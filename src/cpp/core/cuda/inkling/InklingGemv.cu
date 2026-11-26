// Decode-and-dot GEMV for the Inkling block types.  One 256-thread block per
// output row; every thread decodes one weight of each 256-weight block and
// accumulates its product.  This is the parity-first shape from the plan; the
// row/warp tiling and int8 path are the Task 8 optimization.
//
// Types: 40 NVFP4, 42 Q2_0, 19 IQ1_S, 16 IQ2_XXS, 17 IQ2_XS, 22 IQ2_S,
//        23 IQ4_XS (LM head), 0 F32, 30 BF16 (small state projections).
// The F32/BF16 cases use a 32-element "block" so the same kernel covers them.
//
// The expert entry point resolves the expert id on the device (idx[u]) instead
// of on the host: the router writes idx on the same stream, and a host read
// would be a synchronisation inside the token loop.
#include "gpu/InklingGPU.hpp"

#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../gemm/QuantBlocks.cuh"

namespace tinycoder::gpu {
    namespace {

        static const __device__ int8_t kvalues_nvfp4[16] = {
                0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

        static const __device__ uint8_t ksigns[128] = {
                0,
                129,
                130,
                3,
                132,
                5,
                6,
                135,
                136,
                9,
                10,
                139,
                12,
                141,
                142,
                15,
                144,
                17,
                18,
                147,
                20,
                149,
                150,
                23,
                24,
                153,
                154,
                27,
                156,
                29,
                30,
                159,
                160,
                33,
                34,
                163,
                36,
                165,
                166,
                39,
                40,
                169,
                170,
                43,
                172,
                45,
                46,
                175,
                48,
                177,
                178,
                51,
                180,
                53,
                54,
                183,
                184,
                57,
                58,
                187,
                60,
                189,
                190,
                63,
                192,
                65,
                66,
                195,
                68,
                197,
                198,
                71,
                72,
                201,
                202,
                75,
                204,
                77,
                78,
                207,
                80,
                209,
                210,
                83,
                212,
                85,
                86,
                215,
                216,
                89,
                90,
                219,
                92,
                221,
                222,
                95,
                96,
                225,
                226,
                99,
                228,
                101,
                102,
                231,
                232,
                105,
                106,
                235,
                108,
                237,
                238,
                111,
                240,
                113,
                114,
                243,
                116,
                245,
                246,
                119,
                120,
                249,
                250,
                123,
                252,
                125,
                126,
                255,
        };

        /// Elements per quant block, resolved at compile time.
        template<uint32_t Type>
        __host__ __device__ constexpr uint32_t tqk() {
            if (Type == 40 || Type == 42) return 64u;
            if (Type == 0 || Type == 30 || Type == 8) return 32u;
            return 256u;
        }

        /// Bytes per quant block (per tqk<Type>() weights).
        template<uint32_t Type>
        __host__ __device__ constexpr uint32_t tbb() {
            if constexpr (Type == 40) return 36;// NVFP4
            else if constexpr (Type == 42)
                return 18;// Q2_0
            else if constexpr (Type == 19)
                return 50;// IQ1_S
            else if constexpr (Type == 16)
                return 66;// IQ2_XXS
            else if constexpr (Type == 17)
                return 74;// IQ2_XS
            else if constexpr (Type == 22)
                return 82;// IQ2_S
            else if constexpr (Type == 23)
                return 136;// IQ4_XS
            else if constexpr (Type == 0)
                return 32u * 4u;// F32, 32 weights
            else if constexpr (Type == 30)
                return 32u * 2u;// BF16, 32 weights
            else if constexpr (Type == 8)
                return 34;// Q8_0
            else if constexpr (Type == 12)
                return 144;// Q4_K
            else if constexpr (Type == 13)
                return 176;// Q5_K
            else if constexpr (Type == 14)
                return 210;// Q6_K
            else if constexpr (Type == 18)
                return 98;// IQ3_XXS
            else
                return 0;
        }

        __device__ __forceinline__ float bf16ToF32(uint16_t v) {
            const uint32_t u = static_cast<uint32_t>(v) << 16;
            float f;
            memcpy(&f, &u, sizeof(f));
            return f;
        }

        template<uint32_t Type>
        __device__ __forceinline__ float decodeWeightT(const uint8_t *block,
                                                       uint32_t i) {
            if constexpr (Type == 40) {// NVFP4: d[4] UE4M3, qs[32], 64 weights
                const block_nvfp4 *b = reinterpret_cast<const block_nvfp4 *>(block);
                const uint32_t sub = i / QK_NVFP4_SUB;
                const uint32_t j = i % QK_NVFP4_SUB;
                const uint8_t byte = b->qs[sub * (QK_NVFP4_SUB / 2) + (j % 8)];
                const int nib = (j < 8) ? (byte & 0xF) : (byte >> 4);
                const float d = tc_cuda_ue4m3_to_fp32(b->d[sub]);
                return d * static_cast<float>(kvalues_nvfp4[nib]);
            }
            if constexpr (Type == 42) {// Q2_0: fp16 d, 16 bytes, 64 weights
                const uint8_t *b = block;
                const float d = __half2float(*reinterpret_cast<const __half *>(b));
                const uint8_t q = b[2 + i / 4];
                const int v = ((q >> ((i % 4) * 2)) & 3) - 1;
                return d * static_cast<float>(v);
            }
            if constexpr (Type == 19) {// IQ1_S: fp16 d, qs[32], qh[16], 256 weights
                const block_iq1_s *b = reinterpret_cast<const block_iq1_s *>(block);
                const uint32_t ib = i / 32;
                const uint32_t l = (i % 32) / 8;
                const uint32_t j = i % 8;
                const uint16_t qh = b->qh[ib];
                const int idx = b->qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
                const uint32_t grid = iq1s_grid_gpu[idx];
                const int shift = 8 * (j & 3) + (j >= 4 ? 4 : 0);
                const int nib = (grid >> shift) & 0xF;
                const float d1q = __half2float(b->d) *
                                  static_cast<float>((((qh >> 11) & 0x0E) + 1));
                const float delta = -1.0f + 0.125f -
                                    ((qh & 0x8000) ? 0.25f : 0.0f);
                return d1q * (static_cast<float>(nib) + delta);
            }
            if constexpr (Type == 16) {// IQ2_XXS: fp16 d, qs[32] u16, 256 weights
                const block_iq2_xxs *b =
                        reinterpret_cast<const block_iq2_xxs *>(block);
                const uint32_t ib = i / 32;
                const uint32_t l = (i % 32) / 8;
                const uint32_t j = i % 8;
                uint32_t aux[2];
                memcpy(&aux[0], &b->qs[4 * ib], sizeof(uint32_t));
                memcpy(&aux[1], &b->qs[4 * ib + 2], sizeof(uint32_t));
                const uint8_t *aux8 = reinterpret_cast<const uint8_t *>(aux);
                const float d = __half2float(b->d);
                const float db = d * (0.5f + static_cast<float>(aux[1] >> 28)) * 0.25f;
                const uint64_t grid = iq2xxs_grid[aux8[l]];
                const int8_t v = static_cast<int8_t>((grid >> (8 * j)) & 0xFF);
                // ksigns_iq2xs bit j: 1 means negate.

                const uint8_t signs = ksigns[(aux[1] >> (7 * l)) & 127];
                const float sgn = ((signs >> j) & 1) ? -1.0f : 1.0f;
                return db * static_cast<float>(v) * sgn;
            }
            if constexpr (Type == 17) {// IQ2_XS: fp16 d, qs[32] u16, scales[8], 256 weights
                const block_iq2_xs *b =
                        reinterpret_cast<const block_iq2_xs *>(block);
                const uint32_t ib = i / 32;
                const uint32_t l = (i % 32) / 8;
                const uint32_t j = i % 8;
                const uint16_t qval = b->qs[4 * ib + l];
                const uint64_t grid = iq2xs_grid[qval & 0x1FF];
                const uint8_t signs = ksigns[qval >> 9];
                const uint32_t sh = 4u * (l / 2u);
                const float db = __half2float(b->d) *
                                 (0.5f + static_cast<float>((b->scales[ib] >> sh) & 0xF)) *
                                 0.25f;
                const float mag = static_cast<float>((grid >> (8 * j)) & 0xFF);
                return db * mag * (((signs >> j) & 1) ? -1.0f : 1.0f);
            }
            if constexpr (Type == 22) {// IQ2_S: fp16 d, qs[64], qh[8], scales[8], 256 weights
                const block_iq2_s *b = reinterpret_cast<const block_iq2_s *>(block);
                const uint32_t ib = i / 32;
                const uint32_t l = (i % 32) / 8;
                const uint32_t j = i % 8;
                const uint32_t gridIdx =
                        static_cast<uint32_t>(b->qs[4 * ib + l]) |
                        ((static_cast<uint32_t>(b->qh[ib]) << (8 - 2 * l)) & 0x300u);
                const uint64_t grid = iq2s_grid[gridIdx];
                const uint8_t signs = b->qs[32 + 4 * ib + l];// signs = qs + QK_K/8
                const uint32_t sh = 4u * (l / 2u);
                const float db = __half2float(b->d) *
                                 (0.5f + static_cast<float>((b->scales[ib] >> sh) & 0xF)) *
                                 0.25f;
                const float mag = static_cast<float>((grid >> (8 * j)) & 0xFF);
                return db * mag * (((signs >> j) & 1) ? -1.0f : 1.0f);
            }
            if constexpr (Type == 23) {// IQ4_XS: fp16 d, scales_h, scales_l[4], qs[128]
                const block_iq4_xs *b = reinterpret_cast<const block_iq4_xs *>(block);
                const uint32_t ib = i / 32;
                const uint32_t j = i % 32;
                const int ls = static_cast<int>(
                                       (b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                               (static_cast<int>((b->scales_h >> (2 * ib)) & 3) << 4);
                const float dl = __half2float(b->d) * static_cast<float>(ls - 32);
                const uint8_t byte = b->qs[ib * 16 + (j & 15)];
                const int nib = (j < 16) ? (byte & 0xF) : (byte >> 4);
                return dl * static_cast<float>(kvalues_iq4nl[nib]);
            }

            if constexpr (Type == 8) {// Q8_0: fp16 d, qs[32] int8, 32 weights
                const float d = __half2float(*reinterpret_cast<const __half *>(block));
                return d * static_cast<float>(
                                   reinterpret_cast<const int8_t *>(block + 2)[i]);
            }
            if constexpr (Type == 12) {// Q4_K: d, dmin, scales[12], qs[128], 256 weights
                const uint8_t *scales = block + 4;
                const uint8_t *qs = block + 16;
                const float d = __half2float(*reinterpret_cast<const __half *>(block));
                const float dmin =
                        __half2float(*reinterpret_cast<const __half *>(block + 2));
                const uint32_t chunk = i / 64;
                const uint32_t sub = (i % 64) / 32;
                const uint32_t l = i % 32;
                const uint32_t is = chunk * 2 + sub;
                uint32_t scv, m;
                if (is < 4) {
                    scv = scales[is] & 63u;
                    m = scales[is + 4] & 63u;
                } else {
                    scv = (scales[is + 4] & 0xFu) | ((scales[is - 4] >> 6) << 4);
                    m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
                }
                const uint32_t q = sub == 0 ? (qs[chunk * 32 + l] & 0xFu)
                                            : (qs[chunk * 32 + l] >> 4);
                return d * static_cast<float>(scv) * static_cast<float>(q) -
                       dmin * static_cast<float>(m);
            }
            if constexpr (Type == 13) {// Q5_K: d, dmin, scales[12], qh[32], qs[128]
                const uint8_t *scales = block + 4;
                const uint8_t *qh = block + 16;
                const uint8_t *qs = block + 48;
                const float d = __half2float(*reinterpret_cast<const __half *>(block));
                const float dmin =
                        __half2float(*reinterpret_cast<const __half *>(block + 2));
                const uint32_t chunk = i / 64;
                const uint32_t sub = (i % 64) / 32;
                const uint32_t l = i % 32;
                const uint32_t is = chunk * 2 + sub;
                uint32_t scv, m;
                if (is < 4) {
                    scv = scales[is] & 63u;
                    m = scales[is + 4] & 63u;
                } else {
                    scv = (scales[is + 4] & 0xFu) | ((scales[is - 4] >> 6) << 4);
                    m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
                }
                const uint32_t bit = sub == 0 ? (1u << (2 * chunk))
                                              : (2u << (2 * chunk));
                const uint32_t q5 = (sub == 0 ? (qs[chunk * 32 + l] & 0xFu)
                                              : (qs[chunk * 32 + l] >> 4)) |
                                    ((qh[l] & bit) ? 16u : 0u);
                return d * static_cast<float>(scv) * static_cast<float>(q5) -
                       dmin * static_cast<float>(m);
            }
            if constexpr (Type == 14) {// Q6_K: ql[128], qh[64], scales[16], fp16 d
                const uint32_t n = i / 128;
                const uint32_t r = i % 128;
                const uint32_t l = r % 32;
                const uint32_t g = r / 32;
                const uint8_t *ql = block + n * 64;
                const uint8_t *qh = block + 128 + n * 32;
                const int8_t *sc = reinterpret_cast<const int8_t *>(block + 192 + n * 8);
                const float d =
                        __half2float(*reinterpret_cast<const __half *>(block + 208));
                int q;
                if (g == 0) {
                    q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
                } else if (g == 1) {
                    q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                } else if (g == 2) {
                    q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                } else {
                    q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                }
                q -= 32;
                return d * static_cast<float>(sc[l / 16 + 2 * g]) * static_cast<float>(q);
            }
            if constexpr (Type == 18) {// IQ3_XXS: fp16 d, qs[3*256/8], 256 weights
                const uint8_t *qs = block + 2;
                const uint8_t *ss = qs + QK_K / 4;
                const uint32_t ib = i / 32;
                const uint32_t l = (i % 32) / 8;
                const uint32_t j = i % 8;
                uint32_t aux32;
                memcpy(&aux32, ss + 4 * ib, sizeof(uint32_t));
                const float db = __half2float(*reinterpret_cast<const __half *>(block)) *
                                 (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                const uint8_t signs = ksigns[(aux32 >> (7 * l)) & 127];
                const uint32_t gridIdx = qs[ib * 8 + 2 * l + (j < 4 ? 0 : 1)];
                const uint32_t grid = iq3xxs_grid[gridIdx];
                const uint32_t byte = (grid >> (8 * (j & 3))) & 0xFF;
                return db * static_cast<float>(byte) *
                       (((signs >> j) & 1) ? -1.0f : 1.0f);
            }
            if constexpr (Type == 0) {// F32
                return reinterpret_cast<const float *>(block)[i];
            }
            if constexpr (Type == 30) {// BF16
                uint16_t v;
                memcpy(&v, block + 2 * i, sizeof(v));
                return bf16ToF32(v);
            }
            return 0.0f;
        }
        /// Warp-per-row decode-dot.  One warp owns one output row; lane l
        /// handles qk/32 consecutive weights of every quant block, so the
        /// per-row reduction is five warp shuffles instead of a 256-thread
        /// shared-memory tree.  The type is a template argument, so the decode
        /// is branch-free inside the loop.
        template<uint32_t Type>
        __global__ void inklingGemvWarpKernel(const uint8_t *w, const float *x,
                                              float *out, uint32_t totalRows,
                                              uint32_t cols,
                                              uint32_t blocksPerRow,
                                              uint32_t blockBytes,
                                              const int32_t *rowIdx,
                                              uint32_t rowsPer,
                                              uint32_t xStride) {
            constexpr uint32_t qk = tqk<Type>();
            constexpr uint32_t perLane = qk / 32u;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5u;
            const uint32_t o = blockIdx.x * (blockDim.x >> 5u) + warp;
            if (o >= totalRows) return;
            uint32_t row = o;
            const float *xv = x;
            if (rowIdx != nullptr) {
                const uint32_t u = o / rowsPer;
                const uint32_t r = o - u * rowsPer;
                row = static_cast<uint32_t>(rowIdx[u]) * rowsPer + r;
                xv = x + static_cast<size_t>(u) * xStride;
            }
            const uint8_t *rowPtr = w + static_cast<uint64_t>(row) *
                                                blocksPerRow * blockBytes;
            float acc = 0.0f;
            const uint32_t first = lane * perLane;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blk = rowPtr + static_cast<uint64_t>(b) * blockBytes;
                const uint32_t base = b * qk;
#pragma unroll
                for (uint32_t k = 0; k < perLane; ++k) {
                    const uint32_t col = base + first + k;
                    if (col < cols)
                        acc = fmaf(decodeWeightT<Type>(blk, first + k), xv[col], acc);
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) out[o] = acc;
        }


        /// Consume one decoded per-lane weight: either keep it for the batched
        /// GEMV (KeepW) or fold it into the running dot (decode GEMV).  One
        /// body, two instantiations, so the two paths decode with the same
        /// instructions and cannot drift apart -- and the decode path keeps the
        /// codegen it had before the batched kernel existed.
        template<bool KeepW>
        __device__ __forceinline__ void decodeEmit(float v, float *wv, float &acc,
                                                   const float *xb, uint32_t k) {
            if constexpr (KeepW) wv[k] = v;
            else
                acc = fmaf(v, xb[k], acc);
        }

        /// NVFP4's per-16-weight scale, without `ldexpf`.
        ///
        /// tc_cuda_ue4m3_to_fp32 builds its result as
        /// `ldexpf(1 + man/8, exp - 7) / 2` (or `ldexpf(man, -9) / 2` for
        /// exp == 0).  `ldexpf` is a real call on device and it ran once PER
        /// WEIGHT in the decode loop: measured 187.6 us for the NVFP4 down
        /// shape against 65.8 for the Q2_0 down shape at half the bytes, i.e.
        /// the conversion, not the memory, owned that kernel.  Both branches
        /// are exact power-of-two rescalings of small integers, so the same
        /// value comes out of integer exponent arithmetic:
        ///   exp > 0 : (8 + man) * 2^(exp-11)   [== (1 + man/8) * 2^(exp-8)]
        ///   exp == 0: man * 2^-10              [== ldexpf(man, -9) / 2]
        /// `8 + man` and `2^(exp-11)` are both exactly representable and the
        /// rescale is a power of two, so the product is bit-identical to the
        /// reference's.  x == 0x7F keeps the reference's explicit zero.
        __device__ __forceinline__ float ue4m3Fast(uint8_t x) {
            if (x == 0x7Fu) return 0.0f;
            const uint32_t e = (x >> 3) & 0xFu;
            const uint32_t m = x & 7u;
            if (e == 0) return static_cast<float>(m) * (1.0f / 1024.0f);
            const float p = __int_as_float(static_cast<int>(e + 116u) << 23);
            return static_cast<float>(8u + m) * p;
        }

        /// Per-lane weights for one quant block.  For 256-weight blocks
        /// (qk == 256) lane l owns the eight weights [8l, 8l+8); the IQ grid
        /// types therefore need ONE grid lookup + ONE sign byte per lane
        /// instead of eight independent decodes.  The 32/64-weight types use
        /// the generic per-weight decoder (1-2 weights per lane).
        /// KeepW == false: fold every weight into `acc` and return it (decode).
        /// KeepW == true:  store every weight into `wv`, return 0 (batched).
        ///
        /// Fast == true is the single-token decode shape (inklingGemvWarp4):
        /// it uses the shared-memory grid tables the decode kernel stages for
        /// every IQ type and the arithmetic NVFP4 scale.  Fast == false is the
        /// batched/prefill path, whose instructions and results are unchanged.
        template<uint32_t Type, bool KeepW, bool Fast = false>
        __device__ __forceinline__ float decodeLaneImpl(const uint8_t *blk,
                                                        uint32_t lane,
                                                        const float *xb, float *wv,
                                                        const float *iq4Tab,
                                                        const uint8_t *signsTab,
                                                        const uint64_t *gridTab) {
            constexpr uint32_t qk = tqk<Type>();
            constexpr uint32_t perLane = qk / 32u;
            const uint32_t first = lane * perLane;
            float acc = 0.0f;
            // `!KeepW`: the branch folds the weights itself instead of calling
            // decodeEmit, so it must only ever serve the decode consumer.  A
            // (Fast, KeepW) instantiation would otherwise return a value and
            // silently not fill `wv`.  It also assumes perLane == 2, which is
            // tqk<40>() == 64.
            if constexpr (Fast && Type == 40 && !KeepW) {
                // 64-weight NVFP4 block: lane l owns weights 2l and 2l+1, which
                // share one 16-weight sub-block, so the UE4M3 scale is decoded
                // once per lane instead of once per weight.  Same operand order
                // and same `d * value` product as decodeWeightT<40>.
                const uint32_t i = lane * 2u;
                const uint32_t sub = i >> 4;
                const uint32_t j = i & 15u;
                const uint32_t byte0 = 4u + sub * 8u + (j & 7u);
                const uint32_t sh = (j < 8u) ? 0u : 4u;
                const float d = ue4m3Fast(blk[sub]);
                const int8_t k0 = kvalues_nvfp4[(blk[byte0] >> sh) & 0xFu];
                const int8_t k1 = kvalues_nvfp4[(blk[byte0 + 1u] >> sh) & 0xFu];
                acc = fmaf(d * static_cast<float>(k0), xb[0], acc);
                acc = fmaf(d * static_cast<float>(k1), xb[1], acc);
                return acc;
            }
            if constexpr (qk != 256) {
#pragma unroll
                for (uint32_t k = 0; k < perLane; ++k)
                    decodeEmit<KeepW>(decodeWeightT<Type>(blk, first + k), wv, acc,
                                      xb, k);
                return acc;
            }
            const uint32_t ib = lane >> 2;// 32-weight sub-block, 0..7
            const uint32_t g = lane & 3;  // grid slot inside the sub-block
            if constexpr (Type == 16) {   // IQ2_XXS
                // block_iq2_xxs::qs is uint16_t[32], so one 32-weight
                // sub-block spans 8 bytes at qs + 8*ib: the low u32 holds the
                // four grid indices, the high u32 the sign/scale bits.
                uint64_t aux64;
                memcpy(&aux64, blk + 2 + 8 * ib, 8);
                const uint32_t aux1 = static_cast<uint32_t>(aux64 >> 32);
                // The decode shape reads the 2 KiB grid table from the block's
                // staged copy: its indices are random, so a global load costs up
                // to 32 sectors per warp instruction while a shared read is one
                // LDS with bank conflicts.  The batched path keeps the global
                // table (Fast == false), so prefill is untouched.
                const uint8_t gidx = static_cast<uint8_t>((aux64 >> (8 * g)) & 0xFFu);
                uint64_t grid;
                if constexpr (Fast) grid = gridTab[gidx];
                else
                    grid = iq2xxs_grid[gidx];
                const uint8_t signs = signsTab[(aux1 >> (7 * g)) & 127];
                const float db = __half2float(
                                         *reinterpret_cast<const __half *>(blk)) *
                                 (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    const float v = static_cast<float>(
                            static_cast<int8_t>((grid >> (8 * k)) & 0xFF));
                    decodeEmit<KeepW>(db * (((signs >> k) & 1) ? -v : v), wv, acc,
                                      xb, k);
                }
                return acc;
            } else if constexpr (Type == 19) {// IQ1_S
                const uint16_t qh = reinterpret_cast<const block_iq1_s *>(blk)->qh[ib];
                uint32_t qs4;
                memcpy(&qs4, blk + 2 + 4 * ib, 4);
                const int idx = static_cast<int>((qs4 >> (8 * g)) & 0xFFu) |
                                (((qh >> (3 * g)) & 7) << 8);
                // 8 KiB IQ1_S grid: staged for the decode shape only (Fast),
                // same reasoning as the IQ2_XXS table above.
                uint32_t grid;
                if constexpr (Fast)
                    grid = reinterpret_cast<const uint32_t *>(gridTab)[idx];
                else
                    grid = iq1s_grid_gpu[idx];
                const float d1q =
                        __half2float(*reinterpret_cast<const __half *>(blk)) *
                        static_cast<float>((((qh >> 11) & 0x0E) + 1));
                const float delta =
                        -1.0f + 0.125f - ((qh & 0x8000) ? 0.25f : 0.0f);
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    const uint32_t shift = 8 * (k & 3) + (k >= 4 ? 4 : 0);
                    const int nib = (grid >> shift) & 0xF;
                    decodeEmit<KeepW>(d1q * (static_cast<float>(nib) + delta), wv,
                                      acc, xb, k);
                }
                return acc;
            } else if constexpr (Type == 17) {// IQ2_XS
                const uint8_t *scales = blk + 66;
                uint64_t qs64;
                memcpy(&qs64, blk + 2 + 8 * ib, 8);
                const uint16_t qval = static_cast<uint16_t>(qs64 >> (16 * g));
                const uint64_t grid = gridTab[qval & 0x1FF];
                const uint8_t signs = signsTab[qval >> 9];
                const float db = __half2float(
                                         *reinterpret_cast<const __half *>(blk)) *
                                 (0.5f + static_cast<float>(
                                                 (scales[ib] >> (4 * (g / 2))) & 0xF)) *
                                 0.25f;
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    const float v = static_cast<float>((grid >> (8 * k)) & 0xFF);
                    decodeEmit<KeepW>(db * (((signs >> k) & 1) ? -v : v), wv, acc,
                                      xb, k);
                }
                return acc;
            } else if constexpr (Type == 22) {// IQ2_S
                const uint8_t *qh = blk + 66;
                const uint8_t *scales = blk + 74;
                uint32_t qs4, sg4;
                memcpy(&qs4, blk + 2 + 4 * ib, 4);
                memcpy(&sg4, blk + 34 + 4 * ib, 4);
                const uint32_t gridIdx =
                        ((qs4 >> (8 * g)) & 0xFFu) |
                        ((static_cast<uint32_t>(qh[ib]) << (8 - 2 * g)) & 0x300u);
                const uint64_t grid = gridTab[gridIdx];
                const uint8_t signs = static_cast<uint8_t>((sg4 >> (8 * g)) & 0xFFu);
                const float db = __half2float(
                                         *reinterpret_cast<const __half *>(blk)) *
                                 (0.5f + static_cast<float>(
                                                 (scales[ib] >> (4 * (g / 2))) & 0xF)) *
                                 0.25f;
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    const float v = static_cast<float>((grid >> (8 * k)) & 0xFF);
                    decodeEmit<KeepW>(db * (((signs >> k) & 1) ? -v : v), wv, acc,
                                      xb, k);
                }
                return acc;
            } else if constexpr (Type == 23) {// IQ4_XS
                const uint8_t *scales_l = blk + 4;
                const uint8_t *qsv = blk + 8;
                const uint16_t scales_h =
                        *reinterpret_cast<const uint16_t *>(blk + 2);
                const int ls =
                        static_cast<int>((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                        (static_cast<int>((scales_h >> (2 * ib)) & 3) << 4);
                const float dl = __half2float(
                                         *reinterpret_cast<const __half *>(blk)) *
                                 static_cast<float>(ls - 32);
                const uint8_t *qp = qsv + ib * 16 + (g < 2 ? 8 * g : 8 * (g - 2));
                uint64_t q8 = *reinterpret_cast<const uint64_t *>(qp);
                const uint32_t qsh = (g < 2) ? 0u : 4u;
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    const uint32_t nib = (q8 >> (8 * k + qsh)) & 0xFu;
                    decodeEmit<KeepW>(dl * iq4Tab[nib], wv, acc, xb, k);
                }
                return acc;
            } else if constexpr (Type == 14) {// Q6_K
                const uint32_t n = lane >> 4;
                const uint32_t within = lane & 15;
                const uint32_t base = 8 * (within & 3);
                const uint32_t gg = within >> 2;
                const uint32_t qsh = (gg < 2) ? 0u : 4u;
                uint64_t ql8, qh8;
                memcpy(&ql8, blk + n * 64 + base + ((gg & 1) ? 32 : 0), 8);
                memcpy(&qh8, blk + 128 + n * 32 + base, 8);
                const int8_t *sc = reinterpret_cast<const int8_t *>(blk + 192 + n * 8);
                const float d =
                        __half2float(*reinterpret_cast<const __half *>(blk + 208));
                const float scv =
                        d * static_cast<float>(sc[(base >= 16 ? 1 : 0) + 2 * gg]);
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    const uint32_t nib = (ql8 >> (8 * k + qsh)) & 0xFu;
                    const uint32_t hi = (qh8 >> (8 * k + 2 * gg)) & 3u;
                    const int q = static_cast<int>(nib | (hi << 4)) - 32;
                    decodeEmit<KeepW>(scv * static_cast<float>(q), wv, acc, xb, k);
                }
                return acc;
            } else {// Q4_K / Q5_K / IQ3_XXS: 8 generic decodes
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k)
                    decodeEmit<KeepW>(decodeWeightT<Type>(blk, first + k), wv, acc,
                                      xb, k);
                return acc;
            }
        }

        /// Per-lane contribution for one quant block: the decode-time consumer.
        template<uint32_t Type>
        __device__ __forceinline__ float dotLane(const uint8_t *blk, uint32_t lane,
                                                 const float *xb,
                                                 const float *iq4Tab,
                                                 const uint8_t *signsTab,
                                                 const uint64_t *gridTab) {
            return decodeLaneImpl<Type, false>(blk, lane, xb, nullptr, iq4Tab,
                                               signsTab, gridTab);
        }

        /// Decode-only consumer: same arithmetic, but the IQ grid tables come
        /// from the block's staged shared copy and NVFP4 uses the arithmetic
        /// UE4M3 scale.  Only inklingGemvWarp4Kernel calls this; the batched
        /// prefill kernels keep `dotLane`, so their instructions and results
        /// are unchanged.
        template<uint32_t Type>
        __device__ __forceinline__ float dotLaneFast(const uint8_t *blk,
                                                     uint32_t lane,
                                                     const float *xb,
                                                     const float *iq4Tab,
                                                     const uint8_t *signsTab,
                                                     const uint64_t *gridTab) {
            return decodeLaneImpl<Type, false, true>(blk, lane, xb, nullptr,
                                                     iq4Tab, signsTab, gridTab);
        }

        /// Per-lane weights for one quant block: the batched-GEMV consumer.
        template<uint32_t Type>
        __device__ __forceinline__ void decodeLaneVals(const uint8_t *blk,
                                                       uint32_t lane, float *wv,
                                                       const float *iq4Tab,
                                                       const uint8_t *signsTab,
                                                       const uint64_t *gridTab) {
            decodeLaneImpl<Type, true>(blk, lane, nullptr, wv, iq4Tab, signsTab,
                                       gridTab);
        }

        /// Four output rows per warp.  The four weight streams give the warp
        /// four independent memory/ALU chains, and the eight activation values
        /// are loaded once per block and shared by all four rows.  Eight rows
        /// per warp was measured and regressed ~10% (register pressure and
        /// weaker cache locality), so keep four.
        ///
        /// kRowsT == 1 is the same body with one row per warp.  launchT picks
        /// it for a short output matrix: 128 rows at four rows per warp is four
        /// blocks, i.e. four SMs out of 68, and the router's 1.18 MiB F32 matrix
        /// measured 14.3 us there.  One row per warp spreads the same rows over
        /// 16 blocks and keeps the arithmetic identical -- same lane->weight
        /// map, same per-block FMA chain, same butterfly -- so the result is
        /// bit-identical, not merely within a band.
        template<uint32_t Type, uint32_t kRowsT = 4>
        __global__ void inklingGemvWarp4Kernel(
                const uint8_t *__restrict__ w, const float *__restrict__ x,
                float *__restrict__ out, uint32_t totalRows, uint32_t cols,
                uint32_t blocksPerRow, uint32_t blockBytes,
                const int32_t *rowIdx, uint32_t rowsPer, uint32_t xStride,
                bool alignedX) {
            constexpr uint32_t qk = tqk<Type>();
            constexpr uint32_t perLane = qk / 32u;
            constexpr uint32_t kRows = kRowsT;
            constexpr uint32_t kWarps = 8;
            // The grid staging buffer is sized per type.  Only the 256-weight
            // grid types need one, and a flat 8 KiB for every instantiation
            // capped the DOWN shapes (Q2_0, NVFP4 -- they never read it) at 7
            // blocks/SM by the 64 KiB static-shared limit.  Measured effect on
            // the down shape: none (it is not occupancy-limited at 7 blocks),
            // so this is a correctness-of-resource-sizing change, not a win.
            constexpr uint32_t kGridTab =
                    (Type == 19 || Type == 22) ? 1024u
                    : (Type == 17)             ? 512u
                    : (Type == 16)             ? 256u
                                               : 1u;
            __shared__ float sIQ4[16];
            __shared__ uint8_t sSigns[128];
            __shared__ uint64_t sGrid[kGridTab];
            if (threadIdx.x < 16)
                sIQ4[threadIdx.x] = static_cast<float>(kvalues_iq4nl[threadIdx.x]);
            if (threadIdx.x < 128) sSigns[threadIdx.x] = ksigns[threadIdx.x];
            if constexpr (Type == 17 || Type == 22) {
                // IQ2_XS (512 x u64) / IQ2_S (1024 x u64) grid tables: stage in
                // shared memory.  The per-lane lookup is random, which turned
                // each warp instruction into up to 32 global sectors; from
                // shared it is one LDS with a few bank conflicts.
                const uint64_t *src = (Type == 17) ? iq2xs_grid : iq2s_grid;
                constexpr uint32_t nTab = (Type == 17) ? 512u : 1024u;
                for (uint32_t i = threadIdx.x; i < nTab; i += blockDim.x)
                    sGrid[i] = src[i];
            } else if constexpr (Type == 16) {
                // IQ2_XXS: 256 x u64 = 2 KiB.  The decode shape's dot reads it
                // through `gridTab` (see decodeLaneImpl's Fast branch).
                for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x)
                    sGrid[i] = iq2xxs_grid[i];
            } else if constexpr (Type == 19) {
                // IQ1_S: 2048 x u32 = 8 KiB, packed into the same u64 buffer.
                uint32_t *dst = reinterpret_cast<uint32_t *>(sGrid);
                for (uint32_t i = threadIdx.x; i < 2048u; i += blockDim.x)
                    dst[i] = iq1s_grid_gpu[i];
            }
            __syncthreads();
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5u;
            const uint32_t o0 = (blockIdx.x * kWarps + warp) * kRows;
            if (o0 >= totalRows) return;
            const uint32_t nr = min(kRows, totalRows - o0);
            const uint8_t *rp[kRows];
            uint32_t u0 = 0;
            for (uint32_t r = 0; r < nr; ++r) {
                const uint32_t o = o0 + r;
                uint32_t row = o;
                if (rowIdx != nullptr) {
                    const uint32_t u = o / rowsPer;
                    const uint32_t rr = o - u * rowsPer;
                    row = static_cast<uint32_t>(rowIdx[u]) * rowsPer + rr;
                    u0 = u;
                }
                rp[r] = w + static_cast<uint64_t>(row) * blocksPerRow * blockBytes;
            }
            const float *xv = x;
            if (rowIdx != nullptr) xv = x + static_cast<size_t>(u0) * xStride;
            float acc[kRows] = {};
            const uint32_t first = lane * perLane;
#pragma unroll 3
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                float xb[8];
                const uint32_t base = b * qk + first;
                if (perLane == 2 && alignedX) {
                    // 64-weight blocks: base*4 is 8-byte aligned.
                    const float2 a = *reinterpret_cast<const float2 *>(xv + base);
                    xb[0] = a.x;
                    xb[1] = a.y;
                } else if (perLane == 8 && alignedX) {
                    // base*4 is 32-byte aligned when cols is a multiple of qk.
                    const float4 a = *reinterpret_cast<const float4 *>(xv + base);
                    const float4 c = *reinterpret_cast<const float4 *>(xv + base + 4);
                    xb[0] = a.x;
                    xb[1] = a.y;
                    xb[2] = a.z;
                    xb[3] = a.w;
                    xb[4] = c.x;
                    xb[5] = c.y;
                    xb[6] = c.z;
                    xb[7] = c.w;
                } else {
#pragma unroll
                    for (uint32_t k = 0; k < perLane; ++k)
                        xb[k] = (base + k < cols) ? __ldg(&xv[base + k]) : 0.0f;
                }
#pragma unroll
                for (uint32_t r = 0; r < kRows; ++r) {
                    if (r < nr)
                        acc[r] += dotLaneFast<Type>(
                                rp[r] + static_cast<uint64_t>(b) * blockBytes, lane,
                                xb, sIQ4, sSigns, sGrid);
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
                for (uint32_t r = 0; r < kRows; ++r)
                    acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
            }
            if (lane == 0) {
#pragma unroll
                for (uint32_t r = 0; r < kRows; ++r)
                    if (r < nr) out[o0 + r] = acc[r];
            }
        }

        /// Small-row batched GEMV: one warp per (token, output row).
        ///
        /// The 4-rows-per-warp batched kernel packs 32 rows into a block, so a
        /// projection with fewer rows (the attention sink projection is 24 x
        /// 2304) runs as a SINGLE block no matter how many tokens are in
        /// flight -- at m=128 that is 8 launches of one block for a 7 MFLOP
        /// product, and it measured 104 ms per 128-token pass in the batched
        /// prefill graph.  Here the token is a grid dimension instead, so the
        /// same arithmetic (same lane partial order, same shuffle tree) runs
        /// across m*rows warps.
        template<uint32_t Type>
        __global__ void inklingGemvSmallKernel(const uint8_t *__restrict__ w,
                                               const float *__restrict__ x,
                                               float *__restrict__ out,
                                               uint32_t rows, uint32_t cols,
                                               uint32_t m, uint32_t blocksPerRow,
                                               uint32_t blockBytes) {
            constexpr uint32_t qk = tqk<Type>();
            constexpr uint32_t perLane = qk / 32u;
            __shared__ float sIQ4[16];
            __shared__ uint8_t sSigns[128];
            __shared__ uint64_t sGrid[1024];
            if (threadIdx.x < 16)
                sIQ4[threadIdx.x] = static_cast<float>(kvalues_iq4nl[threadIdx.x]);
            if (threadIdx.x < 128) sSigns[threadIdx.x] = ksigns[threadIdx.x];
            if constexpr (Type == 17 || Type == 22) {
                const uint64_t *src = (Type == 17) ? iq2xs_grid : iq2s_grid;
                constexpr uint32_t nTab = (Type == 17) ? 512u : 1024u;
                for (uint32_t i = threadIdx.x; i < nTab; i += blockDim.x)
                    sGrid[i] = src[i];
            }
            __syncthreads();
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t wid = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
            const uint32_t total = m * rows;
            if (wid >= total) return;
            const uint32_t t = wid / rows;
            const uint32_t r = wid - t * rows;
            const uint8_t *rp = w + static_cast<size_t>(r) * blocksPerRow * blockBytes;
            const float *xv = x + static_cast<size_t>(t) * cols;
            const uint32_t first = lane * perLane;
            const bool alignedX = (cols % qk) == 0;
            float acc = 0.0f;
#pragma unroll 3
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                float xb[8];
                const uint32_t base = b * qk + first;
                if (perLane == 2 && alignedX) {
                    const float2 a = *reinterpret_cast<const float2 *>(xv + base);
                    xb[0] = a.x;
                    xb[1] = a.y;
                } else if (perLane == 8 && alignedX) {
                    const float4 a = *reinterpret_cast<const float4 *>(xv + base);
                    const float4 c = *reinterpret_cast<const float4 *>(xv + base + 4);
                    xb[0] = a.x;
                    xb[1] = a.y;
                    xb[2] = a.z;
                    xb[3] = a.w;
                    xb[4] = c.x;
                    xb[5] = c.y;
                    xb[6] = c.z;
                    xb[7] = c.w;
                } else {
#pragma unroll
                    for (uint32_t k = 0; k < perLane; ++k)
                        xb[k] = (base + k < cols) ? __ldg(&xv[base + k]) : 0.0f;
                }
                acc += dotLane<Type>(
                        rp + static_cast<uint64_t>(b) * blockBytes, lane, xb, sIQ4,
                        sSigns, sGrid);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) out[static_cast<size_t>(t) * rows + r] = acc;
        }

        template<uint32_t Type>
        bool launchSmall(const void *w, const float *x, float *out, uint32_t rows,
                         uint32_t cols, uint32_t m, cudaStream_t stream) {
            constexpr uint32_t blockBytes = tbb<Type>();
            constexpr uint32_t qk = tqk<Type>();
            if (rows == 0 || cols == 0 || m == 0 || w == nullptr || x == nullptr ||
                out == nullptr)
                return false;
            const uint32_t blocksPerRow = (cols + qk - 1) / qk;
            const uint32_t total = m * rows;
            const dim3 grid((total + 7u) / 8u);
            inklingGemvSmallKernel<Type><<<grid, 256, 0, stream>>>(
                    static_cast<const uint8_t *>(w), x, out, rows, cols, m,
                    blocksPerRow, blockBytes);
            return true;
        }

        template<uint32_t Type>
        bool launchT(const void *w, const float *x, float *out, uint32_t rows,
                     uint32_t cols, const int32_t *rowIdx, uint32_t rowsPer,
                     uint32_t xStride, cudaStream_t stream) {
            constexpr uint32_t blockBytes = tbb<Type>();
            constexpr uint32_t qk = tqk<Type>();
            if (rows == 0 || cols == 0 || w == nullptr || x == nullptr ||
                out == nullptr)
                return false;
            const uint32_t blocksPerRow = (cols + qk - 1) / qk;
            constexpr uint32_t kWarpsPerBlock = 8;
            constexpr uint32_t kRowsPerWarp = 4;
            // A short output matrix leaves the card idle at four rows per warp:
            // 128 rows is four blocks.  Use one row per warp there instead, so
            // the same rows run on eight times as many blocks.  The per-row
            // arithmetic is identical (see the kernel comment), so only the
            // launch geometry changes.  The cut is 512 rows rather than 1024
            // because the two forms measured the same at 1024 (the K/V
            // projections), so that shape is left on its old geometry.
            const bool oneRow = rowIdx == nullptr && rows <= 512u;
            const uint32_t perBlock = kWarpsPerBlock * (oneRow ? 1u : kRowsPerWarp);
            const uint32_t blocks = (rows + perBlock - 1) / perBlock;
            if (oneRow) {
                inklingGemvWarp4Kernel<Type, 1>
                        <<<blocks, 32 * kWarpsPerBlock, 0, stream>>>(
                                static_cast<const uint8_t *>(w), x, out, rows,
                                cols, blocksPerRow, blockBytes, rowIdx, rowsPer,
                                xStride, (cols % qk) == 0);
            } else {
                inklingGemvWarp4Kernel<Type, kRowsPerWarp>
                        <<<blocks, 32 * kWarpsPerBlock, 0, stream>>>(
                                static_cast<const uint8_t *>(w), x, out, rows,
                                cols, blocksPerRow, blockBytes, rowIdx, rowsPer,
                                xStride, (cols % qk) == 0);
            }
            // No per-launch cudaGetLastError(): the caller's final stream sync
            // surfaces a launch error, and at 688 launches/token the query is
            // measurable host overhead.
            return true;
        }

        /// Stage the decode lookup tables in shared memory.  Same set the
        /// 4-row decode kernel stages: the IQ4 nibble table, the IQ2 sign table
        /// and (only for the two grid-indexed types) the grid table, whose
        /// per-lane lookup is random and would be up to 32 global sectors.
        template<uint32_t Type>
        __device__ __forceinline__ void stageDecodeTables(float *sIQ4,
                                                          uint8_t *sSigns,
                                                          uint64_t *sGrid) {
            if (threadIdx.x < 16)
                sIQ4[threadIdx.x] = static_cast<float>(kvalues_iq4nl[threadIdx.x]);
            if (threadIdx.x < 128) sSigns[threadIdx.x] = ksigns[threadIdx.x];
            if constexpr (Type == 17 || Type == 22) {
                const uint64_t *src = (Type == 17) ? iq2xs_grid : iq2s_grid;
                constexpr uint32_t nTab = (Type == 17) ? 512u : 1024u;
                for (uint32_t i = threadIdx.x; i < nTab; i += blockDim.x)
                    sGrid[i] = src[i];
            }
            __syncthreads();
        }

        /// Batched prefill GEMV: four output rows per warp x M activations.
        ///
        /// The decode is hoisted out of the activation loop: for one weight
        /// block each lane decodes its [8l, 8l+8) weights ONCE into wv[r][k]
        /// (r = the four rows the warp owns), and the M activations then just
        /// multiply them.  That is the whole point of the batched shape: the
        /// decode is 3-20 ALU ops per weight while the multiply it feeds is
        /// one FMA, so a token batch of M amortises the decoder by M.
        ///
        /// Accumulators are fp32, one per (row, token): 4*M registers, which is
        /// why M is capped at 16.  The per-block contribution of (row, token)
        /// is summed over the lane's weights in the same order dotLane uses, so
        /// a batched call is bit-identical to M single-token calls.
        template<uint32_t Type, uint32_t M>
        __global__ void inklingGemvBatchedKernel(
                const uint8_t *__restrict__ w, const float *__restrict__ x,
                float *__restrict__ out, uint32_t rows, uint32_t cols,
                uint32_t blocksPerRow, uint32_t blockBytes, uint32_t m,
                bool alignedX) {
            constexpr uint32_t qk = tqk<Type>();
            constexpr uint32_t perLane = qk / 32u;
            constexpr uint32_t kRows = 4;
            constexpr uint32_t kWarps = 8;
            __shared__ float sIQ4[16];
            __shared__ uint8_t sSigns[128];
            __shared__ uint64_t sGrid[1024];
            stageDecodeTables<Type>(sIQ4, sSigns, sGrid);
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5u;
            const uint32_t o0 = (blockIdx.x * kWarps + warp) * kRows;
            if (o0 >= rows) return;
            const uint32_t nr = min(kRows, rows - o0);
            const uint8_t *rp[kRows];
#pragma unroll
            for (uint32_t r = 0; r < kRows; ++r)
                rp[r] = w + static_cast<uint64_t>(o0 + (r < nr ? r : 0)) *
                                    blocksPerRow * blockBytes;
            float acc[kRows][M];
#pragma unroll
            for (uint32_t r = 0; r < kRows; ++r)
#pragma unroll
                for (uint32_t s = 0; s < M; ++s) acc[r][s] = 0.0f;
            const uint32_t first = lane * perLane;
#pragma unroll 2
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                // One decode per weight block, reused by every activation.
                float wv[kRows][perLane];
#pragma unroll
                for (uint32_t r = 0; r < kRows; ++r)
                    if (r < nr)
                        decodeLaneVals<Type>(
                                rp[r] + static_cast<uint64_t>(b) * blockBytes,
                                lane, wv[r], sIQ4, sSigns, sGrid);
                const uint32_t base = b * qk + first;
#pragma unroll
                for (uint32_t s = 0; s < M; ++s) {
                    // Clamp (not break) so the loop unrolls and acc[4][M] stays
                    // in registers; a duplicated slot is recomputed and skipped
                    // at the write.  M is the launch's rounded-up tile, so at
                    // most half of the slots are duplicates.
                    const float *xp = x + static_cast<size_t>(s < m ? s : 0u) * cols + base;
                    float xb[8];
                    if (perLane == 2 && alignedX) {
                        const float2 a = *reinterpret_cast<const float2 *>(xp);
                        xb[0] = a.x;
                        xb[1] = a.y;
                    } else if (perLane == 8 && alignedX) {
                        const float4 a = *reinterpret_cast<const float4 *>(xp);
                        const float4 c = *reinterpret_cast<const float4 *>(xp + 4);
                        xb[0] = a.x;
                        xb[1] = a.y;
                        xb[2] = a.z;
                        xb[3] = a.w;
                        xb[4] = c.x;
                        xb[5] = c.y;
                        xb[6] = c.z;
                        xb[7] = c.w;
                    } else {
#pragma unroll
                        for (uint32_t k = 0; k < perLane; ++k)
                            xb[k] = (base + k < cols) ? __ldg(&xp[k]) : 0.0f;
                    }
#pragma unroll
                    for (uint32_t r = 0; r < kRows; ++r) {
                        if (r < nr) {
                            float t = 0.0f;
#pragma unroll
                            for (uint32_t k = 0; k < perLane; ++k)
                                t = fmaf(wv[r][k], xb[k], t);
                            acc[r][s] += t;
                        }
                    }
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
                for (uint32_t r = 0; r < kRows; ++r)
#pragma unroll
                    for (uint32_t s = 0; s < M; ++s)
                        acc[r][s] += __shfl_down_sync(0xffffffffu, acc[r][s], off);
            }
            if (lane == 0) {
#pragma unroll
                for (uint32_t s = 0; s < M; ++s) {
                    if (s >= m) break;
                    const uint64_t ob = static_cast<uint64_t>(s) * rows + o0;
#pragma unroll
                    for (uint32_t r = 0; r < kRows; ++r)
                        if (r < nr) out[ob + r] = acc[r][s];
                }
            }
        }


        template<uint32_t Type, uint32_t M>
        __global__ void inklingGemvTokParallelKernel(
                const uint8_t *__restrict__ w, const float *__restrict__ x,
                float *__restrict__ out, uint32_t rows, uint32_t cols,
                uint32_t blocksPerRow, uint32_t blockBytes, uint32_t m,
                bool alignedX) {
            constexpr uint32_t qk = tqk<Type>();
            constexpr uint32_t perLane = qk / 32u;
            constexpr uint32_t kRows = 4;
            constexpr uint32_t kWarps = 8;
            __shared__ float sIQ4[16];
            __shared__ uint8_t sSigns[128];
            __shared__ uint64_t sGrid[1024];
            stageDecodeTables<Type>(sIQ4, sSigns, sGrid);
            // Token-parallel form: blockIdx.y owns M tokens, so the grid is
            // ceil(rows/32) * ceil(m/M) blocks instead of ceil(rows/32).  At
            // the F32 router (128 rows) the row-only grid is 4 blocks on a
            // 68-SM part and the section measured 384 ms per 512-token pass.
            // The per-output dot order is unchanged (lane stride + the same
            // butterfly), so a call is still bit-identical to per-token GEMVs.
            const uint32_t tokBase = blockIdx.y * M;
            const uint32_t ms = m - tokBase < M ? m - tokBase : M;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5u;
            const uint32_t o0 = (blockIdx.x * kWarps + warp) * kRows;
            if (o0 >= rows) return;
            const uint32_t nr = min(kRows, rows - o0);
            const uint8_t *rp[kRows];
#pragma unroll
            for (uint32_t r = 0; r < kRows; ++r)
                rp[r] = w + static_cast<uint64_t>(o0 + (r < nr ? r : 0)) *
                                    blocksPerRow * blockBytes;
            float acc[kRows][M];
#pragma unroll
            for (uint32_t r = 0; r < kRows; ++r)
#pragma unroll
                for (uint32_t s = 0; s < M; ++s) acc[r][s] = 0.0f;
            const uint32_t first = lane * perLane;
#pragma unroll 2
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                // One decode per weight block, reused by every activation.
                float wv[kRows][perLane];
#pragma unroll
                for (uint32_t r = 0; r < kRows; ++r)
                    if (r < nr)
                        decodeLaneVals<Type>(
                                rp[r] + static_cast<uint64_t>(b) * blockBytes,
                                lane, wv[r], sIQ4, sSigns, sGrid);
                const uint32_t base = b * qk + first;
#pragma unroll
                for (uint32_t s = 0; s < M; ++s) {
                    // Clamp (not break) so the loop unrolls and acc[4][M] stays
                    // in registers; a duplicated slot is recomputed and skipped
                    // at the write.  M is the launch's rounded-up tile, so at
                    // most half of the slots are duplicates.
                    const float *xp = x + static_cast<size_t>(tokBase + (s < ms ? s : 0u)) * cols + base;
                    float xb[8];
                    if (perLane == 2 && alignedX) {
                        const float2 a = *reinterpret_cast<const float2 *>(xp);
                        xb[0] = a.x;
                        xb[1] = a.y;
                    } else if (perLane == 8 && alignedX) {
                        const float4 a = *reinterpret_cast<const float4 *>(xp);
                        const float4 c = *reinterpret_cast<const float4 *>(xp + 4);
                        xb[0] = a.x;
                        xb[1] = a.y;
                        xb[2] = a.z;
                        xb[3] = a.w;
                        xb[4] = c.x;
                        xb[5] = c.y;
                        xb[6] = c.z;
                        xb[7] = c.w;
                    } else {
#pragma unroll
                        for (uint32_t k = 0; k < perLane; ++k)
                            xb[k] = (base + k < cols) ? __ldg(&xp[k]) : 0.0f;
                    }
#pragma unroll
                    for (uint32_t r = 0; r < kRows; ++r) {
                        if (r < nr) {
                            float t = 0.0f;
#pragma unroll
                            for (uint32_t k = 0; k < perLane; ++k)
                                t = fmaf(wv[r][k], xb[k], t);
                            acc[r][s] += t;
                        }
                    }
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
                for (uint32_t r = 0; r < kRows; ++r)
#pragma unroll
                    for (uint32_t s = 0; s < M; ++s)
                        acc[r][s] += __shfl_down_sync(0xffffffffu, acc[r][s], off);
            }
            if (lane == 0) {
#pragma unroll
                for (uint32_t s = 0; s < M; ++s) {
                    if (s >= ms) break;
                    const uint64_t ob = static_cast<uint64_t>(tokBase + s) * rows + o0;
#pragma unroll
                    for (uint32_t r = 0; r < kRows; ++r)
                        if (r < nr) out[ob + r] = acc[r][s];
                }
            }
        }


        /// Token-parallel float-decoder launch: M=8 tokens per block (the
        /// register-light bucket) and the token groups in grid.y, so a narrow
        /// matrix still fills the device.
        template<uint32_t Type>
        bool launchTokParallel(const void *w, const float *x, float *out,
                               uint32_t rows, uint32_t cols, uint32_t m,
                               cudaStream_t stream) {
            constexpr uint32_t M = 8;
            constexpr uint32_t blockBytes = tbb<Type>();
            constexpr uint32_t qk = tqk<Type>();
            if (rows == 0 || cols == 0 || m == 0) return false;
            const uint32_t blocksPerRow = (cols + qk - 1) / qk;
            constexpr uint32_t kWarpsPerBlock = 8;
            constexpr uint32_t kRowsPerWarp = 4;
            const uint32_t perBlock = kWarpsPerBlock * kRowsPerWarp;
            const dim3 grid((rows + perBlock - 1) / perBlock, (m + M - 1) / M);
            inklingGemvTokParallelKernel<Type, M>
                    <<<grid, 32 * kWarpsPerBlock, 0, stream>>>(
                            static_cast<const uint8_t *>(w), x, out, rows, cols,
                            blocksPerRow, blockBytes, m, (cols % qk) == 0);
            return true;
        }

        bool launchTokParallelAny(uint32_t type, const void *w, const float *x,
                                  float *out, uint32_t rows, uint32_t cols,
                                  uint32_t m, cudaStream_t stream) {
            switch (type) {
                case 40:
                    return launchTokParallel<40>(w, x, out, rows, cols, m, stream);
                case 42:
                    return launchTokParallel<42>(w, x, out, rows, cols, m, stream);
                case 19:
                    return launchTokParallel<19>(w, x, out, rows, cols, m, stream);
                case 16:
                    return launchTokParallel<16>(w, x, out, rows, cols, m, stream);
                case 17:
                    return launchTokParallel<17>(w, x, out, rows, cols, m, stream);
                case 22:
                    return launchTokParallel<22>(w, x, out, rows, cols, m, stream);
                case 23:
                    return launchTokParallel<23>(w, x, out, rows, cols, m, stream);
                case 0:
                    return launchTokParallel<0>(w, x, out, rows, cols, m, stream);
                case 30:
                    return launchTokParallel<30>(w, x, out, rows, cols, m, stream);
                case 8:
                    return launchTokParallel<8>(w, x, out, rows, cols, m, stream);
                case 12:
                    return launchTokParallel<12>(w, x, out, rows, cols, m, stream);
                case 13:
                    return launchTokParallel<13>(w, x, out, rows, cols, m, stream);
                case 14:
                    return launchTokParallel<14>(w, x, out, rows, cols, m, stream);
                case 18:
                    return launchTokParallel<18>(w, x, out, rows, cols, m, stream);
                default:
                    return false;
            }
        }

        /// Dispatch the token count onto a compile-time M.  The kernel body is
        /// unrolled per M, so the buckets are the only instantiations: rounding
        /// m up wastes the idle `s >= m` iterations of the arm, it does not
        /// change the result.
        template<uint32_t Type, uint32_t M>
        bool launchBatchedT(const void *w, const float *x, float *out,
                            uint32_t rows, uint32_t cols, uint32_t m,
                            cudaStream_t stream) {
            constexpr uint32_t blockBytes = tbb<Type>();
            constexpr uint32_t qk = tqk<Type>();
            const uint32_t blocksPerRow = (cols + qk - 1) / qk;
            constexpr uint32_t kWarpsPerBlock = 8;
            constexpr uint32_t kRowsPerWarp = 4;
            const uint32_t perBlock = kWarpsPerBlock * kRowsPerWarp;
            const uint32_t blocks = (rows + perBlock - 1) / perBlock;
            inklingGemvBatchedKernel<Type, M>
                    <<<blocks, 32 * kWarpsPerBlock, 0, stream>>>(
                            static_cast<const uint8_t *>(w), x, out, rows, cols,
                            blocksPerRow, blockBytes, m, (cols % qk) == 0);
            return true;
        }

        template<uint32_t Type>
        bool launchBatchedM(const void *w, const float *x, float *out,
                            uint32_t rows, uint32_t cols, uint32_t m,
                            cudaStream_t stream) {
            if (m <= 1) return launchBatchedT<Type, 1>(w, x, out, rows, cols, m, stream);
            if (m <= 2) return launchBatchedT<Type, 2>(w, x, out, rows, cols, m, stream);
            if (m <= 4) return launchBatchedT<Type, 4>(w, x, out, rows, cols, m, stream);
            if (m <= 8) return launchBatchedT<Type, 8>(w, x, out, rows, cols, m, stream);
            return launchBatchedT<Type, 16>(w, x, out, rows, cols, m, stream);
        }

        bool launchBatched(uint32_t type, const void *w, const float *x,
                           float *out, uint32_t rows, uint32_t cols, uint32_t m,
                           cudaStream_t stream) {
            if (rows == 0 || cols == 0 || m == 0 || m > 16 || w == nullptr ||
                x == nullptr || out == nullptr)
                return false;
            switch (type) {
                case 40:
                    return launchBatchedM<40>(w, x, out, rows, cols, m, stream);
                case 42:
                    return launchBatchedM<42>(w, x, out, rows, cols, m, stream);
                case 19:
                    return launchBatchedM<19>(w, x, out, rows, cols, m, stream);
                case 16:
                    return launchBatchedM<16>(w, x, out, rows, cols, m, stream);
                case 17:
                    return launchBatchedM<17>(w, x, out, rows, cols, m, stream);
                case 22:
                    return launchBatchedM<22>(w, x, out, rows, cols, m, stream);
                case 23:
                    return launchBatchedM<23>(w, x, out, rows, cols, m, stream);
                case 0:
                    return launchBatchedM<0>(w, x, out, rows, cols, m, stream);
                case 30:
                    return launchBatchedM<30>(w, x, out, rows, cols, m, stream);
                case 8:
                    return launchBatchedM<8>(w, x, out, rows, cols, m, stream);
                case 12:
                    return launchBatchedM<12>(w, x, out, rows, cols, m, stream);
                case 13:
                    return launchBatchedM<13>(w, x, out, rows, cols, m, stream);
                case 14:
                    return launchBatchedM<14>(w, x, out, rows, cols, m, stream);
                case 18:
                    return launchBatchedM<18>(w, x, out, rows, cols, m, stream);
                default:
                    return false;
            }
        }

        /// Grouped batched GEMV: the same kernel with one expert group per
        /// blockIdx.y.  `x` is [nGroups][capacity][cols], `out` is
        /// [nGroups][capacity][rows] (the batched layout with a group index in
        /// front) and counts[g] is how many rows of group g are live; `w` is the
        /// base of group 0's matrix, so group g's rows start at
        /// g*rows*blocksPerRow*blockBytes.  Empty groups (counts[g] == 0) return
        /// before touching memory.
        ///
        /// This is how a routed MoE batch is processed: 8 experts are picked per
        /// token, m tokens pick ~m*nUsed distinct experts in total, and the
        /// weight decode is paid once per *distinct* expert instead of once per
        /// (token, expert) pair.
        template<uint32_t Type, uint32_t M>
        __global__ void inklingGemvBatchedGroupedKernel(
                const uint8_t *__restrict__ w, const float *__restrict__ x,
                float *__restrict__ out, uint32_t rows, uint32_t cols,
                uint32_t blocksPerRow, uint32_t blockBytes, uint32_t capacity,
                const int32_t *__restrict__ counts,
                const int32_t *__restrict__ off) {
            constexpr uint32_t qk = tqk<Type>();
            constexpr uint32_t perLane = qk / 32u;
            constexpr uint32_t kRows = 4;
            constexpr uint32_t kWarps = 8;
            __shared__ float sIQ4[16];
            __shared__ uint8_t sSigns[128];
            __shared__ uint64_t sGrid[1024];
            stageDecodeTables<Type>(sIQ4, sSigns, sGrid);
            const uint32_t g = blockIdx.y;
            const uint32_t m = static_cast<uint32_t>(counts[g]);
            if (m == 0) return;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5u;
            const uint32_t o0 = (blockIdx.x * kWarps + warp) * kRows;
            if (o0 >= rows) return;
            const uint32_t nr = min(kRows, rows - o0);
            const uint8_t *wg = w + static_cast<uint64_t>(g) * rows *
                                            blocksPerRow * blockBytes;
            const uint32_t base =
                    off != nullptr ? static_cast<uint32_t>(off[g]) : g * capacity;
            const float *xg = x + static_cast<size_t>(base) * cols;
            float *og = out + static_cast<size_t>(base) * rows;
            const uint8_t *rp[kRows];
#pragma unroll
            for (uint32_t r = 0; r < kRows; ++r)
                rp[r] = wg + static_cast<uint64_t>(o0 + (r < nr ? r : 0)) *
                                     blocksPerRow * blockBytes;
            const uint32_t first = lane * perLane;
            // The activation tile for one k block, shared by all 8 warps of the
            // block.  Without it every warp re-reads the same group rows for
            // every 32-row output block: at the down shape (rows=2304,
            // cols=1152, 72 blocks per expert) that is 72 x 8 warps x 8 tokens
            // x 1152 floats = 2.7 GB of activation traffic per layer, and it
            // measured 4.3 ms per layer -- exactly the DRAM time of that
            // traffic.  Staged once per block it is 8x less.
            __shared__ float sx[8][qk];
            // A group holds up to `capacity` rows, which is kBatchM (128) in the
            // layer-major path; the M template is 8 because the live
            // accumulators are 4 rows x M tokens per warp.  Walk the live rows
            // in M-token tiles: the mean group at m=128 is 8 tokens, so this is
            // one iteration and the expert's weights are decoded once per chunk.
            for (uint32_t s0 = 0; s0 < m; s0 += M) {
                const uint32_t ms = min(M, m - s0);
                float acc[kRows][M];
#pragma unroll
                for (uint32_t r = 0; r < kRows; ++r)
#pragma unroll
                    for (uint32_t s = 0; s < M; ++s) acc[r][s] = 0.0f;
#pragma unroll 2
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __syncthreads();
                    for (uint32_t i = threadIdx.x; i < M * qk; i += blockDim.x) {
                        const uint32_t si = i / qk;
                        const uint32_t kk = i - si * qk;
                        const uint32_t ts = s0 + (si < ms ? si : 0u);
                        const uint32_t gk = b * qk + kk;
                        sx[si][kk] = (ts < m && gk < cols)
                                             ? xg[static_cast<size_t>(ts) * cols + gk]
                                             : 0.0f;
                    }
                    __syncthreads();
                    float wv[kRows][perLane];
#pragma unroll
                    for (uint32_t r = 0; r < kRows; ++r)
                        if (r < nr)
                            decodeLaneVals<Type>(
                                    rp[r] + static_cast<uint64_t>(b) * blockBytes,
                                    lane, wv[r], sIQ4, sSigns, sGrid);
                    const uint32_t base = b * qk + first;
#pragma unroll
                    for (uint32_t s = 0; s < M; ++s) {
                        // Clamp rather than break: a branch on s keeps the loop
                        // from unrolling and drops the 4 x M accumulators into
                        // local memory, which measured ~25x slower than the
                        // arithmetic predicts.  Duplicated slots are recomputed
                        // and skipped at the write.
                        const uint32_t ts = s < ms ? s : 0u;
                        const float *xp = sx[ts] + first;
                        float xb[8];
                        if (perLane == 2 && (cols % qk) == 0) {
                            const float2 a = *reinterpret_cast<const float2 *>(xp);
                            xb[0] = a.x;
                            xb[1] = a.y;
                        } else if (perLane == 8 && (cols % qk) == 0) {
                            const float4 a = *reinterpret_cast<const float4 *>(xp);
                            const float4 c = *reinterpret_cast<const float4 *>(xp + 4);
                            xb[0] = a.x;
                            xb[1] = a.y;
                            xb[2] = a.z;
                            xb[3] = a.w;
                            xb[4] = c.x;
                            xb[5] = c.y;
                            xb[6] = c.z;
                            xb[7] = c.w;
                        } else {
#pragma unroll
                            for (uint32_t k = 0; k < perLane; ++k)
                                xb[k] = (base + k < cols) ? __ldg(&xp[k]) : 0.0f;
                        }
#pragma unroll
                        for (uint32_t r = 0; r < kRows; ++r) {
                            if (r < nr) {
                                float t = 0.0f;
#pragma unroll
                                for (uint32_t k = 0; k < perLane; ++k)
                                    t = fmaf(wv[r][k], xb[k], t);
                                acc[r][s] += t;
                            }
                        }
                    }
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
                    for (uint32_t r = 0; r < kRows; ++r)
#pragma unroll
                        for (uint32_t s = 0; s < M; ++s)
                            acc[r][s] += __shfl_down_sync(0xffffffffu, acc[r][s], off);
                }
                if (lane == 0) {
#pragma unroll
                    for (uint32_t s = 0; s < M; ++s) {
                        if (s >= ms) break;
                        const uint64_t ob =
                                static_cast<uint64_t>(s0 + s) * rows + o0;
#pragma unroll
                        for (uint32_t r = 0; r < kRows; ++r)
                            if (r < nr) og[ob + r] = acc[r][s];
                    }
                }
            }
        }

        template<uint32_t Type, uint32_t M>
        bool launchGroupedT(const void *w, const float *x, float *out,
                            uint32_t rows, uint32_t cols, uint32_t capacity,
                            const int32_t *counts, const int32_t *off,
                            uint32_t nGroups, cudaStream_t stream) {
            constexpr uint32_t blockBytes = tbb<Type>();
            constexpr uint32_t qk = tqk<Type>();
            const uint32_t blocksPerRow = (cols + qk - 1) / qk;
            constexpr uint32_t kWarpsPerBlock = 8;
            constexpr uint32_t kRowsPerWarp = 4;
            const uint32_t perBlock = kWarpsPerBlock * kRowsPerWarp;
            const dim3 grid((rows + perBlock - 1) / perBlock, nGroups);
            inklingGemvBatchedGroupedKernel<Type, M>
                    <<<grid, 32 * kWarpsPerBlock, 0, stream>>>(
                            static_cast<const uint8_t *>(w), x, out, rows, cols,
                            blocksPerRow, blockBytes, capacity, counts, off);
            return true;
        }

        template<uint32_t Type>
        bool launchGroupedM(const void *w, const float *x, float *out,
                            uint32_t rows, uint32_t cols, uint32_t capacity,
                            const int32_t *counts, const int32_t *off,
                            uint32_t nGroups, cudaStream_t stream) {
            if (capacity <= 1)
                return launchGroupedT<Type, 1>(w, x, out, rows, cols, capacity,
                                               counts, off, nGroups, stream);
            if (capacity <= 2)
                return launchGroupedT<Type, 2>(w, x, out, rows, cols, capacity,
                                               counts, off, nGroups, stream);
            if (capacity <= 4)
                return launchGroupedT<Type, 4>(w, x, out, rows, cols, capacity,
                                               counts, off, nGroups, stream);
            // M is capped at 8, not 16.  Every live row is walked in M-token
            // tiles whatever M is, and the mean group at m=128 is 8 tokens, so
            // M=16 only doubles the wasted slots; it also doubles the register
            // footprint of the 4 x M accumulators, which is what pushes this
            // kernel over the register ceiling.
            return launchGroupedT<Type, 8>(w, x, out, rows, cols, capacity,
                                           counts, off, nGroups, stream);
        }

        bool launchGrouped(uint32_t type, const void *w, const float *x,
                           float *out, uint32_t rows, uint32_t cols,
                           uint32_t capacity, const int32_t *counts,
                           const int32_t *off, uint32_t nGroups,
                           cudaStream_t stream) {
            // capacity is the group *stride* (kBatchM in the layer-major path),
            // not the live count: the kernel walks counts[g] rows in M-token
            // tiles, so any stride is fine and a sparse group still touches its
            // weights once.
            if (rows == 0 || cols == 0 || capacity == 0 ||
                nGroups == 0 || w == nullptr || x == nullptr || out == nullptr ||
                counts == nullptr)
                return false;
            switch (type) {
                case 40:
                    return launchGroupedM<40>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 42:
                    return launchGroupedM<42>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 19:
                    return launchGroupedM<19>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 16:
                    return launchGroupedM<16>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 17:
                    return launchGroupedM<17>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 22:
                    return launchGroupedM<22>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 23:
                    return launchGroupedM<23>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 0:
                    return launchGroupedM<0>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 30:
                    return launchGroupedM<30>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 8:
                    return launchGroupedM<8>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 12:
                    return launchGroupedM<12>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 13:
                    return launchGroupedM<13>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 14:
                    return launchGroupedM<14>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                case 18:
                    return launchGroupedM<18>(w, x, out, rows, cols, capacity, counts, off, nGroups, stream);
                default:
                    return false;
            }
        }

        bool launch(uint32_t type, const void *w, const float *x, float *out,
                    uint32_t rows, uint32_t cols, const int32_t *rowIdx,
                    uint32_t rowsPer, uint32_t xStride, cudaStream_t stream) {
            if (rows == 0 || cols == 0 || w == nullptr || x == nullptr ||
                out == nullptr)
                return false;
            switch (type) {
                case 40:
                    return launchT<40>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 42:
                    return launchT<42>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 19:
                    return launchT<19>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 16:
                    return launchT<16>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 17:
                    return launchT<17>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 22:
                    return launchT<22>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 23:
                    return launchT<23>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 0:
                    return launchT<0>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 30:
                    return launchT<30>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 8:
                    return launchT<8>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 12:
                    return launchT<12>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 13:
                    return launchT<13>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 14:
                    return launchT<14>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                case 18:
                    return launchT<18>(w, x, out, rows, cols, rowIdx, rowsPer, xStride, stream);
                default:
                    return false;
            }
        }

    }// namespace

    bool inklingGemvAsync(uint32_t type, const void *w, const float *x, float *out,
                          uint32_t rows, uint32_t cols, void *stream) {
        return launch(type, w, x, out, rows, cols, nullptr, 0, 0,
                      reinterpret_cast<cudaStream_t>(stream));
    }

    bool inklingGemvExpertAsync(uint32_t type, const void *w, const float *x,
                                uint32_t xStride, float *out,
                                uint32_t rowsPerExpert, uint32_t cols,
                                const int32_t *idx, uint32_t nUsed,
                                void *stream) {
        if (idx == nullptr || nUsed == 0 || rowsPerExpert == 0) return false;
        return launch(type, w, x, out, rowsPerExpert * nUsed, cols, idx,
                      rowsPerExpert, xStride,
                      reinterpret_cast<cudaStream_t>(stream));
    }

    bool inklingGemv(uint32_t type, const void *w, const float *x, float *out,
                     uint32_t rows, uint32_t cols, void *stream) {
        if (!inklingGemvAsync(type, w, x, out, rows, cols, stream)) return false;
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            fprintf(stderr, "[inklingGemv] sync %s\n", cudaGetErrorString(err));
        return err == cudaSuccess;
    }

    bool inklingGemvBatched(uint32_t type, const void *w, const float *x,
                            float *out, uint32_t rows, uint32_t cols, uint32_t m,
                            void *stream) {
        if (!launchBatched(type, w, x, out, rows, cols, m,
                           reinterpret_cast<cudaStream_t>(stream)))
            return false;
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            fprintf(stderr, "[inklingGemvBatched] sync %s\n",
                    cudaGetErrorString(err));
        return err == cudaSuccess;
    }

    bool inklingGemvBatchedAsync(uint32_t type, const void *w, const float *x,
                                 float *out, uint32_t rows, uint32_t cols,
                                 uint32_t m, void *stream) {
        return launchBatched(type, w, x, out, rows, cols, m,
                             reinterpret_cast<cudaStream_t>(stream));
    }

    bool inklingGemvBatchedSmallAsync(uint32_t type, const void *w,
                                      const float *x, float *out, uint32_t rows,
                                      uint32_t cols, uint32_t m, void *stream) {
        if (m == 0 || rows == 0) return false;
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        switch (type) {
            case 0:
                return launchSmall<0>(w, x, out, rows, cols, m, s);
            case 8:
                return launchSmall<8>(w, x, out, rows, cols, m, s);
            case 30:
                return launchSmall<30>(w, x, out, rows, cols, m, s);
            case 12:
                return launchSmall<12>(w, x, out, rows, cols, m, s);
            case 13:
                return launchSmall<13>(w, x, out, rows, cols, m, s);
            case 14:
                return launchSmall<14>(w, x, out, rows, cols, m, s);
            case 16:
                return launchSmall<16>(w, x, out, rows, cols, m, s);
            case 17:
                return launchSmall<17>(w, x, out, rows, cols, m, s);
            case 18:
                return launchSmall<18>(w, x, out, rows, cols, m, s);
            case 19:
                return launchSmall<19>(w, x, out, rows, cols, m, s);
            case 22:
                return launchSmall<22>(w, x, out, rows, cols, m, s);
            case 23:
                return launchSmall<23>(w, x, out, rows, cols, m, s);
            default:
                return false;
        }
    }

    bool inklingGemvBatchedTokenParallelAsync(uint32_t type, const void *w,
                                              const float *x, float *out,
                                              uint32_t rows, uint32_t cols,
                                              uint32_t m, void *stream) {
        if (m == 0 || rows == 0) return false;
        return launchTokParallelAny(type, w, x, out, rows, cols, m,
                                    reinterpret_cast<cudaStream_t>(stream));
    }

    bool inklingGemvBatchedGroupedAsync(uint32_t type, const void *w,
                                        const float *x, float *out,
                                        uint32_t rows, uint32_t cols,
                                        uint32_t capacity, const int32_t *counts,
                                        uint32_t nGroups, void *stream) {
        return launchGrouped(type, w, x, out, rows, cols, capacity, counts,
                             nullptr, nGroups,
                             reinterpret_cast<cudaStream_t>(stream));
    }

    bool inklingGemvBatchedGroupedOffsetsAsync(uint32_t type, const void *w,
                                               const float *x, float *out,
                                               uint32_t rows, uint32_t cols,
                                               const int32_t *counts,
                                               const int32_t *off,
                                               uint32_t nGroups, void *stream) {
        if (off == nullptr || counts == nullptr) return false;
        return launchGrouped(type, w, x, out, rows, cols, 8u, counts, off,
                             nGroups, reinterpret_cast<cudaStream_t>(stream));
    }

}// namespace tinycoder::gpu
