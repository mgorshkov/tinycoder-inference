/*
 ⚡ TinyCoder AI

 Copyright (c) 2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
*/

// Grouped int8 tensor-core MMQ for the routed experts (IQ2_XXS / IQ2_XS /
// IQ2_S / IQ1_S gate/up, Q2_0 / NVFP4 down), the gate/up counterpart of
// gemm/Q2_0Mma.cu.
//
// The batched prefill graph groups the m*nUsed routed (token, expert) pairs by
// expert and runs one projection per distinct expert.  The exact path runs the
// token path's int8 GEMV in 4-row slices, which streams an expert's weight
// matrix once per four tokens; this kernel instead runs one int8 MMA tile per
// (expert, 64 output rows), so an expert's weights are decoded and multiplied
// once per group and the MMA does 8 live tokens in one instruction.
//
// Structure (same shape as Q2_0Mma.cu, which is the working grouped MMQ):
//   * activation Q8_1 quantized once for the whole group, [e][cap][cols] int8
//     payload + [e][cap][cols/32] fp32 scales;
//   * the expert's weight slab for the block's 64 output rows (the whole row,
//     every k block) is staged into shared memory with a contiguous 16-byte
//     copy -- the direct per-row 82-byte global read is one 32 B sector per
//     row and measured ~55 GB/s on this part, while a contiguous slab moves at
//     DRAM speed;
//   * each 64-k chunk is decoded from the slab into the int8 shared tile the
//     mma.s8 loaders expect, [64 rows][16 ints + 4 scale slots];
//   * 8 warps x one m8n8 tile each = 64 output columns, 16-token M tiles.
//
// The default kernel is instead the 32-token k-tile form of llama.cpp's
// mul_mat_q<..., int mmq_x = 32>: one block owns a 32-token M tile and 64
// output rows, stages and decodes only the current weight k-tile, and reuses
// that one decode for all 32 token rows (see "32-token k-tile MMQ" below).
//
// The one numeric difference from the token path is the Q8_1 activation
// quantization, i.e. the int8 format's own floor (~2e-3 relRMS on continuous
// activations), exactly as for the dense int8 GEMM and the Q2_0 down MMQ.
// TINYCODER_INKLING_EXPERT_MMA is on by default (=0 opts out); the
// model-level gate is in tools/inkling_expert_mma_parity.cpp and the
// batched-prefill test.
//
// IQ scale granularity: IQ2_XS/IQ2_S carry a scale per 16 weights (the low/high
// nibble of scales[ib32] for l<2 / l>=2), so the k loop steps 16 weights per
// MMA (4 sub-steps per 64-k chunk) and keeps four weight scales per chunk.
// IQ2_XXS has one scale per 32 weights and simply repeats it in the pair.
// IQ1_S carries one scale per 32 weights and an 8-weight grid entry; its 1-bit
// lattice is asymmetric (grid value +-1/8), which is folded into the int8
// payload as 8*nib - 7 / 8*nib - 9 with a 1/8 scale, so the decode is exact.
// NVFP4 is 36 B per 64 weights: four UE4M3 scales (one per 16 weights) and 32
// packed E2M1 nibbles, so its k loop is exactly the four 16-weight MMA steps of
// one 64-k chunk with the four scale slots used as-is.
#include "../gemm/Int8Mma.cuh"
#include "../gemm/QuantDot.cuh"
#include "gpu/InklingKernels.hpp"

#ifdef USE_CUDA

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tinycoder::gpu {
    namespace {

        constexpr int kMTile = 16;// token rows in one MMA tile
        constexpr int kNRow = 64; // output rows (weight rows) per block
        constexpr int kN8 = 8;    // one m8n8 tile per warp
        constexpr int kWarps = 8; // 8 x n8 = 64 output columns
        constexpr int kThreads = kWarps * 32;
        constexpr int kKChunk = 64; // weights per k iteration (16 ints)
        constexpr int kAInts = 16;  // int8 payload ints per 64-k row
        constexpr int kAStride = 20;// 16 payload ints + 2 activation scales
        constexpr int kBStride = 20;// 16 payload ints + 4 weight scales
        constexpr int kSteps = 4;   // 16-weight MMA steps per 64-k chunk

        /// Bytes per quant block and how many 64-weight k chunks one block
        /// holds, per supported type.
        template<uint32_t Type>
        __host__ __device__ constexpr uint32_t blockBytes() {
            if constexpr (Type == 16) return 66;
            else if constexpr (Type == 17)
                return 74;
            else if constexpr (Type == 19)
                return 50;
            else if constexpr (Type == 22)
                return 82;
            else if constexpr (Type == 40)
                return 36;
            else if constexpr (Type == 42)
                return 18;
            else
                return 0;
        }
        template<uint32_t Type>
        __host__ __device__ constexpr uint32_t kChunksPerBlock() {
            // Q2_0 and NVFP4 are 64 weights per block; the IQ family is 256.
            return (Type == 40 || Type == 42) ? 1u : 4u;
        }

        // 2-bit codes in element order -> four int8 values (d*(code-1) is
        // applied as the scale, so the payload is code-1).  Same sequence as
        // Q2_0Mma.cu's q2Decode4.
        __device__ __forceinline__ uint32_t q2Decode4(const uint32_t b) {
            constexpr uint32_t LUT = 0x020100FFu;// {-1, 0, 1, 2}
            const uint32_t ev = __byte_perm(LUT, LUT, b);
            const uint32_t od = __byte_perm(LUT, LUT, b >> 2);
            return __byte_perm(ev, od, 0x5140);
        }

        /// ksigns_iq2xs as a broadcast sign word: bit j set means weight j is
        /// negative (same arithmetic as QuantDot.cuh's unpack_ksigns).
        __device__ __forceinline__ uint32_t unpackSigns(const uint8_t v) {
            const uint32_t p = __popc(v) & 1u;
            return (v ^ (p << 7)) * 0x01010101u;
        }

        /// NVFP4's E2M1 codebook in nibble order (GGMLDequantize's
        /// kvalues_fp4, also InklingGemv.cu's kvalues_nvfp4): the high bit is
        /// the sign, bits 0-2 the magnitude code.  QuantBlocks.cuh's
        /// kvalues_mxfp4 is the MXFP4 ordering and decodes this layout with the
        /// wrong value for 11 of 16 nibbles.
        static const __device__ int8_t kvaluesNvfp4[16] = {
                0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

        /// One 16-weight MMA step decoded into four ints (16 signed int8 in
        /// weight order) plus its fp32 scale.  The sign application is the
        /// llama.cpp __vcmpne4/__vsub4 trick: build a per-byte 0xFF mask from
        /// the sign bits, then (grid ^ mask) - mask negates exactly the masked
        /// bytes.  A per-byte loop costs ~3x more and is the difference
        /// between this kernel beating the 4-row slice path and losing to it.
        template<uint32_t Type>
        __device__ __forceinline__ void decodeStep(const uint8_t *blk,
                                                   uint32_t s32, uint32_t half,
                                                   int *dst, float *sc) {
            // `half` is the 16-weight step index within the 64-k chunk (0..3):
            // for the IQ family one 32-weight sub-block holds two steps, so the
            // step is the low/high half of that sub-block; Q2_0 has one scale
            // per 64 weights and the step is directly the code quarter.
            const uint32_t l0 = 2u * (half & 1u);
            if constexpr (Type == 22) {// IQ2_S: d, qs[64], qh[8], scales[8]
                // The shared slab's row stride (738 B) is only 2-byte aligned,
                // and a 32-bit load from shared memory must be naturally
                // aligned, so the two qs bytes this step needs come in as one
                // 16-bit load (l0 is even) -- same for the two sign bytes.
                const block_iq2_s *b = reinterpret_cast<const block_iq2_s *>(blk);
                const uint16_t qpair = *reinterpret_cast<const uint16_t *>(
                        b->qs + 4u * s32 + l0);
                const int qh = b->qh[s32];
                const uint16_t spair = *reinterpret_cast<const uint16_t *>(
                        b->qs + 32u + 4u * s32 + l0);
                const uint8_t qsb[2] = {static_cast<uint8_t>(qpair & 0xFFu),
                                        static_cast<uint8_t>(qpair >> 8)};
                const uint8_t sp[2] = {static_cast<uint8_t>(spair & 0xFFu),
                                       static_cast<uint8_t>(spair >> 8)};
                *sc = __half2float(b->d) *
                      (0.5f + static_cast<float>((b->scales[s32] >> (4u * (half & 1u))) & 0xFu)) *
                      0.25f;
#pragma unroll
                for (uint32_t jj = 0; jj < 2; ++jj) {
                    const int *gp = reinterpret_cast<const int *>(
                            iq2s_grid +
                            (qsb[jj] |
                             ((qh << (8 - 2 * static_cast<int>(l0 + jj))) & 0x300)));
                    const int s0 = __vcmpne4(
                            ((sp[jj] & 0x03) << 7) | ((sp[jj] & 0x0C) << 21), 0);
                    const int s1 = __vcmpne4(
                            ((sp[jj] & 0x30) << 3) | ((sp[jj] & 0xC0) << 17), 0);
                    dst[2 * jj] = __vsub4(gp[0] ^ s0, s0);
                    dst[2 * jj + 1] = __vsub4(gp[1] ^ s1, s1);
                }
            } else if constexpr (Type == 17) {// IQ2_XS: d, qs[32] u16, scales[8]
                const block_iq2_xs *b = reinterpret_cast<const block_iq2_xs *>(blk);
                const uint16_t q0 = b->qs[4u * s32 + l0];
                const uint16_t q1 = b->qs[4u * s32 + l0 + 1];
                *sc = __half2float(b->d) *
                      (0.5f + static_cast<float>((b->scales[s32] >> (4u * (half & 1u))) & 0xFu)) *
                      0.25f;
#pragma unroll
                for (uint32_t jj = 0; jj < 2; ++jj) {
                    const uint16_t q2 = jj == 0 ? q0 : q1;
                    const uint2 gp = reinterpret_cast<const uint2 *>(
                            iq2xs_grid)[q2 & 0x1FFu];
                    const uint32_t sv = unpackSigns(static_cast<uint8_t>(q2 >> 9));
                    const int s0 = __vcmpne4(sv & 0x08040201, 0);
                    const int s1 = __vcmpne4(sv & 0x80402010, 0);
                    dst[2 * jj] = __vsub4(static_cast<int>(gp.x) ^ s0, s0);
                    dst[2 * jj + 1] = __vsub4(static_cast<int>(gp.y) ^ s1, s1);
                }
            } else if constexpr (Type == 16) {// IQ2_XXS: d, qs[32] u16
                const block_iq2_xxs *b = reinterpret_cast<const block_iq2_xxs *>(blk);
                // 16-bit loads only: the shared row stride is 2-byte aligned.
                const uint32_t q0 = b->qs[4u * s32 + 0];
                const uint32_t q1 = b->qs[4u * s32 + 1];
                const uint32_t q2 = b->qs[4u * s32 + 2];
                const uint32_t q3 = b->qs[4u * s32 + 3];
                const uint32_t w0 = q0 | (q1 << 16);
                const uint32_t w1 = q2 | (q3 << 16);
                *sc = __half2float(b->d) *
                      (0.5f + static_cast<float>(w1 >> 28)) * 0.25f;
#pragma unroll
                for (uint32_t jj = 0; jj < 2; ++jj) {
                    const uint32_t l = l0 + jj;
                    const uint2 gp = reinterpret_cast<const uint2 *>(
                            iq2xxs_grid)[(w0 >> (8u * l)) & 0xFFu];
                    const uint32_t sv = unpackSigns(
                            static_cast<uint8_t>((w1 >> (7u * l)) & 127u));
                    const int s0 = __vcmpne4(sv & 0x08040201, 0);
                    const int s1 = __vcmpne4(sv & 0x80402010, 0);
                    dst[2 * jj] = __vsub4(static_cast<int>(gp.x) ^ s0, s0);
                    dst[2 * jj + 1] = __vsub4(static_cast<int>(gp.y) ^ s1, s1);
                }
            } else if constexpr (Type == 19) {// IQ1_S: d, qs[32], qh[8] u16, 256 w
                // The shared slab's row stride (50 B) is only 2-byte aligned,
                // so qh comes in as one naturally aligned 16-bit load and the
                // grid indices as single bytes.
                const block_iq1_s *b = reinterpret_cast<const block_iq1_s *>(blk);
                const uint16_t qh = *reinterpret_cast<const uint16_t *>(
                        b->qh + s32);
                // iq1s_grid_gpu stores the 1-bit lattice nibble-packed as
                // (CPU grid value + 1), and the block delta is +-IQ1S_DELTA,
                // so weight = d1q * (grid +- 0.125) = (d1q/8) * (8*nib - 7 or
                // 8*nib - 9).  Folding the offset into the payload keeps one
                // scale per 16-weight step -- i.e. exactly the layout the
                // other types use -- and is exact: 8*nib - c <= 113 fits int8
                // and d1q/8 is an exact power-of-two rescale.
                const uint32_t bias = (qh & 0x8000u) ? 0x09090909u : 0x07070707u;
                *sc = __half2float(b->d) *
                      (0.5f + static_cast<float>((qh >> 12) & 7u)) * 0.25f;
#pragma unroll
                for (uint32_t jj = 0; jj < 2; ++jj) {
                    const uint32_t l = l0 + jj;// 8-weight grid slot in the group
                    const uint32_t grid = iq1s_grid_gpu[b->qs[4u * s32 + l] |
                                                        ((qh >> (3u * l) & 7u) << 8)];
                    dst[2 * jj] = __vsub4(
                            static_cast<int>((grid & 0x0F0F0F0Fu) << 3),
                            static_cast<int>(bias));
                    dst[2 * jj + 1] = __vsub4(
                            static_cast<int>(((grid >> 4) & 0x0F0F0F0Fu) << 3),
                            static_cast<int>(bias));
                }
            } else if constexpr (Type == 40) {// NVFP4: d[4] UE4M3, qs[32], 64 w
                const block_nvfp4 *b = reinterpret_cast<const block_nvfp4 *>(blk);
                // One 16-weight MMA step = one UE4M3 scale slot = 8 nibble
                // bytes.  `half` is that slot (0..3) because a 64-weight block
                // is one k chunk.  get_int_from_table_16 returns the even and
                // odd nibble indices of its 4 bytes as two ints, i.e. for
                // bytes 0-3 the low nibbles are elements 0-3 and the high
                // nibbles elements 8-11, so the four fragment ints are
                // q0.x,q1.x,q0.y,q1.y -- the reference's "elements 0-7 are the
                // low nibbles of the 8 bytes, 8-15 the high nibbles".
                const uint32_t *qs = reinterpret_cast<const uint32_t *>(b->qs);
                const int2 q0 =
                        get_int_from_table_16(qs[2u * half + 0u], kvaluesNvfp4);
                const int2 q1 =
                        get_int_from_table_16(qs[2u * half + 1u], kvaluesNvfp4);
                dst[0] = q0.x;
                dst[1] = q1.x;
                dst[2] = q0.y;
                dst[3] = q1.y;
                *sc = tc_cuda_ue4m3_to_fp32(b->d[half]);
            } else if constexpr (Type == 42) {// Q2_0: fp16 d, 16 code bytes
                const uint8_t *qs = blk + 2;
                *sc = __half2float(*reinterpret_cast<const __half *>(blk));
#pragma unroll
                for (uint32_t jj = 0; jj < 4; ++jj)
                    dst[jj] = static_cast<int>(q2Decode4(qs[4u * half + jj]));
            }
        }

        /// Two m8n8k16 instead of the four that make a logical m16n8k32: the
        /// second 8-token half is dead for a group with <= 8 live rows, which
        /// is the mean at m=128 (1024 routed pairs / 128 experts = 8).
        __device__ __forceinline__ void mmaHalf(
                tc_mma::tile<16, 8, int> &D, const tc_mma::tile<16, 4, int> &A,
                const tc_mma::tile<8, 4, int> &B) {
#if defined(TURING_MMA_AVAILABLE)
            asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
                : "+r"(D.x[0]), "+r"(D.x[1])
                : "r"(A.x[0]), "r"(B.x[0]));
#else
            tc_mma::mma(D, A, B);
#endif
        }

        /// `counts[e]` live token rows of expert e start at row `off[e]` of the
        /// gathered activation / output arrays.  With `off == nullptr` the rows
        /// start at e*capacity (the padded [nExperts][capacity] layout); the
        /// layer-major graph passes a compact prefix-offset array instead, so a
        /// group of any size costs one weight decode and no padded storage.
        template<uint32_t Type>
        __global__ void __launch_bounds__(kThreads) expertMmaKernel(
                const char *__restrict__ w, const int8_t *__restrict__ aq,
                const float *__restrict__ ad, float *__restrict__ out,
                uint32_t rows, uint32_t cols, uint32_t capacity,
                const int32_t *__restrict__ counts,
                const int32_t *__restrict__ off) {
            constexpr uint32_t kBB = blockBytes<Type>();
            const uint32_t e = blockIdx.y;
            const int32_t cnt = counts[e];
            if (cnt <= 0) return;
            const uint32_t base =
                    off != nullptr ? static_cast<uint32_t>(off[e]) : e * capacity;

            constexpr uint32_t kBlocksPerChunk = kChunksPerBlock<Type>();
            const uint32_t kBlocks = cols / (kKChunk * kBlocksPerChunk);
            const uint32_t kbCount = cols / kKChunk;// 64-k chunks per row
            const uint32_t rowStride = kBlocks * kBB;
            const uint32_t n0 = blockIdx.x * kNRow;
            const uint32_t nRows = n0 + kNRow <= rows ? kNRow : rows - n0;
            // tc_mma's loaders index fragments by threadIdx.x, so the block is
            // dim3(32, kWarps): lane in x, warp in y (as in Q2_0Mma.cu).
            const uint32_t lane = threadIdx.x;
            const uint32_t warp = threadIdx.y;
            const uint32_t tid = warp * 32u + lane;

            extern __shared__ __align__(16) char smem[];
            uint8_t *sW = reinterpret_cast<uint8_t *>(smem);
            int *sA = reinterpret_cast<int *>(smem + kNRow * rowStride);
            int *sB = sA + kMTile * kAStride;

            // Stage the whole [kNRow][rowStride] slab contiguously.  Rows are
            // adjacent in the tensor, so the block's region is one contiguous
            // nRows*rowStride span; 16-byte copies keep enough loads in flight
            // to run at DRAM speed (the 82-byte-per-row loader does not).
            const char *wBase = w + static_cast<size_t>(e) * rows * rowStride +
                                static_cast<size_t>(n0) * rowStride;
            const size_t nBytes = static_cast<size_t>(nRows) * rowStride;
            const bool aligned = ((reinterpret_cast<uintptr_t>(wBase) & 15u) == 0u) &&
                                 ((nBytes & 15u) == 0u);
            if (aligned) {
                const size_t n16 = nBytes / 16u;
                const int4 *src = reinterpret_cast<const int4 *>(wBase);
                int4 *dst = reinterpret_cast<int4 *>(sW);
                for (size_t i = tid; i < n16; i += kThreads) dst[i] = src[i];
            } else {
                for (size_t i = tid; i < nBytes; i += kThreads) sW[i] = wBase[i];
            }
            for (uint32_t i = tid; i < kMTile * kAStride; i += kThreads) sA[i] = 0;
            __syncthreads();

            const uint32_t i0 = lane / 4u;
            const uint32_t j0 = (lane % 4u) * 2u;
            const uint32_t bRow = warp * kN8;
            const int8_t *aqe = aq + static_cast<size_t>(base) * cols;
            const float *ade = ad + static_cast<size_t>(base) * (cols / 32u);
            float *oute = out + static_cast<size_t>(base) * rows;

            for (uint32_t s0 = 0; s0 < static_cast<uint32_t>(cnt); s0 += kMTile) {
                const uint32_t rowsLive = static_cast<uint32_t>(cnt) - s0 < kMTile
                                                  ? static_cast<uint32_t>(cnt) - s0
                                                  : kMTile;
                const bool twoHalves = rowsLive > 8u;
                float acc[4] = {};

                for (uint32_t kb = 0; kb < kbCount; ++kb) {
                    __syncthreads();
                    // decode the 64-k chunk of all kNRow weight rows: one
                    // thread per (row, 16-weight step)
                    {
                        const uint32_t n = tid / kSteps;
                        const uint32_t part = tid % kSteps;
                        if (n < kNRow) {
                            const uint8_t *blk = sW +
                                                 static_cast<size_t>(n) * rowStride +
                                                 static_cast<size_t>(kb / kBlocksPerChunk) * kBB;
                            const uint32_t s32 =
                                    kBlocksPerChunk == 1u
                                            ? 0u
                                            : 2u * (kb & 3u) + (part >> 1);
                            int *dst = sB + n * kBStride + part * 4;
                            float *sc = reinterpret_cast<float *>(sB + n * kBStride + kAInts) +
                                        part;
                            decodeStep<Type>(blk, s32, part, dst, sc);
                        }
                    }
                    // activation tile for the same 64-k chunk
                    if (tid < kMTile) {
                        const int *src = reinterpret_cast<const int *>(
                                aqe + static_cast<size_t>(s0 + tid) * cols + kb * kKChunk);
                        int *dst = sA + tid * kAStride;
#pragma unroll
                        for (int j = 0; j < kAInts / 4; ++j)
                            reinterpret_cast<int4 *>(dst)[j] =
                                    reinterpret_cast<const int4 *>(src)[j];
                        dst[16] = __float_as_int(
                                ade[static_cast<size_t>(s0 + tid) * (cols / 32u) + 2u * kb]);
                        dst[17] = __float_as_int(
                                ade[static_cast<size_t>(s0 + tid) * (cols / 32u) + 2u * kb + 1u]);
                    }
                    __syncthreads();

#pragma unroll
                    for (uint32_t ks = 0; ks < kSteps; ++ks) {
                        tc_mma::tile<16, 4, int> A;
                        tc_mma::load_ldmatrix(A, sA + ks * 4, kAStride);
                        const float a0 = __int_as_float(
                                sA[i0 * kAStride + kAInts + (ks >> 1)]);
                        const float a1 = __int_as_float(
                                sA[(i0 + 8u) * kAStride + kAInts + (ks >> 1)]);
                        tc_mma::tile<8, 4, int> B;
                        tc_mma::load_generic(B, sB + bRow * kBStride + ks * 4, kBStride);
                        tc_mma::tile<16, 8, int> C;
                        if (twoHalves)
                            tc_mma::mma(C, A, B);
                        else
                            mmaHalf(C, A, B);
                        const float b0 = *reinterpret_cast<const float *>(
                                sB + (bRow + j0) * kBStride + kAInts + ks);
                        const float b1 = *reinterpret_cast<const float *>(
                                sB + (bRow + j0 + 1) * kBStride + kAInts + ks);
                        acc[0] += a0 * b0 * static_cast<float>(C.x[0]);
                        acc[1] += a0 * b1 * static_cast<float>(C.x[1]);
                        if (twoHalves) {
                            acc[2] += a1 * b0 * static_cast<float>(C.x[2]);
                            acc[3] += a1 * b1 * static_cast<float>(C.x[3]);
                        }
                    }
                }

                const uint32_t col = n0 + bRow + j0;
                if (i0 < rowsLive && col < rows) {
                    oute[static_cast<size_t>(s0 + i0) * rows + col] = acc[0];
                    if (col + 1 < rows)
                        oute[static_cast<size_t>(s0 + i0) * rows + col + 1] = acc[1];
                }
                if (twoHalves && i0 + 8u < rowsLive) {
                    if (col < rows) {
                        oute[static_cast<size_t>(s0 + i0 + 8u) * rows + col] = acc[2];
                        if (col + 1 < rows)
                            oute[static_cast<size_t>(s0 + i0 + 8u) * rows + col + 1] = acc[3];
                    }
                }
            }
        }

        // ---- weight-stationary variant ------------------------------------
        //
        // The kernel above decodes its 64 weight rows once per 16-token M tile,
        // so a group of `cnt` tokens pays ceil(cnt/16) decodes of the whole
        // slab: the grouped gate/up pass at m=512 is sum_e ceil(cnt_e/16) =
        // 11941 tile decodes, and one tile's cost splits decode 26% /
        // activation fill 26% / staging+MMA 39%.  26% of a latency-bound kernel
        // is not tensor throughput -- 8 warps at one block per SM reach 2.3
        // weights/cycle/SM, 1.8% of the int pipes.
        //
        // This kernel makes the slab stationary instead: a block owns only
        // kWsNRow output rows, so the *entire* K range of those rows is decoded
        // into shared once per block, and every token group then multiplies the
        // resident int8 tile.  Decode per pass becomes a single pass over the
        // expert's matrix; sum_e ceil(cnt_e/M) is gone from it.  What remains is
        // the activation side, and that is where the design differs twice over:
        //
        //   * each warp owns a private 16-row activation tile, written by the
        //     warp that reads it, so the k loop carries no block barrier at all
        //     (__syncwarp only) and warps drift across k independently -- a
        //     warp whose group is small still covers its own load latency
        //     because the other warps keep the SM busy;
        //   * two lanes per token row fetch it (one 32 B sector each, the 64 B
        //     payload of a row lands in two adjacent sectors) into registers
        //     one chunk ahead, while the current chunk's MMA runs.
        //
        // Arithmetic is unchanged element for element: for a given (token,row)
        // the k order is still kb ascending then ks ascending, every m8n8k16 is
        // issued with the same operands as the 16-token tile's half, and the
        // Q8_1 quantizer and the epilogue are identical -- the output is
        // bit-identical to the kernel above, not merely inside the band.
        constexpr int kWsNRow = 16;                // output rows per block
        constexpr int kWsNT = kWsNRow / kN8;       // n8 tiles per warp
        constexpr int kWsWarps = 12;               // 12 x 16 tokens = M tile
        constexpr int kWsMTile = kWsWarps * kMTile;// 192-token M tile
        constexpr int kWsThreads = kWsWarps * 32;

        template<uint32_t Type>
        __global__ void __launch_bounds__(kWsThreads) expertMmaStationary(
                const char *__restrict__ w, const int8_t *__restrict__ aq,
                const float *__restrict__ ad, float *__restrict__ out,
                uint32_t rows, uint32_t cols, uint32_t capacity,
                const int32_t *__restrict__ counts,
                const int32_t *__restrict__ off) {
            constexpr uint32_t kBB = blockBytes<Type>();
            constexpr uint32_t kBlocksPerChunk = kChunksPerBlock<Type>();
            const uint32_t e = blockIdx.y;
            const int32_t cnt = counts[e];
            if (cnt <= 0) return;
            const uint32_t base =
                    off != nullptr ? static_cast<uint32_t>(off[e]) : e * capacity;

            const uint32_t kBlocks = cols / (kKChunk * kBlocksPerChunk);
            const uint32_t kbCount = cols / kKChunk;
            const uint32_t rowStride = kBlocks * kBB;
            const uint32_t n0 = blockIdx.x * kWsNRow;
            const uint32_t nRows = n0 + kWsNRow <= rows ? kWsNRow : rows - n0;
            const uint32_t lane = threadIdx.x;
            const uint32_t warp = threadIdx.y;
            const uint32_t tid = warp * 32u + lane;
            const uint32_t i0 = lane / 4u;
            const uint32_t j0 = (lane % 4u) * 2u;

            extern __shared__ __align__(16) char smem[];
            // `smem` first holds the raw slab of this block's rows (the whole
            // row, every k block) for the one-time decode, then the per-warp
            // activation tiles.  sB holds every k chunk of this block's kWsNRow
            // rows: [kb][row][16 int8 payload + 4 scales].
            char *sRaw = smem;
            int *sA = reinterpret_cast<int *>(smem);
            int *sB = reinterpret_cast<int *>(smem + kWsMTile * kAStride * 4);

            // ---- one-time decode of this block's rows ----------------------
            // The rows are adjacent in the tensor, so the block's slab is one
            // contiguous nRows*rowStride span (DRAM speed, not one 32 B sector
            // per row); then one thread per (row, 64-k chunk, 16-weight step).
            {
                const char *wBase = w + static_cast<size_t>(e) * rows * rowStride +
                                    static_cast<size_t>(n0) * rowStride;
                const size_t nBytes = static_cast<size_t>(nRows) * rowStride;
                const bool aligned =
                        ((reinterpret_cast<uintptr_t>(wBase) & 15u) == 0u) &&
                        ((nBytes & 15u) == 0u);
                if (aligned) {
                    // PROBE: staging disabled -- this branch stages nothing for
                    // an aligned row block, which is what "n16 = 0" always did.
                    // The bound is a constant zero, so nvcc reports the (dead)
                    // loop's `unsigned < 0` as #186; the code shape is kept
                    // because it is an experiment switch, not dead weight.
#pragma nv_diag_suppress 186
                    const size_t n16 = 0;
                    const int4 *src = reinterpret_cast<const int4 *>(wBase);
                    int4 *dst = reinterpret_cast<int4 *>(sRaw);
                    for (size_t i = tid; i < n16; i += kWsThreads) dst[i] = src[i];
#pragma nv_diag_default 186
                } else {
                    for (size_t i = tid; i < nBytes; i += kWsThreads)
                        sRaw[i] = wBase[i];
                }
                __syncthreads();
                const uint32_t steps = nRows * kbCount * kSteps;
                for (uint32_t i = tid; i < steps; i += kWsThreads) {
                    const uint32_t part = i % kSteps;
                    const uint32_t rk = i / kSteps;
                    const uint32_t kb = rk % kbCount;
                    const uint32_t r = rk / kbCount;
                    const uint8_t *blk = reinterpret_cast<const uint8_t *>(sRaw) +
                                         static_cast<size_t>(r) * rowStride +
                                         (kb / kBlocksPerChunk) * kBB;
                    const uint32_t s32 = kBlocksPerChunk == 1u
                                                 ? 0u
                                                 : 2u * (kb & 3u) + (part >> 1);
                    int *dst = sB + (kb * kWsNRow + r) * kBStride + part * 4;
                    float *sc =
                            reinterpret_cast<float *>(sB +
                                                      (kb * kWsNRow + r) * kBStride +
                                                      kAInts) +
                            part;
                    decodeStep<Type>(blk, s32, part, dst, sc);
                }
                __syncthreads();
            }

            const int8_t *aqe = aq + static_cast<size_t>(base) * cols;
            const float *ade = ad + static_cast<size_t>(base) * (cols / 32u);
            float *oute = out + static_cast<size_t>(base) * rows;
            const uint32_t aStrideRow = cols / 32u;

            // Each warp owns a private 16-row activation tile, so the k loop
            // needs no block barrier: a warp writes the slice it is about to
            // read (__syncwarp is enough) and warps drift across the k loop
            // independently.
            int *sAw = sA + warp * kMTile * kAStride;
            // Two lanes per token row, 32 B each: the two halves of a row's
            // 64 B payload land in the same 32 B sector, so one 16 B request
            // per row instead of two (the fetch is sector-throughput bound, not
            // bandwidth bound -- see the kernel comment).
            const uint32_t aRow = lane >> 1; // 16 token rows across the warp
            const uint32_t aHalf = lane & 1u;// payload ints [8*aHalf, 8*aHalf+8)
            auto loadAct = [&](const uint32_t kb, const uint32_t tok, int4 &v0,
                               int4 &v1, float &d0, float &d1) {
                const int4 *src = reinterpret_cast<const int4 *>(
                        aqe + static_cast<size_t>(tok) * cols + kb * kKChunk);
                v0 = src[2u * aHalf];
                v1 = src[2u * aHalf + 1u];
                if (aHalf == 0u) {
                    d0 = ade[static_cast<size_t>(tok) * aStrideRow + 2u * kb];
                    d1 = ade[static_cast<size_t>(tok) * aStrideRow + 2u * kb + 1u];
                }
            };
            auto storeAct = [&](const int4 v0, const int4 v1, const float d0,
                                const float d1) {
                int *dst = sAw + aRow * kAStride + aHalf * 8u;
                reinterpret_cast<int4 *>(dst)[0] = v0;
                reinterpret_cast<int4 *>(dst)[1] = v1;
                if (aHalf == 0u) {
                    dst[16] = __float_as_int(d0);
                    dst[17] = __float_as_int(d1);
                }
            };

            const uint32_t cntU = static_cast<uint32_t>(cnt);
            // Warp w owns token rows base+0..15 of each group; warps walk a
            // group's rows with a kWsMTile stride and retire as soon as their
            // rows are exhausted (no barrier holds them behind).
            for (uint32_t base = warp * kMTile; base < cntU; base += kWsMTile) {
                const uint32_t rowStart = base + aRow;
                const uint32_t left = cntU - base;
                // A warp's slice is 16 rows, so only the lanes of live rows
                // may load: the rest would take the next warp's rows.
                const uint32_t liveW = left < kMTile ? left : kMTile;
                const bool laneLive = aRow < liveW;

                int4 v0, v1;
                float d0 = 0.0f, d1 = 0.0f;
                if (laneLive) loadAct(0, rowStart, v0, v1, d0, d1);
                __syncwarp();
                if (laneLive) storeAct(v0, v1, d0, d1);
                __syncwarp();

                const bool twoHalves = liveW > 8u;
                float acc[kWsNT][4] = {};
                for (uint32_t kb = 0; kb < kbCount; ++kb) {
                    const bool more = kb + 1u < kbCount;
                    if (more && laneLive) loadAct(kb + 1u, rowStart, v0, v1, d0, d1);
#pragma unroll
                    for (uint32_t ks = 0; ks < kSteps; ++ks) {
                        tc_mma::tile<16, 4, int> A;
                        tc_mma::load_ldmatrix(A, sAw + ks * 4, kAStride);
                        const float a0 = __int_as_float(
                                sAw[i0 * kAStride + kAInts + (ks >> 1)]);
                        const float a1 = __int_as_float(
                                sAw[(i0 + 8u) * kAStride + kAInts + (ks >> 1)]);
#pragma unroll
                        for (int t = 0; t < kWsNT; ++t) {
                            const int row0 = static_cast<int>(kb) * kWsNRow + t * kN8;
                            tc_mma::tile<8, 4, int> B;
                            tc_mma::load_generic(
                                    B, sB + row0 * kBStride + ks * 4, kBStride);
                            tc_mma::tile<16, 8, int> C;
                            if (twoHalves)
                                tc_mma::mma(C, A, B);
                            else
                                mmaHalf(C, A, B);
                            const float b0 = *reinterpret_cast<const float *>(
                                    sB + (row0 + j0) * kBStride + kAInts + ks);
                            const float b1 = *reinterpret_cast<const float *>(
                                    sB + (row0 + j0 + 1u) * kBStride + kAInts + ks);
                            acc[t][0] += a0 * b0 * static_cast<float>(C.x[0]);
                            acc[t][1] += a0 * b1 * static_cast<float>(C.x[1]);
                            if (twoHalves) {
                                acc[t][2] += a1 * b0 * static_cast<float>(C.x[2]);
                                acc[t][3] += a1 * b1 * static_cast<float>(C.x[3]);
                            }
                        }
                    }
                    if (more) {
                        __syncwarp();
                        if (laneLive) storeAct(v0, v1, d0, d1);
                        __syncwarp();
                    }
                }

                const uint32_t r0 = base + i0;
                const uint32_t r1 = r0 + 8u;
#pragma unroll
                for (int t = 0; t < kWsNT; ++t) {
                    const uint32_t col = n0 + t * kN8 + j0;
                    if (r0 < cntU && col < rows) {
                        oute[static_cast<size_t>(r0) * rows + col] = acc[t][0];
                        if (col + 1 < rows)
                            oute[static_cast<size_t>(r0) * rows + col + 1] = acc[t][1];
                    }
                    if (r1 < cntU && col < rows) {
                        oute[static_cast<size_t>(r1) * rows + col] = acc[t][2];
                        if (col + 1 < rows)
                            oute[static_cast<size_t>(r1) * rows + col + 1] = acc[t][3];
                    }
                }
            }
        }

        // ---- 32-token k-tile MMQ ------------------------------------------
        //
        // The 16-token kernel above re-decodes its 64-row weight tile once per
        // 16-token M tile, and the weight-stationary form below decodes it once
        // but keeps the whole K range of a block's rows resident: that costs one
        // block/SM for every gate/up type and, on the mean group
        // (m*nUsed/nExperts = 32 rows at m=512), leaves all but 2 of its 12
        // warps idle.
        //
        // This form follows llama.cpp's mul_mat_q<..., int mmq_x = 32>: a block
        // owns a 32-token M tile and 64 output rows, one ggml k-tile of the
        // weights (256 weights for the IQ family, 64 for Q2_0/NVFP4) is decoded
        // into shared once per block and then multiplied by all 32 token rows,
        // and only that k-tile is resident.  So the shared footprint is one
        // decoded k-tile -- 4*64*20 ints, 20 KB for gate/up and 5 KB for down --
        // and two blocks stay resident per SM.
        //
        // The activation fragments are loaded straight from the gathered Q8_1
        // array instead of from a shared tile: a lane's mma.s8 A fragment is the
        // four consecutive int8 at row lane/4, k offset 4*(lane%4), which is one
        // 32-bit load.  That removes the per-64-k-chunk activation staging and
        // its barrier: two barriers per k-tile, one after the weight decode and
        // one before the next decode overwrites it.
        //
        // Arithmetic is unchanged element for element: per (row, token) the k
        // order is still kb ascending then ks ascending, with the same int8
        // operands from the same decodeStep and the same Q8_1 scales, and the
        // fp32 epilogue is the same multiply-accumulate expression.  The output
        // is bit-identical to both kernels above, not merely inside a band.
        constexpr int kKtNRow = 128;// output (weight) rows per block
        constexpr int kKtWarps = 8; // 8 warps x two m8n8 tiles = 128 columns
        constexpr int kKtThreads = kKtWarps * 32;
        constexpr int kKtMTile = 32;// token rows per M tile

        template<uint32_t Type>
        __host__ __device__ constexpr int ktileChunks() {
            // One staged k-tile is two 64-k chunks of the IQ family's 256-weight
            // block (one for Q2_0/NVFP4, whose block is 64 weights).  With 128
            // output rows per block that is 25 KB of shared, so two blocks are
            // resident per SM.
            return kChunksPerBlock<Type>() == 1u ? 1 : 2;
        }

        template<uint32_t Type>
        __global__ void __launch_bounds__(kKtThreads, 2) expertMmaKtile(
                const char *__restrict__ w, const int8_t *__restrict__ aq,
                const float *__restrict__ ad, float *__restrict__ out,
                uint32_t rows, uint32_t cols, uint32_t capacity,
                const int32_t *__restrict__ counts,
                const int32_t *__restrict__ off) {
            constexpr uint32_t kBB = blockBytes<Type>();
            constexpr uint32_t kChunksPer = kChunksPerBlock<Type>();
            constexpr int kTileChunks = ktileChunks<Type>();
            const uint32_t e = blockIdx.y;
            const int32_t cnt = counts[e];
            if (cnt <= 0) return;
            const uint32_t base =
                    off != nullptr ? static_cast<uint32_t>(off[e]) : e * capacity;
            const uint32_t kBlocks = cols / (kKChunk * kTileChunks);
            const uint32_t rowStride = (cols / (kKChunk * kChunksPer)) * kBB;
            const uint32_t n0 = blockIdx.x * kKtNRow;
            const uint32_t nRows = n0 + kKtNRow <= rows ? kKtNRow : rows - n0;
            const uint32_t lane = threadIdx.x;
            const uint32_t warp = threadIdx.y;
            const uint32_t tid = warp * 32u + lane;
            const uint32_t i0 = lane / 4u;
            const uint32_t j0 = (lane % 4u) * 2u;
            // 128 output rows / 8 warps = two m8n8 tiles per warp.
            const int bRow = static_cast<int>(warp) * (2 * kN8);

            extern __shared__ __align__(16) char smem[];
            // [chunk][row][16 int8 payload + 4 scale slots] decoded weights,
            // then the same layout for the Q8_1 activation tile.
            int *sB = reinterpret_cast<int *>(smem);
            int *sA = sB + kTileChunks * kKtNRow * kBStride;

            // Decode thread: kStepsPerTile lanes per weight row, so a warp's
            // 16-bit reads land in two rows' blocks (one 32 B sector each)
            // instead of eight rows' worth -- that is the difference between a
            // coalesced and a scattered weight read.
            constexpr uint32_t kStepsPerTile = kSteps * kTileChunks;
            constexpr uint32_t kRowsPerPass = kKtThreads / kStepsPerTile;
            constexpr uint32_t kPasses = kKtNRow / kRowsPerPass;
            const uint32_t dRowBase = tid / kStepsPerTile;
            const uint32_t dStep = tid % kStepsPerTile;
            const uint32_t dChunk = dStep / kSteps;
            const uint32_t dKs = dStep % kSteps;

            const char *wBase = w + static_cast<size_t>(e) * rows * rowStride +
                                static_cast<size_t>(n0) * rowStride;
            if (nRows < static_cast<uint32_t>(kKtNRow)) {
                for (int i = static_cast<int>(tid);
                     i < (kTileChunks * kKtNRow + kTileChunks * kKtMTile) * kBStride;
                     i += kKtThreads)
                    reinterpret_cast<int *>(smem)[i] = 0;
                __syncthreads();
            }

            const int8_t *aqe = aq + static_cast<size_t>(base) * cols;
            const float *ade = ad + static_cast<size_t>(base) * (cols / 32u);
            float *oute = out + static_cast<size_t>(base) * rows;
            const uint32_t aStrideRow = cols / 32u;
            const uint32_t cntU = static_cast<uint32_t>(cnt);

            for (uint32_t s0 = 0; s0 < cntU; s0 += kKtMTile) {
                const uint32_t left = cntU - s0;
                const uint32_t live = left < kKtMTile ? left : kKtMTile;
                const bool half0Full = live > 8u;
                const bool half1Live = live > 16u;
                const bool half1Full = live > 24u;

                // acc[half][tile][element]
                float acc[2][2][4] = {};

                // ---- software pipeline for the activation tile ------------
                // The fill is one 16 B global load per thread and the barrier
                // right after it would expose that latency once per k-tile; it
                // is the largest single phase (>45% of the k-tile).  Instead the
                // next k-tile's tile is loaded into registers at the top of the
                // iteration, the current k-tile's decode and mma run while the
                // load is in flight, and the tile is stored to shared after the
                // second barrier (all warps are then done reading sA).
                constexpr uint32_t kChunkInt4 = kKtMTile * (kAInts / 4);
                constexpr uint32_t kChunkScales = kKtMTile * 2u;
                constexpr uint32_t nInt4 = kTileChunks * kChunkInt4;
                constexpr uint32_t nScale = kTileChunks * kChunkScales;
                const uint32_t fC = tid < nInt4 ? tid / kChunkInt4 : 0u;
                const uint32_t fRem = tid < nInt4 ? tid - fC * kChunkInt4 : 0u;
                const uint32_t fT = fRem / (kAInts / 4);
                const uint32_t fSeg = fRem - fT * (kAInts / 4);
                const uint32_t fTl = fT < live ? fT : live - 1u;
                const uint32_t sC = tid < nScale ? tid / kChunkScales : 0u;
                const uint32_t sRem = tid < nScale ? tid - sC * kChunkScales : 0u;
                const uint32_t sT = sRem >> 1;
                const uint32_t sJ = sRem & 1u;
                const uint32_t sTl = sT < live ? sT : live - 1u;
                auto loadTileA = [&](const uint32_t kt, int4 &v, float &sv) {
                    if (tid < nInt4)
                        v = *reinterpret_cast<const int4 *>(
                                aqe + static_cast<size_t>(s0 + fTl) * cols +
                                static_cast<size_t>(kt * kTileChunks + fC) * kKChunk +
                                fSeg * 16u);
                    if (tid < nScale)
                        sv = ade[static_cast<size_t>(s0 + sTl) * aStrideRow +
                                 2u * (kt * kTileChunks + sC) + sJ];
                };
                auto storeTileA = [&](const int4 v, const float sv) {
                    if (tid < nInt4)
                        *reinterpret_cast<int4 *>(
                                sA + (fC * kKtMTile + fT) * kBStride + fSeg * 4) = v;
                    if (tid < nScale)
                        reinterpret_cast<float *>(
                                sA + (sC * kKtMTile + sT) * kBStride + kAInts)[sJ] = sv;
                };
                {
                    int4 v;
                    float sv = 0.0f;
                    loadTileA(0, v, sv);
                    storeTileA(v, sv);
                }
                __syncthreads();

                for (uint32_t kt = 0; kt < kBlocks; ++kt) {
                    const uint32_t kb0 = kt * kTileChunks;
                    // issue the next k-tile's activation tile into registers
                    int4 av;
                    float asv = 0.0f;
                    const bool moreTile = kt + 1u < kBlocks;
                    if (moreTile) loadTileA(kt + 1u, av, asv);
                    // ---- weight decode for this k-tile ----
                    {
                        // weights: this k-tile of the block's rows
                        const uint32_t blkIdx = kb0 / kChunksPer;
                        const uint32_t dc = (kb0 % kChunksPer) + dChunk;
                        const uint32_t s32 = kChunksPer == 1u
                                                     ? 0u
                                                     : 2u * dc + (dKs >> 1);
                        int *dstRow =
                                sB + static_cast<int>(dChunk * kKtNRow) * kBStride +
                                static_cast<int>(dKs) * 4;
                        float *dstSc = reinterpret_cast<float *>(
                                               sB + static_cast<int>(dChunk * kKtNRow) * kBStride +
                                               kAInts) +
                                       dKs;
#pragma unroll
                        for (uint32_t rp = 0; rp < kPasses; ++rp) {
                            const uint32_t r = dRowBase + rp * kRowsPerPass;
                            if (r < nRows) {
                                const uint8_t *blk = reinterpret_cast<const uint8_t *>(
                                        wBase + static_cast<size_t>(r) * rowStride +
                                        static_cast<size_t>(blkIdx) * kBB);
                                decodeStep<Type>(
                                        blk, s32, dKs,
                                        dstRow + static_cast<int>(r) * kBStride,
                                        dstSc + static_cast<int>(r) * kBStride);
                            }
                        }
                    }
                    __syncthreads();

                    // ---- 32 tokens x 128 rows x this k-tile -----------------
#pragma unroll
                    for (int c = 0; c < kTileChunks; ++c) {
                        tc_mma::tile<16, 4, int> A0[kSteps];
                        tc_mma::tile<16, 4, int> A1[kSteps];
                        tc_mma::tile<8, 4, int> B0[kSteps];
                        tc_mma::tile<8, 4, int> B1[kSteps];
                        const int *aTile = sA + (c * kKtMTile) * kBStride;
                        const int *bTile =
                                sB + (c * kKtNRow + bRow) * kBStride;
#pragma unroll
                        for (int ks = 0; ks < kSteps; ++ks) {
                            // All fragment loads for the chunk are hoisted, so
                            // the mma.s8 below issue back to back instead of
                            // waiting on one load each.
                            tc_mma::load_ldmatrix(A0[ks], aTile + ks * 4, kBStride);
                            if (half1Live)
                                tc_mma::load_ldmatrix(
                                        A1[ks], aTile + 16 * kBStride + ks * 4,
                                        kBStride);
                            tc_mma::load_generic(B0[ks], bTile + ks * 4, kBStride);
                            tc_mma::load_generic(B1[ks],
                                                 bTile + kN8 * kBStride + ks * 4,
                                                 kBStride);
                        }
#pragma unroll
                        for (int ks = 0; ks < kSteps; ++ks) {
                            const uint32_t si = static_cast<uint32_t>(ks) >> 1;
                            // One scale product per (token row, tile, step): the
                            // same product the per-element form computes, in the
                            // same left-to-right order, with the FMULs hoisted.
                            const float sb[2][2] = {
                                    {*reinterpret_cast<const float *>(
                                             bTile + j0 * kBStride + kAInts + ks),
                                     *reinterpret_cast<const float *>(
                                             bTile + (j0 + 1) * kBStride + kAInts + ks)},
                                    {*reinterpret_cast<const float *>(
                                             bTile + (kN8 + j0) * kBStride + kAInts + ks),
                                     *reinterpret_cast<const float *>(
                                             bTile + (kN8 + j0 + 1) * kBStride +
                                             kAInts + ks)}};
#pragma unroll
                            for (int h = 0; h < 2; ++h) {
                                if (h == 1 && !half1Live) continue;
                                const bool full = h == 0 ? half0Full : half1Full;
                                const tc_mma::tile<16, 4, int> &Ah =
                                        h == 0 ? A0[ks] : A1[ks];
                                const float sa0 = __int_as_float(
                                        aTile[(16 * h + i0) * kBStride + kAInts + si]);
                                const float sa1 = __int_as_float(
                                        aTile[(16 * h + i0 + 8) * kBStride + kAInts + si]);
#pragma unroll
                                for (int t = 0; t < 2; ++t) {
                                    tc_mma::tile<16, 8, int> C;
                                    if (full)
                                        tc_mma::mma(C, Ah, t == 0 ? B0[ks] : B1[ks]);
                                    else if (t == 0)
                                        mmaHalf(C, Ah, B0[ks]);
                                    else
                                        mmaHalf(C, Ah, B1[ks]);
                                    const float t0 = sa0 * sb[t][0];
                                    const float t1 = sa0 * sb[t][1];
                                    const float t2 = sa1 * sb[t][0];
                                    const float t3 = sa1 * sb[t][1];
                                    acc[h][t][0] += t0 * static_cast<float>(C.x[0]);
                                    acc[h][t][1] += t1 * static_cast<float>(C.x[1]);
                                    if (full) {
                                        acc[h][t][2] += t2 * static_cast<float>(C.x[2]);
                                        acc[h][t][3] += t3 * static_cast<float>(C.x[3]);
                                    }
                                }
                            }
                        }
                    }
                    __syncthreads();
                    if (moreTile) {
                        storeTileA(av, asv);
                        __syncthreads();
                    }
                }

                const uint32_t rA = s0 + i0;
                const uint32_t rB = rA + 8u;
                const uint32_t rC = rA + 16u;
                const uint32_t rD = rA + 24u;
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const uint32_t col = n0 + static_cast<uint32_t>(bRow) +
                                         static_cast<uint32_t>(t * kN8) + j0;
                    if (rA < cntU && col < rows) {
                        oute[static_cast<size_t>(rA) * rows + col] = acc[0][t][0];
                        if (col + 1u < rows)
                            oute[static_cast<size_t>(rA) * rows + col + 1u] =
                                    acc[0][t][1];
                    }
                    if (half0Full && rB < cntU && col < rows) {
                        oute[static_cast<size_t>(rB) * rows + col] = acc[0][t][2];
                        if (col + 1u < rows)
                            oute[static_cast<size_t>(rB) * rows + col + 1u] =
                                    acc[0][t][3];
                    }
                    if (half1Live && rC < cntU && col < rows) {
                        oute[static_cast<size_t>(rC) * rows + col] = acc[1][t][0];
                        if (col + 1u < rows)
                            oute[static_cast<size_t>(rC) * rows + col + 1u] =
                                    acc[1][t][1];
                    }
                    if (half1Full && rD < cntU && col < rows) {
                        oute[static_cast<size_t>(rD) * rows + col] = acc[1][t][2];
                        if (col + 1u < rows)
                            oute[static_cast<size_t>(rD) * rows + col + 1u] =
                                    acc[1][t][3];
                    }
                }
            }
        }

        /// Q8_1 quantize the gathered activations.  Same convention as
        /// Q2_0Mma.cu's q2QuantizeKernel: d = amax/127, payload = round(v/d).
        ///
        /// One warp quantizes one 32-weight chunk, so the block is
        /// (32, warps) and the grid is (chunk groups, slots).  The previous
        /// form was one block per expert with a serial step over
        /// cnt*chunks/8 chunks (~288 dependent iterations per warp at m=512,
        /// 128 experts), every one of them waiting on its own load: measured
        /// 2.9 ms per layer, i.e. ~137 ms of a 512-token pass between the two
        /// grouped launches (gate/up 91.6, down 45.8), with the expert MMA
        /// kernel and its input bits unchanged.  The arithmetic per chunk is
        /// unchanged and every element is written by exactly the same lane.
        __global__ void __launch_bounds__(256) expertQuantKernel(
                const float *__restrict__ x, int8_t *__restrict__ aq,
                float *__restrict__ ad, uint32_t cols, uint32_t capacity,
                uint32_t nExperts, const int32_t *__restrict__ counts,
                const int32_t *__restrict__ off, uint32_t nSlots) {
            const uint32_t slot = blockIdx.y;
            if (slot >= nSlots) return;
            if (off == nullptr) {
                // Padded [nExperts][capacity] layout: slot e*capacity+r is
                // live only while r < counts[e].
                const uint32_t e = slot / capacity;
                if (e >= nExperts) return;
                if (static_cast<int32_t>(slot - e * capacity) >= counts[e]) return;
            }
            const uint32_t chunks = cols / 32u;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5u;
            const uint32_t warps = blockDim.x >> 5u;
            for (uint32_t kc = blockIdx.x * warps + warp; kc < chunks;
                 kc += gridDim.x * warps) {
                const size_t at = static_cast<size_t>(slot) * cols + kc * 32u;
                const float v = x[at + lane];
                float amax = fabsf(v);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
                const float d = amax > 0.0f ? amax * (1.0f / 127.0f) : 0.0f;
                const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                int q = __float2int_rn(v * inv);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                aq[at + lane] = static_cast<int8_t>(q);
                if (lane == 0)
                    ad[static_cast<size_t>(slot) * chunks + kc] = d;
            }
        }

        void *g_emScratch = nullptr;
        size_t g_emBytes = 0;
        [[maybe_unused]] int32_t g_countsHost[1024];

        bool ensureScratch(size_t bytes) {
            if (bytes <= g_emBytes) return true;
            if (g_emScratch != nullptr) {
                cudaFree(g_emScratch);
                g_emScratch = nullptr;
                g_emBytes = 0;
            }
            if (cudaMalloc(&g_emScratch, bytes) != cudaSuccess) {
                cudaGetLastError();
                g_emScratch = nullptr;
                return false;
            }
            g_emBytes = bytes;
            return true;
        }

        template<uint32_t Type>
        bool launchExpertMma(const void *w, const float *x, float *out,
                             uint32_t rows, uint32_t cols, uint32_t capacity,
                             const int32_t *counts, const int32_t *off,
                             uint32_t nSlots, uint32_t nExperts, cudaStream_t s,
                             bool quantize) {
            constexpr uint32_t kBB = blockBytes<Type>();
            if (kBB == 0) return false;
            constexpr uint32_t kPer = kKChunk * kChunksPerBlock<Type>();
            if (cols == 0u || cols % kPer != 0u || rows == 0) return false;
            const uint32_t kBlocks = cols / kPer;

            const size_t aqBytes = static_cast<size_t>(nSlots) * cols;
            const size_t adBytes = static_cast<size_t>(nSlots) * (cols / 32u) *
                                   sizeof(float);
            if (!ensureScratch(aqBytes + adBytes)) return false;
            int8_t *aq = static_cast<int8_t *>(g_emScratch);
            float *ad = reinterpret_cast<float *>(aq + aqBytes);

            if (quantize) {
                // 4 chunks per warp: enough independent work to amortise the
                // block, few enough that the grid still covers every SM.
                const uint32_t chunks = cols / 32u;
                const uint32_t groups = std::max(1u, (chunks + 31u) / 32u);
                const dim3 qGrid(groups, nSlots);
                expertQuantKernel<<<qGrid, 256, 0, s>>>(x, aq, ad, cols, capacity,
                                                        nExperts, counts, off,
                                                        nSlots);
                if (cudaGetLastError() != cudaSuccess) return false;
            }

            const uint32_t rowStride = kBlocks * kBB;
            const size_t smemBytes = static_cast<size_t>(kNRow) * rowStride +
                                     (kMTile * kAStride + kNRow * kBStride) * sizeof(int);
            // Weight-stationary geometry: kWsNRow output rows per block with
            // every k chunk of them resident in shared, a kWsMTile-token M tile
            // and per-warp activation slices.  The raw slab of those rows is
            // staged in the activation space before the decode, so it has to
            // fit there.
            const uint32_t kbCount = cols / kKChunk;
            static size_t g_smemOptin = 0;
            if (g_smemOptin == 0) {
                int optin = 0;
                int device = 0;
                if (cudaGetDevice(&device) == cudaSuccess &&
                    cudaDeviceGetAttribute(&optin,
                                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                           device) == cudaSuccess &&
                    optin > 0)
                    g_smemOptin = static_cast<size_t>(optin);
                else
                    g_smemOptin = 1;
            }
            // 32-token k-tile form: a block owns a 32-token M tile and 64
            // output rows, and only the current weight k-tile is resident, so
            // the shared footprint is one decoded k-tile (20 KB for the IQ
            // family, 5 KB for Q2_0/NVFP4) and two blocks stay resident per SM.
            // Default ON.  Per (row, token) it computes exactly the same
            // product order as the 16-token and weight-stationary forms below,
            // so its output is bit-identical to them (tools/
            // inkling_expert_mma_parity --dump, all six types), and it is the
            // only one of the three that is deterministic across fresh
            // processes at m=512: the stationary form's per-warp activation
            // tile is not (four fresh processes, four different 512-token
            // logit md5s, and it can answer the Paris question with noise).
            // It is 3-6% slower on the moe-gate section (309 vs 290 ms per
            // pass) because at m=512 sum_e ceil(cnt_e/32) = 199 M tiles per
            // layer against the stationary form's single decode.
            // TINYCODER_INKLING_MOE_KT=0 selects that form for the A/B.
            const size_t ktSmemBytes =
                    static_cast<size_t>(ktileChunks<Type>()) *
                    (kKtNRow + kKtMTile) * kBStride * sizeof(int);
            static const bool ktOn = [] {
                const char *e = std::getenv("TINYCODER_INKLING_MOE_KT");
                return e == nullptr || e[0] != '0';
            }();
            if (ktOn && ktSmemBytes <= g_smemOptin) {
                if (cudaFuncSetAttribute(expertMmaKtile<Type>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(ktSmemBytes)) == cudaSuccess) {
                    const dim3 ktGrid((rows + kKtNRow - 1) / kKtNRow, nExperts);
                    expertMmaKtile<Type>
                            <<<ktGrid, dim3(32, kKtWarps), ktSmemBytes, s>>>(
                                    static_cast<const char *>(w), aq, ad, out, rows,
                                    cols, capacity, counts, off);
                    if (cudaGetLastError() == cudaSuccess) return true;
                }
                cudaGetLastError();
            }
            // Occupancy rule.  Full-K residency costs shared memory, and on
            // this part (64 KB/SM) that is the resource that decides how many
            // blocks -- how many warps -- hide the latency of a kernel that is
            // nowhere near tensor bound.  The 16-token-tile kernel above needs
            // kNRow*rowStride + 6.4 KB, i.e. 53 KB for gate/up (1 block/SM, 8
            // warps) but only 27 KB for the Q2_0 down (2 blocks/SM, 16 warps).
            // Where the old kernel already has 16 warps it beats the stationary
            // one outright (measured: down 359.6 vs 363.9 ms per pass for the
            // two forms), so take the stationary form only where residency
            // costs occupancy.
            const size_t oldTwoBlocks = 2 * smemBytes;
            const size_t wsSmemBytes =
                    static_cast<size_t>(kWsMTile) * kAStride * sizeof(int) +
                    static_cast<size_t>(kbCount) * kWsNRow * kBStride * sizeof(int);
            const bool wsRawFits =
                    static_cast<size_t>(kWsNRow) * rowStride <=
                    static_cast<size_t>(kWsMTile) * kAStride * sizeof(int);
            // TINYCODER_INKLING_MOE_WS=0 forces the 16-token-tile kernel, the
            // A/B reference for the weight-stationary one.
            static const bool wsOn = [] {
                const char *e = std::getenv("TINYCODER_INKLING_MOE_WS");
                return e == nullptr || e[0] != '0';
            }();
            const bool wsWorthIt = wsOn && oldTwoBlocks > g_smemOptin;
            if (wsWorthIt && wsRawFits && wsSmemBytes <= g_smemOptin &&
                rows >= static_cast<uint32_t>(kWsNRow)) {
                if (cudaFuncSetAttribute(expertMmaStationary<Type>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(wsSmemBytes)) == cudaSuccess) {
                    const dim3 wsGrid((rows + kWsNRow - 1) / kWsNRow, nExperts);
                    expertMmaStationary<Type>
                            <<<wsGrid, dim3(32, kWsWarps), wsSmemBytes, s>>>(
                                    static_cast<const char *>(w), aq, ad, out, rows,
                                    cols, capacity, counts, off);
                    if (cudaGetLastError() == cudaSuccess) return true;
                }
                cudaGetLastError();
            }
            if (smemBytes > g_smemOptin) return false;
            if (cudaFuncSetAttribute(expertMmaKernel<Type>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(smemBytes)) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            const dim3 grid((rows + kNRow - 1) / kNRow, nExperts);
            expertMmaKernel<Type><<<grid, dim3(32, kWarps), smemBytes, s>>>(
                    static_cast<const char *>(w), aq, ad, out, rows, cols, capacity,
                    counts, off);
            return cudaGetLastError() == cudaSuccess;
        }

    }// namespace

    bool inklingExpertMmaSupported(uint32_t type) {
        // TINYCODER_INKLING_MOE_TYPES=old keeps IQ1_S (19) and NVFP4 (40) on
        // the float decoder, i.e. the pre-change fast path.  It is the A/B
        // reference for the two types served by expertMmaStationary now.
        static const bool legacyTypes = [] {
            const char *e = std::getenv("TINYCODER_INKLING_MOE_TYPES");
            return e != nullptr && std::strcmp(e, "old") == 0;
        }();
        if (legacyTypes && (type == 19u || type == 40u)) return false;
        return type == 16u || type == 17u || type == 19u || type == 22u ||
               type == 40u || type == 42u;
    }

    bool inklingExpertMmaGroupedOffsets(const void *w, const float *x, float *out,
                                        uint32_t rows, uint32_t cols,
                                        const int32_t *counts, const int32_t *off,
                                        uint32_t nSlots, uint32_t nExperts,
                                        uint32_t type, void *stream,
                                        bool quantize) {
        if (w == nullptr || x == nullptr || out == nullptr || counts == nullptr ||
            off == nullptr)
            return false;
        if (nSlots == 0 || rows == 0 || cols == 0 || nExperts == 0) return false;
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        switch (type) {
            case 16:
                return launchExpertMma<16>(w, x, out, rows, cols, nSlots, counts,
                                           off, nSlots, nExperts, s, quantize);
            case 17:
                return launchExpertMma<17>(w, x, out, rows, cols, nSlots, counts,
                                           off, nSlots, nExperts, s, quantize);
            case 19:
                return launchExpertMma<19>(w, x, out, rows, cols, nSlots, counts,
                                           off, nSlots, nExperts, s, quantize);
            case 22:
                return launchExpertMma<22>(w, x, out, rows, cols, nSlots, counts,
                                           off, nSlots, nExperts, s, quantize);
            case 40:
                return launchExpertMma<40>(w, x, out, rows, cols, nSlots, counts,
                                           off, nSlots, nExperts, s, quantize);
            case 42:
                return launchExpertMma<42>(w, x, out, rows, cols, nSlots, counts,
                                           off, nSlots, nExperts, s, quantize);
            default:
                return false;
        }
    }

    bool inklingExpertMmaGrouped(const void *w, const float *x, float *out,
                                 uint32_t rows, uint32_t cols, uint32_t capacity,
                                 const int32_t *counts, uint32_t nExperts,
                                 uint32_t type, void *stream, bool quantize) {
        if (w == nullptr || x == nullptr || out == nullptr || counts == nullptr)
            return false;
        if (capacity == 0 || capacity > 1024u || rows == 0 || cols == 0 ||
            nExperts == 0)
            return false;
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        const uint32_t nSlots = nExperts * capacity;
        switch (type) {
            case 16:
                return launchExpertMma<16>(w, x, out, rows, cols, capacity, counts,
                                           nullptr, nSlots, nExperts, s, quantize);
            case 17:
                return launchExpertMma<17>(w, x, out, rows, cols, capacity, counts,
                                           nullptr, nSlots, nExperts, s, quantize);
            case 19:
                return launchExpertMma<19>(w, x, out, rows, cols, capacity, counts,
                                           nullptr, nSlots, nExperts, s, quantize);
            case 22:
                return launchExpertMma<22>(w, x, out, rows, cols, capacity, counts,
                                           nullptr, nSlots, nExperts, s, quantize);
            case 40:
                return launchExpertMma<40>(w, x, out, rows, cols, capacity, counts,
                                           nullptr, nSlots, nExperts, s, quantize);
            case 42:
                return launchExpertMma<42>(w, x, out, rows, cols, capacity, counts,
                                           nullptr, nSlots, nExperts, s, quantize);
            default:
                return false;
        }
    }

}// namespace tinycoder::gpu

#endif// USE_CUDA
