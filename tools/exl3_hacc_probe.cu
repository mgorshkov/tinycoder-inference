/*
 * exl3_hacc_probe -- can an fp16-accumulate HMMA kernel with an exact
 * power-of-two pre-scale run the EXL3 prefill GEMM correctly and faster than
 * materialise+cuBLAS?
 *
 * Standalone decision experiment.  It does NOT include the engine's kernel
 * translation unit and does not link its device code: the decode arithmetic
 * and the mma.sync kernel are copied here, so the measurement can neither be
 * perturbed by a stale build of the production tree nor perturb it.  The only
 * engine code it calls is the CPU reference codec (EXL3Codec), used to prove
 * the GPU decode is right.
 *
 * One real projection from the real checkpoint:
 *   model.language_model.layers.0.mlp.gate_proj  (blk.0.ffn_gate, 5120 x 17408)
 *   trellis [320, 1088, 16] -> K = 1 bit, mul1 codebook, k = 5120, n = 17408.
 *
 * Measured, for seq = 512 tokens (the real pp512 shape):
 *   1. y_ref = A^T W in fp64 (cublasDgemm over the fp64-decoded RAW weights):
 *      the exact answer for the values the kernel is given.
 *   2. mma.sync.m16n8k8 with fp32 accumulate (known-good baseline): TFLOPS and
 *      error vs y_ref.
 *   3. The same tiling with fp16 accumulate plus a 2^-s pre-scale on the B
 *      operand, s = 0..12, compensated by 2^+s in the fp32 fold: error and
 *      TFLOPS for every s.  A second sweep applies a 2^12 gain to the operands
 *      first, reproducing the "RAW weights are O(4000)" regime the pre-scale
 *      was proposed for, so the mechanism can be confirmed where it was
 *      claimed to matter and shown unnecessary where it actually is.
 *   4. cuBLAS on the materialised fp16 weights -- the shipped path -- on the
 *      same buffers and with the same call shape as
 *      GPUComputeForward::gemmExl3Batch.
 *   5. A pure-MMA microbenchmark, so the same-session fp32-acc and fp16-acc
 *      peaks are measured instead of quoted.
 *
 * Build (explicit target; see unit_tests/CMakeLists.txt):
 *   cmake --build build-1.0 --target exl3_hacc_probe -j 20
 * Run:
 *   ./build-1.0/unit_tests/exl3_hacc_probe [checkpoint] [seq] [reps]
 * The checkpoint argument defaults to the reference 27B EXL3 directory.
 */
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "EXL3Codec.hpp"
#include "SafetensorsReader.hpp"

#define CK(expr)                                                                      \
    do {                                                                              \
        cudaError_t e_ = (expr);                                                      \
        if (e_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #expr,       \
                         cudaGetErrorString(e_));                                     \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

namespace {

    // =====================================================================
    // Device decode arithmetic -- copied from
    // src/cpp/core/cuda/exl3/EXL3Kernels.cu (EXL3Codec mirrors it on the CPU).
    // =====================================================================
    __device__ __forceinline__ float h2f(uint16_t h) {
        return __half2float(__ushort_as_half(h));
    }
    __device__ __forceinline__ float decode3INST(uint32_t x) {
        x = x * 89226354u + 64248484u;
        x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
        return h2f(static_cast<uint16_t>(x & 0xffff)) +
               h2f(static_cast<uint16_t>(x >> 16));
    }
    __device__ __forceinline__ float decodeMcG(uint32_t x) {
        x = x * 0xCBAC1FEDu;
        x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
        return h2f(static_cast<uint16_t>(x & 0xffff)) +
               h2f(static_cast<uint16_t>(x >> 16));
    }
    __device__ __forceinline__ float decodeMul1(uint32_t x) {
        x = x * 0x83DCD12Du;
        uint32_t sum = 0x6400u + (x & 0xff) + ((x >> 8) & 0xff) +
                       ((x >> 16) & 0xff) + ((x >> 24) & 0xff);
        const float kInv = h2f(0x1eee);
        const float kBias = h2f(0xc931);
        return h2f(static_cast<uint16_t>(sum)) * kInv + kBias;
    }
    __device__ __forceinline__ float decodeCode(uint32_t code, int codebook) {
        if (codebook == 1) return decodeMcG(code);
        if (codebook == 2) return decodeMul1(code);
        return decode3INST(code);
    }

    __device__ __forceinline__ uint32_t exl3NatFshift(uint32_t b, uint32_t a, int sh) {
        const uint64_t m = (static_cast<uint64_t>(a) << 32) | b;
        return static_cast<uint32_t>(m >> sh);
    }
    __device__ __forceinline__ uint32_t exl3NatBfe16(uint32_t x, int off) {
        return (x >> off) & 0xffffu;
    }

    // One lane's 8 decoded weights of a 16x16 tile (integer K).
    __device__ __forceinline__ void exl3LaneFragNative(
            const uint32_t *__restrict__ p, uint32_t K, int codebook, int lane,
            float *f) {
        const int to = lane * 8;
        uint32_t w[8];
        if (K == 1) {
            const uint32_t i1 = static_cast<uint32_t>(to) >> 5;
            const uint32_t i0 = (i1 + 7u) & 7u;
            const uint32_t b = exl3NatFshift(
                    p[i1], p[i0],
                    static_cast<int>(static_cast<uint32_t>(~to) & 24u));
            w[7] = b & 0xffffu;
            for (int i = 6; i >= 0; --i) w[i] = exl3NatBfe16(b, 7 - i);
        } else if (K == 2) {
            const uint32_t i1 = static_cast<uint32_t>(to) >> 4;
            const uint32_t i0 = (i1 + 15u) & 15u;
            const uint32_t b = exl3NatFshift(
                    p[i1], p[i0],
                    static_cast<int>((static_cast<uint32_t>(~to) & 8u) << 1));
            w[7] = b & 0xffffu;
            for (int i = 6; i >= 0; --i) w[i] = exl3NatBfe16(b, 14 - 2 * i);
        } else {
            const int b1 = (to + 257) * 3;
            const int b2 = b1 + 21;
            const int i0 = (b1 - 16) / 32;
            const int i2 = (b2 - 1) / 32;
            const int s2 = (i2 + 1) * 32 - b2;
            const int nw = 3 * 256 / 32;
            const uint32_t a = p[i0 % nw];
            const uint32_t b = p[i2 % nw];
            w[7] = exl3NatFshift(b, a, s2);
            w[6] = w[7] >> 3;
            w[5] = w[6] >> 3;
            w[4] = w[5] >> 3;
            w[3] = exl3NatFshift(b, a, s2 + 12);
            w[2] = w[3] >> 3;
            w[1] = w[2] >> 3;
            w[0] = w[1] >> 3;
        }
        for (int i = 0; i < 8; ++i) {
            f[i] = decodeCode(static_cast<uint16_t>(w[i] & 0xffffu), codebook);
        }
    }

    // =====================================================================
    // Trellis -> [k][n] weight matrix, same walk as the engine's decodeTile*.
    // =====================================================================
    __device__ __forceinline__ void writeTileF64(double *recon, uint32_t n, uint32_t ki,
                                                 uint32_t ni, int lane, const float *f,
                                                 const float *nn) {
        const int r0 = (lane % 4) * 2;
        const int c0 = lane / 8;
        const int c1 = c0 + 4;
        const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
        for (int i = 0; i < 4; ++i) {
            double *row = recon + static_cast<size_t>(ki * 16u + rows[i]) * n + ni * 16u;
            row[c0 * 2 + 0] = f[i];
            row[c0 * 2 + 1] = nn[i];
            row[c1 * 2 + 0] = f[4 + i];
            row[c1 * 2 + 1] = nn[4 + i];
        }
    }

    __global__ void decodeRawF64(const uint16_t *__restrict__ packed, uint32_t K,
                                 int codebook, uint32_t tilesN, uint32_t tileU16,
                                 double *__restrict__ recon, uint32_t n) {
        const uint32_t tile = blockIdx.x;
        const uint32_t ki = tile / tilesN;
        const uint32_t ni = tile % tilesN;
        const int lane = static_cast<int>(threadIdx.x);
        if (lane >= 32 || (lane & 4) != 0) return;
        const uint32_t *p = reinterpret_cast<const uint32_t *>(
                packed + static_cast<size_t>(tile) * tileU16);
        float f[8], nn[8];
        exl3LaneFragNative(p, K, codebook, lane, f);
        exl3LaneFragNative(p, K, codebook, lane + 4, nn);
        writeTileF64(recon, n, ki, ni, lane, f, nn);
    }

    // Production twin of decodeTilesIntNativeH<1> (the RAW materialise pass):
    // eight warps per block, one tile per warp, half2 stores.
    __global__ void decodeRawF16(const uint16_t *__restrict__ packed, uint32_t K,
                                 int codebook, uint32_t tilesN, uint32_t tileU16,
                                 __half *__restrict__ recon, uint32_t n,
                                 uint32_t nTiles) {
        const uint32_t warp = threadIdx.x >> 5;
        const uint32_t tile = blockIdx.x * (blockDim.x >> 5) + warp;
        if (tile >= nTiles) return;
        const uint32_t ki = tile / tilesN;
        const uint32_t ni = tile % tilesN;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        if ((lane & 4) != 0) return;
        const uint32_t *p = reinterpret_cast<const uint32_t *>(
                packed + static_cast<size_t>(tile) * tileU16);
        float f[8], nn[8];
        exl3LaneFragNative(p, K, codebook, lane, f);
        exl3LaneFragNative(p, K, codebook, lane + 4, nn);
        const int r0 = (lane % 4) * 2;
        const int c0 = lane / 8;
        const int c1 = c0 + 4;
        const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
        for (int i = 0; i < 4; ++i) {
            __half *row = recon + static_cast<size_t>(ki * 16u + rows[i]) * n +
                          ni * 16u;
            reinterpret_cast<__half2 *>(row)[c0] = __floats2half2_rn(f[i], nn[i]);
            reinterpret_cast<__half2 *>(row)[c1] =
                    __floats2half2_rn(f[4 + i], nn[4 + i]);
        }
    }

    // |w|max and sum(w^2) of the decoded matrix, for the magnitude statement.
    __global__ void weightStats(const double *__restrict__ w, size_t count,
                                unsigned int *amax, double *sumsq) {
        double s = 0.0;
        unsigned int m = 0u;
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
             i += (size_t) gridDim.x * blockDim.x) {
            const double v = w[i];
            s += v * v;
            const unsigned int b = __float_as_uint(static_cast<float>(std::fabs(v)));
            if (b > m) m = b;
        }
        atomicAdd(sumsq, s);
        atomicMax(amax, m);
    }

    // =====================================================================
    // mma.sync.m16n8k8 (HMMA.1688), fp32 and fp16 accumulate -- copied.
    // =====================================================================
    __device__ __forceinline__ void hmma1688(float *d, const uint32_t *a, uint32_t b) {
        asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                : "r"(a[0]), "r"(a[1]), "r"(b));
    }
    __device__ __forceinline__ void hmma1688H(uint32_t *d, const uint32_t *a, uint32_t b) {
        asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                : "+r"(d[0]), "+r"(d[1])
                : "r"(a[0]), "r"(a[1]), "r"(b));
    }
    __device__ __forceinline__ void ldmX4Trans(uint32_t *r, const void *p) {
        const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(p));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                     "{%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(sa));
    }

    // =====================================================================
    // The fused kernel: trellis decode feeding HMMA directly.  Exact-geometry
    // (FULL) path; copy of gemmFusedHmmaT<..., AREAD=0, FULL=true, HACC> with
    // three additions:
    //   S    -- power-of-two pre-scale exponent on the B operand (2^-s), undone
    //           by 2^+s in the fp32 fold.  S = 0 is no pre-scale.
    //   G    -- power-of-two gain applied before the pre-scale.  G = 12 puts
    //           the operands in the O(4000) regime the pre-scale was proposed
    //           for; G = 0 is this checkpoint's real magnitude.  Measurement
    //           knob only.
    //   MINB -- __launch_bounds__ blocks/SM (the register-budget selector).
    // =====================================================================
    template <int WARPS, int MLOOP, int NT, int KC, int MINB, bool HACC, int S, int G,
              bool NODEC = false>
    __launch_bounds__(WARPS * 32, MINB) __global__ void gemmProbe(
            const uint16_t *__restrict__ packed, uint32_t K, int codebook,
            uint32_t tilesN, uint32_t tileU16, const __half *__restrict__ A,
            float *__restrict__ C, uint32_t m, uint32_t k, uint32_t n) {
        static const int kNativeLane[16] = {0, 1, 2, 3, 8, 9, 10, 11,
                                            16, 17, 18, 19, 24, 25, 26, 27};
        constexpr int kTilesPerPass = WARPS * 2;
        constexpr int kDecodePasses = (NT * KC + kTilesPerPass - 1) / kTilesPerPass;
        constexpr float kPre = (S == G) ? 1.0f
                             : (S > G) ? (1.0f / static_cast<float>(1u << (S - G)))
                                       : static_cast<float>(1u << (G - S));
        constexpr float kUn = (S == G) ? 1.0f
                            : (S > G) ? static_cast<float>(1u << (S - G))
                                      : (1.0f / static_cast<float>(1u << (G - S)));
        __shared__ __align__(16) __half bT[2 * NT * KC * 256];
        const uint32_t warp = threadIdx.x >> 5;
        const uint32_t lane = threadIdx.x & 31u;
        const int vl = kNativeLane[lane & 15u];
        const uint32_t hw = lane >> 4;
        const uint32_t nt0 = blockIdx.x * NT;
        const uint32_t mBase = (blockIdx.y * WARPS * MLOOP) * 16u;
        const uint32_t kTiles = k / 16u;
        const int nChunks = static_cast<int>((kTiles + KC - 1) / KC);
        const uint32_t ar = lane >> 2;
        const uint32_t ac = (lane & 3u) * 2u;
        const __half *aStrip[MLOOP];
#pragma unroll
        for (int ml = 0; ml < MLOOP; ++ml) {
            const uint32_t s = mBase + (ml * WARPS + warp) * 16u;
            aStrip[ml] = A + static_cast<size_t>(s + ar) * k;
        }
        float acc[MLOOP][NT][2][4];
        uint32_t accH[MLOOP][NT][2][2];
#pragma unroll
        for (int ml = 0; ml < MLOOP; ++ml)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[ml][nt][h][q] = 0.0f;
#pragma unroll
                    for (int q = 0; q < 2; ++q) accH[ml][nt][h][q] = 0u;
                }

        auto decodeChunk = [&](int chunk, int buf) {
            if (NODEC) return;// control arm: MMA + ldmatrix with no trellis work
            const uint32_t kc = static_cast<uint32_t>(chunk) * KC;
#pragma unroll
            for (int p = 0; p < kDecodePasses; ++p) {
                const int t = p * kTilesPerPass + static_cast<int>(warp) * 2 +
                              static_cast<int>(hw);
                const int nt = t % NT;
                const int kk = t / NT;
                if (nt >= NT || kk >= KC) continue;// NT*KC < WARPS*2 leaves idle slots
                const uint32_t ni = nt0 + static_cast<uint32_t>(nt);
                const uint32_t kt = kc + static_cast<uint32_t>(kk);
                const uint32_t tile = kt * tilesN + ni;
                const uint32_t *pp = reinterpret_cast<const uint32_t *>(
                        packed + static_cast<size_t>(tile) * tileU16);
                float f[8], nn[8];
                exl3LaneFragNative(pp, K, codebook, vl, f);
                exl3LaneFragNative(pp, K, codebook, vl + 4, nn);
                const int r0 = (vl % 4) * 2, c0 = vl / 8, c1 = c0 + 4;
                const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
                __half *b = bT + (((buf * NT) + nt) * KC + kk) * 256;
                for (int i = 0; i < 4; ++i) {
                    __half *row = b + rows[i] * 16;
                    reinterpret_cast<__half2 *>(row)[c0] =
                            __floats2half2_rn(f[i] * kPre, nn[i] * kPre);
                    reinterpret_cast<__half2 *>(row)[c1] =
                            __floats2half2_rn(f[4 + i] * kPre, nn[4 + i] * kPre);
                }
            }
        };

        auto loadA = [&](uint32_t kbase, uint32_t (&af)[MLOOP][4]) {
#pragma unroll
            for (int ml = 0; ml < MLOOP; ++ml) {
                const __half *p = aStrip[ml] + kbase + ac;
                af[ml][0] = *reinterpret_cast<const uint32_t *>(p);
                af[ml][1] = *reinterpret_cast<const uint32_t *>(
                        p + static_cast<size_t>(8u) * k);
                af[ml][2] = *reinterpret_cast<const uint32_t *>(p + 8u);
                af[ml][3] = *reinterpret_cast<const uint32_t *>(
                        p + static_cast<size_t>(8u) * k + 8u);
            }
        };

        auto mmaChunk = [&](int chunk, int buf) {
            const uint32_t kc = static_cast<uint32_t>(chunk) * KC;
            uint32_t af[2][MLOOP][4];
            loadA(kc * 16u, af[0]);
#pragma unroll
            for (int kk = 0; kk < KC; ++kk) {
                const int cur = kk & 1;
                if (kk + 1 < KC)
                    loadA(kc * 16u + (uint32_t) (kk + 1) * 16u, af[cur ^ 1]);
                uint32_t bf[NT][4];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    ldmX4Trans(bf[nt],
                               bT + (((buf * NT) + nt) * KC + kk) * 256 +
                                       (lane & 15u) * 16u + (lane >> 4) * 8u);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                    for (int ml = 0; ml < MLOOP; ++ml) {
                        const uint32_t *ap = af[cur][ml];
                        if (HACC) {
                            hmma1688H(accH[ml][nt][0], ap + 0, bf[nt][0]);
                            hmma1688H(accH[ml][nt][0], ap + 2, bf[nt][1]);
                            hmma1688H(accH[ml][nt][1], ap + 0, bf[nt][2]);
                            hmma1688H(accH[ml][nt][1], ap + 2, bf[nt][3]);
                        } else {
                            hmma1688(acc[ml][nt][0], ap + 0, bf[nt][0]);
                            hmma1688(acc[ml][nt][0], ap + 2, bf[nt][1]);
                            hmma1688(acc[ml][nt][1], ap + 0, bf[nt][2]);
                            hmma1688(acc[ml][nt][1], ap + 2, bf[nt][3]);
                        }
                    }
                }
            }
        };

        auto foldAcc = [&]() {
#pragma unroll
            for (int ml = 0; ml < MLOOP; ++ml)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 g0 = __half22float2(
                                *reinterpret_cast<const __half2 *>(&accH[ml][nt][h][0]));
                        const float2 g1 = __half22float2(
                                *reinterpret_cast<const __half2 *>(&accH[ml][nt][h][1]));
                        acc[ml][nt][h][0] += g0.x * kUn;
                        acc[ml][nt][h][1] += g0.y * kUn;
                        acc[ml][nt][h][2] += g1.x * kUn;
                        acc[ml][nt][h][3] += g1.y * kUn;
                        accH[ml][nt][h][0] = 0u;
                        accH[ml][nt][h][1] = 0u;
                    }
        };

        if (NODEC) {
            for (int i = static_cast<int>(threadIdx.x); i < 2 * NT * KC * 256;
                 i += static_cast<int>(blockDim.x))
                bT[i] = __ushort_as_half(0);
            __syncthreads();
        }
        decodeChunk(0, 0);
        __syncthreads();
        for (int c = 0; c < nChunks; ++c) {
            const int b = c & 1, nb = b ^ 1;
            if ((c + 1) < nChunks) decodeChunk(c + 1, nb);
            mmaChunk(c, b);
            if (HACC) foldAcc();
            __syncthreads();
        }

#pragma unroll
        for (int ml = 0; ml < MLOOP; ++ml) {
            const uint32_t strip = mBase + (ml * WARPS + warp) * 16u;
            const uint32_t r0 = lane >> 2, r1 = r0 + 8u;
            const uint32_t cOff = (lane & 3u) * 2u;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint32_t ni = nt0 + static_cast<uint32_t>(nt);
                float *dst = C + static_cast<size_t>(strip + r0) * n + ni * 16u + cOff;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t co = static_cast<uint32_t>(h) * 8u;
                    *reinterpret_cast<float2 *>(dst + co) =
                            make_float2(acc[ml][nt][h][0], acc[ml][nt][h][1]);
                    *reinterpret_cast<float2 *>(dst + co +
                                                static_cast<size_t>(8u) * n) =
                            make_float2(acc[ml][nt][h][2], acc[ml][nt][h][3]);
                }
            }
        }
    }

    // =====================================================================
    // Pure-MMA microbenchmark: the same-session peak, not the published one.
    // =====================================================================
    template <bool HACC, int ACCS>
    __global__ void mmaPeak(uint32_t *out, int iters) {
        float d[ACCS][4];
        uint32_t dh[ACCS][2];
        uint32_t a[4], b[4];
#pragma unroll
        for (int i = 0; i < ACCS; ++i) {
#pragma unroll
            for (int q = 0; q < 4; ++q) d[i][q] = 0.0f;
            dh[i][0] = 0u;
            dh[i][1] = 0u;
        }
        for (int i = 0; i < 4; ++i) {
            a[i] = 0x3c003c00u;
            b[i] = 0x3c003c00u;
        }
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int i = 0; i < ACCS; ++i) {
                if (HACC) {
                    hmma1688H(dh[i], &a[0], b[0]);
                    hmma1688H(dh[i], &a[2], b[1]);
                    hmma1688H(dh[i], &a[0], b[2]);
                    hmma1688H(dh[i], &a[2], b[3]);
                } else {
                    hmma1688(d[i], &a[0], b[0]);
                    hmma1688(d[i], &a[2], b[1]);
                    hmma1688(d[i], &a[0], b[2]);
                    hmma1688(d[i], &a[2], b[3]);
                }
            }
        }
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < ACCS; ++i) {
            if (HACC)
                s += __uint_as_float(dh[i][0] + dh[i][1]);
            else
                s += d[i][0] + d[i][1] + d[i][2] + d[i][3];
        }
        if (s == 1234.5678f) out[0] = 1u;// never true: keeps the chain live
    }

    // =====================================================================
    // Host helpers
    // =====================================================================
    float clockGHz() {
        FILE *f = popen(
                "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null",
                "r");
        if (!f) return 0.0f;
        float mhz = 0.0f;
        if (std::fscanf(f, "%f", &mhz) != 1) mhz = 0.0f;
        pclose(f);
        return mhz / 1000.0f;
    }

    template <typename Fn>
    float timeMs(int iters, const Fn &fn) {
        fn();
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) fn();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, a, b));
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return ms / static_cast<float>(iters);
    }

    // Deterministic N(0,1): splitmix64 + Box-Muller.
    struct Rng {
        uint64_t s;
        explicit Rng(uint64_t seed) : s(seed) {}
        uint64_t next() {
            s += 0x9e3779b97f4a7c15ull;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
            return z ^ (z >> 31);
        }
        double uniform() {
            return (static_cast<double>(next() >> 11) + 0.5) / 9007199254740992.0;
        }
        double normal() {
            const double u1 = uniform(), u2 = uniform();
            return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        }
    };

    struct Metrics {
        double maxAbs = 0, rmsErr = 0, rmsRef = 0, maxAbsRef = 0;
        double relMax() const { return maxAbsRef > 0 ? maxAbs / maxAbsRef : 0; }
        double relRms() const { return rmsRef > 0 ? rmsErr / rmsRef : 0; }
    };

    void metrics(const float *hat, const double *ref, size_t count, Metrics &m) {
        double se = 0, sr = 0, ma = 0, mr = 0;
        for (size_t i = 0; i < count; ++i) {
            const double d = static_cast<double>(hat[i]) - ref[i];
            const double a = std::fabs(d);
            if (a > ma) ma = a;
            const double ar = std::fabs(ref[i]);
            if (ar > mr) mr = ar;
            se += d * d;
            sr += ref[i] * ref[i];
        }
        m.maxAbs = ma;
        m.rmsErr = std::sqrt(se / static_cast<double>(count));
        m.rmsRef = std::sqrt(sr / static_cast<double>(count));
        m.maxAbsRef = mr;
    }

    void metricsD(const double *hat, const double *ref, size_t count, Metrics &m) {
        double se = 0, sr = 0, ma = 0, mr = 0;
        for (size_t i = 0; i < count; ++i) {
            const double d = hat[i] - ref[i];
            const double a = std::fabs(d);
            if (a > ma) ma = a;
            const double ar = std::fabs(ref[i]);
            if (ar > mr) mr = ar;
            se += d * d;
            sr += ref[i] * ref[i];
        }
        m.maxAbs = ma;
        m.rmsErr = std::sqrt(se / static_cast<double>(count));
        m.rmsRef = std::sqrt(sr / static_cast<double>(count));
        m.maxAbsRef = mr;
    }

    // Normalized 128-point Hadamard (1/sqrt(128)), in place.
    void fwht128(double *v) {
        for (int len = 1; len < 128; len <<= 1) {
            for (int i = 0; i < 128; i += len << 1) {
                for (int j = 0; j < len; ++j) {
                    const double a = v[i + j], b = v[i + j + len];
                    v[i + j] = a + b;
                    v[i + j + len] = a - b;
                }
            }
        }
        const double k = 1.0 / std::sqrt(128.0);
        for (int i = 0; i < 128; ++i) v[i] *= k;
    }

    // The engine's output transform: y -> svh * H128(y) over 128-column bands.
    void foldOut(double *y, size_t rows, uint32_t n, const std::vector<float> &svh) {
        for (size_t r = 0; r < rows; ++r) {
            double *row = y + r * n;
            for (uint32_t b = 0; b + 128 <= n; b += 128) {
                fwht128(row + b);
                for (uint32_t t = 0; t < 128; ++t) row[b + t] *= svh[b + t];
            }
        }
    }

} // namespace

// =========================================================================
// main
// =========================================================================
int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1]
                               : "/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw";
    const int seq = argc > 2 ? std::atoi(argv[2]) : 512;
    const int repsArg = argc > 3 ? std::atoi(argv[3]) : 0;

    // ---- open the checkpoint, read the one projection -------------------
    std::vector<std::string> shards;
    for (const auto &e : std::filesystem::directory_iterator(dir)) {
        const std::string p = e.path().string();
        if (p.size() > 12 && p.compare(p.size() - 12, 12, ".safetensors") == 0)
            shards.push_back(p);
    }
    std::sort(shards.begin(), shards.end());
    if (shards.empty()) {
        std::fprintf(stderr, "no safetensors in %s\n", dir);
        return 1;
    }
    tinycoder::SafetensorsReader st;
    if (!st.open(shards)) {
        std::fprintf(stderr, "safetensors open failed\n");
        return 1;
    }
    const std::string base = "model.language_model.layers.0.mlp.gate_proj";
    const auto *tr = st.find(base + ".trellis");
    const auto *su = st.find(base + ".suh");
    const auto *sv = st.find(base + ".svh");
    if (!tr || !su || !sv || tr->shape.size() != 3) {
        std::fprintf(stderr, "projection tensors missing\n");
        return 1;
    }
    const uint32_t k = static_cast<uint32_t>(tr->shape[0] * 16);
    const uint32_t n = static_cast<uint32_t>(tr->shape[1] * 16);
    const uint32_t tileU16 = static_cast<uint32_t>(tr->shape[2]);
    const uint32_t K = tileU16 / 16;
    int codebook = 2;// mul1
    if (st.find(base + ".mcg")) codebook = 1;
    else if (!st.find(base + ".mul1")) codebook = 0;
    if (k % 64 != 0 || n % 128 != 0) {
        std::fprintf(stderr, "unsupported geometry k=%u n=%u\n", k, n);
        return 1;
    }
    const uint32_t tilesN = n / 16u;
    const uint32_t nTiles = (k / 16u) * tilesN;
    const size_t trBytes = static_cast<size_t>(tr->end - tr->begin);
    std::printf("projection %s\n  k=%u n=%u K=%u tileU16=%u codebook=%s  seq=%d\n",
                base.c_str(), k, n, K, tileU16,
                codebook == 2 ? "mul1" : codebook == 1 ? "mcg" : "3inst", seq);
    std::printf("  trellis %.1f MB, materialised fp16 %.1f MB, fp64 %.1f MB\n",
                (double) trBytes / 1e6, (double) k * n * 2 / 1e6,
                (double) k * n * 8 / 1e6);

    auto toFloat = [](const tinycoder::SafetensorsReader::TensorDesc &d,
                      const uint8_t *raw) {
        std::vector<float> v((size_t) tinycoder::SafetensorsReader::numel(d));
        const auto *h = reinterpret_cast<const uint16_t *>(raw);
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = tinycoder::EXL3Codec::halfToFloat(h[i]);
        return v;
    };
    std::vector<float> suh = toFloat(*su, st.data(*su));
    std::vector<float> svh = toFloat(*sv, st.data(*sv));
    if (suh.size() != k || svh.size() != n) {
        std::fprintf(stderr, "sign vector length mismatch\n");
        return 1;
    }

    // ---- device buffers -------------------------------------------------
    uint16_t *dT = nullptr;
    float *dSuh = nullptr, *dSvh = nullptr;
    double *dW64 = nullptr;
    __half *dW16 = nullptr;
    __half *dA = nullptr;
    float *dC = nullptr, *dCc = nullptr;
    double *dXd = nullptr, *dWd = nullptr, *dCd = nullptr;
    unsigned int *dAmax = nullptr;
    double *dSumsq = nullptr;
    uint32_t *dMmaOut = nullptr;
    const size_t kn = (size_t) k * n;
    CK(cudaMalloc(&dT, trBytes));
    CK(cudaMalloc(&dSuh, k * sizeof(float)));
    CK(cudaMalloc(&dSvh, n * sizeof(float)));
    CK(cudaMalloc(&dW64, kn * sizeof(double)));
    CK(cudaMalloc(&dW16, kn * sizeof(__half)));
    CK(cudaMalloc(&dA, (size_t) seq * k * sizeof(__half)));
    CK(cudaMalloc(&dC, (size_t) seq * n * sizeof(float)));
    CK(cudaMalloc(&dCc, (size_t) seq * n * sizeof(float)));
    CK(cudaMalloc(&dXd, (size_t) seq * k * sizeof(double)));
    CK(cudaMalloc(&dWd, kn * sizeof(double)));
    CK(cudaMalloc(&dCd, (size_t) seq * n * sizeof(double)));
    CK(cudaMalloc(&dAmax, sizeof(unsigned int)));
    CK(cudaMalloc(&dSumsq, sizeof(double)));
    CK(cudaMalloc(&dMmaOut, sizeof(uint32_t)));
    CK(cudaMemcpy(dT, st.data(*tr), trBytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSuh, suh.data(), k * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSvh, svh.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // ---- activation -----------------------------------------------------
    // A is exactly what the engine hands the GEMM: kExl3FwhtToF16 writes
    // fp16(H(x) * suh).  suh is not a sign vector in this checkpoint but a
    // per-input scale (|suh| ~ 0.0088), so A = suh (.) x with x the
    // post-RMSNorm activation.  x is a deterministic Gaussian whose RMS is the
    // measured RMS of blk.0's post_attention_layernorm.weight -- an RMSNorm
    // whose input is whitened produces an output with exactly that RMS, and
    // the Hadamard of such a vector has the same per-element RMS.
    const float xRms = 1.5902f;
    std::vector<float> x((size_t) seq * k);
    {
        Rng rng(0x5eed1234ull);
        for (auto &v : x) v = static_cast<float>(rng.normal() * xRms);
    }
    std::vector<__half> aH((size_t) seq * k);
    double sumsqA = 0.0, mxA = 0.0, sumsqSuh = 0.0;
    for (float v : suh) sumsqSuh += (double) v * v;
    for (size_t i = 0; i < aH.size(); ++i) {
        const float v = suh[i % k] * x[i];
        aH[i] = __float2half_rn(v);
        const double d = (double) __half2float(aH[i]);
        sumsqA += d * d;
        if (std::fabs(d) > mxA) mxA = std::fabs(d);
    }
    CK(cudaMemcpy(dA, aH.data(), aH.size() * sizeof(__half), cudaMemcpyHostToDevice));
    const double aRms = std::sqrt(sumsqA / (double) aH.size());
    std::printf("A (the GEMM's fp16 input): RMS=%.5f max=%.4f | |suh| RMS=%.5f, "
                "x RMS=%.4f\n",
                aRms, mxA, std::sqrt(sumsqSuh / k), (double) xRms);

    // ---- decode the reference weight (fp64) -----------------------------
    const float msDecode64 = timeMs(3, [&] {
        decodeRawF64<<<nTiles, 32>>>(dT, K, codebook, tilesN, tileU16, dW64, n);
    });
    CK(cudaGetLastError());
    std::printf("decode trellis -> fp64: %.3f ms (%.1f GB/s counting the fp64 write + "
                "trellis read)\n",
                msDecode64, (double) (trBytes + kn * 8) / msDecode64 / 1e6);

    CK(cudaMemset(dAmax, 0, sizeof(unsigned int)));
    CK(cudaMemset(dSumsq, 0, sizeof(double)));
    weightStats<<<512, 256>>>(dW64, kn, dAmax, dSumsq);
    CK(cudaGetLastError());
    unsigned int amax = 0;
    double sumsq = 0;
    CK(cudaMemcpy(&amax, dAmax, sizeof(amax), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&sumsq, dSumsq, sizeof(sumsq), cudaMemcpyDeviceToHost));
    float wmax = 0.0f;
    std::memcpy(&wmax, &amax, sizeof(float));

    // ---- independent decode check against the engine's CPU codec --------
    {
        std::vector<float> cpuTile(256);
        std::vector<double> row(16), gpuTile(256);
        int bad = 0;
        double maxDiff = 0;
        const int tilesToCheck = 48;
        for (int t = 0; t < tilesToCheck; ++t) {
            const uint32_t tile = (uint32_t) ((t * 7919u) % nTiles);
            const uint32_t ki = tile / tilesN, ni = tile % tilesN;
            tinycoder::EXL3Codec::decodeTileNative(
                    reinterpret_cast<const uint16_t *>(
                            st.data(*tr) + (size_t) tile * tileU16 * sizeof(uint16_t)),
                    K, codebook, cpuTile.data());
            for (int r = 0; r < 16; ++r) {
                CK(cudaMemcpy(row.data(),
                              dW64 + (size_t) (ki * 16 + r) * n + ni * 16,
                              row.size() * sizeof(double), cudaMemcpyDeviceToHost));
                for (int c = 0; c < 16; ++c) gpuTile[r * 16 + c] = row[c];
            }
            for (int i = 0; i < 256; ++i) {
                const double d = std::fabs(gpuTile[i] - (double) cpuTile[i]);
                if (d > maxDiff) maxDiff = d;
                if (d > 1e-9) ++bad;
            }
        }
        std::printf("decode check vs EXL3Codec::decodeTileNative: %d tiles, "
                    "maxAbsDiff=%.3g (%s)\n",
                    tilesToCheck, maxDiff, bad == 0 ? "identical" : "MISMATCH");
        if (bad != 0) return 1;
    }
    std::printf("decoded RAW weights: max |w| = %.4f, RMS = %.4f  (the operands the "
                "GEMM sees)\n",
                wmax, std::sqrt(sumsq / (double) kn));

    // ---- y_ref = A^T W in fp64 ------------------------------------------
    {
        std::vector<double> xd(aH.size());
        for (size_t i = 0; i < aH.size(); ++i) xd[i] = __half2float(aH[i]);
        CK(cudaMemcpy(dXd, xd.data(), xd.size() * sizeof(double),
                      cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(dWd, dW64, kn * sizeof(double), cudaMemcpyDeviceToDevice));

    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 1;
    {
        const double one = 1.0, zero = 0.0;
        // C' = W^T x: A = W ([n][k] col-major, lda = n), B = x ([k][seq],
        // ldb = k), C' = [n][seq] col-major, ldc = n -- the engine's layout.
        const cublasStatus_t stt = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int) n,
                                               seq, (int) k, &one, dWd, (int) n, dXd,
                                               (int) k, &zero, dCd, (int) n);
        if (stt != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "dgemm failed %d\n", (int) stt);
            return 1;
        }
        CK(cudaDeviceSynchronize());
    }
    std::vector<double> yRef((size_t) seq * n);
    {
        std::vector<double> ct((size_t) seq * n);
        CK(cudaMemcpy(ct.data(), dCd, ct.size() * sizeof(double),
                      cudaMemcpyDeviceToHost));
        for (int r = 0; r < seq; ++r)
            for (uint32_t c = 0; c < n; ++c)
                // dCd is column-major [n][seq]: element (row c, col r) is at
                // c + r*n.
                yRef[(size_t) r * n + c] = ct[c + (size_t) r * n];
    }
    {
        double s = 0, mx = 0;
        for (double v : yRef) {
            s += v * v;
            if (std::fabs(v) > mx) mx = std::fabs(v);
        }
        std::printf("y_ref (fp64): RMS=%.5f max=%.5f  -> the GEMM answer is O(%.0e)\n",
                    std::sqrt(s / yRef.size()), mx,
                    std::pow(10.0, std::round(std::log10(mx))));
    }
    {
        // Independent spot-check of the fp64 reference: gather W[:,0:8] and
        // recompute four rows x eight columns with a long-double dot product.
        std::vector<double> w8((size_t) k * 8);
        CK(cudaMemcpy2D(w8.data(), 8 * sizeof(double), dW64, n * sizeof(double),
                        8 * sizeof(double), k, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 8; ++c) {
                long double s = 0;
                for (uint32_t p = 0; p < k; ++p)
                    s += (long double) __half2float(aH[(size_t) r * k + p]) *
                         (long double) w8[(size_t) p * 8 + c];
                const double d = std::fabs(
                        (double) (s - (long double) yRef[(size_t) r * n + c]));
                if (d > worst) worst = d;
            }
        std::printf("y_ref spot-check vs CPU long-double dot product (4 rows x 8 cols): "
                    "maxAbsDiff=%.3g\n",
                    worst);
        if (worst > 1e-9) {
            std::fprintf(stderr, "reference layout mismatch\n");
            return 1;
        }
    }
    std::vector<double> yRefF = yRef;
    foldOut(yRefF.data(), (size_t) seq, n, svh);
    {
        double s = 0, mx = 0;
        for (double v : yRefF) {
            s += v * v;
            if (std::fabs(v) > mx) mx = std::fabs(v);
        }
        std::printf("post-fold reference (what layer 0 consumes): RMS=%.5f max=%.5f\n",
                    std::sqrt(s / yRefF.size()), mx);
    }

    // ---- pure-MMA peak --------------------------------------------------
    {
        const int blocks = 2 * 108;
        const int inner = 50, outer = 3;
        // 4 mma per accumulator per iteration x 8 accumulators = 32 mma per
        // warp-iteration.
        const double per = (double) blocks * 8.0 /*warps*/ * 32.0 /*mma per iter*/ *
                           2048.0 /*flop per m16n8k8*/ * (double) inner;
        auto peak32 = [&] { mmaPeak<false, 8><<<blocks, 256>>>(dMmaOut, inner); };
        auto peak16 = [&] { mmaPeak<true, 8><<<blocks, 256>>>(dMmaOut, inner); };
        const float ms32 = timeMs(outer, peak32);
        const float ms16 = timeMs(outer, peak16);
        CK(cudaGetLastError());
        std::printf("\npure MMA peak (2 blk/SM, 8 warps, 8 indep acc/warp): "
                    "fp32-acc %.2f TFLOPS, fp16-acc %.2f TFLOPS @ %.2f GHz\n",
                    per / (ms32 * 1e-3) / 1e12, per / (ms16 * 1e-3) / 1e12, clockGHz());
    }

    // ---- fused kernel runs ----------------------------------------------
    // Timings are taken round-robin over three rounds and the minimum per
    // configuration is reported.  The 2080 Ti here is power-limited: a burst of
    // mixed INT-decode + HMMA work lets the SM clock fall from 1.97 GHz to
    // ~1.35 GHz within a second, so a sequential table reports the run order,
    // not the kernel.  Round-robin + min gives every configuration the same
    // chance to be measured at the clock the fastest ones saw.
    const double flops = 2.0 * (double) k * n * (double) seq;
    const int reps = repsArg > 0 ? repsArg : std::max(5, (int) (6.0e10 / flops));
    std::vector<float> yHat((size_t) seq * n);
    std::vector<double> yFold((size_t) seq * n);

    struct Run {
        std::string name;
        std::function<void()> launch;
        float *out = nullptr;
        bool check = true;
        Metrics mr, mf;
        int regs = -1;
        size_t spill = 0;
        float ms = 1e9f;
    };
    std::vector<Run> runs;

    auto addGemm = [&](const char *name, auto kern, int warps, uint32_t rowsPerBlock,
                       uint32_t colsPerBlock, bool check) {
        Run r;
        r.name = name;
        r.check = check;
        r.out = dC;
        const dim3 grid(n / colsPerBlock, seq / rowsPerBlock);
        r.launch = [&, kern, grid, warps] {
            kern<<<grid, warps * 32>>>(dT, K, codebook, tilesN, tileU16, dA, dC, seq,
                                       k, n);
        };
        cudaFuncAttributes fa{};
        cudaFuncGetAttributes(&fa, kern);
        r.regs = fa.numRegs;
        r.spill = fa.localSizeBytes;
        runs.push_back(std::move(r));
    };
    auto addPlain = [&](const char *name, std::function<void()> fn, float *out,
                        bool check) {
        Run r;
        r.name = name;
        r.launch = std::move(fn);
        r.out = out;
        r.check = check;
        runs.push_back(std::move(r));
    };

    // fp32-accumulate baseline, and the fp16-accumulate sweep at one tiling.
    addGemm("fp32acc 8x2x4x4", gemmProbe<8, 2, 4, 4, 2, false, 0, 0>, 8, 256, 64, true);
    addGemm("fp32acc 8x2x2x4", gemmProbe<8, 2, 2, 4, 2, false, 0, 0>, 8, 256, 32, true);
    addGemm("hacc s=0  8x2x2x4", gemmProbe<8, 2, 2, 4, 2, true, 0, 0>, 8, 256, 32, true);
    addGemm("hacc s=1  8x2x2x4", gemmProbe<8, 2, 2, 4, 2, true, 1, 0>, 8, 256, 32, true);
    addGemm("hacc s=2  8x2x2x4", gemmProbe<8, 2, 2, 4, 2, true, 2, 0>, 8, 256, 32, true);
    addGemm("hacc s=4  8x2x2x4", gemmProbe<8, 2, 2, 4, 2, true, 4, 0>, 8, 256, 32, true);
    addGemm("hacc s=6  8x2x2x4", gemmProbe<8, 2, 2, 4, 2, true, 6, 0>, 8, 256, 32, true);
    addGemm("hacc s=8  8x2x2x4", gemmProbe<8, 2, 2, 4, 2, true, 8, 0>, 8, 256, 32, true);
    addGemm("hacc s=10 8x2x2x4", gemmProbe<8, 2, 2, 4, 2, true, 10, 0>, 8, 256, 32, true);
    addGemm("hacc s=12 8x2x2x4", gemmProbe<8, 2, 2, 4, 2, true, 12, 0>, 8, 256, 32, true);
    // Wider accumulator tile: 96 accumulator registers, so only one block/SM.
    addGemm("hacc 8x2x4x4 blk1", gemmProbe<8, 2, 4, 4, 1, true, 0, 0>, 8, 256, 64, true);
    // Same tile with KC = 8: half as many fp32 folds.  If the fold is what eats
    // the 2x fp16 rate, this closes part of the gap.
    addGemm("hacc KC8 8x2x4x4 blk1", gemmProbe<8, 2, 4, 8, 1, true, 0, 0>, 8, 256, 64,
            true);
    // The O(4000) regime the pre-scale was proposed for, for comparison.
    addGemm("hacc G12 s=0", gemmProbe<8, 2, 2, 4, 2, true, 0, 12>, 8, 256, 32, true);
    addGemm("hacc G12 s=12", gemmProbe<8, 2, 2, 4, 2, true, 12, 12>, 8, 256, 32, true);
    // Decode-free control arm: its output is wrong by construction (the B tiles
    // are zero), so only its timing is meaningful.
    addGemm("NODEC fp32acc 8x2x4x4",
            gemmProbe<8, 2, 4, 4, 2, false, 0, 0, true>, 8, 256, 64, false);
    addGemm("NODEC fp32acc 8x2x2x4",
            gemmProbe<8, 2, 2, 4, 2, false, 0, 0, true>, 8, 256, 32, false);
    addGemm("NODEC hacc 8x2x2x4", gemmProbe<8, 2, 2, 4, 2, true, 0, 0, true>, 8, 256,
            32, false);
    addGemm("NODEC hacc 8x2x4x4 blk1",
            gemmProbe<8, 2, 4, 4, 1, true, 0, 0, true>, 8, 256, 64, false);

    // The shipped path: materialise the whole [k][n] matrix to fp16, then the
    // exact cublasGemmEx call GPUComputeForward::gemmExl3Batch issues.
    addPlain("materialise -> fp16",
             [&] {
                 decodeRawF16<<<(nTiles + 7u) / 8u, 256>>>(dT, K, codebook, tilesN,
                                                           tileU16, dW16, n, nTiles);
             },
             nullptr, false);
    addPlain("cublas fp32-acc",
             [&] {
                 // alpha/beta must live in the lambda: the Run is used after the
                 // enclosing scope ends, and a captured reference would dangle
                 // (which is exactly how a "62 TFLOPS" cublas entry appeared).
                 const float alpha = 1.0f, beta = 0.0f;
                 cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, (int) n, seq, (int) k,
                              &alpha, dW16, CUDA_R_16F, (int) n, dA, CUDA_R_16F,
                              (int) k, &beta, dCc, CUDA_R_32F, (int) n, CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
             },
             dCc, true);

    // errors (one launch + copy each)
    for (auto &r : runs) {
        if (!r.check || r.out == nullptr) continue;
        r.launch();
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(yHat.data(), r.out, yHat.size() * sizeof(float),
                      cudaMemcpyDeviceToHost));
        metrics(yHat.data(), yRef.data(), yHat.size(), r.mr);
        for (size_t i = 0; i < yFold.size(); ++i) yFold[i] = yHat[i];
        foldOut(yFold.data(), (size_t) seq, n, svh);
        metricsD(yFold.data(), yRefF.data(), yFold.size(), r.mf);
    }

    // timing (three interleaved rounds, keep the fastest)
    for (int round = 0; round < 3; ++round)
        for (auto &r : runs) r.ms = std::min(r.ms, timeMs(reps, r.launch));

    std::printf("\n-- one projection, k=%u n=%u seq=%d: errors vs fp64, ms, TFLOPS --\n",
                k, n, seq);
    std::printf("%-22s %9s %9s %9s | %9s %9s | %7s %7s  %s\n", "config", "maxAbs",
                "relMax", "relRms", "foldRelMx", "foldRelRms", "ms", "TFLOPS",
                "regs/spill");
    for (const auto &r : runs) {
        if (r.out == nullptr) {
            std::printf("%-22s %9s %9s %9s | %9s %9s | %7.3f %7s  %s\n", r.name.c_str(),
                        "-", "-", "-", "-", "-", r.ms, "-",
                        r.name.rfind("materialise", 0) == 0 ? "(write pass)" : "");
            continue;
        }
        const double tf = flops / (r.ms * 1e-3) / 1e12;
        std::printf("%-22s %9.4g %9.2e %9.2e | %9.2e %9.2e | %7.3f %7.2f  ", r.name.c_str(),
                    r.mr.maxAbs, r.mr.relMax(), r.mr.relRms(), r.mf.relMax(),
                    r.mf.relRms(), r.ms, tf);
        if (r.regs > 0)
            std::printf("%d r / %zu B", r.regs, r.spill);
        else
            std::printf("%s", "n/a (cublas)");
        std::printf("\n");
    }

    // ---- identity probe: one-hot A gives one decoded weight per output ---
    // This is the check that previously produced "maxAbsDiff 3.889".  A single
    // nonzero input means the only possible difference from the exact value is
    // the fp16 storage of that one operand, which BOTH accumulate types share.
    {
        std::vector<__half> oneHot((size_t) seq * k, __float2half_rn(0.0f));
        oneHot[0] = __float2half_rn(1.0f);
        CK(cudaMemcpy(dA, oneHot.data(), oneHot.size() * sizeof(__half),
                      cudaMemcpyHostToDevice));
        gemmProbe<8, 2, 2, 4, 2, false, 0, 0>
                <<<dim3(n / 32u, seq / 256u), 256>>>(dT, K, codebook, tilesN, tileU16,
                                                     dA, dC, seq, k, n);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<float> y0(n);
        CK(cudaMemcpy(y0.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost));
        gemmProbe<8, 2, 2, 4, 2, true, 0, 0>
                <<<dim3(n / 32u, seq / 256u), 256>>>(dT, K, codebook, tilesN, tileU16,
                                                     dA, dCc, seq, k, n);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<float> y1(n);
        CK(cudaMemcpy(y1.data(), dCc, n * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<double> w0(n);
        for (uint32_t c = 0; c < n; ++c)
            CK(cudaMemcpy(&w0[c], dW64 + c, sizeof(double), cudaMemcpyDeviceToHost));
        double dExact = 0, dAcc = 0;
        for (uint32_t c = 0; c < n; ++c) {
            dExact = std::max(dExact, std::fabs((double) y0[c] - w0[c]));
            dAcc = std::max(dAcc, std::fabs((double) y0[c] - y1[c]));
        }
        std::printf("\nidentity probe (one-hot A, row 0 = W[0][:]): fp32acc vs exact "
                    "decoded weight maxAbsDiff=%.3g (fp16 ulp of a %.2f value is %.3g);"
                    " hacc s=0 vs fp32acc maxAbsDiff=%.3g\n",
                    dExact, wmax,
                    std::ldexp(1.0f, (int) std::floor(std::log2(wmax)) - 10), dAcc);
        CK(cudaMemcpy(dA, aH.data(), aH.size() * sizeof(__half),
                      cudaMemcpyHostToDevice));
    }

    // ---- the end-to-end number for this projection ----------------------
    {
        auto msOf = [&](const char *name) {
            for (const auto &r : runs)
                if (r.name == name) return r.ms;
            return 0.0f;
        };
        const float mat = msOf("materialise -> fp16");
        const float cublas = msOf("cublas fp32-acc");
        const float f32 = msOf("fp32acc 8x2x4x4");
        const float hacc = msOf("hacc 8x2x4x4 blk1");
        std::printf("\n-- this projection, end to end (decode included in the fused "
                    "kernels) --\n");
        std::printf("  materialise + cuBLAS : %.3f + %.3f = %.3f ms  (writes+reads "
                    "%.0f MB of fp16 weights)\n",
                    mat, cublas, mat + cublas, (double) kn * 2 / 1e6);
        std::printf("  fused fp32-acc       : %.3f ms  (%.2fx vs shipped)\n", f32,
                    (mat + cublas) / f32);
        std::printf("  fused fp16-acc (hacc): %.3f ms  (%.2fx vs shipped)\n", hacc,
                    (mat + cublas) / hacc);
        const double rF32 = flops / (f32 * 1e-3) / 1e12;
        const double rHacc = flops / (hacc * 1e-3) / 1e12;
        const double rCub = flops / (cublas * 1e-3) / 1e12;
        const double perTok = 2.0 * 24.75e9 * 512.0;// whole model, pp512, GEMM only
        std::printf("  (extrapolated to the whole model's 24.75e9 GEMM params at this "
                    "projection's rates, GEMM only: fused-fp32 %.0f ms, hacc %.0f ms, "
                    "cuBLAS %.0f ms -- per-shape spread is wider than this, so treat "
                    "as a ceiling)\n",
                    perTok / rF32 / 1e9, perTok / rHacc / 1e9, perTok / rCub / 1e9);
    }

    cublasDestroy(h);
    return 0;
}
