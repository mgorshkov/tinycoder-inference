/*
⚡ TinyCoder AI

Copyright (c) 2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// -----------------------------------------------------------------------------
// Persistent fused FFN mega-kernel (idea #21/A) — extracted translation unit.
//
// Self-contained: it re-declares the K-quant block byte sizes it needs and the
// per-row dequant-dot math (kQGemvQ2KxW4 / kQGemvQ3KxW4: 4 rows per warp, all
// 32 lanes issuing u32 weight loads, 4-way block batching + in-order fold +
// 8-lane shuffle tree), so it shares no hidden state with the main GPU driver.
//
// v2 (2026-10-08) optimizations over the first cut:
//   * `cooperative_groups::grid_group::sync()` replaces the hand-rolled atomic
//     barrier (correct at multi-block + much cheaper; the build targets
//     native sm_75/80/86, and cooperative launch is CUDA-graph-compatible).
//   * The 4-rows-per-warp geometry (the shape the default Q2_K/Q3_K GEMVs use,
//     ~145 GB/s vs ~103 GB/s for 1-row-per-warp) replaces the per-warp
//     single-row dot in phases 2/3.
//   * The barrier after the FFN RMSNorm is dropped: each block redundantly
//     normalizes the WHOLE hidden vector into its own shared memory, so phase 2
//     reads only its own block's shared — there is no cross-block dependency to
//     order.  Only phase 2->3 (productOut) and phase 3->4 (downOut) need a
//     grid sync.
// -----------------------------------------------------------------------------

#ifdef USE_CUDA

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

#include "FFNPersistentKernel.hpp"

namespace cg = cooperative_groups;

namespace tinycoder::gpu {
    namespace {

        // GGML QK_K block byte sizes (kQ2K_BYTES matches GPUCompute.cu).
        constexpr uint32_t kQ2K_BYTES = 84;
        constexpr uint32_t kQ3K_BYTES = 110;

        // One Q2_K weight row's 8-lane partial dot (kQGemvQ2KxW4 geometry: lane
        // I = lane&7 owns 4 contiguous columns per (half, 32-col group); the
        // u32 load feeds 4 columns × 4 jj positions).  EXACT columns: the
        // caller guarantees cols == blocksPerRow*256, so no guard is emitted.
        __device__ __forceinline__ float kFfnQ2KRowDot4W(
                const uint8_t *__restrict__ rp, const float *__restrict__ x,
                uint32_t I, uint32_t isub, uint32_t blocksPerRow) {
            const uint32_t qOfs = 4u * I;
            const uint32_t qb = qOfs;
            float acc = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 4u; ++bb) {
                    const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                    const float df = __half2float(
                            *reinterpret_cast<const __half *>(crp + 80));
                    const float dmf = __half2float(
                            *reinterpret_cast<const __half *>(crp + 82));
                    const uint8_t *sc = crp;
                    const uint8_t *q = crp + 16;
                    float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                            : (bb == 2u)       ? a2
                                                               : a3;
#pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        uint32_t w4;
                        std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                        const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                        for (uint32_t j = 0; j < 4u; ++j) {
                            const uint8_t scv = sc[n * 8u + 2u * j + isub];
                            const float dl = df * static_cast<float>(scv & 0xFu);
                            const float ml = dmf * static_cast<float>(scv >> 4u);
                            const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                            for (uint32_t k = 0; k < 4u; ++k) {
                                const float qv = static_cast<float>(
                                        static_cast<int8_t>(
                                                (w4 >> (8u * k + 2u * j)) & 3u));
                                slot = fmaf(fmaf(dl, qv, -ml), x[cbase + k], slot);
                            }
                        }
                    }
                }
                acc += a0;
                acc += a1;
                acc += a2;
                acc += a3;
                rp += 4u * kQ2K_BYTES;
            }
            for (; b < blocksPerRow; ++b) {
                const float df = __half2float(
                        *reinterpret_cast<const __half *>(rp + 80));
                const float dmf = __half2float(
                        *reinterpret_cast<const __half *>(rp + 82));
                const uint8_t *sc = rp;
                const uint8_t *q = rp + 16;
#pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    uint32_t w4;
                    std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                    const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
                    for (uint32_t j = 0; j < 4u; ++j) {
                        const uint8_t scv = sc[n * 8u + 2u * j + isub];
                        const float dl = df * static_cast<float>(scv & 0xFu);
                        const float ml = dmf * static_cast<float>(scv >> 4u);
                        const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                        for (uint32_t k = 0; k < 4u; ++k) {
                            const float qv = static_cast<float>(
                                    static_cast<int8_t>(
                                            (w4 >> (8u * k + 2u * j)) & 3u));
                            acc = fmaf(fmaf(dl, qv, -ml), x[cbase + k], acc);
                        }
                    }
                }
                rp += kQ2K_BYTES;
            }
            return acc;
        }

        // Q8_1 activation block layout (llama.cpp block_q8_1): 2 B fp16 d, 2 B
        // fp16 s, then 32 int8 qs.  Stride padded to 40 B so the int8 lanes and
        // the 4-way u32 loads stay aligned (matches kQ8_1_STRIDE in
        // GPUCompute.cu).
        constexpr uint32_t kQ8_1_STRIDE = 40;

        // 4-rows-per-warp Q2_K x Q8_1 dp4a partial dot.  Same 8-lane geometry as
        // kFfnQ2KRowDot4W (lane I owns columns [4I, 4I+4) of each 32-col group),
        // but the activation is the block-local Q8_1 in shared memory (sy) and
        // the per-element work is 1 u32 weight load + 1 u32 Q8_1 load + 2 dp4a +
        // 1 float mad, versus the float path's 1 u32 weight load + 4 fp32
        // activation loads + 4 fmaf.  Q8_1 block index for (Q2_K block b, half
        // n, 32-group j): b*8 + n*4 + j (8 groups of 32 per Q2_K block).
        // dq/dmin/sc come from the Q2_K weight block exactly as in the float
        // path; the sc byte's low nibble is the d scale and high nibble the
        // min.  Bit-identical to kQGemvQ2KxQ81_4xW (GPUCompute.cu).
        __device__ __forceinline__ float kFfnQ2KRowDot4W_Q81(
                const uint8_t *__restrict__ rp, const uint8_t *__restrict__ sy,
                uint32_t I, uint32_t isub, uint32_t blocksPerRow) {
            const uint32_t qOfs = 4u * I;
            float acc = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 4u; ++bb) {
                    const uint8_t *blkW =
                            rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                    const float dq = __half2float(
                            *reinterpret_cast<const __half *>(blkW + 80));
                    const float dmin = __half2float(
                            *reinterpret_cast<const __half *>(blkW + 82));
                    const uint8_t *sc = blkW;
                    const uint8_t *q = blkW + 16;
                    float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                            : (bb == 2u)       ? a2
                                                               : a3;
#pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        uint32_t w4;
                        std::memcpy(&w4, q + n * 32u + qOfs, sizeof(uint32_t));
                        uint64_t scU8;
                        std::memcpy(&scU8, sc + n * 8u, sizeof(uint64_t));
#pragma unroll
                        for (uint32_t j = 0; j < 4u; ++j) {
                            const uint8_t *blkY =
                                    sy + static_cast<uint64_t>(
                                                 (b + bb) * 8u + n * 4u + j) *
                                                 kQ8_1_STRIDE;
                            const float d8 = __half2float(
                                    *reinterpret_cast<const __half *>(blkY));
                            uint32_t y4;
                            std::memcpy(&y4, blkY + 4u + qOfs,
                                        sizeof(uint32_t));
                            const uint32_t pk = (w4 >> (2u * j)) & 0x03030303u;
                            const int pdot = __dp4a(static_cast<int>(pk),
                                                    static_cast<int>(y4), 0);
                            const int sumy = __dp4a(0x01010101,
                                                    static_cast<int>(y4), 0);
                            const uint8_t scv = static_cast<uint8_t>(
                                    (scU8 >> (8u * (2u * j + isub))) & 0xFFu);
                            slot += d8 *
                                    (dq * static_cast<float>(scv & 0xFu) *
                                             static_cast<float>(pdot) -
                                     dmin * static_cast<float>(scv >> 4u) *
                                             static_cast<float>(sumy));
                        }
                    }
                }
                acc += a0;
                acc += a1;
                acc += a2;
                acc += a3;
                rp += 4u * kQ2K_BYTES;
            }
            for (; b < blocksPerRow; ++b) {
                const float dq = __half2float(
                        *reinterpret_cast<const __half *>(rp + 80));
                const float dmin = __half2float(
                        *reinterpret_cast<const __half *>(rp + 82));
                const uint8_t *sc = rp;
                const uint8_t *q = rp + 16;
#pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    uint32_t w4;
                    std::memcpy(&w4, q + n * 32u + qOfs, sizeof(uint32_t));
                    uint64_t scU8;
                    std::memcpy(&scU8, sc + n * 8u, sizeof(uint64_t));
#pragma unroll
                    for (uint32_t j = 0; j < 4u; ++j) {
                        const uint8_t *blkY =
                                sy + static_cast<uint64_t>(
                                             b * 8u + n * 4u + j) *
                                             kQ8_1_STRIDE;
                        const float d8 = __half2float(
                                *reinterpret_cast<const __half *>(blkY));
                        uint32_t y4;
                        std::memcpy(&y4, blkY + 4u + qOfs, sizeof(uint32_t));
                        const uint32_t pk = (w4 >> (2u * j)) & 0x03030303u;
                        const int pdot = __dp4a(static_cast<int>(pk),
                                                static_cast<int>(y4), 0);
                        const int sumy = __dp4a(0x01010101,
                                                static_cast<int>(y4), 0);
                        const uint8_t scv = static_cast<uint8_t>(
                                (scU8 >> (8u * (2u * j + isub))) & 0xFFu);
                        acc += d8 *
                               (dq * static_cast<float>(scv & 0xFu) *
                                        static_cast<float>(pdot) -
                                dmin * static_cast<float>(scv >> 4u) *
                                        static_cast<float>(sumy));
                    }
                }
                rp += kQ2K_BYTES;
            }
            return acc;
        }

        // One Q3_K weight row's 8-lane partial dot (kQGemvQ3KxW4 geometry).
        __device__ __forceinline__ float kFfnQ3KRowDot4W(
                const uint8_t *__restrict__ rp, const float *__restrict__ x,
                uint32_t I, uint32_t isub, uint32_t blocksPerRow) {
            const uint32_t qOfs = 4u * I;
            float acc = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 4u; ++bb) {
                    const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ3K_BYTES;
                    const float df = __half2float(
                            *reinterpret_cast<const __half *>(crp + 108));
                    const uint8_t *hm = crp;
                    const uint8_t *q = crp + 32;
                    const uint8_t *scales = crp + 96;
                    uint32_t aux[4];
                    std::memcpy(&aux[0], scales, 12);
                    const uint32_t tmp = aux[2];
                    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) |
                             (((tmp >> 4) & 0x03030303u) << 4);
                    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) |
                             (((tmp >> 6) & 0x03030303u) << 4);
                    aux[0] = (aux[0] & 0x0f0f0f0fu) |
                             (((tmp >> 0) & 0x03030303u) << 4);
                    aux[1] = (aux[1] & 0x0f0f0f0fu) |
                             (((tmp >> 2) & 0x03030303u) << 4);
                    int8_t sc16[16];
                    std::memcpy(sc16, aux, 16);
                    float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                            : (bb == 2u)       ? a2
                                                               : a3;
#pragma unroll
                    for (uint32_t n = 0; n < 2u; ++n) {
                        uint32_t w4;
                        std::memcpy(&w4, q + n * 32u + qOfs, sizeof(uint32_t));
                        const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                        for (uint32_t j = 0; j < 4u; ++j) {
                            const float dl = df * static_cast<float>(
                                                          sc16[n * 8u + j * 2u + isub] - 32);
                            const uint32_t maskBit = 1u << (n * 4u + j);
                            const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                            for (uint32_t k = 0; k < 4u; ++k) {
                                const float qv = static_cast<float>(
                                        static_cast<int8_t>(
                                                ((w4 >> (8u * k + 2u * j)) & 3u) -
                                                ((hm[qOfs + k] & maskBit) ? 0 : 4)));
                                slot = fmaf(dl * qv, x[cbase + k], slot);
                            }
                        }
                    }
                }
                acc += a0;
                acc += a1;
                acc += a2;
                acc += a3;
                rp += 4u * kQ3K_BYTES;
            }
            for (; b < blocksPerRow; ++b) {
                const float df = __half2float(
                        *reinterpret_cast<const __half *>(rp + 108));
                const uint8_t *hm = rp;
                const uint8_t *q = rp + 32;
                const uint8_t *scales = rp + 96;
                uint32_t aux[4];
                std::memcpy(&aux[0], scales, 12);
                const uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) |
                         (((tmp >> 4) & 0x03030303u) << 4);
                aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) |
                         (((tmp >> 6) & 0x03030303u) << 4);
                aux[0] = (aux[0] & 0x0f0f0f0fu) |
                         (((tmp >> 0) & 0x03030303u) << 4);
                aux[1] = (aux[1] & 0x0f0f0f0fu) |
                         (((tmp >> 2) & 0x03030303u) << 4);
                int8_t sc16[16];
                std::memcpy(sc16, aux, 16);
#pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    uint32_t w4;
                    std::memcpy(&w4, q + n * 32u + qOfs, sizeof(uint32_t));
                    const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
                    for (uint32_t j = 0; j < 4u; ++j) {
                        const float dl = df * static_cast<float>(
                                                      sc16[n * 8u + j * 2u + isub] - 32);
                        const uint32_t maskBit = 1u << (n * 4u + j);
                        const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                        for (uint32_t k = 0; k < 4u; ++k) {
                            const float qv = static_cast<float>(
                                    static_cast<int8_t>(
                                            ((w4 >> (8u * k + 2u * j)) & 3u) -
                                            ((hm[qOfs + k] & maskBit) ? 0 : 4)));
                            acc = fmaf(dl * qv, x[cbase + k], acc);
                        }
                    }
                }
                rp += kQ3K_BYTES;
            }
            return acc;
        }

        // 8-lane-of-4 shuffle tree (xor 4,1,2 stays inside each row's 8-lane
        // group -- same tree as kQGemvQ2KxW4 / kQGemvQ3KxW4).
        __device__ __forceinline__ float kFfnReduce8(float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            return v;
        }

        // Grid is clamped to the cooperative-launch residency by the launcher.
        // phases: (1) FFN RMSNorm (redundant per block, into shared) ->
        //         (2) gate+up 4xW + silu -> productOut -> grid.sync
        //         (3) down 4xW + residual (hidden += d) -- the residual is
        //             folded into the down row write, so there is exactly ONE
        //             grid barrier per layer.
        // maxBlocksPerMultiprocessor=2: the fused kernel otherwise compiles to
        // ~219 regs (1 block/SM, 8 warps) -- far too few warps to hide the
        // DRAM latency of the memory-bound GEMV phases.  Asking for 2 caps the
        // reg budget at 128 and roughly doubles resident warps (measured: lb2
        // ~162 tok/s vs lb1 ~156 vs lb3 ~145; see plan 8.5).
        //
        // Q81 (2026-10-08): template flag selecting the gate/up activation
        // path.  false = fp32 block-local norm (kFfnQ2KRowDot4W).  true =
        // phase 1 additionally emits llama.cpp's Q8_1 into shared and gate/up
        // use the integer dp4a dot (kFfnQ2KRowDot4W_Q81), trading a tiny
        // in-kernel quantize for 1 B/elem activation reads in phase 2.  The
        // down phase always consumes the fp32 silu product, so it is shared.
        template<bool Q81>
        __global__ void __launch_bounds__(256, 2) kFfnPersistentLayer(
                const float *__restrict__ hiddenRes,// [H] residual stream in
                float *__restrict__ hiddenOut,      // [H] residual stream out
                const float *__restrict__ ffnNormW, // [H] rmsNormFFN weight
                const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
                const uint8_t *__restrict__ wd,
                float *__restrict__ productOut,// [I] = silu(gate)*up
                float *__restrict__ downOut,   // [H] = down(product)
                uint32_t H, uint32_t I, uint32_t ffnRows, uint32_t downRows,
                uint32_t gateRowBytes, uint32_t downRowBytes,
                uint32_t gateBlocksPerRow, uint32_t downBlocksPerRow, float eps) {
            cg::grid_group grid = cg::this_grid();
            extern __shared__ float s_ffn[];// H norm floats + 256 sum tree
            float *ssum = s_ffn + H;
            // Q8_1 region (only used when Q81): one 40 B block per 32 columns.
            uint8_t *sy = reinterpret_cast<uint8_t *>(ssum + 256u);
            const uint32_t tid = threadIdx.x;
            const uint32_t lane = tid & 31u;
            const uint32_t warp = tid >> 5u;
            const uint32_t nWarps = blockDim.x >> 5u;
            const uint32_t warpId = blockIdx.x * nWarps + warp;
            const uint32_t totalWarps = gridDim.x * nWarps;
            const uint32_t I8 = lane & 7u;         // 8 lanes per row
            const uint32_t rowInGroup = lane >> 3u;// 0..3
            const uint32_t isub = (4u * I8 < 16u) ? 0u : 1u;

            // Phase 1: FFN RMSNorm -> s_ffn (verbatim kRMSNormRow body).  Each
            // block normalizes the whole vector into its OWN shared memory, so
            // no grid sync is needed before phase 2.
            {
                float a = 0.0f;
                for (uint32_t i = tid; i < H; i += blockDim.x) {
                    const float v = hiddenRes[i];
                    a = fmaf(v, v, a);
                }
                ssum[tid] = a;
                __syncthreads();
                for (uint32_t s = blockDim.x >> 1u; s > 0u; s >>= 1u) {
                    if (tid < s) ssum[tid] += ssum[tid + s];
                    __syncthreads();
                }
                const float rms = rsqrtf(ssum[0] / static_cast<float>(H) + eps);
                for (uint32_t i = tid; i < H; i += blockDim.x)
                    s_ffn[i] = hiddenRes[i] * rms * ffnNormW[i];
            }
            // Phase 2 warps each read the WHOLE normalized vector, so the
            // block-local write->read of s_ffn must be ordered.  (Dropping the
            // grid barrier after phase 1 is safe; dropping this __syncthreads
            // is NOT -- grid=1 happened to hide the race, grid>=2 exposed it.)
            __syncthreads();

            // Phase 1b (Q81): quantize the block-local fp32 norm into llama.cpp
            // block_q8_1 in shared (d = amax/127, q = round(x/d), s = d*sum via
            // the fp16 s slot), one warp per 32 columns.  This is the phase the
            // standalone path spent two GLOBAL launches on; here it never leaves
            // shared.  H is a multiple of 32 for every supported FFN (guarded by
            // the caller before the Q81 path is selected).
            if (Q81) {
                const uint32_t q8Blocks = H >> 5u;
                for (uint32_t blk = warp; blk < q8Blocks; blk += nWarps) {
                    const uint32_t e0 = blk * 32u;
                    uint8_t *dst =
                            sy + static_cast<uint64_t>(blk) * kQ8_1_STRIDE;
                    int8_t *qs = reinterpret_cast<int8_t *>(dst + 4);
                    const float xi = s_ffn[e0 + lane];
                    float amax = fabsf(xi);
                    float sum = xi;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        amax = fmaxf(amax,
                                     __shfl_xor_sync(0xffffffffu, amax, o));
                        sum += __shfl_xor_sync(0xffffffffu, sum, o);
                    }
                    const float d = amax / 127.0f;
                    qs[lane] = (amax == 0.0f)
                                       ? static_cast<int8_t>(0)
                                       : static_cast<int8_t>(roundf(xi / d));
                    if (lane == 0u) {
                        __half *ds = reinterpret_cast<__half *>(dst);
                        ds[0] = __float2half(d);
                        ds[1] = __float2half(sum);
                    }
                }
                __syncthreads();
            }

            // Phase 2: gate + up + silu (4 rows per warp, 8 lanes/row), reading
            // the block's OWN shared norm (fp32 or Q8_1).  ffnRows is a multiple
            // of 4 for every supported model, so all 4 rows of a group are valid.
            for (uint32_t base = warpId * 4u; base < ffnRows;
                 base += totalWarps * 4u) {
                const uint32_t row = base + rowInGroup;
                const uint8_t *rpg = wg + static_cast<uint64_t>(row) * gateRowBytes;
                const uint8_t *rpu = wu + static_cast<uint64_t>(row) * gateRowBytes;
                float g, u;
                if (Q81) {
                    g = kFfnQ2KRowDot4W_Q81(rpg, sy, I8, isub, gateBlocksPerRow);
                    u = kFfnQ2KRowDot4W_Q81(rpu, sy, I8, isub, gateBlocksPerRow);
                } else {
                    g = kFfnQ2KRowDot4W(rpg, s_ffn, I8, isub, gateBlocksPerRow);
                    u = kFfnQ2KRowDot4W(rpu, s_ffn, I8, isub, gateBlocksPerRow);
                }
                g = kFfnReduce8(g);
                u = kFfnReduce8(u);
                if (I8 == 0u)
                    productOut[row] = (g / (1.0f + __expf(-g))) * u;
            }
            grid.sync();

            // Phase 3: down (4 rows per warp), reading productOut from global.
            // The down output row r IS residual element r (down.rows == H is
            // guaranteed by the caller), so fold the residual add in here: the
            // row's owning lane (I8==0) writes hiddenOut[row] += d.  This
            // removes phase 4 AND the second grid.sync() outright -- one grid
            // barrier per layer instead of two.  hiddenRes is not read by any
            // phase after phase 1, so the in-place hiddenRes==hiddenOut update
            // is race-free.
            for (uint32_t base = warpId * 4u; base < downRows;
                 base += totalWarps * 4u) {
                const uint32_t row = base + rowInGroup;
                const uint8_t *rpd = wd + static_cast<uint64_t>(row) * downRowBytes;
                float d = kFfnQ3KRowDot4W(rpd, productOut, I8, isub,
                                          downBlocksPerRow);
                d = kFfnReduce8(d);
                if (I8 == 0u) hiddenOut[row] += d;
            }

            (void) I;
            (void) downOut;// residual is fused into the phase-3 write
        }

    }// namespace

    bool launchFfnPersistentLayer(const FfnPersistentArgs &a, cudaStream_t stream) {
        // ---- device capability: cooperative launch --------------------------
        static int smCount = -1;
        static int coopLaunch = 0;
        if (smCount < 0) {
            if (cudaDeviceGetAttribute(&smCount,
                                       cudaDevAttrMultiProcessorCount,
                                       0) != cudaSuccess) {
                smCount = -1;
                return false;
            }
            if (cudaDeviceGetAttribute(&coopLaunch,
                                       cudaDevAttrCooperativeLaunch,
                                       0) != cudaSuccess) {
                coopLaunch = 0;
            }
        }
        if (!coopLaunch) return false;

        // Shared memory: H norm floats + the 256-entry reduction tree (+ one
        // 40 B Q8_1 block per 32 columns when the Q8_1 path is selected).
        const size_t smemBase =
                (static_cast<size_t>(a.H) + 256u) * sizeof(float);
        const size_t smemQ81 =
                smemBase + (static_cast<size_t>(a.H) / 32u) * kQ8_1_STRIDE;

        // Pick the Q8_1 specialization only when the caller asked, H is a
        // multiple of 32 (the Q8_1 block width) and the extra shared fits the
        // device's opt-in limit.  Otherwise fall back to the fp32 path.
        bool q81 = a.q81 && (a.H % 32u == 0u);
        if (q81) {
            static long maxSmemOptin = -1;
            if (maxSmemOptin < 0) {
                int v = 0;
                maxSmemOptin = (cudaDeviceGetAttribute(
                                        &v,
                                        cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                        0) == cudaSuccess)
                                       ? static_cast<long>(v)
                                       : 0L;
            }
            if (maxSmemOptin <= 0 ||
                smemQ81 > static_cast<size_t>(maxSmemOptin)) {
                q81 = false;
            }
        }

        int maxBlocks = 0;
        if (q81 &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &maxBlocks, kFfnPersistentLayer<true>, 256, smemQ81) ==
                    cudaSuccess &&
            maxBlocks >= 1) {
            // Opt-in: the dynamic shared request exceeds the 48 KB default.
            cudaFuncSetAttribute(kFfnPersistentLayer<true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smemQ81));
        } else {
            q81 = false;
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &maxBlocks, kFfnPersistentLayer<false>, 256, smemBase) !=
                cudaSuccess) {
                return false;
            }
        }
        const size_t smem = q81 ? smemQ81 : smemBase;
        if (maxBlocks < 1) return false;
        // Cooperative launch requires grid <= max co-resident blocks x SMs.
        uint32_t grid =
                static_cast<uint32_t>(maxBlocks) * static_cast<uint32_t>(smCount);
        // Experiment knob: cap the grid (debug / occupancy sweep).
        const char *gE = std::getenv("TINYCODER_FFN_PERSIST_GRID");
        if (gE != nullptr) {
            const long gv = std::strtol(gE, nullptr, 10);
            if (gv > 0 && static_cast<uint32_t>(gv) < grid)
                grid = static_cast<uint32_t>(gv);
        }
        if (grid == 0) return false;

        const float *hiddenRes = a.hiddenRes;
        float *hiddenOut = a.hiddenOut;
        const float *ffnNormW = a.ffnNormW;
        const uint8_t *wg = a.wg;
        const uint8_t *wu = a.wu;
        const uint8_t *wd = a.wd;
        float *productOut = a.productOut;
        float *downOut = a.downOut;
        uint32_t H = a.H;
        uint32_t I = a.I;
        uint32_t ffnRows = a.ffnRows;
        uint32_t downRows = a.downRows;
        uint32_t gateRowBytes = a.gateRowBytes;
        uint32_t downRowBytes = a.downRowBytes;
        uint32_t gateBlocksPerRow = a.gateBlocksPerRow;
        uint32_t downBlocksPerRow = a.downBlocksPerRow;
        float eps = a.eps;
        void *args[] = {&hiddenRes, &hiddenOut, &ffnNormW, &wg,
                        &wu, &wd, &productOut, &downOut,
                        &H, &I, &ffnRows, &downRows,
                        &gateRowBytes, &downRowBytes, &gateBlocksPerRow,
                        &downBlocksPerRow, &eps};
        const void *fn = q81
                                 ? reinterpret_cast<const void *>(
                                           kFfnPersistentLayer<true>)
                                 : reinterpret_cast<const void *>(
                                           kFfnPersistentLayer<false>);
        cudaError_t e = cudaLaunchCooperativeKernel(fn, dim3(grid), dim3(256),
                                                    args, smem, stream);
        if (std::getenv("TINYCODER_FFN_PERSIST_DEBUG") != nullptr) {
            std::fprintf(stderr,
                         "[ffn-persist] sms=%d maxBlocksPerSM=%d grid=%u "
                         "smem=%zu q81=%d coopLaunchResult=%s\n",
                         smCount, maxBlocks, grid, smem, q81 ? 1 : 0,
                         e == cudaSuccess ? "ok" : cudaGetErrorString(e));
        }
        return e == cudaSuccess;
    }

}// namespace tinycoder::gpu

#endif// USE_CUDA
