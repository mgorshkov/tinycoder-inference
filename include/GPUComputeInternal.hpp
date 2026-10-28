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

/*
 * CUDA GPU offload engine for TinyCoder.
 *
 * Two execution strategies (RTX 2080 Ti / Turing reference target):
 *
 * 1. Single-token generation (seqLen == 1) is DRAM-bandwidth-bound.
 *
 *   2. Batched prefill (seqLen > 1) is compute-bound.  Turing has no
 *      quantized tensor cores, so we materialize FP16 twins of all weights
 *      once at upload and run cublasGemmEx fp16 GEMMs (fp32 accumulate).
 *
 * Partial layer offload: upload() takes numGpuLayers (default = numLayers)
 * so only the first that many layers live on the GPU; the remainder can be
 * computed on the CPU by the caller (forward() copies the final hidden state
 * back instead of running the LM head when fewer than numLayers layers are
 * offloaded).
 *
 * Layer math (Qwen2 dense block), replicated vs tinycoder's CPU reference:
 *   hidden -> rmsNorm -> Q/K/V (+biases) -> RoPE(Q) + store KV(rotated K)
 *   -> warp flash attention -> attnO -> residual -> rmsNorm -> gate*up
 *   (SwiGLU silu fused) -> down -> residual; final rmsNorm -> LM head.
 */


// ============================================================================
// GPUComputeInternal.hpp -- INTERNAL shared implementation of the CUDA
// offload engine.  Created by splitting the former monolithic GPUCompute.cu
// into several translation units:
//   * GPUComputeGemm.cu      -- debug/parity probes (Q8_K / Q8_0 / MMQ rows)
//   * GPUComputeUpload.cu    -- scratch + weight upload / expert cache
//   * GPUComputeMoE.cu       -- resident-expert arenas + qwen35 prefix
//   * GPUComputeForward.cu   -- qwen35moe hybrid, forward(), CUDA graph
// This header holds the complete ANONYMOUS-NAMESPACE kernel + device-helper
// set.  Anonymous-namespace entities have INTERNAL linkage, so including this
// header in every .cu gives each TU its own copy -- no ODR/link collisions.
// The only cross-TU mutable state (the CUDA streams) lives OUTSIDE the
// anonymous namespace as `inline` variables (one shared instance).
// ============================================================================

#pragma once

#ifdef USE_CUDA

#include "FFNPersistentKernel.hpp"
#include "GPUCompute.hpp"
#include "ThreadPool.hpp"
#if defined(__linux__)
#include <sys/mman.h>
#endif

// Device-side copies of the IQ2_XS / IQ3_S grid + sign lookup tables
// (byte-identical to the host tables in GridTables.cpp, GridTablesIQ3S.cpp
// and GGMLDequantize.hpp; auto-generated into this header).
#include "GridTablesDevice.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace tinycoder::gpu {
    // ---- Cross-translation-unit GPU state ----
    // The CUDA stream / cuBLAS handle / init flag are SINGLE shared instances
    // (inline variables, external linkage) because the kernel + driver code is
    // now split across several translation units that all include this header.
    // A per-TU copy would give each TU a different stream and break kernel
    // ordering between them.
    inline cudaStream_t g_stream = nullptr;
    inline cublasHandle_t g_cublas = nullptr;
    // Secondary non-blocking stream (copy-engine only).  The CPU-expert
    // hybrid handoff (forwardQwen35MoePrefix) uses it for the norm D2H + FFN
    // H2D copies so they OVERLAP the GPU's g_stream compute.
    inline cudaStream_t g_stream2 = nullptr;
    inline std::once_flag g_cudaInit;
    // TINYCODER_GPU_VERBOSE=1: per-stage stream syncs + print (debug aid).
    inline const bool g_verbose = [] {
        const char *e = std::getenv("TINYCODER_GPU_VERBOSE");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();

    namespace {


        // GGML type codes (GGUFLoader.hpp).
        constexpr uint32_t kTypeF32 = 0;
        constexpr uint32_t kTypeQ5_0 = 6;
        constexpr uint32_t kTypeQ8_0 = 8;
        constexpr uint32_t kTypeQ2K = 10;
        constexpr uint32_t kTypeQ3K = 11;
        constexpr uint32_t kTypeQ4K = 12;
        constexpr uint32_t kTypeQ5K = 13;
        constexpr uint32_t kTypeQ6K = 14;
        constexpr uint32_t kTypeIQ2XS = 17;
        constexpr uint32_t kTypeIQ3_XXS = 18;
        constexpr uint32_t kTypeIQ2XXS = 16;
        constexpr uint32_t kTypeIQ4NL = 20;
        constexpr uint32_t kTypeIQ4XS = 23;
        constexpr uint32_t kTypeIQ3S = 21;
        constexpr uint32_t kTypeIQ2S = 22;
        // Prism ternary (Bonsai) — 2 bits/weight, 34-byte / 128-weight blocks.
        constexpr uint32_t kTypePQ2_0 = 142;
        constexpr uint32_t kPQ2_0_BYTES = 34;// d(fp16,2) + qs[32]
        // Prism Hadamard-fold block size (prism.hadamard.block_size).
        constexpr uint32_t kPrismBlock = 1024;

        constexpr uint32_t kQ5_0_BYTES = 22;
        constexpr uint32_t kQ8_0_BYTES = 34;
        constexpr uint32_t kQ2K_BYTES = 84;
        constexpr uint32_t kQ3K_BYTES = 110;
        constexpr uint32_t kQ5K_BYTES = 176;
        constexpr uint32_t kQ4K_BYTES = 144;
        constexpr uint32_t kQ6K_BYTES = 210;
        constexpr uint32_t kIQ2XS_BYTES = 74;
        constexpr uint32_t kIQ3S_BYTES = 110;
        constexpr uint32_t kIQ3XXS_BYTES = 98;
        constexpr uint32_t kIQ2S_BYTES = 82;
        constexpr uint32_t kIQ2XXS_BYTES = 66;
        constexpr uint32_t kIQ4NL_BYTES = 18;// d(fp16,2) + qs[16]
        constexpr uint32_t kIQ4XS_BYTES = 136;

        constexpr float LOG2E = 1.4426950408889634f;


        bool ensureCuda() {
            std::call_once(g_cudaInit, []() {
                if (cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking) !=
                    cudaSuccess)
                    return;
                if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) return;
                cublasSetStream(g_cublas, g_stream);
                if (cudaStreamCreateWithFlags(&g_stream2,
                                              cudaStreamNonBlocking) !=
                    cudaSuccess)
                    return;
            });
            return g_stream && g_cublas && g_stream2;
        }

        // ------------------------------------------------------------------
        // Elementwise
        // ------------------------------------------------------------------

        __global__ void kAddResidual(float *dst, const float *src, uint32_t n) {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) dst[i] += src[i];
        }

        // Broadcast a per-row bias across every row of a [rows][biasLen] tensor
        // (Qwen2 Q/K/V biases).  One block per row; the same bias[c] is added to
        // every row, so the bias is never indexed past its biasLen elements.
        __global__ void kAddBias(float *dst, const float *bias, uint32_t rows,
                                 uint32_t biasLen) {
            uint32_t s = blockIdx.x;
            if (s >= rows) return;
            float *row = dst + static_cast<size_t>(s) * biasLen;
            for (uint32_t c = threadIdx.x; c < biasLen; c += blockDim.x) {
                row[c] += bias[c];
            }
        }

        __global__ void kSiluMul(float *a, const float *b, uint32_t n) {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                float x = a[i];
                a[i] = (x / (1.0f + __expf(-x))) * b[i];
            }
        }

        // ---- KV-cache element access (fp16 KV cache, TINYCODER_KV16) ----
        // The attention / KV-store kernels are templated on the cache element
        // type so an fp16 cache halves the O(context) attention + RoPE DRAM
        // traffic that dominates long-context decode.  float is the default,
        // so every existing instantiation is bit-identical to the legacy path.
        __device__ __forceinline__ float kvLoad(const float *p, uint32_t i) {
            return p[i];
        }
        __device__ __forceinline__ float kvLoad(const __half *p, uint32_t i) {
            return __half2float(p[i]);
        }
        // Scalar form for values already held in registers (4-position load
        // batching in the split-KV attention): same element conversion as
        // kvLoad, no memory access.
        __device__ __forceinline__ float kvScalar(const float &v) {
            return v;
        }
        __device__ __forceinline__ float kvScalar(const __half &v) {
            return __half2float(v);
        }
        __device__ __forceinline__ void kvStore(float *p, uint32_t i, float v) {
            p[i] = v;
        }
        __device__ __forceinline__ void kvStore(__half *p, uint32_t i, float v) {
            p[i] = __float2half(v);
        }

        // Elementwise f32 -> f16: x[rows*n] fp32 -> out[rows*n] fp16.
        __global__ void kF32ToF16(const float *__restrict__ x,
                                  __half2 *__restrict__ out, uint32_t pairs) {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= pairs) return;
            // each thread converts 2 consecutive floats into one __half2
            const uint32_t e0 = 2u * i;
            float a = x[e0];
            float b = x[e0 + 1];
            out[i] = __floats2half2_rn(a, b);
        }

        // One block (256 threads) per row of the [rows][n] matrix.
        // OUT-OF-PLACE: y = rmsNorm(x, w).  `x` is the residual stream and MUST
        // NOT be overwritten (the CPU reference norms into a separate buffer and
        // keeps the residual stream intact for the residual adds).
        __global__ void kRMSNormRow(const float *x, float *y, const float *w,
                                    uint32_t n, uint32_t rows, float eps) {
            uint32_t row = blockIdx.x;
            if (row >= rows) return;
            const float *rp = x + static_cast<size_t>(row) * n;
            float *op = y + static_cast<size_t>(row) * n;
            extern __shared__ float ssum[];
            float acc = 0.0f;
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                float v = rp[i];
                acc = fmaf(v, v, acc);
            }
            ssum[threadIdx.x] = acc;
            __syncthreads();
            for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
                __syncthreads();
            }
            float rms = rsqrtf(ssum[0] / static_cast<float>(n) + eps);
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                op[i] = rp[i] * rms * w[i];
            }
        }

        // kRMSNormRow + Q8_1 fold (2026-10-10, plan section 9.26 lever 1): the
        // rms_norm block already holds the whole row, so it also emits the
        // Q8_1 activation blocks the dp4a GLU mmvq consumes -- deleting the
        // separate kQuantizeQ8_1 launch (2.5 us/layer).  The quantize math is
        // copied VERBATIM from kQuantizeQ8_1 (amax/127, roundf(v/d), s = sum
        // of the FLOATS) over the same float values the float path stores, so
        // the Q8_1 bytes are bit-identical to the two-kernel sequence.
        // Requires blockDim 1024 and n == 1536 (thread t owns elements t and
        // t+1024: warp w covers Q8_1 block w from its first element and, for
        // w < 16, block 32 + w from its second).  Launched only for the dense
        // decode (seqLen==1, H==1536); every other shape keeps the separate
        // launch.  TINYCODER_FUSE_Q81_PROD=0 restores it.
        __global__ void kRMSNormRowQ81(const float *x, float *y, const float *w,
                                       uint8_t *__restrict__ q81, uint32_t n,
                                       uint32_t rows, float eps) {
            uint32_t row = blockIdx.x;
            if (row >= rows) return;
            const float *rp = x + static_cast<size_t>(row) * n;
            float *op = y + static_cast<size_t>(row) * n;
            extern __shared__ float ssum[];
            float acc = 0.0f;
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                float v = rp[i];
                acc = fmaf(v, v, acc);
            }
            ssum[threadIdx.x] = acc;
            __syncthreads();
            for (uint32_t s = blockDim.x / 2; s >= 32u; s >>= 1) {
                if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
                __syncthreads();
            }
            // Last 5 reduction levels (16..1) touch only warp 0.  Replace the
            // five remaining __syncthreads with warp shuffles at the IDENTICAL
            // pairing (lane t adds lane t+delta), so the float sum is
            // bit-identical while the barrier chain is halved (2026-10-11).
            if (threadIdx.x < 32u) {
                float v = ssum[threadIdx.x];
#pragma unroll
                for (uint32_t o = 16u; o > 0u; o >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, o);
                if (threadIdx.x == 0u) ssum[0] = v;
            }
            __syncthreads();
            float rms = rsqrtf(ssum[0] / static_cast<float>(n) + eps);
            const uint32_t i0 = threadIdx.x;
            const uint32_t i1 = threadIdx.x + blockDim.x;
            float v0 = rp[i0] * rms * w[i0];
            op[i0] = v0;
            float v1 = 0.0f;
            const bool has1 = (i1 < n);
            if (has1) {
                v1 = rp[i1] * rms * w[i1];
                op[i1] = v1;
            }
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5u;
            const uint64_t rowBlk = static_cast<uint64_t>(row) * (n / 32u);
            // block(warp) <- v0 (element i0 = warp*32 + lane)
            {
                float amax = fabsf(v0), sum = v0;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                    sum += __shfl_xor_sync(0xffffffffu, sum, o);
                }
                const float d = amax / 127.0f;
                uint8_t *blk = q81 + (rowBlk + warp) * 40u;
                reinterpret_cast<int8_t *>(blk + 4)[lane] =
                        (amax == 0.0f)
                                ? static_cast<int8_t>(0)
                                : static_cast<int8_t>(roundf(v0 / d));
                if (lane == 0) {
                    __half *ds = reinterpret_cast<__half *>(blk);
                    ds[0] = __float2half(d);
                    ds[1] = __float2half(sum);
                }
            }
            // block(32 + warp) <- v1 (element i1 = 1024 + warp*32 + lane)
            if (has1) {
                float amax = fabsf(v1), sum = v1;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                    sum += __shfl_xor_sync(0xffffffffu, sum, o);
                }
                const float d = amax / 127.0f;
                uint8_t *blk = q81 + (rowBlk + 32u + warp) * 40u;
                reinterpret_cast<int8_t *>(blk + 4)[lane] =
                        (amax == 0.0f)
                                ? static_cast<int8_t>(0)
                                : static_cast<int8_t>(roundf(v1 / d));
                if (lane == 0) {
                    __half *ds = reinterpret_cast<__half *>(blk);
                    ds[0] = __float2half(d);
                    ds[1] = __float2half(sum);
                }
            }
        }

        // ------------------------------------------------------------------
        // RoPE on Q, and K/V store with fused K rotation (matches
        // ModelPrimitives.cpp applyRoPE + storeKVWithRoPE element order).
        // ------------------------------------------------------------------

        // q[k] into q[k] (in place). One block per token, 128 threads.
        // (Interleaved (2j,2j+1) was the wrong transform for pos>0 and produced
        // garbage multi-token generations; identity at pos 0 masked it in
        // single-token dumps.)
        __global__ void kRoPEQ(float *q, const float *cosT, const float *sinT,
                               uint32_t seqLen, uint32_t qHeads, uint32_t headDim,
                               uint32_t pos) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            uint32_t p = pos + s;
            uint32_t pairs = headDim / 2;
            const float *c = cosT + static_cast<size_t>(p) * pairs;
            const float *sn = sinT + static_cast<size_t>(p) * pairs;
            // every thread walks its own column slice across all heads
            // Flatten (head, pair): all blockDim threads share the
            // qHeads*pairs rotation instead of 64 threads walking all 12 heads
            // serially -- same per-element fmaf chain, bit-identical.
            const uint32_t total = qHeads * pairs;
            for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
                const uint32_t h = idx / pairs;
                const uint32_t j = idx - h * pairs;
                float cc = c[j], ss = sn[j];
                float *head = q + (static_cast<size_t>(s) * qHeads + h) * headDim;
                float x0 = head[j], x1 = head[j + pairs];
                head[j] = fmaf(x0, cc, -x1 * ss);
                head[j + pairs] = fmaf(x0, ss, x1 * cc);
            }
        }

        // kSrc/vSrc [seqLen][kHeads*headDim] -> kDst/vDst cache at cachePos.
        // One block (256 threads) per token.
        template<typename KVT>
        __global__ void kStoreKVRope(const float *kSrc, const float *vSrc, KVT *kDst,
                                     KVT *vDst, uint32_t seqLen, uint32_t kHeads,
                                     uint32_t headDim, uint32_t cachePos, uint32_t pos,
                                     const float *cosT, const float *sinT) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            uint32_t p = pos + s;
            uint32_t pairs = headDim / 2;
            uint32_t kvSize = kHeads * headDim;
            const float *ks = kSrc + static_cast<size_t>(s) * kvSize;
            const float *vs = vSrc + static_cast<size_t>(s) * kvSize;
            KVT *kd = kDst + static_cast<size_t>(cachePos + s) * kvSize;
            KVT *vd = vDst + static_cast<size_t>(cachePos + s) * kvSize;
            const float *c = cosT + static_cast<size_t>(p) * pairs;
            const float *sn = sinT + static_cast<size_t>(p) * pairs;
            uint32_t t = threadIdx.x;
            // V copied verbatim for ALL columns.
            for (uint32_t e = t; e < kvSize; e += blockDim.x) {
                kvStore(vd, e, vs[e]);
            }
            // K rotated per-head on the NEOX (j, j+pairs) pairs, angle index j.
            const uint32_t numPairs = kHeads * pairs;
            for (uint32_t e = t; e < numPairs; e += blockDim.x) {
                uint32_t h = e / pairs;
                uint32_t j = e - h * pairs;
                uint32_t base = h * headDim + j;
                float k0 = ks[base], k1 = ks[base + pairs];
                kvStore(kd, base, fmaf(k0, c[j], -k1 * sn[j]));
                kvStore(kd, base + pairs, fmaf(k0, sn[j], k1 * c[j]));
            }
        }

        // ------------------------------------------------------------------
        // Flash attention: one warp per (token, q-head).  Online softmax,
        // fp32 accumulate, causal mask via csEnd = cachePos + s.
        // ------------------------------------------------------------------

        __device__ __forceinline__ float kWarpReduceSum(float v) {
            for (uint32_t off = 16; off; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            return __shfl_sync(0xffffffffu, v, 0);
        }

        // Flash attention: one warp per (token, q-head), compile-time headDim.
        // HD/32 register accumulators per lane (lane owns elements lane+i*32),
        // so nothing spills to local memory (vs. the old float o[128] indexed by
        // a runtime loop bound, which NVCC keeps in local memory).  Grid.y must
        // cover nHeads warps; idle warps exit immediately.
        template<uint32_t HD, typename KVT>
        __global__ void kWarpAttention(const float *__restrict__ q,
                                       const KVT *__restrict__ kCache,
                                       const KVT *__restrict__ vCache,
                                       float *__restrict__ out, uint32_t seqLen,
                                       uint32_t nHeads, uint32_t nKVHeads,
                                       uint32_t cachePos, float invSqrtHeadDim) {
            static_assert(HD == 128 || HD == 64 || HD == 32, "HD must be 32/64/128");
            constexpr uint32_t NV = HD / 32;
            uint32_t s = blockIdx.x;
            uint32_t warpId = blockIdx.y * blockDim.y + threadIdx.y;
            uint32_t lane = threadIdx.x;
            if (s >= seqLen || warpId >= nHeads) return;
            uint32_t qHead = warpId;
            uint32_t kvHead = qHead / (nHeads / nKVHeads);
            uint32_t csEnd = cachePos + s;

            const float *qPtr = q + (static_cast<size_t>(s) * nHeads + qHead) * HD;
            float acc[NV];
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
            float m = -1e30f, l = 0.0f;

            for (uint32_t cs = 0; cs <= csEnd; ++cs) {
                const KVT *kPtr =
                        kCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                const KVT *vPtr =
                        vCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                // score = dot(q, k) * invSqrt, warp-reduced
                float sc = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) {
                    sc = fmaf(qPtr[lane + i * 32], kvLoad(kPtr, lane + i * 32), sc);
                }
                sc = kWarpReduceSum(sc) * invSqrtHeadDim;

                if (sc > m) {
                    float mNew = sc;
                    float alpha = exp2f((m - mNew) * LOG2E);
                    if (alpha != 1.0f) {
#pragma unroll
                        for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
                    }
                    l *= alpha;
                    m = mNew;
                }
                float pp = exp2f((sc - m) * LOG2E);
                l += pp;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) {
                    acc[i] = fmaf(pp, kvLoad(vPtr, lane + i * 32), acc[i]);
                }
            }
            float invL = 1.0f / l;
            float *outPtr = out + (static_cast<size_t>(s) * nHeads + qHead) * HD;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) {
                outPtr[lane + i * 32] = acc[i] * invL;
            }
        }

        // ---- CUDA-graph variants (pos from DEVICE memory) ----
        // A captured CUDA graph freezes kernel arguments.  The RoPE / KV-store
        // / attention kernels take `pos`/`cachePos` as plain values; for
        // graph capture the position must be read from a device-resident
        // uint32 that the host refreshes via cudaMemcpyAsync before each
        // replay.  These variants replace the value param with a const
        // uint32_t* (pointer is frozen; the POINTED-TO value is read at kernel
        // time).  The math is line-for-line identical, so graph vs eager
        // outputs are bit-identical (49/49 parity preserved).
        __global__ void kRoPEQPos(float *q, const float *cosT,
                                  const float *sinT, uint32_t seqLen,
                                  uint32_t qHeads, uint32_t headDim,
                                  const uint32_t *pos) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            uint32_t p = *pos + s;
            uint32_t pairs = headDim / 2;
            const float *c = cosT + static_cast<size_t>(p) * pairs;
            const float *sn = sinT + static_cast<size_t>(p) * pairs;
            // Flatten (head, pair): all blockDim threads share the
            // qHeads*pairs rotation instead of 64 threads walking all 12 heads
            // serially -- same per-element fmaf chain, bit-identical.
            const uint32_t total = qHeads * pairs;
            for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
                const uint32_t h = idx / pairs;
                const uint32_t j = idx - h * pairs;
                float cc = c[j], ss = sn[j];
                float *head = q + (static_cast<size_t>(s) * qHeads + h) * headDim;
                float x0 = head[j], x1 = head[j + pairs];
                head[j] = fmaf(x0, cc, -x1 * ss);
                head[j + pairs] = fmaf(x0, ss, x1 * cc);
            }
        }

        template<typename KVT>
        __global__ void kStoreKVRopePos(const float *kSrc, const float *vSrc,
                                        KVT *kDst, KVT *vDst,
                                        uint32_t seqLen, uint32_t kHeads,
                                        uint32_t headDim, const uint32_t *pos,
                                        const float *cosT, const float *sinT) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            uint32_t p = *pos + s;
            uint32_t pairs = headDim / 2;
            uint32_t kvSize = kHeads * headDim;
            const float *ks = kSrc + static_cast<size_t>(s) * kvSize;
            const float *vs = vSrc + static_cast<size_t>(s) * kvSize;
            KVT *kd = kDst + static_cast<size_t>(p) * kvSize;
            KVT *vd = vDst + static_cast<size_t>(p) * kvSize;
            const float *c = cosT + static_cast<size_t>(p) * pairs;
            const float *sn = sinT + static_cast<size_t>(p) * pairs;
            uint32_t t = threadIdx.x;
            for (uint32_t e = t; e < kvSize; e += blockDim.x) {
                kvStore(vd, e, vs[e]);
            }
            const uint32_t numPairs = kHeads * pairs;
            for (uint32_t e = t; e < numPairs; e += blockDim.x) {
                uint32_t h = e / pairs;
                uint32_t j = e - h * pairs;
                uint32_t base = h * headDim + j;
                float k0 = ks[base], k1 = ks[base + pairs];
                kvStore(kd, base, fmaf(k0, c[j], -k1 * sn[j]));
                kvStore(kd, base + pairs, fmaf(k0, sn[j], k1 * c[j]));
            }
        }

        // Attention with the causal end (cachePos + s) read from a device
        // scalar; identical otherwise.
        template<uint32_t HD, typename KVT>
        __global__ void kWarpAttentionPos(const float *__restrict__ q,
                                          const KVT *__restrict__ kCache,
                                          const KVT *__restrict__ vCache,
                                          float *__restrict__ out,
                                          uint32_t seqLen, uint32_t nHeads,
                                          uint32_t nKVHeads,
                                          const uint32_t *cachePos,
                                          float invSqrtHeadDim) {
            static_assert(HD == 128 || HD == 64 || HD == 32, "HD must be 32/64/128");
            constexpr uint32_t NV = HD / 32;
            uint32_t s = blockIdx.x;
            uint32_t warpId = blockIdx.y * blockDim.y + threadIdx.y;
            uint32_t lane = threadIdx.x;
            if (s >= seqLen || warpId >= nHeads) return;
            uint32_t qHead = warpId;
            uint32_t kvHead = qHead / (nHeads / nKVHeads);
            uint32_t csEnd = *cachePos + s;

            const float *qPtr = q + (static_cast<size_t>(s) * nHeads + qHead) * HD;
            float acc[NV];
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
            float m = -1e30f, l = 0.0f;

            for (uint32_t cs = 0; cs <= csEnd; ++cs) {
                const KVT *kPtr =
                        kCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                const KVT *vPtr =
                        vCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                // score = dot(q, k) * invSqrt, warp-reduced
                float sc = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) {
                    sc = fmaf(qPtr[lane + i * 32], kvLoad(kPtr, lane + i * 32), sc);
                }
                sc = kWarpReduceSum(sc) * invSqrtHeadDim;

                if (sc > m) {
                    float mNew = sc;
                    float alpha = exp2f((m - mNew) * LOG2E);
                    if (alpha != 1.0f) {
#pragma unroll
                        for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
                    }
                    l *= alpha;
                    m = mNew;
                }
                float pp = exp2f((sc - m) * LOG2E);
                l += pp;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) {
                    acc[i] = fmaf(pp, kvLoad(vPtr, lane + i * 32), acc[i]);
                }
            }
            float invL = 1.0f / l;
            float *outPtr = out + (static_cast<size_t>(s) * nHeads + qHead) * HD;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) {
                outPtr[lane + i * 32] = acc[i] * invL;
            }
        }

        // ------------------------------------------------------------------
        // Split-KV decode attention (batch == 1 fast path).
        //
        // kWarpAttention launches ONE warp per (token, q-head): at batch==1 that
        // is nHeads (=12) warps for the whole 68-SM GPU, each scanning the FULL
        // cachePos range.  Decode attention is therefore latency-, not
        // bandwidth-bound, and its cost is a hard O(context) wall (measured
        // 0.76 ms @ pp16 -> 20.27 ms @ pp1024 over 28 layers = 81% of the layer
        // loop at 1K context, while the GEMVs stay flat at ~3.6 ms).
        //
        // The split path tiles the KV range into `chunkSize` windows and
        // launches nHeads * numChunks warps, so the SMs stay busy and the
        // critical path shrinks to O(chunkSize).  Each warp folds its window
        // into a flash-attention (m, l, acc) partial; kAttnCombinePartial
        // merges the partials with the standard rescale.  The math is the same
        // online softmax, only the reduction grouping differs, so top-k logit
        // identity is preserved.
        // ------------------------------------------------------------------
        // WSUB sub-warps cooperate on one (head, chunk): each warp scans a
        // contiguous chunkSize/WSUB span (exactly one 4-position load-batch
        // group at WSUB=4, chunkSize=16) and the block merges the WSUB
        // (m, l, acc) partials in shared memory with the same rescale formula
        // as kAttnCombinePartial, so the per-chunk partial layout and the
        // combine kernel are unchanged.  WSUB=1 is the original one-warp layout.
        template<uint32_t HD, typename KVT, uint32_t WSUB = 1>
        __global__ void kWarpAttentionSplit(const float *__restrict__ q,
                                            const KVT *__restrict__ kCache,
                                            const KVT *__restrict__ vCache,
                                            float *__restrict__ partAcc,
                                            float *__restrict__ partM,
                                            float *__restrict__ partL,
                                            uint32_t seqLen, uint32_t nHeads,
                                            uint32_t nKVHeads, uint32_t cachePos,
                                            uint32_t numChunks, uint32_t chunkSize,
                                            float invSqrtHeadDim) {
            static_assert(HD == 128 || HD == 64 || HD == 32, "HD must be 32/64/128");
            constexpr uint32_t NV = HD / 32;
            __shared__ float sAttnM[WSUB];
            __shared__ float sAttnL[WSUB];
            __shared__ float sAttnAcc[WSUB][HD];
            uint32_t s = blockIdx.x;
            uint32_t chunk = blockIdx.y;
            uint32_t lane = threadIdx.x;
            uint32_t qHead;
            if constexpr (WSUB == 1) {
                uint32_t warpId = blockIdx.z * blockDim.y + threadIdx.y;
                if (s >= seqLen || warpId >= nHeads) return;
                qHead = warpId;
            } else {
                if (s >= seqLen || blockIdx.z >= nHeads) return;
                qHead = blockIdx.z;
            }
            uint32_t kvHead = qHead / (nHeads / nKVHeads);
            uint32_t csEnd = cachePos + s;
            const float *qPtr = q + (static_cast<size_t>(s) * nHeads + qHead) * HD;
            const uint32_t pidx = (s * nHeads + qHead) * numChunks + chunk;
            float *pa = partAcc + static_cast<size_t>(pidx) * HD;
            float acc[NV];
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
            float m = -1e30f, l = 0.0f;
            uint32_t lo = chunk * chunkSize;
            uint32_t span = chunkSize;
            if constexpr (WSUB > 1) {
                span = chunkSize / WSUB;
                lo += threadIdx.y * span;
            }
            if (lo <= csEnd) {
                uint32_t hi = lo + span - 1;
                if (hi > csEnd) hi = csEnd;
                // 4-POSITION LOAD BATCHING (2026-10-10, ncu): the online-softmax
                // loop was a serial chain with a DRAM load at the head of every
                // step (load k -> dot -> reduce -> softmax -> load v -> fma), so
                // each warp kept ~1 request in flight -- measured SM 2.7 %,
                // DRAM 0.8 %, 17 us per launch.  Issuing a whole 4-position
                // group's K+V loads together puts 8 independent requests in
                // flight per lane.  The softmax ORDER is unchanged -> results
                // are bit-identical; only the load scheduling moves.
                for (uint32_t g0 = lo; g0 <= hi; g0 += 4u) {
                    KVT kR[4][NV];
                    KVT vR[4][NV];
#pragma unroll
                    for (uint32_t g = 0; g < 4; ++g) {
                        const uint32_t cs = g0 + g;
                        if (cs <= hi) {
                            const KVT *kPtr = kCache +
                                              (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                            const KVT *vPtr = vCache +
                                              (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
#pragma unroll
                            for (uint32_t i = 0; i < NV; ++i) {
                                kR[g][i] = kvLoad(kPtr, lane + i * 32);
                                vR[g][i] = kvLoad(vPtr, lane + i * 32);
                            }
                        }
                    }
#pragma unroll
                    for (uint32_t g = 0; g < 4; ++g) {
                        const uint32_t cs = g0 + g;
                        if (cs > hi) continue;
                        float sc = 0.0f;
#pragma unroll
                        for (uint32_t i = 0; i < NV; ++i) {
                            sc = fmaf(qPtr[lane + i * 32], kvScalar(kR[g][i]), sc);
                        }
                        sc = kWarpReduceSum(sc) * invSqrtHeadDim;
                        if (sc > m) {
                            float mNew = sc;
                            float alpha = exp2f((m - mNew) * LOG2E);
                            if (alpha != 1.0f) {
#pragma unroll
                                for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
                            }
                            l *= alpha;
                            m = mNew;
                        }
                        float pp = exp2f((sc - m) * LOG2E);
                        l += pp;
#pragma unroll
                        for (uint32_t i = 0; i < NV; ++i) {
                            acc[i] = fmaf(pp, kvScalar(vR[g][i]), acc[i]);
                        }
                    }
                }
            }
            if constexpr (WSUB == 1) {
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) pa[lane + i * 32] = acc[i];
                partM[pidx] = m;
                partL[pidx] = l;
            } else {
                // Intra-block merge of the WSUB sub-span partials: same
                // rescale formula as kAttnCombinePartial (running max M,
                // w = exp2((m - M)), acc and l summed with w).
                if (lane == 0) {
                    sAttnM[threadIdx.y] = m;
                    sAttnL[threadIdx.y] = l;
                }
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i)
                    sAttnAcc[threadIdx.y][lane + i * 32] = acc[i];
                __syncthreads();
                float M = -1e30f;
#pragma unroll
                for (uint32_t w = 0; w < WSUB; ++w) M = fmaxf(M, sAttnM[w]);
                float lsum = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
#pragma unroll
                for (uint32_t w = 0; w < WSUB; ++w) {
                    const float wgt = exp2f((sAttnM[w] - M) * LOG2E);
                    lsum += wgt * sAttnL[w];
#pragma unroll
                    for (uint32_t i = 0; i < NV; ++i) {
                        acc[i] = fmaf(wgt, sAttnAcc[w][lane + i * 32], acc[i]);
                    }
                }
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) pa[lane + i * 32] = acc[i];
                if (lane == 0) {
                    partM[pidx] = M;
                    partL[pidx] = lsum;
                }
            }
        }

        // Device-pos variant (CUDA-graph capture): cachePos read from *cachePos.
        // Pos variant of kWarpAttentionSplit (WSUB semantics identical).
        template<uint32_t HD, typename KVT, uint32_t WSUB = 1>
        __global__ void kWarpAttentionSplitPos(const float *__restrict__ q,
                                               const KVT *__restrict__ kCache,
                                               const KVT *__restrict__ vCache,
                                               float *__restrict__ partAcc,
                                               float *__restrict__ partM,
                                               float *__restrict__ partL,
                                               uint32_t seqLen, uint32_t nHeads,
                                               uint32_t nKVHeads,
                                               const uint32_t *cachePos,
                                               uint32_t numChunks, uint32_t chunkSize,
                                               float invSqrtHeadDim) {
            static_assert(HD == 128 || HD == 64 || HD == 32, "HD must be 32/64/128");
            constexpr uint32_t NV = HD / 32;
            __shared__ float sAttnM[WSUB];
            __shared__ float sAttnL[WSUB];
            __shared__ float sAttnAcc[WSUB][HD];
            uint32_t s = blockIdx.x;
            uint32_t chunk = blockIdx.y;
            uint32_t lane = threadIdx.x;
            uint32_t qHead;
            if constexpr (WSUB == 1) {
                uint32_t warpId = blockIdx.z * blockDim.y + threadIdx.y;
                if (s >= seqLen || warpId >= nHeads) return;
                qHead = warpId;
            } else {
                if (s >= seqLen || blockIdx.z >= nHeads) return;
                qHead = blockIdx.z;
            }
            uint32_t kvHead = qHead / (nHeads / nKVHeads);
            uint32_t csEnd = *cachePos + s;
            const float *qPtr = q + (static_cast<size_t>(s) * nHeads + qHead) * HD;
            const uint32_t pidx = (s * nHeads + qHead) * numChunks + chunk;
            float *pa = partAcc + static_cast<size_t>(pidx) * HD;
            float acc[NV];
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
            float m = -1e30f, l = 0.0f;
            uint32_t lo = chunk * chunkSize;
            uint32_t span = chunkSize;
            if constexpr (WSUB > 1) {
                span = chunkSize / WSUB;
                lo += threadIdx.y * span;
            }
            if (lo <= csEnd) {
                uint32_t hi = lo + span - 1;
                if (hi > csEnd) hi = csEnd;
                // 4-position load batching -- see kWarpAttentionSplit.
                for (uint32_t g0 = lo; g0 <= hi; g0 += 4u) {
                    KVT kR[4][NV];
                    KVT vR[4][NV];
#pragma unroll
                    for (uint32_t g = 0; g < 4; ++g) {
                        const uint32_t cs = g0 + g;
                        if (cs <= hi) {
                            const KVT *kPtr = kCache +
                                              (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                            const KVT *vPtr = vCache +
                                              (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
#pragma unroll
                            for (uint32_t i = 0; i < NV; ++i) {
                                kR[g][i] = kvLoad(kPtr, lane + i * 32);
                                vR[g][i] = kvLoad(vPtr, lane + i * 32);
                            }
                        }
                    }
#pragma unroll
                    for (uint32_t g = 0; g < 4; ++g) {
                        const uint32_t cs = g0 + g;
                        if (cs > hi) continue;
                        float sc = 0.0f;
#pragma unroll
                        for (uint32_t i = 0; i < NV; ++i) {
                            sc = fmaf(qPtr[lane + i * 32], kvScalar(kR[g][i]), sc);
                        }
                        sc = kWarpReduceSum(sc) * invSqrtHeadDim;
                        if (sc > m) {
                            float mNew = sc;
                            float alpha = exp2f((m - mNew) * LOG2E);
                            if (alpha != 1.0f) {
#pragma unroll
                                for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
                            }
                            l *= alpha;
                            m = mNew;
                        }
                        float pp = exp2f((sc - m) * LOG2E);
                        l += pp;
#pragma unroll
                        for (uint32_t i = 0; i < NV; ++i) {
                            acc[i] = fmaf(pp, kvScalar(vR[g][i]), acc[i]);
                        }
                    }
                }
            }
            if constexpr (WSUB == 1) {
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) pa[lane + i * 32] = acc[i];
                partM[pidx] = m;
                partL[pidx] = l;
            } else {
                // Intra-block merge of the WSUB sub-span partials -- see
                // kWarpAttentionSplit.
                if (lane == 0) {
                    sAttnM[threadIdx.y] = m;
                    sAttnL[threadIdx.y] = l;
                }
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i)
                    sAttnAcc[threadIdx.y][lane + i * 32] = acc[i];
                __syncthreads();
                float M = -1e30f;
#pragma unroll
                for (uint32_t w = 0; w < WSUB; ++w) M = fmaxf(M, sAttnM[w]);
                float lsum = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
#pragma unroll
                for (uint32_t w = 0; w < WSUB; ++w) {
                    const float wgt = exp2f((sAttnM[w] - M) * LOG2E);
                    lsum += wgt * sAttnL[w];
#pragma unroll
                    for (uint32_t i = 0; i < NV; ++i) {
                        acc[i] = fmaf(wgt, sAttnAcc[w][lane + i * 32], acc[i]);
                    }
                }
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) pa[lane + i * 32] = acc[i];
                if (lane == 0) {
                    partM[pidx] = M;
                    partL[pidx] = lsum;
                }
            }
        }

        // Merge the per-chunk (m, l, acc) partials of one (token, q-head).
        // One warp per (token, q-head); every lane recomputes the same running
        // max / rescale (broadcast loads) and writes its own HD/32 elements.
        template<uint32_t HD>
        __global__ void kAttnCombinePartial(const float *__restrict__ partAcc,
                                            const float *__restrict__ partM,
                                            const float *__restrict__ partL,
                                            float *__restrict__ out, uint32_t seqLen,
                                            uint32_t nHeads, uint32_t numChunks) {
            constexpr uint32_t NV = HD / 32;
            uint32_t s = blockIdx.x;
            uint32_t qHead = blockIdx.y;
            uint32_t lane = threadIdx.x;
            if (s >= seqLen || qHead >= nHeads) return;
            // Partial row index base = (token, q-head) * numChunks; chunk c of
            // this row lives at base + c (NOT s*nHeads+qHead + c).
            const uint64_t base =
                    (static_cast<uint64_t>(s) * nHeads + qHead) * numChunks;
            const float *pm = partM + base;
            const float *pl = partL + base;
            float M = -1e30f;
            for (uint32_t c = 0; c < numChunks; ++c) M = fmaxf(M, pm[c]);
            float acc[NV];
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
            float lsum = 0.0f;
            for (uint32_t c = 0; c < numChunks; ++c) {
                if (pl[c] <= 0.0f) continue;
                float w = exp2f((pm[c] - M) * LOG2E);
                lsum += w * pl[c];
                const float *pac = partAcc + (base + c) * HD;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) {
                    acc[i] = fmaf(w, pac[lane + i * 32], acc[i]);
                }
            }
            float inv = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
            float *outPtr = out + (static_cast<size_t>(s) * nHeads + qHead) * HD;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) outPtr[lane + i * 32] = acc[i] * inv;
        }

        // A/B gate for the 4-warp combine below (TINYCODER_ATTN_COMBINE4).
        // ADOPTED as the default (2026-10-10): interleaved A/B tg128@pp512,
        // 5 rounds: 257.8-261.3 -> 263.4-264.4 tok/s, +1.05..+2.42 %, 5/5
        // wins.  TINYCODER_ATTN_COMBINE4=0 restores the 32-thread combine.
        inline bool attnCombine4() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_ATTN_COMBINE4");
                return (e == nullptr || std::atoi(e) != 0);
            }();
            return v;
        }
        // Producer-epilogue Q8_1 fold (2026-10-10, plan section 9.26 lever 1):
        // the combine W4 and the FFN rms_norm emit the Q8_1 activation blocks
        // their downstream dp4a mmvq consumers (attnO / GLU) need, deleting the
        // two 48-block kQuantizeQ8_1 launches (2.5 us each, ~5 us/layer).  The
        // quantize math is identical to kQuantizeQ8_1 over the same float
        // values, so the Q8_1 bytes -- and therefore the GEMV outputs -- are
        // bit-identical.  TINYCODER_FUSE_Q81_PROD=0 restores the separate
        // launches.
        inline bool fuseQ81Prod() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_FUSE_Q81_PROD");
                return (e == nullptr || std::atoi(e) != 0);
            }();
            return v;
        }
        inline bool qkvDp4a() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_QKV_DP4A");
                return (e == nullptr || std::atoi(e) != 0);
            }();
            return v;
        }
        // Combine warp-count switch (2026-10-10, plan section 9.27 lever 3):
        // NW=8 (256-thread blocks, 3072 threads across 12 blocks) when
        // numChunks >= 64; `0` pins the 9.24 four-warp layout.
        inline bool combineW8() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_ATTN_COMBINE_W8");
                return (e == nullptr || std::atoi(e) != 0);
            }();
            return v;
        }

        // NW-warp parallel chunk merge (2026-10-10, plan section 9.24/9.27):
        // the original combine runs 12 blocks x 32 threads (384 threads on 68
        // SMs), each thread serially folding numChunks=128 partials with
        // HD-strided loads -- 14.5 us at DRAM 0.4 %, pure latency.  Each warp
        // now folds every NWth chunk into its own (m, l, acc) partial and the
        // NW partials are merged flash-style in shared memory.  NW=4 was the
        // 9.24 adoption; NW=8 (256 threads, 3072 threads across 12 blocks)
        // halves the per-warp serial fold again when numChunks >= 64
        // (TINYCODER_ATTN_COMBINE_W8=0 pins NW=4).  The fold ORDER changes
        // (same tolerance class as the split-KV attention itself); verified by
        // the 4-prompt stream + top-10 logits comparison.
        template<uint32_t HDV, uint32_t NW>
        __global__ void kAttnCombinePartialW4(const float *__restrict__ partAcc,
                                              const float *__restrict__ partM,
                                              const float *__restrict__ partL,
                                              float *__restrict__ out,
                                              uint32_t seqLen, uint32_t nHeads,
                                              uint32_t numChunks,
                                              uint8_t *__restrict__ q81) {
            constexpr uint32_t NV = HDV / 32;
            const uint32_t s = blockIdx.x;
            const uint32_t qHead = blockIdx.y;
            const uint32_t warp = threadIdx.x >> 5;// 0..NW-1, block = NW*32 threads
            const uint32_t lane = threadIdx.x & 31;
            if (s >= seqLen || qHead >= nHeads || warp >= NW) return;
            const uint64_t base =
                    (static_cast<uint64_t>(s) * nHeads + qHead) * numChunks;
            // Per-warp local fold over chunks warp, warp+NW, warp+2*NW, ...
            float M = -1e30f;
            for (uint32_t c = warp; c < numChunks; c += NW)
                M = fmaxf(M, partM[base + c]);
            float acc[NV];
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
            float lsum = 0.0f;
            for (uint32_t c = warp; c < numChunks; c += NW) {
                const float lc = partL[base + c];
                if (lc <= 0.0f) continue;
                const float w = exp2f((partM[base + c] - M) * LOG2E);
                lsum += w * lc;
                const float *pac = partAcc + (base + c) * HDV;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i)
                    acc[i] = fmaf(w, pac[lane + i * 32], acc[i]);
            }
            // Flash-style NW-partial merge in shared memory.  sacc is indexed
            // [warp][lane][i]: acc[i] is PER-LANE (dims lane + i*32), so the
            // partial write MUST keep the lane dimension -- a [warp][i] array
            // would race all 32 lanes onto one address (the original bug).
            __shared__ float sm[NW], sl[NW], sacc[NW][32][NV];
            sm[warp] = M;
            sl[warp] = lsum;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) sacc[warp][lane][i] = acc[i];
            __syncthreads();
            if (warp == 0) {
                float gM = -1e30f;
#pragma unroll
                for (uint32_t w2 = 0; w2 < NW; ++w2) gM = fmaxf(gM, sm[w2]);
                float gL = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
#pragma unroll
                for (uint32_t w2 = 0; w2 < NW; ++w2) {
                    const float w = exp2f((sm[w2] - gM) * LOG2E);
                    gL += w * sl[w2];
#pragma unroll
                    for (uint32_t i = 0; i < NV; ++i)
                        acc[i] = fmaf(w, sacc[w2][lane][i], acc[i]);
                }
                const float inv = (gL > 0.0f) ? (1.0f / gL) : 0.0f;
                float *outPtr =
                        out + (static_cast<size_t>(s) * nHeads + qHead) * HDV;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i)
                    outPtr[lane + i * 32] = acc[i] * inv;
                // Q8_1 fold: NV blocks of 32 consecutive outputs; block i is
                // exactly {acc[i] over lanes} (element lane + i*32).  Same
                // math as kQuantizeQ8_1 -> bit-identical bytes.
                if (q81 != nullptr) {
                    const uint64_t blkBase =
                            (static_cast<uint64_t>(s) * nHeads + qHead) * NV;
#pragma unroll
                    for (uint32_t i = 0; i < NV; ++i) {
                        const float v = acc[i] * inv;
                        float amax = fabsf(v), sum = v;
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1) {
                            amax = fmaxf(amax,
                                         __shfl_xor_sync(0xffffffffu, amax, o));
                            sum += __shfl_xor_sync(0xffffffffu, sum, o);
                        }
                        const float d = amax / 127.0f;
                        uint8_t *blk = q81 + (blkBase + i) * 40u;
                        reinterpret_cast<int8_t *>(blk + 4)[lane] =
                                (amax == 0.0f)
                                        ? static_cast<int8_t>(0)
                                        : static_cast<int8_t>(roundf(v / d));
                        if (lane == 0) {
                            __half *ds = reinterpret_cast<__half *>(blk);
                            ds[0] = __float2half(d);
                            ds[1] = __float2half(sum);
                        }
                    }
                }
            }
        }

        // Common launch path for both kernel variants (eager pos vs device
        // pos) at a fixed WSUB.
        template<uint32_t HD, typename KVT, uint32_t WSUB>
        void launchSplitVariant(const float *q, const KVT *kK, const KVT *kV,
                                float *pAcc, float *pM, float *pL,
                                uint32_t seqLen, uint32_t nHeads, uint32_t nKV,
                                uint32_t pos, bool devicePos,
                                const uint32_t *posDev, uint32_t numChunks,
                                uint32_t chunkSize, float invSqrt) {
            dim3 grid(seqLen, numChunks, WSUB > 1 ? nHeads : (nHeads + 3) / 4);
            dim3 blk(32, WSUB > 1 ? WSUB : 4);
            if (devicePos)
                kWarpAttentionSplitPos<HD, KVT, WSUB><<<grid, blk, 0, g_stream>>>(
                        q, kK, kV, pAcc, pM, pL, seqLen, nHeads, nKV, posDev,
                        numChunks, chunkSize, invSqrt);
            else
                kWarpAttentionSplit<HD, KVT, WSUB><<<grid, blk, 0, g_stream>>>(
                        q, kK, kV, pAcc, pM, pL, seqLen, nHeads, nKV, pos,
                        numChunks, chunkSize, invSqrt);
        }

        // Host dispatcher for the split-KV decode attention.  `devicePos`
        // selects the CUDA-graph (device scalar) kernel variant; the combine is
        // position-independent.  Only called for seqLen == 1.
        //
        // TINYCODER_ATTN_SUBWARP=2/4 splits each (head, chunk) across that
        // many warps (contiguous sub-spans of the chunk, merged in shared
        // memory), multiplying the active-warp count without growing the
        // partial layout or the combine.  0/1 keeps the one-warp layout.
        void launchDecodeAttentionSplit(const float *q, const void *kvK,
                                        const void *kvV, float *attnOut,
                                        float *pAcc, float *pM, float *pL,
                                        uint32_t seqLen, uint32_t nHeads,
                                        uint32_t nKV, uint32_t hd, uint32_t pos,
                                        bool devicePos, const uint32_t *posDev,
                                        uint32_t numChunks, uint32_t chunkSize,
                                        float invSqrt, bool kvHalf,
                                        uint8_t *q81Out) {
            static const uint32_t subWarpEnv = []() -> uint32_t {
                if (const char *e = std::getenv("TINYCODER_ATTN_SUBWARP")) {
                    const int v = std::atoi(e);
                    if (v == 2 || v == 4) return static_cast<uint32_t>(v);
                    return 1u;// 0/1 = legacy one-warp layout
                }
                return 2u;// measured default: 8/8 interleaved pair wins, +0.35 %
            }();
            const bool useSub = subWarpEnv > 1 && (chunkSize % subWarpEnv) == 0;
            const uint32_t wsub = useSub ? subWarpEnv : 1u;
#define TC_ATTN_SPLIT(HDV, KVT)                                                                     \
    do {                                                                                            \
        const KVT *kK = static_cast<const KVT *>(kvK);                                              \
        const KVT *kV = static_cast<const KVT *>(kvV);                                              \
        if (wsub == 4)                                                                              \
            launchSplitVariant<HDV, KVT, 4>(q, kK, kV, pAcc, pM, pL,                                \
                                            seqLen, nHeads, nKV, pos, devicePos, posDev, numChunks, \
                                            chunkSize, invSqrt);                                    \
        else if (wsub == 2)                                                                         \
            launchSplitVariant<HDV, KVT, 2>(q, kK, kV, pAcc, pM, pL,                                \
                                            seqLen, nHeads, nKV, pos, devicePos, posDev, numChunks, \
                                            chunkSize, invSqrt);                                    \
        else                                                                                        \
            launchSplitVariant<HDV, KVT, 1>(q, kK, kV, pAcc, pM, pL,                                \
                                            seqLen, nHeads, nKV, pos, devicePos, posDev, numChunks, \
                                            chunkSize, invSqrt);                                    \
    } while (0)
#define TC_DISPATCH(HDV)                        \
    do {                                        \
        if (kvHalf) TC_ATTN_SPLIT(HDV, __half); \
        else                                    \
            TC_ATTN_SPLIT(HDV, float);          \
    } while (0)
            if (hd == 128) TC_DISPATCH(128);
            else if (hd == 64)
                TC_DISPATCH(64);
            else
                TC_DISPATCH(32);
#undef TC_DISPATCH
#undef TC_ATTN_SPLIT
            dim3 cgrid(seqLen, nHeads);
            const bool combine4 = attnCombine4();
            const uint32_t nw =
                    (combine4 && combineW8() && (numChunks >= 64u)) ? 8u : 4u;
            if (hd == 128) {
                if (nw == 8)
                    kAttnCombinePartialW4<128, 8><<<cgrid, 256, 0, g_stream>>>(
                            pAcc, pM, pL, attnOut, seqLen, nHeads, numChunks,
                            q81Out);
                else if (combine4)
                    kAttnCombinePartialW4<128, 4><<<cgrid, 128, 0, g_stream>>>(
                            pAcc, pM, pL, attnOut, seqLen, nHeads, numChunks,
                            q81Out);
                else
                    kAttnCombinePartial<128><<<cgrid, 32, 0, g_stream>>>(
                            pAcc, pM, pL, attnOut, seqLen, nHeads, numChunks);
            } else if (hd == 64) {
                if (nw == 8)
                    kAttnCombinePartialW4<64, 8><<<cgrid, 256, 0, g_stream>>>(
                            pAcc, pM, pL, attnOut, seqLen, nHeads, numChunks,
                            q81Out);
                else if (combine4)
                    kAttnCombinePartialW4<64, 4><<<cgrid, 128, 0, g_stream>>>(
                            pAcc, pM, pL, attnOut, seqLen, nHeads, numChunks,
                            q81Out);
                else
                    kAttnCombinePartial<64><<<cgrid, 32, 0, g_stream>>>(
                            pAcc, pM, pL, attnOut, seqLen, nHeads, numChunks);
            } else {
                if (nw == 8)
                    kAttnCombinePartialW4<32, 8><<<cgrid, 256, 0, g_stream>>>(
                            pAcc, pM, pL, attnOut, seqLen, nHeads, numChunks,
                            q81Out);
                else if (combine4)
                    kAttnCombinePartialW4<32, 4><<<cgrid, 128, 0, g_stream>>>(
                            pAcc, pM, pL, attnOut, seqLen, nHeads, numChunks,
                            q81Out);
                else
                    kAttnCombinePartial<32><<<cgrid, 32, 0, g_stream>>>(
                            pAcc, pM, pL, attnOut, seqLen, nHeads, numChunks);
            }
        }

        // ------------------------------------------------------------------
        // Quantized GEMV: y[rows] = x[cols] @ W^T, warp-per-row.
        //
        // x is staged in shared memory once per block by all 256 threads.
        // Each warp owns one output row.  Each lane of the warp owns elements
        // [lane*8, lane*8+8) of each 256-element block and precomputes the
        // exact (scale, min, byte-base, shift) indices matching
        // GGMLDequantize::dequantize{Q2_K,Q3_K,Q4_K}Block so the dequant loop
        // is branch-free and bit-identical.
        // ------------------------------------------------------------------

        // cols must be a multiple of 32 (hiddenSize = 1536, intermediateSize =
        // 8960, headDim*heads = 1536 are; the launcher pads cols up to the next
        // 32 when they are not).  No shared memory is used: x is read directly
        // from global (coalesced, and tiny enough to stay L1/L2-resident).
        // P1 exp2 (2026-10-06): EXACT=true drops the per-element column
        // guard `(cidx < cols)` when cols is an exact multiple of the
        // 256-wide K-quant block (blocksPerRow*256 == cols).  That holds for
        // every default-model matrix (H=1536, I=8960, KV=256), and the guard
        // is provably dead there, so the arithmetic is BIT-IDENTICAL to the
        // guarded path while shedding ~2 SASS ops per weight (the ISETP +
        // predicated-zero select) -- the exact "reduce per-byte ALU" lever
        // the P0 profile demanded.  The launcher only selects EXACT=true
        // after satisfying the multiple-of-256 test, so non-conforming
        // models keep the guarded instantiation.
        template<int TYPE, bool EXACT = false>
        __global__ void kQGemv(const uint8_t *__restrict__ w,
                               const float *__restrict__ x, float *__restrict__ out,
                               uint32_t rows, uint32_t cols, uint32_t rowBytes,
                               uint32_t blocksPerRow) {
            // x is read DIRECTLY from global (coalesced, and tiny enough to
            // stay L1/L2-resident); no shared-memory staging/barrier needed.
            // cols is padded up to a multiple of 32 (blocksPerRow*256 for the
            // K-quants); the final partial block's pad columns [cols, blk*256)
            // MUST contribute 0 (the CPU's SIMD kernels treat them as zero
            // dequant) and MUST NOT be read from x: the scratch x buffers are
            // allocated to the true H/I size (e.g. 896, 4864), so an unguarded
            // read at idx >= cols reads past the buffer into stale device
            // memory (prefill looked fine only because fresh cudaMalloc pages
            // are zero).  The helper masks those lanes to 0.
            uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;

            float acc = 0.0f;
            const uint32_t lane = threadIdx.x;
            const uint32_t e0 = lane * 8u;

            if (TYPE == kTypeQ2K) {
                // Coalesced byte scheme (bit-exact with dequantizeQ2_KBlock):
                // read x DIRECTLY from global (skip the shared-memory staging,
                // which added latency and occupancy pressure).  Reads of x are
                // coalesced (lanes read consecutive floats within each 32-group)
                // and x is tiny (cols <= 1536 floats), so it stays L1/L2-hot
                // across all rows.
                const uint32_t sub = lane / 16u;
                // 4-way block batching (mirrors kQGemv<kTypeIQ2XS>): 4
                // independent (load, dequant, fmaf-chain) streams hide the
                // serial inter-block dependency that kept the Q2_K gate+up at
                // ~23.4 us/layer (~5x the GB/s floor).  Each block's fmaf
                // chain is untouched and the fold (acc += a0; a1; a2; a3) is
                // in block order -- bit-exact with the serial loop.
                uint32_t b = 0;
                for (; b + 4u <= blocksPerRow; b += 4u) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                    for (uint32_t bb = 0; bb < 4; ++bb) {
                        const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                        __half d = *reinterpret_cast<const __half *>(crp + 80);
                        __half dmin = *reinterpret_cast<const __half *>(crp + 82);
                        float df = __half2float(d), dmf = __half2float(dmin);
                        const uint8_t *sc = crp;
                        const uint8_t *q = crp + 16;
                        float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                : (bb == 2u)       ? a2
                                                                   : a3;
#pragma unroll
                        for (uint32_t half = 0; half < 2; ++half) {
                            const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                            for (uint32_t jj = 0; jj < 4; ++jj) {
                                const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                                float dl = df * static_cast<float>(scv & 0xF);
                                float ml = dmf * static_cast<float>(scv >> 4);
                                float qv = static_cast<float>(
                                        static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                                const uint32_t cidx =
                                        (b + bb) * 256u + half * 128u + jj * 32u + lane;
                                slot = fmaf(fmaf(dl, qv, -ml),
                                            EXACT ? x[cidx]
                                                  : ((cidx < cols) ? x[cidx]
                                                                   : 0.0f),
                                            slot);
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
                    __half d = *reinterpret_cast<const __half *>(rp + 80);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 82);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *sc = rp;
                    const uint8_t *q = rp + 16;
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                            float dl = df * static_cast<float>(scv & 0xF);
                            float ml = dmf * static_cast<float>(scv >> 4);
                            float qv = static_cast<float>(
                                    static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                            const uint32_t cidx =
                                    b * 256u + half * 128u + jj * 32u + lane;
                            acc = fmaf(fmaf(dl, qv, -ml),
                                       EXACT ? x[cidx]
                                             : ((cidx < cols) ? x[cidx] : 0.0f),
                                       acc);
                        }
                    }
                    rp += kQ2K_BYTES;
                }
            } else if (TYPE == kTypeQ3K) {
                // Same coalesced-byte scheme as Q2_K.  hm (32 bytes, block+0)
                // holds the high-bit mask: bit 1<<jj for the same (sub,l)
                // position => hm[lane] with the 1<<jj mask.  qv = (qb>>2jj&3) -
                // (hm[lane]&(1<<jj) ? 0 : 4).  Scale is the repacked int8
                // sc16[half*8+jj*2+sub].  No dmin (Q3_K has none).
                const uint32_t sub = lane / 16u;
                // 4-way block batching (same rationale as the Q2_K branch):
                // independent (load, dequant, fmaf-chain) streams hide the
                // serial inter-block dependency; in-order fold keeps the
                // result bit-identical with the serial loop.
                uint32_t b = 0;
                for (; b + 4u <= blocksPerRow; b += 4u) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                    for (uint32_t bb = 0; bb < 4; ++bb) {
                        const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ3K_BYTES;
                        __half d = *reinterpret_cast<const __half *>(crp + 108);
                        float df = __half2float(d);
                        const uint8_t *hm = crp;
                        const uint8_t *q = crp + 32;
                        const uint8_t *scales = crp + 96;
                        uint32_t aux[4];
                        std::memcpy(&aux[0], scales, 12);
                        uint32_t tmp = aux[2];
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
                        const uint8_t hmb = hm[lane];
                        float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                : (bb == 2u)       ? a2
                                                                   : a3;
#pragma unroll
                        for (uint32_t half = 0; half < 2; ++half) {
                            const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                            for (uint32_t jj = 0; jj < 4; ++jj) {
                                float dl = df * static_cast<float>(
                                                        sc16[half * 8u + jj * 2u + sub] - 32);
                                // the hmask bit advances ACROSS halves too:
                                // half0 uses bits 0..3, half1 bits 4..7.
                                const uint32_t maskBit = 1u << (half * 4u + jj);
                                float qv = static_cast<float>(static_cast<int8_t>(
                                        ((qb >> (jj * 2u)) & 3) -
                                        ((hmb & maskBit) ? 0 : 4)));
                                const uint32_t cidx =
                                        (b + bb) * 256u + half * 128u + jj * 32u + lane;
                                slot = fmaf(dl * qv,
                                            EXACT ? x[cidx]
                                                  : ((cidx < cols) ? x[cidx]
                                                                   : 0.0f),
                                            slot);
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
                    __half d = *reinterpret_cast<const __half *>(rp + 108);
                    float df = __half2float(d);
                    const uint8_t *hm = rp;
                    const uint8_t *q = rp + 32;
                    const uint8_t *scales = rp + 96;
                    uint32_t aux[4];
                    std::memcpy(&aux[0], scales, 12);
                    uint32_t tmp = aux[2];
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
                    const uint8_t hmb = hm[lane];
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            float dl = df * static_cast<float>(
                                                    sc16[half * 8u + jj * 2u + sub] - 32);
                            // the hmask bit advances ACROSS halves too:
                            // half0 uses bits 0..3, half1 bits 4..7.
                            const uint32_t maskBit = 1u << (half * 4u + jj);
                            float qv = static_cast<float>(static_cast<int8_t>(
                                    ((qb >> (jj * 2u)) & 3) -
                                    ((hmb & maskBit) ? 0 : 4)));
                            const uint32_t cidx =
                                    b * 256u + half * 128u + jj * 32u + lane;
                            acc = fmaf(dl * qv,
                                       EXACT ? x[cidx]
                                             : ((cidx < cols) ? x[cidx] : 0.0f),
                                       acc);
                        }
                    }
                    rp += kQ3K_BYTES;
                }
            } else if (TYPE == kTypeQ4K) {
                // Coalesced byte scheme (bit-exact with dequantizeQ4_KBlock):
                // each 64-weight group g reads 32 q bytes.  Byte
                // qs[g*32 + lane] supplies out[g*64 + lane] from its LOW
                // nibble (sub-scale is+0) and out[g*64 + 32 + lane] from its
                // HIGH nibble (sub-scale is+1).  32 lanes x 2 nibbles = 64
                // weights per group, 4 groups = 256 per block.
                auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                      uint8_t *m_out) {
                    if (j < 4) {
                        *d_out = q[j] & 63;
                        *m_out = q[j + 4] & 63;
                    } else {
                        *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                        *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                    }
                };
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 2);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *scales = rp + 4;
                    const uint8_t *qs = rp + 16;
#pragma unroll
                    for (uint32_t g = 0; g < 4; ++g) {
                        const uint8_t qb = qs[g * 32u + lane];
                        uint8_t sc0, mm0, sc1, mm1;
                        getScaleMin(static_cast<int>(g * 2 + 0), scales, &sc0,
                                    &mm0);
                        getScaleMin(static_cast<int>(g * 2 + 1), scales, &sc1,
                                    &mm1);
                        float dl0 = df * static_cast<float>(sc0);
                        float ml0 = dmf * static_cast<float>(mm0);
                        float dl1 = df * static_cast<float>(sc1);
                        float ml1 = dmf * static_cast<float>(mm1);
                        float qv0 = static_cast<float>(qb & 0xF);
                        float qv1 = static_cast<float>(qb >> 4);
                        const uint32_t c0 = b * 256u + g * 64u + lane;
                        const uint32_t c1 = c0 + 32u;
                        acc = fmaf(fmaf(dl0, qv0, -ml0),
                                   (c0 < cols) ? x[c0] : 0.0f, acc);
                        acc = fmaf(fmaf(dl1, qv1, -ml1),
                                   (c1 < cols) ? x[c1] : 0.0f, acc);
                    }
                    rp += kQ4K_BYTES;
                }
            } else if (TYPE == kTypeQ6K) {
                // Q6_K block layout (210 B): ql(128) + qh(64) + sc(16) + d(2).
                // Each 128-weight half: qh[l] holds 2-bit fields for weights at
                // l, l+32, l+64, l+96; q1/q3 use ql[l] (low/high nibble), q2/q4
                // use ql[32+l]; 8 int8 scales, one per 16-weight group:
                //   sc[half*8 + sub*2 + (l/16)], value = d*scale*(q-32).
                const uint32_t half = e0 / 128u;
                const uint32_t sub = (e0 % 128u) / 32u;
                const uint32_t l = e0 % 32u;
                const uint32_t g16 = l / 16u;
                const uint32_t qlBase = half * 64u + ((sub & 1u) ? 32u : 0u);
                const uint32_t qhBase = half * 32u;
                const uint32_t shift = sub * 2u;            // 0,2,4,6
                const uint32_t hiNib = (sub & 2u) ? 4u : 0u;// q3/q4 use high nibble
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    const uint8_t *rb = rp;
                    const uint8_t *ql = rb;
                    const uint8_t *qh = rb + 128;
                    const int8_t *sc = reinterpret_cast<const int8_t *>(rb + 192);
                    __half d = *reinterpret_cast<const __half *>(rb + 208);
                    float dl = __half2float(d) *
                               static_cast<float>(sc[half * 8u + sub * 2u + g16]);
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t k = 0; k < 8; ++k) {
                        uint32_t qb = (hiNib) ? (ql[qlBase + l + k] >> 4)
                                              : (ql[qlBase + l + k] & 0xF);
                        qb |= ((qh[qhBase + l + k] >> shift) & 3u) << 4u;
                        float qv = static_cast<float>(static_cast<int>(qb) - 32);
                        const uint32_t ck = b * 256u + e0 + k;
                        acc = fmaf(dl * qv, (ck < cols) ? xs[k] : 0.0f, acc);
                    }
                    rp += kQ6K_BYTES;
                }
            } else if (TYPE == kTypeIQ2XS) {
                // IQ2_XS block (74 B): d(2) + qs[64] (32 x uint16) +
                // scales[8].  Each uint16 holds a 9-bit grid index + 7-bit
                // sign index.  Each lane owns 8 consecutive columns = exactly
                // one (ib32, l) group's 8 weights (e0 is a multiple of 8), so
                // one (gridIdx, signIdx) lookup feeds all 8 of its columns.
                //
                // BLOCK BATCHING (2026-09-27): the per-lane float acc is
                // INDEPENDENT across blocks (the final 5-shuffle tree sums
                // lanes AFTER the whole loop), so processing 4 blocks into
                // separate accumulators a0..a3 and folding them at the end is
                // bit-exact -- the fmaf chain inside each block is unchanged,
                // and float addition is associative only in order, which we
                // preserve by folding a0+a1+a2+a3 into acc serially.  The
                // motive: 4 independent (load, decode, fmaf-chain) streams hide
                // the inter-block dependency that kept the 7B IQ2_S gate+up at
                // ~215 us/call (~5x the ~44 us GB/s floor).
                uint32_t b = 0;
                for (; b + 4u <= blocksPerRow; b += 4u) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                    for (uint32_t bb = 0; bb < 4; ++bb) {
                        const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kIQ2XS_BYTES;
                        __half d = *reinterpret_cast<const __half *>(crp + 0);
                        float df = __half2float(d);
                        const uint16_t *qs16 =
                                reinterpret_cast<const uint16_t *>(crp + 2);
                        const uint8_t *scales = crp + 66;
                        const uint32_t ib32 = e0 >> 5u;    // sub-block 0..7
                        const uint32_t l = (e0 >> 3u) & 3u;// 0..3
                        float db0 = df * (0.5f + static_cast<float>(scales[ib32] & 0xf)) *
                                    0.25f;
                        float db1 = df * (0.5f + static_cast<float>(scales[ib32] >> 4)) *
                                    0.25f;
                        // CPU reference: dl = db[l/2] (l=0,1 -> db0; l=2,3 -> db1).
                        float dl = (l >= 2u) ? db1 : db0;
                        uint16_t qval = qs16[ib32 * 4u + l];
                        uint16_t gridIdx = qval & 0x1FFu;
                        uint8_t signIdx = static_cast<uint8_t>(qval >> 9);
                        const uint64_t gv = c_iq2xs_grid[gridIdx];
                        uint8_t signs = c_ksigns_iq2xs[signIdx];
                        const float *xs = x + (b + bb) * 256u + e0;
                        float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                : (bb == 2u)       ? a2
                                                                   : a3;
#pragma unroll
                        for (uint32_t j = 0; j < 8; ++j) {
                            float w =
                                    dl * static_cast<float>((gv >> (8u * j)) & 0xFFu) *
                                    ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                            const uint32_t ck = (b + bb) * 256u + e0 + j;
                            slot = fmaf(w, (ck < cols) ? xs[j] : 0.0f, slot);
                        }
                    }
                    acc += a0;
                    acc += a1;
                    acc += a2;
                    acc += a3;
                    rp += 4u * kIQ2XS_BYTES;
                }
                for (; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    const uint16_t *qs16 =
                            reinterpret_cast<const uint16_t *>(rp + 2);
                    const uint8_t *scales = rp + 66;
                    const uint32_t ib32 = e0 >> 5u;    // sub-block 0..7
                    const uint32_t l = (e0 >> 3u) & 3u;// 0..3
                    float db0 = df * (0.5f + static_cast<float>(scales[ib32] & 0xf)) *
                                0.25f;
                    float db1 = df * (0.5f + static_cast<float>(scales[ib32] >> 4)) *
                                0.25f;
                    float dl = (l >= 2u) ? db1 : db0;
                    uint16_t qval = qs16[ib32 * 4u + l];
                    uint16_t gridIdx = qval & 0x1FFu;
                    uint8_t signIdx = static_cast<uint8_t>(qval >> 9);
                    const uint64_t gv = c_iq2xs_grid[gridIdx];
                    uint8_t signs = c_ksigns_iq2xs[signIdx];
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        float w =
                                dl * static_cast<float>((gv >> (8u * j)) & 0xFFu) *
                                ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t ck = b * 256u + e0 + j;
                        acc = fmaf(w, (ck < cols) ? xs[j] : 0.0f, acc);
                    }
                    rp += kIQ2XS_BYTES;
                }
            } else if (TYPE == kTypeIQ2XXS) {
                // IQ2_XXS block (66 B): d(2) + qs[32 x uint16] (64 B).
                // Per (ib32, l) group (ib32 = sub-block 0..7, l = 0..3) the
                // lane reads 8 bytes = qs[4*ib32 .. 4*ib32+7] as two LE
                // uint32 words.  Bytes 0..3 (word0) hold the four grid
                // indices aux8[l] into iq2xxs_grid[256]; word1 bits 7*l..7*l+6
                // hold the sign index and word1 bits 28..31 the scale:
                //   db = d*(0.5 + (word1>>28))*0.25
                //   w  = db * grid[j] * (signs & kmask[j] ? -1 : 1)
                // (bit-identical to GGMLDequantize::dequantizeIQ2_XXSBlock).
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    const uint8_t *qsv = rp + 2;
                    const uint32_t ib32 = e0 >> 5u;
                    const uint32_t l = (e0 >> 3u) & 3u;
                    uint32_t w0, w1;
                    std::memcpy(&w0, qsv + 4u * ib32, sizeof(uint32_t));
                    std::memcpy(&w1, qsv + 4u * ib32 + 4u, sizeof(uint32_t));
                    float db =
                            df * (0.5f + static_cast<float>(w1 >> 28)) * 0.25f;
                    const uint32_t gridIdx = (w0 >> (8u * l)) & 0xFFu;
                    const uint8_t signs =
                            c_ksigns_iq2xs[(w1 >> (7u * l)) & 127u];
                    // Native-word load + byte-extract (same fix as IQ2XS).
                    const uint64_t gv = c_iq2xxs_grid[gridIdx];
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        float wgt = db *
                                    static_cast<float>((gv >> (8u * j)) & 0xFFu) *
                                    ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t ck = b * 256u + e0 + j;
                        acc = fmaf(wgt, (ck < cols) ? xs[j] : 0.0f, acc);
                    }
                    rp += kIQ2XXS_BYTES;
                }
            } else if (TYPE == kTypeIQ4XS) {
                // IQ4_XS block (136 B): d(2) + scales_h(2) + scales_l(4) +
                // qs[128].  Per 32-weight sub-block ib:
                //   ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf)
                //        | (((scales_h >> 2*ib) & 3) << 4)
                //   dl = d*(ls - 32); qs[ib*16 + j] byte -> col ib*32+j (low
                //   nibble) and col ib*32+16+j (high nibble) of kvalues_iq4nl.
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    uint16_t scales_h;
                    std::memcpy(&scales_h, rp + 2, sizeof(uint16_t));
                    const uint8_t *scales_l = rp + 4;
                    const uint8_t *qsv = rp + 8;
                    const uint32_t ib = e0 >> 5u;      // sub-block 0..7
                    const uint32_t l = (e0 >> 3u) & 3u;// 4 bytes -> 8 weights
                    const int ls =
                            static_cast<int>((scales_l[ib / 2u] >>
                                              (4u * (ib % 2u))) &
                                             0xfu) |
                            (static_cast<int>((scales_h >> (2u * ib)) & 0x3u)
                             << 4);
                    float dl = df * static_cast<float>(ls - 32);
                    const uint8_t *qb = qsv + ib * 16u + l * 4u;
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        const uint32_t jc = l * 8u + j;// col within sub-block
                        const uint32_t byteIdx = jc & 15u;
                        const uint32_t nib = (jc < 16u)
                                                     ? (qb[byteIdx] & 0xfu)
                                                     : (qb[byteIdx] >> 4);
                        float wgt = dl * static_cast<float>(c_kvalues_iq4nl[nib]);
                        const uint32_t ck = b * 256u + e0 + j;
                        acc = fmaf(wgt, (ck < cols) ? xs[j] : 0.0f, acc);
                    }
                    rp += kIQ4XS_BYTES;
                }
            } else if (TYPE == kTypeIQ2S) {
                // IQ2_S block (82 B): d(2) + qs[64] + qh[8] + scales[8].
                // Per (ib32, l) group (ib32 = sub-block 0..7, l = 0..3):
                //   dl  = db[l/2] where db0 = d*(0.5+(scales[ib32]&0xf))*0.25,
                //                     db1 = d*(0.5+(scales[ib32]>>4))*0.25
                //   gridIdx = qs[ib32*4 + l] | (qh[ib32] << (8-2l) & 0x300)
                //   signs   = qs[32 + ib32*4 + l]   (the sign byte lives in the
                //             qs tail -- CPU: signs = qs + QK_K/8)
                // The lane owns 8 consecutive columns => exactly one (ib32,l).
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    const uint8_t *qsv = rp + 2;    // 64 q bytes (grid idx + signs)
                    const uint8_t *qh = rp + 66;    // 8 high bits
                    const uint8_t *scales = rp + 74;// 8 scales
                    const uint32_t ib32 = e0 >> 5u;
                    const uint32_t l = (e0 >> 3u) & 3u;
                    float db0 = df *
                                (0.5f + static_cast<float>(scales[ib32] & 0xf)) *
                                0.25f;
                    float db1 = df *
                                (0.5f + static_cast<float>(scales[ib32] >> 4)) *
                                0.25f;
                    float dl = (l >= 2u) ? db1 : db0;
                    uint16_t gridIdx = static_cast<uint16_t>(
                            qsv[ib32 * 4u + l] |
                            ((qh[ib32] << (8 - 2u * l)) & 0x300));
                    const uint8_t *grid =
                            reinterpret_cast<const uint8_t *>(&c_iq2s_grid[gridIdx]);
                    const uint8_t signs = qsv[32u + ib32 * 4u + l];
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        float w =
                                dl * static_cast<float>(grid[j]) *
                                ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t ck = b * 256u + e0 + j;
                        acc = fmaf(w, (ck < cols) ? xs[j] : 0.0f, acc);
                    }
                    rp += kIQ2S_BYTES;
                }
            } else if (TYPE == kTypeIQ3_XXS) {
                // IQ3_XXS block (98 B): d(2) + qs[96].  qs[64..95] doubles as
                // the scales+signs area (scales_and_signs = qs + QK_K/4 = qs+64).
                // Per (ib32, l):
                //   aux32 = u32(scales_and_signs + 4*ib32)
                //   db    = d*(0.5f + (aux32 >> 28))*0.5f
                //   signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]
                //   grid1 = qs[8*ib32 + 2*l + 0], grid2 = qs[8*ib32 + 2*l + 1]
                //   y[0..3] from grid1 bytes, y[4..7] from grid2 bytes.
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    const uint8_t *qsv = rp + 2;
                    const uint32_t ib32 = e0 >> 5u;
                    const uint32_t l = (e0 >> 3u) & 3u;
                    uint32_t aux32;
                    std::memcpy(&aux32, qsv + 64u + 4u * ib32,
                                sizeof(uint32_t));
                    float db = df *
                               (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                    const uint8_t signs =
                            c_ksigns_iq2xs[(aux32 >> (7u * l)) & 127u];
                    const uint32_t g1 = c_iq3xxs_grid[qsv[8u * ib32 + 2u * l]];
                    const uint32_t g2 =
                            c_iq3xxs_grid[qsv[8u * ib32 + 2u * l + 1u]];
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        uint32_t gval = (j < 4u)
                                                ? ((g1 >> (8u * j)) & 0xFFu)
                                                : ((g2 >> (8u * (j - 4u))) &
                                                   0xFFu);
                        float w = db * static_cast<float>(gval) *
                                  ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t ck = b * 256u + e0 + j;
                        acc = fmaf(w, (ck < cols) ? xs[j] : 0.0f, acc);
                    }
                    rp += kIQ3XXS_BYTES;
                }
            } else if (TYPE == kTypeIQ3S) {
                // IQ3_S block (110 B): d(2) + qs[64] + qh[8] + signs[32] +
                // scales[4].  Sub-blocks pair up: pair=ib32>>1, half=ib32&1.
                // First half of a pair uses db1 = d*(1+2*(scales[pair]&0xf));
                // second half db2 = d*(1+2*(scales[pair]>>4)).  qh advance by
                // 2 per pair; qs by 16 per pair (8 per half); signs by 8 per
                // pair (4 per half).  Grid values are 4 x 4-bit bytes packed in
                // a uint32_t (grid1[j] = byte j).
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    const uint8_t *qs = rp + 2;
                    const uint8_t *qh = rp + 66;
                    const uint8_t *signs = rp + 74;
                    const uint8_t *scales = rp + 106;
                    const uint32_t ib32 = e0 >> 5u;
                    const uint32_t half = ib32 & 1u;
                    const uint32_t pair = ib32 >> 1u;
                    const uint32_t l = (e0 >> 3u) & 3u;
                    const uint32_t qhIdx = pair * 2u + half;
                    const uint32_t qsOff = pair * 16u + half * 8u;
                    const uint32_t signOff = pair * 8u + half * 4u;
                    float db =
                            df *
                            (1.0f +
                             2.0f *
                                     static_cast<float>((half == 0)
                                                                ? (scales[pair] & 0xf)
                                                                : (scales[pair] >> 4)));
                    const uint16_t gridIdx1 = static_cast<uint16_t>(
                            qs[qsOff + 2u * l] |
                            ((qh[qhIdx] << (8 - 2u * l)) & 256));
                    const uint16_t gridIdx2 = static_cast<uint16_t>(
                            qs[qsOff + 2u * l + 1u] |
                            ((qh[qhIdx] << (7 - 2u * l)) & 256));
                    const uint32_t g1 = c_iq3s_grid[gridIdx1];
                    const uint32_t g2 = c_iq3s_grid[gridIdx2];
                    const uint8_t sbyte = signs[signOff + l];
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        uint32_t gval = (j < 4u)
                                                ? ((g1 >> (8u * j)) & 0xFFu)
                                                : ((g2 >> (8u * (j - 4u))) &
                                                   0xFFu);
                        float w = db * static_cast<float>(gval) *
                                  ((sbyte & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t ck = b * 256u + e0 + j;
                        acc = fmaf(w, (ck < cols) ? xs[j] : 0.0f, acc);
                    }
                    rp += kIQ3S_BYTES;
                }
            } else if (TYPE == kTypeQ5K) {
                // Q5_K block (176 B): d(2) + dmin(2) + scales[12] + qh[32] +
                // qs[128].  4 chunks of 64; sub-block 0 of a chunk uses qh bit
                // u1 = 1<<(2*g), sub-block 1 uses u2 = 2<<(2*g).  Lane owns 8
                // consecutive columns that never cross a 32-weight sub-block
                // boundary (e0 is a multiple of 8), so qs/qh byte = g*32 + jj.
                auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                      uint8_t *m_out) {
                    if (j < 4) {
                        *d_out = q[j] & 63;
                        *m_out = q[j + 4] & 63;
                    } else {
                        *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                        *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                    }
                };
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 2);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *scales = rp + 4;
                    const uint8_t *qh = rp + 16;
                    const uint8_t *qs = rp + 48;
                    const uint32_t g = e0 / 64u;         // chunk 0..3
                    const uint32_t sub = (e0 / 32u) & 1u;// 0 or 1
                    const uint32_t jj0 = e0 & 31u;       // first of 8
                    uint8_t sc, m;
                    getScaleMin(static_cast<int>(2 * g + sub), scales, &sc, &m);
                    float dl = df * static_cast<float>(sc);
                    float ml = dmf * static_cast<float>(m);
                    const uint8_t u = (sub == 0)
                                              ? static_cast<uint8_t>(1u << (2u * g))
                                              : static_cast<uint8_t>(2u << (2u * g));
                    const uint8_t *qsp = qs + g * 32u;
                    // NOTE: qh does NOT advance per chunk -- the same qh[l]
                    // bytes are reused for all 4 chunks with the u1/u2 bit
                    // masks selecting the (chunk, sub-block) position (matches
                    // dequantizeQ5_KBlock, which advances only qs by 32).
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t kk = 0; kk < 8; ++kk) {
                        const uint8_t qb = qsp[jj0 + kk];
                        uint8_t q5 = (sub == 0)
                                             ? static_cast<uint8_t>(
                                                       (qb & 0xF) |
                                                       ((qh[jj0 + kk] & u) ? 16
                                                                           : 0))
                                             : static_cast<uint8_t>(
                                                       (qb >> 4) |
                                                       ((qh[jj0 + kk] & u) ? 16
                                                                           : 0));
                        float w = dl * static_cast<float>(q5) - ml;
                        const uint32_t ck = b * 256u + e0 + kk;
                        acc = fmaf(w, (ck < cols) ? xs[kk] : 0.0f, acc);
                    }
                    rp += kQ5K_BYTES;
                }
            } else {
                acc = 0.0f;
            }
// Warp reduction: each lane holds the partial dot product over its
// disjoint 8-column-per-block subset; the full row result is the
// sum across all 32 lanes.  (5 shuffles + 5 adds per row.)
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) out[row] = acc;
        }

        // ------------------------------------------------------------------
        // SPLIT-K GEMV (2026-09-29, decode small-row occupancy lever):
        //
        // The warp-per-row kQGemv launches nBlocks = ceil(rows/8) blocks of
        // 8 warps.  On the SMALL decode matrices (attnO / gate+up / down at
        // rows=1536) this leaves only ~2.8 blocks/SM resident (the kernel is
        // register-limited at ~90 regs/thread), i.e. ~22 warps/SM streaming
        // their rows — well under the 32-warp/SM hardware cap, and the
        // measured per-matrix throughput (down: 74 us/layer for 5.9 MB =
        // ~80 GB/s) is far under the ~550 GB/s DRAM peak.
        //
        // THIS kernel raises the concurrent warp count WITHOUT changing any
        // per-warp math: each row is handled by KN chunk-warps (blockDim.y =
        // KN, one block per row), and chunk w owns the CONTIGUOUS block
        // sub-range  [(w*bpr)/KN, ((w+1)*bpr)/KN).  Each chunk warp runs the
        // byte-identical kQGemv float chain (4-way a0..a3 block batching with
        // in-order fold, remainder direct-fold, 5-shuffle warp reduce) over
        // its sub-range; the KN per-chunk partials are summed in shared
        // memory IN CHUNK ORDER by warp 0 lane 0.  For KN=2 on down
        // (rows=1536): 1536 blocks x 2 warps = 3072 warps vs 1536 today, at
        // the SAME register footprint per warp (each warp does half the
        // blocks) — more concurrent DRAM streams, more blocks/SM resident.
        //
        // The cross-chunk float sum (p0+p1+.. in chunk order) is a 1-ulp-class
        // reordering vs the fully-serial fold, so the per-layer logits may
        // differ from the serial path by ~1e-7 relative — within the
        // tolerance-based parity bar that the Q2K/Q3K float GEMV paths already
        // use (the exact-integer Q8_K paths are the bit-exact ones).  Measured
        // KEEP decision (2026-09-29): down-only Q3_K split-K is a WIN (down
        // 1.50 -> 1.40 ms/layer, ~7%, parity PASS); a Q2_K split branch was
        // measured and REVERTED because splitting the gate+up (rows=8960)
        // flooded the L2/x-traffic with duplicate activation reads (gate+up
        // 1.77 -> 3.40 ms/layer, +92%) — Q2_K split only helps the true
        // SMALL-row matrices (attnO at rows=1536) and the 8960-row gate/up
        // far outweighs it.  This kernel is therefore Q3K-only (ffnDown).
        template<int TYPE, int KN>
        __global__ void __launch_bounds__(32 * KN) kQGemvSplitK(
                const uint8_t *__restrict__ w, const float *__restrict__ x,
                float *__restrict__ out, uint32_t rows, uint32_t cols,
                uint32_t rowBytes, uint32_t blocksPerRow) {
            const uint32_t row = blockIdx.x;
            if (row >= rows) return;
            const uint32_t wId = threadIdx.y;// 0..KN-1
            const uint32_t lane = threadIdx.x;
            const uint32_t cLow = (wId * blocksPerRow) / KN;
            const uint32_t cHigh = ((wId + 1u) * blocksPerRow) / KN;
            const uint32_t nBlk = cHigh - cLow;
            if (nBlk == 0) {
                // Chunk with no blocks (tail when bpr < KN): contribute 0.
                __syncthreads();
                if (wId == 0u && lane == 0u) out[row] = 0.0f;
                return;
            }
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes +
                                static_cast<uint64_t>(cLow) * kQ3K_BYTES;
            float acc = 0.0f;
            if (TYPE == kTypeQ3K) {
                const uint32_t sub = lane / 16u;
                uint32_t b = 0;
                for (; b + 4u <= nBlk; b += 4u) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                    for (uint32_t bb = 0; bb < 4; ++bb) {
                        const uint8_t *crp =
                                rp + static_cast<uint64_t>(bb) * kQ3K_BYTES;
                        __half d = *reinterpret_cast<const __half *>(crp + 108);
                        float df = __half2float(d);
                        const uint8_t *hm = crp;
                        const uint8_t *q = crp + 32;
                        const uint8_t *scales = crp + 96;
                        uint32_t aux[4];
                        std::memcpy(&aux[0], scales, 12);
                        uint32_t tmp = aux[2];
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
                        const uint8_t hmb = hm[lane];
                        float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                : (bb == 2u)       ? a2
                                                                   : a3;
#pragma unroll
                        for (uint32_t half = 0; half < 2; ++half) {
                            const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                            for (uint32_t jj = 0; jj < 4; ++jj) {
                                float dl = df * static_cast<float>(
                                                        sc16[half * 8u + jj * 2u + sub] - 32);
                                const uint32_t maskBit = 1u << (half * 4u + jj);
                                float qv = static_cast<float>(static_cast<int8_t>(
                                        ((qb >> (jj * 2u)) & 3) -
                                        ((hmb & maskBit) ? 0 : 4)));
                                const uint32_t cidx =
                                        (cLow + b + bb) * 256u + half * 128u +
                                        jj * 32u + lane;
                                slot = fmaf(dl * qv,
                                            (cidx < cols) ? x[cidx] : 0.0f, slot);
                            }
                        }
                    }
                    acc += a0;
                    acc += a1;
                    acc += a2;
                    acc += a3;
                    rp += 4u * kQ3K_BYTES;
                }
                for (; b < nBlk; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 108);
                    float df = __half2float(d);
                    const uint8_t *hm = rp;
                    const uint8_t *q = rp + 32;
                    const uint8_t *scales = rp + 96;
                    uint32_t aux[4];
                    std::memcpy(&aux[0], scales, 12);
                    uint32_t tmp = aux[2];
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
                    const uint8_t hmb = hm[lane];
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            float dl = df * static_cast<float>(
                                                    sc16[half * 8u + jj * 2u + sub] - 32);
                            const uint32_t maskBit = 1u << (half * 4u + jj);
                            float qv = static_cast<float>(static_cast<int8_t>(
                                    ((qb >> (jj * 2u)) & 3) -
                                    ((hmb & maskBit) ? 0 : 4)));
                            const uint32_t cidx =
                                    (cLow + b) * 256u + half * 128u + jj * 32u + lane;
                            acc = fmaf(dl * qv, (cidx < cols) ? x[cidx] : 0.0f,
                                       acc);
                        }
                    }
                    rp += kQ3K_BYTES;
                }
            } else {
                acc = 0.0f;
            }
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, off);
            }
            __shared__ float sPart[KN];
            if (lane == 0) sPart[wId] = acc;
            __syncthreads();
            if (wId == 0u && lane == 0u) {
                float total = 0.0f;
#pragma unroll
                for (uint32_t k = 0; k < KN; ++k) {
                    total += sPart[k];// chunk order
                }
                out[row] = total;
            }
        }

        // ---- 32-wide-block GEMV (Q5_0 / Q8_0 / IQ4_NL) ----
        // These legacy quant formats use 32-weight blocks (22 B / 34 B / 18 B
        // each) rather than the 256-wide K-quant blocks handled by kQGemv
        // above. One warp handles one output row; each lane owns column
        // (b*32 + lane) of a block, computes its single weight, and the warp
        // reduction sums them.  blocksPerRow = ceil(cols/32); rowBytes =
        // blocksPerRow * blockBytes.
        template<int TYPE>
        __global__ void kQGemvSmall32(const uint8_t *__restrict__ w,
                                      const float *__restrict__ x,
                                      float *__restrict__ out, uint32_t rows,
                                      uint32_t cols, uint32_t rowBytes,
                                      uint32_t blocksPerRow) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint8_t *rp =
                    w + static_cast<uint64_t>(row) * rowBytes;
            float acc = 0.0f;
            constexpr uint32_t BLOCK_BYTES = (TYPE == kTypeQ5_0)
                                                     ? kQ5_0_BYTES
                                             : (TYPE == kTypeQ8_0) ? kQ8_0_BYTES
                                                                   : kIQ4NL_BYTES;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blk = rp + static_cast<uint64_t>(b) * BLOCK_BYTES;
                const uint32_t col = b * 32u + lane;
                if (col >= cols) break;
                __half d = *reinterpret_cast<const __half *>(blk);
                float wv;
                if (TYPE == kTypeQ8_0) {
                    const int8_t q = reinterpret_cast<const int8_t *>(blk + 2)[lane];
                    wv = __half2float(d) * static_cast<float>(q);
                } else if (TYPE == kTypeIQ4NL) {
                    // IQ4_NL (4.5 bpw): d(fp16,2) + qs[16].  Element `lane`:
                    //   j   = lane & 15
                    //   nib = (lane < 16) ? qs[j] & 0x0F : qs[j] >> 4
                    //   wv  = d * kvalues_iq4nl[nib]   (signed 8-bit table)
                    // Matches CPU dequantizeIQ4_NLBlock / dotProductIQ4_NL.
                    const uint8_t *qs = blk + 2;
                    const uint32_t j = lane & 15u;
                    const uint8_t nib =
                            (qs[j] >> (4u * (lane >> 4u))) & 0xFu;
                    wv = __half2float(d) *
                         static_cast<float>(c_kvalues_iq4nl[nib]);
                } else {// Q5_0 (matches CPU dequantizeQ5_0Block)
                    // Element `lane` of the 32-wide block (QK5_0=32):
                    //   j      = lane & 15
                    //   low    = (lane < 16) ? qs[j] & 0x0F : qs[j] >> 4
                    //   high   = bit j of qh for lane < 16, else bit (j+16)
                    //   val    = (low | (high << 4)) - 16
                    // NOTE: the CPU reference reads the upper-half high bit via
                    // `((qh >> (j + 12)) & 0x10)` -- the 0x10 mask selects bit 4
                    // of the shifted word, i.e. ORIGINAL bit (j+16) of qh, NOT
                    // bit (j+12) (the surrounding comment in GGMLDequantize.hpp
                    // is misleading).  Use (j+16) here to match the CPU math.
                    uint32_t qh;
                    std::memcpy(&qh, blk + 2, sizeof(uint32_t));
                    const uint8_t *qs = blk + 6;
                    const uint32_t j = lane & 15u;
                    const uint8_t low =
                            (qs[j] >> (4u * (lane >> 4u))) & 0xFu;
                    const uint32_t hb = (lane & 16u) ? (j + 16u) : j;
                    const uint8_t high =
                            static_cast<uint8_t>((qh >> hb) & 1u);
                    const int val = static_cast<int>(low | (high << 4)) - 16;
                    wv = __half2float(d) * static_cast<float>(val);
                }
                acc = fmaf(wv, x[col], acc);
            }
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) out[row] = acc;
        }

        // ==================================================================
        // Prism PQ2_0 (type 142) ternary GEMV + Hadamard fold (Bonsai).
        //
        // Ternary-Bonsai-2-27B stores its (near-ternary) linear weights in a
        // 2-bit format already folded with a per-input-axis normalized
        // Sylvester-Walsh-Hadamard rotation H (block 1024) and a per-feature
        // sign flip S: W_folded = W @ (H S P).  To recover the primal
        // projection y = W x the ACTIVATION is transformed x' = H S P x first,
        // then a plain (ternary) GEMV computes W_folded @ x' = W @ x.  The
        // PQ2_0 weights therefore stay compressed in VRAM; only the small
        // activation vector is rotated per token.
        //
        // PQ2_0 block = 34 bytes / 128 weights: d(fp16,2) + qs[uint8;32].
        // Code -> value: 0 -> -1, 1 -> 0, 2 -> +1, 3 -> +2 (value = q-1).
        // Element j: byte j>>2, bits (j&3)*2.  Bit-exact with the CPU
        // GGMLDequantize::dequantizePQ2_0Block / dotProductPQ2_0.
        // ==================================================================

        // Row GEMV: one warp per output row; each of the 32 lanes reads ONE qs
        // byte (4 two-bit codes) and owns the 4 columns [lane*4, lane*4+4) of
        // the 128-wide block.  Mirrors kQGemvSmall32's warp-reduce epilogue.
        __global__ void __launch_bounds__(32) kQGemvPQ2_0(
                const uint8_t *__restrict__ w, const float *__restrict__ x,
                float *__restrict__ out, uint32_t rows, uint32_t cols,
                uint32_t rowBytes, uint32_t blocksPerRow) {
            const uint32_t row = blockIdx.x;
            if (row >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            float acc = 0.0f;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blk =
                        rp + static_cast<uint64_t>(b) * kPQ2_0_BYTES;
                const float d =
                        __half2float(*reinterpret_cast<const __half *>(blk));
                const uint8_t qb = blk[2u + lane];
                const uint32_t base = b * 128u + lane * 4u;
                const uint32_t c0 = base + 0u, c1 = base + 1u;
                const uint32_t c2 = base + 2u, c3 = base + 3u;
                const float x0 = (c0 < cols) ? x[c0] : 0.0f;
                const float x1 = (c1 < cols) ? x[c1] : 0.0f;
                const float x2 = (c2 < cols) ? x[c2] : 0.0f;
                const float x3 = (c3 < cols) ? x[c3] : 0.0f;
                acc = fmaf(d * (static_cast<float>((qb >> 0u) & 3u) - 1.0f), x0,
                           acc);
                acc = fmaf(d * (static_cast<float>((qb >> 2u) & 3u) - 1.0f), x1,
                           acc);
                acc = fmaf(d * (static_cast<float>((qb >> 4u) & 3u) - 1.0f), x2,
                           acc);
                acc = fmaf(d * (static_cast<float>((qb >> 6u) & 3u) - 1.0f), x3,
                           acc);
            }
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) out[row] = acc;
        }

        // GDN value-head permutation index (matches prism::permuteGdnValueGrouped):
        // destination grouped feature g = d + r*headV + k*headV*rep maps to the
        // tiled source feature hv*headV + d with hv = k + r*nKHeads.
        // GDN value-head permutation index (inverse of prism::
        // permuteGdnValueGrouped): destination grouped feature
        //   g = d + r*headV + k*headV*rep
        // maps to the tiled source feature hv*headV + d with
        //   hv = k + r*nKHeads.
        // Decompose g: inner = g % headV; rest = g / headV = r + k*rep, so
        // r = rest % rep and k = rest / rep; then recompose the tiled head with
        // nKHeads (= nk).  rep and nk differ (3 vs 16 for Bonsai), so both are
        // required.
        __device__ __forceinline__ uint32_t prismGdnSrcIndex(uint32_t g,
                                                             uint32_t headV,
                                                             uint32_t nk,
                                                             uint32_t rep) {
            const uint32_t inner = g % headV;
            const uint32_t rest = g / headV;// r + k*rep
            const uint32_t r = rest % rep;
            const uint32_t k = rest / rep;
            return (k + r * nk) * headV + inner;
        }

        // Normalized blockwise Sylvester-Walsh-Hadamard rotation of ONE
        // kPrismBlock-wide group held in shared memory (in place).
        __device__ __forceinline__ void prismFwhtShared(float *tile, uint32_t n,
                                                        uint32_t tid,
                                                        uint32_t nThreads) {
            for (uint32_t hs = 1; hs < n; hs <<= 1) {
                for (uint32_t p = tid; p < n / 2u; p += nThreads) {
                    const uint32_t i = (p / hs) * (hs << 1) + (p % hs);
                    const float a = tile[i], b = tile[i + hs];
                    tile[i] = a + b;
                    tile[i + hs] = a - b;
                }
                __syncthreads();
            }
            const float scale = rsqrtf(static_cast<float>(n));
            for (uint32_t i = tid; i < n; i += nThreads) tile[i] *= scale;
            __syncthreads();
        }

        // Forward Prism activation transform (per 1024-wide tile): optional GDN
        // value-head permutation, then the per-feature sign flip, then the
        // blockwise normalized Hadamard.  grid = ceil(n / kPrismBlock);
        // block = 256.  `v` (primal activation) and `dst` (folded activation)
        // must NOT alias.
        __global__ void kPrismFold(const float *__restrict__ v,
                                   float *__restrict__ dst, uint32_t n,
                                   const float *__restrict__ signs,
                                   uint32_t gdnHeadV, uint32_t gdnNk,
                                   uint32_t gdnRep) {
            __shared__ float tile[kPrismBlock];
            const uint32_t base = blockIdx.x * kPrismBlock;
            const uint32_t tid = threadIdx.x;
            const uint32_t have = (base < n) ? ((n - base < kPrismBlock)
                                                        ? (n - base)
                                                        : kPrismBlock)
                                             : 0u;
            if (have == 0u) return;
            const bool perm = (gdnRep > 1u && gdnHeadV > 0u && gdnNk > 0u);
            for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x) {
                float val = 0.0f;
                if (i < have) {
                    const uint32_t g = base + i;
                    const uint32_t src =
                            perm ? prismGdnSrcIndex(g, gdnHeadV, gdnNk, gdnRep) : g;
                    val = v[src];
                    if (signs != nullptr) val *= signs[g];
                }
                tile[i] = val;
            }
            __syncthreads();
            prismFwhtShared(tile, kPrismBlock, tid, blockDim.x);
            for (uint32_t i = tid; i < have; i += blockDim.x) {
                dst[base + i] = tile[i];
            }
        }

        // PQ2_0 token-embedding dequant + INVERSE fold (S then H -> rotate then
        // sign).  grid = seqLen tokens; block = 256.  Dequantizes the ternary
        // row into `hidden`, then applies the blockwise FWHT + signs in place so
        // the latent (rotated) embedding row is restored to the primal basis.
        __global__ void kEmbedDequantPQ2_0(
                const uint8_t *__restrict__ embed, const int32_t *__restrict__ tokens,
                float *__restrict__ hidden, uint32_t seqLen, uint32_t hiddenSize,
                uint32_t blocksPerRow, uint32_t rowBytes, uint32_t vocabSize,
                const float *__restrict__ signs) {
            const uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            const int32_t t = tokens[s];
            if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
            const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
            float *h = hidden + static_cast<size_t>(s) * hiddenSize;
            __shared__ float tile[kPrismBlock];
            const uint32_t tid = threadIdx.x;
            const uint32_t lane = tid & 31u, warp = tid >> 5;
            for (uint32_t b = warp; b < blocksPerRow; b += (blockDim.x >> 5)) {
                const uint8_t *blk =
                        row + static_cast<uint64_t>(b) * kPQ2_0_BYTES;
                const float d =
                        __half2float(*reinterpret_cast<const __half *>(blk));
                const uint8_t qb = blk[2u + lane];
                const uint32_t base = b * 128u + lane * 4u;
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint32_t c = base + j;
                    if (c < hiddenSize)
                        h[c] = d * (static_cast<float>((qb >> (j * 2u)) & 3u) -
                                    1.0f);
                }
            }
            __syncthreads();
            for (uint32_t base = 0; base + kPrismBlock <= hiddenSize;
                 base += kPrismBlock) {
                for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x)
                    tile[i] = h[base + i];
                __syncthreads();
                prismFwhtShared(tile, kPrismBlock, tid, blockDim.x);
                for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x) {
                    float val = tile[i];
                    if (signs != nullptr) val *= signs[base + i];
                    h[base + i] = val;
                }
                __syncthreads();
            }
        }

        // ------------------------------------------------------------------
        // Q8_K activation quantization + integer vec-dot GEMV (decode path)
        // for the IQ2_S / IQ3_XXS / IQ3_S weight types.
        //
        // ------------------------------------------------------------------

        // Q8_K per-block stride: int8 qs[256] + int16 bsums[16] + float d,
        // 16-byte aligned (float at 288).
        static constexpr uint32_t kQ8K_Q = 0;
        static constexpr uint32_t kQ8K_B = 256;
        static constexpr uint32_t kQ8K_D = 256 + 32;
        static constexpr uint32_t kQ8K_STRIDE = 256 + 32 + 16;

        // Quantize one fp32 activation vector into Q8_K blocks in q8k.
        // Bit-exact with GGMLDequantize::quantizeQ8K (nearest-int rintf,
        // clamp to +-127, d = -max/vscale, bsums per 16-group).
        // grid = ceil(cols/256), block = 256 threads.
        __global__ void kQuantizeQ8K(const float *__restrict__ x,
                                     uint8_t *__restrict__ q8k,
                                     uint32_t cols) {
            const uint32_t b = blockIdx.x;
            const uint32_t e0 = b * 256u;
            const uint32_t n = (cols - e0) < 256u ? (cols - e0) : 256u;
            if (n == 0) return;
            uint8_t *blk = q8k + static_cast<uint64_t>(b) * kQ8K_STRIDE;
            float maxv = 0.0f;
            float amax = 0.0f;
            for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
                float v = x[e0 + j];
                float av = fabsf(v);
                if (av > amax) {
                    amax = av;
                    maxv = v;
                }
            }
            // reduce the signed max (signed-ness travels with the abs-max).
            // NOTE: NO divergent early-exit here -- every thread must reach the
            // same __syncthreads() count or the block's barrier is undefined.
            // An all-zero block simply yields iscale/d = 0 (bsums stay 0).
            __shared__ float sMax[256];
            sMax[threadIdx.x] = (amax > 0.0f) ? maxv : 0.0f;
            __syncthreads();
            for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    float a0 = sMax[threadIdx.x], a1 = sMax[threadIdx.x + s];
                    if (fabsf(a1) > fabsf(a0)) sMax[threadIdx.x] = a1;
                }
                __syncthreads();
            }
            const float max = sMax[0];
            // __fdiv_rn forces the two IEEE correctly-rounded divisions
            // (-127/max then 1/iscale) that the CPU's quantizeQ8K emits as two
            // divss.  A plain '/' lets nvcc algebraically contract
            // 1/(-127/max) -> -max/127, which rounds differently (measured:
            // exact 1-ulp drift on the stored d, corrupting every downstream
            // Q4K/Q6K x Q8_K dot by 1 ulp even though qs/bsums stay bit-exact).
            const float iscale = (max != 0.0f) ? __fdiv_rn(-127.0f, max) : 0.0f;
            const float d = (max != 0.0f) ? __fdiv_rn(1.0f, iscale) : 0.0f;
            for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
                if (max == 0.0f) {
                    blk[kQ8K_Q + j] = 0;
                    continue;
                }
                int v = static_cast<int>(rintf(iscale * x[e0 + j]));
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                blk[kQ8K_Q + j] = static_cast<uint8_t>(static_cast<int8_t>(v));
            }
            if (threadIdx.x == 0) {
                *reinterpret_cast<float *>(blk + kQ8K_D) = d;
            }
            __syncthreads();
            for (uint32_t j = threadIdx.x; j < 16; j += blockDim.x) {
                int sum = 0;
                for (uint32_t ii = 0; ii < 16; ++ii) {
                    sum += static_cast<int>(
                            static_cast<int8_t>(blk[kQ8K_Q + j * 16u + ii]));
                }
                *reinterpret_cast<int16_t *>(blk + kQ8K_B + j * 2) =
                        static_cast<int16_t>(sum);
            }
        }

        // ==================================================================
        // int8 tensor-core MMQ (mma.s8.m8n8k16) for the Int8Pack format
        // ==================================================================
        // Target: Turing (sm_75) provides int8 tensor cores ONLY for the
        // m8n8k16 / m8n8k32 shapes; mma.m16n8k16 for int8 requires sm_80+.  The
        // pack (include/Int8Pack.hpp) therefore targets m8n8k16, whose B
        // fragment (b_i = B[(lane%4)*4+i][lane>>2]) is IDENTICAL to the
        // m16n8k16 B fragment, so the same packed weights feed both shapes.
        //
        // GEMM: y[seq][rows] = A[seq][cols] @ W[rows][cols]^T with the
        // 256-wide Q8_K activation (kQuantizeQ8K) and the exact IQ3_XXS
        // epilogue 4 * actd * sum_g db_g * sum_int_g (matches
        // Int8Pack::dotBlockQ8K / dotProductIQ3_XXS_Q8K).  decode (seq==1) runs
        // the SAME kernel -- the identical mma family, just one real token.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
#define TINYCODER_MMA_S8 1
#endif

        // D += A @ B (m8n8k16, s8 x s8 -> s32).  a: 1 b32/lane (4 int8),
        // b: 1 b32/lane (4 int8), c: 2 s32/lane.  Guarded so sm_80/sm_90 also
        // compile (the shape is valid on every arch >= sm_75).
        __device__ __forceinline__ void mmaS8_8x8x16(int32_t c[2], uint32_t a,
                                                     uint32_t b) {
#ifdef TINYCODER_MMA_S8
            asm volatile(
                    "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                    "{%0, %1}, {%2}, {%3}, {%0, %1};\n"
                    : "+r"(c[0]), "+r"(c[1])
                    : "r"(a), "r"(b));
#else
            (void) c;
            (void) a;
            (void) b;
#endif
        }

        // One lane's 4 int8 of the packed B cell (cell = nt*k16Tiles + kt).
        // The pack stores lane L's 4 int8 at bytes [L*4, L*4+4); reading them
        // as one little-endian b32 yields (b0,b1,b2,b3) in PTX order.
        __device__ __forceinline__ uint32_t mmqLoadBCell(const int8_t *bpk,
                                                         int cell, int lane) {
            const int32_t *p = reinterpret_cast<const int32_t *>(
                    bpk + static_cast<size_t>(cell) * 128);
            return static_cast<uint32_t>(p[lane]);
        }

        // One lane's 4 int8 of the A fragment: the Q8_K qs of `token` at
        // columns k0 + (lane%4)*4 .. +3.  k0 is a multiple of 16 and 16 | 256,
        // so the 4 bytes never straddle a 256 block.
        __device__ __forceinline__ uint32_t mmqLoadAFrag(const int8_t *xq,
                                                         int actStride,
                                                         int token, int k0,
                                                         int lane) {
            const int k = k0 + (lane & 3) * 4;
            const int8_t *q = xq + static_cast<size_t>(token) * actStride +
                              static_cast<size_t>(k >> 8) * kQ8K_STRIDE +
                              (k & 255);
            return static_cast<uint32_t>(static_cast<uint8_t>(q[0])) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(q[1])) << 8) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(q[2])) << 16) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(q[3])) << 24);
        }

        // Prefill GEMM / decode GEMV for IQ3_XXS packed weights.
        //   one block = 8 warps (256 threads); each warp owns an 8-token
        //   (m8) x 8-row (n8) output tile; the whole block shares ONE n8 row
        //   tile (n0 = blockIdx.x*8) and spans BM = 64 tokens.  Because all 8
        //   warps read the SAME per-(n8,k16) B cell, the L1 broadcast serves
        //   it after the first warp touches DRAM.
        //   grid = (n8Tiles, ceil(seq / 64)).
        // Accumulates the integer mma over each 32-group (2k16) then applies
        // the per-row group db; the float group sums of a 256 block are then
        // folded with that block's activation scale exactly once -- the same
        // operation order as dotBlockQ8K.
        __global__ void __launch_bounds__(256) kMmqGemm_8x8x16(
                const int8_t *__restrict__ bpk, const float *__restrict__ sc,
                const int8_t *__restrict__ xq, float *__restrict__ y, int seq,
                int rows, int cols, int k16Tiles, int actStride) {
            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;
            const int nt = blockIdx.x;
            const int n0 = nt * 8;
            const int m0 = (blockIdx.y * 8 + warp) * 8;
            const int token = m0 + (lane >> 2);
            const bool active = token < seq;
            const int nrow0 = (lane & 3) * 2;
            const int nrow1 = nrow0 + 1;
            const int scRowBase = nt * k16Tiles;
            const int nGroups = (cols + 31) >> 5;

            int32_t D[2] = {0, 0};
            float grp0 = 0.0f, grp1 = 0.0f;
            float row0 = 0.0f, row1 = 0.0f;
            for (int gg = 0; gg < nGroups; ++gg) {
                const int kt = 2 * gg;
                {
                    const uint32_t a = active
                                               ? mmqLoadAFrag(xq, actStride,
                                                              token, kt * 16,
                                                              lane)
                                               : 0u;
                    const uint32_t b = mmqLoadBCell(bpk, scRowBase + kt, lane);
                    mmaS8_8x8x16(D, a, b);
                }
                if (kt + 1 < k16Tiles) {
                    const uint32_t a = active
                                               ? mmqLoadAFrag(xq, actStride,
                                                              token,
                                                              (kt + 1) * 16,
                                                              lane)
                                               : 0u;
                    const uint32_t b =
                            mmqLoadBCell(bpk, scRowBase + kt + 1, lane);
                    mmaS8_8x8x16(D, a, b);
                }
                const int scIdx = (scRowBase + kt) * 8;
                grp0 += sc[scIdx + nrow0] * static_cast<float>(D[0]);
                grp1 += sc[scIdx + nrow1] * static_cast<float>(D[1]);
                D[0] = 0;
                D[1] = 0;
                if (((gg & 7) == 7) || (gg == nGroups - 1)) {
                    const float ad = active
                                             ? *reinterpret_cast<const float *>(
                                                       xq + static_cast<size_t>(token) * actStride +
                                                       static_cast<size_t>(
                                                               gg >> 3) *
                                                               kQ8K_STRIDE +
                                                       kQ8K_D)
                                             : 0.0f;
                    row0 += 4.0f * ad * grp0;
                    row1 += 4.0f * ad * grp1;
                    grp0 = 0.0f;
                    grp1 = 0.0f;
                }
            }
            if (active) {
                const int nOut0 = n0 + nrow0;
                const int nOut1 = n0 + nrow1;
                if (nOut0 < rows)
                    y[static_cast<size_t>(token) * rows + nOut0] = row0;
                if (nOut1 < rows)
                    y[static_cast<size_t>(token) * rows + nOut1] = row1;
            }
        }

        // Affine Q4_K prefill/decode GEMM reading the RAW GGML Q4_K weights
        // (no pre-pack), so the int8 tensor-core path costs ZERO extra VRAM --
        // the packed kMmqGemmQ4K_Affine needs a ~1.5x weight copy that does not
        // fit the 11 GB card for a 9B Q4_K model.  Q4_K is affine
        // (w = d*sc*q - dmin*m); lane L of the mma B fragment needs weight row
        // (n0 + L/4) at columns k0 + (L%4)*4 .. +3.  In the raw block those 4
        // 4-bit q live in 4 consecutive bytes
        //   qs[32*(i>>1) + (k0%32) + (L%4)*4 .. +3]   (i = covering 32-group)
        // (low nibble for even i, high for odd), so one 4-aligned u32 load +
        // nibble select yields the whole per-lane B word.  The dsc=d*sc /
        // dmn=dmin*m of the covering 32-group come straight from the block
        // header.  Geometry identical to kMmqGemm_8x8x16 (8 warps x 8 tokens x
        // 8 rows); grid = (n8Tiles, ceil(seq/64)); actStride = blocksPerRow*kQ8K_STRIDE.
        // Same accumulation class as kxQ8KBlockDot<Q4_K> (d8*(main-min)), NOT
        // bit-exact; the batch-vs-sequential contract tolerates that.
        __global__ void __launch_bounds__(256) kMmqGemmQ4K_AffineRaw(
                const uint8_t *__restrict__ w, const int8_t *__restrict__ xq,
                float *__restrict__ y, int seq, int rows, int cols,
                int rowBytes, int k16Tiles, int actStride) {
            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;
            const int nt = blockIdx.x;
            const int n0 = nt * 8;
            const int m0 = (blockIdx.y * 8 + warp) * 8;
            const int token = m0 + (lane >> 2);
            const bool active = token < seq;
            const int nrow0 = (lane & 3) * 2;
            const int nrow1 = nrow0 + 1;
            const int nGroups = (cols + 31) >> 5;
            const int nB = n0 + (lane >> 2);// mma B row = W row for this lane
            const uint8_t *rowBase = w + static_cast<size_t>(nB) * rowBytes;
            const bool bRowValid = nB < rows;

            int32_t D[2] = {0, 0};
            float main0 = 0.0f, main1 = 0.0f;
            float min0 = 0.0f, min1 = 0.0f;
            float row0 = 0.0f, row1 = 0.0f;
            for (int gg = 0; gg < nGroups; ++gg) {
                const int kt = 2 * gg;
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int k0 = (kt + half) * 16;
                    if (kt + half >= k16Tiles) break;
                    const int gblock = k0 >> 8;
                    const int i = (k0 & 255) >> 5;
                    const int jb = (k0 & 31) + (lane & 3) * 4;
                    uint32_t qw = 0;
                    if (bRowValid) {
                        std::memcpy(&qw,
                                    rowBase +
                                            static_cast<size_t>(gblock) *
                                                    kQ4K_BYTES +
                                            16 + 32 * (i >> 1) + jb,
                                    sizeof(uint32_t));
                        qw = ((i & 1) == 0) ? (qw & 0x0F0F0F0Fu)
                                            : ((qw >> 4) & 0x0F0F0F0Fu);
                    }
                    const uint32_t a = active
                                               ? mmqLoadAFrag(xq, actStride,
                                                              token, k0, lane)
                                               : 0u;
                    mmaS8_8x8x16(D, a, qw);
                }
                // dsc/dmn of the covering 32-group gg for the two output rows
                // this lane owns.
                const int i = gg & 7;
                const int gblk = gg >> 3;
                float dsc0 = 0.0f, dmn0 = 0.0f, dsc1 = 0.0f, dmn1 = 0.0f;
                {
                    const int n = n0 + nrow0;
                    if (n < rows) {
                        const uint8_t *b = w +
                                           static_cast<size_t>(n) * rowBytes +
                                           static_cast<size_t>(gblk) *
                                                   kQ4K_BYTES;
                        const float d = __half2float(
                                *reinterpret_cast<const __half *>(b));
                        const float dmin = __half2float(
                                *reinterpret_cast<const __half *>(b + 2));
                        const uint8_t *scales = b + 4;
                        uint8_t sc, m;
                        if (i < 4) {
                            sc = scales[i] & 63u;
                            m = scales[i + 4] & 63u;
                        } else {
                            sc = static_cast<uint8_t>(
                                    (scales[i + 4] & 0xFu) |
                                    ((scales[i - 4] >> 6) << 4));
                            m = static_cast<uint8_t>(
                                    (scales[i + 4] >> 4) |
                                    ((scales[i] >> 6) << 4));
                        }
                        dsc0 = d * static_cast<float>(sc);
                        dmn0 = dmin * static_cast<float>(m);
                    }
                }
                {
                    const int n = n0 + nrow1;
                    if (n < rows) {
                        const uint8_t *b = w +
                                           static_cast<size_t>(n) * rowBytes +
                                           static_cast<size_t>(gblk) *
                                                   kQ4K_BYTES;
                        const float d = __half2float(
                                *reinterpret_cast<const __half *>(b));
                        const float dmin = __half2float(
                                *reinterpret_cast<const __half *>(b + 2));
                        const uint8_t *scales = b + 4;
                        uint8_t sc, m;
                        if (i < 4) {
                            sc = scales[i] & 63u;
                            m = scales[i + 4] & 63u;
                        } else {
                            sc = static_cast<uint8_t>(
                                    (scales[i + 4] & 0xFu) |
                                    ((scales[i - 4] >> 6) << 4));
                            m = static_cast<uint8_t>(
                                    (scales[i + 4] >> 4) |
                                    ((scales[i] >> 6) << 4));
                        }
                        dsc1 = d * static_cast<float>(sc);
                        dmn1 = dmin * static_cast<float>(m);
                    }
                }
                int32_t bsum32 = 0;
                if (active) {
                    const int8_t *blkA =
                            xq + static_cast<size_t>(token) * actStride +
                            static_cast<size_t>(gblk) * kQ8K_STRIDE;
                    const int16_t *bs = reinterpret_cast<const int16_t *>(
                            blkA + kQ8K_B);
                    bsum32 = static_cast<int>(bs[2 * i]) +
                             static_cast<int>(bs[2 * i + 1]);
                }
                main0 += dsc0 * static_cast<float>(D[0]);
                main1 += dsc1 * static_cast<float>(D[1]);
                min0 += dmn0 * static_cast<float>(bsum32);
                min1 += dmn1 * static_cast<float>(bsum32);
                D[0] = 0;
                D[1] = 0;
                if (((gg & 7) == 7) || (gg == nGroups - 1)) {
                    const float ad = active
                                             ? *reinterpret_cast<const float *>(
                                                       xq + static_cast<size_t>(token) * actStride +
                                                       static_cast<size_t>(
                                                               gg >> 3) *
                                                               kQ8K_STRIDE +
                                                       kQ8K_D)
                                             : 0.0f;
                    row0 += ad * (main0 - min0);
                    row1 += ad * (main1 - min1);
                    main0 = 0.0f;
                    main1 = 0.0f;
                    min0 = 0.0f;
                    min1 = 0.0f;
                }
            }
            if (active) {
                const int nOut0 = n0 + nrow0;
                const int nOut1 = n0 + nrow1;
                if (nOut0 < rows)
                    y[static_cast<size_t>(token) * rows + nOut0] = row0;
                if (nOut1 < rows)
                    y[static_cast<size_t>(token) * rows + nOut1] = row1;
            }
        }

        // Affine Q4_K prefill/decode GEMM on the int8 tensor cores (2026-10-29).
        // Geometry identical to kMmqGemm_8x8x16 (8 warps x 8 tokens x 8 rows),
        // but the weight is the AFFINE Q4_K decode  w = d*sc*q - dmin*m  (q in
        // 0..15, exact), which the pure-scale kMmqGemm kernel does not model.
        // Per 32-column Q8_K group the dot is
        //   sum_c w*x = d8 * ( dsc*D_int - dmn*bsum32 )
        // where D_int = sum_c q*y8 is produced by the tensor-core mma,
        // dsc = d*sc and dmn = dmin*m are the packMmaQ4K per-(n8,k16,nrow)
        // scales, bsum32 = bsums[2s]+bsums[2s+1] is the activation 32-group
        // sum, and d8 the Q8_K block scale.  main/min are kept separate per
        // 256-block and folded by d8, matching kxQ8KBlockDot<Q4_K>'s
        // d8*(ms-mns) -- the same accumulation class, so it matches
        // kQGemmKxQ8K<Q4_K> to ~1e-3 (it is NOT bit-exact: the Q8_K activation
        // quantization and the float fold differ; the engine's batch-vs-
        // sequential contract allows that, see ModelTest).
        // grid = (n8Tiles, ceil(seq / 64)); actStride = blocksPerRow*kQ8K_STRIDE.
        __global__ void __launch_bounds__(256) kMmqGemmQ4K_Affine(
                const int8_t *__restrict__ qpk, const float *__restrict__ dsc,
                const float *__restrict__ dmn, const int8_t *__restrict__ xq,
                float *__restrict__ y, int seq, int rows, int cols,
                int k16Tiles, int actStride) {
            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;
            const int nt = blockIdx.x;
            const int n0 = nt * 8;
            const int m0 = (blockIdx.y * 8 + warp) * 8;
            const int token = m0 + (lane >> 2);
            const bool active = token < seq;
            const int nrow0 = (lane & 3) * 2;
            const int nrow1 = nrow0 + 1;
            const int scRowBase = nt * k16Tiles;
            const int nGroups = (cols + 31) >> 5;

            int32_t D[2] = {0, 0};
            float main0 = 0.0f, main1 = 0.0f;
            float min0 = 0.0f, min1 = 0.0f;
            float row0 = 0.0f, row1 = 0.0f;
            for (int gg = 0; gg < nGroups; ++gg) {
                const int kt = 2 * gg;
                {
                    const uint32_t a = active ? mmqLoadAFrag(xq, actStride,
                                                             token, kt * 16, lane)
                                              : 0u;
                    const uint32_t b = mmqLoadBCell(qpk, scRowBase + kt, lane);
                    mmaS8_8x8x16(D, a, b);
                }
                if (kt + 1 < k16Tiles) {
                    const uint32_t a = active
                                               ? mmqLoadAFrag(xq, actStride,
                                                              token, (kt + 1) * 16,
                                                              lane)
                                               : 0u;
                    const uint32_t b =
                            mmqLoadBCell(qpk, scRowBase + kt + 1, lane);
                    mmaS8_8x8x16(D, a, b);
                }
                int32_t bsum32 = 0;
                if (active) {
                    const int8_t *blkA =
                            xq + static_cast<size_t>(token) * actStride +
                            static_cast<size_t>(gg >> 3) * kQ8K_STRIDE;
                    const int16_t *bs = reinterpret_cast<const int16_t *>(
                            blkA + kQ8K_B);
                    bsum32 = static_cast<int>(bs[2 * (gg & 7)]) +
                             static_cast<int>(bs[2 * (gg & 7) + 1]);
                }
                const int scIdx = (scRowBase + kt) * 8;
                main0 += dsc[scIdx + nrow0] * static_cast<float>(D[0]);
                main1 += dsc[scIdx + nrow1] * static_cast<float>(D[1]);
                min0 += dmn[scIdx + nrow0] * static_cast<float>(bsum32);
                min1 += dmn[scIdx + nrow1] * static_cast<float>(bsum32);
                D[0] = 0;
                D[1] = 0;
                if (((gg & 7) == 7) || (gg == nGroups - 1)) {
                    const float ad = active
                                             ? *reinterpret_cast<const float *>(
                                                       xq + static_cast<size_t>(token) * actStride +
                                                       static_cast<size_t>(gg >> 3) * kQ8K_STRIDE +
                                                       kQ8K_D)
                                             : 0.0f;
                    row0 += ad * (main0 - min0);
                    row1 += ad * (main1 - min1);
                    main0 = 0.0f;
                    main1 = 0.0f;
                    min0 = 0.0f;
                    min1 = 0.0f;
                }
            }
            if (active) {
                const int nOut0 = n0 + nrow0;
                const int nOut1 = n0 + nrow1;
                if (nOut0 < rows)
                    y[static_cast<size_t>(token) * rows + nOut0] = row0;
                if (nOut1 < rows)
                    y[static_cast<size_t>(token) * rows + nOut1] = row1;
            }
        }

        // ------------------------------------------------------------------
        // Q8_1 activation path (2026-10-08, Q8_1-Q2_K experiment). It differs
        // from our existing Q8_K path (256-wide, sign-forced d<0, per-16 bsums)
        // and from the float path (no quantize at all). The dot: dot =
        // sum_{j=0..3} d8_j * ( dq.(sc&0xF).dp4a(pk_j, y_j) -
        // dmin.(sc>>4).dp4a(0x01010101, y_j) ) reproduces the float kernel
        // exactly (all-integer inside, one float mad per group), with 1 B/elem
        // activation instead of 4 (removes the fp32-scalar-per-element load +
        // fmaf that pins the float 4xW).
        // ------------------------------------------------------------------
        static constexpr uint32_t kQ8_1_STRIDE = 40;// 2B d + 2B s + 32 int8, padded
        // grid = cols/32, block = 32 (one warp per 32-block).
        __global__ void kQuantizeQ8_1(const float *__restrict__ x,
                                      uint8_t *__restrict__ y1, uint32_t cols) {
            const uint32_t b = blockIdx.x;
            const uint32_t e0 = b * 32u;
            const uint32_t t = threadIdx.x;
            uint8_t *blk = y1 + static_cast<uint64_t>(b) * kQ8_1_STRIDE;
            int8_t *qs = reinterpret_cast<int8_t *>(blk + 4);
            const float xi = (e0 + t < cols) ? x[e0 + t] : 0.0f;
            float amax = fabsf(xi);
            float sum = xi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            const float d = amax / 127.0f;
            qs[t] = (amax == 0.0f)
                            ? static_cast<int8_t>(0)
                            : static_cast<int8_t>(roundf(xi / d));
            if (t == 0) {
                __half *ds = reinterpret_cast<__half *>(blk);
                ds[0] = __float2half(d);
                ds[1] = __float2half(sum);
            }
        }

        // 4-rows-per-warp Q2_K x Q8_1 dp4a GEMV.  Geometry identical to
        // kQGemvQ2KxW4 (all 32 lanes issue u32 weight loads, 8 lanes/row,
        // 4 rows/warp, no smem); the per-element body is 1 u32 weight load +
        // 1 u32 Q8_1 load + 2 dp4a + 1 float mad (vs the float kernel's 1 u32
        // weight load + 4 fp32 activation loads + 4 fmaf).  4 Q2_K blocks are
        // batched into independent accumulators (memory-level parallelism).
        __global__ void __launch_bounds__(256) kQGemvQ2KxQ81_4xW(
                const uint8_t *__restrict__ w, const uint8_t *__restrict__ y1,
                float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow,
                uint32_t rowBytes) {
            const uint32_t warpRowBase = blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * 4u;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t isub = I >> 2u;
            const uint32_t row = base + (lane >> 3u);
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            float sumf = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 4u; ++bb) {
                    const uint8_t *blkW =
                            rp + static_cast<uint64_t>(b + bb) * kQ2K_BYTES;
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
                        std::memcpy(&w4, q + n * 32u + 4u * I, sizeof(uint32_t));
                        uint64_t scU8;
                        std::memcpy(&scU8, sc + n * 8u, sizeof(uint64_t));
#pragma unroll
                        for (uint32_t j = 0; j < 4u; ++j) {
                            const uint8_t *blkY =
                                    y1 + static_cast<uint64_t>(
                                                 (b + bb) * 8u + n * 4u + j) *
                                                 kQ8_1_STRIDE;
                            const float d8 = __half2float(
                                    *reinterpret_cast<const __half *>(blkY));
                            uint32_t y4;
                            std::memcpy(&y4, blkY + 4u + 4u * I,
                                        sizeof(uint32_t));
                            const uint32_t pk = (w4 >> (2u * j)) & 0x03030303u;
                            const int pdot = __dp4a(static_cast<int>(pk),
                                                    static_cast<int>(y4), 0);
                            const int sumy = __dp4a(0x01010101,
                                                    static_cast<int>(y4), 0);
                            const uint8_t scv = static_cast<uint8_t>(
                                    (scU8 >> (8u * (2u * j + isub))) & 0xFFu);
                            slot += d8 * (dq * static_cast<float>(scv & 0xFu) *
                                                  static_cast<float>(pdot) -
                                          dmin * static_cast<float>(scv >> 4u) *
                                                  static_cast<float>(sumy));
                        }
                    }
                }
                sumf = a0 + sumf;
                sumf = a1 + sumf;
                sumf = a2 + sumf;
                sumf = a3 + sumf;
            }
            for (; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * kQ2K_BYTES;
                const float dq = __half2float(
                        *reinterpret_cast<const __half *>(blkW + 80));
                const float dmin = __half2float(
                        *reinterpret_cast<const __half *>(blkW + 82));
                const uint8_t *sc = blkW;
                const uint8_t *q = blkW + 16;
#pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    uint32_t w4;
                    std::memcpy(&w4, q + n * 32u + 4u * I, sizeof(uint32_t));
                    uint64_t scU8;
                    std::memcpy(&scU8, sc + n * 8u, sizeof(uint64_t));
#pragma unroll
                    for (uint32_t j = 0; j < 4u; ++j) {
                        const uint8_t *blkY =
                                y1 + static_cast<uint64_t>(b * 8u + n * 4u + j) *
                                             kQ8_1_STRIDE;
                        const float d8 = __half2float(
                                *reinterpret_cast<const __half *>(blkY));
                        uint32_t y4;
                        std::memcpy(&y4, blkY + 4u + 4u * I, sizeof(uint32_t));
                        const uint32_t pk = (w4 >> (2u * j)) & 0x03030303u;
                        const int pdot = __dp4a(static_cast<int>(pk),
                                                static_cast<int>(y4), 0);
                        const int sumy = __dp4a(0x01010101,
                                                static_cast<int>(y4), 0);
                        const uint8_t scv = static_cast<uint8_t>(
                                (scU8 >> (8u * (2u * j + isub))) & 0xFFu);
                        sumf += d8 *
                                (dq * static_cast<float>(scv & 0xFu) *
                                         static_cast<float>(pdot) -
                                 dmin * static_cast<float>(scv >> 4u) *
                                         static_cast<float>(sumy));
                    }
                }
            }
            const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
            const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
            if (I == 0u) out[row] = rr;
        }

        // ------------------------------------------------------------------
        // Split-K Q2_K x Q8_1 dp4a GEMV (2026-10-09,
        // TINYCODER_Q2K_Q81_SPLITK=1, requires TINYCODER_Q2K_Q81=1). Here the
        // block is 64 threads = 2 warps, each warp is 4 eight-lane stripes (the
        // 8-lane-of-4 column coverage reused from kQGemvQ2KxQ81_4xW), so ONE
        // output row's K blocks are split 8 ways (stripe s takes b = s, s+8,
        // s+16, ...). Each lane's dependent chain is blocksPerRow/8 instead of
        // blocksPerRow. A 4-warp variant compiles from the same source but its
        // different summation order flips sampled near-ties, so only the 2-warp
        // shape is offered.
        __global__ void __launch_bounds__(64) kQGemvQ2KxQ81_SplitK(
                const uint8_t *__restrict__ w, const uint8_t *__restrict__ y1,
                float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow,
                uint32_t rowBytes) {
            const uint32_t lane = threadIdx.x;      // 0..31
            const uint32_t warp = threadIdx.y;      // 0..1
            const uint32_t I = lane & 7u;           // 8-lane column group
            const uint32_t stripe = lane >> 3u;     // 0..3 (K stripe in warp)
            const uint32_t nSplit = blockDim.y * 4u;// 8 (2 warps; see below)
            const uint32_t ksplit = warp * 4u + stripe;
            const uint32_t row = blockIdx.x;
            if (row >= rows) return;
            const uint32_t isub = (4u * I < 16u) ? 0u : 1u;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            float acc = 0.0f;
            for (uint32_t b = ksplit; b < blocksPerRow; b += nSplit) {
                const uint8_t *blkW =
                        rp + static_cast<uint64_t>(b) * kQ2K_BYTES;
                const float dq = __half2float(
                        *reinterpret_cast<const __half *>(blkW + 80));
                const float dmin = __half2float(
                        *reinterpret_cast<const __half *>(blkW + 82));
                const uint8_t *sc = blkW;
                const uint8_t *q = blkW + 16;
#pragma unroll
                for (uint32_t n = 0; n < 2u; ++n) {
                    uint32_t w4;
                    std::memcpy(&w4, q + n * 32u + 4u * I, sizeof(uint32_t));
                    uint64_t scU8;
                    std::memcpy(&scU8, sc + n * 8u, sizeof(uint64_t));
#pragma unroll
                    for (uint32_t j = 0; j < 4u; ++j) {
                        const uint8_t *blkY =
                                y1 + static_cast<uint64_t>(
                                             b * 8u + n * 4u + j) *
                                             kQ8_1_STRIDE;
                        const float d8 = __half2float(
                                *reinterpret_cast<const __half *>(blkY));
                        uint32_t y4;
                        std::memcpy(&y4, blkY + 4u + 4u * I, sizeof(uint32_t));
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
            }
            // 8-lane tree within the stripe (xor 4/1/2 stays inside the block).
            acc += __shfl_xor_sync(0xffffffffu, acc, 4);
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            acc += __shfl_xor_sync(0xffffffffu, acc, 2);
            __shared__ float red[16];
            if (I == 0u) red[ksplit] = acc;
            __syncthreads();
            if (threadIdx.x == 0u && warp == 0u) {
                float s = 0.0f;
                for (uint32_t i = 0; i < nSplit; ++i) s += red[i];
                out[row] = s;
            }
        }

        // QI2_K = QI3_K = 16, QR2_K = QR3_K = 4, QI8_1 = 8 for QK_K=256.
        // -------- Q2_K x Q8_1 --------------------------------------------
        __global__ void __launch_bounds__(64, 8) kQGemvQ2KxQ81_Mmvq(
                const uint8_t *__restrict__ w, const uint8_t *__restrict__ y1,
                float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow) {
            constexpr int nwarps = 2;
            constexpr int QK = 256, QI = 16, QR = 4, QI8_1 = 8;
            const int row = blockIdx.x;
            if (row >= rows) return;
            const int t = threadIdx.x;// lane 0..31
            const int tid = threadIdx.y * 32 + t;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * (6u * 84u);
            const int blocks_per_iter = nwarps * 32 / QI;// 4
            float tmp = 0.0f;
            for (int kbx = tid / QI; kbx < static_cast<int>(blocksPerRow);
                 kbx += blocks_per_iter) {
                const int kby = kbx * (QK / 32);// *8
                const int kqs = tid % QI;
                const uint8_t *bq = rp + static_cast<uint64_t>(kbx) * 84u;
                // vec_dot_q2_K_q8_1
                const int bq8_offset = QR * (kqs / QI8_1);
                const int scale_offset =
                        kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1 / 2);
                const uint8_t *scales = bq + scale_offset;// scales @ block+0
                const uint8_t *qs = bq + 16;              // qs @ block+16
                int v;
                std::memcpy(&v, qs + 4 * kqs, sizeof(int));// get_int_b4(qs,kqs)
                int u[QR];
                float d8[QR];
#pragma unroll
                for (int i = 0; i < QR; ++i) {
                    const uint8_t *by = y1 +
                                        static_cast<uint64_t>(kby + bq8_offset + i) * 40u;
                    std::memcpy(&u[i], by + 4 + 4 * (kqs % QI8_1), sizeof(int));
                    d8[i] = __half2float(
                            *reinterpret_cast<const __half *>(by));
                }
                float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
                for (int i = 0; i < QR; ++i) {
                    const int sc = scales[2 * i];
                    const int vi = (v >> (2 * i)) & 0x03030303;
                    sumf_d += d8[i] *
                              (__dp4a(vi, u[i], 0) * (sc & 0xF));
                    int m = sc >> 4;
                    m |= m << 8;
                    m |= m << 16;
                    sumf_m += d8[i] * __dp4a(m, u[i], 0);
                }
                const float2 dm2f = __half22float2(
                        *reinterpret_cast<const __half2 *>(bq + 80));
                tmp += dm2f.x * sumf_d - dm2f.y * sumf_m;
            }
            // cross-warp reduce (nwarps-1 partials) then warp tree.
            __shared__ float red[32];
            if (threadIdx.y == 1) red[t] = tmp;
            __syncthreads();
            if (threadIdx.y == 0) {
                float v = tmp + red[t];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_xor_sync(0xffffffffu, v, off);
                if (t == 0) out[row] = v;
            }
        }

        // Same bytes, same values. ADOPTED as the default (2026-10-10):
        // interleaved A/B tg128@pp512, 5 rounds: 211.5-212.1 -> 249.9-251.7
        // tok/s, +18.0..+19.0 %, 5/5 wins, 4/4 prompt stream parity.
        // TINYCODER_MMVQ_WIDE=0 restores the memcpy byte-load form (the
        // pre-9.23 default).
        inline bool mmvqWide() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_MMVQ_WIDE");
                return (e == nullptr || std::atoi(e) != 0);
            }();
            return v;
        }

        // 2026-10-11: IQ2_S / IQ3_XXS / IQ3_S decode GEMVs were ALU-bound on the
        // scalar grid-lookup dot (ncu: IQ2_S gate+up 867 us/layer at DRAM 7 %).
        // ADOPTED default-on: the __dp4a + XOR-subtract-sign lane body (llama's
        // vec_dot_iq*_q8_1 structure) is bit-identical and measured
        // +5.6 % (1.5B iq3_xxs, 3/3) and +7.6 % (7B iq3_xxs, 2/2), parity
        // byte-identical on 4/4 prompts.  TINYCODER_IQ_DP4A=0 restores the
        // scalar body for A/B.
        inline bool iqDp4a() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_IQ_DP4A");
                return (e == nullptr || std::atoi(e) != 0);
            }();
            return v;
        }

        // 2026-10-11: route IQ2_XS (GGML type 17) through the Q8_K integer
        // GEMV family instead of the float dequant-dot kernels
        // (kQGemvFusedGU_IQ2XS 869 us/layer at 25 % occupancy, kQGemv 499 us at
        // 83 % LG-throttle).  ADOPTED default-on after A/B; TINYCODER_IQ2XS_INT=0
        // restores the float path for A/B and for greedy-stream parity checks.
        inline bool iq2xsInt() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_IQ2XS_INT");
                return (e == nullptr || std::atoi(e) != 0);
            }();
            return v;
        }

        // 2026-10-11: the IQ decode GEMVs are LG-throttle bound (42-53 % of
        // stall cycles) on the divergent codebook lookups.  The fused gate+up
        // kernel now stages the small grid table (1-8 KB) in shared memory so
        // those reads use the MIO path.  ADOPTED default-on: +7 % (7B iq2_s,
        // 2/2) and +2 % (7B iq3_xxs, 2/2).  TINYCODER_IQ_GRID_SMEM=0 restores
        // the global grid for A/B.
        inline bool iqGridSmem() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_IQ_GRID_SMEM");
                return (e == nullptr || std::atoi(e) != 0);
            }();
            return v;
        }

        // 2026-10-12: 4-rows-per-warp IQ Q8_K GEMV (kQGemvQ8K_4xW).  The
        // generic kQGemvQ8K runs one row per warp, so every row re-reads the
        // whole Q8_K activation and only one weight block is in flight per
        // warp.  The 4xW kernel gives each 8-lane group its own row and each
        // lane one ib32 sub-block: the activation load is shared across the
        // warp's 4 rows (the L2 re-read cost of 8960 rows is eliminated) and 4
        // weight blocks stay in flight.  Bit-exact with kQGemvQ8K (per-row
        // block-order float accumulation unchanged).  Requires rows%4==0.
        // MEASURED 2026-10-12: REJECTED (default OFF).  The IQ decode GEMV is
        // codebook-lookup (LG-throttle) bound, not activation-re-read bound:
        // the 4xW form multiplies the per-lane divergent grid lookups without
        // reducing the total, and loses warp parallelism.  A/B on the 1.5B
        // iq3_xxs: 162.0 tok/s (4xW) vs 174.2 (generic) -- a -7% regression.
        // Kept for the record; TINYCODER_IQ_4XW=1 re-enables it.
        inline bool iq4xw() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_IQ_4XW");
                return (e != nullptr && std::atoi(e) != 0);
            }();
            return v;
        }

        // 2026-10-29: 4-rows-per-warp Q4_K Q8_K decode GEMV (kQGemvQ4KxQ8K_4xW).
        // The generic kQGemvKxQ8K<kTypeQ4K> runs one row per warp (only 8 of 32
        // lanes issue weight loads) and scalar-loads each 32-byte q's strip, so
        // the LARGEST quant class of the dense qwen35 models (Q4_K ~2.7 GB of
        // Ornith-1.5-9B's 5.9 GB) has no 4xW kernel while Q5_K/Q6_K do.  The
        // new kernel mirrors kQGemvQ5KxQ8K_4xW (4 rows/warp, all 32 lanes load)
        // with an 8x __dp4a integer body; bit-exact with the generic kernel.
        // DEFAULT ON: TINYCODER_Q4K_4XW=0 restores the generic kernel for A/B.
        inline bool q4k4xw() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_Q4K_4XW");
                return (e == nullptr || std::atoi(e) != 0);
            }();
            return v;
        }

        // 2026-10-29: warps per block for the Q4_K 4xW decode GEMV.  The 4xW
        // grid is rows/4/warps, so for the dense-qwen35 shapes (rows 4096/12288)
        // the default 8-warp block yields only ~128-384 blocks -- often under
        // one wave on the 68-SM 2080 Ti, i.e. occupancy-starved.  Smaller
        // blocks give more blocks (better SM fill) at more block-fixed cost.
        // TINYCODER_Q4K_BSY=N overrides the default 8.
        inline int q4kBsy() {
            static const int v = [] {
                const char *e = std::getenv("TINYCODER_Q4K_BSY");
                int n = (e != nullptr) ? std::atoi(e) : 8;
                return (n >= 1 && n <= 32) ? n : 8;
            }();
            return v;
        }

        // 2026-10-29: kQ35L2NormRepeat previously recomputed the same headK-wide
        // double L2 norm for every element of a slice's headK*nVHeads output
        // (the per-slice norm depends only on the value head).  The fast path
        // computes each slice's iq/ik once (shared memory), bit-identically.
        // DEFAULT ON: TINYCODER_L2NORM_SMEM=0 restores the recompute body.
        inline bool l2NormSmem() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_L2NORM_SMEM");
                return (e == nullptr || std::atoi(e) != 0);
            }();
            return v;
        }

        // 2026-10-29: int8 tensor-core (mma.s8.m8n8k16) prefill GEMM for Q4_K.
        // The affine Q4_K path (kMmqGemmQ4K_AffineRaw) reads the RAW GGML
        // weights (no pre-pack), costs zero extra VRAM, and feeds the same
        // Q8_K activations the integer kQGemmKxQ8K<Q4_K> path already produces.
        // It is NOT bit-exact with that path (different float fold), so the
        // batch-vs-sequential contract (top-1 coherence) is the acceptance bar;
        // DEFAULT OFF until an A/B shows a clear prefill win.
        // TINYCODER_MMQ_Q4K=1 enables; TINYCODER_MMQ_Q4K=0 forces it off.
        inline bool mmqQ4k() {
            static const bool v = [] {
                const char *e = std::getenv("TINYCODER_MMQ_Q4K");
                return (e != nullptr && std::atoi(e) != 0);
            }();
            return v;
        }

        // Runtime gate: does THIS device expose the int8 m8n8k16 mma the MMQ
        // kernels need (Turing sm_75+)?  Cached after the first query.  The
        // kernels are only COMPILED for the gencode arch list; this is the
        // runtime half (so a pre-Turing device never launches them).
        inline bool mmqS8Ok() {
            static const int v = [] {
                int dev = 0;
                if (cudaGetDevice(&dev) != cudaSuccess) return 0;
                int major = 0, minor = 0;
                if (cudaDeviceGetAttribute(
                            &major, cudaDevAttrComputeCapabilityMajor, dev) !=
                    cudaSuccess)
                    return 0;
                if (cudaDeviceGetAttribute(
                            &minor, cudaDevAttrComputeCapabilityMinor, dev) !=
                    cudaSuccess)
                    return 0;
                return (major > 7 || (major == 7 && minor >= 5)) ? 1 : 0;
            }();
            return v != 0;
        }

        // bytes in flight -- not a different inner loop. Dot values are the
        // Q8_1 integer domain (~1e-6 from the float path), so parity is
        // asserted via the deterministic greedy stream like the other mmvq
        // adoptions.
        template<bool WIDE = false>
        __global__ void __launch_bounds__(64, 8) kQGemvQ2KxQ81_GLU(
                const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
                const uint8_t *__restrict__ y1, float *__restrict__ out,
                uint32_t rows, uint32_t blocksPerRow) {
            constexpr int QK = 256, QI = 16, QR = 4, QI8_1 = 8;
            const int row = blockIdx.x;
            if (row >= rows) return;
            const int t = threadIdx.x;
            const int tid = threadIdx.y * 32 + t;
            const uint8_t *rg = wg + static_cast<uint64_t>(row) * (6u * 84u);
            const uint8_t *ru = wu + static_cast<uint64_t>(row) * (6u * 84u);
            const int blocks_per_iter = 4;// nwarps(2) * 32 / QI
            float tmp = 0.0f, tmpg = 0.0f;
            for (int kbx = tid / QI; kbx < static_cast<int>(blocksPerRow);
                 kbx += blocks_per_iter) {
                const int kby = kbx * (QK / 32);
                const int kqs = tid % QI;
                const uint8_t *bg = rg + static_cast<uint64_t>(kbx) * 84u;
                const uint8_t *bu = ru + static_cast<uint64_t>(kbx) * 84u;
                // vec_dot_q2_K_q8_1, shared Q8_1 operand loads.
                const int bq8_offset = QR * (kqs / QI8_1);
                const int scale_offset =
                        kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1 / 2);
                const uint8_t *scales_g = bg + scale_offset;// scales @ block+0
                const uint8_t *scales_u = bu + scale_offset;
                const uint8_t *qs_g = bg + 16;// qs @ block+16
                const uint8_t *qs_u = bu + 16;
                int vg, vu;
                if constexpr (WIDE) {
                    vg = *reinterpret_cast<const int *>(qs_g + 4 * kqs);
                    vu = *reinterpret_cast<const int *>(qs_u + 4 * kqs);
                } else {
                    std::memcpy(&vg, qs_g + 4 * kqs, sizeof(int));
                    std::memcpy(&vu, qs_u + 4 * kqs, sizeof(int));
                }
                int u[QR];
                float d8[QR];
#pragma unroll
                for (int i = 0; i < QR; ++i) {
                    const uint8_t *by = y1 +
                                        static_cast<uint64_t>(kby + bq8_offset + i) * 40u;
                    if constexpr (WIDE) {
                        u[i] = *reinterpret_cast<const int *>(
                                by + 4 + 4 * (kqs % QI8_1));
                    } else {
                        std::memcpy(&u[i], by + 4 + 4 * (kqs % QI8_1),
                                    sizeof(int));
                    }
                    d8[i] = __half2float(
                            *reinterpret_cast<const __half *>(by));
                }
                float sd = 0.0f, sm = 0.0f, sgd = 0.0f, sgm = 0.0f;
#pragma unroll
                for (int i = 0; i < QR; ++i) {
                    const int scg = scales_g[2 * i];
                    const int scu = scales_u[2 * i];
                    const int vig = (vg >> (2 * i)) & 0x03030303;
                    const int viu = (vu >> (2 * i)) & 0x03030303;
                    sd += d8[i] * (__dp4a(vig, u[i], 0) * (scg & 0xF));
                    sgd += d8[i] * (__dp4a(viu, u[i], 0) * (scu & 0xF));
                    int mg = scg >> 4;
                    mg |= mg << 8;
                    mg |= mg << 16;
                    int mu = scu >> 4;
                    mu |= mu << 8;
                    mu |= mu << 16;
                    sm += d8[i] * __dp4a(mg, u[i], 0);
                    sgm += d8[i] * __dp4a(mu, u[i], 0);
                }
                const float2 dmg = __half22float2(
                        *reinterpret_cast<const __half2 *>(bg + 80));
                const float2 dmu = __half22float2(
                        *reinterpret_cast<const __half2 *>(bu + 80));
                tmp += dmg.x * sd - dmg.y * sm;
                tmpg += dmu.x * sgd - dmu.y * sgm;
            }
            // Cross-warp reduce for both partials, then the warp tree.
            __shared__ float red[32], redg[32];
            if (threadIdx.y == 1) {
                red[t] = tmp;
                redg[t] = tmpg;
            }
            __syncthreads();
            if (threadIdx.y == 0) {
                // v = gate partial-sum tree, vg = up partial-sum tree (the
                // loop's int vg/vu are out of scope here).
                float g = tmp + red[t];
                float u = tmpg + redg[t];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    g += __shfl_xor_sync(0xffffffffu, g, off);
                    u += __shfl_xor_sync(0xffffffffu, u, off);
                }
                if (t == 0) {
                    // silu(gate)*up -- the exact kSiluMul expression/order.
                    out[row] = (g / (1.0f + __expf(-g))) * u;
                }
            }
        }

        // -------- Q3_K x Q8_1 --------------------------------------------
        template<bool WIDE = false>
        __global__ void __launch_bounds__(64, 8) kQGemvQ3KxQ81_Mmvq(
                const uint8_t *__restrict__ w, const uint8_t *__restrict__ y1,
                float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow,
                uint32_t rowBytes, float *__restrict__ resid) {
            constexpr int nwarps = 2;
            constexpr int QK = 256, QI = 16, QR = 4, QI8_1 = 8;
            const int row = blockIdx.x;
            if (row >= rows) return;
            const int t = threadIdx.x;
            const int tid = threadIdx.y * 32 + t;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            const int blocks_per_iter = nwarps * 32 / QI;// 4
            float tmp = 0.0f;
            for (int kbx = tid / QI; kbx < static_cast<int>(blocksPerRow);
                 kbx += blocks_per_iter) {
                const int kby = kbx * (QK / 32);
                const int kqs = tid % QI;
                const uint8_t *bq = rp + static_cast<uint64_t>(kbx) * 110u;
                // vec_dot_q3_K_q8_1
                const int bq8_offset = QR * (kqs / (QI / 2));// 4*(kqs/8)
                const int scale_offset =
                        kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1 / 2);
                const uint8_t *scales = bq + 96;// scales @ +96
                const float d3 = __half2float(
                        *reinterpret_cast<const __half *>(bq + 108));
                const uint8_t *qs = bq + 32;// qs @ +32
                // get_int_b2(x, i32) == ((const int *)x)[i32] == the 32-bit
                // little-endian word at byte offset 4*i32 (NOT a 16-bit read!)
                // -- it assembles x16[2i] | (x16[2i+1]<<16), so it is identical
                // to get_int_b4.  Reading only 2 bytes here would zero the high
                // 16 bits and corrupt the 0x03030303/0x04040404 masks, dropping
                // half the elements per kqs (the parity bug).
                // WIDE (down variant): the Q3_K row stride is 110 B, so odd
                // (row,kbx) pairs leave the weight ints 2-mod-4 — a scalar 4-B
                // load there is NOT legal (unlike GLU's 84-B stride, where the
                // int deref is exact and was verified by the 4/4 stream).  The
                // aligned-safe wide form is 2x LDG.E.U16 per int (all these
                // addresses are even), halving the byte-load count.
                int vl, vh;
                if constexpr (WIDE) {
                    const unsigned short *vl16 =
                            reinterpret_cast<const unsigned short *>(
                                    qs + 4 * kqs);
                    vl = vl16[0] | (static_cast<int>(vl16[1]) << 16);
                    const unsigned short *vh16 =
                            reinterpret_cast<const unsigned short *>(
                                    bq + 4 * (kqs % (QI / 2)));
                    vh = vh16[0] | (static_cast<int>(vh16[1]) << 16);
                } else {
                    std::memcpy(&vl, qs + 4 * kqs,
                                sizeof(int));// get_int_b2(qs, kqs)
                    std::memcpy(&vh, bq + 4 * (kqs % (QI / 2)),
                                sizeof(int));// get_int_b2(hmask, kqs%8)
                }
                vh = ~vh >> bq8_offset;
                int u[QR];
                float d8[QR];
#pragma unroll
                for (int i = 0; i < QR; ++i) {
                    const uint8_t *by = y1 +
                                        static_cast<uint64_t>(kby + bq8_offset + i) * 40u;
                    if constexpr (WIDE) {
                        // y blocks stride 40 B: always 4-aligned.
                        u[i] = *reinterpret_cast<const int *>(
                                by + 4 + 4 * (kqs % QI8_1));
                    } else {
                        std::memcpy(&u[i], by + 4 + 4 * (kqs % QI8_1),
                                    sizeof(int));
                    }
                    d8[i] = __half2float(
                            *reinterpret_cast<const __half *>(by));
                }
                float sumf = 0.0f;
#pragma unroll
                for (int i = 0; i < QR; ++i) {
                    const int isc = scale_offset + 2 * i;
                    const int isc_low = isc % (QK / 32);
                    const int sc_shift_low = 4 * (isc / (QK / 32));
                    const int sc_low =
                            (scales[isc_low] >> sc_shift_low) & 0xF;
                    const int isc_high = isc % (QK / 64);
                    const int sc_shift_high = 2 * (isc / (QK / 64));
                    const int sc_high =
                            ((scales[(QK / 32) + isc_high] >> sc_shift_high) & 3)
                            << 4;
                    const int sc = (sc_low | sc_high) - 32;
                    const int vil = (vl >> (2 * i)) & 0x03030303;
                    const int vih = ((vh >> i) << 2) & 0x04040404;
                    const int vi = __vsubss4(vil, vih);
                    sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
                }
                tmp += d3 * sumf;
            }
            __shared__ float red[32];
            if (threadIdx.y == 1) red[t] = tmp;
            __syncthreads();
            if (threadIdx.y == 0) {
                float v = tmp + red[t];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_xor_sync(0xffffffffu, v, off);
                if (t == 0) {
                    out[row] = v;
                    // Folded residual (2026-10-10): kAddResidual's
                    // hidden[i] = hidden[i] + out[i] is THIS float add, so
                    // passing s.hidden here keeps the residual stream
                    // byte-identical while deleting one launch per layer.
                    if (resid != nullptr) resid[row] = resid[row] + v;
                }
            }
        }

        // ------------------------------------------------------------------
        // Q8_0 x Q8_32 (activation Q8_32) integer GEMV — bit-exact with the
        // CPU's matMulVecBatchQ8_0_Q8K_AVX2.
        //
        // The CPU reference quantizes the fp32 activation per 32-wide block
        // (Q8_0's natural block width — NOT the 256-wide Q8_K blocking used by
        // the IQ* types above), with:
        //   amax/maxv = signed abs-max scan over the block's 32 floats
        //   iscale = (amax>0) ? -127/maxv : 0;  xq = clamp(lrintf(x*iscale))
        //   d8 = (amax>0) ? 1/iscale : 0   (lrintf = nearest, ties to even)
        // and accumulates the dot per 32-block via the sign-substitution
        // identity (|xq| in [0,127] as the unsigned slot, w*sign(xq) as the
        // signed slot) — an EXACT int8x8->int16, pairwise int16->int32 madd,
        // THEN a float hsum of d*d8*sumi_group.  The float side is the AVX
        // tree; the ONLY roundings are d*d8 (fp16 halfToFloat d x fp32 d8) and
        // the per-group float mults.  Mirrored below:
        //   * kQuantizeQ8_0x32: per-32-block Q8_0 activation quantize
        //   * kQGemvQ8_0xQ8K: warp-per-row integer madd GEMV equivalent to
        //     one AVX batch row (the CPU's per-block float tree IS the same
        //     sequential d*d8*sum group accumulation, so lane-0 serial
        //     accumulation reproduces it bit-exactly).
        // 64-byte block stride: real data is 2-byte d + 32 int8 q (34 bytes at
        // offsets 0..33), but the float d8 must be 4-byte (ideally 16-byte)
        // aligned for `*reinterpret_cast<const float *>` reads in
        // kQGemvQ8_0xQ8K.  A 34-byte stride puts block b's d at offset b*34
        // -- 2 bytes out of alignment for every odd block -- and every odd
        // block's kQGemvQ8_0xQ8K read faults with CUDA_ERROR_MISALIGNED_ADDRESS
        // (sticky: falls back to CPU for the whole session).  Pad to 64 so the
        // float lands at offset b*64 (16-aligned) and the pad bytes after the
        // 34 used ones are never read (kQGemvQ8_0xQ8K's blkA indexing is
        // strictly within the 34 used bytes, and the writer memset-fills the
        // whole block so no stale-data hazard exists).
        static constexpr uint32_t kQ8_0x32_STRIDE = 64;
        // grid = blocksPerRow (= cols/32), block = 32 threads.  Bit-exact with
        // matMulVecBatchQ8_0_Q8K_AVX2's quantize loop: the CPU scans the block
        // SERIALLY (strict `>` keeps the FIRST abs-max, whose SIGNED value
        // becomes maxv), so lane 0 reproduces that exact scan and broadcasts
        // (amax, maxv) via shuffles; every lane then quantizes its own column
        // with iscale = -127/maxv, xq = clamp(lrintf(x*iscale), -127, 127),
        // d8 = 1/iscale (0 for the all-zero block).
        // Layout: [0..3] fp32 d8 (written only by lane 0), [4..35] the 32 int8
        // column values (lane -> offset 4+lane), pad [36..63] zeroed.
        // NOTE (2026-09-14): the OLD layout put the int8 columns at 2+lane,
        // which OVERLAPPED the 4-byte fp32 d8 at [0..3]: lane 0's float store
        // (program-ordered last) clobbered columns 0 and 1 of every block with
        // d8 byte garbage -- the root cause of a ~6% column corruption that
        // made the Q8_0 GEMV diverge 5-80% from the CPU AVX2 reference
        // (GPU-vs-CPU logits parity failure on qwen35moe).  Bytes are now
        // placed strictly after the float.
        __global__ void kQuantizeQ8_0x32(const float *__restrict__ x,
                                         uint8_t *__restrict__ q32,
                                         uint32_t cols) {
            const uint32_t b = blockIdx.x;
            const uint32_t lane = threadIdx.x;
            const uint32_t cidx = b * 32u + lane;
            if (cidx >= cols) {
                // Pad lanes (cols multiple of 32 for all real GEMVs; pad the
                // output block to zero so a partial final block contributes 0,
                // exactly like the CPU's 32-block padding).
                q32[b * kQ8_0x32_STRIDE + 4u + lane] = 0;
                // Zero the pad block for determinism.  NOTE: strictly within
                // [36..63] -- lane<30 would write offsets 64/65, i.e. the NEXT
                // block's d8 low 2 bytes (nondeterministic race with that
                // block's parallel lane-0 float store).
                if (lane < 28u) {
                    q32[b * kQ8_0x32_STRIDE + 36u + lane] = 0;
                }
                if (lane == 0)
                    *reinterpret_cast<float *>(q32 + b * kQ8_0x32_STRIDE) = 0.0f;
                return;
            }
            // Lane 0 serial scan over the block's 32 floats (first-wins max).
            float amax = 0.0f, maxv = 0.0f;
            if (lane == 0) {
                const float *xs = x + b * 32u;
                for (uint32_t j = 0; j < 32u; ++j) {
                    const float av = fabsf(xs[j]);
                    if (av > amax) {
                        amax = av;
                        maxv = xs[j];
                    }
                }
            }
            amax = __shfl_sync(0xffffffffu, amax, 0);
            maxv = __shfl_sync(0xffffffffu, maxv, 0);
            if (amax == 0.0f) {
                q32[b * kQ8_0x32_STRIDE + 4u + lane] = 0;
                // Zero the pad block for determinism.
                if (lane < 28u) {
                    q32[b * kQ8_0x32_STRIDE + 36u + lane] = 0;
                }
                if (lane == 0)
                    *reinterpret_cast<float *>(q32 + b * kQ8_0x32_STRIDE) = 0.0f;
                return;
            }
            // IEEE fp32 division (NOT the fast approximate reciprocal nvcc
            // otherwise emits for a bare `1.0f / x` -- measured 1-ULP off from
            // the CPU's divss, which propagates through all blocks as ~1e-3
            // row error).  The CPU reference computes `iscale = -127.0f/maxv`
            // then `d8 = 1.0f/iscale` with IEEE division twice; mirror exactly.
            const float iscale = __fdiv_rn(-127.0f, maxv);
            int vq = static_cast<int>(rintf(iscale * x[cidx]));
            if (vq > 127) vq = 127;
            if (vq < -127) vq = -127;
            q32[b * kQ8_0x32_STRIDE + 4u + lane] =
                    static_cast<uint8_t>(static_cast<int8_t>(vq));
            // Zero the pad block for determinism (inside [36..63] only).
            if (lane < 28u) {
                q32[b * kQ8_0x32_STRIDE + 36u + lane] = 0;
            }
            if (lane == 0)
                *reinterpret_cast<float *>(q32 + b * kQ8_0x32_STRIDE) =
                        __fdiv_rn(1.0f, iscale);
        }

        // Q8_0 weights x Q8_32 activation: warp-per-row, exact int8 madd
        // (bit-exact with matMulVecBatchQ8_0_Q8K_AVX2).  Per 32-wide block, the
        // AVX reference computes the EXACT integer dot per 4-column group
        // (maddubs + pairwise int16->int32 madd of the sign-substituted
        // lanes), then the float tree:
        //   t_g = (d*d8)*s_g           8 SEPARATE roundings (g=0..7)
        //   u_g = t_g + t_{g+4}        pairs (0-4, 1-5, 2-6, 3-7)
        //   v0 = u_0 + u_1; v1 = u_2 + u_3     (hadd 1)
        //   blk = v0 + v1                      (hadd 2)
        // and the block result is added SERIALLY into the row accumulator with
        // plain mul+add (no fmaf — fmaf differs by 1 ulp per block,
        // compounding through decode and flipping near-tie rows).
        //
        // PERFORMANCE NOTE (measured 2026-09-13): the previous version ran the
        // float tree REDUNDANTLY on all 32 lanes (8 __shfl_sync gathers + 8
        // mults per lane, per block) for "bit-exactness".  Bit-exactness only
        // requires the SAME float ops in the SAME order, computed ONCE (the
        // redundant lanes' t[]/blk are dead — only lane 0's sumf is used).
        // With the Q8_0 integer GEMV routed through EVERY dense projection
        // (trunk, shared expert, LM head: 8192-row and 248320-row matrices
        // over 64 blocks/row), the redundant per-lane work made decode ~4-5x
        // slower than the memory floor: ~197 ms/token spent in moeBlockMs.
        // Now only lane 0 gathers the 8 converged group sums, runs the tree,
        // and accumulates; the ops (8 roundings, pair/hadd order, serial
        // mul+add without fmaf) are byte-identical to the CPU reference.
        __global__ void kQGemvQ8_0xQ8K(const uint8_t *__restrict__ w,
                                       const uint8_t *__restrict__ x,
                                       float *__restrict__ out, uint32_t rows,
                                       uint32_t blocksPerRow, uint32_t rowBytes) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t lane = threadIdx.x;
            float sumf = 0.0f;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * 34u;
                const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
                const float d = __half2float(*reinterpret_cast<const __half *>(blkW));
                const float d8 = *reinterpret_cast<const float *>(blkA);
                if (d8 == 0.0f) continue;
                const int wq = static_cast<int>(
                        static_cast<int8_t>(blkW[2u + lane]));
                const int xq = static_cast<int>(
                        static_cast<int8_t>(blkA[4u + lane]));
                // Folded per-column contribution (sign-substitution identity):
                // |xq| * w * sign(xq) == xq * w — the exact per-lane product
                // the AVX kernel's maddubs computes (|xq| in slot A unsigned,
                // w*sign(xq) in slot B signed; the magnitudes multiply, the
                // sign travels).  127*127 = 16129 fits int32 trivially.
                int c = xq * wq;
                // Per-4-column integer group sum s_g (the AVX maddubs groups of
                // 4 lanes: cols [4g..4g+3]); after the two XOR reduces, lane 4g
                // holds s_g (all lanes of the group hold the same value).
                c += __shfl_xor_sync(0xffffffffu, c, 1);
                c += __shfl_xor_sync(0xffffffffu, c, 2);
                // Gather the 8 group sums -- WARP-UNIFORM (__shfl_sync is a
                // warp-collective; it must be reached by ALL 32 lanes or it is
                // UB / hangs).  Every lane receives lane 4g's value, so the
                // gathered s[g] are identical across lanes.
                int s[8];
#pragma unroll
                for (uint32_t g = 0; g < 8; ++g) {
                    s[g] = __shfl_sync(0xffffffffu, c, 4u * g);
                }
                // Lane 0 only: run the exact float tree on the gathered sums
                // (same op order as the AVX reference, no fmaf).  The dead
                // lanes' float mults/adds were pure waste (measured: the
                // redundant per-lane tree dominated decode time on every
                // Q8_0 GEMV -- trunk, experts, LM head).
                if (lane == 0) {
                    const float scale = d * d8;
                    float t[8];
#pragma unroll
                    for (uint32_t g = 0; g < 8; ++g) {
                        t[g] = scale * static_cast<float>(s[g]);
                    }
                    const float u0 = t[0] + t[4];
                    const float u1 = t[1] + t[5];
                    const float u2 = t[2] + t[6];
                    const float u3 = t[3] + t[7];
                    const float v0 = u0 + u1;
                    const float v1 = u2 + u3;
                    const float blk = v0 + v1;
                    sumf += blk;
                }
            }
            if (lane == 0) out[row] = sumf;
        }

        // =====================================================================
        // BATCHED decode fast path (2026-09-14): launch ALL expertUsed ranks in
        // ONE kernel with grid.y = rank.  All ranks' GEMV blocks co-schedule on
        // the SMs (real parallelism instead of the old 40-serialized-kernels
        // chain), and rank r reads its OWN slap (per-rank out slices + the
        // per-rank weight pointer table moeRankWq_, [3*expertUsed] void*: gate /
        // up / down slab base per rank).  Each kernel is a bit-exact clone of the
        // single-matrix math it replaces — identical integer group sums, float
        // trees, op order, and no fmaf — so the batched path produces byte-equal
        // outputs to the old per-rank launches.
        // =====================================================================

        // Batched gate+up GEMV against the pre-quantized norm (q8kN_), then a
        // fused silu(gate)*up write into rank r's [expertFF] slab.  Per rank:
        // gate rows [0, expertFF) and up rows [expertFF, 2*expertFF) of the
        // cache slab (wq[r*2] = gate base, wq[r*2+1] = up base); grid.x covers
        // the 1024 combined rows (block (32,8)), grid.y = rank.
        __global__ void kQGemvQ8_0xQ8K_BatchGU(const uint8_t *const *__restrict__ wq,
                                               const uint8_t *__restrict__ x,
                                               float *__restrict__ gateOut,
                                               float *__restrict__ upOut,
                                               uint32_t expertFF,
                                               uint32_t blocksPerRow,
                                               uint32_t rowBytes,
                                               uint32_t combinedRows) {
            const uint32_t r = blockIdx.y;
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= combinedRows) return;
            // First half = gate GEMV, second half = up GEMV.
            const uint32_t isUp = (row >= expertFF) ? 1u : 0u;
            const uint32_t wrow = row - isUp * expertFF;
            // moeRankWq_ table: [r*3 + 0] = gate base, [r*3 + 1] = up base,
            // [r*3 + 2] = down base.
            const uint8_t *rp = wq[r * 3u + isUp] +
                                static_cast<uint64_t>(wrow) * rowBytes;
            const uint32_t lane = threadIdx.x;
            float sumf = 0.0f;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * 34u;
                const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
                const float d = __half2float(*reinterpret_cast<const __half *>(blkW));
                const float d8 = *reinterpret_cast<const float *>(blkA);
                if (d8 == 0.0f) continue;
                const int wq_v = static_cast<int>(
                        static_cast<int8_t>(blkW[2u + lane]));
                const int xq = static_cast<int>(
                        static_cast<int8_t>(blkA[4u + lane]));
                int c = xq * wq_v;
                c += __shfl_xor_sync(0xffffffffu, c, 1);
                c += __shfl_xor_sync(0xffffffffu, c, 2);
                int s[8];
#pragma unroll
                for (uint32_t g = 0; g < 8; ++g) {
                    s[g] = __shfl_sync(0xffffffffu, c, 4u * g);
                }
                if (lane == 0) {
                    const float scale = d * d8;
                    float t[8];
#pragma unroll
                    for (uint32_t g = 0; g < 8; ++g) {
                        t[g] = scale * static_cast<float>(s[g]);
                    }
                    const float u0 = t[0] + t[4];
                    const float u1 = t[1] + t[5];
                    const float u2 = t[2] + t[6];
                    const float u3 = t[3] + t[7];
                    const float v0 = u0 + u1;
                    const float v1 = u2 + u3;
                    const float blk = v0 + v1;
                    sumf += blk;
                }
            }
            if (lane == 0) {
                float *dest = isUp ? upOut : gateOut;
                dest[r * expertFF + wrow] = sumf;
            }
        }

        // Batch silu(gate)*up: rank r's fused activation, written in-place over
        // gateOut slab r.  (The up slab is consumed read-only.)  Plain expression
        // identical to kSiluMul: x/(1+__expf(-x)) * b (bit-exact with the CPU's
        // geluInPlace? NO: the GPU fast path uses silu — see kSiluMul).
        __global__ void kSiluMulBatch(float *__restrict__ gateOut,
                                      const float *__restrict__ upOut,
                                      uint32_t expertFF, uint32_t ranks) {
            const uint32_t r = blockIdx.y;
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= expertFF) return;
            float *a = gateOut + r * expertFF;
            float x = a[i];
            a[i] = (x / (1.0f + __expf(-x))) * upOut[r * expertFF + i];
        }

        // Batched per-rank quantize of the silu product (rank r reads gateOut
        // slab r, writes moeQ8kDownP_ slab r) using the EXACT kQuantizeQ8_0x32
        // math (__fdiv_rn IEEE divisions).  grid = (blocksPerRow, ranks).
        __global__ void kQuantizeQ8_0x32_Batch(const float *__restrict__ x,
                                               uint8_t *__restrict__ q32,
                                               uint32_t cols, uint32_t ranks,
                                               uint32_t strideBytes) {
            const uint32_t r = blockIdx.y;
            const uint32_t b = blockIdx.x;
            const uint32_t lane = threadIdx.x;
            const uint32_t cidx = b * 32u + lane;
            const float *xs = x + r * cols;
            uint8_t *q = q32 + (static_cast<uint64_t>(r) * strideBytes) +
                         static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
            if (cidx >= cols) {
                q[4u + lane] = 0;
                if (lane < 28u) q[36u + lane] = 0;
                if (lane == 0)
                    *reinterpret_cast<float *>(q) = 0.0f;
                return;
            }
            float amax = 0.0f, maxv = 0.0f;
            if (lane == 0) {
                const float *xblk = xs + b * 32u;
                for (uint32_t j = 0; j < 32u; ++j) {
                    const float av = fabsf(xblk[j]);
                    if (av > amax) {
                        amax = av;
                        maxv = xblk[j];
                    }
                }
            }
            amax = __shfl_sync(0xffffffffu, amax, 0);
            maxv = __shfl_sync(0xffffffffu, maxv, 0);
            if (amax == 0.0f) {
                q[4u + lane] = 0;
                if (lane < 28u) q[36u + lane] = 0;
                if (lane == 0) *reinterpret_cast<float *>(q) = 0.0f;
                return;
            }
            const float iscale = __fdiv_rn(-127.0f, maxv);
            int vq = static_cast<int>(rintf(iscale * xs[cidx]));
            if (vq > 127) vq = 127;
            if (vq < -127) vq = -127;
            q[4u + lane] = static_cast<uint8_t>(static_cast<int8_t>(vq));
            if (lane < 28u) q[36u + lane] = 0;
            if (lane == 0)
                *reinterpret_cast<float *>(q) = __fdiv_rn(1.0f, iscale);
        }

        // Batched down GEMV: rank r's [hidden] output from the quantized silu
        // product (moeQ8kDownP_ slab r).  The [H][expertFF] down matrix rows =
        // stack of [expertFF]-wide blocks; grid.x rows (block (32,8)), grid.y
        // rank; per-rank down base from wq[r*2+2] + r's explicit slab offset.
        __global__ void kQGemvQ8_0xQ8K_BatchDown(const uint8_t *const *__restrict__ wq,
                                                 const uint8_t *__restrict__ x,
                                                 float *__restrict__ out,
                                                 uint32_t hidden,
                                                 uint32_t blocksPerRow,
                                                 uint32_t rowBytes,
                                                 uint32_t xStrideBytes) {
            const uint32_t r = blockIdx.y;
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= hidden) return;
            // moeRankWq_ table: [r*3 + 2] = down base.
            const uint8_t *rp = wq[r * 3u + 2u] +
                                static_cast<uint64_t>(row) * rowBytes;
            const uint8_t *xr = x + (static_cast<uint64_t>(r) * xStrideBytes);
            const uint32_t lane = threadIdx.x;
            float sumf = 0.0f;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * 34u;
                const uint8_t *blkA = xr + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
                const float d = __half2float(*reinterpret_cast<const __half *>(blkW));
                const float d8 = *reinterpret_cast<const float *>(blkA);
                if (d8 == 0.0f) continue;
                const int wq_v = static_cast<int>(
                        static_cast<int8_t>(blkW[2u + lane]));
                const int xq = static_cast<int>(
                        static_cast<int8_t>(blkA[4u + lane]));
                int c = xq * wq_v;
                c += __shfl_xor_sync(0xffffffffu, c, 1);
                c += __shfl_xor_sync(0xffffffffu, c, 2);
                int s[8];
#pragma unroll
                for (uint32_t g = 0; g < 8; ++g) {
                    s[g] = __shfl_sync(0xffffffffu, c, 4u * g);
                }
                if (lane == 0) {
                    const float scale = d * d8;
                    float t[8];
#pragma unroll
                    for (uint32_t g = 0; g < 8; ++g) {
                        t[g] = scale * static_cast<float>(s[g]);
                    }
                    const float u0 = t[0] + t[4];
                    const float u1 = t[1] + t[5];
                    const float u2 = t[2] + t[6];
                    const float u3 = t[3] + t[7];
                    const float v0 = u0 + u1;
                    const float v1 = u2 + u3;
                    const float blk = v0 + v1;
                    sumf += blk;
                }
            }
            if (lane == 0) out[r * hidden + row] = sumf;
        }

        // Batched down GEMV with rank REMAP (Option D resident split):
        // identical math to kQGemvQ8_0xQ8K_BatchDown, but the OUTPUT rank
        // index comes from the per-slot rankOf table — the resident ranks are
        // COMPACTED into slots 0..residentCount-1 for the wq/silu/quantize
        // pipeline, while the down rows must land in the FULL rank-major
        // combined table at [rankOf[j]*outStride + row] so the rank-order
        // accumulation (kQ35MoeAccumScaled(kQ35MoeAccumScaledBatch}) reads the
        // CPU-filled non-resident slots and the GPU-filled resident slots in
        // the exact reference order.  outStride == seqLen*H (per-rank slab).
        __global__ void kQGemvQ8_0xQ8K_BatchDownR(const uint8_t *const *__restrict__ wq,
                                                  const uint8_t *__restrict__ x,
                                                  float *__restrict__ out,
                                                  const int32_t *__restrict__ rankOf,
                                                  uint32_t hidden,
                                                  uint32_t blocksPerRow,
                                                  uint32_t rowBytes,
                                                  uint32_t xStrideBytes,
                                                  uint32_t outStride) {
            const uint32_t j = blockIdx.y;// compact resident slot
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= hidden) return;
            const int32_t r0 = (rankOf != nullptr) ? rankOf[j] : static_cast<int32_t>(j);
            if (r0 < 0) return;
            // wq table: [j*3 + 2] = down base of the compact resident slot j.
            const uint8_t *rp = wq[j * 3u + 2u] +
                                static_cast<uint64_t>(row) * rowBytes;
            const uint8_t *xr = x + (static_cast<uint64_t>(j) * xStrideBytes);
            const uint32_t lane = threadIdx.x;
            float sumf = 0.0f;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * 34u;
                const uint8_t *blkA = xr + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
                const float d = __half2float(*reinterpret_cast<const __half *>(blkW));
                const float d8 = *reinterpret_cast<const float *>(blkA);
                if (d8 == 0.0f) continue;
                const int wq_v = static_cast<int>(
                        static_cast<int8_t>(blkW[2u + lane]));
                const int xq = static_cast<int>(
                        static_cast<int8_t>(blkA[4u + lane]));
                int c = xq * wq_v;
                c += __shfl_xor_sync(0xffffffffu, c, 1);
                c += __shfl_xor_sync(0xffffffffu, c, 2);
                int s[8];
#pragma unroll
                for (uint32_t g = 0; g < 8; ++g) {
                    s[g] = __shfl_sync(0xffffffffu, c, 4u * g);
                }
                if (lane == 0) {
                    const float scale = d * d8;
                    float t[8];
#pragma unroll
                    for (uint32_t g = 0; g < 8; ++g) {
                        t[g] = scale * static_cast<float>(s[g]);
                    }
                    const float u0 = t[0] + t[4];
                    const float u1 = t[1] + t[5];
                    const float u2 = t[2] + t[6];
                    const float u3 = t[3] + t[7];
                    const float v0 = u0 + u1;
                    const float v1 = u2 + u3;
                    const float blk = v0 + v1;
                    sumf += blk;
                }
            }
            if (lane == 0) {
                out[static_cast<uint64_t>(r0) * outStride + row] = sumf;
            }
        }

        // =====================================================================
        // Q4_K / Q6_K x Q8_K-activation integer GEMV (Option D resident fast
        // path for Q4_K_M models, 2026-09-16).
        //
        // The Ornith-1.5-35B-Q4_K_M expert matrices are gate/up = Q4_K
        // (type 12, 144 B/256-element block) + down = Q6_K (type 14, 210
        // B/block).  The CPU reference (matMulVecBatchQ4K_Q8K_AVX2 /
        // matMulVecBatchQ6K_Q8K_AVX2) quantizes the fp32 activation to Q8_K
        // (256-wide blocks, kQ8K_STRIDE layout) and computes the dot with
        // EXACT integer products plus specific fp32 trees.  These kernels
        // reproduce BOTH the integer math and the float op order so the GPU
        // resident path is bit-exact with the CPU grouped-batch path.
        //
        // DECODED COMPILER BEHAVIOR (verified by disassembly of
        // SIMDMatMulVecAVX2.o under -O2 -mavx2 -mfma, the real flags):
        //   Q4_K:  mainS = fma(d*sc_i, s32q_i, mainS)   (i = 0..7 serial)
        //          minS  = fma(dmin*m_i, bsum_i, minS)  (i = 0..7 serial)
        //          acc   = fma(d8, mainS - minS, acc)   (vsubss + vfmadd132ss)
        //   Q6_K:  per-lane offset = (-32*d*d8) * bprod       (2 muls + 1 mul)
        //          acc_j = fma(d*d8, sumi0_j, off_j)          (chunk 0)
        //          acc_j = fma(d*d8, sumi1_j, acc_j)          (chunk 1)
        //          running_j += acc_j (per block, serial 8-lane add)
        //          final hsum tree: p=a0+a4..a3+a7 (add), q=p0+p1,p2+p3,
        //          r=q0+q1  (the AVX hadd chain: 7 roundings, matching the
        //          _mm_hadd_ps tree exactly).
        // All int->float conversions are exact (values < 2^24), so only the
        // float op ORDER matters.
        // =====================================================================

        // ---- Warp-per-row Q4_K / Q6_K x Q8_K GEMV: bit-exact with the CPU
        // matMulVecBatchQ4K_Q8K_AVX2 / matMulVecBatchQ6K_Q8K_AVX2 kernels.
        // blockDim = (32, 8): threadIdx.y = row, 32 lanes = sub-blocks /
        // column lanes.  All per-lane integer products are EXACT (int32
        // accumulators), so only the float op ORDER matters:
        //
        // Q4_K (NOT FMA-contracted: the SIMD TU is compiled with -mfma but
        // WITHOUT -ffp-contract, so `mainS += ds*s32q` etc. emit separate
        // vmulss+vaddss pairs, verified by objdump at the tile lambda and by
        // the mode0 bit-exact match of the no-FMA host model; hence the
        // volatile-intermediate trick below):
        //   mainS = dsc_i*s32q_i + mainS    i = 0..7 serial (mul then add)
        //   minS  = dmn_i*bsum_i + minS     i = 0..7 serial (mul then add)
        //   sumf  = d8*(mainS - minS) + sumf  (vsubss then mul then add)
        // Q6_K (explicit intrinsics, no contraction beyond the literal
        // _mm256_fmadd_ps):
        //   off_j = ((-32*d) * d8) * bprod_j        (3 muls)
        //   acc_j = fma(d*d8, sumi0_j, off_j)       (chunk 0)
        //   acc_j = fma(d*d8, sumi1_j, acc_j)       (chunk 1)
        //   running_j += acc_j per block (vaddps)
        //   final: p_j = a_j + a_{j+4}; q = p + shfl_xor(p, 1);
        //          r = q + shfl_xor(q, 2)  (== the AVX hadd chain: 7 adds)
        //
        // Q6_K scale identity (from k_shuffle + cvtepi8_epi16 in the AVX2
        // kernel): for global sub-block s' in [0,8), the madd lanes of that
        // sub use scale sc[2s'] for its columns 0..15 and sc[2s'+1] for its
        // columns 16..31 (sc = sign-extended int8[16] block scales at
        // blk+192).  sumi lane J (J in 0..7) of chunk c accumulates sub
        // s = 4c..4c+3, global columns s*32 + 4J .. s*32 + 4J+3, weighted by
        // sc[2s*] (J<4) or sc[2s*+1] (J>=4).
        // Shared per-block dot contribution for the Q4_K / Q6_K x Q8_K GEMV
        // kernels (single-row + batched).  Warp-collective: ALL 32 lanes call
        // it (the __shfl_sync gathers are reached unconditionally).  For Q4_K
        // only lane 0 updates `running` (the per-block scalar addend, applied
        // as fmaf(d8, ms - mns, running) — the exact vsubss+vfmadd132ss CPU
        // order); for Q6_K lanes 0..7 each update their own lane-j running
        // accumulator by `acc_j`.  NOTE: NO d8==0 skip — the CPU Q4K/Q6K
        // kernels have no zero-guard, and skipping would change the sign of a
        // -0 running accumulator (bit-exactness requires the +0 add).
        // WIDE 4-byte load for 210-B-stride Q6_K weight ints (plan section 9.23,
        // extended to the LM head 2026-10-11): the block base is always even
        // (210 = 2*105) and every offset read here is a multiple of 2, so a pair
        // of 16-bit loads is legal at EVERY address and assembles the same
        // little-endian 32-bit word that std::memcpy from a uint8_t* otherwise
        // splits into 4 byte LDGs (the LG-throttle disease section 9.23
        // root-caused and fixed on the mmvq family).  Two LDG.E.U16 instead of
        // four LDG.E.U8 per weight int, bit-identical bytes.
        __device__ __forceinline__ uint32_t kxWideU32(const uint8_t *p) {
            const unsigned short *p16 =
                    reinterpret_cast<const unsigned short *>(p);
            return static_cast<uint32_t>(p16[0]) |
                   (static_cast<uint32_t>(p16[1]) << 16);
        }

        template<int TYPE, bool WIDE = false, bool GROUPED = false>
        __device__ __forceinline__ void kxQ8KBlockDot(float &running,
                                                      const uint8_t *blkW,
                                                      const uint8_t *blkA,
                                                      uint32_t lane) {
            // GROUPED=false (legacy 1-row-per-warp call): only warp lanes 0-7
            // (the sub-block owners) work; the reduction gathers lanes 0-7
            // (shuffle width 32) and lane 0 folds into `running`.  EVERY
            // expression below is unchanged vs the pre-2026-10-12 kernel.
            // GROUPED=true (4-rows-per-warp call): every lane owns sub-block
            // Iid = lane&7 of ITS own 8-lane group's row, so the gather uses
            // shuffle width 8 (each group reduces its own row) and each group
            // leader (Iid==0) folds its row's value into its own `running`.
            // The per-(row, Iid) arithmetic is byte-identical, only the lane
            // that ends up holding a given row's scalar changes -- which the
            // caller's 4-row shuffle tree coalesces within each group.
            const uint32_t Iid = GROUPED ? (lane & 7u) : lane;
            const uint32_t shW = GROUPED ? 8u : 32u;
            const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA);
            const int16_t *bsums =
                    reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
            const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
            if (TYPE == kTypeQ4K) {
                int s32q = 0;
                float dsv = 0.0f, mnv = 0.0f;
                if (Iid < 8) {
                    const uint32_t i = Iid;
                    const float d = __half2float(
                            *reinterpret_cast<const __half *>(blkW));
                    const float dmin = __half2float(
                            *reinterpret_cast<const __half *>(blkW + 2));
                    const uint8_t *scales = blkW + 4;
                    const uint8_t *qsv = blkW + 16;
                    uint8_t sc, m;
                    if (i < 4) {
                        sc = scales[i] & 63u;
                        m = scales[i + 4] & 63u;
                    } else {
                        sc = static_cast<uint8_t>(
                                (scales[i + 4] & 0xFu) |
                                ((scales[i - 4] >> 6) << 4));
                        m = static_cast<uint8_t>(
                                (scales[i + 4] >> 4) |
                                ((scales[i] >> 6) << 4));
                    }
                    dsv = d * static_cast<float>(sc);
                    mnv = dmin * static_cast<float>(m);
                    const uint32_t half = i & 1u;
                    const uint8_t *qsb = qsv + 32u * (i >> 1u);
                    const int8_t *yp = y8 + i * 32u;
                    // VECTORIZED integer body (2026-10-29): 8x __dp4a replaces
                    // the 32 scalar byte-loads + per-byte nibble select that
                    // made this the scalar bottleneck of BOTH the generic
                    // 1-row GEMV and the batched prefill GEMM
                    // (kQGemmKxQ8K).  qsb is 4-aligned (rowBytes 144 % 4 == 0,
                    // qsv=blkW+16, 32*(i>>1), 4-B steps), so the u32 loads are
                    // aligned.  w32 holds the 4 packed 0..15 weights, yb4 the 4
                    // signed int8 activations; the integer sum is exact and
                    // order-free, so s32q is bit-identical to the scalar body
                    // and the float fold below is unchanged.  This is the same
                    // dp4a body as kxQ4KBlockDotVec (4-rows-per-warp kernel);
                    // it lifts the prefill GEMM (the dense-qwen35 prefill cost)
                    // without touching any output bit.
#pragma unroll
                    for (uint32_t j = 0; j < 32u; j += 4u) {
                        const uint32_t wb4 =
                                *reinterpret_cast<const uint32_t *>(qsb + j);
                        const uint32_t yb4 =
                                *reinterpret_cast<const uint32_t *>(yp + j);
                        const uint32_t w32 =
                                (half == 0u) ? (wb4 & 0x0F0F0F0Fu)
                                             : ((wb4 >> 4) & 0x0F0F0F0Fu);
                        s32q = __dp4a(static_cast<int>(w32),
                                      static_cast<int>(yb4), s32q);
                    }
                }
                int ss8[8];
                float dss[8], mns8[8];
#pragma unroll
                for (uint32_t i = 0; i < 8; ++i) {
                    ss8[i] = __shfl_sync(0xffffffffu, s32q, i, shW);
                    dss[i] = __shfl_sync(0xffffffffu, dsv, i, shW);
                    mns8[i] = __shfl_sync(0xffffffffu, mnv, i, shW);
                }
                if (Iid == 0) {
                    // The compiled CPU kernel (matMulVecBatchQ4K_Q8K_AVX2, built
                    // with -mfma but WITHOUT -ffp-contract) executes
                    // `mainS += ds*s32q` as SEPARATE vmulss+vaddss (NO FMA
                    // contraction — verified by disassembly and the mode0
                    // bit-exact match of the no-FMA host model).  nvcc WOULD
                    // fuse a plain `a*b + c` into FFMA, so the multiply must be
                    // forced through a volatile local (memory materialization
                    // makes fusion impossible) to reproduce the CPU's two
                    // independent roundings bit-exactly.
                    float ms = 0.0f, mns = 0.0f;
#pragma unroll
                    for (uint32_t i = 0; i < 8; ++i) {
                        const float bi = static_cast<float>(
                                static_cast<int>(bsums[2 * i]) +
                                static_cast<int>(bsums[2 * i + 1]));
                        volatile float pMain = dss[i] *
                                               static_cast<float>(ss8[i]);
                        ms = pMain + ms;
                        volatile float pMin = mns8[i] * bi;
                        mns = pMin + mns;
                    }
                    volatile float pFin = d8 * (ms - mns);
                    running = pFin + running;
                }
            } else if (TYPE == kTypeQ5K) {
                // Q5_K (176 B/block): d(2) + dmin(2) + scales[12] + qh[32] +
                // qs[128].  Mirror matMulVecBatchQ5K_Q8K_AVX2 EXACTLY:
                //   * per sub-block i (0..7): 5-bit weights from the LOW/HIGH
                //     nibbles of qs[32c] (c = i>>1) plus the qh bit
                //     1<<(2c+half) selecting the 16-bit value, integer-dotted
                //     with the Q8_K activation y8[i*32..i*32+31];
                //   * mainS/MS += ds[i]*s32q (mul-then-add, NO FMA — forced
                //     through volatile locals since nvcc would otherwise
                //     contract the multiply+add into a single FFMA);
                //   * minS/MNS += mn[i]*(bsums[2i]+bsums[2i+1]);
                //   * lane 0: running += d8*(mainS-minS) (vsubss then vmulss
                //     then vaddss — same 3-rounding chain as the AVX2 kernel).
                int s32q = 0;
                float dsv = 0.0f, mnv = 0.0f;
                if (Iid < 8) {
                    const uint32_t i = Iid;
                    const float d = __half2float(
                            *reinterpret_cast<const __half *>(blkW));
                    const float dmin = __half2float(
                            *reinterpret_cast<const __half *>(blkW + 2));
                    const uint8_t *scales = blkW + 4;
                    const uint8_t *qh = blkW + 16;
                    const uint8_t *qs = blkW + 48;
                    uint8_t sc, m;
                    if (i < 4) {
                        sc = scales[i] & 63u;
                        m = scales[i + 4] & 63u;
                    } else {
                        sc = static_cast<uint8_t>(
                                (scales[i + 4] & 0xFu) |
                                ((scales[i - 4] >> 6) << 4));
                        m = static_cast<uint8_t>(
                                (scales[i + 4] >> 4) |
                                ((scales[i] >> 6) << 4));
                    }
                    dsv = d * static_cast<float>(sc);
                    mnv = dmin * static_cast<float>(m);
                    const uint32_t c = i >> 1u;
                    const uint32_t half = i & 1u;
                    const uint32_t u = 1u << (2u * c + half);
                    const uint8_t *qsp = qs + 32u * c;
                    for (uint32_t j = 0; j < 32u; ++j) {
                        const uint8_t qb = qsp[j];
                        const uint8_t low = (half == 0u) ? (qb & 0xFu)
                                                         : (qb >> 4);
                        const uint8_t q5 = static_cast<uint8_t>(
                                low | ((qh[j] & u) != 0u ? 0x10u : 0u));
                        s32q += static_cast<int>(q5) *
                                static_cast<int>(y8[i * 32u + j]);
                    }
                }
                int ss8[8];
                float dss[8], mns8[8];
#pragma unroll
                for (uint32_t i = 0; i < 8; ++i) {
                    ss8[i] = __shfl_sync(0xffffffffu, s32q, i, shW);
                    dss[i] = __shfl_sync(0xffffffffu, dsv, i, shW);
                    mns8[i] = __shfl_sync(0xffffffffu, mnv, i, shW);
                }
                if (Iid == 0) {
                    float ms = 0.0f, mns = 0.0f;
#pragma unroll
                    for (uint32_t i = 0; i < 8; ++i) {
                        const float bi = static_cast<float>(
                                static_cast<int>(bsums[2 * i]) +
                                static_cast<int>(bsums[2 * i + 1]));
                        volatile float pMain = dss[i] *
                                               static_cast<float>(ss8[i]);
                        ms = pMain + ms;
                        volatile float pMin = mns8[i] * bi;
                        mns = pMin + mns;
                    }
                    volatile float pFin = d8 * (ms - mns);
                    running = pFin + running;
                }
            } else if (TYPE == kTypeIQ4XS) {
                // IQ4_XS (136 B/block): d(2) + scales_h(2) + scales_l(4) +
                // qs[128].  Mirror matMulVecBatchIQ4XS_Q8K_AVX2 EXACTLY:
                //   * per sub-block i (0..7): 6-bit scale ls (nibble of
                //     scales_l[i/2] + 2 bits of scales_h), dl = d*(ls-32);
                //   * the 32 weights are kvalues_iq4nl[nib] (SIGNED int8), so
                //     the AVX2 |w|*sign(w)*x integer dot reduces to a plain
                //     signed kvalues_iq4nl[nib]*y8[col] dot (exact int32);
                //   * mainS/MS += dl[i]*s32q (mul-then-add, NO FMA — volatile
                //     locals, same no-ffp-contract discipline as Q4_K);
                //   * NO min term (the (ls-32) centers the scale);
                //   * lane 0: running += d8*mainS (vmulss then vaddss).
                int s32q = 0;
                float dlsv = 0.0f;
                if (Iid < 8) {
                    const uint32_t i = Iid;
                    const float d = __half2float(
                            *reinterpret_cast<const __half *>(blkW));
                    const uint16_t scales_h =
                            *reinterpret_cast<const uint16_t *>(blkW + 2);
                    const uint8_t *scales_l = blkW + 4;
                    const uint8_t *qsv = blkW + 8;
                    const int ls = static_cast<int>(
                                           (scales_l[i >> 1u] >>
                                            (4u * (i & 1u))) &
                                           0xFu) |
                                   static_cast<int>(
                                           ((scales_h >> (2u * i)) & 0x3u)
                                           << 4);
                    dlsv = d * static_cast<float>(ls - 32);
                    const uint8_t *qsp = qsv + 16u * i;
                    for (uint32_t j = 0; j < 32u; ++j) {
                        const uint8_t nib = (j < 16u)
                                                    ? static_cast<uint8_t>(
                                                              qsp[j] & 0xFu)
                                                    : static_cast<uint8_t>(
                                                              qsp[j - 16u] >>
                                                              4);
                        s32q += static_cast<int>(c_kvalues_iq4nl[nib]) *
                                static_cast<int>(y8[i * 32u + j]);
                    }
                }
                int ss8[8];
                float dls8[8];
#pragma unroll
                for (uint32_t i = 0; i < 8; ++i) {
                    ss8[i] = __shfl_sync(0xffffffffu, s32q, i, shW);
                    dls8[i] = __shfl_sync(0xffffffffu, dlsv, i, shW);
                }
                if (Iid == 0) {
                    float ms = 0.0f;
#pragma unroll
                    for (uint32_t i = 0; i < 8; ++i) {
                        volatile float pMain = dls8[i] *
                                               static_cast<float>(ss8[i]);
                        ms = pMain + ms;
                    }
                    volatile float pFin = d8 * ms;
                    running = pFin + running;
                }
            } else {
                // Q6_K (210 B/block): ql[128] + qh[64] + sc[16] + d(2).
                // Bit-exact mirror of matMulVecBatchQ6K_Q8K_AVX2 blockDot +
                // 8-lane store: per active lane I (0..7):
                //   * bprod = sc[2I]*bsums[2I] + sc[2I+1]*bsums[2I+1] (the
                //     CPU's madd_epi16(s16, bsums) lane I);
                //   * acc  = kOff*bprod, kOff = (-32*d)*d8 (CPU: set1_ps
                //     mul_ps of the bprod lane product);
                //   * per chunk c (0/1), 4 subs s (idx = 4c+s): sumi over the
                //     4 columns col = idx*32 + 4*I + k, each wv = (ql nibble
                //     [low if (idx&2)==0 else high]) | (qh 2-bit field
                //     shift 2*(idx&3)) << 4, scaled by sc[2*idx] (I<4) or
                //     sc[2*idx+1] (I>=4) matching the CPU shuffle broadcast;
                //     then acc = fmaf(d*d8, sumi, acc) (CPU fmadd_ps).
                // 2026-09-28 VECTORIZED: lanes 0-7 (the only workers) used to
                // load every weight byte individually (64 ld.b8/block with 24
                // of 32 lanes idle) -- the Q6_K LM head (151936 rows x 6
                // blocks, 8960 rows x 35 blocks) ran 5x above the 448 GB/s
                // floor.  Now each lane pulls its 4 contiguous ql bytes
                // (W[a][c] = blkW + 32*a + 64*c + 4*I) and 4 qh bytes
                // (H[c] = blkW + 128 + 32*c + 4*I) as u32 registers and its
                // 4 activation bytes per sub-block (Y[idx] = blkA + 32*idx +
                // 4*I) as u32.  Same bytes, same products, same (c,s,k)
                // accumulation order -- bit-identical results.
                float acc = 0.0f;
                if (lane < 8) {
                    const uint32_t I = lane;
                    const float d = __half2float(
                            *reinterpret_cast<const __half *>(blkW + 208));
                    const int8_t *sc =
                            reinterpret_cast<const int8_t *>(blkW + 192);
                    const int s0 = static_cast<int>(sc[2 * lane]);
                    const int s1 = static_cast<int>(sc[2 * lane + 1]);
                    const int bprod =
                            s0 * static_cast<int>(bsums[2 * lane]) +
                            s1 * static_cast<int>(bsums[2 * lane + 1]);
                    const float kOff = (-32.0f * d) * d8;
                    acc = kOff * static_cast<float>(bprod);
                    const float k2 = d * d8;
                    // u32 register loads (unaligned-safe memcpy; fully unrolled
                    // so the W/H/Y arrays stay in registers).
                    uint32_t W[2][2], H[2], Y[8];
                    // NOTE: 210-byte Q6_K blocks are NOT 4-byte aligned at row
                    // granularity (210 = 2x105), so these MUST stay unaligned-
                    // safe loads -- a u32 __ldcs here faults (tried 2026-10-10,
                    // killed the kernel -> CPU fallback).  The WIDE form
                    // (section 9.23) issues 2x 16-bit loads (always legal: the
                    // block base and every weight offset are even) instead of
                    // the 4 byte LDGs std::memcpy emits through uint8_t*; the
                    // Y (activation) loads are 4-aligned (kQ8K_STRIDE 292,
                    // 4-B k-cols) so they take a single int deref.
                    if constexpr (WIDE) {
#pragma unroll
                        for (uint32_t a = 0; a < 2; ++a) {
                            H[a] = kxWideU32(blkW + 128u + 32u * a + 4u * I);
#pragma unroll
                            for (uint32_t c = 0; c < 2; ++c) {
                                W[a][c] = kxWideU32(
                                        blkW + 32u * a + 64u * c + 4u * I);
                            }
                        }
#pragma unroll
                        for (uint32_t q = 0; q < 8; ++q) {
                            Y[q] = *reinterpret_cast<const uint32_t *>(
                                    blkA + 32u * q + 4u * I);
                        }
                    } else {
#pragma unroll
                        for (uint32_t a = 0; a < 2; ++a) {
                            std::memcpy(&H[a], blkW + 128u + 32u * a + 4u * I,
                                        sizeof(uint32_t));
#pragma unroll
                            for (uint32_t c = 0; c < 2; ++c) {
                                std::memcpy(&W[a][c],
                                            blkW + 32u * a + 64u * c + 4u * I,
                                            sizeof(uint32_t));
                            }
                        }
#pragma unroll
                        for (uint32_t q = 0; q < 8; ++q) {
                            std::memcpy(&Y[q], blkA + 32u * q + 4u * I,
                                        sizeof(uint32_t));
                        }
                    }
#pragma unroll
                    for (uint32_t c = 0; c < 2; ++c) {
                        int sumi = 0;
                        const uint32_t hw = H[c];
#pragma unroll
                        for (uint32_t s = 0; s < 4; ++s) {
                            const uint32_t idx = 4u * c + s;
                            const uint32_t qw = W[s & 1u][c];
                            const uint32_t yw = Y[idx];
                            // Scale for lane I: I<4 -> sc[2*idx],
                            // I>=4 -> sc[2*idx+1] (CPU shuffle broadcast).
                            const int sca = (lane < 4)
                                                    ? static_cast<int>(
                                                              sc[2 * idx])
                                                    : static_cast<int>(
                                                              sc[2 * idx + 1]);
                            const uint32_t nibShift =
                                    ((s & 2u) != 0u) ? 4u : 0u;
                            const uint32_t hiShift = 2u * s;
                            int acc4 = 0;
#pragma unroll
                            for (uint32_t k = 0; k < 4; ++k) {
                                const uint32_t b8 =
                                        (qw >> (8u * k + nibShift)) & 0xFu;
                                const uint32_t h2 =
                                        (hw >> (8u * k + hiShift)) & 3u;
                                const int wv =
                                        static_cast<int>(b8 | (h2 << 4));
                                const int yv = static_cast<int>(
                                        static_cast<int8_t>(static_cast<uint8_t>(
                                                yw >> (8u * k))));
                                acc4 += wv * yv;
                            }
                            sumi += sca * acc4;
                        }
                        acc = fmaf(k2, static_cast<float>(sumi), acc);
                    }
                }
                if (lane < 8) running += acc;
            }
        }

        // ---- Single-matrix row GEMV (Q4_K or Q6_K weight x Q8_K activation).
        // blockDim = (32, 8) (8 warps = 8 rows per block).
        template<int TYPE>
        __global__ void kQGemvKxQ8K(const uint8_t *__restrict__ w,
                                    const uint8_t *__restrict__ x,
                                    float *__restrict__ out, uint32_t rows,
                                    uint32_t blocksPerRow, uint32_t rowBytes) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t lane = threadIdx.x;
            float sumf = 0.0f;
            constexpr uint32_t kBlkBytes =
                    (TYPE == kTypeQ4K)     ? kQ4K_BYTES
                    : (TYPE == kTypeQ5K)   ? kQ5K_BYTES
                    : (TYPE == kTypeIQ4XS) ? kIQ4XS_BYTES
                                           : kQ6K_BYTES;
            // BLOCK BATCHING (2026-09-27): the per-block kxQ8KBlockDot ends in
            // a lane-0 serial `running = pFin + running`, giving a fully serial
            // dependency chain across the blocksPerRow blocks (the LM head at
            // 6 blocks/row measured 3.49 ms vs ~360 us at the 448 GB/s floor).
            // Compute 4 blocks into INDEPENDENT fresh accumulators (each
            // `pFin + 0.0f`, byte-identical to the original first-block add),
            // then serialize the four sumf adds IN BLOCK ORDER.  Float addition
            // is commutative, and each block's pFin chain (s32q -> shuffles ->
            // volatile dss*ss8/mns8*bi -> d8*(ms-mns)) is UNCHANGED, so the
            // bit-exact running sum is preserved exactly.
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
                const uint8_t *bw0 = rp + static_cast<uint64_t>(b + 0u) * kBlkBytes;
                const uint8_t *bw1 = rp + static_cast<uint64_t>(b + 1u) * kBlkBytes;
                const uint8_t *bw2 = rp + static_cast<uint64_t>(b + 2u) * kBlkBytes;
                const uint8_t *bw3 = rp + static_cast<uint64_t>(b + 3u) * kBlkBytes;
                const uint8_t *ba0 = x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE;
                const uint8_t *ba1 = x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE;
                const uint8_t *ba2 = x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE;
                const uint8_t *ba3 = x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE;
                kxQ8KBlockDot<TYPE>(r0, bw0, ba0, lane);
                kxQ8KBlockDot<TYPE>(r1, bw1, ba1, lane);
                kxQ8KBlockDot<TYPE>(r2, bw2, ba2, lane);
                kxQ8KBlockDot<TYPE>(r3, bw3, ba3, lane);
                if (TYPE == kTypeQ6K) {
                    // Q6K accumulates its row partial on ALL 8 active lanes
                    // (each lane owns 32 distinct columns per block via
                    // col = idx*32 + 4*lane + k), and the final shuffle-tree
                    // store sums every lane.  The lane-0-only fold silently
                    // dropped lanes 1-7's blocks 0-3 (7/8 of the columns of
                    // 4 of 6 blocks = ~58% of the LM head dot product), which
                    // the tolerant argmax tests masked -- the head's logits
                    // measured ~3.3 low for a 12.6-wide range.  The fold's
                    // add order (r0, then r1, r2, r3 left-assoc into the
                    // running sum) is byte-identical to the serial tail loop
                    // (running = pFin + running), so per-lane results are
                    // bit-exact with the unbatched path.
                    sumf = r0 + sumf;
                    sumf = r1 + sumf;
                    sumf = r2 + sumf;
                    sumf = r3 + sumf;
                } else if (lane == 0) {
                    // Q4_K / Q5_K / IQ4_XS accumulate the whole block on
                    // lane 0 only (their scale/bsum reduction converges to
                    // lane 0), so the fold stays lane-0-only there.
                    sumf = r0 + sumf;
                    sumf = r1 + sumf;
                    sumf = r2 + sumf;
                    sumf = r3 + sumf;
                }
            }
            for (; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * kBlkBytes;
                const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
                kxQ8KBlockDot<TYPE>(sumf, blkW, blkA, lane);
            }
            if (TYPE == kTypeQ4K || TYPE == kTypeQ5K ||
                TYPE == kTypeIQ4XS) {
                if (lane == 0) out[row] = sumf;
            } else {
                const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
                const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
                const float r = q + __shfl_xor_sync(0xffffffffu, q, 2);
                if (lane == 0) out[row] = r;
            }
        }

        // ---- Q6_K row GEMV (Q8_K activation), 4 ROWS PER WARP (2026-09-28).
        // The generic kQGemvKxQ8K<kTypeQ6K> runs one row per warp and only
        // sub-lanes 0-7 issue weight loads (24 of 32 lanes idle per
        // instruction) -- the Q6_K LM head (151936 rows x 6 blocks x 210 B =
        // 191 MB/token) measured 1850 us vs the ~370 us 448 GB/s floor.  Here
        // lane group L>>3 selects the row (4 rows/warp) and sub-lane I = L&7
        // keeps the EXACT per-(row, I) float chain (bprod, kOff, two chunk
        // FMAs) of kxQ8KBlockDot<kTypeQ6K>, so each row's result is
        // bit-identical (int sums order-free; float chain unchanged); the
        // 8-lane shuffle-tree store (xor strides 4,1,2) stays within each
        // 8-lane group.  Every lane now issues weight loads, quadrupling
        // memory-level parallelism (each load covers 4 rows x 32 B, one
        // sector per row).  Requires rows % 4 == 0 (all real uses: LM head
        // 151936, ffn_down 1536, gate/up 8960); dispatch falls back to
        // kQGemvKxQ8K<kTypeQ6K> otherwise.
        // kQGemvQ6KxQ8K_4xW (~616 GB/s LM-head): all 32 lanes issue u32
        // weight loads (see the block comment above), with WIDE selecting the
        // section 9.23 2x-U16 weight loads / int-deref Y loads for the LM head.
        template<bool WIDE = false>
        __global__ void kQGemvQ6KxQ8K_4xW(const uint8_t *__restrict__ w,
                                          const uint8_t *__restrict__ x,
                                          float *__restrict__ out,
                                          uint32_t rows,
                                          uint32_t blocksPerRow,
                                          uint32_t rowBytes) {
            const uint32_t warpRowBase =
                    blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * 4u;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t row = base + (lane >> 3u);
            const uint8_t *rp =
                    w + static_cast<uint64_t>(row) * rowBytes;
            float sumf = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
                kxQ8KBlockDot<kTypeQ6K, WIDE>(
                        r0, rp + static_cast<uint64_t>(b + 0u) * kQ6K_BYTES,
                        x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE, I);
                kxQ8KBlockDot<kTypeQ6K, WIDE>(
                        r1, rp + static_cast<uint64_t>(b + 1u) * kQ6K_BYTES,
                        x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE, I);
                kxQ8KBlockDot<kTypeQ6K, WIDE>(
                        r2, rp + static_cast<uint64_t>(b + 2u) * kQ6K_BYTES,
                        x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE, I);
                kxQ8KBlockDot<kTypeQ6K, WIDE>(
                        r3, rp + static_cast<uint64_t>(b + 3u) * kQ6K_BYTES,
                        x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE, I);
                sumf = r0 + sumf;
                sumf = r1 + sumf;
                sumf = r2 + sumf;
                sumf = r3 + sumf;
            }
            for (; b < blocksPerRow; ++b) {
                kxQ8KBlockDot<kTypeQ6K, WIDE>(
                        sumf, rp + static_cast<uint64_t>(b) * kQ6K_BYTES,
                        x + static_cast<uint64_t>(b) * kQ8K_STRIDE, I);
            }
            // 8-lane shuffle-tree per row within each 4-row lane group
            // (xor strides 4,1,2 stay inside the group's 8 lanes, same tree
            // as the 1-row-per-warp kernel: 4 XOR 4 = 0, ...).
            const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
            const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = q + __shfl_xor_sync(0xffffffffu, q, 2);
            if (I == 0) out[row] = rr;
        }

        // ---- Batched Q8_K prefill GEMM (K-quant weights x a TOKEN BATCH). ----
        // The dense-qwen35 prefill used to run launchQGemv once per prompt
        // token, re-reading every weight matrix seqLen times (pp == tg).  This
        // kernel reads each weight block ONCE per 4-token group and dots it
        // against 4 Q8_K activation planes, so prefill becomes weight-bound.
        //
        // BIT-EXACT with the per-token launchQGemv(Q8_K) path: one warp owns
        // 4 tokens x 1 weight row -- lane group (lane>>3) = token rr in
        // {0,1,2,3}, sub-lane Iid = lane&7 = the 8 sub-blocks of that row.
        // kxQ8KBlockDot is called with GROUPED=true so the 8-lane reduction
        // stays inside each token group (shuffle width 8) and every (row,
        // token) reproduces the exact single-token association:
        //   * Q4_K/Q5_K/IQ4_XS: lane-0 (Iid==0) serial block fold -- identical
        //     `running = pFin + running` chain;
        //   * Q6_K: the 8-lane tree (xor 4,1,2) of kQGemvKxQ8K, confined to the
        //     token's group.
        // So y[t*rows + row] is byte-identical to launchQGemv(m, token t).
        // `x` is [seqLen][blocksPerRow] Q8_K blocks (actStride bytes/token);
        // grid = (ceil(rows/8),), block = (32,8).  Rows must be %4-free: any
        // rows count works (each warp handles 8 rows).
        template<int TYPE>
        __global__ void kQGemmKxQ8K(const uint8_t *__restrict__ w,
                                    const uint8_t *__restrict__ x,
                                    float *__restrict__ out, uint32_t rows,
                                    uint32_t blocksPerRow, uint32_t rowBytes,
                                    uint32_t actStride, uint32_t seqLen) {
            const uint32_t lane = threadIdx.x;
            // lane group (lane>>3) = token rr in {0,1,2,3}; sub-lane Iid = the
            // 8 sub-blocks of the row.  grid.y steps 4 tokens, so every weight
            // block is dotted against 4 token activations per read (vs ONE in
            // the per-token GEMV), which is the whole prefill win.
            const uint32_t rr = lane >> 3u;
            const uint32_t Iid = lane & 7u;
            const uint32_t tok = blockIdx.y * 4u + rr;
            const bool tokValid = tok < seqLen;
            const uint32_t tokc = tokValid ? tok : 0u;
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            const uint8_t *xt = x + static_cast<uint64_t>(tokc) * actStride;
            constexpr uint32_t kBlkBytes =
                    (TYPE == kTypeQ4K)     ? kQ4K_BYTES
                    : (TYPE == kTypeQ5K)   ? kQ5K_BYTES
                    : (TYPE == kTypeIQ4XS) ? kIQ4XS_BYTES
                                           : kQ6K_BYTES;
            float sumf = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
                kxQ8KBlockDot<TYPE, false, true>(
                        r0, rp + static_cast<uint64_t>(b + 0u) * kBlkBytes,
                        xt + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE, Iid);
                kxQ8KBlockDot<TYPE, false, true>(
                        r1, rp + static_cast<uint64_t>(b + 1u) * kBlkBytes,
                        xt + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE, Iid);
                kxQ8KBlockDot<TYPE, false, true>(
                        r2, rp + static_cast<uint64_t>(b + 2u) * kBlkBytes,
                        xt + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE, Iid);
                kxQ8KBlockDot<TYPE, false, true>(
                        r3, rp + static_cast<uint64_t>(b + 3u) * kBlkBytes,
                        xt + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE, Iid);
                if (TYPE == kTypeQ6K) {
                    sumf = r0 + sumf;
                    sumf = r1 + sumf;
                    sumf = r2 + sumf;
                    sumf = r3 + sumf;
                } else if (Iid == 0) {
                    sumf = r0 + sumf;
                    sumf = r1 + sumf;
                    sumf = r2 + sumf;
                    sumf = r3 + sumf;
                }
            }
            for (; b < blocksPerRow; ++b) {
                kxQ8KBlockDot<TYPE, false, true>(
                        sumf, rp + static_cast<uint64_t>(b) * kBlkBytes,
                        xt + static_cast<uint64_t>(b) * kQ8K_STRIDE, Iid);
            }
            float *op = out + static_cast<uint64_t>(tokc) * rows + row;
            if (TYPE == kTypeQ4K || TYPE == kTypeQ5K || TYPE == kTypeIQ4XS) {
                if (Iid == 0 && tokValid) *op = sumf;
            } else {
                const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
                const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
                const float r = q + __shfl_xor_sync(0xffffffffu, q, 2);
                if (Iid == 0 && tokValid) *op = r;
            }
        }

        // ---- Q5_K row GEMV (Q8_K activation), 4 ROWS PER WARP (2026-10-12).
        // Direct analogue of kQGemvQ6KxQ8K_4xW for the Q5_K LM head.  The
        // generic kQGemvKxQ8K<kTypeQ5K> runs one row per warp and only
        // sub-lanes 0-7 issue weight loads (24 of 32 lanes idle per
        // instruction): the IQ models' Q5_K LM head (151936 rows x 6 blocks x
        // 176 B = 153-357 MB/token) measured 2.4-5.8 ms vs the ~0.36-0.85 ms
        // 448 GB/s floor, while the SAME 4xW geometry already takes the Q6_K
        // head to ~616 GB/s.  Here lane group (lane>>3) selects the row (4
        // rows/warp) and sub-lane I = lane&7 keeps the EXACT per-(row, I)
        // float chain of kxQ8KBlockDot<kTypeQ5K>: the s32q integer dot is
        // order-free and the volatile no-FMA ds*s32q / dm*bsum fold plus the
        // d8*(ms-mns) epilogue are executed by every I, so each row's output
        // is byte-identical to the generic kernel's lane-0 result.  The
        // terminal 8-lane shuffle tree (xor strides 4,1,2) stays inside each
        // 4-row group and is the SAME association as the generic kernel's
        // full-warp tree restricted to its 8 active lanes.  Requires
        // rows % 4 == 0 (all real uses: LM head 151936/152064, ffn_down 1536,
        // gate/up 8960); dispatch falls back to the generic kernel otherwise.
        // Env gate TINYCODER_Q5K_LM_4XW (default ON).
        // Group-local (4-rows-per-warp) Q5_K x Q8_K block dot, VECTORIZED.
        // Same math as kxQ8KBlockDot<kTypeQ5K, false, true> but the 32-column
        // integer dot is done 4 columns at a time with __dp4a (signed int8,
        // exact) instead of 32 scalar byte-loads + per-byte nibble/sign ops.
        // The Q5_K weight byte is (qs 4-bit nibble) | (qh bit ? 0x10), so a
        // u32 of 4 consecutive columns yields the 4 packed weights exactly:
        //   low4 = (half==0 ? wb4 : wb4>>4) & 0x0F0F0F0F
        //   nz   = ((qhb4 & (u*0x01010101)) + 0x7F7F7F7F) & 0x80808080
        //   w32  = low4 | (nz >> 3)                 // 0x10 per set qh bit
        // Because integer addition is exact and associative, s32q is bit-
        // identical to the scalar body; the dsv/mnv float fold and the
        // d8*(ms-mns) epilogue are unchanged, so every row's output matches
        // the generic kernel byte-for-byte.
        __device__ __forceinline__ void kxQ5KBlockDotVec(
                float &running, const uint8_t *__restrict__ blkW,
                const uint8_t *__restrict__ blkA, uint32_t lane) {
            const uint32_t Iid = lane & 7u;
            const int16_t *bsums =
                    reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
            const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
            int s32q = 0;
            float dsv = 0.0f, mnv = 0.0f;
            {
                const uint32_t i = Iid;
                const float d = __half2float(
                        *reinterpret_cast<const __half *>(blkW));
                const float dmin = __half2float(
                        *reinterpret_cast<const __half *>(blkW + 2));
                const uint8_t *scales = blkW + 4;
                const uint8_t *qh = blkW + 16;
                const uint8_t *qs = blkW + 48;
                uint8_t sc, m;
                if (i < 4) {
                    sc = scales[i] & 63u;
                    m = scales[i + 4] & 63u;
                } else {
                    sc = static_cast<uint8_t>(
                            (scales[i + 4] & 0xFu) |
                            ((scales[i - 4] >> 6) << 4));
                    m = static_cast<uint8_t>(
                            (scales[i + 4] >> 4) |
                            ((scales[i] >> 6) << 4));
                }
                dsv = d * static_cast<float>(sc);
                mnv = dmin * static_cast<float>(m);
                const uint32_t c = i >> 1u;
                const uint32_t half = i & 1u;
                const uint32_t u = 1u << (2u * c + half);
                const uint8_t *qsp = qs + 32u * c;
                const int8_t *yp =
                        reinterpret_cast<const int8_t *>(blkA) + i * 32u;
#pragma unroll
                for (uint32_t j = 0; j < 32u; j += 4u) {
                    // Q5_K rows are 176-B aligned (176 = 16*11), so qs (block
                    // +48) and qh (block +16) stay 16-B aligned at 4-B steps
                    // -> direct aligned u32 loads (the generic kernel's even
                    // offsets made these memcpy for WIDE; here they are
                    // provably aligned).  Same bytes, bit-identical.
                    const uint32_t wb4 =
                            *reinterpret_cast<const uint32_t *>(qsp + j);
                    const uint32_t qhb4 =
                            *reinterpret_cast<const uint32_t *>(qh + j);
                    const uint32_t yb4 =
                            *reinterpret_cast<const uint32_t *>(yp + j);
                    const uint32_t low4 =
                            (half == 0u) ? (wb4 & 0x0F0F0F0Fu)
                                         : ((wb4 >> 4) & 0x0F0F0F0Fu);
                    const uint32_t masked = qhb4 & (u * 0x01010101u);
                    const uint32_t nz =
                            (masked + 0x7F7F7F7Fu) & 0x80808080u;
                    const uint32_t w32 = low4 | (nz >> 3);
                    s32q = __dp4a(static_cast<int>(w32),
                                  static_cast<int>(yb4), s32q);
                }
            }
            int ss8[8];
            float dss[8], mns8[8];
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                ss8[i] = __shfl_sync(0xffffffffu, s32q, i, 8u);
                dss[i] = __shfl_sync(0xffffffffu, dsv, i, 8u);
                mns8[i] = __shfl_sync(0xffffffffu, mnv, i, 8u);
            }
            if (Iid == 0) {
                float ms = 0.0f, mns = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < 8; ++i) {
                    const float bi = static_cast<float>(
                            static_cast<int>(bsums[2 * i]) +
                            static_cast<int>(bsums[2 * i + 1]));
                    volatile float pMain = dss[i] * static_cast<float>(ss8[i]);
                    ms = pMain + ms;
                    volatile float pMin = mns8[i] * bi;
                    mns = pMin + mns;
                }
                volatile float pFin = d8 * (ms - mns);
                running = pFin + running;
            }
        }

        __global__ void kQGemvQ5KxQ8K_4xW(const uint8_t *__restrict__ w,
                                          const uint8_t *__restrict__ x,
                                          float *__restrict__ out, uint32_t rows,
                                          uint32_t blocksPerRow,
                                          uint32_t rowBytes) {
            const uint32_t warpRowBase = blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * 4u;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t row = base + (lane >> 3u);
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            float sumf = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
                kxQ5KBlockDotVec(
                        r0,
                        rp + static_cast<uint64_t>(b + 0u) * kQ5K_BYTES,
                        x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE, I);
                kxQ5KBlockDotVec(
                        r1,
                        rp + static_cast<uint64_t>(b + 1u) * kQ5K_BYTES,
                        x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE, I);
                kxQ5KBlockDotVec(
                        r2,
                        rp + static_cast<uint64_t>(b + 2u) * kQ5K_BYTES,
                        x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE, I);
                kxQ5KBlockDotVec(
                        r3,
                        rp + static_cast<uint64_t>(b + 3u) * kQ5K_BYTES,
                        x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE, I);
                sumf = r0 + sumf;
                sumf = r1 + sumf;
                sumf = r2 + sumf;
                sumf = r3 + sumf;
            }
            for (; b < blocksPerRow; ++b) {
                kxQ5KBlockDotVec(
                        sumf, rp + static_cast<uint64_t>(b) * kQ5K_BYTES,
                        x + static_cast<uint64_t>(b) * kQ8K_STRIDE, I);
            }
            // 8-lane shuffle-tree per row within each 4-row lane group (xor
            // strides 4,1,2 stay inside the group's 8 lanes -- identical tree
            // and store predicate to the 1-row-per-warp kernel).
            const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
            const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = q + __shfl_xor_sync(0xffffffffu, q, 2);
            if (I == 0) out[row] = rr;
        }

        // Group-local (4-rows-per-warp) Q4_K x Q8_K block dot, VECTORIZED.
        // Direct analogue of kxQ5KBlockDotVec.  Same math as
        // kxQ8KBlockDot<kTypeQ4K, false, true> (lines above) but the 32-column
        // integer dot is done 4 columns at a time with __dp4a (signed int8,
        // exact) instead of 32 scalar byte-loads + per-byte nibble/sign ops.
        // Q4_K element map (bit-exact with the scalar body and with
        // matMulVecBatchQ4K_Q8K_AVX2): for sub-block i (0..7), q-byte base
        // qsv + 32*(i>>1), column j: byte j -> low nibble when (i&1)==0 else
        // high nibble.  A u32 of 4 consecutive cols therefore yields the 4
        // packed 4-bit weights exactly:
        //   w32 = (half==0 ? wb4 : wb4>>4) & 0x0F0F0F0F
        // Because integer addition is exact and associative, s32q is bit-
        // identical to the scalar body; the dsv/mnv float fold (volatile
        // no-FMA mul-then-add) and the d8*(ms-mns) epilogue are unchanged, so
        // every row's output matches the generic kernel byte-for-byte.
        __device__ __forceinline__ void kxQ4KBlockDotVec(
                float &running, const uint8_t *__restrict__ blkW,
                const uint8_t *__restrict__ blkA, uint32_t lane) {
            const uint32_t Iid = lane & 7u;
            const int16_t *bsums =
                    reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
            const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
            int s32q = 0;
            float dsv = 0.0f, mnv = 0.0f;
            {
                const uint32_t i = Iid;
                const float d = __half2float(
                        *reinterpret_cast<const __half *>(blkW));
                const float dmin = __half2float(
                        *reinterpret_cast<const __half *>(blkW + 2));
                const uint8_t *scales = blkW + 4;
                const uint8_t *qsv = blkW + 16;
                uint8_t sc, m;
                if (i < 4) {
                    sc = scales[i] & 63u;
                    m = scales[i + 4] & 63u;
                } else {
                    sc = static_cast<uint8_t>(
                            (scales[i + 4] & 0xFu) |
                            ((scales[i - 4] >> 6) << 4));
                    m = static_cast<uint8_t>(
                            (scales[i + 4] >> 4) |
                            ((scales[i] >> 6) << 4));
                }
                dsv = d * static_cast<float>(sc);
                mnv = dmin * static_cast<float>(m);
                const uint32_t half = i & 1u;
                const uint8_t *qsp = qsv + 32u * (i >> 1u);
                const int8_t *yp =
                        reinterpret_cast<const int8_t *>(blkA) + i * 32u;
#pragma unroll
                for (uint32_t j = 0; j < 32u; j += 4u) {
                    // Q4_K rows are 144-B aligned (144 = 16*9), so qs (block
                    // +16) stays 16-B aligned at 4-B steps -> direct aligned
                    // u32 loads (same bytes as the scalar body, bit-identical).
                    const uint32_t wb4 =
                            *reinterpret_cast<const uint32_t *>(qsp + j);
                    const uint32_t yb4 =
                            *reinterpret_cast<const uint32_t *>(yp + j);
                    const uint32_t w32 =
                            (half == 0u) ? (wb4 & 0x0F0F0F0Fu)
                                         : ((wb4 >> 4) & 0x0F0F0F0Fu);
                    s32q = __dp4a(static_cast<int>(w32),
                                  static_cast<int>(yb4), s32q);
                }
            }
            int ss8[8];
            float dss[8], mns8[8];
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                ss8[i] = __shfl_sync(0xffffffffu, s32q, i, 8u);
                dss[i] = __shfl_sync(0xffffffffu, dsv, i, 8u);
                mns8[i] = __shfl_sync(0xffffffffu, mnv, i, 8u);
            }
            if (Iid == 0) {
                // Same volatile no-FMA discipline as kxQ8KBlockDot<kTypeQ4K>
                // (the compiled CPU kernel has no ffp-contract; nvcc would
                // otherwise fuse a*b+c into FFMA and break the bit-exact
                // match).  Every expression below is verbatim from the scalar
                // body so each row's output is byte-identical.
                float ms = 0.0f, mns = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < 8; ++i) {
                    const float bi = static_cast<float>(
                            static_cast<int>(bsums[2 * i]) +
                            static_cast<int>(bsums[2 * i + 1]));
                    volatile float pMain = dss[i] * static_cast<float>(ss8[i]);
                    ms = pMain + ms;
                    volatile float pMin = mns8[i] * bi;
                    mns = pMin + mns;
                }
                volatile float pFin = d8 * (ms - mns);
                running = pFin + running;
            }
        }

        // ---- Q4_K row GEMV (Q8_K activation), 4 ROWS PER WARP (2026-10-29).
        // Direct analogue of kQGemvQ5KxQ8K_4xW.  The generic
        // kQGemvKxQ8K<kTypeQ4K> runs one row per warp and only sub-lanes 0-7
        // issue weight loads (24 of 32 lanes idle per instruction) AND serial-
        // loads each 32-byte q's strip scalar (one LDG per column).  This
        // kernel brings the two levers proven on the Q6_K/Q5_K 4xW kernels to
        // the LARGEST quant class of the dense qwen35 models (Q4_K ~2.7 GB of
        // Ornith-1.5-9B's 5.9 GB):
        //   * all 32 lanes issue weight loads (8 sub-blocks x 4 rows per warp);
        //   * the 32-column integer dot is 8x __dp4a instead of 32 byte loads.
        // Bit-exact with kQGemvKxQ8K<kTypeQ4K>: the s32q integer sum is
        // order-free and the float chain (volatile no-FMA dsv*ss8 / mnv*bi +
        // d8*(ms-mns) epilogue) is verbatim, executed by every I, so each
        // row's output is byte-identical to the generic kernel's lane-0 result.
        // The terminal 8-lane shuffle tree stays inside each 4-row group and is
        // the SAME association as the generic full-warp tree restricted to its
        // 8 active lanes.  Requires rows % 4 == 0 (all real uses: dense qwen35
        // attnO/ffnDown/ffnGate/ffnUp and the Q4_K LM head).
        // Env gate TINYCODER_Q4K_4XW (default ON).
        __global__ void kQGemvQ4KxQ8K_4xW(const uint8_t *__restrict__ w,
                                          const uint8_t *__restrict__ x,
                                          float *__restrict__ out, uint32_t rows,
                                          uint32_t blocksPerRow,
                                          uint32_t rowBytes) {
            const uint32_t warpRowBase = blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * 4u;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t row = base + (lane >> 3u);
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            float sumf = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
                kxQ4KBlockDotVec(
                        r0,
                        rp + static_cast<uint64_t>(b + 0u) * kQ4K_BYTES,
                        x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE, I);
                kxQ4KBlockDotVec(
                        r1,
                        rp + static_cast<uint64_t>(b + 1u) * kQ4K_BYTES,
                        x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE, I);
                kxQ4KBlockDotVec(
                        r2,
                        rp + static_cast<uint64_t>(b + 2u) * kQ4K_BYTES,
                        x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE, I);
                kxQ4KBlockDotVec(
                        r3,
                        rp + static_cast<uint64_t>(b + 3u) * kQ4K_BYTES,
                        x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE, I);
                sumf = r0 + sumf;
                sumf = r1 + sumf;
                sumf = r2 + sumf;
                sumf = r3 + sumf;
            }
            for (; b < blocksPerRow; ++b) {
                kxQ4KBlockDotVec(
                        sumf, rp + static_cast<uint64_t>(b) * kQ4K_BYTES,
                        x + static_cast<uint64_t>(b) * kQ8K_STRIDE, I);
            }
            // 8-lane shuffle-tree per row within each 4-row lane group (xor
            // strides 4,1,2 stay inside the group's 8 lanes -- identical tree
            // and store predicate to the 1-row-per-warp kernel).
            const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
            const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = q + __shfl_xor_sync(0xffffffffu, q, 2);
            if (I == 0) out[row] = rr;
        }

        // ---- Q2_K float-dequant GEMV, 4 rows per warp (2026-09-30, A/B gate
        // TINYCODER_Q2K_4XW=1).  Replicates the bandwidth shape proven by
        // kQGemvQ6KxQ8K_4xW (~616 GB/s LM-head): all 32 lanes issue u32
        // weight loads, 8 lanes per row, 4 independent row-streams per warp,
        // NO shared memory / __syncthreads (both avoided because
        // kQGemvQ2K2xW's smem staging + barrier serialized the block).
        //
        // Q2_K element map (bit-exact with dequantizeQ2_KBlock, verified):
        //   col c -> q byte q[n*32 + (c%32) - 16*((c%32)>=16)], 2-bit shift
        //   2*j, scale scales[n*8 + 2*j + ((c%32)>=16)], where
        //   n = c/128, j = (c%128)/32.  Lane I (0..7) owns the 4 contiguous
        //   cols qOfs = 4*I of each 32-col group: for qOfs<16 the 4 cols map
        //   to q bytes [qOfs..qOfs+3] (scale sub 0), else [qOfs-16..qOfs-13]
        //   (scale sub 1) -- ONE unaligned-safe u32 loads all 4 weight bytes.
        //   Per-element float math is IDENTICAL to kQGemv<kTypeQ2K>:
        //   fmaf(fmaf(dl, qv, -ml), x[cidx], acc).  The reduce/lane-partition
        //   order differs, so results stay within the Q2_K float-path ~1e-7
        //   tolerance of the CPU reference -- same class as the existing
        //   1-row-per-warp kernel (validated by SequentialDecodeArgmaxAgrees
        //   + ParisPromptLogitsAgree).
        // MODE (P1 exp2 -> campaign 2026-10-10):
        //   0 = guarded scalar x[] loads ((cidx<cols) ? x[cidx] : 0).
        //   1 = EXACT: drop the guard when blocksPerRow*256 == cols (always
        //       true for the default gate/up 8960x1536).  Bit-identical.
        //   2 = EXACT + 128-bit activation load: the 4 k-cols of a group are
        //       contiguous, so 4 scalar x[] loads -> 1 LDG.128 (issue-bound
        //       GEMV, plan ch.4).  Bit-identical fmaf order.
        __device__ __forceinline__ float qvOf(uint32_t w4, uint32_t j,
                                              uint32_t k) {
            return static_cast<float>(static_cast<int8_t>(
                    (w4 >> (8u * k + 2u * j)) & 3u));
        }
        // ncu (2026-10-10): SM 44 % / DRAM 19 %, achieved occupancy 66.6 %,
        // register-limited to 14 blocks/SM.  __launch_bounds__(64, 16) was
        // tried to reclaim the 2 missing blocks: it caps registers at 64 and
        // the resulting spills cost more than the extra occupancy buys
        // (171.1 vs 173.1 tok/s) -> REVERTED.  Latency-bound at 44 % SM with
        // no occupancy lever left.
        template<int MODE = 0>
        __global__ void kQGemvQ2KxW4(const uint8_t *__restrict__ w,
                                     const float *__restrict__ x,
                                     float *__restrict__ out,
                                     uint32_t rows, uint32_t cols,
                                     uint32_t rowBytes,
                                     uint32_t blocksPerRow) {
            const uint32_t warpRowBase =
                    blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * 4u;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t row = base + (lane >> 3u);
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;

            // Element map (bit-exact with dequantizeQ2_KBlock): col c uses
            // q byte q[n*32 + (c%32)] (SAME offset for both 16-col halves of
            // a group; only the 2-bit shift 2*j and the scale sub differ) --
            // so the u32 load is at q + n*32 + qOfs for EVERY lane, and the
            // scale sub (0/1) is the only half-dependent index.
            const uint32_t qOfs = 4u * I;
            const uint32_t qb = qOfs;
            const uint32_t isub = (qOfs < 16u) ? 0u : 1u;

            float acc = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                // MODE 3: one accumulator per (block, n-half) -> 8 independent
                // FMA chains of 16 instead of 4 chains of 32.  The dmon decode
                // profile (sm 93% busy / mem 28% -- memory idle 72%) says the
                // SMs are stalled, not memory-bound; the serial-FMA latency per
                // chain is the only structural stall left, so halve it.
                float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 4; ++bb) {
                    const uint8_t *crp =
                            rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                    float df, dmf;
                    uint32_t scw[2][2];
                    if constexpr (MODE >= 4) {
                        // MODE 4/5: packed weight-side loads.  ONE u32 covers
                        // d|dm (the f16 pair at +80) and one u32 covers 4 scale
                        // bytes, vs 2 half loads + 4 byte loads before -- the
                        // ncu LG-throttle stall (38.6 % of the issue gap on
                        // this kernel) is an instruction-COUNT problem on the
                        // LSU queue, so cut load instructions.  All extracted
                        // values are identical -> results byte-identical.
                        uint32_t dm4;
                        std::memcpy(&dm4, crp + 80, sizeof(uint32_t));
                        df = __half2float(__ushort_as_half(
                                static_cast<unsigned short>(dm4 & 0xFFFFu)));
                        dmf = __half2float(__ushort_as_half(
                                static_cast<unsigned short>(dm4 >> 16u)));
                        std::memcpy(&scw[0][0], crp, sizeof(uint32_t));
                        std::memcpy(&scw[0][1], crp + 4, sizeof(uint32_t));
                        std::memcpy(&scw[1][0], crp + 8, sizeof(uint32_t));
                        std::memcpy(&scw[1][1], crp + 12, sizeof(uint32_t));
                    } else {
                        df = __half2float(
                                *reinterpret_cast<const __half *>(crp + 80));
                        dmf = __half2float(
                                *reinterpret_cast<const __half *>(crp + 82));
                    }
                    const uint8_t *sc = crp;
                    const uint8_t *q = crp + 16;
#pragma unroll
                    for (uint32_t n = 0; n < 2; ++n) {
                        float &slot =
                                (MODE == 3) ? (n == 0u ? (bb == 0u   ? a0
                                                          : bb == 1u ? a2
                                                          : bb == 2u ? a4
                                                                     : a6)
                                                       : (bb == 0u   ? a1
                                                          : bb == 1u ? a3
                                                          : bb == 2u ? a5
                                                                     : a7))
                                            : ((bb == 0u)   ? a0
                                               : (bb == 1u) ? a1
                                               : (bb == 2u) ? a2
                                                            : a3);
                        uint32_t w4;
                        if constexpr (MODE == 6) {
                            // MODE 6: weight loads bypass L1 (ld.global.cg,
                            // L2-only).  The ncu profile shows L1/TEX at 73-75 %
                            // on this kernel with DRAM at only 19 % -- the
                            // zero-reuse weight stream occupies the L1 pipe the
                            // broadcast x reads need.  504-B rows are 4-aligned,
                            // so u32 __ldcg is legal.  Same bytes -> byte-identical.
                            w4 = __ldcg(reinterpret_cast<const uint32_t *>(
                                    q + n * 32u + qb));
                        } else {
                            std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                        }
                        uint32_t scvA[4];
                        if constexpr (MODE >= 4) {
                            // scv[j] == sc[n*8 + 2j + isub], extracted from the
                            // two packed u32 words of this n-half.
                            const uint32_t A = scw[n][0], Bv = scw[n][1];
                            if (isub == 0u) {
                                scvA[0] = A & 0xFFu;
                                scvA[1] = (A >> 16) & 0xFFu;
                                scvA[2] = Bv & 0xFFu;
                                scvA[3] = (Bv >> 16) & 0xFFu;
                            } else {
                                scvA[0] = (A >> 8) & 0xFFu;
                                scvA[1] = (A >> 24) & 0xFFu;
                                scvA[2] = (Bv >> 8) & 0xFFu;
                                scvA[3] = (Bv >> 24) & 0xFFu;
                            }
                        }
                        const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            const uint8_t scv = (MODE >= 4)
                                                        ? static_cast<uint8_t>(scvA[j])
                                                : (MODE == 6)
                                                        ? static_cast<uint8_t>(__ldcg(
                                                                  reinterpret_cast<const char *>(
                                                                          sc + n * 8u + 2u * j +
                                                                          isub)))
                                                        : sc[n * 8u + 2u * j + isub];
                            const float dl =
                                    df * static_cast<float>(scv & 0xFu);
                            const float ml =
                                    dmf * static_cast<float>(scv >> 4u);
                            const uint32_t cbase = ebase + j * 32u + qOfs;
                            if constexpr (MODE == 2 || MODE == 5) {
                                // MODE 2: the 4 k-cols are CONTIGUOUS (cbase
                                // is a multiple of 4), so the 4 scalar x[]
                                // loads become ONE LDG.128.  Attack on the
                                // issue-bound Q2_K GEMV (SM ~97% busy / DRAM
                                // ~20%, plan ch.4): activation-load issue slots
                                // drop 4x.  Same 4 fmaf in the same order ->
                                // parity byte-identical.
                                const float4 xv = *reinterpret_cast<const float4 *>(
                                        x + cbase);
                                const float m0 = fmaf(dl, qvOf(w4, j, 0), -ml);
                                const float m1 = fmaf(dl, qvOf(w4, j, 1), -ml);
                                const float m2 = fmaf(dl, qvOf(w4, j, 2), -ml);
                                const float m3 = fmaf(dl, qvOf(w4, j, 3), -ml);
                                slot = fmaf(m0, xv.x, slot);
                                slot = fmaf(m1, xv.y, slot);
                                slot = fmaf(m2, xv.z, slot);
                                slot = fmaf(m3, xv.w, slot);
                            } else {
#pragma unroll
                                for (uint32_t k = 0; k < 4; ++k) {
                                    const float qv = qvOf(w4, j, k);
                                    const uint32_t cidx = cbase + k;
                                    const float xv = (MODE == 1 || MODE == 6 ||
                                                      MODE >= 4)
                                                             ? x[cidx]
                                                             : ((cidx < cols)
                                                                        ? x[cidx]
                                                                        : 0.0f);
                                    slot = fmaf(fmaf(dl, qv, -ml), xv, slot);
                                }
                            }
                        }
                    }
                }
                acc += a0;
                acc += a1;
                acc += a2;
                acc += a3;
                if constexpr (MODE == 3) {
                    acc += a4;
                    acc += a5;
                    acc += a6;
                    acc += a7;
                }
                rp += 4u * kQ2K_BYTES;
            }
            for (; b < blocksPerRow; ++b) {
                float df, dmf;
                uint32_t scw[2][2];
                if constexpr (MODE >= 4) {
                    // Packed weight loads -- see the 4-block loop above.
                    uint32_t dm4;
                    std::memcpy(&dm4, rp + 80, sizeof(uint32_t));
                    df = __half2float(__ushort_as_half(
                            static_cast<unsigned short>(dm4 & 0xFFFFu)));
                    dmf = __half2float(__ushort_as_half(
                            static_cast<unsigned short>(dm4 >> 16u)));
                    std::memcpy(&scw[0][0], rp, sizeof(uint32_t));
                    std::memcpy(&scw[0][1], rp + 4, sizeof(uint32_t));
                    std::memcpy(&scw[1][0], rp + 8, sizeof(uint32_t));
                    std::memcpy(&scw[1][1], rp + 12, sizeof(uint32_t));
                } else if (MODE == 6) {
                    uint32_t dm4 = __ldcg(
                            reinterpret_cast<const uint32_t *>(rp + 80));
                    df = __half2float(__ushort_as_half(
                            static_cast<unsigned short>(dm4 & 0xFFFFu)));
                    dmf = __half2float(__ushort_as_half(
                            static_cast<unsigned short>(dm4 >> 16u)));
                } else {
                    df = __half2float(
                            *reinterpret_cast<const __half *>(rp + 80));
                    dmf = __half2float(
                            *reinterpret_cast<const __half *>(rp + 82));
                }
                const uint8_t *sc = rp;
                const uint8_t *q = rp + 16;
#pragma unroll
                for (uint32_t n = 0; n < 2; ++n) {
                    uint32_t w4;
                    if constexpr (MODE == 6) {
                        w4 = __ldcg(reinterpret_cast<const uint32_t *>(
                                q + n * 32u + qb));
                    } else {
                        std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                    }
                    uint32_t scvA[4];
                    if constexpr (MODE >= 4) {
                        const uint32_t A = scw[n][0], Bv = scw[n][1];
                        if (isub == 0u) {
                            scvA[0] = A & 0xFFu;
                            scvA[1] = (A >> 16) & 0xFFu;
                            scvA[2] = Bv & 0xFFu;
                            scvA[3] = (Bv >> 16) & 0xFFu;
                        } else {
                            scvA[0] = (A >> 8) & 0xFFu;
                            scvA[1] = (A >> 24) & 0xFFu;
                            scvA[2] = (Bv >> 8) & 0xFFu;
                            scvA[3] = (Bv >> 24) & 0xFFu;
                        }
                    }
                    const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        const uint8_t scv = (MODE >= 4)
                                                    ? static_cast<uint8_t>(scvA[j])
                                            : (MODE == 6)
                                                    ? static_cast<uint8_t>(__ldcg(
                                                              reinterpret_cast<const char *>(
                                                                      sc + n * 8u + 2u * j + isub)))
                                                    : sc[n * 8u + 2u * j + isub];
                        const float dl = df * static_cast<float>(scv & 0xFu);
                        const float ml = dmf * static_cast<float>(scv >> 4u);
                        const uint32_t cbase = ebase + j * 32u + qOfs;
                        if constexpr (MODE == 2 || MODE == 5) {
                            // One LDG.128 instead of 4 scalar x[] loads
                            // (identical fmaf order; parity byte-identical).
                            const float4 xv = *reinterpret_cast<const float4 *>(
                                    x + cbase);
                            const float m0 = fmaf(dl, qvOf(w4, j, 0), -ml);
                            const float m1 = fmaf(dl, qvOf(w4, j, 1), -ml);
                            const float m2 = fmaf(dl, qvOf(w4, j, 2), -ml);
                            const float m3 = fmaf(dl, qvOf(w4, j, 3), -ml);
                            acc = fmaf(m0, xv.x, acc);
                            acc = fmaf(m1, xv.y, acc);
                            acc = fmaf(m2, xv.z, acc);
                            acc = fmaf(m3, xv.w, acc);
                        } else {
#pragma unroll
                            for (uint32_t k = 0; k < 4; ++k) {
                                const float qv = qvOf(w4, j, k);
                                const uint32_t cidx = cbase + k;
                                const float xv = (MODE == 1 || MODE == 6 ||
                                                  MODE >= 4)
                                                         ? x[cidx]
                                                         : ((cidx < cols)
                                                                    ? x[cidx]
                                                                    : 0.0f);
                                acc = fmaf(fmaf(dl, qv, -ml), xv, acc);
                            }
                        }
                    }
                }
                rp += kQ2K_BYTES;
            }
            // 8-lane shuffle tree per row (xor 4,1,2 stays inside the row's
            // 8-lane group -- same tree as kQGemvQ6KxQ8K_4xW).
            const float p = acc + __shfl_xor_sync(0xffffffffu, acc, 4);
            const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
            if (I == 0) out[row] = rr;
        }

        // ---- REPACKED 16-byte-lane Q2_K GEMV (2026-10-10, §9.19,
        // TINYCODER_Q2K_RPACK=1).  The ncu profile shows every Q2_K GEMV
        // L1-pipe bound (73-75 %) with 4-B-per-lane weight loads.  The repack
        // (kRepackQ2K, run once per matrix) gives lane I of each 8-lane row
        // group 16 CONTIGUOUS bytes per block -- 8 B q (h0@4I, h1@32+4I) + 8 B
        // scales (sc[2j+isub], sc[8+2j+isub]) -- plus a broadcast 4 B dm, in
        // 136-B 8-aligned blocks.  The GEMV then loads u64 q + u64 scales +
        // u32 dm per block (3 weight loads vs 12), each warp-instruction
        // covering 64 contiguous bytes per row.  Column mapping, fmaf order
        // and reduce tree are IDENTICAL to kQGemvQ2KxW4<1> -> byte-identical.
        __global__ void kRepackQ2K(const uint8_t *__restrict__ w,
                                   uint8_t *__restrict__ dst,
                                   uint32_t rowBytes, uint32_t blocksPerRow) {
            const uint32_t row = blockIdx.x;
            const uint32_t b = blockIdx.y;
            const uint32_t I = threadIdx.x & 7u;
            const uint32_t isub = I >> 2u;// (4I < 16) <=> I < 4
            const uint8_t *crp = w + static_cast<uint64_t>(row) * rowBytes +
                                 static_cast<uint64_t>(b) * kQ2K_BYTES;
            uint8_t *dblk =
                    dst + (static_cast<uint64_t>(row) * blocksPerRow + b) * 136u;
            std::memcpy(dblk + 16u * I, crp + 16u + 4u * I, 4);           // h0 q bytes
            std::memcpy(dblk + 16u * I + 4u, crp + 16u + 32u + 4u * I, 4);// h1
            for (uint32_t m = 0; m < 8; ++m) {
                dblk[16u * I + 8u + m] =
                        crp[(m < 4u) ? (2u * m + isub)
                                     : (8u + 2u * (m - 4u) + isub)];
            }
            if (threadIdx.x == 0) std::memcpy(dblk + 128u, crp + 80u, 4);// dm
        }

        template<bool EXACT = false>
        __global__ void __launch_bounds__(64) kQGemvQ2KxW4_RP(
                const uint8_t *__restrict__ w, const float *__restrict__ x,
                float *__restrict__ out, uint32_t rows, uint32_t cols,
                uint32_t blocksPerRow) {
            constexpr uint32_t kRP_BLK = 136u;
            const uint32_t warpRowBase = blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * 4u;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t row = base + (lane >> 3u);
            const uint8_t *rp = w + static_cast<uint64_t>(row) * blocksPerRow *
                                            kRP_BLK;
            const uint32_t qOfs = 4u * I;
            const uint32_t isub = I >> 2u;
            (void) isub;
            float acc = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 4; ++bb) {
                    const uint8_t *blk = rp + static_cast<uint64_t>(b + bb) *
                                                      kRP_BLK;
                    // 3 weight loads per block (vs 12): u64 q, u64 scales,
                    // u32 dm (broadcast across the row's 8 lanes).
                    unsigned long long q8, s8;
                    std::memcpy(&q8, blk + 16u * I, sizeof(unsigned long long));
                    std::memcpy(&s8, blk + 16u * I + 8u,
                                sizeof(unsigned long long));
                    uint32_t dm4;
                    std::memcpy(&dm4, blk + 128u, sizeof(uint32_t));
                    const float df = __half2float(__ushort_as_half(
                            static_cast<unsigned short>(dm4 & 0xFFFFu)));
                    const float dmf = __half2float(__ushort_as_half(
                            static_cast<unsigned short>(dm4 >> 16u)));
                    float &slot = (bb == 0u)   ? a0
                                  : (bb == 1u) ? a1
                                  : (bb == 2u) ? a2
                                               : a3;
#pragma unroll
                    for (uint32_t n = 0; n < 2; ++n) {
                        const uint32_t w4 = static_cast<uint32_t>(
                                (n == 0u) ? q8 : (q8 >> 32u));
                        const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            const uint32_t scv =
                                    (s8 >> (8u * (4u * n + j))) & 0xFFu;
                            const float dl =
                                    df * static_cast<float>(scv & 0xFu);
                            const float ml =
                                    dmf * static_cast<float>(scv >> 4u);
                            const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                            for (uint32_t k = 0; k < 4; ++k) {
                                const float qv = qvOf(w4, j, k);
                                const uint32_t cidx = cbase + k;
                                const float xv = EXACT ? x[cidx]
                                                       : ((cidx < cols)
                                                                  ? x[cidx]
                                                                  : 0.0f);
                                slot = fmaf(fmaf(dl, qv, -ml), xv, slot);
                            }
                        }
                    }
                }
                acc += a0;
                acc += a1;
                acc += a2;
                acc += a3;
            }
            for (; b < blocksPerRow; ++b) {
                const uint8_t *blk = rp + static_cast<uint64_t>(b) * kRP_BLK;
                unsigned long long q8, s8;
                std::memcpy(&q8, blk + 16u * I, sizeof(unsigned long long));
                std::memcpy(&s8, blk + 16u * I + 8u, sizeof(unsigned long long));
                uint32_t dm4;
                std::memcpy(&dm4, blk + 128u, sizeof(uint32_t));
                const float df = __half2float(__ushort_as_half(
                        static_cast<unsigned short>(dm4 & 0xFFFFu)));
                const float dmf = __half2float(__ushort_as_half(
                        static_cast<unsigned short>(dm4 >> 16u)));
#pragma unroll
                for (uint32_t n = 0; n < 2; ++n) {
                    const uint32_t w4 =
                            static_cast<uint32_t>((n == 0u) ? q8 : (q8 >> 32u));
                    const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        const uint32_t scv = (s8 >> (8u * (4u * n + j))) & 0xFFu;
                        const float dl = df * static_cast<float>(scv & 0xFu);
                        const float ml = dmf * static_cast<float>(scv >> 4u);
                        const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                        for (uint32_t k = 0; k < 4; ++k) {
                            const float qv = qvOf(w4, j, k);
                            const uint32_t cidx = cbase + k;
                            const float xv = EXACT ? x[cidx]
                                                   : ((cidx < cols) ? x[cidx]
                                                                    : 0.0f);
                            acc = fmaf(fmaf(dl, qv, -ml), xv, acc);
                        }
                    }
                }
            }
            const float p = acc + __shfl_xor_sync(0xffffffffu, acc, 4);
            const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
            if (I == 0) out[row] = rr;
        }

        // ---- Fused-residual GEMV epilogues (2026-10-09, per-layer launch/
        // drain campaign; opt-in via TINYCODER_FUSE_ATTNO_RESID and
        // TINYCODER_FUSE_DOWN_RESID, DEFAULT OFF).  These are EXACT copies of
        // kQGemvQ2KxW4<EXACT> / kQGemv<kTypeQ3K,EXACT> that additionally fold
        // the following kAddResidual into the GEMV epilogue, deleting one
        // kernel launch per FFN/attn block:
        //   * decode attnO : s.attnProj = Wo*attnOut ; s.hidden += s.attnProj
        //   * decode down  : s.ffnOut  = Wd*gate    ; s.hidden += s.ffnOut
        // The dot math, per-lane partition and reduce tree are IDENTICAL to
        // the base kernels, so out[row] is written byte-for-byte as before
        // (out is NOT dead -- kAddResidual reads src=out); the residual add
        // is the same `acc[row] + rr` float add kAddResidual performs, hence
        // the residual stream is byte-identical too (parity is asserted by the
        // usual greedy-token probe: deterministic argmax => identical stream).
        template<bool EXACT = false>
        __global__ void kQGemvQ2KxW4Resid(const uint8_t *__restrict__ w,
                                          const float *__restrict__ x,
                                          float *__restrict__ out,
                                          float *__restrict__ acc,
                                          uint32_t rows, uint32_t cols,
                                          uint32_t rowBytes,
                                          uint32_t blocksPerRow) {
            const uint32_t warpRowBase =
                    blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * 4u;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t row = base + (lane >> 3u);
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t qOfs = 4u * I;
            const uint32_t qb = qOfs;
            const uint32_t isub = (qOfs < 16u) ? 0u : 1u;

            float accv = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 4; ++bb) {
                    const uint8_t *crp =
                            rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
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
                    for (uint32_t n = 0; n < 2; ++n) {
                        uint32_t w4;
                        std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                        const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            const uint8_t scv = sc[n * 8u + 2u * j + isub];
                            const float dl =
                                    df * static_cast<float>(scv & 0xFu);
                            const float ml =
                                    dmf * static_cast<float>(scv >> 4u);
                            const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                            for (uint32_t k = 0; k < 4; ++k) {
                                const float qv = static_cast<float>(
                                        static_cast<int8_t>(
                                                (w4 >> (8u * k + 2u * j)) & 3u));
                                const uint32_t cidx = cbase + k;
                                slot = fmaf(fmaf(dl, qv, -ml),
                                            EXACT ? x[cidx]
                                                  : ((cidx < cols) ? x[cidx]
                                                                   : 0.0f),
                                            slot);
                            }
                        }
                    }
                }
                accv += a0;
                accv += a1;
                accv += a2;
                accv += a3;
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
                for (uint32_t n = 0; n < 2; ++n) {
                    uint32_t w4;
                    std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                    const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        const uint8_t scv = sc[n * 8u + 2u * j + isub];
                        const float dl = df * static_cast<float>(scv & 0xFu);
                        const float ml = dmf * static_cast<float>(scv >> 4u);
                        const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                        for (uint32_t k = 0; k < 4; ++k) {
                            const float qv = static_cast<float>(
                                    static_cast<int8_t>(
                                            (w4 >> (8u * k + 2u * j)) & 3u));
                            const uint32_t cidx = cbase + k;
                            accv = fmaf(fmaf(dl, qv, -ml),
                                        EXACT ? x[cidx]
                                              : ((cidx < cols) ? x[cidx] : 0.0f),
                                        accv);
                        }
                    }
                }
                rp += kQ2K_BYTES;
            }
            // 8-lane shuffle tree per row -- IDENTICAL to kQGemvQ2KxW4.
            const float p = accv + __shfl_xor_sync(0xffffffffu, accv, 4);
            const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
            if (I == 0) {
                out[row] = rr;
                acc[row] = acc[row] + rr;
            }
        }

        // Q3_K fused-residual (ffnDown).  1-row-per-warp, 32-lane reduce --
        // IDENTICAL to kQGemv<kTypeQ3K,EXACT>; only the epilogue adds the
        // residual.  See the Q2_KResid comment above for the parity argument.
        template<bool EXACT = false>
        __global__ void kQGemvQ3KResid(const uint8_t *__restrict__ w,
                                       const float *__restrict__ x,
                                       float *__restrict__ out,
                                       float *__restrict__ acc, uint32_t rows,
                                       uint32_t cols, uint32_t rowBytes,
                                       uint32_t blocksPerRow) {
            uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            float accv = 0.0f;
            const uint32_t lane = threadIdx.x;
            const uint32_t sub = lane / 16u;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                for (uint32_t bb = 0; bb < 4; ++bb) {
                    const uint8_t *crp =
                            rp + static_cast<uint64_t>(bb) * kQ3K_BYTES;
                    __half d = *reinterpret_cast<const __half *>(crp + 108);
                    float df = __half2float(d);
                    const uint8_t *hm = crp;
                    const uint8_t *q = crp + 32;
                    const uint8_t *scales = crp + 96;
                    uint32_t aux[4];
                    std::memcpy(&aux[0], scales, 12);
                    uint32_t tmp = aux[2];
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
                    const uint8_t hmb = hm[lane];
                    float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                            : (bb == 2u)       ? a2
                                                               : a3;
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            float dl = df * static_cast<float>(
                                                    sc16[half * 8u + jj * 2u + sub] - 32);
                            const uint32_t maskBit = 1u << (half * 4u + jj);
                            float qv = static_cast<float>(static_cast<int8_t>(
                                    ((qb >> (jj * 2u)) & 3) -
                                    ((hmb & maskBit) ? 0 : 4)));
                            const uint32_t cidx = (b + bb) * 256u +
                                                  half * 128u + jj * 32u + lane;
                            slot = fmaf(dl * qv,
                                        EXACT ? x[cidx]
                                              : ((cidx < cols) ? x[cidx] : 0.0f),
                                        slot);
                        }
                    }
                }
                accv += a0;
                accv += a1;
                accv += a2;
                accv += a3;
                rp += 4u * kQ3K_BYTES;
            }
            for (; b < blocksPerRow; ++b) {
                __half d = *reinterpret_cast<const __half *>(rp + 108);
                float df = __half2float(d);
                const uint8_t *hm = rp;
                const uint8_t *q = rp + 32;
                const uint8_t *scales = rp + 96;
                uint32_t aux[4];
                std::memcpy(&aux[0], scales, 12);
                uint32_t tmp = aux[2];
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
                const uint8_t hmb = hm[lane];
#pragma unroll
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                    for (uint32_t jj = 0; jj < 4; ++jj) {
                        float dl = df * static_cast<float>(
                                                sc16[half * 8u + jj * 2u + sub] - 32);
                        const uint32_t maskBit = 1u << (half * 4u + jj);
                        float qv = static_cast<float>(static_cast<int8_t>(
                                ((qb >> (jj * 2u)) & 3) -
                                ((hmb & maskBit) ? 0 : 4)));
                        const uint32_t cidx =
                                b * 256u + half * 128u + jj * 32u + lane;
                        accv = fmaf(dl * qv,
                                    EXACT ? x[cidx]
                                          : ((cidx < cols) ? x[cidx] : 0.0f),
                                    accv);
                    }
                }
                rp += kQ3K_BYTES;
            }
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1) {
                accv += __shfl_xor_sync(0xffffffffu, accv, off);
            }
            if (lane == 0) {
                out[row] = accv;
                acc[row] = acc[row] + accv;
            }
        }

        // ---- Q3_K 4-rows-per-warp decode GEMV (2026-10-06, P1 bandwidth
        // campaign; A/B gate TINYCODER_Q3K_4XW=0 disables, DEFAULT ON).  The
        // 1-row-per-warp kQGemv<kTypeQ3K> measures ~103 GB/s on the FFN down
        // projection (rows=1536), while the byte-shape-identical Q2_K 4xW
        // kernel reaches ~145 GB/s on the gate/up (rows=8960).  The only
        // structural difference was the warp geometry: this kernel gives all
        // 32 lanes a u32 weight load (8 lanes/row, 4 independent row-streams
        // per warp), NO shared memory / __syncthreads, mirroring
        // kQGemvQ2KxW4 exactly.
        //
        // Q3_K layout (bit-exact with the kQGemv<kTypeQ3K> branch above):
        //   hm   (32 B at block+0):  high-bit mask, bit 1<<jj for the (half,
        //                            lane) position with qv = low2 - (bit?0:4)
        //   q    (64 B at block+32): 2-bit quants, q[half*32 + lane]
        //   scales (12 B at block+96): repacked int8 scales, decoded via the
        //                            same aux[4] shuffle to sc16[16]
        //   d    (f16 at block+108)
        // Lane I (0..7) owns 4 contiguous columns per 32-col group:
        //   qOfs = 4*I;  sub = (qOfs < 16) ? 0 : 1.
        // The hmask bit (1<<jj) is the SAME for all 4 columns of a lane's
        // slice (it indexes jj, not the column), so one hm byte load per
        // (block, half) suffices.  Per-element math is IDENTICAL to the
        // 1-row-per-warp kernel (same int8 sub-32 scale, same hmask fold,
        // same fmaf); only the lane partition/reduce order differs, so results
        // stay within the Q3_K float-path tolerance (validated by
        // SequentialDecodeArgmaxAgrees + ParisPromptLogitsAgree).
        __global__ void kQGemvQ3KxW4(const uint8_t *__restrict__ w,
                                     const float *__restrict__ x,
                                     float *__restrict__ out,
                                     uint32_t rows, uint32_t cols,
                                     uint32_t rowBytes,
                                     uint32_t blocksPerRow) {
            const uint32_t warpRowBase =
                    blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * 4u;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t row = base + (lane >> 3u);
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;

            const uint32_t qOfs = 4u * I;
            const uint32_t isub = (qOfs < 16u) ? 0u : 1u;

            float acc = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 4; ++bb) {
                    const uint8_t *crp =
                            rp + static_cast<uint64_t>(bb) * kQ3K_BYTES;
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
                    for (uint32_t half = 0; half < 2; ++half) {
                        // ONE u32 load fetches the lane's 4 contiguous 2-bit
                        // quants (cols qOfs..qOfs+3) -- the Q2_K 4xW shape.
                        // The hmask byte is per-COLUMN (hm[qOfs+k]); the mask
                        // bit (1<<jj) indexes jj, not the column.
                        uint32_t w4;
                        std::memcpy(&w4, q + half * 32u + qOfs,
                                    sizeof(uint32_t));
                        const uint32_t ebase = (b + bb) * 256u + half * 128u;
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            const float dl = df * static_cast<float>(
                                                          sc16[half * 8u + jj * 2u + isub] - 32);
                            const uint32_t maskBit = 1u << (half * 4u + jj);
                            const uint32_t cbase = ebase + jj * 32u + qOfs;
#pragma unroll
                            for (uint32_t k = 0; k < 4; ++k) {
                                const float qv = static_cast<float>(
                                        static_cast<int8_t>(
                                                ((w4 >> (8u * k + 2u * jj)) & 3u) -
                                                ((hm[qOfs + k] & maskBit) ? 0 : 4)));
                                const uint32_t cidx = cbase + k;
                                slot = fmaf(dl * qv,
                                            (cidx < cols) ? x[cidx] : 0.0f,
                                            slot);
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
                for (uint32_t half = 0; half < 2; ++half) {
                    uint32_t w4;
                    std::memcpy(&w4, q + half * 32u + qOfs, sizeof(uint32_t));
                    const uint32_t ebase = b * 256u + half * 128u;
#pragma unroll
                    for (uint32_t jj = 0; jj < 4; ++jj) {
                        const float dl = df * static_cast<float>(
                                                      sc16[half * 8u + jj * 2u + isub] - 32);
                        const uint32_t maskBit = 1u << (half * 4u + jj);
                        const uint32_t cbase = ebase + jj * 32u + qOfs;
#pragma unroll
                        for (uint32_t k = 0; k < 4; ++k) {
                            const float qv = static_cast<float>(
                                    static_cast<int8_t>(
                                            ((w4 >> (8u * k + 2u * jj)) & 3u) -
                                            ((hm[qOfs + k] & maskBit) ? 0 : 4)));
                            const uint32_t cidx = cbase + k;
                            acc = fmaf(dl * qv,
                                       (cidx < cols) ? x[cidx] : 0.0f, acc);
                        }
                    }
                }
                rp += kQ3K_BYTES;
            }
            const float p = acc + __shfl_xor_sync(0xffffffffu, acc, 4);
            const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
            if (I == 0) out[row] = rr;
        }

        // ---- Q2_K MULTI-ROW-PER-WARP decode GEMV (2026-09-30, A/B gate
        // TINYCODER_Q2K_NM=<RPB>, DEFAULT OFF).  The 4xW kernel gives every
        // lane ONE row's 4-col slice; the FFN gate/up measured ~78 GB/s vs
        // the ~616 GB/s Q6_K LM-head ceiling, i.e. the FFN is limited by
        // MEMORY-LATENCY-BOUND occupancy (only ~8 warps/SM resident at ~90
        // regs) rather than by DRAM bandwidth.  The Q6_K LM head reaches
        // 616 GB/s with the SAME 4xW geometry purely because it has 4748
        // full blocks (~68 per SM) to fill every latency-hiding slot.
        //
        // This kernel raises the per-WARP memory-level parallelism instead:
        // the 4 lane groups each own R = RPB/4 independent rows, so every
        // lane issues R interleaved weight streams (4 blocks x R rows per
        // outer iteration) with NO shared memory and NO __syncthreads --
        // exactly the pattern that wins on the FFN's tiny 280-block grid
        // (68 SMs x ~4 resident blocks).  RPB must divide `rows` (8960 and
        // 1536 both do: RPB in {8,16,32}).
        //
        // Per-element float math is IDENTICAL to kQGemvQ2KxW4 (same u32
        // q-byte load, same 2-bit shift, same scale sub-index, same fmaf
        // chain and in-order 4-block fold); only the per-lane accumulation
        // state is replicated R-fold.  The 8-lane shuffle tree per row is
        // unchanged, so results stay within the Q2_K float-path ~1e-7
        // tolerance (validated by SequentialDecodeArgmaxAgrees +
        // ParisPromptLogitsAgree).
        template<uint32_t RPB>
        __global__ void kQGemvQ2KxW4_Nm(const uint8_t *__restrict__ w,
                                        const float *__restrict__ x,
                                        float *__restrict__ out, uint32_t rows,
                                        uint32_t cols, uint32_t rowBytes,
                                        uint32_t blocksPerRow) {
            constexpr uint32_t R = RPB / 4u;// rows owned by each lane group
            const uint32_t warpRowBase =
                    blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * RPB;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t g = lane >> 3u;// 0..3 lane group
            const uint32_t I = lane & 7u; // sub-lane within the row
            const uint32_t qOfs = 4u * I;
            const uint32_t qb = qOfs;
            const uint32_t isub = (qOfs < 16u) ? 0u : 1u;
            const uint32_t row0 = base + g * R;

            float acc[R];
            const uint8_t *rp[R];
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                acc[r] = 0.0f;
                rp[r] = w + static_cast<uint64_t>(row0 + r) * rowBytes;
            }

            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
#pragma unroll
                for (uint32_t r = 0; r < R; ++r) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                    for (uint32_t bb = 0; bb < 4; ++bb) {
                        const uint8_t *crp =
                                rp[r] + static_cast<uint64_t>(bb) * kQ2K_BYTES;
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
                        for (uint32_t n = 0; n < 2; ++n) {
                            uint32_t w4;
                            std::memcpy(&w4, q + n * 32u + qb,
                                        sizeof(uint32_t));
                            const uint32_t ebase =
                                    (b + bb) * 256u + n * 128u;
#pragma unroll
                            for (uint32_t j = 0; j < 4; ++j) {
                                const uint8_t scv = sc[n * 8u + 2u * j + isub];
                                const float dl =
                                        df * static_cast<float>(scv & 0xFu);
                                const float ml =
                                        dmf * static_cast<float>(scv >> 4u);
                                const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                                for (uint32_t k = 0; k < 4; ++k) {
                                    const float qv = static_cast<float>(
                                            static_cast<int8_t>(
                                                    (w4 >> (8u * k + 2u * j)) &
                                                    3u));
                                    const uint32_t cidx = cbase + k;
                                    slot = fmaf(
                                            fmaf(dl, qv, -ml),
                                            (cidx < cols) ? x[cidx] : 0.0f,
                                            slot);
                                }
                            }
                        }
                    }
                    acc[r] += a0;
                    acc[r] += a1;
                    acc[r] += a2;
                    acc[r] += a3;
                    rp[r] += 4u * kQ2K_BYTES;
                }
            }
            for (; b < blocksPerRow; ++b) {
#pragma unroll
                for (uint32_t r = 0; r < R; ++r) {
                    const float df = __half2float(
                            *reinterpret_cast<const __half *>(rp[r] + 80));
                    const float dmf = __half2float(
                            *reinterpret_cast<const __half *>(rp[r] + 82));
                    const uint8_t *sc = rp[r];
                    const uint8_t *q = rp[r] + 16;
#pragma unroll
                    for (uint32_t n = 0; n < 2; ++n) {
                        uint32_t w4;
                        std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                        const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            const uint8_t scv = sc[n * 8u + 2u * j + isub];
                            const float dl =
                                    df * static_cast<float>(scv & 0xFu);
                            const float ml =
                                    dmf * static_cast<float>(scv >> 4u);
                            const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                            for (uint32_t k = 0; k < 4; ++k) {
                                const float qv = static_cast<float>(
                                        static_cast<int8_t>(
                                                (w4 >> (8u * k + 2u * j)) &
                                                3u));
                                const uint32_t cidx = cbase + k;
                                acc[r] = fmaf(fmaf(dl, qv, -ml),
                                              (cidx < cols) ? x[cidx] : 0.0f,
                                              acc[r]);
                            }
                        }
                    }
                    rp[r] += kQ2K_BYTES;
                }
            }
            // 8-lane shuffle tree per row (xor 4,1,2 stays inside the row's
            // 8-lane group -- same tree as kQGemvQ2KxW4).
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                const float p = acc[r] + __shfl_xor_sync(0xffffffffu, acc[r], 4);
                const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
                const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
                if (I == 0) out[row0 + r] = rr;
            }
        }

        // ---- Q2_K 4-WAY K-SPLIT decode GEMV (2026-09-30, A/B gate
        // TINYCODER_Q2K_SPLIT4=1, DEFAULT OFF). (round-robin) so the per-lane
        // SERIAL block chain is ~4x SHORTER than kQGemvQ2KxW4 (which walks all
        // blocksPerRow blocks serially). This is the latency-hiding lever the
        // multi-row kQGemvQ2KxW4_Nm attempt did NOT provide (it kept the full
        // serial chain per lane). One row per warp (8 rows/block), no smem.
        //
        // K-splitting the warp cuts the FFN's serial chain 4x.
        //
        // Per-element float math is IDENTICAL to kQGemvQ2KxW4.  The
        // cross-group join reorders the accumulation (~1e-7 float tolerance,
        // same class as the other Q2_K float paths).
        __global__ void kQGemvQ2KxSplit4(const uint8_t *__restrict__ w,
                                         const float *__restrict__ x,
                                         float *__restrict__ out, uint32_t rows,
                                         uint32_t cols, uint32_t rowBytes,
                                         uint32_t blocksPerRow) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t g = lane >> 3u;// 0..3 lane group
            const uint32_t I = lane & 7u; // sub-lane within the row
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;

            const uint32_t qOfs = 4u * I;
            const uint32_t qb = qOfs;
            const uint32_t isub = (qOfs < 16u) ? 0u : 1u;

            float acc = 0.0f;
            // Round-robin block ownership: group g owns b = g, g+4, g+8...
            // 4-way batching over the group's own blocks (block stride 4).
            uint32_t b = g;
            for (; b + 12u <= blocksPerRow; b += 16u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 4; ++bb) {
                    const uint8_t *crp =
                            rp + static_cast<uint64_t>(b + 4u * bb) * kQ2K_BYTES;
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
                    for (uint32_t n = 0; n < 2; ++n) {
                        uint32_t w4;
                        std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                        const uint32_t ebase = (b + 4u * bb) * 256u + n * 128u;
#pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            const uint8_t scv = sc[n * 8u + 2u * j + isub];
                            const float dl = df * static_cast<float>(scv & 0xFu);
                            const float ml = dmf * static_cast<float>(scv >> 4u);
                            const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                            for (uint32_t k = 0; k < 4; ++k) {
                                const float qv = static_cast<float>(
                                        static_cast<int8_t>(
                                                (w4 >> (8u * k + 2u * j)) & 3u));
                                const uint32_t cidx = cbase + k;
                                slot = fmaf(fmaf(dl, qv, -ml),
                                            (cidx < cols) ? x[cidx] : 0.0f, slot);
                            }
                        }
                    }
                }
                acc += a0;
                acc += a1;
                acc += a2;
                acc += a3;
            }
            for (; b < blocksPerRow; b += 4u) {
                const uint8_t *crp = rp + static_cast<uint64_t>(b) * kQ2K_BYTES;
                const float df = __half2float(
                        *reinterpret_cast<const __half *>(crp + 80));
                const float dmf = __half2float(
                        *reinterpret_cast<const __half *>(crp + 82));
                const uint8_t *sc = crp;
                const uint8_t *q = crp + 16;
#pragma unroll
                for (uint32_t n = 0; n < 2; ++n) {
                    uint32_t w4;
                    std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                    const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        const uint8_t scv = sc[n * 8u + 2u * j + isub];
                        const float dl = df * static_cast<float>(scv & 0xFu);
                        const float ml = dmf * static_cast<float>(scv >> 4u);
                        const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                        for (uint32_t k = 0; k < 4; ++k) {
                            const float qv = static_cast<float>(
                                    static_cast<int8_t>(
                                            (w4 >> (8u * k + 2u * j)) & 3u));
                            const uint32_t cidx = cbase + k;
                            acc = fmaf(fmaf(dl, qv, -ml),
                                       (cidx < cols) ? x[cidx] : 0.0f, acc);
                        }
                    }
                }
            }
            // Join the 4 lane-groups: xor 8 then 16 sums corresponding lanes
            // across the groups, so each lane holds sum over groups of its
            // group-lane value.  Lanes 0..7 then run the 8-lane tree.
            acc += __shfl_xor_sync(0xffffffffu, acc, 8);
            acc += __shfl_xor_sync(0xffffffffu, acc, 16);
            const float p = acc + __shfl_xor_sync(0xffffffffu, acc, 4);
            const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
            if (I == 0) out[row] = rr;
        }

        // ---- Q2_K 8-way block batching in the 4xW shape (2026-09-30, lever-3
        // A/B TINYCODER_Q2K_8W=1, DEFAULT OFF): identical geometry to
        // kQGemvQ2KxW4 (all 32 lanes issue u32 loads, 8 lanes/row, 4 rows per
        // warp, no smem/barrier) but batches 8 blocks per outer iteration.
        // The 4xW version already interleaves 4 independent block chains per
        // row; 8-way doubles the in-flight weight loads per lane to hide DRAM
        // latency, at the cost of more registers.  Per-element float math is
        // IDENTICAL to kQGemvQ2KxW4 / kQGemv<kTypeQ2K> -- only the block
        // accumulation order changes (a0..a7 folded in order, then the warp
        // tree), so results stay within the Q2_K float-path ~1e-7 tolerance.
        __global__ void kQGemvQ2KxW4b8(const uint8_t *__restrict__ w,
                                       const float *__restrict__ x,
                                       float *__restrict__ out,
                                       uint32_t rows, uint32_t cols,
                                       uint32_t rowBytes,
                                       uint32_t blocksPerRow) {
            const uint32_t warpRowBase =
                    blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * 4u;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t row = base + (lane >> 3u);
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t qOfs = 4u * I;
            const uint32_t qb = qOfs;
            const uint32_t isub = (qOfs < 16u) ? 0u : 1u;
            float acc = 0.0f;
            uint32_t b = 0;
            for (; b + 8u <= blocksPerRow; b += 8u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 8; ++bb) {
                    const uint8_t *crp =
                            rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                    const float df = __half2float(
                            *reinterpret_cast<const __half *>(crp + 80));
                    const float dmf = __half2float(
                            *reinterpret_cast<const __half *>(crp + 82));
                    const uint8_t *sc = crp;
                    const uint8_t *q = crp + 16;
                    float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                            : (bb == 2u)       ? a2
                                            : (bb == 3u)       ? a3
                                            : (bb == 4u)       ? a4
                                            : (bb == 5u)       ? a5
                                            : (bb == 6u)       ? a6
                                                               : a7;
#pragma unroll
                    for (uint32_t n = 0; n < 2; ++n) {
                        uint32_t w4;
                        std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                        const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            const uint8_t scv = sc[n * 8u + 2u * j + isub];
                            const float dl =
                                    df * static_cast<float>(scv & 0xFu);
                            const float ml =
                                    dmf * static_cast<float>(scv >> 4u);
                            const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                            for (uint32_t k = 0; k < 4; ++k) {
                                const float qv = static_cast<float>(
                                        static_cast<int8_t>(
                                                (w4 >> (8u * k + 2u * j)) & 3u));
                                const uint32_t cidx = cbase + k;
                                slot = fmaf(fmaf(dl, qv, -ml),
                                            (cidx < cols) ? x[cidx] : 0.0f,
                                            slot);
                            }
                        }
                    }
                }
                acc += a0;
                acc += a1;
                acc += a2;
                acc += a3;
                acc += a4;
                acc += a5;
                acc += a6;
                acc += a7;
                rp += 8u * kQ2K_BYTES;
            }
            // Remainder blocks (blocksPerRow % 8) -- 4xW-compatible single/4
            // block bodies fall back to the 4xW remainder loop by just reusing
            // the 4-way body on the leftover (handles 0..7 leftover blocks).
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (uint32_t bb = 0; bb < 4; ++bb) {
                    const uint8_t *crp =
                            rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
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
                    for (uint32_t n = 0; n < 2; ++n) {
                        uint32_t w4;
                        std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                        const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                        for (uint32_t j = 0; j < 4; ++j) {
                            const uint8_t scv = sc[n * 8u + 2u * j + isub];
                            const float dl =
                                    df * static_cast<float>(scv & 0xFu);
                            const float ml =
                                    dmf * static_cast<float>(scv >> 4u);
                            const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                            for (uint32_t k = 0; k < 4; ++k) {
                                const float qv = static_cast<float>(
                                        static_cast<int8_t>(
                                                (w4 >> (8u * k + 2u * j)) & 3u));
                                const uint32_t cidx = cbase + k;
                                slot = fmaf(fmaf(dl, qv, -ml),
                                            (cidx < cols) ? x[cidx] : 0.0f,
                                            slot);
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
                for (uint32_t n = 0; n < 2; ++n) {
                    uint32_t w4;
                    std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                    const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        const uint8_t scv = sc[n * 8u + 2u * j + isub];
                        const float dl = df * static_cast<float>(scv & 0xFu);
                        const float ml = dmf * static_cast<float>(scv >> 4u);
                        const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                        for (uint32_t k = 0; k < 4; ++k) {
                            const float qv = static_cast<float>(
                                    static_cast<int8_t>(
                                            (w4 >> (8u * k + 2u * j)) & 3u));
                            const uint32_t cidx = cbase + k;
                            acc = fmaf(fmaf(dl, qv, -ml),
                                       (cidx < cols) ? x[cidx] : 0.0f, acc);
                        }
                    }
                }
                rp += kQ2K_BYTES;
            }
            // 8-lane shuffle tree per row (xor 4,1,2 stays inside the row's
            // 8-lane group -- same tree as kQGemvQ6KxQ8K_4xW).
            const float p = acc + __shfl_xor_sync(0xffffffffu, acc, 4);
            const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
            if (I == 0) out[row] = rr;
        }

        // ---- Q2_K x Q8_K int-dot GEMV, 4 rows per warp (2026-09-30, A/B
        // gate TINYCODER_Q2K_MQ4XW=1, DEFAULT OFF until measured).  This is
        // reference-mmq's decode route (kQuantizeQ8K the fp32 activation ONCE,
        // then an exact-integer 2-bit x Q8_K dot), brought to the Q6_K-proven
        // 4xW shape.  The float kQGemvQ2KxW4 reads the fp32 activation SCALAR
        // per element (32 predicated 4B loads + 32 fmaf/block/lane) which
        // keeps the Q2_K FFN at ~75-113 GB/s while the Q6_K LM head (same
        // geometry, Q8_K-int math) hits ~616 GB/s -- the FFN bandwidth gap
        // measured by [gpu fwd] decode matrix sums (gate+up 2.24 ms, down
        // 2.11 ms of the ~7.5 ms layer loop).
        //
        // STRUCTURE: identical to kQGemvQ6KxQ8K_4xW -- all 32 lanes issue u32
        // loads, 8 lanes per row (lane I owns the 4 contiguous cols 4I..4I+3
        // of every 32-col group), 4 rows per warp, 4-way block batching into
        // INDEPENDENT per-lane float accumulators r0..r3, and a SINGLE 8-lane
        // float tree at the END (no per-block shuffles / lane-0 serial folds
        // -- that was the failure mode of the earlier int-dot variants).
        //
        // MATH (int side bit-identical to dotProductQ2_K_PrePacked_Q8_Scalar /
        // the compact AVX2 makeSetup+blockDot): col c -> weight byte
        // q[n*32 + c%32], value (byte >> 2*((c%128)/32)) & 3; 16-group g =
        // n*8 + 2*j + sub (n = c/128, j = (c%128)/32, sub = (c%32)/16);
        // dl_g = dq*(scales[g]&0xF), ml_g = dmin*(scales[g]>>4);
        // block = d8*(sum_g dl_g*S_g - sum_g ml_g*bsums_g).
        //   * lane I: isub = I>>2 is CONSTANT, so dl_g == dq*sc[n*8+2j+isub]
        //     for every group the lane touches, and the per-lane partial
        //     dl_g*S_g-partial sums over the 8-lane tree to sum_g dl_g*S_g.
        //   * per (n,j): one u32 weight load q[n*32+4I..+3] (SAME bytes for
        //     all 4 j -- only the 2-bit shift 2*j changes) x one u32 Q8
        //     activation load y8[n*128+j*32+4I..+3] via __dp4a (signed bytes;
        //     weight bytes are 0..3 so signed==unsigned).
        //   * the ml term (bsums_g is a FULL 16-col sum, same for all 4 lanes
        //     of a sub-group) is folded on the sub-group representative lanes
        //     (I&3)==0 -- lanes 0 and 4 -- so the single end tree sums the
        //     dl*S terms from all 8 lanes and the ml terms exactly once.
        __device__ __forceinline__ void q2kXQ8KBlockDot(
                float &acc, const uint8_t *__restrict__ blkW,
                const uint8_t *__restrict__ blkA, uint32_t I) {
            const uint32_t isub = I >> 2u;     // 0 (lanes 0-3) or 1 (lanes 4-7)
            const bool mrep = ((I & 3u) == 0u);// sub-group rep folds ml
            const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA);
            const int16_t *bs =
                    reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
            const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
            const float dq = __half2float(
                    *reinterpret_cast<const __half *>(blkW + 80));
            const float dmin = __half2float(
                    *reinterpret_cast<const __half *>(blkW + 82));
#pragma unroll
            for (uint32_t n = 0; n < 2; ++n) {
                // 4 weight bytes for this lane (cols 4I..4I+3 of chunk n);
                // reused across all 4 groups j via the 2-bit shift 2*j.
                uint32_t wb;
                std::memcpy(&wb, blkW + 16u + n * 32u + 4u * I,
                            sizeof(uint32_t));
                uint64_t scU;
                std::memcpy(&scU, blkW + n * 8u, sizeof(uint64_t));
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    uint32_t yv;
                    std::memcpy(&yv, y8 + n * 128u + j * 32u + 4u * I,
                                sizeof(uint32_t));
                    const uint32_t pk = (wb >> (2u * j)) & 0x03030303u;
                    const int pdot = __dp4a(static_cast<int>(pk),
                                            static_cast<int>(yv), 0);
                    const uint8_t scv = static_cast<uint8_t>(
                            (scU >> (8u * (2u * j + isub))) & 0xFFu);
                    const float dl = dq * static_cast<float>(scv & 0xFu);
                    acc += d8 * (dl * static_cast<float>(pdot));
                    if (mrep) {
                        const float ml = dmin * static_cast<float>(scv >> 4u);
                        // Direct indexed int16 load: a packed u64 shift by
                        // >64 bits (groups 4..7) is UB and silently zeroed
                        // the ml terms for the high groups.
                        const int bg = static_cast<int>(bs[n * 8u + 2u * j +
                                                           isub]);
                        acc -= d8 * (ml * static_cast<float>(bg));
                    }
                }
            }
        }

        __global__ void kQGemvQ2KxQ8K_4xW(const uint8_t *__restrict__ w,
                                          const uint8_t *__restrict__ x,
                                          float *__restrict__ out,
                                          uint32_t rows,
                                          uint32_t blocksPerRow,
                                          uint32_t rowBytes) {
            const uint32_t warpRowBase =
                    blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t base = warpRowBase * 4u;
            if (base >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t row = base + (lane >> 3u);
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            float sumf = 0.0f;
            uint32_t b = 0;
            // 4-way block batching into INDEPENDENT per-lane accumulators
            // (same memory-level-parallelism pattern as kQGemvQ6KxQ8K_4xW).
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
                q2kXQ8KBlockDot(
                        r0, rp + static_cast<uint64_t>(b + 0u) * kQ2K_BYTES,
                        x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE, I);
                q2kXQ8KBlockDot(
                        r1, rp + static_cast<uint64_t>(b + 1u) * kQ2K_BYTES,
                        x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE, I);
                q2kXQ8KBlockDot(
                        r2, rp + static_cast<uint64_t>(b + 2u) * kQ2K_BYTES,
                        x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE, I);
                q2kXQ8KBlockDot(
                        r3, rp + static_cast<uint64_t>(b + 3u) * kQ2K_BYTES,
                        x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE, I);
                sumf = r0 + sumf;
                sumf = r1 + sumf;
                sumf = r2 + sumf;
                sumf = r3 + sumf;
            }
            // Remainder (blocksPerRow % 4) -- single-block body, same math.
            for (; b < blocksPerRow; ++b) {
                q2kXQ8KBlockDot(
                        sumf, rp + static_cast<uint64_t>(b) * kQ2K_BYTES,
                        x + static_cast<uint64_t>(b) * kQ8K_STRIDE, I);
            }
            // 8-lane shuffle tree per row (xor 4,1,2 stays inside the row's
            // 8-lane group -- same tree as kQGemvQ6KxQ8K_4xW).
            const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
            const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
            if (I == 0) out[row] = rr;
        }

        // MEASURED (eager, interleaved A/B, n16/g8/r5): MMVQ 153.3/144.0 vs
        // float 4xW default 167.4/168.4 -> REGRESSION. This REFUTES the
        // wave-granularity hypothesis that motivated it: the 4xW/MQ4XW
        // 280-block grid is a SINGLE full wave (68 SMs x 8 resident 256-thread
        // blocks = 544 >= 280), so row-granular blocks bought only block
        // launch/scheduling overhead (4480 tiny launches) with no bandwidth
        // gain. Combined with MQ4XW (int, 280 blocks: 153-156) this closes the
        // FFN-kernel lever: neither the int-dot route nor the grid shape beats
        // the float 4xW -- the decode kernel is NOT the 168 vs 261 gap. MATH is
        // bit-identical to kQGemvQ2KxQ8K_4xW (same q2kXQ8KBlockDot); only the
        // reduce ORDER across lane-groups differs (xor 16/8 instead of serial
        // batch fold) -- same float-tolerance class as the 4xW.
        __global__ void kQGemvQ2KxQ8K_Mmvq(const uint8_t *__restrict__ w,
                                           const uint8_t *__restrict__ x,
                                           float *__restrict__ out,
                                           uint32_t rows,
                                           uint32_t blocksPerRow,
                                           uint32_t rowBytes) {
            // block = (32, 2): 2 warps, one row per warp, grid = rows/2.
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t I = lane & 7u;
            const uint32_t g = lane >> 3u;// 4 lane-groups split the blocks
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            float sumf = 0.0f;
            // Row's Q2_K blocks round-robin across the 4 groups (FFN/attnO
            // blocksPerRow = 6: group g handles blocks g and g+4).  All 32
            // lanes stay busy; groups with no blocks idle into the tree.
            for (uint32_t b = g; b < blocksPerRow; b += 4u) {
                q2kXQ8KBlockDot(sumf, rp + static_cast<uint64_t>(b) * kQ2K_BYTES,
                                x + static_cast<uint64_t>(b) * kQ8K_STRIDE, I);
            }
            // 8-lane tree folds each group's partial onto all its lanes; xor
            // 16 then 8 folds the four group partials (g0..g3 live on lanes
            // 0..7, 8..15, 16..23, 24..31 after the first tree) down to lane 0.
            const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
            const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
            const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
            const float s = rr + __shfl_xor_sync(0xffffffffu, rr, 16);
            const float t = s + __shfl_xor_sync(0xffffffffu, s, 8);
            if (lane == 0) out[row] = t;
        }

        // ---- Q2_K int-dot fused gate+up+silu (2026-09-30, lever 1, A/B
        // TINYCODER_Q2K_INTGU=1).
        //
        // MATH (per 256-wide block; integer side bit-identical to the CPU
        // reference dotProductQ2_K_PrePacked_Q8_Scalar and the compact AVX2
        // kernel's makeSetup+blockDot):
        //   * col c: chunk n = c/128, group j = (c%128)/32, sub = (c%32)/16,
        //     weight byte = q[n*32 + c%32], 2-bit value = (byte >> 2*j) & 3;
        //     scale byte = scales[n*8 + j*2 + sub] (16 bytes, chunk-major);
        //     dl = dq*(sc&0xF), ml = dmin*(sc>>4).
        //   * S_g = 16-col int dot of weights vs Q8 y8; bsums[g] = Q8 block's
        //     group-g int16 sum.  block = dq*d8*sum_g((sc&0xF)*S_g)
        //                               - dmin*d8*sum_g((sc>>4)*bsums_g).
        //
        // LANE FACTORIZATION (exact): lane i (0..31) owns 8 CONTIGUOUS cols
        // c0 = i*8, so n = i>>4, j = (i>>2)&3, sub = (i&3)>>1:
        //   * weight bytes are CONTIGUOUS at q + (i>>4)*32 + (i&3)*8 (byte k
        //     covers col c0+k; c%32 == (i&3)*8 + k).  Verified vs dequant.
        //   * the 2-bit shift is CONSTANT over the lane: 2*j = 2*((i>>2)&3).
        //   * scale byte index == i>>1 (identity (i>>4)*8 + ((i>>2)&3)*2 +
        //     ((i&3)>>1) == i>>1 for all i<32, verified).
        // So the dl term is ONE 32-lane int tree of (scales[i>>1]&0xF)*S_i and
        // the ml term folds on lane 0 from the 16 Q8 bsums:
        // mlt = sum_g (scales[g]>>4)*bsums[g].  4-way block batching (8
        // independent per-lane int chains), lane-0 serial float fold in block
        // order (same convention as kQGemvQ8KFusedGU), silu(gate)*up out[row].
        //
        // Per-lane Q2_K 8-col int dot against shared Q8 y8: the lane's weight
        // bytes are contiguous (see above); sh is the constant 2-bit shift.
        __device__ __forceinline__ void q2kIntDot8(
                const uint8_t *__restrict__ qb, const uint8_t *__restrict__ qu,
                const int8_t *__restrict__ y8, uint32_t e0, uint32_t sh,
                int &gacc, int &uacc) {
            uint32_t wAg, wBg, wAu, wBu;
            std::memcpy(&wAg, qb, sizeof(uint32_t));
            std::memcpy(&wBg, qb + 4, sizeof(uint32_t));
            std::memcpy(&wAu, qu, sizeof(uint32_t));
            std::memcpy(&wBu, qu + 4, sizeof(uint32_t));
            const uint64_t yv = __ldg(
                    reinterpret_cast<const uint64_t *>(&y8[e0]));
#pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                const uint32_t bv = (k < 4u) ? ((wAg >> (8u * k)) & 0xFFu)
                                             : ((wBg >> (8u * (k & 3u))) & 0xFFu);
                const uint32_t bvU = (k < 4u) ? ((wAu >> (8u * k)) & 0xFFu)
                                              : ((wBu >> (8u * (k & 3u))) & 0xFFu);
                const int vy = static_cast<int>(static_cast<int8_t>(
                        (yv >> (8u * k)) & 0xFFu));
                gacc += static_cast<int>((bv >> sh) & 3u) * vy;
                uacc += static_cast<int>((bvU >> sh) & 3u) * vy;
            }
        }

        // Full 32-lane int xor-tree (1,2,4,8,16): exact warp-wide sum on every
        // lane (integer addition is order-free, so tree order is irrelevant).
        __device__ __forceinline__ int q2kWarpSumi(int v) {
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            return v;
        }

        // Lane-0-only ml term: sum_g (scales[g]>>4) * bsums[g].
        __device__ __forceinline__ int q2kMlTerm(const uint8_t *__restrict__ blk,
                                                 const int16_t *__restrict__ bs) {
            int mlt = 0;
#pragma unroll
            for (uint32_t g = 0; g < 16; ++g) {
                mlt += static_cast<int>(blk[g] >> 4u) *
                       static_cast<int>(bs[g]);
            }
            return mlt;
        }

        __global__ void __launch_bounds__(256) kQGemvQ2KIntFusedGU(
                const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
                const uint8_t *__restrict__ x, float *__restrict__ out,
                uint32_t rows, uint32_t blocksPerRow, uint32_t rowBytes) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rg = wg + static_cast<uint64_t>(row) * rowBytes;
            const uint8_t *ru = wu + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t lane = threadIdx.x;
            const uint32_t qOff = 16u + (lane >> 4u) * 32u + (lane & 3u) * 8u;
            const uint32_t sh = 2u * ((lane >> 2u) & 3u);
            const uint32_t p = lane >> 1u;// scale byte index (0..15)
            const uint32_t e0 = lane * 8u;// first col of the lane's 8
            float sumfG = 0.0f, sumfU = 0.0f;
            uint32_t b = 0;
            // 4-way block batching: 8 independent per-lane int chains feed 8
            // full-warp int trees; lane 0 folds gate then up in block order.
            for (; b + 4u <= blocksPerRow; b += 4u) {
                const uint8_t *gw0 = rg + static_cast<uint64_t>(b + 0u) * kQ2K_BYTES;
                const uint8_t *gw1 = rg + static_cast<uint64_t>(b + 1u) * kQ2K_BYTES;
                const uint8_t *gw2 = rg + static_cast<uint64_t>(b + 2u) * kQ2K_BYTES;
                const uint8_t *gw3 = rg + static_cast<uint64_t>(b + 3u) * kQ2K_BYTES;
                const uint8_t *uw0 = ru + static_cast<uint64_t>(b + 0u) * kQ2K_BYTES;
                const uint8_t *uw1 = ru + static_cast<uint64_t>(b + 1u) * kQ2K_BYTES;
                const uint8_t *uw2 = ru + static_cast<uint64_t>(b + 2u) * kQ2K_BYTES;
                const uint8_t *uw3 = ru + static_cast<uint64_t>(b + 3u) * kQ2K_BYTES;
                const uint8_t *ba0 = x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE;
                const uint8_t *ba1 = x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE;
                const uint8_t *ba2 = x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE;
                const uint8_t *ba3 = x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE;
                const int8_t *y0 = reinterpret_cast<const int8_t *>(ba0 + kQ8K_Q);
                const int8_t *y1 = reinterpret_cast<const int8_t *>(ba1 + kQ8K_Q);
                const int8_t *y2 = reinterpret_cast<const int8_t *>(ba2 + kQ8K_Q);
                const int8_t *y3 = reinterpret_cast<const int8_t *>(ba3 + kQ8K_Q);
                const float d8_0 = *reinterpret_cast<const float *>(ba0 + kQ8K_D);
                const float d8_1 = *reinterpret_cast<const float *>(ba1 + kQ8K_D);
                const float d8_2 = *reinterpret_cast<const float *>(ba2 + kQ8K_D);
                const float d8_3 = *reinterpret_cast<const float *>(ba3 + kQ8K_D);
                // Per-lane scales: low nibble = dl multiplier for that lane's
                // 16-group (index p == i>>1, exact identity, see comment).
                const int scG0 = static_cast<int>(gw0[p] & 0xFu);
                const int scU0 = static_cast<int>(uw0[p] & 0xFu);
                const int scG1 = static_cast<int>(gw1[p] & 0xFu);
                const int scU1 = static_cast<int>(uw1[p] & 0xFu);
                const int scG2 = static_cast<int>(gw2[p] & 0xFu);
                const int scU2 = static_cast<int>(uw2[p] & 0xFu);
                const int scG3 = static_cast<int>(gw3[p] & 0xFu);
                const int scU3 = static_cast<int>(uw3[p] & 0xFu);
                int gci0 = 0, gci1 = 0, gci2 = 0, gci3 = 0;
                int uci0 = 0, uci1 = 0, uci2 = 0, uci3 = 0;
                q2kIntDot8(gw0 + qOff, uw0 + qOff, y0, e0, sh, gci0, uci0);
                q2kIntDot8(gw1 + qOff, uw1 + qOff, y1, e0, sh, gci1, uci1);
                q2kIntDot8(gw2 + qOff, uw2 + qOff, y2, e0, sh, gci2, uci2);
                q2kIntDot8(gw3 + qOff, uw3 + qOff, y3, e0, sh, gci3, uci3);
                const int gt0 = q2kWarpSumi(gci0 * scG0);
                const int gt1 = q2kWarpSumi(gci1 * scG1);
                const int gt2 = q2kWarpSumi(gci2 * scG2);
                const int gt3 = q2kWarpSumi(gci3 * scG3);
                const int ut0 = q2kWarpSumi(uci0 * scU0);
                const int ut1 = q2kWarpSumi(uci1 * scU1);
                const int ut2 = q2kWarpSumi(uci2 * scU2);
                const int ut3 = q2kWarpSumi(uci3 * scU3);
                if (lane == 0) {
                    const float dq0 = __half2float(
                            *reinterpret_cast<const __half *>(gw0 + 80));
                    const float dmin0 = __half2float(
                            *reinterpret_cast<const __half *>(gw0 + 82));
                    const float dqU0 = __half2float(
                            *reinterpret_cast<const __half *>(uw0 + 80));
                    const float dminU0 = __half2float(
                            *reinterpret_cast<const __half *>(uw0 + 82));
                    sumfG += dq0 * d8_0 * static_cast<float>(gt0) -
                             dmin0 * d8_0 *
                                     static_cast<float>(q2kMlTerm(
                                             gw0, reinterpret_cast<const int16_t *>(ba0 + kQ8K_B)));
                    sumfU += dqU0 * d8_0 * static_cast<float>(ut0) -
                             dminU0 * d8_0 *
                                     static_cast<float>(q2kMlTerm(
                                             uw0, reinterpret_cast<const int16_t *>(ba0 + kQ8K_B)));
                    const float dq1 = __half2float(
                            *reinterpret_cast<const __half *>(gw1 + 80));
                    const float dmin1 = __half2float(
                            *reinterpret_cast<const __half *>(gw1 + 82));
                    const float dqU1 = __half2float(
                            *reinterpret_cast<const __half *>(uw1 + 80));
                    const float dminU1 = __half2float(
                            *reinterpret_cast<const __half *>(uw1 + 82));
                    sumfG += dq1 * d8_1 * static_cast<float>(gt1) -
                             dmin1 * d8_1 *
                                     static_cast<float>(q2kMlTerm(
                                             gw1, reinterpret_cast<const int16_t *>(ba1 + kQ8K_B)));
                    sumfU += dqU1 * d8_1 * static_cast<float>(ut1) -
                             dminU1 * d8_1 *
                                     static_cast<float>(q2kMlTerm(
                                             uw1, reinterpret_cast<const int16_t *>(ba1 + kQ8K_B)));
                    const float dq2 = __half2float(
                            *reinterpret_cast<const __half *>(gw2 + 80));
                    const float dmin2 = __half2float(
                            *reinterpret_cast<const __half *>(gw2 + 82));
                    const float dqU2 = __half2float(
                            *reinterpret_cast<const __half *>(uw2 + 80));
                    const float dminU2 = __half2float(
                            *reinterpret_cast<const __half *>(uw2 + 82));
                    sumfG += dq2 * d8_2 * static_cast<float>(gt2) -
                             dmin2 * d8_2 *
                                     static_cast<float>(q2kMlTerm(
                                             gw2, reinterpret_cast<const int16_t *>(ba2 + kQ8K_B)));
                    sumfU += dqU2 * d8_2 * static_cast<float>(ut2) -
                             dminU2 * d8_2 *
                                     static_cast<float>(q2kMlTerm(
                                             uw2, reinterpret_cast<const int16_t *>(ba2 + kQ8K_B)));
                    const float dq3 = __half2float(
                            *reinterpret_cast<const __half *>(gw3 + 80));
                    const float dmin3 = __half2float(
                            *reinterpret_cast<const __half *>(gw3 + 82));
                    const float dqU3 = __half2float(
                            *reinterpret_cast<const __half *>(uw3 + 80));
                    const float dminU3 = __half2float(
                            *reinterpret_cast<const __half *>(uw3 + 82));
                    sumfG += dq3 * d8_3 * static_cast<float>(gt3) -
                             dmin3 * d8_3 *
                                     static_cast<float>(q2kMlTerm(
                                             gw3, reinterpret_cast<const int16_t *>(ba3 + kQ8K_B)));
                    sumfU += dqU3 * d8_3 * static_cast<float>(ut3) -
                             dminU3 * d8_3 *
                                     static_cast<float>(q2kMlTerm(
                                             uw3, reinterpret_cast<const int16_t *>(ba3 + kQ8K_B)));
                }
            }
            // Remainder (blocksPerRow % 4) -- single-block body, same math.
            for (; b < blocksPerRow; ++b) {
                const uint8_t *blkWG = rg + static_cast<uint64_t>(b) * kQ2K_BYTES;
                const uint8_t *blkWU = ru + static_cast<uint64_t>(b) * kQ2K_BYTES;
                const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
                const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA + kQ8K_Q);
                const int16_t *bs = reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
                const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
                const int scG = static_cast<int>(blkWG[p] & 0xFu);
                const int scU = static_cast<int>(blkWU[p] & 0xFu);
                int gci = 0, uci = 0;
                q2kIntDot8(blkWG + qOff, blkWU + qOff, y8, e0, sh, gci, uci);
                const int gt = q2kWarpSumi(gci * scG);
                const int ut = q2kWarpSumi(uci * scU);
                if (lane == 0) {
                    const float dq = __half2float(
                            *reinterpret_cast<const __half *>(blkWG + 80));
                    const float dmin = __half2float(
                            *reinterpret_cast<const __half *>(blkWG + 82));
                    const float dqU = __half2float(
                            *reinterpret_cast<const __half *>(blkWU + 80));
                    const float dminU = __half2float(
                            *reinterpret_cast<const __half *>(blkWU + 82));
                    sumfG += dq * d8 * static_cast<float>(gt) -
                             dmin * d8 * static_cast<float>(q2kMlTerm(blkWG, bs));
                    sumfU += dqU * d8 * static_cast<float>(ut) -
                             dminU * d8 * static_cast<float>(q2kMlTerm(blkWU, bs));
                }
            }
            if (lane == 0) {
                const float g = sumfG;
                const float u = sumfU;
                out[row] = (g / (1.0f + __expf(-g))) * u;
            }
        }


        // ---- Batched gate+up GEMV (Q4_K/Q6_K x Q8_K): grid.y = rank/slot j,
        // gates rows [0, expertFF) and up rows [expertFF, 2*expertFF) via the
        // slot-major wq table ([j*3+g/u] = gate/up base).  Identical math to
        // kQGemvKxQ8K, bit-exact with the CPU grouped path.
        template<int TYPE>
        __global__ void kQGemvKxQ8K_BatchGU(const uint8_t *const *__restrict__ wq,
                                            const uint8_t *__restrict__ x,
                                            float *__restrict__ gateOut,
                                            float *__restrict__ upOut,
                                            uint32_t expertFF,
                                            uint32_t blocksPerRow,
                                            uint32_t rowBytes,
                                            uint32_t combinedRows) {
            const uint32_t r = blockIdx.y;
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= combinedRows) return;
            const uint32_t isUp = (row >= expertFF) ? 1u : 0u;
            const uint32_t wrow = row - isUp * expertFF;
            const uint8_t *rp = wq[r * 3u + isUp] +
                                static_cast<uint64_t>(wrow) * rowBytes;
            const uint32_t lane = threadIdx.x;
            float sumf = 0.0f;
            constexpr uint32_t kBlkBytes =
                    (TYPE == kTypeQ4K)     ? kQ4K_BYTES
                    : (TYPE == kTypeQ5K)   ? kQ5K_BYTES
                    : (TYPE == kTypeIQ4XS) ? kIQ4XS_BYTES
                                           : kQ6K_BYTES;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * kBlkBytes;
                const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
                kxQ8KBlockDot<TYPE>(sumf, blkW, blkA, lane);
            }
            if (TYPE == kTypeQ4K || TYPE == kTypeQ5K ||
                TYPE == kTypeIQ4XS) {
                if (lane == 0) {
                    float *dest = isUp ? upOut : gateOut;
                    dest[r * expertFF + wrow] = sumf;
                }
            } else {
                const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
                const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
                const float rr = q + __shfl_xor_sync(0xffffffffu, q, 2);
                if (lane == 0) {
                    float *dest = isUp ? upOut : gateOut;
                    dest[r * expertFF + wrow] = rr;
                }
            }
        }

        // ---- Batched down GEMV with rank remap (Q4_K/Q6_K x Q8_K): resident
        // slots 0..residentCount-1 are compacted; slot j's down rows land at
        // out[rankOf[j]*outStride + row] in the full rank-major combined table
        // (outStride == seqLen*H).  Same math as the GU kernel.
        template<int TYPE>
        __global__ void kQGemvKxQ8K_BatchDownR(const uint8_t *const *__restrict__ wq,
                                               const uint8_t *__restrict__ x,
                                               float *__restrict__ out,
                                               const int32_t *__restrict__ rankOf,
                                               uint32_t hidden,
                                               uint32_t blocksPerRow,
                                               uint32_t rowBytes,
                                               uint32_t xStrideBytes,
                                               uint32_t outStride) {
            const uint32_t j = blockIdx.y;
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= hidden) return;
            const int32_t r0 = (rankOf != nullptr)
                                       ? rankOf[j]
                                       : static_cast<int32_t>(j);
            if (r0 < 0) return;
            const uint8_t *rp = wq[j * 3u + 2u] +
                                static_cast<uint64_t>(row) * rowBytes;
            const uint8_t *xr = x + static_cast<uint64_t>(j) * xStrideBytes;
            const uint32_t lane = threadIdx.x;
            float sumf = 0.0f;
            constexpr uint32_t kBlkBytes =
                    (TYPE == kTypeQ4K)     ? kQ4K_BYTES
                    : (TYPE == kTypeQ5K)   ? kQ5K_BYTES
                    : (TYPE == kTypeIQ4XS) ? kIQ4XS_BYTES
                                           : kQ6K_BYTES;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * kBlkBytes;
                const uint8_t *blkA = xr + static_cast<uint64_t>(b) * kQ8K_STRIDE;
                kxQ8KBlockDot<TYPE>(sumf, blkW, blkA, lane);
            }
            if (TYPE == kTypeQ4K || TYPE == kTypeQ5K ||
                TYPE == kTypeIQ4XS) {
                if (lane == 0) {
                    out[static_cast<uint64_t>(r0) * outStride + row] = sumf;
                }
            } else {
                const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
                const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
                const float rr = q + __shfl_xor_sync(0xffffffffu, q, 2);
                if (lane == 0) {
                    out[static_cast<uint64_t>(r0) * outStride + row] = rr;
                }
            }
        }

        // ---- Batched 256-wide Q8_K quantize (per-rank slabs): rank r's
        // [cols] fp32 activation -> q8k + r*strideBytes, block-by-block with
        // the EXACT kQuantizeQ8K math (bit-exact with
        // GGMLDequantize::quantizeQ8K that the CPU kernels use).  grid =
        // (blocks, ranks), block = 256 threads.
        __global__ void kQuantizeQ8K_Batch(const float *__restrict__ x,
                                           uint8_t *__restrict__ q8k,
                                           uint32_t cols, uint32_t ranks,
                                           uint32_t strideBytes) {
            const uint32_t r = blockIdx.y;
            const uint32_t b = blockIdx.x;
            const uint32_t tid = threadIdx.x;
            const uint32_t e0 = b * 256u;
            const uint32_t n = (cols - e0) < 256u ? (cols - e0) : 256u;
            if (n == 0) return;
            uint8_t *blk = q8k + static_cast<uint64_t>(r) * strideBytes +
                           static_cast<uint64_t>(b) * kQ8K_STRIDE;
            const float *xs = x + static_cast<uint64_t>(r) * cols;
            float maxv = 0.0f, amax = 0.0f;
            for (uint32_t j = tid; j < n; j += blockDim.x) {
                const float v = xs[e0 + j];
                const float av = fabsf(v);
                if (av > amax) {
                    amax = av;
                    maxv = v;
                }
            }
            __shared__ float sMax[256];
            sMax[tid] = (amax > 0.0f) ? maxv : 0.0f;
            __syncthreads();
            for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    const float a0 = sMax[tid], a1 = sMax[tid + s];
                    if (fabsf(a1) > fabsf(a0)) sMax[tid] = a1;
                }
                __syncthreads();
            }
            const float max = sMax[0];
            // See kQuantizeQ8K: __fdiv_rn keeps the two IEEE divisions the CPU
            // quantizeQ8K performs (1/(-127/max) contracts to -max/127 with a
            // plain '/' and drifts the stored d by 1 ulp).
            const float iscale = (max != 0.0f) ? __fdiv_rn(-127.0f, max) : 0.0f;
            const float d = (max != 0.0f) ? __fdiv_rn(1.0f, iscale) : 0.0f;
            for (uint32_t j = tid; j < n; j += blockDim.x) {
                if (max == 0.0f) {
                    blk[kQ8K_Q + j] = 0;
                    continue;
                }
                int v = static_cast<int>(rintf(iscale * xs[e0 + j]));
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                blk[kQ8K_Q + j] = static_cast<uint8_t>(static_cast<int8_t>(v));
            }
            if (tid == 0) {
                *reinterpret_cast<float *>(blk + kQ8K_D) = d;
            }
            __syncthreads();
            for (uint32_t j = tid; j < 16; j += blockDim.x) {
                int sum = 0;
                for (uint32_t ii = 0; ii < 16; ++ii) {
                    sum += static_cast<int>(
                            static_cast<int8_t>(blk[kQ8K_Q + j * 16u + ii]));
                }
                *reinterpret_cast<int16_t *>(blk + kQ8K_B + j * 2) =
                        static_cast<int16_t>(sum);
            }
        }
        // down(r,s) over ranks ASCENDING, matching the CPU reference
        // (computeQwen35MoE step 5: expertOut[i] += weight * down[i], with
        // expertOut zero-initialized).  Plain mul+add (no fmaf), left-to-right
        // association, ONE thread per element looping ranks serially so the
        // accumulation ORDER is exactly the CPU's (0 + w0*d0, then + w1*d1, ...)
        // and there is NO cross-rank race on o[i].  grid = 1 block (decode
        // path is seqLen==1); 256 threads stride over hidden.
        __global__ void kQ35MoeAccumScaledBatch(const float *__restrict__ wgt,
                                                uint32_t wgtStride,
                                                const float *__restrict__ down,
                                                float *__restrict__ out,
                                                uint32_t seqLen, uint32_t hidden,
                                                uint32_t ranks) {
            const uint32_t s = 0;// batched decode path is seqLen==1 only
            float *o = out + static_cast<size_t>(s) * hidden;
            for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
                float acc = 0.0f;
#pragma unroll 4
                for (uint32_t r = 0; r < ranks; ++r) {
                    const float g = wgt[static_cast<size_t>(s) * wgtStride + r];
                    // NO FMA CONTRACTION: the CPU reference (ModelMoE.cpp
                    // computeQwen35MoE step 5: expertOut[i] += weight * down[i])
                    // is a strict mul-then-add.  nvcc's default -fmad=true
                    // contracts `g*down + acc` into a single fmaf, which rounds
                    // ONCE instead of twice — a 1-ULP difference per element
                    // that compounds through the recurrent GDN state and flips
                    // top-1 (root-cause of SequentialDecodeArgmaxAgrees failing
                    // in BOTH cache paths while the CPU-expert fallback
                    // passed; verified 2026-09-15: L0 moeOut identical in both,
                    // first 1-ULP norm divergence at L1 fwd3).  __fmul_rn/
                    // __fadd_rn are never contracted by the compiler.
                    acc = __fadd_rn(
                            __fmul_rn(g, down[static_cast<size_t>(r) * hidden + i]),
                            acc);
                }
                o[i] = acc;
            }
        }

        // Debug twin of kQGemvQ8_0xQ8K: identical integer group-sum logic and
        // identical float tree, but lane 0 additionally records per-block
        // cumulative sums into trace[0..blocksPerRow), the block-0 t[g]
        // intermediates into trace[blocksPerRow + g], and block-0's blk into
        // trace[blocksPerRow + 8].  Used by the parity unit test to localize
        // the FIRST block where the kernel's internal float tree diverges from
        // the host recompute (which is bit-exact with the CPU AVX2 reference).
        // Diagnostic only -- never launched in the inference path.
        __global__ void kQGemvQ8_0xQ8KTrace(const uint8_t *__restrict__ w,
                                            const uint8_t *__restrict__ x,
                                            float *__restrict__ out,
                                            float *__restrict__ trace,
                                            uint32_t rows,
                                            uint32_t blocksPerRow,
                                            uint32_t rowBytes) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t lane = threadIdx.x;
            float sumf = 0.0f;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * 34u;
                const uint8_t *blkA =
                        x + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
                const float d =
                        __half2float(*reinterpret_cast<const __half *>(blkW));
                const float d8 = *reinterpret_cast<const float *>(blkA);
                if (d8 == 0.0f) continue;
                const int wq = static_cast<int>(
                        static_cast<int8_t>(blkW[2u + lane]));
                const int xq = static_cast<int>(
                        static_cast<int8_t>(blkA[4u + lane]));
                int c = xq * wq;
                c += __shfl_xor_sync(0xffffffffu, c, 1);
                c += __shfl_xor_sync(0xffffffffu, c, 2);
                int s[8];
#pragma unroll
                for (uint32_t g = 0; g < 8; ++g) {
                    s[g] = __shfl_sync(0xffffffffu, c, 4u * g);
                }
                if (lane == 0) {
                    const float scale = d * d8;
                    float t[8];
#pragma unroll
                    for (uint32_t g = 0; g < 8; ++g) {
                        t[g] = scale * static_cast<float>(s[g]);
                    }
                    const float u0 = t[0] + t[4];
                    const float u1 = t[1] + t[5];
                    const float u2 = t[2] + t[6];
                    const float u3 = t[3] + t[7];
                    const float v0 = u0 + u1;
                    const float v1 = u2 + u3;
                    const float blk = v0 + v1;
                    sumf += blk;
                    trace[b] = sumf;
                    if (b == 0) {
                        for (uint32_t g = 0; g < 8; ++g) {
                            trace[blocksPerRow + g] = t[g];
                        }
                        trace[blocksPerRow + 8u] = blk;
                    }
                    if (b == 1) {
                        for (uint32_t g = 0; g < 8; ++g) {
                            trace[2u * blocksPerRow + g] =
                                    static_cast<float>(s[g]);
                        }
                        trace[2u * blocksPerRow + 8u] = blk;
                        trace[2u * blocksPerRow + 9u] = d;
                        trace[2u * blocksPerRow + 10u] = d8;
                    }
                }
            }
            if (lane == 0) out[row] = sumf;
        }

        // Per-lane integer contribution for ONE (ib32, l) group's 8 columns.
        //
        // MEMORY OPTIMIZATION (2026-09-27): the old body read the lane's 8
        // activation bytes with EIGHT separate y8[...] byte loads at stride-8
        // addresses. Across the warp that is a perfectly contiguous 256-byte
        // block, but issued as 8 independent 1-byte transactions per lane, each
        // dependent on the previous byte's load -> a serial ~8-deep L2 latency
        // chain per (block, lane). Measured: the IQ2_S/IQ3_XXS/IQ3_S Q8K GEMV
        // (kQGemvQ8K) ran at ~2000 cycles per block-iteration -- ~700x off the
        // 448 GB/s memory floor (7B IQ2_S decode: gate+up 146.6ms, down 70.8ms
        // for ~131 MB of weight bytes/token). Each lane's 8 columns are
        // CONTIGUOUS (base = ib32*32 + l*8, 8-aligned), so we now load them as
        // ONE 8-byte __ldg (coalesced: the 32 lanes cover the whole 256-byte
        // block in one transaction set) and byte-extract.
        // Forward declaration (the shared body is defined just below; the
        // wrapper and the multi-row kernel both call it).
        template<bool DP4A>
        __device__ __forceinline__ int q8kLaneSumiV(
                int type, const uint8_t *__restrict__ w, uint64_t y8v,
                uint32_t ib32, uint32_t l,
                const uint8_t *__restrict__ gsm = nullptr);

        // Thin wrapper: load the lane's activation word then run the shared
        // body.  Bit-identical to the historical q8kLaneSumi.
        template<bool DP4A = false>
        __device__ __forceinline__ int q8kLaneSumi(
                int type, const uint8_t *__restrict__ w,
                const int8_t *__restrict__ y8, uint32_t ib32, uint32_t l,
                const uint8_t *__restrict__ gsm = nullptr) {
            // The lane's 8 columns are contiguous and 8-byte aligned (cudaMalloc
            // aligns q8k to >= 256; kQ8K_STRIDE=304 is a multiple of 8; ib32*32
            // and l*8 are multiples of 8).  __ldg routes through the read-only
            // cache so the streaming q8k block does not evict weight lines.
            const uint64_t y8v = __ldg(reinterpret_cast<const uint64_t *>(
                    &y8[ib32 * 32u + l * 8u]));
            return q8kLaneSumiV<DP4A>(type, w, y8v, ib32, l, gsm);
        }

        // Body shared by q8kLaneSumi (which loads y8v itself) and the
        // multi-row kernel (which loads one activation word per warp-lane and
        // reuses it across all rows).  y8v is the lane's 8 Q8_K activation
        // bytes at [ib32*32 + l*8].
        template<bool DP4A>
        __device__ __forceinline__ int q8kLaneSumiV(
                int type, const uint8_t *__restrict__ w, uint64_t y8v,
                uint32_t ib32, uint32_t l,
                const uint8_t *__restrict__ gsm) {
            if constexpr (DP4A) {
                // IQ2_S / IQ3_XXS / IQ3_S lane dot via __dp4a + the branchless
                // XOR-subtract sign application from llama's
                // vec_dot_iq3_xxs_q8_1 (__vcmpne4/__vsub4).  Bit-identical to
                // the scalar body: dp4a accumulates exact signed int8 products,
                // and the XOR-sub yields the exact signed grid byte (every grid
                // magnitude is <= 62 <= 127, so -g is int8-representable).
                // Replaces 8 scalar IMAD + 8 sign selects per (ib32, l) with
                // 2 dp4a + 4 mask/vector ops -- the ALU lever for the IQ decode
                // kernels (2026-10-11).  A/B: TINYCODER_IQ_DP4A.
                const int u0 = static_cast<int>(y8v & 0xFFFFFFFFull);
                const int u1 = static_cast<int>(y8v >> 32);
                if (type == kTypeIQ2S) {
                    const uint8_t *qsv = w + 2;
                    const uint8_t *qh = w + 66;
                    const uint16_t gridIdx = static_cast<uint16_t>(
                            qsv[ib32 * 4u + l] |
                            ((qh[ib32] << (8 - 2u * l)) & 0x300u));
                    const uint64_t gv =
                            gsm ? reinterpret_cast<const uint64_t *>(gsm)[gridIdx]
                                : c_iq2s_grid[gridIdx];
                    const int signs =
                            static_cast<int>(qsv[32u + ib32 * 4u + l]);
                    const int s0 = __vcmpne4(
                            static_cast<uint32_t>(signs) * 0x01010101u &
                                    0x08040201u,
                            0);
                    const int s1 = __vcmpne4(
                            static_cast<uint32_t>(signs) * 0x01010101u &
                                    0x80402010u,
                            0);
                    const int w0 = __vsub4(
                            static_cast<int>(gv & 0xFFFFFFFFull) ^ s0, s0);
                    const int w1 = __vsub4(
                            static_cast<int>(gv >> 32) ^ s1, s1);
                    return __dp4a(w1, u1, __dp4a(w0, u0, 0));
                }
                if (type == kTypeIQ2XS) {
                    // IQ2_XS block (74 B): d(2) + qs[32 x uint16](64) +
                    // scales[8].  Per (ib32, l): qval = qs16[ib32*4+l]; grid
                    // index = low 9 bits (512-entry uint64 grid = 8 columns);
                    // signs = ksigns[qval>>9].  ONE grid entry per group (vs
                    // IQ3_XXS's two) -> the sign masks span bytes 0..7 through
                    // the low/high halves, exactly as the scalar body.
                    const uint16_t *qs16 =
                            reinterpret_cast<const uint16_t *>(w + 2);
                    const uint16_t qval = qs16[ib32 * 4u + l];
                    const uint64_t gv =
                            gsm ? reinterpret_cast<const uint64_t *>(
                                          gsm)[qval & 0x1FFu]
                                : c_iq2xs_grid[qval & 0x1FFu];
                    const int signs =
                            static_cast<int>(c_ksigns_iq2xs[qval >> 9]);
                    const int s0 = __vcmpne4(
                            static_cast<uint32_t>(signs) * 0x01010101u &
                                    0x08040201u,
                            0);
                    const int s1 = __vcmpne4(
                            static_cast<uint32_t>(signs) * 0x01010101u &
                                    0x80402010u,
                            0);
                    const int w0 = __vsub4(
                            static_cast<int>(gv & 0xFFFFFFFFull) ^ s0, s0);
                    const int w1 = __vsub4(
                            static_cast<int>(gv >> 32) ^ s1, s1);
                    return __dp4a(w1, u1, __dp4a(w0, u0, 0));
                }
                if (type == kTypeIQ3_XXS) {
                    const uint8_t *q3 = w + 2;
                    uint32_t aux32;
                    std::memcpy(&aux32, q3 + 64u + 4u * ib32,
                                sizeof(uint32_t));
                    const uint32_t g1 =
                            gsm ? reinterpret_cast<const uint32_t *>(
                                          gsm)[q3[8u * ib32 + 2u * l]]
                                : c_iq3xxs_grid[q3[8u * ib32 + 2u * l]];
                    const uint32_t g2 =
                            gsm ? reinterpret_cast<const uint32_t *>(
                                          gsm)[q3[8u * ib32 + 2u * l + 1u]]
                                : c_iq3xxs_grid[q3[8u * ib32 + 2u * l + 1u]];
                    const int signs = static_cast<int>(
                            c_ksigns_iq2xs[(aux32 >> (7u * l)) & 127u]);
                    const int s0 = __vcmpne4(
                            static_cast<uint32_t>(signs) * 0x01010101u &
                                    0x08040201u,
                            0);
                    const int s1 = __vcmpne4(
                            static_cast<uint32_t>(signs) * 0x01010101u &
                                    0x80402010u,
                            0);
                    const int w0 = __vsub4(static_cast<int>(g1) ^ s0, s0);
                    const int w1 = __vsub4(static_cast<int>(g2) ^ s1, s1);
                    return __dp4a(w1, u1, __dp4a(w0, u0, 0));
                }
                // IQ3_S
                const uint8_t *qs = w + 2;
                const uint8_t *qh = w + 66;
                const uint8_t *signsP = w + 74;
                const uint32_t qhIdx = (ib32 >> 1u) * 2u + (ib32 & 1u);
                const uint32_t qsOff = (ib32 >> 1u) * 16u + (ib32 & 1u) * 8u;
                const uint32_t signOff =
                        (ib32 >> 1u) * 8u + (ib32 & 1u) * 4u;
                const uint16_t gridIdx1 = static_cast<uint16_t>(
                        qs[qsOff + 2u * l] |
                        ((qh[qhIdx] << (8 - 2u * l)) & 256u));
                const uint16_t gridIdx2 = static_cast<uint16_t>(
                        qs[qsOff + 2u * l + 1u] |
                        ((qh[qhIdx] << (7 - 2u * l)) & 256u));
                const uint32_t g1 =
                        gsm ? reinterpret_cast<const uint32_t *>(gsm)[gridIdx1]
                            : c_iq3s_grid[gridIdx1];
                const uint32_t g2 =
                        gsm ? reinterpret_cast<const uint32_t *>(gsm)[gridIdx2]
                            : c_iq3s_grid[gridIdx2];
                const int signs = static_cast<int>(signsP[signOff + l]);
                const int s0 = __vcmpne4(static_cast<uint32_t>(signs) *
                                                         0x01010101u &
                                                 0x08040201u,
                                         0);
                const int s1 = __vcmpne4(static_cast<uint32_t>(signs) *
                                                         0x01010101u &
                                                 0x80402010u,
                                         0);
                const int w0 = __vsub4(static_cast<int>(g1) ^ s0, s0);
                const int w1 = __vsub4(static_cast<int>(g2) ^ s1, s1);
                return __dp4a(w1, u1, __dp4a(w0, u0, 0));
            } else {
                auto yb = [&](uint32_t j) -> int8_t {
                    return static_cast<int8_t>((y8v >> (8u * j)) & 0xFFu);
                };
                (void) y8v;
                if (type == kTypeIQ2S) {
                    const uint8_t *qsv = w + 2;
                    const uint8_t *qh = w + 66;
                    const uint16_t gridIdx = static_cast<uint16_t>(
                            qsv[ib32 * 4u + l] | ((qh[ib32] << (8 - 2u * l)) & 0x300u));
                    // Native-word load + register byte-extraction.  The old body
                    // dereferenced grid[j] for j=0..7 -- EIGHT separate ld.const.u8
                    // with a per-lane (divergent) address.  A divergent constant
                    // load costs one constant-cache port access PER UNIQUE ADDRESS
                    // (up to 32 serialized cycles for a full warp).  That was the
                    // real ~2000 cycles/block-iteration serialization point (the
                    // 4-way block batching only hides latency, not this throughput
                    // limit -- hence batching measured zero change).  One uint64
                    // load + 8 register shifts is byte-identical (little-endian,
                    // low byte = grid[0]) and costs 1 (or a few) constant accesses.
                    const uint64_t gv =
                            gsm ? reinterpret_cast<const uint64_t *>(gsm)[gridIdx]
                                : c_iq2s_grid[gridIdx];
                    const uint8_t signs = qsv[32u + ib32 * 4u + l];
                    int sumi = 0;
                    for (uint32_t j = 0; j < 8; ++j) {
                        sumi += static_cast<int>(yb(j)) *
                                static_cast<int>((gv >> (8u * j)) & 0xFFu) *
                                ((signs & c_kmask_iq2xs[j]) ? -1 : 1);
                    }
                    return sumi;
                }
                if (type == kTypeIQ2XS) {
                    const uint16_t *qs16 =
                            reinterpret_cast<const uint16_t *>(w + 2);
                    const uint16_t qval = qs16[ib32 * 4u + l];
                    const uint64_t gv =
                            gsm ? reinterpret_cast<const uint64_t *>(
                                          gsm)[qval & 0x1FFu]
                                : c_iq2xs_grid[qval & 0x1FFu];
                    const uint8_t signs = c_ksigns_iq2xs[qval >> 9];
                    int sumi = 0;
                    for (uint32_t j = 0; j < 8; ++j) {
                        sumi += static_cast<int>((gv >> (8u * j)) & 0xFFu) *
                                static_cast<int>(yb(j)) *
                                ((signs & c_kmask_iq2xs[j]) ? -1 : 1);
                    }
                    return sumi;
                }
                if (type == kTypeIQ3_XXS) {
                    const uint8_t *q3 = w + 2;
                    uint32_t aux32;
                    std::memcpy(&aux32, q3 + 64u + 4u * ib32, sizeof(uint32_t));
                    // Same native-word trick as IQ2_S: the old body did 8 divergent
                    // byte loads on the two uint32 grid entries; now 2 divergent
                    // uint32 loads + register byte-extraction.
                    const uint32_t g1 = c_iq3xxs_grid[q3[8u * ib32 + 2u * l]];
                    const uint32_t g2 = c_iq3xxs_grid[q3[8u * ib32 + 2u * l + 1u]];
                    const uint8_t signs = c_ksigns_iq2xs[(aux32 >> (7u * l)) & 127u];
                    int sumi = 0;
                    for (uint32_t j = 0; j < 4; ++j) {
                        sumi += static_cast<int>((g1 >> (8u * j)) & 0xFFu) *
                                static_cast<int>(yb(j)) *
                                ((signs & c_kmask_iq2xs[j + 0]) ? -1 : 1);
                        sumi += static_cast<int>((g2 >> (8u * j)) & 0xFFu) *
                                static_cast<int>(yb(j + 4)) *
                                ((signs & c_kmask_iq2xs[j + 4]) ? -1 : 1);
                    }
                    return sumi;
                }
                // IQ3_S
                const uint8_t *qs = w + 2;
                const uint8_t *qh = w + 66;
                const uint8_t *signs = w + 74;
                const uint32_t qhIdx = (ib32 >> 1u) * 2u + (ib32 & 1u);
                const uint32_t qsOff = (ib32 >> 1u) * 16u + (ib32 & 1u) * 8u;
                const uint32_t signOff = (ib32 >> 1u) * 8u + (ib32 & 1u) * 4u;
                const uint16_t gridIdx1 = static_cast<uint16_t>(
                        qs[qsOff + 2u * l] | ((qh[qhIdx] << (8 - 2u * l)) & 256u));
                const uint16_t gridIdx2 = static_cast<uint16_t>(
                        qs[qsOff + 2u * l + 1u] | ((qh[qhIdx] << (7 - 2u * l)) & 256u));
                const uint32_t g1 =
                        gsm ? reinterpret_cast<const uint32_t *>(gsm)[gridIdx1]
                            : c_iq3s_grid[gridIdx1];
                const uint32_t g2 =
                        gsm ? reinterpret_cast<const uint32_t *>(gsm)[gridIdx2]
                            : c_iq3s_grid[gridIdx2];
                const uint8_t sbyte = signs[signOff + l];
                int sumi = 0;
                for (uint32_t j = 0; j < 4; ++j) {
                    sumi += static_cast<int>((g1 >> (8u * j)) & 0xFFu) *
                            static_cast<int>(yb(j)) *
                            ((sbyte & c_kmask_iq2xs[j + 0]) ? -1 : 1);
                    // grid2 is a uint32 whose bytes 0..3 are grid2[j] for j=0..3
                    // (the CPU reference reads them via
                    // reinterpret_cast<const uint8_t*>(&IQ3S_GRID[...])[j], i.e.
                    // byte j of the little-endian word).  Shifting by (j+4) would
                    // shift the 32-bit word out entirely (>>= 32..56) and silently
                    // DROP all four grid2 contributions -- the attnO corruption.
                    sumi += static_cast<int>((g2 >> (8u * j)) & 0xFFu) *
                            static_cast<int>(yb(j + 4)) *
                            ((sbyte & c_kmask_iq2xs[j + 4]) ? -1 : 1);
                }
                return sumi;
            }
        }

        // Multiply a lane's integer sumi by its per-(ib32, l) ls multiplier.
        template<int TYPE>
        __device__ __forceinline__ int q8kScaleMuli(const uint8_t *__restrict__ w,
                                                    int ci, uint32_t ib32, uint32_t l) {
            if (TYPE == kTypeIQ2S) {
                const uint8_t *scales = w + 74;
                const int ls = (l < 2) ? (1 + 2 * (scales[ib32] & 0xf))
                                       : (1 + 2 * (scales[ib32] >> 4));
                return ci * ls;
            }
            if (TYPE == kTypeIQ2XS) {
                // IQ2_XS: scales[8] at offset 66; low nibble for l=0,1 and high
                // nibble for l=2,3; ls = 2*nib+1 (the CPU db = d*(0.5+nib)*0.25
                // = d*ls/8, hence the 0.125 final scale, same as IQ2_S).
                const uint8_t *scales = w + 66;
                const int nib = (l < 2) ? (scales[ib32] & 0xf)
                                        : (scales[ib32] >> 4);
                return ci * (2 * nib + 1);
            }
            if (TYPE == kTypeIQ3_XXS) {
                uint32_t aux32;
                std::memcpy(&aux32, w + 2 + 64u + 4u * ib32, sizeof(uint32_t));
                return ci * static_cast<int>(2 * (aux32 >> 28) + 1);
            }
            // kTypeIQ3S
            const uint8_t *scales = w + 106;
            const int ls = static_cast<int>(
                    2 * ((ib32 & 1u) ? (scales[ib32 >> 1u] >> 4)
                                     : (scales[ib32 >> 1u] & 0xf)) +
                    1);
            return ci * ls;
        }

        // Warp-per-row Q8K GEMV over the pre-quantized activation (q8k).
        //
        // Per (ib32, l) lane: integer sumi (matches vec_dot_iq*_q8_K generic).
        // The per-ib32 bsum is IQ2_S: ls1*sumi(l=0..1) + ls2*sumi(l=2..3)  (per
        // 32-group ib32) IQ3_XXS: ls*sumi(l=0..3) IQ3_S:  ls*sumi(l=0..3) per
        // 32-wide half pairing (ib32 pairs) Since everything is integer,
        // multiplying each lane's sumi by its own ls FIRST and then doing a
        // plain 32-lane tree reduction reproduces the exact same integer bsum
        // (addition distributes over the scalar ls, including the split ls1/ls2
        // case).
        template<int TYPE, bool DP4A = false>
        __global__ void __launch_bounds__(256) kQGemvQ8K(
                const uint8_t *__restrict__ w, const uint8_t *__restrict__ x,
                float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow,
                uint32_t rowBytes, bool grsm = false) {
            extern __shared__ uint8_t sGrid[];
            const uint8_t *gsm = nullptr;
            if (grsm) {
                const uint32_t nw = (TYPE == kTypeIQ2S)    ? 2048u
                                    : (TYPE == kTypeIQ2XS) ? 1024u
                                    : (TYPE == kTypeIQ3S)  ? 512u
                                                           : 256u;
                const uint32_t *src =
                        (TYPE == kTypeIQ2S)
                                ? reinterpret_cast<const uint32_t *>(c_iq2s_grid)
                        : (TYPE == kTypeIQ2XS)
                                ? reinterpret_cast<const uint32_t *>(c_iq2xs_grid)
                        : (TYPE == kTypeIQ3S) ? c_iq3s_grid
                                              : c_iq3xxs_grid;
                uint32_t *dst = reinterpret_cast<uint32_t *>(sGrid);
                const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
                const uint32_t nth = blockDim.x * blockDim.y;
                for (uint32_t i = tid; i < nw; i += nth) dst[i] = src[i];
                __syncthreads();
                gsm = sGrid;
            }
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t lane = threadIdx.x;
            const uint32_t ib32 = lane >> 2u;// 0..7 within a block
            const uint32_t l = lane & 3u;    // 0..3
            const uint32_t bwBytes = (TYPE == kTypeIQ2S)      ? kIQ2S_BYTES
                                     : (TYPE == kTypeIQ2XS)   ? kIQ2XS_BYTES
                                     : (TYPE == kTypeIQ3_XXS) ? kIQ3XXS_BYTES
                                                              : kIQ3S_BYTES;
            float sumf = 0.0f;

            // BLOCK BATCHING (2026-09-27): process blocks in groups of 4.  The
            // old loop was one fully-serial dependency chain per block
            // (q8kLaneSumi loads -> ls scale -> 5-deep shuffle tree -> lane-0
            // float add), measured at ~2000 cycles/block-iteration.  Each group
            // now issues 4 INDEPENDENT (load, scale, tree) chains, giving
            // 4-way memory-level and instruction-level parallelism.  The four
            // float `sumf += d*bsum` updates still run serially at lane 0 in
            // block order, so the float accumulation order -- and therefore the
            // bit-exact per-layer logits -- is unchanged (the integer sumi/bsum
            // are order-free; the serial d*bsum sequence is byte-identical).
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                const uint8_t *bw0 = rp + static_cast<uint64_t>(b + 0u) * bwBytes;
                const uint8_t *bw1 = rp + static_cast<uint64_t>(b + 1u) * bwBytes;
                const uint8_t *bw2 = rp + static_cast<uint64_t>(b + 2u) * bwBytes;
                const uint8_t *bw3 = rp + static_cast<uint64_t>(b + 3u) * bwBytes;
                const uint8_t *ba0 = x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE;
                const uint8_t *ba1 = x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE;
                const uint8_t *ba2 = x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE;
                const uint8_t *ba3 = x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE;

                // 4 independent per-lane integer sums (all loads issue before
                // any of them is consumed -> 4-deep memory-level parallelism).
                int ci0 = q8kLaneSumi<DP4A>(TYPE, bw0,
                                            reinterpret_cast<const int8_t *>(ba0 + kQ8K_Q), ib32, l, gsm);
                int ci1 = q8kLaneSumi<DP4A>(TYPE, bw1,
                                            reinterpret_cast<const int8_t *>(ba1 + kQ8K_Q), ib32, l, gsm);
                int ci2 = q8kLaneSumi<DP4A>(TYPE, bw2,
                                            reinterpret_cast<const int8_t *>(ba2 + kQ8K_Q), ib32, l, gsm);
                int ci3 = q8kLaneSumi<DP4A>(TYPE, bw3,
                                            reinterpret_cast<const int8_t *>(ba3 + kQ8K_Q), ib32, l, gsm);

                // Per-lane ls scaling (order-free integer math).
                ci0 = q8kScaleMuli<TYPE>(bw0, ci0, ib32, l);
                ci1 = q8kScaleMuli<TYPE>(bw1, ci1, ib32, l);
                ci2 = q8kScaleMuli<TYPE>(bw2, ci2, ib32, l);
                ci3 = q8kScaleMuli<TYPE>(bw3, ci3, ib32, l);

                // 4 independent 32-lane trees -> per-block integer bsums.  The
                // trees use the same __shfl_xor_sync mask/order as before, so
                // each bsum is bit-identical to the unbatched loop's.
                int bs0 = ci0 + __shfl_xor_sync(0xffffffffu, ci0, 1);
                bs0 += __shfl_xor_sync(0xffffffffu, bs0, 2);
                bs0 += __shfl_xor_sync(0xffffffffu, bs0, 4);
                bs0 += __shfl_xor_sync(0xffffffffu, bs0, 8);
                bs0 += __shfl_xor_sync(0xffffffffu, bs0, 16);
                int bs1 = ci1 + __shfl_xor_sync(0xffffffffu, ci1, 1);
                bs1 += __shfl_xor_sync(0xffffffffu, bs1, 2);
                bs1 += __shfl_xor_sync(0xffffffffu, bs1, 4);
                bs1 += __shfl_xor_sync(0xffffffffu, bs1, 8);
                bs1 += __shfl_xor_sync(0xffffffffu, bs1, 16);
                int bs2 = ci2 + __shfl_xor_sync(0xffffffffu, ci2, 1);
                bs2 += __shfl_xor_sync(0xffffffffu, bs2, 2);
                bs2 += __shfl_xor_sync(0xffffffffu, bs2, 4);
                bs2 += __shfl_xor_sync(0xffffffffu, bs2, 8);
                bs2 += __shfl_xor_sync(0xffffffffu, bs2, 16);
                int bs3 = ci3 + __shfl_xor_sync(0xffffffffu, ci3, 1);
                bs3 += __shfl_xor_sync(0xffffffffu, bs3, 2);
                bs3 += __shfl_xor_sync(0xffffffffu, bs3, 4);
                bs3 += __shfl_xor_sync(0xffffffffu, bs3, 8);
                bs3 += __shfl_xor_sync(0xffffffffu, bs3, 16);

                // Uniform per-block scale products (identical for every lane;
                // lane 0 consumes them in the serial accumulation below).
                const float d0 =
                        __half2float(*reinterpret_cast<const __half *>(bw0)) *
                        *reinterpret_cast<const float *>(ba0 + kQ8K_D);
                const float d1 =
                        __half2float(*reinterpret_cast<const __half *>(bw1)) *
                        *reinterpret_cast<const float *>(ba1 + kQ8K_D);
                const float d2 =
                        __half2float(*reinterpret_cast<const __half *>(bw2)) *
                        *reinterpret_cast<const float *>(ba2 + kQ8K_D);
                const float d3 =
                        __half2float(*reinterpret_cast<const __half *>(bw3)) *
                        *reinterpret_cast<const float *>(ba3 + kQ8K_D);

                // Lane 0 accumulates serially IN BLOCK ORDER (bit-exact). fmaf
                // would differ by 1 ulp per block, compounding through the KV
                // cache across decode steps and flipping the distribution by
                // step 3+.
                if (lane == 0) {
                    sumf += d0 * static_cast<float>(bs0);
                    sumf += d1 * static_cast<float>(bs1);
                    sumf += d2 * static_cast<float>(bs2);
                    sumf += d3 * static_cast<float>(bs3);
                }
            }
            // Remainder (blocksPerRow % 4) -- original single-block body.
            for (; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * bwBytes;
                const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
                const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA + kQ8K_Q);
                const float yd = *reinterpret_cast<const float *>(blkA + kQ8K_D);
                int ci = q8kLaneSumi<DP4A>(TYPE, blkW, y8, ib32, l, gsm);
                ci = q8kScaleMuli<TYPE>(blkW, ci, ib32, l);
                int bsum = ci + __shfl_xor_sync(0xffffffffu, ci, 1);
                bsum += __shfl_xor_sync(0xffffffffu, bsum, 2);
                bsum += __shfl_xor_sync(0xffffffffu, bsum, 4);
                bsum += __shfl_xor_sync(0xffffffffu, bsum, 8);
                bsum += __shfl_xor_sync(0xffffffffu, bsum, 16);
                if (lane == 0) {
                    const float dval =
                            __half2float(*reinterpret_cast<const __half *>(blkW)) * yd;
                    sumf += dval * static_cast<float>(bsum);
                }
            }
            const float scale = (TYPE == kTypeIQ2S)      ? 0.125f
                                : (TYPE == kTypeIQ2XS)   ? 0.125f
                                : (TYPE == kTypeIQ3_XXS) ? 0.25f
                                                         : 1.0f;
            if (lane == 0) out[row] = scale * sumf;
        }

        // ---- IQ Q8_K GEMV, 4 ROWS PER WARP (2026-10-12).  Bit-exact
        // restructure of kQGemvQ8K<TYPE,DP4A>.  The generic kernel runs one
        // ROW per warp: every row re-reads the whole Q8_K activation and only
        // its own weight block, giving both an activation-re-read cost (all
        // 8960 rows re-read the 1.5 KB activation through L2) and one weight
        // block in flight per warp.  Here the 32 lanes are partitioned into 4
        // groups of 8; group G = lane>>3 owns row = base+G, and the 8 lanes of
        // each group split the block's 8 ib32 sub-blocks (ib32 = lane&7, l =
        // 0).  The lane loads its 8 activation bytes ONCE per block and reuses
        // them for its row's weight (activation-re-read eliminated), and the
        // warp keeps 4 weight blocks in flight.  Each lane holds the WHOLE
        // per-block scalar for its row (no 32-lane tree), so the per-row
        // block-order float accumulation is byte-identical to kQGemvQ8K's
        // lane-0 `sumf += d*bsum`; only the final per-type scale is applied by
        // the group leader (lane&7==0).  Requires rows%4==0.
        // Env gate TINYCODER_IQ_4XW (default ON).
        template<int TYPE, bool DP4A = false>
        __device__ __forceinline__ int q8kBlockSum(
                const uint8_t *__restrict__ bw, const uint8_t *__restrict__ ba,
                uint32_t ype, const uint8_t *__restrict__ gsm) {
            const int8_t *y8b = reinterpret_cast<const int8_t *>(ba + kQ8K_Q);
            int sumi = 0;
#pragma unroll
            for (uint32_t ll = 0; ll < 4; ++ll) {
                const uint64_t yv = __ldg(reinterpret_cast<const uint64_t *>(
                        &y8b[ype * 32u + ll * 8u]));
                int c = q8kLaneSumiV<DP4A>(TYPE, bw, yv, ype, ll, gsm);
                c = q8kScaleMuli<TYPE>(bw, c, ype, ll);
                sumi += c;
            }
            sumi += __shfl_xor_sync(0xffffffffu, sumi, 1);
            sumi += __shfl_xor_sync(0xffffffffu, sumi, 2);
            sumi += __shfl_xor_sync(0xffffffffu, sumi, 4);
            return sumi;
        }

        template<int TYPE, bool DP4A = false>
        __global__ void __launch_bounds__(256) kQGemvQ8K_4xW(
                const uint8_t *__restrict__ w, const uint8_t *__restrict__ x,
                float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow,
                uint32_t rowBytes, bool grsm = false) {
            extern __shared__ uint8_t sGrid[];
            const uint8_t *gsm = nullptr;
            if (grsm) {
                const uint32_t nw = (TYPE == kTypeIQ2S)    ? 2048u
                                    : (TYPE == kTypeIQ2XS) ? 1024u
                                    : (TYPE == kTypeIQ3S)  ? 512u
                                                           : 256u;
                const uint32_t *src =
                        (TYPE == kTypeIQ2S)
                                ? reinterpret_cast<const uint32_t *>(c_iq2s_grid)
                        : (TYPE == kTypeIQ2XS)
                                ? reinterpret_cast<const uint32_t *>(c_iq2xs_grid)
                        : (TYPE == kTypeIQ3S) ? c_iq3s_grid
                                              : c_iq3xxs_grid;
                uint32_t *dst = reinterpret_cast<uint32_t *>(sGrid);
                const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
                const uint32_t nth = blockDim.x * blockDim.y;
                for (uint32_t i = tid; i < nw; i += nth) dst[i] = src[i];
                __syncthreads();
                gsm = sGrid;
            }
            const uint32_t lane = threadIdx.x;
            const uint32_t ype = lane & 7u;// ib32 sub-block this lane owns
            const uint32_t rr = lane >> 3u;// 0..3 row within the warp
            const uint32_t base = (blockIdx.x * blockDim.y + threadIdx.y) * 4u;
            if (base >= rows) return;
            const uint32_t row = base + rr;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t bwBytes = (TYPE == kTypeIQ2S)      ? kIQ2S_BYTES
                                     : (TYPE == kTypeIQ2XS)   ? kIQ2XS_BYTES
                                     : (TYPE == kTypeIQ3_XXS) ? kIQ3XXS_BYTES
                                                              : kIQ3S_BYTES;
            float sumf = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                const uint8_t *ba0 = x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE;
                const uint8_t *ba1 = x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE;
                const uint8_t *ba2 = x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE;
                const uint8_t *ba3 = x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE;
                const uint8_t *bw0 = rp + static_cast<uint64_t>(b + 0u) * bwBytes;
                const uint8_t *bw1 = rp + static_cast<uint64_t>(b + 1u) * bwBytes;
                const uint8_t *bw2 = rp + static_cast<uint64_t>(b + 2u) * bwBytes;
                const uint8_t *bw3 = rp + static_cast<uint64_t>(b + 3u) * bwBytes;
                const int bs0 = q8kBlockSum<TYPE, DP4A>(bw0, ba0, ype, gsm);
                const int bs1 = q8kBlockSum<TYPE, DP4A>(bw1, ba1, ype, gsm);
                const int bs2 = q8kBlockSum<TYPE, DP4A>(bw2, ba2, ype, gsm);
                const int bs3 = q8kBlockSum<TYPE, DP4A>(bw3, ba3, ype, gsm);
                const float d0 =
                        __half2float(*reinterpret_cast<const __half *>(bw0)) *
                        *reinterpret_cast<const float *>(ba0 + kQ8K_D);
                const float d1 =
                        __half2float(*reinterpret_cast<const __half *>(bw1)) *
                        *reinterpret_cast<const float *>(ba1 + kQ8K_D);
                const float d2 =
                        __half2float(*reinterpret_cast<const __half *>(bw2)) *
                        *reinterpret_cast<const float *>(ba2 + kQ8K_D);
                const float d3 =
                        __half2float(*reinterpret_cast<const __half *>(bw3)) *
                        *reinterpret_cast<const float *>(ba3 + kQ8K_D);
                sumf += d0 * static_cast<float>(bs0);
                sumf += d1 * static_cast<float>(bs1);
                sumf += d2 * static_cast<float>(bs2);
                sumf += d3 * static_cast<float>(bs3);
            }
            for (; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) * bwBytes;
                const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
                const int bsum = q8kBlockSum<TYPE, DP4A>(blkW, blkA, ype, gsm);
                const float dval =
                        __half2float(*reinterpret_cast<const __half *>(blkW)) *
                        *reinterpret_cast<const float *>(blkA + kQ8K_D);
                sumf += dval * static_cast<float>(bsum);
            }
            const float scale = (TYPE == kTypeIQ2S)      ? 0.125f
                                : (TYPE == kTypeIQ2XS)   ? 0.125f
                                : (TYPE == kTypeIQ3_XXS) ? 0.25f
                                                         : 1.0f;
            if (ype == 0u) out[row] = scale * sumf;
        }

        // ------------------------------------------------------------------
        // FUSED gate+up+silu (decode seqLen==1, dense FFN, 2026-09-28).
        //
        // The per-layer FFN path used to launch gate, up and kSiluMul as THREE
        // kernels per layer (40 layers => ~120 launches/token just for the
        // FFN).  The fused kernels below compute BOTH row dots in one warp and
        // fold the silu(gate)*up elementwise product into the gate store, so
        // ONE launch replaces the three, and (for the Q8K-int path) the fp32
        // activation is quantized to Q8_K ONCE and shared by both matrices.
        //
        // BIT-EXACTNESS: each matrix's accumulation is byte-identical to the
        // separate-kernel math:
        //   * float path (kQGemvFusedGU_IQ2XS): identical per-lane fmaf chain
        //     over blocks (incl. the 4-way a0..a3 block batching with serial
        //     folds) and the identical 5-shuffle warp tree per matrix;
        //   * Q8K-int path (kQGemvQ8KFusedGU): identical q8kLaneSumi +
        //     q8kScaleMuli integer sums, identical per-block int tree, and the
        //     serial lane-0 `sumf += d*bsum` in block order, with the trailing
        //     per-type scale applied before the silu product exactly like the
        //     separate kernels + kSiluMul.
        // The fused silu = x/(1+__expf(-x)) -- the kSiluMul expression --
        // applied to the two fully-reduced row values.

        // ---- Fused gate+up+silu for IQ2_XS (float dequant-dot).
        __global__ void __launch_bounds__(256) kQGemvFusedGU_IQ2XS(
                const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
                const float *__restrict__ x, float *__restrict__ out,
                uint32_t rows, uint32_t cols, uint32_t rowBytes,
                uint32_t blocksPerRow) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rg = wg + static_cast<uint64_t>(row) * rowBytes;
            const uint8_t *ru = wu + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t lane = threadIdx.x;
            const uint32_t e0 = lane * 8u;
            const uint32_t ib32 = e0 >> 5u;    // sub-block 0..7
            const uint32_t l = (e0 >> 3u) & 3u;// 0..3
            float accg = 0.0f, accu = 0.0f;
            uint32_t b = 0;
            // 4-way block batching (mirrors kQGemv<kTypeIQ2XS>): each group
            // issues 8 independent (load, decode, fmaf-chain) streams (4 per
            // matrix) sharing the same xs activation window per block.
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
#pragma unroll 4
                for (uint32_t bb = 0; bb < 4; ++bb) {
                    const uint8_t *crpG =
                            rg + static_cast<uint64_t>(bb) * kIQ2XS_BYTES;
                    const uint8_t *crpU =
                            ru + static_cast<uint64_t>(bb) * kIQ2XS_BYTES;
                    __half dG = *reinterpret_cast<const __half *>(crpG + 0);
                    __half dU = *reinterpret_cast<const __half *>(crpU + 0);
                    float dfG = __half2float(dG), dfU = __half2float(dU);
                    const uint16_t *qsG =
                            reinterpret_cast<const uint16_t *>(crpG + 2);
                    const uint16_t *qsU =
                            reinterpret_cast<const uint16_t *>(crpU + 2);
                    const uint8_t *scG = crpG + 66;
                    const uint8_t *scU = crpU + 66;
                    float gb0 = dfG * (0.5f + static_cast<float>(scG[ib32] & 0xf)) * 0.25f;
                    float gb1 = dfG * (0.5f + static_cast<float>(scG[ib32] >> 4)) * 0.25f;
                    float ub0 = dfU * (0.5f + static_cast<float>(scU[ib32] & 0xf)) * 0.25f;
                    float ub1 = dfU * (0.5f + static_cast<float>(scU[ib32] >> 4)) * 0.25f;
                    float dlG = (l >= 2u) ? gb1 : gb0;
                    float dlU = (l >= 2u) ? ub1 : ub0;
                    uint16_t qg = qsG[ib32 * 4u + l];
                    uint16_t qu = qsU[ib32 * 4u + l];
                    const uint64_t gvG = c_iq2xs_grid[qg & 0x1FFu];
                    const uint64_t gvU = c_iq2xs_grid[qu & 0x1FFu];
                    uint8_t sgnG = c_ksigns_iq2xs[qg >> 9];
                    uint8_t sgnU = c_ksigns_iq2xs[qu >> 9];
                    const float *xs = x + (b + bb) * 256u + e0;
                    float &slotG = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                             : (bb == 2u)       ? a2
                                                                : a3;
                    float &slotU = (bb == 0u) ? u0 : (bb == 1u) ? u1
                                             : (bb == 2u)       ? u2
                                                                : u3;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        float wG = dlG *
                                   static_cast<float>((gvG >> (8u * j)) & 0xFFu) *
                                   ((sgnG & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        float wU = dlU *
                                   static_cast<float>((gvU >> (8u * j)) & 0xFFu) *
                                   ((sgnU & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t ck = (b + bb) * 256u + e0 + j;
                        const float xv = (ck < cols) ? xs[j] : 0.0f;
                        slotG = fmaf(wG, xv, slotG);
                        slotU = fmaf(wU, xv, slotU);
                    }
                }
                accg += a0;
                accg += a1;
                accg += a2;
                accg += a3;
                accu += u0;
                accu += u1;
                accu += u2;
                accu += u3;
                rg += 4u * kIQ2XS_BYTES;
                ru += 4u * kIQ2XS_BYTES;
            }
            // Remainder (blocksPerRow % 4): original single-block serial chain.
            for (; b < blocksPerRow; ++b) {
                __half dG = *reinterpret_cast<const __half *>(rg + 0);
                __half dU = *reinterpret_cast<const __half *>(ru + 0);
                float dfG = __half2float(dG), dfU = __half2float(dU);
                const uint16_t *qsG = reinterpret_cast<const uint16_t *>(rg + 2);
                const uint16_t *qsU = reinterpret_cast<const uint16_t *>(ru + 2);
                const uint8_t *scG = rg + 66;
                const uint8_t *scU = ru + 66;
                float gb0 = dfG * (0.5f + static_cast<float>(scG[ib32] & 0xf)) * 0.25f;
                float gb1 = dfG * (0.5f + static_cast<float>(scG[ib32] >> 4)) * 0.25f;
                float ub0 = dfU * (0.5f + static_cast<float>(scU[ib32] & 0xf)) * 0.25f;
                float ub1 = dfU * (0.5f + static_cast<float>(scU[ib32] >> 4)) * 0.25f;
                float dlG = (l >= 2u) ? gb1 : gb0;
                float dlU = (l >= 2u) ? ub1 : ub0;
                uint16_t qg = qsG[ib32 * 4u + l];
                uint16_t qu = qsU[ib32 * 4u + l];
                const uint64_t gvG = c_iq2xs_grid[qg & 0x1FFu];
                const uint64_t gvU = c_iq2xs_grid[qu & 0x1FFu];
                uint8_t sgnG = c_ksigns_iq2xs[qg >> 9];
                uint8_t sgnU = c_ksigns_iq2xs[qu >> 9];
                const float *xs = x + b * 256u + e0;
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    float wG = dlG * static_cast<float>((gvG >> (8u * j)) & 0xFFu) *
                               ((sgnG & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    float wU = dlU * static_cast<float>((gvU >> (8u * j)) & 0xFFu) *
                               ((sgnU & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    const uint32_t ck = b * 256u + e0 + j;
                    const float xv = (ck < cols) ? xs[j] : 0.0f;
                    accg = fmaf(wG, xv, accg);
                    accu = fmaf(wU, xv, accu);
                }
                rg += kIQ2XS_BYTES;
                ru += kIQ2XS_BYTES;
            }
            // Two independent 5-shuffle warp trees (identical order per matrix).
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1) {
                accg += __shfl_xor_sync(0xffffffffu, accg, off);
                accu += __shfl_xor_sync(0xffffffffu, accu, off);
            }
            if (lane == 0) {
                const float g = accg;
                out[row] = (g / (1.0f + __expf(-g))) * accu;
            }
        }

        // ---- Fused gate+up+silu for Q2_K (float dequant-dot) ----
        // The qwen2.5-coder-1.5b q2_k FFN gate/up are Q2_K (type 10), which
        // the IQ-only fused dispatcher did NOT cover -- the FFN ran as THREE
        // launches (gate + up + kSiluMul) per layer.  This kernel computes both
        // row dots in one warp and folds silu(gate)*up into the store, exactly
        // like the IQ2XS fused kernel: each matrix's fmaf chain + 4-way a0..a3
        // block batching + in-order folds + 5-shuffle warp tree are the
        // byte-identical kQGemv<kTypeQ2K> math; the silu is the kSiluMul
        // expression.  4-way block batching gives 8 independent (load, dequant,
        // fmaf) streams (4 per matrix) sharing the x activation window.
        __global__ void __launch_bounds__(256) kQGemvFusedGU_Q2K(
                const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
                const float *__restrict__ x, float *__restrict__ out,
                uint32_t rows, uint32_t cols, uint32_t rowBytes,
                uint32_t blocksPerRow) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rg = wg + static_cast<uint64_t>(row) * rowBytes;
            const uint8_t *ru = wu + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t lane = threadIdx.x;
            const uint32_t sub = lane / 16u;
            float accg = 0.0f, accu = 0.0f;
            uint32_t b = 0;
            for (; b + 4u <= blocksPerRow; b += 4u) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
#pragma unroll 4
                for (uint32_t bb = 0; bb < 4; ++bb) {
                    const uint8_t *crpG = rg + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                    const uint8_t *crpU = ru + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                    __half dG = *reinterpret_cast<const __half *>(crpG + 80);
                    __half dminG = *reinterpret_cast<const __half *>(crpG + 82);
                    __half dU = *reinterpret_cast<const __half *>(crpU + 80);
                    __half dminU = *reinterpret_cast<const __half *>(crpU + 82);
                    float dfG = __half2float(dG), dmfG = __half2float(dminG);
                    float dfU = __half2float(dU), dmfU = __half2float(dminU);
                    const uint8_t *scG = crpG;
                    const uint8_t *qG = crpG + 16;
                    const uint8_t *scU = crpU;
                    const uint8_t *qU = crpU + 16;
                    float &slotG = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                             : (bb == 2u)       ? a2
                                                                : a3;
                    float &slotU = (bb == 0u) ? u0 : (bb == 1u) ? u1
                                             : (bb == 2u)       ? u2
                                                                : u3;
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qbG = qG[half * 32u + lane];
                        const uint8_t qbU = qU[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            const uint8_t scvG = scG[half * 8u + jj * 2u + sub];
                            const uint8_t scvU = scU[half * 8u + jj * 2u + sub];
                            float dlG = dfG * static_cast<float>(scvG & 0xF);
                            float mlG = dmfG * static_cast<float>(scvG >> 4);
                            float dlU = dfU * static_cast<float>(scvU & 0xF);
                            float mlU = dmfU * static_cast<float>(scvU >> 4);
                            float qvG = static_cast<float>(
                                    static_cast<int8_t>((qbG >> (jj * 2u)) & 3));
                            float qvU = static_cast<float>(
                                    static_cast<int8_t>((qbU >> (jj * 2u)) & 3));
                            const uint32_t cidx =
                                    (b + bb) * 256u + half * 128u + jj * 32u + lane;
                            const float xv = (cidx < cols) ? x[cidx] : 0.0f;
                            slotG = fmaf(fmaf(dlG, qvG, -mlG), xv, slotG);
                            slotU = fmaf(fmaf(dlU, qvU, -mlU), xv, slotU);
                        }
                    }
                }
                accg += a0;
                accg += a1;
                accg += a2;
                accg += a3;
                accu += u0;
                accu += u1;
                accu += u2;
                accu += u3;
                rg += 4u * kQ2K_BYTES;
                ru += 4u * kQ2K_BYTES;
            }
            for (; b < blocksPerRow; ++b) {
                __half dG = *reinterpret_cast<const __half *>(rg + 80);
                __half dminG = *reinterpret_cast<const __half *>(rg + 82);
                __half dU = *reinterpret_cast<const __half *>(ru + 80);
                __half dminU = *reinterpret_cast<const __half *>(ru + 82);
                float dfG = __half2float(dG), dmfG = __half2float(dminG);
                float dfU = __half2float(dU), dmfU = __half2float(dminU);
                const uint8_t *scG = rg;
                const uint8_t *qG = rg + 16;
                const uint8_t *scU = ru;
                const uint8_t *qU = ru + 16;
#pragma unroll
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint8_t qbG = qG[half * 32u + lane];
                    const uint8_t qbU = qU[half * 32u + lane];
#pragma unroll
                    for (uint32_t jj = 0; jj < 4; ++jj) {
                        const uint8_t scvG = scG[half * 8u + jj * 2u + sub];
                        const uint8_t scvU = scU[half * 8u + jj * 2u + sub];
                        float dlG = dfG * static_cast<float>(scvG & 0xF);
                        float mlG = dmfG * static_cast<float>(scvG >> 4);
                        float dlU = dfU * static_cast<float>(scvU & 0xF);
                        float mlU = dmfU * static_cast<float>(scvU >> 4);
                        float qvG = static_cast<float>(
                                static_cast<int8_t>((qbG >> (jj * 2u)) & 3));
                        float qvU = static_cast<float>(
                                static_cast<int8_t>((qbU >> (jj * 2u)) & 3));
                        const uint32_t cidx =
                                b * 256u + half * 128u + jj * 32u + lane;
                        const float xv = (cidx < cols) ? x[cidx] : 0.0f;
                        accg = fmaf(fmaf(dlG, qvG, -mlG), xv, accg);
                        accu = fmaf(fmaf(dlU, qvU, -mlU), xv, accu);
                    }
                }
                rg += kQ2K_BYTES;
                ru += kQ2K_BYTES;
            }
            // Two independent 5-shuffle warp trees (identical order per matrix).
#pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1) {
                accg += __shfl_xor_sync(0xffffffffu, accg, off);
                accu += __shfl_xor_sync(0xffffffffu, accu, off);
            }
            if (lane == 0) {
                const float g = accg;
                out[row] = (g / (1.0f + __expf(-g))) * accu;
            }
        }

        // ---- Fused Q+K (decode, Q2_K float dequant-dot) ----
        // qwen2.5-coder-1.5b q2_k attn_q/attn_k are BOTH Q2_K against the SAME
        // activation (s.norm).  The decode path launched them as two separate
        // kQGemv<kTypeQ2K> kernels; this fused kernel computes BOTH rows in one
        // warp so one launch replaces two.  Q rows are [qRows] and K rows are
        // [kRows]; each warp handles Q row `row` and, when `row < kRows`, also
        // K row `row`.  The per-matrix math is byte-identical to
        // kQGemv<kTypeQ2K> (same fmaf chain, same 4-way batching + in-order
        // folds, same 5-shuffle tree), so bit-exactness is preserved.
        __global__ void __launch_bounds__(256) kQGemvFusedQK_Q2K(
                const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wk,
                const float *__restrict__ x, float *__restrict__ outQ,
                float *__restrict__ outK, uint32_t qRows, uint32_t kRows,
                uint32_t cols, uint32_t qRowBytes, uint32_t kRowBytes,
                uint32_t qBlocksPerRow, uint32_t kBlocksPerRow) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= qRows) return;
            const uint32_t lane = threadIdx.x;
            const uint32_t sub = lane / 16u;

            // ---- Q row ----
            {
                const uint8_t *rp = wq + static_cast<uint64_t>(row) * qRowBytes;
                float acc = 0.0f;
                uint32_t b = 0;
                for (; b + 4u <= qBlocksPerRow; b += 4u) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                    for (uint32_t bb = 0; bb < 4; ++bb) {
                        const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                        __half d = *reinterpret_cast<const __half *>(crp + 80);
                        __half dmin = *reinterpret_cast<const __half *>(crp + 82);
                        float df = __half2float(d), dmf = __half2float(dmin);
                        const uint8_t *sc = crp;
                        const uint8_t *q = crp + 16;
                        float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                : (bb == 2u)       ? a2
                                                                   : a3;
#pragma unroll
                        for (uint32_t half = 0; half < 2; ++half) {
                            const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                            for (uint32_t jj = 0; jj < 4; ++jj) {
                                const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                                float dl = df * static_cast<float>(scv & 0xF);
                                float ml = dmf * static_cast<float>(scv >> 4);
                                float qv = static_cast<float>(
                                        static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                                const uint32_t cidx =
                                        (b + bb) * 256u + half * 128u + jj * 32u + lane;
                                slot = fmaf(fmaf(dl, qv, -ml),
                                            (cidx < cols) ? x[cidx] : 0.0f, slot);
                            }
                        }
                    }
                    acc += a0;
                    acc += a1;
                    acc += a2;
                    acc += a3;
                    rp += 4u * kQ2K_BYTES;
                }
                for (; b < qBlocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 80);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 82);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *sc = rp;
                    const uint8_t *q = rp + 16;
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                            float dl = df * static_cast<float>(scv & 0xF);
                            float ml = dmf * static_cast<float>(scv >> 4);
                            float qv = static_cast<float>(
                                    static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                            const uint32_t cidx =
                                    b * 256u + half * 128u + jj * 32u + lane;
                            acc = fmaf(fmaf(dl, qv, -ml),
                                       (cidx < cols) ? x[cidx] : 0.0f, acc);
                        }
                    }
                    rp += kQ2K_BYTES;
                }
#pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1) {
                    acc += __shfl_xor_sync(0xffffffffu, acc, off);
                }
                if (lane == 0) outQ[row] = acc;
            }

            // ---- K row (only warps with row < kRows) ----
            if (row < kRows) {
                const uint8_t *rp = wk + static_cast<uint64_t>(row) * kRowBytes;
                float acc = 0.0f;
                uint32_t b = 0;
                for (; b + 4u <= kBlocksPerRow; b += 4u) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                    for (uint32_t bb = 0; bb < 4; ++bb) {
                        const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                        __half d = *reinterpret_cast<const __half *>(crp + 80);
                        __half dmin = *reinterpret_cast<const __half *>(crp + 82);
                        float df = __half2float(d), dmf = __half2float(dmin);
                        const uint8_t *sc = crp;
                        const uint8_t *q = crp + 16;
                        float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                : (bb == 2u)       ? a2
                                                                   : a3;
#pragma unroll
                        for (uint32_t half = 0; half < 2; ++half) {
                            const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                            for (uint32_t jj = 0; jj < 4; ++jj) {
                                const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                                float dl = df * static_cast<float>(scv & 0xF);
                                float ml = dmf * static_cast<float>(scv >> 4);
                                float qv = static_cast<float>(
                                        static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                                const uint32_t cidx =
                                        (b + bb) * 256u + half * 128u + jj * 32u + lane;
                                slot = fmaf(fmaf(dl, qv, -ml),
                                            (cidx < cols) ? x[cidx] : 0.0f, slot);
                            }
                        }
                    }
                    acc += a0;
                    acc += a1;
                    acc += a2;
                    acc += a3;
                    rp += 4u * kQ2K_BYTES;
                }
                for (; b < kBlocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 80);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 82);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *sc = rp;
                    const uint8_t *q = rp + 16;
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                            float dl = df * static_cast<float>(scv & 0xF);
                            float ml = dmf * static_cast<float>(scv >> 4);
                            float qv = static_cast<float>(
                                    static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                            const uint32_t cidx =
                                    b * 256u + half * 128u + jj * 32u + lane;
                            acc = fmaf(fmaf(dl, qv, -ml),
                                       (cidx < cols) ? x[cidx] : 0.0f, acc);
                        }
                    }
                    rp += kQ2K_BYTES;
                }
#pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1) {
                    acc += __shfl_xor_sync(0xffffffffu, acc, off);
                }
                if (lane == 0) outK[row] = acc;
            }
        }

        // ---- Fused Q+K+V (decode; qwen2.5-coder-1.5b q2_k: attn_q/attn_k are
        // Q2_K and attn_v is Q4_K against the SAME fp32 activation).  Extends
        // kQGemvFusedQK_Q2K with a V section (float dequant-dot, faithful to
        // dequantizeQ4_KBlock) so ONE launch replaces fused-QK + the Q8_K
        // quantize + the v GEMV, and the three bias adds fold into the
        // epilogues (out = acc + bias[row] -- the SAME float add kAddBias
        // performed after the store, hence bit-exact for Q/K).  Q/K per-matrix
        // math is byte-identical to kQGemvFusedQK_Q2K.  Warps [0, qRows) do Q
        // (and K when row < kRows); warps [qRows, qRows+vRows) do V.
        // TINYCODER_FUSE_QKV=0 opts out (A/B record).
        __global__ void __launch_bounds__(256) kQGemvFusedQKV(
                const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wk,
                const uint8_t *__restrict__ wv, const float *__restrict__ x,
                float *__restrict__ outQ, float *__restrict__ outK,
                float *__restrict__ outV, uint32_t qRows, uint32_t kRows,
                uint32_t vRows, uint32_t cols, uint32_t qRowBytes,
                uint32_t kRowBytes, uint32_t vRowBytes, uint32_t qBlocksPerRow,
                uint32_t kBlocksPerRow, uint32_t vBlocksPerRow,
                const float *__restrict__ biasQ, const float *__restrict__ biasK,
                const float *__restrict__ biasV,
                const uint8_t *__restrict__ xq81) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            const uint32_t lane = threadIdx.x;
            const uint32_t sub = lane / 16u;
            if (row < qRows) {
                // ---- Q row (byte-identical to kQGemvFusedQK_Q2K) ----
                {
                    const uint8_t *rp = wq + static_cast<uint64_t>(row) * qRowBytes;
                    if (xq81 != nullptr) {
                        // Weight int loads are int-cast derefs: 504-B row
                        // stride and 84-B block stride are 4-aligned.
                        // Accumulation order differs from the float body (same
                        // tolerance class as the GLU/down mmvq adoptions).
                        float acc = 0.0f;
                        const uint32_t kqs = lane % 16u;
                        for (uint32_t kbx = lane / 16u; kbx < qBlocksPerRow;
                             kbx += 2u) {
                            const uint8_t *bq =
                                    rp + static_cast<uint64_t>(kbx) * 84u;
                            const int bq8_offset = 4 * (kqs / 8);
                            const int scale_offset =
                                    kqs - kqs % 8 + (kqs % 8) / 4;
                            const uint8_t *scales_bq = bq + scale_offset;
                            const uint8_t *qs_bq = bq + 16;
                            const int v = *reinterpret_cast<const int *>(
                                    qs_bq + 4 * kqs);
                            int u[4];
                            float d8[4];
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const uint8_t *by =
                                        xq81 + static_cast<uint64_t>(
                                                       kbx * 8u + bq8_offset + i) *
                                                       40u;
                                u[i] = *reinterpret_cast<const int *>(
                                        by + 4 + 4 * (kqs % 8));
                                d8[i] = __half2float(
                                        *reinterpret_cast<const __half *>(by));
                            }
                            float sd = 0.0f, sm = 0.0f;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int sc = scales_bq[2 * i];
                                const int vi = (v >> (2 * i)) & 0x03030303;
                                sd += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF));
                                int m = sc >> 4;
                                m |= m << 8;
                                m |= m << 16;
                                sm += d8[i] * __dp4a(m, u[i], 0);
                            }
                            const float2 dm = __half22float2(
                                    *reinterpret_cast<const __half2 *>(bq + 80));
                            acc += dm.x * sd - dm.y * sm;
                        }
#pragma unroll
                        for (uint32_t off = 16; off > 0; off >>= 1) {
                            acc += __shfl_xor_sync(0xffffffffu, acc, off);
                        }
                        if (lane == 0)
                            outQ[row] = biasQ ? (acc + biasQ[row]) : acc;
                    } else {
                        float acc = 0.0f;
                        uint32_t b = 0;
                        for (; b + 4u <= qBlocksPerRow; b += 4u) {
                            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                            for (uint32_t bb = 0; bb < 4; ++bb) {
                                const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                                __half d = *reinterpret_cast<const __half *>(crp + 80);
                                __half dmin = *reinterpret_cast<const __half *>(crp + 82);
                                float df = __half2float(d), dmf = __half2float(dmin);
                                const uint8_t *sc = crp;
                                const uint8_t *q = crp + 16;
                                float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                        : (bb == 2u)       ? a2
                                                                           : a3;
#pragma unroll
                                for (uint32_t half = 0; half < 2; ++half) {
                                    const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                                    for (uint32_t jj = 0; jj < 4; ++jj) {
                                        const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                                        float dl = df * static_cast<float>(scv & 0xF);
                                        float ml = dmf * static_cast<float>(scv >> 4);
                                        float qv = static_cast<float>(
                                                static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                                        const uint32_t cidx =
                                                (b + bb) * 256u + half * 128u + jj * 32u + lane;
                                        slot = fmaf(fmaf(dl, qv, -ml),
                                                    (cidx < cols) ? x[cidx] : 0.0f, slot);
                                    }
                                }
                            }
                            acc += a0;
                            acc += a1;
                            acc += a2;
                            acc += a3;
                            rp += 4u * kQ2K_BYTES;
                        }
                        for (; b < qBlocksPerRow; ++b) {
                            __half d = *reinterpret_cast<const __half *>(rp + 80);
                            __half dmin = *reinterpret_cast<const __half *>(rp + 82);
                            float df = __half2float(d), dmf = __half2float(dmin);
                            const uint8_t *sc = rp;
                            const uint8_t *q = rp + 16;
#pragma unroll
                            for (uint32_t half = 0; half < 2; ++half) {
                                const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                                for (uint32_t jj = 0; jj < 4; ++jj) {
                                    const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                                    float dl = df * static_cast<float>(scv & 0xF);
                                    float ml = dmf * static_cast<float>(scv >> 4);
                                    float qv = static_cast<float>(
                                            static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                                    const uint32_t cidx =
                                            b * 256u + half * 128u + jj * 32u + lane;
                                    acc = fmaf(fmaf(dl, qv, -ml),
                                               (cidx < cols) ? x[cidx] : 0.0f, acc);
                                }
                            }
                            rp += kQ2K_BYTES;
                        }
#pragma unroll
                        for (uint32_t off = 16; off > 0; off >>= 1) {
                            acc += __shfl_xor_sync(0xffffffffu, acc, off);
                        }
                        // Folded bias (bit-exact: same float add kAddBias does).
                        if (lane == 0)
                            outQ[row] = biasQ ? (acc + biasQ[row]) : acc;
                    }
                }
                // ---- K row (warps with row < kRows also run K after Q --
                // a split-K-per-warp variant measured neutral/-0.3 % (the
                // stage is not critical-path bound at 224 blocks), so the
                // simpler serial layout ships) ----
                if (row < kRows) {
                    const uint8_t *rp = wk + static_cast<uint64_t>(row) * kRowBytes;
                    if (xq81 != nullptr) {
                        // dp4a Q2_K x Q8_1, same structure as the Q branch
                        // above.  kBlocksPerRow == 1 here: only lanes 0-15
                        // carry work, the rest contribute zero to the reduce.
                        float acc = 0.0f;
                        const uint32_t kqs = lane % 16u;
                        for (uint32_t kbx = lane / 16u; kbx < kBlocksPerRow;
                             kbx += 2u) {
                            const uint8_t *bq =
                                    rp + static_cast<uint64_t>(kbx) * 84u;
                            const int bq8_offset = 4 * (kqs / 8);
                            const int scale_offset =
                                    kqs - kqs % 8 + (kqs % 8) / 4;
                            const uint8_t *scales_bq = bq + scale_offset;
                            const uint8_t *qs_bq = bq + 16;
                            const int v = *reinterpret_cast<const int *>(
                                    qs_bq + 4 * kqs);
                            int u[4];
                            float d8[4];
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const uint8_t *by =
                                        xq81 + static_cast<uint64_t>(
                                                       kbx * 8u + bq8_offset + i) *
                                                       40u;
                                u[i] = *reinterpret_cast<const int *>(
                                        by + 4 + 4 * (kqs % 8));
                                d8[i] = __half2float(
                                        *reinterpret_cast<const __half *>(by));
                            }
                            float sd = 0.0f, sm = 0.0f;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int sc = scales_bq[2 * i];
                                const int vi = (v >> (2 * i)) & 0x03030303;
                                sd += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF));
                                int m = sc >> 4;
                                m |= m << 8;
                                m |= m << 16;
                                sm += d8[i] * __dp4a(m, u[i], 0);
                            }
                            const float2 dm = __half22float2(
                                    *reinterpret_cast<const __half2 *>(bq + 80));
                            acc += dm.x * sd - dm.y * sm;
                        }
#pragma unroll
                        for (uint32_t off = 16; off > 0; off >>= 1) {
                            acc += __shfl_xor_sync(0xffffffffu, acc, off);
                        }
                        if (lane == 0)
                            outK[row] = biasK ? (acc + biasK[row]) : acc;
                    } else {
                        float acc = 0.0f;
                        uint32_t b = 0;
                        for (; b + 4u <= kBlocksPerRow; b += 4u) {
                            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                            for (uint32_t bb = 0; bb < 4; ++bb) {
                                const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                                __half d = *reinterpret_cast<const __half *>(crp + 80);
                                __half dmin = *reinterpret_cast<const __half *>(crp + 82);
                                float df = __half2float(d), dmf = __half2float(dmin);
                                const uint8_t *sc = crp;
                                const uint8_t *q = crp + 16;
                                float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                        : (bb == 2u)       ? a2
                                                                           : a3;
#pragma unroll
                                for (uint32_t half = 0; half < 2; ++half) {
                                    const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                                    for (uint32_t jj = 0; jj < 4; ++jj) {
                                        const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                                        float dl = df * static_cast<float>(scv & 0xF);
                                        float ml = dmf * static_cast<float>(scv >> 4);
                                        float qv = static_cast<float>(
                                                static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                                        const uint32_t cidx =
                                                (b + bb) * 256u + half * 128u + jj * 32u + lane;
                                        slot = fmaf(fmaf(dl, qv, -ml),
                                                    (cidx < cols) ? x[cidx] : 0.0f, slot);
                                    }
                                }
                            }
                            acc += a0;
                            acc += a1;
                            acc += a2;
                            acc += a3;
                            rp += 4u * kQ2K_BYTES;
                        }
                        for (; b < kBlocksPerRow; ++b) {
                            __half d = *reinterpret_cast<const __half *>(rp + 80);
                            __half dmin = *reinterpret_cast<const __half *>(rp + 82);
                            float df = __half2float(d), dmf = __half2float(dmin);
                            const uint8_t *sc = rp;
                            const uint8_t *q = rp + 16;
#pragma unroll
                            for (uint32_t half = 0; half < 2; ++half) {
                                const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                                for (uint32_t jj = 0; jj < 4; ++jj) {
                                    const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                                    float dl = df * static_cast<float>(scv & 0xF);
                                    float ml = dmf * static_cast<float>(scv >> 4);
                                    float qv = static_cast<float>(
                                            static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                                    const uint32_t cidx =
                                            b * 256u + half * 128u + jj * 32u + lane;
                                    acc = fmaf(fmaf(dl, qv, -ml),
                                               (cidx < cols) ? x[cidx] : 0.0f, acc);
                                }
                            }
                            rp += kQ2K_BYTES;
                        }
#pragma unroll
                        for (uint32_t off = 16; off > 0; off >>= 1) {
                            acc += __shfl_xor_sync(0xffffffffu, acc, off);
                        }
                        if (lane == 0)
                            outK[row] = biasK ? (acc + biasK[row]) : acc;
                    }
                }
            } else {
                // ---- V row: Q4_K float dequant-dot (bit-faithful to
                // dequantizeQ4_KBlock), 4-way block batching with in-order
                // folds (same structure as the Q2_K sections above).
                const uint32_t vrow = row - qRows;
                if (vrow >= vRows) return;
                const uint8_t *rp = wv + static_cast<uint64_t>(vrow) * vRowBytes;
                auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                      uint8_t *m_out) {
                    if (j < 4) {
                        *d_out = q[j] & 63;
                        *m_out = q[j + 4] & 63;
                    } else {
                        *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                        *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                    }
                };
                float acc = 0.0f;
                uint32_t b = 0;
                for (; b + 4u <= vBlocksPerRow; b += 4u) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
                    for (uint32_t bb = 0; bb < 4; ++bb) {
                        const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ4K_BYTES;
                        __half d = *reinterpret_cast<const __half *>(crp + 0);
                        __half dmin = *reinterpret_cast<const __half *>(crp + 2);
                        float df = __half2float(d), dmf = __half2float(dmin);
                        const uint8_t *scales = crp + 4;
                        const uint8_t *qs = crp + 16;
                        float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                : (bb == 2u)       ? a2
                                                                   : a3;
#pragma unroll
                        for (uint32_t g = 0; g < 4; ++g) {
                            const uint8_t qb = qs[g * 32u + lane];
                            uint8_t sc0, mm0, sc1, mm1;
                            getScaleMin(static_cast<int>(g * 2 + 0), scales, &sc0,
                                        &mm0);
                            getScaleMin(static_cast<int>(g * 2 + 1), scales, &sc1,
                                        &mm1);
                            float dl0 = df * static_cast<float>(sc0);
                            float ml0 = dmf * static_cast<float>(mm0);
                            float dl1 = df * static_cast<float>(sc1);
                            float ml1 = dmf * static_cast<float>(mm1);
                            float qv0 = static_cast<float>(qb & 0xF);
                            float qv1 = static_cast<float>(qb >> 4);
                            const uint32_t c0 = (b + bb) * 256u + g * 64u + lane;
                            const uint32_t c1 = c0 + 32u;
                            slot = fmaf(fmaf(dl0, qv0, -ml0),
                                        (c0 < cols) ? x[c0] : 0.0f, slot);
                            slot = fmaf(fmaf(dl1, qv1, -ml1),
                                        (c1 < cols) ? x[c1] : 0.0f, slot);
                        }
                    }
                    acc += a0;
                    acc += a1;
                    acc += a2;
                    acc += a3;
                    rp += 4u * kQ4K_BYTES;
                }
                for (; b < vBlocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 2);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *scales = rp + 4;
                    const uint8_t *qs = rp + 16;
#pragma unroll
                    for (uint32_t g = 0; g < 4; ++g) {
                        const uint8_t qb = qs[g * 32u + lane];
                        uint8_t sc0, mm0, sc1, mm1;
                        getScaleMin(static_cast<int>(g * 2 + 0), scales, &sc0, &mm0);
                        getScaleMin(static_cast<int>(g * 2 + 1), scales, &sc1, &mm1);
                        float dl0 = df * static_cast<float>(sc0);
                        float ml0 = dmf * static_cast<float>(mm0);
                        float dl1 = df * static_cast<float>(sc1);
                        float ml1 = dmf * static_cast<float>(mm1);
                        float qv0 = static_cast<float>(qb & 0xF);
                        float qv1 = static_cast<float>(qb >> 4);
                        const uint32_t c0 = b * 256u + g * 64u + lane;
                        const uint32_t c1 = c0 + 32u;
                        acc = fmaf(fmaf(dl0, qv0, -ml0),
                                   (c0 < cols) ? x[c0] : 0.0f, acc);
                        acc = fmaf(fmaf(dl1, qv1, -ml1),
                                   (c1 < cols) ? x[c1] : 0.0f, acc);
                    }
                    rp += kQ4K_BYTES;
                }
#pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1) {
                    acc += __shfl_xor_sync(0xffffffffu, acc, off);
                }
                if (lane == 0)
                    outV[vrow] = biasV ? (acc + biasV[vrow]) : acc;
            }
        }

        // ---- Fused gate+up+silu for the Q8_K integer path (IQ2_S / IQ3_XXS /
        // IQ3_S): ONE shared kQuantizeQ8K of the activation, then both row
        // dots against the pre-quantized q8k in one warp.
        template<int TYPE, bool DP4A = false>
        __global__ void __launch_bounds__(256) kQGemvQ8KFusedGU(
                const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
                const uint8_t *__restrict__ x, float *__restrict__ out,
                uint32_t rows, uint32_t blocksPerRow, uint32_t rowBytes,
                bool grsm) {
            extern __shared__ uint8_t sGrid[];
            const uint8_t *gsm = nullptr;
            if (grsm) {
                // Stage the codebook grid for this TYPE (words of 4 B):
                // IQ2_S 1024x u64 = 2048, IQ2_XS 512x u64 = 1024,
                // IQ3_S 512x u32 = 512, IQ3_XXS 256x u32 = 256.
                const uint32_t nw = (TYPE == kTypeIQ2S)    ? 2048u
                                    : (TYPE == kTypeIQ2XS) ? 1024u
                                    : (TYPE == kTypeIQ3S)  ? 512u
                                                           : 256u;
                const uint32_t *src =
                        (TYPE == kTypeIQ2S)
                                ? reinterpret_cast<const uint32_t *>(c_iq2s_grid)
                        : (TYPE == kTypeIQ2XS)
                                ? reinterpret_cast<const uint32_t *>(c_iq2xs_grid)
                        : (TYPE == kTypeIQ3S) ? c_iq3s_grid
                                              : c_iq3xxs_grid;
                uint32_t *dst = reinterpret_cast<uint32_t *>(sGrid);
                const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
                const uint32_t nth = blockDim.x * blockDim.y;
                for (uint32_t i = tid; i < nw; i += nth) dst[i] = src[i];
                __syncthreads();
                gsm = sGrid;
            }
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rg = wg + static_cast<uint64_t>(row) * rowBytes;
            const uint8_t *ru = wu + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t lane = threadIdx.x;
            const uint32_t ib32 = lane >> 2u;// 0..7 within a block
            const uint32_t l = lane & 3u;    // 0..3
            const uint32_t bwBytes = (TYPE == kTypeIQ2S)      ? kIQ2S_BYTES
                                     : (TYPE == kTypeIQ2XS)   ? kIQ2XS_BYTES
                                     : (TYPE == kTypeIQ3_XXS) ? kIQ3XXS_BYTES
                                                              : kIQ3S_BYTES;
            float sumfG = 0.0f, sumfU = 0.0f;
            uint32_t b = 0;
            // 4-way block batching: 8 independent per-lane integer chains
            // (4 gate + 4 up) feed 8 int trees; lane 0 folds gate then up
            // serially in block order.
            for (; b + 4u <= blocksPerRow; b += 4u) {
                const uint8_t *gw0 = rg + static_cast<uint64_t>(b + 0u) * bwBytes;
                const uint8_t *gw1 = rg + static_cast<uint64_t>(b + 1u) * bwBytes;
                const uint8_t *gw2 = rg + static_cast<uint64_t>(b + 2u) * bwBytes;
                const uint8_t *gw3 = rg + static_cast<uint64_t>(b + 3u) * bwBytes;
                const uint8_t *uw0 = ru + static_cast<uint64_t>(b + 0u) * bwBytes;
                const uint8_t *uw1 = ru + static_cast<uint64_t>(b + 1u) * bwBytes;
                const uint8_t *uw2 = ru + static_cast<uint64_t>(b + 2u) * bwBytes;
                const uint8_t *uw3 = ru + static_cast<uint64_t>(b + 3u) * bwBytes;
                const uint8_t *ba0 = x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE;
                const uint8_t *ba1 = x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE;
                const uint8_t *ba2 = x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE;
                const uint8_t *ba3 = x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE;
                const int8_t *y0 = reinterpret_cast<const int8_t *>(ba0 + kQ8K_Q);
                const int8_t *y1 = reinterpret_cast<const int8_t *>(ba1 + kQ8K_Q);
                const int8_t *y2 = reinterpret_cast<const int8_t *>(ba2 + kQ8K_Q);
                const int8_t *y3 = reinterpret_cast<const int8_t *>(ba3 + kQ8K_Q);
                int gci0 = q8kLaneSumi<DP4A>(TYPE, gw0, y0, ib32, l, gsm);
                int gci1 = q8kLaneSumi<DP4A>(TYPE, gw1, y1, ib32, l, gsm);
                int gci2 = q8kLaneSumi<DP4A>(TYPE, gw2, y2, ib32, l, gsm);
                int gci3 = q8kLaneSumi<DP4A>(TYPE, gw3, y3, ib32, l, gsm);
                int uci0 = q8kLaneSumi<DP4A>(TYPE, uw0, y0, ib32, l, gsm);
                int uci1 = q8kLaneSumi<DP4A>(TYPE, uw1, y1, ib32, l, gsm);
                int uci2 = q8kLaneSumi<DP4A>(TYPE, uw2, y2, ib32, l, gsm);
                int uci3 = q8kLaneSumi<DP4A>(TYPE, uw3, y3, ib32, l, gsm);
                gci0 = q8kScaleMuli<TYPE>(gw0, gci0, ib32, l);
                gci1 = q8kScaleMuli<TYPE>(gw1, gci1, ib32, l);
                gci2 = q8kScaleMuli<TYPE>(gw2, gci2, ib32, l);
                gci3 = q8kScaleMuli<TYPE>(gw3, gci3, ib32, l);
                uci0 = q8kScaleMuli<TYPE>(uw0, uci0, ib32, l);
                uci1 = q8kScaleMuli<TYPE>(uw1, uci1, ib32, l);
                uci2 = q8kScaleMuli<TYPE>(uw2, uci2, ib32, l);
                uci3 = q8kScaleMuli<TYPE>(uw3, uci3, ib32, l);
                int gb0 = gci0 + __shfl_xor_sync(0xffffffffu, gci0, 1);
                gb0 += __shfl_xor_sync(0xffffffffu, gb0, 2);
                gb0 += __shfl_xor_sync(0xffffffffu, gb0, 4);
                gb0 += __shfl_xor_sync(0xffffffffu, gb0, 8);
                gb0 += __shfl_xor_sync(0xffffffffu, gb0, 16);
                int gb1 = gci1 + __shfl_xor_sync(0xffffffffu, gci1, 1);
                gb1 += __shfl_xor_sync(0xffffffffu, gb1, 2);
                gb1 += __shfl_xor_sync(0xffffffffu, gb1, 4);
                gb1 += __shfl_xor_sync(0xffffffffu, gb1, 8);
                gb1 += __shfl_xor_sync(0xffffffffu, gb1, 16);
                int gb2 = gci2 + __shfl_xor_sync(0xffffffffu, gci2, 1);
                gb2 += __shfl_xor_sync(0xffffffffu, gb2, 2);
                gb2 += __shfl_xor_sync(0xffffffffu, gb2, 4);
                gb2 += __shfl_xor_sync(0xffffffffu, gb2, 8);
                gb2 += __shfl_xor_sync(0xffffffffu, gb2, 16);
                int gb3 = gci3 + __shfl_xor_sync(0xffffffffu, gci3, 1);
                gb3 += __shfl_xor_sync(0xffffffffu, gb3, 2);
                gb3 += __shfl_xor_sync(0xffffffffu, gb3, 4);
                gb3 += __shfl_xor_sync(0xffffffffu, gb3, 8);
                gb3 += __shfl_xor_sync(0xffffffffu, gb3, 16);
                int ub0 = uci0 + __shfl_xor_sync(0xffffffffu, uci0, 1);
                ub0 += __shfl_xor_sync(0xffffffffu, ub0, 2);
                ub0 += __shfl_xor_sync(0xffffffffu, ub0, 4);
                ub0 += __shfl_xor_sync(0xffffffffu, ub0, 8);
                ub0 += __shfl_xor_sync(0xffffffffu, ub0, 16);
                int ub1 = uci1 + __shfl_xor_sync(0xffffffffu, uci1, 1);
                ub1 += __shfl_xor_sync(0xffffffffu, ub1, 2);
                ub1 += __shfl_xor_sync(0xffffffffu, ub1, 4);
                ub1 += __shfl_xor_sync(0xffffffffu, ub1, 8);
                ub1 += __shfl_xor_sync(0xffffffffu, ub1, 16);
                int ub2 = uci2 + __shfl_xor_sync(0xffffffffu, uci2, 1);
                ub2 += __shfl_xor_sync(0xffffffffu, ub2, 2);
                ub2 += __shfl_xor_sync(0xffffffffu, ub2, 4);
                ub2 += __shfl_xor_sync(0xffffffffu, ub2, 8);
                ub2 += __shfl_xor_sync(0xffffffffu, ub2, 16);
                int ub3 = uci3 + __shfl_xor_sync(0xffffffffu, uci3, 1);
                ub3 += __shfl_xor_sync(0xffffffffu, ub3, 2);
                ub3 += __shfl_xor_sync(0xffffffffu, ub3, 4);
                ub3 += __shfl_xor_sync(0xffffffffu, ub3, 8);
                ub3 += __shfl_xor_sync(0xffffffffu, ub3, 16);
                const float dg0 =
                        __half2float(*reinterpret_cast<const __half *>(gw0)) *
                        *reinterpret_cast<const float *>(ba0 + kQ8K_D);
                const float dg1 =
                        __half2float(*reinterpret_cast<const __half *>(gw1)) *
                        *reinterpret_cast<const float *>(ba1 + kQ8K_D);
                const float dg2 =
                        __half2float(*reinterpret_cast<const __half *>(gw2)) *
                        *reinterpret_cast<const float *>(ba2 + kQ8K_D);
                const float dg3 =
                        __half2float(*reinterpret_cast<const __half *>(gw3)) *
                        *reinterpret_cast<const float *>(ba3 + kQ8K_D);
                const float du0 =
                        __half2float(*reinterpret_cast<const __half *>(uw0)) *
                        *reinterpret_cast<const float *>(ba0 + kQ8K_D);
                const float du1 =
                        __half2float(*reinterpret_cast<const __half *>(uw1)) *
                        *reinterpret_cast<const float *>(ba1 + kQ8K_D);
                const float du2 =
                        __half2float(*reinterpret_cast<const __half *>(uw2)) *
                        *reinterpret_cast<const float *>(ba2 + kQ8K_D);
                const float du3 =
                        __half2float(*reinterpret_cast<const __half *>(uw3)) *
                        *reinterpret_cast<const float *>(ba3 + kQ8K_D);
                if (lane == 0) {
                    sumfG += dg0 * static_cast<float>(gb0);
                    sumfG += dg1 * static_cast<float>(gb1);
                    sumfG += dg2 * static_cast<float>(gb2);
                    sumfG += dg3 * static_cast<float>(gb3);
                    sumfU += du0 * static_cast<float>(ub0);
                    sumfU += du1 * static_cast<float>(ub1);
                    sumfU += du2 * static_cast<float>(ub2);
                    sumfU += du3 * static_cast<float>(ub3);
                }
            }
            // Remainder (blocksPerRow % 4) -- original single-block body.
            for (; b < blocksPerRow; ++b) {
                const uint8_t *blkWG = rg + static_cast<uint64_t>(b) * bwBytes;
                const uint8_t *blkWU = ru + static_cast<uint64_t>(b) * bwBytes;
                const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
                const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA + kQ8K_Q);
                const float yd = *reinterpret_cast<const float *>(blkA + kQ8K_D);
                int gci = q8kLaneSumi<DP4A>(TYPE, blkWG, y8, ib32, l, gsm);
                gci = q8kScaleMuli<TYPE>(blkWG, gci, ib32, l);
                int uci = q8kLaneSumi<DP4A>(TYPE, blkWU, y8, ib32, l, gsm);
                uci = q8kScaleMuli<TYPE>(blkWU, uci, ib32, l);
                int gbs = gci + __shfl_xor_sync(0xffffffffu, gci, 1);
                gbs += __shfl_xor_sync(0xffffffffu, gbs, 2);
                gbs += __shfl_xor_sync(0xffffffffu, gbs, 4);
                gbs += __shfl_xor_sync(0xffffffffu, gbs, 8);
                gbs += __shfl_xor_sync(0xffffffffu, gbs, 16);
                int ubs = uci + __shfl_xor_sync(0xffffffffu, uci, 1);
                ubs += __shfl_xor_sync(0xffffffffu, ubs, 2);
                ubs += __shfl_xor_sync(0xffffffffu, ubs, 4);
                ubs += __shfl_xor_sync(0xffffffffu, ubs, 8);
                ubs += __shfl_xor_sync(0xffffffffu, ubs, 16);
                if (lane == 0) {
                    const float dg =
                            __half2float(*reinterpret_cast<const __half *>(blkWG)) * yd;
                    const float du =
                            __half2float(*reinterpret_cast<const __half *>(blkWU)) * yd;
                    sumfG += dg * static_cast<float>(gbs);
                    sumfU += du * static_cast<float>(ubs);
                }
            }
            const float scale = (TYPE == kTypeIQ2S)      ? 0.125f
                                : (TYPE == kTypeIQ2XS)   ? 0.125f
                                : (TYPE == kTypeIQ3_XXS) ? 0.25f
                                                         : 1.0f;
            if (lane == 0) {
                const float g = scale * sumfG;
                const float u = scale * sumfU;
                out[row] = (g / (1.0f + __expf(-g))) * u;
            }
        }

        // ------------------------------------------------------------------
        // Device-side FP16 dequant (per-layer streaming for prefill GEMMs).
        //
        // Instead of persisting a ~2 B/element FP16 twin of EVERY weight in
        // VRAM (~14 GB for a 7B), dequantize ONE matrix on-device into the
        // reusable wF16_ scratch, run its GEMM(s), then overwrite it with the
        // next matrix.  Decode (seqLen==1) never touches this path (the
        // quantized kQGemv runs straight from the compact blocks), so the
        // scratch only ever holds one matrix at a time.
        //
        // The kernel mirrors GGMLDequantize::dequantizeBlock for each type.
        // One warp per 32-column strip of a row: lane l owns columns
        // [stripStart + l*8, stripStart + l*8 + 8) of a 256-element block, so
        // an even strip boundary keeps each lane inside one (ib32,...,l) group.
        // pad (cols not a multiple of 256) is zero-filled.
        template<int TYPE>
        __global__ void kDequantF16(const uint8_t *__restrict__ w,
                                    __half *__restrict__ out, uint32_t rows,
                                    uint32_t cols, uint32_t blocksPerRow,
                                    uint32_t rowBytes) {
            const uint32_t lane = threadIdx.x;
            const uint32_t e0 = lane * 8u;
            const uint32_t r = blockIdx.x;
            if (r >= rows || cols == 0) return;
            const uint8_t *rp = w + static_cast<uint64_t>(r) * rowBytes;
            __half *orow = out + static_cast<uint64_t>(r) * cols;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                if (TYPE == kTypeQ5_0 || TYPE == kTypeQ8_0 ||
                    TYPE == kTypeIQ4NL) {
                    // 32-wide legacy block: only lanes 0..3 own valid columns.
                    const uint32_t base = b * 32u;
                    if (e0 < 32u) {
                        if (TYPE == kTypeQ8_0) {
                            __half d = *reinterpret_cast<const __half *>(rp);
                            const int8_t *qs =
                                    reinterpret_cast<const int8_t *>(rp + 2);
                            float dl = __half2float(d);
#pragma unroll
                            for (uint32_t k = 0; k < 8; ++k) {
                                const uint32_t col = base + e0 + k;
                                if (col < cols)
                                    orow[col] = __float2half(
                                            dl * static_cast<float>(qs[e0 + k]));
                            }
                        } else if (TYPE == kTypeIQ4NL) {
                            // IQ4_NL (4.5 bpw): d(fp16,2) + qs[16].  Per-column
                            // byte j = col&15; nib = (col<16) ? qs[j]&0xF :
                            // qs[j]>>4; weight = d*kvalues_iq4nl[nib].  Matches
                            // CPU dequantizeIQ4_NLBlock.
                            __half d = *reinterpret_cast<const __half *>(rp);
                            float dl = __half2float(d);
                            const uint8_t *qs = rp + 2;
#pragma unroll
                            for (uint32_t k = 0; k < 8; ++k) {
                                const uint32_t col = base + e0 + k;
                                if (col >= cols) continue;
                                const uint32_t j = col & 15u;
                                const uint8_t nib =
                                        (qs[j] >> (4u * ((col >> 4) & 1u))) &
                                        0xFu;
                                orow[col] = __float2half(
                                        dl * static_cast<float>(
                                                     c_kvalues_iq4nl[nib]));
                            }
                        } else {// Q5_0
                            __half d = *reinterpret_cast<const __half *>(rp);
                            float dl = __half2float(d);
                            uint32_t qh;
                            std::memcpy(&qh, rp + 2, sizeof(uint32_t));
                            const uint8_t *qs = rp + 6;
#pragma unroll
                            for (uint32_t k = 0; k < 8; ++k) {
                                const uint32_t col = base + e0 + k;
                                if (col >= cols) continue;
                                const uint32_t j = col & 15u;
                                const uint8_t low =
                                        (qs[j] >> (4u * ((col >> 4) & 1u))) &
                                        0xFu;
                                const uint32_t hb =
                                        (col & 16u) ? (j + 16u) : j;
                                const uint8_t high =
                                        static_cast<uint8_t>((qh >> hb) & 1u);
                                const int val =
                                        static_cast<int>(low | (high << 4)) - 16;
                                orow[col] = __float2half(
                                        dl * static_cast<float>(val));
                            }
                        }
                    }
                    rp += (TYPE == kTypeQ5_0)
                                  ? kQ5_0_BYTES
                          : (TYPE == kTypeQ8_0) ? kQ8_0_BYTES
                                                : kIQ4NL_BYTES;
                    continue;
                }
                const uint32_t base = b * 256u;
                // The final partial block's pad columns get zero fp16.
                if (base + e0 >= cols) {
                    rp += (TYPE == kTypeQ2K)      ? kQ2K_BYTES
                          : (TYPE == kTypeQ3K)    ? kQ3K_BYTES
                          : (TYPE == kTypeQ4K)    ? kQ4K_BYTES
                          : (TYPE == kTypeQ6K)    ? kQ6K_BYTES
                          : (TYPE == kTypeIQ2XS)  ? kIQ2XS_BYTES
                          : (TYPE == kTypeIQ2XXS) ? kIQ2XXS_BYTES
                          : (TYPE == kTypeIQ4XS)  ? kIQ4XS_BYTES
                          : (TYPE == kTypeIQ3S)   ? kIQ3S_BYTES
                                                  : kQ5K_BYTES;
                    continue;
                }
                if (TYPE == kTypeQ2K) {
                    __half d = *reinterpret_cast<const __half *>(rp + 80);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 82);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *q = rp + 16;
                    const uint8_t *sc = rp;
                    const uint32_t half = e0 / 128u;
                    const uint32_t jj = (e0 % 128u) / 32u;
                    const uint32_t sub = (e0 % 32u) / 16u;
                    const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                    float dl = df * static_cast<float>(scv & 0xF);
                    float ml = dmf * static_cast<float>(scv >> 4);
#pragma unroll
                    for (uint32_t k = 0; k < 8; ++k) {
                        // Per-column byte: the 8 lane columns stay within one
                        // 16-column sub group (e0%16 is 0 or 8), so only the
                        // low 4 bits of the q index advance with k.
                        const uint8_t qb = q[half * 32u + sub * 16u + ((e0 + k) & 15u)];
                        float qv = static_cast<float>(static_cast<int8_t>(
                                ((qb >> (jj * 2u)) & 3)));
                        const uint32_t col = base + e0 + k;
                        if (col < cols) orow[col] = __float2half(fmaf(dl, qv, -ml));
                    }
                } else if (TYPE == kTypeQ3K) {
                    __half d = *reinterpret_cast<const __half *>(rp + 108);
                    float df = __half2float(d);
                    const uint8_t *hm = rp;
                    const uint8_t *q = rp + 32;
                    const uint8_t *scales = rp + 96;
                    uint32_t aux[4];
                    std::memcpy(&aux[0], scales, 12);
                    uint32_t tmp = aux[2];
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
                    const uint32_t half = e0 / 128u;
                    const uint32_t jj = (e0 % 128u) / 32u;
                    const uint32_t sub = (e0 % 32u) / 16u;
                    float dl = df * static_cast<float>(
                                            sc16[half * 8u + jj * 2u + sub] - 32);
                    // CPU reference: hm is ONLY 32 bytes (QK_K/8) and does NOT
                    // advance per 128-half -- the half distinction is carried by
                    // the mask bit 1u<<(half*4+jj) (m starts at 1 and shifts once
                    // per j, so half 1 uses bits 4..7).  The byte index within
                    // the block is sub*16 + lk (hm[l] for sub 0, hm[l+16] for
                    // sub 1), matching kQGemv's hm[lane].  Advancing hm by
                    // half*32 overreads into the q array for half 1.
                    const uint32_t maskBit = 1u << (half * 4u + jj);
#pragma unroll
                    for (uint32_t k = 0; k < 8; ++k) {
                        const uint32_t lk = (e0 + k) & 15u;
                        const uint8_t qb = q[half * 32u + sub * 16u + lk];
                        const uint8_t hmb = hm[sub * 16u + lk];
                        float qv = static_cast<float>(static_cast<int8_t>(
                                ((qb >> (jj * 2u)) & 3) -
                                ((hmb & maskBit) ? 0 : 4)));
                        const uint32_t col = base + e0 + k;
                        if (col < cols) orow[col] = __float2half(dl * qv);
                    }
                } else if (TYPE == kTypeQ4K) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 2);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *scales = rp + 4;
                    const uint8_t *qs = rp + 16;
                    const uint32_t g = e0 / 64u;
                    const uint32_t sub = (e0 / 32u) & 1u;
                    const uint32_t jj0 = e0 & 31u;
                    auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                          uint8_t *m_out) {
                        if (j < 4) {
                            *d_out = q[j] & 63;
                            *m_out = q[j + 4] & 63;
                        } else {
                            *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                            *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                        }
                    };
                    uint8_t sc, m;
                    getScaleMin(static_cast<int>(2u * g + sub), scales, &sc, &m);
                    float dl = df * static_cast<float>(sc);
                    float ml = dmf * static_cast<float>(m);
                    // CPU reference: per-column byte l = jj0+k advances with the
                    // column (0..31 within the 32-byte qs group for chunk g);
                    // sub==0 reads the low nibble, sub==1 the high nibble.
#pragma unroll
                    for (uint32_t k = 0; k < 8; ++k) {
                        const uint8_t qb = qs[g * 32u + jj0 + k];
                        const uint8_t low = (sub == 0) ? (qb & 0xF) : (qb >> 4);
                        float wv = fmaf(dl, static_cast<float>(low), -ml);
                        const uint32_t col = base + e0 + k;
                        if (col < cols) orow[col] = __float2half(wv);
                    }
                } else if (TYPE == kTypeQ6K) {
                    __half d = *reinterpret_cast<const __half *>(rp + 208);
                    float df = __half2float(d);
                    const uint8_t *ql = rp;
                    const uint8_t *qh = rp + 128;
                    const int8_t *sc = reinterpret_cast<const int8_t *>(rp + 192);
                    const uint32_t half = e0 / 128u;
                    const uint32_t sub = (e0 % 128u) / 32u;
                    const uint32_t l = e0 % 32u;
                    const uint32_t g16 = l / 16u;
                    const uint32_t qlBase = half * 64u + ((sub & 1u) ? 32u : 0u);
                    const uint32_t qhBase = half * 32u;
                    const uint32_t shift = sub * 2u;
                    const uint32_t hiNib = (sub & 2u) ? 4u : 0u;
                    float dl = df *
                               static_cast<float>(sc[half * 8u + sub * 2u + g16]);
#pragma unroll
                    for (uint32_t k = 0; k < 8; ++k) {
                        uint32_t qb = (hiNib) ? (ql[qlBase + l + k] >> 4)
                                              : (ql[qlBase + l + k] & 0xF);
                        qb |= ((qh[qhBase + l + k] >> shift) & 3u) << 4u;
                        float qv = static_cast<float>(static_cast<int>(qb) - 32);
                        const uint32_t col = base + e0 + k;
                        if (col < cols) orow[col] = __float2half(dl * qv);
                    }
                } else if (TYPE == kTypeIQ2XS) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    const uint16_t *qs16 =
                            reinterpret_cast<const uint16_t *>(rp + 2);
                    const uint8_t *scales = rp + 66;
                    const uint32_t ib32 = e0 >> 5u;
                    const uint32_t l = (e0 >> 3u) & 3u;
                    float db0 = df * (0.5f + static_cast<float>(scales[ib32] & 0xf)) *
                                0.25f;
                    float db1 = df * (0.5f + static_cast<float>(scales[ib32] >> 4)) *
                                0.25f;
                    // CPU reference: dl = db[l/2] (l=0,1 -> db0; l=2,3 -> db1).
                    float dl = (l >= 2u) ? db1 : db0;
                    uint16_t qval = qs16[ib32 * 4u + l];
                    uint16_t gridIdx = qval & 0x1FFu;
                    uint8_t signIdx = static_cast<uint8_t>(qval >> 9);
                    // ggml semantics: grid entry packs 8 byte values; byte j
                    // is grid[j]. sign flip from c_ksigns_iq2xs / c_kmask_iq2xs.
                    const uint8_t *grid =
                            reinterpret_cast<const uint8_t *>(&c_iq2xs_grid[gridIdx]);
                    uint8_t signs = c_ksigns_iq2xs[signIdx];
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        float w = dl * static_cast<float>(grid[j]) *
                                  ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t col = base + e0 + j;
                        if (col < cols) orow[col] = __float2half(w);
                    }
                } else if (TYPE == kTypeIQ2XXS) {
                    // IQ2_XXS block (66 B): d(2) + qs[32 x uint16] (64 B).
                    // Same per-(ib32,l) math as the kQGemv branch -- grid idx
                    // from word0 byte l, sign idx + scale from word1.
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    const uint8_t *qsv = rp + 2;
                    const uint32_t ib32 = e0 >> 5u;
                    const uint32_t l = (e0 >> 3u) & 3u;
                    uint32_t w0, w1;
                    std::memcpy(&w0, qsv + 4u * ib32, sizeof(uint32_t));
                    std::memcpy(&w1, qsv + 4u * ib32 + 4u, sizeof(uint32_t));
                    float db =
                            df * (0.5f + static_cast<float>(w1 >> 28)) * 0.25f;
                    const uint32_t gridIdx = (w0 >> (8u * l)) & 0xFFu;
                    const uint8_t signs =
                            c_ksigns_iq2xs[(w1 >> (7u * l)) & 127u];
                    const uint8_t *grid = reinterpret_cast<const uint8_t *>(
                            &c_iq2xxs_grid[gridIdx]);
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        float wgt = db * static_cast<float>(grid[j]) *
                                    ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t col = base + e0 + j;
                        if (col < cols) orow[col] = __float2half(wgt);
                    }
                } else if (TYPE == kTypeIQ4XS) {
                    // IQ4_XS block (136 B): d(2) + scales_h(2) + scales_l(4) +
                    // qs[128].  Per 32-weight sub-block ib (lane owns its
                    // 8 columns within one sub-block):
                    //   ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf)
                    //        | (((scales_h >> 2*ib) & 3) << 4);  dl = d*(ls-32)
                    //   qs[ib*16 + j] byte -> col ib*32+j (low nibble) or
                    //   col ib*32+16+j (high nibble) of kvalues_iq4nl.
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    uint16_t scales_h;
                    std::memcpy(&scales_h, rp + 2, sizeof(uint16_t));
                    const uint8_t *scales_l = rp + 4;
                    const uint8_t *qsv = rp + 8;
                    const uint32_t ib = e0 >> 5u;
                    const uint32_t l = (e0 >> 3u) & 3u;
                    const int ls =
                            static_cast<int>((scales_l[ib / 2u] >>
                                              (4u * (ib % 2u))) &
                                             0xfu) |
                            (static_cast<int>((scales_h >> (2u * ib)) & 0x3u)
                             << 4);
                    float dl = df * static_cast<float>(ls - 32);
                    const uint8_t *qb = qsv + ib * 16u + l * 4u;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        const uint32_t jc = l * 8u + j;
                        const uint32_t byteIdx = jc & 15u;
                        const uint32_t nib = (jc < 16u)
                                                     ? (qb[byteIdx] & 0xfu)
                                                     : (qb[byteIdx] >> 4);
                        float wgt = dl * static_cast<float>(c_kvalues_iq4nl[nib]);
                        const uint32_t col = base + e0 + j;
                        if (col < cols) orow[col] = __float2half(wgt);
                    }
                } else if (TYPE == kTypeIQ2S) {
                    // IQ2_S block (82 B): d(2) + qs[64] + qh[8] + scales[8].
                    // Per (ib32, l): dl = db[l/2]; gridIdx = qs[ib32*4+l] |
                    // (qh[ib32] << (8-2l) & 0x300); sign = qs[32+ib32*4+l].
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    const uint8_t *qsv = rp + 2;
                    const uint8_t *qh = rp + 66;
                    const uint8_t *scales = rp + 74;
                    const uint32_t ib32 = e0 >> 5u;
                    const uint32_t l = (e0 >> 3u) & 3u;
                    float db0 = df *
                                (0.5f + static_cast<float>(scales[ib32] & 0xf)) *
                                0.25f;
                    float db1 = df *
                                (0.5f + static_cast<float>(scales[ib32] >> 4)) *
                                0.25f;
                    float dl = (l >= 2u) ? db1 : db0;
                    uint16_t gridIdx = static_cast<uint16_t>(
                            qsv[ib32 * 4u + l] |
                            ((qh[ib32] << (8 - 2u * l)) & 0x300));
                    const uint8_t *grid =
                            reinterpret_cast<const uint8_t *>(&c_iq2s_grid[gridIdx]);
                    const uint8_t signs = qsv[32u + ib32 * 4u + l];
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        float w = dl * static_cast<float>(grid[j]) *
                                  ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t col = base + e0 + j;
                        if (col < cols) orow[col] = __float2half(w);
                    }
                } else if (TYPE == kTypeIQ3_XXS) {
                    // IQ3_XXS block (98 B): d(2) + qs[96]; qs[64..95] is the
                    // scales+signs area.  Per (ib32, l):
                    //   aux32 = u32(qs+64+4*ib32); db = d*(0.5+(aux32>>28))*0.5
                    //   signs = c_ksigns_iq2xs[(aux32>>7l)&127]
                    //   grid1 = qs[8*ib32+2l+0], grid2 = qs[8*ib32+2l+1]
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    const uint8_t *qsv = rp + 2;
                    const uint32_t ib32 = e0 >> 5u;
                    const uint32_t l = (e0 >> 3u) & 3u;
                    uint32_t aux32;
                    std::memcpy(&aux32, qsv + 64u + 4u * ib32,
                                sizeof(uint32_t));
                    float db = df *
                               (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                    const uint8_t signs =
                            c_ksigns_iq2xs[(aux32 >> (7u * l)) & 127u];
                    const uint32_t g1 = c_iq3xxs_grid[qsv[8u * ib32 + 2u * l]];
                    const uint32_t g2 =
                            c_iq3xxs_grid[qsv[8u * ib32 + 2u * l + 1u]];
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        uint32_t gval = (j < 4u) ? ((g1 >> (8u * j)) & 0xFFu)
                                                 : ((g2 >> (8u * (j - 4u))) &
                                                    0xFFu);
                        float w = db * static_cast<float>(gval) *
                                  ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t col = base + e0 + j;
                        if (col < cols) orow[col] = __float2half(w);
                    }
                } else if (TYPE == kTypeIQ3S) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    float df = __half2float(d);
                    const uint8_t *qs = rp + 2;
                    const uint8_t *qh = rp + 66;
                    const uint8_t *signs = rp + 74;
                    const uint8_t *scales = rp + 106;
                    const uint32_t ib32 = e0 >> 5u;
                    const uint32_t half = ib32 & 1u;
                    const uint32_t pair = ib32 >> 1u;
                    const uint32_t l = (e0 >> 3u) & 3u;
                    const uint32_t qhIdx = pair * 2u + half;
                    const uint32_t qsOff = pair * 16u + half * 8u;
                    const uint32_t signOff = pair * 8u + half * 4u;
                    float db = df *
                               (1.0f +
                                2.0f * static_cast<float>(
                                               (half == 0) ? (scales[pair] & 0xf)
                                                           : (scales[pair] >> 4)));
                    const uint16_t gridIdx1 = static_cast<uint16_t>(
                            qs[qsOff + 2u * l] |
                            ((qh[qhIdx] << (8 - 2u * l)) & 256));
                    const uint16_t gridIdx2 = static_cast<uint16_t>(
                            qs[qsOff + 2u * l + 1u] |
                            ((qh[qhIdx] << (7 - 2u * l)) & 256));
                    const uint32_t g1 = c_iq3s_grid[gridIdx1];
                    const uint32_t g2 = c_iq3s_grid[gridIdx2];
                    const uint8_t sbyte = signs[signOff + l];
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        uint32_t gval = (j < 4u) ? ((g1 >> (8u * j)) & 0xFFu)
                                                 : ((g2 >> (8u * (j - 4u))) &
                                                    0xFFu);
                        float w = db * static_cast<float>(gval) *
                                  ((sbyte & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                        const uint32_t col = base + e0 + j;
                        if (col < cols) orow[col] = __float2half(w);
                    }
                } else if (TYPE == kTypeQ5K) {
                    __half d = *reinterpret_cast<const __half *>(rp + 0);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 2);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *scales = rp + 4;
                    const uint8_t *qh = rp + 16;
                    const uint8_t *qs = rp + 48;
                    const uint32_t g = e0 / 64u;
                    const uint32_t sub = (e0 / 32u) & 1u;
                    const uint32_t jj0 = e0 & 31u;
                    auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                          uint8_t *m_out) {
                        if (j < 4) {
                            *d_out = q[j] & 63;
                            *m_out = q[j + 4] & 63;
                        } else {
                            *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                            *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                        }
                    };
                    uint8_t sc, m;
                    getScaleMin(static_cast<int>(2u * g + sub), scales, &sc, &m);
                    float dl = df * static_cast<float>(sc);
                    float ml = dmf * static_cast<float>(m);
                    const uint8_t u = (sub == 0) ? static_cast<uint8_t>(1u << (2u * g))
                                                 : static_cast<uint8_t>(2u << (2u * g));
                    const uint8_t *qsp = qs + g * 32u;
#pragma unroll
                    for (uint32_t kk = 0; kk < 8; ++kk) {
                        uint8_t q5 = (sub == 0)
                                             ? static_cast<uint8_t>(
                                                       (qsp[jj0 + kk] & 0xF) |
                                                       ((qh[jj0 + kk] & u) ? 16 : 0))
                                             : static_cast<uint8_t>(
                                                       (qsp[jj0 + kk] >> 4) |
                                                       ((qh[jj0 + kk] & u) ? 16 : 0));
                        float wv = dl * static_cast<float>(q5) - ml;
                        const uint32_t col = base + e0 + kk;
                        if (col < cols) orow[col] = __float2half(wv);
                    }
                } else {
                    // Unknown type: zero the strip.
#pragma unroll
                    for (uint32_t k = 0; k < 8; ++k) {
                        const uint32_t col = base + e0 + k;
                        if (col < cols) orow[col] = __float2half(0.0f);
                    }
                }
                static constexpr uint32_t SZ = (TYPE == kTypeQ5_0 || TYPE == kTypeQ8_0 ||
                                                TYPE == kTypeIQ4NL)
                                                       ? (TYPE == kTypeQ5_0
                                                                  ? kQ5_0_BYTES
                                                          : TYPE == kTypeQ8_0
                                                                  ? kQ8_0_BYTES
                                                                  : kIQ4NL_BYTES)
                                                       : (TYPE == kTypeQ2K
                                                                  ? kQ2K_BYTES
                                                          : TYPE == kTypeQ3K
                                                                  ? kQ3K_BYTES
                                                          : TYPE == kTypeQ4K
                                                                  ? kQ4K_BYTES
                                                          : TYPE == kTypeQ6K
                                                                  ? kQ6K_BYTES
                                                          : TYPE == kTypeIQ2XS
                                                                  ? kIQ2XS_BYTES
                                                          : TYPE == kTypeIQ3_XXS
                                                                  ? kIQ3XXS_BYTES
                                                          : TYPE == kTypeIQ3S
                                                                  ? kIQ3S_BYTES
                                                          : TYPE == kTypeIQ2S
                                                                  ? kIQ2S_BYTES
                                                          : TYPE == kTypeIQ2XXS
                                                                  ? kIQ2XXS_BYTES
                                                          : TYPE == kTypeIQ4XS
                                                                  ? kIQ4XS_BYTES
                                                                  : kQ5K_BYTES);
                rp += SZ;
            }
        }

        // Forward declaration: defined after launchQGemv (qwen35 section).
        __global__ void kQ35F32Gemv(const float *x, const float *w, float *out,
                                    uint32_t rows, uint32_t cols);

        // Batched prefill GEMM: out[seqLen][rows] = X[seqLen][cols] @ W^T.
        // Quantizes every token's activation to Q8_K once (into the caller's
        // batched scratch) and runs kQGemmKxQ8K for the K-quant family, reading
        // each weight block ONCE per 4-token group (vs once per token in the
        // per-token launchQGemv loop).  Bit-exact with the per-token Q8_K
        // launchQGemv path, so it is a drop-in for the qwen35 prefill that must
        // stay numerically identical to decode.  Returns false (leaving `out`
        // untouched) when the type/shape is not covered, so the caller can fall
        // back to the exact per-token launchQGemv loop.
        bool launchQGemm(int type, const void *wq, const float *x, float *out,
                         uint32_t seqLen, uint32_t rows, uint32_t cols,
                         uint32_t rowBytes, uint32_t blocksPerRow,
                         uint8_t *q8kBatch, uint64_t q8kBatchBytes) {
            if (seqLen == 0 || rows == 0 || cols == 0) return true;
            // Which kernel template + block bytes for this type.
            const uint32_t kBlkBytes =
                    (type == kTypeQ4K)     ? kQ4K_BYTES
                    : (type == kTypeQ5K)   ? kQ5K_BYTES
                    : (type == kTypeIQ4XS) ? kIQ4XS_BYTES
                    : (type == kTypeQ6K)   ? kQ6K_BYTES
                                           : 0u;
            if (kBlkBytes == 0) return false;
            // The MMA-style kxQ8KBlockDot kernels write the Q6_K result on all
            // 8 sub-lanes (shuffle tree); Q4_K/Q5_K/IQ4_XS write lane 0 only.
            // Both are handled inside kQGemmKxQ8K, so no shape constraint here.
            const uint64_t actStride =
                    static_cast<uint64_t>(blocksPerRow) * kQ8K_STRIDE;
            const uint64_t need = static_cast<uint64_t>(seqLen) * actStride;
            if (q8kBatch == nullptr || q8kBatchBytes < need) return false;
            // Quantize all tokens' activations (one block-row per token).
            dim3 qgrid(blocksPerRow, seqLen);
            kQuantizeQ8K_Batch<<<qgrid, 256, 0, g_stream>>>(
                    x, q8kBatch, cols, seqLen,
                    static_cast<uint32_t>(actStride));
            cudaError_t qe = cudaGetLastError();
            if (qe != cudaSuccess) {
                std::fprintf(stderr, "kQuantizeQ8K_Batch: %s\n",
                             cudaGetErrorString(qe));
                return false;
            }
            dim3 block(32, 8);
            // grid.x covers rows in 8-row blocks; grid.y steps 4 tokens/group.
            dim3 grid((rows + 7) / 8, (seqLen + 3) / 4);
            const uint32_t aStride = static_cast<uint32_t>(actStride);
            if (type == kTypeQ4K) {
                // int8 tensor-core (mma.s8.m8n8k16) affine Q4_K GEMM reading the
                // RAW weights (kMmqGemmQ4K_AffineRaw, zero extra VRAM).  Opt-in
                // (TINYCODER_MMQ_Q4K=1), gated on sm_75+ and a real prefill
                // batch; NOT bit-exact with kQGemmKxQ8K<Q4_K> (accepted via the
                // batch-vs-sequential coherence contract).
                if (seqLen >= 64 && mmqQ4k() && mmqS8Ok()) {
                    const uint32_t n8Tiles = (rows + 7u) / 8u;
                    dim3 mgrid(n8Tiles, (seqLen + 63u) / 64u);
                    kMmqGemmQ4K_AffineRaw<<<mgrid, 256, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq),
                            reinterpret_cast<const int8_t *>(q8kBatch), out,
                            static_cast<int>(seqLen), static_cast<int>(rows),
                            static_cast<int>(cols), static_cast<int>(rowBytes),
                            static_cast<int>((cols + 15u) / 16u),
                            static_cast<int>(aStride));
                    cudaError_t me = cudaGetLastError();
                    if (me == cudaSuccess) return true;
                    // fall through to the integer GEMM on any launch error
                }
                kQGemmKxQ8K<kTypeQ4K><<<grid, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8kBatch, out, rows,
                        blocksPerRow, rowBytes, aStride, seqLen);
            } else if (type == kTypeQ5K) {
                kQGemmKxQ8K<kTypeQ5K><<<grid, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8kBatch, out, rows,
                        blocksPerRow, rowBytes, aStride, seqLen);
            } else if (type == kTypeQ6K) {
                kQGemmKxQ8K<kTypeQ6K><<<grid, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8kBatch, out, rows,
                        blocksPerRow, rowBytes, aStride, seqLen);
            } else {
                kQGemmKxQ8K<kTypeIQ4XS><<<grid, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8kBatch, out, rows,
                        blocksPerRow, rowBytes, aStride, seqLen);
            }
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                std::fprintf(stderr, "kQGemmKxQ8K: %s\n", cudaGetErrorString(e));
                return false;
            }
            return true;
        }

        void launchQGemv(int type, const void *wq, const float *x, float *out,
                         uint32_t rows, uint32_t cols, uint32_t rowBytes,
                         uint32_t blocksPerRow, uint8_t *&q8k,
                         uint64_t &q8kBytes) {
            // x is read directly from global inside the kernel (coalesced, and
            // tiny enough to stay L1/L2-resident), so no shared memory is used.
            // cols must be a multiple of 32.  For this model hiddenSize=1536,
            // intermediateSize=8960, qLen=1536 and kvLen=256 are all multiples;
            // the Model layer pads any non-multiple up when building the desc.
            // 8 warps per block (256 threads) => 8 rows per block.
            dim3 block(32, 8);
            uint32_t nBlocks = (rows + 7) / 8;
            if (std::getenv("TINYCODER_TRACE_GEMV") != nullptr) {
                std::fprintf(stderr, "[gemv trace] type=%d q8k=%d rows=%u cols=%u "
                                     "blocksPerRow=%u rowBytes=%u nBlocks=%u\n",
                             type, (type == kTypeIQ2S || type == kTypeIQ3_XXS || type == kTypeIQ3S),
                             rows, cols, blocksPerRow, rowBytes, nBlocks);
            }
            // Rows+nBlocks combined line (decode reads as (rows,nBlocks)).
            if (std::getenv("TINYCODER_TRACE_GEMV") != nullptr) {
                std::fprintf(stderr, "[gemv rnb] r=%u nB=%u c=%u\n", rows, nBlocks,
                             cols);
            }
            // P1 exp2 (2026-10-06): EXACT-columns fast path.  When
            // blocksPerRow*256 == cols the per-element `(cidx < cols)` guard
            // in the K-quant GEMVs is provably dead (all default-model
            // matrices: H=1536, I=8960, KV=256); the EXACT=true
            // instantiations drop it -- BIT-IDENTICAL arithmetic (max cidx
            // = bpr*256-1 = cols-1) with ~2 fewer SASS ops per weight.
            // MEASURED 2026-10-06 (Release, AVX2, interleaved + ABBA,
            // n-gen 8, reps 8): guarded 165.9/166.0/166.0/165.9/165.7/165.8
            // (mean 165.9) vs EXACT 170.1/170.2/170.1/169.2/170.2/170.3
            // (mean 170.0) -- a solid +2.5% tg8, position-independent.  The
            // gain is real (the guarded kernel's dead compare/select was NOT
            // free once the FFN is issue-bound) and costs nothing, so it is
            // DEFAULT ON.  TINYCODER_EXACT_COLS=0 restores the guarded
            // kernels for A/B.  Non-multiple-of-256 models keep the guard.
            const char *exactE = std::getenv("TINYCODER_EXACT_COLS");
            const bool exactCols =
                    (exactE == nullptr || std::atoi(exactE) != 0) &&
                    (static_cast<uint64_t>(blocksPerRow) * 256u ==
                     static_cast<uint64_t>(cols));
            if (type == kTypeIQ2S || (type == kTypeIQ2XS && iq2xsInt()) ||
                type == kTypeIQ3_XXS || type == kTypeIQ3S ||
                type == kTypeQ5K || type == kTypeQ4K || type == kTypeQ6K ||
                type == kTypeIQ4XS) {
                // The float dequant-dot differs by up to ~5%, flipping near-tie
                // argmax at knife-edge rows -- so mirror the Q8_K integer path
                // here: quantize x once into the reusable q8k_ scratch, then
                // run the integer GEMV.
                const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                                      static_cast<uint64_t>(kQ8K_STRIDE);
                if (q8k == nullptr || q8kBytes < need) {
                    if (q8k) cudaFree(q8k);
                    q8k = nullptr;
                    q8kBytes = 0;
                    cudaError_t e = cudaMalloc(&q8k, need);
                    if (e != cudaSuccess) {
                        std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                                     cudaGetErrorString(e));
                        cudaMemsetAsync(out, 0, sizeof(float) * rows, g_stream);
                        return;
                    }
                    q8kBytes = need;
                }
                kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(x, q8k, cols);
                const bool d4 = iqDp4a();
                const bool grsm = iqGridSmem();
                const bool iq4 = iq4xw() && (rows & 3u) == 0u;
#define TC_IQ_GEMV(T, GB)                                               \
    do {                                                                \
        if (iq4) {                                                      \
            const uint32_t nB4 = (rows / 4u + 7u) / 8u;                 \
            if (d4)                                                     \
                kQGemvQ8K_4xW<T, true>                                  \
                        <<<nB4, block, grsm ? (GB) : 0u, g_stream>>>(   \
                                static_cast<const uint8_t *>(wq), q8k,  \
                                out, rows, blocksPerRow, rowBytes,      \
                                grsm);                                  \
            else                                                        \
                kQGemvQ8K_4xW<T, false>                                 \
                        <<<nB4, block, grsm ? (GB) : 0u, g_stream>>>(   \
                                static_cast<const uint8_t *>(wq), q8k,  \
                                out, rows, blocksPerRow, rowBytes,      \
                                grsm);                                  \
        } else if (d4)                                                  \
            kQGemvQ8K<T, true>                                          \
                    <<<nBlocks, block, grsm ? (GB) : 0u, g_stream>>>(   \
                            static_cast<const uint8_t *>(wq), q8k, out, \
                            rows, blocksPerRow, rowBytes, grsm);        \
        else                                                            \
            kQGemvQ8K<T, false>                                         \
                    <<<nBlocks, block, grsm ? (GB) : 0u, g_stream>>>(   \
                            static_cast<const uint8_t *>(wq), q8k, out, \
                            rows, blocksPerRow, rowBytes, grsm);        \
    } while (0)
                if (type == kTypeIQ2S) {
                    TC_IQ_GEMV(kTypeIQ2S, 8192u);
                } else if (type == kTypeIQ2XS) {
                    TC_IQ_GEMV(kTypeIQ2XS, 4096u);
                } else if (type == kTypeIQ3_XXS) {
                    TC_IQ_GEMV(kTypeIQ3_XXS, 1024u);
                } else if (type == kTypeIQ3S) {
                    TC_IQ_GEMV(kTypeIQ3S, 2048u);
                } else if (type == kTypeQ5K) {
                    // Q5_K: the CPU reference (matMulVecBatchQ5K_SIMD ->
                    // matMulVecBatchQ5K_Q8K_AVX2) quantizes the fp32 activation
                    // to Q8_K and dots it against the 5-bit weights with exact
                    // integer math; the float dequant-dot of kQGemv<kTypeQ5K>
                    // diverged 0.5-5% per row (Q5KKernelParityWithCPU), which
                    // compounds through the recurrent GDN state and flips the
                    // logits.  Route through kQuantizeQ8K + the bit-exact
                    // kQGemvKxQ8K<kTypeQ5K> (kxQ8KBlockDot's Q5K branch mirrors
                    // the AVX2 float tree, including the volatile no-FMA
                    // mul-then-add discipline).
                    //
                    // 4-ROWS-PER-WARP LM head (2026-10-12): the 8-active-lane
                    // generic kernel ran the Q5_K LM head (all three IQ models,
                    // 151936 rows) at 2.4-5.8 ms vs the ~0.36-0.85 ms floor.
                    // kQGemvQ5KxQ8K_4xW is a bit-exact drop-in (all 32 lanes
                    // issue weight loads, 4 rows/warp) for rows % 4 == 0;
                    // TINYCODER_Q5K_LM_4XW=0 restores the generic kernel.
                    const char *q5wideE = std::getenv("TINYCODER_Q5K_LM_4XW");
                    const bool q5wide4 =
                            (q5wideE == nullptr || std::atoi(q5wideE) != 0);
                    if (q5wide4 && (rows & 3u) == 0u) {
                        const uint32_t gWarp = (rows / 4u + 7u) / 8u;
                        kQGemvQ5KxQ8K_4xW<<<gWarp, block, 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), q8k, out,
                                rows, blocksPerRow, rowBytes);
                    } else {
                        kQGemvKxQ8K<kTypeQ5K><<<nBlocks, block, 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), q8k, out,
                                rows, blocksPerRow, rowBytes);
                    }
                } else if (type == kTypeQ4K) {
                    // Q4_K: the CPU reference (matMulVecBatchQ4K_SIMD ->
                    // matMulVecBatchQ4K_Q8K_AVX2) quantizes the activation to
                    // Q8_K and does integer dots; the float dequant-dot
                    // kQGemv<kTypeQ4K> diverged (Q4K_Q6KKernelParityWithCPU
                    // proved the Q8_K path bit-exact).  Route through
                    // kQuantizeQ8K + kQGemvKxQ8K<kTypeQ4K>.
                    // 4-ROWS-PER-WARP (2026-10-29, TINYCODER_Q4K_4XW): the
                    // kQGemvQ4KxQ8K_4xW drop-in for rows % 4 == 0 (all real
                    // uses: dense qwen35 attnO/ffnDown/ffnGate/ffnUp and the
                    // Q4_K LM head) is bit-exact and lifts the LARGEST quant
                    // class of this model to the Q5_K/Q6_K 4xW bandwidth shape.
                    if (q4k4xw() && (rows & 3u) == 0u) {
                        const uint32_t w4 =
                                static_cast<uint32_t>(q4kBsy());
                        const uint32_t gWarp = (rows / 4u + w4 - 1u) / w4;
                        kQGemvQ4KxQ8K_4xW<<<gWarp, dim3(32, w4), 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), q8k, out,
                                rows, blocksPerRow, rowBytes);
                    } else {
                        kQGemvKxQ8K<kTypeQ4K><<<nBlocks, block, 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), q8k, out, rows,
                                blocksPerRow, rowBytes);
                    }
                } else if (type == kTypeQ6K) {
                    // Q6_K decode path: 4 rows per warp (all 32 lanes issue
                    // weight loads; the 8-active-lane generic kernel ran the
                    // LM head at ~103 GB/s vs the ~616 GB/s FFN peak because
                    // 24 of 32 lanes idled per instruction).  rows%4==0 for
                    // every real Q6_K use (LM head 151936, ffn_down 1536,
                    // gate/up 8960); fall back to the generic path otherwise.
                    if ((rows & 3u) == 0u) {
                        const uint32_t gBlocks = rows / 4u;
                        const uint32_t gWarp = (gBlocks + 7u) / 8u;
                        const uint32_t nBlocks4 = gWarp;
                        // WIDE (section 9.23) applied to the Q6_K LM head
                        // (2026-10-11): 2x-U16 weight loads + int-deref Y
                        // loads instead of the std::memcpy 4-byte-into-4x-U8
                        // split.  Gated by TINYCODER_MMVQ_WIDE (default on)
                        // for A/B.
                        if (mmvqWide()) {
                            kQGemvQ6KxQ8K_4xW<true>
                                    <<<nBlocks4, block, 0, g_stream>>>(
                                            static_cast<const uint8_t *>(wq),
                                            q8k, out, rows, blocksPerRow,
                                            rowBytes);
                        } else {
                            kQGemvQ6KxQ8K_4xW<false>
                                    <<<nBlocks4, block, 0, g_stream>>>(
                                            static_cast<const uint8_t *>(wq),
                                            q8k, out, rows, blocksPerRow,
                                            rowBytes);
                        }
                    } else {
                        kQGemvKxQ8K<kTypeQ6K><<<nBlocks, block, 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), q8k, out,
                                rows, blocksPerRow, rowBytes);
                    }
                } else if (type == kTypeIQ4XS) {
                    kQGemvKxQ8K<kTypeIQ4XS><<<nBlocks, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), q8k, out, rows,
                            blocksPerRow, rowBytes);
                } else {
                    kQGemvQ8K<kTypeIQ3S><<<nBlocks, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), q8k, out, rows,
                            blocksPerRow, rowBytes);
                }
                return;
            }
            if (type == kTypeQ2K) {
                // Q2_K x Q8_1 dp4a 4xW (2026-10-08, A/B TINYCODER_Q2K_Q81=1,
                // DEFAULT OFF until measured). Removes the float 4xW's
                // fp32-scalar activation load + fmaf per element (the
                // issue-bound cost).
                const char *q81E = std::getenv("TINYCODER_Q2K_Q81");
                const char *q81mE = std::getenv("TINYCODER_Q2K_Q81_MMVQ");
                const bool q81On =
                        (q81E != nullptr && std::atoi(q81E) != 0) ||
                        (q81mE != nullptr && std::atoi(q81mE) != 0);
                if (q81On && (rows & 3u) == 0u) {
                    const uint64_t need =
                            static_cast<uint64_t>(blocksPerRow) * 8u *
                            kQ8_1_STRIDE;
                    if (q8k == nullptr || q8kBytes < need) {
                        if (q8k) cudaFree(q8k);
                        q8k = nullptr;
                        q8kBytes = 0;
                        cudaError_t e = cudaMalloc(&q8k, need);
                        if (e != cudaSuccess) {
                            std::fprintf(stderr, "cudaMalloc(q8_1): %s\n",
                                         cudaGetErrorString(e));
                            cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                            g_stream);
                            return;
                        }
                        q8kBytes = need;
                    }
                    const uint32_t nBlk32 = cols / 32u;
                    kQuantizeQ8_1<<<nBlk32, 32, 0, g_stream>>>(x, q8k, cols);
                    const char *fmm = std::getenv("TINYCODER_Q2K_Q81_MMVQ");
                    if (fmm != nullptr && std::atoi(fmm) != 0 &&
                        blocksPerRow == 6u && rowBytes == 6u * 84u) {
                        kQGemvQ2KxQ81_Mmvq<<<rows, dim3(32, 2), 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), q8k, out,
                                rows, blocksPerRow);
                        return;
                    }
                    const char *skE = std::getenv("TINYCODER_Q2K_Q81_SPLITK");
                    if (skE != nullptr && std::atoi(skE) != 0) {
                        kQGemvQ2KxQ81_SplitK<<<rows, dim3(32, 2), 0,
                                               g_stream>>>(
                                static_cast<const uint8_t *>(wq), q8k, out,
                                rows, blocksPerRow, rowBytes);
                    } else {
                        const uint32_t nW4 = (rows / 4u + 7u) / 8u;
                        kQGemvQ2KxQ81_4xW<<<nW4, block, 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), q8k, out,
                                rows, blocksPerRow, rowBytes);
                    }
                    return;
                }
                // DEFAULT OFF until measured).  Row-granular 64-thread blocks
                // (one row per warp, grid = rows/2) cover the ENTIRE row range
                // so the machine stays full -- the 4xW/MQ4XW 280-block
                // launches only fill ~51% of the 68-SM thread slots and stream
                // the FFN at ~78-113 GB/s.  Requires the same Q8_K activation
                // quantization as MQ4XW; math identical (q2kXQ8KBlockDot).
                const char *mmv = std::getenv("TINYCODER_Q2K_MMVQ");
                if (mmv != nullptr && std::atoi(mmv) != 0 && rows >= 2u) {
                    const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                                          static_cast<uint64_t>(kQ8K_STRIDE);
                    if (q8k == nullptr || q8kBytes < need) {
                        if (q8k) cudaFree(q8k);
                        q8k = nullptr;
                        q8kBytes = 0;
                        cudaError_t e = cudaMalloc(&q8k, need);
                        if (e != cudaSuccess) {
                            std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                                         cudaGetErrorString(e));
                            cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                            g_stream);
                            return;
                        }
                        q8kBytes = need;
                    }
                    kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(x, q8k,
                                                                     cols);
                    const uint32_t nB = (rows + 1u) / 2u;
                    kQGemvQ2KxQ8K_Mmvq<<<nB, dim3(32, 2), 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), q8k, out, rows,
                            blocksPerRow, rowBytes);
                    return;
                }
                // Q2_K x Q8_K int-dot 4xW (2026-09-30, A/B
                // TINYCODER_Q2K_MQ4XW=1, DEFAULT OFF until measured).
                // reference-mmq decode route: quantize the activation to Q8_K
                // once (kQuantizeQ8K) then run the integer 2-bit x Q8_K dot in
                // the Q6_K-proven 4-rows-per-warp shape -- every lane issues
                // u32 loads, 8 lanes per row, 4 rows per warp, no smem.  The
                // float kQGemvQ2KxW4 reads the fp32 activation SCALAR per
                // element, keeping the FFN at ~75 GB/s; this int path is the
                // fix for that (see kernel comment).  Requires rows%4==0
                // (all real Q2_K uses: gate/up 8960, attnO 1536).
                const char *mqx = std::getenv("TINYCODER_Q2K_MQ4XW");
                if (mqx != nullptr && std::atoi(mqx) != 0 &&
                    (rows & 3u) == 0u) {
                    const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                                          static_cast<uint64_t>(kQ8K_STRIDE);
                    if (q8k == nullptr || q8kBytes < need) {
                        if (q8k) cudaFree(q8k);
                        q8k = nullptr;
                        q8kBytes = 0;
                        cudaError_t e = cudaMalloc(&q8k, need);
                        if (e != cudaSuccess) {
                            std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                                         cudaGetErrorString(e));
                            cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                            g_stream);
                            return;
                        }
                        q8kBytes = need;
                    }
                    kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(x, q8k,
                                                                     cols);
                    // MQ4XW BLOCK-SHAPE A/B (2026-10-10, TINYCODER_Q2K_MQ4XW_BSY):
                    // this int body has NO shared memory, so unlike the Q8_1
                    // 4xW it can run at the winning BSY=2 block shape.  It was
                    // only ever measured at the old 8-warp default.  Default
                    // stays 8 (unchanged) until the A/B decides.
                    const char *mbsyE = std::getenv("TINYCODER_Q2K_MQ4XW_BSY");
                    const uint32_t mwarps = (mbsyE != nullptr)
                                                    ? static_cast<uint32_t>(
                                                              std::atoi(mbsyE))
                                                    : 8u;
                    const uint32_t mWarps = rows / 4u;
                    const uint32_t nW4 = (mWarps + mwarps - 1u) / mwarps;
                    kQGemvQ2KxQ8K_4xW<<<nW4, dim3(32, mwarps), 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), q8k, out, rows,
                            blocksPerRow, rowBytes);
                    return;
                }
                // 4-rows-per-warp Q2_K float kernel (2026-09-30,
                // TINYCODER_Q2K_4XW=0 disables; DEFAULT ON): replicates the
                // kQGemvQ6KxQ8K_4xW bandwidth shape (~616 GB/s) -- all 32
                // lanes issue u32 weight loads, 8 lanes per row, 4 rows per
                // warp, NO shared memory / barrier (both avoided because
                // kQGemvQ2K2xW's smem staging serialized the block).
                // rows%4==0 for all real Q2_K uses (gate/up 8960, attnO
                // 1536); per-element math identical to kQGemv<kTypeQ2K>,
                // reduce order differs (float-path ~1e-7 tolerance).
                // Measured +2.0-2.1% tg8 (165.1 -> 168.4 steady, interleaved
                // A/B, both rounds stable), parity 11/11 PASS.
                // 8-way block batching A/B (2026-09-30, lever-3,
                // TINYCODER_Q2K_8W=1, DEFAULT OFF): same 4xW geometry with
                // 8-block batch chains (see kernel comment).  Overrides the
                // 4xW path when enabled.
                const char *q2k8 = std::getenv("TINYCODER_Q2K_8W");
                const bool use8W = (q2k8 != nullptr && std::atoi(q2k8) != 0);
                const char *q2k4 = std::getenv("TINYCODER_Q2K_4XW");
                // MULTI-ROW-PER-WARP A/B (2026-09-30, TINYCODER_Q2K_NM=RPB,
                // DEFAULT OFF): RPB in {8,16,32}; overrides the 4xW path when
                // set.  See kQGemvQ2KxW4_Nm -- raises per-warp memory-level
                // parallelism on the tiny-grid Q2_K FFN decode GEMV.
                const char *q2knm = std::getenv("TINYCODER_Q2K_NM");
                const int nmv = (q2knm != nullptr) ? std::atoi(q2knm) : 0;
                // 4-WAY K-SPLIT A/B (2026-09-30, TINYCODER_Q2K_SPLIT4=1,
                // DEFAULT OFF): reference-mmvq-style round-robin block split
                // across the warp's 4 lane groups.  See kQGemvQ2KxSplit4.
                const char *q2ks4 = std::getenv("TINYCODER_Q2K_SPLIT4");
                if (q2ks4 != nullptr && std::atoi(q2ks4) != 0) {
                    kQGemvQ2KxSplit4<<<nBlocks, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out, rows,
                            cols, rowBytes, blocksPerRow);
                } else if (nmv == 8 && (rows % 8u) == 0u) {
                    kQGemvQ2KxW4_Nm<8>
                            <<<(rows / 8u + 7u) / 8u, block, 0, g_stream>>>(
                                    static_cast<const uint8_t *>(wq), x, out,
                                    rows, cols, rowBytes, blocksPerRow);
                } else if (nmv == 16 && (rows % 16u) == 0u) {
                    kQGemvQ2KxW4_Nm<16>
                            <<<(rows / 16u + 7u) / 8u, block, 0, g_stream>>>(
                                    static_cast<const uint8_t *>(wq), x, out,
                                    rows, cols, rowBytes, blocksPerRow);
                } else if (nmv == 32 && (rows % 32u) == 0u) {
                    kQGemvQ2KxW4_Nm<32>
                            <<<(rows / 32u + 7u) / 8u, block, 0, g_stream>>>(
                                    static_cast<const uint8_t *>(wq), x, out,
                                    rows, cols, rowBytes, blocksPerRow);
                } else if (use8W && (rows & 3u) == 0u) {
                    const uint32_t nW4 = (rows / 4u + 7u) / 8u;
                    kQGemvQ2KxW4b8<<<nW4, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out, rows,
                            cols, rowBytes, blocksPerRow);
                } else if ((q2k4 == nullptr || std::atoi(q2k4) != 0) &&
                           (rows & 3u) == 0u) {
                    // BLOCK-SHAPE A/B (2026-09-30, TINYCODER_Q2K_BSY=W): warps
                    // per block for the 4xW kernel. 2-warp (64-thread) blocks
                    // push the gate/up grid from 280 to 1120 blocks; the kernel
                    // body and per-value math are unchanged (parity-identical).
                    const char *bsyE = std::getenv("TINYCODER_Q2K_BSY");
                    const uint32_t warps = (bsyE != nullptr)
                                                   ? static_cast<uint32_t>(
                                                             std::atoi(bsyE))
                                                   : 2u;
                    const uint32_t nWarps = rows / 4u;
                    const uint32_t nW4 =
                            (nWarps + warps - 1u) / warps;
                    // REPACKED 16-byte-lane layout (2026-10-10, §9.19,
                    // TINYCODER_Q2K_RPACK=1): repack once per matrix (cached by
                    // weight pointer; the first eager decode repacks BEFORE any
                    // graph capture, so the captured graph replays the RP
                    // kernel), then run kQGemvQ2KxW4_RP -- 3 weight loads per
                    // block instead of 12, each covering 64 contiguous bytes.
                    // Byte-identical to MODE 1.
                    const char *rpE = std::getenv("TINYCODER_Q2K_RPACK");
                    if (rpE != nullptr && std::atoi(rpE) != 0 &&
                        (cols & 255u) == 0u && (rows & 3u) == 0u) {
                        static std::mutex rpMtx;
                        static std::vector<std::pair<const void *, uint8_t *>>
                                rpCache;
                        uint8_t *rpBuf = nullptr;
                        {
                            std::lock_guard<std::mutex> lk(rpMtx);
                            for (auto &kv: rpCache)
                                if (kv.first == wq) {
                                    rpBuf = kv.second;
                                    break;
                                }
                            // cudaMalloc is illegal during stream capture, and
                            // launchQGemv is a free function (no member flag):
                            // query the stream directly.  The first eager
                            // decode repacks, so capture always hits the cache.
                            cudaStreamCaptureStatus capSt =
                                    cudaStreamCaptureStatusNone;
                            cudaStreamIsCapturing(g_stream, &capSt);
                            const bool capturing =
                                    (capSt != cudaStreamCaptureStatusNone);
                            if (rpBuf == nullptr && !capturing) {
                                void *p = nullptr;
                                const uint64_t rpBytes =
                                        static_cast<uint64_t>(rows) *
                                        blocksPerRow * 136u;
                                if (cudaMalloc(&p, rpBytes) == cudaSuccess) {
                                    kRepackQ2K<<<dim3(rows, blocksPerRow), 32, 0,
                                                 g_stream>>>(
                                            static_cast<const uint8_t *>(wq),
                                            static_cast<uint8_t *>(p), rowBytes,
                                            blocksPerRow);
                                    rpBuf = static_cast<uint8_t *>(p);
                                    rpCache.push_back({wq, rpBuf});
                                }
                            }
                        }
                        if (rpBuf != nullptr) {
                            if (exactCols)
                                kQGemvQ2KxW4_RP<true><<<nW4, dim3(32, warps), 0,
                                                        g_stream>>>(
                                        rpBuf, x, out, rows, cols, blocksPerRow);
                            else
                                kQGemvQ2KxW4_RP<false><<<nW4, dim3(32, warps), 0,
                                                         g_stream>>>(
                                        rpBuf, x, out, rows, cols, blocksPerRow);
                            return;
                        }
                    }
                    // MODE A/B (2026-10-10, TINYCODER_Q2K_MODE=0/1/2): 0 =
                    // guarded scalar, 1 = EXACT (guard dropped when cols are
                    // exact), 2 = EXACT + 128-bit activation load (LDG.128 per
                    // 4 contiguous cols; attacks the issue-bound SM 97%/DRAM
                    // 20% profile, plan ch.4).  Modes 1/2 are bit-identical to
                    // each other; both require exactCols to be safe.
                    // Default 1 (EXACT scalar): MODE 2 (LDG.128) measured
                    // within noise / -0.2% over MODE 1 in 4 clean interleaved
                    // rounds -> REJECTED (activation loads are not the Q2_K
                    // gate/up limiter).  Kept env-selectable as an A/B record.
                    const char *modeE = std::getenv("TINYCODER_Q2K_MODE");
                    const int q2kMode = (modeE != nullptr) ? std::atoi(modeE) : 1;
                    if (exactCols && q2kMode == 3) {
                        kQGemvQ2KxW4<3><<<nW4, dim3(32, warps), 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), x, out, rows,
                                cols, rowBytes, blocksPerRow);
                    } else if (exactCols && q2kMode == 6) {
                        // MODE 6 (2026-10-10): MODE 1 with __ldcg (L2-only)
                        // weight loads -- L1 pipe relief per the fresh ncu
                        // (L1 73-75 % / DRAM 19 %).
                        kQGemvQ2KxW4<6><<<nW4, dim3(32, warps), 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), x, out, rows,
                                cols, rowBytes, blocksPerRow);
                    } else if (exactCols && q2kMode == 5) {
                        // MODE 5 (2026-10-10): packed weight loads + LDG.128 x.
                        kQGemvQ2KxW4<5><<<nW4, dim3(32, warps), 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), x, out, rows,
                                cols, rowBytes, blocksPerRow);
                    } else if (exactCols && q2kMode == 4) {
                        // MODE 4 (2026-10-10): packed weight loads only (u32
                        // d|dm + u32 scale words instead of half/byte loads).
                        kQGemvQ2KxW4<4><<<nW4, dim3(32, warps), 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), x, out, rows,
                                cols, rowBytes, blocksPerRow);
                    } else if (exactCols && q2kMode >= 2) {
                        kQGemvQ2KxW4<2><<<nW4, dim3(32, warps), 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), x, out, rows,
                                cols, rowBytes, blocksPerRow);
                    } else if (exactCols) {
                        kQGemvQ2KxW4<1><<<nW4, dim3(32, warps), 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), x, out, rows,
                                cols, rowBytes, blocksPerRow);
                    } else {
                        kQGemvQ2KxW4<0><<<nW4, dim3(32, warps), 0,
                                          g_stream>>>(
                                static_cast<const uint8_t *>(wq), x, out, rows,
                                cols, rowBytes, blocksPerRow);
                    }
                } else {
                    kQGemv<kTypeQ2K><<<nBlocks, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out, rows,
                            cols, rowBytes, blocksPerRow);
                }
            } else if (type == kTypeQ3K) {
                // Quantizes the fp32 activation to Q8_1 once (per-32 d+s), in
                // the (b*8+n) numbering the n-q3 row index (== cols/32/8)
                // requires. Targets the FFN down (the 130 GB/s stage): measured
                // down 1.40 -> 1.12 ms/token (~130 -> 163 GB/s), whole-decode
                // +5.0-5.4% (eager + graph), greedy parity byte-identical on
                // 4/4 prompts (plan 9.9).
                const char *fmm3 = std::getenv("TINYCODER_Q3K_MMVQ");
                if ((fmm3 == nullptr || std::atoi(fmm3) != 0) &&
                    (cols & 31u) == 0u) {
                    const uint64_t need =
                            static_cast<uint64_t>(blocksPerRow) * 8u *
                            kQ8_1_STRIDE;
                    if (q8k == nullptr || q8kBytes < need) {
                        if (q8k) cudaFree(q8k);
                        q8k = nullptr;
                        q8kBytes = 0;
                        cudaError_t e = cudaMalloc(&q8k, need);
                        if (e != cudaSuccess) {
                            std::fprintf(stderr, "cudaMalloc(q8_1): %s\n",
                                         cudaGetErrorString(e));
                            cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                            g_stream);
                            return;
                        }
                        q8kBytes = need;
                    }
                    const uint32_t nBlk32 = cols / 32u;
                    kQuantizeQ8_1<<<nBlk32, 32, 0, g_stream>>>(x, q8k, cols);
                    if (mmvqWide())
                        kQGemvQ3KxQ81_Mmvq<true><<<rows, dim3(32, 2), 0,
                                                   g_stream>>>(
                                static_cast<const uint8_t *>(wq), q8k, out,
                                rows, blocksPerRow, rowBytes, nullptr);
                    else
                        kQGemvQ3KxQ81_Mmvq<false><<<rows, dim3(32, 2), 0,
                                                    g_stream>>>(
                                static_cast<const uint8_t *>(wq), q8k, out,
                                rows, blocksPerRow, rowBytes, nullptr);
                    return;
                }
                // SPLIT-K occupancy A/B (2026-09-29, TINYCODER_SPLITK_DOWN=n):
                // the small-row Q3_K ffn down (rows=1536) runs at ~2.8
                // blocks/SM resident; split each row's block range across n
                // chunk-warps in ONE block to raise the concurrent-stream
                // count.  Per-chunk math is byte-identical; the cross-chunk
                // join is a 1-ulp-class reorder (see kQGemvSplitK).
                const char *sk = std::getenv("TINYCODER_SPLITK_DOWN");
                if (sk != nullptr) {
                    const int skn = std::atoi(sk);
                    if (skn == 2) {
                        kQGemvSplitK<kTypeQ3K, 2><<<rows, dim3(32, 2), 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), x, out, rows,
                                cols, rowBytes, blocksPerRow);
                        return;
                    }
                    if (skn == 4) {
                        kQGemvSplitK<kTypeQ3K, 4><<<rows, dim3(32, 4), 0, g_stream>>>(
                                static_cast<const uint8_t *>(wq), x, out, rows,
                                cols, rowBytes, blocksPerRow);
                        return;
                    }
                }
                // 4-rows-per-warp Q3_K kernel (2026-10-06, P1 bandwidth
                // campaign, TINYCODER_Q3K_4XW=1 opts IN; DEFAULT OFF).
                // Mirrors the byte-shape-proven kQGemvQ2KxW4 geometry (all 32
                // lanes issue u32 weight loads, 8 lanes/row, 4 row-streams per
                // warp, NO smem/barrier), but the Q3_K hmask fold adds ~1 ALU
                // op per weight byte that the 1-row-per-warp kernel does NOT
                // pay.  MEASURED 2026-10-06 (Release, AVX2, interleaved,
                // n-gen 8, reps 8): baseline ~163 tok/s vs 4xW ~150 tok/s
                // (~-8%); block-size sweep BSY=1/2/4 also ~154-156, none
                // recovers it.  The ffn down (1536 rows) is ISSUE/LATENCY-bound
                // (SM ~96% busy, DRAM ~20%), so the extra per-byte ALU is not
                // hidden -- this REFUTES the "Q3_K down is bandwidth-limited by
                // its 1-row geometry" hypothesis and reconfirms the README
                // campaign verdict.  Kept opt-in as an A/B record; parity
                // verified (44 PASS / 0 FAIL with the kernel ON).
                const char *q3k4 = std::getenv("TINYCODER_Q3K_4XW");
                const bool use4W = (q3k4 != nullptr && std::atoi(q3k4) != 0);
                if (use4W && (rows & 3u) == 0u) {
                    const char *bsyE3 = std::getenv("TINYCODER_Q3K_BSY");
                    const uint32_t warps3 = (bsyE3 != nullptr)
                                                    ? static_cast<uint32_t>(
                                                              std::atoi(bsyE3))
                                                    : 8u;
                    const uint32_t nW4 = (rows / 4u + warps3 - 1u) / warps3;
                    kQGemvQ3KxW4<<<nW4, dim3(32, warps3), 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out, rows,
                            cols, rowBytes, blocksPerRow);
                    return;
                }
                if (exactCols) {
                    kQGemv<kTypeQ3K, true><<<nBlocks, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out, rows,
                            cols, rowBytes, blocksPerRow);
                } else {
                    kQGemv<kTypeQ3K, false><<<nBlocks, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out, rows,
                            cols, rowBytes, blocksPerRow);
                }
            } else if (type == kTypeIQ2XS) {
                kQGemv<kTypeIQ2XS><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeIQ2XXS) {
                kQGemv<kTypeIQ2XXS><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeIQ3S) {
                kQGemv<kTypeIQ3S><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeIQ2S) {
                kQGemv<kTypeIQ2S><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeIQ3_XXS) {
                kQGemv<kTypeIQ3_XXS><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeQ5_0) {
                kQGemvSmall32<kTypeQ5_0><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeIQ4NL) {
                // IQ4_NL (4.5 bpw) GEMV: 32-wide legacy blocks, weight =
                // d * kvalues_iq4nl[nibble] (float dequant-dot, like Q5_0).
                kQGemvSmall32<kTypeIQ4NL><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypePQ2_0) {
                // Prism PQ2_0 (Bonsai) ternary GEMV: 128-wide blocks (34 B), one
                // warp per output row (grid = rows, NOT nBlocks).  The `x` here
                // is already the folded activation H S P x (the caller applies
                // kPrismFold first).
                kQGemvPQ2_0<<<rows, 32, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeQ8_0) {
                // Q8_0: integer Q8_32 activation GEMV, bit-exact with the CPU
                // matMulVecBatchQ8_0_Q8K_AVX2 (quantize x per 32-block, exact
                // int8 madd + AVX float tree).  The fp32 dequant-dot
                // (kQGemvSmall32) rounds differently and diverges the logits
                // through a 40-layer decode, flipping near-tie argmax rows
                // (batch-vs-sequential and GPU-vs-CPU parity failures on
                // qwen35moe).  Every Q8_0 path — trunk GEMVs, routed experts,
                // shared expert, LM head — must run this identical math so the
                // GPU matches the CPU reference bit-for-bit.
                // The Q8_32 activation scratch REUSES the caller-supplied
                // q8k/q8kBytes (sized blocksPerRow * kQ8K_STRIDE = 304 B/block,
                // far larger than the 34 B/block this kernel needs), so no
                // separate allocation is required.  kQuantizeQ8_0x32 writes the
                // 32-wide Q8_0 activation blocks into the SAME buffer the
                // 256-wide kQuantizeQ8K would use — launchQGemv never has both
                // in flight on one call.
                // kQ8_0x32_STRIDE (64) < kQ8K_STRIDE (304): the reused buffer's
                // cap always covers the denser Q8_32 layout; the writer never
                // exceeds blocksPerRow*64 bytes.
                const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                                      static_cast<uint64_t>(kQ8_0x32_STRIDE);
                const uint64_t have = static_cast<uint64_t>(blocksPerRow) *
                                      static_cast<uint64_t>(kQ8K_STRIDE);
                if (q8k == nullptr || q8kBytes < have) {
                    if (q8k) cudaFree(q8k);
                    q8k = nullptr;
                    q8kBytes = 0;
                    cudaError_t e = cudaMalloc(&q8k, have);
                    if (e != cudaSuccess) {
                        std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                                     cudaGetErrorString(e));
                        cudaMemsetAsync(out, 0, sizeof(float) * rows, g_stream);
                        return;
                    }
                    q8kBytes = have;
                }
                (void) need;
                kQuantizeQ8_0x32<<<blocksPerRow, 32, 0, g_stream>>>(x, q8k, cols);
                kQGemvQ8_0xQ8K<<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out, rows,
                        blocksPerRow, rowBytes);
            } else if (type == kTypeF32) {
                // Plain F32 GEMV: weight blob is rows*cols floats row-major
                // (the qwen35 ssm_alpha / ssm_beta small matrices).  The device
                // buffer was uploaded with exactly this layout.
                kQ35F32Gemv<<<rows, 256, 0, g_stream>>>(
                        x, static_cast<const float *>(wq), out, rows, cols);
            } else {
                cudaMemsetAsync(out, 0, sizeof(float) * rows, g_stream);
            }
        }

        // Placeholder q8_1-buffer references for launchQGemvResid's default
        // arguments (only the ffnDown caller passes the real member buffers).
        uint8_t *g_q8k_unused = nullptr;
        uint64_t g_q8k_unused_bytes = 0;

        // ---- launchQGemvResid (2026-10-09): the Q2_K (attnO) and Q3_K
        // (ffnDown) decode GEMVs fused with the trailing residual
        // (acc[row] += out[row]).  Deletes the kAddResidual launch after these
        // two projections; the dot math and reduce tree are IDENTICAL to the
        // base kernels, so out[] is written byte-for-byte as before and the
        // residual stream stays byte-identical.  The activation is always fp32
        // (these projections do not use the Q8_K int path by default), and the
        // EXACT-columns flag mirrors launchQGemv.  Returns true when it
        // launched; false for an unsupported type/shape, so the caller falls
        // back to launchQGemv + kAddResidual.
        bool launchQGemvResid(int type, const void *wq, const float *x,
                              float *out, float *acc, uint32_t rows,
                              uint32_t cols, uint32_t rowBytes,
                              uint32_t blocksPerRow, bool allowMmvq = false,
                              uint8_t *&q8k = g_q8k_unused,
                              uint64_t &q8kBytes = g_q8k_unused_bytes,
                              const uint8_t *preQ81 = nullptr) {
            const char *exactE = std::getenv("TINYCODER_EXACT_COLS");
            const bool exactCols =
                    (exactE == nullptr || std::atoi(exactE) != 0) &&
                    (static_cast<uint64_t>(blocksPerRow) * 256u ==
                     static_cast<uint64_t>(cols));
            if (type == kTypeQ2K) {
                // Match the DEFAULT 4xW path (BSY=2 warps/block, empty-env
                // enabled).  If the caller has switched off 4xW (or enabled a
                // different Q2_K geometry A/B), decline so the base
                // launchQGemv + kAddResidual runs instead -- keeps the fused
                // path exactly on top of whatever the default geometry is.
                const char *q2k4 = std::getenv("TINYCODER_Q2K_4XW");
                const bool use4W = (q2k4 == nullptr || std::atoi(q2k4) != 0);
                if (!use4W || (rows & 3u) != 0u) return false;
                const char *bsyE = std::getenv("TINYCODER_Q2K_BSY");
                const uint32_t warps =
                        (bsyE != nullptr)
                                ? static_cast<uint32_t>(std::atoi(bsyE))
                                : 2u;
                const uint32_t nW4 = (rows / 4u + warps - 1u) / warps;
                if (exactCols) {
                    kQGemvQ2KxW4Resid<true><<<nW4, dim3(32, warps), 0,
                                              g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out, acc, rows,
                            cols, rowBytes, blocksPerRow);
                } else {
                    kQGemvQ2KxW4Resid<false><<<nW4, dim3(32, warps), 0,
                                               g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out, acc, rows,
                            cols, rowBytes, blocksPerRow);
                }
                return true;
            }
            if (type == kTypeQ3K && allowMmvq) {
                // Used for ffnDown only (allowMmvq); attnO keeps the float
                // fused-resid path.
                const char *fmm3 = std::getenv("TINYCODER_Q3K_MMVQ");
                const bool mmvq =
                        (fmm3 == nullptr || std::atoi(fmm3) != 0) &&
                        (cols & 31u) == 0u;
                if (mmvq) {
                    // preQ81 (2026-10-10, plan section 9.26 lever 1): the
                    // caller's producer (combine W4 for attnO) already wrote
                    // the Q8_1 activation blocks -- skip the allocation guard
                    // and the separate kQuantizeQ8_1 launch entirely.
                    if (preQ81 == nullptr) {
                        const uint64_t need =
                                static_cast<uint64_t>(blocksPerRow) * 8u *
                                kQ8_1_STRIDE;
                        if (q8k == nullptr || q8kBytes < need) {
                            if (q8k) cudaFree(q8k);
                            q8k = nullptr;
                            q8kBytes = 0;
                            cudaError_t e = cudaMalloc(&q8k, need);
                            if (e != cudaSuccess) {
                                std::fprintf(stderr, "cudaMalloc(q8_1): %s\n",
                                             cudaGetErrorString(e));
                                return false;
                            }
                            q8kBytes = need;
                        }
                        const uint32_t nBlk32 = cols / 32u;
                        kQuantizeQ8_1<<<nBlk32, 32, 0, g_stream>>>(x, q8k, cols);
                    }
                    const uint8_t *y1 = (preQ81 != nullptr) ? preQ81 : q8k;
                    if (mmvqWide())
                        kQGemvQ3KxQ81_Mmvq<true><<<rows, dim3(32, 2), 0,
                                                   g_stream>>>(
                                static_cast<const uint8_t *>(wq), y1, out, rows,
                                blocksPerRow, rowBytes, acc);
                    else
                        kQGemvQ3KxQ81_Mmvq<false><<<rows, dim3(32, 2), 0,
                                                    g_stream>>>(
                                static_cast<const uint8_t *>(wq), y1, out, rows,
                                blocksPerRow, rowBytes, acc);
                    return true;
                }
            }
            if (type == kTypeQ3K) {
                const uint32_t nBlocks = (rows + 7u) / 8u;
                dim3 block(32, 8);
                if (exactCols) {
                    kQGemvQ3KResid<true><<<nBlocks, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out, acc, rows,
                            cols, rowBytes, blocksPerRow);
                } else {
                    kQGemvQ3KResid<false><<<nBlocks, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out, acc, rows,
                            cols, rowBytes, blocksPerRow);
                }
                return true;
            }
            return false;
        }

        // ---- Fused gate+up+silu dispatcher (decode seqLen==1, FFN).
        // Replaces launchQGemv(gate) + launchQGemv(up) + kSiluMul with ONE
        // launch.  `wg`/`wu` are the gate/up row-major weight bases and both
        // matrices MUST be identical (rows, cols, rowBytes, blocksPerRow) --
        // guaranteed by the Model layer for the dense Qwen2 FFN gate/up pair.
        // ffnRows = rows (gate rows == up rows).  out[ffnRows] receives
        // silu(gate(x))*up(x) -- the same product that kSiluMul was computing
        // into s.gate from the two separate launchQGemv results.  The raw up
        // row values, and s.up, are dead after this (the down QGEMV consumes
        // out), so they are not written.
        void launchQGemvFusedGU(int type, const void *wg, const void *wu,
                                const float *x, float *out, uint32_t ffnRows,
                                uint32_t cols, uint32_t rowBytes,
                                uint32_t blocksPerRow, uint8_t *&q8k,
                                uint64_t &q8kBytes,
                                const uint8_t *preQ81 = nullptr) {
            dim3 block(32, 8);
            uint32_t nBlocks = (ffnRows + 7) / 8;
            const uint32_t rows = ffnRows;
            if (type == kTypeQ2K) {
                // Q2_K FFN gate+up.  Two routes:
                //   * TINYCODER_Q2K_INTGU=1 (2026-09-30, lever-1 A/B, DEFAULT
                //     OFF): reference-mmq-style integer path -- kQuantizeQ8K
                //     the fp32 activation ONCE, then BOTH row dots run as
                //     Q2_K-int x Q8_K-int DP4A-style dots (kQGemvQ2KIntFusedGU)
                //     against the shared Q8_K activation, replacing the
                //     ALU-bound float dequant.  Bit-exact int side with the
                //     CPU reference dotProductQ2_K_PrePacked_Q8_Scalar / the
                //     compact AVX2 kernel; float fold in serial block order
                //     like kQGemvQ8KFusedGU.
                //   * otherwise (DEFAULT): float dequant-dot path, computes
                //     both row dots + the silu product in ONE launch (was 3).
                //     Bit-exact with kQGemv<kTypeQ2K> + kSiluMul.
                const char *q2kInt = std::getenv("TINYCODER_Q2K_INTGU");
                if (q2kInt != nullptr && std::atoi(q2kInt) != 0) {
                    const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                                          static_cast<uint64_t>(kQ8K_STRIDE);
                    if (q8k == nullptr || q8kBytes < need) {
                        if (q8k) cudaFree(q8k);
                        q8k = nullptr;
                        q8kBytes = 0;
                        cudaError_t e = cudaMalloc(&q8k, need);
                        if (e != cudaSuccess) {
                            std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                                         cudaGetErrorString(e));
                            cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                            g_stream);
                            return;
                        }
                        q8kBytes = need;
                    }
                    kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(
                            x, q8k, cols);
                    kQGemvQ2KIntFusedGU<<<nBlocks, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wg),
                            static_cast<const uint8_t *>(wu), q8k, out, rows,
                            blocksPerRow, rowBytes);
                    return;
                }
                // TINYCODER_FUSE_GU_Q2K=2 (2026-10-10): reference-style fused
                // GLU on the dp4a Q8_1 path -- ONE launch quantizes the fp32
                // activation to Q8_1 (the SAME kQuantizeQ8_1 the down mmvq
                // uses), then kQGemvQ2KxQ81_GLU computes both weight dots with
                // the shared Q8_1 operand and writes silu(gate)*up. Default = 2
                // when the env is unset (see the call-site comment); the
                // kernel's 504-B row stride is hardcoded for cols==1536
                // (blocksPerRow==6), so anything else falls to the float path.
                const char *fgMode = std::getenv("TINYCODER_FUSE_GU_Q2K");
                const int gluMode = (fgMode != nullptr) ? std::atoi(fgMode) : 2;
                if (gluMode == 2 && (cols & 31u) == 0u &&
                    rowBytes == 6u * 84u) {
                    const uint64_t need =
                            static_cast<uint64_t>(blocksPerRow) * 8u *
                            kQ8_1_STRIDE;
                    // preQ81 (2026-10-10, plan section 9.26 lever 1): the FFN
                    // rms_norm epilogue already wrote the Q8_1 activation
                    // blocks -- skip the allocation guard and the separate
                    // kQuantizeQ8_1 launch entirely.
                    const uint8_t *y1;
                    if (preQ81 != nullptr) {
                        y1 = preQ81;
                    } else {
                        if (q8k == nullptr || q8kBytes < need) {
                            if (q8k) cudaFree(q8k);
                            q8k = nullptr;
                            q8kBytes = 0;
                            cudaError_t e = cudaMalloc(&q8k, need);
                            if (e != cudaSuccess) {
                                std::fprintf(stderr,
                                             "cudaMalloc(q8_1 glu): %s\n",
                                             cudaGetErrorString(e));
                                cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                                g_stream);
                                return;
                            }
                            q8kBytes = need;
                        }
                        kQuantizeQ8_1<<<cols / 32u, 32, 0, g_stream>>>(x, q8k,
                                                                       cols);
                        y1 = q8k;
                    }
                    if (mmvqWide())
                        kQGemvQ2KxQ81_GLU<true><<<rows, dim3(32, 2), 0,
                                                  g_stream>>>(
                                static_cast<const uint8_t *>(wg),
                                static_cast<const uint8_t *>(wu), y1, out,
                                rows, blocksPerRow);
                    else
                        kQGemvQ2KxQ81_GLU<false><<<rows, dim3(32, 2), 0,
                                                   g_stream>>>(
                                static_cast<const uint8_t *>(wg),
                                static_cast<const uint8_t *>(wu), y1, out,
                                rows, blocksPerRow);
                    return;
                }
                kQGemvFusedGU_Q2K<<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wg),
                        static_cast<const uint8_t *>(wu), x, out, rows, cols,
                        rowBytes, blocksPerRow);
                return;
            }
            if (type == kTypeIQ2XS && !iq2xsInt()) {
                // Legacy float dequant-dot path (A/B only): no quantize
                // needed; reads x directly.
                kQGemvFusedGU_IQ2XS<<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wg),
                        static_cast<const uint8_t *>(wu), x, out, rows, cols,
                        rowBytes, blocksPerRow);
                return;
            }
            if (type == kTypeIQ2S || (type == kTypeIQ2XS && iq2xsInt()) ||
                type == kTypeIQ3_XXS || type == kTypeIQ3S) {
                // Q8_K integer path: quantize the activation ONCE, share it.
                const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                                      static_cast<uint64_t>(kQ8K_STRIDE);
                if (q8k == nullptr || q8kBytes < need) {
                    if (q8k) cudaFree(q8k);
                    q8k = nullptr;
                    q8kBytes = 0;
                    cudaError_t e = cudaMalloc(&q8k, need);
                    if (e != cudaSuccess) {
                        std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                                     cudaGetErrorString(e));
                        cudaMemsetAsync(out, 0, sizeof(float) * rows, g_stream);
                        return;
                    }
                    q8kBytes = need;
                }
                kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(x, q8k, cols);
                const bool d4 = iqDp4a();
                const bool grsm = iqGridSmem();
#define TC_FUSED_GU(T, GB)                                              \
    do {                                                                \
        if (d4)                                                         \
            kQGemvQ8KFusedGU<T, true>                                   \
                    <<<nBlocks, block, grsm ? (GB) : 0u, g_stream>>>(   \
                            static_cast<const uint8_t *>(wg),           \
                            static_cast<const uint8_t *>(wu), q8k, out, \
                            rows, blocksPerRow, rowBytes, grsm);        \
        else                                                            \
            kQGemvQ8KFusedGU<T, false>                                  \
                    <<<nBlocks, block, grsm ? (GB) : 0u, g_stream>>>(   \
                            static_cast<const uint8_t *>(wg),           \
                            static_cast<const uint8_t *>(wu), q8k, out, \
                            rows, blocksPerRow, rowBytes, grsm);        \
    } while (0)
                if (type == kTypeIQ2S)
                    TC_FUSED_GU(kTypeIQ2S, 8192u);
                else if (type == kTypeIQ2XS)
                    TC_FUSED_GU(kTypeIQ2XS, 4096u);
                else if (type == kTypeIQ3_XXS)
                    TC_FUSED_GU(kTypeIQ3_XXS, 1024u);
                else
                    TC_FUSED_GU(kTypeIQ3S, 2048u);
#undef TC_FUSED_GU
                return;
            }
            // Unsupported pair: fall back to the two separate kernels + silu
            // (the caller only takes this path when gate/up types match a
            // supported fused type, so this is a bare safety net).
            launchQGemv(type, wg, x, out, rows, cols, rowBytes, blocksPerRow,
                        q8k, q8kBytes);
        }

        // Q8_0 decode-helper split (added 2026-09-13): the decode expert path
        // runs up to 8 experts per layer, each with gate/up/down GEMVs against
        // the SAME fp32 activation (s.norm).  The old path re-quantized
        // s.norm for every (expert, matrix) pair -- ~1920 quantize+GEMV
        // kernel launches per decode token (8 ranks x 3 matrices x 40 layers),
        // each kernel under-occupying the GPU.  Split the Q8_0 path so the
        // caller quantizes the activation ONCE per layer into the shared q8k_
        // scratch and runs the pre-quantized GEMVs.

        // Quantize x (cols floats) into the caller-supplied q8k scratch using
        // the Q8_32 layout (kQ8_0x32_STRIDE=64).  grid = blocksPerRow.
        void launchQuantQ8_0x32(const float *x, uint8_t *q8k, uint32_t cols) {
            const uint32_t blocksPerRow = (cols + 31) / 32;
            kQuantizeQ8_0x32<<<blocksPerRow, 32, 0, g_stream>>>(x, q8k, cols);
        }

        // Integer Q8_0 GEMV against a PRE-QUANTIZED activation (the Q8_32
        // layout from launchQuantQ8_0x32).  Bit-exact with the CPU reference:
        // same float tree, same op order, no fmaf -- identical to
        // launchQGemv's Q8_0 branch WITHOUT the re-quantization.
        void launchQGemvQ8_0Pre(const void *wq, const uint8_t *q8k, float *out,
                                uint32_t rows, uint32_t blocksPerRow,
                                uint32_t rowBytes) {
            dim3 block(32, 8);
            uint32_t nBlocks = (rows + 7) / 8;
            kQGemvQ8_0xQ8K<<<nBlocks, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), q8k, out, rows,
                    blocksPerRow, rowBytes);
        }

        // Dequant ONE [rows][cols] quantized matrix into the reusable wF16_
        // scratch (fp16, row-major).  The kernel uses ONE warp per row (the
        // 32 lanes cover all 256 columns of a K-quant block via lane*8), so
        // grid = rows, block = 32.
        void launchDequantF16(int type, const void *wq, void *out,
                              uint32_t rows, uint32_t cols, uint32_t rowBytes,
                              uint32_t blocksPerRow) {
            dim3 blk(32);
            if (type == kTypeQ2K) {
                kDequantF16<kTypeQ2K><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeQ3K) {
                kDequantF16<kTypeQ3K><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeQ4K) {
                kDequantF16<kTypeQ4K><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeQ6K) {
                kDequantF16<kTypeQ6K><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeIQ2XS) {
                kDequantF16<kTypeIQ2XS><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeIQ2XXS) {
                kDequantF16<kTypeIQ2XXS><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeIQ4XS) {
                kDequantF16<kTypeIQ4XS><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeIQ3S) {
                kDequantF16<kTypeIQ3S><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeIQ2S) {
                kDequantF16<kTypeIQ2S><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeIQ3_XXS) {
                kDequantF16<kTypeIQ3_XXS><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeQ5K) {
                kDequantF16<kTypeQ5K><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeQ5_0) {
                kDequantF16<kTypeQ5_0><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeQ8_0) {
                kDequantF16<kTypeQ8_0><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else if (type == kTypeIQ4NL) {
                kDequantF16<kTypeIQ4NL><<<rows, blk, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq),
                        static_cast<__half *>(out), rows, cols, blocksPerRow,
                        rowBytes);
            } else {
                cudaMemsetAsync(out, 0,
                                static_cast<size_t>(rows) * cols * sizeof(__half),
                                g_stream);
            }
        }

        // Token embedding dequant (quantized Q2_K rows, one warp per token).
        __global__ void kEmbedDequant(const uint8_t *__restrict__ embed,
                                      const int32_t *__restrict__ tokens,
                                      float *__restrict__ hidden, uint32_t seqLen,
                                      uint32_t hiddenSize, uint32_t blocksPerRow,
                                      uint32_t rowBytes, uint32_t vocabSize) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            int32_t t = tokens[s];
            if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
            const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
            float *h = hidden + static_cast<size_t>(s) * hiddenSize;
            const uint32_t lane = threadIdx.x;
            const uint32_t e0 = lane * 8u;
            const uint32_t half = e0 / 128u, jj = (e0 % 128u) / 32u,
                           sub = (e0 % 32u) / 16u, l16 = e0 % 16u;
            const uint32_t is = half * 8u + jj * 2u + sub;
            // q indexing matches dequantizeQ2_KBlock: same 32 q bytes per half,
            // reused for all four shifts.
            const uint32_t qbase = half * 32u + sub * 16u + l16;
            const uint32_t shift = jj * 2u;
            const uint32_t typeBytes = rowBytes / blocksPerRow;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                __half d = *reinterpret_cast<const __half *>(row + 80);
                __half dmin = *reinterpret_cast<const __half *>(row + 82);
                float df = __half2float(d), dmf = __half2float(dmin);
                float dl = df * static_cast<float>(row[is] & 0xF);
                float ml = dmf * static_cast<float>(row[is] >> 4);
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    // Bound the write to the row width.  blocksPerRow pads up
                    // to the next 256-col block, so the last block's columns
                    // [b*256, b*256+256) can exceed hiddenSize; writing past it
                    // corrupts the next token's row (or the buffer end) and
                    // triggers an illegal memory access on non-256-multiple
                    // hidden sizes.
                    const uint32_t col = b * 256u + e0 + k;
                    if (col < hiddenSize) {
                        float qv = static_cast<float>(static_cast<int8_t>(
                                ((row[16 + qbase + k]) >> shift) & 3));
                        h[col] = fmaf(dl, qv, -ml);
                    }
                }
                row += typeBytes;
            }
        }

        // Token embedding dequant for IQ3_S rows (one warp per token).  Layout
        // matches dequantizeIQ3_SBlock (110 B blocks).  Each lane owns 8
        // consecutive columns; exactly one (ib32, l) group.  The sign byte
        // for a (ib32, l) group is signs[pair*8 + half*4 + l] and the qh byte
        // is qh[pair*2 + half] -- both advance as in the reference.
        __global__ void kEmbedDequantIQ3S(const uint8_t *__restrict__ embed,
                                          const int32_t *__restrict__ tokens,
                                          float *__restrict__ hidden,
                                          uint32_t seqLen, uint32_t hiddenSize,
                                          uint32_t blocksPerRow, uint32_t rowBytes,
                                          uint32_t vocabSize) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            int32_t t = tokens[s];
            if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
            const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
            float *h = hidden + static_cast<size_t>(s) * hiddenSize;
            const uint32_t lane = threadIdx.x;
            const uint32_t e0 = lane * 8u;
            const uint32_t ib32 = e0 >> 5u;
            const uint32_t half = ib32 & 1u;
            const uint32_t pair = ib32 >> 1u;
            const uint32_t l = (e0 >> 3u) & 3u;
            const uint32_t qhIdx = pair * 2u + half;
            const uint32_t qsOff = pair * 16u + half * 8u;
            const uint32_t signOff = pair * 8u + half * 4u;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                __half d = *reinterpret_cast<const __half *>(row + 0);
                float df = __half2float(d);
                float db = df *
                           (1.0f +
                            2.0f *
                                    static_cast<float>((half == 0)
                                                               ? (row[106 + pair] & 0xf)
                                                               : (row[106 + pair] >> 4)));
                const uint16_t gridIdx1 = static_cast<uint16_t>(
                        row[2 + qsOff + 2u * l] |
                        ((row[66 + qhIdx] << (8 - 2u * l)) & 256));
                const uint16_t gridIdx2 = static_cast<uint16_t>(
                        row[2 + qsOff + 2u * l + 1u] |
                        ((row[66 + qhIdx] << (7 - 2u * l)) & 256));
                const uint32_t g1 = c_iq3s_grid[gridIdx1];
                const uint32_t g2 = c_iq3s_grid[gridIdx2];
                const uint8_t sbyte = row[74 + signOff + l];
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    const uint32_t col = b * 256u + e0 + j;
                    if (col >= hiddenSize) continue;
                    uint32_t gval = (j < 4u)
                                            ? ((g1 >> (8u * j)) & 0xFFu)
                                            : ((g2 >> (8u * (j - 4u))) & 0xFFu);
                    float w = db * static_cast<float>(gval) *
                              ((sbyte & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    h[col] = w;
                }
                row += kIQ3S_BYTES;
            }
        }

        // Token embedding dequant for Q5_K rows (one warp per token).  Layout
        // matches dequantizeQ5_KBlock (176 B blocks).  Each lane owns 8
        // consecutive columns within one 32-weight sub-block (e0 is a multiple
        // of 8), so the qs/qh byte is g*32 + jj and the sub-block scale/min
        // come from getScaleMin(2*g + sub).
        __global__ void kEmbedDequantQ5K(const uint8_t *__restrict__ embed,
                                         const int32_t *__restrict__ tokens,
                                         float *__restrict__ hidden,
                                         uint32_t seqLen, uint32_t hiddenSize,
                                         uint32_t blocksPerRow, uint32_t rowBytes,
                                         uint32_t vocabSize) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            int32_t t = tokens[s];
            if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
            const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
            float *h = hidden + static_cast<size_t>(s) * hiddenSize;
            const uint32_t lane = threadIdx.x;
            const uint32_t e0 = lane * 8u;
            const uint32_t g = e0 / 64u;
            const uint32_t sub = (e0 / 32u) & 1u;
            const uint32_t jj0 = e0 & 31u;
            auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                  uint8_t *m_out) {
                if (j < 4) {
                    *d_out = q[j] & 63;
                    *m_out = q[j + 4] & 63;
                } else {
                    *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                }
            };
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                __half d = *reinterpret_cast<const __half *>(row + 0);
                __half dmin = *reinterpret_cast<const __half *>(row + 2);
                float df = __half2float(d), dmf = __half2float(dmin);
                const uint8_t *scales = row + 4;
                const uint8_t *qh = row + 16;
                const uint8_t *qs = row + 48;
                uint8_t sc, m;
                getScaleMin(static_cast<int>(2u * g + sub), scales, &sc, &m);
                float dl = df * static_cast<float>(sc);
                float ml = dmf * static_cast<float>(m);
                const uint8_t u =
                        (sub == 0) ? static_cast<uint8_t>(1u << (2u * g))
                                   : static_cast<uint8_t>(2u << (2u * g));
                const uint8_t *qsp = qs + g * 32u;
#pragma unroll
                for (uint32_t kk = 0; kk < 8; ++kk) {
                    uint8_t q5 = (sub == 0)
                                         ? static_cast<uint8_t>(
                                                   (qsp[jj0 + kk] & 0xF) |
                                                   ((qh[jj0 + kk] & u) ? 16 : 0))
                                         : static_cast<uint8_t>(
                                                   (qsp[jj0 + kk] >> 4) |
                                                   ((qh[jj0 + kk] & u) ? 16 : 0));
                    const uint32_t col = b * 256u + e0 + kk;
                    if (col < hiddenSize) h[col] = dl * static_cast<float>(q5) - ml;
                }
                row += kQ5K_BYTES;
            }
        }

        // Token embedding dequant for Q4_K rows (one warp per token).  Layout
        // matches dequantizeQ4_KBlock (144 B blocks: d, dmin, scales[12], qs[128]
        // = 2+2+12+128).  Each lane owns 8 consecutive columns within one
        // 32-weight sub-block (e0 is a multiple of 8), so the qs byte is
        // g*32 + jj0 + k and the sub-block scale/min come from
        // getScaleMin(2*g + sub) -- identical decomposition to kEmbedDequantQ5K.
        __global__ void kEmbedDequantQ4K(const uint8_t *__restrict__ embed,
                                         const int32_t *__restrict__ tokens,
                                         float *__restrict__ hidden,
                                         uint32_t seqLen, uint32_t hiddenSize,
                                         uint32_t blocksPerRow, uint32_t rowBytes,
                                         uint32_t vocabSize) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            int32_t t = tokens[s];
            if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
            const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
            float *h = hidden + static_cast<size_t>(s) * hiddenSize;
            const uint32_t lane = threadIdx.x;
            const uint32_t e0 = lane * 8u;
            const uint32_t g = e0 / 64u;
            const uint32_t sub = (e0 / 32u) & 1u;
            const uint32_t jj0 = e0 & 31u;
            auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                                  uint8_t *m_out) {
                if (j < 4) {
                    *d_out = q[j] & 63;
                    *m_out = q[j + 4] & 63;
                } else {
                    *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                    *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
                }
            };
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                __half d = *reinterpret_cast<const __half *>(row + 0);
                __half dmin = *reinterpret_cast<const __half *>(row + 2);
                float df = __half2float(d), dmf = __half2float(dmin);
                const uint8_t *scales = row + 4;
                const uint8_t *qs = row + 16;
                uint8_t sc, m;
                getScaleMin(static_cast<int>(2u * g + sub), scales, &sc, &m);
                float dl = df * static_cast<float>(sc);
                float ml = dmf * static_cast<float>(m);
                const uint8_t *qsp = qs + g * 32u;
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k) {
                    const uint8_t lo = (sub == 0) ? (qsp[jj0 + k] & 0xF)
                                                  : (qsp[jj0 + k] >> 4);
                    const uint32_t col = b * 256u + e0 + k;
                    if (col < hiddenSize) h[col] = fmaf(dl, static_cast<float>(lo), -ml);
                }
                row += kQ4K_BYTES;
            }
        }

        // Token embedding dequant for the 32-wide-block legacy quants
        // (Q5_0 / Q8_0): one warp per token, each lane writes its own column
        // (b*32 + lane) of every block.  blocksPerRow = ceil(hiddenSize/32),
        // rowBytes = blocksPerRow * blockBytes.
        template<int TYPE>
        __global__ void kEmbedDequantSmall32(const uint8_t *__restrict__ embed,
                                             const int32_t *__restrict__ tokens,
                                             float *__restrict__ hidden,
                                             uint32_t seqLen, uint32_t hiddenSize,
                                             uint32_t blocksPerRow, uint32_t rowBytes,
                                             uint32_t vocabSize) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            int32_t t = tokens[s];
            if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
            const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
            float *h = hidden + static_cast<size_t>(s) * hiddenSize;
            const uint32_t lane = threadIdx.x;
            constexpr uint32_t BLOCK_BYTES = (TYPE == kTypeQ5_0)
                                                     ? kQ5_0_BYTES
                                             : (TYPE == kTypeQ8_0) ? kQ8_0_BYTES
                                                                   : kIQ4NL_BYTES;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blk = row + static_cast<uint64_t>(b) * BLOCK_BYTES;
                const uint32_t col = b * 32u + lane;
                if (col >= hiddenSize) break;
                __half d = *reinterpret_cast<const __half *>(blk);
                if (TYPE == kTypeQ8_0) {
                    const int8_t q = reinterpret_cast<const int8_t *>(blk + 2)[lane];
                    h[col] = __half2float(d) * static_cast<float>(q);
                } else if (TYPE == kTypeIQ4NL) {
                    // IQ4_NL (4.5 bpw): d(fp16,2) + qs[16].  Element `lane`:
                    //   j   = lane & 15
                    //   nib = (lane < 16) ? qs[j] & 0x0F : qs[j] >> 4
                    //   wv  = d * kvalues_iq4nl[nib]   (signed 8-bit table)
                    // Matches CPU dequantizeIQ4_NLBlock.
                    const uint8_t *qs = blk + 2;
                    const uint32_t j = lane & 15u;
                    const uint8_t nib =
                            (qs[j] >> (4u * (lane >> 4u))) & 0xFu;
                    h[col] = __half2float(d) *
                             static_cast<float>(c_kvalues_iq4nl[nib]);
                } else {// Q5_0 (matches CPU dequantizeQ5_0Block)
                    // Element `lane` of the 32-wide block (QK5_0=32):
                    //   j      = lane & 15
                    //   low    = (lane < 16) ? qs[j] & 0x0F : qs[j] >> 4
                    //   high   = bit j of qh for lane < 16, else bit (j+16)
                    //   val    = (low | (high << 4)) - 16
                    // NOTE: the CPU reference reads the upper-half high bit via
                    // `((qh >> (j + 12)) & 0x10)` -- the 0x10 mask selects bit 4
                    // of the shifted word, i.e. ORIGINAL bit (j+16) of qh, NOT
                    // bit (j+12) (the surrounding comment in GGMLDequantize.hpp
                    // is misleading).  Use (j+16) here to match the CPU math.
                    uint32_t qh;
                    std::memcpy(&qh, blk + 2, sizeof(uint32_t));
                    const uint8_t *qs = blk + 6;
                    const uint32_t j = lane & 15u;
                    const uint8_t low =
                            (qs[j] >> (4u * (lane >> 4u))) & 0xFu;
                    const uint32_t hb = (lane & 16u) ? (j + 16u) : j;
                    const uint8_t high =
                            static_cast<uint8_t>((qh >> hb) & 1u);
                    const int val = static_cast<int>(low | (high << 4)) - 16;
                    h[col] = __half2float(d) * static_cast<float>(val);
                }
            }
        }

        // Number of blocks per row for a given quant type and row width, and
        // the corresponding row byte stride.  Mirrors the host-side layout in
        // ModelGPU.cpp so both sides agree on where each block lives.
        uint32_t hostBlocksPerRow(uint32_t type, uint32_t cols) {
            if (type == kTypeQ5_0 || type == kTypeQ8_0 ||
                type == kTypeIQ4NL) {
                return (cols + 31) / 32;// 32-wide legacy blocks
            }
            if (type == kTypePQ2_0) {
                return (cols + 127) / 128;// PQ2_0 (Prism ternary) 128-wide blocks
            }
            return (cols + 255) / 256;// K-quant blocks (256 wide)
        }

        // ====================================================================
        // Qwen35 (dense gated-delta-net) kernels.
        //
        // The MTP block (layer >= n_layer) is never executed in the main decode
        // pass (the CPU forward skips it), so kernels only process the real
        // layers. All matmuls re-use the existing launchQGemv / cuBLAS fp16
        // GEMM machinery; only the qwen35-specific glue is new.
        // ====================================================================

        __forceinline__ __device__ float q35Sigmoid(float x) {
            return 1.0f / (1.0f + __expf(-x));
        }

        __forceinline__ __device__ float q35Softplus(float x) {
            return x > 20.0f ? x : log1pf(__expf(x));
        }

        // RMSNorm over a slice of length `n` with weight `w`, results OUT-OF-
        // PLACE into `out`.  One block per (token, head) slice, 128 threads.
        // Grid: total = number of [n]-wide slices (= seqLen * heads).
        __global__ void kQ35HeadRmsNorm(const float *in, float *out,
                                        const float *__restrict__ w, uint32_t n,
                                        uint32_t total) {
            if (blockIdx.x >= total) return;
            // per-element: out[i] = rms * w[i % n] * in[i]
            extern __shared__ float ssum[];
            uint32_t lane = threadIdx.x;
            // One block covers a contiguous [n]-wide slice (total == k*n).
            uint32_t chunk = blockIdx.x;
            const uint32_t base = chunk * n;
            const float *rp = in + base;
            float *op = out + base;
            float acc = 0.0f;
            for (uint32_t j = lane; j < n; j += blockDim.x) {
                float v = rp[j];
                acc = fmaf(v, v, acc);
            }
            ssum[lane] = acc;
            __syncthreads();
            for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (lane < s) ssum[lane] += ssum[lane + s];
                __syncthreads();
            }
            float rms = rsqrtf(ssum[0] / static_cast<float>(n) + 1e-6f);
            for (uint32_t j = lane; j < n; j += blockDim.x) {
                op[j] = rp[j] * rms * w[j];
            }
        }

        // RMSNorm over a STRIDED slice: the qwen35 fused Q+gate projection puts
        // head h's Q slice at [h*2*headDim .. h*2*headDim+headDim).  One block
        // per (token, head) slice reads `in[base + j*stride]`, writes
        // `out[base + j]` (dense).  K (attn_k_norm) uses stride=1 and is served
        // by the plain kQ35HeadRmsNorm above.
        __global__ void kQ35HeadRmsNorm2(const float *__restrict__ in,
                                         float *__restrict__ out,
                                         const float *__restrict__ w,
                                         uint32_t n, uint32_t stride,
                                         uint32_t total) {
            if (blockIdx.x >= total) return;
            extern __shared__ float ssum[];
            float acc = 0.0f;
            uint32_t lane = threadIdx.x;
            const float *rp = in + static_cast<size_t>(blockIdx.x) * stride;
            for (uint32_t j = lane; j < n; j += blockDim.x) {
                float v = rp[j];
                acc = fmaf(v, v, acc);
            }
            ssum[lane] = acc;
            __syncthreads();
            for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (lane < s) ssum[lane] += ssum[lane + s];
                __syncthreads();
            }
            float rms = rsqrtf(ssum[0] / static_cast<float>(n) + 1e-6f);
            float *op = out + static_cast<size_t>(blockIdx.x) * n;
            for (uint32_t j = lane; j < n; j += blockDim.x) {
                op[j] = rp[j] * rms * w[j];
            }
        }

        // Batched sigmoid(beta) + softplus(alpha)+dt.bias scaled by ssm_a,
        // producing the per-value-head decay gate.  One block per (token),
        // one thread per value head.
        __global__ void kQ35BetaGate(const float *betaIn, const float *alphaIn,
                                     const float *__restrict__ dtBias,
                                     const float *__restrict__ aBroadcast,
                                     float *betaOut, float *gateVOut,
                                     uint32_t seqLen, uint32_t nVHeads) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            const float *br = betaIn + static_cast<size_t>(s) * nVHeads;
            const float *ar = alphaIn + static_cast<size_t>(s) * nVHeads;
            float *bo = betaOut + static_cast<size_t>(s) * nVHeads;
            float *go = gateVOut + static_cast<size_t>(s) * nVHeads;
            for (uint32_t v = threadIdx.x; v < nVHeads; v += blockDim.x) {
                bo[v] = q35Sigmoid(br[v]);
                float alpha = q35Softplus(ar[v] + dtBias[v]);
                go[v] = alpha * aBroadcast[v];
            }
        }

        // MRoPE on one (q or k) tensor.  One block per (token, head), 32 lanes
        // per half.  Mirrors Model::applyMRoPE: pairs (j, j+nDimsHalf) rotated
        // with cache index j; channels [nDims, headDim) untouched.  The cache
        // entries were precomputed on the host with theta_scale =
        // freq_base^(-2/n_dims) seeded from theta=1 (matches the CPU path).
        __global__ void kQ35MRoPE(float *__restrict__ t, uint32_t seqLen,
                                  uint32_t heads, uint32_t headDim,
                                  uint32_t nDims, uint32_t pos,
                                  const float *__restrict__ cosT,
                                  const float *__restrict__ sinT) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            uint32_t h = blockIdx.y;
            if (h >= heads) return;
            const uint32_t p = pos + s;
            const uint32_t half = nDims / 2;
            float *head = t + (static_cast<size_t>(s) * heads + h) * headDim;
            const float *c = cosT + static_cast<size_t>(p) * half;
            const float *sn = sinT + static_cast<size_t>(p) * half;
            for (uint32_t j = threadIdx.x; j < half; j += blockDim.x) {
                float cc = c[j], ss = sn[j];
                float x0 = head[j], x1 = head[j + half];
                head[j] = fmaf(x0, cc, -x1 * ss);
                head[j + half] = fmaf(x0, ss, x1 * cc);
            }
        }

        // L2-norm of [headK]-wide slices repeated over value heads, then copy v.
        // Each of the nVHeads value heads reads k-head (hv % nKHeads).  One block
        // per token (128 threads).
        // `smemFast` (2026-10-29, TINYCODER_L2NORM_SMEM, default on): the
        // original body recomputed the SAME headK-wide double-precision L2 norm
        // for every one of the headK*nVHeads elements (an O(headK^2 * nVHeads)
        // double-add chain -- the per-slice norm depends only on hv, not on the
        // element index d).  On the dense qwen35 prefill this was the single
        // hottest kernel (nvprof: 4286 us / 6528 calls, #1 above the Q4_K
        // projections).  The fast path computes each slice's iq/ik ONCE (one
        // thread per value head, shared memory broadcast) using the SAME
        // j=0..headK-1 double accumulation order, so iq/ik -- and therefore
        // every qDst/kDst element -- are bit-identical.  nVHeads > 128 falls
        // back to the original recompute body.
        __global__ void kQ35L2NormRepeat(const float *__restrict__ convOut,
                                         float *__restrict__ qN, float *__restrict__ kN,
                                         float *__restrict__ vN, uint32_t headK,
                                         uint32_t headV, uint32_t nKHeads,
                                         uint32_t nVHeads, uint32_t keyDim,
                                         uint32_t valueDim, uint32_t seqLen,
                                         uint32_t smemFast) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            const float *conv = convOut + static_cast<size_t>(s) *
                                                  (2 * keyDim + valueDim);
            const float *convQ = conv;
            const float *convK = conv + keyDim;
            const float *convV = conv + 2 * keyDim;
            float *qDst = qN + static_cast<size_t>(s) * headK * nVHeads;
            float *kDst = kN + static_cast<size_t>(s) * headK * nVHeads;
            float *vDst = vN + static_cast<size_t>(s) * headV * nVHeads;
            const bool fast = (smemFast != 0u) && (nVHeads <= 128u);
            __shared__ float siq[128];
            __shared__ float sik[128];
            if (fast) {
                for (uint32_t hv = threadIdx.x; hv < nVHeads; hv += blockDim.x) {
                    uint32_t hk = hv % nKHeads;
                    const float *qSrc = convQ + static_cast<size_t>(hk) * headK;
                    const float *kSrc = convK + static_cast<size_t>(hk) * headK;
                    double sq = 0.0, sk = 0.0;
                    for (uint32_t j = 0; j < headK; ++j) {
                        float qv = qSrc[j], kv = kSrc[j];
                        sq += static_cast<double>(qv) * qv;
                        sk += static_cast<double>(kv) * kv;
                    }
                    siq[hv] = static_cast<float>(
                            1.0 / std::sqrt(fmax(sq, 1e-6)));
                    sik[hv] = static_cast<float>(
                            1.0 / std::sqrt(fmax(sk, 1e-6)));
                }
                __syncthreads();
            }
            for (uint32_t i = threadIdx.x; i < headK * nVHeads; i += blockDim.x) {
                uint32_t hv = i / headK;
                uint32_t d = i - hv * headK;
                float iq, ik;
                if (fast) {
                    iq = siq[hv];
                    ik = sik[hv];
                } else {
                    uint32_t hk = hv % nKHeads;
                    const float *qSrc = convQ + static_cast<size_t>(hk) * headK;
                    const float *kSrc = convK + static_cast<size_t>(hk) * headK;
                    double sq = 0.0, sk = 0.0;
                    for (uint32_t j = 0; j < headK; ++j) {
                        float qv = qSrc[j], kv = kSrc[j];
                        sq += static_cast<double>(qv) * qv;
                        sk += static_cast<double>(kv) * kv;
                    }
                    iq = static_cast<float>(
                            1.0 / std::sqrt(fmax(sq, 1e-6)));
                    ik = static_cast<float>(
                            1.0 / std::sqrt(fmax(sk, 1e-6)));
                }
                uint32_t hk = hv % nKHeads;
                const float *qSrc = convQ + static_cast<size_t>(hk) * headK;
                const float *kSrc = convK + static_cast<size_t>(hk) * headK;
                qDst[i] = qSrc[d] * iq;
                kDst[i] = kSrc[d] * ik;
            }
            // v: [headV, nVHeads] copied verbatim.
            for (uint32_t i = threadIdx.x; i < headV * nVHeads; i += blockDim.x) {
                vDst[i] = convV[i];
            }
        }

        // Gated delta net recurrence: one block per (token, value-head), 128
        // threads.  Per value-head state M is [headV x headV] floats stored
        // transposed (M[j*headV+i] = S[i][j], matching ggml).  The recurrence
        // for ONE token:
        //   M *= exp(gateV[hv])                                (decay)
        //   delta[j] = (v[j] - dot(row_j(M), k)) * beta[hv]
        //   row_j(M) += delta[j] * k
        //   out[j] = dot(row_j(M), q) * invSqrt(headV)
        // The state read-modify-write is inherently sequential across j, so a
        // single block owns one head and threads cooperatively process rows in
        // [0, headV) with a write-visibility barrier (__threadfence_block) --
        // each row j is handled by thread t = j (headV=128 => 128 threads).
        template<uint32_t HEADV>
        __global__ void kQ35GatedDeltaNet(const float *__restrict__ qN,
                                          const float *__restrict__ kN,
                                          const float *__restrict__ vN,
                                          const float *__restrict__ beta,
                                          const float *__restrict__ decayV,
                                          float *__restrict__ gdnState,
                                          float *__restrict__ attnOut,
                                          uint32_t nVHeads, uint32_t seqLen,
                                          float invSqrtHeadV) {
            static_assert(HEADV == 128, "specialized for headV=128");
            uint32_t s = blockIdx.x;
            uint32_t hv = blockIdx.y;
            if (s >= seqLen || hv >= nVHeads) return;
            const uint32_t row = threadIdx.x;// j
            const float dec = decayV[hv];
            const float *qHead = qN + (static_cast<size_t>(s) * nVHeads + hv) * HEADV;
            const float *kHead = kN + (static_cast<size_t>(s) * nVHeads + hv) * HEADV;
            const float *vHead = vN + (static_cast<size_t>(s) * nVHeads + hv) * HEADV;
            float *state = gdnState + (static_cast<size_t>(hv) * HEADV + row) * HEADV;
            for (uint32_t i = 0; i < HEADV; ++i) {
                state[i] *= dec;
            }
            __syncthreads();
            // delta[row] = (v[row] - dot(state_row, k)) * beta[hv]
            float acc = 0.0f;
            for (uint32_t i = 0; i < HEADV; ++i) {
                acc = fmaf(state[i], kHead[i], acc);
            }
            float delta = (vHead[row] - acc) * beta[hv];
            // row_j(M) += delta[j] * k  (write to the SAME row j only, so no
            // cross-row race) then out[j] = dot(row_j(M), q) * invSqrt.
            float outRow = 0.0f;
            for (uint32_t i = 0; i < HEADV; ++i) {
                float m = state[i] + delta * kHead[i];
                state[i] = m;
                outRow = fmaf(m, qHead[i], outRow);
            }
            attnOut[(static_cast<size_t>(s) * nVHeads + hv) * HEADV + row] =
                    outRow * invSqrtHeadV;
        }

        // Gated RMSNorm + silu(z): out[i] = rmsnorm(out, ssm_norm)[i] * silu(z[i]).
        // One block per (token), 256 threads; per value-head slice normalized
        // with the shared [headV] weight.
        __global__ void kQ35GatedRmsNorm(const float *__restrict__ z,
                                         float *__restrict__ out,
                                         const float *__restrict__ ssmNorm,
                                         uint32_t headV, uint32_t nVHeads,
                                         uint32_t seqLen) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            const uint32_t t = threadIdx.x;
            const float *zRow = z + static_cast<size_t>(s) * headV * nVHeads;
            float *oRow = out + static_cast<size_t>(s) * headV * nVHeads;
            // compute the per-slice sums into shared
            extern __shared__ float ssum[];
            for (uint32_t hv = 0; hv < nVHeads; ++hv) {
                float *slice = oRow + static_cast<size_t>(hv) * headV;
                float acc = 0.0f;
                for (uint32_t i = t; i < headV; i += blockDim.x) {
                    float v = slice[i];
                    acc = fmaf(v, v, acc);
                }
                ssum[t] = acc;
                __syncthreads();
                for (uint32_t st = blockDim.x / 2; st > 0; st >>= 1) {
                    if (t < st) ssum[t] += ssum[t + st];
                    __syncthreads();
                }
                float rms = rsqrtf(ssum[0] / static_cast<float>(headV) + 1e-6f);
                for (uint32_t i = t; i < headV; i += blockDim.x) {
                    float sv = zRow[static_cast<size_t>(hv) * headV + i];
                    float silu = sv * q35Sigmoid(sv);
                    slice[i] = slice[i] * rms * ssmNorm[i] * silu;
                }
                __syncthreads();
            }
        }

        // Sigmoid-gate multiply for qwen35 full-attention: attnOut *= sigmoid(gate)
        // where gate is the SECOND half of each head's fused Q+gate projection
        // (qGate layout: [seqLen][h][2*headDim], gate slice = +headDim).
        __global__ void kQ35SigmoidGate(float *__restrict__ attnOut,
                                        const float *__restrict__ qGate,
                                        uint32_t seqLen, uint32_t nHeads,
                                        uint32_t headDim) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            const uint32_t t = threadIdx.x;
            float *oRow = attnOut + static_cast<size_t>(s) * nHeads * headDim;
            const float *gRow = qGate + static_cast<size_t>(s) * nHeads * 2 * headDim;
            for (uint32_t h = t / headDim; h < nHeads; h += blockDim.x / headDim) {
                uint32_t e = t % headDim;
                float g = gRow[static_cast<size_t>(h) * 2 * headDim + headDim + e];
                oRow[static_cast<size_t>(h) * headDim + e] *= q35Sigmoid(g);
            }
        }

        // Store K/V into the qwen35 full-attention KV cache (MRoPE already
        // applied to k).  One block per token, 256 threads.
        __global__ void kQ35StoreKV(const float *__restrict__ k,
                                    const float *__restrict__ v,
                                    float *__restrict__ kDst,
                                    float *__restrict__ vDst,
                                    uint32_t seqLen, uint32_t nKVHeads,
                                    uint32_t headDim, uint32_t cachePos) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            const uint32_t kvSize = nKVHeads * headDim;
            const float *ks = k + static_cast<size_t>(s) * kvSize;
            const float *vs = v + static_cast<size_t>(s) * kvSize;
            float *kd = kDst + static_cast<size_t>(cachePos + s) * kvSize;
            float *vd = vDst + static_cast<size_t>(cachePos + s) * kvSize;
            for (uint32_t e = threadIdx.x; e < kvSize; e += blockDim.x) {
                kd[e] = ks[e];
                vd[e] = vs[e];
            }
        }

        // Plain F32 GEMV: out[row] = dot(x[0..cols), W[row*cols..]) for the
        // small qwen35 F32 matrices (ssm_alpha / ssm_beta are [rows=nVHeads,
        // cols=hiddenSize]).  One block per row, 256 threads.
        __global__ void kQ35F32Gemv(const float *__restrict__ x,
                                    const float *__restrict__ w,
                                    float *__restrict__ out, uint32_t rows,
                                    uint32_t cols) {
            uint32_t row = blockIdx.x;
            if (row >= rows) return;
            const float *wRow = w + static_cast<size_t>(row) * cols;
            float acc = 0.0f;
            for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
                acc = fmaf(x[i], wRow[i], acc);
            }
            // warp-reduce
            for (uint32_t off = 16; off; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }
            __shared__ float srow[32];
            if (threadIdx.x % 32 == 0) srow[threadIdx.x / 32] = acc;
            __syncthreads();
            if (threadIdx.x == 0) {
                float s = 0.0f;
                for (uint32_t wq = 0; wq < blockDim.x / 32; ++wq) s += srow[wq];
                out[row] = s;
            }
        }

        // ------------------------------------------------------------------
        // Qwen35MoE (qwen35moe) MoE FFN kernels.
        //
        // Output for token s: moeIdx[s][r]   expert id of rank r (top-k)
        // moeWgt[s][r]   renormalized weight of rank r moeWsum[s]     max(wsum,
        // 6.103515625e-5) (for debug / scale)
        // ------------------------------------------------------------------
        __global__ void kQ35MoeRouter(const float *__restrict__ logits,
                                      int32_t *__restrict__ moeIdx,
                                      float *__restrict__ moeWgt,
                                      float *__restrict__ moeWsum,
                                      uint32_t seqLen, uint32_t expertCount,
                                      uint32_t expertUsed) {
            const uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            const float *lk = logits + static_cast<size_t>(s) * expertCount;
            const uint32_t t = threadIdx.x;
            const uint32_t warp = t >> 5u;// 0..7 (8 warps)
            const uint32_t lane = t & 31u;// 0..31

            // ---- stage logits to shared, find max ----
            extern __shared__ float slogits[];// dynamic: expertCount floats
            float vmax = -3.402823e38f;
            for (uint32_t e = t; e < expertCount; e += blockDim.x) {
                slogits[e] = lk[e];
                vmax = fmaxf(vmax, lk[e]);
            }
            // block-reduce max (8 warps)
            for (uint32_t off = 16; off; off >>= 1)
                vmax = fmaxf(vmax, __shfl_down_sync(0xffffffffu, vmax, off));
            __shared__ float smax[8];
            if (lane == 0) smax[warp] = vmax;
            __syncthreads();
            if (t == 0) {
                vmax = smax[0];
                for (uint32_t wq = 1; wq < 8; ++wq) vmax = fmaxf(vmax, smax[wq]);
                smax[0] = vmax;
            }
            __syncthreads();
            vmax = smax[0];

            // ---- softmax over ALL experts (double sum like the CPU path) ----
            // CPU reference accumulates exp() in double; the GPU uses float
            // expf (a few ulp apart) -- acceptable for a router: ties are
            // broken identically by the strength of the top-k separation.
            double dsum = 0.0;
            for (uint32_t e = t; e < expertCount; e += blockDim.x) {
                float p = __expf(slogits[e] - vmax);
                slogits[e] = p;
                dsum += static_cast<double>(p);
            }
            // block-reduce the double sum
            __syncthreads();
            double dpart = dsum;
            for (uint32_t off = 16; off; off >>= 1)
                dpart += __shfl_down_sync(0xffffffffu, dpart, off);
            __shared__ double ssum[8];
            if (lane == 0) ssum[warp] = dpart;
            __syncthreads();
            if (t == 0) {
                double acc = 0.0;
                for (uint32_t wq = 0; wq < 8; ++wq) acc += ssum[wq];
                ssum[0] = acc;
            }
            __syncthreads();
            const double invSum = 1.0 / ssum[0];
            for (uint32_t e = t; e < expertCount; e += blockDim.x) {
                slogits[e] = static_cast<float>(slogits[e] * invSum);
            }
            __syncthreads();

            // ---- top-k by probability, k = expertUsed (8) ----
            // Serial insertion per warp is fine: 256 experts, 8 warps => 32
            // experts each, then warp 0 merges the 8 partial top-lists.  For
            // simplicity (correctness first) we do the FULL serial selection
            // in warp 0 (256*8 compares is trivial vs. the expert GEMVs).
            if (t == 0) {
                // serial insertion sort in a local top-k list (expertUsed <= 8);
                // 256 experts x 8 slots is trivial vs. the per-expert GEMVs that
                // follow.  Ties (equal probability) keep the lower expert index:
                // the insertion replaces the FIRST slot with a strictly
                // smaller-or-equal value, so a later expert never displaces an
                // earlier one at equal probability -- identical to the CPU
                // std::partial_sort stable-by-index behavior for equal probs.
                float topBest[8];
                int32_t topIdx[8];
                for (uint32_t r = 0; r < expertUsed; ++r) {
                    topBest[r] = -3.402823e38f;
                    topIdx[r] = -1;
                }
                for (uint32_t e = 0; e < expertCount; ++e) {
                    const float p = slogits[e];
                    // insertion into the descending top-k list
                    for (uint32_t r = 0; r < expertUsed; ++r) {
                        if (p > topBest[r]) {
                            for (uint32_t rr = expertUsed - 1; rr > r; --rr) {
                                topBest[rr] = topBest[rr - 1];
                                topIdx[rr] = topIdx[rr - 1];
                            }
                            topBest[r] = p;
                            topIdx[r] = static_cast<int32_t>(e);
                            break;
                        }
                    }
                }
                float wsum = 0.0f;
                for (uint32_t r = 0; r < expertUsed; ++r) {
                    wsum += topBest[r];
                }
                const float wsumClamped =
                        fmaxf(wsum, 6.103515625e-5f);// norm_w clamp
                for (uint32_t r = 0; r < expertUsed; ++r) {
                    const size_t off = static_cast<size_t>(s) * expertUsed + r;
                    moeIdx[off] = topIdx[r];
                    moeWgt[off] = topBest[r] / wsumClamped;
                }
                moeWsum[s] = wsumClamped;
            }
        }

        // Shared-expert gate: g = sigmoid(ffn_gate_inp_shexp @ x) for every
        // token.  ffn_gate_inp_shexp is F32 [1 x hidden].  One block per token.
        __global__ void kQ35MoeShexpGate(const float *__restrict__ x,
                                         const float *__restrict__ w,
                                         float *__restrict__ gateOut,
                                         uint32_t seqLen, uint32_t hidden,
                                         uint32_t cols) {
            const uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            const float *xr = x + static_cast<size_t>(s) * hidden;
            // w is a single row of `cols` floats (cols == hidden)
            float acc = 0.0f;
            for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
                acc = fmaf(xr[i], w[i], acc);
            }
            for (uint32_t off = 16; off; off >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            __shared__ float sg[32];
            if (threadIdx.x % 32 == 0) sg[threadIdx.x / 32] = acc;
            __syncthreads();
            if (threadIdx.x == 0) {
                float sum = 0.0f;
                for (uint32_t wq = 0; wq < blockDim.x / 32; ++wq) sum += sg[wq];
                gateOut[s] = 1.0f / (1.0f + __expf(-sum));// sigmoid
            }
        }

        // Accumulate: out[s] += gate[s] * src[s] over hiddenSize.  Uses plain
        // mul+add (NOT fmaf) to stay bit-identical with the CPU reference
        // (computeQwen35MoE: expertOut[i] += weight * down[i]) — fmaf rounds
        // the fused product+add once instead of twice, diverging ~1 ULP per
        // element and compounding through 40 layers of decode into a different
        // argmax on knife-edge top-k rows (measured GPU-vs-CPU logits
        // divergence 2026-09-13).
        __global__ void kAddScaled(const float *__restrict__ gate,
                                   const float *__restrict__ src,
                                   float *__restrict__ out, uint32_t seqLen,
                                   uint32_t hidden) {
            const uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            const float g = gate[s];
            float *o = out + static_cast<size_t>(s) * hidden;
            const float *x = src + static_cast<size_t>(s) * hidden;
            for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
                o[i] = g * x[i] + o[i];
            }
        }

        // Single-token scaled accumulate (MoE per-token fast path): out += g*src
        // over n elements.  One block, 256 threads.  Plain mul+add to match the
        // CPU reference accumulation (see kAddScaled).
        __global__ void kAddScaled1(const float g, const float *__restrict__ src,
                                    float *__restrict__ out, uint32_t n) {
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                out[i] = g * src[i] + out[i];
            }
        }

        // MoE per-expert weighted accumulate: out[s] += wgt[s*stride + r] *
        // src[s].  One block per token; wgt is the per-(token, rank) routing
        // weight array laid out [seqLen][expertUsed] (r = fixed rank).  Plain
        // mul+add to match the CPU reference (see kAddScaled).
        __global__ void kQ35MoeAccumScaled(const float *__restrict__ wgt,
                                           uint32_t wgtStride,
                                           const float *__restrict__ src,
                                           float *__restrict__ out,
                                           uint32_t seqLen, uint32_t hidden) {
            const uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            const float g = wgt[static_cast<size_t>(s) * wgtStride];
            float *o = out + static_cast<size_t>(s) * hidden;
            const float *x = src + static_cast<size_t>(s) * hidden;
            for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
                // Strict mul-then-add (no FMA contraction): the CPU reference
                // accumulate is expertOut[i] += weight*down[i] (float round of
                // the product, then float round of the sum).  See the analysis
                // on kQ35MoeAccumScaledBatch — nvcc -fmad=true would otherwise
                // fuse g*x[i]+o[i] into ONE rounded fmaf and diverge 1 ULP per
                // element from the CPU path (root-cause of the parity break in
                // BOTH the batched fast path and this generic per-rank path).
                o[i] = __fadd_rn(__fmul_rn(g, x[i]), o[i]);
            }
        }

        // Conv1d step for ONE token (the qwen35 recurrent path): builds the
        // sliding-window input [state(K-1) | qkv] in convIn, computes
        // conv_out = silu(conv1d(qkv)), and shifts the persistent conv state
        // window to [t-(K-2) .. t].  Grid-stride over output channels; thread c
        // owns channel c exclusively (its state window row is private).
        __global__ void kQ35ConvStep(const float *__restrict__ convWeight,
                                     const float *__restrict__ qkv,
                                     float *__restrict__ convIn,
                                     float *__restrict__ convOut,
                                     float *__restrict__ convState,
                                     uint32_t channels, uint32_t K) {
            const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= channels) return;
            // convIn[0..K-2] = convState[0..K-2] (window shift), convIn[K-1] = qkv
            // (single block: all threads see the same shift, per-channel).
            for (uint32_t i = 0; i < K - 1; ++i) {
                convIn[i * channels + c] = convState[i * channels + c];
            }
            convIn[(K - 1) * channels + c] = qkv[c];
            // conv1d (kernel-major weight [c][i])
            const float *wRow = convWeight + static_cast<size_t>(c) * K;
            float sum = 0.0f;
            for (uint32_t i = 0; i < K; ++i) {
                sum = fmaf(wRow[i], convIn[i * channels + c], sum);
            }
            // silu
            convOut[c] = sum * q35Sigmoid(sum);
            // shift state: state[i] = state[i+1] for i in [0, K-2), state[K-2] = qkv
            for (uint32_t i = 0; i < K - 2; ++i) {
                convState[i * channels + c] = convState[(i + 1) * channels + c];
            }
            if (K >= 2) convState[(K - 2) * channels + c] = qkv[c];
        }

        // Gated-delta-net state update + output for ONE token, ONE value head.
        // 128 threads, each owning row j of the [headV][headV] state (row j
        // maps to thread j).  Matches forwardQwen35Recurrent's M layout
        // (M[j*Sv + i] = S[i][j]).
        template<uint32_t HEADV>
        __global__ void kQ35GdnStep(const float *__restrict__ qHead,
                                    const float *__restrict__ kHead,
                                    const float *__restrict__ vHead,
                                    const float *__restrict__ beta,
                                    const float *__restrict__ decayV,
                                    float *__restrict__ state,
                                    float *__restrict__ outHead,
                                    uint32_t nVHeads, uint32_t hv,
                                    float invSqrtHeadV) {
            const uint32_t j = threadIdx.x;
            if (j >= HEADV) return;
            constexpr uint32_t Sv = HEADV;
            float *rowJ = state + static_cast<size_t>(j) * Sv;
            // decay S *= exp(gateV[hv])
            const float dec = __expf(decayV[hv]);
            for (uint32_t i = 0; i < Sv; ++i) rowJ[i] *= dec;
            __syncthreads();
            // delta[j] = (v[j] - dot(row_j(M), k)) * beta[hv]
            float dot = 0.0f;
            for (uint32_t i = 0; i < Sv; ++i) dot = fmaf(rowJ[i], kHead[i], dot);
            const float del = (vHead[j] - dot) * beta[hv];
            // M[j][i] += delta[j] * k[i]  (only row j touched)
            for (uint32_t i = 0; i < Sv; ++i) rowJ[i] = fmaf(del, kHead[i], rowJ[i]);
            // out[j] = dot(row_j(M), q) * scale
            float outRow = 0.0f;
            for (uint32_t i = 0; i < Sv; ++i) outRow = fmaf(rowJ[i], qHead[i], outRow);
            outHead[j] = outRow * invSqrtHeadV;
            (void) nVHeads;
            (void) qHead;
            (void) vHead;
        }

        // Batched gated-delta-net step: ONE launch with grid.x = nVHeads instead
        // of nVHeads grid(1) launches.  Each block (hv = blockIdx.x) computes the
        // identical math as kQ35GdnStep<HEADV> for ONE value head -- the heads
        // are fully independent (each has its own private [headV][headV] state
        // and output slice), so block-parallel execution is bit-exact with the
        // sequential loop.  Replaces the per-head launch loop (48 grid(1)
        // launches per recurrent layer per token; at parked idle clocks each
        // launch carries multi-µs latency that inflated the per-layer stream
        // timeline ~35 ms).
        template<uint32_t HEADV>
        __global__ void kQ35GdnStepBatch(const float *__restrict__ qN,
                                         const float *__restrict__ kN,
                                         const float *__restrict__ vN,
                                         const float *__restrict__ beta,
                                         const float *__restrict__ decayV,
                                         float *__restrict__ state,
                                         float *__restrict__ gdnOut,
                                         uint32_t nVHeads, float invSqrtHeadV) {
            const uint32_t hv = blockIdx.x;
            if (hv >= nVHeads) return;
            const uint32_t j = threadIdx.x;
            if (j >= HEADV) return;
            constexpr uint32_t Sv = HEADV;
            const float *qHead = qN + static_cast<size_t>(hv) * HEADV;
            const float *kHead = kN + static_cast<size_t>(hv) * HEADV;
            const float *vHead = vN + static_cast<size_t>(hv) * HEADV;
            float *rowJ = state + static_cast<size_t>(hv) * HEADV * HEADV +
                          static_cast<size_t>(j) * Sv;
            // decay S *= exp(gateV[hv])
            const float dec = __expf(decayV[hv]);
            for (uint32_t i = 0; i < Sv; ++i) rowJ[i] *= dec;
            __syncthreads();
            // delta[j] = (v[j] - dot(row_j(M), k)) * beta[hv]
            float dot = 0.0f;
            for (uint32_t i = 0; i < Sv; ++i) dot = fmaf(rowJ[i], kHead[i], dot);
            const float del = (vHead[j] - dot) * beta[hv];
            // M[j][i] += delta[j] * k[i]  (only row j touched)
            for (uint32_t i = 0; i < Sv; ++i) rowJ[i] = fmaf(del, kHead[i], rowJ[i]);
            // out[j] = dot(row_j(M), q) * scale
            float outRow = 0.0f;
            for (uint32_t i = 0; i < Sv; ++i) outRow = fmaf(rowJ[i], qHead[i], outRow);
            gdnOut[static_cast<size_t>(hv) * HEADV + j] = outRow * invSqrtHeadV;
        }

        // Warp flash attention for qwen35 full-attention layers (headDim=256:
        // 8 fp32 accumulators per lane, warp per (token, q-head), causal from
        // cachePos).  Mirrors kWarpAttention with template HD=256.
        template<uint32_t HD>
        __global__ void kQ35WarpAttention(const float *__restrict__ q,
                                          const float *__restrict__ kCache,
                                          const float *__restrict__ vCache,
                                          float *__restrict__ out, uint32_t seqLen,
                                          uint32_t nHeads, uint32_t nKVHeads,
                                          uint32_t cachePos, float invSqrt) {
            static_assert(HD == 256, "qwen35 uses headDim=256");
            constexpr uint32_t NV = HD / 32;
            uint32_t s = blockIdx.x;
            uint32_t warpId = blockIdx.y * blockDim.y + threadIdx.y;
            uint32_t lane = threadIdx.x;
            if (s >= seqLen || warpId >= nHeads) return;
            uint32_t qHead = warpId;
            uint32_t kvHead = qHead / (nHeads / nKVHeads);
            uint32_t csEnd = cachePos + s;
            const float *qPtr = q + (static_cast<size_t>(s) * nHeads + qHead) * HD;
            float acc[NV];
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
            float m = -1e30f, l = 0.0f;
            for (uint32_t cs = 0; cs <= csEnd; ++cs) {
                const float *kPtr =
                        kCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                const float *vPtr =
                        vCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                float sc = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) {
                    sc = fmaf(qPtr[lane + i * 32], kPtr[lane + i * 32], sc);
                }
                sc = kWarpReduceSum(sc) * invSqrt;
                if (sc > m) {
                    float mNew = sc;
                    float alpha = exp2f((m - mNew) * LOG2E);
                    for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
                    l *= alpha;
                    m = mNew;
                }
                float p = exp2f((sc - m) * LOG2E);
                l += p;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) {
                    acc[i] = fmaf(p, vPtr[lane + i * 32], acc[i]);
                }
            }
            float invL = 1.0f / l;
            float *oPtr = out + (static_cast<size_t>(s) * nHeads + qHead) * HD;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) {
                oPtr[lane + i * 32] = acc[i] * invL;
            }
        }

    }// namespace

    // ---- Cross-translation-unit device helpers ----
    // Defined in GPUComputeUpload.cu, called from GPUComputeMoE.cu.  Declared
    // at namespace scope (external linkage) so the call links across TUs.
    double bwSelfTestHostToPinned(const uint8_t *src, uint8_t *dst, size_t bytes,
                                  int iters);

}// namespace tinycoder::gpu

#endif// USE_CUDA
