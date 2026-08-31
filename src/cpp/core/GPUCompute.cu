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
 *   1. Single-token generation (seqLen == 1) is DRAM-bandwidth-bound.
 *      We stream the COMPACT quantized weights (Q2_K 84 B/block, Q3_K
 *      110 B/block, Q4_K 144 B/block) straight from VRAM with a warp-per-row
 *      dequant-GEMV kernel — the same memory traffic llama.cpp's CUDA GEMV
 *      uses.  The FP16 twin (2 B/elem) would read ~5x the bytes and cap
 *      generation near ~270 tok/s, below llama.cpp's 295 tok/s on this part.
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

#ifdef USE_CUDA

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
        constexpr uint32_t kTypeIQ4XS = 23;
        constexpr uint32_t kTypeIQ3S = 21;
        constexpr uint32_t kTypeIQ2S = 22;

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
        constexpr uint32_t kIQ4XS_BYTES = 136;

        constexpr float LOG2E = 1.4426950408889634f;

        cudaStream_t g_stream = nullptr;
        cublasHandle_t g_cublas = nullptr;
        // Secondary non-blocking stream (copy-engine only).  The CPU-expert
        // hybrid handoff (forwardQwen35MoePrefix) uses it for the
        // norm D2H + FFN H2D copies so they OVERLAP the GPU's g_stream compute
        // (attention of layer L+1 while the CPU computes layer L's experts).
        // Never used with cuBLAS: g_cublas stays bound to g_stream and all
        // GEMM/GEMV launches go to g_stream.
        cudaStream_t g_stream2 = nullptr;
        std::once_flag g_cudaInit;
        // TINYCODER_GPU_VERBOSE=1: per-stage stream syncs + print (debug aid).
        const bool g_verbose = [] {
            const char *e = std::getenv("TINYCODER_GPU_VERBOSE");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();

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

        // ------------------------------------------------------------------
        // RoPE on Q, and K/V store with fused K rotation (matches
        // ModelPrimitives.cpp applyRoPE + storeKVWithRoPE element order).
        // ------------------------------------------------------------------

        // q[k] into q[k] (in place).  One block per token, 128 threads.
        // NEOX (rotate-half) pairing, matching llama.cpp for qwen2/qwen3:
        // rotate the pair (head[j], head[j + headDim/2]) with angle index j.
        // (Interleaved (2j,2j+1) was the wrong transform for pos>0 and
        // produced garbage multi-token generations; identity at pos 0 masked
        // it in single-token dumps.)
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
            for (uint32_t j = threadIdx.x; j < pairs; j += blockDim.x) {
                float cc = c[j], ss = sn[j];
                for (uint32_t h = 0; h < qHeads; ++h) {
                    float *head = q + (static_cast<size_t>(s) * qHeads + h) * headDim;
                    float x0 = head[j], x1 = head[j + pairs];
                    head[j] = fmaf(x0, cc, -x1 * ss);
                    head[j + pairs] = fmaf(x0, ss, x1 * cc);
                }
            }
        }

        // kSrc/vSrc [seqLen][kHeads*headDim] -> kDst/vDst cache at cachePos.
        // One block (256 threads) per token.
        // K rotation pairing MUST match the CPU reference (ModelPrimitives.cpp
        // storeKVWithRoPE) and llama.cpp NEOX: rotate the pair
        // (kh[j], kh[j + headDim/2]) with angle index j.
        __global__ void kStoreKVRope(const float *kSrc, const float *vSrc, float *kDst,
                                     float *vDst, uint32_t seqLen, uint32_t kHeads,
                                     uint32_t headDim, uint32_t cachePos, uint32_t pos,
                                     const float *cosT, const float *sinT) {
            uint32_t s = blockIdx.x;
            if (s >= seqLen) return;
            uint32_t p = pos + s;
            uint32_t pairs = headDim / 2;
            uint32_t kvSize = kHeads * headDim;
            const float *ks = kSrc + static_cast<size_t>(s) * kvSize;
            const float *vs = vSrc + static_cast<size_t>(s) * kvSize;
            float *kd = kDst + static_cast<size_t>(cachePos + s) * kvSize;
            float *vd = vDst + static_cast<size_t>(cachePos + s) * kvSize;
            const float *c = cosT + static_cast<size_t>(p) * pairs;
            const float *sn = sinT + static_cast<size_t>(p) * pairs;
            uint32_t t = threadIdx.x;
            // V copied verbatim for ALL columns.
            for (uint32_t e = t; e < kvSize; e += blockDim.x) {
                vd[e] = vs[e];
            }
            // K rotated per-head on the NEOX (j, j+pairs) pairs, angle index j.
            const uint32_t numPairs = kHeads * pairs;
            for (uint32_t e = t; e < numPairs; e += blockDim.x) {
                uint32_t h = e / pairs;
                uint32_t j = e - h * pairs;
                uint32_t base = h * headDim + j;
                float k0 = ks[base], k1 = ks[base + pairs];
                kd[base] = fmaf(k0, c[j], -k1 * sn[j]);
                kd[base + pairs] = fmaf(k0, sn[j], k1 * c[j]);
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
        template<uint32_t HD>
        __global__ void kWarpAttention(const float *__restrict__ q,
                                       const float *__restrict__ kCache,
                                       const float *__restrict__ vCache,
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
                const float *kPtr =
                        kCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                const float *vPtr =
                        vCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
                // score = dot(q, k) * invSqrt, warp-reduced
                float sc = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < NV; ++i) {
                    sc = fmaf(qPtr[lane + i * 32], kPtr[lane + i * 32], sc);
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
                    acc[i] = fmaf(pp, vPtr[lane + i * 32], acc[i]);
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
        template<int TYPE>
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
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
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
            } else if (TYPE == kTypeQ3K) {
                // Same coalesced-byte scheme as Q2_K.  hm (32 bytes, block+0)
                // holds the high-bit mask: bit 1<<jj for the same (sub,l)
                // position => hm[lane] with the 1<<jj mask.  qv = (qb>>2jj&3) -
                // (hm[lane]&(1<<jj) ? 0 : 4).  Scale is the repacked int8
                // sc16[half*8+jj*2+sub].  No dmin (Q3_K has none).
                const uint32_t sub = lane / 16u;
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
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
                            acc = fmaf(dl * qv, (cidx < cols) ? x[cidx] : 0.0f,
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
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
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
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        float w =
                                dl * static_cast<float>(grid[j]) *
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
                    const uint8_t *grid = reinterpret_cast<const uint8_t *>(
                            &c_iq2xxs_grid[gridIdx]);
                    const float *xs = x + b * 256u + e0;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        float wgt = db * static_cast<float>(grid[j]) *
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

        // ---- 32-wide-block GEMV (Q5_0 / Q8_0) ----
        // These legacy quant formats use 32-weight blocks (22 B / 34 B each)
        // rather than the 256-wide K-quant blocks handled by kQGemv above. One
        // warp handles one output row; each lane owns column (b*32 + lane) of a
        // block, computes its single weight, and the warp reduction sums them.
        // blocksPerRow = ceil(cols/32); rowBytes = blocksPerRow * blockBytes.
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
            constexpr uint32_t BLOCK_BYTES = (TYPE == kTypeQ5_0) ? kQ5_0_BYTES : kQ8_0_BYTES;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blk = rp + static_cast<uint64_t>(b) * BLOCK_BYTES;
                const uint32_t col = b * 32u + lane;
                if (col >= cols) break;
                __half d = *reinterpret_cast<const __half *>(blk);
                float wv;
                if (TYPE == kTypeQ8_0) {
                    const int8_t q = reinterpret_cast<const int8_t *>(blk + 2)[lane];
                    wv = __half2float(d) * static_cast<float>(q);
                } else {// Q5_0 (llama.cpp layout; matches CPU dequantizeQ5_0Block)
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

        // ------------------------------------------------------------------
        // Q8_K activation quantization + integer vec-dot GEMV (decode path)
        // for the IQ2_S / IQ3_XXS / IQ3_S weight types.
        //
        // llama.cpp computes these types' CPU and CUDA mul_mat by first
        // quantizing the fp32 activation to Q8_K (vec_dot_type == Q8_K) and
        // accumulating with exact integer math (vec_dot_iq*_q8_K generic):
        //   per 256-element block:
        //     d   = fp16(w.d) * q8b.d
        //     bsum = sum_i ls(i) * sumi(i)          (INTEGER across 8x(ib32,l))
        //     sumf += d * bsum                      (FLOAT, one op per block)
        //   result = [0.125f | 0.25f | 1.0f] * sumf (per-type trailing scale)
        // A float dequant-dot differs numerically (up to ~5%), which flips
        // near-tie argmax at knife-edge positions -- so the GPU decode must
        // reproduce the EXACT integer-bsum-first accumulation to match llama.
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
            const float iscale = (max != 0.0f) ? (-127.0f / max) : 0.0f;
            const float d = (max != 0.0f) ? (1.0f / iscale) : 0.0f;
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

        // Batched weighted accumulation: out[s] = sum_r wgt[s*stride + r] *
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
        __device__ __forceinline__ int q8kLaneSumi(
                int type, const uint8_t *__restrict__ w,
                const int8_t *__restrict__ y8, uint32_t ib32, uint32_t l) {
            if (type == kTypeIQ2S) {
                const uint8_t *qsv = w + 2;
                const uint8_t *qh = w + 66;
                const uint16_t gridIdx = static_cast<uint16_t>(
                        qsv[ib32 * 4u + l] | ((qh[ib32] << (8 - 2u * l)) & 0x300u));
                const uint8_t *grid =
                        reinterpret_cast<const uint8_t *>(&c_iq2s_grid[gridIdx]);
                const uint8_t signs = qsv[32u + ib32 * 4u + l];
                int sumi = 0;
                for (uint32_t j = 0; j < 8; ++j) {
                    sumi += static_cast<int>(y8[ib32 * 32u + l * 8u + j]) *
                            static_cast<int>(grid[j]) *
                            ((signs & c_kmask_iq2xs[j]) ? -1 : 1);
                }
                return sumi;
            }
            if (type == kTypeIQ3_XXS) {
                const uint8_t *q3 = w + 2;
                uint32_t aux32;
                std::memcpy(&aux32, q3 + 64u + 4u * ib32, sizeof(uint32_t));
                const uint8_t *grid1 = reinterpret_cast<const uint8_t *>(
                        &c_iq3xxs_grid[q3[8u * ib32 + 2u * l]]);
                const uint8_t *grid2 = reinterpret_cast<const uint8_t *>(
                        &c_iq3xxs_grid[q3[8u * ib32 + 2u * l + 1u]]);
                const uint8_t signs = c_ksigns_iq2xs[(aux32 >> (7u * l)) & 127u];
                int sumi = 0;
                for (uint32_t j = 0; j < 4; ++j) {
                    sumi += static_cast<int>(grid1[j]) *
                            static_cast<int>(y8[ib32 * 32u + l * 8u + j + 0]) *
                            ((signs & c_kmask_iq2xs[j + 0]) ? -1 : 1);
                    sumi += static_cast<int>(grid2[j]) *
                            static_cast<int>(y8[ib32 * 32u + l * 8u + j + 4]) *
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
            const uint32_t g1 = c_iq3s_grid[gridIdx1];
            const uint32_t g2 = c_iq3s_grid[gridIdx2];
            const uint8_t sbyte = signs[signOff + l];
            int sumi = 0;
            for (uint32_t j = 0; j < 4; ++j) {
                sumi += static_cast<int>((g1 >> (8u * j)) & 0xFFu) *
                        static_cast<int>(y8[ib32 * 32u + l * 8u + j + 0]) *
                        ((sbyte & c_kmask_iq2xs[j + 0]) ? -1 : 1);
                // grid2 is a uint32 whose bytes 0..3 are grid2[j] for j=0..3
                // (the CPU reference reads them via
                // reinterpret_cast<const uint8_t*>(&IQ3S_GRID[...])[j], i.e.
                // byte j of the little-endian word).  Shifting by (j+4) would
                // shift the 32-bit word out entirely (>>= 32..56) and silently
                // DROP all four grid2 contributions -- the attnO corruption.
                sumi += static_cast<int>((g2 >> (8u * j)) & 0xFFu) *
                        static_cast<int>(y8[ib32 * 32u + l * 8u + j + 4]) *
                        ((sbyte & c_kmask_iq2xs[j + 4]) ? -1 : 1);
            }
            return sumi;
        }

        // Warp-per-row Q8K GEMV over the pre-quantized activation (q8k).
        //
        // Per (ib32, l) lane: integer sumi (matches vec_dot_iq*_q8_K generic).
        // The per-ib32 bsum is
        //   IQ2_S: ls1*sumi(l=0..1) + ls2*sumi(l=2..3)  (per 32-group ib32)
        //   IQ3_XXS: ls*sumi(l=0..3)
        //   IQ3_S:  ls*sumi(l=0..3) per 32-wide half pairing (ib32 pairs)
        // Since everything is integer, multiplying each lane's sumi by its own
        // ls FIRST and then doing a plain 32-lane tree reduction reproduces the
        // exact same integer bsum (addition distributes over the scalar ls,
        // including the split ls1/ls2 case).  The per-BLOCK integer bsum, its
        // float d*bsum, and the serial float `sumf += d*bsum` across blocks all
        // match llama's generic order; lane 0 then applies the trailing scale.
        template<int TYPE>
        __global__ void kQGemvQ8K(const uint8_t *__restrict__ w,
                                  const uint8_t *__restrict__ x,
                                  float *__restrict__ out, uint32_t rows,
                                  uint32_t blocksPerRow, uint32_t rowBytes) {
            const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
            if (row >= rows) return;
            const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
            const uint32_t lane = threadIdx.x;
            const uint32_t ib32 = lane >> 2u;// 0..7 within a block
            const uint32_t l = lane & 3u;    // 0..3
            float sumf = 0.0f;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blkW = rp + static_cast<uint64_t>(b) *
                                                   ((TYPE == kTypeIQ2S)      ? kIQ2S_BYTES
                                                    : (TYPE == kTypeIQ3_XXS) ? kIQ3XXS_BYTES
                                                                             : kIQ3S_BYTES);
                const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
                const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA + kQ8K_Q);
                const float yd = *reinterpret_cast<const float *>(blkA + kQ8K_D);

                // Per-lane (ib32,l) integer sumi.
                int ci = q8kLaneSumi(TYPE, blkW, y8, ib32, l);

                // Multiply each lane's sumi by its own ls multiplier FIRST. The
                // distributivity of integer arithmetic makes the subsequent
                // plain 32-lane tree reduction land on the identical per-block
                // bsum that llama computes serially (ls1(s0+s1)+ls2(s2+s3) ==
                // ls1*s0 + ls1*s1 + ls2*s2 + ls2*s3).
                if (TYPE == kTypeIQ2S) {
                    const uint8_t *scales = blkW + 74;
                    const int ls = (l < 2) ? (1 + 2 * (scales[ib32] & 0xf))
                                           : (1 + 2 * (scales[ib32] >> 4));
                    ci *= ls;
                } else if (TYPE == kTypeIQ3_XXS) {
                    uint32_t aux32;
                    std::memcpy(&aux32, blkW + 2 + 64u + 4u * ib32,
                                sizeof(uint32_t));
                    ci *= static_cast<int>(2 * (aux32 >> 28) + 1);
                } else {// kTypeIQ3S
                    const uint8_t *scales = blkW + 106;
                    const int ls = static_cast<int>(
                            2 * ((ib32 & 1u) ? (scales[ib32 >> 1u] >> 4)
                                             : (scales[ib32 >> 1u] & 0xf)) +
                            1);
                    ci *= ls;
                }

                // 32-lane tree reduction -> per-BLOCK integer bsum.
                int bsum = ci + __shfl_xor_sync(0xffffffffu, ci, 1);
                bsum += __shfl_xor_sync(0xffffffffu, bsum, 2);
                bsum += __shfl_xor_sync(0xffffffffu, bsum, 4);
                bsum += __shfl_xor_sync(0xffffffffu, bsum, 8);
                bsum += __shfl_xor_sync(0xffffffffu, bsum, 16);

                // lane 0 accumulates the serial float `sumf += d * bsum`.
                // NOTE: use plain mul+add (NOT fmaf) -- llama's generic
                // vec_dot_iq*_q8_K and TinyCoder's CPU path both round
                // d*bsum and the running sum separately.  fmaf would differ
                // by 1 ulp per block, compounding through the KV cache across
                // decode steps and flipping the distribution by step 3+.
                if (lane == 0) {
                    const float dval =
                            __half2float(*reinterpret_cast<const __half *>(blkW)) * yd;
                    sumf += dval * static_cast<float>(bsum);
                }
            }
            const float scale = (TYPE == kTypeIQ2S)      ? 0.125f
                                : (TYPE == kTypeIQ3_XXS) ? 0.25f
                                                         : 1.0f;
            if (lane == 0) out[row] = scale * sumf;
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
                if (TYPE == kTypeQ5_0 || TYPE == kTypeQ8_0) {
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
                    rp += (TYPE == kTypeQ5_0) ? kQ5_0_BYTES : kQ8_0_BYTES;
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
                static constexpr uint32_t SZ = (TYPE == kTypeQ5_0 || TYPE == kTypeQ8_0)
                                                       ? (TYPE == kTypeQ5_0 ? kQ5_0_BYTES
                                                                            : kQ8_0_BYTES)
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
                                     "blocksPerRow=%u rowBytes=%u\n",
                             type, (type == kTypeIQ2S || type == kTypeIQ3_XXS || type == kTypeIQ3S),
                             rows, cols, blocksPerRow, rowBytes);
            }
            if (type == kTypeIQ2S || type == kTypeIQ3_XXS || type == kTypeIQ3S) {
                // llama.cpp's CUDA backend quantizes the fp32 activation to
                // Q8_K (vec_dot_type == Q8_K) and accumulates with exact
                // integer math (vec_dot_iq*_q8_K).  The float dequant-dot
                // differs by up to ~5%, flipping near-tie argmax at knife-edge
                // rows -- so mirror the Q8_K integer path here: quantize x once
                // into the reusable q8k_ scratch, then run the integer GEMV.
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
                if (type == kTypeIQ2S) {
                    kQGemvQ8K<kTypeIQ2S><<<nBlocks, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), q8k, out, rows,
                            blocksPerRow, rowBytes);
                } else if (type == kTypeIQ3_XXS) {
                    kQGemvQ8K<kTypeIQ3_XXS><<<nBlocks, block, 0, g_stream>>>(
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
                kQGemv<kTypeQ2K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeQ3K) {
                kQGemv<kTypeQ3K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeQ4K) {
                kQGemv<kTypeQ4K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeQ6K) {
                kQGemv<kTypeQ6K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeIQ2XS) {
                kQGemv<kTypeIQ2XS><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeIQ2XXS) {
                kQGemv<kTypeIQ2XXS><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeIQ4XS) {
                kQGemv<kTypeIQ4XS><<<nBlocks, block, 0, g_stream>>>(
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
            } else if (type == kTypeQ5K) {
                kQGemv<kTypeQ5K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows, cols,
                        rowBytes, blocksPerRow);
            } else if (type == kTypeQ5_0) {
                kQGemvSmall32<kTypeQ5_0><<<nBlocks, block, 0, g_stream>>>(
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
            constexpr uint32_t BLOCK_BYTES =
                    (TYPE == kTypeQ5_0) ? kQ5_0_BYTES : kQ8_0_BYTES;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint8_t *blk = row + static_cast<uint64_t>(b) * BLOCK_BYTES;
                const uint32_t col = b * 32u + lane;
                if (col >= hiddenSize) break;
                __half d = *reinterpret_cast<const __half *>(blk);
                if (TYPE == kTypeQ8_0) {
                    const int8_t q = reinterpret_cast<const int8_t *>(blk + 2)[lane];
                    h[col] = __half2float(d) * static_cast<float>(q);
                } else {// Q5_0 (llama.cpp layout; matches CPU dequantizeQ5_0Block)
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
            if (type == kTypeQ5_0 || type == kTypeQ8_0) {
                return (cols + 31) / 32;// 32-wide legacy blocks
            }
            return (cols + 255) / 256;// K-quant blocks (256 wide)
        }

        // ====================================================================
        // Qwen35 (dense gated-delta-net) kernels.
        //
        // Layer classification (llama.cpp qwen35.cpp):
        //   is_recr(i) = (i < n_layer) && ((i+1) % full_attention_interval != 0)
        // Recurrent layers run the gated delta net (linear attention with a
        // [headV x headV] state per value head); full-attention layers run Q/K/V
        // projections + per-head RMSNorm + MRoPE + attention with sigmoid(gate).
        // The MTP block (layer >= n_layer) is never executed in the main decode
        // pass (the CPU forward skips it), so kernels only process the real
        // layers.  All matmuls re-use the existing launchQGemv / cuBLAS fp16
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
        __global__ void kQ35HeadRmsNorm(const float *__restrict__ in,
                                        float *__restrict__ out,
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
        __global__ void kQ35BetaGate(const float *__restrict__ betaIn,
                                     const float *__restrict__ alphaIn,
                                     const float *__restrict__ dtBias,
                                     const float *__restrict__ aBroadcast,
                                     float *__restrict__ betaOut,
                                     float *__restrict__ gateVOut,
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
        __global__ void kQ35L2NormRepeat(const float *__restrict__ convOut,
                                         float *__restrict__ qN, float *__restrict__ kN,
                                         float *__restrict__ vN, uint32_t headK,
                                         uint32_t headV, uint32_t nKHeads,
                                         uint32_t nVHeads, uint32_t keyDim,
                                         uint32_t valueDim, uint32_t seqLen) {
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
            for (uint32_t i = threadIdx.x; i < headK * nVHeads; i += blockDim.x) {
                uint32_t hv = i / headK;
                uint32_t hk = hv % nKHeads;
                uint32_t d = i - hv * headK;
                const float *qSrc = convQ + static_cast<size_t>(hk) * headK;
                const float *kSrc = convK + static_cast<size_t>(hk) * headK;
                // per-slice L2 scale is computed per (hv,hk) once below; for
                // simplicity recompute here (small: headK*nVHeads = 6144).
                double sq = 0.0, sk = 0.0;
                for (uint32_t j = 0; j < headK; ++j) {
                    float qv = qSrc[j], kv = kSrc[j];
                    sq += static_cast<double>(qv) * qv;
                    sk += static_cast<double>(kv) * kv;
                }
                float iq = static_cast<float>(
                        1.0 / std::sqrt(fmax(sq, 1e-6)));
                float ik = static_cast<float>(
                        1.0 / std::sqrt(fmax(sk, 1e-6)));
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
        // Routing (mirrors llama build_moe_ffn with SOFTMAX gating + norm_w):
        //   logits = ffn_gate_inp (F32 [expertCount x hidden]) @ x
        //   probs  = softmax(logits) over ALL experts
        //   top-k by probs (descending), k = expertUsedCount (8)
        //   weight = probs[sel] / max(wsum_topk, 6.103515625e-5)   (renorm)
        // The router runs ONE block per token with one warp per expert
        // (grid = (seqLen, ), block = 256 threads = 8 warps of 32 lanes).
        // Output for token s:
        //   moeIdx[s][r]   expert id of rank r (top-k)
        //   moeWgt[s][r]   renormalized weight of rank r
        //   moeWsum[s]     max(wsum, 6.103515625e-5) (for debug / scale)
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

    // ----------------------------------------------------------------------
    // Public API
    // ----------------------------------------------------------------------

    // Host-side diagnostic: run ONE weight row through the GPU Q8_K GEMV
    // (kQuantizeQ8K + kQGemvQ8K) for the IQ2_S / IQ3_XXS / IQ3_S types and
    // return the scalar row output.  Used by unit tests to verify the GPU
    // integer-dot kernels reproduce the CPU Q8K path bit-for-bit.
    float GPUModel::debugQ8KRow(uint32_t type, const uint8_t *wRow,
                                uint32_t blocksPerRow, const float *x,
                                uint32_t cols) {
        // Host->device: one weight row + one activation vector.
        uint32_t rowBytes = blocksPerRow *
                            (type == kTypeIQ2S      ? kIQ2S_BYTES
                             : type == kTypeIQ3_XXS ? kIQ3XXS_BYTES
                                                    : kIQ3S_BYTES);
        uint8_t *dW = nullptr;
        float *dX = nullptr, *dOut = nullptr;
        cudaMalloc(&dW, rowBytes);
        cudaMalloc(&dX, cols * sizeof(float));
        cudaMalloc(&dOut, sizeof(float));
        uint8_t *dQ8 = nullptr;
        cudaMalloc(&dQ8, static_cast<uint64_t>(blocksPerRow) * kQ8K_STRIDE);
        cudaMemcpy(dW, wRow, rowBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dX, x, cols * sizeof(float), cudaMemcpyHostToDevice);
        kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(dX, dQ8, cols);
        if (type == kTypeIQ2S) {
            kQGemvQ8K<kTypeIQ2S><<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        } else if (type == kTypeIQ3_XXS) {
            kQGemvQ8K<kTypeIQ3_XXS><<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        } else {
            kQGemvQ8K<kTypeIQ3S><<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        }
        // The kernels ran on the non-blocking g_stream; a synchronous memcpy on
        // the default stream would NOT wait for them, and the immediatelly
        // following cudaFree()s could hand the same addresses back to the next
        // call's cudaMalloc() while the previous kernels are still writing.
        // Sync the stream before reading the result (diagnostic-only path --
        // correctness over speed).
        cudaStreamSynchronize(g_stream);
        float result = 0.0f;
        cudaMemcpy(&result, dOut, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dW);
        cudaFree(dX);
        cudaFree(dOut);
        cudaFree(dQ8);
        return result;
    }

    // Host-side diagnostic: run ONE weight row through the GPU Q8_0 INTEGER
    // GEMV (kQuantizeQ8_0x32 + kQGemvQ8_0xQ8K) -- the exact path launchQGemv's
    // Q8_0 branch uses for the qwen35moe trunk projections and LM head -- and
    // return the scalar row output.  Used by unit tests to verify the GPU
    // reproduces the CPU matMulVecBatchQ8_0_Q8K_AVX2 reference bit-for-bit.
    float GPUModel::debugQ8_0Row(const uint8_t *wRow, uint32_t blocksPerRow,
                                 const float *x, uint32_t cols) {
        const uint32_t rowBytes = blocksPerRow * kQ8_0_BYTES;
        uint8_t *dW = nullptr;
        float *dX = nullptr, *dOut = nullptr;
        cudaMalloc(&dW, rowBytes);
        cudaMalloc(&dX, cols * sizeof(float));
        cudaMalloc(&dOut, sizeof(float));
        uint8_t *dQ8 = nullptr;
        cudaMalloc(&dQ8,
                   static_cast<uint64_t>(blocksPerRow) * kQ8_0x32_STRIDE);
        cudaMemcpy(dW, wRow, rowBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dX, x, cols * sizeof(float), cudaMemcpyHostToDevice);
        kQuantizeQ8_0x32<<<blocksPerRow, 32, 0, g_stream>>>(dX, dQ8, cols);
        kQGemvQ8_0xQ8K<<<1, dim3(32, 1), 0, g_stream>>>(
                dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        cudaStreamSynchronize(g_stream);
        float result = 0.0f;
        cudaMemcpy(&result, dOut, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dW);
        cudaFree(dX);
        cudaFree(dOut);
        cudaFree(dQ8);
        return result;
    }

    // Host-side diagnostic: run kQuantizeQ8_0x32 on x and copy the quantized
    // block bytes (kQ8_0x32_STRIDE each) back to the host so a unit test can
    // diff the GPU activation quantization against a host-side scalar copy of
    // the CPU's quantize_row_q8_0 loop block by block.
    void GPUModel::debugQ8_0Blocks(const uint8_t *wRow, uint32_t blocksPerRow,
                                   const float *x, uint32_t cols,
                                   float *outBlk, uint8_t *outQuant,
                                   float *outTrace) {
        const uint32_t rowBytes = blocksPerRow * kQ8_0_BYTES;
        float *dX = nullptr;
        uint8_t *dQ8 = nullptr, *dW = nullptr;
        float *dTrace = nullptr, *dOut = nullptr;
        cudaMalloc(&dX, cols * sizeof(float));
        cudaMalloc(&dQ8,
                   static_cast<uint64_t>(blocksPerRow) * kQ8_0x32_STRIDE);
        cudaMalloc(&dW, rowBytes);
        cudaMemcpy(dX, x, cols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dW, wRow, rowBytes, cudaMemcpyHostToDevice);
        kQuantizeQ8_0x32<<<blocksPerRow, 32, 0, g_stream>>>(dX, dQ8, cols);
        if (outTrace != nullptr) {
            // Launch the tracing twin so the kernel's OWN per-block cumulative
            // sums (not the host recompute) can be diffed block-by-block.
            const uint32_t traceN = 2u * blocksPerRow + 16u;
            cudaMalloc(&dTrace, traceN * sizeof(float));
            cudaMalloc(&dOut, sizeof(float));
            kQGemvQ8_0xQ8KTrace<<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, dTrace, 1, blocksPerRow, rowBytes);
        }
        cudaStreamSynchronize(g_stream);
        // The trace kernel writes directly in the documented layout:
        // [0..blocksPerRow) cumulative sums, [blocksPerRow+g] block-0 t[g],
        // [blocksPerRow+8] block-0 blk; [2*blocksPerRow+g] block-1 s[g];
        // [2*blocksPerRow+8] block-1 blk.  Raw copy to the host, no reshaping.
        if (outTrace != nullptr) {
            const uint32_t traceN = 2u * blocksPerRow + 16u;
            cudaMemcpy(outTrace, dTrace, traceN * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
        if (dTrace != nullptr) cudaFree(dTrace);
        if (dOut != nullptr) cudaFree(dOut);
        if (dW != nullptr) cudaFree(dW);
        cudaMemcpy(outQuant, dQ8,
                   static_cast<uint64_t>(blocksPerRow) * kQ8_0x32_STRIDE,
                   cudaMemcpyDeviceToHost);
        // Recompute the per-block float contributions on the host using the
        // SAME float tree kQGemvQ8_0xQ8K runs (from the GPU's own quantized
        // bytes) so a test can compare against the CPU AVX2 per-block sums.
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
            const uint8_t *blkW =
                    wRow + static_cast<uint64_t>(b) * kQ8_0_BYTES;
            const uint8_t *blkA =
                    outQuant + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
            const float d = __half2float(*(const __half *) (blkW + 0));
            const float d8 = *reinterpret_cast<const float *>(blkA);
            if (d8 == 0.0f) {
                outBlk[b] = 0.0f;
                continue;
            }
            // Per-4-group integer sums (mirror of the kernel's shfl logic).
            int s[8];
            for (uint32_t g = 0; g < 8; ++g) {
                int c = 0;
                for (uint32_t j = 0; j < 4; ++j) {
                    const int wq = static_cast<int>(static_cast<int8_t>(
                            blkW[2u + 4u * g + j]));
                    const int xq = static_cast<int>(static_cast<int8_t>(
                            blkA[4u + 4u * g + j]));
                    c += xq * wq;
                }
                s[g] = c;
            }
            const float scale = d * d8;
            float t[8];
            for (uint32_t g = 0; g < 8; ++g) t[g] = scale * static_cast<float>(s[g]);
            const float u0 = t[0] + t[4];
            const float u1 = t[1] + t[5];
            const float u2 = t[2] + t[6];
            const float u3 = t[3] + t[7];
            const float v0 = u0 + u1;
            const float v1 = u2 + u3;
            outBlk[b] = v0 + v1;
        }
        cudaFree(dX);
        cudaFree(dQ8);
    }

    // GPU is enabled by default in any build compiled with USE_CUDA
    // (scripts/build.sh and npm run build:native produce such builds).
    // $TINYCODER_GPU=0 disables the offload engine (CPU-only), which is the
    // opt-out for systems without a usable CUDA device.
    bool gpuEnabled() {
        const char *e = std::getenv("TINYCODER_GPU");
        if (e == nullptr || e[0] == '\0') return true;
        return e[0] != '0';
    }

    bool GPUModel::ensureScratch(uint32_t seqLen, std::string &errMsg) {
        if (scratchAlloc_ && scratch_.seqCap >= seqLen) return true;
        destroyScratch();
        // Allocate for the requested capacity (grow-on-demand; never shrink).
        uint32_t cap = std::max(seqLen, 1u);
        uint32_t H = geom_.hiddenSize, I = geom_.intermediateSize;
        uint32_t qLen = geom_.numAttentionHeads * geom_.headDim;
        uint32_t kvLen = geom_.numKVHeads * geom_.headDim;
        // Width of the "other" fp16 X-twin bucket (attnOutF16) used by the
        // prefill gemv helper for nCols that is neither H (hiddenF16) nor I
        // (gateF16).  Must cover the widest such input: the qwen35[MoE]
        // recurrent ssm_out GEMM reads valueDim = ssmInnerSize columns, and
        // the qwen35moe MoE down projections read expertFF / sharedFF
        // columns.  Allocating the bucket by qLen alone is too small when
        // qLen (nHeads*headDim) < ssmInnerSize (qwen35moe: 2048 < 4096).
        uint32_t f16OtherW = qLen;
        if (geom_.architecture == 1 || geom_.architecture == 2) {
            f16OtherW = std::max(f16OtherW, geom_.ssmInnerSize);
        }
        if (geom_.architecture == 2) {
            f16OtherW = std::max(f16OtherW, geom_.expertFF);
            f16OtherW = std::max(f16OtherW, geom_.expertSharedFF);
        }
        uint64_t v = static_cast<uint64_t>(cap) * geom_.vocabSize * sizeof(float);
        auto allocT = [&](auto *&p, uint64_t n, const char *what) -> bool {
            p = nullptr;
            cudaError_t e = cudaMalloc(&p, n);
            if (e != cudaSuccess) {
                p = nullptr;
                errMsg = std::string("cudaMalloc(scratch ") + what + "): " +
                         cudaGetErrorString(e);
                return false;
            }
            return true;
        };
        if (!allocT(scratch_.tokens, static_cast<uint64_t>(cap) * sizeof(int32_t),
                    "tokens") ||
            !allocT(scratch_.hidden, static_cast<uint64_t>(cap) * H * sizeof(float),
                    "hidden") ||
            !allocT(scratch_.norm, static_cast<uint64_t>(cap) * H * sizeof(float),
                    "norm") ||
            !allocT(scratch_.hiddenF16,
                    static_cast<uint64_t>(cap) * H * sizeof(uint16_t), "hiddenF16") ||
            !allocT(scratch_.q, static_cast<uint64_t>(cap) * qLen * sizeof(float),
                    "q") ||
            !allocT(scratch_.k, static_cast<uint64_t>(cap) * kvLen * sizeof(float),
                    "k") ||
            !allocT(scratch_.v, static_cast<uint64_t>(cap) * kvLen * sizeof(float),
                    "v") ||
            !allocT(scratch_.attnOut,
                    static_cast<uint64_t>(cap) * qLen * sizeof(float), "attnOut") ||
            !allocT(scratch_.attnOutF16,
                    static_cast<uint64_t>(cap) * f16OtherW * sizeof(uint16_t),
                    "attnOutF16") ||
            !allocT(scratch_.attnProj,
                    static_cast<uint64_t>(cap) * H * sizeof(float), "attnProj") ||
            !allocT(scratch_.gate, static_cast<uint64_t>(cap) * I * sizeof(float),
                    "gate") ||
            !allocT(scratch_.gateF16,
                    static_cast<uint64_t>(cap) * I * sizeof(uint16_t), "gateF16") ||
            !allocT(scratch_.up, static_cast<uint64_t>(cap) * I * sizeof(float),
                    "up") ||
            !allocT(scratch_.ffnOut,
                    static_cast<uint64_t>(cap) * H * sizeof(float), "ffnOut") ||
            !allocT(scratch_.logits, v, "logits")) {
            destroyScratch();
            return false;
        }

        // ---- Qwen35 recurrent-layer scratch (dense gated delta net) ----
        // The dense projection outputs (qkv/z/beta/gateV) are BATCHED over
        // tokens ([seqLen][...]) because attnQKV/attnGate/ssmBetaQ/ssmAlphaQ are
        // plain dense GEMM/GEMVs; the conv + gated-delta-net recurrence is
        // inherently sequential, so those buffers are sized for ONE token
        // (the driver loops tokens feeding one token at a time).
        //   qkvDim = 2*keyDim + valueDim = 2*(headK*nKHeads) + dInner (10240)
        //   convChannels = qkvDim              convKernel = ssmConvKernel (4)
        //   headV = dInner / nVHeads (128)     qN/kN = headK*nVHeads (6144)
        //   vN/gdnOut/q35Norm = headV*nVHeads (6144)
        if (geom_.architecture == 1 || geom_.architecture == 2) {
            // Shared by the dense qwen35 and the qwen35moe recurrent layers
            // (qwen35moe uses the same gated-delta-net block; architecture 2
            // adds the MoE FFN scratch below).
            const uint32_t q35KeyDim = geom_.ssmStateSize * geom_.ssmGroupCount;
            const uint32_t q35QkvDim = 2 * q35KeyDim + geom_.ssmInnerSize;
            const uint32_t q35HeadV =
                    geom_.ssmInnerSize / std::max(geom_.ssmTimeStepRank, 1u);
            const uint32_t q35Kvn = geom_.ssmStateSize * geom_.ssmTimeStepRank;
            const uint32_t q35Vn = q35HeadV * geom_.ssmTimeStepRank;
            const uint32_t q35QGateDim =
                    geom_.numAttentionHeads * 2 * geom_.headDim;
            if (!allocT(scratch_.qkv,
                        static_cast<uint64_t>(cap) * q35QkvDim * sizeof(float),
                        "q35qkv") ||
                !allocT(scratch_.q35z,
                        static_cast<uint64_t>(cap) * geom_.ssmInnerSize *
                                sizeof(float),
                        "q35z") ||
                !allocT(scratch_.q35Beta,
                        static_cast<uint64_t>(cap) * geom_.ssmTimeStepRank *
                                sizeof(float),
                        "q35beta") ||
                !allocT(scratch_.q35GateV,
                        static_cast<uint64_t>(cap) * geom_.ssmTimeStepRank *
                                sizeof(float),
                        "q35gatev") ||
                !allocT(scratch_.q35QGate,
                        static_cast<uint64_t>(cap) * q35QGateDim * sizeof(float),
                        "q35qgate") ||
                !allocT(scratch_.convIn,
                        static_cast<uint64_t>(geom_.ssmConvKernel) *
                                q35QkvDim * sizeof(float),
                        "q35convin") ||
                !allocT(scratch_.convOut,
                        static_cast<uint64_t>(q35QkvDim) * sizeof(float),
                        "q35convout") ||
                !allocT(scratch_.qN,
                        static_cast<uint64_t>(q35Kvn) * sizeof(float), "q35qn") ||
                !allocT(scratch_.kN,
                        static_cast<uint64_t>(q35Kvn) * sizeof(float), "q35kn") ||
                !allocT(scratch_.vN,
                        static_cast<uint64_t>(q35Vn) * sizeof(float), "q35vn") ||
                !allocT(scratch_.gdnOut,
                        static_cast<uint64_t>(cap) * q35Vn * sizeof(float),
                        "q35gdn") ||
                !allocT(scratch_.q35Norm,
                        static_cast<uint64_t>(q35Vn) * sizeof(float),
                        "q35norm")) {
                destroyScratch();
                return false;
            }
        }
        // ---- Qwen35MoE (qwen35moe) MoE FFN scratch ----
        // Router logits [seqLen][expertCount], top-k routing [seqLen]x2*
        // expertUsed (int32 idx + float wgt), clamped-sum [seqLen].  Per-expert
        // gate/up + down are REUSED across ranks (one expert at a time, sized to
        // expertFF / hidden).  Shared-expert buffers are always present even when
        // the model disables the shared expert (geom fields then 0 -> 1-byte
        // allocs; the driver skips by geom.expertSharedFF > 0).
        if (geom_.architecture == 2 &&
            (geom_.expertCount > 0 && geom_.expertUsedCount > 0)) {
            const uint32_t ef = std::max(geom_.expertFF, 1u);
            const uint32_t sf = std::max(geom_.expertSharedFF, 1u);
            const uint32_t eu = geom_.expertUsedCount;
            if (!allocT(scratch_.moeLogits,
                        static_cast<uint64_t>(cap) * geom_.expertCount *
                                sizeof(float),
                        "moeLogits") ||
                !allocT(scratch_.moeIdx,
                        static_cast<uint64_t>(cap) * eu * sizeof(int32_t),
                        "moeIdx") ||
                !allocT(scratch_.moeWgt,
                        static_cast<uint64_t>(cap) * eu * sizeof(float),
                        "moeWgt") ||
                !allocT(scratch_.moeWsum,
                        static_cast<uint64_t>(cap) * sizeof(float), "moeWsum") ||
                !allocT(scratch_.moeGateUp,
                        static_cast<uint64_t>(cap) * ef * sizeof(float),
                        "moeGateUp") ||
                !allocT(scratch_.moeGateUp2,
                        static_cast<uint64_t>(cap) * ef * sizeof(float),
                        "moeGateUp2") ||
                !allocT(scratch_.moeDown,
                        static_cast<uint64_t>(cap) * H * sizeof(float),
                        "moeDown") ||
                // Per-rank slabs for the BATCHED decode fast path
                // (2026-09-14): all expertUsed ranks' gate/up/down single-
                // token buffers at once so the batched kernels can run the
                // ranks CONCURRENTLY (grid.y = rank) instead of the serialized
                // per-rank loop (was ~40 stream-serialized kernels/layer).
                !allocT(scratch_.moeGateUpP,
                        static_cast<uint64_t>(eu) * ef * sizeof(float),
                        "moeGateUpP") ||
                !allocT(scratch_.moeGateUp2P,
                        static_cast<uint64_t>(eu) * ef * sizeof(float),
                        "moeGateUp2P") ||
                !allocT(scratch_.moeDownP,
                        static_cast<uint64_t>(eu) * H * sizeof(float),
                        "moeDownP") ||
                !allocT(scratch_.moeExpertOut,
                        static_cast<uint64_t>(cap) * H * sizeof(float),
                        "moeExpertOut") ||
                !allocT(scratch_.moeShexpGate,
                        static_cast<uint64_t>(cap) * sizeof(float),
                        "moeShexpGate") ||
                !allocT(scratch_.moeShexpGateUp,
                        static_cast<uint64_t>(cap) * sf * sizeof(float),
                        "moeShexpGateUp") ||
                !allocT(scratch_.moeShexpGateUp2,
                        static_cast<uint64_t>(cap) * sf * sizeof(float),
                        "moeShexpGateUp2") ||
                !allocT(scratch_.moeShexpDown,
                        static_cast<uint64_t>(cap) * H * sizeof(float),
                        "moeShexpDown") ||
                // Double-buffered device snapshots of s.norm for the CPU-expert
                // hybrid handoff (see the moeNormSnap comment in the header).
                !allocT(scratch_.moeNormSnap[0],
                        static_cast<uint64_t>(cap) * H * sizeof(float),
                        "moeNormSnap0") ||
                !allocT(scratch_.moeNormSnap[1],
                        static_cast<uint64_t>(cap) * H * sizeof(float),
                        "moeNormSnap1")) {
                destroyScratch();
                return false;
            }
        }
        scratch_.seqCap = cap;
        scratchAlloc_ = true;
        return true;
    }

    void GPUModel::destroyScratch() {
        if (!scratchAlloc_) return;
        cudaFree(scratch_.tokens);
        cudaFree(scratch_.hidden);
        cudaFree(scratch_.norm);
        cudaFree(scratch_.hiddenF16);
        scratch_.hiddenF16 = nullptr;
        cudaFree(scratch_.q);
        cudaFree(scratch_.k);
        cudaFree(scratch_.v);
        cudaFree(scratch_.attnOut);
        cudaFree(scratch_.attnOutF16);
        cudaFree(scratch_.attnProj);
        cudaFree(scratch_.gate);
        cudaFree(scratch_.gateF16);
        cudaFree(scratch_.up);
        cudaFree(scratch_.ffnOut);
        cudaFree(scratch_.logits);
        // Qwen35 recurrent + full-attention scratch.
        cudaFree(scratch_.qkv);
        cudaFree(scratch_.q35z);
        cudaFree(scratch_.q35Beta);
        cudaFree(scratch_.q35GateV);
        cudaFree(scratch_.q35QGate);
        cudaFree(scratch_.convIn);
        cudaFree(scratch_.convOut);
        cudaFree(scratch_.qN);
        cudaFree(scratch_.kN);
        cudaFree(scratch_.vN);
        cudaFree(scratch_.gdnOut);
        cudaFree(scratch_.q35Norm);
        // Qwen35MoE (qwen35moe) MoE FFN scratch.
        cudaFree(scratch_.moeLogits);
        cudaFree(scratch_.moeIdx);
        cudaFree(scratch_.moeWgt);
        cudaFree(scratch_.moeWsum);
        cudaFree(scratch_.moeGateUp);
        cudaFree(scratch_.moeGateUp2);
        cudaFree(scratch_.moeDown);
        cudaFree(scratch_.moeGateUpP);
        cudaFree(scratch_.moeGateUp2P);
        cudaFree(scratch_.moeDownP);
        cudaFree(scratch_.moeExpertOut);
        cudaFree(scratch_.moeShexpGate);
        cudaFree(scratch_.moeShexpGateUp);
        cudaFree(scratch_.moeShexpGateUp2);
        cudaFree(scratch_.moeShexpDown);
        cudaFree(scratch_.moeNormSnap[0]);
        cudaFree(scratch_.moeNormSnap[1]);
        // Value-initialize (zero all pointers) instead of std::memset: the
        // struct has non-trivial members, and std::memset on it trips
        // -Wclass-memaccess.
        scratch_ = DeviceScratch();
        scratchAlloc_ = false;
    }

    bool GPUModel::upload(const std::vector<DeviceLayer> &layers,
                          const ModelGeometry &geom, const void *embedQ,
                          uint32_t embedType, uint32_t embedRowBytes,
                          const void *lmHeadQ, uint32_t lmHeadType,
                          uint32_t lmHeadRowBytes, std::string &errMsg) {
        if (allocated_) destroy();
        // Clear any sticky CUDA error from a previous failed attempt (e.g. an
        // earlier upload's cudaMalloc OOM).  cudaGetLastError() consumes it;
        // leaving it set makes the NEXT forward's first cudaGetLastError()
        // (dequantMatrixF16) report the stale error and abort a healthy retry.
        if (cudaGetLastError() != cudaSuccess) {
            std::fprintf(stderr, "[gpu] upload: cleared sticky CUDA error\n");
        }
        if (!ensureCuda()) {
            errMsg = "CUDA runtime initialization failed";
            return false;
        }
        // numGpuLayers defaults to numLayers; caller may pass a smaller value in
        // geom (via the numGpuLayers field below) for partial offload.
        uint32_t nGpu = geom.numGpuLayers;
        if (nGpu == 0) nGpu = geom.numLayers;
        if (nGpu > geom.numLayers || nGpu > layers.size()) nGpu = geom.numLayers;

        geom_ = geom;
        geom_.numGpuLayers = nGpu;

        auto alloc = [&](void **p, size_t bytes, const char *what) -> bool {
            cudaError_t e = cudaMalloc(p, bytes);
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMalloc(") + what + "): " +
                         cudaGetErrorString(e);
                return false;
            }
            return true;
        };
        auto uploadBytes = [&](void *dst, const void *src, size_t bytes,
                               const char *what) -> bool {
            cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMemcpy(upload ") + what + "): " +
                         cudaGetErrorString(e);
                return false;
            }
            return true;
        };
        // On ANY allocation/upload failure, free everything already allocated
        // in this (or a previous failed) attempt.  Without this, each failed
        // upload leaks ~10.5 GB of VRAM (layer matrices + embed + KV slots are
        // allocated sequentially, and `allocated_` only flips true at the very
        // end), so every subsequent fresh session re-attempts the full upload
        // from an already-VRAM-starved device and eventually exhausts even the
        // CPU fallback's headroom.  The caller latches the failure (see
        // ModelGPUAdapter::gpuUnavailable_) so this cleanup runs at most once
        // per load.
        auto failCleanup = [&](const char *where) {
            std::fprintf(stderr,
                         "[gpu] upload failed at %s (%s); freeing %zu partial "
                         "device layers\n",
                         where, errMsg.c_str(), (layers_ ? nGpu : 0u));
            // destroy() starts with `if (!allocated_) return;`, and upload()
            // only flips allocated_ to true at the VERY END of a successful
            // upload.  Until then destroy() is a no-op, which previously leaked
            // every partially-allocated device buffer on a mid-upload OOM.
            // Bypass the guard so the partial allocations (layer matrices,
            // embedQ_, rope/MRoPE tables, q35 state, KV slots) are released.
            allocated_ = true;
            destroy();
            // Consume the sticky CUDA error left by the failed cudaMalloc.
            // cudaGetLastError() returns AND clears it; without this, the next
            // forward's first cudaGetLastError() (dequantMatrixF16) reports the
            // stale "out of memory" and aborts a healthy hybrid retry.
            (void) cudaGetLastError();
        };
        // Early-return sites below route through a common helper so the
        // partial allocations are always released on failure.
        auto uploadMat = [&](const DeviceMatrix &src, DeviceMatrix &dst,
                             const char *name) -> bool {
            if (src.q == nullptr) return true;
            dst = src;
            if (!alloc(&dst.q, static_cast<uint64_t>(src.rows) * src.rowBytes, name)) {
                failCleanup(name);
                return false;
            }
            if (!uploadBytes(dst.q, src.q,
                             static_cast<uint64_t>(src.rows) * src.rowBytes, name)) {
                failCleanup(name);
                return false;
            }
            // FP16 twin: NO persistent twin.  The prefill path dequantizes
            // ONE matrix per layer on-device into the reusable wF16_ scratch
            // (see dequantMatrixF16), and decode (seqLen==1) never uses fp16
            // GEMMs at all.  This keeps 7B+ quantized models inside VRAM
            // (~14 GB of persistent fp16 twins would OOM the 11 GB card).
            return true;
        };
        auto uploadF32 = [&](float *&dst, const float *src, uint32_t n,
                             const char *name) -> bool {
            if (src == nullptr || n == 0) return true;
            if (!alloc(reinterpret_cast<void **>(&dst), sizeof(float) * n, name)) {
                failCleanup(name);
                return false;
            }
            if (!uploadBytes(dst, src, sizeof(float) * n, name)) {
                failCleanup(name);
                return false;
            }
            return true;
        };
        auto allocRaw = [&](void **p, size_t bytes, const char *what) -> bool {
            if (!alloc(p, bytes, what)) {
                failCleanup(what);
                return false;
            }
            return true;
        };
        auto uploadRaw = [&](void *dst, const void *src, size_t bytes,
                             const char *what) -> bool {
            if (!uploadBytes(dst, src, bytes, what)) {
                failCleanup(what);
                return false;
            }
            return true;
        };

        // ---- Device LayerWeights (first nGpu layers only) ----
        layers_ = new DeviceLayer[nGpu];
        for (uint32_t L = 0; L < nGpu; ++L) {
            const DeviceLayer &hl = layers[L];
            DeviceLayer &dl = layers_[L];
            if (!uploadMat(hl.attnQ, dl.attnQ, "attnQ") ||
                !uploadMat(hl.attnK, dl.attnK, "attnK") ||
                !uploadMat(hl.attnV, dl.attnV, "attnV") ||
                !uploadMat(hl.attnO, dl.attnO, "attnO") ||
                !uploadMat(hl.ffnGate, dl.ffnGate, "ffnGate") ||
                !uploadMat(hl.ffnUp, dl.ffnUp, "ffnUp") ||
                !uploadMat(hl.ffnDown, dl.ffnDown, "ffnDown")) {
                return false;
            }
            // ---- Qwen35 extras (shared GDN block; qwen35moe adds the MoE FFN
            // extras below in its own architecture==2 block) ----
            if (geom_.architecture == 1 || geom_.architecture == 2) {
                // Recurrent / full-attention quantized matrices.
                if (!uploadMat(hl.attnQKV, dl.attnQKV, "attnQKV") ||
                    !uploadMat(hl.attnGate, dl.attnGate, "attnGate") ||
                    !uploadMat(hl.ssmAlphaQ, dl.ssmAlphaQ, "ssmAlphaQ") ||
                    !uploadMat(hl.ssmBetaQ, dl.ssmBetaQ, "ssmBetaQ") ||
                    !uploadMat(hl.ssmOut, dl.ssmOut, "ssmOut")) {
                    return false;
                }
                // ssm_conv1d: F32 [convChannels][convKernel]. The DeviceMatrix
                // descriptor came through toDeviceMatrix (rows/cols set, but
                // rowBytes==0 because type==F32 has no block encoding).  Upload
                // the raw rows*cols*4 bytes as an F32 blob for kQ35Conv1d.
                if (hl.ssmConv1d.q != nullptr && hl.ssmConv1d.rows > 0 &&
                    hl.ssmConv1d.cols > 0) {
                    dl.ssmConv1d.rows = hl.ssmConv1d.rows;
                    dl.ssmConv1d.cols = hl.ssmConv1d.cols;
                    uint64_t cb =
                            static_cast<uint64_t>(hl.ssmConv1d.rows) *
                            hl.ssmConv1d.cols * sizeof(float);
                    if (!alloc(&dl.ssmConv1d.q, static_cast<size_t>(cb),
                               "ssmConv1d"))
                        return false;
                    if (!uploadBytes(dl.ssmConv1d.q, hl.ssmConv1d.q, cb,
                                     "ssmConv1d"))
                        return false;
                }
                // F32 vectors.
                if (!uploadF32(dl.ssmABroadcast, hl.ssmABroadcast,
                               geom_.ssmTimeStepRank, "ssmA") ||
                    !uploadF32(dl.ssmDtBiasFull, hl.ssmDtBiasFull,
                               geom_.ssmTimeStepRank, "ssmDt") ||
                    !uploadF32(dl.ssmNorm, hl.ssmNorm,
                               geom_.ssmInnerSize / std::max(geom_.ssmTimeStepRank, 1u),
                               "ssmNorm") ||
                    !uploadF32(dl.attnQNorm, hl.attnQNorm, geom_.headDim, "attnQNorm") ||
                    !uploadF32(dl.attnKNorm, hl.attnKNorm, geom_.headDim, "attnKNorm") ||
                    !uploadF32(dl.postAttnNorm, hl.postAttnNorm, geom_.hiddenSize,
                               "postAttnNorm")) {
                    return false;
                }
            }
            // ---- Qwen35MoE (qwen35moe) MoE FFN extras ----
            // Full offload (`!moeCpuFn_`): upload the softmax router
            // (ffnGateInpMoe), the 256 per-expert matrices
            // (ffnGateExps/ffnUpExps/ffnDownExpsMoe, ~32B of the model's 35B
            // params — the VRAM hogs) and the shared expert (ffn*Shexp).
            //
            // CPU-expert hybrid (`moeCpuFn_` set): the 256 per-expert matrices
            // STAY OFF the device (~32B in host RAM; only 8/256 are touched
            // per token and those run on the CPU via the callback).  The
            // ROUTER and the SHARED EXPERT are small (router F32 [256x2048]
            // ~2 MB; ffn*Shexp ~12 MB) and are uploaded here so the GPU runs
            // them: the router kernel writes the top-k expert ids + weights
            // into the tiny host mirrors the callback consumes, and the shared
            // expert adds its contribution on the device.  That is what makes
            // IQ2_M (~11.5 GB) and Q4_K_M (~22 GB) fit an 11 GB card with a
            // busy GPU (shared expert runs there) while the CPU covers the
            // routed 8-of-256 experts.
            if (geom_.architecture == 2 &&
                hl.ffnGateInpMoe.q != nullptr) {
                // Router (F32 [expertCount x hidden]) is a raw float blob.
                if (!uploadMat(hl.ffnGateInpMoe, dl.ffnGateInpMoe,
                               "ffnGateInpMoe")) {
                    return false;
                }
                if (!moeCpuFn_) {
                    if (!uploadMat(hl.ffnGateExps, dl.ffnGateExps, "ffnGateExps") ||
                        !uploadMat(hl.ffnUpExps, dl.ffnUpExps, "ffnUpExps") ||
                        !uploadMat(hl.ffnDownExpsMoe, dl.ffnDownExpsMoe,
                                   "ffnDownExpsMoe")) {
                        return false;
                    }
                } else {
                    // ---- CPU-expert hybrid: record the HOST slice geometry for
                    // the per-expert device cache built later (buildExpertCache).
                    // The packed expert matrices stay in host RAM
                    // (ffnGateExps [expertCount*expertFF][H], ffnUpExps same,
                    // ffnDownExpsMoe [expertCount*H][expertFF]); the cache H2Ds
                    // ONE expert (3 slices) at a time.  Row strides come from
                    // the descriptors' block encoding.
                    //
                    // CRITICAL (2026-09-14): the H2D SOURCE must be recorded
                    // PER LAYER into expertCache_[L] (hostGate/hostUp/hostDown).
                    // Every layer has its OWN ffnGateExps/ffnUpExps/
                    // ffnDownExpsMoe host blob, and this loop runs once per
                    // layer; a single global expertHostGate_ would be
                    // overwritten every iteration and end up pointing at the
                    // LAST uploaded layer's matrices — making EVERY layer's
                    // cache execute the last layer's routed-expert weights
                    // (the L0 routed-output divergence root cause).
                    if (hl.ffnGateExps.q != nullptr) {
                        expertHostGate_ = static_cast<const uint8_t *>(hl.ffnGateExps.q);
                        expertRowBytesGate_ = hl.ffnGateExps.rowBytes;
                        expertType_ = hl.ffnGateExps.type;
                        expertCache_[L].hostGate = expertHostGate_;
                    }
                    if (hl.ffnUpExps.q != nullptr) {
                        expertHostUp_ = static_cast<const uint8_t *>(hl.ffnUpExps.q);
                        expertCache_[L].hostUp = expertHostUp_;
                    }
                    if (hl.ffnDownExpsMoe.q != nullptr) {
                        expertHostDown_ = static_cast<const uint8_t *>(hl.ffnDownExpsMoe.q);
                        expertRowBytesDown_ = hl.ffnDownExpsMoe.rowBytes;
                        expertCache_[L].hostDown = expertHostDown_;
                    }
                    expertFF_ = geom_.expertFF;
                    expertCount_ = geom_.expertCount;
                    expertUsed_ = geom_.expertUsedCount;
                    expertHM_ = geom_.hiddenSize;
                }
                // Shared-expert matrices (gate/up/down quantized; the gate
                // vector ffn_gate_inp_shexp is F32 [1 x hidden]).
                if (hl.ffnGateShexp.q != nullptr) {
                    if (!uploadMat(hl.ffnGateShexp, dl.ffnGateShexp,
                                   "ffnGateShexp") ||
                        !uploadMat(hl.ffnUpShexp, dl.ffnUpShexp, "ffnUpShexp") ||
                        !uploadMat(hl.ffnDownShexp, dl.ffnDownShexp,
                                   "ffnDownShexp")) {
                        return false;
                    }
                }
                // ffn_gate_inp_shexp: F32 [1 x hidden] — ONE row blob.  The
                // DeviceMatrix came through toDeviceMatrix with type==F32,
                // which sets block encoding = raw floats (rowBytes=cols*4),
                // rows==1, cols==hidden.  uploadMat would need rows*rowBytes;
                // we handle the F32 blob directly (same as ssm_conv1d).
                if (hl.ffnGateInpShexp.q != nullptr &&
                    hl.ffnGateInpShexp.rows > 0) {
                    const uint64_t cb = static_cast<uint64_t>(hl.ffnGateInpShexp.rows) *
                                        hl.ffnGateInpShexp.cols * sizeof(float);
                    dl.ffnGateInpShexp.rows = hl.ffnGateInpShexp.rows;
                    dl.ffnGateInpShexp.cols = hl.ffnGateInpShexp.cols;
                    if (!allocRaw(&dl.ffnGateInpShexp.q, static_cast<size_t>(cb),
                                  "ffnGateInpShexp"))
                        return false;
                    if (!uploadRaw(dl.ffnGateInpShexp.q, hl.ffnGateInpShexp.q, cb,
                                   "ffnGateInpShexp"))
                        return false;
                }
            }
            uint32_t qLen = geom_.numAttentionHeads * geom_.headDim;
            uint32_t kvLen = geom_.numKVHeads * geom_.headDim;
            // For Qwen2, biases are [qLen]/[kvLen]; for Gemma/Qwen35MoE they are
            // absent (qwen2Bias == 0 routes them to null).
            uint32_t qb = geom_.qwen2Bias ? qLen : 0;
            uint32_t kb = geom_.qwen2Bias ? kvLen : 0;
            if (!uploadF32(dl.attnQBias, hl.attnQBias, qb, "attnQBias") ||
                !uploadF32(dl.attnKBias, hl.attnKBias, kb, "attnKBias") ||
                !uploadF32(dl.attnVBias, hl.attnVBias, kb, "attnVBias") ||
                !uploadF32(dl.rmsNormAttn, hl.rmsNormAttn, geom_.hiddenSize, "rmsAttn") ||
                !uploadF32(dl.rmsNormFFN, hl.rmsNormFFN, geom_.hiddenSize, "rmsFFN")) {
                return false;
            }
        }

        // ---- Final norm ----
        if (geom.finalNorm != nullptr) {
            if (!allocRaw(reinterpret_cast<void **>(&finalNorm_),
                          geom.hiddenSize * sizeof(float), "finalNorm"))
                return false;
            if (!uploadRaw(finalNorm_, geom.finalNorm,
                           geom.hiddenSize * sizeof(float), "finalNorm"))
                return false;
        }

        // ---- Embedding (quantized rows) ----
        if (embedQ != nullptr) {
            uint64_t embedBytes = static_cast<uint64_t>(geom.vocabSize) * embedRowBytes;
            if (!allocRaw(reinterpret_cast<void **>(&embedQ_), embedBytes, "embedQ"))
                return false;
            if (!uploadRaw(embedQ_, embedQ, embedBytes, "embedQ")) return false;
            embedType_ = embedType;
            embedRowBytes_ = embedRowBytes;
        }

        // ---- Separate LM head (quantized rows; nullable for tied heads) ----
        // In partial-offload mode (nGpu < numLayers) the shared tail runs the
        // final RMSNorm + LM head on the CPU (see Model::forward continuation),
        // so a separate output.weight matrix would NEVER be used on the GPU.
        // Skipping the upload keeps ~1 GB of VRAM free for extra offloaded
        // layers (matches llama.cpp's --cpu-moe philosophy: evict the tensors
        // the CPU can compute).  lmHeadQ_ stays null; the full-offload tail
        // falls back to the tied embedding matrix, which is correct because a
        // full offload never has a separate head without uploading it first.
        const bool useGpuLmHead = (lmHeadQ != nullptr) && (nGpu == geom.numLayers);
        if (useGpuLmHead) {
            uint64_t lmBytes = static_cast<uint64_t>(geom.vocabSize) * lmHeadRowBytes;
            if (!allocRaw(reinterpret_cast<void **>(&lmHeadQ_), lmBytes, "lmHeadQ"))
                return false;
            if (!uploadRaw(lmHeadQ_, lmHeadQ, lmBytes, "lmHeadQ")) return false;
            lmHeadType_ = lmHeadType;
            lmHeadRowBytes_ = lmHeadRowBytes;
        }

        // ---- RoPE tables [maxSeqLen][headDim/2] ----
        {
            uint32_t pairs = geom_.headDim / 2;
            std::vector<float> cosT(static_cast<size_t>(geom_.maxSeqLen) * pairs);
            std::vector<float> sinT(static_cast<size_t>(geom_.maxSeqLen) * pairs);
            std::vector<float> freq(pairs);
            for (uint32_t d2 = 0; d2 < pairs; ++d2) {
                freq[d2] = 1.0f / std::pow(geom_.ropeTheta,
                                           static_cast<float>(2 * d2) / geom_.headDim);
            }
            for (uint32_t p = 0; p < geom_.maxSeqLen; ++p) {
                for (uint32_t d2 = 0; d2 < pairs; ++d2) {
                    float ang = static_cast<float>(p) * freq[d2];
                    cosT[static_cast<size_t>(p) * pairs + d2] = std::cos(ang);
                    sinT[static_cast<size_t>(p) * pairs + d2] = std::sin(ang);
                }
            }
            if (!allocRaw(reinterpret_cast<void **>(&ropeCos_),
                          cosT.size() * sizeof(float), "ropeCos") ||
                !allocRaw(reinterpret_cast<void **>(&ropeSin_),
                          sinT.size() * sizeof(float), "ropeSin"))
                return false;
            if (!uploadRaw(ropeCos_, cosT.data(), cosT.size() * sizeof(float),
                           "ropeCos") ||
                !uploadRaw(ropeSin_, sinT.data(), sinT.size() * sizeof(float),
                           "ropeSin"))
                return false;
        }

        // ---- Qwen35 MRoPE cache [maxSeqLen][ropeDimensionCount/2] ----
        // Mirrors Model::applyMRoPE's theta_scale = thetaBase^(-2/nDims) with the
        // cache STARTING theta at 1.0f (not thetaBase: ggml seeds mrope cache
        // theta with the token POSITION, so freq_base enters only through
        // theta_scale).  The standard rope tables above are unused for qwen35.
        if (geom_.architecture == 1 || geom_.architecture == 2) {
            // Shared by qwen35 and qwen35moe (same gated-delta-net block; the
            // MoE arch reuses the MRoPE cache + persistent conv/GDN state).
            const uint32_t nDims = geom_.ropeDimensionCount > 0
                                           ? geom_.ropeDimensionCount
                                           : 64;
            const uint32_t nDimsHalf = nDims / 2;
            const float thetaBase = geom_.ropeTheta;
            const float thetaScale =
                    std::pow(thetaBase, -2.0f / static_cast<float>(nDims));
            std::vector<float> mcos(static_cast<size_t>(geom_.maxSeqLen) * nDimsHalf);
            std::vector<float> msin(static_cast<size_t>(geom_.maxSeqLen) * nDimsHalf);
            for (uint32_t p = 0; p < geom_.maxSeqLen; ++p) {
                float theta = 1.0f;
                for (uint32_t d2 = 0; d2 < nDimsHalf; ++d2) {
                    mcos[static_cast<size_t>(p) * nDimsHalf + d2] =
                            std::cos(static_cast<float>(p) * theta);
                    msin[static_cast<size_t>(p) * nDimsHalf + d2] =
                            std::sin(static_cast<float>(p) * theta);
                    theta *= thetaScale;
                }
            }
            if (!allocRaw(reinterpret_cast<void **>(&mropeCos_),
                          mcos.size() * sizeof(float), "mropeCos") ||
                !allocRaw(reinterpret_cast<void **>(&mropeSin_),
                          msin.size() * sizeof(float), "mropeSin"))
                return false;
            if (!uploadRaw(mropeCos_, mcos.data(), mcos.size() * sizeof(float),
                           "mropeCos") ||
                !uploadRaw(mropeSin_, msin.data(), msin.size() * sizeof(float),
                           "mropeSin"))
                return false;

            // Persistent recurrent-layer state (qwen35) — conv sliding window
            // [(convKernel-1)*convChannels] and gated-delta-net [nVHeads*headV^2]
            // per offloaded GPU layer.  Zero-initialized (fresh session).
            const uint32_t q35KeyDim = geom_.ssmStateSize * geom_.ssmGroupCount;
            const uint32_t q35QkvDim = 2 * q35KeyDim + geom_.ssmInnerSize;
            const uint32_t q35HeadV =
                    geom_.ssmInnerSize / std::max(geom_.ssmTimeStepRank, 1u);
            uint64_t q35ConvBytes =
                    static_cast<uint64_t>(nGpu) * (geom_.ssmConvKernel - 1) *
                    q35QkvDim * sizeof(float);
            uint64_t q35GdnBytes =
                    static_cast<uint64_t>(nGpu) * geom_.ssmTimeStepRank * q35HeadV *
                    q35HeadV * sizeof(float);
            if (q35ConvBytes == 0) q35ConvBytes = 1;
            if (q35GdnBytes == 0) q35GdnBytes = 1;
            if (!allocRaw(reinterpret_cast<void **>(&q35ConvState_),
                          static_cast<size_t>(q35ConvBytes), "q35ConvState") ||
                !allocRaw(reinterpret_cast<void **>(&q35GdnState_),
                          static_cast<size_t>(q35GdnBytes), "q35GdnState"))
                return false;
            cudaMemsetAsync(q35ConvState_, 0,
                            static_cast<size_t>(q35ConvBytes), g_stream);
            cudaMemsetAsync(q35GdnState_, 0,
                            static_cast<size_t>(q35GdnBytes), g_stream);
        }

        // ---- KV cache (device, only nGpu layers) ----
        uint64_t kvBytes = static_cast<uint64_t>(geom_.maxSeqLen) *
                           geom_.numKVHeads * geom_.headDim * sizeof(float);
        if (!allocRaw(reinterpret_cast<void **>(&kvK_),
                      static_cast<size_t>(nGpu) * kvBytes, "kvK") ||
            !allocRaw(reinterpret_cast<void **>(&kvV_),
                      static_cast<size_t>(nGpu) * kvBytes, "kvV"))
            return false;
        cudaMemsetAsync(kvK_, 0, static_cast<size_t>(nGpu) * kvBytes, g_stream);
        cudaMemsetAsync(kvV_, 0, static_cast<size_t>(nGpu) * kvBytes, g_stream);

        kvPos_ = 0;
        allocated_ = true;
        cudaStreamSynchronize(g_stream);
        return true;
    }

    // 2-way set-associative expert cache set index for expert e: the
    // upper bits of the Knuth multiply-high hash, masked to the runtime
    // SET count (TINYCODER_MOE_WAYS = sets, default 16).  The set is a
    // power of 2 <= kExpertCacheSets so the static tag grid covers the
    // prefix of sets used.  Slot index within the arena =
    // set*kExpertCacheWays + way (way chosen by the tag/LRU lookup in
    // ensureExpertCached).
    // Set index for expert e (2-way set-associative): the upper bits of
    // the Knuth multiply-high hash, masked to the runtime SET count
    // (TINYCODER_MOE_WAYS = sets, default 16).  The set is a power of 2
    // <= kExpertCacheSets so the static tag grid covers the prefix used.
    // Slot index within the arena = set*kExpertCacheWays + way (way from
    // the tag/LRU lookup in ensureExpertCached).
    uint32_t GPUModel::expertCacheSet(uint32_t e) {
        const uint32_t sets = expertCacheSets_ ? expertCacheSets_
                                               : kExpertCacheSetsDefault;
        const uint32_t h = static_cast<uint32_t>(
                (static_cast<uint64_t>(e) * 2654435761u) >> 27u);
        // Power-of-2 -> cheap mask; otherwise modulo (harmless; only used
        // for the TINYCODER_MOE_WAYS capacity A/B with e.g. 24 sets).
        return (sets & (sets - 1u)) == 0u ? (h & (sets - 1u)) : (h % sets);
    }

    static double bwSelfTestHostToPinned(const uint8_t *src, uint8_t *dst,
                                         size_t bytes, int iters) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            std::memcpy(dst, src, bytes);
        double sec = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
        return (double) bytes * iters / sec / (1ull << 30);
    }

    bool GPUModel::buildExpertCache(std::string &errMsg) {
        // Hybrid-only: the 256 per-expert matrices stay in host RAM, so the
        // per-expert GEMVs cannot read layers_[L].ffnGateExps.q (host pointer).
        // Allocate a 2-way SET-ASSOCIATIVE device cache: kExpertCacheSets sets
        // x 2 ways = 32 slots per layer (TINYCODER_MOE_WAYS = sets, default
        // 16).  Each slot is ONE CONTIGUOUS [gate][up][down] block
        // (slotBytes = 2*gateSlice + downSlice) so a miss is a single pinned
        // DMA H2D.  With 40 layers this costs ~4.1 GB at 32 slots / 3.3 MB
        // per expert (see the VRAM budget in the header comment).
        if (!moeCpuFn_ || expertCacheBuilt_) return true;
        // Runtime set count: TINYCODER_MOE_WAYS (power of 2, clamped to
        // kExpertCacheSets).  A/B tool: more sets shrink conflicts (more
        // VRAM); fewer sets shrink VRAM (more conflicts).
        {
            const char *wv = std::getenv("TINYCODER_MOE_WAYS");
            if (wv != nullptr && wv[0] != '\0') {
                long v = std::strtol(wv, nullptr, 10);
                // Accept non-power-of-2 too (e.g. 24 sets = 48 slots for the
                // capacity A/B); the hash still masks to a power-of-2 PREFIX
                // (hash % sets via rejection-free mul-based range) so any set
                // count <= kExpertCacheSets works.
                if (v >= 2 && v <= static_cast<long>(kExpertCacheSets)) {
                    expertCacheSets_ = static_cast<uint32_t>(v);
                }
            } else {
                expertCacheSets_ = kExpertCacheSetsDefault;
            }
        }
        {
            const char *dnE = std::getenv("TINYCODER_DUMP_MOE");
            if (dnE != nullptr && dnE[0] != '\0' && dnE[0] != '0') {
                std::fprintf(stderr,
                             "[cache build] expertFF=%u expertHM=%u count=%u "
                             "gateRowBytes=%u downRowBytes=%u type=%u "
                             "sets=%u slots=%u\n",
                             expertFF_, expertHM_, expertCount_,
                             expertRowBytesGate_, expertRowBytesDown_,
                             expertType_, expertCacheSets_,
                             expertCacheSets_ * kExpertCacheWays);
            }
        }
        const uint32_t nGpu = geom_.numGpuLayers ? geom_.numGpuLayers
                                                 : geom_.numLayers;
        if (expertFF_ == 0 || expertHM_ == 0 || expertCount_ == 0 ||
            expertHostGate_ == nullptr || expertHostDown_ == nullptr) {
            errMsg = "buildExpertCache: missing host expert geometry (hybrid "
                     "upload did not record it)";
            return false;
        }
        const uint64_t gateSlice = static_cast<uint64_t>(expertFF_) *
                                   expertRowBytesGate_;
        const uint64_t downSlice = static_cast<uint64_t>(expertHM_) *
                                   expertRowBytesDown_;
        const uint64_t slotBytes = 2 * gateSlice + downSlice;
        const uint32_t sets = expertCacheSets_ ? expertCacheSets_
                                               : kExpertCacheSetsDefault;
        for (uint32_t L = 0; L < nGpu; ++L) {
            // One contiguous slot-major arena per layer: slot s owns
            // [s*slotBytes, (s+1)*slotBytes) = [gate][up][down]; slot index =
            // set*2 + way.
            uint8_t *arena = nullptr;
            cudaError_t e = cudaMalloc(
                    &arena, static_cast<size_t>(slotBytes * sets *
                                                kExpertCacheWays));
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMalloc(expertCache): ") +
                         cudaGetErrorString(e);
                // Free what we already built, leave expertCacheBuilt_ false.
                for (uint32_t l2 = 0; l2 < L; ++l2) {
                    if (expertCache_[l2].gateQ) cudaFree(expertCache_[l2].gateQ);
                    expertCache_[l2] = ExpertSlot();
                }
                return false;
            }
            expertCache_[L].gateQ = arena;
            // Slot-0 interior pointers (per-slot offsets are re-derived in
            // ensureExpertCached from the slot-major stride).
            expertCache_[L].upQ = arena + static_cast<size_t>(gateSlice);
            expertCache_[L].downQ =
                    arena + static_cast<size_t>(2 * gateSlice);
            for (uint32_t s2 = 0; s2 < sets; ++s2) {
                expertCacheTag_[L][s2][0] = -1;
                expertCacheTag_[L][s2][1] = -1;
                expertCacheLru_[L][s2] = 0;
            }
            // ---- Keep the host expert blobs resident (2026-09-16).  The
            // weights are zero-copy mmap'd (GGUFLoader), so with a ~35 GB
            // model on a 31 GB host the kernel evicts expert pages while the
            // router touches other layers; the per-miss fill memcpy then
            // page-faults and re-reads 3.3 MB from the file (measured:
            // per-miss fillEnq 0.31-0.54 ms warm vs 3-18 ms cold — a 30x
            // swing that exactly tracks rankLoopMs/fillMs ballooning to
            // 2000-5000 ms/token).  Long-term (per-expert) residency is a
            // kExpertCacheWays-way per-layer working set; MADV_WILLNEED is a
            // pure hint that prefetches the pages without pinning them.
            // Opt-out with TINYCODER_NO_WILLNEED=1 (measurement A/B).
            {
                const char *noWn = std::getenv("TINYCODER_NO_WILLNEED");
                if (noWn == nullptr || noWn[0] == '\0' || noWn[0] == '0') {
#if defined(__linux__)
                    const uint8_t *hGate = expertCache_[L].hostGate
                                                   ? expertCache_[L].hostGate
                                                   : expertHostGate_;
                    const uint8_t *hUp = expertCache_[L].hostUp
                                                 ? expertCache_[L].hostUp
                                                 : expertHostUp_;
                    const uint8_t *hDown = expertCache_[L].hostDown
                                                   ? expertCache_[L].hostDown
                                                   : expertHostDown_;
                    if (hGate)
                        ::madvise(const_cast<void *>(static_cast<const void *>(hGate)),
                                  static_cast<size_t>(expertCount_) * gateSlice,
                                  MADV_WILLNEED);
                    if (hUp)
                        ::madvise(const_cast<void *>(static_cast<const void *>(hUp)),
                                  static_cast<size_t>(expertCount_) * gateSlice,
                                  MADV_WILLNEED);
                    if (hDown)
                        ::madvise(const_cast<void *>(static_cast<const void *>(hDown)),
                                  static_cast<size_t>(expertCount_) * downSlice,
                                  MADV_WILLNEED);
#endif
                }
            }
        }
        expertCacheBuilt_ = true;
        // TINYCODER_BW_SELFTEST=1: measure the two H2D legs (host->pinned
        // memcpy fill and pinned->device DMA) so a bandwidth ceiling can be
        // attributed to the right one.  The MoE block remains ~225 ms/token
        // even with pinned staging, so this is the decisive measurement.
        if (std::getenv("TINYCODER_BW_SELFTEST") != nullptr) {
            const uint8_t *src = expertHostGate_;
            // Use the same registration scheme as the real path so the
            // selftest measures the ACTUAL staging buffer behavior.
            void *pp = nullptr;
            posix_memalign(&pp, 4096, static_cast<size_t>(slotBytes));
            cudaHostRegister(pp, static_cast<size_t>(slotBytes),
                             cudaHostRegisterMapped | cudaHostRegisterPortable);
            expertStage_[0] = static_cast<uint8_t *>(pp);
            double fillGB = bwSelfTestHostToPinned(src, expertStage_[0],
                                                   static_cast<size_t>(slotBytes),
                                                   64);
            // Pinned -> device DMA (one slot, repeated).
            uint8_t *dst = expertCache_[0].gateQ;
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < 64; ++i) {
                cudaMemcpyAsync(dst, expertStage_[0],
                                static_cast<size_t>(slotBytes),
                                cudaMemcpyHostToDevice, g_stream);
            }
            cudaStreamSynchronize(g_stream);
            double dmaSec = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - t1)
                                    .count();
            double dmaGB = (double) slotBytes * 64 / dmaSec / (1ull << 30);
            std::fprintf(stderr,
                         "[bw selftest] slotBytes=%llu hostFill=%.1f GB/s "
                         "pinnedDMA=%.1f GB/s\n",
                         static_cast<unsigned long long>(slotBytes), fillGB,
                         dmaGB);
        }
        return true;
    }

    const uint8_t *GPUModel::ensureExpertCached(uint32_t L, uint32_t e,
                                                std::string &errMsg,
                                                const uint8_t **downQOut) {
        if (!expertCacheBuilt_ || e >= expertCount_) return nullptr;
        const uint32_t set = expertCacheSet(e);
        ExpertSlot &s = expertCache_[L];
        const uint64_t gateSlice = static_cast<uint64_t>(expertFF_) *
                                   expertRowBytesGate_;
        const uint64_t downSlice = static_cast<uint64_t>(expertHM_) *
                                   expertRowBytesDown_;
        const uint64_t slotBytes = 2 * gateSlice + downSlice;
        const uint32_t sets = expertCacheSets_ ? expertCacheSets_
                                               : kExpertCacheSetsDefault;
        if (set >= sets) {
            errMsg = "ensureExpertCached: set out of runtime set count";
            return nullptr;
        }
        // ---- 2-way set-associative lookup.  The set is set; the two ways
        // (0 and 1) hold their own tags.  Arena slot = set*2 + way.
        // ---- SLOT-MAJOR arena (2026-09-14): slot s owns the CONTIGUOUS range
        // [s*slotBytes, (s+1)*slotBytes) = [gate][up][down].  gateQ is the
        // layer arena base; up = base + gateSlice; down = base + 2*gateSlice.
        const uint8_t *g0 = static_cast<const uint8_t *>(s.gateQ) +
                            static_cast<size_t>(set * kExpertCacheWays) *
                                    static_cast<size_t>(slotBytes);
        const uint8_t *g1 = g0 + static_cast<size_t>(slotBytes);
        // Hit: one of the two ways holds expert e.
        if (expertCacheTag_[L][set][0] == static_cast<int32_t>(e)) {
            // Touch makeLRU = way1 (so way0 is now the MRU of the pair).
            expertCacheLru_[L][set] = 1;
            if (downQOut) *downQOut = g0 + 2 * gateSlice;
            return g0;
        }
        if (expertCacheTag_[L][set][1] == static_cast<int32_t>(e)) {
            expertCacheLru_[L][set] = 0;
            if (downQOut) *downQOut = g1 + 2 * gateSlice;
            return g1;
        }
        // ---- Miss: choose the LRU way (expertCacheLru_ bit; initialize to
        // way0 if both are empty so the first miss fills way0).
        uint32_t victim = expertCacheLru_[L][set];
        const uint8_t *gBase = (victim == 0) ? g0 : g1;
        if (expertCacheTag_[L][set][0] < 0 && expertCacheTag_[L][set][1] < 0)
            victim = 0, gBase = g0;
        // Flip the LRU bit so the OTHER way is now LRU (2-way pseudo-LRU:
        // after filling victim, the survivor becomes LRU next).
        expertCacheLru_[L][set] = (victim == 0) ? 1u : 0u;
        if (downQOut) *downQOut = gBase + 2 * gateSlice;
        // ---- Miss path: ONE contiguous pinned H2D of the whole slot on
        // g_stream.  The host expert bytes are PINNED into a per-miss ring
        // buffer (host-side memcpy is synchronous, THEN one cudaMemcpyAsync
        // from PINNED memory reaches the PCIe 3.0 x16 DMA ceiling ~6 GB/s —
        // the old path did 3 pageable bounces at ~2.4 GB/s; measured 500
        // MB/token ≈ 225 ms/token in moeBlockMs).  Copies stay on g_stream
        // (NOT g_stream2): the GEMVs that read the slices run on g_stream,
        // and a g_stream2 H2D without a matching stream-wait lets the GEMVs
        // read stale bytes (measured: broke SequentialDecodeArgmaxAgrees with
        // the 2026-09-14 prefetch attempt).  A torn read between the two
        // stage copies is harmless because both copies are bit-identical
        // (the HOST expert blobs are immutable constants, so every fill
        // writes the exact same bytes).
        // ----
        // Per-layer H2D source (recorded in upload(): every layer has its OWN
        // expert host blobs — see expertCache_[L].hostGate/Up/Down).
        const uint8_t *srcGate = s.hostGate ? s.hostGate : expertHostGate_;
        const uint8_t *srcUp = s.hostUp ? s.hostUp : expertHostUp_;
        const uint8_t *srcDown = s.hostDown ? s.hostDown : expertHostDown_;
        // Allocate (grow-on-demand) the pinned stage ring.  MEASURED
        // 2026-09-14 (TINYCODER_BW_SELFTEST): cudaHostAlloc (write-combined
        // mapped pages) drops the HOST-SIDE fill to 0.8 GB/s — the memcpy into
        // WC memory is the bottleneck, not the DMA (11.5 GB/s to the device).
        // malloc + cudaHostRegister keeps the pages CACHEABLE so the host fill
        // runs at DRAM speed, and the bits then DMA over at the same wire
        // speed.  Registered pages must be page-aligned (posix_memalign).
        if (expertStageBytes_ < slotBytes) {
            for (uint32_t b = 0; b < kExpertStageBufs; ++b) {
                if (expertStage_[b]) {
                    cudaHostUnregister(expertStage_[b]);
                    std::free(expertStage_[b]);
                }
                expertStage_[b] = nullptr;
                if (expertStageEv_[b]) {
                    cudaEventDestroy(expertStageEv_[b]);
                    expertStageEv_[b] = nullptr;
                }
            }
            for (uint32_t b = 0; b < kExpertStageBufs; ++b) {
                void *pp = nullptr;
                if (posix_memalign(&pp, 4096, static_cast<size_t>(slotBytes)) !=
                    0) {
                    errMsg = "posix_memalign(expertStage) failed";
                    return nullptr;
                }
                cudaError_t eP = cudaHostRegister(
                        pp, static_cast<size_t>(slotBytes),
                        cudaHostRegisterMapped | cudaHostRegisterPortable);
                if (eP != cudaSuccess) {
                    std::free(pp);
                    errMsg = std::string("cudaHostRegister(expertStage): ") +
                             cudaGetErrorString(eP);
                    return nullptr;
                }
                expertStage_[b] = static_cast<uint8_t *>(pp);
                cudaEventCreateWithFlags(&expertStageEv_[b],
                                         cudaEventDisableTiming);
            }
            expertStageBytes_ = slotBytes;
        }
        const uint32_t st = (expertStageFlag_++) % kExpertStageBufs;
        // TINYCODER_TRACE_MISS=1: per-miss wall time split (host fill + memcpy
        // enqueue) to attribute the moeBlockMs budget.
        bool trMiss = false;
        auto tMiss0 = std::chrono::steady_clock::now();
        if (std::getenv("TINYCODER_TRACE_MISS") != nullptr) trMiss = true;
        // ---- DRAIN-BEFORE-REFILL (2026-09-15 root-cause, parity): the stage
        // ring is shared by ALL fills, and with >kExpertStageBufs misses per
        // forward the round-robin claim wraps — miss k+8 refills stage st=a
        // with a DIFFERENT expert's bytes while the copy engine may still be
        // DMA-ing miss k's H2D OUT of that stage.  That tears the pinned
        // source mid-transfer and lands blended expert bytes in the device
        // slot, deterministically corrupting the routed-expert weights (both
        // the batched fast path AND the TINYCODER_MOE_SLOW per-rank path
        // failed SequentialDecodeArgmaxAgrees with the SAME corrupted tokens
        // while the CPU-expert fallback — which never touches this ring —
        // passed).  The old "the host blobs are immutable so re-fill writes
        // the same bytes" argument only holds when the SAME expert refills
        // the stage; a wrap refills with DIFFERENT bytes.  Wait for the
        // previous owner's DMA to drain before overwriting (cheap when the
        // copy already completed — the common case when fills run ahead).
        if (cudaEventQuery(expertStageEv_[st]) != cudaSuccess) {
            cudaEventSynchronize(expertStageEv_[st]);
        }
        // Synchronous host fill into PINNED memory (memory-bandwidth bound;
        // ~3.3 MB ≈ tens of µs) — then ONE DMA H2D per miss.
        std::memcpy(expertStage_[st],
                    srcGate + static_cast<uint64_t>(e) * gateSlice,
                    static_cast<size_t>(gateSlice));
        std::memcpy(expertStage_[st] + gateSlice,
                    srcUp + static_cast<uint64_t>(e) * gateSlice,
                    static_cast<size_t>(gateSlice));
        std::memcpy(expertStage_[st] + 2 * gateSlice,
                    srcDown + static_cast<uint64_t>(e) * downSlice,
                    static_cast<size_t>(downSlice));
        // H2D on g_stream2 (the COPY engine): the GEMVs that read this slot
        // run on g_stream and WAIT on expertStageEv_[st] (see the caller's
        // cudaStreamWaitEvent before the per-rank GEMV launches).  This lets
        // the copy engine fill slot i+1 CONCURRENTLY with the SMs executing
        // slot i's GEMVs — the DMA and compute overlap instead of serializing
        // on one stream.  Previously the fill ran in-order on g_stream, so the
        // GPU sat idle while the copy engine H2D'd every miss (~0.48 ms each
        // at cold DRAM) before the next GEMV could start: the fill+DMA was
        // on the GPU's critical path (~120-170 ms/token of g_stream serial
        // time measured 2026-09-15).  g_stream2's WAR hazard is handled by
        // the event wait (never a raw stream2 launch without the wait, which
        // is what broke SequentialDecodeArgmaxAgrees in the 2026-09-14
        // prefetch attempt).
        cudaMemcpyAsync(const_cast<uint8_t *>(gBase), expertStage_[st],
                        static_cast<size_t>(slotBytes), cudaMemcpyHostToDevice,
                        g_stream2);
        cudaEventRecord(expertStageEv_[st], g_stream2);
        // WAR barrier: the GEMVs that read this slot run on g_stream; make
        // them WAIT for the g_stream2 fill to complete.  Without this the SMs
        // could read the victim slot's OLD bytes (or a half-filled slot) while
        // the copy engine is still writing it — the exact race that broke
        // SequentialDecodeArgmaxAgrees in the 2026-09-14 prefetch attempt.
        // The wait costs nothing when the copy already finished (fills stay
        // ahead), and the pipelining gain is: while g_stream's SMs execute
        // slot i's GEMVs, the copy engine is already H2D-ing slot i+1.
        cudaStreamWaitEvent(g_stream, expertStageEv_[st], 0);
        expertCacheTag_[L][set][victim] = static_cast<int32_t>(e);
        if (trMiss) {
            double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - tMiss0)
                                .count();
            std::fprintf(stderr,
                         "[miss] L=%u e=%u set=%u way=%u fillEnq=%.3f ms "
                         "slotBytes=%llu\n",
                         L, e, set, victim, ms,
                         static_cast<unsigned long long>(slotBytes));
        }
        // TINYCODER_DUMP_MOE=1: byte-compare the JUST-COPIED gate slice
        // against the host rows for the same expert (L==0 only).  This is the
        // decisive check for "cache slice content wrong" vs "kernel wrong":
        // if the first bytes differ, the H2D/slot layout feeds wrong weight
        // bytes into the expert GEMVs.
        {
            const char *dnC = std::getenv("TINYCODER_DUMP_MOE");
            if (L == 0 && dnC != nullptr && dnC[0] != '\0' && dnC[0] != '0') {
                cudaStreamSynchronize(g_stream);
                const size_t cmpN = 2 * 34;// 2 Q8_0 blocks (row 0, col 0..63)
                std::vector<uint8_t> dev(cmpN), host(cmpN);
                cudaMemcpy(dev.data(), gBase, cmpN, cudaMemcpyDeviceToHost);
                // Compare against the PER-LAYER host blob (layer 0) — the
                // global expertHostGate_ is only a fallback now.
                const uint8_t *hsrc = srcGate +
                                      static_cast<uint64_t>(e) * expertFF_ *
                                              expertRowBytesGate_;
                std::memcpy(host.data(), hsrc, cmpN);
                size_t ndiff = 0;
                size_t firstDiff = cmpN;
                for (size_t i = 0; i < cmpN; ++i) {
                    if (dev[i] != host[i]) {
                        if (ndiff == 0) firstDiff = i;
                        ++ndiff;
                    }
                }
                std::fprintf(stderr,
                             "[cache cmp] L=0 set=%u way=%u e=%u dev[0..3]=%02x%02x"
                             "%02x%02x host[0..3]=%02x%02x%02x%02x "
                             "ndiff=%zu firstDiff=%zu\n",
                             set, victim, e, dev[0], dev[1], dev[2], dev[3],
                             host[0], host[1], host[2], host[3], ndiff,
                             firstDiff);
            }
        }
        return gBase;
    }

    bool GPUModel::dequantMatrixF16(const DeviceMatrix &m, void **outBuf,
                                    std::string &errMsg) {
        if (m.q == nullptr || m.rows == 0 || m.cols == 0) {
            *outBuf = nullptr;
            return true;
        }
        // Cache by source matrix: re-dequantizing the same matrix (e.g. two
        // GEMM calls sharing one layer's matrix) is wasted work.
        if (wF16SrcQ_ == m.q) {
            *outBuf = wF16_;
            return true;
        }
        const uint64_t need = static_cast<uint64_t>(m.rows) * m.cols * sizeof(__half);
        if (need > wF16Bytes_) {
            if (wF16_) cudaFree(wF16_);
            wF16_ = nullptr;
            wF16Bytes_ = 0;
            cudaError_t e = cudaMalloc(&wF16_, need);
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMalloc(wF16): ") + cudaGetErrorString(e);
                return false;
            }
            wF16Bytes_ = need;
        }
        launchDequantF16(m.type, m.q, wF16_, m.rows, m.cols, m.rowBytes,
                         m.blocksPerRow);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            errMsg = std::string("launchDequantF16: ") + cudaGetErrorString(e);
            return false;
        }
        wF16SrcQ_ = m.q;
        *outBuf = wF16_;
        return true;
    }

    bool GPUModel::forwardQwen35Prefix(const std::vector<int32_t> &tokens,
                                       bool computeAllLogits, float *logitsOut,
                                       std::string &errMsg) {
        if (!allocated_) {
            errMsg = "GPU model not uploaded";
            return false;
        }
        if (tokens.empty()) return true;
        if (geom_.architecture != 1) {
            errMsg = "forwardQwen35Prefix called for non-qwen35 geometry";
            return false;
        }
        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t nGpu = geom_.numGpuLayers;
        if (nGpu == 0) nGpu = geom_.numLayers;
        bool fullOffload = (nGpu == geom_.numLayers);
        const uint32_t H = geom_.hiddenSize, I = geom_.intermediateSize;
        const uint32_t nHeads = geom_.numAttentionHeads;
        const uint32_t nKV = geom_.numKVHeads;
        const uint32_t hd = geom_.headDim;
        const uint32_t qLen = nHeads * hd, kvLen = nKV * hd;
        const uint32_t pos = static_cast<uint32_t>(kvPos_);

        // Qwen35 dense-hybrid geometry.
        const uint32_t dInner = geom_.ssmInnerSize;           // 6144
        const uint32_t headK = geom_.ssmStateSize;            // 128
        const uint32_t nKHeads = geom_.ssmGroupCount;         // 16
        const uint32_t nVHeads = geom_.ssmTimeStepRank;       // 48
        const uint32_t headV = dInner / std::max(nVHeads, 1u);// 128
        const uint32_t keyDim = headK * nKHeads;              // 2048
        const uint32_t valueDim = headV * nVHeads;            // 6144
        const uint32_t convChannels = 2 * keyDim + valueDim;  // 10240
        const uint32_t convK = geom_.ssmConvKernel;           // 4
        const uint32_t qkvDim = 2 * keyDim + valueDim;
        const uint32_t qgDims = nHeads * 2 * hd;

        if (!ensureScratch(seqLen, errMsg)) return false;
        DeviceScratch &s = scratch_;
        cudaMemcpyAsync(s.tokens, tokens.data(), seqLen * sizeof(int32_t),
                        cudaMemcpyHostToDevice, g_stream);
        cudaStreamSynchronize(g_stream);

        // ---- Embedding (K-quant / legacy).  Mirror forward()'s dispatch. ----
        {
            const uint32_t blocksPerRow = hostBlocksPerRow(embedType_, H);
            const uint8_t *eQ = reinterpret_cast<const uint8_t *>(embedQ_);
            if (embedType_ == kTypeQ5K || embedType_ == kTypeQ5_0) {
                kEmbedDequantQ5K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ4K) {
                kEmbedDequantQ4K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ8_0) {
                kEmbedDequantSmall32<kTypeQ8_0><<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else {
                kEmbedDequant<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            }
        }

        // Dense projection helper: GEMV (decode) or fp16 cuBLAS GEMM (prefill).
        // The fp16 X-side twin buffer is selected by the input width (hidden /
        // intermediate / attention-out widths) from the shared scratch.
        auto gemv = [&](const DeviceMatrix &m, const float *x, float *y,
                        uint32_t nRows, uint32_t nCols,
                        const char *what) -> bool {
            if (m.q == nullptr || nRows == 0) return true;
            if (seqLen == 1) {
                launchQGemv(m.type, m.q, x, y, nRows, nCols, m.rowBytes,
                            m.blocksPerRow, q8k_, q8kBytes_);
                return true;
            }
            // Batched: y[seqLen][nRows] = x[seqLen][nCols] @ W[nRows][nCols]^T
            uint16_t *h16 = (nCols == H)   ? s.hiddenF16
                            : (nCols == I) ? s.gateF16
                                           : s.attnOutF16;
            kF32ToF16<<<(seqLen * nCols + 511) / 512, 256, 0, g_stream>>>(
                    x, reinterpret_cast<__half2 *>(h16), (seqLen * nCols) / 2);
            const uint16_t *w16 = nullptr;
            if (m.type == kTypeF32) {
                // Small F32 matrices (ssm_alpha / ssm_beta): convert the raw
                // weight blob to fp16 in the reusable wF16_ scratch, then GEMM.
                const uint64_t need =
                        static_cast<uint64_t>(m.rows) * m.cols * sizeof(__half);
                if (need > wF16Bytes_) {
                    if (wF16_) cudaFree(wF16_);
                    wF16_ = nullptr;
                    wF16Bytes_ = 0;
                    cudaError_t e = cudaMalloc(&wF16_, need);
                    if (e != cudaSuccess) {
                        errMsg = std::string("cudaMalloc(wF16 q35 F32): ") +
                                 cudaGetErrorString(e);
                        return false;
                    }
                    wF16Bytes_ = need;
                }
                kF32ToF16<<<(m.rows * m.cols + 511) / 512, 256, 0, g_stream>>>(
                        static_cast<const float *>(m.q),
                        reinterpret_cast<__half2 *>(wF16_), (m.rows * m.cols) / 2);
                w16 = static_cast<const uint16_t *>(wF16_);
            } else {
                void *wF = nullptr;
                if (!dequantMatrixF16(m, &wF, errMsg)) return false;
                w16 = static_cast<const uint16_t *>(wF);
            }
            float alpha = 1.0f, beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(
                    g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(nRows),
                    static_cast<int>(seqLen), static_cast<int>(nCols), &alpha,
                    w16, CUDA_R_16F, static_cast<int>(nCols), h16, CUDA_R_16F,
                    static_cast<int>(nCols), &beta, y, CUDA_R_32F,
                    static_cast<int>(nRows), CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                errMsg = std::string("cublasGemmEx ") + what + " failed";
                return false;
            }
            return true;
        };

        const float invSqrtV = 1.0f / std::sqrt(static_cast<float>(headV));
        const uint32_t nDims = geom_.ropeDimensionCount > 0
                                       ? geom_.ropeDimensionCount
                                       : 64;

        // ---- Offloaded layer prefix ----
        // The MTP (NextN) block (layer >= nRealLayers) is never executed in the
        // main decode pass (the CPU forwardQwen35Layer guards layer >= nLayer;
        // llama.cpp runs it only via a separate draft graph).
        const uint32_t nRealLayers =
                geom_.numLayers - geom_.nextnPredictLayers;
        for (uint32_t L = 0; L < nGpu && L < nRealLayers; ++L) {
            const DeviceLayer &w = layers_[L];

            // ---- Attention / recurrent block ----
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.rmsNormAttn, H, seqLen, 1e-6f);

            if ((L + 1) % geom_.fullAttentionInterval != 0) {
                // =================== Recurrent (gated delta net) ===========
                if (!gemv(w.attnQKV, s.norm, s.qkv, qkvDim, H, "attnQKV") ||
                    !gemv(w.attnGate, s.norm, s.q35z, dInner, H, "attnGate") ||
                    !gemv(w.ssmBetaQ, s.norm, s.q35Beta, nVHeads, H,
                          "ssmBetaQ") ||
                    !gemv(w.ssmAlphaQ, s.norm, s.q35GateV, nVHeads, H,
                          "ssmAlphaQ")) {
                    return false;
                }
                // beta=sigmoid(beta); gateV=softplus(alpha+dt.bias)*ssm_a
                kQ35BetaGate<<<seqLen, 64, 0, g_stream>>>(
                        s.q35Beta, s.q35GateV, w.ssmDtBiasFull, w.ssmABroadcast,
                        s.q35Beta, s.q35GateV, seqLen, nVHeads);

                // Sequential recurrence over tokens (per-layer persistent state).
                float *q35ConvState =
                        q35ConvState_ + static_cast<size_t>(L) * (convK - 1) * convChannels;
                float *q35GdnState =
                        q35GdnState_ + static_cast<size_t>(L) * nVHeads * headV * headV;
                const float *convW = static_cast<const float *>(w.ssmConv1d.q);
                const uint32_t convBlocks = (convChannels + 255) / 256;
                for (uint32_t st = 0; st < seqLen; ++st) {
                    const float *qkv = s.qkv + static_cast<size_t>(st) * qkvDim;
                    kQ35ConvStep<<<convBlocks, 256, 0, g_stream>>>(
                            convW, qkv, s.convIn, s.convOut, q35ConvState,
                            convChannels, convK);
                    kQ35L2NormRepeat<<<1, 128, 0, g_stream>>>(
                            s.convOut, s.qN, s.kN, s.vN, headK, headV, nKHeads,
                            nVHeads, keyDim, valueDim, 1);
                    const float *betaRow = s.q35Beta + static_cast<size_t>(st) * nVHeads;
                    const float *gateRow = s.q35GateV + static_cast<size_t>(st) * nVHeads;
                    float *gdnRow = s.gdnOut + static_cast<size_t>(st) * valueDim;
                    // TINYCODER_GDN_SEQ=1: sequential per-head loop (parity
                    // bisection, same as the decode site below).
                    const bool gdnSeq = (std::getenv("TINYCODER_GDN_SEQ") != nullptr);
                    if (gdnSeq) {
                        for (uint32_t hv = 0; hv < nVHeads; ++hv) {
                            kQ35GdnStep<128><<<1, 128, 0, g_stream>>>(
                                    s.qN + static_cast<size_t>(hv) * headK,
                                    s.kN + static_cast<size_t>(hv) * headK,
                                    s.vN + static_cast<size_t>(hv) * headV,
                                    betaRow, gateRow,
                                    q35GdnState + static_cast<size_t>(hv) * headV * headV,
                                    gdnRow + static_cast<size_t>(hv) * headV, nVHeads,
                                    hv, invSqrtV);
                        }
                    } else {
                        kQ35GdnStepBatch<128><<<nVHeads, 128, 0, g_stream>>>(
                                s.qN, s.kN, s.vN, betaRow, gateRow, q35GdnState,
                                gdnRow, nVHeads, invSqrtV);
                    }
                }
                // Gated RMSNorm (per value-head) × silu(z) over the batch.
                kQ35GatedRmsNorm<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                        s.q35z, s.gdnOut, w.ssmNorm, headV, nVHeads, seqLen);
                if (!gemv(w.ssmOut, s.gdnOut, s.attnProj, H, valueDim,
                          "ssmOut")) {
                    return false;
                }
            } else {
                // =================== Full attention =======================
                if (!gemv(w.attnQ, s.norm, s.q35QGate, qgDims, H, "attnQ") ||
                    !gemv(w.attnK, s.norm, s.k, kvLen, H, "attnK") ||
                    !gemv(w.attnV, s.norm, s.v, kvLen, H, "attnV")) {
                    return false;
                }
                // Per-head Q RMSNorm (Q slice = first headDim of each 2*headDim
                // fused Q+gate row) and K RMSNorm (dense, in place).
                kQ35HeadRmsNorm2<<<seqLen * nHeads, 128, 128 * sizeof(float),
                                   g_stream>>>(s.q35QGate, s.q, w.attnQNorm, hd,
                                               2 * hd, seqLen * nHeads);
                kQ35HeadRmsNorm<<<seqLen * nKV, 128, 128 * sizeof(float),
                                  g_stream>>>(s.k, s.k, w.attnKNorm, hd,
                                              seqLen * nKV);
                // MRoPE on Q and K.
                kQ35MRoPE<<<dim3(seqLen, nHeads), 32, 0, g_stream>>>(
                        s.q, seqLen, nHeads, hd, nDims, pos, mropeCos_, mropeSin_);
                kQ35MRoPE<<<dim3(seqLen, nKV), 32, 0, g_stream>>>(
                        s.k, seqLen, nKV, hd, nDims, pos, mropeCos_, mropeSin_);
                // Store K/V into the layer's KV cache.
                float *kvK = kvK_ + static_cast<size_t>(L) * geom_.maxSeqLen * kvLen;
                float *kvV = kvV_ + static_cast<size_t>(L) * geom_.maxSeqLen * kvLen;
                kQ35StoreKV<<<seqLen, 256, 0, g_stream>>>(s.k, s.v, kvK, kvV,
                                                          seqLen, nKV, hd, pos);
                // Flash attention (warp per (token, q-head), HD=256).
                const float invSqrtA = 1.0f / std::sqrt(static_cast<float>(hd));
                dim3 blk(32, 4);
                kQ35WarpAttention<256><<<dim3(seqLen, (nHeads + 3) / 4), blk, 0,
                                         g_stream>>>(s.q, kvK, kvV, s.attnOut,
                                                     seqLen, nHeads, nKV, pos,
                                                     invSqrtA);
                // attnOut *= sigmoid(gate) (gate = second headDim of the fused
                // Q+gate projection).
                kQ35SigmoidGate<<<seqLen, 256, 0, g_stream>>>(
                        s.attnOut, s.q35QGate, seqLen, nHeads, hd);
                if (!gemv(w.attnO, s.attnOut, s.attnProj, H, qLen, "attnO")) {
                    return false;
                }
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.attnProj, seqLen * H);

            // ---- Post-attention RMSNorm + SwiGLU FFN + residual ----
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.postAttnNorm, H, seqLen, 1e-6f);
            if (!gemv(w.ffnGate, s.norm, s.gate, I, H, "ffnGate") ||
                !gemv(w.ffnUp, s.norm, s.up, I, H, "ffnUp")) {
                return false;
            }
            kSiluMul<<<(seqLen * I + 255) / 256, 256, 0, g_stream>>>(s.gate,
                                                                     s.up,
                                                                     seqLen * I);
            if (!gemv(w.ffnDown, s.gate, s.ffnOut, H, I, "ffnDown")) {
                return false;
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.ffnOut, seqLen * H);
        }

        cudaError_t llErr = cudaStreamSynchronize(g_stream);
        if (llErr != cudaSuccess) {
            errMsg = std::string("qwen35 prefix sync: ") +
                     cudaGetErrorString(llErr);
            return false;
        }
        kvPos_ += seqLen;

        if (!fullOffload) {
            // Partial offload: the caller copies the hidden state (copyHiddenOut)
            // and continues the remaining layers on the CPU against the host KV
            // cache.  kvPos_ has already advanced on the GPU side (the host KV
            // cache position tracks it via kvCache_.pos in Model::forward).
            return true;
        }

        // ---- Full offload: final RMSNorm + LM head ----
        if (finalNorm_) {
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.hidden, finalNorm_, H, seqLen, 1e-6f);
        }
        const void *lmQ = lmHeadQ_ != nullptr ? lmHeadQ_ : embedQ_;
        uint32_t lmType = lmHeadQ_ != nullptr ? lmHeadType_ : embedType_;
        uint32_t lmRowBytes =
                lmHeadQ_ != nullptr ? lmHeadRowBytes_ : embedRowBytes_;
        uint32_t lmBlocks = hostBlocksPerRow(lmType, H);
        if (computeAllLogits) {
            for (uint32_t si = 0; si < seqLen; ++si) {
                launchQGemv(lmType, lmQ, s.hidden + si * H,
                            s.logits + si * geom_.vocabSize, geom_.vocabSize, H,
                            lmRowBytes, lmBlocks, q8k_, q8kBytes_);
            }
            cudaMemcpyAsync(logitsOut, s.logits,
                            seqLen * geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        } else {
            uint32_t si = seqLen - 1;
            launchQGemv(lmType, lmQ, s.hidden + si * H, s.logits,
                        geom_.vocabSize, H, lmRowBytes, lmBlocks, q8k_, q8kBytes_);
            cudaMemcpyAsync(logitsOut, s.logits, geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        }
        cudaError_t fin = cudaStreamSynchronize(g_stream);
        if (fin != cudaSuccess) {
            errMsg = std::string("qwen35 lmhead sync: ") +
                     cudaGetErrorString(fin);
            return false;
        }
        return true;
    }

    bool GPUModel::forwardQwen35MoePrefix(const std::vector<int32_t> &tokens,
                                          bool computeAllLogits, float *logitsOut,
                                          std::string &errMsg) {
        if (!allocated_) {
            errMsg = "GPU model not uploaded";
            return false;
        }
        if (tokens.empty()) return true;
        if (geom_.architecture != 2 || geom_.expertCount == 0 ||
            geom_.expertUsedCount == 0) {
            errMsg = "forwardQwen35MoePrefix called for non-qwen35moe geometry";
            return false;
        }
        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t nGpu = geom_.numGpuLayers;
        if (nGpu == 0) nGpu = geom_.numLayers;
        // qwen35moe is FULL-OFFLOAD ONLY (the CPU has no partial-offload
        // continuation for it; TINYCODER_NGL is ignored in the adapter).
        const bool fullOffload = (nGpu == geom_.numLayers);
        if (!fullOffload) {
            errMsg = "qwen35moe requires full offload (no partial)";
            return false;
        }
        const uint32_t H = geom_.hiddenSize;
        const uint32_t I = geom_.intermediateSize;
        const uint32_t nHeads = geom_.numAttentionHeads;
        const uint32_t nKV = geom_.numKVHeads;
        const uint32_t hd = geom_.headDim;
        const uint32_t qLen = nHeads * hd, kvLen = nKV * hd;
        const uint32_t pos = static_cast<uint32_t>(kvPos_);

        // Qwen35MoE dense-hybrid geometry (shared with qwen35).
        const uint32_t dInner = geom_.ssmInnerSize;           // 4096
        const uint32_t headK = geom_.ssmStateSize;            // 128
        const uint32_t nKHeads = geom_.ssmGroupCount;         // 16
        const uint32_t nVHeads = geom_.ssmTimeStepRank;       // 32
        const uint32_t headV = dInner / std::max(nVHeads, 1u);// 128
        const uint32_t keyDim = headK * nKHeads;              // 2048
        const uint32_t valueDim = headV * nVHeads;            // 4096
        const uint32_t convK = geom_.ssmConvKernel;           // 4
        const uint32_t convChannels = 2 * keyDim + valueDim;  // 8192
        const uint32_t qkvDim = convChannels;
        const uint32_t qgDims = nHeads * 2 * hd;

        // MoE geometry.
        const uint32_t expertCount = geom_.expertCount;   // 256
        const uint32_t expertUsed = geom_.expertUsedCount;// 8
        const uint32_t expertFF = geom_.expertFF;         // 512
        const uint32_t sharedFF = geom_.expertSharedFF;   // 512

        if (!ensureScratch(seqLen, errMsg)) return false;
        DeviceScratch &s = scratch_;
        cudaMemcpyAsync(s.tokens, tokens.data(), seqLen * sizeof(int32_t),
                        cudaMemcpyHostToDevice, g_stream);
        cudaStreamSynchronize(g_stream);

        // ---- MoE stats instrumentation.  Cheap steady_clock hooks only (no
        // per-layer stream syncs, no prints in the hot loop): enabled by
        // TINYCODER_MOE_STATS=1 OR the legacy TINYCODER_GPU_VERBOSE=1 (which
        // ALSO adds slower per-stage debug elsewhere -- never use it for
        // timing).  The [moe fwd] line is the ONLY print (once per forward).
        const bool moeTr = g_verbose ||
                           (std::getenv("TINYCODER_MOE_STATS") != nullptr);
        auto tF0 = std::chrono::steady_clock::now();
        if (moeTr) {
            if (!moeEvStart_) cudaEventCreate(&moeEvStart_);
            if (!moeEvEnd_) cudaEventCreate(&moeEvEnd_);
            cudaEventRecord(moeEvStart_, g_stream);
        }
        uint64_t cacheHits = 0, cacheMisses = 0, cacheMissBytes = 0;
        double cpuRouterMs = 0.0, layerGpuMs = 0.0, moeBlockMs = 0.0;
        double syncWaitMs = 0.0;
        double rankLoopMs = 0.0;// rank GEMV loop: fills + enqueues (host-side)
        double fillMs = 0.0;    // batched-path parallel fill (PHASE 1b/1c)
        bool moeFallbackWarned = false;

        // ---- Device-side phase profile (TINYCODER_PHASE_PROFILE=1) ----
        //
        // The [moe fwd] host timers (syncWaitMs / rankLoopMs / fillMs /
        // cpuRouterMs) measure HOST wall between enqueues, so a host-stall
        // (the pinned D2H handoff, the blocking sync) gets attributed to
        // "sync" even though the DEVICE may have already finished -- and the
        // device may still be draining the trunk while the host runs the
        // router.  To separate DEVICE time from HOST time inside the per-layer
        // wall, TINYCODER_PHASE_PROFILE=1 records stream-ordered CUDA events
        // at three boundaries per layer (layer start, post-trunk+shexp+D2H
        // enqueue, post-rank+addScaled enqueue), then accumulates the DEVICE
        // deltas:
        //   trunkDev  = ev1 - ev0  (true SM/engine time the trunk+D2H+shexp
        //                            consumed on the stream -- INCLUDES any
        //                            queue-wait on the previous layer's tail)
        //   gapDev    = ev2 - ev1  (device-idle stretch while the host ran the
        //                            CPU router + fills + enqueues -- if this
        //                            dominates, the WALL is host-side and the
        //                            fix is CPU/GPU overlap, not more kernels)
        // Events are local (created lazily, destroyed before return); nothing
        // is recorded unless the env flag is set.
        static thread_local const bool phaseProf = [] {
            const char *e = std::getenv("TINYCODER_PHASE_PROFILE");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        struct PhaseAcc {
            double trunkMs = 0.0;     // ev1-ev0 per layer (trunk+D2H+shexp dev)
            double gapMs = 0.0;       // ev2-ev1 per layer (router+fill host gap)
            double tailMs = 0.0;      // next-ev0 - ev2 (residual + addScaled dev)
            double lmHeadMs = 0.0;    // final norm + LM-head GEMV + D2H (dev)
            double hostRouterMs = 0.0;// cpuRouterMs (reported for the diff)
        } ph;
        // Per-layer event ring [L][slot]: 3 stream-ordered events per layer.
        // Recorded with NO host syncs; all deltas are computed ONCE after the
        // final cudaStreamSynchronize(g_stream) (the true completion point),
        // so this probe adds zero extra syncs to the critical path.
        std::vector<std::array<cudaEvent_t, 3>> phaseEv;
        if (phaseProf) {
            // Sizing by nGpu (the loop caps at nGpu && nRealLayers, so the
            // worst case is nGpu entries; MTP layers are never executed).
            phaseEv.resize(nGpu);
            for (auto &row: phaseEv) {
                for (auto &ev: row) {
                    cudaEventCreate(&ev);
                }
            }
        }
        auto phaseMark = [&](uint32_t layerIdx, uint32_t slot) {
            if (!phaseProf) return;
            cudaEventRecord(phaseEv[layerIdx][slot], g_stream);
        };
        if (moeTr) {
            std::fprintf(stderr,
                         "[moe tr] entry seqLen=%u expertCacheBuilt_=%d "
                         "moeCpuFn_=%d nGpu=%u\n",
                         seqLen, expertCacheBuilt_ ? 1 : 0, moeCpuFn_ ? 1 : 0,
                         nGpu);
        }
        // Per-layer wall-clock hooks: layerGpuMs accumulates the enqueue-to-D2H
        // interval (GPU work + launch overhead); moeBlockMs accumulates the
        // hybrid-handoff interval (D2H sync .. residual) minus the CPU router.
        auto tLayer0 = std::chrono::steady_clock::now();
        auto startLayer = [&]() {
            tLayer0 = std::chrono::steady_clock::now();
        };
        auto endLayerGpu = [&]() {
            layerGpuMs += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - tLayer0)
                                  .count();
        };
        auto tMoe0 = std::chrono::steady_clock::now();
        auto startMoeBlock = [&]() {
            tMoe0 = std::chrono::steady_clock::now();
        };
        auto endMoeBlock = [&]() {
            moeBlockMs += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - tMoe0)
                                  .count();
        };
        const uint64_t gateSliceB =
                static_cast<uint64_t>(expertFF_) * expertRowBytesGate_;
        const uint64_t downSliceB =
                static_cast<uint64_t>(expertHM_) * expertRowBytesDown_;

        // ---- Embedding (K-quant / legacy) — same dispatch as qwen35. ----
        {
            const uint32_t blocksPerRow = hostBlocksPerRow(embedType_, H);
            const uint8_t *eQ = reinterpret_cast<const uint8_t *>(embedQ_);
            if (embedType_ == kTypeQ5K || embedType_ == kTypeQ5_0) {
                kEmbedDequantQ5K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ4K) {
                kEmbedDequantQ4K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ8_0) {
                kEmbedDequantSmall32<kTypeQ8_0><<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else {
                kEmbedDequant<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            }
        }
        // TINYCODER_DUMP_NORM=1: dump the GPU's embedding output (s.hidden)
        // before any layer, so a GPU-vs-CPU divergence can be localized to the
        // embedding vs the L0 recurrent/attention block.
        {
            const char *dn4 = std::getenv("TINYCODER_DUMP_NORM");
            if (dn4 != nullptr && dn4[0] != '\0' && dn4[0] != '0' && seqLen >= 1) {
                cudaStreamSynchronize(g_stream);
                float e0[8];
                cudaMemcpy(e0, s.hidden, 8 * sizeof(float), cudaMemcpyDeviceToHost);
                double esum = 0.0;
                for (int i = 0; i < 2048; ++i) {
                    float t;
                    cudaMemcpy(&t, s.hidden + i, sizeof(float),
                               cudaMemcpyDeviceToHost);
                    esum += static_cast<double>(t) * t;
                }
                std::fprintf(stderr,
                             "[gpu emb] hidden[0..7]={%.6f %.6f %.6f %.6f %.6f "
                             "%.6f %.6f %.6f} ||h||=%.3f\n",
                             e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                             std::sqrt(esum));
            }
        }

        // Dense projection helper.  qwen35moe runs the SAME math for prefill
        // (seqLen>1) and decode (seqLen==1): a per-token loop of the exact
        // quantized GEMV (launchQGemv), NEVER an fp16 cuBLAS GEMM.
        //
        // Exact per-token GEMVs are what make batch == sequential (and
        // GPU == CPU) here.  The hybrid CPU router (moeCpuFn_) consumes a D2H
        // mirror of the post-attn norm, and the prefill changed it: the old
        // fp16 cuBLAS GEMM rounded every projection differently from the
        // fp32 decode GEMV (visible as ~1e-3 router-logit deltas at layer 0),
        // and the MoE top-8 set is knife-edge (probabilities clustered within
        // ~1e-3), so a borderline expert flipped by layer 2 and diverged
        // chaotically over the 40 layers (maxDiff ~19, different top-1).  With
        // identical activations the CPU router selects bit-identical experts
        // and batch collapses to the (already verified) decode path.
        //
        // launchQGemv covers every type these projections use (K-quants via
        // kQGemv*, Q8_0 via kQGemvSmall32, F32 via kQ35F32Gemv), so no fp16
        // conversion path is needed at all.
        auto gemv = [&](const DeviceMatrix &m, const float *x, float *y,
                        uint32_t nRows, uint32_t nCols,
                        const char *what) -> bool {
            if (m.q == nullptr || nRows == 0) return true;
            for (uint32_t st = 0; st < seqLen; ++st) {
                launchQGemv(m.type, m.q, x + static_cast<size_t>(st) * nCols,
                            y + static_cast<size_t>(st) * nRows, nRows, nCols,
                            m.rowBytes, m.blocksPerRow, q8k_, q8kBytes_);
            }
            return true;
        };

        const float invSqrtV = 1.0f / std::sqrt(static_cast<float>(headV));
        const uint32_t nDims = geom_.ropeDimensionCount > 0
                                       ? geom_.ropeDimensionCount
                                       : 64;

        // ---- Offloaded layer prefix (all layers; qwen35moe is full offload).
        // The MTP (NextN) block is never executed on the main decode path.
        const uint32_t nRealLayers =
                geom_.numLayers - geom_.nextnPredictLayers;
        for (uint32_t L = 0; L < nGpu && L < nRealLayers; ++L) {
            const DeviceLayer &w = layers_[L];
            if (moeTr) startLayer();
            phaseMark(L, 0);// ev0: layer start (trunk boundary)

            // ---- Attention / recurrent block (identical to qwen35) ----
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.rmsNormAttn, H, seqLen, 1e-6f);

            // TINYCODER_DUMP_NORM=1: dump the GPU's L0 attention norm (the
            // recurrent block's input, BEFORE any conv/GDN math) so it can be
            // compared against the CPU's `[cpuattnnorm]` -- both are
            // rmsNormAttn(embedding), so any difference isolates
            // kRMSNormRow / the rmsNormAttn upload (embedding is already
            // verified byte-identical).
            {
                const char *dnA = std::getenv("TINYCODER_DUMP_NORM");
                if (L == 0 && dnA != nullptr && dnA[0] != '\0' &&
                    dnA[0] != '0') {
                    cudaStreamSynchronize(g_stream);
                    float a0[8];
                    cudaMemcpy(a0, s.norm, 8 * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    std::fprintf(stderr,
                                 "[gpuattnnorm] L0 attnNorm[0..4]={%.6f %.6f "
                                 "%.6f %.6f %.6f} (8th=%.6f)\n",
                                 a0[0], a0[1], a0[2], a0[3], a0[4], a0[7]);
                }
            }

            if ((L + 1) % geom_.fullAttentionInterval != 0) {
                // =================== Recurrent (gated delta net) ===========
                if (!gemv(w.attnQKV, s.norm, s.qkv, qkvDim, H, "attnQKV") ||
                    !gemv(w.attnGate, s.norm, s.q35z, dInner, H, "attnGate") ||
                    !gemv(w.ssmBetaQ, s.norm, s.q35Beta, nVHeads, H,
                          "ssmBetaQ") ||
                    !gemv(w.ssmAlphaQ, s.norm, s.q35GateV, nVHeads, H,
                          "ssmAlphaQ")) {
                    return false;
                }
                // beta=sigmoid(beta); gateV=softplus(alpha+dt.bias)*ssm_a
                kQ35BetaGate<<<seqLen, 64, 0, g_stream>>>(
                        s.q35Beta, s.q35GateV, w.ssmDtBiasFull, w.ssmABroadcast,
                        s.q35Beta, s.q35GateV, seqLen, nVHeads);

                // Sequential recurrence over tokens (per-layer persistent state).
                float *q35ConvState =
                        q35ConvState_ + static_cast<size_t>(L) * (convK - 1) * convChannels;
                float *q35GdnState =
                        q35GdnState_ + static_cast<size_t>(L) * nVHeads * headV * headV;
                const float *convW = static_cast<const float *>(w.ssmConv1d.q);
                const uint32_t convBlocks = (convChannels + 255) / 256;
                for (uint32_t st = 0; st < seqLen; ++st) {
                    const float *qkv = s.qkv + static_cast<size_t>(st) * qkvDim;
                    kQ35ConvStep<<<convBlocks, 256, 0, g_stream>>>(
                            convW, qkv, s.convIn, s.convOut, q35ConvState,
                            convChannels, convK);
                    // TINYCODER_DUMP_NORM=1: dump GPU L0 token-0 recurrent
                    // intermediates (qkv, convOut post-silu) to bisect against
                    // the CPU's [cpuqkv]/[cpuconv] (zero recurrent state at
                    // st==0, so convOut = silu(conv(qkv)) is a pure function).
                    const char *dnR = std::getenv("TINYCODER_DUMP_NORM");
                    if (L == 0 && st == 0 && dnR != nullptr && dnR[0] != '\0' &&
                        dnR[0] != '0') {
                        cudaStreamSynchronize(g_stream);
                        float q0[5], c0[5];
                        cudaMemcpy(q0, s.qkv, 5 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        cudaMemcpy(c0, s.convOut, 5 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        std::fprintf(stderr,
                                     "[gpuqkv] L0T0 qkv[0..4]={%.6f %.6f %.6f "
                                     "%.6f %.6f}\n",
                                     q0[0], q0[1], q0[2], q0[3], q0[4]);
                        std::fprintf(stderr,
                                     "[gpuconv] L0T0 convOut[0..4]={%.6f %.6f "
                                     "%.6f %.6f %.6f}\n",
                                     c0[0], c0[1], c0[2], c0[3], c0[4]);
                    }
                    kQ35L2NormRepeat<<<1, 128, 0, g_stream>>>(
                            s.convOut, s.qN, s.kN, s.vN, headK, headV, nKHeads,
                            nVHeads, keyDim, valueDim, 1);
                    const float *betaRow = s.q35Beta + static_cast<size_t>(st) * nVHeads;
                    const float *gateRow = s.q35GateV + static_cast<size_t>(st) * nVHeads;
                    float *gdnRow = s.gdnOut + static_cast<size_t>(st) * valueDim;
                    // TINYCODER_GDN_SEQ=1: restore the original sequential
                    // per-head launch loop (48 grid(1) launches) for parity
                    // bisection against the batched grid.x=nVHeads launch.
                    const bool gdnSeq = (std::getenv("TINYCODER_GDN_SEQ") != nullptr);
                    if (gdnSeq) {
                        for (uint32_t hv = 0; hv < nVHeads; ++hv) {
                            kQ35GdnStep<128><<<1, 128, 0, g_stream>>>(
                                    s.qN + static_cast<size_t>(hv) * headK,
                                    s.kN + static_cast<size_t>(hv) * headK,
                                    s.vN + static_cast<size_t>(hv) * headV,
                                    betaRow, gateRow,
                                    q35GdnState + static_cast<size_t>(hv) * headV * headV,
                                    gdnRow + static_cast<size_t>(hv) * headV, nVHeads,
                                    hv, invSqrtV);
                        }
                    } else {
                        kQ35GdnStepBatch<128><<<nVHeads, 128, 0, g_stream>>>(
                                s.qN, s.kN, s.vN, betaRow, gateRow, q35GdnState,
                                gdnRow, nVHeads, invSqrtV);
                    }
                }
                // Gated RMSNorm (per value-head) × silu(z) over the batch.
                kQ35GatedRmsNorm<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                        s.q35z, s.gdnOut, w.ssmNorm, headV, nVHeads, seqLen);
                if (!gemv(w.ssmOut, s.gdnOut, s.attnProj, H, valueDim,
                          "ssmOut")) {
                    return false;
                }
                // TINYCODER_DUMP_NORM=1: dump GPU L0 token-0 gdnOut (post
                // gated-RMSNorm × silu(z)) and attnProj (post ssmOut) to bisect
                // against the CPU reference.  gdnOut[0..4] is token s's
                // [headV*nVHeads] slice; attnProj[0..4] is the ssmOut output.
                const char *dnO = std::getenv("TINYCODER_DUMP_NORM");
                if (L == 0 && dnO != nullptr && dnO[0] != '\0' &&
                    dnO[0] != '0') {
                    cudaStreamSynchronize(g_stream);
                    float g0[5], p0[5];
                    cudaMemcpy(g0, s.gdnOut, 5 * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(p0, s.attnProj, 5 * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    std::fprintf(stderr,
                                 "[gpuout] L0 gdnOut[0..4]={%.6f %.6f %.6f "
                                 "%.6f %.6f} attnProj[0..4]={%.6f %.6f %.6f "
                                 "%.6f %.6f}\n",
                                 g0[0], g0[1], g0[2], g0[3], g0[4], p0[0], p0[1],
                                 p0[2], p0[3], p0[4]);
                }
            } else {
                // =================== Full attention =======================
                if (!gemv(w.attnQ, s.norm, s.q35QGate, qgDims, H, "attnQ") ||
                    !gemv(w.attnK, s.norm, s.k, kvLen, H, "attnK") ||
                    !gemv(w.attnV, s.norm, s.v, kvLen, H, "attnV")) {
                    return false;
                }
                // Per-head Q RMSNorm (Q slice) + K RMSNorm.
                kQ35HeadRmsNorm2<<<seqLen * nHeads, 128, 128 * sizeof(float),
                                   g_stream>>>(s.q35QGate, s.q, w.attnQNorm, hd,
                                               2 * hd, seqLen * nHeads);
                kQ35HeadRmsNorm<<<seqLen * nKV, 128, 128 * sizeof(float),
                                  g_stream>>>(s.k, s.k, w.attnKNorm, hd,
                                              seqLen * nKV);
                // MRoPE on Q and K.
                kQ35MRoPE<<<dim3(seqLen, nHeads), 32, 0, g_stream>>>(
                        s.q, seqLen, nHeads, hd, nDims, pos, mropeCos_, mropeSin_);
                kQ35MRoPE<<<dim3(seqLen, nKV), 32, 0, g_stream>>>(
                        s.k, seqLen, nKV, hd, nDims, pos, mropeCos_, mropeSin_);
                // Store K/V into the layer's KV cache.
                float *kvK = kvK_ + static_cast<size_t>(L) * geom_.maxSeqLen * kvLen;
                float *kvV = kvV_ + static_cast<size_t>(L) * geom_.maxSeqLen * kvLen;
                kQ35StoreKV<<<seqLen, 256, 0, g_stream>>>(s.k, s.v, kvK, kvV,
                                                          seqLen, nKV, hd, pos);
                // Flash attention (warp per (token, q-head), HD=256).
                const float invSqrtA = 1.0f / std::sqrt(static_cast<float>(hd));
                dim3 blk(32, 4);
                kQ35WarpAttention<256><<<dim3(seqLen, (nHeads + 3) / 4), blk, 0,
                                         g_stream>>>(s.q, kvK, kvV, s.attnOut,
                                                     seqLen, nHeads, nKV, pos,
                                                     invSqrtA);
                // attnOut *= sigmoid(gate).
                kQ35SigmoidGate<<<seqLen, 256, 0, g_stream>>>(
                        s.attnOut, s.q35QGate, seqLen, nHeads, hd);
                if (!gemv(w.attnO, s.attnOut, s.attnProj, H, qLen, "attnO")) {
                    return false;
                }
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.attnProj, seqLen * H);

            // ---- Post-attention RMSNorm + MoE FFN + residual ----
            // TINYCODER_DUMP_NORM=1: dump the GPU's pre-post-norm hidden and
            // the resulting post-attn norm at L0 (seqLen==1) so a GPU-vs-CPU
            // layer-0 divergence can be localized (hidden vs norm scale).
            {
                const char *dn3 = std::getenv("TINYCODER_DUMP_NORM");
                if (dn3 != nullptr && dn3[0] != '\0' && dn3[0] != '0' &&
                    L == 0) {
                    cudaStreamSynchronize(g_stream);
                    float h0[8], hm2[8];
                    cudaMemcpy(h0, s.hidden, 8 * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    double hsum = 0.0;
                    for (int i = 0; i < 2048; ++i) {
                        float t;
                        cudaMemcpy(&t, s.hidden + i, sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        hsum += static_cast<double>(t) * t;
                    }
                    std::fprintf(stderr,
                                 "[gpu l0] pre-post-norm hidden[0..7]={%.6f "
                                 "%.6f %.6f %.6f %.6f %.6f %.6f %.6f} "
                                 "||h||=%.3f\n",
                                 h0[0], h0[1], h0[2], h0[3], h0[4], h0[5], h0[6],
                                 h0[7], std::sqrt(hsum));
                }
            }
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.postAttnNorm, H, seqLen, 1e-6f);

            // ---------- MoE FFN: CPU-expert hybrid OR full-GPU ----------
            if (moeCpuFn_) {
                // ---- CPU-expert hybrid (llama.cpp `--cpu-moe` style) ----
                // The GPU runs the SHARED EXPERT (uploaded with the layer); the
                // CPU runs the ROUTER (fp32 ffnGateInpMoe @ norm — reference
                // math).  When the per-layer expert cache was built
                // (buildExpertCache success), the 8-of-256 routed expert FFNs
                // ALSO run on the GPU from the cached slices (VRAM bandwidth):
                // the CPU callback only fills the router-selection tables, so
                // NO routed-output H2D crosses the PCIe bus and the ~32 MB/
                // layer of host-RAM expert weights (the decode bottleneck on
                // models bigger than the page cache) is never streamed.  The
                // host does ~µs of router work per layer (a 256x2048 fp32
                // mat-vec per token).  Without the cache (device-allocation
                // failure fallback), the CPU runs the full routed expert FFNs
                // into moeFfnOutHost_ (the old --cpu-moe path).
                //
                // SYNCHRONOUS handoff (no event ring).  Events proved racy (see
                // the historical note below); every layer hands the norm to the
                // host with a blocking in-order D2H and takes the routed output
                // back in-order on g_stream — no events, no cross-stream
                // ordering, no reuse hazard.  The CPU router is deterministic
                // on the exact device norm, so batch and single-token prefills
                // that produce bit-identical norms produce bit-identical MoE
                // outputs BY CONSTRUCTION, and the shared-expert kernels (step
                // 2) still run on-device while the CPU callback computes.
                const size_t normBytes =
                        static_cast<size_t>(seqLen) * H * sizeof(float);
                // Ensure the double-buffer mirrors exist (grow on demand).
                if (moeNormHost_.size() < 2) {
                    moeNormHost_.resize(2);
                    moeFfnOutHost_.resize(2);
                }
                if (moeNormHost_[0].size() < static_cast<size_t>(seqLen) * H) {
                    // Unpin the OLD ranges FIRST: the resize below may
                    // reallocate storage, which would leave the registration
                    // pointing at freed/repurposed memory (registering or
                    // freeing it later is UB).  moeNormPinned_[i] tracks each
                    // slot's current pinned range (nullptr = not pinned).
                    for (uint32_t pi = 0; pi < 2; ++pi) {
                        if (moeNormPinned_[pi] != nullptr) {
                            cudaHostUnregister(moeNormPinned_[pi]);
                            moeNormPinned_[pi] = nullptr;
                        }
                    }
                    for (auto &slot: moeNormHost_)
                        slot.resize(static_cast<size_t>(seqLen) * H);
                    for (auto &slot: moeFfnOutHost_)
                        slot.resize(static_cast<size_t>(seqLen) * H);
                    // PIN the D2H norm mirrors ONCE here (2026-09-14 root-cause
                    // fix): a pageable cudaMemcpyAsync D2H forces the driver to
                    // bounce 8 KB transfers through ITS internal pinned staging
                    // with a global lock, so the host-side enqueue BLOCKS until
                    // the copy completes — and the copy waits for the whole
                    // unflushed trunk burst, stranding the GPU at idle clocks.
                    // Per-layer profiling saw wall ~39 ms/layer with only ~4 ms
                    // of sync/router/rank/fill accounted — the unaccounted
                    // ~35 ms is exactly this hidden pageable-D2H serialization
                    // (invisible to the CUDA-event timeline: dev == total
                    // because the stream is EMPTY during the stall).
                    // cudaHostRegister pins the buffer, turning each per-layer
                    // transfer into a true async DMA.
                    // Opt-out for measurement: TINYCODER_NO_PIN_NORM=1.
                    const char *noPin = std::getenv("TINYCODER_NO_PIN_NORM");
                    const bool pinNorm =
                            (noPin == nullptr || noPin[0] == '\0' ||
                             noPin[0] == '0');
                    if (pinNorm) {
                        for (uint32_t pi = 0; pi < 2; ++pi) {
                            cudaError_t eP = cudaHostRegister(
                                    moeNormHost_[pi].data(),
                                    static_cast<size_t>(seqLen) * H *
                                            sizeof(float),
                                    cudaHostRegisterMapped |
                                            cudaHostRegisterPortable);
                            if (eP == cudaSuccess) {
                                moeNormPinned_[pi] =
                                        moeNormHost_[pi].data();
                            } else {
                                // Non-fatal: fall back to the pageable D2H (old
                                // behavior) rather than abort decode.
                                std::fprintf(stderr,
                                             "[gpu] cudaHostRegister(moeNormHost) "
                                             "failed: %s\n",
                                             cudaGetErrorString(eP));
                                moeNormPinned_[pi] = nullptr;
                            }
                        }
                    }
                }
                const uint32_t buf = moeBufFlip_ & 1u;
                if (moeTr) endLayerGpu();
                if (moeTr) startMoeBlock();
                // 1) D2H the post-attn norm on g_stream: the CPU router's exact
                //    input.  In-order AFTER this layer's RMSNorm (s.norm is
                //    still this layer's; L+1's RMSNorm is not enqueued until
                //    the next iteration), race-free by the sync in step 3.
                cudaMemcpyAsync(moeNormHost_[buf].data(), s.norm, normBytes,
                                cudaMemcpyDeviceToHost, g_stream);
                phaseMark(L, 1);// ev1: after trunk+D2H+shexp enqueues
                // 2) Shared-expert compute on g_stream (runs CONCURRENTLY with
                //    the CPU callback below — this keeps the GPU busy): gate
                //    sigmoid + SwiGLU + down into s.moeShexpDown.  It only
                //    reads s.norm (g_stream-ordered after the RMSNorm) and
                //    nothing here depends on the routed experts.
                if (w.ffnGateShexp.q != nullptr && w.ffnUpShexp.q != nullptr &&
                    w.ffnDownShexp.q != nullptr && sharedFF > 0) {
                    kQ35MoeShexpGate<<<seqLen, 256, 0, g_stream>>>(
                            s.norm, static_cast<const float *>(w.ffnGateInpShexp.q),
                            s.moeShexpGate, seqLen, H, H);
                    if (!gemv(w.ffnGateShexp, s.norm, s.moeShexpGateUp, sharedFF, H,
                              "ffnGateShexp") ||
                        !gemv(w.ffnUpShexp, s.norm, s.moeShexpGateUp2, sharedFF, H,
                              "ffnUpShexp")) {
                        return false;
                    }
                    kSiluMul<<<(seqLen * sharedFF + 255) / 256, 256, 0, g_stream>>>(
                            s.moeShexpGateUp, s.moeShexpGateUp2, seqLen * sharedFF);
                    if (!gemv(w.ffnDownShexp, s.moeShexpGateUp, s.moeShexpDown, H,
                              sharedFF, "ffnDownShexp")) {
                        return false;
                    }
                }
                // 3) Blocking sync: this layer's norm D2H is complete and the
                //    step-2 shared-expert kernels (reading s.norm) have
                //    finished.  The CPU callback now sees the exact norm.
                if (moeTr) {
                    auto tSync0 = std::chrono::steady_clock::now();
                    cudaStreamSynchronize(g_stream);
                    syncWaitMs += std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() -
                                          tSync0)
                                          .count();
                } else {
                    cudaStreamSynchronize(g_stream);
                }
                // TINYCODER_DUMP_NORM=1: verify the D2H mirror the CPU router
                // is about to read equals a FRESH device read of s.norm at the
                // SAME moment (post-sync).  A mismatch proves the mirror is
                // stale/raced (D2H ordering) vs an on-device s.norm clobber by
                // the MoE block (mirror matches but the post-block readback
                // differs from the pre-block value).
                {
                    const char *dnP = std::getenv("TINYCODER_DUMP_NORM");
                    if (dnP != nullptr && dnP[0] != '\0' && dnP[0] != '0' &&
                        seqLen >= 1) {
                        float dv[2];
                        cudaMemcpy(dv, s.norm, 2 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        const float *mv = moeNormHost_[buf].data();
                        std::fprintf(stderr,
                                     "[normprobe] L%u mirror[0..1]={%.6f %.6f} "
                                     "dev[0..1]={%.6f %.6f} %s\n",
                                     L, mv[0], mv[1], dv[0], dv[1],
                                     (mv[0] == dv[0] && mv[1] == dv[1])
                                             ? "MATCH"
                                             : "MISMATCH");
                    }
                }
                // 4) CPU callback: ROUTER-ONLY (expert cache built) or FULL
                //    routed experts (fallback), dispatched inside the callback
                //    on expertCacheBuilt().  While it runs, the GPU is idle (no
                //    L+1 work is enqueued yet), so nothing races device state.
                if (moeTr) {
                    auto tRouter0 = std::chrono::steady_clock::now();
                    const bool rOk = moeCpuFn_(L, moeNormHost_[buf].data(),
                                               moeFfnOutHost_[buf].data(),
                                               seqLen);
                    cpuRouterMs += std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() -
                                           tRouter0)
                                           .count();
                    if (!rOk) {
                        errMsg = "CPU MoE expert callback failed";
                        return false;
                    }
                } else if (!moeCpuFn_(L, moeNormHost_[buf].data(),
                                      moeFfnOutHost_[buf].data(), seqLen)) {
                    errMsg = "CPU MoE expert callback failed";
                    return false;
                }
                // TINYCODER_MOE_FALLBACK=1: force the old CPU-expert fallback
                // (adapter computes the FULL routed experts with
                // computeQwen35MoEFromLogits into moeFfnOutHost_, GPU only adds
                // the shared expert + residual) even when the device cache
                // exists.  Used to bisect the GPU-vs-CPU logits divergence:
                // if the fallback path AGREES with pure CPU but the
                // cache-driven GPU-expert path does NOT, the bug is in the
                // cache slices / expert GEMVs (not the trunk + router).
                const bool forceMoeFallback =
                        (std::getenv("TINYCODER_MOE_FALLBACK") != nullptr);
                if (expertCacheBuilt_ && !forceMoeFallback) {
                    auto cacheStat = [&](uint32_t ee) {
                        const uint32_t cset = expertCacheSet(ee);
                        const bool hit =
                                (expertCacheTag_[L][cset][0] == static_cast<int32_t>(ee)) ||
                                (expertCacheTag_[L][cset][1] == static_cast<int32_t>(ee));
                        if (hit) {
                            ++cacheHits;
                        } else {
                            ++cacheMisses;
                            cacheMissBytes += 2 * gateSliceB + downSliceB;
                        }
                    };
                    // ---- Router-only: run the 8 routed experts on the GPU
                    // from the per-layer cache.  The callback wrote the
                    // per-(token, rank) (expertId, weight) tables into
                    // moeRouterIdxHost_/moeRouterWgtHost_.  Zero the
                    // accumulator and H2D the weights (the accumulation kernel
                    // reads them on-device) — both in-order on g_stream.
                    auto tRank0 = std::chrono::steady_clock::now();
                    cudaMemsetAsync(s.moeExpertOut, 0,
                                    static_cast<uint64_t>(seqLen) * H *
                                            sizeof(float),
                                    g_stream);
                    cudaMemcpyAsync(s.moeWgt, moeRouterWgtHost_.data(),
                                    static_cast<size_t>(seqLen) * expertUsed *
                                            sizeof(float),
                                    cudaMemcpyHostToDevice, g_stream);
                    // Cache slice geometry: the cached experts share the
                    // layer's recorded quant type (Q8_0 for Ornith); rebuild
                    // the block count for the sliced row widths (cols == H for
                    // gate/up, cols == expertFF for down).
                    const uint32_t gBlocks =
                            (expertType_ == kTypeQ5_0 || expertType_ == kTypeQ8_0)
                                    ? (H + 31) / 32
                                    : (H + 255) / 256;
                    const uint32_t dBlocks =
                            (expertType_ == kTypeQ5_0 || expertType_ == kTypeQ8_0)
                                    ? (expertFF + 31) / 32
                                    : (expertFF + 255) / 256;
                    // Decode fast path: all 8 ranks' gate/up GEMVs share the
                    // SAME activation (s.norm) and the down GEMVs share the
                    // silu output.  Quantize the norm ONCE per layer into the
                    // DEDICATED q8kN_ scratch (Q8_32 layout) instead of
                    // per-(rank, matrix) re-quantize kernels (added
                    // 2026-09-13; measured ~1920 wasteful quantize
                    // launches/token before this split).  q8kN_ must NOT be
                    // q8k_: the per-rank down path re-quantizes s.moeGateUp
                    // into q8k_, and sharing one buffer made ranks 1..7 read
                    // rank 0's silu as the norm (correctness regression,
                    // measured 2026-09-13: garbage MoE output + cache misses
                    // 180->236, missMB 529->752, moeBlockMs 236.7->264.1).
                    // TINYCODER_MOE_SLOW=1 forces the generic per-(rank, matrix)
                    // launchQGemv path (identical to prefill) so a parity
                    // failure can be bisected: fast path vs the shared Q8_0
                    // GEMV branch.  Defaults OFF (fast path on).
                    const bool moeFastPath =
                            (std::getenv("TINYCODER_MOE_SLOW") == nullptr);
                    if (seqLen == 1 && expertType_ == kTypeQ8_0 && moeFastPath) {
                        const uint64_t needN =
                                static_cast<uint64_t>(gBlocks) *
                                kQ8_0x32_STRIDE;
                        if (q8kN_ == nullptr || q8kNBytes_ < needN) {
                            if (q8kN_) cudaFree(q8kN_);
                            q8kN_ = nullptr;
                            q8kNBytes_ = 0;
                            cudaError_t eN = cudaMalloc(&q8kN_, needN);
                            if (eN != cudaSuccess) {
                                errMsg = std::string("cudaMalloc(q8kN): ") +
                                         cudaGetErrorString(eN);
                                return false;
                            }
                            q8kNBytes_ = needN;
                        }
                        launchQuantQ8_0x32(s.norm, q8kN_, H);
                        // ============ BATCHED decode fast path (2026-09-14) ====
                        // Replaces the per-rank serial 40-kernel chain with 4
                        // rank-indexed launches (grid.y = rank) so all expertUsed
                        // ranks' GEMVs co-schedule on the SMs.  The host gathers
                        // the per-rank cache bases (PHASE 1 — the miss fills
                        // already pipeline on g_stream2 via expertStageEv_) and
                        // publishes ONE tiny pointer-table H2D, then PHASE 2
                        // enqueues: GU GEMV (fused gate+up) -> silu -> per-rank
                        // down-input quantize -> down GEMV -> serialized weight
                        // accumulate (bit-exact CPU order, one thread per element
                        // looping ranks so there is no cross-rank race).
                        //
                        // SLOT-CONFLICT GUARD (2026-09-14): the batched path
                        // enqueues ALL 8 fills on g_stream2 BEFORE any GEMV, so
                        // a later rank's fill can EVICT an earlier rank's slot
                        // (same 2-way set, LRU way) and overwrite its bytes.  All
                        // batched GEMVs wait for all fills, so the earlier rank
                        // would read the WRONG expert (measured: norms diverge
                        // from L3 onward while L0-L2 matched — the ~23% of
                        // layers whose 8 routed experts collide in a set).  The
                        // interleaved per-rank loop below enqueues each GEMV
                        // right after its own fill (stream-ordered BEFORE any
                        // later eviction), so it is race-free: on a detected
                        // collision (two different experts mapping to the SAME
                        // slot base) we fall back to it — the old function
                        // refills each evicted slot right before its GEMV.
                        const uint32_t eu = expertUsed;
                        bool moeSlotConflict = false;
                        std::vector<const uint8_t *> moeSlotSeen;
                        moeSlotSeen.reserve(eu);
                        const uint64_t downStrideB =
                                static_cast<uint64_t>(dBlocks) * kQ8_0x32_STRIDE;
                        const uint64_t needDownP =
                                static_cast<uint64_t>(eu) * downStrideB;
                        if (moeQ8kDownP_ == nullptr ||
                            moeQ8kDownPBytes_ < needDownP) {
                            if (moeQ8kDownP_) cudaFree(moeQ8kDownP_);
                            moeQ8kDownP_ = nullptr;
                            moeQ8kDownPBytes_ = 0;
                            cudaError_t eD = cudaMalloc(&moeQ8kDownP_, needDownP);
                            if (eD != cudaSuccess) {
                                errMsg = std::string("cudaMalloc(moeQ8kDownP): ") +
                                         cudaGetErrorString(eD);
                                return false;
                            }
                            moeQ8kDownPBytes_ = needDownP;
                        }
                        const uint64_t needWq = static_cast<uint64_t>(eu) * 3 *
                                                sizeof(uint8_t *);
                        if (moeRankWq_ == nullptr || moeRankWqBytes_ < needWq) {
                            if (moeRankWq_) cudaFree(moeRankWq_);
                            moeRankWq_ = nullptr;
                            moeRankWqBytes_ = 0;
                            cudaError_t eW = cudaMalloc(&moeRankWq_, needWq);
                            if (eW != cudaSuccess) {
                                errMsg = std::string("cudaMalloc(moeRankWq): ") +
                                         cudaGetErrorString(eW);
                                return false;
                            }
                            moeRankWqBytes_ = needWq;
                        }
                        // PHASE 1 (LOCATE-ONLY): per-rank cache slot bases via
                        // the tag/LRU tables WITHOUT copying any bytes.  The
                        // (hit) bases + (miss) victim reservations are gathered
                        // here; the memcpy fills are dispatched to the
                        // ThreadPool in PARALLEL (PHASE 1b) so the g_stream2
                        // copy engine never idles behind the host's serial
                        // memcpy (measured 2026-09-15: 0.47 ms/miss single-
                        // threaded = ~61 ms/token of host fill on the critical
                        // path — the largest single item in moeBlockMs).
                        // SLOT-CONFLICT guard (unchanged): two DIFFERENT
                        // experts mapping to the SAME slot base means a later
                        // fill would evict an earlier rank's bytes before the
                        // batched GEMVs ran -> fall back to the interleaved
                        // per-rank loop, which refills each slot right before
                        // its own GEMVs and is therefore race-free.
                        struct MoeMissFill {
                            uint32_t expert;
                            const uint8_t *dest;// victim slot base (device)
                            uint32_t victim;    // way reserved in the set
                        };
                        // gateSliceB / downSliceB / slot geometry come from the
                        // enclosing function scope (declared above the layer
                        // loop), so no shadow copies are needed here.
                        const uint64_t slotBytesB = 2 * gateSliceB + downSliceB;
                        // Grow the pinned stage ring to one slot if needed
                        // (mirrors ensureExpertCached's grow-on-demand).
                        if (expertStageBytes_ < slotBytesB) {
                            for (uint32_t b = 0; b < kExpertStageBufs; ++b) {
                                if (expertStage_[b]) {
                                    cudaHostUnregister(expertStage_[b]);
                                    std::free(expertStage_[b]);
                                }
                                expertStage_[b] = nullptr;
                                if (expertStageEv_[b]) {
                                    cudaEventDestroy(expertStageEv_[b]);
                                    expertStageEv_[b] = nullptr;
                                }
                            }
                            for (uint32_t b = 0; b < kExpertStageBufs; ++b) {
                                void *pp = nullptr;
                                if (posix_memalign(
                                            &pp, 4096,
                                            static_cast<size_t>(slotBytesB)) !=
                                    0) {
                                    errMsg = "posix_memalign(expertStage) failed";
                                    return false;
                                }
                                cudaError_t eP = cudaHostRegister(
                                        pp, static_cast<size_t>(slotBytesB),
                                        cudaHostRegisterMapped |
                                                cudaHostRegisterPortable);
                                if (eP != cudaSuccess) {
                                    std::free(pp);
                                    errMsg = std::string(
                                                     "cudaHostRegister(expertStage): ") +
                                             cudaGetErrorString(eP);
                                    return false;
                                }
                                expertStage_[b] = static_cast<uint8_t *>(pp);
                                cudaEventCreateWithFlags(
                                        &expertStageEv_[b],
                                        cudaEventDisableTiming);
                            }
                            expertStageBytes_ = slotBytesB;
                        }
                        std::vector<const uint8_t *> moeRankWqHost;
                        moeRankWqHost.reserve(3u * eu);
                        std::vector<MoeMissFill> moeMisses;
                        moeMisses.reserve(eu);
                        std::vector<uint32_t> moeMissStage;// stage idx per miss
                        moeMissStage.reserve(eu);
                        for (uint32_t r = 0; r < eu; ++r) {
                            const int32_t e0 = moeRouterIdxHost_.data()[r];
                            cacheStat(static_cast<uint32_t>(e0));
                            if (e0 < 0) {
                                errMsg = "qwen35moe expert cache miss: negative "
                                         "expert id in decode fast path";
                                return false;
                            }
                            const uint32_t ue = static_cast<uint32_t>(e0);
                            const uint32_t cset = expertCacheSet(ue);
                            const uint8_t *g0 =
                                    static_cast<const uint8_t *>(
                                            expertCache_[L].gateQ) +
                                    static_cast<size_t>(cset *
                                                        kExpertCacheWays) *
                                            static_cast<size_t>(slotBytesB);
                            const uint8_t *g1 =
                                    g0 + static_cast<size_t>(slotBytesB);
                            const uint8_t *cacheBase = nullptr;
                            const uint8_t *downQ = nullptr;
                            if (expertCacheTag_[L][cset][0] ==
                                static_cast<int32_t>(ue)) {
                                expertCacheLru_[L][cset] = 1;
                                cacheBase = g0;
                                downQ = g0 + 2 * gateSliceB;
                            } else if (expertCacheTag_[L][cset][1] ==
                                       static_cast<int32_t>(ue)) {
                                expertCacheLru_[L][cset] = 0;
                                cacheBase = g1;
                                downQ = g1 + 2 * gateSliceB;
                            } else {
                                // Miss: reserve the LRU way (bytes filled in
                                // PHASE 1b; tag committed after the H2D).
                                uint32_t victim = expertCacheLru_[L][cset];
                                const uint8_t *gBase =
                                        (victim == 0) ? g0 : g1;
                                if (expertCacheTag_[L][cset][0] < 0 &&
                                    expertCacheTag_[L][cset][1] < 0) {
                                    victim = 0;
                                    gBase = g0;
                                }
                                expertCacheLru_[L][cset] =
                                        (victim == 0) ? 1u : 0u;
                                cacheBase = gBase;
                                downQ = gBase + 2 * gateSliceB;
                                moeMisses.push_back(
                                        {ue, gBase, victim});
                                moeMissStage.push_back(
                                        (expertStageFlag_++) %
                                        kExpertStageBufs);
                            }
                            for (size_t q = 0; q < moeSlotSeen.size(); ++q) {
                                if (moeSlotSeen[q] == cacheBase &&
                                    moeRouterIdxHost_.data()[q] != e0) {
                                    moeSlotConflict = true;
                                    break;
                                }
                            }
                            moeSlotSeen.push_back(cacheBase);
                            moeRankWqHost.push_back(cacheBase);
                            moeRankWqHost.push_back(
                                    cacheBase + static_cast<uint64_t>(expertFF_) *
                                                        expertRowBytesGate_);
                            moeRankWqHost.push_back(downQ);
                            if (moeSlotConflict) break;
                        }
                        // Conflict detected: the batched plan would read
                        // corrupted bytes.  Fall back to the interleaved
                        // per-rank loop (its ensureExpertCached refills each
                        // expert right before its own GEMVs on g_stream).
                        if (moeSlotConflict) goto moePerRank;
                        // PHASE 1b (SERIAL FILL — REVERTED 2026-09-16): the
                        // ThreadPool parallel-fill experiment (pool parallelFor
                        // gathering 3*misses zones into the stage ring, then
                        // back-to-back H2Ds) caused a catastrophic regression —
                        // 0.32-0.40 tok/s vs the 5.61 tok/s baseline, fillMs
                        // 91-2773 ms/token vs ~58, syncWaitMs 3.4 -> 90 ms.
                        // Root causes: (1) with ~130-180 misses/token and only
                        // kExpertStageBufs=8 stage buffers, the round-robin
                        // stage claim WRAPS within one token, so two pool
                        // workers concurrently memcpy DIFFERENT experts into
                        // the same stage while the enqueue loop H2Ds that
                        // stage twice (torn, non-bit-identical bytes — the
                        // "immutable source" safety argument is void); (2) the
                        // parallel gather fully serializes in phase 1b BEFORE
                        // phase 1c's H2Ds, so the copy engine idles during the
                        // gather while 8 workers hammer the same pinned pages
                        // (DMA snoop contention ~0.3 GB/s vs 7-11 GB/s
                        // serial); (3) fine-grained fetch_add + std::function
                        // dispatch per 1.1 MB job is the wrong tool for
                        // near-uniform slab copies.  The serial fill below
                        // preserves the original ring semantics (each stage's
                        // H2D drains before the same stage refills) and
                        // reproduces the 5.61 tok/s baseline.
                        if (!moeMisses.empty()) {
                            auto tFill0 = std::chrono::steady_clock::now();
                            const uint8_t *srcGate =
                                    expertCache_[L].hostGate
                                            ? expertCache_[L].hostGate
                                            : expertHostGate_;
                            const uint8_t *srcUp =
                                    expertCache_[L].hostUp
                                            ? expertCache_[L].hostUp
                                            : expertHostUp_;
                            const uint8_t *srcDown =
                                    expertCache_[L].hostDown
                                            ? expertCache_[L].hostDown
                                            : expertHostDown_;
                            for (size_t mi = 0; mi < moeMisses.size(); ++mi) {
                                const MoeMissFill &mf = moeMisses[mi];
                                const uint32_t st = moeMissStage[mi];
                                uint8_t *stage = expertStage_[st];
                                const uint64_t eoff =
                                        static_cast<uint64_t>(mf.expert) *
                                        gateSliceB;
                                std::memcpy(stage, srcGate + eoff, gateSliceB);
                                std::memcpy(stage + gateSliceB, srcUp + eoff,
                                            gateSliceB);
                                std::memcpy(stage + 2 * gateSliceB,
                                            srcDown +
                                                    static_cast<uint64_t>(
                                                            mf.expert) *
                                                            downSliceB,
                                            downSliceB);
                                cudaMemcpyAsync(
                                        const_cast<uint8_t *>(mf.dest), stage,
                                        static_cast<size_t>(slotBytesB),
                                        cudaMemcpyHostToDevice, g_stream2);
                                cudaEventRecord(expertStageEv_[st], g_stream2);
                                cudaStreamWaitEvent(g_stream,
                                                    expertStageEv_[st], 0);
                                expertCacheTag_[L][expertCacheSet(mf.expert)]
                                               [mf.victim] =
                                                       static_cast<int32_t>(mf.expert);
                            }
                            if (moeTr) {
                                fillMs += std::chrono::duration<double,
                                                                std::milli>(
                                                  std::chrono::steady_clock::now() -
                                                  tFill0)
                                                  .count();
                            }
                        }
                        cudaMemcpyAsync(moeRankWq_, moeRankWqHost.data(), needWq,
                                        cudaMemcpyHostToDevice, g_stream);
                        // PHASE 2: batched launches.
                        const uint8_t *const *wqTable =
                                reinterpret_cast<const uint8_t *const *>(
                                        moeRankWq_);
                        const uint32_t combinedRows = 2u * expertFF;
                        kQGemvQ8_0xQ8K_BatchGU<<<
                                dim3((combinedRows + 7) / 8, eu), dim3(32, 8), 0,
                                g_stream>>>(wqTable, q8kN_, s.moeGateUpP,
                                            s.moeGateUp2P, expertFF, gBlocks,
                                            expertRowBytesGate_, combinedRows);
                        kSiluMulBatch<<<dim3((expertFF + 255) / 256, eu), 256, 0,
                                        g_stream>>>(s.moeGateUpP, s.moeGateUp2P,
                                                    expertFF, eu);
                        kQuantizeQ8_0x32_Batch<<<dim3(dBlocks, eu), 32, 0,
                                                 g_stream>>>(
                                s.moeGateUpP, moeQ8kDownP_, expertFF, eu,
                                static_cast<uint32_t>(downStrideB));
                        kQGemvQ8_0xQ8K_BatchDown<<<dim3((H + 7) / 8, eu),
                                                   dim3(32, 8), 0, g_stream>>>(
                                wqTable, moeQ8kDownP_, s.moeDownP, H, dBlocks,
                                expertRowBytesDown_,
                                static_cast<uint32_t>(downStrideB));
                        kQ35MoeAccumScaledBatch<<<1, 256, 0, g_stream>>>(
                                s.moeWgt, expertUsed, s.moeDownP, s.moeExpertOut,
                                seqLen, H, eu);
                        // TINYCODER_DUMP_MOE=1 + L0: dump rank 0's batched
                        // gate/up + down products for bisection (replaces the
                        // old per-rank [gpu fast L0] dump).
                        {
                            const char *dnR = std::getenv("TINYCODER_DUMP_MOE");
                            if (L == 0 && dnR != nullptr && dnR[0] != '\0' &&
                                dnR[0] != '0') {
                                cudaStreamSynchronize(g_stream);
                                float gu[4], gd[4];
                                cudaMemcpy(gu, s.moeGateUpP, 4 * sizeof(float),
                                           cudaMemcpyDeviceToHost);
                                cudaMemcpy(gd, s.moeDownP, 4 * sizeof(float),
                                           cudaMemcpyDeviceToHost);
                                std::fprintf(stderr,
                                             "[gpu fast batch L0] rank0 e=%d "
                                             "gateUp[0..3]={%.6f %.6f %.6f %.6f} "
                                             "down[0..3]={%.6f %.6f %.6f %.6f}\n",
                                             moeRouterIdxHost_.data()[0], gu[0],
                                             gu[1], gu[2], gu[3], gd[0], gd[1],
                                             gd[2], gd[3]);
                            }
                        }
                    } else
                    moePerRank: {
                        for (uint32_t r = 0; r < expertUsed; ++r) {
                            const int32_t *row = moeRouterIdxHost_.data() + r;
                            // Same-expert fast path: the whole batch selected the
                            // same expert at rank r (the common prefill case).
                            bool sameExpert = true;
                            const int32_t e0 = row[0];
                            for (uint32_t st = 1; st < seqLen; ++st) {
                                if (row[st * expertUsed] != e0) {
                                    sameExpert = false;
                                    break;
                                }
                            }
                            if (seqLen == 1 || (sameExpert && e0 >= 0)) {
                                const uint32_t e = static_cast<uint32_t>(e0);
                                cacheStat(e);
                                std::string cErr;
                                const uint8_t *downQ = nullptr;
                                const uint8_t *cacheBase =
                                        ensureExpertCached(L, e, cErr, &downQ);
                                if (cacheBase == nullptr) {
                                    errMsg = "qwen35moe expert cache miss: " + cErr;
                                    return false;
                                }
                                // Slice the cached arena: gate at cacheBase, up one
                                // gateSlice later (both [expertFF][H]), down in the
                                // slot's separate region ([H][expertFF]).
                                DeviceMatrix gExps;
                                gExps.q = const_cast<void *>(
                                        static_cast<const void *>(cacheBase));
                                gExps.rows = expertFF;
                                gExps.cols = H;
                                gExps.type = expertType_;
                                gExps.blocksPerRow = gBlocks;
                                gExps.rowBytes = expertRowBytesGate_;
                                DeviceMatrix uExps = gExps;
                                uExps.q = const_cast<void *>(
                                        static_cast<const void *>(
                                                cacheBase + static_cast<uint64_t>(expertFF_) *
                                                                    expertRowBytesGate_));
                                DeviceMatrix dExps;
                                dExps.q = const_cast<void *>(
                                        static_cast<const void *>(downQ));
                                dExps.rows = H;
                                dExps.cols = expertFF;
                                dExps.type = expertType_;
                                dExps.blocksPerRow = dBlocks;
                                dExps.rowBytes = expertRowBytesDown_;
                                if (L < 4) {
                                    const char *dnG = std::getenv(
                                            "TINYCODER_DUMP_MOE");
                                    if (dnG != nullptr && dnG[0] != '\0' &&
                                        dnG[0] != '0') {
                                        cudaStreamSynchronize(g_stream);
                                        float nrm[4];
                                        cudaMemcpy(nrm, s.norm, 4 * sizeof(float),
                                                   cudaMemcpyDeviceToHost);
                                        std::fprintf(
                                                stderr,
                                                "[gpu gatein L%u] r=%u e=%u "
                                                "snorm[0..3]={%.6f %.6f %.6f %.6f} "
                                                "cacheBase=%p\n",
                                                L, r, static_cast<uint32_t>(e0),
                                                nrm[0], nrm[1], nrm[2], nrm[3],
                                                cacheBase);
                                    }
                                }
                                if (seqLen == 1 && expertType_ == kTypeQ8_0 &&
                                    moeFastPath) {
                                    // Pre-quantized decode fast path: s.norm is
                                    // already Q8_32-quantized once into q8kN_
                                    // above; the down input (s.moeGateUp, the silu
                                    // output) is fp32 and needs its own 16-block
                                    // quantize into the transient q8k_.
                                    launchQGemvQ8_0Pre(gExps.q, q8kN_, s.moeGateUp,
                                                       expertFF, gBlocks,
                                                       expertRowBytesGate_);
                                    launchQGemvQ8_0Pre(uExps.q, q8kN_, s.moeGateUp2,
                                                       expertFF, gBlocks,
                                                       expertRowBytesGate_);
                                    kSiluMul<<<(seqLen * expertFF + 255) / 256, 256,
                                               0, g_stream>>>(
                                            s.moeGateUp, s.moeGateUp2,
                                            seqLen * expertFF);
                                    launchQuantQ8_0x32(s.moeGateUp, q8k_, expertFF);
                                    launchQGemvQ8_0Pre(dExps.q, q8k_, s.moeDown, H,
                                                       dBlocks,
                                                       expertRowBytesDown_);
                                    // TINYCODER_DUMP_MOE=1 + L0: dump the fast
                                    // path's per-rank output values so the routed
                                    // contribution can be compared against the
                                    // slow path / CPU reference.
                                    {
                                        const char *dnR = std::getenv(
                                                "TINYCODER_DUMP_MOE");
                                        if (L < 4 && dnR != nullptr &&
                                            dnR[0] != '\0' && dnR[0] != '0') {
                                            cudaStreamSynchronize(g_stream);
                                            float gu[4], gd[4], gwt = moeRouterWgtHost_[r];
                                            cudaMemcpy(gu, s.moeGateUp,
                                                       4 * sizeof(float),
                                                       cudaMemcpyDeviceToHost);
                                            cudaMemcpy(gd, s.moeDown,
                                                       4 * sizeof(float),
                                                       cudaMemcpyDeviceToHost);
                                            std::fprintf(
                                                    stderr,
                                                    "[gpu fast L%u] r=%u e=%u wgt=%.6f "
                                                    "gateUp[0..3]={%.6f %.6f %.6f "
                                                    "%.6f} down[0..3]={%.6f %.6f "
                                                    "%.6f %.6f}\n",
                                                    L, r, static_cast<uint32_t>(e0),
                                                    gwt, gu[0], gu[1], gu[2], gu[3],
                                                    gd[0], gd[1], gd[2], gd[3]);
                                        }
                                    }
                                } else if (!gemv(gExps, s.norm, s.moeGateUp, expertFF,
                                                 H, "ffnGateExps") ||
                                           !gemv(uExps, s.norm, s.moeGateUp2,
                                                 expertFF, H, "ffnUpExps")) {
                                    return false;
                                } else {
                                    kSiluMul<<<(seqLen * expertFF + 255) / 256, 256,
                                               0, g_stream>>>(
                                            s.moeGateUp, s.moeGateUp2,
                                            seqLen * expertFF);
                                    if (!gemv(dExps, s.moeGateUp, s.moeDown, H,
                                              expertFF, "ffnDownExpsMoe")) {
                                        return false;
                                    }
                                }
                                // wgt[s*expertUsed + r] via base+offset (kernel
                                // reads the fixed-rank slice at s.moeWgt + r).
                                kQ35MoeAccumScaled<<<seqLen, 256, 0, g_stream>>>(
                                        s.moeWgt + r, expertUsed, s.moeDown,
                                        s.moeExpertOut, seqLen, H);
                            } else {
                                // Per-token fallback: each token has its own expert
                                // at this rank; run per-token GEMVs with the token's
                                // own cached slice + weight.
                                for (uint32_t st = 0; st < seqLen; ++st) {
                                    const int32_t e = row[st * expertUsed];
                                    if (e < 0) continue;
                                    const uint32_t ue = static_cast<uint32_t>(e);
                                    cacheStat(ue);
                                    std::string cErr;
                                    const uint8_t *downQ = nullptr;
                                    const uint8_t *cacheBase =
                                            ensureExpertCached(L, ue, cErr, &downQ);
                                    if (cacheBase == nullptr) {
                                        errMsg = "qwen35moe expert cache miss: " +
                                                 cErr;
                                        return false;
                                    }
                                    float *gateRow = s.moeGateUp +
                                                     static_cast<size_t>(st) * expertFF;
                                    float *upRow = s.moeGateUp2 +
                                                   static_cast<size_t>(st) * expertFF;
                                    const float *normRow =
                                            s.norm + static_cast<size_t>(st) * H;
                                    float *downRow =
                                            s.moeDown + static_cast<size_t>(st) * H;
                                    float *outRow = s.moeExpertOut +
                                                    static_cast<size_t>(st) * H;
                                    launchQGemv(expertType_, cacheBase, normRow,
                                                gateRow, expertFF, H,
                                                expertRowBytesGate_, gBlocks, q8k_,
                                                q8kBytes_);
                                    launchQGemv(expertType_,
                                                cacheBase +
                                                        static_cast<uint64_t>(expertFF_) *
                                                                expertRowBytesGate_,
                                                normRow, upRow, expertFF, H,
                                                expertRowBytesGate_, gBlocks, q8k_,
                                                q8kBytes_);
                                    kSiluMul<<<(expertFF + 255) / 256, 256, 0,
                                               g_stream>>>(gateRow, upRow, expertFF);
                                    launchQGemv(expertType_, downQ, gateRow, downRow,
                                                H, expertFF, expertRowBytesDown_,
                                                dBlocks, q8k_, q8kBytes_);
                                    const float g = moeRouterWgtHost_[static_cast<size_t>(st) * expertUsed + r];
                                    kAddScaled1<<<1, 256, 0, g_stream>>>(
                                            g, downRow, outRow, H);
                                }
                            }
                        }// end per-rank loop (generic prefill / slow path)
                    }
                        if (moeTr) {
                            rankLoopMs += std::chrono::duration<double, std::milli>(
                                                  std::chrono::steady_clock::now() -
                                                  tRank0)
                                                  .count();
                        }
                    // TINYCODER_PER_LAYER=1: per-layer MoE block breakdown —
                    // wall time since startMoeBlock split into the instrumented
                    // sub-stages vs the (un-instrumented) gap.  This isolates
                    // the ~1s/token unexplained budget outside fill/router/rank.
                    {
                        const char *perL = std::getenv("TINYCODER_PER_LAYER");
                        if (perL != nullptr && perL[0] != '\0' && perL[0] != '0') {
                            double moeWallMs = std::chrono::duration<double,
                                                                     std::milli>(
                                                       std::chrono::steady_clock::now() - tMoe0)
                                                       .count();
                            // Accumulated so far this forward (per-token
                            // callback runs once per layer, so these are
                            // deltas by construction).
                            std::fprintf(
                                    stderr,
                                    "[moe L%u] wall=%.2f sync=%.2f router=%.2f "
                                    "rank=%.2f fill=%.2f hits=%llu misses=%llu\n",
                                    L, moeWallMs, syncWaitMs, cpuRouterMs,
                                    rankLoopMs, fillMs,
                                    static_cast<unsigned long long>(cacheHits),
                                    static_cast<unsigned long long>(cacheMisses));
                        }
                    }
                } else {
                    // ---- Default (no expert cache): the callback ran the
                    // full routed experts into moeFfnOutHost_; H2D the output.
                    // Option B (GPU trunk + CPU experts) is the DEFAULT expert
                    // placement for qwen35moe (TINYCODER_MOE_CACHE opts back
                    // into the device LRU cache); this is no longer a degraded
                    // fallback — it is the preferred hybrid.
                    if (!moeFallbackWarned) {
                        std::fprintf(stderr,
                                     "[gpu] L=%u: CPU-expert hybrid (GPU "
                                     "trunk + CPU experts; TINYCODER_MOE_CACHE "
                                     "enables the device expert cache)\n",
                                     L);
                        moeFallbackWarned = true;
                    }
                    // on g_stream and sync so kAddScaled sees the copy.
                    cudaMemcpyAsync(s.moeExpertOut, moeFfnOutHost_[buf].data(),
                                    normBytes, cudaMemcpyHostToDevice, g_stream);
                    cudaStreamSynchronize(g_stream);
                }
                // 6) Add the shared expert into the routed output
                //    (moeShexpDown was computed in step 2).
                if (w.ffnGateShexp.q != nullptr && w.ffnUpShexp.q != nullptr &&
                    w.ffnDownShexp.q != nullptr && sharedFF > 0) {
                    kAddScaled<<<seqLen, 256, 0, g_stream>>>(
                            s.moeShexpGate, s.moeShexpDown, s.moeExpertOut, seqLen,
                            H);
                    // TINYCODER_DUMP_MOE=1: dump the shared-expert contribution
                    // (sigmoid * sharedDown) so routed vs shared can be isolated.
                    {
                        const char *dnS = std::getenv("TINYCODER_DUMP_MOE");
                        if (moeCpuFn_ != nullptr && L == 0 && seqLen >= 1 &&
                            dnS != nullptr && dnS[0] != '\0' && dnS[0] != '0') {
                            cudaStreamSynchronize(g_stream);
                            float sd[4], sg = 0.0f;
                            cudaMemcpy(sd, s.moeShexpDown, 4 * sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            cudaMemcpy(&sg, s.moeShexpGate, sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            std::fprintf(stderr,
                                         "[gpu L0 shared] sig=%.6f shexpDown"
                                         "[0..3]={%.6f %.6f %.6f %.6f}\n",
                                         sg, sd[0], sd[1], sd[2], sd[3]);
                        }
                    }
                }
                // TINYCODER_DUMP_MOE=1: dump L0's completed MoE output (routed
                // experts + shared expert, BEFORE the residual add) so the
                // cache-driven decode fast path can be bisected against the
                // TINYCODER_MOE_SLOW generic path and the pure-CPU fallback
                // (both verified passing the SequentialDecode gate).
                // TOKEN-0 ONLY (the batch token order is deterministic):  dump
                // moeExpertOut[0..3] so it can be compared against the CPU
                // reference L0 ffnOut for the same token.
                {
                    const char *dnT = std::getenv("TINYCODER_DUMP_MOE");
                    if (moeCpuFn_ != nullptr && L == 0 && dnT != nullptr &&
                        dnT[0] != '\0' && dnT[0] != '0' && seqLen >= 1) {
                        cudaStreamSynchronize(g_stream);
                        float mo[4];
                        cudaMemcpy(mo, s.moeExpertOut, 4 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        std::fprintf(stderr,
                                     "[gpu L0 moeOut] moeExpertOut[0..3]={%.6f "
                                     "%.6f %.6f %.6f}\n",
                                     mo[0], mo[1], mo[2], mo[3]);
                    }
                }
                {
                    const char *dnM = std::getenv("TINYCODER_DUMP_MOE");
                    if (dnM != nullptr && dnM[0] != '\0' && dnM[0] != '0' &&
                        moeCpuFn_ != nullptr) {
                        // Dump the norm for THIS layer (all layers, token 0)
                        // so the first divergence layer can be identified
                        // against the CPU `[cpunorm]` dumps.
                        cudaStreamSynchronize(g_stream);
                        float n0[4];
                        cudaMemcpy(n0, s.norm, 4 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        std::fprintf(stderr,
                                     "[gpu norm] L%u norm[0..3]={%.6f %.6f "
                                     "%.6f %.6f}\n",
                                     L, n0[0], n0[1], n0[2], n0[3]);
                    }
                }
                {
                    const char *dnM = std::getenv("TINYCODER_DUMP_MOE");
                    if (L == 0 && dnM != nullptr && dnM[0] != '\0' &&
                        dnM[0] != '0' && moeCpuFn_ != nullptr) {
                        cudaStreamSynchronize(g_stream);
                        float mo[8], norm0[4];
                        cudaMemcpy(mo, s.moeExpertOut, 8 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        cudaMemcpy(norm0, s.norm, 4 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        std::fprintf(stderr,
                                     "[gpu moe] L0 moeExpertOut[0..7]={%.6f "
                                     "%.6f %.6f %.6f %.6f %.6f %.6f %.6f} "
                                     "norm[0..3]={%.6f %.6f %.6f %.6f}\n",
                                     mo[0], mo[1], mo[2], mo[3], mo[4], mo[5],
                                     mo[6], mo[7], norm0[0], norm0[1], norm0[2],
                                     norm0[3]);
                    }
                }
                // Flip for the next layer.
                moeBufFlip_ ^= 1u;
            } else {
                // ---------- Full-GPU routed MoE FFN (softmax top-k + shared expert) ----------
                // 1) Router logits = ffn_gate_inp (F32 [expertCount x H]) @ norm.
                if (!gemv(w.ffnGateInpMoe, s.norm, s.moeLogits, expertCount, H,
                          "ffnGateInpMoe")) {
                    return false;
                }
                // 2) softmax over ALL experts + top-k + renormalize (norm_w clamp).
                const uint32_t moeShmem =
                        sizeof(float) * (expertCount + 2 * expertUsed);
                kQ35MoeRouter<<<seqLen, 256, moeShmem, g_stream>>>(
                        s.moeLogits, s.moeIdx, s.moeWgt, s.moeWsum, seqLen,
                        expertCount, expertUsed);
                // Copy the tiny top-k selection back to the host: the per-expert
                // GEMVs slice rows from the packed expert matrices and need the
                // SELECTED expert ids (moeIdx) as a host-side row offset per token.
                // The renormalized weights (moeWgt) are copied too: the per-token
                // fallback scales by the token's OWN rank-r weight, and reading
                // that from the device buffer (s.moeWgt) on the host would be UB.
                if (moeIdxHost_.size() < static_cast<size_t>(seqLen) * expertUsed) {
                    moeIdxHost_.resize(static_cast<size_t>(seqLen) * expertUsed);
                    moeWgtHost_.resize(static_cast<size_t>(seqLen) * expertUsed);
                }
                cudaMemcpyAsync(moeIdxHost_.data(), s.moeIdx,
                                static_cast<size_t>(seqLen) * expertUsed *
                                        sizeof(int32_t),
                                cudaMemcpyDeviceToHost, g_stream);
                cudaMemcpyAsync(moeWgtHost_.data(), s.moeWgt,
                                static_cast<size_t>(seqLen) * expertUsed *
                                        sizeof(float),
                                cudaMemcpyDeviceToHost, g_stream);
                cudaStreamSynchronize(g_stream);

                // 3) Per-expert GEMVs.  Expert e of the packed ffnGateExps /
                //    ffnUpExps spans rows [e*expertFF, (e+1)*expertFF) (cols == H);
                //    ffnDownExpsMoe spans rows [e*H, (e+1)*H) (cols == expertFF).
                //    The routing weight array is laid out [seqLen][expertUsed]
                //    (fixed rank r).  We iterate ranks ASCENDING and accumulate
                //    each rank's weighted expert output into moeExpertOut.  The
                //    per-expert gate/up/down buffers are reused across ranks.
                cudaMemsetAsync(s.moeExpertOut, 0,
                                static_cast<uint64_t>(seqLen) * H * sizeof(float),
                                g_stream);
                // 3) Per-expert GEMVs + SwiGLU + down + weighted accumulation.
                //
                // The per-expert matrices are PACKED: expert e occupies rows
                // [e*expertFF, (e+1)*expertFF) of ffnGateExps / ffnUpExps
                // (cols == H) and rows [e*H, (e+1)*H) of ffnDownExpsMoe
                // (cols == expertFF).  The GPU kernels read a contiguous row
                // range (rowStart = e*rowsPerExpert), so the driver launches ONE
                // GEMV per selected expert with the sliced row base; the whole
                // token batch shares the expert at rank r (the router kernel's
                // top-k builds the SAME rank ordering for every token only when
                // the probs are strictly ordered — but rank r may differ across
                // tokens!).  Exact parity with the CPU (matMulVecRows per token)
                // requires per-(token, rank) GEMVs: the CPU accumulates
                // routingWeight[e] * down(e, token).  We batch over tokens ONLY
                // when the whole batch selected the SAME expert at rank r (the
                // overwhelmingly common prefill case); otherwise fall back to a
                // per-token GEMV with the token's own expert slice + weight.
                //
                // moeWgt is laid out [seqLen][expertUsed] (rank r fixed); the
                // kQ35MoeAccumScaled kernel takes wgt[token][r] with
                // wgtStride == expertUsed and accumulates out += w * src.
                for (uint32_t r = 0; r < expertUsed; ++r) {
                    const int32_t *row = moeIdxHost_.data() + r;
                    const float *wgtRow = moeWgtHost_.data() + r;
                    // Detect the same-expert fast path: all tokens choose the
                    // same expert at this rank.
                    bool sameExpert = true;
                    const int32_t e0 = row[0];
                    for (uint32_t st = 1; st < seqLen; ++st) {
                        if (row[st * expertUsed] != e0) {
                            sameExpert = false;
                            break;
                        }
                    }
                    if (seqLen == 1 || (sameExpert && e0 >= 0)) {
                        const uint32_t e = static_cast<uint32_t>(e0);
                        // Row-sliced copies of the expert's matrices: the `gemv`
                        // lambda handles decode (launchQGemv) AND prefill (fp16
                        // cuBLAS GEMM) uniformly.  `gemv` dequantizes the slice
                        // into wF16_ on demand (cached by q pointer).
                        DeviceMatrix gExps = w.ffnGateExps;
                        gExps.q = static_cast<void *>(
                                static_cast<uint8_t *>(w.ffnGateExps.q) +
                                static_cast<size_t>(e) * expertFF *
                                        w.ffnGateExps.rowBytes);
                        gExps.rows = expertFF;
                        DeviceMatrix uExps = w.ffnUpExps;
                        uExps.q = static_cast<void *>(
                                static_cast<uint8_t *>(w.ffnUpExps.q) +
                                static_cast<size_t>(e) * expertFF *
                                        w.ffnUpExps.rowBytes);
                        uExps.rows = expertFF;
                        DeviceMatrix dExps = w.ffnDownExpsMoe;
                        dExps.q = static_cast<void *>(
                                static_cast<uint8_t *>(w.ffnDownExpsMoe.q) +
                                static_cast<size_t>(e) * H *
                                        w.ffnDownExpsMoe.rowBytes);
                        dExps.rows = H;
                        if (!gemv(gExps, s.norm, s.moeGateUp, expertFF, H,
                                  "ffnGateExps") ||
                            !gemv(uExps, s.norm, s.moeGateUp2, expertFF, H,
                                  "ffnUpExps")) {
                            return false;
                        }
                        // SwiGLU: gate = silu(gate) * up (in-place, [seqLen][ff]).
                        kSiluMul<<<(seqLen * expertFF + 255) / 256, 256, 0,
                                   g_stream>>>(s.moeGateUp, s.moeGateUp2,
                                               seqLen * expertFF);
                        if (!gemv(dExps, s.moeGateUp, s.moeDown, H, expertFF,
                                  "ffnDownExpsMoe")) {
                            return false;
                        }
                        // Accumulate: moeExpertOut[s] += wgt[s][r] * moeDown[s].
                        // kQ35MoeAccumScaled reads `wgt[s*stride]` from the BASE
                        // pointer passed in, so pass s.moeWgt + r to reach the
                        // per-(token, rank) weight moeWgt[s*expertUsed + r].
                        kQ35MoeAccumScaled<<<seqLen, 256, 0, g_stream>>>(
                                s.moeWgt + r, expertUsed, s.moeDown, s.moeExpertOut,
                                seqLen, H);
                    } else {
                        // Per-token fallback: each token has its own expert.
                        for (uint32_t st = 0; st < seqLen; ++st) {
                            const int32_t e = row[st * expertUsed];
                            if (e < 0) continue;
                            const uint32_t ue = static_cast<uint32_t>(e);
                            const uint32_t gRow0 = ue * expertFF;
                            const uint32_t dRow0 = ue * H;
                            float *gateRow = s.moeGateUp + static_cast<size_t>(st) * expertFF;
                            float *upRow = s.moeGateUp2 + static_cast<size_t>(st) * expertFF;
                            float *normRow = s.norm + static_cast<size_t>(st) * H;
                            float *downRow = s.moeDown + static_cast<size_t>(st) * H;
                            float *outRow = s.moeExpertOut + static_cast<size_t>(st) * H;
                            DeviceMatrix gExps = w.ffnGateExps;
                            gExps.q = static_cast<void *>(
                                    static_cast<uint8_t *>(w.ffnGateExps.q) +
                                    static_cast<size_t>(gRow0) * w.ffnGateExps.rowBytes);
                            gExps.rows = expertFF;
                            DeviceMatrix uExps = w.ffnUpExps;
                            uExps.q = static_cast<void *>(
                                    static_cast<uint8_t *>(w.ffnUpExps.q) +
                                    static_cast<size_t>(gRow0) * w.ffnUpExps.rowBytes);
                            uExps.rows = expertFF;
                            DeviceMatrix dExps = w.ffnDownExpsMoe;
                            dExps.q = static_cast<void *>(
                                    static_cast<uint8_t *>(w.ffnDownExpsMoe.q) +
                                    static_cast<size_t>(dRow0) * w.ffnDownExpsMoe.rowBytes);
                            dExps.rows = H;
                            launchQGemv(gExps.type, gExps.q, normRow, gateRow,
                                        gExps.rows, gExps.cols, gExps.rowBytes,
                                        gExps.blocksPerRow, q8k_, q8kBytes_);
                            launchQGemv(uExps.type, uExps.q, normRow, upRow,
                                        uExps.rows, uExps.cols, uExps.rowBytes,
                                        uExps.blocksPerRow, q8k_, q8kBytes_);
                            kSiluMul<<<(expertFF + 255) / 256, 256, 0, g_stream>>>(
                                    gateRow, upRow, expertFF);
                            launchQGemv(dExps.type, dExps.q, gateRow, downRow,
                                        dExps.rows, dExps.cols, dExps.rowBytes,
                                        dExps.blocksPerRow, q8k_, q8kBytes_);
                            const float g = wgtRow[st * expertUsed];
                            kAddScaled1<<<1, 256, 0, g_stream>>>(g, downRow, outRow,
                                                                 H);
                        }
                    }
                }

                // ---------- Shared expert (sigmoid-gated SwiGLU) ----------
                if (w.ffnGateShexp.q != nullptr && w.ffnUpShexp.q != nullptr &&
                    w.ffnDownShexp.q != nullptr && sharedFF > 0) {
                    // Gate: g[s] = sigmoid(ffn_gate_inp_shexp @ norm[s]).
                    kQ35MoeShexpGate<<<seqLen, 256, 0, g_stream>>>(
                            s.norm, static_cast<const float *>(w.ffnGateInpShexp.q),
                            s.moeShexpGate, seqLen, H, H);
                    // gate + up (shared-expert matrices, full [sharedFF x H]).
                    if (!gemv(w.ffnGateShexp, s.norm, s.moeShexpGateUp, sharedFF, H,
                              "ffnGateShexp") ||
                        !gemv(w.ffnUpShexp, s.norm, s.moeShexpGateUp2, sharedFF, H,
                              "ffnUpShexp")) {
                        return false;
                    }
                    // SwiGLU: [seqLen][sharedFF].
                    kSiluMul<<<(seqLen * sharedFF + 255) / 256, 256, 0, g_stream>>>(
                            s.moeShexpGateUp, s.moeShexpGateUp2, seqLen * sharedFF);
                    // Down: full [H x sharedFF] matrix.
                    if (!gemv(w.ffnDownShexp, s.moeShexpGateUp, s.moeShexpDown, H,
                              sharedFF, "ffnDownShexp")) {
                        return false;
                    }
                    // Accumulate: moeExpertOut[s] += gate[s] * shexpDown[s].
                    kAddScaled<<<seqLen, 256, 0, g_stream>>>(
                            s.moeShexpGate, s.moeShexpDown, s.moeExpertOut, seqLen,
                            H);
                }
            }// end full-GPU MoE (else of moeCpuFn_)

            // ---- Residual: hidden += moeExpertOut ----
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.moeExpertOut, seqLen * H);
            phaseMark(L, 2);// ev2: after rank + addScaled + residual enqueues
            if (moeTr) endMoeBlock();
        }

        cudaError_t llErr = cudaStreamSynchronize(g_stream);
        if (llErr != cudaSuccess) {
            errMsg = std::string("qwen35moe prefix sync: ") +
                     cudaGetErrorString(llErr);
            return false;
        }
        kvPos_ += seqLen;

        // ---- Full offload: final RMSNorm + LM head ----
        if (finalNorm_) {
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.hidden, finalNorm_, H, seqLen, 1e-6f);
        }
        const void *lmQ = lmHeadQ_ != nullptr ? lmHeadQ_ : embedQ_;
        uint32_t lmType = lmHeadQ_ != nullptr ? lmHeadType_ : embedType_;
        uint32_t lmRowBytes =
                lmHeadQ_ != nullptr ? lmHeadRowBytes_ : embedRowBytes_;
        uint32_t lmBlocks = hostBlocksPerRow(lmType, H);
        if (computeAllLogits) {
            for (uint32_t si = 0; si < seqLen; ++si) {
                launchQGemv(lmType, lmQ, s.hidden + si * H,
                            s.logits + si * geom_.vocabSize, geom_.vocabSize, H,
                            lmRowBytes, lmBlocks, q8k_, q8kBytes_);
            }
            cudaMemcpyAsync(logitsOut, s.logits,
                            seqLen * geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        } else {
            uint32_t si = seqLen - 1;
            launchQGemv(lmType, lmQ, s.hidden + si * H, s.logits,
                        geom_.vocabSize, H, lmRowBytes, lmBlocks, q8k_, q8kBytes_);
            cudaMemcpyAsync(logitsOut, s.logits, geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        }
        cudaError_t fin = cudaStreamSynchronize(g_stream);
        if (fin != cudaSuccess) {
            errMsg = std::string("qwen35moe lmhead sync: ") +
                     cudaGetErrorString(fin);
            return false;
        }
        // ---- Phase profile accumulation (TINYCODER_PHASE_PROFILE=1) ----
        // All events are complete (we just synced the stream).  Per layer:
        //   trunkDev[i]  = ev1-ev0  device time: trunk + D2H (stream-ordered)
        //   rankDev[i]   = ev2-ev1  device time: rank GEMVs + addScaled +
        //                           residual (the host gap while the router/
        //                           fills ran is NOT on the stream, so it does
        //                           not inflate this span)
        // Summed over layers vs devMs (= device span moeEvStart..moeEvEnd,
        // which includes the LM head) gives:
        //   deviceTrunkMs / deviceRankMs: true GPU work per phase
        //   deviceOtherMs  = devMs - trunk - rank: LM head + tails
        //   hostBubbleMs   = totalMs - devMs: HOST-side time the GPU was idle
        //                    (router + fills + sync waits + enqueue) -- the
        //                    CPU/GPU overlap lever.
        if (phaseProf && !phaseEv.empty()) {
            double devTrunk = 0.0, devRank = 0.0;
            for (size_t li = 0; li < phaseEv.size(); ++li) {
                float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f;
                cudaEventElapsedTime(&t0, phaseEv[li][0], phaseEv[li][1]);
                cudaEventElapsedTime(&t1, phaseEv[li][1], phaseEv[li][2]);
                devTrunk += t0;
                devRank += t1;
                (void) t2;
                (void) ph;
            }
            float devMs2 = 0.0f;
            if (moeEvStart_ && moeEvEnd_) {
                cudaEventElapsedTime(&devMs2, moeEvStart_, moeEvEnd_);
            }
            double wallMs = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - tF0)
                                    .count();
            double devOther = static_cast<double>(devMs2) - devTrunk - devRank;
            if (devOther < 0.0) devOther = 0.0;
            std::fprintf(stderr,
                         "[phase] wall=%.1f dev=%.1f hostBubble=%.1f "
                         "devTrunk=%.1f(%u L) devRank=%.1f(%u L) devOther=%.1f "
                         "perLayerTrunk=%.2f perLayerRank=%.2f "
                         "perLayerBubble=%.2f\n",
                         wallMs, static_cast<double>(devMs2),
                         wallMs - static_cast<double>(devMs2), devTrunk,
                         static_cast<unsigned>(phaseEv.size()), devRank,
                         static_cast<unsigned>(phaseEv.size()), devOther,
                         devTrunk / static_cast<double>(phaseEv.size()),
                         devRank / static_cast<double>(phaseEv.size()),
                         (wallMs - static_cast<double>(devMs2)) /
                                 static_cast<double>(phaseEv.size()));
            // LM-head device span: devOther / (the final sync is not on the
            // timeline) approximates the finalNorm + LM-head GEMV + D2H.
        }
        if (moeTr) {
            double totalMs = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - tF0)
                                     .count();
            // Device elapsed (g_stream): everything the GPU actually ran from
            // the token H2D to the lmhead sync — the true GPU busy time.  The
            // difference (totalMs - devMs) is host-side enqueue/fill/CPU work
            // that the GPU was NOT busy with.
            float devMs = 0.0f;
            if (moeEvStart_ && moeEvEnd_) {
                cudaEventRecord(moeEvEnd_, g_stream);
                cudaEventSynchronize(moeEvEnd_);
                cudaEventElapsedTime(&devMs, moeEvStart_, moeEvEnd_);
            }
            std::fprintf(stderr,
                         "[moe fwd] seqLen=%u total=%.2f ms dev=%.1f "
                         "cacheHits=%llu cacheMisses=%llu missMB=%.0f "
                         "layerGpuMs=%.1f moeBlockMs=%.1f cpuRouterMs=%.1f "
                         "syncWaitMs=%.1f rankLoopMs=%.1f fillMs=%.1f%s\n",
                         seqLen, totalMs, devMs,
                         static_cast<unsigned long long>(cacheHits),
                         static_cast<unsigned long long>(cacheMisses),
                         cacheMissBytes / 1048576.0, layerGpuMs, moeBlockMs,
                         cpuRouterMs, syncWaitMs, rankLoopMs, fillMs,
                         moeFallbackWarned ? "  ** CPU-EXPERT FALLBACK ACTIVE **"
                                           : "");
        }
        // Destroy the phase-profile events (env-gated probe; no-op otherwise).
        for (auto &row: phaseEv) {
            for (auto &ev: row) {
                cudaEventDestroy(ev);
            }
        }
        return true;
    }

    bool GPUModel::forward(const std::vector<int32_t> &tokens, bool computeAllLogits,
                           float *logitsOut, std::string &errMsg) {
        if (!allocated_) {
            errMsg = "GPU model not uploaded";
            return false;
        }
        if (tokens.empty()) return true;
        auto t0 = std::chrono::steady_clock::now();
#define TIMING_TAG(what)                                               \
    do {                                                               \
        if (g_verbose) {                                               \
            double ms = std::chrono::duration<double, std::milli>(     \
                                std::chrono::steady_clock::now() - t0) \
                                .count();                              \
            std::fprintf(stderr, "[gpu fwd %6.2f ms] %s\n", ms, what); \
        }                                                              \
    } while (0)

        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t nGpu = geom_.numGpuLayers;
        if (nGpu == 0) nGpu = geom_.numLayers;
        uint32_t H = geom_.hiddenSize, I = geom_.intermediateSize;
        uint32_t nHeads = geom_.numAttentionHeads, nKV = geom_.numKVHeads;
        uint32_t hd = geom_.headDim;
        uint32_t qLen = nHeads * hd, kvLen = nKV * hd;
        bool fullOffload = (nGpu == geom_.numLayers);

        if (!ensureScratch(seqLen, errMsg)) return false;
        DeviceScratch &s = scratch_;
        cudaMemcpyAsync(s.tokens, tokens.data(), seqLen * sizeof(int32_t),
                        cudaMemcpyHostToDevice, g_stream);
        cudaStreamSynchronize(g_stream);

        uint32_t pos = static_cast<uint32_t>(kvPos_);

        // ---- Embedding ----
        {
            uint32_t blocksPerRow = hostBlocksPerRow(embedType_, H);
            const uint8_t *eQ = reinterpret_cast<const uint8_t *>(embedQ_);
            if (embedType_ == kTypeQ5_0) {
                kEmbedDequantSmall32<kTypeQ5_0><<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ8_0) {
                kEmbedDequantSmall32<kTypeQ8_0><<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeIQ3S) {
                kEmbedDequantIQ3S<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ4K) {
                kEmbedDequantQ4K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ5K) {
                kEmbedDequantQ5K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else {
                kEmbedDequant<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            }
        }
        TIMING_TAG("embed queued");

        // ---- Layer loop (only the offloaded prefix) ----
        cudaEvent_t evL0 = nullptr, evL1 = nullptr;
        std::vector<cudaEvent_t> evQKV, evAttn, evFFN;
        // per-matrix decode breakdown: attnO, gate(+up, up to silu),
        // down.  Only meaningful for seqLen == 1 (decode).
        std::vector<cudaEvent_t> evMtxAttnO, evMtxGateUp, evMtxDown;
        const bool traceStages = g_verbose && seqLen == 1;
        if (g_verbose) {
            cudaEventCreate(&evL0);
            cudaEventCreate(&evL1);
            cudaEventRecord(evL0, g_stream);
        }
        if (traceStages) {
            evQKV.resize(nGpu);
            evAttn.resize(nGpu);
            evFFN.resize(nGpu);
            evMtxAttnO.resize(nGpu);
            evMtxGateUp.resize(nGpu);
            evMtxDown.resize(nGpu);
            for (uint32_t i = 0; i < nGpu; ++i) {
                cudaEventCreate(&evQKV[i]);
                cudaEventCreate(&evAttn[i]);
                cudaEventCreate(&evFFN[i]);
                cudaEventCreate(&evMtxAttnO[i]);
                cudaEventCreate(&evMtxGateUp[i]);
                cudaEventCreate(&evMtxDown[i]);
            }
        }
        for (uint32_t L = 0; L < nGpu; ++L) {
            if (L == 0) TIMING_TAG("layer loop start");
            const DeviceLayer &w = layers_[L];
            float *kvK = kvK_ + static_cast<size_t>(L) * geom_.maxSeqLen * kvLen;
            float *kvV = kvV_ + static_cast<size_t>(L) * geom_.maxSeqLen * kvLen;


            // Attention RMSNorm -> s.norm (OUT-OF-PLACE: s.hidden is the
            // residual stream and must survive until the residual add below).
            // Grid = seqLen: the kernel uses ONE BLOCK PER ROW (row ==
            // blockIdx.x), so `(seqLen + 255) / 256` would only normalize the
            // first row for seqLen <= 256.
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.rmsNormAttn, H, seqLen, 1e-6f);

            // Q/K/V projections read the NORMED hidden (s.norm), not the
            // residual stream (s.hidden).
            if (seqLen == 1) {
                launchQGemv(w.attnQ.type, w.attnQ.q, s.norm, s.q, w.attnQ.rows,
                            w.attnQ.cols, w.attnQ.rowBytes, w.attnQ.blocksPerRow,
                            q8k_, q8kBytes_);
                launchQGemv(w.attnK.type, w.attnK.q, s.norm, s.k, w.attnK.rows,
                            w.attnK.cols, w.attnK.rowBytes, w.attnK.blocksPerRow,
                            q8k_, q8kBytes_);
                launchQGemv(w.attnV.type, w.attnV.q, s.norm, s.v, w.attnV.rows,
                            w.attnV.cols, w.attnV.rowBytes, w.attnV.blocksPerRow,
                            q8k_, q8kBytes_);
            } else {
                // cuBLAS fp16 GEMM (batch): cublasGemmEx tensor-op requires BOTH
                // A and B in fp16 (fp32 x fp16 returns CUBLAS_STATUS_NOT_SUPPORTED),
                // so keep an fp16 twin of the RMSNorm'd hidden here.
                kF32ToF16<<<(seqLen * H + 511) / 512, 256, 0, g_stream>>>(
                        s.norm, reinterpret_cast<__half2 *>(s.hiddenF16),
                        (seqLen * H) / 2);
                // out[seqLen][rows] = x[seqLen][cols] @ W[rows][cols]^T
                // CUBLAS is column-major: put W in the A slot (op=T gives
                // W^T) and x in the B slot (op=N), with C sized [rows][seqLen]
                // in column-major order == the row-major [seqLen][rows] output
                // buffer.  (The old transposed parameterization -- m=seqLen,
                // n=rows, ldc=seqLen -- wrote element (i,j) at flat[i+j*seqLen]
                // instead of flat[i*rows+j], scrambling every matrix output
                // whenever seqLen > 1.  Decode (seqLen==1) was unaffected,
                // which is why single-token generation looked sane while the
                // 40-token prefill produced NaN.)
                auto gemmXWt = [&](float *out, const void *x, const void *wF16,
                                   uint32_t rows, uint32_t cols) -> bool {
                    // computeType CUDA_R_32F requires alpha/beta as float*.
                    // (Passing __half* made cuBLAS read the half 1.0/0.0 bits
                    // plus the adjacent half as one float -- effectively a
                    // tiny alpha that underflowed the prefill GEMM results.)
                    float alpha = 1.0f, beta = 0.0f;
                    cublasStatus_t st = cublasGemmEx(
                            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(rows), static_cast<int>(seqLen),
                            static_cast<int>(cols), &alpha, wF16, CUDA_R_16F,
                            static_cast<int>(cols), x, CUDA_R_16F,
                            static_cast<int>(cols), &beta, out, CUDA_R_32F,
                            static_cast<int>(rows), CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    return st == CUBLAS_STATUS_SUCCESS;
                };
                // Stream each weight matrix's fp16 form on-device (no persistent
                // twins).  Q/K/V share the hiddenF16 X buffer; the three GEMMs
                // each re-dequant its own matrix into wF16_.
                void *wF = nullptr;
                if (!dequantMatrixF16(w.attnQ, &wF, errMsg) ||
                    !gemmXWt(s.q, s.hiddenF16, wF, w.attnQ.rows, w.attnQ.cols) ||
                    !dequantMatrixF16(w.attnK, &wF, errMsg) ||
                    !gemmXWt(s.k, s.hiddenF16, wF, w.attnK.rows, w.attnK.cols) ||
                    !dequantMatrixF16(w.attnV, &wF, errMsg) ||
                    !gemmXWt(s.v, s.hiddenF16, wF, w.attnV.rows, w.attnV.cols)) {
                    errMsg = errMsg.empty() ? "cublasGemmEx Q/K/V failed" : errMsg;
                    return false;
                }
                if (g_verbose) {
                    cudaError_t e = cudaStreamSynchronize(g_stream);
                    std::fprintf(stderr, "[gpu loop] L=%u QKV: %s\n", L,
                                 e == cudaSuccess ? "ok"
                                                  : cudaGetErrorString(e));
                }
            }

            // Biases (Qwen2): bias is ONE row (qLen/kvLen floats) broadcast
            // across every token — kAddBias, never kAddResidual (which would
            // index the bias array up to seqLen*biasLen and overread it).
            if (w.attnQBias) {
                kAddBias<<<seqLen, 256, 0, g_stream>>>(s.q, w.attnQBias, seqLen,
                                                       qLen);
            }
            if (w.attnKBias) {
                kAddBias<<<seqLen, 256, 0, g_stream>>>(s.k, w.attnKBias, seqLen,
                                                       kvLen);
            }
            if (w.attnVBias) {
                kAddBias<<<seqLen, 256, 0, g_stream>>>(s.v, w.attnVBias, seqLen,
                                                       kvLen);
            }
            if (traceStages) cudaEventRecord(evQKV[L], g_stream);

            // RoPE(Q) + store K/V (K rotation fused)
            kRoPEQ<<<seqLen, 128, 0, g_stream>>>(s.q, ropeCos_, ropeSin_, seqLen,
                                                 nHeads, hd, pos);
            kStoreKVRope<<<seqLen, 256, 0, g_stream>>>(
                    s.k, s.v, kvK, kvV, seqLen, nKV, hd, pos, pos, ropeCos_, ropeSin_);

            // Flash attention (warp per token-head).  Compile-time headDim
            // so the accumulators live in registers; grid.y covers exactly
            // nHeads warps (no redundant warps).
            float invSqrt = 1.0f / std::sqrt(static_cast<float>(hd));
            dim3 blk(32, 4);
            if (hd == 128) {
                kWarpAttention<128><<<dim3(seqLen, (nHeads + 3) / 4), blk, 0,
                                      g_stream>>>(s.q, kvK, kvV, s.attnOut, seqLen,
                                                  nHeads, nKV, pos, invSqrt);
            } else if (hd == 64) {
                kWarpAttention<64><<<dim3(seqLen, (nHeads + 3) / 4), blk, 0,
                                     g_stream>>>(s.q, kvK, kvV, s.attnOut, seqLen,
                                                 nHeads, nKV, pos, invSqrt);
            } else {
                kWarpAttention<32><<<dim3(seqLen, (nHeads + 3) / 4), blk, 0,
                                     g_stream>>>(s.q, kvK, kvV, s.attnOut, seqLen,
                                                 nHeads, nKV, pos, invSqrt);
            }
            if (g_verbose) {
                cudaError_t e = cudaStreamSynchronize(g_stream);
                std::fprintf(stderr, "[gpu loop] L=%u kv+attn: %s\n", L,
                             e == cudaSuccess ? "ok"
                                              : cudaGetErrorString(e));
            }
            if (traceStages) cudaEventRecord(evAttn[L], g_stream);

            // attnO projection + residual
            if (traceStages) cudaEventRecord(evMtxAttnO[L], g_stream);
            if (seqLen == 1) {
                launchQGemv(w.attnO.type, w.attnO.q, s.attnOut, s.attnProj,
                            w.attnO.rows, w.attnO.cols, w.attnO.rowBytes,
                            w.attnO.blocksPerRow, q8k_, q8kBytes_);
            } else {
                // attnOut is the fp32 attention result; make an fp16 twin for
                // the fp16 tensor-core GEMM.
                kF32ToF16<<<(seqLen * qLen + 511) / 512, 256, 0, g_stream>>>(
                        s.attnOut, reinterpret_cast<__half2 *>(s.attnOutF16),
                        (seqLen * qLen) / 2);
                // CUDA_R_32F compute type -> alpha/beta must be float*.
                float alpha = 1.0f, beta = 0.0f;
                void *wF = nullptr;
                if (!dequantMatrixF16(w.attnO, &wF, errMsg)) return false;
                // attnProj[seqLen][rows] = attnOut[seqLen][cols] @ W[rows][cols]^T
                // (see the Q/K/V gemmXWt comment: CUBLAS column-major requires
                // W in the A slot, x in B, ldc=rows).
                cublasStatus_t st = cublasGemmEx(
                        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w.attnO.rows), static_cast<int>(seqLen),
                        static_cast<int>(w.attnO.cols), &alpha, wF, CUDA_R_16F,
                        static_cast<int>(w.attnO.cols), s.attnOutF16, CUDA_R_16F,
                        static_cast<int>(w.attnO.cols), &beta, s.attnProj, CUDA_R_32F,
                        static_cast<int>(w.attnO.rows), CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) {
                    errMsg = "cublasGemmEx attnO failed";
                    return false;
                }
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.attnProj, seqLen * H);

            // FFN RMSNorm -> s.norm (OUT-OF-PLACE, one block per row).
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.rmsNormFFN, H, seqLen, 1e-6f);

            // gate / up + SwiGLU: inputs are the FFN RMSNorm'd vector (s.norm).
            if (traceStages) cudaEventRecord(evMtxGateUp[L], g_stream);
            if (seqLen == 1) {
                launchQGemv(w.ffnGate.type, w.ffnGate.q, s.norm, s.gate,
                            w.ffnGate.rows, w.ffnGate.cols, w.ffnGate.rowBytes,
                            w.ffnGate.blocksPerRow, q8k_, q8kBytes_);
                launchQGemv(w.ffnUp.type, w.ffnUp.q, s.norm, s.up, w.ffnUp.rows,
                            w.ffnUp.cols, w.ffnUp.rowBytes, w.ffnUp.blocksPerRow,
                            q8k_, q8kBytes_);
            } else {
                // hiddenF16 was converted before Q/K/V, but s.norm has since
                // been refreshed by the FFN RMSNorm, so rebuild the fp16 twin
                // from s.norm for the FFN GEMMs.
                kF32ToF16<<<(seqLen * H + 511) / 512, 256, 0, g_stream>>>(
                        s.norm, reinterpret_cast<__half2 *>(s.hiddenF16),
                        (seqLen * H) / 2);
                // CUDA_R_32F compute type -> alpha/beta must be float*.
                float alpha = 1.0f, beta = 0.0f;
                void *wF = nullptr;
                // gate/up[seqLen][rows] = norm[seqLen][cols] @ W[rows][cols]^T
                // (CUBLAS column-major: W in A, x in B, ldc=rows).
                if (!dequantMatrixF16(w.ffnGate, &wF, errMsg)) return false;
                cublasStatus_t st = cublasGemmEx(
                        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w.ffnGate.rows), static_cast<int>(seqLen),
                        static_cast<int>(w.ffnGate.cols), &alpha, wF, CUDA_R_16F,
                        static_cast<int>(w.ffnGate.cols), s.hiddenF16,
                        CUDA_R_16F, static_cast<int>(w.ffnGate.cols), &beta,
                        s.gate, CUDA_R_32F, static_cast<int>(w.ffnGate.rows),
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (!dequantMatrixF16(w.ffnUp, &wF, errMsg)) return false;
                cublasStatus_t st2 = cublasGemmEx(
                        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w.ffnUp.rows), static_cast<int>(seqLen),
                        static_cast<int>(w.ffnUp.cols), &alpha, wF, CUDA_R_16F,
                        static_cast<int>(w.ffnUp.cols), s.hiddenF16,
                        CUDA_R_16F, static_cast<int>(w.ffnUp.cols), &beta,
                        s.up, CUDA_R_32F, static_cast<int>(w.ffnUp.rows),
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS || st2 != CUBLAS_STATUS_SUCCESS) {
                    errMsg = "cublasGemmEx gate/up failed";
                    return false;
                }
            }
            kSiluMul<<<(seqLen * I + 255) / 256, 256, 0, g_stream>>>(s.gate, s.up,
                                                                     seqLen * I);

            // down projection + residual
            if (traceStages) cudaEventRecord(evMtxDown[L], g_stream);
            if (seqLen == 1) {
                launchQGemv(w.ffnDown.type, w.ffnDown.q, s.gate, s.ffnOut,
                            w.ffnDown.rows, w.ffnDown.cols, w.ffnDown.rowBytes,
                            w.ffnDown.blocksPerRow, q8k_, q8kBytes_);
            } else {
                // gate holds silu(gate)*up in fp32; fp16 twin for the GEMM.
                kF32ToF16<<<(seqLen * I + 511) / 512, 256, 0, g_stream>>>(
                        s.gate, reinterpret_cast<__half2 *>(s.gateF16),
                        (seqLen * I) / 2);
                // CUDA_R_32F compute type -> alpha/beta must be float*.
                float alpha = 1.0f, beta = 0.0f;
                void *wF = nullptr;
                if (!dequantMatrixF16(w.ffnDown, &wF, errMsg)) return false;
                // ffnOut[seqLen][rows] = gate[seqLen][cols] @ W[rows][cols]^T
                // (CUBLAS column-major: W in A, x in B, ldc=rows).
                cublasStatus_t st = cublasGemmEx(
                        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w.ffnDown.rows), static_cast<int>(seqLen),
                        static_cast<int>(w.ffnDown.cols), &alpha, wF, CUDA_R_16F,
                        static_cast<int>(w.ffnDown.cols), s.gateF16,
                        CUDA_R_16F, static_cast<int>(w.ffnDown.cols), &beta,
                        s.ffnOut, CUDA_R_32F, static_cast<int>(w.ffnDown.rows),
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) {
                    errMsg = "cublasGemmEx ffnDown failed";
                    return false;
                }
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.ffnOut, seqLen * H);

            if (g_verbose) {
                cudaError_t e = cudaStreamSynchronize(g_stream);
                std::fprintf(stderr, "[gpu loop] L=%u ffn: %s\n", L,
                             e == cudaSuccess ? "ok"
                                              : cudaGetErrorString(e));
            }
            if (traceStages) cudaEventRecord(evFFN[L], g_stream);
        }
        cudaError_t llErr = cudaStreamSynchronize(g_stream);
        if (llErr != cudaSuccess) {
            if (evL0) cudaEventDestroy(evL0);
            if (evL1) cudaEventDestroy(evL1);
            errMsg = std::string("GPU layer loop sync: ") + cudaGetErrorString(llErr);
            return false;
        }
        if (g_verbose) {
            cudaEventRecord(evL1, g_stream);
            cudaEventSynchronize(evL1);
            float gpuMs = 0.0f;
            cudaEventElapsedTime(&gpuMs, evL0, evL1);
            std::fprintf(stderr, "[gpu fwd   gpu-ms] layer loop GPU elapsed = %.2f ms (%u layers)\n",
                         gpuMs, nGpu);
            cudaEventDestroy(evL0);
            cudaEventDestroy(evL1);
            evL0 = evL1 = nullptr;
        }
        if (traceStages && g_verbose) {
            double accQKV = 0.0, accAttn = 0.0, accFFN = 0.0;
            for (uint32_t i = 0; i < nGpu; ++i) {
                float a = 0.0f, b = 0.0f, c = 0.0f;
                if (i == 0) {
                    cudaEventElapsedTime(&a, evL0, evQKV[0]);
                } else {
                    cudaEventElapsedTime(&a, evFFN[i - 1], evQKV[i]);
                }
                cudaEventElapsedTime(&b, evQKV[i], evAttn[i]);
                cudaEventElapsedTime(&c, evAttn[i], evFFN[i]);
                accQKV += a;
                accAttn += b;
                accFFN += c;
            }
            std::fprintf(stderr,
                         "[gpu fwd   gpu-ms] decode stage sums: QKV/rms=%.2f ms, kv/attn/rope=%.2f ms, attnO/ffn=%.2f ms\n",
                         accQKV, accAttn, accFFN);
            // per-matrix sums inside the FFN-heavy stage (event stream order per
            // layer: evMtxAttnO, evMtxGateUp, evMtxDown, evFFN).  attnO span =
            // attnO GEMV + attnO residual + FFN RMSNorm; gate+up span = gate
            // GEMV + up GEMV + kSiluMul; down span = down GEMV + residual.
            double accAO = 0.0, accGU = 0.0, accDn = 0.0;
            for (uint32_t i = 0; i < nGpu; ++i) {
                float aa = 0.0f, gu = 0.0f, dn = 0.0f;
                cudaEventElapsedTime(&aa, evMtxAttnO[i], evMtxGateUp[i]);
                cudaEventElapsedTime(&gu, evMtxGateUp[i], evMtxDown[i]);
                cudaEventElapsedTime(&dn, evMtxDown[i], evFFN[i]);
                accAO += aa;
                accGU += gu;
                accDn += dn;
            }
            std::fprintf(stderr,
                         "[gpu fwd   gpu-ms] decode matrix sums: attnO=%.2f ms, gate+up=%.2f ms, down=%.2f ms\n",
                         accAO, accGU, accDn);
            for (uint32_t i = 0; i < nGpu; ++i) {
                cudaEventDestroy(evQKV[i]);
                cudaEventDestroy(evAttn[i]);
                cudaEventDestroy(evFFN[i]);
                cudaEventDestroy(evMtxAttnO[i]);
                cudaEventDestroy(evMtxGateUp[i]);
                cudaEventDestroy(evMtxDown[i]);
            }
        }

        kvPos_ += seqLen;

        if (!fullOffload) {
            // Partial offload: forward() stops after the offloaded layer prefix
            // and leaves the hidden state on the device.  The caller picks it up
            // via copyHiddenOut() and continues the remaining layers on the CPU
            // against the host KV cache.  (kvPos_ has already advanced.)
            cudaStreamSynchronize(g_stream);
            return true;
        }

        // ---- Full offload: final RMSNorm + LM head ----
        TIMING_TAG("layer loop done");
        // Final RMSNorm -> place the result INTO s.hidden (the residual stream
        // is no longer needed once the last layer's residual add has run).
        // One block per row.
        if (finalNorm_) {
            kRMSNormRow<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                    s.hidden, s.hidden, finalNorm_, H, seqLen, 1e-6f);
        }

        // LM head: quantized GEMV. Tied head -> token-embedding matrix
        // (embedQ_). Separate output.weight -> lmHeadQ_ (when uploaded).
        TIMING_TAG("lmhead start");
        const void *lmQ = lmHeadQ_ != nullptr ? lmHeadQ_ : embedQ_;
        uint32_t lmType = lmHeadQ_ != nullptr ? lmHeadType_ : embedType_;
        uint32_t lmRowBytes = lmHeadQ_ != nullptr ? lmHeadRowBytes_ : embedRowBytes_;
        uint32_t blocksPerRow = hostBlocksPerRow(lmType, H);
        if (computeAllLogits) {
            for (uint32_t si = 0; si < seqLen; ++si) {
                launchQGemv(lmType, lmQ, s.hidden + si * H,
                            s.logits + si * geom_.vocabSize, geom_.vocabSize, H,
                            lmRowBytes, blocksPerRow, q8k_, q8kBytes_);
            }
            cudaMemcpyAsync(logitsOut, s.logits,
                            seqLen * geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        } else {
            uint32_t si = seqLen - 1;
            launchQGemv(lmType, lmQ, s.hidden + si * H, s.logits,
                        geom_.vocabSize, H, lmRowBytes, blocksPerRow, q8k_,
                        q8kBytes_);
            cudaMemcpyAsync(logitsOut, s.logits, geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        }
        cudaError_t lerr = cudaGetLastError();
        if (lerr != cudaSuccess) {
            errMsg = std::string("GPU lmhead launch: ") + cudaGetErrorString(lerr);
            return false;
        }

        cudaError_t fin = cudaStreamSynchronize(g_stream);
        if (fin != cudaSuccess) {
            errMsg = std::string("GPU forward final sync: ") +
                     cudaGetErrorString(fin);
            return false;
        }
        TIMING_TAG("forward complete");
        return true;
    }

    bool GPUModel::copyHiddenOut(float *hiddenOut, uint32_t seqLen,
                                 std::string &errMsg) {
        if (!allocated_ || !scratchAlloc_) {
            errMsg = "GPU model not uploaded / no hidden state";
            return false;
        }
        if (seqLen > scratch_.seqCap) {
            errMsg = "copyHiddenOut: seqLen exceeds scratch capacity";
            return false;
        }
        cudaError_t e = cudaMemcpy(hiddenOut, scratch_.hidden,
                                   static_cast<size_t>(seqLen) * geom_.hiddenSize *
                                           sizeof(float),
                                   cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            errMsg = std::string("copyHiddenOut: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    void GPUModel::clearKVCache() {
        if (!allocated_) return;
        uint32_t nGpu = geom_.numGpuLayers;
        uint64_t kvBytes = static_cast<uint64_t>(geom_.maxSeqLen) *
                           geom_.numKVHeads * geom_.headDim * sizeof(float);
        cudaMemsetAsync(kvK_, 0, static_cast<size_t>(nGpu) * kvBytes, g_stream);
        cudaMemsetAsync(kvV_, 0, static_cast<size_t>(nGpu) * kvBytes, g_stream);
        // Qwen35 / Qwen35MoE: also reset the persistent gated-delta-net + conv1d
        // state so a fresh session starts from an empty recurrent state (matches
        // the CPU kvCache_.q35ConvState / q35GdnState zeroing on clearKVCache).
        // The MoE arch shares the same recurrent block, so both must reset.
        if (geom_.architecture == 1 || geom_.architecture == 2) {
            const uint32_t q35KeyDim = geom_.ssmStateSize * geom_.ssmGroupCount;
            const uint32_t q35QkvDim = 2 * q35KeyDim + geom_.ssmInnerSize;
            const uint32_t q35HeadV =
                    geom_.ssmInnerSize / std::max(geom_.ssmTimeStepRank, 1u);
            uint64_t q35ConvBytes =
                    static_cast<uint64_t>(nGpu) * (geom_.ssmConvKernel - 1) *
                    q35QkvDim * sizeof(float);
            uint64_t q35GdnBytes =
                    static_cast<uint64_t>(nGpu) * geom_.ssmTimeStepRank * q35HeadV *
                    q35HeadV * sizeof(float);
            if (q35ConvBytes == 0) q35ConvBytes = 1;
            if (q35GdnBytes == 0) q35GdnBytes = 1;
            cudaMemsetAsync(q35ConvState_, 0,
                            static_cast<size_t>(q35ConvBytes), g_stream);
            cudaMemsetAsync(q35GdnState_, 0,
                            static_cast<size_t>(q35GdnBytes), g_stream);
        }
        kvPos_ = 0;
        cudaStreamSynchronize(g_stream);
    }

    void GPUModel::destroy() {
        if (!allocated_) return;
        if (layers_) {
            uint32_t nGpu = geom_.numGpuLayers;
            for (uint32_t L = 0; L < nGpu; ++L) {
                DeviceLayer &l = layers_[L];
                auto freeMat = [](DeviceMatrix &m) {
                    cudaFree(m.q);
                    m.q = nullptr;
                };
                auto freeF32 = [](float *&p) {
                    cudaFree(p);
                    p = nullptr;
                };
                freeMat(l.attnQ);
                freeMat(l.attnK);
                freeMat(l.attnV);
                freeMat(l.attnO);
                freeMat(l.ffnGate);
                freeMat(l.ffnUp);
                freeMat(l.ffnDown);
                freeMat(l.attnQKV);
                freeMat(l.attnGate);
                freeMat(l.ssmAlphaQ);
                freeMat(l.ssmBetaQ);
                freeMat(l.ssmOut);
                freeMat(l.ssmConv1d);
                // Qwen35MoE (qwen35moe) MoE FFN matrices.
                freeMat(l.ffnGateInpMoe);
                freeMat(l.ffnGateExps);
                freeMat(l.ffnUpExps);
                freeMat(l.ffnDownExpsMoe);
                freeMat(l.ffnGateShexp);
                freeMat(l.ffnUpShexp);
                freeMat(l.ffnDownShexp);
                freeMat(l.ffnGateInpShexp);
                freeF32(l.attnQBias);
                freeF32(l.attnKBias);
                freeF32(l.attnVBias);
                freeF32(l.rmsNormAttn);
                freeF32(l.rmsNormFFN);
                freeF32(l.ssmABroadcast);
                freeF32(l.ssmDtBiasFull);
                freeF32(l.ssmNorm);
                freeF32(l.attnQNorm);
                freeF32(l.attnKNorm);
                freeF32(l.postAttnNorm);
            }
            delete[] layers_;
            layers_ = nullptr;
        }
        cudaFree(kvK_);
        cudaFree(kvV_);
        cudaFree(finalNorm_);
        cudaFree(embedQ_);
        cudaFree(lmHeadQ_);
        cudaFree(ropeCos_);
        cudaFree(ropeSin_);
        cudaFree(mropeCos_);
        cudaFree(mropeSin_);
        cudaFree(q35ConvState_);
        cudaFree(q35GdnState_);
        kvK_ = kvV_ = nullptr;
        finalNorm_ = embedQ_ = nullptr;
        lmHeadQ_ = nullptr;
        ropeCos_ = ropeSin_ = nullptr;
        mropeCos_ = mropeSin_ = nullptr;
        q35ConvState_ = q35GdnState_ = nullptr;
        if (wF16_) {
            cudaFree(wF16_);
            wF16_ = nullptr;
        }
        wF16Bytes_ = 0;
        wF16SrcQ_ = nullptr;
        if (q8k_) {
            cudaFree(q8k_);
            q8k_ = nullptr;
        }
        q8kBytes_ = 0;
        if (q8kN_) {
            cudaFree(q8kN_);
            q8kN_ = nullptr;
        }
        q8kNBytes_ = 0;
        if (moeQ8kDownP_) {
            cudaFree(moeQ8kDownP_);
            moeQ8kDownP_ = nullptr;
        }
        moeQ8kDownPBytes_ = 0;
        if (moeRankWq_) {
            cudaFree(moeRankWq_);
            moeRankWq_ = nullptr;
        }
        moeRankWqBytes_ = 0;
        // CPU-expert hybrid handoff events (lazily created, freed at destroy).
        for (uint32_t i = 0; i < kMoeEventRing; ++i) {
            if (eNormReady[i]) {
                cudaEventDestroy(eNormReady[i]);
                eNormReady[i] = nullptr;
            }
            if (eCpuDone[i]) {
                cudaEventDestroy(eCpuDone[i]);
                eCpuDone[i] = nullptr;
            }
        }
        moeBufFlip_ = 0;
        // Unregister the pinned norm-mirror ranges BEFORE the vectors free
        // their storage (cudaHostUnregister on a freed pointer is UB).
        for (uint32_t pi = 0; pi < 2; ++pi) {
            if (moeNormPinned_[pi] != nullptr) {
                cudaHostUnregister(moeNormPinned_[pi]);
                moeNormPinned_[pi] = nullptr;
            }
        }
        moeNormHost_.clear();
        moeFfnOutHost_.clear();
        moeLogitsHost_.clear();
        moeRouterIdxHost_.clear();
        moeRouterWgtHost_.clear();
        expertCacheBuilt_ = false;
        expertType_ = 0;
        expertCacheSets_ = kExpertCacheSetsDefault;
        // Per-layer expert cache arenas (hybrid qwen35moe mode).  Each layer
        // owns ONE contiguous arena anchored at ExpertSlot::gateQ (upQ/downQ
        // are interior pointers; only the anchor is freed).
        for (uint32_t L = 0; L < kExpertCacheLayers; ++L) {
            if (expertCache_[L].gateQ) {
                cudaFree(expertCache_[L].gateQ);
                expertCache_[L] = ExpertSlot();
            }
            for (uint32_t s2 = 0; s2 < kExpertCacheSets; ++s2) {
                expertCacheTag_[L][s2][0] = -1;
                expertCacheTag_[L][s2][1] = -1;
                expertCacheLru_[L][s2] = 0;
            }
        }
        expertCacheLayerBytes_ = 0;
        // Free the pinned H2D staging ring (registered malloc'd pages +
        // per-buffer events).
        for (uint32_t b = 0; b < kExpertStageBufs; ++b) {
            if (expertStage_[b]) {
                cudaHostUnregister(expertStage_[b]);
                std::free(expertStage_[b]);
                expertStage_[b] = nullptr;
            }
            if (expertStageEv_[b]) {
                cudaEventDestroy(expertStageEv_[b]);
                expertStageEv_[b] = nullptr;
            }
        }
        expertStageBytes_ = 0;
        expertStageFlag_ = 0;
        destroyScratch();
        allocated_ = false;
    }

    GPUModel::~GPUModel() { destroy(); }

}// namespace tinycoder::gpu

#endif// USE_CUDA
