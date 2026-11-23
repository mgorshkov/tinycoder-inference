// Batched prefill graph for Inkling-2-Mini: m tokens in flight, [H, m].
//
// The decode graph in InklingForward.cu processes one token at a time, so every
// dense projection re-reads its weight matrix for every prompt token.  This
// path keeps m (=kInklingBatchM = 512, or less for the tail) hidden rows alive
// and hands each projection the whole batch at once:
//
//   * dense projections the int8 tensor-core GEMM accepts (Q4_K/Q5_K/Q6_K/Q8_0/
//     IQ2_S, plus IQ2_XS once wired) go through int8Gemm: one activation
//     quantization for all m rows, then one stream-k launch whose weight tile is
//     loaded once and multiplied by tile_x (64/128) token rows.  The activation
//     quantize pads each row to a multiple of 512 values, so K only has to be a
//     multiple of 128 for the y layout (2304 and 1152 both are).  Everything
//     else (IQ1_S/IQ2_XXS/IQ4_XS/NVFP4/F32/BF16 -- and the small m<8 tail) takes
//     inklingGemvBatchedAsync in 16-row tiles, which is bit-identical to one
//     inklingGemv per token;
//   * the routed MoE counting-sorts the m*nUsed (token, expert) pairs by expert
//     into a compact [nPairs] array and runs one grouped projection per expert,
//     so an expert's weight matrix is decoded once per pass instead of once per
//     pair.  The compact form is what makes the outer batch pay: at m=512 one
//     expert receives >128 of the 512 tokens on this checkpoint, so a
//     [nExperts][512] padded gather is 2.0 GiB and a capacity-128 gather has to
//     run four times, while the compact gather is m*nUsed rows (~94 MB here).
//     The down projection can take the Q2_0 int8 MMQ
//     (TINYCODER_INKLING_Q2_0_MMA) or the grouped float decoder.
//
// Everything else (RMS norm, clip, short convolution, relative-bias table,
// attention, depth mix, router, SITU, MoE accumulation) has a batched kernel
// below that reproduces the single-token kernel's arithmetic in the same order.
// The loop controller is the exception: its one-block-per-rank form lands
// 0-20 ULP off the decode kernel for a few tokens, which a later router tie
// amplifies, so the default runs one block per (rank, token) -- the decode
// kernel's body with the token as a grid dimension -- with the two RMS norms
// hoisted to one launch per token and the gain/forget mix blocked over 64
// hidden elements x 128 tokens (TINYCODER_INKLING_CTRL=batched opts back into
// the one-block-per-rank form, =token into the per-token decode launches).
//
// With all of that, a prompt is bit-identical to the token-major graph (top-1,
// top-5 and every block's hidden state) on the exact path, and bit-identical
// across fresh processes on the default one; both are what
// tools/inkling_gpu_batched_prefill_test.cpp and
// tools/inkling_gpu_forward_test.cpp gate.
//
// Two defects found by that gate, both parity-only and both fixed here, are
// worth knowing about because they are invisible to a "sensible output" check:
//
//   * the chunk's embedding rows used to be uploaded with cudaMemcpy, i.e. on
//     the legacy default stream, which has no implicit ordering with this
//     engine's cudaStreamNonBlocking stream.  A pageable-source H2D copy may
//     return with its device DMA still in flight, so the token-norm kernel could
//     read rows whose copy had not landed: rows 12..15 of a 16-token chunk came
//     out holding the PREVIOUS chunk's embeddings.  The upload is now an async
//     copy on the compute stream from a pinned ring.
//   * the batched attention picked one warp count for the whole chunk from
//     attentionWarps(pos0+m), while the decode path picks it per query from
//     attentionWarps(pos+1).  A chunk straddling a step (nKeys 128/256/512) then
//     partitions the keys, and the online-softmax combine order, differently
//     from the decode path -- which is why 192-token prompts diverged while
//     128-token ones did not.  The launch is now split into token ranges that
//     share one warp count.
//
// Layouts: every buffer is [m][...] (token-major) except the block bank, which
// is [nBlocks][m][hidden] so banking a block is one flat copy of m rows, and the
// depth-mix / message outputs, which are [query][m][hidden] so one launch writes
// all queries as contiguous token planes.
#include "Int8Gemm.hpp"
#include "gpu/InklingGPU.hpp"
#include "gpu/InklingKernels.hpp"

#include "InklingGPUImpl.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// See InklingForward.cu: Model.hpp pulls in np -> immintrin, and GCC's
// amxtileintrin.h emits builtins cudafe++ does not know.
#if defined(__CUDACC__) || defined(__NVCC__)
#define __builtin_ia32_ldtilecfg(x) ((void) 0)
#define __builtin_ia32_sttilecfg(x) ((void) 0)
#endif
#ifndef __CUDA_ARCH__
#include "Model.hpp"
#endif

namespace tinycoder::gpu {
    namespace {

        [[maybe_unused]] constexpr float kRmsEps = 1e-6f;
        constexpr uint32_t kMaxMixStates = 64;

        __device__ __forceinline__ uint32_t bMin(uint32_t a, uint32_t b) {
            return a < b ? a : b;
        }

        __device__ __forceinline__ float bWarpSum(float v) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffffu, v, off);
            return v;
        }

        // ---- depth mix ------------------------------------------------------
        //
        // One block per (token, query).  The state rows live in the block bank
        // as [nStates][m][hidden], hence `stateStride` = m*hidden and the
        // per-token offset; the appended current row is read from `cur`.
        __global__ void mixCurBatchKernel(const float *__restrict__ states,
                                          uint32_t nStates,
                                          uint32_t stateStride,
                                          const float *__restrict__ cur,
                                          uint32_t curStride,
                                          const float *__restrict__ query,
                                          uint32_t hidden,
                                          float *__restrict__ out,
                                          uint32_t outPlaneStride) {
            __shared__ float inv[kMaxMixStates];
            __shared__ float score[kMaxMixStates];
            const uint32_t t = blockIdx.y;
            const uint32_t q = blockIdx.x;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t w = threadIdx.x >> 5u;
            const uint32_t nwarps = blockDim.x >> 5u;
            const uint32_t n = nStates + 1u;
            const float *qv = query + static_cast<size_t>(q) * hidden;
            const float *ct = cur + static_cast<size_t>(t) * curStride;
            for (uint32_t st = w; st < n; st += nwarps) {
                const float *sv =
                        st < nStates
                                ? states + static_cast<size_t>(st) * stateStride +
                                          static_cast<size_t>(t) * hidden
                                : ct;
                float ss = 0.0f;
                for (uint32_t i = lane; i < hidden; i += 32u)
                    ss += sv[i] * sv[i];
                ss = bWarpSum(ss);
                const float iv = rsqrtf(ss / float(hidden) + 1e-6f);
                float dot = 0.0f;
                for (uint32_t i = lane; i < hidden; i += 32u)
                    dot += sv[i] * iv * qv[i];
                dot = bWarpSum(dot);
                if (lane == 0) {
                    inv[st] = iv;
                    score[st] = dot * rsqrtf(float(hidden));
                }
            }
            __syncthreads();
            // One lane, not the whole warp: the softmax below reads and
            // rewrites score[] in shared memory, so letting all 32 lanes run
            // it is a cross-lane read-after-write race under independent
            // thread scheduling (compute-sanitizer racecheck reports it).
            // Every lane used to write the same value, which hid it.
            if (w == 0 && lane == 0) {
                float mx = -INFINITY;
                for (uint32_t st = 0; st < n; ++st) mx = fmaxf(mx, score[st]);
                float sum = 0.0f;
                for (uint32_t st = 0; st < n; ++st) {
                    score[st] = expf(score[st] - mx);
                    sum += score[st];
                }
                for (uint32_t st = 0; st < n; ++st) score[st] /= sum;
            }
            __syncthreads();
            // State-outer, element-inner.  The old nest read the bank
            // element-major: for every output element, n state rows at a 4.7 MB
            // stride, i.e. one 32 B sector per 4 B used.  At m=512 that is
            // ~138 GB of sectors for a 512-token pass (measured 290 ms, the
            // second largest section).  Here each thread keeps kMaxK elements
            // in registers so a state row is read once, coalesced.
            //
            // The per-element sum is unchanged: the same addends in the same
            // (st ascending) order, and still (score*sv[i])*inv as two
            // multiplies, so the batched path stays bit-exact with the decode
            // kernel.
            float *dst = out + static_cast<size_t>(q) * outPlaneStride +
                         static_cast<size_t>(t) * hidden;
            constexpr uint32_t kMaxK = 9;
            for (uint32_t base = 0; base < hidden; base += kMaxK * blockDim.x) {
                float acc[kMaxK];
#pragma unroll
                for (uint32_t k = 0; k < kMaxK; ++k) acc[k] = 0.0f;
                for (uint32_t st = 0; st < n; ++st) {
                    const float *sv =
                            st < nStates
                                    ? states +
                                              static_cast<size_t>(st) * stateStride +
                                              static_cast<size_t>(t) * hidden
                                    : ct;
                    const float sc = score[st];
                    const float iv = inv[st];
#pragma unroll
                    for (uint32_t k = 0; k < kMaxK; ++k) {
                        const uint32_t i = base + threadIdx.x + k * blockDim.x;
                        if (i < hidden) acc[k] += (sc * sv[i]) * iv;
                    }
                }
#pragma unroll
                for (uint32_t k = 0; k < kMaxK; ++k) {
                    const uint32_t i = base + threadIdx.x + k * blockDim.x;
                    if (i < hidden) dst[i] = acc[k];
                }
            }
        }

        __global__ void mixBatchKernel(const float *__restrict__ states,
                                       uint32_t nStates, uint32_t stateStride,
                                       const float *__restrict__ query,
                                       uint32_t hidden, float *__restrict__ out,
                                       uint32_t outPlaneStride) {
            __shared__ float inv[kMaxMixStates];
            __shared__ float score[kMaxMixStates];
            const uint32_t t = blockIdx.y;
            const uint32_t q = blockIdx.x;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t w = threadIdx.x >> 5u;
            const uint32_t nwarps = blockDim.x >> 5u;
            const float *qv = query + static_cast<size_t>(q) * hidden;
            for (uint32_t st = w; st < nStates; st += nwarps) {
                const float *sv = states + static_cast<size_t>(st) * stateStride +
                                  static_cast<size_t>(t) * hidden;
                float ss = 0.0f;
                for (uint32_t i = lane; i < hidden; i += 32u)
                    ss += sv[i] * sv[i];
                ss = bWarpSum(ss);
                const float iv = rsqrtf(ss / float(hidden) + 1e-6f);
                float dot = 0.0f;
                for (uint32_t i = lane; i < hidden; i += 32u)
                    dot += sv[i] * iv * qv[i];
                dot = bWarpSum(dot);
                if (lane == 0) {
                    inv[st] = iv;
                    score[st] = dot * rsqrtf(float(hidden));
                }
            }
            __syncthreads();
            // One lane, not the whole warp: the softmax below reads and
            // rewrites score[] in shared memory, so letting all 32 lanes run
            // it is a cross-lane read-after-write race under independent
            // thread scheduling (compute-sanitizer racecheck reports it).
            // Every lane used to write the same value, which hid it.
            if (w == 0 && lane == 0) {
                float mx = -INFINITY;
                for (uint32_t st = 0; st < nStates; ++st)
                    mx = fmaxf(mx, score[st]);
                float sum = 0.0f;
                for (uint32_t st = 0; st < nStates; ++st) {
                    score[st] = expf(score[st] - mx);
                    sum += score[st];
                }
                for (uint32_t st = 0; st < nStates; ++st) score[st] /= sum;
            }
            __syncthreads();
            // Same state-outer reorder as mixCurBatchKernel, and the same
            // per-element arithmetic order.
            float *dst = out + static_cast<size_t>(q) * outPlaneStride +
                         static_cast<size_t>(t) * hidden;
            constexpr uint32_t kMaxK = 9;
            for (uint32_t base = 0; base < hidden; base += kMaxK * blockDim.x) {
                float acc[kMaxK];
#pragma unroll
                for (uint32_t k = 0; k < kMaxK; ++k) acc[k] = 0.0f;
                for (uint32_t st = 0; st < nStates; ++st) {
                    const float *sv =
                            states + static_cast<size_t>(st) * stateStride +
                            static_cast<size_t>(t) * hidden;
                    const float sc = score[st];
                    const float iv = inv[st];
#pragma unroll
                    for (uint32_t k = 0; k < kMaxK; ++k) {
                        const uint32_t i = base + threadIdx.x + k * blockDim.x;
                        if (i < hidden) acc[k] += (sc * sv[i]) * iv;
                    }
                }
#pragma unroll
                for (uint32_t k = 0; k < kMaxK; ++k) {
                    const uint32_t i = base + threadIdx.x + k * blockDim.x;
                    if (i < hidden) dst[i] = acc[k];
                }
            }
        }

        // ---- loop control ---------------------------------------------------

        __device__ __forceinline__ float rmsInvShared(const float *x, uint32_t n,
                                                      float *s) {
            // Barrier before the first write: controlDownExact/Batch call this
            // twice back to back over the SAME shared buffer, so without it the
            // second call's `s[threadIdx.x]` (thread 0 first) can overwrite
            // s[0] while a slower warp is still reading the first call's
            // reduction.  That is the same race the token path's copy in
            // InklingElementwise.cu documents and guards; this batched copy was
            // added without the guard, and compute-sanitizer racecheck reports
            // 2048 hazards in controlDownExactKernel because of it.
            __syncthreads();
            double ss = 0.0;
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
                ss += double(x[i]) * x[i];
            s[threadIdx.x] = static_cast<float>(ss);
            __syncthreads();
            for (uint32_t k = 128; k > 0; k >>= 1) {
                if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
                __syncthreads();
            }
            return rsqrtf(s[0] / float(n) + 1e-6f);
        }

        /// Batched loop-controller down projection, bit-exact with
        /// controlDownKernel: one block per rank row holds the weight row in
        /// registers and walks the m tokens in order, so every token's dot is
        /// summed in exactly the token kernel's order (256 thread partials over
        /// j = tid, tid+256, ..., then the same shared-memory tree).  The two
        /// RMS norms are the same rmsInvShared reduction the token kernel does,
        /// so g[t][r] is bit-identical to the decode path's dCtrlG.
        __global__ void controlDownBatchKernel(const float *__restrict__ cur,
                                               uint32_t curStride,
                                               const float *__restrict__ msg,
                                               uint32_t msgStride,
                                               const float *__restrict__ w,
                                               float *__restrict__ g, uint32_t rank,
                                               uint32_t hidden, uint32_t m) {
            __shared__ float s[256];
            __shared__ float inv0[kInklingBatchM];
            __shared__ float inv1[kInklingBatchM];
            const uint32_t r = blockIdx.x;
            for (uint32_t t = 0; t < m; ++t) {
                __syncthreads();
                inv0[t] = rmsInvShared(cur + static_cast<size_t>(t) * curStride,
                                       hidden, s);
                __syncthreads();
                inv1[t] = rmsInvShared(msg + static_cast<size_t>(t) * msgStride,
                                       hidden, s);
                // The next write to s[] must not race the s[0] read above.
                __syncthreads();
            }
            const uint32_t n2 = 2 * hidden;
            const float *wr = w + static_cast<size_t>(r) * n2;
            // Read the weight row through the same `wr[j] * v` expression the
            // decode kernel uses: the identity of the FFMA contraction matters
            // at the ULP level, and a register preload with a running index
            // compiles differently.
            for (uint32_t t = 0; t < m; ++t) {
                const float *c = cur + static_cast<size_t>(t) * curStride;
                const float *mg = msg + static_cast<size_t>(t) * msgStride;
                float acc = 0.0f;
                for (uint32_t j = threadIdx.x; j < n2; j += blockDim.x) {
                    const float v = (j < hidden)
                                            ? c[j] * inv0[t]
                                            : mg[j - hidden] * inv1[t];
                    acc += wr[j] * v;
                }
                s[threadIdx.x] = acc;
                __syncthreads();
                for (uint32_t q = 128; q > 0; q >>= 1) {
                    if (threadIdx.x < q)
                        s[threadIdx.x] += s[threadIdx.x + q];
                    __syncthreads();
                }
                if (threadIdx.x == 0)
                    g[static_cast<size_t>(t) * rank + r] = s[0];
                __syncthreads();// next token reuses s[]
            }
        }

        /// Exact batched controller: the decode kernel's body with one block per
        /// (rank, token) instead of a per-token launch, so its arithmetic is the
        /// decode kernel's by construction (same block size, same thread
        /// mapping, same expressions).  The one-block-per-rank form below loops
        /// the tokens in-kernel, which nvcc contracts a few ULP differently --
        /// enough that a router tie can flip.  Here the token is blockIdx.y.
        /// The controller's two RMS norms, once per token instead of once per
        /// (rank, token).  controlDownExactKernel is one block per (rank, token)
        /// to reproduce the decode kernel's dot order, so it recomputed the same
        /// two norms `rank` = 64 times per token -- 290 ms of a 512-token pass,
        /// 19% of prefill.  rmsInvShared is deterministic for a given row and
        /// block size (same fp64 accumulation, same shared tree), so hoisting it
        /// out is bit-identical.
        [[maybe_unused]]
        __global__ void controlInvKernel(const float *__restrict__ cur,
                                         const float *__restrict__ msg,
                                         uint32_t hidden, float *__restrict__ inv) {
            __shared__ float s[256];
            const uint32_t t = blockIdx.x;
            const float i0 = rmsInvShared(cur + static_cast<size_t>(t) * hidden,
                                          hidden, s);
            const float i1 = rmsInvShared(msg + static_cast<size_t>(t) * hidden,
                                          hidden, s);
            if (threadIdx.x == 0) {
                inv[2u * t] = i0;
                inv[2u * t + 1u] = i1;
            }
        }

        [[maybe_unused]]
        __global__ void controlDownExactKernel(const float *__restrict__ cur,
                                               uint32_t curStride,
                                               const float *__restrict__ msg,
                                               uint32_t msgStride,
                                               const float *__restrict__ w,
                                               float *__restrict__ g,
                                               uint32_t rank, uint32_t hidden,
                                               const float *__restrict__ inv) {
            __shared__ float s[256];
            const uint32_t r = blockIdx.x;
            const uint32_t t = blockIdx.y;
            const float *c = cur + static_cast<size_t>(t) * curStride;
            const float *m = msg + static_cast<size_t>(t) * msgStride;
            const float inv0 = inv[2u * t];
            const float inv1 = inv[2u * t + 1u];
            const uint32_t n2 = 2 * hidden;
            const float *wr = w + static_cast<size_t>(r) * n2;
            float acc = 0.0f;
            for (uint32_t j = threadIdx.x; j < n2; j += blockDim.x) {
                const float v = (j < hidden) ? c[j] * inv0 : m[j - hidden] * inv1;
                acc += wr[j] * v;
            }
            __syncthreads();
            s[threadIdx.x] = acc;
            __syncthreads();
            for (uint32_t k = 128; k > 0; k >>= 1) {
                if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
                __syncthreads();
            }
            if (threadIdx.x == 0)
                g[static_cast<size_t>(t) * rank + r] = s[0];
        }

        /// Controller down projection, warp-per-rank and weight-stationary.
        ///
        /// controlDownExactKernel reproduces the decode kernel's reduction -- a
        /// 256-thread strided partial sum followed by a specific binary tree --
        /// and that tree is why a generic GEMV cannot replace it.  But the tree
        /// does not need 256 threads: one warp holds the same 256 partials as
        /// eight per lane (partial `lane + 32*jj`), so the tree is
        /// p[jj] += p[jj+4], p[jj] += p[jj+2], p[0] += p[1], then four shuffle
        /// steps -- bit-identical, with no shared memory and no block barrier.
        /// With one rank row per warp the row lives in registers for the whole
        /// block instead of being read once per token: the exact form read
        /// rank(64) x 2H(4608) x 512 tokens x 4 B = 604 MB per launch (measured
        /// 0.85 ms, 96 launches = 82 ms of a 512-token pass); this reads one
        /// row per rank per 32-token tile, 19 MB for the launch.
        [[maybe_unused]]
        __global__ void __launch_bounds__(256, 1)
                controlDownWarpKernel(const float *__restrict__ cur,
                                      uint32_t curStride,
                                      const float *__restrict__ msg,
                                      uint32_t msgStride,
                                      const float *__restrict__ w,
                                      float *__restrict__ g, uint32_t rank,
                                      uint32_t hidden,
                                      const float *__restrict__ inv, uint32_t m) {
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5u;
            const uint32_t r = blockIdx.x * (blockDim.x >> 5u) + warp;
            if (r >= rank) return;
            constexpr uint32_t PER = 18u;// n2 = 4608 for a 2304-wide model
            const uint32_t n2 = 2u * hidden;
            if (n2 != PER * 256u) return;// the wrapper falls back to the exact form
            float wv[PER][8];
            const float *wr = w + static_cast<size_t>(r) * n2;
#pragma unroll
            for (uint32_t k = 0; k < PER; ++k) {
#pragma unroll
                for (uint32_t jj = 0; jj < 8u; ++jj)
                    wv[k][jj] = wr[lane + 32u * jj + 256u * k];
            }
            const uint32_t t0 = blockIdx.y * 32u;
            const uint32_t t1 = t0 + 32u < m ? t0 + 32u : m;
            for (uint32_t t = t0; t < t1; ++t) {
                const float *c = cur + static_cast<size_t>(t) * curStride;
                const float *mg = msg + static_cast<size_t>(t) * msgStride;
                const float inv0 = inv[2u * t];
                const float inv1 = inv[2u * t + 1u];
                float p[8] = {};
#pragma unroll
                for (uint32_t k = 0; k < PER; ++k) {
                    constexpr uint32_t HID = 2304u;
                    const uint32_t jb = 256u * k;
#pragma unroll
                    for (uint32_t jj = 0; jj < 8u; ++jj) {
                        const uint32_t j = lane + 32u * jj + jb;
                        const float v = j < HID ? c[j] * inv0
                                                : mg[j - HID] * inv1;
                        p[jj] = fmaf(wv[k][jj], v, p[jj]);
                    }
                }
                // The 256-leaf tree of the decode kernel, levels 128/64/32
                // folded in registers, levels 16..1 by shuffle.
                p[0] += p[4];
                p[1] += p[5];
                p[2] += p[6];
                p[3] += p[7];
                p[0] += p[2];
                p[1] += p[3];
                p[0] += p[1];
                float s = __shfl_down_sync(0xffffffffu, p[0], 16);
                if (lane < 16u) p[0] += s;
                s = __shfl_down_sync(0xffffffffu, p[0], 8);
                if (lane < 8u) p[0] += s;
                s = __shfl_down_sync(0xffffffffu, p[0], 4);
                if (lane < 4u) p[0] += s;
                s = __shfl_down_sync(0xffffffffu, p[0], 2);
                if (lane < 2u) p[0] += s;
                s = __shfl_down_sync(0xffffffffu, p[0], 1);
                if (lane < 1u) p[0] += s;
                if (lane == 0u) g[static_cast<size_t>(t) * rank + r] = p[0];
            }
        }

        [[maybe_unused]]
        __global__ void controlMixExactKernel(const float *__restrict__ cur,
                                              uint32_t curStride,
                                              const float *__restrict__ msg,
                                              uint32_t msgStride,
                                              const float *__restrict__ gainW,
                                              const float *__restrict__ forgetW,
                                              const float *__restrict__ g,
                                              float *__restrict__ out,
                                              uint32_t hidden, uint32_t rank) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= hidden) return;
            const uint32_t t = blockIdx.y;
            const float *c = cur + static_cast<size_t>(t) * curStride;
            const float *m = msg + static_cast<size_t>(t) * msgStride;
            const float *gr = gainW + static_cast<size_t>(i) * rank;
            const float *fr = forgetW + static_cast<size_t>(i) * rank;
            const float *gt = g + static_cast<size_t>(t) * rank;
            float gv = 0.0f, fv = 0.0f;
            for (uint32_t j = 0; j < rank; ++j) {
                gv += gr[j] * gt[j];
                fv += fr[j] * gt[j];
            }
            out[static_cast<size_t>(t) * curStride + i] =
                    c[i] - tanhf(fv) * c[i] + tanhf(gv) * m[i];
        }

        /// Blocked form of controlMixExactKernel, which was 265 ms of a
        /// 512-token pass (18% of prefill): one block per (hidden element,
        /// token) reads the [hidden][rank] gain/forget rows with a 4 B access
        /// per 256 B row, so every 32 B sector carries 4 B of useful data and
        /// every token re-reads the whole matrix.  Here one block owns 64
        /// hidden elements and 128 tokens: the 1.2 MB of weights is staged
        /// once, coalesced, into padded shared (stride 65 makes the per-thread
        /// row reads conflict-free), and a warp covers 32 hidden elements for
        /// one token, so gt[j] is a shared broadcast.
        ///
        /// The per-element arithmetic is identical -- j ascending, gv then fv,
        /// the same `c - tanh(fv)*c + tanh(gv)*m` expression -- so this stays
        /// bit-exact with controlMixKernel2 and the decode graph.
        [[maybe_unused]]
        __global__ void controlMixBlockedKernel(const float *__restrict__ cur,
                                                uint32_t curStride,
                                                const float *__restrict__ msg,
                                                uint32_t msgStride,
                                                const float *__restrict__ gainW,
                                                const float *__restrict__ forgetW,
                                                const float *__restrict__ g,
                                                float *__restrict__ out,
                                                uint32_t hidden, uint32_t rank,
                                                uint32_t m) {
            constexpr uint32_t NI = 64;
            constexpr uint32_t NT = 128;
            __shared__ float sG[NI][65];
            __shared__ float sF[NI][65];
            const uint32_t i0 = blockIdx.x * NI;
            const uint32_t t0 = blockIdx.y * NT;
            for (uint32_t idx = threadIdx.x; idx < NI * rank; idx += blockDim.x) {
                const uint32_t i = idx / rank;
                const uint32_t j = idx - i * rank;
                sG[i][j] = gainW[(static_cast<size_t>(i0) + i) * rank + j];
                sF[i][j] = forgetW[(static_cast<size_t>(i0) + i) * rank + j];
            }
            __syncthreads();
            const uint32_t iL = threadIdx.x & (NI - 1u);
            const uint32_t tStride = blockDim.x / NI;
            for (uint32_t t = t0 + (threadIdx.x / NI); t < t0 + NT && t < m;
                 t += tStride) {
                const float *c = cur + static_cast<size_t>(t) * curStride;
                const float *mg = msg + static_cast<size_t>(t) * msgStride;
                const float *gt = g + static_cast<size_t>(t) * rank;
                float gv = 0.0f, fv = 0.0f;
                for (uint32_t j = 0; j < rank; ++j) {
                    gv += sG[iL][j] * gt[j];
                    fv += sF[iL][j] * gt[j];
                }
                const uint32_t i = i0 + iL;
                if (i < hidden)
                    out[static_cast<size_t>(t) * curStride + i] =
                            c[i] - tanhf(fv) * c[i] + tanhf(gv) * mg[i];
            }
        }

        /// Controller mix, weight-stationary per hidden element.  The blocked
        /// form above stages 64 hidden elements' weights in shared and every
        /// token re-reads them: at rank 64 that is 128 shared loads per element
        /// per token, which is what kept the section at 23 ms of a 512-token
        /// pass.  Here one thread owns one hidden element, keeps its 64 gain
        /// and 64 forget weights in registers for the whole block, and only the
        /// per-token rank vector comes from shared.  The two dots keep the same
        /// ascending-j order and the same `c - tanh(fv)*c + tanh(gv)*m`
        /// expression, so the result is bit-identical.
        [[maybe_unused]]
        __global__ void __launch_bounds__(256, 1)
                controlMixRegKernel(const float *__restrict__ cur,
                                    uint32_t curStride,
                                    const float *__restrict__ msg,
                                    uint32_t msgStride,
                                    const float *__restrict__ gainW,
                                    const float *__restrict__ forgetW,
                                    const float *__restrict__ g,
                                    float *__restrict__ out, uint32_t hidden,
                                    uint32_t rank, uint32_t m) {
            constexpr uint32_t R = 64u;
            constexpr uint32_t TT = 128u;
            extern __shared__ float sG[];// [TT][R]
            const uint32_t t0 = blockIdx.y * TT;
            const uint32_t nT = (t0 + TT < m ? t0 + TT : m) - t0;
            for (uint32_t idx = threadIdx.x; idx < nT * R; idx += blockDim.x) {
                const uint32_t tt = idx / R;
                const uint32_t j = idx - tt * R;
                sG[idx] = g[(static_cast<size_t>(t0) + tt) * rank + j];
            }
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            float gw[R], fw[R];
            if (i < hidden) {
                const float *gp = gainW + static_cast<size_t>(i) * rank;
                const float *fp = forgetW + static_cast<size_t>(i) * rank;
#pragma unroll
                for (uint32_t j = 0; j < R; ++j) {
                    gw[j] = gp[j];
                    fw[j] = fp[j];
                }
            }
            __syncthreads();
            if (i >= hidden) return;
            for (uint32_t tt = 0; tt < nT; ++tt) {
                const float *gt = sG + tt * R;
                float gv = 0.0f, fv = 0.0f;
#pragma unroll
                for (uint32_t j = 0; j < R; ++j) {
                    gv = fmaf(gw[j], gt[j], gv);
                    fv = fmaf(fw[j], gt[j], fv);
                }
                const uint32_t t = t0 + tt;
                const float c = cur[static_cast<size_t>(t) * curStride + i];
                const float mg = msg[static_cast<size_t>(t) * msgStride + i];
                out[static_cast<size_t>(t) * curStride + i] =
                        c - tanhf(fv) * c + tanhf(gv) * mg;
            }
        }

        /// Batched gain/forget + mix, bit-exact with controlMixKernel2: one
        /// thread per hidden element, the rank-64 dots accumulated in the same
        /// sequential order.
        __global__ void controlMixBatchKernel(const float *__restrict__ cur,
                                              uint32_t curStride,
                                              const float *__restrict__ msg,
                                              uint32_t msgStride,
                                              const float *__restrict__ gainW,
                                              const float *__restrict__ forgetW,
                                              const float *__restrict__ g,
                                              float *__restrict__ out,
                                              uint32_t hidden, uint32_t rank) {
            const uint32_t t = blockIdx.y;
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= hidden) return;
            const float *c = cur + static_cast<size_t>(t) * curStride;
            const float *m = msg + static_cast<size_t>(t) * msgStride;
            const float *gr = gainW + static_cast<size_t>(i) * rank;
            const float *fr = forgetW + static_cast<size_t>(i) * rank;
            const float *gt = g + static_cast<size_t>(t) * rank;
            float gv = 0.0f, fv = 0.0f;
            for (uint32_t j = 0; j < rank; ++j) {
                gv += gr[j] * gt[j];
                fv += fr[j] * gt[j];
            }
            out[static_cast<size_t>(t) * curStride + i] =
                    c[i] - tanhf(fv) * c[i] + tanhf(gv) * m[i];
        }

        // ---- clip -----------------------------------------------------------

        __global__ void clipBatchKernel(float *x, uint32_t rows, uint32_t len,
                                        const float *scale, float threshold) {
            __shared__ float s[256];
            const uint32_t r = blockIdx.x % rows;
            const uint32_t t = blockIdx.x / rows;
            float *row = x + (static_cast<size_t>(t) * rows + r) * len;
            double ss = 0.0;
            if (scale != nullptr) {
                const float *sc = scale + static_cast<size_t>(r) * len;
                for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
                    row[i] *= sc[i];
                    ss += double(row[i]) * row[i];
                }
            } else {
                for (uint32_t i = threadIdx.x; i < len; i += blockDim.x)
                    ss += double(row[i]) * row[i];
            }
            s[threadIdx.x] = static_cast<float>(ss);
            __syncthreads();
            for (uint32_t k = 128; k > 0; k >>= 1) {
                if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
                __syncthreads();
            }
            const float norm = sqrtf(s[0] + 1e-6f);
            const float f = fminf(1.0f, threshold / norm);
            for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) row[i] *= f;
        }

        // ---- relative bias table --------------------------------------------

        /// Fallback for a relative width above one warp (32): the original
        /// one-block-per-(token, head) scatter, kept for shape safety only.
        __global__ void relBiasScatterKernel(const float *__restrict__ r,
                                             const float *__restrict__ bias,
                                             const float *__restrict__ sink,
                                             float *__restrict__ table,
                                             uint32_t heads, uint32_t nRel,
                                             uint32_t extent) {
            const uint32_t h = blockIdx.x % heads;
            const uint32_t t = blockIdx.x / heads;
            const float *rh = r + (static_cast<size_t>(t) * heads + h) * nRel;
            const float sk = sink[static_cast<size_t>(t) * heads + h];
            float *tb = table +
                        (static_cast<size_t>(t) * heads + h) * (extent + 1);
            for (uint32_t d = threadIdx.x; d <= extent; d += blockDim.x) {
                float v = 0.0f;
                if (d < extent) {
                    const float *br = bias + static_cast<size_t>(d) * nRel;
                    for (uint32_t k = 0; k < nRel; ++k) v += br[k] * rh[k];
                    v -= sk;
                }
                tb[d] = v;
            }
        }

        /// Relative-bias table, bias-stationary.
        ///
        /// table[t][h][d] = sum_k r[t][h][k]*bias[d][k] - sink[t][h] (d <
        /// extent), 0 at d == extent.  The scatter form above launches one block
        /// per (token, head) and every block streams the whole [extent][nRel]
        /// bias matrix: at m=512, heads=24, extent=1024 that is 12288 blocks x
        /// 64 KiB = 786 MB of bias reads per layer, 37 GB per 512-token pass.
        /// It measured 78 ms of the pass, all of it redundant.
        ///
        /// Here one block stages a bias tile once into shared, transposed so a
        /// warp's consecutive-distance reads hit distinct banks, and one warp
        /// then walks whole (token, head) rows out of it.  Two further cuts:
        /// the tile is capped at the largest distance this pass can still read
        /// (pos0+m-1; for a fresh 512-token context that halves the row), and
        /// each lane accumulates four distances with the same ascending-k FMA
        /// chain and the same trailing `- sink` as the token path, so the table
        /// is bit-identical.
        template <uint32_t DT, uint32_t TT>
        __global__ void relBiasTableKernel(const float *__restrict__ r,
                                           const float *__restrict__ bias,
                                           const float *__restrict__ sink,
                                           float *__restrict__ table,
                                           uint32_t heads, uint32_t nRel,
                                           uint32_t extent, uint32_t dCount,
                                           uint32_t m) {
            extern __shared__ float sh[];
            float *sB = sh;// [nRel][DT + 1], transposed
            const uint32_t d0 = blockIdx.x * DT;
            const uint32_t t0 = blockIdx.y * TT;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t w = threadIdx.x >> 5u;
            const uint32_t nw = blockDim.x >> 5u;
            for (uint32_t idx = threadIdx.x; idx < nRel * DT;
                 idx += blockDim.x) {
                const uint32_t k = idx / DT;
                const uint32_t d = idx - k * DT;
                const uint32_t dg = d0 + d;
                sB[k * (DT + 1u) + d] =
                        dg < dCount
                                ? bias[static_cast<size_t>(dg) * nRel + k]
                                : 0.0f;
            }
            __syncthreads();
            const uint32_t rows = TT * heads;
            for (uint32_t row = w; row < rows; row += nw) {
                const uint32_t tt = row / heads;
                const uint32_t h = row - tt * heads;
                const uint32_t t = t0 + tt;
                if (t >= m) continue;
                const float *rh =
                        r + (static_cast<size_t>(t) * heads + h) * nRel;
                const float rv = lane < nRel ? rh[lane] : 0.0f;
                const float sk = sink[static_cast<size_t>(t) * heads + h];
                float *tb = table +
                            (static_cast<size_t>(t) * heads + h) * (extent + 1);
                for (uint32_t dc = 0; dc < DT; dc += 128u) {
                    const uint32_t dl = dc + lane * 4u;
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                    for (uint32_t k = 0; k < 32u; ++k) {
                        if (k >= nRel) break;
                        const float rk = __shfl_sync(0xffffffffu, rv, k);
                        const float *brow = sB + k * (DT + 1u) + dl;
                        a0 = fmaf(rk, brow[0], a0);
                        a1 = fmaf(rk, brow[1], a1);
                        a2 = fmaf(rk, brow[2], a2);
                        a3 = fmaf(rk, brow[3], a3);
                    }
                    const uint32_t d = d0 + dl;
                    if (d + 3u < dCount && d + 3u < extent) {
                        tb[d] = a0 - sk;
                        tb[d + 1u] = a1 - sk;
                        tb[d + 2u] = a2 - sk;
                        tb[d + 3u] = a3 - sk;
                    } else {
                        const float av[4] = {a0, a1, a2, a3};
                        for (uint32_t j = 0; j < 4u; ++j) {
                            const uint32_t dd = d + j;
                            if (dd >= dCount) continue;
                            tb[dd] = dd == extent ? 0.0f : av[j] - sk;
                        }
                    }
                }
            }
        }

        // ---- K/V store + causal multi-query attention ------------------------

        __global__ void storeKvBatchKernel(const float *__restrict__ kcur,
                                           const float *__restrict__ vcur,
                                           float *__restrict__ kCache,
                                           float *__restrict__ vCache,
                                           uint32_t pos0, uint32_t kvDim) {
            const uint32_t t = blockIdx.y;
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= kvDim) return;
            const size_t off = static_cast<size_t>(pos0 + t) * kvDim + i;
            kCache[off] = kcur[static_cast<size_t>(t) * kvDim + i];
            vCache[off] = vcur[static_cast<size_t>(t) * kvDim + i];
        }

        /// One block per (head, query token), W warps splitting the key range:
        /// the single-token kernel with the query's own position.  Causality is
        /// the key range itself -- query t attends [firstKey, pos0+t], and the
        /// store kernel above has already written every key up to pos0+m-1, so
        /// no explicit mask is needed and no future key is in range.
        template <uint32_t W>
        __global__ void attentionBatchKernel(
                const float *__restrict__ q, const float *__restrict__ kCache,
                const float *__restrict__ vCache,
                const float *__restrict__ table, uint32_t pos0, uint32_t nHeads,
                uint32_t nKvHeads, uint32_t extent, float qkScale,
                float *__restrict__ out, uint32_t window, uint32_t isSwa,
                uint32_t tokOffset) {
            const uint32_t t = tokOffset + blockIdx.y;
            const uint32_t pos = pos0 + t;
            const uint32_t firstKey =
                    (isSwa && window > 0 && pos + 1 > window) ? pos + 1 - window
                                                              : 0;
            const uint32_t nKeys = pos + 1 - firstKey;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t w = threadIdx.x >> 5u;
            const uint32_t hq = blockIdx.x;
            if (hq >= nHeads) return;
            const uint32_t group = nHeads / nKvHeads;
            const uint32_t hkv = hq / group;
            const uint32_t qDim = nHeads * 128u;
            const float *qh = q + static_cast<size_t>(t) * qDim +
                              static_cast<size_t>(hq) * 128u;
            const float q0 = qh[lane], q1 = qh[lane + 32], q2 = qh[lane + 64],
                        q3 = qh[lane + 96];
            const uint32_t strideK = nKvHeads * 128u;
            const float *kb = kCache + static_cast<size_t>(hkv) * 128u + lane;
            const float *vb = vCache + static_cast<size_t>(hkv) * 128u + lane;
            const float *trow =
                    table + (static_cast<size_t>(t) * nHeads + hq) * (extent + 1u);

            extern __shared__ float smem[];
            float *sOut = smem;                     // [W][128]
            float *trowS = smem + W * 128u;         // [extent + 1]
            __shared__ float sMax[W];
            __shared__ float sSum[W];

            for (uint32_t i = threadIdx.x; i <= extent; i += 32u * W)
                trowS[i] = trow[i];
            __syncthreads();

            const uint32_t k0 = (nKeys * w) / W;
            const uint32_t k1 = (nKeys * (w + 1u)) / W;
            float mx = -INFINITY;
            float sum = 0.0f;
            float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
            for (uint32_t ki = k0; ki < k1; ++ki) {
                const uint32_t kp = firstKey + ki;
                const float *kh = kb + static_cast<size_t>(kp) * strideK;
                float d = q0 * kh[0] + q1 * kh[32] + q2 * kh[64] + q3 * kh[96];
                d = bWarpSum(d);
                const uint32_t dist = bMin(pos - kp, extent);
                const float logit = d * qkScale + trowS[dist];
                float p;
                if (logit > mx) {
                    const float corr = expf(mx - logit);
                    sum = sum * corr + 1.0f;
                    o0 *= corr;
                    o1 *= corr;
                    o2 *= corr;
                    o3 *= corr;
                    mx = logit;
                    p = 1.0f;
                } else {
                    p = expf(logit - mx);
                    sum += p;
                }
                const float *vh = vb + static_cast<size_t>(kp) * strideK;
                o0 += p * vh[0];
                o1 += p * vh[32];
                o2 += p * vh[64];
                o3 += p * vh[96];
            }
            if (lane == 0) {
                sMax[w] = mx;
                sSum[w] = sum;
            }
            sOut[w * 128u + lane] = o0;
            sOut[w * 128u + lane + 32u] = o1;
            sOut[w * 128u + lane + 64u] = o2;
            sOut[w * 128u + lane + 96u] = o3;
            __syncthreads();

            if (w == 0) {
                float M = 0.0f;// sink logit
                for (uint32_t j = 0; j < W; ++j) M = fmaxf(M, sMax[j]);
                float gsum = expf(0.0f - M);
                for (uint32_t j = 0; j < W; ++j) {
                    if (sSum[j] > 0.0f)
                        gsum += sSum[j] * expf(sMax[j] - M);
                }
                const float inv = 1.0f / gsum;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (uint32_t j = 0; j < W; ++j) {
                    if (sSum[j] > 0.0f) {
                        const float sc = expf(sMax[j] - M) * inv;
                        a0 += sc * sOut[j * 128u + lane];
                        a1 += sc * sOut[j * 128u + lane + 32u];
                        a2 += sc * sOut[j * 128u + lane + 64u];
                        a3 += sc * sOut[j * 128u + lane + 96u];
                    }
                }
                float *oh = out + static_cast<size_t>(t) * qDim +
                            static_cast<size_t>(hq) * 128u;
                oh[lane] = a0;
                oh[lane + 32] = a1;
                oh[lane + 64] = a2;
                oh[lane + 96] = a3;
            }
        }

        /// Query-tiled attention: the per-token kernel above, with T query
        /// tokens sharing one block and one K/V read.
        ///
        /// The per-token kernel gives every (head, token) its own block, so each
        /// block streams the whole [nKeys][128] K and V slice of its kv head.
        /// At m=128 that is 3072 blocks x nKeys x 1 KB of K/V per layer, which
        /// is the section's cost: the 8-token tile below reuses one K/V read for
        /// 8 tokens, and stages the relative-bias rows once instead of once per
        /// token.
        ///
        /// Arithmetic is unchanged per token: warp w still owns the key slice
        /// [nKeys*w/W, nKeys*(w+1)/W) of every token in the tile (the same
        /// attentionWarps() partition and the same sequential key order), and
        /// the online-softmax combine still runs over w = 0..W-1 with warp 0's
        /// sink term, so the result is bit-identical to the token-major path.
        /// The union key range is walked once and each token's own slice bounds
        /// filter it; those bounds are warp-uniform, so the predication never
        /// splits a warp.
        template <uint32_t W>
        __global__ void attentionTileKernel(
                const float *__restrict__ q, const float *__restrict__ kCache,
                const float *__restrict__ vCache,
                const float *__restrict__ table, uint32_t pos0, uint32_t nHeads,
                uint32_t nKvHeads, uint32_t extent, float qkScale,
                float *__restrict__ out, uint32_t window, uint32_t isSwa,
                uint32_t tokOffset, uint32_t tEnd) {
            constexpr uint32_t T = 8;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t w = threadIdx.x >> 5u;
            const uint32_t hq = blockIdx.x;
            if (hq >= nHeads) return;
            const uint32_t group = nHeads / nKvHeads;
            const uint32_t hkv = hq / group;
            const uint32_t qDim = nHeads * 128u;
            const uint32_t strideK = nKvHeads * 128u;
            const uint32_t tBase = tokOffset + blockIdx.y * T;
            const size_t tStride = static_cast<size_t>(extent + 1u);

            __shared__ float sMax[T][W];
            __shared__ float sSum[T][W];
            __shared__ float sOut[T][W][128];

            float qv[T][4];
            float ov[T][4];
            float mx[T], sm[T];
            uint32_t fk[T], ki0[T], ki1[T], posv[T];
            uint32_t kpLo = 0xFFFFFFFFu, kpHi = 0u;
#pragma unroll
            for (uint32_t i = 0; i < T; ++i) {
                // Clamp instead of masking: an out-of-range slot recomputes the
                // last valid token (its write is skipped).  A branch on i keeps
                // nvcc from unrolling and drops these arrays into local memory,
                // which costs more than the duplicated work.
                const uint32_t t = bMin(tBase + i, tEnd - 1u);
                const uint32_t pos = pos0 + t;
                const uint32_t firstKey =
                        (isSwa != 0u && window > 0u && pos + 1u > window)
                                ? pos + 1u - window
                                : 0u;
                const uint32_t nKeys = pos + 1u - firstKey;
                const uint32_t k0 = (nKeys * w) / W;
                const uint32_t k1 = (nKeys * (w + 1u)) / W;
                posv[i] = pos;
                fk[i] = firstKey;
                ki0[i] = k0;
                ki1[i] = k1;
                mx[i] = -INFINITY;
                sm[i] = 0.0f;
                kpLo = bMin(kpLo, firstKey + k0);
                kpHi = kpHi > firstKey + k1 ? kpHi : firstKey + k1;
                const float *qh = q + static_cast<size_t>(t) * qDim +
                                  static_cast<size_t>(hq) * 128u;
                qv[i][0] = qh[lane];
                qv[i][1] = qh[lane + 32u];
                qv[i][2] = qh[lane + 64u];
                qv[i][3] = qh[lane + 96u];
                ov[i][0] = 0.0f;
                ov[i][1] = 0.0f;
                ov[i][2] = 0.0f;
                ov[i][3] = 0.0f;
            }

            const float *kb = kCache + static_cast<size_t>(hkv) * 128u + lane;
            const float *vb = vCache + static_cast<size_t>(hkv) * 128u + lane;
            for (uint32_t kp = kpLo; kp < kpHi; ++kp) {
                const float *kh = kb + static_cast<size_t>(kp) * strideK;
                const float k0v = kh[0], k1v = kh[32], k2v = kh[64], k3v = kh[96];
                const float *vh = vb + static_cast<size_t>(kp) * strideK;
                const float v0 = vh[0], v1 = vh[32], v2 = vh[64], v3 = vh[96];
#pragma unroll
                for (uint32_t i = 0; i < T; ++i) {
                    const uint32_t ki = kp - fk[i];
                    if (ki < ki0[i] || ki >= ki1[i]) continue;
                    float d = qv[i][0] * k0v + qv[i][1] * k1v +
                              qv[i][2] * k2v + qv[i][3] * k3v;
                    d = bWarpSum(d);
                    const uint32_t dist = bMin(posv[i] - kp, extent);
                    const float logit =
                            d * qkScale +
                            table[(static_cast<size_t>(tBase + i) * nHeads + hq) *
                                          tStride +
                                  dist];
                    float p;
                    if (logit > mx[i]) {
                        const float corr = expf(mx[i] - logit);
                        sm[i] = sm[i] * corr + 1.0f;
                        ov[i][0] *= corr;
                        ov[i][1] *= corr;
                        ov[i][2] *= corr;
                        ov[i][3] *= corr;
                        mx[i] = logit;
                        p = 1.0f;
                    } else {
                        p = expf(logit - mx[i]);
                        sm[i] += p;
                    }
                    ov[i][0] += p * v0;
                    ov[i][1] += p * v1;
                    ov[i][2] += p * v2;
                    ov[i][3] += p * v3;
                }
            }

            // Every warp writes its own slice of every token's partial output.
#pragma unroll
            for (uint32_t i = 0; i < T; ++i) {
                sMax[i][w] = mx[i];
                sSum[i][w] = sm[i];
                sOut[i][w][lane] = ov[i][0];
                sOut[i][w][lane + 32u] = ov[i][1];
                sOut[i][w][lane + 64u] = ov[i][2];
                sOut[i][w][lane + 96u] = ov[i][3];
            }
            __syncthreads();

            for (uint32_t i = w; i < T; i += W) {
                const uint32_t t = tBase + i;
                if (t >= tEnd) break;
                float M = 0.0f;// sink logit
                for (uint32_t j = 0; j < W; ++j) M = fmaxf(M, sMax[i][j]);
                float gsum = expf(0.0f - M);
                for (uint32_t j = 0; j < W; ++j)
                    if (sSum[i][j] > 0.0f)
                        gsum += sSum[i][j] * expf(sMax[i][j] - M);
                const float inv = 1.0f / gsum;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (uint32_t j = 0; j < W; ++j) {
                    if (sSum[i][j] > 0.0f) {
                        const float sc = expf(sMax[i][j] - M) * inv;
                        a0 += sc * sOut[i][j][lane];
                        a1 += sc * sOut[i][j][lane + 32u];
                        a2 += sc * sOut[i][j][lane + 64u];
                        a3 += sc * sOut[i][j][lane + 96u];
                    }
                }
                float *oh = out + static_cast<size_t>(t) * qDim +
                            static_cast<size_t>(hq) * 128u;
                oh[lane] = a0;
                oh[lane + 32u] = a1;
                oh[lane + 64u] = a2;
                oh[lane + 96u] = a3;
            }
        }

        /// Flash-style causal attention for a long batch.
        ///
        /// The tiled kernel above gives every (query head, 8-token tile) its own
        /// block and lets W warps split that tile's key range, so the 3 q heads
        /// of a GQA group each re-read the same K/V and the cross-warp
        /// online-softmax combine needs a [T][W][128] shared buffer (33 KiB,
        /// one block per SM).  It measures 0.71 ns per (token, key) pair.
        ///
        /// Here a block owns one KV head and a TT-token tile: all 3 q heads of
        /// the group share it, K/V are staged into shared memory in 16-key
        /// tiles, and each warp walks NT of the block's tokens with its own
        /// online softmax (so no cross-warp combine and no output buffer).  The
        /// default instantiation is TT = 8, NT = 1: 78 registers, 16 KiB of
        /// shared, three blocks per SM, and 0.35 ns per (token, key) pair on
        /// the 512-token geometry -- a 2x gain over the tiled kernel at 0.70.
        /// Larger TT/NT keep more token state in registers and lose occupancy
        /// (TT=32/NT=4 measures 0.75 ns), which is why TT=8 is the default.
        ///
        /// This is a *different numeric trajectory* from the token path -- keys
        /// are accumulated sequentially instead of split across warps and
        /// folded -- which is why it is gated by cosine against the token
        /// kernel rather than bit equality (the isolated kernel agrees to
        /// 3.6e-7 absolute, cosine 1.0).  It is used for batches of at least
        /// 128 tokens; smaller batches keep the exact tiled kernel.
        template <uint32_t TT, uint32_t KT, uint32_t NT>
        __global__ void attentionFlashKernel(const float *__restrict__ q,
                                             const float *__restrict__ kCache,
                                             const float *__restrict__ vCache,
                                             const float *__restrict__ table,
                                             uint32_t pos0, uint32_t nHeads,
                                             uint32_t nKvHeads, uint32_t extent,
                                             float qkScale, float *__restrict__ out,
                                             uint32_t window, uint32_t isSwa,
                                             uint32_t m) {
            __shared__ float sK[KT][128];
            __shared__ float sV[KT][128];
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t w = threadIdx.x >> 5u;
            const uint32_t hkv = blockIdx.x;
            const uint32_t group = nHeads / nKvHeads;
            const uint32_t qDim = nHeads * 128u;
            const uint32_t kvStride = nKvHeads * 128u;
            const uint32_t t0 = blockIdx.y * TT;
            if (t0 >= m) return;
            const uint32_t tLast = t0 + TT < m ? t0 + TT : m;
            const uint32_t hq0 = hkv * group;

            // NT tokens per warp, TT = 8 warps * NT.
            const uint32_t posFirst = pos0 + t0;
            const uint32_t fkFirst =
                    (isSwa != 0u && window > 0u && posFirst + 1u > window)
                            ? posFirst + 1u - window
                            : 0u;
            const uint32_t kEnd = pos0 + tLast - 1u;
            const uint32_t kStart = (fkFirst / KT) * KT;

            float qv[NT][3][4];
            float ov[NT][3][4];
            float mx[NT][3], sm[NT][3];
            // Position/first-key as float: keeps these per-token arrays in
            // registers (an integer array indexed the same way spills).
            float posv[NT], fkv[NT];
#pragma unroll
            for (uint32_t i = 0; i < NT; ++i) {
                const uint32_t t = t0 + w + 8u * i;
                if (t >= tLast) continue;
                const uint32_t pos = pos0 + t;
                posv[i] = float(pos);
                const uint32_t fk =
                        (isSwa != 0u && window > 0u && pos + 1u > window)
                                ? pos + 1u - window
                                : 0u;
                fkv[i] = float(fk);
                const float *qh = q + static_cast<size_t>(t) * qDim +
                                  static_cast<size_t>(hq0) * 128u;
#pragma unroll
                for (uint32_t g = 0; g < 3u; ++g) {
                    const float *qg = qh + static_cast<size_t>(g) * 128u;
                    qv[i][g][0] = qg[lane];
                    qv[i][g][1] = qg[lane + 32u];
                    qv[i][g][2] = qg[lane + 64u];
                    qv[i][g][3] = qg[lane + 96u];
                    mx[i][g] = 0.0f;// the sink is a virtual key with logit 0
                    sm[i][g] = 1.0f;
                    ov[i][g][0] = 0.0f;
                    ov[i][g][1] = 0.0f;
                    ov[i][g][2] = 0.0f;
                    ov[i][g][3] = 0.0f;
                }
            }

            for (uint32_t kt = kStart; kt <= kEnd; kt += KT) {
                for (uint32_t idx = threadIdx.x; idx < KT * 128u;
                     idx += blockDim.x) {
                    const uint32_t row = idx >> 7u;
                    const uint32_t col = idx & 127u;
                    uint32_t kp = kt + row;
                    if (kp > kEnd) kp = kEnd;
                    const size_t off = static_cast<size_t>(kp) * kvStride +
                                       static_cast<size_t>(hkv) * 128u + col;
                    sK[row][col] = kCache[off];
                    sV[row][col] = vCache[off];
                }
                __syncthreads();
#pragma unroll
                for (uint32_t i = 0; i < NT; ++i) {
                    const uint32_t t = t0 + w + 8u * i;
                    if (t >= tLast) continue;
                    const uint32_t pos = static_cast<uint32_t>(posv[i]);
                    const uint32_t fk = static_cast<uint32_t>(fkv[i]);
                    if (pos < kt) continue;
#pragma unroll
                    for (uint32_t kk = 0; kk < KT; ++kk) {
                        const uint32_t kp = kt + kk;
                        if (kp < fk || kp > pos) continue;
                        const float k0 = sK[kk][lane];
                        const float k1 = sK[kk][lane + 32u];
                        const float k2 = sK[kk][lane + 64u];
                        const float k3 = sK[kk][lane + 96u];
                        const float v0 = sV[kk][lane];
                        const float v1 = sV[kk][lane + 32u];
                        const float v2 = sV[kk][lane + 64u];
                        const float v3 = sV[kk][lane + 96u];
                        const uint32_t dist = bMin(pos - kp, extent);
#pragma unroll
                        for (uint32_t g = 0; g < 3u; ++g) {
                            float d = qv[i][g][0] * k0 + qv[i][g][1] * k1 +
                                      qv[i][g][2] * k2 + qv[i][g][3] * k3;
                            d = bWarpSum(d);
                            const float logit =
                                    d * qkScale +
                                    table[(static_cast<size_t>(t) * nHeads +
                                           hq0 + g) *
                                                  (extent + 1u) +
                                          dist];
                            float p;
                            if (logit > mx[i][g]) {
                                const float corr = expf(mx[i][g] - logit);
                                sm[i][g] = sm[i][g] * corr + 1.0f;
                                ov[i][g][0] *= corr;
                                ov[i][g][1] *= corr;
                                ov[i][g][2] *= corr;
                                ov[i][g][3] *= corr;
                                mx[i][g] = logit;
                                p = 1.0f;
                            } else {
                                p = expf(logit - mx[i][g]);
                                sm[i][g] += p;
                            }
                            ov[i][g][0] += p * v0;
                            ov[i][g][1] += p * v1;
                            ov[i][g][2] += p * v2;
                            ov[i][g][3] += p * v3;
                        }
                    }
                }
                __syncthreads();
            }

#pragma unroll
            for (uint32_t i = 0; i < NT; ++i) {
                const uint32_t t = t0 + w + 8u * i;
                if (t >= tLast) continue;
                float *oh = out + static_cast<size_t>(t) * qDim +
                            static_cast<size_t>(hq0) * 128u;
#pragma unroll
                for (uint32_t g = 0; g < 3u; ++g) {
                    const float inv = 1.0f / sm[i][g];
                    float *og = oh + static_cast<size_t>(g) * 128u;
                    og[lane] = ov[i][g][0] * inv;
                    og[lane + 32u] = ov[i][g][1] * inv;
                    og[lane + 64u] = ov[i][g][2] * inv;
                    og[lane + 96u] = ov[i][g][3] * inv;
                }
            }
        }

        inline uint32_t attentionWarps(uint32_t nKeys) {
            uint32_t w = nKeys / 64u;
            if (w < 1u) w = 1u;
            if (w > 8u) w = 8u;
            if (w <= 2u) return 2u;
            if (w <= 4u) return 4u;
            return 8u;
        }

        // ---- log-scaled positions -------------------------------------------

        __global__ void tauScaleBatchKernel(float *q, uint32_t qDim, float *rel,
                                            uint32_t relDim, uint32_t pos0,
                                            float alpha, float floor) {
            const uint32_t t = blockIdx.x;
            const uint32_t pos = pos0 + t;
            const float arg =
                    fmaxf(1.0f, float(pos + 1) / fmaxf(1.0f, floor));
            const float tau = 1.0f + alpha * logf(arg);
            float *qp = q + static_cast<size_t>(t) * qDim;
            float *rp = rel + static_cast<size_t>(t) * relDim;
            for (uint32_t i = threadIdx.x; i < qDim; i += blockDim.x)
                qp[i] *= tau;
            for (uint32_t i = threadIdx.x; i < relDim; i += blockDim.x)
                rp[i] *= tau;
        }

        // ---- short convolution ----------------------------------------------
        //
        // The state is per slot and strictly sequential, so one thread owns one
        // channel and walks the chunk in order: token t's taps see the history
        // left by token t-1.  Same arithmetic as shortConvKernel, same order.
        /// Generic form, kept for a tap count the tiled kernel does not
        /// instantiate: the state is per slot and strictly sequential, so one
        /// thread owns one channel and walks the whole chunk in order.
        __global__ void shortConvBatchSerialKernel(const float *__restrict__ cur,
                                                   const float *__restrict__ w,
                                                   const float *__restrict__ dyn,
                                                   float *__restrict__ hist,
                                                   float *__restrict__ out,
                                                   uint32_t channels,
                                                   uint32_t nConv, uint32_t nHeads,
                                                   bool residual, float scale,
                                                   uint32_t m) {
            const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= channels) return;
            const uint32_t h = c / (channels / nHeads);
            for (uint32_t t = 0; t < m; ++t) {
                const float *ct = cur + static_cast<size_t>(t) * channels;
                float *ot = out + static_cast<size_t>(t) * channels;
                float acc = residual ? ct[c] : 0.0f;
                for (uint32_t k = 0; k < nConv; ++k) {
                    const float v = (k + 1 < nConv)
                                            ? hist[static_cast<size_t>(k) * channels + c]
                                            : ct[c];
                    acc += w[static_cast<size_t>(c) * nConv + k] * v;
                }
                if (dyn != nullptr) {
                    const float *dt =
                            dyn + (static_cast<size_t>(t) * nHeads + h) * nConv;
                    for (uint32_t lag = 0; lag < nConv; ++lag) {
                        const uint32_t src = nConv - 1 - lag;
                        const float v = (src + 1 < nConv)
                                                ? hist[static_cast<size_t>(src) * channels + c]
                                                : ct[c];
                        acc += dt[lag] * v;
                    }
                }
                ot[c] = acc * scale;
                for (uint32_t k = 0; k + 2 < nConv; ++k)
                    hist[static_cast<size_t>(k) * channels + c] =
                            hist[static_cast<size_t>(k + 1) * channels + c];
                hist[static_cast<size_t>(nConv - 2) * channels + c] = ct[c];
            }
        }

        /// Short convolution over a token tile.  The serial kernel launched one
        /// block per 256 channels (9 blocks for 2304 channels, so 9 of 68 SMs)
        /// and carried a load/store dependency chain through the whole chunk:
        /// measured 29 ms of a 512-token pass.  The state after token t is just
        /// the last nConv-1 inputs, and those are already in `cur`, so a block
        /// can start any token tile by seeding the delay registers from `cur`
        /// (or from the incoming `hist` for the first tile), walk its tiles with
        /// a register window, and write the outgoing history from the last tile
        /// only.  The per-token tap order is the serial kernel's exactly
        /// (`residual`, static taps k ascending, then dynamic lags ascending),
        /// so this is bit-identical.
        template <uint32_t NC>
        __global__ void shortConvTileKernel(const float *__restrict__ cur,
                                            const float *__restrict__ w,
                                            const float *__restrict__ dyn,
                                            float *__restrict__ hist,
                                            float *__restrict__ out,
                                            uint32_t channels, uint32_t nHeads,
                                            bool residual, float scale,
                                            uint32_t m) {
            constexpr uint32_t TT = 64;
            const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= channels) return;
            const uint32_t h = c / (channels / nHeads);
            const uint32_t t0 = blockIdx.y * TT;
            const uint32_t t1 = t0 + TT < m ? t0 + TT : m;
            float wv[NC];
#pragma unroll
            for (uint32_t k = 0; k < NC; ++k) wv[k] = w[c * NC + k];
            float prev[NC];// prev[d] = input at delay d, padded at d = 0
            prev[0] = 0.0f;
#pragma unroll
            for (uint32_t d = 1; d < NC; ++d) {
                prev[d] = t0 >= d
                                  ? cur[static_cast<size_t>(t0 - d) * channels + c]
                                  : hist[static_cast<size_t>(NC - 1 - d) * channels + c];
            }
            for (uint32_t t = t0; t < t1; ++t) {
                const float cur0 = cur[static_cast<size_t>(t) * channels + c];
                float acc = residual ? cur0 : 0.0f;
#pragma unroll
                for (uint32_t k = 0; k + 1 < NC; ++k)
                    acc += wv[k] * prev[NC - 1 - k];
                acc += wv[NC - 1] * cur0;
                if (dyn != nullptr) {
                    const float *dt =
                            dyn + (static_cast<size_t>(t) * nHeads + h) * NC;
                    acc += dt[0] * cur0;
#pragma unroll
                    for (uint32_t lag = 1; lag < NC; ++lag)
                        acc += dt[lag] * prev[lag];
                }
                out[static_cast<size_t>(t) * channels + c] = acc * scale;
#pragma unroll
                for (uint32_t d = NC - 1; d > 1; --d) prev[d] = prev[d - 1];
                prev[1] = cur0;
            }
            if (t1 == m) {
#pragma unroll
                for (uint32_t k = 0; k + 1 < NC; ++k)
                    hist[static_cast<size_t>(k) * channels + c] =
                            prev[NC - 1 - k];
            }
        }

        // ---- router ----------------------------------------------------------

        __global__ void routerBatchKernel(const float *__restrict__ logits,
                                          const float *__restrict__ bias,
                                          int32_t *__restrict__ idx,
                                          float *__restrict__ wgt,
                                          uint32_t nExperts, uint32_t nUsed,
                                          float scale) {
            __shared__ float probs[256];
            __shared__ float sel[256];
            const uint32_t t = blockIdx.x;
            const float *lg = logits + static_cast<size_t>(t) * nExperts;
            const uint32_t e = threadIdx.x;
            if (e < nExperts) {
                probs[e] = 1.0f / (1.0f + expf(-lg[e]));
                sel[e] = probs[e] + (bias ? bias[e] : 0.0f);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int32_t *ix = idx + static_cast<size_t>(t) * nUsed;
                float *wt = wgt + static_cast<size_t>(t) * nUsed;
                float sum = 0.0f;
                for (uint32_t u = 0; u < nUsed; ++u) {
                    uint32_t best = 0;
                    float bv = -INFINITY;
                    for (uint32_t k = 0; k < nExperts; ++k) {
                        if (sel[k] > bv) {
                            bv = sel[k];
                            best = k;
                        }
                    }
                    ix[u] = static_cast<int32_t>(best);
                    sum += probs[best];
                    sel[best] = -INFINITY;
                }
                const float norm = scale / fmaxf(sum, 1e-20f);
                for (uint32_t u = 0; u < nUsed; ++u) {
                    const int32_t e2 = ix[u];
                    wt[u] = (1.0f / (1.0f + expf(-lg[e2]))) * norm;
                }
            }
        }

        /// gateUpRank[u] = gateUpScale[idx[u]], downRank[u] = downScale[idx[u]]
        /// for every routed pair of every token.
        [[maybe_unused]]
        __global__ void gatherScalesBatchKernel(float *__restrict__ gateUp,
                                                float *__restrict__ down,
                                                const int32_t *__restrict__ idx,
                                                const float *__restrict__ gs,
                                                const float *__restrict__ ds,
                                                uint32_t n) {
            const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
            if (u >= n) return;
            const int32_t e = idx[u];
            gateUp[u] = gs[e];
            down[u] = ds[e];
        }

        // ---- routed MoE ------------------------------------------------------

        /// Same fused per-expert scale + SITU as situAllKernel, but reading and
        /// writing the grouped layout [nExperts][capacity][2n] and compacting
        /// the activation into act[g][slot][n] (the down projection's input).
        __global__ void situGroupedKernel(float *__restrict__ gateUp,
                                          float *__restrict__ act,
                                          const int32_t *__restrict__ gexp,
                                          const float *__restrict__ scale,
                                          uint32_t n, float beta, float lb) {
            // One block per routed (token, expert) pair, in the compact pair
            // order the gather wrote: the pair's slot IS blockIdx.x.
            const uint32_t p = blockIdx.x;
            const uint32_t e = static_cast<uint32_t>(gexp[p]);
            const float s = scale[e];
            float *base = gateUp + static_cast<size_t>(p) * 2 * n;
            float *dst = act + static_cast<size_t>(p) * n;
            for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
                const float g = base[j] * s;
                const float up = base[n + j] * s;
                const float v = beta * tanhf(g / beta) *
                                (1.0f / (1.0f + expf(-g))) *
                                (lb > 0.0f ? lb * tanhf(up / lb) : up);
                dst[j] = v;
            }
        }

        /// SITU for a per-token gate/up buffer (the exact path, where the gate/up
        /// GEMV stays per token), compacted into the grouped down projection's
        /// [expert][slot][n] activation.  Same arithmetic as situAllKernel.
        __global__ void situGatherKernel(const float *__restrict__ gateUp,
                                         const int32_t *__restrict__ idx,
                                         const int32_t *__restrict__ gpos,
                                         float *__restrict__ act,
                                         const float *__restrict__ scale,
                                         uint32_t n, uint32_t nUsed, float beta,
                                         float lb) {
            const uint32_t u = blockIdx.x;
            const uint32_t t = blockIdx.y;
            if (u >= nUsed) return;
            const uint32_t p = t * nUsed + u;
            const uint32_t e = static_cast<uint32_t>(idx[p]);
            const uint32_t slot = static_cast<uint32_t>(gpos[p]);
            const float s = scale[e];
            const float *base = gateUp + static_cast<size_t>(p) * 2 * n;
            float *dst = act + static_cast<size_t>(slot) * n;
            for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
                const float g = base[j] * s;
                const float up = base[n + j] * s;
                dst[j] = beta * tanhf(g / beta) *
                         (1.0f / (1.0f + expf(-g))) *
                         (lb > 0.0f ? lb * tanhf(up / lb) : up);
            }
        }

        /// Counting sort of the m*nUsed routed (token, expert) pairs by expert,
        /// into a compact [nPairs] array with per-expert offsets.  A compact
        /// layout is what lets one gather cover the whole outer batch however
        /// skewed the router is: the padded [nExperts][capacity] form has to be
        /// sized for the worst case (every token on one expert) and that is
        /// 1.5 GiB at m=512, while the compact form is exactly m*nUsed rows.
        /// counts[e] and goff[e] are both written; expert e lives in
        /// goff[e]..goff[e]+counts[e].  Pair order inside a slab is token order,
        /// so the gather and every consumer see a deterministic layout.
        __global__ void moeGroupBuildKernel(const int32_t *__restrict__ idx,
                                            const float *__restrict__ wgt,
                                            int32_t *__restrict__ counts,
                                            int32_t *__restrict__ goff,
                                            int32_t *__restrict__ gtok,
                                            int32_t *__restrict__ gpos,
                                            float *__restrict__ gwgt,
                                            int32_t *__restrict__ gexp, uint32_t m,
                                            uint32_t nUsed, uint32_t nExperts) {
            // One block of 256 threads.  The serial form (thread 0 only) was
            // 0.41 ms per layer on 4096 pairs -- 20 ms per 512-token pass -- and
            // all of it was latency on the dependent global writes.  Three
            // phases: parallel histogram + prefix (unchanged result), then a
            // stable scatter in tiles of blockDim.x pairs.  Inside a tile a
            // pair's rank is the number of earlier pairs in the tile with the
            // same expert, so the layout (expert slabs in expert order, tokens
            // in pair order inside a slab) is bit-identical to the serial one
            // and needs no atomics for the slot assignment.
            __shared__ int cursor[256];// nExperts <= 256 (ensureGraph)
            __shared__ int plane[256];// this tile's expert ids, -1 = dead lane
            __shared__ int hist[256];
            const uint32_t n = m * nUsed;
            const uint32_t tid = threadIdx.x;
            const uint32_t nthr = blockDim.x;
            for (uint32_t e = tid; e < nExperts; e += nthr) {
                counts[e] = 0;
                hist[e] = 0;
            }
            __syncthreads();
            for (uint32_t p = tid; p < n; p += nthr)
                atomicAdd(&hist[idx[p]], 1);
            __syncthreads();
            if (tid == 0) {
                int32_t acc = 0;
                for (uint32_t e = 0; e < nExperts; ++e) {
                    counts[e] = hist[e];
                    goff[e] = acc;
                    cursor[e] = acc;
                    acc += hist[e];
                }
                goff[nExperts] = acc;
            }
            __syncthreads();
            for (uint32_t p0 = 0; p0 < n; p0 += nthr) {
                const uint32_t p = p0 + tid;
                const bool live = p < n;
                const int e = live ? idx[p] : -1;
                plane[tid] = e;
                __syncthreads();
                uint32_t before = 0, after = 0;
                for (uint32_t j = 0; j < nthr; ++j) {
                    const int pe = plane[j];
                    if (pe == e) {
                        if (j < tid) ++before;
                        else if (j > tid) ++after;
                    }
                }
                if (live) {
                    const int32_t slot = cursor[e] + static_cast<int32_t>(before);
                    gtok[slot] = static_cast<int32_t>(p / nUsed);
                    gwgt[slot] = wgt[p];
                    gpos[p] = slot;
                    gexp[slot] = e;
                }
                __syncthreads();
                // The last pair of this expert inside the tile advances the
                // cursor by the tile's count for that expert; exactly one
                // thread per expert does it, so no atomic is needed.
                if (live && after == 0) cursor[e] += static_cast<int>(before) + 1;
                __syncthreads();
            }
        }

        __global__ void gatherExpertXKernel(const float *__restrict__ x,
                                            const int32_t *__restrict__ gtok,
                                            float *__restrict__ gx,
                                            uint32_t cols) {
            // One block per compact slot.  The per-expert form (one block per
            // expert, its rows copied in a serial loop) was 0.84 ms per layer
            // = 40 ms per 512-token pass at only ~90 GB/s: 64 blocks and a
            // dependent row loop cannot fill the memory pipe.  With one block
            // per row the slots are independent, `tok` is one broadcast load,
            // and the copy is float4.
            const uint32_t slot = blockIdx.x;
            const uint32_t t = static_cast<uint32_t>(gtok[slot]);
            const float *src = x + static_cast<size_t>(t) * cols;
            float *dst = gx + static_cast<size_t>(slot) * cols;
            if ((cols & 3u) == 0u) {
                const uint4 *s4 = reinterpret_cast<const uint4 *>(src);
                uint4 *d4 = reinterpret_cast<uint4 *>(dst);
                const uint32_t n4 = cols >> 2u;
                for (uint32_t j = threadIdx.x; j < n4; j += blockDim.x)
                    d4[j] = s4[j];
            } else {
                for (uint32_t j = threadIdx.x; j < cols; j += blockDim.x)
                    dst[j] = src[j];
            }
        }

        __global__ void moeAccumGroupedKernel(
                const float *__restrict__ expertOut,
                const int32_t *__restrict__ idx, const int32_t *__restrict__ gpos,
                const float *__restrict__ wgt,
                const float *__restrict__ downScale, float *__restrict__ out,
                uint32_t nUsed, uint32_t hidden) {
            const uint32_t t = blockIdx.x;
            for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
                float a = 0.0f;
                for (uint32_t u = 0; u < nUsed; ++u) {
                    const uint32_t p = t * nUsed + u;
                    const uint32_t e = static_cast<uint32_t>(idx[p]);
                    const uint32_t slot = static_cast<uint32_t>(gpos[p]);
                    const float w = wgt[p] * downScale[e];
                    a += w * expertOut[static_cast<size_t>(slot) * hidden + i];
                }
                out[static_cast<size_t>(t) * hidden + i] = a;
            }
        }

        // ---- elementwise glue over m rows ------------------------------------

        [[maybe_unused]]
        __global__ void bCombineKernel(float *out, const float *a, const float *b,
                                       uint32_t n, float sa, float sb) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += gridDim.x * blockDim.x)
                out[i] = sa * a[i] + sb * b[i];
        }

        [[maybe_unused]]
        __global__ void bSigmoidMulKernel(float *x, const float *g, uint32_t n) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += gridDim.x * blockDim.x)
                x[i] *= 1.0f / (1.0f + expf(-g[i]));
        }

        [[maybe_unused]]
        __global__ void bSoftcapKernel(float *x, uint32_t n, float cap) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += gridDim.x * blockDim.x)
                x[i] = cap * tanhf(x[i] / cap);
        }

    }// namespace

#ifndef __CUDA_ARCH__
    namespace {
        inline bool bcopyD2D(void *dst, const void *src, uint64_t bytes,
                             cudaStream_t s) {
            if (bytes == 0) return true;
            return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s) ==
                   cudaSuccess;
        }
        inline void bCombine(float *out, const float *a, const float *b, uint32_t n,
                             float sa, float sb, cudaStream_t s) {
            bCombineKernel<<<(n + 255u) / 256u, 256, 0, s>>>(out, a, b, n, sa, sb);
        }
        inline void bSigmoidMul(float *x, const float *g, uint32_t n,
                                cudaStream_t s) {
            bSigmoidMulKernel<<<(n + 255u) / 256u, 256, 0, s>>>(x, g, n);
        }
        inline void bSoftcap(float *x, uint32_t n, float cap, cudaStream_t s) {
            bSoftcapKernel<<<(n + 255u) / 256u, 256, 0, s>>>(x, n, cap);
        }

        /// Debug dump (TINYCODER_INKLING_BATCHED_DUMP): one file per tag, written
        /// after a stream sync.  Only used by the parity tool.
        void batchDump(const char *dir, const char *name, const float *dev,
                       uint32_t n, cudaStream_t s) {
            if (dir == nullptr) return;
            std::vector<float> h(n);
            if (cudaMemcpyAsync(h.data(), dev, static_cast<size_t>(n) * sizeof(float),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess)
                return;
            if (cudaStreamSynchronize(s) != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
            FILE *f = std::fopen(path, "wb");
            if (f != nullptr) {
                std::fwrite(h.data(), sizeof(float), n, f);
                std::fclose(f);
            }
        }

        /// Per-token stage dump for all m rows: "<dir>/bt-<stage>-t<t>-s<slot>.f32".
        void dumpStageSet(const char *dir, uint32_t slot, const char *stage,
                          const float *dev, uint32_t width, uint32_t m,
                          cudaStream_t s) {
            if (dir == nullptr) return;
            const char *only = std::getenv("TINYCODER_INKLING_STAGE_SLOT");
            if (only != nullptr && static_cast<uint32_t>(std::atoi(only)) != slot)
                return;
            for (uint32_t t = 0; t < m; ++t) {
                char tag[64];
                std::snprintf(tag, sizeof(tag), "%s-t%u-s%u", stage, t, slot);
                batchDump(dir, tag, dev + static_cast<size_t>(t) * width, width, s);
            }
        }

        /// Router ids as float (exact below 2^24) so the parity tool can tell a
        /// changed expert selection from a numerically shifted one.
        void batchDumpI32(const char *dir, const char *name, const int32_t *dev,
                          uint32_t n, cudaStream_t s) {
            if (dir == nullptr) return;
            std::vector<int32_t> h(n);
            std::vector<float> f(n);
            if (cudaMemcpyAsync(h.data(), dev, static_cast<size_t>(n) * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess)
                return;
            if (cudaStreamSynchronize(s) != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            for (uint32_t i = 0; i < n; ++i) f[i] = static_cast<float>(h[i]);
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
            FILE *fp = std::fopen(path, "wb");
            if (fp != nullptr) {
                std::fwrite(f.data(), sizeof(float), n, fp);
                std::fclose(fp);
            }
        }

        // ---- optional section timer (TINYCODER_INKLING_BATCHED_PROF=1) -----
        // The graph is a single stream, so a host timestamp pair per section
        // boundary measures that section's device time exactly.  Each boundary
        // drains the stream, which inflates the total; the shares are the point.
        struct BatchProf {
            static constexpr int kSections = 36;
            bool on = std::getenv("TINYCODER_INKLING_BATCHED_PROF") != nullptr;
            double sec[kSections] = {};
            const char *name[kSections] = {
                    "embed", "attn-pre", "qkv", "conv", "attention", "attn-out",
                    "resid", "moe-gate", "moe-down", "ffn-tail", "ctl-down",
                    "attn-norm", "attn-clip", "attn-relb", "attn-store",
                    "attn-kern", "c-q", "c-k", "n-v", "c-rel", "sink", "tau",
                    "router-gemm", "ctl-mix", "ctl-inv", "ffn-mix",
                    "router-kern", "lm-head",
                    // MoE sub-sections: the grouped path is a four-stage
                    // pipeline (sort, gather, gate/up+SITU, down+accum) and the
                    // aggregate "moe-gate"/"moe-down" hides which stage to cut.
                    "moe-build", "moe-gather", "moe-gu", "moe-situ",
                    "moe-down-mma", "moe-accum", "moe-quant", "moe-x"};
            std::chrono::steady_clock::time_point last;
            void begin() {
                if (on) last = std::chrono::steady_clock::now();
            }
            void mark(int next, cudaStream_t s) {
                if (!on) return;
                cudaStreamSynchronize(s);
                const auto n = std::chrono::steady_clock::now();
                // Attribute the interval to the section that just finished.
                sec[cur] += std::chrono::duration<double>(n - last).count();
                last = n;
                cur = next;
            }
            int cur = 0;
            /// Re-target the next interval without attributing the current one
            /// (used between MoE chunks, whose gate and down sections alternate:
            /// the gap after a down section belongs to the next gate section).
            void flip(int next) {
                if (!on) return;
                cur = next;
                last = std::chrono::steady_clock::now();
            }
            void print() {
                if (!on) return;
                double total = 0.0;
                for (double v: sec) total += v;
                std::fprintf(stderr, "[batch-prof]");
                for (int i = 0; i < kSections; ++i)
                    if (sec[i] > 0.0)
                        std::fprintf(stderr, " %s=%.1fms", name[i], sec[i] * 1e3);
                std::fprintf(stderr, " total=%.1fms\n", total * 1e3);
            }
        };

        /// FAST mode: accept the int8 multi-token accumulation order and the
        /// float decoder for every expert type.  It is the faster graph; it is
        /// not the token path's trajectory, so it is opt-in and the tool
        /// reports its parity separately.  Retained as a diagnostic: the
        /// default path no longer needs it, because the dense projections use
        /// the engine's int8 tensor-core GEMM instead of per-token GEMVs.
        bool fastMode() {
            static const bool f =
                    std::getenv("TINYCODER_INKLING_BATCHED_FAST") != nullptr;
            return f;
        }

        /// Dense int8 tensor-core GEMM: one Q8_1 activation quantization for all
        /// m rows, then one stream-k launch whose weight tile is loaded once and
        /// multiplied by tile_x (64/128) rows at a time.  This is the piece that
        /// removes the per-token weight stream from the dense projections; at
        /// m=128 it replaces 128 int8Gemv launches with one GEMM per matrix.
        ///
        /// OFF by default: it is the fastest dense path but its Q8_1 int8
        /// accumulation is a different trajectory from the token-major int8
        /// GEMV, and the model-level gate (top-1 / top-5 set on every prompt
        /// position) does not survive it.  TINYCODER_INKLING_GEMM=1 opts in;
        /// the measured pp512 and parity counts are in the tool output.
        bool int8GemmDenseEnabled() {
            static const bool f = [] {
                const char *e = std::getenv("TINYCODER_INKLING_GEMM");
                return e == nullptr || e[0] != '0';
            }();
            return f;
        }

        /// Types `int8Gemm` accepts and Inkling ships in a dense projection.
        /// IQ2_XS is in the list because the dispatch supports it, but no dense
        /// Inkling tensor uses it today.
        bool gemmDenseType(uint32_t type) {
            return type == 8u || type == 12u || type == 13u || type == 14u ||
                   type == 17u || type == 22u;
        }

        /// K is consumed in 128-value activation-quantize groups, so the only
        /// requirement is cols % 128 == 0: 2304 = 18*128 and 1152 = 9*128 are
        /// the Inkling widths, and the quantizer's tail groups are simply never
        /// read (see GemmDispatch.cu).
        bool int8GemmDense(uint32_t type, const void *w, const float *x,
                           float *out, uint32_t rows, uint32_t cols, uint32_t m,
                           cudaStream_t s) {
            if (!int8GemmDenseEnabled() || m < 8 || !gemmDenseType(type) ||
                (cols & 127u) != 0u)
                return false;
            const uint32_t qk = inklingTypeQk(type);
            const uint32_t blocksPerRow = (cols + qk - 1u) / qk;
            if (int8Gemm(static_cast<int>(type), w, x, out, m, rows, cols,
                         blocksPerRow, reinterpret_cast<void *>(s)))
                return true;
            cudaGetLastError();
            return false;
        }

        /// Opt-in int8 MMQ for the Q2_0 expert down projection.  OFF by default
        /// for the same reason as the dense int8 GEMM: its Q8_1 activation
        /// quantization is a different trajectory from the float decoder the
        /// token path uses (q2_0_mmq_parity measures 3.6e-3 relRMS on random
        /// activations), and the model-level gate reflects that.
        /// TINYCODER_INKLING_Q2_0_MMA=1 opts in (moe-down 375 -> 187 ms per
        /// 128-token chunk, 4 top-1 flips at 128 tokens).
        /// Grouped int8 MMQ for the routed experts (both directions).
        /// TINYCODER_INKLING_EXPERT_MMA=1 opts in.  OFF by default because the
        /// Q8_1 activation quantization is a different trajectory from the
        /// token path's GEMV decoders, exactly like the Q2_0 down MMQ; the
        /// model-level gate and tools/inkling_expert_mma_parity.cpp decide.
        /// TINYCODER_INKLING_EXPERT_MMA selects the direction: 1 = both,
        /// "gate" = gate/up only, "down" = down only.  The two have different
        /// parity deltas (the gate/up activation error is amplified by SITU
        /// and the down decoder), so they are measured separately.
        bool expertMmaGate() {
            static const int mode = [] {
                const char *e = std::getenv("TINYCODER_INKLING_EXPERT_MMA");
                if (e != nullptr && (e[0] == '\0' || e[0] == '0')) return 0;
                if (e != nullptr && std::strcmp(e, "down") == 0) return 2;
                if (e != nullptr && std::strcmp(e, "gate") == 0) return 1;
                return 3;
            }();
            return (mode & 1) != 0;
        }
        bool expertMmaDown() {
            static const int mode = [] {
                const char *e = std::getenv("TINYCODER_INKLING_EXPERT_MMA");
                if (e != nullptr && (e[0] == '\0' || e[0] == '0')) return 0;
                if (e != nullptr && std::strcmp(e, "down") == 0) return 2;
                if (e != nullptr && std::strcmp(e, "gate") == 0) return 1;
                return 3;
            }();
            return (mode & 2) != 0;
        }

        bool q20MmaEnabled() {
            static const bool f = [] {
                const char *e = std::getenv("TINYCODER_INKLING_Q2_0_MMA");
                return e == nullptr || e[0] != '0';
            }();
            return f;
        }

        /// One dense projection over the batch.
        ///   * a type/shape the engine's int8 tensor-core GEMM accepts goes
        ///     through int8Gemm once for all m rows (weight tile reused across
        ///     the M dimension);
        ///   * otherwise the token path's int8 GEMV wins if it accepts the type,
        ///     called once per token: that keeps the exact token-major
        ///     trajectory for the types/shapes int8Gemm refused;
        ///   * everything else (NVFP4/Q2_0/IQ1_S/IQ2_XXS/IQ4_XS/F32/BF16) goes
        ///     through inklingGemvBatchedAsync in 16-row tiles, which is
        ///     bit-identical to one inklingGemv per token.
        bool gemvBatchedAny(uint32_t type, const void *w, const float *x,
                            float *out, uint32_t rows, uint32_t cols, uint32_t m,
                            cudaStream_t s) {
            if (int8GemmDense(type, w, x, out, rows, cols, m, s)) return true;
            // A projection with fewer rows than the batched GEMV's 32-row
            // block (the attention sink projection, 24 x 2304) runs as one
            // block in the general path however many tokens are in flight; the
            // one-warp-per-(token,row) form is bit-identical and scales with
            // the token count (104 ms -> ~2 ms per 128-token pass).
            if (m >= 8u && rows < 32u &&
                inklingGemvBatchedSmallAsync(type, w, x, out, rows, cols, m, s))
                return true;
            cudaGetLastError();
            if (fastMode()) {
                bool int8Ok = true;
                for (uint32_t off = 0; off < m; off += 8) {
                    const uint32_t mm = std::min(8u, m - off);
                    if (!int8Gemv(type, w, x + static_cast<size_t>(off) * cols,
                                  out + static_cast<size_t>(off) * rows, mm,
                                  rows, cols, s)) {
                        int8Ok = false;
                        break;
                    }
                }
                if (int8Ok) return true;
                cudaGetLastError();
            } else if (int8Gemv(type, w, x, out, 4u < m ? 4u : m, rows, cols, s)) {
                // Exact: the decode path owns this type, so keep its per-token
                // accumulation order rather than the float decoder's.  A 4-token
                // slice is bit-identical to four seqLen=1 calls: gemvCoreImpl
                // picks nwarps=4 for ncols<=4 (and 2 warps above, a different
                // reduction tree), and the per-row dot order does not depend on
                // how many rows the block carries.  8-token slices are NOT
                // exact for that reason; 4 is the largest exact slice and
                // halves the dense weight traffic again versus per-token.
                for (uint32_t off = 4; off < m; off += 4) {
                    const uint32_t mm = std::min(4u, m - off);
                    if (!int8Gemv(type, w, x + static_cast<size_t>(off) * cols,
                                  out + static_cast<size_t>(off) * rows, mm,
                                  rows, cols, s))
                        return false;
                }
                return true;
            } else {
                cudaGetLastError();
            }
            // Float decoder.  A narrow matrix (fewer than one block per SM:
            // the F32 router is 128 rows = 4 blocks, the BF16 conv projections
            // are 32 rows = 1 block) goes through the token-parallel form,
            // which puts the token groups in grid.y; the row-only grid measured
            // 384 ms per 512-token pass on the router.  Same per-output dot
            // order, so this stays bit-identical to the per-token GEMVs.  Wide
            // matrices keep the 16-row tiles, which re-read the weights once
            // per 16 tokens instead of once per 8.
            const uint32_t rowBlocks = (rows + 31u) / 32u;
            if (m >= 8u && rowBlocks < 68u &&
                inklingGemvBatchedTokenParallelAsync(type, w, x, out, rows, cols,
                                                     m, s))
                return true;
            cudaGetLastError();
            for (uint32_t off = 0; off < m; off += 16) {
                const uint32_t mm = std::min(16u, m - off);
                if (!inklingGemvBatchedAsync(type, w,
                                             x + static_cast<size_t>(off) * cols,
                                             out + static_cast<size_t>(off) * rows,
                                             rows, cols, mm, s))
                    return false;
            }
            return true;
        }

        /// Types the decode path's gemvExpertToken sends to int8GemvExpert
        /// with a shared activation row (gate/up).  Everything else falls to
        /// the float expert GEMV, which a grouped float kernel reproduces
        /// bit-for-bit.
        bool expertInt8GateUp(uint32_t type) {
            if (!int8GemmAvailable()) return false;
            switch (type) {
                case 8:
                case 12:
                case 13:
                case 14:
                case 17:
                case 18:
                case 21:
                case 22:
                    return true;
                default:
                    return false;
            }
        }

        bool gemvExpertToken(uint32_t type, const void *w, const float *x,
                             uint32_t xStride, float *out, uint32_t rowsPer,
                             uint32_t cols, const int32_t *idx, uint32_t nUsed,
                             cudaStream_t s) {
            if (int8GemvExpert(type, w, x, out, rowsPer, cols, idx, nUsed,
                               xStride, s))
                return true;
            cudaGetLastError();
            return inklingGemvExpertAsync(type, w, x, xStride, out, rowsPer, cols,
                                          idx, nUsed, s);
        }

    }// namespace

    // -------------------------------------------------------------------------
    // Batched scratch: one arena, sized from the geometry for kBatchM tokens.
    // -------------------------------------------------------------------------
    bool InklingGPU::Impl::ensureBatch(std::string &err) {
        auto &b = batch;
        if (b.ready) return true;
        const uint64_t m = kBatchM;
        const uint64_t H = hidden;
        const uint64_t kvDim = this->kvDim;
        const uint64_t qDim = static_cast<uint64_t>(nHeads) * headDim;
        const uint64_t bankRows = static_cast<uint64_t>(slots) + 2;
        const uint64_t nExp = nExperts;
        // Compact routed-MoE pairs: one row per (token, expert).
        const uint64_t nPairs = m * nUsed;

        uint64_t need = 0;
        need += 12 * m * H;                    // x prelude passOut inpSA ffnInp mixH
                                               // normed postO convH o moeOut emb
        need += 3 * m * H;                     // mixAttn: 3 query planes
        need += 4 * m * H;                     // msgMixed: 4 planes
        need += 3 * m * H;                     // pre0..2 (control outputs)
        need += 3 * m * H;                     // nq nk nv (attention-input norms)
        need += 2 * bankRows * m * H;          // block bank + snapshot
        need += 3 * m * qDim;                  // q gateProj attnOut
        need += 4 * m * kvDim;                 // k v kProj vProj
        need += m * nHeads * nRel;             // rel
        need += m * nHeads;                    // sink
        need += m * nHeads * (extent + 1);     // rel-bias table
        need += m * nExp;                      // router
        need += 2 * m * nUsed;                 // gateUpRank + downRank
        need += m * nUsed * 2 * expertFf;      // gateUp
        need += m * nUsed * H;                 // expertOut
        need += m * nConv * nKvHeads;          // dynamic conv taps
        need += m * 64;                        // control rank activation
        need += 2 * m;                         // control per-token RMS inverses
        need += m * vocab;                     // logits
        need += 2 * nPairs * H;                // gx + grouped down output
        need += nPairs * 2 * expertFf;         // grouped gate/up
        need += nPairs * expertFf;             // grouped activation
        need += nPairs;                        // gwgt
        need += m * nUsed;                     // wgt
        need += 2 * m * nUsed;                 // idx + gpos (int32)
        need += 2 * nPairs + 2 * (nExp + 1);   // gtok+gexp, gcount+goff (int32)
        b.bytes = need * sizeof(float);
        size_t freeBefore = 0, totalBytes = 0;
        cudaMemGetInfo(&freeBefore, &totalBytes);
        const cudaError_t ce = cudaMalloc(reinterpret_cast<void **>(&b.arena),
                                          b.bytes);
        if (ce != cudaSuccess) {
            cudaGetLastError();
            size_t mf = 0, mt = 0;
            cudaMemGetInfo(&mf, &mt);
            err = std::string("cudaMalloc(batched prefill scratch, ") +
                  std::to_string(b.bytes >> 20) + " MiB) failed: " +
                  cudaGetErrorString(ce) + " [free " +
                  std::to_string(mf >> 20) + " MiB of " +
                  std::to_string(mt >> 20) + " MiB]";
            b.arena = nullptr;
            return false;
        }
        float *cur = b.arena;
        auto take = [&](uint64_t n) {
            float *p = cur;
            cur += n;
            return p;
        };
        auto takeI = [&](uint64_t n) {
            return reinterpret_cast<int32_t *>(take(n));
        };
        b.x = take(m * H);
        b.prelude = take(m * H);
        b.passOut = take(m * H);
        b.inpSA = take(m * H);
        b.ffnInp = take(m * H);
        b.mixH = take(m * H);
        b.normed = take(m * H);
        b.postO = take(m * H);
        b.convH = take(m * H);
        b.o = take(m * H);
        b.moeOut = take(m * H);
        b.emb = take(m * H);
        b.mixAttn = take(3 * m * H);
        b.msgMixed = take(4 * m * H);
        b.pre0 = take(m * H);
        b.pre1 = take(m * H);
        b.pre2 = take(m * H);
        b.nq = take(m * H);
        b.nk = take(m * H);
        b.nv = take(m * H);
        b.blockBank = take(bankRows * m * H);
        b.bankSnap = take(bankRows * m * H);
        b.q = take(m * qDim);
        b.gateProj = take(m * qDim);
        b.attnOut = take(m * qDim);
        b.k = take(m * kvDim);
        b.v = take(m * kvDim);
        b.kProj = take(m * kvDim);
        b.vProj = take(m * kvDim);
        b.rel = take(m * nHeads * nRel);
        b.sink = take(m * nHeads);
        b.table = take(m * nHeads * (extent + 1));
        b.router = take(m * nExp);
        b.gateUpRank = take(m * nUsed);
        b.downRank = take(m * nUsed);
        b.gateUp = take(m * nUsed * 2 * expertFf);
        b.expertOut = take(m * nUsed * H);
        b.dyn = take(m * nConv * nKvHeads);
        b.ctrlG = take(m * 64);
        b.ctrlInv = take(2 * m);
        b.logits = take(m * vocab);
        b.gx = take(nPairs * H);
        b.gout = take(nPairs * H);
        b.gact = take(nPairs * expertFf);
        b.gwgt = take(nPairs);
        b.wgt = take(m * nUsed);
        b.gtok = takeI(nPairs);
        b.gexp = takeI(nPairs);
        b.gpos = takeI(m * nUsed);
        b.gcount = takeI(nExp + 1);
        b.goff = takeI(nExp + 1);
        b.idx = takeI(m * nUsed);
        // Pinned staging for the per-chunk embedding upload: a non-blocking
        // compute stream has no implicit ordering with the legacy stream, so the
        // upload has to be an async copy on the compute stream itself.
        if (cudaHostAlloc(reinterpret_cast<void **>(&b.embPinned),
                          static_cast<size_t>(2) * m * H * sizeof(float),
                          cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            b.embPinned = nullptr;
            return (err = "cudaHostAlloc(batched embedding staging) failed",
                    false);
        }
        b.embPinnedIdx = 0;
        // Host mirror of the per-expert live-row counts: one page is plenty for
        // the 128-expert router and it is pinned because a per-layer D2H copy
        // must not be staged through the driver.
        if (cudaHostAlloc(reinterpret_cast<void **>(&b.countsPinned),
                          static_cast<size_t>(nExp) * sizeof(int32_t),
                          cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            b.countsPinned = nullptr;
            return (err = "cudaHostAlloc(expert counts) failed", false);
        }
        if (cudaHostAlloc(reinterpret_cast<void **>(&b.goffPinned),
                          static_cast<size_t>(nExp + 1) * sizeof(int32_t),
                          cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            b.goffPinned = nullptr;
            return (err = "cudaHostAlloc(expert offsets) failed", false);
        }
        b.ready = true;
        // The arena is one allocation, so a short `need` would silently let the
        // last buffers run past it into the int8 GEMV scratch.
        const uint64_t usedFloats = static_cast<uint64_t>(cur - b.arena);
        if (usedFloats > need)
            return (err = "batched prefill arena overflow: " +
                          std::to_string(usedFloats) + " > " +
                          std::to_string(need) + " floats",
                    false);
        std::fprintf(stderr,
                     "[inkling] batched prefill scratch: %.1f MiB (m=%u, "
                     "pairs=%llu, %llu/%llu floats, free before %llu MiB)\n",
                     double(b.bytes) / 1048576.0, kBatchM,
                     static_cast<unsigned long long>(nPairs),
                     static_cast<unsigned long long>(usedFloats),
                     static_cast<unsigned long long>(need),
                     static_cast<unsigned long long>(freeBefore >> 20));
        size_t freeAfter = 0;
        cudaMemGetInfo(&freeAfter, &totalBytes);
        std::fprintf(stderr,
                     "[inkling] VRAM after scratch: used %.0f MiB of %.0f MiB "
                     "(free %llu MiB)\n",
                     double(totalBytes - freeAfter) / 1048576.0,
                     double(totalBytes) / 1048576.0,
                     static_cast<unsigned long long>(freeAfter >> 20));
        return true;
    }

    void InklingGPU::Impl::dumpBatchPrelude(const char *dir, const float *dev,
                                            uint32_t m, cudaStream_t s) {
        batchDump(dir, "bt-prelude", dev, m * hidden, s);
    }

    // -------------------------------------------------------------------------
    // The batched prefill graph.
    // -------------------------------------------------------------------------
    bool InklingGPU::forwardBatch(const Model &model,
                                  const std::vector<int32_t> &tokens,
                                  bool allLogits, float *logits,
                                  std::string &err) {
        err.clear();
        if (!available_ || impl_ == nullptr) {
            err = "CUDA device unavailable";
            return false;
        }
        const uint32_t m = static_cast<uint32_t>(tokens.size());
        if (m == 0 || m > Impl::kBatchM) {
            err = "forwardBatch token count " + std::to_string(m) +
                  " outside 1.." + std::to_string(Impl::kBatchM);
            return false;
        }
        if (logits == nullptr) {
            err = "null logits buffer";
            return false;
        }
        const ModelConfig &cfg = model.config();
        // Geometry the batched attention kernel and the [H, m] graph assume.
        // Refused *before* any state changes, so the caller can fall back to the
        // token-major path safely.
        if (model.config().headDim != 128u || cfg.numAttentionHeads > 32u ||
            cfg.relativeExtent > 4096u || cfg.numAttentionHeads %
                                                            cfg.numKVHeads != 0) {
            err = "batched prefill needs headDim 128, <=32 heads, extent<=4096";
            return false;
        }
        if (!ensureGraph(model, err)) return false;
        if (!impl_->ensureBatch(err)) return false;
        Impl *im = impl_;
        auto &b = im->batch;
        const uint32_t H = im->hidden;
        const uint32_t qDim = im->nHeads * im->headDim;
        const float qkScale = 1.0f / std::sqrt(float(im->headDim));
        const float postScale = 1.0f / std::sqrt(float(im->nConv + 1));
        cudaStream_t s = im->stream;
        const char *dumpDir = std::getenv("TINYCODER_INKLING_BATCHED_DUMP");
        if (im->kvPos < 0 ||
            static_cast<uint64_t>(im->kvPos) + m > im->maxSeq) {
            err = "context window exceeded: pos " + std::to_string(im->kvPos) +
                  " + " + std::to_string(m) + " > " +
                  std::to_string(im->maxSeq);
            return false;
        }
        const uint32_t pos0 = static_cast<uint32_t>(im->kvPos);
        if (std::getenv("TINYCODER_INKLING_BATCHDBG") != nullptr)
            std::fprintf(stderr,
                         "[batch] pos0=%u m=%u first=%d last=%d\n", pos0, m,
                         tokens.front(), tokens.back());

        auto CK = [&]() {
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                err = std::string("CUDA error: ") + cudaGetErrorString(e);
                return false;
            }
            return true;
        };

        BatchProf bp;
        bp.begin();

        // ---- embeddings + token norm ----------------------------------------
        {
            // Upload into a pinned ring slot with an ASYNC copy on the compute
            // stream.  A plain cudaMemcpy would run on the legacy default stream,
            // which has no implicit ordering with a cudaStreamNonBlocking stream:
            // its device-side DMA may still be in flight when the token-norm
            // kernel reads this buffer, and a pageable source makes that window
            // real (CUDA only guarantees the copy was staged).  The symptom was
            // rows 12..15 of a 16-token chunk holding the PREVIOUS chunk's
            // embeddings, i.e. the last tokens of the chunk.
            float *pin = b.embPinned +
                         static_cast<size_t>(b.embPinnedIdx % 2u) * m * H;
            ++b.embPinnedIdx;
            for (uint32_t i = 0; i < m; ++i) {
                std::vector<float> row = model.quantizedEmbeddings_.getRow(
                        static_cast<uint32_t>(tokens[i]));
                if (row.size() != H) {
                    err = "bad embedding row for token " +
                          std::to_string(tokens[i]);
                    return false;
                }
                std::memcpy(pin + static_cast<size_t>(i) * H, row.data(),
                            H * sizeof(float));
            }
            const cudaError_t ue =
                    cudaMemcpyAsync(b.emb, pin,
                                    static_cast<size_t>(m) * H * sizeof(float),
                                    cudaMemcpyHostToDevice, s);
            if (ue != cudaSuccess) {
                cudaGetLastError();
                err = std::string("batched embedding upload failed: ") +
                      cudaGetErrorString(ue);
                return false;
            }
        }
        kInklingRmsNormRows(b.emb, static_cast<const float *>(im->tokNorm), b.x,
                            H, m, kRmsEps, s);
        if (!CK()) return false;
        bp.mark(1, s);

        uint32_t nBlocks = 0;

        auto GE = [&](uint32_t type, const void *dw, const float *x, float *out,
                      uint32_t rows, uint32_t cols) -> bool {
            if (gemvBatchedAny(type, dw, x, out, rows, cols, m, s)) return true;
            err = "batched GEMV failed (type " + std::to_string(type) +
                  ", rows " + std::to_string(rows) + ", cols " +
                  std::to_string(cols) + ")";
            return false;
        };

        // The loop controller is the one stage whose naive batched form (one
        // block per rank, tokens looped in-kernel) does not reproduce the
        // decode kernel's dot bit-for-bit: the FMA chain contracts a few ULP
        // differently, which a router tie in a later pass turns into a discrete
        // trajectory change.  ctrlLaunches therefore uses controlDownExact/
        // controlMixExact -- the decode kernel's body with the token as a grid
        // dimension, one launch for the whole batch.  TINYCODER_INKLING_CTRL=
        // token restores the per-token decode launches for a bisect; =batched
        // selects the old one-block-per-rank form (faster, not exact).
        const bool ctrlToken = [] {
            const char *e = std::getenv("TINYCODER_INKLING_CTRL");
            return e != nullptr && std::strcmp(e, "token") == 0;
        }();
        const bool ctrlLoopBatch = [] {
            const char *e = std::getenv("TINYCODER_INKLING_CTRL");
            return e != nullptr && std::strcmp(e, "batched") == 0;
        }();
        auto ctrlLaunches = [&](const Impl::LayerDev &dev, const float *cur,
                                const float *msg, uint32_t c, float *out) {
            if (!ctrlToken && !ctrlLoopBatch) {
                // Exact batched form: one block per (rank|element, token), so
                // the arithmetic is the decode kernel's by construction.
                bp.mark(24, s);
                controlInvKernel<<<m, 256, 0, s>>>(cur, msg, H, b.ctrlInv);
                bp.mark(10, s);
                static const bool downOld =
                        std::getenv("TINYCODER_INKLING_CTRL_DOWN_OLD") != nullptr;
                if (H == 2304u && !downOld) {
                    controlDownWarpKernel<<<dim3((dev.ctrlRank + 7u) / 8u,
                                                     (m + 31u) / 32u),
                                                256, 0, s>>>(
                            cur, H, msg, H,
                            static_cast<const float *>(dev.ctrlDown[c]), b.ctrlG,
                            dev.ctrlRank, H, b.ctrlInv, m);
                } else {
                    controlDownExactKernel<<<dim3(dev.ctrlRank, m), 256, 0, s>>>(
                            cur, H, msg, H,
                            static_cast<const float *>(dev.ctrlDown[c]), b.ctrlG,
                            dev.ctrlRank, H, b.ctrlInv);
                }
                bp.mark(23, s);
                static const bool mixReg =
                        std::getenv("TINYCODER_INKLING_CTRL_MIX_OLD") == nullptr;
                if (mixReg && dev.ctrlRank == 64u && H == 2304u) {
                    constexpr uint32_t kTT = 128u;
                    controlMixRegKernel<<<dim3((H + 255u) / 256u,
                                               (m + kTT - 1u) / kTT),
                                          256, kTT * 64u * sizeof(float), s>>>(
                            cur, H, msg, H,
                            static_cast<const float *>(dev.ctrlGain[c]),
                            static_cast<const float *>(dev.ctrlForget[c]),
                            b.ctrlG, out, H, dev.ctrlRank, m);
                } else {
                    controlMixBlockedKernel<<<dim3((H + 63u) / 64u,
                                                   (m + 127u) / 128u),
                                              256, 0, s>>>(
                            cur, H, msg, H,
                            static_cast<const float *>(dev.ctrlGain[c]),
                            static_cast<const float *>(dev.ctrlForget[c]),
                            b.ctrlG, out, H, dev.ctrlRank, m);
                }
                bp.mark(24, s);
                return;
            }
            if (ctrlToken) {
                for (uint32_t t = 0; t < m; ++t) {
                    kInklingControlDown(
                            cur + static_cast<size_t>(t) * H,
                            msg + static_cast<size_t>(t) * H,
                            static_cast<const float *>(dev.ctrlDown[c]),
                            b.ctrlG + static_cast<size_t>(t) * dev.ctrlRank,
                            dev.ctrlRank, H, s);
                    kInklingControlMix(
                            cur + static_cast<size_t>(t) * H,
                            msg + static_cast<size_t>(t) * H,
                            static_cast<const float *>(dev.ctrlGain[c]),
                            static_cast<const float *>(dev.ctrlForget[c]),
                            b.ctrlG + static_cast<size_t>(t) * dev.ctrlRank,
                            out + static_cast<size_t>(t) * H, H, dev.ctrlRank, s);
                }
                return;
            }
            kInklingControlDownBatch(cur, H, msg, H,
                                     static_cast<const float *>(dev.ctrlDown[c]),
                                     b.ctrlG, dev.ctrlRank, H, m, s);
            kInklingControlMixBatch(cur, H, msg, H,
                                    static_cast<const float *>(dev.ctrlGain[c]),
                                    static_cast<const float *>(dev.ctrlForget[c]),
                                    b.ctrlG, out, H, dev.ctrlRank, m, s);
        };

        // =====================================================================
        // One physical block over the whole chunk.  Mirrors runLayer in
        // InklingForward.cu stage for stage.
        // =====================================================================
        auto runLayer = [&](uint32_t physIdx, uint32_t slot,
                            const float *dMessage) -> bool {
            const auto &w = model.layers_[physIdx];
            const auto &ink = w.inkling;
            const auto &dev = im->layers[physIdx];
            const bool isSwa = physIdx < cfg.slidingWindowPattern.size()
                                       ? cfg.slidingWindowPattern[physIdx] != 0
                                       : true;
            float *hist = im->convState + static_cast<size_t>(slot) *
                                                 im->histLen * im->sconvChannels;

            // inpSA = x (before attention)
            if (!bcopyD2D(b.inpSA, b.x, static_cast<uint64_t>(m) * H * 4, s))
                return false;

            // ---- attention inputs: depth mix, then optional message control --
            float *pre[3];
            if (dev.attnResQuery != nullptr) {
                const uint32_t nq = isSwa ? 1 : 3;
                kInklingMixCurrentBatch(b.blockBank, nBlocks, b.x, H,
                                        static_cast<const float *>(dev.attnResQuery),
                                        nq, H, b.mixAttn, m * H, m, s);
                pre[0] = b.mixAttn;
                pre[1] = isSwa ? b.mixAttn : b.mixAttn + m * H;
                pre[2] = isSwa ? b.mixAttn : b.mixAttn + 2 * m * H;
            } else {
                pre[0] = pre[1] = pre[2] = b.x;
            }
            if (dumpDir != nullptr) dumpStageSet(dumpDir, slot, "mix", pre[0], H,
                                                 m, s);
            bp.mark(10, s);
            if (dMessage != nullptr) {
                float *outs[3] = {b.pre0, b.pre1, b.pre2};
                for (uint32_t c = 0; c < 3; ++c) {
                    const float *msg = dMessage + static_cast<size_t>(c) * m * H;
                    ctrlLaunches(dev, pre[c], msg, c, outs[c]);
                    pre[c] = outs[c];
                }
            }

            bp.mark(11, s);
            if (dumpDir != nullptr) dumpStageSet(dumpDir, slot, "pre", pre[0], H,
                                                 m, s);
            if (dumpDir != nullptr && dMessage != nullptr)
                dumpStageSet(dumpDir, slot, "ctg", b.ctrlG, dev.ctrlRank, m, s);
            if (dumpDir != nullptr && dMessage != nullptr) {
                dumpStageSet(dumpDir, slot, "msg", dMessage, H, m, s);
                const char *only = std::getenv("TINYCODER_INKLING_STAGE_SLOT");
                if (only == nullptr ||
                    static_cast<uint32_t>(std::atoi(only)) == slot) {
                    char tag[64];
                    std::snprintf(tag, sizeof(tag), "msgall-s%u", slot);
                    batchDump(dumpDir, tag, dMessage, 4 * m * H, s);
                }
            }
            if (isSwa) {
                kInklingRmsNormRows(pre[0],
                                    static_cast<const float *>(dev.rmsNormAttn),
                                    b.nq, H, m, kRmsEps, s);
                kInklingRmsNormRows(pre[1],
                                    static_cast<const float *>(dev.rmsNormAttn),
                                    b.nk, H, m, kRmsEps, s);
                kInklingRmsNormRows(pre[2],
                                    static_cast<const float *>(dev.rmsNormAttn),
                                    b.nv, H, m, kRmsEps, s);
            } else {
                kInklingRmsNormRows(pre[0],
                                    static_cast<const float *>(dev.qInpNorm),
                                    b.nq, H, m, kRmsEps, s);
                kInklingRmsNormRows(pre[1],
                                    static_cast<const float *>(dev.kInpNorm),
                                    b.nk, H, m, kRmsEps, s);
                kInklingRmsNormRows(pre[2],
                                    static_cast<const float *>(dev.vInpNorm),
                                    b.nv, H, m, kRmsEps, s);
            }

            bp.mark(2, s);
            // ---- Q/K/V/rel/sink projections ---------------------------------
            if (!GE(w.attnQ.type, dev.attnQ, b.nq, b.q, qDim, H)) return false;
            if (!GE(w.attnK.type, dev.attnK, b.nk, b.kProj, im->kvDim, H))
                return false;
            if (!GE(w.attnV.type, dev.attnV, b.nv, b.vProj, im->kvDim, H))
                return false;
            if (!GE(ink.attnRel.type, dev.attnRel, b.nq, b.rel,
                    im->nHeads * im->nRel, H))
                return false;

            bp.mark(3, s);
            // ---- short convolutions (K at 0, V at kvDim in the history) -----
            const float convScale =
                    isSwa ? 1.0f / std::sqrt(float(im->nConv + 1))
                          : 1.0f / std::sqrt(float(2 * im->nConv + 1));
            const float *dyn = nullptr;
            if (!isSwa && dev.kConvProj != nullptr) {
                if (!GE(ink.attnKConvProj.type, dev.kConvProj, b.nk, b.dyn,
                        im->nConv * im->nKvHeads, H))
                    return false;
                dyn = b.dyn;
            }
            kInklingShortConvBatch(b.kProj, static_cast<const float *>(dev.kConvW),
                                   dyn, hist, b.k, im->kvDim, im->nConv,
                                   im->nKvHeads, true, convScale, m, s);
            dyn = nullptr;
            if (!isSwa && dev.vConvProj != nullptr) {
                if (!GE(ink.attnVConvProj.type, dev.vConvProj, b.nv, b.dyn,
                        im->nConv * im->nKvHeads, H))
                    return false;
                dyn = b.dyn;
            }
            kInklingShortConvBatch(
                    b.vProj, static_cast<const float *>(dev.vConvW), dyn,
                    hist + static_cast<size_t>(im->histLen) * im->kvDim, b.v,
                    im->kvDim, im->nConv, im->nKvHeads, true, convScale, m, s);

            bp.mark(4, s);
            bp.mark(12, s);
            // ---- clip / per-head value norm / rel clip / sink ----------------
            bp.mark(16, s);
            kInklingClipBatch(b.q, im->nHeads, im->headDim,
                              static_cast<const float *>(dev.attnQClip),
                              cfg.qkClip, m, s);
            bp.mark(17, s);
            kInklingClipBatch(b.k, im->nKvHeads, im->headDim, nullptr, cfg.qkClip,
                              m, s);
            bp.mark(18, s);
            kInklingRmsNormRows(b.v, nullptr, b.v, im->headDim,
                                m * im->nKvHeads, kRmsEps, s);
            bp.mark(19, s);
            kInklingClipBatch(b.rel, im->nHeads, im->nRel,
                              static_cast<const float *>(dev.attnRelClip),
                              cfg.relativeClip, m, s);
            bp.mark(20, s);
            if (!GE(ink.attnSinkProj.type, dev.sinkProj, b.nq, b.sink,
                    im->nHeads, H))
                return false;
            bp.mark(21, s);

            // ---- log-scaled positions (full attention only) ------------------
            if (!isSwa)
                kInklingTauScaleBatch(b.q, qDim, b.rel, im->nHeads * im->nRel,
                                      pos0, cfg.logScaleAlpha, cfg.logScaleFloor,
                                      m, s);
            bp.mark(22, s);

            // ---- relative bias table, K/V store, attention -------------------
            bp.mark(13, s);
            kInklingRelBiasBatch(b.rel, static_cast<const float *>(dev.relBias),
                                 b.sink, b.table, im->nHeads, im->nRel,
                                 im->extent, pos0, m, s);
            float *kSlot = im->kCache + static_cast<size_t>(slot) * im->maxSeq *
                                                im->kvDim;
            float *vSlot = im->vCache + static_cast<size_t>(slot) * im->maxSeq *
                                                im->kvDim;
            bp.mark(14, s);
            kInklingStoreKvBatch(b.k, b.v, kSlot, vSlot, pos0, im->kvDim, m, s);
            bp.mark(15, s);
            kInklingAttentionBatch(b.q, kSlot, vSlot, b.table, pos0, im->nHeads,
                                   im->nKvHeads, im->extent, qkScale, b.attnOut,
                                   m, cfg.slidingWindow, isSwa, s);

            if (dumpDir != nullptr) dumpStageSet(dumpDir, slot, "ao", b.attnOut,
                                                 qDim, m, s);
            bp.mark(5, s);
            // ---- attention gate + output projection + post conv --------------
            if (!GE(ink.attnGate.type, dev.attnGate, b.nq, b.gateProj, qDim, H))
                return false;
            bSigmoidMul(b.attnOut, b.gateProj, m * qDim, s);
            if (!GE(w.attnO.type, dev.attnO, b.attnOut, b.o, H, qDim))
                return false;
            kInklingShortConvBatch(
                    b.o, static_cast<const float *>(dev.attnPostConvW), nullptr,
                    hist + static_cast<size_t>(im->histLen) * (2 * im->kvDim),
                    b.postO, H, im->nConv, 1, true, postScale, m, s);

            bp.mark(6, s);
            // ---- block bank / residual --------------------------------------
            if (physIdx % im->blockSize == 0) {
                if (!bcopyD2D(b.blockBank +
                                      static_cast<size_t>(nBlocks) * m * H,
                              b.inpSA, static_cast<uint64_t>(m) * H * 4, s))
                    return false;
                ++nBlocks;
                if (!bcopyD2D(b.x, b.postO, static_cast<uint64_t>(m) * H * 4, s))
                    return false;
            } else {
                bCombine(b.x, b.postO, b.inpSA, m * H, 1.0f, 1.0f, s);
            }

            // ---- FFN: depth mix, optional message, norm, routed MoE ----------
            bp.mark(25, s);
            if (!bcopyD2D(b.ffnInp, b.x, static_cast<uint64_t>(m) * H * 4, s))
                return false;
            kInklingMixCurrentBatch(b.blockBank, nBlocks, b.x, H,
                                    static_cast<const float *>(dev.ffnResQuery),
                                    1, H, b.mixH, m * H, m, s);
            if (dMessage != nullptr) {
                const float *msg = dMessage + static_cast<size_t>(3) * m * H;
                ctrlLaunches(dev, b.mixH, msg, 3, b.normed);
                kInklingRmsNormRows(b.normed,
                                    static_cast<const float *>(dev.rmsNormFFN),
                                    b.normed, H, m, kRmsEps, s);
            } else {
                kInklingRmsNormRows(b.mixH,
                                    static_cast<const float *>(dev.rmsNormFFN),
                                    b.normed, H, m, kRmsEps, s);
            }

            if (dumpDir != nullptr) dumpStageSet(dumpDir, slot, "nrm", b.normed, H,
                                                 m, s);
            // ---- router ------------------------------------------------------
            bp.mark(22, s);
            if (!GE(w.ffnGateInp.type, dev.router, b.normed, b.router,
                    im->nExperts, H))
                return false;
            bp.mark(23, s);
            kInklingRouterBatch(b.router, static_cast<const float *>(dev.probsBias),
                                b.idx, b.wgt, im->nExperts, im->nUsed,
                                cfg.expertWeightsScale, m, s);
            bp.mark(26, s);
            if (dumpDir != nullptr) dumpStageSet(dumpDir, slot, "rou", b.router,
                                                 im->nExperts, m, s);

            bp.mark(7, s);
            // ---- routed experts ---------------------------------------------
            // Exact path: when the decode graph reaches the engine's int8
            // expert GEMV for this tensor (gate/up with a supported type), a
            // grouped float decode would be a different trajectory, so the
            // gate/up projection runs per token through the same call.  The
            // down projection always takes the float decoder in the token path
            // (int8GemvExpert refuses the 2*ff activation stride), so it stays
            // grouped -- bit-exact and amortised.
            // Diagnostic: run the whole routed MoE exactly as the decode graph
            // does (per token, per slot), to isolate the grouped path.
            static const bool moeToken = std::getenv("TINYCODER_INKLING_MOE") !=
                                         nullptr &&
                                         std::strcmp(std::getenv("TINYCODER_INKLING_MOE"),
                                                     "token") == 0;
            if (moeToken) {
                gatherScalesBatchKernel<<<1, 256, 0, s>>>(
                        b.gateUpRank, b.downRank, b.idx,
                        static_cast<const float *>(dev.gateUpScale),
                        static_cast<const float *>(dev.downScale), m * im->nUsed);
                for (uint32_t t = 0; t < m; ++t) {
                    const float *dn = b.normed + static_cast<size_t>(t) * H;
                    int32_t *ix = b.idx + static_cast<size_t>(t) * im->nUsed;
                    float *gu = b.gateUp + static_cast<size_t>(t) * im->nUsed *
                                                   2 * im->expertFf;
                    float *eo = b.expertOut + static_cast<size_t>(t) * im->nUsed * H;
                    if (!gemvExpertToken(w.ffnGateUpExps.type, dev.gateUp, dn, 0,
                                         gu, 2 * im->expertFf, H, ix, im->nUsed,
                                         s)) {
                        err = "expert gate_up GEMV failed (token MoE)";
                        return false;
                    }
                    kInklingSituAll(gu, im->nUsed, im->expertFf,
                                    b.gateUpRank + static_cast<size_t>(t) * im->nUsed,
                                    cfg.situBeta, cfg.situLinearBeta, s);
                    if (!gemvExpertToken(w.ffnDownExps.type, dev.down, gu,
                                         2 * im->expertFf, eo, H, im->expertFf, ix,
                                         im->nUsed, s)) {
                        err = "expert down GEMV failed (token MoE)";
                        return false;
                    }
                    kInklingMoeAccum(eo, b.wgt + static_cast<size_t>(t) * im->nUsed,
                                     b.downRank + static_cast<size_t>(t) * im->nUsed,
                                     b.moeOut + static_cast<size_t>(t) * H,
                                     im->nUsed, H, s);
                }
            }
            // Which grouped path this tensor takes.  Decided once, because it
            // is a property of the (type, env) pair and not of the chunk.
            enum MoeMode { MoeMma, MoeSliced, MoeProbe, MoeGrouped };
            MoeMode moeMode = MoeGrouped;
            if (!moeToken) {
                if (fastMode()) {
                    moeMode = MoeGrouped;
                } else if (expertMmaGate() &&
                           inklingExpertMmaSupported(w.ffnGateUpExps.type)) {
                    // Grouped int8 MMQ for the IQ2 gate/up experts.
                    moeMode = MoeMma;
                } else if (expertInt8GateUp(w.ffnGateUpExps.type)) {
                    static const bool sliceOn = [] {
                        const char *e =
                                std::getenv("TINYCODER_INKLING_EXPERT_SLICE");
                        return e == nullptr || e[0] != '0';
                    }();
                    moeMode = sliceOn ? MoeSliced : MoeProbe;
                }
                if (moeMode == MoeProbe) {
                    // Probe the first token: true means the int8 expert kernel
                    // owns this tensor in the decode path, so run every token
                    // through it.  A refusal drops to the grouped float decoder,
                    // which is what the decode path uses for that type.
                    if (!gemvExpertToken(w.ffnGateUpExps.type, dev.gateUp,
                                         b.normed, 0, b.gateUp, 2 * im->expertFf, H,
                                         b.idx, im->nUsed, s)) {
                        cudaGetLastError();
                        moeMode = MoeGrouped;
                    }
                }
            }
            if (!moeToken) {
                // ---- routed MoE over the outer batch -------------------------
                // The m*nUsed routed (token, expert) pairs are counting-sorted
                // by expert into a COMPACT pair array: expert e owns
                // goff[e]..goff[e]+counts[e].  The compact form is what lets one
                // gather cover the whole outer batch: the padded
                // [nExperts][capacity] form must be sized for the worst case
                // (every token on one expert), 1.5 GiB at m=512, while the
                // compact form is exactly m*nUsed rows (~94 MB of scratch).
                //
                // The gain is on the expert side: one gather per pass decodes
                // each expert's weight matrix once for all m tokens instead of
                // once per 128-token chunk, and the routed MoE is ~65% of
                // prefill and decode-ALU bound, not bandwidth bound.  It also
                // matters more the more skewed the router is: this model's
                // router puts >128 of 512 tokens on one expert, so a
                // capacity-128 gather has to run four times.
                //
                // Arithmetic is unchanged: a token's routed dot and the
                // accumulation order over its nUsed slots are its own
                // (moeAccumGrouped walks u = 0..nUsed-1 per token), and every
                // pair appears in exactly one gather.
                const uint32_t nPairs = m * im->nUsed;
                const uint32_t rowsPer = 2 * im->expertFf;
                auto moeBuild = [&]() {
                    kInklingMoeGroupBuild(b.idx, b.wgt, b.gcount, b.goff, b.gtok,
                                          b.gpos, b.gwgt, b.gexp, m, im->nUsed,
                                          im->nExperts, s);
                };
                auto moeCompute = [&]() -> bool {
                    bp.mark(29, s);
                    if (moeMode == MoeMma) {
                        kInklingGatherExpertX(b.normed, b.gtok, nPairs, b.gx,
                                              H, s);
                        bp.mark(30, s);
                        if (!inklingExpertMmaGroupedOffsets(
                                    dev.gateUp, b.gx, b.gout, rowsPer, H,
                                    b.gcount, b.goff, nPairs, im->nExperts,
                                    w.ffnGateUpExps.type, s)) {
                            err = "grouped expert gate_up MMA failed (type " +
                                  std::to_string(w.ffnGateUpExps.type) + ")";
                            return false;
                        }
                        bp.mark(31, s);
                        kInklingSituGrouped(
                                b.gout, b.gact, b.gexp,
                                static_cast<const float *>(dev.gateUpScale),
                                nPairs, im->expertFf, cfg.situBeta,
                                cfg.situLinearBeta, s);
                        bp.mark(32, s);
                    } else if (moeMode == MoeSliced) {
                        kInklingGatherExpertX(b.normed, b.gtok, nPairs, b.gx,
                                              H, s);
                        // The slice count per expert is a host launch decision,
                        // so the live-row counts come back once per layer.  The
                        // copy runs on the compute stream and the sync only
                        // waits for this layer's group build.
                        if (cudaMemcpyAsync(b.countsPinned, b.gcount,
                                            static_cast<size_t>(im->nExperts) *
                                                    sizeof(int32_t),
                                            cudaMemcpyDeviceToHost, s) !=
                                            cudaSuccess ||
                            cudaStreamSynchronize(s) != cudaSuccess) {
                            cudaGetLastError();
                            err = "expert group-count readback failed";
                            return false;
                        }
                        const uint32_t qk = inklingTypeQk(w.ffnGateUpExps.type);
                        const uint32_t bb =
                                inklingTypeBlockBytes(w.ffnGateUpExps.type);
                        if (qk == 0 || bb == 0) {
                            err = "unsupported expert gate_up quant type";
                            return false;
                        }
                        const uint64_t rowBytes =
                                ((static_cast<uint64_t>(H) + qk - 1) / qk) * bb;
                        const uint8_t *wbase =
                                static_cast<const uint8_t *>(dev.gateUp);
                        // The slab base is a device value too, so it comes
                        // back with the counts (both are tiny and pinned).
                        int32_t *goffHost = b.goffPinned;
                        if (cudaMemcpyAsync(goffHost, b.goff,
                                            static_cast<size_t>(im->nExperts + 1) *
                                                    sizeof(int32_t),
                                            cudaMemcpyDeviceToHost, s) !=
                                cudaSuccess ||
                            cudaStreamSynchronize(s) != cudaSuccess) {
                            cudaGetLastError();
                            err = "expert group-offset readback failed";
                            return false;
                        }
                        for (uint32_t e = 0; e < im->nExperts; ++e) {
                            const int32_t cnt = b.countsPinned[e];
                            if (cnt <= 0) continue;
                            const uint32_t off =
                                    static_cast<uint32_t>(goffHost[e]);
                            const float *xg = b.gx + static_cast<size_t>(off) * H;
                            float *og = b.gout +
                                        static_cast<size_t>(off) * rowsPer;
                            const void *we =
                                    wbase + static_cast<uint64_t>(e) * rowsPer *
                                                    rowBytes;
                            for (uint32_t s0 = 0;
                                 s0 < static_cast<uint32_t>(cnt); s0 += 4) {
                                const uint32_t mm =
                                        std::min(4u, static_cast<uint32_t>(cnt) -
                                                             s0);
                                if (!int8Gemv(w.ffnGateUpExps.type, we,
                                              xg + static_cast<size_t>(s0) * H,
                                              og + static_cast<size_t>(s0) * rowsPer,
                                              mm, rowsPer, H, s)) {
                                    err = "expert gate_up int8 slice failed (type " +
                                          std::to_string(w.ffnGateUpExps.type) +
                                          ")";
                                    return false;
                                }
                            }
                        }
                        kInklingSituGrouped(
                                b.gout, b.gact, b.gexp,
                                static_cast<const float *>(dev.gateUpScale),
                                nPairs, im->expertFf, cfg.situBeta,
                                cfg.situLinearBeta, s);
                    } else if (moeMode == MoeProbe) {
                        for (uint32_t t = 0; t < m; ++t) {
                            if (!gemvExpertToken(
                                        w.ffnGateUpExps.type, dev.gateUp,
                                        b.normed + static_cast<size_t>(t) * H, 0,
                                        b.gateUp + static_cast<size_t>(t) *
                                                           im->nUsed * rowsPer,
                                        rowsPer, H,
                                        b.idx + static_cast<size_t>(t) * im->nUsed,
                                        im->nUsed, s)) {
                                err = "expert gate_up GEMV failed (per-token "
                                      "exact)";
                                return false;
                            }
                        }
                        kInklingSituGather(
                                b.gateUp, b.idx, b.gpos, b.gact,
                                static_cast<const float *>(dev.gateUpScale),
                                im->expertFf, im->nUsed, m, cfg.situBeta,
                                cfg.situLinearBeta, s);
                    } else {
                        kInklingGatherExpertX(b.normed, b.gtok, nPairs, b.gx,
                                              H, s);
                        if (!inklingGemvBatchedGroupedOffsetsAsync(
                                    w.ffnGateUpExps.type, dev.gateUp, b.gx,
                                    b.gout, rowsPer, H, b.gcount, b.goff,
                                    im->nExperts, s)) {
                            err = "grouped expert gate_up GEMV failed (type " +
                                  std::to_string(w.ffnGateUpExps.type) + ")";
                            return false;
                        }
                        kInklingSituGrouped(
                                b.gout, b.gact, b.gexp,
                                static_cast<const float *>(dev.gateUpScale),
                                nPairs, im->expertFf, cfg.situBeta,
                                cfg.situLinearBeta, s);
                    }
                    if (moeMode != MoeMma) bp.mark(8, s);
                    // Q2_0 expert down is 18 B / 64 weights and runs through
                    // the int8 MMQ when TINYCODER_INKLING_Q2_0_MMA=1; any
                    // refusal (unsupported shape, launch failure) falls through
                    // to the grouped float decoder, which is the default.
                    bool downOk = false;
                    if (expertMmaDown() &&
                        inklingExpertMmaSupported(w.ffnDownExps.type))
                        downOk = inklingExpertMmaGroupedOffsets(
                                dev.down, b.gact, b.gx, H, im->expertFf,
                                b.gcount, b.goff, nPairs, im->nExperts,
                                w.ffnDownExps.type, s);
                    if (!downOk && q20MmaEnabled() && w.ffnDownExps.type == 42)
                        downOk = int8GemmQ2_0GroupedOffsets(
                                dev.down, b.gact, b.gx, b.gcount, b.goff, nPairs,
                                H, im->expertFf, im->nExperts, s);
                    if (!downOk && !inklingGemvBatchedGroupedOffsetsAsync(
                                            w.ffnDownExps.type, dev.down, b.gact,
                                            b.gx, H, im->expertFf, b.gcount,
                                            b.goff, im->nExperts, s)) {
                        err = "grouped expert down GEMV failed (type " +
                              std::to_string(w.ffnDownExps.type) + ")";
                        return false;
                    }
                    bp.mark(33, s);
                    kInklingMoeAccumGrouped(
                            b.gx, b.idx, b.gpos, b.wgt,
                            static_cast<const float *>(dev.downScale), b.moeOut,
                            im->nUsed, H, m, s);
                    bp.mark(34, s);
                    bp.mark(9, s);
                    return true;
                };
                bp.mark(28, s);
                moeBuild();
                if (!moeCompute()) return false;
            }
            if (dumpDir != nullptr) dumpStageSet(dumpDir, slot, "moe", b.moeOut, H,
                                                 m, s);
            bp.mark(9, s);
            // ---- FFN post conv + residual -----------------------------------
            kInklingShortConvBatch(
                    b.moeOut, static_cast<const float *>(dev.ffnPostConvW),
                    nullptr,
                    hist + static_cast<size_t>(im->histLen) *
                                   (2 * im->kvDim + H),
                    b.convH, H, im->nConv, 1, true, postScale, m, s);
            bCombine(b.x, b.ffnInp, b.convH, m * H, 1.0f, 1.0f, s);

            if (dumpDir != nullptr) {
                for (uint32_t t = 0; t < m; ++t) {
                    char tag[64];
                    std::snprintf(tag, sizeof(tag), "bt-t%u-s%u", t, slot);
                    batchDump(dumpDir, tag, b.x + static_cast<size_t>(t) * H, H, s);
                }
                char tag[64];
                std::snprintf(tag, sizeof(tag), "bt-idx-s%u", slot);
                batchDumpI32(dumpDir, tag, b.idx, m * im->nUsed, s);
            }
            bp.mark(1, s);
            return CK();
        };

        // ---- prelude --------------------------------------------------------
        for (uint32_t il = 0; il < cfg.loopStart; ++il)
            if (!runLayer(il, il, nullptr)) return false;
        const uint32_t preludeBlocks = nBlocks;
        if (!bcopyD2D(b.prelude, b.x, static_cast<uint64_t>(m) * H * 4, s)) {
            err = "prelude copy failed";
            return false;
        }
        if (dumpDir != nullptr) im->dumpBatchPrelude(dumpDir, b.x, m, s);

        // ---- three loop passes ---------------------------------------------
        for (uint32_t pass = 0; pass < cfg.loopCount; ++pass) {
            uint32_t msgStates = 0;
            if (pass > 0) {
                const uint32_t skip = preludeBlocks + 1;
                const uint32_t nBank = nBlocks > skip ? nBlocks - skip : 0;
                // The message bank is the previous pass's bank rows after its
                // first core block, compacted to [0, nBank), plus that pass's
                // output as the last state (the token path's dBankSnap layout).
                if (!bcopyD2D(b.bankSnap,
                              b.blockBank + static_cast<size_t>(skip) * m * H,
                              static_cast<uint64_t>(nBank) * m * H * 4, s) ||
                    !bcopyD2D(b.passOut, b.x, static_cast<uint64_t>(m) * H * 4,
                              s)) {
                    err = "message bank copy failed";
                    return false;
                }
                nBlocks = preludeBlocks;
                if (!bcopyD2D(b.x, b.prelude, static_cast<uint64_t>(m) * H * 4,
                              s) ||
                    !bcopyD2D(b.bankSnap + static_cast<size_t>(nBank) * m * H,
                              b.passOut, static_cast<uint64_t>(m) * H * 4, s)) {
                    err = "message bank copy failed";
                    return false;
                }
                msgStates = nBank + 1;
            }
            for (uint32_t ip = cfg.loopStart; ip < cfg.loopEnd; ++ip) {
                const uint32_t slot =
                        pass == 0 ? ip
                                  : im->phys + (pass - 1) * im->core +
                                            (ip - cfg.loopStart);
                if (pass == 0) {
                    if (!runLayer(ip, slot, nullptr)) return false;
                    continue;
                }
                const void *mq = im->layers[ip].msgQuery;
                if (mq == nullptr)
                    mq = static_cast<const float *>(im->legacyMsg) +
                         static_cast<size_t>(ip - cfg.loopStart) * 4 * H;
                kInklingMixBatch(b.bankSnap, msgStates,
                                 static_cast<const float *>(mq), 4, H, b.msgMixed,
                                 m * H, m, s);
                if (!runLayer(ip, slot, b.msgMixed)) return false;
            }
        }

        // ---- postlude -------------------------------------------------------
        for (uint32_t il = cfg.loopEnd; il < im->phys; ++il)
            if (!runLayer(il, il, nullptr)) return false;

        // ---- output depth mix + norm + LM head + softcap ---------------------
        bp.mark(10, s);
        kInklingMixCurrentBatch(b.blockBank, nBlocks, b.x, H,
                                static_cast<const float *>(im->outResQuery), 1,
                                H, b.mixH, m * H, m, s);
        kInklingRmsNormRows(b.mixH, static_cast<const float *>(im->finalNorm),
                            b.normed, H, m, kRmsEps, s);
        bp.mark(27, s);
        // Only the last position's logits are needed when the caller did not
        // ask for all of them (the benchmark's prefill does not), and the LM
        // head is 200058 x 2304 IQ4_XS: computing all m positions reads 230 MB
        // of weights once per 16-token tile.  Restricting the batch to the last
        // token changes no arithmetic, only how many positions are evaluated.
        const uint32_t lmRows = allLogits ? m : 1u;
        const float *lmIn = b.normed + (allLogits ? 0u : static_cast<size_t>(m - 1u) * H);
        float *lmOut = b.logits +
                       (allLogits ? 0u
                                  : static_cast<size_t>(m - 1u) * im->vocab);
        if (!gemvBatchedAny(im->lmHeadType, im->lmHead, lmIn, lmOut, im->vocab,
                            H, lmRows, s)) {
            err = "batched LM head GEMV failed";
            return false;
        }
        if (cfg.finalLogitSoftcapping > 0.0f)
            bSoftcap(lmOut, lmRows * im->vocab, cfg.finalLogitSoftcapping, s);
        if (!CK()) return false;
        bp.mark(11, s);

        // ---- logits ---------------------------------------------------------
        if (allLogits) {
            if (cudaMemcpyAsync(logits, b.logits,
                                static_cast<size_t>(m) * im->vocab * sizeof(float),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess) {
                cudaGetLastError();
                err = "batched logits copy failed";
                return false;
            }
        } else {
            if (cudaMemcpyAsync(logits,
                                b.logits + static_cast<size_t>(m - 1) * im->vocab,
                                static_cast<size_t>(im->vocab) * sizeof(float),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess) {
                cudaGetLastError();
                err = "batched logits copy failed";
                return false;
            }
        }
        const cudaError_t e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) {
            cudaGetLastError();
            err = std::string("cudaStreamSynchronize failed: ") +
                  cudaGetErrorString(e);
            return false;
        }
        im->kvPos += static_cast<int>(m);
        bp.print();
        return true;
    }

#endif// __CUDA_ARCH__

    // ---- batched kernel entry points -----------------------------------------
    void kInklingMixCurrentBatch(const float *states, uint32_t nStates,
                                 const float *cur, uint32_t curStride,
                                 const float *query, uint32_t nQueries,
                                 uint32_t hidden, float *out,
                                 uint32_t outStride, uint32_t m, void *stream) {
        const uint32_t n = nStates + 1u;
        // Cap at 8 warps, not 32: one warp owns one state row, so the block's
        // live working set is nw * hidden floats.  With 32 warps that is 288 KiB
        // for the 50-row block bank, which thrashes the 128 KiB L1 between the
        // norm pass and the dot pass on the same row; 8 warps keeps 72 KiB and
        // makes the second read an L1 hit.  The per-state arithmetic is
        // unchanged (one warp still sums one row in lane-stride order).
        const uint32_t nw = n < 1u ? 1u : (n > 8u ? 8u : n);
        const dim3 grid(nQueries < 1u ? 1u : nQueries, m);
        mixCurBatchKernel<<<grid, nw * 32u, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
                states, nStates, m * hidden, cur, curStride, query, hidden, out,
                outStride);
    }

    void kInklingMixBatch(const float *states, uint32_t nStates,
                          const float *query, uint32_t nQueries, uint32_t hidden,
                          float *out, uint32_t outStride, uint32_t m,
                          void *stream) {
        // Same 8-warp cap as kInklingMixCurrentBatch: the row-per-warp working
        // set must stay inside L1 or the norm/dot/mix passes re-read it.
        const uint32_t nw = nStates < 1u ? 1u : (nStates > 8u ? 8u : nStates);
        const dim3 grid(nQueries < 1u ? 1u : nQueries, m);
        mixBatchKernel<<<grid, nw * 32u, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
                states, nStates, m * hidden, query, hidden, out, outStride);
    }

    void kInklingClipBatch(float *x, uint32_t rows, uint32_t len,
                           const float *scale, float threshold, uint32_t m,
                           void *stream) {
        clipBatchKernel<<<rows * m, 256, 0,
                          reinterpret_cast<cudaStream_t>(stream)>>>(
                x, rows, len, scale, threshold);
    }

    void kInklingRelBiasBatch(const float *r, const float *bias,
                              const float *sink, float *table, uint32_t heads,
                              uint32_t nRel, uint32_t extent, uint32_t pos0,
                              uint32_t m, void *stream) {
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        static const bool scatterDbg =
                std::getenv("TINYCODER_INKLING_RELB_SCATTER") != nullptr;
        if (nRel > 32u || scatterDbg) {
            relBiasScatterKernel<<<heads * m, 256, 0, s>>>(
                    r, bias, sink, table, heads, nRel, extent);
            return;
        }
        // Only distances this pass can still read are built: the attention
        // kernel's dist is min(pos - kp, extent) with pos < pos0 + m.
        const uint32_t maxDist = pos0 + m - 1u;
        const uint32_t dCount = (maxDist < extent ? maxDist : extent) + 1u;
        constexpr uint32_t kDT = 256;
        constexpr uint32_t kTT = 8;
        const dim3 grid((dCount + kDT - 1u) / kDT, (m + kTT - 1u) / kTT);
        const size_t shmem = static_cast<size_t>(nRel) * (kDT + 1u) * sizeof(float);
        relBiasTableKernel<kDT, kTT><<<grid, 256, shmem, s>>>(
                r, bias, sink, table, heads, nRel, extent, dCount, m);
    }

    void kInklingStoreKvBatch(const float *kcur, const float *vcur, float *kCache,
                              float *vCache, uint32_t pos0, uint32_t kvDim,
                              uint32_t m, void *stream) {
        const dim3 grid((kvDim + 255u) / 256u, m);
        storeKvBatchKernel<<<grid, 256, 0,
                             reinterpret_cast<cudaStream_t>(stream)>>>(
                kcur, vcur, kCache, vCache, pos0, kvDim);
    }

    void kInklingAttentionBatch(const float *q, const float *kCache,
                                const float *vCache, const float *table,
                                uint32_t pos0, uint32_t nHeads,
                                uint32_t nKvHeads, uint32_t extent,
                                float qkScale, float *out, uint32_t m,
                                uint32_t window, bool isSwa, void *stream) {
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        const uint32_t isS = isSwa ? 1u : 0u;
        // The decode path picks the warp count per QUERY from its own key count
        // (attentionWarps(pos+1)); a batched launch must therefore be split into
        // token ranges that share one warp count, or a chunk straddling an
        // attentionWarps step (nKeys 128/256/512) would partition the keys --
        // and the online-softmax combine order -- differently from the decode
        // path.  At most one step is crossed per 16-token chunk.
        const size_t shmem =
                (static_cast<size_t>(8u) * 128u + extent + 1u) * sizeof(float);
        // Query-tiled form: T tokens per block share one K/V read.  Opt-in
        // until the model gate says it is bit-identical (it is arithmetic-for-
        // arithmetic the same per token, so it should be).
        const char *tileEnv = std::getenv("TINYCODER_INKLING_ATTN_TILE");
        const bool tiled = tileEnv == nullptr || tileEnv[0] != '0';
        // Flash form for a real prefill batch: a different numeric trajectory
        // (sequential key accumulation per token), so it is opt-out via
        // TINYCODER_INKLING_ATTN_FLASH=0.  Small batches keep the exact form.
        const char *flashEnv = std::getenv("TINYCODER_INKLING_ATTN_FLASH");
        const bool flash = flashEnv == nullptr || flashEnv[0] != '0';
        if (flash && m >= 128u && nHeads == 3u * nKvHeads) {
            const uint32_t tt =
                    std::getenv("TINYCODER_INKLING_ATTN_FLASH_TT") != nullptr
                            ? static_cast<uint32_t>(std::atoi(
                                      std::getenv("TINYCODER_INKLING_ATTN_FLASH_TT")))
                            : 8u;
            const dim3 gf(nKvHeads, (m + tt - 1u) / tt);
            switch (tt) {
                case 8:
                    attentionFlashKernel<8, 16, 1><<<gf, 256, 0, s>>>(
                            q, kCache, vCache, table, pos0, nHeads, nKvHeads,
                            extent, qkScale, out, window, isS, m);
                    break;
                case 16:
                    attentionFlashKernel<16, 16, 2><<<gf, 256, 0, s>>>(
                            q, kCache, vCache, table, pos0, nHeads, nKvHeads,
                            extent, qkScale, out, window, isS, m);
                    break;
                default:
                    attentionFlashKernel<32, 16, 4><<<gf, 256, 0, s>>>(
                            q, kCache, vCache, table, pos0, nHeads, nKvHeads,
                            extent, qkScale, out, window, isS, m);
                    break;
            }
            return;
        }
        constexpr uint32_t kT = 8;
        for (uint32_t t0 = 0; t0 < m;) {
            const uint32_t w = attentionWarps(pos0 + t0 + 1u);
            uint32_t t1 = t0 + 1u;
            while (t1 < m && attentionWarps(pos0 + t1 + 1u) == w) ++t1;
            if (tiled) {
                const uint32_t nTiles = (t1 - t0 + kT - 1u) / kT;
                const dim3 g2(nHeads, nTiles);
                switch (w) {
                    case 2:
                        attentionTileKernel<2><<<g2, 64, 0, s>>>(
                                q, kCache, vCache, table, pos0, nHeads, nKvHeads,
                                extent, qkScale, out, window, isS, t0, t1);
                        break;
                    case 4:
                        attentionTileKernel<4><<<g2, 128, 0, s>>>(
                                q, kCache, vCache, table, pos0, nHeads, nKvHeads,
                                extent, qkScale, out, window, isS, t0, t1);
                        break;
                    default:
                        attentionTileKernel<8><<<g2, 256, 0, s>>>(
                                q, kCache, vCache, table, pos0, nHeads, nKvHeads,
                                extent, qkScale, out, window, isS, t0, t1);
                        break;
                }
                t0 = t1;
                continue;
            }
            const dim3 grid(nHeads, t1 - t0);
            switch (w) {
                case 2:
                    attentionBatchKernel<2><<<grid, 64, shmem, s>>>(
                            q, kCache, vCache, table, pos0, nHeads, nKvHeads,
                            extent, qkScale, out, window, isS, t0);
                    break;
                case 4:
                    attentionBatchKernel<4><<<grid, 128, shmem, s>>>(
                            q, kCache, vCache, table, pos0, nHeads, nKvHeads,
                            extent, qkScale, out, window, isS, t0);
                    break;
                default:
                    attentionBatchKernel<8><<<grid, 256, shmem, s>>>(
                            q, kCache, vCache, table, pos0, nHeads, nKvHeads,
                            extent, qkScale, out, window, isS, t0);
                    break;
            }
            t0 = t1;
        }
    }

    void kInklingTauScaleBatch(float *q, uint32_t qDim, float *rel,
                               uint32_t relDim, uint32_t pos0, float alpha,
                               float floor, uint32_t m, void *stream) {
        tauScaleBatchKernel<<<m, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                q, qDim, rel, relDim, pos0, alpha, floor);
    }

    void kInklingShortConvBatch(const float *cur, const float *weight,
                                const float *dynamic, float *history, float *out,
                                uint32_t channels, uint32_t nConv,
                                uint32_t nHeads, bool residual, float scale,
                                uint32_t m, void *stream) {
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        static const bool serialDbg =
                std::getenv("TINYCODER_INKLING_CONV_SERIAL") != nullptr;
        if (serialDbg) {
            shortConvBatchSerialKernel<<<(channels + 255u) / 256u, 256, 0, s>>>(
                    cur, weight, dynamic, history, out, channels, nConv, nHeads,
                    residual, scale, m);
            return;
        }
        const dim3 grid((channels + 255u) / 256u, (m + 63u) / 64u);
        switch (nConv) {
            case 2:
                shortConvTileKernel<2><<<grid, 256, 0, s>>>(
                        cur, weight, dynamic, history, out, channels, nHeads,
                        residual, scale, m);
                return;
            case 3:
                shortConvTileKernel<3><<<grid, 256, 0, s>>>(
                        cur, weight, dynamic, history, out, channels, nHeads,
                        residual, scale, m);
                return;
            case 4:
                shortConvTileKernel<4><<<grid, 256, 0, s>>>(
                        cur, weight, dynamic, history, out, channels, nHeads,
                        residual, scale, m);
                return;
            case 5:
                shortConvTileKernel<5><<<grid, 256, 0, s>>>(
                        cur, weight, dynamic, history, out, channels, nHeads,
                        residual, scale, m);
                return;
            default:
                shortConvBatchSerialKernel<<<(channels + 255u) / 256u, 256, 0, s>>>(
                        cur, weight, dynamic, history, out, channels, nConv,
                        nHeads, residual, scale, m);
                return;
        }
    }

    void kInklingRouterBatch(const float *logits, const float *bias, int32_t *idx,
                             float *wgt, uint32_t nExperts, uint32_t nUsed,
                             float scale, uint32_t m, void *stream) {
        routerBatchKernel<<<m, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                logits, bias, idx, wgt, nExperts, nUsed, scale);
    }

    void kInklingControlDownBatch(const float *cur, uint32_t curStride,
                                  const float *msg, uint32_t msgStride,
                                  const float *downW, float *g, uint32_t rank,
                                  uint32_t hidden, uint32_t m, void *stream) {
        controlDownBatchKernel<<<rank, 256, 0,
                                 reinterpret_cast<cudaStream_t>(stream)>>>(
                cur, curStride, msg, msgStride, downW, g, rank, hidden, m);
    }

    void kInklingControlMixBatch(const float *cur, uint32_t curStride,
                                 const float *msg, uint32_t msgStride,
                                 const float *gainW, const float *forgetW,
                                 const float *g, float *out, uint32_t hidden,
                                 uint32_t rank, uint32_t m, void *stream) {
        const dim3 grid((hidden + 255u) / 256u, m);
        controlMixBatchKernel<<<grid, 256, 0,
                                reinterpret_cast<cudaStream_t>(stream)>>>(
                cur, curStride, msg, msgStride, gainW, forgetW, g, out, hidden,
                rank);
    }

    void kInklingSituGrouped(float *gateUp, float *act, const int32_t *gexp,
                             const float *scale, uint32_t nPairs, uint32_t n,
                             float beta, float lb, void *stream) {
        situGroupedKernel<<<nPairs, 256, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
                gateUp, act, gexp, scale, n, beta, lb);
    }

    void kInklingSituGather(const float *gateUp, const int32_t *idx,
                            const int32_t *gpos, float *act,
                            const float *scale, uint32_t n, uint32_t nUsed,
                            uint32_t m, float beta, float lb, void *stream) {
        const dim3 grid(nUsed, m);
        situGatherKernel<<<grid, 256, 0,
                           reinterpret_cast<cudaStream_t>(stream)>>>(
                gateUp, idx, gpos, act, scale, n, nUsed, beta, lb);
    }

    void kInklingMoeGroupBuild(const int32_t *idx, const float *wgt,
                               int32_t *counts, int32_t *goff, int32_t *gtok,
                               int32_t *gpos, float *gwgt, int32_t *gexp,
                               uint32_t m, uint32_t nUsed, uint32_t nExperts,
                               void *stream) {
        moeGroupBuildKernel<<<1, 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                idx, wgt, counts, goff, gtok, gpos, gwgt, gexp, m, nUsed,
                nExperts);
    }

    void kInklingGatherExpertX(const float *x, const int32_t *gtok,
                               uint32_t nPairs, float *gx, uint32_t cols,
                               void *stream) {
        gatherExpertXKernel<<<nPairs, 256, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
                x, gtok, gx, cols);
    }

    void kInklingMoeAccumGrouped(const float *expertOut, const int32_t *idx,
                                 const int32_t *gpos, const float *wgt,
                                 const float *downScale, float *out,
                                 uint32_t nUsed, uint32_t hidden, uint32_t m,
                                 void *stream) {
        moeAccumGroupedKernel<<<m, 256, 0,
                                reinterpret_cast<cudaStream_t>(stream)>>>(
                expertOut, idx, gpos, wgt, downScale, out, nUsed, hidden);
    }

}// namespace tinycoder::gpu
