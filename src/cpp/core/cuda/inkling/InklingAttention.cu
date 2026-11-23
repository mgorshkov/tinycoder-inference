#include "gpu/InklingKernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace tinycoder::gpu {
    namespace {
        __device__ __forceinline__ uint32_t umin(uint32_t a, uint32_t b) {
            return a < b ? a : b;
        }

        __global__ void relBiasKernel(const float *r, const float *bias,
                                      const float *sink, float *table,
                                      uint32_t nRel, uint32_t extent) {
            const uint32_t h = blockIdx.x;
            for (uint32_t d = threadIdx.x; d <= extent; d += blockDim.x) {
                float v = 0.0f;
                if (d < extent) {
                    const float *br = bias + static_cast<size_t>(d) * nRel;
                    const float *rh = r + static_cast<size_t>(h) * nRel;
                    for (uint32_t k = 0; k < nRel; ++k) v += br[k] * rh[k];
                    v -= sink[h];
                }
                table[static_cast<size_t>(h) * (extent + 1) + d] = v;
            }
        }

        __device__ __forceinline__ float warpSum(float v);

        // One warp handles one state, and strided over the warps when there are
        // more states than warps.  The loop message mix has nBlocks-skip+1 = 18
        // states and the block-bank mix nBlocks+1 = 25, against 8/9 warps in
        // the launch, so a single warp-per-state round is not enough: the
        // earlier version dropped every state past the warp count and read
        // uninitialised scores for the rest, which is an O(20) logit error.
        // 24 physical blocks + the current x = 25 at most today; 64 leaves
        // headroom so a deeper loop config cannot silently overrun the shared
        // score arrays (the strided loop handles more states than warps).
        constexpr uint32_t kMaxMixStates = 64;

        // Launched with 1024 threads.  Without an explicit bound nvcc is free
        // to pick a register count a 1024-thread block cannot launch (observed:
        // 72 registers, i.e. 73,728 per block, which makes every forward fail
        // with "too many resources requested for launch" as soon as a rebuild
        // shifts codegen).  The bound caps the allocation at 64, so the launch
        // is valid whatever the surrounding translation unit does.
        __global__ void __launch_bounds__(1024, 1)
        mixKernel(const float *states, uint32_t nStates,
                  const float *query, uint32_t hidden, float *out) {
            __shared__ float inv[kMaxMixStates];
            __shared__ float score[kMaxMixStates];
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t w = threadIdx.x >> 5u;
            const uint32_t nwarps = blockDim.x >> 5u;
            const uint32_t q = blockIdx.x;
            const float *qv = query + static_cast<size_t>(q) * hidden;
            for (uint32_t s = w; s < nStates; s += nwarps) {
                const float *sv = states + static_cast<size_t>(s) * hidden;
                // Four loads in flight per lane, accumulated in the original
                // ascending i order (i, i+32, i+64, i+96) -- one dependent FMA
                // chain over 32-strided elements is latency-bound at six warps
                // per block, and a four-wide load prefix is the only change
                // that keeps the reduction bit-identical.
                float ss = 0.0f;
                uint32_t i = lane;
                for (; i + 96u < hidden; i += 128u) {
                    const float v0 = sv[i], v1 = sv[i + 32u],
                                v2 = sv[i + 64u], v3 = sv[i + 96u];
                    ss += v0 * v0;
                    ss += v1 * v1;
                    ss += v2 * v2;
                    ss += v3 * v3;
                }
                for (; i < hidden; i += 32u) ss += sv[i] * sv[i];
                ss = warpSum(ss);
                const float iv = rsqrtf(ss / float(hidden) + 1e-6f);
                float dot = 0.0f;
                i = lane;
                for (; i + 96u < hidden; i += 128u) {
                    const float v0 = sv[i], v1 = sv[i + 32u],
                                v2 = sv[i + 64u], v3 = sv[i + 96u];
                    dot += v0 * iv * qv[i];
                    dot += v1 * iv * qv[i + 32u];
                    dot += v2 * iv * qv[i + 64u];
                    dot += v3 * iv * qv[i + 96u];
                }
                for (; i < hidden; i += 32u) dot += sv[i] * iv * qv[i];
                dot = warpSum(dot);
                if (lane == 0) {
                    inv[s] = iv;
                    score[s] = dot * rsqrtf(float(hidden));
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
                for (uint32_t s = 0; s < nStates; ++s) mx = fmaxf(mx, score[s]);
                float sum = 0.0f;
                for (uint32_t s = 0; s < nStates; ++s) {
                    score[s] = expf(score[s] - mx);
                    sum += score[s];
                }
                for (uint32_t s = 0; s < nStates; ++s) score[s] /= sum;
            }
            __syncthreads();
            float *dst = out + static_cast<size_t>(q) * hidden;
            for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
                float acc = 0.0f;
                for (uint32_t s = 0; s < nStates; ++s)
                    acc += score[s] * states[static_cast<size_t>(s) * hidden + i] * inv[s];
                dst[i] = acc;
            }
        }

        __device__ __forceinline__ float warpSum(float v) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffffu, v, off);
            return v;
        }

        /// One block per query head, W warps splitting the key range.  The
        /// earlier version ran all 24 heads as warps of a single block, which
        /// left 67 of the 68 SMs idle at long context; a block per head spreads
        /// the work over 24 SMs, and W warps per block give the SM enough
        /// in-flight loads to hide the K/V latency.  The QK dot is computed once
        /// per key with an online softmax (the earlier version made three full
        /// passes over the cache), and the relative-bias row is staged in shared
        /// memory so the inner loop does not issue a global load per key.
        template <uint32_t W>
        __global__ void attentionWarpKernel(
                const float *__restrict__ q, const float *__restrict__ kCache,
                const float *__restrict__ vCache, const float *__restrict__ table,
                uint32_t pos, uint32_t firstKey, uint32_t nKeys, uint32_t nHeads,
                uint32_t nKvHeads, uint32_t extent, float qkScale,
                float *__restrict__ out, const int32_t *__restrict__ posDev,
                uint32_t window, uint32_t isSwa) {
            if (posDev != nullptr) {
                // CUDA-graph form: pos and the SWA window come from the device.
                pos = static_cast<uint32_t>(*posDev);
                firstKey = (isSwa && window > 0 && pos + 1 > window)
                                   ? pos + 1 - window
                                   : 0;
                nKeys = pos + 1 - firstKey;
            }
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t w = threadIdx.x >> 5u;
            const uint32_t hq = blockIdx.x;
            if (hq >= nHeads) return;
            const uint32_t group = nHeads / nKvHeads;
            const uint32_t hkv = hq / group;
            const float *qh = q + static_cast<size_t>(hq) * 128u;
            const float q0 = qh[lane], q1 = qh[lane + 32], q2 = qh[lane + 64],
                        q3 = qh[lane + 96];
            const uint32_t strideK = nKvHeads * 128u;
            const float *kb = kCache + static_cast<size_t>(hkv) * 128u + lane;
            const float *vb = vCache + static_cast<size_t>(hkv) * 128u + lane;
            const float *trow = table + static_cast<size_t>(hq) * (extent + 1u);

            extern __shared__ float smem[];
            float *sOut = smem;                 // [W][128]
            float *trowS = smem + W * 128u;     // [extent + 1]
            __shared__ float sMax[W];
            __shared__ float sSum[W];

            for (uint32_t i = threadIdx.x; i <= extent; i += 32u * W)
                trowS[i] = trow[i];
            __syncthreads();

            // Each warp owns a contiguous slice of the key range and folds it
            // into a partial online softmax.  The sink logit (0) is added once
            // in the combine, not once per warp.
            const uint32_t k0 = (nKeys * w) / W;
            const uint32_t k1 = (nKeys * (w + 1u)) / W;
            float mx = -INFINITY;
            float sum = 0.0f;
            float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
            for (uint32_t ki = k0; ki < k1; ++ki) {
                const uint32_t kp = firstKey + ki;
                const float *kh = kb + static_cast<size_t>(kp) * strideK;
                float d = q0 * kh[0] + q1 * kh[32] + q2 * kh[64] + q3 * kh[96];
                d = warpSum(d);
                const uint32_t dist = umin(pos - kp, extent);
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
                    if (sSum[j] > 0.0f) gsum += sSum[j] * expf(sMax[j] - M);
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
                float *oh = out + static_cast<size_t>(hq) * 128u;
                oh[lane] = a0;
                oh[lane + 32] = a1;
                oh[lane + 64] = a2;
                oh[lane + 96] = a3;
            }
        }

        /// Dispatch the warp count from the key count: 64 keys per warp keeps
        /// the per-warp loop long enough to amortise the combine.
        inline uint32_t attentionWarps(uint32_t nKeys) {
            uint32_t w = nKeys / 64u;
            if (w < 1u) w = 1u;
            if (w > 8u) w = 8u;
            if (w <= 2u) return 2u;
            if (w <= 4u) return 4u;
            return 8u;
        }

        template <uint32_t W>
        inline void launchAttentionWarp(
                const float *q, const float *kCache, const float *vCache,
                const float *table, uint32_t pos, uint32_t firstKey,
                uint32_t nKeys, uint32_t nHeads, uint32_t nKvHeads,
                uint32_t extent, float qkScale, float *out, cudaStream_t s,
                const int32_t *posDev = nullptr, uint32_t window = 0,
                uint32_t isSwa = 0) {
            const size_t shmem = (static_cast<size_t>(W) * 128u + extent + 1u) *
                                 sizeof(float);
            attentionWarpKernel<W><<<nHeads, 32u * W, shmem, s>>>(
                    q, kCache, vCache, table, pos, firstKey, nKeys, nHeads,
                    nKvHeads, extent, qkScale, out, posDev, window, isSwa);
        }

        /// Store the current token's K/V row.  Split out of the attention
        /// kernel because the per-head grid no longer shares a block-wide
        /// barrier: each head block must see the full row before it reads it.
        __global__ void storeKvKernel(const float *__restrict__ kcur,
                                      const float *__restrict__ vcur,
                                      float *__restrict__ kCache,
                                      float *__restrict__ vCache, uint32_t pos,
                                      uint32_t kvDim,
                                      const int32_t *__restrict__ posDev) {
            const uint32_t p = posDev != nullptr
                                       ? static_cast<uint32_t>(*posDev)
                                       : pos;
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= kvDim) return;
            const size_t off = static_cast<size_t>(p) * kvDim + i;
            kCache[off] = kcur[i];
            vCache[off] = vcur[i];
        }

        /// mixKernel with the current hidden state appended as one more state.
        __global__ void __launch_bounds__(1024, 1)
        mixCurKernel(const float *states, uint32_t nStates,
                     const float *cur, const float *query, uint32_t hidden,
                     float *out) {
            __shared__ float inv[kMaxMixStates];
            __shared__ float score[kMaxMixStates];
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t w = threadIdx.x >> 5u;
            const uint32_t nwarps = blockDim.x >> 5u;
            const uint32_t n = nStates + 1;
            const uint32_t q = blockIdx.x;
            const float *qv = query + static_cast<size_t>(q) * hidden;
            for (uint32_t s = w; s < n; s += nwarps) {
                const float *sv = s < nStates
                                          ? states + static_cast<size_t>(s) * hidden
                                          : cur;
                float ss = 0.0f;
                uint32_t i = lane;
                for (; i + 96u < hidden; i += 128u) {
                    const float v0 = sv[i], v1 = sv[i + 32u],
                                v2 = sv[i + 64u], v3 = sv[i + 96u];
                    ss += v0 * v0;
                    ss += v1 * v1;
                    ss += v2 * v2;
                    ss += v3 * v3;
                }
                for (; i < hidden; i += 32u) ss += sv[i] * sv[i];
                ss = warpSum(ss);
                const float iv = rsqrtf(ss / float(hidden) + 1e-6f);
                float dot = 0.0f;
                i = lane;
                for (; i + 96u < hidden; i += 128u) {
                    const float v0 = sv[i], v1 = sv[i + 32u],
                                v2 = sv[i + 64u], v3 = sv[i + 96u];
                    dot += v0 * iv * qv[i];
                    dot += v1 * iv * qv[i + 32u];
                    dot += v2 * iv * qv[i + 64u];
                    dot += v3 * iv * qv[i + 96u];
                }
                for (; i < hidden; i += 32u) dot += sv[i] * iv * qv[i];
                dot = warpSum(dot);
                if (lane == 0) {
                    inv[s] = iv;
                    score[s] = dot * rsqrtf(float(hidden));
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
                for (uint32_t s = 0; s < n; ++s) mx = fmaxf(mx, score[s]);
                float sum = 0.0f;
                for (uint32_t s = 0; s < n; ++s) {
                    score[s] = expf(score[s] - mx);
                    sum += score[s];
                }
                for (uint32_t s = 0; s < n; ++s) score[s] /= sum;
            }
            __syncthreads();
            float *dst = out + static_cast<size_t>(q) * hidden;
            for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
                float acc = 0.0f;
                for (uint32_t s = 0; s < n; ++s) {
                    const float *sv = s < nStates
                                              ? states + static_cast<size_t>(s) * hidden
                                              : cur;
                    acc += score[s] * sv[i] * inv[s];
                }
                dst[i] = acc;
            }
        }

        __global__ void attentionKernel(const float *q, const float *kCache,
                                        const float *vCache, const float *table,
                                        uint32_t pos, uint32_t firstKey,
                                        uint32_t nKeys, uint32_t headDim,
                                        uint32_t nHeads, uint32_t nKvHeads,
                                        uint32_t extent, float qkScale,
                                        float *out) {
            const uint32_t hq = threadIdx.x;
            if (hq >= nHeads) return;
            const uint32_t group = nHeads / nKvHeads;
            const uint32_t hkv = hq / group;
            const float *qh = q + static_cast<size_t>(hq) * headDim;
            float mx = 0.0f;
            for (uint32_t ki = 0; ki < nKeys; ++ki) {
                const uint32_t kp = firstKey + ki;
                const float *kh = kCache + static_cast<size_t>(kp) * nKvHeads * headDim +
                                  static_cast<size_t>(hkv) * headDim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < headDim; ++d) dot += qh[d] * kh[d];
                const uint32_t dist = umin(pos - kp, extent);
                mx = fmaxf(mx, dot * qkScale +
                                   table[static_cast<size_t>(hq) * (extent + 1) + dist]);
            }
            float sum = expf(-mx);
            for (uint32_t ki = 0; ki < nKeys; ++ki) {
                const uint32_t kp = firstKey + ki;
                const float *kh = kCache + static_cast<size_t>(kp) * nKvHeads * headDim +
                                  static_cast<size_t>(hkv) * headDim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < headDim; ++d) dot += qh[d] * kh[d];
                const uint32_t dist = umin(pos - kp, extent);
                sum += expf(dot * qkScale +
                            table[static_cast<size_t>(hq) * (extent + 1) + dist] - mx);
            }
            float *oh = out + static_cast<size_t>(hq) * headDim;
            for (uint32_t d = 0; d < headDim; ++d) oh[d] = 0.0f;
            const float inv = 1.0f / sum;
            for (uint32_t ki = 0; ki < nKeys; ++ki) {
                const uint32_t kp = firstKey + ki;
                const float *kh = kCache + static_cast<size_t>(kp) * nKvHeads * headDim +
                                  static_cast<size_t>(hkv) * headDim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < headDim; ++d) dot += qh[d] * kh[d];
                const uint32_t dist = umin(pos - kp, extent);
                const float w = expf(dot * qkScale +
                                     table[static_cast<size_t>(hq) * (extent + 1) + dist] -
                                     mx) * inv;
                const float *vh = vCache + static_cast<size_t>(kp) * nKvHeads * headDim +
                                  static_cast<size_t>(hkv) * headDim;
                for (uint32_t d = 0; d < headDim; ++d) oh[d] += w * vh[d];
            }
        }
    }// namespace

    void kInklingRelBias(const float *r, const float *bias, const float *sink,
                         float *table, uint32_t heads, uint32_t nRel,
                         uint32_t extent, void *stream) {
        relBiasKernel<<<heads, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                r, bias, sink, table, nRel, extent);
    }

    void kInklingAttention(const float *q, const float *kCache,
                           const float *vCache, const float *table, uint32_t pos,
                           uint32_t firstKey, uint32_t nKeys, uint32_t headDim,
                           uint32_t nHeads, uint32_t nKvHeads, uint32_t extent,
                           float qkScale, float *out, void *stream) {
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        // The model's geometry (headDim 128, <=32 heads) takes the warp kernel;
        // anything else falls back to the reference one-thread-per-head shape.
        // The bias row is staged in shared memory, so cap `extent` at 16 KiB.
        if (headDim == 128u && nHeads <= 32u && nHeads % nKvHeads == 0 &&
            extent <= 4096u) {
            switch (attentionWarps(nKeys)) {
                case 2:
                    launchAttentionWarp<2>(q, kCache, vCache, table, pos,
                                           firstKey, nKeys, nHeads, nKvHeads,
                                           extent, qkScale, out, s);
                    break;
                case 4:
                    launchAttentionWarp<4>(q, kCache, vCache, table, pos,
                                           firstKey, nKeys, nHeads, nKvHeads,
                                           extent, qkScale, out, s);
                    break;
                default:
                    launchAttentionWarp<8>(q, kCache, vCache, table, pos,
                                           firstKey, nKeys, nHeads, nKvHeads,
                                           extent, qkScale, out, s);
                    break;
            }
            return;
        }
        attentionKernel<<<1, 32, 0, s>>>(
                q, kCache, vCache, table, pos, firstKey, nKeys, headDim, nHeads,
                nKvHeads, extent, qkScale, out);
    }

    void kInklingMix(const float *states, uint32_t nStates, const float *query,
                     uint32_t nQueries, uint32_t hidden, float *out,
                     void *stream) {
        // A full 32-warp block, not one warp per state: the states are reduced
        // one per warp either way (the kernel strides), but the trailing
        // hidden-wide combine is the expensive part of this kernel and it runs
        // on the whole block, so 1024 threads make it three rounds instead of
        // one round per state.
        mixKernel<<<nQueries, 1024u, 0,
                    reinterpret_cast<cudaStream_t>(stream)>>>(
                states, nStates, query, hidden, out);
    }
    void kInklingMixCurrent(const float *states, uint32_t nStates,
                            const float *cur, const float *query,
                            uint32_t nQueries, uint32_t hidden, float *out,
                            void *stream) {
        // 32 warps: state s is still owned by warp s (the kernel strides past
        // the state count), and the hidden-wide combine gets 1024 threads.
        mixCurKernel<<<nQueries, 1024u, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
                states, nStates, cur, query, hidden, out);
    }
    void kInklingAttentionStore(const float *q, const float *kcur,
                                const float *vcur, float *kCache, float *vCache,
                                const float *table, uint32_t pos,
                                uint32_t firstKey, uint32_t nKeys,
                                uint32_t nHeads, uint32_t nKvHeads,
                                uint32_t extent, float qkScale, float *out,
                                void *stream) {
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        const uint32_t kvDim = nKvHeads * 128u;
        if (nHeads <= 32u && nHeads % nKvHeads == 0 && extent <= 4096u) {
            storeKvKernel<<<(kvDim + 255u) / 256u, 256, 0, s>>>(
                    kcur, vcur, kCache, vCache, pos, kvDim, nullptr);
            switch (attentionWarps(nKeys)) {
                case 2:
                    launchAttentionWarp<2>(q, kCache, vCache, table, pos,
                                           firstKey, nKeys, nHeads, nKvHeads,
                                           extent, qkScale, out, s);
                    break;
                case 4:
                    launchAttentionWarp<4>(q, kCache, vCache, table, pos,
                                           firstKey, nKeys, nHeads, nKvHeads,
                                           extent, qkScale, out, s);
                    break;
                default:
                    launchAttentionWarp<8>(q, kCache, vCache, table, pos,
                                           firstKey, nKeys, nHeads, nKvHeads,
                                           extent, qkScale, out, s);
                    break;
            }
            return;
        }
        // Fallback: store, then the reference kernel.
        float *ks = kCache + static_cast<size_t>(pos) * kvDim;
        float *vs = vCache + static_cast<size_t>(pos) * kvDim;
        cudaMemcpyAsync(ks, kcur, kvDim * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(vs, vcur, kvDim * sizeof(float), cudaMemcpyDeviceToDevice, s);
        attentionKernel<<<1, 32, 0, s>>>(
                q, kCache, vCache, table, pos, firstKey, nKeys, 128u, nHeads,
                nKvHeads, extent, qkScale, out);
    }

    void kInklingAttentionStorePos(const float *q, const float *kcur,
                                   const float *vcur, float *kCache,
                                   float *vCache, const float *table,
                                   const int32_t *posDev, bool isSwa,
                                   uint32_t window, uint32_t nHeads,
                                   uint32_t nKvHeads, uint32_t extent,
                                   float qkScale, float *out, void *stream) {
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        const uint32_t kvDim = nKvHeads * 128u;
        // Fixed 8 warps: the graph does not know nKeys on the host, and 8 is
        // what attentionWarps() picks for the >= 512-key decode range.  The
        // online softmax kernel handles any nKeys with any W.
        storeKvKernel<<<(kvDim + 255u) / 256u, 256, 0, s>>>(
                kcur, vcur, kCache, vCache, 0, kvDim, posDev);
        launchAttentionWarp<8>(q, kCache, vCache, table, 0, 0, 0, nHeads,
                               nKvHeads, extent, qkScale, out, s, posDev, window,
                               isSwa ? 1u : 0u);
    }

    namespace {
        __global__ void tauScaleKernel(float *q, uint32_t qDim, float *rel,
                                       uint32_t relDim, const int32_t *posDev,
                                       float alpha, float floor) {
            const uint32_t pos = static_cast<uint32_t>(*posDev);
            const float arg =
                    fmaxf(1.0f, float(pos + 1) / fmaxf(1.0f, floor));
            const float tau = 1.0f + alpha * logf(arg);
            for (uint32_t i = threadIdx.x; i < qDim; i += blockDim.x)
                q[i] *= tau;
            for (uint32_t i = threadIdx.x; i < relDim; i += blockDim.x)
                rel[i] *= tau;
        }
    }// namespace

    void kInklingTauScale(float *q, uint32_t qDim, float *rel, uint32_t relDim,
                          const int32_t *posDev, float alpha, float floor,
                          void *stream) {
        tauScaleKernel<<<1, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                q, qDim, rel, relDim, posDev, alpha, floor);
    }

}// namespace tinycoder::gpu
