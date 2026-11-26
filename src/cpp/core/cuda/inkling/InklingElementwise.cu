#include "gpu/InklingKernels.hpp"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace tinycoder::gpu {
    namespace {
        __global__ void rmsNormKernel(const float *x, const float *w, float *out,
                                      uint32_t n, float eps) {
            __shared__ float s[256];
            double ss = 0.0;
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
                ss += double(x[i]) * x[i];
            s[threadIdx.x] = static_cast<float>(ss);
            __syncthreads();
            for (uint32_t k = 128; k > 0; k >>= 1) {
                if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
                __syncthreads();
            }
            const float inv = rsqrtf(s[0] / float(n) + eps);
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
                out[i] = x[i] * inv * (w ? w[i] : 1.0f);
        }

        __global__ void rmsNormRowsKernel(const float *x, const float *w,
                                          float *out, uint32_t n, float eps) {
            __shared__ float s[256];
            const uint32_t row = blockIdx.x;
            const float *xr = x + static_cast<size_t>(row) * n;
            float *orow = out + static_cast<size_t>(row) * n;
            double ss = 0.0;
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
                ss += double(xr[i]) * xr[i];
            s[threadIdx.x] = static_cast<float>(ss);
            __syncthreads();
            for (uint32_t k = 128; k > 0; k >>= 1) {
                if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
                __syncthreads();
            }
            const float inv = rsqrtf(s[0] / float(n) + eps);
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
                orow[i] = xr[i] * inv * (w ? w[i] : 1.0f);
        }

        __device__ __forceinline__ float rmsInvShared(const float *x, uint32_t n,
                                                      float *s) {
            // Barrier before the first write: controlDownKernel calls this twice
            // back to back, and without a barrier the second call can overwrite
            // s[0] (and the rest of s) while a slower warp is still reading the
            // first call's reduction.  That is a real race -- it moved a 32-token
            // prefill's logits by 0.13 between runs of the same binary, which is
            // visible on a near-tied router expert.
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

        __global__ void controlDownKernel(const float *cur, const float *msg,
                                          const float *w, float *g, uint32_t rank,
                                          uint32_t hidden) {
            __shared__ float s[256];
            const uint32_t r = blockIdx.x;
            const float inv0 = rmsInvShared(cur, hidden, s);
            const float inv1 = rmsInvShared(msg, hidden, s);
            const uint32_t n2 = 2 * hidden;
            const float *wr = w + static_cast<size_t>(r) * n2;
            float acc = 0.0f;
            for (uint32_t j = threadIdx.x; j < n2; j += blockDim.x) {
                const float t = (j < hidden) ? cur[j] * inv0
                                             : msg[j - hidden] * inv1;
                acc += wr[j] * t;
            }
            // Barrier before reusing s[]: rmsInvShared's return already read
            // s[0] on every thread, and thread 0's `s[0] = acc` below would
            // otherwise land before a slower warp reads it (racecheck: 768
            // definite hazards before this line was added).
            __syncthreads();
            s[threadIdx.x] = acc;
            __syncthreads();
            for (uint32_t k = 128; k > 0; k >>= 1) {
                if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
                __syncthreads();
            }
            if (threadIdx.x == 0) g[r] = s[0];
        }

        __global__ void controlMixKernel2(const float *cur, const float *msg,
                                          const float *gainW,
                                          const float *forgetW, const float *g,
                                          float *out, uint32_t hidden,
                                          uint32_t rank) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= hidden) return;
            const float *gr = gainW + static_cast<size_t>(i) * rank;
            const float *fr = forgetW + static_cast<size_t>(i) * rank;
            float gv = 0.0f, fv = 0.0f;
            for (uint32_t j = 0; j < rank; ++j) {
                gv += gr[j] * g[j];
                fv += fr[j] * g[j];
            }
            out[i] = cur[i] - tanhf(fv) * cur[i] + tanhf(gv) * msg[i];
        }

        // ---- decode loop controllers, batched by layer -----------------------
        // Same arithmetic as controlDownKernel/controlMixKernel2 (the reference
        // forms above, which the batched prefill path is also bit-exact with),
        // with every controller of one layer sharing a launch per stage.  Two
        // costs were removed:
        //   * the mix read the [hidden][rank] gain/forget rows with a 4 B access
        //     per 256 B row -- 8x the sectors of the data it needs, on a matrix
        //     that has to come from DRAM once per token (measured 19.7 us per
        //     call); it now reads the row four floats at a time.
        //   * the down and the mix were six launches per layer (three controls);
        //     the three downs are one launch and the three mixes another.
        // What was NOT changed is the per-block norm: see controlDownFusedKernel
        // for why hoisting it costs the last two ULP and the bit-exact
        // trajectory.

        /// controlDownKernel, one launch for every controller of the layer:
        /// blockIdx.x = c*rank + r, and each block keeps the reference kernel's
        /// own two RMS norms (rmsInvShared, the same 256-thread strided fp64
        /// partials and the same shared float tree).  Re-deriving the norms in
        /// the block instead of hoisting them into a separate kernel is not
        /// laziness: `rmsInvShared` is exact only in the reference shape, and
        /// the two-calls-in-one-block form is what every other path in this
        /// tree is gated against -- a one-norm-per-block variant was measured to
        /// differ by 2 ULP on real rows, which is enough to flip a near-tied
        /// router expert.  The norms are recomputed by all `rank` blocks, but
        /// the blocks run concurrently, so batching the layers' controllers into
        /// one launch is what removes the cost, not hoisting the norms.
        __global__ void controlDownFusedKernel(InklingControlArgs a) {
            __shared__ float s[256];
            const uint32_t c = blockIdx.x / a.rank;
            const uint32_t r = blockIdx.x - c * a.rank;
            const float *cv = a.cur[c];
            const float *mv = a.msg[c];
            const float inv0 = rmsInvShared(cv, a.hidden, s);
            const float inv1 = rmsInvShared(mv, a.hidden, s);
            const uint32_t n2 = 2u * a.hidden;
            const float *wr = a.downW[c] + static_cast<size_t>(r) * n2;
            float acc = 0.0f;
            for (uint32_t j = threadIdx.x; j < n2; j += blockDim.x) {
                const float t = (j < a.hidden) ? cv[j] * inv0
                                               : mv[j - a.hidden] * inv1;
                acc += wr[j] * t;
            }
            __syncthreads();
            s[threadIdx.x] = acc;
            __syncthreads();
            for (uint32_t k = 128; k > 0; k >>= 1) {
                if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
                __syncthreads();
            }
            if (threadIdx.x == 0) a.g[static_cast<size_t>(c) * a.rank + r] = s[0];
        }

        /// controlMixKernel2 with the per-element weight row read as float4.
        /// The four lane components are accumulated in ascending j into the same
        /// single accumulator, so the FMA chain is the reference chain; the only
        /// change is the width of the load.  A rank that is not a multiple of
        /// four keeps the scalar walk.
        __global__ void controlMixRegKernel(InklingControlArgs a) {
            const uint32_t c = blockIdx.y;
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= a.hidden) return;
            const uint32_t R = a.rank;
            const float *gr = a.gainW[c] + static_cast<size_t>(i) * R;
            const float *fr = a.forgetW[c] + static_cast<size_t>(i) * R;
            const float *gt = a.g + static_cast<size_t>(c) * R;
            float gv = 0.0f, fv = 0.0f;
            if ((R & 3u) == 0u) {
                const float4 *g4 = reinterpret_cast<const float4 *>(gr);
                const float4 *f4 = reinterpret_cast<const float4 *>(fr);
                for (uint32_t j = 0; j < R; j += 4u) {
                    const float4 x = g4[j >> 2];
                    const float4 y = f4[j >> 2];
                    gv = fmaf(x.x, gt[j], gv);
                    gv = fmaf(x.y, gt[j + 1u], gv);
                    gv = fmaf(x.z, gt[j + 2u], gv);
                    gv = fmaf(x.w, gt[j + 3u], gv);
                    fv = fmaf(y.x, gt[j], fv);
                    fv = fmaf(y.y, gt[j + 1u], fv);
                    fv = fmaf(y.z, gt[j + 2u], fv);
                    fv = fmaf(y.w, gt[j + 3u], fv);
                }
            } else {
                for (uint32_t j = 0; j < R; ++j) {
                    gv = fmaf(gr[j], gt[j], gv);
                    fv = fmaf(fr[j], gt[j], fv);
                }
            }
            const float c0 = a.cur[c][i];
            const float m0 = a.msg[c][i];
            a.out[c][i] = c0 - tanhf(fv) * c0 + tanhf(gv) * m0;
        }

        __global__ void clipKernel(float *x, uint32_t len, const float *scale, float threshold) {
            const uint32_t r = blockIdx.x;
            float *row = x + static_cast<size_t>(r) * len;
            __shared__ float s[256];
            double ss = 0.0;
            for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
                if (scale) row[i] *= scale[static_cast<size_t>(r) * len + i];
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

        __global__ void shortConvKernel(const float *cur, const float *w,
                                        const float *dyn, float *hist, float *out,
                                        uint32_t channels, uint32_t nConv,
                                        uint32_t nHeads, bool residual,
                                        float scale) {
            const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= channels) return;
            const uint32_t h = c / (channels / nHeads);
            float acc = residual ? cur[c] : 0.0f;
            for (uint32_t k = 0; k < nConv; ++k) {
                const float v = (k + 1 < nConv)
                                        ? hist[static_cast<size_t>(k) * channels + c]
                                        : cur[c];
                acc += w[static_cast<size_t>(c) * nConv + k] * v;
            }
            if (dyn) {
                for (uint32_t lag = 0; lag < nConv; ++lag) {
                    const uint32_t src = nConv - 1 - lag;
                    const float v = (src + 1 < nConv)
                                            ? hist[static_cast<size_t>(src) * channels + c]
                                            : cur[c];
                    acc += dyn[static_cast<size_t>(h) * nConv + lag] * v;
                }
            }
            out[c] = acc * scale;
            for (uint32_t k = 0; k + 2 < nConv; ++k)
                hist[static_cast<size_t>(k) * channels + c] =
                        hist[static_cast<size_t>(k + 1) * channels + c];
            hist[static_cast<size_t>(nConv - 2) * channels + c] = cur[c];
        }
    }// namespace

    void kInklingRmsNorm(const float *x, const float *weight, float *out,
                         uint32_t n, float eps, void *stream) {
        rmsNormKernel<<<1, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                x, weight, out, n, eps);
    }
    void kInklingRmsNormRows(const float *x, const float *weight, float *out,
                             uint32_t n, uint32_t rows, float eps, void *stream) {
        rmsNormRowsKernel<<<rows, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                x, weight, out, n, eps);
    }
    namespace {
        __global__ void posStepKernel(int32_t *p) { *p += 1; }
    }// namespace

    void kInklingPosStep(int32_t *posDev, void *stream) {
        posStepKernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(posDev);
    }

    void kInklingControlDown(const float *cur, const float *msg,
                             const float *downW, float *g, uint32_t rank,
                             uint32_t hidden, void *stream) {
        controlDownKernel<<<rank, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                cur, msg, downW, g, rank, hidden);
    }
    void kInklingControlMix(const float *cur, const float *msg,
                            const float *gainW, const float *forgetW,
                            const float *g, float *out, uint32_t hidden,
                            uint32_t rank, void *stream) {
        controlMixKernel2<<<(hidden + 255u) / 256u, 256, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
                cur, msg, gainW, forgetW, g, out, hidden, rank);
    }

    void kInklingControlDownFused(const InklingControlArgs &a, void *stream) {
        if (a.nCtl == 0) return;
        controlDownFusedKernel<<<a.nCtl * a.rank, 256, 0,
                                 reinterpret_cast<cudaStream_t>(stream)>>>(a);
    }
    void kInklingControlMixFused(const InklingControlArgs &a, void *stream) {
        if (a.nCtl == 0) return;
        dim3 grid((a.hidden + 255u) / 256u, a.nCtl);
        controlMixRegKernel<<<grid, 256, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(a);
    }

    void kInklingClip(float *x, uint32_t rows, uint32_t len, const float *scale,
                      float threshold, void *stream) {
        clipKernel<<<rows, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                x, len, scale, threshold);
    }
    void kInklingShortConv(const float *cur, const float *weight,
                           const float *dynamic, float *history, float *out,
                           uint32_t channels, uint32_t nConv, uint32_t nHeads,
                           bool residual, float scale, void *stream) {
        const uint32_t blocks = (channels + 255u) / 256u;
        shortConvKernel<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                cur, weight, dynamic, history, out, channels, nConv, nHeads,
                residual, scale);
    }
}// namespace tinycoder::gpu
