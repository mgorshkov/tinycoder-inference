#include "gpu/InklingKernels.hpp"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace tinycoder::gpu {
    namespace {
        __global__ void routerKernel(const float *logits, const float *bias,
                                     int32_t *idx, float *wgt, uint32_t nExperts,
                                     uint32_t nUsed, float scale) {
            __shared__ float probs[256];
            __shared__ float sel[256];
            const uint32_t e = threadIdx.x;
            if (e < nExperts) {
                probs[e] = 1.0f / (1.0f + expf(-logits[e]));
                sel[e] = probs[e] + (bias ? bias[e] : 0.0f);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
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
                    idx[u] = static_cast<int32_t>(best);
                    sum += probs[best];
                    sel[best] = -INFINITY;
                }
                const float norm = scale / fmaxf(sum, 1e-20f);
                // second pass: weights need the original probs, recover from
                // logits (the id list is already fixed).
                for (uint32_t u = 0; u < nUsed; ++u) {
                    const int e2 = idx[u];
                    wgt[u] = (1.0f / (1.0f + expf(-logits[e2]))) * norm;
                }
            }
        }

        __global__ void situKernel(float *gateUp, uint32_t n, float beta, float lb) {
            for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
                const float g = gateUp[j];
                const float u = gateUp[n + j];
                gateUp[j] = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g))) *
                            (lb * tanhf(u / lb));
            }
        }

        __global__ void situAllKernel(float *gateUp, uint32_t nUsed, uint32_t n,
                                      const float *scale, float beta, float lb) {
            const uint32_t u = blockIdx.x;
            if (u >= nUsed) return;
            const float s = scale[u];
            float *base = gateUp + static_cast<size_t>(u) * 2 * n;
            for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
                const float g = base[j] * s;
                const float up = base[n + j] * s;
                base[j] = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g))) *
                          (lb > 0.0f ? lb * tanhf(up / lb) : up);
            }
        }

        __global__ void moeAccumKernel(const float *expertOut, const float *wgt,
                                       const float *scale, float *out,
                                       uint32_t nUsed, uint32_t hidden) {
            for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
                float a = 0.0f;
                for (uint32_t u = 0; u < nUsed; ++u)
                    a += wgt[u] * scale[u] * expertOut[static_cast<size_t>(u) * hidden + i];
                out[i] = a;
            }
        }
    }// namespace

    void kInklingRouter(const float *logits, const float *bias, int32_t *idx,
                        float *wgt, uint32_t nExperts, uint32_t nUsed,
                        float scale, void *stream) {
        routerKernel<<<1, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                logits, bias, idx, wgt, nExperts, nUsed, scale);
    }
    void kInklingSitu(float *gateUp, uint32_t n, float beta, float lb,
                      void *stream) {
        situKernel<<<(n + 255u) / 256u, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                gateUp, n, beta, lb);
    }
    void kInklingSituAll(float *gateUp, uint32_t nUsed, uint32_t n,
                         const float *scale, float beta, float lb, void *stream) {
        situAllKernel<<<nUsed, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                gateUp, nUsed, n, scale, beta, lb);
    }
    void kInklingMoeAccum(const float *expertOut, const float *wgt,
                          const float *scale, float *out, uint32_t nUsed,
                          uint32_t hidden, void *stream) {
        moeAccumKernel<<<(hidden + 255u) / 256u, 256, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
                expertOut, wgt, scale, out, nUsed, hidden);
    }
}// namespace tinycoder::gpu
