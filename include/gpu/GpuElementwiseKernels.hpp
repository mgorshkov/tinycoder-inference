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

// GpuElementwiseKernels.hpp -- residual add, RMSNorm, f32->f16, bias add, embedding gather.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


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

// TINYCODER_NODE_PAD=N (measurement only, default 0): launch this
// empty kernel N times per layer inside the captured decode graph.
// Every node captured on one stream is a link in the same dependency
// chain, so the marginal replay cost of a node is
// (tg(N) - tg(0)) / (28*N) -- the number that decides whether fusing
// launches can pay for itself.  It touches nothing, so all it adds is
// a node.
__global__ void kGraphPadNoop() {}

// Fused residual add + RMSNorm: one kernel instead of
// kAddResidual + kRMSNormRow (fused residual + RMSNorm).
__global__ void kAddResidualRmsNorm(float *x, const float *add,
                                    float *y, const float *w,
                                    uint32_t n, uint32_t rows,
                                    float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    float *xp = x + static_cast<size_t>(row) * n;
    const float *ap = add + static_cast<size_t>(row) * n;
    float *op = y + static_cast<size_t>(row) * n;
    extern __shared__ float ssum[];
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xp[i] + ap[i];
        xp[i] = v;
        acc = fmaf(v, v, acc);
    }
    ssum[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    const float rms = rsqrtf(ssum[0] / static_cast<float>(n) + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        op[i] = xp[i] * rms * w[i];
    }
}

// kRMSNormRow + Q8_1 fold (2026-10-10, plan section 9.26 lever 1): the
// rms_norm block already holds the whole row, so it also emits the
// Q8_1 activation blocks the dp4a GLU GEMV consumes -- deleting the
// separate kQuantizeQ8_1 launch (2.5 us/layer).  The quantize math is
// copied VERBATIM from kQuantizeQ8_1 (amax/127, roundf(v/d), s = sum
// of the FLOATS) over the same float values the float path stores, so
// the Q8_1 bytes are bit-identical to the two-kernel sequence.
// Requires blockDim 1024 and n == 1536 (thread t owns elements t and
// t+1024: warp w covers Q8_1 block w from its first element and, for
// w < 16, block 32 + w from its second).  Launched only for the dense
// decode (seqLen==1, H==1536); every other shape keeps the separate
// launch.  TINYCODER_FUSE_Q81_PROD=0 restores it.
// One launch for the Qwen2 Q/K/V biases (was three kAddBias launches).
__global__ void kAddBias3(float *q, const float *qb, float *k,
                          const float *kb, float *v, const float *vb,
                          uint32_t seqLen, uint32_t qLen,
                          uint32_t kvLen) {
    const uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    float *qp = q + static_cast<size_t>(s) * qLen;
    float *kp = k + static_cast<size_t>(s) * kvLen;
    float *vp = v + static_cast<size_t>(s) * kvLen;
    for (uint32_t i = threadIdx.x; i < qLen; i += blockDim.x)
        if (qb) qp[i] += qb[i];
    for (uint32_t i = threadIdx.x; i < kvLen; i += blockDim.x) {
        if (kb) kp[i] += kb[i];
        if (vb) vp[i] += vb[i];
    }
}

// ---- residual + RMSNorm + Q8_1, one launch -------------------------
// Decode-only (rows == seqLen == 1 here), blockDim 1024, n <= 2048.
// Replaces the kAddResidual/kRMSNormRowQ81 pair AND the separate
// gemvLaunchQuantizeQ8F32 that every int8Gemv call used to issue
// against the normed row, so one GEMV group costs 1 node instead of
// (residual + norm + quantize + GEMV).
//
// Both outputs are byte-identical to the sequence it replaces:
//   * s.norm: same per-thread partition (thread t owns t and t+1024)
//     and the same reduction tree as kRMSNormRowQ81, i.e. the same
//     float sum, and the residual is the same `x[i] += add[i]` that
//     kAddResidual performed in place.
//   * q81: the 36-byte block_q8_1 layout gemv/Gemv.cu's
//     gemvQuantizeQ8F32 writes (2B d, 2B s = sum of the raw floats,
//     32 int8), same warp partition of 32 consecutive values, same
//     `amax/127 + roundf(v/d)` math -- so the GEMV reading it is
//     the same kernel on the same bytes.
// `add == nullptr` means this row has no residual to fold.
// x and y are deliberately NOT restrict-qualified: the final norm runs
// in place (x == y == the residual stream) so the caller passes the same
// pointer for both, and a restrict-qualified pair is UB there -- g++
// says so at the call site with -Wrestrict.  Only the read-only operands
// (add, w) and the q8_1 output alias nothing.
// y == nullptr means "write the normed row back into x" (the final norm
// reuses the residual stream once the last layer has consumed it).  The
// two pointer arguments never alias -- the in-place case passes nullptr
// rather than x -- so `__restrict__` still holds and the compiler may
// keep the loads/stores disjoint.  Passing x twice here instead is what
// -Wrestrict flagged, and dropping `__restrict__` to silence it cost
// 0.6% of pp512 (727 -> 723).
__global__ void kResidualNormQ81(float *__restrict__ x,
                                 const float *__restrict__ add,
                                 float *__restrict__ y,
                                 const float *__restrict__ w,
                                 uint8_t *__restrict__ q81, uint32_t n,
                                 uint32_t rows, float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    float *xp = x + static_cast<size_t>(row) * n;
    const float *ap = add != nullptr
                              ? add + static_cast<size_t>(row) * n
                              : nullptr;
    float *op = y != nullptr ? y + static_cast<size_t>(row) * n : xp;
    extern __shared__ float ssum[];
    float acc = 0.0f;
    if (ap != nullptr) {
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            const float v = xp[i] + ap[i];
            xp[i] = v;
            acc = fmaf(v, v, acc);
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            const float v = xp[i];
            acc = fmaf(v, v, acc);
        }
    }
    ssum[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t st = blockDim.x / 2; st >= 32u; st >>= 1) {
        if (threadIdx.x < st) ssum[threadIdx.x] += ssum[threadIdx.x + st];
        __syncthreads();
    }
    // Last five levels touch only warp 0 (same pairing as
    // kRMSNormRowQ81, so the float sum is identical).
    if (threadIdx.x < 32u) {
        float v = ssum[threadIdx.x];
#pragma unroll
        for (uint32_t o = 16u; o > 0u; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0u) ssum[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(ssum[0] / static_cast<float>(n) + eps);
    const uint32_t i0 = threadIdx.x;
    const uint32_t i1 = threadIdx.x + blockDim.x;
    const float v0 = xp[i0] * rms * w[i0];
    op[i0] = v0;
    const bool has1 = (i1 < n);
    float v1 = 0.0f;
    if (has1) {
        v1 = xp[i1] * rms * w[i1];
        op[i1] = v1;
    }
    // Warp w owns Q8_1 block w (elements from i0) and block 32+w
    // (elements from i1); blocks are 36 B and contiguous.
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t rowBlk = static_cast<uint64_t>(row) * (n / 32u);
    {
        float amax = fabsf(v0), sum = v0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
        }
        const float d = amax / 127.0f;
        uint8_t *blk = q81 + (rowBlk + warp) * 36u;
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
    if (has1) {
        float amax = fabsf(v1), sum = v1;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
        }
        const float d = amax / 127.0f;
        uint8_t *blk = q81 + (rowBlk + 32u + warp) * 36u;
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
