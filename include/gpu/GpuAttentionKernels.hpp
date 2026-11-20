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

// GpuAttentionKernels.hpp -- warp reductions, KV store/RoPE, flash-attention split + combine.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


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
// Prefill attention, two tokens per warp: the K/V loads are shared by
// the pair, and the earlier token simply stops one position sooner.
template<uint32_t HD, typename KVT>
__global__ void kWarpAttention2Tok(
        const float *__restrict__ q, const KVT *__restrict__ kCache,
        const KVT *__restrict__ vCache, float *__restrict__ out,
        uint32_t seqLen, uint32_t nHeads, uint32_t nKVHeads,
        uint32_t cachePos, float invSqrt) {
    constexpr uint32_t NV = HD / 32;
    const uint32_t s0 = blockIdx.x * 2;
    const uint32_t s1 = s0 + 1;
    const uint32_t warpId = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t lane = threadIdx.x;
    if (s0 >= seqLen || warpId >= nHeads) return;
    const bool has1 = s1 < seqLen;
    const uint32_t qHead = warpId;
    const uint32_t kvHead = qHead / (nHeads / nKVHeads);
    const uint32_t end0 = cachePos + s0;
    const uint32_t end1 = has1 ? (cachePos + s1) : end0;
    const float *q0 = q + (static_cast<size_t>(s0) * nHeads + qHead) * HD;
    const float *q1 = has1
                              ? q + (static_cast<size_t>(s1) * nHeads + qHead) * HD
                              : q0;
    float a0[NV], a1[NV];
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) {
        a0[i] = 0.0f;
        a1[i] = 0.0f;
    }
    float m0 = -1e30f, l0 = 0.0f, m1 = -1e30f, l1 = 0.0f;
    for (uint32_t cs = 0; cs <= end1; ++cs) {
        const KVT *kp =
                kCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
        const KVT *vp =
                vCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
        float sc0 = 0.0f, sc1 = 0.0f;
        const bool use0 = cs <= end0;
#pragma unroll
        for (uint32_t i = 0; i < NV; ++i) {
            const float kv = kvLoad(kp, lane + i * 32);
            if (use0) sc0 = fmaf(q0[lane + i * 32], kv, sc0);
            if (has1) sc1 = fmaf(q1[lane + i * 32], kv, sc1);
        }
        if (use0) {
            sc0 = kWarpReduceSum(sc0) * invSqrt;
            if (sc0 > m0) {
                const float al = exp2f((m0 - sc0) * LOG2E);
                for (uint32_t i = 0; i < NV; ++i) a0[i] *= al;
                l0 *= al;
                m0 = sc0;
            }
            const float p0 = exp2f((sc0 - m0) * LOG2E);
            l0 += p0;
            for (uint32_t i = 0; i < NV; ++i)
                a0[i] = fmaf(p0, kvLoad(vp, lane + i * 32), a0[i]);
        }
        if (has1) {
            sc1 = kWarpReduceSum(sc1) * invSqrt;
            if (sc1 > m1) {
                const float al = exp2f((m1 - sc1) * LOG2E);
                for (uint32_t i = 0; i < NV; ++i) a1[i] *= al;
                l1 *= al;
                m1 = sc1;
            }
            const float p1 = exp2f((sc1 - m1) * LOG2E);
            l1 += p1;
            for (uint32_t i = 0; i < NV; ++i)
                a1[i] = fmaf(p1, kvLoad(vp, lane + i * 32), a1[i]);
        }
    }
    float *o0 = out + (static_cast<size_t>(s0) * nHeads + qHead) * HD;
    const float i0 = 1.0f / l0;
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) o0[lane + i * 32] = a0[i] * i0;
    if (has1) {
        float *o1 = out + (static_cast<size_t>(s1) * nHeads + qHead) * HD;
        const float i1 = 1.0f / l1;
#pragma unroll
        for (uint32_t i = 0; i < NV; ++i)
            o1[lane + i * 32] = a1[i] * i1;
    }
}

// Prefill attention, two warps per (token, head): each warp scans half
// the causal KV range, then the pair merges (m,l,acc) in shared memory.
template<uint32_t HD, typename KVT>
__global__ void kWarpAttention2W(
        const float *__restrict__ q, const KVT *__restrict__ kCache,
        const KVT *__restrict__ vCache, float *__restrict__ out,
        uint32_t seqLen, uint32_t nHeads, uint32_t nKVHeads,
        uint32_t cachePos, float invSqrt) {
    constexpr uint32_t NV = HD / 32;
    const uint32_t s = blockIdx.x;
    const uint32_t pair = threadIdx.y / 2;
    const uint32_t sub = threadIdx.y % 2;
    const uint32_t head = blockIdx.y * 2 + pair;
    const uint32_t lane = threadIdx.x;
    if (s >= seqLen || head >= nHeads) return;
    const uint32_t kvHead = head / (nHeads / nKVHeads);
    const uint32_t csEnd = cachePos + s;
    const uint32_t half = (csEnd + 1) / 2;
    const uint32_t lo = sub == 0 ? 0 : half;
    const uint32_t hi = sub == 0 ? (half == 0 ? 0 : half - 1) : csEnd;
    const float *qPtr =
            q + (static_cast<size_t>(s) * nHeads + head) * HD;
    __shared__ float sm[2][2], sl[2][2], sacc[2][2][HD];
    float acc[NV];
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
    float m = -1e30f, l = 0.0f;
    if (lo <= hi && lo <= csEnd) {
        for (uint32_t cs = lo; cs <= hi; ++cs) {
            const KVT *kp = kCache +
                            (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
            const KVT *vp = vCache +
                            (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
            float sc = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i)
                sc = fmaf(qPtr[lane + i * 32], kvLoad(kp, lane + i * 32), sc);
            sc = kWarpReduceSum(sc) * invSqrt;
            if (sc > m) {
                const float al = exp2f((m - sc) * LOG2E);
                for (uint32_t i = 0; i < NV; ++i) acc[i] *= al;
                l *= al;
                m = sc;
            }
            const float p = exp2f((sc - m) * LOG2E);
            l += p;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i)
                acc[i] = fmaf(p, kvLoad(vp, lane + i * 32), acc[i]);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) sacc[pair][sub][lane + i * 32] = acc[i];
    if (lane == 0) {
        sm[pair][sub] = m;
        sl[pair][sub] = l;
    }
    __syncthreads();
    if (sub == 0) {
        const float m0 = sm[pair][0], m1 = sm[pair][1];
        const float mNew = fmaxf(m0, m1);
        const float a0 = exp2f((m0 - mNew) * LOG2E);
        const float a1 = exp2f((m1 - mNew) * LOG2E);
        const float lNew = sl[pair][0] * a0 + sl[pair][1] * a1;
        float *op = out +
                    (static_cast<size_t>(s) * nHeads + head) * HD;
        const float inv = lNew > 0.0f ? 1.0f / lNew : 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < NV; ++i) {
            const uint32_t idx = lane + i * 32;
            op[idx] = (sacc[pair][0][idx] * a0 +
                       sacc[pair][1][idx] * a1) *
                      inv;
        }
    }
}

template<uint32_t HD, typename KVT>
__global__ void kWarpAttention(const float *__restrict__ q,
                               const KVT *__restrict__ kCache,
                               const KVT *__restrict__ vCache,
                               float *__restrict__ out, uint32_t seqLen,
                               uint32_t nHeads, uint32_t nKVHeads,
                               uint32_t cachePos, float invSqrtHeadDim) {
    static_assert(HD == 256 || HD == 128 || HD == 64 || HD == 32,
                  "HD must be 32/64/128/256");
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

#pragma unroll 32
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
// Materialised prefill attention: causal softmax of the scores that
// cublasGemmStridedBatchedEx produced, written out as fp16
// probabilities for the second (PV) GEMM.
//
// Why materialise: QK^T and PV are plain GEMMs, and this GPU's tensor
// cores do them at ~35-54 TFLOPS through cuBLAS while the FMA
// warp-per-token kernel reaches ~1.8 TFLOPS (measured; see the ledger
// entry for 2026-11-16).  llama.cpp's flash kernel lands at ~14.
// The cost of materialising is one extra pass over the score matrix
// (read + write), which is cheap next to the GEMMs at pp512.
//
// scores: [head][token][key] fp32 (row-contiguous in key).
// pOut:   [head][token][key] fp16, zero outside the causal range so
//         the PV GEMM needs no mask of its own.
// One block per (head, token) row.
// Block reduction used by kAttnSoftmaxCausal.  The leading
// __syncthreads() is NOT redundant: `red` is reused by the caller's
// next reduction (max then sum), and without it a fast thread can
// overwrite red[0] in the second call before a slower thread has read
// the first call's result -- measured as a huge attention divergence
// on every row except row 0 (whose reduction is single-threaded).
__device__ __forceinline__ float kAttnBlockReduce(float v,
                                                  float *red,
                                                  uint32_t nWarps,
                                                  bool isMax) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t w = threadIdx.x >> 5;
    __syncthreads();
#pragma unroll
    for (uint32_t off = 16; off; off >>= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, off);
        v = isMax ? fmaxf(v, o) : (v + o);
    }
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = lane < nWarps ? red[lane] : (isMax ? -1e30f : 0.0f);
#pragma unroll
        for (uint32_t off = 16; off; off >>= 1) {
            const float o = __shfl_xor_sync(0xffffffffu, t, off);
            t = isMax ? fmaxf(t, o) : (t + o);
        }
        if (lane == 0) red[0] = t;
    }
    __syncthreads();
    return red[0];
}

__global__ void kAttnSoftmaxCausal(const float *__restrict__ scores,
                                   __half *__restrict__ pOut,
                                   uint32_t seqLen, uint32_t headStride,
                                   uint32_t cachePos, float scale) {
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const size_t base =
            static_cast<size_t>(head) * headStride +
            static_cast<size_t>(row) * seqLen;
    const float *s = scores + base;
    __half *p = pOut + base;
    // Last attended column for this query row: the causal bound the
    // warp kernel uses (cachePos + row), clamped to the row width.
    uint32_t lim = cachePos + row;
    if (lim >= seqLen) lim = seqLen - 1u;
    const uint32_t t = threadIdx.x, nt = blockDim.x;
    __shared__ float red[32];
    const uint32_t nWarps = (nt + 31u) >> 5;
    float mx = -1e30f;
    for (uint32_t c = t; c <= lim; c += nt) mx = fmaxf(mx, s[c]);
    mx = kAttnBlockReduce(mx, red, nWarps, true) * scale;
    float sum = 0.0f;
    for (uint32_t c = t; c <= lim; c += nt)
        sum += exp2f((s[c] * scale - mx) * LOG2E);
    sum = kAttnBlockReduce(sum, red, nWarps, false);
    const float inv = 1.0f / sum;
    for (uint32_t c = t; c <= lim; c += nt)
        p[c] = __float2half(exp2f((s[c] * scale - mx) * LOG2E) * inv);
    for (uint32_t c = lim + 1u + t; c < seqLen; c += nt)
        p[c] = __float2half(0.0f);
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

// ------------------------------------------------------------------
// FUSED RoPE(Q) + store K/V (K rotated): ONE launch per layer
// instead of two.  Decode is launch-latency bound (a graph node costs
// ~0.83 us -- measured with TINYCODER_NODE_PAD), so collapsing the
// two independent per-token passes into one kernel removes 28 nodes
// (one per layer) from the 312-node decode graph.
//
// The two halves touch DISJOINT buffers (s.q vs s.k/s.v and the KV
// cache) and each element's arithmetic is copied line for line from
// kRoPEQ(Pos) / kStoreKVRope(Pos): no reduction is reordered and no
// element is recomputed, so the output is bit-identical.  The thread
// partition changed (256 threads for both halves instead of 128/256),
// which reorders WHICH thread does which element -- never the value of
// an element.
//
// The fused kernel must be launched where the two originals were
// (RoPE(Q) before KV store, both before attention); the KV half's
// writes complete before the attention kernel reads them because they
// are the same grid and the next node waits on this one.
// ------------------------------------------------------------------
template<typename KVT>
__global__ void kRoPEQStoreKVPos(float *q, const float *kSrc,
                                 const float *vSrc, KVT *kDst,
                                 KVT *vDst, uint32_t seqLen,
                                 uint32_t qHeads, uint32_t kHeads,
                                 uint32_t headDim, const uint32_t *pos,
                                 const float *cosT,
                                 const float *sinT) {
    uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    uint32_t p = *pos + s;
    uint32_t pairs = headDim / 2;
    const float *c = cosT + static_cast<size_t>(p) * pairs;
    const float *sn = sinT + static_cast<size_t>(p) * pairs;
    uint32_t t = threadIdx.x;
    // (1) RoPE on Q, exactly kRoPEQPos.
    const uint32_t qTotal = qHeads * pairs;
    for (uint32_t idx = t; idx < qTotal; idx += blockDim.x) {
        const uint32_t h = idx / pairs;
        const uint32_t j = idx - h * pairs;
        float cc = c[j], ss = sn[j];
        float *head = q + (static_cast<size_t>(s) * qHeads + h) * headDim;
        float x0 = head[j], x1 = head[j + pairs];
        head[j] = fmaf(x0, cc, -x1 * ss);
        head[j + pairs] = fmaf(x0, ss, x1 * cc);
    }
    // (2) store V verbatim, store K rotated -- exactly
    // kStoreKVRopePos (which caches at `p`, i.e. *pos + s).
    const uint32_t kvSize = kHeads * headDim;
    const float *ks = kSrc + static_cast<size_t>(s) * kvSize;
    const float *vs = vSrc + static_cast<size_t>(s) * kvSize;
    KVT *kd = kDst + static_cast<size_t>(p) * kvSize;
    KVT *vd = vDst + static_cast<size_t>(p) * kvSize;
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

// Eager (host position) twin: identical body, pos/cachePos as values.
// `cachePos` is the KV write row; `pos` is the RoPE angle row (they
// differ for multi-token batches, matching kStoreKVRope).
template<typename KVT>
__global__ void kRoPEQStoreKV(float *q, const float *kSrc,
                              const float *vSrc, KVT *kDst, KVT *vDst,
                              uint32_t seqLen, uint32_t qHeads,
                              uint32_t kHeads, uint32_t headDim,
                              uint32_t cachePos, uint32_t pos,
                              const float *cosT, const float *sinT) {
    uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    uint32_t p = pos + s;
    uint32_t cp = cachePos + s;
    uint32_t pairs = headDim / 2;
    const float *c = cosT + static_cast<size_t>(p) * pairs;
    const float *sn = sinT + static_cast<size_t>(p) * pairs;
    uint32_t t = threadIdx.x;
    const uint32_t qTotal = qHeads * pairs;
    for (uint32_t idx = t; idx < qTotal; idx += blockDim.x) {
        const uint32_t h = idx / pairs;
        const uint32_t j = idx - h * pairs;
        float cc = c[j], ss = sn[j];
        float *head = q + (static_cast<size_t>(s) * qHeads + h) * headDim;
        float x0 = head[j], x1 = head[j + pairs];
        head[j] = fmaf(x0, cc, -x1 * ss);
        head[j + pairs] = fmaf(x0, ss, x1 * cc);
    }
    const uint32_t kvSize = kHeads * headDim;
    const float *ks = kSrc + static_cast<size_t>(s) * kvSize;
    const float *vs = vSrc + static_cast<size_t>(s) * kvSize;
    KVT *kd = kDst + static_cast<size_t>(cp) * kvSize;
    KVT *vd = vDst + static_cast<size_t>(cp) * kvSize;
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
    static_assert(HD == 256 || HD == 128 || HD == 64 || HD == 32,
                  "HD must be 32/64/128/256");
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
    static_assert(HD == 256 || HD == 128 || HD == 64 || HD == 32,
                  "HD must be 32/64/128/256");
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
    static_assert(HD == 256 || HD == 128 || HD == 64 || HD == 32,
                  "HD must be 32/64/128/256");
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
// their downstream dp4a gemv consumers (attnO / GLU) need, deleting the
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
    const uint32_t subWarpEnv = [&]() -> uint32_t {
        if (const char *e = std::getenv("TINYCODER_ATTN_SUBWARP")) {
            const int v = std::atoi(e);
            if (v == 2 || v == 4) return static_cast<uint32_t>(v);
            return 1u;// 0/1 = legacy one-warp layout
        }
        // Qwen2.5-1.5B (12 heads, hd 128): 4 sub-warps measured best.
        if (nHeads == 12 && hd == 128) return 4u;
        return 2u;// measured default: 8/8 interleaved pair wins, +0.35 %
    }();
    const bool useSub = subWarpEnv > 1 && (chunkSize % subWarpEnv) == 0;
    const uint32_t wsub = useSub ? subWarpEnv : 1u;
    // Split-KV by the real length, not the pool window: short contexts
    // launch only chunks that contain KV data, avoiding idle blocks.
    // Device-position graph capture keeps the fixed shape.
    uint32_t activeChunks = numChunks;
    if (!devicePos && chunkSize != 0) {
        const uint32_t need = (pos + 1u + chunkSize - 1u) / chunkSize;
        activeChunks = numChunks < need ? numChunks : need;
        if (activeChunks == 0) activeChunks = 1;
    }
#define TC_ATTN_SPLIT(HDV, KVT)                                                                        \
    do {                                                                                               \
        const KVT *kK = static_cast<const KVT *>(kvK);                                                 \
        const KVT *kV = static_cast<const KVT *>(kvV);                                                 \
        if (wsub == 4)                                                                                 \
            launchSplitVariant<HDV, KVT, 4>(q, kK, kV, pAcc, pM, pL,                                   \
                                            seqLen, nHeads, nKV, pos, devicePos, posDev, activeChunks, \
                                            chunkSize, invSqrt);                                       \
        else if (wsub == 2)                                                                            \
            launchSplitVariant<HDV, KVT, 2>(q, kK, kV, pAcc, pM, pL,                                   \
                                            seqLen, nHeads, nKV, pos, devicePos, posDev, activeChunks, \
                                            chunkSize, invSqrt);                                       \
        else                                                                                           \
            launchSplitVariant<HDV, KVT, 1>(q, kK, kV, pAcc, pM, pL,                                   \
                                            seqLen, nHeads, nKV, pos, devicePos, posDev, activeChunks, \
                                            chunkSize, invSqrt);                                       \
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
            (combine4 && combineW8() && (activeChunks >= 16u)) ? 8u : 4u;
    if (hd == 128) {
        if (nw == 8)
            kAttnCombinePartialW4<128, 8><<<cgrid, 256, 0, g_stream>>>(
                    pAcc, pM, pL, attnOut, seqLen, nHeads, activeChunks,
                    q81Out);
        else if (combine4)
            kAttnCombinePartialW4<128, 4><<<cgrid, 128, 0, g_stream>>>(
                    pAcc, pM, pL, attnOut, seqLen, nHeads, activeChunks,
                    q81Out);
        else
            kAttnCombinePartial<128><<<cgrid, 32, 0, g_stream>>>(
                    pAcc, pM, pL, attnOut, seqLen, nHeads, activeChunks);
    } else if (hd == 64) {
        if (nw == 8)
            kAttnCombinePartialW4<64, 8><<<cgrid, 256, 0, g_stream>>>(
                    pAcc, pM, pL, attnOut, seqLen, nHeads, activeChunks,
                    q81Out);
        else if (combine4)
            kAttnCombinePartialW4<64, 4><<<cgrid, 128, 0, g_stream>>>(
                    pAcc, pM, pL, attnOut, seqLen, nHeads, activeChunks,
                    q81Out);
        else
            kAttnCombinePartial<64><<<cgrid, 32, 0, g_stream>>>(
                    pAcc, pM, pL, attnOut, seqLen, nHeads, activeChunks);
    } else {
        if (nw == 8)
            kAttnCombinePartialW4<32, 8><<<cgrid, 256, 0, g_stream>>>(
                    pAcc, pM, pL, attnOut, seqLen, nHeads, activeChunks,
                    q81Out);
        else if (combine4)
            kAttnCombinePartialW4<32, 4><<<cgrid, 128, 0, g_stream>>>(
                    pAcc, pM, pL, attnOut, seqLen, nHeads, activeChunks,
                    q81Out);
        else
            kAttnCombinePartial<32><<<cgrid, 32, 0, g_stream>>>(
                    pAcc, pM, pL, attnOut, seqLen, nHeads, activeChunks);
    }
}
