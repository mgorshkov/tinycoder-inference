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

// GpuQwen35Kernels.hpp -- qwen35 norms/conv/GDN scan, MoE router, attention, int8 KV.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


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
// Fused q RMSNorm + partial MRoPE (qwen35 full attention).
__global__ void kQ35QNormRoPE(const float *__restrict__ in,
                              float *__restrict__ out,
                              const float *__restrict__ w, uint32_t hd,
                              uint32_t nDims, uint32_t nHeads,
                              uint32_t seqLen, uint32_t pos,
                              const uint32_t *__restrict__ dpos,
                              const float *__restrict__ cosT,
                              const float *__restrict__ sinT) {
    const uint32_t s = blockIdx.x, h = blockIdx.y;
    if (s >= seqLen || h >= nHeads) return;
    extern __shared__ float ssum[];
    const uint32_t lane = threadIdx.x;
    const float *rp = in + (static_cast<size_t>(s) * nHeads + h) * 2u * hd;
    float *op = out + (static_cast<size_t>(s) * nHeads + h) * hd;
    float acc = 0.0f;
    for (uint32_t j = lane; j < hd; j += blockDim.x) {
        const float v = rp[j];
        acc = fmaf(v, v, acc);
    }
    ssum[lane] = acc;
    __syncthreads();
    for (uint32_t st = blockDim.x / 2; st > 0; st >>= 1) {
        if (lane < st) ssum[lane] += ssum[lane + st];
        __syncthreads();
    }
    const float rms = rsqrtf(ssum[0] / static_cast<float>(hd) + 1e-6f);
    const uint32_t p = (dpos != nullptr ? dpos[0] : pos) + s;
    const uint32_t half = nDims / 2;
    const float *c = cosT + static_cast<size_t>(p) * half;
    const float *sn = sinT + static_cast<size_t>(p) * half;
    for (uint32_t j = lane; j < half; j += blockDim.x) {
        const float x0 = rp[j] * rms * w[j];
        const float x1 = rp[j + half] * rms * w[j + half];
        op[j] = fmaf(x0, c[j], -x1 * sn[j]);
        op[j + half] = fmaf(x0, sn[j], x1 * c[j]);
    }
    for (uint32_t j = nDims + lane; j < hd; j += blockDim.x) {
        op[j] = rp[j] * rms * w[j];
    }
}

// Fused k RMSNorm + partial MRoPE (in place).
__global__ void kQ35KNormRoPE(float *__restrict__ t,
                              const float *__restrict__ w, uint32_t hd,
                              uint32_t nDims, uint32_t nKV,
                              uint32_t seqLen, uint32_t pos,
                              const uint32_t *__restrict__ dpos,
                              const float *__restrict__ cosT,
                              const float *__restrict__ sinT) {
    const uint32_t s = blockIdx.x, h = blockIdx.y;
    if (s >= seqLen || h >= nKV) return;
    extern __shared__ float ssum[];
    const uint32_t lane = threadIdx.x;
    float *rp = t + (static_cast<size_t>(s) * nKV + h) * hd;
    float acc = 0.0f;
    for (uint32_t j = lane; j < hd; j += blockDim.x) {
        const float v = rp[j];
        acc = fmaf(v, v, acc);
    }
    ssum[lane] = acc;
    __syncthreads();
    for (uint32_t st = blockDim.x / 2; st > 0; st >>= 1) {
        if (lane < st) ssum[lane] += ssum[lane + st];
        __syncthreads();
    }
    const float rms = rsqrtf(ssum[0] / static_cast<float>(hd) + 1e-6f);
    const uint32_t p = (dpos != nullptr ? dpos[0] : pos) + s;
    const uint32_t half = nDims / 2;
    const float *c = cosT + static_cast<size_t>(p) * half;
    const float *sn = sinT + static_cast<size_t>(p) * half;
    for (uint32_t j = lane; j < half; j += blockDim.x) {
        const float x0 = rp[j] * rms * w[j];
        const float x1 = rp[j + half] * rms * w[j + half];
        rp[j] = fmaf(x0, c[j], -x1 * sn[j]);
        rp[j + half] = fmaf(x0, sn[j], x1 * c[j]);
    }
    for (uint32_t j = nDims + lane; j < hd; j += blockDim.x) {
        rp[j] = rp[j] * rms * w[j];
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
// L2-norm of [headK]-wide slices repeated over value heads, then copy v.
// Each of the nVHeads value heads reads k-head hv / (nVHeads / nKHeads).
// One block per token (128 threads).
//
// The mapping is CONTIGUOUS GROUPS, not hv % nKHeads.  That periodic
// form was the last correctness bug in this engine: it agrees with the
// reference only for hv = 0 (the one head where both mappings pick
// k-head 0), so layer 0's delta-net output matched exllamav3 bit-for-bit
// in its first six channels while its 6144-wide norm was 23 % low
// (7.0799 vs 9.1996) and the branch came out 3.6 % small at every
// layer -- enough to flatten the logits 2x at long context and loop.
// HF Qwen3Next/Qwen3_5 GatedDeltaNet expands q/k with
// `repeat_interleave(num_v_heads // num_k_heads)`, which is exactly
// hk = hv / (nVHeads/nKHeads); every vendored fla kernel
// (chunk_o, chunk_delta_h, chunk_scaled_dot_kkt, gdn_wy_fast) indexes
// `i_h // (HV // H)`.
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
                                 uint32_t smemFast,
                                 uint32_t vHeadsTiled) {
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
    // GQA head mapping: value head hv reads k-head hv / (nVHeads /
    // nKHeads), i.e. CONTIGUOUS groups (HF `repeat_interleave`, fla's
    // `i_h // (HV // H)`), NOT hv % nKHeads.  See the note on
    // kQ35L2NormHeads.
    const uint32_t ng = nKHeads > 0u ? nVHeads / nKHeads : 1u;
    auto keyHeadOf = [&](uint32_t hv) -> uint32_t {
        return vHeadsTiled != 0u ? hv % (nKHeads > 0u ? nKHeads : 1u)
                                 : hv / ng;
    };
    __shared__ float siq[128];
    __shared__ float sik[128];
    if (fast) {
        for (uint32_t hv = threadIdx.x; hv < nVHeads; hv += blockDim.x) {
            uint32_t hk = keyHeadOf(hv);
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
            uint32_t hk = keyHeadOf(hv);
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
        uint32_t hk = keyHeadOf(hv);
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

// Per-head gated RMS norm (2026-11-04): the original kernel used ONE
// block per token and walked the 32 value heads serially with a
// __syncthreads per head (~32 us/token, ~0.8 ms of the 9B token).  One
// block per (token, head) keeps every SM busy; the per-head reduction
// order is independent of seqLen, so prefill and decode stay
// bit-identical.
__global__ void kQ35GatedRmsNormHeads(const float *__restrict__ z,
                                      float *__restrict__ out,
                                      const float *__restrict__ ssmNorm,
                                      uint32_t headV, uint32_t nVHeads,
                                      uint32_t seqLen) {
    const uint32_t s = blockIdx.x;
    const uint32_t hv = blockIdx.y;
    if (s >= seqLen || hv >= nVHeads) return;
    const uint32_t t = threadIdx.x;
    const float *zRow =
            z + (static_cast<size_t>(s) * nVHeads + hv) * headV;
    float *o = out + (static_cast<size_t>(s) * nVHeads + hv) * headV;
    float acc = 0.0f;
    for (uint32_t i = t; i < headV; i += blockDim.x) {
        const float v = o[i];
        acc = fmaf(v, v, acc);
    }
    __shared__ float red[32];
    float w = acc;
    for (uint32_t off = 16; off; off >>= 1)
        w += __shfl_down_sync(0xffffffffu, w, off);
    if ((t & 31u) == 0u) red[t >> 5] = w;
    __syncthreads();
    if (t == 0u) {
        float sum = 0.0f;
        const uint32_t nw = (blockDim.x + 31u) >> 5;
        for (uint32_t i = 0; i < nw; ++i) sum += red[i];
        red[0] = rsqrtf(sum / static_cast<float>(headV) + 1e-6f);
    }
    __syncthreads();
    const float rms = red[0];
    for (uint32_t i = t; i < headV; i += blockDim.x) {
        const float sv = zRow[i];
        const float silu = sv * q35Sigmoid(sv);
        o[i] = o[i] * rms * ssmNorm[i] * silu;
    }
}

// Per-head L2 normalization + q/k/v scatter (2026-11-04).  Same
// reasoning as kQ35GatedRmsNormHeads: the old kernel handled all 32
// value heads in one 128-thread block.
__global__ void kQ35L2NormHeads(const float *__restrict__ convOut,
                                float *__restrict__ qN,
                                float *__restrict__ kN,
                                float *__restrict__ vN, uint32_t headK,
                                uint32_t headV, uint32_t nKHeads,
                                uint32_t nVHeads, uint32_t keyDim,
                                uint32_t valueDim, uint32_t seqLen,
                                uint32_t vHeadsTiled) {
    const uint32_t s = blockIdx.x;
    const uint32_t hv = blockIdx.y;
    if (s >= seqLen || hv >= nVHeads) return;
    // GGUF (llama.cpp converter): tiled V-head order, k-head = hv % nK.
    // HF/exllamav3/Prism: grouped, k-head = hv / (nVHeads / nKHeads).
    const uint32_t hk = vHeadsTiled != 0u
                                ? hv % (nKHeads > 0u ? nKHeads : 1u)
                                : hv / (nVHeads / (nKHeads > 0u ? nKHeads : 1u));
    const float *conv = convOut + static_cast<size_t>(s) *
                                          (2 * keyDim + valueDim);
    const float *qSrc = conv + static_cast<size_t>(hk) * headK;
    const float *kSrc = conv + keyDim + static_cast<size_t>(hk) * headK;
    const float *vSrc = conv + 2 * keyDim +
                        static_cast<size_t>(hv) * headV;
    const uint32_t t = threadIdx.x;
    double sq = 0.0, sk = 0.0;
    for (uint32_t j = t; j < headK; j += blockDim.x) {
        const double qv = qSrc[j];
        const double kv = kSrc[j];
        sq += qv * qv;
        sk += kv * kv;
    }
    __shared__ double redq[32];
    __shared__ double redk[32];
    for (uint32_t off = 16; off; off >>= 1) {
        sq += __shfl_down_sync(0xffffffffu, sq, off);
        sk += __shfl_down_sync(0xffffffffu, sk, off);
    }
    if ((t & 31u) == 0u) {
        redq[t >> 5] = sq;
        redk[t >> 5] = sk;
    }
    __syncthreads();
    __shared__ float iq, ik;
    if (t == 0u) {
        double tq = 0.0, tk = 0.0;
        const uint32_t nw = (blockDim.x + 31u) >> 5;
        for (uint32_t i = 0; i < nw; ++i) {
            tq += redq[i];
            tk += redk[i];
        }
        iq = static_cast<float>(1.0 / std::sqrt(fmax(tq, 1e-6)));
        ik = static_cast<float>(1.0 / std::sqrt(fmax(tk, 1e-6)));
    }
    __syncthreads();
    float *qDst = qN + (static_cast<size_t>(s) * nVHeads + hv) * headK;
    float *kDst = kN + (static_cast<size_t>(s) * nVHeads + hv) * headK;
    float *vDst = vN + (static_cast<size_t>(s) * nVHeads + hv) * headV;
    for (uint32_t j = t; j < headK; j += blockDim.x) {
        qDst[j] = qSrc[j] * iq;
        kDst[j] = kSrc[j] * ik;
    }
    for (uint32_t j = t; j < headV; j += blockDim.x) {
        vDst[j] = vSrc[j];
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
                            uint32_t headDim, uint32_t cachePos,
                            const uint32_t *__restrict__ dpos) {
    uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    if (dpos != nullptr) cachePos = dpos[0];
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

// Int8 split-KV decode attention (one warp per (q-head, chunk)).
__global__ void kQ35WarpAttention8Split(
        const float *__restrict__ q, const int8_t *__restrict__ k8,
        const int8_t *__restrict__ v8, const float *__restrict__ ks,
        const float *__restrict__ vs, float *__restrict__ partAcc,
        float *__restrict__ partM, float *__restrict__ partL,
        uint32_t nHeads, uint32_t nKV, uint32_t cachePos,
        uint32_t numChunks, uint32_t chunkSize, uint32_t hd,
        float invSqrt) {
    constexpr uint32_t NV = 256 / 32;
    const uint32_t chunk = blockIdx.y;
    const uint32_t qHead = blockIdx.z;
    const uint32_t lane = threadIdx.x;
    if (qHead >= nHeads) return;
    const uint32_t kvHead = qHead / (nHeads / nKV);
    const uint32_t pidx = qHead * numChunks + chunk;
    float *pa = partAcc + static_cast<size_t>(pidx) * hd;
    const float *qPtr = q + static_cast<size_t>(qHead) * hd;
    float acc[NV];
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
    float m = -1e30f, l = 0.0f;
    uint32_t lo = chunk * chunkSize;
    if (lo <= cachePos) {
        uint32_t hi = lo + chunkSize - 1;
        if (hi > cachePos) hi = cachePos;
        for (uint32_t cs = lo; cs <= hi; ++cs) {
            const int8_t *kp =
                    k8 + (static_cast<size_t>(cs) * nKV + kvHead) * hd;
            const int8_t *vp =
                    v8 + (static_cast<size_t>(cs) * nKV + kvHead) * hd;
            const float sk = ks[static_cast<size_t>(cs) * nKV + kvHead];
            const float sv = vs[static_cast<size_t>(cs) * nKV + kvHead];
            float sc = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) {
                sc = fmaf(qPtr[lane + i * 32],
                          static_cast<float>(kp[lane + i * 32]) * sk,
                          sc);
            }
            sc = kWarpReduceSum(sc) * invSqrt;
            if (sc > m) {
                const float alpha = exp2f((m - sc) * LOG2E);
                for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
                l *= alpha;
                m = sc;
            }
            const float pp = exp2f((sc - m) * LOG2E);
            l += pp;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) {
                acc[i] = fmaf(pp,
                              static_cast<float>(vp[lane + i * 32]) * sv,
                              acc[i]);
            }
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) pa[lane + i * 32] = acc[i];
    if (lane == 0) {
        partM[pidx] = m;
        partL[pidx] = l;
    }
}

// Quantize F32 K/V (layout [token][nKV][hd]) into the decode int8 pool:
// int8 values + one scale per (token, kv-head).  srcToken starts at 0;
// destinations start at dstTokenBase.
__global__ void kQ35QuantKV8(const float *__restrict__ k,
                             const float *__restrict__ v,
                             int8_t *__restrict__ k8,
                             int8_t *__restrict__ v8,
                             float *__restrict__ ks,
                             float *__restrict__ vs, uint32_t tokens,
                             uint32_t nKV, uint32_t hd,
                             uint32_t dstTokenBase,
                             uint32_t srcTokenBase) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= tokens || h >= nKV) return;
    // srcTokenBase lets the rebuild of the int8 decode pool start at the
    // first row it is missing instead of re-quantizing the whole prefix.
    const uint32_t base = ((srcTokenBase + t) * nKV + h) * hd;
    const uint32_t dbase = (dstTokenBase + t) * nKV + h;
    __shared__ float smax[2][128];
    const uint32_t lane = threadIdx.x;
    float mk = 0.0f, mv = 0.0f;
    for (uint32_t i = lane; i < hd; i += blockDim.x) {
        mk = fmaxf(mk, fabsf(k[base + i]));
        mv = fmaxf(mv, fabsf(v[base + i]));
    }
    smax[0][lane] = mk;
    smax[1][lane] = mv;
    __syncthreads();
    for (uint32_t st = blockDim.x / 2; st > 0; st >>= 1) {
        if (lane < st) {
            smax[0][lane] = fmaxf(smax[0][lane], smax[0][lane + st]);
            smax[1][lane] = fmaxf(smax[1][lane], smax[1][lane + st]);
        }
        __syncthreads();
    }
    const float sk = smax[0][0] > 0.0f ? smax[0][0] / 127.0f : 1.0f;
    const float sv = smax[1][0] > 0.0f ? smax[1][0] / 127.0f : 1.0f;
    if (lane == 0) {
        ks[dbase] = sk;
        vs[dbase] = sv;
    }
    // 64-bit: (dstTokenBase + t) * nKVHeads * headDim reaches 2^32 for a
    // 1M window with wide K/V heads, and this kernel now runs at every
    // window the planner can grant.
    const size_t outBase =
            (static_cast<size_t>(dstTokenBase) + t) * nKV * hd +
            static_cast<size_t>(h) * hd;
    for (uint32_t i = lane; i < hd; i += blockDim.x) {
        const long qk = lrintf(k[base + i] / sk);
        const long qv = lrintf(v[base + i] / sv);
        k8[outBase + i] = static_cast<int8_t>(qk < -128 ? -128 : (qk > 127 ? 127 : qk));
        v8[outBase + i] = static_cast<int8_t>(qv < -128 ? -128 : (qv > 127 ? 127 : qv));
    }
}

// Int8 KV decode attention (eager, one warp per q-head).  Used only for
// seqLen == 1; prefill keeps the F32 cache.
__global__ void kQ35WarpAttention8(const float *__restrict__ q,
                                   const int8_t *__restrict__ k8,
                                   const int8_t *__restrict__ v8,
                                   const float *__restrict__ ks,
                                   const float *__restrict__ vs,
                                   float *__restrict__ out,
                                   uint32_t nHeads, uint32_t nKV,
                                   uint32_t cachePos, uint32_t hd,
                                   float invSqrt) {
    constexpr uint32_t NV = 256 / 32;
    const uint32_t warpId = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t lane = threadIdx.x;
    if (warpId >= nHeads) return;
    const uint32_t qHead = warpId;
    const uint32_t kvHead = qHead / (nHeads / nKV);
    const float *qPtr = q + static_cast<size_t>(qHead) * hd;
    float acc[NV];
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
    float m = -1e30f, l = 0.0f;
    for (uint32_t cs = 0; cs <= cachePos; ++cs) {
        const int8_t *kp = k8 + (static_cast<size_t>(cs) * nKV + kvHead) * hd;
        const int8_t *vp = v8 + (static_cast<size_t>(cs) * nKV + kvHead) * hd;
        const float sk = ks[static_cast<size_t>(cs) * nKV + kvHead];
        const float sv = vs[static_cast<size_t>(cs) * nKV + kvHead];
        float sc = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < NV; ++i) {
            sc = fmaf(qPtr[lane + i * 32],
                      static_cast<float>(kp[lane + i * 32]) * sk, sc);
        }
        sc = kWarpReduceSum(sc) * invSqrt;
        if (sc > m) {
            const float mNew = sc;
            const float alpha = exp2f((m - mNew) * LOG2E);
            for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
            l *= alpha;
            m = mNew;
        }
        const float p = exp2f((sc - m) * LOG2E);
        l += p;
#pragma unroll
        for (uint32_t i = 0; i < NV; ++i) {
            acc[i] = fmaf(p, static_cast<float>(vp[lane + i * 32]) * sv,
                          acc[i]);
        }
    }
    const float invL = 1.0f / l;
    float *op = out + static_cast<size_t>(qHead) * hd;
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) op[lane + i * 32] = acc[i] * invL;
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

// Decode-only fusion of the two 48-row F32 GDN projections
// (ssm_alpha, ssm_beta) with their per-head epilogue.  One launch
// instead of three: blockIdx < rows computes alpha's row (=head) and
// applies gateV = softplus(alpha + dt_bias) * ssm_a; the rest compute
// beta's row and apply beta = sigmoid(beta).  The per-row dot is the
// SAME per-thread strided accumulation plus warp/shared reduction as
// kQ35F32Gemv (bit-identical), and the epilogue is the same expression
// as kQ35BetaGate, so this is a pure launch fusion.
__global__ void kQ35F32GemvDualBeta(
        const float *__restrict__ x, const float *__restrict__ wAlpha,
        const float *__restrict__ wBeta, float *__restrict__ outAlpha,
        float *__restrict__ outBeta,
        const float *__restrict__ dtBias,
        const float *__restrict__ aBroadcast, uint32_t rows,
        uint32_t cols) {
    const uint32_t bi = blockIdx.x;
    const bool isBeta = bi >= rows;
    const uint32_t row = isBeta ? (bi - rows) : bi;
    if (row >= rows) return;
    const float *w = (isBeta ? wBeta : wAlpha) +
                     static_cast<size_t>(row) * cols;
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        acc = fmaf(x[i], w[i], acc);
    }
    for (uint32_t off = 16; off; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    __shared__ float srow[32];
    if (threadIdx.x % 32 == 0) srow[threadIdx.x / 32] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t wq = 0; wq < blockDim.x / 32; ++wq)
            sum += srow[wq];
        if (isBeta) {
            outBeta[row] = q35Sigmoid(sum);
        } else {
            const float a = q35Softplus(sum + dtBias[row]);
            outAlpha[row] = a * aBroadcast[row];
        }
    }
}

// Decode-step fusion of kQ35L2NormHeads and the beta/alpha GEMV.
// Both are per (token, head) and both must finish before the scan, so
// they were two launches back to back for no reason: the norm's 128
// values and the two 5120-dots are microseconds of work each and the
// trace put four GDN launches at 26 us/layer.  One 256-thread block per
// head: lanes 0..127 normalise q, 128..255 normalise k (and copy v), then
// all 256 threads run the beta dot, then the alpha dot, with the same
// warp-shfl + smem reduce and the same epilogue as kQ35F32GemvDualBeta.
// The norm keeps its double accumulation and 1e-6 floor verbatim.
__global__ void kQ35NormBetaAlpha(
        const float *__restrict__ convOut, float *__restrict__ qN,
        float *__restrict__ kN, float *__restrict__ vN,
        const float *__restrict__ x, const float *__restrict__ wAlpha,
        const float *__restrict__ wBeta, float *__restrict__ outAlpha,
        float *__restrict__ outBeta, const float *__restrict__ dtBias,
        const float *__restrict__ aBroadcast, uint32_t headK,
        uint32_t headV, uint32_t nKHeads, uint32_t nVHeads,
        uint32_t keyDim, uint32_t valueDim, uint32_t vHeadsTiled,
        uint32_t cols) {
    const uint32_t s = blockIdx.x;
    const uint32_t hv = blockIdx.y;
    if (s >= 1u || hv >= nVHeads) return;
    const uint32_t nk = nKHeads > 0u ? nKHeads : 1u;
    const uint32_t hk = vHeadsTiled != 0u ? hv % nk : hv / (nVHeads / nk);
    const float *conv = convOut + static_cast<size_t>(s) *
                                          (2 * keyDim + valueDim);
    const float *qSrc = conv + static_cast<size_t>(hk) * headK;
    const float *kSrc = conv + keyDim + static_cast<size_t>(hk) * headK;
    const float *vSrc = conv + 2 * keyDim +
                        static_cast<size_t>(hv) * headV;
    const uint32_t t = threadIdx.x;
    __shared__ float siq, sik;
    __shared__ float red[32];
    // ---- phase 1: per-head L2 norms (double accumulate, 1e-6 floor)
    if (t < 128u) {
        const bool isK = t >= 64u;
        const uint32_t lane = t & 63u;
        const float *src = isK ? kSrc : qSrc;
        double acc = 0.0;
        for (uint32_t j = lane; j < headK; j += 64u) {
            const double v = src[j];
            acc += v * v;
        }
        for (uint32_t off = 16u; off; off >>= 1u)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if ((t & 31u) == 0u) red[t >> 5] = static_cast<float>(acc);
    }
    __syncthreads();
    if (t == 0) {
        float sq = 0.0f, sk = 0.0f;
        for (uint32_t g = 0; g < 2u; ++g) sq += red[g];
        for (uint32_t g = 2u; g < 4u; ++g) sk += red[g];
        siq = rsqrtf(sq > 1e-6f ? sq : 1e-6f);
        sik = rsqrtf(sk > 1e-6f ? sk : 1e-6f);
    }
    __syncthreads();
    // ---- phase 2: write qN/kN/vN (no reduction needed)
    // Destination heads are indexed by VALUE head, not key head: the
    // source (convOut) is indexed by hk.  Getting this wrong silently
    // mixes q/k heads and still answers short questions.
    const size_t qBase = static_cast<size_t>(s * nVHeads + hv) * headK;
    const size_t vBase = static_cast<size_t>(s * nVHeads + hv) * headV;
    for (uint32_t j = t; j < headK; j += blockDim.x) {
        qN[qBase + j] = qSrc[j] * siq;
        kN[qBase + j] = kSrc[j] * sik;
    }
    for (uint32_t j = t; j < headV; j += blockDim.x)
        vN[vBase + j] = vSrc[j];
    // ---- phase 3: beta and alpha dots (same math as the dual GEMV)
    const uint32_t bi = hv;
    float acc = 0.0f;
    const float *wb = wBeta + static_cast<size_t>(bi) * cols;
    for (uint32_t i = t; i < cols; i += blockDim.x)
        acc = fmaf(x[i], wb[i], acc);
    for (uint32_t off = 16u; off; off >>= 1u)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __syncthreads();
    if ((t & 31u) == 0u) red[t >> 5] = acc;
    __syncthreads();
    if (t == 0) {
        float sum = 0.0f;
        for (uint32_t g = 0; g < blockDim.x / 32u; ++g) sum += red[g];
        outBeta[bi] = q35Sigmoid(sum);
    }
    __syncthreads();
    const float *wa = wAlpha + static_cast<size_t>(bi) * cols;
    acc = 0.0f;
    for (uint32_t i = t; i < cols; i += blockDim.x)
        acc = fmaf(x[i], wa[i], acc);
    for (uint32_t off = 16u; off; off >>= 1u)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __syncthreads();
    if ((t & 31u) == 0u) red[t >> 5] = acc;
    __syncthreads();
    if (t == 0) {
        float sum = 0.0f;
        for (uint32_t g = 0; g < blockDim.x / 32u; ++g) sum += red[g];
        const float a = q35Softplus(sum + dtBias[bi]);
        outAlpha[bi] = a * aBroadcast[bi];
    }
}

// Batched F32 projection: one block per (row, token) running the SAME
// per-thread strided dot + warp/shared reduction as kQ35F32Gemv, so
// every element is bit-identical to the per-token decode GEMV.  The
// ssm_alpha/ssm_beta matrices feed the GDN decay/beta, where any
// reassociation propagates into the recurrent state and breaks the
// batch-vs-sequential contract (measured maxDiff 7.9 when prefill used
// cuBLAS Sgemm instead).
__global__ void kQ35F32GemmBatch(const float *__restrict__ x,
                                 const float *__restrict__ w,
                                 float *__restrict__ out, uint32_t rows,
                                 uint32_t cols, uint32_t seqLen) {
    constexpr uint32_t TOKENS_PER_BLOCK = 8;
    const uint32_t row = blockIdx.x;
    const uint32_t s0 = blockIdx.y * TOKENS_PER_BLOCK;
    if (row >= rows || s0 >= seqLen) return;
    const float *wRow = w + static_cast<size_t>(row) * cols;
    __shared__ float srow[32];
    const uint32_t nTok = min(TOKENS_PER_BLOCK, seqLen - s0);
    for (uint32_t t = 0; t < nTok; ++t) {
        const uint32_t s = s0 + t;
        const float *xRow = x + static_cast<size_t>(s) * cols;
        // float4 + 4 accumulators (2026-11-04): the scalar strided
        // loop was 1 fma per thread per step; this covers 4x the
        // columns per step with 4 independent chains.  The reduction
        // order changes, so the batch-vs-sequential gate (top-1,
        // maxDiff < 40) is what this must keep.
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        const uint32_t stride = blockDim.x * 4u;
        uint32_t i = threadIdx.x * 4u;
        for (; i + 3u < cols; i += stride) {
            const float4 xv = *reinterpret_cast<const float4 *>(xRow + i);
            const float4 wv = *reinterpret_cast<const float4 *>(wRow + i);
            acc.x = fmaf(xv.x, wv.x, acc.x);
            acc.y = fmaf(xv.y, wv.y, acc.y);
            acc.z = fmaf(xv.z, wv.z, acc.z);
            acc.w = fmaf(xv.w, wv.w, acc.w);
        }
        float accS = (acc.x + acc.y) + (acc.z + acc.w);
        for (; i < cols; ++i) accS = fmaf(xRow[i], wRow[i], accS);
        float acc2 = accS;
        for (uint32_t off = 16; off; off >>= 1) {
            acc2 += __shfl_down_sync(0xffffffffu, acc2, off);
        }
        if (threadIdx.x % 32 == 0) srow[threadIdx.x / 32] = acc2;
        __syncthreads();
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (uint32_t wq = 0; wq < blockDim.x / 32; ++wq)
                sum += srow[wq];
            out[static_cast<size_t>(s) * rows + row] = sum;
        }
        // srow is reused by the next token in this block.
        __syncthreads();
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

// kQ35ConvStep with the K taps spread over K threads instead of one.
// The per-thread form was 3.1 us/layer at 60 blocks x 256 = 15360
// threads (480 warps, ~7 per SM) with SEVEN dependent global accesses
// per thread, so it is latency-bound with a ninth of the machine's warp
// slots used.  Here one thread owns (channel, tap): 61K threads over 240
// blocks, one load and one fma each, then a 4-lane shuffle reduction.
// The state shift stays on the tap-0 thread; it is inherent to the
// shift-register layout.  Fixed at K == 4 (the qwen35 GDN conv).
__global__ void kQ35ConvStep4(const float *__restrict__ convWeight,
                              const float *__restrict__ qkv,
                              float *__restrict__ convIn,
                              float *__restrict__ convOut,
                              float *__restrict__ convState,
                              uint32_t channels) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t c = t >> 2;
    const uint32_t i = t & 3u;
    if (c >= channels) return;
    // window: convIn[0..2] = state[0..2], convIn[3] = qkv
    const float xin = i < 3u ? convState[i * channels + c] : qkv[c];
    convIn[i * channels + c] = xin;
    const float wv = convWeight[static_cast<size_t>(c) * 4u + i];
    float sum = wv * xin;
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    if (i == 0u) {
        convOut[c] = sum * q35Sigmoid(sum);
        convState[0 * channels + c] = convState[1 * channels + c];
        convState[1 * channels + c] = convState[2 * channels + c];
        convState[2 * channels + c] = qkv[c];
    }
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
// Vectorized (float4) row helpers for the GDN recurrence.  The FP
// accumulation ORDER is unchanged (x,y,z,w in index order) -- only the
// load width changes -- so results stay bit-identical to the scalar
// loops.  The scalar form spent ~33 ms/layer at 512 tokens on serial
// 4-byte state loads (L2-latency bound, 16 blocks); float4 lowers the
// load count 4x and lets the compiler keep many loads in flight.
//
// FOUR independent accumulators (one per float4 lane) break the
// 128-long serial FMA chain: the recurrence runs only 16 blocks (one
// per value head), so the chain latency dominated the prefill (4.3 ms
// per 512-token layer with one accumulator).  The smem transposed
// kernel uses the SAME i%4 grouping and the same (a0+a1)+(a2+a3)
// combine, so batch prefill and decode remain bit-identical to each
// other.  (They differ from the old strictly-sequential sum by ~1e-6
// relative -- fp32 reassociation only.)
__device__ __forceinline__ float q35RowDotF4(const float *__restrict__ a,
                                             const float *__restrict__ b,
                                             uint32_t n4) {
    const float4 *a4 = reinterpret_cast<const float4 *>(a);
    const float4 *b4 = reinterpret_cast<const float4 *>(b);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 32
    for (uint32_t i = 0; i < n4; ++i) {
        const float4 x = a4[i];
        const float4 y = b4[i];
        a0 = fmaf(x.x, y.x, a0);
        a1 = fmaf(x.y, y.y, a1);
        a2 = fmaf(x.z, y.z, a2);
        a3 = fmaf(x.w, y.w, a3);
    }
    return (a0 + a1) + (a2 + a3);
}
// Same i%4 grouping for a strided (transposed smem) row.
__device__ __forceinline__ float q35ColDot4(const float *__restrict__ sm,
                                            uint32_t stride, uint32_t j,
                                            const float *__restrict__ b,
                                            uint32_t n) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 32
    for (uint32_t i = 0; i < n; i += 4) {
        a0 = fmaf(sm[static_cast<size_t>(i + 0u) * stride + j], b[i + 0u],
                  a0);
        a1 = fmaf(sm[static_cast<size_t>(i + 1u) * stride + j], b[i + 1u],
                  a1);
        a2 = fmaf(sm[static_cast<size_t>(i + 2u) * stride + j], b[i + 2u],
                  a2);
        a3 = fmaf(sm[static_cast<size_t>(i + 3u) * stride + j], b[i + 3u],
                  a3);
    }
    return (a0 + a1) + (a2 + a3);
}
__device__ __forceinline__ void q35RowScaleF4(float *__restrict__ a,
                                              float dec, uint32_t n4) {
    float4 *a4 = reinterpret_cast<float4 *>(a);
#pragma unroll 32
    for (uint32_t i = 0; i < n4; ++i) {
        float4 v = a4[i];
        v.x *= dec;
        v.y *= dec;
        v.z *= dec;
        v.w *= dec;
        a4[i] = v;
    }
}
__device__ __forceinline__ void q35RowAxpyF4(float *__restrict__ a,
                                             const float *__restrict__ b,
                                             float del, uint32_t n4) {
    float4 *a4 = reinterpret_cast<float4 *>(a);
    const float4 *b4 = reinterpret_cast<const float4 *>(b);
#pragma unroll 32
    for (uint32_t i = 0; i < n4; ++i) {
        float4 x = a4[i];
        const float4 y = b4[i];
        x.x = fmaf(del, y.x, x.x);
        x.y = fmaf(del, y.y, x.y);
        x.z = fmaf(del, y.z, x.z);
        x.w = fmaf(del, y.w, x.w);
        a4[i] = x;
    }
}

// Warp-per-row GDN chunk scan (2026-11-04): the fastest mapping for
// the recurrence.  One warp owns ONE state row; each lane holds 4
// contiguous columns of that row in REGISTERS for the whole chunk, so
// the 64 KiB state is never re-read from smem/global between tokens.
// Rows are independent, so blocks contain 4 warps = 4 rows and the 16
// heads x 128 rows become 512 blocks (the 128-thread/row-per-thread
// scans left 52 SMs idle and were latency-bound at 3.5 ms/layer).
//
// Numerics: the per-lane 4-element partial is summed in order, then
// the 32 lanes combine with a shfl tree.  This is a deterministic fp32
// reassociation of the sequential reference (~1e-6 relative); the same
// kernel serves chunk prefill and single-token decode, so batch and
// sequential stay identical to each other.
// Bit-exact chunk scan: same warp-per-row mapping as
// kQ35GdnWarpScan, but the per-token dot is reduced in the EXACT
// reference order (4 accumulators over i%4, i ascending, then
// (a0+a1)+(a2+a3)) instead of a warp tree.  The warp tree is ~1e-6
// off the per-token kernel per token; on the qwen35moe trunk that
// drift flips the knife-edge expert router, so prefill and decode must
// agree bit-for-bit.  Lane 0 walks the 32 float4 groups in order while
// every lane shfl-broadcasts its 4 state and 4 activation values, so
// the fma chain is identical to kQ35GdnStepBatch's q35RowDotF4.
template<uint32_t HEADV>
__global__ void kQ35GdnScanExact(const float *__restrict__ qN,
                                 const float *__restrict__ kN,
                                 const float *__restrict__ vN,
                                 const float *__restrict__ beta,
                                 const float *__restrict__ decayV,
                                 float *__restrict__ state,
                                 float *__restrict__ gdnOut,
                                 uint32_t nVHeads, uint32_t valueDim,
                                 uint32_t seqLen, float invSqrtHeadV) {
    constexpr uint32_t RPB = 4u;
    constexpr uint32_t NBLK = HEADV / RPB;
    const uint32_t hv = blockIdx.x / NBLK;
    const uint32_t rowBase = (blockIdx.x % NBLK) * RPB;
    if (hv >= nVHeads) return;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = rowBase + warp;
    if (row >= HEADV) return;
    const uint32_t col = lane * 4u;
    const uint32_t kvn = static_cast<uint32_t>(HEADV) * nVHeads;
    float4 s = *reinterpret_cast<const float4 *>(
            state + (static_cast<size_t>(hv) * HEADV + row) * HEADV +
            col);
    auto dotRef = [&](const float4 &x4) -> float {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
        for (uint32_t g = 0; g < 32u; ++g) {
            const float sx = __shfl_sync(0xffffffffu, s.x, g);
            const float sy = __shfl_sync(0xffffffffu, s.y, g);
            const float sz = __shfl_sync(0xffffffffu, s.z, g);
            const float sw = __shfl_sync(0xffffffffu, s.w, g);
            const float xx = __shfl_sync(0xffffffffu, x4.x, g);
            const float xy = __shfl_sync(0xffffffffu, x4.y, g);
            const float xz = __shfl_sync(0xffffffffu, x4.z, g);
            const float xw = __shfl_sync(0xffffffffu, x4.w, g);
            a0 = fmaf(sx, xx, a0);
            a1 = fmaf(sy, xy, a1);
            a2 = fmaf(sz, xz, a2);
            a3 = fmaf(sw, xw, a3);
        }
        return (a0 + a1) + (a2 + a3);
    };
    // Software prefetch (2026-11-04): the loop is a serial chain over
    // tokens, so without lookahead every step stalled on the k/q load
    // (~0.86 ms/layer for 512 tokens).  Issue the NEXT token's k/q
    // loads before the current token's math so the latency hides
    // behind the dot/update.
    auto loadTok = [&](uint32_t t, float4 &kk, float4 &qq) {
        kk = *reinterpret_cast<const float4 *>(
                kN + static_cast<size_t>(t) * kvn + hv * HEADV + col);
        qq = *reinterpret_cast<const float4 *>(
                qN + static_cast<size_t>(t) * kvn + hv * HEADV + col);
    };
    float4 k4, q4;
    if (seqLen > 0) loadTok(0, k4, q4);
    for (uint32_t t = 0; t < seqLen; ++t) {
        float4 kNext, qNext;
        if (t + 1u < seqLen) loadTok(t + 1u, kNext, qNext);
        const float dec =
                __expf(decayV[static_cast<size_t>(t) * nVHeads + hv]);
        s.x *= dec;
        s.y *= dec;
        s.z *= dec;
        s.w *= dec;
        const float dot = dotRef(k4);
        const float del =
                (vN[static_cast<size_t>(t) * kvn + hv * HEADV + row] -
                 dot) *
                beta[static_cast<size_t>(t) * nVHeads + hv];
        s.x = fmaf(del, k4.x, s.x);
        s.y = fmaf(del, k4.y, s.y);
        s.z = fmaf(del, k4.z, s.z);
        s.w = fmaf(del, k4.w, s.w);
        const float outRow = dotRef(q4);
        if (lane == 0u)
            gdnOut[static_cast<size_t>(t) * valueDim + hv * HEADV + row] =
                    outRow * invSqrtHeadV;
    }
    *reinterpret_cast<float4 *>(
            state + (static_cast<size_t>(hv) * HEADV + row) * HEADV +
            col) = s;
}

// Full-warp sum with a butterfly, so EVERY lane ends up holding the
// total.  For any fixed lane the butterfly performs exactly the
// additions the down-tree + broadcast pair performed, in the same
// order (lane 0 adds p16, then p8 which already holds p8+p24, ...), so
// it is bit-identical to `down-tree then __shfl_sync(.., 0)` and simply
// costs one shuffle instead of two.
__device__ __forceinline__ float q35WarpSumAll(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// One warp owns RPW consecutive state rows; a block owns
// RPW * RPB of them.  RPW = 1, RPB = 4 is the original geometry.
//
// The recurrence for a row is a 128-long dot spread over the warp's 32
// lanes (4 columns each), so growing RPW does NOT change any row's
// arithmetic or summation order -- it only lets one warp amortise the
// per-(token, head) k/q vector over several rows and, far more
// importantly, drops the number of BLOCKS per head.  That block count
// was the cost: with 32 blocks per head the same 1 KiB (k,q) slice was
// fetched once per block per token, i.e. 32x per token per head, and
// the re-reads are spread far enough apart in time that L2 (6 MiB) did
// not hold them: 512 tok x 48 heads x 32 blocks x 1 KiB = 768 MiB of
// DRAM traffic per layer, which at ~550 GB/s is 1.4 ms -- the measured
// per-layer cost of this kernel.
// MAXBLK is a minimum-blocks-per-SM hint (1 = no constraint).  The
// kernel is latency-bound, so the resident-warp count is a lever: 47
// registers caps it at 10 blocks (40 warps) of 128 threads.
// 8 lanes per row, 4 rows per warp, RPB warps per block.
//
// The 32-lane scan needs a 5-step shuffle tree for each of its two dots
// (10 SHFL per (token, row)).  With 8 lanes on a row, each lane owns the
// four column groups i, i+8, i+16, i+24, so the first two tree levels
// (d=16 and d=8) collapse into two register adds and only 3 lane steps
// remain.  The tree still combines
//   (p_i + p_{i+16}) + (p_{i+8} + p_{i+24})  then d=4, d=2, d=1
// with the same partial-product order (mul, then three fmaf), so the
// output is BIT-IDENTICAL to kQ35GdnWarpScan; only the shuffle count
// and the number of independent rows in flight change.
template<uint32_t HEADV, uint32_t RPB = 4u>
__global__ void __launch_bounds__(RPB * 32u, 1)
        kQ35GdnWarpScan8(const float *__restrict__ qN,
                         const float *__restrict__ kN,
                         const float *__restrict__ vN,
                         const float *__restrict__ beta,
                         const float *__restrict__ decayV,
                         float *__restrict__ state,
                         float *__restrict__ gdnOut, uint32_t nVHeads,
                         uint32_t valueDim, uint32_t seqLen,
                         float invSqrtHeadV) {
    constexpr uint32_t RPW = 4u;// rows per warp = lane groups
    constexpr uint32_t ROWS_PER_BLOCK = RPB * RPW;
    constexpr uint32_t NBLK = HEADV / ROWS_PER_BLOCK;
    const uint32_t hv = blockIdx.x / NBLK;
    const uint32_t rowBase = (blockIdx.x % NBLK) * ROWS_PER_BLOCK;
    if (hv >= nVHeads) return;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t g = lane >> 3;// row owned by this lane group
    const uint32_t j = lane & 7u;// lane inside the 8-lane row group
    const uint32_t rowWarp = rowBase + warp * RPW;
    const uint32_t row = rowWarp + g;
    if (row >= HEADV) return;
    const uint32_t kvn = static_cast<uint32_t>(HEADV) * nVHeads;
    // column group gg covers columns 4*(j + 8*gg) .. +3
    const size_t sBase =
            (static_cast<size_t>(hv) * HEADV + row) * HEADV + 4u * j;
    const size_t tokBase = static_cast<size_t>(hv) * HEADV + 4u * j;
    float4 s[4];
#pragma unroll
    for (uint32_t gg = 0; gg < 4u; ++gg)
        s[gg] = *reinterpret_cast<const float4 *>(state + sBase + 32u * gg);
    // 3-step tree over the 8 lanes of a row group: same partial sums as
    // the 32-lane XOR butterfly, hence identical on every lane.
    auto groupSum = [&](float v) {
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        return v;
    };
    const uint32_t vBase = hv * HEADV + rowWarp + g;
    for (uint32_t t = 0; t < seqLen; ++t) {
        float4 kv[4], qv[4];
        const float *kb = kN + static_cast<size_t>(t) * kvn + tokBase;
        const float *qb = qN + static_cast<size_t>(t) * kvn + tokBase;
#pragma unroll
        for (uint32_t gg = 0; gg < 4u; ++gg) {
            kv[gg] = *reinterpret_cast<const float4 *>(kb + 32u * gg);
            qv[gg] = *reinterpret_cast<const float4 *>(qb + 32u * gg);
        }
        const float dec =
                __expf(decayV[static_cast<size_t>(t) * nVHeads + hv]);
        const float bt = beta[static_cast<size_t>(t) * nVHeads + hv];
        const float vv =
                vN[static_cast<size_t>(t) * kvn + vBase];
        float part[4];
#pragma unroll
        for (uint32_t gg = 0; gg < 4u; ++gg) {
            s[gg].x *= dec;
            s[gg].y *= dec;
            s[gg].z *= dec;
            s[gg].w *= dec;
            float p = s[gg].x * kv[gg].x;
            p = fmaf(s[gg].y, kv[gg].y, p);
            p = fmaf(s[gg].z, kv[gg].z, p);
            p = fmaf(s[gg].w, kv[gg].w, p);
            part[gg] = p;
        }
        // d=16 then d=8 collapse into registers
        const float tot =
                groupSum((part[0] + part[2]) + (part[1] + part[3]));
        const float del = (vv - tot) * bt;
#pragma unroll
        for (uint32_t gg = 0; gg < 4u; ++gg) {
            s[gg].x = fmaf(del, kv[gg].x, s[gg].x);
            s[gg].y = fmaf(del, kv[gg].y, s[gg].y);
            s[gg].z = fmaf(del, kv[gg].z, s[gg].z);
            s[gg].w = fmaf(del, kv[gg].w, s[gg].w);
        }
        float pq[4];
#pragma unroll
        for (uint32_t gg = 0; gg < 4u; ++gg) {
            float p = s[gg].x * qv[gg].x;
            p = fmaf(s[gg].y, qv[gg].y, p);
            p = fmaf(s[gg].z, qv[gg].z, p);
            p = fmaf(s[gg].w, qv[gg].w, p);
            pq[gg] = p;
        }
        const float po = groupSum((pq[0] + pq[2]) + (pq[1] + pq[3]));
        if (j == 0u)
            gdnOut[static_cast<size_t>(t) * valueDim + hv * HEADV +
                   row] = po * invSqrtHeadV;
    }
#pragma unroll
    for (uint32_t gg = 0; gg < 4u; ++gg)
        *reinterpret_cast<float4 *>(state + sBase + 32u * gg) = s[gg];
}

template<uint32_t HEADV, uint32_t RPB = 4u, uint32_t RPW = 1u,
         bool PF = true, bool PFS = false, uint32_t MAXBLK = 1u>
__global__ void __launch_bounds__(RPB * 32u, MAXBLK) kQ35GdnWarpScan(const float *__restrict__ qN,
                                                                     const float *__restrict__ kN,
                                                                     const float *__restrict__ vN,
                                                                     const float *__restrict__ beta,
                                                                     const float *__restrict__ decayV,
                                                                     float *__restrict__ state,
                                                                     float *__restrict__ gdnOut,
                                                                     uint32_t nVHeads, uint32_t valueDim,
                                                                     uint32_t seqLen, float invSqrtHeadV) {
    // rows (warps) per block; more rows per block share the same
    // per-(token, head) k/q vectors through L1.
    constexpr uint32_t ROWS_PER_BLOCK = RPB * RPW;
    constexpr uint32_t NBLK = HEADV / ROWS_PER_BLOCK;
    const uint32_t hv = blockIdx.x / NBLK;
    const uint32_t rowBase = (blockIdx.x % NBLK) * ROWS_PER_BLOCK;
    if (hv >= nVHeads) return;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = rowBase + warp * RPW;
    if (row >= HEADV) return;
    const uint32_t col = lane * 4u;// this lane's 4 contiguous columns
    const uint32_t kvn = static_cast<uint32_t>(HEADV) * nVHeads;
    const size_t sBase =
            (static_cast<size_t>(hv) * HEADV + row) * HEADV + col;
    float4 s[RPW];
#pragma unroll
    for (uint32_t r = 0; r < RPW; ++r)
        s[r] = *reinterpret_cast<const float4 *>(
                state + sBase + static_cast<size_t>(r) * HEADV);
    auto loadTokW = [&](uint32_t t, float4 &kk, float4 &qq) {
        kk = *reinterpret_cast<const float4 *>(
                kN + static_cast<size_t>(t) * kvn + hv * HEADV + col);
        qq = *reinterpret_cast<const float4 *>(
                qN + static_cast<size_t>(t) * kvn + hv * HEADV + col);
    };
    float4 k4 = {0.f, 0.f, 0.f, 0.f}, q4 = {0.f, 0.f, 0.f, 0.f};
    if (PF && seqLen > 0) loadTokW(0, k4, q4);
    const uint32_t vBase = hv * HEADV + row;
    // The per-token scalars (decay, beta, v) are COLD loads: each
    // (token, head) element is touched once per layer, so they miss
    // L1 and sit in the dependency chain.  PFS hoists them one token
    // ahead like k/q, which is a pure scheduling change.
    float dec = 0.f, bt = 0.f;
    float vv[RPW];
    auto loadScalars = [&](uint32_t t, float &d, float &b, float *v) {
        d = __expf(decayV[static_cast<size_t>(t) * nVHeads + hv]);
        b = beta[static_cast<size_t>(t) * nVHeads + hv];
        const size_t vOff = static_cast<size_t>(t) * kvn + vBase;
        if (RPW == 4u) {
            // RPW consecutive rows = one aligned 16 B load
            const float4 v4 =
                    *reinterpret_cast<const float4 *>(vN + vOff);
            v[0] = v4.x;
            v[1] = v4.y;
            v[2] = v4.z;
            v[3] = v4.w;
        } else {
#pragma unroll
            for (uint32_t r = 0; r < RPW; ++r) v[r] = vN[vOff + r];
        }
    };
    if (PFS && seqLen > 0) loadScalars(0, dec, bt, vv);
    for (uint32_t t = 0; t < seqLen; ++t) {
        float4 kNext, qNext;
        if (PF && t + 1u < seqLen) loadTokW(t + 1u, kNext, qNext);
        if (!PF) loadTokW(t, k4, q4);
        float decCur = 0.f, btCur = 0.f;
        float vCur[RPW];
        if (PFS) {
            decCur = dec;
            btCur = bt;
#pragma unroll
            for (uint32_t r = 0; r < RPW; ++r) vCur[r] = vv[r];
            if (t + 1u < seqLen) loadScalars(t + 1u, dec, bt, vv);
        } else {
            loadScalars(t, decCur, btCur, vCur);
        }
        float part[RPW];
#pragma unroll
        for (uint32_t r = 0; r < RPW; ++r) {
            s[r].x *= decCur;
            s[r].y *= decCur;
            s[r].z *= decCur;
            s[r].w *= decCur;
            float p = s[r].x * k4.x;
            p = fmaf(s[r].y, k4.y, p);
            p = fmaf(s[r].z, k4.z, p);
            p = fmaf(s[r].w, k4.w, p);
            part[r] = p;
        }
        // `del` drives the state update on EVERY lane (each lane owns 4
        // of the 128 columns), so the total must reach all lanes, not
        // just lane 0.  Without that each lane subtracts its own
        // suffix-sum remnant and the persistent delta-net state is
        // corrupted: gdnOut still looked right (it is written from lane
        // 0), which is why the corruption hid until the state
        // overflowed at ~39 tokens.
#pragma unroll
        for (uint32_t r = 0; r < RPW; ++r) part[r] = q35WarpSumAll(part[r]);
#pragma unroll
        for (uint32_t r = 0; r < RPW; ++r) {
            const float del = (vCur[r] - part[r]) * btCur;
            s[r].x = fmaf(del, k4.x, s[r].x);
            s[r].y = fmaf(del, k4.y, s[r].y);
            s[r].z = fmaf(del, k4.z, s[r].z);
            s[r].w = fmaf(del, k4.w, s[r].w);
        }
#pragma unroll
        for (uint32_t r = 0; r < RPW; ++r) {
            float po = s[r].x * q4.x;
            po = fmaf(s[r].y, q4.y, po);
            po = fmaf(s[r].z, q4.z, po);
            po = fmaf(s[r].w, q4.w, po);
            po += __shfl_down_sync(0xffffffffu, po, 16);
            po += __shfl_down_sync(0xffffffffu, po, 8);
            po += __shfl_down_sync(0xffffffffu, po, 4);
            po += __shfl_down_sync(0xffffffffu, po, 2);
            po += __shfl_down_sync(0xffffffffu, po, 1);
            if (lane == 0u)
                gdnOut[static_cast<size_t>(t) * valueDim + hv * HEADV +
                       row + r] = po * invSqrtHeadV;
        }
        if (PF && t + 1u < seqLen) {
            k4 = kNext;
            q4 = qNext;
        }
    }
#pragma unroll
    for (uint32_t r = 0; r < RPW; ++r)
        *reinterpret_cast<float4 *>(
                state + sBase + static_cast<size_t>(r) * HEADV) = s[r];
}

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
    constexpr uint32_t Sv4 = Sv / 4u;
    const float *qHead = qN + static_cast<size_t>(hv) * HEADV;
    const float *kHead = kN + static_cast<size_t>(hv) * HEADV;
    const float *vHead = vN + static_cast<size_t>(hv) * HEADV;
    float *rowJ = state + static_cast<size_t>(hv) * HEADV * HEADV +
                  static_cast<size_t>(j) * Sv;
    // decay S *= exp(gateV[hv])
    const float dec = __expf(decayV[hv]);
    q35RowScaleF4(rowJ, dec, Sv4);
    __syncthreads();
    // delta[j] = (v[j] - dot(row_j(M), k)) * beta[hv]
    const float dot = q35RowDotF4(rowJ, kHead, Sv4);
    const float del = (vHead[j] - dot) * beta[hv];
    // M[j][i] += delta[j] * k[i]  (only row j touched)
    q35RowAxpyF4(rowJ, kHead, del, Sv4);
    // out[j] = dot(row_j(M), q) * scale
    const float outRow = q35RowDotF4(rowJ, qHead, Sv4);
    gdnOut[static_cast<size_t>(hv) * HEADV + j] = outRow * invSqrtHeadV;
}

// Batched chunk-scan gated-delta-net: ONE block per value head, each
// block walks the WHOLE token chunk in order.  The heads are fully
// independent (private [headV][headV] state), and the per-head state
// stays resident in the block's L1/register working set across the
// scan, so this is bit-exact with seqLen launches of kQ35GdnStepBatch
// while removing the launch + cold-state latency per token.  The math
// per token is the identical code, in the identical order.
template<uint32_t HEADV>
__global__ void kQ35GdnScan(const float *__restrict__ qN,
                            const float *__restrict__ kN,
                            const float *__restrict__ vN,
                            const float *__restrict__ beta,
                            const float *__restrict__ decayV,
                            float *__restrict__ state,
                            float *__restrict__ gdnOut,
                            uint32_t nVHeads, uint32_t valueDim,
                            uint32_t seqLen, float invSqrtHeadV) {
    const uint32_t hv = blockIdx.x;
    if (hv >= nVHeads) return;
    const uint32_t j = threadIdx.x;
    if (j >= HEADV) return;
    constexpr uint32_t Sv = HEADV;
    constexpr uint32_t Sv4 = Sv / 4u;
    const uint32_t kvn = static_cast<uint32_t>(HEADV) * nVHeads;
    float *rowJ = state + static_cast<size_t>(hv) * HEADV * HEADV +
                  static_cast<size_t>(j) * Sv;
    for (uint32_t s = 0; s < seqLen; ++s) {
        const float *qHead =
                qN + static_cast<size_t>(s) * kvn + hv * HEADV;
        const float *kHead =
                kN + static_cast<size_t>(s) * kvn + hv * HEADV;
        const float *vHead =
                vN + static_cast<size_t>(s) * kvn + hv * HEADV;
        const float dec = __expf(decayV[static_cast<size_t>(s) * nVHeads +
                                        hv]);
        q35RowScaleF4(rowJ, dec, Sv4);
        __syncthreads();
        const float dot = q35RowDotF4(rowJ, kHead, Sv4);
        const float del =
                (vHead[j] - dot) *
                beta[static_cast<size_t>(s) * nVHeads + hv];
        q35RowAxpyF4(rowJ, kHead, del, Sv4);
        const float outRow = q35RowDotF4(rowJ, qHead, Sv4);
        gdnOut[static_cast<size_t>(s) * valueDim + hv * HEADV + j] =
                outRow * invSqrtHeadV;
        __syncthreads();
    }
}

// Shared-memory chunk-scan GDN: identical math to kQ35GdnScan, but the
// per-head state lives in dynamic shared memory for the whole chunk and
// is written back once at the end.  The global-state scan re-read 64 KiB
// per head per token from L2 (global stores bypass L1 and evict the
// lines), which made it latency-bound at ~33 ms per 512-token layer.
// Rows are private to their thread, so no barrier is needed inside the
// token loop.
//
// The state is kept TRANSPOSED in smem: sm[i*RPB + j] = M[rowBase+j][i].
// Thread j then reads its own COLUMN for a fixed i, so the 32 lanes of
// a warp touch 32 consecutive floats (all 32 banks); a row-major layout
// with a 128-float row stride gives a 32-way bank conflict per load.
//
// RPB = 32 rows per block: the 16 value heads become 64 independent
// blocks (each 16 KiB smem, one warp), so all 64 SMs of the 2080 Ti run
// the recurrence instead of 16 -- the 128-thread/64 KiB variant was
// limited to one block per SM (4 warps) and cost 4.2 ms/layer.
template<uint32_t HEADV, uint32_t RPB>
__global__ void kQ35GdnScanSmem(const float *__restrict__ qN,
                                const float *__restrict__ kN,
                                const float *__restrict__ vN,
                                const float *__restrict__ beta,
                                const float *__restrict__ decayV,
                                float *__restrict__ state,
                                float *__restrict__ gdnOut,
                                uint32_t nVHeads, uint32_t valueDim,
                                uint32_t seqLen, float invSqrtHeadV) {
    constexpr uint32_t NBLK = HEADV / RPB;
    const uint32_t hv = blockIdx.x / NBLK;
    const uint32_t rowBase = (blockIdx.x % NBLK) * RPB;
    if (hv >= nVHeads) return;
    const uint32_t j = threadIdx.x;
    if (j >= RPB) return;
    constexpr uint32_t Sv = HEADV;
    extern __shared__ float smState[];
    float *stateG = state + static_cast<size_t>(hv) * HEADV * HEADV;
    const uint32_t row = rowBase + j;
    for (uint32_t i = 0; i < Sv; ++i)
        smState[static_cast<size_t>(i) * RPB + j] =
                stateG[static_cast<size_t>(row) * Sv + i];
    __syncthreads();
    const uint32_t kvn = static_cast<uint32_t>(HEADV) * nVHeads;
    for (uint32_t s = 0; s < seqLen; ++s) {
        const float *qHead =
                qN + static_cast<size_t>(s) * kvn + hv * HEADV;
        const float *kHead =
                kN + static_cast<size_t>(s) * kvn + hv * HEADV;
        const float *vHead =
                vN + static_cast<size_t>(s) * kvn + hv * HEADV;
        const float dec = __expf(decayV[static_cast<size_t>(s) * nVHeads +
                                        hv]);
        for (uint32_t i = 0; i < Sv; ++i)
            smState[static_cast<size_t>(i) * RPB + j] *= dec;
        const float dot =
                q35ColDot4(smState, RPB, j, kHead, Sv);
        const float del =
                (vHead[row] - dot) *
                beta[static_cast<size_t>(s) * nVHeads + hv];
        for (uint32_t i = 0; i < Sv; ++i)
            smState[static_cast<size_t>(i) * RPB + j] =
                    fmaf(del, kHead[i],
                         smState[static_cast<size_t>(i) * RPB + j]);
        const float outRow =
                q35ColDot4(smState, RPB, j, qHead, Sv);
        gdnOut[static_cast<size_t>(s) * valueDim + hv * HEADV + row] =
                outRow * invSqrtHeadV;
    }
    __syncthreads();
    for (uint32_t i = 0; i < Sv; ++i)
        stateG[static_cast<size_t>(row) * Sv + i] =
                smState[static_cast<size_t>(i) * RPB + j];
}
// Token-parallel causal conv (K<=8): every (token, channel) is an
// independent dot over qkv[s-K+1..s]; only the first K-1 tokens read
// the persistent pre-chunk state.  Replaces the serial per-channel
// kQ35ConvScan for prefill.
__global__ void kQ35ConvParallel(const float *__restrict__ convWeight,
                                 const float *__restrict__ qkv,
                                 float *__restrict__ convOut,
                                 const float *__restrict__ convState,
                                 uint32_t channels, uint32_t K,
                                 uint32_t seqLen) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t s = blockIdx.y;
    if (c >= channels || s >= seqLen || K > 8) return;
    const float *wRow = convWeight + static_cast<size_t>(c) * K;
    float sum = 0.0f;
    for (uint32_t i = 0; i < K; ++i) {
        const int idx = static_cast<int>(s) - static_cast<int>(K - 1) +
                        static_cast<int>(i);
        float v;
        if (idx >= 0)
            v = qkv[static_cast<size_t>(idx) * channels + c];
        else
            v = convState[static_cast<size_t>(idx + K - 1) * channels + c];
        sum = fmaf(wRow[i], v, sum);
    }
    convOut[static_cast<size_t>(s) * channels + c] =
            sum * q35Sigmoid(sum);
}

__global__ void kQ35ConvStateUpdate(const float *__restrict__ qkv,
                                    float *__restrict__ convState,
                                    uint32_t channels, uint32_t K,
                                    uint32_t seqLen) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels || K <= 1 || K > 8) return;
    float tmp[8];
    for (uint32_t i = 0; i + 1 < K; ++i) {
        const int src = static_cast<int>(seqLen) - static_cast<int>(K - 1) +
                        static_cast<int>(i);
        tmp[i] = src >= 0
                         ? qkv[static_cast<size_t>(src) * channels + c]
                         : convState[static_cast<size_t>(src + K - 1) *
                                             channels +
                                     c];
    }
    for (uint32_t i = 0; i + 1 < K; ++i)
        convState[static_cast<size_t>(i) * channels + c] = tmp[i];
}

__global__ void kQ35ConvScan(const float *__restrict__ convWeight,
                             const float *__restrict__ qkv,
                             float *__restrict__ convOut,
                             float *__restrict__ convState,
                             uint32_t channels, uint32_t K,
                             uint32_t seqLen) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float *wRow = convWeight + static_cast<size_t>(c) * K;
    float win[8];
    if (K > 8) return;// geometry guard (qwen35 uses K=4)
    for (uint32_t i = 0; i < K - 1; ++i)
        win[i] = convState[static_cast<size_t>(i) * channels + c];
    for (uint32_t s = 0; s < seqLen; ++s) {
        win[K - 1] = qkv[static_cast<size_t>(s) * channels + c];
        float sum = 0.0f;
        for (uint32_t i = 0; i < K; ++i)
            sum = fmaf(wRow[i], win[i], sum);
        convOut[static_cast<size_t>(s) * channels + c] =
                sum * q35Sigmoid(sum);
        // shift state: state[i] = state[i+1], last = qkv
        for (uint32_t i = 0; i + 1 < K - 1; ++i) win[i] = win[i + 1];
        if (K >= 2) win[K - 2] = qkv[static_cast<size_t>(s) * channels + c];
    }
    for (uint32_t i = 0; i < K - 1; ++i)
        convState[static_cast<size_t>(i) * channels + c] = win[i];
}

// Warp flash attention for qwen35 full-attention layers (headDim=256:
// 8 fp32 accumulators per lane, warp per (token, q-head), causal from
// cachePos).  Mirrors kWarpAttention with template HD=256.


// ---- EXL3 prefill Hadamard on the ACTIVATIONS -------------------
// W = Dsu Hk What Hn Dsv, so W^T x = Dsv Hn What^T Hk Dsu x: both
// 128-point transforms can be applied to the activations (O(T*k) and
// O(T*n)) instead of to the weight matrix (O(k*n) twice per prefill).
// That removes 420 MB of traffic per linear from the decode path.
// MUST match exl3gpu's fwht128Smem (XOR butterflies, L = 1,2,4,...,64)
// or the weight-side and activation-side transforms differ and the
// prefill logits diverge from the per-token GEMV reference.
// ---- Warp-level 128-point FWHT --------------------------------
// One warp owns one contiguous 128-element chunk, four consecutive
// elements per lane, so the whole transform is registers + shuffles:
// no shared memory and no __syncthreads.  The BUTTERFLY ORDER is
// identical to the shared-memory version it replaces
// (L = 1,2,4,...,64; b[i] = (i&L) ? a[i^L]-a[i] : a[i]+a[i^L]), so the
// output is bit-identical -- only the transport changed.  Measured
// reason: the smem version ran at 190-210 GB/s (3x under this card's
// streaming peak) because every 128-element block paid 14 syncthreads
// and two smem round trips for 512 B of payload.
__device__ __forceinline__ void fwht128Warp(float *v, int lane) {
    // L = 1, 2: both partners live in this lane's four elements.
    {
        const float a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
        v[0] = a0 + a1;
        v[1] = a0 - a1;
        v[2] = a2 + a3;
        v[3] = a2 - a3;
        const float b0 = v[0], b1 = v[1], b2 = v[2], b3 = v[3];
        v[0] = b0 + b2;
        v[2] = b0 - b2;
        v[1] = b1 + b3;
        v[3] = b1 - b3;
    }
    // Butterfly span L = 4..64: the partner of element i = 4*lane + j is
    // 4*(lane ^ (L/4)) + j, i.e. the same slot in lane ^ (L/4).  Only
    // log2(L) = s is needed, as the shfl.xor mask.
#pragma unroll
    for (int s = 0; s < 5; ++s) {
        const int mask = 1 << s;
        const bool hi = (lane & mask) != 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float part = __shfl_xor_sync(0xffffffffu, v[j], mask);
            v[j] = hi ? (part - v[j]) : (v[j] + part);
        }
    }
}

// Legacy shared-memory twin of fwht128Warp, kept as a bit-exact A/B
// oracle (TINYCODER_WHT_LEGACY=1) and as the reference the warp version
// was validated against.  Same butterfly order, so the two produce
// identical bytes.
__device__ __forceinline__ void fwht128BlkLegacy(float *a, float *b) {
    for (uint32_t L = 1; L < 128u; L <<= 1) {
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) {
            const float x = a[i], y = a[i ^ L];
            b[i] = (i & L) ? (y - x) : (x + y);
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) a[i] = b[i];
        __syncthreads();
    }
}
__global__ void kExl3FwhtToF16Legacy(const float *__restrict__ in,
                                     __half *__restrict__ out,
                                     const float *__restrict__ signs,
                                     uint32_t C) {
    __shared__ float s[128];
    __shared__ float t[128];
    const uint32_t r = blockIdx.y;
    const uint32_t c0 = blockIdx.x * 128u;
    const float *src = in + static_cast<size_t>(r) * C + c0;
    const float sc = 0.08838834764831845f;
    for (int i = threadIdx.x; i < 128; i += blockDim.x)
        s[i] = src[i] * (signs ? signs[c0 + i] : 1.0f);
    __syncthreads();
    fwht128BlkLegacy(s, t);
    for (int i = threadIdx.x; i < 128; i += blockDim.x)
        out[static_cast<size_t>(r) * C + c0 + i] = __float2half_rn(s[i] * sc);
}
__global__ void kExl3FwhtF32Legacy(float *__restrict__ y,
                                   const float *__restrict__ signs,
                                   uint32_t C) {
    __shared__ float s[128];
    __shared__ float t[128];
    const uint32_t r = blockIdx.y;
    const uint32_t c0 = blockIdx.x * 128u;
    float *dst = y + static_cast<size_t>(r) * C + c0;
    const float sc = 0.08838834764831845f;
    for (int i = threadIdx.x; i < 128; i += blockDim.x) s[i] = dst[i];
    __syncthreads();
    fwht128BlkLegacy(s, t);
    for (int i = threadIdx.x; i < 128; i += blockDim.x)
        dst[i] = s[i] * sc * (signs ? signs[c0 + i] : 1.0f);
}

// x(FP32, row-major [rows][C]) -> H128(signs * x) * scale as FP16.
// Launch: dim3(ceil((C/128)/WARPS), rows), WARPS*32 threads.
// `mul`, when non-null, fuses the dense-FFN activation into this pass:
// x = silu(x) * mul, evaluated with EXACTLY the kSiluMul expression so
// the result is bit-identical to running kSiluMul first.  That removes
// a full read+read+write trip over the gate and up buffers (measured
// 197 us per layer, 12.6 ms per 512-token prefill).
template<int WARPS>
__global__ void kExl3FwhtToF16(const float *__restrict__ in,
                               __half *__restrict__ out,
                               const float *__restrict__ signs,
                               uint32_t C,
                               const float *__restrict__ mul = nullptr) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const uint32_t chunk =
            blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (chunk >= (C >> 7)) return;
    const uint32_t r = blockIdx.y;
    const uint32_t c0 = chunk * 128u;
    const size_t off = static_cast<size_t>(r) * C + c0 + 4u * lane;
    const float4 f4 = *reinterpret_cast<const float4 *>(in + off);
    float v[4] = {f4.x, f4.y, f4.z, f4.w};
    if (mul != nullptr) {
        const float4 m4 = *reinterpret_cast<const float4 *>(mul + off);
        const float mv[4] = {m4.x, m4.y, m4.z, m4.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float x = v[j];
            v[j] = __fmul_rn(x / (1.0f + __expf(-x)), mv[j]);
        }
    }
    float sg[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (signs != nullptr) {
        const float4 s4 = *reinterpret_cast<const float4 *>(
                signs + c0 + 4u * lane);
        sg[0] = s4.x;
        sg[1] = s4.y;
        sg[2] = s4.z;
        sg[3] = s4.w;
    }
    // __fmul_rn, not `*=`: nvcc contracts a bare multiply followed by
    // the first butterfly into an FMA, which the shared-memory version
    // physically cannot do (the product goes through smem).  That one
    // rounding difference moved 5 of 15360 elements by 1 ulp and the
    // 27B's top-1 logit by 1.6e-4, i.e. it IS observable.
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __fmul_rn(v[j], sg[j]);
    fwht128Warp(v, lane);
    const float sc = 0.08838834764831845f;// 1/sqrt(128)
    const __half2 a = __floats2half2_rn(v[0] * sc, v[1] * sc);
    const __half2 b = __floats2half2_rn(v[2] * sc, v[3] * sc);
    uint2 pk;
    pk.x = *reinterpret_cast<const uint32_t *>(&a);
    pk.y = *reinterpret_cast<const uint32_t *>(&b);
    *reinterpret_cast<uint2 *>(out + static_cast<size_t>(r) * C + c0 +
                               4u * lane) = pk;
}

// y = signs * H128(y) * scale, in place, FP32 [rows][C].
template<int WARPS>
__global__ void kExl3FwhtF32(float *__restrict__ y,
                             const float *__restrict__ signs,
                             uint32_t C) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const uint32_t chunk =
            blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (chunk >= (C >> 7)) return;
    const uint32_t r = blockIdx.y;
    const uint32_t c0 = chunk * 128u;
    float *dst = y + static_cast<size_t>(r) * C + c0 + 4u * lane;
    const float4 f4 = *reinterpret_cast<const float4 *>(dst);
    float v[4] = {f4.x, f4.y, f4.z, f4.w};
    fwht128Warp(v, lane);
    const float sc = 0.08838834764831845f;
    float sg[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (signs != nullptr) {
        const float4 s4 = *reinterpret_cast<const float4 *>(
                signs + c0 + 4u * lane);
        sg[0] = s4.x;
        sg[1] = s4.y;
        sg[2] = s4.z;
        sg[3] = s4.w;
    }
    float4 o;
    o.x = __fmul_rn(__fmul_rn(v[0], sc), sg[0]);
    o.y = __fmul_rn(__fmul_rn(v[1], sc), sg[1]);
    o.z = __fmul_rn(__fmul_rn(v[2], sc), sg[2]);
    o.w = __fmul_rn(__fmul_rn(v[3], sc), sg[3]);
    *reinterpret_cast<float4 *>(dst) = o;
}

// ---- GPU router (TINYCODER_GPU_ROUTER=1) ------------------------
// The hybrid path routes on the CPU: fp32 norm @ routerW, double
// softmax, stable top-k, renormalise.  Measured 295 ms per 512-token
// prefill (7.2 ms/layer) and 6.2 ms per decode token -- more than the
// GPU spends on the shared expert that is supposed to hide it.  This
// kernel does the identical selection on the device: one block per
// token, one thread per expert (fp32 sequential dot, same summation
// order as the per-token matMulVec reference), then a serial top-k
// over the probabilities.
// Router logits: one WARP per (token, expert) pair, lane-strided dot
// plus a warp reduction.  The naive one-thread-per-expert version is
// latency-bound on the dependent FMA chain (measured 0.24 ms/layer at
// decode, which cost 18% of decode); this spreads each dot over 32
// lanes so the chain is 32x shorter.
__global__ void kQ35MoeRouterLogits(const float *__restrict__ norm,
                                    const float *__restrict__ W,
                                    float *__restrict__ logits,
                                    uint32_t H, uint32_t E,
                                    uint32_t nTokExp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t idx = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (idx >= nTokExp) return;
    const uint32_t s = idx / E;
    const uint32_t e = idx % E;
    const float *x = norm + static_cast<size_t>(s) * H;
    const float *wr = W + static_cast<size_t>(e) * H;
    float acc = 0.0f;
    for (uint32_t i = lane; i < H; i += 32u)
        acc = fmaf(x[i], wr[i], acc);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) logits[idx] = acc;
}

// Softmax + stable top-k over one token's logits (thread 0 does the
// selection; E <= 256 so this is a few hundred scalar ops per token).
__global__ void kQ35MoeRouterSelect(const float *__restrict__ logits,
                                    int32_t *__restrict__ idx,
                                    float *__restrict__ wgt, uint32_t E,
                                    uint32_t K) {
    extern __shared__ float sh[];// probs[E]
    const uint32_t s = blockIdx.x;
    if (threadIdx.x != 0) return;
    const float *lg = logits + static_cast<size_t>(s) * E;
    float maxL = -3.4e38f;
    for (uint32_t i = 0; i < E; ++i) maxL = fmaxf(maxL, lg[i]);
    double sumExp = 0.0;
    for (uint32_t i = 0; i < E; ++i) {
        const double v = exp(static_cast<double>(lg[i]) - maxL);
        sh[i] = static_cast<float>(v);
        sumExp += v;
    }
    const float invSum = static_cast<float>(1.0 / sumExp);
    for (uint32_t i = 0; i < E; ++i) sh[i] *= invSum;
    int32_t *io = idx + static_cast<size_t>(s) * K;
    float *wo = wgt + static_cast<size_t>(s) * K;
    double wsum = 0.0;
    for (uint32_t r = 0; r < K; ++r) {
        float best = -1.0f;
        uint32_t bi = 0;
        for (uint32_t i = 0; i < E; ++i) {
            if (sh[i] > best) {
                best = sh[i];
                bi = i;
            }
        }
        io[r] = static_cast<int32_t>(bi);
        wo[r] = best;
        wsum += static_cast<double>(best);
        sh[bi] = -1.0f;
    }
    const float wsumClamped =
            fmaxf(static_cast<float>(wsum), 6.103515625e-5f);
    for (uint32_t r = 0; r < K; ++r) wo[r] /= wsumClamped;
}

// ---- Prefill attention with a shared K/V row (2026-11-10) --------
// kQ35WarpAttention gives every (token, head) warp its own private
// stream of the K/V cache: for a 512-token prompt that is O(seqLen^2)
// x nHeads of global traffic and it is why prefill attention runs ~8x
// slower than llama's tiled kernel (~25 ms/layer vs ~7 ms).
//
// Here one block owns ALL heads of ONE token, and the K/V row for a
// position is staged ONCE in shared memory and read by every head
// warp.  Global K/V traffic drops by the head count; the arithmetic
// per head is IDENTICAL (same dot order, same online-softmax order),
// so the output is bit-identical to kQ35WarpAttention and can be
// swapped behind TINYCODER_ATTN_MMA=1 without a parity risk.
template<uint32_t HD, uint32_t W>
__global__ void kQ35PrefillAttnSharedKV(
        const float *__restrict__ q, const float *__restrict__ kCache,
        const float *__restrict__ vCache, float *__restrict__ out,
        uint32_t seqLen, uint32_t nHeads, uint32_t nKVHeads,
        uint32_t cachePos, const uint32_t *__restrict__ dpos,
        float invSqrt) {
    static_assert(HD == 256, "qwen35 uses headDim=256");
    constexpr uint32_t NV = HD / 32;
    extern __shared__ float sh[];
    float *shK = sh;// HD floats
    float *shV = sh + HD;
    if (dpos != nullptr) cachePos = dpos[0];
    const uint32_t s = blockIdx.x;
    const uint32_t head0 = blockIdx.y * W + threadIdx.y;
    const uint32_t lane = threadIdx.x;
    if (s >= seqLen) return;
    const bool active = head0 < nHeads;
    const uint32_t kvHead = active ? head0 / (nHeads / nKVHeads) : 0;
    const uint32_t csEnd = cachePos + s;
    const float *qPtr =
            q + (static_cast<size_t>(s) * nHeads + head0) * HD;
    float acc[NV];
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
    float m = -1e30f, l = 0.0f;
    for (uint32_t cs = 0; cs <= csEnd; ++cs) {
        // One warp stages the row; every head warp then reads it from
        // shared memory instead of re-fetching it from global.
        if (threadIdx.y == 0) {
            const float *kPtr =
                    kCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) *
                                     HD;
            const float *vPtr =
                    vCache + (static_cast<size_t>(cs) * nKVHeads + kvHead) *
                                     HD;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) {
                shK[lane + i * 32] = kPtr[lane + i * 32];
                shV[lane + i * 32] = vPtr[lane + i * 32];
            }
        }
        __syncthreads();
        if (active) {
            float sc = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i)
                sc = fmaf(qPtr[lane + i * 32], shK[lane + i * 32], sc);
            sc = kWarpReduceSum(sc) * invSqrt;
            if (sc > m) {
                const float alpha = exp2f((m - sc) * LOG2E);
                for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
                l *= alpha;
                m = sc;
            }
            const float p = exp2f((sc - m) * LOG2E);
            l += p;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i)
                acc[i] = fmaf(p, shV[lane + i * 32], acc[i]);
        }
        __syncthreads();
    }
    if (!active) return;
    const float invL = 1.0f / l;
    float *oPtr = out + (static_cast<size_t>(s) * nHeads + head0) * HD;
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) oPtr[lane + i * 32] = acc[i] * invL;
}

// ---- int8 K/V store: prefill attention over the quantized cache ----
// Same math as the f32 kernels above; the row is dequantized as
// int8 * scale[token][kvHead].  The store (kQ35QuantKV8) runs first, so
// these read exactly what they wrote.  Used when the planned window is
// granted only with an int8 K/V store; the whole point is that the row
// costs 1 byte instead of 4 at every position, so a 64k window fits.
// One block per (query token, kv head): every warp of the block owns one
// QUERY head of that kv head's group, so the row staged into shared
// memory is the one all of them need.  The first version of this kernel
// spread a block over `W` arbitrary query heads (like the f32 kernel it
// was copied from), which mixes kv heads inside one block whenever
// nHeads/nKVHeads does not divide W — on Bonsai (24 heads / 4 kv, group
// 6, W=8) heads 6..7 of block 0 were attending block 0's staging of kv
// head 0 and the model answered "000000: routine maintenance log entry"
// instead of the access code.  The f32 original never hit it because it
// is opt-in (TINYCODER_ATTN_MMA) and defaults off.
// The block shape is (32 lanes, one warp per query head of the kv
// head's group), so there is no compile-time warp count to specialize
// on -- the group comes from nHeads/nKVHeads at launch.
template<uint32_t HD>
__global__ void kQ35PrefillAttnSharedKV8(
        const float *__restrict__ q, const int8_t *__restrict__ kCache8,
        const int8_t *__restrict__ vCache8, const float *__restrict__ ks,
        const float *__restrict__ vs, float *__restrict__ out,
        uint32_t seqLen, uint32_t nHeads, uint32_t nKVHeads,
        uint32_t cachePos, const uint32_t *__restrict__ dpos,
        float invSqrt) {
    static_assert(HD == 256, "qwen35 uses headDim=256");
    constexpr uint32_t NV = HD / 32;
    extern __shared__ float sh[];
    float *shK = sh;// HD floats
    float *shV = sh + HD;
    if (dpos != nullptr) cachePos = dpos[0];
    const uint32_t s = blockIdx.x;
    const uint32_t kvHead = blockIdx.y;
    const uint32_t group = nHeads / nKVHeads;
    const uint32_t head0 = kvHead * group + threadIdx.y;
    const uint32_t lane = threadIdx.x;
    if (s >= seqLen) return;
    const bool active = threadIdx.y < group;
    const uint32_t csEnd = cachePos + s;
    const float *qPtr =
            q + (static_cast<size_t>(s) * nHeads + head0) * HD;
    float acc[NV];
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
    float m = -1e30f, l = 0.0f;
    for (uint32_t cs = 0; cs <= csEnd; ++cs) {
        // One warp stages the row, dequantizing on the way into shared
        // memory so the head warps' inner loop is identical to f32.
        if (threadIdx.y == 0) {
            const int8_t *kPtr =
                    kCache8 + (static_cast<size_t>(cs) * nKVHeads + kvHead) *
                                      HD;
            const int8_t *vPtr =
                    vCache8 + (static_cast<size_t>(cs) * nKVHeads + kvHead) *
                                      HD;
            const float sk = ks[static_cast<size_t>(cs) * nKVHeads + kvHead];
            const float sv = vs[static_cast<size_t>(cs) * nKVHeads + kvHead];
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i) {
                shK[lane + i * 32] = static_cast<float>(kPtr[lane + i * 32]) * sk;
                shV[lane + i * 32] = static_cast<float>(vPtr[lane + i * 32]) * sv;
            }
        }
        __syncthreads();
        if (active) {
            float sc = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i)
                sc = fmaf(qPtr[lane + i * 32], shK[lane + i * 32], sc);
            sc = kWarpReduceSum(sc) * invSqrt;
            if (sc > m) {
                const float alpha = exp2f((m - sc) * LOG2E);
                for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
                l *= alpha;
                m = sc;
            }
            const float p = exp2f((sc - m) * LOG2E);
            l += p;
#pragma unroll
            for (uint32_t i = 0; i < NV; ++i)
                acc[i] = fmaf(p, shV[lane + i * 32], acc[i]);
        }
        __syncthreads();
    }
    if (!active) return;
    const float invL = 1.0f / l;
    float *oPtr = out + (static_cast<size_t>(s) * nHeads + head0) * HD;
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) oPtr[lane + i * 32] = acc[i] * invL;
}

template<uint32_t HD, uint32_t W>
__global__ void kQ35WarpAttention8Prefill(
        const float *__restrict__ q, const int8_t *__restrict__ k8,
        const int8_t *__restrict__ v8, const float *__restrict__ ks,
        const float *__restrict__ vs, float *__restrict__ out,
        uint32_t seqLen, uint32_t nHeads, uint32_t nKVHeads,
        uint32_t cachePos, const uint32_t *__restrict__ dpos,
        float invSqrt) {
    if (dpos != nullptr) cachePos = dpos[0];
    static_assert(HD == 256, "qwen35 uses headDim=256");
    constexpr uint32_t NV = HD / 32;
    const uint32_t s = blockIdx.x;
    const uint32_t warpId = blockIdx.y * W + threadIdx.y;
    const uint32_t lane = threadIdx.x;
    if (s >= seqLen || warpId >= nHeads) return;
    const uint32_t qHead = warpId;
    const uint32_t kvHead = qHead / (nHeads / nKVHeads);
    const uint32_t csEnd = cachePos + s;
    const float *qPtr = q + (static_cast<size_t>(s) * nHeads + qHead) * HD;
    float acc[NV];
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) acc[i] = 0.0f;
    float m = -1e30f, l = 0.0f;
    for (uint32_t cs = 0; cs <= csEnd; ++cs) {
        const int8_t *kPtr =
                k8 + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
        const int8_t *vPtr =
                v8 + (static_cast<size_t>(cs) * nKVHeads + kvHead) * HD;
        const float sk = ks[static_cast<size_t>(cs) * nKVHeads + kvHead];
        const float sv = vs[static_cast<size_t>(cs) * nKVHeads + kvHead];
        float sc = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < NV; ++i) {
            sc = fmaf(qPtr[lane + i * 32],
                      static_cast<float>(kPtr[lane + i * 32]) * sk, sc);
        }
        sc = kWarpReduceSum(sc) * invSqrt;
        if (sc > m) {
            const float alpha = exp2f((m - sc) * LOG2E);
            for (uint32_t i = 0; i < NV; ++i) acc[i] *= alpha;
            l *= alpha;
            m = sc;
        }
        const float p = exp2f((sc - m) * LOG2E);
        l += p;
#pragma unroll
        for (uint32_t i = 0; i < NV; ++i) {
            acc[i] = fmaf(p, static_cast<float>(vPtr[lane + i * 32]) * sv,
                          acc[i]);
        }
    }
    const float invL = 1.0f / l;
    float *oPtr = out + (static_cast<size_t>(s) * nHeads + qHead) * HD;
#pragma unroll
    for (uint32_t i = 0; i < NV; ++i) oPtr[lane + i * 32] = acc[i] * invL;
}

template<uint32_t HD>
__global__ void kQ35WarpAttention(const float *__restrict__ q,
                                  const float *__restrict__ kCache,
                                  const float *__restrict__ vCache,
                                  float *__restrict__ out, uint32_t seqLen,
                                  uint32_t nHeads, uint32_t nKVHeads,
                                  uint32_t cachePos,
                                  const uint32_t *__restrict__ dpos,
                                  float invSqrt) {
    if (dpos != nullptr) cachePos = dpos[0];
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
