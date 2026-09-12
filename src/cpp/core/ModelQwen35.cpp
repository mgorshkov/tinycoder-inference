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

// ============================================================================
// Qwen35 (dense) architecture forward pass.
//
// Qwen3.8-27B is a dense hybrid transformer: 48 gated-delta-net recurrent
// layers + 16 full-attention layers + 1 MTP (Multi-Token Prediction) block.
// Reference: llama.cpp src/models/qwen35.cpp + delta-net-base.cpp.
//
// Layer classification (llama.cpp is_recr_impl):
//   is_recr(i) = (i < n_layer) && ((i+1) % full_attention_interval != 0)
//   n_layer = n_layer_all - nextn_predict_layers (64 for Qwen3.8-27B).
//
// Full-attention layer (i+1)%4 == 0:
//   attn_norm -> Q+G fused projection (attn_q, outputs [nHeads*2*headDim])
//   -> Q view (first headDim of each head), per-head Q RMSNorm (attn_q_norm,
//   single [headDim] shared across all heads) -> K/V projections, K per-head
//   RMSNorm (attn_k_norm) -> MRoPE (n_rot=64) -> attention
//   (kq_scale = 1/sqrt(headDim)) -> sigmoid(gate) elementwise -> attn_output
//   projection -> residual add.
//
// Recurrent layer (gated delta net):
//   attn_norm -> attn_qkv (Q|K|V fused: [keyDim*2 + valueDim] output),
//   attn_gate (z) -> conv1d (kernel 4 over conv state, silu) -> slice
//   q_conv/k_conv/v_conv -> L2-norm q/k -> repeat q,k over value heads
//   (periodic: value-head hv reads k-head hv % nKHeads) -> gated delta net
//   recurrence (per value-head [headV, headV] state M stored transposed:
//   M[j*Sv+i] = S[i][j], decay S *= exp(gate), delta, outer product,
//   output M@q * 1/sqrt(headV)) -> gated RMSNorm with silu(z) -> ssm_out
//   projection -> residual add.
//
// Both layer types then run: rmsNorm(post_attention_norm) -> SwiGLU FFN
// (ffn_gate/ffn_up/ffn_down) -> residual add onto the pre-post-norm tensor.
// ============================================================================

#include "Model.hpp"

#include "ModelInternal.hpp"
#include "SIMDMatMulVec.hpp"
#include "ThreadPool.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace tinycoder {

    namespace {

        /// @brief sigmoid(x) = 1/(1+exp(-x)).
        float q35Sigmoid(float x) {
            return 1.0f / (1.0f + std::exp(-x));
        }

        /// @brief softplus(x) = log(1+exp(x)), numerically stable.
        float q35Softplus(float x) {
            if (x > 20.0f) {
                return x;
            }
            return std::log1p(std::exp(x));
        }

        /// @brief RMSNorm over a head slice with a per-head shared weight.
        /// @param x Input slice of length n.
        /// @param out Output slice (may alias x).
        /// @param weight Norm weight (length n).
        /// @param n Slice length.
        static void q35HeadRmsNorm(const float *x, float *out,
                                   const float *weight, uint32_t n) {
            rmsNormSIMD(x, out, weight, n);
        }

    }// namespace

    // -----------------------------------------------------------------------
    // MRoPE (Multi-section RoPE) for full-attention qwen35 layers.
    //
    // Mirrors ggml_rope_multi with:
    //   - mode = GGML_ROPE_TYPE_IMROPE (40): mrope path with all theta bases
    //     equal to freq_base (qwen35 uses a single position id for text).
    //   - n_dims = n_rot = 64 (rope.dimension_count); headDim = 256.
    //   - sections = [11, 11, 10, 0]; sect_dims = 32 (pairs per period).
    //   - theta_scale = freq_base^(-2/n_dims).
    //   - rotate_pairs Neox-style: pair (x[i], x[i + n_dims/2]) rotated with
    //     cache entry i0 (i in [0, n_dims)). Remaining channels [n_dims, headDim)
    //     are copied unchanged.
    // Since all theta bases are identical for text, the sector pattern reduces
    // to the standard single-theta frequency progression: cache[i0] =
    // cos(p * freq_base * theta_scale^(i0/2)) with the rotation applied to
    // pairs (i, i+32) — equivalent to Neox RoPE with headDim-period 64.
    // -----------------------------------------------------------------------
    void Model::applyMRoPE(float *q, float *k, uint32_t qSeqLen, uint32_t kSeqLen,
                           uint32_t qHeads, uint32_t kHeads, uint32_t pos) {
        const uint32_t nDims = config_.ropeDimensionCount > 0
                                       ? config_.ropeDimensionCount
                                       : 64;
        const uint32_t nDimsHalf = nDims / 2;
        const uint32_t headDim = config_.headDim;
        const float thetaBase = config_.ropeTheta;
        const float thetaScale =
                std::pow(thetaBase, -2.0f / static_cast<float>(nDims));
        const uint32_t neededPos = pos + std::max(qSeqLen, kSeqLen);

        // The existing rope tables use freq = theta^{-2*d/headDim} (headDim=256),
        // NOT the MRoPE theta_scale (computed with nDims=64), so the MRoPE cache
        // is computed inline here. The standard rope tables are unused for qwen35.
        //
        // CRITICAL: ggml's ggml_mrope_cache_init() seeds theta_base with the
        // token POSITION (ggml_compute_forward_rope_flt passes p = pos[i2]),
        // so the angular frequency of pair k is  theta(k) = p * theta_scale^k
        // (freq_base enters ONLY through theta_scale = freq_base^(-2/n_dims)).
        // The cache must therefore START theta at 1.0f -- NOT at freq_base
        // (1e7). Seeding it with thetaBase injected an extra factor of 1e7 into
        // every angle, pseudo-randomizing cos/sin and scrambling every
        // full-attention layer (the recurrent layers 0-2 matched llama exactly
        // because they contain no RoPE; layer 3 -- the first full-attention
        // layer -- was the first to diverge).
        std::vector<float> cosCache(static_cast<size_t>(neededPos) * nDimsHalf);
        std::vector<float> sinCache(static_cast<size_t>(neededPos) * nDimsHalf);
        for (uint32_t p = 0; p < neededPos; ++p) {
            float theta = 1.0f;
            for (uint32_t d2 = 0; d2 < nDimsHalf; ++d2) {
                cosCache[static_cast<size_t>(p) * nDimsHalf + d2] =
                        std::cos(static_cast<float>(p) * theta);
                sinCache[static_cast<size_t>(p) * nDimsHalf + d2] =
                        std::sin(static_cast<float>(p) * theta);
                theta *= thetaScale;
            }
        }

        // Rotation helper (Neox-style: pair (k, k + nDimsHalf)).
        // Matches ggml's rotate_pairs(n_dims, n_dims/2): for k in [0, n_dims/2),
        // rotate the pair (head[k], head[k + n_dims/2]) with cache index k.
        auto rotateHead = [&](float *head, uint32_t p) {
            const float *cosRow =
                    cosCache.data() + static_cast<size_t>(p) * nDimsHalf;
            const float *sinRow =
                    sinCache.data() + static_cast<size_t>(p) * nDimsHalf;
            for (uint32_t k = 0; k < nDimsHalf; ++k) {
                float c = cosRow[k];
                float sn = sinRow[k];
                float x0 = head[k];
                float x1 = head[k + nDimsHalf];
                head[k] = x0 * c - x1 * sn;
                head[k + nDimsHalf] = x0 * sn + x1 * c;
            }
            // Remaining channels [nDims, headDim) are left untouched.
        };

        for (uint32_t s = 0; s < qSeqLen; ++s) {
            uint32_t p = pos + s;
            for (uint32_t h = 0; h < qHeads; ++h) {
                rotateHead(q + (static_cast<size_t>(s) * qHeads + h) * headDim, p);
            }
        }
        for (uint32_t s = 0; s < kSeqLen; ++s) {
            uint32_t p = pos + s;
            for (uint32_t h = 0; h < kHeads; ++h) {
                rotateHead(k + (static_cast<size_t>(s) * kHeads + h) * headDim, p);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Full-attention layer for qwen35: fused Q+gate projection, per-head Q/K
    // norms, MRoPE, attention with sigmoid(gate), output projection.
    // Mirrors llama.cpp qwen35 graph::build_layer_attn.
    // -----------------------------------------------------------------------
    void Model::forwardQwen35FullAttention(uint32_t layer,
                                           const LayerWeights &w, float *hidden,
                                           float *attnNorm, float *q, float *k,
                                           float *v, float *attnOut, float *qGate,
                                           float *attnProj, uint32_t seqLen,
                                           uint32_t hiddenSize, uint32_t nHeads,
                                           uint32_t nKVHeads, uint32_t headDim) {
        detail::ScopedProfile spQ35Attn("qwen35_full_attn");
        // RMSNorm: attnNorm = rmsNorm(hidden, attn_norm)
        for (uint32_t s = 0; s < seqLen; ++s) {
            rmsNormSIMD(hidden + static_cast<size_t>(s) * hiddenSize,
                        attnNorm + static_cast<size_t>(s) * hiddenSize,
                        w.rmsNormAttn.data(), hiddenSize);
        }

        // Q+G fused projection: w.attnQ = [nHeads*2*headDim, hiddenSize].
        // Row layout per head h: [Q(h) (headDim) | gate(h) (headDim)].
        const uint32_t qgDims = nHeads * 2 * headDim;
        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *hRow = attnNorm + static_cast<size_t>(s) * hiddenSize;
            w.attnQ.matMulVec(hRow, qGate + static_cast<size_t>(s) * qgDims);
        }

        // K/V projections.
        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *hRow = attnNorm + static_cast<size_t>(s) * hiddenSize;
            w.attnK.matMulVec(hRow, k + static_cast<size_t>(s) * nKVHeads * headDim);
            w.attnV.matMulVec(hRow, v + static_cast<size_t>(s) * nKVHeads * headDim);
        }

        // Per-head Q norm (shared weight, [headDim]) + per-head K norm.
        // Q slice of each head is the FIRST headDim of the 2*headDim slice.
        for (uint32_t s = 0; s < seqLen; ++s) {
            float *qRow = q + static_cast<size_t>(s) * nHeads * headDim;
            float *kRow = k + static_cast<size_t>(s) * nKVHeads * headDim;
            const float *qgRow = qGate + static_cast<size_t>(s) * qgDims;
            const float *qNormW = w.attnQNorm.data();
            const float *kNormW = w.attnKNorm.data();
            for (uint32_t h = 0; h < nHeads; ++h) {
                q35HeadRmsNorm(qgRow + static_cast<size_t>(h) * 2 * headDim,
                               qRow + static_cast<size_t>(h) * headDim,
                               qNormW, headDim);
            }
            for (uint32_t h = 0; h < nKVHeads; ++h) {
                q35HeadRmsNorm(kRow + static_cast<size_t>(h) * headDim,
                               kRow + static_cast<size_t>(h) * headDim,
                               kNormW, headDim);
            }
        }

        // MRoPE on Q and K (fused into the same buffers).
        uint32_t cachePos = static_cast<uint32_t>(kvCache_.pos);
        applyMRoPE(q, k, seqLen, seqLen, nHeads, nKVHeads, cachePos);

        // Store K, V into the KV cache (MRoPE already applied).
        const uint32_t maxSeqLen = config_.maxSeqLen;
        float *kCacheLayer =
                kvCache_.k.data() + static_cast<size_t>(layer) * maxSeqLen *
                                            nKVHeads * headDim;
        float *vCacheLayer =
                kvCache_.v.data() + static_cast<size_t>(layer) * maxSeqLen *
                                            nKVHeads * headDim;
        for (uint32_t s = 0; s < seqLen; ++s) {
            std::memcpy(kCacheLayer + static_cast<size_t>(cachePos + s) * nKVHeads * headDim,
                        k + static_cast<size_t>(s) * nKVHeads * headDim,
                        static_cast<size_t>(nKVHeads) * headDim * sizeof(float));
            std::memcpy(vCacheLayer + static_cast<size_t>(cachePos + s) * nKVHeads * headDim,
                        v + static_cast<size_t>(s) * nKVHeads * headDim,
                        static_cast<size_t>(nKVHeads) * headDim * sizeof(float));
        }

        // Attention. headDim = 256 for Qwen3.8-27B; attentionFused supports
        // headDim <= MAX_HEAD_DIM (128) only, so use the generic FMHA path.
        uint32_t totalCacheLen = cachePos + seqLen;
        if (headDim <= 128) {
            attentionFused(q, kCacheLayer, vCacheLayer, attnOut, seqLen,
                           cachePos, totalCacheLen, layer);
        } else {
            // Generic causal attention with online softmax (headDim up to 512).
            const uint32_t nGroups = nHeads / nKVHeads;
            const float invSqrt = 1.0f / std::sqrt(static_cast<float>(headDim));
            constexpr float LOG2E = 1.4426950408889634f;
            // Reusable per-query accumulation buffer (headDim floats) — heap was
            // allocated once per (query, head) pair before.
            ScratchPool &scratch = scratchPool();
            scratch.q35K.resize(headDim);
            float *o = scratch.q35K.data();
            for (uint32_t s = 0; s < seqLen; ++s) {
                uint32_t csEnd = cachePos + s;
                for (uint32_t g = 0; g < nKVHeads; ++g) {
                    for (uint32_t h = 0; h < nGroups; ++h) {
                        uint32_t qHead = g * nGroups + h;
                        const float *qPtr =
                                q + (static_cast<size_t>(s) * nHeads + qHead) * headDim;
                        float *oPtr = attnOut +
                                      (static_cast<size_t>(s) * nHeads + qHead) * headDim;
                        std::memset(o, 0, headDim * sizeof(float));
                        float m = -std::numeric_limits<float>::infinity();
                        float l = 0.0f;
                        for (uint32_t cs = 0; cs <= csEnd; ++cs) {
                            const float *kPtr =
                                    kCacheLayer +
                                    (static_cast<size_t>(cs) * nKVHeads + g) * headDim;
                            float score = dotProductFMA(qPtr, kPtr, headDim) * invSqrt;
                            if (score > m) {
                                float mNew = score;
                                float alpha = std::exp2f((m - mNew) * LOG2E);
                                for (uint32_t d = 0; d < headDim; ++d) {
                                    o[d] *= alpha;
                                }
                                l *= alpha;
                                m = mNew;
                            }
                            const float *vPtr =
                                    vCacheLayer +
                                    (static_cast<size_t>(cs) * nKVHeads + g) * headDim;
                            float p = std::exp2f((score - m) * LOG2E);
                            l += p;
                            for (uint32_t d = 0; d < headDim; ++d) {
                                o[d] += p * vPtr[d];
                            }
                        }
                        float invL = 1.0f / l;
                        for (uint32_t d = 0; d < headDim; ++d) {
                            oPtr[d] = o[d] * invL;
                        }
                    }
                }
            }
        }

        // Elementwise multiply by sigmoid(gate).
        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *gRow = qGate + static_cast<size_t>(s) * qgDims;
            float *oRow = attnOut + static_cast<size_t>(s) * nHeads * headDim;
            for (uint32_t h = 0; h < nHeads; ++h) {
                const float *gHead =
                        gRow + static_cast<size_t>(h) * 2 * headDim + headDim;
                float *oHead = oRow + static_cast<size_t>(h) * headDim;
                for (uint32_t d = 0; d < headDim; ++d) {
                    oHead[d] *= q35Sigmoid(gHead[d]);
                }
            }
        }

        // Output projection: attnProj = attn_output @ attnOut.
        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *oRow = attnOut + static_cast<size_t>(s) * nHeads * headDim;
            w.attnO.matMulVec(oRow, attnProj + static_cast<size_t>(s) * hiddenSize);
        }

        // Residual: hidden += attnProj.
        for (uint32_t s = 0; s < seqLen; ++s) {
            addSIMD(hidden + static_cast<size_t>(s) * hiddenSize,
                    attnProj + static_cast<size_t>(s) * hiddenSize, hiddenSize);
        }
    }

    // -----------------------------------------------------------------------
    // Recurrent (gated delta net) layer for qwen35.
    //
    // Per token:
    //   qkv = attn_qkv @ x              (keyDim*2 + valueDim)
    //   z   = attn_gate @ x             (dInner)
    //   beta = sigmoid(ssm_beta @ x)    (nVHeads)
    //   alpha = softplus(ssm_alpha @ x + ssm_dt.bias)   (nVHeads)
    //   gate = alpha * ssm_a            (ssm_a < 0, so decay = exp(gate) < 1)
    //   conv_in = concat(conv_state[3], qkv)  (conv kernel = 4)
    //   conv_out = silu(conv1d(conv_in))      (keyDim*2 + valueDim)
    //   q_conv = l2_norm(conv_out[:keyDim])   reshaped [headK, nKHeads]
    //   k_conv = l2_norm(conv_out[keyDim:2*keyDim])
    //   v_conv = conv_out[2*keyDim:]          reshaped [headV, nVHeads]
    //   repeat q,k nVHeads/nKHeads times (periodic: head hv reads head hv%16)
    //   per value-head h (state M_h [headV, headV], M[j*Sv+i] = S[i][j]):
    //     M *= exp(gate[h])
    //     delta[j] = (v[j] - dot(row_j(M), k)) * beta[h]
    //     M[j][:] += delta[j] * k          (outer product)
    //     out[j] = dot(row_j(M), q) * (1/sqrt(headV))
    //   out = rmsnorm(out, ssm_norm) * silu(z)
    //   out = ssm_out @ out
    // -----------------------------------------------------------------------
    void Model::forwardQwen35Recurrent(uint32_t layer, const LayerWeights &w,
                                       float *hidden, float *attnNorm,
                                       float *attnProj, uint32_t seqLen,
                                       uint32_t hiddenSize) {
        detail::ScopedProfile spQ35Rec("qwen35_recurrent");
        const uint32_t dInner = config_.ssmInnerSize;       // 6144
        const uint32_t headK = config_.ssmStateSize;        // 128
        const uint32_t nKHeads = config_.ssmGroupCount;     // 16
        const uint32_t nVHeads = config_.ssmTimeStepRank;   // 48
        const uint32_t headV = dInner / nVHeads;            // 128
        const uint32_t keyDim = headK * nKHeads;            // 2048
        const uint32_t valueDim = headV * nVHeads;          // 6144
        const uint32_t convChannels = 2 * keyDim + valueDim;// 10240
        const uint32_t convKernel = config_.ssmConvKernel;  // 4
        const uint32_t qkvDim = 2 * keyDim + valueDim;
        const uint32_t repeat = nVHeads / nKHeads;

        // RMSNorm: attnNorm = rmsNorm(hidden, attn_norm)
        for (uint32_t s = 0; s < seqLen; ++s) {
            rmsNormSIMD(hidden + static_cast<size_t>(s) * hiddenSize,
                        attnNorm + static_cast<size_t>(s) * hiddenSize,
                        w.rmsNormAttn.data(), hiddenSize);
        }

        // Per-layer recurrent state (only recurrent layers have it allocated).
        auto &convState = kvCache_.q35ConvState[layer];// [(convKernel-1)*channels]
        auto &gdnState = kvCache_.q35GdnState[layer];  // [nVHeads * headV * headV]

        const float epsNorm = 1e-6f;
        const float scale = 1.0f / std::sqrt(static_cast<float>(headV));

        // Reusable per-token buffers (allocated once per layer, NOT per token):
        // these were heap-allocated per token before, causing ~12 allocations ×
        // 48 recurrent layers = 576 malloc/free pairs per forward token.
        ScratchPool &scratch = scratchPool();
        std::vector<float> &qkv = scratch.q35QkvMixed;
        std::vector<float> &z = scratch.q35Z;
        std::vector<float> &beta = scratch.q35Beta;
        std::vector<float> &gateV = scratch.q35Gate;
        std::vector<float> &convInput = scratch.q35ConvInput;
        std::vector<float> &convOut = scratch.q35ConvOut;
        std::vector<float> &qN = scratch.q35Q;
        std::vector<float> &kN = scratch.q35K;
        std::vector<float> &vN = scratch.q35V;
        std::vector<float> &attnOut = scratch.q35AttnOut;
        std::vector<float> &delta = scratch.q35NormScratch;
        std::vector<float> &normOut = scratch.q35NormOut;
        qkv.resize(qkvDim);
        z.resize(dInner);
        beta.resize(nVHeads);
        gateV.resize(nVHeads);
        // Debug hook: retain softplus(alpha + dt.bias) per value head (used by
        // debugQwen35Layer0Intm to fingerprint llama's a_softplus-0 tensor).
        scratch.q35Alpha.resize(nVHeads);
        convInput.resize(static_cast<size_t>(convKernel) * convChannels);
        convOut.resize(convChannels);
        qN.resize(static_cast<size_t>(headK) * nVHeads);
        kN.resize(static_cast<size_t>(headK) * nVHeads);
        vN.resize(static_cast<size_t>(headV) * nVHeads);
        attnOut.resize(static_cast<size_t>(headV) * nVHeads);
        delta.resize(headV);
        normOut.resize(static_cast<size_t>(headV) * nVHeads);

        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *hRow = attnNorm + static_cast<size_t>(s) * hiddenSize;

            // ---- Projections ----
            w.attnQKV.matMulVec(hRow, qkv.data());
            w.attnGate.matMulVec(hRow, z.data());
            w.ssmBetaQ.matMulVec(hRow, beta.data());
            w.ssmAlphaQ.matMulVec(hRow, gateV.data());
            for (uint32_t v = 0; v < nVHeads; ++v) {
                beta[v] = q35Sigmoid(beta[v]);
                float alpha =
                        q35Softplus(gateV[v] + w.ssmDtBiasFull.data()[v]);
                scratch.q35Alpha[v] = alpha;// debug hook (a_softplus-0)
                gateV[v] = alpha * w.ssmABroadcast.data()[v];
            }

            // ---- Conv state + conv1d ----
            // conv state holds the (convKernel-1) most recent qkv inputs, oldest
            // first: [t-(K-1) ... t-1][channel]. conv_in = state ++ qkv gives
            // the sliding window [t-(K-1) .. t][channel].
            std::memcpy(convInput.data(), convState.data(),
                        (convKernel - 1) * static_cast<size_t>(convChannels) * sizeof(float));
            std::memcpy(convInput.data() + (convKernel - 1) * static_cast<size_t>(convChannels),
                        qkv.data(), qkvDim * sizeof(float));

            // conv1d: weight [convKernel, convChannels] row-major; input
            // [convKernel, convChannels] (kernel index stride = channels).
            const float *convW = reinterpret_cast<const float *>(w.ssmConv1d.data.data());
            for (uint32_t c = 0; c < convChannels; ++c) {
                const float *wRow =
                        convW + static_cast<size_t>(c) * convKernel;
                float sum = 0.0f;
                for (uint32_t i = 0; i < convKernel; ++i) {
                    sum += wRow[i] *
                           convInput[static_cast<size_t>(i) * convChannels + c];
                }
                convOut[c] = sum;
            }
            siluSIMD(convOut.data(), convChannels);

            // Shift conv state: drop the oldest window, append the new qkv.
            std::memmove(convState.data(),
                         convState.data() + convChannels,
                         (convKernel - 2) * static_cast<size_t>(convChannels) * sizeof(float));
            std::memcpy(convState.data() + (convKernel - 2) * static_cast<size_t>(convChannels),
                        qkv.data(), qkvDim * sizeof(float));

            // ---- Slice q/k/v ----
            const float *convQ = convOut.data();
            const float *convK = convOut.data() + keyDim;
            const float *convV = convOut.data() + 2 * keyDim;

            // L2-norm per k-head on q/k, then repeat over value heads
            // (periodic: value head hv reads k-head hv % nKHeads).
            for (uint32_t hv = 0; hv < nVHeads; ++hv) {
                uint32_t hk = hv % nKHeads;
                const float *qSrc = convQ + static_cast<size_t>(hk) * headK;
                const float *kSrc = convK + static_cast<size_t>(hk) * headK;
                double sq = 0.0, sk = 0.0;
                for (uint32_t d = 0; d < headK; ++d) {
                    sq += static_cast<double>(qSrc[d]) * qSrc[d];
                    sk += static_cast<double>(kSrc[d]) * kSrc[d];
                }
                float iq = static_cast<float>(1.0 / std::sqrt(std::max(sq, static_cast<double>(epsNorm))));
                float ik = static_cast<float>(1.0 / std::sqrt(std::max(sk, static_cast<double>(epsNorm))));
                float *qDst = qN.data() + static_cast<size_t>(hv) * headK;
                float *kDst = kN.data() + static_cast<size_t>(hv) * headK;
                for (uint32_t d = 0; d < headK; ++d) {
                    qDst[d] = qSrc[d] * iq;
                    kDst[d] = kSrc[d] * ik;
                }
            }
            // v_conv: [headV, nVHeads] copied directly.
            (void) repeat;
            for (uint32_t hv = 0; hv < nVHeads; ++hv) {
                std::memcpy(vN.data() + static_cast<size_t>(hv) * headV,
                            convV + static_cast<size_t>(hv) * headV,
                            headV * sizeof(float));
            }

            // ---- Gated delta net recurrence ----
            // M layout per head: M[j*Sv + i] = S[i][j] (transposed store, so
            // row j of M = column j of S, matching ggml's gated_delta_net).
            for (uint32_t hv = 0; hv < nVHeads; ++hv) {
                float *state = gdnState.data() + static_cast<size_t>(hv) * headV * headV;
                const float *qHead = qN.data() + static_cast<size_t>(hv) * headK;
                const float *kHead = kN.data() + static_cast<size_t>(hv) * headK;
                const float *vHead = vN.data() + static_cast<size_t>(hv) * headV;

                // Decay: S *= exp(gateV[hv]).
                const float decay = std::exp(gateV[hv]);
                for (uint32_t i = 0; i < headV * headV; ++i) {
                    state[i] *= decay;
                }

                // delta[j] = (v[j] - dot(row_j(M), k)) * beta[hv]
                for (uint32_t j = 0; j < headV; ++j) {
                    double sum = 0.0;
                    const float *rowJ = state + static_cast<size_t>(j) * headV;
                    for (uint32_t i = 0; i < headV; ++i) {
                        sum += static_cast<double>(rowJ[i]) * kHead[i];
                    }
                    delta[j] = (vHead[j] - static_cast<float>(sum)) * beta[hv];
                }

                // M[j][i] += delta[j] * k[i]  (outer product row j).
                for (uint32_t j = 0; j < headV; ++j) {
                    float dj = delta[j];
                    float *rowJ = state + static_cast<size_t>(j) * headV;
                    for (uint32_t i = 0; i < headV; ++i) {
                        rowJ[i] += dj * kHead[i];
                    }
                }

                // out[j] = dot(row_j(M), q) * scale
                float *outHead = attnOut.data() + static_cast<size_t>(hv) * headV;
                for (uint32_t j = 0; j < headV; ++j) {
                    double sum = 0.0;
                    const float *rowJ = state + static_cast<size_t>(j) * headV;
                    for (uint32_t i = 0; i < headV; ++i) {
                        sum += static_cast<double>(rowJ[i]) * qHead[i];
                    }
                    outHead[j] = static_cast<float>(sum) * scale;
                }
            }

            // ---- Gated RMSNorm: out = rmsnorm(out, ssm_norm) * silu(z) ----
            // ssm_norm is [headV] applied per value-head over the headV dim.
            // z is [dInner] = [headV, nVHeads] (same index order as out).
            for (uint32_t hv = 0; hv < nVHeads; ++hv) {
                rmsNormSIMD(attnOut.data() + static_cast<size_t>(hv) * headV,
                            normOut.data() + static_cast<size_t>(hv) * headV,
                            w.ssmNorm.data(), headV);
            }
            for (uint32_t i = 0; i < dInner; ++i) {
                // silu(z) = z * sigmoid(z)
                normOut[i] *= z[i] * q35Sigmoid(z[i]);
            }

            // ---- Output projection: ssm_out [valueDim -> hiddenSize] ----
            w.ssmOut.matMulVec(normOut.data(), attnProj + static_cast<size_t>(s) * hiddenSize);
        }

        // Residual: hidden += attnProj.
        for (uint32_t s = 0; s < seqLen; ++s) {
            addSIMD(hidden + static_cast<size_t>(s) * hiddenSize,
                    attnProj + static_cast<size_t>(s) * hiddenSize, hiddenSize);
        }
    }

    // -----------------------------------------------------------------------
    // One full qwen35 layer (both recurrent and full-attention variants),
    // including the post-attention-norm + SwiGLU FFN + residual structure.
    // Mirrors llama.cpp qwen35 graph::graph layer loop.
    // -----------------------------------------------------------------------
    void Model::forwardQwen35Layer(uint32_t layer, float *hidden, float *attnNorm,
                                   float *attnProj, float *q, float *k, float *v,
                                   float *attnOut, float *gate, float *up,
                                   float *ffnNorm, float *ffnOut, uint32_t seqLen,
                                   uint32_t hiddenSize, uint32_t nHeads,
                                   uint32_t nKVHeads, uint32_t headDim,
                                   uint32_t intermediateSize) {
        // The MTP (NextN) block (layer >= n_layer) is NOT executed in the main
        // decode pass — llama.cpp runs it only via the separate
        // LLM_GRAPH_TYPE_DECODER_MTP draft graph (llama.cpp qwen35.cpp:158).
        // Guard against any caller iterating over the full numLayers range.
        const uint32_t nLayer = config_.numLayers - config_.nextnPredictLayers;
        if (layer >= nLayer) {
            return;
        }

        const auto &w = layers_[layer];

        if (detail::isQwen35RecurrentLayer(config_, layer)) {
            // ---- Recurrent (gated delta net) ----
            forwardQwen35Recurrent(layer, w, hidden, attnNorm, attnProj, seqLen,
                                   hiddenSize);
        } else {
            // ---- Full attention (incl. MTP block, which is not executed in
            //      the main decode pass — the caller loops only over the non-MTP
            //      layers, so layer 64 is never dispatched here) ----
            // qGate is [seqLen, nHeads*2*headDim] (fused Q+gate output). It is
            // reused across full-attention layers via the scratch pool instead of
            // being heap-allocated per layer (16 layers × 12.5 MB allocs per token
            // before this change).
            ScratchPool &scratch = scratchPool();
            scratch.q35FinalOut.resize(static_cast<size_t>(seqLen) * nHeads * 2 * headDim);
            forwardQwen35FullAttention(layer, w, hidden, attnNorm, q, k, v,
                                       attnOut, scratch.q35FinalOut.data(), attnProj,
                                       seqLen, hiddenSize, nHeads, nKVHeads, headDim);
        }

        // ---- Post-attention norm + FFN + residual ----
        // The FFN residual connects to the tensor BEFORE the post-attention norm
        // (llama.cpp: ffn_residual = cur after the attention residual).
        for (uint32_t s = 0; s < seqLen; ++s) {
            rmsNormSIMD(hidden + static_cast<size_t>(s) * hiddenSize,
                        ffnNorm + static_cast<size_t>(s) * hiddenSize,
                        w.postAttnNorm.data(), hiddenSize);
        }

        if (config_.architecture == ARCH_QWEN35MOE && config_.expertCount > 0) {
            // ---- Routed MoE FFN (qwen35moe) ----
            // llama.cpp qwen35moe build_layer_ffn: softmax-router top-8 experts
            // (renormalized weights) + sigmoid-gated shared expert.
            computeQwen35MoE(ffnNorm, ffnOut, seqLen, hiddenSize,
                             intermediateSize, w);
        } else {
            // ---- Dense SwiGLU FFN (qwen35) ----
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *x = ffnNorm + static_cast<size_t>(s) * hiddenSize;
                float *g = gate + static_cast<size_t>(s) * intermediateSize;
                float *u = up + static_cast<size_t>(s) * intermediateSize;
                w.ffnGate.matMulVecFusedGateUp(w.ffnUp, x, g, u);
                for (uint32_t i = 0; i < intermediateSize; ++i) {
                    // silu(gate) * up
                    g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
                }
                w.ffnDown.matMulVec(g, ffnOut + static_cast<size_t>(s) * hiddenSize);
            }
        }
        // Residual: hidden += ffnOut (residual is the pre-post-norm tensor, which
        // is simply `hidden`'s current value after the attention residual).
        for (uint32_t s = 0; s < seqLen; ++s) {
            addSIMD(hidden + static_cast<size_t>(s) * hiddenSize,
                    ffnOut + static_cast<size_t>(s) * hiddenSize, hiddenSize);
        }
    }

    // -----------------------------------------------------------------------
    // Debug: run qwen35 token-by-token through the EXACT shared math used by
    // forward() (forwardQwen35Layer), snapshotting the hidden state after every
    // layer for the LAST token. Used by ReferenceCompareTest.Qwen35LayerwiseDivergence
    // to localize where the SIMD batch kernels and the scalar baseline first
    // drift apart, and to detect hidden-state norm pathologies (explosion /
    // collapse) that indicate a shared-math bug rather than kernel noise.
    //
    // NOTE: forwardTokenByToken() in ModelForwardDebug.cpp does NOT dispatch
    // ARCH_QWEN35 (it only handles the old qwen35moe SSM path + generic
    // attention), so it cannot be used to probe the real qwen35 forward. This
    // method is the qwen35-aware equivalent.
    // -----------------------------------------------------------------------
    std::vector<std::vector<float>>
    Model::debugQwen35PerLayer(const std::vector<int32_t> &tokens) {
        std::vector<std::vector<float>> perLayer;
        if (tokens.empty()) {
            return perLayer;
        }
        const uint32_t hiddenSize = config_.hiddenSize;
        const uint32_t nHeads = config_.numAttentionHeads;
        const uint32_t nKVHeads = config_.numKVHeads;
        const uint32_t headDim = config_.headDim;
        const uint32_t nLayers = config_.numLayers;
        const uint32_t intermediateSize = config_.intermediateSize;
        const uint32_t nLayer = nLayers - config_.nextnPredictLayers;
        const uint32_t seqLen = static_cast<uint32_t>(tokens.size());

        clearKVCache();

        ScratchPool &scratch = scratchPool();
        scratch.hidden.resize(static_cast<size_t>(seqLen) * hiddenSize);
        scratch.attnNorm.resize(static_cast<size_t>(seqLen) * hiddenSize);
        scratch.attnProj.resize(static_cast<size_t>(seqLen) * hiddenSize);
        scratch.q.resize(static_cast<size_t>(seqLen) * nHeads * headDim);
        scratch.k.resize(static_cast<size_t>(seqLen) * nKVHeads * headDim);
        scratch.v.resize(static_cast<size_t>(seqLen) * nKVHeads * headDim);
        scratch.attnOut.resize(static_cast<size_t>(seqLen) * nHeads * headDim);
        scratch.gate.resize(static_cast<size_t>(seqLen) * intermediateSize);
        scratch.up.resize(static_cast<size_t>(seqLen) * intermediateSize);
        scratch.ffnNorm.resize(static_cast<size_t>(seqLen) * hiddenSize);
        scratch.ffnOut.resize(static_cast<size_t>(seqLen) * hiddenSize);

        float *hiddenData = scratch.hidden.data();
        float *attnNormData = scratch.attnNorm.data();
        float *attnProjData = scratch.attnProj.data();
        float *qData = scratch.q.data();
        float *kData = scratch.k.data();
        float *vData = scratch.v.data();
        float *attnOutData = scratch.attnOut.data();
        float *gateData = scratch.gate.data();
        float *upData = scratch.up.data();
        float *ffnNormData = scratch.ffnNorm.data();
        float *ffnOutData = scratch.ffnOut.data();

        perLayer.reserve(nLayer);
        for (uint32_t li = 0; li < nLayer; ++li) {
            perLayer.emplace_back(hiddenSize, 0.0f);
        }

        for (uint32_t pos = 0; pos < seqLen; ++pos) {
            const int32_t tokenId = tokens[pos];
            // Mirror llama.cpp's AR decode: token i is decoded at position i.
            // The real forward() advances kvCache_.pos += seqLen AFTER the layer
            // loop; this debug method decodes one token at a time, so it must
            // set the position explicitly BEFORE each token, otherwise every
            // token would use position 0 and KV slot 0 (full-attention layers
            // would attend only to themselves, corrupting every layer >= 3 —
            // the first full-attention layer).
            kvCache_.pos = pos;
            // Cut to a single token: re-use the first slot of each scratch buffer
            // (seqLen == 1 semantics — exactly what forward() does for decode).
            uint32_t curLen = 1;
            if (tokenId >= 0 &&
                tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
                auto embRow = quantizedEmbeddings_.getRow(tokenId);
                std::memcpy(hiddenData, embRow.data(), hiddenSize * sizeof(float));
            } else {
                std::fill(hiddenData, hiddenData + hiddenSize, 0.0f);
            }

            for (uint32_t layer = 0; layer < nLayer; ++layer) {
                forwardQwen35Layer(layer, hiddenData, attnNormData, attnProjData,
                                   qData, kData, vData, attnOutData, gateData,
                                   upData, ffnNormData, ffnOutData, curLen,
                                   hiddenSize, nHeads, nKVHeads, headDim,
                                   intermediateSize);
                // Snapshot the last (only) token's post-FFN-residual hidden.
                if (pos == seqLen - 1) {
                    std::memcpy(perLayer[layer].data(), hiddenData,
                                hiddenSize * sizeof(float));
                }
            }
        }

        return perLayer;
    }

    // -----------------------------------------------------------------------
    // Debug bisection: fingerprint every layer-0 recurrent intermediate for
    // the LAST token of a sequence, mirroring llama_ref_probe.cpp's
    // "[ar] layer-0 recurrent intermediates" captures (the probe's intm map
    // is overwritten on every eval, so it holds the last token's values).
    // Runs the REAL shared forward math (forwardQwen35Layer) token-by-token
    // for layer 0 so the conv/GDN recurrent state accumulates exactly like
    // llama's AR decode; then fingerprints the retained scratch buffers.
    // -----------------------------------------------------------------------
    std::vector<std::pair<std::string, std::vector<float>>>
    Model::debugQwen35Layer0Intm(const std::vector<int32_t> &tokens) {
        std::vector<std::pair<std::string, std::vector<float>>> out;
        const uint32_t hiddenSize = config_.hiddenSize;
        const uint32_t nHeads = config_.numAttentionHeads;
        const uint32_t nKVHeads = config_.numKVHeads;
        const uint32_t headDim = config_.headDim;
        const uint32_t nLayers = config_.numLayers;
        const uint32_t intermediateSize = config_.intermediateSize;
        const uint32_t nLayer = nLayers - config_.nextnPredictLayers;
        const uint32_t seqLen = 1;

        clearKVCache();

        ScratchPool &scratch = scratchPool();
        scratch.hidden.resize(static_cast<size_t>(seqLen) * hiddenSize);
        scratch.attnNorm.resize(static_cast<size_t>(seqLen) * hiddenSize);
        scratch.attnProj.resize(static_cast<size_t>(seqLen) * hiddenSize);
        scratch.q.resize(static_cast<size_t>(seqLen) * nHeads * headDim);
        scratch.k.resize(static_cast<size_t>(seqLen) * nKVHeads * headDim);
        scratch.v.resize(static_cast<size_t>(seqLen) * nKVHeads * headDim);
        scratch.attnOut.resize(static_cast<size_t>(seqLen) * nHeads * headDim);
        scratch.gate.resize(static_cast<size_t>(seqLen) * intermediateSize);
        scratch.up.resize(static_cast<size_t>(seqLen) * intermediateSize);
        scratch.ffnNorm.resize(static_cast<size_t>(seqLen) * hiddenSize);
        scratch.ffnOut.resize(static_cast<size_t>(seqLen) * hiddenSize);

        float *hiddenData = scratch.hidden.data();
        float *attnNormData = scratch.attnNorm.data();
        float *attnProjData = scratch.attnProj.data();
        float *qData = scratch.q.data();
        float *kData = scratch.k.data();
        float *vData = scratch.v.data();
        float *attnOutData = scratch.attnOut.data();
        float *gateData = scratch.gate.data();
        float *upData = scratch.up.data();
        float *ffnNormData = scratch.ffnNorm.data();
        float *ffnOutData = scratch.ffnOut.data();

        if (tokens.empty()) {
            return out;
        }

        // Run ONLY layer 0 for every token (recurrent conv/GDN state
        // accumulates across tokens exactly like llama's AR decode). The
        // scratch buffers retain the LAST token's values afterwards.
        for (size_t pos = 0; pos < tokens.size(); ++pos) {
            const int32_t tokenId = tokens[pos];
            if (tokenId >= 0 &&
                tokenId < static_cast<int32_t>(quantizedEmbeddings_.vocabSize)) {
                auto embRow = quantizedEmbeddings_.getRow(tokenId);
                std::memcpy(hiddenData, embRow.data(), hiddenSize * sizeof(float));
            } else {
                std::fill(hiddenData, hiddenData + hiddenSize, 0.0f);
            }
            forwardQwen35Layer(0, hiddenData, attnNormData, attnProjData, qData,
                               kData, vData, attnOutData, gateData, upData,
                               ffnNormData, ffnOutData, seqLen, hiddenSize,
                               nHeads, nKVHeads, headDim, intermediateSize);
        }

        const uint32_t dInner = config_.ssmInnerSize;       // 6144
        const uint32_t headK = config_.ssmStateSize;        // 128
        const uint32_t nKHeads = config_.ssmGroupCount;     // 16
        const uint32_t nVHeads = config_.ssmTimeStepRank;   // 48
        const uint32_t headV = dInner / nVHeads;            // 128
        const uint32_t keyDim = headK * nKHeads;            // 2048
        const uint32_t valueDim = headV * nVHeads;          // 6144
        const uint32_t convChannels = 2 * keyDim + valueDim;// 10240

        auto push = [&](const char *name, std::vector<float> v) {
            out.emplace_back(name, std::move(v));
        };

        // 1. attn_norm = rmsNorm(emb, attn_norm_weight)
        push("attn_norm-0", std::vector<float>(attnNormData, attnNormData + hiddenSize));

        // 2. linear_attn_qkv_mixed = attnQKV @ attn_norm
        push("linear_attn_qkv_mixed-0", scratch.q35QkvMixed);

        // 3. z (attn_gate projection)
        push("z-0", scratch.q35Z);

        // 4. beta_sigmoid = sigmoid(beta) after ssmBetaQ projection
        push("beta_sigmoid-0", scratch.q35Beta);

        // 5. a_softplus = softplus(alpha + ssm_dt.bias); gate = a_softplus *
        //    ssmA (decay). forwardQwen35Recurrent stores the softplus alpha in
        //    scratch.q35Alpha (debug hook) and the final gate in q35Gate.
        push("a_softplus-0", scratch.q35Alpha);
        push("gate-0", scratch.q35Gate);

        // 6. conv outputs: q35ConvInput (= conv output raw pre-silu is lost;
        //    only post-silu survives in q35ConvOut). The silu buffer that
        //    forwardQwen35Recurrent applies in-place overwrites the raw conv
        //    output, so we reconstruct conv_output_raw analytically is NOT
        //    possible from scratch alone -- instead re-run the conv on the
        //    retained inputs:
        {
            // Re-run the conv1d on the retained inputs to recover the PRE-silu
            // raw output (forwardQwen35Recurrent applies silu in-place).
            const auto &w = layers_[0];
            const uint32_t convKernel = config_.ssmConvKernel;// 4
            const float *convW = reinterpret_cast<const float *>(w.ssmConv1d.data.data());
            std::vector<float> convRaw(convChannels, 0.0f);
            // convInput layout identical to forward path: [3 old states][qkv].
            for (uint32_t c = 0; c < convChannels; ++c) {
                const float *wRow = convW + static_cast<size_t>(c) * convKernel;
                float sum = 0.0f;
                for (uint32_t i = 0; i < convKernel; ++i) {
                    sum += wRow[i] * scratch.q35ConvInput[static_cast<size_t>(i) * convChannels + c];
                }
                convRaw[c] = sum;
            }
            push("conv_output_raw-0", std::move(convRaw));
        }
        push("conv_output_silu-0", scratch.q35ConvOut);

        // 7. q/k/v_conv_predelta: L2-normalized per repeated k-head slices.
        // scratch.q35Q / q35K hold [nVHeads * headK] repeated views (qN/kN).
        push("q_conv_predelta-0", scratch.q35Q);
        push("k_conv_predelta-0", scratch.q35K);
        push("v_conv_predelta-0", scratch.q35V);

        // 8. attn_output = GDN (post-recurrence, pre gated-norm) output
        push("attn_output-0", scratch.q35AttnOut);

        // 9. final_output = gated rmsnorm(ssm_norm, silu(z))
        push("final_output-0", scratch.q35NormOut);

        // 10. linear_attn_out = ssm_out @ final_output
        push("linear_attn_out-0", std::vector<float>(attnProjData, attnProjData + hiddenSize));

        // 11. attn_residual = hidden(emb_last) + linear_attn_out (attnProjData
        //     retains the last token's output projection).
        {
            std::vector<float> attnResidual(hiddenSize);
            auto embRow = quantizedEmbeddings_.getRow(tokens.back());
            for (uint32_t i = 0; i < hiddenSize; ++i) {
                attnResidual[i] = embRow[i] + attnProjData[i];
            }
            push("attn_residual-0", std::move(attnResidual));
        }

        // 12. attn_post_norm = rmsNorm(attn_residual, postAttnNorm)
        std::vector<float> postNorm(hiddenSize);
        // ffnNormData was overwritten by the layer's post-attn RMSNorm of the
        // pre-FFN hidden; the residual hidden (emb+attnProj) == pre-post-norm.
        // ffnNormData holds EXACTLY that: post-attn RMSNorm(attn_residual).
        push("attn_post_norm-0", std::vector<float>(ffnNormData, ffnNormData + hiddenSize));

        // 13. ffn_out = SwiGLU FFN output (residual NOT included)
        push("ffn_out-0", std::vector<float>(ffnOutData, ffnOutData + hiddenSize));

        // 14. post_ffn = attn_residual + ffn_out == final hiddenData
        push("post_ffn-0", std::vector<float>(hiddenData, hiddenData + hiddenSize));

        return out;
    }

}// namespace tinycoder
