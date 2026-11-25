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

// Inkling-2-Mini: looped mixture-of-experts decoder.
//
// The middle [loopStart, loopEnd) blocks run `loopCount` times with shared
// weights.  Every run owns its own K/V and short-convolution state, the passes
// exchange gated messages, attention carries a learned relative-position bias
// and a sink logit, and several projections are wrapped in a causal depthwise
// short convolution.  This is a self-contained token-at-a-time implementation:
// correctness against llama.cpp first, batching/GPU afterwards.
//

#include "Model.hpp"
#ifdef USE_CUDA
#include "gpu/InklingGPU.hpp"
#endif
#include "SIMDMatMulVec.hpp"
#include "ThreadPool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace tinycoder {
    namespace {

        constexpr float kRmsEps = 1e-6f;

        inline float sigmoidf_(float x) {
            return 1.0f / (1.0f + std::exp(-x));
        }

        void rmsNormVec(const float *x, const float *weight, float *out,
                        uint32_t n, float eps = kRmsEps) {
            double ss = 0.0;
            for (uint32_t i = 0; i < n; ++i) ss += double(x[i]) * x[i];
            const float scale = 1.0f / std::sqrt(float(ss / n) + eps);
            if (weight != nullptr) {
                for (uint32_t i = 0; i < n; ++i) out[i] = x[i] * scale * weight[i];
            } else {
                for (uint32_t i = 0; i < n; ++i) out[i] = x[i] * scale;
            }
        }

        void cpuMatVec(const QuantizedMatrix &m, const float *x, float *out) {
            if (m.empty() || m.data.empty()) {
                std::fill(out, out + (m.rows ? m.rows : 1), 0.0f);
                return;
            }
            m.matMulVec(x, out);
        }

        // Per-head L2 clip after an optional per-row-item scale.
        void clipRows(float *x, uint32_t nRows, uint32_t rowLen,
                      const float *scale, float threshold, float eps = 1e-6f) {
            for (uint32_t r = 0; r < nRows; ++r) {
                float *row = x + static_cast<size_t>(r) * rowLen;
                double ss = 0.0;
                if (scale != nullptr) {
                    const float *s = scale + static_cast<size_t>(r) * rowLen;
                    for (uint32_t i = 0; i < rowLen; ++i) {
                        row[i] *= s[i];
                        ss += double(row[i]) * row[i];
                    }
                } else {
                    for (uint32_t i = 0; i < rowLen; ++i) ss += double(row[i]) * row[i];
                }
                const float norm = std::sqrt(float(ss) + eps);
                const float factor = std::min(1.0f, threshold / norm);
                if (factor < 1.0f) {
                    for (uint32_t i = 0; i < rowLen; ++i) row[i] *= factor;
                }
            }
        }

        // Causal depthwise short convolution.  `history` is (nConv-1) x
        // channels, oldest first; on return it holds the last nConv-1 inputs.
        // `dynamic` optionally holds nConv input-dependent taps per head.
        void shortConv(const float *cur, uint32_t channels, uint32_t nConv,
                       const float *weight, const float *dynamic, uint32_t nHeads,
                       float *history, float *out) {
            const uint32_t hist = nConv - 1;
            for (uint32_t c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < nConv; ++k) {
                    const float v = (k < hist)
                                            ? history[static_cast<size_t>(k) * channels + c]
                                            : cur[c];
                    acc += weight[static_cast<size_t>(c) * nConv + k] * v;
                }
                out[c] = acc;
            }
            if (dynamic != nullptr) {
                const uint32_t perHead = channels / nHeads;
                for (uint32_t h = 0; h < nHeads; ++h) {
                    for (uint32_t lag = 0; lag < nConv; ++lag) {
                        const float tap = dynamic[static_cast<size_t>(h) * nConv + lag];
                        if (tap == 0.0f) continue;
                        const uint32_t src = nConv - 1 - lag;// 0..hist, hist = cur
                        const float *vv = src < hist
                                                  ? history + static_cast<size_t>(src) * channels
                                                  : cur;
                        float *dst = out + static_cast<size_t>(h) * perHead;
                        const float *sv = vv + static_cast<size_t>(h) * perHead;
                        for (uint32_t j = 0; j < perHead; ++j) dst[j] += tap * sv[j];
                    }
                }
            }
            for (uint32_t k = 0; k + 1 < hist; ++k) {
                std::memcpy(history + static_cast<size_t>(k) * channels,
                            history + static_cast<size_t>(k + 1) * channels,
                            channels * sizeof(float));
            }
            if (hist > 0) {
                std::memcpy(history + static_cast<size_t>(hist - 1) * channels, cur,
                            channels * sizeof(float));
            }
        }

        // Weighted RMS-normalized mix of `states` with `query`; writes nQueries
        // x hidden vectors.
        void mixStates(const std::vector<const float *> &states, const float *query,
                       uint32_t nQueries, uint32_t hidden, float *out,
                       std::vector<float> &scratchNormed,
                       std::vector<float> &scratchScores) {
            const uint32_t n = static_cast<uint32_t>(states.size());
            scratchNormed.resize(static_cast<size_t>(n) * hidden);
            for (uint32_t s = 0; s < n; ++s) {
                rmsNormVec(states[s], nullptr,
                           scratchNormed.data() + static_cast<size_t>(s) * hidden,
                           hidden);
            }
            scratchScores.resize(n);
            const float scale = 1.0f / std::sqrt(float(hidden));
            for (uint32_t q = 0; q < nQueries; ++q) {
                const float *qv = query + static_cast<size_t>(q) * hidden;
                float mx = -std::numeric_limits<float>::infinity();
                for (uint32_t s = 0; s < n; ++s) {
                    double dot = 0.0;
                    const float *sv = scratchNormed.data() +
                                      static_cast<size_t>(s) * hidden;
                    for (uint32_t i = 0; i < hidden; ++i) dot += double(sv[i]) * qv[i];
                    scratchScores[s] = float(dot) * scale;
                    mx = std::max(mx, scratchScores[s]);
                }
                double sum = 0.0;
                for (uint32_t s = 0; s < n; ++s) {
                    scratchScores[s] = std::exp(scratchScores[s] - mx);
                    sum += scratchScores[s];
                }
                float *dst = out + static_cast<size_t>(q) * hidden;
                std::fill(dst, dst + hidden, 0.0f);
                for (uint32_t s = 0; s < n; ++s) {
                    const float wgt = float(scratchScores[s] / sum);
                    const float *sv = scratchNormed.data() +
                                      static_cast<size_t>(s) * hidden;
                    for (uint32_t i = 0; i < hidden; ++i) dst[i] += wgt * sv[i];
                }
            }
        }

        // Gated cross-pass message: (1-forget)*cur + gain*message.
        void applyControl(const float *cur, const float *message,
                          const QuantizedMatrix &down, const QuantizedMatrix &gain,
                          const QuantizedMatrix &forget, uint32_t hidden,
                          float *out, std::vector<float> &scratch) {
            scratch.resize(static_cast<size_t>(2) * hidden);
            rmsNormVec(cur, nullptr, scratch.data(), hidden);
            rmsNormVec(message, nullptr, scratch.data() + hidden, hidden);
            std::vector<float> g(static_cast<size_t>(down.rows));
            cpuMatVec(down, scratch.data(), g.data());
            std::vector<float> gv(static_cast<size_t>(gain.rows));
            std::vector<float> fv(static_cast<size_t>(forget.rows));
            cpuMatVec(gain, g.data(), gv.data());
            cpuMatVec(forget, g.data(), fv.data());
            for (uint32_t i = 0; i < hidden; ++i) {
                const float gn = std::tanh(gv[i]);
                const float fg = std::tanh(fv[i]);
                out[i] = cur[i] - fg * cur[i] + gn * message[i];
            }
        }

        // SITU(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) *
        //                  lb*tanh(up/lb)
        inline float situ(float gate, float up, float beta, float lb) {
            const float g = beta * std::tanh(gate / beta) * sigmoidf_(gate);
            const float u = lb > 0.0f ? lb * std::tanh(up / lb) : up;
            return g * u;
        }

    }// namespace

    tinycoder::Array<float> Model::forwardInkling(const std::vector<int32_t> &tokens,
                                                  bool computeAllLogits) {
        const uint32_t hidden = config_.hiddenSize;
        const uint32_t nHeads = config_.numAttentionHeads;
        const uint32_t nKvHeads = config_.numKVHeads;
        const uint32_t headDim = config_.headDim;
        const uint32_t kvDim = nKvHeads * headDim;
        const uint32_t nConv = config_.shortconvLCache;
        const uint32_t nRel = config_.relativeLength;
        const uint32_t extent = config_.relativeExtent;
        const uint32_t nExperts = config_.expertCount;
        const uint32_t nUsed = config_.expertUsedCount;
        const uint32_t expertFf = config_.expertFeedForwardLength;
        const uint32_t phys = config_.numPhysicalLayers;
        const uint32_t slots = config_.numKvSlots;
        const uint32_t core = config_.loopEnd - config_.loopStart;
        const uint32_t blockSize = config_.attnResBlockSize;
        const uint32_t vocab = config_.vocabSizeUnpadded ? config_.vocabSizeUnpadded
                                                          : config_.vocabSize;
        const float qkScale = 1.0f / std::sqrt(float(headDim));
        const uint32_t group = nHeads / nKvHeads;
        const uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        const uint32_t histLen = nConv > 1 ? nConv - 1 : 0;
        const uint32_t sconvChannels = 2 * kvDim + 2 * hidden;

        if (phys == 0 || slots == 0 || core == 0 || nConv < 2 || nRel == 0 ||
            extent == 0 || nExperts == 0 || nUsed == 0) {
            std::cerr << "[TinyCoder] Inkling forward: incomplete geometry"
                      << std::endl;
            return tinycoder::Array<float>{};
        }
        if (inklingConvState_.size() != slots) {
            inklingConvState_.assign(
                    slots, std::vector<float>(
                                    static_cast<size_t>(histLen) * sconvChannels,
                                    0.0f));
        }

        // Per-call scratch (reused across tokens).
        std::vector<float> x(hidden), normed(hidden), inpSA(hidden);
        std::vector<float> nq(hidden), nk(hidden), nv(hidden);
        std::vector<float> pre0(hidden), pre1(hidden), pre2(hidden);
        std::vector<float> q(static_cast<size_t>(nHeads) * headDim);
        std::vector<float> k(static_cast<size_t>(kvDim));
        std::vector<float> v(static_cast<size_t>(kvDim));
        std::vector<float> kProj(static_cast<size_t>(kvDim));
        std::vector<float> vProj(static_cast<size_t>(kvDim));
        std::vector<float> convK(static_cast<size_t>(kvDim));
        std::vector<float> convV(static_cast<size_t>(kvDim));
        std::vector<float> convH(hidden);
        std::vector<float> rel(static_cast<size_t>(nRel) * nHeads);
        std::vector<float> sink(nHeads);
        std::vector<float> table(static_cast<size_t>(extent + 1) * nHeads);
        std::vector<float> attnOut(static_cast<size_t>(nHeads) * headDim);
        std::vector<float> o(hidden);
        std::vector<float> gateProj(static_cast<size_t>(nHeads) * headDim);
        std::vector<float> gateUp(static_cast<size_t>(2) * expertFf);
        std::vector<float> expertAct(static_cast<size_t>(expertFf));
        std::vector<float> expertOut(hidden);
        std::vector<float> moeOut(hidden);
        std::vector<float> ffnInp(hidden);
        std::vector<float> mixOut(static_cast<size_t>(3) * hidden);
        std::vector<float> router(static_cast<size_t>(nExperts));
        std::vector<float> probs(static_cast<size_t>(nExperts));
        std::vector<float> selection(static_cast<size_t>(nExperts));
        std::vector<std::pair<float, uint32_t>> scored(static_cast<size_t>(nExperts));
        std::vector<float> logitsRow(static_cast<size_t>(
                std::max(config_.vocabSize, lmHead_.empty() ? 0u : lmHead_.rows)));
        std::vector<float> mixScratch, scoreScratch, controlScratch;
        std::vector<float> moeExpertOuts;
        std::vector<float> msgMixed(static_cast<size_t>(4) * hidden);
        std::vector<float> legacyQuery(static_cast<size_t>(4) * hidden);
        std::vector<float> bankNormed;
        std::vector<float> bankScores;

        std::vector<std::vector<float>> blocks;
        blocks.reserve(8);
        std::vector<const float *> states;
        states.reserve(8);

        tinycoder::Array<float> logits(
                tinycoder::Shape{computeAllLogits ? size_t(seqLen) : size_t(1),
                                 size_t(vocab)});

        double profQ = 0, profC = 0, profT = 0, profA = 0, profO = 0, profF = 0, profM = 0;
        auto nowT = [] { return std::chrono::steady_clock::now(); };
#ifdef USE_CUDA
        const char *inklingGpuEnv = std::getenv("TINYCODER_INKLING_GPU");
        if (inklingGpuEnv != nullptr && inklingGpuEnv[0] != '\0' &&
            inklingGpuEnv[0] != '0' && !inklingGpu_) {
            inklingGpu_ = std::make_shared<gpu::InklingGPU>();
            std::fprintf(stderr, "[inkling] hybrid GPU matvecs: %s\n",
                         inklingGpu_->available() ? "enabled" : "unavailable");
        }
        auto *gpuMv = inklingGpu_.get();
#else
        constexpr void *gpuMv = nullptr;
#endif
        auto mv = [&](const QuantizedMatrix &m, const float *x, float *out) {
#ifdef USE_CUDA
            if (gpuMv != nullptr && gpuMv->available() && !m.data.empty() &&
                gpuMv->gemv(m.data.data(), m.type, m.rows, m.cols, x, out)) {
                return;
            }
#endif
            cpuMatVec(m, x, out);
        };
        [[maybe_unused]] auto mvRows = [&](const QuantizedMatrix &m,
                                           const float *x, uint32_t rowStart,
                                           uint32_t numRows, float *out) {
#ifdef USE_CUDA
            if (gpuMv != nullptr && gpuMv->available() && !m.data.empty() &&
                gpuMv->gemvRows(m.data.data(), m.type, m.rows, m.cols, rowStart,
                                numRows, x, out)) {
                return;
            }
#endif
            m.matMulVecRows(x, rowStart, numRows, out);
        };

        const uint32_t start = config_.loopStart;
        const uint32_t end = config_.loopEnd;
        const float beta = config_.situBeta;
        const float lb = config_.situLinearBeta;

        for (uint32_t s = 0; s < seqLen; ++s) {
            const uint32_t pos = static_cast<uint32_t>(kvCache_.pos) + s;
            const int32_t token = tokens[s];

            {
                std::vector<float> emb =
                        quantizedEmbeddings_.getRow(static_cast<uint32_t>(token));
                if (emb.size() != hidden) {
                    std::cerr << "[TinyCoder] Inkling: bad embedding row"
                              << std::endl;
                    return tinycoder::Array<float>{};
                }
                rmsNormVec(emb.data(), inklingTokNorm_.data(), x.data(), hidden);
                if (const char *dd = std::getenv("TINYCODER_INKLING_DUMP")) {
                    char path[1024];
                    std::snprintf(path, sizeof(path), "%s/emb-%u.f32", dd,
                                  static_cast<unsigned>(token));
                    FILE *f = std::fopen(path, "wb");
                    if (f) { std::fwrite(x.data(), sizeof(float), hidden, f); std::fclose(f); }
                }
            }

            blocks.clear();
            if (std::getenv("TINYCODER_INKLING_TRACE") != nullptr) {
                double ne = 0.0;
                for (uint32_t i = 0; i < hidden; ++i) ne += double(x[i]) * x[i];
                std::fprintf(stderr, "\n[emb %.4g] ", std::sqrt(ne));
            }

            auto mixWithCurrent = [&](const float *query, uint32_t nqOut,
                                      float *out) {
                states.clear();
                for (const auto &b: blocks) states.push_back(b.data());
                states.push_back(x.data());
                mixStates(states, query, nqOut, hidden, out, mixScratch,
                          scoreScratch);
            };

            auto dumpVec = [&](const char *tag, const float *p, uint32_t n,
                               uint32_t slot) {
                if (const char *dd = std::getenv("TINYCODER_INKLING_DUMP")) {
                    char path[1024];
                    std::snprintf(path, sizeof(path), "%s/%s-%u.f32", dd, tag, slot);
                    FILE *f = std::fopen(path, "wb");
                    if (f) { std::fwrite(p, sizeof(float), n, f); std::fclose(f); }
                }
            };

            auto runLayer = [&](uint32_t il, uint32_t slot,
                                const float *message) {
                if (std::getenv("TINYCODER_INKLING_TRACE") != nullptr)
                    std::fprintf(stderr, "L%u ", il);
                auto &w = layers_[il];
                auto &ink = w.inkling;
                const bool isSwa = il < config_.slidingWindowPattern.size()
                                           ? config_.slidingWindowPattern[il] != 0
                                           : true;
                float *hist = inklingConvState_[slot].data();

                std::memcpy(inpSA.data(), x.data(), hidden * sizeof(float));

                // Attention inputs, depth-mixed and optionally message-gated.
                if (!ink.attnResQuery.empty()) {
                    const uint32_t nq = isSwa ? 1 : 3;
                    mixWithCurrent(ink.attnResQuery.data(), nq, mixOut.data());
                    std::memcpy(pre0.data(), mixOut.data(), hidden * sizeof(float));
                    if (isSwa) {
                        std::memcpy(pre1.data(), pre0.data(), hidden * sizeof(float));
                        std::memcpy(pre2.data(), pre0.data(), hidden * sizeof(float));
                    } else {
                        std::memcpy(pre1.data(), mixOut.data() + hidden,
                                    hidden * sizeof(float));
                        std::memcpy(pre2.data(), mixOut.data() + 2 * hidden,
                                    hidden * sizeof(float));
                    }
                } else {
                    std::memcpy(pre0.data(), x.data(), hidden * sizeof(float));
                    std::memcpy(pre1.data(), x.data(), hidden * sizeof(float));
                    std::memcpy(pre2.data(), x.data(), hidden * sizeof(float));
                }
                if (message != nullptr) {
                    applyControl(pre0.data(), message + 0 * hidden,
                                 ink.ctrlDown[0], ink.ctrlGain[0],
                                 ink.ctrlForget[0], hidden, nq.data(),
                                 controlScratch);
                    std::memcpy(pre0.data(), nq.data(), hidden * sizeof(float));
                    applyControl(pre1.data(), message + 1 * hidden,
                                 ink.ctrlDown[1], ink.ctrlGain[1],
                                 ink.ctrlForget[1], hidden, nk.data(),
                                 controlScratch);
                    std::memcpy(pre1.data(), nk.data(), hidden * sizeof(float));
                    applyControl(pre2.data(), message + 2 * hidden,
                                 ink.ctrlDown[2], ink.ctrlGain[2],
                                 ink.ctrlForget[2], hidden, nv.data(),
                                 controlScratch);
                    std::memcpy(pre2.data(), nv.data(), hidden * sizeof(float));
                }

                if (isSwa) {
                    rmsNormVec(pre0.data(), w.rmsNormAttn.data(), nq.data(), hidden);
                    rmsNormVec(pre1.data(), w.rmsNormAttn.data(), nk.data(), hidden);
                    rmsNormVec(pre2.data(), w.rmsNormAttn.data(), nv.data(), hidden);
                } else {
                    rmsNormVec(pre0.data(), ink.attnQInpNorm.data(), nq.data(), hidden);
                    rmsNormVec(pre1.data(), ink.attnKInpNorm.data(), nk.data(), hidden);
                    rmsNormVec(pre2.data(), ink.attnVInpNorm.data(), nv.data(), hidden);
                }

                dumpVec("nq", nq.data(), hidden, slot);
                auto tp0 = nowT();
                mv(w.attnQ, nq.data(), q.data());
                mv(w.attnK, nk.data(), kProj.data());
                mv(w.attnV, nv.data(), vProj.data());
                mv(ink.attnRel, nq.data(), rel.data());

                // Short convolutions: K at offset 0, V at kvDim.
                std::vector<float> dyn;
                if (!isSwa && !ink.attnKConvProj.empty()) {
                    dyn.resize(static_cast<size_t>(nConv) * nKvHeads);
                    mv(ink.attnKConvProj, nk.data(), dyn.data());
                }
                shortConv(kProj.data(), kvDim, nConv,
                          reinterpret_cast<const float *>(ink.attnKConv.data.data()),
                          dyn.empty() ? nullptr : dyn.data(), nKvHeads, hist,
                          convK.data());
                dyn.clear();
                if (!isSwa && !ink.attnVConvProj.empty()) {
                    dyn.resize(static_cast<size_t>(nConv) * nKvHeads);
                    mv(ink.attnVConvProj, nv.data(), dyn.data());
                }
                shortConv(vProj.data(), kvDim, nConv,
                          reinterpret_cast<const float *>(ink.attnVConv.data.data()),
                          dyn.empty() ? nullptr : dyn.data(), nKvHeads,
                          hist + static_cast<size_t>(histLen) * kvDim,
                          convV.data());
                profQ += std::chrono::duration<double>(nowT() - tp0).count();
                auto tp1 = nowT();
                const float convScale = isSwa
                                                ? 1.0f / std::sqrt(float(nConv + 1))
                                                : 1.0f / std::sqrt(float(2 * nConv + 1));
                for (uint32_t i = 0; i < kvDim; ++i) {
                    k[i] = (kProj[i] + convK[i]) * convScale;
                    v[i] = (vProj[i] + convV[i]) * convScale;
                }

                clipRows(q.data(), nHeads, headDim, ink.attnQClip.data(),
                         config_.qkClip);
                clipRows(k.data(), nKvHeads, headDim, nullptr, config_.qkClip);
                for (uint32_t h = 0; h < nKvHeads; ++h) {
                    rmsNormVec(v.data() + static_cast<size_t>(h) * headDim, nullptr,
                               v.data() + static_cast<size_t>(h) * headDim, headDim);
                }
                clipRows(rel.data(), nHeads, nRel, ink.attnRelClip.data(),
                         config_.relativeClip);
                mv(ink.attnSinkProj, nq.data(), sink.data());
                profC += std::chrono::duration<double>(nowT() - tp1).count();
                dumpVec("Qcur", q.data(), nHeads * headDim, slot);
                dumpVec("Kcur", k.data(), kvDim, slot);
                dumpVec("Vcur", v.data(), kvDim, slot);
                dumpVec("Rcur", rel.data(), nRel * nHeads, slot);

                if (!isSwa) {
                    const float arg = std::max(
                            1.0f, float(pos + 1) /
                                          std::max(1.0f, config_.logScaleFloor));
                    const float tau = 1.0f + config_.logScaleAlpha * std::log(arg);
                    for (float &qv: q) qv *= tau;
                    for (float &rv: rel) rv *= tau;
                }

                // Relative bias table [extent+1][nHeads]; row extent is the
                // zero pad the reference appends.
                // table[d][h] = dot(attn_rel_bias[d], rel[h]) - sink[h].
                // One fused pass: 1024x16x24 MACs is far too small to pay a
                // parallel matvec dispatch per head (24 launches per layer made
                // the tiny relative-bias projection the dominant cost).
                {
                    const float *bias =
                            reinterpret_cast<const float *>(ink.attnRelBias.data.data());
                    for (uint32_t h = 0; h < nHeads; ++h) {
                        table[static_cast<size_t>(h) * (extent + 1) + extent] = 0.0f;
                    }
                    for (uint32_t d = 0; d < extent; ++d) {
                        const float *br = bias + static_cast<size_t>(d) * nRel;
                        for (uint32_t h = 0; h < nHeads; ++h) {
                            const float *rh = rel.data() + static_cast<size_t>(h) * nRel;
                            float acc = 0.0f;
                            for (uint32_t r = 0; r < nRel; ++r) acc += br[r] * rh[r];
                            table[static_cast<size_t>(h) * (extent + 1) + d] =
                                    acc - sink[h];
                        }
                    }
                }

                auto tp2 = nowT();
                const uint32_t firstKey =
                        isSwa && config_.slidingWindow > 0 &&
                                        pos + 1 > config_.slidingWindow
                                ? pos + 1 - config_.slidingWindow
                                : 0;
                const uint32_t nKeys = pos + 1 - firstKey;
                float *kSlot = kvCache_.k.data() +
                               static_cast<size_t>(slot) * config_.maxSeqLen * kvDim;
                float *vSlot = kvCache_.v.data() +
                               static_cast<size_t>(slot) * config_.maxSeqLen * kvDim;
                std::memcpy(kSlot + static_cast<size_t>(pos) * kvDim, k.data(),
                            kvDim * sizeof(float));
                std::memcpy(vSlot + static_cast<size_t>(pos) * kvDim, v.data(),
                            kvDim * sizeof(float));

                std::fill(attnOut.begin(), attnOut.end(), 0.0f);
                std::vector<float> scores(nKeys);
                for (uint32_t hq = 0; hq < nHeads; ++hq) {
                    const uint32_t hkv = hq / group;
                    const float *qh = q.data() + static_cast<size_t>(hq) * headDim;
                    float mx = 0.0f;// sink logit is 0
                    for (uint32_t ki = 0; ki < nKeys; ++ki) {
                        const uint32_t kp = firstKey + ki;
                        const float *kh = kSlot + static_cast<size_t>(kp) * kvDim +
                                          static_cast<size_t>(hkv) * headDim;
                        double dot = 0.0;
                        for (uint32_t d = 0; d < headDim; ++d)
                            dot += double(qh[d]) * kh[d];
                        const uint32_t dist = std::min(pos - kp, extent);
                        const float bias =
                                table[static_cast<size_t>(hq) * (extent + 1) + dist];
                        scores[ki] = float(dot) * qkScale + bias;
                        mx = std::max(mx, scores[ki]);
                    }
                    double sum = std::exp(-mx);// sink: logit 0, value 0
                    for (uint32_t ki = 0; ki < nKeys; ++ki) {
                        scores[ki] = std::exp(scores[ki] - mx);
                        sum += scores[ki];
                    }
                    const double inv = 1.0 / sum;
                    float *oh = attnOut.data() + static_cast<size_t>(hq) * headDim;
                    for (uint32_t ki = 0; ki < nKeys; ++ki) {
                        const float wgt = float(scores[ki] * inv);
                        const uint32_t kp = firstKey + ki;
                        const float *vh = vSlot + static_cast<size_t>(kp) * kvDim +
                                          static_cast<size_t>(hkv) * headDim;
                        for (uint32_t d = 0; d < headDim; ++d) oh[d] += wgt * vh[d];
                    }
                }

                profA += std::chrono::duration<double>(nowT() - tp2).count();
                dumpVec("kqv", attnOut.data(), nHeads * headDim, slot);
                auto tp3 = nowT();
                mv(ink.attnGate, nq.data(), gateProj.data());
                dumpVec("gateProj", gateProj.data(), nHeads * headDim, slot);
                for (uint32_t i = 0; i < nHeads * headDim; ++i) {
                    attnOut[i] *= sigmoidf_(gateProj[i]);
                }
                mv(w.attnO, attnOut.data(), o.data());
                dumpVec("wo_out", o.data(), hidden, slot);
                shortConv(o.data(), hidden, nConv,
                          reinterpret_cast<const float *>(ink.attnPostConv.data.data()),
                          nullptr, 1,
                          hist + static_cast<size_t>(histLen) * (2 * kvDim),
                          convH.data());
                const float postScale = 1.0f / std::sqrt(float(nConv + 1));
                for (uint32_t i = 0; i < hidden; ++i) {
                    o[i] = (o[i] + convH[i]) * postScale;
                }

                profO += std::chrono::duration<double>(nowT() - tp3).count();
                dumpVec("attn_out", o.data(), hidden, slot);
                if (il % blockSize == 0) {
                    blocks.emplace_back(inpSA);
                } else {
                    for (uint32_t i = 0; i < hidden; ++i) o[i] += inpSA[i];
                }
                std::memcpy(x.data(), o.data(), hidden * sizeof(float));
                if (std::getenv("TINYCODER_INKLING_TRACE") != nullptr) {
                    double na = 0.0;
                    for (uint32_t i = 0; i < hidden; ++i)
                        na += double(x[i]) * x[i];
                    std::fprintf(stderr, "a%u:%.3g ", il, std::sqrt(na));
                }

                // FFN: depth mix, optional message, norm, routed MoE.
                std::memcpy(ffnInp.data(), x.data(), hidden * sizeof(float));
                mixWithCurrent(ink.ffnResQuery.data(), 1, mixOut.data());
                if (message != nullptr) {
                    applyControl(mixOut.data(), message + 3 * hidden,
                                 ink.ctrlDown[3], ink.ctrlGain[3],
                                 ink.ctrlForget[3], hidden, normed.data(),
                                 controlScratch);
                    rmsNormVec(normed.data(), w.rmsNormFFN.data(), normed.data(),
                               hidden);
                } else {
                    rmsNormVec(mixOut.data(), w.rmsNormFFN.data(), normed.data(),
                               hidden);
                }

                profF += std::chrono::duration<double>(nowT() - tp3).count();
                auto tp4 = nowT();
                mv(w.ffnGateInp, normed.data(), router.data());
                for (uint32_t e = 0; e < nExperts; ++e) {
                    probs[e] = sigmoidf_(router[e]);
                    const float bias = e < ink.expertProbsBias.size()
                                                ? ink.expertProbsBias.data()[e]
                                                : 0.0f;
                    selection[e] = probs[e] + bias;
                    scored[e] = {selection[e], e};
                }
                std::partial_sort(scored.begin(), scored.begin() + nUsed, scored.end(),
                                  [](const auto &a, const auto &b) {
                                      return a.first > b.first;
                                  });
                float wsum = 0.0f;
                for (uint32_t u = 0; u < nUsed; ++u) wsum += probs[scored[u].second];
                const float wnorm = config_.expertWeightsScale /
                                    std::max(wsum, 1e-20f);
                // One parallel dispatch for all routed experts. Calling the
                // expert matvec once per expert (16 dispatches per layer, 768
                // per token) spent more time waking the pool than doing the
                // arithmetic; the outer expert-parallel loop makes each inner
                // matvec serial (the pool's re-entrancy guard) and keeps 8
                // threads busy.
                moeExpertOuts.resize(static_cast<size_t>(nUsed) * hidden);
                std::vector<float> gateUpLocal(static_cast<size_t>(2) * expertFf);
                std::vector<float> actLocal(static_cast<size_t>(expertFf));
                ThreadPool::current().parallelForSlab(
                        0, nUsed, [&](uint32_t u) {
                            const uint32_t e = scored[u].second;
                            float *gu = gateUpLocal.data();
                            w.ffnGateUpExps.matMulVecRows(normed.data(),
                                                          e * 2 * expertFf,
                                                          2 * expertFf, gu);
                            const float gs = e < ink.expertGateUpScale.size()
                                                     ? ink.expertGateUpScale.data()[e]
                                                     : 1.0f;
                            for (uint32_t j = 0; j < 2 * expertFf; ++j) gu[j] *= gs;
                            float *act = actLocal.data();
                            for (uint32_t j = 0; j < expertFf; ++j) {
                                act[j] = situ(gu[j], gu[expertFf + j], beta, lb);
                            }
                            float *dst = moeExpertOuts.data() +
                                         static_cast<size_t>(u) * hidden;
                            w.ffnDownExps.matMulVecRows(act, e * hidden, hidden,
                                                        dst);
                        });
                if (il == 0 && std::getenv("TINYCODER_INKLING_TRACE") != nullptr) {
                    std::fprintf(stderr, "\n[topk]");
                    for (uint32_t u = 0; u < nUsed; ++u)
                        std::fprintf(stderr, " %u(%.4f)", scored[u].second,
                                     scored[u].first);
                }
                std::fill(moeOut.begin(), moeOut.end(), 0.0f);
                for (uint32_t u = 0; u < nUsed; ++u) {
                    const uint32_t e = scored[u].second;
                    const float wgt = probs[e] * wnorm;
                    const float ds = e < ink.expertDownScale.size()
                                             ? ink.expertDownScale.data()[e]
                                             : 1.0f;
                    const float *src = moeExpertOuts.data() +
                                       static_cast<size_t>(u) * hidden;
                    for (uint32_t j = 0; j < hidden; ++j)
                        moeOut[j] += wgt * ds * src[j];
                }

                profM += std::chrono::duration<double>(nowT() - tp4).count();
                shortConv(moeOut.data(), hidden, nConv,
                          reinterpret_cast<const float *>(ink.ffnPostConv.data.data()),
                          nullptr, 1,
                          hist + static_cast<size_t>(histLen) * (2 * kvDim + hidden),
                          convH.data());
                for (uint32_t i = 0; i < hidden; ++i) {
                    moeOut[i] = (moeOut[i] + convH[i]) * postScale;
                    x[i] = ffnInp[i] + moeOut[i];
                }
                dumpVec("ffn_out", moeOut.data(), hidden, slot);
                if (std::getenv("TINYCODER_INKLING_TRACE") != nullptr) {
                    double nx = 0.0;
                    for (uint32_t i = 0; i < hidden; ++i)
                        nx += double(x[i]) * x[i];
                    std::fprintf(stderr, "L%u:%.3g ", il, std::sqrt(nx));
                }
                if (const char *dd = std::getenv("TINYCODER_INKLING_DUMP")) {
                    char path[1024];
                    std::snprintf(path, sizeof(path), "%s/l_out-%u.f32", dd, slot);
                    FILE *f = std::fopen(path, "wb");
                    if (f) { std::fwrite(x.data(), sizeof(float), hidden, f); std::fclose(f); }
                }
            };

            for (uint32_t il = 0; il < start; ++il) runLayer(il, il, nullptr);

            const std::vector<float> prelude = x;
            const std::vector<std::vector<float>> preludeBlocks = blocks;

            for (uint32_t pass = 0; pass < config_.loopCount; ++pass) {
                std::vector<std::vector<float>> bank;
                std::vector<float> bankOut;
                if (pass > 0) {
                    const size_t skip = preludeBlocks.size() + 1;
                    if (blocks.size() > skip) {
                        bank.assign(blocks.begin() + skip, blocks.end());
                    }
                    bankOut = x;
                    x = prelude;
                    blocks = preludeBlocks;
                }
                for (uint32_t ip = start; ip < end; ++ip) {
                    const uint32_t il = pass == 0
                                                ? ip
                                                : phys + (pass - 1) * core + (ip - start);
                    if (pass == 0) {
                        runLayer(ip, il, nullptr);
                        continue;
                    }

                    // Message queries: mainline stores 4 rows per block;
                    // legacy groups them as
                    // [res(2i), kv(2i), kv(2i+1), res(2i+1)] = q,k,v,ffn.
                    const float *mq = nullptr;
                    std::vector<float> legacy(static_cast<size_t>(4) * hidden);
                    if (!layers_[ip].inkling.msgQuery.empty()) {
                        mq = reinterpret_cast<const float *>(
                                layers_[ip].inkling.msgQuery.data.data());
                    } else if (!inklingLoopResQuery_.empty()) {
                        const uint32_t i = ip - start;
                        std::memcpy(legacy.data() + 0 * hidden,
                                    inklingLoopResQuery_.data() +
                                            static_cast<size_t>(2 * i) * hidden,
                                    hidden * sizeof(float));
                        std::memcpy(legacy.data() + 1 * hidden,
                                    inklingLoopKvResQuery_.data() +
                                            static_cast<size_t>(2 * i) * hidden,
                                    hidden * sizeof(float));
                        std::memcpy(legacy.data() + 2 * hidden,
                                    inklingLoopKvResQuery_.data() +
                                            static_cast<size_t>(2 * i + 1) * hidden,
                                    hidden * sizeof(float));
                        std::memcpy(legacy.data() + 3 * hidden,
                                    inklingLoopResQuery_.data() +
                                            static_cast<size_t>(2 * i + 1) * hidden,
                                    hidden * sizeof(float));
                        mq = legacy.data();
                    }
                    if (mq == nullptr) {
                        std::cerr << "[TinyCoder] Inkling: missing loop_msg_query"
                                  << std::endl;
                        return tinycoder::Array<float>{};
                    }
                    // The bank is the previous pass's blocks after its first
                    // core block, plus that pass's output.
                    std::vector<const float *> bankStates;
                    bankStates.reserve(bank.size() + 1);
                    for (const auto &b: bank) bankStates.push_back(b.data());
                    if (!bankOut.empty()) bankStates.push_back(bankOut.data());
                    if (bankStates.empty()) {
                        std::cerr << "[TinyCoder] Inkling: empty message bank"
                                  << std::endl;
                        return tinycoder::Array<float>{};
                    }
                    mixStates(bankStates, mq, 4, hidden, msgMixed.data(),
                              bankNormed, bankScores);
                    runLayer(ip, il, msgMixed.data());
                }
            }

            for (uint32_t il = end; il < phys; ++il) runLayer(il, il, nullptr);

            mixWithCurrent(inklingOutputResQuery_.data(), 1, mixOut.data());
            rmsNormVec(mixOut.data(), finalNorm_.data(), normed.data(), hidden);
            if (std::getenv("TINYCODER_INKLING_TRACE") != nullptr) {
                double nrm = 0.0, an = 0.0;
                for (uint32_t i = 0; i < hidden; ++i) {
                    nrm += double(normed[i]) * normed[i];
                    an += double(mixOut[i]) * mixOut[i];
                }
                std::fprintf(stderr,
                             "\n[inkling] out_mix=%.6f normed=%.6f lmhead_rows=%u "
                             "outq=%zu finalnorm=%zu toknorm=%zu\n",
                             std::sqrt(an), std::sqrt(nrm), lmHead_.rows,
                             inklingOutputResQuery_.size(), finalNorm_.size(),
                             inklingTokNorm_.size());
            }

            if (!lmHead_.empty()) {
                lmHead_.matMulVec(normed.data(), logitsRow.data());
            } else {
                for (uint32_t id = 0; id < vocab; ++id) {
                    logitsRow[id] = quantizedEmbeddings_.dotRow(normed.data(), id);
                }
            }
            const float cap = config_.finalLogitSoftcapping;
            if (cap > 0.0f) {
                for (uint32_t id = 0; id < vocab; ++id) {
                    logitsRow[id] = cap * std::tanh(logitsRow[id] / cap);
                }
            }
            if (computeAllLogits) {
                std::memcpy(logits.data() + static_cast<size_t>(s) * vocab,
                            logitsRow.data(), vocab * sizeof(float));
            } else if (s + 1 == seqLen) {
                std::memcpy(logits.data(), logitsRow.data(), vocab * sizeof(float));
            }
        }

        if (std::getenv("TINYCODER_INKLING_PROF") != nullptr) {
            std::fprintf(stderr,
                         "\n[prof] qkv=%.3fs conv=%.3fs preattn=%.3fs attn=%.3fs out=%.3fs ffn=%.3fs moe=%.3fs\n",
                         profQ, profC, profT, profA, profO, profF, profM);
        }
        kvCache_.pos += seqLen;
        return logits;
    }

}// namespace tinycoder
