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

#include "Model.hpp"
#include "SIMDMatMulVec.hpp"
#include "ThreadPool.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace tinycoder {

    void Model::computeGemma4MoE(const float *ffnNorm, float *ffnOut,
                                 uint32_t seqLen, uint32_t hiddenSize,
                                 uint32_t intermediateSize,
                                 const LayerWeights &w) const {
        (void) intermediateSize;
        // Gemma4 MoE FFN with top-k expert routing.
        //
        // Architecture:
        //   1. Second pre-FFN norm (preFFWNorm2) applied to input
        //   2. Router (ffnGateInp) computes expert scores
        //   3. Top-k experts selected (k = expertUsedCount)
        //   4. For each selected expert:
        //      a. Gate+Up projection from ffnGateUpExps (fused gate+up per expert)
        //      b. GeGLU activation: gelu(gate) * up
        //      c. Down projection from ffnDownExps
        //   5. Weighted sum of expert outputs by routing probabilities
        //   6. Post-FFN norm 1 (postFFWNorm1)
        //   7. Post-FFN norm 2 (postFFWNorm2)
        //
        // Tensor layouts (after loadQuantized handles 3D→2D flattening):
        //   ffnGateUpExps: [expertCount * expertFF * 2, hiddenSize]
        //     - Expert e: rows [e * expertFF * 2, (e+1) * expertFF * 2)
        //       - Gate: rows [e * expertFF * 2, e * expertFF * 2 + expertFF)
        //       - Up:   rows [e * expertFF * 2 + expertFF, (e+1) * expertFF * 2)
        //   ffnDownExps: [expertCount * hiddenSize, expertFF]
        //     - Expert e: rows [e * hiddenSize, (e+1) * hiddenSize)
        //   ffnGateInp: [hiddenSize, expertCount] (router)

        uint32_t expertCount = config_.expertCount;
        uint32_t expertUsedCount = config_.expertUsedCount;
        uint32_t expertFF = config_.expertFeedForwardLength;

        if (expertCount == 0 || expertUsedCount == 0) {
            std::cerr << "[TinyCoder] computeGemma4MoE: MoE not configured" << std::endl;
            return;
        }

        // Temporary buffers per token
        std::vector<float> routerScores(expertCount);
        std::vector<float> gateBuf(expertFF);
        std::vector<float> upBuf(expertFF);
        std::vector<float> expertOut(hiddenSize);

        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *inputPtr = ffnNorm + s * hiddenSize;

            // Step 1: Apply second pre-FFN norm (preFFWNorm2)
            std::vector<float> normedInput(hiddenSize);
            if (!w.preFFWNorm2.empty()) {
                rmsNormInPlace(inputPtr, normedInput.data(), w.preFFWNorm2.data(), hiddenSize);
            } else {
                std::memcpy(normedInput.data(), inputPtr, hiddenSize * sizeof(float));
            }

            // Step 2: Router - compute expert scores
            np::Array<float> routerOut = w.ffnGateInp.matMulVec(normedInput.data());
            std::memcpy(routerScores.data(), routerOut.data(), expertCount * sizeof(float));

            // Step 3: Select top-k experts
            // Build list of (score, expert_idx) pairs
            std::vector<std::pair<float, uint32_t>> scoredExperts(expertCount);
            for (uint32_t e = 0; e < expertCount; ++e) {
                scoredExperts[e] = {routerScores[e], e};
            }
            // Partial sort to get top-k
            std::partial_sort(scoredExperts.begin(), scoredExperts.begin() + expertUsedCount,
                              scoredExperts.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });

            // Step 4-5: Compute expert outputs and combine
            std::fill(expertOut.begin(), expertOut.end(), 0.0f);

            for (uint32_t r = 0; r < expertUsedCount; ++r) {
                uint32_t expertIdx = scoredExperts[r].second;
                float routingWeight = scoredExperts[r].first;

                // Step 4a: Gate+Up projection for this expert
                // Gate: rows [expertIdx * expertFF * 2, expertIdx * expertFF * 2 + expertFF)
                np::Array<float> gateRow = w.ffnGateUpExps.matMulVecRows(
                        normedInput.data(),
                        expertIdx * expertFF * 2,
                        expertFF);
                std::memcpy(gateBuf.data(), gateRow.data(), expertFF * sizeof(float));

                // Up: rows [expertIdx * expertFF * 2 + expertFF, (expertIdx+1) * expertFF * 2)
                np::Array<float> upRow = w.ffnGateUpExps.matMulVecRows(
                        normedInput.data(),
                        expertIdx * expertFF * 2 + expertFF,
                        expertFF);
                std::memcpy(upBuf.data(), upRow.data(), expertFF * sizeof(float));

                // Step 4b: GeGLU activation: gelu(gate) * up
                geluInPlace(gateBuf.data(), expertFF);
                for (uint32_t i = 0; i < expertFF; ++i) {
                    gateBuf[i] *= upBuf[i];
                }

                // Step 4c: Down projection for this expert
                // rows [expertIdx * hiddenSize, (expertIdx+1) * hiddenSize)
                np::Array<float> downRow = w.ffnDownExps.matMulVecRows(
                        gateBuf.data(),
                        expertIdx * hiddenSize,
                        hiddenSize);

                // Step 5: Weighted sum (routing weight * expert output)
                float weight = routingWeight;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    expertOut[i] += weight * downRow.data()[i];
                }
            }

            // Step 6-7: Post-FFN norms
            float *outPtr = ffnOut + s * hiddenSize;
            if (!w.postFFWNorm1.empty()) {
                rmsNormInPlace(expertOut.data(), outPtr, w.postFFWNorm1.data(), hiddenSize);
            } else {
                std::memcpy(outPtr, expertOut.data(), hiddenSize * sizeof(float));
            }
            if (!w.postFFWNorm2.empty()) {
                rmsNormInPlace(outPtr, outPtr, w.postFFWNorm2.data(), hiddenSize);
            }
        }
    }

    void Model::computeQwen35MoE(const float *ffnNorm, float *ffnOut,
                                 uint32_t seqLen, uint32_t hiddenSize,
                                 uint32_t intermediateSize,
                                 const LayerWeights &w) const {
        (void) intermediateSize;
        // Qwen35MoE MoE FFN with softmax-gated top-k expert routing + shared
        // expert.  Mirrors llama.cpp qwen35moe build_layer_ffn + build_moe_ffn
        // (LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, norm_w=true):
        //
        //   1. Router logits = ffn_gate_inp @ x         [expertCount]
        //   2. probs = softmax(logits)                   (over ALL experts)
        //   3. top-k (expertUsedCount) experts by probs
        //   4. weights = probs[selected] RE-NORMALIZED (sum to 1)
        //   5. per expert: gate/up = ffn_gate_exps/ffn_up_exps @ x, SwiGLU,
        //      down = ffn_down_exps @ silu(gate)*up; out += weight[e] * down
        //   6. shared expert: ffn_*_shexp @ x with sigmoid(ffn_gate_inp_shexp @ x)
        //
        // Tensor layouts (after loadQuantized flattens 3D→2D):
        //   ffnGateExps/ffnUpExps: [expertCount * expertFF, hiddenSize]
        //     - Expert e: rows [e * expertFF, (e+1) * expertFF)
        //   ffnDownExpsMoe: [expertCount * hiddenSize, expertFF]
        //     - Expert e: rows [e * hiddenSize, (e+1) * hiddenSize)
        //   ffnGateInpMoe: [hiddenSize, expertCount] (router)
        //
        // expert_weights_scale: llama applies hparams.expert_weights_scale to the
        // normalized weights; the Qwen3.6 models export none (1.0), so it is
        // skipped here.

        uint32_t expertCount = config_.expertCount;
        uint32_t expertUsedCount = config_.expertUsedCount;
        uint32_t expertFF = config_.expertFeedForwardLength;
        uint32_t sharedExpertFF = config_.expertSharedFeedForwardLength;

        if (expertCount == 0 || expertUsedCount == 0) {
            std::cerr << "[TinyCoder] computeQwen35MoE: MoE not configured" << std::endl;
            return;
        }

        // Temporary buffers per token
        std::vector<float> routerLogits(expertCount);
        std::vector<float> gateBuf(expertFF);
        std::vector<float> upBuf(expertFF);
        std::vector<float> expertOut(hiddenSize);

        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *inputPtr = ffnNorm + s * hiddenSize;

            // Step 1: Router logits
            w.ffnGateInpMoe.matMulVec(inputPtr, routerLogits.data());

            // Step 2: softmax over ALL experts (llama ggml_soft_max).
            float maxL = -std::numeric_limits<float>::infinity();
            for (uint32_t e = 0; e < expertCount; ++e) {
                maxL = std::max(maxL, routerLogits[e]);
            }
            double sumExp = 0.0;
            std::vector<float> probs(expertCount);
            for (uint32_t e = 0; e < expertCount; ++e) {
                probs[e] = std::exp(static_cast<double>(routerLogits[e]) - maxL);
                sumExp += probs[e];
            }
            const float invSum = static_cast<float>(1.0 / sumExp);
            for (uint32_t e = 0; e < expertCount; ++e) {
                probs[e] *= invSum;
            }

            // Step 3: select top-k by probability (descending).
            std::vector<std::pair<float, uint32_t>> scoredExperts(expertCount);
            for (uint32_t e = 0; e < expertCount; ++e) {
                scoredExperts[e] = {probs[e], e};
            }
            std::partial_sort(scoredExperts.begin(),
                              scoredExperts.begin() + expertUsedCount,
                              scoredExperts.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });

            // Step 4: renormalize weights over the selected top-k
            // (llama norm_w=true: clamp the sum to >= 6.1e-5, then divide).
            double wsum = 0.0;
            for (uint32_t r = 0; r < expertUsedCount; ++r) {
                wsum += scoredExperts[r].first;
            }
            const float wsumClamped = static_cast<float>(std::max(wsum, 6.103515625e-5));

            // Step 5: compute expert outputs and combine.
            std::fill(expertOut.begin(), expertOut.end(), 0.0f);
            {
                // Per-expert down buffers hold hiddenSize floats (not expertFF).
                std::vector<float> downBuf(hiddenSize);
                // Fused vs separate expert layout (llama ffn_gate_up_exps vs
                // ffn_gate_exps + ffn_up_exps).
                const bool fused =
                        !w.ffnGateUpExpsMoe.empty() && w.ffnUpExps.empty();
                const QuantizedMatrix &gateM =
                        fused ? w.ffnGateUpExpsMoe : w.ffnGateExps;
                const QuantizedMatrix *upM =
                        fused ? nullptr : &w.ffnUpExps;
                for (uint32_t r = 0; r < expertUsedCount; ++r) {
                    uint32_t expertIdx = scoredExperts[r].second;
                    float routingWeight = scoredExperts[r].first / wsumClamped;

                    if (fused) {
                        // gate half: [e*ff*2, e*ff*2+ff); up half: [e*ff*2+ff, ...)
                        gateM.matMulVecRows(inputPtr, expertIdx * expertFF * 2,
                                            expertFF, gateBuf.data());
                        gateM.matMulVecRows(inputPtr,
                                            expertIdx * expertFF * 2 + expertFF,
                                            expertFF, upBuf.data());
                    } else {
                        // separate gate/up
                        gateM.matMulVecRows(inputPtr, expertIdx * expertFF,
                                            expertFF, gateBuf.data());
                        upM->matMulVecRows(inputPtr, expertIdx * expertFF,
                                           expertFF, upBuf.data());
                    }

                    // SwiGLU: silu(gate) * up
                    for (uint32_t i = 0; i < expertFF; ++i) {
                        gateBuf[i] =
                                (gateBuf[i] / (1.0f + std::exp(-gateBuf[i]))) *
                                upBuf[i];
                    }

                    // Down projection
                    w.ffnDownExpsMoe.matMulVecRows(gateBuf.data(),
                                                   expertIdx * hiddenSize,
                                                   hiddenSize, downBuf.data());
                    for (uint32_t i = 0; i < hiddenSize; ++i) {
                        expertOut[i] += routingWeight * downBuf[i];
                    }
                }
            }

            // Shared expert (sigmoid-gated SwiGLU; llama build_layer_ffn).
            if (!w.ffnGateShexp.empty() && !w.ffnUpShexp.empty() &&
                !w.ffnDownShexp.empty()) {
                float sharedGateWeight = 1.0f;
                if (!w.ffnGateInpShexp.empty()) {
                    float g = 0.0f;
                    w.ffnGateInpShexp.matMulVec(inputPtr, &g);
                    sharedGateWeight = 1.0f / (1.0f + std::exp(-g));// sigmoid
                }

                std::vector<float> sharedGate(sharedExpertFF);
                std::vector<float> sharedUp(sharedExpertFF);
                w.ffnGateShexp.matMulVec(inputPtr, sharedGate.data());
                w.ffnUpShexp.matMulVec(inputPtr, sharedUp.data());
                for (uint32_t i = 0; i < sharedExpertFF; ++i) {
                    sharedGate[i] =
                            (sharedGate[i] / (1.0f + std::exp(-sharedGate[i]))) *
                            sharedUp[i];
                }
                std::vector<float> sharedDown(hiddenSize);
                w.ffnDownShexp.matMulVec(sharedGate.data(), sharedDown.data());
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    expertOut[i] += sharedGateWeight * sharedDown[i];
                }
            }

            // Write output
            std::memcpy(ffnOut + s * hiddenSize, expertOut.data(),
                        hiddenSize * sizeof(float));
        }
    }

    void Model::computeQwen35MoERouterOnly(const float *ffnNorm, uint32_t seqLen,
                                           uint32_t hiddenSize,
                                           const LayerWeights &w,
                                           int32_t *expertIdxOut,
                                           float *expertWgtOut) const {
        // CPU-side ROUTER ONLY (hybrid GPU-expert mode): identical selection
        // math to computeQwen35MoEFromLogits' reference router (fp32
        // ffnGateInpMoe @ norm + double softmax + stable-by-index top-k +
        // renormalize with the norm_w clamp), writing the per-(token, rank)
        // tables WITHOUT running the expert FFNs — those run on the GPU via
        // the per-layer expert cache (GPUCompute.cu ensureExpertCached).
        uint32_t expertCount = config_.expertCount;
        uint32_t expertUsedCount = config_.expertUsedCount;
        const uint32_t eu = expertUsedCount;
        if (expertCount == 0 || eu == 0) return;

        std::vector<float> routerLogits(expertCount);
        std::vector<float> probs(expertCount);
        std::vector<std::pair<float, uint32_t>> scored(expertCount);
        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *x = ffnNorm + static_cast<size_t>(s) * hiddenSize;
            w.ffnGateInpMoe.matMulVec(x, routerLogits.data());
            float maxL = -std::numeric_limits<float>::infinity();
            for (uint32_t e = 0; e < expertCount; ++e) {
                maxL = std::max(maxL, routerLogits[e]);
            }
            double sumExp = 0.0;
            for (uint32_t e = 0; e < expertCount; ++e) {
                probs[e] = std::exp(static_cast<double>(routerLogits[e]) - maxL);
                sumExp += probs[e];
            }
            const float invSum = static_cast<float>(1.0 / sumExp);
            for (uint32_t e = 0; e < expertCount; ++e) {
                probs[e] *= invSum;
            }
            for (uint32_t e = 0; e < expertCount; ++e) {
                scored[e] = {probs[e], e};
            }
            std::partial_sort(scored.begin(), scored.begin() + eu, scored.end(),
                              [](const auto &a, const auto &b) {
                                  return a.first > b.first;
                              });
            double wsum = 0.0;
            for (uint32_t r = 0; r < eu; ++r) {
                wsum += scored[r].first;
            }
            const float wsumClamped =
                    static_cast<float>(std::max(wsum, 6.103515625e-5));
            for (uint32_t r = 0; r < eu; ++r) {
                expertIdxOut[static_cast<size_t>(s) * eu + r] =
                        static_cast<int32_t>(scored[r].second);
                expertWgtOut[static_cast<size_t>(s) * eu + r] =
                        scored[r].first / wsumClamped;
            }
        }
    }

    void Model::computeQwen35MoEFromLogits(const float *ffnNorm, float *ffnOut,
                                           uint32_t seqLen, uint32_t hiddenSize,
                                           const LayerWeights &w) const {
        // Routed-expert half of computeQwen35MoE ONLY (no shared expert):
        // the GPU hybrid driver handed us the post-attn norm mirror, and THIS
        // function runs the ROUTER on the CPU (fp32 ffnGateInpMoe @ norm —
        // reference math) + softmax + top-k + renormalization + the 8 routed
        // expert FFNs — the exact math of computeQwen35MoE steps 1-5.
        //
        // Running the router here (instead of consuming GPU-computed logits)
        // is what makes the expert selection IDENTICAL for BATCH and
        // SINGLE-TOKEN prefills: the GPU's fp16 prefill GEMM and its fp32
        // decode GEMV round the router logits slightly differently, which can
        // flip a borderline expert in the top-8, break batch-vs-sequential
        // parity and even diverge from the CPU reference.  The fp32 CPU router
        // is bit-identical for both, and matches the pure-CPU path (which also
        // uses w.ffnGateInpMoe.matMulVec).  The GPU still runs attention, the
        // shared expert and the residual — the router is a 256x2048 mat-vec per
        // token (~1% of the layer FLOPs), so the CPU cost is negligible.
        //
        // The expert FFN part is parallelized over (token, rank) pairs: for a
        // single-token decode that is expertUsed (=8) independent tasks across
        // the pool — the routed experts are the ONLY CPU work left in the
        // hybrid, so they decide decode speed.
        //
        // Math (must stay bit-identical to the reference loop):
        //   per token s:
        //     logits = ffnGateInpMoe @ norm[s]     (fp32, host matrix)
        //     probs = softmax(logits)               (double accumulation)
        //     top-k by probs (descending, stable-by-index for ties)
        //     wsum = sum(probs[top-k]); wsumClamped = max(wsum, 6.1e-5)
        //     per rank r: e, wg = probs[selected]/wsumClamped
        //       gate = ffnGateExps @ x      [expertFF]  (or fused ffnGateUpExps)
        //       up   = ffnUpExps @ x        [expertFF]
        //       gate = silu(gate) * up
        //       down = ffnDownExpsMoe @ gate [hiddenSize]
        //       ffnOut[s] += wg * down
        uint32_t expertCount = config_.expertCount;
        uint32_t expertUsedCount = config_.expertUsedCount;
        uint32_t expertFF = config_.expertFeedForwardLength;

        if (expertCount == 0 || expertUsedCount == 0) {
            std::cerr << "[TinyCoder] computeQwen35MoEFromLogits: MoE not "
                         "configured"
                      << std::endl;
            return;
        }

        // Per-expert matrices: fused (ffnGateUpExpsMoe) vs separate
        // (ffnGateExps + ffnUpExps); same rule as computeQwen35MoE.
        const bool fused = !w.ffnGateUpExpsMoe.empty() && w.ffnUpExps.empty();
        const QuantizedMatrix &gateM = fused ? w.ffnGateUpExpsMoe : w.ffnGateExps;
        const QuantizedMatrix *upM = fused ? nullptr : &w.ffnUpExps;

        const uint32_t eu = expertUsedCount;

        // ---- Reference router (CPU, fp32): softmax + top-k + renormalize ----
        // Build the per-token (rank -> expertId, weight) tables.
        std::vector<int32_t> expertIdx(static_cast<size_t>(seqLen) * eu);
        std::vector<float> expertWgt(static_cast<size_t>(seqLen) * eu);
        {
            std::vector<float> routerLogits(expertCount);
            std::vector<float> probs(expertCount);
            std::vector<std::pair<float, uint32_t>> scored(expertCount);
            for (uint32_t s = 0; s < seqLen; ++s) {
                const float *x = ffnNorm + static_cast<size_t>(s) * hiddenSize;
                // Router: logits[e] = sum_k ffnGateInpMoe[e][k] * norm[s][k].
                // ffnGateInpMoe is the F32 [expertCount x hiddenSize] router;
                // the same host matrix the CPU reference uses, so the logits
                // are bit-identical to the pure-CPU path AND to a seqLen=1
                // decode (which runs the identical fp32 matMulVec).
                w.ffnGateInpMoe.matMulVec(x, routerLogits.data());
                float maxL = -std::numeric_limits<float>::infinity();
                for (uint32_t e = 0; e < expertCount; ++e) {
                    maxL = std::max(maxL, routerLogits[e]);
                }
                double sumExp = 0.0;
                for (uint32_t e = 0; e < expertCount; ++e) {
                    probs[e] = std::exp(static_cast<double>(routerLogits[e]) - maxL);
                    sumExp += probs[e];
                }
                const float invSum = static_cast<float>(1.0 / sumExp);
                for (uint32_t e = 0; e < expertCount; ++e) {
                    probs[e] *= invSum;
                }
                for (uint32_t e = 0; e < expertCount; ++e) {
                    scored[e] = {probs[e], e};
                }
                std::partial_sort(scored.begin(), scored.begin() + eu, scored.end(),
                                  [](const auto &a, const auto &b) {
                                      return a.first > b.first;
                                  });
                double wsum = 0.0;
                for (uint32_t r = 0; r < eu; ++r) {
                    wsum += scored[r].first;
                }
                const float wsumClamped =
                        static_cast<float>(std::max(wsum, 6.103515625e-5));
                for (uint32_t r = 0; r < eu; ++r) {
                    expertIdx[static_cast<size_t>(s) * eu + r] =
                            static_cast<int32_t>(scored[r].second);
                    expertWgt[static_cast<size_t>(s) * eu + r] =
                            scored[r].first / wsumClamped;
                }
            }
        }

        // ---- Routed experts: BATCH over per-expert token groups ----
        // (2026-09-15 rewrite)  The old body dispatched parallelFor2D(seqLen,
        // eu) per-(s, r) tasks; each task called the SINGLE-token
        // matMulVecRows, which from a pool worker (reentrancy guard) ran its
        // internal slab tiles SERIALLY.  A 36-token prefill therefore executed
        // 288 expert GEMVs on ONE thread each (~2.1 s/token wall, the
        // cpuRouterMs in the [moe fwd] stats) — and each token re-read the
        // expert's weight bytes independently.
        //
        // New structure: group the (s, r) selections by expert id and run EACH
        // group as ONE batch GEMM through matMulVecBatchQ8_0_SIMD (Q8_0
        // expert matrices; the kernel takes an arbitrary CONTIGUOUS row slice
        // [expertFF rows offset ue*expertFF] and a seqLen token batch).  Since
        // this function runs on the MAIN thread (the GPU hybrid callback), the
        // kernel's internal parallelForSlab is a TOP-LEVEL pool dispatch (not
        // nested), so all 8 threads tile the expert rows — the row parallelism
        // that was previously dead under the reentrancy guard.  The group
        // weights are also read ONCE per expert instead of once per member
        // (weight-stationary; up to seqLen× less DRAM traffic per prefill).
        //
        // Bit-exactness: matMulVecBatchQ8_0_SIMD computes each (token, row)
        // dot with the identical per-block int8 math and per-token x
        // quantization bytes as the per-token matMulVecRows reference (same
        // algorithm, same op order).  The per-(s, r) partials are written into
        // the same disjoint slab rows and reduced in the exact rank order
        // below, so no accumulation order changes.
        const size_t slabRows = static_cast<size_t>(seqLen) * eu;
        std::vector<float> partial(static_cast<size_t>(slabRows) * hiddenSize, 0.0f);

        // Group the (s, r) selections by expert id.  O(eu * seqLen) per layer.
        // Decode (seqLen == 1) yields eu single-member groups; prefill yields
        // up to min(expertCount, seqLen*eu) non-empty groups.
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> expertGroups(
                expertCount);
        uint32_t maxGrp = 0;
        for (uint32_t s = 0; s < seqLen; ++s) {
            for (uint32_t r = 0; r < eu; ++r) {
                const int32_t e = expertIdx[static_cast<size_t>(s) * eu + r];
                if (e < 0) continue;
                expertGroups[static_cast<size_t>(e)].emplace_back(
                        std::make_pair(s, r));
                maxGrp = std::max(maxGrp,
                                  static_cast<uint32_t>(
                                          expertGroups[static_cast<size_t>(e)]
                                                  .size()));
            }
        }

        // One-time batch capability check: the grouped path needs all three
        // expert matrices in Q8_0 (the only type matMulVecBatchQ8_0_SIMD
        // covers), 32-block-aligned input/output dims (the Q8_0 block size),
        // and the SIMD kernels enabled — the exact conditions
        // QuantizedMatrix::matMulVecRows applies to its own Q8_0 fast path.
        // Anything else falls back to the bit-exact per-member loop below.
        const bool canBatch =
                !forceScalarPath() && (hiddenSize & 31) == 0 &&
                (expertFF & 31) == 0 && gateM.type == GGML_TYPE_Q8_0 &&
                (fused || upM->type == GGML_TYPE_Q8_0) &&
                w.ffnDownExpsMoe.type == GGML_TYPE_Q8_0;

        // Reusable grow-on-demand scratch (batch of x rows + batched outputs).
        std::vector<float> grpX, grpG, grpU, grpD;
        std::vector<uint32_t> grpS(maxGrp), grpR(maxGrp);
        if (canBatch) {
            grpX.resize(static_cast<size_t>(maxGrp) * hiddenSize);
            grpG.resize(static_cast<size_t>(maxGrp) * expertFF);
            grpU.resize(static_cast<size_t>(maxGrp) * expertFF);
            grpD.resize(static_cast<size_t>(maxGrp) * hiddenSize);
        }

        // Row stride in bytes for a quantized matrix — mirrors the
        // blocksPerRow * typeSize math of QuantizedMatrix::matMulVecRows.
        auto rowBytes = [](const QuantizedMatrix &m) -> uint64_t {
            const uint32_t bs = ggmlBlockSize(m.type);
            const uint32_t blocks = bs ? (m.cols + bs - 1) / bs : 0;
            return static_cast<uint64_t>(blocks) * ggmlTypeSize(m.type);
        };

        // routeBatch: ONE expert's gate/up/down for a grpCount-token group as
        // three batch GEMMs (grpX must already hold the group's x rows).
        // Each (token, row) dot is computed by the identical per-block int8
        // math + per-token x quantization as the single-token reference, so
        // results are bit-exact per element.  Returns false -> caller falls
        // back to the exact per-member loop (e.g. no-AVX2 host).
        auto routeBatch = [&](uint32_t grpCount, const uint8_t *gateRows,
                              const uint8_t *upRows, const uint8_t *downRows) {
            if (!matMulVecBatchQ8_0_SIMD(gateRows, grpX.data(), grpCount,
                                         expertFF, hiddenSize, grpG.data())) {
                return false;
            }
            if (!matMulVecBatchQ8_0_SIMD(upRows, grpX.data(), grpCount,
                                         expertFF, hiddenSize, grpU.data())) {
                return false;
            }
            // SwiGLU: g = silu(g) * u (per token, per element; same expression
            // as the reference loops, line for line).
            for (uint32_t m = 0; m < grpCount; ++m) {
                const float *uRow = grpU.data() + static_cast<size_t>(m) * expertFF;
                float *gRow = grpG.data() + static_cast<size_t>(m) * expertFF;
                for (uint32_t i = 0; i < expertFF; ++i) {
                    gRow[i] = (gRow[i] / (1.0f + std::exp(-gRow[i]))) * uRow[i];
                }
            }
            if (!matMulVecBatchQ8_0_SIMD(downRows, grpG.data(), grpCount,
                                         hiddenSize, expertFF, grpD.data())) {
                return false;
            }
            return true;
        };

        for (uint32_t ue = 0; ue < expertCount; ++ue) {
            const auto &members = expertGroups[ue];
            const uint32_t grpCount = static_cast<uint32_t>(members.size());
            if (grpCount == 0) continue;

            const uint32_t gRows0 = ue * expertFF;
            const uint32_t dRows0 = ue * hiddenSize;

            if (canBatch) {
                // Gather the group's x rows (contiguous [grpCount][hiddenSize]).
                // Stream from ffnNorm (read-only) into the reusable scratch.
                for (uint32_t m = 0; m < grpCount; ++m) {
                    const uint32_t s = members[m].first;
                    const uint32_t r = members[m].second;
                    grpS[m] = s;
                    grpR[m] = r;
                    std::memcpy(grpX.data() + static_cast<size_t>(m) * hiddenSize,
                                ffnNorm + static_cast<size_t>(s) * hiddenSize,
                                hiddenSize * sizeof(float));
                }

                const uint8_t *gateRows, *upRows, *downRows;
                if (fused) {
                    // Fused [exp*ff*2, hidden]: gate half then up half, both
                    // inside the same matrix (rows e*ff*2 .. e*ff*2 + 2*ff).
                    const uint64_t gBpr = rowBytes(gateM);
                    gateRows = gateM.data.data() +
                               static_cast<uint64_t>(gRows0) * 2 * gBpr;
                    upRows = gateRows + static_cast<uint64_t>(expertFF) * gBpr;
                } else {
                    // Separate gate/up matrices, same row base.
                    gateRows = gateM.data.data() +
                               static_cast<uint64_t>(gRows0) * rowBytes(gateM);
                    upRows = upM->data.data() +
                             static_cast<uint64_t>(gRows0) * rowBytes(*upM);
                }
                downRows = w.ffnDownExpsMoe.data.data() +
                           static_cast<uint64_t>(dRows0) *
                                   rowBytes(w.ffnDownExpsMoe);

                if (routeBatch(grpCount, gateRows, upRows, downRows)) {
                    // Scatter the weighted partials into the disjoint slab
                    // rows (bit-identical wg * down per (s, r)).
                    for (uint32_t m = 0; m < grpCount; ++m) {
                        const uint32_t s = grpS[m];
                        const uint32_t r = grpR[m];
                        const float wg =
                                expertWgt[static_cast<size_t>(s) * eu + r];
                        const float *src =
                                grpD.data() + static_cast<size_t>(m) * hiddenSize;
                        float *dst = partial.data() +
                                     (static_cast<size_t>(s) * eu + r) * hiddenSize;
                        for (uint32_t i = 0; i < hiddenSize; ++i) {
                            dst[i] = wg * src[i];
                        }
                    }
                    continue;
                }
            }

            // Fallback (non-AVX2 host / forceScalar / non-batchable layout):
            // the exact per-member single-token math (identical results).  The
            // fused row base is e*ff*2 (gate half then up half) — same as the
            // reference computeQwen35MoE.
            std::vector<float> g(expertFF), u(expertFF), d(hiddenSize);
            for (uint32_t m = 0; m < grpCount; ++m) {
                const uint32_t s = members[m].first;
                const uint32_t r = members[m].second;
                const float wg = expertWgt[static_cast<size_t>(s) * eu + r];
                const float *x = ffnNorm + static_cast<size_t>(s) * hiddenSize;
                if (fused) {
                    gateM.matMulVecRows(x, gRows0 * 2, expertFF, g.data());
                    gateM.matMulVecRows(x, gRows0 * 2 + expertFF, expertFF,
                                        u.data());
                } else {
                    gateM.matMulVecRows(x, gRows0, expertFF, g.data());
                    upM->matMulVecRows(x, gRows0, expertFF, u.data());
                }
                for (uint32_t i = 0; i < expertFF; ++i) {
                    g[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
                }
                w.ffnDownExpsMoe.matMulVecRows(g.data(), dRows0, hiddenSize,
                                               d.data());
                float *dst = partial.data() +
                             (static_cast<size_t>(s) * eu + r) * hiddenSize;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    dst[i] = wg * d[i];
                }
            }
        }

        // Reduce the per-(s, r) partials into ffnOut in rank order (matches the
        // reference accumulation: rank 0 first ... rank eu-1 last).
        std::fill(ffnOut, ffnOut + static_cast<size_t>(seqLen) * hiddenSize, 0.0f);
        for (uint32_t s = 0; s < seqLen; ++s) {
            float *o = ffnOut + static_cast<size_t>(s) * hiddenSize;
            for (uint32_t r = 0; r < eu; ++r) {
                const float *src = partial.data() +
                                   (static_cast<size_t>(s) * eu + r) * hiddenSize;
                for (uint32_t i = 0; i < hiddenSize; ++i) {
                    o[i] += src[i];
                }
            }
        }
    }

}// namespace tinycoder
