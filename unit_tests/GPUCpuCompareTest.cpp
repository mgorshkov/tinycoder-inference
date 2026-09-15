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

/**
 * TinyCoder GPU vs CPU comparison tests.
 *
 * These tests run the SAME forward pass through both the CPU engine and the
 * GPU offload engine and compare the resulting logits. The goal is to catch
 * GPU-only regressions that produce "garbage" output while the CPU path is
 * correct (e.g. in-place RMSNorm destroying the residual stream, or an
 * RMSNorm launch grid that only normalizes the first token).
 *
 * The comparison is intentionally lossy-tolerant:
 *  - Q2_K quantization is inherently lossy vs the CPU reference math
 *    (fp16 GEMM prefill vs fp32), so exact logit equality is NOT expected.
 *  - The top-1/top-5 token *identity* and the logit magnitude/scale must
 *    agree - a broken GPU engine produces a nearly flat or random
 *    distribution whose argmax diverges.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "Model.hpp"
#include "ModelConfig.hpp"
#include "SIMDMatMulVec.hpp"
#include "SharedTestEnv.hpp"
#include "Tokenizer.hpp"

#ifdef USE_CUDA
#include "GPUCompute.hpp"
#endif

// ---------------------------------------------------------------------------
// Fixture + helpers
// ---------------------------------------------------------------------------

class GPUCpuCompareTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_NE(SharedTestEnv::model, nullptr) << "Model not loaded -- skipping test";
        ASSERT_TRUE(SharedTestEnv::modelLoaded);
    }
};

/// @brief The chat-formatted Paris prompt (the failing generation in the bug
/// report: 40 prompt tokens).
static std::string parisChatPrompt() {
    return SharedTestEnv::model->formatChat(
            {{"system", "You are TinyCoder, an AI coding assistant. Be concise."},
             {"user", "What is the capital of France?"}},
            true);
}

/// @brief Normalize a forward() result to the flat LAST-token logit row.
///
/// Shape normalization: the CPU engine allocates the full [seqLen, vocabSize]
/// array but (with computeAllLogits=false) only computes the LAST token's logit
/// row - the other rows are uninitialized padding. The GPU engine returns a
/// compact [1, vocabSize] array. Normalize both to the flat last-token row so
/// the comparisons operate on identical shapes.
static np::Array<float> lastLogitRow(np::Array<float> logits) {
    const uint32_t vocab = SharedTestEnv::config.vocabSize;
    if (logits.size() <= vocab) {
        // Single-token pass: shapes already agree ([1, vocabSize]).
        return logits;
    }

    // Multi-token pass: strip everything except the last token's row.
    np::Array<float> lastRow = np::Array<float>(np::Shape{vocab});
    const float *src = logits.data() + (logits.size() - vocab);
    std::memcpy(lastRow.data(), src, vocab * sizeof(float));
    return lastRow;
}

/// @brief Run one forward pass with the GPU engine explicitly enabled or
/// disabled via $TINYCODER_GPU, returning ONLY the last token's logits.
/// Clears BOTH KV caches first (fresh-session semantics).
static np::Array<float> forwardLastLogits(const std::vector<int32_t> &tokens,
                                          bool wantGpu) {
    if (wantGpu) {
        setenv("TINYCODER_GPU", "1", 1);
    } else {
        setenv("TINYCODER_GPU", "0", 1);
    }
    SharedTestEnv::model->clearKVCache();
    return lastLogitRow(SharedTestEnv::model->forward(tokens,
                                                      /*computeAllLogits=*/false));
}

/// @brief Top-N (logit, id) pairs in descending logit order.
static std::vector<std::pair<float, int32_t>> topN(const np::Array<float> &logits,
                                                   size_t n) {
    std::vector<std::pair<float, int32_t>> v;
    v.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        v.emplace_back(logits.get(static_cast<uint32_t>(i)),
                       static_cast<int32_t>(i));
    }
    std::partial_sort(v.begin(), v.begin() + std::min(n, v.size()), v.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    v.resize(std::min(n, v.size()));
    return v;
}

static void printTop(tinycoder::Tokenizer &tok, const std::string &label,
                     const np::Array<float> &logits, size_t n) {
    std::cout << "  " << label << ":";
    auto top = topN(logits, n);
    for (const auto &kv: top) {
        std::string text = tok.decodeToken(kv.second);
        std::cout << " [" << kv.second << " " << text << " logit=" << kv.first
                  << "]";
    }
    std::cout << std::endl;
}

static void printTopPairs(tinycoder::Tokenizer &tok, const std::string &label,
                          const std::vector<std::pair<float, int32_t>> &top) {
    std::cout << "  " << label << ":";
    for (const auto &kv: top) {
        std::string text = tok.decodeToken(kv.second);
        std::cout << " [" << kv.second << " " << text << " logit=" << kv.first
                  << "]";
    }
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// GPU vs CPU logits comparison
// ---------------------------------------------------------------------------

// The GPU engine is compiled in only when USE_CUDA is defined. Without CUDA
// this test trivially passes (both "GPU" and "CPU" runs are the CPU engine).
#ifdef USE_CUDA

TEST_F(GPUCpuCompareTest, ParisPromptLogitsAgree) {
    tinycoder::Model *model = SharedTestEnv::model;
    tinycoder::Tokenizer &tokenizer = model->tokenizer();

    // Skip when no usable CUDA device is present (the GPU forward falls back
    // to CPU internally, so the test would compare CPU vs CPU).
    if (!tinycoder::gpu::gpuEnabled()) {
        GTEST_SKIP() << "GPU disabled ($TINYCODER_GPU=0) or no CUDA device";
    }

    std::string prompt = parisChatPrompt();
    auto tokens = tokenizer.encode(prompt);
    ASSERT_GT(tokens.size(), 1u) << "Prompt must tokenize to >= 2 tokens";

    // TINYCODER_DUMP_NORM=1: dump the CPU reference embedding row for the
    // first token (so it can be compared against the GPU's [gpu emb] dump).
    const char *dnT = std::getenv("TINYCODER_DUMP_NORM");
    if (dnT != nullptr && dnT[0] != '\0' && dnT[0] != '0') {
        auto embRow = model->debugGetEmbedding(tokens[0]);
        if (embRow.size() >= 8u) {
            double esum = 0.0;
            for (float x: embRow) esum += static_cast<double>(x) * x;
            std::fprintf(stderr,
                         "[cpuemb] t=%d emb[0..7]={%.6f %.6f %.6f %.6f %.6f "
                         "%.6f %.6f %.6f} ||h||=%.3f\n",
                         tokens[0], embRow[0], embRow[1], embRow[2], embRow[3],
                         embRow[4], embRow[5], embRow[6], embRow[7],
                         std::sqrt(esum));
        }
    }

    std::cout << "GPUCpuCompareTest: " << prompt.size() << " chars -> "
              << tokens.size() << " prompt tokens" << std::endl;

    // CPU logits (the known-good reference path).
    np::Array<float> cpuLogits = forwardLastLogits(tokens, /*wantGpu=*/false);
    ASSERT_FALSE(cpuLogits.empty());

    // GPU logits (the offload engine).
    np::Array<float> gpuLogits = forwardLastLogits(tokens, /*wantGpu=*/true);
    ASSERT_FALSE(gpuLogits.empty());

    const uint32_t vocab = static_cast<uint32_t>(cpuLogits.size());
    ASSERT_EQ(gpuLogits.size(), cpuLogits.size());

    printTop(tokenizer, "CPU top-5", cpuLogits, 5);
    printTop(tokenizer, "GPU top-5", gpuLogits, 5);

    auto cpuTop = topN(cpuLogits, 5);
    auto gpuTop = topN(gpuLogits, 5);
    ASSERT_FALSE(cpuTop.empty());
    ASSERT_FALSE(gpuTop.empty());

    // The top-1 token must agree.  A GPU engine that silently computes garbage
    // (broken RMSNorm / residual stream) yields a nearly flat distribution and
    // a different argmax.
    EXPECT_EQ(cpuTop[0].second, gpuTop[0].second)
            << "GPU top-1 token differs from CPU top-1: cpu=" << cpuTop[0].second
            << " gpu=" << gpuTop[0].second;

    // Allow >= 60% of the CPU top-5 to appear in the GPU top-5 (Q2_K noise).
    size_t overlap = 0;
    for (const auto &c: cpuTop) {
        for (const auto &g: gpuTop) {
            if (c.second == g.second) {
                ++overlap;
                break;
            }
        }
    }
    EXPECT_GE(overlap, 3u) << "GPU top-5 shares too few tokens with CPU top-5";

    // Logits must have comparable magnitude/scale.  Garbage distributions are
    // typically flat (near-zero range) or wildly inflated.
    float cpuMax = -std::numeric_limits<float>::max();
    float gpuMax = -std::numeric_limits<float>::max();
    float cpuMin = std::numeric_limits<float>::max();
    float gpuMin = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < vocab; ++i) {
        cpuMax = std::max(cpuMax, cpuLogits.get(i));
        cpuMin = std::min(cpuMin, cpuLogits.get(i));
        gpuMax = std::max(gpuMax, gpuLogits.get(i));
        gpuMin = std::min(gpuMin, gpuLogits.get(i));
    }
    const float cpuRange = cpuMax - cpuMin;
    const float gpuRange = gpuMax - gpuMin;
    std::cout << "  CPU logit range: [" << cpuMin << ", " << cpuMax
              << "] (width " << cpuRange << ")" << std::endl;
    std::cout << "  GPU logit range: [" << gpuMin << ", " << gpuMax
              << "] (width " << gpuRange << ")" << std::endl;
    // The GPU prefill uses fp16 GEMMs (lossy vs CPU fp32) but the global scale
    // must match within a generous factor.  A garbage engine has a range off
    // by >> 2x.
    EXPECT_GT(cpuRange, 1.0f) << "CPU logits unexpectedly flat -- test is broken";
    EXPECT_GT(gpuRange, cpuRange / 4.0f)
            << "GPU logit range collapsed vs CPU -- residual/RMSNorm bug";
    EXPECT_LT(gpuRange, cpuRange * 8.0f)
            << "GPU logit range inflated vs CPU -- garbage distribution";
}

// Single-token decode parity: replay an IDENTICAL token stream one token at a
// time through both engines (KV cache growing on each side) - the exact path
// from Model::generate's loop - and compare each step's top-1 token.
//
// Both engines consume the same fixed stream, so this is an apples-to-apples
// comparison: differences can only come from the GPU engine's own math, never
// from divergent context.
TEST_F(GPUCpuCompareTest, SequentialDecodeArgmaxAgrees) {
    tinycoder::Model *model = SharedTestEnv::model;
    tinycoder::Tokenizer &tokenizer = model->tokenizer();

    if (!tinycoder::gpu::gpuEnabled()) {
        GTEST_SKIP() << "GPU disabled ($TINYCODER_GPU=0) or no CUDA device";
    }

    // A short deterministic token stream (start-of-assistant + a few tokens),
    // exercising the multi-position KV-cache decode path.
    std::string prompt = parisChatPrompt();
    auto tokens = tokenizer.encode(prompt);
    ASSERT_GT(tokens.size(), 1u);

    std::vector<int32_t> stream(tokens.begin(), tokens.begin() + 8);
    std::cout << "GPUCpuCompareTest sequential: " << stream.size()
              << " tokens fed one-by-one" << std::endl;

    // CPU reference session: clear once, then let the KV cache grow.
    setenv("TINYCODER_GPU", "0", 1);
    model->clearKVCache();
    // Per-step CPU top-5 reference (the corruption signal: a broken GPU engine
    // ranks EOG/chat-special tokens first, which never appear in the CPU's top-5).
    std::vector<std::vector<std::pair<float, int32_t>>> cpuTop5;
    std::vector<std::vector<std::pair<float, int32_t>>> cpuTop10;
    cpuTop5.reserve(stream.size());
    cpuTop10.reserve(stream.size());
    for (size_t i = 0; i < stream.size(); ++i) {
        auto logits = lastLogitRow(model->forward({stream[i]},
                                                  /*computeAllLogits=*/false));
        ASSERT_FALSE(logits.empty());
        cpuTop5.push_back(topN(logits, 5));
        cpuTop10.push_back(topN(logits, 10));
        ASSERT_FALSE(cpuTop5[i].empty());
    }

    // GPU session: same stream, own KV cache.
    setenv("TINYCODER_GPU", "1", 1);
    model->clearKVCache();
    for (size_t i = 0; i < stream.size(); ++i) {
        std::string piece = tokenizer.decodeToken(stream[i]);
        std::cout << "\n-- step " << i << " (token " << stream[i] << " '" << piece
                  << "') --" << std::endl;
        auto logits = lastLogitRow(model->forward({stream[i]},
                                                  /*computeAllLogits=*/false));
        ASSERT_FALSE(logits.empty());
        // Logit magnitude/range check at this step (a collapsed GPU range is the
        // signature of a broken GEMV/attention path).
        float gMin = std::numeric_limits<float>::max(),
              gMax = -std::numeric_limits<float>::max();
        for (uint32_t v = 0; v < logits.size(); ++v) {
            gMin = std::min(gMin, logits.get(v));
            gMax = std::max(gMax, logits.get(v));
        }
        std::cout << "  GPU logit range: [" << gMin << ", " << gMax
                  << "] (width " << (gMax - gMin) << ")" << std::endl;
        printTopPairs(tokenizer, "CPU top-10", cpuTop10[i]);
        printTop(tokenizer, "GPU top-10", logits, 10);
        auto gpuTop = topN(logits, 5);
        ASSERT_FALSE(gpuTop.empty());

        // The two engines compute the SAME math with different precision on the
        // decode kernels (GPU streams quantized weights via warp-per-row GEMV vs
        // CPU's fp32 reference).  At position 0 / minimal-context steps the
        // per-vocabulary logits sit on sharp quantization noise, so an adjacent
        // token can flip CPU vs GPU top-1 within the same near-tie cluster.
        //
        // Assertion intent: catch GROSS corruption (the pre-fix symptom) where
        // the GPU ranks EOG/chat-special tokens (151644/151645/151643) first,
        // which never rank in the CPU's top-5.  Tolerate near-tie reorderings by
        // requiring the GPU top-1 token to be a plausible CPU-top-5 candidate.
        bool plausible = false;
        for (const auto &kv: cpuTop5[i]) {
            if (kv.second == gpuTop[0].second) {
                plausible = true;
                break;
            }
        }
        EXPECT_TRUE(plausible)
                << "Step " << i << " (token " << stream[i]
                << "): GPU top-1=" << gpuTop[0].second
                << " not among CPU top-5 (GPU distribution diverged - corruption?)";
    }
}

/// @brief Kernel-level Q8K parity: run the GPU kQuantizeQ8K + kQGemvQ8K
/// kernels on ONE real weight row and a deterministic activation, and compare
/// against the CPU reference (matMulVecFusedQ8K, the path the CPU forward uses).
/// This isolates the Q8K integer-dot kernels from the rest of the forward
/// pass: a mismatch here means the GPU kernel itself diverges from llama's
/// vec_dot_iq*_q8_K; a match here points the blame at the surrounding
/// attention/KV-cache/layer plumbing instead.
///
/// Runs on the first `kRows` rows of layer-0 attnQ (IQ2_S), attnO (IQ3_S),
/// ffnGate (IQ3_XXS) -- the three quant types that use the Q8K decode path.
TEST_F(GPUCpuCompareTest, Q8_0KernelParityWithCPU) {
    if (!tinycoder::gpu::gpuEnabled()) {
        GTEST_SKIP() << "GPU disabled ($TINYCODER_GPU=0) or no CUDA device";
    }
    tinycoder::Model *model = SharedTestEnv::model;
    const auto &layers = model->debugGetLayers();
    ASSERT_FALSE(layers.empty());

    tinycoder::gpu::GPUModel gm;

    // ---- Q8_0 quantizer + float-tree bisection on the REAL L0 attnQKV ----
    // The GPU integer kernel diverges ~5-80% from the CPU AVX2 kernel on
    // attnQKV rows (measured).  Both float trees are identical, so this block
    // (1) diffs the GPU's quantized activation bytes vs a host scalar copy of
    // the CPU quantize_row_q8_0 loop, and (2) recomputes per-block float
    // contributions from each side's bytes to see if the divergence is in the
    // quantization or the float accumulation.
    if (!layers.empty()) {
        const auto &qkv = layers[0].attnQKV;
        if (!qkv.empty() && qkv.type == GGML_TYPE_Q8_0) {
            const uint32_t bpr = (qkv.cols + 31) / 32;
            const uint8_t *row0 = qkv.data.data();
            std::vector<float> xx(qkv.cols);
            for (uint32_t i = 0; i < xx.size(); ++i) {
                const double t = static_cast<double>(i);
                xx[i] = 0.9f * std::sin(0.021 * t) +
                        0.35f * std::cos(0.0037 * t) +
                        0.15f * std::sin(0.1103 * t + 0.7);
            }
            std::vector<float> gpuBlk(bpr);
            std::vector<uint8_t> gpuQuant(static_cast<size_t>(bpr) * 64);
            std::vector<float> gpuTrace(static_cast<size_t>(2) * bpr + 16);
            gm.debugQ8_0Blocks(row0, bpr, xx.data(), qkv.cols, gpuBlk.data(),
                               gpuQuant.data(), gpuTrace.data());
            std::vector<int8_t> cpuQ(qkv.cols);
            std::vector<float> cpuD8(bpr);
            for (uint32_t b = 0; b < bpr; ++b) {
                float amax = 0.0f, maxv = 0.0f;
                for (uint32_t j = 0; j < 32; ++j) {
                    const float ax = std::fabs(xx[b * 32 + j]);
                    if (ax > amax) {
                        amax = ax;
                        maxv = xx[b * 32 + j];
                    }
                }
                if (amax == 0.0f) continue;
                const float iscale = -127.0f / maxv;
                for (uint32_t j = 0; j < 32; ++j) {
                    int v = static_cast<int>(std::lrintf(iscale * xx[b * 32 + j]));
                    cpuQ[b * 32 + j] = static_cast<int8_t>(std::min(127, v));
                }
                cpuD8[b] = 1.0f / iscale;
            }
            // Host recompute of the SAME per-block float contributions, but
            // from the CPU-quantized bytes and using the CPU kernel's exact
            // tree (d = halfToFloat, dd8 = d*d8, t_g = dd8*s_g, pair/hadd).
            std::vector<float> cpuBlk(bpr, 0.0f);
            for (uint32_t b = 0; b < bpr; ++b) {
                const uint16_t dh = *reinterpret_cast<const uint16_t *>(
                        row0 + static_cast<size_t>(b) * 34);
                const float d =
                        tinycoder::GGMLDequantize::halfToFloatBranchFree(dh);
                const float d8 = cpuD8[b];
                if (d8 == 0.0f) continue;
                int s[8];
                for (uint32_t g = 0; g < 8; ++g) {
                    int c = 0;
                    for (uint32_t j = 0; j < 4; ++j) {
                        const int wq = static_cast<int>(static_cast<int8_t>(
                                row0[static_cast<size_t>(b) * 34 + 2u + 4u * g + j]));
                        c += wq * static_cast<int>(cpuQ[b * 32 + 4u * g + j]);
                    }
                    s[g] = c;
                }
                const float scale = d * d8;
                float t[8];
                for (uint32_t g = 0; g < 8; ++g) t[g] = scale * static_cast<float>(s[g]);
                const float u0 = t[0] + t[4];
                const float u1 = t[1] + t[5];
                const float u2 = t[2] + t[6];
                const float u3 = t[3] + t[7];
                const float v0 = u0 + u1;
                const float v1 = u2 + u3;
                cpuBlk[b] = v0 + v1;
            }
            // Diff the KERNEL's OWN cumulative sums (gpuTrace, cumulative) vs
            // the host recompute's cumulative sums (prefix sum of gpuBlk,
            // which is per-block).  gpuTrace[b] and sum(gpuBlk[0..b]) must be
            // bit-identical for every b if the kernel's float tree matches the
            // host tree on the same quantized bytes.
            int firstKernelDiv = -1;
            float hostCum = 0.0f;
            for (uint32_t b = 0; b < bpr; ++b) {
                hostCum += gpuBlk[b];
                if (gpuTrace[b] != hostCum) {
                    firstKernelDiv = static_cast<int>(b);
                    break;
                }
            }
            if (firstKernelDiv >= 0) {
                const uint32_t bb = static_cast<uint32_t>(firstKernelDiv);
                // If block 1 diverges, dump its kernel-side s[8] and blk vs
                // the host's block-1 blk (gpuBlk[1]-gpuBlk[0]) to see whether
                // the integer group sums or the float tree differs.
                std::cout << "[Q8_0 trace] kernel != host recompute at b=" << bb
                          << " kernelCum=" << gpuTrace[bb]
                          << " hostCum=" << gpuBlk[bb]
                          << " blk0=" << gpuTrace[bpr + 8]
                          << " hostblk0=" << gpuBlk[0]
                          << " t0..t7=(";
                for (uint32_t g = 0; g < 8; ++g) {
                    std::cout << (g ? "," : "") << gpuTrace[bpr + g];
                }
                std::cout << ")";
                if (bb == 1 && bpr > 1) {
                    std::cout << " blk1Host="
                              << (gpuBlk[1] - gpuBlk[0])
                              << " blk1Kernel=" << gpuTrace[2 * bpr + 8]
                              << " kerneld1=" << gpuTrace[2 * bpr + 9]
                              << " kerneld81=" << gpuTrace[2 * bpr + 10]
                              << " s1=(";
                    for (uint32_t g = 0; g < 8; ++g) {
                        std::cout << (g ? "," : "") << gpuTrace[2 * bpr + g];
                    }
                    std::cout << ")";
                }
                std::cout << std::endl;
            } else {
                std::cout << "[Q8_0 trace] kernel matches host recompute on all "
                          << bpr << " blocks" << std::endl;
            }
            // Also recompute the host's block-1 s[8] and scale so we can judge
            // whether the kernel's integer group sums or its float tree differ.
            if (bpr > 1) {
                int hs[8];
                for (uint32_t g = 0; g < 8; ++g) {
                    int c = 0;
                    for (uint32_t j = 0; j < 4; ++j) {
                        const int wq = static_cast<int>(static_cast<int8_t>(
                                row0[static_cast<size_t>(1) * 34 + 2u + 4u * g + j]));
                        c += wq * static_cast<int>(cpuQ[32 + 4u * g + j]);
                    }
                    hs[g] = c;
                }
                const uint16_t dh1 = *reinterpret_cast<const uint16_t *>(
                        row0 + static_cast<size_t>(1) * 34);
                const float d1 =
                        tinycoder::GGMLDequantize::halfToFloatBranchFree(dh1);
                // Reconstruct the kernel's blk1 from ITS captured inputs
                // (kerneld1/kerneld81/s1) using the exact float tree, to see
                // which side (kernel=0.116698 or host=0.0547496) is
                // self-consistent with the shared inputs.
                float kscale = gpuTrace[2 * bpr + 9] * gpuTrace[2 * bpr + 10];
                float kt[8];
                for (uint32_t g = 0; g < 8; ++g) {
                    kt[g] = kscale * gpuTrace[2 * bpr + g];
                }
                const float ku0 = kt[0] + kt[4];
                const float ku1 = kt[1] + kt[5];
                const float ku2 = kt[2] + kt[6];
                const float ku3 = kt[3] + kt[7];
                const float kv0 = ku0 + ku1;
                const float kv1 = ku2 + ku3;
                const float kblk = kv0 + kv1;
                std::cout << "[Q8_0 host1] d1=" << d1 << " d81=" << cpuD8[1]
                          << " hs=(";
                for (uint32_t g = 0; g < 8; ++g) {
                    std::cout << (g ? "," : "") << hs[g];
                }
                std::cout << ") treeFromKerInputs=" << kblk << std::endl;
            }
            // Compare block-by-block: GPU kernel per-block (gpuBlk, host
            // recompute from GPU bytes) vs CPU-tree per-block (cpuBlk, from
            // CPU bytes).  Byte-identical quant with identical tree MUST give
            // bit-identical per-block values; any diff localizes the bug.
            float gpuSum = 0.0f, cpuSum = 0.0f;
            int firstBlockDiff = -1;
            for (uint32_t b = 0; b < bpr; ++b) {
                gpuSum += gpuBlk[b];
                cpuSum += cpuBlk[b];
                if (firstBlockDiff < 0 &&
                    gpuBlk[b] != cpuBlk[b]) {
                    firstBlockDiff = static_cast<int>(b);
                }
            }
            std::cout << "[Q8_0 blk] attnQKV row0: gpuSum=" << gpuSum
                      << " cpuSum=" << cpuSum
                      << " delta=" << (gpuSum - cpuSum)
                      << " firstBlockDiff=" << firstBlockDiff;
            if (firstBlockDiff >= 0) {
                const uint32_t b = static_cast<uint32_t>(firstBlockDiff);
                const uint16_t dh = *reinterpret_cast<const uint16_t *>(
                        row0 + static_cast<size_t>(b) * 34);
                const float dCpu =
                        tinycoder::GGMLDequantize::halfToFloatBranchFree(dh);
                // GPU kernel uses __half2float on the block's fp16 d; emulate
                // it on the host via the same CUDA __half2float bit math
                // (IEEE fp16->fp32), using the project's half helpers.
                const float dGpu = tinycoder::GGMLDequantize::halfToFloat(dh);
                const float gpuD8f =
                        *reinterpret_cast<const float *>(gpuQuant.data() +
                                                         static_cast<size_t>(b) * 64);
                auto bits = [](float f) -> uint32_t {
                    uint32_t u;
                    std::memcpy(&u, &f, 4);
                    return u;
                };
                std::cout << " b=" << b << " dh=" << std::hex << dh << std::dec
                          << " dCpuBits=" << std::hex << bits(dCpu)
                          << " dGpuBits=" << bits(dGpu) << std::dec
                          << " cpuD8Bits=" << std::hex << bits(cpuD8[b])
                          << " gpuD8Bits=" << bits(gpuD8f) << std::dec
                          << " gpuBlkBits=" << std::hex << bits(gpuBlk[b])
                          << " cpuBlkBits=" << bits(cpuBlk[b]) << std::dec;
            }
            std::cout << std::endl;
            // CPU byte-exact quantize loop (cpuQ/cpuD8 hoisted above, computed
            // once for both the byte diff and the block-level float compare).
            int byteDiff = 0, firstByteDiff = -1;
            for (uint32_t b = 0; b < bpr; ++b) {
                for (uint32_t j = 0; j < 32; ++j) {
                    const int8_t gb = static_cast<int8_t>(gpuQuant[b * 64 + 4 + j]);
                    if (gb != cpuQ[b * 32 + j]) {
                        ++byteDiff;
                        if (firstByteDiff < 0) firstByteDiff = static_cast<int>(b * 32 + j);
                    }
                }
            }
            std::cout << "[Q8_0 quant] attnQKV row0: GPU byte diffs=" << byteDiff
                      << " first@idx=" << firstByteDiff
                      << " cpuD8[0]=" << cpuD8[0]
                      << " gpuD8[0]="
                      << *reinterpret_cast<const float *>(gpuQuant.data())
                      << std::endl;
            // Dump the first 8 differing (idx, cpuByte, gpuByte, x[idx]) triples.
            int shown = 0;
            for (uint32_t b = 0; b < bpr && shown < 8; ++b) {
                for (uint32_t j = 0; j < 32; ++j) {
                    const int8_t gb = static_cast<int8_t>(gpuQuant[b * 64 + 4 + j]);
                    if (gb != cpuQ[b * 32 + j]) {
                        const uint32_t idx = b * 32 + j;
                        std::cout << "[Q8_0 quant] diff idx=" << idx
                                  << " x=" << xx[idx] << " cpuQ="
                                  << static_cast<int>(cpuQ[b * 32 + j])
                                  << " gpuQ=" << static_cast<int>(gb)
                                  << std::endl;
                        ++shown;
                        if (shown >= 8) break;
                    }
                }
            }
        }
    }

    // Gather the Q8_0 trunk matrices that launchQGemv routes through the
    // integer path (attnQKV is the recurrent trunk's projection; the gate/up
    // MoE experts are also Q8_0 in this model).
    struct MatrixRef {
        const char *name;
        const tinycoder::QuantizedMatrix *m;
    };
    std::vector<MatrixRef> mats;
    auto addIf = [&](const char *name, const tinycoder::QuantizedMatrix &m) {
        if (!m.empty() && m.type == GGML_TYPE_Q8_0) {
            mats.push_back({name, &m});
        }
    };
    if (!layers.empty()) {
        addIf("L0 attnQKV (Q8_0)", layers[0].attnQKV);
        addIf("L0 attnGate (Q8_0)", layers[0].attnGate);
        addIf("L0 attnO (Q8_0)", layers[0].attnO);
        addIf("L0 ffnGate (Q8_0)", layers[0].ffnGate);
        addIf("L0 ffnUp (Q8_0)", layers[0].ffnUp);
        addIf("L0 ffnDown (Q8_0)", layers[0].ffnDown);
        addIf("L0 ffnGateShexp (Q8_0)", layers[0].ffnGateShexp);
        addIf("L0 ffnUpShexp (Q8_0)", layers[0].ffnUpShexp);
        addIf("L0 ffnDownShexp (Q8_0)", layers[0].ffnDownShexp);
        addIf("L0 ffnGateExps (Q8_0)", layers[0].ffnGateExps);
        addIf("L0 ffnUpExps (Q8_0)", layers[0].ffnUpExps);
        addIf("L0 ffnDownExpsMoe (Q8_0)", layers[0].ffnDownExpsMoe);
    }
    if (mats.empty()) {
        GTEST_SKIP() << "Model has no Q8_0 trunk matrices; skipping Q8_0 "
                        "kernel parity regression guard";
    }

    constexpr uint32_t kRows = 64;
    // Deterministic, non-degenerate activation vector (like the Q8K test).
    std::vector<float> x(mats[0].m->cols);
    for (uint32_t i = 0; i < x.size(); ++i) {
        const double t = static_cast<double>(i);
        x[i] = 0.9f * std::sin(0.021 * t) + 0.35f * std::cos(0.0037 * t) +
               0.15f * std::sin(0.1103 * t + 0.7);
    }

    for (const auto &mr: mats) {
        SCOPED_TRACE(mr.name);
        const auto &m = *mr.m;
        const uint32_t blocksPerRow = (m.cols + 31) / 32;
        const uint32_t rowBytes = blocksPerRow * 34;
        for (uint32_t r = 0; r < std::min<uint32_t>(kRows, m.rows); ++r) {
            const uint8_t *rowData = m.data.data() +
                                     static_cast<size_t>(r) * rowBytes;
            // CPU reference: the SAME Q8_0 SIMD the CPU forward uses
            // (matMulVecBatchQ8_0_SIMD dispatches to the AVX2 kernel).
            std::vector<float> cpuOut(1);
            tinycoder::matMulVecBatchQ8_0_SIMD(rowData, x.data(), 1, 1, m.cols,
                                               cpuOut.data());
            // GPU: the exact integer Q8_0 path launchQGemv uses.
            float gpuOut = gm.debugQ8_0Row(rowData, blocksPerRow, x.data(), m.cols);
            const float absErr = std::fabs(gpuOut - cpuOut[0]);
            const float relErr = absErr / (std::fabs(cpuOut[0]) + 1e-3f);
            if (r == 0 || absErr > 1e-5f) {
                std::cout << "[Q8_0 parity] " << mr.name << " row=" << r
                          << ": cpu=" << cpuOut[0] << " gpu=" << gpuOut
                          << " absErr=" << absErr << " relErr=" << relErr
                          << std::endl;
            }
            EXPECT_TRUE(absErr < 1e-5f || relErr < 1e-4f)
                    << "GPU kQGemvQ8_0xQ8K diverges from CPU "
                       "matMulVecBatchQ8_0_SIMD on "
                    << mr.name << " row " << r;
        }
    }
}

TEST_F(GPUCpuCompareTest, Q8KKernelParityWithCPU) {
    if (!tinycoder::gpu::gpuEnabled()) {
        GTEST_SKIP() << "GPU disabled ($TINYCODER_GPU=0) or no CUDA device";
    }
    tinycoder::Model *model = SharedTestEnv::model;
    const auto &layers = model->debugGetLayers();
    ASSERT_FALSE(layers.empty());

    // Standalone GPUModel: only debugQ8KRow is used (no upload() call), and the
    // default-constructed instance's destructor early-returns on !allocated_, so
    // it is safe to stack-allocate.
    tinycoder::gpu::GPUModel gm;
    struct MatrixRef {
        const char *name;
        uint32_t type;
        const tinycoder::AlignedVector<uint8_t> *data;
        uint32_t rows, cols;
    };
    std::vector<MatrixRef> mats;
    if (!layers[0].attnQ.empty() && layers[0].attnQ.type == GGML_TYPE_IQ2_S) {
        mats.push_back({"L0 attnQ (IQ2_S)", layers[0].attnQ.type,
                        &layers[0].attnQ.data, layers[0].attnQ.rows,
                        layers[0].attnQ.cols});
    }
    if (!layers[0].attnO.empty() && layers[0].attnO.type == GGML_TYPE_IQ3_S) {
        mats.push_back({"L0 attnO (IQ3_S)", layers[0].attnO.type,
                        &layers[0].attnO.data, layers[0].attnO.rows,
                        layers[0].attnO.cols});
    }
    if (!layers[0].ffnGate.empty() &&
        layers[0].ffnGate.type == GGML_TYPE_IQ3_XXS) {
        mats.push_back({"L0 ffnGate (IQ3_XXS)", layers[0].ffnGate.type,
                        &layers[0].ffnGate.data, layers[0].ffnGate.rows,
                        layers[0].ffnGate.cols});
    }
    if (mats.empty()) {
        GTEST_SKIP() << "Model lacks the Q8K decode types (expected "
                        "IQ3_XXS-imat layout: attnQ=IQ2_S, attnO=IQ3_S, "
                        "ffnGate=IQ3_XXS); skipping kernel parity "
                        "regression guard";
    }

    constexpr uint32_t kBlock = 256;
    constexpr uint32_t kRows = 48;

    // Deterministic, non-degenerate activation vector (scaled soft-sines with
    // several sign changes; a "plain" spread would mask kernel bugs behind the
    // dominant term).  Same vector for every matrix (each is Q8K-quantized
    // independently on both sides).  Size = the first matrix's input width
    // (hidden size for attnQ/attnO, intermediate for ffnGate).
    std::vector<float> x(mats[0].cols);
    for (uint32_t i = 0; i < x.size(); ++i) {
        const double t = static_cast<double>(i);
        x[i] = 0.9f * std::sin(0.021 * t) + 0.35f * std::cos(0.0037 * t) +
               0.15f * std::sin(0.1103 * t + 0.7);
    }

    for (const auto &m: mats) {
        SCOPED_TRACE(m.name);
        const uint32_t blocksPerRow = (m.cols + kBlock - 1) / kBlock;
        const uint32_t typeSize = m.type == GGML_TYPE_IQ2_S     ? 82
                                  : m.type == GGML_TYPE_IQ3_XXS ? 98
                                                                : 110;
        for (uint32_t r = 0; r < std::min<uint32_t>(kRows, m.rows); ++r) {
            // Weight pointer offset to row r (rows are contiguous in data).
            const uint8_t *rowData = m.data->data() +
                                     static_cast<size_t>(r) * blocksPerRow * typeSize;
            // CPU reference: the same Q8K integer dot the CPU forward uses.
            std::vector<float> cpuOut(1);
            tinycoder::GGMLDequantize::matMulVecFusedQ8K(
                    m.type, rowData, x.data(), 1, m.cols, cpuOut.data());
            // GPU: kQuantizeQ8K on x + kQGemvQ8K on row r.
            float gpuOut =
                    gm.debugQ8KRow(m.type, rowData, blocksPerRow, x.data(), m.cols);
            // Q8K quantization is integer-exact on both sides; only the final
            // float d*bsum rounding can differ (~1-2 ulp of the accumulated
            // value, which is typically O(1)).  Near-zero cpu outputs inflate a
            // purely relative bound, so use an absolute cap of 1e-5 (well above
            // a 2-ulp rounding of a ~50-magnitude sum) plus a loose relative
            // cap for large values.  Any divergence larger than this is a real
            // kernel bug (like the pre-fix IQ3_S grid2 byte-extraction error,
            // which produced relErr 0.3..34).
            const float absErr = std::fabs(gpuOut - cpuOut[0]);
            const float relErr = absErr / (std::fabs(cpuOut[0]) + 1e-3f);
            if (r == 0 || absErr > 1e-5f) {
                std::cout << "[Q8K parity] " << m.name << " row=" << r
                          << ": cpu=" << cpuOut[0] << " gpu=" << gpuOut
                          << " absErr=" << absErr << " relErr=" << relErr
                          << std::endl;
            }
            EXPECT_TRUE(absErr < 1e-5f || relErr < 1e-4f)
                    << "GPU kQGemvQ8K diverges from CPU matMulVecFusedQ8K on "
                    << m.name << " row " << r;
        }
    }
}

#else// !USE_CUDA

TEST_F(GPUCpuCompareTest, ParityTestSkippedWithoutCuda) {
    GTEST_SKIP() << "Built without USE_CUDA -- no GPU engine to compare";
}

#endif// USE_CUDA
