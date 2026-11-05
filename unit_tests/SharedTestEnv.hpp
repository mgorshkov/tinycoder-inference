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

#pragma once

#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "Model.hpp"
#include "ModelConfig.hpp"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// SharedTestEnv: Google Test global environment that loads the model once
// for ALL test suites. Registered in main() via ::testing::AddGlobalTestEnvironment().
// ---------------------------------------------------------------------------

class SharedTestEnv : public ::testing::Environment {
public:
    static std::string modelPath;
    static tinycoder::Model *model;
    static bool modelLoaded;
    static tinycoder::ModelConfig config;
    // Set when the bench-parity page-cache warmup ran in SetUp().  Lets the
    // question tests annotate their reported tok/s with the actual cache
    // state (warm vs cold) instead of implying a fixed state.
    static bool warmedUp;
    // Diagnostic explaining why the shared model is unavailable (missing path or
    // a load failure).  Empty when the model loaded.  Model-dependent fixtures
    // surface this via GTEST_SKIP() instead of failing the whole run.
    static std::string loadError;

    void SetUp() override {
        // The shared model is an OPTIONAL convenience for the model-dependent
        // suites.  It must NOT be loaded with a fatal assertion: a fatal
        // failure here makes Google Test skip EVERY test in the binary
        // (gtest's TestSuite::Run treats a global-environment failure as
        // skip_all), including the self-contained suites that need no model
        // at all (e.g. EXL3CodecTest, EXL3SafetensorsTest, EXL3LoaderTest).
        // Instead, record the reason and let the model-dependent fixtures call
        // GTEST_SKIP() themselves.
        if (modelPath.empty()) {
            loadError = "Model path must be provided via TINYCODER_MODEL_PATH env "
                        "var or --model-path argument";
            std::cerr << "[SharedTestEnv] " << loadError
                      << "; model-dependent tests will be skipped." << std::endl;
            modelLoaded = false;
            return;
        }
        if (!fs::exists(modelPath)) {
            loadError = "Model not found: " + modelPath;
            std::cerr << "[SharedTestEnv] " << loadError
                      << "; model-dependent tests will be skipped." << std::endl;
            modelLoaded = false;
            return;
        }

        model = new tinycoder::Model();
        std::string err;
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = model->load(modelPath, &err);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto loadMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        if (!ok) {
            // Non-fatal: record the reason so the model-dependent fixtures can
            // GTEST_SKIP() while the self-contained suites still run.
            loadError = "Model load failed: " + err;
            std::cerr << "[SharedTestEnv] " << loadError
                      << "; model-dependent tests will be skipped." << std::endl;
            delete model;
            model = nullptr;
            modelLoaded = false;
        } else {
            loadError.clear();
            modelLoaded = true;
            config = model->config();
            std::cout << "Model loaded in " << loadMs << " ms" << std::endl;
            std::cout << "  Model: \"" << config.modelName << "\" ("
                      << config.architecture << ")" << std::endl;
            std::cout << "  Config: " << config.numLayers << " layers, "
                      << config.hiddenSize << " hidden, "
                      << config.numAttentionHeads << " heads, "
                      << config.numKVHeads << " KV heads, "
                      << config.vocabSize << " vocab, "
                      << config.intermediateSize << " intermediate, "
                      << config.headDim << " headDim, "
                      << "ropeTheta=" << config.ropeTheta << std::endl;
            std::cout << "  Tokenizer: " << model->tokenizer().vocabSize()
                      << " tokens" << std::endl;

            // ---- Bench-parity page-cache warmup (2026-09-15) ----
            // tinycoder_bench measures WARM (the reference benchmark tgen protocol: a
            // warmup prefill+decode rep runs before the timed reps), while this
            // gtest previously measured COLD first-touch of the weight pages.
            // On a 35 GB qwen35moe model in a RAM-constrained host the cold/warm
            // delta is large (measured: 0.4 vs 9.5 pp tok/s, 0.5 vs 7.3 tg
            // tok/s), which made the two harnesses report wildly different
            // numbers for the SAME engine.  To make the test's reported tok/s
            // comparable to the bench's, run the same deterministic
            // page-cache warmup the bench uses (64-token prompt prefill + 8
            // greedy decode tokens) once here.  The 35 GB model cannot stay
            // fully resident, so this warms the test's own token stream — the
            // same self-selection the bench gets from its repeated reps.
            // Opt out with TINYCODER_TEST_NO_WARMUP=1 (e.g. small models where
            // the warmup is unnecessary and slows the suite).
            warmedUp = false;
            const bool isBigMoE =
                    (config.architecture == tinycoder::ARCH_QWEN35MOE &&
                     config.expertCount > 0);
            const char *noWarmup = std::getenv("TINYCODER_TEST_NO_WARMUP");
            if (isBigMoE && (noWarmup == nullptr || noWarmup[0] == '\0' ||
                             noWarmup[0] == '0')) {
                // Same deterministic prompt as tinycoder_bench::buildPrompt.
                std::vector<int32_t> prompt;
                prompt.reserve(64);
                uint64_t seed = 0x5EED123456789ULL;
                for (int32_t i = 0; i < 64; ++i) {
                    seed = seed * 6364136223846793005ULL +
                           1442695040888963407ULL;
                    prompt.push_back(static_cast<int32_t>(seed % 3000) + 10);
                }
                auto w0 = std::chrono::high_resolution_clock::now();
                model->clearKVCache();
                auto prefillLogits = model->forward(prompt, false);
                auto w1 = std::chrono::high_resolution_clock::now();
                auto prefillMs = std::chrono::duration_cast<
                                         std::chrono::milliseconds>(w1 - w0)
                                         .count();
                // 8 greedy decode tokens (same as the bench's decode region).
                const int32_t vocab = static_cast<int32_t>(config.vocabSize);
                auto d0 = std::chrono::high_resolution_clock::now();
                int32_t tok = 0;
                if (prefillLogits.size() >= static_cast<size_t>(vocab)) {
                    float best = prefillLogits.get(0);
                    for (int32_t i = 1; i < vocab; ++i) {
                        if (prefillLogits.get(static_cast<size_t>(i)) > best) {
                            best = prefillLogits.get(static_cast<size_t>(i));
                            tok = i;
                        }
                    }
                }
                int32_t gen = 0;
                for (int32_t i = 0; i < 8; ++i) {
                    auto logits = model->forward({tok}, false);
                    if (logits.size() < static_cast<size_t>(vocab)) break;
                    float best = logits.get(0);
                    int32_t bt = 0;
                    for (int32_t j = 1; j < vocab; ++j) {
                        if (logits.get(static_cast<size_t>(j)) > best) {
                            best = logits.get(static_cast<size_t>(j));
                            bt = j;
                        }
                    }
                    tok = bt;
                    ++gen;
                }
                auto d1 = std::chrono::high_resolution_clock::now();
                auto decodeMs = std::chrono::duration_cast<
                                        std::chrono::milliseconds>(d1 - d0)
                                        .count();
                std::cout << "  [bench-parity] page-cache warmup: prefill 64 tok"
                          << " in " << prefillMs << " ms ("
                          << (64.0 / (prefillMs / 1000.0)) << " pp tok/s); "
                          << gen << " tg tok in " << decodeMs << " ms ("
                          << (gen / (decodeMs / 1000.0)) << " tg tok/s)"
                          << std::endl;
                model->clearKVCache();
                warmedUp = true;
            } else {
                std::cout << "  [bench-parity] page-cache warmup skipped"
                          << " (small/dense model or TINYCODER_TEST_NO_WARMUP)"
                          << std::endl;
            }
        }
    }

    void TearDown() override {
        delete model;
        model = nullptr;
        modelLoaded = false;
    }
};
