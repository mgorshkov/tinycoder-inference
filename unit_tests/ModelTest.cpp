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
 * TinyCoder Model Unit Test (Google Test)
 *
 * Tests model loading, tokenization, forward pass, KV cache management,
 * and sample question answering.
 *
 * GPU acceleration is used automatically when the library is compiled
 * with ENABLE_CUDA=ON and a CUDA-capable GPU is available.
 *
 * Usage:
 *   TINYCODER_MODEL_PATH=<path_to_model.gguf> ./tinycoder_test
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include "GGUFLoader.hpp"
#include "Model.hpp"
#include "ModelConfig.hpp"
#include "ModelInternal.hpp"
#include "SharedTestEnv.hpp"
#include "Tokenizer.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace fs = std::filesystem;
using namespace tinycoder;

// ---------------------------------------------------------------------------
// Test fixture: uses the model loaded once globally by SharedTestEnv
// ---------------------------------------------------------------------------

class ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_NE(SharedTestEnv::model, nullptr) << "Model not loaded — skipping test";
        ASSERT_TRUE(SharedTestEnv::modelLoaded);
    }
};

// ---------------------------------------------------------------------------
// Test: GGUF Metadata Loading
// ---------------------------------------------------------------------------

TEST(GGUFMetadataTest, LoadMetadata) {
    ASSERT_FALSE(SharedTestEnv::modelPath.empty());
    ASSERT_TRUE(fs::exists(SharedTestEnv::modelPath));

    GGUFLoader loader;
    ASSERT_TRUE(loader.loadMetadata(SharedTestEnv::modelPath));

    const auto &cfg = loader.config();
    EXPECT_GT(cfg.numLayers, 0);
    EXPECT_GT(cfg.hiddenSize, 0);
    EXPECT_GT(cfg.numAttentionHeads, 0);
    EXPECT_GT(cfg.numKVHeads, 0);
    EXPECT_GT(cfg.vocabSize, 0);
    if (cfg.architecture == ARCH_QWEN35 ||
        cfg.architecture == ARCH_QWEN35MOE) {
        // Both the Qwen35 (Qwen3.8) dense and the Qwen35MoE (Qwen3.6-35B-A3B)
        // hybrids declare explicit per-head dims via attention.key_length /
        // attention.value_length (256 for 27B / 35B-A3B).  headDim is the
        // explicit key length, NOT hiddenSize/numAttentionHeads (5120/24 =
        // 213.33 and 2048/16 = 128 would both be wrong).
        EXPECT_EQ(cfg.headDim, cfg.attentionKeyLength);
        EXPECT_EQ(cfg.attentionKeyLength, cfg.attentionValueLength);
    }
    if (cfg.architecture == ARCH_QWEN35) {
        // Qwen35 (dense) hybrid: 48 recurrent + 16 full-attention + 1 MTP.
        EXPECT_EQ(cfg.numLayers - cfg.nextnPredictLayers, 64u);
        EXPECT_EQ(cfg.fullAttentionInterval, 4u);
        EXPECT_EQ(cfg.ropeDimensionCount, 64u);
        EXPECT_EQ(cfg.ssmInnerSize, 6144u);
        EXPECT_EQ(cfg.ssmStateSize, 128u);
        EXPECT_EQ(cfg.ssmGroupCount, 16u);
        EXPECT_EQ(cfg.ssmTimeStepRank, 48u);
        EXPECT_EQ(cfg.ssmConvKernel, 4u);
        // MRoPE sections [t, h, w, e] = [11, 11, 10, 0].
        EXPECT_EQ(cfg.ropeDimensionSections[0], 11u);
        EXPECT_EQ(cfg.ropeDimensionSections[1], 11u);
        EXPECT_EQ(cfg.ropeDimensionSections[2], 10u);
        EXPECT_EQ(cfg.ropeDimensionSections[3], 0u);
    } else if (cfg.architecture != ARCH_QWEN35MOE) {
        EXPECT_EQ(cfg.headDim, cfg.hiddenSize / cfg.numAttentionHeads);
    }
}

// ---------------------------------------------------------------------------
// Test: Memory Estimation
// ---------------------------------------------------------------------------

TEST(GGUFMetadataTest, MemoryEstimation) {
    ASSERT_FALSE(SharedTestEnv::modelPath.empty());

    uint64_t estimatedBytes = Model::estimateMemory(SharedTestEnv::modelPath);
    EXPECT_GT(estimatedBytes, 0);
}

// ---------------------------------------------------------------------------
// Test: Model Loading
// ---------------------------------------------------------------------------

TEST_F(ModelTest, ModelLoaded) {
    ASSERT_NE(SharedTestEnv::model, nullptr);
    ASSERT_TRUE(SharedTestEnv::modelLoaded);

    const auto &config = SharedTestEnv::config;
    EXPECT_GT(config.numLayers, 0);
    EXPECT_GT(config.hiddenSize, 0);
    EXPECT_GT(config.numAttentionHeads, 0);
    EXPECT_GT(config.numKVHeads, 0);
    EXPECT_GT(config.vocabSize, 0);
    EXPECT_GT(config.intermediateSize, 0);
    EXPECT_GT(config.headDim, 0);
    EXPECT_GT(config.ropeTheta, 0.0f);
}

// ---------------------------------------------------------------------------
// Test: Tokenizer Encoding/Decoding
// ---------------------------------------------------------------------------

TEST_F(ModelTest, TokenizerEncodeDecode) {
    auto &tokenizer = SharedTestEnv::model->tokenizer();

    // Test encoding
    std::string testStr = "Hello";
    auto tokens = tokenizer.encode(testStr);
    EXPECT_GT(tokens.size(), 0);

    // Test round-trip
    std::string decoded;
    for (auto t: tokens) {
        decoded += tokenizer.decodeToken(t);
    }
    EXPECT_FALSE(decoded.empty());

    // Test special tokens
    int32_t bos = tokenizer.bosTokenId();
    int32_t eos = tokenizer.eosTokenId();
    EXPECT_GE(bos, 0);
    EXPECT_GE(eos, 0);
}

// ---------------------------------------------------------------------------
// Test: Single Token Forward Pass
// ---------------------------------------------------------------------------

TEST_F(ModelTest, SingleTokenForward) {
    auto &tokenizer = SharedTestEnv::model->tokenizer();
    auto tokens = tokenizer.encode("Hello");
    ASSERT_GT(tokens.size(), 0);

    SharedTestEnv::model->clearKVCache();
    auto logits = SharedTestEnv::model->forward({tokens[0]});

    const auto &config = SharedTestEnv::config;
    EXPECT_EQ(logits.size(), config.vocabSize);

    // Check logits are finite
    bool hasNan = false, hasInf = false;
    for (uint32_t i = 0; i < logits.size(); ++i) {
        if (std::isnan(logits.get(i))) hasNan = true;
        if (std::isinf(logits.get(i))) hasInf = true;
    }
    EXPECT_FALSE(hasNan) << "Logits contain NaN";
    EXPECT_FALSE(hasInf) << "Logits contain Inf";
}

// ---------------------------------------------------------------------------
// Test: Multi-Token Forward Pass
// ---------------------------------------------------------------------------

TEST_F(ModelTest, MultiTokenForward) {
    auto &tokenizer = SharedTestEnv::model->tokenizer();
    auto tokens = tokenizer.encode("Hello world");
    ASSERT_GT(tokens.size(), 1);

    SharedTestEnv::model->clearKVCache();
    auto logits = SharedTestEnv::model->forward(tokens);

    const auto &config = SharedTestEnv::config;
    EXPECT_EQ(logits.size(), tokens.size() * config.vocabSize);

    // Check logits are finite
    bool hasNan = false, hasInf = false;
    for (uint32_t i = 0; i < logits.size(); ++i) {
        if (std::isnan(logits.get(i))) hasNan = true;
        if (std::isinf(logits.get(i))) hasInf = true;
    }
    EXPECT_FALSE(hasNan) << "Logits contain NaN";
    EXPECT_FALSE(hasInf) << "Logits contain Inf";
}

// ---------------------------------------------------------------------------
// Test: KV Cache Management
// ---------------------------------------------------------------------------

TEST_F(ModelTest, KVCacheClear) {
    auto &tokenizer = SharedTestEnv::model->tokenizer();
    auto tokens = tokenizer.encode("Hello");
    ASSERT_GT(tokens.size(), 0);

    // First forward pass
    SharedTestEnv::model->clearKVCache();
    auto logits1 = SharedTestEnv::model->forward({tokens[0]});

    // Second forward pass (should use KV cache)
    auto logits2 = SharedTestEnv::model->forward({tokens[0]});

    // Clear and re-run
    SharedTestEnv::model->clearKVCache();
    auto logits3 = SharedTestEnv::model->forward({tokens[0]});

    // logits1 and logits3 should match (both start fresh)
    const auto &config = SharedTestEnv::config;
    float maxDiff = 0.0f;
    for (uint32_t i = 0; i < config.vocabSize; ++i) {
        float diff = std::abs(logits1.get(i) - logits3.get(i));
        maxDiff = std::max(maxDiff, diff);
    }
    EXPECT_LT(maxDiff, 1e-4f) << "KV cache clear should produce identical results";
}

// ---------------------------------------------------------------------------
// Test: Token Generation (basic)
// ---------------------------------------------------------------------------

TEST_F(ModelTest, GenerateTokens) {
    InferenceParams params;
    params.maxTokens = 10;
    params.temperature = 0.0f;// greedy decoding for reproducibility
    params.seed = 42;

    std::string generatedText;
    int tokenCount = 0;

    SharedTestEnv::model->generate("Hello", params,
                                   [&](int32_t /*token*/, const std::string &text) -> bool {
                                       generatedText += text;
                                       tokenCount++;
                                       return true;
                                   });

    EXPECT_GT(tokenCount, 0);
    EXPECT_FALSE(generatedText.empty());
}

// ---------------------------------------------------------------------------
// Test: Sample question answering (parameterized)
// ---------------------------------------------------------------------------

struct QuestionAnswer {
    std::string question;
    std::string expectedKeyword; // keyword expected in the answer
    std::string forbiddenKeyword;// keyword that must NOT appear (language guard)
    int minTokens;               // minimum expected tokens
};

class SampleQuestionTest
    : public ModelTest,
      public ::testing::WithParamInterface<QuestionAnswer> {
public:
    static bool quickMode;
};
bool SampleQuestionTest::quickMode = false;

TEST_P(SampleQuestionTest, AnswersQuestion) {
    const auto &qa = GetParam();

    // End-to-end generation checks asserting on sampled keywords. The original
    // keywords were validated against the IQ3_XXS Qwen2-0.5B reference model;
    // the Qwen2.5-Coder-7B-Instruct-IQ2_S model was verified (via llama.cpp and
    // the CPU engine) to answer each question with the expected keyword using
    // the same chat template. The assertions below are robust: they check the
    // keyword is PRESENT anywhere in the generated text (not position-locked).
    const auto &cfg = SharedTestEnv::config;
    const bool isGptQwen025B = (cfg.architecture == "qwen2" &&
                                cfg.hiddenSize == 1024 && cfg.numLayers == 24);
    const bool isQwen25Coder7B = (cfg.architecture == "qwen2" &&
                                  cfg.hiddenSize == 3584 &&
                                  cfg.numLayers == 28);
    // 1.5B IQ3_XXS-imat (qwen2.5-coder-1.5b-iq3_xxs-imat.gguf) is the same
    // quant family the keyword assertions were originally validated against
    // (IQ3_XXS); run them here too instead of skipping.
    const bool isQwen25Coder15B = (cfg.architecture == "qwen2" &&
                                   cfg.hiddenSize == 1536 &&
                                   cfg.numLayers == 28);
    // Qwen3.6-27B (qwen35 arch, dense gated-delta-net + full-attention hybrid,
    // hidden 5120, 65 layers) is far more capable than the original 0.5B/1.5B
    // reference models, so the same keyword assertions (which are content-based,
    // not position-locked) carry over; the token floors are trivially met.
    const bool isQwen3527B = (cfg.architecture == ARCH_QWEN35 &&
                              cfg.hiddenSize == 5120 && cfg.numLayers == 65);
    // Qwen3.6-35B-A3B (qwen35moe arch, MoE gated-delta-net hybrid, hidden 2048,
    // 40 layers) is a sparse MoE of comparable intelligence; the same keyword
    // assertions carry over (content-based, not position-locked).
    const bool isQwen35MoE = (cfg.architecture == ARCH_QWEN35MOE &&
                              cfg.hiddenSize == 2048 && cfg.numLayers == 40);
    if (!isGptQwen025B && !isQwen25Coder7B && !isQwen25Coder15B &&
        !isQwen3527B && !isQwen35MoE) {
        GTEST_SKIP() << "generation keyword assertions validated against the "
                        "IQ3_XXS Qwen2-0.5B and Qwen2.5-Coder-7B reference "
                        "models plus the Qwen3.6-27B (qwen35) model; loaded "
                     << cfg.modelName << " (" << cfg.architecture << ", hidden "
                     << cfg.hiddenSize << ", layers " << cfg.numLayers << ")";
    }
    // On the 7B model the sampled output is longer and may include a trailing
    // code block; cap the token budget so the Paris keyword is not cut off.
    // On the 27B model answers can include a full C++ snippet with a preamble,
    // which needs a larger budget than the 64-token cap used for the small
    // reference models.
    const uint32_t maxGenTokens =
            isQwen25Coder7B ? 96u
                            : ((isQwen3527B || isQwen35MoE) ? 160u : 64u);

    // Use formatChat() to build the prompt in an architecture-aware way
    std::string prompt = SharedTestEnv::model->formatChat(
            {{"system", "You are TinyCoder, an AI coding assistant. Be concise."},
             {"user", qa.question}},
            true);

    InferenceParams params;
    params.maxTokens = maxGenTokens;
    // Temperature sampling with repeat penalty.  The 2.57-bit IQ2_S model's
    // greedy argmax collapses onto ubiquitous tokens (e.g. ","); sampling lets
    // the answer emerge (the golden reference run by llama.cpp used sampling).
    params.temperature = 0.7f;
    params.topP = 0.9f;
    params.topK = 40;
    params.repeatPenalty = 1.1f;
    params.seed = 42;

    std::string generatedText;
    int tokenCount = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    SharedTestEnv::model->generate(prompt, params,
                                   [&](int32_t /*token*/, const std::string &text) -> bool {
                                       generatedText += text;
                                       tokenCount++;
                                       return true;
                                   });
    auto t1 = std::chrono::high_resolution_clock::now();
    auto genMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    float tokPerSec = tokenCount > 0 ? (tokenCount / (genMs / 1000.0f)) : 0.0f;

    std::cout << "  Q: " << qa.question << std::endl;
    std::cout << "  Generated " << tokenCount << " tokens in " << genMs << " ms ("
              << tokPerSec << " tok/s)" << std::endl;
    std::cout << "  Output: \"" << generatedText << "\"" << std::endl;
    std::cout << "  (Prefill and generation timing printed by Model::generate above)"
              << std::endl;

    // Check minimum token count.  The floor guards against the pathological
    // case where the model emits only the end-of-turn token with no answer.
    // The heavily-quantized 0.5B/1.5B models ramble, which is why their floors
    // are 5-10 tokens; the 7B model is concise enough to answer completely in a
    // single token (e.g. Q2 -> "Paris" followed immediately by <|im_end|>),
    // which is correct behaviour.  The keyword-presence assertion below remains
    // the substantive correctness check on every model.
    const int minTokensEffective = isQwen25Coder7B ? 1 : qa.minTokens;
    EXPECT_GE(tokenCount, minTokensEffective);

    // Check keyword presence (case-insensitive)
    std::string lowerOutput = generatedText;
    std::string lowerKeyword = qa.expectedKeyword;
    std::transform(lowerOutput.begin(), lowerOutput.end(), lowerOutput.begin(),
                   ::tolower);
    std::transform(lowerKeyword.begin(), lowerKeyword.end(), lowerKeyword.begin(),
                   ::tolower);
    EXPECT_NE(lowerOutput.find(lowerKeyword), std::string::npos)
            << "Expected keyword '" << qa.expectedKeyword << "' not found in output";

    // Language guard: a forbidden keyword (e.g. model answering the C++ question
    // in Python) must NOT appear. Catches the lossy Q2_K re-quant quality
    // regression where Q1 starts emitting "def add(a, b):" instead of C++.
    if (!qa.forbiddenKeyword.empty()) {
        std::string lowerForbidden = qa.forbiddenKeyword;
        std::transform(lowerForbidden.begin(), lowerForbidden.end(),
                       lowerForbidden.begin(), ::tolower);
        EXPECT_EQ(lowerOutput.find(lowerForbidden), std::string::npos)
                << "Output must NOT contain forbidden keyword '"
                << qa.forbiddenKeyword << "'";
    }

    // Clear KV cache between questions
    SharedTestEnv::model->clearKVCache();
}

// Define test questions
// Note: These are end-to-end generation tests with sampling. The model is
// heavily quantized (IQ3_XXS, 3.06 bpw), so output quality is limited.
// Keywords are chosen to match the model's actual sampled output.
// Language-guard keywords: Q1 and Q3 ask for C++ code, so a bare-Python answer
// ("def ...:") is a regression signal (lossy Q2_K re-quant). Q2 (factual) and
// Q4 (Python) have no forbidden keyword.
static const QuestionAnswer fullQuestions[] = {
        {"Write a C++ function to add two numbers.", "return", "def ", 10},
        {"What is the capital of France?", "Paris", "", 5},
        {"Explain what a pointer is in C++.", "address", "def ", 10},
        {"Write a for loop in Python that prints numbers 1 to 5.", "for", "", 10},
};

INSTANTIATE_TEST_SUITE_P(FullQuestions, SampleQuestionTest,
                         ::testing::ValuesIn(fullQuestions));

// ===========================================================================
// Diagnostic test: Process Paris prompt tokens one by one and track how
// results change as the KV cache grows. This tests whether the bug is in
// the FIRST token (position 0) or only appears with a multi-position cache.
// ===========================================================================
TEST_F(ModelTest, TracePrefillTokenByToken) {
    std::string question = "What is the capital of France?";
    std::string prompt = SharedTestEnv::model->formatChat(
            {{"system", "You are TinyCoder, an AI coding assistant. Be concise."},
             {"user", question}},
            true);

    auto &tokenizer = SharedTestEnv::model->tokenizer();
    const auto &config = SharedTestEnv::config;
    uint32_t vocabSize = config.vocabSize;

    auto tokens = tokenizer.encode(prompt);
    std::cout << "Prompt: " << tokens.size() << " tokens" << std::endl;

    // First, compute the "fresh" result for each token alone at position 0
    // This is the same code path tested by CompareLogitsWithReference
    std::cout << "\n=== Fresh single-token results ===" << std::endl;
    std::cout << "(each token processed alone at position 0, should all be correct)" << std::endl;
    for (size_t t = 0; t < std::min(tokens.size(), size_t(5)); ++t) {
        SharedTestEnv::model->clearKVCache();
        auto logits = SharedTestEnv::model->forward({tokens[t]});

        std::vector<std::pair<float, int32_t>> top5;
        for (uint32_t i = 0; i < vocabSize; ++i) {
            top5.emplace_back(logits.get(i), static_cast<int32_t>(i));
        }
        std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });

        std::string text = tokenizer.decodeToken(tokens[t]);
        std::cout << "Token " << t << ": id=" << tokens[t] << " text=\"";
        for (char c: text) {
            if (c >= 32 && c < 127) std::cout << c;
            else
                std::cout << "\\x" << std::hex << (unsigned) (unsigned char) c << std::dec;
        }
        std::cout << "\"" << std::endl;
        std::cout << "  top-1: id=" << top5[0].second
                  << " logit=" << top5[0].first
                  << " text=\"" << tokenizer.decodeToken(top5[0].second) << "\"" << std::endl;
    }

    // Now process tokens sequentially and check how results change
    std::cout << "\n=== Sequential processing ===" << std::endl;
    std::cout << "(building KV cache incrementally - tests multi-position cache code path)" << std::endl;
    SharedTestEnv::model->clearKVCache();

    for (size_t t = 0; t < std::min(tokens.size(), size_t(5)); ++t) {
        auto logits = SharedTestEnv::model->forward({tokens[t]});

        std::vector<std::pair<float, int32_t>> top5;
        for (uint32_t i = 0; i < vocabSize; ++i) {
            top5.emplace_back(logits.get(i), static_cast<int32_t>(i));
        }
        std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });

        std::string text = tokenizer.decodeToken(tokens[t]);
        std::cout << "Token " << t << ": id=" << tokens[t] << " text=\"";
        for (char c: text) {
            if (c >= 32 && c < 127) std::cout << c;
            else
                std::cout << "\\x" << std::hex << (unsigned) (unsigned char) c << std::dec;
        }
        std::cout << "\"" << std::endl;
        std::cout << "  top-1: id=" << top5[0].second
                  << " logit=" << top5[0].first
                  << " text=\"" << tokenizer.decodeToken(top5[0].second) << "\"" << std::endl;
    }
}

// ===========================================================================
// DIAGNOSTIC (temp): dump TinyCoder's per-layer post-FFN hidden norms for the
// Paris chat prompt, to diff against llama.cpp's `[ar] per-layer` capture and
// find which layer first diverges.  Runs only on the 1.5B IQ3_XXS-imat model.
// Single token (151644) at position 0 to match llama's AR path captures.
// ===========================================================================
TEST_F(ModelTest, DumpLayerHiddenNormsQwen2) {
    const auto &cfg = SharedTestEnv::config;
    const bool is15BIQ3 = (cfg.architecture == "qwen2" && cfg.hiddenSize == 1536 &&
                           cfg.numLayers == 28);
    if (!is15BIQ3) {
        GTEST_SKIP() << "only for 1.5B IQ3_XXS-imat";
    }
    uint32_t hiddenSize = cfg.hiddenSize;
    auto &tok = SharedTestEnv::model->tokenizer();
    uint32_t vocabSize = cfg.vocabSize;

    // Matches llama_ref_probe's fnv1a64 exactly: one FNV round per float,
    // hashing the 32-bit word (memcpy to avoid aliasing/padding issues).
    auto fnv1a = [](const float *v, size_t n) {
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i < n; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &v[i], sizeof(bits));
            h ^= bits;
            h *= 1099511628211ULL;
        }
        return h;
    };

    const int32_t tokId = 151644;

    // 1) Token embedding row — vs llama tok[0] fnv=28c708690e356a43.
    {
        auto emb = SharedTestEnv::model->debugGetEmbedding(tokId);
        double s = 0.0;
        for (float v: emb) s += (double) v * v;
        printf("  [embedding] norm=%12.6f rms=%.6f first8=[% .7f % .7f % .7f % .7f % .7f % .7f % .7f % .7f] fnv=%016llx\n",
               std::sqrt(s), std::sqrt(s / hiddenSize),
               emb[0], emb[1], emb[2], emb[3], emb[4], emb[5], emb[6], emb[7],
               (unsigned long long) fnv1a(emb.data(), emb.size()));
    }

    // 2) Per-layer post-FFN hidden states (before final RMSNorm) — vs llama's
    //    [ar] per-layer capture (layer 0 fnv=ab99f24a4f92b8eb ...).
    {
        SharedTestEnv::model->clearKVCache();
        auto perLayer = SharedTestEnv::model->forwardWithPerLayerStates({tokId});
        ASSERT_EQ(perLayer.size(), cfg.numLayers);
        printf("  [per-layer] post-FFN hidden (layer / norm / fnv / first8):\n");
        for (uint32_t l = 0; l < cfg.numLayers; ++l) {
            const auto &v = perLayer[l];
            double s = 0.0;
            for (float x: v) s += (double) x * x;
            printf("    layer %3u: norm=%12.6f fnv=%016llx first8=[% .6f % .6f % .6f % .6f % .6f % .6f % .6f % .6f]\n",
                   l, std::sqrt(s), (unsigned long long) fnv1a(v.data(), v.size()),
                   v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
        }
    }

    // 2b) Layer-0 intermediates — vs llama's [ar] layer-0 captures
    //     (attn_norm-0 fnv=fb052b95b11864c0, Qcur-0 fnv=5d7c06f08573c920,
    //      Kcur-0 fnv=524e15b66b171ffd, Vcur-0 fnv=6de4b04fc4456dbf,
    //      ffn_inp-0 fnv=346376922a81d368, ffn_norm-0 fnv=0f49e373692481ef,
    //      ffn_out-0 fnv=7d6b1dfefaf8d39c, l_out-0 fnv=ab99f24a4f92b8eb).
    {
        SharedTestEnv::model->clearKVCache();
        auto intm = SharedTestEnv::model->debugQwen2Layer0Intm({tokId});
        printf("  [layer0] intermediates (name / n / norm / fnv / first4):\n");
        for (const auto &kv: intm) {
            const auto &v = kv.second;
            double s = 0.0;
            for (float x: v) s += (double) x * x;
            printf("    %-16s n=%-6zu norm=%12.6f fnv=%016llx first4=[% .6f % .6f % .6f % .6f]\n",
                   kv.first.c_str(), v.size(), std::sqrt(s),
                   (unsigned long long) fnv1a(v.data(), v.size()),
                   v[0], v.size() > 1 ? v[1] : 0.0f,
                   v.size() > 2 ? v[2] : 0.0f, v.size() > 3 ? v[3] : 0.0f);
        }
    }

    // 3) Final hidden (after final RMSNorm) + top-5 logits — vs llama's [ar]
    //    result_norm fnv=b93fe705ddebe39d / post-output_norm embedding.
    {
        SharedTestEnv::model->clearKVCache();
        auto [hidden, logits] = SharedTestEnv::model->debugForwardWithHidden(tokId);
        double ssq = 0.0;
        for (float v: hidden) ssq += (double) v * v;
        printf("  [final] norm=%12.6f rms=%.6f first8=[% .7f % .7f % .7f % .7f % .7f % .7f % .7f % .7f] fnv=%016llx\n",
               std::sqrt(ssq), std::sqrt(ssq / hiddenSize), hidden[0], hidden[1],
               hidden[2], hidden[3], hidden[4], hidden[5], hidden[6], hidden[7],
               (unsigned long long) fnv1a(hidden.data(), hidden.size()));
        std::vector<std::pair<float, int32_t>> top5;
        for (uint32_t i = 0; i < vocabSize; ++i) {
            top5.emplace_back(logits[i], static_cast<int32_t>(i));
        }
        std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });
        for (int r = 0; r < 5; ++r) {
            printf("  top%d: id=%d logit=%12.6f text=\"%s\"\n", r + 1,
                   top5[r].second, top5[r].first,
                   tok.decodeToken(top5[r].second).c_str());
        }
    }
}

// ===========================================================================
// Diagnostic test: Compare batch prefill vs sequential prefill for the
// full Paris prompt (56 tokens). If they produce the same (wrong) result,
// the bug is not in batch-specific code paths.
// ===========================================================================
TEST_F(ModelTest, CompareBatchVsSequentialPrefill) {
    std::string question = "What is the capital of France?";
    std::string prompt = SharedTestEnv::model->formatChat(
            {{"system", "You are TinyCoder, an AI coding assistant. Be concise."},
             {"user", question}},
            true);

    auto &tokenizer = SharedTestEnv::model->tokenizer();
    const auto &config = SharedTestEnv::config;
    uint32_t vocabSize = config.vocabSize;

    // Tokenize prompt
    auto tokens = tokenizer.encode(prompt);
    ASSERT_GE(tokens.size(), 2) << "Prompt must have at least 2 tokens";
    std::cout << "Prompt: " << tokens.size() << " tokens" << std::endl;

    // ---- Method 1: Batch prefill (forward with all tokens) ----
    SharedTestEnv::model->clearKVCache();
    auto batchLogitsAll = SharedTestEnv::model->forward(tokens);
    ASSERT_FALSE(batchLogitsAll.empty());

    // Extract last token logits
    uint32_t lastIdx = static_cast<uint32_t>(tokens.size()) - 1;
    np::Array<float> batchLastLogits(np::Shape{vocabSize});
    for (uint32_t i = 0; i < vocabSize; ++i) {
        batchLastLogits.set(i, batchLogitsAll.get(lastIdx * vocabSize + i));
    }

    // Batch top-5
    float batchMax = -std::numeric_limits<float>::max();
    float batchMin = std::numeric_limits<float>::max();
    std::vector<std::pair<float, int32_t>> batchTop5;
    for (uint32_t i = 0; i < vocabSize; ++i) {
        float v = batchLastLogits.get(i);
        batchMax = std::max(batchMax, v);
        batchMin = std::min(batchMin, v);
        batchTop5.emplace_back(v, static_cast<int32_t>(i));
    }
    std::partial_sort(batchTop5.begin(), batchTop5.begin() + 5, batchTop5.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    std::cout << "\nBatch prefill logits: min=" << batchMin << " max=" << batchMax << std::endl;
    std::cout << "Batch top-5:" << std::endl;
    for (int r = 0; r < 5; ++r) {
        std::string dt = tokenizer.decodeToken(batchTop5[r].second);
        std::cout << "  [" << r << "] id=" << batchTop5[r].second
                  << " logit=" << batchTop5[r].first
                  << " text=\"" << dt << "\"" << std::endl;
    }

    // ---- Method 2: Sequential (token-by-token, building KV cache) ----
    SharedTestEnv::model->clearKVCache();

    // Fill cache with all but the last token
    for (size_t t = 0; t + 1 < tokens.size(); ++t) {
        SharedTestEnv::model->forward({tokens[t]});
    }

    // Now process the last token (KV cache has positions 0..n-2 filled)
    auto seqLogitsAll = SharedTestEnv::model->forward({tokens.back()});
    ASSERT_FALSE(seqLogitsAll.empty());

    // Extract logits (seqLen=1, so position 0)
    np::Array<float> seqLastLogits(np::Shape{vocabSize});
    for (uint32_t i = 0; i < vocabSize; ++i) {
        seqLastLogits.set(i, seqLogitsAll.get(i));
    }

    // Sequential top-5
    float seqMax = -std::numeric_limits<float>::max();
    float seqMin = std::numeric_limits<float>::max();
    std::vector<std::pair<float, int32_t>> seqTop5;
    for (uint32_t i = 0; i < vocabSize; ++i) {
        float v = seqLastLogits.get(i);
        seqMax = std::max(seqMax, v);
        seqMin = std::min(seqMin, v);
        seqTop5.emplace_back(v, static_cast<int32_t>(i));
    }
    std::partial_sort(seqTop5.begin(), seqTop5.begin() + 5, seqTop5.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
    std::cout << "\nSequential prefill logits: min=" << seqMin << " max=" << seqMax << std::endl;
    std::cout << "Sequential top-5:" << std::endl;
    for (int r = 0; r < 5; ++r) {
        std::string dt = tokenizer.decodeToken(seqTop5[r].second);
        std::cout << "  [" << r << "] id=" << seqTop5[r].second
                  << " logit=" << seqTop5[r].first
                  << " text=\"" << dt << "\"" << std::endl;
    }

    // ---- Compare batch vs sequential ----
    float maxDiff = 0.0f;
    int diffCount = 0;
    float avgDiff = 0.0f;
    for (uint32_t i = 0; i < vocabSize; ++i) {
        float diff = std::abs(batchLastLogits.get(i) - seqLastLogits.get(i));
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
        if (diff > 1e-4f) diffCount++;
    }
    avgDiff /= static_cast<float>(vocabSize);

    std::cout << "\n=== Batch vs Sequential Comparison ===" << std::endl;
    std::cout << "  maxDiff=" << maxDiff << " avgDiff=" << avgDiff
              << " mismatches (>1e-4): " << diffCount << "/" << vocabSize << std::endl;

    // Top-1 token match check
    bool top1Match = (batchTop5[0].second == seqTop5[0].second);
    std::cout << "  Top-1 token match: " << (top1Match ? "YES" : "NO") << std::endl;
    if (!top1Match) {
        std::cout << "  Batch top-1: id=" << batchTop5[0].second
                  << " (" << tokenizer.decodeToken(batchTop5[0].second) << ")"
                  << " logit=" << batchTop5[0].first << std::endl;
        std::cout << "  Sequential top-1: id=" << seqTop5[0].second
                  << " (" << tokenizer.decodeToken(seqTop5[0].second) << ")"
                  << " logit=" << seqTop5[0].first << std::endl;
    }

    if (maxDiff < 1e-3f) {
        std::cout << "\nBatch and Sequential produce IDENTICAL results." << std::endl;
        // Reference value check is model-specific; only check for Qwen2 0.5B
        if (config.architecture == "qwen2" && config.hiddenSize == 1024 && config.numLayers == 24) {
            std::cout << "Comparing against reference (prefill):" << std::endl;
            std::cout << "  Reference top-1: id=785 (\"The\") logit=22.8516" << std::endl;
            std::cout << "  Our top-1: id=" << batchTop5[0].second
                      << " logit=" << batchTop5[0].first << std::endl;
            EXPECT_NEAR(batchTop5[0].first, 22.8516f, 5.0f)
                    << "Top-1 logit far from reference (22.8516).";
        } else {
            std::cout << "\n(Reference value check skipped for model: "
                      << config.modelName << " [" << config.architecture << "])" << std::endl;
        }
    } else {
        std::cout << "\n*** BATCH AND SEQUENTIAL DIFFER! ***" << std::endl;
        std::cout << "maxDiff=" << maxDiff << std::endl;
    }

    // Numeric gate: batch and sequential run identical deterministic kernels, so
    // a large diff signals a routing/math bug (e.g. a scalar-broadcast bsum
    // compensation counted 8x). top-1 must match exactly (Q3_K precision keeps
    // the coherence guard meaningful).
    EXPECT_LT(maxDiff, 5.0f)
            << "Batch prefill differs significantly from sequential! maxDiff=" << maxDiff;
    EXPECT_TRUE(top1Match)
            << "Batch prefill produces different top-1 token than sequential!";
}

// ===========================================================================
// Qwen35 (Qwen3.8-27B) architecture tests. Only run when the loaded model is
// the qwen35 architecture (gated delta net + full attention hybrid).
// ===========================================================================

class Qwen35Test : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_NE(SharedTestEnv::model, nullptr) << "Model not loaded — skipping test";
        ASSERT_TRUE(SharedTestEnv::modelLoaded);
        // qwen35 (dense) and qwen35moe share the identical gated-delta-net +
        // full-attention hybrid; BOTH run these GDN tests.
        const auto &arch = SharedTestEnv::config.architecture;
        if (arch != ARCH_QWEN35 && arch != ARCH_QWEN35MOE) {
            GTEST_SKIP() << "Not a Qwen35/qwen35moe model; skipping qwen35-specific test";
        }
    }

    // True if the test should skip (non-qwen35/qwen35moe model).
    bool skip() const {
        const auto &arch = SharedTestEnv::config.architecture;
        return arch != ARCH_QWEN35 && arch != ARCH_QWEN35MOE;
    }
};

// ---------------------------------------------------------------------------
// Test: recurrent layer classification matches the query (i+1)%4 != 0 pattern.
// ---------------------------------------------------------------------------
TEST_F(Qwen35Test, RecurrentLayerClassification) {
    if (skip()) return;
    const auto &cfg = SharedTestEnv::config;
    const uint32_t nLayer = cfg.numLayers - cfg.nextnPredictLayers;// 64
    for (uint32_t i = 0; i < nLayer; ++i) {
        bool expected = ((i + 1) % cfg.fullAttentionInterval) != 0;
        EXPECT_EQ(detail::isQwen35RecurrentLayer(cfg, i), expected)
                << "layer " << i << " classification mismatch";
    }
    // MTP block is always full-attention.
    for (uint32_t i = nLayer; i < cfg.numLayers; ++i) {
        EXPECT_FALSE(detail::isQwen35RecurrentLayer(cfg, i)) << "MTP layer " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: a fresh (zero-init) forward pass produces finite, well-formed logits
// and the top-1 token is a real vocab entry. This exercises the full qwen35
// path (recurrent + full-attention + FFN + final norm + LM head) without NaN
// or inf from the gated delta net recurrence or MRoPE.
// ---------------------------------------------------------------------------
TEST_F(Qwen35Test, SingleTokenForwardFinite) {
    if (skip()) return;
    SharedTestEnv::model->clearKVCache();
    auto logits = SharedTestEnv::model->forward({0});
    ASSERT_FALSE(logits.empty());

    const auto &config = SharedTestEnv::config;
    const size_t vocab = config.vocabSize;
    // The logits buffer must cover the full LM head width (the tokenizer vocab
    // from GGUF), so the top-1 search below can see every candidate.
    ASSERT_GE(logits.size(), vocab) << "logits shorter than vocab";
    ASSERT_GT(vocab, 0u);

    float maxAbs = 0.0f;
    float mean = 0.0f;
    size_t nanCount = 0;
    for (size_t i = 0; i < vocab; ++i) {
        float v = logits.get(i);
        if (std::isnan(v) || std::isinf(v)) {
            ++nanCount;
            continue;
        }
        maxAbs = std::max(maxAbs, std::fabs(v));
        mean += v;
    }
    EXPECT_EQ(nanCount, 0u) << "NaN/Inf logits in qwen35 forward pass";
    mean /= static_cast<float>(vocab);
    // Raw (pre-softmax) logits have no sign invariant: a healthy distribution
    // just needs finite values with a nonzero dynamic range.
    EXPECT_GT(maxAbs, 0.0f) << "All logits are zero — forward pass is degenerate";
    EXPECT_GT(std::fabs(mean), 0.0f) << "Logits look degenerate (mean==0)";

    // Sanity: the argmax should be a plausible token (not a pad id like 0).
    int32_t argmax = 0;
    float best = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab; ++i) {
        float v = logits.get(i);
        if (v > best) {
            best = v;
            argmax = static_cast<int32_t>(i);
        }
    }
    std::cout << "Qwen35 top-1: id=" << argmax << " logit=" << best
              << " text=\"" << SharedTestEnv::model->tokenizer().decodeToken(argmax)
              << "\" maxAbs=" << maxAbs << " mean=" << mean << std::endl;
    EXPECT_GT(argmax, 0);
}

// ---------------------------------------------------------------------------
// Test: batch prefill vs sequential prefill produce coherent (matching top-1)
// results through the mixed recurrent/full-attention stack.
// ---------------------------------------------------------------------------
TEST_F(Qwen35Test, BatchVsSequentialCoherent) {
    if (skip()) return;
    auto &tokenizer = SharedTestEnv::model->tokenizer();
    const auto &config = SharedTestEnv::config;
    const size_t vocab = config.vocabSize;

    // Prompt long enough to cover >1 BPE token in the qwen35 vocab (the word
    // "Hello" alone is a single token). Kept short for speed on the 27B model.
    std::string prompt = "Hello world, how are you today?";
    auto tokens = tokenizer.encode(prompt);
    ASSERT_GE(tokens.size(), 2) << "prompt did not tokenize to >=2 tokens";

    // Batch prefill.
    SharedTestEnv::model->clearKVCache();
    auto batchAll = SharedTestEnv::model->forward(tokens);
    ASSERT_FALSE(batchAll.empty());

    // Sequential prefill (token-by-token).
    SharedTestEnv::model->clearKVCache();
    for (size_t t = 0; t + 1 < tokens.size(); ++t) {
        SharedTestEnv::model->forward({tokens[t]});
    }
    auto seqAll = SharedTestEnv::model->forward({tokens.back()});
    ASSERT_FALSE(seqAll.empty());

    uint32_t lastIdx = static_cast<uint32_t>(tokens.size()) - 1;
    int32_t batchTop = -1, seqTop = -1;
    float batchBest = -std::numeric_limits<float>::infinity();
    float seqBest = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab; ++i) {
        float b = batchAll.get(lastIdx * vocab + i);
        float s = seqAll.get(i);
        if (b > batchBest) {
            batchBest = b;
            batchTop = static_cast<int32_t>(i);
        }
        if (s > seqBest) {
            seqBest = s;
            seqTop = static_cast<int32_t>(i);
        }
    }
    std::cout << "Qwen35 batch top-1: id=" << batchTop << " logit=" << batchBest
              << " text=\"" << tokenizer.decodeToken(batchTop) << "\"" << std::endl;
    std::cout << "Qwen35 seq   top-1: id=" << seqTop << " logit=" << seqBest
              << " text=\"" << tokenizer.decodeToken(seqTop) << "\"" << std::endl;

    // The recurrence is exact and deterministic in both paths, so the top-1
    // must agree; allow a relaxed tolerance on the score itself (quantized
    // weights, different accumulation order for the recurrent state).
    EXPECT_TRUE(std::isfinite(batchBest) && std::isfinite(seqBest));
    EXPECT_EQ(batchTop, seqTop)
            << "Batch and sequential prefill disagree on top-1 for qwen35";
}

// SharedTestEnv static members are defined in SharedTestEnv.cpp

// ---------------------------------------------------------------------------
// Custom main: reads TINYCODER_MODEL_PATH from environment, registers the
// shared test environment, and runs all tests.
// ---------------------------------------------------------------------------

// Main function moved to ModelTestMain.cpp
