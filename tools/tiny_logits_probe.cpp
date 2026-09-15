// tiny_logits_probe.cpp
//
// Dump TinyCoder's per-token logits for a prompt, apples-to-apples with
// llama_ref_probe's "[ar] final logits (last prompt token) top-10" block.
// Loads the FULL model (like the unit tests) so this is slow (~5-6 min on
// the 35B-A3B Q8_0) — run with TINYCODER_GPU=0/1 to switch engines.
//
// Build:
//   g++ -O2 -std=c++17 tools/tiny_logits_probe.cpp -I include -Isrc/cpp/core \
//       build/libtinycoder_core.a -lpthread -o /tmp/tiny_logits_probe
//
// Run:
//   TINYCODER_MODEL_PATH=<model.gguf> tiny_logits_probe \
//       [--file <prompt-file> | "prompt text"] [--tokens] [--generate]
//   or: tiny_logits_probe <model.gguf> --file <prompt-file> [--generate]
//   (default prompt: the Q1 wrapped prompt — the exact bytes tinycoder feeds
//    generate(); --tokens dumps the full tokenization; --generate runs the
//    SampleQuestionTest generation path — formatChat + generate() with the
//    test's exact InferenceParams — and dumps the sampled token stream)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Model.hpp"
#include "ModelConfig.hpp"

using namespace tinycoder;

static std::string readFile(const char *path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

int main(int argc, char **argv) {
    const char *modelPath = std::getenv("TINYCODER_MODEL_PATH");
    std::string prompt;
    std::string promptLabel;
    bool dumpTokens = false;
    bool runGenerate = false;
    int32_t genMaxTokens = 160;
    uint32_t genSeed = 42;

    int i = 1;
    while (i < argc) {
        const char *a = argv[i];
        if (strcmp(a, "--file") == 0 && i + 1 < argc) {
            prompt = readFile(argv[i + 1]);
            promptLabel = std::string("file:") + argv[i + 1];
            i += 2;
        } else if (strcmp(a, "--tokens") == 0) {
            dumpTokens = true;
            ++i;
        } else if (strcmp(a, "--generate") == 0) {
            runGenerate = true;
            ++i;
        } else if (strcmp(a, "--max-tokens") == 0 && i + 1 < argc) {
            genMaxTokens = atoi(argv[i + 1]);
            i += 2;
        } else if (strcmp(a, "--seed") == 0 && i + 1 < argc) {
            genSeed = (uint32_t) strtoul(argv[i + 1], nullptr, 10);
            i += 2;
        } else if (a[0] != '-') {
            if (modelPath == nullptr) {
                modelPath = a;
            } else if (promptLabel.empty()) {
                prompt = a;
                promptLabel = "arg";
            } else {
                fprintf(stderr, "unexpected positional arg: %s\n", a);
                return 1;
            }
            ++i;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a);
            return 1;
        }
    }
    if (!modelPath) {
        fprintf(stderr,
                "usage: %s [<model.gguf>] [--file <prompt> | \"prompt\"] [--tokens]\n",
                argv[0]);
        return 1;
    }
    if (promptLabel.empty()) {
        // Byte-exact reproduction of formatChat() for qwen35moe Q1.
        prompt = "<|im_start|>user\n$"
                 "<|im_start|>system\nYou are TinyCoder, an AI coding assistant. Be concise.<|im_end|>\n"
                 "<|im_start|>user\nWrite a C++ function to add two numbers.<|im_end|>\n"
                 "<|im_start|>assistant\n"
                 "<|im_end|>\n<|im_start|>assistant\n";
        promptLabel = "default-wrapped-Q1";
    }

    fprintf(stderr, "loading model %s ...\n", modelPath);
    Model model;
    std::string loadErr;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!model.load(modelPath, &loadErr)) {
        fprintf(stderr, "model load failed: %s\n", loadErr.c_str());
        return 2;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "model loaded in %.1f s\n",
            std::chrono::duration<double>(t1 - t0).count());

    const ModelConfig &cfg = model.config();
    const int64_t vocabSize = cfg.vocabSize;

    printf("== tinycoder logits probe ==\n");
    printf("  model=%s arch=%s vocab=%lld\n", cfg.modelName.c_str(),
           cfg.architecture.c_str(), (long long) vocabSize);
    printf("  prompt=%s (%zu bytes)\n", promptLabel.c_str(), prompt.size());

    std::vector<int32_t> toks = model.tokenize(prompt);
    printf("  tokens: %zu\n", toks.size());

    if (runGenerate) {
        // Replicate SampleQuestionTest.AnswersQuestion exactly: formatChat with
        // the Q1 system/user pair + addGenerationPrompt=true, then generate()
        // with the test's InferenceParams.
        std::string chatPrompt = model.formatChat(
                {{"system", "You are TinyCoder, an AI coding assistant. Be concise."},
                 {"user", "Write a C++ function to add two numbers."}},
                true);
        printf("  formatChat prompt: %zu bytes\n", chatPrompt.size());
        printf("  generate params: maxTokens=%d temp=0.7 topP=0.9 topK=40 "
               "repeatPenalty=1.1 seed=%u\n",
               (int) genMaxTokens, genSeed);

        InferenceParams params;
        params.maxTokens = genMaxTokens;
        params.temperature = 0.7f;
        params.topP = 0.9f;
        params.topK = 40;
        params.repeatPenalty = 1.1f;
        params.seed = genSeed;

        std::string generatedText;
        int tokenCount = 0;
        auto g0 = std::chrono::high_resolution_clock::now();
        model.generate(chatPrompt, params,
                       [&](int32_t token, const std::string &text) -> bool {
                           generatedText += text;
                           tokenCount++;
                           if (tokenCount <= 200) {
                               printf("    step%4d: id=%d \"%s\"%s\n", tokenCount,
                                      token, text.c_str(),
                                      model.tokenizer().isSpecialToken(token)
                                              ? " !SPECIAL"
                                              : "");
                           }
                           return true;
                       });
        auto g1 = std::chrono::high_resolution_clock::now();
        double genS = std::chrono::duration<double>(g1 - g0).count();
        printf("  == generated %d tokens ==\n", tokenCount);
        printf("  full output (%d tokens, %.2f tok/s):\n", tokenCount,
               tokenCount > 0 ? tokenCount / genS : 0.0);
        printf("  >>> %s <<<\n", generatedText.c_str());

        // Keyword check as the test does (case-insensitive "return").
        std::string lowerOutput = generatedText;
        std::transform(lowerOutput.begin(), lowerOutput.end(), lowerOutput.begin(),
                       ::tolower);
        printf("  keyword-check \"return\": %s\n",
               lowerOutput.find("return") != std::string::npos ? "FOUND" : "MISSING");
        printf("  keyword-check \"def \": %s\n",
               lowerOutput.find("def ") != std::string::npos ? "FOUND(ban!)" : "absent(ok)");
        return 0;
    }

    if (dumpTokens) {
        const Tokenizer &tok0 = model.tokenizer();
        for (size_t t = 0; t < toks.size(); ++t) {
            std::string piece = tok0.decodeToken(toks[t]);
            printf("    tok[%zu]: id=%d \"%s\"%s\n", t, toks[t], piece.c_str(),
                   tok0.isSpecialToken(toks[t]) ? " !SPECIAL" : "");
        }
    }

    printf("  forward(computeAllLogits=false) ...\n");
    auto f0 = std::chrono::high_resolution_clock::now();
    np::Array<float> logits = model.forward(toks, false);
    auto f1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "  forward took %.1f s\n",
            std::chrono::duration<double>(f1 - f0).count());

    // Shape normalization (see ModelGeneration.cpp:201-208): the CPU engine
    // returns [seqLen, vocabSize] (last row populated), the GPU engine returns
    // compact [1, vocabSize].
    const float *lastRow = logits.size() <= (size_t) vocabSize
                                   ? logits.data()
                                   : logits.data() + (logits.size() - (size_t) vocabSize);
    printf("  logits array size=%zu (stride=vocab=%lld)\n", logits.size(),
           (long long) vocabSize);

    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve((size_t) vocabSize);
    for (int64_t i = 0; i < vocabSize; ++i) scored.emplace_back(lastRow[i], (int32_t) i);
    std::partial_sort(scored.begin(), scored.begin() + 10, scored.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

    const Tokenizer &tok = model.tokenizer();
    printf("  [ar-equivalent] final logits (last prompt token) top-10:\n");
    for (int i = 0; i < 10; ++i) {
        const auto &p = scored[i];
        std::string piece = tok.decodeToken(p.second);
        printf("    top%d: id=%d logit=%12.6f text=\"%s\"%s\n", i + 1,
               p.second, p.first, piece.c_str(),
               tok.isSpecialToken(p.second) ? " !SPECIAL" : "");
    }
    return 0;
}