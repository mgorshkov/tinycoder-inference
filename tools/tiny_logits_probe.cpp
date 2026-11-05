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

// tiny_logits_probe.cpp
//
// Loads the FULL model (like the unit tests) so this is slow (~5-6 min on the
// 35B-A3B Q8_0) — run with TINYCODER_GPU=0/1 to switch engines.
//
// Build:
//   g++ -O2 -std=c++17 tools/tiny_logits_probe.cpp -I include -Isrc/cpp/core
//       build/libtinycoder_core.a -lpthread -o /tmp/tiny_logits_probe
//
// Run:
//   TINYCODER_MODEL_PATH=<model.gguf> tiny_logits_probe --file prompt.txt
//      [--tokens] [--generate]
//   (with an inline prompt: TINYCODER_MODEL_PATH=<model.gguf>
//      tiny_logits_probe "prompt text" [--tokens] [--generate] )
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
    std::vector<int32_t> idsPrompt;   // --ids-file: raw ids, no tokenizer
    int seqGen = 0;                   // --seq-gen: decode-vs-prefill chain
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
        } else if (strcmp(a, "--ids-file") == 0 && i + 1 < argc) {
            // Raw token ids, whitespace/comma separated.  Lets an oracle be fed
            // the exact same ids without a tokenizer round-trip, which is the
            // only way to compare logits at a position whose text form the
            // tokenizer does not round-trip (special tokens like ' thinking').
            std::ifstream in(argv[i + 1]);
            long long v = 0;
            while (in >> v) {
                idsPrompt.push_back((int32_t) v);
                in >> std::ws;
                if (in.peek() == ',') in.get();
                in >> std::ws;
            }
            promptLabel = std::string("ids:") + argv[i + 1];
            i += 2;
        } else if (strcmp(a, "--seq-gen") == 0 && i + 1 < argc) {
            seqGen = std::atoi(argv[i + 1]);
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

    std::vector<int32_t> toks =
            idsPrompt.empty() ? model.tokenize(prompt) : idsPrompt;
    printf("  tokens: %zu\n", toks.size());

    if (runGenerate) {
        // Replicate SampleQuestionTest.AnswersQuestion exactly: formatChat with
        // the Q1 system/user pair + addGenerationPrompt=true, then generate()
        // with the test's InferenceParams.
        //
        // TINYCODER_GEN_QUESTION overrides the user turn and TINYCODER_GEN_TEMP
        // the temperature, so the same binary can run the acceptance question
        // greedily (TINYCODER_GEN_TEMP=0) on any prompt; defaults reproduce the
        // test byte-for-byte.
        const char *qEnv = std::getenv("TINYCODER_GEN_QUESTION");
        const std::string question =
                qEnv != nullptr ? std::string(qEnv)
                                : std::string("Write a C++ function to add two numbers.");
        const char *tEnv = std::getenv("TINYCODER_GEN_TEMP");
        const float genTemp = tEnv != nullptr ? std::stof(tEnv) : 0.7f;
        const char *rpEnv = std::getenv("TINYCODER_GEN_REPEAT");
        const float genRepeat = rpEnv != nullptr ? std::stof(rpEnv) : 1.1f;
        // topK=0 disables the exact top-K LM-head pruning inside forward(), i.e.
        // the full vocabulary path.  Diffing the two isolates a pruning bug.
        const char *tkEnv = std::getenv("TINYCODER_GEN_TOPK");
        const int genTopK = tkEnv != nullptr ? std::atoi(tkEnv) : 40;
        // --file / positional prompt: use those bytes verbatim as the generate
        // prompt (no chat template), so an arbitrary prompt -- including an
        // oracle-verified one -- can be run through the real generate() path.
        std::string chatPrompt =
                promptLabel.empty() || promptLabel == "default-wrapped-Q1"
                        ? model.formatChat(
                                  {{"system",
                                    "You are TinyCoder, an AI coding assistant. Be concise."},
                                   {"user", question}},
                                  true)
                        : prompt;
        printf("  formatChat prompt: %zu bytes, question=\"%s\"\n",
               chatPrompt.size(), question.c_str());
        // Escaped dump of the exact bytes generate() sees, so an oracle can be
        // fed the identical prompt (a chat-template difference otherwise hides
        // a math difference behind a different token stream).
        {
            std::string esc;
            for (char c: chatPrompt) {
                if (c == '\n') esc += "\\n";
                else if (c == '\r') esc += "\\r";
                else esc += c;
            }
            printf("  formatChat prompt escaped: \"%s\"\n", esc.c_str());
            if (const char *po = std::getenv("TINYCODER_GEN_PROMPT_OUT");
                po != nullptr) {
                std::ofstream out(po, std::ios::binary);
                out.write(chatPrompt.data(), (std::streamsize) chatPrompt.size());
                printf("  wrote prompt bytes to %s\n", po);
            }
        }
        printf("  generate params: maxTokens=%d temp=%.3f topP=0.9 topK=%d "
               "repeatPenalty=%.3f seed=%u\n",
               (int) genMaxTokens, (double) genTemp, genTopK, (double) genRepeat,
               genSeed);

        InferenceParams params;
        params.maxTokens = genMaxTokens;
        params.temperature = genTemp;
        params.topP = 0.9f;
        params.topK = genTopK;
        params.repeatPenalty = genRepeat;
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


    if (seqGen > 0) {
        // Decode-vs-prefill chain.  Greedy decoding samples from BOTH paths at
        // every step and prints them side by side:
        //   prefill: forward(ids[0..n-1])            (batched)
        //   decode : forward(ids[0..n-2]) then forward(ids[n-1])  (seqLen==1)
        // This is the only probe that walks the *stateful* decode path (GDN
        // recurrent state + conv state, EXL3 fused GEMV) across steps, which is
        // what Model::generate() actually uses; exl3_decode_agree restarts from
        // a fresh batched prefill for every position it checks.
        auto top1 = [&](const np::Array<float> &l, float *outLogit,
                        int32_t *outId) {
            const float *row = l.size() <= (size_t) vocabSize
                                       ? l.data()
                                       : l.data() + (l.size() - (size_t) vocabSize);
            int best = -1;
            float bv = -1e30f;
            for (int64_t k = 0; k < vocabSize; ++k)
                if (std::isfinite(row[k]) && row[k] > bv) { bv = row[k]; best = (int32_t) k; }
            *outId = best;
            *outLogit = bv;
        };
        std::vector<int32_t> ids = toks;
        const size_t base = ids.size();
        printf("  == decode-vs-prefill chain (%d steps) ==\n", seqGen);
        // Pass A: the authoritative greedy chain, each step re-anchored by a
        // fresh batched prefill (this is the path the oracle comparison uses).
        std::vector<int32_t> chainIds;
        std::vector<int32_t> chainTop;
        std::vector<float> chainLogit;
        for (int step = 0; step < seqGen; ++step) {
            model.clearKVCache();
            np::Array<float> pre = model.forward(ids, false);
            float l = 0.f;
            int32_t id = -1;
            top1(pre, &l, &id);
            chainTop.push_back(id);
            chainLogit.push_back(l);
            ids.push_back(id);
        }
        // Pass B: ONE base prefill followed by seqLen==1 forwards only -- the
        // stateful path Model::generate() actually uses.  If the two chains
        // disagree, the decode-time state update (GDN recurrent + conv state)
        // is diverging from what a batched prefill computes for the same ids.
        model.clearKVCache();
        np::Array<float> live = model.forward(
                std::vector<int32_t>(ids.begin(), ids.begin() + (long) base), false);
        for (int step = 0; step < seqGen; ++step) {
            float l = 0.f;
            int32_t id = -1;
            top1(live, &l, &id);
            printf("    step %2d  pos=%3zu  prefill: id=%-7d logit=%9.4f   "
                   "seqdec: id=%-7d logit=%9.4f   %s\n",
                   step, base - 1 + (size_t) step, chainTop[(size_t) step],
                   chainLogit[(size_t) step], id, l,
                   chainTop[(size_t) step] == id ? "AGREE" : "DIFFER");
            fflush(stdout);
            if (step + 1 < seqGen) {
                live = model.forward({chainTop[(size_t) step]}, false);
            }
        }
        model.clearKVCache();
        return 0;
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
