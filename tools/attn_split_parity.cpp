// -----------------------------------------------------------------------------
// attn_split_parity — correctness harness for the split-KV decode attention.
//
// Runs a long-prompt prefill + greedy decode and prints the generated token id
// stream.  Run twice (TINYCODER_ATTN_SPLIT=1 and =0) and diff the streams:
// the split path changes only the online-softmax reduction grouping, so the
// top-1 token stream must be identical.
//
// "cmp" mode runs BOTH paths in ONE process (GPU graph disabled so the path is
// chosen per-forward) back-to-back on the identical prompt, which removes
// cross-process variation, and prints a per-token diff.
//
//   TINYCODER_GPU=1 TINYCODER_ATTN_SPLIT=1 tinycoder_attn_parity <model> [nGen] [repeats] [eager|split|cmp]
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Model.hpp"
#include "ModelConfig.hpp"

namespace {
    std::vector<int32_t> runDecode(tinycoder::Model &model,
                                   const std::vector<int32_t> &prompt, int nGen) {
        model.clearKVCache();
        auto logits = model.forward(prompt, false);
        const int32_t vocab = static_cast<int32_t>(model.config().vocabSize);
        auto argmax = [&](const float *l) {
            int32_t b = 0;
            float bv = l[0];
            for (int32_t i = 1; i < vocab; ++i) {
                if (l[i] > bv) {
                    bv = l[i];
                    b = i;
                }
            }
            return b;
        };
        std::vector<int32_t> out;
        int32_t tok = argmax(logits.data());
        out.push_back(tok);
        for (int i = 0; i < nGen; ++i) {
            logits = model.forward({tok}, false);
            tok = argmax(logits.data());
            out.push_back(tok);
        }
        return out;
    }

    void printStream(const char *tag, const std::vector<int32_t> &v) {
        std::printf("%s", tag);
        for (int32_t t: v) std::printf(" %d", t);
        std::printf("\n");
    }
}// namespace

int main(int argc, char **argv) {
    const std::string modelPath =
            argc > 1 ? argv[1]
                     : "/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf";
    const int nGen = argc > 2 ? std::atoi(argv[2]) : 48;
    const int nRepeat = argc > 3 ? std::atoi(argv[3]) : 40;
    const std::string mode = argc > 4 ? argv[4] : "eager";
    setenv("TINYCODER_GPU", "1", 1);

    tinycoder::Model model;
    std::string err;
    if (!model.load(modelPath, &err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    std::string prompt;
    for (int i = 0; i < nRepeat; ++i) {
        prompt += "def f(x): return x*x + 1  # parity probe line\n";
    }
    auto toks = model.tokenize(prompt);
    std::fprintf(stderr, "prompt tokens=%zu\n", toks.size());

    if (mode == "cmp") {
        // In-process A/B: disable the CUDA graph so the attention path is
        // selected per forward from the env var, then run split-off and
        // split-on back to back on the same prompt.
        setenv("TINYCODER_GPU_GRAPH", "0", 1);
        setenv("TINYCODER_ATTN_SPLIT", "0", 1);
        auto a = runDecode(model, toks, nGen);
        setenv("TINYCODER_ATTN_SPLIT", "1", 1);
        auto b = runDecode(model, toks, nGen);
        printStream("EAGER", a);
        printStream("SPLIT", b);
        int diff = -1;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
            if (a[i] != b[i]) {
                diff = static_cast<int>(i);
                break;
            }
        }
        if (diff < 0) {
            std::printf("RESULT IDENTICAL (%zu tokens)\n", a.size());
        } else {
            std::printf("RESULT DIFFER at token %d: eager=%d split=%d\n", diff,
                        a[diff], b[diff]);
        }
        return 0;
    }

    setenv("TINYCODER_ATTN_SPLIT", mode == "split" ? "1" : "0", 1);
    auto s = runDecode(model, toks, nGen);
    printStream("TOKENS", s);
    return 0;
}
