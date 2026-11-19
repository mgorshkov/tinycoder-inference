// ctx_probe.cpp
//
// Needle-in-a-haystack probe for the context window.
//
// Puts an access code near the start of a long prompt, then asks for it with a
// question at the very end, so the model has to attend across the whole window
// to answer.  A window that is "granted" but not actually cached fails this: the
// question is at token ~T while the answer is at token ~20.
//
// Usage:
//   ctx_probe <model.gguf> --ctx N [--needle-tokens T] [--gen G]
//
//   --ctx N           request N tokens (0 = auto, i.e. 2048)
//   --needle-tokens T place the question at ~T tokens of prompt (default: the
//                     granted window, minus a margin)
//   --gen G           generate at most G tokens (default 16)
//
// Prints `granted=<n> maxAllowed=<n> kv=<f32|int8>` and the generated text, and
// exits 0 only if the answer contains the code.  Exit 1 means the model did not
// answer; exit 2 means the engine refused the prompt (window too small), 3 means
// bad usage.
//
// Build: cmake --build build-1.0 --target ctx_probe

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Model.hpp"

namespace {

constexpr const char *kCode = "pineapple-7731";

std::string fillerLine(int32_t i) {
    char buf[128];
    // Deliberately varied: an identical line repeated 3600 times collapses a
    // weak (ternary 2-bit) model into an immediate EOS -- Bonsai-2-27B answered
    // "pineapple-7731" at 86 tokens and emitted only `|im_end|` at 3600 with the
    // old constant filler, in BOTH the f32 and the int8 store, i.e. the prompt
    // was the problem and not the KV path.
    std::snprintf(buf, sizeof(buf),
                  "Record %06d: maintenance log, unit %d, batch %d, shift %d, "
                  "checked, no action taken.",
                  i, i % 7 + 1, i % 13 + 1, i % 3 + 1);
    return std::string(buf) + "\n";
}

}  // namespace

int main(int argc, char **argv) {
    const char *modelPath = nullptr;
    uint32_t ctx = 0;
    int32_t needleTokens = -1;
    int32_t gen = 16;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (std::strcmp(a, "--ctx") == 0 && i + 1 < argc) {
            ctx = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(a, "--needle-tokens") == 0 && i + 1 < argc) {
            needleTokens = std::atoi(argv[++i]);
        } else if (std::strcmp(a, "--gen") == 0 && i + 1 < argc) {
            gen = std::atoi(argv[++i]);
        } else if (a[0] != '-') {
            modelPath = a;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a);
            return 3;
        }
    }
    if (modelPath == nullptr) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> --ctx N [--needle-tokens T] [--gen G]\n",
                     argv[0]);
        return 3;
    }

    tinycoder::Model model;
    if (ctx > 0 && !model.setContextLength(ctx)) {
        std::fprintf(stderr, "cannot set context length %u\n", ctx);
        return 3;
    }
    std::string err;
    if (!model.load(modelPath, &err)) {
        std::fprintf(stderr, "model load failed: %s\n", err.c_str());
        return 3;
    }
    const auto &plan = model.contextPlan();
    std::printf("granted=%u maxAllowed=%u kv=%s mirror=%d\n", plan.seqLen,
                plan.maxAllowed,
                plan.kv == tinycoder::detail::KvPrecision::I8 ? "int8" : "f32",
                plan.kvMirror ? 1 : 0);
    std::printf("reason=%s\n", plan.reason.c_str());
    // Whether a reduced grant is a failure is the caller's call (the gate
    // compares granted against the window it asked for); the probe's own job is
    // the needle, so it only refuses when the needle cannot fit at all.
    if (ctx > 0 && plan.seqLen < ctx) {
        std::printf("note: requested %u, granted %u (maxAllowed %u)\n", ctx,
                    plan.seqLen, plan.maxAllowed);
    }

    // Grant-only mode (`--needle-tokens 0`): some checkpoints cannot retrieve a
    // needle at any useful distance (Ternary-Bonsai-2-27B answers at 600 tokens
    // and emits EOS at 3600 in BOTH the f32 and the int8 store), so for those the
    // gate can only assert the grant and that the model still speaks.  Say so
    // instead of dressing a model limitation up as a window result.
    if (needleTokens == 0) {
        std::printf("grant-only: %u tokens granted (%s K/V), no needle check\n",
                    plan.seqLen,
                    plan.kv == tinycoder::detail::KvPrecision::I8 ? "int8" : "f32");
        std::printf("PASS (grant)\n");
        return plan.seqLen > 0 ? 0 : 4;
    }

    // Target prompt length: the requested needle position, clamped to the
    // window minus room for the question.
    const int32_t window = static_cast<int32_t>(model.contextLength());
    const int32_t want = needleTokens > 0 ? needleTokens : window;
    const int32_t target = want > window - 64 ? window - 64 : want;

    // One tokenize of the whole prompt: the BPE is not free at 64k, and
    // retokenizing per appended line (the first version) took minutes.
    const std::string head = std::string("The access code is ") + kCode + ".\n";
    // Ask for the first line verbatim: a question whose answer is a copy is far
    // more stable across models than "what is the code" (a small model will
    // answer "7731" and drop the prefix), and it still requires reading token 0
    // from the far end of the window.
    const std::string tail =
            "Question: repeat the first line of this document verbatim.\nFirst line: ";
    const int32_t perLine = std::max(1, static_cast<int32_t>(
                                             model.tokenize(fillerLine(0)).size()));
    std::string prompt = head;
    const int32_t headToks = static_cast<int32_t>(model.tokenize(head + tail).size());
    // Two lines of slack: the per-line token count varies with the digits in the
    // filler (a 65024-token target came out at 65649 and the prompt did not fit
    // its own window).  Trim and retokenize if it still overshoots.
    int32_t lines = target > headToks ? (target - headToks) / perLine - 2 : 0;
    if (lines < 0) lines = 0;
    for (int32_t line = 0; line < lines; ++line) prompt += fillerLine(line);
    prompt += tail;
    const auto tTok0 = std::chrono::steady_clock::now();
    std::vector<int32_t> toks = model.tokenize(prompt);
    // Shrink until it fits: a prompt that does not fit its own granted window
    // must not be mistaken for a broken window.
    for (int32_t guard = 0; guard < 8 && static_cast<int32_t>(toks.size()) > window &&
                           lines > 0;
         ++guard) {
        lines = std::max(0, lines - 8);
        prompt = head;
        for (int32_t line = 0; line < lines; ++line) prompt += fillerLine(line);
        prompt += tail;
        toks = model.tokenize(prompt);
    }
    const auto tTok1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "  tokenize: %.1f s for %zu tokens (%zu bytes)\n",
                 std::chrono::duration<double>(tTok1 - tTok0).count(),
                 toks.size(), prompt.size());

    std::printf("prompt_tokens=%zu target=%d\n", toks.size(), target);

    if (static_cast<int32_t>(toks.size()) > window) {
        std::fprintf(stderr, "prompt does not fit: %zu > %d granted\n",
                     toks.size(), window);
        return 2;
    }

    tinycoder::InferenceParams params;
    params.maxTokens = gen;
    params.temperature = 0.0f;// greedy
    params.topK = 1;
    params.repeatPenalty = 1.0f;
    std::string out;
    // generate() takes the prompt text; it tokenizes to the same ids counted
    // above (same tokenizer, same bytes).
    const auto ids = model.generate(
            prompt, params,
            [&out](int32_t, const std::string &text) {
                out += text;
                return out.size() < 4096;
            });
    std::printf("generated_tokens=%zu\n", ids.size());
    std::printf("answer=<<<%s>>>\n", out.c_str());
    if (model.contextFull()) {
        std::fprintf(stderr, "the session hit the end of the window\n");
    }

    if (out.find(kCode) != std::string::npos) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("FAIL: the answer does not contain %s\n", kCode);
    return 1;
}
