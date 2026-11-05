/*
 * Decode-vs-prefill agreement probe.
 *
 * The prefill path (seqLen > 1) is bit-exact against the CPU reference for this
 * checkpoint; the decode path (seqLen == 1, i.e. the EXL3 fused GEMV) is the
 * suspect when generation produces garbage.  This tool removes sampling, the
 * tokenizer and the sampler from the picture: it feeds one fixed token stream
 * through both paths and compares the logits they produce for the SAME token at
 * the SAME position:
 *
 *   A (prefill): forward(t[0..N-1])                     -> logits for t[N-1]
 *   B (decode) : forward(t[0..N-2]) ; forward(t[N-1..])  -> logits for t[N-1]
 *
 * Any difference is the decode path's own error.  --pos P runs the comparison
 * for the token at position P for several P values, and --layers dumps
 * TINYCODER_DUMP_NORM-style first values so the divergence can be localised.
 */
#include "Model.hpp"
#include "Tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tinycoder;

namespace {

    struct Top {
        std::vector<std::pair<float, int32_t>> v;
    };

    Top topN(const float *logits, int64_t vocab, int n) {
        Top t;
        t.v.reserve(static_cast<size_t>(vocab));
        for (int64_t i = 0; i < vocab; ++i) {
            t.v.emplace_back(logits[i], static_cast<int32_t>(i));
        }
        const int k = static_cast<int>(std::min<int64_t>(n, vocab));
        std::partial_sort(t.v.begin(), t.v.begin() + k, t.v.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });
        t.v.resize(static_cast<size_t>(k));
        return t;
    }

    struct Stats {
        double maxAbs = 0.0;
        double rms = 0.0;
        int nanA = 0, nanB = 0;
        float minA = 1e30f, maxA = -1e30f, minB = 1e30f, maxB = -1e30f;
        int argA = -1, argB = -1;
    };

    Stats compare(const float *a, const float *b, int64_t vocab) {
        Stats s;
        double ssq = 0.0;
        for (int64_t i = 0; i < vocab; ++i) {
            const float x = a[i], y = b[i];
            if (std::isnan(x) || std::isinf(x)) ++s.nanA;
            if (std::isnan(y) || std::isinf(y)) ++s.nanB;
            s.minA = std::min(s.minA, x);
            s.maxA = std::max(s.maxA, x);
            s.minB = std::min(s.minB, y);
            s.maxB = std::max(s.maxB, y);
            const double d = static_cast<double>(x) - static_cast<double>(y);
            ssq += d * d;
            s.maxAbs = std::max(s.maxAbs, std::fabs(d));
        }
        s.rms = std::sqrt(ssq / static_cast<double>(vocab));
        return s;
    }

    void printTop(const Tokenizer &tok, const char *tag, const Top &t) {
        printf("    %s:", tag);
        for (size_t i = 0; i < t.v.size(); ++i) {
            std::string piece = tok.decodeToken(t.v[i].second);
            std::string esc;
            for (char c: piece) {
                if (c >= 32 && c < 127) esc += c;
                else {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\x%02x", (unsigned) (unsigned char) c);
                    esc += buf;
                }
            }
            printf(" [%zu] id=%d logit=%.6f \"%s\"", i, t.v[i].second, t.v[i].first,
                   esc.c_str());
        }
        printf("\n");
    }

}// namespace

int main(int argc, char **argv) {
    const char *modelPath = std::getenv("TINYCODER_MODEL_PATH");
    std::string prompt;
    int maxPos = 4;
    int nTop = 3;
    int mode = 0;// 0: compare, 1: both paths' top only

    std::string promptArg;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--model") == 0 && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (strcmp(a, "--max-pos") == 0 && i + 1 < argc) {
            maxPos = atoi(argv[++i]);
        } else if (strcmp(a, "--top") == 0 && i + 1 < argc) {
            nTop = atoi(argv[++i]);
        } else if (strcmp(a, "--tokens-only") == 0) {
            mode = 1;
        } else if (a[0] != '-') {
            if (modelPath == nullptr) modelPath = a;
            else promptArg = a;
        }
    }
    if (modelPath == nullptr) {
        fprintf(stderr, "usage: %s --model <dir> [--max-pos N] [\"prompt\"]\n", argv[0]);
        return 1;
    }
    if (promptArg.empty()) {
        prompt = "<|im_start|>user\n$"
                 "<|im_start|>system\nYou are TinyCoder, an AI coding assistant. Be concise.<|im_end|>\n"
                 "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
                 "<|im_start|>assistant\n";
    } else {
        prompt = promptArg;
    }

    Model model;
    std::string loadErr;
    fprintf(stderr, "loading model %s ...\n", modelPath);
    if (!model.load(modelPath, &loadErr)) {
        fprintf(stderr, "model load failed: %s\n", loadErr.c_str());
        return 2;
    }
    const ModelConfig &cfg = model.config();
    const int64_t vocab = cfg.vocabSize;
    const Tokenizer &tok = model.tokenizer();
    std::vector<int32_t> tokens = model.tokenize(prompt);
    printf("prompt tokens: %zu (vocab %lld)\n", tokens.size(), (long long) vocab);
    printf("ids:");
    for (int32_t id : tokens) printf(" %d", id);
    printf("\n");
    if (tokens.size() < 3) {
        fprintf(stderr, "prompt too short\n");
        return 2;
    }
    if (static_cast<size_t>(maxPos) >= tokens.size()) {
        maxPos = static_cast<int>(tokens.size()) - 1;
    }
    (void) mode;

    // Deterministic token stream: prompt + a few extra tokens from the prompt's
    // own tail so position N is always inside the real tokenizer vocabulary.
    std::vector<int32_t> stream = tokens;
    for (int i = 0; i < maxPos + 2; ++i) stream.push_back(tokens[1 + (i % (tokens.size() - 1))]);

    auto runSeq = [&](size_t begin, size_t end) -> std::vector<float> {
        std::vector<int32_t> chunk(stream.begin() + static_cast<long>(begin),
                                   stream.begin() + static_cast<long>(end));
        np::Array<float> lg = model.forward(chunk, false);
        const size_t n = lg.size();
        const float *row = n <= static_cast<size_t>(vocab)
                                   ? lg.data()
                                   : lg.data() + (n - static_cast<size_t>(vocab));
        return std::vector<float>(row, row + static_cast<size_t>(vocab));
    };

    if (std::getenv("TINYCODER_AGREE_SWEEP") != nullptr) {
        // Length sweep: forward(stream[0..n-1]) for increasing n, report the
        // health of the last-position logits.  First NaN/RMS blow-up localises
        // the sequence length at which the forward pass breaks.
        printf("\n=== length sweep ===\n");
        const char *lenEnv = std::getenv("TINYCODER_AGREE_LEN");
        size_t onlyLen = lenEnv != nullptr ? static_cast<size_t>(atoi(lenEnv)) : 0;
        for (size_t n = 1; n <= stream.size(); ++n) {
            if (onlyLen != 0 && n != onlyLen) continue;
            model.clearKVCache();
            std::vector<int32_t> chunk(stream.begin(),
                                       stream.begin() + static_cast<long>(n));
            np::Array<float> lg = model.forward(chunk, false);
            const size_t sz = lg.size();
            const float *row = sz <= static_cast<size_t>(vocab)
                                       ? lg.data()
                                       : lg.data() + (sz - static_cast<size_t>(vocab));
            int bad = 0;
            float mn = 1e30f, mx = -1e30f;
            double ssq = 0.0;
            for (int64_t i = 0; i < vocab; ++i) {
                const float v = row[i];
                if (!std::isfinite(v)) ++bad;
                else {
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                    ssq += static_cast<double>(v) * v;
                }
            }
            Top t = topN(row, vocab, 5);
            printf("  n=%3zu  nonfinite=%7d  range=[%.4f, %.4f]  rms=%.4f  top1=id%d "
                   "logit=%.6f",
                   n, bad, mn, mx, std::sqrt(ssq / std::max(1.0, (double) (vocab - bad))),
                   t.v.empty() ? -1 : t.v[0].second, t.v.empty() ? 0.f : t.v[0].first);
            // Machine-readable top-5 for comparison against an external oracle
            // (scripts/exl3_oracle_logits.py) on the same token ids.
            printf("  top5=");
            for (const auto &kv : t.v) printf("%d:%.4f,", kv.second, kv.first);
            printf("\n");
            fflush(stdout);
            if (bad > 0) break;
        }
        return 0;
    }

    if (const char *genEnv = std::getenv("TINYCODER_AGREE_GEN")) {
        // Greedy continuation of the stream with the KV cache and recurrent
        // state carried across steps - the same thing Model::generate does, but
        // printing the ids so the sequence can be diffed against an oracle.
        const int nGen = std::max(1, atoi(genEnv));
        printf("\n=== greedy continuation (%d tokens) ===\n", nGen);
        model.clearKVCache();
        np::Array<float> lg = model.forward(stream, false);
        auto top1Of = [&](const np::Array<float> &l) {
            const size_t sz = l.size();
            const float *row = sz <= static_cast<size_t>(vocab)
                                       ? l.data()
                                       : l.data() + (sz - static_cast<size_t>(vocab));
            int best = -1;
            float bv = -1e30f;
            for (int64_t i = 0; i < vocab; ++i)
                if (std::isfinite(row[i]) && row[i] > bv) { bv = row[i]; best = static_cast<int>(i); }
            return std::make_pair(best, bv);
        };
        auto px = top1Of(lg);
        printf("  gen[%2d] id=%-7d logit=%.6f   (from prefill)\n", 0, px.first, px.second);
        int32_t cur = static_cast<int32_t>(px.first);
        for (int i = 1; i < nGen; ++i) {
            std::vector<int32_t> one{cur};
            np::Array<float> l2 = model.forward(one, false);
            auto p2 = top1Of(l2);
            printf("  gen[%2d] id=%-7d logit=%.6f\n", i, p2.first, p2.second);
            fflush(stdout);
            cur = static_cast<int32_t>(p2.first);
        }
        return 0;
    }

    if (std::getenv("TINYCODER_AGREE_SEQ") != nullptr) {
        // Sequential decode: clear the cache, then feed the stream ONE token at a
        // time (seqLen==1 every step, i.e. the decode kernels + recurrent-state
        // updates).  Compare against a single batched prefill of the same stream:
        // if prefill NaNs and this stays finite, the fault is in a batched
        // (seqLen>1) kernel, not in the decode path.
        printf("\n=== sequential 1-token forwards ===\n");
        model.clearKVCache();
        for (size_t n = 1; n <= stream.size(); ++n) {
            std::vector<int32_t> one{stream[n - 1]};
            np::Array<float> lg = model.forward(one, false);
            const size_t sz = lg.size();
            const float *row = sz <= static_cast<size_t>(vocab)
                                       ? lg.data()
                                       : lg.data() + (sz - static_cast<size_t>(vocab));
            int bad = 0;
            float mn = 1e30f, mx = -1e30f;
            for (int64_t i = 0; i < vocab; ++i) {
                const float v = row[i];
                if (!std::isfinite(v)) ++bad;
                else {
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                }
            }
            Top t = topN(row, vocab, 1);
            printf("  step=%3zu tok=%6d nonfinite=%7d range=[%.3f, %.3f] top1=id%d "
                   "logit=%.6f\n",
                   n, stream[n - 1], bad, mn, mx, t.v.empty() ? -1 : t.v[0].second,
                   t.v.empty() ? 0.f : t.v[0].first);
            fflush(stdout);
            if (bad > 0) break;
        }
        return 0;
    }

    printf("\n=== single-position comparison (pos, prefill-logits vs decode-logits) ===\n");
    for (int pos = 0; pos <= maxPos; ++pos) {
        // A: prefill of stream[0..pos]
        model.clearKVCache();
        std::vector<float> A = runSeq(0, static_cast<size_t>(pos) + 1);

        // B: prefill stream[0..pos-1] then decode stream[pos]
        model.clearKVCache();
        std::vector<float> B;
        if (pos == 0) {
            B = runSeq(0, 1);
            A = B;// identical path at pos 0; nothing to compare
        } else {
            runSeq(0, static_cast<size_t>(pos));
            B = runSeq(static_cast<size_t>(pos), static_cast<size_t>(pos) + 1);
        }

        Stats s = compare(A.data(), B.data(), vocab);
        printf("  pos=%d token=%d\n", pos, stream[static_cast<size_t>(pos)]);
        printf("    prefill: range [%.4f, %.4f]  decode: range [%.4f, %.4f]  "
               "nan=%d/%d  maxAbsDiff=%.6g rmsDiff=%.6g\n",
               s.minA, s.maxA, s.minB, s.maxB, s.nanA, s.nanB, s.maxAbs, s.rms);
        Top ta = topN(A.data(), vocab, nTop);
        Top tb = topN(B.data(), vocab, nTop);
        printTop(tok, "prefill", ta);
        printTop(tok, "decode ", tb);
        printf("    argmax %s\n",
               (!ta.v.empty() && !tb.v.empty() && ta.v[0].second == tb.v[0].second)
                       ? "AGREE"
                       : "DIFFER");
        fflush(stdout);
    }
    return 0;
}
