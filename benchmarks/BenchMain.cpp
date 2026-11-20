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

// -----------------------------------------------------------------------------
// tinycoder_bench — prefill/generation benchmark.
//
// Usage:
//   tinycoder_bench [--model <path>] [--n-prompts 512] [--n-gen 128]
//                   [--reps 5] [--threads N]
//   TINYCODER_THREADS=N env var is honored by ThreadPool::recommendedThreadCount.
//   TINYCODER_TOKEN_TIMES=1 prints per-token decode ms (tagged [tg-times]).
// -----------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Model.hpp"
#include "ModelConfig.hpp"
#include "ThreadPool.hpp"

namespace {

    struct BenchOpts {
        std::string modelPath;
        int32_t nPrompts = 512; // prompt token count
        int32_t nGen = 128;     // generated token count
        int32_t reps = 5;       // warm repeats
        int32_t threads = 0;    // 0 = recommendedThreadCount()
        uint32_t ctx = 0;        // --ctx: requested window, 0 = auto (2048)
        int forceGpu = -1;       // --gpu = 1, --cpu/--no-gpu = 0, unset = -1
    };

    double nowMs() {
        return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
    }

    std::vector<int32_t> buildPrompt(int32_t nTokens) {
        std::vector<int32_t> t;
        t.reserve(nTokens);
        uint64_t s = 0x5EED123456789ULL;
        for (int32_t i = 0; i < nTokens; ++i) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            // BPE ids in the low range (safe general-token range, avoids special
            // tokens that may be filtered).
            t.push_back(static_cast<int32_t>(s % 3000) + 10);
        }
        return t;
    }

    int32_t argmaxTop(const float *logits, int32_t vocabSize) {
        int32_t best = 0;
        float bestVal = logits[0];
        for (int32_t i = 1; i < vocabSize; ++i) {
            if (logits[i] > bestVal) {
                bestVal = logits[i];
                best = i;
            }
        }
        return best;
    }

    struct RepResult {
        double ms = 0.0;    // decode loop wall time (excludes prefill)
        int32_t tokens = 0; // generated tokens this rep
    };

    // TINYCODER_TOKEN_TIMES=1: per-token decode timing. One summary line answers
    // "is this run cold or just slow": first tokens slow and later tokens steady
    // means a one-time cost inside the timed region (lazy kernel build, allocator
    // growth, first touch); every token uniformly slow means clocks/bandwidth.
    // The full series is emitted too so a run can be re-plotted or diffed.
    void reportTokenTimes(const char *label, const std::vector<double> &ms) {
        if (ms.empty()) return;
        std::vector<double> sorted = ms;
        std::sort(sorted.begin(), sorted.end());
        auto pct = [&sorted](double p) {
            const size_t i = static_cast<size_t>(p * static_cast<double>(sorted.size() - 1) + 0.5);
            return sorted[i];
        };
        double mean = 0.0;
        for (double v: ms) mean += v;
        mean /= static_cast<double>(ms.size());
        const size_t nf = std::min<size_t>(5, ms.size());
        const size_t nl = std::min<size_t>(5, ms.size());
        double first = 0.0, last = 0.0;
        for (size_t i = 0; i < nf; ++i) first += ms[i];
        for (size_t i = ms.size() - nl; i < ms.size(); ++i) last += ms[i];
        first /= static_cast<double>(nf);
        last /= static_cast<double>(nl);
        std::fprintf(stderr,
                     "[tg-times] %s: n=%zu mean=%.2f ms p50=%.2f p95=%.2f max=%.2f "
                     "first%zu=%.2f last%zu=%.2f\n",
                     label, ms.size(), mean, pct(0.50), pct(0.95), sorted.back(),
                     nf, first, nl, last);
        std::fprintf(stderr, "[tg-times] %s series_ms=", label);
        for (double v: ms) std::fprintf(stderr, "%.2f,", v);
        std::fprintf(stderr, "\n");
    }

    // Prefill (untimed), then decode
    // nGen tokens greedily, timing ONLY the decode loop.
    RepResult runOneRep(tinycoder::Model &model, const std::vector<int32_t> &prompt,
                        int32_t nGen, double &prefillMsOut,
                        const int32_t vocabSize, const char *label = "") {
        // Prefill on a cleared cache, timed separately as a diagnostic
        // (EXCLUDED from the decode rate).
        model.clearKVCache();
        auto t0 = nowMs();
        auto logits = model.forward(prompt, false);
        auto t1 = nowMs();
        prefillMsOut = t1 - t0;

        // Decode loop — THE measured region.
        double tDec0 = nowMs();
        int32_t tok = argmaxTop(logits.data(), vocabSize);
        // TINYCODER_DUMP_TOKENS=1: print the greedy continuation so an A/B of
        // two execution paths can be diffed token-for-token.
        const char *dumpTok = std::getenv("TINYCODER_DUMP_TOKENS");
        const bool dump = dumpTok != nullptr && dumpTok[0] != '\0' &&
                          dumpTok[0] != '0';
        const char *timesTok = std::getenv("TINYCODER_TOKEN_TIMES");
        const bool tokenTimes = timesTok != nullptr && timesTok[0] != '\0' &&
                                timesTok[0] != '0';
        std::vector<double> tokenMs;
        if (tokenTimes) tokenMs.reserve(static_cast<size_t>(nGen));
        if (dump) std::fprintf(stderr, "[tokens]");
        int32_t gen = 0;
        for (int32_t i = 0; i < nGen; ++i) {
            if (dump) std::fprintf(stderr, " %d", tok);
            const double tTok0 = tokenTimes ? nowMs() : 0.0;
            logits = model.forward({tok}, false);
            tok = argmaxTop(logits.data(), vocabSize);
            ++gen;
            if (tokenTimes) tokenMs.push_back(nowMs() - tTok0);
        }
        if (dump) std::fprintf(stderr, " %d\n", tok);
        double tDec1 = nowMs();
        if (tokenTimes) reportTokenTimes(label, tokenMs);

        RepResult r;
        r.ms = tDec1 - tDec0;
        r.tokens = gen;
        return r;
    }

    void printUsage(const char *argv0) {
        std::fprintf(stderr,
                     "Usage: %s [options]\n"
                     "  --model <path>   GGUF model (default: $TINYCODER_MODEL_PATH)\n"
                     "  --n-prompts N    prompt token count (default 512)\n"
                     "  --n-gen N        generated token count (default 128)\n"
                     "  --reps N         warm repeat count (default 5)\n"
                     "  --threads N      thread pool size (default: logical CPUs)\n"
                     "  --ctx N          context window in tokens (default 0 = auto, 2048;\n"
                     "                   granted = min(request, model max, what fits))\n"
                     "  --gpu            use the CUDA offload engine (the default in a CUDA\n"
                     "                   build; overrides TINYCODER_GPU=0)\n"
                     "  --cpu, --no-gpu  force the CPU path (CPU baseline of the same run)\n"
                     "  TINYCODER_TOKEN_TIMES=1  print per-token decode ms tagged\n"
                     "                   [tg-times] (cold/warm diagnosis: first tokens vs rest)\n"
#ifndef USE_CUDA
                     "                   NOTE: this binary was built WITHOUT CUDA, so --gpu\n"
                     "                   is unavailable: reconfigure with -DENABLE_CUDA=ON\n"
                     "                   (the default when nvcc is installed)\n"
#endif
                     ,
                     argv0);
    }

} // namespace

int main(int argc, char **argv) {
    BenchOpts opt;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", argv[i]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--model") == 0) {
            opt.modelPath = next();
        } else if (std::strcmp(argv[i], "--n-prompts") == 0) {
            opt.nPrompts = std::atoi(next().c_str());
        } else if (std::strcmp(argv[i], "--ctx") == 0) {
            opt.ctx = static_cast<uint32_t>(std::atoi(next().c_str()));
        } else if (std::strcmp(argv[i], "--n-gen") == 0) {
            opt.nGen = std::atoi(next().c_str());
        } else if (std::strcmp(argv[i], "--reps") == 0) {
            opt.reps = std::atoi(next().c_str());
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            opt.threads = std::atoi(next().c_str());
        } else if (std::strcmp(argv[i], "--gpu") == 0) {
            // Accepted in every build on purpose: "unknown option: --gpu" is what
            // a CPU-only binary used to say, and it reads like a typo rather than
            // "this build has no offload engine".  The clear error comes below.
            opt.forceGpu = 1;
        } else if (std::strcmp(argv[i], "--cpu") == 0 ||
                   std::strcmp(argv[i], "--no-gpu") == 0) {
            opt.forceGpu = 0;
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 2;
        }
    }

    if (opt.modelPath.empty()) {
        const char *env = std::getenv("TINYCODER_MODEL_PATH");
        if (env != nullptr && env[0] != '\0') {
            opt.modelPath = env;
        } else {
            opt.modelPath = "/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf";
        }
    }
    if (opt.modelPath.empty()) {
        opt.modelPath = "/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf";
    }
    if (opt.forceGpu == 1) {
#ifdef USE_CUDA
        // The default in a CUDA build; --gpu also overrides a TINYCODER_GPU=0
        // opt-out.  Must happen before Model::load() so weights are uploaded
        // once, at first forward.
        std::fprintf(stderr, "[bench] CUDA offload engine requested\n");
        setenv("TINYCODER_GPU", "1", 1);
#else
        std::fprintf(stderr,
                     "[bench] --gpu needs a CUDA build, and this binary has none: "
                     "reconfigure with -DENABLE_CUDA=ON (the default when nvcc is "
                     "installed) and rebuild.\n");
        return 2;
#endif
    } else if (opt.forceGpu == 0) {
        // CPU baseline of the same run: the engine reads this at load time.
        std::fprintf(stderr, "[bench] CPU path forced (no GPU offload)\n");
        setenv("TINYCODER_GPU", "0", 1);
    }
    if (opt.reps < 1) opt.reps = 1;

    size_t threads = opt.threads > 0
                             ? static_cast<size_t>(opt.threads)
                             : tinycoder::ThreadPool::recommendedThreadCount();
    tinycoder::ThreadPool::instance().initialize(threads);

    std::fprintf(stderr, "Loading model: %s\n", opt.modelPath.c_str());
    tinycoder::Model model;
    std::string err;
    // The window has to be requested before load: it sizes the caches.
    if (opt.ctx > 0 && !model.setContextLength(opt.ctx)) {
        std::fprintf(stderr, "FATAL: cannot set context length %u\n", opt.ctx);
        return 1;
    }
    if (!model.load(opt.modelPath, &err)) {
        std::fprintf(stderr, "FATAL: model load failed: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "  %s, %u layers, hidden %u, intermediate %u, vocab %u\n",
                 model.config().modelName.c_str(), model.config().numLayers,
                 model.config().hiddenSize, model.config().intermediateSize,
                 model.config().vocabSize);

    const auto prompt = buildPrompt(opt.nPrompts);
    const int32_t vocab = static_cast<int32_t>(model.config().vocabSize);

    std::fprintf(stderr,
                 "Bench: pp=%d tg=%d reps=%d threads=%zu  (greedy, decode-only "
                 "timing, warmup+reps)\n\n",
                 opt.nPrompts, opt.nGen, opt.reps, threads);

    // ---- Warmup rep ----
    // The warmup is the COLD first-touch pass (page cache miss): on a 35 GB
    // qwen35moe model in <35 GB RAM it is 15-23x slower than the warm reps and
    // matches the COLD numbers the question tests used to report — the exact
    // mismatch ([bench-parity] below reconciles the two harnesses).
    double coldTg = 0.0, coldPp = 0.0;
    {
        double prefillMs = 0.0;
        auto r = runOneRep(model, prompt, opt.nGen, prefillMs, vocab, "cold");
        coldTg = r.tokens / (r.ms / 1000.0);
        coldPp = opt.nPrompts / (prefillMs / 1000.0);
        std::fprintf(stderr,
                     "  warmup (COLD): %d tg tok in %.1f ms (%.1f tg tok/s); "
                     "prefill %d tok in %.1f ms (%.1f pp tok/s)\n",
                     r.tokens, r.ms, coldTg, opt.nPrompts, prefillMs, coldPp);
    }

    // ---- Measured reps ----
    std::vector<double> tokPerSec;
    std::vector<double> decodeMs;
    std::vector<double> prefillTokPerSec;
    std::vector<double> prefillMsVec;
    for (int r = 0; r < opt.reps; ++r) {
        double prefillMs = 0.0;
        char repLabel[32];
        std::snprintf(repLabel, sizeof(repLabel), "rep %d", r + 1);
        auto res = runOneRep(model, prompt, opt.nGen, prefillMs, vocab, repLabel);
        prefillMsVec.push_back(prefillMs);
        double rate = res.tokens / (res.ms / 1000.0);
        double ppRate = opt.nPrompts / (prefillMs / 1000.0);
        tokPerSec.push_back(rate);
        decodeMs.push_back(res.ms);
        prefillTokPerSec.push_back(ppRate);
        std::fprintf(stderr,
                     "  rep %d: %d tok in %.1f ms (%.1f tg tok/s); prefill %d tok "
                     "in %.1f ms (%.1f pp tok/s)\n",
                     r + 1, res.tokens, res.ms, rate, opt.nPrompts, prefillMs,
                     ppRate);
    }

    // ---- Aggregate (mean ± stdev over reps) ----
    double mean = 0.0;
    for (double v: tokPerSec) mean += v;
    mean /= static_cast<double>(opt.reps);
    double var = 0.0;
    for (double v: tokPerSec) var += (v - mean) * (v - mean);
    var /= static_cast<double>(std::max(1, opt.reps - 1));
    double stdev = std::sqrt(var);

    double meanMs = 0.0;
    for (double v: decodeMs) meanMs += v;
    meanMs /= static_cast<double>(opt.reps);

    double meanPrefillMs = 0.0;
    for (double v: prefillMsVec) meanPrefillMs += v;
    meanPrefillMs /= static_cast<double>(opt.reps);

    double meanPp = 0.0;
    for (double v: prefillTokPerSec) meanPp += v;
    meanPp /= static_cast<double>(opt.reps);

    std::printf("tg%d | %d threads | %.2f +/- %.2f tok/s | %.1f ms/decode (mean) || "
                "pp%d | %.1f +/- %.1f tok/s | %.1f ms (mean)\n",
                opt.nGen, static_cast<int>(threads), mean, stdev, meanMs,
                opt.nPrompts, meanPp, 0.0, meanPrefillMs);

    // Bench-parity summary: the two harnesses (this bench and the gtest
    // question tests) intentionally report the SAME quantity — decode-only
    // tg tok/s — with the cache state made explicit.  The question tests
    // warm the page cache once (SharedTestEnv) and emit
    //   [bench-parity] generation N tok @ X tg tok/s (...warm/cold)
    // so bench-warm should match test-warm within measurement noise.
    std::printf(
            "[bench-parity] pp %d tok @ %.2f tok/s (cold warmup) || %.2f +/- %.2f "
            "tok/s (warm reps) || tg %d tok @ %.2f tok/s (cold warmup) || "
            "%.2f +/- %.2f tok/s (warm reps)\n",
            opt.nPrompts, coldPp, meanPp, 0.0, opt.nGen, coldTg, mean, stdev);

    return 0;
}
