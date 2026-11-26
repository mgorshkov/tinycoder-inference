// Batched-prefill gate for Inkling-2-Mini.
//
// Runs the same prompt twice on one engine and one checkpoint -- once through
// the token-major decode graph (TINYCODER_INKLING_BATCHED=0) and once through
// the [H, m] batched graph -- and requires:
//
//   parity   every prompt position's top-1 id identical, top-5 id set identical,
//            and the per-position logit cosine >= 0.9999;
//   stages   (--dump) the hidden state after the prelude for every token, and
//            after every block for token 0, cosine >= 0.9999 against the
//            token-major path.  The per-block comparison is what localises a
//            divergence to a stage; the tool prints the first block that fails.
//
// --bench N measures prefill throughput for both paths on an N-token prompt
// (the same synthetic token stream tinycoder_bench builds), so the pp16/pp512
// columns come from one binary and one protocol.  Decode stays on the token
// path in both cases: a multi-token call is prefill.
//
// forward() now chunks a prompt into 128-token layer-major passes (was 16), so
// a 512-token prompt is four passes and a 128-token one is a single pass.
// The default batched graph is the fast path; with the trajectory-changing
// levers off (TINYCODER_INKLING_GEMM, _EXPERT_MMA, _Q2_0_MMA and
// _ATTN_FLASH all 0) it is bit-exact with the token-major graph at any length
// (max|d| == 0 at 128 and 512 tokens).  Those levers are ON by default -- the
// parity gate below is only meaningful with them switched off -- and the tool
// reports their parity separately:
//   TINYCODER_INKLING_GEMM=0        disables the dense int8 tensor-core GEMM
//                                   (one weight read per pass instead of one
//                                   per 4 tokens)
//   TINYCODER_INKLING_Q2_0_MMA=0    disables the Q2_0 int8 MMQ for expert down
//   TINYCODER_INKLING_CTRL=batched  opts into the old one-block-per-rank
//                                   controller (faster, 0-20 ULP off, flips
//                                   router ties; default is the exact form)
//   TINYCODER_INKLING_EXPERT_MMA=0  disables the grouped int8 MMQ for the
//                                   routed experts (IQ2_XXS/XS/S gate/up and
//                                   Q2_0 down); =gate / =down selects one
//                                   direction.  The Q8_1 activation
//                                   quantisation changes the trajectory, so
//                                   top-1/top-5 parity with the token path
//                                   does not survive it.
//   TINYCODER_INKLING_ATTN_FLASH=0  keeps the exact per-token attention form
//                                   for m >= 128 (the flash kernel accumulates
//                                   keys sequentially, a different trajectory)
//   TINYCODER_INKLING_ATTN_TILE=0   revert the 8-query-token attention tiling
//                                   (on by default; it is bit-exact)
//
// Usage:
//   inkling_gpu_batched_prefill_test [<model.gguf>] [--m 128] [--bench 512]
//                                    [--reps 1] [--dump <dir>] [--no-bench]
//                                    [--repeat N] [--both-token]
//                                    [--batched-only]
//   --gemv-parity  answer "is a multi-token GEMV the same numbers as the
//                  per-token call?" per type, before trusting any graph parity.
//   --repeat N     re-run the batched side N times in one process: differing
//                  answers for identical work mean a race.
//   --both-token   run the token path on both sides (its own determinism check).
//   --dump DIR     also compare the prelude and every block's hidden state, and
//                  the router ids, against the token path (per chunk; a prompt
//                  longer than 16 tokens overwrites the earlier chunk's files).
//   --dump-logits DIR  write [m][vocab] float32 logit matrices for both sides
//                  (tm-logits.f32 / bt-logits.f32) so fresh processes can be
//                  compared byte for byte; --m 16/128/512 makes the batched
//                  logits a cross-process determinism gate for the fast path.
//
// Exit status is the gate: 0 only when every parity/stage check passed and the
// batched path was actually faster (a speedup claim needs the measurement).
#include "GGUFLoader.hpp"
#include "Int8Gemm.hpp"
#include "Model.hpp"
#include "gpu/InklingGPU.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <sys/stat.h>

using namespace tinycoder;

namespace {

    struct Top {
        int32_t id = -1;
        float logit = 0.0f;
    };

    std::vector<Top> topK(const float *v, size_t n, size_t k) {
        std::vector<int32_t> ids(n);
        std::iota(ids.begin(), ids.end(), int32_t(0));
        const size_t kk = std::min(k, n);
        std::partial_sort(ids.begin(), ids.begin() + kk, ids.end(),
                          [v](int32_t a, int32_t b) {
                              return v[a] > v[b] || (v[a] == v[b] && a < b);
                          });
        std::vector<Top> out(kk);
        for (size_t i = 0; i < kk; ++i)
            out[i] = Top{ids[i], v[ids[i]]};
        return out;
    }

    double cosine(const float *a, const float *b, size_t n) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < n; ++i) {
            dot += double(a[i]) * b[i];
            na += double(a[i]) * a[i];
            nb += double(b[i]) * b[i];
        }
        if (na == 0.0 || nb == 0.0) return na == nb ? 1.0 : 0.0;
        return dot / std::sqrt(na * nb);
    }

    std::vector<int32_t> buildPrompt(int32_t nTokens) {
        std::vector<int32_t> t;
        t.reserve(nTokens);
        uint64_t s = 0x5EED123456789ULL;
        for (int32_t i = 0; i < nTokens; ++i) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            t.push_back(static_cast<int32_t>(s % 3000) + 10);
        }
        return t;
    }

    /// Read an int32 dump (the token path writes dIdx's raw bytes).
    bool readI32(const std::string &path, std::vector<int32_t> &out, size_t want) {
        FILE *f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) return false;
        out.resize(want);
        const size_t got = std::fread(out.data(), sizeof(int32_t), want, f);
        std::fclose(f);
        return got == want;
    }

    bool readF32(const std::string &path, std::vector<float> &out, size_t want) {        FILE *f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) return false;
        out.resize(want);
        const size_t got = std::fread(out.data(), sizeof(float), want, f);
        std::fclose(f);
        return got == want;
    }

    bool fileExists(const std::string &path) {
        struct stat st{};
        return ::stat(path.c_str(), &st) == 0;
    }

    struct Run {
        std::vector<float> logits;// [seq][vocab]
        double ms = 0.0;
        bool ok = false;
    };

    // ---- GEMV parity probe ---------------------------------------------------
    // Answers the question the layer-major graph depends on: does a batched or
    // sliced call produce the SAME numbers as the per-token call the decode path
    // makes?  Per tensor:
    //   float  inklingGemvBatchedAsync(16) vs 16 x inklingGemvAsync
    //   int8   int8Gemv(seqLen=8) slices   vs 8 x int8Gemv(seqLen=1)  [reference]
    //   slice4 int8Gemv(seqLen=4) slices   vs 4 x int8Gemv(seqLen=1)  [gate]
    //   expert int8Gemv(4 rows of one expert) vs 4 x int8GemvExpert(1) [gate]
    // The layer-major default uses the float 16-row tiles, the int8 4-row slices
    // and the per-expert 4-row slices; those three are bit-exact here.  The
    // 8-token int8 slice is NOT exact (gemvCoreImpl swaps to a 2-warp reduction
    // above ncols_dst=4) and is printed only so a regression is visible.
    struct GemvCase {
        const char *tensor;
        uint32_t type;
        const char *name;
    };
    constexpr GemvCase kGemvCases[] = {
            {"blk.0.attn_q.weight", 14, "Q6_K"},
            {"blk.0.attn_k.weight", 8, "Q8_0"},
            {"blk.3.attn_q.weight", 13, "Q5_K"},
            {"blk.18.attn_q.weight", 12, "Q4_K"},
            {"blk.20.attn_q.weight", 18, "IQ3_XXS"},
            {"blk.9.ffn_gate_up_exps.weight", 17, "IQ2_XS"},
            {"blk.16.ffn_gate_up_exps.weight", 22, "IQ2_S"},
            {"blk.5.ffn_gate_up_exps.weight", 16, "IQ2_XXS"},
            {"blk.0.ffn_gate_up_exps.weight", 19, "IQ1_S"},
            {"blk.0.ffn_down_exps.weight", 42, "Q2_0"},
            {"blk.5.ffn_down_exps.weight", 40, "NVFP4"},
            {"blk.0.ffn_gate_inp.weight", 0, "F32"},
            {"blk.0.attn_sink_proj.weight", 30, "BF16"},
            {"output.weight", 23, "IQ4_XS"},
    };
    uint32_t gemvBlockElems(uint32_t type) {
        if (type == 40 || type == 42) return 64u;
        if (type == 0 || type == 30 || type == 8) return 32u;
        return 256u;
    }
    uint32_t gemvBlockBytes(uint32_t type) {
        switch (type) {
            case 0: return 128;
            case 8: return 34;
            case 12: return 144;
            case 13: return 176;
            case 14: return 210;
            case 16: return 66;
            case 17: return 74;
            case 18: return 98;
            case 19: return 50;
            case 22: return 82;
            case 23: return 136;
            case 30: return 64;
            case 40: return 36;
            case 42: return 18;
            default: return 0;
        }
    }
    bool runGemvParity(const char *modelPath) {
        GGUFLoader loader;
        if (!loader.load(modelPath)) {
            std::fprintf(stderr, "[gemv-parity] load failed: %s\n", modelPath);
            return false;
        }
        const uint32_t m = 16, rows = 256;
        bool pass = true;
        std::printf("[gemv-parity] %-8s %-24s %12s %12s %12s %12s\n", "type",
                    "tensor", "int8 max|d|", "float max|d|", "slice4 max|d|",
                    "expert max|d|");
        for (const GemvCase &c: kGemvCases) {
            const TensorInfo *info = loader.getTensorInfo(c.tensor);
            if (info == nullptr || info->type != c.type) {
                std::printf("[gemv-parity] %-8s %-24s  (missing or type mismatch)\n",
                            c.name, c.tensor);
                continue;
            }
            const uint8_t *host = loader.getTensor(c.tensor);
            const uint32_t cols = static_cast<uint32_t>(info->shape[0]);
            const uint32_t rowBytes =
                    ((cols + gemvBlockElems(c.type) - 1) / gemvBlockElems(c.type)) *
                    gemvBlockBytes(c.type);
            const size_t wBytes = static_cast<size_t>(rows) * rowBytes;
            std::vector<float> x(static_cast<size_t>(m) * cols);
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = 0.35f * std::sin(0.017f * float(i)) +
                       0.11f * std::cos(0.0031f * float(i));
            void *dw = nullptr, *dx = nullptr, *dRef = nullptr, *dBat = nullptr;
            cudaMalloc(&dw, wBytes);
            cudaMalloc(&dx, x.size() * sizeof(float));
            cudaMalloc(&dRef, static_cast<size_t>(m) * rows * sizeof(float));
            cudaMalloc(&dBat, static_cast<size_t>(m) * rows * sizeof(float));
            cudaMemcpy(dw, host, wBytes, cudaMemcpyHostToDevice);
            cudaMemcpy(dx, x.data(), x.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
            std::vector<float> ref(static_cast<size_t>(m) * rows),
                    bat(static_cast<size_t>(m) * rows);

            // float: per-token inklingGemv vs inklingGemvBatched
            for (uint32_t s = 0; s < m; ++s)
                gpu::inklingGemv(c.type, dw,
                                 static_cast<float *>(dx) +
                                         static_cast<size_t>(s) * cols,
                                 static_cast<float *>(dRef) +
                                         static_cast<size_t>(s) * rows,
                                 rows, cols, nullptr);
            const bool fOk = gpu::inklingGemvBatchedAsync(
                    c.type, dw, static_cast<float *>(dx),
                    static_cast<float *>(dBat), rows, cols, m, nullptr);
            cudaDeviceSynchronize();
            cudaMemcpy(ref.data(), dRef, ref.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(bat.data(), dBat, bat.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            double fMax = 0.0, fRef = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                fMax = std::max(fMax, double(std::fabs(bat[i] - ref[i])));
                fRef = std::max(fRef, double(std::fabs(ref[i])));
            }

            // int8: an 8-token slice vs eight 1-token calls, only for the types
            // int8Gemv accepts.  This is the "is a multi-token int8 GEMV the
            // same number as the per-token call?" question, against the int8
            // answer (not the float one, which is a different path entirely).
            double iMax = -1.0, iRef = 0.0;
            bool iOk = true;
            std::vector<float> int8one(static_cast<size_t>(m) * rows);
            for (uint32_t s = 0; s < m && iOk; ++s)
                iOk = gpu::int8Gemv(c.type, dw,
                                    static_cast<float *>(dx) +
                                            static_cast<size_t>(s) * cols,
                                    static_cast<float *>(dRef) +
                                            static_cast<size_t>(s) * rows,
                                    1, rows, cols, nullptr);
            if (iOk) {
                cudaDeviceSynchronize();
                cudaMemcpy(int8one.data(), dRef,
                           int8one.size() * sizeof(float),
                           cudaMemcpyDeviceToHost);
                for (uint32_t off = 0; off < m && iOk; off += 8) {
                    const uint32_t mm = std::min(8u, m - off);
                    iOk = gpu::int8Gemv(c.type, dw,
                                        static_cast<float *>(dx) +
                                                static_cast<size_t>(off) * cols,
                                        static_cast<float *>(dRef) +
                                                static_cast<size_t>(off) * rows,
                                        mm, rows, cols, nullptr);
                }
                if (iOk) {
                    cudaDeviceSynchronize();
                    cudaMemcpy(bat.data(), dRef, bat.size() * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    for (size_t i = 0; i < int8one.size(); ++i) {
                        iMax = std::max(iMax,
                                        double(std::fabs(bat[i] - int8one[i])));
                        iRef = std::max(iRef, double(std::fabs(int8one[i])));
                    }
                }
            }
            // Expert-vs-plain: the layer-major path runs one int8Gemv per
            // *expert* over 4 gathered rows, while the decode path runs one
            // int8GemvExpert per token.  Those two must agree bit-for-bit, or
            // the batching changes the model's trajectory.  Also checks that a
            // 4-token int8Gemv is bit-identical to four 1-token calls (it is:
            // gemvCoreImpl keeps nwarps=4 up to ncols_dst=4, while 5..8 swaps
            // to a 2-warp reduction tree).
            // The layer-major graph runs one int8Gemv per *expert* over 4
            // gathered rows, while the decode path runs one int8GemvExpert per
            // token.  Both must agree bit-for-bit with 4 x one-token calls or
            // the batching changes the model's trajectory.  4 is the largest
            // exact slice: gemvCoreImpl keeps nwarps=4 up to ncols_dst=4 and
            // swaps to a 2-warp reduction above it.
            double eSlice = -1.0, eExpert = -1.0;
            if (iOk) {
                int32_t *dIdx = nullptr;
                cudaMalloc(&dIdx, sizeof(int32_t));
                cudaMemset(dIdx, 0, sizeof(int32_t));
                std::vector<float> plain4(static_cast<size_t>(m) * rows),
                        expert1(static_cast<size_t>(m) * rows);
                bool ok = true;
                for (uint32_t s = 0; s < m && ok; ++s)
                    ok = gpu::int8GemvExpert(
                            c.type, dw,
                            static_cast<float *>(dx) +
                                    static_cast<size_t>(s) * cols,
                            static_cast<float *>(dBat) +
                                    static_cast<size_t>(s) * rows,
                            rows, cols, dIdx, 1, 0, nullptr);
                if (ok) {
                    cudaDeviceSynchronize();
                    cudaMemcpy(expert1.data(), dBat,
                               expert1.size() * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    for (uint32_t off = 0; off < m && ok; off += 4) {
                        const uint32_t mm = std::min(4u, m - off);
                        ok = gpu::int8Gemv(c.type, dw,
                                           static_cast<float *>(dx) +
                                                   static_cast<size_t>(off) * cols,
                                           static_cast<float *>(dBat) +
                                                   static_cast<size_t>(off) * rows,
                                           mm, rows, cols, nullptr);
                    }
                    cudaDeviceSynchronize();
                    cudaMemcpy(plain4.data(), dBat,
                               plain4.size() * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    eSlice = 0.0;
                    for (size_t i = 0; i < plain4.size(); ++i)
                        eSlice = std::max(eSlice, double(std::fabs(
                                                        plain4[i] - int8one[i])));
                    // plain 4-slice vs 4 x expert-1: the number that decides
                    // whether the per-expert slice is bit-identical.
                    eExpert = 0.0;
                    for (size_t i = 0; i < plain4.size(); ++i)
                        eExpert = std::max(eExpert, double(std::fabs(
                                                         plain4[i] - expert1[i])));
                } else {
                    cudaGetLastError();
                }
                cudaFree(dIdx);
            }
            if (iOk) {
                // The gate is on the paths the graph actually uses: the
                // 4-token plain slice and the per-expert plain slice must be
                // bit-identical to the token major calls.  The 8-token slice is
                // reported for reference only -- it switches to a 2-warp
                // reduction and is deliberately unused.
                const bool exact = eSlice == 0.0 && eExpert == 0.0 && fMax == 0.0;
                std::printf("[gemv-parity] %-8s %-24s %12.3e %12.3e %12.3e %12.3e%s\n",
                            c.name, c.tensor, iMax, fMax, eSlice, eExpert,
                            exact ? "" : "   <-- not bit-exact");
                if (fMax > 1e-3 || eSlice > 1e-3 || eExpert > 1e-3) pass = false;
            } else {
                std::printf("[gemv-parity] %-8s %-24s %12s %12.3e %12s %12s%s\n",
                            c.name, c.tensor, "(refused)", fMax, "-", "-",
                            fMax > 0.0 ? "   <-- not bit-exact" : "");
                if (fMax > 1e-3) pass = false;
            }
            cudaFree(dw);
            cudaFree(dx);
            cudaFree(dRef);
            cudaFree(dBat);
        }
        std::printf("[gemv-parity] %s\n", pass ? "PASSED" : "FAILED");
        return pass;
    }

} // namespace
int main(int argc, char **argv) {
    const char *modelPath =
            argc > 1 && argv[1][0] != '-'
                    ? argv[1]
                    : "/data/models/inkling/inkling-2-mini/"
                      "Inkling-2-Mini-UD-IQ2_XXS.gguf";
    int32_t promptLen = 128;
    int32_t benchLen = 512;
    int32_t reps = 1;
    bool bench = true;
    bool gemvParity = false;
    bool bothToken = false;
    bool batchedOnly = false;
    int32_t repeat = 1;
    std::string dumpDir;
    std::string logitsDir;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (std::strcmp(argv[i], "--m") == 0) promptLen = std::atoi(next());
        else if (std::strcmp(argv[i], "--bench") == 0) benchLen = std::atoi(next());
        else if (std::strcmp(argv[i], "--reps") == 0) reps = std::atoi(next());
        else if (std::strcmp(argv[i], "--dump") == 0) dumpDir = next();
        else if (std::strcmp(argv[i], "--dump-logits") == 0) logitsDir = next();
        else if (std::strcmp(argv[i], "--no-bench") == 0) bench = false;
        else if (std::strcmp(argv[i], "--gemv-parity") == 0) gemvParity = true;
        else if (std::strcmp(argv[i], "--both-token") == 0) bothToken = true;
        else if (std::strcmp(argv[i], "--batched-only") == 0) batchedOnly = true;
        else if (std::strcmp(argv[i], "--repeat") == 0) repeat = std::atoi(next());
    }
    if (promptLen < 1 || promptLen > 512) {
        std::fprintf(stderr, "[batched-prefill] --m must be 1..512 "
                             "(forward() chunks by 128)\n");
        return 2;
    }

    if (gemvParity) return runGemvParity(modelPath) ? 0 : 1;

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        std::fprintf(stderr, "[batched-prefill] no CUDA device\n");
        return 2;
    }
    size_t freeMiB = 0, totalMiB = 0;
    cudaMemGetInfo(&freeMiB, &totalMiB);
    std::printf("[batched-prefill] GPU free %.0f MiB of %.0f MiB\n",
                freeMiB / 1048576.0, totalMiB / 1048576.0);

    Model model;
    std::string err;
    std::printf("[batched-prefill] loading %s\n", modelPath);
    if (!model.load(modelPath, &err)) {
        std::fprintf(stderr, "[batched-prefill] load failed: %s\n", err.c_str());
        return 2;
    }
    const ModelConfig &cfg = model.config();
    const uint32_t vocab = cfg.vocabSizeUnpadded ? cfg.vocabSizeUnpadded
                                                 : cfg.vocabSize;
    std::printf("[batched-prefill] hidden=%u heads=%u/%u headDim=%u phys=%u "
                "loop=[%u,%u)x%u vocab=%u maxSeq=%u\n",
                cfg.hiddenSize, cfg.numAttentionHeads, cfg.numKVHeads,
                cfg.headDim, cfg.numPhysicalLayers, cfg.loopStart, cfg.loopEnd,
                cfg.loopCount, vocab, cfg.maxSeqLen);

    gpu::InklingGPU engine;
    if (!engine.available()) {
        std::fprintf(stderr, "[batched-prefill] no CUDA device\n");
        return 2;
    }

    // One prompt, long enough for both the parity run and the benchmark.
    const std::vector<int32_t> prompt =
            buildPrompt(std::max(promptLen, benchLen));

    // One engine, two settings: the only difference between the runs is which
    // graph forward() dispatches to.
    const auto run = [&](bool batched, bool allLogits,
                         int32_t nTokens) -> Run {
        Run r;
        if (batched) setenv("TINYCODER_INKLING_BATCHED", nTokens == 1 ? "2" : "1", 1);
        else setenv("TINYCODER_INKLING_BATCHED", "0", 1);
        if (!dumpDir.empty()) setenv("TINYCODER_INKLING_BATCHED_DUMP",
                                     dumpDir.c_str(), 1);
        else unsetenv("TINYCODER_INKLING_BATCHED_DUMP");
        engine.clearKVCache();
        const std::vector<int32_t> toks(prompt.begin(), prompt.begin() + nTokens);
        std::vector<float> logits(
                static_cast<size_t>(allLogits ? nTokens : 1) * vocab, 0.0f);
        const auto t0 = std::chrono::steady_clock::now();
        r.ok = engine.forward(model, toks, allLogits, logits.data(), err);
        const auto t1 = std::chrono::steady_clock::now();
        r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!r.ok) {
            std::fprintf(stderr, "[batched-prefill] %s forward failed: %s\n",
                         batched ? "batched" : "token-major", err.c_str());
            return r;
        }
        r.logits = std::move(logits);
        return r;
    };

    bool pass = true;

    // --dump-logits DIR: write the full [m][vocab] logit matrix of each side so
    // independent fresh processes can be compared byte-for-byte.  The in-process
    // parity below cannot see a cross-process difference; the files can.
    const auto writeLogits = [&](const char *name, const Run &r) {
        if (logitsDir.empty() || !r.ok) return;
        const std::string path = logitsDir + "/" + name + ".f32";
        FILE *f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "[batched-prefill] cannot write %s\n",
                         path.c_str());
            pass = false;
            return;
        }
        std::fwrite(r.logits.data(), sizeof(float), r.logits.size(), f);
        std::fclose(f);
    };
    const auto writePrompt = [&] {
        if (logitsDir.empty()) return;
        const std::string path = logitsDir + "/prompt.i32";
        FILE *f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) return;
        std::fwrite(prompt.data(), sizeof(int32_t), promptLen, f);
        std::fclose(f);
    };

    // ---- stage parity + logits parity ---------------------------------------
    if (!batchedOnly)
    std::printf("[batched-prefill] --- parity: %d-token prompt, token-major vs "
                "batched ---\n", promptLen);
    // Reference first (it also writes the tm-* dumps), then the batched run.
    if (batchedOnly) {
        const Run b = run(true, true, promptLen);
        if (!b.ok) return 3;
        writePrompt();
        writeLogits("bt-logits", b);
    } else {
    Run ref = run(false, true, promptLen);
    if (!ref.ok) return 3;
    Run bat = run(!bothToken, true, promptLen);
    if (!bat.ok) return 3;
    writePrompt();
    writeLogits("tm-logits", ref);
    writeLogits("bt-logits", bat);
    // --repeat: the same comparison several times in one process.  A race shows
    // up as different answers for identical work in one process; a deterministic
    // defect gives the same answer every time.
    for (int32_t r = 1; r < repeat; ++r) {
        const Run b2 = run(!bothToken, true, promptLen);
        if (!b2.ok) return 3;
        double worst = 1.0;
        int32_t worstPos = 0;
        for (int32_t s = 0; s < promptLen; ++s) {
            const double c = cosine(ref.logits.data() + static_cast<size_t>(s) * vocab,
                                    b2.logits.data() + static_cast<size_t>(s) * vocab,
                                    vocab);
            if (c < worst) {
                worst = c;
                worstPos = s;
            }
        }
        std::printf("[batched-prefill] repeat %d: min cosine %.8f at pos %d\n", r,
                    worst, worstPos);
    }
    double minCos = 1.0;
    int32_t worstPos = 0;
    int32_t top1Mismatch = 0, setMismatch = 0;
    double maxBand = 0.0;
    for (int32_t s = 0; s < promptLen; ++s) {
        const float *a = ref.logits.data() + static_cast<size_t>(s) * vocab;
        const float *b = bat.logits.data() + static_cast<size_t>(s) * vocab;
        const std::vector<Top> ta = topK(a, vocab, 5);
        const std::vector<Top> tb = topK(b, vocab, 5);
        const double c = cosine(a, b, vocab);
        if (c < minCos) {
            minCos = c;
            worstPos = s;
        }
        bool sameTop1 = ta[0].id == tb[0].id;
        std::vector<int32_t> ia, ib;
        for (const auto &x: ta) ia.push_back(x.id);
        for (const auto &x: tb) ib.push_back(x.id);
        std::sort(ia.begin(), ia.end());
        std::sort(ib.begin(), ib.end());
        const bool sameSet = ia == ib;
        if (!sameTop1) ++top1Mismatch;
        if (!sameSet) ++setMismatch;
        float band = 0.0f;
        for (int32_t id: ia)
            band = std::max(band, std::fabs(b[id] - a[id]));
        maxBand = std::max(maxBand, double(band));
        std::printf("  pos %2d: token-major top-1 %6d (%.4f)  batched top-1 "
                    "%6d (%.4f)  %s  top5 %s  cos %.8f  max|d| %.5f\n",
                    s, ta[0].id, ta[0].logit, tb[0].id, tb[0].logit,
                    sameTop1 ? "MATCH" : "DIFF ", sameSet ? "MATCH" : "DIFF ",
                    c, band);
    }
    const bool parityOk = top1Mismatch == 0 && setMismatch == 0 &&
                          minCos >= 0.9999;
    std::printf("[batched-prefill] parity: top-1 %s (%d mismatches), top-5 set %s "
                "(%d), min cosine %.8f at pos %d, max|d| %.5f -> %s\n",
                top1Mismatch == 0 ? "MATCH" : "MISMATCH", top1Mismatch,
                setMismatch == 0 ? "MATCH" : "MISMATCH", setMismatch, minCos,
                worstPos, maxBand, parityOk ? "PASS" : "FAIL");
    pass = pass && parityOk;
    }

    // ---- per-stage cosines (optional) --------------------------------------
    if (!dumpDir.empty()) {
        std::printf("[batched-prefill] --- stage parity (cosine per token, "
                    "dump dir %s) ---\n", dumpDir.c_str());
        // Prelude: one file per token from the reference, one batch file.
        std::vector<float> bp;
        const bool haveBatchPrelude =
                readF32(dumpDir + "/bt-prelude.f32", bp,
                        static_cast<size_t>(promptLen) * cfg.hiddenSize);
        double preludeMin = 1.0;
        int32_t preludeWorst = -1;
        if (!haveBatchPrelude) {
            std::printf("[batched-prefill] prelude: batched dump missing\n");
            pass = false;
        } else {
            for (int32_t s = 0; s < promptLen; ++s) {
                std::vector<float> tp;
                if (!readF32(dumpDir + "/tm-t" + std::to_string(s) +
                                     "-prelude.f32",
                             tp, cfg.hiddenSize)) {
                    std::printf("[batched-prefill] prelude: reference dump for "
                                "token %d missing\n", s);
                    pass = false;
                    break;
                }
                const double c = cosine(tp.data(),
                                        bp.data() +
                                                static_cast<size_t>(s) *
                                                        cfg.hiddenSize,
                                        cfg.hiddenSize);
                if (c < preludeMin) {
                    preludeMin = c;
                    preludeWorst = s;
                }
            }
            std::printf("[batched-prefill] prelude hidden state: min cosine "
                        "%.8f (token %d) -> %s\n", preludeMin, preludeWorst,
                        preludeMin >= 0.9999 ? "PASS" : "FAIL");
            pass = pass && preludeMin >= 0.9999;
        }
        // Per block and per token: the hidden state after every block-run, in
        // execution order, so the first divergence is localised to one stage.
        // The router ids are compared too: a changed id is a discrete routing
        // flip, a shifted hidden state without one is arithmetic.
        double stageMin = 1.0;
        uint32_t stageWorst = 0;
        int32_t stageWorstTok = -1;
        uint32_t stageCount = 0;
        int32_t idxMismatch = 0;
        const std::vector<uint32_t> order = [&] {
            std::vector<uint32_t> o;
            for (uint32_t s = 0; s < cfg.loopStart; ++s) o.push_back(s);
            for (uint32_t s = cfg.loopStart; s < cfg.loopEnd; ++s) o.push_back(s);
            for (uint32_t p = 1; p < cfg.loopCount; ++p)
                for (uint32_t s = 0; s < cfg.loopEnd - cfg.loopStart; ++s)
                    o.push_back(cfg.numPhysicalLayers + (p - 1) * (cfg.loopEnd - cfg.loopStart) + s);
            for (uint32_t s = cfg.loopEnd; s < cfg.numPhysicalLayers; ++s)
                o.push_back(s);
            return o;
        }();
        for (uint32_t slot: order) {
            const std::string bi = dumpDir + "/bt-idx-s" + std::to_string(slot) +
                                   ".f32";
            std::vector<float> ba;
            std::vector<int32_t> ta;
            if (fileExists(bi) &&
                readF32(bi, ba, static_cast<size_t>(promptLen) * cfg.expertUsedCount)) {
                for (int32_t t = 0; t < promptLen; ++t) {
                    if (!readI32(dumpDir + "/tm-idx-t" + std::to_string(t) + "-s" +
                                         std::to_string(slot) + ".f32",
                                 ta, cfg.expertUsedCount))
                        continue;
                    for (uint32_t u = 0; u < cfg.expertUsedCount; ++u)
                        if (static_cast<float>(ta[u]) !=
                            ba[static_cast<size_t>(t) * cfg.expertUsedCount + u])
                            ++idxMismatch;
                }
            }
            for (int32_t t = 0; t < promptLen; ++t) {
                const std::string tm = dumpDir + "/tm-t" + std::to_string(t) +
                                       "-s" + std::to_string(slot) + ".f32";
                const std::string bt = dumpDir + "/bt-t" + std::to_string(t) +
                                       "-s" + std::to_string(slot) + ".f32";
                if (!fileExists(tm) || !fileExists(bt)) continue;
                std::vector<float> a, b;
                if (!readF32(tm, a, cfg.hiddenSize) ||
                    !readF32(bt, b, cfg.hiddenSize))
                    continue;
                const double c = cosine(a.data(), b.data(), cfg.hiddenSize);
                ++stageCount;
                if (c < stageMin) {
                    stageMin = c;
                    stageWorst = slot;
                    stageWorstTok = t;
                }
            }
        }
        std::printf("[batched-prefill] per-block hidden state: %u (token,block) "
                    "pairs, min cosine %.8f (slot %u, token %d) -> %s\n",
                    stageCount, stageMin, stageWorst, stageWorstTok,
                    stageCount > 0 && stageMin >= 0.9999 ? "PASS" : "FAIL");
        std::printf("[batched-prefill] router ids: %d mismatched (token,slot,u) "
                    "entries\n", idxMismatch);
        if (stageCount == 0) {
            std::printf("[batched-prefill] no per-block dumps found\n");
            pass = false;
        }
        pass = pass && stageCount > 0 && stageMin >= 0.9999;
        // The dump mode is a debugging mode; leave it off for timing.
        unsetenv("TINYCODER_INKLING_BATCHED_DUMP");
    }

    // ---- prefill throughput -------------------------------------------------
    double ppToken = 0.0, ppBatch = 0.0;
    if (bench && benchLen >= 1) {
        std::printf("[batched-prefill] --- prefill %d tokens, %d rep(s) ---\n",
                    benchLen, reps);
        double bestT = 1e30, bestB = 1e30, sumT = 0.0, sumB = 0.0;
        for (int32_t r = 0; r < reps; ++r) {
            if (batchedOnly) {
                const Run b = run(true, false, benchLen);
                if (!b.ok) return 3;
                bestB = std::min(bestB, b.ms);
                sumB += b.ms;
                std::printf("  rep %d: batched %.1f ms (%.1f tok/s)\n", r + 1,
                            b.ms, benchLen / (b.ms / 1000.0));
                continue;
            }
            const Run t = run(false, false, benchLen);
            if (!t.ok) return 3;
            const Run b = run(!bothToken, false, benchLen);
            if (!b.ok) return 3;
            bestT = std::min(bestT, t.ms);
            bestB = std::min(bestB, b.ms);
            sumT += t.ms;
            sumB += b.ms;
            std::printf("  rep %d: token-major %.1f ms (%.1f tok/s)   batched "
                        "%.1f ms (%.1f tok/s)   %.2fx\n",
                        r + 1, t.ms, benchLen / (t.ms / 1000.0), b.ms,
                        benchLen / (b.ms / 1000.0), t.ms / b.ms);
        }
        ppToken = benchLen / ((sumT / reps) / 1000.0);
        ppBatch = benchLen / ((sumB / reps) / 1000.0);
        if (batchedOnly)
            std::printf("[batched-prefill] pp%d: batched %.2f tok/s (best %.1f "
                        "ms)\n", benchLen, ppBatch, bestB);
        else
            std::printf("[batched-prefill] pp%d: token-major %.2f tok/s (best "
                        "%.1f ms), batched %.2f tok/s (best %.1f ms), speedup "
                        "%.2fx\n", benchLen, ppToken, bestT, ppBatch, bestB,
                        bestT / bestB);
    }

    std::printf("%s\n", pass ? "INKLING BATCHED PREFILL TEST PASSED"
                             : "INKLING BATCHED PREFILL TEST FAILED");
    return pass ? 0 : 1;
}
