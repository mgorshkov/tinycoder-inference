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

// Measured prototype: can "dequantize an expert layer to FP32 once, then run
// cuBLAS SGEMM over the token columns" beat the current grouped float GEMV for
// the routed MoE?
//
// It answers, in this order:
//   1. dequant exactness: the device IQ1_S/Q2_0 kernels against
//      GGMLDequantize::dequantize on the real blk.0 tensors (max|d| must be 0);
//   2. dequant and SGEMM rates on their own (GB/s, TFLOPS) for 8 and 128
//      experts, at m = 8/16/32/64/128/512 token columns;
//   3. the same shapes through the current path
//      (inklingGemvBatchedGroupedAsync);
//   4. one real MoE block: the engine's own dumped hidden state for m=128
//      tokens is replayed through the exact grouped path (which must reproduce
//      the engine's dumped moeOut bit-exactly) and through the FP32
//      dequant+SGEMM path, then compared per token;
//   5. the real routed row counts, which are the crux: at m=128 the top-8
//      experts hold only part of the 1024 (token,expert) pairs, the rest are
//      spread over 30-116 distinct experts, so a per-layer FP32 copy moves 20x
//      the quantized bytes.  The per-slot cost uses those real counts;
//   6. a projected pp512 under two models (drop-in at the engine's m=128, and a
//      hypothetical single m=512 pass) plus the VRAM a full-layer FP32 scratch
//      needs.
//
// Nothing here is wired into the shipped graph.
//
// Usage:
//   inkling_fp32_gemm_probe [<model.gguf>] [--dump <dir>] [--m 128]
//       [--reps 5] [--slots 0,12,23,47] [--reuse-dump] [--keep-dump]
//
// Exit status: 0 when the dequant is bit-exact against the host reference and
// the block replay reproduces the engine (the numeric preconditions); the speed
// verdict is printed, not gated.

#include "GGMLDequantize.hpp"
#include "Model.hpp"
#include "gpu/InklingFp32Probe.hpp"
#include "gpu/InklingGPU.hpp"
#include "gpu/InklingKernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <sys/stat.h>

using namespace tinycoder;

namespace {

    // ---------------------------------------------------------------------
    // Small helpers
    // ---------------------------------------------------------------------
    [[noreturn]] void die(const char *what, const char *detail = nullptr) {
        std::fprintf(stderr, "[fp32-probe] FATAL: %s%s%s\n", what,
                     detail != nullptr ? ": " : "",
                     detail != nullptr ? detail : "");
        std::exit(2);
    }

    void checkCuda(const char *what) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) die(what, cudaGetErrorString(e));
    }

    void *devAlloc(size_t bytes, const char *what) {
        void *p = nullptr;
        const cudaError_t e = cudaMalloc(&p, bytes);
        if (e != cudaSuccess) {
            cudaGetLastError();
            size_t mf = 0, mt = 0;
            cudaMemGetInfo(&mf, &mt);
            std::fprintf(stderr,
                         "[fp32-probe] cudaMalloc(%s, %.1f MiB) failed: %s "
                         "[free %.0f of %.0f MiB]\n",
                         what, static_cast<double>(bytes) / 1048576.0,
                         cudaGetErrorString(e), mf / 1048576.0,
                         mt / 1048576.0);
            std::exit(2);
        }
        return p;
    }

    void up(void *dst, const void *src, size_t bytes, const char *what) {
        if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            die(what);
    }

    void down(void *dst, const void *src, size_t bytes, const char *what) {
        if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
            die(what);
    }

    struct Timer {
        cudaEvent_t a{}, b{};
        Timer() {
            cudaEventCreate(&a);
            cudaEventCreate(&b);
        }
        ~Timer() {
            cudaEventDestroy(a);
            cudaEventDestroy(b);
        }
        void start(cudaStream_t s) { cudaEventRecord(a, s); }
        double stop(cudaStream_t s) {
            cudaEventRecord(b, s);
            cudaEventSynchronize(b);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            return static_cast<double>(ms);
        }
    };

    double median(const std::vector<double> &v) {
        if (v.empty()) return 0.0;
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        return s[s.size() / 2];
    }

    bool readFile(const std::string &path, void *dst, size_t bytes) {
        FILE *f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) return false;
        const size_t got = std::fread(dst, 1, bytes, f);
        std::fclose(f);
        return got == bytes;
    }

    bool readF32(const std::string &path, std::vector<float> &v, size_t want) {
        v.resize(want);
        return readFile(path, v.data(), want * sizeof(float));
    }

    /// bt-idx-s<slot>.f32 holds the router ids as FLOAT (batchDumpI32 writes
    /// "ids as float, exact below 2^24"), not as int32.
    bool readIdxF32(const std::string &path, std::vector<int32_t> &v, size_t want) {
        std::vector<float> f;
        if (!readF32(path, f, want)) return false;
        v.resize(want);
        for (size_t i = 0; i < want; ++i)
            v[i] = static_cast<int32_t>(std::lrint(f[i]));
        return true;
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

    struct CountStats {
        uint32_t distinct = 0;
        int32_t minCount = 0;
        int32_t medCount = 0;
        int32_t maxCount = 0;
        int64_t sum = 0;
        int32_t top8 = 0;
    };

    CountStats countStats(const std::vector<int32_t> &counts) {
        CountStats s;
        std::vector<int32_t> live;
        for (int32_t c: counts) {
            if (c > 0) {
                live.push_back(c);
                ++s.distinct;
                s.sum += c;
            }
        }
        if (!live.empty()) {
            std::sort(live.begin(), live.end());
            s.minCount = live.front();
            s.medCount = live[live.size() / 2];
            s.maxCount = live.back();
            for (size_t i = live.size() > 8 ? live.size() - 8 : 0; i < live.size(); ++i)
                s.top8 += live[i];
        }
        return s;
    }

    struct Options {
        std::string model =
                "/data/models/inkling/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";
        std::string dump = "/tmp/inkling_fp32_probe_dump";
        int32_t m = 128;
        int reps = 5;
        std::vector<uint32_t> slots = {0u, 12u, 23u, 47u};
        bool reuseDump = false;
        bool keepDump = false;
    };

}// namespace

int main(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (std::strcmp(a, "--dump") == 0) opt.dump = next();
        else if (std::strcmp(a, "--m") == 0) opt.m = std::atoi(next());
        else if (std::strcmp(a, "--reps") == 0) opt.reps = std::atoi(next());
        else if (std::strcmp(a, "--reuse-dump") == 0) opt.reuseDump = true;
        else if (std::strcmp(a, "--keep-dump") == 0) opt.keepDump = true;
        else if (std::strcmp(a, "--slots") == 0) {
            opt.slots.clear();
            const char *p = next();
            if (std::strcmp(p, "all") == 0) {
                for (uint32_t sl = 0; sl < 48u; ++sl) opt.slots.push_back(sl);
                continue;
            }
            while (*p) {
                char *end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p) break;
                opt.slots.push_back(static_cast<uint32_t>(v));
                p = end;
                while (*p == ',' || *p == ' ') ++p;
            }
        } else if (a[0] != '-') {
            opt.model = a;
        } else {
            std::fprintf(stderr, "[fp32-probe] unknown argument: %s\n", a);
            return 2;
        }
    }
    if (opt.m < 8 || opt.m > 128) {
        std::fprintf(stderr, "[fp32-probe] --m must be 8..128 (the engine's "
                             "batch pass is 128 tokens)\n");
        return 2;
    }
    if (opt.reps < 1) opt.reps = 1;

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0)
        die("no CUDA device");

    // ---------------------------------------------------------------------
    // 0. Load the checkpoint, run one real batched pass (also the current
    //    pp128 measurement) and dump the block inputs this probe replays.
    // ---------------------------------------------------------------------
    Model model;
    std::string err;
    std::printf("[fp32-probe] loading %s\n", opt.model.c_str());
    if (!model.load(opt.model, &err)) die("model load", err.c_str());
    const ModelConfig &cfg = model.config();
    const uint32_t H = cfg.hiddenSize;
    const uint32_t nExp = cfg.expertCount;
    const uint32_t nUsed = cfg.expertUsedCount;
    const uint32_t ff = cfg.expertFeedForwardLength;
    const uint32_t vocab = cfg.vocabSizeUnpadded ? cfg.vocabSizeUnpadded
                                                 : cfg.vocabSize;
    const auto &layers = model.debugGetLayers();
    if (layers.empty()) die("no layers");
    const auto &w0 = layers[0];
    const auto &gu = w0.ffnGateUpExps;
    const auto &dn = w0.ffnDownExps;
    if (gu.type != 19u || dn.type != 42u) {
        std::fprintf(stderr,
                     "[fp32-probe] blk.0 experts are type %u/%u, this probe "
                     "targets IQ1_S(19)/Q2_0(42)\n",
                     gu.type, dn.type);
        return 2;
    }
    const uint32_t rowsGu = 2u * ff;// 2304
    const uint32_t rowsDn = H;      // 2304
    const uint32_t colsDn = ff;     // 1152
    if (gu.rows != nExp * rowsGu || gu.cols != H || dn.rows != nExp * rowsDn ||
        dn.cols != colsDn)
        die("unexpected blk.0 expert tensor shape");
    const uint32_t rowBytesGu = (H / 256u) * 50u;
    const uint32_t rowBytesDn = (colsDn / 64u) * 18u;
    if (gu.data.size() < static_cast<uint64_t>(nExp) * rowsGu * rowBytesGu)
        die("gate/up tensor smaller than its expert stride");
    if (dn.data.size() < static_cast<uint64_t>(nExp) * rowsDn * rowBytesDn)
        die("down tensor smaller than its expert stride");
    const size_t guBytes = static_cast<size_t>(nExp) * rowsGu * rowBytesGu;
    const size_t dnBytes = static_cast<size_t>(nExp) * rowsDn * rowBytesDn;
    std::printf("[fp32-probe] hidden=%u experts=%u used=%u ff=%u  "
                "blk.0 gate/up %ux%u IQ1_S (%.0f KB/expert, %.0f MB all)  "
                "down %ux%u Q2_0 (%.0f KB/expert, %.0f MB all)\n",
                H, nExp, nUsed, ff, rowsGu, H,
                static_cast<double>(rowsGu) * rowBytesGu / 1024.0,
                static_cast<double>(guBytes) / 1048576.0, rowsDn, colsDn,
                static_cast<double>(rowsDn) * rowBytesDn / 1024.0,
                static_cast<double>(dnBytes) / 1048576.0);

    mkdir(opt.dump.c_str(), 0755);
    double currentPassMs = 0.0;
    size_t graphFreeMiB = 0;
    size_t gpuTotalMiB = 0;
    {
        gpu::InklingGPU engine;
        if (!engine.available()) die("InklingGPU not available");
        if (!opt.reuseDump) {
            const std::vector<int32_t> toks = buildPrompt(opt.m);
            {
                // The warm-up slice must not read past a short prompt: --m is
                // allowed down to 8, and slicing 16 out of an 8-token vector is
                // out of bounds (it segfaulted).
                const size_t warmN = std::min<size_t>(16u, toks.size());
                std::vector<int32_t> warm(toks.begin(), toks.begin() + warmN);
                std::vector<float> wl(vocab);
                engine.clearKVCache();
                if (!engine.forwardBatch(model, warm, false, wl.data(), err))
                    die("batched warmup forward", err.c_str());
            }
            // The timed pass follows the bench protocol exactly: no dumps, no
            // allLogits (the prefill number is for the last position only).
            std::vector<float> lastLogits(vocab);
            engine.clearKVCache();
            const auto t0 = std::chrono::steady_clock::now();
            if (!engine.forwardBatch(model, toks, false, lastLogits.data(), err))
                die("batched timed pass", err.c_str());
            const auto t1 = std::chrono::steady_clock::now();
            currentPassMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            size_t free1 = 0, total = 0;
            cudaMemGetInfo(&free1, &total);
            graphFreeMiB = free1 / 1048576;
            gpuTotalMiB = total / 1048576;
            std::printf("[fp32-probe] current batched pass: m=%d %.1f ms "
                        "(%.1f tok/s)  graph live: free %.0f of %.0f MiB\n",
                        opt.m, currentPassMs, opt.m / (currentPassMs / 1e3),
                        static_cast<double>(graphFreeMiB),
                        static_cast<double>(gpuTotalMiB));
            // Separate dump pass: every dump call syncs the stream, so its
            // timing is meaningless and is not reported.
            setenv("TINYCODER_INKLING_BATCHED_DUMP", opt.dump.c_str(), 1);
            std::vector<float> logits(static_cast<size_t>(opt.m) * vocab, 0.0f);
            engine.clearKVCache();
            if (!engine.forwardBatch(model, toks, true, logits.data(), err))
                die("batched forward (dump pass)", err.c_str());
            unsetenv("TINYCODER_INKLING_BATCHED_DUMP");
        } else {
            std::printf("[fp32-probe] reusing the dump at %s\n", opt.dump.c_str());
        }
    }// engine destroyed here: its ~9 GiB upload must be gone
    cudaDeviceSynchronize();

    std::vector<float> row;

    // ---------------------------------------------------------------------
    // 1. Upload the real blk.0 expert tensors + per-expert scales.
    // ---------------------------------------------------------------------
    void *guDev = devAlloc(guBytes, "blk.0 gate/up experts");
    void *dnDev = devAlloc(dnBytes, "blk.0 down experts");
    up(guDev, gu.data.data(), guBytes, "upload gate/up");
    up(dnDev, dn.data.data(), dnBytes, "upload down");

    std::vector<float> gateUpScale(nExp, 1.0f), downScale(nExp, 1.0f);
    std::vector<float> probsBias(nExp, 0.0f);
    if (w0.inkling.expertGateUpScale.size() >= nExp)
        std::copy_n(w0.inkling.expertGateUpScale.data(), nExp, gateUpScale.begin());
    if (w0.inkling.expertDownScale.size() >= nExp)
        std::copy_n(w0.inkling.expertDownScale.data(), nExp, downScale.begin());
    if (w0.inkling.expertProbsBias.size() >= nExp)
        std::copy_n(w0.inkling.expertProbsBias.data(), nExp, probsBias.begin());
    float *gateUpScaleDev = static_cast<float *>(devAlloc(nExp * 4, "gateUpScale"));
    float *downScaleDev = static_cast<float *>(devAlloc(nExp * 4, "downScale"));
    float *probsBiasDev = static_cast<float *>(devAlloc(nExp * 4, "probsBias"));
    up(gateUpScaleDev, gateUpScale.data(), nExp * 4, "upload gateUpScale");
    up(downScaleDev, downScale.data(), nExp * 4, "upload downScale");
    up(probsBiasDev, probsBias.data(), nExp * 4, "upload probsBias");

    // ---------------------------------------------------------------------
    // 2. Dequant exactness against the host reference.
    // ---------------------------------------------------------------------
    bool dequantExact = true;
    {
        float *dW = static_cast<float *>(devAlloc(
                static_cast<size_t>(rowsGu) * H * 4, "one expert gate/up FP32"));
        float *dWd = static_cast<float *>(devAlloc(
                static_cast<size_t>(rowsDn) * colsDn * 4, "one expert down FP32"));
        for (uint32_t e: {0u, 7u, nExp - 1u}) {
            const uint8_t *guHost = gu.data.data() +
                                    static_cast<uint64_t>(e) * rowsGu * rowBytesGu;
            if (!gpu::inklingFp32Dequant(
                        19u,
                        static_cast<const uint8_t *>(guDev) +
                                static_cast<uint64_t>(e) * rowsGu * rowBytesGu,
                        dW, static_cast<uint64_t>(rowsGu) * H, nullptr))
                die("dequant gate/up launch");
            std::vector<float> got(static_cast<size_t>(rowsGu) * H);
            down(got.data(), dW, got.size() * 4, "read dequant gate/up");
            const std::vector<float> ref = GGMLDequantize::dequantizeIQ1_S(
                    guHost, static_cast<uint64_t>(rowsGu) * H);
            if (ref.size() != got.size()) die("host dequant size");
            double maxAbs = 0.0;
            size_t worst = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
                if (d > maxAbs) {
                    maxAbs = d;
                    worst = i;
                }
            }
            if (maxAbs != 0.0) dequantExact = false;
            std::printf("[fp32-probe] dequant exactness gate/up expert %2u: "
                        "max|d| = %.3e at %zu%s\n",
                        e, maxAbs, worst,
                        maxAbs == 0.0 ? " (bit-exact)" : "  <-- MISMATCH");
        }
        {
            const uint8_t *dnHost = dn.data.data();
            if (!gpu::inklingFp32Dequant(42u, dnDev, dWd,
                                          static_cast<uint64_t>(rowsDn) * colsDn,
                                          nullptr))
                die("dequant down launch");
            std::vector<float> got(static_cast<size_t>(rowsDn) * colsDn);
            down(got.data(), dWd, got.size() * 4, "read dequant down");
            const std::vector<float> ref = GGMLDequantize::dequantizeQ2_0(
                    dnHost, static_cast<uint64_t>(rowsDn) * colsDn);
            if (ref.size() != got.size()) die("host dequant size (down)");
            double maxAbs = 0.0;
            for (size_t i = 0; i < got.size(); ++i)
                maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(got[i]) -
                                                    ref[i]));
            if (maxAbs != 0.0) dequantExact = false;
            std::printf("[fp32-probe] dequant exactness down  expert  0: "
                        "max|d| = %.3e%s\n",
                        maxAbs, maxAbs == 0.0 ? " (bit-exact)" : "  <-- MISMATCH");
        }
        cudaFree(dW);
        cudaFree(dWd);
    }

    // ---------------------------------------------------------------------
    // 3. Dequant rate: 8 experts and all 128, written to a full-size FP32
    //    buffer (that allocation is also the VRAM answer for section 8).
    //    Traffic = quantized bytes read + FP32 bytes written.
    // ---------------------------------------------------------------------
    std::printf("\n[fp32-probe] === dequant to FP32 (device kernel) ===\n");
    {
        float *dW = static_cast<float *>(devAlloc(
                static_cast<size_t>(nExp) * rowsGu * H * 4, "gate/up FP32 (all)"));
        float *dWd = static_cast<float *>(devAlloc(
                static_cast<size_t>(nExp) * rowsDn * colsDn * 4, "down FP32 (all)"));
        const auto benchDequant = [&](const char *name, uint32_t nExps,
                                      uint32_t type, const void *w,
                                      uint64_t elemsPerExpert, uint32_t rows,
                                      uint32_t rowBytes, float *out) -> double {
            const double deqBytes = static_cast<double>(nExps) * rows * rowBytes;
            const double f32Bytes = static_cast<double>(nExps) * elemsPerExpert * 4.0;
            std::vector<double> ms;
            for (int r = 0; r < opt.reps; ++r) {
                Timer t;
                t.start(nullptr);
                for (uint32_t e = 0; e < nExps; ++e)
                    if (!gpu::inklingFp32Dequant(
                                type,
                                static_cast<const uint8_t *>(w) +
                                        static_cast<uint64_t>(e) * rows * rowBytes,
                                out + static_cast<uint64_t>(e) * elemsPerExpert,
                                elemsPerExpert, nullptr))
                        die("dequant bench launch");
                ms.push_back(t.stop(nullptr));
            }
            const double best = *std::min_element(ms.begin(), ms.end());
            const double gb = (deqBytes + f32Bytes) / 1e9;
            std::printf("[fp32-probe] %-14s experts=%3u  %8.3f ms  %7.1f GB/s "
                        "(%6.2f GB moved: %.3f read + %.3f written)\n",
                        name, nExps, best, gb / (best * 1e-3), gb, deqBytes / 1e9,
                        f32Bytes / 1e9);
            return best;
        };
        benchDequant("gate/up IQ1_S", 8u, 19u, guDev,
                     static_cast<uint64_t>(rowsGu) * H, rowsGu, rowBytesGu, dW);
        benchDequant("gate/up IQ1_S", nExp, 19u, guDev,
                     static_cast<uint64_t>(rowsGu) * H, rowsGu, rowBytesGu, dW);
        benchDequant("down Q2_0", 8u, 42u, dnDev,
                     static_cast<uint64_t>(rowsDn) * colsDn, rowsDn,
                     rowBytesDn, dWd);
        benchDequant("down Q2_0", nExp, 42u, dnDev,
                     static_cast<uint64_t>(rowsDn) * colsDn, rowsDn,
                     rowBytesDn, dWd);
        cudaFree(dW);
        cudaFree(dWd);
    }

    // ---------------------------------------------------------------------
    // 4. SGEMM rate vs the current grouped GEMV for the same shapes.
    //    "8 experts x m token columns" is the hypothesis' shape; its FLOPs
    //    equal the real m*nUsed routed pairs, but the real router spreads
    //    those pairs over 30-128 distinct experts (section 6).
    // ---------------------------------------------------------------------
    std::printf("\n[fp32-probe] === GEMM: 8 experts x m token columns ===\n");
    {
        const uint32_t nExps = 8;
        float *dW8 = static_cast<float *>(devAlloc(
                static_cast<size_t>(nExps) * rowsGu * H * 4, "gate/up 8 FP32"));
        float *dW8d = static_cast<float *>(devAlloc(
                static_cast<size_t>(nExps) * rowsDn * colsDn * 4, "down 8 FP32"));
        for (uint32_t e = 0; e < nExps; ++e) {
            if (!gpu::inklingFp32Dequant(
                        19u,
                        static_cast<const uint8_t *>(guDev) +
                                static_cast<uint64_t>(e) * rowsGu * rowBytesGu,
                        dW8 + static_cast<uint64_t>(e) * rowsGu * H,
                        static_cast<uint64_t>(rowsGu) * H, nullptr))
                die("pre-dequant gate/up");
            if (!gpu::inklingFp32Dequant(
                        42u,
                        static_cast<const uint8_t *>(dnDev) +
                                static_cast<uint64_t>(e) * rowsDn * rowBytesDn,
                        dW8d + static_cast<uint64_t>(e) * rowsDn * colsDn,
                        static_cast<uint64_t>(rowsDn) * colsDn, nullptr))
                die("pre-dequant down");
        }
        const uint32_t mMax = 512;
        float *xC = static_cast<float *>(devAlloc(
                static_cast<size_t>(mMax) * H * 4, "gemm x"));
        float *outC = static_cast<float *>(devAlloc(
                static_cast<size_t>(nExps) * mMax * rowsGu * 4, "gemm out"));
        // Real token rows from the dump pass (cycled when m > dump rows).
        std::vector<float> xHost(static_cast<size_t>(mMax) * H);
        {
            const uint32_t have = static_cast<uint32_t>(opt.m);
            for (uint32_t t = 0; t < mMax; ++t) {
                char path[1024];
                std::snprintf(path, sizeof(path), "%s/nrm-t%u-s0.f32",
                              opt.dump.c_str(), t % have);
                if (!readF32(path, row, H))
                    die("missing dump row for SGEMM input", path);
                std::memcpy(xHost.data() + static_cast<size_t>(t) * H, row.data(),
                            H * sizeof(float));
            }
            up(xC, xHost.data(), xHost.size() * 4, "upload gemm x");
        }
        {
            // Verify the cuBLAS argument mapping against a host fp64 dot:
            // expert 0, token 0, the first 8 output rows.
            const std::vector<float> wHost = GGMLDequantize::dequantizeIQ1_S(
                    gu.data.data(), static_cast<uint64_t>(rowsGu) * H);
            if (!gpu::inklingFp32Gemm(dW8, xC, outC, rowsGu, H, 1, nullptr))
                die("sgemm self-test");
            std::vector<float> got(rowsGu);
            down(got.data(), outC, rowsGu * 4, "read sgemm self-test");
            double maxAbs = 0.0, maxRef = 0.0;
            for (uint32_t r = 0; r < 8; ++r) {
                double ref = 0.0;
                for (uint32_t k = 0; k < H; ++k)
                    ref += static_cast<double>(
                                   wHost[static_cast<size_t>(r) * H + k]) *
                           xHost[k];
                maxAbs = std::max(maxAbs, std::fabs(got[r] - ref));
                maxRef = std::max(maxRef, std::fabs(ref));
            }
            std::printf("[fp32-probe] sgemm mapping self-test: 8 rows, max|d| "
                        "%.2e (max|ref| %.2f) -> %s\n",
                        maxAbs, maxRef,
                        maxAbs <= 1e-3 * maxRef ? "OK" : "MISMATCH");
            if (maxAbs > 1e-3 * maxRef) die("sgemm argument mapping is wrong");
        }
        float *xG = static_cast<float *>(devAlloc(
                static_cast<size_t>(nExps) * mMax * rowsGu * 4, "grouped x"));
        float *outG = static_cast<float *>(devAlloc(
                static_cast<size_t>(nExps) * mMax * rowsGu * 4, "grouped out"));
        int32_t *counts8 = static_cast<int32_t *>(devAlloc(nExps * 4, "counts8"));

        const auto benchGemm = [&](const char *name, uint32_t m, const float *w8,
                                   uint32_t rows, uint32_t cols, const float *x,
                                   float *out) {
            for (uint32_t e = 0; e < nExps; ++e)// untimed warm call
                gpu::inklingFp32Gemm(w8 + static_cast<uint64_t>(e) * rows * cols,
                                     x, out + static_cast<uint64_t>(e) * m * rows,
                                     rows, cols, m, nullptr);
            std::vector<double> ms;
            for (int r = 0; r < opt.reps; ++r) {
                Timer t;
                t.start(nullptr);
                for (uint32_t e = 0; e < nExps; ++e)
                    if (!gpu::inklingFp32Gemm(
                                w8 + static_cast<uint64_t>(e) * rows * cols, x,
                                out + static_cast<uint64_t>(e) * m * rows, rows,
                                cols, m, nullptr))
                        die("sgemm launch");
                ms.push_back(t.stop(nullptr));
            }
            const double best = *std::min_element(ms.begin(), ms.end());
            const double tflops =
                    2.0 * nExps * m * rows * cols / (best * 1e-3) / 1e12;
            std::printf("[fp32-probe] %-14s m=%3u n=%4u k=%4u  %8.3f ms  %6.2f "
                        "TFLOPS  (cuBLAS SGEMM)\n",
                        name, m, rows, cols, best, tflops);
        };
        const auto benchGrouped = [&](const char *name, uint32_t m, uint32_t type,
                                      const void *w, uint32_t rows, uint32_t cols,
                                      const float *x, float *out) {
            std::vector<int32_t> counts(nExps, static_cast<int32_t>(m));
            up(counts8, counts.data(), nExps * 4, "upload counts8");
            std::vector<double> ms;
            for (int r = 0; r < opt.reps; ++r) {
                Timer t;
                t.start(nullptr);
                if (!gpu::inklingGemvBatchedGroupedAsync(type, w, x, out, rows,
                                                        cols, m, counts8, nExps,
                                                        nullptr))
                    die("grouped gemv launch");
                ms.push_back(t.stop(nullptr));
            }
            const double best = *std::min_element(ms.begin(), ms.end());
            const double tflops =
                    2.0 * nExps * m * rows * cols / (best * 1e-3) / 1e12;
            std::printf("[fp32-probe] %-14s m=%3u n=%4u k=%4u  %8.3f ms  %6.2f "
                        "TFLOPS  (current grouped float GEMV)\n",
                        name, m, rows, cols, best, tflops);
        };
        const auto fillGroupedX = [&](uint32_t m, uint32_t cols) {
            std::vector<float> tmp(static_cast<size_t>(nExps) * m * cols);
            for (uint32_t e = 0; e < nExps; ++e)
                for (uint32_t t = 0; t < m; ++t)
                    std::memcpy(tmp.data() +
                                        (static_cast<size_t>(e) * m + t) * cols,
                                xHost.data() + static_cast<size_t>(t) * H,
                                cols * sizeof(float));
            up(xG, tmp.data(), tmp.size() * 4, "upload grouped x");
        };

        std::printf("[fp32-probe] -- gate/up (n=2304, k=2304) --\n");
        for (uint32_t m: {8u, 16u, 32u, 64u, 128u, 512u}) {
            benchGemm("sgemm", m, dW8, rowsGu, H, xC, outC);
            fillGroupedX(m, H);
            benchGrouped("grouped-float", m, 19u, guDev, rowsGu, H, xG, outG);
        }
        std::printf("[fp32-probe] -- down (n=2304, k=1152) --\n");
        for (uint32_t m: {8u, 16u, 32u, 64u, 128u, 512u}) {
            benchGemm("sgemm", m, dW8d, rowsDn, colsDn, xC, outC);
            fillGroupedX(m, colsDn);
            benchGrouped("grouped-float", m, 42u, dnDev, rowsDn, colsDn, xG, outG);
        }
        cudaFree(dW8);
        cudaFree(dW8d);
        cudaFree(xC);
        cudaFree(outC);
        cudaFree(xG);
        cudaFree(outG);
        cudaFree(counts8);
    }

    // ---------------------------------------------------------------------
    // 5. One real MoE block, exact grouped path vs FP32 dequant+SGEMM.
    // ---------------------------------------------------------------------
    std::printf("\n[fp32-probe] === one real MoE block (m=%d tokens, real "
                "routing) ===\n", opt.m);
    const uint32_t cap = static_cast<uint32_t>(opt.m);
    const uint32_t m = static_cast<uint32_t>(opt.m);
    float *xDev = static_cast<float *>(devAlloc(static_cast<size_t>(m) * H * 4, "x"));
    float *rouDev = static_cast<float *>(devAlloc(static_cast<size_t>(m) * nExp * 4, "router logits"));
    int32_t *idxDev = static_cast<int32_t *>(devAlloc(static_cast<size_t>(m) * nUsed * 4, "idx"));
    float *wgtDev = static_cast<float *>(devAlloc(static_cast<size_t>(m) * nUsed * 4, "wgt"));
    int32_t *gposDev = static_cast<int32_t *>(devAlloc(static_cast<size_t>(m) * nUsed * 4, "gpos"));
    int32_t *countsDev = static_cast<int32_t *>(devAlloc(nExp * 4, "counts"));
    int32_t *gtokDev = static_cast<int32_t *>(devAlloc(static_cast<size_t>(nExp) * cap * 4, "gtok"));
    int32_t *gexpDev = static_cast<int32_t *>(devAlloc(static_cast<size_t>(nExp) * cap * 4, "gexp"));
    int32_t *goffDev = static_cast<int32_t *>(devAlloc((nExp + 1) * 4, "goff"));
    float *gwgtDev = static_cast<float *>(devAlloc(static_cast<size_t>(nExp) * cap * 4, "gwgt"));
    float *gx = static_cast<float *>(devAlloc(static_cast<size_t>(nExp) * cap * H * 4, "gx"));
    float *gout = static_cast<float *>(devAlloc(static_cast<size_t>(nExp) * cap * rowsGu * 4, "gout"));
    float *gact = static_cast<float *>(devAlloc(static_cast<size_t>(nExp) * cap * ff * 4, "gact"));
    float *moeExact = static_cast<float *>(devAlloc(static_cast<size_t>(m) * H * 4, "moeExact"));
    float *gxIn = static_cast<float *>(devAlloc(static_cast<size_t>(nExp) * cap * H * 4, "gxIn"));
    float *gx2 = static_cast<float *>(devAlloc(static_cast<size_t>(nExp) * cap * H * 4, "gx2"));
    float *gout2 = static_cast<float *>(devAlloc(static_cast<size_t>(nExp) * cap * rowsGu * 4, "gout2"));
    float *gact2 = static_cast<float *>(devAlloc(static_cast<size_t>(nExp) * cap * ff * 4, "gact2"));
    float *moeFp32 = static_cast<float *>(devAlloc(static_cast<size_t>(m) * H * 4, "moeFp32"));
    float *dW = static_cast<float *>(devAlloc(static_cast<size_t>(rowsGu) * H * 4, "one expert gate/up FP32"));
    float *dWd = static_cast<float *>(devAlloc(static_cast<size_t>(rowsDn) * colsDn * 4, "one expert down FP32"));

    double sumExact = 0.0, sumDeq = 0.0, sumGemm = 0.0, sumFp32 = 0.0;
    uint32_t measuredSlots = 0;
    double worstRelRms = 0.0, worstMaxDiff = 0.0, worstCos = 1.0;
    bool slotParityOk = true;
    std::printf("[fp32-probe] slot distinct  maxc  top8  exact-ms  deq-ms  "
                "gemm-ms  fp32-ms  fp32/exact  max|d|    min-cos\n");
    for (uint32_t slot: opt.slots) {
        std::vector<float> xHost(static_cast<size_t>(m) * H);
        std::vector<float> rouHost(static_cast<size_t>(m) * nExp);
        std::vector<int32_t> idxRef(static_cast<size_t>(m) * nUsed);
        std::vector<float> moeRef(static_cast<size_t>(m) * H);
        bool haveSlot = true;
        for (uint32_t t = 0; t < m && haveSlot; ++t) {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/nrm-t%u-s%u.f32",
                          opt.dump.c_str(), t, slot);
            if (!readF32(path, row, H)) haveSlot = false;
            else std::memcpy(xHost.data() + static_cast<size_t>(t) * H, row.data(),
                             H * sizeof(float));
            std::snprintf(path, sizeof(path), "%s/rou-t%u-s%u.f32",
                          opt.dump.c_str(), t, slot);
            if (haveSlot && !readF32(path, row, nExp)) haveSlot = false;
            else if (haveSlot)
                std::memcpy(rouHost.data() + static_cast<size_t>(t) * nExp,
                            row.data(), nExp * sizeof(float));
            std::snprintf(path, sizeof(path), "%s/moe-t%u-s%u.f32",
                          opt.dump.c_str(), t, slot);
            if (haveSlot && !readF32(path, row, H)) haveSlot = false;
            else if (haveSlot)
                std::memcpy(moeRef.data() + static_cast<size_t>(t) * H, row.data(),
                            H * sizeof(float));
        }
        {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/bt-idx-s%u.f32", opt.dump.c_str(),
                          slot);
            if (haveSlot && !readIdxF32(path, idxRef, idxRef.size())) haveSlot = false;
        }
        if (!haveSlot) {
            std::printf("[fp32-probe] slot %u: dumps missing, skipped\n", slot);
            slotParityOk = false;
            continue;
        }
        up(xDev, xHost.data(), xHost.size() * 4, "upload x");
        up(rouDev, rouHost.data(), rouHost.size() * 4, "upload rou");
        // Replay the router for the weights.  On slot 0 (blk.0, whose bias this
        // probe loaded) the replay must reproduce the engine's ids exactly; a
        // later slot carries a later layer's selection, so the engine's dumped
        // ids are the ones the block is run with below.
        gpu::kInklingRouterBatch(rouDev, probsBiasDev, idxDev, wgtDev, nExp, nUsed,
                                 cfg.expertWeightsScale, m, nullptr);
        checkCuda("router");
        std::vector<int32_t> idxGot(idxRef.size());
        down(idxGot.data(), idxDev, idxGot.size() * 4, "read idx");
        size_t idxMismatch = 0;
        for (size_t i = 0; i < idxGot.size(); ++i)
            if (idxGot[i] != idxRef[i]) ++idxMismatch;
        up(idxDev, idxRef.data(), idxRef.size() * 4, "upload engine idx");
        std::vector<float> wgtHost(static_cast<size_t>(m) * nUsed);
        down(wgtHost.data(), wgtDev, wgtHost.size() * 4, "read wgt");
        // The kernel entry points take explicit per-expert offsets: the engine
        // passes compact prefix offsets, this probe keeps the padded
        // [nGroups][capacity] layout it was written around, so offsets are
        // e*capacity and the group arrays are the same counting sort the old
        // in-kernel build produced.
        std::vector<int32_t> counts(nExp, 0);
        std::vector<int32_t> goff(nExp + 1), gexp(static_cast<size_t>(nExp) * cap, 0),
                gtok(static_cast<size_t>(nExp) * cap, 0), gpos(idxRef.size(), 0);
        std::vector<float> gwgt(static_cast<size_t>(nExp) * cap, 0.0f);
        for (int32_t e: idxRef) ++counts[e];
        for (uint32_t e = 0; e < nExp; ++e) goff[e] = static_cast<int32_t>(e * cap);
        goff[nExp] = static_cast<int32_t>(nExp * cap);
        {
            std::vector<int32_t> cur(goff.begin(), goff.begin() + nExp);
            for (uint32_t t = 0; t < m; ++t)
                for (uint32_t u = 0; u < nUsed; ++u) {
                    const uint32_t p = t * nUsed + u;
                    const uint32_t e = static_cast<uint32_t>(idxRef[p]);
                    const int32_t slot = cur[e]++;
                    gtok[slot] = static_cast<int32_t>(t);
                    gwgt[slot] = wgtHost[p];
                    gpos[p] = slot;
                    gexp[slot] = static_cast<int32_t>(e);
                }
        }
        up(countsDev, counts.data(), counts.size() * 4, "upload counts");
        up(goffDev, goff.data(), goff.size() * 4, "upload goff");
        up(gexpDev, gexp.data(), gexp.size() * 4, "upload gexp");
        up(gtokDev, gtok.data(), gtok.size() * 4, "upload gtok");
        up(gposDev, gpos.data(), gpos.size() * 4, "upload gpos");
        up(gwgtDev, gwgt.data(), gwgt.size() * 4, "upload gwgt");
        const CountStats cs = countStats(counts);
        // The entry point takes the compact pair count and no per-expert bounds
        // (it gathers every slot in [0, nPairs)).  This probe keeps the padded
        // [nGroups][capacity] layout, so the count is the whole padded array;
        // unused slots have gtok == 0 and are never read, exactly as before.
        gpu::kInklingGatherExpertX(xDev, gtokDev, nExp * cap, gx, H, nullptr);
        checkCuda("gather expert x");
        // The exact path below reuses `gx` for its down output, exactly as the
        // engine does (one arena buffer).  Keep the gathered input for the FP32
        // path in its own buffer.
        cudaMemcpyAsync(gxIn, gx, static_cast<size_t>(nExp) * cap * H * 4,
                        cudaMemcpyDeviceToDevice, nullptr);
        checkCuda("copy gathered x");

        // ---- exact path (the engine's default grouped float GEMV) ----------
        double exactSlotMs = 0.0;
        {
            Timer t;
            t.start(nullptr);
            if (!gpu::inklingGemvBatchedGroupedAsync(19u, guDev, gx, gout, rowsGu,
                                                     H, cap, countsDev, nExp,
                                                     nullptr))
                die("exact gate/up grouped gemv");
            const double msUp = t.stop(nullptr);
            gpu::kInklingSituGrouped(gout, gact, gexpDev, gateUpScaleDev,
                                     static_cast<uint32_t>(nExp) * cap, ff,
                                     cfg.situBeta, cfg.situLinearBeta, nullptr);
            Timer t2;
            t2.start(nullptr);
            if (!gpu::inklingGemvBatchedGroupedAsync(42u, dnDev, gact, gx, rowsDn,
                                                     colsDn, cap, countsDev, nExp,
                                                     nullptr))
                die("exact down grouped gemv");
            const double msDn = t2.stop(nullptr);
            gpu::kInklingMoeAccumGrouped(gx, idxDev, gposDev, wgtDev,
                                         downScaleDev, moeExact, nUsed, H, m,
                                         nullptr);
            checkCuda("exact accum");
            exactSlotMs = msUp + msDn;
        }
        std::vector<float> moeExactHost(static_cast<size_t>(m) * H);
        down(moeExactHost.data(), moeExact, moeExactHost.size() * 4,
             "read exact moe");
        double replayMax = 0.0;
        for (size_t i = 0; i < moeExactHost.size(); ++i)
            replayMax = std::max(replayMax,
                                 std::fabs(static_cast<double>(moeExactHost[i]) -
                                           moeRef[i]));

        // ---- FP32 dequant + SGEMM path -------------------------------------
        const auto fp32Fused = [&]() {
            Timer t;
            t.start(nullptr);
            for (uint32_t e = 0; e < nExp; ++e) {
                const int32_t cnt = counts[e];
                if (cnt <= 0) continue;
                const float *xe = gxIn + static_cast<size_t>(e) * cap * H;
                float *oe = gout2 + static_cast<size_t>(e) * cap * rowsGu;
                if (!gpu::inklingFp32Dequant(
                            19u,
                            static_cast<const uint8_t *>(guDev) +
                                    static_cast<uint64_t>(e) * rowsGu * rowBytesGu,
                            dW, static_cast<uint64_t>(rowsGu) * H, nullptr))
                    die("fp32 dequant gate/up");
                if (!gpu::inklingFp32Gemm(dW, xe, oe, rowsGu, H,
                                          static_cast<uint32_t>(cnt), nullptr))
                    die("fp32 sgemm gate/up");
            }
            gpu::kInklingSituGrouped(gout2, gact2, gexpDev, gateUpScaleDev,
                                     static_cast<uint32_t>(nExp) * cap, ff,
                                     cfg.situBeta, cfg.situLinearBeta, nullptr);
            for (uint32_t e = 0; e < nExp; ++e) {
                const int32_t cnt = counts[e];
                if (cnt <= 0) continue;
                const float *ae = gact2 + static_cast<size_t>(e) * cap * ff;
                float *oe = gx2 + static_cast<size_t>(e) * cap * H;
                if (!gpu::inklingFp32Dequant(
                            42u,
                            static_cast<const uint8_t *>(dnDev) +
                                    static_cast<uint64_t>(e) * rowsDn * rowBytesDn,
                            dWd, static_cast<uint64_t>(rowsDn) * colsDn, nullptr))
                    die("fp32 dequant down");
                if (!gpu::inklingFp32Gemm(dWd, ae, oe, rowsDn, colsDn,
                                          static_cast<uint32_t>(cnt), nullptr))
                    die("fp32 sgemm down");
            }
            gpu::kInklingMoeAccumGrouped(gx2, idxDev, gposDev, wgtDev,
                                         downScaleDev, moeFp32, nUsed, H, m,
                                         nullptr);
            checkCuda("fp32 accum");
            return t.stop(nullptr);
        };
        const auto fp32DequantOnly = [&]() {
            Timer t;
            t.start(nullptr);
            for (uint32_t e = 0; e < nExp; ++e) {
                if (counts[e] <= 0) continue;
                if (!gpu::inklingFp32Dequant(
                            19u,
                            static_cast<const uint8_t *>(guDev) +
                                    static_cast<uint64_t>(e) * rowsGu * rowBytesGu,
                            dW, static_cast<uint64_t>(rowsGu) * H, nullptr))
                    die("fp32 dequant gate/up (only)");
                if (!gpu::inklingFp32Dequant(
                            42u,
                            static_cast<const uint8_t *>(dnDev) +
                                    static_cast<uint64_t>(e) * rowsDn * rowBytesDn,
                            dWd, static_cast<uint64_t>(rowsDn) * colsDn, nullptr))
                    die("fp32 dequant down (only)");
            }
            return t.stop(nullptr);
        };
        const auto fp32GemmOnly = [&]() {
            Timer t;
            t.start(nullptr);
            for (uint32_t e = 0; e < nExp; ++e) {
                const int32_t cnt = counts[e];
                if (cnt <= 0) continue;
                const float *xe = gx + static_cast<size_t>(e) * cap * H;
                float *oe = gout2 + static_cast<size_t>(e) * cap * rowsGu;
                if (!gpu::inklingFp32Gemm(dW, xe, oe, rowsGu, H,
                                          static_cast<uint32_t>(cnt), nullptr))
                    die("fp32 sgemm gate/up (only)");
            }
            gpu::kInklingSituGrouped(gout2, gact2, gexpDev, gateUpScaleDev,
                                     static_cast<uint32_t>(nExp) * cap, ff,
                                     cfg.situBeta, cfg.situLinearBeta, nullptr);
            for (uint32_t e = 0; e < nExp; ++e) {
                const int32_t cnt = counts[e];
                if (cnt <= 0) continue;
                const float *ae = gact2 + static_cast<size_t>(e) * cap * ff;
                float *oe = gx2 + static_cast<size_t>(e) * cap * H;
                if (!gpu::inklingFp32Gemm(dWd, ae, oe, rowsDn, colsDn,
                                          static_cast<uint32_t>(cnt), nullptr))
                    die("fp32 sgemm down (only)");
            }
            gpu::kInklingMoeAccumGrouped(gx2, idxDev, gposDev, wgtDev,
                                         downScaleDev, moeFp32, nUsed, H, m,
                                         nullptr);
            checkCuda("fp32 accum (only)");
            return t.stop(nullptr);
        };
        fp32Fused();// warm caches, and leave a real FP32 result behind
        std::vector<double> tot, deq, gem;
        for (int r = 0; r < opt.reps; ++r) {
            deq.push_back(fp32DequantOnly());
            gem.push_back(fp32GemmOnly());
            tot.push_back(fp32Fused());
        }
        const double fp32SlotMs = median(tot);
        const double deqSlotMs = median(deq);
        const double gemmSlotMs = median(gem);
        std::vector<float> moeFp32Host(static_cast<size_t>(m) * H);
        down(moeFp32Host.data(), moeFp32, moeFp32Host.size() * 4, "read fp32 moe");
        double maxDiff = 0.0;
        double sumDiff2 = 0.0, sumRef2 = 0.0;
        double minCos = 1.0;
        for (uint32_t t = 0; t < m; ++t) {
            const float *a = moeExactHost.data() + static_cast<size_t>(t) * H;
            const float *b = moeFp32Host.data() + static_cast<size_t>(t) * H;
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (uint32_t i = 0; i < H; ++i) {
                const double d = static_cast<double>(b[i]) - a[i];
                maxDiff = std::max(maxDiff, std::fabs(d));
                sumDiff2 += d * d;
                sumRef2 += static_cast<double>(a[i]) * a[i];
                dot += static_cast<double>(a[i]) * b[i];
                na += static_cast<double>(a[i]) * a[i];
                nb += static_cast<double>(b[i]) * b[i];
            }
            minCos = std::min(minCos, dot / std::sqrt(na * nb));
        }
        const double relRms = sumRef2 > 0.0 ? std::sqrt(sumDiff2 / sumRef2) : 0.0;
        // Localise any divergence: gate/up output and down output, per stage.
        // The buffers are device memory, so copy them back first (one slot).
        double guRel = 0.0, guMax = 0.0, dnRel = 0.0, dnMax = 0.0;
        if (slot == opt.slots.front()) {
            const size_t guN = static_cast<size_t>(nExp) * cap * rowsGu;
            const size_t dnN = static_cast<size_t>(nExp) * cap * H;
            std::vector<float> hGuEx(guN), hGuFp(guN), hDnEx(dnN), hDnFp(dnN);
            down(hGuEx.data(), gout, guN * 4, "read exact gate/up");
            down(hGuFp.data(), gout2, guN * 4, "read fp32 gate/up");
            down(hDnEx.data(), gx, dnN * 4, "read exact down");
            down(hDnFp.data(), gx2, dnN * 4, "read fp32 down");
            const auto stageRel = [](const std::vector<float> &exact,
                                     const std::vector<float> &fp32,
                                     double *maxAbs) {
                double d2 = 0.0, r2 = 0.0;
                *maxAbs = 0.0;
                for (size_t i = 0; i < exact.size(); ++i) {
                    const double d = static_cast<double>(fp32[i]) - exact[i];
                    d2 += d * d;
                    r2 += static_cast<double>(exact[i]) * exact[i];
                    *maxAbs = std::max(*maxAbs, std::fabs(d));
                }
                return r2 > 0.0 ? std::sqrt(d2 / r2) : 0.0;
            };
            guRel = stageRel(hGuEx, hGuFp, &guMax);
            dnRel = stageRel(hDnEx, hDnFp, &dnMax);
        }
        std::printf("%4u %8u %6d %6d %9.2f %7.2f %8.2f %8.2f %10.2fx %9.2e "
                    "%9.7f\n",
                    slot, cs.distinct, cs.maxCount, cs.top8, exactSlotMs,
                    deqSlotMs, gemmSlotMs, fp32SlotMs, exactSlotMs / fp32SlotMs,
                    maxDiff, minCos);
        if (slot == 0u)
            std::printf("[fp32-probe]   slot %u: router ids %s (%zu/%zu "
                        "mismatched), replay vs engine moeOut max|d| %.3e\n",
                        slot, idxMismatch == 0 ? "MATCH" : "MISMATCH",
                        idxMismatch, idxGot.size(), replayMax);
        else
            std::printf("[fp32-probe]   slot %u: engine ids replayed for the "
                        "routing, blk.0 weights/bias used (timing + fp32-vs-exact "
                        "only)\n", slot);
        std::printf("[fp32-probe]   slot %u: fp32 vs exact  gate/up relRMS %.2e "
                    "(max %.2e)  down relRMS %.2e (max %.2e)  moeOut relRMS %.2e "
                    "(max %.2e, min cos %.8f)\n",
                    slot, guRel, guMax, dnRel, dnMax, relRms, maxDiff, minCos);
        if (slot == 0u && replayMax > 0.0) slotParityOk = false;
        if (slot == 0u && idxMismatch != 0) slotParityOk = false;
        if (relRms > 1e-4) slotParityOk = false;
        worstRelRms = std::max(worstRelRms, relRms);
        worstMaxDiff = std::max(worstMaxDiff, maxDiff);
        worstCos = std::min(worstCos, minCos);
        sumExact += exactSlotMs;
        sumDeq += deqSlotMs;
        sumGemm += gemmSlotMs;
        sumFp32 += fp32SlotMs;
        ++measuredSlots;
    }
    const double meanExact = measuredSlots > 0 ? sumExact / measuredSlots : 0.0;
    const double meanDeq = measuredSlots > 0 ? sumDeq / measuredSlots : 0.0;
    const double meanGemm = measuredSlots > 0 ? sumGemm / measuredSlots : 0.0;
    const double meanFp32 = measuredSlots > 0 ? sumFp32 / measuredSlots : 0.0;
    std::printf("[fp32-probe] mean over %u slots: exact %.2f ms (dequant %.2f + "
                "gemm %.2f = %.2f ms, overhead %.2f), fp32 %.2f ms, speed ratio "
                "%.2fx\n",
                measuredSlots, meanExact, meanDeq, meanGemm, meanDeq + meanGemm,
                meanDeq + meanGemm - meanFp32, meanFp32, meanExact / meanFp32);

    // ---------------------------------------------------------------------
    // 6. Routing reality: at m=128 the pairs are spread over many experts.
    // ---------------------------------------------------------------------
    std::printf("\n[fp32-probe] === routing reality at m=%d ===\n", opt.m);
    {
        double sumDistinct = 0.0, sumTop8Frac = 0.0;
        uint32_t nSlots = 0;
        for (uint32_t slot = 0; slot < cfg.numKvSlots; ++slot) {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/bt-idx-s%u.f32", opt.dump.c_str(),
                          slot);
            std::vector<int32_t> idx;
            if (!readIdxF32(path, idx, static_cast<size_t>(m) * nUsed)) continue;
            std::vector<int32_t> counts(nExp, 0);
            for (int32_t e: idx)
                if (e >= 0 && e < static_cast<int32_t>(nExp)) ++counts[e];
            const CountStats cs = countStats(counts);
            sumDistinct += cs.distinct;
            sumTop8Frac += static_cast<double>(cs.top8) / (m * nUsed);
            ++nSlots;
        }
        if (nSlots > 0) {
            std::printf("[fp32-probe] %u slots: mean distinct experts %.1f of %u, "
                        "mean top-8 share of the routed pairs %.1f%%\n",
                        nSlots, sumDistinct / nSlots, nExp,
                        100.0 * sumTop8Frac / nSlots);
            std::printf("[fp32-probe] one layer FP32: gate/up %.0f MB + down %.0f "
                        "MB = %.0f MB for %u experts (%.0f MB for 8); quantized "
                        "it is %.0f MB\n",
                        static_cast<double>(nExp) * rowsGu * H * 4.0 / 1048576.0,
                        static_cast<double>(nExp) * rowsDn * colsDn * 4.0 / 1048576.0,
                        static_cast<double>(nExp) * (rowsGu * H + rowsDn * colsDn) *
                                4.0 / 1048576.0,
                        nExp,
                        8.0 * (rowsGu * H + rowsDn * colsDn) * 4.0 / 1048576.0,
                        static_cast<double>(guBytes + dnBytes) / 1048576.0);
        }
    }

    // ---------------------------------------------------------------------
    // 7. m=512 in one pass: counts scaled 4x (the same router, 4x the tokens).
    //    Dequant cost per slot does not fall with m; the GEMM rows do.  The
    //    current grouped float path is measured at the same m and the same
    //    counts, so this is a like-for-like MoE-only comparison.  This probe
    //    keeps the padded [nExperts][512] layout; the engine now uses a compact
    //    prefix-offset layout at m=512 (see kInklingMoeGroupBuild).
    // ---------------------------------------------------------------------
    std::printf("\n[fp32-probe] === m=512, one pass (counts = 4x the measured "
                "m=128 counts) ===\n");
    double fp32Slot512Ms = 0.0, exactSlot512Ms = 0.0;
    {
        const uint32_t cap512 = 512u;
        std::vector<int32_t> idxRef(static_cast<size_t>(m) * nUsed);
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/bt-idx-s0.f32", opt.dump.c_str());
        if (readIdxF32(path, idxRef, idxRef.size())) {
            std::vector<int32_t> counts128(nExp, 0);
            for (int32_t e: idxRef) counts128[e]++;
            std::vector<int32_t> counts512(nExp, 0);
            uint32_t sum512 = 0;
            for (uint32_t e = 0; e < nExp; ++e) {
                counts512[e] = counts128[e] * 4;
                sum512 += counts512[e];
            }
            up(countsDev, counts512.data(), nExp * 4, "upload counts512");
            // SITU walks one block per padded slot and needs the slot's expert.
            int32_t *gexp512Dev = static_cast<int32_t *>(
                    devAlloc(static_cast<size_t>(nExp) * cap512 * 4, "gexp512"));
            std::vector<int32_t> gexp512(static_cast<size_t>(nExp) * cap512, 0);
            for (uint32_t e = 0; e < nExp; ++e)
                for (int32_t r = 0; r < counts512[e]; ++r)
                    gexp512[static_cast<size_t>(e) * cap512 + r] =
                            static_cast<int32_t>(e);
            up(gexp512Dev, gexp512.data(), gexp512.size() * 4, "upload gexp512");
            std::vector<float> xHost(static_cast<size_t>(m) * H);
            for (uint32_t t = 0; t < m; ++t) {
                std::snprintf(path, sizeof(path), "%s/nrm-t%u-s0.f32",
                              opt.dump.c_str(), t);
                if (!readF32(path, row, H)) die("missing nrm row (512)");
                std::memcpy(xHost.data() + static_cast<size_t>(t) * H, row.data(),
                            H * sizeof(float));
            }
            std::vector<std::vector<uint32_t>> toks(nExp);
            for (uint32_t t = 0; t < m; ++t)
                for (uint32_t u = 0; u < nUsed; ++u)
                    toks[static_cast<uint32_t>(
                                 idxRef[static_cast<size_t>(t) * nUsed + u])]
                            .push_back(t);
            float *gxIn512 = static_cast<float *>(devAlloc(
                    static_cast<size_t>(nExp) * cap512 * H * 4, "gxIn512"));
            std::vector<float> gxHost(static_cast<size_t>(nExp) * cap512 * H, 0.0f);
            for (uint32_t e = 0; e < nExp; ++e) {
                if (counts512[e] <= 0) continue;
                for (uint32_t s = 0; s < static_cast<uint32_t>(counts512[e]); ++s) {
                    const uint32_t src = toks[e][s % toks[e].size()];
                    std::memcpy(gxHost.data() +
                                        (static_cast<size_t>(e) * cap512 + s) * H,
                                xHost.data() + static_cast<size_t>(src) * H,
                                H * sizeof(float));
                }
            }
            up(gxIn512, gxHost.data(), gxHost.size() * 4, "upload gxIn512");

            // ---- current grouped float path at m=512 -----------------------
            {
                float *goutEx = static_cast<float *>(devAlloc(
                        static_cast<size_t>(nExp) * cap512 * rowsGu * 4, "goutEx512"));
                float *gactEx = static_cast<float *>(devAlloc(
                        static_cast<size_t>(nExp) * cap512 * ff * 4, "gactEx512"));
                float *gxDnEx = static_cast<float *>(devAlloc(
                        static_cast<size_t>(nExp) * cap512 * H * 4, "gxDnEx512"));
                std::vector<double> ms;
                for (int r = 0; r < opt.reps; ++r) {
                    Timer t;
                    t.start(nullptr);
                    if (!gpu::inklingGemvBatchedGroupedAsync(
                                19u, guDev, gxIn512, goutEx, rowsGu, H, cap512,
                                countsDev, nExp, nullptr))
                        die("exact gate/up grouped (512)");
                    gpu::kInklingSituGrouped(
                            goutEx, gactEx, gexp512Dev, gateUpScaleDev,
                            static_cast<uint32_t>(nExp) * cap512, ff,
                            cfg.situBeta, cfg.situLinearBeta, nullptr);
                    if (!gpu::inklingGemvBatchedGroupedAsync(
                                42u, dnDev, gactEx, gxDnEx, rowsDn, colsDn, cap512,
                                countsDev, nExp, nullptr))
                        die("exact down grouped (512)");
                    ms.push_back(t.stop(nullptr));
                }
                exactSlot512Ms = median(ms);
                cudaFree(goutEx);
                cudaFree(gactEx);
                cudaFree(gxDnEx);
            }

            // ---- FP32 dequant + SGEMM at m=512 -----------------------------
            {
                float *gout2 = static_cast<float *>(devAlloc(
                        static_cast<size_t>(nExp) * cap512 * rowsGu * 4, "gout2-512"));
                float *gact2 = static_cast<float *>(devAlloc(
                        static_cast<size_t>(nExp) * cap512 * ff * 4, "gact2-512"));
                float *gxDn2 = static_cast<float *>(devAlloc(
                        static_cast<size_t>(nExp) * cap512 * H * 4, "gxDn2-512"));
                const auto fp32Block512 = [&]() {
                    Timer t;
                    t.start(nullptr);
                    for (uint32_t e = 0; e < nExp; ++e) {
                        const int32_t cnt = counts512[e];
                        if (cnt <= 0) continue;
                        const float *xe =
                                gxIn512 + static_cast<size_t>(e) * cap512 * H;
                        float *oe =
                                gout2 + static_cast<size_t>(e) * cap512 * rowsGu;
                        if (!gpu::inklingFp32Dequant(
                                    19u,
                                    static_cast<const uint8_t *>(guDev) +
                                            static_cast<uint64_t>(e) * rowsGu *
                                                    rowBytesGu,
                                    dW, static_cast<uint64_t>(rowsGu) * H, nullptr))
                            die("fp32 dequant gate/up (512)");
                        if (!gpu::inklingFp32Gemm(dW, xe, oe, rowsGu, H,
                                                  static_cast<uint32_t>(cnt),
                                                  nullptr))
                            die("fp32 sgemm gate/up (512)");
                    }
                    gpu::kInklingSituGrouped(
                            gout2, gact2, gexp512Dev, gateUpScaleDev,
                            static_cast<uint32_t>(nExp) * cap512, ff,
                            cfg.situBeta, cfg.situLinearBeta, nullptr);
                    for (uint32_t e = 0; e < nExp; ++e) {
                        const int32_t cnt = counts512[e];
                        if (cnt <= 0) continue;
                        const float *ae =
                                gact2 + static_cast<size_t>(e) * cap512 * ff;
                        float *oe = gxDn2 + static_cast<size_t>(e) * cap512 * H;
                        if (!gpu::inklingFp32Dequant(
                                    42u,
                                    static_cast<const uint8_t *>(dnDev) +
                                            static_cast<uint64_t>(e) * rowsDn *
                                                    rowBytesDn,
                                    dWd, static_cast<uint64_t>(rowsDn) * colsDn,
                                    nullptr))
                            die("fp32 dequant down (512)");
                        if (!gpu::inklingFp32Gemm(dWd, ae, oe, rowsDn, colsDn,
                                                  static_cast<uint32_t>(cnt),
                                                  nullptr))
                            die("fp32 sgemm down (512)");
                    }
                    return t.stop(nullptr);
                };
                fp32Block512();
                std::vector<double> ms;
                for (int r = 0; r < opt.reps; ++r) ms.push_back(fp32Block512());
                fp32Slot512Ms = median(ms);
                cudaFree(gout2);
                cudaFree(gact2);
                cudaFree(gxDn2);
            }
            cudaFree(gxIn512);
            const double deqMb = static_cast<double>(nExp) *
                                 (rowsGu * H + rowsDn * colsDn) * 4.0 / 1048576.0;
            const double gflop =
                    2.0 * sum512 * (rowsGu * H + rowsDn * colsDn) / 1e9;
            std::printf("[fp32-probe] m=512 slot 0: %u routed pairs over %u "
                        "experts, %.2f GFLOP\n", sum512, nExp, gflop);
            std::printf("[fp32-probe] m=512 slot 0: current grouped %.2f ms; "
                        "fp32 dequant(%.0f MB)+sgemm %.2f ms  => %.2fx\n",
                        exactSlot512Ms, deqMb, fp32Slot512Ms,
                        exactSlot512Ms / fp32Slot512Ms);
        }
    }

    // 8. VRAM for a full-layer FP32 scratch (section 3 already allocated it).
    // ---------------------------------------------------------------------
    std::printf("\n[fp32-probe] === VRAM for a full-layer FP32 scratch ===\n");
    {
        size_t free0 = 0, total = 0;
        cudaMemGetInfo(&free0, &total);
        const double fullLayerMb =
                static_cast<double>(nExp) * (rowsGu * H + rowsDn * colsDn) * 4.0 /
                1048576.0;
        std::printf("[fp32-probe] all %u experts FP32 = %.0f MiB (gate/up %.0f + "
                    "down %.0f); 8 experts = %.0f MiB; live graph at m=%d leaves "
                    "%.0f free MiB of %.0f\n",
                    nExp, fullLayerMb,
                    static_cast<double>(nExp) * rowsGu * H * 4.0 / 1048576.0,
                    static_cast<double>(nExp) * rowsDn * colsDn * 4.0 / 1048576.0,
                    8.0 * (rowsGu * H + rowsDn * colsDn) * 4.0 / 1048576.0, opt.m,
                    static_cast<double>(graphFreeMiB),
                    static_cast<double>(gpuTotalMiB));
        float *guF32 = nullptr;
        float *dnF32 = nullptr;
        const cudaError_t e1 = cudaMalloc(reinterpret_cast<void **>(&guF32),
                                          static_cast<size_t>(nExp) * rowsGu * H * 4);
        const cudaError_t e2 =
                e1 == cudaSuccess
                        ? cudaMalloc(reinterpret_cast<void **>(&dnF32),
                                     static_cast<size_t>(nExp) * rowsDn * colsDn * 4)
                        : cudaErrorMemoryAllocation;
        size_t free1 = 0;
        cudaMemGetInfo(&free1, &total);
        std::printf("[fp32-probe] cudaMalloc of that scratch: gate/up %s, down "
                    "%s; free %.0f -> %.0f MiB of %.0f\n",
                    e1 == cudaSuccess ? "OK" : cudaGetErrorString(e1),
                    e1 == cudaSuccess && e2 == cudaSuccess
                            ? "OK"
                            : cudaGetErrorString(e1 == cudaSuccess ? e2 : e1),
                    free0 / 1048576.0, free1 / 1048576.0, total / 1048576.0);
        if (guF32 != nullptr) cudaFree(guF32);
        if (dnF32 != nullptr) cudaFree(dnF32);
    }

    // ---------------------------------------------------------------------
    // 9. Projection.
    // ---------------------------------------------------------------------
    {
        // The non-MoE part of a pass is what the measured pass leaves after
        // removing the 48 slot GEMV pairs (the profile's moe-gate + moe-down).
        const double restPerPassMs =
                currentPassMs > 0.0 ? currentPassMs - 48.0 * meanExact : 354.3;
        const double fp32PassMs = restPerPassMs + 48.0 * meanFp32;
        const double exactPassMs = restPerPassMs + 48.0 * meanExact;
        const double ppDropIn = 128.0 / (fp32PassMs / 1e3);
        std::printf("\n[fp32-probe] === projection ===\n");
        std::printf("[fp32-probe] current pass (m=%d) %.1f ms, non-MoE %.1f ms, "
                    "exact experts %.1f ms/pass (exact pass %.1f ms)\n",
                    opt.m, currentPassMs, restPerPassMs, 48.0 * meanExact,
                    exactPassMs);
        std::printf("[fp32-probe] drop-in fp32 at m=%d: %.1f ms/pass => %.1f "
                    "tok/s, i.e. %.2fx of today's %.1f tok/s\n",
                    opt.m, fp32PassMs, ppDropIn,
                    ppDropIn / (opt.m / (currentPassMs / 1e3)),
                    opt.m / (currentPassMs / 1e3));
        // MoE-only, per 512-token prompt: 4 m=128 passes x 48 slots, or 48
        // slots at m=512.  The m=512 row is hypothetical for the engine
        // (kBatchM=128) and the rest of the graph is not re-measured at m=512,
        // so no whole-model pp512 is claimed from it.
        const double moE512Now = 4.0 * 48.0 * meanExact;
        const double moE512Fp32 = 4.0 * 48.0 * meanFp32;
        const double moE512At512Current = 48.0 * exactSlot512Ms;
        const double moE512At512Fp32 = 48.0 * fp32Slot512Ms;
        std::printf("[fp32-probe] MoE per 512 tokens: engine (4x m=%d) current "
                    "%.0f ms, fp32 %.0f ms (%.2fx)\n",
                    opt.m, moE512Now, moE512Fp32, moE512Now / moE512Fp32);
        std::printf("[fp32-probe] MoE per 512 tokens at m=512 (hypothetical): "
                    "current %.0f ms, fp32 %.0f ms (%.2fx) -- at equal m the fp32 "
                    "copy loses; batching to 512 would help both paths\n",
                    moE512At512Current, moE512At512Fp32,
                    moE512At512Current / moE512At512Fp32);
    }

    // ---------------------------------------------------------------------
    // 10. Cleanup / verdict.
    // ---------------------------------------------------------------------
    cudaFree(guDev);
    cudaFree(dnDev);
    cudaFree(gateUpScaleDev);
    cudaFree(downScaleDev);
    cudaFree(probsBiasDev);
    cudaFree(xDev);
    cudaFree(rouDev);
    cudaFree(idxDev);
    cudaFree(wgtDev);
    cudaFree(gposDev);
    cudaFree(countsDev);
    cudaFree(gtokDev);
    cudaFree(gwgtDev);
    cudaFree(moeExact);
    cudaFree(moeFp32);
    cudaFree(dW);
    cudaFree(dWd);
    if (gx != nullptr) cudaFree(gx);
    if (gout != nullptr) cudaFree(gout);
    if (gact != nullptr) cudaFree(gact);
    if (gx2 != nullptr) cudaFree(gx2);
    if (gout2 != nullptr) cudaFree(gout2);
    if (gact2 != nullptr) cudaFree(gact2);

    std::printf("\n[fp32-probe] dequant gate: %s\n",
                dequantExact ? "BIT-EXACT vs GGMLDequantize" : "MISMATCH");
    std::printf("[fp32-probe] block parity: %s (worst relRMS %.2e, worst "
                "max|d| %.2e, worst cos %.8f)\n",
                slotParityOk ? "PASS (fp32 summation-order only)"
                             : "FAIL (see rows above)",
                worstRelRms, worstMaxDiff, worstCos);
    if (!opt.keepDump)
        std::printf("[fp32-probe] dump kept at %s (remove manually)\n",
                    opt.dump.c_str());
    return dequantExact ? 0 : 1;
}
