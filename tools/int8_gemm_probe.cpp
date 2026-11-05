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

// int8_gemm_probe.cpp
//
// Per-shape timing probe for the GGUF prefill int8 tensor-core GEMM
// (int8Gemm -> int8GemmStreamK<q>Tile<tileX>).  Answers one question: for the
// exact (type, rows, cols) of a real checkpoint's projections, what TFLOPS
// does the shipped kernel reach at seqLen=512?
//
// Why: a whole-model pp512 number is a weighted average over ~7 projections x
// N layers plus attention/GDN/LM-head, so it cannot say WHICH shape is short.
// This probe times each shape in isolation, which localises the deficit.
//
// Synthetic weights (no GGUF payload read): MMA throughput is data
// independent, and this probe is for TIMING only.  Numeric correctness of this
// path is covered by the model-level logits gates and the question tests.
//
// Build/run:
//   cmake --build build-1.0 --target int8_gemm_probe -j 20
//   ./build-1.0/unit_tests/int8_gemm_probe [--models 7b|9b|all] [--reps N]
//                                          [--seqlen N] [--csv]
// Env knobs are the shipped ones (TINYCODER_INT8_GEMM_X / _BLOCKS) so a sweep
// needs no rebuild.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "Int8Gemm.hpp"

namespace {

    constexpr int GGML_TYPE_Q8_0 = 8;
    constexpr int GGML_TYPE_Q4_K = 12;
    constexpr int GGML_TYPE_Q5_K = 13;
    constexpr int GGML_TYPE_Q6_K = 14;
    constexpr int GGML_TYPE_IQ2_S = 22;
    constexpr int GGML_TYPE_IQ3_XXS = 18;
    constexpr int GGML_TYPE_IQ3_S = 21;

    const char *typeName(int t) {
        switch (t) {
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_IQ2_S: return "IQ2_S";
        case GGML_TYPE_IQ3_XXS: return "IQ3XXS";
        case GGML_TYPE_IQ3_S: return "IQ3S";
        default: return "?";
        }
    }

    // bytes per quant block and values per block
    size_t typeSize(int t) {
        switch (t) {
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_IQ2_S: return 82;
        case GGML_TYPE_IQ3_XXS: return 98;
        case GGML_TYPE_IQ3_S: return 110;
        default: return 0;
        }
    }
    size_t typeBlock(int t) {
        return t == GGML_TYPE_Q8_0 ? 32u : 256u;
    }

    struct Shape {
        const char *name;
        int type;
        uint32_t rows;
        uint32_t cols;
    };

    // Cross product rows x cols at one type, to separate a K (cols) effect from
    // an n (rows) effect.
    const Shape kSweepQ4K[] = {
            {"Q4K 3584x3584", GGML_TYPE_Q4_K, 3584, 3584},
            {"Q4K 3584x4096", GGML_TYPE_Q4_K, 3584, 4096},
            {"Q4K 4096x3584", GGML_TYPE_Q4_K, 4096, 3584},
            {"Q4K 4096x4096", GGML_TYPE_Q4_K, 4096, 4096},
            {"Q4K 12288x3584", GGML_TYPE_Q4_K, 12288, 3584},
            {"Q4K 12288x4096", GGML_TYPE_Q4_K, 12288, 4096},
            {"Q4K 18944x3584", GGML_TYPE_Q4_K, 18944, 3584},
            {"Q4K 18944x4096", GGML_TYPE_Q4_K, 18944, 4096},
            {"Q4K 8192x3584", GGML_TYPE_Q4_K, 8192, 3584},
            {"Q4K 8192x4096", GGML_TYPE_Q4_K, 8192, 4096},
            {"Q4K 3584x7168", GGML_TYPE_Q4_K, 3584, 7168},
            {"Q4K 3584x18944", GGML_TYPE_Q4_K, 3584, 18944},
            {"Q4K 3584x12288", GGML_TYPE_Q4_K, 3584, 12288},
    };

    // The projection inventory of the two checkpoint families, copied from the
    // GGUF tensor table.  rows = output features, cols = input features.
    const Shape kSevenB[] = {
            {"attn_q   (IQ2_S)", GGML_TYPE_IQ2_S, 3584, 3584},
            {"attn_k   (IQ2_S)", GGML_TYPE_IQ2_S, 512, 3584},
            {"attn_v   (Q4_K) ", GGML_TYPE_Q4_K, 512, 3584},
            {"attn_o   (Q4_K) ", GGML_TYPE_Q4_K, 3584, 3584},
            {"ffn_gate (Q4_K) ", GGML_TYPE_Q4_K, 18944, 3584},
            {"ffn_up   (Q4_K) ", GGML_TYPE_Q4_K, 18944, 3584},
            {"ffn_down (Q4_K) ", GGML_TYPE_Q4_K, 3584, 18944},
            {"lm_head  (Q5_K) ", GGML_TYPE_Q5_K, 152064, 3584},
    };

    const Shape kNineB[] = {
            {"attn_qkv (Q6_K) ", GGML_TYPE_Q6_K, 8192, 4096},
            {"attn_gat (Q4_K) ", GGML_TYPE_Q4_K, 4096, 4096},
            {"attn_o   (Q5_K) ", GGML_TYPE_Q5_K, 4096, 4096},
            {"ssm_out  (Q8_0) ", GGML_TYPE_Q8_0, 4096, 4096},
            {"ffn_gate (Q4_K) ", GGML_TYPE_Q4_K, 12288, 4096},
            {"ffn_up   (Q4_K) ", GGML_TYPE_Q4_K, 12288, 4096},
            {"ffn_down (Q6_K) ", GGML_TYPE_Q6_K, 4096, 12288},
            {"lm_head  (Q6_K) ", GGML_TYPE_Q6_K, 248320, 4096},
    };

    // Qwen2.5-Coder-7B `iq3_xxs`: every projection is IQ3_XXS.  Compared
    // against kSevenB this isolates the quant type from the shape, because the
    // rows/cols are the same inventory (see BENCHMARKS.md).
    const Shape kSevenBIq3[] = {
            {"q   IQ3XXS 3584", GGML_TYPE_IQ3_XXS, 3584, 3584},
            {"k   IQ3XXS  512", GGML_TYPE_IQ3_XXS, 512, 3584},
            {"v   IQ3XXS  512", GGML_TYPE_IQ3_XXS, 512, 3584},
            {"o   IQ3XXS 3584", GGML_TYPE_IQ3_XXS, 3584, 3584},
            {"gate IQ3XXS 18944", GGML_TYPE_IQ3_XXS, 18944, 3584},
            {"up  IQ3XXS 18944", GGML_TYPE_IQ3_XXS, 18944, 3584},
            {"down IQ3XXS 3584x18944", GGML_TYPE_IQ3_XXS, 3584, 18944},
            {"o   IQ3S  3584", GGML_TYPE_IQ3_S, 3584, 3584},
            {"q   IQ2S  3584", GGML_TYPE_IQ2_S, 3584, 3584},
            {"k   IQ2S   512", GGML_TYPE_IQ2_S, 512, 3584},
            {"q4k ref 3584", GGML_TYPE_Q4_K, 3584, 3584},
            {"q4k ref 18944", GGML_TYPE_Q4_K, 18944, 3584},
            {"q4k ref 512", GGML_TYPE_Q4_K, 512, 3584},
            {"q4k ref 3584x18944", GGML_TYPE_Q4_K, 3584, 18944},
    };

    // Qwen2.5-Coder-1.5B `iq3_xxs`: hidden 1536, intermediate 8960, 28 layers,
    // 12 query / 2 KV heads.  This is the prefill row whose shapes are SMALL in
    // n -- q/k/v are 256-1536 rows and attnO/down are 12-16 tiles of 128 -- so
    // it probes the regime where the stream-k grid cannot fill 68 SMs from
    // whole tiles.
    const Shape kOneFiveBIq3[] = {
            {"q   IQ2S  1536", GGML_TYPE_IQ2_S, 1536, 1536},
            {"k   IQ2S   256", GGML_TYPE_IQ2_S, 256, 1536},
            {"v   Q4K    256", GGML_TYPE_Q4_K, 256, 1536},
            {"qkv merged 2048", GGML_TYPE_IQ2_S, 2048, 1536},
            {"o   IQ3S  1536", GGML_TYPE_IQ3_S, 1536, 1536},
            {"o   IQ3XXS 1536", GGML_TYPE_IQ3_XXS, 1536, 1536},
            {"gate IQ3XXS 8960", GGML_TYPE_IQ3_XXS, 8960, 1536},
            {"up  IQ3XXS 8960", GGML_TYPE_IQ3_XXS, 8960, 1536},
            {"down IQ3XXS 1536x8960", GGML_TYPE_IQ3_XXS, 1536, 8960},
    };

    // The card's SM clock moves with load: a 1350 MHz sample reads ~30% low
    // against a 1965 MHz one, which is larger than any effect this probe is
    // meant to resolve.  Every line is therefore tagged with the clock that
    // was in force, and the probe refuses to print a comparison it knows to be
    // clock-inconsistent.
    unsigned int smClock() {
        static bool inited = false;
        static nvmlDevice_t dev = nullptr;
        if (!inited) {
            inited = true;
            if (nvmlInit_v2() == NVML_SUCCESS)
                nvmlDeviceGetHandleByIndex_v2(0, &dev);
        }
        if (dev == nullptr) return 0;
        unsigned int mhz = 0;
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz) != NVML_SUCCESS) return 0;
        return mhz;
    }

    bool g_scan = false;

    struct Result {
        float ms;
        double tflops;
    };


    // Non-finite scan + first-magnitude report for one output buffer.  A NaN/inf
    // here with benign synthetic weights and sin() activations means the kernel
    // itself produced it, not the data.
    struct Scan {
        unsigned long nanCount = 0, infCount = 0;
        float maxAbs = 0.0f, minAbs = 1e30f;
        double sumAbs = 0.0;
        unsigned long finiteCount = 0;
    };
    Scan scanOut(const float *d, size_t n) {
        std::vector<float> h(n);
        cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
        Scan r;
        for (size_t i = 0; i < n; ++i) {
            const float v = h[i];
            if (std::isnan(v)) { ++r.nanCount; continue; }
            if (std::isinf(v)) { ++r.infCount; continue; }
            const float a = std::fabs(v);
            if (a > r.maxAbs) r.maxAbs = a;
            if (a < r.minAbs) r.minAbs = a;
            r.sumAbs += a;
            ++r.finiteCount;
        }
        if (r.finiteCount == 0) r.minAbs = 0.0f;
        return r;
    }
    void printScan(const char *tag, const Shape &s, const Scan &sc) {
        const double rel = sc.finiteCount ? sc.sumAbs / static_cast<double>(sc.finiteCount) : 0.0;
        std::printf("  %-5s %-16s %6u %6u nan=%lu inf=%lu finite=%lu maxAbs=%.6g meanAbs=%.6g\n",
                    tag, s.name, s.rows, s.cols, sc.nanCount, sc.infCount, sc.finiteCount,
                    sc.maxAbs, rel);
        std::fflush(stdout);
    }
    Result timeShape(const Shape &s, uint32_t seqLen, int reps) {
        const size_t bp = s.cols / typeBlock(s.type);
        const size_t wBytes = static_cast<size_t>(s.rows) * bp * typeSize(s.type);
        void *w = nullptr;
        void *x = nullptr;
        void *out = nullptr;
        cudaMalloc(&w, wBytes);
        cudaMemset(w, 0x11, wBytes);
        cudaMalloc(&x, static_cast<size_t>(seqLen) * s.cols * sizeof(float));
        cudaMemset(x, 0x11, static_cast<size_t>(seqLen) * s.cols * sizeof(float));
        cudaMalloc(&out, static_cast<size_t>(seqLen) * s.rows * sizeof(float));

        // Fill the activations with a deterministic smooth pattern.
        {
            std::vector<float> h(static_cast<size_t>(seqLen) * s.cols);
            for (size_t i = 0; i < h.size(); ++i)
                h[i] = std::sin(0.001f * static_cast<float>(i));
            cudaMemcpy(x, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
        }

        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        std::vector<float> samples;
        for (int r = 0; r < reps + 3; ++r) {
            cudaEventRecord(a);
            const bool ok = tinycoder::gpu::int8Gemm(
                    s.type, w, static_cast<const float *>(x), static_cast<float *>(out),
                    seqLen, s.rows, s.cols, static_cast<uint32_t>(bp), nullptr);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            if (!ok) {
                std::fprintf(stderr, "[probe] int8Gemm returned false for %s\n", s.name);
                return {0.0f, 0.0};
            }
            if (r >= 3) samples.push_back(ms);
        }
        std::sort(samples.begin(), samples.end());
        const float med = samples[samples.size() / 2];
        const double flops = 2.0 * s.rows * s.cols * seqLen;
        if (g_scan) {
            printScan("int8", s, scanOut(static_cast<const float *>(out),
                                        static_cast<size_t>(seqLen) * s.rows));
        }
        cudaFree(w);
        cudaFree(x);
        cudaFree(out);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return {med, flops / (static_cast<double>(med) * 1e-3) / 1e12};
    }

    // What the SAME shape costs on the path the engine falls back to when
    // int8Gemm() refuses it: cuBLAS fp16 tensor-op GEMM over an fp16 twin of
    // the weights (the engine dequantizes each matrix once at upload and keeps
    // the twin resident), plus the kF32ToF16 activation convert that the fp16
    // path issues before every call.  The GEMM call mirrors
    // GPUComputeForward's exactly: CUBLAS_OP_T / CUBLAS_OP_N, A = the twin
    // (ld = cols), B = the activations (ld = cols), C fp32 (ld = rows).
    //
    // The convert is timed as a D2D copy of the fp32 activation volume scaled
    // by 0.75 (it reads 4 B and writes 2 B per element, a copy reads and
    // writes 4+4), so it is an estimate on a bandwidth-bound launch, not a
    // measurement of the kernel itself.
    Result timeCublasF16(const Shape &s, uint32_t seqLen, int reps, cublasHandle_t h,
                         float *convertMs) {
        void *w = nullptr;
        void *x = nullptr;
        void *out = nullptr;
        void *cvtBuf = nullptr;
        const size_t wBytes = static_cast<size_t>(s.rows) * s.cols * sizeof(__half);
        const size_t xBytes = static_cast<size_t>(seqLen) * s.cols * sizeof(__half);
        const size_t cvtBytes = static_cast<size_t>(seqLen) * s.cols * sizeof(float);
        cudaMalloc(&w, wBytes);
        cudaMemset(w, 0x11, wBytes);
        cudaMalloc(&x, xBytes);
        cudaMemset(x, 0x11, xBytes);
        cudaMalloc(&out, static_cast<size_t>(seqLen) * s.rows * sizeof(float));
        cudaMalloc(&cvtBuf, cvtBytes);
        float alpha = 1.0f, beta = 0.0f;
        const int m = static_cast<int>(s.rows);
        const int n = static_cast<int>(seqLen);
        const int k = static_cast<int>(s.cols);
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        std::vector<float> gemm, cvt;
        for (int r = 0; r < reps + 3; ++r) {
            cudaEventRecord(a);
            const cublasStatus_t st = cublasGemmEx(
                    h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, w, CUDA_R_16F, k,
                    x, CUDA_R_16F, k, &beta, out, CUDA_R_32F, m, CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            if (st != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "[probe] cublasGemmEx failed (%d) for %s\n", (int) st, s.name);
                return {0.0f, 0.0};
            }
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            if (r >= 3) gemm.push_back(ms);
        }
        // Convert-volume proxy (see above).
        for (int r = 0; r < reps + 3; ++r) {
            cudaEventRecord(a);
            cudaMemcpyAsync(cvtBuf, x, cvtBytes, cudaMemcpyDeviceToDevice, nullptr);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            if (r >= 3) cvt.push_back(ms);
        }
        std::sort(gemm.begin(), gemm.end());
        std::sort(cvt.begin(), cvt.end());
        const float med = gemm[gemm.size() / 2];
        *convertMs = cvt[cvt.size() / 2] * 0.75f;
        const double flops = 2.0 * s.rows * s.cols * seqLen;
        if (g_scan) {
            printScan("fp16", s, scanOut(static_cast<const float *>(out),
                                        static_cast<size_t>(seqLen) * s.rows));
        }
        cudaFree(w);
        cudaFree(x);
        cudaFree(out);
        cudaFree(cvtBuf);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return {med, flops / (static_cast<double>(med) * 1e-3) / 1e12};
    }

}// namespace

int main(int argc, char **argv) {
    std::string models = "all";
    int reps = 10;
    uint32_t seqLen = 512;
    bool csv = false;
    bool compare = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--models") == 0 && i + 1 < argc) models = argv[++i];
        else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seqlen") == 0 && i + 1 < argc) seqLen = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--csv") == 0) csv = true;
        else if (std::strcmp(argv[i], "--compare") == 0) compare = true;
        else if (std::strcmp(argv[i], "--nan") == 0) g_scan = true;
        else if (std::strcmp(argv[i], "--sweep") == 0) models = "sweep";
        else {
            std::fprintf(stderr, "usage: %s [--models 7b|9b|7biq3|all] [--reps N] [--seqlen N] [--csv]\n", argv[0]);
            return 2;
        }
    }
    if (!tinycoder::gpu::int8GemmAvailable()) {
        std::fprintf(stderr, "[probe] int8Gemm not available\n");
        return 1;
    }
    const char *ex = std::getenv("TINYCODER_INT8_GEMM_X");
    const char *bl = std::getenv("TINYCODER_INT8_GEMM_BLOCKS");
    if (!csv) {
        std::fprintf(stderr,
                     "int8 prefill GEMM probe: seqLen=%u reps=%d  X=%s blocks=%s\n",
                     seqLen, reps, ex ? ex : "(default 128)", bl ? bl : "(default)");
        std::fprintf(stderr, "%-18s %-6s %7s %7s %9s %8s\n", "shape", "type", "rows", "cols", "ms", "TFLOPS");
    }
    // Boost the clock before any measurement and hold it: a burst of ~50 ms of
    // GEMM work is not enough on its own to leave the idle DVFS state.
    {
        const unsigned int c0 = smClock();
        const Shape warm = {"warm", GGML_TYPE_Q4_K, 8192, 4096};
        // Boost is slow to engage (idle 300 MHz -> 1350 -> 1965 MHz takes ~10 s
        // of sustained load), so wait for it instead of guessing a duration.
        for (int i = 0; i < 20000 && smClock() < 1900; ++i) timeShape(warm, seqLen, 2);
        if (!csv)
            std::fprintf(stderr, "clock after warmup: %u MHz (was %u MHz)\n", smClock(), c0);
    }
    // --compare: int8 MMA vs the cuBLAS fp16 fallback for every shape, which is
    // the table the per-shape dispatch decision is made from.
    cublasHandle_t cb = nullptr;
    if (compare) {
        cublasCreate(&cb);
        std::fprintf(stderr, "%-18s %-6s %7s %7s %9s %9s %8s %9s  %s\n", "shape", "type", "rows",
                     "cols", "int8_ms", "f16_ms", "cvt_ms", "f16_tot", "faster");
    }
    auto runCompare = [&](const Shape *v, size_t n) {
        double lostBar = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const unsigned int clk = smClock();
            // A pending error from the previous shape's cuBLAS/copy would make
            // int8Gemm()'s cudaGetLastError() gate refuse a shape that is in
            // fact launchable (it did: identical shapes alternating refused).
            cudaGetLastError();
            const Result r8 = timeShape(v[i], seqLen, reps);
            cudaGetLastError();
            float cvt = 0.0f;
            const Result rf = timeCublasF16(v[i], seqLen, reps, cb, &cvt);
            cudaGetLastError();
            const double tot = rf.ms + cvt;
            if (r8.ms <= 0.0f) {
                std::printf("%-18s %-6s %7u %7u  %7s %9.3f %8.3f %9.3f  fp16 (int8 refused)  %4u MHz\n",
                            v[i].name, typeName(v[i].type), v[i].rows, v[i].cols, "-", rf.ms, cvt,
                            tot, clk);
                continue;
            }
            const bool int8Wins = r8.ms <= tot;
            if (!int8Wins) lostBar += (r8.ms - tot);
            std::printf("%-18s %-6s %7u %7u %9.3f %9.3f %8.3f %9.3f  %s  %4u MHz\n", v[i].name,
                        typeName(v[i].type), v[i].rows, v[i].cols, r8.ms, rf.ms, cvt, tot,
                        int8Wins ? "int8" : "fp16", clk);
        }
        if (n > 0)
            std::fprintf(stderr, "sum(int8 - fp16) over rows where fp16 wins: %.3f ms/layer-call\n",
                         lostBar);
    };
    auto run = [&](const Shape *v, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            const unsigned int clk = smClock();
            const Result r = timeShape(v[i], seqLen, reps);
            if (csv) std::printf("%s,%s,%u,%u,%.3f,%.2f,%u\n", v[i].name, typeName(v[i].type),
                                 v[i].rows, v[i].cols, r.ms, r.tflops, clk);
            else std::printf("%-18s %-6s %7u %7u %9.3f %8.2f  %4u MHz\n", v[i].name, typeName(v[i].type),
                             v[i].rows, v[i].cols, r.ms, r.tflops, clk);
        }
    };
    const auto runSel = [&](const Shape *v, size_t n) { compare ? runCompare(v, n) : run(v, n); };
    if (models == "all" || models == "7b")
        runSel(kSevenB, sizeof(kSevenB) / sizeof(kSevenB[0]));
    if (models == "all" || models == "9b")
        runSel(kNineB, sizeof(kNineB) / sizeof(kNineB[0]));
    if (models == "7biq3")
        runSel(kSevenBIq3, sizeof(kSevenBIq3) / sizeof(kSevenBIq3[0]));
    if (models == "1p5b" || models == "all")
        runSel(kOneFiveBIq3, sizeof(kOneFiveBIq3) / sizeof(kOneFiveBIq3[0]));
    if (models == "sweep")
        runSel(kSweepQ4K, sizeof(kSweepQ4K) / sizeof(kSweepQ4K[0]));
    if (compare) cublasDestroy(cb);
    return 0;
}
