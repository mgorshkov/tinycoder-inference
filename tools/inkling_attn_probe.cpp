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

// Isolation probe for the Inkling-2-Mini batched attention kernel.
//
// The batched-prefill section timer puts "attention" (clip + relbias + store +
// the kernel) at ~135 ms per 128-token pass.  A section number that large
// needs to be attributed before anything is rewritten, so this tool times the
// kernel alone on the real geometry (24 q heads / 8 kv heads / headDim 128 /
// extent 1024) and prints cycles per (token, key) pair.
//
// Usage:
//   ./build-inkling/unit_tests/inkling_attn_probe [--heads 24] [--kv 8]
//       [--tokens 128] [--keys 512] [--extent 1024] [--window 1024]
//       [--swa 1] [--reps 20]
//
// --keys is the position of the last query (nKeys = keys for a full-attention
// block).  The probe always times BOTH the 8-query-token tiled kernel and the
// per-token kernel (TINYCODER_INKLING_ATTN_TILE) and prints their speedup, so
// there is no flag to select one.  --parity compares the two outputs.
#include "gpu/InklingKernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    uint32_t nHeads = 24, nKv = 8, tokens = 128, keys = 512, extent = 1024,
             window = 1024, swa = 1;
    int reps = 20;
    bool parity = false, flash = false;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        auto next = [&]() -> uint32_t {
            return i + 1 < argc ? static_cast<uint32_t>(std::atoi(argv[++i])) : 0u;
        };
        if (!std::strcmp(a, "--heads")) nHeads = next();
        else if (!std::strcmp(a, "--kv"))
            nKv = next();
        else if (!std::strcmp(a, "--tokens"))
            tokens = next();
        else if (!std::strcmp(a, "--keys"))
            keys = next();
        else if (!std::strcmp(a, "--extent"))
            extent = next();
        else if (!std::strcmp(a, "--window"))
            window = next();
        else if (!std::strcmp(a, "--swa"))
            swa = next();
        else if (!std::strcmp(a, "--reps"))
            reps = static_cast<int>(next());
        else if (!std::strcmp(a, "--parity"))
            parity = true;
        else if (!std::strcmp(a, "--flash"))
            flash = true;
        else {
            std::fprintf(stderr, "[attn] unknown argument %s\n", a);
            return 2;
        }
    }
    if (tokens == 0 || keys == 0 || nHeads % nKv != 0) {
        std::fprintf(stderr, "[attn] bad geometry\n");
        return 2;
    }

    const uint32_t qDim = nHeads * 128u;
    const uint32_t kvDim = nKv * 128u;
    const uint32_t maxSeq = keys + tokens + 8u;
    const uint32_t pos0 = keys >= tokens ? keys - tokens : 0u;

    std::vector<float> qh(static_cast<size_t>(tokens) * qDim);
    std::vector<float> kh(static_cast<size_t>(maxSeq) * kvDim);
    std::vector<float> vh(static_cast<size_t>(maxSeq) * kvDim);
    std::vector<float> th(static_cast<size_t>(tokens) * nHeads * (extent + 1u));
    uint32_t s = 0x12345678u;
    auto rnd = [&]() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    };
    for (float &v: qh) v = rnd();
    for (float &v: kh) v = rnd();
    for (float &v: vh) v = rnd();
    for (float &v: th) v = rnd() * 0.25f;

    float *q = nullptr, *k = nullptr, *v = nullptr, *t = nullptr;
    float *o0 = nullptr, *o1 = nullptr;
    cudaMalloc(&q, qh.size() * 4);
    cudaMalloc(&k, kh.size() * 4);
    cudaMalloc(&v, vh.size() * 4);
    cudaMalloc(&t, th.size() * 4);
    cudaMalloc(&o0, static_cast<size_t>(tokens) * qDim * 4);
    cudaMalloc(&o1, static_cast<size_t>(tokens) * qDim * 4);
    cudaMemcpy(q, qh.data(), qh.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(k, kh.data(), kh.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(v, vh.data(), vh.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(t, th.data(), th.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(o0, 0, static_cast<size_t>(tokens) * qDim * 4);
    cudaMemset(o1, 0, static_cast<size_t>(tokens) * qDim * 4);

    const float qkScale = 1.0f / std::sqrt(128.0f);
    auto launch = [&](float *out, bool tiled, bool useFlash = false) {
        setenv("TINYCODER_INKLING_ATTN_TILE", tiled ? "1" : "0", 1);
        setenv("TINYCODER_INKLING_ATTN_FLASH", useFlash ? "1" : "0", 1);
        tinycoder::gpu::kInklingAttentionBatch(
                q, k, v, t, pos0, nHeads, nKv, extent, qkScale, out, tokens,
                window, swa != 0u, nullptr);
    };

    // warm up + time
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto time = [&](bool tiled, bool useFlash = false) {
        launch(o0, tiled, useFlash);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) {
            std::printf("[attn] launch failed: %s\n",
                        cudaGetErrorString(cudaGetLastError()));
            return -1.0;
        }
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) launch(o0, tiled, useFlash);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        return static_cast<double>(ms) / reps;
    };

    const double msTiled = time(true);
    const double msToken = time(false);
    const double msFlash = flash ? time(true, true) : -1.0;
    // (token, key) pairs actually evaluated: nHeads * sum over tokens of nKeys.
    double pairs = 0.0;
    for (uint32_t tt = 0; tt < tokens; ++tt) {
        const uint32_t pos = pos0 + tt;
        const uint32_t firstKey = (swa != 0u && window > 0 && pos + 1 > window)
                                          ? pos + 1 - window
                                          : 0u;
        pairs += static_cast<double>(nHeads) * (pos + 1 - firstKey);
    }
    std::printf("[attn] geometry heads=%u kv=%u tokens=%u keys=%u extent=%u "
                "window=%u swa=%u pos0=%u\n",
                nHeads, nKv, tokens, keys, extent, window, swa, pos0);
    std::printf("[attn] pairs=%.0f  (token,key)\n", pairs);
    for (auto pr: {std::pair<const char *, double>("tile", msTiled),
                   std::pair<const char *, double>("token", msToken),
                   std::pair<const char *, double>("flash", msFlash)}) {
        if (pr.second < 0.0) continue;
        const double nsPerPair = pr.second * 1e6 / pairs;
        const double cycles = nsPerPair * 1.5;// ~1.5 GHz
        std::printf("[attn] %-5s %8.3f ms  %.2f ns/pair  %.1f cycles/pair  "
                    "%.1f Gpair/s\n",
                    pr.first, pr.second, nsPerPair, cycles, pairs / pr.second / 1e6);
    }
    if (msTiled > 0.0 && msToken > 0.0)
        std::printf("[attn] tile/token speedup %.2fx\n", msToken / msTiled);

    if (parity) {
        std::vector<float> a(static_cast<size_t>(tokens) * qDim),
                b(static_cast<size_t>(tokens) * qDim);
        launch(o0, false);
        cudaDeviceSynchronize();
        cudaMemcpy(a.data(), o0, a.size() * 4, cudaMemcpyDeviceToHost);
        launch(o1, true);
        cudaDeviceSynchronize();
        cudaMemcpy(b.data(), o1, b.size() * 4, cudaMemcpyDeviceToHost);
        auto compare = [&](const char *what) {
            double mx = 0.0, ref = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
            size_t diff = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                mx = std::fmax(mx, std::fabs(double(a[i]) - b[i]));
                ref = std::fmax(ref, std::fabs(double(a[i])));
                dot += double(a[i]) * b[i];
                na += double(a[i]) * a[i];
                nb += double(b[i]) * b[i];
                if (a[i] != b[i]) ++diff;
            }
            const double cos = dot / std::sqrt(na * nb);
            std::printf("[attn] %s: max|d|=%.3e (ref max %.3e) cos %.10f "
                        "%zu/%zu elements differ -> %s\n",
                        what, mx, ref, cos, diff, a.size(),
                        mx == 0.0 ? "BIT-EXACT" : "differs");
            return cos;
        };
        compare("parity tile vs token");
        if (flash) {
            // b currently holds the tiled kernel; reload the flash output.
            launch(o1, true, true);
            cudaDeviceSynchronize();
            cudaMemcpy(b.data(), o1, b.size() * 4, cudaMemcpyDeviceToHost);
            compare("parity flash vs token");
        }
    }
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(t);
    cudaFree(o0);
    cudaFree(o1);
    return 0;
}
