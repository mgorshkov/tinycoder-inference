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

// exl3_gemv_probe -- time ONE production decode projection through the shipped
// fused GEMV entry point (gemvFused), at a chosen (k, n, K).
//
// Why this exists: the engine-level benchmark can only report a whole decode
// step, and the model takes ~40 s to load, so a config sweep inside the engine
// costs minutes per point.  This probe registers a synthetic linear of the same
// geometry and bitrate with uploadFusedLinearDevice() -- the same registry and
// the same launch code the model uses -- and times gemvFused in a loop.  The
// trellis contents do not matter for a bandwidth measurement (they are zeros),
// and every kernel-selection env knob (TINYCODER_EXL3_GEMV_K1,
// TINYCODER_EXL3_GEMV_K1TT, TINYCODER_EXL3_GEMV_SPLIT, TINYCODER_EXL3_K1DIAG)
// is honoured because the probe enters through the production dispatch.
//
// GB/s here is (k*n*K/8) / (ms per call), i.e. the trellis bytes actually
// streamed per projection, the same definition as the in-engine table
// (TINYCODER_EXL3_GEMV_TIME=1 TINYCODER_EXL3_GEMV_TABLE=1).
//
// Build (explicit target; see unit_tests/CMakeLists.txt):
//   cmake --build build-1.0 --target exl3_gemv_probe -j 20
// Run:
//   ./build-1.0/unit_tests/exl3_gemv_probe                 # all 27B shapes
//   ./build-1.0/unit_tests/exl3_gemv_probe 17408 5120 1    # one shape
// Optional 4th arg = iterations (default 40).

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "EXL3Kernels.hpp"

#define CKP(expr)                                                                     \
    do {                                                                              \
        cudaError_t e_ = (expr);                                                      \
        if (e_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s:%d cuda error %s\n", __FILE__, __LINE__,         \
                         cudaGetErrorString(e_));                                     \
            std::exit(2);                                                             \
        }                                                                             \
    } while (0)

namespace {

    struct Shape {
        uint32_t k, n, K;
        const char *what;
    };

    // The 27B 1.60bpw decode projections (Qwen3.5-style: hidden 5120, MLP
    // 17408, GDN value 6144, 4 attention heads x 256).  K = 1 for the 1.60 bpw
    // trellis, the lm_head is 2.0 bpw in this checkpoint.
    const Shape kShapes[] = {
            {5120, 17408, 1, "ffn_gate/up"},
            {17408, 5120, 1, "ffn_down"},
            {5120, 12288, 1, "attn_qkv"},
            {6144, 5120, 3, "attn_out"},
            {5120, 10240, 3, "gdn_in"},
            {5120, 6144, 3, "gdn_out"},
            {5120, 248320, 3, "lm_head"},
    };

    // One projection read repeatedly would be half resident in the 5.5 MB L2 of
    // this card (an 11 MB trellis!), which flatters every number by 2-3x.  The
    // engine streams 60 DISTINCT projections per decode step, so the probe
    // registers `copies` distinct trellises of the same geometry and cycles
    // through them: the working set is then far larger than L2 and the rate is
    // a DRAM rate.  All copies are memset to the same value, so a copy is only
    // distinguished by its address.
    void runShape(const Shape &s, int iters, int copies) {
        const size_t bytes = static_cast<size_t>(s.k) * s.n * s.K / 8u;
        std::vector<void *> trellis(static_cast<size_t>(copies), nullptr);
        std::vector<tinycoder::exl3gpu::Exl3FusedLinear> lins(
                static_cast<size_t>(copies));
        float *suh = nullptr, *x = nullptr, *y = nullptr;
        CKP(cudaMalloc(&suh, sizeof(float) * s.k));
        CKP(cudaMalloc(&x, sizeof(float) * s.k));
        CKP(cudaMalloc(&y, sizeof(float) * s.n));
        std::vector<float> ones(static_cast<size_t>(s.n) > s.k ? s.n : s.k, 1.0f);
        CKP(cudaMemcpy(suh, ones.data(), sizeof(float) * s.k, cudaMemcpyHostToDevice));
        CKP(cudaMemcpy(x, ones.data(), sizeof(float) * s.k, cudaMemcpyHostToDevice));

        std::string err;
        for (int c = 0; c < copies; ++c) {
            CKP(cudaMalloc(&trellis[static_cast<size_t>(c)], bytes));
            CKP(cudaMemset(trellis[static_cast<size_t>(c)], 0, bytes));
            float *svh = nullptr;
            CKP(cudaMalloc(&svh, sizeof(float) * s.n));
            CKP(cudaMemcpy(svh, ones.data(), sizeof(float) * s.n,
                           cudaMemcpyHostToDevice));
            if (!tinycoder::exl3gpu::uploadFusedLinearDevice(
                        trellis[static_cast<size_t>(c)], suh, svh, s.k, s.n, s.K * 10u, 2,
                        lins[static_cast<size_t>(c)], err)) {
                std::fprintf(stderr, "upload failed for %ux%u: %s\n", s.k, s.n,
                             err.c_str());
                std::exit(3);
            }
        }

        // Warm-up (also faults the pages in, so the timed loop is pure kernel).
        // 30 ms of warm-up matters: this card idles at 300 MHz and needs a
        // sustained load before it boosts, so a 2 ms measurement reports the
        // clock, not the kernel (a 27% swing was observed before this loop
        // existed).
        for (int i = 0; i < 2 * copies; ++i) {
            const int c = i % copies;
            if (!tinycoder::exl3gpu::gemvFused(trellis[static_cast<size_t>(c)], x, y, 0,
                                               err)) {
                std::fprintf(stderr, "gemv failed: %s\n", err.c_str());
                std::exit(3);
            }
        }
        CKP(cudaDeviceSynchronize());

        cudaEvent_t a, b;
        CKP(cudaEventCreate(&a));
        CKP(cudaEventCreate(&b));

        // This card idles at 300 MHz and needs sustained load before it boosts;
        // a 2 ms measurement once reported 143 GB/s where a 400 ms measurement
        // reports 168 for the same kernel, and a cold first shape in a batch
        // reads 20% slow.  So: preheat to >= 150 ms, calibrate, then time three
        // ~300 ms passes and report the best (min) one.
        auto runN = [&](int n) {
            for (int i = 0; i < n; ++i) {
                const int c = i % copies;
                if (!tinycoder::exl3gpu::gemvFused(trellis[static_cast<size_t>(c)], x,
                                                   y, 0, err)) {
                    std::fprintf(stderr, "gemv failed: %s\n", err.c_str());
                    std::exit(3);
                }
            }
        };
        auto timed = [&](int n) {
            CKP(cudaEventRecord(a, 0));
            runN(n);
            CKP(cudaEventRecord(b, 0));
            CKP(cudaEventSynchronize(b));
            float ms = 0.0f;
            CKP(cudaEventElapsedTime(&ms, a, b));
            return ms;
        };

        // 1 s of preheat, not 150 ms: a per-process first measurement was
        // still reading 40% slow (88 vs 128 GB/s on the same config) with the
        // shorter ramp, which made a config sweep report the order it ran in.
        float heat = timed(16);
        while (heat < 1000.0f) heat += timed(16);
        const double perEst = static_cast<double>(heat) / 16.0;
        if (iters <= 0) {
            iters = static_cast<int>(300.0 / (perEst > 1e-6 ? perEst : 1e-6));
            if (iters < 20) iters = 20;
            if (iters > 400000) iters = 400000;
        }
        double best = 1e30;
        for (int pass = 0; pass < 3; ++pass) {
            const double per = timed(iters) / static_cast<double>(iters);
            if (per < best) best = per;
        }
        const double per = best;
        const double gbs = static_cast<double>(bytes) / (per * 1e6);

        // Per-call shape facts that explain the rate.
        const uint32_t tilesN = s.n / 16u;
        const uint32_t chunks = s.k / 128u;
        std::printf("%-6ux%-7u K=%u %-11s %8.3f ms  %7.1f GB/s  tilesN=%-5u chunks=%-5u %s\n",
                    s.k, s.n, s.K, s.what, per, gbs, tilesN, chunks,
                    std::getenv("TINYCODER_EXL3_K1DIAG") ? "[DIAG]" : "");

        for (int c = 0; c < copies; ++c) {
            tinycoder::exl3gpu::Exl3FusedLinear &lin = lins[static_cast<size_t>(c)];
            tinycoder::exl3gpu::releaseFusedLinear(lin);
            cudaFree(trellis[static_cast<size_t>(c)]);
            cudaFree(const_cast<float *>(lin.svh));
        }
        cudaFree(suh);
        cudaFree(x);
        cudaFree(y);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }

}// namespace

int main(int argc, char **argv) {
    if (!tinycoder::exl3gpu::available()) {
        std::fprintf(stderr, "EXL3 CUDA module unavailable\n");
        return 2;
    }
    int iters = 0;   // 0 = auto: enough iterations for ~400 ms
    int copies = 8;
    if (const char *e = std::getenv("XLP_ITERS")) iters = std::atoi(e);
    if (const char *e = std::getenv("XLP_COPIES")) copies = std::atoi(e);
    if (copies < 1) copies = 1;
    if (argc == 2) iters = std::atoi(argv[1]);
    if (argc >= 4) {
        Shape s{static_cast<uint32_t>(std::atoi(argv[1])),
                static_cast<uint32_t>(std::atoi(argv[2])),
                static_cast<uint32_t>(std::atoi(argv[3])), "arg"};
        if (argc >= 5) iters = std::atoi(argv[4]);
        runShape(s, iters, copies);
    } else {
        for (const Shape &s : kShapes) runShape(s, iters, copies);
    }
    tinycoder::exl3gpu::shutdown();
    return 0;
}
