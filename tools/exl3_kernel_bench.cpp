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

// exl3_kernel_bench.cpp
//
// EXL3 GPU (Turing / Ampere / Ada, incl. RTX 2080 Ti / 3090 / 4090) kernel-layer
// benchmark.  Exercises the same
// module validated by tools/exl3_gpu_parity.cpp -- uploadLinear -> lazy
// reconstruct (decode trellis -> left-H128*suh -> right-H128*svh -> transpose)
// -> F32 GEMV -- and reports:
//
//   * one-off reconstruction latency + achieved bandwidth, and
//   * steady-state GEMV latency + effective F32-weight bandwidth
//
// for the layer shapes of a real checkpoint (default: Qwen3.5-27B EXL3 1.60bpw,
// config.json: hidden 5120, 64 layers = 48 GDN + 16 full-attention, MLP 17408,
// GQA 24/4 heads x 256, GDN key_dim 2048 / value_dim 6144).  It then multiplies
// the per-shape numbers by their occurrence count to project a single decode
// step (seqLen = 1) across the whole model.
//
// The trellis is synthetic (random codes) -- the numbers are shape-driven, so a
// real checkpoint would give the same timings while letting this run without a
// model file.  Requires a CUDA build (target is only added when ENABLE_CUDA=ON).
//
// Usage:
//   ./build-avx2/unit_tests/exl3_kernel_bench

#include "EXL3Codec.hpp"
#include "EXL3Kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

    using Clock = std::chrono::steady_clock;

    double elapsedMs(Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    // Pack `codes[256]` (one per tile) into the LSB-first wrapped-funnel layout
    // (inverse of EXL3Codec::unpackTile).  Writes 16*K uint16 words.
    void packTile(const uint16_t *codes, uint32_t K, uint16_t *packed) {
        const uint32_t nWords = 8u * K;
        std::vector<uint32_t> words(nWords, 0u);
        const uint32_t mask = (K >= 16u) ? 0xFFFFu : ((1u << K) - 1u);
        for (uint32_t j = 0; j < 256; ++j) {
            const uint32_t pos = j * K;
            const uint32_t word = pos >> 5;
            const uint32_t within = pos & 31u;
            uint32_t code = codes[j] & mask;
            words[word] |= (code << within);
            if (within + K > 32u) {
                const uint32_t spill = within + K - 32u;
                words[(word + 1u) % nWords] |= (code >> (K - spill));
            }
        }
        for (uint32_t i = 0; i < 16u * K; ++i) {
            packed[i] = static_cast<uint16_t>((words[i >> 1] >> (16u * (i & 1u))) & 0xFFFFu);
        }
    }

    // Build a synthetic signed trellis for a k x n linear at K bpw.
    void buildSynthetic(uint32_t k, uint32_t n, uint32_t K, uint32_t seed,
                        std::vector<uint16_t> &packed, std::vector<float> &suh,
                        std::vector<float> &svh) {
        const uint32_t tilesK = k / 16u, tilesN = n / 16u;
        const uint32_t tileU16 = 16u * K;
        packed.assign(static_cast<size_t>(tilesK) * tilesN * tileU16, 0u);
        std::mt19937 rng(seed);
        const uint32_t codeMax = (1u << K) - 1u;
        std::vector<uint16_t> codes(256);
        for (uint32_t t = 0; t < tilesK * tilesN; ++t) {
            for (uint32_t j = 0; j < 256; ++j) {
                codes[j] = static_cast<uint16_t>(rng() % (codeMax + 1u));
            }
            packTile(codes.data(), K, packed.data() + static_cast<size_t>(t) * tileU16);
        }
        suh.resize(k);
        svh.resize(n);
        for (auto &v: suh) v = (rng() & 1u) ? -1.0f : 1.0f;
        for (auto &v: svh) v = (rng() & 1u) ? -1.0f : 1.0f;
    }

    struct Shape {
        const char *name;
        uint32_t k;
        uint32_t n;
        int count;
    };

    // One measured shape.
    struct Result {
        const char *name;
        uint32_t k, n;
        double reconMs;
        double gemvMs;
        double gemvGBs;
    };

    constexpr int64_t kTargetMacs = 800000000;// ~0.8 GMAC of work per timing round
    constexpr int kRounds = 7;

    // Upload `s`, run enough warmups to reach steady clocks, then measure the
    // steady-state gemv() call as the MINIMUM mean over several rounds (the GPU
    // is shared, so the min is the least-contended, most reproducible estimate).
    // `coldOverrideMs`, when non-null, receives the min (cold - warm) delta over
    // a few fresh uploads = the one-off reconstruct cost.
    Result benchmarkShape(const Shape &s, uint32_t K, int codebook, uint32_t seed,
                          std::string &err, double *coldOverrideMs = nullptr) {
        std::vector<uint16_t> packed;
        std::vector<float> suh, svh;
        buildSynthetic(s.k, s.n, K, seed, packed, suh, svh);

        tinycoder::exl3gpu::Exl3HostLinear lin{};
        lin.packed = packed.data();
        lin.k = s.k;
        lin.n = s.n;
        lin.bitrate10 = K * 10u;
        lin.codebook = codebook;
        lin.suh = suh.data();
        lin.svh = svh.data();

        std::vector<float> x(s.k);
        std::mt19937 rng(seed ^ 0x9e3779b9u);
        for (auto &v: x) v = static_cast<float>(rng() % 2000u) / 1000.0f - 1.0f;
        std::vector<float> y(s.n);

        const int reps = static_cast<int>(std::clamp(
                static_cast<double>(kTargetMacs) / (static_cast<double>(s.k) * s.n),
                10.0, 4000.0));

        double bestWarm = 1e30;
        double bestCold = 1e30;
        const int uploads = coldOverrideMs ? 3 : 1;
        for (int u = 0; u < uploads; ++u) {
            int64_t h = tinycoder::exl3gpu::uploadLinear(lin, err);
            if (h < 0) {
                std::fprintf(stderr, "uploadLinear(%s): %s\n", s.name, err.c_str());
                std::exit(1);
            }
            // Cold call = lazy reconstruct + one GEMV.
            auto t0 = Clock::now();
            tinycoder::exl3gpu::gemv(h, x.data(), y.data(), err);
            auto t1 = Clock::now();
            const double coldMs = elapsedMs(t0, t1);

            // Warmups, then the timed rounds.
            for (int r = 0; r < 30; ++r) {
                tinycoder::exl3gpu::gemv(h, x.data(), y.data(), err);
            }
            for (int round = 0; round < kRounds; ++round) {
                t0 = Clock::now();
                for (int r = 0; r < reps; ++r) {
                    tinycoder::exl3gpu::gemv(h, x.data(), y.data(), err);
                }
                t1 = Clock::now();
                bestWarm = std::min(bestWarm, elapsedMs(t0, t1) / reps);
            }
            bestCold = std::min(bestCold, coldMs);
            tinycoder::exl3gpu::releaseLinear(h);
        }

        Result r{};
        r.name = s.name;
        r.k = s.k;
        r.n = s.n;
        r.reconMs = std::max(0.0, bestCold - bestWarm);
        r.gemvMs = bestWarm;
        // The GEMV reads the reconstructed F32 weight (n*k*4 bytes) per token.
        r.gemvGBs = bestWarm > 0.0
                            ? (static_cast<double>(s.k) * s.n * 4.0 / 1e9) / (bestWarm / 1e3)
                            : 0.0;
        if (coldOverrideMs) *coldOverrideMs = r.reconMs;
        return r;
    }

}// namespace

int main(int /*argc*/, char ** /*argv*/) {
    if (!tinycoder::exl3gpu::available()) {
        std::fprintf(stderr, "EXL3 CUDA module unavailable (no device?)\n");
        return 2;
    }

    cudaDeviceProp prop{};
    int dev = 0;
    cudaGetDevice(&dev);
    const bool haveProp = (cudaGetDeviceProperties(&prop, dev) == cudaSuccess);

    std::printf("=====================================================================\n");
    std::printf(" EXL3 GPU kernel benchmark  |  %s (sm_%d%d)%s\n",
                haveProp ? prop.name : "CUDA device", haveProp ? prop.major : 7,
                haveProp ? prop.minor : 5,
                haveProp ? "" : " [props unavailable]");
    std::printf(" shapes: Qwen3.5-27B EXL3 1.60bpw  (hidden 5120, MLP 17408,\n");
    std::printf("         48 GDN + 16 full-attn layers, GQA 24/4 x 256,\n");
    std::printf("         GDN key_dim 2048 / value_dim 6144)\n");
    std::printf(" synthetic trellis (random codes); times are shape-driven\n");
    std::printf("=====================================================================\n\n");

    std::string err;

    // ---- 1. Reconstruct-latency sweep over bitrate / codebook --------------
    // Reconstruct is a one-off per linear (done once at upload for the
    // correctness-first path, or once lazily on the native path).
    std::printf("--- reconstruction pipeline (decode -> H128 -> transpose), one-off ---\n");
    std::printf("%-22s %8s %10s %12s %12s\n", "shape (k x n)", "bpw", "packed MB",
                "recon ms", "recon GB/s");
    {
        const Shape rshape{"attn_qkv 5120x10240", 5120, 10240, 1};
        struct B {
            uint32_t K;
            int cb;
            const char *label;
        };
        const B books[] = {
                {1, 0, "3INST"}, {2, 0, "3INST"}, {3, 0, "3INST"},
                {2, 1, "mcg"},   {2, 2, "mul1"}, {3, 2, "mul1"},
        };
        for (const B &b: books) {
            double reconMs = 0.0;
            Result r = benchmarkShape(rshape, b.K, b.cb, 1u, err, &reconMs);
            // packed bytes = tilesK*tilesN*16*K*2
            const double packedMb = static_cast<double>(r.k / 16) * (r.n / 16) * 16 * b.K * 2 / 1e6;
            // raw trellis read + F32 scratch write + F32 wT write
            const double bytes = packedMb * 1e6 + static_cast<double>(r.k) * r.n * 8.0;
            const double gbs = reconMs > 0 ? (bytes / 1e9) / (reconMs / 1e3) : 0.0;
            std::printf("%-22s %8.1f %10.1f %12.3f %12.1f  (%s)\n", rshape.name, b.K * 1.0,
                        packedMb, reconMs, gbs, b.label);
        }
    }

    // ---- 2. Steady-state GEMV over the real decode shapes ------------------
    const Shape shapes[] = {
            {"recurrent.attn_qkv", 5120, 10240, 48},
            {"recurrent.attn_gate", 5120, 6144, 48},
            {"recurrent.ssm_out", 6144, 5120, 48},
            {"mlp.gate", 5120, 17408, 64},
            {"mlp.up", 5120, 17408, 64},
            {"mlp.down", 17408, 5120, 64},
            {"attn.q", 5120, 12288, 16},
            {"attn.k", 5120, 1024, 16},
            {"attn.v", 5120, 1024, 16},
            {"attn.o", 6144, 5120, 16},
    };

    std::printf("\n--- steady-state GEMV (F32 reconstructed weight, 2.0 bpw) ---\n");
    std::printf("%-20s %7s %7s %6s %11s %14s\n", "shape", "k", "n", "count", "gemv ms",
                "GB/s (F32 W)");
    double totalMs = 0.0;
    double totalMacs = 0.0;
    for (const Shape &s: shapes) {
        Result r = benchmarkShape(s, 2, 0, 7u, err);
        std::printf("%-20s %7u %7u %6d %11.4f %14.1f\n", r.name, r.k, r.n, s.count, r.gemvMs,
                    r.gemvGBs);
        totalMs += r.gemvMs * s.count;
        totalMacs += static_cast<double>(s.k) * s.n * s.count;
    }

    std::printf("\n--- model decode projection (one token, seqLen = 1) ---\n");
    std::printf("  layer GEMV workload          : %.2f GMAC\n", totalMacs / 1e9);
    std::printf("  total layer GEMV time        : %.3f ms\n", totalMs);
    std::printf("  projected decode throughput  : %.1f tok/s\n", totalMs > 0 ? 1000.0 / totalMs : 0.0);
    std::printf("  (excludes GDN recurrence, attention, conv1d, norms and the LM\n");
    std::printf("   head -- the F32 GEMV of the layer projections dominates decode)\n");

    const double f32Gb = totalMacs * 4.0 / 1e9;   // F32 weight bytes (1 MAC = 1 weight)
    const double packedGb = totalMacs * 1.6 / 8.0 / 1e9;// 1.6 bpw packed trellis
    std::printf("\n  note: the GEMV re-reads %.1f GB of F32 weights per token;\n", f32Gb);
    std::printf("        the native (lazy reconstruct) path instead keeps the\n");
    std::printf("        packed trellis resident (~%.1f GB at 1.6 bpw) and\n", packedGb);
    std::printf("        materializes a single layer's F32 weight at a time.\n");

    tinycoder::exl3gpu::shutdown();
    return 0;
}
