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

// Parity + microbenchmark for the grouped gate/up expert MMQ
// (src/cpp/core/cuda/inkling/InklingExpertMma.cu).
//
// Parity: reads a real blk.*.ffn_gate_up_exps.weight of type IQ2_XXS (16),
// IQ2_XS (17), IQ1_S (19) or IQ2_S (22), or a blk.*.ffn_down_exps.weight of
// type NVFP4 (40) / Q2_0 (42), from the checkpoint, builds the layer-major
// graph's grouped layout ([nExperts][capacity][cols] activations, counts[e]
// live rows, mean m*nUsed/nExperts = 8), runs the kernel and compares sampled
// output rows against a float64 dot over the fp32-dequantized weights.
// Default activations are exactly representable in Q8_1 (integers in
// [-127,127] x 2^-7, one 127 per 32 values) so d == 2^-7 and the 2e-3 gate
// measures the decode/tiling/scale structure, not the int8 activation floor;
// --random prints that floor.  `--onehot K` makes every activation row e_K, so
// output column r must equal W[r][K] exactly and a permutation or a scale
// mix-up shows up per element.
//
// Microbench: times the whole grouped layer (128 experts, capacity 128, the
// real gate/up shape) for three paths so the section delta can be attributed:
//   float  inklingGemvBatchedGroupedAsync -- the grouped float decoder, exact
//   slice  int8Gemv in 4-row per-expert slices -- the current exact path for
//          IQ2_XS/IQ2_S (types 17/22 only; int8Gemv refuses 16)
//   mma    inklingExpertMmaGrouped -- this kernel
//
// Usage:
//   ./build/unit_tests/inkling_expert_mma_parity <model.gguf>
//       [--type 22] [--seqlen 8] [--rows 128] [--tolerance 2e-3]
//       [--bench 10] [--random] [--onehot K] [--verbose]
//
// Exit code: 0 when the parity gate passed, 1 otherwise.
#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"
#include "Int8Gemm.hpp"
#include "gpu/InklingGPU.hpp"
#include "gpu/InklingKernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

    struct Options {
        std::string model;
        uint32_t type = 22;
        uint32_t seqLen = 8;// mean routed rows per expert at m=128, nUsed=8
        uint32_t rowsToCheck = 128;
        uint32_t capacity = 128;
        double tolerance = 2e-3;
        int benchReps = 10;
        int onehot = -1;
        bool randomActs = false;
        bool verbose = false;
        // --dump PATH: write the raw [nExperts][capacity][rows] fp32 output so a
        // fresh process running a different kernel form can be compared bit for
        // bit (cmp), which is the gate a "same numerics, faster tile" claim
        // needs; the float64 relRMS below cannot see an accumulation reorder.
        std::string dump;
    };

    struct Tensor {
        std::string name;
        const uint8_t *data = nullptr;
        uint32_t type = 0;
        uint32_t cols = 0;
        uint32_t rows = 0;
        uint32_t nExperts = 0;
    };

    uint32_t blockElems(uint32_t type) {
        if (type == 40 || type == 42) return 64;
        if (type == 0 || type == 30 || type == 8) return 32;
        return 256;
    }
    uint32_t blockBytes(uint32_t type) {
        switch (type) {
            case 16: return 66;
            case 17: return 74;
            case 19: return 50;
            case 22: return 82;
            case 40: return 36;
            case 42: return 18;
            default: return 0;
        }
    }

    double eventMs(cudaEvent_t a, cudaEvent_t b) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        return static_cast<double>(ms);
    }

    bool findTensor(tinycoder::GGUFLoader &loader, uint32_t type, Tensor &out) {
        bool found = false;
        for (const auto &rec: loader.tensorRecords()) {
            if (rec.type != type || rec.shape.size() != 3) continue;
            // Q2_0 (42) and NVFP4 (40) ship on the down projection, the IQ
            // family on gate/up.
            const bool down = type == 40 || type == 42;
            if (rec.name.find(down ? "ffn_down_exps" : "ffn_gate_up_exps") ==
                std::string::npos)
                continue;
            const uint32_t cols = static_cast<uint32_t>(rec.shape[0]);
            const uint32_t rows = static_cast<uint32_t>(rec.shape[1]);
            const uint32_t nExperts = static_cast<uint32_t>(rec.shape[2]);
            const uint32_t kPer =
                    (type == 40 || type == 42) ? 64u : 256u;
            if (cols % kPer != 0u || rows % 64u != 0u || nExperts == 0) continue;
            const uint8_t *w = loader.getTensor(rec.name);
            if (w == nullptr) continue;
            out.name = rec.name;
            out.data = w;
            out.type = type;
            out.cols = cols;
            out.rows = rows;
            out.nExperts = nExperts;
            found = true;
            break;
        }
        return found;
    }

}// namespace

int main(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (std::strcmp(a, "--type") == 0 && i + 1 < argc) {
            opt.type = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(a, "--seqlen") == 0 && i + 1 < argc) {
            opt.seqLen = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(a, "--rows") == 0 && i + 1 < argc) {
            opt.rowsToCheck = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(a, "--tolerance") == 0 && i + 1 < argc) {
            opt.tolerance = std::atof(argv[++i]);
        } else if (std::strcmp(a, "--bench") == 0 && i + 1 < argc) {
            opt.benchReps = std::atoi(argv[++i]);
        } else if (std::strcmp(a, "--onehot") == 0 && i + 1 < argc) {
            opt.onehot = std::atoi(argv[++i]);
        } else if (std::strcmp(a, "--dump") == 0 && i + 1 < argc) {
            opt.dump = argv[++i];
        } else if (std::strcmp(a, "--random") == 0) {
            opt.randomActs = true;
        } else if (std::strcmp(a, "--verbose") == 0) {
            opt.verbose = true;
        } else if (a[0] != '-') {
            opt.model = a;
        } else {
            std::fprintf(stderr, "[expert-mma] unknown argument: %s\n", a);
            return 2;
        }
    }
    if (opt.model.empty()) {
        std::fprintf(stderr,
                     "usage: inkling_expert_mma_parity <model.gguf> [--type 22]\n"
                     "       [--seqlen 8] [--rows 128] [--tolerance 2e-3]\n"
                     "       [--bench 10] [--random] [--onehot K] [--verbose]\n"
                     "       [--dump out.f32]\n");
        return 2;
    }
    if (opt.type != 16u && opt.type != 17u && opt.type != 19u &&
        opt.type != 22u && opt.type != 40u && opt.type != 42u) {
        std::fprintf(stderr, "[expert-mma] --type must be 16 (IQ2_XXS), 17 "
                             "(IQ2_XS), 19 (IQ1_S), 22 (IQ2_S), 40 (NVFP4 "
                             "down) or 42 (Q2_0 down)\n");
        return 2;
    }

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        std::fprintf(stderr, "[expert-mma] no CUDA device\n");
        return 2;
    }

    tinycoder::GGUFLoader loader;
    if (!loader.load(opt.model)) {
        std::fprintf(stderr, "[expert-mma] cannot read %s\n", opt.model.c_str());
        return 2;
    }
    Tensor t;
    if (!findTensor(loader, opt.type, t)) {
        std::fprintf(stderr, "[expert-mma] no 3D ffn_gate_up_exps tensor of "
                             "type %u in %s\n", opt.type, opt.model.c_str());
        return 2;
    }
    const uint32_t rowBytes = (t.cols / blockElems(t.type)) * blockBytes(t.type);
    std::printf("[expert-mma] tensor %s\n", t.name.c_str());
    std::printf("[expert-mma] type=%u cols=%u rows=%u experts=%u rowBytes=%u "
                "capacity=%u mean_rows=%u\n",
                t.type, t.cols, t.rows, t.nExperts, rowBytes, opt.capacity,
                opt.seqLen);
    std::printf("[expert-mma] gate %.3g relRMS%s\n", opt.tolerance,
                opt.randomActs ? " (--random: format floor, gate 5e-3)"
                               : " (Q8_1-exact activations)");

    // One device-resident copy, as the engine holds it (dev.gateUp).
    const size_t wBytes = static_cast<size_t>(t.nExperts) * t.rows * rowBytes;
    uint8_t *wDev = nullptr;
    if (cudaMalloc(&wDev, wBytes) != cudaSuccess) {
        std::fprintf(stderr, "[expert-mma] cudaMalloc(weights, %.1f MB) failed\n",
                     double(wBytes) / 1048576.0);
        return 2;
    }
    if (cudaMemcpy(wDev, t.data, wBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "[expert-mma] cudaMemcpy(weights) failed\n");
        return 2;
    }

    // ---- activations -------------------------------------------------------
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    const uint32_t cap = opt.capacity;
    std::vector<float> xHost(static_cast<size_t>((opt.seqLen * t.nExperts) / 8u) *
                             t.cols);
    if (opt.onehot >= 0 && static_cast<uint32_t>(opt.onehot) < t.cols) {
        std::fill(xHost.begin(), xHost.end(), 0.0f);
        for (uint32_t tt = 0;
             tt < xHost.size() / t.cols; ++tt)
            xHost[static_cast<size_t>(tt) * t.cols + opt.onehot] = 1.0f;
        std::printf("[expert-mma] one-hot k=%d\n", opt.onehot);
    } else if (opt.randomActs) {
        for (float &v: xHost) v = dist(rng);
    } else {
        const float actScale = std::ldexp(1.0f, -7);
        for (uint32_t tt = 0; tt < xHost.size() / t.cols; ++tt) {
            for (uint32_t k = 0; k < t.cols; k += 32u) {
                float *blk = xHost.data() + static_cast<size_t>(tt) * t.cols + k;
                for (uint32_t i = 0; i < 32u; ++i)
                    blk[i] = static_cast<float>(
                                     static_cast<int>(rng() % 255u) - 127) *
                             actScale;
                blk[rng() % 32u] = 127.0f * actScale;
            }
        }
    }

    // Grouped layout: nUsed=8 experts per token, m pairs spread over the
    // expert set, exactly as moeGroupBuildKernel packs them.
    const uint32_t nUsed = 8;
    const uint32_t mTokens = (opt.seqLen * t.nExperts) / nUsed;
    std::vector<int32_t> counts(t.nExperts, 0);
    std::vector<std::vector<uint32_t>> tokens(t.nExperts);
    for (uint32_t tt = 0; tt < mTokens; ++tt) {
        for (uint32_t u = 0; u < nUsed; ++u) {
            const uint32_t e = (tt * nUsed + u) % t.nExperts;
            tokens[e].push_back(tt);
            ++counts[e];
        }
    }
    uint32_t maxCount = 0;
    for (uint32_t e = 0; e < t.nExperts; ++e)
        maxCount = std::max(maxCount, static_cast<uint32_t>(counts[e]));
    if (maxCount > cap) {
        std::fprintf(stderr, "[expert-mma] capacity %u too small for %u rows\n",
                     cap, maxCount);
        return 2;
    }
    const size_t gxCount = static_cast<size_t>(t.nExperts) * cap * t.cols;
    const size_t goutCount = static_cast<size_t>(t.nExperts) * cap * t.rows;
    std::vector<float> gxHost(gxCount, 0.0f);
    for (uint32_t e = 0; e < t.nExperts; ++e) {
        for (uint32_t slot = 0; slot < tokens[e].size(); ++slot) {
            const uint32_t tt = tokens[e][slot];
            std::memcpy(gxHost.data() +
                                (static_cast<size_t>(e) * cap + slot) * t.cols,
                        xHost.data() + static_cast<size_t>(tt) * t.cols,
                        t.cols * sizeof(float));
        }
    }
    float *gx = nullptr;
    float *gout = nullptr;
    int32_t *countsDev = nullptr;
    if (cudaMalloc(&gx, gxCount * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&gout, goutCount * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&countsDev, t.nExperts * sizeof(int32_t)) != cudaSuccess) {
        std::fprintf(stderr, "[expert-mma] cudaMalloc(grouped) failed\n");
        return 2;
    }
    if (cudaMemcpy(gx, gxHost.data(), gxCount * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(countsDev, counts.data(), t.nExperts * sizeof(int32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "[expert-mma] cudaMemcpy(grouped) failed\n");
        return 2;
    }

    // ---- run the MMA -------------------------------------------------------
    if (!tinycoder::gpu::inklingExpertMmaSupported(t.type)) {
        std::fprintf(stderr, "[expert-mma] type %u not supported\n", t.type);
        return 2;
    }
    if (!tinycoder::gpu::inklingExpertMmaGrouped(wDev, gx, gout, t.rows, t.cols,
                                                 cap, countsDev, t.nExperts,
                                                 t.type, nullptr)) {
        std::fprintf(stderr, "[expert-mma] kernel refused the shape\n");
        return 2;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "[expert-mma] kernel failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return 2;
    }
    std::vector<float> goutHost(goutCount);
    cudaMemcpy(goutHost.data(), gout, goutCount * sizeof(float),
               cudaMemcpyDeviceToHost);
    if (!opt.dump.empty()) {
        FILE *f = std::fopen(opt.dump.c_str(), "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "[expert-mma] cannot write %s\n",
                         opt.dump.c_str());
            return 2;
        }
        std::fwrite(goutHost.data(), sizeof(float), goutHost.size(), f);
        std::fclose(f);
        std::printf("[expert-mma] dumped %zu floats to %s\n", goutHost.size(),
                    opt.dump.c_str());
    }

    // ---- float64 reference -------------------------------------------------
    const uint32_t stride = std::max<uint32_t>(1u, t.rows / opt.rowsToCheck);
    double sumSqErr = 0.0, sumSqRef = 0.0, maxAbsErr = 0.0, maxAbsRef = 0.0;
    size_t compared = 0;
    const std::vector<uint32_t> experts = {0u, t.nExperts / 2u, t.nExperts - 1u};
    for (uint32_t e: experts) {
        const uint8_t *we = t.data + static_cast<uint64_t>(e) * t.rows * rowBytes;
        for (uint32_t r = 0; r < t.rows; r += stride) {
            const std::vector<float> deq = tinycoder::GGMLDequantize::dequantize(
                    t.type, we + static_cast<uint64_t>(r) * rowBytes, t.cols);
            if (deq.size() != t.cols) {
                std::fprintf(stderr, "[expert-mma] dequantize failed\n");
                return 2;
            }
            for (uint32_t slot = 0; slot < tokens[e].size(); ++slot) {
                const float *xr =
                        xHost.data() + static_cast<size_t>(tokens[e][slot]) * t.cols;
                double ref = 0.0;
                if (opt.onehot >= 0) {
                    ref = static_cast<double>(deq[opt.onehot]);
                } else {
                    for (uint32_t k = 0; k < t.cols; ++k)
                        ref += static_cast<double>(deq[k]) *
                               static_cast<double>(xr[k]);
                }
                const double got = static_cast<double>(
                        goutHost[(static_cast<size_t>(e) * cap + slot) * t.rows + r]);
                const double err = got - ref;
                sumSqErr += err * err;
                sumSqRef += ref * ref;
                maxAbsErr = std::max(maxAbsErr, std::fabs(err));
                maxAbsRef = std::max(maxAbsRef, std::fabs(ref));
                ++compared;
            }
        }
    }
    const double relRms = sumSqRef > 0.0 ? std::sqrt(sumSqErr / sumSqRef) : 0.0;
    const double relMax = maxAbsRef > 0.0 ? maxAbsErr / maxAbsRef : 0.0;
    // Continuous activations sit at the Q8_1 format's own floor (~3.7e-3
    // relRMS here, the same number q2_0_mmq_parity reports for the Q2_0 MMQ),
    // which is above the 2e-3 structural gate by construction; --random
    // documents that floor and gates it at 5e-3 instead of failing on it.
    const double gate = opt.randomActs ? 5e-3 : opt.tolerance;
    bool parityOk = relRms <= gate;
    std::printf("[expert-mma] parity %-4s  relRMS=%.3e relMax=%.3e  pairs=%zu "
                "experts=%zu rows-stride=%u\n",
                parityOk ? "PASS" : "FAIL", relRms, relMax, compared,
                experts.size(), stride);
    if (opt.verbose) {
        std::printf("[expert-mma]         maxAbsErr=%.6e maxAbsRef=%.6e\n",
                    maxAbsErr, maxAbsRef);
    }
    if (opt.onehot >= 0) {
        // Every live output element must equal W[r][K] exactly.
        double worst = 0.0;
        for (uint32_t e: experts) {
            const uint8_t *we =
                    t.data + static_cast<uint64_t>(e) * t.rows * rowBytes;
            for (uint32_t r = 0; r < t.rows; r += std::max(1u, t.rows / 64u)) {
                const std::vector<float> deq =
                        tinycoder::GGMLDequantize::dequantize(
                                t.type, we + static_cast<uint64_t>(r) * rowBytes,
                                t.cols);
                for (uint32_t slot = 0; slot < tokens[e].size(); ++slot) {
                    const double got = static_cast<double>(
                            goutHost[(static_cast<size_t>(e) * cap + slot) *
                                             t.rows +
                                     r]);
                    worst = std::max(worst, std::fabs(got - deq[opt.onehot]));
                }
            }
        }
        std::printf("[expert-mma] one-hot maxAbsDiff=%.3e (must be 0 for an "
                    "exact decode)\n", worst);
        // First rows of expert 0, to separate a k permutation from a scale or
        // column mix-up.
        if (opt.verbose) {
            const uint8_t *we = t.data;
            for (uint32_t r = 0; r < 6 && r < t.rows; ++r) {
                const std::vector<float> deq =
                        tinycoder::GGMLDequantize::dequantize(
                                t.type, we + static_cast<uint64_t>(r) * rowBytes,
                                t.cols);
                const float got = goutHost[static_cast<size_t>(r)];
                int bestK = -1;
                float best = 1e30f;
                for (uint32_t k = 0; k < t.cols; ++k) {
                    const float d = std::fabs(got - deq[k]);
                    if (d < best) {
                        best = d;
                        bestK = static_cast<int>(k);
                    }
                }
                std::printf("[expert-mma] r=%u got=%.6f want W[r][K]=%.6f  "
                            "best-match-in-W[r][*]=k%d (|d|=%.2e)\n",
                            r, got, deq[opt.onehot], bestK, best);
            }
            // Worst sampled elements with their coordinates: a whole n-tile or
            // warp wrong points at the staging/column mapping.
            for (uint32_t e: experts) {
                if (tokens[e].empty()) continue;
                const uint8_t *wee =
                        t.data + static_cast<uint64_t>(e) * t.rows * rowBytes;
                for (uint32_t r = 0; r < t.rows; r += 36u) {
                    const std::vector<float> deq =
                            tinycoder::GGMLDequantize::dequantize(
                                    t.type,
                                    wee + static_cast<uint64_t>(r) * rowBytes,
                                    t.cols);
                    const float got = goutHost[(static_cast<size_t>(e) * cap) *
                                                       t.rows +
                                               r];
                    if (std::fabs(got - deq[opt.onehot]) > 1.0f)
                        std::printf("[expert-mma] BAD e=%u slot=0 r=%u "
                                    "(warp %u, n-tile %u) got=%.6f want=%.6f\n",
                                    e, r, (r % 64u) / 8u, r / 64u, got,
                                    deq[opt.onehot]);
                }
            }
        }
    }

    // ---- microbench: the whole grouped layer, three paths ------------------
    if (opt.benchReps > 0) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        auto time = [&](const char *name, auto &&fn) {
            // warm up
            fn();
            if (cudaDeviceSynchronize() != cudaSuccess) {
                std::printf("[expert-mma] bench %-6s FAILED: %s\n", name,
                            cudaGetErrorString(cudaGetLastError()));
                return -1.0;
            }
            cudaEventRecord(ev0);
            for (int r = 0; r < opt.benchReps; ++r) fn();
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            const double ms = eventMs(ev0, ev1) / opt.benchReps;
            if (cudaGetLastError() != cudaSuccess) {
                std::printf("[expert-mma] bench %-6s FAILED: %s\n", name,
                            cudaGetErrorString(cudaGetLastError()));
                return -1.0;
            }
            std::printf("[expert-mma] bench %-6s %8.3f ms/grouped-layer  "
                        "(%.1f GB/s of weight bytes)\n",
                        name, ms, wBytes / (ms * 1e-3) / 1e9);
            return ms;
        };
        const double gb = static_cast<double>(wBytes);

        // MMA: one launch per layer (quantize + gemm).
        const double tMma = time("mma", [&] {
            tinycoder::gpu::inklingExpertMmaGrouped(wDev, gx, gout, t.rows,
                                                   t.cols, cap, countsDev,
                                                   t.nExperts, t.type, nullptr);
        });
        // Same kernel with the activation quantization already done, so the
        // GEMM alone is visible.
        const double tMmaQ = time("mma-q0", [&] {
            tinycoder::gpu::inklingExpertMmaGrouped(wDev, gx, gout, t.rows,
                                                   t.cols, cap, countsDev,
                                                   t.nExperts, t.type, nullptr,
                                                   false);
        });
        (void) tMmaQ;

        // Float grouped decoder: the exact path's down-style kernel.
        const double tFloat = time("float", [&] {
            tinycoder::gpu::inklingGemvBatchedGroupedAsync(
                    t.type, wDev, gx, gout, t.rows, t.cols, cap, countsDev,
                    t.nExperts, nullptr);
        });
        (void) tFloat;

        // Exact int8 4-row per-expert slices: what the graph runs today for
        // types 17/22.
        double tSlice = -1.0;
        if (t.type == 17u || t.type == 22u) {
            std::vector<int32_t> countsHost = counts;
            std::vector<const uint8_t *> base(t.nExperts);
            tSlice = time("slice", [&] {
                for (uint32_t e = 0; e < t.nExperts; ++e) {
                    const int32_t cnt = countsHost[e];
                    if (cnt <= 0) continue;
                    const uint8_t *we =
                            reinterpret_cast<const uint8_t *>(wDev) +
                            static_cast<uint64_t>(e) * t.rows * rowBytes;
                    const float *xg =
                            gx + static_cast<size_t>(e) * cap * t.cols;
                    float *og = gout + static_cast<size_t>(e) * cap * t.rows;
                    for (uint32_t s0 = 0; s0 < static_cast<uint32_t>(cnt);
                         s0 += 4) {
                        const uint32_t mm = std::min(4u, static_cast<uint32_t>(cnt) - s0);
                        tinycoder::gpu::int8Gemv(
                                t.type, we, xg + static_cast<size_t>(s0) * t.cols,
                                og + static_cast<size_t>(s0) * t.rows, mm, t.rows,
                                t.cols, nullptr);
                    }
                }
            });
            (void) base;
        }
        if (tMma > 0.0 && tFloat > 0.0)
            std::printf("[expert-mma] mma vs float: %.2fx; weight bytes %.1f MB\n",
                        tFloat / tMma, gb / 1048576.0);
        if (tMma > 0.0 && tSlice > 0.0)
            std::printf("[expert-mma] mma vs slice: %.2fx\n", tSlice / tMma);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }

    cudaFree(gx);
    cudaFree(gout);
    cudaFree(countsDev);
    cudaFree(wDev);
    std::printf("[expert-mma] %s\n", parityOk ? "PASSED" : "FAILED");
    return parityOk ? 0 : 1;
}
