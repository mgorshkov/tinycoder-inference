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

// q2_0_mmq_parity.cpp
//
// Gate + microbenchmark for the Q2_0 int8 MMQ used by Inkling batched prefill
// (src/cpp/core/cuda/gemm/Q2_0Mma.cu).
//
// Parity: reads the real blk.*.ffn_down_exps.weight Q2_0 tensor from the
// checkpoint (18 B / 64 weights, fp16 d, value = d*(code-1)), runs the kernel
// on `--seqlen` activation rows, and compares every output against a float64
// dot over the fp32-dequantized weights.  Default activations are exactly
// representable in Q8_1 (integers in [-127,127] x 2^-7, one 127 per block), so
// d == 2^-7 and the gate (2e-3 relRMS, the same band int8_gemm_parity uses)
// measures decode/tiling/scale structure, not the int8 format's own activation
// floor.  `--random` draws continuous values instead and prints that floor;
// on this checkpoint it is 3.6e-3, which is why the model-level gate cannot be
// passed by this kernel alone.
//
// Microbench: reports the kernel's TFLOPS for the real down shape
// (cols=1152, rows=2304, m=16), both as one expert and as the full grouped
// layer (128 experts x capacity 16), so a pp change can be attributed to the
// kernel instead of the whole graph.
//
// Usage:
//   ./build-inkling/unit_tests/q2_0_mmq_parity <model.gguf> [--seqlen 16]
//       [--rows 256] [--tolerance 2e-3] [--bench 20] [--verbose]
//
// Exit code: 0 when the parity check passed and the benchmark ran, 1 otherwise.
#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"
#include "gpu/InklingKernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

    constexpr uint32_t kQ2_0Block = 64;
    constexpr uint32_t kQ2_0BlockBytes = 18;

    struct Options {
        std::string model;
        uint32_t seqLen = 16;
        uint32_t rowsToCheck = 256;
        double tolerance = 2e-3;
        int benchReps = 20;
        int onehot = -1;
        bool randomActs = false;
        bool verbose = false;
    };

    struct Q20Tensor {
        std::string name;
        const uint8_t *data = nullptr;
        uint32_t cols = 0;
        uint32_t rows = 0;
        uint32_t nExperts = 0;
    };



    double eventMs(cudaEvent_t a, cudaEvent_t b) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        return static_cast<double>(ms);
    }

    bool findTensor(tinycoder::GGUFLoader &loader, Q20Tensor &out) {
        bool found = false;
        for (const auto &rec: loader.tensorRecords()) {
            if (rec.type != 42 || rec.shape.size() != 3) continue;
            const uint32_t cols = static_cast<uint32_t>(rec.shape[0]);
            const uint32_t rows = static_cast<uint32_t>(rec.shape[1]);
            const uint32_t nExperts = static_cast<uint32_t>(rec.shape[2]);
            if (cols % kQ2_0Block != 0 || rows == 0 || nExperts == 0) continue;
            const uint8_t *w = loader.getTensor(rec.name);
            if (w == nullptr) continue;
            const bool prefer = rec.name.find("ffn_down_exps") != std::string::npos;
            if (!found || prefer) {
                out.name = rec.name;
                out.data = w;
                out.cols = cols;
                out.rows = rows;
                out.nExperts = nExperts;
                found = true;
                if (prefer) break;
            }
        }
        return out.data != nullptr;
    }

}// namespace

int main(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (std::strcmp(a, "--seqlen") == 0 && i + 1 < argc) {
            opt.seqLen = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(a, "--rows") == 0 && i + 1 < argc) {
            opt.rowsToCheck = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(a, "--tolerance") == 0 && i + 1 < argc) {
            opt.tolerance = std::atof(argv[++i]);
        } else if (std::strcmp(a, "--bench") == 0 && i + 1 < argc) {
            opt.benchReps = std::atoi(argv[++i]);
        } else if (std::strcmp(a, "--onehot") == 0 && i + 1 < argc) {
            opt.onehot = std::atoi(argv[++i]);
        } else if (std::strcmp(a, "--random") == 0) {
            opt.randomActs = true;
        } else if (std::strcmp(a, "--verbose") == 0) {
            opt.verbose = true;
        } else if (a[0] != '-') {
            opt.model = a;
        } else {
            std::fprintf(stderr, "[q2_0] unknown argument: %s\n", a);
            return 2;
        }
    }
    if (opt.model.empty()) {
        std::fprintf(stderr,
                     "usage: q2_0_mmq_parity <model.gguf> [--seqlen 16] [--rows N]\n"
                     "       [--tolerance 2e-3] [--bench N] [--verbose]\n");
        return 2;
    }
    if (opt.seqLen == 0 || opt.seqLen > 16) {
        std::fprintf(stderr, "[q2_0] --seqlen must be 1..16 (the kernel's M tile)\n");
        return 2;
    }

    tinycoder::GGUFLoader loader;
    if (!loader.load(opt.model)) {
        std::fprintf(stderr, "[q2_0] cannot read %s\n", opt.model.c_str());
        return 2;
    }
    Q20Tensor t;
    if (!findTensor(loader, t)) {
        std::fprintf(stderr, "[q2_0] no 3D Q2_0 (type 42) tensor in %s\n",
                     opt.model.c_str());
        return 2;
    }

    const uint32_t rowBytes = (t.cols / kQ2_0Block) * kQ2_0BlockBytes;
    std::printf("[q2_0] tensor  %s\n", t.name.c_str());
    std::printf("[q2_0] shape   cols=%u rows=%u experts=%u  rowBytes=%u "
                "(18 B / 64)\n",
                t.cols, t.rows, t.nExperts, rowBytes);
    std::printf("[q2_0] m=%u  tolerance %.3g relRMS\n", opt.seqLen,
                opt.tolerance);

    // The engine hands the kernel a device-resident copy (dev.down).  Passing
    // the loader's file-backed mmap instead would make every 18-byte weight
    // load a separate zero-copy PCIe transaction: measured ~0.3 TFLOPS and
    // ~600 ms for one grouped layer, i.e. two orders of magnitude of pure
    // transport.  Upload once and gate the thing the engine actually runs.
    const size_t wBytes = static_cast<size_t>(t.nExperts) * t.rows * rowBytes;
    uint8_t *wDev = nullptr;
    if (cudaMalloc(&wDev, wBytes) != cudaSuccess) {
        std::fprintf(stderr, "[q2_0] cudaMalloc(weights, %.1f MB) failed\n",
                     double(wBytes) / 1048576.0);
        return 2;
    }
    if (cudaMemcpy(wDev, t.data, wBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "[q2_0] cudaMemcpy(weights) failed\n");
        return 2;
    }

    // ---- activations: one row per token, independent per-token draws --------
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    // Default activations are exactly representable in Q8_1: integers in
    // [-127,127] scaled by 2^-7 with one 127 per 32-block, so d is exactly
    // 2^-7 and the quantizer is lossless.  The fp64 reference then gates the
    // kernel's decode/tiling/scale structure, not the int8 activation floor
    // (which is a property of the format, not of this kernel).  --random draws
    // continuous values instead and prints the floor that convention costs.
    std::vector<float> xHost(static_cast<size_t>(opt.seqLen) * t.cols);
    if (opt.randomActs) {
        for (float &v: xHost) v = dist(rng);
    } else {
        const float actScale = std::ldexp(1.0f, -7);
        for (uint32_t tt = 0; tt < opt.seqLen; ++tt) {
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
    if (opt.onehot >= 0 && static_cast<uint32_t>(opt.onehot) < t.cols) {
        // Diagnostic: x = e_K.  The output column r is then exactly W[r][K], so
        // a permutation, a scale mix-up or a wrong k mapping shows up as a
        // per-element diff instead of a 1.1 relRMS smudge.
        std::fill(xHost.begin(), xHost.end(), 0.0f);
        for (uint32_t tt = 0; tt < opt.seqLen; ++tt)
            xHost[static_cast<size_t>(tt) * t.cols + opt.onehot] = 1.0f;
        std::printf("[q2_0] one-hot k=%d\n", opt.onehot);
    }

    const size_t outCount = static_cast<size_t>(opt.seqLen) * t.rows;
    float *xDev = nullptr;
    float *outDev = nullptr;
    if (cudaMalloc(&xDev, xHost.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&outDev, outCount * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "[q2_0] cudaMalloc failed\n");
        return 2;
    }
    cudaMemcpy(xDev, xHost.data(), xHost.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(outDev, 0, outCount * sizeof(float));

    // expert 0: the reference points at t.data; the kernel is exercised on a
    // few experts below to check the [expert][row][k] stride.
    if (!tinycoder::gpu::int8GemmQ2_0(wDev, xDev, outDev, opt.seqLen, t.rows,
                                       t.cols, nullptr)) {
        std::fprintf(stderr, "[q2_0] int8GemmQ2_0 refused the shape\n");
        return 2;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "[q2_0] kernel failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return 2;
    }
    std::vector<float> outHost(outCount);
    cudaMemcpy(outHost.data(), outDev, outCount * sizeof(float),
               cudaMemcpyDeviceToHost);

    // ---- float64 reference over the dequantized rows -----------------------
    const uint32_t stride = std::max<uint32_t>(1u, t.rows / opt.rowsToCheck);
    double sumSqErr = 0.0, sumSqRef = 0.0, maxAbsErr = 0.0, maxAbsRef = 0.0;
    size_t compared = 0;
    // The standalone call covers expert 0 only; the grouped call below checks
    // the expert stride on two more experts.
    const uint32_t experts[] = {0u};
    for (uint32_t e : experts) {
        const uint8_t *we = t.data + static_cast<uint64_t>(e) * t.rows * rowBytes;
        for (uint32_t r = 0; r < t.rows; r += stride) {
            const std::vector<float> deq = tinycoder::GGMLDequantize::dequantize(
                    42, we + static_cast<uint64_t>(r) * rowBytes, t.cols);
            if (deq.size() != t.cols) {
                std::fprintf(stderr, "[q2_0] dequantize failed\n");
                return 2;
            }
            for (uint32_t tt = 0; tt < opt.seqLen; ++tt) {
                const float *xr = xHost.data() + static_cast<size_t>(tt) * t.cols;
                double ref = 0.0;
                if (opt.onehot >= 0) {
                    ref = static_cast<double>(deq[opt.onehot]);
                } else {
                    for (uint32_t k = 0; k < t.cols; ++k)
                        ref += static_cast<double>(deq[k]) * static_cast<double>(xr[k]);
                }
                const double got = static_cast<double>(
                        outHost[static_cast<size_t>(tt) * t.rows + r]);
                const double err = got - ref;
                sumSqErr += err * err;
                sumSqRef += ref * ref;
                maxAbsErr = std::max(maxAbsErr, std::fabs(err));
                maxAbsRef = std::max(maxAbsRef, std::fabs(ref));
            }
            ++compared;
        }
    }
    const double relRms = sumSqRef > 0.0 ? std::sqrt(sumSqErr / sumSqRef) : 0.0;
    const double relMax = maxAbsRef > 0.0 ? maxAbsErr / maxAbsRef : 0.0;
    bool parityOk = relRms <= opt.tolerance;
    std::printf("[q2_0] parity  %-4s  relRMS=%.3e relMax=%.3e  rows=%zu "
                "expert=0 m=%u\n",
                parityOk ? "PASS" : "FAIL", relRms, relMax, compared, opt.seqLen);
    if (opt.onehot >= 0 && opt.verbose) {
        const std::vector<float> d0 = tinycoder::GGMLDequantize::dequantize(
                42, t.data, t.cols);
        const uint32_t diagRows[] = {0, 1, 2, 7, 8, 9, 15, 16, 31, 32, 33, 64, 127};
        for (uint32_t r : diagRows) {
            const float got = outHost[r];
            int bestK = -1;
            float best = 1e30f;
            for (uint32_t k = 0; k < t.cols; ++k) {
                const float d = std::fabs(got - d0[k]);
                if (d < best) { best = d; bestK = static_cast<int>(k); }
            }
            const std::vector<float> dr = tinycoder::GGMLDequantize::dequantize(
                    42, t.data + static_cast<uint64_t>(r) * rowBytes, t.cols);
            std::printf("[q2_0] diag r=%u got=%.6f want W[r][K]=%.6f  "
                        "best-match-in-W[0][*]=k%d (|d|=%.2e)  W[K][r]=%.6f\n",
                        r, got, dr[opt.onehot], bestK, best,
                        opt.onehot < (int) t.rows
                                ? tinycoder::GGMLDequantize::dequantize(
                                          42,
                                          t.data + static_cast<uint64_t>(opt.onehot) * rowBytes,
                                          t.cols)[r]
                                : 0.0f);
        }
    }
    if (opt.verbose) {
        std::printf("[q2_0]         maxAbsErr=%.6e maxAbsRef=%.6e\n", maxAbsErr,
                    maxAbsRef);
    }

    // ---- grouped parity: expert stride + the [nExperts][cap][rows] layout -
    {
        const uint32_t cap = 16;
        const size_t gxCount = static_cast<size_t>(t.nExperts) * cap * t.cols;
        const size_t goutCount = static_cast<size_t>(t.nExperts) * cap * t.rows;
        float *gx = nullptr;
        float *gout = nullptr;
        int32_t *counts = nullptr;
        if (cudaMalloc(&gx, gxCount * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&gout, goutCount * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&counts, t.nExperts * sizeof(int32_t)) != cudaSuccess) {
            std::fprintf(stderr, "[q2_0] cudaMalloc(grouped parity) failed\n");
            return 2;
        }
        std::vector<float> gxHost(gxCount);
        for (uint32_t e = 0; e < t.nExperts; ++e)
            for (uint32_t r = 0; r < cap; ++r)
                for (uint32_t k = 0; k < t.cols; ++k)
                    gxHost[(static_cast<size_t>(e) * cap + r) * t.cols + k] =
                            xHost[static_cast<size_t>(r % opt.seqLen) * t.cols + k];
        std::vector<int32_t> countsHost(t.nExperts, opt.seqLen);
        cudaMemcpy(gx, gxHost.data(), gxCount * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(counts, countsHost.data(), t.nExperts * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        if (!tinycoder::gpu::int8GemmQ2_0Grouped(wDev, gx, gout, counts, cap,
                                                  t.rows, t.cols, t.nExperts,
                                                  nullptr) ||
            cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "[q2_0] grouped kernel failed\n");
            return 2;
        }
        std::vector<float> goutHost(goutCount);
        cudaMemcpy(goutHost.data(), gout, goutCount * sizeof(float),
                   cudaMemcpyDeviceToHost);
        double gErr = 0.0, gRef = 0.0;
        size_t gCompared = 0;
        const uint32_t gExperts[] = {0u, t.nExperts / 2u, t.nExperts - 1u};
        for (uint32_t e : gExperts) {
            const uint8_t *we =
                    t.data + static_cast<uint64_t>(e) * t.rows * rowBytes;
            for (uint32_t r = 0; r < t.rows; r += stride) {
                const std::vector<float> deq =
                        tinycoder::GGMLDequantize::dequantize(
                                42, we + static_cast<uint64_t>(r) * rowBytes,
                                t.cols);
                for (uint32_t tt = 0; tt < opt.seqLen; ++tt) {
                    const float *xr =
                            xHost.data() + static_cast<size_t>(tt) * t.cols;
                    double ref = 0.0;
                    for (uint32_t k = 0; k < t.cols; ++k)
                        ref += static_cast<double>(deq[k]) *
                               static_cast<double>(xr[k]);
                    const double got = static_cast<double>(
                            goutHost[(static_cast<size_t>(e) * cap + tt) *
                                             t.rows +
                                     r]);
                    const double err = got - ref;
                    gErr += err * err;
                    gRef += ref * ref;
                }
                ++gCompared;
            }
        }
        const double gRel = gRef > 0.0 ? std::sqrt(gErr / gRef) : 0.0;
        const bool gOk = gRel <= opt.tolerance;
        parityOk = parityOk && gOk;
        std::printf("[q2_0] grouped %-4s  relRMS=%.3e  rows=%zu "
                    "experts={0,%u,%u} m=%u\n",
                    gOk ? "PASS" : "FAIL", gRel, gCompared, t.nExperts / 2u,
                    t.nExperts - 1u, opt.seqLen);
        cudaFree(gx);
        cudaFree(gout);
        cudaFree(counts);
    }

    // ---- microbench: one expert and the full grouped layer -----------------
    int benchOk = 1;
    if (opt.benchReps > 0) {
        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        const double flopsOne = 2.0 * opt.seqLen * t.rows * t.cols;

        auto timeIt = [&](const std::function<bool()> &fn) -> double {
            for (int i = 0; i < 5; ++i) fn();
            cudaEventRecord(ev0);
            for (int i = 0; i < opt.benchReps; ++i) fn();
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            return eventMs(ev0, ev1) / opt.benchReps;
        };

        const double fullMs = timeIt([&] {
            return tinycoder::gpu::int8GemmQ2_0(
                    wDev, xDev, outDev, opt.seqLen, t.rows, t.cols, nullptr);
        });
        const double kernelMs = timeIt([&] {
            return tinycoder::gpu::int8GemmQ2_0(
                    wDev, xDev, outDev, opt.seqLen, t.rows, t.cols, nullptr,
                    false);
        });
        std::printf("[q2_0] bench   1 expert m=%u: quantize+gemm %.3f ms "
                    "%.1f TFLOPS | gemm %.3f ms %.1f TFLOPS\n",
                    opt.seqLen, fullMs, flopsOne / (fullMs * 1e9), kernelMs,
                    flopsOne / (kernelMs * 1e9));

        // Full grouped layer: 128 experts x capacity 16.  All groups full is
        // the worst case (two row halves issued); counts=1 is the common
        // runtime case (one token per expert at m=16 x nUsed=8 / 128).
        const uint32_t cap = 16;
        const size_t gxCount = static_cast<size_t>(t.nExperts) * cap * t.cols;
        const size_t goutCount = static_cast<size_t>(t.nExperts) * cap * t.rows;
        float *gx = nullptr;
        float *gout = nullptr;
        int32_t *counts = nullptr;
        if (cudaMalloc(&gx, gxCount * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&gout, goutCount * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&counts, t.nExperts * sizeof(int32_t)) != cudaSuccess) {
            std::fprintf(stderr, "[q2_0] cudaMalloc(grouped) failed\n");
            cudaFree(xDev);
            cudaFree(outDev);
            return 2;
        }
        std::vector<float> gxHost(gxCount);
        std::vector<int32_t> countsHost(t.nExperts, 16);
        for (size_t i = 0; i < gxCount; ++i)
            gxHost[i] = xHost[i % xHost.size()];
        cudaMemcpy(gx, gxHost.data(), gxCount * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(counts, countsHost.data(), t.nExperts * sizeof(int32_t),
                   cudaMemcpyHostToDevice);

        const double flopsLayer = 2.0 * t.nExperts * cap * t.rows * t.cols;
        const double gFull = timeIt([&] {
            return tinycoder::gpu::int8GemmQ2_0Grouped(
                    wDev, gx, gout, counts, cap, t.rows, t.cols, t.nExperts,
                    nullptr);
        });
        const double gKern = timeIt([&] {
            return tinycoder::gpu::int8GemmQ2_0Grouped(
                    wDev, gx, gout, counts, cap, t.rows, t.cols, t.nExperts,
                    nullptr, false);
        });
        std::printf("[q2_0] bench   %u experts x cap %u (full): quantize+gemm "
                    "%.3f ms %.1f TFLOPS | gemm %.3f ms %.1f TFLOPS\n",
                    t.nExperts, cap, gFull, flopsLayer / (gFull * 1e9), gKern,
                    flopsLayer / (gKern * 1e9));

        countsHost.assign(t.nExperts, 1);
        cudaMemcpy(counts, countsHost.data(), t.nExperts * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        // The common runtime case: one token per expert (m=16 x nUsed=8 over
        // 128 experts), so only the first row half is issued.
        const double gOneFull = timeIt([&] {
            return tinycoder::gpu::int8GemmQ2_0Grouped(
                    wDev, gx, gout, counts, cap, t.rows, t.cols, t.nExperts,
                    nullptr);
        });
        const double gOneKern = timeIt([&] {
            return tinycoder::gpu::int8GemmQ2_0Grouped(
                    wDev, gx, gout, counts, cap, t.rows, t.cols, t.nExperts,
                    nullptr, false);
        });
        std::printf("[q2_0] bench   %u experts x count 1: quantize+gemm %.3f ms "
                    "| gemm %.3f ms (one row half issued)\n",
                    t.nExperts, gOneFull, gOneKern);

        cudaFree(gx);
        cudaFree(gout);
        cudaFree(counts);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }

    cudaFree(wDev);
    cudaFree(xDev);
    cudaFree(outDev);

    std::printf("[q2_0] %s\n", parityOk && benchOk ? "PASSED" : "FAILED");
    return parityOk && benchOk ? 0 : 1;
}
