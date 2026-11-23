// gemv_down_probe.cpp
//
// Does the Inkling routed-expert DOWN projection want the int8 Q8_1 path?
//
// The decode graph's gate/up projection is served by int8GemvExpert for every
// type the checkpoint uses, but down is not: its activation rows are strided by
// 2*ff (the SITU output is written into the first half of a 2*ff row) while the
// reduction length is ff, and int8GemvExpert accepts only a shared row
// (xStride == 0) or contiguous rows (xStride == cols).  Before adding a strided
// quantizer, a new int8 dot for NVFP4, and a wider dispatch guard, this probe
// answers the prior question: is the down shape limited by its dot product at
// all?
//
// It times, on the real 8-expert matrices of a checkpoint, three launches of
// the SAME shape:
//
//   float   inklingGemvExpertAsync   what runs today (xStride = 2*ff)
//   int8    int8GemvExpert           xStride = cols: the stride the int8 path
//                                    accepts, so this is its best case (its
//                                    activation reads are contiguous, which the
//                                    real down call cannot be)
//   gateup  the same pair on the gate/up shape of a layer whose type has an
//           int8 entry, as the control: this is the pair the engine already
//           prefers int8 for
//
// The interesting number is the int8/float ratio on down versus on gate/up.  If
// int8 does not win on down even in its best case, the dot product is not what
// bounds that shape and an NVFP4 int8 dot would buy nothing.
//
// Usage:
//   cmake --build build --target gemv_down_probe -j 20
//   ./build/unit_tests/gemv_down_probe <iq2.gguf> [--iters 200]
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <cstring>
#include <string>
#include <vector>

#include "GGUFLoader.hpp"
#include "Int8Gemm.hpp"
#include "gpu/InklingGPU.hpp"

namespace {

    double timeIters(const std::function<void()> &launch, uint32_t iters,
                     const char *what) {
        cudaEvent_t a = nullptr, b = nullptr;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        launch();// warmup / allocate
        cudaDeviceSynchronize();
        const cudaError_t warm = cudaGetLastError();
        if (warm != cudaSuccess) {
            std::printf("  %-8s launch error: %s\n", what,
                        cudaGetErrorString(warm));
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return -1.0;
        }
        cudaEventRecord(a);
        for (uint32_t i = 0; i < iters; ++i) launch();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return static_cast<double>(ms) / static_cast<double>(iters);
    }

    uint32_t typeBytes(uint32_t t) {
        switch (t) {
            case 40: return 36;
            case 42: return 18;
            case 16: return 66;
            case 17: return 74;
            case 19: return 50;
            case 22: return 82;
            default: return 0;
        }
    }

    struct Case {
        const char *name = nullptr;
        uint32_t rowsPer = 0;
        uint32_t cols = 0;
        uint32_t xStride = 0;// in floats
        bool int8Ok = false;
    };

} // namespace

int main(int argc, char **argv) {
    std::string model;
    uint32_t iters = 200;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (argv[i][0] != '-') {
            model = argv[i];
        }
    }
    if (model.empty()) {
        std::fprintf(stderr, "usage: gemv_down_probe <model.gguf> [--iters N]\n");
        return 2;
    }
    tinycoder::GGUFLoader loader;
    if (!loader.load(model)) {
        std::fprintf(stderr, "cannot read %s\n", model.c_str());
        return 2;
    }
    if (!tinycoder::gpu::int8GemmAvailable()) {
        std::fprintf(stderr, "int8 GEMV unavailable\n");
        return 2;
    }

    const uint32_t nUsed = 8;
    const int32_t ids[8] = {7, 3, 11, 0, 1, 2, 4, 5};
    int32_t *idx = nullptr;
    cudaMalloc(&idx, sizeof(ids));
    cudaMemcpy(idx, ids, sizeof(ids), cudaMemcpyHostToDevice);

    // Every expert shape the IQ2_XXS checkpoint actually runs in decode: the
    // two down types (Q2_0 and NVFP4) and the four gate/up types.
    struct Job {
        const char *tensor;
        uint32_t rowsPer;
        uint32_t cols;
        uint32_t xStride;
        bool gateUp;
    };
    const Job jobs[] = {
            {"blk.0.ffn_down_exps.weight", 2304, 1152, 2304, false},// Q2_0
            {"blk.5.ffn_down_exps.weight", 2304, 1152, 2304, false},// NVFP4
            {"blk.0.ffn_gate_up_exps.weight", 2304, 2304, 0, true}, // IQ1_S
            {"blk.5.ffn_gate_up_exps.weight", 2304, 2304, 0, true}, // IQ2_XXS
            {"blk.10.ffn_gate_up_exps.weight", 2304, 2304, 0, true},// IQ2_XS
            {"blk.16.ffn_gate_up_exps.weight", 2304, 2304, 0, true},// IQ2_S
    };

    for (const Job &j: jobs) {
        auto info = loader.getTensorInfo(j.tensor);
        if (!info) {
            std::printf("%s: missing\n", j.tensor);
            return 2;
        }
        const uint8_t *w = loader.getTensor(j.tensor);
        const uint32_t typeSize = typeBytes(info->type);
        if (typeSize == 0) {
            std::printf("%s: unsupported type %u\n", j.tensor, info->type);
            return 2;
        }
        const uint32_t bs = (info->type == 40 || info->type == 42) ? 64u : 256u;
        const uint64_t rowBytes = (uint64_t(j.cols) / bs) * typeSize;
        const uint64_t bytes =
                uint64_t(nUsed) * j.rowsPer * rowBytes;
        const size_t outElems = size_t(j.rowsPer) * nUsed;

        void *dw = nullptr, *dx = nullptr, *dy = nullptr;
        // Only experts 0..11 are ever indexed (ids max = 11), so 12 rows of
        // matrices is the whole working set the kernel can touch.
        cudaMalloc(&dw, (size_t(j.rowsPer) * 12) * rowBytes);
        cudaMalloc(&dx, std::max<uint32_t>(j.cols, j.xStride) * nUsed * sizeof(float));
        cudaMalloc(&dy, outElems * sizeof(float));
        cudaMemcpy(dw, w, (size_t(j.rowsPer) * 12) * rowBytes,
                   cudaMemcpyHostToDevice);
        cudaMemset(dx, 0, std::max<uint32_t>(j.cols, j.xStride) * nUsed *
                                  sizeof(float));

        const double msFloat = timeIters(
                [&] {
                    tinycoder::gpu::inklingGemvExpertAsync(
                            info->type, dw, static_cast<const float *>(dx),
                            j.xStride, static_cast<float *>(dy), j.rowsPer,
                            j.cols, idx, nUsed, nullptr);
                },
                iters, "float");
        // int8GemvExpert returns false for a type it has no kernel for (NVFP4,
        // the minority down type) WITHOUT launching anything.  Timing that
        // empty loop reported 2.8 us and "4335 GB/s" -- a meaningless number
        // that reads like a 45x win.  Ask the entry point first and report
        // REFUSED instead.
        const auto int8Launch = [&] {
            return tinycoder::gpu::int8GemvExpert(
                    static_cast<int>(info->type), dw,
                    static_cast<const float *>(dx), static_cast<float *>(dy),
                    j.rowsPer, j.cols, idx, nUsed, j.cols, nullptr);
        };
        const bool int8Serves = int8Launch();
        cudaGetLastError();
        const double msInt8 =
                int8Serves ? timeIters(int8Launch, iters, "int8") : -1.0;
        double ms4 = -1.0;
        if (j.gateUp) {
            // Decode gate/up now prefers the four-rows-per-warp entry.
            ms4 = timeIters(
                    [&] {
                        tinycoder::gpu::int8GemvExpert4(
                                static_cast<int>(info->type), dw,
                                static_cast<const float *>(dx),
                                static_cast<float *>(dy), j.rowsPer, j.cols,
                                idx, nUsed, nullptr);
                    },
                    iters, "int8x4");
        }
        const double gb = double(bytes) / 1e9;
        std::printf("%-30s type=%2u %s rowsPer=%u cols=%u\n", j.tensor,
                    info->type, j.gateUp ? "gate/up" : "down   ", j.rowsPer,
                    j.cols);
        std::printf("  float  %8.1f us  %6.1f GB/s over %.1f MB\n", msFloat * 1e3,
                    msFloat > 0 ? gb / (msFloat * 1e-3) : 0.0, gb * 1e3);
        if (msInt8 > 0) {
            std::printf("  int8   %8.1f us  %6.1f GB/s   int8/float = %.2fx\n",
                        msInt8 * 1e3, gb / (msInt8 * 1e-3), msFloat / msInt8);
        } else {
            std::printf("  int8   REFUSED for this type/stride\n");
        }
        if (ms4 > 0) {
            std::printf("  int8x4 %8.1f us  %6.1f GB/s   x4/int8 = %.2fx  "
                        "x4/float = %.2fx\n",
                        ms4 * 1e3, gb / (ms4 * 1e-3), msInt8 / ms4,
                        msFloat / ms4);
        }
        cudaFree(dw);
        cudaFree(dx);
        cudaFree(dy);
    }
    cudaFree(idx);
    return 0;
}
