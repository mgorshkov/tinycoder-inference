// int8_m16_probe.cu
//
// Throughput probe for the MoE token-block shape: how fast is the int8
// tensor-core GEMM at m = 8/16/32/64 with n = 512, k = 2048 (one routed
// expert's gate matrix), when the per-call launch cost is amortised over
// many back-to-back calls on one stream.
//
// Why it exists: a prefill with 512 tokens routes ~16 tokens to each of 256
// experts, so an expert GEMM is m~16.  llama.cpp computes those on the GPU
// while streaming the weights over PCIe at ~8.8 GB/s, which only needs
// ~264 GFLOPS to keep up.  Our resident experiment measured 36 GFLOPS for the
// same work, but that number included one int8Gemm call per (expert, matrix),
// i.e. 3000 launches per prefill.  This probe separates kernel rate from
// launch cost: if the rate at m=16 is far above 264 GFLOPS, a grouped kernel
// (grid.z = expert) would pay; if the rate itself is ~36 GFLOPS, no amount of
// batching saves it and the GPU-offload route is closed for good.
//
// Usage:
//   cmake --build build --target int8_m16_probe -j 20
//   ./build/unit_tests/int8_m16_probe [--m 8,16,32,64] [--iters 500]
//
// Exit code: 0 always (it reports; it does not judge).

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "Int8Gemm.hpp"

namespace {

    constexpr uint32_t kRows = 512;
    constexpr uint32_t kCols = 2048;
    constexpr uint32_t kBlockBytes = 34;// Q8_0: 1 half scale + 32 int8
    constexpr uint32_t kBlocksPerRow = kCols / 32;

    // Deterministic pseudo-random bytes: the probe measures time, not values,
    // but keeping magnitudes realistic avoids denormal/overflow surprises.
    uint32_t nextRand(uint32_t &s) {
        s = s * 1664525u + 1013904223u;
        return s;
    }

    double timeCalls(uint32_t m, uint32_t iters, bool &ok) {
        // Weights: one Q8_0 expert matrix in device memory.
        const size_t wBytes = static_cast<size_t>(kRows) * kBlocksPerRow * kBlockBytes;
        std::vector<uint8_t> wHost(wBytes);
        uint32_t s = 12345u;
        for (size_t i = 0; i < wBytes; ++i) wHost[i] = static_cast<uint8_t>(nextRand(s));
        // Valid fp16 scales (0x3C00 = 1.0) so the kernels take their normal path.
        for (uint32_t r = 0; r < kRows; ++r) {
            for (uint32_t b = 0; b < kBlocksPerRow; ++b) {
                uint8_t *blk = wHost.data() +
                               (static_cast<size_t>(r) * kBlocksPerRow + b) * kBlockBytes;
                blk[0] = 0x00;
                blk[1] = 0x3C;
            }
        }
        std::vector<float> xHost(static_cast<size_t>(m) * kCols);
        for (size_t i = 0; i < xHost.size(); ++i) {
            xHost[i] = (static_cast<int>(nextRand(s) % 2001) - 1000) / 1000.0f;
        }
        std::vector<float> outHost(static_cast<size_t>(m) * kRows);

        uint8_t *wDev = nullptr;
        float *xDev = nullptr;
        float *outDev = nullptr;
        if (cudaMalloc(&wDev, wBytes) != cudaSuccess ||
            cudaMalloc(&xDev, xHost.size() * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&outDev, outHost.size() * sizeof(float)) != cudaSuccess) {
            ok = false;
            return 0.0;
        }
        cudaMemcpy(wDev, wHost.data(), wBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(xDev, xHost.data(), xHost.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemset(outDev, 0, outHost.size() * sizeof(float));

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        // Warm-up: one call so any lazy init (smem attributes, JIT) is done.
        ok = tinycoder::gpu::int8Gemm(8, wDev, xDev, outDev, m, kRows, kCols,
                                      kBlocksPerRow, nullptr);
        cudaDeviceSynchronize();
        if (!ok) {
            cudaFree(wDev);
            cudaFree(xDev);
            cudaFree(outDev);
            return 0.0;
        }
        cudaEventRecord(e0);
        for (uint32_t i = 0; i < iters; ++i) {
            tinycoder::gpu::int8Gemm(8, wDev, xDev, outDev, m, kRows, kCols,
                                     kBlocksPerRow, nullptr);
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        cudaFree(wDev);
        cudaFree(xDev);
        cudaFree(outDev);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return ms;
    }

}// namespace

int main(int argc, char **argv) {
    std::vector<uint32_t> ms = {8, 16, 32, 64};
    uint32_t iters = 500;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            ms.clear();
            const char *p = argv[++i];
            while (*p != '\0') {
                ms.push_back(static_cast<uint32_t>(std::strtoul(p, nullptr, 10)));
                while (*p != '\0' && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        } else {
            std::fprintf(stderr, "usage: %s [--m 8,16,32,64] [--iters N]\n", argv[0]);
            return 2;
        }
    }
    if (!tinycoder::gpu::int8GemmAvailable()) {
        std::printf("[m16] int8 GEMM not available on this device\n");
        return 0;
    }
    std::printf("[m16] shape n=%u k=%u, iters=%u per m\n", kRows, kCols, iters);
    std::printf("%6s %12s %12s %12s\n", "m", "ms/call", "GFLOP/s", "vs 264 GF/s");
    for (uint32_t m : ms) {
        bool ok = true;
        const double msTotal = timeCalls(m, iters, ok);
        if (!ok || msTotal <= 0.0) {
            std::printf("%6u %12s\n", m, "REFUSED/FAILED");
            continue;
        }
        const double msPer = msTotal / iters;
        const double gflops = 2.0 * m * kRows * kCols / (msPer * 1e6);
        std::printf("%6u %12.1f %12.1f %12.2fx\n", m, msPer, gflops,
                    264.0 / gflops);
    }
    return 0;
}
