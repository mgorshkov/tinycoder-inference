// int8_gemm_shape_probe.cpp
//
// Synthetic shape probe for the int8 prefill GEMM.
//
// The rented 3090 report (docs/reports/2026-11-20-3090-recheck.md, defect 1)
// logged `[int8] launch failed (type=14 k=2048 n=8192 m=495 …)` thousands of
// times per run.  Type 14 is Q6_K, and no checkpoint in this tree has a Q6_K
// tensor with k=2048 or k=6144, so the parity tools cannot reach those shapes.
// This tool builds them: it lifts a *real* Q6_K block run out of a checkpoint
// (so the block layout and magnitudes are the ones the kernels will see),
// repeats it to the requested width, and compares the int8 GEMM against an
// fp64 dot over the same blocks dequantized by GGMLDequantize.
//
// Usage:
//   cmake --build build --target int8_gemm_shape_probe -j 20
//   ./build/unit_tests/int8_gemm_shape_probe [--shape k,n,m,type ...]
//       [--source <model.gguf>] [--tolerance <rel>]
//
// Default shapes are the three the report logged plus m=8 (the documented
// floor) and m=7 (below it, which must be refused, not attempted).
//
// Exit code: 0 when every shape passed (or was refused for m<8), 1 otherwise,
// 2 on bad usage or a missing Q6_K source block.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"
#include "Int8Gemm.hpp"

namespace {

    struct Probe {
        uint32_t k;    // cols
        uint32_t n;    // rows
        uint32_t m;    // seqLen / tokens
        uint32_t type; // GGML type
    };

    const char *typeName(uint32_t t) {
        switch (t) {
            case 8: return "Q8_0";
            case 12: return "Q4_K";
            case 13: return "Q5_K";
            case 14: return "Q6_K";
            default: return "?";
        }
    }

    void usage(const char *argv0) {
        std::printf("usage: %s [--shape k,n,m,type ...] [--source <model.gguf>] "
                    "[--tolerance <rel>]\n"
                    "  default shapes: 2048,8192,495,14  6144,5120,512,14  "
                    "2048,512,495,14  6144,5120,8,14  6144,5120,7,14\n"
                    "  m<8 is expected to be refused by int8Gemm (it is the "
                    "documented floor), not to fail.\n",
                    argv0);
    }

} // namespace

int main(int argc, char **argv) {
    std::vector<Probe> probes;
    std::string source = "/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf";
    double tolerance = 2e-3;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (std::strcmp(a, "--shape") == 0 && i + 1 < argc) {
            unsigned k = 0, n = 0, m = 0, t = 0;
            if (std::sscanf(argv[++i], "%u,%u,%u,%u", &k, &n, &m, &t) != 4) {
                usage(argv[0]);
                return 2;
            }
            probes.push_back({k, n, m, t});
        } else if (std::strcmp(a, "--source") == 0 && i + 1 < argc) {
            source = argv[++i];
        } else if (std::strcmp(a, "--tolerance") == 0 && i + 1 < argc) {
            tolerance = std::atof(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (probes.empty()) {
        probes = {{2048, 8192, 495, 14}, {6144, 5120, 512, 14},
                  {2048, 512, 495, 14},  {6144, 5120, 8, 14},
                  {6144, 5120, 7, 14}};
    }
    if (!tinycoder::gpu::int8GemmAvailable()) {
        std::fprintf(stderr, "[probe] int8 GEMM not available on this device/build\n");
        return 2;
    }

    // ---- a real Q6_K block run to build the synthetic rows from --------------
    tinycoder::GGUFLoader loader;
    if (!loader.load(source)) {
        std::fprintf(stderr, "[probe] cannot read %s\n", source.c_str());
        return 2;
    }
    const uint32_t blockSize = tinycoder::ggmlBlockSize(14);
    const uint32_t typeSize = tinycoder::ggmlTypeSize(14);
    if (blockSize != 256 || typeSize != 210) {
        std::fprintf(stderr, "[probe] unexpected Q6_K block geometry %u/%u\n", blockSize,
                     typeSize);
        return 2;
    }
    const uint8_t *srcRow = nullptr;
    uint32_t srcBlocks = 0;
    for (const auto &rec: loader.tensorRecords()) {
        if (rec.type != 14 || rec.shape.size() != 2) continue;
        if (rec.shape[0] % blockSize != 0 || rec.shape[0] < 8u * blockSize) continue;
        srcRow = loader.getTensor(rec.name);
        srcBlocks = rec.shape[0] / blockSize;
        if (srcRow != nullptr) break;
    }
    if (srcRow == nullptr || srcBlocks < 8) {
        std::fprintf(stderr, "[probe] no Q6_K tensor with at least 8 blocks in %s\n",
                     source.c_str());
        return 2;
    }
    constexpr uint32_t kPatternBlocks = 8; // 2048 values, the report's smallest k
    std::printf("[probe] source %s: %u Q6_K blocks per row, using %u as the pattern\n",
                source.c_str(), srcBlocks, kPatternBlocks);

    // Activations: integers in [-127, 127] scaled by 2^-7, so every 32-wide
    // Q8_1 block quantises exactly and an fp64 dot over the unquantised values
    // is the honest reference (same construction as int8_gemm_parity).
    std::mt19937 rng(20261122);
    const float actScale = std::ldexp(1.0f, -7);

    int failures = 0;
    int verified = 0;
    for (const Probe &p: probes) {
        const uint32_t blocksPerRow = p.k / blockSize;
        if (p.k % blockSize != 0 || p.k < kPatternBlocks * blockSize) {
            std::printf("[probe] %s k=%-5u n=%-6u m=%-4u  SKIP (k must be a multiple of %u "
                        "and at least %u)\n",
                        typeName(p.type), p.k, p.n, p.m, blockSize,
                        kPatternBlocks * blockSize);
            continue;
        }

        // Weights: row r starts at block offset (r % (srcBlocks - kPatternBlocks + 1)),
        // so adjacent rows are different blocks of the same real tensor.
        const size_t rowBytes = static_cast<size_t>(blocksPerRow) * typeSize;
        std::vector<uint8_t> w(static_cast<size_t>(p.n) * rowBytes);
        const uint32_t offsets = srcBlocks - kPatternBlocks + 1;
        for (uint32_t r = 0; r < p.n; ++r) {
            uint8_t *dst = w.data() + static_cast<size_t>(r) * rowBytes;
            const uint32_t base = r % offsets;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                const uint32_t srcBlock = base + (b % kPatternBlocks);
                std::memcpy(dst + static_cast<size_t>(b) * typeSize,
                            srcRow + static_cast<size_t>(srcBlock) * typeSize, typeSize);
            }
        }

        std::vector<float> xHost(static_cast<size_t>(p.m) * p.k, 0.0f);
        for (uint32_t t = 0; t < p.m; ++t) {
            for (uint32_t k = 0; k < p.k; k += 32u) {
                float *blk = xHost.data() + static_cast<size_t>(t) * p.k + k;
                const uint32_t nb = std::min<uint32_t>(32u, p.k - k);
                for (uint32_t i = 0; i < nb; ++i) {
                    blk[i] = static_cast<float>(static_cast<int>(rng() % 255u) - 127);
                }
                blk[rng() % nb] = 127.0f; // pin the block max so d is well defined
                for (uint32_t i = 0; i < nb; ++i) blk[i] *= actScale;
            }
        }

        uint8_t *wDev = nullptr;
        float *xDev = nullptr, *outDev = nullptr;
        const size_t outN = static_cast<size_t>(p.m) * p.n;
        if (cudaMalloc(&wDev, w.size()) != cudaSuccess ||
            cudaMalloc(&xDev, xHost.size() * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&outDev, outN * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "[probe] cudaMalloc failed for %s k=%u n=%u m=%u\n",
                         typeName(p.type), p.k, p.n, p.m);
            cudaFree(wDev);
            cudaFree(xDev);
            cudaFree(outDev);
            return 2;
        }
        cudaMemcpy(wDev, w.data(), w.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(xDev, xHost.data(), xHost.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemset(outDev, 0, outN * sizeof(float));

        const bool ok = tinycoder::gpu::int8Gemm(
                static_cast<int>(p.type), wDev, xDev, outDev, p.m, p.n, p.k,
                blocksPerRow, nullptr);
        if (!ok) {
            if (p.m < 8) {
                std::printf("[probe] %s k=%-5u n=%-6u m=%-4u  ok (m<8 refused by design)\n",
                            typeName(p.type), p.k, p.n, p.m);
                ++verified;
            } else {
                const cudaError_t e = cudaGetLastError();
                std::printf("[probe] %s k=%-5u n=%-6u m=%-4u  REFUSED%s%s\n",
                            typeName(p.type), p.k, p.n, p.m,
                            e == cudaSuccess ? "" : " (last CUDA error: ",
                            e == cudaSuccess ? "" : cudaGetErrorString(e));
                ++failures;
            }
            cudaFree(outDev);
            cudaFree(xDev);
            cudaFree(wDev);
            continue;
        }

        std::vector<float> outHost(outN);
        cudaMemcpy(outHost.data(), outDev, outN * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // Reference: fp64 dot over every dequantised row for a sample of rows.
        const uint32_t rowsToCheck = std::min<uint32_t>(p.n, 16u);
        const uint32_t stride = std::max<uint32_t>(1u, p.n / rowsToCheck);
        std::vector<float> rowF(static_cast<size_t>(p.k));
        double sumSqErr = 0.0, sumSqRef = 0.0;
        for (uint32_t r = 0; r < p.n; r += stride) {
            const uint8_t *row = w.data() + static_cast<size_t>(r) * rowBytes;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                tinycoder::GGMLDequantize::dequantizeBlock(
                        14, row + static_cast<size_t>(b) * typeSize,
                        rowF.data() + static_cast<size_t>(b) * blockSize, blockSize);
            }
            for (uint32_t t = 0; t < p.m; ++t) {
                const float *xr = xHost.data() + static_cast<size_t>(t) * p.k;
                double ref = 0.0;
                for (uint32_t i = 0; i < p.k; ++i) {
                    ref += static_cast<double>(rowF[i]) * xr[i];
                }
                const double err =
                        static_cast<double>(outHost[static_cast<size_t>(t) * p.n + r]) - ref;
                sumSqErr += err * err;
                sumSqRef += ref * ref;
            }
        }
        const double relRms = sumSqRef > 0.0 ? std::sqrt(sumSqErr / sumSqRef) : 0.0;
        const bool pass = relRms <= tolerance;
        std::printf("[probe] %s k=%-5u n=%-6u m=%-4u  %s relRMS=%.3e\n",
                    typeName(p.type), p.k, p.n, p.m, pass ? "PASS" : "FAIL", relRms);
        if (pass) ++verified;
        else ++failures;

        cudaFree(outDev);
        cudaFree(xDev);
        cudaFree(wDev);
    }

    std::printf("[probe] %d verified, %d failed\n", verified, failures);
    return failures == 0 ? 0 : 1;
}
