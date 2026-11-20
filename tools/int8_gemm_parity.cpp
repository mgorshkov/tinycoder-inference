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

// int8_gemm_parity.cpp
//
// Numeric gate for the GGUF prefill int8 tensor-core GEMM (`int8Gemm`), on REAL
// weights read straight from a checkpoint. Answers one question: for the exact
// (type, rows, cols) of a real projection, does the kernel agree with a float64
// dot product over the dequantized weights, and by how much?
//
// Why this exists: every planned sm_86 prefill change (two-stage cp.async
// staging, a different scale epilogue, retuned tiles) perturbs the same loop.
// The model-level gates in AGENTS.md catch a wrong answer, but they cannot say
// which shape broke, and a whole-model logit is a sum over ~250 such shapes.
// This probe names the shape.
//
// What it can prove: the kernel's tiling, k-splitting, stream-k fixup and
// per-block scale folding are structurally right -- a wrong tile index, a
// mis-strided operand or a dropped fixup produces an error of order 1 here.
// What it measures beyond structure: the type's requantization floor.  Q6_K
// comes out at 1e-7 (the path is exact; only accumulation order remains) and
// Q4_K at 6.7e-4, because a 4-bit-plus-subblock-min block cannot be represented
// exactly in int8 for mma.s8. The gate is 2e-3 so that floor fits; read the
// printed number, not just the verdict.
//
// Usage:
//   cmake --build build-1.0 --target int8_gemm_parity -j 20
//   ./build-1.0/unit_tests/int8_gemm_parity <model.gguf> [--seqlen 8]
//       [--rows 32] [--max-shapes 12] [--tolerance 0.015] [--verbose]
//
// Exit code: 0 when every probed shape is inside tolerance, 1 otherwise.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"
#include "Int8Gemm.hpp"

namespace {

    const char *typeName(uint32_t t) {
        switch (t) {
            case 8: return "Q8_0";
            case 12: return "Q4_K";
            case 13: return "Q5_K";
            case 14: return "Q6_K";
            case 17: return "IQ2_XS";
            case 18: return "IQ3_XXS";
            case 21: return "IQ3_S";
            case 22: return "IQ2_S";
            case 142: return "PQ2_0";
            case 143: return "PTQ1_0";
            default: return "?";
        }
    }

    // The gate. See the header comment: this band is the Q8_1 activation
    // convention's width, not the kernel's error.
    // Measured floors on ornith/Ornith-1.5-9B-Q4_K_M: Q6_K 1.0e-7 (the path is
    // exact -- only accumulation order), Q4_K 6.7e-4.  The Q4_K term is the
    // kernel requantizing 4-bit-plus-subblock-min values into int8 for
    // mma.s8, which is inherent to an int8 tensor path and not a defect; the
    // gate therefore sits above it at 2e-3.  A structural error is 1e-1 or
    // worse, so nothing real hides in the gap -- but read the number: a Q6_K
    // shape drifting from 1e-7 to 1e-4 is a regression the verdict would still
    // call a pass.
    constexpr double kDefaultTolerance = 2e-3;

    struct Options {
        std::string model;
        // A list, not a scalar: the dispatch picks the tile width from seqLen
        // (>= 64 -> the 128-row tile, < 64 -> 64, < 32 -> 32), so a probe that
        // only ever ran at one seqLen verified one of the three kernel variants
        // and left the others unexercised -- the blind spot AGENTS.md names for
        // tiling dimensions.  `opt.seqLen` is the one being probed right now.
        std::vector<uint32_t> seqLens = {8};
        uint32_t seqLen = 8;
        uint32_t rowsToCheck = 32;
        uint32_t maxShapes = 12;
        double tolerance = kDefaultTolerance;
        bool verbose = false;
    };

    struct Shape {
        uint32_t type = 0;
        uint32_t rows = 0;
        uint32_t cols = 0;
        uint32_t blocksPerRow = 0;
        const uint8_t *weights = nullptr;
        std::string name;
    };

}// namespace

int main(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (std::strcmp(a, "--seqlen") == 0 && i + 1 < argc) {
            opt.seqLens.clear();
            for (const char *p = argv[++i]; p != nullptr && *p != '\0';) {
                char *end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p) break;
                if (v > 0 && v <= 8192) opt.seqLens.push_back(static_cast<uint32_t>(v));
                p = (*end == ',') ? end + 1 : nullptr;
            }
            if (opt.seqLens.empty()) opt.seqLens.push_back(8);
            opt.seqLen = opt.seqLens[0];
        } else if (std::strcmp(a, "--rows") == 0 && i + 1 < argc) {
            opt.rowsToCheck = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(a, "--max-shapes") == 0 && i + 1 < argc) {
            opt.maxShapes = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(a, "--tolerance") == 0 && i + 1 < argc) {
            opt.tolerance = std::atof(argv[++i]);
        } else if (std::strcmp(a, "--verbose") == 0) {
            opt.verbose = true;
        } else if (a[0] != '-') {
            opt.model = a;
        } else {
            std::fprintf(stderr, "[parity] unknown argument: %s\n", a);
            return 2;
        }
    }
    if (opt.model.empty()) {
        std::fprintf(stderr,
                     "usage: int8_gemm_parity <model.gguf> [--seqlen N[,N...]] [--rows N]\n"
                     "       [--max-shapes N] [--tolerance F] [--verbose]\n"
                     "  --seqlen takes a comma list and sweeps it: 1 and 8 exercise the\n"
                     "  32- and 64-row tile, 64 and 128 the 128-row one.\n");
        return 2;
    }

    tinycoder::GGUFLoader loader;
    // load(), not loadMetadata(): the tensor pointers come from the
    // file-backed mmap that only load() establishes, and getTensor() on a
    // metadata-only loader dereferences null.
    if (!loader.load(opt.model)) {
        std::fprintf(stderr, "[parity] cannot read %s\n", opt.model.c_str());
        return 2;
    }
    if (!tinycoder::gpu::int8GemmAvailable()) {
        std::fprintf(stderr, "[parity] int8 GEMM not available on this device/build\n");
        return 2;
    }

    // ---- collect one projection per (type, rows, cols) -----------------------
    // Same-shape duplicates add nothing: the kernel is selected by shape, so the
    // second occurrence runs identical code on identical data.
    std::vector<Shape> shapes;
    for (const auto &rec: loader.tensorRecords()) {
        if (shapes.size() >= opt.maxShapes) break;
        if (rec.shape.size() != 2) continue;
        const uint32_t cols = rec.shape[0];
        const uint32_t rows = rec.shape[1];
        // int8Gemm handles these; a GEMV-only type would report "unsupported"
        // and fall back, which is a different question than the one asked here.
        // Every type int8Gemm/int8Gemv claim to handle: Q8_0, the K-quants,
        // the i-quants and the two Prism ternary layouts.  An earlier version of
        // this filter listed only the K-quants and the ternaries, so the i-quant
        // checkpoints probed two shapes and looked clean.
        if (rec.type != 8 && rec.type != 12 && rec.type != 13 && rec.type != 14 &&
            rec.type != 22 && rec.type != 21 && rec.type != 17 && rec.type != 18 &&
            rec.type != 142 && rec.type != 143) {
            continue;
        }
        const uint32_t blockSize = tinycoder::ggmlBlockSize(rec.type);
        const uint32_t typeSize = tinycoder::ggmlTypeSize(rec.type);
        if (blockSize == 0 || typeSize == 0 || cols % blockSize != 0) continue;
        // The reference dequantizes a block at a time in fp32 and the kernel's
        // int8 staging wants a multiple of the 32-wide quant block; anything
        // else is a shape neither side claims to support.
        if (cols % 32u != 0u) continue;
        bool seen = false;
        for (const auto &s: shapes) {
            if (s.type == rec.type && s.rows == rows && s.cols == cols) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        const uint8_t *w = loader.getTensor(rec.name);
        if (w == nullptr) continue;
        shapes.push_back({rec.type, rows, cols, cols / blockSize, w, rec.name});
    }

    if (shapes.empty()) {
        std::fprintf(stderr, "[parity] no probed-type 2D tensors found in %s\n",
                     opt.model.c_str());
        return 2;
    }

    std::printf("[parity] model   %s\n", opt.model.c_str());
    std::printf("[parity] seqlen  ");
    for (uint32_t sv : opt.seqLens) std::printf("%u ", sv);
    std::printf("tokens, %u rows per shape, tolerance %.3g relRMS\n", opt.rowsToCheck,
                opt.tolerance);

    // Activation values are integers in [-127, 127] scaled by 2^-7, so each
    // 32-block's Q8_1 scale lands on 2^-7 and the quantized activations are the
    // integers themselves; that is what lets an fp64 dot over the *un*quantized
    // values serve as the reference.  See the header comment for the residual
    // one-LSB band this leaves.
    std::mt19937 rng(12345);
    const float actScale = std::ldexp(1.0f, -7);

    int failures = 0;
    int verified = 0;
    int skipped = 0;
    for (const auto &s: shapes) {
      for (const uint32_t seqLen : opt.seqLens) {
        opt.seqLen = seqLen;
        const size_t n = static_cast<size_t>(opt.seqLen) * s.rows;
        // Activations carry the shape's own row stride: int8Gemm indexes x as
        // [seqLen][cols], so a buffer padded to the largest probed cols would be
        // read at the wrong offset for every narrower shape.  (It was, and every
        // shape "failed" at relRMS ~1.3 -- the probe's first bug, and a reminder
        // that this tool is itself code that can be wrong.)
        std::vector<float> xHost(static_cast<size_t>(opt.seqLen) * s.cols, 0.0f);
        for (uint32_t t = 0; t < opt.seqLen; ++t) {
            for (uint32_t k = 0; k < s.cols; k += 32u) {
                float *blk = xHost.data() + static_cast<size_t>(t) * s.cols + k;
                const uint32_t nb = std::min<uint32_t>(32u, s.cols - k);
                for (uint32_t i = 0; i < nb; ++i) {
                    blk[i] = static_cast<float>(static_cast<int>(rng() % 255u) - 127);
                }
                blk[rng() % nb] = 127.0f;// pin the block max, so d is well defined
                for (uint32_t i = 0; i < nb; ++i) blk[i] *= actScale;
            }
        }
        // Every shape gets the same token rows only in the sense that the values
        // are drawn from one stream; the reference uses this same buffer, so the
        // two sides agree on the inputs by construction.
        float *xDev = nullptr;
        if (cudaMalloc(&xDev, xHost.size() * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "[parity] cudaMalloc(x) failed\n");
            return 2;
        }
        cudaMemcpy(xDev, xHost.data(), xHost.size() * sizeof(float), cudaMemcpyHostToDevice);
        float *outDev = nullptr;
        if (cudaMalloc(&outDev, n * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "[parity] cudaMalloc(out) failed for %s\n", s.name.c_str());
            return 2;
        }
        cudaMemset(outDev, 0, n * sizeof(float));
        const bool ok = tinycoder::gpu::int8Gemm(static_cast<int>(s.type), s.weights, xDev,
                                                 outDev, opt.seqLen, s.rows, s.cols,
                                                 s.blocksPerRow, nullptr);
        if (!ok) {
            ++skipped;
            std::printf("[parity] %-24s %-6s rows=%-6u cols=%-5u  SKIP (int8Gemm refused the shape)\n",
                        s.name.c_str(), typeName(s.type), s.rows, s.cols);
            cudaFree(outDev);
            cudaFree(xDev);
            continue;
        }
        std::vector<float> outHost(n);
        cudaMemcpy(outHost.data(), outDev, n * sizeof(float), cudaMemcpyDeviceToHost);

        // ---- float64 reference on a sample of rows --------------------------
        // One row at a time, straight from the mmap: the dequantize path is the
        // engine's own (GGMLDequantize), so the reference weights are by
        // definition the weights the fp16 twin would hold.
        const uint32_t rowBytes =
                static_cast<uint32_t>((static_cast<uint64_t>(s.cols) /
                                       tinycoder::ggmlBlockSize(s.type)) *
                                      tinycoder::ggmlTypeSize(s.type));
        const uint32_t stride = std::max<uint32_t>(1u, s.rows / opt.rowsToCheck);
        double sumSqErr = 0.0, sumSqRef = 0.0, maxAbsErr = 0.0, maxAbsRef = 0.0;
        size_t compared = 0;
        for (uint32_t r = 0; r < s.rows && compared < opt.rowsToCheck; r += stride) {
            // dequantize(), not dequantizeToF16(): the fp16 variant rounds every
            // dequantized weight to fp16 before returning it, and that rounding
            // (~2.4e-4 relative) was the dominant term in this probe's error
            // band -- it made the kernel look 4e-4 loose against a reference
            // that was itself the loose one.  fp32 here and the only remaining
            // difference is accumulation order.
            const std::vector<float> deq =
                    tinycoder::GGMLDequantize::dequantize(s.type, s.weights + static_cast<uint64_t>(r) * rowBytes,
                                                    s.cols);
            if (deq.size() != s.cols) {
                std::fprintf(stderr, "[parity] dequantize failed for type %u\n", s.type);
                cudaFree(outDev);
                cudaFree(xDev);
                return 2;
            }
            for (uint32_t t = 0; t < opt.seqLen; ++t) {
                const float *xr = xHost.data() + static_cast<size_t>(t) * s.cols;
                double ref = 0.0;
                for (uint32_t k = 0; k < s.cols; ++k) {
                    ref += static_cast<double>(deq[k]) * static_cast<double>(xr[k]);
                }
                const double got = static_cast<double>(outHost[static_cast<size_t>(t) * s.rows + r]);
                const double err = got - ref;
                sumSqErr += err * err;
                sumSqRef += ref * ref;
                maxAbsErr = std::max(maxAbsErr, std::fabs(err));
                maxAbsRef = std::max(maxAbsRef, std::fabs(ref));
            }
            ++compared;
        }
        cudaFree(outDev);
        cudaFree(xDev);

        const double relRms = sumSqRef > 0.0 ? std::sqrt(sumSqErr / sumSqRef) : 0.0;
        const double relMax = maxAbsRef > 0.0 ? maxAbsErr / maxAbsRef : 0.0;
        const bool pass = relRms <= opt.tolerance;
        ++verified;
        if (!pass) ++failures;
        std::printf("[parity] %-4s %-24s %-6s rows=%-6u cols=%-5u m=%-4u relRMS=%.3e "
                    "relMax=%.3e\n",
                    pass ? "PASS" : "FAIL", s.name.c_str(), typeName(s.type), s.rows,
                    s.cols, seqLen, relRms, relMax);
        if (opt.verbose || !pass) {
            std::printf("           weights=%s bpr=%u rowBytes=%u sampled=%zu\n",
                        typeName(s.type), s.blocksPerRow, rowBytes, compared);
        }
      }
    }

    // A skip is not a pass: this tool exits non-zero when it verified nothing,
    // because "every shape skipped" printed the same "0 shape(s) outside
    // tolerance" line as a clean run (it did, on a PQ2_0 checkpoint whose decode
    // is served by the ternary GEMV path, not by this entry point).
    std::printf("[parity] %d verified, %d skipped, %d outside tolerance\n", verified,
                skipped, failures);
    if (verified == 0) {
        std::printf("[parity] nothing was verified -- this probe cannot gate anything on this "
                    "checkpoint\n");
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
