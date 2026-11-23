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
// What it CANNOT prove: that the numbers are unchanged to the last bit, because
// the reference and the kernel sum k in different orders, and because the
// kernel quantizes the activations to Q8_1 while the reference uses the values
// that quantization was built from. The gate below is therefore 1.5e-2 relative
// RMS, which is the width of the Q8_1 scale-convention band, not the width of
// the kernel's own error: on an exact-activation basis the two agree to ~1e-6.
// Read the printed number, not just the verdict -- a band that drifts from
// 1.1e-2 to 1.4e-2 is a regression even while it passes.
//
// Usage:
//   cmake --build build-1.0 --target int8_gemm_parity -j 20
//   ./build-1.0/unit_tests/int8_gemm_parity <model.gguf> [--seqlen 8]
//       [--rows 32] [--max-shapes 12] [--tolerance 0.015] [--verbose]
//
// Exit code: 0 when every probed shape is inside tolerance, 1 otherwise.
// gemv_decode_parity.cpp
//
// Numeric gate for the GGUF decode mat-vec (`int8Gemv`) -- the kernel that runs
// once per token per projection, i.e. the whole of decode. Answers the same
// question int8_gemm_parity asks of prefill, per (type, rows, cols), on real
// weights: does the kernel agree with a float64 dot product over the
// dequantized row?
//
// Why a separate tool: prefill and decode are different kernels with different
// coverage. The int8 GEMM has a gate (int8_gemm_parity), and the EXL3 decode
// GEMV has two tools -- but nothing reached the GGUF decode path, so a change
// there could only be judged by whole-model tok/s, which is 40 shapes averaged
// together and moves by ~1% at a time.
//
// Tolerance is 2e-4 relative. Measured floors: the K-quants and Q8_0 land at
// 3e-8..1.3e-7, i.e. exact to accumulation order (a GEMV accumulates
// one int32 dot per 32-block (dp4a is exact in int32) and applies one fp16 scale
// per block, so against an fp64 reference the only error is the order of a
// handful of int32 dots, one fp16 scale each). The i-quants (IQ2_S, IQ3_XXS,
// IQ3_S) land at 2.5e-5..3.1e-5, which is not accumulation order: their lattice
// reconstruction evidently rounds an intermediate somewhere the host dequant
// does not. That is 100x below fp16 noise and irrelevant to output quality, but
// it is a real divergence between the engine's dequant and the kernel's, so it
// is recorded here rather than smoothed away -- and it is the reason the gate is
// 2e-4 and not 1e-5. A failure is still always structural (wrong row stride,
// wrong block scale, a dropped block: 1e-1 or worse).
//
// Routed-expert tensors are 3D ([cols, rowsPer, nExperts]) and reach their
// weight rows through a device-side expert index, so they are probed through
// int8GemvExpert rather than int8Gemv: nUsed = 8 (the model's used-expert
// count), unsorted real expert ids, and the same shared activation row the
// decode graph's gate/up passes (xStride = 0).  That path is what carries
// IQ2_XXS and IQ1_S on the Inkling checkpoint, and before this probe existed
// nothing in the tree ran those two dot products at all.
//
// The reference identities come from the same trick as the GEMM probe:
// activations are integers in [-127,127] scaled by 2^-7, so Q8_1 quantisation
// recovers them exactly and an fp64 dot over the *un*quantized values is the
// reference. Weights are dequantized by the engine's own GGMLDequantize.
//
// Usage:
//   cmake --build build-1.0 --target gemv_decode_parity -j 20
//   ./build-1.0/unit_tests/gemv_decode_parity <model.gguf> [--seqlen 1]
//       [--rows 32] [--max-shapes 12] [--tolerance 1e-5] [--verbose]
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
            case 16: return "IQ2_XXS";
            case 17: return "IQ2_XS";
            case 18: return "IQ3_XXS";
            case 19: return "IQ1_S";
            case 21: return "IQ3_S";
            case 22: return "IQ2_S";
            case 40: return "NVFP4";
            case 42: return "Q2_0";
            case 142: return "PQ2_0";
            case 143: return "PTQ1_0";
            default: return "?";
        }
    }

    // The gate. See the header comment: this band is the Q8_1 activation
    // convention's width, not the kernel's error.
    constexpr double kDefaultTolerance = 2e-4;

    struct Options {
        std::string model;
        // A list: int8Gemv declares 1..8 tokens and swaps kernel variants on the
        // batch size, so a probe run at one seqLen verifies one of eight kernels
        // (and, until it was swept, the multi-token path was quietly wrong -- see
        // the stride_col_y fix in GemmDispatch.cu).  `opt.seqLen` is the one
        // being probed right now.
        std::vector<uint32_t> seqLens = {1};
        uint32_t seqLen = 1;
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
        // 3D routed-expert tensor: `rows` is rowsPer*nExperts and the kernel is
        // reached through int8GemvExpert, which maps output row r to
        // (expert u = r/rowsPer, local row r%rowsPer). Zero for a plain 2D
        // projection.
        uint32_t rowsPer = 0;
        uint32_t nExperts = 0;
        bool expert = false;
    };

    struct ExpertProbe {
        bool ran = false;// false: int8GemvExpert refused the shape
        bool pass = false;
        double relRms = 0.0;
        double relMax = 0.0;
        size_t compared = 0;
    };

    // Expert-indexed probe for a [cols, rowsPer, nExperts] tensor.
    //
    // Same reference construction as the 2D path: activations are integers in
    // [-127,127] scaled by 2^-7, so Q8_1 recovers them exactly and an fp64 dot
    // over the *un*quantized values is the reference.  What differs is the
    // route: nUsed = 8 with unsorted expert ids (the model's used-expert count,
    // and ids that include 0 so a "use u instead of idx[u]" bug cannot pass),
    // and a shared activation row (xStride = 0), which is the gate/up shape.
    ExpertProbe probeExpertShape(const Shape &s, uint32_t rowsToCheck,
                                 double tolerance, bool fourRow) {
        ExpertProbe r;
        const uint32_t nUsed = std::min<uint32_t>(8u, s.nExperts);
        const int32_t ids[8] = {7, 3, 11, 0, 1, 2, 4, 5};
        const uint32_t blockSize = tinycoder::ggmlBlockSize(s.type);
        const uint32_t typeSize = tinycoder::ggmlTypeSize(s.type);
        if (blockSize == 0 || typeSize == 0 || nUsed == 0) return r;
        const uint64_t rowBytes =
                (static_cast<uint64_t>(s.cols) / blockSize) * typeSize;

        // Its own stream, so the 2D shapes' activations (and therefore the
        // recorded floors they are compared against) do not move.
        std::mt19937 rng(12345);
        const float actScale = std::ldexp(1.0f, -7);
        std::vector<float> xHost(s.cols, 0.0f);
        for (uint32_t k = 0; k < s.cols; k += 32u) {
            float *blk = xHost.data() + k;
            const uint32_t nb = std::min<uint32_t>(32u, s.cols - k);
            for (uint32_t i = 0; i < nb; ++i)
                blk[i] = static_cast<float>(static_cast<int>(rng() % 255u) - 127);
            blk[rng() % nb] = 127.0f;// pin the block max, so d is well defined
            for (uint32_t i = 0; i < nb; ++i) blk[i] *= actScale;
        }

        int32_t maxId = 0;
        for (uint32_t u = 0; u < nUsed; ++u) maxId = std::max(maxId, ids[u]);
        const uint64_t copyRows = static_cast<uint64_t>(maxId + 1) * s.rowsPer;
        const size_t outElems = static_cast<size_t>(s.rowsPer) * nUsed;

        void *dw = nullptr, *dx = nullptr, *dy = nullptr;
        int32_t *didx = nullptr;
        cudaMalloc(&dw, copyRows * rowBytes);
        cudaMalloc(&dx, xHost.size() * sizeof(float));
        cudaMalloc(&dy, outElems * sizeof(float));
        cudaMalloc(&didx, nUsed * sizeof(int32_t));
        cudaMemcpy(dw, s.weights, copyRows * rowBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, xHost.data(), xHost.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(didx, ids, nUsed * sizeof(int32_t), cudaMemcpyHostToDevice);
        std::vector<float> gpu(outElems, 0.0f);
        // Both entries take the shared activation row (xStride = 0); fourRow
        // selects the 4-rows-per-warp decode kernel instead of the one-row one.
        const bool launched =
                fourRow ? tinycoder::gpu::int8GemvExpert4(
                                  static_cast<int>(s.type), dw,
                                  static_cast<float *>(dx),
                                  static_cast<float *>(dy), s.rowsPer, s.cols,
                                  didx, nUsed, nullptr)
                        : tinycoder::gpu::int8GemvExpert(
                                  static_cast<int>(s.type), dw,
                                  static_cast<float *>(dx),
                                  static_cast<float *>(dy), s.rowsPer, s.cols,
                                  didx, nUsed, 0u, nullptr);
        if (launched) {
            cudaMemcpy(gpu.data(), dy, outElems * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
        cudaFree(dw);
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(didx);
        r.ran = launched;
        if (!launched) return r;

        const uint32_t stride = std::max<uint32_t>(1u, s.rowsPer / rowsToCheck);
        double sumSqErr = 0.0, sumSqRef = 0.0;
        double maxAbsErr = 0.0, maxAbsRef = 0.0;
        for (uint32_t u = 0; u < nUsed; ++u) {
            for (uint32_t rr = 0; rr < s.rowsPer; rr += stride) {
                const uint64_t row = static_cast<uint64_t>(ids[u]) * s.rowsPer + rr;
                const std::vector<float> deq = tinycoder::GGMLDequantize::dequantize(
                        s.type, s.weights + row * rowBytes, s.cols);
                if (deq.size() != s.cols) return r;
                double ref = 0.0;
                for (uint32_t k = 0; k < s.cols; ++k)
                    ref += static_cast<double>(deq[k]) * static_cast<double>(xHost[k]);
                const double got =
                        static_cast<double>(gpu[static_cast<size_t>(u) * s.rowsPer + rr]);
                const double err = got - ref;
                sumSqErr += err * err;
                sumSqRef += ref * ref;
                maxAbsErr = std::max(maxAbsErr, std::fabs(err));
                maxAbsRef = std::max(maxAbsRef, std::fabs(ref));
                ++r.compared;
            }
        }
        r.relRms = sumSqRef > 0.0 ? std::sqrt(sumSqErr / sumSqRef) : 0.0;
        r.relMax = maxAbsRef > 0.0 ? maxAbsErr / maxAbsRef : 0.0;
        r.pass = r.relRms <= tolerance;
        return r;
    }

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
                if (v >= 1 && v <= 8) opt.seqLens.push_back(static_cast<uint32_t>(v));
                p = (*end == ',') ? end + 1 : nullptr;
            }
            if (opt.seqLens.empty()) opt.seqLens.push_back(1);
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
                     "usage: gemv_decode_parity <model.gguf> [--seqlen N] [--rows N]\n"
                     "       [--max-shapes N] [--tolerance F] [--verbose]\n");
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
        std::fprintf(stderr, "[parity] int8 GEMV not available on this device/build\n");
        return 2;
    }

    // ---- collect one projection per (type, rows, cols) -----------------------
    // Same-shape duplicates add nothing: the kernel is selected by shape, so the
    // second occurrence runs identical code on identical data.
    std::vector<Shape> shapes;

    // ---- 3D routed-expert tensors, own pass and own cap ----------------------
    // [cols, rowsPer, nExperts].  Only the two types this probe was extended for
    // are taken: they have an expert-indexed int8 entry and, before it existed,
    // nothing in the tree reached their dot product at all.  The cap is
    // separate from --max-shapes because the expert tensors sit late in a
    // checkpoint's tensor list; a budget already spent on the dense projections
    // would silently probe none of them and the gate would look clean.
    constexpr uint32_t kMaxExpertShapes = 8;
    uint32_t nExpertShapes = 0;
    for (const auto &rec: loader.tensorRecords()) {
        if (nExpertShapes >= kMaxExpertShapes) break;
        if (rec.shape.size() != 3) continue;
        // Every type with an expert-indexed int8 entry, plus NVFP4 so its
        // absence is reported rather than silently skipped.
        if (rec.type != 16 && rec.type != 17 && rec.type != 19 &&
            rec.type != 22 && rec.type != 40 && rec.type != 42)
            continue;
        const uint32_t cols = rec.shape[0];
        const uint32_t rowsPer = rec.shape[1];
        const uint32_t nExperts = rec.shape[2];
        const uint32_t blockSize = tinycoder::ggmlBlockSize(rec.type);
        const uint32_t typeSize = tinycoder::ggmlTypeSize(rec.type);
        if (blockSize == 0 || typeSize == 0 || cols % blockSize != 0) continue;
        if (rowsPer == 0 || nExperts == 0) continue;
        bool seen = false;
        for (const auto &s: shapes) {
            if (s.expert && s.type == rec.type && s.rowsPer == rowsPer &&
                s.cols == cols) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        const uint8_t *w = loader.getTensor(rec.name);
        if (w == nullptr) continue;
        Shape s;
        s.type = rec.type;
        s.rows = rowsPer * nExperts;
        s.cols = cols;
        s.blocksPerRow = cols / blockSize;
        s.weights = w;
        s.name = rec.name;
        s.rowsPer = rowsPer;
        s.nExperts = nExperts;
        s.expert = true;
        shapes.push_back(s);
        ++nExpertShapes;
    }

    // ---- 2D dense projections ------------------------------------------------
    uint32_t nPlainShapes = 0;
    for (const auto &rec: loader.tensorRecords()) {
        if (nPlainShapes >= opt.maxShapes) break;
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
            if (!s.expert && s.type == rec.type && s.rows == rows &&
                s.cols == cols) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        const uint8_t *w = loader.getTensor(rec.name);
        if (w == nullptr) continue;
        shapes.push_back({rec.type, rows, cols, cols / blockSize, w, rec.name});
        ++nPlainShapes;
    }

    if (shapes.empty()) {
        std::fprintf(stderr, "[parity] no probed-type tensors found in %s\n",
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
      if (s.expert) {
        // Single-token path only (the expert GEMV has no batch dimension), so
        // the seqLen sweep does not apply.  Both kernels are probed: the
        // one-row entry the prefill path uses and the 4-rows-per-warp entry the
        // decode gate/up now prefers.
        const ExpertProbe pr =
                probeExpertShape(s, opt.rowsToCheck, opt.tolerance, false);
        if (!pr.ran) {
            ++skipped;
            std::printf("[parity] %-24s %-6s rowsPer=%-5u cols=%-5u  SKIP "
                        "(no one-row int8 expert entry for this type)\n",
                        s.name.c_str(), typeName(s.type), s.rowsPer, s.cols);
        } else {
            ++verified;
            if (!pr.pass) ++failures;
            std::printf("[parity] %-4s %-24s %-6s rowsPer=%-5u cols=%-5u experts=%-4u "
                        "expert-1row relRMS=%.3e relMax=%.3e\n",
                        pr.pass ? "PASS" : "FAIL", s.name.c_str(),
                        typeName(s.type), s.rowsPer, s.cols, s.nExperts, pr.relRms,
                        pr.relMax);
            if (opt.verbose || !pr.pass)
                std::printf("           expert-1row rows=%zu id=[7,3,11,0,1,2,4,5]\n",
                            pr.compared);
        }
        // The four-rows-per-warp entry serves only the gate/up types.
        if (s.type == 16 || s.type == 17 || s.type == 19 || s.type == 22) {
            const ExpertProbe p4 =
                    probeExpertShape(s, opt.rowsToCheck, opt.tolerance, true);
            if (!p4.ran) {
                ++skipped;
                std::printf("[parity] %-24s %-6s rowsPer=%-5u cols=%-5u  SKIP "
                            "(no 4-row int8 expert entry for this type)\n",
                            s.name.c_str(), typeName(s.type), s.rowsPer, s.cols);
            } else {
                ++verified;
                if (!p4.pass) ++failures;
                std::printf("[parity] %-4s %-24s %-6s rowsPer=%-5u cols=%-5u experts=%-4u "
                            "expert-4row relRMS=%.3e relMax=%.3e\n",
                            p4.pass ? "PASS" : "FAIL", s.name.c_str(),
                            typeName(s.type), s.rowsPer, s.cols, s.nExperts,
                            p4.relRms, p4.relMax);
                if (opt.verbose || !p4.pass)
                    std::printf("           expert-4row rows=%zu id=[7,3,11,0,1,2,4,5]\n",
                                p4.compared);
            }
        }
        continue;
      }
      for (const uint32_t seqLen : opt.seqLens) {
        opt.seqLen = seqLen;
        const size_t n = static_cast<size_t>(opt.seqLen) * s.rows;
        // Activations carry the shape's own row stride: int8Gemv indexes x as
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
        const bool ok = tinycoder::gpu::int8Gemv(static_cast<int>(s.type), s.weights, xDev,
                                                 outDev, opt.seqLen, s.rows, s.cols,
                                                 nullptr);
        if (!ok) {
            ++skipped;
            std::printf("[parity] %-24s %-6s rows=%-6u cols=%-5u  SKIP (int8Gemv refused the shape)\n",
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
        std::printf("[parity] %-4s %-24s %-6s rows=%-6u cols=%-5u m=%-3u relRMS=%.3e "
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
