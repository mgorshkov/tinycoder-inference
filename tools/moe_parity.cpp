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

// moe_parity.cpp
//
// Numeric gate for the MoE expert path, which had NO coverage: `int8_gemm_parity`
// and `gemv_decode_parity` only walk 2-D tensors, so the 3-D expert stacks
// (`ffn_{gate,up}_exps` [hidden, ff, experts], `ffn_down_exps` [ff, hidden,
// experts]) and their row-window GEMV (`QuantizedMatrix::matMulVecRows`, the
// call every routed expert uses) were never compared against a reference.
//
// This asks the same question those two tools ask, of the shapes they skip: for
// a real checkpoint, does the engine's expert-slice mat-vec agree with a float64
// dot over the dequantized rows, and by how much?  It also checks the router
// matrix, because a transposed or mis-strided router produces plausible-shaped
// logits and total garbage downstream, exactly like a broken expert GEMV.
//
// Usage:
//   cmake --build build-1.0 --target moe_parity -j 20
//   ./build-1.0/unit_tests/moe_parity <model.gguf> [--layers 0,1]
//       [--experts 0,1,137,255] [--tolerance 2e-4] [--verbose]
//
// Exit code: 0 when every probed slice is inside tolerance, 1 otherwise.
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
#include "Model.hpp"

namespace {

    struct Options {
        std::string model;
        std::vector<uint32_t> layers = {0};
        std::vector<uint32_t> experts = {0, 1, 137, 255};
        double tolerance = 2e-4;
        bool verbose = false;
    };

    struct TResult {
        double relRms = 0.0;
        double relMax = 0.0;
    };

    // The same relative metric the other two gates use, so the numbers are
    // comparable across tools.
    TResult compare(const std::vector<float> &got, const std::vector<float> &ref) {
        double num = 0.0, den = 0.0, maxAbs = 0.0, maxRef = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = static_cast<double>(got[i]) - ref[i];
            num += d * d;
            den += static_cast<double>(ref[i]) * ref[i];
            maxAbs = std::max(maxAbs, std::fabs(d));
            maxRef = std::max(maxRef, std::fabs(static_cast<double>(ref[i])));
        }
        TResult r;
        r.relRms = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
        r.relMax = maxRef > 0.0 ? maxAbs / maxRef : maxAbs;
        return r;
    }

    // Copy a slice of the checkpoint's mmap into an owning QuantizedMatrix (the
    // engine's containers own their bytes; aliasing them would test a different
    // layout than the model runs).
    bool adopt(const tinycoder::GGUFLoader &loader, const std::string &name,
               uint32_t rows, uint32_t cols, uint32_t type,
               tinycoder::QuantizedMatrix &out) {
        const uint8_t *src = loader.getTensor(name);
        if (src == nullptr) return false;
        const uint32_t bs = tinycoder::ggmlBlockSize(type);
        const uint32_t ts = tinycoder::ggmlTypeSize(type);
        if (bs == 0 || ts == 0 || cols % bs != 0) return false;
        const size_t bytes = static_cast<size_t>(rows) * (cols / bs) * ts;
        out.data.resize(bytes);
        std::memcpy(out.data.data(), src, bytes);
        out.rows = rows;
        out.cols = cols;
        out.type = type;
        return true;
    }

    // fp64 reference over rows [rowStart, rowStart + nRows) of a tensor.
    std::vector<float> referenceRows(const tinycoder::GGUFLoader &loader,
                                     const std::string &name, uint32_t type,
                                     uint32_t cols, uint32_t rowStart,
                                     uint32_t nRows, const std::vector<float> &x) {
        const uint8_t *base = loader.getTensor(name);
        const uint32_t bs = tinycoder::ggmlBlockSize(type);
        const uint32_t ts = tinycoder::ggmlTypeSize(type);
        const size_t bytesPerRow = static_cast<size_t>(cols / bs) * ts;
        std::vector<float> out(nRows, 0.0f);
        if (base == nullptr) return out;
        for (uint32_t r = 0; r < nRows; ++r) {
            const uint8_t *row = base + static_cast<size_t>(rowStart + r) * bytesPerRow;
            const std::vector<float> w = tinycoder::GGMLDequantize::dequantize(type, row, cols);
            double dot = 0.0;
            for (uint32_t i = 0; i < cols; ++i) {
                dot += static_cast<double>(w[i]) * static_cast<double>(x[i]);
            }
            out[r] = static_cast<float>(dot);
        }
        return out;
    }

}// namespace

int main(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (std::strcmp(a, "--layers") == 0 && i + 1 < argc) {
            opt.layers.clear();
            for (const char *p = argv[++i]; p != nullptr && *p != '\0';) {
                char *end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p) break;
                if (v >= 0) opt.layers.push_back(static_cast<uint32_t>(v));
                p = (*end == ',') ? end + 1 : nullptr;
            }
            if (opt.layers.empty()) opt.layers.push_back(0);
        } else if (std::strcmp(a, "--experts") == 0 && i + 1 < argc) {
            opt.experts.clear();
            for (const char *p = argv[++i]; p != nullptr && *p != '\0';) {
                char *end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p) break;
                if (v >= 0) opt.experts.push_back(static_cast<uint32_t>(v));
                p = (*end == ',') ? end + 1 : nullptr;
            }
        } else if (std::strcmp(a, "--tolerance") == 0 && i + 1 < argc) {
            opt.tolerance = std::atof(argv[++i]);
        } else if (std::strcmp(a, "--verbose") == 0) {
            opt.verbose = true;
        } else if (a[0] != '-') {
            opt.model = a;
        } else {
            std::fprintf(stderr, "[moe] unknown argument: %s\n", a);
            return 2;
        }
    }
    if (opt.model.empty()) {
        std::fprintf(stderr,
                     "usage: moe_parity <model.gguf> [--layers 0,1] "
                     "[--experts 0,1,137,255] [--tolerance 2e-4] [--verbose]\n");
        return 2;
    }

    tinycoder::GGUFLoader loader;
    if (!loader.load(opt.model)) {
        std::fprintf(stderr, "[moe] cannot read %s\n", opt.model.c_str());
        return 2;
    }

    // One activation vector, deterministic, in the same [-1,1]-ish range the
    // expert path actually sees (the GEMV is linear, so the scale is irrelevant
    // to a relative check as long as it is not degenerate).
    // Activations must be exactly representable in the kernel's int8 staging:
    // integers in [-127,127] scaled by 2^-7, so every 32-block's Q8_1 scale is
    // 2^-7 and the quantised values are the integers themselves.  With random
    // floats the reference disagrees by the quantisation band (2^-8 = 3.9e-3
    // relative RMS) and every slice "fails" for a reason that is not a defect --
    // measured at 4.0e-3 before this was fixed, exactly 2^-8.
    std::mt19937 rng(20261121);
    auto makeX = [&rng](uint32_t n) {
        std::vector<float> x(n);
        for (uint32_t k = 0; k < n; k += 32u) {
            const uint32_t nb = std::min<uint32_t>(32u, n - k);
            for (uint32_t i = 0; i < nb; ++i)
                x[k + i] = static_cast<float>(static_cast<int>(rng() % 255u) - 127);
            x[k + rng() % nb] = 127.0f;// pin the block max so d is well defined
            for (uint32_t i = 0; i < nb; ++i) x[k + i] *= std::ldexp(1.0f, -7);
        }
        return x;
    };

    std::printf("[moe] model %s\n", opt.model.c_str());
    int failures = 0, checked = 0;

    for (const uint32_t L: opt.layers) {
        const std::string p = "blk." + std::to_string(L) + ".";
        // ---- router: [experts][hidden] F32 or quantized ----
        for (const auto &rec: loader.tensorRecords()) {
            if (rec.name != p + "ffn_gate_inp.weight") continue;
            const uint32_t cols = rec.shape[0];
            const uint32_t rows = rec.shape[1];
            tinycoder::QuantizedMatrix m;
            if (!adopt(loader, rec.name, rows, cols, rec.type, m)) break;
            const std::vector<float> x = makeX(cols);
            std::vector<float> got(rows, 0.0f);
            m.matMulVec(x.data(), got.data());
            const auto ref = referenceRows(loader, rec.name, rec.type, cols, 0, rows, x);
            const TResult t = compare(got, ref);
            ++checked;
            if (t.relRms > opt.tolerance) ++failures;
            std::printf("[moe] %s %-8s L%u router      rows=%u cols=%u relRMS=%.3e relMax=%.3e\n",
                        t.relRms <= opt.tolerance ? "PASS" : "FAIL",
                        rec.name.c_str(), L, rows, cols, t.relRms, t.relMax);
            break;
        }

        // ---- routed experts: 3-D stacks, one contiguous expert per row block ----
        struct Stack {
            std::string suffix;
            uint32_t perExpertRows; // rows in one expert
            uint32_t cols;
        };
        const Stack stacks[] = {
                {"ffn_gate_exps.weight", 0, 0},// rows filled from the tensor
                {"ffn_up_exps.weight", 0, 0},
                {"ffn_down_exps.weight", 0, 0},
        };
        for (const auto &st: stacks) {
            const std::string name = p + st.suffix;
            const auto &recs = loader.tensorRecords();
            const tinycoder::TensorInfo *rec = nullptr;
            for (const auto &r: recs) {
                if (r.name == name) {
                    rec = &r;
                    break;
                }
            }
            if (rec == nullptr || rec->shape.size() != 3) continue;
            const uint32_t cols = rec->shape[0];
            const uint32_t perExpert = rec->shape[1];
            const uint32_t experts = rec->shape[2];
            if (experts == 0) continue;
            tinycoder::QuantizedMatrix m;
            if (!adopt(loader, name, perExpert * experts, cols, rec->type, m)) continue;
            const std::vector<float> x = makeX(cols);
            for (const uint32_t e: opt.experts) {
                if (e >= experts) continue;
                std::vector<float> got(perExpert, 0.0f);
                m.matMulVecRows(x.data(), e * perExpert, perExpert, got.data());
                const auto ref = referenceRows(loader, name, rec->type, cols,
                                               e * perExpert, perExpert, x);
                const TResult t = compare(got, ref);
                ++checked;
                if (t.relRms > opt.tolerance) ++failures;
                std::printf("[moe] %s %-8s L%u expert %3u rows=%u cols=%u relRMS=%.3e relMax=%.3e\n",
                            t.relRms <= opt.tolerance ? "PASS" : "FAIL",
                            st.suffix.c_str(), L, e, perExpert, cols, t.relRms, t.relMax);
            }
        }
    }

    std::printf("[moe] %d checked, %d outside tolerance\n", checked, failures);
    return failures == 0 ? 0 : 1;
}
