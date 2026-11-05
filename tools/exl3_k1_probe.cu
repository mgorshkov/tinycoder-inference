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

// exl3_k1_probe -- per-output-element correctness probe for the DECODE GEMV.
//
// Why this exists: the prefill path has bit-exact probes, the decode GEMV has
// none reaching it, and `exl3GemvK1` (the K=1 fast path) was numerically wrong
// while every shipped gate stayed green -- the 4-token logits probe never runs
// a decode GEMV at all, and `exl3_gpu_parity`'s K=1 cases are cb=0/cb=1, which
// the codebook gate routes away from the fast path.
//
// What it does, for ONE real 1-bit projection from the reference checkpoint:
//   1. decodes the trellis on the CPU with the reference codec
//      (EXL3Codec::decodeTileNative) into the ROTATED-basis weight What[k][n],
//   2. computes y_ref = What^T x in double (plus the 1/sqrt(128) Hadamards and
//      svh, exactly as the kernel's out transform does), for several x,
//   3. runs the shipped `gemvFused` on the same x and compares PER OUTPUT
//      ELEMENT, reporting max abs/rel error and WHICH output indices are wrong,
//      also inverted back to the pre-out-transform domain so a wrong index
//      localises to a (16-column tile, 128-column block) instead of smearing
//      over the Hadamard.
//
// x patterns: the most informative one is a "focused row": because the kernel
// computes H128(suh*x)/sqrt(128) before the dot, feeding
// x = suh * H128(e_j)/sqrt(128) makes the kernel's transformed activation a
// pure unit spike at row j, so the kernel's pre-transform output IS row j of
// What.  Sweeping j therefore says exactly "these columns are wrong for this
// weight row", which separates a lane->byte/row mapping error from a
// code->weight-position error.
//
// Two runs (one per kernel) are compared automatically: pass --tag k1 and
// --tag v3 (or any names) and a later run compares against the dumps the
// earlier one left behind, so "k1 vs v3" is separated from "k1 vs CPU".
//
// Build:  cmake --build build-1.0 --target exl3_k1_probe -j 20
// Run:    ./build-1.0/unit_tests/exl3_k1_probe <model_dir> [base] [--tag T]
//         (default base = model.language_model.layers.0.mlp.down_proj, K=1)

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "EXL3Codec.hpp"
#include "EXL3Kernels.hpp"
#include "SafetensorsReader.hpp"

using tinycoder::EXL3Codec;
using tinycoder::SafetensorsReader;

namespace {

#define CKP(expr)                                                                     \
    do {                                                                              \
        cudaError_t e_ = (expr);                                                      \
        if (e_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s:%d cuda error %s\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(e_));                                     \
            std::exit(2);                                                             \
        }                                                                             \
    } while (0)

    constexpr double kScale128 = 0.08838834764831845;// 1/sqrt(128)

    // Minimal parallel-for (nvcc's host pass does not take -fopenmp here, and a
    // pragma that is silently ignored would make this probe 30x slower).
    template<class F>
    void parFor(int64_t count, F f) {
        unsigned nt = std::thread::hardware_concurrency();
        if (nt == 0) nt = 1;
        if (nt > 32) nt = 32;
        if (count < 256) nt = 1;
        std::vector<std::thread> th;
        th.reserve(nt - 1);
        for (unsigned t = 1; t < nt; ++t) {
            th.emplace_back([&, t] {
                for (int64_t i = t; i < count; i += nt) f(i);
            });
        }
        for (int64_t i = 0; i < count; i += nt) f(i);
        for (std::thread &x: th) x.join();
    }

    // Unnormalized 128-point Hadamard, then * kScale128.  Same butterfly order
    // as exl3Fwht128 (not that exact arithmetic cares; kept for reviewability).
    void h128Scaled(const double *in, double *out) {
        double v[128];
        for (int i = 0; i < 128; ++i) v[i] = in[i];
        for (int len = 1; len < 128; len <<= 1) {
            for (int a = 0; a < 128; a += (len << 1)) {
                for (int j = 0; j < len; ++j) {
                    const double u = v[a + j], w = v[a + j + len];
                    v[a + j] = u + w;
                    v[a + j + len] = u - w;
                }
            }
        }
        for (int i = 0; i < 128; ++i) out[i] = v[i] * kScale128;
    }

    struct BadIdx {
        int idx;
        double d;
        double ref;
        double got;
    };

    struct Cmp {
        double maxAbs = 0.0;
        double maxRel = 0.0;
        double refScale = 0.0;
        std::vector<BadIdx> bad;
        int nBad = 0;
        int tiles16 = 0;
        int blocks128 = 0;
        std::string worstTileList;
    };

    Cmp compare(const std::vector<double> &ref, const std::vector<double> &got,
                double tol) {
        Cmp c;
        std::vector<uint8_t> tileHit((ref.size() + 15) / 16, 0);
        std::vector<uint8_t> blockHit((ref.size() + 127) / 128, 0);
        for (size_t i = 0; i < ref.size(); ++i) {
            const double d = std::fabs(ref[i] - got[i]);
            if (d > c.maxAbs) c.maxAbs = d;
            if (std::fabs(ref[i]) > c.refScale) c.refScale = std::fabs(ref[i]);
            const double rel = d / (std::fabs(ref[i]) + 1e-30);
            if (rel > c.maxRel) c.maxRel = rel;
            if (d > tol) {
                ++c.nBad;
                tileHit[i / 16] = 1;
                blockHit[i / 128] = 1;
                if (static_cast<int>(c.bad.size()) < 12) {
                    c.bad.push_back(BadIdx{static_cast<int>(i), d, ref[i], got[i]});
                }
            }
        }
        for (uint8_t t: tileHit) c.tiles16 += t;
        for (uint8_t b: blockHit) c.blocks128 += b;
        char buf[512];
        int off = 0;
        int shown = 0;
        for (size_t t = 0; t < tileHit.size() && shown < 24; ++t) {
            if (!tileHit[t]) continue;
            off += std::snprintf(buf + off, sizeof(buf) - static_cast<size_t>(off),
                                 "%s%zu", shown ? "," : "", t);
            ++shown;
            if (off >= static_cast<int>(sizeof(buf)) - 32) break;
        }
        if (shown == 24) {
            std::snprintf(buf + off, sizeof(buf) - static_cast<size_t>(off), ",...");
        }
        c.worstTileList = buf;
        return c;
    }

    void printCmp(const char *what, const Cmp &c, double tol, int nOut) {
        std::printf("    %-22s maxAbs=%.6e maxRel=%.3e  (|ref|max=%.3e tol=%.1e) "
                    "bad=%d/%d  nTiles16=%d nBlocks128=%d\n",
                    what, c.maxAbs, c.maxRel, c.refScale, tol, c.nBad, nOut,
                    c.tiles16, c.blocks128);
        if (c.nBad > 0) {
            std::printf("      bad16-tiles: [%s]\n", c.worstTileList.c_str());
            for (const BadIdx &b: c.bad) {
                std::printf("      out[%d] ref=%+.6f got=%+.6f  d=%.3e\n", b.idx,
                            b.ref, b.got, b.d);
            }
        }
    }

    bool readFloats(const SafetensorsReader &st, const std::string &name,
                    uint32_t count, std::vector<float> &dst) {
        const auto *t = st.find(name);
        if (t == nullptr) return false;
        const uint8_t *raw = st.data(*t);
        if (raw == nullptr || SafetensorsReader::numel(*t) < count) return false;
        dst.resize(count);
        if (t->dtype == SafetensorsReader::DType::F32) {
            std::memcpy(dst.data(), raw, count * sizeof(float));
            return true;
        }
        if (t->dtype == SafetensorsReader::DType::F16 ||
            t->dtype == SafetensorsReader::DType::BF16) {
            const uint16_t *h = reinterpret_cast<const uint16_t *>(raw);
            for (uint32_t i = 0; i < count; ++i) {
                if (t->dtype == SafetensorsReader::DType::BF16) {
                    const uint32_t bits = static_cast<uint32_t>(h[i]) << 16;
                    std::memcpy(&dst[i], &bits, sizeof(float));
                } else {
                    dst[i] = EXL3Codec::halfToFloat(h[i]);
                }
            }
            return true;
        }
        return false;
    }

    struct Pattern {
        std::string name;
        std::vector<float> x;
    };

    bool writeDump(const std::string &path, const std::vector<float> &y) {
        FILE *f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) return false;
        const uint64_t n = y.size();
        std::fwrite(&n, sizeof(n), 1, f);
        std::fwrite(y.data(), sizeof(float), y.size(), f);
        std::fclose(f);
        return true;
    }

    bool readDump(const std::string &path, std::vector<double> &y) {
        FILE *f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) return false;
        uint64_t n = 0;
        if (std::fread(&n, sizeof(n), 1, f) != 1) {
            std::fclose(f);
            return false;
        }
        std::vector<float> yf(n);
        const size_t got = std::fread(yf.data(), sizeof(float), n, f);
        std::fclose(f);
        if (got != n) return false;
        y.assign(yf.begin(), yf.end());
        return true;
    }

}// namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <model_dir> [tensor_base] [--tag NAME] [--dump DIR]\n"
                     "          [--split-order]\n"
                     "  default tensor_base = model.language_model.layers.0.mlp.down_proj\n"
                     "  --split-order: the gemv under test re-associates its k sum\n"
                     "    (grid.y k-split) on purpose, so the cross-kernel comparison\n"
                     "    against the other tag cannot be exact.  The comparison is\n"
                     "    then printed and held to the same tolerance as the double\n"
                     "    reference instead of 0.  The double-reference check is\n"
                     "    NEVER relaxed.\n",
                     argv[0]);
        return 1;
    }
    const std::string modelDir = argv[1];
    std::string base = "model.language_model.layers.0.mlp.down_proj";
    std::string tag = "run";
    std::string dumpDir = "/tmp/exl3_k1_probe";
    bool splitOrder = false;
    int argi = 2;
    if (argi < argc && argv[argi][0] != '-') base = argv[argi++];
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (a == "--tag" && argi + 1 < argc) {
            tag = argv[++argi];
        } else if (a == "--dump" && argi + 1 < argc) {
            dumpDir = argv[++argi];
        } else if (a == "--split-order") {
            splitOrder = true;
        } else {
            std::fprintf(stderr, "unknown argument '%s'\n", a.c_str());
            return 1;
        }
    }

    // ---- open the checkpoint ----
    std::vector<std::string> shards;
    {
        // Directory listing without <filesystem>: the shard names in this
        // checkpoint are the standard 5-digit form.
        for (int i = 1; i <= 99; ++i) {
            char p[512];
            std::snprintf(p, sizeof(p), "%s/model-%05d-of-00002.safetensors",
                          modelDir.c_str(), i);
            FILE *f = std::fopen(p, "rb");
            if (f) {
                std::fclose(f);
                shards.emplace_back(p);
            }
        }
        if (shards.empty()) {
            char p[512];
            std::snprintf(p, sizeof(p), "%s/model.safetensors", modelDir.c_str());
            FILE *f = std::fopen(p, "rb");
            if (f) {
                std::fclose(f);
                shards.emplace_back(p);
            }
        }
    }
    if (shards.empty()) {
        std::fprintf(stderr, "no *.safetensors found under %s\n", modelDir.c_str());
        return 1;
    }
    SafetensorsReader st;
    if (!st.open(shards)) {
        std::fprintf(stderr, "safetensors open failed\n");
        return 1;
    }
    const auto *tr = st.find(base + ".trellis");
    if (tr == nullptr || tr->shape.size() != 3) {
        std::fprintf(stderr, "tensor '%s.trellis' missing or malformed\n", base.c_str());
        return 1;
    }
    const uint32_t tilesK = static_cast<uint32_t>(tr->shape[0]);
    const uint32_t tilesN = static_cast<uint32_t>(tr->shape[1]);
    const uint32_t tileU16 = static_cast<uint32_t>(tr->shape[2]);
    const uint32_t k = tilesK * 16u, n = tilesN * 16u;
    const uint32_t K = tileU16 / 16u;
    if (tileU16 % 16u != 0) {
        std::fprintf(stderr, "half-integer bitrate unsupported here\n");
        return 1;
    }
    int codebook = EXL3Codec::CB_3INST;
    if (st.find(base + ".mcg")) codebook = EXL3Codec::CB_MCG;
    else if (st.find(base + ".mul1")) codebook = EXL3Codec::CB_MUL1;

    std::printf("tensor   %s\n", base.c_str());
    std::printf("geometry k=%u n=%u K=%u codebook=%d tileU16=%u  tag=%s\n", k, n, K,
                codebook, tileU16, tag.c_str());

    std::vector<float> suh, svh;
    const bool haveSu = readFloats(st, base + ".suh", k, suh);
    const bool haveSv = readFloats(st, base + ".svh", n, svh);
    std::printf("signs    suh=%d svh=%d\n", haveSu ? 1 : 0, haveSv ? 1 : 0);
    // EXL3's suh/svh are NOT ±1: they carry a per-row/column scale (the ~2^-7
    // of the raw basis lives there).  Every inversion below therefore divides
    // by them instead of multiplying.
    std::vector<double> isuh(k, 1.0), isvh(n, 1.0);
    if (haveSu) {
        double mx = 0.0, mn = 1e30, rms = 0.0;
        for (uint32_t i = 0; i < k; ++i) {
            isuh[i] = 1.0 / static_cast<double>(suh[i]);
            const double a = std::fabs(static_cast<double>(suh[i]));
            if (a > mx) mx = a;
            if (a < mn) mn = a;
            rms += a * a;
        }
        std::printf("suh      |v| in [%.6g, %.6g] rms=%.6g (not +-1 => a scale vector)\n",
                    mn, mx, std::sqrt(rms / k));
    }
    if (haveSv) {
        double mx = 0.0, mn = 1e30, rms = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            isvh[i] = 1.0 / static_cast<double>(svh[i]);
            const double a = std::fabs(static_cast<double>(svh[i]));
            if (a > mx) mx = a;
            if (a < mn) mn = a;
            rms += a * a;
        }
        std::printf("svh      |v| in [%.6g, %.6g] rms=%.6g\n", mn, mx,
                    std::sqrt(rms / n));
    }

    // ---- CPU reference: rotated-basis weight from the reference codec ----
    const uint16_t *packed = reinterpret_cast<const uint16_t *>(st.data(*tr));
    if (packed == nullptr) {
        std::fprintf(stderr, "trellis payload missing\n");
        return 1;
    }
    std::vector<float> what(static_cast<size_t>(k) * n);
    {
        const auto t0 = std::chrono::steady_clock::now();
        parFor(static_cast<int64_t>(tilesK) * tilesN, [&](int64_t t) {
            const uint32_t ki = static_cast<uint32_t>(t / tilesN);
            const uint32_t ni = static_cast<uint32_t>(t % tilesN);
            float tile[256];
            EXL3Codec::decodeTileNative(packed + t * tileU16, K, codebook, tile);
            for (uint32_t r = 0; r < 16u; ++r) {
                float *dst = what.data() + (static_cast<size_t>(ki) * 16u + r) * n +
                             ni * 16u;
                for (uint32_t c = 0; c < 16u; ++c) dst[c] = tile[r * 16u + c];
            }
        });
        const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
        std::printf("cpu      decoded %ux%u in %.0f ms\n", k, n, ms);
    }

    // ---- device: register the same linear through the shipping path ----
    const size_t trellisBytes = static_cast<size_t>(tilesK) * tilesN * tileU16 * 2u;
    void *dTrellis = nullptr;
    float *dSuh = nullptr, *dSvh = nullptr, *dX = nullptr, *dY = nullptr;
    CKP(cudaMalloc(&dTrellis, trellisBytes));
    CKP(cudaMemcpy(dTrellis, packed, trellisBytes, cudaMemcpyHostToDevice));
    if (haveSu) {
        CKP(cudaMalloc(&dSuh, sizeof(float) * k));
        CKP(cudaMemcpy(dSuh, suh.data(), sizeof(float) * k, cudaMemcpyHostToDevice));
    }
    if (haveSv) {
        CKP(cudaMalloc(&dSvh, sizeof(float) * n));
        CKP(cudaMemcpy(dSvh, svh.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
    }
    CKP(cudaMalloc(&dX, sizeof(float) * k));
    CKP(cudaMalloc(&dY, sizeof(float) * n));
    tinycoder::exl3gpu::Exl3FusedLinear lin{};
    std::string err;
    if (!tinycoder::exl3gpu::uploadFusedLinearDevice(dTrellis, dSuh, dSvh, k, n,
                                                     K * 10u, codebook, lin, err)) {
        std::fprintf(stderr, "uploadFusedLinearDevice: %s\n", err.c_str());
        return 2;
    }

    // ---- x patterns ----
    std::vector<Pattern> pats;
    auto addPat = [&](const std::string &nm, std::vector<float> &&v) {
        pats.push_back(Pattern{nm, std::move(v)});
    };
    {
        std::vector<float> ones(k, 1.0f);
        addPat("ones", std::move(ones));
    }
    {
        // Two PRNG patterns (fixed seed, reproducible).
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (int rep = 0; rep < 2; ++rep) {
            std::vector<float> v(k);
            for (uint32_t i = 0; i < k; ++i) {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                v[i] = static_cast<float>(static_cast<double>(s >> 11) /
                                              static_cast<double>(1ull << 53) *
                                              2.0 -
                                          1.0);
            }
            addPat(rep == 0 ? "rand_a" : "rand_b", std::move(v));
        }
    }
    {
        // Focused rows: x = suh * H128(e_j) / sqrt(128) makes the kernel's
        // internal transformed activation e_j, so the pre-transform output is
        // exactly weight row j.  j = 0..15 covers one full k-tile's 16 rows.
        const uint32_t rows[] = {0u, 1u, 2u, 3u, 4u, 7u, 8u, 9u,
                                 12u, 15u, 16u, 17u, 127u, 128u, 129u};
        for (uint32_t j: rows) {
            if (j >= k) continue;
            std::vector<double> e(128, 0.0), h(128, 0.0), xd(k, 0.0);
            const uint32_t chunk = (j / 128u) * 128u;
            e[j % 128u] = 1.0;
            h128Scaled(e.data(), h.data());
            for (uint32_t i = 0; i < 128u; ++i) {
                if (chunk + i < k) xd[chunk + i] = h[i];
            }
            std::vector<float> x(k);
            for (uint32_t i = 0; i < k; ++i) {
                x[i] = static_cast<float>(xd[i] * isuh[i]);
            }
            char nm[64];
            std::snprintf(nm, sizeof(nm), "row%u", j);
            addPat(nm, std::move(x));
        }
    }
    {
        // One-hot rows (through the transform, so they mix a 128-row chunk):
        // these are what a "which k rows are wired wrong" sweep looks like from
        // the outside.
        for (uint32_t j: {0u, 1u, 15u, 16u, 127u, 128u, k - 1u}) {
            if (j >= k) continue;
            std::vector<float> x(k, 0.0f);
            x[j] = 1.0f;
            char nm[64];
            std::snprintf(nm, sizeof(nm), "spike%u", j);
            addPat(nm, std::move(x));
        }
    }

    // ---- the reference is What^T xh with xh = H128(suh*x)/sqrt(128) ----
    // A "focused row" pattern must land exactly on one row of What.
    int failures = 0;
    const std::string otherTag = (tag == "k1") ? "v3" : "k1";
    for (const Pattern &p: pats) {
        std::vector<double> xh(k, 0.0), zref(n, 0.0), yref(n, 0.0);
        for (uint32_t kb = 0; kb + 128u <= k; kb += 128u) {
            std::vector<double> in(128), out(128);
            for (uint32_t i = 0; i < 128u; ++i) {
                in[i] = static_cast<double>(p.x[kb + i]) *
                        (haveSu ? static_cast<double>(suh[kb + i]) : 1.0);
            }
            h128Scaled(in.data(), out.data());
            for (uint32_t i = 0; i < 128u; ++i) xh[kb + i] = out[i];
        }
        parFor(static_cast<int64_t>(n), [&](int64_t c) {
            double acc = 0.0;
            const float *col = what.data() + c;
            for (uint32_t i = 0; i < k; ++i) {
                acc += static_cast<double>(col[static_cast<size_t>(i) * n]) * xh[i];
            }
            zref[c] = acc;
        });
        for (uint32_t nb = 0; nb + 128u <= n; nb += 128u) {
            std::vector<double> in(128), out(128);
            for (uint32_t i = 0; i < 128u; ++i) in[i] = zref[nb + i];
            h128Scaled(in.data(), out.data());
            for (uint32_t i = 0; i < 128u; ++i) {
                yref[nb + i] = out[i] * (haveSv ? static_cast<double>(svh[nb + i]) : 1.0);
            }
        }

        // GPU
        CKP(cudaMemcpy(dX, p.x.data(), sizeof(float) * k, cudaMemcpyHostToDevice));
        if (!tinycoder::exl3gpu::gemvFused(dTrellis, dX, dY, 0, err)) {
            std::fprintf(stderr, "gemvFused(%s): %s\n", p.name.c_str(), err.c_str());
            return 2;
        }
        CKP(cudaDeviceSynchronize());
        std::vector<float> yg(n);
        CKP(cudaMemcpy(yg.data(), dY, sizeof(float) * n, cudaMemcpyDeviceToHost));

        // Undo the out transform: z = H128(y * svh) / sqrt(128).
        std::vector<double> ygSv(n), zg(n);
        for (uint32_t i = 0; i < n; ++i) {
            ygSv[i] = static_cast<double>(yg[i]) * isvh[i];
        }
        for (uint32_t nb = 0; nb + 128u <= n; nb += 128u) {
            std::vector<double> in(128), out(128);
            for (uint32_t i = 0; i < 128u; ++i) in[i] = ygSv[nb + i];
            h128Scaled(in.data(), out.data());
            for (uint32_t i = 0; i < 128u; ++i) zg[nb + i] = out[i];
        }
        std::vector<double> ygd(yg.begin(), yg.end());

        std::printf("  x=%-9s\n", p.name.c_str());
        if (std::getenv("EXL3_K1_PROBE_DEBUG")) {
            double wr = 0.0;
            for (size_t i = 0; i < what.size(); ++i) wr += double(what[i]) * what[i];
            wr = std::sqrt(wr / double(what.size()));
            double xr = 0.0;
            for (uint32_t i = 0; i < k; ++i) xr += xh[i] * xh[i];
            xr = std::sqrt(xr / double(k));
            double nz = 0.0, ng = 0.0, ny = 0.0, nyg = 0.0;
            for (uint32_t i = 0; i < n; ++i) {
                nz += zref[i] * zref[i];
                ng += zg[i] * zg[i];
                ny += yref[i] * yref[i];
                nyg += double(yg[i]) * yg[i];
            }
            std::vector<double> ta(128), tb(128);
            for (int i = 0; i < 128; ++i) ta[i] = zref[i];
            h128Scaled(ta.data(), tb.data());
            h128Scaled(tb.data(), ta.data());
            double rt = 0.0;
            for (int i = 0; i < 128; ++i) rt = std::fmax(rt, std::fabs(ta[i] - zref[i]));
            {
                double zmaxv = 0.0, gmaxv = 0.0, smaxv = 0.0;
                int zi = -1, gi = -1, si = -1;
                for (uint32_t i = 0; i < n; ++i) {
                    if (std::fabs(zg[i]) > gmaxv) { gmaxv = std::fabs(zg[i]); gi = int(i); }
                    if (std::fabs(yg[i]) > smaxv) { smaxv = std::fabs(yg[i]); si = int(i); }
                    if (std::fabs(zref[i]) > zmaxv) { zmaxv = std::fabs(zref[i]); zi = int(i); }
                }
                double direct = 0.0;
                for (int i = 0; i < 128; ++i) {
                    const double sgn = (i == (i & i - 1)) ? 1.0 : 1.0;
                    (void) sgn;
                    direct += ygSv[i];
                }
                direct *= kScale128;
                std::printf("      dbg max|zref|=%.4f@%d max|zg|=%.4f@%d "
                            "max|yg|=%.4f@%d  sum128*kScale=%.4f zg[0]=%.6f "
                            "rms(svh)=%.4f\n",
                            zmaxv, zi, gmaxv, gi, smaxv, si, direct, zg[0],
                            [&] {
                                double a = 0.0;
                                for (uint32_t i = 0; i < n; ++i) a += double(svh[i]) * svh[i];
                                return std::sqrt(a / double(n));
                            }());
            }
            std::printf("      dbg |zref|=%.4f |zg|=%.4f |yref|=%.4f |yg|=%.4f "
                        "involution_err=%.3e\n", std::sqrt(nz), std::sqrt(ng),
                        std::sqrt(ny), std::sqrt(nyg), rt);
            std::printf("      dbg rms(what)=%.4f rms(xh)=%.4f z[0..3]=%.6f %.6f %.6f "
                        "%.6f | yref[0]=%.6f yg[0]=%.6f | zg[0..1]=%.6f %.6f\n",
                        wr, xr, zref[0], zref[1], zref[2], zref[3], yref[0], yg[0],
                        zg[0], zg[1]);
        }
        const double tolY = 4e-5 * (1.0 + std::fabs(yref[0]));
        const Cmp cy = compare(yref, ygd, tolY);
        printCmp("gpu_vs_cpu [y]", cy, tolY, static_cast<int>(n));
        double zmax = 0.0;
        for (double v: zref) zmax = std::fmax(zmax, std::fabs(v));
        const double tolZ = 4e-4 * (1.0 + zmax);
        const Cmp cz = compare(zref, zg, tolZ);
        printCmp("gpu_vs_cpu [pre-xform]", cz, tolZ, static_cast<int>(n));
        if (cy.maxAbs > tolY || cz.maxAbs > tolZ) ++failures;

        // Dump for the cross-kernel comparison.
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/y_%s_%s.bin", dumpDir.c_str(),
                      p.name.c_str(), tag.c_str());
        writeDump(path, yg);
        std::snprintf(path, sizeof(path), "%s/y_%s_%s.bin", dumpDir.c_str(),
                      p.name.c_str(), otherTag.c_str());
        std::vector<double> yOther;
        if (readDump(path, yOther) && yOther.size() == n) {
            // Two kernels must agree exactly per weight -- unless the kernel
            // under test was asked to re-associate its k sum over grid.y, in
            // which case only the double reference can be the arbiter.
            const double tolK = splitOrder ? tolY : 0.0;
            const Cmp ck = compare(yOther, ygd, tolK);
            printCmp((std::string("gpu_vs_") + otherTag + " [y]").c_str(), ck, tolK,
                     static_cast<int>(n));
            if (splitOrder) {
                std::printf("    (re-associated k order: cross-kernel diff is "
                            "accumulation order, not a defect)\n");
                if (ck.maxAbs > tolY) ++failures;
            }
        }
    }

    std::printf("%s: %d pattern(s) outside tolerance\n", failures ? "FAIL" : "PASS",
                failures);
    tinycoder::exl3gpu::releaseFusedLinear(lin);
    cudaFree(dTrellis);
    if (dSuh) cudaFree(dSuh);
    if (dSvh) cudaFree(dSvh);
    cudaFree(dX);
    cudaFree(dY);
    return failures ? 1 : 0;
}
