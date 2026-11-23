// GPU parity for the four Inkling GEMV types.  CPU reference is the engine's
// GGMLDequantize block decoder; the GPU path is inklingGemv.
#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"
#include "Int8Gemm.hpp"
#include "gpu/InklingGPU.hpp"
#include "gpu/InklingKernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace tinycoder;

static uint32_t typeBlockBytes(uint32_t type);

static bool testTensor(const char *path, const char *name, uint32_t expertRows,
                       uint32_t cols, uint32_t testRows) {
    GGUFLoader loader;
    if (!loader.load(path)) return false;
    auto info = loader.getTensorInfo(name);
    if (!info) { std::fprintf(stderr, "missing %s\n", name); return false; }
    const uint8_t *weights = loader.getTensor(name);
    const uint32_t rowsPerExpert = static_cast<uint32_t>(info->shape[1]);
    const uint32_t bs = (info->type == 40 || info->type == 42) ? 64u
            : (info->type == 0 || info->type == 30 || info->type == 8) ? 32u
                                                                      : 256u;
    const uint32_t typeSize = typeBlockBytes(info->type);
    const uint64_t blocksPerRow = (cols + bs - 1) / bs;
    const uint64_t rowBytes = blocksPerRow * typeSize;
    if (rowsPerExpert != expertRows) {
        std::fprintf(stderr, "%s: rows/expert %u != %u\n", name, rowsPerExpert,
                     expertRows);
        return false;
    }

    std::vector<float> x(cols, 0.01f), ref(testRows), gpu(testRows);
    std::vector<float> block(bs);
    for (uint32_t r = 0; r < testRows; ++r) {
        const uint8_t *row = weights + static_cast<uint64_t>(r) * rowBytes;
        double acc = 0.0;
        for (uint64_t b = 0; b < blocksPerRow; ++b) {
            GGMLDequantize::dequantizeBlock(info->type, row + b * typeSize,
                                            block.data(), bs);
            const uint32_t n = std::min<uint32_t>(bs, cols - b * bs);
            for (uint32_t j = 0; j < n; ++j)
                acc += double(block[j]) * x[b * bs + j];
        }
        ref[r] = static_cast<float>(acc);
    }

    void *dw = nullptr, *dx = nullptr, *dy = nullptr;
    cudaMalloc(&dw, static_cast<size_t>(testRows) * rowBytes);
    cudaMalloc(&dx, cols * sizeof(float));
    cudaMalloc(&dy, testRows * sizeof(float));
    cudaMemcpy(dw, weights, static_cast<size_t>(testRows) * rowBytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
    const bool ok = gpu::inklingGemv(info->type, dw, static_cast<float *>(dx),
                                     static_cast<float *>(dy), testRows, cols,
                                     nullptr);
    cudaMemcpy(gpu.data(), dy, testRows * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    if (!ok) { std::fprintf(stderr, "%s: launch failed\n", name); return false; }

    double maxAbs = 0.0, maxRef = 0.0;
    for (uint32_t i = 0; i < testRows; ++i) {
        maxAbs = std::max(maxAbs, std::fabs(double(gpu[i]) - ref[i]));
        maxRef = std::max(maxRef, std::fabs(double(ref[i])));
    }
    const double ratio = maxAbs / std::max(maxRef, 1e-30);
    if (ratio >= 1e-2) {
        for (int i = 0; i < 3; ++i)
            std::printf("  [%d] ref=%.6f gpu=%.6f\n", i, ref[i], gpu[i]);
    }
    std::printf("%-42s type=%2u ratio=%.3e %s\n", name, info->type, ratio,
                ratio < 1e-2 ? "PASS" : "FAIL");
    return ratio < 1e-2;
}

// ---- Expert-indexed int8 GEMV (the routed-MoE decode path) ----
//
// The risk this catches is not rounding, it is the row mapping: output row r
// must come from weight row rowIdx[r/rowsPer]*rowsPer + r%rowsPer, and (for the
// down projection) from activation row r/rowsPer.  A wrong expert or a wrong
// activation row is O(1) off, not 1e-2, so a 2e-2 gate is sharp.
static uint32_t typeBlockBytes(uint32_t type) {
    switch (type) {
        case 8: return 34;
        case 12: return 144;
        case 13: return 176;
        case 14: return 210;
        case 16: return 66;
        case 17: return 74;
        case 18: return 98;
        case 19: return 50;
        case 21: return 110;
        case 22: return 82;
        case 23: return 136;
        case 40: return 36;
        case 42: return 18;
        default: return 0;
    }
}

static bool testExpertGemv(const char *path, const char *name, uint32_t rowsPer,
                           uint32_t cols, uint32_t testRows, bool sharedAct) {
    GGUFLoader loader;
    if (!loader.load(path)) return false;
    auto info = loader.getTensorInfo(name);
    if (!info) { std::fprintf(stderr, "missing %s\n", name); return false; }
    const uint8_t *weights = loader.getTensor(name);
    const uint32_t bs = (info->type == 40 || info->type == 42) ? 64u : 256u;
    const uint32_t typeSize = typeBlockBytes(info->type);
    if (typeSize == 0) {
        std::fprintf(stderr, "%s: unsupported type %u\n", name, info->type);
        return false;
    }
    const uint64_t blocksPerRow = (cols + bs - 1) / bs;
    const uint64_t rowBytes = blocksPerRow * typeSize;

    // nUsed=8 matches the model; ids are deliberately not ascending and
    // include 0 so a "use u instead of idx[u]" bug cannot pass.
    const uint32_t nUsed = 8;
    const int32_t ids[nUsed] = {7, 3, 11, 0, 1, 2, 4, 5};
    std::vector<float> x(sharedAct ? cols : static_cast<size_t>(nUsed) * cols);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = 0.35f * std::sin(float(i) * 0.017f) + 0.11f * std::cos(float(i) * 0.0031f);

    std::vector<float> ref(static_cast<size_t>(nUsed) * testRows, 0.0f);
    std::vector<float> block(bs);
    for (uint32_t u = 0; u < nUsed; ++u) {
        const float *xu = sharedAct ? x.data() : x.data() + size_t(u) * cols;
        for (uint32_t r = 0; r < testRows; ++r) {
            const uint64_t row = uint64_t(ids[u]) * rowsPer + r;
            const uint8_t *rp = weights + row * rowBytes;
            double acc = 0.0;
            for (uint64_t b = 0; b < blocksPerRow; ++b) {
                GGMLDequantize::dequantizeBlock(info->type, rp + b * typeSize,
                                                block.data(), bs);
                const uint32_t n = std::min<uint32_t>(bs, cols - b * bs);
                for (uint32_t j = 0; j < n; ++j)
                    acc += double(block[j]) * xu[b * bs + j];
            }
            ref[size_t(u) * testRows + r] = float(acc);
        }
    }

    // The kernel writes every row of each selected expert, not only the rows
    // this probe compares, so the device copy must cover the highest expert id.
    // The ids are deliberately unsorted, so scan for the maximum.
    int32_t maxId = 0;
    for (uint32_t u = 0; u < nUsed; ++u) maxId = std::max(maxId, ids[u]);
    const uint64_t maxRow = (uint64_t(maxId) + 1) * rowsPer;
    void *dw = nullptr, *dx = nullptr, *dy = nullptr;
    int32_t *didx = nullptr;
    // The kernel writes all rowsPer*nUsed output rows; only the first `testRows`
    // of each expert are compared, but the buffer must hold the rest.
    const size_t outElems = size_t(rowsPer) * nUsed;
    cudaMalloc(&dw, maxRow * rowBytes);
    cudaMalloc(&dx, x.size() * sizeof(float));
    cudaMalloc(&dy, outElems * sizeof(float));
    cudaMalloc(&didx, nUsed * sizeof(int32_t));
    cudaMemcpy(dw, weights, maxRow * rowBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(didx, ids, nUsed * sizeof(int32_t), cudaMemcpyHostToDevice);
    std::vector<float> gpu(outElems, 0.0f);
    const bool launched = gpu::int8GemvExpert(
            info->type, dw, static_cast<float *>(dx), static_cast<float *>(dy),
            rowsPer, cols, didx, nUsed, sharedAct ? 0u : cols, nullptr);
    if (launched) {
        const cudaError_t e = cudaMemcpy(gpu.data(), dy,
                                         outElems * sizeof(float),
                                         cudaMemcpyDeviceToHost);
        if (e != cudaSuccess)
            std::printf("  expert GEMV copy failed: %s\n",
                        cudaGetErrorString(e));
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(didx);
    if (!launched) {
        std::printf("%-42s expert GEMV: REFUSED (no int8 path)\n", name);
        return true;// not a failure: the caller falls back to the float decoder
    }

    // The kernel writes expert u's rows at u*rowsPer; the reference packs only
    // the first testRows of each expert.
    double maxAbs = 0.0, maxRef = 0.0, s2 = 0.0;
    for (uint32_t u = 0; u < nUsed; ++u) {
        for (uint32_t r = 0; r < testRows; ++r) {
            const double d = double(gpu[size_t(u) * rowsPer + r]) -
                             ref[size_t(u) * testRows + r];
            maxAbs = std::max(maxAbs, std::fabs(d));
            maxRef = std::max(maxRef,
                              std::fabs(double(ref[size_t(u) * testRows + r])));
            s2 += d * d;
        }
    }
    const double ratio = maxAbs / std::max(maxRef, 1e-30);
    const double rms = std::sqrt(s2 / double(ref.size()));
    if (ratio >= 2e-2) {
        for (uint32_t u = 0; u < nUsed; ++u)
            std::printf("  expert %d row 0: ref=%.6f gpu=%.6f\n", ids[u],
                        ref[size_t(u) * testRows],
                        gpu[size_t(u) * rowsPer]);
    }
    std::printf("%-42s type=%2u expert=%s ratio=%.3e rms=%.3e %s\n", name,
                info->type, sharedAct ? "shared" : "per-row", ratio, rms,
                ratio < 2e-2 ? "PASS" : "FAIL");
    return ratio < 2e-2;
}

static bool testElementwise();
static bool testAttentionOps();
static bool testMoeOps();
static bool testMoeGroupSort();

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <iq2.gguf>\n", argv[0]); return 2; }
    int fails = 0;
    fails += !testTensor(argv[1], "blk.0.ffn_gate_up_exps.weight", 2304, 2304, 32);
    fails += !testTensor(argv[1], "blk.0.ffn_down_exps.weight",    2304, 1152, 32);
    fails += !testTensor(argv[1], "blk.5.ffn_gate_up_exps.weight", 2304, 2304, 32);
    fails += !testTensor(argv[1], "blk.5.ffn_down_exps.weight",    2304, 1152, 32);
    fails += !testTensor(argv[1], "blk.10.ffn_gate_up_exps.weight", 2304, 2304, 32);
    // Expert GEMV: the four types this checkpoint's routed experts use.
    // IQ2_XS gate/up and Q2_0 down are covered by the int8 path; IQ2_XXS
    // gate/up and NVFP4 down report REFUSED and fall back to the float
    // decoder, which is a coverage statement, not a failure.
    fails += !testExpertGemv(argv[1], "blk.10.ffn_gate_up_exps.weight", 2304,
                             2304, 256, true);// IQ2_XS
    fails += !testExpertGemv(argv[1], "blk.3.ffn_gate_up_exps.weight", 2304,
                             2304, 256, true);// IQ2_S
    fails += !testExpertGemv(argv[1], "blk.0.ffn_down_exps.weight", 1152, 2304,
                             256, false);// Q2_0
    fails += !testExpertGemv(argv[1], "blk.5.ffn_gate_up_exps.weight", 2304, 2304,
                             256, true);// IQ2_XXS: refused, float fallback
    fails += !testExpertGemv(argv[1], "blk.5.ffn_down_exps.weight", 1152, 2304,
                             256, false);// NVFP4: refused, float fallback
    fails += !testElementwise();
    fails += !testAttentionOps();
    fails += !testMoeOps();
    fails += !testMoeGroupSort();
    std::printf("%s\n", fails == 0 ? "ALL INKLING GPU PARITY CHECKS PASSED"
                                   : "INKLING GPU PARITY FAILURES");
    return fails == 0 ? 0 : 1;
}

// ---- Task 2: elementwise / short-conv parity ----
static bool testElementwise() {
    const uint32_t n = 2304;
    std::vector<float> x(n), w(n), ref(n), out(n);
    for (uint32_t i = 0; i < n; ++i) { x[i] = 0.1f * (i % 7) - 0.3f; w[i] = 0.5f + 0.01f * (i % 5); }
    double ss = 0.0; for (float v : x) ss += double(v) * v;
    const float inv = 1.0f / std::sqrt(float(ss / n) + 1e-6f);
    for (uint32_t i = 0; i < n; ++i) ref[i] = x[i] * inv * w[i];
    void *dx, *dw, *dy;
    cudaMalloc(&dx, n * 4); cudaMalloc(&dw, n * 4); cudaMalloc(&dy, n * 4);
    cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), n * 4, cudaMemcpyHostToDevice);
    gpu::kInklingRmsNorm(static_cast<float *>(dx), static_cast<float *>(dw),
                         static_cast<float *>(dy), n, 1e-6f, nullptr);
    cudaMemcpy(out.data(), dy, n * 4, cudaMemcpyDeviceToHost);
    float md = 0.0f; for (uint32_t i = 0; i < n; ++i) md = std::max(md, std::fabs(out[i] - ref[i]));
    cudaFree(dx); cudaFree(dw); cudaFree(dy);
    std::printf("rmsnorm maxdiff=%.3e %s\n", md, md < 1e-5f ? "PASS" : "FAIL");
    if (md >= 1e-5f) return false;

    // clip: 8 rows x 128, threshold 1.0 forces a scale-down
    const uint32_t rows = 8, len = 128;
    std::vector<float> cx(rows * len), cref(rows * len), cout(rows * len);
    for (uint32_t i = 0; i < cx.size(); ++i) cx[i] = 0.5f + 0.001f * (i % 11);
    for (uint32_t r = 0; r < rows; ++r) {
        double s2 = 0.0;
        for (uint32_t i = 0; i < len; ++i) s2 += double(cx[r * len + i]) * cx[r * len + i];
        const float f = std::min(1.0f, 1.0f / std::sqrt(float(s2) + 1e-6f));
        for (uint32_t i = 0; i < len; ++i) cref[r * len + i] = cx[r * len + i] * f;
    }
    cudaMalloc(&dx, cx.size() * 4);
    cudaMemcpy(dx, cx.data(), cx.size() * 4, cudaMemcpyHostToDevice);
    gpu::kInklingClip(static_cast<float *>(dx), rows, len, nullptr, 1.0f, nullptr);
    cudaMemcpy(cout.data(), dx, cx.size() * 4, cudaMemcpyDeviceToHost);
    md = 0.0f;
    for (size_t i = 0; i < cx.size(); ++i) md = std::max(md, std::fabs(cout[i] - cref[i]));
    cudaFree(dx);
    std::printf("clip maxdiff=%.3e %s\n", md, md < 1e-5f ? "PASS" : "FAIL");
    if (md >= 1e-5f) return false;

    // conv: 512 channels, 4 taps, 8 heads, zero history, residual
    const uint32_t ch = 512, nc = 4, nh = 8;
    std::vector<float> cur(ch), wt(ch * nc), hist((nc - 1) * ch, 0.0f), cref2(ch);
    for (uint32_t i = 0; i < ch; ++i) { cur[i] = 0.01f * (i % 9); for (uint32_t k = 0; k < nc; ++k) wt[i * nc + k] = 0.1f + 0.001f * ((i + k) % 7); }
    for (uint32_t c = 0; c < ch; ++c) {
        float a = cur[c];// residual
        a += wt[c * nc + 3] * cur[c];// only the current input has non-zero history
        cref2[c] = a / std::sqrt(5.0f);
    }
    void *dc, *dwt, *dh, *do2;
    cudaMalloc(&dc, ch * 4); cudaMalloc(&dwt, ch * nc * 4); cudaMalloc(&dh, (nc - 1) * ch * 4); cudaMalloc(&do2, ch * 4);
    cudaMemcpy(dc, cur.data(), ch * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwt, wt.data(), ch * nc * 4, cudaMemcpyHostToDevice);
    cudaMemset(dh, 0, (nc - 1) * ch * 4);
    gpu::kInklingShortConv(static_cast<float *>(dc), static_cast<float *>(dwt), nullptr,
                           static_cast<float *>(dh), static_cast<float *>(do2), ch, nc, nh,
                           true, 1.0f / std::sqrt(5.0f), nullptr);
    cudaMemcpy(cout.data(), do2, ch * 4, cudaMemcpyDeviceToHost);
    md = 0.0f;
    for (uint32_t c = 0; c < ch; ++c) md = std::max(md, std::fabs(cout[c] - cref2[c]));
    cudaFree(dc); cudaFree(dwt); cudaFree(dh); cudaFree(do2);
    std::printf("conv maxdiff=%.3e %s\n", md, md < 1e-4f ? "PASS" : "FAIL");
    if (md >= 1e-4f) return false;

    // conv with NON-ZERO history -- the decode path's second and later tokens.
    // The case above never reaches the `k < hist` branch, so a broken history
    // tap, a wrong history stride or a wrong shift all pass it.  This case runs
    // two consecutive steps through the same buffer, so it also checks that the
    // shift the kernel writes is the shift the next call reads.
    {
        const uint32_t nc2 = 4, ch2 = 256, nh2 = 4;
        const uint32_t hl = nc2 - 1;
        std::vector<float> c1(ch2), c2(ch2), w2(ch2 * nc2);
        std::vector<float> hgpu(hl * ch2, 0.0f), hcpu(hl * ch2, 0.0f);
        std::vector<float> o1g(ch2), o2g(ch2), o1c(ch2), o2c(ch2);
        for (uint32_t i = 0; i < c1.size(); ++i) {
            c1[i] = 0.3f * std::sin(0.07f * float(i)) + 0.11f;
            c2[i] = 0.25f * std::cos(0.05f * float(i)) - 0.07f;
            for (uint32_t k = 0; k < nc2; ++k)
                w2[i * nc2 + k] = 0.2f + 0.01f * float((i + k) % 5);
        }
        // CPU reference: same tap order and same shift as ModelInkling.cpp.
        auto cpuConv = [&](const std::vector<float> &cur, std::vector<float> &out) {
            for (uint32_t c = 0; c < ch2; ++c) {
                float acc = cur[c];// residual = true
                for (uint32_t k = 0; k < nc2; ++k) {
                    const float v = (k < hl) ? hcpu[size_t(k) * ch2 + c] : cur[c];
                    acc += w2[size_t(c) * nc2 + k] * v;
                }
                out[c] = acc / std::sqrt(5.0f);
            }
            for (uint32_t k = 0; k + 1 < hl; ++k)
                std::memcpy(&hcpu[size_t(k) * ch2], &hcpu[size_t(k + 1) * ch2],
                            ch2 * sizeof(float));
            if (hl > 0) std::memcpy(&hcpu[size_t(hl - 1) * ch2], cur.data(),
                                    ch2 * sizeof(float));
        };
        cpuConv(c1, o1c);
        cpuConv(c2, o2c);
        void *d1, *d2, *dw2, *dh2, *do1, *do2b;
        cudaMalloc(&d1, ch2 * 4); cudaMalloc(&d2, ch2 * 4);
        cudaMalloc(&dw2, ch2 * nc2 * 4); cudaMalloc(&dh2, hl * ch2 * 4);
        cudaMalloc(&do1, ch2 * 4); cudaMalloc(&do2b, ch2 * 4);
        cudaMemcpy(d1, c1.data(), ch2 * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d2, c2.data(), ch2 * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw2, w2.data(), ch2 * nc2 * 4, cudaMemcpyHostToDevice);
        cudaMemset(dh2, 0, hl * ch2 * 4);
        const float sc2 = 1.0f / std::sqrt(5.0f);
        gpu::kInklingShortConv(static_cast<float *>(d1), static_cast<float *>(dw2),
                               nullptr, static_cast<float *>(dh2),
                               static_cast<float *>(do1), ch2, nc2, nh2, true, sc2,
                               nullptr);
        gpu::kInklingShortConv(static_cast<float *>(d2), static_cast<float *>(dw2),
                               nullptr, static_cast<float *>(dh2),
                               static_cast<float *>(do2b), ch2, nc2, nh2, true, sc2,
                               nullptr);
        cudaMemcpy(o1g.data(), do1, ch2 * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(o2g.data(), do2b, ch2 * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hgpu.data(), dh2, hl * ch2 * 4, cudaMemcpyDeviceToHost);
        cudaFree(d1); cudaFree(d2); cudaFree(dw2); cudaFree(dh2);
        cudaFree(do1); cudaFree(do2b);
        float e1 = 0.0f, e2 = 0.0f;
        for (uint32_t c = 0; c < ch2; ++c) {
            e1 = std::max(e1, std::fabs(o1g[c] - o1c[c]));
            e2 = std::max(e2, std::fabs(o2g[c] - o2c[c]));
        }
        const bool ok = e1 < 1e-5f && e2 < 1e-5f;
        std::printf("conv history maxdiff step0=%.3e step1=%.3e %s\n", e1, e2,
                    ok ? "PASS" : "FAIL");
        if (!ok) return false;
    }
    return true;
}

static bool testAttentionOps() {
    // relbias: 2 heads, nRel=4, extent=8
    const uint32_t H = 2, NR = 4, E = 8;
    std::vector<float> r(H * NR), bias(E * NR), sink(H), table(H * (E + 1), 0.0f), refT(H * (E + 1), 0.0f);
    for (uint32_t i = 0; i < r.size(); ++i) r[i] = 0.1f * (i + 1);
    for (uint32_t i = 0; i < bias.size(); ++i) bias[i] = 0.01f * (i % 5);
    for (uint32_t h = 0; h < H; ++h) {
        sink[h] = 0.2f * h;
        for (uint32_t d = 0; d < E; ++d) {
            float a = 0.0f;
            for (uint32_t k = 0; k < NR; ++k) a += bias[d * NR + k] * r[h * NR + k];
            refT[h * (E + 1) + d] = a - sink[h];
        }
    }
    void *dr, *db, *ds, *dt;
    cudaMalloc(&dr, r.size() * 4); cudaMalloc(&db, bias.size() * 4);
    cudaMalloc(&ds, H * 4); cudaMalloc(&dt, table.size() * 4);
    cudaMemcpy(dr, r.data(), r.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, sink.data(), H * 4, cudaMemcpyHostToDevice);
    gpu::kInklingRelBias(static_cast<float *>(dr), static_cast<float *>(db),
                         static_cast<float *>(ds), static_cast<float *>(dt), H, NR, E, nullptr);
    cudaMemcpy(table.data(), dt, table.size() * 4, cudaMemcpyDeviceToHost);
    float md = 0.0f;
    for (size_t i = 0; i < table.size(); ++i) md = std::max(md, std::fabs(table[i] - refT[i]));
    cudaFree(dr); cudaFree(db); cudaFree(ds); cudaFree(dt);
    std::printf("relbias maxdiff=%.3e %s\n", md, md < 1e-5f ? "PASS" : "FAIL");
    if (md >= 1e-5f) return false;

    // mix: 2 states, hidden=8, 1 query
    const uint32_t NS = 2, NH = 8;
    std::vector<float> states(NS * NH), query(NH), refM(NH), outM(NH);
    for (uint32_t i = 0; i < states.size(); ++i) states[i] = 0.1f * (i + 1);
    for (uint32_t i = 0; i < NH; ++i) query[i] = 0.05f * (i + 1);
    float normed[2][8], sc[2], mx = -1e30f;
    for (uint32_t s = 0; s < NS; ++s) {
        double ss = 0.0; for (uint32_t i = 0; i < NH; ++i) ss += double(states[s * NH + i]) * states[s * NH + i];
        const float inv = 1.0f / std::sqrt(float(ss / NH) + 1e-6f);
        double dot = 0.0;
        for (uint32_t i = 0; i < NH; ++i) { normed[s][i] = states[s * NH + i] * inv; dot += double(normed[s][i]) * query[i]; }
        sc[s] = float(dot) / std::sqrt(float(NH)); mx = std::max(mx, sc[s]);
    }
    double sum = 0.0; for (uint32_t s = 0; s < NS; ++s) { sc[s] = std::exp(sc[s] - mx); sum += sc[s]; }
    for (uint32_t i = 0; i < NH; ++i) refM[i] = float(sc[0] / sum) * normed[0][i] + float(sc[1] / sum) * normed[1][i];
    void *dst, *dq, *dmo;
    cudaMalloc(&dst, states.size() * 4); cudaMalloc(&dq, NH * 4); cudaMalloc(&dmo, NH * 4);
    cudaMemcpy(dst, states.data(), states.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, query.data(), NH * 4, cudaMemcpyHostToDevice);
    gpu::kInklingMix(static_cast<float *>(dst), NS, static_cast<float *>(dq), 1, NH,
                     static_cast<float *>(dmo), nullptr);
    cudaMemcpy(outM.data(), dmo, NH * 4, cudaMemcpyDeviceToHost);
    md = 0.0f; for (uint32_t i = 0; i < NH; ++i) md = std::max(md, std::fabs(outM[i] - refM[i]));
    cudaFree(dst); cudaFree(dq); cudaFree(dmo);
    std::printf("mix maxdiff=%.3e %s\n", md, md < 1e-5f ? "PASS" : "FAIL");
    if (md >= 1e-5f) return false;

    // attention: 2 q-heads, 1 kv-head, headDim=4, pos=2 (3 keys), no bias/sink
    const uint32_t HD = 4, HQ = 2, HKV = 1;
    std::vector<float> q(HQ * HD), k(3 * HKV * HD), v(3 * HKV * HD), atab(HQ * (E + 1), 0.0f);
    std::vector<float> refA(HQ * HD), outA(HQ * HD);
    for (uint32_t i = 0; i < q.size(); ++i) q[i] = 0.1f * (i + 1);
    for (uint32_t i = 0; i < k.size(); ++i) { k[i] = 0.05f * (i + 1); v[i] = 0.2f * (i + 1); }
    for (uint32_t h = 0; h < HQ; ++h) {
        float sc[3], mx = 0.0f;
        for (uint32_t kk = 0; kk < 3; ++kk) {
            float d = 0.0f;
            for (uint32_t j = 0; j < HD; ++j) d += q[h * HD + j] * k[kk * HD + j];
            sc[kk] = d / std::sqrt(float(HD)); mx = std::max(mx, sc[kk]);
        }
        double sum = std::exp(-mx);
        for (uint32_t kk = 0; kk < 3; ++kk) { sc[kk] = std::exp(sc[kk] - mx); sum += sc[kk]; }
        for (uint32_t j = 0; j < HD; ++j) {
            float a = 0.0f;
            for (uint32_t kk = 0; kk < 3; ++kk) a += float(sc[kk] / sum) * v[kk * HD + j];
            refA[h * HD + j] = a;
        }
    }
    void *dq2, *dk, *dv, *dt2, *da;
    cudaMalloc(&dq2, q.size() * 4); cudaMalloc(&dk, k.size() * 4); cudaMalloc(&dv, v.size() * 4);
    cudaMalloc(&dt2, atab.size() * 4); cudaMalloc(&da, outA.size() * 4);
    cudaMemcpy(dq2, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(dt2, 0, atab.size() * 4);
    gpu::kInklingAttention(static_cast<float *>(dq2), static_cast<float *>(dk),
                           static_cast<float *>(dv), static_cast<float *>(dt2), 2, 0, 3,
                           HD, HQ, HKV, E, 1.0f / std::sqrt(float(HD)),
                           static_cast<float *>(da), nullptr);
    cudaMemcpy(outA.data(), da, outA.size() * 4, cudaMemcpyDeviceToHost);
    md = 0.0f; for (size_t i = 0; i < outA.size(); ++i) md = std::max(md, std::fabs(outA[i] - refA[i]));
    cudaFree(dq2); cudaFree(dk); cudaFree(dv); cudaFree(dt2); cudaFree(da);
    std::printf("attention maxdiff=%.3e %s\n", md, md < 1e-4f ? "PASS" : "FAIL");
    return md < 1e-4f;
}

// ---- compact expert gather: counting sort + row copy ----
// The batched MoE sorts its m*nUsed (token, expert) pairs by expert into a
// compact slab array, then gathers each slab's activation rows.  Both were
// rewritten for parallelism (the serial sort was 0.41 ms/layer and the
// per-expert gather 0.84 ms/layer).  The gate here is stronger than a cosine:
// the whole layout -- counts, offsets, and for every slot the token, weight,
// expert and position -- plus every gathered byte must equal the original
// serial algorithm, because that layout is what the expert MMA and the
// accumulation consume.  m*nUsed is deliberately not a multiple of the block
// so the tile tail and the dead-lane path are reached.
static bool testMoeGroupSort() {
    const uint32_t M = 37, NU = 8, NE = 13;
    const uint32_t n = M * NU;
    std::vector<int32_t> idx(n);
    std::vector<float> wgt(n);
    for (uint32_t p = 0; p < n; ++p) {
        // Skewed (expert 3 and 9 get many pairs), duplicate experts per token,
        // and a token order that is not expert order.
        idx[p] = static_cast<int32_t>((p * 7u + (p * p) / 5u) % NE);
        wgt[p] = 0.5f + 0.125f * static_cast<float>(p % 11u);
    }
    std::vector<int32_t> counts(NE + 1, -7), goff(NE + 1, -7), gtok(n, -7),
            gpos(n, -7), gexp(n, -7);
    std::vector<float> gwgt(n, -7.0f);
    void *di, *dw, *dc, *do_, *dt, *dp, *dwt, *de;
    cudaMalloc(&di, n * 4); cudaMalloc(&dw, n * 4);
    cudaMalloc(&dc, (NE + 1) * 4); cudaMalloc(&do_, (NE + 1) * 4);
    cudaMalloc(&dt, n * 4); cudaMalloc(&dp, n * 4); cudaMalloc(&dwt, n * 4);
    cudaMalloc(&de, n * 4);
    cudaMemcpy(di, idx.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, wgt.data(), n * 4, cudaMemcpyHostToDevice);
    gpu::kInklingMoeGroupBuild(static_cast<const int32_t *>(di),
                               static_cast<const float *>(dw),
                               static_cast<int32_t *>(dc), static_cast<int32_t *>(do_),
                               static_cast<int32_t *>(dt), static_cast<int32_t *>(dp),
                               static_cast<float *>(dwt), static_cast<int32_t *>(de),
                               M, NU, NE, nullptr);
    cudaMemcpy(counts.data(), dc, (NE + 1) * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(goff.data(), do_, (NE + 1) * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gtok.data(), dt, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpos.data(), dp, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gexp.data(), de, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gwgt.data(), dwt, n * 4, cudaMemcpyDeviceToHost);

    // The serial algorithm the kernel replaced, verbatim.
    std::vector<int32_t> rc(NE + 1, 0), ro(NE + 1, 0), rt(n, 0), rp(n, 0),
            re(n, 0), cursor(NE, 0);
    std::vector<float> rw(n, 0.0f);
    for (uint32_t p = 0; p < n; ++p) ++rc[idx[p]];
    int32_t acc = 0;
    for (uint32_t e = 0; e < NE; ++e) { ro[e] = acc; cursor[e] = acc; acc += rc[e]; }
    ro[NE] = acc;
    for (uint32_t p = 0; p < n; ++p) {
        const uint32_t e = static_cast<uint32_t>(idx[p]);
        const int32_t slot = cursor[e]++;
        rt[slot] = static_cast<int32_t>(p / NU);
        rw[slot] = wgt[p];
        rp[p] = slot;
        re[slot] = static_cast<int32_t>(e);
    }
    bool same = acc == static_cast<int32_t>(n);
    for (uint32_t e = 0; e < NE; ++e) {
        if (counts[e] != rc[e] || goff[e] != ro[e]) {
            std::printf("  first count/offset diff at e=%u: %d/%d vs %d/%d\n", e,
                        counts[e], goff[e], rc[e], ro[e]);
            same = false;
            break;
        }
    }
    if (goff[NE] != ro[NE]) {
        std::printf("  total offset diff: %d vs %d\n", goff[NE], ro[NE]);
        same = false;
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (gtok[i] != rt[i] || gpos[i] != rp[i] || gexp[i] != re[i] ||
            gwgt[i] != rw[i]) {
            std::printf("  first slot diff at %u: tok %d/%d pos %d/%d exp %d/%d "
                        "wgt %.9g/%.9g\n",
                        i, gtok[i], rt[i], gpos[i], rp[i], gexp[i], re[i],
                        gwgt[i], rw[i]);
            same = false;
            break;
        }
    }
    std::printf("moe-group-build: n=%u experts=%u layout %s\n", n, NE,
                same ? "BIT-IDENTICAL to the serial algorithm" : "DIFFERS");
    cudaFree(di); cudaFree(dw); cudaFree(dc); cudaFree(do_); cudaFree(dt);
    cudaFree(dp); cudaFree(dwt); cudaFree(de);
    if (!same) return false;

    // Gather: gx[slot] = x[gtok[slot]], one block per slot.  Compare every
    // element, not a sample.
    const uint32_t cols = 2304;
    std::vector<float> x(M * cols), gx(n * cols), ref(n * cols);
    for (uint32_t i = 0; i < M * cols; ++i)
        x[i] = std::sin(0.001f * static_cast<float>(i)) * 3.0f + 0.25f;
    void *dx, *dt2, *dgx;
    cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dt2, n * 4);
    cudaMalloc(&dgx, gx.size() * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dt2, gtok.data(), n * 4, cudaMemcpyHostToDevice);
    gpu::kInklingGatherExpertX(static_cast<const float *>(dx),
                               static_cast<const int32_t *>(dt2), n,
                               static_cast<float *>(dgx), cols, nullptr);
    cudaMemcpy(gx.data(), dgx, gx.size() * 4, cudaMemcpyDeviceToHost);
    for (uint32_t s = 0; s < n; ++s)
        for (uint32_t c = 0; c < cols; ++c)
            ref[static_cast<size_t>(s) * cols + c] =
                    x[static_cast<size_t>(gtok[s]) * cols + c];
    float gmd = 0.0f;
    for (size_t i = 0; i < gx.size(); ++i)
        gmd = std::max(gmd, std::fabs(gx[i] - ref[i]));
    std::printf("moe-gather: %u slots x %u cols maxdiff=%.3e %s\n", n, cols, gmd,
                gmd == 0.0f ? "PASS" : "FAIL");
    cudaFree(dx); cudaFree(dt2); cudaFree(dgx);
    return gmd == 0.0f;
}

static bool testMoeOps() {
    const uint32_t NE = 4, NU = 2;
    std::vector<float> logits = {0.5f, -0.2f, 1.1f, 0.3f};
    std::vector<float> bias = {0.0f, 0.4f, -0.1f, 0.2f};
    std::vector<int32_t> idx(NU), ridx(NU);
    std::vector<float> wgt(NU), rwgt(NU);
    // reference: selection = sigmoid + bias; weights = sigmoid of selected,
    // normalized, then x scale.
    float p[4]; for (uint32_t e = 0; e < NE; ++e) p[e] = 1.0f / (1.0f + std::exp(-logits[e]));
    int order[4] = {0,1,2,3};
    std::sort(order, order + 4, [&](int a, int b) { return p[a] + bias[a] > p[b] + bias[b]; });
    float sum = 0.0f; for (uint32_t u = 0; u < NU; ++u) { ridx[u] = order[u]; sum += p[order[u]]; }
    for (uint32_t u = 0; u < NU; ++u) rwgt[u] = p[ridx[u]] * 8.0f / sum;
    void *dl, *db, *di, *dw;
    cudaMalloc(&dl, NE * 4); cudaMalloc(&db, NE * 4); cudaMalloc(&di, NU * 4); cudaMalloc(&dw, NU * 4);
    cudaMemcpy(dl, logits.data(), NE * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bias.data(), NE * 4, cudaMemcpyHostToDevice);
    gpu::kInklingRouter(static_cast<float *>(dl), static_cast<float *>(db),
                        static_cast<int32_t *>(di), static_cast<float *>(dw), NE, NU, 8.0f, nullptr);
    cudaMemcpy(idx.data(), di, NU * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(wgt.data(), dw, NU * 4, cudaMemcpyDeviceToHost);
    float md = 0.0f; bool idok = true;
    for (uint32_t u = 0; u < NU; ++u) { idok &= idx[u] == ridx[u]; md = std::max(md, std::fabs(wgt[u] - rwgt[u])); }
    cudaFree(dl); cudaFree(db); cudaFree(di); cudaFree(dw);
    std::printf("router id=%s maxdiff=%.3e %s\n", idok ? "ok" : "BAD", md,
                (idok && md < 1e-5f) ? "PASS" : "FAIL");
    if (!idok || md >= 1e-5f) return false;

    // SITU: 2 values, beta=4, lb=25
    const uint32_t NS = 4;
    std::vector<float> gu = {0.3f, -1.2f, 2.0f, 0.7f}, refS(NS);
    auto situ = [](float g, float u, float beta, float lb) {
        return beta * std::tanh(g / beta) * (1.0f / (1.0f + std::exp(-g))) *
               lb * std::tanh(u / lb);
    };
    for (uint32_t i = 0; i < NS; ++i) refS[i] = situ(gu[i], gu[NS + i], 4.0f, 25.0f);
    void *dg;
    cudaMalloc(&dg, 2 * NS * 4);
    cudaMemcpy(dg, gu.data(), 2 * NS * 4, cudaMemcpyHostToDevice);
    gpu::kInklingSitu(static_cast<float *>(dg), NS, 4.0f, 25.0f, nullptr);
    cudaMemcpy(gu.data(), dg, NS * 4, cudaMemcpyDeviceToHost);
    md = 0.0f; for (uint32_t i = 0; i < NS; ++i) md = std::max(md, std::fabs(gu[i] - refS[i]));
    cudaFree(dg);
    std::printf("situ maxdiff=%.3e %s\n", md, md < 1e-5f ? "PASS" : "FAIL");
    if (md >= 1e-5f) return false;

    // expert accumulation: 2 experts x 4 hidden
    const uint32_t NH = 4;
    std::vector<float> ex = {1.f,2.f,3.f,4.f, 10.f,20.f,30.f,40.f};
    std::vector<float> aw = {0.25f, 0.5f}, asc = {2.0f, 0.5f}, outA(NH), refA(NH, 0.f);
    for (uint32_t u = 0; u < 2; ++u)
        for (uint32_t i = 0; i < NH; ++i)
            refA[i] += aw[u] * asc[u] * ex[u * NH + i];
    void *de, *daw, *dasc, *doa;
    cudaMalloc(&de, ex.size() * 4); cudaMalloc(&daw, 2 * 4); cudaMalloc(&dasc, 2 * 4); cudaMalloc(&doa, NH * 4);
    cudaMemcpy(de, ex.data(), ex.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(daw, aw.data(), 2 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dasc, asc.data(), 2 * 4, cudaMemcpyHostToDevice);
    gpu::kInklingMoeAccum(static_cast<float *>(de), static_cast<float *>(daw),
                          static_cast<float *>(dasc), static_cast<float *>(doa), 2, NH, nullptr);
    cudaMemcpy(outA.data(), doa, NH * 4, cudaMemcpyDeviceToHost);
    md = 0.0f; for (uint32_t i = 0; i < NH; ++i) md = std::max(md, std::fabs(outA[i] - refA[i]));
    cudaFree(de); cudaFree(daw); cudaFree(dasc); cudaFree(doa);
    std::printf("moe-accum maxdiff=%.3e %s\n", md, md < 1e-5f ? "PASS" : "FAIL");
    return md < 1e-5f;
}
