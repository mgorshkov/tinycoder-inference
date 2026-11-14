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

// EXL3 GPU kernel parity probe: builds a synthetic trellis, reconstructs it on
// both the CPU reference (EXL3Codec) and the CUDA module (EXL3Kernels — runs on
// Turing/Ampere/Ada, incl. RTX 2080 Ti / 3090 / 4090), and cross-checks the full
// matrix plus a GEMV.  Returns nonzero on any failure so it can gate a
// build/CI step.  No model file required.

#include "EXL3Codec.hpp"
#include "EXL3Kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using tinycoder::EXL3Codec;

namespace {

    // Pack `codes[256]` (one per tile) into the LSB-first wrapped-funnel layout,
    // the exact inverse of EXL3Codec::unpackTile.  Writes 16*K uint16 words.
    void packTileIndependent(const uint16_t *codes, uint32_t K, uint16_t *packed) {
        const uint32_t nWords = 8u * K;
        std::vector<uint32_t> words(nWords, 0u);
        const uint32_t mask = (K >= 16u) ? 0xFFFFu : ((1u << K) - 1u);
        for (uint32_t j = 0; j < 256; ++j) {
            const uint32_t pos = j * K;
            const uint32_t word = pos >> 5;
            const uint32_t within = pos & 31u;
            uint32_t code = codes[j] & mask;
            words[word] |= (code << within);
            if (within + K > 32u) {
                const uint32_t spill = within + K - 32u;
                words[(word + 1u) % nWords] |= (code >> (K - spill));
            }
        }
        for (uint32_t i = 0; i < 16u * K; ++i) {
            packed[i] = static_cast<uint16_t>((words[i >> 1] >> (16u * (i & 1u))) & 0xFFFFu);
        }
    }

    // Half-integer pack: code j takes (KA+1) bits when even, KA when odd.
    // The tile holds exactly 16*KA + 8 uint16 words; the reference
    // unpackTileFrac loads only 8*KA + 4 uint32 words (the wrap modulus is the
    // larger 8*(KA+1), matching EXL3Codec::unpackTileFrac), so the word buffer
    // is sized 8*(KA+1) but only the first 16*KA+8 uint16 are emitted.
    void packTileFracIndependent(const uint16_t *codes, uint32_t KA, uint16_t *packed) {
        const uint32_t KA1 = KA + 1u;
        const uint32_t nWords = 8u * KA1;
        std::vector<uint32_t> words(nWords, 0u);
        uint32_t bitpos = 0;
        for (uint32_t j = 0; j < 256; ++j) {
            const uint32_t width = (j % 2u == 0u) ? KA1 : KA;
            const uint32_t mask = (1u << width) - 1u;
            uint32_t code = codes[j] & mask;
            const uint32_t word = bitpos >> 5;
            const uint32_t within = bitpos & 31u;
            words[word] |= (code << within);
            if (within + width > 32u) {
                const uint32_t spill = within + width - 32u;
                words[(word + 1u) % nWords] |= (code >> (width - spill));
            }
            bitpos += width;
        }
        for (uint32_t i = 0; i < 16u * KA + 8u; ++i) {
            packed[i] = static_cast<uint16_t>((words[i >> 1] >> (16u * (i & 1u))) & 0xFFFFu);
        }
    }

    float maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
        float m = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
        return m;
    }


    int failures = 0;

    void check(bool ok, const std::string &what) {
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
        if (!ok) ++failures;
    }

    // Batched-prefill parity: y = W @ x with x = e_j returns row j of the [k][n]
    // basis matrix, so comparing that row against the F16 the batched path
    // materialises pins the convention the batched GEMM must reproduce.
    void checkMaterializeVsFused(const tinycoder::exl3gpu::Exl3HostLinear &lin,
                                 const void *trellis,
                                 const std::vector<float> &refCpu) {
        const uint32_t k = lin.k;
        const uint32_t n = lin.n;
        const size_t nelem = static_cast<size_t>(k) * n;
        __half *matF16 = nullptr;
        __half *rawF16 = nullptr;
        float *xDev = nullptr;
        float *yDev = nullptr;
        cudaMalloc(&matF16, nelem * sizeof(__half));
        cudaMalloc(&rawF16, nelem * sizeof(__half));
        cudaMalloc(&xDev, k * sizeof(float));
        cudaMalloc(&yDev, n * sizeof(float));

        std::string err;
        bool ok = tinycoder::exl3gpu::materializeF16(trellis, matF16, 0, err);
        check(ok, "materializeF16 call" + (ok ? std::string() : (": " + err)));
        ok = tinycoder::exl3gpu::materializeRawF16(trellis, rawF16, 0, err);
        check(ok, "materializeRawF16 call" + (ok ? std::string() : (": " + err)));

        std::vector<__half> hMat(nelem);
        std::vector<__half> hRaw(nelem);
        cudaMemcpy(hMat.data(), matF16, nelem * sizeof(__half), cudaMemcpyDeviceToHost);
        cudaMemcpy(hRaw.data(), rawF16, nelem * sizeof(__half), cudaMemcpyDeviceToHost);

        // Gate (a) for the fused kernel: decode+WMMA in one pass (raw basis) must
        // reproduce the materialised raw matrix rows.
        {
            std::vector<__half> xh(static_cast<size_t>(16) * k, __float2half(0.0f));
            for (uint32_t r = 0; r < 16u && r < k; ++r)
                xh[static_cast<size_t>(r) * k + r] = __float2half(1.0f);
            __half *xhDev = nullptr;
            float *yG = nullptr;
            cudaMalloc(&xhDev, xh.size() * sizeof(__half));
            cudaMalloc(&yG, static_cast<size_t>(16) * n * sizeof(float));
            cudaMemcpy(xhDev, xh.data(), xh.size() * sizeof(__half), cudaMemcpyHostToDevice);
            std::string ferr;
            const bool fok = tinycoder::exl3gpu::gemmFusedRawLaunch(trellis, xhDev, yG, 16u, 0, ferr);
            check(fok, "fused gemm launch" + (fok ? std::string() : (": " + ferr)));
            if (fok) {
                std::vector<float> hy(static_cast<size_t>(16) * n);
                cudaMemcpy(hy.data(), yG, hy.size() * sizeof(float), cudaMemcpyDeviceToHost);
                float w = 0.0f;
                for (uint32_t r = 0; r < 16u && r < k; ++r)
                    for (uint32_t c = 0; c < n; ++c)
                        w = std::fmax(w, std::fabs(hy[static_cast<size_t>(r) * n + c] -
                                                   __half2float(hRaw[static_cast<size_t>(r) * n + c])));
                check(w < 0.05f,
                      "fused gemm vs raw matrix (maxAbsDiff=" +
                              std::to_string(w) + ")");
            }
            cudaFree(xhDev);
            cudaFree(yG);
        }

        // Multi-strip variant of the check above.  seqLen=16 only ever exercises
        // the first m-strip, so a kernel that walks several strips per block
        // (MLOOP > 1, or any multi-warp/multi-strip tiling) could have a broken
        // strip index and still pass -- the logits probe uses a 12-token prompt,
        // which does not reach a second strip either.  With x = I the GEMM must
        // return rows 0..seqLen-1 of the raw matrix verbatim.
        {
            const uint32_t stripRows = (k < 512u) ? k : 512u;
            const uint32_t mRows = (stripRows < n) ? stripRows : n;
            std::vector<__half> xh(static_cast<size_t>(mRows) * k, __float2half(0.0f));
            for (uint32_t r = 0; r < mRows; ++r)
                xh[static_cast<size_t>(r) * k + r] = __float2half(1.0f);
            __half *xhDev = nullptr;
            float *yG = nullptr;
            cudaMalloc(&xhDev, xh.size() * sizeof(__half));
            cudaMalloc(&yG, static_cast<size_t>(mRows) * n * sizeof(float));
            cudaMemcpy(xhDev, xh.data(), xh.size() * sizeof(__half), cudaMemcpyHostToDevice);
            std::string ferr;
            const bool fok =
                    tinycoder::exl3gpu::gemmFusedRawLaunch(trellis, xhDev, yG, mRows, 0, ferr);
            check(fok, "fused gemm multi-strip launch" + (fok ? std::string() : (": " + ferr)));
            if (fok) {
                std::vector<float> hy(static_cast<size_t>(mRows) * n);
                cudaMemcpy(hy.data(), yG, hy.size() * sizeof(float), cudaMemcpyDeviceToHost);
                float w = 0.0f;
                for (uint32_t r = 0; r < mRows; ++r)
                    for (uint32_t c = 0; c < n; ++c)
                        w = std::fmax(w, std::fabs(hy[static_cast<size_t>(r) * n + c] -
                                                   __half2float(hRaw[static_cast<size_t>(r) * n + c])));
                check(w < 0.05f,
                      "fused gemm multi-strip (" + std::to_string(mRows) +
                              " rows) vs raw matrix (maxAbsDiff=" + std::to_string(w) + ")");
            }
            cudaFree(xhDev);
            cudaFree(yG);
        }

        const uint32_t probes[4] = {0u, 1u, k / 2u, k - 1u};
        float worstMat = 0.0f;
        float worstRaw = 0.0f;
        float worstCpu = 0.0f;
        for (uint32_t p = 0; p < 4u; ++p) {
            const uint32_t j = probes[p];
            std::vector<float> x(k, 0.0f);
            x[j] = 1.0f;
            cudaMemcpy(xDev, x.data(), k * sizeof(float), cudaMemcpyHostToDevice);
            ok = tinycoder::exl3gpu::gemvFused(trellis, xDev, yDev, 0, err);
            check(ok, "gemvFused call" + (ok ? std::string() : (": " + err)));
            std::vector<float> yG(n, 0.0f);
            cudaMemcpy(yG.data(), yDev, n * sizeof(float), cudaMemcpyDeviceToHost);

            float dMat = 0.0f;
            float dRaw = 0.0f;
            float dCpu = 0.0f;
            for (uint32_t c = 0; c < n; ++c) {
                const float g = yG[c];
                dMat = std::fmax(dMat, std::fabs(g - __half2float(hMat[static_cast<size_t>(j) * n + c])));
                dRaw = std::fmax(dRaw, std::fabs(g - __half2float(hRaw[static_cast<size_t>(j) * n + c])));
                dCpu = std::fmax(dCpu, std::fabs(g - refCpu[static_cast<size_t>(j) * n + c]));
            }
            float dMatCpu = 0.0f;
            float dRawCpu = 0.0f;
            for (uint32_t c = 0; c < n; ++c) {
                const float r = refCpu[static_cast<size_t>(j) * n + c];
                dMatCpu = std::fmax(dMatCpu, std::fabs(r - __half2float(hMat[static_cast<size_t>(j) * n + c])));
                dRawCpu = std::fmax(dRawCpu, std::fabs(r - __half2float(hRaw[static_cast<size_t>(j) * n + c])));
            }
            std::printf("    j=%u  mat-vs-fused %.6f  raw-vs-fused %.6f  fused-vs-cpu %.6f  mat-vs-cpu %.6f\n",
                        j, dMat, dRaw, dCpu, dMatCpu);
            worstMat = std::fmax(worstMat, dMat);
            worstRaw = std::fmax(worstRaw, dRaw);
            worstCpu = std::fmax(worstCpu, dCpu);
        }
        check(worstCpu < 1e-2f, "fused gemv row vs cpu (maxAbsDiff=" + std::to_string(worstCpu) + ")");
        check(worstMat < 1e-2f, "materialize row vs fused gemv (maxAbsDiff=" + std::to_string(worstMat) + ")");
        check(worstRaw < 1e-2f, "materializeRaw row vs fused gemv (maxAbsDiff=" + std::to_string(worstRaw) + ")");

        cudaFree(matF16);
        cudaFree(rawF16);
        cudaFree(xDev);
        cudaFree(yDev);
    }

    void runCase(uint32_t k, uint32_t n, uint32_t K10, int codebook, bool signs,
                 uint32_t seed) {
        const bool frac = (K10 % 10u) != 0u;
        const uint32_t K = K10 / 10u;
        const uint32_t tileU16 = frac ? (16u * K + 8u) : (16u * K);
        const uint32_t tilesK = k / 16u;
        const uint32_t tilesN = n / 16u;
        const uint32_t nTiles = tilesK * tilesN;

        std::mt19937 rng(seed);
        // Code values: bounded so the reconstruct math stays in a sane range.
        const uint32_t codeMax = frac ? (1u << (K + 1u)) - 1u : (1u << K) - 1u;
        std::vector<uint16_t> packed(static_cast<size_t>(nTiles) * tileU16);
        {
            std::vector<uint16_t> codes(256);
            for (uint32_t t = 0; t < nTiles; ++t) {
                for (uint32_t j = 0; j < 256; ++j) {
                    codes[j] = static_cast<uint16_t>(rng() % (codeMax + 1u));
                }
                uint16_t *dst = packed.data() + static_cast<size_t>(t) * tileU16;
                if (frac) {
                    packTileFracIndependent(codes.data(), K, dst);
                } else {
                    packTileIndependent(codes.data(), K, dst);
                }
            }
        }

        std::vector<float> suh, svh;
        std::vector<uint16_t> suhP, svhP;
        if (signs) {
            suh.resize(k);
            svh.resize(n);
            for (auto &v : suh) v = (rng() & 1u) ? -1.0f : 1.0f;
            for (auto &v : svh) v = (rng() & 1u) ? -1.0f : 1.0f;
            suhP.assign((k + 15u) / 16u, 0u);
            svhP.assign((n + 15u) / 16u, 0u);
        }

        char label[128];
        std::snprintf(label, sizeof(label), "k=%u n=%u b=%.1f cb=%d signs=%d", k, n,
                      K10 / 10.0, codebook, signs ? 1 : 0);
        std::printf("case %s\n", label);

        // ---- CPU reference ----
        std::vector<float> refCpu(static_cast<size_t>(k) * n);
        if (frac) {
            EXL3Codec::reconstructMatrixFrac(packed.data(), K, codebook,
                                             signs ? suh.data() : nullptr,
                                             signs ? svh.data() : nullptr, k, n,
                                             refCpu.data());
        } else {
            EXL3Codec::reconstructMatrix(packed.data(), K, codebook,
                                         signs ? suh.data() : nullptr,
                                         signs ? svh.data() : nullptr, k, n,
                                         refCpu.data());
        }

        // ---- GPU ----
        tinycoder::exl3gpu::Exl3HostLinear lin{};
        lin.packed = packed.data();
        lin.k = k;
        lin.n = n;
        lin.bitrate10 = K10;
        lin.codebook = codebook;
        lin.suh = signs ? suh.data() : nullptr;
        lin.svh = signs ? svh.data() : nullptr;
        std::string err;
        int64_t h = tinycoder::exl3gpu::uploadLinear(lin, err);
        if (h < 0) {
            check(false, "uploadLinear: " + err);
            return;
        }
        std::vector<float> refGpu(static_cast<size_t>(k) * n);
        bool ok = tinycoder::exl3gpu::reconstruct(h, refGpu.data(), err);
        check(ok, "reconstruct call" + (ok ? std::string() : (": " + err)));
        if (ok) {
            const float d = maxAbsDiff(refCpu, refGpu);
            check(d < 1e-3f, "matrix parity (maxAbsDiff=" + std::to_string(d) + ")");
        }

        // ---- GEMV parity: y = W @ x with W[row][col] = recon[col][row] ----
        std::vector<float> x(k);
        for (auto &v : x) v = static_cast<float>(rng() % 1000u) / 250.0f - 2.0f;
        std::vector<float> yRef(n, 0.0f);
        for (uint32_t row = 0; row < n; ++row) {
            float acc = 0.0f;
            for (uint32_t col = 0; col < k; ++col) {
                acc += x[col] * refCpu[static_cast<size_t>(col) * n + row];
            }
            yRef[row] = acc;
        }
        std::vector<float> yGpu(n, 0.0f);
        ok = tinycoder::exl3gpu::gemv(h, x.data(), yGpu.data(), err);
        check(ok, "gemv call" + (ok ? std::string() : (": " + err)));
        if (ok) {
            const float d = maxAbsDiff(yRef, yGpu);
            check(d < 2e-2f, "gemv parity (maxAbsDiff=" + std::to_string(d) + ")");
        }
        tinycoder::exl3gpu::releaseLinear(h);

        // ---- Batched-prefill basis check (integer bitrates only: the fused
        // upload rejects half-integer ones) ----
        if (!frac) {
            tinycoder::exl3gpu::Exl3FusedLinear fused{};
            std::string ferr;
            if (tinycoder::exl3gpu::uploadFusedLinear(lin, fused, ferr)) {
                checkMaterializeVsFused(lin, fused.trellis, refCpu);
                tinycoder::exl3gpu::releaseFusedLinear(fused);
            } else {
                check(false, "uploadFusedLinear: " + ferr);
            }
        }
        std::printf("\n");
    }

}// namespace

int main() {
    if (!tinycoder::exl3gpu::available()) {
        std::fprintf(stderr, "EXL3 CUDA module unavailable (no device?)\n");
        return 2;
    }
    std::printf("EXL3 GPU parity probe (SM %d%d)\n\n", 7, 5);

    // Integer bitrates across all three codebooks, with and without signs.
    runCase(256, 256, 20, 0, false, 1);
    runCase(256, 256, 20, 0, true, 2);
    runCase(256, 512, 40, 0, true, 3);
    runCase(512, 256, 10, 0, true, 4);
    runCase(256, 256, 20, 1, true, 6);// mcg
    // Half-integer bitrate (mul1 only): 2.5 and 3.5 bpw.
    runCase(256, 256, 25, 2, true, 5);
    runCase(256, 256, 25, 2, false, 7);
    runCase(256, 256, 35, 2, true, 8);
    // Larger shapes / non-square to exercise the transpose + 128-wide bands.
    runCase(512, 512, 40, 0, true, 9);
    runCase(1024, 512, 40, 0, true, 9);
    runCase(1024, 256, 40, 0, true, 10);
    runCase(256, 1024, 40, 0, true, 11);

    // Gate (a): the 27B's real geometries at their production bitrate (1.60 bpw => K=1).
    // The synthetic squares above are 256-1024 on a side; a transpose or tile-order
    // convention error only shows up at the aspect ratios and absolute offsets the
    // model actually uses, which is how three convention bugs survived this long.
    //   attention/MLP in  : k=5120, n=10240 (qkv+gate/up style, n >> k)
    //   attention/MLP out : k=6144, n=5120  (k >> n)
    runCase(5120, 2048, 16, 0, true, 20);
    runCase(5120, 4096, 16, 0, true, 21);
    runCase(5120, 8192, 16, 0, true, 22);
    runCase(5120, 10240, 16, 0, true, 12);
    runCase(6144, 5120, 16, 0, true, 13);

    tinycoder::exl3gpu::shutdown();
    if (failures == 0) {
        std::printf("ALL EXL3 GPU PARITY CHECKS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d EXL3 GPU parity check(s) FAILED\n", failures);
    return 1;
}
