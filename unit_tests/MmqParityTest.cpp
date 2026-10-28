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

/**
 * TinyCoder int8 tensor-core MMQ harness (mma.s8.m8n8k16).
 *
 * Step 2 of the int8 MMQ project: the kernels themselves.  Verifies the
 * GEMM/GEMV kernel (kQuantizeQ8K + kMmqGemm_8x8x16) against the Int8Pack
 * reference computed on the SAME packed weights and the SAME GPU-quantized
 * activations:
 *
 *   1. Instruction layout (emulateMmaCell): the packed B cells really ARE the
 *      m8n8k16 B fragment, checked from the PTX fragment positions (independent
 *      of the pack's own pack/unpack symmetry, which a self-consistent but
 *      wrong layout would pass).
 *   2. Decode  (seq == 1): kernel == host reference (float tolerance).
 *   3. Prefill (seq > 1): kernel == host reference.
 *   4. Real model IQ3_XXS matrices: pack -> kernel == host reference.
 *
 * The target GPU is a Turing RTX 2080 Ti (sm_75): int8 tensor cores exist ONLY
 * for m8n8k16 / m8n8k32 there (mma.m16n8k16 for int8 needs sm_80+), so the pack
 * and the kernel use m8n8k16.  The kernel is skipped (GTEST_SKIP) on a GPU
 * without int8 mma.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "GGMLDequantize.hpp"
#include "Int8Pack.hpp"
#include "Model.hpp"
#include "SIMDMatMulVec.hpp"
#include "SharedTestEnv.hpp"

#if defined(USE_CUDA)
#include "GPUCompute.hpp"
#endif

using namespace tinycoder;

namespace {

    // The device Q8_K block stride (kQ8K_STRIDE in GPUComputeInternal.hpp);
    // independently asserted by the byte-exact quantizer parity tests.
    constexpr uint32_t kMmqQ8KStride = 304;

    /// @brief Deterministic synthetic GGML IQ3_XXS block (98 bytes).
    void makeSyntheticIQ3XXSBlock(uint8_t *blk, uint64_t seed) {
        std::mt19937_64 rng(seed);
        const float d = std::ldexp(
                1.0f + static_cast<float>(rng() % 1000) / 1000.0f,
                static_cast<int>(rng() % 9) - 4);
        const uint16_t dh = GGMLDequantize::floatToHalf(d);
        std::memcpy(blk, &dh, 2);
        for (uint32_t i = 0; i < 96; ++i)
            blk[2 + i] = static_cast<uint8_t>(rng() & 0xFFu);
    }

    /// @brief Deterministic synthetic GGML Q4_K block (144 bytes:
    /// d(2) + dmin(2) + scales[12] + qs[128]).  All 4-bit q values and the
    /// 6-bit (sc, m) fields are random; the dequant w = d*sc*q - dmin*m is the
    /// affine form the kernel must reproduce.
    void makeSyntheticQ4KBlock(uint8_t *blk, uint64_t seed) {
        std::mt19937_64 rng(seed);
        const float d = std::ldexp(
                1.0f + static_cast<float>(rng() % 1000) / 1000.0f,
                static_cast<int>(rng() % 7) - 5);
        const float dmin = std::ldexp(
                1.0f + static_cast<float>(rng() % 1000) / 1000.0f,
                static_cast<int>(rng() % 7) - 6);
        const uint16_t dh = GGMLDequantize::floatToHalf(d);
        const uint16_t dmh = GGMLDequantize::floatToHalf(dmin);
        std::memcpy(blk, &dh, 2);
        std::memcpy(blk + 2, &dmh, 2);
        for (uint32_t i = 0; i < 12; ++i)
            blk[4 + i] = static_cast<uint8_t>(rng() & 0xFFu);
        for (uint32_t i = 0; i < 128; ++i)
            blk[16 + i] = static_cast<uint8_t>(rng() & 0xFFu);
    }

    /// @brief Q4_K sub-block (sc, m) decode (mirror of Int8Pack::q4kScaleAt
    /// and kxQ8KBlockDot<Q4_K>): returns d*sc and dmin*m for group i of blk.
    void q4kSubScale(const uint8_t *blk, uint32_t i, float &dsc, float &dmn) {
        const float d = GGMLDequantize::halfToFloatBranchFree(
                *reinterpret_cast<const uint16_t *>(blk));
        const float dmin = GGMLDequantize::halfToFloatBranchFree(
                *reinterpret_cast<const uint16_t *>(blk + 2));
        const uint8_t *scales = blk + 4;
        uint8_t sc, m;
        if (i < 4) {
            sc = scales[i] & 63u;
            m = scales[i + 4] & 63u;
        } else {
            sc = static_cast<uint8_t>((scales[i + 4] & 0xFu) |
                                      ((scales[i - 4] >> 6) << 4));
            m = static_cast<uint8_t>((scales[i + 4] >> 4) |
                                     ((scales[i] >> 6) << 4));
        }
        dsc = d * static_cast<float>(sc);
        dmn = dmin * static_cast<float>(m);
    }

    /// @brief Q4_K 4-bit q at column c (0..255) of blk (mirror of
    /// Int8Pack::q4kWeightAt): qs[32*(i>>1) + j], low nibble for even i.
    uint8_t q4kQ(const uint8_t *blk, uint32_t c) {
        const uint32_t i = c / 32u;
        const uint32_t j = c % 32u;
        const uint8_t qb = blk[16 + 32u * (i >> 1u) + j];
        return ((i & 1u) == 0u) ? (qb & 0xFu) : (qb >> 4);
    }

    /// @brief Deterministic activation row (scaled soft-sines, several sign
    /// changes) with a per-token phase so batch rows differ.
    std::vector<float> makeActivation(uint32_t n, double phase) {
        std::vector<float> x(n);
        for (uint32_t i = 0; i < n; ++i) {
            const double t = static_cast<double>(i);
            x[i] = 0.9f * std::sin(0.021 * t + phase) +
                   0.35f * std::cos(0.0037 * t + 0.5 * phase) +
                   0.15f * std::sin(0.1103 * t + 0.7);
        }
        return x;
    }

    /// @brief Host reference MMQ: the EXACT float tree of kMmqGemm_8x8x16 on
    /// the canonical PackedBlock weights and the GPU-quantized Q8_K bytes.
    /// The kernel folds a whole 32-column group into ONE exact int32 mma
    /// accumulator (integer add is associative) then applies that group's db
    /// once, so the reference does the same -- per-k16 rounded multiplies would
    /// drift by a few ulps per group and mask a real layout bug.
    /// (cols must be a multiple of 256 so every block is full.)
    void refMmq(const Int8Pack::PackedBlock *packed, uint32_t blocksPerRow,
                uint32_t rows, uint32_t cols, const uint8_t *xq,
                uint32_t seq, float *out) {
        (void) cols;
        for (uint32_t s = 0; s < seq; ++s) {
            const uint8_t *xqTok = xq + static_cast<size_t>(s) * blocksPerRow *
                                                kMmqQ8KStride;
            for (uint32_t n = 0; n < rows; ++n) {
                float total = 0.0f;
                for (uint32_t g = 0; g < blocksPerRow; ++g) {
                    const Int8Pack::PackedBlock &pb =
                            packed[static_cast<size_t>(n) * blocksPerRow + g];
                    const uint8_t *q8 = xqTok + static_cast<size_t>(g) *
                                                        kMmqQ8KStride;
                    const int8_t *yq = reinterpret_cast<const int8_t *>(q8);
                    const float ad =
                            *reinterpret_cast<const float *>(q8 + 288);
                    float grp = 0.0f;
                    // 256 columns per block = 8 groups of 32; one db per group.
                    for (uint32_t gg = 0; gg < Int8Pack::GROUPS_PER_BLOCK; ++gg) {
                        int32_t acc = 0;
                        for (uint32_t i = 0; i < Int8Pack::GROUP_SIZE; ++i) {
                            const uint32_t k = gg * Int8Pack::GROUP_SIZE + i;
                            acc += static_cast<int32_t>(pb.qs[k]) *
                                   static_cast<int32_t>(yq[k]);
                        }
                        grp += pb.scales[gg] * static_cast<float>(acc);
                    }
                    total += 4.0f * ad * grp;
                }
                out[static_cast<size_t>(s) * rows + n] = total;
            }
        }
    }

    /// @brief Host reference for the AFFINE Q4_K MMQ: the EXACT float tree of
    /// kMmqGemmQ4K_Affine on the canonical Q4_K blocks and the GPU-quantized
    /// Q8_K bytes.  Per 256-block the kernel computes
    ///   d8 * ( sum_g dsc_g*D_g - sum_g dmn_g*bsum32_g )
    /// where D_g = sum_{c in g} q_c * y8_c (exact int) and
    /// bsum32_g = bsums[2g] + bsums[2g+1]; the reference does the same.
    void refMmqQ4K(const uint8_t *src, uint32_t blocksPerRow, uint32_t rows,
                   uint32_t cols, const uint8_t *xq, uint32_t seq, float *out) {
        (void) cols;
        for (uint32_t s = 0; s < seq; ++s) {
            const uint8_t *xqTok = xq + static_cast<size_t>(s) * blocksPerRow *
                                                kMmqQ8KStride;
            for (uint32_t n = 0; n < rows; ++n) {
                float total = 0.0f;
                for (uint32_t g = 0; g < blocksPerRow; ++g) {
                    const uint8_t *blkW =
                            src + (static_cast<size_t>(n) * blocksPerRow + g) *
                                          Int8Pack::Q4K_SRC_BLOCK_BYTES;
                    const uint8_t *q8 = xqTok + static_cast<size_t>(g) *
                                                         kMmqQ8KStride;
                    const int8_t *yq = reinterpret_cast<const int8_t *>(q8);
                    const int16_t *bsums =
                            reinterpret_cast<const int16_t *>(q8 + 256);
                    const float d8 = *reinterpret_cast<const float *>(q8 + 288);
                    float mainS = 0.0f, minS = 0.0f;
                    for (uint32_t gg = 0; gg < 8; ++gg) {
                        int32_t acc = 0;
                        for (uint32_t i = 0; i < 32; ++i) {
                            const uint32_t c = gg * 32u + i;
                            acc += static_cast<int32_t>(q4kQ(blkW, c)) *
                                   static_cast<int32_t>(yq[c]);
                        }
                        float dsc, dmn;
                        q4kSubScale(blkW, gg, dsc, dmn);
                        const float bi = static_cast<float>(
                                static_cast<int>(bsums[2 * gg]) +
                                static_cast<int>(bsums[2 * gg + 1]));
                        mainS += dsc * static_cast<float>(acc);
                        minS += dmn * bi;
                    }
                    total += d8 * (mainS - minS);
                }
                out[static_cast<size_t>(s) * rows + n] = total;
            }
        }
    }

    /// @brief Collect the model's IQ3_XXS matrices (name, data, rows, cols).
    struct Iq3XxsMat {
        std::string name;
        const AlignedVector<uint8_t> *data;
        uint32_t rows, cols;
    };

    std::vector<Iq3XxsMat> collectIq3XxsMatrices() {
        std::vector<Iq3XxsMat> mats;
        if (SharedTestEnv::model == nullptr) return mats;
        const auto &layers = SharedTestEnv::model->debugGetLayers();
        for (size_t li = 0; li < layers.size(); ++li) {
            const auto &L = layers[li];
            const QuantizedMatrix *ms[] = {
                    &L.attnQ, &L.attnK, &L.attnV, &L.attnO,
                    &L.ffnGate, &L.ffnUp, &L.ffnDown, &L.attnQKV,
                    &L.attnGate, &L.ssmAlphaQ, &L.ssmBetaQ, &L.ssmOut,
                    &L.ffnGateExps, &L.ffnUpExps, &L.ffnDownExpsMoe,
                    &L.ffnGateShexp, &L.ffnUpShexp, &L.ffnDownShexp};
            const char *ns[] = {
                    "attnQ", "attnK", "attnV", "attnO",
                    "ffnGate", "ffnUp", "ffnDown", "attnQKV",
                    "attnGate", "ssmAlphaQ", "ssmBetaQ", "ssmOut",
                    "ffnGateExps", "ffnUpExps", "ffnDownExpsMoe",
                    "ffnGateShexp", "ffnUpShexp", "ffnDownShexp"};
            for (size_t i = 0; i < sizeof(ms) / sizeof(ms[0]); ++i) {
                if (!ms[i]->empty() && ms[i]->type == GGML_TYPE_IQ3_XXS)
                    mats.push_back({"L" + std::to_string(li) + " " + ns[i],
                                &ms[i]->data, ms[i]->rows, ms[i]->cols});
        }
    }
    return mats;
}

/// @brief Collect the model's Q4_K matrices (name, data, rows, cols).
struct Q4KMat {
    std::string name;
    const AlignedVector<uint8_t> *data;
    uint32_t rows, cols;
};

std::vector<Q4KMat> collectQ4KMatrices() {
    std::vector<Q4KMat> mats;
    if (SharedTestEnv::model == nullptr) return mats;
    const auto &layers = SharedTestEnv::model->debugGetLayers();
    for (size_t li = 0; li < layers.size(); ++li) {
        const auto &L = layers[li];
        const QuantizedMatrix *ms[] = {
                &L.attnQ, &L.attnK, &L.attnV, &L.attnO,
                &L.ffnGate, &L.ffnUp, &L.ffnDown, &L.attnQKV,
                &L.attnGate, &L.ssmAlphaQ, &L.ssmBetaQ, &L.ssmOut,
                &L.ffnGateExps, &L.ffnUpExps, &L.ffnDownExpsMoe,
                &L.ffnGateShexp, &L.ffnUpShexp, &L.ffnDownShexp};
        const char *ns[] = {
                "attnQ", "attnK", "attnV", "attnO",
                "ffnGate", "ffnUp", "ffnDown", "attnQKV",
                "attnGate", "ssmAlphaQ", "ssmBetaQ", "ssmOut",
                "ffnGateExps", "ffnUpExps", "ffnDownExpsMoe",
                "ffnGateShexp", "ffnUpShexp", "ffnDownShexp"};
        for (size_t i = 0; i < sizeof(ms) / sizeof(ms[0]); ++i) {
            if (!ms[i]->empty() && ms[i]->type == GGML_TYPE_Q4_K)
                mats.push_back({"L" + std::to_string(li) + " " + ns[i],
                                &ms[i]->data, ms[i]->rows, ms[i]->cols});
        }
    }
    return mats;
}

/// @brief Pack a synthetic matrix into Int8Pack blocks + mma cells.
struct PackedMatrix {
        std::vector<Int8Pack::PackedBlock> blocks;
        uint32_t blocksPerRow = 0;
        std::vector<int8_t> b;
        std::vector<float> scales;
        uint32_t n8Tiles = 0, k16Tiles = 0;
        uint32_t rows = 0, cols = 0;
    };

    void packSynthetic(PackedMatrix &pm, const uint8_t *src, uint32_t rows,
                       uint32_t cols) {
        pm.rows = rows;
        pm.cols = cols;
        pm.blocksPerRow = (cols + Int8Pack::BLOCK_SIZE - 1) / Int8Pack::BLOCK_SIZE;
        pm.blocks.resize(static_cast<size_t>(rows) * pm.blocksPerRow);
        Int8Pack::pack(src, rows * pm.blocksPerRow, pm.blocks.data());
        pm.n8Tiles = (rows + Int8Pack::MMA_N - 1) / Int8Pack::MMA_N;
        pm.k16Tiles = (cols + Int8Pack::MMA_K - 1) / Int8Pack::MMA_K;
        pm.b.assign(static_cast<size_t>(pm.n8Tiles) * pm.k16Tiles *
                            Int8Pack::MMA_TILE_BYTES,
                    0);
        pm.scales.assign(static_cast<size_t>(pm.n8Tiles) * pm.k16Tiles *
                                 Int8Pack::MMA_N,
                         0.0f);
        Int8Pack::packMma(src, rows, cols, pm.b.data(), pm.scales.data());
    }

}// namespace

class MmqParityTest : public ::testing::Test {};

// ---------------------------------------------------------------------------
// 1. The packed B cell IS the mma.s8.m8n8k16 B fragment (instruction layout)
// ---------------------------------------------------------------------------
TEST_F(MmqParityTest, PackedCellIsMmaBFragment) {
    // One 8x16 tile = the smallest mma cell.  rows == 8 (m-tile), cols == 16
    // (one k16).  The grid reference reads the fragment positions; the
    // PackedBlock reference reads the canonical weights -- they must agree
    // bit-for-bit for a random A.
    constexpr uint32_t rows = 8, cols = 16;
    // 8 rows x 1 block each (blocksPerRow = 1); every row needs its OWN block.
    std::vector<uint8_t> src(rows * Int8Pack::SRC_BLOCK_BYTES);
    for (uint32_t n = 0; n < rows; ++n)
        makeSyntheticIQ3XXSBlock(src.data() +
                                         static_cast<size_t>(n) *
                                                 Int8Pack::SRC_BLOCK_BYTES,
                                 0xABCDEF00ull + n);
    int8_t bCell[128];
    float scales[8];
    Int8Pack::packMma(src.data(), rows, cols, bCell, scales);

    // Logical A[8][16]: the kernel feeds 4 int8 per lane as one b32 word.
    std::mt19937_64 rng(0x1234ull);
    int8_t a[8][16];
    for (uint32_t r = 0; r < 8; ++r)
        for (uint32_t c = 0; c < 16; ++c)
            a[r][c] = static_cast<int8_t>(static_cast<int>(rng() % 31) - 15);

    int32_t dGot[8][8];
    Int8Pack::emulateMmaCell(&a[0][0], bCell, dGot);

    // Reference D[m][n] = sum_k A[m][k] * W[n][k] from the canonical blocks.
    // The 8 rows of W each carry their own block at (n * blocksPerRow + 0)
    // with cols=16 -> blocksPerRow = 1, i.e. block index n.
    Int8Pack::PackedBlock pb[8];
    for (uint32_t n = 0; n < 8; ++n)
        Int8Pack::packBlock(src.data() +
                                    static_cast<size_t>(n) *
                                            Int8Pack::SRC_BLOCK_BYTES,
                            pb[n]);
    for (uint32_t m = 0; m < 8; ++m) {
        for (uint32_t n = 0; n < 8; ++n) {
            int32_t ref = 0;
            // emulateMmaCell returns the RAW integer sum (the kernel applies
            // the per-row db afterwards), so compare against the raw dot.
            for (uint32_t k = 0; k < 16; ++k)
                ref += static_cast<int32_t>(a[m][k]) *
                       static_cast<int32_t>(pb[n].qs[k]);
            EXPECT_EQ(dGot[m][n], ref)
                    << "cell fragment mismatch at row " << m << " col " << n;
        }
    }
    std::cout << "[Mmq] packed B cell == mma.s8.m8n8k16 B fragment (bit-exact)"
              << std::endl;
}

#if !defined(USE_CUDA)
TEST_F(MmqParityTest, KernelsMatchReference) {
    GTEST_SKIP() << "CUDA disabled; MMQ kernels are CUDA-only";
}
#else

// ---------------------------------------------------------------------------
// 2 + 3. Synthetic kernel parity: decode (seq==1) and prefill (seq>1)
// ---------------------------------------------------------------------------
TEST_F(MmqParityTest, SyntheticKernelMatchesReference) {
    tinycoder::gpu::GPUModel gm;
    constexpr uint32_t kCols[][2] = {{24, 512}, {20, 768}, {8, 256}, {16, 1024}};
    bool ran = false;
    for (const auto &g: kCols) {
        const uint32_t rows = g[0], cols = g[1];
        SCOPED_TRACE(std::to_string(rows) + "x" + std::to_string(cols));
        const uint32_t blocksPerRow =
                (cols + Int8Pack::BLOCK_SIZE - 1) / Int8Pack::BLOCK_SIZE;
        std::vector<uint8_t> src(static_cast<size_t>(rows) * blocksPerRow *
                                 Int8Pack::SRC_BLOCK_BYTES);
        for (uint32_t b = 0; b < rows * blocksPerRow; ++b)
            makeSyntheticIQ3XXSBlock(
                    src.data() +
                            static_cast<size_t>(b) * Int8Pack::SRC_BLOCK_BYTES,
                    0x5AA50000ull + b + rows * 131u);

        PackedMatrix pm;
        packSynthetic(pm, src.data(), rows, cols);

        for (uint32_t seq: {1u, 40u}) {
            // Batch activations (seq rows).
            std::vector<float> x(static_cast<size_t>(seq) * cols);
            for (uint32_t s = 0; s < seq; ++s) {
                const auto row = makeActivation(cols, 0.013 * s);
                std::copy(row.begin(), row.end(),
                          x.begin() + static_cast<size_t>(s) * cols);
            }
            std::vector<float> yGpu(static_cast<size_t>(seq) * rows);
            std::vector<uint8_t> xqGpu(static_cast<size_t>(seq) * blocksPerRow *
                                       kMmqQ8KStride);
            if (!gm.debugMmqGemm(pm.b.data(), pm.scales.data(), rows, cols,
                                 x.data(), seq, yGpu.data(), xqGpu.data())) {
                GTEST_SKIP() << "This GPU has no int8 m8n8k16 tensor-core mma";
            }
            ran = true;

            std::vector<float> yRef(static_cast<size_t>(seq) * rows);
            refMmq(pm.blocks.data(), blocksPerRow, rows, cols, xqGpu.data(),
                   seq, yRef.data());

            float maxAbs = 0.0f;
            for (size_t i = 0; i < yGpu.size(); ++i) {
                const float absErr = std::fabs(yGpu[i] - yRef[i]);
                const float relErr = absErr / (std::fabs(yRef[i]) + 1e-3f);
                maxAbs = std::max(maxAbs, absErr);
                EXPECT_TRUE(absErr < 1e-3f || relErr < 1e-4f)
                        << "seq=" << seq << " row=" << (i / rows) << " n="
                        << (i % rows) << " gpu=" << yGpu[i]
                        << " ref=" << yRef[i] << " absErr=" << absErr;
            }
            std::cout << "[Mmq] synthetic " << rows << "x" << cols
                      << " seq=" << seq << " kernel==reference (maxAbs="
                      << maxAbs << ")" << std::endl;
        }
    }
    EXPECT_TRUE(ran);
}

// ---------------------------------------------------------------------------
// 4. Real model IQ3_XXS matrices: prefill + decode parity
// ---------------------------------------------------------------------------
TEST_F(MmqParityTest, RealModelKernelMatchesReference) {
    const auto mats = collectIq3XxsMatrices();
    if (mats.empty())
        GTEST_SKIP() << "Model has no IQ3_XXS matrices";

    tinycoder::gpu::GPUModel gm;
    bool ran = false;
    for (const auto &m: mats) {
        // The kernel handles any cols, but the host reference folds whole
        // 256-blocks; every real IQ3_XXS matrix has cols % 256 == 0.
        if (m.cols % Int8Pack::BLOCK_SIZE != 0) continue;
        SCOPED_TRACE(m.name);
        const uint32_t blocksPerRow = m.cols / Int8Pack::BLOCK_SIZE;

        int8_t *b = nullptr;
        float *scales = nullptr;
        const uint32_t n8Tiles = (m.rows + Int8Pack::MMA_N - 1) / Int8Pack::MMA_N;
        const uint32_t k16Tiles =
                (m.cols + Int8Pack::MMA_K - 1) / Int8Pack::MMA_K;
        std::vector<int8_t> bVec(static_cast<size_t>(n8Tiles) * k16Tiles * 128);
        std::vector<float> scVec(static_cast<size_t>(n8Tiles) * k16Tiles * 8);
        Int8Pack::packMma(m.data->data(), m.rows, m.cols, bVec.data(),
                          scVec.data());
        b = bVec.data();
        scales = scVec.data();

        std::vector<Int8Pack::PackedBlock> packed(
                static_cast<size_t>(m.rows) * blocksPerRow);
        Int8Pack::pack(m.data->data(), m.rows * blocksPerRow, packed.data());

        const uint32_t seq = 16;
        std::vector<float> x(static_cast<size_t>(seq) * m.cols);
        for (uint32_t s = 0; s < seq; ++s) {
            const auto row = makeActivation(m.cols, 0.017 * s + 0.3);
            std::copy(row.begin(), row.end(),
                      x.begin() + static_cast<size_t>(s) * m.cols);
        }
        std::vector<float> yGpu(static_cast<size_t>(seq) * m.rows);
        std::vector<uint8_t> xqGpu(static_cast<size_t>(seq) * blocksPerRow *
                                   kMmqQ8KStride);
        if (!gm.debugMmqGemm(b, scales, m.rows, m.cols, x.data(), seq,
                             yGpu.data(), xqGpu.data())) {
            GTEST_SKIP() << "This GPU has no int8 m8n8k16 tensor-core mma";
        }
        ran = true;

        std::vector<float> yRef(static_cast<size_t>(seq) * m.rows);
        refMmq(packed.data(), blocksPerRow, m.rows, m.cols, xqGpu.data(), seq,
               yRef.data());

        float maxAbs = 0.0f, maxRel = 0.0f;
        for (size_t i = 0; i < yGpu.size(); ++i) {
            const float absErr = std::fabs(yGpu[i] - yRef[i]);
            const float relErr = absErr / (std::fabs(yRef[i]) + 1e-3f);
            maxAbs = std::max(maxAbs, absErr);
            maxRel = std::max(maxRel, relErr);
            EXPECT_TRUE(absErr < 1e-3f || relErr < 1e-4f)
                    << m.name << " seq=" << (i / m.rows) << " n=" << (i % m.rows)
                    << " gpu=" << yGpu[i] << " ref=" << yRef[i];
        }
        std::cout << "[Mmq] " << m.name << " " << m.rows << "x" << m.cols
                  << " seq=" << seq << ": kernel==reference (maxAbs=" << maxAbs
                  << " maxRel=" << maxRel << ")" << std::endl;
    }
    EXPECT_TRUE(ran);
}

// ---------------------------------------------------------------------------
// 5. Affine Q4_K MMQ (kMmqGemmQ4K_Affine): synthetic decode + prefill parity
// ---------------------------------------------------------------------------
TEST_F(MmqParityTest, AffineQ4KSyntheticMatchesReference) {
    tinycoder::gpu::GPUModel gm;
    constexpr uint32_t kCols[][2] = {{24, 512}, {20, 768}, {8, 256}, {16, 1024}};
    bool ran = false;
    for (const auto &g: kCols) {
        const uint32_t rows = g[0], cols = g[1];
        SCOPED_TRACE(std::to_string(rows) + "x" + std::to_string(cols));
        const uint32_t blocksPerRow =
                (cols + Int8Pack::BLOCK_SIZE - 1) / Int8Pack::BLOCK_SIZE;
        std::vector<uint8_t> src(static_cast<size_t>(rows) * blocksPerRow *
                                 Int8Pack::Q4K_SRC_BLOCK_BYTES);
        for (uint32_t b = 0; b < rows * blocksPerRow; ++b)
            makeSyntheticQ4KBlock(
                    src.data() +
                            static_cast<size_t>(b) *
                                    Int8Pack::Q4K_SRC_BLOCK_BYTES,
                    0x4B4B0000ull + b + rows * 101u);

        const uint32_t n8Tiles = (rows + Int8Pack::MMA_N - 1) / Int8Pack::MMA_N;
        const uint32_t k16Tiles =
                (cols + Int8Pack::MMA_K - 1) / Int8Pack::MMA_K;
        const size_t tileN = static_cast<size_t>(n8Tiles) * k16Tiles;
        std::vector<int8_t> qPk(tileN * Int8Pack::MMA_TILE_BYTES);
        std::vector<float> dsc(tileN * Int8Pack::MMA_N);
        std::vector<float> dmn(tileN * Int8Pack::MMA_N);
        Int8Pack::packMmaQ4K(src.data(), rows, cols, qPk.data(), dsc.data(),
                             dmn.data());

        for (uint32_t seq: {1u, 40u}) {
            std::vector<float> x(static_cast<size_t>(seq) * cols);
            for (uint32_t s = 0; s < seq; ++s) {
                const auto row = makeActivation(cols, 0.013 * s + 0.7);
                std::copy(row.begin(), row.end(),
                          x.begin() + static_cast<size_t>(s) * cols);
            }
            std::vector<float> yGpu(static_cast<size_t>(seq) * rows);
            std::vector<uint8_t> xqGpu(static_cast<size_t>(seq) * blocksPerRow *
                                       kMmqQ8KStride);
            if (!gm.debugMmqQ4K(qPk.data(), dsc.data(), dmn.data(), rows, cols,
                                x.data(), seq, yGpu.data(), xqGpu.data())) {
                GTEST_SKIP() << "This GPU has no int8 m8n8k16 tensor-core mma";
            }
            ran = true;

            std::vector<float> yRef(static_cast<size_t>(seq) * rows);
            refMmqQ4K(src.data(), blocksPerRow, rows, cols, xqGpu.data(), seq,
                      yRef.data());

            float maxAbs = 0.0f;
            for (size_t i = 0; i < yGpu.size(); ++i) {
                const float absErr = std::fabs(yGpu[i] - yRef[i]);
                const float relErr = absErr / (std::fabs(yRef[i]) + 1e-3f);
                maxAbs = std::max(maxAbs, absErr);
                EXPECT_TRUE(absErr < 1e-3f || relErr < 1e-4f)
                        << "seq=" << seq << " row=" << (i / rows) << " n="
                        << (i % rows) << " gpu=" << yGpu[i]
                        << " ref=" << yRef[i] << " absErr=" << absErr;
            }
            std::cout << "[Mmq-Q4K] synthetic " << rows << "x" << cols
                      << " seq=" << seq << " kernel==reference (maxAbs="
                      << maxAbs << ")" << std::endl;
        }
    }
    EXPECT_TRUE(ran);
}

// ---------------------------------------------------------------------------
// 6. Real model Q4_K matrices: affine mma parity vs the block reference
// ---------------------------------------------------------------------------
TEST_F(MmqParityTest, AffineQ4KRealModelMatchesReference) {
    const auto mats = collectQ4KMatrices();
    if (mats.empty()) GTEST_SKIP() << "Model has no Q4_K matrices";

    tinycoder::gpu::GPUModel gm;
    bool ran = false;
    for (const auto &m: mats) {
        // The host reference folds whole 256-blocks; real Q4_K rows are
        // 256-aligned, but guard anyway.
        if (m.cols % Int8Pack::BLOCK_SIZE != 0) continue;
        SCOPED_TRACE(m.name);
        const uint32_t blocksPerRow = m.cols / Int8Pack::BLOCK_SIZE;
        const uint32_t n8Tiles = (m.rows + Int8Pack::MMA_N - 1) / Int8Pack::MMA_N;
        const uint32_t k16Tiles =
                (m.cols + Int8Pack::MMA_K - 1) / Int8Pack::MMA_K;
        const size_t tileN = static_cast<size_t>(n8Tiles) * k16Tiles;
        std::vector<int8_t> qPk(tileN * Int8Pack::MMA_TILE_BYTES);
        std::vector<float> dsc(tileN * Int8Pack::MMA_N);
        std::vector<float> dmn(tileN * Int8Pack::MMA_N);
        Int8Pack::packMmaQ4K(m.data->data(), m.rows, m.cols, qPk.data(),
                             dsc.data(), dmn.data());

        const uint32_t seq = 16;
        std::vector<float> x(static_cast<size_t>(seq) * m.cols);
        for (uint32_t s = 0; s < seq; ++s) {
            const auto row = makeActivation(m.cols, 0.017 * s + 0.3);
            std::copy(row.begin(), row.end(),
                      x.begin() + static_cast<size_t>(s) * m.cols);
        }
        std::vector<float> yGpu(static_cast<size_t>(seq) * m.rows);
        std::vector<uint8_t> xqGpu(static_cast<size_t>(seq) * blocksPerRow *
                                   kMmqQ8KStride);
        if (!gm.debugMmqQ4K(qPk.data(), dsc.data(), dmn.data(), m.rows, m.cols,
                            x.data(), seq, yGpu.data(), xqGpu.data())) {
            GTEST_SKIP() << "This GPU has no int8 m8n8k16 tensor-core mma";
        }
        ran = true;

        std::vector<float> yRef(static_cast<size_t>(seq) * m.rows);
        refMmqQ4K(m.data->data(), blocksPerRow, m.rows, m.cols, xqGpu.data(),
                  seq, yRef.data());

        float maxAbs = 0.0f, maxRel = 0.0f;
        for (size_t i = 0; i < yGpu.size(); ++i) {
            const float absErr = std::fabs(yGpu[i] - yRef[i]);
            const float relErr = absErr / (std::fabs(yRef[i]) + 1e-3f);
            maxAbs = std::max(maxAbs, absErr);
            maxRel = std::max(maxRel, relErr);
            EXPECT_TRUE(absErr < 1e-3f || relErr < 1e-4f)
                    << m.name << " seq=" << (i / m.rows) << " n=" << (i % m.rows)
                    << " gpu=" << yGpu[i] << " ref=" << yRef[i];
        }
        std::cout << "[Mmq-Q4K] " << m.name << " " << m.rows << "x" << m.cols
                  << " seq=" << seq << ": kernel==reference (maxAbs=" << maxAbs
                  << " maxRel=" << maxRel << ")" << std::endl;
    }
    EXPECT_TRUE(ran);
}

#endif// USE_CUDA
