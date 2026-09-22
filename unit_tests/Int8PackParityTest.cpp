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
 * TinyCoder int8 pack parity harness (step 1 of the mma.s8 MMQ project).
 *
 * Verifies the Int8Pack conversion of GGML IQ3_XXS blocks into the
 * signed-int8 + per-group-scale form (and the mma.s8.m16n8k16 B-fragment
 * layout) against the reference dequantization / dot paths:
 *
 *   1. Weight reconstruction is BIT-EXACT vs dequantizeIQ3_XXSBlock
 *      (the pack is a lossless re-encoding: qs[i] = grid*sign, w = db*qs).
 *   2. The packed Q8_K dot matches dotProductIQ3_XXS_Q8K up to float
 *      rounding (per-group scale multiply vs the reference's d*ls fold).
 *   3. The mma B-fragment layout round-trips bit-exactly (packMma ->
 *      unpackMma == canonical unpack), including non-multiple-of-8/16
 *      padding geometry.
 *   4. The mma-layout dot (dotMma, reads the B fragments in lane order)
 *      matches the canonical float dot.
 *
 * Runs on synthetic blocks (always) and on the loaded model's real IQ3_XXS
 * matrices (when a model is present).  This is the shared foundation for the
 * prefill int8 tensor-core GEMM and the decode mma.s8 GEMV paths: any bug in
 * the pack (wrong grid index, sign, scale, or fragment mapping) is caught
 * here before any kernel work starts.
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
#include "SharedTestEnv.hpp"

using namespace tinycoder;

namespace {

    /// @brief Bit pattern of a float (for bit-exact comparisons).
    uint32_t fbits(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    }

    /// @brief Deterministic synthetic GGML IQ3_XXS block (98 bytes).
    /// Random d (fp16, sane range) + random qs[96] exercises every grid
    /// entry, sign pattern and scale nibble.
    void makeSyntheticIQ3XXSBlock(uint8_t *blk, uint64_t seed) {
        std::mt19937_64 rng(seed);
        const float d = std::ldexp(
                1.0f + static_cast<float>(rng() % 1000) / 1000.0f,
                static_cast<int>(rng() % 9) - 4);
        const uint16_t dh = GGMLDequantize::floatToHalf(d);
        std::memcpy(blk, &dh, 2);
        for (uint32_t i = 0; i < 96; ++i) {
            blk[2 + i] = static_cast<uint8_t>(rng() & 0xFFu);
        }
    }

    /// @brief Deterministic activation vector (scaled soft-sines with several
    /// sign changes; a "plain" spread would mask pack bugs behind the
    /// dominant term).
    std::vector<float> makeActivation(uint32_t n) {
        std::vector<float> x(n);
        for (uint32_t i = 0; i < n; ++i) {
            const double t = static_cast<double>(i);
            x[i] = 0.9f * std::sin(0.021 * t) + 0.35f * std::cos(0.0037 * t) +
                   0.15f * std::sin(0.1103 * t + 0.7);
        }
        return x;
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
                    &L.attnQ,       &L.attnK,       &L.attnV,       &L.attnO,
                    &L.ffnGate,     &L.ffnUp,       &L.ffnDown,     &L.attnQKV,
                    &L.attnGate,    &L.ssmAlphaQ,   &L.ssmBetaQ,    &L.ssmOut,
                    &L.ffnGateExps, &L.ffnUpExps,   &L.ffnDownExpsMoe,
                    &L.ffnGateShexp, &L.ffnUpShexp, &L.ffnDownShexp};
            const char *ns[] = {
                    "attnQ",       "attnK",       "attnV",       "attnO",
                    "ffnGate",     "ffnUp",       "ffnDown",     "attnQKV",
                    "attnGate",    "ssmAlphaQ",   "ssmBetaQ",    "ssmOut",
                    "ffnGateExps", "ffnUpExps",   "ffnDownExpsMoe",
                    "ffnGateShexp", "ffnUpShexp", "ffnDownShexp"};
            for (size_t i = 0; i < sizeof(ms) / sizeof(ms[0]); ++i) {
                if (!ms[i]->empty() && ms[i]->type == GGML_TYPE_IQ3_XXS) {
                    mats.push_back({"L" + std::to_string(li) + " " + ns[i],
                                    &ms[i]->data, ms[i]->rows, ms[i]->cols});
                }
            }
        }
        return mats;
    }

    /// @brief Dequantize a full [rows x cols] matrix of IQ3_XXS blocks into
    /// float (row-major), the canonical reference.
    std::vector<float> dequantizeMatrix(const uint8_t *src, uint32_t rows,
                                        uint32_t cols) {
        const uint32_t blocksPerRow =
                (cols + Int8Pack::BLOCK_SIZE - 1) / Int8Pack::BLOCK_SIZE;
        std::vector<float> out(static_cast<size_t>(rows) * cols);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t bc = 0; bc < blocksPerRow; ++bc) {
                const uint8_t *blk =
                        src + (static_cast<size_t>(r) * blocksPerRow + bc) *
                                      Int8Pack::SRC_BLOCK_BYTES;
                float tmp[Int8Pack::BLOCK_SIZE];
                GGMLDequantize::dequantizeIQ3_XXSBlock(blk, tmp);
                const uint32_t colStart = bc * Int8Pack::BLOCK_SIZE;
                const uint32_t colEnd =
                        std::min(colStart + Int8Pack::BLOCK_SIZE, cols);
                for (uint32_t c = colStart; c < colEnd; ++c) {
                    out[static_cast<size_t>(r) * cols + c] = tmp[c - colStart];
                }
            }
        }
        return out;
    }

    /// @brief Canonical float dot of a packed matrix row: y[n] = sum_k W[n][k]
    /// * x[k] (from the canonical PackedBlock form).
    void dotPackedMatrix(const Int8Pack::PackedBlock *blocks,
                         uint32_t blocksPerRow, uint32_t rows, uint32_t cols,
                         const float *x, float *y) {
        std::fill(y, y + rows, 0.0f);
        for (uint32_t r = 0; r < rows; ++r) {
            float acc = 0.0f;
            for (uint32_t bc = 0; bc < blocksPerRow; ++bc) {
                const Int8Pack::PackedBlock &blk =
                        blocks[static_cast<size_t>(r) * blocksPerRow + bc];
                const uint32_t colStart = bc * Int8Pack::BLOCK_SIZE;
                const uint32_t colEnd =
                        std::min(colStart + Int8Pack::BLOCK_SIZE, cols);
                for (uint32_t c = colStart; c < colEnd; ++c) {
                    const uint32_t g = (c - colStart) / Int8Pack::GROUP_SIZE;
                    const uint32_t i = (c - colStart) % Int8Pack::GROUP_SIZE;
                    acc += blk.scales[g] *
                           static_cast<float>(blk.qs[g * Int8Pack::GROUP_SIZE +
                                                     i]) *
                           x[c];
                }
            }
            y[r] = acc;
        }
    }

    /// @brief Check one packed block against the reference dequantization
    /// (bit-exact) and the reference Q8_K dot (tolerance).  Returns the
    /// number of weights checked.
    uint32_t checkBlock(const uint8_t *blk, const Q8KBlock *q8,
                        const char *label) {
        Int8Pack::PackedBlock packed;
        Int8Pack::packBlock(blk, packed);

        // 1. Bit-exact weight reconstruction.
        float ref[Int8Pack::BLOCK_SIZE], got[Int8Pack::BLOCK_SIZE];
        GGMLDequantize::dequantizeIQ3_XXSBlock(blk, ref);
        Int8Pack::unpackBlock(packed, got);
        for (uint32_t i = 0; i < Int8Pack::BLOCK_SIZE; ++i) {
            EXPECT_EQ(fbits(got[i]), fbits(ref[i]))
                    << label << " weight " << i << " packed=" << got[i]
                    << " ref=" << ref[i];
        }

        // 2. Q8_K dot parity (float rounding tolerance).
        if (q8 != nullptr) {
            const float refDot = GGMLDequantize::dotProductIQ3_XXS_Q8K(blk, q8);
            const float gotDot = Int8Pack::dotBlockQ8K(packed, *q8);
            const float absErr = std::fabs(gotDot - refDot);
            const float relErr = absErr / (std::fabs(refDot) + 1e-3f);
            EXPECT_TRUE(absErr < 1e-3f || relErr < 1e-4f)
                    << label << " Q8K dot: ref=" << refDot
                    << " packed=" << gotDot << " absErr=" << absErr
                    << " relErr=" << relErr;
        }
        return Int8Pack::BLOCK_SIZE;
    }

}// namespace

class Int8PackParityTest : public ::testing::Test {};

// ---------------------------------------------------------------------------
// 1. Weight reconstruction parity (BIT-EXACT) + Q8_K dot parity
// ---------------------------------------------------------------------------
TEST_F(Int8PackParityTest, PackedBlockMatchesDequantized) {
    const auto x = makeActivation(Int8Pack::BLOCK_SIZE);
    Q8KBlock q8;
    GGMLDequantize::quantizeQ8K(x.data(), Int8Pack::BLOCK_SIZE, &q8);

    uint32_t checked = 0;

    // Synthetic blocks: every grid entry / sign pattern is exercised.
    for (uint32_t trial = 0; trial < 128; ++trial) {
        uint8_t blk[Int8Pack::SRC_BLOCK_BYTES];
        makeSyntheticIQ3XXSBlock(blk, 0x5EED0000ull + trial);
        checked += checkBlock(blk, &q8, "synthetic");
    }

    // Real model blocks (first 64 blocks of each IQ3_XXS matrix).
    for (const auto &m: collectIq3XxsMatrices()) {
        const uint32_t blocksPerRow =
                (m.cols + Int8Pack::BLOCK_SIZE - 1) / Int8Pack::BLOCK_SIZE;
        const uint32_t nBlocks =
                std::min<uint32_t>(64, m.rows * blocksPerRow);
        for (uint32_t b = 0; b < nBlocks; ++b) {
            const uint8_t *blk =
                    m.data->data() +
                    static_cast<size_t>(b) * Int8Pack::SRC_BLOCK_BYTES;
            checked += checkBlock(blk, &q8, m.name.c_str());
        }
    }

    EXPECT_GT(checked, 0u);
    std::cout << "[Int8Pack] bit-exact weight parity on " << checked
              << " weights (" << (checked / Int8Pack::BLOCK_SIZE)
              << " blocks)" << std::endl;
}

// ---------------------------------------------------------------------------
// 2. mma.s8.m16n8k16 B-fragment layout round-trip (BIT-EXACT)
// ---------------------------------------------------------------------------
TEST_F(Int8PackParityTest, MmaLayoutRoundTrip) {
    // Synthetic matrix with NON-multiple-of-8/16 geometry (13 x 400) to prove
    // the padding path, plus a multiple-of-tile geometry (24 x 512).
    constexpr uint32_t kGeoms[][2] = {{13, 400}, {24, 512}, {8, 16}, {1, 1}};
    for (const auto &g: kGeoms) {
        const uint32_t rows = g[0], cols = g[1];
        const uint32_t blocksPerRow =
                (cols + Int8Pack::BLOCK_SIZE - 1) / Int8Pack::BLOCK_SIZE;
        std::vector<uint8_t> src(static_cast<size_t>(rows) * blocksPerRow *
                                 Int8Pack::SRC_BLOCK_BYTES);
        for (uint32_t b = 0; b < rows * blocksPerRow; ++b) {
            makeSyntheticIQ3XXSBlock(
                    src.data() + static_cast<size_t>(b) *
                                         Int8Pack::SRC_BLOCK_BYTES,
                    0x77770000ull + static_cast<uint64_t>(rows) * 1000 + b);
        }

        // Canonical reference (dequantize the source matrix).
        const auto ref = dequantizeMatrix(src.data(), rows, cols);

        // Pack to the mma B-fragment layout, then unpack back.
        const uint32_t n8Tiles = (rows + Int8Pack::MMA_N - 1) / Int8Pack::MMA_N;
        const uint32_t k16Tiles =
                (cols + Int8Pack::MMA_K - 1) / Int8Pack::MMA_K;
        std::vector<int8_t> b(static_cast<size_t>(n8Tiles) * k16Tiles *
                              Int8Pack::MMA_TILE_BYTES);
        std::vector<float> scales(static_cast<size_t>(n8Tiles) * k16Tiles *
                                  Int8Pack::MMA_N);
        Int8Pack::packMma(src.data(), rows, cols, b.data(), scales.data());
        std::vector<float> got(static_cast<size_t>(rows) * cols);
        Int8Pack::unpackMma(b.data(), scales.data(), rows, cols, got.data());

        for (uint32_t i = 0; i < rows * cols; ++i) {
            EXPECT_EQ(fbits(got[i]), fbits(ref[i]))
                    << "mma round-trip " << rows << "x" << cols << " weight "
                    << i << " got=" << got[i] << " ref=" << ref[i];
        }
        std::cout << "[Int8Pack] mma layout round-trip " << rows << "x" << cols
                  << " bit-exact (" << (rows * cols) << " weights)"
                  << std::endl;
    }
}

// ---------------------------------------------------------------------------
// 3. mma-layout dot vs canonical float dot
// ---------------------------------------------------------------------------
TEST_F(Int8PackParityTest, MmaLayoutDotMatchesReference) {
    constexpr uint32_t rows = 24, cols = 512;
    const uint32_t blocksPerRow =
            (cols + Int8Pack::BLOCK_SIZE - 1) / Int8Pack::BLOCK_SIZE;
    std::vector<uint8_t> src(static_cast<size_t>(rows) * blocksPerRow *
                             Int8Pack::SRC_BLOCK_BYTES);
    for (uint32_t b = 0; b < rows * blocksPerRow; ++b) {
        makeSyntheticIQ3XXSBlock(
                src.data() + static_cast<size_t>(b) * Int8Pack::SRC_BLOCK_BYTES,
                0x88880000ull + b);
    }

    // Canonical packed form + its float dot.
    std::vector<Int8Pack::PackedBlock> packed(rows * blocksPerRow);
    Int8Pack::pack(src.data(), rows * blocksPerRow, packed.data());
    const auto x = makeActivation(cols);
    std::vector<float> yRef(rows);
    dotPackedMatrix(packed.data(), blocksPerRow, rows, cols, x.data(),
                    yRef.data());

    // mma-layout dot (reads the B fragments in lane order).
    const uint32_t n8Tiles = (rows + Int8Pack::MMA_N - 1) / Int8Pack::MMA_N;
    const uint32_t k16Tiles = (cols + Int8Pack::MMA_K - 1) / Int8Pack::MMA_K;
    std::vector<int8_t> b(static_cast<size_t>(n8Tiles) * k16Tiles *
                          Int8Pack::MMA_TILE_BYTES);
    std::vector<float> scales(static_cast<size_t>(n8Tiles) * k16Tiles *
                              Int8Pack::MMA_N);
    Int8Pack::packMma(src.data(), rows, cols, b.data(), scales.data());
    std::vector<float> yGot(rows);
    Int8Pack::dotMma(b.data(), scales.data(), rows, cols, x.data(),
                     yGot.data());

    for (uint32_t r = 0; r < rows; ++r) {
        const float absErr = std::fabs(yGot[r] - yRef[r]);
        const float relErr = absErr / (std::fabs(yRef[r]) + 1e-3f);
        EXPECT_TRUE(absErr < 1e-3f || relErr < 1e-4f)
                << "mma dot row " << r << ": ref=" << yRef[r]
                << " mma=" << yGot[r] << " absErr=" << absErr
                << " relErr=" << relErr;
    }
    std::cout << "[Int8Pack] mma-layout dot matches canonical float dot on "
              << rows << " rows" << std::endl;
}

// ---------------------------------------------------------------------------
// 4. Real-model matrices: mma layout round-trip + dot (the actual use case)
// ---------------------------------------------------------------------------
TEST_F(Int8PackParityTest, RealModelMmaLayoutParity) {
    const auto mats = collectIq3XxsMatrices();
    if (mats.empty()) {
        GTEST_SKIP() << "Model has no IQ3_XXS matrices; skipping real-model "
                        "mma layout parity";
    }

    // Activation sized to the WIDEST matrix (ffnDown is [H x intermediate],
    // e.g. 1536 x 8960 -- wider than the hidden-size gate/up matrices).
    uint32_t widest = 0;
    for (const auto &m: mats) widest = std::max(widest, m.cols);
    const auto x = makeActivation(widest);

    for (const auto &m: mats) {
        SCOPED_TRACE(m.name);
        const uint32_t blocksPerRow =
                (m.cols + Int8Pack::BLOCK_SIZE - 1) / Int8Pack::BLOCK_SIZE;
        const uint32_t n8Tiles = (m.rows + Int8Pack::MMA_N - 1) / Int8Pack::MMA_N;
        const uint32_t k16Tiles =
                (m.cols + Int8Pack::MMA_K - 1) / Int8Pack::MMA_K;

        // Pack the whole matrix to the mma layout.
        std::vector<int8_t> b(static_cast<size_t>(n8Tiles) * k16Tiles *
                              Int8Pack::MMA_TILE_BYTES);
        std::vector<float> scales(static_cast<size_t>(n8Tiles) * k16Tiles *
                                  Int8Pack::MMA_N);
        Int8Pack::packMma(m.data->data(), m.rows, m.cols, b.data(),
                          scales.data());

        // Round-trip: unpackMma vs the canonical dequantization (bit-exact).
        const auto ref = dequantizeMatrix(m.data->data(), m.rows, m.cols);
        std::vector<float> got(static_cast<size_t>(m.rows) * m.cols);
        Int8Pack::unpackMma(b.data(), scales.data(), m.rows, m.cols,
                            got.data());
        uint32_t firstDiff = UINT32_MAX;
        for (uint32_t i = 0; i < m.rows * m.cols; ++i) {
            if (fbits(got[i]) != fbits(ref[i])) {
                firstDiff = i;
                break;
            }
        }
        EXPECT_EQ(firstDiff, UINT32_MAX)
                << m.name << " mma round-trip diverges at weight " << firstDiff
                << " got=" << (firstDiff < m.rows * m.cols ? got[firstDiff] : 0.0f)
                << " ref=" << (firstDiff < m.rows * m.cols ? ref[firstDiff] : 0.0f);

        // Dot parity on the first 64 rows (mma layout vs canonical packed).
        std::vector<Int8Pack::PackedBlock> packed(
                static_cast<size_t>(m.rows) * blocksPerRow);
        Int8Pack::pack(m.data->data(), m.rows * blocksPerRow, packed.data());
        const uint32_t kRows = std::min<uint32_t>(64, m.rows);
        std::vector<float> yRef(kRows), yGot(kRows);
        dotPackedMatrix(packed.data(), blocksPerRow, kRows, m.cols, x.data(),
                        yRef.data());
        Int8Pack::dotMma(b.data(), scales.data(), kRows, m.cols, x.data(),
                         yGot.data());
        for (uint32_t r = 0; r < kRows; ++r) {
            const float absErr = std::fabs(yGot[r] - yRef[r]);
            const float relErr = absErr / (std::fabs(yRef[r]) + 1e-3f);
            EXPECT_TRUE(absErr < 1e-3f || relErr < 1e-4f)
                    << m.name << " mma dot row " << r << ": ref=" << yRef[r]
                    << " mma=" << yGot[r] << " absErr=" << absErr
                    << " relErr=" << relErr;
        }
        std::cout << "[Int8Pack] " << m.name << " " << m.rows << "x" << m.cols
                  << ": mma round-trip bit-exact, dot parity on " << kRows
                  << " rows" << std::endl;
    }
}
