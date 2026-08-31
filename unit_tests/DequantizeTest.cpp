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
 * TinyCoder Dequantization Unit Test
 *
 * Tests the GGML dequantization functions against known constants.
 * These test vectors are derived from the reference implementation
 * to verify correctness of the dequantization routines.
 *
 * Each test creates a known quantized block with specific values and
 * verifies that dequantization produces the expected float values.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <regex>
#include <vector>

#include "GGMLDequantize.hpp"
#include "GGUFLoader.hpp"
#include "SIMDMatMulVec.hpp"

using namespace tinycoder;

// ---------------------------------------------------------------------------
// Test: Q5_1 block dequantization with known constants
// ---------------------------------------------------------------------------
// Q5_1 block format: 32 weights in 32 bytes
//   - 2 bytes: d (half precision scale)
//   - 2 bytes: m (half precision min)
//   - 28 bytes: 32 5-bit values packed in 7 uint32_t's (4 values per uint32_t)
//
// Test vector: create a block where all values are 0, so dequantized = m
TEST(DequantizeTest, Q5_1BlockAllZeros) {
    // Create a Q5_1 block with d=1.0, m=0.0, all weights = 0
    uint8_t blockData[32] = {};
    // Set d = 1.0 (half precision: 0x3C00)
    blockData[0] = 0x00;
    blockData[1] = 0x3C;
    // Set m = 0.0 (half precision: 0x0000)
    blockData[2] = 0x00;
    blockData[3] = 0x00;
    // All qm values are 0 (already zero-initialized)

    float out[32];
    GGMLDequantize::dequantizeQ5_1Block(blockData, out);

    // All values should be 0.0 (since m=0 and all weights are 0)
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(out[i], 0.0f, 1e-6f) << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: Q5_1 block with known values
// ---------------------------------------------------------------------------
// Q5_1 block format (32 bytes):
//   Bytes 0-1:  d (half precision scale)
//   Bytes 2-3:  m (half precision min)
//   Bytes 4-7:  qh (32-bit, bit i = high bit of weight i)
//   Bytes 8-23: ql (16 bytes, byte j holds two 4-bit low nibbles:
//                   ql[j] & 0xF = low bits of weight 2*j,
//                   ql[j] >> 4  = low bits of weight 2*j+1)
//   Bytes 24-31: padding (unused)
//
// Formula: out[i] = q * d + m, where q = (highBit << 4) | low4
TEST(DequantizeTest, Q5_1BlockKnownValues) {
    // Create a Q5_1 block with d=2.0, m=-1.0
    // Half precision: 2.0 = 0x4000, -1.0 = 0xBC00
    uint8_t blockData[32] = {};
    blockData[0] = 0x00;
    blockData[1] = 0x40;// d = 2.0
    blockData[2] = 0x00;
    blockData[3] = 0xBC;// m = -1.0

    // Set all weights to 16 (0b10000):
    //   low4 = 0, highBit = 1
    //   ql bytes: each nibble = 0
    //   qh: all 32 bits = 1
    uint32_t qh = 0xFFFFFFFF;// all high bits = 1
    std::memcpy(blockData + 4, &qh, sizeof(uint32_t));
    // ql: all nibbles = 0 (already zero-initialized)

    float out[32];
    GGMLDequantize::dequantizeQ5_1Block(blockData, out);

    // q = (1 << 4) | 0 = 16
    // out = 16 * 2.0 + (-1.0) = 32.0 - 1.0 = 31.0
    float expected = 16.0f * 2.0f + (-1.0f);// = 31.0
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(out[i], expected, 1e-4f) << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: Q5_K block dequantization
// ---------------------------------------------------------------------------
// Q5_K block format: 256 weights in 176 bytes
//   - 2 bytes: d (half precision super-block scale)
//   - 2 bytes: dmin (half precision super-block min)
//   - 16 bytes: scales (8 high bits + 8 low bits)
//   - 32 bytes: qh (32 x 8-bit high bits)
//   - 128 bytes: ql (128 x 8-bit low bits)
//   - 4 bytes: padding
//
// Test: create a minimal block and verify no crashes
TEST(DequantizeTest, Q5_KBlockNoCrash) {
    uint8_t blockData[176] = {};
    // Set d = 1.0
    blockData[0] = 0x00;
    blockData[1] = 0x3C;
    // Set dmin = 0.0
    blockData[2] = 0x00;
    blockData[3] = 0x00;

    float out[256];
    GGMLDequantize::dequantizeQ5_KBlock(blockData, out);

    // Just verify no crash and values are finite
    for (int i = 0; i < 256; ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: Q4_K block dequantization
// ---------------------------------------------------------------------------
TEST(DequantizeTest, Q4_KBlockNoCrash) {
    uint8_t blockData[144] = {};
    // Set d = 1.0
    blockData[0] = 0x00;
    blockData[1] = 0x3C;
    // Set dmin = 0.0
    blockData[2] = 0x00;
    blockData[3] = 0x00;

    float out[256];
    GGMLDequantize::dequantizeQ4_KBlock(blockData, out);

    for (int i = 0; i < 256; ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: IQ3_XXS block dequantization
// ---------------------------------------------------------------------------
// IQ3_XXS block format: 256 weights in 98 bytes
//   - 2 bytes: d (half precision scale)
//   - 64 bytes: qs (quantized indices)
//   - 32 bytes: scales_and_signs
//
// Test: create a block with d=1.0, all indices=0, all signs=0
TEST(DequantizeTest, IQ3_XXSBlockNoCrash) {
    uint8_t blockData[98] = {};
    // Set d = 1.0
    blockData[0] = 0x00;
    blockData[1] = 0x3C;

    float out[256];
    GGMLDequantize::dequantizeIQ3_XXSBlock(blockData, out);

    for (int i = 0; i < 256; ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: IQ3_S block dequantization
// ---------------------------------------------------------------------------
TEST(DequantizeTest, IQ3_SBlockNoCrash) {
    uint8_t blockData[110] = {};
    // Set d = 1.0
    blockData[0] = 0x00;
    blockData[1] = 0x3C;

    float out[256];
    GGMLDequantize::dequantizeIQ3_SBlock(blockData, out);

    for (int i = 0; i < 256; ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: IQ2_S block dequantization
// ---------------------------------------------------------------------------
TEST(DequantizeTest, IQ2_SBlockNoCrash) {
    uint8_t blockData[82] = {};
    // Set d = 1.0
    blockData[0] = 0x00;
    blockData[1] = 0x3C;

    float out[256];
    GGMLDequantize::dequantizeIQ2_SBlock(blockData, out);

    for (int i = 0; i < 256; ++i) {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: IQ2_XS + IQ3_S fused dot products match dequantize-then-dot reference
// ---------------------------------------------------------------------------
TEST(DequantizeTest, IQ2XS_IQ3S_FusedDotMatchesReference) {
    // Deterministic pseudo-random block bytes (seeded, so reproducible).
    auto fill = [](uint8_t *dst, size_t n, uint64_t seed) {
        uint64_t s = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        for (size_t i = 0; i < n; ++i) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            dst[i] = static_cast<uint8_t>(s >> 33);
        }
    };

    auto reference = [](uint32_t type, const uint8_t *blk, const float *x) {
        float out[256];
        GGMLDequantize::dequantizeBlock(type, blk, out, 256);
        double dot = 0.0;
        for (int i = 0; i < 256; ++i) dot += static_cast<double>(x[i]) * out[i];
        return static_cast<float>(dot);
    };

    // x must be a column vector of sane floats (NOT raw byte patterns, which
    // can be NaN/Inf bit patterns).
    float x[256];
    {
        uint64_t s = 777 * 6364136223846793005ULL + 1442695040888963407ULL;
        for (int i = 0; i < 256; ++i) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            double u = static_cast<double>(s >> 11) / static_cast<double>(1ULL << 53);
            x[i] = static_cast<float>(2.0 * u - 1.0);
        }
    }

    // IQ2_XS: 74-byte block
    {
        uint8_t blk[74];
        fill(blk, sizeof(blk), 1234);
        // d is at bytes 0-1 (fp16); fix to 1.25 (0x3D20 LE)
        blk[0] = 0x20;
        blk[1] = 0x3D;
        float ref = reference(GGML_TYPE_IQ2_XS, blk, x);
        float got = GGMLDequantize::dotProductIQ2_XS(blk, x);
        EXPECT_NEAR(got, ref, 1e-3f * std::max(1.0f, std::fabs(ref)))
                << "IQ2_XS fused dot deviates from reference";
    }

    // IQ3_S: 110-byte block
    {
        uint8_t blk[110];
        fill(blk, sizeof(blk), 5678);
        blk[0] = 0x20;
        blk[1] = 0x3D;
        float ref = reference(GGML_TYPE_IQ3_S, blk, x);
        float got = GGMLDequantize::dotProductIQ3_S(blk, x);
        EXPECT_NEAR(got, ref, 1e-3f * std::max(1.0f, std::fabs(ref)))
                << "IQ3_S fused dot deviates from reference";
    }

    // Also verify the generic dotProductFused dispatch reaches the new dots.
    {
        uint8_t blk[74];
        fill(blk, sizeof(blk), 9999);
        blk[0] = 0x20;
        blk[1] = 0x3D;
        float viaDispatch = GGMLDequantize::dotProductFused(
                GGML_TYPE_IQ2_XS, blk, x, 256);
        float direct = GGMLDequantize::dotProductIQ2_XS(blk, x);
        EXPECT_EQ(viaDispatch, direct)
                << "dotProductFused did not dispatch to dotProductIQ2_XS";
    }
}

// ---------------------------------------------------------------------------
// Test: Generic dequantizeBlock dispatcher
// ---------------------------------------------------------------------------
TEST(DequantizeTest, GenericDequantizeBlockAllTypes) {
    // Test that the generic dequantizeBlock dispatcher works for all types
    // by comparing with the type-specific functions

    // Create a Q5_1 block with known values
    uint8_t blockData[256] = {};// max block size
    // Set d = 1.5, m = 0.5
    // Half: 1.5 = 0x3E00, 0.5 = 0x3800
    blockData[0] = 0x00;
    blockData[1] = 0x3E;
    blockData[2] = 0x00;
    blockData[3] = 0x38;
    // Set qh: alternating high bits
    uint32_t qh = 0xAAAAAAAA;
    std::memcpy(blockData + 4, &qh, sizeof(uint32_t));
    // Set ql: alternating nibbles
    for (int i = 0; i < 16; ++i) {
        blockData[8 + i] = 0xAB;// nibbles: A=10, B=11
    }

    float outDirect[32];
    float outGeneric[32];
    GGMLDequantize::dequantizeQ5_1Block(blockData, outDirect);
    GGMLDequantize::dequantizeBlock(GGML_TYPE_Q5_1, blockData, outGeneric, 32);

    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(outDirect[i], outGeneric[i], 1e-6f) << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Q8_1 / Q8_K block + bulk + fused dot + fused matmul correctness.
//
// Q8_1 (36 B/block: d fp16 + s fp16 + 32 x int8) and Q8_K (292 B/block:
// d f32 + 256 x int8 + 16 x int16 bsums) are used by the qwen35 recurrent
// gated-delta-net weights (ssm_alpha/ssm_beta as Q8_1, ssm_out as Q8_K).
// They were previously missing from ggmlTypeSize()/dequantizeBlock(), which
// both corrupted the load (overread) and zeroed the fused matmul output.
// ---------------------------------------------------------------------------
TEST(DequantizeTest, Q8_1AndQ8_K_DequantDotMatmul) {
    // ---- Q8_1: fabricate two 32-wide blocks with known d/s/qs ----
    uint8_t q8_1[2][36] = {};
    {
        // Block 0: d = 0.5 (0x3800), s = 1.0 (0x3C00), qs[i] = i - 16
        uint8_t *blk = q8_1[0];
        blk[0] = 0x00;
        blk[1] = 0x38;// d = 0.5
        blk[2] = 0x00;
        blk[3] = 0x3C;// s = 1.0
        for (int i = 0; i < 32; ++i) blk[4 + i] = static_cast<uint8_t>(static_cast<int8_t>(i - 16));
        // Block 1: d = -0.25 (0xB000), qs = alternating -8/+8
        uint8_t *blk1 = q8_1[1];
        blk1[0] = 0x00;
        blk1[1] = 0xB0;// d = -0.25
        for (int i = 0; i < 32; ++i) blk1[4 + i] = static_cast<uint8_t>((i & 1) ? 8 : -8);
    }

    float x[64];
    for (int i = 0; i < 64; ++i) x[i] = 0.01f * static_cast<float>(i + 1);

    // dequantizeBlock for Q8_1
    float deq[32];
    GGMLDequantize::dequantizeBlock(GGML_TYPE_Q8_1, q8_1[0], deq, 32);
    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(deq[i], 0.5f * static_cast<float>(i - 16), 1e-5f)
                << "Q8_1 block0 dequant idx " << i;
    }

    // dotProductFused for Q8_1
    double refDot0 = 0.0;
    for (int i = 0; i < 32; ++i) refDot0 += static_cast<double>(x[i]) * deq[i];
    EXPECT_NEAR(GGMLDequantize::dotProductFused(GGML_TYPE_Q8_1, q8_1[0], x, 32),
                static_cast<float>(refDot0), 1e-4f);

    // ---- Q8_K: fabricate one 256-wide block ----
    uint8_t q8k[292] = {};
    {
        float d = 0.001f;
        std::memcpy(q8k, &d, sizeof(float));
        for (int i = 0; i < 256; ++i) q8k[4 + i] = static_cast<uint8_t>(static_cast<int8_t>(i - 128));
    }
    float deqK[256];
    GGMLDequantize::dequantizeBlock(GGML_TYPE_Q8_K, q8k, deqK, 256);
    double refDotK = 0.0;
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(deqK[i], 0.001f * static_cast<float>(i - 128), 1e-6f)
                << "Q8_K dequant idx " << i;
        refDotK += static_cast<double>(x[i % 32] + 0.0 * i) * deqK[i];// reuse x[0..31]
    }
    // Separate x vector covering all 256
    std::vector<float> xK(256);
    for (int i = 0; i < 256; ++i) xK[i] = 0.001f * static_cast<float>(i - 128);
    double refDotK2 = 0.0;
    for (int i = 0; i < 256; ++i) refDotK2 += static_cast<double>(xK[i]) * deqK[i];
    EXPECT_NEAR(GGMLDequantize::dotProductFused(GGML_TYPE_Q8_K, q8k, xK.data(), 256),
                static_cast<float>(refDotK2), 1e-5f);

    // ---- Full fused matmul: rows x cols from Q8_1 blocks ----
    constexpr uint32_t ROWS = 4;
    constexpr uint32_t COLS = 64;// 2 blocks of 32 per row
    std::vector<uint8_t> matQ8_1(ROWS * 2 * 36);
    for (uint32_t r = 0; r < ROWS; ++r) {
        std::memcpy(matQ8_1.data() + r * 72 + 0, q8_1[0], 36);
        std::memcpy(matQ8_1.data() + r * 72 + 36, q8_1[1], 36);
    }
    std::vector<float> yFused(ROWS);
    GGMLDequantize::matMulVecFused(GGML_TYPE_Q8_1, matQ8_1.data(), x, ROWS, COLS, yFused.data());
    // Reference: dequantize whole matrix then dot per row
    auto deqAll = GGMLDequantize::dequantize(GGML_TYPE_Q8_1, matQ8_1.data(), ROWS * COLS);
    ASSERT_EQ(deqAll.size(), ROWS * COLS);
    for (uint32_t r = 0; r < ROWS; ++r) {
        double dot = 0.0;
        for (uint32_t i = 0; i < COLS; ++i) {
            dot += static_cast<double>(x[i]) * deqAll[r * COLS + i];
        }
        EXPECT_NEAR(yFused[r], static_cast<float>(dot), 1e-4f)
                << "Q8_1 fused matmul row " << r;
    }

    // ---- Full fused matmul for Q8_K: one row of 256 ----
    std::vector<float> yK(1);
    GGMLDequantize::matMulVecFused(GGML_TYPE_Q8_K, q8k, xK.data(), 1, 256, yK.data());
    EXPECT_NEAR(yK[0], static_cast<float>(refDotK2), 1e-5f)
            << "Q8_K fused matmul mismatch";
}
// ---------------------------------------------------------------------------
TEST(DequantizeTest, MatMulVecFusedMatchesDequantizeDot) {
    // Create a small quantized matrix and verify that matMulVecFused produces
    // the same result as full dequantize + manual dot product.
    //
    // GGUF stores weight matrices as (rows x cols) = (out_features x in_features)
    // in row-major order. Each output row j has blocksPerRow quantized blocks,
    // each covering blockSize input features.
    // W[j][i] = deq[j * cols + i] (within each row's blocks).
    // The computation is: y_j = sum_i x[i] * W[j][i]
    //
    // We use COLS < BLOCK_SIZE to test partial-block handling.

    constexpr uint32_t ROWS = 32;// output features
    constexpr uint32_t COLS = 2; // input features
    constexpr uint32_t TYPE = GGML_TYPE_Q5_1;
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr uint32_t TYPE_SIZE = 24;// Q5_1 block size: d(2) + m(2) + qh(4) + ql(16) = 24

    // Create quantized data for the matrix stored as (rows x cols).
    // With ROWS=32 output rows and COLS=2 input features, each output row has
    // blocksPerRow = ceil(2/32) = 1 block covering 2 input features.
    // The remaining 30 elements of each block are padding (not used).
    uint32_t blocksPerRow = (COLS + BLOCK_SIZE - 1) / BLOCK_SIZE;// 1
    uint64_t totalBytes = static_cast<uint64_t>(ROWS) * blocksPerRow * TYPE_SIZE;
    std::vector<uint8_t> quantData(totalBytes, 0);

    // Helper to fill a Q5_1 block at a given byte offset
    auto fillBlock = [&](uint64_t byteOffset, float d, float m, uint8_t qValue) {
        // d as fp16
        uint16_t dHalf = 0;
        if (d == 1.0f) dHalf = 0x3C00;
        else if (d == 2.0f)
            dHalf = 0x4000;
        std::memcpy(&quantData[byteOffset], &dHalf, sizeof(uint16_t));
        // m as fp16
        uint16_t mHalf = 0;
        if (m == 0.0f) mHalf = 0x0000;
        else if (m == -0.5f)
            mHalf = 0xB800;
        std::memcpy(&quantData[byteOffset + 2], &mHalf, sizeof(uint16_t));
        // qh: high bit for each of 32 weights
        uint32_t qh = 0;
        uint8_t low4 = qValue & 0xF;
        uint8_t highBit = (qValue >> 4) & 1;
        for (int i = 0; i < 32; ++i) {
            if (highBit) qh |= (1u << i);
        }
        std::memcpy(&quantData[byteOffset + 4], &qh, sizeof(uint32_t));
        // ql: low nibbles, packed 2 per byte
        uint8_t nibblePair = static_cast<uint8_t>(low4 | (low4 << 4));
        for (int i = 0; i < 16; ++i) {
            quantData[byteOffset + 8 + i] = nibblePair;
        }
    };

    // Fill blocks for each output row.
    // Row 0: d=1.0, m=0.0, q=10 → deq = 10*1.0 + 0.0 = 10.0
    // Row 1: d=2.0, m=-0.5, q=20 → deq = 20*2.0 + (-0.5) = 39.5
    // Rows 2-31: d=1.0, m=0.0, q=10 → deq = 10.0
    fillBlock(0, 1.0f, 0.0f, 10);         // row 0
    fillBlock(TYPE_SIZE, 2.0f, -0.5f, 20);// row 1
    for (uint32_t j = 2; j < ROWS; ++j) {
        fillBlock(static_cast<uint64_t>(j) * TYPE_SIZE, 1.0f, 0.0f, 10);
    }

    // Input vector x: all 1.0
    std::vector<float> x(COLS, 1.0f);

    // Method 1: matMulVecFused
    std::vector<float> resultFused(ROWS, 0.0f);
    GGMLDequantize::matMulVecFused(TYPE, quantData.data(), x.data(), ROWS, COLS,
                                   resultFused.data());

    // Method 2: full dequantize + dot product
    // Dequantize ALL blocks (ROWS blocks = ROWS * BLOCK_SIZE elements).
    // Then W[j][i] = deq[j * BLOCK_SIZE + i] for i in [0, COLS).
    auto deq = GGMLDequantize::dequantize(
            TYPE, quantData.data(),
            static_cast<uint64_t>(ROWS) * BLOCK_SIZE);
    ASSERT_FALSE(deq.empty());
    std::vector<float> resultManual(ROWS, 0.0f);
    for (uint32_t j = 0; j < ROWS; ++j) {
        float dot = 0.0f;
        for (uint32_t i = 0; i < COLS; ++i) {
            // W[j][i] is at position j * BLOCK_SIZE + i in the flat deq array
            dot += x[i] * deq[static_cast<size_t>(j) * BLOCK_SIZE + i];
        }
        resultManual[j] = dot;
    }

    // Compare
    for (uint32_t j = 0; j < ROWS; ++j) {
        EXPECT_NEAR(resultFused[j], resultManual[j], 1e-4f)
                << "Mismatch at row " << j;
    }
}

// ---------------------------------------------------------------------------
// Test: SIMD dotProductFMA produces correct results
// ---------------------------------------------------------------------------
TEST(DequantizeTest, DotProductFMA) {
    // Test dotProductFMA with known vectors
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    float expected = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        expected += a[i] * b[i];
    }

    float result = dotProductFMA(a.data(), b.data(), static_cast<uint32_t>(a.size()));
    EXPECT_NEAR(result, expected, 1e-4f);

    // Test with different sizes
    expected = 1.0f * 8.0f + 2.0f * 7.0f + 3.0f * 6.0f;
    result = dotProductFMA(a.data(), b.data(), 3);
    EXPECT_NEAR(result, expected, 1e-4f);
}

// ---------------------------------------------------------------------------
// Test: accumulateFMA produces correct results
// ---------------------------------------------------------------------------
TEST(DequantizeTest, AccumulateFMA) {
    std::vector<float> local = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> blockOut = {0.5f, 1.0f, 1.5f, 2.0f};
    float alpha = 2.0f;

    accumulateFMA(local.data(), blockOut.data(), alpha, 4);

    EXPECT_NEAR(local[0], 1.0f + 2.0f * 0.5f, 1e-6f);
    EXPECT_NEAR(local[1], 2.0f + 2.0f * 1.0f, 1e-6f);
    EXPECT_NEAR(local[2], 3.0f + 2.0f * 1.5f, 1e-6f);
    EXPECT_NEAR(local[3], 4.0f + 2.0f * 2.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Test: typeSize and blockSize return correct values
// ---------------------------------------------------------------------------
TEST(DequantizeTest, TypeSizeAndBlockSize) {
    // Verify that type sizes match expected values
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_F32), 4u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_F16), 2u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_Q5_1), 24u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_Q5_K), 176u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_Q4_K), 144u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_IQ3_XXS), 98u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_IQ3_S), 110u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_IQ2_S), 82u);
    // Q8_1 (d fp16 + s fp16 + 32 x int8) and Q8_K (d f32 + 256 x int8 + 16 x int16)
    // are used by qwen35 ssm_alpha/ssm_beta/ssm_out weights. They must resolve to
    // the correct per-block byte sizes or loadQuantized overreads the file.
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_Q8_1), 36u);
    EXPECT_EQ(ggmlTypeSize(GGML_TYPE_Q8_K), 292u);

    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_F32), 1u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_Q5_1), 32u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_Q5_K), 256u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_Q4_K), 256u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_IQ3_XXS), 256u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_IQ3_S), 256u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_IQ2_S), 256u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_Q8_1), 32u);
    EXPECT_EQ(ggmlBlockSize(GGML_TYPE_Q8_K), 256u);
}

// ---------------------------------------------------------------------------
// Test: Dequantize roundtrip for all supported types
// ---------------------------------------------------------------------------
TEST(DequantizeTest, DequantizeRoundtrip) {
    // Create a small buffer of quantized data for each type and verify
    // that dequantize produces the right number of output elements.

    struct TypeTest {
        uint32_t type;
        uint32_t blockSize;
        uint32_t typeSize;
        const char *name;
    };

    TypeTest types[] = {
            {GGML_TYPE_Q5_1, 32, 32, "Q5_1"},
            {GGML_TYPE_Q5_K, 256, 176, "Q5_K"},
            {GGML_TYPE_Q4_K, 256, 144, "Q4_K"},
            {GGML_TYPE_IQ3_XXS, 256, 98, "IQ3_XXS"},
            {GGML_TYPE_IQ3_S, 256, 110, "IQ3_S"},
            {GGML_TYPE_IQ2_S, 256, 82, "IQ2_S"},
    };

    for (const auto &t: types) {
        // Create 2 blocks of quantized data
        uint64_t numElements = static_cast<uint64_t>(t.blockSize) * 2;
        uint64_t numBlocks = (numElements + t.blockSize - 1) / t.blockSize;
        uint64_t dataBytes = numBlocks * t.typeSize;
        std::vector<uint8_t> data(dataBytes, 0);

        // Set scale to 1.0 for the first block
        if (t.type == GGML_TYPE_Q5_1) {
            data[0] = 0x00;
            data[1] = 0x3C;// d = 1.0
        } else {
            data[0] = 0x00;
            data[1] = 0x3C;// d = 1.0 (half)
        }

        auto deq = GGMLDequantize::dequantize(t.type, data.data(), numElements);
        ASSERT_EQ(deq.size(), numElements)
                << "Dequantize returned wrong size for " << t.name;

        // All values should be finite
        for (uint64_t i = 0; i < numElements; ++i) {
            EXPECT_FALSE(std::isnan(deq[i])) << "NaN at index " << i << " for " << t.name;
            EXPECT_FALSE(std::isinf(deq[i])) << "Inf at index " << i << " for " << t.name;
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Tokenizer pretokenize with GPT-2 pattern
// ---------------------------------------------------------------------------
TEST(DequantizeTest, TokenizerPretokenizePattern) {
    // Verify that the GPT-2 regex pattern compiles and matches correctly
    // Uses POSIX character classes for GCC std::regex compatibility
    std::string pattern =
            R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^[:space:][:alpha:][:digit:]]+|[[:space:]]+(?![^[:space:]])|[[:space:]]+)";

    try {
        std::regex re(pattern, std::regex::ECMAScript);
        SUCCEED() << "GPT-2 regex pattern compiles successfully";

        // Test matching on a sample string
        std::string test = "Hello world! 42";
        std::sregex_iterator iter(test.begin(), test.end(), re);
        std::sregex_iterator end;
        std::vector<std::string> matches;
        for (; iter != end; ++iter) {
            matches.push_back(iter->str());
        }

        // GPT-2 should produce: ["Hello", " world", "!", " 42"]
        ASSERT_EQ(matches.size(), 4u) << "Expected 4 tokens";
        EXPECT_EQ(matches[0], "Hello");
        EXPECT_EQ(matches[1], " world");
        EXPECT_EQ(matches[2], "!");
        EXPECT_EQ(matches[3], " 42");
    } catch (const std::regex_error &e) {
        FAIL() << "GPT-2 regex pattern failed to compile: " << e.what();
    }
}

// ---------------------------------------------------------------------------
// Test: Q2_K original vs pre-packed kernel comparison
// ---------------------------------------------------------------------------
// Creates a synthetic Q2_K block, pre-packs it, and compares the dot product
// results from the original and pre-packed kernels for the same input vector.
// This validates that the pre-packed kernel produces identical results.
TEST(DequantizeTest, Q2_K_PrePackedVsOriginal) {
    // Create a synthetic Q2_K block (84 bytes)
    uint8_t blockData[84] = {};

    // Set d = 2.0 (fp16: 0x4000)
    blockData[80] = 0x00;
    blockData[81] = 0x40;

    // Set dmin = 0.5 (fp16: 0x3800)
    blockData[82] = 0x00;
    blockData[83] = 0x38;

    // Set scales[16]: each byte has 4-bit scale (low) and 4-bit min (high)
    // Make them vary so we test all combinations
    for (int i = 0; i < 16; ++i) {
        blockData[i] = static_cast<uint8_t>((i << 4) | (15 - i));
    }

    // Set qs[64]: pack 2-bit values (0, 1, 2, 3) into each byte
    // Byte k contains values for elements [4k, 4k+1, 4k+2, 4k+3]
    // bit0-1 = element 4k, bit2-3 = element 4k+1, bit4-5 = element 4k+2, bit6-7 = element 4k+3
    for (int i = 0; i < 64; ++i) {
        uint8_t byteVal = 0;
        for (int b = 0; b < 4; ++b) {
            uint8_t val = static_cast<uint8_t>((i * 4 + b) % 4);
            byteVal |= (val << (b * 2));
        }
        blockData[16 + i] = byteVal;
    }

    // Pre-pack the block
    auto prepacked = GGMLDequantize::prepackQ2_K(blockData, 256);
    ASSERT_EQ(prepacked.size(), 276u);

    // Create a random x vector of 256 floats
    float x[256];
    std::srand(42);
    for (int i = 0; i < 256; ++i) {
        x[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // Compute dot product using original kernel
    float originalResult = dotProductQ2_K_SIMD(blockData, x);

    // Compute dot product using pre-packed kernel
    float prepackedResult = dotProductQ2_K_PrePacked_SIMD(prepacked.data(), x);

    // Also compute using the scalar reference (dequantize then dot)
    float deqRef[256];
    GGMLDequantize::dequantizeQ2_KBlock(blockData, deqRef);
    double refDot = 0.0;
    for (int i = 0; i < 256; ++i) {
        refDot += static_cast<double>(x[i]) * deqRef[i];
    }
    float referenceResult = static_cast<float>(refDot);

    // Print results for debugging
    std::cout << "  Original kernel:  " << originalResult << std::endl;
    std::cout << "  Pre-packed kernel: " << prepackedResult << std::endl;
    std::cout << "  Reference (deq):   " << referenceResult << std::endl;

    // Both kernels should match the reference
    EXPECT_NEAR(originalResult, referenceResult, 1e-4f)
            << "Original kernel differs from reference dequantize-then-dot";

    EXPECT_NEAR(prepackedResult, referenceResult, 1e-4f)
            << "Pre-packed kernel differs from reference dequantize-then-dot";

    // The two kernels should match each other
    EXPECT_NEAR(originalResult, prepackedResult, 1e-4f)
            << "Original and pre-packed kernels produce different results";
}

// ---------------------------------------------------------------------------
// Test: Q8_K dot product kernel vs reference
// ---------------------------------------------------------------------------
// Validates that the Q8_K-quantized dot product kernel (which uses
// _mm256_maddubs_epi16) produces results matching the reference
// dequantize-then-dot computation within Q8_K quantization tolerance.
TEST(DequantizeTest, Q2_K_Q8KernelVsReference) {
    // Create a synthetic Q2_K block (84 bytes)
    uint8_t blockData[84] = {};

    // Set d = 2.0 (fp16: 0x4000)
    blockData[80] = 0x00;
    blockData[81] = 0x40;

    // Set dmin = 0.5 (fp16: 0x3800)
    blockData[82] = 0x00;
    blockData[83] = 0x38;

    // Set scales[16]: each byte has 4-bit scale (low) and 4-bit min (high)
    for (int i = 0; i < 16; ++i) {
        blockData[i] = static_cast<uint8_t>((i << 4) | (15 - i));
    }

    // Set qs[64]: pack 2-bit values (0, 1, 2, 3) into each byte
    for (int i = 0; i < 64; ++i) {
        uint8_t byteVal = 0;
        for (int b = 0; b < 4; ++b) {
            uint8_t val = static_cast<uint8_t>((i * 4 + b) % 4);
            byteVal |= (val << (b * 2));
        }
        blockData[16 + i] = byteVal;
    }

    // Pre-pack the block
    auto prepacked = GGMLDequantize::prepackQ2_K(blockData, 256);
    ASSERT_EQ(prepacked.size(), 276u);

    // Create a random x vector of 256 floats
    float x[256];
    std::srand(42);
    for (int i = 0; i < 256; ++i) {
        x[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // Quantize x to Q8_K
    Q8KBlock q8;
    GGMLDequantize::quantizeQ8K(x, 256, &q8);

    // Compute dot product using the Q8_K kernel
    float q8Result = dotProductQ2_K_PrePacked_Q8_SIMD(prepacked.data(), &q8);

    // Reference: dequantize then dot
    float deqRef[256];
    GGMLDequantize::dequantizeQ2_KBlock(blockData, deqRef);
    double refDot = 0.0;
    for (int i = 0; i < 256; ++i) {
        refDot += static_cast<double>(x[i]) * deqRef[i];
    }
    float referenceResult = static_cast<float>(refDot);

    std::cout << "  Q8_K kernel:   " << q8Result << std::endl;
    std::cout << "  Reference:     " << referenceResult << std::endl;

    // Q8_K quantization introduces ~1/127 relative error per element, so use a
    // relative tolerance rather than an absolute one. The reference magnitude is
    // ~350 here, so 1e-2 absolute would be far too tight.
    const float relTol = 0.01f * std::max(1.0f, std::fabs(referenceResult));
    EXPECT_NEAR(q8Result, referenceResult, relTol)
            << "Q8_K kernel differs from reference dequantize-then-dot";
}

// ---------------------------------------------------------------------------
// Test: quantizeQ2KRow round-trip (Lever C load-time quantizer).
// Quantizes a float row to compact Q2_K, dequantizes with dequantizeQ2_KBlock,
// and checks the mean abs error is small (Q2_K is inherently lossy ~1/3 scale).
// This localizes quantizer bugs (layout/scale) vs kernel bugs.
// ---------------------------------------------------------------------------
TEST(DequantizeTest, QuantizeQ2KRowRoundtrip) {
    constexpr uint32_t N = 256;
    float x[N];
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    for (uint32_t i = 0; i < N; ++i) x[i] = dist(rng);

    uint8_t q2k[84];
    GGMLDequantize::quantizeQ2KRow(x, N, q2k);
    float deq[N];
    GGMLDequantize::dequantizeQ2_KBlock(q2k, deq);

    double sumErr = 0.0, sumAbs = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        sumErr += std::fabs(deq[i] - x[i]);
        sumAbs += std::fabs(x[i]);
    }
    double meanErr = sumErr / N;
    double meanAbs = sumAbs / N;
    std::cout << "  meanErr=" << meanErr << " meanAbs=" << meanAbs
              << " rel=" << (meanAbs > 0 ? meanErr / meanAbs : 0.0) << std::endl;
    // Q2_K is a ~2-bit codebook: worst-case per-element error is ~1/3 of the
    // group range, so ~35% mean relative error is inherent (the +ml bug above
    // showed ~140%, a layout/scale corruption — well beyond this bound).
    EXPECT_LT(meanErr, 0.5 * meanAbs + 1e-6)
            << "Q2_K round-trip error too large: quantizer layout/scale bug";
}

// ---------------------------------------------------------------------------
// Test: fused gate+up+down Q2K variant vs scalar reference (Lever C kernel).
// Validates matMulVecFusedGateUpDownQ2K_Q2K_SIMD against a scalar reference
// (dequantize-then-dot), localizing bugs in the new Q2_K ffnDown phase.
// ---------------------------------------------------------------------------
TEST(DequantizeTest, FusedGateUpDownQ2KDownVsReference) {
    constexpr uint32_t BLOCK = 256;
    constexpr uint32_t ROWS = 256;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-0.03f, 0.03f);

    // x (hidden), gate/up/down rows: 256 x 256 (single block per row).
    // Matrix layout is row-major [rows][blocksPerRow], so a rows x cols matrix
    // needs rows*ceil(cols/256) blocks (the kernel reads per-row stride).
    std::vector<float> x(BLOCK);
    std::vector<std::vector<float>> gateF(ROWS, std::vector<float>(BLOCK));
    std::vector<std::vector<float>> upF(ROWS, std::vector<float>(BLOCK));
    std::vector<std::vector<float>> downF(ROWS, std::vector<float>(BLOCK));
    for (uint32_t i = 0; i < BLOCK; ++i) x[i] = dist(rng);
    for (uint32_t r = 0; r < ROWS; ++r)
        for (uint32_t j = 0; j < BLOCK; ++j) {
            gateF[r][j] = dist(rng);
            upF[r][j] = dist(rng);
            downF[r][j] = dist(rng);
        }

    std::vector<uint8_t> gateQ(ROWS * 84), upQ(ROWS * 84), downQ(ROWS * 84);
    for (uint32_t r = 0; r < ROWS; ++r) {
        GGMLDequantize::quantizeQ2KRow(gateF[r].data(), BLOCK, gateQ.data() + r * 84);
        GGMLDequantize::quantizeQ2KRow(upF[r].data(), BLOCK, upQ.data() + r * 84);
        GGMLDequantize::quantizeQ2KRow(downF[r].data(), BLOCK, downQ.data() + r * 84);
    }

    std::vector<float> out(BLOCK, 0.0f), residual(BLOCK, 0.01f), ref(BLOCK);
    bool used = matMulVecFusedGateUpDownQ2K_Q2K_SIMD(
            gateQ.data(), upQ.data(), downQ.data(), x.data(),
            BLOCK, BLOCK, BLOCK, out.data(), residual.data());
    ASSERT_TRUE(used) << "AVX2 fused Q2K kernel not dispatched";

    // Scalar reference: dequantize the full ROWS x BLOCK matrices, then
    // act[j] = silu(sum_k x[k]*gate[j,k]) * sum_k x[k]*up[j,k], and
    // ref[i] = sum_j act[j]*down[i,j] + residual[i].
    std::vector<float> gateDeq(ROWS * BLOCK), upDeq(ROWS * BLOCK);
    std::vector<float> downDeq(ROWS * BLOCK);
    for (uint32_t r = 0; r < ROWS; ++r) {
        GGMLDequantize::dequantizeQ2_KBlock(gateQ.data() + r * 84,
                                            gateDeq.data() + r * BLOCK);
        GGMLDequantize::dequantizeQ2_KBlock(upQ.data() + r * 84,
                                            upDeq.data() + r * BLOCK);
        GGMLDequantize::dequantizeQ2_KBlock(downQ.data() + r * 84,
                                            downDeq.data() + r * BLOCK);
    }
    std::vector<float> act(BLOCK);
    for (uint32_t j = 0; j < BLOCK; ++j) {
        double g = 0.0, u = 0.0;
        for (uint32_t k = 0; k < BLOCK; ++k) {
            g += static_cast<double>(x[k]) * gateDeq[j * BLOCK + k];
            u += static_cast<double>(x[k]) * upDeq[j * BLOCK + k];
        }
        float gv = static_cast<float>(g);
        act[j] = gv / (1.0f + std::exp(-gv)) * static_cast<float>(u);
    }
    for (uint32_t i = 0; i < BLOCK; ++i) {
        double d = 0.0;
        for (uint32_t k = 0; k < BLOCK; ++k) {
            d += static_cast<double>(act[k]) * downDeq[i * BLOCK + k];
        }
        ref[i] = static_cast<float>(d) + residual[i];
    }

    double maxAbs = 0.0, maxRef = 0.0;
    for (uint32_t i = 0; i < BLOCK; ++i) {
        maxAbs = std::max(maxAbs, static_cast<double>(std::fabs(out[i] - ref[i])));
        maxRef = std::max(maxRef, static_cast<double>(std::fabs(ref[i])));
    }
    std::cout << "  maxAbsDiff=" << maxAbs << " maxRef=" << maxRef << std::endl;
    const double tol = 0.05 * std::max(1.0, maxRef) + 1e-3;
    EXPECT_LT(maxAbs, tol)
            << "Fused Q2K-down kernel deviates from scalar reference";
}

// ---------------------------------------------------------------------------
// Test: matMulVecBatchQ2K_Compact_SIMD vs the per-block scalar Q2_K dot path.
// Exercises the LM-head compact Q2_K batch kernel (Lever C) over multiple
// vocab rows and tokens (seqLen=2 to cover the [row][token] accumulator
// indexing); every output is compared against the dequantize-then-dot
// reference the non-AVX2 fallback uses.
// ---------------------------------------------------------------------------
TEST(DequantizeTest, BatchQ2KCompactVsScalarDot) {
    constexpr uint32_t BLOCK = 256;
    constexpr uint32_t ROWS = 24;
    constexpr uint32_t SEQ = 2;
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    // Quantize each row of an ROWS x BLOCK matrix to compact Q2_K.
    std::vector<std::vector<float>> wF(ROWS, std::vector<float>(BLOCK));
    std::vector<uint8_t> wQ(ROWS * 84);
    for (uint32_t r = 0; r < ROWS; ++r) {
        for (uint32_t j = 0; j < BLOCK; ++j) wF[r][j] = dist(rng);
        GGMLDequantize::quantizeQ2KRow(wF[r].data(), BLOCK, wQ.data() + r * 84);
    }
    std::vector<float> wDeq(ROWS * BLOCK);
    for (uint32_t r = 0; r < ROWS; ++r) {
        GGMLDequantize::dequantizeQ2_KBlock(wQ.data() + r * 84,
                                            wDeq.data() + r * BLOCK);
    }

    // Two token vectors; out is [seqLen][rows] row-major (LM-head convention).
    std::vector<float> x(SEQ * BLOCK);
    for (uint32_t s = 0; s < SEQ; ++s)
        for (uint32_t j = 0; j < BLOCK; ++j) x[s * BLOCK + j] = dist(rng);
    std::vector<float> out(SEQ * ROWS, 0.0f);

    bool used = matMulVecBatchQ2K_Compact_SIMD(wQ.data(), x.data(), SEQ, ROWS,
                                               BLOCK, out.data());
    ASSERT_TRUE(used) << "AVX2 compact Q2_K batch kernel not dispatched";

    // Reference: direct float dot against the dequantized matrix.
    double maxAbs = 0.0, maxRef = 0.0;
    for (uint32_t s = 0; s < SEQ; ++s) {
        for (uint32_t r = 0; r < ROWS; ++r) {
            double d = 0.0;
            for (uint32_t j = 0; j < BLOCK; ++j) {
                d += static_cast<double>(x[s * BLOCK + j]) * wDeq[r * BLOCK + j];
            }
            float ref = static_cast<float>(d);
            maxAbs = std::max(maxAbs, static_cast<double>(std::fabs(out[s * ROWS + r] - ref)));
            maxRef = std::max(maxRef, static_cast<double>(std::fabs(ref)));
        }
    }
    std::cout << "  maxAbsDiff=" << maxAbs << " maxRef=" << maxRef << std::endl;
    // Q8_K quant of x contributes ~1/127 relative error; the reference above is
    // exact-float so the tolerance must absorb it.
    const double tol = 0.01 * std::max(1.0, maxRef) + 1e-3;
    EXPECT_LT(maxAbs, tol)
            << "Compact Q2_K batch kernel deviates from scalar dot reference";
}

// ---------------------------------------------------------------------------
// Test: Q5_K IQ4_XS IQ4_NL batch SIMD kernels vs exact scalar dequantize+dot.
// Fabricates quantized block bytes hand-per-layout (no encoder exists for
// these types), calls the same matMulVecBatch*_SIMD kernels the generic
// QuantizedMatrix::matMulVec dispatches for the qwen35 layer matmuls, and
// compares against dequantizeBlock + float dot. This isolates the kernel math
// from the forward pipeline (the Qwen3.8-27B sample-question regression).
// ---------------------------------------------------------------------------
namespace {

    // Write a float as an fp16 bit pattern (little-endian) at `dst`.
    void writeFp16(uint8_t *dst, float v) {
        uint16_t h = GGMLDequantize::floatToHalf(v);
        std::memcpy(dst, &h, sizeof(uint16_t));
    }

    // Reference: dot(x, dequantize(block)) using the exact scalar block path.
    float scalarBlockDot(uint32_t type, const uint8_t *block, const float *x) {
        float blockOut[256];
        GGMLDequantize::dequantizeBlock(type, block, blockOut, 256);
        double dot = 0.0;
        for (uint32_t i = 0; i < 256; ++i) {
            dot += static_cast<double>(x[i]) * blockOut[i];
        }
        return static_cast<float>(dot);
    }

    // One Q5_K block (176 B). Sub-block i (0..7) has scale d*sc[i], min dmin*m[i],
    // and 32 weights of 0..31 where the 5th bit for sub-block element j comes
    // from qh[j] bit (2*chunk + half). We iterate the SAME indexing scheme the
    // reference dequantizeQ5_KBlock uses (chunk c of 4, half h of 2) to keep the
    // qs/qh offsets and qh bit positions in lockstep.
    void fabricateQ5KBlock(uint8_t *blk, float d, float dmin, const float sc[8],
                           const float m[8], const uint8_t w[256]) {
        std::memset(blk, 0, 176);
        writeFp16(blk + 0, d);
        writeFp16(blk + 2, dmin);
        uint8_t *scales = blk + 4;
        uint8_t *qh = blk + 16;
        uint8_t *qs = blk + 48;
        // scales: 6-bit d per sub-block + 6-bit m per sub-block (the packed
        // 12-byte layout getScaleMin reads back).
        for (int j = 0; j < 8; ++j) {
            uint8_t sv = static_cast<uint8_t>(sc[j]) & 63;
            uint8_t mv = static_cast<uint8_t>(m[j]) & 63;
            if (j < 4) {
                scales[j] |= sv;
                scales[j + 4] |= mv;
            } else {
                scales[j + 4] |= (sv & 0xF) | ((mv & 0xF) << 4);
                scales[j - 4] |= (sv >> 4) << 6;
                scales[j] |= (mv >> 4) << 6;
            }
        }
        for (int c = 0; c < 4; ++c) {
            const int u1 = 1 << (2 * c), u2 = 1 << (2 * c + 1);
            for (int l = 0; l < 32; ++l) {
                uint8_t v0 = w[c * 64 + l];
                uint8_t v1 = w[c * 64 + 32 + l];
                qs[c * 32 + l] = (v0 & 0xF) | ((v1 & 0xF) << 4);
                if (v0 & 16) qh[l] |= static_cast<uint8_t>(u1);
                if (v1 & 16) qh[l] |= static_cast<uint8_t>(u2);
            }
        }
    }

    // One IQ4_XS block (136 B). Sub-block ib (0..7) has scale dl = d*(ls-32),
    // ls packed as scales_l[ib/2] nibbles + 2-bit fields of the 16-bit scales_h
    // word at bit positions 2*ib (`(scales_h >> 2*ib) & 3`, little-endian);
    // qs advances 16 bytes per sub-block (reference dequantizeIQ4_XS).
    void fabricateIQ4XSBlock(uint8_t *blk, float d, const uint8_t ls[8],
                             const uint8_t nib[128]) {
        std::memset(blk, 0, 136);
        writeFp16(blk + 0, d);
        uint16_t scales_h16 = 0;
        uint8_t *scales_l = blk + 4;
        uint8_t *qs = blk + 8;
        for (int ib = 0; ib < 8; ++ib) {
            scales_l[ib / 2] |= (ls[ib] & 0xF) << (4 * (ib % 2));
            scales_h16 |= static_cast<uint16_t>(((ls[ib] >> 4) & 0x3)
                                                << (2 * ib));
            std::memcpy(qs + 16 * ib, nib + 16 * ib, 16);
        }
        std::memcpy(blk + 2, &scales_h16, sizeof(uint16_t));
    }

    // One IQ4_NL block (18 B): d + 16 nibbles, weight j = kvalues_iq4nl[nib].
    void fabricateIQ4NLBlock(uint8_t *blk, float d, const uint8_t nib[16]) {
        std::memset(blk, 0, 18);
        writeFp16(blk + 0, d);
        std::memcpy(blk + 2, nib, 16);
    }

}// namespace

TEST(DequantizeTest, Q5KBatchSIMDVsScalarDot) {
    constexpr uint32_t BLOCK = 256;
    constexpr uint32_t ROWS = 16;
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    std::vector<uint8_t> wQ(ROWS * 176);
    for (uint32_t r = 0; r < ROWS; ++r) {
        float sc[8], m[8];
        uint8_t w[256];
        for (int i = 0; i < 8; ++i) {
            sc[i] = static_cast<float>(1 + rng() % 63);
            m[i] = static_cast<float>(rng() % 32);
        }
        for (int i = 0; i < 256; ++i) w[i] = static_cast<uint8_t>(rng() % 32);
        fabricateQ5KBlock(wQ.data() + r * 176, 0.01f, 0.002f, sc, m, w);
    }

    std::vector<float> x(BLOCK);
    for (uint32_t i = 0; i < BLOCK; ++i) x[i] = dist(rng);
    std::vector<float> out(ROWS, 0.0f);
    bool used = matMulVecBatchQ5K_SIMD(wQ.data(), x.data(), 1, ROWS, BLOCK, out.data());
    ASSERT_TRUE(used) << "Q5_K SIMD batch kernel not dispatched";

    double maxAbs = 0.0, maxRef = 0.0;
    for (uint32_t r = 0; r < ROWS; ++r) {
        float ref = scalarBlockDot(GGML_TYPE_Q5_K, wQ.data() + r * 176, x.data());
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(out[r] - ref)));
        maxRef = std::max(maxRef, std::fabs(static_cast<double>(ref)));
    }
    std::cout << "  Q5_K maxAbsDiff=" << maxAbs << " maxRef=" << maxRef << std::endl;
    const double tol = 0.02 * std::max(1.0, maxRef) + 1e-3;
    EXPECT_LT(maxAbs, tol) << "Q5_K SIMD kernel deviates from scalar dot";
}

TEST(DequantizeTest, IQ4XSBatchSIMDVsScalarDot) {
    constexpr uint32_t BLOCK = 256;
    constexpr uint32_t ROWS = 16;
    std::mt19937 rng(202);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    std::vector<uint8_t> wQ(ROWS * 136);
    for (uint32_t r = 0; r < ROWS; ++r) {
        uint8_t ls[8], nib[128];
        for (int i = 0; i < 8; ++i) ls[i] = static_cast<uint8_t>(rng() % 64);
        for (int i = 0; i < 128; ++i) nib[i] = static_cast<uint8_t>(rng() % 256);
        fabricateIQ4XSBlock(wQ.data() + r * 136, 0.01f, ls, nib);
    }

    std::vector<float> x(BLOCK);
    for (uint32_t i = 0; i < BLOCK; ++i) x[i] = dist(rng);
    std::vector<float> out(ROWS, 0.0f);
    bool used = matMulVecBatchIQ4XS_SIMD(wQ.data(), x.data(), 1, ROWS, BLOCK, out.data());
    ASSERT_TRUE(used) << "IQ4_XS SIMD batch kernel not dispatched";

    double maxAbs = 0.0, maxRef = 0.0;
    for (uint32_t r = 0; r < ROWS; ++r) {
        float ref = scalarBlockDot(GGML_TYPE_IQ4_XS, wQ.data() + r * 136, x.data());
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(out[r] - ref)));
        maxRef = std::max(maxRef, std::fabs(static_cast<double>(ref)));
    }
    std::cout << "  IQ4_XS maxAbsDiff=" << maxAbs << " maxRef=" << maxRef << std::endl;
    const double tol = 0.02 * std::max(1.0, maxRef) + 1e-3;
    EXPECT_LT(maxAbs, tol) << "IQ4_XS SIMD kernel deviates from scalar dot";
}

TEST(DequantizeTest, IQ4NLBatchSIMDVsScalarDot) {
    // IQ4_NL blocks are 32-wide, so one row = 8 blocks of 18 B.
    constexpr uint32_t BLOCK = 32;
    constexpr uint32_t BLOCKS_PER_ROW = 8;
    constexpr uint32_t ROWS = 16;
    constexpr uint32_t COLS = BLOCK * BLOCKS_PER_ROW;
    std::mt19937 rng(303);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);

    std::vector<uint8_t> wQ(ROWS * BLOCKS_PER_ROW * 18);
    for (uint32_t r = 0; r < ROWS; ++r) {
        for (uint32_t b = 0; b < BLOCKS_PER_ROW; ++b) {
            uint8_t nib[16];
            for (int i = 0; i < 16; ++i) nib[i] = static_cast<uint8_t>(rng() % 256);
            fabricateIQ4NLBlock(wQ.data() + (r * BLOCKS_PER_ROW + b) * 18,
                                0.02f, nib);
        }
    }

    std::vector<float> x(COLS);
    for (uint32_t i = 0; i < COLS; ++i) x[i] = dist(rng);
    std::vector<float> out(ROWS, 0.0f);
    bool used = matMulVecBatchIQ4NL_SIMD(wQ.data(), x.data(), 1, ROWS, COLS, out.data());
    ASSERT_TRUE(used) << "IQ4_NL SIMD batch kernel not dispatched";

    double maxAbs = 0.0, maxRef = 0.0;
    for (uint32_t r = 0; r < ROWS; ++r) {
        double dot = 0.0;
        for (uint32_t b = 0; b < BLOCKS_PER_ROW; ++b) {
            float blockOut[32];
            GGMLDequantize::dequantizeIQ4_NLBlock(
                    wQ.data() + (r * BLOCKS_PER_ROW + b) * 18, blockOut);
            float *d = blockOut;
            for (uint32_t i = 0; i < 32; ++i) {
                dot += static_cast<double>(x[b * 32 + i]) * d[i];
            }
        }
        float ref = static_cast<float>(dot);
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(out[r] - ref)));
        maxRef = std::max(maxRef, std::fabs(static_cast<double>(ref)));
    }
    std::cout << "  IQ4_NL maxAbsDiff=" << maxAbs << " maxRef=" << maxRef << std::endl;
    const double tol = 0.02 * std::max(1.0, maxRef) + 1e-3;
    EXPECT_LT(maxAbs, tol) << "IQ4_NL SIMD kernel deviates from scalar dot";
}
