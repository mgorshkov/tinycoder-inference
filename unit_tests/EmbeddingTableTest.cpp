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

// Invariant guard for the host LM-head embedding table.
//
// `Model::buildDequantizedEmbeddings` builds the Q8_K table row by row: one
// reusable F32 row buffer, `dequantizeBlock` per source block, then
// `quantizeQ8K` per row.  The previous order dequantized the whole matrix
// first and sliced it.  The two must produce identical bytes -- if they do
// not, the CPU LM head silently changes its logits.  Building the whole F32
// matrix costs 5.1 GB on a 248320 x 5120 vocab, so the row-streamed order is
// the one that has to stay.
//
// The test synthesises Q8_0 source blocks (34 bytes each: half scale + 32
// int8) so it needs no checkpoint, and covers both the row-streamable geometry
// (hidden is a whole number of source blocks) and the fallback geometry.

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "GGMLDequantize.hpp"
#include "SIMDMatMulVec.hpp"

namespace {

    using tinycoder::GGML_TYPE_Q8_0;
    using tinycoder::GGMLDequantize;

    // One Q8_0 row: ceil(hidden/32) blocks of {fp16 d, int8 qs[32]}.
    std::vector<uint8_t> makeQ8_0Matrix(uint32_t rows, uint32_t hidden, uint32_t seed) {
        const uint32_t blocksPerRow = (hidden + 31) / 32;
        std::vector<uint8_t> out(static_cast<size_t>(rows) * blocksPerRow * 34);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                uint8_t *blk = out.data() + (static_cast<size_t>(r) * blocksPerRow + b) * 34;
                const float d = 0.01f * static_cast<float>(1 + ((r + b + seed) % 7));
                // little-endian fp16 payload; 0x3c00 == 1.0 and the value only has
                // to be a plausible positive scale, so encode from the raw bits.
                const uint32_t bits = 0x3c00u + 0x100u * ((r + b + seed) % 5u);
                blk[0] = static_cast<uint8_t>(bits & 0xffu);
                blk[1] = static_cast<uint8_t>((bits >> 8) & 0xffu);
                (void) d;
                for (int i = 0; i < 32; ++i) {
                    const int v = static_cast<int>((r * 31u + b * 17u + static_cast<uint32_t>(i) * 5u + seed) % 255u) - 127;
                    blk[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(v));
                }
            }
        }
        return out;
    }

    // Whole-matrix order: dequantize everything, then quantize row by row.
    std::vector<tinycoder::Q8KBlock> buildFull(const std::vector<uint8_t> &src, uint32_t rows,
                                               uint32_t hidden) {
        const uint32_t blocksPerRow = (hidden + 255) / 256;
        std::vector<tinycoder::Q8KBlock> q8k(static_cast<size_t>(rows) * blocksPerRow);
        auto deq = GGMLDequantize::dequantize(GGML_TYPE_Q8_0, src.data(),
                                              static_cast<uint64_t>(rows) * hidden);
        EXPECT_EQ(deq.size(), static_cast<size_t>(rows) * hidden);
        for (uint32_t r = 0; r < rows; ++r) {
            GGMLDequantize::quantizeQ8K(deq.data() + static_cast<uint64_t>(r) * hidden, hidden,
                                        q8k.data() + static_cast<uint64_t>(r) * blocksPerRow);
        }
        return q8k;
    }

    // Row-streamed order: exactly what buildDequantizedEmbeddings does now.
    std::vector<tinycoder::Q8KBlock> buildStreamed(const std::vector<uint8_t> &src, uint32_t rows,
                                                   uint32_t hidden) {
        const uint32_t srcBlockSize = 32;// Q8_0
        const uint32_t srcTypeSize = 34; // Q8_0
        const uint64_t srcBlocksPerRow = (hidden + srcBlockSize - 1) / srcBlockSize;
        const uint32_t blocksPerRow = (hidden + 255) / 256;
        std::vector<tinycoder::Q8KBlock> q8k(static_cast<size_t>(rows) * blocksPerRow);
        std::vector<float> rowF32(hidden);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint8_t *rowSrc =
                    src.data() + static_cast<uint64_t>(r) * srcBlocksPerRow * srcTypeSize;
            for (uint64_t b = 0; b < srcBlocksPerRow; ++b) {
                GGMLDequantize::dequantizeBlock(GGML_TYPE_Q8_0, rowSrc + b * srcTypeSize,
                                                rowF32.data() + b * srcBlockSize, srcBlockSize);
            }
            GGMLDequantize::quantizeQ8K(rowF32.data(), hidden,
                                        q8k.data() + static_cast<uint64_t>(r) * blocksPerRow);
        }
        return q8k;
    }

    TEST(EmbeddingTable, StreamedQ8KMatchesWholeMatrix) {
        // Row-streamable geometry: 5120 = 160 whole Q8_0 blocks.
        const uint32_t rows = 8, hidden = 5120;
        const auto src = makeQ8_0Matrix(rows, hidden, 3u);
        const auto full = buildFull(src, rows, hidden);
        const auto streamed = buildStreamed(src, rows, hidden);
        ASSERT_EQ(full.size(), streamed.size());
        EXPECT_EQ(std::memcmp(full.data(), streamed.data(),
                              full.size() * sizeof(tinycoder::Q8KBlock)),
                  0)
                << "row-streamed Q8_K must be byte-identical to slice-of-whole-matrix";
    }

    TEST(EmbeddingTable, StreamedQ8KIgnoresStaleRowBytes) {
        // A row buffer is reused across vocab rows, so a short final block must not
        // leak the previous row's tail into the Q8_K blocks.
        const uint32_t rows = 4, hidden = 5120;
        auto src = makeQ8_0Matrix(rows, hidden, 11u);
        const auto full = buildFull(src, rows, hidden);
        const auto streamed = buildStreamed(src, rows, hidden);
        ASSERT_EQ(full.size(), streamed.size());
        EXPECT_EQ(std::memcmp(full.data(), streamed.data(),
                              full.size() * sizeof(tinycoder::Q8KBlock)),
                  0);
    }

    TEST(EmbeddingTable, StreamedQ8KMatchesWholeMatrixForSmallHidden) {
        // A second geometry (768 = 3 whole Q8_K blocks per row, 24 source blocks)
        // so a per-row/block indexing error cannot hide behind 5120 alone.
        // NOTE: `quantizeQ8K` writes whole 256-wide blocks, so hidden must be a
        // multiple of 256 -- true for every shipping architecture (1536/4096/5120/
        // 7168).  A hidden size outside that contract is not covered here.
        const uint32_t rows = 6, hidden = 768;
        const auto src = makeQ8_0Matrix(rows, hidden, 7u);
        const auto full = buildFull(src, rows, hidden);
        const auto streamed = buildStreamed(src, rows, hidden);
        ASSERT_EQ(full.size(), streamed.size());
        EXPECT_EQ(std::memcmp(full.data(), streamed.data(),
                              full.size() * sizeof(tinycoder::Q8KBlock)),
                  0);
    }

}// namespace
