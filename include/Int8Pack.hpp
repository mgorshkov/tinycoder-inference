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

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "GGMLDequantize.hpp"
#include "SIMDMatMulVec.hpp"
#include "ThreadPool.hpp"

namespace tinycoder {

    /// @brief int8 tensor-core pack for IQ3_XXS weights (mma.s8.m8n8k16 family).
    ///
    /// Shared foundation for the prefill (int8 tensor-core GEMM) and decode
    /// (mma.s8 GEMV) MMQ paths.  Converts GGML IQ3_XXS blocks (256 weights,
    /// 98 B each) into a signed-int8 weight array + per-32-group float scales,
    /// and provides the mma.s8.m16n8k16 B-fragment layout the tensor-core
    /// kernels consume.
    ///
    /// IQ3_XXS dequantization (per 256-weight block):
    ///   w[i] = db[g(i)] * grid_value[i] * sign[i]
    /// where db[g] = d * (0.5 + scale6[g]) * 0.5 is the per-32-group scale.
    /// The int8 pack stores qs[i] = grid_value[i] * sign[i] (range [-15, 15])
    /// plus the 8 group scales, so w[i] = db[g] * qs[i] EXACTLY (the pack is a
    /// lossless re-encoding: unpacking reproduces dequantizeIQ3_XXSBlock
    /// bit-for-bit, verified by Int8PackParityTest).
    ///
    /// The mma.s8.m8n8k16 B-fragment layout (packMma) targets the GEMM
    /// y = x @ W^T (C[M x N] = A[M x K] @ B[K x N] with B[k][n] = W[n][k]):
    /// for each (n8-tile, k16-tile) the 128 int8 weights are stored in the
    /// exact per-thread order the PTX mma instruction reads them (B: 4 int8
    /// per lane, k = (lane%4)*4 + i, n = lane/4), so a warp can load a whole
    /// B fragment with one coalesced 128-byte transaction.  The B mapping is
    /// shared by m8n8k16 (Turing: the only int8 tensor-core shape) and
    /// m16n8k16 (Ampere+), so one pack feeds both.
    struct Int8Pack {
        // ---- IQ3_XXS source geometry (GGML) ----
        static constexpr uint32_t BLOCK_SIZE = 256;                          // weights per block
        static constexpr uint32_t GROUP_SIZE = 32;                           // weights per scale group
        static constexpr uint32_t GROUPS_PER_BLOCK = BLOCK_SIZE / GROUP_SIZE;// 8
        static constexpr uint32_t SRC_BLOCK_BYTES = 98;                      // d(2) + qs[96]

        // ---- Q4_K source geometry (GGML): d(2) + dmin(2) + scales[12] + qs[128]
        static constexpr uint32_t Q4K_SRC_BLOCK_BYTES = 144;

        // ---- mma.s8.m8n8k16 geometry ----
        // Turing (sm_75) has no mma.m16n8k16 for int8 (that needs sm_80+); the
        // m8n8k16 shape IS available on Turing and Ampere+, so the pack targets
        // it.  m16n8k16 shares the identical B fragment (k=(lane%4)*4+i,
        // n=lane/4) and only widens M (and hence the A / C-D fragments), so the
        // same packed weights feed both instructions.
        static constexpr uint32_t MMA_M = 8;
        static constexpr uint32_t MMA_N = 8;
        static constexpr uint32_t MMA_K = 16;
        static constexpr uint32_t MMA_THREADS = 32;
        static constexpr uint32_t MMA_ELEMS_PER_THREAD = 4;// B: 4 int8 / lane
        static constexpr uint32_t MMA_ACC_PER_THREAD = 2;  // C/D: 2 s32 / lane
        static constexpr uint32_t MMA_A_PER_THREAD = 4;    // A: 4 int8 / lane
        static constexpr uint32_t MMA_TILE_BYTES =
                MMA_THREADS * MMA_ELEMS_PER_THREAD;// 128

        /// @brief One packed IQ3_XXS block: 256 signed int8 weights + 8 group
        /// scales.  This is the canonical int8 form both MMQ paths consume.
        struct PackedBlock {
            int8_t qs[BLOCK_SIZE];         // signed grid values, weight order
            float scales[GROUPS_PER_BLOCK];// per-32-group weight scale db
        };
        static_assert(sizeof(PackedBlock) ==
                              BLOCK_SIZE + GROUPS_PER_BLOCK * sizeof(float),
                      "PackedBlock must be 256 int8 + 8 float");

        /// @brief Pack one GGML IQ3_XXS block (98 bytes) into the int8 form.
        static void packBlock(const uint8_t *src, PackedBlock &dst);

        /// @brief Pack numBlocks contiguous IQ3_XXS blocks (thread-pool
        /// parallel; one-time cost at upload, not in the inference hot path).
        static void pack(const uint8_t *src, uint32_t numBlocks,
                         PackedBlock *dst);

        /// @brief Reconstruct float weights from a packed block.
        /// Bit-exact with GGMLDequantize::dequantizeIQ3_XXSBlock.
        static void unpackBlock(const PackedBlock &src, float *out);

        /// @brief Reference dot: packed int8 weights x Q8_K-quantized
        /// activation.  Matches GGMLDequantize::dotProductIQ3_XXS_Q8K up to
        /// float rounding (per-group scale multiply vs the reference's d*ls
        /// fold; the parity harness checks this with a tight tolerance).
        static float dotBlockQ8K(const PackedBlock &w, const Q8KBlock &x);

        /// @brief Reference dot: packed int8 weights x float activation
        /// (the true weight dot, no activation quantization).
        static float dotBlockFloat(const PackedBlock &w, const float *x);

        // ---- mma.s8.m8n8k16 B-fragment layout ----
        /// @brief Pack a full weight matrix [rows x cols] into the mma
        /// B-fragment layout: for each (n8-tile, k16-tile), 32 lanes x 4 int8
        /// in the exact PTX mma.m8n8k16 / mma.m16n8k16 (s8) order -- the 4
        /// int8 of lane L are W[n0 + (L>>2)][k0 + (L&3)*4 + i], i = 0..3.
        /// @param src IQ3_XXS block bytes (rows * blocksPerRow * 98)
        /// @param rows, cols matrix geometry (rows padded to 8, cols to 16;
        ///        padding weights are zeroed)
        /// @param bOut [n8Tiles * k16Tiles * 128] int8
        /// @param scaleOut [n8Tiles * k16Tiles * MMA_N] floats: per
        ///        (n8-tile, k16-tile, n-row) scale = the db of the 32-group
        ///        covering column kt*16 of WEIGHT ROW (n0+nrow).  The scale is
        ///        per-row because every row of W is its own IQ3_XXS block
        ///        sequence with its own d/scales; a k16-tile never straddles
        ///        two 32-groups within a row.
        static void packMma(const uint8_t *src, uint32_t rows, uint32_t cols,
                            int8_t *bOut, float *scaleOut);

        /// @brief Unpack the mma B-fragment layout back to float weights
        /// [rows x cols] (parity harness / host reference).
        static void unpackMma(const int8_t *b, const float *scales,
                              uint32_t rows, uint32_t cols, float *out);

        /// @brief Reference dot on the mma layout: y[n] = sum_k W[n][k] * x[k]
        /// (reads the B fragments in lane order, proving the layout mapping).
        static void dotMma(const int8_t *b, const float *scales, uint32_t rows,
                           uint32_t cols, const float *x, float *y);

        /// @brief Host emulation of ONE mma.s8.m8n8k16 instruction on a packed
        /// B cell, built strictly from the PTX fragment positions (not from the
        /// pack's own assumption): A[m][k] with m = lane>>2, k = (lane&3)*4+i;
        /// B[k][n] with k = (lane&3)*4+i, n = lane>>2, and D = A @ B.  This is
        /// the ground-truth check that the packed B cell really IS the
        /// instruction's B fragment (a self-consistent but wrong mapping passes
        /// packMma/unpackMma/dotMma yet fails here).
        /// @param aTile logical A, 8 rows x 16 cols, row-major int8
        /// @param bCell 128 int8 B cell from packMma
        /// @param dOut  logical D[8][8] = sum_k aTile[m][k] * bCellFrag[k][n]
        static void emulateMmaCell(const int8_t *aTile, const int8_t *bCell,
                                   int32_t dOut[8][8]);

        // ---- Q4_K (affine) mma pack (2026-10-29) ----
        // Q4_K dequantizes AFFINELY: w = d*sc*q - dmin*m (q in [0,15]).  The
        // pure-scale mma path common to IQ3_XXS/Q6_K folds the per-row scale
        // into an integer mma and needs no extra term, but the Q4_K min term
        // is a per-(row, 32-block) constant that multiplies the activation
        // block-sum, so it lives OUTSIDE the tensor core.  This pack therefore
        // emits, per (n8-tile, k16-tile):
        //   qOut   : 128 int8 = the 4-bit q values (0..15), same B-fragment
        //            layout as packMma (lane L byte i = W[n0+L/4][k0+(L&3)*4+i]);
        //   dscOut : per n-row (8) float = d*sc of the 32-block covering the
        //            k16 tile's columns;
        //   dmnOut : per n-row (8) float = dmin*m of that same 32-block.
        // The kernel computes  sum_k q*x  on the tensor core, then
        //   y[n] += ad * ( dsc * D - dmn * bsum32 )
        // where ad is the Q8_K activation scale and bsum32 the activation
        // 32-group sum.  This re-quantizes the Q4_K weights 1:1 (no extra error
        // from the affine fit -- the q are exact 4-bit values), so it matches
        // kQGemmKxQ8K<Q4_K> up to the same float-accumulation class.
        static void packMmaQ4K(const uint8_t *src, uint32_t rows, uint32_t cols,
                               int8_t *qOut, float *dscOut, float *dmnOut);

    private:
        /// @brief int8 weight at matrix position (n, k) from packed blocks
        /// (0 for mma padding outside [rows x cols]).
        static int8_t weightAt(const PackedBlock *blocks, uint32_t blocksPerRow,
                               uint32_t n, uint32_t k, uint32_t rows,
                               uint32_t cols);

        /// @brief Q4_K 4-bit q (0..15) at (n, k); 0 outside [rows x cols].
        static uint8_t q4kWeightAt(const uint8_t *src, uint32_t blocksPerRow,
                                   uint32_t n, uint32_t k, uint32_t rows,
                                   uint32_t cols);

        /// @brief Q4_K (d*sc, dmin*m) of the 32-sub-block covering column
        /// `k16tile*16` of weight row n.  Returns (0,0) outside [rows x cols].
        static void q4kScaleAt(const uint8_t *src, uint32_t blocksPerRow,
                               uint32_t n, uint32_t k16tile, uint32_t rows,
                               float &dscOut, float &dmnOut);
    };

    // -----------------------------------------------------------------------
    // Implementation
    // -----------------------------------------------------------------------

    inline void Int8Pack::packBlock(const uint8_t *src, PackedBlock &dst) {
        const float d = GGMLDequantize::halfToFloatBranchFree(
                *reinterpret_cast<const uint16_t *>(src));
        const uint8_t *q3 = src + 2;
        const uint8_t *gas = q3 + BLOCK_SIZE / 4;// scales_and_signs at qs+64

        for (uint32_t g = 0; g < GROUPS_PER_BLOCK; ++g) {
            uint32_t aux32;
            std::memcpy(&aux32, gas + 4u * g, sizeof(uint32_t));
            const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
            dst.scales[g] = db;
            for (uint32_t l = 0; l < 4; ++l) {
                const uint8_t signs =
                        GGMLDequantize::ksigns_iq2xs[(aux32 >> (7u * l)) & 127u];
                const uint8_t *grid1 = reinterpret_cast<const uint8_t *>(
                        &GGMLDequantize::IQ3XXS_GRID[q3[8u * g + 2u * l]]);
                const uint8_t *grid2 = reinterpret_cast<const uint8_t *>(
                        &GGMLDequantize::IQ3XXS_GRID[q3[8u * g + 2u * l + 1u]]);
                for (uint32_t j = 0; j < 4; ++j) {
                    const int s1 =
                            (signs & GGMLDequantize::kmask_iq2xs[j + 0u]) ? -1 : 1;
                    const int s2 =
                            (signs & GGMLDequantize::kmask_iq2xs[j + 4u]) ? -1 : 1;
                    dst.qs[g * GROUP_SIZE + l * 8u + j + 0u] =
                            static_cast<int8_t>(static_cast<int>(grid1[j]) * s1);
                    dst.qs[g * GROUP_SIZE + l * 8u + j + 4u] =
                            static_cast<int8_t>(static_cast<int>(grid2[j]) * s2);
                }
            }
        }
    }

    inline void Int8Pack::pack(const uint8_t *src, uint32_t numBlocks,
                               PackedBlock *dst) {
        ThreadPool::instance().parallelFor(0, numBlocks, [&](uint32_t b) {
            packBlock(src + static_cast<size_t>(b) * SRC_BLOCK_BYTES, dst[b]);
        });
    }

    inline void Int8Pack::unpackBlock(const PackedBlock &src, float *out) {
        for (uint32_t g = 0; g < GROUPS_PER_BLOCK; ++g) {
            const float db = src.scales[g];
            for (uint32_t i = 0; i < GROUP_SIZE; ++i) {
                out[g * GROUP_SIZE + i] =
                        db * static_cast<float>(src.qs[g * GROUP_SIZE + i]);
            }
        }
    }

    inline float Int8Pack::dotBlockQ8K(const PackedBlock &w, const Q8KBlock &x) {
        // Reference (dotProductIQ3_XXS_Q8K): (d * xd) * sum_g(ls_g * sumi_g)
        // with d * ls_g == 4 * db_g.  The packed form multiplies each group's
        // integer dot by its own db in float, then applies the constant 4 and
        // the activation scale -- the same math, different rounding
        // (tolerance-checked by the parity harness).
        float acc = 0.0f;
        for (uint32_t g = 0; g < GROUPS_PER_BLOCK; ++g) {
            int32_t sumi = 0;
            const int8_t *wq = w.qs + g * GROUP_SIZE;
            const int8_t *xq = x.qs + g * GROUP_SIZE;
            for (uint32_t i = 0; i < GROUP_SIZE; ++i) {
                sumi += static_cast<int32_t>(wq[i]) *
                        static_cast<int32_t>(xq[i]);
            }
            acc += w.scales[g] * static_cast<float>(sumi);
        }
        return 4.0f * x.d * acc;
    }

    inline float Int8Pack::dotBlockFloat(const PackedBlock &w, const float *x) {
        float acc = 0.0f;
        for (uint32_t g = 0; g < GROUPS_PER_BLOCK; ++g) {
            const float db = w.scales[g];
            const int8_t *wq = w.qs + g * GROUP_SIZE;
            for (uint32_t i = 0; i < GROUP_SIZE; ++i) {
                acc += db * static_cast<float>(wq[i]) * x[g * GROUP_SIZE + i];
            }
        }
        return acc;
    }

    inline int8_t Int8Pack::weightAt(const PackedBlock *blocks,
                                     uint32_t blocksPerRow, uint32_t n,
                                     uint32_t k, uint32_t rows, uint32_t cols) {
        if (n >= rows || k >= cols) return 0;// mma padding
        const uint32_t block = n * blocksPerRow + k / BLOCK_SIZE;
        return blocks[block].qs[k % BLOCK_SIZE];
    }

    inline void Int8Pack::packMma(const uint8_t *src, uint32_t rows,
                                  uint32_t cols, int8_t *bOut,
                                  float *scaleOut) {
        const uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const uint32_t numBlocks = rows * blocksPerRow;
        std::vector<PackedBlock> packed(numBlocks);
        pack(src, numBlocks, packed.data());

        const uint32_t n8Tiles = (rows + MMA_N - 1) / MMA_N;
        const uint32_t k16Tiles = (cols + MMA_K - 1) / MMA_K;

        // Per-(n8-tile, k16-tile, n-row) scale: the db of the 32-group covering
        // column kt*16 of WEIGHT ROW (n0+nrow).  Every row of W is its own
        // IQ3_XXS block sequence with its own d/scales, so the scale is
        // per-row; a k16-tile spans 16 consecutive columns, always inside one
        // 32-group.
        for (uint32_t nt = 0; nt < n8Tiles; ++nt) {
            const uint32_t n0 = nt * MMA_N;
            for (uint32_t kt = 0; kt < k16Tiles; ++kt) {
                const uint32_t c = kt * MMA_K;
                const uint32_t group = (c % BLOCK_SIZE) / GROUP_SIZE;
                for (uint32_t nr = 0; nr < MMA_N; ++nr) {
                    const uint32_t n = n0 + nr;
                    if (n >= rows) {
                        scaleOut[(nt * k16Tiles + kt) * MMA_N + nr] = 0.0f;
                        continue;
                    }
                    const uint32_t block = n * blocksPerRow + c / BLOCK_SIZE;
                    scaleOut[(nt * k16Tiles + kt) * MMA_N + nr] =
                            packed[block].scales[group];
                }
            }
        }

        // B fragments: for each (nt, kt) tile, 32 lanes x 4 int8 in the exact
        // PTX mma.m8n8k16 / mma.m16n8k16 (s8) B-fragment order:
        //   k-row = (lane % 4) * 4 + i,  n-col = lane / 4,  i = 0..3
        // i.e. lane L holds W[n0 + (L>>2)][k0 + (L&3)*4 + i].  With B[k][n] =
        // W[n][k] this is the col-major B for the row-major GEMM y = x @ W^T.
        ThreadPool::instance().parallelFor(0, n8Tiles * k16Tiles, [&](uint32_t t) {
            const uint32_t nt = t / k16Tiles;
            const uint32_t kt = t % k16Tiles;
            const uint32_t n0 = nt * MMA_N;
            const uint32_t k0 = kt * MMA_K;
            int8_t *tile = bOut + static_cast<size_t>(t) * MMA_TILE_BYTES;
            for (uint32_t lane = 0; lane < MMA_THREADS; ++lane) {
                const uint32_t n = n0 + (lane >> 2u);
                const uint32_t kBase = k0 + (lane & 3u) * 4u;
                for (uint32_t i = 0; i < 4; ++i) {
                    tile[lane * 4u + i] = weightAt(packed.data(), blocksPerRow,
                                                   n, kBase + i, rows, cols);
                }
            }
        });
    }

    inline void Int8Pack::unpackMma(const int8_t *b, const float *scales,
                                    uint32_t rows, uint32_t cols, float *out) {
        const uint32_t n8Tiles = (rows + MMA_N - 1) / MMA_N;
        const uint32_t k16Tiles = (cols + MMA_K - 1) / MMA_K;
        std::memset(out, 0, static_cast<size_t>(rows) * cols * sizeof(float));
        for (uint32_t nt = 0; nt < n8Tiles; ++nt) {
            const uint32_t n0 = nt * MMA_N;
            for (uint32_t kt = 0; kt < k16Tiles; ++kt) {
                const uint32_t k0 = kt * MMA_K;
                // Per-row scale: dbRow[nrow] is the db of WEIGHT ROW
                // (n0+nrow)'s 32-group covering column k0.
                const float *dbRow = scales + (nt * k16Tiles + kt) * MMA_N;
                const int8_t *tile =
                        b + (static_cast<size_t>(nt) * k16Tiles + kt) *
                                    MMA_TILE_BYTES;
                for (uint32_t lane = 0; lane < MMA_THREADS; ++lane) {
                    const uint32_t n = n0 + (lane >> 2u);
                    if (n >= rows) continue;
                    const uint32_t kBase = k0 + (lane & 3u) * 4u;
                    const float db = dbRow[lane >> 2u];
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint32_t k = kBase + i;
                        if (k >= cols) continue;
                        out[static_cast<size_t>(n) * cols + k] =
                                db * static_cast<float>(tile[lane * 4u + i]);
                    }
                }
            }
        }
    }

    inline void Int8Pack::dotMma(const int8_t *b, const float *scales,
                                 uint32_t rows, uint32_t cols, const float *x,
                                 float *y) {
        const uint32_t n8Tiles = (rows + MMA_N - 1) / MMA_N;
        const uint32_t k16Tiles = (cols + MMA_K - 1) / MMA_K;
        std::fill(y, y + rows, 0.0f);
        for (uint32_t nt = 0; nt < n8Tiles; ++nt) {
            const uint32_t n0 = nt * MMA_N;
            for (uint32_t kt = 0; kt < k16Tiles; ++kt) {
                const uint32_t k0 = kt * MMA_K;
                // Per-row scale (see unpackMma): dbRow[nrow] for WEIGHT ROW
                // (n0+nrow).
                const float *dbRow = scales + (nt * k16Tiles + kt) * MMA_N;
                const int8_t *tile =
                        b + (static_cast<size_t>(nt) * k16Tiles + kt) *
                                    MMA_TILE_BYTES;
                for (uint32_t lane = 0; lane < MMA_THREADS; ++lane) {
                    const uint32_t n = n0 + (lane >> 2u);
                    if (n >= rows) continue;
                    const uint32_t kBase = k0 + (lane & 3u) * 4u;
                    const float db = dbRow[lane >> 2u];
                    float acc = 0.0f;
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint32_t k = kBase + i;
                        if (k < cols)
                            acc += static_cast<float>(tile[lane * 4u + i]) *
                                   x[k];
                    }
                    y[n] += db * acc;
                }
            }
        }
    }

    inline void Int8Pack::emulateMmaCell(const int8_t *aTile, const int8_t *bCell,
                                         int32_t dOut[8][8]) {
        // Rebuild the logical A[8][16] and B[16][8] from the fragment
        // positions the PTX instruction actually reads (B is the *packed* cell,
        // so a wrong pack layout corrupts B here and the D comparison fails).
        int8_t bFrag[16][8];
        for (uint32_t lane = 0; lane < 32; ++lane) {
            const uint32_t n = lane >> 2u;
            const uint32_t kBase = (lane & 3u) * 4u;
            for (uint32_t i = 0; i < 4; ++i)
                bFrag[kBase + i][n] = bCell[lane * 4u + i];
        }
        for (uint32_t m = 0; m < 8; ++m) {
            for (uint32_t n = 0; n < 8; ++n) {
                int32_t acc = 0;
                for (uint32_t k = 0; k < 16; ++k)
                    acc += static_cast<int32_t>(aTile[m * 16u + k]) *
                           static_cast<int32_t>(bFrag[k][n]);
                dOut[m][n] = acc;
            }
        }
    }

    // ---- Q4_K (affine) mma pack ----
    inline uint8_t Int8Pack::q4kWeightAt(const uint8_t *src,
                                         uint32_t blocksPerRow, uint32_t n,
                                         uint32_t k, uint32_t rows,
                                         uint32_t cols) {
        if (n >= rows || k >= cols) return 0;
        const uint8_t *blk =
                src + (static_cast<size_t>(n) * blocksPerRow + (k / BLOCK_SIZE)) *
                              Q4K_SRC_BLOCK_BYTES;
        const uint32_t c = k % BLOCK_SIZE;   // 0..255
        const uint32_t i = c / GROUP_SIZE;   // sub-block 0..7
        const uint32_t j = c % GROUP_SIZE;   // 0..31
        const uint8_t *qsv = blk + 16;
        const uint8_t qb = qsv[32u * (i >> 1u) + j];
        return ((i & 1u) == 0u) ? (qb & 0xFu) : (qb >> 4);
    }

    inline void Int8Pack::q4kScaleAt(const uint8_t *src, uint32_t blocksPerRow,
                                     uint32_t n, uint32_t k16tile, uint32_t rows,
                                     float &dscOut, float &dmnOut) {
        dscOut = 0.0f;
        dmnOut = 0.0f;
        if (n >= rows) return;
        const uint32_t c = k16tile * MMA_K;   // a k16 tile stays inside a 32-group
        const uint8_t *blk =
                src + (static_cast<size_t>(n) * blocksPerRow + (c / BLOCK_SIZE)) *
                              Q4K_SRC_BLOCK_BYTES;
        const float d =
                GGMLDequantize::halfToFloatBranchFree(*reinterpret_cast<const uint16_t *>(blk));
        const float dmin = GGMLDequantize::halfToFloatBranchFree(
                *reinterpret_cast<const uint16_t *>(blk + 2));
        const uint8_t *scales = blk + 4;
        const uint32_t i = (c % BLOCK_SIZE) / GROUP_SIZE;
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
        dscOut = d * static_cast<float>(sc);
        dmnOut = dmin * static_cast<float>(m);
    }

    inline void Int8Pack::packMmaQ4K(const uint8_t *src, uint32_t rows,
                                     uint32_t cols, int8_t *qOut,
                                     float *dscOut, float *dmnOut) {
        const uint32_t n8Tiles = (rows + MMA_N - 1) / MMA_N;
        const uint32_t k16Tiles = (cols + MMA_K - 1) / MMA_K;
        const uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Per (n8, k16) tile: the q B-cell, plus the per-row dsc/dmn.
        ThreadPool::instance().parallelFor(0, n8Tiles * k16Tiles, [&](uint32_t t) {
            const uint32_t nt = t / k16Tiles;
            const uint32_t kt = t % k16Tiles;
            const uint32_t n0 = nt * MMA_N;
            const uint32_t k0 = kt * MMA_K;
            int8_t *tile = qOut + static_cast<size_t>(t) * MMA_TILE_BYTES;
            for (uint32_t lane = 0; lane < MMA_THREADS; ++lane) {
                const uint32_t n = n0 + (lane >> 2u);
                const uint32_t kBase = k0 + (lane & 3u) * 4u;
                for (uint32_t i = 0; i < 4; ++i) {
                    tile[lane * 4u + i] = static_cast<int8_t>(
                            q4kWeightAt(src, blocksPerRow, n, kBase + i, rows,
                                        cols));
                }
            }
            for (uint32_t nr = 0; nr < MMA_N; ++nr) {
                float dsc, dmn;
                q4kScaleAt(src, blocksPerRow, n0 + nr, kt, rows, dsc, dmn);
                dscOut[t * MMA_N + nr] = dsc;
                dmnOut[t * MMA_N + nr] = dmn;
            }
        });
    }

}// namespace tinycoder
