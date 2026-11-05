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

#include "EXL3Codec.hpp"

#include <array>
#include <cmath>
#include <cstring>

namespace tinycoder {

    // ---- Tensor-core permutation (mirror of tensor_core_perm in quantize.py) ----
    //
    // Packed index j corresponds to row-major tile element perm[j] (row*16+col).
    // The table is built once and cached.
    static const std::array<uint8_t, 256> &tensorCorePermTable() {
        static const std::array<uint8_t, 256> table = [] {
            std::array<uint8_t, 256> t{};
            for (int tile = 0; tile < 32; ++tile) {
                const int r0 = (tile % 4) * 2;
                const int r1 = r0 + 1;
                const int r2 = r0 + 8;
                const int r3 = r0 + 9;
                const int c0 = tile / 4;
                const int c1 = c0 + 8;
                t[tile * 8 + 0] = static_cast<uint8_t>(r0 * 16 + c0);
                t[tile * 8 + 1] = static_cast<uint8_t>(r1 * 16 + c0);
                t[tile * 8 + 2] = static_cast<uint8_t>(r2 * 16 + c0);
                t[tile * 8 + 3] = static_cast<uint8_t>(r3 * 16 + c0);
                t[tile * 8 + 4] = static_cast<uint8_t>(r0 * 16 + c1);
                t[tile * 8 + 5] = static_cast<uint8_t>(r1 * 16 + c1);
                t[tile * 8 + 6] = static_cast<uint8_t>(r2 * 16 + c1);
                t[tile * 8 + 7] = static_cast<uint8_t>(r3 * 16 + c1);
            }
            return t;
        }();
        return table;
    }

    static const std::array<uint8_t, 256> &tensorCorePermInvTable() {
        static const std::array<uint8_t, 256> inv = [] {
            std::array<uint8_t, 256> t{};
            const auto &perm = tensorCorePermTable();
            for (int j = 0; j < 256; ++j) {
                t[perm[j]] = static_cast<uint8_t>(j);
            }
            return t;
        }();
        return inv;
    }

    const uint8_t *EXL3Codec::tensorCorePerm() {
        return tensorCorePermTable().data();
    }

    const uint8_t *EXL3Codec::tensorCorePermInv() {
        return tensorCorePermInvTable().data();
    }

    // ---- Bit-plane unpack ----
    void EXL3Codec::unpackTile(const uint16_t *packed, uint32_t K, uint16_t *codes) {
        const uint32_t mask = (K >= 16) ? 0xFFFFu : ((1u << K) - 1u);
        // 16*K uint16 == 8*K uint32 words. Reinterpret via memcpy to stay
        // alignment- and aliasing-safe.
        const uint32_t nWords = 8u * K;
        std::array<uint32_t, 128> words{};// max K=8 -> 64 words
        std::memcpy(words.data(), packed, sizeof(uint32_t) * nWords);

        for (uint32_t j = 0; j < 256; ++j) {
            const uint32_t pos = j * K;
            const uint32_t word = pos >> 5;
            const uint32_t within = pos & 31;
            const uint32_t lo = words[word];
            const uint32_t hi = words[(word + 1) % nWords];
            const uint64_t merged = static_cast<uint64_t>(lo) |
                                    (static_cast<uint64_t>(hi) << 32);
            codes[j] = static_cast<uint16_t>((merged >> within) & mask);
        }
    }

    // ---- Half-integer bitrate tile unpack ----
    void EXL3Codec::unpackTileFrac(const uint16_t *packed, uint32_t KA, uint16_t *codes) {
        // 16 * (KA + 0.5) uint16 words per tile == 8 * KA + 4 uint32 words.
        const uint32_t K = KA + 1;          // max bits per code
        const uint32_t nWords = 8u * K;     // upper bound on the uint32 word count
        std::array<uint32_t, 128> words{};
        std::memcpy(words.data(), packed, sizeof(uint32_t) * (8u * KA + 4u));

        // Alternating period-16 mask: code j is (KA+1)-bit when j is even, KA-bit
        // when j is odd (the reference frac pattern 0xAAAA).
        uint32_t bitpos = 0;
        for (uint32_t j = 0; j < 256; ++j) {
            const uint32_t width = (j % 2 == 0) ? (KA + 1u) : KA;
            const uint32_t mask = (width >= 32) ? 0xFFFFFFFFu : ((1u << width) - 1u);
            const uint32_t word = bitpos >> 5;
            const uint32_t within = bitpos & 31;
            const uint32_t lo = words[word];
            const uint32_t hi = words[(word + 1) % nWords];
            const uint64_t merged = static_cast<uint64_t>(lo) |
                                    (static_cast<uint64_t>(hi) << 32);
            codes[j] = static_cast<uint16_t>((merged >> within) & mask);
            bitpos += width;
        }
    }

    // ---- Tile decode ----
    void EXL3Codec::decodeTile(const uint16_t *packed, uint32_t K, int codebook,
                               float *tileRowMajor256) {
        uint16_t codes[256];
        unpackTile(packed, K, codes);
        const uint8_t *perm = tensorCorePerm();
        for (uint32_t j = 0; j < 256; ++j) {
            tileRowMajor256[perm[j]] = decodeCode(codes[j], codebook);
        }
    }

    void EXL3Codec::decodeTileNative(const uint16_t *packed, uint32_t K, int codebook,
                                    float *tileRowMajor256) {
        if (K < 1 || K > 3) {
            decodeTile(packed, K, codebook, tileRowMajor256);
            return;
        }
        // CPU port of the EXL3 reference dq8_aligned_{1,2,3}bit + reconstruct_tile's
        // tensor-core shuffle.  The runtime layout is a per-lane 16-bit window
        // stream, not a straightforward K-bit-per-code stream: each lane decodes
        // 8 overlapping 16-bit windows through the codebook, then the tensor-core
        // permuted halves are shuffled into a row-major 16x16 tile.
        auto fshift = [](uint32_t b, uint32_t a, int sh) -> uint32_t {
            const uint64_t m = (static_cast<uint64_t>(a) << 32) | b;
            return static_cast<uint32_t>(m >> sh);
        };
        auto bfe16 = [](uint32_t x, int off) -> uint32_t {
            return (x >> off) & 0xffffu;
        };
        const uint32_t *p = reinterpret_cast<const uint32_t *>(packed);
        float tile[16][16];
        auto laneFrag = [&](int lane, float *f) {
            const int to = lane * 8;
            uint32_t w[8];
            if (K == 1) {
                const uint32_t i1 = static_cast<uint32_t>(to) >> 5;
                const uint32_t i0 = (i1 + 7u) & 7u;
                uint32_t b = fshift(p[i1], p[i0], (static_cast<uint32_t>(~to)) & 24u);
                w[7] = b & 0xffffu;
                for (int i = 6; i >= 0; --i) w[i] = bfe16(b, 7 - i);
            } else if (K == 2) {
                const uint32_t i1 = static_cast<uint32_t>(to) >> 4;
                const uint32_t i0 = (i1 + 15u) & 15u;
                uint32_t b = fshift(p[i1], p[i0],
                                    ((static_cast<uint32_t>(~to)) & 8u) << 1);
                w[7] = b & 0xffffu;
                for (int i = 6; i >= 0; --i) w[i] = bfe16(b, 14 - 2 * i);
            } else {
                const int b1 = (to + 257) * 3, b2 = b1 + 21;
                const int i0 = (b1 - 16) / 32, i2 = (b2 - 1) / 32;
                const int s2 = (i2 + 1) * 32 - b2;
                const int nw = 3 * 256 / 32;
                const uint32_t a = p[i0 % nw], b = p[i2 % nw];
                w[7] = fshift(b, a, s2);
                w[6] = w[7] >> 3;
                w[5] = w[6] >> 3;
                w[4] = w[5] >> 3;
                w[3] = fshift(b, a, s2 + 12);
                w[2] = w[3] >> 3;
                w[1] = w[2] >> 3;
                w[0] = w[1] >> 3;
            }
            for (int i = 0; i < 8; ++i) {
                f[i] = decodeCode(w[i] & 0xffffu, codebook);
            }
        };
        for (int lane = 0; lane < 32; ++lane) {
            if (lane & 4) continue;
            float f[8], n[8];
            laneFrag(lane, f);
            laneFrag(lane + 4, n);
            const int r0 = (lane % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
            const int c0 = lane / 8, c1 = c0 + 4;
            auto put = [&](int r, int c, float a, float b) {
                tile[r][c] = a;
                tile[r][c + 1] = b;
            };
            put(r0, c0 * 2, f[0], n[0]);
            put(r1, c0 * 2, f[1], n[1]);
            put(r2, c0 * 2, f[2], n[2]);
            put(r3, c0 * 2, f[3], n[3]);
            put(r0, c1 * 2, f[4], n[4]);
            put(r1, c1 * 2, f[5], n[5]);
            put(r2, c1 * 2, f[6], n[6]);
            put(r3, c1 * 2, f[7], n[7]);
        }
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                tileRowMajor256[r * 16 + c] = tile[r][c];
            }
        }
    }

    void EXL3Codec::decodeTileFrac(const uint16_t *packed, uint32_t KA,
                                   float *tileRowMajor256) {
        uint16_t codes[256];
        unpackTileFrac(packed, KA, codes);
        const uint8_t *perm = tensorCorePerm();
        for (uint32_t j = 0; j < 256; ++j) {
            tileRowMajor256[perm[j]] = decodeMul1(codes[j]);
        }
    }

    // ---- Sign bitfield expansion ----
    void EXL3Codec::expandPackedSigns(const uint16_t *packed, size_t groups,
                                      float *out, bool reverse) {
        // Reference (LinearEXL3.unpack_bf): masks = 1 << arange(16); a set bit
        // maps to -1 and a clear bit to +1; bit i of group g corresponds to
        // element g*16 + i.
        for (size_t g = 0; g < groups; ++g) {
            const uint16_t bits = packed[g];
            for (uint32_t i = 0; i < 16; ++i) {
                const uint32_t bit = reverse ? (15u - i) : i;
                const float s = ((bits >> bit) & 1u) ? -1.0f : 1.0f;
                out[g * 16 + i] = s;
            }
        }
    }

    // ---- Normalized Hadamard (Sylvester) ----
    void EXL3Codec::hadamard128(float *v) {
        for (uint32_t len = 1; len < kHad; len <<= 1) {
            for (uint32_t i = 0; i < kHad; i += (len << 1)) {
                for (uint32_t j = 0; j < len; ++j) {
                    const float a = v[i + j];
                    const float b = v[i + j + len];
                    v[i + j] = a + b;
                    v[i + j + len] = a - b;
                }
            }
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(kHad));
        for (uint32_t i = 0; i < kHad; ++i) {
            v[i] *= scale;
        }
    }

    // ---- Reconstruct from a decoded "rotated basis" tile grid ----
    //
    // Shared tail of the reconstruct paths, mirroring the reference
    // LinearEXL3.get_weight_tensor():
    //     w = preapply_had_l_(w, 128)    (left Hadamard over input dims)
    //     w *= suh                        (input-side signs)
    //     w = preapply_had_r_(w, 128)    (right Hadamard over output dims)
    //     w *= svh                        (output-side signs)
    template <typename DecodeTile>
    static void exl3ReconstructFromTiles(const uint16_t *packed, uint32_t tileU16,
                                         uint32_t tilesK, uint32_t tilesN,
                                         const float *suh, const float *svh,
                                         uint32_t k, uint32_t n, float *out,
                                         DecodeTile decodeTileFn) {
        // 1) Decode every tile into the "rotated basis" weight matrix.
        // Tiles are independent; decode them in parallel (the loader is a
        // one-time cost but 27B checkpoints are ~54 GB of F16 output).
#pragma omp parallel for schedule(static)
        for (uint32_t ki = 0; ki < static_cast<int>(tilesK); ++ki) {
            for (uint32_t ni = 0; ni < tilesN; ++ni) {
                const uint16_t *tile =
                        packed + (static_cast<size_t>(ki) * tilesN + ni) * tileU16;
                float tileRM[EXL3Codec::kTileElems];
                decodeTileFn(tile, tileRM);
                for (uint32_t r = 0; r < 16; ++r) {
                    float *dst = out + (static_cast<size_t>(ki) * 16 + r) * n + ni * 16;
                    std::memcpy(dst, tileRM + r * 16, 16 * sizeof(float));
                }
            }
        }

        // 2) Left Hadamard: 128-wide blocks over the k (row) axis, per column.
        //
        // CACHE-BLOCKED.  The straightforward implementation gathers a column
        // of 128 floats (stride n*4 bytes) for EVERY column, then scatters it
        // back: for a 5120x17408 FFN matrix that is ~220 GB of cache-line
        // traffic per matrix and dominated EXL3 checkpoint loading (a 27B
        // checkpoint took >20 minutes, single-threaded).  Loading a
        // [128 rows x 128 cols] tile once into a 64 KB stack buffer (L1/L2
        // resident) and doing the 128 column transforms there moves the same
        // bytes once in each direction and is bit-identical: hadamard128() is
        // unchanged and every column sees the same inputs in the same order.
        {
            constexpr uint32_t kBlk = EXL3Codec::kHad;// 128
#pragma omp parallel for schedule(static)
            for (uint32_t kb = 0; kb < static_cast<int>(k); kb += kBlk) {
                alignas(64) float blk[kBlk * kBlk];
                std::array<float, kBlk> tmp{};
                const uint32_t rows = std::min(kBlk, k - kb);
                for (uint32_t nb = 0; nb < n; nb += kBlk) {
                    const uint32_t cols = std::min(kBlk, n - nb);
                    // Gather the tile (rows are contiguous in `out`).
                    for (uint32_t r = 0; r < rows; ++r) {
                        std::memcpy(blk + static_cast<size_t>(r) * cols,
                                    out + static_cast<size_t>(kb + r) * n + nb,
                                    cols * sizeof(float));
                    }
                    // Transform each column inside the tile.
                    for (uint32_t c = 0; c < cols; ++c) {
                        for (uint32_t r = 0; r < rows; ++r) {
                            tmp[r] = blk[static_cast<size_t>(r) * cols + c];
                        }
                        EXL3Codec::hadamard128(tmp.data());
                        for (uint32_t r = 0; r < rows; ++r) {
                            blk[static_cast<size_t>(r) * cols + c] = tmp[r];
                        }
                    }
                    // Scatter back.
                    for (uint32_t r = 0; r < rows; ++r) {
                        std::memcpy(out + static_cast<size_t>(kb + r) * n + nb,
                                    blk + static_cast<size_t>(r) * cols,
                                    cols * sizeof(float));
                    }
                }
            }
        }

        // 3) suh: scale each row by its input-side sign.
        if (suh) {
#pragma omp parallel for schedule(static)
            for (uint32_t r = 0; r < static_cast<int>(k); ++r) {
                const float s = suh[r];
                float *row = out + static_cast<size_t>(r) * n;
                for (uint32_t c = 0; c < n; ++c) {
                    row[c] *= s;
                }
            }
        }

        // 4) Right Hadamard: 128-wide blocks over the n (col) axis, per row.
        {
            std::array<float, EXL3Codec::kHad> tmp{};
            for (uint32_t r = 0; r < k; ++r) {
                float *row = out + static_cast<size_t>(r) * n;
                for (uint32_t nb = 0; nb < n; nb += EXL3Codec::kHad) {
                    std::memcpy(tmp.data(), row + nb, EXL3Codec::kHad * sizeof(float));
                    EXL3Codec::hadamard128(tmp.data());
                    std::memcpy(row + nb, tmp.data(), EXL3Codec::kHad * sizeof(float));
                }
            }
        }

        // 5) svh: scale each column by its output-side sign.
        if (svh) {
#pragma omp parallel for schedule(static)
            for (uint32_t r = 0; r < static_cast<int>(k); ++r) {
                float *row = out + static_cast<size_t>(r) * n;
                for (uint32_t c = 0; c < n; ++c) {
                    row[c] *= svh[c];
                }
            }
        }
    }

    // ---- Full reconstruct in the original basis ----
    void EXL3Codec::reconstructMatrix(const uint16_t *packed, uint32_t K, int codebook,
                                      const float *suh, const float *svh,
                                      uint32_t k, uint32_t n, float *out) {
        exl3ReconstructFromTiles(packed, 16u * K, k / 16, n / 16, suh, svh, k, n, out,
                                 [K, codebook](const uint16_t *t, float *rm) {
                                     EXL3Codec::decodeTileNative(t, K, codebook, rm);
                                 });
    }

    void EXL3Codec::reconstructMatrixFrac(const uint16_t *packed, uint32_t KA, int,
                                          const float *suh, const float *svh,
                                          uint32_t k, uint32_t n, float *out) {
        // Half-integer bitrate: 16 * (KA + 0.5) == 16*KA + 8 uint16 words per tile.
        exl3ReconstructFromTiles(packed, 16u * KA + 8u, k / 16, n / 16, suh, svh, k, n, out,
                                 [KA](const uint16_t *t, float *rm) {
                                     EXL3Codec::decodeTileFrac(t, KA, rm);
                                 });
    }

}// namespace tinycoder
