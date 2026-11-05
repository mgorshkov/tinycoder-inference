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

#include <cstdint>
#include <vector>

namespace tinycoder {

    /// @brief EXL3 (EXL3) trellis codec — CPU reference.
    ///
    /// EXL3 stores each linear as a 3-D trellis tensor of packed 16x16 tiles
    /// [k/16, n/16, 16*K] (uint16), where K is the bitrate (integer 1..8, or a
    /// half-integer 1.5/2.5/3.5 with the mul1 codebook). Each 256-weight tile
    /// holds 256 K-bit codes packed contiguously (little-endian bit order,
    /// wrapped within the tile). A code is expanded to a weight by a procedural
    /// codebook (3INST / mcg / mul1).
    ///
    /// The tile's codes are in the "tensor core" permuted order (see
    /// tensor_core_perm in the reference quantizer); element `j` of the packed
    /// code array holds the row-major tile element `tensorCorePerm()[j]`.
    /// Reconstructing a weight therefore: unpack codes -> codebook-expand ->
    /// un-permute to a row-major 16x16 tile -> assemble the [k, n] matrix in the
    /// rotated basis -> apply the fused suh/H128/W_hat/H128/svh transform to
    /// recover the original basis.
    ///
    /// This class is the correctness reference; the CUDA kernels in
    /// EXL3Kernels.cuh mirror it bit-for-bit (modulo half-vs-float rounding in
    /// the codebook FMA).
    struct EXL3Codec {
        static constexpr uint32_t kTile = 16;    // tile edge (16x16 = 256 weights)
        static constexpr uint32_t kTileElems = 256;
        static constexpr uint32_t kHad = 128;    // Hadamard block size

        /// @brief Codebook selector stored next to a trellis.
        enum Codebook : int {
            CB_3INST = 0,// default procedural codebook
            CB_MCG = 1,  // marker "mcg"
            CB_MUL1 = 2, // marker "mul1" (required for half-integer bitrates)
        };

        // ---- IEEE-754 half conversion (matches GGMLDequantize / the CUDA path) ----
        static float halfToFloat(uint16_t h) {
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f32;
            if (exp == 0) {
                if (mant == 0) {
                    f32 = sign << 31;
                } else {
                    int n = 0;
                    while ((mant & 0x200) == 0 && n < 10) {
                        mant <<= 1;
                        ++n;
                    }
                    mant &= 0x3FF;
                    f32 = (sign << 31) | ((112 - n) << 23) | ((mant - 512) << 14);
                }
            } else if (exp == 31) {
                f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
            } else {
                f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            }
            float out;
            for (size_t i = 0; i < sizeof(float); ++i) {
                reinterpret_cast<uint8_t *>(&out)[i] =
                        reinterpret_cast<const uint8_t *>(&f32)[i];
            }
            return out;
        }

        // ---- Codebook decode (a single K-bit code -> weight) ----

        /// @brief 3INST: x*89226354 + 64248484, lop3-fold to fp16 patterns, add
        /// the two halves.
        static float decode3INST(uint32_t x) {
            x = x * 89226354u + 64248484u;
            x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
            return halfToFloat(static_cast<uint16_t>(x & 0xffff)) +
                   halfToFloat(static_cast<uint16_t>(x >> 16));
        }

        /// @brief mcg: x*0xCBAC1FED, lop3-fold, add the two halves.
        static float decodeMcG(uint32_t x) {
            x = x * 0xCBAC1FEDu;
            x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
            return halfToFloat(static_cast<uint16_t>(x & 0xffff)) +
                   halfToFloat(static_cast<uint16_t>(x >> 16));
        }

        /// @brief mul1: x*0x83DCD12D, byte-sum + 0x6400 as an fp16 pattern, then
        /// hfma with the two locked constants.
        static float decodeMul1(uint32_t x) {
            x = x * 0x83DCD12Du;
            uint32_t sum = 0x6400u + (x & 0xff) + ((x >> 8) & 0xff) +
                           ((x >> 16) & 0xff) + ((x >> 24) & 0xff);
            const float kInv = halfToFloat(0x1eee);
            const float kBias = halfToFloat(0xc931);
            return halfToFloat(static_cast<uint16_t>(sum)) * kInv + kBias;
        }

        static float decodeCode(uint32_t code, int codebook) {
            switch (codebook) {
                case CB_MCG: return decodeMcG(code);
                case CB_MUL1: return decodeMul1(code);
                default: return decode3INST(code);
            }
        }

        /// @brief The 16x16 tensor-core permutation: packed index -> row-major
        /// tile index (row*16 + col). Mirrors tensor_core_perm() in quantize.py.
        static const uint8_t *tensorCorePerm();
        /// @brief Inverse permutation (row-major -> packed index).
        static const uint8_t *tensorCorePermInv();

        /// @brief Unpack one 256-weight tile's K-bit codes from `packed`
        /// (16*K uint16) into codes[256]. Integer K (1..8) only.
        static void unpackTile(const uint16_t *packed, uint32_t K, uint16_t *codes);

        /// @brief Unpack one tile at a half-integer bitrate KA+0.5 (mul1 only).
        /// `packed` holds 16*K uint16 (8*K uint32) words; code j takes KA+1 bits
        /// when j is even and KA bits when j is odd (the reference 0xAAAA period).
        static void unpackTileFrac(const uint16_t *packed, uint32_t KA, uint16_t *codes);

        /// @brief Decode one packed tile with the EXL3 runtime bit layout.
        ///
        /// This is a CPU port of the EXL3 reference dq8_aligned_{1,2,3}bit +
        /// reconstruct_tile shuffle: the packed stream is NOT a plain LSB-first
        /// K-bit stream (see the implementation comment).  K = 1..3 is
        /// supported natively; other bitrates fall back to decodeTile().
        static void decodeTileNative(const uint16_t *packed, uint32_t K, int codebook,
                                     float *tileRowMajor256);

        /// @brief Decode one packed tile into a row-major 16x16 float tile
        /// (integer K).
        static void decodeTile(const uint16_t *packed, uint32_t K, int codebook,
                               float *tileRowMajor256);

        /// @brief Decode one packed tile at a half-integer bitrate (mul1 only).
        static void decodeTileFrac(const uint16_t *packed, uint32_t KA,
                                   float *tileRowMajor256);

        /// @brief Expand a packed ±1 bitfield (su/sv) into `out`. `groups`
        /// uint16 words cover groups*16 elements; bit i of a group maps to
        /// element i (the reference pack_signs/unpack_bf convention) and a SET
        /// bit is -1 (0 → +1). Pass reverse=true to flip the within-group order
        /// (debug hedge only).
        static void expandPackedSigns(const uint16_t *packed, size_t groups,
                                      float *out, bool reverse = false);

        /// @brief Number of packed uint16 sign words covering `count` elements.
        static size_t packedSignWords(size_t count) { return (count + 15) / 16; }

        /// @brief Normalized Hadamard transform of a 128-element vector,
        /// in place (unnormalized FWHT followed by a 1/sqrt(128) scale).
        static void hadamard128(float *v);

        /// @brief Reconstruct a full EXL3 linear in the ORIGINAL basis.
        ///
        /// Mirrors LinearEXL3.get_weight_tensor(): decode the rotated-basis
        /// weight, then left-H128 -> *suh -> right-H128 -> *svh (see exl3.py).
        /// The result is row-major [k][n] float.
        ///
        /// @param packed   Trellis tensor: [k/16, n/16, 16*K] uint16.
        /// @param K        Bitrate (integer 1..8).
        /// @param codebook Codebook selector.
        /// @param suh      Per-row (k) sign vector (±1), or nullptr for identity.
        /// @param svh      Per-col (n) sign vector (±1), or nullptr for identity.
        /// @param k        in_features (rows of the [k, n] weight).
        /// @param n        out_features (cols of the [k, n] weight).
        /// @param out      Output buffer, row-major [k][n] float.
        static void reconstructMatrix(const uint16_t *packed, uint32_t K, int codebook,
                                      const float *suh, const float *svh,
                                      uint32_t k, uint32_t n, float *out);

        /// @brief reconstructMatrix() for a half-integer bitrate (mul1 only).
        /// @param KA Integer part of the bitrate (fractional part is 0.5).
        static void reconstructMatrixFrac(const uint16_t *packed, uint32_t KA, int codebook,
                                          const float *suh, const float *svh,
                                          uint32_t k, uint32_t n, float *out);

        /// @brief True when K is an integer in [1, 8].
        static bool isIntegerK(float K) {
            return K >= 1.0f && K <= 8.0f && (K - static_cast<int>(K)) == 0.0f;
        }
    };

}// namespace tinycoder
