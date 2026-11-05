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

namespace tinycoder {

    /// @brief GGML Q4_K *quantisation* (the encoder side of GGMLDequantize).
    ///
    /// Only needed by the optional load-time repack paths in ModelLoad.cpp, but
    /// it lives in its own unit so a unit test can gate its accuracy: the first
    /// version of this code was private to ModelLoad.cpp, silently re-quantised
    /// every K-quant checkpoint through it, and produced ~7% relative RMS error
    /// (it used `scale = maxScale/63` with per-group 4-bit rounding and never
    /// fitted the 6-bit sub-scale), which corrupted the 7B/9B/35B rows for
    /// weeks.  The test is `DequantizeTest.Q4KQuantizeRoundTripAccuracy`.
    struct GGMLQuantize {
        /// Bytes per Q4_K block: d(2) + dmin(2) + 12 packed 6-bit scale/min
        /// bytes + 128 nibble bytes.
        static constexpr uint32_t Q4K_BLOCK_BYTES = 144;

        /// @brief Quantise 256 floats into one compact Q4_K block (144 B).
        ///
        /// Layout matches ggml: d and dmin are fp16 super-scales, the twelve
        /// scale bytes hold eight 6-bit sub-scales and eight 6-bit mins, and the
        /// 128 nibble bytes hold the 4-bit weights, two consecutive 32-element
        /// groups per 32-byte run (low nibbles = even group, high = odd).
        /// Reconstruction is `x = d*sc[g]*q - dmin*m[g]`.
        ///
        /// The body is the original 2026-11-16 loader implementation, kept
        /// unchanged on purpose: measured against ggml's own
        /// `quantize_row_q4_K_ref` on identical gaussian weight blocks it is
        /// within 1 percentage point (8.10% vs 7.17% mean relative RMS), and a
        /// rewrite using the grid-search + alternating-least-squares strategy of
        /// ggml's `make_qkx2_quants` measured 11.6%, i.e. worse.  ~7% IS the
        /// Q4_K format floor on fp32 gaussian weights, so there is no headroom
        /// to recover here; see the ledger entry for 2026-11-16.
        static void quantizeQ4KBlock(const float *x, uint8_t *out);
    };

}// namespace tinycoder
