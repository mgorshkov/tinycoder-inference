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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// ============================================================================
// Prism "Hadamard fold" support (Ternary-Bonsai-2-27B PQ2_0).
//
// Prism stores the linear weights of a model already folded with a per-input-
// dimension rotation, so that the (near-ternary) weight matrices are better
// conditioned.  A folded weight row was produced as
//
//     W_folded = W @ H        (H applied to the input/feature axis)
//
// where H is the normalized Sylvester-Walsh-Hadamard matrix
//
//     H[r][c] = (-1)^popcount(r & c) / sqrt(N)
//
// applied independently to every contiguous block of N = 1024 features, with an
// optional per-feature sign flip S applied first:
//
//     W_folded = W @ (H S)    i.e.  x_folded = H S x
//
// To recover the primal function at inference time the *activation* fed to such
// a matmul must undergo the same transform
//
//     y = W_folded @ x_folded = W @ H S @ H S x = W @ x   (since H S is orthogonal)
//
// The token embedding table is stored "latent" (rotated) instead, so its looked-up
// rows are restored to the primal basis with the *inverse* transform (H then S —
// the reverse order of the forward fold).
//
// The Gemma-style gated-delta-net `ssm_out` projection additionally receives its
// activation in a tiled value-head order and must be permuted into the grouped
// order the fold was computed in before the signs + rotation are applied.
// ============================================================================

namespace tinycoder::prism {

    /// @brief Normalized Sylvester-Walsh-Hadamard transform, in place, applied
    ///        independently to every contiguous block of `blockSize` elements.
    ///
    /// The butterfly yields the unnormalized Walsh-Hadamard transform in natural
    /// (Hadamard) order; the 1/sqrt(N) scale makes the matrix orthonormal so the
    /// same routine is its own inverse (H = H^-1).
    inline void fwhtBlockwise(float *v, uint32_t n, uint32_t blockSize) {
        if (blockSize <= 1 || n < blockSize) {
            return;
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(blockSize));
        for (uint32_t base = 0; base + blockSize <= n; base += blockSize) {
            float *x = v + base;
            for (uint32_t h = 1; h < blockSize; h <<= 1) {
                for (uint32_t i = 0; i < blockSize; i += (h << 1)) {
                    for (uint32_t j = 0; j < h; ++j) {
                        const float a = x[i + j];
                        const float b = x[i + j + h];
                        x[i + j] = a + b;
                        x[i + j + h] = a - b;
                    }
                }
            }
            for (uint32_t i = 0; i < blockSize; ++i) {
                x[i] *= scale;
            }
        }
    }

    /// @brief Reorder the gated-delta-net value feature axis from the engine's
    ///        tiled order [headV, nKHeads, rep] into the grouped order
    ///        [headV, rep, nKHeads] the Prism fold was computed in.
    ///
    /// The engine lays the ssm_out activation out as `v[hv * headV + d]` where hv
    /// is the value head (0..nVHeads-1) and d the in-head dim.  Grouping value
    /// heads by their periodic key head gives hv = r*nKHeads + k with r in
    /// [0, rep), k in [0, nKHeads).  The fold expects the feature index ordered
    /// d + r*headV + k*headV*rep instead.
    inline void permuteGdnValueGrouped(float *v, uint32_t headV, uint32_t nKHeads,
                                       uint32_t rep) {
        if (rep <= 1) {
            return;
        }
        // Engine layout: value head hv = k + r*nKHeads (k = hv % nKHeads is the
        // periodic key head, r = hv / nKHeads). Source linear index for (hv, d)
        // is hv*headV + d. The Prism fold was computed with the feature axis
        // grouped as d + r*headV + k*headV*rep, i.e. the in-head dim d is the
        // contiguous (stride-1) axis, r strides by headV and k by headV*rep.
        const uint32_t n = headV * nKHeads * rep;
        std::vector<float> tmp(n);
        for (uint32_t hv = 0; hv < nKHeads * rep; ++hv) {
            const uint32_t k = hv % nKHeads;
            const uint32_t r = hv / nKHeads;
            const float *src = v + static_cast<size_t>(hv) * headV;
            float *dst = tmp.data() + static_cast<size_t>(r) * headV +
                         static_cast<size_t>(k) * headV * rep;
            std::memcpy(dst, src, headV * sizeof(float));
        }
        std::memcpy(v, tmp.data(), n * sizeof(float));
    }

    /// @brief Inverse of permuteGdnValueGrouped: reorder a grouped feature axis
    ///        [headV, rep, nKHeads] back into the engine's tiled order
    ///        [headV, nKHeads, rep].  Used to bring a folded ssm_out weight's
    ///        columns back to the layout the engine's activation uses.
    inline void permuteGdnGroupedToTiled(float *v, uint32_t headV, uint32_t nKHeads,
                                         uint32_t rep) {
        if (rep <= 1) {
            return;
        }
        const uint32_t n = headV * nKHeads * rep;
        std::vector<float> tmp(n);
        for (uint32_t hv = 0; hv < nKHeads * rep; ++hv) {
            const uint32_t k = hv % nKHeads;
            const uint32_t r = hv / nKHeads;
            const float *src = v + static_cast<size_t>(r) * headV +
                               static_cast<size_t>(k) * headV * rep;
            float *dst = tmp.data() + static_cast<size_t>(hv) * headV;
            std::memcpy(dst, src, headV * sizeof(float));
        }
        std::memcpy(v, tmp.data(), n * sizeof(float));
    }

    /// @brief Apply the forward activation transform for a folded weight:
    ///        optional GDN value-head permutation, then the sign flip, then the
    ///        blockwise normalized Hadamard rotation (H S x).
    /// @param v         Activation vector (modified in place), size = weight cols.
    /// @param n         Number of elements (= weight cols).
    /// @param blockSize Hadamard block size (prism.hadamard.block_size).
    /// @param signs     Per-feature sign vector (may be null for identity mode).
    /// @param headV/nKHeads/rep  GDN geometry for permuteGdnValueGrouped; pass
    ///        rep <= 1 (or 0) to skip the permutation.
    inline void applyFoldForward(float *v, uint32_t n, uint32_t blockSize,
                                 const float *signs, uint32_t headV = 0,
                                 uint32_t nKHeads = 0, uint32_t rep = 0) {
        if (rep > 1 && headV > 0 && nKHeads > 0) {
            permuteGdnValueGrouped(v, headV, nKHeads, rep);
        }
        if (signs != nullptr) {
            for (uint32_t i = 0; i < n; ++i) {
                v[i] *= signs[i];
            }
        }
        fwhtBlockwise(v, n, blockSize);
    }

    /// @brief Restore a latent (rotated) embedding row to the primal basis.
    ///        Inverse of the forward fold: rotate first (H), then sign (S).
    inline void applyInverseLookup(float *v, uint32_t n, uint32_t blockSize,
                                   const float *signs) {
        fwhtBlockwise(v, n, blockSize);
        if (signs != nullptr) {
            for (uint32_t i = 0; i < n; ++i) {
                v[i] *= signs[i];
            }
        }
    }

    /// @brief Unfold one stored folded weight row (row-major) back to the primal
    ///        basis: W_primal_row = W_folded_row @ (S H) = S ⊙ (H · row).  This
    ///        is the load-time alternative to transforming the activation at
    ///        inference time and is preferred because the unfolded matrix
    ///        streams once per token.
    /// @param W         Row-major matrix data (modified in place), rows*cols.
    /// @param rows      Number of rows.
    /// @param cols      Number of columns (= the folded input width).
    /// @param blockSize Hadamard block size (must divide cols).
    /// @param signs     Per-column sign vector (nullptr for identity).
    /// @param headV/nKHeads/rep  GDN geometry: pass rep > 1 for a folded ssm_out
    ///        weight to also reorder its columns from the grouped order back to
    ///        the engine's tiled value-head order (after unfolding).
    inline void unfoldMatrixRows(float *W, uint32_t rows, uint32_t cols,
                                 uint32_t blockSize, const float *signs,
                                 uint32_t headV = 0, uint32_t nKHeads = 0,
                                 uint32_t rep = 0) {
        for (uint32_t r = 0; r < rows; ++r) {
            float *row = W + static_cast<size_t>(r) * cols;
            applyInverseLookup(row, cols, blockSize, signs);
            if (rep > 1 && headV > 0 && nKHeads > 0) {
                permuteGdnGroupedToTiled(row, headV, nKHeads, rep);
            }
        }
    }

    /// @brief Apply the forward activation transform (fold) to every row of an
    ///        `m x cols` matrix independently, writing the result to `out`
    ///        (`out` may alias `W` when m == 1 for the in-place single-row case).
    ///        This is the "folded weight" path: a stored folded weight row
    ///        W_folded is transformed row-wise, so that afterwards the plain dot
    ///        product y = x · out_row equals W_folded @ (H S x) — i.e. the model's
    ///        primal output without ever materializing an unfolded matrix.
    /// @param colPermute When true, apply the GDN value-head permutation as the
    ///        first step (ssm_out only); otherwise the transform is public.
    inline void transformRows(const float *W, float *out, uint32_t m, uint32_t cols,
                              uint32_t blockSize, const float *signs,
                              bool colPermute, uint32_t headV, uint32_t nKHeads,
                              uint32_t rep) {
        const uint32_t pv = colPermute ? headV : 0;
        const uint32_t pk = colPermute ? nKHeads : 0;
        const uint32_t pr = colPermute ? rep : 0;
        if (W == out) {
            // Single-row in-place transform (m must be 1).
            applyFoldForward(out, cols, blockSize, signs, pv, pk, pr);
            return;
        }
        for (uint32_t r = 0; r < m; ++r) {
            float *dst = out + static_cast<size_t>(r) * cols;
            std::memcpy(dst, W + static_cast<size_t>(r) * cols,
                        cols * sizeof(float));
            applyFoldForward(dst, cols, blockSize, signs, pv, pk, pr);
        }
    }

}// namespace tinycoder::prism
