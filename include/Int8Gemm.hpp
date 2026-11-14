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

// -----------------------------------------------------------------------------
// TinyCoder INT8/GEMV — int8 tensor-core prefill + decode GEMM dispatch.
//
// The kernels in src/cpp/core/cuda/ read the RAW GGUF quantized weights,
// so no pre-pack and no extra VRAM.  Activations are quantized once per
// (matrix, batch) to the Q8_1 layout consumed by mma.s8.m8n8k16.
//
// This header is the only entry point the engine needs: int8_dispatch.cu owns
// the activation scratch and the stream-k fixup buffer, queries the device
// once, and dispatches Q4_K / Q5_K / Q6_K / Q8_0 (prefill) and the small-batch
// GEMV types (decode).
// -----------------------------------------------------------------------------
#pragma once

#include <cstdint>

#ifdef USE_CUDA
namespace tinycoder::gpu {

    /// @brief True when INT8 is compiled in, the device is sm_75+, and
    ///        TINYCODER_INT8_GEMM is not disabled.
    bool int8GemmAvailable();

    /// @brief out[seqLen][rows] = x[seqLen][cols] @ W[rows][cols]^T using the
    ///        int8 tensor-core INT8 kernels.
    /// @param type        GGUF type id (8 (Q8_0), 12 (Q4_K), 13 (Q5_K), 14 (Q6_K)).
    /// @param w           raw quantized weight rows (rowBytes per row).
    /// @param x           fp32 activations, row-major [seqLen][cols].
    /// @param out         fp32 output, row-major [seqLen][rows].
    /// @param blocksPerRow cols / quant block size (256 for K-quants).
    /// @param stream      CUDA stream (void* to keep this header CUDA-free).
    /// @return false when the shape/type is unsupported or a launch failed; the
    ///         caller then falls back to the integer kQGemmKxQ8K kernel.
    bool int8Gemm(int type, const void *w, const float *x, float *out,
                 uint32_t seqLen, uint32_t rows, uint32_t cols,
                 uint32_t blocksPerRow, void *stream);

    /// @brief Small-batch (1..8 tokens) quantized mat-vec (Q8_1 activations,
    ///        one warp per row). This is the decode path: the per-token weight
    ///        stream is the only wall.
    bool int8Gemv(int type, const void *w, const float *x, float *out,
                  uint32_t seqLen, uint32_t rows, uint32_t cols, void *stream);

    /// @brief Pre-allocate the GEMV activation scratch (no launch).  Required
    ///        before a CUDA graph capture: cudaMalloc is not capturable.
    void int8GemvReserve(uint32_t cols, uint32_t seqLen);

/// @brief Enable/disable the Q8_0 int8-MMA path at runtime.  The sm_75 Q80
/// tile loader is broken for Q8-dominant checkpoints (illegal access on
/// Ornith-9B-Q8_0), while Q4_K checkpoints use it for their Q8_0 side
/// matrices.  Set false only when the model's main projections are Q8_0.
void int8GemmSetQ8Enabled(bool enabled);

/// @brief Fused IQ3_XXS gate+up GEMV: one Q8_1 activation quantization, both
/// row dots in one launch; gout receives silu(gate)*up.  Returns false for
/// other types/shapes.
bool int8GemvDualIQ3(const void *wg, const void *wu, const float *x,
                     float *gout, float *uout, uint32_t rows, uint32_t cols,
                     void *stream);

/// @brief Fused IQ2_XS gate+up GEMV (one Q8_1 activation, both dots, SwiGLU).
bool int8GemvDualIQ2XS(const void *wg, const void *wu, const float *x,
                       float *gout, float *uout, uint32_t rows, uint32_t cols,
                       void *stream);

/// @brief Fused Q4_K gate+up GEMV (one Q8_1 activation, SwiGLU output).
bool int8GemvDualQ4K(const void *wg, const void *wu, const float *x,
                     float *gout, float *uout, uint32_t rows, uint32_t cols,
                     void *stream);

}// namespace tinycoder::gpu
#endif
