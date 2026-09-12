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

#include "GGMLDequantize.hpp"
#include "Model.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <np/Array.hpp>

#ifdef USE_CUDA
#include <np/internal/cuda/Dot1d2d.hpp>
#endif

namespace tinycoder {

    np::Array<float> QuantizedMatrix::matMulVec(const float *x) const {
        // Allocate result and delegate to the out-parameter version
        np::Array<float> result(np::Shape{rows});
        matMulVec(x, result.data());
        return result;
    }

    uint32_t ggmlTypeSize(uint32_t type);

    // Dispatch statistics (A/B diagnostics): count how many matMulVec calls
    // went through the SIMD batch kernels vs the scalar fallback, per quant
    // type. Printed once at process exit via std::atexit — zero runtime cost
    // (relaxed atomic increments).
    namespace {
        std::atomic<uint64_t> g_batchKernelCalls{0};
        std::atomic<uint64_t> g_scalarFallbackCalls{0};
        std::atomic<int> g_batchKernelTypes[64]{};
        struct BatchKernelStatsDumper {
            BatchKernelStatsDumper() {
                std::atexit([] {
                    if (g_batchKernelCalls.load() == 0 &&
                        g_scalarFallbackCalls.load() == 0)
                        return;
                    std::fprintf(stderr,
                                 "[TinyCoder] matMulVec dispatch: SIMD-batch=%llu "
                                 "scalar=%llu | per-type (type:calls) ",
                                 static_cast<unsigned long long>(
                                         g_batchKernelCalls.load()),
                                 static_cast<unsigned long long>(
                                         g_scalarFallbackCalls.load()));
                    for (int t = 0; t < 64; ++t) {
                        int c = g_batchKernelTypes[t].load();
                        if (c > 0)
                            std::fprintf(stderr, "[%d:%d] ", t, c);
                    }
                    std::fprintf(stderr, "\n");
                });
            }
        };
        inline BatchKernelStatsDumper g_batchKernelStatsDumper;
    }// namespace

    // Single-token (seqLen==1) dispatcher into the register-tiled AVX2 batch
    // GEMM kernels. These are the same kernels that give the LM head ~36
    // GFLOP/s (vs the scalar double-precision dotProductFused path at ~0.3
    // GFLOP/s) — the dominant qwen35 per-token cost was the scalar path.
    //
    // Contract: all kernels require cols % 256 == 0 (a whole number of K-quant
    // blocks), which holds for every real qwen2/qwen35 matrix (embedding 5120,
    // FFN intermediate 17408, headDim 128/256). Non-multiple cols fall back to
    // the scalar fused path below.
    static bool matMulVecBatchSIMD(uint32_t type, const uint8_t *data,
                                   const float *x, uint32_t rows, uint32_t cols,
                                   float *out) {
        if (cols == 0 || (cols & 255) != 0) {
            return false;
        }
        // Each case dispatches to the register-tiled AVX2 batch kernel for that
        // quant type (see SIMDMatMulVecAVX2.cpp). All kernels require the
        // cols % 256 == 0 contract checked above. Q5_K needs its own kernel
        // (176 B/block: the high bit lives in a separate qh array, not in the
        // low nibbles), IQ4_XS its own (136 B/block with per-32 scale), and
        // IQ4_NL has 32-wide blocks (18 B/block) — none can share another
        // type's layout.
        switch (type) {
            case GGML_TYPE_Q6_K:
                return matMulVecBatchQ6K_SIMD(data, x, 1, rows, cols, out);
            case GGML_TYPE_Q5_K:
                // Q5_K yields exactly the same block size (256) and the same
                // scale/min extraction as Q4_K except the weights are 5-bit
                // (low 4 bits in qs, high bit in qh). Use a dedicated AVX2
                // kernel (see below); no generic fallback here.
                return matMulVecBatchQ5K_SIMD(data, x, 1, rows, cols, out);
            case GGML_TYPE_Q4_K:
                return matMulVecBatchQ4K_SIMD(data, x, 1, rows, cols, out);
            case GGML_TYPE_IQ4_XS:
                return matMulVecBatchIQ4XS_SIMD(data, x, 1, rows, cols, out);
            case GGML_TYPE_IQ4_NL:
                return matMulVecBatchIQ4NL_SIMD(data, x, 1, rows, cols, out);
            case GGML_TYPE_Q3_K:
                return matMulVecBatchQ3K_SIMD(data, x, 1, rows, cols, out);
            case GGML_TYPE_Q8_K:
                return matMulVecBatchQ8K_SIMD(
                        reinterpret_cast<const Q8KBlock *>(data), x, 1, rows,
                        cols, out);
            default:
                return false;
        }
    }

    void QuantizedMatrix::matMulVec(const float *x, float *out) const {
        // Compute y = x * W where W is this quantized matrix.
        // W has dimensions (rows x cols) = (out_features x in_features), stored
        // row-major in GGUF format. x is a row vector of size cols (in_features).
        // y is a row vector of size rows (out_features).
        //
        // GGUF stores weight matrices as (out_features x in_features) row-major:
        //   W[j][i] = data[j * cols + i]
        // where j indexes output features and i indexes input features.
        //
        // The computation is: y_j = sum_i x[i] * W[j][i]  for j in [0, rows), i in
        // [0, cols)
        //
        // OPTIMIZATION: Instead of dequantizing the entire matrix to float and then
        // doing the dot product, we use a block-level fused dequantize-dot approach:
        // for each output row, we dequantize one block at a time and compute the
        // dot product directly. This avoids:
        //   1. Allocating a large float buffer for the full dequantized matrix
        //      (e.g., 13.76M elements = ~55 MB for ffnGate)
        //   2. The memory bandwidth bottleneck of writing/reading all those floats
        //   3. The overhead of copying into np::Array for the dot product
        //
        // For F32 matrices, we still use the CUDA/CPU path.

        // For F32 type, use the CUDA/CPU path (no dequantization needed)
        if (type == GGML_TYPE_F32) {
            const float *W_f32 = reinterpret_cast<const float *>(data.data());
            // For F32, compute directly into out (avoiding the CUDA path's allocation)
            for (uint32_t j = 0; j < rows; ++j) {
                double dot = 0.0;
                for (uint32_t i = 0; i < cols; ++i) {
                    dot += static_cast<double>(x[i]) * W_f32[static_cast<size_t>(j) * cols + i];
                }
                out[j] = static_cast<float>(dot);
            }
            return;
        }

        // For quantized types, use block-level fused dequantize-dot
        // matMulVecFused computes y_j = sum_i x[i] * W[j][i]
        // where x has size cols, result has size rows

        // OPTIMIZATION: register-tiled AVX2 batch GEMM fast path (single token).
        // The LM head reaches ~36 GFLOP/s through these kernels while the
        // generic scalar dotProductFused path below crawls at ~0.3 GFLOP/s.
        // This is the dominant qwen35 per-token cost (FFN gate/up/down +
        // attention projections all call matMulVec with seqLen==1). All real
        // model matrices have cols % 256 == 0; non-multiple cols fall back.
        // TINYCODER_FORCE_SCALAR=1 bypasses the kernels to establish a scalar
        // correctness baseline (A/B the same generation between the SIMD batch
        // kernels and the reference dequantize+dot path).
        if (!forceScalarPath() &&
            matMulVecBatchSIMD(type, data.data(), x, rows, cols, out)) {
            g_batchKernelCalls.fetch_add(1, std::memory_order_relaxed);
            if (type >= 0 && type < 64)
                g_batchKernelTypes[type].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        g_scalarFallbackCalls.fetch_add(1, std::memory_order_relaxed);

        // OPTIMIZATION: Use pre-packed kernel for Q2_K matrices that have been
        // pre-packed at load time. The pre-packed format eliminates the 2-bit
        // extraction overhead in the SIMD kernel.
        if (type == GGML_TYPE_Q2_K && !prepackedData.empty()) {
            // quantize x to Q8_K (int8) once per matmul,
            // then use _mm256_maddubs_epi16 (32 int8×int8->int16 multiply-adds
            // per instruction) instead of float FMAs (8 per instruction).
            GGMLDequantize::matMulVecFusedQ2_K_PrePacked_Q8(prepackedData.data(), x, rows, cols, out);
            return;
        }

        // Use cache-blocked version for large matrices (rows > 256) to keep
        // the x vector in L1 cache. For small matrices, the non-blocked version
        // is fine and avoids the memset overhead.
        if (rows > 256) {
            GGMLDequantize::matMulVecFusedCacheBlocked(type, data.data(), x, rows, cols, out);
        } else {
            GGMLDequantize::matMulVecFused(type, data.data(), x, rows, cols, out);
        }
    }

    np::Array<float> QuantizedMatrix::matMulVecRows(const float *x, uint32_t rowStart, uint32_t numRows) const {
        // Allocate result and delegate to the out-parameter version
        np::Array<float> result(np::Shape{numRows});
        matMulVecRows(x, rowStart, numRows, result.data());
        return result;
    }

    void QuantizedMatrix::matMulVecFusedGateUp(const QuantizedMatrix &other,
                                               const float *x, float *gateOut,
                                               float *upOut,
                                               bool applySwish) const {
        // Compute gate = x * this^T and up = x * other^T in a single pass over x.
        // Both matrices must share the same dimensions and quantized type.
        if (rows != other.rows || cols != other.cols || type != other.type) {
            // Fallback: two separate matmuls
            matMulVec(x, gateOut);
            other.matMulVec(x, upOut);
            return;
        }

        // llama-parity Q8K path: for IQ2_S / IQ3_XXS / IQ3_S weights llama's CPU
        // mul_mat quantizes the activations to Q8_K and accumulates with exact
        // integer math (vec_dot_iq*_q8_K). Route single-token generation
        // through matMulVecFusedQ8K to reproduce that bit-for-bit instead of
        // the lossy float dequant-dot (diverges ~4.7% from llama).
        if (GGMLDequantize::supportsQ8KDot(type)) {
            GGMLDequantize::matMulVecFusedQ8K(type, data.data(), x, rows, cols,
                                              gateOut);
            GGMLDequantize::matMulVecFusedQ8K(type, other.data.data(), x, rows,
                                              cols, upOut);
            if (applySwish) {
                for (uint32_t j = 0; j < rows; ++j) {
                    float g = gateOut[j];
                    gateOut[j] = (g / (1.0f + std::exp(-g))) * upOut[j];
                }
            }
            return;
        }

        // Fast path: fused Q2_K gate+up kernel over the COMPACT (raw) blocks.
        // This fn is used for single-token generation, which is
        // DRAM-bandwidth-bound; the compact 84-byte Q2_K blocks (vs 276-byte
        // prepacked) cut the gate+up weight traffic ~3.3x — the dominant
        // per-token cost — matching llama.cpp's working set. The kernel unpacks
        // the 2-bit quants on the fly (verified index-identical to the prepacked
        // expansion) and computes both dot products per row, quantizing x to
        // Q8_K once.
        if (!forceScalarPath() && type == GGML_TYPE_Q2_K && !data.empty() &&
            !other.data.empty()) {
            GGMLDequantize::matMulVecFusedGateUpQ2_K_Compact_Q8(
                    data.data(), other.data.data(), x, rows, cols,
                    gateOut, upOut, applySwish);
            return;
        }

        // Fallback fast path: pre-packed Q2_K gate+up fused kernel (same maths,
        // 3.3x larger weight read — used when the compact data is unavailable).
        if (!forceScalarPath() && type == GGML_TYPE_Q2_K && !prepackedData.empty() &&
            !other.prepackedData.empty()) {
            GGMLDequantize::matMulVecFusedGateUpQ2_K_PrePacked_Q8(
                    prepackedData.data(), other.prepackedData.data(), x, rows, cols,
                    gateOut, upOut);
            return;
        }

        // General fallback: two separate matmuls
        matMulVec(x, gateOut);
        other.matMulVec(x, upOut);

        // The Q2_K fast paths fuse the SwiGLU activation into their epilogues,
        // so the caller relies on this function to apply it consistently for
        // EVERY dispatch route.  The general fallback must therefore honour
        // applySwish too — otherwise single-token generation skips the
        // silu(gate)*up activation entirely for non-Q2_K quant types (Q5_0 /
        // Q8_0 / Q4_K / Q6_K), producing garbage logits and diverging from the
        // batched prefill path (which always applies it).
        if (applySwish) {
            for (uint32_t j = 0; j < rows; ++j) {
                float g = gateOut[j];
                gateOut[j] = (g / (1.0f + std::exp(-g))) * upOut[j];
            }
        }
    }

    void QuantizedMatrix::matMulVecRows(const float *x, uint32_t rowStart, uint32_t numRows, float *out) const {
        // Compute y = x * W for a contiguous range of rows [rowStart, rowStart + numRows).
        // W has dimensions (rows x cols) = (out_features x in_features), stored row-major.
        // This is used for expert sub-matrices in MoE architectures where multiple
        // experts are stored in a single QuantizedMatrix.
        //
        // The data layout is: expert 0 rows, expert 1 rows, ..., expert N-1 rows.
        // Each expert has expertFF rows (for gate/up) or hiddenSize rows (for down).
        //
        // We compute y_j = sum_i x[i] * W[rowStart + j][i] for j in [0, numRows)

        if (rowStart + numRows > rows) {
            std::cerr << "[TinyCoder] matMulVecRows: rowStart=" << rowStart
                      << " numRows=" << numRows << " exceeds rows=" << rows << std::endl;
            std::memset(out, 0, static_cast<size_t>(numRows) * sizeof(float));
            return;
        }

        if (type == GGML_TYPE_F32) {
            const float *W_f32 = reinterpret_cast<const float *>(data.data());
            const float *W_start = W_f32 + static_cast<size_t>(rowStart) * cols;
            for (uint32_t j = 0; j < numRows; ++j) {
                double dot = 0.0;
                for (uint32_t i = 0; i < cols; ++i) {
                    dot += static_cast<double>(x[i]) * W_start[static_cast<size_t>(j) * cols + i];
                }
                out[j] = static_cast<float>(dot);
            }
        } else {
            // For quantized types, compute row by row using the fused quantized dot product.
            // This eliminates the float blockBuf[256] temporary and the extra memory pass.
            uint32_t blockSize = ggmlBlockSize(type);
            uint32_t typeSize = ggmlTypeSize(type);
            uint64_t blocksPerRow = (static_cast<uint64_t>(cols) + blockSize - 1) / blockSize;
            uint64_t bytesPerRow = blocksPerRow * typeSize;
            const uint8_t *rowData = data.data() + static_cast<size_t>(rowStart) * bytesPerRow;

            for (uint32_t j = 0; j < numRows; ++j) {
                const uint8_t *rowPtr = rowData + j * bytesPerRow;
                double dot = 0.0;
                uint32_t remaining = cols;
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    uint32_t blockN = std::min(blockSize, remaining);
                    // Fused quantized dot product: dequantize and dot in one pass
                    dot += static_cast<double>(GGMLDequantize::dotProductFused(type, rowPtr + b * typeSize, x + b * blockSize, blockN));
                    remaining -= blockN;
                }
                out[j] = static_cast<float>(dot);
            }
        }
    }

}// namespace tinycoder
