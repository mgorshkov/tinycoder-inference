#pragma once
#include <cstdint>

// Prototype-only entry points for the "dequantize an expert layer to FP32 once,
// then run cuBLAS SGEMM over the token columns" experiment.
//
// Nothing in the shipped forward graph calls these: they exist so
// tools/inkling_fp32_gemm_probe can measure the dequant + SGEMM cost against
// the exact grouped float GEMV path on the real blk.0 expert tensors, and can
// check that the dequantized values are the exact GGMLDequantize values.
namespace tinycoder::gpu {

    /// GGML types with a device dequant-to-FP32 kernel here: 19 (IQ1_S) and
    /// 42 (Q2_0), the two types blk.0's routed experts carry in the
    /// IQ2_XXS checkpoint.
    bool inklingFp32DequantSupported(uint32_t type);

    /// Dequantize `numElements` weights (a multiple of the type's block size)
    /// from `w` into `out` as FP32.  Bit-exact with
    /// GGMLDequantize::dequantize(): the kernel repeats the host arithmetic
    /// operation for operation, and every dequantized value is exactly
    /// representable in FP32 (fp16 scale * small integer).
    bool inklingFp32Dequant(uint32_t type, const void *w, float *out,
                            uint64_t numElements, void *stream);

    /// out[m][rows] = x[m][cols] * W[rows][cols]^T with W and x FP32
    /// row-major, via cublasSgemm (op(A)=T, op(B)=N in column-major terms).
    /// cuBLAS is used as a plain GEMM, which the repo rules allow.
    bool inklingFp32Gemm(const float *w, const float *x, float *out,
                         uint32_t rows, uint32_t cols, uint32_t m,
                         void *stream);

}// namespace tinycoder::gpu
