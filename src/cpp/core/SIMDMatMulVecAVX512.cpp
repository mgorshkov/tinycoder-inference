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

// ============================================================================
// AVX-512 implementations of the SIMD mat-vec kernels.
//
// This translation unit is compiled with -mavx512f -mfma (see CMakeLists.txt)
// so the AVX-512/FMA intrinsics below are always available here. The kernels
// are declared in SIMDMatMulVecInternal.hpp and referenced by the runtime
// dispatch in SIMDMatMulVec.cpp.
// ============================================================================

#include "SIMDMatMulVecInternal.hpp"

// Block dequantizers (dequantizeNVFP4Block / dequantizeQ2_0Block) and the
// thread pool, shared with the AVX2 kernels so both decode weights through the
// same code.
#include "GGMLDequantize.hpp"
#include "ThreadPool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>

namespace tinycoder::simd {

    void accumulateFMA_AVX512(float *local, const float *blockOut, float alpha,
                              uint32_t n) {
        __m512 alphaVec = _mm512_set1_ps(alpha);
        uint32_t j = 0;
        for (; j + 16 <= n; j += 16) {
            __m512 blockVec = _mm512_loadu_ps(blockOut + j);
            __m512 localVec = _mm512_loadu_ps(local + j);
            __m512 result = _mm512_fmadd_ps(alphaVec, blockVec, localVec);
            _mm512_storeu_ps(local + j, result);
        }
        for (; j < n; ++j) {
            local[j] += alpha * blockOut[j];
        }
    }

    float dotProductFMA_AVX512(const float *hidden, const float *blockOut,
                               uint32_t n) {
        __m512 sumVec = _mm512_setzero_ps();
        uint32_t j = 0;
        for (; j + 16 <= n; j += 16) {
            __m512 hVec = _mm512_loadu_ps(hidden + j);
            __m512 bVec = _mm512_loadu_ps(blockOut + j);
            sumVec = _mm512_fmadd_ps(hVec, bVec, sumVec);
        }
        float result = _mm512_reduce_add_ps(sumVec);
        for (; j < n; ++j) {
            result += hidden[j] * blockOut[j];
        }
        return result;
    }

    void rmsNorm_AVX512(const float *x, float *out, const float *weight,
                        uint32_t n, float eps) {
        __m512 sumSqVec = _mm512_setzero_ps();
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            sumSqVec = _mm512_fmadd_ps(xVec, xVec, sumSqVec);
        }
        double sumSq = static_cast<double>(_mm512_reduce_add_ps(sumSqVec));
        for (; i < n; ++i) {
            sumSq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        }
        float invRms = 1.0f / (std::sqrt(static_cast<float>(sumSq / static_cast<double>(n))) + eps);

        __m512 invRmsVec = _mm512_set1_ps(invRms);
        i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            __m512 wVec = _mm512_loadu_ps(weight + i);
            _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(xVec, invRmsVec), wVec));
        }
        for (; i < n; ++i) {
            out[i] = x[i] * invRms * weight[i];
        }
    }

    void softmax_AVX512(float *x, uint32_t n) {
        // Use scalar for simplicity and accuracy
        float maxVal = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < n; ++i) {
            if (x[i] > maxVal) maxVal = x[i];
        }
        __m512 maxVec = _mm512_set1_ps(maxVal);
        __m512 sumExpVec = _mm512_setzero_ps();
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            __m512 subVec = _mm512_sub_ps(xVec, maxVec);
            // AVX-512 has no native exp, use scalar
            float vals[16];
            _mm512_storeu_ps(vals, subVec);
            for (int k = 0; k < 16; ++k) {
                vals[k] = std::exp(vals[k]);
            }
            __m512 expVec = _mm512_loadu_ps(vals);
            _mm512_storeu_ps(x + i, expVec);
            sumExpVec = _mm512_add_ps(sumExpVec, expVec);
        }
        double sumExp = static_cast<double>(_mm512_reduce_add_ps(sumExpVec));
        for (; i < n; ++i) {
            float ev = std::exp(x[i] - maxVal);
            x[i] = ev;
            sumExp += static_cast<double>(ev);
        }
        float invSum = static_cast<float>(1.0 / sumExp);
        __m512 invSumVec = _mm512_set1_ps(invSum);
        i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            _mm512_storeu_ps(x + i, _mm512_mul_ps(xVec, invSumVec));
        }
        for (; i < n; ++i) {
            x[i] *= invSum;
        }
    }

    void silu_AVX512(float *x, uint32_t n) {
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            float vals[16];
            _mm512_storeu_ps(vals, _mm512_loadu_ps(x + i));
            for (int k = 0; k < 16; ++k) {
                vals[k] = vals[k] / (1.0f + std::exp(-vals[k]));
            }
            _mm512_storeu_ps(x + i, _mm512_loadu_ps(vals));
        }
        for (; i < n; ++i) {
            x[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    void swiGLU_AVX512(float *x, const float *y, uint32_t n) {
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            float xVals[16], yVals[16];
            _mm512_storeu_ps(xVals, _mm512_loadu_ps(x + i));
            _mm512_storeu_ps(yVals, _mm512_loadu_ps(y + i));
            for (int k = 0; k < 16; ++k) {
                float siluVal = xVals[k] / (1.0f + std::exp(-xVals[k]));
                xVals[k] = siluVal * yVals[k];
            }
            _mm512_storeu_ps(x + i, _mm512_loadu_ps(xVals));
        }
        for (; i < n; ++i) {
            float siluVal = x[i] / (1.0f + std::exp(-x[i]));
            x[i] = siluVal * y[i];
        }
    }

    void add_AVX512(float *x, const float *y, uint32_t n) {
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            __m512 yVec = _mm512_loadu_ps(y + i);
            _mm512_storeu_ps(x + i, _mm512_add_ps(xVec, yVec));
        }
        for (; i < n; ++i) {
            x[i] += y[i];
        }
    }

    void scale_AVX512(float *x, float alpha, uint32_t n) {
        __m512 alphaVec = _mm512_set1_ps(alpha);
        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 xVec = _mm512_loadu_ps(x + i);
            _mm512_storeu_ps(x + i, _mm512_mul_ps(xVec, alphaVec));
        }
        for (; i < n; ++i) {
            x[i] *= alpha;
        }
    }

    // ---- AVX-512 F16 dot product ----
    // Computes dot(hidden, W) where W is stored as FP16 (uint16_t).
    // Uses F16C _mm512_cvtph_ps to convert 16 FP16 values to FP32 in one
    // instruction (vs 8 per instruction in the AVX2 kernel), doubling the
    // throughput of the memory-bound FFN-down / attnO mat-vecs.
    float dotProductFMA_F16_AVX512(const float *hidden, const uint16_t *W_f16,
                                   uint32_t n) {
        __m512 sumVec = _mm512_setzero_ps();
        uint32_t j = 0;
        for (; j + 16 <= n; j += 16) {
            // Load 16 FP16 values and convert to FP32 using F16C
            __m256i f16 = _mm256_loadu_si256((const __m256i *) (W_f16 + j));
            __m512 wVec = _mm512_cvtph_ps(f16);
            __m512 hVec = _mm512_loadu_ps(hidden + j);
            sumVec = _mm512_fmadd_ps(hVec, wVec, sumVec);
        }
        float result = _mm512_reduce_add_ps(sumVec);
        for (; j < n; ++j) {
            // Scalar tail: convert FP16 to float manually
            uint16_t h = W_f16[j];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f32;
            if (exp == 0) {
                if (mant == 0) {
                    f32 = sign << 31;
                } else {
                    int nz = 0;
                    while ((mant & 0x200) == 0 && nz < 10) {
                        mant <<= 1;
                        nz++;
                    }
                    mant &= 0x3FF;
                    uint32_t mant_low = mant - 512;
                    exp = 112 - nz;
                    f32 = (sign << 31) | (exp << 23) | (mant_low << 14);
                }
            } else if (exp == 31) {
                f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
            } else {
                exp = exp + (127 - 15);
                f32 = (sign << 31) | (exp << 23) | (mant << 13);
            }
            float wVal;
            std::memcpy(&wVal, &f32, sizeof(float));
            result += hidden[j] * wVal;
        }
        return result;
    }

    // _mm512_sign_epi8 does not exist (VPSSIGNB is AVX2-only). Emulate
    // a * sgn(b) per byte with a compare + masked blend: -a when b < 0, a
    // otherwise. Note: when b == 0 VPSSIGNB yields 0 but this yields a; that
    // only matters for lanes where the paired |x| byte is also 0 (so the
    // maddubs product is 0 either way), keeping results bit-identical.
    static inline __m512i signEPI8_AVX512(__m512i a, __m512i b) {
        const __m512i zero = _mm512_setzero_si512();
        __mmask64 neg = _mm512_cmpgt_epi8_mask(zero, b);// mask where b < 0
        return _mm512_mask_blend_epi8(neg, a, _mm512_sub_epi8(zero, a));
    }

    // ---- AVX-512 Q8_K x Q8_K dot product ----
    // Computes dot(x, w) where both x and w are Q8_K blocks (256 int8 values +
    // a block scale). Uses _mm512_maddubs_epi16 (64 int8xint8->int16
    // multiply-adds per instruction) instead of 32 per instruction in the AVX2
    // kernel. This is the hot kernel of the LM head Q8_K path.
    float dotProductQ8K_Q8K_AVX512(const tinycoder::Q8KBlock *x,
                                   const tinycoder::Q8KBlock *w) {
        __m512i sumi = _mm512_setzero_si512();
        for (int i = 0; i < 256; i += 64) {
            // _mm512_maddubs_epi16 treats its FIRST operand as unsigned bytes
            // and the second as signed bytes. Both qs arrays are signed int8,
            // so raw signed values cannot go into the unsigned slot (a negative
            // byte would be reinterpreted as value+256). Products are bounded
            // by 127*127, so the int16 intermediate in maddubs cannot saturate.
            __m512i xv = _mm512_loadu_si512((const __m512i *) &x->qs[i]);
            __m512i wv = _mm512_loadu_si512((const __m512i *) &w->qs[i]);
            // xabs = |xv| (VPABSB, AVX-512BW); the sign trick uses
            // _mm512_sign_epi8 semantics, which only exist as VPSSIGNB in AVX2.
            __m512i xabs = _mm512_abs_epi8(xv);
            __m512i wsig = signEPI8_AVX512(wv, xv);
            __m512i p = _mm512_maddubs_epi16(xabs, wsig);
            sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(p, _mm512_set1_epi16(1)));
        }
        int sum = _mm512_reduce_add_epi32(sumi);
        return x->d * w->d * static_cast<float>(sum);
    }

    // ---- AVX-512 Q8_K dot product for pre-packed Q2_K ----
    // x is quantized to int8 (Q8_K) once per matmul. The dot product then uses
    // _mm512_maddubs_epi16 (64 int8xint8->int16 multiply-adds per instruction,
    // vs 32 in the AVX2 kernel). This is the hot kernel of the FFN gate+up
    // fused mat-vec.
    //
    // Pre-packed block layout (276 bytes):
    //   Offset 0-15:   scales[16]   (4-bit scale low, 4-bit min high)
    //   Offset 16-17:  d            (fp16)
    //   Offset 18-19:  dmin         (fp16)
    //   Offset 20-275: qs_expanded[256] (each byte is 0-3, in element order)
    //
    // For each 16-element group g with scale dl and min ml:
    //   sum_xq = d * sum(qs[i] * q2[i])   via maddubs
    //   sum_x  = d * bsums[g]             via madd (mins * bsums)
    //   dot   += dl * sum_xq - ml * sum_x
    float dotProductQ2_K_PrePacked_Q8_AVX512(const uint8_t *prepackedBlock,
                                             const tinycoder::Q8KBlock *q8) {
        auto halfToFloat = [](uint16_t h) -> float {
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
                        n++;
                    }
                    mant &= 0x3FF;
                    uint32_t mant_low = mant - 512;
                    exp = 112 - n;
                    f32 = (sign << 31) | (exp << 23) | (mant_low << 14);
                }
            } else if (exp == 31) {
                f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
            } else {
                exp = exp + (127 - 15);
                f32 = (sign << 31) | (exp << 23) | (mant << 13);
            }
            float result;
            std::memcpy(&result, &f32, sizeof(float));
            return result;
        };

        const float d = q8->d;
        const float dq = halfToFloat(*(const uint16_t *) (prepackedBlock + 16));
        const float dmin = halfToFloat(*(const uint16_t *) (prepackedBlock + 18));
        const uint8_t *scales = prepackedBlock;
        const uint8_t *qs_expanded = prepackedBlock + 20;

        __m512 acc = _mm512_setzero_ps();

        // Fold the min term: acc += -dmin * d * sum_g mins[g] * bsums[g].
        // mins are 4-bit (0-15), bsums are int16. Only 16 groups, so a scalar
        // loop is negligible compared to the 256-element main loop.
        {
            int minsSum = 0;
            for (int g = 0; g < 16; ++g) {
                uint8_t m = (scales[g] >> 4) & 0xF;
                minsSum += static_cast<int>(m) * q8->bsums[g];
            }
            acc = _mm512_fmadd_ps(_mm512_set1_ps(-dmin * d),
                                  _mm512_set1_ps(static_cast<float>(minsSum)), acc);
        }

        for (int n = 0; n < 256; n += 128) {
            const int chunk = n / 128;// 0 or 1
            __m512i sumi = _mm512_setzero_si512();

            // 2 sub-chunks of 64 elements each = 128 elements.
            // Each 64-element sub-chunk spans 4 groups of 16.
            //   chunk 0, sub 0 -> groups 0-3   (scales 0-3)
            //   chunk 0, sub 1 -> groups 4-7   (scales 4-7)
            //   chunk 1, sub 0 -> groups 8-11  (scales 8-11)
            //   chunk 1, sub 1 -> groups 12-15 (scales 12-15)
            for (int sub = 0; sub < 2; ++sub) {
                __m512i q2 = _mm512_loadu_si512((const __m512i *) (qs_expanded + sub * 64));
                __m512i q8v = _mm512_loadu_si512((const __m512i *) &q8->qs[n + sub * 64]);
                // q2 values are 0-3 (unsigned), q8 are int8 (signed).
                // maddubs treats the first operand as unsigned, second as signed.
                __m512i p = _mm512_maddubs_epi16(q2, q8v);

                // Build the scale vector for the 4 groups in this sub-chunk.
                // Each group's 16-bit scale is duplicated across 8 int16 lanes
                // (16 bytes). _mm512_set_epi16 lists args high-to-low, so the
                // lowest 8 int16 (bytes 0-15) = group 0 = s0.
                const int scaleBase = chunk * 8 + sub * 4;
                const int s0 = scales[scaleBase + 0] & 0xF;
                const int s1 = scales[scaleBase + 1] & 0xF;
                const int s2 = scales[scaleBase + 2] & 0xF;
                const int s3 = scales[scaleBase + 3] & 0xF;
                __m512i scales_v = _mm512_set_epi16(
                        s3, s3, s3, s3, s3, s3, s3, s3, // bytes 48-63 (group 3)
                        s2, s2, s2, s2, s2, s2, s2, s2, // bytes 32-47 (group 2)
                        s1, s1, s1, s1, s1, s1, s1, s1, // bytes 16-31 (group 1)
                        s0, s0, s0, s0, s0, s0, s0, s0);// bytes 0-15 (group 0)

                sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(scales_v, p));
            }

            // acc += dq * d * sumi
            acc = _mm512_fmadd_ps(_mm512_set1_ps(dq * d), _mm512_cvtepi32_ps(sumi), acc);

            qs_expanded += 128;
        }

        return _mm512_reduce_add_ps(acc);
    }

    // ---- AVX-512 batch GEMM for NVFP4 / Q2_0 (Inkling-2-Mini MoE experts) ----
    //
    // Same contract and same block math as the AVX2 kernels in
    // SIMDMatMulVecAVX2.cpp: out[s*rows + j], each 64-element block decoded
    // through the GGMLDequantize block function (so the reconstruction is
    // bit-identical to the scalar path) and dotted with the activation using
    // 16-wide FMA. Only the accumulation order differs from the AVX2 kernel
    // (16 lanes / four chains instead of 8 lanes / four chains), which is the
    // same tolerance class as the Q4_K AVX2-vs-AVX-512 pair.
    //
    // AMX is deliberately NOT used here. AMX is an int8/BF16 tile engine: to
    // feed it an NVFP4/Q2_0 block you would first expand every 2/4-bit code to
    // an int8 tile and separately apply the per-block UE4M3/fp16 scale, which
    // is exactly the dequant + FMA performed below (the scale cannot ride the
    // int8 accumulate because it changes every 16 or 64 elements). That adds
    // tile setup and copy cost while saving no arithmetic, so there is no AMX
    // path for these types. (The engine has no AMX code at all; ENABLE_AMX only
    // forwards to the scalar fallback.)
    namespace {

        inline float dotBlockFMA16_AVX512(const float *w, const float *x,
                                          uint32_t n) {
            __m512 a0 = _mm512_setzero_ps();
            __m512 a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps();
            __m512 a3 = _mm512_setzero_ps();
            uint32_t i = 0;
            for (; i + 64u <= n; i += 64u) {
                a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i + 0u),
                                     _mm512_loadu_ps(x + i + 0u), a0);
                a1 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i + 16u),
                                     _mm512_loadu_ps(x + i + 16u), a1);
                a2 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i + 32u),
                                     _mm512_loadu_ps(x + i + 32u), a2);
                a3 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i + 48u),
                                     _mm512_loadu_ps(x + i + 48u), a3);
            }
            for (; i + 16u <= n; i += 16u) {
                a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i),
                                     _mm512_loadu_ps(x + i), a0);
            }
            __m512 s = _mm512_add_ps(_mm512_add_ps(a0, a1),
                                     _mm512_add_ps(a2, a3));
            float sum = _mm512_reduce_add_ps(s);
            // Partial last block only (cols not a multiple of the block size).
            for (; i < n; ++i) {
                sum += w[i] * x[i];
            }
            return sum;
        }

        // QK = elements per block, BLOCK_BYTES = storage per block. blockFn is
        // one of GGMLDequantize::dequantize{NVFP4,Q2_0}Block.
        template<uint32_t QK, uint32_t BLOCK_BYTES>
        void matMulVecBatchBlocks_AVX512(const uint8_t *W, const float *X,
                                         uint32_t seqLen, uint32_t rows,
                                         uint32_t cols, float *out,
                                         void (*blockFn)(const uint8_t *, float *)) {
            if (seqLen == 0 || rows == 0 || cols == 0) {
                return;
            }
            const uint32_t blocksPerRow = (cols + QK - 1) / QK;
            const uint64_t rowStride =
                    static_cast<uint64_t>(blocksPerRow) * BLOCK_BYTES;

            // One row at a time (streaming), parallel over rows. Each thread
            // keeps seqLen accumulators and decodes each weight block once,
            // reusing it for every token.
            ThreadPool::current().parallelForSlab(0, rows, [&](uint32_t row) {
                static thread_local std::vector<float> acc;
                acc.assign(seqLen, 0.0f);

                const uint8_t *rp = W + static_cast<uint64_t>(row) * rowStride;
                float blockOut[QK];
                for (uint32_t b = 0; b < blocksPerRow; ++b) {
                    if (b + 2u < blocksPerRow) {
                        _mm_prefetch(
                                reinterpret_cast<const char *>(
                                        rp + static_cast<uint64_t>(b + 2u) *
                                                     BLOCK_BYTES),
                                _MM_HINT_T0);
                    }
                    const uint8_t *blockData =
                            rp + static_cast<uint64_t>(b) * BLOCK_BYTES;
                    blockFn(blockData, blockOut);

                    const uint32_t start = b * QK;
                    const uint32_t n = std::min(QK, cols - start);
                    for (uint32_t s = 0; s < seqLen; ++s) {
                        acc[s] += dotBlockFMA16_AVX512(
                                blockOut,
                                X + static_cast<uint64_t>(s) * cols + start,
                                n);
                    }
                }
                for (uint32_t s = 0; s < seqLen; ++s) {
                    out[static_cast<uint64_t>(s) * rows + row] = acc[s];
                }
            });
        }

    }// namespace

    void matMulVecBatchNVFP4_AVX512(const uint8_t *W_nvfp4, const float *X,
                                    uint32_t seqLen, uint32_t rows,
                                    uint32_t cols, float *out) {
        matMulVecBatchBlocks_AVX512<tinycoder::GGMLDequantize::QK_NVFP4,
                                    tinycoder::GGMLDequantize::NVFP4_BLOCK_BYTES>(
                W_nvfp4, X, seqLen, rows, cols, out,
                tinycoder::GGMLDequantize::dequantizeNVFP4Block);
    }

    void matMulVecBatchQ2_0_AVX512(const uint8_t *W_q2_0, const float *X,
                                   uint32_t seqLen, uint32_t rows,
                                   uint32_t cols, float *out) {
        matMulVecBatchBlocks_AVX512<tinycoder::GGMLDequantize::QK_Q2_0,
                                    tinycoder::GGMLDequantize::Q2_0_BLOCK_BYTES>(
                W_q2_0, X, seqLen, rows, cols, out,
                tinycoder::GGMLDequantize::dequantizeQ2_0Block);
    }

    // IQ1_S (1.5625 bpw, 256-weight blocks): same block decode as the AVX2
    // twin, 16-wide FMA. See the AMX note above -- a 1-bit grid plus a
    // per-group 0.125 delta still needs the fp32 reconstruct-and-scale pass,
    // which an AMX int8 tile cannot absorb.
    void matMulVecBatchIQ1S_AVX512(const uint8_t *W_iq1s, const float *X,
                                   uint32_t seqLen, uint32_t rows,
                                   uint32_t cols, float *out) {
        matMulVecBatchBlocks_AVX512<tinycoder::GGMLDequantize::QK_IQ1,
                                    tinycoder::GGMLDequantize::IQ1_S_BLOCK_BYTES>(
                W_iq1s, X, seqLen, rows, cols, out,
                tinycoder::GGMLDequantize::dequantizeIQ1_SBlock);
    }

    // ---- AVX-512 Q8_0 batch GEMM (qwen35moe routed experts) ----
    // Same contract and block math as matMulVecBatchQ8_0_Q8K_AVX2: out[s*rows +
    // row] = sum_k X[s*cols + k] * W[row*cols + k], with each token's activation
    // quantised to Q8_0 once (two-phase dispatch, rows split across the pool)
    // and every 32-weight block's exact int8 dot scaled by d * d8.
    //
    // The 512-bit part is the byte work only: two consecutive 32-weight blocks
    // share one 64-byte activation load, one sign fold and one
    // _mm512_maddubs_epi16 + _mm512_madd_epi16.  The low and high 256-bit
    // halves of the result are exactly the two blocks' int32 sums, and each is
    // scaled and reduced with the same _mm256_mul_ps(d*d8, cvt) + hsum sequence
    // the AVX2 kernel uses, in block order.  The FP accumulation order is
    // therefore identical and this kernel is bit-exact with the AVX2 one.
    // ---- AVX-512 Q8_0 4-row x 2-token tile -------------------------------
    // Same tile as the AVX2 4x2 kernel (tinyBLAS_Q0_AVX's AVX2 choice), with
    // the 512-bit op covering TWO 32-weight blocks of a token: xq is
    // token-major, so bytes [b*32, b*32+64) are blocks b and b+1.  The low and
    // high 256-bit halves are accumulated into the same (row, token)
    // accumulator in block order, so batch and sequential stay bit-identical
    // to each other and to the AVX2 kernel.
    template<uint32_t RN, uint32_t SN>
    static inline void q8DotTile512(const uint8_t *const *rowBase,
                                    const int8_t *const *xqS,
                                    const float *const *xdS,
                                    uint32_t blocksPerRow,
                                    float *const *outPtr) {
        static constexpr uint32_t BLOCK_SIZE = 32;
        static constexpr uint32_t Q8_0_BLOCK_BYTES = 34;
        __m256 acc[RN][SN];
        const __m256i one16 = _mm256_set1_epi16(1);
        const __m512i one16_512 = _mm512_set1_epi16(1);
#pragma GCC unroll 2
        for (uint32_t t = 0; t < SN; ++t)
#pragma GCC unroll 4
            for (uint32_t q = 0; q < RN; ++q)
                acc[q][t] = _mm256_setzero_ps();
        uint32_t b = 0;
        for (; b + 2u <= blocksPerRow; b += 2u) {
            __m512i xabs[SN];
            __mmask64 neg[SN];
#pragma GCC unroll 2
            for (uint32_t t = 0; t < SN; ++t) {
                const __m512i xqb = _mm512_loadu_si512(
                        reinterpret_cast<const void *>(xqS[t] +
                                                       b * BLOCK_SIZE));
                xabs[t] = _mm512_abs_epi8(xqb);
                neg[t] = _mm512_movepi8_mask(xqb);
            }
#pragma GCC unroll 4
            for (uint32_t q = 0; q < RN; ++q) {
                const uint8_t *blk0 =
                        rowBase[q] +
                        static_cast<uint64_t>(b) * Q8_0_BLOCK_BYTES;
                const uint8_t *blk1 = blk0 + Q8_0_BLOCK_BYTES;
                const float d0 = _cvtsh_ss(
                        *reinterpret_cast<const uint16_t *>(blk0));
                const float d1 = _cvtsh_ss(
                        *reinterpret_cast<const uint16_t *>(blk1));
                const __m256i w0 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(blk0 + 2));
                const __m256i w1 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(blk1 + 2));
                const __m512i wq = _mm512_inserti64x4(
                        _mm512_castsi256_si512(w0), w1, 1);
#pragma GCC unroll 2
                for (uint32_t t = 0; t < SN; ++t) {
                    const __m512i wsign = _mm512_mask_sub_epi8(
                            wq, neg[t], _mm512_setzero_si512(), wq);
                    const __m512i prod =
                            _mm512_maddubs_epi16(xabs[t], wsign);
                    const __m512i sumi =
                            _mm512_madd_epi16(prod, one16_512);
                    acc[q][t] = _mm256_fmadd_ps(
                            _mm256_set1_ps(d0 * xdS[t][b]),
                            _mm256_cvtepi32_ps(
                                    _mm512_castsi512_si256(sumi)),
                            acc[q][t]);
                    acc[q][t] = _mm256_fmadd_ps(
                            _mm256_set1_ps(d1 * xdS[t][b + 1]),
                            _mm256_cvtepi32_ps(
                                    _mm512_extracti64x4_epi64(sumi, 1)),
                            acc[q][t]);
                }
            }
        }
        for (; b < blocksPerRow; ++b) {
            __m256i xqb[SN];
            __m256i xabs[SN];
#pragma GCC unroll 2
            for (uint32_t t = 0; t < SN; ++t) {
                xqb[t] = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(xqS[t] +
                                                          b * BLOCK_SIZE));
                xabs[t] = _mm256_sign_epi8(xqb[t], xqb[t]);
            }
#pragma GCC unroll 4
            for (uint32_t q = 0; q < RN; ++q) {
                const uint8_t *block =
                        rowBase[q] +
                        static_cast<uint64_t>(b) * Q8_0_BLOCK_BYTES;
                const float d = _cvtsh_ss(
                        *reinterpret_cast<const uint16_t *>(block));
                const __m256i wq = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(block + 2));
#pragma GCC unroll 2
                for (uint32_t t = 0; t < SN; ++t) {
                    const __m256i wsign = _mm256_sign_epi8(wq, xqb[t]);
                    const __m256i prod =
                            _mm256_maddubs_epi16(xabs[t], wsign);
                    const __m256i sumi = _mm256_madd_epi16(prod, one16);
                    acc[q][t] = _mm256_fmadd_ps(
                            _mm256_set1_ps(d * xdS[t][b]),
                            _mm256_cvtepi32_ps(sumi), acc[q][t]);
                }
            }
        }
        auto hsum8 = [](__m256 v) -> float {
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                                  _mm256_extractf128_ps(v, 1));
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            return _mm_cvtss_f32(s);
        };
#pragma GCC unroll 2
        for (uint32_t t = 0; t < SN; ++t)
#pragma GCC unroll 4
            for (uint32_t q = 0; q < RN; ++q)
                outPtr[t][q] = hsum8(acc[q][t]);
    }

    void matMulVecBatchQ8_0_Q8K_AVX512(const uint8_t *W_q8_0, const float *X,
                                       uint32_t seqLen, uint32_t rows,
                                       uint32_t cols, float *out) {
        // Same llama.cpp-shaped loop as the AVX2 kernel: token-outer,
        // row-tile-inner, one accumulator per (row, token), hsum once.  The
        // 512-bit op covers TWO consecutive 32-weight blocks of the same
        // token (xq is token-major, so 64 bytes are contiguous); the low/high
        // 256-bit halves are fmadd'ed into the row's accumulator in block
        // order, which keeps it bit-exact with the AVX2 kernel.
        static constexpr uint32_t BLOCK_SIZE = 32;
        static constexpr uint32_t Q8_0_BLOCK_BYTES = 34;
        static constexpr uint32_t TILE_ROWS = 4;
        static constexpr uint32_t ROW_GROUP = 4;
        const uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const uint64_t rowStride =
                static_cast<uint64_t>(blocksPerRow) * Q8_0_BLOCK_BYTES;
        auto halfToFloat = [](uint16_t h) -> float {
            return _cvtsh_ss(h);
        };
        auto hsum8 = [](__m256 v) -> float {
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                                  _mm256_extractf128_ps(v, 1));
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            return _mm_cvtss_f32(s);
        };

        std::vector<float> xd(static_cast<size_t>(seqLen) * blocksPerRow);
        std::vector<int8_t> xq(static_cast<size_t>(seqLen) * blocksPerRow *
                               BLOCK_SIZE);
        const __m256i one16 = _mm256_set1_epi16(1);
        const __m512i one16_512 = _mm512_set1_epi16(1);
        const uint32_t numTiles = (rows + TILE_ROWS - 1) / TILE_ROWS;

        ThreadPool::current().parallelForSlab2(
                0, seqLen,
                [&](uint32_t s) {
                    const float *xs = X + static_cast<size_t>(s) * cols;
                    int8_t *xqs = xq.data() +
                                  static_cast<size_t>(s) * blocksPerRow *
                                          BLOCK_SIZE;
                    for (uint32_t b = 0; b < blocksPerRow; ++b) {
                        float amax = 0.0f, maxv = 0.0f;
                        for (uint32_t j = 0; j < BLOCK_SIZE; ++j) {
                            float ax = std::fabs(xs[b * BLOCK_SIZE + j]);
                            if (ax > amax) {
                                amax = ax;
                                maxv = xs[b * BLOCK_SIZE + j];
                            }
                        }
                        const uint32_t idx = s * blocksPerRow + b;
                        if (amax == 0.0f) {
                            xd[idx] = 0.0f;
                            std::memset(xqs + b * BLOCK_SIZE, 0, BLOCK_SIZE);
                            continue;
                        }
                        const float iscale = -127.0f / maxv;
                        for (uint32_t j = 0; j < BLOCK_SIZE; ++j) {
                            int v = static_cast<int>(std::lrintf(iscale * xs[b * BLOCK_SIZE + j]));
                            xqs[b * BLOCK_SIZE + j] = static_cast<int8_t>(std::min(127, v));
                        }
                        xd[idx] = 1.0f / iscale;
                    }
                },
                [] {},
                0, numTiles,
                [&](uint32_t tile) {
                    const uint32_t rowStart = tile * TILE_ROWS;
                    const uint32_t nRows =
                            std::min(TILE_ROWS, rows - rowStart);
                    for (uint32_t s0 = 0; s0 < seqLen; s0 += 2) {
                        const bool two = (s0 + 1 < seqLen);
                        const int8_t *xqP[2] = {
                                xq.data() +
                                        static_cast<size_t>(s0) * blocksPerRow *
                                                BLOCK_SIZE,
                                xq.data() +
                                        static_cast<size_t>(s0 + 1) *
                                                blocksPerRow * BLOCK_SIZE};
                        const float *xdP[2] = {
                                xd.data() + static_cast<size_t>(s0) * blocksPerRow,
                                xd.data() +
                                        static_cast<size_t>(s0 + 1) *
                                                blocksPerRow};
                        const uint8_t *rowBase[4];
                        for (uint32_t q = 0; q < nRows; ++q)
                            rowBase[q] = W_q8_0 +
                                         static_cast<uint64_t>(rowStart + q) *
                                                 rowStride;
                        float *dst = out + static_cast<size_t>(s0) * rows +
                                     rowStart;
                        float *outPtr2[2] = {dst, dst + rows};
                        float *outPtr1[1] = {dst};
                        if (two) {
                            switch (nRows) {
                                case 4:
                                    q8DotTile512<4, 2>(rowBase, xqP, xdP,
                                                       blocksPerRow, outPtr2);
                                    break;
                                case 3:
                                    q8DotTile512<3, 2>(rowBase, xqP, xdP,
                                                       blocksPerRow, outPtr2);
                                    break;
                                case 2:
                                    q8DotTile512<2, 2>(rowBase, xqP, xdP,
                                                       blocksPerRow, outPtr2);
                                    break;
                                default:
                                    q8DotTile512<1, 2>(rowBase, xqP, xdP,
                                                       blocksPerRow, outPtr2);
                                    break;
                            }
                        } else {
                            switch (nRows) {
                                case 4:
                                    q8DotTile512<4, 1>(rowBase, xqP, xdP,
                                                       blocksPerRow, outPtr1);
                                    break;
                                case 3:
                                    q8DotTile512<3, 1>(rowBase, xqP, xdP,
                                                       blocksPerRow, outPtr1);
                                    break;
                                case 2:
                                    q8DotTile512<2, 1>(rowBase, xqP, xdP,
                                                       blocksPerRow, outPtr1);
                                    break;
                                default:
                                    q8DotTile512<1, 1>(rowBase, xqP, xdP,
                                                       blocksPerRow, outPtr1);
                                    break;
                            }
                        }
                    }
                });
    }

    // ---- Fused Q8_0 expert groups (AVX-512 counterpart) -------------------
    // Same one-op-per-layer shape as the AVX2 kernel; the 4x2 tile uses the
    // 512-bit two-block op inside q8DotTile512.
    bool matMulVecBatchQ8_0Groups_AVX512(
            const uint8_t *W_q8_0, uint32_t rowsPerExpert, uint64_t rowBytes,
            const uint32_t *groupExpert, const uint32_t *groupFirst,
            const uint32_t *groupCount, uint32_t nGroups,
            const uint32_t *memberPair, const uint32_t *memberXRow,
            uint32_t xRows, const float *X, uint32_t cols, float *out) {
        static constexpr uint32_t BLOCK_SIZE = 32;
        static constexpr uint32_t TILE_ROWS = 4;
        const uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const uint32_t tilesPerExpert =
                (rowsPerExpert + TILE_ROWS - 1) / TILE_ROWS;
        const uint32_t numTasks = nGroups * tilesPerExpert;
        if (numTasks == 0 || xRows == 0) return true;

        std::vector<float> xd(static_cast<size_t>(xRows) * blocksPerRow);
        std::vector<int8_t> xq(static_cast<size_t>(xRows) * blocksPerRow *
                               BLOCK_SIZE);
        auto groupsPhase1 = [&](uint32_t m) {
            const float *xs =
                    X + static_cast<size_t>(memberXRow[m]) * cols;
            int8_t *xqs = xq.data() +
                          static_cast<size_t>(m) * blocksPerRow *
                                  BLOCK_SIZE;
            for (uint32_t b = 0; b < blocksPerRow; ++b) {
                float amax = 0.0f, maxv = 0.0f;
                for (uint32_t j = 0; j < BLOCK_SIZE; ++j) {
                    float ax = std::fabs(xs[b * BLOCK_SIZE + j]);
                    if (ax > amax) {
                        amax = ax;
                        maxv = xs[b * BLOCK_SIZE + j];
                    }
                }
                const uint32_t idx = m * blocksPerRow + b;
                if (amax == 0.0f) {
                    xd[idx] = 0.0f;
                    std::memset(xqs + b * BLOCK_SIZE, 0, BLOCK_SIZE);
                    continue;
                }
                const float iscale = -127.0f / maxv;
                for (uint32_t j = 0; j < BLOCK_SIZE; ++j) {
                    int v = static_cast<int>(
                            std::lrintf(iscale * xs[b * BLOCK_SIZE + j]));
                    xqs[b * BLOCK_SIZE + j] =
                            static_cast<int8_t>(std::min(127, v));
                }
                xd[idx] = 1.0f / iscale;
            }
        };
        auto groupsPhase2 = [&](uint32_t task) {
            const uint32_t g = task / tilesPerExpert;
            const uint32_t tile = task % tilesPerExpert;
            const uint32_t rowStart = tile * TILE_ROWS;
            const uint32_t nRows =
                    std::min(TILE_ROWS, rowsPerExpert - rowStart);
            const uint32_t first = groupFirst[g];
            const uint32_t count = groupCount[g];
            const uint8_t *base =
                    W_q8_0 +
                    (static_cast<uint64_t>(groupExpert[g]) *
                             rowsPerExpert +
                     rowStart) *
                            rowBytes;
            const uint8_t *rowBase[4];
            for (uint32_t q = 0; q < nRows; ++q)
                rowBase[q] = base + static_cast<uint64_t>(q) * rowBytes;
            for (uint32_t m = 0; m < count; m += 2) {
                const uint32_t p0 = memberPair[first + m];
                const uint32_t x0 = first + m;
                if (m + 1 < count) {
                    const uint32_t p1 = memberPair[first + m + 1];
                    const uint32_t x1 = first + m + 1;
                    const int8_t *xqP[2] = {
                            xq.data() +
                                    static_cast<size_t>(x0) *
                                            blocksPerRow * BLOCK_SIZE,
                            xq.data() +
                                    static_cast<size_t>(x1) *
                                            blocksPerRow * BLOCK_SIZE};
                    const float *xdP[2] = {
                            xd.data() +
                                    static_cast<size_t>(x0) * blocksPerRow,
                            xd.data() +
                                    static_cast<size_t>(x1) * blocksPerRow};
                    float *outP[2] = {
                            out + static_cast<size_t>(p0) * rowsPerExpert +
                                    rowStart,
                            out + static_cast<size_t>(p1) * rowsPerExpert +
                                    rowStart};
                    switch (nRows) {
                        case 4:
                            q8DotTile512<4, 2>(rowBase, xqP, xdP,
                                               blocksPerRow, outP);
                            break;
                        case 3:
                            q8DotTile512<3, 2>(rowBase, xqP, xdP,
                                               blocksPerRow, outP);
                            break;
                        case 2:
                            q8DotTile512<2, 2>(rowBase, xqP, xdP,
                                               blocksPerRow, outP);
                            break;
                        default:
                            q8DotTile512<1, 2>(rowBase, xqP, xdP,
                                               blocksPerRow, outP);
                            break;
                    }
                } else {
                    const int8_t *xqP[1] = {
                            xq.data() +
                            static_cast<size_t>(x0) * blocksPerRow *
                                    BLOCK_SIZE};
                    const float *xdP[1] = {
                            xd.data() +
                            static_cast<size_t>(x0) * blocksPerRow};
                    float *outP[1] = {
                            out + static_cast<size_t>(p0) * rowsPerExpert +
                            rowStart};
                    switch (nRows) {
                        case 4:
                            q8DotTile512<4, 1>(rowBase, xqP, xdP,
                                               blocksPerRow, outP);
                            break;
                        case 3:
                            q8DotTile512<3, 1>(rowBase, xqP, xdP,
                                               blocksPerRow, outP);
                            break;
                        case 2:
                            q8DotTile512<2, 1>(rowBase, xqP, xdP,
                                               blocksPerRow, outP);
                            break;
                        default:
                            q8DotTile512<1, 1>(rowBase, xqP, xdP,
                                               blocksPerRow, outP);
                            break;
                    }
                }
            }
        };
        if (nGroups > 32) {
            ThreadPool::current().parallelForSteal2(
                    0, xRows, 4, groupsPhase1, [] {}, 0, numTasks, 1,
                    groupsPhase2);
        } else {
            ThreadPool::current().parallelForSlab2(
                    0, xRows, groupsPhase1, [] {}, 0, numTasks, groupsPhase2);
        }
        return true;
    }

}// namespace tinycoder::simd
#endif// __AVX512F__ && __FMA__
