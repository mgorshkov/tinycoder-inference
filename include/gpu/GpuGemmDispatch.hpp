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

// GpuGemmDispatch.hpp -- kDequantF16 and the launchQGemv* dispatchers.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// ---- Fused gate+up+silu dispatcher (decode seqLen==1, FFN).
// Replaces launchQGemv(gate) + launchQGemv(up) + kSiluMul with ONE
// launch.  `wg`/`wu` are the gate/up row-major weight bases and both
// matrices MUST be identical (rows, cols, rowBytes, blocksPerRow) --
// guaranteed by the Model layer for the dense Qwen2 FFN gate/up pair.
// ffnRows = rows (gate rows == up rows).  out[ffnRows] receives
// silu(gate(x))*up(x) -- the same product that kSiluMul was computing
// into s.gate from the two separate launchQGemv results.  The raw up
// row values, and s.up, are dead after this (the down QGEMV consumes
// out), so they are not written.
void launchQGemvFusedGU(int type, const void *wg, const void *wu,
                        const float *x, float *out, uint32_t ffnRows,
                        uint32_t cols, uint32_t rowBytes,
                        uint32_t blocksPerRow, uint8_t *&q8k,
                        uint64_t &q8kBytes,
                        const uint8_t *preQ81 = nullptr) {
    dim3 block(32, 8);
    uint32_t nBlocks = (ffnRows + 7) / 8;
    const uint32_t rows = ffnRows;
    if (type == kTypeQ2K) {
        // Q2_K FFN gate+up.  Two routes:
        //   * TINYCODER_Q2K_INTGU=1 (2026-09-30, lever-1 A/B, DEFAULT
        //     OFF): reference-int8-style integer path -- kQuantizeQ8K
        //     the fp32 activation ONCE, then BOTH row dots run as
        //     Q2_K-int x Q8_K-int DP4A-style dots (kQGemvQ2KIntFusedGU)
        //     against the shared Q8_K activation, replacing the
        //     ALU-bound float dequant.  Bit-exact int side with the
        //     CPU reference dotProductQ2_K_PrePacked_Q8_Scalar / the
        //     compact AVX2 kernel; float fold in serial block order
        //     like kQGemvQ8KFusedGU.
        //   * otherwise (DEFAULT): float dequant-dot path, computes
        //     both row dots + the silu product in ONE launch (was 3).
        //     Bit-exact with kQGemv<kTypeQ2K> + kSiluMul.
        const char *q2kInt = std::getenv("TINYCODER_Q2K_INTGU");
        if (q2kInt != nullptr && std::atoi(q2kInt) != 0) {
            const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                                  static_cast<uint64_t>(kQ8K_STRIDE);
            if (q8k == nullptr || q8kBytes < need) {
                if (q8k) cudaFree(q8k);
                q8k = nullptr;
                q8kBytes = 0;
                cudaError_t e = cudaMalloc(&q8k, need);
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                                 cudaGetErrorString(e));
                    cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                    g_stream);
                    return;
                }
                q8kBytes = need;
            }
            kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(
                    x, q8k, cols);
            kQGemvQ2KIntFusedGU<<<nBlocks, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wg),
                    static_cast<const uint8_t *>(wu), q8k, out, rows,
                    blocksPerRow, rowBytes);
            return;
        }
        // TINYCODER_FUSE_GU_Q2K=2 (2026-10-10): reference-style fused
        // GLU on the dp4a Q8_1 path -- ONE launch quantizes the fp32
        // activation to Q8_1 (the SAME kQuantizeQ8_1 the down gemv
        // uses), then kQGemvQ2KxQ81_GLU computes both weight dots with
        // the shared Q8_1 operand and writes silu(gate)*up. Default = 2
        // when the env is unset (see the call-site comment); the
        // kernel's 504-B row stride is hardcoded for cols==1536
        // (blocksPerRow==6), so anything else falls to the float path.
        const char *fgMode = std::getenv("TINYCODER_FUSE_GU_Q2K");
        const int gluMode = (fgMode != nullptr) ? std::atoi(fgMode) : 2;
        if (gluMode == 2 && (cols & 31u) == 0u &&
            rowBytes == 6u * 84u) {
            const uint64_t need =
                    static_cast<uint64_t>(blocksPerRow) * 8u *
                    kQ8_1_STRIDE;
            // preQ81 (2026-10-10, plan section 9.26 lever 1): the FFN
            // rms_norm epilogue already wrote the Q8_1 activation
            // blocks -- skip the allocation guard and the separate
            // kQuantizeQ8_1 launch entirely.
            const uint8_t *y1;
            if (preQ81 != nullptr) {
                y1 = preQ81;
            } else {
                if (q8k == nullptr || q8kBytes < need) {
                    if (q8k) cudaFree(q8k);
                    q8k = nullptr;
                    q8kBytes = 0;
                    cudaError_t e = cudaMalloc(&q8k, need);
                    if (e != cudaSuccess) {
                        std::fprintf(stderr,
                                     "cudaMalloc(q8_1 glu): %s\n",
                                     cudaGetErrorString(e));
                        cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                        g_stream);
                        return;
                    }
                    q8kBytes = need;
                }
                kQuantizeQ8_1<<<cols / 32u, 32, 0, g_stream>>>(x, q8k,
                                                               cols);
                y1 = q8k;
            }
            if (gemvWide())
                kQGemvQ2KxQ81_GLU<true><<<rows, dim3(32, 2), 0,
                                          g_stream>>>(
                        static_cast<const uint8_t *>(wg),
                        static_cast<const uint8_t *>(wu), y1, out,
                        rows, blocksPerRow);
            else
                kQGemvQ2KxQ81_GLU<false><<<rows, dim3(32, 2), 0,
                                           g_stream>>>(
                        static_cast<const uint8_t *>(wg),
                        static_cast<const uint8_t *>(wu), y1, out,
                        rows, blocksPerRow);
            return;
        }
        kQGemvFusedGU_Q2K<<<nBlocks, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wg),
                static_cast<const uint8_t *>(wu), x, out, rows, cols,
                rowBytes, blocksPerRow);
        return;
    }
    if (type == kTypeIQ2XS && !iq2xsInt()) {
        // Legacy float dequant-dot path (A/B only): no quantize
        // needed; reads x directly.
        kQGemvFusedGU_IQ2XS<<<nBlocks, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wg),
                static_cast<const uint8_t *>(wu), x, out, rows, cols,
                rowBytes, blocksPerRow);
        return;
    }
    if (type == kTypeIQ2S || (type == kTypeIQ2XS && iq2xsInt()) ||
        type == kTypeIQ3_XXS || type == kTypeIQ3S) {
        // Q8_K integer path: quantize the activation ONCE, share it.
        const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                              static_cast<uint64_t>(kQ8K_STRIDE);
        if (q8k == nullptr || q8kBytes < need) {
            if (q8k) cudaFree(q8k);
            q8k = nullptr;
            q8kBytes = 0;
            cudaError_t e = cudaMalloc(&q8k, need);
            if (e != cudaSuccess) {
                std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                             cudaGetErrorString(e));
                cudaMemsetAsync(out, 0, sizeof(float) * rows, g_stream);
                return;
            }
            q8kBytes = need;
        }
        kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(x, q8k, cols);
        const bool d4 = iqDp4a();
        const bool grsm = iqGridSmem();
#define TC_FUSED_GU(T, GB)                                              \
    do {                                                                \
        if (d4)                                                         \
            kQGemvQ8KFusedGU<T, true>                                   \
                    <<<nBlocks, block, grsm ? (GB) : 0u, g_stream>>>(   \
                            static_cast<const uint8_t *>(wg),           \
                            static_cast<const uint8_t *>(wu), q8k, out, \
                            rows, blocksPerRow, rowBytes, grsm);        \
        else                                                            \
            kQGemvQ8KFusedGU<T, false>                                  \
                    <<<nBlocks, block, grsm ? (GB) : 0u, g_stream>>>(   \
                            static_cast<const uint8_t *>(wg),           \
                            static_cast<const uint8_t *>(wu), q8k, out, \
                            rows, blocksPerRow, rowBytes, grsm);        \
    } while (0)
        if (type == kTypeIQ2S)
            TC_FUSED_GU(kTypeIQ2S, 8192u);
        else if (type == kTypeIQ2XS)
            TC_FUSED_GU(kTypeIQ2XS, 4096u);
        else if (type == kTypeIQ3_XXS)
            TC_FUSED_GU(kTypeIQ3_XXS, 1024u);
        else
            TC_FUSED_GU(kTypeIQ3S, 2048u);
#undef TC_FUSED_GU
        return;
    }
    // Unsupported pair: fall back to the two separate kernels + silu
    // (the caller only takes this path when gate/up types match a
    // supported fused type, so this is a bare safety net).
    launchQGemv(type, wg, x, out, rows, cols, rowBytes, blocksPerRow,
                q8k, q8kBytes);
}

// Q8_0 decode-helper split (added 2026-09-13): the decode expert path
// runs up to 8 experts per layer, each with gate/up/down GEMVs against
// the SAME fp32 activation (s.norm).  The old path re-quantized
// s.norm for every (expert, matrix) pair -- ~1920 quantize+GEMV
// kernel launches per decode token (8 ranks x 3 matrices x 40 layers),
// each kernel under-occupying the GPU.  Split the Q8_0 path so the
// caller quantizes the activation ONCE per layer into the shared q8k_
// scratch and runs the pre-quantized GEMVs.

// Quantize x (cols floats) into the caller-supplied q8k scratch using
// the Q8_32 layout (kQ8_0x32_STRIDE=64).  grid = blocksPerRow.
void launchQuantQ8_0x32(const float *x, uint8_t *q8k, uint32_t cols) {
    const uint32_t blocksPerRow = (cols + 31) / 32;
    kQuantizeQ8_0x32<<<blocksPerRow, 32, 0, g_stream>>>(x, q8k, cols);
}

// Integer Q8_0 GEMV against a PRE-QUANTIZED activation (the Q8_32
// layout from launchQuantQ8_0x32).  Bit-exact with the CPU reference:
// same float tree, same op order, no fmaf -- identical to
// launchQGemv's Q8_0 branch WITHOUT the re-quantization.
void launchQGemvQ8_0Pre(const void *wq, const uint8_t *q8k, float *out,
                        uint32_t rows, uint32_t blocksPerRow,
                        uint32_t rowBytes) {
    dim3 block(32, 8);
    uint32_t nBlocks = (rows + 7) / 8;
    kQGemvQ8_0xQ8K<<<nBlocks, block, 0, g_stream>>>(
            static_cast<const uint8_t *>(wq), q8k, out, rows,
            blocksPerRow, rowBytes);
}

// Dequant ONE [rows][cols] quantized matrix into the reusable wF16_
// scratch (fp16, row-major).  The kernel uses ONE warp per row (the
// 32 lanes cover all 256 columns of a K-quant block via lane*8), so
// grid = rows, block = 32.
void launchDequantF16(int type, const void *wq, void *out,
                      uint32_t rows, uint32_t cols, uint32_t rowBytes,
                      uint32_t blocksPerRow,
                      cudaStream_t stream = nullptr) {
    cudaStream_t ds = stream != nullptr ? stream : g_stream;
    dim3 blk(32);
    if (type == kTypeQ2K) {
        kDequantF16<kTypeQ2K><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeQ3K) {
        kDequantF16<kTypeQ3K><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeQ4K) {
        kDequantF16<kTypeQ4K><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeQ6K) {
        kDequantF16<kTypeQ6K><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeIQ2XS) {
        kDequantF16<kTypeIQ2XS><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeIQ2XXS) {
        kDequantF16<kTypeIQ2XXS><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeIQ4XS) {
        kDequantF16<kTypeIQ4XS><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeIQ3S) {
        kDequantF16<kTypeIQ3S><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeIQ2S) {
        kDequantF16<kTypeIQ2S><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeIQ3_XXS) {
        kDequantF16<kTypeIQ3_XXS><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeQ5K) {
        kDequantF16<kTypeQ5K><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeQ5_0) {
        kDequantF16<kTypeQ5_0><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeQ8_0) {
        kDequantF16Q8<<<rows, 256, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else if (type == kTypeIQ4NL) {
        kDequantF16<kTypeIQ4NL><<<rows, blk, 0, ds>>>(
                static_cast<const uint8_t *>(wq),
                static_cast<__half *>(out), rows, cols, blocksPerRow,
                rowBytes);
    } else {
        cudaMemsetAsync(out, 0,
                        static_cast<size_t>(rows) * cols * sizeof(__half),
                        ds);
    }
}

// Token embedding dequant (quantized Q2_K rows, one warp per token).
__global__ void kEmbedDequant(const uint8_t *__restrict__ embed,
                              const int32_t *__restrict__ tokens,
                              float *__restrict__ hidden, uint32_t seqLen,
                              uint32_t hiddenSize, uint32_t blocksPerRow,
                              uint32_t rowBytes, uint32_t vocabSize) {
    uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    int32_t t = tokens[s];
    if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
    const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
    float *h = hidden + static_cast<size_t>(s) * hiddenSize;
    const uint32_t lane = threadIdx.x;
    const uint32_t e0 = lane * 8u;
    const uint32_t half = e0 / 128u, jj = (e0 % 128u) / 32u,
                   sub = (e0 % 32u) / 16u, l16 = e0 % 16u;
    const uint32_t is = half * 8u + jj * 2u + sub;
    // q indexing matches dequantizeQ2_KBlock: same 32 q bytes per half,
    // reused for all four shifts.
    const uint32_t qbase = half * 32u + sub * 16u + l16;
    const uint32_t shift = jj * 2u;
    const uint32_t typeBytes = rowBytes / blocksPerRow;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        __half d = *reinterpret_cast<const __half *>(row + 80);
        __half dmin = *reinterpret_cast<const __half *>(row + 82);
        float df = __half2float(d), dmf = __half2float(dmin);
        float dl = df * static_cast<float>(row[is] & 0xF);
        float ml = dmf * static_cast<float>(row[is] >> 4);
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) {
            // Bound the write to the row width.  blocksPerRow pads up
            // to the next 256-col block, so the last block's columns
            // [b*256, b*256+256) can exceed hiddenSize; writing past it
            // corrupts the next token's row (or the buffer end) and
            // triggers an illegal memory access on non-256-multiple
            // hidden sizes.
            const uint32_t col = b * 256u + e0 + k;
            if (col < hiddenSize) {
                float qv = static_cast<float>(static_cast<int8_t>(
                        ((row[16 + qbase + k]) >> shift) & 3));
                h[col] = fmaf(dl, qv, -ml);
            }
        }
        row += typeBytes;
    }
}

// Token embedding copy for a plain F16 checkpoint table (EXL3 keeps
// the embedding as BF16->F16 rather than trellis-quantized).
__global__ void kEmbedF16(const uint8_t *__restrict__ embed,
                          const int32_t *__restrict__ tokens,
                          float *__restrict__ hidden, uint32_t seqLen,
                          uint32_t hiddenSize, uint32_t rowBytes,
                          uint32_t vocabSize) {
    const uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    const int32_t t = tokens[s];
    if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
    const __half *row = reinterpret_cast<const __half *>(
            embed + static_cast<uint64_t>(t) * rowBytes);
    float *h = hidden + static_cast<size_t>(s) * hiddenSize;
    for (uint32_t i = threadIdx.x; i < hiddenSize; i += blockDim.x) {
        h[i] = __half2float(row[i]);
    }
}

// Token embedding dequant for IQ3_S rows (one warp per token).  Layout
// matches dequantizeIQ3_SBlock (110 B blocks).  Each lane owns 8
// consecutive columns; exactly one (ib32, l) group.  The sign byte
// for a (ib32, l) group is signs[pair*8 + half*4 + l] and the qh byte
// is qh[pair*2 + half] -- both advance as in the reference.
__global__ void kEmbedDequantIQ3S(const uint8_t *__restrict__ embed,
                                  const int32_t *__restrict__ tokens,
                                  float *__restrict__ hidden,
                                  uint32_t seqLen, uint32_t hiddenSize,
                                  uint32_t blocksPerRow, uint32_t rowBytes,
                                  uint32_t vocabSize) {
    uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    int32_t t = tokens[s];
    if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
    const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
    float *h = hidden + static_cast<size_t>(s) * hiddenSize;
    const uint32_t lane = threadIdx.x;
    const uint32_t e0 = lane * 8u;
    const uint32_t ib32 = e0 >> 5u;
    const uint32_t half = ib32 & 1u;
    const uint32_t pair = ib32 >> 1u;
    const uint32_t l = (e0 >> 3u) & 3u;
    const uint32_t qhIdx = pair * 2u + half;
    const uint32_t qsOff = pair * 16u + half * 8u;
    const uint32_t signOff = pair * 8u + half * 4u;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        __half d = *reinterpret_cast<const __half *>(row + 0);
        float df = __half2float(d);
        float db = df *
                   (1.0f +
                    2.0f *
                            static_cast<float>((half == 0)
                                                       ? (row[106 + pair] & 0xf)
                                                       : (row[106 + pair] >> 4)));
        const uint16_t gridIdx1 = static_cast<uint16_t>(
                row[2 + qsOff + 2u * l] |
                ((row[66 + qhIdx] << (8 - 2u * l)) & 256));
        const uint16_t gridIdx2 = static_cast<uint16_t>(
                row[2 + qsOff + 2u * l + 1u] |
                ((row[66 + qhIdx] << (7 - 2u * l)) & 256));
        const uint32_t g1 = c_iq3s_grid[gridIdx1];
        const uint32_t g2 = c_iq3s_grid[gridIdx2];
        const uint8_t sbyte = row[74 + signOff + l];
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t col = b * 256u + e0 + j;
            if (col >= hiddenSize) continue;
            uint32_t gval = (j < 4u)
                                    ? ((g1 >> (8u * j)) & 0xFFu)
                                    : ((g2 >> (8u * (j - 4u))) & 0xFFu);
            float w = db * static_cast<float>(gval) *
                      ((sbyte & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
            h[col] = w;
        }
        row += kIQ3S_BYTES;
    }
}

// Token embedding dequant for Q5_K rows (one warp per token).  Layout
// matches dequantizeQ5_KBlock (176 B blocks).  Each lane owns 8
// consecutive columns within one 32-weight sub-block (e0 is a multiple
// of 8), so the qs/qh byte is g*32 + jj and the sub-block scale/min
// come from getScaleMin(2*g + sub).
__global__ void kEmbedDequantQ5K(const uint8_t *__restrict__ embed,
                                 const int32_t *__restrict__ tokens,
                                 float *__restrict__ hidden,
                                 uint32_t seqLen, uint32_t hiddenSize,
                                 uint32_t blocksPerRow, uint32_t rowBytes,
                                 uint32_t vocabSize) {
    uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    int32_t t = tokens[s];
    if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
    const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
    float *h = hidden + static_cast<size_t>(s) * hiddenSize;
    const uint32_t lane = threadIdx.x;
    const uint32_t e0 = lane * 8u;
    const uint32_t g = e0 / 64u;
    const uint32_t sub = (e0 / 32u) & 1u;
    const uint32_t jj0 = e0 & 31u;
    auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                          uint8_t *m_out) {
        if (j < 4) {
            *d_out = q[j] & 63;
            *m_out = q[j + 4] & 63;
        } else {
            *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
        }
    };
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        __half d = *reinterpret_cast<const __half *>(row + 0);
        __half dmin = *reinterpret_cast<const __half *>(row + 2);
        float df = __half2float(d), dmf = __half2float(dmin);
        const uint8_t *scales = row + 4;
        const uint8_t *qh = row + 16;
        const uint8_t *qs = row + 48;
        uint8_t sc, m;
        getScaleMin(static_cast<int>(2u * g + sub), scales, &sc, &m);
        float dl = df * static_cast<float>(sc);
        float ml = dmf * static_cast<float>(m);
        const uint8_t u =
                (sub == 0) ? static_cast<uint8_t>(1u << (2u * g))
                           : static_cast<uint8_t>(2u << (2u * g));
        const uint8_t *qsp = qs + g * 32u;
#pragma unroll
        for (uint32_t kk = 0; kk < 8; ++kk) {
            uint8_t q5 = (sub == 0)
                                 ? static_cast<uint8_t>(
                                           (qsp[jj0 + kk] & 0xF) |
                                           ((qh[jj0 + kk] & u) ? 16 : 0))
                                 : static_cast<uint8_t>(
                                           (qsp[jj0 + kk] >> 4) |
                                           ((qh[jj0 + kk] & u) ? 16 : 0));
            const uint32_t col = b * 256u + e0 + kk;
            if (col < hiddenSize) h[col] = dl * static_cast<float>(q5) - ml;
        }
        row += kQ5K_BYTES;
    }
}

// Token embedding dequant for Q4_K rows (one warp per token).  Layout
// matches dequantizeQ4_KBlock (144 B blocks: d, dmin, scales[12], qs[128]
// = 2+2+12+128).  Each lane owns 8 consecutive columns within one
// 32-weight sub-block (e0 is a multiple of 8), so the qs byte is
// g*32 + jj0 + k and the sub-block scale/min come from
// getScaleMin(2*g + sub) -- identical decomposition to kEmbedDequantQ5K.
__global__ void kEmbedDequantQ4K(const uint8_t *__restrict__ embed,
                                 const int32_t *__restrict__ tokens,
                                 float *__restrict__ hidden,
                                 uint32_t seqLen, uint32_t hiddenSize,
                                 uint32_t blocksPerRow, uint32_t rowBytes,
                                 uint32_t vocabSize) {
    uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    int32_t t = tokens[s];
    if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
    const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
    float *h = hidden + static_cast<size_t>(s) * hiddenSize;
    const uint32_t lane = threadIdx.x;
    const uint32_t e0 = lane * 8u;
    const uint32_t g = e0 / 64u;
    const uint32_t sub = (e0 / 32u) & 1u;
    const uint32_t jj0 = e0 & 31u;
    auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                          uint8_t *m_out) {
        if (j < 4) {
            *d_out = q[j] & 63;
            *m_out = q[j + 4] & 63;
        } else {
            *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
        }
    };
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        __half d = *reinterpret_cast<const __half *>(row + 0);
        __half dmin = *reinterpret_cast<const __half *>(row + 2);
        float df = __half2float(d), dmf = __half2float(dmin);
        const uint8_t *scales = row + 4;
        const uint8_t *qs = row + 16;
        uint8_t sc, m;
        getScaleMin(static_cast<int>(2u * g + sub), scales, &sc, &m);
        float dl = df * static_cast<float>(sc);
        float ml = dmf * static_cast<float>(m);
        const uint8_t *qsp = qs + g * 32u;
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) {
            const uint8_t lo = (sub == 0) ? (qsp[jj0 + k] & 0xF)
                                          : (qsp[jj0 + k] >> 4);
            const uint32_t col = b * 256u + e0 + k;
            if (col < hiddenSize) h[col] = fmaf(dl, static_cast<float>(lo), -ml);
        }
        row += kQ4K_BYTES;
    }
}

// Token embedding dequant for Q6_K rows (one warp per token).
//
// Q6_K is the type the rented 3090 report caught going to the CPU path: the
// support gate listed every other embedding type, so a bartowski Q2_K file
// (token_embd stored as Q6_K) silently ran at ~20 tok/s on 12 CPU threads.
// Layout matches GGMLDequantize case 14 / dequantize_row_q6_K: ql[128] holds
// the low nibbles, qh[64] the two high bits (shift 2*g), scales[16] one int8
// per 16 values, d at byte 208.  A lane owns 8 consecutive columns, which sit
// inside one 16-value sub-block, so the scale is constant across them.
__global__ void kEmbedDequantQ6K(const uint8_t *__restrict__ embed,
                                 const int32_t *__restrict__ tokens,
                                 float *__restrict__ hidden,
                                 uint32_t seqLen, uint32_t hiddenSize,
                                 uint32_t blocksPerRow, uint32_t rowBytes,
                                 uint32_t vocabSize) {
    const uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    const int32_t t = tokens[s];
    if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
    const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
    float *h = hidden + static_cast<size_t>(s) * hiddenSize;
    const uint32_t lane = threadIdx.x;
    const uint32_t e0 = lane * 8u;          // 0..248 inside the 256-value block
    const uint32_t half = e0 / 128u;        // the block holds two 128-value halves
    const uint32_t r = e0 % 128u;           // position inside the half
    const uint32_t g = r / 32u;             // which qh shift / ql pair
    const uint32_t sub = (r % 32u) / 16u;   // 16-value sub-block inside the group
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        // Per-block pointers: every 210-byte block carries its own ql/qh/scales/d.
        // Hoisting them out of the loop made blocks 1..n-1 decode block 0's bytes
        // (caught by Q6KEmbeddingGatherMatchesDequant at ~0.11 absolute error on
        // 15/16 of a 4096-wide row).
        const int8_t *sc = reinterpret_cast<const int8_t *>(row + 192);
        const uint8_t *ql = row;            // 128 bytes
        const uint8_t *qh = row + 128u;     // 64 bytes
        const float d = __half2float(*reinterpret_cast<const __half *>(row + 208));
        const float scale = d * static_cast<float>(sc[half * 8u + 2u * g + sub]);
        const uint8_t *qlp = ql + half * 64u + (g & 1u) * 32u + (r % 32u);
        const uint8_t *qhp = qh + half * 32u + (r % 32u);
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) {
            const uint32_t nib = (g < 2u) ? (qlp[k] & 0xFu) : (qlp[k] >> 4);
            const uint32_t hi = (qhp[k] >> (2u * g)) & 3u;
            const int q = static_cast<int>(nib | (hi << 4)) - 32;
            const uint32_t col = b * 256u + e0 + k;
            if (col < hiddenSize) h[col] = scale * static_cast<float>(q);
        }
        row += kQ6K_BYTES;
    }
}

// Token embedding dequant for the 32-wide-block legacy quants
// (Q5_0 / Q8_0): one warp per token, each lane writes its own column
// (b*32 + lane) of every block.  blocksPerRow = ceil(hiddenSize/32),
// rowBytes = blocksPerRow * blockBytes.
template<int TYPE>
__global__ void kEmbedDequantSmall32(const uint8_t *__restrict__ embed,
                                     const int32_t *__restrict__ tokens,
                                     float *__restrict__ hidden,
                                     uint32_t seqLen, uint32_t hiddenSize,
                                     uint32_t blocksPerRow, uint32_t rowBytes,
                                     uint32_t vocabSize) {
    uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    int32_t t = tokens[s];
    if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
    const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
    float *h = hidden + static_cast<size_t>(s) * hiddenSize;
    const uint32_t lane = threadIdx.x;
    constexpr uint32_t BLOCK_BYTES = (TYPE == kTypeQ5_0)
                                             ? kQ5_0_BYTES
                                     : (TYPE == kTypeQ8_0) ? kQ8_0_BYTES
                                                           : kIQ4NL_BYTES;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        const uint8_t *blk = row + static_cast<uint64_t>(b) * BLOCK_BYTES;
        const uint32_t col = b * 32u + lane;
        if (col >= hiddenSize) break;
        __half d = *reinterpret_cast<const __half *>(blk);
        if (TYPE == kTypeQ8_0) {
            const int8_t q = reinterpret_cast<const int8_t *>(blk + 2)[lane];
            h[col] = __half2float(d) * static_cast<float>(q);
        } else if (TYPE == kTypeIQ4NL) {
            // IQ4_NL (4.5 bpw): d(fp16,2) + qs[16].  Element `lane`:
            //   j   = lane & 15
            //   nib = (lane < 16) ? qs[j] & 0x0F : qs[j] >> 4
            //   wv  = d * kvalues_iq4nl[nib]   (signed 8-bit table)
            // Matches CPU dequantizeIQ4_NLBlock.
            const uint8_t *qs = blk + 2;
            const uint32_t j = lane & 15u;
            const uint8_t nib =
                    (qs[j] >> (4u * (lane >> 4u))) & 0xFu;
            h[col] = __half2float(d) *
                     static_cast<float>(c_kvalues_iq4nl[nib]);
        } else {// Q5_0 (matches CPU dequantizeQ5_0Block)
            // Element `lane` of the 32-wide block (QK5_0=32):
            //   j      = lane & 15
            //   low    = (lane < 16) ? qs[j] & 0x0F : qs[j] >> 4
            //   high   = bit j of qh for lane < 16, else bit (j+16)
            //   val    = (low | (high << 4)) - 16
            // NOTE: the CPU reference reads the upper-half high bit via
            // `((qh >> (j + 12)) & 0x10)` -- the 0x10 mask selects bit 4
            // of the shifted word, i.e. ORIGINAL bit (j+16) of qh, NOT
            // bit (j+12) (the surrounding comment in GGMLDequantize.hpp
            // is misleading).  Use (j+16) here to match the CPU math.
            uint32_t qh;
            std::memcpy(&qh, blk + 2, sizeof(uint32_t));
            const uint8_t *qs = blk + 6;
            const uint32_t j = lane & 15u;
            const uint8_t low =
                    (qs[j] >> (4u * (lane >> 4u))) & 0xFu;
            const uint32_t hb = (lane & 16u) ? (j + 16u) : j;
            const uint8_t high =
                    static_cast<uint8_t>((qh >> hb) & 1u);
            const int val = static_cast<int>(low | (high << 4)) - 16;
            h[col] = __half2float(d) * static_cast<float>(val);
        }
    }
}

// Number of blocks per row for a given quant type and row width, and
// the corresponding row byte stride.  Mirrors the host-side layout in
// ModelGPU.cpp so both sides agree on where each block lives.
uint32_t hostBlocksPerRow(uint32_t type, uint32_t cols) {
    if (type == kTypeQ5_0 || type == kTypeQ8_0 ||
        type == kTypeIQ4NL) {
        return (cols + 31) / 32;// 32-wide legacy blocks
    }
    if (type == kTypePQ2_0 || type == kTypePTQ1_0) {
        return (cols + 127) / 128;// Prism ternary 128-wide blocks
    }
    return (cols + 255) / 256;// K-quant blocks (256 wide)
}
