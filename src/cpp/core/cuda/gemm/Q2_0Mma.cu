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

// Q2_0Mma.cu — int8 tensor-core MMQ for the Inkling Q2_0 expert-down weights.
//
// Q2_0 is 18 B per 64 weights: one fp16 scale, then 16 bytes of 2-bit codes in
// element order, value = d * (code - 1).  The expert down projection is
//   gate_up_exps : [nExperts][1152][2304]  (IQ2_S / ...)
//   down_exps    : [nExperts][2304][1152]  Q2_0, i.e. rows=2304, cols=1152.
// The batched prefill path groups the m*nUsed routed pairs by expert and runs
// one grouped float GEMV per distinct expert; that path decodes every weight
// block to float.  This kernel instead decodes each 18-byte block to int8 once
// per 64-k tile and feeds it to mma.s8, with the fp32 scale applied in the
// epilogue.
//
// Shape notes:
//   * cols is 1152, not a multiple of 512.  This kernel never assumes the
//     engine's Q8_1 512-value row padding: it pads nothing, and K is consumed
//     in 64-weight units (one Q2_0 block), so cols % 64 == 0 is the only
//     requirement.
//   * The output tile is 16 rows (tokens) x 128 columns.  rows=2304 gives 18
//     column tiles per expert, which fills the 68-SM grid with 128 experts.
//   * Turing has no mma.m16n8k32.s8 (ptxas: "requires .target sm_80 or
//     higher"), so the m16n8k32 logical tile is built from 4x m8n8k16, exactly
//     as llama.cpp's mmq.cuh does on this architecture.  For an expert with
//     count <= 8 only the first row half is issued, which is the common case:
//     m=16 tokens x nUsed=8 over 128 experts averages one token per expert.
//
// Activation layout: Q8_1 per 32 weights (one fp32 scale + 32 int8), produced
// once for all experts by q2QuantizeKernel, so the scale is folded into the
// epilogue as dA[i][k32] * dB[j][k32].
//
// The shared-memory int8 tile layout is the one Int8Mma.cuh's ldmatrix helpers
// expect (row-major, 32-bit elements, 16-byte-aligned row stride):
//   A: [16][20] ints per 64-k step  (16 ints of int8 + 2 scales + pad)
//   B: [128][20] ints per 64-k step (same layout, one row per output column)
// The 20-int stride makes an 8-row ldmatrix x2 bank-conflict free (the eight
// row starts land on disjoint 4-bank groups).

#include "Int8Mma.cuh"
#include "gpu/InklingKernels.hpp"

#ifdef USE_CUDA

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace tinycoder::gpu {
    namespace {

        constexpr int kMTile = 16;   // token rows in one tile
        constexpr int kNTile = 128;  // output columns in one tile
        constexpr int kWarps = 4;    // one warp per 32 output columns
        constexpr int kNPerWarp = 32;// 4 x n8 per warp
        constexpr int kKBlock = 64;  // one Q2_0 block
        constexpr int kAStride = 20; // ints per 64-k row
        constexpr int kBStride = 20;

        struct Q8Pack {
            int8_t qs[32];
            float d;
        };
        static_assert(sizeof(Q8Pack) == 36, "Q8_1 pack must be 36 B");

        // 2-bit codes in element order -> four int8 values (d*(code-1) is
        // applied as a scale in the epilogue, so the payload is code-1).
        // The selector nibble holds two 2-bit fields; a=b=LUT being 4-periodic
        // makes the low field win, so the even and odd codes need one perm each
        // and one interleave.  Same sequence as llama.cpp's q2_0 loader.
        __device__ __forceinline__ uint32_t q2Decode4(const uint32_t b) {
            constexpr uint32_t LUT = 0x020100FFu;// {-1, 0, 1, 2}
            const uint32_t ev = __byte_perm(LUT, LUT, b);
            const uint32_t od = __byte_perm(LUT, LUT, b >> 2);
            return __byte_perm(ev, od, 0x5140);
        }

        // fp32 -> Q8_1, one warp per 32-value block.  Writes the int8 payload
        // and the per-block scale to separate contiguous arrays so the GEMM's
        // A loader can copy a 64-k row with two vector loads.
        __global__ void __launch_bounds__(128) q2QuantizeKernel(
                const float *__restrict__ x, int8_t *__restrict__ aq,
                float *__restrict__ ad, uint32_t cols, uint32_t cap,
                uint32_t nExperts, const int32_t *__restrict__ counts,
                const int32_t *__restrict__ off) {
            const uint32_t e = blockIdx.x;
            if (e >= nExperts) return;
            if (counts != nullptr && counts[e] <= 0) return;
            const uint32_t rows = counts != nullptr
                                          ? static_cast<uint32_t>(counts[e])
                                          : cap;
            const uint32_t chunks = cols / 32u;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5u;
            const uint32_t total = rows * chunks;
            const uint32_t base =
                    off != nullptr ? static_cast<uint32_t>(off[e]) : e * cap;
            for (uint32_t c = warp; c < total; c += 4u) {
                const uint32_t r = c / chunks;
                const uint32_t kc = c - r * chunks;
                const float v =
                        x[(static_cast<size_t>(base + r)) * cols + kc * 32u + lane];
                float amax = fabsf(v);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
                const float d = amax > 0.0f ? amax * (1.0f / 127.0f) : 0.0f;
                const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                int q = __float2int_rn(v * inv);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                aq[(static_cast<size_t>(base + r)) * cols + kc * 32u + lane] =
                        static_cast<int8_t>(q);
                if (lane == 0)
                    ad[(static_cast<size_t>(base + r)) * chunks + kc] = d;
            }
        }

        // Two m8n8k16 instead of the four that make a logical m16n8k32: the
        // second row half is dead for an expert with <= 8 tokens.
        __device__ __forceinline__ void mmaFirstHalf(
                tc_mma::tile<16, 8, int> &D, const tc_mma::tile<16, 8, int> &A,
                const tc_mma::tile<8, 8, int> &B) {
#if defined(TURING_MMA_AVAILABLE)
            asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
                : "+r"(D.x[0]), "+r"(D.x[1])
                : "r"(A.x[0]), "r"(B.x[0]));
            asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
                : "+r"(D.x[0]), "+r"(D.x[1])
                : "r"(A.x[2]), "r"(B.x[1]));
#else
            tc_mma::mma(D, A, B);
#endif
        }

        __global__ void __launch_bounds__(kWarps * 32) q2_0MmaKernel(
                const char *__restrict__ w, const int8_t *__restrict__ aq,
                const float *__restrict__ ad, float *__restrict__ out,
                uint32_t cols, uint32_t rows, uint32_t cap, uint32_t nExperts,
                const int32_t *__restrict__ counts,
                const int32_t *__restrict__ off) {
            const uint32_t e = blockIdx.y;
            if (e >= nExperts) return;
            const int32_t cnt = counts != nullptr ? counts[e] : static_cast<int32_t>(cap);
            if (cnt <= 0) return;
            const uint32_t base =
                    off != nullptr ? static_cast<uint32_t>(off[e]) : e * cap;

            const uint32_t nt = blockIdx.x;
            const uint32_t n0 = nt * kNTile;
            const uint32_t kbCount = cols / kKBlock;
            const uint32_t lane = threadIdx.x;
            const uint32_t warp = threadIdx.y;
            const uint32_t tid = warp * 32u + lane;

            __shared__ int sA[kMTile * kAStride];
            __shared__ int sB[kNTile * kBStride];

            const char *wr = w + static_cast<size_t>(e) * rows * kbCount * 18u;
            const int8_t *aqr = aq + static_cast<size_t>(base) * cols;
            const float *adr = ad + static_cast<size_t>(base) * (cols / 32u);
            float *outr = out + static_cast<size_t>(base) * rows;

            // `cnt` can be up to cap (=128 in the layer-major graph); walk it in
            // 16-row MMA tiles.  The mean group is 8 tokens, so this is one
            // iteration and the expert's weights are decoded exactly once.
            for (uint32_t s0 = 0; s0 < static_cast<uint32_t>(cnt); s0 += kMTile) {
                const uint32_t rowsLive =
                        static_cast<uint32_t>(cnt) - s0 < kMTile
                                ? static_cast<uint32_t>(cnt) - s0
                                : kMTile;
                const bool twoHalves = rowsLive > 8;
                float acc[4][4] = {};

                for (uint32_t kb = 0; kb < kbCount; ++kb) {
                    __syncthreads();
                    if (tid < kMTile) {
                        const uint32_t r = tid;
                        const int *src = reinterpret_cast<const int *>(
                                aqr + static_cast<size_t>(s0 + r) * cols + kb * kKBlock);
                        int *dst = sA + r * kAStride;
#pragma unroll
                        for (int j = 0; j < 16; ++j) dst[j] = src[j];
                        dst[16] = __float_as_int(adr[static_cast<size_t>(s0 + r) * (cols / 32u) + 2u * kb]);
                        dst[17] = __float_as_int(adr[static_cast<size_t>(s0 + r) * (cols / 32u) + 2u * kb + 1u]);
                    }

                    {
                        const uint32_t n = tid;// 0..127
                        int *dst = sB + n * kBStride;
                        uint32_t v[16];
                        float d = 0.0f;
                        if (n0 + n < rows) {
                            const uint8_t *blk = reinterpret_cast<const uint8_t *>(
                                    wr + (static_cast<size_t>(n0 + n) * kbCount + kb) * 18u);
                            d = __half2float(*reinterpret_cast<const __half *>(blk));
                            const uint8_t *qs = blk + 2;
#pragma unroll
                            for (int j = 0; j < 16; ++j) v[j] = q2Decode4(qs[j]);
                        } else {
#pragma unroll
                            for (int j = 0; j < 16; ++j) v[j] = 0u;
                        }
#pragma unroll
                        for (int j = 0; j < 16; ++j) dst[j] = static_cast<int>(v[j]);
                        dst[16] = __float_as_int(d);
                        dst[17] = __float_as_int(d);
                    }
                    __syncthreads();

                    const uint32_t i0 = lane / 4u;       // 0..7
                    const uint32_t j0 = (lane % 4u) * 2u;// 0..7
                    for (int ks = 0; ks < 2; ++ks) {
                        tc_mma::tile<16, 8, int> A;
                        tc_mma::load_ldmatrix(A, sA + ks * 8, kAStride);
                        const float a0 = __int_as_float(sA[i0 * kAStride + 16 + ks]);
                        const float a1 = __int_as_float(sA[(i0 + 8u) * kAStride + 16 + ks]);
                        for (int t = 0; t < 4; ++t) {
                            const int bRow = static_cast<int>(warp) * kNPerWarp + t * 8;
                            tc_mma::tile<8, 8, int> B;
                            // load_generic, not load_ldmatrix: the mma B fragment is
                            // column-major (n,k) and the x2 ldmatrix layout does not
                            // produce it.  The generic int8 kernel makes the same
                            // choice on this path ("faster than load_ldmatrix").
                            tc_mma::load_generic(
                                    B, sB + bRow * kBStride + ks * 8, kBStride);
                            tc_mma::tile<16, 8, int> C;
                            if (twoHalves)
                                tc_mma::mma(C, A, B);
                            else
                                mmaFirstHalf(C, A, B);
                            const float b0 = __int_as_float(
                                    sB[(bRow + static_cast<int>(j0)) * kBStride + 16 + ks]);
                            const float b1 = __int_as_float(
                                    sB[(bRow + static_cast<int>(j0) + 1) * kBStride + 16 + ks]);
                            acc[t][0] += a0 * b0 * static_cast<float>(C.x[0]);
                            acc[t][1] += a0 * b1 * static_cast<float>(C.x[1]);
                            if (twoHalves) {
                                acc[t][2] += a1 * b0 * static_cast<float>(C.x[2]);
                                acc[t][3] += a1 * b1 * static_cast<float>(C.x[3]);
                            }
                        }
                    }
                }

                const uint32_t i0 = lane / 4u;
                const uint32_t j0 = (lane % 4u) * 2u;
                for (int t = 0; t < 4; ++t) {
                    const uint32_t col = n0 + warp * kNPerWarp + t * 8 + j0;
                    if (i0 < rowsLive && col < rows) {
                        outr[static_cast<size_t>(s0 + i0) * rows + col] =
                                acc[t][0];
                        if (col + 1 < rows)
                            outr[static_cast<size_t>(s0 + i0) * rows + col + 1] =
                                    acc[t][1];
                    }
                    if (twoHalves && i0 + 8u < rowsLive) {
                        if (col < rows) {
                            outr[static_cast<size_t>(s0 + i0 + 8u) * rows + col] =
                                    acc[t][2];
                            if (col + 1 < rows)
                                outr[static_cast<size_t>(s0 + i0 + 8u) * rows + col + 1] =
                                        acc[t][3];
                        }
                    }
                }
            }// s0 tile
        }

        // Coalesced variant: one block stages the whole [128][cols] weight slab
        // of its expert in shared memory with a contiguous 4-byte copy, then
        // decodes each 18-byte Q2_0 block from shared.  The direct loader above
        // reads 18 B per thread at a 324 B row stride, i.e. one 32 B sector per
        // row and ~55 GB/s measured; the slab is a single contiguous
        // 128*rowStride region, so the same bytes move at DRAM speed.  Shared
        // is ~53 KB for cols=1152, so this is one block per SM.
        __global__ void __launch_bounds__(kWarps * 32) q2_0MmaCoalesced(
                const char *__restrict__ w, const int8_t *__restrict__ aq,
                const float *__restrict__ ad, float *__restrict__ out,
                uint32_t cols, uint32_t rows, uint32_t cap, uint32_t nExperts,
                const int32_t *__restrict__ counts,
                const int32_t *__restrict__ off) {
            const uint32_t e = blockIdx.y;
            if (e >= nExperts) return;
            const int32_t cnt = counts != nullptr ? counts[e] : static_cast<int32_t>(cap);
            if (cnt <= 0) return;
            const uint32_t base =
                    off != nullptr ? static_cast<uint32_t>(off[e]) : e * cap;

            const uint32_t nt = blockIdx.x;
            const uint32_t n0 = nt * kNTile;
            const uint32_t kbCount = cols / kKBlock;
            const uint32_t rowStride = kbCount * 18u;
            const uint32_t lane = threadIdx.x;
            const uint32_t warp = threadIdx.y;
            const uint32_t tid = warp * 32u + lane;

            extern __shared__ __align__(16) char smem[];
            char *sW = smem;
            int *sA = reinterpret_cast<int *>(smem + kNTile * rowStride);
            int *sB = sA + kMTile * kAStride;

            const char *wr = w + static_cast<size_t>(e) * rows * rowStride;
            const char *wBase = wr + static_cast<size_t>(n0) * rowStride;
            const uint32_t nRows = n0 + kNTile <= rows ? kNTile : rows - n0;

            if ((rowStride & 3u) == 0u) {
                const size_t nInts =
                        static_cast<size_t>(nRows) * rowStride / 4u;
                const int *src = reinterpret_cast<const int *>(wBase);
                int *dst = reinterpret_cast<int *>(sW);
                for (size_t i = tid; i < nInts; i += 128u) dst[i] = src[i];
            } else {
                const size_t nBytes = static_cast<size_t>(nRows) * rowStride;
                for (size_t i = tid; i < nBytes; i += 128u) sW[i] = wBase[i];
            }
            __syncthreads();

            const int8_t *aqr = aq + static_cast<size_t>(base) * cols;
            const float *adr = ad + static_cast<size_t>(base) * (cols / 32u);
            float *outr = out + static_cast<size_t>(base) * rows;

            // The [128][cols] slab above is resident for every live-row tile, so
            // a group of up to 128 tokens still streams its weights once.
            for (uint32_t s0 = 0; s0 < static_cast<uint32_t>(cnt); s0 += kMTile) {
                const uint32_t rowsLive =
                        static_cast<uint32_t>(cnt) - s0 < kMTile
                                ? static_cast<uint32_t>(cnt) - s0
                                : kMTile;
                const bool twoHalves = rowsLive > 8;
                float acc[4][4] = {};

                for (uint32_t kb = 0; kb < kbCount; ++kb) {
                    __syncthreads();
                    if (tid < kMTile) {
                        const uint32_t r = tid;
                        const int *src = reinterpret_cast<const int *>(
                                aqr + static_cast<size_t>(s0 + r) * cols + kb * kKBlock);
                        int *dst = sA + r * kAStride;
#pragma unroll
                        for (int j = 0; j < 16; ++j) dst[j] = src[j];
                        dst[16] = __float_as_int(adr[static_cast<size_t>(s0 + r) * (cols / 32u) + 2u * kb]);
                        dst[17] = __float_as_int(adr[static_cast<size_t>(s0 + r) * (cols / 32u) + 2u * kb + 1u]);
                    }

                    {
                        const uint32_t n = tid;
                        int *dst = sB + n * kBStride;
                        uint32_t v[16];
                        float d = 0.0f;
                        if (n < nRows) {
                            const uint8_t *blk = reinterpret_cast<const uint8_t *>(sW) +
                                                 static_cast<size_t>(n) * rowStride +
                                                 static_cast<size_t>(kb) * 18u;
                            d = __half2float(*reinterpret_cast<const __half *>(blk));
                            const uint8_t *qs = blk + 2;
#pragma unroll
                            for (int j = 0; j < 16; ++j) v[j] = q2Decode4(qs[j]);
                        } else {
#pragma unroll
                            for (int j = 0; j < 16; ++j) v[j] = 0u;
                        }
#pragma unroll
                        for (int j = 0; j < 16; ++j) dst[j] = static_cast<int>(v[j]);
                        dst[16] = __float_as_int(d);
                        dst[17] = __float_as_int(d);
                    }
                    __syncthreads();

                    const uint32_t i0 = lane / 4u;
                    const uint32_t j0 = (lane % 4u) * 2u;
                    for (int ks = 0; ks < 2; ++ks) {
                        tc_mma::tile<16, 8, int> A;
                        tc_mma::load_ldmatrix(A, sA + ks * 8, kAStride);
                        const float a0 = __int_as_float(sA[i0 * kAStride + 16 + ks]);
                        const float a1 = __int_as_float(sA[(i0 + 8u) * kAStride + 16 + ks]);
                        for (int t = 0; t < 4; ++t) {
                            const int bRow = static_cast<int>(warp) * kNPerWarp + t * 8;
                            tc_mma::tile<8, 8, int> B;
                            tc_mma::load_generic(
                                    B, sB + bRow * kBStride + ks * 8, kBStride);
                            tc_mma::tile<16, 8, int> C;
                            if (twoHalves)
                                tc_mma::mma(C, A, B);
                            else
                                mmaFirstHalf(C, A, B);
                            const float b0 = __int_as_float(
                                    sB[(bRow + static_cast<int>(j0)) * kBStride + 16 + ks]);
                            const float b1 = __int_as_float(
                                    sB[(bRow + static_cast<int>(j0) + 1) * kBStride + 16 + ks]);
                            acc[t][0] += a0 * b0 * static_cast<float>(C.x[0]);
                            acc[t][1] += a0 * b1 * static_cast<float>(C.x[1]);
                            if (twoHalves) {
                                acc[t][2] += a1 * b0 * static_cast<float>(C.x[2]);
                                acc[t][3] += a1 * b1 * static_cast<float>(C.x[3]);
                            }
                        }
                    }
                }

                const uint32_t i0 = lane / 4u;
                const uint32_t j0 = (lane % 4u) * 2u;
                for (int t = 0; t < 4; ++t) {
                    const uint32_t col = n0 + warp * kNPerWarp + t * 8 + j0;
                    if (i0 < rowsLive && col < rows) {
                        outr[static_cast<size_t>(s0 + i0) * rows + col] = acc[t][0];
                        if (col + 1 < rows)
                            outr[static_cast<size_t>(s0 + i0) * rows + col + 1] = acc[t][1];
                    }
                    if (twoHalves && i0 + 8u < rowsLive) {
                        if (col < rows) {
                            outr[static_cast<size_t>(s0 + i0 + 8u) * rows + col] = acc[t][2];
                            if (col + 1 < rows)
                                outr[static_cast<size_t>(s0 + i0 + 8u) * rows + col + 1] = acc[t][3];
                        }
                    }
                }
            }// s0 tile
        }

        void *g_q20Scratch = nullptr;
        size_t g_q20Bytes = 0;

        bool ensureScratch(size_t bytes) {
            if (bytes <= g_q20Bytes) return true;
            if (g_q20Scratch != nullptr) {
                cudaFree(g_q20Scratch);
                g_q20Scratch = nullptr;
                g_q20Bytes = 0;
            }
            if (cudaMalloc(&g_q20Scratch, bytes) != cudaSuccess) {
                cudaGetLastError();
                g_q20Scratch = nullptr;
                return false;
            }
            g_q20Bytes = bytes;
            return true;
        }

    }// namespace

    bool int8GemmQ2_0GroupedOffsets(const void *w, const float *x, float *out,
                                    const int32_t *counts, const int32_t *off,
                                    uint32_t nSlots, uint32_t rows, uint32_t cols,
                                    uint32_t nExperts, void *stream,
                                    bool quantize) {
        if (w == nullptr || x == nullptr || out == nullptr) return false;
        // `counts[e]` live rows of expert e start at row off[e] of the gathered
        // activation/output arrays; the packed Q8_1 scratch is nSlots rows
        // (nExperts*capacity padded, m*nUsed compact).
        if (nSlots == 0 || rows == 0 || cols == 0 || nExperts == 0)
            return false;
        if (cols % kKBlock != 0) return false;

        const size_t aqBytes = static_cast<size_t>(nSlots) * cols;
        const size_t adBytes =
                static_cast<size_t>(nSlots) * (cols / 32u) * sizeof(float);
        if (!ensureScratch(aqBytes + adBytes)) return false;
        int8_t *aq = static_cast<int8_t *>(g_q20Scratch);
        float *ad = reinterpret_cast<float *>(aq + aqBytes);

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

        if (quantize) {
            q2QuantizeKernel<<<nExperts, 128, 0, s>>>(x, aq, ad, cols,
                                                      nSlots, nExperts, counts,
                                                      off);
            if (cudaGetLastError() != cudaSuccess) return false;
        }

        const dim3 grid((rows + kNTile - 1) / kNTile, nExperts);
        const size_t rowStride = static_cast<size_t>(cols / kKBlock) * 18u;
        const size_t smemBytes = static_cast<size_t>(kNTile) * rowStride +
                                 (kMTile * kAStride + kNTile * kBStride) * sizeof(int);
        // The coalesced slab needs the whole [128][cols] weight slice resident
        // in shared memory; fall back to the direct loader when it does not fit
        // (cols beyond ~1.3k on this 64 KB/SM part).
        static size_t g_smemOptin = 0;
        if (g_smemOptin == 0) {
            int optin = 0;
            int device = 0;
            if (cudaGetDevice(&device) == cudaSuccess &&
                cudaDeviceGetAttribute(
                        &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                        device) == cudaSuccess &&
                optin > 0)
                g_smemOptin = static_cast<size_t>(optin);
            else
                g_smemOptin = 1;// cannot query: never take the slab path
        }
        if (smemBytes <= g_smemOptin &&
            cudaFuncSetAttribute(q2_0MmaCoalesced,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smemBytes)) == cudaSuccess) {
            q2_0MmaCoalesced<<<grid, dim3(32, kWarps), smemBytes, s>>>(
                    static_cast<const char *>(w), aq, ad, out, cols, rows,
                    nSlots, nExperts, counts, off);
            if (cudaGetLastError() == cudaSuccess) return true;
        }
        cudaGetLastError();
        q2_0MmaKernel<<<grid, dim3(32, kWarps), 0, s>>>(
                static_cast<const char *>(w), aq, ad, out, cols, rows, nSlots,
                nExperts, counts, off);
        return cudaGetLastError() == cudaSuccess;
    }

    bool int8GemmQ2_0Grouped(const void *w, const float *x, float *out,
                             const int32_t *counts, uint32_t capacity,
                             uint32_t rows, uint32_t cols, uint32_t nExperts,
                             void *stream, bool quantize) {
        if (capacity == 0 || capacity > 1024u || nExperts == 0) return false;
        return int8GemmQ2_0GroupedOffsets(
                w, x, out, counts, nullptr, nExperts * capacity, rows, cols,
                nExperts, stream, quantize);
    }

    bool int8GemmQ2_0(const void *w, const float *x, float *out,
                      uint32_t seqLen, uint32_t rows, uint32_t cols,
                      void *stream, bool quantize) {
        return int8GemmQ2_0Grouped(w, x, out, nullptr, seqLen, rows, cols, 1,
                                   stream, quantize);
    }

}// namespace tinycoder::gpu

#endif// USE_CUDA
