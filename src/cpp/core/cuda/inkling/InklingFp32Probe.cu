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

// Prototype: dequantize an Inkling expert weight block to FP32 on the device
// (IQ1_S and Q2_0) and run the resulting dense matrix through cuBLAS SGEMM.
//
// This is measurement plumbing for tools/inkling_fp32_gemm_probe, not part of
// the shipped graph.  The dequant kernels repeat
// GGMLDequantize::dequantizeIQ1_SBlock / dequantizeQ2_0Block operation for
// operation, so their output is bit-identical to the host dequantizer; the
// probe checks that (max|d| == 0) before timing anything.
//
// The kernels are store-bound by construction: one byte of IQ1_S decodes to
// 20 bytes of FP32 (50 B / 256 weights), one Q2_0 byte pack decodes to 64*4 B.
// Each warp owns one quant block and writes it as contiguous float4s.

#include "gpu/InklingFp32Probe.hpp"

#include "GGMLDequantize.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <mutex>

namespace tinycoder::gpu {
    namespace {

        constexpr uint32_t kQkIq1 = 256;
        constexpr uint32_t kIq1Bytes = 50;
        constexpr uint32_t kQkQ20 = 64;
        constexpr uint32_t kQ20Bytes = 18;

        /// One warp per IQ1_S quant block: lane l writes weights [8l, 8l+8),
        /// i.e. group ib = l/4, 8-weight sub-index l%4.  The 16 KB grid table
        /// is passed as a device pointer (it stays resident in L2).
        __global__ void kDequantIq1s(const uint8_t *__restrict__ w,
                                     float *__restrict__ out, uint32_t nBlocks,
                                     const int8_t *__restrict__ grid) {
            const uint32_t warps = blockDim.x >> 5;
            const uint32_t warp = blockIdx.x * warps + (threadIdx.x >> 5);
            if (warp >= nBlocks) return;
            const uint32_t lane = threadIdx.x & 31u;
            const uint8_t *b = w + static_cast<uint64_t>(warp) * kIq1Bytes;
            const float d = __half2float(*reinterpret_cast<const __half *>(b));
            const uint8_t *qs = b + 2;
            const uint8_t *qh = b + 34;
            const uint32_t ib = lane >> 2;
            const uint32_t sub = lane & 3u;
            const uint16_t qhb = static_cast<uint16_t>(
                    static_cast<uint16_t>(qh[2 * ib]) |
                    (static_cast<uint16_t>(qh[2 * ib + 1]) << 8));
            const float dl =
                    d * static_cast<float>(2 * ((qhb >> 12) & 7) + 1);
            const float delta = (qhb & 0x8000) ? -0.125f : 0.125f;
            const uint32_t idx = static_cast<uint32_t>(qs[4 * ib + sub]) |
                                 (static_cast<uint32_t>((qhb >> (3 * sub)) & 7)
                                  << 8);
            const int8_t *g = grid + static_cast<uint64_t>(idx) * 8;
            float *y = out + static_cast<uint64_t>(warp) * kQkIq1 +
                       static_cast<uint64_t>(lane) * 8;
            // Two float4 stores per lane: 32 contiguous bytes, one 1 KB block
            // per warp.
#pragma unroll
            for (uint32_t j = 0; j < 8; j += 4) {
                float4 v;
                v.x = dl * (static_cast<float>(g[j + 0]) + delta);
                v.y = dl * (static_cast<float>(g[j + 1]) + delta);
                v.z = dl * (static_cast<float>(g[j + 2]) + delta);
                v.w = dl * (static_cast<float>(g[j + 3]) + delta);
                *reinterpret_cast<float4 *>(y + j) = v;
            }
        }

        /// One warp per Q2_0 quant block; lane l writes weights [2l, 2l+2) as
        /// one float2.
        __global__ void kDequantQ20(const uint8_t *__restrict__ w,
                                    float *__restrict__ out, uint32_t nBlocks) {
            const uint32_t warps = blockDim.x >> 5;
            const uint32_t warp = blockIdx.x * warps + (threadIdx.x >> 5);
            if (warp >= nBlocks) return;
            const uint32_t lane = threadIdx.x & 31u;
            const uint8_t *b = w + static_cast<uint64_t>(warp) * kQ20Bytes;
            const float d = __half2float(*reinterpret_cast<const __half *>(b));
            const uint8_t *qs = b + 2;
            const uint32_t j = lane * 2u;
            const uint8_t byte = qs[j >> 2];
            const int q0 = static_cast<int>((byte >> ((j & 3u) * 2u)) & 3u) - 1;
            const int q1 = static_cast<int>((byte >> (((j + 1u) & 3u) * 2u)) & 3u) - 1;
            float2 v;
            v.x = static_cast<float>(q0) * d;
            v.y = static_cast<float>(q1) * d;
            *reinterpret_cast<float2 *>(out + static_cast<uint64_t>(warp) * kQkQ20 +
                                        j) = v;
        }

        /// The 2048 x 8 int8 IQ1_S grid, uploaded once from the host table.
        const int8_t *iq1GridDevice() {
            static std::mutex mu;
            static int8_t *dev = nullptr;
            std::lock_guard<std::mutex> lock(mu);
            if (dev == nullptr) {
                if (cudaMalloc(reinterpret_cast<void **>(&dev),
                               2048u * 8u * sizeof(int8_t)) != cudaSuccess) {
                    cudaGetLastError();
                    return nullptr;
                }
                if (cudaMemcpy(dev, &GGMLDequantize::iq1s_grid[0],
                               2048u * 8u * sizeof(int8_t),
                               cudaMemcpyHostToDevice) != cudaSuccess) {
                    cudaGetLastError();
                    cudaFree(dev);
                    dev = nullptr;
                    return nullptr;
                }
            }
            return dev;
        }

        /// cuBLAS handle for the prototype; the probe is single-threaded and
        /// single-stream.
        cublasHandle_t sgemmHandle() {
            static std::mutex mu;
            static cublasHandle_t handle = nullptr;
            std::lock_guard<std::mutex> lock(mu);
            if (handle == nullptr) {
                if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
                    handle = nullptr;
                }
            }
            return handle;
        }

    }// namespace

    bool inklingFp32DequantSupported(uint32_t type) {
        return type == 19u || type == 42u;
    }

    bool inklingFp32Dequant(uint32_t type, const void *w, float *out,
                            uint64_t numElements, void *stream) {
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        if (type == 19u) {
            if (numElements % kQkIq1 != 0u) return false;
            const int8_t *grid = iq1GridDevice();
            if (grid == nullptr) return false;
            const uint32_t nBlocks = static_cast<uint32_t>(numElements / kQkIq1);
            const uint32_t threads = 256;
            const uint32_t blocks = (nBlocks + threads / 32u - 1u) / (threads / 32u);
            kDequantIq1s<<<blocks, threads, 0, s>>>(
                    static_cast<const uint8_t *>(w), out, nBlocks, grid);
            return cudaGetLastError() == cudaSuccess;
        }
        if (type == 42u) {
            if (numElements % kQkQ20 != 0u) return false;
            const uint32_t nBlocks = static_cast<uint32_t>(numElements / kQkQ20);
            const uint32_t threads = 256;
            const uint32_t blocks = (nBlocks + threads / 32u - 1u) / (threads / 32u);
            kDequantQ20<<<blocks, threads, 0, s>>>(
                    static_cast<const uint8_t *>(w), out, nBlocks);
            return cudaGetLastError() == cudaSuccess;
        }
        return false;
    }

    bool inklingFp32Gemm(const float *w, const float *x, float *out,
                         uint32_t rows, uint32_t cols, uint32_t m,
                         void *stream) {
        if (rows == 0u || cols == 0u || m == 0u) return false;
        cublasHandle_t handle = sgemmHandle();
        if (handle == nullptr) return false;
        if (cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream)) !=
            CUBLAS_STATUS_SUCCESS)
            return false;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // C(col-major n x m) = op(A) * op(B): out row-major [m][rows] is the
        // same memory as out^T column-major [rows][m], W row-major [rows][cols]
        // is W^T column-major with lda = cols, and x row-major [m][cols] is
        // x^T column-major with ldb = cols.
        const cublasStatus_t st = cublasSgemm(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows),
                static_cast<int>(m), static_cast<int>(cols), &alpha, w,
                static_cast<int>(cols), x, static_cast<int>(cols), &beta, out,
                static_cast<int>(rows));
        return st == CUBLAS_STATUS_SUCCESS;
    }

}// namespace tinycoder::gpu
