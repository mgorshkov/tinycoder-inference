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

// Grouped-expert GPU MMQ for prefill: one expert at a time is streamed to the
// device, its token group is batched through int8Gemm, and the weighted down
// rows are accumulated on the host.  This keeps the full expert set in host
// memory while still using tensor cores for the expert projections.

#include "GPUComputeInternal.hpp"
#include "Int8Gemm.hpp"

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <vector>

namespace tinycoder::gpu {

    namespace {
        __global__ void kMoeBatchSwiGLU(float *g, const float *u, uint32_t n,
                                        uint32_t ff, uint32_t stride) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n * ff) return;
            const uint32_t t = i / ff;
            const uint32_t j = i % ff;
            const float gv = g[t * stride + j];
            const float uv = u ? u[t * ff + j] : g[t * stride + ff + j];
            g[t * ff + j] = (gv / (1.0f + expf(-gv))) * uv;
        }
    }// namespace

    bool GPUModel::moeBatchExperts(
            const uint8_t *gate, uint32_t gateType, uint32_t gateRowsPerExpert,
            const uint8_t *up, uint32_t upType, uint32_t upRowsPerExpert,
            const uint8_t *down, uint32_t downType, uint32_t downRowsPerExpert,
            const float *normHost, float *outHost, uint32_t seqLen,
            uint32_t hidden, uint32_t expertFF, const int32_t *expertIdx,
            const float *expertWgt, uint32_t expertsUsed, uint32_t expertCount,
            std::string &errMsg) {
        if (seqLen < 2 || gate == nullptr || down == nullptr ||
            expertIdx == nullptr || expertWgt == nullptr ||
            expertCount == 0 || expertsUsed == 0) {
            return false;
        }
        const bool fused = up == nullptr;
        const uint32_t gateCols = hidden;
        const uint32_t downCols = expertFF;
        auto rowBytes = [](uint32_t type, uint32_t cols) -> uint64_t {
            const uint32_t bs = ggmlBlockSize(type);
            const uint32_t ts = ggmlTypeSize(type);
            return ((static_cast<uint64_t>(cols) + bs - 1) / bs) * ts;
        };
        const uint64_t gRB = rowBytes(gateType, gateCols);
        const uint64_t uRB = fused ? 0 : rowBytes(upType, gateCols);
        const uint64_t dRB = rowBytes(downType, downCols);
        const uint64_t gBytes = static_cast<uint64_t>(gateRowsPerExpert) * gRB;
        const uint64_t uBytes = static_cast<uint64_t>(upRowsPerExpert) * uRB;
        const uint64_t dBytes = static_cast<uint64_t>(downRowsPerExpert) * dRB;
        const uint32_t maxN = seqLen * expertsUsed;
        void *dG = nullptr, *dU = nullptr, *dD = nullptr, *dX = nullptr;
        void *dGO = nullptr, *dUO = nullptr, *dDO = nullptr;
        auto alloc = [&](void **p, uint64_t n) -> bool {
            if (n == 0) { *p = nullptr; return true; }
            return cudaMalloc(p, n) == cudaSuccess;
        };
        const uint64_t allG = gBytes * expertCount;
        const uint64_t allU = uBytes * expertCount;
        const uint64_t allD = dBytes * expertCount;
        bool ok = alloc(&dG, allG) && alloc(&dU, allU) && alloc(&dD, allD) &&
                  alloc(&dX, static_cast<uint64_t>(maxN) * hidden * sizeof(float)) &&
                  alloc(&dGO, static_cast<uint64_t>(maxN) * gateRowsPerExpert * sizeof(float)) &&
                  alloc(&dUO, fused ? 0 : static_cast<uint64_t>(maxN) * upRowsPerExpert * sizeof(float)) &&
                  alloc(&dDO, static_cast<uint64_t>(maxN) * downRowsPerExpert * sizeof(float));
        if (!ok) {
            errMsg = "moeBatchExperts: cudaMalloc failed";
            for (void *p : {dG, dU, dD, dX, dGO, dUO, dDO}) if (p) cudaFree(p);
            return false;
        }
        // Upload all experts of this layer once; the per-expert loop then only
        // streams the gathered activations and the down result.
        cudaMemcpyAsync(dG, gate, allG, cudaMemcpyHostToDevice, g_stream);
        if (!fused) cudaMemcpyAsync(dU, up, allU, cudaMemcpyHostToDevice, g_stream);
        cudaMemcpyAsync(dD, down, allD, cudaMemcpyHostToDevice, g_stream);
        std::vector<uint32_t> sList, rList;
        std::vector<float> xHost, yHost;
        for (uint32_t e = 0; e < expertCount; ++e) {
            sList.clear(); rList.clear();
            for (uint32_t s = 0; s < seqLen; ++s) {
                for (uint32_t r = 0; r < expertsUsed; ++r) {
                    if (expertIdx[static_cast<size_t>(s) * expertsUsed + r] ==
                        static_cast<int32_t>(e)) {
                        sList.push_back(s);
                        rList.push_back(r);
                    }
                }
            }
            const uint32_t n = static_cast<uint32_t>(sList.size());
            if (n == 0) continue;
            const uint32_t n8 = n < 8 ? 8 : n;
            xHost.resize(static_cast<size_t>(n8) * hidden);
            for (uint32_t t = 0; t < n8; ++t) {
                const uint32_t s = sList[t < n ? t : n - 1];
                std::memcpy(xHost.data() + static_cast<size_t>(t) * hidden,
                            normHost + static_cast<size_t>(s) * hidden,
                            hidden * sizeof(float));
            }
            const uint64_t gOff = static_cast<uint64_t>(e) * gBytes;
            const uint64_t uOff = static_cast<uint64_t>(e) * uBytes;
            const uint64_t dOff = static_cast<uint64_t>(e) * dBytes;
            cudaMemcpyAsync(dX, xHost.data(), static_cast<size_t>(n8) * hidden * sizeof(float),
                            cudaMemcpyHostToDevice, g_stream);
            const uint32_t gBpr = (gateCols + ggmlBlockSize(gateType) - 1) /
                                  ggmlBlockSize(gateType);
            const uint32_t uBpr = fused ? 0 : (gateCols + ggmlBlockSize(upType) - 1) /
                                                  ggmlBlockSize(upType);
            const uint32_t dBpr = (downCols + ggmlBlockSize(downType) - 1) /
                                  ggmlBlockSize(downType);
            bool good = int8Gemm(gateType, static_cast<uint8_t *>(dG) + gOff,
                                 reinterpret_cast<const float *>(dX),
                                 reinterpret_cast<float *>(dGO), n8,
                                 gateRowsPerExpert, gateCols, gBpr,
                                 reinterpret_cast<void *>(g_stream));
            if (good && !fused) {
                good = int8Gemm(upType, static_cast<uint8_t *>(dU) + uOff,
                                reinterpret_cast<const float *>(dX),
                                reinterpret_cast<float *>(dUO), n8,
                                upRowsPerExpert, gateCols, uBpr,
                                reinterpret_cast<void *>(g_stream));
            }
            if (!good) {
                errMsg = "moeBatchExperts: gate/up int8Gemm unsupported";
                break;
            }
            const uint32_t stride = fused ? gateRowsPerExpert : gateRowsPerExpert;
            const uint32_t ff = fused ? gateRowsPerExpert / 2 : gateRowsPerExpert;
            kMoeBatchSwiGLU<<<(n8 * ff + 255) / 256, 256, 0, g_stream>>>(
                    reinterpret_cast<float *>(dGO),
                    fused ? nullptr : reinterpret_cast<const float *>(dUO), n8, ff,
                    stride);
            good = int8Gemm(downType, static_cast<uint8_t *>(dD) + dOff,
                            reinterpret_cast<const float *>(dGO),
                            reinterpret_cast<float *>(dDO), n8,
                            downRowsPerExpert, downCols, dBpr,
                            reinterpret_cast<void *>(g_stream));
            if (!good) {
                errMsg = "moeBatchExperts: down int8Gemm unsupported";
                break;
            }
            yHost.resize(static_cast<size_t>(n8) * downRowsPerExpert);
            cudaMemcpyAsync(yHost.data(), dDO,
                            static_cast<size_t>(n8) * downRowsPerExpert * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
            if (cudaStreamSynchronize(g_stream) != cudaSuccess) {
                errMsg = "moeBatchExperts: stream sync failed";
                break;
            }
            for (uint32_t t = 0; t < n; ++t) {
                const uint32_t s = sList[t];
                const float wg = expertWgt[static_cast<size_t>(s) * expertsUsed + rList[t]];
                for (uint32_t i = 0; i < downRowsPerExpert; ++i) {
                    outHost[static_cast<size_t>(s) * downRowsPerExpert + i] +=
                            wg * yHost[static_cast<size_t>(t) * downRowsPerExpert + i];
                }
            }
        }
        for (void *p : {dG, dU, dD, dX, dGO, dUO, dDO}) if (p) cudaFree(p);
        return errMsg.empty();
    }

}// namespace tinycoder::gpu
#endif// USE_CUDA
