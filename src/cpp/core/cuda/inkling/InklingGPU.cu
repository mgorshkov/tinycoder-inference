#include "InklingGPUImpl.hpp"

#include "gpu/InklingGPU.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>

namespace tinycoder::gpu {

    void InklingGPU::Impl::release() {
        for (auto &kv: mats) cudaFree(kv.second.p);
        mats.clear();
        cudaFree(dx);
        cudaFree(dy);
        dx = dy = nullptr;
        cudaFree(kCache);
        cudaFree(vCache);
        cudaFree(convState);
        cudaFree(arena);
        kCache = vCache = convState = arena = nullptr;
        if (batch.arena != nullptr) {
            cudaFree(batch.arena);
        }
        if (batch.embPinned != nullptr) cudaFreeHost(batch.embPinned);
        if (batch.countsPinned != nullptr) cudaFreeHost(batch.countsPinned);
        batch = BatchDev{};
        if (graphExec != nullptr) {
            cudaGraphExecDestroy(graphExec);
            graphExec = nullptr;
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
        graphReady = false;
        for (float *p: pinnedEmb) cudaFreeHost(p);
        pinnedEmb.clear();
        if (captureStream != nullptr) {
            cudaStreamDestroy(captureStream);
            captureStream = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        ready = false;
        uploadedBytes = 0;
    }

    void *InklingGPU::Impl::getRaw(const void *host, uint64_t bytes,
                                   std::string &err, const char *what) {
        if (host == nullptr || bytes == 0) {
            err = std::string("missing host data for ") + what;
            return nullptr;
        }
        auto it = mats.find(host);
        if (it != mats.end() && it->second.bytes >= bytes) return it->second.p;
        void *p = nullptr;
        if (it != mats.end()) {// grow an undersized entry
            cudaFree(it->second.p);
            mats.erase(it);
        }
        const cudaError_t a = cudaMalloc(&p, bytes);
        if (a != cudaSuccess) {
            cudaGetLastError();
            err = std::string("cudaMalloc(") + what + ", " +
                  std::to_string(bytes) + " bytes) failed: " +
                  cudaGetErrorString(a);
            return nullptr;
        }
        // Pageable H2D: CUDA stages the source before returning, so the copy is
        // valid and ordered on this stream without an explicit host sync.
        const cudaError_t c = cudaMemcpyAsync(p, host, bytes,
                                              cudaMemcpyHostToDevice, stream);
        if (c != cudaSuccess) {
            cudaFree(p);
            cudaGetLastError();
            err = std::string("cudaMemcpyAsync(") + what + ", " +
                  std::to_string(bytes) + " bytes) failed: " +
                  cudaGetErrorString(c);
            return nullptr;
        }
        DevMat dm;
        dm.p = p;
        dm.bytes = bytes;
        mats.emplace(host, dm);
        uploadedBytes += bytes;
        return p;
    }

    void *InklingGPU::Impl::getMat(const void *host, uint32_t type,
                                   uint32_t rows, uint32_t cols,
                                   std::string &err, const char *what) {
        const uint32_t bb = inklingTypeBlockBytes(type);
        const uint32_t qk = inklingTypeQk(type);
        if (bb == 0 || qk == 0) {
            err = std::string("unsupported quant type ") + std::to_string(type) +
                  " for " + what;
            return nullptr;
        }
        const uint64_t blocksPerRow = (static_cast<uint64_t>(cols) + qk - 1) / qk;
        const uint64_t bytes = static_cast<uint64_t>(rows) * blocksPerRow * bb;
        return getRaw(host, bytes, err, what);
    }

    InklingGPU::InklingGPU() : impl_(new Impl) {
        int dev = 0;
        available_ = cudaGetDevice(&dev) == cudaSuccess && dev >= 0;
    }

    InklingGPU::~InklingGPU() {
        if (!impl_) return;
        impl_->release();
        delete impl_;
    }

    uint64_t InklingGPU::uploadedBytes() const {
        return impl_ ? impl_->uploadedBytes : 0;
    }

    bool InklingGPU::gemvRows(const void *hostW, uint32_t type, uint32_t rowsTotal,
                              uint32_t cols, uint32_t rowStart, uint32_t numRows,
                              const float *x, float *out) {
        if (!available_ || !impl_ || hostW == nullptr || rowStart + numRows > rowsTotal)
            return false;
        const uint32_t bb = inklingTypeBlockBytes(type);
        const uint32_t qk = inklingTypeQk(type);
        if (bb == 0) return false;
        std::string err;
        void *dw = impl_->getMat(hostW, type, rowsTotal, cols, err, "hybrid weight");
        if (dw == nullptr) {
            std::fprintf(stderr, "[inkling] %s\n", err.c_str());
            available_ = false;
            return false;
        }
        if (cols > impl_->dxBytes) {
            cudaFree(impl_->dx);
            if (cudaMalloc(&impl_->dx, cols * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                available_ = false;
                return false;
            }
            impl_->dxBytes = cols;
        }
        if (numRows > impl_->dyBytes) {
            cudaFree(impl_->dy);
            if (cudaMalloc(&impl_->dy, numRows * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                available_ = false;
                return false;
            }
            impl_->dyBytes = numRows;
        }
        if (cudaMemcpy(impl_->dx, x, cols * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        const uint64_t blocksPerRow = (static_cast<uint64_t>(cols) + qk - 1) / qk;
        const uint64_t rowBytes = blocksPerRow * bb;
        const uint8_t *w = static_cast<const uint8_t *>(dw) +
                           static_cast<uint64_t>(rowStart) * rowBytes;
        if (!inklingGemv(type, w, static_cast<float *>(impl_->dx),
                         static_cast<float *>(impl_->dy), numRows, cols, nullptr))
            return false;
        if (cudaMemcpy(out, impl_->dy, numRows * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        return true;
    }

    bool InklingGPU::gemv(const void *hostW, uint32_t type, uint32_t rows,
                          uint32_t cols, const float *x, float *out) {
        return gemvRows(hostW, type, rows, cols, 0, rows, x, out);
    }

    void InklingGPU::clearKVCache() {
        if (!impl_) return;
        impl_->kvPos = 0;
        if (impl_->stream == nullptr) return;
        if (impl_->dPos != nullptr) {
            cudaMemsetAsync(impl_->dPos, 0, sizeof(int32_t), impl_->stream);
            impl_->dPosValid = true;
        }
        const uint64_t kvFloats = static_cast<uint64_t>(impl_->slots) *
                                  impl_->maxSeq * impl_->kvDim;
        const uint64_t convFloats = static_cast<uint64_t>(impl_->slots) *
                                    impl_->histLen * impl_->sconvChannels;
        if (impl_->kCache)
            cudaMemsetAsync(impl_->kCache, 0, kvFloats * sizeof(float),
                            impl_->stream);
        if (impl_->vCache)
            cudaMemsetAsync(impl_->vCache, 0, kvFloats * sizeof(float),
                            impl_->stream);
        if (impl_->convState)
            cudaMemsetAsync(impl_->convState, 0, convFloats * sizeof(float),
                            impl_->stream);
    }

}// namespace tinycoder::gpu
