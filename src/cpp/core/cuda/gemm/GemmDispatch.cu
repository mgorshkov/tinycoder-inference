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
// GemmDispatch — host side of the int8 tensor-core prefill/decode GEMMs.
//
// The kernels in src/cpp/core/cuda/ implement
// int8 tensor-core `int8Matmul` implemented in this repository.
// On sm_75 the engine runs in **stream-k** mode: the
// grid is `nSM` blocks, each walks a slice of the (weight-row tile, token-tile,
// k-chunk) work, and the partial sums of tiles split across blocks are folded
// by `int8GemmFixup*`.  Stream-k is what the Turing numbers
// are measured with.
//
// This translation unit owns the device query, the Q8_1 activation scratch
// (144 B per 128 values, INT8 layout), the stream-k fixup buffer, and tile
// selection; the engine calls it through include/Int8Gemm.hpp only.
// -----------------------------------------------------------------------------
#ifdef USE_CUDA

#include "Int8Gemm.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#define DECL_INT8_SK(N)                                                         \
    __global__ void int8GemmStreamKQ4KTile##N(                                   \
            const char *x, const int *y, float *dst, float *tmp_fixup,         \
            const int ncols_x, const int nrows_x, const int ncols_dst,         \
            const int stride_row_x, const int ncols_y, const int stride_col_dst, \
            const int ncols_max);                                              \
    __global__ void int8GemmStreamKQ5KTile##N(                                   \
            const char *x, const int *y, float *dst, float *tmp_fixup,         \
            const int ncols_x, const int nrows_x, const int ncols_dst,         \
            const int stride_row_x, const int ncols_y, const int stride_col_dst, \
            const int ncols_max);                                              \
    __global__ void int8GemmStreamKQ6KTile##N(                                   \
            const char *x, const int *y, float *dst, float *tmp_fixup,         \
            const int ncols_x, const int nrows_x, const int ncols_dst,         \
            const int stride_row_x, const int ncols_y, const int stride_col_dst, \
            const int ncols_max);                                              \
    __global__ void int8GemmStreamKQ80Tile##N(                                   \
            const char *x, const int *y, float *dst, float *tmp_fixup,         \
            const int ncols_x, const int nrows_x, const int ncols_dst,         \
            const int stride_row_x, const int ncols_y, const int stride_col_dst, \
            const int ncols_max);                                              \
    __global__ void int8GemmFixupQ4KTile##N(                                \
            float *dst, const float *tmp_fixup, const int ncols_x,             \
            const int nrows_x, const int ncols_dst, const int stride_col_dst,  \
            const int ncols_max);                                              \
    __global__ void int8GemmFixupQ5KTile##N(                                \
            float *dst, const float *tmp_fixup, const int ncols_x,             \
            const int nrows_x, const int ncols_dst, const int stride_col_dst,  \
            const int ncols_max);                                              \
    __global__ void int8GemmFixupQ6KTile##N(                                \
            float *dst, const float *tmp_fixup, const int ncols_x,             \
            const int nrows_x, const int ncols_dst, const int stride_col_dst,  \
            const int ncols_max);                                              \
    __global__ void int8GemmFixupQ80Tile##N(                                \
            float *dst, const float *tmp_fixup, const int ncols_x,             \
            const int nrows_x, const int ncols_dst, const int stride_col_dst,  \
            const int ncols_max);

DECL_INT8_SK(32)
DECL_INT8_SK(64)
DECL_INT8_SK(128)

    __global__ void int8GemmStreamKIQ2STile32(const char *, const int *, float *, float *, int, int, int, int, int, int, int);
    __global__ void int8GemmStreamKIQ2STile64(const char *, const int *, float *, float *, int, int, int, int, int, int, int);
    __global__ void int8GemmStreamKIQ2STile128(const char *, const int *, float *, float *, int, int, int, int, int, int, int);
    __global__ void int8GemmStreamKIQ3XXSTile32(const char *, const int *, float *, float *, int, int, int, int, int, int, int);
    __global__ void int8GemmStreamKIQ3XXSTile64(const char *, const int *, float *, float *, int, int, int, int, int, int, int);
    __global__ void int8GemmStreamKIQ3XXSTile128(const char *, const int *, float *, float *, int, int, int, int, int, int, int);
    __global__ void int8GemmFixupIQ2STile32(float *, const float *, int, int, int, int, int);
    __global__ void int8GemmFixupIQ2STile64(float *, const float *, int, int, int, int, int);
    __global__ void int8GemmFixupIQ2STile128(float *, const float *, int, int, int, int, int);
    __global__ void int8GemmFixupIQ3XXSTile32(float *, const float *, int, int, int, int, int);
    __global__ void int8GemmFixupIQ3XXSTile64(float *, const float *, int, int, int, int, int);
    __global__ void int8GemmFixupIQ3XXSTile128(float *, const float *, int, int, int, int, int);

void int8QuantizeQ8Ds4(const void *x_f32, const int32_t *ids,
                                  void *vy, int type_x, int64_t ne00,
                                  int64_t s01, int64_t s02, int64_t s03,
                                  int64_t ne0, int64_t ne1, int64_t ne2,
                                  int64_t ne3, void *stream);
void int8QuantizeQ8D4(const void *x_f32, const int32_t *ids,
                                 void *vy, int type_x, int64_t ne00,
                                 int64_t s01, int64_t s02, int64_t s03,
                                 int64_t ne0, int64_t ne1, int64_t ne2,
                                 int64_t ne3, void *stream);    // GEMV (decode/small-batch) launchers, Q8_1 plain layout.
    void gemvLaunchQ2KF32Plain(const void *vx, const void *vy,
                                         void *dst, int ncols_x, int nrows_x,
                                         int stride_col_y, int stride_col_dst,
                                         int b_size, void *stream);
    void gemvLaunchQ3KF32Plain(const void *vx, const void *vy,
                                         void *dst, int ncols_x, int nrows_x,
                                         int stride_col_y, int stride_col_dst,
                                         int b_size, void *stream);
    void gemvLaunchQ4KF32Plain(const void *vx, const void *vy,
                                         void *dst, int ncols_x, int nrows_x,
                                         int stride_col_y, int stride_col_dst,
                                         int b_size, void *stream);
    void gemvLaunchQ5KF32Plain(const void *vx, const void *vy,
                                         void *dst, int ncols_x, int nrows_x,
                                         int stride_col_y, int stride_col_dst,
                                         int b_size, void *stream);
    void gemvLaunchIQ2XSF32Plain(const void *vx, const void *vy,
                                 void *dst, int cols, int rows,
                                 int strideY, int strideDst, int b,
                                 void *stream);
    void gemvLaunchIQ3SF32Plain(const void *vx, const void *vy,
                                void *dst, int cols, int rows,
                                int strideY, int strideDst, int b,
                                void *stream);
    void gemvLaunchIQ2SF32Plain(const void *vx, const void *vy,
                                void *dst, int cols, int rows,
                                int strideY, int strideDst, int b,
                                void *stream);
    void gemvLaunchIQ3XXSF32Plain(const void *vx, const void *vy,
                                  void *dst, int cols, int rows,
                                  int strideY, int strideDst, int b,
                                  void *stream);
    void gemvLaunchQ6KF32Plain(const void *vx, const void *vy,
                                         void *dst, int ncols_x, int nrows_x,
                                         int stride_col_y, int stride_col_dst,
                                         int b_size, void *stream);
    void gemvLaunchQ80F32Plain(const void *vx, const void *vy,
                                         void *dst, int ncols_x, int nrows_x,
                                         int stride_col_y, int stride_col_dst,
                                         int b_size, void *stream);
    void gemvLaunchQuantizeQ8F32(const void *x, void *vy, int kx,
                                            int kx_padded, int num_rows,
                                            void *stream);
}

namespace tinycoder::gpu {
    namespace {
        constexpr int kTypeQ4K_ = 12;
        constexpr int kTypeQ5K_ = 13;
        constexpr int kTypeQ6K_ = 14;
        constexpr int kTypeQ8_0_ = 8;
        constexpr int kTypeQ2K_ = 10;
        constexpr int kTypeQ3K_ = 11;
        constexpr int kTypeIQ2S_ = 22;
        constexpr int kTypeIQ2XS_ = 17;
        constexpr int kTypeIQ3S_ = 21;
        constexpr int kTypeIQ3XXS_ = 18;
        constexpr int kInt8Y = 128;// weight rows per tile (Turing+)

        struct Int8Ctx {
            bool ok = false;
            int cc = 0;
            int nsm = 0;
            int smpbo = 48 * 1024;
        };

        Int8Ctx &ctx() {
            static Int8Ctx c = [] {
                Int8Ctx r;
                int dev = 0;
                if (cudaGetDevice(&dev) != cudaSuccess) return r;
                int major = 0, minor = 0, nsm = 0, smem = 0;
                if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                           dev) != cudaSuccess)
                    return r;
                if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                           dev) != cudaSuccess)
                    return r;
                if (cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount,
                                           dev) != cudaSuccess)
                    return r;
                if (cudaDeviceGetAttribute(
                            &smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                            dev) != cudaSuccess)
                    smem = 48 * 1024;
                r.cc = major * 100 + minor;
                r.nsm = nsm;
                r.smpbo = smem;
                r.ok = (major > 7) || (major == 7 && minor >= 5);
                return r;
            }();
            return c;
        }

        void *g_yq = nullptr;
        size_t g_yqBytes = 0;
        void *g_fixup = nullptr;
        size_t g_fixupBytes = 0;

        bool ensure(void *&p, size_t &cap, size_t need) {
            if (cap >= need) return true;
            if (p != nullptr) cudaFree(p);
            p = nullptr;
            cap = 0;
            if (cudaMalloc(&p, need) != cudaSuccess) {
                p = nullptr;
                return false;
            }
            cap = need;
            return true;
        }

        // Shared-memory sizing for the K-quants:
        //   int8_tile_x_k = 84 ints for Q4_K/Q5_K, 76 for Q6_K;
        //   block_q8_1_int8 = 144 B per 128 activation values.
        size_t gemmSmemBytes(int tileX, int type) {
            const bool q8Tile = type == kTypeQ6K_ || type == kTypeQ8_0_ ||
                                type == kTypeIQ3XXS_;
            const size_t tileXK = q8Tile ? 76u : 84u;
            const size_t nbsIds = static_cast<size_t>(tileX) * sizeof(int);
            const size_t nbsX =
                    static_cast<size_t>(kInt8Y) * tileXK * sizeof(int);
            const size_t nbsY = static_cast<size_t>(tileX) * 144u;
            const size_t pad = 8u * 32u * sizeof(int);
            return nbsIds + nbsX + ((nbsY + pad - 1u) / pad) * pad;
        }

        template<typename Fn>
        bool setSmemAttrImpl(Fn fn, int bytes) {
            if (bytes <= 48 * 1024) return true;
            return cudaFuncSetAttribute(
                           reinterpret_cast<const void *>(fn),
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           bytes) == cudaSuccess;
        }

        bool setSmemAttr(int tileX, int type) {
            const int bytes = static_cast<int>(gemmSmemBytes(tileX, type));
            if (type == kTypeQ4K_) {
                if (tileX == 128) return setSmemAttrImpl(int8GemmStreamKQ4KTile128, bytes);
                if (tileX == 64) return setSmemAttrImpl(int8GemmStreamKQ4KTile64, bytes);
                return setSmemAttrImpl(int8GemmStreamKQ4KTile32, bytes);
            }
            if (type == kTypeQ5K_) {
                if (tileX == 128) return setSmemAttrImpl(int8GemmStreamKQ5KTile128, bytes);
                if (tileX == 64) return setSmemAttrImpl(int8GemmStreamKQ5KTile64, bytes);
                return setSmemAttrImpl(int8GemmStreamKQ5KTile32, bytes);
            }
            if (type == kTypeQ8_0_) {
                if (tileX == 128) return setSmemAttrImpl(int8GemmStreamKQ80Tile128, bytes);
                if (tileX == 64) return setSmemAttrImpl(int8GemmStreamKQ80Tile64, bytes);
                return setSmemAttrImpl(int8GemmStreamKQ80Tile32, bytes);
            }
            if (type == kTypeIQ2S_) {
                if (tileX == 128) return setSmemAttrImpl(int8GemmStreamKIQ2STile128, bytes);
                if (tileX == 64) return setSmemAttrImpl(int8GemmStreamKIQ2STile64, bytes);
                return setSmemAttrImpl(int8GemmStreamKIQ2STile32, bytes);
            }
            if (type == kTypeIQ3XXS_) {
                if (tileX == 128) return setSmemAttrImpl(int8GemmStreamKIQ3XXSTile128, bytes);
                if (tileX == 64) return setSmemAttrImpl(int8GemmStreamKIQ3XXSTile64, bytes);
                return setSmemAttrImpl(int8GemmStreamKIQ3XXSTile32, bytes);
            }
            if (tileX == 128) return setSmemAttrImpl(int8GemmStreamKQ6KTile128, bytes);
            if (tileX == 64) return setSmemAttrImpl(int8GemmStreamKQ6KTile64, bytes);
            return setSmemAttrImpl(int8GemmStreamKQ6KTile32, bytes);
        }

        bool launchSk(int tileX, int type, const void *w, const int *yq,
                      float *out, int k, int n, int m, int strideRowX,
                      float *fixup, int nblocks, int ntiles, cudaStream_t s) {
            const size_t smem = gemmSmemBytes(tileX, type);
            const dim3 grid(nblocks, 1, 1);
            const dim3 block(32, 8, 1);
            const char *wc = static_cast<const char *>(w);
            if (type == kTypeQ4K_) {
                if (tileX == 128)
                    int8GemmStreamKQ4KTile128<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else if (tileX == 64)
                    int8GemmStreamKQ4KTile64<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else
                    int8GemmStreamKQ4KTile32<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
            } else if (type == kTypeQ5K_) {
                if (tileX == 128)
                    int8GemmStreamKQ5KTile128<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else if (tileX == 64)
                    int8GemmStreamKQ5KTile64<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else
                    int8GemmStreamKQ5KTile32<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
            } else if (type == kTypeQ8_0_) {
                if (tileX == 128)
                    int8GemmStreamKQ80Tile128<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else if (tileX == 64)
                    int8GemmStreamKQ80Tile64<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else
                    int8GemmStreamKQ80Tile32<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
            } else if (type == kTypeIQ2S_) {
                if (tileX == 128)
                    int8GemmStreamKIQ2STile128<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else if (tileX == 64)
                    int8GemmStreamKIQ2STile64<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else
                    int8GemmStreamKIQ2STile32<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
            } else if (type == kTypeIQ3XXS_) {
                if (tileX == 128)
                    int8GemmStreamKIQ3XXSTile128<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else if (tileX == 64)
                    int8GemmStreamKIQ3XXSTile64<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else
                    int8GemmStreamKIQ3XXSTile32<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
            } else if (type == kTypeQ6K_) {
                if (tileX == 128)
                    int8GemmStreamKQ6KTile128<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else if (tileX == 64)
                    int8GemmStreamKQ6KTile64<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else
                    int8GemmStreamKQ6KTile32<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
            } else {
                return false;
            }
            if (cudaGetLastError() != cudaSuccess) return false;
            if (ntiles % nblocks == 0) return true;
            // Fold the partial tiles stream-k split across blocks.
            const dim3 fblock(32, 8, 1);
            if (type == kTypeQ4K_) {
                if (tileX == 128)
                    int8GemmFixupQ4KTile128<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else if (tileX == 64)
                    int8GemmFixupQ4KTile64<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else
                    int8GemmFixupQ4KTile32<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
            } else if (type == kTypeQ5K_) {
                if (tileX == 128)
                    int8GemmFixupQ5KTile128<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else if (tileX == 64)
                    int8GemmFixupQ5KTile64<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else
                    int8GemmFixupQ5KTile32<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
            } else if (type == kTypeQ8_0_) {
                if (tileX == 128)
                    int8GemmFixupQ80Tile128<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else if (tileX == 64)
                    int8GemmFixupQ80Tile64<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else
                    int8GemmFixupQ80Tile32<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
            } else if (type == kTypeIQ2S_) {
                if (tileX == 128)
                    int8GemmFixupIQ2STile128<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else if (tileX == 64)
                    int8GemmFixupIQ2STile64<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else
                    int8GemmFixupIQ2STile32<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
            } else if (type == kTypeIQ3XXS_) {
                if (tileX == 128)
                    int8GemmFixupIQ3XXSTile128<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else if (tileX == 64)
                    int8GemmFixupIQ3XXSTile64<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else
                    int8GemmFixupIQ3XXSTile32<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
            } else {
                if (tileX == 128)
                    int8GemmFixupQ6KTile128<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else if (tileX == 64)
                    int8GemmFixupQ6KTile64<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else
                    int8GemmFixupQ6KTile32<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
            }
            return cudaGetLastError() == cudaSuccess;
        }
    }// namespace

    bool int8GemmAvailable() {
        static const bool enabled = [] {
            const char *e = std::getenv("TINYCODER_INT8_GEMM");
            if (e != nullptr && e[0] != '\0' && e[0] == '0') return false;
            return ctx().ok;
        }();
        return enabled;
    }

    namespace {
        bool g_q8GemmEnabled = true;
    }

    void int8GemmSetQ8Enabled(bool enabled) { g_q8GemmEnabled = enabled; }

    bool int8Gemm(int type, const void *w, const float *x, float *out,
                 uint32_t seqLen, uint32_t rows, uint32_t cols,
                 uint32_t blocksPerRow, void *stream) {
        if (!int8GemmAvailable()) return false;
        if (type != kTypeQ4K_ && type != kTypeQ5K_ && type != kTypeQ6K_ &&
            type != kTypeQ8_0_ && type != kTypeIQ2S_ && type != kTypeIQ3XXS_)
            return false;
        // Q8-dominant checkpoints hit the broken sm_75 Q80 tile loader; the
        // adapter disables it for those models only.
        if (type == kTypeQ8_0_ && !g_q8GemmEnabled) return false;
        if (seqLen < 8 || rows == 0 || cols == 0) return false;
        // The activation quantizer pads each row to 512 values and the kernel
        // derives its y row stride from that padding; every dense qwen35
        // projection is aligned (4096 / 12288 / 2048).
        if ((cols & 511u) != 0u) return false;

        const Int8Ctx &c = ctx();
        const size_t yBytes =
                static_cast<size_t>(seqLen) * (static_cast<size_t>(cols) / 128u) * 144u;
        if (!ensure(g_yq, g_yqBytes, yBytes)) return false;

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        if (type == kTypeQ6K_ || type == kTypeQ8_0_ || type == kTypeIQ2S_ ||
            type == kTypeIQ3XXS_) {
            int8QuantizeQ8D4(
                    x, nullptr, g_yq, type, static_cast<int64_t>(cols),
                    static_cast<int64_t>(cols), 0, 0, static_cast<int64_t>(cols),
                    static_cast<int64_t>(seqLen), 1, 1, static_cast<void *>(s));
        } else {
            int8QuantizeQ8Ds4(
                    x, nullptr, g_yq, type, static_cast<int64_t>(cols),
                    static_cast<int64_t>(cols), 0, 0, static_cast<int64_t>(cols),
                    static_cast<int64_t>(seqLen), 1, 1, static_cast<void *>(s));
        }
        if (cudaGetLastError() != cudaSuccess) return false;

        int tileX = 128;
        if (seqLen < 64 || gemmSmemBytes(128, type) > static_cast<size_t>(c.smpbo))
            tileX = 64;
        if (const char *mx = std::getenv("TINYCODER_INT8_GEMM_X")) {
            const int v = std::atoi(mx);
            if (v == 32 || v == 64 || v == 128) tileX = v;
        }
        if (seqLen < 32 || gemmSmemBytes(64, type) > static_cast<size_t>(c.smpbo))
            tileX = 32;
        if (!setSmemAttr(tileX, type)) return false;

        const int ntx = (static_cast<int>(seqLen) + tileX - 1) / tileX;
        const int nty = (static_cast<int>(rows) + kInt8Y - 1) / kInt8Y;
        const int ntiles = ntx * nty;
        // Stream-k splits the tile space across `nblocks`.  The baseline uses one
        // block per SM; that is 8 warps/SM and can leave the memory system
        // under-driven for the 128-row tiles.  Tunable via
        // TINYCODER_INT8_GEMM_BLOCKS (1 = baseline behaviour).
        // Q8_0 wants one block per tile: stream-k splitting costs it the fixup
        // round-trip and measured 2,497 -> 2,828 pp512 (9B q8_0) when the block
        // count reaches ntiles.  The K-quant and IQ kernels are unaffected or
        // regress (iq3_xxs 2,745 -> 2,552), so this stays Q8_0-only.
        int blocksPerSm = (type == kTypeQ8_0_) ? 8 : 1;
        if (const char *bm = std::getenv("TINYCODER_INT8_GEMM_BLOCKS")) {
            const int v = std::atoi(bm);
            // Raised from 4: with blocksPerSm >= ceil(ntiles/nSM) the block count
            // reaches ntiles, so no tile is split and the stream-k fixup traffic
            // disappears (each block owns whole tiles).
            if (v >= 1 && v <= 64) blocksPerSm = v;
        }
        const int nmax = c.nsm * blocksPerSm;
        const int nblocks = ntiles < nmax ? ntiles : nmax;
        const size_t fixBytes =
                static_cast<size_t>(nblocks) * tileX * kInt8Y * sizeof(float);
        if (!ensure(g_fixup, g_fixupBytes, fixBytes)) return false;

        if (!launchSk(tileX, type, w, static_cast<const int *>(g_yq), out,
                      static_cast<int>(cols), static_cast<int>(rows),
                      static_cast<int>(seqLen), static_cast<int>(blocksPerRow),
                      static_cast<float *>(g_fixup), nblocks, ntiles, s)) {
            std::fprintf(stderr,
                         "[int8] launch failed (type=%d k=%u n=%u m=%u x=%d "
                         "tiles=%d blocks=%d)\n",
                         type, cols, rows, seqLen, tileX, ntiles, nblocks);
            return false;
        }
        return true;
    }

    namespace {
        void *g_mvq = nullptr;
        size_t g_mvqBytes = 0;
    }// namespace

    void int8GemvReserve(uint32_t cols, uint32_t seqLen) {
        const uint32_t colsPadded = (cols + 511u) & ~511u;
        const size_t need =
                static_cast<size_t>(seqLen) * (colsPadded / 32u) * 36u;
        (void) ensure(g_mvq, g_mvqBytes, need);
    }

    extern "C" __global__ void gemvIQ3XXSF32Row4(
            const void *vx, const void *vy, float *dst, int ncols, int nrows,
            int strideY);

    extern "C" __global__ void gemvIQ2XSFusedGUF32PlainCuda1(
            const void *vg, const void *vu, const void *vy, float *gout,
            float *uout, int ncols, int nrows, int strideY);

    extern "C" __global__ void gemvIQ3XXSFusedGUF32PlainCuda1(
            const void *vg, const void *vu, const void *vy, float *gout,
            float *uout, int ncols, int nrows, int strideY);

    extern "C" __global__ void gemvQ4KFusedGUF32PlainCuda1(
            const void *vg, const void *vu, const void *vy, float *gout,
            float *uout, int ncols, int nrows, int strideY);

    extern "C" __global__ void gemvQ6KF32W8(
            const void *vx, const void *vy, float *dst, int ncols, int nrows,
            int strideY);

    bool int8GemvDualQ4K(const void *wg, const void *wu, const float *x,
                         float *gout, float *uout, uint32_t rows,
                         uint32_t cols, void *stream) {
        if (!int8GemmAvailable() || rows == 0 || (cols & 255u) != 0)
            return false;
        const uint32_t colsPadded = (cols + 511u) & ~511u;
        const size_t need = static_cast<size_t>(colsPadded / 32u) * 36u;
        if (!ensure(g_mvq, g_mvqBytes, need)) return false;
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        gemvLaunchQuantizeQ8F32(x, g_mvq, static_cast<int>(cols),
                                static_cast<int>(colsPadded), 1,
                                static_cast<void *>(s));
        gemvQ4KFusedGUF32PlainCuda1<<<dim3((rows + 3) / 4), dim3(32, 4), 0, s>>>(
                wg, wu, g_mvq, gout, uout, static_cast<int>(cols),
                static_cast<int>(rows), 0);
        return cudaGetLastError() == cudaSuccess;
    }

    bool int8GemvDualIQ2XS(const void *wg, const void *wu, const float *x,
                           float *gout, float *uout, uint32_t rows,
                           uint32_t cols, void *stream) {
        if (!int8GemmAvailable() || rows == 0 || cols == 0 ||
            (cols & 255u) != 0)
            return false;
        const uint32_t colsPadded = (cols + 511u) & ~511u;
        const size_t need = static_cast<size_t>(colsPadded / 32u) * 36u;
        if (!ensure(g_mvq, g_mvqBytes, need)) return false;
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        gemvLaunchQuantizeQ8F32(x, g_mvq, static_cast<int>(cols),
                                static_cast<int>(colsPadded), 1,
                                static_cast<void *>(s));
        gemvIQ2XSFusedGUF32PlainCuda1<<<dim3((rows + 3) / 4), dim3(32, 4), 0, s>>>(
                wg, wu, g_mvq, gout, uout, static_cast<int>(cols),
                static_cast<int>(rows), 0);
        return cudaGetLastError() == cudaSuccess;
    }

    bool int8GemvDualIQ3(const void *wg, const void *wu, const float *x,
                         float *gout, float *uout, uint32_t rows,
                         uint32_t cols, void *stream) {
        if (!int8GemmAvailable() || rows == 0 || cols == 0 ||
            (cols & 255u) != 0)
            return false;
        const uint32_t colsPadded = (cols + 511u) & ~511u;
        const size_t need = static_cast<size_t>(colsPadded / 32u) * 36u;
        if (!ensure(g_mvq, g_mvqBytes, need)) return false;
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        gemvLaunchQuantizeQ8F32(x, g_mvq, static_cast<int>(cols),
                                static_cast<int>(colsPadded), 1,
                                static_cast<void *>(s));
        gemvIQ3XXSFusedGUF32PlainCuda1<<<dim3((rows+3)/4), dim3(32, 4), 0, s>>>(
                wg, wu, g_mvq, gout, uout, static_cast<int>(cols),
                static_cast<int>(rows), 0);
        return cudaGetLastError() == cudaSuccess;
    }

    bool int8Gemv(int type, const void *w, const float *x, float *out,
                  uint32_t seqLen, uint32_t rows, uint32_t cols, void *stream) {
        if (!int8GemmAvailable()) return false;
        if (seqLen == 0 || seqLen > 8 || rows == 0 || cols == 0) return false;
        if (type != kTypeQ4K_ && type != kTypeQ5K_ && type != kTypeQ6K_ &&
            type != kTypeQ8_0_ && type != kTypeQ2K_ && type != kTypeQ3K_ &&
            type != kTypeIQ2S_ && type != kTypeIQ2XS_ &&
            type != kTypeIQ3S_ && type != kTypeIQ3XXS_)
            return false;
        // block_q8_1 is 36 B per 32 values; rows are padded to 512 values.
        const uint32_t colsPadded = (cols + 511u) & ~511u;
        const size_t need = static_cast<size_t>(seqLen) * (colsPadded / 32u) * 36u;
        if (!ensure(g_mvq, g_mvqBytes, need)) return false;

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        gemvLaunchQuantizeQ8F32(x, g_mvq, static_cast<int>(cols),
                                           static_cast<int>(colsPadded),
                                           static_cast<int>(seqLen),
                                           static_cast<void *>(s));
        if (cudaGetLastError() != cudaSuccess) return false;
        switch (type) {
            case kTypeQ4K_:
                gemvLaunchQ4KF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeQ5K_:
                gemvLaunchQ5KF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeQ6K_:
                gemvLaunchQ6KF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeIQ2XS_:
                gemvLaunchIQ2XSF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeIQ3S_:
                gemvLaunchIQ3SF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeIQ2S_:
                gemvLaunchIQ2SF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeIQ3XXS_:
                gemvLaunchIQ3XXSF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeQ2K_:
                gemvLaunchQ2KF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeQ3K_:
                gemvLaunchQ3KF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            default:
                gemvLaunchQ80F32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
        }
        return cudaGetLastError() == cudaSuccess;
    }

}// namespace tinycoder::gpu

#endif// USE_CUDA
