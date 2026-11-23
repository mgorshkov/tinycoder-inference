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
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#define DECL_INT8_SK(N)                                                          \
    __global__ void int8GemmStreamKQ4KTile##N(                                   \
            const char *x, const int *y, float *dst, float *tmp_fixup,           \
            const int ncols_x, const int nrows_x, const int ncols_dst,           \
            const int stride_row_x, const int ncols_y, const int stride_col_dst, \
            const int ncols_max);                                                \
    __global__ void int8GemmStreamKQ5KTile##N(                                   \
            const char *x, const int *y, float *dst, float *tmp_fixup,           \
            const int ncols_x, const int nrows_x, const int ncols_dst,           \
            const int stride_row_x, const int ncols_y, const int stride_col_dst, \
            const int ncols_max);                                                \
    __global__ void int8GemmStreamKQ6KTile##N(                                   \
            const char *x, const int *y, float *dst, float *tmp_fixup,           \
            const int ncols_x, const int nrows_x, const int ncols_dst,           \
            const int stride_row_x, const int ncols_y, const int stride_col_dst, \
            const int ncols_max);                                                \
    __global__ void int8GemmStreamKQ80Tile##N(                                   \
            const char *x, const int *y, float *dst, float *tmp_fixup,           \
            const int ncols_x, const int nrows_x, const int ncols_dst,           \
            const int stride_row_x, const int ncols_y, const int stride_col_dst, \
            const int ncols_max);                                                \
    __global__ void int8GemmFixupQ4KTile##N(                                     \
            float *dst, const float *tmp_fixup, const int ncols_x,               \
            const int nrows_x, const int ncols_dst, const int stride_col_dst,    \
            const int ncols_max);                                                \
    __global__ void int8GemmFixupQ5KTile##N(                                     \
            float *dst, const float *tmp_fixup, const int ncols_x,               \
            const int nrows_x, const int ncols_dst, const int stride_col_dst,    \
            const int ncols_max);                                                \
    __global__ void int8GemmFixupQ6KTile##N(                                     \
            float *dst, const float *tmp_fixup, const int ncols_x,               \
            const int nrows_x, const int ncols_dst, const int stride_col_dst,    \
            const int ncols_max);                                                \
    __global__ void int8GemmStreamKPQ20Tile##N(                                  \
            const char *x, const int *y, float *dst, float *tmp_fixup,           \
            const int ncols_x, const int nrows_x, const int ncols_dst,           \
            const int stride_row_x, const int ncols_y, const int stride_col_dst, \
            const int ncols_max);                                                \
    __global__ void int8GemmStreamKPTQ10Tile##N(                                 \
            const char *x, const int *y, float *dst, float *tmp_fixup,           \
            const int ncols_x, const int nrows_x, const int ncols_dst,           \
            const int stride_row_x, const int ncols_y, const int stride_col_dst, \
            const int ncols_max);                                                \
    __global__ void int8GemmFixupPTQ10Tile##N(                                   \
            float *dst, const float *tmp_fixup, const int ncols_x,               \
            const int nrows_x, const int ncols_dst, const int stride_col_dst,    \
            const int ncols_max);                                                \
    __global__ void int8GemmFixupPQ20Tile##N(                                    \
            float *dst, const float *tmp_fixup, const int ncols_x,               \
            const int nrows_x, const int ncols_dst, const int stride_col_dst,    \
            const int ncols_max);                                                \
    __global__ void int8GemmFixupQ80Tile##N(                                     \
            float *dst, const float *tmp_fixup, const int ncols_x,               \
            const int nrows_x, const int ncols_dst, const int stride_col_dst,    \
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
__global__ void int8GemmStreamKIQ3STile32(const char *, const int *, float *, float *, int, int, int, int, int, int, int);
__global__ void int8GemmStreamKIQ3STile64(const char *, const int *, float *, float *, int, int, int, int, int, int, int);
__global__ void int8GemmStreamKIQ3STile128(const char *, const int *, float *, float *, int, int, int, int, int, int, int);
__global__ void int8GemmFixupIQ2STile32(float *, const float *, int, int, int, int, int);
__global__ void int8GemmFixupIQ2STile64(float *, const float *, int, int, int, int, int);
__global__ void int8GemmFixupIQ2STile128(float *, const float *, int, int, int, int, int);
__global__ void int8GemmFixupIQ3XXSTile32(float *, const float *, int, int, int, int, int);
__global__ void int8GemmFixupIQ3XXSTile64(float *, const float *, int, int, int, int, int);
__global__ void int8GemmFixupIQ3XXSTile128(float *, const float *, int, int, int, int, int);
__global__ void int8GemmFixupIQ3STile32(float *, const float *, int, int, int, int, int);
__global__ void int8GemmFixupIQ3STile64(float *, const float *, int, int, int, int, int);
__global__ void int8GemmFixupIQ3STile128(float *, const float *, int, int, int, int, int);

void int8QuantizeQ8Ds4(const void *x_f32, const int32_t *ids,
                       void *vy, int type_x, int64_t ne00,
                       int64_t s01, int64_t s02, int64_t s03,
                       int64_t ne0, int64_t ne1, int64_t ne2,
                       int64_t ne3, void *stream);
void int8QuantizeQ8D4(const void *x_f32, const int32_t *ids,
                      void *vy, int type_x, int64_t ne00,
                      int64_t s01, int64_t s02, int64_t s03,
                      int64_t ne0, int64_t ne1, int64_t ne2,
                      int64_t ne3, void *stream);// GEMV (decode/small-batch) launchers, Q8_1 plain layout.
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
// Expert-indexed Q8_1 GEMV: rowsPer*nUsed rows in one launch, weight rows
// reached through a device-side expert index (routed MoE, no host sync).
#define DECL_GEMV_EXPERT(tag)                                                 \
    void gemvLaunchExpert##tag(const void *vx, const void *vy, void *dst,     \
                               int ncols_x, int nrows_x, int stride_col_y,    \
                               const int *rowIdx, int rowsPer, void *stream);
DECL_GEMV_EXPERT(Q4K)
DECL_GEMV_EXPERT(Q5K)
DECL_GEMV_EXPERT(Q6K)
DECL_GEMV_EXPERT(Q80)
DECL_GEMV_EXPERT(IQ2S)
DECL_GEMV_EXPERT(IQ2XS)
DECL_GEMV_EXPERT(IQ3S)
DECL_GEMV_EXPERT(IQ3XXS)
DECL_GEMV_EXPERT(Q20)
// gemv/GemvExpertIq.cu: the same expert shape for the two i-quants the dense
// decode file does not carry.  They reuse gemm/QuantDot.cuh's vec_dots.
DECL_GEMV_EXPERT(IQ2XXS)
DECL_GEMV_EXPERT(IQ1S)
// Four-rows-per-warp variants of the same expert entries (decode gate/up):
// one warp per four rows, shared Q8_1 activation row, no stride_col_y.
#define DECL_GEMV_EXPERT4(tag)                                                \
    void gemvLaunchExpert4##tag(const void *vx, const void *vy, void *dst,    \
                                int ncols_x, int nrows_x, const int *rowIdx,  \
                                int rowsPer, void *stream);
DECL_GEMV_EXPERT4(IQ2XXS)
DECL_GEMV_EXPERT4(IQ2XS)
DECL_GEMV_EXPERT4(IQ2S)
DECL_GEMV_EXPERT4(IQ1S)
}

namespace tinycoder::gpu {
    namespace {
        constexpr int kTypeQ4K_ = 12;
        constexpr int kTypeQ5K_ = 13;
        constexpr int kTypeQ6K_ = 14;
        constexpr int kTypeQ8_0_ = 8;
        constexpr int kTypePQ2_0_ = 142; // Prism PQ2_0 ternary
        constexpr int kTypePTQ1_0_ = 143;// Prism native ternary, 28 B/128
        constexpr int kTypeQ2K_ = 10;
        constexpr int kTypeQ3K_ = 11;
        constexpr int kTypeIQ2S_ = 22;
        constexpr int kTypeIQ2XS_ = 17;
        constexpr int kTypeIQ3S_ = 21;
        constexpr int kTypeIQ3XXS_ = 18;
        constexpr int kTypeIQ2XXS_ = 16;
        constexpr int kTypeIQ1S_ = 19;
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
            // IQ3_S shares the IQ3_XXS/IQ8_0 activation-tile stride
            // (INT8_MMA_TILE_X_K_Q8_0); 76 ints/row covers it.
            const bool q8Tile = type == kTypeQ6K_ || type == kTypeQ8_0_ ||
                                type == kTypeIQ3XXS_ || type == kTypeIQ3S_ ||
                                type == kTypePQ2_0_ || type == kTypePTQ1_0_;
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
                if (tileX == 128)
                    setSmemAttrImpl(int8GemmStreamKQ4KTile128, bytes);
                else if (tileX == 64)
                    setSmemAttrImpl(int8GemmStreamKQ4KTile64, bytes);
                else
                    setSmemAttrImpl(int8GemmStreamKQ4KTile32, bytes);
            }
            if (type == kTypeQ5K_) {
                if (tileX == 128)
                    setSmemAttrImpl(int8GemmStreamKQ5KTile128, bytes);
                else if (tileX == 64)
                    setSmemAttrImpl(int8GemmStreamKQ5KTile64, bytes);
                else
                    setSmemAttrImpl(int8GemmStreamKQ5KTile32, bytes);
            }
            if (type == kTypeQ8_0_) {
                if (tileX == 128)
                    setSmemAttrImpl(int8GemmStreamKQ80Tile128, bytes);
                else if (tileX == 64)
                    setSmemAttrImpl(int8GemmStreamKQ80Tile64, bytes);
                else
                    setSmemAttrImpl(int8GemmStreamKQ80Tile32, bytes);
            }
            if (type == kTypePQ2_0_) {
                if (tileX == 128)
                    setSmemAttrImpl(int8GemmStreamKPQ20Tile128, bytes);
                else if (tileX == 64)
                    setSmemAttrImpl(int8GemmStreamKPQ20Tile64, bytes);
                else
                    setSmemAttrImpl(int8GemmStreamKPQ20Tile32, bytes);
            }
            if (type == kTypePTQ1_0_) {
                if (tileX == 128)
                    setSmemAttrImpl(int8GemmStreamKPTQ10Tile128, bytes);
                else if (tileX == 64)
                    setSmemAttrImpl(int8GemmStreamKPTQ10Tile64, bytes);
                else
                    setSmemAttrImpl(int8GemmStreamKPTQ10Tile32, bytes);
            }
            if (type == kTypeIQ2S_) {
                if (tileX == 128)
                    setSmemAttrImpl(int8GemmStreamKIQ2STile128, bytes);
                else if (tileX == 64)
                    setSmemAttrImpl(int8GemmStreamKIQ2STile64, bytes);
                else
                    setSmemAttrImpl(int8GemmStreamKIQ2STile32, bytes);
            }
            if (type == kTypeIQ3XXS_) {
                if (tileX == 128)
                    setSmemAttrImpl(int8GemmStreamKIQ3XXSTile128, bytes);
                else if (tileX == 64)
                    setSmemAttrImpl(int8GemmStreamKIQ3XXSTile64, bytes);
                else
                    setSmemAttrImpl(int8GemmStreamKIQ3XXSTile32, bytes);
            }
            if (type == kTypeIQ3S_) {
                if (tileX == 128)
                    setSmemAttrImpl(int8GemmStreamKIQ3STile128, bytes);
                else if (tileX == 64)
                    setSmemAttrImpl(int8GemmStreamKIQ3STile64, bytes);
                else
                    setSmemAttrImpl(int8GemmStreamKIQ3STile32, bytes);
            }
            if (tileX == 128) return setSmemAttrImpl(int8GemmStreamKQ6KTile128, bytes);
            if (tileX == 64) return setSmemAttrImpl(int8GemmStreamKQ6KTile64, bytes);
            return setSmemAttrImpl(int8GemmStreamKQ6KTile32, bytes);
        }

    namespace {
        // ---------------------------------------------------------------------
        // Fallback accounting for the int8 GEMM.
        //
        // `int8Gemm` returning false means the caller silently runs the integer
        // `kQGemmKxQ8K` kernel instead: correct, several times slower on a
        // prefill, and invisible in every number the engine reports.  That is
        // how a missing Q6_K launch branch survived here -- the only symptom
        // was one stderr line per call, and nothing read it.
        //
        // Only the *unexpected* failures are reported: the early "not my shape"
        // returns (small batch, unaligned cols, unsupported type) are the
        // dispatch doing its job, and logging them would fire on every decode
        // step.  Each distinct (reason, type, shape) warns once; a single total
        // line is printed at exit so a benchmark cannot end without the reader
        // learning that some fraction of the prefill never reached the tensor
        // cores.
        struct Int8Fallback {
            const char *reason;
            int type;
            uint32_t rows, cols, seqLen;
        };
        std::mutex &fallbackMutex() {
            static std::mutex m;
            return m;
        }
        std::vector<Int8Fallback> &fallbackSeen() {
            static std::vector<Int8Fallback> v;
            return v;
        }
        unsigned long &fallbackCount() {
            static unsigned long n = 0;
            return n;
        }

        void printFallbackTotal() {
            const unsigned long n = fallbackCount();
            if (n == 0) return;
            std::fprintf(stderr,
                         "[int8] %lu int8-GEMM call(s) fell back to the integer kernel "
                         "(%zu distinct shape(s) reported above); prefill did not use the "
                         "tensor cores for those\n",
                         n, fallbackSeen().size());
        }

        void noteInt8Fallback(const char *reason, int type, uint32_t rows,
                              uint32_t cols, uint32_t seqLen) {
            std::lock_guard<std::mutex> lock(fallbackMutex());
            ++fallbackCount();
            for (const auto &f: fallbackSeen()) {
                if (f.reason == reason && f.type == type && f.rows == rows &&
                    f.cols == cols && f.seqLen == seqLen) {
                    return;
                }
            }
            if (fallbackSeen().empty()) std::atexit(printFallbackTotal);
            fallbackSeen().push_back({reason, type, rows, cols, seqLen});
            std::fprintf(stderr,
                         "[int8] fallback: %s (type=%d k=%u n=%u m=%u) -- this shape now "
                         "runs the integer kernel, not the tensor cores\n",
                         reason, type, cols, rows, seqLen);
        }
    }// namespace

        // Q6_K on the int8 tensor path: on by default (it agrees with the fp64
        // reference on real weights to 3.4e-4, tools/int8_gemm_parity) and off
        // only to reproduce the old silent-fallback numbers.  The fallback is
        // kQGemmKxQ8K, which is correct but not tensor-core.
        bool q6kInt8Enabled() {
            static const bool on = [] {
                const char *e = std::getenv("TINYCODER_INT8_Q6K");
                return e == nullptr || e[0] != '0';
            }();
            return on;
        }

        bool launchSk(int tileX, int type, const void *w, const int *yq,
                      float *out, int k, int n, int m, int strideRowX,
                      float *fixup, int nblocks, int ntiles, cudaStream_t s) {
            const size_t smem = gemmSmemBytes(tileX, type);
            const dim3 block(32, 8, 1);
            const dim3 grid(nblocks, 1, 1);
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
            } else if (type == kTypeQ6K_ && q6kInt8Enabled()) {
                // Q6_K was wired everywhere around this switch -- the extern
                // declarations, the dynamic-smem attribute, the fixup kernels, the
                // supported-type guard at int8Gemm() and even the q8-wide tile loader
                // selection -- but this branch was missing, so every Q6_K projection
                // fell through to `return false`, logged "launch failed" and silently
                // ran the integer kQGemmKxQ8K fallback instead.  Q6_K is attn_qkv,
                // ffn_down and output on every Q4_K_M checkpoint.
                if (tileX == 128)
                    int8GemmStreamKQ6KTile128<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else if (tileX == 64)
                    int8GemmStreamKQ6KTile64<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else
                    int8GemmStreamKQ6KTile32<<<grid, block, smem, s>>>(
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
            } else if (type == kTypePQ2_0_) {
                if (tileX == 128)
                    int8GemmStreamKPQ20Tile128<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else if (tileX == 64)
                    int8GemmStreamKPQ20Tile64<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else
                    int8GemmStreamKPQ20Tile32<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
            } else if (type == kTypePTQ1_0_) {
                if (tileX == 128)
                    int8GemmStreamKPTQ10Tile128<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else if (tileX == 64)
                    int8GemmStreamKPTQ10Tile64<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else
                    int8GemmStreamKPTQ10Tile32<<<grid, block, smem, s>>>(
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
            } else if (type == kTypeIQ3S_) {
                if (tileX == 128)
                    int8GemmStreamKIQ3STile128<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else if (tileX == 64)
                    int8GemmStreamKIQ3STile64<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
                else
                    int8GemmStreamKIQ3STile32<<<grid, block, smem, s>>>(
                            wc, yq, out, fixup, k, n, m, strideRowX, m, n, m);
            } else {
                // A type that passed int8Gemm's supported-type guard but has no
                // branch here is exactly the defect the Q6_K omission was: the
                // caller falls back and nothing says why.
                noteInt8Fallback("no launch branch for this type", type,
                                 static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                                 static_cast<uint32_t>(m));
                return false;
            }
            if (cudaGetLastError() != cudaSuccess) {
                noteInt8Fallback("kernel launch error", type, static_cast<uint32_t>(n),
                                 static_cast<uint32_t>(k), static_cast<uint32_t>(m));
                return false;
            }
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
            } else if (type == kTypeIQ3S_) {
                if (tileX == 128)
                    int8GemmFixupIQ3STile128<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else if (tileX == 64)
                    int8GemmFixupIQ3STile64<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
                else
                    int8GemmFixupIQ3STile32<<<grid, fblock, 0, s>>>(out, fixup, k, n, m, n, m);
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

    // -------------------------------------------------------------------------
    // Per-shape path selection: int8 MMA vs the cuBLAS fp16 twin.
    //
    // Measured with `tools/int8_gemm_probe --models <set> --compare`, which
    // times BOTH paths for one shape in the same process and at the same SM
    // clock (int8 = int8Gemm including its activation quantise; fp16 = the
    // cublasGemmEx the engine issues over the resident f16Twin plus the
    // kF32ToF16 convert).  Ratios are clock-invariant, so the readings below
    // (1,350 / 1,890 MHz) are usable on a 1,965 MHz run.
    //
    // fps512 = 512; ms = one layer's call.
    //
    //   type           rows     cols     int8      fp16     winner
    //   IQ3_XXS       18,944   3,584    1.724     1.777    int8  (+3%)
    //   IQ3_XXS        8,960   1,536    0.420     0.394    fp16  (+7%)
    //   IQ3_XXS        3,584   3,584    0.403     0.370    fp16  (+9%)
    //   IQ3_XXS          512   3,584    0.122     0.078    fp16  (+56%)
    //   IQ3_XXS        3,584  18,944    1.587     1.320    fp16  (+20%)
    //   IQ3_XXS        1,536   1,536    0.124     0.094    fp16  (+32%)
    //   IQ3_S          3,584   3,584    0.319     0.271    fp16  (+18%)
    //   IQ3_S          1,536   1,536    0.125     0.094    fp16  (+33%)
    //   IQ2_S          3,584   3,584    0.387     0.271    fp16  (+43%)
    //   IQ2_S          2,048   1,536    0.165     0.097    fp16  (+70%)
    //   IQ2_S          1,536   1,536    0.147     0.093    fp16  (+58%)
    //   Q4_K          18,944   3,584    1.389     1.288    fp16  (+8%)
    //   Q4_K           3,584  18,944    1.604     1.318    fp16  (+22%)
    //   Q4_K           3,584   3,584    0.341     0.271    fp16  (+26%)
    //   Q4_K             512   3,584    0.102     0.065    fp16  (+57%)
    //
    // The pattern is not "small shapes lose": the int8 kernel wins exactly one
    // cell, the 18,944-row IQ3_XXS gate/up, because at 3.06 bpw its DRAM traffic
    // is low enough for the MMA to be the limit, while every wider-block type
    // (Q4_K 4.5 bpw) and every narrower shape is faster on cuBLAS fp16.  That is
    // why the table is a table and not a rows threshold.
    //
    // Only shapes listed here answer "fp16"; anything else keeps int8, so a
    // checkpoint that was never measured (9B q4_k_m, 27B, MoE) behaves exactly
    // as before.  TINYCODER_INT8_GEMM_DISPATCH=0 restores int8 everywhere.
    //
    // NOTE (2026-11-16): adding the dense qwen35 9B's shapes here was tried and
    // reverted -- `int8_gemm_probe --models 9b --compare` does show cuBLAS fp16
    // winning all of them (Q4_K 12,288x4,096 +18%, Q4_K 4,096x4,096 +33%,
    // Q5_K 4,096x4,096 +34%, Q8_0 4,096x4,096 +50%), but the 9B's prefill never
    // reaches this table: forwardQwen35Prefix routes every projection through
    // `dequantMatrixF16` (persistent fp16 twins) and does not consult
    // int8GemmPreferred.  Measured end to end: pp512 2,560.6 -> 2,559.5 tok/s,
    // i.e. no change.  See the ledger for 2026-11-16.
    namespace {
        struct Int8ShapeVerdict {
            uint32_t type, rows, cols;
        };
        constexpr Int8ShapeVerdict kFp16FasterShapes[] = {
                {kTypeIQ3XXS_, 8960, 1536},
                {kTypeIQ3XXS_, 3584, 3584},
                {kTypeIQ3XXS_, 512, 3584},
                {kTypeIQ3XXS_, 3584, 18944},
                {kTypeIQ3XXS_, 1536, 1536},
                {kTypeIQ3S_, 3584, 3584},
                {kTypeIQ3S_, 1536, 1536},
                {kTypeIQ2S_, 3584, 3584},
                {kTypeIQ2S_, 2048, 1536},
                {kTypeIQ2S_, 1536, 1536},
                {kTypeQ4K_, 18944, 3584},
                {kTypeQ4K_, 3584, 18944},
                {kTypeQ4K_, 3584, 3584},
                {kTypeQ4K_, 512, 3584},
        };
    }// namespace

    bool int8GemmPreferred(uint32_t type, uint32_t rows, uint32_t cols) {
        static const bool dispatchOn = [] {
            const char *e = std::getenv("TINYCODER_INT8_GEMM_DISPATCH");
            return !(e != nullptr && e[0] != '\0' && e[0] == '0');
        }();
        if (!dispatchOn) return true;
        for (const Int8ShapeVerdict &v: kFp16FasterShapes)
            if (v.type == type && v.rows == rows && v.cols == cols) return false;
        return true;
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
            type != kTypeQ8_0_ && type != kTypeIQ2S_ && type != kTypeIQ3XXS_ &&
            type != kTypeIQ3S_ && type != kTypePQ2_0_ &&
            type != kTypePTQ1_0_)
            return false;
        // Q8-dominant checkpoints hit the broken sm_75 Q80 tile loader; the
        // adapter disables it for those models only.
        if (type == kTypeQ8_0_ && !g_q8GemmEnabled) return false;
        if (type == kTypeQ6K_ && !q6kInt8Enabled()) {
            noteInt8Fallback("Q6_K int8 path disabled by TINYCODER_INT8_Q6K", type,
                             rows, cols, seqLen);
            return false;
        }
        if (seqLen < 8 || rows == 0 || cols == 0) {
            noteInt8Fallback("seqLen/rows/cols below the int8 floor", type, rows,
                             cols, seqLen);
            return false;
        }
        // The activation quantizer packs one Q8_1 block per 32 values and writes
        // them in 128-value groups; a partial final group is simply not read by
        // the kernel (it loops k/qk weight blocks, i.e. floor(k/128) groups for
        // every type that reaches this point).  So K only has to be a multiple
        // of 128, not of 512: Inkling's 2304 = 18*128 and 1152 = 9*128 dense
        // projections are aligned, and their quantized rows are exactly
        // (cols/128)*144 bytes, which is what yBytes below allocates.
        if ((cols & 127u) != 0u) {
            noteInt8Fallback("cols is not a multiple of 128", type, rows, cols,
                             seqLen);
            return false;
        }

        const Int8Ctx &c = ctx();
        const size_t yBytes =
                static_cast<size_t>(seqLen) * (static_cast<size_t>(cols) / 128u) * 144u;
        if (!ensure(g_yq, g_yqBytes, yBytes)) {
            noteInt8Fallback("cannot allocate the Q8_1 activation scratch", type,
                             rows, cols, seqLen);
            return false;
        }

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        if (type == kTypeQ6K_ || type == kTypeQ8_0_ || type == kTypeIQ2S_ ||
            type == kTypeIQ3XXS_ || type == kTypeIQ3S_ ||
            type == kTypePQ2_0_ || type == kTypePTQ1_0_) {
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
        if (cudaGetLastError() != cudaSuccess) {
            noteInt8Fallback("activation quantize launch failed", type, rows,
                             cols, seqLen);
            return false;
        }

        int tileX = 128;
        if (seqLen < 64 || gemmSmemBytes(128, type) > static_cast<size_t>(c.smpbo))
            tileX = 64;
        if (const char *mx = std::getenv("TINYCODER_INT8_GEMM_X")) {
            const int v = std::atoi(mx);
            if (v == 32 || v == 64 || v == 128) tileX = v;
        }
        if (seqLen < 32 || gemmSmemBytes(64, type) > static_cast<size_t>(c.smpbo))
            tileX = 32;
        if (!setSmemAttr(tileX, type)) {
            noteInt8Fallback("cannot raise the dynamic-smem limit", type, rows, cols, seqLen);
            return false;
        }

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
        // PQ2_0 wants ONE block per SM: with the cheap byte_perm decode the
        // stream-k k-split only adds the fixup round-trip.  Measured pp512:
        // B=1 823.4, B=2 787.7, B=4 770.6, B=8 756.6, B=16 707.1, B=32 631.4.
        // (Q8_0 separately wants 8; the two types disagree.)
        int blocksPerSm = type == kTypeQ8_0_ ? 8 : 1;
        if (const char *bm = std::getenv("TINYCODER_INT8_GEMM_BLOCKS")) {
            const int v = std::atoi(bm);
            // Raised from 4: with blocksPerSm >= ceil(ntiles/nSM) the block count
            // reaches ntiles, so no tile is split and the stream-k fixup traffic
            // disappears (each block owns whole tiles).
            if (v >= 1 && v <= 64) blocksPerSm = v;
        }
        const int nmax = c.nsm * blocksPerSm;
        // Stream-k may legitimately ask for MORE blocks than there are tiles:
        // the continuous k-block partition then splits a tile's k range across
        // blocks and `int8GemmFixup*` folds the partial sums.  Clamping to
        // `ntiles` instead leaves thin output matrices on a fraction of the
        // card: the 7B GQA k/v projections are 512 rows x 512 tokens = 4x4 =
        // 16 tiles, i.e. 16 of 68 SMs, measured at 12.95 TFLOPS against 55.1
        // for a shape that fills the grid.  `TINYCODER_INT8_GEMM_FILL=0`
        // restores the old clamp.
        static const bool fillGrid = [] {
            const char *e = std::getenv("TINYCODER_INT8_GEMM_FILL");
            return e == nullptr || e[0] != '0';
        }();
        const bool underfilled = ntiles < nmax;
        const int nblocks = !underfilled ? nmax : (fillGrid ? nmax : ntiles);
        const size_t fixBytes =
                static_cast<size_t>(nblocks) * tileX * kInt8Y * sizeof(float);
        if (!ensure(g_fixup, g_fixupBytes, fixBytes)) return false;

        if (!launchSk(tileX, type, w, static_cast<const int *>(g_yq), out,
                      static_cast<int>(cols), static_cast<int>(rows),
                      static_cast<int>(seqLen), static_cast<int>(blocksPerRow),
                      static_cast<float *>(g_fixup), nblocks, ntiles, s)) {
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
        gemvIQ3XXSFusedGUF32PlainCuda1<<<dim3((rows + 3) / 4), dim3(32, 4), 0, s>>>(
                wg, wu, g_mvq, gout, uout, static_cast<int>(cols),
                static_cast<int>(rows), 0);
        return cudaGetLastError() == cudaSuccess;
    }

    bool int8GemvExpert(int type, const void *w, const float *x, float *out,
                        uint32_t rowsPer, uint32_t cols, const int32_t *idx,
                        uint32_t nUsed, uint32_t xStride, void *stream) {
        if (!int8GemmAvailable()) return false;
        if (rowsPer == 0 || cols == 0 || nUsed == 0 || idx == nullptr)
            return false;
        // xStride 0: every expert reads the same activation row (gate/up).
        // xStride == cols: the per-expert rows are contiguous (down).
        if (xStride != 0 && xStride != cols) return false;
        const uint32_t rows = rowsPer * nUsed;
        const uint32_t colsPadded = (cols + 511u) & ~511u;
        const uint32_t quantRows = (xStride == 0) ? 1u : nUsed;
        const size_t need = static_cast<size_t>(quantRows) *
                            (colsPadded / 32u) * 36u;
        if (!ensure(g_mvq, g_mvqBytes, need)) return false;

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        gemvLaunchQuantizeQ8F32(x, g_mvq, static_cast<int>(cols),
                                static_cast<int>(colsPadded),
                                static_cast<int>(quantRows),
                                static_cast<void *>(s));
        if (cudaGetLastError() != cudaSuccess) return false;
        const int strideY =
                (xStride == 0) ? 0 : static_cast<int>(colsPadded / 32u);
        switch (type) {
            case kTypeQ4K_:
                gemvLaunchExpertQ4K(w, g_mvq, out, static_cast<int>(cols),
                                    static_cast<int>(rows), strideY, idx,
                                    static_cast<int>(rowsPer), s);
                break;
            case kTypeQ5K_:
                gemvLaunchExpertQ5K(w, g_mvq, out, static_cast<int>(cols),
                                    static_cast<int>(rows), strideY, idx,
                                    static_cast<int>(rowsPer), s);
                break;
            case kTypeQ6K_:
                gemvLaunchExpertQ6K(w, g_mvq, out, static_cast<int>(cols),
                                    static_cast<int>(rows), strideY, idx,
                                    static_cast<int>(rowsPer), s);
                break;
            case kTypeQ8_0_:
                gemvLaunchExpertQ80(w, g_mvq, out, static_cast<int>(cols),
                                    static_cast<int>(rows), strideY, idx,
                                    static_cast<int>(rowsPer), s);
                break;
            case kTypeIQ2XS_:
                gemvLaunchExpertIQ2XS(w, g_mvq, out, static_cast<int>(cols),
                                      static_cast<int>(rows), strideY, idx,
                                      static_cast<int>(rowsPer), s);
                break;
            case kTypeIQ2S_:
                gemvLaunchExpertIQ2S(w, g_mvq, out, static_cast<int>(cols),
                                     static_cast<int>(rows), strideY, idx,
                                     static_cast<int>(rowsPer), s);
                break;
            case kTypeIQ3S_:
                gemvLaunchExpertIQ3S(w, g_mvq, out, static_cast<int>(cols),
                                     static_cast<int>(rows), strideY, idx,
                                     static_cast<int>(rowsPer), s);
                break;
            case kTypeIQ3XXS_:
                gemvLaunchExpertIQ3XXS(w, g_mvq, out, static_cast<int>(cols),
                                       static_cast<int>(rows), strideY, idx,
                                       static_cast<int>(rowsPer), s);
                break;
            case 42:// Q2_0, the expert-down type on the IQ2_XXS checkpoint
                gemvLaunchExpertQ20(w, g_mvq, out, static_cast<int>(cols),
                                    static_cast<int>(rows), strideY, idx,
                                    static_cast<int>(rowsPer), s);
                break;
            // IQ2_XXS / IQ1_S: the routed gate/up experts of the Inkling
            // IQ2_XXS checkpoint.  Same expert index, same Q8_1 activation
            // buffer and same launch shape as the types above; only the
            // dot-product body differs (gemv/GemvExpertIq.cu).
            case kTypeIQ2XXS_:
                gemvLaunchExpertIQ2XXS(w, g_mvq, out, static_cast<int>(cols),
                                       static_cast<int>(rows), strideY, idx,
                                       static_cast<int>(rowsPer), s);
                break;
            case kTypeIQ1S_:
                gemvLaunchExpertIQ1S(w, g_mvq, out, static_cast<int>(cols),
                                     static_cast<int>(rows), strideY, idx,
                                     static_cast<int>(rowsPer), s);
                break;
            default:
                return false;
        }
        return cudaGetLastError() == cudaSuccess;
    }

    // Gate/up counterpart of int8GemvExpert, four rows per warp.
    //
    // The one-row kernel launches one block per output row, so 18432 blocks per
    // gate/up call each re-read the same Q8_1 activation row through a
    // four-warp reduction.  This entry serves only the shared-activation shape
    // (every expert reads the same row), which is exactly gate/up; it quantizes
    // that row once with the same gemvLaunchQuantizeQ8F32 and launches the
    // 4-rows-per-warp kernel in gemv/GemvExpertIq.cu.  The prefill path keeps
    // calling int8GemvExpert, so its arithmetic is untouched.
    bool int8GemvExpert4(int type, const void *w, const float *x, float *out,
                         uint32_t rowsPer, uint32_t cols, const int32_t *idx,
                         uint32_t nUsed, void *stream) {
        if (!int8GemmAvailable()) return false;
        if (rowsPer == 0 || cols == 0 || nUsed == 0 || idx == nullptr)
            return false;
        if (type != kTypeIQ2XXS_ && type != kTypeIQ2XS_ &&
            type != kTypeIQ2S_ && type != kTypeIQ1S_)
            return false;
        // One quant block per 256 weights is what the kernel strides over.
        if ((cols & 255u) != 0u) return false;
        const uint32_t colsPadded = (cols + 511u) & ~511u;
        const size_t need = static_cast<size_t>(colsPadded / 32u) * 36u;
        if (!ensure(g_mvq, g_mvqBytes, need)) return false;
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        gemvLaunchQuantizeQ8F32(x, g_mvq, static_cast<int>(cols),
                                static_cast<int>(colsPadded), 1,
                                static_cast<void *>(s));
        if (cudaGetLastError() != cudaSuccess) return false;
        const int rows = static_cast<int>(rowsPer * nUsed);
        const int c = static_cast<int>(cols);
        const int rp = static_cast<int>(rowsPer);
        switch (type) {
            case kTypeIQ2XXS_:
                gemvLaunchExpert4IQ2XXS(w, g_mvq, out, c, rows, idx, rp, s);
                break;
            case kTypeIQ2XS_:
                gemvLaunchExpert4IQ2XS(w, g_mvq, out, c, rows, idx, rp, s);
                break;
            case kTypeIQ2S_:
                gemvLaunchExpert4IQ2S(w, g_mvq, out, c, rows, idx, rp, s);
                break;
            case kTypeIQ1S_:
                gemvLaunchExpert4IQ1S(w, g_mvq, out, c, rows, idx, rp, s);
                break;
            default:
                return false;
        }
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
                        static_cast<int>(rows), static_cast<int>(colsPadded / 32u),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeQ5K_:
                gemvLaunchQ5KF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded / 32u),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeQ6K_:
                gemvLaunchQ6KF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded / 32u),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeIQ2XS_:
                gemvLaunchIQ2XSF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded / 32u),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeIQ3S_:
                gemvLaunchIQ3SF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded / 32u),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeIQ2S_:
                gemvLaunchIQ2SF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded / 32u),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeIQ3XXS_:
                gemvLaunchIQ3XXSF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded / 32u),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeQ2K_:
                gemvLaunchQ2KF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded / 32u),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            case kTypeQ3K_:
                gemvLaunchQ3KF32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded / 32u),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
            default:
                gemvLaunchQ80F32Plain(
                        w, g_mvq, out, static_cast<int>(cols),
                        static_cast<int>(rows), static_cast<int>(colsPadded / 32u),
                        static_cast<int>(rows), static_cast<int>(seqLen),
                        static_cast<void *>(s));
                break;
        }
        return cudaGetLastError() == cudaSuccess;
    }

}// namespace tinycoder::gpu

#endif// USE_CUDA
