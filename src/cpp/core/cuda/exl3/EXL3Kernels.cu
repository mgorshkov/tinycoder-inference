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

// EXL3 trellis decode + reconstruct + GEMV kernels (Turing / Ampere / Ada; the
// reference target is the sm_75 RTX 2080 Ti, and the same code runs on 3090/4090
// and newer GPUs via forward-compatible PTX).
// See include/EXL3Kernels.hpp for the design overview.  This TU is deliberately
// self-contained (cuda_runtime + EXL3Kernels.hpp only) so the kernel layer can
// be built and parity-tested against EXL3Codec without pulling in the Model.

#include "EXL3Kernels.hpp"

#ifdef USE_CUDA

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace tinycoder::exl3gpu {

    namespace {

        // ---- Device tensor-core permutation (packed index -> row-major) ----
        // Mirror of EXL3Codec::tensorCorePerm() (quantize.py tensor_core_perm).
        __constant__ uint8_t c_perm[256];

        void ensurePermTable() {
            static std::once_flag once;
            std::call_once(once, [] {
                uint8_t t[256] = {0};
                for (int tile = 0; tile < 32; ++tile) {
                    const int r0 = (tile % 4) * 2;
                    const int r1 = r0 + 1;
                    const int r2 = r0 + 8;
                    const int r3 = r0 + 9;
                    const int c0 = tile / 4;
                    const int c1 = c0 + 8;
                    t[tile * 8 + 0] = static_cast<uint8_t>(r0 * 16 + c0);
                    t[tile * 8 + 1] = static_cast<uint8_t>(r1 * 16 + c0);
                    t[tile * 8 + 2] = static_cast<uint8_t>(r2 * 16 + c0);
                    t[tile * 8 + 3] = static_cast<uint8_t>(r3 * 16 + c0);
                    t[tile * 8 + 4] = static_cast<uint8_t>(r0 * 16 + c1);
                    t[tile * 8 + 5] = static_cast<uint8_t>(r1 * 16 + c1);
                    t[tile * 8 + 6] = static_cast<uint8_t>(r2 * 16 + c1);
                    t[tile * 8 + 7] = static_cast<uint8_t>(r3 * 16 + c1);
                }
                cudaMemcpyToSymbol(c_perm, t, 256);
            });
        }

        cudaStream_t &stream() {
            static cudaStream_t s = nullptr;
            return s;
        }
        std::once_flag &initFlag() {
            static std::once_flag f;
            return f;
        }
        bool g_ok = false;
        std::string g_initErr;

        bool initOnce() {
            std::call_once(initFlag(), [] {
                int dev = 0;
                if (cudaGetDevice(&dev) != cudaSuccess) {
                    g_initErr = "no CUDA device";
                    g_ok = false;
                    return;
                }
                if (cudaStreamCreateWithFlags(&stream(), cudaStreamNonBlocking) !=
                    cudaSuccess) {
                    g_initErr = "cudaStreamCreate failed";
                    g_ok = false;
                    return;
                }
                ensurePermTable();
                g_ok = true;
            });
            return g_ok;
        }

        // ---- Codebook decode (device mirror of EXL3Codec) ----
        __device__ __forceinline__ float h2f(uint16_t h) {
            return __half2float(__ushort_as_half(h));
        }
        __device__ __forceinline__ float decode3INST(uint32_t x) {
            x = x * 89226354u + 64248484u;
            x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
            return h2f(static_cast<uint16_t>(x & 0xffff)) +
                   h2f(static_cast<uint16_t>(x >> 16));
        }
        __device__ __forceinline__ float decodeMcG(uint32_t x) {
            x = x * 0xCBAC1FEDu;
            x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
            return h2f(static_cast<uint16_t>(x & 0xffff)) +
                   h2f(static_cast<uint16_t>(x >> 16));
        }
        __device__ __forceinline__ float decodeMul1(uint32_t x) {
            x = x * 0x83DCD12Du;
            uint32_t sum = 0x6400u + (x & 0xff) + ((x >> 8) & 0xff) +
                           ((x >> 16) & 0xff) + ((x >> 24) & 0xff);
            const float kInv = h2f(0x1eee);
            const float kBias = h2f(0xc931);
            return h2f(static_cast<uint16_t>(sum)) * kInv + kBias;
        }
        __device__ __forceinline__ float decodeCode(uint32_t code, int codebook) {
            if (codebook == 1) return decodeMcG(code);
            if (codebook == 2) return decodeMul1(code);
            return decode3INST(code);
        }

        // One thread unpacks one code (j = threadIdx.x < 256).
        __device__ __forceinline__ uint16_t unpackCode(const uint16_t *packed, uint32_t K,
                                                       uint32_t j) {
            const uint32_t mask = (K >= 16) ? 0xFFFFu : ((1u << K) - 1u);
            const uint32_t nWords = 8u * K;
            const uint32_t *w32 = reinterpret_cast<const uint32_t *>(packed);
            const uint32_t pos = j * K;
            const uint32_t word = pos >> 5;
            const uint32_t within = pos & 31;
            const uint32_t lo = w32[word];
            const uint32_t hi = w32[(word + 1) % nWords];
            const uint64_t merged = static_cast<uint64_t>(lo) |
                                    (static_cast<uint64_t>(hi) << 32);
            return static_cast<uint16_t>((merged >> within) & mask);
        }

        // Half-integer bitrate: code j is (KA+1) bits when j is even, KA when odd.
        __device__ __forceinline__ uint16_t unpackCodeFrac(const uint16_t *packed,
                                                           uint32_t KA, uint32_t j) {
            const uint32_t K = KA + 1;
            const uint32_t nWords = 8u * K;
            const uint32_t *w32 = reinterpret_cast<const uint32_t *>(packed);
            const uint32_t eCnt = (j + 1u) / 2u;// even j' < j
            const uint32_t oCnt = j / 2u;       // odd  j' < j
            const uint32_t bitpos = eCnt * (KA + 1u) + oCnt * KA;
            const uint32_t width = (j % 2u == 0u) ? (KA + 1u) : KA;
            const uint32_t mask = (width >= 32u) ? 0xFFFFFFFFu : ((1u << width) - 1u);
            const uint32_t word = bitpos >> 5;
            const uint32_t within = bitpos & 31;
            const uint32_t lo = w32[word];
            const uint32_t hi = w32[(word + 1) % nWords];
            const uint64_t merged = static_cast<uint64_t>(lo) |
                                    (static_cast<uint64_t>(hi) << 32);
            return static_cast<uint16_t>((merged >> within) & mask);
        }

        // ---- Decode one tile grid into the rotated-basis weight [k][n] ----
        // Grid: one block per (ki, ni) tile, 256 threads (one per code).
        __global__ void decodeTilesInt(const uint16_t *__restrict__ packed, uint32_t K,
                                        int codebook, uint32_t tilesN, uint32_t tileU16,
                                        float *__restrict__ recon, uint32_t n) {
            const uint32_t tile = blockIdx.x;
            const uint32_t ki = tile / tilesN;
            const uint32_t ni = tile % tilesN;
            const uint16_t *t = packed + static_cast<size_t>(tile) * tileU16;
            const uint32_t j = threadIdx.x;// 0..255
            const uint16_t code = unpackCode(t, K, j);
            const float v = decodeCode(code, codebook);
            const uint32_t rmIdx = c_perm[j];// r*16 + c
            const uint32_t r = rmIdx >> 4;
            const uint32_t cc = rmIdx & 15u;
            recon[(static_cast<size_t>(ki) * 16 + r) * n + ni * 16 + cc] = v;
        }

        __global__ void decodeTilesFrac(const uint16_t *__restrict__ packed, uint32_t KA,
                                         uint32_t tilesN, uint32_t tileU16,
                                         float *__restrict__ recon, uint32_t n) {
            const uint32_t tile = blockIdx.x;
            const uint32_t ki = tile / tilesN;
            const uint32_t ni = tile % tilesN;
            const uint16_t *t = packed + static_cast<size_t>(tile) * tileU16;
            const uint32_t j = threadIdx.x;
            const uint16_t code = unpackCodeFrac(t, KA, j);
            const float v = decodeMul1(code);
            const uint32_t rmIdx = c_perm[j];
            const uint32_t r = rmIdx >> 4;
            const uint32_t cc = rmIdx & 15u;
            recon[(static_cast<size_t>(ki) * 16 + r) * n + ni * 16 + cc] = v;
        }

        // ---- 128-wide normalized Hadamard (shared-memory FWHT) ----
        // 128 threads cooperate on one 128-vector staged in shared memory.
        //
        // Implemented as a statically-unrolled ping-pong between two shared
        // buffers (A -> B -> A ...).  Each butterfly stage reads one buffer and
        // writes the other with a single barrier in between, so every read is
        // guaranteed to precede every write without the read/write aliasing a
        // single in-place buffer would create.  The double buffer must use
        // STATIC (fully unrolled) indices: an in-place or dynamically-indexed
        // variant makes NVRTC ptxas (CUDA 12.4, sm_75/80/86) fail with
        // "Register allocation failed with register count of '7'", so the
        // alternation is written out literally.
        __device__ __forceinline__ void fwht128Smem(float *s) {
            __shared__ float A[128];
            __shared__ float B[128];
            const uint32_t i = threadIdx.x;
            A[i] = s[i];
            __syncthreads();
#define EXL3_FWHT_STAGE(src, dst, L)                          \
    do {                                                      \
        const float a = src[i];                               \
        const float b = src[i ^ (L)];                         \
        dst[i] = (i & (L)) ? (b - a) : (a + b);               \
    } while (0)
            EXL3_FWHT_STAGE(A, B, 1u);
            __syncthreads();
            EXL3_FWHT_STAGE(B, A, 2u);
            __syncthreads();
            EXL3_FWHT_STAGE(A, B, 4u);
            __syncthreads();
            EXL3_FWHT_STAGE(B, A, 8u);
            __syncthreads();
            EXL3_FWHT_STAGE(A, B, 16u);
            __syncthreads();
            EXL3_FWHT_STAGE(B, A, 32u);
            __syncthreads();
            EXL3_FWHT_STAGE(A, B, 64u);
            __syncthreads();
            s[i] = B[i];
#undef EXL3_FWHT_STAGE
        }

        // Left Hadamard over k (128-wide row bands), per column, times suh.
        // Grid: one block per k-band.  128 threads.
        __global__ void leftH128(const float *__restrict__ suh, float *__restrict__ recon,
                                  uint32_t k, uint32_t n) {
            const uint32_t kb = blockIdx.x * 128u;
            if (kb >= k) return;
            __shared__ float s[128];
            const float scale = rsqrtf(128.0f);
            for (uint32_t c = 0; c < n; ++c) {
                s[threadIdx.x] = recon[static_cast<size_t>(kb + threadIdx.x) * n + c];
                __syncthreads();
                fwht128Smem(s);
                const float sign = suh ? suh[kb + threadIdx.x] : 1.0f;
                recon[static_cast<size_t>(kb + threadIdx.x) * n + c] =
                        s[threadIdx.x] * scale * sign;
                __syncthreads();
            }
        }

        // Right Hadamard over n (128-wide column bands), per row, times svh.
        // Grid: one block per row.  128 threads.
        __global__ void rightH128(const float *__restrict__ svh, float *__restrict__ recon,
                                   uint32_t k, uint32_t n) {
            const uint32_t r = blockIdx.x;
            if (r >= k) return;
            __shared__ float s[128];
            const float scale = rsqrtf(128.0f);
            float *row = recon + static_cast<size_t>(r) * n;
            for (uint32_t nb = 0; nb < n; nb += 128u) {
                s[threadIdx.x] = row[nb + threadIdx.x];
                __syncthreads();
                fwht128Smem(s);
                const float sign = svh ? svh[nb + threadIdx.x] : 1.0f;
                row[nb + threadIdx.x] = s[threadIdx.x] * scale * sign;
                __syncthreads();
            }
        }

        // [k][n] row-major -> [n][k] row-major transpose (tiled, canonical form).
        // Block (bx, by) reads in-tile rows bx*32.., cols by*32.. and writes the
        // transposed tile; out[n][k] holds out[col][row] = in[row][col].
        // Grid must be dim3(ceil(k/32), ceil(n/32)).
        __global__ void transposeKN(const float *__restrict__ in, float *__restrict__ out,
                                     uint32_t k, uint32_t n) {
            __shared__ float t[32][33];
            const uint32_t inRow = blockIdx.x * 32u + threadIdx.y;
            const uint32_t inCol = blockIdx.y * 32u + threadIdx.x;
            if (inRow < k && inCol < n) {
                t[threadIdx.y][threadIdx.x] = in[static_cast<size_t>(inRow) * n + inCol];
            }
            __syncthreads();
            const uint32_t outRow = blockIdx.y * 32u + threadIdx.x;// = in col
            const uint32_t outCol = blockIdx.x * 32u + threadIdx.y;// = in row
            if (outRow < n && outCol < k) {
                out[static_cast<size_t>(outRow) * k + outCol] = t[threadIdx.y][threadIdx.x];
            }
        }

        // y[o] = dot(x, WT[o][0..k)) ; WT is [n][k] row-major.
        __global__ void gemvF32(const float *__restrict__ x, const float *__restrict__ wt,
                                 float *__restrict__ out, uint32_t n, uint32_t k) {
            const uint32_t o = blockIdx.x;
            if (o >= n) return;
            const float *wRow = wt + static_cast<size_t>(o) * k;
            float acc = 0.0f;
            for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
                acc = fmaf(x[i], wRow[i], acc);
            }
            for (uint32_t off = 16; off; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }
            __shared__ float srow[8];
            if ((threadIdx.x & 31u) == 0u) srow[threadIdx.x >> 5] = acc;
            __syncthreads();
            if (threadIdx.x == 0) {
                float s = 0.0f;
                for (uint32_t w = 0; w < blockDim.x / 32u; ++w) s += srow[w];
                out[o] = s;
            }
        }

        // Expand a packed ±1 bitfield into floats: set bit -> -1, clear -> +1;
        // bit i of word (e/16) maps to element e (mirrors expandPackedSigns).
        __global__ void expandSigns(const uint16_t *__restrict__ packed,
                                     float *__restrict__ out, uint32_t count) {
            const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
            if (e >= count) return;
            const uint16_t bits = packed[e >> 4];
            const uint32_t bit = e & 15u;
            out[e] = ((bits >> bit) & 1u) ? -1.0f : 1.0f;
        }

        // ---- Module state ----
        struct Linear {
            Exl3DeviceLinear d;
            float *wT = nullptr;// transposed [n][k]
        };
        std::unordered_map<int64_t, Linear> g_linears;
        std::mutex g_mutex;
        int64_t g_nextHandle = 1;

        template <typename T>
        bool deviceAlloc(T **p, size_t bytes, std::string &err) {
            if (bytes == 0) {
                *p = nullptr;
                return true;
            }
            cudaError_t e = cudaMalloc(reinterpret_cast<void **>(p), bytes);
            if (e != cudaSuccess) {
                err = std::string("cudaMalloc: ") + cudaGetErrorString(e);
                *p = nullptr;
                return false;
            }
            return true;
        }

        bool ensureReconstructed(Linear &lin, std::string &err);

    }// namespace

    bool available() { return initOnce(); }

    int64_t uploadLinear(const Exl3HostLinear &lin, std::string &err) {
        if (!initOnce()) {
            err = g_initErr;
            return -1;
        }
        if (lin.k == 0 || lin.n == 0 || (lin.k % 128u) != 0 || (lin.n % 128u) != 0) {
            err = "EXL3 linear dims must be nonzero multiples of 128";
            return -1;
        }
        const uint32_t K10 = lin.bitrate10;
        const bool frac = (K10 % 10u) != 0u;
        const uint32_t K = K10 / 10u;
        if (!frac && (K < 1u || K > 8u)) {
            err = "EXL3 bitrate must be in [1.0, 8.0]";
            return -1;
        }
        if (frac && K < 1u) {
            err = "EXL3 fractional bitrate below 1.5";
            return -1;
        }
        const uint32_t tilesK = lin.k / 16u;
        const uint32_t tilesN = lin.n / 16u;
        const uint32_t tileU16 = frac ? (16u * K + 8u) : (16u * K);
        const size_t packedBytes = static_cast<size_t>(tilesK) * tilesN * tileU16 *
                                   sizeof(uint16_t);

        Linear lin2{};
        lin2.d.k = lin.k;
        lin2.d.n = lin.n;
        lin2.d.tileU16 = tileU16;
        lin2.d.bitrate10 = K10;
        lin2.d.codebook = lin.codebook;
        lin2.d.frac = frac;

        if (!deviceAlloc(&lin2.d.packed, packedBytes, err)) return -1;
        cudaMemcpy(lin2.d.packed, lin.packed, packedBytes, cudaMemcpyHostToDevice);

        auto uploadSigns = [&](const float *f, const uint16_t *pk, uint32_t count,
                               float **dst) -> bool {
            if (f) {
                if (!deviceAlloc(dst, static_cast<size_t>(count) * sizeof(float), err))
                    return false;
                cudaMemcpy(*dst, f, static_cast<size_t>(count) * sizeof(float),
                           cudaMemcpyHostToDevice);
                return true;
            }
            if (pk) {
                if (!deviceAlloc(dst, static_cast<size_t>(count) * sizeof(float), err))
                    return false;
                const uint32_t words = (count + 15u) / 16u;
                uint16_t *tmp = nullptr;
                if (!deviceAlloc(&tmp, static_cast<size_t>(words) * sizeof(uint16_t), err)) {
                    return false;
                }
                cudaMemcpy(tmp, pk, static_cast<size_t>(words) * sizeof(uint16_t),
                           cudaMemcpyHostToDevice);
                const uint32_t threads = 256;
                const uint32_t blocks = (count + threads - 1u) / threads;
                expandSigns<<<blocks, threads, 0, stream()>>>(tmp, *dst, count);
                cudaStreamSynchronize(stream());
                cudaFree(tmp);
                return true;
            }
            *dst = nullptr;
            return true;
        };
        if (!uploadSigns(lin.suh, lin.suhPacked, lin.k, &lin2.d.suh)) return -1;
        if (!uploadSigns(lin.svh, lin.svhPacked, lin.n, &lin2.d.svh)) return -1;

        if (!deviceAlloc(&lin2.d.scratch,
                         static_cast<size_t>(lin.k) * lin.n * sizeof(float), err))
            return -1;
        if (!deviceAlloc(&lin2.wT, static_cast<size_t>(lin.n) * lin.k * sizeof(float), err))
            return -1;

        std::lock_guard<std::mutex> lk(g_mutex);
        const int64_t h = g_nextHandle++;
        g_linears.emplace(h, lin2);
        return h;
    }

    void releaseLinear(int64_t handle) {
        std::lock_guard<std::mutex> lk(g_mutex);
        auto it = g_linears.find(handle);
        if (it == g_linears.end()) return;
        Linear &lin = it->second;
        if (stream()) cudaStreamSynchronize(stream());
        cudaFree(lin.d.packed);
        cudaFree(lin.d.suh);
        cudaFree(lin.d.svh);
        cudaFree(lin.d.scratch);
        cudaFree(lin.wT);
        g_linears.erase(it);
    }

    namespace {
        bool ensureReconstructed(Linear &lin, std::string &err) {
            if (lin.d.reconstructed) return true;
            const uint32_t k = lin.d.k, n = lin.d.n;
            const uint32_t tilesK = k / 16u;
            const uint32_t tilesN = n / 16u;
            if (lin.d.frac) {
                decodeTilesFrac<<<tilesK * tilesN, 256, 0, stream()>>>(
                        static_cast<const uint16_t *>(lin.d.packed), lin.d.bitrate10 / 10u,
                        tilesN, lin.d.tileU16, lin.d.scratch, n);
            } else {
                decodeTilesInt<<<tilesK * tilesN, 256, 0, stream()>>>(
                        static_cast<const uint16_t *>(lin.d.packed), lin.d.bitrate10 / 10u,
                        lin.d.codebook, tilesN, lin.d.tileU16, lin.d.scratch, n);
            }
            leftH128<<<tilesK, 128, 0, stream()>>>(lin.d.suh, lin.d.scratch, k, n);
            rightH128<<<k, 128, 0, stream()>>>(lin.d.svh, lin.d.scratch, k, n);
            dim3 tblock(32, 32);
            // Grid.x covers the k (in-row) axis, grid.y the n (in-col) axis,
            // matching the swapped-role indexing inside transposeKN.
            dim3 tgrid((k + 31u) / 32u, (n + 31u) / 32u);
            transposeKN<<<tgrid, tblock, 0, stream()>>>(lin.d.scratch, lin.wT, k, n);
            cudaError_t e = cudaStreamSynchronize(stream());
            if (e != cudaSuccess) {
                err = std::string("EXL3 reconstruct: ") + cudaGetErrorString(e);
                return false;
            }
            lin.d.reconstructed = true;
            return true;
        }
    }// namespace

    bool reconstruct(int64_t handle, float *outHost, std::string &err) {
        if (!initOnce()) {
            err = g_initErr;
            return false;
        }
        std::lock_guard<std::mutex> lk(g_mutex);
        auto it = g_linears.find(handle);
        if (it == g_linears.end()) {
            err = "invalid EXL3 handle";
            return false;
        }
        Linear &lin = it->second;
        if (!ensureReconstructed(lin, err)) return false;
        const size_t bytes = static_cast<size_t>(lin.d.k) * lin.d.n * sizeof(float);
        cudaError_t e = cudaMemcpy(outHost, lin.d.scratch, bytes, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            err = std::string("EXL3 D2H: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    bool gemv(int64_t handle, const float *xHost, float *outHost, std::string &err) {
        if (!initOnce()) {
            err = g_initErr;
            return false;
        }
        std::lock_guard<std::mutex> lk(g_mutex);
        auto it = g_linears.find(handle);
        if (it == g_linears.end()) {
            err = "invalid EXL3 handle";
            return false;
        }
        Linear &lin = it->second;
        if (!ensureReconstructed(lin, err)) return false;
        const uint32_t k = lin.d.k, n = lin.d.n;

        float *dx = nullptr, *dy = nullptr;
        if (!deviceAlloc(&dx, static_cast<size_t>(k) * sizeof(float), err)) return false;
        if (!deviceAlloc(&dy, static_cast<size_t>(n) * sizeof(float), err)) {
            cudaFree(dx);
            return false;
        }
        bool ok = true;
        cudaError_t e = cudaMemcpy(dx, xHost, static_cast<size_t>(k) * sizeof(float),
                                   cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            err = std::string("EXL3 x H2D: ") + cudaGetErrorString(e);
            ok = false;
        } else {
            gemvF32<<<n, 256, 0, stream()>>>(dx, lin.wT, dy, n, k);
            // stream() is cudaStreamNonBlocking, so the synchronous D2H
            // cudaMemcpy (default stream) does NOT wait for the kernel; sync
            // explicitly or the copy races the GEMV and reads stale dy.
            e = cudaStreamSynchronize(stream());
            if (e != cudaSuccess) {
                err = std::string("EXL3 gemv sync: ") + cudaGetErrorString(e);
                ok = false;
            } else {
                e = cudaMemcpy(outHost, dy, static_cast<size_t>(n) * sizeof(float),
                               cudaMemcpyDeviceToHost);
                if (e != cudaSuccess) {
                    err = std::string("EXL3 y D2H: ") + cudaGetErrorString(e);
                    ok = false;
                }
            }
        }
        cudaFree(dx);
        cudaFree(dy);
        return ok;
    }

    // =======================================================================
    // Fused native GEMV (see include/EXL3Kernels.hpp)
    // =======================================================================

    namespace {

        struct FusedRegistry {
            std::mutex mx;
            std::unordered_map<const void *, Exl3FusedLinear> map;
        };
        FusedRegistry &fusedRegistry() {
            static FusedRegistry r;
            return r;
        }

        // Decode one code out of the warp-staged 32-bit words.
        template <bool FRAC>
        __device__ __forceinline__ uint32_t codeAt(const uint32_t *words, uint32_t nWords,
                                                   uint32_t j, uint32_t K) {
            uint32_t pos, width;
            if constexpr (FRAC) {
                const uint32_t even = (j + 1u) / 2u;
                const uint32_t odd = j / 2u;
                pos = even * (K + 1u) + odd * K;
                width = (j & 1u) ? K : (K + 1u);
            } else {
                pos = j * K;
                width = K;
            }
            const uint32_t word = pos >> 5;
            const uint32_t sh = pos & 31u;
            // Runtime modulo compiles to an integer-division CALL here (one per
            // code!), which dominated the hot loop.  The wrap can only happen
            // at the last word, so a select is equivalent and branch-free.
            const uint32_t next = (word + 1u == nWords) ? 0u : (word + 1u);
            const uint32_t lo = words[word];
            const uint32_t hi = words[next];
            const uint32_t raw = __funnelshift_r(lo, hi, sh);
            return (width >= 32u) ? raw : (raw & ((1u << width) - 1u));
        }

        // Each warp owns TWO output tiles (32 columns) and each lane owns one
        // output column.  The 16 weights of a column are decoded into registers
        // and accumulated immediately: no shared decoded-tile round trip, no
        // intra-tile barriers, no half-warp divergence.  SPLIT warps split k
        // and their partial sums are reduced through sRed.
        //
        // Codes are read from the upload-permuted trellis (row-major order), so
        // a column's weight at row r is simply code index t = r*16 + c.
        __device__ __forceinline__ uint32_t exl3Fshift(uint32_t b, uint32_t a, int sh) {
            const uint64_t m = (static_cast<uint64_t>(a) << 32) | b;
            return static_cast<uint32_t>(m >> sh);
        }
        __device__ __forceinline__ uint32_t exl3Bfe16(uint32_t x, int off) {
            return (x >> off) & 0xffffu;
        }

        // Decode 8 weights of one 16x16 EXL3 tile for one lane (CPU port of
        // the EXL3 reference dq8_aligned_{1,2,3}bit).  The packed stream is a per-lane
        // 16-bit-window stream, NOT a plain K-bit code stream.
        template <int CB>
        __device__ __forceinline__ void exl3LaneFrag(const uint32_t *p, uint32_t K,
                                                     int lane, float *f) {
            const int to = lane * 8;
            uint32_t w[8];
            if (K == 1) {
                const uint32_t i1 = static_cast<uint32_t>(to) >> 5;
                const uint32_t i0 = (i1 + 7u) & 7u;
                uint32_t b = exl3Fshift(p[i1], p[i0],
                                        static_cast<int>((static_cast<uint32_t>(~to)) & 24u));
                w[7] = b & 0xffffu;
#pragma unroll
                for (int i = 6; i >= 0; --i) w[i] = exl3Bfe16(b, 7 - i);
            } else if (K == 2) {
                const uint32_t i1 = static_cast<uint32_t>(to) >> 4;
                const uint32_t i0 = (i1 + 15u) & 15u;
                uint32_t b = exl3Fshift(p[i1], p[i0],
                                        static_cast<int>(((static_cast<uint32_t>(~to)) & 8u) << 1));
                w[7] = b & 0xffffu;
#pragma unroll
                for (int i = 6; i >= 0; --i) w[i] = exl3Bfe16(b, 14 - 2 * i);
            } else {
                const int b1 = (to + 257) * 3, b2 = b1 + 21;
                const int i0 = (b1 - 16) / 32, i2 = (b2 - 1) / 32;
                const int s2 = (i2 + 1) * 32 - b2;
                const int nw = 3 * 256 / 32;
                const uint32_t a = p[i0 % nw], b = p[i2 % nw];
                w[7] = exl3Fshift(b, a, s2);
                w[6] = w[7] >> 3;
                w[5] = w[6] >> 3;
                w[4] = w[5] >> 3;
                w[3] = exl3Fshift(b, a, s2 + 12);
                w[2] = w[3] >> 3;
                w[1] = w[2] >> 3;
                w[0] = w[1] >> 3;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) f[i] = decodeCode(w[i] & 0xffffu, CB);
        }

        template <int CB, int SPLIT>
        __global__ void exl3FusedMatVec(const uint16_t *__restrict__ trellis,
                                         const float *__restrict__ suh,
                                         uint32_t k, uint32_t K,
                                         uint32_t tilesN, uint32_t tileU16,
                                         const float *__restrict__ x,
                                         float *__restrict__ z) {
            constexpr int TPB = 8 / SPLIT;
            const int warp = threadIdx.x >> 5;
            const int lane = threadIdx.x & 31;
            const int group = warp / SPLIT;
            const int wi = warp % SPLIT;
            const uint32_t pair = blockIdx.x * TPB + group;
            const uint32_t c = static_cast<uint32_t>(lane) & 15u;
            // One warp-group handles ONE 16-column output tile; all its lanes
            // decode that tile cooperatively (SPLIT warps split k).
            const uint32_t ni = pair;

            const uint32_t kChunk = k / SPLIT;
            const uint32_t kStart = wi * kChunk;
            const uint32_t kEnd = kStart + kChunk;
            const uint32_t nWords = 8u * K;
            const uint32_t mask = (K >= 32u) ? 0xFFFFFFFFu : ((1u << K) - 1u);
            // Column-major bit window for this lane's output column c: all 16
            // row codes are contiguous at bit c*16*K.
            const uint32_t startBit = c * 16u * K;
            const uint32_t sw = startBit >> 5;
            const uint32_t sh0 = startBit & 31u;

            __shared__ float sX[8][128];   // input Hadamard (per warp)
            __shared__ float sDec[8][256]; // decoded 16x16 tile (per warp)
            __shared__ float sRed[8][32];  // per-warp partial outputs

            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

            for (uint32_t kb = kStart; kb < kEnd; kb += 128u) {
                // 128-point FWHT with four values per lane and shuffle-xor
                // butterflies.  The previous shared-memory version needed one
                // __syncwarp per stage (7 per block); on sm_75 nvcc lowers each
                // __syncwarp to an out-of-line CALL, which dominated this loop.
                float v0, v1, v2, v3;
                {
                    const uint32_t i0 = static_cast<uint32_t>(lane) * 4u;
                    v0 = x[kb + i0];
                    v1 = x[kb + i0 + 1u];
                    v2 = x[kb + i0 + 2u];
                    v3 = x[kb + i0 + 3u];
                    // xh = H128 (suh ⊙ x): W = Dsu Hk What Hn Dsv, so
                    // W^T x = Dsv Hn What^T Hk (Dsu x).
                    if (suh) {
                        v0 *= suh[kb + i0];
                        v1 *= suh[kb + i0 + 1u];
                        v2 *= suh[kb + i0 + 2u];
                        v3 *= suh[kb + i0 + 3u];
                    }
                }
                // bits 0 and 1 of the element index live inside the lane
                {
                    const float a = v0, b = v1;
                    v0 = a + b;
                    v1 = a - b;
                    const float c = v2, d = v3;
                    v2 = c + d;
                    v3 = c - d;
                }
                {
                    const float a = v0, b = v2;
                    v0 = a + b;
                    v2 = a - b;
                    const float c = v1, d = v3;
                    v1 = c + d;
                    v3 = c - d;
                }
                // bits 2..6: butterfly with the lane whose id differs in bit s
#pragma unroll
                for (int st = 0; st < 5; ++st) {
                    const int bit = 1 << st;
                    float o0 = __shfl_xor_sync(0xffffffffu, v0, bit);
                    float o1 = __shfl_xor_sync(0xffffffffu, v1, bit);
                    float o2 = __shfl_xor_sync(0xffffffffu, v2, bit);
                    float o3 = __shfl_xor_sync(0xffffffffu, v3, bit);
                    if (lane & bit) {
                        v0 = o0 - v0;
                        v1 = o1 - v1;
                        v2 = o2 - v2;
                        v3 = o3 - v3;
                    } else {
                        v0 = v0 + o0;
                        v1 = v1 + o1;
                        v2 = v2 + o2;
                        v3 = v3 + o3;
                    }
                }
                constexpr float kScale = 0.08838834764831845f;// 1/sqrt(128)
                // Publish the transformed block for the column dot products.
                const uint32_t i0 = static_cast<uint32_t>(lane) * 4u;
                sX[warp][i0] = v0 * kScale;
                sX[warp][i0 + 1u] = v1 * kScale;
                sX[warp][i0 + 2u] = v2 * kScale;
                sX[warp][i0 + 3u] = v3 * kScale;
                __syncwarp();

                const uint32_t kt0 = kb >> 4;
                for (uint32_t kt = kt0; kt < kt0 + 8u; ++kt) {
                    const uint16_t *tp = trellis +
                            (static_cast<size_t>(kt) * tilesN + ni) * tileU16;
                    const uint32_t *p32 = reinterpret_cast<const uint32_t *>(tp);
                    if ((lane & 4) == 0) {
                        float f[8], nb[8];
                        exl3LaneFrag<CB>(p32, K, lane, f);
                        exl3LaneFrag<CB>(p32, K, lane + 4, nb);
                        const int r0 = (lane % 4) * 2, r1 = r0 + 1;
                        const int r2 = r0 + 8, r3 = r0 + 9;
                        const int c0 = lane / 8, c1 = c0 + 4;
                        auto put = [&](int r, int cc, float a, float b) {
                            sDec[warp][r * 16 + cc] = a;
                            sDec[warp][r * 16 + cc + 1] = b;
                        };
                        put(r0, c0 * 2, f[0], nb[0]);
                        put(r1, c0 * 2, f[1], nb[1]);
                        put(r2, c0 * 2, f[2], nb[2]);
                        put(r3, c0 * 2, f[3], nb[3]);
                        put(r0, c1 * 2, f[4], nb[4]);
                        put(r1, c1 * 2, f[5], nb[5]);
                        put(r2, c1 * 2, f[6], nb[6]);
                        put(r3, c1 * 2, f[7], nb[7]);
                    }
                    __syncwarp();
                    if (lane < 16) {
                        const float *xs = sX[warp] + ((kt << 4) - kb);
#pragma unroll
                        for (int r = 0; r < 16; ++r) {
                            const float wv = sDec[warp][r * 16 + lane];
                            const float xv = xs[r];
                            if ((r & 3) == 0) {
                                acc0 = fmaf(wv, xv, acc0);
                            } else if ((r & 3) == 1) {
                                acc1 = fmaf(wv, xv, acc1);
                            } else if ((r & 3) == 2) {
                                acc2 = fmaf(wv, xv, acc2);
                            } else {
                                acc3 = fmaf(wv, xv, acc3);
                            }
                        }
                    }
                    __syncwarp();
                }
            }
            sRed[warp][lane] = acc0 + acc1 + acc2 + acc3;
            __syncthreads();
            if (wi == 0 && lane < 16) {
                float s = sRed[group * SPLIT][lane];
#pragma unroll
                for (int q = 1; q < SPLIT; ++q) {
                    s += sRed[group * SPLIT + q][lane];
                }
                z[pair * 16u + static_cast<uint32_t>(lane)] = s;
            }
        }

        // y = svh ⊙ H128(z) for each 128-wide output block.
        __global__ void exl3FusedOutTransform(const float *__restrict__ z,
                                               const float *__restrict__ svh,
                                               float *__restrict__ y) {
            __shared__ float s[128];
            const uint32_t base = blockIdx.x * 128u;
            const int t = threadIdx.x;
            s[t] = z[base + static_cast<uint32_t>(t)];
            __syncthreads();
            for (uint32_t len = 1, lg = 0; len < 128u; len <<= 1, ++lg) {
                if (t < 64) {
                    const int grp = t >> lg;
                    const int off = t & static_cast<int>(len - 1u);
                    const int a = (grp << (lg + 1)) + off;
                    const int bb = a + static_cast<int>(len);
                    const float u = s[a];
                    const float v = s[bb];
                    s[a] = u + v;
                    s[bb] = u - v;
                }
                __syncthreads();
            }
            constexpr float kScale = 0.08838834764831845f;// 1/sqrt(128)
            float v = s[t] * kScale;
            if (svh) v *= svh[base + static_cast<uint32_t>(t)];
            y[base + static_cast<uint32_t>(t)] = v;
        }

        template <int CB>
        void launchFusedSplit(int split, const Exl3FusedLinear &lin,
                              const float *xDev, float *yDev,
                              cudaStream_t stream) {
            const uint32_t tilesN = lin.n / 16u;
            const uint32_t tileU16 = lin.tileU16;
            const uint32_t gridTiles = lin.n / 16u;// tiles total
#define LAUNCH_SPLIT(SP)                                                         \
    do {                                                                         \
        const uint32_t grid = gridTiles / (8u / (SP));                           \
        exl3FusedMatVec<CB, SP><<<grid, 256, 0, stream>>>(                      \
                reinterpret_cast<const uint16_t *>(lin.trellis), lin.suh, lin.k, \
                lin.K, tilesN, tileU16, xDev, yDev);                             \
    } while (0)
            switch (split) {
                case 8: LAUNCH_SPLIT(8); break;
                case 4: LAUNCH_SPLIT(4); break;
                case 2: LAUNCH_SPLIT(2); break;
                default: LAUNCH_SPLIT(1); break;
            }
#undef LAUNCH_SPLIT
        }

    }// namespace

    bool uploadFusedLinear(const Exl3HostLinear &lin, Exl3FusedLinear &out,
                           std::string &err) {
        if (!initOnce()) {
            err = g_initErr;
            return false;
        }
        if (lin.packed == nullptr || lin.k == 0 || lin.n == 0) {
            err = "EXL3 fused upload: empty linear";
            return false;
        }
        if ((lin.k & 127u) != 0 || (lin.n & 127u) != 0) {
            err = "EXL3 fused upload: k/n must be 128-divisible";
            return false;
        }
        const bool frac = (lin.bitrate10 % 10u) != 0u;
        if (frac) {
            err = "EXL3 fused upload: half-integer bitrate unsupported";
            return false;
        }
        const uint32_t K = lin.bitrate10 / 10u;
        const uint32_t tileU16 = 16u * K;
        const uint64_t trellisBytes =
                static_cast<uint64_t>(lin.k / 16u) * (lin.n / 16u) * tileU16 *
                sizeof(uint16_t);
        void *dTrellis = nullptr;
        float *dSuh = nullptr, *dSvh = nullptr;
        if (cudaMalloc(&dTrellis, trellisBytes) != cudaSuccess) {
            err = "EXL3 fused upload: cudaMalloc(trellis) failed";
            return false;
        }
        if (cudaMemcpy(dTrellis, lin.packed, trellisBytes, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            cudaFree(dTrellis);
            err = "EXL3 fused upload: trellis H2D failed";
            return false;
        }
        auto uploadSigns = [&](const float *src, uint32_t count,
                               float **dst) -> bool {
            if (src == nullptr || count == 0) {
                *dst = nullptr;
                return true;
            }
            if (cudaMalloc(dst, static_cast<size_t>(count) * sizeof(float)) !=
                cudaSuccess) {
                return false;
            }
            if (cudaMemcpy(*dst, src, static_cast<size_t>(count) * sizeof(float),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaFree(*dst);
                *dst = nullptr;
                return false;
            }
            return true;
        };
        if (!uploadSigns(lin.suh, lin.k, &dSuh) ||
            !uploadSigns(lin.svh, lin.n, &dSvh)) {
            cudaFree(dTrellis);
            if (dSuh) cudaFree(dSuh);
            if (dSvh) cudaFree(dSvh);
            err = "EXL3 fused upload: sign H2D failed";
            return false;
        }
        out = Exl3FusedLinear{};
        out.trellis = dTrellis;
        out.suh = dSuh;
        out.svh = dSvh;
        out.k = lin.k;
        out.n = lin.n;
        out.K = K;
        out.codebook = lin.codebook;
        out.frac = false;
        out.tileU16 = tileU16;
        registerFusedLinear(out);
        return true;
    }

    void releaseFusedLinear(Exl3FusedLinear &lin) {
        if (lin.trellis != nullptr) {
            unregisterFusedLinear(lin.trellis);
            cudaFree(const_cast<void *>(lin.trellis));
        }
        if (lin.suh != nullptr) cudaFree(const_cast<float *>(lin.suh));
        if (lin.svh != nullptr) cudaFree(const_cast<float *>(lin.svh));
        lin = Exl3FusedLinear{};
    }

    void registerFusedLinear(const Exl3FusedLinear &lin) {
        if (lin.trellis == nullptr || lin.k == 0 || lin.n == 0) return;
        std::lock_guard<std::mutex> lk(fusedRegistry().mx);
        fusedRegistry().map[lin.trellis] = lin;
    }

    void unregisterFusedLinear(const void *trellisDev) {
        std::lock_guard<std::mutex> lk(fusedRegistry().mx);
        fusedRegistry().map.erase(trellisDev);
    }

    void clearFusedLinears() {
        std::lock_guard<std::mutex> lk(fusedRegistry().mx);
        fusedRegistry().map.clear();
    }

    bool gemvFused(const void *trellisDev, const float *xDev, float *yDev,
                   cudaStream_t stream, std::string &err) {
        Exl3FusedLinear lin;
        {
            std::lock_guard<std::mutex> lk(fusedRegistry().mx);
            auto it = fusedRegistry().map.find(trellisDev);
            if (it == fusedRegistry().map.end()) {
                err = "EXL3 fused: unregistered linear";
                return false;
            }
            lin = it->second;
        }
        if (lin.frac) {
            err = "EXL3 fused: half-integer bitrate not supported";
            return false;
        }
        if ((lin.k & 127u) != 0 || (lin.n & 127u) != 0) {
            err = "EXL3 fused: k/n must be 128-divisible";
            return false;
        }
        int split = 8;
        while (split > 1 && (lin.k % (128u * static_cast<uint32_t>(split))) != 0) {
            split >>= 1;
        }
        if (lin.codebook == 2) {
            launchFusedSplit<2>(split, lin, xDev, yDev, stream);
        } else if (lin.codebook == 1) {
            launchFusedSplit<1>(split, lin, xDev, yDev, stream);
        } else {
            launchFusedSplit<0>(split, lin, xDev, yDev, stream);
        }
        exl3FusedOutTransform<<<lin.n / 128u, 128, 0, stream>>>(
                yDev, lin.svh, yDev);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            err = std::string("EXL3 fused launch: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    void shutdown() {
        clearFusedLinears();
        std::vector<int64_t> handles;
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            for (auto &kv : g_linears) handles.push_back(kv.first);
        }
        for (int64_t h : handles) releaseLinear(h);
        if (stream()) {
            cudaStreamDestroy(stream());
            stream() = nullptr;
        }
    }

}// namespace tinycoder::exl3gpu

#endif// USE_CUDA
