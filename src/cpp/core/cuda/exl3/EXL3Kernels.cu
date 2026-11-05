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
#include <mma.h>
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

        // ---- Integer-K lane shuffle (port of EXL3Codec::decodeTileNative) ----
        // For K = 1..3 the CPU decodes 16-bit windows per lane and writes each
        // (f[i], n[i]) pair into ADJACENT columns, which is not the layout
        // c_perm encodes (columns 8 apart).  Ported verbatim so the GPU
        // reconstruct/materialise reproduce the CPU reference matrix.
        __device__ __forceinline__ uint32_t exl3NatFshift(uint32_t b, uint32_t a, int sh) {
            const uint64_t m = (static_cast<uint64_t>(a) << 32) | b;
            return static_cast<uint32_t>(m >> sh);
        }
        __device__ __forceinline__ uint32_t exl3NatBfe16(uint32_t x, int off) {
            return (x >> off) & 0xffffu;
        }

        __device__ __forceinline__ void exl3LaneFragNative(
                const uint32_t *__restrict__ p, uint32_t K, int codebook, int lane,
                float *f) {
            const int to = lane * 8;
            uint32_t w[8];
            if (K == 1) {
                const uint32_t i1 = static_cast<uint32_t>(to) >> 5;
                const uint32_t i0 = (i1 + 7u) & 7u;
                const uint32_t b = exl3NatFshift(p[i1], p[i0],
                                              static_cast<int>(static_cast<uint32_t>(~to) & 24u));
                w[7] = b & 0xffffu;
                for (int i = 6; i >= 0; --i) w[i] = exl3NatBfe16(b, 7 - i);
            } else if (K == 2) {
                const uint32_t i1 = static_cast<uint32_t>(to) >> 4;
                const uint32_t i0 = (i1 + 15u) & 15u;
                const uint32_t b = exl3NatFshift(p[i1], p[i0],
                                              static_cast<int>((static_cast<uint32_t>(~to) & 8u) << 1));
                w[7] = b & 0xffffu;
                for (int i = 6; i >= 0; --i) w[i] = exl3NatBfe16(b, 14 - 2 * i);
            } else {
                const int b1 = (to + 257) * 3;
                const int b2 = b1 + 21;
                const int i0 = (b1 - 16) / 32;
                const int i2 = (b2 - 1) / 32;
                const int s2 = (i2 + 1) * 32 - b2;
                const int nw = 3 * 256 / 32;
                const uint32_t a = p[i0 % nw];
                const uint32_t b = p[i2 % nw];
                w[7] = exl3NatFshift(b, a, s2);
                w[6] = w[7] >> 3;
                w[5] = w[6] >> 3;
                w[4] = w[5] >> 3;
                w[3] = exl3NatFshift(b, a, s2 + 12);
                w[2] = w[3] >> 3;
                w[1] = w[2] >> 3;
                w[0] = w[1] >> 3;
            }
            for (int i = 0; i < 8; ++i) {
                f[i] = decodeCode(static_cast<uint16_t>(w[i] & 0xffffu), codebook);
            }
        }

        // Half-output twin for the RAW path: writes the decoded weight straight to
        // fp16, so materialise is read-trellis + write-fp16 with no 10.6 GB F32
        // staging write/read pair (profile: 54% of the batch path before this).
        // Eight tiles per 256-thread block (one warp each).  A single warp per block
        // left the SM under-occupied and the decode latency-bound (it was 32% of
        // prefill); this keeps eight warps resident with independent work.
        // Each warp walks TILES_PER_WARP independent tiles with the loop unrolled, so
        // the trellis loads of all of them are in flight at once.  The decode was
        // bound by memory-level parallelism, not occupancy: one 32-byte sector per
        // tile left too few outstanding requests to hide DRAM latency (more warps and
        // more active lanes both measured neutral).
        template <int TILES_PER_WARP>
        __global__ void decodeTilesIntNativeH(const uint16_t *__restrict__ packed, uint32_t K,
                                              int codebook, uint32_t tilesN, uint32_t tileU16,
                                              __half *__restrict__ recon, uint32_t n,
                                              uint32_t nTiles) {
            const uint32_t warp = threadIdx.x >> 5;
            const uint32_t tile0 = (blockIdx.x * (blockDim.x >> 5) + warp) * TILES_PER_WARP;
#pragma unroll
          for (int tt = 0; tt < TILES_PER_WARP; ++tt) {
            const uint32_t tile = tile0 + tt;
            if (tile >= nTiles) return;
            const uint32_t ki = tile / tilesN;
            const uint32_t ni = tile % tilesN;
            const uint32_t *p = reinterpret_cast<const uint32_t *>(
                    packed + static_cast<size_t>(tile) * tileU16);
            const int lane = static_cast<int>(threadIdx.x) & 31;
            if ((lane & 4) != 0) return;
            float f[8];
            float nn[8];
            exl3LaneFragNative(p, K, codebook, lane, f);
            exl3LaneFragNative(p, K, codebook, lane + 4, nn);
            const int r0 = (lane % 4) * 2;
            const int c0 = lane / 8;
            const int c1 = c0 + 4;
            const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
            for (int i = 0; i < 4; ++i) {
                __half *row = recon + static_cast<size_t>(ki * 16u + rows[i]) * n + ni * 16u;
                // 4-byte stores: the two adjacent columns are one __half2, which
                // halves the store transactions (2-byte scalar stores were ~39 GB/s).
                reinterpret_cast<__half2 *>(row)[c0] =
                        __floats2half2_rn(f[i], nn[i]);
                reinterpret_cast<__half2 *>(row)[c1] =
                        __floats2half2_rn(f[4 + i], nn[4 + i]);
            }
          }
        }

        // One warp per tile: lane 0..31, lanes with bit 2 set are the "n" half of
        // lane&~4 and are loaded by their partner's loop iteration.
        __global__ void decodeTilesIntNative(const uint16_t *__restrict__ packed, uint32_t K,
                                             int codebook, uint32_t tilesN, uint32_t tileU16,
                                             float *__restrict__ recon, uint32_t n) {
            const uint32_t tile = blockIdx.x;
            const uint32_t ki = tile / tilesN;
            const uint32_t ni = tile % tilesN;
            const uint32_t *p = reinterpret_cast<const uint32_t *>(
                    packed + static_cast<size_t>(tile) * tileU16);
            const int lane = static_cast<int>(threadIdx.x);
            if (lane >= 32 || (lane & 4) != 0) return;
            float f[8];
            float nn[8];
            exl3LaneFragNative(p, K, codebook, lane, f);
            exl3LaneFragNative(p, K, codebook, lane + 4, nn);
            const int r0 = (lane % 4) * 2;
            const int c0 = lane / 8;
            const int c1 = c0 + 4;
            // Rows are r0, r0+1, r0+8, r0+9 (the tensor-core half split), NOT
            // four consecutive rows.
            const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
            for (int i = 0; i < 4; ++i) {
                float *row = recon + static_cast<size_t>(ki * 16u + rows[i]) * n + ni * 16u;
                row[c0 * 2 + 0] = f[i];
                row[c0 * 2 + 1] = nn[i];
                row[c1 * 2 + 0] = f[4 + i];
                row[c1 * 2 + 1] = nn[4 + i];
            }
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

        // Fused decode + WMMA GEMM, register-blocked in n and software-pipelined.
        //
        // A block covers WARPS*16 rows of A and NT*16 columns of B.  v2 covered only
        // 16 columns, so every block re-read the whole A matrix: (n/16) x seq x k
        // halfs = 3.2 GB per forward at k=5120/n=10240, which was measured to be the
        // dominant cost (widening m instead of n changed nothing, ruling out decode
        // amplification).  With NT accumulators per warp one A fragment load feeds
        // NT MMAs, so A traffic drops by NT.
        //
        // Decode: NT*KC trellis tiles per k-chunk are decoded into shared memory and
        // shared by all warps in the block (one decode per tile per block).  Both
        // HALF-warps decode a different tile -- v2 fed lane 16..31 through the same
        // lane index as lane 0..15 and simply duplicated their work, so half the
        // decode throughput was being thrown away.
        //
        // Pipeline (this is what makes it competitive): with the barriers in the
        // decode-then-MMA form, every chunk paid the FULL latency of its A-fragment
        // loads and of the trellis loads before any MMA could issue -- measured as
        // 0.7 ms of pure stall per 5120x10240 matrix, more than the MMA itself.
        // Both operand sources are therefore fetched one chunk ahead:
        //   * B tiles are decoded into a double-buffered smem pair, chunk c+1 while
        //     chunk c is still being consumed;
        //   * A fragments live in registers as a two-deep pair, so the global loads
        //     issued for chunk c+1 are in flight for the whole of chunk c's MMA.
        // The two buffers also cut the barrier count to one per chunk.
        //
        // The epilogue stages each accumulator tile through smem and copies only the
        // rows below m, so a partial row strip cannot write past y (v1's bug).
        // MLOOP: how many m-strips (of WARPS warps each) one block covers.  With
        // MLOOP*WARPS*16 >= seqLen a block spans the whole activation matrix, so
        // every trellis tile is decoded exactly once per k-chunk instead of once
        // per m-block (the 4x duplication that made the fused kernel lose to the
        // materialise+cuBLAS path).  MLOOP > 1 loads A inside the MMA stage: the
        // two-deep A preload would otherwise cost MLOOP*KC fragments of registers.
        template <int WARPS, int NT, int KC, bool DECODE_FIRST, int MLOOP = 1>
        __global__ void gemmFusedRawT(const uint16_t *__restrict__ packed, uint32_t K,
                                      int codebook, uint32_t tilesN, uint32_t tileU16,
                                      const __half *__restrict__ A, float *__restrict__ C,
                                      uint32_t m, uint32_t k, uint32_t n,
                                      uint32_t dbgMode) {
            using namespace nvcuda;
            static const int kNativeLane[16] = {0, 1, 2, 3, 8, 9, 10, 11,
                                                16, 17, 18, 19, 24, 25, 26, 27};
            using AFrag = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                                         wmma::row_major>;
            constexpr int kTilesPerPass = WARPS * 2;   // two half-warps per warp
            constexpr int kDecodePasses = (NT * KC + kTilesPerPass - 1) / kTilesPerPass;
            // One shared buffer, two uses: the double-buffered B staging tiles
            // during the k-loop, then (the loop is over and a barrier has passed)
            // the per-warp fp32 epilogue tile.  Sharing them keeps the block at
            // 16 KB for the default shape, which is what allows 4 resident blocks
            // per SM (a separate epilogue array cost a whole extra residency slot).
            constexpr int kBBytes = 2 * NT * KC * 16 * 16 * 2;
            constexpr int kCBytes = WARPS * 16 * 16 * 4;
            constexpr int kSmemBytes = kBBytes > kCBytes ? kBBytes : kCBytes;
            __shared__ __align__(16) unsigned char smemRaw[kSmemBytes];
            __half *const bT = reinterpret_cast<__half *>(smemRaw);
            const uint32_t warp = threadIdx.x >> 5;
            const uint32_t lane = threadIdx.x & 31u;
            const int vl = kNativeLane[lane & 15u];
            const uint32_t hw = lane >> 4;             // 0/1: which tile this half decodes
            const uint32_t nt0 = blockIdx.x * NT;      // first n-tile of this block
            const uint32_t mBase = (blockIdx.y * WARPS * MLOOP) * 16u;
            const uint32_t m0 = mBase + warp * 16u;   // first strip (MLOOP == 1 path)
            const uint32_t kTiles = k / 16u;
            const int nChunks = static_cast<int>((kTiles + KC - 1) / KC);
            const __half *aRow = A + static_cast<size_t>(m0) * k;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MLOOP][NT];
#pragma unroll
            for (int ml = 0; ml < MLOOP; ++ml)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) wmma::fill_fragment(acc[ml][nt], 0.0f);

            // Decode one k-chunk of NT*KC tiles into bT[buf].
            auto decodeChunk = [&](int chunk, int buf) {
                if ((dbgMode & 1u) != 0u) return;
                const uint32_t kc = static_cast<uint32_t>(chunk) * KC;
#pragma unroll
                for (int p = 0; p < kDecodePasses; ++p) {
                    const int t = p * kTilesPerPass + static_cast<int>(warp) * 2 +
                                  static_cast<int>(hw);
                    const int nt = t % NT;
                    const int kk = t / NT;
                    const uint32_t kt = kc + static_cast<uint32_t>(kk);
                    const uint32_t ni = nt0 + static_cast<uint32_t>(nt);
                    if (nt < NT && kk < KC && kt < kTiles && ni < tilesN) {
                        const uint32_t tile = kt * tilesN + ni;
                        const uint32_t *pp = reinterpret_cast<const uint32_t *>(
                                packed + static_cast<size_t>(tile) * tileU16);
                        float f[8], nn[8];
                        exl3LaneFragNative(pp, K, codebook, vl, f);
                        exl3LaneFragNative(pp, K, codebook, vl + 4, nn);
                        const int r0 = (vl % 4) * 2, c0 = vl / 8, c1 = c0 + 4;
                        const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
                        __half *b = bT + (((buf * NT) + nt) * KC + kk) * 256;
                        for (int i = 0; i < 4; ++i) {
                            __half *row = b + rows[i] * 16;
                            reinterpret_cast<__half2 *>(row)[c0] =
                                    __floats2half2_rn(f[i], nn[i]);
                            reinterpret_cast<__half2 *>(row)[c1] =
                                    __floats2half2_rn(f[4 + i], nn[4 + i]);
                        }
                    }
                }
            };
            auto loadA = [&](int chunk, AFrag (&aF)[KC]) {
                if (MLOOP > 1) return;   // MLOOP>1 loads A per strip inside mmaChunk
                if (m0 >= m || (dbgMode & 4u) != 0u) return;
                const uint32_t kc = static_cast<uint32_t>(chunk) * KC;
#pragma unroll
                for (int kk = 0; kk < KC; ++kk) {
                    if (kc + static_cast<uint32_t>(kk) < kTiles)
                        wmma::load_matrix_sync(aF[kk], aRow + kc * 16u + kk * 16u, k);
                }
            };
            auto mmaChunk = [&](int chunk, AFrag (&aF)[KC], int buf) {
                if ((dbgMode & 2u) != 0u) return;
                const uint32_t kc = static_cast<uint32_t>(chunk) * KC;
#pragma unroll
                for (int ml = 0; ml < MLOOP; ++ml) {
                    const uint32_t strip = mBase + (ml * WARPS + warp) * 16u;
                    if (strip >= m) continue;
                    AFrag aLocal[KC];
                    const __half *row = A + static_cast<size_t>(strip) * k;
                    if (MLOOP > 1) {
#pragma unroll
                        for (int kk = 0; kk < KC; ++kk)
                            if (kc + static_cast<uint32_t>(kk) < kTiles)
                                wmma::load_matrix_sync(aLocal[kk], row + kc * 16u + kk * 16u, k);
                    }
#pragma unroll
                    for (int kk = 0; kk < KC; ++kk) {
                        if (kc + static_cast<uint32_t>(kk) >= kTiles) break;
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt) {
                            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                                           wmma::row_major> bF;
                            wmma::load_matrix_sync(
                                    bF, bT + (((buf * NT) + nt) * KC + kk) * 256, 16);
                            const AFrag &aUse = (MLOOP > 1) ? aLocal[kk] : aF[kk];
                            wmma::mma_sync(acc[ml][nt], aUse, bF, acc[ml][nt]);
                        }
                    }
                }
            };

            AFrag aF[2][KC];
            decodeChunk(0, 0);
            loadA(0, aF[0]);
            __syncthreads();
            for (int c = 0; c < nChunks; ++c) {
                const int b = c & 1, nb = b ^ 1;
                const bool more = (c + 1) < nChunks;
                if (DECODE_FIRST) {
                    // Decode the next chunk first: its ALU work then interleaves with
                    // this chunk's MMAs across warps instead of stalling in lockstep.
                    if (more) decodeChunk(c + 1, nb);
                    if (more) loadA(c + 1, aF[nb]);
                    mmaChunk(c, aF[b], b);
                } else {
                    loadA(c + 1, aF[nb]);
                    mmaChunk(c, aF[b], b);
                    if (more) decodeChunk(c + 1, nb);
                }
                __syncthreads();
            }
#pragma unroll
            for (int ml = 0; ml < MLOOP; ++ml) {
            const uint32_t strip = mBase + (ml * WARPS + warp) * 16u;
            if (strip >= m) continue;
            const uint32_t rows = (m - strip < 16u) ? (m - strip) : 16u;
            for (int nt = 0; nt < NT; ++nt) {
                const uint32_t ni = nt0 + static_cast<uint32_t>(nt);
                if (ni >= tilesN) break;
                float *cTw = reinterpret_cast<float *>(smemRaw) + warp * 256;
                wmma::store_matrix_sync(cTw, acc[ml][nt], 16, wmma::mem_row_major);
                __syncwarp();
                if (lane < 16u && lane < rows) {
                    const float *src = cTw + lane * 16u;
                    float *dst = C + static_cast<size_t>(strip + lane) * n + ni * 16u;
                    if ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0u) {
                        const float4 *s4 = reinterpret_cast<const float4 *>(src);
                        float4 *d4 = reinterpret_cast<float4 *>(dst);
#pragma unroll
                        for (int q = 0; q < 4; ++q) d4[q] = s4[q];
                    } else {
                        for (int c = 0; c < 16; ++c) dst[c] = src[c];
                    }
                }
                __syncwarp();
            }
            }
        }

        // ---- Native HMMA stage: mma.sync.m16n8k8 driven by ldmatrix ----
        // Replaces the wmma::m16n16k16 stage above (spec docs/specs/
        // exl3-hmma-fused-gemm.md).  decodeChunk is byte-identical to the wmma
        // kernel's, so the trellis decode, the tile layout and the double buffer
        // are unchanged; only the MMA stage and the epilogue differ.
        //
        // NOTE on the instruction: the spec asks for m16n8k16, which ptxas
        // rejects here ("Feature '.m16n8k16' requires .target sm_80 or higher")
        // -- Turing's widest fp16 tensor-core shape is HMMA.1688, i.e.
        // mma.sync.aligned.m16n8k8.  That is also exactly what
        // wmma::m16n16k16 decomposes into (2 n-halves x 2 k-halves = 4 HMMA per
        // wmma::mma_sync), so this kernel issues the same tensor instructions as
        // the wmma version but schedules them, and their operand loads, by hand.
        //
        // A 16x8 fp16 B fragment is ONE .b32 per lane:
        //   b0,b1 -> rows 2*(t%4)+{0,1}, col t/4          (rows = k, col = n)
        // decodeChunk stages the tile as [k][n] with n contiguous, i.e. the exact
        // transpose of that, so fragments come from `ldmatrix.x4.trans` on the
        // staging buffer.  One x4.trans yields all four B fragments of a k-tile:
        //   r0 = (k 0..7,   n 0..7)   r1 = (k 8..15, n 0..7)
        //   r2 = (k 0..7,   n 8..15)  r3 = (k 8..15, n 8..15)
        // i.e. the two n-halves of an m16n16 output and both k-halves, from one
        // instruction and with no per-fragment address arithmetic.
        //
        // A 16x8 fp16 A fragment is 2 x .b32: a0,a1 = row t/4, cols 2*(t%4)+{0,1};
        // a2,a3 = row t/4+8.  For a 16-k chunk that is 4 x .b32 per lane:
        // rows t/4, t/4+8 x cols 2*(t%4)+{0,1} and +8.  A does not have to go
        // through shared memory: the four 4-byte loads per lane land on 16 whole
        // 32-byte rows per warp, so every sector fetched is fully consumed.
        // AREAD=1 stages those rows through shared memory and uses
        // ldmatrix.x4.b16 instead, for A/B comparison only.
        //
        // Register budget (spec section 7: <= 64 accumulator regs): the
        // accumulators are MLOOP*NT*2 m16n8 tiles x 4 regs, so every shape
        // instantiated below is 32 or 64.
        __device__ __forceinline__ void hmma1688(float *d, const uint32_t *a, uint32_t b) {
            asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                    : "r"(a[0]), "r"(a[1]), "r"(b));
        }
        // fp16-accumulate twin of hmma1688: 2x the tensor-core rate on consumer
        // Turing (sm_75), where fp32-acc HMMA runs at half rate.  The D/C operand
        // is 2 x .b32 (d0,d1 = row t/4; d2,d3 = row t/4+8), half the registers of
        // the fp32 form.  Precision is protected by folding into the fp32
        // accumulators once per k-chunk, exactly as exllamav3 does
        // (EXL3_GEMM_H_ACC in exl3_gemm_inner.cuh, which is guarded to sm_86 and
        // reports ~1% of output RMS error at k=4096 -- below quantization noise).
        __device__ __forceinline__ void hmma1688H(uint32_t *d, const uint32_t *a, uint32_t b) {
            asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                    "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                    : "+r"(d[0]), "+r"(d[1])
                    : "r"(a[0]), "r"(a[1]), "r"(b));
        }
        __device__ __forceinline__ void ldmX4(uint32_t *r, const void *p) {
            const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(p));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                         "{%0,%1,%2,%3}, [%4];\n"
                         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                         : "r"(sa));
        }
        __device__ __forceinline__ void ldmX4Trans(uint32_t *r, const void *p) {
            const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(p));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                         "{%0,%1,%2,%3}, [%4];\n"
                         : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                         : "r"(sa));
        }

        // FULL: the host has verified that m covers every m-strip of the block
        // exactly, that n is a whole multiple of the block's column span and
        // that k is a whole number of KC chunks.  Every inner-loop guard then
        // disappears, which is what lets ptxas hoist the ldmatrix loads and
        // software-pipeline the mma stream instead of branching around each
        // group (the guarded and unguarded variants are both kept: the parity
        // harness runs seqLen=16, which is not a whole block).
        template <int WARPS, int MLOOP, int NT, int KC, bool DECODE_FIRST, int AREAD,
                  bool FULL, bool HACC = false>
        __launch_bounds__(WARPS * 32, 2) __global__ void gemmFusedHmmaT(const uint16_t *__restrict__ packed, uint32_t K,
                                       int codebook, uint32_t tilesN, uint32_t tileU16,
                                       const __half *__restrict__ A, float *__restrict__ C,
                                       uint32_t m, uint32_t k, uint32_t n,
                                       uint32_t dbgMode) {
            static const int kNativeLane[16] = {0, 1, 2, 3, 8, 9, 10, 11,
                                                16, 17, 18, 19, 24, 25, 26, 27};
            constexpr int kTilesPerPass = WARPS * 2;
            constexpr int kDecodePasses = (NT * KC + kTilesPerPass - 1) / kTilesPerPass;
            constexpr int kRowsPerWarp = MLOOP * 16;
            // bT: the double-buffered decoded B tiles ([k][n], pitch 16 halfs).
            // aS: optional A staging, warp-local and per k-tile; each warp owns
            // kRowsPerWarp rows of 16 halfs = 32 bytes each.
            __shared__ __align__(16) __half bT[2 * NT * KC * 256];
            __shared__ __align__(16) __half aS[AREAD != 0 ? WARPS * kRowsPerWarp * 16 : 1];
            const uint32_t warp = threadIdx.x >> 5;
            const uint32_t lane = threadIdx.x & 31u;
            const int vl = kNativeLane[lane & 15u];
            const uint32_t hw = lane >> 4;
            const uint32_t nt0 = blockIdx.x * NT;
            const uint32_t mBase = (blockIdx.y * WARPS * MLOOP) * 16u;
            const uint32_t kTiles = k / 16u;
            const int nChunks = static_cast<int>((kTiles + KC - 1) / KC);
            // A fragment mapping: row t/4 and t/4+8, cols 2*(t%4)+{0,8}.
            const uint32_t ar = lane >> 2;
            const uint32_t ac = (lane & 3u) * 2u;
            const __half *aStrip[MLOOP];    // row (strip + t/4), col 0
            bool stripOk[MLOOP];
#pragma unroll
            for (int ml = 0; ml < MLOOP; ++ml) {
                const uint32_t s = mBase + (ml * WARPS + warp) * 16u;
                stripOk[ml] = FULL || s < m;
                aStrip[ml] = A + static_cast<size_t>(s + ar) * k;
            }
            float acc[MLOOP][NT][2][4];
            // HACC: fp16 partials, folded into acc once per k-chunk (see hmma1688H).
            uint32_t accH[MLOOP][NT][2][2];
#pragma unroll
            for (int ml = 0; ml < MLOOP; ++ml)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) acc[ml][nt][h][q] = 0.0f;
#pragma unroll
                        for (int q = 0; q < 2; ++q) accH[ml][nt][h][q] = 0u;
                    }

            // ---- decode one k-chunk of NT*KC tiles into bT[buf] ----
            // Identical to gemmFusedRawT's decodeChunk (validated bit-exact).
            auto decodeChunk = [&](int chunk, int buf) {
                if ((dbgMode & 1u) != 0u) return;
                const uint32_t kc = static_cast<uint32_t>(chunk) * KC;
#pragma unroll
                for (int p = 0; p < kDecodePasses; ++p) {
                    const int t = p * kTilesPerPass + static_cast<int>(warp) * 2 +
                                  static_cast<int>(hw);
                    const int nt = t % NT;
                    const int kk = t / NT;
                    const uint32_t kt = kc + static_cast<uint32_t>(kk);
                    const uint32_t ni = nt0 + static_cast<uint32_t>(nt);
                    if (nt < NT && kk < KC && kt < kTiles && ni < tilesN) {
                        const uint32_t tile = kt * tilesN + ni;
                        const uint32_t *pp = reinterpret_cast<const uint32_t *>(
                                packed + static_cast<size_t>(tile) * tileU16);
                        float f[8], nn[8];
                        exl3LaneFragNative(pp, K, codebook, vl, f);
                        exl3LaneFragNative(pp, K, codebook, vl + 4, nn);
                        const int r0 = (vl % 4) * 2, c0 = vl / 8, c1 = c0 + 4;
                        const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
                        __half *b = bT + (((buf * NT) + nt) * KC + kk) * 256;
                        for (int i = 0; i < 4; ++i) {
                            __half *row = b + rows[i] * 16;
                            reinterpret_cast<__half2 *>(row)[c0] =
                                    __floats2half2_rn(f[i], nn[i]);
                            reinterpret_cast<__half2 *>(row)[c1] =
                                    __floats2half2_rn(f[4 + i], nn[4 + i]);
                        }
                    }
                }
            };

            // ---- A fragments for one k-tile (16 k values) ----
            auto loadA = [&](uint32_t kbase, uint32_t (&af)[MLOOP][4]) {
                if (AREAD == 0) {
#pragma unroll
                    for (int ml = 0; ml < MLOOP; ++ml) {
                        const __half *p = aStrip[ml] + kbase + ac;
#pragma unroll
                        for (int q = 0; q < 4; ++q) af[ml][q] = 0u;
                        if (FULL || stripOk[ml]) {
                            af[ml][0] = *reinterpret_cast<const uint32_t *>(p);
                            af[ml][1] = *reinterpret_cast<const uint32_t *>(
                                    p + static_cast<size_t>(8u) * k);
                            af[ml][2] = *reinterpret_cast<const uint32_t *>(p + 8u);
                            af[ml][3] = *reinterpret_cast<const uint32_t *>(
                                    p + static_cast<size_t>(8u) * k + 8u);
                        }
                    }
                    return;
                }
                // AREAD == 1: coalesced 32-byte-row copy into this warp's staging
                // area, then ldmatrix.x4.b16 per strip.
                __half *w = aS + warp * (kRowsPerWarp * 16);
                const uint32_t row = lane >> 1;            // 0..15
                const uint32_t half16 = (lane & 1u) * 8u;  // 16-byte half of the row
#pragma unroll
                for (int ml = 0; ml < MLOOP; ++ml) {
                    const uint32_t s = mBase + (ml * WARPS + warp) * 16u;
                    __half *dst = w + (ml * 16u + row) * 16u + half16;
                    if (FULL || s + row < m) {
                        const float4 v = *reinterpret_cast<const float4 *>(
                                A + static_cast<size_t>(s + row) * k + kbase + half16);
                        *reinterpret_cast<float4 *>(dst) = v;
                    }
                }
                __syncwarp();
#pragma unroll
                for (int ml = 0; ml < MLOOP; ++ml)
                    ldmX4(af[ml], w + ((ml * 16u) + (lane & 15u)) * 16u + (lane >> 4) * 8u);
            };

            // ---- MMA stage for one k-chunk ----
            auto mmaChunk = [&](int chunk, int buf) {
                if ((dbgMode & 2u) != 0u) return;
                const uint32_t kc = static_cast<uint32_t>(chunk) * KC;
                uint32_t af[2][MLOOP][4];
                loadA(kc * 16u, af[0]);
#pragma unroll
                for (int kk = 0; kk < KC; ++kk) {
                    if constexpr (!FULL) {
                        if (kc + static_cast<uint32_t>(kk) >= kTiles) break;
                    }
                    const int cur = kk & 1;
                    // A for the next k-tile is fetched while this tile's MMAs
                    // drain: the global-load latency never sits in front of an
                    // MMA (the wmma kernel's biggest stall).
                    if (kk + 1 < KC) loadA(kc * 16u + (uint32_t)(kk + 1) * 16u, af[cur ^ 1]);
                    uint32_t bf[NT][4];
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        if constexpr (!FULL) {
                            if (nt0 + static_cast<uint32_t>(nt) >= tilesN) break;
                        }
                        ldmX4Trans(bf[nt],
                                  bT + (((buf * NT) + nt) * KC + kk) * 256 +
                                          (lane & 15u) * 16u + (lane >> 4) * 8u);
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        if constexpr (!FULL) {
                            if (nt0 + static_cast<uint32_t>(nt) >= tilesN) break;
                        }
#pragma unroll
                        for (int ml = 0; ml < MLOOP; ++ml) {
                            if (!stripOk[ml]) continue;
                            const uint32_t *ap = af[cur][ml];
                            if constexpr (HACC) {
                                hmma1688H(accH[ml][nt][0], ap + 0, bf[nt][0]);
                                hmma1688H(accH[ml][nt][0], ap + 2, bf[nt][1]);
                                hmma1688H(accH[ml][nt][1], ap + 0, bf[nt][2]);
                                hmma1688H(accH[ml][nt][1], ap + 2, bf[nt][3]);
                            } else {
                                hmma1688(acc[ml][nt][0], ap + 0, bf[nt][0]);
                                hmma1688(acc[ml][nt][0], ap + 2, bf[nt][1]);
                                hmma1688(acc[ml][nt][1], ap + 0, bf[nt][2]);
                                hmma1688(acc[ml][nt][1], ap + 2, bf[nt][3]);
                            }
                        }
                    }
                }
            };

            // HACC fold: once per k-chunk, fp16 partials -> fp32 masters.
            auto foldAcc = [&]() {
#pragma unroll
                for (int ml = 0; ml < MLOOP; ++ml)
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const float2 g0 = __half22float2(
                                    *reinterpret_cast<const __half2 *>(&accH[ml][nt][h][0]));
                            const float2 g1 = __half22float2(
                                    *reinterpret_cast<const __half2 *>(&accH[ml][nt][h][1]));
                            acc[ml][nt][h][0] += g0.x;
                            acc[ml][nt][h][1] += g0.y;
                            acc[ml][nt][h][2] += g1.x;
                            acc[ml][nt][h][3] += g1.y;
                            accH[ml][nt][h][0] = 0u;
                            accH[ml][nt][h][1] = 0u;
                        }
            };
            decodeChunk(0, 0);
            __syncthreads();
            for (int c = 0; c < nChunks; ++c) {
                const int b = c & 1, nb = b ^ 1;
                const bool more = (c + 1) < nChunks;
                if (DECODE_FIRST) {
                    if (more) decodeChunk(c + 1, nb);
                    mmaChunk(c, b);
                    if constexpr (HACC) foldAcc();
                } else {
                    mmaChunk(c, b);
                    if constexpr (HACC) foldAcc();
                    if (more) decodeChunk(c + 1, nb);
                }
                __syncthreads();
            }

            // ---- epilogue: store straight from the accumulators ----
            // C/lane holds rows t/4, t/4+8 x cols 2*(t%4)+{0,1}, so a float2 per
            // row half writes the tile with no shared-memory bounce.  Rows past
            // m are masked, so a partial strip cannot write past y.
#pragma unroll
            for (int ml = 0; ml < MLOOP; ++ml) {
                if (!stripOk[ml]) continue;
                const uint32_t strip = mBase + (ml * WARPS + warp) * 16u;
                const uint32_t rows = FULL ? 16u
                                           : ((m - strip < 16u) ? (m - strip) : 16u);
                const uint32_t r0 = lane >> 2, r1 = r0 + 8u;
                const uint32_t cOff = (lane & 3u) * 2u;
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const uint32_t ni = nt0 + static_cast<uint32_t>(nt);
                    if constexpr (!FULL) {
                        if (ni >= tilesN) break;
                    }
                    float *dst = C + static_cast<size_t>(strip + r0) * n +
                                 ni * 16u + cOff;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint32_t co = static_cast<uint32_t>(h) * 8u;
                        if (r0 < rows)
                            *reinterpret_cast<float2 *>(dst + co) =
                                    make_float2(acc[ml][nt][h][0], acc[ml][nt][h][1]);
                        if (r1 < rows)
                            *reinterpret_cast<float2 *>(dst + co +
                                                        static_cast<size_t>(8u) * n) =
                                    make_float2(acc[ml][nt][h][2], acc[ml][nt][h][3]);
                    }
                }
            }
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
            // blockIdx.y splits the n columns: the original looped over all n
            // serially inside k/128 blocks, which left the GPU 99% idle and
            // dominated EXL3 prefill (5 ms per matrix).
            const uint32_t c0 = blockIdx.y * 8u;
            const uint32_t c1 = (c0 + 8u) < n ? (c0 + 8u) : n;
            for (uint32_t c = c0; c < c1; ++c) {
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
            // blockIdx.y splits the 128-wide n bands across blocks.
            const uint32_t nb0 = blockIdx.y * 128u;
            const uint32_t nb1 = (nb0 + 128u) < n ? (nb0 + 128u) : n;
            for (uint32_t nb = nb0; nb < nb1; nb += 128u) {
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
            } else if (lin.d.bitrate10 / 10u <= 3u) {
                decodeTilesIntNative<<<tilesK * tilesN, 32, 0, stream()>>>(
                        static_cast<const uint16_t *>(lin.d.packed), lin.d.bitrate10 / 10u,
                        lin.d.codebook, tilesN, lin.d.tileU16, lin.d.scratch, n);
            } else {
                decodeTilesInt<<<tilesK * tilesN, 256, 0, stream()>>>(
                        static_cast<const uint16_t *>(lin.d.packed), lin.d.bitrate10 / 10u,
                        lin.d.codebook, tilesN, lin.d.tileU16, lin.d.scratch, n);
            }
            // leftH128 covers 8 columns per block, so grid.y must span n/8;
            // launching tilesK blocks only transformed the first 8 columns and
            // made reconstruct() disagree with the CPU codec (matrix parity
            // 6.5-99.7 in tools/exl3_gpu_parity).
            leftH128<<<dim3(tilesK, (n + 7u) / 8u), 128, 0, stream()>>>(
                    lin.d.suh, lin.d.scratch, k, n);
            rightH128<<<dim3(k, (n + 127u) / 128u), 128, 0, stream()>>>(
                    lin.d.svh, lin.d.scratch, k, n);
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

        // -------------------------------------------------------------------
        // V2 decode GEMV (opt-in: TINYCODER_EXL3_GEMV=2).
        //
        // Three defects of exl3FusedMatVec are fixed here:
        //  * all 32 lanes own distinct columns (V1 used `lane & 15`, so lanes
        //    16-31 duplicated 0-15, and only 16 lanes ever decoded);
        //  * the 128-point activation Hadamard is computed once per 128-k
        //    chunk by the whole block into shared memory, not per warp;
        //  * the trellis is read by consecutive lanes at consecutive word
        //    addresses and redistributed with __shfl_sync, instead ofeach lane
        //    gathering its own strided bit window.
        //
        // Block = 256 threads (8 warps).  A group of SPLIT warps owns one
        // 16-column output tile and splits the k range between its warps;
        // 8/SPLIT groups per block.  Within a 128-k chunk the 8 row-tiles are
        // spread over the group's warps, and every one of them needs the whole
        // transformed chunk, so the eight warps transform eight chunks in
        // parallel into shared memory once per round and all of them consume
        // it with a single barrier per round (no per-tile barriers at all).
        //
        // Lane role inside one 16x16 tile (same decomposition as
        // exl3LaneFrag, so the decode is bit-identical to the V1 path):
        //   rowpair = lane & 3, frag = (lane >> 2) & 1, colgroup = lane >> 3
        //   rows = 2*rowpair + {0,1,8,9}; cols = 2*(colgroup + 4*frag) + {0,1}
        // 32 lanes x 8 weights = 256 weights = one tile, one pass, no idle
        // lanes.  The 4 lanes sharing a column pair are contiguous, so the
        // partial sums reduce with two shfl_xor.
        // -------------------------------------------------------------------
        __device__ __forceinline__ int exl3RowOff(int i) {
            return (i & 1) + ((i & 2) ? 8 : 0);   // 0,1,8,9
        }

        // 128-point FWHT of x[kb..kb+128) with the suh pre-scale, one warp,
        // four values per lane.  Butterfly order is copied verbatim from
        // exl3FusedMatVec so the transform stays numerically identical.
        __device__ __forceinline__ void exl3Fwht128(const float *__restrict__ x,
                                                    const float *__restrict__ suh,
                                                    uint32_t kb, int lane,
                                                    float *__restrict__ out) {
            const uint32_t i0 = static_cast<uint32_t>(lane) * 4u;
            float v0 = x[kb + i0];
            float v1 = x[kb + i0 + 1u];
            float v2 = x[kb + i0 + 2u];
            float v3 = x[kb + i0 + 3u];
            if (suh) {
                v0 *= suh[kb + i0];
                v1 *= suh[kb + i0 + 1u];
                v2 *= suh[kb + i0 + 2u];
                v3 *= suh[kb + i0 + 3u];
            }
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
#pragma unroll
            for (int st = 0; st < 5; ++st) {
                const int bit = 1 << st;
                const float o0 = __shfl_xor_sync(0xffffffffu, v0, bit);
                const float o1 = __shfl_xor_sync(0xffffffffu, v1, bit);
                const float o2 = __shfl_xor_sync(0xffffffffu, v2, bit);
                const float o3 = __shfl_xor_sync(0xffffffffu, v3, bit);
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
            out[i0] = v0 * kScale;
            out[i0 + 1u] = v1 * kScale;
            out[i0 + 2u] = v2 * kScale;
            out[i0 + 3u] = v3 * kScale;
        }

        // Which two words of the tile a lane needs, and at what bit offset.
        // Mirrors the per-lane 16-bit-window stream of exl3LaneFrag; depends
        // only on the lane id, so it is hoisted out of the k loop.
        __device__ __forceinline__ void exl3LaneWords(uint32_t K, int lane,
                                                      uint32_t &loIdx,
                                                      uint32_t &hiIdx,
                                                      uint32_t &sh) {
            const uint32_t to = static_cast<uint32_t>(lane) * 8u;
            if (K == 1u) {
                const uint32_t i1 = to >> 5;
                loIdx = i1;
                hiIdx = (i1 + 7u) & 7u;
                sh = (~to) & 24u;
            } else if (K == 2u) {
                const uint32_t i1 = to >> 4;
                loIdx = i1;
                hiIdx = (i1 + 15u) & 15u;
                sh = ((~to) & 8u) << 1;
            } else {
                const int nw = static_cast<int>(8u * K);
                const int b1 = static_cast<int>(to + 257u) * 3;
                const int b2 = b1 + 21;
                const int i0 = (b1 - 16) / 32;
                const int i2 = (b2 - 1) / 32;
                loIdx = static_cast<uint32_t>(i2 % nw);
                hiIdx = static_cast<uint32_t>(i0 % nw);
                sh = static_cast<uint32_t>((i2 + 1) * 32 - b2);
            }
        }

        template <int CB, int SPLIT>
        __global__ void exl3GemvV2(const uint16_t *__restrict__ trellis,
                                   const float *__restrict__ suh,
                                   uint32_t k, uint32_t K,
                                   uint32_t tilesN, uint32_t tileU16,
                                   const float *__restrict__ x,
                                   float *__restrict__ z) {
            constexpr int GROUPS = 8 / SPLIT;
            const int warp = threadIdx.x >> 5;
            const int lane = threadIdx.x & 31;
            const int group = warp / SPLIT;
            const int wi = warp % SPLIT;
            const uint32_t pair = blockIdx.x * GROUPS + static_cast<uint32_t>(group);
            const uint32_t nWords = 8u * K;
            const uint32_t nChunks = k >> 7;

            uint32_t loIdx, hiIdx, sh;
            exl3LaneWords(K, lane, loIdx, hiIdx, sh);
            const uint32_t rowOff = static_cast<uint32_t>(lane & 3) * 2u;
            // A frag's first four weights land on column B, the last four on
            // column B+8 (see exl3FusedMatVec's put(): f[4..7] go to c1*2 with
            // c1 = c0+4).  B covers 0..7 over (colgroup, frag).
            const int frag = (lane >> 2) & 1;
            const uint32_t colA = static_cast<uint32_t>(2 * (lane >> 3) + frag);

            __shared__ float sX[8][128];// eight transformed 128-k chunks
            __shared__ float sRed[GROUPS][SPLIT][16];

            float accA = 0.0f, accB = 0.0f;

            for (uint32_t c0 = 0; c0 < nChunks; c0 += 8u) {
                // Every warp transforms one chunk: all 256 threads cooperate,
                // each chunk is transformed exactly once per block.
                const uint32_t ch = c0 + static_cast<uint32_t>(warp);
                if (ch < nChunks) {
                    exl3Fwht128(x, suh, ch * 128u, lane, sX[warp]);
                }
                __syncthreads();
                uint32_t nC = nChunks - c0;
                if (nC > 8u) nC = 8u;
                for (uint32_t cc = 0; cc < nC; ++cc) {
                    const float *xs = sX[cc];
                    const uint32_t kt0 = (c0 + cc) * 8u;
                    for (uint32_t t = static_cast<uint32_t>(wi); t < 8u;
                         t += SPLIT) {
                        const size_t off =
                                (static_cast<size_t>(kt0 + t) * tilesN + pair) *
                                static_cast<size_t>(tileU16);
                        const uint32_t *p32 =
                                reinterpret_cast<const uint32_t *>(trellis + off);
                        // Consecutive lanes read consecutive words: 24 lanes
                        // (K=3) cover one tile's 96 bytes as three sectors.
                        uint32_t w = (static_cast<uint32_t>(lane) < nWords)
                                             ? p32[lane]
                                             : 0u;
                        const uint32_t lo =
                                __shfl_sync(0xffffffffu, w, static_cast<int>(loIdx));
                        const uint32_t hi =
                                __shfl_sync(0xffffffffu, w, static_cast<int>(hiIdx));
                        const uint64_t m = (static_cast<uint64_t>(hi) << 32) | lo;
                        const float *xsr = xs + t * 16u + rowOff;
                        // Code i of the lane stream is w[i] = the window at
                        // sh + K*(7-i): exl3LaneFrag's w[] runs from the last
                        // code (w[7], lowest window) back to the first.
                        uint32_t bp = sh + 7u * K;
                        float aA = 0.0f, aB = 0.0f;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            // 16-bit window of the per-lane stream: same value
                            // as exl3LaneFrag's w[i] for every codebook.
                            const float f = decodeCode(
                                    static_cast<uint32_t>((m >> bp) & 0xffffu),
                                    CB);
                            bp -= K;
                            const float xv = xsr[exl3RowOff(i & 3)];
                            if (i < 4) {
                                aA = fmaf(f, xv, aA);
                            } else {
                                aB = fmaf(f, xv, aB);
                            }
                        }
                        accA += aA;
                        accB += aB;
                    }
                }
                __syncthreads();
            }

            // The four lanes sharing a column pair hold the four row groups.
            float sA = accA, sB = accB;
            sA += __shfl_xor_sync(0xffffffffu, sA, 1);
            sB += __shfl_xor_sync(0xffffffffu, sB, 1);
            sA += __shfl_xor_sync(0xffffffffu, sA, 2);
            sB += __shfl_xor_sync(0xffffffffu, sB, 2);
            if ((lane & 3) == 0) {
                sRed[group][wi][colA] = sA;
                sRed[group][wi][colA + 8u] = sB;
            }
            __syncthreads();
            const int tid = threadIdx.x;
            if (tid < GROUPS * 16) {
                const int g = tid >> 4;
                const int col = tid & 15;
                float sum = 0.0f;
#pragma unroll
                for (int q = 0; q < SPLIT; ++q) sum += sRed[g][q][col];
                z[(blockIdx.x * GROUPS + static_cast<uint32_t>(g)) * 16u +
                  static_cast<uint32_t>(col)] = sum;
            }
        }

        // -------------------------------------------------------------------
        // V3 decode GEMV: V2 with a deeper memory pipeline.
        //
        // V2 issues ONE trellis load per warp per k-chunk and then immediately
        // consumes it, so a warp has a single request in flight and the two
        // __shfl_sync calls between loads act as scheduling barriers: the
        // compiler cannot overlap adjacent tiles' loads.  Instrumented on the
        // 27B that shows up as a constant ~420 G weights/s across EVERY shape
        // and every SPLIT, and as bandwidth that scales with bytes-per-tile
        // (K=3 -> 128-200 GB/s, K=1 -> 46-58 GB/s) rather than with anything
        // the kernel does differently: the limit is loads-in-flight, not DRAM.
        //
        // V3 gives each warp TT CONSECUTIVE output tiles and issues all
        // 8/SPLIT x TT loads for the chunk BEFORE decoding any of them, so a
        // warp keeps TT (or more) requests outstanding.  The per-column
        // accumulation order is untouched (each tile still sees its k-chunk in
        // the same order, the four-lane shuffle tree and the SPLIT reduction
        // are unchanged), so the output stays bit-identical to V2.
        // -------------------------------------------------------------------
        template <int CB, int SPLIT, int TT, int L8 = 0>
        __global__ void exl3GemvV3(const uint16_t *__restrict__ trellis,
                                   const float *__restrict__ suh,
                                   uint32_t k, uint32_t K,
                                   uint32_t tilesN, uint32_t tileU16,
                                   const float *__restrict__ x,
                                   float *__restrict__ z) {
            constexpr int GROUPS = 8 / SPLIT;
            constexpr int KT = 8 / SPLIT;// k-tiles per warp per chunk
            const int warp = threadIdx.x >> 5;
            const int lane = threadIdx.x & 31;
            const int group = warp / SPLIT;
            const int wi = warp % SPLIT;
            const uint32_t pair0 =
                    blockIdx.x * static_cast<uint32_t>(GROUPS * TT) +
                    static_cast<uint32_t>(group * TT);
            const uint32_t nWords = 8u * K;
            const uint32_t nChunks = k >> 7;
            const bool loads = (static_cast<uint32_t>(lane) < nWords);

            uint32_t loIdx, hiIdx, sh;
            exl3LaneWords(K, lane, loIdx, hiIdx, sh);
            const uint32_t rowOff = static_cast<uint32_t>(lane & 3) * 2u;
            const int frag = (lane >> 2) & 1;
            const uint32_t colA = static_cast<uint32_t>(2 * (lane >> 3) + frag);

            __shared__ float sX[8][128];
            // K<=3: a code is 0..7, so one 32-bit shared lookup replaces the
            // nine-instruction 3INST sequence (IMAD, LOP3, LOP3, two
            // HADD2.F32, FADD) with LDS.U.32 + FFMA.  The table is built from
            // decodeCode(), so the values are bit-identical.
            __shared__ float sLut8[8];
            __shared__ float sRed[GROUPS][SPLIT][TT * 16];

            if (L8 && threadIdx.x < 8) {
                sLut8[threadIdx.x] = decodeCode(static_cast<uint32_t>(threadIdx.x), CB);
            }
            float accA[TT], accB[TT];
#pragma unroll
            for (int i = 0; i < TT; ++i) {
                accA[i] = 0.0f;
                accB[i] = 0.0f;
            }

            for (uint32_t c0 = 0; c0 < nChunks; c0 += 8u) {
                const uint32_t ch = c0 + static_cast<uint32_t>(warp);
                if (ch < nChunks) {
                    exl3Fwht128(x, suh, ch * 128u, lane, sX[warp]);
                }
                __syncthreads();
                uint32_t nC = nChunks - c0;
                if (nC > 8u) nC = 8u;
                // Double-buffered trellis loads: the next chunk's requests are
                // issued BEFORE this chunk is decoded, so the decode hides
                // their latency instead of stalling on it.  Without this the
                // kernel serialises load-latency and ALU work: with the decode
                // arithmetic removed it runs 2.15x faster (measured on the
                // 27B), i.e. the two halves never overlapped.
                uint32_t wc[KT][TT], wn[KT][TT];
#define EXL3_LOAD_CHUNK(dst, cc_)                                                \
    do {                                                                         \
        const uint32_t kt_ = (c0 + (cc_)) * 8u;                                  \
        _Pragma("unroll") for (int kk = 0; kk < KT; ++kk) {                       \
            const uint32_t t_ = static_cast<uint32_t>(wi + kk * SPLIT);          \
            _Pragma("unroll") for (int tt = 0; tt < TT; ++tt) {                   \
                const size_t off_ =                                              \
                        (static_cast<size_t>(kt_ + t_) * tilesN +                \
                         (pair0 + static_cast<uint32_t>(tt))) *                  \
                        static_cast<size_t>(tileU16);                            \
                const uint32_t *p32_ =                                           \
                        reinterpret_cast<const uint32_t *>(trellis + off_);       \
                dst[kk][tt] = loads ? p32_[lane] : 0u;                            \
            }                                                                    \
        }                                                                        \
    } while (0)
                EXL3_LOAD_CHUNK(wc, 0u);
                for (uint32_t cc = 0; cc < nC; ++cc) {
                    if (cc + 1u < nC) EXL3_LOAD_CHUNK(wn, cc + 1u);
                    const float *xs = sX[cc];
                    const uint32_t kt0 = (c0 + cc) * 8u;
#pragma unroll
                    for (int kk = 0; kk < KT; ++kk) {
                        const uint32_t t = static_cast<uint32_t>(wi + kk * SPLIT);
                        const float *xsr = xs + t * 16u + rowOff;
#pragma unroll
                        for (int tt = 0; tt < TT; ++tt) {
                            const uint32_t lo =
                                    __shfl_sync(0xffffffffu, wc[kk][tt], static_cast<int>(loIdx));
                            const uint32_t hi =
                                    __shfl_sync(0xffffffffu, wc[kk][tt], static_cast<int>(hiIdx));
                            const uint64_t m = (static_cast<uint64_t>(hi) << 32) | lo;
                            uint32_t bp = sh + 7u * K;
                            float aA = 0.0f, aB = 0.0f;
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const float f =
                                        L8 ? sLut8[static_cast<uint32_t>((m >> bp) & 0x7u)]
                                           : decodeCode(
                                                     static_cast<uint32_t>((m >> bp) &
                                                                           0xffffu),
                                                     CB);
                                bp -= K;
                                const float xv = xsr[exl3RowOff(i & 3)];
                                if (i < 4) {
                                    aA = fmaf(f, xv, aA);
                                } else {
                                    aB = fmaf(f, xv, aB);
                                }
                            }
                            accA[tt] += aA;
                            accB[tt] += aB;
                        }
                    }
#pragma unroll
                    for (int kk = 0; kk < KT; ++kk)
#pragma unroll
                        for (int tt = 0; tt < TT; ++tt) wc[kk][tt] = wn[kk][tt];
                }
#undef EXL3_LOAD_CHUNK
                __syncthreads();
            }

            const uint32_t laneGroup = static_cast<uint32_t>(lane & 3);
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) {
                float sA = accA[tt], sB = accB[tt];
                sA += __shfl_xor_sync(0xffffffffu, sA, 1);
                sB += __shfl_xor_sync(0xffffffffu, sB, 1);
                sA += __shfl_xor_sync(0xffffffffu, sA, 2);
                sB += __shfl_xor_sync(0xffffffffu, sB, 2);
                if (laneGroup == 0u) {
                    sRed[group][wi][tt * 16 + static_cast<int>(colA)] = sA;
                    sRed[group][wi][tt * 16 + static_cast<int>(colA) + 8] = sB;
                }
            }
            __syncthreads();
            const int tid = threadIdx.x;
            if (tid < GROUPS * TT * 16) {
                const int g = tid / (TT * 16);
                const int rem = tid % (TT * 16);
                const int tt = rem >> 4;
                const int col = rem & 15;
                float sum = 0.0f;
#pragma unroll
                for (int q = 0; q < SPLIT; ++q) sum += sRed[g][q][tt * 16 + col];
                const uint32_t pair =
                        blockIdx.x * static_cast<uint32_t>(GROUPS * TT) +
                        static_cast<uint32_t>(g * TT + tt);
                z[pair * 16u + static_cast<uint32_t>(col)] = sum;
            }
        }

        // -------------------------------------------------------------------
        // V4: the K=1 decode GEMV.
        //
        // At one bit per weight a code is a SYMBOL FROM A TWO-ELEMENT
        // ALPHABET: the 16-bit window a lane reads carries a single bit, so
        // decodeCode() is only ever asked for decode(0) or decode(1).  The
        // general path still spends the whole 3INST sequence on it -
        // IMAD, LOP3, LOP3, two HADD2.F32 and an FADD - to answer a yes/no
        // question, i.e. ~9 instructions per weight where one FMA suffices.
        // Measured on the 27B: the K=1 shapes are 64% of decode time for 42%
        // of the bytes, and the rate is FLAT in weights/s across every K,
        // which is the signature of an instruction-bound kernel.
        //
        // Two things change here, both of which keep the arithmetic
        // bit-identical (the two floats are exact fp16 values, and the FMA
        // order per accumulator is untouched):
        //
        //  1. The code pair is resolved with a 16-entry shared LUT of float4
        //     (one nibble -> four decoded floats, one LDS.128), so a lane
        //     decodes eight weights with two LDS.128 + eight FMAs.  The nibble
        //     split falls out of the layout: codes 0..3 (aA) are bits 7..4 of
        //     the lane's byte and codes 4..7 (aB) are bits 3..0.
        //  2. The load is one 128-byte warp transaction for TT=4 tiles
        //     (2 x 64 B for TT=2, 1 x 32 B for TT=1) instead of TT separate
        //     8-lane 32-byte requests, and the second (hi) word shuffle is
        //     dead: for K=1 all eight code bits of a lane live in ONE word.
        // -------------------------------------------------------------------
        template <int CB, int SPLIT, int TT, int DIAG = 0>
        __global__ void exl3GemvK1(const uint16_t *__restrict__ trellis,
                                   const float *__restrict__ suh, uint32_t k,
                                   uint32_t tilesN, uint32_t tileU16,
                                   const float *__restrict__ x,
                                   float *__restrict__ z) {
            constexpr int GROUPS = 8 / SPLIT;
            constexpr int KT = 8 / SPLIT;// k-tiles per warp per chunk
            const int warp = threadIdx.x >> 5;
            const int lane = threadIdx.x & 31;
            const int group = warp / SPLIT;
            const int wi = warp % SPLIT;
            const uint32_t pair0 =
                    blockIdx.x * static_cast<uint32_t>(GROUPS * TT) +
                    static_cast<uint32_t>(group * TT);
            const uint32_t nChunks = k >> 7;
            // Byte of the tile this lane's eight codes live in, and the word
            // that owns it.  exl3LaneWords(K=1) is loIdx = lane>>2 with
            // sh = ~lane*8 & 24, i.e. byte 4*(lane>>2) + 3-(lane&3).
            const uint32_t byteShift = 8u * ((~static_cast<uint32_t>(lane)) & 3u);
            const uint32_t wordInTile = static_cast<uint32_t>(lane) >> 2;
            const uint32_t rowOff = static_cast<uint32_t>(lane & 3) * 2u;
            const int frag = (lane >> 2) & 1;
            const uint32_t colA = static_cast<uint32_t>(2 * (lane >> 3) + frag);

            __shared__ float sX[8][128];
            __shared__ float4 sLut[16];
            __shared__ float sRed[GROUPS][SPLIT][TT * 16];

            if (threadIdx.x < 16) {
                const uint32_t n = static_cast<uint32_t>(threadIdx.x);
                // Entry n = the four weights selected by its four bits, MSB
                // first, because code i of a nibble is bit (3-i) of it.
                float4 v;
                v.x = decodeCode((n >> 3) & 1u, CB);
                v.y = decodeCode((n >> 2) & 1u, CB);
                v.z = decodeCode((n >> 1) & 1u, CB);
                v.w = decodeCode(n & 1u, CB);
                sLut[n] = v;
            }

            float accA[TT], accB[TT];
#pragma unroll
            for (int i = 0; i < TT; ++i) {
                accA[i] = 0.0f;
                accB[i] = 0.0f;
            }

            // Whole-round software pipeline: a round of eight 128-value k
            // chunks is prefetched into registers, and the NEXT round's loads
            // are issued before the barrier that publishes this round's sX.
            // They do not depend on sX, so this is free, and it puts eight
            // 128 B warp loads (1 KB) in flight instead of one.  That matters
            // because with the decode ALU made cheap the K=1 shapes stopped
            // being instruction-bound and became LATENCY-bound: TT=4 beat
            // TT=2 beat TT=1 in the sweep (84.9 / 78.4 / 68.2 GB/s), which is
            // the signature of too few requests outstanding, not of DRAM.
            uint32_t wc[8][KT], wn[8][KT];
#define EXL3_LOAD_ROUND_K1(dst, base_)                                           \
    _Pragma("unroll") for (int cc_ = 0; cc_ < 8; ++cc_) {                         \
        if ((base_) + static_cast<uint32_t>(cc_) < nChunks) {                     \
            _Pragma("unroll") for (int kk = 0; kk < KT; ++kk) {                   \
                const uint32_t t_ = static_cast<uint32_t>(wi + kk * SPLIT);       \
                const size_t off_ =                                              \
                        (static_cast<size_t>(((base_) + static_cast<uint32_t>(cc_)) * 8u + \
                                             t_) * tilesN + pair0) *             \
                        static_cast<size_t>(tileU16);                             \
                const uint32_t *p32_ =                                           \
                        reinterpret_cast<const uint32_t *>(trellis + off_);       \
                dst[cc_][kk] = (lane < 8 * TT) ? p32_[lane] : 0u;                 \
            }                                                                    \
        }                                                                        \
    } while (0)
            EXL3_LOAD_ROUND_K1(wc, 0u);
            for (uint32_t c0 = 0; c0 < nChunks; c0 += 8u) {
                const uint32_t ch = c0 + static_cast<uint32_t>(warp);
                if (ch < nChunks) {
                    exl3Fwht128(x, suh, ch * 128u, lane, sX[warp]);
                }
                EXL3_LOAD_ROUND_K1(wn, c0 + 8u);
                __syncthreads();
                uint32_t nC = nChunks - c0;
                if (nC > 8u) nC = 8u;
#pragma unroll
                for (int cc = 0; cc < 8; ++cc) {
                    if (cc >= static_cast<int>(nC)) continue;
                    const float *xs = sX[cc];
#pragma unroll
                    for (int kk = 0; kk < KT; ++kk) {
                        const uint32_t t = static_cast<uint32_t>(wi + kk * SPLIT);
                        const float *xsr = xs + t * 16u + rowOff;
                        const float x0 = xsr[0];
                        const float x1 = xsr[1];
                        const float x8 = xsr[8];
                        const float x9 = xsr[9];
#pragma unroll
                        for (int tt = 0; tt < TT; ++tt) {
                            // Word (lane>>2) of tile tt.
                            const uint32_t w = __shfl_sync(
                                    0xffffffffu, wc[cc][kk],
                                    static_cast<int>(static_cast<uint32_t>(tt) * 8u +
                                                     wordInTile));
                            const uint32_t byte = (w >> byteShift) & 0xFFu;
                            if (DIAG == 1) {   // diagnostic: no LUT, no FMA
                                accA[tt] += __uint_as_float(byte);
                                accB[tt] += __uint_as_float(w);
                                continue;
                            }
                            const float4 hi = sLut[byte >> 4];
                            const float4 lo = sLut[byte & 15u];
                            float aA = 0.0f, aB = 0.0f;
                            aA = fmaf(hi.x, x0, aA);
                            aA = fmaf(hi.y, x1, aA);
                            aA = fmaf(hi.z, x8, aA);
                            aA = fmaf(hi.w, x9, aA);
                            aB = fmaf(lo.x, x0, aB);
                            aB = fmaf(lo.y, x1, aB);
                            aB = fmaf(lo.z, x8, aB);
                            aB = fmaf(lo.w, x9, aB);
                            accA[tt] += aA;
                            accB[tt] += aB;
                        }
                    }
                }
                __syncthreads();
#pragma unroll
                for (int cc = 0; cc < 8; ++cc)
#pragma unroll
                    for (int kk = 0; kk < KT; ++kk) wc[cc][kk] = wn[cc][kk];
            }
#undef EXL3_LOAD_ROUND_K1

            const uint32_t laneGroup = static_cast<uint32_t>(lane & 3);
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) {
                float sA = accA[tt], sB = accB[tt];
                sA += __shfl_xor_sync(0xffffffffu, sA, 1);
                sB += __shfl_xor_sync(0xffffffffu, sB, 1);
                sA += __shfl_xor_sync(0xffffffffu, sA, 2);
                sB += __shfl_xor_sync(0xffffffffu, sB, 2);
                if (laneGroup == 0u) {
                    sRed[group][wi][tt * 16 + static_cast<int>(colA)] = sA;
                    sRed[group][wi][tt * 16 + static_cast<int>(colA) + 8] = sB;
                }
            }
            __syncthreads();
            const int tid = threadIdx.x;
            if (tid < GROUPS * TT * 16) {
                const int g = tid / (TT * 16);
                const int rem = tid % (TT * 16);
                const int tt = rem >> 4;
                const int col = rem & 15;
                float sum = 0.0f;
#pragma unroll
                for (int q = 0; q < SPLIT; ++q) sum += sRed[g][q][tt * 16 + col];
                const uint32_t pair =
                        blockIdx.x * static_cast<uint32_t>(GROUPS * TT) +
                        static_cast<uint32_t>(g * TT + tt);
                z[pair * 16u + static_cast<uint32_t>(col)] = sum;
            }
        }

        // Tiles per warp for V3.  0 = auto: the largest of {4,2} that still
        // leaves at least two blocks per SM (144 blocks on this card), else 1.
        // A shape that cannot fill the GPU wants blocks, not depth.
        inline int gemvTtEnv() {
            static const int s = [] {
                const char *e = std::getenv("TINYCODER_EXL3_GEMV_TT");
                int v = e ? std::atoi(e) : 0;
                if (v != 0 && v != 1 && v != 2 && v != 4) v = 0;
                return v;
            }();
            return s;
        }

        // TINYCODER_EXL3_GEMV_SPLIT = warps cooperating on one 16-column tile
        // (1/2/4/8).  Larger values give more blocks and a wider k split.
        inline int gemvSplitEnv() {
            static const int s = [] {
                const char *e = std::getenv("TINYCODER_EXL3_GEMV_SPLIT");
                int v = e ? std::atoi(e) : 8;
                if (v != 1 && v != 2 && v != 4 && v != 8) v = 8;
                return v;
            }();
            return s;
        }

        // TINYCODER_EXL3_GEMV_LUT=0 disables the 8-entry code table in V3.
        inline bool gemvLut8On() {
            static const bool s = [] {
                const char *e = std::getenv("TINYCODER_EXL3_GEMV_LUT");
                return !(e && std::atoi(e) == 0);
            }();
            return s;
        }

        // TINYCODER_EXL3_GEMV_K1=0 disables the K=1 kernel (A/B against V3).
        inline bool gemvK1On() {
            static const bool s = [] {
                const char *e = std::getenv("TINYCODER_EXL3_GEMV_K1");
                return !(e && std::atoi(e) == 0);
            }();
            return s;
        }

        // DIAGNOSTIC ONLY (TINYCODER_EXL3_K1DIAG=1): skips the LUT decode and
        // the FMAs, keeping loads and shuffles.  Results are wrong by design;
        // it isolates the memory side of the K=1 kernel.
        inline bool gemvK1Diag() {
            static const bool s = std::getenv("TINYCODER_EXL3_K1DIAG") != nullptr;
            return s;
        }

        // Tiles per warp for the K=1 kernel; 0 = auto.
        inline int gemvK1TtEnv() {
            static const int s = [] {
                const char *e = std::getenv("TINYCODER_EXL3_GEMV_K1TT");
                int v = e ? std::atoi(e) : 0;
                if (v != 0 && v != 1 && v != 2 && v != 4) v = 0;
                return v;
            }();
            return s;
        }

        template <int CB>
        bool launchGemvK1(int split, int tt, const Exl3FusedLinear &lin,
                          const float *xDev, float *yDev, cudaStream_t stream) {
            const uint32_t tilesN = lin.n / 16u;
            const uint32_t tileU16 = lin.tileU16;
            const uint32_t groups = 8u / static_cast<uint32_t>(split);
            uint32_t use = (tt == 4) ? 4u : (tt == 2) ? 2u : (tt == 1) ? 1u : 0u;
            if (use == 0u) {
                use = 1u;
                for (uint32_t cand : {4u, 2u}) {
                    const uint32_t perCand = groups * cand;
                    if (tilesN % perCand != 0) continue;
                    if (tilesN / perCand < 144u) continue;
                    use = cand;
                    break;
                }
            }
            const uint32_t per = groups * use;
            if (per == 0u || tilesN % per != 0) return false;
            const uint32_t grid = tilesN / per;
            if (grid == 0u) return false;
#define LAUNCH_K1(SP, TTV)                                                       \
    do {                                                                         \
        if (gemvK1Diag()) {                                                      \
            exl3GemvK1<CB, SP, TTV, 1><<<grid, 256, 0, stream>>>(                \
                    reinterpret_cast<const uint16_t *>(lin.trellis), lin.suh,    \
                    lin.k, tilesN, tileU16, xDev, yDev);                         \
        } else {                                                                 \
            exl3GemvK1<CB, SP, TTV, 0><<<grid, 256, 0, stream>>>(                \
                    reinterpret_cast<const uint16_t *>(lin.trellis), lin.suh,    \
                    lin.k, tilesN, tileU16, xDev, yDev);                         \
        }                                                                        \
    } while (0)
            if (use == 4u) {
                switch (split) {
                    case 8: LAUNCH_K1(8, 4); break;
                    case 4: LAUNCH_K1(4, 4); break;
                    case 2: LAUNCH_K1(2, 4); break;
                    default: LAUNCH_K1(1, 4); break;
                }
            } else if (use == 2u) {
                switch (split) {
                    case 8: LAUNCH_K1(8, 2); break;
                    case 4: LAUNCH_K1(4, 2); break;
                    case 2: LAUNCH_K1(2, 2); break;
                    default: LAUNCH_K1(1, 2); break;
                }
            } else {
                switch (split) {
                    case 8: LAUNCH_K1(8, 1); break;
                    case 4: LAUNCH_K1(4, 1); break;
                    case 2: LAUNCH_K1(2, 1); break;
                    default: LAUNCH_K1(1, 1); break;
                }
            }
#undef LAUNCH_K1
            return true;
        }

        template <int CB>
        void launchGemvV2(int split, const Exl3FusedLinear &lin,
                          const float *xDev, float *yDev, cudaStream_t stream) {
            const uint32_t tilesN = lin.n / 16u;
            const uint32_t tileU16 = lin.tileU16;
#define LAUNCH_V2(SP)                                                            \
    do {                                                                         \
        const uint32_t grid = tilesN / (8u / (SP));                              \
        exl3GemvV2<CB, SP><<<grid, 256, 0, stream>>>(                            \
                reinterpret_cast<const uint16_t *>(lin.trellis), lin.suh, lin.k,  \
                lin.K, tilesN, tileU16, xDev, yDev);                             \
    } while (0)
            switch (split) {
                case 8: LAUNCH_V2(8); break;
                case 4: LAUNCH_V2(4); break;
                case 2: LAUNCH_V2(2); break;
                default: LAUNCH_V2(1); break;
            }
#undef LAUNCH_V2
        }

        // V3 launcher: `tt` is the caller's tiles-per-warp request (0 = auto).
        // Returns false when no V3 geometry divides this n, so gemvFused can
        // fall back to V2 rather than launch a partial grid.
        template <int CB>
        bool launchGemvV3(int split, int tt, const Exl3FusedLinear &lin,
                          const float *xDev, float *yDev, cudaStream_t stream) {
            const uint32_t tilesN = lin.n / 16u;
            const uint32_t tileU16 = lin.tileU16;
            const uint32_t groups = 8u / static_cast<uint32_t>(split);
            uint32_t use = (tt == 4) ? 4u : (tt == 2) ? 2u : (tt == 1) ? 1u : 0u;
            if (use == 0u) {
                // Auto: depth costs blocks, so take 4 while two blocks per SM
                // (144 on this card) still fit, else 2, else 1.
                use = 1u;
                for (uint32_t cand : {4u, 2u}) {
                    const uint32_t per = groups * cand;
                    if (tilesN % per != 0) continue;
                    if (tilesN / per < 144u) continue;
                    use = cand;
                    break;
                }
            }
            const uint32_t per = groups * use;
            if (per == 0u || tilesN % per != 0) return false;
            const uint32_t grid = tilesN / per;
            if (grid == 0u) return false;
#define LAUNCH_V3(SP, TT)                                                        \
    do {                                                                         \
        if (gemvLut8On() && lin.K <= 3u) {                                        \
            exl3GemvV3<CB, SP, TT, 1><<<grid, 256, 0, stream>>>(                  \
                    reinterpret_cast<const uint16_t *>(lin.trellis), lin.suh,     \
                    lin.k, lin.K, tilesN, tileU16, xDev, yDev);                   \
        } else {                                                                 \
            exl3GemvV3<CB, SP, TT, 0><<<grid, 256, 0, stream>>>(                  \
                    reinterpret_cast<const uint16_t *>(lin.trellis), lin.suh,     \
                    lin.k, lin.K, tilesN, tileU16, xDev, yDev);                   \
        }                                                                        \
    } while (0)
            if (use == 4u) {
                switch (split) {
                    case 8: LAUNCH_V3(8, 4); break;
                    case 4: LAUNCH_V3(4, 4); break;
                    case 2: LAUNCH_V3(2, 4); break;
                    default: LAUNCH_V3(1, 4); break;
                }
            } else if (use == 2u) {
                switch (split) {
                    case 8: LAUNCH_V3(8, 2); break;
                    case 4: LAUNCH_V3(4, 2); break;
                    case 2: LAUNCH_V3(2, 2); break;
                    default: LAUNCH_V3(1, 2); break;
                }
            } else {
                switch (split) {
                    case 8: LAUNCH_V3(8, 1); break;
                    case 4: LAUNCH_V3(4, 1); break;
                    case 2: LAUNCH_V3(2, 1); break;
                    default: LAUNCH_V3(1, 1); break;
                }
            }
#undef LAUNCH_V3
            return true;
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

    bool uploadFusedLinearDevice(const void *dTrellis, const float *dSuh,
                                 const float *dSvh, uint32_t k, uint32_t n,
                                 uint32_t bitrate10, int32_t codebook,
                                 Exl3FusedLinear &out, std::string &err) {
        if (!initOnce()) {
            err = g_initErr;
            return false;
        }
        if (dTrellis == nullptr || k == 0 || n == 0) {
            err = "EXL3 fused alias: empty linear";
            return false;
        }
        if ((k & 127u) != 0 || (n & 127u) != 0) {
            err = "EXL3 fused alias: k/n must be 128-divisible";
            return false;
        }
        if ((bitrate10 % 10u) != 0u) {
            err = "EXL3 fused alias: half-integer bitrate unsupported";
            return false;
        }
        const uint32_t K = bitrate10 / 10u;
        if (K == 0 || K > 8) {
            err = "EXL3 fused alias: bitrate out of range";
            return false;
        }
        out = Exl3FusedLinear{};
        out.trellis = dTrellis;
        out.suh = dSuh;
        out.svh = dSvh;
        out.k = k;
        out.n = n;
        out.K = K;
        out.codebook = codebook;
        out.frac = false;
        out.tileU16 = 16u * K;
        // Caller keeps ownership: releaseFusedLinear() must not free these.
        out.ownsBuffers = false;
        registerFusedLinear(out);
        return true;
    }

    namespace {
        // True when `p` is a device allocation.  A safetensors mmap pointer
        // (pageable host memory) makes cudaPointerGetAttributes fail and is
        // reported as NOT device memory, which is exactly the signal the
        // prefer-alias path needs: only alias what the GPU already owns.
        bool isDeviceMemory(const void *p) {
            if (p == nullptr) return false;
            cudaPointerAttributes attr{};
            if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
                // Consume the sticky error so a host pointer cannot abort a
                // later unrelated CUDA call.
                (void) cudaGetLastError();
                return false;
            }
            return attr.type == cudaMemoryTypeDevice;
        }
    }// namespace

    bool uploadFusedLinearPreferDevice(const Exl3HostLinear &lin,
                                       Exl3FusedLinear &out, bool &aliased,
                                       std::string &err) {
        aliased = false;
        // Alias only when EVERY byte the fused layer reads is already on the
        // device; otherwise the host-copy path is the only correct one.  The
        // signs are a few tens of KB, so requiring all three keeps ownership
        // a single boolean (all-or-nothing) instead of a per-pointer flag set.
        const bool trellisDev = isDeviceMemory(lin.packed);
        const bool signsDev =
                (lin.suh == nullptr || isDeviceMemory(lin.suh)) &&
                (lin.svh == nullptr || isDeviceMemory(lin.svh));
        if (trellisDev && signsDev) {
            if (!uploadFusedLinearDevice(lin.packed, lin.suh, lin.svh, lin.k,
                                         lin.n, lin.bitrate10, lin.codebook, out,
                                         err)) {
                return false;
            }
            aliased = true;
            return true;
        }
        if (trellisDev) {
            // A device trellis with host signs cannot take the copy path (a
            // HostToDevice memcpy from device memory is invalid), and it cannot
            // take the alias path without a per-pointer ownership flag.  Fail
            // loudly instead of uploading garbage.
            err = "EXL3 fused upload: device trellis with host sign vectors "
                  "must be registered via uploadFusedLinearDevice";
            return false;
        }
        return uploadFusedLinear(lin, out, err);
    }

    void releaseFusedLinear(Exl3FusedLinear &lin) {
        if (lin.trellis != nullptr) unregisterFusedLinear(lin.trellis);
        // Free only what uploadFusedLinear() allocated.  A linear registered by
        // uploadFusedLinearDevice() aliases caller-owned device memory; freeing
        // it here would be a double free against the owning per-layer buffer.
        if (lin.ownsBuffers) {
            if (lin.trellis != nullptr) cudaFree(const_cast<void *>(lin.trellis));
            if (lin.suh != nullptr) cudaFree(const_cast<float *>(lin.suh));
            if (lin.svh != nullptr) cudaFree(const_cast<float *>(lin.svh));
        }
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

    // ---- Batched prefill path -------------------------------------------
    // The fused GEMV re-reads the whole packed trellis for EVERY token, which
    // makes a 512-token prefill stream ~2.7 TB (5.3 GB x 512) and caps EXL3 at
    // 5.8 tok/s.  For a batch the trellis is instead decoded ONCE per matrix
    // into F16 (row-major [n][k], the layout the cuBLAS fp16 GEMM consumes as
    // its transposed operand), after which the usual tensor-core GEMM applies.
    // Decode traffic per prefill becomes 5.3 GB read + 54 GB F16 write + 54 GB
    // GEMM read instead of 2.7 TB.
    bool materializeF16Ex(const void *trellisDev, void *dstF16,
                          cudaStream_t stream, std::string &err, bool raw);

    namespace {
        __global__ void f32ToF16K(const float *__restrict__ in, __half *__restrict__ out,
                                  size_t n2) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n2) return;
            reinterpret_cast<__half2 *>(out)[i] =
                    __floats2half2_rn(in[2 * i], in[2 * i + 1]);
        }

        float *g_matScratch = nullptr;    // [k][n] F32
        size_t g_matScratchFloats = 0;
        float *g_matScratchT = nullptr;   // [n][k] F32
        size_t g_matScratchTFloats = 0;
    }// namespace

    namespace {
        template <int WARPS, int NT, int KC, bool DECODE_FIRST, int MLOOP = 1>
        cudaError_t launchFusedRaw(const uint16_t *trellis, const Exl3FusedLinear &lin,
                                   uint32_t tilesN, const __half *aF16, float *yDev,
                                   uint32_t seqLen, cudaStream_t stream) {
            constexpr uint32_t kRowsPerBlock = WARPS * MLOOP * 16u;
            const dim3 grid(lin.n / (NT * 16u),
                            (seqLen + kRowsPerBlock - 1u) / kRowsPerBlock);
            static const uint32_t dbgMode = [] {
                const char *e = std::getenv("TINYCODER_EXL3_DBGMODE");
                return e ? (uint32_t) std::atoi(e) : 0u;
            }();
            gemmFusedRawT<WARPS, NT, KC, DECODE_FIRST, MLOOP><<<grid, WARPS * 32, 0, stream>>>(
                    trellis, lin.K, lin.codebook, tilesN, lin.tileU16, aF16, yDev,
                    seqLen, lin.k, lin.n, dbgMode);
            return cudaGetLastError();
        }

        // Native-HMMA twin of launchFusedRaw.  Any geometry the tile shape
        // cannot cover exactly is rejected here rather than silently dropping
        // columns: the caller (gemmExl3Batch) treats a failure as "disable the
        // fused path and use materialise+cuBLAS", so a shape mismatch costs
        // speed and never correctness.
        template <int WARPS, int MLOOP, int NT, int KC, bool DECODE_FIRST, int AREAD,
                  bool HACC = false>
        cudaError_t launchFusedHmma(const uint16_t *trellis, const Exl3FusedLinear &lin,
                                    const __half *aF16, float *yDev,
                                    uint32_t seqLen, cudaStream_t stream) {
            if ((lin.n % (NT * 16u)) != 0u) return cudaErrorInvalidValue;
            constexpr uint32_t kRowsPerBlock = WARPS * MLOOP * 16u;
            const dim3 grid(lin.n / (NT * 16u),
                            (seqLen + kRowsPerBlock - 1u) / kRowsPerBlock);
            static const uint32_t dbgMode = [] {
                const char *e = std::getenv("TINYCODER_EXL3_DBGMODE");
                return e ? (uint32_t) std::atoi(e) : 0u;
            }();
            // Exact geometry -> drop every inner-loop guard (see gemmFusedHmmaT).
            const bool full = (seqLen % kRowsPerBlock) == 0u &&
                              (lin.k % (KC * 16u)) == 0u;
            auto launch = [&](auto tag) {
                constexpr bool F = decltype(tag)::value;
                gemmFusedHmmaT<WARPS, MLOOP, NT, KC, DECODE_FIRST, AREAD, F, HACC>
                        <<<grid, WARPS * 32, 0, stream>>>(
                                trellis, lin.K, lin.codebook, lin.n / 16u, lin.tileU16,
                                aF16, yDev, seqLen, lin.k, lin.n, dbgMode);
                return cudaGetLastError();
            };
            if (full) return launch(std::integral_constant<bool, true>{});
            return launch(std::integral_constant<bool, false>{});
        }
    }// namespace

    bool gemmFusedRawLaunch(const void *trellisDev, const void *aF16Dev, float *yDev,
                            uint32_t seqLen, cudaStream_t stream, std::string &err) {
        Exl3FusedLinear lin;
        {
            std::lock_guard<std::mutex> lk(fusedRegistry().mx);
            auto it = fusedRegistry().map.find(trellisDev);
            if (it == fusedRegistry().map.end()) {
                err = "EXL3 fused gemm: unregistered linear";
                return false;
            }
            lin = it->second;
        }
        if (lin.frac || lin.K > 3u || (lin.n % 16u) != 0 || (lin.k % 16u) != 0) {
            err = "EXL3 fused gemm: unsupported geometry";
            return false;
        }
        const uint32_t tilesN = lin.n / 16u;
        const uint16_t *trellis = static_cast<const uint16_t *>(trellisDev);
        const __half *aF16 = static_cast<const __half *>(aF16Dev);
        // WARPS = row tiles per block (16 rows each), NT = n tiles per warp (one
        // accumulator each, 16 columns), KC = k tiles per shared-memory chunk,
        // suffix a/b = decode-next-chunk-first vs MMA-first inside a pipeline stage.
        // TINYCODER_EXL3_TILE selects an alternative shape for measurement; the
        // default is the fastest measured configuration.
        static const char *cfgEnv = std::getenv("TINYCODER_EXL3_TILE");
        const std::string cfg = cfgEnv != nullptr ? std::string(cfgEnv) : std::string();
        cudaError_t le = cudaSuccess;
#define EXL3_FUSED_TRY(NAME, W, NT, KC, DF) EXL3_FUSED_TRY_(NAME, W, NT, KC, DF, 1)
#define EXL3_FUSED_TRYM(NAME, W, NT, KC, DF, ML) EXL3_FUSED_TRY_(NAME, W, NT, KC, DF, ML)
#define EXL3_FUSED_TRY_(NAME, W, NT, KC, DF, ML)                   \
    if (le == cudaSuccess && cfg == NAME)                          \
        le = launchFusedRaw<W, NT, KC, DF, ML>(trellis, lin, tilesN, aF16, yDev, \
                                               seqLen, stream)
        EXL3_FUSED_TRY("w8n4k4a", 8, 4, 4, true);
        EXL3_FUSED_TRY("w8n4k4b", 8, 4, 4, false);
        EXL3_FUSED_TRY("w8n8k4a", 8, 8, 4, true);
        EXL3_FUSED_TRY("w8n8k4b", 8, 8, 4, false);
        EXL3_FUSED_TRY("w8n4k8a", 8, 4, 8, true);
        EXL3_FUSED_TRY("w8n4k2a", 8, 4, 2, true);
        EXL3_FUSED_TRY("w16n4k4a", 16, 4, 4, true);
        EXL3_FUSED_TRY("w16n8k4a", 16, 8, 4, true);
        EXL3_FUSED_TRYM("w8n4k4m2a", 8, 4, 4, true, 2);
        EXL3_FUSED_TRYM("w8n2k4m2a", 8, 2, 4, true, 2);
        EXL3_FUSED_TRY("w32n4k4a", 32, 4, 4, true);
        EXL3_FUSED_TRY("w4n4k4a", 4, 4, 4, true);
#undef EXL3_FUSED_TRY
        // Native mma.sync.m16n8k16 shapes (spec section 8): TINYCODER_EXL3_TILE=
        // hmma* selects these.  "hmma" is the tuned default of this family; the
        // rest are the measurement variants.  AREAD picks A from global
        // registers (0) or through shared memory + ldmatrix (1).
#define EXL3_HMMA_TRY(NAME, W, ML, NT, KC, AR)                                  \
    if (le == cudaSuccess && cfg == NAME)                                       \
        le = launchFusedHmma<W, ML, NT, KC, true, AR>(trellis, lin, aF16, yDev,  \
                                                      seqLen, stream)
        // One block covers rows(16*W*ML) x cols(16*NT).  With A read straight
        // from global, per-forward A traffic is inversely proportional to the
        // column span and trellis decode amplification inversely proportional to
        // the row span, so the shape is a trade between the two; measured
        // efficiencies (% of the same-session pure-HMMA peak, k=5120 n=10240
        // seq=512) are noted per shape.
        EXL3_HMMA_TRY("hmma", 8, 4, 2, 4, 0);   // 512 x  32   33%
        EXL3_HMMA_TRY("hmmaA", 8, 2, 2, 4, 0);  // 256 x  32   35%
        EXL3_HMMA_TRY("hmmaB", 8, 2, 4, 4, 0);  // 256 x  64   51%  <- tuned default
        EXL3_HMMA_TRY("hmmaE", 8, 1, 8, 4, 0);  // 128 x 128   45%
        EXL3_HMMA_TRY("hmmaF", 8, 2, 4, 8, 0);  // 256 x  64   44%  (KC=8)
        EXL3_HMMA_TRY("hmmaS", 8, 2, 2, 4, 1);  // 256 x  32, A via smem
        EXL3_HMMA_TRY("hmmaT", 8, 4, 2, 4, 1);  // 512 x  32, A via smem
#undef EXL3_HMMA_TRY
        // fp16-accumulate variants (2x HMMA rate on sm_75).  The fp16 partials
        // cost MLOOP*NT*2*2 extra registers, so only the shapes whose fp32
        // masters already fit well under the 128-register/2-block budget are
        // instantiated here.
#define EXL3_HMMA_TRYH(NAME, W, ML, NT, KC)                                     \
    if (le == cudaSuccess && cfg == NAME)                                       \
        le = launchFusedHmma<W, ML, NT, KC, true, 0, true>(trellis, lin, aF16,   \
                                                           yDev, seqLen, stream)
        EXL3_HMMA_TRYH("hmmaH1", 8, 1, 4, 4);   // 128 x 64, 32+16 acc regs
        EXL3_HMMA_TRYH("hmmaH2", 8, 2, 2, 4);   // 256 x 32, 32+16 acc regs
        EXL3_HMMA_TRYH("hmmaH3", 8, 1, 8, 4);   // 128 x128, 64+32 acc regs
#undef EXL3_HMMA_TRYH
        if (le == cudaSuccess && cfg.empty()) {
            // Default: the fastest measured shape, narrowed when n is too short
            // for the block's column span (n % 16 == 0 is guaranteed above).
            // The native mma.sync shapes beat their wmma predecessors on every
            // geometry measured (hmmaB 425 -> 504 pp512 after the exact-geometry
            // path), so they are the default of the fused family; the wmma
            // kernel stays as the narrow-n fallback that covers n % 16 == 0
            // without a 32-column block.
            if ((lin.n % 64u) == 0u)
                le = launchFusedHmma<8, 2, 4, 4, true, 0>(trellis, lin, aF16, yDev,
                                                          seqLen, stream);
            else if ((lin.n % 32u) == 0u)
                le = launchFusedHmma<8, 2, 2, 4, true, 0>(trellis, lin, aF16, yDev,
                                                          seqLen, stream);
            else
                le = launchFusedRaw<8, 1, 4, true>(trellis, lin, tilesN, aF16, yDev,
                                                   seqLen, stream);
        }
        if (le != cudaSuccess) {
            err = std::string("EXL3 fused gemm: ") + cudaGetErrorString(le);
            return false;
        }
        return true;
    }

    bool materializeF16(const void *trellisDev, void *dstF16, cudaStream_t stream,
                        std::string &err) {
        return materializeF16Ex(trellisDev, dstF16, stream, err, false);
    }

    bool rawFastPath(const void *trellisDev) {
        Exl3FusedLinear lin;
        {
            std::lock_guard<std::mutex> lk(fusedRegistry().mx);
            auto it = fusedRegistry().map.find(trellisDev);
            if (it == fusedRegistry().map.end()) return false;
            lin = it->second;
        }
        return !lin.frac && lin.K <= 3u;
    }

    bool materializeRawF16(const void *trellisDev, void *dstF16,
                           cudaStream_t stream, std::string &err) {
        return materializeF16Ex(trellisDev, dstF16, stream, err, true);
    }

    bool materializeF16Ex(const void *trellisDev, void *dstF16, cudaStream_t stream,
                          std::string &err, bool raw) {
        if (!initOnce()) {
            err = g_initErr;
            return false;
        }
        Exl3FusedLinear lin;
        {
            std::lock_guard<std::mutex> lk(fusedRegistry().mx);
            auto it = fusedRegistry().map.find(trellisDev);
            if (it == fusedRegistry().map.end()) {
                err = "EXL3 materialize: unregistered linear";
                return false;
            }
            lin = it->second;
        }
        if ((lin.k & 127u) != 0 || (lin.n & 127u) != 0) {
            err = "EXL3 materialize: k/n must be 128-divisible";
            return false;
        }
        const size_t needF = static_cast<size_t>(lin.k) * lin.n;
        if (g_matScratchFloats < needF) {
            if (g_matScratch) cudaFree(g_matScratch);
            g_matScratch = nullptr;
            g_matScratchFloats = 0;
            if (cudaMalloc(&g_matScratch, needF * sizeof(float)) != cudaSuccess) {
                err = "EXL3 materialize: scratch alloc failed";
                return false;
            }
            g_matScratchFloats = needF;
        }
        if (g_matScratchTFloats < needF) {
            if (g_matScratchT) cudaFree(g_matScratchT);
            g_matScratchT = nullptr;
            g_matScratchTFloats = 0;
            if (cudaMalloc(&g_matScratchT, needF * sizeof(float)) != cudaSuccess) {
                err = "EXL3 materialize: transpose alloc failed";
                return false;
            }
            g_matScratchTFloats = needF;
        }
        const uint32_t tilesN = lin.n / 16u;
        const uint32_t nTiles = (lin.k / 16u) * tilesN;
        const uint16_t *trellis =
                static_cast<const uint16_t *>(const_cast<void *>(trellisDev));
        // raw: skip BOTH Hadamard passes over the weight; the caller applies
        // Hk to the activations and Hn to the output instead.
        if (raw && !lin.frac && lin.K <= 3u) {
            // Fused path: trellis -> fp16 directly, no F32 staging.
            // TILES_PER_WARP stays 1: batching 4 adjacent n-tiles per warp
            // (for 128-byte store runs) measured 283.2 ms against 114.2 ms for
            // one tile per warp -- the extra live fragments spill.
            decodeTilesIntNativeH<1><<<(nTiles + 7u) / 8u, 256, 0, stream>>>(
                    trellis, lin.K, lin.codebook, tilesN, lin.tileU16,
                    static_cast<__half *>(dstF16), lin.n, nTiles);
            const cudaError_t eh = cudaGetLastError();
            if (eh != cudaSuccess) {
                err = std::string("EXL3 materialize raw f16: ") + cudaGetErrorString(eh);
                return false;
            }
            return true;
        }
        if (lin.frac) {
            decodeTilesFrac<<<nTiles, 256, 0, stream>>>(
                    trellis, lin.K, tilesN, lin.tileU16, g_matScratch, lin.n);
        } else if (lin.K <= 3u) {
            decodeTilesIntNative<<<nTiles, 32, 0, stream>>>(
                    trellis, lin.K, lin.codebook, tilesN, lin.tileU16, g_matScratch, lin.n);
        } else {
            decodeTilesInt<<<nTiles, 256, 0, stream>>>(
                    trellis, lin.K, lin.codebook, tilesN, lin.tileU16,
                    g_matScratch, lin.n);
        }
        if (!raw) {
            leftH128<<<dim3(lin.k / 128u, (lin.n + 7u) / 8u), 128, 0, stream>>>(
                    lin.suh, g_matScratch, lin.k, lin.n);
            rightH128<<<dim3(lin.k, (lin.n + 127u) / 128u), 128, 0, stream>>>(
                    lin.svh, g_matScratch, lin.k, lin.n);
        }
        // No transpose: the decoded matrix is [k][n] row-major, which cuBLAS
        // reads directly as an (n x k) column-major operand with lda = n, so
        // CUBLAS_OP_N consumes it as-is.  The old path materialised a full
        // [n][k] F32 copy first, costing two extra 105 MB passes per matrix.
        const size_t npairs = needF / 2u;
        f32ToF16K<<<(npairs + 255u) / 256u, 256, 0, stream>>>(
                g_matScratch, static_cast<__half *>(dstF16), npairs);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            err = std::string("EXL3 materialize: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    bool linearSides(const void *trellisDev, const float *&suh, const float *&svh) {
        std::lock_guard<std::mutex> lk(fusedRegistry().mx);
        auto it = fusedRegistry().map.find(trellisDev);
        if (it == fusedRegistry().map.end()) return false;
        suh = it->second.suh;
        svh = it->second.svh;
        return true;
    }

    bool linearGeometry(const void *trellisDev, uint32_t &k, uint32_t &n) {
        std::lock_guard<std::mutex> lk(fusedRegistry().mx);
        auto it = fusedRegistry().map.find(trellisDev);
        if (it == fusedRegistry().map.end()) return false;
        k = it->second.k;
        n = it->second.n;
        return true;
    }

        // Diagnostic only (TINYCODER_EXL3_GEMV_COUNT=1): counts how many fused
        // decode GEMV calls the model actually issues and how many weight bytes
        // they move, so the kernel's share of decode time is measurable.
        struct GemvCounter {
            long long calls = 0;
            long long bytes = 0;
            ~GemvCounter() {
                if (std::getenv("TINYCODER_EXL3_GEMV_COUNT")) {
                    std::fprintf(stderr, "[exl3 gemv] calls=%lld bytes=%lld (%.2f MB)\n",
                                 calls, bytes, static_cast<double>(bytes) / 1e6);
                }
            }
        };
        inline GemvCounter &gemvCounter() {
            static GemvCounter c;
            return c;
        }

        // Diagnostic only (TINYCODER_EXL3_GEMV_TIME=1): GPU time spent inside
        // gemvFused, split into the trellis matvec and the out transform, per
        // (k,n) shape, plus the idle gap between consecutive gemv calls.  The
        // per-shape split is what says WHICH projections are below the
        // bandwidth the kernel reaches in isolation.
        struct GemvShape {
            uint32_t k = 0, n = 0, K = 0;
            long long calls = 0;
            double mvMs = 0.0, xfMs = 0.0;
        };
        struct GemvTimer {
            struct S {
                int shape;
                cudaEvent_t a, m, b;
            };
            std::vector<S> ev;
            std::vector<GemvShape> shapes;
            int shapeOf(uint32_t k, uint32_t n, uint32_t K) {
                for (size_t i = 0; i < shapes.size(); ++i)
                    if (shapes[i].k == k && shapes[i].n == n)
                        return static_cast<int>(i);
                GemvShape s;
                s.k = k;
                s.n = n;
                s.K = K;
                shapes.push_back(s);
                return static_cast<int>(shapes.size() - 1);
            }
            ~GemvTimer() {
                if (!std::getenv("TINYCODER_EXL3_GEMV_TIME")) return;
                double mv = 0.0, xf = 0.0, gap = 0.0;
                size_t gapN = 0;
                for (size_t i = 0; i < ev.size(); ++i) {
                    float d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                    cudaEventElapsedTime(&d1, ev[i].a, ev[i].m);
                    cudaEventElapsedTime(&d2, ev[i].m, ev[i].b);
                    GemvShape &s = shapes[static_cast<size_t>(ev[i].shape)];
                    s.calls += 1;
                    s.mvMs += d1;
                    s.xfMs += d2;
                    mv += d1;
                    xf += d2;
                    if (i + 1 < ev.size()) {
                        cudaEventElapsedTime(&d3, ev[i].b, ev[i + 1].a);
                        if (d3 > 0.0f && d3 < 5.0f) {
                            gap += d3;
                            ++gapN;
                        }
                    }
                }
                std::fprintf(stderr,
                             "[exl3 gemv] gpu_ms=%.1f (matvec %.1f + xform %.1f) "
                             "calls=%zu avg_us=%.1f inter-call gap=%.1f ms "
                             "(%zu pairs)\n",
                             mv + xf, mv, xf, ev.size(),
                             1000.0 * (mv + xf) / static_cast<double>(ev.size() + 1),
                             gap, gapN);
                if (std::getenv("TINYCODER_EXL3_GEMV_TABLE")) {
                    std::fprintf(stderr,
                                 "[exl3 gemv] %-11s %-8s %8s %9s %9s %9s %8s\n",
                                 "shape", "K", "calls", "mv_ms", "xf_ms",
                                 "GB/s", "xf%");
                    for (const GemvShape &s : shapes) {
                        const double bytes =
                                0.125 * s.k * static_cast<double>(s.n) * s.K * s.calls;
                        const double ms = s.mvMs + s.xfMs;
                        std::fprintf(stderr,
                                     "[exl3 gemv] %5ux%-5u %-8u %8lld %9.2f %9.2f "
                                     "%9.1f %7.1f\n",
                                     s.k, s.n, s.K, s.calls, s.mvMs, s.xfMs,
                                     ms > 0.0 ? bytes / (ms * 1e6) : 0.0,
                                     ms > 0.0 ? 100.0 * s.xfMs / ms : 0.0);
                    }
                }
            }
        };
        inline GemvTimer &gemvTimer() {
            static GemvTimer t;
            return t;
        }
        inline bool gemvTimeOn() {
            static const bool on = std::getenv("TINYCODER_EXL3_GEMV_TIME") != nullptr;
            return on;
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
        {
            GemvCounter &gc = gemvCounter();
            gc.calls += 1;
            gc.bytes += static_cast<long long>(lin.k) * lin.n * lin.K / 8;
            if (std::getenv("TINYCODER_EXL3_GEMV_COUNT") && gc.calls <= 4) {
                std::fprintf(stderr, "[exl3 gemv] call %lld: k=%u n=%u K=%u\n",
                             gc.calls, lin.k, lin.n, lin.K);
            }
        }

        // TINYCODER_EXL3_GEMV selects the decode matvec.  Default is V2: it
        // reads each trellis tile as 96 contiguous bytes instead of a
        // per-column gather, keeps all 32 lanes on distinct columns, and
        // Hadamards the activation once per block instead of once per warp.
        // Measured 5.50 -> 14.28 tok/s at SPLIT=4 and 14.84 at SPLIT=8 on the
        // 1.60bpw checkpoint, with per-projection output bit-identical to the
        // old path.  V2 covers K<=3; K=4 (2.0bpw and above) takes the old
        // matvec.  TINYCODER_EXL3_GEMV=1 restores the old path byte-for-byte.
        static const int gemvVer = [] {
            const char *e = std::getenv("TINYCODER_EXL3_GEMV");
            return e ? std::atoi(e) : 3;
        }();
        const bool timing = gemvTimeOn();
        if (timing) {
            cudaEvent_t a = nullptr, m = nullptr, b = nullptr;
            cudaEventCreate(&a);
            cudaEventCreate(&m);
            cudaEventCreate(&b);
            cudaEventRecord(a, stream);
            gemvTimer().ev.push_back(
                    GemvTimer::S{gemvTimer().shapeOf(lin.k, lin.n, lin.K), a, m, b});
        }
        if (gemvVer >= 3 && lin.K >= 1 && lin.K <= 3) {
            const int vsplit = gemvSplitEnv();
            const int tt = gemvTtEnv();
            bool ok = false;
            // K=1 has its own kernel: a 1-bit code is a two-symbol alphabet,
            // so the general decode is answering a yes/no question with nine
            // instructions per weight (see exl3GemvK1).
            if (lin.K == 1 && gemvK1On()) {
                const int k1tt = gemvK1TtEnv();
                if (lin.codebook == 2) {
                    ok = launchGemvK1<2>(vsplit, k1tt, lin, xDev, yDev, stream);
                } else if (lin.codebook == 1) {
                    ok = launchGemvK1<1>(vsplit, k1tt, lin, xDev, yDev, stream);
                } else {
                    ok = launchGemvK1<0>(vsplit, k1tt, lin, xDev, yDev, stream);
                }
            }
            if (!ok && lin.codebook == 2) {
                ok = launchGemvV3<2>(vsplit, tt, lin, xDev, yDev, stream);
            } else if (!ok && lin.codebook == 1) {
                ok = launchGemvV3<1>(vsplit, tt, lin, xDev, yDev, stream);
            } else if (!ok) {
                ok = launchGemvV3<0>(vsplit, tt, lin, xDev, yDev, stream);
            }
            if (!ok) {
                if (lin.codebook == 2) {
                    launchGemvV2<2>(vsplit, lin, xDev, yDev, stream);
                } else if (lin.codebook == 1) {
                    launchGemvV2<1>(vsplit, lin, xDev, yDev, stream);
                } else {
                    launchGemvV2<0>(vsplit, lin, xDev, yDev, stream);
                }
            }
        } else if (gemvVer == 2 && lin.K >= 1 && lin.K <= 3) {
            const int vsplit = gemvSplitEnv();
            if (lin.codebook == 2) {
                launchGemvV2<2>(vsplit, lin, xDev, yDev, stream);
            } else if (lin.codebook == 1) {
                launchGemvV2<1>(vsplit, lin, xDev, yDev, stream);
            } else {
                launchGemvV2<0>(vsplit, lin, xDev, yDev, stream);
            }
        } else {
            int split = 8;
            while (split > 1 &&
                   (lin.k % (128u * static_cast<uint32_t>(split))) != 0) {
                split >>= 1;
            }
            if (lin.codebook == 2) {
                launchFusedSplit<2>(split, lin, xDev, yDev, stream);
            } else if (lin.codebook == 1) {
                launchFusedSplit<1>(split, lin, xDev, yDev, stream);
            } else {
                launchFusedSplit<0>(split, lin, xDev, yDev, stream);
            }
        }
        if (timing) {
            cudaEventRecord(gemvTimer().ev.back().m, stream);
        }
        exl3FusedOutTransform<<<lin.n / 128u, 128, 0, stream>>>(
                yDev, lin.svh, yDev);
        if (timing) {
            cudaEventRecord(gemvTimer().ev.back().b, stream);
        }
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
