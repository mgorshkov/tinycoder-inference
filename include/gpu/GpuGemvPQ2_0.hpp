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

// GpuGemvPQ2_0.hpp -- Prism ternary (PQ2_0) GEMV, incl. the 32-wide-block GEMV.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// ------------------------------------------------------------------
// SPLIT-K GEMV (2026-09-29, decode small-row occupancy lever):
//
// The warp-per-row kQGemv launches nBlocks = ceil(rows/8) blocks of
// 8 warps.  On the SMALL decode matrices (attnO / gate+up / down at
// rows=1536) this leaves only ~2.8 blocks/SM resident (the kernel is
// register-limited at ~90 regs/thread), i.e. ~22 warps/SM streaming
// their rows — well under the 32-warp/SM hardware cap, and the
// measured per-matrix throughput (down: 74 us/layer for 5.9 MB =
// ~80 GB/s) is far under the ~550 GB/s DRAM peak.
//
// THIS kernel raises the concurrent warp count WITHOUT changing any
// per-warp math: each row is handled by KN chunk-warps (blockDim.y =
// KN, one block per row), and chunk w owns the CONTIGUOUS block
// sub-range  [(w*bpr)/KN, ((w+1)*bpr)/KN).  Each chunk warp runs the
// byte-identical kQGemv float chain (4-way a0..a3 block batching with
// in-order fold, remainder direct-fold, 5-shuffle warp reduce) over
// its sub-range; the KN per-chunk partials are summed in shared
// memory IN CHUNK ORDER by warp 0 lane 0.  For KN=2 on down
// (rows=1536): 1536 blocks x 2 warps = 3072 warps vs 1536 today, at
// the SAME register footprint per warp (each warp does half the
// blocks) — more concurrent DRAM streams, more blocks/SM resident.
//
// The cross-chunk float sum (p0+p1+.. in chunk order) is a 1-ulp-class
// reordering vs the fully-serial fold, so the per-layer logits may
// differ from the serial path by ~1e-7 relative — within the
// tolerance-based parity bar that the Q2K/Q3K float GEMV paths already
// use (the exact-integer Q8_K paths are the bit-exact ones).  Measured
// KEEP decision (2026-09-29): down-only Q3_K split-K is a WIN (down
// 1.50 -> 1.40 ms/layer, ~7%, parity PASS); a Q2_K split branch was
// measured and REVERTED because splitting the gate+up (rows=8960)
// flooded the L2/x-traffic with duplicate activation reads (gate+up
// 1.77 -> 3.40 ms/layer, +92%) — Q2_K split only helps the true
// SMALL-row matrices (attnO at rows=1536) and the 8960-row gate/up
// far outweighs it.  This kernel is therefore Q3K-only (ffnDown).
template<int TYPE, int KN>
__global__ void __launch_bounds__(32 * KN) kQGemvSplitK(
        const uint8_t *__restrict__ w, const float *__restrict__ x,
        float *__restrict__ out, uint32_t rows, uint32_t cols,
        uint32_t rowBytes, uint32_t blocksPerRow) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t wId = threadIdx.y;// 0..KN-1
    const uint32_t lane = threadIdx.x;
    const uint32_t cLow = (wId * blocksPerRow) / KN;
    const uint32_t cHigh = ((wId + 1u) * blocksPerRow) / KN;
    const uint32_t nBlk = cHigh - cLow;
    if (nBlk == 0) {
        // Chunk with no blocks (tail when bpr < KN): contribute 0.
        __syncthreads();
        if (wId == 0u && lane == 0u) out[row] = 0.0f;
        return;
    }
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes +
                        static_cast<uint64_t>(cLow) * kQ3K_BYTES;
    float acc = 0.0f;
    if (TYPE == kTypeQ3K) {
        const uint32_t sub = lane / 16u;
        uint32_t b = 0;
        for (; b + 4u <= nBlk; b += 4u) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 32
            for (uint32_t bb = 0; bb < 4; ++bb) {
                const uint8_t *crp =
                        rp + static_cast<uint64_t>(bb) * kQ3K_BYTES;
                __half d = *reinterpret_cast<const __half *>(crp + 108);
                float df = __half2float(d);
                const uint8_t *hm = crp;
                const uint8_t *q = crp + 32;
                const uint8_t *scales = crp + 96;
                uint32_t aux[4];
                std::memcpy(&aux[0], scales, 12);
                uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) |
                         (((tmp >> 4) & 0x03030303u) << 4);
                aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) |
                         (((tmp >> 6) & 0x03030303u) << 4);
                aux[0] = (aux[0] & 0x0f0f0f0fu) |
                         (((tmp >> 0) & 0x03030303u) << 4);
                aux[1] = (aux[1] & 0x0f0f0f0fu) |
                         (((tmp >> 2) & 0x03030303u) << 4);
                int8_t sc16[16];
                std::memcpy(sc16, aux, 16);
                const uint8_t hmb = hm[lane];
                float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                        : (bb == 2u)       ? a2
                                                           : a3;
#pragma unroll
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                    for (uint32_t jj = 0; jj < 4; ++jj) {
                        float dl = df * static_cast<float>(
                                                sc16[half * 8u + jj * 2u + sub] - 32);
                        const uint32_t maskBit = 1u << (half * 4u + jj);
                        float qv = static_cast<float>(static_cast<int8_t>(
                                ((qb >> (jj * 2u)) & 3) -
                                ((hmb & maskBit) ? 0 : 4)));
                        const uint32_t cidx =
                                (cLow + b + bb) * 256u + half * 128u +
                                jj * 32u + lane;
                        slot = fmaf(dl * qv,
                                    (cidx < cols) ? x[cidx] : 0.0f, slot);
                    }
                }
            }
            acc += a0;
            acc += a1;
            acc += a2;
            acc += a3;
            rp += 4u * kQ3K_BYTES;
        }
        for (; b < nBlk; ++b) {
            __half d = *reinterpret_cast<const __half *>(rp + 108);
            float df = __half2float(d);
            const uint8_t *hm = rp;
            const uint8_t *q = rp + 32;
            const uint8_t *scales = rp + 96;
            uint32_t aux[4];
            std::memcpy(&aux[0], scales, 12);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) |
                     (((tmp >> 4) & 0x03030303u) << 4);
            aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) |
                     (((tmp >> 6) & 0x03030303u) << 4);
            aux[0] = (aux[0] & 0x0f0f0f0fu) |
                     (((tmp >> 0) & 0x03030303u) << 4);
            aux[1] = (aux[1] & 0x0f0f0f0fu) |
                     (((tmp >> 2) & 0x03030303u) << 4);
            int8_t sc16[16];
            std::memcpy(sc16, aux, 16);
            const uint8_t hmb = hm[lane];
#pragma unroll
            for (uint32_t half = 0; half < 2; ++half) {
                const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    float dl = df * static_cast<float>(
                                            sc16[half * 8u + jj * 2u + sub] - 32);
                    const uint32_t maskBit = 1u << (half * 4u + jj);
                    float qv = static_cast<float>(static_cast<int8_t>(
                            ((qb >> (jj * 2u)) & 3) -
                            ((hmb & maskBit) ? 0 : 4)));
                    const uint32_t cidx =
                            (cLow + b) * 256u + half * 128u + jj * 32u + lane;
                    acc = fmaf(dl * qv, (cidx < cols) ? x[cidx] : 0.0f,
                               acc);
                }
            }
            rp += kQ3K_BYTES;
        }
    } else {
        acc = 0.0f;
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    __shared__ float sPart[KN];
    if (lane == 0) sPart[wId] = acc;
    __syncthreads();
    if (wId == 0u && lane == 0u) {
        float total = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < KN; ++k) {
            total += sPart[k];// chunk order
        }
        out[row] = total;
    }
}

// ---- 32-wide-block GEMV (Q5_0 / Q8_0 / IQ4_NL) ----
// These legacy quant formats use 32-weight blocks (22 B / 34 B / 18 B
// each) rather than the 256-wide K-quant blocks handled by kQGemv
// above. One warp handles one output row; each lane owns column
// (b*32 + lane) of a block, computes its single weight, and the warp
// reduction sums them.  blocksPerRow = ceil(cols/32); rowBytes =
// blocksPerRow * blockBytes.
template<int TYPE>
__global__ void kQGemvSmall32(const uint8_t *__restrict__ w,
                              const float *__restrict__ x,
                              float *__restrict__ out, uint32_t rows,
                              uint32_t cols, uint32_t rowBytes,
                              uint32_t blocksPerRow) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint8_t *rp =
            w + static_cast<uint64_t>(row) * rowBytes;
    float acc = 0.0f;
    constexpr uint32_t BLOCK_BYTES = (TYPE == kTypeQ5_0)
                                             ? kQ5_0_BYTES
                                     : (TYPE == kTypeQ8_0) ? kQ8_0_BYTES
                                                           : kIQ4NL_BYTES;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        const uint8_t *blk = rp + static_cast<uint64_t>(b) * BLOCK_BYTES;
        const uint32_t col = b * 32u + lane;
        if (col >= cols) break;
        __half d = *reinterpret_cast<const __half *>(blk);
        float wv;
        if (TYPE == kTypeQ8_0) {
            const int8_t q = reinterpret_cast<const int8_t *>(blk + 2)[lane];
            wv = __half2float(d) * static_cast<float>(q);
        } else if (TYPE == kTypeIQ4NL) {
            // IQ4_NL (4.5 bpw): d(fp16,2) + qs[16].  Element `lane`:
            //   j   = lane & 15
            //   nib = (lane < 16) ? qs[j] & 0x0F : qs[j] >> 4
            //   wv  = d * kvalues_iq4nl[nib]   (signed 8-bit table)
            // Matches CPU dequantizeIQ4_NLBlock / dotProductIQ4_NL.
            const uint8_t *qs = blk + 2;
            const uint32_t j = lane & 15u;
            const uint8_t nib =
                    (qs[j] >> (4u * (lane >> 4u))) & 0xFu;
            wv = __half2float(d) *
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
            wv = __half2float(d) * static_cast<float>(val);
        }
        acc = fmaf(wv, x[col], acc);
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) out[row] = acc;
}

// ==================================================================
// Prism PQ2_0 (type 142) ternary GEMV + Hadamard fold (Bonsai).
//
// Ternary-Bonsai-2-27B stores its (near-ternary) linear weights in a
// 2-bit format already folded with a per-input-axis normalized
// Sylvester-Walsh-Hadamard rotation H (block 1024) and a per-feature
// sign flip S: W_folded = W @ (H S P).  To recover the primal
// projection y = W x the ACTIVATION is transformed x' = H S P x first,
// then a plain (ternary) GEMV computes W_folded @ x' = W @ x.  The
// PQ2_0 weights therefore stay compressed in VRAM; only the small
// activation vector is rotated per token.
//
// PQ2_0 block = 34 bytes / 128 weights: d(fp16,2) + qs[uint8;32].
// Code -> value: 0 -> -1, 1 -> 0, 2 -> +1, 3 -> +2 (value = q-1).
// Element j: byte j>>2, bits (j&3)*2.  Bit-exact with the CPU
// GGMLDequantize::dequantizePQ2_0Block / dotProductPQ2_0.
// ==================================================================

// ------------------------------------------------------------------
// Prism PQ2_0 (type 142) ternary GEMV.
//
// 2026-11-17, measured in situ (this is the whole Bonsai decode path):
// the original one-warp-per-row, one-block-at-a-time version ran the
// model at 15.28 tg128 = 105 GB/s of PACKED weight (6.85 GB/token), and
// every projection sat at 82-120 GB/s no matter its shape.  The weights
// were already 2 bits; the kernel was latency-bound, because a warp had
// exactly one 34-byte weight fetch in flight and the loop is serial.
//
// Fix, in two parts:
//   * PQ2_GEMV_UNROLL blocks are fetched into registers before any FMA
//     (8 independent 34 B fetches per warp in flight instead of 1) and a
//     block's four activation floats load as one float4 instead of four
//     scalar loads: 15.28 -> 27.3 tg128 in situ (1.78x).
//   * PQ2_ROWS rows per warp: all R rows share the one activation load.
//     The activation is fp32 (4 B) against 0.266 B of weight per column,
//     so a warp re-reads 15x the weight bytes it consumes; sharing x
//     across R rows divides that by R.  TINYCODER_PQ2_R selects the
//     (rows-per-warp, unroll) geometry for A/B: 1=1x8, 2=2x4, 4=4x2,
//     8=2x8 (default), 16=1x16, 32=4x4, 64=2x16.  Measured on this card
//     at n-gen 32: 1x8 33.7, 2x4 35.3, 4x2 32.7, 2x8 39.0, 1x16 25.6,
//     4x4 39.4, 2x16 31.0 tok/s -- in-flight bytes dominate up to ~550 B
//     per warp, past which register pressure (2x16) loses more than the
//     extra prefetch wins.
// Accumulation order per output is unchanged (b ascending, 4 FMAs in
// column order within a block), so the result is bit-identical to the
// pre-change kernel and to the CPU dotProductPQ2_0.
// ------------------------------------------------------------------
template<uint32_t R, uint32_t U>
__global__ void __launch_bounds__(128) kQGemvPQ2_0T(
        const uint8_t *__restrict__ w, const float *__restrict__ x,
        float *__restrict__ out, uint32_t rows, uint32_t cols,
        uint32_t rowBytes, uint32_t blocksPerRow) {
    const uint32_t warp = blockIdx.x * 4u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = warp * R;
    if (row0 >= rows) return;
    const uint32_t nr = (rows - row0 < R) ? (rows - row0) : R;
    const uint8_t *rp[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        rp[r] = w + static_cast<uint64_t>(row0 + (r < nr ? r : nr - 1u)) *
                            rowBytes;
    }
    // A float4 activation load needs a 16-byte aligned pointer and four
    // in-range columns; both tests are uniform across the warp.
    const bool vec = ((reinterpret_cast<uintptr_t>(x) & 15u) == 0u);
    float acc[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) acc[r] = 0.0f;
    uint32_t b = 0;
    for (; b + U <= blocksPerRow; b += U) {
        uint8_t qb[R][U];
        float ds[R][U];
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) {
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                const uint8_t *blk =
                        rp[r] + static_cast<uint64_t>(b + u) * kPQ2_0_BYTES;
                qb[r][u] = blk[2u + lane];
                ds[r][u] = __half2float(
                        *reinterpret_cast<const __half *>(blk));
            }
        }
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) {
            const uint32_t base = (b + u) * 128u + lane * 4u;
            float x0, x1, x2, x3;
            if (vec && base + 3u < cols) {
                const float4 v =
                        *reinterpret_cast<const float4 *>(x + base);
                x0 = v.x;
                x1 = v.y;
                x2 = v.z;
                x3 = v.w;
            } else {
                x0 = (base + 0u < cols) ? x[base + 0u] : 0.0f;
                x1 = (base + 1u < cols) ? x[base + 1u] : 0.0f;
                x2 = (base + 2u < cols) ? x[base + 2u] : 0.0f;
                x3 = (base + 3u < cols) ? x[base + 3u] : 0.0f;
            }
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                if (r >= nr) break;
                const uint32_t q = qb[r][u];
                const float d = ds[r][u];
                float a = acc[r];
                a = fmaf(d * (static_cast<float>((q >> 0u) & 3u) - 1.0f),
                         x0, a);
                a = fmaf(d * (static_cast<float>((q >> 2u) & 3u) - 1.0f),
                         x1, a);
                a = fmaf(d * (static_cast<float>((q >> 4u) & 3u) - 1.0f),
                         x2, a);
                a = fmaf(d * (static_cast<float>((q >> 6u) & 3u) - 1.0f),
                         x3, a);
                acc[r] = a;
            }
        }
    }
    for (; b < blocksPerRow; ++b) {
        const uint32_t base = b * 128u + lane * 4u;
        const uint32_t c0 = base + 0u, c1 = base + 1u;
        const uint32_t c2 = base + 2u, c3 = base + 3u;
        const float x0 = (c0 < cols) ? x[c0] : 0.0f;
        const float x1 = (c1 < cols) ? x[c1] : 0.0f;
        const float x2 = (c2 < cols) ? x[c2] : 0.0f;
        const float x3 = (c3 < cols) ? x[c3] : 0.0f;
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) {
            if (r >= nr) break;
            const uint8_t *blk =
                    rp[r] + static_cast<uint64_t>(b) * kPQ2_0_BYTES;
            const float d =
                    __half2float(*reinterpret_cast<const __half *>(blk));
            const uint8_t q = blk[2u + lane];
            float a = acc[r];
            a = fmaf(d * (static_cast<float>((q >> 0u) & 3u) - 1.0f), x0, a);
            a = fmaf(d * (static_cast<float>((q >> 2u) & 3u) - 1.0f), x1, a);
            a = fmaf(d * (static_cast<float>((q >> 4u) & 3u) - 1.0f), x2, a);
            a = fmaf(d * (static_cast<float>((q >> 6u) & 3u) - 1.0f), x3, a);
            acc[r] = a;
        }
    }
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        float a = acc[r];
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            a += __shfl_xor_sync(0xffffffffu, a, off);
        }
        if (lane == 0 && row0 + r < rows) out[row0 + r] = a;
    }
}

// Host-side selection of the rows-per-warp geometry (A/B switch; set
// TINYCODER_PQ2_R=1 for the previous one-row-per-warp kernel).
inline uint32_t pq2GemvGeom() {
    static const uint32_t r = [] {
        const char *e = std::getenv("TINYCODER_PQ2_R");
        const long v = e ? std::strtol(e, nullptr, 10) : 8;
        return (v == 1 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64)
                       ? static_cast<uint32_t>(v)
                       : static_cast<uint32_t>(8);
    }();
    return r;
}

inline void launchQGemvPQ2_0(const uint8_t *w, const float *x, float *out,
                             uint32_t rows, uint32_t cols,
                             uint32_t rowBytes, uint32_t blocksPerRow) {
    // NOTE: every launch below must name g_stream explicitly.  Without
    // it the kernel goes to the LEGACY DEFAULT stream, and g_stream was
    // created with cudaStreamNonBlocking, so there is no implicit
    // ordering between them: the GEMV would race with the fold that
    // produces its activation and with the consumer of `out` (this was
    // the state of the tree when the 36.4 tok/s baseline was measured;
    // it also hid the GEMV from every event-based profile).
    switch (pq2GemvGeom()) {
        case 1u:
            kQGemvPQ2_0T<1u, 8u><<<(rows + 3u) / 4u, 128, 0, g_stream>>>(
                    w, x, out, rows, cols, rowBytes, blocksPerRow);
            break;
        case 4u:
            kQGemvPQ2_0T<4u, 2u><<<(rows + 15u) / 16u, 128, 0, g_stream>>>(
                    w, x, out, rows, cols, rowBytes, blocksPerRow);
            break;
        case 8u:
            kQGemvPQ2_0T<2u, 8u><<<(rows + 7u) / 8u, 128, 0, g_stream>>>(
                    w, x, out, rows, cols, rowBytes, blocksPerRow);
            break;
        case 16u:
            kQGemvPQ2_0T<1u, 16u><<<(rows + 3u) / 4u, 128, 0, g_stream>>>(
                    w, x, out, rows, cols, rowBytes, blocksPerRow);
            break;
        case 32u:
            kQGemvPQ2_0T<4u, 4u><<<(rows + 15u) / 16u, 128, 0, g_stream>>>(
                    w, x, out, rows, cols, rowBytes, blocksPerRow);
            break;
        case 64u:
            kQGemvPQ2_0T<2u, 16u><<<(rows + 7u) / 8u, 128, 0, g_stream>>>(
                    w, x, out, rows, cols, rowBytes, blocksPerRow);
            break;
        default:
            kQGemvPQ2_0T<2u, 4u><<<(rows + 7u) / 8u, 128, 0, g_stream>>>(
                    w, x, out, rows, cols, rowBytes, blocksPerRow);
            break;
    }
}

// GDN value-head permutation index (matches prism::permuteGdnValueGrouped):
// destination grouped feature g = d + r*headV + k*headV*rep maps to the
// tiled source feature hv*headV + d with hv = k + r*nKHeads.
// GDN value-head permutation index (inverse of prism::
// permuteGdnValueGrouped): destination grouped feature
//   g = d + r*headV + k*headV*rep
// maps to the tiled source feature hv*headV + d with
//   hv = k + r*nKHeads.
// Decompose g: inner = g % headV; rest = g / headV = r + k*rep, so
// r = rest % rep and k = rest / rep; then recompose the tiled head with
// nKHeads (= nk).  rep and nk differ (3 vs 16 for Bonsai), so both are
// required.
__device__ __forceinline__ uint32_t prismGdnSrcIndex(uint32_t g,
                                                     uint32_t headV,
                                                     uint32_t nk,
                                                     uint32_t rep) {
    const uint32_t inner = g % headV;
    const uint32_t rest = g / headV;// r + k*rep
    const uint32_t r = rest % rep;
    const uint32_t k = rest / rep;
    return (k + r * nk) * headV + inner;
}

// Normalized blockwise Sylvester-Walsh-Hadamard rotation of ONE
// kPrismBlock-wide group held in shared memory (in place).
__device__ __forceinline__ void prismFwhtShared(float *tile, uint32_t n,
                                                uint32_t tid,
                                                uint32_t nThreads) {
    for (uint32_t hs = 1; hs < n; hs <<= 1) {
        for (uint32_t p = tid; p < n / 2u; p += nThreads) {
            const uint32_t i = (p / hs) * (hs << 1) + (p % hs);
            const float a = tile[i], b = tile[i + hs];
            tile[i] = a + b;
            tile[i + hs] = a - b;
        }
        __syncthreads();
    }
    const float scale = rsqrtf(static_cast<float>(n));
    for (uint32_t i = tid; i < n; i += nThreads) tile[i] *= scale;
    __syncthreads();
}

// Forward Prism activation transform (per 1024-wide tile): optional GDN
// value-head permutation, then the per-feature sign flip, then the
// blockwise normalized Hadamard.  grid = ceil(n / kPrismBlock);
// block = 256.  `v` (primal activation) and `dst` (folded activation)
// must NOT alias.
__global__ void kPrismFold(const float *__restrict__ v,
                           float *__restrict__ dst, uint32_t n,
                           const float *__restrict__ signs,
                           uint32_t gdnHeadV, uint32_t gdnNk,
                           uint32_t gdnRep) {
    __shared__ float tile[kPrismBlock];
    const uint32_t base = blockIdx.x * kPrismBlock;
    const uint32_t tid = threadIdx.x;
    const uint32_t have = (base < n) ? ((n - base < kPrismBlock)
                                                ? (n - base)
                                                : kPrismBlock)
                                     : 0u;
    if (have == 0u) return;
    const bool perm = (gdnRep > 1u && gdnHeadV > 0u && gdnNk > 0u);
    for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x) {
        float val = 0.0f;
        if (i < have) {
            const uint32_t g = base + i;
            const uint32_t src =
                    perm ? prismGdnSrcIndex(g, gdnHeadV, gdnNk, gdnRep) : g;
            val = v[src];
            if (signs != nullptr) val *= signs[g];
        }
        tile[i] = val;
    }
    __syncthreads();
    prismFwhtShared(tile, kPrismBlock, tid, blockDim.x);
    for (uint32_t i = tid; i < have; i += blockDim.x) {
        dst[base + i] = tile[i];
    }
}

// Batched forward Prism activation transform for prefill: identical math
// to kPrismFold (sign flip, optional GDN value-head permutation, then the
// blockwise normalized Hadamard), one row per blockIdx.y, written
// straight out as fp16 so the fp16 tensor-core GEMM consumes it without a
// second pass.  grid = (ceil(n / kPrismBlock), seqLen); block = 256.
// `rowStride` is the activation row pitch in floats/halves.
__global__ void kPrismFoldToF16(const float *__restrict__ v,
                                __half *__restrict__ dst, uint32_t n,
                                const float *__restrict__ signs,
                                uint32_t gdnHeadV, uint32_t gdnNk,
                                uint32_t gdnRep, uint32_t rowStride) {
    __shared__ float tile[kPrismBlock];
    const uint32_t base = blockIdx.x * kPrismBlock;
    const uint32_t s = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t have = (base < n) ? ((n - base < kPrismBlock)
                                                ? (n - base)
                                                : kPrismBlock)
                                     : 0u;
    if (have == 0u) return;
    const float *src = v + static_cast<size_t>(s) * rowStride;
    __half *dp = dst + static_cast<size_t>(s) * rowStride;
    const bool perm = (gdnRep > 1u && gdnHeadV > 0u && gdnNk > 0u);
    for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x) {
        float val = 0.0f;
        if (i < have) {
            const uint32_t g = base + i;
            const uint32_t si =
                    perm ? prismGdnSrcIndex(g, gdnHeadV, gdnNk, gdnRep) : g;
            val = src[si];
            if (signs != nullptr) val *= signs[g];
        }
        tile[i] = val;
    }
    __syncthreads();
    prismFwhtShared(tile, kPrismBlock, tid, blockDim.x);
    for (uint32_t i = tid; i < have; i += blockDim.x) {
        dp[base + i] = __float2half(tile[i]);
    }
}

// Prism PQ2_0 q8_1 activation + dp4a decode GEMV (2026-11-17).
//
// Why: the fp32-activation GEMV above reads 4 B of activation per
// weight column (a float4 per 128-block per lane) against 0.25 B of
// packed weight, i.e. ~16 B of activation traffic per 4 weights.  That
// is what caps decode at ~233 GB/s of packed weight on this card (the
// same block of 128 weights costs 7 wavefronts of L1 traffic instead
// of 4).  llama.cpp's route -- and the route the PrismML fork takes for
// this exact format -- is to quantise the (folded) activation to int8
// once per projection and dot it against the packed 2-bit codes with
// `dp4a`, so each lane moves 4 B of activation per 16 weights.
//
// The arithmetic difference from the fp32 kernel is ONLY the int8
// quantisation of the activation: the weight codes are integers already,
// so sum((code-1)*x) is exact in int32 (|sum| <= 128*127*2 per block),
// and the per-128-block scale d2 and per-32-block scale d8 are applied
// once, in float, per block.  The per-output accumulation order (blocks
// ascending) is unchanged; what changes is that the four intra-block
// products are summed exactly in integer instead of in fp32, so the
// result differs from the fp32 path by the int8 quantisation error
// (~2^-8 relative on the activation) and not by an accumulation-order
// artefact.  TINYCODER_PQ2_Q8=0 restores the fp32-activation path.
// ==================================================================

// Fold + quantise to q8 (per 32 columns, symmetric).  Same transform as
// kPrismFold above (sign flip, optional GDN value-head permutation,
// blockwise normalized Hadamard), then
//   amax = max|x| over the 32-group, d8 = amax/127, q = round(x/d8).
// grid = ceil(n / kPrismBlock); block = 256.  qs is int8[n] and d8 is
// float[n/32]; both must be 16-byte aligned and n a multiple of 128.
// ------------------------------------------------------------------
// q8_1 activation quantiser for the PQ2_0 dp4a GEMV.
//
// The q8 layout is the PrismML fork's (ggml-cuda/quantize.cu,
// layout GGML_CUDA_Q8_1_PQ2): inside each 16-column group the quantised
// value of column e is stored at position
//     (e % 4) * 4 + (e % 16) / 4
// so that the four bytes of one 32-bit activation word hold columns
// {j, j+4, j+8, j+12}.  That is exactly the column order that
// `(code_word >> 2k) & 0x03030303` selects out of the packed 2-bit
// weights, so a lane can dot 16 weights with four `dp4a` and no
// unpacking table and no cross-lane shuffles (see kQGemvPQ2_0Q8F).
//
// Per 32-column group we also store the exact integer sum of the
// quantised values (the fork packs it into the half2's high half; an
// int32 here keeps it exact, which matters because it is the large
// cancellation term of sum((code-1)*x) = sum(code*x) - sum(x)).
//
// Same H S P transform as kPrismFold above (sign flip, optional GDN
// value-head permutation, blockwise normalized Hadamard), then the
// per-32 symmetric quantisation amax/127.  qs is int8[n] and xd is
// float[n/32] (scale) followed by int32[n/32] (integer sums).
// grid = ceil(n / kPrismBlock); block = 256.
// Column e -> position, and the inverse (position p -> column).  One
// thread owns the four consecutive POSITIONS 4t..4t+3 so its store is a
// single aligned 32-bit word; the four columns it needs are
// {j, j+4, j+8, j+12} of a 16-column group, i.e. scattered in natural
// order, which is why the gather below indexes the shared tile by
// prismQ8Pq2Col().
__device__ __forceinline__ uint32_t prismQ8Pq2Col(uint32_t p) {
    const uint32_t b = p & 15u;
    return (p & ~31u) + (((p >> 4) & 1u) << 4) + ((b & 3u) << 2) +
           (b >> 2);
}

__global__ void kPrismFoldToQ8(const float *__restrict__ v,
                               int8_t *__restrict__ qs,
                               float *__restrict__ xd, uint32_t n,
                               const float *__restrict__ signs,
                               uint32_t gdnHeadV, uint32_t gdnNk,
                               uint32_t gdnRep,
                               const float *__restrict__ mul,
                               int8_t *__restrict__ qsT) {
    __shared__ float tile[kPrismBlock];
    const uint32_t base = blockIdx.x * kPrismBlock;
    const uint32_t tid = threadIdx.x;
    const uint32_t have = (base < n) ? ((n - base < kPrismBlock)
                                                ? (n - base)
                                                : kPrismBlock)
                                     : 0u;
    if (have == 0u) return;
    const bool perm = (gdnRep > 1u && gdnHeadV > 0u && gdnNk > 0u);
    for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x) {
        float val = 0.0f;
        if (i < have) {
            const uint32_t g = base + i;
            const uint32_t si =
                    perm ? prismGdnSrcIndex(g, gdnHeadV, gdnNk, gdnRep) : g;
            val = v[si];
            // Optional silu(x)*mul fused into the fold: ffn_down would
            // otherwise launch kSiluMul over 17408 elements and then
            // read them back here.  Same expression order as kSiluMul
            // (`x / (1 + exp(-x)) * b`), so the elementwise result is
            // unchanged.
            if (mul != nullptr) {
                val = (val / (1.0f + __expf(-val))) * mul[si];
            }
            if (signs != nullptr) val *= signs[g];
        }
        tile[i] = val;
    }
    __syncthreads();
    prismFwhtShared(tile, kPrismBlock, tid, blockDim.x);
    // Thread t writes the 4 q8 bytes at positions 4t..4t+3 of the tile,
    // i.e. columns {j, j+4, j+8, j+12} of a 16-column group (the q8
    // permutation above), so the store is one 32-bit word.
    const uint32_t e = tid * 4u;
    float vv[4];
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t col = prismQ8Pq2Col(e + j);
        vv[j] = (col < have) ? tile[col] : 0.0f;
    }
    float amax = fmaxf(fmaxf(fabsf(vv[0]), fabsf(vv[1])),
                       fmaxf(fabsf(vv[2]), fabsf(vv[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    const float d = amax * (1.0f / 127.0f);
    const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q[4];
    int isum = 0;
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        q[j] = __float2int_rn(vv[j] * inv);
        q[j] = max(-127, min(127, q[j]));
        isum += q[j];
    }
    isum += __shfl_xor_sync(0xffffffffu, isum, 1);
    isum += __shfl_xor_sync(0xffffffffu, isum, 2);
    isum += __shfl_xor_sync(0xffffffffu, isum, 4);
    const uint32_t packed =
            (static_cast<uint32_t>(static_cast<uint8_t>(
                     static_cast<int8_t>(q[0]))) |
             (static_cast<uint32_t>(static_cast<uint8_t>(
                      static_cast<int8_t>(q[1])))
              << 8) |
             (static_cast<uint32_t>(static_cast<uint8_t>(
                      static_cast<int8_t>(q[2])))
              << 16) |
             (static_cast<uint32_t>(static_cast<uint8_t>(
                      static_cast<int8_t>(q[3])))
              << 24));
    if (e + 3u < have) {
        *reinterpret_cast<uint32_t *>(qs + base + e) = packed;
    } else {
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            if (prismQ8Pq2Col(e + j) < have) {
                qs[base + e + j] = static_cast<int8_t>(q[j]);
            }
        }
    }
    if (qsT != nullptr) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t col = base + prismQ8Pq2Col(e + j);
            if (col < n) {
                const uint32_t blk = col >> 7;
                qsT[static_cast<uint64_t>(blk >> 5) * 4096u +
                    static_cast<uint64_t>((col >> 4) & 7u) * 512u +
                    static_cast<uint64_t>(blk & 31u) * 16u + (col & 15u)] =
                        static_cast<int8_t>(q[j]);
            }
        }
    }
    // Global chunk index: the GEMV reads xd[c] for chunk c of the
    // WHOLE activation, so the tile's base has to be added here (a
    // tile-local index made every tile clobber the first 32 entries
    // and left the rest reading uninitialised device memory).
    const uint32_t blk = base / 32u + e / 32u;
    if ((tid & 7u) == 0u) {
        xd[blk] = d;
        reinterpret_cast<int32_t *>(xd + (n + 31u) / 32u)[blk] = isum;
    }
}

// ------------------------------------------------------------------
// PQ2_0 packed ternary x q8 activation GEMV with `dp4a`, ported from the
// PrismML fork's `mul_mat_vec_pq2_0_ordered` (ggml-cuda/ggml-cuda.cu).
//
// One warp owns R output rows.  Each lane owns whole 32-column chunks:
// for chunk c it reads 8 bytes of packed weight (32 codes, four aligned
// uint16 loads at qs + (c&3)*8, which is 2-byte aligned because QA2_0
// blocks are 34 bytes) and the 32 permuted q8 bytes of that chunk, and
// runs 8 `dp4a` against the four (w >> 2k) & 0x03030303 code words.  The
// -1 code bias is one integer subtraction per chunk: the quantiser
// stored sum(q) exactly, so sum((code-1)*x) = sum(code*x) - sum(q).
//
// The four `w` passes reproduce the fork's accumulation order: lane l
// accumulates chunks l, l+128, ... into v[w] for w = 0..3 and
// s = ((v0 + v1) + v2) + v3, then the warp butterfly sums the lanes.
// Weight bytes per lane per chunk are 8, so one warp-load instruction
// touches eight 32-byte weight sectors (the fork's key MLP property);
// the activation load is one 32-byte chunk shared by all R rows.
//
// Numerics differ from the fp32-activation path only by the int8
// quantisation of the activation (~2^-8 relative); the integer dots are
// exact and the per-32 float scale is applied once per chunk.
// TINYCODER_PQ2_Q8=0 restores the fp32-activation GEMV.
// ------------------------------------------------------------------
template<uint32_t R, uint32_t NW = 4u>
__global__ void __launch_bounds__(NW * 32u) kQGemvPQ2_0Q8F(
        const uint8_t *__restrict__ w, const int8_t *__restrict__ xq,
        const float *__restrict__ xd, float *__restrict__ out,
        uint32_t rows, uint32_t cols, uint32_t rowBytes,
        uint32_t blocksPerRow) {
    const uint32_t warp = blockIdx.x * NW + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = warp * R;
    if (row0 >= rows) return;
    const uint32_t nr = (rows - row0 < R) ? (rows - row0) : R;
    const uint8_t *rp[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        rp[r] = w + static_cast<uint64_t>(row0 + (r < nr ? r : nr - 1u)) *
                            rowBytes;
    }
    const uint32_t nchunk = cols / 32u;
    const int32_t *xsum = reinterpret_cast<const int32_t *>(
            xd + ((cols + 31u) / 32u));
    float s[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) s[r] = 0.0f;
#pragma unroll
    for (uint32_t wp = 0; wp < 4u; ++wp) {
        float v[R];
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) v[r] = 0.0f;
        for (uint32_t c = 32u * wp + lane; c < nchunk; c += 128u) {
            uint32_t w0[R], w1[R];
            float d2[R];
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                const uint8_t *blk = rp[r] +
                                     static_cast<uint64_t>(c >> 2) *
                                             kPQ2_0_BYTES;
                const uint16_t *qp = reinterpret_cast<const uint16_t *>(
                        blk + 2u + (c & 3u) * 8u);
                d2[r] = __half2float(
                        *reinterpret_cast<const __half *>(blk));
                w0[r] = static_cast<uint32_t>(qp[0]) |
                        (static_cast<uint32_t>(qp[1]) << 16);
                w1[r] = static_cast<uint32_t>(qp[2]) |
                        (static_cast<uint32_t>(qp[3]) << 16);
            }
            const int4 a0 = *reinterpret_cast<const int4 *>(
                    xq + static_cast<uint64_t>(c) * 32u);
            const int4 a1 = *reinterpret_cast<const int4 *>(
                    xq + static_cast<uint64_t>(c) * 32u + 16u);
            const float d8 = xd[c];
            const int qsum = xsum[c];
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                if (r >= nr) break;
                int sumi = 0;
                sumi = __dp4a(static_cast<int>((w0[r] >> 0) &
                                               0x03030303u),
                              a0.x, sumi);
                sumi = __dp4a(static_cast<int>((w0[r] >> 2) &
                                               0x03030303u),
                              a0.y, sumi);
                sumi = __dp4a(static_cast<int>((w0[r] >> 4) &
                                               0x03030303u),
                              a0.z, sumi);
                sumi = __dp4a(static_cast<int>((w0[r] >> 6) &
                                               0x03030303u),
                              a0.w, sumi);
                sumi = __dp4a(static_cast<int>((w1[r] >> 0) &
                                               0x03030303u),
                              a1.x, sumi);
                sumi = __dp4a(static_cast<int>((w1[r] >> 2) &
                                               0x03030303u),
                              a1.y, sumi);
                sumi = __dp4a(static_cast<int>((w1[r] >> 4) &
                                               0x03030303u),
                              a1.z, sumi);
                sumi = __dp4a(static_cast<int>((w1[r] >> 6) &
                                               0x03030303u),
                              a1.w, sumi);
                sumi -= qsum;
                v[r] = fmaf(d2[r] * d8, static_cast<float>(sumi), v[r]);
            }
        }
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) s[r] += v[r];
    }
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        float a = s[r];
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            a += __shfl_xor_sync(0xffffffffu, a, off);
        }
        if (lane == 0 && row0 + r < rows) out[row0 + r] = a;
    }
}


// ==================================================================
// Native PTQ1_0 (type 143, 28 B / 128 ternary weights) decode.
//
// The native block is qs[24] (5 trits/byte, elements 16n+m at byte m,
// 80+8n+m at byte 16+m) + qh[2] (4 trits/byte, elements 120+2n+h) +
// one fp16 d.  It is the same ternary source as PQ2_0 with 17.6% fewer
// bytes; the transcode to the 34-byte PQ2 layout existed only because
// no PTQ1_0 reader existed.  Decode is the reference's incremental
// chain (ggml-cuda/mmvq-ptq1_0.cuh:ptq1_0_trit_step), 3 int ops per
// four codes, with the -1 code bias left as one `- qsum` per 32-chunk
// exactly as the PQ2_0 GEMV does it.
//
// The decode GEMV keeps the PQ2_0 kernel's exact partition and fp32
// order: a lane loads and unpacks a whole 128-weight block, then hands
// each 32-chunk integer dot to the lane that owns it in the PQ2_0
// partition through a warp-local shared exchange.  The receiver applies
// d2*d8 and accumulates into v[pass] with the identical expression, so
// switching the loader from transcoded to native does not move one bit
// of the output.
//
// The activation side gains one buffer: the q8 bytes in the 32-block
// group are stored piece-major (group g, 16-byte piece i, lane l at
// g*4096 + i*512 + l*16) so the eight int4 loads of a warp are each a
// contiguous 512 B, instead of touching 32 cache lines apiece.  The
// fold writes it alongside the PQ2 layout; PQ2 models leave it null.
// The L2-resident probe measured 306 -> 428 GB/s for this layout; at
// model scale the exchange path lands at 293 GB/s (see
// docs/specs/ptq1_0-native-decode.md).
// ==================================================================
__device__ __forceinline__ int ptq1Dp4aUs(unsigned int a, int b,
                                          int c) {
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(c) : "r"(a), "r"(b), "r"(c));
    return c;
}

__device__ __forceinline__ uint32_t ptq1TritStep(uint32_t &vlo,
                                                 uint32_t &vhi) {
    const uint32_t wlo = vlo * 3u;
    const uint32_t whi = vhi * 3u;
    vlo = wlo & 0x00FF00FFu;
    vhi = whi & 0x00FF00FFu;
    return __byte_perm(wlo, whi, 0x7531);
}

__device__ __forceinline__ int ptq1IntAt(const int4 &v, uint32_t k) {
    return (k == 0u) ? v.x : (k == 1u) ? v.y
                     : (k == 2u)       ? v.z
                                       : v.w;
}

__device__ __forceinline__ int ptq1PlaneDot4(uint32_t (&vlo)[4],
                                             uint32_t (&vhi)[4],
                                             const int4 &a, int acc) {
    const uint32_t q0 = ptq1TritStep(vlo[0], vhi[0]);
    const uint32_t q1 = ptq1TritStep(vlo[1], vhi[1]);
    const uint32_t q2 = ptq1TritStep(vlo[2], vhi[2]);
    const uint32_t q3 = ptq1TritStep(vlo[3], vhi[3]);
    acc = ptq1Dp4aUs(q0, a.x, acc);
    acc = ptq1Dp4aUs(q1, a.y, acc);
    acc = ptq1Dp4aUs(q2, a.z, acc);
    acc = ptq1Dp4aUs(q3, a.w, acc);
    return acc;
}

__device__ __forceinline__ int ptq1PlaneDot2(uint32_t (&vlo)[2],
                                             uint32_t (&vhi)[2],
                                             int a0, int a1, int acc) {
    const uint32_t q0 = ptq1TritStep(vlo[0], vhi[0]);
    const uint32_t q1 = ptq1TritStep(vlo[1], vhi[1]);
    acc = ptq1Dp4aUs(q0, a0, acc);
    acc = ptq1Dp4aUs(q1, a1, acc);
    return acc;
}

// 128 natural-order activation bytes (piece-major group g) -> int4[8].
__device__ __forceinline__ void ptq1LoadActivation(
        const int8_t *__restrict__ group, uint32_t lane, int4 (&a0)[4],
        int4 (&a1)[4]) {
#pragma unroll
    for (uint32_t ci = 0; ci < 4u; ++ci) {
        const int8_t *ap =
                group + static_cast<uint64_t>(2u * ci) * 512u + lane * 16u;
        a0[ci] = *reinterpret_cast<const int4 *>(ap);
        a1[ci] = *reinterpret_cast<const int4 *>(ap + 512);
    }
}

// Unpack one native block and return its four raw (0..2 digit) 32-chunk
// integer dots plus the fp16 scale bits.
__device__ __forceinline__ void ptq1BlockDots(const uint8_t *blk,
                                              const int4 (&a0)[4],
                                              const int4 (&a1)[4],
                                              int sumi[4],
                                              uint32_t &d2bits) {
    const uint32_t *wb = reinterpret_cast<const uint32_t *>(blk);
    d2bits = wb[6] >> 16;// bytes 26,27 = fp16 d
    uint32_t vlo[4], vhi[4], vlo2[2], vhi2[2];
#pragma unroll
    for (uint32_t g = 0; g < 4u; ++g) {
        vlo[g] = __byte_perm(wb[g], 0u, 0x4140);
        vhi[g] = __byte_perm(wb[g], 0u, 0x4342);
    }
#pragma unroll
    for (uint32_t g = 0; g < 2u; ++g) {
        vlo2[g] = __byte_perm(wb[4u + g], 0u, 0x4140);
        vhi2[g] = __byte_perm(wb[4u + g], 0u, 0x4342);
    }
    int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    s0 = ptq1PlaneDot4(vlo, vhi, a0[0], s0);             // qs plane 0 -> j   0..15
    s0 = ptq1PlaneDot4(vlo, vhi, a1[0], s0);             // qs plane 1 -> j  16..31
    s1 = ptq1PlaneDot4(vlo, vhi, a0[1], s1);             // qs plane 2 -> j  32..47
    s1 = ptq1PlaneDot4(vlo, vhi, a1[1], s1);             // qs plane 3 -> j  48..63
    s2 = ptq1PlaneDot4(vlo, vhi, a0[2], s2);             // qs plane 4 -> j  64..79
    s2 = ptq1PlaneDot2(vlo2, vhi2, a1[2].x, a1[2].y, s2);// j 80..87
    s2 = ptq1PlaneDot2(vlo2, vhi2, a1[2].z, a1[2].w, s2);// j 88..95
    s3 = ptq1PlaneDot2(vlo2, vhi2, a0[3].x, a0[3].y, s3);// j 96..103
    s3 = ptq1PlaneDot2(vlo2, vhi2, a0[3].z, a0[3].w, s3);// j104..111
    s3 = ptq1PlaneDot2(vlo2, vhi2, a1[3].x, a1[3].y, s3);// j112..119
    uint32_t vq = (wb[6] & 0xFFu) | ((wb[6] & 0xFF00u) << 8);
    const uint32_t q0 = vq * 3u;
    vq = q0 & 0x00FF00FFu;
    const uint32_t q1 = vq * 3u;
    vq = q1 & 0x00FF00FFu;
    const uint32_t qw0 = __byte_perm(q0, q1, 0x7531);// j120..123
    const uint32_t q2 = vq * 3u;
    vq = q2 & 0x00FF00FFu;
    const uint32_t q3 = vq * 3u;
    const uint32_t qw1 = __byte_perm(q2, q3, 0x7531);// j124..127
    s3 = ptq1Dp4aUs(qw0, a1[3].z, s3);
    s3 = ptq1Dp4aUs(qw1, a1[3].w, s3);
    sumi[0] = s0;
    sumi[1] = s1;
    sumi[2] = s2;
    sumi[3] = s3;
}

// One int (four columns 4p..4p+3) of a native block, biased -1,0,1.
// Used by the fp32 oracle, the embedding dequant and -- one int per
// thread -- the int8 prefill tile loader.
__device__ __forceinline__ uint32_t ptq1DecodeInt(
        const uint8_t *__restrict__ blk, uint32_t p) {
    if (p < 30u) {
        uint32_t off, plane;
        if (p < 20u) {
            plane = p >> 2;
            off = (p & 3u) * 4u;
        } else {
            plane = (p - 20u) >> 1;
            off = 16u + ((p - 20u) & 1u) * 4u;
        }
        static const uint32_t pow3[5] = {1u, 3u, 9u, 27u, 81u};
        const uint32_t packed =
                *reinterpret_cast<const uint32_t *>(blk + off);
        uint32_t a = __byte_perm(packed, 0u, 0x4140) * pow3[plane];
        uint32_t b = __byte_perm(packed, 0u, 0x4342) * pow3[plane];
        a = (a & 0x00FF00FFu) * 3u;
        b = (b & 0x00FF00FFu) * 3u;
        return __vsub4(__byte_perm(a, b, 0x7531), 0x01010101u);
    }
    uint32_t v = static_cast<uint32_t>(blk[24]) |
                 (static_cast<uint32_t>(blk[25]) << 16);
    const uint32_t a = v * 3u;
    v = a & 0x00FF00FFu;
    const uint32_t b = v * 3u;
    v = b & 0x00FF00FFu;
    if (p == 30u)
        return __vsub4(__byte_perm(a, b, 0x7531), 0x01010101u);
    const uint32_t c = v * 3u;
    v = c & 0x00FF00FFu;
    const uint32_t d = v * 3u;
    return __vsub4(__byte_perm(c, d, 0x7531), 0x01010101u);
}

template<uint32_t R>
__global__ void __launch_bounds__(128) kQGemvPTQ1_0Q8(
        const uint8_t *__restrict__ w, const int8_t *__restrict__ xqT,
        const float *__restrict__ xd, float *__restrict__ out,
        uint32_t rows, uint32_t cols, uint32_t rowBytes,
        uint32_t blocksPerRow) {
    constexpr uint32_t kW = 4u;
    const uint32_t wl = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = blockIdx.x * kW + wl;
    const uint32_t row0 = warp * R;
    if (row0 >= rows) return;
    const uint32_t nr = (rows - row0 < R) ? (rows - row0) : R;
    const uint8_t *rp[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r)
        rp[r] = w + static_cast<uint64_t>(row0 + (r < nr ? r : nr - 1u)) *
                            rowBytes;
    const uint32_t nchunk = cols / 32u;
    const int32_t *xsum =
            reinterpret_cast<const int32_t *>(xd + ((cols + 31u) / 32u));
    __shared__ int shm[kW][4u * R][33];
    float v[4][R];
#pragma unroll
    for (uint32_t k = 0; k < 4u; ++k)
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) v[k][r] = 0.0f;
    const uint32_t ngroups = (blocksPerRow + 31u) / 32u;
    for (uint32_t g = 0; g < ngroups; ++g) {
        const uint32_t b = 32u * g + lane;
        if (b < blocksPerRow) {
            int4 a0[4], a1[4];
            ptq1LoadActivation(xqT + static_cast<uint64_t>(g) * 4096u,
                               lane, a0, a1);
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                int sumi[4];
                uint32_t d2bits = 0;
                ptq1BlockDots(rp[r] + static_cast<uint64_t>(b) * kPTQ1_0_BYTES,
                              a0, a1, sumi, d2bits);
#pragma unroll
                for (uint32_t ci = 0; ci < 4u; ++ci)
                    shm[wl][ci * R + r][lane] =
                            static_cast<int>(
                                    static_cast<uint32_t>(sumi[ci]) &
                                    0xFFFFu) |
                            (static_cast<int>(d2bits) << 16);
            }
        }
        __syncwarp();
        // Receiver role: pass wp, global chunk 128g+32wp+lane, loaded
        // by lane 8wp+(lane>>2) of this group.
#pragma unroll
        for (uint32_t wp = 0; wp < 4u; ++wp) {
            const uint32_t c = 128u * g + 32u * wp + lane;
            if (c >= nchunk) continue;
            const uint32_t src = 8u * wp + (lane >> 2);
            const uint32_t k = lane & 3u;
            const float d8 = xd[c];
            const int qs = xsum[c];
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                const int e = shm[wl][k * R + r][src];
                const __half d2 = __ushort_as_half(
                        static_cast<unsigned short>((e >> 16) & 0xFFFFu));
                const int sumi = static_cast<int16_t>(e & 0xFFFF);
                v[wp][r] = fmaf(__half2float(d2) * d8,
                                static_cast<float>(sumi - qs),
                                v[wp][r]);
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        const float a = ((v[0][r] + v[1][r]) + v[2][r]) + v[3][r];
        float s = a;
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0 && row0 + r < rows) out[row0 + r] = s;
    }
}

// fp32-activation oracle: the A/B and LM-head path.  Same geometry as
// kQGemvPQ2_0T; one lane owns four weights and the block scale.
template<uint32_t R, uint32_t U>
__global__ void __launch_bounds__(128) kQGemvPTQ1_0T(
        const uint8_t *__restrict__ w, const float *__restrict__ x,
        float *__restrict__ out, uint32_t rows, uint32_t cols,
        uint32_t rowBytes, uint32_t blocksPerRow) {
    const uint32_t warp = blockIdx.x * 4u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = warp * R;
    if (row0 >= rows) return;
    const uint32_t nr = (rows - row0 < R) ? (rows - row0) : R;
    const uint8_t *rp[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r)
        rp[r] = w + static_cast<uint64_t>(row0 + (r < nr ? r : nr - 1u)) *
                            rowBytes;
    const bool vec = ((reinterpret_cast<uintptr_t>(x) & 15u) == 0u);
    float acc[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) acc[r] = 0.0f;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        const uint32_t base = b * 128u + lane * 4u;
        float x0, x1, x2, x3;
        if (vec && base + 3u < cols) {
            const float4 v =
                    *reinterpret_cast<const float4 *>(x + base);
            x0 = v.x;
            x1 = v.y;
            x2 = v.z;
            x3 = v.w;
        } else {
            x0 = (base + 0u < cols) ? x[base + 0u] : 0.0f;
            x1 = (base + 1u < cols) ? x[base + 1u] : 0.0f;
            x2 = (base + 2u < cols) ? x[base + 2u] : 0.0f;
            x3 = (base + 3u < cols) ? x[base + 3u] : 0.0f;
        }
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) {
            if (r >= nr) break;
            const uint8_t *blk =
                    rp[r] + static_cast<uint64_t>(b) * kPTQ1_0_BYTES;
            const uint32_t q = ptq1DecodeInt(blk, lane);
            const float d = __half2float(
                    *reinterpret_cast<const __half *>(blk + 26u));
            float a = acc[r];
            a = fmaf(d * static_cast<float>(
                                 static_cast<int8_t>(q & 0xFFu)),
                     x0, a);
            a = fmaf(d * static_cast<float>(
                                 static_cast<int8_t>((q >> 8u) & 0xFFu)),
                     x1, a);
            a = fmaf(d * static_cast<float>(
                                 static_cast<int8_t>((q >> 16u) & 0xFFu)),
                     x2, a);
            a = fmaf(d * static_cast<float>(
                                 static_cast<int8_t>((q >> 24u) & 0xFFu)),
                     x3, a);
            acc[r] = a;
        }
    }
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        float a = acc[r];
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1)
            a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < rows) out[row0 + r] = a;
    }
}
