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

// GpuGemvQ2KFused.hpp -- Q2_K int-dot GEMV, fused gate+up+silu, grouped prefill helpers.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// ---- Q2_K 8-way block batching in the 4xW shape (2026-09-30, lever-3
// A/B TINYCODER_Q2K_8W=1, DEFAULT OFF): identical geometry to
// kQGemvQ2KxW4 (all 32 lanes issue u32 loads, 8 lanes/row, 4 rows per
// warp, no smem/barrier) but batches 8 blocks per outer iteration.
// The 4xW version already interleaves 4 independent block chains per
// row; 8-way doubles the in-flight weight loads per lane to hide DRAM
// latency, at the cost of more registers.  Per-element float math is
// IDENTICAL to kQGemvQ2KxW4 / kQGemv<kTypeQ2K> -- only the block
// accumulation order changes (a0..a7 folded in order, then the warp
// tree), so results stay within the Q2_K float-path ~1e-7 tolerance.
__global__ void kQGemvQ2KxW4b8(const uint8_t *__restrict__ w,
                               const float *__restrict__ x,
                               float *__restrict__ out,
                               uint32_t rows, uint32_t cols,
                               uint32_t rowBytes,
                               uint32_t blocksPerRow) {
    const uint32_t warpRowBase =
            blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t base = warpRowBase * 4u;
    if (base >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t I = lane & 7u;
    const uint32_t row = base + (lane >> 3u);
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    const uint32_t qOfs = 4u * I;
    const uint32_t qb = qOfs;
    const uint32_t isub = (qOfs < 16u) ? 0u : 1u;
    float acc = 0.0f;
    uint32_t b = 0;
    for (; b + 8u <= blocksPerRow; b += 8u) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
#pragma unroll
        for (uint32_t bb = 0; bb < 8; ++bb) {
            const uint8_t *crp =
                    rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
            const float df = __half2float(
                    *reinterpret_cast<const __half *>(crp + 80));
            const float dmf = __half2float(
                    *reinterpret_cast<const __half *>(crp + 82));
            const uint8_t *sc = crp;
            const uint8_t *q = crp + 16;
            float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                    : (bb == 2u)       ? a2
                                    : (bb == 3u)       ? a3
                                    : (bb == 4u)       ? a4
                                    : (bb == 5u)       ? a5
                                    : (bb == 6u)       ? a6
                                                       : a7;
#pragma unroll
            for (uint32_t n = 0; n < 2; ++n) {
                uint32_t w4;
                std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint8_t scv = sc[n * 8u + 2u * j + isub];
                    const float dl =
                            df * static_cast<float>(scv & 0xFu);
                    const float ml =
                            dmf * static_cast<float>(scv >> 4u);
                    const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                    for (uint32_t k = 0; k < 4; ++k) {
                        const float qv = static_cast<float>(
                                static_cast<int8_t>(
                                        (w4 >> (8u * k + 2u * j)) & 3u));
                        const uint32_t cidx = cbase + k;
                        slot = fmaf(fmaf(dl, qv, -ml),
                                    (cidx < cols) ? x[cidx] : 0.0f,
                                    slot);
                    }
                }
            }
        }
        acc += a0;
        acc += a1;
        acc += a2;
        acc += a3;
        acc += a4;
        acc += a5;
        acc += a6;
        acc += a7;
        rp += 8u * kQ2K_BYTES;
    }
    // Remainder blocks (blocksPerRow % 8) -- 4xW-compatible single/4
    // block bodies fall back to the 4xW remainder loop by just reusing
    // the 4-way body on the leftover (handles 0..7 leftover blocks).
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
        for (uint32_t bb = 0; bb < 4; ++bb) {
            const uint8_t *crp =
                    rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
            const float df = __half2float(
                    *reinterpret_cast<const __half *>(crp + 80));
            const float dmf = __half2float(
                    *reinterpret_cast<const __half *>(crp + 82));
            const uint8_t *sc = crp;
            const uint8_t *q = crp + 16;
            float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                    : (bb == 2u)       ? a2
                                                       : a3;
#pragma unroll
            for (uint32_t n = 0; n < 2; ++n) {
                uint32_t w4;
                std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint8_t scv = sc[n * 8u + 2u * j + isub];
                    const float dl =
                            df * static_cast<float>(scv & 0xFu);
                    const float ml =
                            dmf * static_cast<float>(scv >> 4u);
                    const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                    for (uint32_t k = 0; k < 4; ++k) {
                        const float qv = static_cast<float>(
                                static_cast<int8_t>(
                                        (w4 >> (8u * k + 2u * j)) & 3u));
                        const uint32_t cidx = cbase + k;
                        slot = fmaf(fmaf(dl, qv, -ml),
                                    (cidx < cols) ? x[cidx] : 0.0f,
                                    slot);
                    }
                }
            }
        }
        acc += a0;
        acc += a1;
        acc += a2;
        acc += a3;
        rp += 4u * kQ2K_BYTES;
    }
    for (; b < blocksPerRow; ++b) {
        const float df = __half2float(
                *reinterpret_cast<const __half *>(rp + 80));
        const float dmf = __half2float(
                *reinterpret_cast<const __half *>(rp + 82));
        const uint8_t *sc = rp;
        const uint8_t *q = rp + 16;
#pragma unroll
        for (uint32_t n = 0; n < 2; ++n) {
            uint32_t w4;
            std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
            const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                const uint8_t scv = sc[n * 8u + 2u * j + isub];
                const float dl = df * static_cast<float>(scv & 0xFu);
                const float ml = dmf * static_cast<float>(scv >> 4u);
                const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                for (uint32_t k = 0; k < 4; ++k) {
                    const float qv = static_cast<float>(
                            static_cast<int8_t>(
                                    (w4 >> (8u * k + 2u * j)) & 3u));
                    const uint32_t cidx = cbase + k;
                    acc = fmaf(fmaf(dl, qv, -ml),
                               (cidx < cols) ? x[cidx] : 0.0f, acc);
                }
            }
        }
        rp += kQ2K_BYTES;
    }
    // 8-lane shuffle tree per row (xor 4,1,2 stays inside the row's
    // 8-lane group -- same tree as kQGemvQ6KxQ8K_4xW).
    const float p = acc + __shfl_xor_sync(0xffffffffu, acc, 4);
    const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
    const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
    if (I == 0) out[row] = rr;
}

// ---- Q2_K x Q8_K int-dot GEMV, 4 rows per warp (2026-09-30, A/B
// gate TINYCODER_Q2K_MQ4XW=1, DEFAULT OFF until measured).  This is
// reference-int8's decode route (kQuantizeQ8K the fp32 activation ONCE,
// then an exact-integer 2-bit x Q8_K dot), brought to the Q6_K-proven
// 4xW shape.  The float kQGemvQ2KxW4 reads the fp32 activation SCALAR
// per element (32 predicated 4B loads + 32 fmaf/block/lane) which
// keeps the Q2_K FFN at ~75-113 GB/s while the Q6_K LM head (same
// geometry, Q8_K-int math) hits ~616 GB/s -- the FFN bandwidth gap
// measured by [gpu fwd] decode matrix sums (gate+up 2.24 ms, down
// 2.11 ms of the ~7.5 ms layer loop).
//
// STRUCTURE: identical to kQGemvQ6KxQ8K_4xW -- all 32 lanes issue u32
// loads, 8 lanes per row (lane I owns the 4 contiguous cols 4I..4I+3
// of every 32-col group), 4 rows per warp, 4-way block batching into
// INDEPENDENT per-lane float accumulators r0..r3, and a SINGLE 8-lane
// float tree at the END (no per-block shuffles / lane-0 serial folds
// -- that was the failure mode of the earlier int-dot variants).
//
// MATH (int side bit-identical to dotProductQ2_K_PrePacked_Q8_Scalar /
// the compact AVX2 makeSetup+blockDot): col c -> weight byte
// q[n*32 + c%32], value (byte >> 2*((c%128)/32)) & 3; 16-group g =
// n*8 + 2*j + sub (n = c/128, j = (c%128)/32, sub = (c%32)/16);
// dl_g = dq*(scales[g]&0xF), ml_g = dmin*(scales[g]>>4);
// block = d8*(sum_g dl_g*S_g - sum_g ml_g*bsums_g).
//   * lane I: isub = I>>2 is CONSTANT, so dl_g == dq*sc[n*8+2j+isub]
//     for every group the lane touches, and the per-lane partial
//     dl_g*S_g-partial sums over the 8-lane tree to sum_g dl_g*S_g.
//   * per (n,j): one u32 weight load q[n*32+4I..+3] (SAME bytes for
//     all 4 j -- only the 2-bit shift 2*j changes) x one u32 Q8
//     activation load y8[n*128+j*32+4I..+3] via __dp4a (signed bytes;
//     weight bytes are 0..3 so signed==unsigned).
//   * the ml term (bsums_g is a FULL 16-col sum, same for all 4 lanes
//     of a sub-group) is folded on the sub-group representative lanes
//     (I&3)==0 -- lanes 0 and 4 -- so the single end tree sums the
//     dl*S terms from all 8 lanes and the ml terms exactly once.
__device__ __forceinline__ void q2kXQ8KBlockDot(
        float &acc, const uint8_t *__restrict__ blkW,
        const uint8_t *__restrict__ blkA, uint32_t I) {
    const uint32_t isub = I >> 2u;     // 0 (lanes 0-3) or 1 (lanes 4-7)
    const bool mrep = ((I & 3u) == 0u);// sub-group rep folds ml
    const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA);
    const int16_t *bs =
            reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
    const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
    const float dq = __half2float(
            *reinterpret_cast<const __half *>(blkW + 80));
    const float dmin = __half2float(
            *reinterpret_cast<const __half *>(blkW + 82));
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
        // 4 weight bytes for this lane (cols 4I..4I+3 of chunk n);
        // reused across all 4 groups j via the 2-bit shift 2*j.
        uint32_t wb;
        std::memcpy(&wb, blkW + 16u + n * 32u + 4u * I,
                    sizeof(uint32_t));
        uint64_t scU;
        std::memcpy(&scU, blkW + n * 8u, sizeof(uint64_t));
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            uint32_t yv;
            std::memcpy(&yv, y8 + n * 128u + j * 32u + 4u * I,
                        sizeof(uint32_t));
            const uint32_t pk = (wb >> (2u * j)) & 0x03030303u;
            const int pdot = __dp4a(static_cast<int>(pk),
                                    static_cast<int>(yv), 0);
            const uint8_t scv = static_cast<uint8_t>(
                    (scU >> (8u * (2u * j + isub))) & 0xFFu);
            const float dl = dq * static_cast<float>(scv & 0xFu);
            acc += d8 * (dl * static_cast<float>(pdot));
            if (mrep) {
                const float ml = dmin * static_cast<float>(scv >> 4u);
                // Direct indexed int16 load: a packed u64 shift by
                // >64 bits (groups 4..7) is UB and silently zeroed
                // the ml terms for the high groups.
                const int bg = static_cast<int>(bs[n * 8u + 2u * j +
                                                   isub]);
                acc -= d8 * (ml * static_cast<float>(bg));
            }
        }
    }
}

__global__ void kQGemvQ2KxQ8K_4xW(const uint8_t *__restrict__ w,
                                  const uint8_t *__restrict__ x,
                                  float *__restrict__ out,
                                  uint32_t rows,
                                  uint32_t blocksPerRow,
                                  uint32_t rowBytes) {
    const uint32_t warpRowBase =
            blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t base = warpRowBase * 4u;
    if (base >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t I = lane & 7u;
    const uint32_t row = base + (lane >> 3u);
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    float sumf = 0.0f;
    uint32_t b = 0;
    // 4-way block batching into INDEPENDENT per-lane accumulators
    // (same memory-level-parallelism pattern as kQGemvQ6KxQ8K_4xW).
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        q2kXQ8KBlockDot(
                r0, rp + static_cast<uint64_t>(b + 0u) * kQ2K_BYTES,
                x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE, I);
        q2kXQ8KBlockDot(
                r1, rp + static_cast<uint64_t>(b + 1u) * kQ2K_BYTES,
                x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE, I);
        q2kXQ8KBlockDot(
                r2, rp + static_cast<uint64_t>(b + 2u) * kQ2K_BYTES,
                x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE, I);
        q2kXQ8KBlockDot(
                r3, rp + static_cast<uint64_t>(b + 3u) * kQ2K_BYTES,
                x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE, I);
        sumf = r0 + sumf;
        sumf = r1 + sumf;
        sumf = r2 + sumf;
        sumf = r3 + sumf;
    }
    // Remainder (blocksPerRow % 4) -- single-block body, same math.
    for (; b < blocksPerRow; ++b) {
        q2kXQ8KBlockDot(
                sumf, rp + static_cast<uint64_t>(b) * kQ2K_BYTES,
                x + static_cast<uint64_t>(b) * kQ8K_STRIDE, I);
    }
    // 8-lane shuffle tree per row (xor 4,1,2 stays inside the row's
    // 8-lane group -- same tree as kQGemvQ6KxQ8K_4xW).
    const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
    const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
    const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
    if (I == 0) out[row] = rr;
}

// MEASURED (eager, interleaved A/B, n16/g8/r5): GEMV 153.3/144.0 vs
// float 4xW default 167.4/168.4 -> REGRESSION. This REFUTES the
// wave-granularity hypothesis that motivated it: the 4xW/MQ4XW
// 280-block grid is a SINGLE full wave (68 SMs x 8 resident 256-thread
// blocks = 544 >= 280), so row-granular blocks bought only block
// launch/scheduling overhead (4480 tiny launches) with no bandwidth
// gain. Combined with MQ4XW (int, 280 blocks: 153-156) this closes the
// FFN-kernel lever: neither the int-dot route nor the grid shape beats
// the float 4xW -- the decode kernel is NOT the 168 vs 261 gap. MATH is
// bit-identical to kQGemvQ2KxQ8K_4xW (same q2kXQ8KBlockDot); only the
// reduce ORDER across lane-groups differs (xor 16/8 instead of serial
// batch fold) -- same float-tolerance class as the 4xW.
__global__ void kQGemvQ2KxQ8K_Gemv(const uint8_t *__restrict__ w,
                                   const uint8_t *__restrict__ x,
                                   float *__restrict__ out,
                                   uint32_t rows,
                                   uint32_t blocksPerRow,
                                   uint32_t rowBytes) {
    // block = (32, 2): 2 warps, one row per warp, grid = rows/2.
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t I = lane & 7u;
    const uint32_t g = lane >> 3u;// 4 lane-groups split the blocks
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    float sumf = 0.0f;
    // Row's Q2_K blocks round-robin across the 4 groups (FFN/attnO
    // blocksPerRow = 6: group g handles blocks g and g+4).  All 32
    // lanes stay busy; groups with no blocks idle into the tree.
    for (uint32_t b = g; b < blocksPerRow; b += 4u) {
        q2kXQ8KBlockDot(sumf, rp + static_cast<uint64_t>(b) * kQ2K_BYTES,
                        x + static_cast<uint64_t>(b) * kQ8K_STRIDE, I);
    }
    // 8-lane tree folds each group's partial onto all its lanes; xor
    // 16 then 8 folds the four group partials (g0..g3 live on lanes
    // 0..7, 8..15, 16..23, 24..31 after the first tree) down to lane 0.
    const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
    const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
    const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
    const float s = rr + __shfl_xor_sync(0xffffffffu, rr, 16);
    const float t = s + __shfl_xor_sync(0xffffffffu, s, 8);
    if (lane == 0) out[row] = t;
}

// ---- Q2_K int-dot fused gate+up+silu (2026-09-30, lever 1, A/B
// TINYCODER_Q2K_INTGU=1).
//
// MATH (per 256-wide block; integer side bit-identical to the CPU
// reference dotProductQ2_K_PrePacked_Q8_Scalar and the compact AVX2
// kernel's makeSetup+blockDot):
//   * col c: chunk n = c/128, group j = (c%128)/32, sub = (c%32)/16,
//     weight byte = q[n*32 + c%32], 2-bit value = (byte >> 2*j) & 3;
//     scale byte = scales[n*8 + j*2 + sub] (16 bytes, chunk-major);
//     dl = dq*(sc&0xF), ml = dmin*(sc>>4).
//   * S_g = 16-col int dot of weights vs Q8 y8; bsums[g] = Q8 block's
//     group-g int16 sum.  block = dq*d8*sum_g((sc&0xF)*S_g)
//                               - dmin*d8*sum_g((sc>>4)*bsums_g).
//
// LANE FACTORIZATION (exact): lane i (0..31) owns 8 CONTIGUOUS cols
// c0 = i*8, so n = i>>4, j = (i>>2)&3, sub = (i&3)>>1:
//   * weight bytes are CONTIGUOUS at q + (i>>4)*32 + (i&3)*8 (byte k
//     covers col c0+k; c%32 == (i&3)*8 + k).  Verified vs dequant.
//   * the 2-bit shift is CONSTANT over the lane: 2*j = 2*((i>>2)&3).
//   * scale byte index == i>>1 (identity (i>>4)*8 + ((i>>2)&3)*2 +
//     ((i&3)>>1) == i>>1 for all i<32, verified).
// So the dl term is ONE 32-lane int tree of (scales[i>>1]&0xF)*S_i and
// the ml term folds on lane 0 from the 16 Q8 bsums:
// mlt = sum_g (scales[g]>>4)*bsums[g].  4-way block batching (8
// independent per-lane int chains), lane-0 serial float fold in block
// order (same convention as kQGemvQ8KFusedGU), silu(gate)*up out[row].
//
// Per-lane Q2_K 8-col int dot against shared Q8 y8: the lane's weight
// bytes are contiguous (see above); sh is the constant 2-bit shift.
__device__ __forceinline__ void q2kIntDot8(
        const uint8_t *__restrict__ qb, const uint8_t *__restrict__ qu,
        const int8_t *__restrict__ y8, uint32_t e0, uint32_t sh,
        int &gacc, int &uacc) {
    uint32_t wAg, wBg, wAu, wBu;
    std::memcpy(&wAg, qb, sizeof(uint32_t));
    std::memcpy(&wBg, qb + 4, sizeof(uint32_t));
    std::memcpy(&wAu, qu, sizeof(uint32_t));
    std::memcpy(&wBu, qu + 4, sizeof(uint32_t));
    const uint64_t yv = __ldg(
            reinterpret_cast<const uint64_t *>(&y8[e0]));
#pragma unroll
    for (uint32_t k = 0; k < 8; ++k) {
        const uint32_t bv = (k < 4u) ? ((wAg >> (8u * k)) & 0xFFu)
                                     : ((wBg >> (8u * (k & 3u))) & 0xFFu);
        const uint32_t bvU = (k < 4u) ? ((wAu >> (8u * k)) & 0xFFu)
                                      : ((wBu >> (8u * (k & 3u))) & 0xFFu);
        const int vy = static_cast<int>(static_cast<int8_t>(
                (yv >> (8u * k)) & 0xFFu));
        gacc += static_cast<int>((bv >> sh) & 3u) * vy;
        uacc += static_cast<int>((bvU >> sh) & 3u) * vy;
    }
}

// Full 32-lane int xor-tree (1,2,4,8,16): exact warp-wide sum on every
// lane (integer addition is order-free, so tree order is irrelevant).
__device__ __forceinline__ int q2kWarpSumi(int v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    return v;
}

// Lane-0-only ml term: sum_g (scales[g]>>4) * bsums[g].
__device__ __forceinline__ int q2kMlTerm(const uint8_t *__restrict__ blk,
                                         const int16_t *__restrict__ bs) {
    int mlt = 0;
#pragma unroll
    for (uint32_t g = 0; g < 16; ++g) {
        mlt += static_cast<int>(blk[g] >> 4u) *
               static_cast<int>(bs[g]);
    }
    return mlt;
}

__global__ void __launch_bounds__(256) kQGemvQ2KIntFusedGU(
        const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
        const uint8_t *__restrict__ x, float *__restrict__ out,
        uint32_t rows, uint32_t blocksPerRow, uint32_t rowBytes) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint8_t *rg = wg + static_cast<uint64_t>(row) * rowBytes;
    const uint8_t *ru = wu + static_cast<uint64_t>(row) * rowBytes;
    const uint32_t lane = threadIdx.x;
    const uint32_t qOff = 16u + (lane >> 4u) * 32u + (lane & 3u) * 8u;
    const uint32_t sh = 2u * ((lane >> 2u) & 3u);
    const uint32_t p = lane >> 1u;// scale byte index (0..15)
    const uint32_t e0 = lane * 8u;// first col of the lane's 8
    float sumfG = 0.0f, sumfU = 0.0f;
    uint32_t b = 0;
    // 4-way block batching: 8 independent per-lane int chains feed 8
    // full-warp int trees; lane 0 folds gate then up in block order.
    for (; b + 4u <= blocksPerRow; b += 4u) {
        const uint8_t *gw0 = rg + static_cast<uint64_t>(b + 0u) * kQ2K_BYTES;
        const uint8_t *gw1 = rg + static_cast<uint64_t>(b + 1u) * kQ2K_BYTES;
        const uint8_t *gw2 = rg + static_cast<uint64_t>(b + 2u) * kQ2K_BYTES;
        const uint8_t *gw3 = rg + static_cast<uint64_t>(b + 3u) * kQ2K_BYTES;
        const uint8_t *uw0 = ru + static_cast<uint64_t>(b + 0u) * kQ2K_BYTES;
        const uint8_t *uw1 = ru + static_cast<uint64_t>(b + 1u) * kQ2K_BYTES;
        const uint8_t *uw2 = ru + static_cast<uint64_t>(b + 2u) * kQ2K_BYTES;
        const uint8_t *uw3 = ru + static_cast<uint64_t>(b + 3u) * kQ2K_BYTES;
        const uint8_t *ba0 = x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE;
        const uint8_t *ba1 = x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE;
        const uint8_t *ba2 = x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE;
        const uint8_t *ba3 = x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE;
        const int8_t *y0 = reinterpret_cast<const int8_t *>(ba0 + kQ8K_Q);
        const int8_t *y1 = reinterpret_cast<const int8_t *>(ba1 + kQ8K_Q);
        const int8_t *y2 = reinterpret_cast<const int8_t *>(ba2 + kQ8K_Q);
        const int8_t *y3 = reinterpret_cast<const int8_t *>(ba3 + kQ8K_Q);
        const float d8_0 = *reinterpret_cast<const float *>(ba0 + kQ8K_D);
        const float d8_1 = *reinterpret_cast<const float *>(ba1 + kQ8K_D);
        const float d8_2 = *reinterpret_cast<const float *>(ba2 + kQ8K_D);
        const float d8_3 = *reinterpret_cast<const float *>(ba3 + kQ8K_D);
        // Per-lane scales: low nibble = dl multiplier for that lane's
        // 16-group (index p == i>>1, exact identity, see comment).
        const int scG0 = static_cast<int>(gw0[p] & 0xFu);
        const int scU0 = static_cast<int>(uw0[p] & 0xFu);
        const int scG1 = static_cast<int>(gw1[p] & 0xFu);
        const int scU1 = static_cast<int>(uw1[p] & 0xFu);
        const int scG2 = static_cast<int>(gw2[p] & 0xFu);
        const int scU2 = static_cast<int>(uw2[p] & 0xFu);
        const int scG3 = static_cast<int>(gw3[p] & 0xFu);
        const int scU3 = static_cast<int>(uw3[p] & 0xFu);
        int gci0 = 0, gci1 = 0, gci2 = 0, gci3 = 0;
        int uci0 = 0, uci1 = 0, uci2 = 0, uci3 = 0;
        q2kIntDot8(gw0 + qOff, uw0 + qOff, y0, e0, sh, gci0, uci0);
        q2kIntDot8(gw1 + qOff, uw1 + qOff, y1, e0, sh, gci1, uci1);
        q2kIntDot8(gw2 + qOff, uw2 + qOff, y2, e0, sh, gci2, uci2);
        q2kIntDot8(gw3 + qOff, uw3 + qOff, y3, e0, sh, gci3, uci3);
        const int gt0 = q2kWarpSumi(gci0 * scG0);
        const int gt1 = q2kWarpSumi(gci1 * scG1);
        const int gt2 = q2kWarpSumi(gci2 * scG2);
        const int gt3 = q2kWarpSumi(gci3 * scG3);
        const int ut0 = q2kWarpSumi(uci0 * scU0);
        const int ut1 = q2kWarpSumi(uci1 * scU1);
        const int ut2 = q2kWarpSumi(uci2 * scU2);
        const int ut3 = q2kWarpSumi(uci3 * scU3);
        if (lane == 0) {
            const float dq0 = __half2float(
                    *reinterpret_cast<const __half *>(gw0 + 80));
            const float dmin0 = __half2float(
                    *reinterpret_cast<const __half *>(gw0 + 82));
            const float dqU0 = __half2float(
                    *reinterpret_cast<const __half *>(uw0 + 80));
            const float dminU0 = __half2float(
                    *reinterpret_cast<const __half *>(uw0 + 82));
            sumfG += dq0 * d8_0 * static_cast<float>(gt0) -
                     dmin0 * d8_0 *
                             static_cast<float>(q2kMlTerm(
                                     gw0, reinterpret_cast<const int16_t *>(ba0 + kQ8K_B)));
            sumfU += dqU0 * d8_0 * static_cast<float>(ut0) -
                     dminU0 * d8_0 *
                             static_cast<float>(q2kMlTerm(
                                     uw0, reinterpret_cast<const int16_t *>(ba0 + kQ8K_B)));
            const float dq1 = __half2float(
                    *reinterpret_cast<const __half *>(gw1 + 80));
            const float dmin1 = __half2float(
                    *reinterpret_cast<const __half *>(gw1 + 82));
            const float dqU1 = __half2float(
                    *reinterpret_cast<const __half *>(uw1 + 80));
            const float dminU1 = __half2float(
                    *reinterpret_cast<const __half *>(uw1 + 82));
            sumfG += dq1 * d8_1 * static_cast<float>(gt1) -
                     dmin1 * d8_1 *
                             static_cast<float>(q2kMlTerm(
                                     gw1, reinterpret_cast<const int16_t *>(ba1 + kQ8K_B)));
            sumfU += dqU1 * d8_1 * static_cast<float>(ut1) -
                     dminU1 * d8_1 *
                             static_cast<float>(q2kMlTerm(
                                     uw1, reinterpret_cast<const int16_t *>(ba1 + kQ8K_B)));
            const float dq2 = __half2float(
                    *reinterpret_cast<const __half *>(gw2 + 80));
            const float dmin2 = __half2float(
                    *reinterpret_cast<const __half *>(gw2 + 82));
            const float dqU2 = __half2float(
                    *reinterpret_cast<const __half *>(uw2 + 80));
            const float dminU2 = __half2float(
                    *reinterpret_cast<const __half *>(uw2 + 82));
            sumfG += dq2 * d8_2 * static_cast<float>(gt2) -
                     dmin2 * d8_2 *
                             static_cast<float>(q2kMlTerm(
                                     gw2, reinterpret_cast<const int16_t *>(ba2 + kQ8K_B)));
            sumfU += dqU2 * d8_2 * static_cast<float>(ut2) -
                     dminU2 * d8_2 *
                             static_cast<float>(q2kMlTerm(
                                     uw2, reinterpret_cast<const int16_t *>(ba2 + kQ8K_B)));
            const float dq3 = __half2float(
                    *reinterpret_cast<const __half *>(gw3 + 80));
            const float dmin3 = __half2float(
                    *reinterpret_cast<const __half *>(gw3 + 82));
            const float dqU3 = __half2float(
                    *reinterpret_cast<const __half *>(uw3 + 80));
            const float dminU3 = __half2float(
                    *reinterpret_cast<const __half *>(uw3 + 82));
            sumfG += dq3 * d8_3 * static_cast<float>(gt3) -
                     dmin3 * d8_3 *
                             static_cast<float>(q2kMlTerm(
                                     gw3, reinterpret_cast<const int16_t *>(ba3 + kQ8K_B)));
            sumfU += dqU3 * d8_3 * static_cast<float>(ut3) -
                     dminU3 * d8_3 *
                             static_cast<float>(q2kMlTerm(
                                     uw3, reinterpret_cast<const int16_t *>(ba3 + kQ8K_B)));
        }
    }
    // Remainder (blocksPerRow % 4) -- single-block body, same math.
    for (; b < blocksPerRow; ++b) {
        const uint8_t *blkWG = rg + static_cast<uint64_t>(b) * kQ2K_BYTES;
        const uint8_t *blkWU = ru + static_cast<uint64_t>(b) * kQ2K_BYTES;
        const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
        const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA + kQ8K_Q);
        const int16_t *bs = reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
        const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
        const int scG = static_cast<int>(blkWG[p] & 0xFu);
        const int scU = static_cast<int>(blkWU[p] & 0xFu);
        int gci = 0, uci = 0;
        q2kIntDot8(blkWG + qOff, blkWU + qOff, y8, e0, sh, gci, uci);
        const int gt = q2kWarpSumi(gci * scG);
        const int ut = q2kWarpSumi(uci * scU);
        if (lane == 0) {
            const float dq = __half2float(
                    *reinterpret_cast<const __half *>(blkWG + 80));
            const float dmin = __half2float(
                    *reinterpret_cast<const __half *>(blkWG + 82));
            const float dqU = __half2float(
                    *reinterpret_cast<const __half *>(blkWU + 80));
            const float dminU = __half2float(
                    *reinterpret_cast<const __half *>(blkWU + 82));
            sumfG += dq * d8 * static_cast<float>(gt) -
                     dmin * d8 * static_cast<float>(q2kMlTerm(blkWG, bs));
            sumfU += dqU * d8 * static_cast<float>(ut) -
                     dminU * d8 * static_cast<float>(q2kMlTerm(blkWU, bs));
        }
    }
    if (lane == 0) {
        const float g = sumfG;
        const float u = sumfU;
        out[row] = (g / (1.0f + __expf(-g))) * u;
    }
}


// ---- Batched gate+up GEMV (Q4_K/Q6_K x Q8_K): grid.y = rank/slot j,
// gates rows [0, expertFF) and up rows [expertFF, 2*expertFF) via the
// slot-major wq table ([j*3+g/u] = gate/up base).  Identical math to
// kQGemvKxQ8K, bit-exact with the CPU grouped path.
template<int TYPE>
__global__ void kQGemvKxQ8K_BatchGU(const uint8_t *const *__restrict__ wq,
                                    const uint8_t *__restrict__ x,
                                    float *__restrict__ gateOut,
                                    float *__restrict__ upOut,
                                    uint32_t expertFF,
                                    uint32_t blocksPerRow,
                                    uint32_t rowBytes,
                                    uint32_t combinedRows) {
    const uint32_t r = blockIdx.y;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= combinedRows) return;
    const uint32_t isUp = (row >= expertFF) ? 1u : 0u;
    const uint32_t wrow = row - isUp * expertFF;
    const uint8_t *rp = wq[r * 3u + isUp] +
                        static_cast<uint64_t>(wrow) * rowBytes;
    const uint32_t lane = threadIdx.x;
    float sumf = 0.0f;
    constexpr uint32_t kBlkBytes =
            (TYPE == kTypeQ4K)     ? kQ4K_BYTES
            : (TYPE == kTypeQ5K)   ? kQ5K_BYTES
            : (TYPE == kTypeIQ4XS) ? kIQ4XS_BYTES
                                   : kQ6K_BYTES;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * kBlkBytes;
        const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
        kxQ8KBlockDot<TYPE>(sumf, blkW, blkA, lane);
    }
    if (TYPE == kTypeQ4K || TYPE == kTypeQ5K ||
        TYPE == kTypeIQ4XS) {
        if (lane == 0) {
            float *dest = isUp ? upOut : gateOut;
            dest[r * expertFF + wrow] = sumf;
        }
    } else {
        const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
        const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
        const float rr = q + __shfl_xor_sync(0xffffffffu, q, 2);
        if (lane == 0) {
            float *dest = isUp ? upOut : gateOut;
            dest[r * expertFF + wrow] = rr;
        }
    }
}

// ---- Grouped prefill helpers (int8 MMA path, 2026-11-10) --------
// xg[p][cols] = x[tok[p]][cols]: gathers each (token, rank) pair's
// activation row into one contiguous matrix, with pairs ordered by
// expert so an expert's rows are a contiguous M-slab for int8Gemm.
__global__ void kMoeGatherRows(const float *__restrict__ x,
                               const int32_t *__restrict__ tok,
                               float *__restrict__ out, uint32_t cols) {
    const uint32_t p = blockIdx.y;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    out[static_cast<uint64_t>(p) * cols + i] =
            x[static_cast<size_t>(tok[p]) * cols + i];
}

// ---- Token-reuse DOWN GEMV (2026-11-10) ------------------------
// The down projection contracts over expertFF (256), which the int8
// MMA path cannot take (it needs a 512-aligned contraction width), so
// it stays a Q4_K/Q6_K x Q8_K warp dot.  The per-pair launch read the
// expert's whole 0.86 MB down matrix once PER TOKEN; here one block
// owns a row tile of ONE expert and loops over that expert's token
// group, so the weight bytes are decoded once and reused for every
// token routed to the expert (measured: 3.5 GB -> 132 MB of down
// weight reads per layer).
template<int TYPE, uint32_t TOK>
__global__ void kQGemvKxQ8K_DownExpert(
        const uint8_t *const *__restrict__ wq,
        const uint8_t *__restrict__ xq,
        const int32_t *__restrict__ ranked,
        const int32_t *__restrict__ eStart, float *__restrict__ out,
        uint32_t hidden, uint32_t blocksPerRow, uint32_t rowBytes,
        uint32_t xStride, uint32_t outStride) {
    const uint32_t e = blockIdx.y;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= hidden) return;
    const uint32_t p0 = static_cast<uint32_t>(eStart[e]);
    const uint32_t p1 = static_cast<uint32_t>(eStart[e + 1]);
    if (p0 >= p1) return;
    const uint8_t *rp = wq[p0 * 3u + 2u] +
                        static_cast<uint64_t>(row) * rowBytes;
    const uint32_t lane = threadIdx.x;
    constexpr uint32_t kBlkBytes =
            (TYPE == kTypeQ4K)     ? kQ4K_BYTES
            : (TYPE == kTypeQ5K)   ? kQ5K_BYTES
            : (TYPE == kTypeIQ4XS) ? kIQ4XS_BYTES
                                   : kQ6K_BYTES;
    for (uint32_t p = p0; p < p1; p += TOK) {
        const uint32_t n = (p1 - p < TOK) ? (p1 - p) : TOK;
        float acc[TOK];
#pragma unroll
        for (uint32_t t = 0; t < TOK; ++t) acc[t] = 0.0f;
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
            const uint8_t *blkW = rp + static_cast<uint64_t>(b) * kBlkBytes;
            for (uint32_t t = 0; t < n; ++t) {
                const uint8_t *blkA =
                        xq + static_cast<uint64_t>(p + t) * xStride +
                        static_cast<uint64_t>(b) * kQ8K_STRIDE;
                kxQ8KBlockDot<TYPE>(acc[t], blkW, blkA, lane);
            }
        }
        // The cross-lane reduction MUST run on every lane (it is a
        // sync shuffle); only the store is lane-0.  Q4_K/Q5_K/IQ4XS
        // leave the complete dot in every lane (same asymmetry as
        // kQGemvKxQ8K_BatchDownR).
        for (uint32_t t = 0; t < n; ++t) {
            float v = acc[t];
            if (TYPE != kTypeQ4K && TYPE != kTypeQ5K &&
                TYPE != kTypeIQ4XS) {
                const float r1 = v + __shfl_xor_sync(0xffffffffu, v, 4);
                const float r2 =
                        r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
                v = r2 + __shfl_xor_sync(0xffffffffu, r2, 2);
            }
            if (lane == 0)
                out[static_cast<uint64_t>(ranked[p + t]) * outStride +
                    row] = v;
        }
    }
}

// act[t][j] = silu(gu[t][2ff + j] ... ) from the fused [n][2*ff] slab.
__global__ void kMoeSwiGLUToPacked(const float *__restrict__ gu,
                                   float *__restrict__ act, uint32_t ff) {
    const uint32_t t = blockIdx.y;
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ff) return;
    const float g = gu[static_cast<uint64_t>(t) * 2u * ff + j];
    const float u = gu[static_cast<uint64_t>(t) * 2u * ff + ff + j];
    act[static_cast<uint64_t>(t) * ff + j] =
            (g / (1.0f + __expf(-g))) * u;
}

// ---- Grouped PREFILL weighted accumulation over tokens -----------
// out[s][i] = sum_r wgt[s*eu + r] * down[(s*eu + r)][i] with the strict
// mul-then-add order of the CPU reference (no FMA contraction).
__global__ void kQ35MoeAccumScaledBatchT(
        const float *__restrict__ wgt, uint32_t eu,
        const float *__restrict__ down, float *__restrict__ out,
        uint32_t hidden) {
    const uint32_t s = blockIdx.x;
    float *o = out + static_cast<size_t>(s) * hidden;
    const float *w = wgt + static_cast<size_t>(s) * eu;
    const float *dn = down + static_cast<size_t>(s) * eu * hidden;
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        float acc = 0.0f;
#pragma unroll 8
        for (uint32_t r = 0; r < eu; ++r) {
            acc = __fadd_rn(
                    __fmul_rn(w[r],
                              dn[static_cast<size_t>(r) * hidden + i]),
                    acc);
        }
        o[i] = acc;
    }
}

// ---- Batched down GEMV with rank remap (Q4_K/Q6_K x Q8_K): resident
// slots 0..residentCount-1 are compacted; slot j's down rows land at
// out[rankOf[j]*outStride + row] in the full rank-major combined table
// (outStride == seqLen*H).  Same math as the GU kernel.
template<int TYPE>
__global__ void kQGemvKxQ8K_BatchDownR(const uint8_t *const *__restrict__ wq,
                                       const uint8_t *__restrict__ x,
                                       float *__restrict__ out,
                                       const int32_t *__restrict__ rankOf,
                                       uint32_t hidden,
                                       uint32_t blocksPerRow,
                                       uint32_t rowBytes,
                                       uint32_t xStrideBytes,
                                       uint32_t outStride) {
    const uint32_t j = blockIdx.y;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= hidden) return;
    const int32_t r0 = (rankOf != nullptr)
                               ? rankOf[j]
                               : static_cast<int32_t>(j);
    if (r0 < 0) return;
    const uint8_t *rp = wq[j * 3u + 2u] +
                        static_cast<uint64_t>(row) * rowBytes;
    const uint8_t *xr = x + static_cast<uint64_t>(j) * xStrideBytes;
    const uint32_t lane = threadIdx.x;
    float sumf = 0.0f;
    constexpr uint32_t kBlkBytes =
            (TYPE == kTypeQ4K)     ? kQ4K_BYTES
            : (TYPE == kTypeQ5K)   ? kQ5K_BYTES
            : (TYPE == kTypeIQ4XS) ? kIQ4XS_BYTES
                                   : kQ6K_BYTES;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * kBlkBytes;
        const uint8_t *blkA = xr + static_cast<uint64_t>(b) * kQ8K_STRIDE;
        kxQ8KBlockDot<TYPE>(sumf, blkW, blkA, lane);
    }
    if (TYPE == kTypeQ4K || TYPE == kTypeQ5K ||
        TYPE == kTypeIQ4XS) {
        if (lane == 0) {
            out[static_cast<uint64_t>(r0) * outStride + row] = sumf;
        }
    } else {
        const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
        const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
        const float rr = q + __shfl_xor_sync(0xffffffffu, q, 2);
        if (lane == 0) {
            out[static_cast<uint64_t>(r0) * outStride + row] = rr;
        }
    }
}

// ---- Batched 256-wide Q8_K quantize (per-rank slabs): rank r's
// [cols] fp32 activation -> q8k + r*strideBytes, block-by-block with
// the EXACT kQuantizeQ8K math (bit-exact with
// GGMLDequantize::quantizeQ8K that the CPU kernels use).  grid =
// (blocks, ranks), block = 256 threads.
__global__ void kQuantizeQ8K_Batch(const float *__restrict__ x,
                                   uint8_t *__restrict__ q8k,
                                   uint32_t cols, uint32_t ranks,
                                   uint32_t strideBytes) {
    const uint32_t r = blockIdx.y;
    const uint32_t b = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t e0 = b * 256u;
    const uint32_t n = (cols - e0) < 256u ? (cols - e0) : 256u;
    if (n == 0) return;
    uint8_t *blk = q8k + static_cast<uint64_t>(r) * strideBytes +
                   static_cast<uint64_t>(b) * kQ8K_STRIDE;
    const float *xs = x + static_cast<uint64_t>(r) * cols;
    float maxv = 0.0f, amax = 0.0f;
    for (uint32_t j = tid; j < n; j += blockDim.x) {
        const float v = xs[e0 + j];
        const float av = fabsf(v);
        if (av > amax) {
            amax = av;
            maxv = v;
        }
    }
    __shared__ float sMax[256];
    sMax[tid] = (amax > 0.0f) ? maxv : 0.0f;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a0 = sMax[tid], a1 = sMax[tid + s];
            if (fabsf(a1) > fabsf(a0)) sMax[tid] = a1;
        }
        __syncthreads();
    }
    const float max = sMax[0];
    // See kQuantizeQ8K: __fdiv_rn keeps the two IEEE divisions the CPU
    // quantizeQ8K performs (1/(-127/max) contracts to -max/127 with a
    // plain '/' and drifts the stored d by 1 ulp).
    const float iscale = (max != 0.0f) ? __fdiv_rn(-127.0f, max) : 0.0f;
    const float d = (max != 0.0f) ? __fdiv_rn(1.0f, iscale) : 0.0f;
    for (uint32_t j = tid; j < n; j += blockDim.x) {
        if (max == 0.0f) {
            blk[kQ8K_Q + j] = 0;
            continue;
        }
        int v = static_cast<int>(rintf(iscale * xs[e0 + j]));
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        blk[kQ8K_Q + j] = static_cast<uint8_t>(static_cast<int8_t>(v));
    }
    if (tid == 0) {
        *reinterpret_cast<float *>(blk + kQ8K_D) = d;
    }
    __syncthreads();
    for (uint32_t j = tid; j < 16; j += blockDim.x) {
        int sum = 0;
        for (uint32_t ii = 0; ii < 16; ++ii) {
            sum += static_cast<int>(
                    static_cast<int8_t>(blk[kQ8K_Q + j * 16u + ii]));
        }
        *reinterpret_cast<int16_t *>(blk + kQ8K_B + j * 2) =
                static_cast<int16_t>(sum);
    }
}
// down(r,s) over ranks ASCENDING, matching the CPU reference
// (computeQwen35MoE step 5: expertOut[i] += weight * down[i], with
// expertOut zero-initialized).  Plain mul+add (no fmaf), left-to-right
// association, ONE thread per element looping ranks serially so the
// accumulation ORDER is exactly the CPU's (0 + w0*d0, then + w1*d1, ...)
// and there is NO cross-rank race on o[i].  grid = 1 block (decode
// path is seqLen==1); 256 threads stride over hidden.
__global__ void kQ35MoeAccumScaledBatch(const float *__restrict__ wgt,
                                        uint32_t wgtStride,
                                        const float *__restrict__ down,
                                        float *__restrict__ out,
                                        uint32_t seqLen, uint32_t hidden,
                                        uint32_t ranks) {
    const uint32_t s = 0;// batched decode path is seqLen==1 only
    float *o = out + static_cast<size_t>(s) * hidden;
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        float acc = 0.0f;
#pragma unroll 32
        for (uint32_t r = 0; r < ranks; ++r) {
            const float g = wgt[static_cast<size_t>(s) * wgtStride + r];
            // NO FMA CONTRACTION: the CPU reference (ModelMoE.cpp
            // computeQwen35MoE step 5: expertOut[i] += weight * down[i])
            // is a strict mul-then-add.  nvcc's default -fmad=true
            // contracts `g*down + acc` into a single fmaf, which rounds
            // ONCE instead of twice — a 1-ULP difference per element
            // that compounds through the recurrent GDN state and flips
            // top-1 (root-cause of SequentialDecodeArgmaxAgrees failing
            // in BOTH cache paths while the CPU-expert fallback
            // passed; verified 2026-09-15: L0 moeOut identical in both,
            // first 1-ULP norm divergence at L1 fwd3).  __fmul_rn/
            // __fadd_rn are never contracted by the compiler.
            acc = __fadd_rn(
                    __fmul_rn(g, down[static_cast<size_t>(r) * hidden + i]),
                    acc);
        }
        o[i] = acc;
    }
}

// Debug twin of kQGemvQ8_0xQ8K: identical integer group-sum logic and
// identical float tree, but lane 0 additionally records per-block
// cumulative sums into trace[0..blocksPerRow), the block-0 t[g]
// intermediates into trace[blocksPerRow + g], and block-0's blk into
// trace[blocksPerRow + 8].  Used by the parity unit test to localize
// the FIRST block where the kernel's internal float tree diverges from
// the host recompute (which is bit-exact with the CPU AVX2 reference).
// Diagnostic only -- never launched in the inference path.
__global__ void kQGemvQ8_0xQ8KTrace(const uint8_t *__restrict__ w,
                                    const uint8_t *__restrict__ x,
                                    float *__restrict__ out,
                                    float *__restrict__ trace,
                                    uint32_t rows,
                                    uint32_t blocksPerRow,
                                    uint32_t rowBytes) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    const uint32_t lane = threadIdx.x;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * 34u;
        const uint8_t *blkA =
                x + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
        const float d =
                __half2float(*reinterpret_cast<const __half *>(blkW));
        const float d8 = *reinterpret_cast<const float *>(blkA);
        if (d8 == 0.0f) continue;
        const int wq = static_cast<int>(
                static_cast<int8_t>(blkW[2u + lane]));
        const int xq = static_cast<int>(
                static_cast<int8_t>(blkA[4u + lane]));
        int c = xq * wq;
        c += __shfl_xor_sync(0xffffffffu, c, 1);
        c += __shfl_xor_sync(0xffffffffu, c, 2);
        int s[8];
#pragma unroll
        for (uint32_t g = 0; g < 8; ++g) {
            s[g] = __shfl_sync(0xffffffffu, c, 4u * g);
        }
        if (lane == 0) {
            const float scale = d * d8;
            float t[8];
#pragma unroll
            for (uint32_t g = 0; g < 8; ++g) {
                t[g] = scale * static_cast<float>(s[g]);
            }
            const float u0 = t[0] + t[4];
            const float u1 = t[1] + t[5];
            const float u2 = t[2] + t[6];
            const float u3 = t[3] + t[7];
            const float v0 = u0 + u1;
            const float v1 = u2 + u3;
            const float blk = v0 + v1;
            sumf += blk;
            trace[b] = sumf;
            if (b == 0) {
                for (uint32_t g = 0; g < 8; ++g) {
                    trace[blocksPerRow + g] = t[g];
                }
                trace[blocksPerRow + 8u] = blk;
            }
            if (b == 1) {
                for (uint32_t g = 0; g < 8; ++g) {
                    trace[2u * blocksPerRow + g] =
                            static_cast<float>(s[g]);
                }
                trace[2u * blocksPerRow + 8u] = blk;
                trace[2u * blocksPerRow + 9u] = d;
                trace[2u * blocksPerRow + 10u] = d8;
            }
        }
    }
    if (lane == 0) out[row] = sumf;
}

// Per-lane integer contribution for ONE (ib32, l) group's 8 columns.
//
// MEMORY OPTIMIZATION (2026-09-27): the old body read the lane's 8
// activation bytes with EIGHT separate y8[...] byte loads at stride-8
// addresses. Across the warp that is a perfectly contiguous 256-byte
// block, but issued as 8 independent 1-byte transactions per lane, each
// dependent on the previous byte's load -> a serial ~8-deep L2 latency
// chain per (block, lane). Measured: the IQ2_S/IQ3_XXS/IQ3_S Q8K GEMV
// (kQGemvQ8K) ran at ~2000 cycles per block-iteration -- ~700x off the
// 448 GB/s memory floor (7B IQ2_S decode: gate+up 146.6ms, down 70.8ms
// for ~131 MB of weight bytes/token). Each lane's 8 columns are
// CONTIGUOUS (base = ib32*32 + l*8, 8-aligned), so we now load them as
// ONE 8-byte __ldg (coalesced: the 32 lanes cover the whole 256-byte
// block in one transaction set) and byte-extract.
// Forward declaration (the shared body is defined just below; the
// wrapper and the multi-row kernel both call it).
template<bool DP4A>
__device__ __forceinline__ int q8kLaneSumiV(
        int type, const uint8_t *__restrict__ w, uint64_t y8v,
        uint32_t ib32, uint32_t l,
        const uint8_t *__restrict__ gsm = nullptr);

// Thin wrapper: load the lane's activation word then run the shared
// body.  Bit-identical to the historical q8kLaneSumi.
template<bool DP4A = false>
__device__ __forceinline__ int q8kLaneSumi(
        int type, const uint8_t *__restrict__ w,
        const int8_t *__restrict__ y8, uint32_t ib32, uint32_t l,
        const uint8_t *__restrict__ gsm = nullptr) {
    // The lane's 8 columns are contiguous and 8-byte aligned (cudaMalloc
    // aligns q8k to >= 256; kQ8K_STRIDE=304 is a multiple of 8; ib32*32
    // and l*8 are multiples of 8).  __ldg routes through the read-only
    // cache so the streaming q8k block does not evict weight lines.
    const uint64_t y8v = __ldg(reinterpret_cast<const uint64_t *>(
            &y8[ib32 * 32u + l * 8u]));
    return q8kLaneSumiV<DP4A>(type, w, y8v, ib32, l, gsm);
}

// Body shared by q8kLaneSumi (which loads y8v itself) and the
// multi-row kernel (which loads one activation word per warp-lane and
// reuses it across all rows).  y8v is the lane's 8 Q8_K activation
// bytes at [ib32*32 + l*8].
template<bool DP4A>
__device__ __forceinline__ int q8kLaneSumiV(
        int type, const uint8_t *__restrict__ w, uint64_t y8v,
        uint32_t ib32, uint32_t l,
        const uint8_t *__restrict__ gsm) {
    if constexpr (DP4A) {
        // IQ2_S / IQ3_XXS / IQ3_S lane dot via __dp4a + the branchless
        // XOR-subtract sign application from the reference
        // vec_dot_iq3_xxs_q8_1 (__vcmpne4/__vsub4).  Bit-identical to
        // the scalar body: dp4a accumulates exact signed int8 products,
        // and the XOR-sub yields the exact signed grid byte (every grid
        // magnitude is <= 62 <= 127, so -g is int8-representable).
        // Replaces 8 scalar IMAD + 8 sign selects per (ib32, l) with
        // 2 dp4a + 4 mask/vector ops -- the ALU lever for the IQ decode
        // kernels (2026-10-11).  A/B: TINYCODER_IQ_DP4A.
        const int u0 = static_cast<int>(y8v & 0xFFFFFFFFull);
        const int u1 = static_cast<int>(y8v >> 32);
        if (type == kTypeIQ2S) {
            const uint8_t *qsv = w + 2;
            const uint8_t *qh = w + 66;
            const uint16_t gridIdx = static_cast<uint16_t>(
                    qsv[ib32 * 4u + l] |
                    ((qh[ib32] << (8 - 2u * l)) & 0x300u));
            const uint64_t gv =
                    gsm ? reinterpret_cast<const uint64_t *>(gsm)[gridIdx]
                        : c_iq2s_grid[gridIdx];
            const int signs =
                    static_cast<int>(qsv[32u + ib32 * 4u + l]);
            const int s0 = __vcmpne4(
                    static_cast<uint32_t>(signs) * 0x01010101u &
                            0x08040201u,
                    0);
            const int s1 = __vcmpne4(
                    static_cast<uint32_t>(signs) * 0x01010101u &
                            0x80402010u,
                    0);
            const int w0 = __vsub4(
                    static_cast<int>(gv & 0xFFFFFFFFull) ^ s0, s0);
            const int w1 = __vsub4(
                    static_cast<int>(gv >> 32) ^ s1, s1);
            return __dp4a(w1, u1, __dp4a(w0, u0, 0));
        }
        if (type == kTypeIQ2XS) {
            // IQ2_XS block (74 B): d(2) + qs[32 x uint16](64) +
            // scales[8].  Per (ib32, l): qval = qs16[ib32*4+l]; grid
            // index = low 9 bits (512-entry uint64 grid = 8 columns);
            // signs = ksigns[qval>>9].  ONE grid entry per group (vs
            // IQ3_XXS's two) -> the sign masks span bytes 0..7 through
            // the low/high halves, exactly as the scalar body.
            const uint16_t *qs16 =
                    reinterpret_cast<const uint16_t *>(w + 2);
            const uint16_t qval = qs16[ib32 * 4u + l];
            const uint64_t gv =
                    gsm ? reinterpret_cast<const uint64_t *>(
                                  gsm)[qval & 0x1FFu]
                        : c_iq2xs_grid[qval & 0x1FFu];
            const int signs =
                    static_cast<int>(c_ksigns_iq2xs[qval >> 9]);
            const int s0 = __vcmpne4(
                    static_cast<uint32_t>(signs) * 0x01010101u &
                            0x08040201u,
                    0);
            const int s1 = __vcmpne4(
                    static_cast<uint32_t>(signs) * 0x01010101u &
                            0x80402010u,
                    0);
            const int w0 = __vsub4(
                    static_cast<int>(gv & 0xFFFFFFFFull) ^ s0, s0);
            const int w1 = __vsub4(
                    static_cast<int>(gv >> 32) ^ s1, s1);
            return __dp4a(w1, u1, __dp4a(w0, u0, 0));
        }
        if (type == kTypeIQ3_XXS) {
            const uint8_t *q3 = w + 2;
            uint32_t aux32;
            std::memcpy(&aux32, q3 + 64u + 4u * ib32,
                        sizeof(uint32_t));
            const uint32_t g1 =
                    gsm ? reinterpret_cast<const uint32_t *>(
                                  gsm)[q3[8u * ib32 + 2u * l]]
                        : c_iq3xxs_grid[q3[8u * ib32 + 2u * l]];
            const uint32_t g2 =
                    gsm ? reinterpret_cast<const uint32_t *>(
                                  gsm)[q3[8u * ib32 + 2u * l + 1u]]
                        : c_iq3xxs_grid[q3[8u * ib32 + 2u * l + 1u]];
            const int signs = static_cast<int>(
                    c_ksigns_iq2xs[(aux32 >> (7u * l)) & 127u]);
            const int s0 = __vcmpne4(
                    static_cast<uint32_t>(signs) * 0x01010101u &
                            0x08040201u,
                    0);
            const int s1 = __vcmpne4(
                    static_cast<uint32_t>(signs) * 0x01010101u &
                            0x80402010u,
                    0);
            const int w0 = __vsub4(static_cast<int>(g1) ^ s0, s0);
            const int w1 = __vsub4(static_cast<int>(g2) ^ s1, s1);
            return __dp4a(w1, u1, __dp4a(w0, u0, 0));
        }
        // IQ3_S
        const uint8_t *qs = w + 2;
        const uint8_t *qh = w + 66;
        const uint8_t *signsP = w + 74;
        const uint32_t qhIdx = (ib32 >> 1u) * 2u + (ib32 & 1u);
        const uint32_t qsOff = (ib32 >> 1u) * 16u + (ib32 & 1u) * 8u;
        const uint32_t signOff =
                (ib32 >> 1u) * 8u + (ib32 & 1u) * 4u;
        const uint16_t gridIdx1 = static_cast<uint16_t>(
                qs[qsOff + 2u * l] |
                ((qh[qhIdx] << (8 - 2u * l)) & 256u));
        const uint16_t gridIdx2 = static_cast<uint16_t>(
                qs[qsOff + 2u * l + 1u] |
                ((qh[qhIdx] << (7 - 2u * l)) & 256u));
        const uint32_t g1 =
                gsm ? reinterpret_cast<const uint32_t *>(gsm)[gridIdx1]
                    : c_iq3s_grid[gridIdx1];
        const uint32_t g2 =
                gsm ? reinterpret_cast<const uint32_t *>(gsm)[gridIdx2]
                    : c_iq3s_grid[gridIdx2];
        const int signs = static_cast<int>(signsP[signOff + l]);
        const int s0 = __vcmpne4(static_cast<uint32_t>(signs) *
                                                 0x01010101u &
                                         0x08040201u,
                                 0);
        const int s1 = __vcmpne4(static_cast<uint32_t>(signs) *
                                                 0x01010101u &
                                         0x80402010u,
                                 0);
        const int w0 = __vsub4(static_cast<int>(g1) ^ s0, s0);
        const int w1 = __vsub4(static_cast<int>(g2) ^ s1, s1);
        return __dp4a(w1, u1, __dp4a(w0, u0, 0));
    } else {
        auto yb = [&](uint32_t j) -> int8_t {
            return static_cast<int8_t>((y8v >> (8u * j)) & 0xFFu);
        };
        (void) y8v;
        if (type == kTypeIQ2S) {
            const uint8_t *qsv = w + 2;
            const uint8_t *qh = w + 66;
            const uint16_t gridIdx = static_cast<uint16_t>(
                    qsv[ib32 * 4u + l] | ((qh[ib32] << (8 - 2u * l)) & 0x300u));
            // Native-word load + register byte-extraction.  The old body
            // dereferenced grid[j] for j=0..7 -- EIGHT separate ld.const.u8
            // with a per-lane (divergent) address.  A divergent constant
            // load costs one constant-cache port access PER UNIQUE ADDRESS
            // (up to 32 serialized cycles for a full warp).  That was the
            // real ~2000 cycles/block-iteration serialization point (the
            // 4-way block batching only hides latency, not this throughput
            // limit -- hence batching measured zero change).  One uint64
            // load + 8 register shifts is byte-identical (little-endian,
            // low byte = grid[0]) and costs 1 (or a few) constant accesses.
            const uint64_t gv =
                    gsm ? reinterpret_cast<const uint64_t *>(gsm)[gridIdx]
                        : c_iq2s_grid[gridIdx];
            const uint8_t signs = qsv[32u + ib32 * 4u + l];
            int sumi = 0;
            for (uint32_t j = 0; j < 8; ++j) {
                sumi += static_cast<int>(yb(j)) *
                        static_cast<int>((gv >> (8u * j)) & 0xFFu) *
                        ((signs & c_kmask_iq2xs[j]) ? -1 : 1);
            }
            return sumi;
        }
        if (type == kTypeIQ2XS) {
            const uint16_t *qs16 =
                    reinterpret_cast<const uint16_t *>(w + 2);
            const uint16_t qval = qs16[ib32 * 4u + l];
            const uint64_t gv =
                    gsm ? reinterpret_cast<const uint64_t *>(
                                  gsm)[qval & 0x1FFu]
                        : c_iq2xs_grid[qval & 0x1FFu];
            const uint8_t signs = c_ksigns_iq2xs[qval >> 9];
            int sumi = 0;
            for (uint32_t j = 0; j < 8; ++j) {
                sumi += static_cast<int>((gv >> (8u * j)) & 0xFFu) *
                        static_cast<int>(yb(j)) *
                        ((signs & c_kmask_iq2xs[j]) ? -1 : 1);
            }
            return sumi;
        }
        if (type == kTypeIQ3_XXS) {
            const uint8_t *q3 = w + 2;
            uint32_t aux32;
            std::memcpy(&aux32, q3 + 64u + 4u * ib32, sizeof(uint32_t));
            // Same native-word trick as IQ2_S: the old body did 8 divergent
            // byte loads on the two uint32 grid entries; now 2 divergent
            // uint32 loads + register byte-extraction.
            const uint32_t g1 = c_iq3xxs_grid[q3[8u * ib32 + 2u * l]];
            const uint32_t g2 = c_iq3xxs_grid[q3[8u * ib32 + 2u * l + 1u]];
            const uint8_t signs = c_ksigns_iq2xs[(aux32 >> (7u * l)) & 127u];
            int sumi = 0;
            for (uint32_t j = 0; j < 4; ++j) {
                sumi += static_cast<int>((g1 >> (8u * j)) & 0xFFu) *
                        static_cast<int>(yb(j)) *
                        ((signs & c_kmask_iq2xs[j + 0]) ? -1 : 1);
                sumi += static_cast<int>((g2 >> (8u * j)) & 0xFFu) *
                        static_cast<int>(yb(j + 4)) *
                        ((signs & c_kmask_iq2xs[j + 4]) ? -1 : 1);
            }
            return sumi;
        }
        // IQ3_S
        const uint8_t *qs = w + 2;
        const uint8_t *qh = w + 66;
        const uint8_t *signs = w + 74;
        const uint32_t qhIdx = (ib32 >> 1u) * 2u + (ib32 & 1u);
        const uint32_t qsOff = (ib32 >> 1u) * 16u + (ib32 & 1u) * 8u;
        const uint32_t signOff = (ib32 >> 1u) * 8u + (ib32 & 1u) * 4u;
        const uint16_t gridIdx1 = static_cast<uint16_t>(
                qs[qsOff + 2u * l] | ((qh[qhIdx] << (8 - 2u * l)) & 256u));
        const uint16_t gridIdx2 = static_cast<uint16_t>(
                qs[qsOff + 2u * l + 1u] | ((qh[qhIdx] << (7 - 2u * l)) & 256u));
        const uint32_t g1 =
                gsm ? reinterpret_cast<const uint32_t *>(gsm)[gridIdx1]
                    : c_iq3s_grid[gridIdx1];
        const uint32_t g2 =
                gsm ? reinterpret_cast<const uint32_t *>(gsm)[gridIdx2]
                    : c_iq3s_grid[gridIdx2];
        const uint8_t sbyte = signs[signOff + l];
        int sumi = 0;
        for (uint32_t j = 0; j < 4; ++j) {
            sumi += static_cast<int>((g1 >> (8u * j)) & 0xFFu) *
                    static_cast<int>(yb(j)) *
                    ((sbyte & c_kmask_iq2xs[j + 0]) ? -1 : 1);
            // grid2 is a uint32 whose bytes 0..3 are grid2[j] for j=0..3
            // (the CPU reference reads them via
            // reinterpret_cast<const uint8_t*>(&IQ3S_GRID[...])[j], i.e.
            // byte j of the little-endian word).  Shifting by (j+4) would
            // shift the 32-bit word out entirely (>>= 32..56) and silently
            // DROP all four grid2 contributions -- the attnO corruption.
            sumi += static_cast<int>((g2 >> (8u * j)) & 0xFFu) *
                    static_cast<int>(yb(j + 4)) *
                    ((sbyte & c_kmask_iq2xs[j + 4]) ? -1 : 1);
        }
        return sumi;
    }
}
