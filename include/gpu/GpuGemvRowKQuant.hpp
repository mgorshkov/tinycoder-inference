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

// GpuGemvRowKQuant.hpp -- warp-per-row Q4_K/Q6_K/Q5_K/Q2_K GEMV and 4xW variants.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// ---- Q4_K row GEMV (Q8_K activation), 4 ROWS PER WARP (2026-10-29).
// Direct analogue of kQGemvQ5KxQ8K_4xW.  The generic
// kQGemvKxQ8K<kTypeQ4K> runs one row per warp and only sub-lanes 0-7
// issue weight loads (24 of 32 lanes idle per instruction) AND serial-
// loads each 32-byte q's strip scalar (one LDG per column).  This
// kernel brings the two levers proven on the Q6_K/Q5_K 4xW kernels to
// the LARGEST quant class of the dense qwen35 models (Q4_K ~2.7 GB of
// Ornith-1.5-9B's 5.9 GB):
//   * all 32 lanes issue weight loads (8 sub-blocks x 4 rows per warp);
//   * the 32-column integer dot is 8x __dp4a instead of 32 byte loads.
// Bit-exact with kQGemvKxQ8K<kTypeQ4K>: the s32q integer sum is
// order-free and the float chain (volatile no-FMA dsv*ss8 / mnv*bi +
// d8*(ms-mns) epilogue) is verbatim, executed by every I, so each
// row's output is byte-identical to the generic kernel's lane-0 result.
// The terminal 8-lane shuffle tree stays inside each 4-row group and is
// the SAME association as the generic full-warp tree restricted to its
// 8 active lanes.  Requires rows % 4 == 0 (all real uses: dense qwen35
// attnO/ffnDown/ffnGate/ffnUp and the Q4_K LM head).
// Env gate TINYCODER_Q4K_4XW (default ON).
__global__ void kQGemvQ4KxQ8K_4xW(const uint8_t *__restrict__ w,
                                  const uint8_t *__restrict__ x,
                                  float *__restrict__ out, uint32_t rows,
                                  uint32_t blocksPerRow,
                                  uint32_t rowBytes) {
    const uint32_t warpRowBase = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t base = warpRowBase * 4u;
    if (base >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t I = lane & 7u;
    const uint32_t row = base + (lane >> 3u);
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    float sumf = 0.0f;
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        kxQ4KBlockDotVec(
                r0,
                rp + static_cast<uint64_t>(b + 0u) * kQ4K_BYTES,
                x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE, I);
        kxQ4KBlockDotVec(
                r1,
                rp + static_cast<uint64_t>(b + 1u) * kQ4K_BYTES,
                x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE, I);
        kxQ4KBlockDotVec(
                r2,
                rp + static_cast<uint64_t>(b + 2u) * kQ4K_BYTES,
                x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE, I);
        kxQ4KBlockDotVec(
                r3,
                rp + static_cast<uint64_t>(b + 3u) * kQ4K_BYTES,
                x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE, I);
        sumf = r0 + sumf;
        sumf = r1 + sumf;
        sumf = r2 + sumf;
        sumf = r3 + sumf;
    }
    for (; b < blocksPerRow; ++b) {
        kxQ4KBlockDotVec(
                sumf, rp + static_cast<uint64_t>(b) * kQ4K_BYTES,
                x + static_cast<uint64_t>(b) * kQ8K_STRIDE, I);
    }
    // 8-lane shuffle-tree per row within each 4-row lane group (xor
    // strides 4,1,2 stay inside the group's 8 lanes -- identical tree
    // and store predicate to the 1-row-per-warp kernel).
    const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
    const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
    const float rr = q + __shfl_xor_sync(0xffffffffu, q, 2);
    if (I == 0) out[row] = rr;
}

// ---- Q2_K float-dequant GEMV, 4 rows per warp (2026-09-30, A/B gate
// TINYCODER_Q2K_4XW=1).  Replicates the bandwidth shape proven by
// kQGemvQ6KxQ8K_4xW (~616 GB/s LM-head): all 32 lanes issue u32
// weight loads, 8 lanes per row, 4 independent row-streams per warp,
// NO shared memory / __syncthreads (both avoided because
// kQGemvQ2K2xW's smem staging + barrier serialized the block).
//
// Q2_K element map (bit-exact with dequantizeQ2_KBlock, verified):
//   col c -> q byte q[n*32 + (c%32) - 16*((c%32)>=16)], 2-bit shift
//   2*j, scale scales[n*8 + 2*j + ((c%32)>=16)], where
//   n = c/128, j = (c%128)/32.  Lane I (0..7) owns the 4 contiguous
//   cols qOfs = 4*I of each 32-col group: for qOfs<16 the 4 cols map
//   to q bytes [qOfs..qOfs+3] (scale sub 0), else [qOfs-16..qOfs-13]
//   (scale sub 1) -- ONE unaligned-safe u32 loads all 4 weight bytes.
//   Per-element float math is IDENTICAL to kQGemv<kTypeQ2K>:
//   fmaf(fmaf(dl, qv, -ml), x[cidx], acc).  The reduce/lane-partition
//   order differs, so results stay within the Q2_K float-path ~1e-7
//   tolerance of the CPU reference -- same class as the existing
//   1-row-per-warp kernel (validated by SequentialDecodeArgmaxAgrees
//   + ParisPromptLogitsAgree).
// MODE (P1 exp2 -> campaign 2026-10-10):
//   0 = guarded scalar x[] loads ((cidx<cols) ? x[cidx] : 0).
//   1 = EXACT: drop the guard when blocksPerRow*256 == cols (always
//       true for the default gate/up 8960x1536).  Bit-identical.
//   2 = EXACT + 128-bit activation load: the 4 k-cols of a group are
//       contiguous, so 4 scalar x[] loads -> 1 LDG.128 (issue-bound
//       GEMV, plan ch.4).  Bit-identical fmaf order.
__device__ __forceinline__ float qvOf(uint32_t w4, uint32_t j,
                                      uint32_t k) {
    return static_cast<float>(static_cast<int8_t>(
            (w4 >> (8u * k + 2u * j)) & 3u));
}
// ncu (2026-10-10): SM 44 % / DRAM 19 %, achieved occupancy 66.6 %,
// register-limited to 14 blocks/SM.  __launch_bounds__(64, 16) was
// tried to reclaim the 2 missing blocks: it caps registers at 64 and
// the resulting spills cost more than the extra occupancy buys
// (171.1 vs 173.1 tok/s) -> REVERTED.  Latency-bound at 44 % SM with
// no occupancy lever left.
template<int MODE = 0>
__global__ void kQGemvQ2KxW4(const uint8_t *__restrict__ w,
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

    // Element map (bit-exact with dequantizeQ2_KBlock): col c uses
    // q byte q[n*32 + (c%32)] (SAME offset for both 16-col halves of
    // a group; only the 2-bit shift 2*j and the scale sub differ) --
    // so the u32 load is at q + n*32 + qOfs for EVERY lane, and the
    // scale sub (0/1) is the only half-dependent index.
    const uint32_t qOfs = 4u * I;
    const uint32_t qb = qOfs;
    const uint32_t isub = (qOfs < 16u) ? 0u : 1u;

    float acc = 0.0f;
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        // MODE 3: one accumulator per (block, n-half) -> 8 independent
        // FMA chains of 16 instead of 4 chains of 32.  The dmon decode
        // profile (sm 93% busy / mem 28% -- memory idle 72%) says the
        // SMs are stalled, not memory-bound; the serial-FMA latency per
        // chain is the only structural stall left, so halve it.
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
#pragma unroll
        for (uint32_t bb = 0; bb < 4; ++bb) {
            const uint8_t *crp =
                    rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
            float df, dmf;
            uint32_t scw[2][2];
            if constexpr (MODE >= 4) {
                // MODE 4/5: packed weight-side loads.  ONE u32 covers
                // d|dm (the f16 pair at +80) and one u32 covers 4 scale
                // bytes, vs 2 half loads + 4 byte loads before -- the
                // ncu LG-throttle stall (38.6 % of the issue gap on
                // this kernel) is an instruction-COUNT problem on the
                // LSU queue, so cut load instructions.  All extracted
                // values are identical -> results byte-identical.
                uint32_t dm4;
                std::memcpy(&dm4, crp + 80, sizeof(uint32_t));
                df = __half2float(__ushort_as_half(
                        static_cast<unsigned short>(dm4 & 0xFFFFu)));
                dmf = __half2float(__ushort_as_half(
                        static_cast<unsigned short>(dm4 >> 16u)));
                std::memcpy(&scw[0][0], crp, sizeof(uint32_t));
                std::memcpy(&scw[0][1], crp + 4, sizeof(uint32_t));
                std::memcpy(&scw[1][0], crp + 8, sizeof(uint32_t));
                std::memcpy(&scw[1][1], crp + 12, sizeof(uint32_t));
            } else {
                df = __half2float(
                        *reinterpret_cast<const __half *>(crp + 80));
                dmf = __half2float(
                        *reinterpret_cast<const __half *>(crp + 82));
            }
            const uint8_t *sc = crp;
            const uint8_t *q = crp + 16;
#pragma unroll
            for (uint32_t n = 0; n < 2; ++n) {
                float &slot =
                        (MODE == 3) ? (n == 0u ? (bb == 0u   ? a0
                                                  : bb == 1u ? a2
                                                  : bb == 2u ? a4
                                                             : a6)
                                               : (bb == 0u   ? a1
                                                  : bb == 1u ? a3
                                                  : bb == 2u ? a5
                                                             : a7))
                                    : ((bb == 0u)   ? a0
                                       : (bb == 1u) ? a1
                                       : (bb == 2u) ? a2
                                                    : a3);
                uint32_t w4;
                if constexpr (MODE == 6) {
                    // MODE 6: weight loads bypass L1 (ld.global.cg,
                    // L2-only).  The ncu profile shows L1/TEX at 73-75 %
                    // on this kernel with DRAM at only 19 % -- the
                    // zero-reuse weight stream occupies the L1 pipe the
                    // broadcast x reads need.  504-B rows are 4-aligned,
                    // so u32 __ldcg is legal.  Same bytes -> byte-identical.
                    w4 = __ldcg(reinterpret_cast<const uint32_t *>(
                            q + n * 32u + qb));
                } else {
                    std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                }
                uint32_t scvA[4];
                if constexpr (MODE >= 4) {
                    // scv[j] == sc[n*8 + 2j + isub], extracted from the
                    // two packed u32 words of this n-half.
                    const uint32_t A = scw[n][0], Bv = scw[n][1];
                    if (isub == 0u) {
                        scvA[0] = A & 0xFFu;
                        scvA[1] = (A >> 16) & 0xFFu;
                        scvA[2] = Bv & 0xFFu;
                        scvA[3] = (Bv >> 16) & 0xFFu;
                    } else {
                        scvA[0] = (A >> 8) & 0xFFu;
                        scvA[1] = (A >> 24) & 0xFFu;
                        scvA[2] = (Bv >> 8) & 0xFFu;
                        scvA[3] = (Bv >> 24) & 0xFFu;
                    }
                }
                const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint8_t scv = (MODE >= 4)
                                                ? static_cast<uint8_t>(scvA[j])
                                        : (MODE == 6)
                                                ? static_cast<uint8_t>(__ldcg(
                                                          reinterpret_cast<const char *>(
                                                                  sc + n * 8u + 2u * j +
                                                                  isub)))
                                                : sc[n * 8u + 2u * j + isub];
                    const float dl =
                            df * static_cast<float>(scv & 0xFu);
                    const float ml =
                            dmf * static_cast<float>(scv >> 4u);
                    const uint32_t cbase = ebase + j * 32u + qOfs;
                    if constexpr (MODE == 2 || MODE == 5) {
                        // MODE 2: the 4 k-cols are CONTIGUOUS (cbase
                        // is a multiple of 4), so the 4 scalar x[]
                        // loads become ONE LDG.128.  Attack on the
                        // issue-bound Q2_K GEMV (SM ~97% busy / DRAM
                        // ~20%, plan ch.4): activation-load issue slots
                        // drop 4x.  Same 4 fmaf in the same order ->
                        // parity byte-identical.
                        const float4 xv = *reinterpret_cast<const float4 *>(
                                x + cbase);
                        const float m0 = fmaf(dl, qvOf(w4, j, 0), -ml);
                        const float m1 = fmaf(dl, qvOf(w4, j, 1), -ml);
                        const float m2 = fmaf(dl, qvOf(w4, j, 2), -ml);
                        const float m3 = fmaf(dl, qvOf(w4, j, 3), -ml);
                        slot = fmaf(m0, xv.x, slot);
                        slot = fmaf(m1, xv.y, slot);
                        slot = fmaf(m2, xv.z, slot);
                        slot = fmaf(m3, xv.w, slot);
                    } else {
#pragma unroll
                        for (uint32_t k = 0; k < 4; ++k) {
                            const float qv = qvOf(w4, j, k);
                            const uint32_t cidx = cbase + k;
                            const float xv = (MODE == 1 || MODE == 6 ||
                                              MODE >= 4)
                                                     ? x[cidx]
                                                     : ((cidx < cols)
                                                                ? x[cidx]
                                                                : 0.0f);
                            slot = fmaf(fmaf(dl, qv, -ml), xv, slot);
                        }
                    }
                }
            }
        }
        acc += a0;
        acc += a1;
        acc += a2;
        acc += a3;
        if constexpr (MODE == 3) {
            acc += a4;
            acc += a5;
            acc += a6;
            acc += a7;
        }
        rp += 4u * kQ2K_BYTES;
    }
    for (; b < blocksPerRow; ++b) {
        float df, dmf;
        uint32_t scw[2][2];
        if constexpr (MODE >= 4) {
            // Packed weight loads -- see the 4-block loop above.
            uint32_t dm4;
            std::memcpy(&dm4, rp + 80, sizeof(uint32_t));
            df = __half2float(__ushort_as_half(
                    static_cast<unsigned short>(dm4 & 0xFFFFu)));
            dmf = __half2float(__ushort_as_half(
                    static_cast<unsigned short>(dm4 >> 16u)));
            std::memcpy(&scw[0][0], rp, sizeof(uint32_t));
            std::memcpy(&scw[0][1], rp + 4, sizeof(uint32_t));
            std::memcpy(&scw[1][0], rp + 8, sizeof(uint32_t));
            std::memcpy(&scw[1][1], rp + 12, sizeof(uint32_t));
        } else if (MODE == 6) {
            uint32_t dm4 = __ldcg(
                    reinterpret_cast<const uint32_t *>(rp + 80));
            df = __half2float(__ushort_as_half(
                    static_cast<unsigned short>(dm4 & 0xFFFFu)));
            dmf = __half2float(__ushort_as_half(
                    static_cast<unsigned short>(dm4 >> 16u)));
        } else {
            df = __half2float(
                    *reinterpret_cast<const __half *>(rp + 80));
            dmf = __half2float(
                    *reinterpret_cast<const __half *>(rp + 82));
        }
        const uint8_t *sc = rp;
        const uint8_t *q = rp + 16;
#pragma unroll
        for (uint32_t n = 0; n < 2; ++n) {
            uint32_t w4;
            if constexpr (MODE == 6) {
                w4 = __ldcg(reinterpret_cast<const uint32_t *>(
                        q + n * 32u + qb));
            } else {
                std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
            }
            uint32_t scvA[4];
            if constexpr (MODE >= 4) {
                const uint32_t A = scw[n][0], Bv = scw[n][1];
                if (isub == 0u) {
                    scvA[0] = A & 0xFFu;
                    scvA[1] = (A >> 16) & 0xFFu;
                    scvA[2] = Bv & 0xFFu;
                    scvA[3] = (Bv >> 16) & 0xFFu;
                } else {
                    scvA[0] = (A >> 8) & 0xFFu;
                    scvA[1] = (A >> 24) & 0xFFu;
                    scvA[2] = (Bv >> 8) & 0xFFu;
                    scvA[3] = (Bv >> 24) & 0xFFu;
                }
            }
            const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                const uint8_t scv = (MODE >= 4)
                                            ? static_cast<uint8_t>(scvA[j])
                                    : (MODE == 6)
                                            ? static_cast<uint8_t>(__ldcg(
                                                      reinterpret_cast<const char *>(
                                                              sc + n * 8u + 2u * j + isub)))
                                            : sc[n * 8u + 2u * j + isub];
                const float dl = df * static_cast<float>(scv & 0xFu);
                const float ml = dmf * static_cast<float>(scv >> 4u);
                const uint32_t cbase = ebase + j * 32u + qOfs;
                if constexpr (MODE == 2 || MODE == 5) {
                    // One LDG.128 instead of 4 scalar x[] loads
                    // (identical fmaf order; parity byte-identical).
                    const float4 xv = *reinterpret_cast<const float4 *>(
                            x + cbase);
                    const float m0 = fmaf(dl, qvOf(w4, j, 0), -ml);
                    const float m1 = fmaf(dl, qvOf(w4, j, 1), -ml);
                    const float m2 = fmaf(dl, qvOf(w4, j, 2), -ml);
                    const float m3 = fmaf(dl, qvOf(w4, j, 3), -ml);
                    acc = fmaf(m0, xv.x, acc);
                    acc = fmaf(m1, xv.y, acc);
                    acc = fmaf(m2, xv.z, acc);
                    acc = fmaf(m3, xv.w, acc);
                } else {
#pragma unroll
                    for (uint32_t k = 0; k < 4; ++k) {
                        const float qv = qvOf(w4, j, k);
                        const uint32_t cidx = cbase + k;
                        const float xv = (MODE == 1 || MODE == 6 ||
                                          MODE >= 4)
                                                 ? x[cidx]
                                                 : ((cidx < cols)
                                                            ? x[cidx]
                                                            : 0.0f);
                        acc = fmaf(fmaf(dl, qv, -ml), xv, acc);
                    }
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

// ---- REPACKED 16-byte-lane Q2_K GEMV (2026-10-10, §9.19,
// TINYCODER_Q2K_RPACK=1).  The ncu profile shows every Q2_K GEMV
// L1-pipe bound (73-75 %) with 4-B-per-lane weight loads.  The repack
// (kRepackQ2K, run once per matrix) gives lane I of each 8-lane row
// group 16 CONTIGUOUS bytes per block -- 8 B q (h0@4I, h1@32+4I) + 8 B
// scales (sc[2j+isub], sc[8+2j+isub]) -- plus a broadcast 4 B dm, in
// 136-B 8-aligned blocks.  The GEMV then loads u64 q + u64 scales +
// u32 dm per block (3 weight loads vs 12), each warp-instruction
// covering 64 contiguous bytes per row.  Column mapping, fmaf order
// and reduce tree are IDENTICAL to kQGemvQ2KxW4<1> -> byte-identical.
__global__ void kRepackQ2K(const uint8_t *__restrict__ w,
                           uint8_t *__restrict__ dst,
                           uint32_t rowBytes, uint32_t blocksPerRow) {
    const uint32_t row = blockIdx.x;
    const uint32_t b = blockIdx.y;
    const uint32_t I = threadIdx.x & 7u;
    const uint32_t isub = I >> 2u;// (4I < 16) <=> I < 4
    const uint8_t *crp = w + static_cast<uint64_t>(row) * rowBytes +
                         static_cast<uint64_t>(b) * kQ2K_BYTES;
    uint8_t *dblk =
            dst + (static_cast<uint64_t>(row) * blocksPerRow + b) * 136u;
    std::memcpy(dblk + 16u * I, crp + 16u + 4u * I, 4);           // h0 q bytes
    std::memcpy(dblk + 16u * I + 4u, crp + 16u + 32u + 4u * I, 4);// h1
    for (uint32_t m = 0; m < 8; ++m) {
        dblk[16u * I + 8u + m] =
                crp[(m < 4u) ? (2u * m + isub)
                             : (8u + 2u * (m - 4u) + isub)];
    }
    if (threadIdx.x == 0) std::memcpy(dblk + 128u, crp + 80u, 4);// dm
}

template<bool EXACT = false>
__global__ void __launch_bounds__(64) kQGemvQ2KxW4_RP(
        const uint8_t *__restrict__ w, const float *__restrict__ x,
        float *__restrict__ out, uint32_t rows, uint32_t cols,
        uint32_t blocksPerRow) {
    constexpr uint32_t kRP_BLK = 136u;
    const uint32_t warpRowBase = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t base = warpRowBase * 4u;
    if (base >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t I = lane & 7u;
    const uint32_t row = base + (lane >> 3u);
    const uint8_t *rp = w + static_cast<uint64_t>(row) * blocksPerRow *
                                    kRP_BLK;
    const uint32_t qOfs = 4u * I;
    const uint32_t isub = I >> 2u;
    (void) isub;
    float acc = 0.0f;
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
        for (uint32_t bb = 0; bb < 4; ++bb) {
            const uint8_t *blk = rp + static_cast<uint64_t>(b + bb) *
                                              kRP_BLK;
            // 3 weight loads per block (vs 12): u64 q, u64 scales,
            // u32 dm (broadcast across the row's 8 lanes).
            unsigned long long q8, s8;
            std::memcpy(&q8, blk + 16u * I, sizeof(unsigned long long));
            std::memcpy(&s8, blk + 16u * I + 8u,
                        sizeof(unsigned long long));
            uint32_t dm4;
            std::memcpy(&dm4, blk + 128u, sizeof(uint32_t));
            const float df = __half2float(__ushort_as_half(
                    static_cast<unsigned short>(dm4 & 0xFFFFu)));
            const float dmf = __half2float(__ushort_as_half(
                    static_cast<unsigned short>(dm4 >> 16u)));
            float &slot = (bb == 0u)   ? a0
                          : (bb == 1u) ? a1
                          : (bb == 2u) ? a2
                                       : a3;
#pragma unroll
            for (uint32_t n = 0; n < 2; ++n) {
                const uint32_t w4 = static_cast<uint32_t>(
                        (n == 0u) ? q8 : (q8 >> 32u));
                const uint32_t ebase = (b + bb) * 256u + n * 128u;
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint32_t scv =
                            (s8 >> (8u * (4u * n + j))) & 0xFFu;
                    const float dl =
                            df * static_cast<float>(scv & 0xFu);
                    const float ml =
                            dmf * static_cast<float>(scv >> 4u);
                    const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                    for (uint32_t k = 0; k < 4; ++k) {
                        const float qv = qvOf(w4, j, k);
                        const uint32_t cidx = cbase + k;
                        const float xv = EXACT ? x[cidx]
                                               : ((cidx < cols)
                                                          ? x[cidx]
                                                          : 0.0f);
                        slot = fmaf(fmaf(dl, qv, -ml), xv, slot);
                    }
                }
            }
        }
        acc += a0;
        acc += a1;
        acc += a2;
        acc += a3;
    }
    for (; b < blocksPerRow; ++b) {
        const uint8_t *blk = rp + static_cast<uint64_t>(b) * kRP_BLK;
        unsigned long long q8, s8;
        std::memcpy(&q8, blk + 16u * I, sizeof(unsigned long long));
        std::memcpy(&s8, blk + 16u * I + 8u, sizeof(unsigned long long));
        uint32_t dm4;
        std::memcpy(&dm4, blk + 128u, sizeof(uint32_t));
        const float df = __half2float(__ushort_as_half(
                static_cast<unsigned short>(dm4 & 0xFFFFu)));
        const float dmf = __half2float(__ushort_as_half(
                static_cast<unsigned short>(dm4 >> 16u)));
#pragma unroll
        for (uint32_t n = 0; n < 2; ++n) {
            const uint32_t w4 =
                    static_cast<uint32_t>((n == 0u) ? q8 : (q8 >> 32u));
            const uint32_t ebase = b * 256u + n * 128u;
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t scv = (s8 >> (8u * (4u * n + j))) & 0xFFu;
                const float dl = df * static_cast<float>(scv & 0xFu);
                const float ml = dmf * static_cast<float>(scv >> 4u);
                const uint32_t cbase = ebase + j * 32u + qOfs;
#pragma unroll
                for (uint32_t k = 0; k < 4; ++k) {
                    const float qv = qvOf(w4, j, k);
                    const uint32_t cidx = cbase + k;
                    const float xv = EXACT ? x[cidx]
                                           : ((cidx < cols) ? x[cidx]
                                                            : 0.0f);
                    acc = fmaf(fmaf(dl, qv, -ml), xv, acc);
                }
            }
        }
    }
    const float p = acc + __shfl_xor_sync(0xffffffffu, acc, 4);
    const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
    const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
    if (I == 0) out[row] = rr;
}

// ---- Fused-residual GEMV epilogues (2026-10-09, per-layer launch/
// drain campaign; opt-in via TINYCODER_FUSE_ATTNO_RESID and
// TINYCODER_FUSE_DOWN_RESID, DEFAULT OFF).  These are EXACT copies of
// kQGemvQ2KxW4<EXACT> / kQGemv<kTypeQ3K,EXACT> that additionally fold
// the following kAddResidual into the GEMV epilogue, deleting one
// kernel launch per FFN/attn block:
//   * decode attnO : s.attnProj = Wo*attnOut ; s.hidden += s.attnProj
//   * decode down  : s.ffnOut  = Wd*gate    ; s.hidden += s.ffnOut
// The dot math, per-lane partition and reduce tree are IDENTICAL to
// the base kernels, so out[row] is written byte-for-byte as before
// (out is NOT dead -- kAddResidual reads src=out); the residual add
// is the same `acc[row] + rr` float add kAddResidual performs, hence
// the residual stream is byte-identical too (parity is asserted by the
// usual greedy-token probe: deterministic argmax => identical stream).
template<bool EXACT = false>
__global__ void kQGemvQ2KxW4Resid(const uint8_t *__restrict__ w,
                                  const float *__restrict__ x,
                                  float *__restrict__ out,
                                  float *__restrict__ acc,
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

    float accv = 0.0f;
    uint32_t b = 0;
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
                                    EXACT ? x[cidx]
                                          : ((cidx < cols) ? x[cidx]
                                                           : 0.0f),
                                    slot);
                    }
                }
            }
        }
        accv += a0;
        accv += a1;
        accv += a2;
        accv += a3;
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
                    accv = fmaf(fmaf(dl, qv, -ml),
                                EXACT ? x[cidx]
                                      : ((cidx < cols) ? x[cidx] : 0.0f),
                                accv);
                }
            }
        }
        rp += kQ2K_BYTES;
    }
    // 8-lane shuffle tree per row -- IDENTICAL to kQGemvQ2KxW4.
    const float p = accv + __shfl_xor_sync(0xffffffffu, accv, 4);
    const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
    const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
    if (I == 0) {
        out[row] = rr;
        acc[row] = acc[row] + rr;
    }
}

// Q3_K fused-residual (ffnDown).  1-row-per-warp, 32-lane reduce --
// IDENTICAL to kQGemv<kTypeQ3K,EXACT>; only the epilogue adds the
// residual.  See the Q2_KResid comment above for the parity argument.
template<bool EXACT = false>
__global__ void kQGemvQ3KResid(const uint8_t *__restrict__ w,
                               const float *__restrict__ x,
                               float *__restrict__ out,
                               float *__restrict__ acc, uint32_t rows,
                               uint32_t cols, uint32_t rowBytes,
                               uint32_t blocksPerRow) {
    uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    float accv = 0.0f;
    const uint32_t lane = threadIdx.x;
    const uint32_t sub = lane / 16u;
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
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
                    const uint32_t cidx = (b + bb) * 256u +
                                          half * 128u + jj * 32u + lane;
                    slot = fmaf(dl * qv,
                                EXACT ? x[cidx]
                                      : ((cidx < cols) ? x[cidx] : 0.0f),
                                slot);
                }
            }
        }
        accv += a0;
        accv += a1;
        accv += a2;
        accv += a3;
        rp += 4u * kQ3K_BYTES;
    }
    for (; b < blocksPerRow; ++b) {
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
                        b * 256u + half * 128u + jj * 32u + lane;
                accv = fmaf(dl * qv,
                            EXACT ? x[cidx]
                                  : ((cidx < cols) ? x[cidx] : 0.0f),
                            accv);
            }
        }
        rp += kQ3K_BYTES;
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        accv += __shfl_xor_sync(0xffffffffu, accv, off);
    }
    if (lane == 0) {
        out[row] = accv;
        acc[row] = acc[row] + accv;
    }
}

// ---- Q3_K 4-rows-per-warp decode GEMV (2026-10-06, P1 bandwidth
// campaign; A/B gate TINYCODER_Q3K_4XW=0 disables, DEFAULT ON).  The
// 1-row-per-warp kQGemv<kTypeQ3K> measures ~103 GB/s on the FFN down
// projection (rows=1536), while the byte-shape-identical Q2_K 4xW
// kernel reaches ~145 GB/s on the gate/up (rows=8960).  The only
// structural difference was the warp geometry: this kernel gives all
// 32 lanes a u32 weight load (8 lanes/row, 4 independent row-streams
// per warp), NO shared memory / __syncthreads, mirroring
// kQGemvQ2KxW4 exactly.
//
// Q3_K layout (bit-exact with the kQGemv<kTypeQ3K> branch above):
//   hm   (32 B at block+0):  high-bit mask, bit 1<<jj for the (half,
//                            lane) position with qv = low2 - (bit?0:4)
//   q    (64 B at block+32): 2-bit quants, q[half*32 + lane]
//   scales (12 B at block+96): repacked int8 scales, decoded via the
//                            same aux[4] shuffle to sc16[16]
//   d    (f16 at block+108)
// Lane I (0..7) owns 4 contiguous columns per 32-col group:
//   qOfs = 4*I;  sub = (qOfs < 16) ? 0 : 1.
// The hmask bit (1<<jj) is the SAME for all 4 columns of a lane's
// slice (it indexes jj, not the column), so one hm byte load per
// (block, half) suffices.  Per-element math is IDENTICAL to the
// 1-row-per-warp kernel (same int8 sub-32 scale, same hmask fold,
// same fmaf); only the lane partition/reduce order differs, so results
// stay within the Q3_K float-path tolerance (validated by
// SequentialDecodeArgmaxAgrees + ParisPromptLogitsAgree).
__global__ void kQGemvQ3KxW4(const uint8_t *__restrict__ w,
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
    const uint32_t isub = (qOfs < 16u) ? 0u : 1u;

    float acc = 0.0f;
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
        for (uint32_t bb = 0; bb < 4; ++bb) {
            const uint8_t *crp =
                    rp + static_cast<uint64_t>(bb) * kQ3K_BYTES;
            const float df = __half2float(
                    *reinterpret_cast<const __half *>(crp + 108));
            const uint8_t *hm = crp;
            const uint8_t *q = crp + 32;
            const uint8_t *scales = crp + 96;
            uint32_t aux[4];
            std::memcpy(&aux[0], scales, 12);
            const uint32_t tmp = aux[2];
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
            float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                    : (bb == 2u)       ? a2
                                                       : a3;
#pragma unroll
            for (uint32_t half = 0; half < 2; ++half) {
                // ONE u32 load fetches the lane's 4 contiguous 2-bit
                // quants (cols qOfs..qOfs+3) -- the Q2_K 4xW shape.
                // The hmask byte is per-COLUMN (hm[qOfs+k]); the mask
                // bit (1<<jj) indexes jj, not the column.
                uint32_t w4;
                std::memcpy(&w4, q + half * 32u + qOfs,
                            sizeof(uint32_t));
                const uint32_t ebase = (b + bb) * 256u + half * 128u;
#pragma unroll
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    const float dl = df * static_cast<float>(
                                                  sc16[half * 8u + jj * 2u + isub] - 32);
                    const uint32_t maskBit = 1u << (half * 4u + jj);
                    const uint32_t cbase = ebase + jj * 32u + qOfs;
#pragma unroll
                    for (uint32_t k = 0; k < 4; ++k) {
                        const float qv = static_cast<float>(
                                static_cast<int8_t>(
                                        ((w4 >> (8u * k + 2u * jj)) & 3u) -
                                        ((hm[qOfs + k] & maskBit) ? 0 : 4)));
                        const uint32_t cidx = cbase + k;
                        slot = fmaf(dl * qv,
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
        rp += 4u * kQ3K_BYTES;
    }
    for (; b < blocksPerRow; ++b) {
        const float df = __half2float(
                *reinterpret_cast<const __half *>(rp + 108));
        const uint8_t *hm = rp;
        const uint8_t *q = rp + 32;
        const uint8_t *scales = rp + 96;
        uint32_t aux[4];
        std::memcpy(&aux[0], scales, 12);
        const uint32_t tmp = aux[2];
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
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
            uint32_t w4;
            std::memcpy(&w4, q + half * 32u + qOfs, sizeof(uint32_t));
            const uint32_t ebase = b * 256u + half * 128u;
#pragma unroll
            for (uint32_t jj = 0; jj < 4; ++jj) {
                const float dl = df * static_cast<float>(
                                              sc16[half * 8u + jj * 2u + isub] - 32);
                const uint32_t maskBit = 1u << (half * 4u + jj);
                const uint32_t cbase = ebase + jj * 32u + qOfs;
#pragma unroll
                for (uint32_t k = 0; k < 4; ++k) {
                    const float qv = static_cast<float>(
                            static_cast<int8_t>(
                                    ((w4 >> (8u * k + 2u * jj)) & 3u) -
                                    ((hm[qOfs + k] & maskBit) ? 0 : 4)));
                    const uint32_t cidx = cbase + k;
                    acc = fmaf(dl * qv,
                               (cidx < cols) ? x[cidx] : 0.0f, acc);
                }
            }
        }
        rp += kQ3K_BYTES;
    }
    const float p = acc + __shfl_xor_sync(0xffffffffu, acc, 4);
    const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
    const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
    if (I == 0) out[row] = rr;
}

// ---- Q2_K MULTI-ROW-PER-WARP decode GEMV (2026-09-30, A/B gate
// TINYCODER_Q2K_NM=<RPB>, DEFAULT OFF).  The 4xW kernel gives every
// lane ONE row's 4-col slice; the FFN gate/up measured ~78 GB/s vs
// the ~616 GB/s Q6_K LM-head ceiling, i.e. the FFN is limited by
// MEMORY-LATENCY-BOUND occupancy (only ~8 warps/SM resident at ~90
// regs) rather than by DRAM bandwidth.  The Q6_K LM head reaches
// 616 GB/s with the SAME 4xW geometry purely because it has 4748
// full blocks (~68 per SM) to fill every latency-hiding slot.
//
// This kernel raises the per-WARP memory-level parallelism instead:
// the 4 lane groups each own R = RPB/4 independent rows, so every
// lane issues R interleaved weight streams (4 blocks x R rows per
// outer iteration) with NO shared memory and NO __syncthreads --
// exactly the pattern that wins on the FFN's tiny 280-block grid
// (68 SMs x ~4 resident blocks).  RPB must divide `rows` (8960 and
// 1536 both do: RPB in {8,16,32}).
//
// Per-element float math is IDENTICAL to kQGemvQ2KxW4 (same u32
// q-byte load, same 2-bit shift, same scale sub-index, same fmaf
// chain and in-order 4-block fold); only the per-lane accumulation
// state is replicated R-fold.  The 8-lane shuffle tree per row is
// unchanged, so results stay within the Q2_K float-path ~1e-7
// tolerance (validated by SequentialDecodeArgmaxAgrees +
// ParisPromptLogitsAgree).
template<uint32_t RPB>
__global__ void kQGemvQ2KxW4_Nm(const uint8_t *__restrict__ w,
                                const float *__restrict__ x,
                                float *__restrict__ out, uint32_t rows,
                                uint32_t cols, uint32_t rowBytes,
                                uint32_t blocksPerRow) {
    constexpr uint32_t R = RPB / 4u;// rows owned by each lane group
    const uint32_t warpRowBase =
            blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t base = warpRowBase * RPB;
    if (base >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t g = lane >> 3u;// 0..3 lane group
    const uint32_t I = lane & 7u; // sub-lane within the row
    const uint32_t qOfs = 4u * I;
    const uint32_t qb = qOfs;
    const uint32_t isub = (qOfs < 16u) ? 0u : 1u;
    const uint32_t row0 = base + g * R;

    float acc[R];
    const uint8_t *rp[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        acc[r] = 0.0f;
        rp[r] = w + static_cast<uint64_t>(row0 + r) * rowBytes;
    }

    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
            for (uint32_t bb = 0; bb < 4; ++bb) {
                const uint8_t *crp =
                        rp[r] + static_cast<uint64_t>(bb) * kQ2K_BYTES;
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
                    std::memcpy(&w4, q + n * 32u + qb,
                                sizeof(uint32_t));
                    const uint32_t ebase =
                            (b + bb) * 256u + n * 128u;
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
                                            (w4 >> (8u * k + 2u * j)) &
                                            3u));
                            const uint32_t cidx = cbase + k;
                            slot = fmaf(
                                    fmaf(dl, qv, -ml),
                                    (cidx < cols) ? x[cidx] : 0.0f,
                                    slot);
                        }
                    }
                }
            }
            acc[r] += a0;
            acc[r] += a1;
            acc[r] += a2;
            acc[r] += a3;
            rp[r] += 4u * kQ2K_BYTES;
        }
    }
    for (; b < blocksPerRow; ++b) {
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) {
            const float df = __half2float(
                    *reinterpret_cast<const __half *>(rp[r] + 80));
            const float dmf = __half2float(
                    *reinterpret_cast<const __half *>(rp[r] + 82));
            const uint8_t *sc = rp[r];
            const uint8_t *q = rp[r] + 16;
#pragma unroll
            for (uint32_t n = 0; n < 2; ++n) {
                uint32_t w4;
                std::memcpy(&w4, q + n * 32u + qb, sizeof(uint32_t));
                const uint32_t ebase = b * 256u + n * 128u;
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
                                        (w4 >> (8u * k + 2u * j)) &
                                        3u));
                        const uint32_t cidx = cbase + k;
                        acc[r] = fmaf(fmaf(dl, qv, -ml),
                                      (cidx < cols) ? x[cidx] : 0.0f,
                                      acc[r]);
                    }
                }
            }
            rp[r] += kQ2K_BYTES;
        }
    }
    // 8-lane shuffle tree per row (xor 4,1,2 stays inside the row's
    // 8-lane group -- same tree as kQGemvQ2KxW4).
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        const float p = acc[r] + __shfl_xor_sync(0xffffffffu, acc[r], 4);
        const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
        const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
        if (I == 0) out[row0 + r] = rr;
    }
}

// ---- Q2_K 4-WAY K-SPLIT decode GEMV (2026-09-30, A/B gate
// TINYCODER_Q2K_SPLIT4=1, DEFAULT OFF). (round-robin) so the per-lane
// SERIAL block chain is ~4x SHORTER than kQGemvQ2KxW4 (which walks all
// blocksPerRow blocks serially). This is the latency-hiding lever the
// multi-row kQGemvQ2KxW4_Nm attempt did NOT provide (it kept the full
// serial chain per lane). One row per warp (8 rows/block), no smem.
//
// K-splitting the warp cuts the FFN's serial chain 4x.
//
// Per-element float math is IDENTICAL to kQGemvQ2KxW4.  The
// cross-group join reorders the accumulation (~1e-7 float tolerance,
// same class as the other Q2_K float paths).
__global__ void kQGemvQ2KxSplit4(const uint8_t *__restrict__ w,
                                 const float *__restrict__ x,
                                 float *__restrict__ out, uint32_t rows,
                                 uint32_t cols, uint32_t rowBytes,
                                 uint32_t blocksPerRow) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t g = lane >> 3u;// 0..3 lane group
    const uint32_t I = lane & 7u; // sub-lane within the row
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;

    const uint32_t qOfs = 4u * I;
    const uint32_t qb = qOfs;
    const uint32_t isub = (qOfs < 16u) ? 0u : 1u;

    float acc = 0.0f;
    // Round-robin block ownership: group g owns b = g, g+4, g+8...
    // 4-way batching over the group's own blocks (block stride 4).
    uint32_t b = g;
    for (; b + 12u <= blocksPerRow; b += 16u) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
        for (uint32_t bb = 0; bb < 4; ++bb) {
            const uint8_t *crp =
                    rp + static_cast<uint64_t>(b + 4u * bb) * kQ2K_BYTES;
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
                const uint32_t ebase = (b + 4u * bb) * 256u + n * 128u;
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
                        slot = fmaf(fmaf(dl, qv, -ml),
                                    (cidx < cols) ? x[cidx] : 0.0f, slot);
                    }
                }
            }
        }
        acc += a0;
        acc += a1;
        acc += a2;
        acc += a3;
    }
    for (; b < blocksPerRow; b += 4u) {
        const uint8_t *crp = rp + static_cast<uint64_t>(b) * kQ2K_BYTES;
        const float df = __half2float(
                *reinterpret_cast<const __half *>(crp + 80));
        const float dmf = __half2float(
                *reinterpret_cast<const __half *>(crp + 82));
        const uint8_t *sc = crp;
        const uint8_t *q = crp + 16;
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
    }
    // Join the 4 lane-groups: xor 8 then 16 sums corresponding lanes
    // across the groups, so each lane holds sum over groups of its
    // group-lane value.  Lanes 0..7 then run the 8-lane tree.
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 16);
    const float p = acc + __shfl_xor_sync(0xffffffffu, acc, 4);
    const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
    const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
    if (I == 0) out[row] = rr;
}
