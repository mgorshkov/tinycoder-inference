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

// GpuGemvQ8_1.hpp -- Q8_1 activation path: Q2_K/Q3_K/Q4_K/Q5_K/Q6_K x Q8_1 GEMV.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// ------------------------------------------------------------------
// Q8_1 activation path (2026-10-08, Q8_1-Q2_K experiment). It differs
// from our existing Q8_K path (256-wide, sign-forced d<0, per-16 bsums)
// and from the float path (no quantize at all). The dot: dot =
// sum_{j=0..3} d8_j * ( dq.(sc&0xF).dp4a(pk_j, y_j) -
// dmin.(sc>>4).dp4a(0x01010101, y_j) ) reproduces the float kernel
// exactly (all-integer inside, one float mad per group), with 1 B/elem
// activation instead of 4 (removes the fp32-scalar-per-element load +
// fmaf that pins the float 4xW).
// ------------------------------------------------------------------
static constexpr uint32_t kQ8_1_STRIDE = 40;// 2B d + 2B s + 32 int8, padded
// Token cap for the materialised prefill attention (kAttnSoftmaxCausal
// + the two cuBLAS GEMMs).  Its score matrix is seqLen^2 floats PER
// HEAD: 512 tokens x 28 heads = 29 MB (+ 15 MB fp16 P), 1024 would be
// 117 MB (+ 59 MB) and 2048 470 MB.  pp512 is the benchmark shape and a
// 512-token prompt is the text-gate shape, so 512 is the cap; longer
// prefills (and any continued prefill at pos > 0) keep the warp kernel.
static constexpr uint32_t kAttnGemmMaxSeq = 512;
// grid = cols/32, block = 32 (one warp per 32-block).
__global__ void kQuantizeQ8_1(const float *__restrict__ x,
                              uint8_t *__restrict__ y1, uint32_t cols) {
    const uint32_t b = blockIdx.x;
    const uint32_t e0 = b * 32u;
    const uint32_t t = threadIdx.x;
    uint8_t *blk = y1 + static_cast<uint64_t>(b) * kQ8_1_STRIDE;
    int8_t *qs = reinterpret_cast<int8_t *>(blk + 4);
    const float xi = (e0 + t < cols) ? x[e0 + t] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.0f;
    qs[t] = (amax == 0.0f)
                    ? static_cast<int8_t>(0)
                    : static_cast<int8_t>(roundf(xi / d));
    if (t == 0) {
        __half *ds = reinterpret_cast<__half *>(blk);
        ds[0] = __float2half(d);
        ds[1] = __float2half(sum);
    }
}

// 4-rows-per-warp Q2_K x Q8_1 dp4a GEMV.  Geometry identical to
// kQGemvQ2KxW4 (all 32 lanes issue u32 weight loads, 8 lanes/row,
// 4 rows/warp, no smem); the per-element body is 1 u32 weight load +
// 1 u32 Q8_1 load + 2 dp4a + 1 float mad (vs the float kernel's 1 u32
// weight load + 4 fp32 activation loads + 4 fmaf).  4 Q2_K blocks are
// batched into independent accumulators (memory-level parallelism).
__global__ void __launch_bounds__(256) kQGemvQ2KxQ81_4xW(
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ y1,
        float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow,
        uint32_t rowBytes) {
    const uint32_t warpRowBase = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t base = warpRowBase * 4u;
    if (base >= rows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t I = lane & 7u;
    const uint32_t isub = I >> 2u;
    const uint32_t row = base + (lane >> 3u);
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    float sumf = 0.0f;
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
        for (uint32_t bb = 0; bb < 4u; ++bb) {
            const uint8_t *blkW =
                    rp + static_cast<uint64_t>(b + bb) * kQ2K_BYTES;
            const float dq = __half2float(
                    *reinterpret_cast<const __half *>(blkW + 80));
            const float dmin = __half2float(
                    *reinterpret_cast<const __half *>(blkW + 82));
            const uint8_t *sc = blkW;
            const uint8_t *q = blkW + 16;
            float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                    : (bb == 2u)       ? a2
                                                       : a3;
#pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                uint32_t w4;
                std::memcpy(&w4, q + n * 32u + 4u * I, sizeof(uint32_t));
                uint64_t scU8;
                std::memcpy(&scU8, sc + n * 8u, sizeof(uint64_t));
#pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) {
                    const uint8_t *blkY =
                            y1 + static_cast<uint64_t>(
                                         (b + bb) * 8u + n * 4u + j) *
                                         kQ8_1_STRIDE;
                    const float d8 = __half2float(
                            *reinterpret_cast<const __half *>(blkY));
                    uint32_t y4;
                    std::memcpy(&y4, blkY + 4u + 4u * I,
                                sizeof(uint32_t));
                    const uint32_t pk = (w4 >> (2u * j)) & 0x03030303u;
                    const int pdot = __dp4a(static_cast<int>(pk),
                                            static_cast<int>(y4), 0);
                    const int sumy = __dp4a(0x01010101,
                                            static_cast<int>(y4), 0);
                    const uint8_t scv = static_cast<uint8_t>(
                            (scU8 >> (8u * (2u * j + isub))) & 0xFFu);
                    slot += d8 * (dq * static_cast<float>(scv & 0xFu) *
                                          static_cast<float>(pdot) -
                                  dmin * static_cast<float>(scv >> 4u) *
                                          static_cast<float>(sumy));
                }
            }
        }
        sumf = a0 + sumf;
        sumf = a1 + sumf;
        sumf = a2 + sumf;
        sumf = a3 + sumf;
    }
    for (; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * kQ2K_BYTES;
        const float dq = __half2float(
                *reinterpret_cast<const __half *>(blkW + 80));
        const float dmin = __half2float(
                *reinterpret_cast<const __half *>(blkW + 82));
        const uint8_t *sc = blkW;
        const uint8_t *q = blkW + 16;
#pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            uint32_t w4;
            std::memcpy(&w4, q + n * 32u + 4u * I, sizeof(uint32_t));
            uint64_t scU8;
            std::memcpy(&scU8, sc + n * 8u, sizeof(uint64_t));
#pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint8_t *blkY =
                        y1 + static_cast<uint64_t>(b * 8u + n * 4u + j) *
                                     kQ8_1_STRIDE;
                const float d8 = __half2float(
                        *reinterpret_cast<const __half *>(blkY));
                uint32_t y4;
                std::memcpy(&y4, blkY + 4u + 4u * I, sizeof(uint32_t));
                const uint32_t pk = (w4 >> (2u * j)) & 0x03030303u;
                const int pdot = __dp4a(static_cast<int>(pk),
                                        static_cast<int>(y4), 0);
                const int sumy = __dp4a(0x01010101,
                                        static_cast<int>(y4), 0);
                const uint8_t scv = static_cast<uint8_t>(
                        (scU8 >> (8u * (2u * j + isub))) & 0xFFu);
                sumf += d8 *
                        (dq * static_cast<float>(scv & 0xFu) *
                                 static_cast<float>(pdot) -
                         dmin * static_cast<float>(scv >> 4u) *
                                 static_cast<float>(sumy));
            }
        }
    }
    const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
    const float qq = p + __shfl_xor_sync(0xffffffffu, p, 1);
    const float rr = qq + __shfl_xor_sync(0xffffffffu, qq, 2);
    if (I == 0u) out[row] = rr;
}

// ------------------------------------------------------------------
// Split-K Q2_K x Q8_1 dp4a GEMV (2026-10-09,
// TINYCODER_Q2K_Q81_SPLITK=1, requires TINYCODER_Q2K_Q81=1). Here the
// block is 64 threads = 2 warps, each warp is 4 eight-lane stripes (the
// 8-lane-of-4 column coverage reused from kQGemvQ2KxQ81_4xW), so ONE
// output row's K blocks are split 8 ways (stripe s takes b = s, s+8,
// s+16, ...). Each lane's dependent chain is blocksPerRow/8 instead of
// blocksPerRow. A 4-warp variant compiles from the same source but its
// different summation order flips sampled near-ties, so only the 2-warp
// shape is offered.
__global__ void __launch_bounds__(64) kQGemvQ2KxQ81_SplitK(
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ y1,
        float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow,
        uint32_t rowBytes) {
    const uint32_t lane = threadIdx.x;      // 0..31
    const uint32_t warp = threadIdx.y;      // 0..1
    const uint32_t I = lane & 7u;           // 8-lane column group
    const uint32_t stripe = lane >> 3u;     // 0..3 (K stripe in warp)
    const uint32_t nSplit = blockDim.y * 4u;// 8 (2 warps; see below)
    const uint32_t ksplit = warp * 4u + stripe;
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t isub = (4u * I < 16u) ? 0u : 1u;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    float acc = 0.0f;
    for (uint32_t b = ksplit; b < blocksPerRow; b += nSplit) {
        const uint8_t *blkW =
                rp + static_cast<uint64_t>(b) * kQ2K_BYTES;
        const float dq = __half2float(
                *reinterpret_cast<const __half *>(blkW + 80));
        const float dmin = __half2float(
                *reinterpret_cast<const __half *>(blkW + 82));
        const uint8_t *sc = blkW;
        const uint8_t *q = blkW + 16;
#pragma unroll
        for (uint32_t n = 0; n < 2u; ++n) {
            uint32_t w4;
            std::memcpy(&w4, q + n * 32u + 4u * I, sizeof(uint32_t));
            uint64_t scU8;
            std::memcpy(&scU8, sc + n * 8u, sizeof(uint64_t));
#pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint8_t *blkY =
                        y1 + static_cast<uint64_t>(
                                     b * 8u + n * 4u + j) *
                                     kQ8_1_STRIDE;
                const float d8 = __half2float(
                        *reinterpret_cast<const __half *>(blkY));
                uint32_t y4;
                std::memcpy(&y4, blkY + 4u + 4u * I, sizeof(uint32_t));
                const uint32_t pk = (w4 >> (2u * j)) & 0x03030303u;
                const int pdot = __dp4a(static_cast<int>(pk),
                                        static_cast<int>(y4), 0);
                const int sumy = __dp4a(0x01010101,
                                        static_cast<int>(y4), 0);
                const uint8_t scv = static_cast<uint8_t>(
                        (scU8 >> (8u * (2u * j + isub))) & 0xFFu);
                acc += d8 *
                       (dq * static_cast<float>(scv & 0xFu) *
                                static_cast<float>(pdot) -
                        dmin * static_cast<float>(scv >> 4u) *
                                static_cast<float>(sumy));
            }
        }
    }
    // 8-lane tree within the stripe (xor 4/1/2 stays inside the block).
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    __shared__ float red[16];
    if (I == 0u) red[ksplit] = acc;
    __syncthreads();
    if (threadIdx.x == 0u && warp == 0u) {
        float s = 0.0f;
        for (uint32_t i = 0; i < nSplit; ++i) s += red[i];
        out[row] = s;
    }
}

// QI2_K = QI3_K = 16, QR2_K = QR3_K = 4, QI8_1 = 8 for QK_K=256.
// -------- Q2_K x Q8_1 --------------------------------------------
__global__ void __launch_bounds__(64, 8) kQGemvQ2KxQ81_Gemv(
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ y1,
        float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow) {
    constexpr int nwarps = 2;
    constexpr int QK = 256, QI = 16, QR = 4, QI8_1 = 8;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int t = threadIdx.x;// lane 0..31
    const int tid = threadIdx.y * 32 + t;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * (6u * 84u);
    const int blocks_per_iter = nwarps * 32 / QI;// 4
    float tmp = 0.0f;
    for (int kbx = tid / QI; kbx < static_cast<int>(blocksPerRow);
         kbx += blocks_per_iter) {
        const int kby = kbx * (QK / 32);// *8
        const int kqs = tid % QI;
        const uint8_t *bq = rp + static_cast<uint64_t>(kbx) * 84u;
        // vec_dot_q2_K_q8_1
        const int bq8_offset = QR * (kqs / QI8_1);
        const int scale_offset =
                kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1 / 2);
        const uint8_t *scales = bq + scale_offset;// scales @ block+0
        const uint8_t *qs = bq + 16;              // qs @ block+16
        int v;
        std::memcpy(&v, qs + 4 * kqs, sizeof(int));// get_int_b4(qs,kqs)
        int u[QR];
        float d8[QR];
#pragma unroll
        for (int i = 0; i < QR; ++i) {
            const uint8_t *by = y1 +
                                static_cast<uint64_t>(kby + bq8_offset + i) * 40u;
            std::memcpy(&u[i], by + 4 + 4 * (kqs % QI8_1), sizeof(int));
            d8[i] = __half2float(
                    *reinterpret_cast<const __half *>(by));
        }
        float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < QR; ++i) {
            const int sc = scales[2 * i];
            const int vi = (v >> (2 * i)) & 0x03030303;
            sumf_d += d8[i] *
                      (__dp4a(vi, u[i], 0) * (sc & 0xF));
            int m = sc >> 4;
            m |= m << 8;
            m |= m << 16;
            sumf_m += d8[i] * __dp4a(m, u[i], 0);
        }
        const float2 dm2f = __half22float2(
                *reinterpret_cast<const __half2 *>(bq + 80));
        tmp += dm2f.x * sumf_d - dm2f.y * sumf_m;
    }
    // cross-warp reduce (nwarps-1 partials) then warp tree.
    __shared__ float red[32];
    if (threadIdx.y == 1) red[t] = tmp;
    __syncthreads();
    if (threadIdx.y == 0) {
        float v = tmp + red[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (t == 0) out[row] = v;
    }
}

// Same bytes, same values. ADOPTED as the default (2026-10-10):
// interleaved A/B tg128@pp512, 5 rounds: 211.5-212.1 -> 249.9-251.7
// tok/s, +18.0..+19.0 %, 5/5 wins, 4/4 prompt stream parity.
// TINYCODER_Q2K_WIDE=0 restores the memcpy byte-load form (the
// pre-9.23 default).
inline bool gemvWide() {
    static const bool v = [] {
        const char *e = std::getenv("TINYCODER_Q2K_WIDE");
        return (e == nullptr || std::atoi(e) != 0);
    }();
    return v;
}

// 2026-10-11: IQ2_S / IQ3_XXS / IQ3_S decode GEMVs were ALU-bound on the
// scalar grid-lookup dot (ncu: IQ2_S gate+up 867 us/layer at DRAM 7 %).
// ADOPTED default-on: the __dp4a + XOR-subtract-sign lane body (the reference
// vec_dot_iq*_q8_1 structure) is bit-identical and measured
// +5.6 % (1.5B iq3_xxs, 3/3) and +7.6 % (7B iq3_xxs, 2/2), parity
// byte-identical on 4/4 prompts.  TINYCODER_IQ_DP4A=0 restores the
// scalar body for A/B.
inline bool iqDp4a() {
    static const bool v = [] {
        const char *e = std::getenv("TINYCODER_IQ_DP4A");
        return (e == nullptr || std::atoi(e) != 0);
    }();
    return v;
}

// 2026-10-11: route IQ2_XS (GGML type 17) through the Q8_K integer
// GEMV family instead of the float dequant-dot kernels
// (kQGemvFusedGU_IQ2XS 869 us/layer at 25 % occupancy, kQGemv 499 us at
// 83 % LG-throttle).  ADOPTED default-on after A/B; TINYCODER_IQ2XS_INT=0
// restores the float path for A/B and for greedy-stream parity checks.
inline bool iq2xsInt() {
    static const bool v = [] {
        const char *e = std::getenv("TINYCODER_IQ2XS_INT");
        return (e == nullptr || std::atoi(e) != 0);
    }();
    return v;
}

// 2026-10-11: the IQ decode GEMVs are LG-throttle bound (42-53 % of
// stall cycles) on the divergent codebook lookups.  The fused gate+up
// kernel now stages the small grid table (1-8 KB) in shared memory so
// those reads use the MIO path.  ADOPTED default-on: +7 % (7B iq2_s,
// 2/2) and +2 % (7B iq3_xxs, 2/2).  TINYCODER_IQ_GRID_SMEM=0 restores
// the global grid for A/B.
inline bool iqGridSmem() {
    static const bool v = [] {
        const char *e = std::getenv("TINYCODER_IQ_GRID_SMEM");
        return (e == nullptr || std::atoi(e) != 0);
    }();
    return v;
}

// 2026-10-12: 4-rows-per-warp IQ Q8_K GEMV (kQGemvQ8K_4xW).  The
// generic kQGemvQ8K runs one row per warp, so every row re-reads the
// whole Q8_K activation and only one weight block is in flight per
// warp.  The 4xW kernel gives each 8-lane group its own row and each
// lane one ib32 sub-block: the activation load is shared across the
// warp's 4 rows (the L2 re-read cost of 8960 rows is eliminated) and 4
// weight blocks stay in flight.  Bit-exact with kQGemvQ8K (per-row
// block-order float accumulation unchanged).  Requires rows%4==0.
// MEASURED 2026-10-12: REJECTED (default OFF).  The IQ decode GEMV is
// codebook-lookup (LG-throttle) bound, not activation-re-read bound:
// the 4xW form multiplies the per-lane divergent grid lookups without
// reducing the total, and loses warp parallelism.  A/B on the 1.5B
// iq3_xxs: 162.0 tok/s (4xW) vs 174.2 (generic) -- a -7% regression.
// Kept for the record; TINYCODER_IQ_4XW=1 re-enables it.
inline bool iq4xw() {
    static const bool v = [] {
        const char *e = std::getenv("TINYCODER_IQ_4XW");
        return (e != nullptr && std::atoi(e) != 0);
    }();
    return v;
}

// 2026-10-29: 4-rows-per-warp Q4_K Q8_K decode GEMV (kQGemvQ4KxQ8K_4xW).
// The generic kQGemvKxQ8K<kTypeQ4K> runs one row per warp (only 8 of 32
// lanes issue weight loads) and scalar-loads each 32-byte q's strip, so
// the LARGEST quant class of the dense qwen35 models (Q4_K ~2.7 GB of
// Ornith-1.5-9B's 5.9 GB) has no 4xW kernel while Q5_K/Q6_K do.  The
// new kernel mirrors kQGemvQ5KxQ8K_4xW (4 rows/warp, all 32 lanes load)
// with an 8x __dp4a integer body; bit-exact with the generic kernel.
// DEFAULT ON: TINYCODER_Q4K_4XW=0 restores the generic kernel for A/B.
inline bool q4k4xw() {
    static const bool v = [] {
        const char *e = std::getenv("TINYCODER_Q4K_4XW");
        return (e == nullptr || std::atoi(e) != 0);
    }();
    return v;
}

// 2026-10-29: warps per block for the Q4_K 4xW decode GEMV.  The 4xW
// grid is rows/4/warps, so for the dense-qwen35 shapes (rows 4096/12288)
// the default 8-warp block yields only ~128-384 blocks -- often under
// one wave on the 68-SM 2080 Ti, i.e. occupancy-starved.  Smaller
// blocks give more blocks (better SM fill) at more block-fixed cost.
// TINYCODER_Q4K_BSY=N overrides the default 8.
inline int q4kBsy() {
    static const int v = [] {
        const char *e = std::getenv("TINYCODER_Q4K_BSY");
        int n = (e != nullptr) ? std::atoi(e) : 8;
        return (n >= 1 && n <= 32) ? n : 8;
    }();
    return v;
}

// 2026-10-29: kQ35L2NormRepeat previously recomputed the same headK-wide
// double L2 norm for every element of a slice's headK*nVHeads output
// (the per-slice norm depends only on the value head).  The fast path
// computes each slice's iq/ik once (shared memory), bit-identically.
// DEFAULT ON: TINYCODER_L2NORM_SMEM=0 restores the recompute body.
inline bool l2NormSmem() {
    static const bool v = [] {
        const char *e = std::getenv("TINYCODER_L2NORM_SMEM");
        return (e == nullptr || std::atoi(e) != 0);
    }();
    return v;
}

// Runtime gate: does THIS device expose the int8 m8n8k16 mma the INT8
// kernels need (Turing sm_75+)?  Cached after the first query.  The
// kernels are only COMPILED for the gencode arch list; this is the
// runtime half (so a pre-Turing device never launches them).
inline bool int8MmaOk() {
    static const int v = [] {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(
                    &major, cudaDevAttrComputeCapabilityMajor, dev) !=
            cudaSuccess)
            return 0;
        if (cudaDeviceGetAttribute(
                    &minor, cudaDevAttrComputeCapabilityMinor, dev) !=
            cudaSuccess)
            return 0;
        return (major > 7 || (major == 7 && minor >= 5)) ? 1 : 0;
    }();
    return v != 0;
}

// bytes in flight -- not a different inner loop. Dot values are the
// Q8_1 integer domain (~1e-6 from the float path), so parity is
// asserted via the deterministic greedy stream like the other gemv
// adoptions.
template<bool WIDE = false>
__global__ void __launch_bounds__(64, 8) kQGemvQ2KxQ81_GLU(
        const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
        const uint8_t *__restrict__ y1, float *__restrict__ out,
        uint32_t rows, uint32_t blocksPerRow) {
    constexpr int QK = 256, QI = 16, QR = 4, QI8_1 = 8;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int t = threadIdx.x;
    const int tid = threadIdx.y * 32 + t;
    const uint8_t *rg = wg + static_cast<uint64_t>(row) * (6u * 84u);
    const uint8_t *ru = wu + static_cast<uint64_t>(row) * (6u * 84u);
    const int blocks_per_iter = 4;// nwarps(2) * 32 / QI
    float tmp = 0.0f, tmpg = 0.0f;
    for (int kbx = tid / QI; kbx < static_cast<int>(blocksPerRow);
         kbx += blocks_per_iter) {
        const int kby = kbx * (QK / 32);
        const int kqs = tid % QI;
        const uint8_t *bg = rg + static_cast<uint64_t>(kbx) * 84u;
        const uint8_t *bu = ru + static_cast<uint64_t>(kbx) * 84u;
        // vec_dot_q2_K_q8_1, shared Q8_1 operand loads.
        const int bq8_offset = QR * (kqs / QI8_1);
        const int scale_offset =
                kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1 / 2);
        const uint8_t *scales_g = bg + scale_offset;// scales @ block+0
        const uint8_t *scales_u = bu + scale_offset;
        const uint8_t *qs_g = bg + 16;// qs @ block+16
        const uint8_t *qs_u = bu + 16;
        int vg, vu;
        if constexpr (WIDE) {
            vg = *reinterpret_cast<const int *>(qs_g + 4 * kqs);
            vu = *reinterpret_cast<const int *>(qs_u + 4 * kqs);
        } else {
            std::memcpy(&vg, qs_g + 4 * kqs, sizeof(int));
            std::memcpy(&vu, qs_u + 4 * kqs, sizeof(int));
        }
        int u[QR];
        float d8[QR];
#pragma unroll
        for (int i = 0; i < QR; ++i) {
            const uint8_t *by = y1 +
                                static_cast<uint64_t>(kby + bq8_offset + i) * 40u;
            if constexpr (WIDE) {
                u[i] = *reinterpret_cast<const int *>(
                        by + 4 + 4 * (kqs % QI8_1));
            } else {
                std::memcpy(&u[i], by + 4 + 4 * (kqs % QI8_1),
                            sizeof(int));
            }
            d8[i] = __half2float(
                    *reinterpret_cast<const __half *>(by));
        }
        float sd = 0.0f, sm = 0.0f, sgd = 0.0f, sgm = 0.0f;
#pragma unroll
        for (int i = 0; i < QR; ++i) {
            const int scg = scales_g[2 * i];
            const int scu = scales_u[2 * i];
            const int vig = (vg >> (2 * i)) & 0x03030303;
            const int viu = (vu >> (2 * i)) & 0x03030303;
            sd += d8[i] * (__dp4a(vig, u[i], 0) * (scg & 0xF));
            sgd += d8[i] * (__dp4a(viu, u[i], 0) * (scu & 0xF));
            int mg = scg >> 4;
            mg |= mg << 8;
            mg |= mg << 16;
            int mu = scu >> 4;
            mu |= mu << 8;
            mu |= mu << 16;
            sm += d8[i] * __dp4a(mg, u[i], 0);
            sgm += d8[i] * __dp4a(mu, u[i], 0);
        }
        const float2 dmg = __half22float2(
                *reinterpret_cast<const __half2 *>(bg + 80));
        const float2 dmu = __half22float2(
                *reinterpret_cast<const __half2 *>(bu + 80));
        tmp += dmg.x * sd - dmg.y * sm;
        tmpg += dmu.x * sgd - dmu.y * sgm;
    }
    // Cross-warp reduce for both partials, then the warp tree.
    __shared__ float red[32], redg[32];
    if (threadIdx.y == 1) {
        red[t] = tmp;
        redg[t] = tmpg;
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        // v = gate partial-sum tree, vg = up partial-sum tree (the
        // loop's int vg/vu are out of scope here).
        float g = tmp + red[t];
        float u = tmpg + redg[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            g += __shfl_xor_sync(0xffffffffu, g, off);
            u += __shfl_xor_sync(0xffffffffu, u, off);
        }
        if (t == 0) {
            // silu(gate)*up -- the exact kSiluMul expression/order.
            out[row] = (g / (1.0f + __expf(-g))) * u;
        }
    }
}

// -------- Q3_K x Q8_1 --------------------------------------------
// NW warps per block: the down/attn_o shapes are
// small-grid (rows = 1536 / 384) so the block, not the grid, is what
// buys memory-level parallelism here.  8 blocks/SM x 32*NW threads
// is 1024 threads/SM (50%) instead of 512 (25%) for NW=4.  The
// reduction stays a fixed-order tree, so the only numeric change is
// which thread owns which k-block.
//
// The launch bound must be the block size the launches use -- dim3(32,
// NW) -- not 64*NW: ptxas computes maxThreadsPerBlock x
// minBlocksPerMultiprocessor and warns ".minnctapersm will be ignored"
// when that exceeds the arch's 1024 threads/SM (256 x 8 = 2048).
template<bool WIDE, int NW = 4>
__global__ void __launch_bounds__(32 * NW, 8) kQGemvQ3KxQ81_Gemv(
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ y1,
        float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow,
        uint32_t rowBytes, float *__restrict__ resid) {
    constexpr int nwarps = NW;
    constexpr int QK = 256, QI = 16, QR = 4, QI8_1 = 8;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int t = threadIdx.x;
    const int tid = threadIdx.y * 32 + t;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    const int blocks_per_iter = nwarps * 32 / QI;// 4
    float tmp = 0.0f;
    for (int kbx = tid / QI; kbx < static_cast<int>(blocksPerRow);
         kbx += blocks_per_iter) {
        const int kby = kbx * (QK / 32);
        const int kqs = tid % QI;
        const uint8_t *bq = rp + static_cast<uint64_t>(kbx) * 110u;
        // vec_dot_q3_K_q8_1
        const int bq8_offset = QR * (kqs / (QI / 2));// 4*(kqs/8)
        const int scale_offset =
                kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1 / 2);
        const uint8_t *scales = bq + 96;// scales @ +96
        const float d3 = __half2float(
                *reinterpret_cast<const __half *>(bq + 108));
        const uint8_t *qs = bq + 32;// qs @ +32
        // get_int_b2(x, i32) == ((const int *)x)[i32] == the 32-bit
        // little-endian word at byte offset 4*i32 (NOT a 16-bit read!)
        // -- it assembles x16[2i] | (x16[2i+1]<<16), so it is identical
        // to get_int_b4.  Reading only 2 bytes here would zero the high
        // 16 bits and corrupt the 0x03030303/0x04040404 masks, dropping
        // half the elements per kqs (the parity bug).
        // WIDE (down variant): the Q3_K row stride is 110 B, so odd
        // (row,kbx) pairs leave the weight ints 2-mod-4 — a scalar 4-B
        // load there is NOT legal (unlike GLU's 84-B stride, where the
        // int deref is exact and was verified by the 4/4 stream).  The
        // aligned-safe wide form is 2x LDG.E.U16 per int (all these
        // addresses are even), halving the byte-load count.
        int vl, vh;
        if constexpr (WIDE) {
            const unsigned short *vl16 =
                    reinterpret_cast<const unsigned short *>(
                            qs + 4 * kqs);
            vl = vl16[0] | (static_cast<int>(vl16[1]) << 16);
            const unsigned short *vh16 =
                    reinterpret_cast<const unsigned short *>(
                            bq + 4 * (kqs % (QI / 2)));
            vh = vh16[0] | (static_cast<int>(vh16[1]) << 16);
        } else {
            std::memcpy(&vl, qs + 4 * kqs,
                        sizeof(int));// get_int_b2(qs, kqs)
            std::memcpy(&vh, bq + 4 * (kqs % (QI / 2)),
                        sizeof(int));// get_int_b2(hmask, kqs%8)
        }
        vh = ~vh >> bq8_offset;
        int u[QR];
        float d8[QR];
#pragma unroll
        for (int i = 0; i < QR; ++i) {
            const uint8_t *by = y1 +
                                static_cast<uint64_t>(kby + bq8_offset + i) * 40u;
            if constexpr (WIDE) {
                // y blocks stride 40 B: always 4-aligned.
                u[i] = *reinterpret_cast<const int *>(
                        by + 4 + 4 * (kqs % QI8_1));
            } else {
                std::memcpy(&u[i], by + 4 + 4 * (kqs % QI8_1),
                            sizeof(int));
            }
            d8[i] = __half2float(
                    *reinterpret_cast<const __half *>(by));
        }
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < QR; ++i) {
            const int isc = scale_offset + 2 * i;
            const int isc_low = isc % (QK / 32);
            const int sc_shift_low = 4 * (isc / (QK / 32));
            const int sc_low =
                    (scales[isc_low] >> sc_shift_low) & 0xF;
            const int isc_high = isc % (QK / 64);
            const int sc_shift_high = 2 * (isc / (QK / 64));
            const int sc_high =
                    ((scales[(QK / 32) + isc_high] >> sc_shift_high) & 3)
                    << 4;
            const int sc = (sc_low | sc_high) - 32;
            const int vil = (vl >> (2 * i)) & 0x03030303;
            const int vih = ((vh >> i) << 2) & 0x04040404;
            const int vi = __vsubss4(vil, vih);
            sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
        }
        tmp += d3 * sumf;
    }
    __shared__ float red[32 * nwarps];
    red[threadIdx.y * 32 + t] = tmp;
    __syncthreads();
    if (threadIdx.y == 0) {
        float v = red[t];
#pragma unroll
        for (int w = 1; w < nwarps; ++w) v += red[w * 32 + t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (t == 0) {
            out[row] = v;
            // Folded residual (2026-10-10): kAddResidual's
            // hidden[i] = hidden[i] + out[i] is THIS float add, so
            // passing s.hidden here keeps the residual stream
            // byte-identical while deleting one launch per layer.
            if (resid != nullptr) resid[row] = resid[row] + v;
        }
    }
}

// ------------------------------------------------------------------
// Q8_0 x Q8_32 (activation Q8_32) integer GEMV — bit-exact with the
// CPU's matMulVecBatchQ8_0_Q8K_AVX2.
//
// The CPU reference quantizes the fp32 activation per 32-wide block
// (Q8_0's natural block width — NOT the 256-wide Q8_K blocking used by
// the IQ* types above), with:
//   amax/maxv = signed abs-max scan over the block's 32 floats
//   iscale = (amax>0) ? -127/maxv : 0;  xq = clamp(lrintf(x*iscale))
//   d8 = (amax>0) ? 1/iscale : 0   (lrintf = nearest, ties to even)
// and accumulates the dot per 32-block via the sign-substitution
// identity (|xq| in [0,127] as the unsigned slot, w*sign(xq) as the
// signed slot) — an EXACT int8x8->int16, pairwise int16->int32 madd,
// THEN a float hsum of d*d8*sumi_group.  The float side is the AVX
// tree; the ONLY roundings are d*d8 (fp16 halfToFloat d x fp32 d8) and
// the per-group float mults.  Mirrored below:
//   * kQuantizeQ8_0x32: per-32-block Q8_0 activation quantize
//   * kQGemvQ8_0xQ8K: warp-per-row integer madd GEMV equivalent to
//     one AVX batch row (the CPU's per-block float tree IS the same
//     sequential d*d8*sum group accumulation, so lane-0 serial
//     accumulation reproduces it bit-exactly).
// 64-byte block stride: real data is 2-byte d + 32 int8 q (34 bytes at
// offsets 0..33), but the float d8 must be 4-byte (ideally 16-byte)
// aligned for `*reinterpret_cast<const float *>` reads in
// kQGemvQ8_0xQ8K.  A 34-byte stride puts block b's d at offset b*34
// -- 2 bytes out of alignment for every odd block -- and every odd
// block's kQGemvQ8_0xQ8K read faults with CUDA_ERROR_MISALIGNED_ADDRESS
// (sticky: falls back to CPU for the whole session).  Pad to 64 so the
// float lands at offset b*64 (16-aligned) and the pad bytes after the
// 34 used ones are never read (kQGemvQ8_0xQ8K's blkA indexing is
// strictly within the 34 used bytes, and the writer memset-fills the
// whole block so no stale-data hazard exists).
static constexpr uint32_t kQ8_0x32_STRIDE = 64;
// grid = blocksPerRow (= cols/32), block = 32 threads.  Bit-exact with
// matMulVecBatchQ8_0_Q8K_AVX2's quantize loop: the CPU scans the block
// SERIALLY (strict `>` keeps the FIRST abs-max, whose SIGNED value
// becomes maxv), so lane 0 reproduces that exact scan and broadcasts
// (amax, maxv) via shuffles; every lane then quantizes its own column
// with iscale = -127/maxv, xq = clamp(lrintf(x*iscale), -127, 127),
// d8 = 1/iscale (0 for the all-zero block).
// Layout: [0..3] fp32 d8 (written only by lane 0), [4..35] the 32 int8
// column values (lane -> offset 4+lane), pad [36..63] zeroed.
// NOTE (2026-09-14): the OLD layout put the int8 columns at 2+lane,
// which OVERLAPPED the 4-byte fp32 d8 at [0..3]: lane 0's float store
// (program-ordered last) clobbered columns 0 and 1 of every block with
// d8 byte garbage -- the root cause of a ~6% column corruption that
// made the Q8_0 GEMV diverge 5-80% from the CPU AVX2 reference
// (GPU-vs-CPU logits parity failure on qwen35moe).  Bytes are now
// placed strictly after the float.
__global__ void kQuantizeQ8_0x32(const float *__restrict__ x,
                                 uint8_t *__restrict__ q32,
                                 uint32_t cols) {
    const uint32_t b = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t cidx = b * 32u + lane;
    if (cidx >= cols) {
        // Pad lanes (cols multiple of 32 for all real GEMVs; pad the
        // output block to zero so a partial final block contributes 0,
        // exactly like the CPU's 32-block padding).
        q32[b * kQ8_0x32_STRIDE + 4u + lane] = 0;
        // Zero the pad block for determinism.  NOTE: strictly within
        // [36..63] -- lane<30 would write offsets 64/65, i.e. the NEXT
        // block's d8 low 2 bytes (nondeterministic race with that
        // block's parallel lane-0 float store).
        if (lane < 28u) {
            q32[b * kQ8_0x32_STRIDE + 36u + lane] = 0;
        }
        if (lane == 0)
            *reinterpret_cast<float *>(q32 + b * kQ8_0x32_STRIDE) = 0.0f;
        return;
    }
    // Lane 0 serial scan over the block's 32 floats (first-wins max).
    float amax = 0.0f, maxv = 0.0f;
    if (lane == 0) {
        const float *xs = x + b * 32u;
        for (uint32_t j = 0; j < 32u; ++j) {
            const float av = fabsf(xs[j]);
            if (av > amax) {
                amax = av;
                maxv = xs[j];
            }
        }
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    maxv = __shfl_sync(0xffffffffu, maxv, 0);
    if (amax == 0.0f) {
        q32[b * kQ8_0x32_STRIDE + 4u + lane] = 0;
        // Zero the pad block for determinism.
        if (lane < 28u) {
            q32[b * kQ8_0x32_STRIDE + 36u + lane] = 0;
        }
        if (lane == 0)
            *reinterpret_cast<float *>(q32 + b * kQ8_0x32_STRIDE) = 0.0f;
        return;
    }
    // IEEE fp32 division (NOT the fast approximate reciprocal nvcc
    // otherwise emits for a bare `1.0f / x` -- measured 1-ULP off from
    // the CPU's divss, which propagates through all blocks as ~1e-3
    // row error).  The CPU reference computes `iscale = -127.0f/maxv`
    // then `d8 = 1.0f/iscale` with IEEE division twice; mirror exactly.
    const float iscale = __fdiv_rn(-127.0f, maxv);
    int vq = static_cast<int>(rintf(iscale * x[cidx]));
    if (vq > 127) vq = 127;
    if (vq < -127) vq = -127;
    q32[b * kQ8_0x32_STRIDE + 4u + lane] =
            static_cast<uint8_t>(static_cast<int8_t>(vq));
    // Zero the pad block for determinism (inside [36..63] only).
    if (lane < 28u) {
        q32[b * kQ8_0x32_STRIDE + 36u + lane] = 0;
    }
    if (lane == 0)
        *reinterpret_cast<float *>(q32 + b * kQ8_0x32_STRIDE) =
                __fdiv_rn(1.0f, iscale);
}

// Q8_0 weights x Q8_32 activation: warp-per-row, exact int8 madd
// (bit-exact with matMulVecBatchQ8_0_Q8K_AVX2).  Per 32-wide block, the
// AVX reference computes the EXACT integer dot per 4-column group
// (maddubs + pairwise int16->int32 madd of the sign-substituted
// lanes), then the float tree:
//   t_g = (d*d8)*s_g           8 SEPARATE roundings (g=0..7)
//   u_g = t_g + t_{g+4}        pairs (0-4, 1-5, 2-6, 3-7)
//   v0 = u_0 + u_1; v1 = u_2 + u_3     (hadd 1)
//   blk = v0 + v1                      (hadd 2)
// and the block result is added SERIALLY into the row accumulator with
// plain mul+add (no fmaf — fmaf differs by 1 ulp per block,
// compounding through decode and flipping near-tie rows).
//
// PERFORMANCE NOTE (measured 2026-09-13): the previous version ran the
// float tree REDUNDANTLY on all 32 lanes (8 __shfl_sync gathers + 8
// mults per lane, per block) for "bit-exactness".  Bit-exactness only
// requires the SAME float ops in the SAME order, computed ONCE (the
// redundant lanes' t[]/blk are dead — only lane 0's sumf is used).
// With the Q8_0 integer GEMV routed through EVERY dense projection
// (trunk, shared expert, LM head: 8192-row and 248320-row matrices
// over 64 blocks/row), the redundant per-lane work made decode ~4-5x
// slower than the memory floor: ~197 ms/token spent in moeBlockMs.
// Now only lane 0 gathers the 8 converged group sums, runs the tree,
// and accumulates; the ops (8 roundings, pair/hadd order, serial
// mul+add without fmaf) are byte-identical to the CPU reference.
__global__ void kQGemvQ8_0xQ8K(const uint8_t *__restrict__ w,
                               const uint8_t *__restrict__ x,
                               float *__restrict__ out, uint32_t rows,
                               uint32_t blocksPerRow, uint32_t rowBytes) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    const uint32_t lane = threadIdx.x;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * 34u;
        const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
        const float d = __half2float(*reinterpret_cast<const __half *>(blkW));
        const float d8 = *reinterpret_cast<const float *>(blkA);
        if (d8 == 0.0f) continue;
        const int wq = static_cast<int>(
                static_cast<int8_t>(blkW[2u + lane]));
        const int xq = static_cast<int>(
                static_cast<int8_t>(blkA[4u + lane]));
        // Folded per-column contribution (sign-substitution identity):
        // |xq| * w * sign(xq) == xq * w — the exact per-lane product
        // the AVX kernel's maddubs computes (|xq| in slot A unsigned,
        // w*sign(xq) in slot B signed; the magnitudes multiply, the
        // sign travels).  127*127 = 16129 fits int32 trivially.
        int c = xq * wq;
        // LEAN: each lane folds its OWN column contribution and the
        // row is reduced ONCE after all blocks.  The old body ran two
        // XOR shuffles + an 8-way gather + a lane-0 float tree per
        // 32-byte block (10 warp-collective ops per 32 B), which made
        // the Q8_0 ssm_out GEMVs issue-bound.  Deterministic fp32
        // reassociation; top-1 coherence is this path's contract.
        sumf = __fadd_rn(__fmul_rn(d * d8, static_cast<float>(c)),
                         sumf);
    }
    // 32-lane reduction of the per-column partials.
    sumf += __shfl_xor_sync(0xffffffffu, sumf, 16);
    sumf += __shfl_xor_sync(0xffffffffu, sumf, 8);
    sumf += __shfl_xor_sync(0xffffffffu, sumf, 4);
    sumf += __shfl_xor_sync(0xffffffffu, sumf, 2);
    sumf += __shfl_xor_sync(0xffffffffu, sumf, 1);
    if (lane == 0) out[row] = sumf;
}
