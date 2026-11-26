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

// GpuBatchGemmKernels.hpp -- batched Q8_0xQ8_K GEMM, fused silu(gate)*up, batched down.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// =====================================================================
// BATCHED decode fast path (2026-09-14): launch ALL expertUsed ranks in
// ONE kernel with grid.y = rank.  All ranks' GEMV blocks co-schedule on
// the SMs (real parallelism instead of the old 40-serialized-kernels
// chain), and rank r reads its OWN slap (per-rank out slices + the
// per-rank weight pointer table moeRankWq_, [3*expertUsed] void*: gate /
// up / down slab base per rank).  Each kernel is a bit-exact clone of the
// single-matrix math it replaces — identical integer group sums, float
// trees, op order, and no fmaf — so the batched path produces byte-equal
// outputs to the old per-rank launches.
// =====================================================================

// Batched gate+up GEMV against the pre-quantized norm (q8kN_), then a
// fused silu(gate)*up write into rank r's [expertFF] slab.  Per rank:
// gate rows [0, expertFF) and up rows [expertFF, 2*expertFF) of the
// cache slab (wq[r*2] = gate base, wq[r*2+1] = up base); grid.x covers
// the 1024 combined rows (block (32,8)), grid.y = rank.
__global__ void kQGemvQ8_0xQ8K_BatchGU(const uint8_t *const *__restrict__ wq,
                                       const uint8_t *__restrict__ x,
                                       float *__restrict__ gateOut,
                                       float *__restrict__ upOut,
                                       uint32_t expertFF,
                                       uint32_t blocksPerRow,
                                       uint32_t rowBytes,
                                       uint32_t combinedRows,
                                       const uint8_t *const *__restrict__ xTab) {
    const uint32_t r = blockIdx.y;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= combinedRows) return;
    // First half = gate GEMV, second half = up GEMV.
    const uint32_t isUp = (row >= expertFF) ? 1u : 0u;
    const uint32_t wrow = row - isUp * expertFF;
    // moeRankWq_ table: [r*3 + 0] = gate base, [r*3 + 1] = up base,
    // [r*3 + 2] = down base.
    const uint8_t *rp = wq[r * 3u + isUp] +
                        static_cast<uint64_t>(wrow) * rowBytes;
    const uint32_t lane = threadIdx.x;
    // Prefill: every job is a distinct (token, rank) pair and carries its own
    // quantised activation row; decode passes xTab == nullptr (one token).
    const uint8_t *xr = (xTab != nullptr) ? xTab[r] : x;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * 34u;
        const uint8_t *blkA = xr + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
        const float d = __half2float(*reinterpret_cast<const __half *>(blkW));
        const float d8 = *reinterpret_cast<const float *>(blkA);
        if (d8 == 0.0f) continue;
        const int wq_v = static_cast<int>(
                static_cast<int8_t>(blkW[2u + lane]));
        const int xq = static_cast<int>(
                static_cast<int8_t>(blkA[4u + lane]));
        int c = xq * wq_v;
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
        }
    }
    if (lane == 0) {
        float *dest = isUp ? upOut : gateOut;
        dest[r * expertFF + wrow] = sumf;
    }
}

// Batch silu(gate)*up: rank r's fused activation, written in-place over
// gateOut slab r.  (The up slab is consumed read-only.)  Plain expression
// identical to kSiluMul: x/(1+__expf(-x)) * b (bit-exact with the CPU's
// geluInPlace? NO: the GPU fast path uses silu — see kSiluMul).
__global__ void kSiluMulBatch(float *__restrict__ gateOut,
                              const float *__restrict__ upOut,
                              uint32_t expertFF, uint32_t ranks) {
    const uint32_t r = blockIdx.y;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= expertFF) return;
    float *a = gateOut + r * expertFF;
    float x = a[i];
    a[i] = (x / (1.0f + __expf(-x))) * upOut[r * expertFF + i];
}

// Batched per-rank quantize of the silu product (rank r reads gateOut
// slab r, writes moeQ8kDownP_ slab r) using the EXACT kQuantizeQ8_0x32
// math (__fdiv_rn IEEE divisions).  grid = (blocksPerRow, ranks).
__global__ void kQuantizeQ8_0x32_Batch(const float *__restrict__ x,
                                       uint8_t *__restrict__ q32,
                                       uint32_t cols, uint32_t ranks,
                                       uint32_t strideBytes) {
    const uint32_t r = blockIdx.y;
    const uint32_t b = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t cidx = b * 32u + lane;
    const float *xs = x + r * cols;
    uint8_t *q = q32 + (static_cast<uint64_t>(r) * strideBytes) +
                 static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
    if (cidx >= cols) {
        q[4u + lane] = 0;
        if (lane < 28u) q[36u + lane] = 0;
        if (lane == 0)
            *reinterpret_cast<float *>(q) = 0.0f;
        return;
    }
    float amax = 0.0f, maxv = 0.0f;
    if (lane == 0) {
        const float *xblk = xs + b * 32u;
        for (uint32_t j = 0; j < 32u; ++j) {
            const float av = fabsf(xblk[j]);
            if (av > amax) {
                amax = av;
                maxv = xblk[j];
            }
        }
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    maxv = __shfl_sync(0xffffffffu, maxv, 0);
    if (amax == 0.0f) {
        q[4u + lane] = 0;
        if (lane < 28u) q[36u + lane] = 0;
        if (lane == 0) *reinterpret_cast<float *>(q) = 0.0f;
        return;
    }
    const float iscale = __fdiv_rn(-127.0f, maxv);
    int vq = static_cast<int>(rintf(iscale * xs[cidx]));
    if (vq > 127) vq = 127;
    if (vq < -127) vq = -127;
    q[4u + lane] = static_cast<uint8_t>(static_cast<int8_t>(vq));
    if (lane < 28u) q[36u + lane] = 0;
    if (lane == 0)
        *reinterpret_cast<float *>(q) = __fdiv_rn(1.0f, iscale);
}

// Batched down GEMV: rank r's [hidden] output from the quantized silu
// product (moeQ8kDownP_ slab r).  The [H][expertFF] down matrix rows =
// stack of [expertFF]-wide blocks; grid.x rows (block (32,8)), grid.y
// rank; per-rank down base from wq[r*2+2] + r's explicit slab offset.
__global__ void kQGemvQ8_0xQ8K_BatchDown(const uint8_t *const *__restrict__ wq,
                                         const uint8_t *__restrict__ x,
                                         float *__restrict__ out,
                                         uint32_t hidden,
                                         uint32_t blocksPerRow,
                                         uint32_t rowBytes,
                                         uint32_t xStrideBytes) {
    const uint32_t r = blockIdx.y;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= hidden) return;
    // moeRankWq_ table: [r*3 + 2] = down base.
    const uint8_t *rp = wq[r * 3u + 2u] +
                        static_cast<uint64_t>(row) * rowBytes;
    const uint8_t *xr = x + (static_cast<uint64_t>(r) * xStrideBytes);
    const uint32_t lane = threadIdx.x;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * 34u;
        const uint8_t *blkA = xr + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
        const float d = __half2float(*reinterpret_cast<const __half *>(blkW));
        const float d8 = *reinterpret_cast<const float *>(blkA);
        if (d8 == 0.0f) continue;
        const int wq_v = static_cast<int>(
                static_cast<int8_t>(blkW[2u + lane]));
        const int xq = static_cast<int>(
                static_cast<int8_t>(blkA[4u + lane]));
        int c = xq * wq_v;
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
        }
    }
    if (lane == 0) out[r * hidden + row] = sumf;
}

// Batched down GEMV with rank REMAP (Option D resident split):
// identical math to kQGemvQ8_0xQ8K_BatchDown, but the OUTPUT rank
// index comes from the per-slot rankOf table — the resident ranks are
// COMPACTED into slots 0..residentCount-1 for the wq/silu/quantize
// pipeline, while the down rows must land in the FULL rank-major
// combined table at [rankOf[j]*outStride + row] so the rank-order
// accumulation (kQ35MoeAccumScaled(kQ35MoeAccumScaledBatch}) reads the
// CPU-filled non-resident slots and the GPU-filled resident slots in
// the exact reference order.  outStride == seqLen*H (per-rank slab).
__global__ void kQGemvQ8_0xQ8K_BatchDownR(const uint8_t *const *__restrict__ wq,
                                          const uint8_t *__restrict__ x,
                                          float *__restrict__ out,
                                          const int32_t *__restrict__ rankOf,
                                          uint32_t hidden,
                                          uint32_t blocksPerRow,
                                          uint32_t rowBytes,
                                          uint32_t xStrideBytes,
                                          uint32_t outStride,
                                          const uint32_t *__restrict__ tokOf) {
    const uint32_t j = blockIdx.y;// compact resident slot
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= hidden) return;
    const int32_t r0 = (rankOf != nullptr) ? rankOf[j] : static_cast<int32_t>(j);
    if (r0 < 0) return;
    // wq table: [j*3 + 2] = down base of the compact resident slot j.
    const uint8_t *rp = wq[j * 3u + 2u] +
                        static_cast<uint64_t>(row) * rowBytes;
    const uint8_t *xr = x + (static_cast<uint64_t>(j) * xStrideBytes);
    const uint32_t lane = threadIdx.x;
    float sumf = 0.0f;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * 34u;
        const uint8_t *blkA = xr + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
        const float d = __half2float(*reinterpret_cast<const __half *>(blkW));
        const float d8 = *reinterpret_cast<const float *>(blkA);
        if (d8 == 0.0f) continue;
        const int wq_v = static_cast<int>(
                static_cast<int8_t>(blkW[2u + lane]));
        const int xq = static_cast<int>(
                static_cast<int8_t>(blkA[4u + lane]));
        int c = xq * wq_v;
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
        }
    }
    if (lane == 0) {
        // Combined table is rank-major [eu][seqLen][hidden]; for prefill each
        // job carries the token whose row it writes (tokOf == nullptr for
        // decode, where the caller already offset the base by s*hidden).
        const uint64_t tokOff =
                (tokOf != nullptr) ? static_cast<uint64_t>(tokOf[j]) * hidden : 0;
        out[static_cast<uint64_t>(r0) * outStride + tokOff + row] = sumf;
    }
}

// =====================================================================
// Q4_K / Q6_K x Q8_K-activation integer GEMV (Option D resident fast
// path for Q4_K_M models, 2026-09-16).
//
// The Ornith-1.5-35B-Q4_K_M expert matrices are gate/up = Q4_K
// (type 12, 144 B/256-element block) + down = Q6_K (type 14, 210
// B/block).  The CPU reference (matMulVecBatchQ4K_Q8K_AVX2 /
// matMulVecBatchQ6K_Q8K_AVX2) quantizes the fp32 activation to Q8_K
// (256-wide blocks, kQ8K_STRIDE layout) and computes the dot with
// EXACT integer products plus specific fp32 trees.  These kernels
// reproduce BOTH the integer math and the float op order so the GPU
// resident path is bit-exact with the CPU grouped-batch path.
//
// DECODED COMPILER BEHAVIOR (verified by disassembly of
// SIMDMatMulVecAVX2.o under -O2 -mavx2 -mfma, the real flags):
//   Q4_K:  mainS = fma(d*sc_i, s32q_i, mainS)   (i = 0..7 serial)
//          minS  = fma(dmin*m_i, bsum_i, minS)  (i = 0..7 serial)
//          acc   = fma(d8, mainS - minS, acc)   (vsubss + vfmadd132ss)
//   Q6_K:  per-lane offset = (-32*d*d8) * bprod       (2 muls + 1 mul)
//          acc_j = fma(d*d8, sumi0_j, off_j)          (chunk 0)
//          acc_j = fma(d*d8, sumi1_j, acc_j)          (chunk 1)
//          running_j += acc_j (per block, serial 8-lane add)
//          final hsum tree: p=a0+a4..a3+a7 (add), q=p0+p1,p2+p3,
//          r=q0+q1  (the AVX hadd chain: 7 roundings, matching the
//          _mm_hadd_ps tree exactly).
// All int->float conversions are exact (values < 2^24), so only the
// float op ORDER matters.
// =====================================================================

// ---- Warp-per-row Q4_K / Q6_K x Q8_K GEMV: bit-exact with the CPU
// matMulVecBatchQ4K_Q8K_AVX2 / matMulVecBatchQ6K_Q8K_AVX2 kernels.
// blockDim = (32, 8): threadIdx.y = row, 32 lanes = sub-blocks /
// column lanes.  All per-lane integer products are EXACT (int32
// accumulators), so only the float op ORDER matters:
//
// Q4_K (NOT FMA-contracted: the SIMD TU is compiled with -mfma but
// WITHOUT -ffp-contract, so `mainS += ds*s32q` etc. emit separate
// vmulss+vaddss pairs, verified by objdump at the tile lambda and by
// the mode0 bit-exact match of the no-FMA host model; hence the
// volatile-intermediate trick below):
//   mainS = dsc_i*s32q_i + mainS    i = 0..7 serial (mul then add)
//   minS  = dmn_i*bsum_i + minS     i = 0..7 serial (mul then add)
//   sumf  = d8*(mainS - minS) + sumf  (vsubss then mul then add)
// Q6_K (explicit intrinsics, no contraction beyond the literal
// _mm256_fmadd_ps):
//   off_j = ((-32*d) * d8) * bprod_j        (3 muls)
//   acc_j = fma(d*d8, sumi0_j, off_j)       (chunk 0)
//   acc_j = fma(d*d8, sumi1_j, acc_j)       (chunk 1)
//   running_j += acc_j per block (vaddps)
//   final: p_j = a_j + a_{j+4}; q = p + shfl_xor(p, 1);
//          r = q + shfl_xor(q, 2)  (== the AVX hadd chain: 7 adds)
//
// Q6_K scale identity (from k_shuffle + cvtepi8_epi16 in the AVX2
// kernel): for global sub-block s' in [0,8), the madd lanes of that
// sub use scale sc[2s'] for its columns 0..15 and sc[2s'+1] for its
// columns 16..31 (sc = sign-extended int8[16] block scales at
// blk+192).  sumi lane J (J in 0..7) of chunk c accumulates sub
// s = 4c..4c+3, global columns s*32 + 4J .. s*32 + 4J+3, weighted by
// sc[2s*] (J<4) or sc[2s*+1] (J>=4).
// Shared per-block dot contribution for the Q4_K / Q6_K x Q8_K GEMV
// kernels (single-row + batched).  Warp-collective: ALL 32 lanes call
// it (the __shfl_sync gathers are reached unconditionally).  For Q4_K
// only lane 0 updates `running` (the per-block scalar addend, applied
// as fmaf(d8, ms - mns, running) — the exact vsubss+vfmadd132ss CPU
// order); for Q6_K lanes 0..7 each update their own lane-j running
// accumulator by `acc_j`.  NOTE: NO d8==0 skip — the CPU Q4K/Q6K
// kernels have no zero-guard, and skipping would change the sign of a
// -0 running accumulator (bit-exactness requires the +0 add).
// WIDE 4-byte load for 210-B-stride Q6_K weight ints (plan section 9.23,
// extended to the LM head 2026-10-11): the block base is always even
// (210 = 2*105) and every offset read here is a multiple of 2, so a pair
// of 16-bit loads is legal at EVERY address and assembles the same
// little-endian 32-bit word that std::memcpy from a uint8_t* otherwise
// splits into 4 byte LDGs (the LG-throttle disease section 9.23
// root-caused and fixed on the gemv family).  Two LDG.E.U16 instead of
// four LDG.E.U8 per weight int, bit-identical bytes.
__device__ __forceinline__ uint32_t kxWideU32(const uint8_t *p) {
    const unsigned short *p16 =
            reinterpret_cast<const unsigned short *>(p);
    return static_cast<uint32_t>(p16[0]) |
           (static_cast<uint32_t>(p16[1]) << 16);
}

template<int TYPE, bool WIDE = false, bool GROUPED = false>
__device__ __forceinline__ void kxQ8KBlockDot(float &running,
                                              const uint8_t *blkW,
                                              const uint8_t *blkA,
                                              uint32_t lane) {
    // GROUPED=false (legacy 1-row-per-warp call): only warp lanes 0-7
    // (the sub-block owners) work; the reduction gathers lanes 0-7
    // (shuffle width 32) and lane 0 folds into `running`.  EVERY
    // expression below is unchanged vs the pre-2026-10-12 kernel.
    // GROUPED=true (4-rows-per-warp call): every lane owns sub-block
    // Iid = lane&7 of ITS own 8-lane group's row, so the gather uses
    // shuffle width 8 (each group reduces its own row) and each group
    // leader (Iid==0) folds its row's value into its own `running`.
    // The per-(row, Iid) arithmetic is byte-identical, only the lane
    // that ends up holding a given row's scalar changes -- which the
    // caller's 4-row shuffle tree coalesces within each group.
    const uint32_t Iid = GROUPED ? (lane & 7u) : lane;
    const uint32_t shW = GROUPED ? 8u : 32u;
    const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA);
    const int16_t *bsums =
            reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
    const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
    if (TYPE == kTypeQ4K) {
        int s32q = 0;
        float dsv = 0.0f, mnv = 0.0f;
        if (Iid < 8) {
            const uint32_t i = Iid;
            const float d = __half2float(
                    *reinterpret_cast<const __half *>(blkW));
            const float dmin = __half2float(
                    *reinterpret_cast<const __half *>(blkW + 2));
            const uint8_t *scales = blkW + 4;
            const uint8_t *qsv = blkW + 16;
            uint8_t sc, m;
            if (i < 4) {
                sc = scales[i] & 63u;
                m = scales[i + 4] & 63u;
            } else {
                sc = static_cast<uint8_t>(
                        (scales[i + 4] & 0xFu) |
                        ((scales[i - 4] >> 6) << 4));
                m = static_cast<uint8_t>(
                        (scales[i + 4] >> 4) |
                        ((scales[i] >> 6) << 4));
            }
            dsv = d * static_cast<float>(sc);
            mnv = dmin * static_cast<float>(m);
            const uint32_t half = i & 1u;
            const uint8_t *qsb = qsv + 32u * (i >> 1u);
            const int8_t *yp = y8 + i * 32u;
            // VECTORIZED integer body (2026-10-29): 8x __dp4a replaces
            // the 32 scalar byte-loads + per-byte nibble select that
            // made this the scalar bottleneck of BOTH the generic
            // 1-row GEMV and the batched prefill GEMM
            // (kQGemmKxQ8K).  qsb is 4-aligned (rowBytes 144 % 4 == 0,
            // qsv=blkW+16, 32*(i>>1), 4-B steps), so the u32 loads are
            // aligned.  w32 holds the 4 packed 0..15 weights, yb4 the 4
            // signed int8 activations; the integer sum is exact and
            // order-free, so s32q is bit-identical to the scalar body
            // and the float fold below is unchanged.  This is the same
            // dp4a body as kxQ4KBlockDotVec (4-rows-per-warp kernel);
            // it lifts the prefill GEMM (the dense-qwen35 prefill cost)
            // without touching any output bit.
#pragma unroll
            for (uint32_t j = 0; j < 32u; j += 4u) {
                const uint32_t wb4 =
                        *reinterpret_cast<const uint32_t *>(qsb + j);
                const uint32_t yb4 =
                        *reinterpret_cast<const uint32_t *>(yp + j);
                const uint32_t w32 =
                        (half == 0u) ? (wb4 & 0x0F0F0F0Fu)
                                     : ((wb4 >> 4) & 0x0F0F0F0Fu);
                s32q = __dp4a(static_cast<int>(w32),
                              static_cast<int>(yb4), s32q);
            }
        }
        int ss8[8];
        float dss[8], mns8[8];
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            ss8[i] = __shfl_sync(0xffffffffu, s32q, i, shW);
            dss[i] = __shfl_sync(0xffffffffu, dsv, i, shW);
            mns8[i] = __shfl_sync(0xffffffffu, mnv, i, shW);
        }
        if (Iid == 0) {
            // The compiled CPU kernel (matMulVecBatchQ4K_Q8K_AVX2, built
            // with -mfma but WITHOUT -ffp-contract) executes
            // `mainS += ds*s32q` as SEPARATE vmulss+vaddss (NO FMA
            // contraction — verified by disassembly and the mode0
            // bit-exact match of the no-FMA host model).  nvcc WOULD
            // fuse a plain `a*b + c` into FFMA, so the multiply must be
            // forced through a volatile local (memory materialization
            // makes fusion impossible) to reproduce the CPU's two
            // independent roundings bit-exactly.
            float ms = 0.0f, mns = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                const float bi = static_cast<float>(
                        static_cast<int>(bsums[2 * i]) +
                        static_cast<int>(bsums[2 * i + 1]));
                ms = __fadd_rn(__fmul_rn(dss[i], static_cast<float>(ss8[i])), ms);
                mns = __fadd_rn(__fmul_rn(mns8[i], bi), mns);
            }
            running = __fadd_rn(__fmul_rn(d8, (ms - mns)), running);
        }
    } else if (TYPE == kTypeQ5K) {
        // Q5_K (176 B/block): d(2) + dmin(2) + scales[12] + qh[32] +
        // qs[128].  Mirror matMulVecBatchQ5K_Q8K_AVX2 EXACTLY:
        //   * per sub-block i (0..7): 5-bit weights from the LOW/HIGH
        //     nibbles of qs[32c] (c = i>>1) plus the qh bit
        //     1<<(2c+half) selecting the 16-bit value, integer-dotted
        //     with the Q8_K activation y8[i*32..i*32+31];
        //   * mainS/MS += ds[i]*s32q (mul-then-add, NO FMA — forced
        //     through volatile locals since nvcc would otherwise
        //     contract the multiply+add into a single FFMA);
        //   * minS/MNS += mn[i]*(bsums[2i]+bsums[2i+1]);
        //   * lane 0: running += d8*(mainS-minS) (vsubss then vmulss
        //     then vaddss — same 3-rounding chain as the AVX2 kernel).
        int s32q = 0;
        float dsv = 0.0f, mnv = 0.0f;
        if (Iid < 8) {
            const uint32_t i = Iid;
            const float d = __half2float(
                    *reinterpret_cast<const __half *>(blkW));
            const float dmin = __half2float(
                    *reinterpret_cast<const __half *>(blkW + 2));
            const uint8_t *scales = blkW + 4;
            const uint8_t *qh = blkW + 16;
            const uint8_t *qs = blkW + 48;
            uint8_t sc, m;
            if (i < 4) {
                sc = scales[i] & 63u;
                m = scales[i + 4] & 63u;
            } else {
                sc = static_cast<uint8_t>(
                        (scales[i + 4] & 0xFu) |
                        ((scales[i - 4] >> 6) << 4));
                m = static_cast<uint8_t>(
                        (scales[i + 4] >> 4) |
                        ((scales[i] >> 6) << 4));
            }
            dsv = d * static_cast<float>(sc);
            mnv = dmin * static_cast<float>(m);
            const uint32_t c = i >> 1u;
            const uint32_t half = i & 1u;
            const uint32_t u = 1u << (2u * c + half);
            const uint8_t *qsp = qs + 32u * c;
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint8_t qb = qsp[j];
                const uint8_t low = (half == 0u) ? (qb & 0xFu)
                                                 : (qb >> 4);
                const uint8_t q5 = static_cast<uint8_t>(
                        low | ((qh[j] & u) != 0u ? 0x10u : 0u));
                s32q += static_cast<int>(q5) *
                        static_cast<int>(y8[i * 32u + j]);
            }
        }
        int ss8[8];
        float dss[8], mns8[8];
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            ss8[i] = __shfl_sync(0xffffffffu, s32q, i, shW);
            dss[i] = __shfl_sync(0xffffffffu, dsv, i, shW);
            mns8[i] = __shfl_sync(0xffffffffu, mnv, i, shW);
        }
        if (Iid == 0) {
            float ms = 0.0f, mns = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                const float bi = static_cast<float>(
                        static_cast<int>(bsums[2 * i]) +
                        static_cast<int>(bsums[2 * i + 1]));
                ms = __fadd_rn(__fmul_rn(dss[i], static_cast<float>(ss8[i])), ms);
                mns = __fadd_rn(__fmul_rn(mns8[i], bi), mns);
            }
            running = __fadd_rn(__fmul_rn(d8, (ms - mns)), running);
        }
    } else if (TYPE == kTypeIQ4XS) {
        // IQ4_XS (136 B/block): d(2) + scales_h(2) + scales_l(4) +
        // qs[128].  Mirror matMulVecBatchIQ4XS_Q8K_AVX2 EXACTLY:
        //   * per sub-block i (0..7): 6-bit scale ls (nibble of
        //     scales_l[i/2] + 2 bits of scales_h), dl = d*(ls-32);
        //   * the 32 weights are kvalues_iq4nl[nib] (SIGNED int8), so
        //     the AVX2 |w|*sign(w)*x integer dot reduces to a plain
        //     signed kvalues_iq4nl[nib]*y8[col] dot (exact int32);
        //   * mainS/MS += dl[i]*s32q (mul-then-add, NO FMA — volatile
        //     locals, same no-ffp-contract discipline as Q4_K);
        //   * NO min term (the (ls-32) centers the scale);
        //   * lane 0: running += d8*mainS (vmulss then vaddss).
        int s32q = 0;
        float dlsv = 0.0f;
        if (Iid < 8) {
            const uint32_t i = Iid;
            const float d = __half2float(
                    *reinterpret_cast<const __half *>(blkW));
            const uint16_t scales_h =
                    *reinterpret_cast<const uint16_t *>(blkW + 2);
            const uint8_t *scales_l = blkW + 4;
            const uint8_t *qsv = blkW + 8;
            const int ls = static_cast<int>(
                                   (scales_l[i >> 1u] >>
                                    (4u * (i & 1u))) &
                                   0xFu) |
                           static_cast<int>(
                                   ((scales_h >> (2u * i)) & 0x3u)
                                   << 4);
            dlsv = d * static_cast<float>(ls - 32);
            const uint8_t *qsp = qsv + 16u * i;
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint8_t nib = (j < 16u)
                                            ? static_cast<uint8_t>(
                                                      qsp[j] & 0xFu)
                                            : static_cast<uint8_t>(
                                                      qsp[j - 16u] >>
                                                      4);
                s32q += static_cast<int>(c_kvalues_iq4nl[nib]) *
                        static_cast<int>(y8[i * 32u + j]);
            }
        }
        int ss8[8];
        float dls8[8];
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            ss8[i] = __shfl_sync(0xffffffffu, s32q, i, shW);
            dls8[i] = __shfl_sync(0xffffffffu, dlsv, i, shW);
        }
        if (Iid == 0) {
            float ms = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                ms = __fadd_rn(__fmul_rn(dls8[i], static_cast<float>(ss8[i])), ms);
            }
            running = __fadd_rn(__fmul_rn(d8, ms), running);
        }
    } else {
        // Q6_K (210 B/block): ql[128] + qh[64] + sc[16] + d(2).
        // Bit-exact mirror of matMulVecBatchQ6K_Q8K_AVX2 blockDot +
        // 8-lane store: per active lane I (0..7):
        //   * bprod = sc[2I]*bsums[2I] + sc[2I+1]*bsums[2I+1] (the
        //     CPU's madd_epi16(s16, bsums) lane I);
        //   * acc  = kOff*bprod, kOff = (-32*d)*d8 (CPU: set1_ps
        //     mul_ps of the bprod lane product);
        //   * per chunk c (0/1), 4 subs s (idx = 4c+s): sumi over the
        //     4 columns col = idx*32 + 4*I + k, each wv = (ql nibble
        //     [low if (idx&2)==0 else high]) | (qh 2-bit field
        //     shift 2*(idx&3)) << 4, scaled by sc[2*idx] (I<4) or
        //     sc[2*idx+1] (I>=4) matching the CPU shuffle broadcast;
        //     then acc = fmaf(d*d8, sumi, acc) (CPU fmadd_ps).
        // 2026-09-28 VECTORIZED: lanes 0-7 (the only workers) used to
        // load every weight byte individually (64 ld.b8/block with 24
        // of 32 lanes idle) -- the Q6_K LM head (151936 rows x 6
        // blocks, 8960 rows x 35 blocks) ran 5x above the 448 GB/s
        // floor.  Now each lane pulls its 4 contiguous ql bytes
        // (W[a][c] = blkW + 32*a + 64*c + 4*I) and 4 qh bytes
        // (H[c] = blkW + 128 + 32*c + 4*I) as u32 registers and its
        // 4 activation bytes per sub-block (Y[idx] = blkA + 32*idx +
        // 4*I) as u32.  Same bytes, same products, same (c,s,k)
        // accumulation order -- bit-identical results.
        float acc = 0.0f;
        if (lane < 8) {
            const uint32_t I = lane;
            const float d = __half2float(
                    *reinterpret_cast<const __half *>(blkW + 208));
            const int8_t *sc =
                    reinterpret_cast<const int8_t *>(blkW + 192);
            const int s0 = static_cast<int>(sc[2 * lane]);
            const int s1 = static_cast<int>(sc[2 * lane + 1]);
            const int bprod =
                    s0 * static_cast<int>(bsums[2 * lane]) +
                    s1 * static_cast<int>(bsums[2 * lane + 1]);
            const float kOff = (-32.0f * d) * d8;
            acc = kOff * static_cast<float>(bprod);
            const float k2 = d * d8;
            // u32 register loads (unaligned-safe memcpy; fully unrolled
            // so the W/H/Y arrays stay in registers).
            uint32_t W[2][2], H[2], Y[8];
            // NOTE: 210-byte Q6_K blocks are NOT 4-byte aligned at row
            // granularity (210 = 2x105), so these MUST stay unaligned-
            // safe loads -- a u32 __ldcs here faults (tried 2026-10-10,
            // killed the kernel -> CPU fallback).  The WIDE form
            // (section 9.23) issues 2x 16-bit loads (always legal: the
            // block base and every weight offset are even) instead of
            // the 4 byte LDGs std::memcpy emits through uint8_t*; the
            // Y (activation) loads are 4-aligned (kQ8K_STRIDE 292,
            // 4-B k-cols) so they take a single int deref.
            if constexpr (WIDE) {
#pragma unroll
                for (uint32_t a = 0; a < 2; ++a) {
                    H[a] = kxWideU32(blkW + 128u + 32u * a + 4u * I);
#pragma unroll
                    for (uint32_t c = 0; c < 2; ++c) {
                        W[a][c] = kxWideU32(
                                blkW + 32u * a + 64u * c + 4u * I);
                    }
                }
#pragma unroll
                for (uint32_t q = 0; q < 8; ++q) {
                    Y[q] = *reinterpret_cast<const uint32_t *>(
                            blkA + 32u * q + 4u * I);
                }
            } else {
#pragma unroll
                for (uint32_t a = 0; a < 2; ++a) {
                    std::memcpy(&H[a], blkW + 128u + 32u * a + 4u * I,
                                sizeof(uint32_t));
#pragma unroll
                    for (uint32_t c = 0; c < 2; ++c) {
                        std::memcpy(&W[a][c],
                                    blkW + 32u * a + 64u * c + 4u * I,
                                    sizeof(uint32_t));
                    }
                }
#pragma unroll
                for (uint32_t q = 0; q < 8; ++q) {
                    std::memcpy(&Y[q], blkA + 32u * q + 4u * I,
                                sizeof(uint32_t));
                }
            }
#pragma unroll
            for (uint32_t c = 0; c < 2; ++c) {
                int sumi = 0;
                const uint32_t hw = H[c];
#pragma unroll
                for (uint32_t s = 0; s < 4; ++s) {
                    const uint32_t idx = 4u * c + s;
                    const uint32_t qw = W[s & 1u][c];
                    const uint32_t yw = Y[idx];
                    // Scale for lane I: I<4 -> sc[2*idx],
                    // I>=4 -> sc[2*idx+1] (CPU shuffle broadcast).
                    const int sca = (lane < 4)
                                            ? static_cast<int>(
                                                      sc[2 * idx])
                                            : static_cast<int>(
                                                      sc[2 * idx + 1]);
                    const uint32_t nibShift =
                            ((s & 2u) != 0u) ? 4u : 0u;
                    const uint32_t hiShift = 2u * s;
                    int acc4 = 0;
#pragma unroll
                    for (uint32_t k = 0; k < 4; ++k) {
                        const uint32_t b8 =
                                (qw >> (8u * k + nibShift)) & 0xFu;
                        const uint32_t h2 =
                                (hw >> (8u * k + hiShift)) & 3u;
                        const int wv =
                                static_cast<int>(b8 | (h2 << 4));
                        const int yv = static_cast<int>(
                                static_cast<int8_t>(static_cast<uint8_t>(
                                        yw >> (8u * k))));
                        acc4 += wv * yv;
                    }
                    sumi += sca * acc4;
                }
                acc = fmaf(k2, static_cast<float>(sumi), acc);
            }
        }
        if (lane < 8) running += acc;
    }
}

// ---- Single-matrix row GEMV (Q4_K or Q6_K weight x Q8_K activation).
// blockDim = (32, 8) (8 warps = 8 rows per block).
template<int TYPE>
__global__ void kQGemvKxQ8K(const uint8_t *__restrict__ w,
                            const uint8_t *__restrict__ x,
                            float *__restrict__ out, uint32_t rows,
                            uint32_t blocksPerRow, uint32_t rowBytes) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    const uint32_t lane = threadIdx.x;
    float sumf = 0.0f;
    constexpr uint32_t kBlkBytes =
            (TYPE == kTypeQ4K)     ? kQ4K_BYTES
            : (TYPE == kTypeQ5K)   ? kQ5K_BYTES
            : (TYPE == kTypeIQ4XS) ? kIQ4XS_BYTES
                                   : kQ6K_BYTES;
    // BLOCK BATCHING (2026-09-27): the per-block kxQ8KBlockDot ends in
    // a lane-0 serial `running = pFin + running`, giving a fully serial
    // dependency chain across the blocksPerRow blocks (the LM head at
    // 6 blocks/row measured 3.49 ms vs ~360 us at the 448 GB/s floor).
    // Compute 4 blocks into INDEPENDENT fresh accumulators (each
    // `pFin + 0.0f`, byte-identical to the original first-block add),
    // then serialize the four sumf adds IN BLOCK ORDER.  Float addition
    // is commutative, and each block's pFin chain (s32q -> shuffles ->
    // volatile dss*ss8/mns8*bi -> d8*(ms-mns)) is UNCHANGED, so the
    // bit-exact running sum is preserved exactly.
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        const uint8_t *bw0 = rp + static_cast<uint64_t>(b + 0u) * kBlkBytes;
        const uint8_t *bw1 = rp + static_cast<uint64_t>(b + 1u) * kBlkBytes;
        const uint8_t *bw2 = rp + static_cast<uint64_t>(b + 2u) * kBlkBytes;
        const uint8_t *bw3 = rp + static_cast<uint64_t>(b + 3u) * kBlkBytes;
        const uint8_t *ba0 = x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE;
        const uint8_t *ba1 = x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE;
        const uint8_t *ba2 = x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE;
        const uint8_t *ba3 = x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE;
        kxQ8KBlockDot<TYPE>(r0, bw0, ba0, lane);
        kxQ8KBlockDot<TYPE>(r1, bw1, ba1, lane);
        kxQ8KBlockDot<TYPE>(r2, bw2, ba2, lane);
        kxQ8KBlockDot<TYPE>(r3, bw3, ba3, lane);
        if (TYPE == kTypeQ6K) {
            // Q6K accumulates its row partial on ALL 8 active lanes
            // (each lane owns 32 distinct columns per block via
            // col = idx*32 + 4*lane + k), and the final shuffle-tree
            // store sums every lane.  The lane-0-only fold silently
            // dropped lanes 1-7's blocks 0-3 (7/8 of the columns of
            // 4 of 6 blocks = ~58% of the LM head dot product), which
            // the tolerant argmax tests masked -- the head's logits
            // measured ~3.3 low for a 12.6-wide range.  The fold's
            // add order (r0, then r1, r2, r3 left-assoc into the
            // running sum) is byte-identical to the serial tail loop
            // (running = pFin + running), so per-lane results are
            // bit-exact with the unbatched path.
            sumf = r0 + sumf;
            sumf = r1 + sumf;
            sumf = r2 + sumf;
            sumf = r3 + sumf;
        } else if (lane == 0) {
            // Q4_K / Q5_K / IQ4_XS accumulate the whole block on
            // lane 0 only (their scale/bsum reduction converges to
            // lane 0), so the fold stays lane-0-only there.
            sumf = r0 + sumf;
            sumf = r1 + sumf;
            sumf = r2 + sumf;
            sumf = r3 + sumf;
        }
    }
    for (; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * kBlkBytes;
        const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
        kxQ8KBlockDot<TYPE>(sumf, blkW, blkA, lane);
    }
    if (TYPE == kTypeQ4K || TYPE == kTypeQ5K ||
        TYPE == kTypeIQ4XS) {
        if (lane == 0) out[row] = sumf;
    } else {
        const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
        const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
        const float r = q + __shfl_xor_sync(0xffffffffu, q, 2);
        if (lane == 0) out[row] = r;
    }
}

// ---- Q6_K row GEMV (Q8_K activation), 4 ROWS PER WARP (2026-09-28).
// The generic kQGemvKxQ8K<kTypeQ6K> runs one row per warp and only
// sub-lanes 0-7 issue weight loads (24 of 32 lanes idle per
// instruction) -- the Q6_K LM head (151936 rows x 6 blocks x 210 B =
// 191 MB/token) measured 1850 us vs the ~370 us 448 GB/s floor.  Here
// lane group L>>3 selects the row (4 rows/warp) and sub-lane I = L&7
// keeps the EXACT per-(row, I) float chain (bprod, kOff, two chunk
// FMAs) of kxQ8KBlockDot<kTypeQ6K>, so each row's result is
// bit-identical (int sums order-free; float chain unchanged); the
// 8-lane shuffle-tree store (xor strides 4,1,2) stays within each
// 8-lane group.  Every lane now issues weight loads, quadrupling
// memory-level parallelism (each load covers 4 rows x 32 B, one
// sector per row).  Requires rows % 4 == 0 (all real uses: LM head
// 151936, ffn_down 1536, gate/up 8960); dispatch falls back to
// kQGemvKxQ8K<kTypeQ6K> otherwise.
// kQGemvQ6KxQ8K_4xW (~616 GB/s LM-head): all 32 lanes issue u32
// weight loads (see the block comment above), with WIDE selecting the
// section 9.23 2x-U16 weight loads / int-deref Y loads for the LM head.
template<bool WIDE = false>
__global__ void kQGemvQ6KxQ8K_4xW(const uint8_t *__restrict__ w,
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
    const uint8_t *rp =
            w + static_cast<uint64_t>(row) * rowBytes;
    float sumf = 0.0f;
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        kxQ8KBlockDot<kTypeQ6K, WIDE>(
                r0, rp + static_cast<uint64_t>(b + 0u) * kQ6K_BYTES,
                x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE, I);
        kxQ8KBlockDot<kTypeQ6K, WIDE>(
                r1, rp + static_cast<uint64_t>(b + 1u) * kQ6K_BYTES,
                x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE, I);
        kxQ8KBlockDot<kTypeQ6K, WIDE>(
                r2, rp + static_cast<uint64_t>(b + 2u) * kQ6K_BYTES,
                x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE, I);
        kxQ8KBlockDot<kTypeQ6K, WIDE>(
                r3, rp + static_cast<uint64_t>(b + 3u) * kQ6K_BYTES,
                x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE, I);
        sumf = r0 + sumf;
        sumf = r1 + sumf;
        sumf = r2 + sumf;
        sumf = r3 + sumf;
    }
    for (; b < blocksPerRow; ++b) {
        kxQ8KBlockDot<kTypeQ6K, WIDE>(
                sumf, rp + static_cast<uint64_t>(b) * kQ6K_BYTES,
                x + static_cast<uint64_t>(b) * kQ8K_STRIDE, I);
    }
    // 8-lane shuffle-tree per row within each 4-row lane group
    // (xor strides 4,1,2 stay inside the group's 8 lanes, same tree
    // as the 1-row-per-warp kernel: 4 XOR 4 = 0, ...).
    const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
    const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
    const float rr = q + __shfl_xor_sync(0xffffffffu, q, 2);
    if (I == 0) out[row] = rr;
}

// ---- Batched Q8_K prefill GEMM (K-quant weights x a TOKEN BATCH). ----
// The dense-qwen35 prefill used to run launchQGemv once per prompt
// token, re-reading every weight matrix seqLen times (pp == tg).  This
// kernel reads each weight block ONCE per 4-token group and dots it
// against 4 Q8_K activation planes, so prefill becomes weight-bound.
//
// BIT-EXACT with the per-token launchQGemv(Q8_K) path: one warp owns
// 4 tokens x 1 weight row -- lane group (lane>>3) = token rr in
// {0,1,2,3}, sub-lane Iid = lane&7 = the 8 sub-blocks of that row.
// kxQ8KBlockDot is called with GROUPED=true so the 8-lane reduction
// stays inside each token group (shuffle width 8) and every (row,
// token) reproduces the exact single-token association:
//   * Q4_K/Q5_K/IQ4_XS: lane-0 (Iid==0) serial block fold -- identical
//     `running = pFin + running` chain;
//   * Q6_K: the 8-lane tree (xor 4,1,2) of kQGemvKxQ8K, confined to the
//     token's group.
// So y[t*rows + row] is byte-identical to launchQGemv(m, token t).
// `x` is [seqLen][blocksPerRow] Q8_K blocks (actStride bytes/token);
// grid = (ceil(rows/8),), block = (32,8).  Rows must be %4-free: any
// rows count works (each warp handles 8 rows).
template<int TYPE>
__global__ void kQGemmKxQ8K(const uint8_t *__restrict__ w,
                            const uint8_t *__restrict__ x,
                            float *__restrict__ out, uint32_t rows,
                            uint32_t blocksPerRow, uint32_t rowBytes,
                            uint32_t actStride, uint32_t seqLen) {
    const uint32_t lane = threadIdx.x;
    // lane group (lane>>3) = token rr in {0,1,2,3}; sub-lane Iid = the
    // 8 sub-blocks of the row.  grid.y steps 4 tokens, so every weight
    // block is dotted against 4 token activations per read (vs ONE in
    // the per-token GEMV), which is the whole prefill win.
    const uint32_t rr = lane >> 3u;
    const uint32_t Iid = lane & 7u;
    const uint32_t tok = blockIdx.y * 4u + rr;
    const bool tokValid = tok < seqLen;
    const uint32_t tokc = tokValid ? tok : 0u;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    const uint8_t *xt = x + static_cast<uint64_t>(tokc) * actStride;
    constexpr uint32_t kBlkBytes =
            (TYPE == kTypeQ4K)     ? kQ4K_BYTES
            : (TYPE == kTypeQ5K)   ? kQ5K_BYTES
            : (TYPE == kTypeIQ4XS) ? kIQ4XS_BYTES
                                   : kQ6K_BYTES;
    float sumf = 0.0f;
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        kxQ8KBlockDot<TYPE, false, true>(
                r0, rp + static_cast<uint64_t>(b + 0u) * kBlkBytes,
                xt + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE, Iid);
        kxQ8KBlockDot<TYPE, false, true>(
                r1, rp + static_cast<uint64_t>(b + 1u) * kBlkBytes,
                xt + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE, Iid);
        kxQ8KBlockDot<TYPE, false, true>(
                r2, rp + static_cast<uint64_t>(b + 2u) * kBlkBytes,
                xt + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE, Iid);
        kxQ8KBlockDot<TYPE, false, true>(
                r3, rp + static_cast<uint64_t>(b + 3u) * kBlkBytes,
                xt + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE, Iid);
        if (TYPE == kTypeQ6K) {
            sumf = r0 + sumf;
            sumf = r1 + sumf;
            sumf = r2 + sumf;
            sumf = r3 + sumf;
        } else if (Iid == 0) {
            sumf = r0 + sumf;
            sumf = r1 + sumf;
            sumf = r2 + sumf;
            sumf = r3 + sumf;
        }
    }
    for (; b < blocksPerRow; ++b) {
        kxQ8KBlockDot<TYPE, false, true>(
                sumf, rp + static_cast<uint64_t>(b) * kBlkBytes,
                xt + static_cast<uint64_t>(b) * kQ8K_STRIDE, Iid);
    }
    float *op = out + static_cast<uint64_t>(tokc) * rows + row;
    if (TYPE == kTypeQ4K || TYPE == kTypeQ5K || TYPE == kTypeIQ4XS) {
        if (Iid == 0 && tokValid) *op = sumf;
    } else {
        const float p = sumf + __shfl_xor_sync(0xffffffffu, sumf, 4);
        const float q = p + __shfl_xor_sync(0xffffffffu, p, 1);
        const float r = q + __shfl_xor_sync(0xffffffffu, q, 2);
        if (Iid == 0 && tokValid) *op = r;
    }
}

// ---- Q5_K row GEMV (Q8_K activation), 4 ROWS PER WARP (2026-10-12).
// Direct analogue of kQGemvQ6KxQ8K_4xW for the Q5_K LM head.  The
// generic kQGemvKxQ8K<kTypeQ5K> runs one row per warp and only
// sub-lanes 0-7 issue weight loads (24 of 32 lanes idle per
// instruction): the IQ models' Q5_K LM head (151936 rows x 6 blocks x
// 176 B = 153-357 MB/token) measured 2.4-5.8 ms vs the ~0.36-0.85 ms
// 448 GB/s floor, while the SAME 4xW geometry already takes the Q6_K
// head to ~616 GB/s.  Here lane group (lane>>3) selects the row (4
// rows/warp) and sub-lane I = lane&7 keeps the EXACT per-(row, I)
// float chain of kxQ8KBlockDot<kTypeQ5K>: the s32q integer dot is
// order-free and the volatile no-FMA ds*s32q / dm*bsum fold plus the
// d8*(ms-mns) epilogue are executed by every I, so each row's output
// is byte-identical to the generic kernel's lane-0 result.  The
// terminal 8-lane shuffle tree (xor strides 4,1,2) stays inside each
// 4-row group and is the SAME association as the generic kernel's
// full-warp tree restricted to its 8 active lanes.  Requires
// rows % 4 == 0 (all real uses: LM head 151936/152064, ffn_down 1536,
// gate/up 8960); dispatch falls back to the generic kernel otherwise.
// Env gate TINYCODER_Q5K_LM_4XW (default ON).
// Group-local (4-rows-per-warp) Q5_K x Q8_K block dot, VECTORIZED.
// Same math as kxQ8KBlockDot<kTypeQ5K, false, true> but the 32-column
// integer dot is done 4 columns at a time with __dp4a (signed int8,
// exact) instead of 32 scalar byte-loads + per-byte nibble/sign ops.
// The Q5_K weight byte is (qs 4-bit nibble) | (qh bit ? 0x10), so a
// u32 of 4 consecutive columns yields the 4 packed weights exactly:
//   low4 = (half==0 ? wb4 : wb4>>4) & 0x0F0F0F0F
//   nz   = ((qhb4 & (u*0x01010101)) + 0x7F7F7F7F) & 0x80808080
//   w32  = low4 | (nz >> 3)                 // 0x10 per set qh bit
// Because integer addition is exact and associative, s32q is bit-
// identical to the scalar body; the dsv/mnv float fold and the
// d8*(ms-mns) epilogue are unchanged, so every row's output matches
// the generic kernel byte-for-byte.
__device__ __forceinline__ void kxQ5KBlockDotVec(
        float &running, const uint8_t *__restrict__ blkW,
        const uint8_t *__restrict__ blkA, uint32_t lane) {
    const uint32_t Iid = lane & 7u;
    const int16_t *bsums =
            reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
    const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
    int s32q = 0;
    float dsv = 0.0f, mnv = 0.0f;
    {
        const uint32_t i = Iid;
        const float d = __half2float(
                *reinterpret_cast<const __half *>(blkW));
        const float dmin = __half2float(
                *reinterpret_cast<const __half *>(blkW + 2));
        const uint8_t *scales = blkW + 4;
        const uint8_t *qh = blkW + 16;
        const uint8_t *qs = blkW + 48;
        uint8_t sc, m;
        if (i < 4) {
            sc = scales[i] & 63u;
            m = scales[i + 4] & 63u;
        } else {
            sc = static_cast<uint8_t>(
                    (scales[i + 4] & 0xFu) |
                    ((scales[i - 4] >> 6) << 4));
            m = static_cast<uint8_t>(
                    (scales[i + 4] >> 4) |
                    ((scales[i] >> 6) << 4));
        }
        dsv = d * static_cast<float>(sc);
        mnv = dmin * static_cast<float>(m);
        const uint32_t c = i >> 1u;
        const uint32_t half = i & 1u;
        const uint32_t u = 1u << (2u * c + half);
        const uint8_t *qsp = qs + 32u * c;
        const int8_t *yp =
                reinterpret_cast<const int8_t *>(blkA) + i * 32u;
#pragma unroll
        for (uint32_t j = 0; j < 32u; j += 4u) {
            // Q5_K rows are 176-B aligned (176 = 16*11), so qs (block
            // +48) and qh (block +16) stay 16-B aligned at 4-B steps
            // -> direct aligned u32 loads (the generic kernel's even
            // offsets made these memcpy for WIDE; here they are
            // provably aligned).  Same bytes, bit-identical.
            const uint32_t wb4 =
                    *reinterpret_cast<const uint32_t *>(qsp + j);
            const uint32_t qhb4 =
                    *reinterpret_cast<const uint32_t *>(qh + j);
            const uint32_t yb4 =
                    *reinterpret_cast<const uint32_t *>(yp + j);
            const uint32_t low4 =
                    (half == 0u) ? (wb4 & 0x0F0F0F0Fu)
                                 : ((wb4 >> 4) & 0x0F0F0F0Fu);
            const uint32_t masked = qhb4 & (u * 0x01010101u);
            const uint32_t nz =
                    (masked + 0x7F7F7F7Fu) & 0x80808080u;
            const uint32_t w32 = low4 | (nz >> 3);
            s32q = __dp4a(static_cast<int>(w32),
                          static_cast<int>(yb4), s32q);
        }
    }
    // LEAN per-lane fold (see kxQ4KBlockDotVec): each lane folds its
    // own sub-block; the kernel's final 8-lane tree sums the row.
    const float bi = static_cast<float>(
            static_cast<int>(bsums[2 * Iid]) +
            static_cast<int>(bsums[2 * Iid + 1]));
    running = __fadd_rn(
            __fmul_rn(d8,
                      __fsub_rn(__fmul_rn(dsv,
                                          static_cast<float>(s32q)),
                                __fmul_rn(mnv, bi))),
            running);
}

__global__ void kQGemvQ5KxQ8K_4xW(const uint8_t *__restrict__ w,
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
        kxQ5KBlockDotVec(
                r0,
                rp + static_cast<uint64_t>(b + 0u) * kQ5K_BYTES,
                x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE, I);
        kxQ5KBlockDotVec(
                r1,
                rp + static_cast<uint64_t>(b + 1u) * kQ5K_BYTES,
                x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE, I);
        kxQ5KBlockDotVec(
                r2,
                rp + static_cast<uint64_t>(b + 2u) * kQ5K_BYTES,
                x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE, I);
        kxQ5KBlockDotVec(
                r3,
                rp + static_cast<uint64_t>(b + 3u) * kQ5K_BYTES,
                x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE, I);
        sumf = r0 + sumf;
        sumf = r1 + sumf;
        sumf = r2 + sumf;
        sumf = r3 + sumf;
    }
    for (; b < blocksPerRow; ++b) {
        kxQ5KBlockDotVec(
                sumf, rp + static_cast<uint64_t>(b) * kQ5K_BYTES,
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

// Group-local (4-rows-per-warp) Q4_K x Q8_K block dot, VECTORIZED.
// Direct analogue of kxQ5KBlockDotVec.  Same math as
// kxQ8KBlockDot<kTypeQ4K, false, true> (lines above) but the 32-column
// integer dot is done 4 columns at a time with __dp4a (signed int8,
// exact) instead of 32 scalar byte-loads + per-byte nibble/sign ops.
// Q4_K element map (bit-exact with the scalar body and with
// matMulVecBatchQ4K_Q8K_AVX2): for sub-block i (0..7), q-byte base
// qsv + 32*(i>>1), column j: byte j -> low nibble when (i&1)==0 else
// high nibble.  A u32 of 4 consecutive cols therefore yields the 4
// packed 4-bit weights exactly:
//   w32 = (half==0 ? wb4 : wb4>>4) & 0x0F0F0F0F
// Because integer addition is exact and associative, s32q is bit-
// identical to the scalar body; the dsv/mnv float fold (volatile
// no-FMA mul-then-add) and the d8*(ms-mns) epilogue are unchanged, so
// every row's output matches the generic kernel byte-for-byte.
__device__ __forceinline__ void kxQ4KBlockDotVec(
        float &running, const uint8_t *__restrict__ blkW,
        const uint8_t *__restrict__ blkA, uint32_t lane) {
    const uint32_t Iid = lane & 7u;
    const int16_t *bsums =
            reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
    const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
    int s32q = 0;
    float dsv = 0.0f, mnv = 0.0f;
    {
        const uint32_t i = Iid;
        const float d = __half2float(
                *reinterpret_cast<const __half *>(blkW));
        const float dmin = __half2float(
                *reinterpret_cast<const __half *>(blkW + 2));
        const uint8_t *scales = blkW + 4;
        const uint8_t *qsv = blkW + 16;
        uint8_t sc, m;
        if (i < 4) {
            sc = scales[i] & 63u;
            m = scales[i + 4] & 63u;
        } else {
            sc = static_cast<uint8_t>(
                    (scales[i + 4] & 0xFu) |
                    ((scales[i - 4] >> 6) << 4));
            m = static_cast<uint8_t>(
                    (scales[i + 4] >> 4) |
                    ((scales[i] >> 6) << 4));
        }
        dsv = d * static_cast<float>(sc);
        mnv = dmin * static_cast<float>(m);
        const uint32_t half = i & 1u;
        const uint8_t *qsp = qsv + 32u * (i >> 1u);
        const int8_t *yp =
                reinterpret_cast<const int8_t *>(blkA) + i * 32u;
#pragma unroll
        for (uint32_t j = 0; j < 32u; j += 4u) {
            // Q4_K rows are 144-B aligned (144 = 16*9), so qs (block
            // +16) stays 16-B aligned at 4-B steps -> direct aligned
            // u32 loads (same bytes as the scalar body, bit-identical).
            const uint32_t wb4 =
                    *reinterpret_cast<const uint32_t *>(qsp + j);
            const uint32_t yb4 =
                    *reinterpret_cast<const uint32_t *>(yp + j);
            const uint32_t w32 =
                    (half == 0u) ? (wb4 & 0x0F0F0F0Fu)
                                 : ((wb4 >> 4) & 0x0F0F0F0Fu);
            s32q = __dp4a(static_cast<int>(w32),
                          static_cast<int>(yb4), s32q);
        }
    }
    // LEAN: each lane folds its OWN sub-block contribution and the
    // kernel's final 8-lane shuffle tree combines the lanes once per
    // ROW.  The old body gathered all 8 sub-block partials into lane 0
    // per 256-block with 24 shuffles (plus a serialized lane-0 float
    // chain), which made this the decode bottleneck: the kernel is
    // instruction-issue-bound, not bandwidth-bound, and cutting the
    // per-block gather is worth ~1.5-2x.  The dot is a deterministic
    // fp32 reassociation of the reference order (top-1 coherence is
    // the contract for this path, like the INT8 prefill).
    const float bi = static_cast<float>(
            static_cast<int>(bsums[2 * Iid]) +
            static_cast<int>(bsums[2 * Iid + 1]));
    running = __fadd_rn(
            __fmul_rn(d8,
                      __fsub_rn(__fmul_rn(dsv,
                                          static_cast<float>(s32q)),
                                __fmul_rn(mnv, bi))),
            running);
}
