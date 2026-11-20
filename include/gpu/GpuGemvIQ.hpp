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

// GpuGemvIQ.hpp -- IQ2_XS/IQ2_S/IQ3_XXS/IQ4_XS GEMV and fused Q/K/V decode.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// Multiply a lane's integer sumi by its per-(ib32, l) ls multiplier.
template<int TYPE>
__device__ __forceinline__ int q8kScaleMuli(const uint8_t *__restrict__ w,
                                            int ci, uint32_t ib32, uint32_t l) {
    if (TYPE == kTypeIQ2S) {
        const uint8_t *scales = w + 74;
        const int ls = (l < 2) ? (1 + 2 * (scales[ib32] & 0xf))
                               : (1 + 2 * (scales[ib32] >> 4));
        return ci * ls;
    }
    if (TYPE == kTypeIQ2XS) {
        // IQ2_XS: scales[8] at offset 66; low nibble for l=0,1 and high
        // nibble for l=2,3; ls = 2*nib+1 (the CPU db = d*(0.5+nib)*0.25
        // = d*ls/8, hence the 0.125 final scale, same as IQ2_S).
        const uint8_t *scales = w + 66;
        const int nib = (l < 2) ? (scales[ib32] & 0xf)
                                : (scales[ib32] >> 4);
        return ci * (2 * nib + 1);
    }
    if (TYPE == kTypeIQ3_XXS) {
        uint32_t aux32;
        std::memcpy(&aux32, w + 2 + 64u + 4u * ib32, sizeof(uint32_t));
        return ci * static_cast<int>(2 * (aux32 >> 28) + 1);
    }
    // kTypeIQ3S
    const uint8_t *scales = w + 106;
    const int ls = static_cast<int>(
            2 * ((ib32 & 1u) ? (scales[ib32 >> 1u] >> 4)
                             : (scales[ib32 >> 1u] & 0xf)) +
            1);
    return ci * ls;
}

// Warp-per-row Q8K GEMV over the pre-quantized activation (q8k).
//
// Per (ib32, l) lane: integer sumi (matches vec_dot_iq*_q8_K generic).
// The per-ib32 bsum is IQ2_S: ls1*sumi(l=0..1) + ls2*sumi(l=2..3)  (per
// 32-group ib32) IQ3_XXS: ls*sumi(l=0..3) IQ3_S:  ls*sumi(l=0..3) per
// 32-wide half pairing (ib32 pairs) Since everything is integer,
// multiplying each lane's sumi by its own ls FIRST and then doing a
// plain 32-lane tree reduction reproduces the exact same integer bsum
// (addition distributes over the scalar ls, including the split ls1/ls2
// case).
template<int TYPE, bool DP4A = false>
__global__ void __launch_bounds__(256) kQGemvQ8K(
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ x,
        float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow,
        uint32_t rowBytes, bool grsm = false) {
    extern __shared__ uint8_t sGrid[];
    const uint8_t *gsm = nullptr;
    if (grsm) {
        const uint32_t nw = (TYPE == kTypeIQ2S)    ? 2048u
                            : (TYPE == kTypeIQ2XS) ? 1024u
                            : (TYPE == kTypeIQ3S)  ? 512u
                                                   : 256u;
        const uint32_t *src =
                (TYPE == kTypeIQ2S)
                        ? reinterpret_cast<const uint32_t *>(c_iq2s_grid)
                : (TYPE == kTypeIQ2XS)
                        ? reinterpret_cast<const uint32_t *>(c_iq2xs_grid)
                : (TYPE == kTypeIQ3S) ? c_iq3s_grid
                                      : c_iq3xxs_grid;
        uint32_t *dst = reinterpret_cast<uint32_t *>(sGrid);
        const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
        const uint32_t nth = blockDim.x * blockDim.y;
        for (uint32_t i = tid; i < nw; i += nth) dst[i] = src[i];
        __syncthreads();
        gsm = sGrid;
    }
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    const uint32_t lane = threadIdx.x;
    const uint32_t ib32 = lane >> 2u;// 0..7 within a block
    const uint32_t l = lane & 3u;    // 0..3
    const uint32_t bwBytes = (TYPE == kTypeIQ2S)      ? kIQ2S_BYTES
                             : (TYPE == kTypeIQ2XS)   ? kIQ2XS_BYTES
                             : (TYPE == kTypeIQ3_XXS) ? kIQ3XXS_BYTES
                                                      : kIQ3S_BYTES;
    float sumf = 0.0f;

    // BLOCK BATCHING (2026-09-27): process blocks in groups of 4.  The
    // old loop was one fully-serial dependency chain per block
    // (q8kLaneSumi loads -> ls scale -> 5-deep shuffle tree -> lane-0
    // float add), measured at ~2000 cycles/block-iteration.  Each group
    // now issues 4 INDEPENDENT (load, scale, tree) chains, giving
    // 4-way memory-level and instruction-level parallelism.  The four
    // float `sumf += d*bsum` updates still run serially at lane 0 in
    // block order, so the float accumulation order -- and therefore the
    // bit-exact per-layer logits -- is unchanged (the integer sumi/bsum
    // are order-free; the serial d*bsum sequence is byte-identical).
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        const uint8_t *bw0 = rp + static_cast<uint64_t>(b + 0u) * bwBytes;
        const uint8_t *bw1 = rp + static_cast<uint64_t>(b + 1u) * bwBytes;
        const uint8_t *bw2 = rp + static_cast<uint64_t>(b + 2u) * bwBytes;
        const uint8_t *bw3 = rp + static_cast<uint64_t>(b + 3u) * bwBytes;
        const uint8_t *ba0 = x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE;
        const uint8_t *ba1 = x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE;
        const uint8_t *ba2 = x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE;
        const uint8_t *ba3 = x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE;

        // 4 independent per-lane integer sums (all loads issue before
        // any of them is consumed -> 4-deep memory-level parallelism).
        int ci0 = q8kLaneSumi<DP4A>(TYPE, bw0,
                                    reinterpret_cast<const int8_t *>(ba0 + kQ8K_Q), ib32, l, gsm);
        int ci1 = q8kLaneSumi<DP4A>(TYPE, bw1,
                                    reinterpret_cast<const int8_t *>(ba1 + kQ8K_Q), ib32, l, gsm);
        int ci2 = q8kLaneSumi<DP4A>(TYPE, bw2,
                                    reinterpret_cast<const int8_t *>(ba2 + kQ8K_Q), ib32, l, gsm);
        int ci3 = q8kLaneSumi<DP4A>(TYPE, bw3,
                                    reinterpret_cast<const int8_t *>(ba3 + kQ8K_Q), ib32, l, gsm);

        // Per-lane ls scaling (order-free integer math).
        ci0 = q8kScaleMuli<TYPE>(bw0, ci0, ib32, l);
        ci1 = q8kScaleMuli<TYPE>(bw1, ci1, ib32, l);
        ci2 = q8kScaleMuli<TYPE>(bw2, ci2, ib32, l);
        ci3 = q8kScaleMuli<TYPE>(bw3, ci3, ib32, l);

        // 4 independent 32-lane trees -> per-block integer bsums.  The
        // trees use the same __shfl_xor_sync mask/order as before, so
        // each bsum is bit-identical to the unbatched loop's.
        int bs0 = ci0 + __shfl_xor_sync(0xffffffffu, ci0, 1);
        bs0 += __shfl_xor_sync(0xffffffffu, bs0, 2);
        bs0 += __shfl_xor_sync(0xffffffffu, bs0, 4);
        bs0 += __shfl_xor_sync(0xffffffffu, bs0, 8);
        bs0 += __shfl_xor_sync(0xffffffffu, bs0, 16);
        int bs1 = ci1 + __shfl_xor_sync(0xffffffffu, ci1, 1);
        bs1 += __shfl_xor_sync(0xffffffffu, bs1, 2);
        bs1 += __shfl_xor_sync(0xffffffffu, bs1, 4);
        bs1 += __shfl_xor_sync(0xffffffffu, bs1, 8);
        bs1 += __shfl_xor_sync(0xffffffffu, bs1, 16);
        int bs2 = ci2 + __shfl_xor_sync(0xffffffffu, ci2, 1);
        bs2 += __shfl_xor_sync(0xffffffffu, bs2, 2);
        bs2 += __shfl_xor_sync(0xffffffffu, bs2, 4);
        bs2 += __shfl_xor_sync(0xffffffffu, bs2, 8);
        bs2 += __shfl_xor_sync(0xffffffffu, bs2, 16);
        int bs3 = ci3 + __shfl_xor_sync(0xffffffffu, ci3, 1);
        bs3 += __shfl_xor_sync(0xffffffffu, bs3, 2);
        bs3 += __shfl_xor_sync(0xffffffffu, bs3, 4);
        bs3 += __shfl_xor_sync(0xffffffffu, bs3, 8);
        bs3 += __shfl_xor_sync(0xffffffffu, bs3, 16);

        // Uniform per-block scale products (identical for every lane;
        // lane 0 consumes them in the serial accumulation below).
        const float d0 =
                __half2float(*reinterpret_cast<const __half *>(bw0)) *
                *reinterpret_cast<const float *>(ba0 + kQ8K_D);
        const float d1 =
                __half2float(*reinterpret_cast<const __half *>(bw1)) *
                *reinterpret_cast<const float *>(ba1 + kQ8K_D);
        const float d2 =
                __half2float(*reinterpret_cast<const __half *>(bw2)) *
                *reinterpret_cast<const float *>(ba2 + kQ8K_D);
        const float d3 =
                __half2float(*reinterpret_cast<const __half *>(bw3)) *
                *reinterpret_cast<const float *>(ba3 + kQ8K_D);

        // Lane 0 accumulates serially IN BLOCK ORDER (bit-exact). fmaf
        // would differ by 1 ulp per block, compounding through the KV
        // cache across decode steps and flipping the distribution by
        // step 3+.
        if (lane == 0) {
            sumf += d0 * static_cast<float>(bs0);
            sumf += d1 * static_cast<float>(bs1);
            sumf += d2 * static_cast<float>(bs2);
            sumf += d3 * static_cast<float>(bs3);
        }
    }
    // Remainder (blocksPerRow % 4) -- original single-block body.
    for (; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * bwBytes;
        const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
        const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA + kQ8K_Q);
        const float yd = *reinterpret_cast<const float *>(blkA + kQ8K_D);
        int ci = q8kLaneSumi<DP4A>(TYPE, blkW, y8, ib32, l, gsm);
        ci = q8kScaleMuli<TYPE>(blkW, ci, ib32, l);
        int bsum = ci + __shfl_xor_sync(0xffffffffu, ci, 1);
        bsum += __shfl_xor_sync(0xffffffffu, bsum, 2);
        bsum += __shfl_xor_sync(0xffffffffu, bsum, 4);
        bsum += __shfl_xor_sync(0xffffffffu, bsum, 8);
        bsum += __shfl_xor_sync(0xffffffffu, bsum, 16);
        if (lane == 0) {
            const float dval =
                    __half2float(*reinterpret_cast<const __half *>(blkW)) * yd;
            sumf += dval * static_cast<float>(bsum);
        }
    }
    const float scale = (TYPE == kTypeIQ2S)      ? 0.125f
                        : (TYPE == kTypeIQ2XS)   ? 0.125f
                        : (TYPE == kTypeIQ3_XXS) ? 0.25f
                                                 : 1.0f;
    if (lane == 0) out[row] = scale * sumf;
}

// ---- IQ Q8_K GEMV, 4 ROWS PER WARP (2026-10-12).  Bit-exact
// restructure of kQGemvQ8K<TYPE,DP4A>.  The generic kernel runs one
// ROW per warp: every row re-reads the whole Q8_K activation and only
// its own weight block, giving both an activation-re-read cost (all
// 8960 rows re-read the 1.5 KB activation through L2) and one weight
// block in flight per warp.  Here the 32 lanes are partitioned into 4
// groups of 8; group G = lane>>3 owns row = base+G, and the 8 lanes of
// each group split the block's 8 ib32 sub-blocks (ib32 = lane&7, l =
// 0).  The lane loads its 8 activation bytes ONCE per block and reuses
// them for its row's weight (activation-re-read eliminated), and the
// warp keeps 4 weight blocks in flight.  Each lane holds the WHOLE
// per-block scalar for its row (no 32-lane tree), so the per-row
// block-order float accumulation is byte-identical to kQGemvQ8K's
// lane-0 `sumf += d*bsum`; only the final per-type scale is applied by
// the group leader (lane&7==0).  Requires rows%4==0.
// Env gate TINYCODER_IQ_4XW (default ON).
template<int TYPE, bool DP4A = false>
__device__ __forceinline__ int q8kBlockSum(
        const uint8_t *__restrict__ bw, const uint8_t *__restrict__ ba,
        uint32_t ype, const uint8_t *__restrict__ gsm) {
    const int8_t *y8b = reinterpret_cast<const int8_t *>(ba + kQ8K_Q);
    int sumi = 0;
#pragma unroll
    for (uint32_t ll = 0; ll < 4; ++ll) {
        const uint64_t yv = __ldg(reinterpret_cast<const uint64_t *>(
                &y8b[ype * 32u + ll * 8u]));
        int c = q8kLaneSumiV<DP4A>(TYPE, bw, yv, ype, ll, gsm);
        c = q8kScaleMuli<TYPE>(bw, c, ype, ll);
        sumi += c;
    }
    sumi += __shfl_xor_sync(0xffffffffu, sumi, 1);
    sumi += __shfl_xor_sync(0xffffffffu, sumi, 2);
    sumi += __shfl_xor_sync(0xffffffffu, sumi, 4);
    return sumi;
}

template<int TYPE, bool DP4A = false>
__global__ void __launch_bounds__(256) kQGemvQ8K_4xW(
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ x,
        float *__restrict__ out, uint32_t rows, uint32_t blocksPerRow,
        uint32_t rowBytes, bool grsm = false) {
    extern __shared__ uint8_t sGrid[];
    const uint8_t *gsm = nullptr;
    if (grsm) {
        const uint32_t nw = (TYPE == kTypeIQ2S)    ? 2048u
                            : (TYPE == kTypeIQ2XS) ? 1024u
                            : (TYPE == kTypeIQ3S)  ? 512u
                                                   : 256u;
        const uint32_t *src =
                (TYPE == kTypeIQ2S)
                        ? reinterpret_cast<const uint32_t *>(c_iq2s_grid)
                : (TYPE == kTypeIQ2XS)
                        ? reinterpret_cast<const uint32_t *>(c_iq2xs_grid)
                : (TYPE == kTypeIQ3S) ? c_iq3s_grid
                                      : c_iq3xxs_grid;
        uint32_t *dst = reinterpret_cast<uint32_t *>(sGrid);
        const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
        const uint32_t nth = blockDim.x * blockDim.y;
        for (uint32_t i = tid; i < nw; i += nth) dst[i] = src[i];
        __syncthreads();
        gsm = sGrid;
    }
    const uint32_t lane = threadIdx.x;
    const uint32_t ype = lane & 7u;// ib32 sub-block this lane owns
    const uint32_t rr = lane >> 3u;// 0..3 row within the warp
    const uint32_t base = (blockIdx.x * blockDim.y + threadIdx.y) * 4u;
    if (base >= rows) return;
    const uint32_t row = base + rr;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    const uint32_t bwBytes = (TYPE == kTypeIQ2S)      ? kIQ2S_BYTES
                             : (TYPE == kTypeIQ2XS)   ? kIQ2XS_BYTES
                             : (TYPE == kTypeIQ3_XXS) ? kIQ3XXS_BYTES
                                                      : kIQ3S_BYTES;
    float sumf = 0.0f;
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        const uint8_t *ba0 = x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE;
        const uint8_t *ba1 = x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE;
        const uint8_t *ba2 = x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE;
        const uint8_t *ba3 = x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE;
        const uint8_t *bw0 = rp + static_cast<uint64_t>(b + 0u) * bwBytes;
        const uint8_t *bw1 = rp + static_cast<uint64_t>(b + 1u) * bwBytes;
        const uint8_t *bw2 = rp + static_cast<uint64_t>(b + 2u) * bwBytes;
        const uint8_t *bw3 = rp + static_cast<uint64_t>(b + 3u) * bwBytes;
        const int bs0 = q8kBlockSum<TYPE, DP4A>(bw0, ba0, ype, gsm);
        const int bs1 = q8kBlockSum<TYPE, DP4A>(bw1, ba1, ype, gsm);
        const int bs2 = q8kBlockSum<TYPE, DP4A>(bw2, ba2, ype, gsm);
        const int bs3 = q8kBlockSum<TYPE, DP4A>(bw3, ba3, ype, gsm);
        const float d0 =
                __half2float(*reinterpret_cast<const __half *>(bw0)) *
                *reinterpret_cast<const float *>(ba0 + kQ8K_D);
        const float d1 =
                __half2float(*reinterpret_cast<const __half *>(bw1)) *
                *reinterpret_cast<const float *>(ba1 + kQ8K_D);
        const float d2 =
                __half2float(*reinterpret_cast<const __half *>(bw2)) *
                *reinterpret_cast<const float *>(ba2 + kQ8K_D);
        const float d3 =
                __half2float(*reinterpret_cast<const __half *>(bw3)) *
                *reinterpret_cast<const float *>(ba3 + kQ8K_D);
        sumf += d0 * static_cast<float>(bs0);
        sumf += d1 * static_cast<float>(bs1);
        sumf += d2 * static_cast<float>(bs2);
        sumf += d3 * static_cast<float>(bs3);
    }
    for (; b < blocksPerRow; ++b) {
        const uint8_t *blkW = rp + static_cast<uint64_t>(b) * bwBytes;
        const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
        const int bsum = q8kBlockSum<TYPE, DP4A>(blkW, blkA, ype, gsm);
        const float dval =
                __half2float(*reinterpret_cast<const __half *>(blkW)) *
                *reinterpret_cast<const float *>(blkA + kQ8K_D);
        sumf += dval * static_cast<float>(bsum);
    }
    const float scale = (TYPE == kTypeIQ2S)      ? 0.125f
                        : (TYPE == kTypeIQ2XS)   ? 0.125f
                        : (TYPE == kTypeIQ3_XXS) ? 0.25f
                                                 : 1.0f;
    if (ype == 0u) out[row] = scale * sumf;
}

// ------------------------------------------------------------------
// FUSED gate+up+silu (decode seqLen==1, dense FFN, 2026-09-28).
//
// The per-layer FFN path used to launch gate, up and kSiluMul as THREE
// kernels per layer (40 layers => ~120 launches/token just for the
// FFN).  The fused kernels below compute BOTH row dots in one warp and
// fold the silu(gate)*up elementwise product into the gate store, so
// ONE launch replaces the three, and (for the Q8K-int path) the fp32
// activation is quantized to Q8_K ONCE and shared by both matrices.
//
// BIT-EXACTNESS: each matrix's accumulation is byte-identical to the
// separate-kernel math:
//   * float path (kQGemvFusedGU_IQ2XS): identical per-lane fmaf chain
//     over blocks (incl. the 4-way a0..a3 block batching with serial
//     folds) and the identical 5-shuffle warp tree per matrix;
//   * Q8K-int path (kQGemvQ8KFusedGU): identical q8kLaneSumi +
//     q8kScaleMuli integer sums, identical per-block int tree, and the
//     serial lane-0 `sumf += d*bsum` in block order, with the trailing
//     per-type scale applied before the silu product exactly like the
//     separate kernels + kSiluMul.
// The fused silu = x/(1+__expf(-x)) -- the kSiluMul expression --
// applied to the two fully-reduced row values.

// ---- Fused gate+up+silu for IQ2_XS (float dequant-dot).
__global__ void __launch_bounds__(256) kQGemvFusedGU_IQ2XS(
        const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
        const float *__restrict__ x, float *__restrict__ out,
        uint32_t rows, uint32_t cols, uint32_t rowBytes,
        uint32_t blocksPerRow) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint8_t *rg = wg + static_cast<uint64_t>(row) * rowBytes;
    const uint8_t *ru = wu + static_cast<uint64_t>(row) * rowBytes;
    const uint32_t lane = threadIdx.x;
    const uint32_t e0 = lane * 8u;
    const uint32_t ib32 = e0 >> 5u;    // sub-block 0..7
    const uint32_t l = (e0 >> 3u) & 3u;// 0..3
    float accg = 0.0f, accu = 0.0f;
    uint32_t b = 0;
    // 4-way block batching (mirrors kQGemv<kTypeIQ2XS>): each group
    // issues 8 independent (load, decode, fmaf-chain) streams (4 per
    // matrix) sharing the same xs activation window per block.
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
#pragma unroll 32
        for (uint32_t bb = 0; bb < 4; ++bb) {
            const uint8_t *crpG =
                    rg + static_cast<uint64_t>(bb) * kIQ2XS_BYTES;
            const uint8_t *crpU =
                    ru + static_cast<uint64_t>(bb) * kIQ2XS_BYTES;
            __half dG = *reinterpret_cast<const __half *>(crpG + 0);
            __half dU = *reinterpret_cast<const __half *>(crpU + 0);
            float dfG = __half2float(dG), dfU = __half2float(dU);
            const uint16_t *qsG =
                    reinterpret_cast<const uint16_t *>(crpG + 2);
            const uint16_t *qsU =
                    reinterpret_cast<const uint16_t *>(crpU + 2);
            const uint8_t *scG = crpG + 66;
            const uint8_t *scU = crpU + 66;
            float gb0 = dfG * (0.5f + static_cast<float>(scG[ib32] & 0xf)) * 0.25f;
            float gb1 = dfG * (0.5f + static_cast<float>(scG[ib32] >> 4)) * 0.25f;
            float ub0 = dfU * (0.5f + static_cast<float>(scU[ib32] & 0xf)) * 0.25f;
            float ub1 = dfU * (0.5f + static_cast<float>(scU[ib32] >> 4)) * 0.25f;
            float dlG = (l >= 2u) ? gb1 : gb0;
            float dlU = (l >= 2u) ? ub1 : ub0;
            uint16_t qg = qsG[ib32 * 4u + l];
            uint16_t qu = qsU[ib32 * 4u + l];
            const uint64_t gvG = c_iq2xs_grid[qg & 0x1FFu];
            const uint64_t gvU = c_iq2xs_grid[qu & 0x1FFu];
            uint8_t sgnG = c_ksigns_iq2xs[qg >> 9];
            uint8_t sgnU = c_ksigns_iq2xs[qu >> 9];
            const float *xs = x + (b + bb) * 256u + e0;
            float &slotG = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                     : (bb == 2u)       ? a2
                                                        : a3;
            float &slotU = (bb == 0u) ? u0 : (bb == 1u) ? u1
                                     : (bb == 2u)       ? u2
                                                        : u3;
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                float wG = dlG *
                           static_cast<float>((gvG >> (8u * j)) & 0xFFu) *
                           ((sgnG & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                float wU = dlU *
                           static_cast<float>((gvU >> (8u * j)) & 0xFFu) *
                           ((sgnU & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t ck = (b + bb) * 256u + e0 + j;
                const float xv = (ck < cols) ? xs[j] : 0.0f;
                slotG = fmaf(wG, xv, slotG);
                slotU = fmaf(wU, xv, slotU);
            }
        }
        accg += a0;
        accg += a1;
        accg += a2;
        accg += a3;
        accu += u0;
        accu += u1;
        accu += u2;
        accu += u3;
        rg += 4u * kIQ2XS_BYTES;
        ru += 4u * kIQ2XS_BYTES;
    }
    // Remainder (blocksPerRow % 4): original single-block serial chain.
    for (; b < blocksPerRow; ++b) {
        __half dG = *reinterpret_cast<const __half *>(rg + 0);
        __half dU = *reinterpret_cast<const __half *>(ru + 0);
        float dfG = __half2float(dG), dfU = __half2float(dU);
        const uint16_t *qsG = reinterpret_cast<const uint16_t *>(rg + 2);
        const uint16_t *qsU = reinterpret_cast<const uint16_t *>(ru + 2);
        const uint8_t *scG = rg + 66;
        const uint8_t *scU = ru + 66;
        float gb0 = dfG * (0.5f + static_cast<float>(scG[ib32] & 0xf)) * 0.25f;
        float gb1 = dfG * (0.5f + static_cast<float>(scG[ib32] >> 4)) * 0.25f;
        float ub0 = dfU * (0.5f + static_cast<float>(scU[ib32] & 0xf)) * 0.25f;
        float ub1 = dfU * (0.5f + static_cast<float>(scU[ib32] >> 4)) * 0.25f;
        float dlG = (l >= 2u) ? gb1 : gb0;
        float dlU = (l >= 2u) ? ub1 : ub0;
        uint16_t qg = qsG[ib32 * 4u + l];
        uint16_t qu = qsU[ib32 * 4u + l];
        const uint64_t gvG = c_iq2xs_grid[qg & 0x1FFu];
        const uint64_t gvU = c_iq2xs_grid[qu & 0x1FFu];
        uint8_t sgnG = c_ksigns_iq2xs[qg >> 9];
        uint8_t sgnU = c_ksigns_iq2xs[qu >> 9];
        const float *xs = x + b * 256u + e0;
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            float wG = dlG * static_cast<float>((gvG >> (8u * j)) & 0xFFu) *
                       ((sgnG & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
            float wU = dlU * static_cast<float>((gvU >> (8u * j)) & 0xFFu) *
                       ((sgnU & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
            const uint32_t ck = b * 256u + e0 + j;
            const float xv = (ck < cols) ? xs[j] : 0.0f;
            accg = fmaf(wG, xv, accg);
            accu = fmaf(wU, xv, accu);
        }
        rg += kIQ2XS_BYTES;
        ru += kIQ2XS_BYTES;
    }
    // Two independent 5-shuffle warp trees (identical order per matrix).
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        accg += __shfl_xor_sync(0xffffffffu, accg, off);
        accu += __shfl_xor_sync(0xffffffffu, accu, off);
    }
    if (lane == 0) {
        const float g = accg;
        out[row] = (g / (1.0f + __expf(-g))) * accu;
    }
}

// ---- Fused gate+up+silu for Q2_K (float dequant-dot) ----
// The qwen2.5-coder-1.5b q2_k FFN gate/up are Q2_K (type 10), which
// the IQ-only fused dispatcher did NOT cover -- the FFN ran as THREE
// launches (gate + up + kSiluMul) per layer.  This kernel computes both
// row dots in one warp and folds silu(gate)*up into the store, exactly
// like the IQ2XS fused kernel: each matrix's fmaf chain + 4-way a0..a3
// block batching + in-order folds + 5-shuffle warp tree are the
// byte-identical kQGemv<kTypeQ2K> math; the silu is the kSiluMul
// expression.  4-way block batching gives 8 independent (load, dequant,
// fmaf) streams (4 per matrix) sharing the x activation window.
__global__ void __launch_bounds__(256) kQGemvFusedGU_Q2K(
        const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
        const float *__restrict__ x, float *__restrict__ out,
        uint32_t rows, uint32_t cols, uint32_t rowBytes,
        uint32_t blocksPerRow) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint8_t *rg = wg + static_cast<uint64_t>(row) * rowBytes;
    const uint8_t *ru = wu + static_cast<uint64_t>(row) * rowBytes;
    const uint32_t lane = threadIdx.x;
    const uint32_t sub = lane / 16u;
    float accg = 0.0f, accu = 0.0f;
    uint32_t b = 0;
    for (; b + 4u <= blocksPerRow; b += 4u) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
#pragma unroll 32
        for (uint32_t bb = 0; bb < 4; ++bb) {
            const uint8_t *crpG = rg + static_cast<uint64_t>(bb) * kQ2K_BYTES;
            const uint8_t *crpU = ru + static_cast<uint64_t>(bb) * kQ2K_BYTES;
            __half dG = *reinterpret_cast<const __half *>(crpG + 80);
            __half dminG = *reinterpret_cast<const __half *>(crpG + 82);
            __half dU = *reinterpret_cast<const __half *>(crpU + 80);
            __half dminU = *reinterpret_cast<const __half *>(crpU + 82);
            float dfG = __half2float(dG), dmfG = __half2float(dminG);
            float dfU = __half2float(dU), dmfU = __half2float(dminU);
            const uint8_t *scG = crpG;
            const uint8_t *qG = crpG + 16;
            const uint8_t *scU = crpU;
            const uint8_t *qU = crpU + 16;
            float &slotG = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                     : (bb == 2u)       ? a2
                                                        : a3;
            float &slotU = (bb == 0u) ? u0 : (bb == 1u) ? u1
                                     : (bb == 2u)       ? u2
                                                        : u3;
#pragma unroll
            for (uint32_t half = 0; half < 2; ++half) {
                const uint8_t qbG = qG[half * 32u + lane];
                const uint8_t qbU = qU[half * 32u + lane];
#pragma unroll
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    const uint8_t scvG = scG[half * 8u + jj * 2u + sub];
                    const uint8_t scvU = scU[half * 8u + jj * 2u + sub];
                    float dlG = dfG * static_cast<float>(scvG & 0xF);
                    float mlG = dmfG * static_cast<float>(scvG >> 4);
                    float dlU = dfU * static_cast<float>(scvU & 0xF);
                    float mlU = dmfU * static_cast<float>(scvU >> 4);
                    float qvG = static_cast<float>(
                            static_cast<int8_t>((qbG >> (jj * 2u)) & 3));
                    float qvU = static_cast<float>(
                            static_cast<int8_t>((qbU >> (jj * 2u)) & 3));
                    const uint32_t cidx =
                            (b + bb) * 256u + half * 128u + jj * 32u + lane;
                    const float xv = (cidx < cols) ? x[cidx] : 0.0f;
                    slotG = fmaf(fmaf(dlG, qvG, -mlG), xv, slotG);
                    slotU = fmaf(fmaf(dlU, qvU, -mlU), xv, slotU);
                }
            }
        }
        accg += a0;
        accg += a1;
        accg += a2;
        accg += a3;
        accu += u0;
        accu += u1;
        accu += u2;
        accu += u3;
        rg += 4u * kQ2K_BYTES;
        ru += 4u * kQ2K_BYTES;
    }
    for (; b < blocksPerRow; ++b) {
        __half dG = *reinterpret_cast<const __half *>(rg + 80);
        __half dminG = *reinterpret_cast<const __half *>(rg + 82);
        __half dU = *reinterpret_cast<const __half *>(ru + 80);
        __half dminU = *reinterpret_cast<const __half *>(ru + 82);
        float dfG = __half2float(dG), dmfG = __half2float(dminG);
        float dfU = __half2float(dU), dmfU = __half2float(dminU);
        const uint8_t *scG = rg;
        const uint8_t *qG = rg + 16;
        const uint8_t *scU = ru;
        const uint8_t *qU = ru + 16;
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
            const uint8_t qbG = qG[half * 32u + lane];
            const uint8_t qbU = qU[half * 32u + lane];
#pragma unroll
            for (uint32_t jj = 0; jj < 4; ++jj) {
                const uint8_t scvG = scG[half * 8u + jj * 2u + sub];
                const uint8_t scvU = scU[half * 8u + jj * 2u + sub];
                float dlG = dfG * static_cast<float>(scvG & 0xF);
                float mlG = dmfG * static_cast<float>(scvG >> 4);
                float dlU = dfU * static_cast<float>(scvU & 0xF);
                float mlU = dmfU * static_cast<float>(scvU >> 4);
                float qvG = static_cast<float>(
                        static_cast<int8_t>((qbG >> (jj * 2u)) & 3));
                float qvU = static_cast<float>(
                        static_cast<int8_t>((qbU >> (jj * 2u)) & 3));
                const uint32_t cidx =
                        b * 256u + half * 128u + jj * 32u + lane;
                const float xv = (cidx < cols) ? x[cidx] : 0.0f;
                accg = fmaf(fmaf(dlG, qvG, -mlG), xv, accg);
                accu = fmaf(fmaf(dlU, qvU, -mlU), xv, accu);
            }
        }
        rg += kQ2K_BYTES;
        ru += kQ2K_BYTES;
    }
    // Two independent 5-shuffle warp trees (identical order per matrix).
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        accg += __shfl_xor_sync(0xffffffffu, accg, off);
        accu += __shfl_xor_sync(0xffffffffu, accu, off);
    }
    if (lane == 0) {
        const float g = accg;
        out[row] = (g / (1.0f + __expf(-g))) * accu;
    }
}

// ---- Fused Q+K (decode, Q2_K float dequant-dot) ----
// qwen2.5-coder-1.5b q2_k attn_q/attn_k are BOTH Q2_K against the SAME
// activation (s.norm).  The decode path launched them as two separate
// kQGemv<kTypeQ2K> kernels; this fused kernel computes BOTH rows in one
// warp so one launch replaces two.  Q rows are [qRows] and K rows are
// [kRows]; each warp handles Q row `row` and, when `row < kRows`, also
// K row `row`.  The per-matrix math is byte-identical to
// kQGemv<kTypeQ2K> (same fmaf chain, same 4-way batching + in-order
// folds, same 5-shuffle tree), so bit-exactness is preserved.
__global__ void __launch_bounds__(256) kQGemvFusedQK_Q2K(
        const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wk,
        const float *__restrict__ x, float *__restrict__ outQ,
        float *__restrict__ outK, uint32_t qRows, uint32_t kRows,
        uint32_t cols, uint32_t qRowBytes, uint32_t kRowBytes,
        uint32_t qBlocksPerRow, uint32_t kBlocksPerRow) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= qRows) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t sub = lane / 16u;

    // ---- Q row ----
    {
        const uint8_t *rp = wq + static_cast<uint64_t>(row) * qRowBytes;
        float acc = 0.0f;
        uint32_t b = 0;
        for (; b + 4u <= qBlocksPerRow; b += 4u) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 32
            for (uint32_t bb = 0; bb < 4; ++bb) {
                const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                __half d = *reinterpret_cast<const __half *>(crp + 80);
                __half dmin = *reinterpret_cast<const __half *>(crp + 82);
                float df = __half2float(d), dmf = __half2float(dmin);
                const uint8_t *sc = crp;
                const uint8_t *q = crp + 16;
                float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                        : (bb == 2u)       ? a2
                                                           : a3;
#pragma unroll
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                    for (uint32_t jj = 0; jj < 4; ++jj) {
                        const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                        float dl = df * static_cast<float>(scv & 0xF);
                        float ml = dmf * static_cast<float>(scv >> 4);
                        float qv = static_cast<float>(
                                static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                        const uint32_t cidx =
                                (b + bb) * 256u + half * 128u + jj * 32u + lane;
                        slot = fmaf(fmaf(dl, qv, -ml),
                                    (cidx < cols) ? x[cidx] : 0.0f, slot);
                    }
                }
            }
            acc += a0;
            acc += a1;
            acc += a2;
            acc += a3;
            rp += 4u * kQ2K_BYTES;
        }
        for (; b < qBlocksPerRow; ++b) {
            __half d = *reinterpret_cast<const __half *>(rp + 80);
            __half dmin = *reinterpret_cast<const __half *>(rp + 82);
            float df = __half2float(d), dmf = __half2float(dmin);
            const uint8_t *sc = rp;
            const uint8_t *q = rp + 16;
#pragma unroll
            for (uint32_t half = 0; half < 2; ++half) {
                const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                    float dl = df * static_cast<float>(scv & 0xF);
                    float ml = dmf * static_cast<float>(scv >> 4);
                    float qv = static_cast<float>(
                            static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                    const uint32_t cidx =
                            b * 256u + half * 128u + jj * 32u + lane;
                    acc = fmaf(fmaf(dl, qv, -ml),
                               (cidx < cols) ? x[cidx] : 0.0f, acc);
                }
            }
            rp += kQ2K_BYTES;
        }
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) outQ[row] = acc;
    }

    // ---- K row (only warps with row < kRows) ----
    if (row < kRows) {
        const uint8_t *rp = wk + static_cast<uint64_t>(row) * kRowBytes;
        float acc = 0.0f;
        uint32_t b = 0;
        for (; b + 4u <= kBlocksPerRow; b += 4u) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 32
            for (uint32_t bb = 0; bb < 4; ++bb) {
                const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                __half d = *reinterpret_cast<const __half *>(crp + 80);
                __half dmin = *reinterpret_cast<const __half *>(crp + 82);
                float df = __half2float(d), dmf = __half2float(dmin);
                const uint8_t *sc = crp;
                const uint8_t *q = crp + 16;
                float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                        : (bb == 2u)       ? a2
                                                           : a3;
#pragma unroll
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                    for (uint32_t jj = 0; jj < 4; ++jj) {
                        const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                        float dl = df * static_cast<float>(scv & 0xF);
                        float ml = dmf * static_cast<float>(scv >> 4);
                        float qv = static_cast<float>(
                                static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                        const uint32_t cidx =
                                (b + bb) * 256u + half * 128u + jj * 32u + lane;
                        slot = fmaf(fmaf(dl, qv, -ml),
                                    (cidx < cols) ? x[cidx] : 0.0f, slot);
                    }
                }
            }
            acc += a0;
            acc += a1;
            acc += a2;
            acc += a3;
            rp += 4u * kQ2K_BYTES;
        }
        for (; b < kBlocksPerRow; ++b) {
            __half d = *reinterpret_cast<const __half *>(rp + 80);
            __half dmin = *reinterpret_cast<const __half *>(rp + 82);
            float df = __half2float(d), dmf = __half2float(dmin);
            const uint8_t *sc = rp;
            const uint8_t *q = rp + 16;
#pragma unroll
            for (uint32_t half = 0; half < 2; ++half) {
                const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                    float dl = df * static_cast<float>(scv & 0xF);
                    float ml = dmf * static_cast<float>(scv >> 4);
                    float qv = static_cast<float>(
                            static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                    const uint32_t cidx =
                            b * 256u + half * 128u + jj * 32u + lane;
                    acc = fmaf(fmaf(dl, qv, -ml),
                               (cidx < cols) ? x[cidx] : 0.0f, acc);
                }
            }
            rp += kQ2K_BYTES;
        }
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) outK[row] = acc;
    }
}

// ---- Fused Q+K+V (decode; qwen2.5-coder-1.5b q2_k: attn_q/attn_k are
// Q2_K and attn_v is Q4_K against the SAME fp32 activation).  Extends
// kQGemvFusedQK_Q2K with a V section (float dequant-dot, faithful to
// dequantizeQ4_KBlock) so ONE launch replaces fused-QK + the Q8_K
// quantize + the v GEMV, and the three bias adds fold into the
// epilogues (out = acc + bias[row] -- the SAME float add kAddBias
// performed after the store, hence bit-exact for Q/K).  Q/K per-matrix
// math is byte-identical to kQGemvFusedQK_Q2K.  Warps [0, qRows) do Q
// (and K when row < kRows); warps [qRows, qRows+vRows) do V.
// TINYCODER_FUSE_QKV=0 opts out (A/B record).
__global__ void __launch_bounds__(256) kQGemvFusedQKV(
        const uint8_t *__restrict__ wq, const uint8_t *__restrict__ wk,
        const uint8_t *__restrict__ wv, const float *__restrict__ x,
        float *__restrict__ outQ, float *__restrict__ outK,
        float *__restrict__ outV, uint32_t qRows, uint32_t kRows,
        uint32_t vRows, uint32_t cols, uint32_t qRowBytes,
        uint32_t kRowBytes, uint32_t vRowBytes, uint32_t qBlocksPerRow,
        uint32_t kBlocksPerRow, uint32_t vBlocksPerRow,
        const float *__restrict__ biasQ, const float *__restrict__ biasK,
        const float *__restrict__ biasV,
        const uint8_t *__restrict__ xq81) {
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t sub = lane / 16u;
    if (row < qRows) {
        // ---- Q row (byte-identical to kQGemvFusedQK_Q2K) ----
        {
            const uint8_t *rp = wq + static_cast<uint64_t>(row) * qRowBytes;
            if (xq81 != nullptr) {
                // Weight int loads are int-cast derefs: 504-B row
                // stride and 84-B block stride are 4-aligned.
                // Accumulation order differs from the float body (same
                // tolerance class as the GLU/down gemv adoptions).
                float acc = 0.0f;
                const uint32_t kqs = lane % 16u;
                for (uint32_t kbx = lane / 16u; kbx < qBlocksPerRow;
                     kbx += 2u) {
                    const uint8_t *bq =
                            rp + static_cast<uint64_t>(kbx) * 84u;
                    const int bq8_offset = 4 * (kqs / 8);
                    const int scale_offset =
                            kqs - kqs % 8 + (kqs % 8) / 4;
                    const uint8_t *scales_bq = bq + scale_offset;
                    const uint8_t *qs_bq = bq + 16;
                    const int v = *reinterpret_cast<const int *>(
                            qs_bq + 4 * kqs);
                    int u[4];
                    float d8[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const uint8_t *by =
                                xq81 + static_cast<uint64_t>(
                                               kbx * 8u + bq8_offset + i) *
                                               40u;
                        u[i] = *reinterpret_cast<const int *>(
                                by + 4 + 4 * (kqs % 8));
                        d8[i] = __half2float(
                                *reinterpret_cast<const __half *>(by));
                    }
                    float sd = 0.0f, sm = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int sc = scales_bq[2 * i];
                        const int vi = (v >> (2 * i)) & 0x03030303;
                        sd += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF));
                        int m = sc >> 4;
                        m |= m << 8;
                        m |= m << 16;
                        sm += d8[i] * __dp4a(m, u[i], 0);
                    }
                    const float2 dm = __half22float2(
                            *reinterpret_cast<const __half2 *>(bq + 80));
                    acc += dm.x * sd - dm.y * sm;
                }
#pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1) {
                    acc += __shfl_xor_sync(0xffffffffu, acc, off);
                }
                if (lane == 0)
                    outQ[row] = biasQ ? (acc + biasQ[row]) : acc;
            } else {
                float acc = 0.0f;
                uint32_t b = 0;
                for (; b + 4u <= qBlocksPerRow; b += 4u) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 32
                    for (uint32_t bb = 0; bb < 4; ++bb) {
                        const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                        __half d = *reinterpret_cast<const __half *>(crp + 80);
                        __half dmin = *reinterpret_cast<const __half *>(crp + 82);
                        float df = __half2float(d), dmf = __half2float(dmin);
                        const uint8_t *sc = crp;
                        const uint8_t *q = crp + 16;
                        float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                : (bb == 2u)       ? a2
                                                                   : a3;
#pragma unroll
                        for (uint32_t half = 0; half < 2; ++half) {
                            const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                            for (uint32_t jj = 0; jj < 4; ++jj) {
                                const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                                float dl = df * static_cast<float>(scv & 0xF);
                                float ml = dmf * static_cast<float>(scv >> 4);
                                float qv = static_cast<float>(
                                        static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                                const uint32_t cidx =
                                        (b + bb) * 256u + half * 128u + jj * 32u + lane;
                                slot = fmaf(fmaf(dl, qv, -ml),
                                            (cidx < cols) ? x[cidx] : 0.0f, slot);
                            }
                        }
                    }
                    acc += a0;
                    acc += a1;
                    acc += a2;
                    acc += a3;
                    rp += 4u * kQ2K_BYTES;
                }
                for (; b < qBlocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 80);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 82);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *sc = rp;
                    const uint8_t *q = rp + 16;
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                            float dl = df * static_cast<float>(scv & 0xF);
                            float ml = dmf * static_cast<float>(scv >> 4);
                            float qv = static_cast<float>(
                                    static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                            const uint32_t cidx =
                                    b * 256u + half * 128u + jj * 32u + lane;
                            acc = fmaf(fmaf(dl, qv, -ml),
                                       (cidx < cols) ? x[cidx] : 0.0f, acc);
                        }
                    }
                    rp += kQ2K_BYTES;
                }
#pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1) {
                    acc += __shfl_xor_sync(0xffffffffu, acc, off);
                }
                // Folded bias (bit-exact: same float add kAddBias does).
                if (lane == 0)
                    outQ[row] = biasQ ? (acc + biasQ[row]) : acc;
            }
        }
        // ---- K row (warps with row < kRows also run K after Q --
        // a split-K-per-warp variant measured neutral/-0.3 % (the
        // stage is not critical-path bound at 224 blocks), so the
        // simpler serial layout ships) ----
        if (row < kRows) {
            const uint8_t *rp = wk + static_cast<uint64_t>(row) * kRowBytes;
            if (xq81 != nullptr) {
                // dp4a Q2_K x Q8_1, same structure as the Q branch
                // above.  kBlocksPerRow == 1 here: only lanes 0-15
                // carry work, the rest contribute zero to the reduce.
                float acc = 0.0f;
                const uint32_t kqs = lane % 16u;
                for (uint32_t kbx = lane / 16u; kbx < kBlocksPerRow;
                     kbx += 2u) {
                    const uint8_t *bq =
                            rp + static_cast<uint64_t>(kbx) * 84u;
                    const int bq8_offset = 4 * (kqs / 8);
                    const int scale_offset =
                            kqs - kqs % 8 + (kqs % 8) / 4;
                    const uint8_t *scales_bq = bq + scale_offset;
                    const uint8_t *qs_bq = bq + 16;
                    const int v = *reinterpret_cast<const int *>(
                            qs_bq + 4 * kqs);
                    int u[4];
                    float d8[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const uint8_t *by =
                                xq81 + static_cast<uint64_t>(
                                               kbx * 8u + bq8_offset + i) *
                                               40u;
                        u[i] = *reinterpret_cast<const int *>(
                                by + 4 + 4 * (kqs % 8));
                        d8[i] = __half2float(
                                *reinterpret_cast<const __half *>(by));
                    }
                    float sd = 0.0f, sm = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int sc = scales_bq[2 * i];
                        const int vi = (v >> (2 * i)) & 0x03030303;
                        sd += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF));
                        int m = sc >> 4;
                        m |= m << 8;
                        m |= m << 16;
                        sm += d8[i] * __dp4a(m, u[i], 0);
                    }
                    const float2 dm = __half22float2(
                            *reinterpret_cast<const __half2 *>(bq + 80));
                    acc += dm.x * sd - dm.y * sm;
                }
#pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1) {
                    acc += __shfl_xor_sync(0xffffffffu, acc, off);
                }
                if (lane == 0)
                    outK[row] = biasK ? (acc + biasK[row]) : acc;
            } else {
                float acc = 0.0f;
                uint32_t b = 0;
                for (; b + 4u <= kBlocksPerRow; b += 4u) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 32
                    for (uint32_t bb = 0; bb < 4; ++bb) {
                        const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ2K_BYTES;
                        __half d = *reinterpret_cast<const __half *>(crp + 80);
                        __half dmin = *reinterpret_cast<const __half *>(crp + 82);
                        float df = __half2float(d), dmf = __half2float(dmin);
                        const uint8_t *sc = crp;
                        const uint8_t *q = crp + 16;
                        float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                                : (bb == 2u)       ? a2
                                                                   : a3;
#pragma unroll
                        for (uint32_t half = 0; half < 2; ++half) {
                            const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                            for (uint32_t jj = 0; jj < 4; ++jj) {
                                const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                                float dl = df * static_cast<float>(scv & 0xF);
                                float ml = dmf * static_cast<float>(scv >> 4);
                                float qv = static_cast<float>(
                                        static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                                const uint32_t cidx =
                                        (b + bb) * 256u + half * 128u + jj * 32u + lane;
                                slot = fmaf(fmaf(dl, qv, -ml),
                                            (cidx < cols) ? x[cidx] : 0.0f, slot);
                            }
                        }
                    }
                    acc += a0;
                    acc += a1;
                    acc += a2;
                    acc += a3;
                    rp += 4u * kQ2K_BYTES;
                }
                for (; b < kBlocksPerRow; ++b) {
                    __half d = *reinterpret_cast<const __half *>(rp + 80);
                    __half dmin = *reinterpret_cast<const __half *>(rp + 82);
                    float df = __half2float(d), dmf = __half2float(dmin);
                    const uint8_t *sc = rp;
                    const uint8_t *q = rp + 16;
#pragma unroll
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint8_t qb = q[half * 32u + lane];
#pragma unroll
                        for (uint32_t jj = 0; jj < 4; ++jj) {
                            const uint8_t scv = sc[half * 8u + jj * 2u + sub];
                            float dl = df * static_cast<float>(scv & 0xF);
                            float ml = dmf * static_cast<float>(scv >> 4);
                            float qv = static_cast<float>(
                                    static_cast<int8_t>((qb >> (jj * 2u)) & 3));
                            const uint32_t cidx =
                                    b * 256u + half * 128u + jj * 32u + lane;
                            acc = fmaf(fmaf(dl, qv, -ml),
                                       (cidx < cols) ? x[cidx] : 0.0f, acc);
                        }
                    }
                    rp += kQ2K_BYTES;
                }
#pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1) {
                    acc += __shfl_xor_sync(0xffffffffu, acc, off);
                }
                if (lane == 0)
                    outK[row] = biasK ? (acc + biasK[row]) : acc;
            }
        }
    } else {
        // ---- V row: Q4_K float dequant-dot (bit-faithful to
        // dequantizeQ4_KBlock), 4-way block batching with in-order
        // folds (same structure as the Q2_K sections above).
        const uint32_t vrow = row - qRows;
        if (vrow >= vRows) return;
        const uint8_t *rp = wv + static_cast<uint64_t>(vrow) * vRowBytes;
        auto getScaleMin = [](int j, const uint8_t *q, uint8_t *d_out,
                              uint8_t *m_out) {
            if (j < 4) {
                *d_out = q[j] & 63;
                *m_out = q[j + 4] & 63;
            } else {
                *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                *m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
            }
        };
        float acc = 0.0f;
        uint32_t b = 0;
        for (; b + 4u <= vBlocksPerRow; b += 4u) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 32
            for (uint32_t bb = 0; bb < 4; ++bb) {
                const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ4K_BYTES;
                __half d = *reinterpret_cast<const __half *>(crp + 0);
                __half dmin = *reinterpret_cast<const __half *>(crp + 2);
                float df = __half2float(d), dmf = __half2float(dmin);
                const uint8_t *scales = crp + 4;
                const uint8_t *qs = crp + 16;
                float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                        : (bb == 2u)       ? a2
                                                           : a3;
#pragma unroll
                for (uint32_t g = 0; g < 4; ++g) {
                    const uint8_t qb = qs[g * 32u + lane];
                    uint8_t sc0, mm0, sc1, mm1;
                    getScaleMin(static_cast<int>(g * 2 + 0), scales, &sc0,
                                &mm0);
                    getScaleMin(static_cast<int>(g * 2 + 1), scales, &sc1,
                                &mm1);
                    float dl0 = df * static_cast<float>(sc0);
                    float ml0 = dmf * static_cast<float>(mm0);
                    float dl1 = df * static_cast<float>(sc1);
                    float ml1 = dmf * static_cast<float>(mm1);
                    float qv0 = static_cast<float>(qb & 0xF);
                    float qv1 = static_cast<float>(qb >> 4);
                    const uint32_t c0 = (b + bb) * 256u + g * 64u + lane;
                    const uint32_t c1 = c0 + 32u;
                    slot = fmaf(fmaf(dl0, qv0, -ml0),
                                (c0 < cols) ? x[c0] : 0.0f, slot);
                    slot = fmaf(fmaf(dl1, qv1, -ml1),
                                (c1 < cols) ? x[c1] : 0.0f, slot);
                }
            }
            acc += a0;
            acc += a1;
            acc += a2;
            acc += a3;
            rp += 4u * kQ4K_BYTES;
        }
        for (; b < vBlocksPerRow; ++b) {
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            __half dmin = *reinterpret_cast<const __half *>(rp + 2);
            float df = __half2float(d), dmf = __half2float(dmin);
            const uint8_t *scales = rp + 4;
            const uint8_t *qs = rp + 16;
#pragma unroll
            for (uint32_t g = 0; g < 4; ++g) {
                const uint8_t qb = qs[g * 32u + lane];
                uint8_t sc0, mm0, sc1, mm1;
                getScaleMin(static_cast<int>(g * 2 + 0), scales, &sc0, &mm0);
                getScaleMin(static_cast<int>(g * 2 + 1), scales, &sc1, &mm1);
                float dl0 = df * static_cast<float>(sc0);
                float ml0 = dmf * static_cast<float>(mm0);
                float dl1 = df * static_cast<float>(sc1);
                float ml1 = dmf * static_cast<float>(mm1);
                float qv0 = static_cast<float>(qb & 0xF);
                float qv1 = static_cast<float>(qb >> 4);
                const uint32_t c0 = b * 256u + g * 64u + lane;
                const uint32_t c1 = c0 + 32u;
                acc = fmaf(fmaf(dl0, qv0, -ml0),
                           (c0 < cols) ? x[c0] : 0.0f, acc);
                acc = fmaf(fmaf(dl1, qv1, -ml1),
                           (c1 < cols) ? x[c1] : 0.0f, acc);
            }
            rp += kQ4K_BYTES;
        }
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        }
        if (lane == 0)
            outV[vrow] = biasV ? (acc + biasV[vrow]) : acc;
    }
}
