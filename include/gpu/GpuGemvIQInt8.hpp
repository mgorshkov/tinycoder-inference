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

// GpuGemvIQInt8.hpp -- IQ Q8_K integer GEMV and fused gate+up+silu on that path.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// ---- Fused gate+up+silu for the Q8_K integer path (IQ2_S / IQ3_XXS /
// IQ3_S): ONE shared kQuantizeQ8K of the activation, then both row
// dots against the pre-quantized q8k in one warp.
template<int TYPE, bool DP4A = false>
__global__ void __launch_bounds__(256) kQGemvQ8KFusedGU(
        const uint8_t *__restrict__ wg, const uint8_t *__restrict__ wu,
        const uint8_t *__restrict__ x, float *__restrict__ out,
        uint32_t rows, uint32_t blocksPerRow, uint32_t rowBytes,
        bool grsm) {
    extern __shared__ uint8_t sGrid[];
    const uint8_t *gsm = nullptr;
    if (grsm) {
        // Stage the codebook grid for this TYPE (words of 4 B):
        // IQ2_S 1024x u64 = 2048, IQ2_XS 512x u64 = 1024,
        // IQ3_S 512x u32 = 512, IQ3_XXS 256x u32 = 256.
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
    const uint8_t *rg = wg + static_cast<uint64_t>(row) * rowBytes;
    const uint8_t *ru = wu + static_cast<uint64_t>(row) * rowBytes;
    const uint32_t lane = threadIdx.x;
    const uint32_t ib32 = lane >> 2u;// 0..7 within a block
    const uint32_t l = lane & 3u;    // 0..3
    const uint32_t bwBytes = (TYPE == kTypeIQ2S)      ? kIQ2S_BYTES
                             : (TYPE == kTypeIQ2XS)   ? kIQ2XS_BYTES
                             : (TYPE == kTypeIQ3_XXS) ? kIQ3XXS_BYTES
                                                      : kIQ3S_BYTES;
    float sumfG = 0.0f, sumfU = 0.0f;
    uint32_t b = 0;
    // 4-way block batching: 8 independent per-lane integer chains
    // (4 gate + 4 up) feed 8 int trees; lane 0 folds gate then up
    // serially in block order.
    for (; b + 4u <= blocksPerRow; b += 4u) {
        const uint8_t *gw0 = rg + static_cast<uint64_t>(b + 0u) * bwBytes;
        const uint8_t *gw1 = rg + static_cast<uint64_t>(b + 1u) * bwBytes;
        const uint8_t *gw2 = rg + static_cast<uint64_t>(b + 2u) * bwBytes;
        const uint8_t *gw3 = rg + static_cast<uint64_t>(b + 3u) * bwBytes;
        const uint8_t *uw0 = ru + static_cast<uint64_t>(b + 0u) * bwBytes;
        const uint8_t *uw1 = ru + static_cast<uint64_t>(b + 1u) * bwBytes;
        const uint8_t *uw2 = ru + static_cast<uint64_t>(b + 2u) * bwBytes;
        const uint8_t *uw3 = ru + static_cast<uint64_t>(b + 3u) * bwBytes;
        const uint8_t *ba0 = x + static_cast<uint64_t>(b + 0u) * kQ8K_STRIDE;
        const uint8_t *ba1 = x + static_cast<uint64_t>(b + 1u) * kQ8K_STRIDE;
        const uint8_t *ba2 = x + static_cast<uint64_t>(b + 2u) * kQ8K_STRIDE;
        const uint8_t *ba3 = x + static_cast<uint64_t>(b + 3u) * kQ8K_STRIDE;
        const int8_t *y0 = reinterpret_cast<const int8_t *>(ba0 + kQ8K_Q);
        const int8_t *y1 = reinterpret_cast<const int8_t *>(ba1 + kQ8K_Q);
        const int8_t *y2 = reinterpret_cast<const int8_t *>(ba2 + kQ8K_Q);
        const int8_t *y3 = reinterpret_cast<const int8_t *>(ba3 + kQ8K_Q);
        int gci0 = q8kLaneSumi<DP4A>(TYPE, gw0, y0, ib32, l, gsm);
        int gci1 = q8kLaneSumi<DP4A>(TYPE, gw1, y1, ib32, l, gsm);
        int gci2 = q8kLaneSumi<DP4A>(TYPE, gw2, y2, ib32, l, gsm);
        int gci3 = q8kLaneSumi<DP4A>(TYPE, gw3, y3, ib32, l, gsm);
        int uci0 = q8kLaneSumi<DP4A>(TYPE, uw0, y0, ib32, l, gsm);
        int uci1 = q8kLaneSumi<DP4A>(TYPE, uw1, y1, ib32, l, gsm);
        int uci2 = q8kLaneSumi<DP4A>(TYPE, uw2, y2, ib32, l, gsm);
        int uci3 = q8kLaneSumi<DP4A>(TYPE, uw3, y3, ib32, l, gsm);
        gci0 = q8kScaleMuli<TYPE>(gw0, gci0, ib32, l);
        gci1 = q8kScaleMuli<TYPE>(gw1, gci1, ib32, l);
        gci2 = q8kScaleMuli<TYPE>(gw2, gci2, ib32, l);
        gci3 = q8kScaleMuli<TYPE>(gw3, gci3, ib32, l);
        uci0 = q8kScaleMuli<TYPE>(uw0, uci0, ib32, l);
        uci1 = q8kScaleMuli<TYPE>(uw1, uci1, ib32, l);
        uci2 = q8kScaleMuli<TYPE>(uw2, uci2, ib32, l);
        uci3 = q8kScaleMuli<TYPE>(uw3, uci3, ib32, l);
        int gb0 = gci0 + __shfl_xor_sync(0xffffffffu, gci0, 1);
        gb0 += __shfl_xor_sync(0xffffffffu, gb0, 2);
        gb0 += __shfl_xor_sync(0xffffffffu, gb0, 4);
        gb0 += __shfl_xor_sync(0xffffffffu, gb0, 8);
        gb0 += __shfl_xor_sync(0xffffffffu, gb0, 16);
        int gb1 = gci1 + __shfl_xor_sync(0xffffffffu, gci1, 1);
        gb1 += __shfl_xor_sync(0xffffffffu, gb1, 2);
        gb1 += __shfl_xor_sync(0xffffffffu, gb1, 4);
        gb1 += __shfl_xor_sync(0xffffffffu, gb1, 8);
        gb1 += __shfl_xor_sync(0xffffffffu, gb1, 16);
        int gb2 = gci2 + __shfl_xor_sync(0xffffffffu, gci2, 1);
        gb2 += __shfl_xor_sync(0xffffffffu, gb2, 2);
        gb2 += __shfl_xor_sync(0xffffffffu, gb2, 4);
        gb2 += __shfl_xor_sync(0xffffffffu, gb2, 8);
        gb2 += __shfl_xor_sync(0xffffffffu, gb2, 16);
        int gb3 = gci3 + __shfl_xor_sync(0xffffffffu, gci3, 1);
        gb3 += __shfl_xor_sync(0xffffffffu, gb3, 2);
        gb3 += __shfl_xor_sync(0xffffffffu, gb3, 4);
        gb3 += __shfl_xor_sync(0xffffffffu, gb3, 8);
        gb3 += __shfl_xor_sync(0xffffffffu, gb3, 16);
        int ub0 = uci0 + __shfl_xor_sync(0xffffffffu, uci0, 1);
        ub0 += __shfl_xor_sync(0xffffffffu, ub0, 2);
        ub0 += __shfl_xor_sync(0xffffffffu, ub0, 4);
        ub0 += __shfl_xor_sync(0xffffffffu, ub0, 8);
        ub0 += __shfl_xor_sync(0xffffffffu, ub0, 16);
        int ub1 = uci1 + __shfl_xor_sync(0xffffffffu, uci1, 1);
        ub1 += __shfl_xor_sync(0xffffffffu, ub1, 2);
        ub1 += __shfl_xor_sync(0xffffffffu, ub1, 4);
        ub1 += __shfl_xor_sync(0xffffffffu, ub1, 8);
        ub1 += __shfl_xor_sync(0xffffffffu, ub1, 16);
        int ub2 = uci2 + __shfl_xor_sync(0xffffffffu, uci2, 1);
        ub2 += __shfl_xor_sync(0xffffffffu, ub2, 2);
        ub2 += __shfl_xor_sync(0xffffffffu, ub2, 4);
        ub2 += __shfl_xor_sync(0xffffffffu, ub2, 8);
        ub2 += __shfl_xor_sync(0xffffffffu, ub2, 16);
        int ub3 = uci3 + __shfl_xor_sync(0xffffffffu, uci3, 1);
        ub3 += __shfl_xor_sync(0xffffffffu, ub3, 2);
        ub3 += __shfl_xor_sync(0xffffffffu, ub3, 4);
        ub3 += __shfl_xor_sync(0xffffffffu, ub3, 8);
        ub3 += __shfl_xor_sync(0xffffffffu, ub3, 16);
        const float dg0 =
                __half2float(*reinterpret_cast<const __half *>(gw0)) *
                *reinterpret_cast<const float *>(ba0 + kQ8K_D);
        const float dg1 =
                __half2float(*reinterpret_cast<const __half *>(gw1)) *
                *reinterpret_cast<const float *>(ba1 + kQ8K_D);
        const float dg2 =
                __half2float(*reinterpret_cast<const __half *>(gw2)) *
                *reinterpret_cast<const float *>(ba2 + kQ8K_D);
        const float dg3 =
                __half2float(*reinterpret_cast<const __half *>(gw3)) *
                *reinterpret_cast<const float *>(ba3 + kQ8K_D);
        const float du0 =
                __half2float(*reinterpret_cast<const __half *>(uw0)) *
                *reinterpret_cast<const float *>(ba0 + kQ8K_D);
        const float du1 =
                __half2float(*reinterpret_cast<const __half *>(uw1)) *
                *reinterpret_cast<const float *>(ba1 + kQ8K_D);
        const float du2 =
                __half2float(*reinterpret_cast<const __half *>(uw2)) *
                *reinterpret_cast<const float *>(ba2 + kQ8K_D);
        const float du3 =
                __half2float(*reinterpret_cast<const __half *>(uw3)) *
                *reinterpret_cast<const float *>(ba3 + kQ8K_D);
        if (lane == 0) {
            sumfG += dg0 * static_cast<float>(gb0);
            sumfG += dg1 * static_cast<float>(gb1);
            sumfG += dg2 * static_cast<float>(gb2);
            sumfG += dg3 * static_cast<float>(gb3);
            sumfU += du0 * static_cast<float>(ub0);
            sumfU += du1 * static_cast<float>(ub1);
            sumfU += du2 * static_cast<float>(ub2);
            sumfU += du3 * static_cast<float>(ub3);
        }
    }
    // Remainder (blocksPerRow % 4) -- original single-block body.
    for (; b < blocksPerRow; ++b) {
        const uint8_t *blkWG = rg + static_cast<uint64_t>(b) * bwBytes;
        const uint8_t *blkWU = ru + static_cast<uint64_t>(b) * bwBytes;
        const uint8_t *blkA = x + static_cast<uint64_t>(b) * kQ8K_STRIDE;
        const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA + kQ8K_Q);
        const float yd = *reinterpret_cast<const float *>(blkA + kQ8K_D);
        int gci = q8kLaneSumi<DP4A>(TYPE, blkWG, y8, ib32, l, gsm);
        gci = q8kScaleMuli<TYPE>(blkWG, gci, ib32, l);
        int uci = q8kLaneSumi<DP4A>(TYPE, blkWU, y8, ib32, l, gsm);
        uci = q8kScaleMuli<TYPE>(blkWU, uci, ib32, l);
        int gbs = gci + __shfl_xor_sync(0xffffffffu, gci, 1);
        gbs += __shfl_xor_sync(0xffffffffu, gbs, 2);
        gbs += __shfl_xor_sync(0xffffffffu, gbs, 4);
        gbs += __shfl_xor_sync(0xffffffffu, gbs, 8);
        gbs += __shfl_xor_sync(0xffffffffu, gbs, 16);
        int ubs = uci + __shfl_xor_sync(0xffffffffu, uci, 1);
        ubs += __shfl_xor_sync(0xffffffffu, ubs, 2);
        ubs += __shfl_xor_sync(0xffffffffu, ubs, 4);
        ubs += __shfl_xor_sync(0xffffffffu, ubs, 8);
        ubs += __shfl_xor_sync(0xffffffffu, ubs, 16);
        if (lane == 0) {
            const float dg =
                    __half2float(*reinterpret_cast<const __half *>(blkWG)) * yd;
            const float du =
                    __half2float(*reinterpret_cast<const __half *>(blkWU)) * yd;
            sumfG += dg * static_cast<float>(gbs);
            sumfU += du * static_cast<float>(ubs);
        }
    }
    const float scale = (TYPE == kTypeIQ2S)      ? 0.125f
                        : (TYPE == kTypeIQ2XS)   ? 0.125f
                        : (TYPE == kTypeIQ3_XXS) ? 0.25f
                                                 : 1.0f;
    if (lane == 0) {
        const float g = scale * sumfG;
        const float u = scale * sumfU;
        out[row] = (g / (1.0f + __expf(-g))) * u;
    }
}

// ------------------------------------------------------------------
// Device-side FP16 dequant (per-layer streaming for prefill GEMMs).
//
// Instead of persisting a ~2 B/element FP16 twin of EVERY weight in
// VRAM (~14 GB for a 7B), dequantize ONE matrix on-device into the
// reusable wF16_ scratch, run its GEMM(s), then overwrite it with the
// next matrix.  Decode (seqLen==1) never touches this path (the
// quantized kQGemv runs straight from the compact blocks), so the
// scratch only ever holds one matrix at a time.
//
// The kernel mirrors GGMLDequantize::dequantizeBlock for each type.
// One warp per 32-column strip of a row: lane l owns columns
// [stripStart + l*8, stripStart + l*8 + 8) of a 256-element block, so
// an even strip boundary keeps each lane inside one (ib32,...,l) group.
// pad (cols not a multiple of 256) is zero-filled.
// Q8_0 -> fp16 with one 256-thread block per row (the legacy
// kDequantF16<Q8_0> path uses only 4 threads per row: lane*8 < 32).
__global__ void kDequantF16Q8(const uint8_t *__restrict__ w,
                              __half *__restrict__ out, uint32_t rows,
                              uint32_t cols, uint32_t blocksPerRow,
                              uint32_t rowBytes) {
    const uint32_t r = blockIdx.x;
    if (r >= rows || cols == 0) return;
    const uint8_t *rp = w + static_cast<uint64_t>(r) * rowBytes;
    __half *orow = out + static_cast<uint64_t>(r) * cols;
    const uint32_t total = blocksPerRow * 32u;
    for (uint32_t g = threadIdx.x; g < total; g += blockDim.x) {
        const uint32_t b = g >> 5;
        const uint32_t e = g & 31u;
        if (g >= cols) break;
        const uint8_t *bp = rp + static_cast<uint64_t>(b) * 34u;
        const float d = __half2float(*reinterpret_cast<const __half *>(bp));
        const int8_t q = reinterpret_cast<const int8_t *>(bp + 2)[e];
        orow[g] = __float2half(d * static_cast<float>(q));
    }
}

template<int TYPE>
__global__ void kDequantF16(const uint8_t *__restrict__ w,
                            __half *__restrict__ out, uint32_t rows,
                            uint32_t cols, uint32_t blocksPerRow,
                            uint32_t rowBytes) {
    const uint32_t lane = threadIdx.x;
    const uint32_t e0 = lane * 8u;
    const uint32_t r = blockIdx.x;
    if (r >= rows || cols == 0) return;
    const uint8_t *rp = w + static_cast<uint64_t>(r) * rowBytes;
    __half *orow = out + static_cast<uint64_t>(r) * cols;
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
        if (TYPE == kTypeQ5_0 || TYPE == kTypeQ8_0 ||
            TYPE == kTypeIQ4NL) {
            // 32-wide legacy block: only lanes 0..3 own valid columns.
            const uint32_t base = b * 32u;
            if (e0 < 32u) {
                if (TYPE == kTypeQ8_0) {
                    __half d = *reinterpret_cast<const __half *>(rp);
                    const int8_t *qs =
                            reinterpret_cast<const int8_t *>(rp + 2);
                    float dl = __half2float(d);
#pragma unroll
                    for (uint32_t k = 0; k < 8; ++k) {
                        const uint32_t col = base + e0 + k;
                        if (col < cols)
                            orow[col] = __float2half(
                                    dl * static_cast<float>(qs[e0 + k]));
                    }
                } else if (TYPE == kTypeIQ4NL) {
                    // IQ4_NL (4.5 bpw): d(fp16,2) + qs[16].  Per-column
                    // byte j = col&15; nib = (col<16) ? qs[j]&0xF :
                    // qs[j]>>4; weight = d*kvalues_iq4nl[nib].  Matches
                    // CPU dequantizeIQ4_NLBlock.
                    __half d = *reinterpret_cast<const __half *>(rp);
                    float dl = __half2float(d);
                    const uint8_t *qs = rp + 2;
#pragma unroll
                    for (uint32_t k = 0; k < 8; ++k) {
                        const uint32_t col = base + e0 + k;
                        if (col >= cols) continue;
                        const uint32_t j = col & 15u;
                        const uint8_t nib =
                                (qs[j] >> (4u * ((col >> 4) & 1u))) &
                                0xFu;
                        orow[col] = __float2half(
                                dl * static_cast<float>(
                                             c_kvalues_iq4nl[nib]));
                    }
                } else {// Q5_0
                    __half d = *reinterpret_cast<const __half *>(rp);
                    float dl = __half2float(d);
                    uint32_t qh;
                    std::memcpy(&qh, rp + 2, sizeof(uint32_t));
                    const uint8_t *qs = rp + 6;
#pragma unroll
                    for (uint32_t k = 0; k < 8; ++k) {
                        const uint32_t col = base + e0 + k;
                        if (col >= cols) continue;
                        const uint32_t j = col & 15u;
                        const uint8_t low =
                                (qs[j] >> (4u * ((col >> 4) & 1u))) &
                                0xFu;
                        const uint32_t hb =
                                (col & 16u) ? (j + 16u) : j;
                        const uint8_t high =
                                static_cast<uint8_t>((qh >> hb) & 1u);
                        const int val =
                                static_cast<int>(low | (high << 4)) - 16;
                        orow[col] = __float2half(
                                dl * static_cast<float>(val));
                    }
                }
            }
            rp += (TYPE == kTypeQ5_0)
                          ? kQ5_0_BYTES
                  : (TYPE == kTypeQ8_0) ? kQ8_0_BYTES
                                        : kIQ4NL_BYTES;
            continue;
        }
        const uint32_t base = b * 256u;
        // The final partial block's pad columns get zero fp16.
        if (base + e0 >= cols) {
            rp += (TYPE == kTypeQ2K)      ? kQ2K_BYTES
                  : (TYPE == kTypeQ3K)    ? kQ3K_BYTES
                  : (TYPE == kTypeQ4K)    ? kQ4K_BYTES
                  : (TYPE == kTypeQ6K)    ? kQ6K_BYTES
                  : (TYPE == kTypeIQ2XS)  ? kIQ2XS_BYTES
                  : (TYPE == kTypeIQ2XXS) ? kIQ2XXS_BYTES
                  : (TYPE == kTypeIQ4XS)  ? kIQ4XS_BYTES
                  : (TYPE == kTypeIQ3S)   ? kIQ3S_BYTES
                                          : kQ5K_BYTES;
            continue;
        }
        if (TYPE == kTypeQ2K) {
            __half d = *reinterpret_cast<const __half *>(rp + 80);
            __half dmin = *reinterpret_cast<const __half *>(rp + 82);
            float df = __half2float(d), dmf = __half2float(dmin);
            const uint8_t *q = rp + 16;
            const uint8_t *sc = rp;
            const uint32_t half = e0 / 128u;
            const uint32_t jj = (e0 % 128u) / 32u;
            const uint32_t sub = (e0 % 32u) / 16u;
            const uint8_t scv = sc[half * 8u + jj * 2u + sub];
            float dl = df * static_cast<float>(scv & 0xF);
            float ml = dmf * static_cast<float>(scv >> 4);
#pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                // Per-column byte: the 8 lane columns stay within one
                // 16-column sub group (e0%16 is 0 or 8), so only the
                // low 4 bits of the q index advance with k.
                const uint8_t qb = q[half * 32u + sub * 16u + ((e0 + k) & 15u)];
                float qv = static_cast<float>(static_cast<int8_t>(
                        ((qb >> (jj * 2u)) & 3)));
                const uint32_t col = base + e0 + k;
                if (col < cols) orow[col] = __float2half(fmaf(dl, qv, -ml));
            }
        } else if (TYPE == kTypeQ3K) {
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
            const uint32_t half = e0 / 128u;
            const uint32_t jj = (e0 % 128u) / 32u;
            const uint32_t sub = (e0 % 32u) / 16u;
            float dl = df * static_cast<float>(
                                    sc16[half * 8u + jj * 2u + sub] - 32);
            // CPU reference: hm is ONLY 32 bytes (QK_K/8) and does NOT
            // advance per 128-half -- the half distinction is carried by
            // the mask bit 1u<<(half*4+jj) (m starts at 1 and shifts once
            // per j, so half 1 uses bits 4..7).  The byte index within
            // the block is sub*16 + lk (hm[l] for sub 0, hm[l+16] for
            // sub 1), matching kQGemv's hm[lane].  Advancing hm by
            // half*32 overreads into the q array for half 1.
            const uint32_t maskBit = 1u << (half * 4u + jj);
#pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                const uint32_t lk = (e0 + k) & 15u;
                const uint8_t qb = q[half * 32u + sub * 16u + lk];
                const uint8_t hmb = hm[sub * 16u + lk];
                float qv = static_cast<float>(static_cast<int8_t>(
                        ((qb >> (jj * 2u)) & 3) -
                        ((hmb & maskBit) ? 0 : 4)));
                const uint32_t col = base + e0 + k;
                if (col < cols) orow[col] = __float2half(dl * qv);
            }
        } else if (TYPE == kTypeQ4K) {
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            __half dmin = *reinterpret_cast<const __half *>(rp + 2);
            float df = __half2float(d), dmf = __half2float(dmin);
            const uint8_t *scales = rp + 4;
            const uint8_t *qs = rp + 16;
            const uint32_t g = e0 / 64u;
            const uint32_t sub = (e0 / 32u) & 1u;
            const uint32_t jj0 = e0 & 31u;
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
            uint8_t sc, m;
            getScaleMin(static_cast<int>(2u * g + sub), scales, &sc, &m);
            float dl = df * static_cast<float>(sc);
            float ml = dmf * static_cast<float>(m);
            // CPU reference: per-column byte l = jj0+k advances with the
            // column (0..31 within the 32-byte qs group for chunk g);
            // sub==0 reads the low nibble, sub==1 the high nibble.
#pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                const uint8_t qb = qs[g * 32u + jj0 + k];
                const uint8_t low = (sub == 0) ? (qb & 0xF) : (qb >> 4);
                float wv = fmaf(dl, static_cast<float>(low), -ml);
                const uint32_t col = base + e0 + k;
                if (col < cols) orow[col] = __float2half(wv);
            }
        } else if (TYPE == kTypeQ6K) {
            __half d = *reinterpret_cast<const __half *>(rp + 208);
            float df = __half2float(d);
            const uint8_t *ql = rp;
            const uint8_t *qh = rp + 128;
            const int8_t *sc = reinterpret_cast<const int8_t *>(rp + 192);
            const uint32_t half = e0 / 128u;
            const uint32_t sub = (e0 % 128u) / 32u;
            const uint32_t l = e0 % 32u;
            const uint32_t g16 = l / 16u;
            const uint32_t qlBase = half * 64u + ((sub & 1u) ? 32u : 0u);
            const uint32_t qhBase = half * 32u;
            const uint32_t shift = sub * 2u;
            const uint32_t hiNib = (sub & 2u) ? 4u : 0u;
            float dl = df *
                       static_cast<float>(sc[half * 8u + sub * 2u + g16]);
#pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                uint32_t qb = (hiNib) ? (ql[qlBase + l + k] >> 4)
                                      : (ql[qlBase + l + k] & 0xF);
                qb |= ((qh[qhBase + l + k] >> shift) & 3u) << 4u;
                float qv = static_cast<float>(static_cast<int>(qb) - 32);
                const uint32_t col = base + e0 + k;
                if (col < cols) orow[col] = __float2half(dl * qv);
            }
        } else if (TYPE == kTypeIQ2XS) {
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            float df = __half2float(d);
            const uint16_t *qs16 =
                    reinterpret_cast<const uint16_t *>(rp + 2);
            const uint8_t *scales = rp + 66;
            const uint32_t ib32 = e0 >> 5u;
            const uint32_t l = (e0 >> 3u) & 3u;
            float db0 = df * (0.5f + static_cast<float>(scales[ib32] & 0xf)) *
                        0.25f;
            float db1 = df * (0.5f + static_cast<float>(scales[ib32] >> 4)) *
                        0.25f;
            // CPU reference: dl = db[l/2] (l=0,1 -> db0; l=2,3 -> db1).
            float dl = (l >= 2u) ? db1 : db0;
            uint16_t qval = qs16[ib32 * 4u + l];
            uint16_t gridIdx = qval & 0x1FFu;
            uint8_t signIdx = static_cast<uint8_t>(qval >> 9);
            // ggml semantics: grid entry packs 8 byte values; byte j
            // is grid[j]. sign flip from c_ksigns_iq2xs / c_kmask_iq2xs.
            const uint8_t *grid =
                    reinterpret_cast<const uint8_t *>(&c_iq2xs_grid[gridIdx]);
            uint8_t signs = c_ksigns_iq2xs[signIdx];
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                float w = dl * static_cast<float>(grid[j]) *
                          ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t col = base + e0 + j;
                if (col < cols) orow[col] = __float2half(w);
            }
        } else if (TYPE == kTypeIQ2XXS) {
            // IQ2_XXS block (66 B): d(2) + qs[32 x uint16] (64 B).
            // Same per-(ib32,l) math as the kQGemv branch -- grid idx
            // from word0 byte l, sign idx + scale from word1.
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            float df = __half2float(d);
            const uint8_t *qsv = rp + 2;
            const uint32_t ib32 = e0 >> 5u;
            const uint32_t l = (e0 >> 3u) & 3u;
            uint32_t w0, w1;
            std::memcpy(&w0, qsv + 4u * ib32, sizeof(uint32_t));
            std::memcpy(&w1, qsv + 4u * ib32 + 4u, sizeof(uint32_t));
            float db =
                    df * (0.5f + static_cast<float>(w1 >> 28)) * 0.25f;
            const uint32_t gridIdx = (w0 >> (8u * l)) & 0xFFu;
            const uint8_t signs =
                    c_ksigns_iq2xs[(w1 >> (7u * l)) & 127u];
            const uint8_t *grid = reinterpret_cast<const uint8_t *>(
                    &c_iq2xxs_grid[gridIdx]);
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                float wgt = db * static_cast<float>(grid[j]) *
                            ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t col = base + e0 + j;
                if (col < cols) orow[col] = __float2half(wgt);
            }
        } else if (TYPE == kTypeIQ4XS) {
            // IQ4_XS block (136 B): d(2) + scales_h(2) + scales_l(4) +
            // qs[128].  Per 32-weight sub-block ib (lane owns its
            // 8 columns within one sub-block):
            //   ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf)
            //        | (((scales_h >> 2*ib) & 3) << 4);  dl = d*(ls-32)
            //   qs[ib*16 + j] byte -> col ib*32+j (low nibble) or
            //   col ib*32+16+j (high nibble) of kvalues_iq4nl.
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            float df = __half2float(d);
            uint16_t scales_h;
            std::memcpy(&scales_h, rp + 2, sizeof(uint16_t));
            const uint8_t *scales_l = rp + 4;
            const uint8_t *qsv = rp + 8;
            const uint32_t ib = e0 >> 5u;
            const uint32_t l = (e0 >> 3u) & 3u;
            const int ls =
                    static_cast<int>((scales_l[ib / 2u] >>
                                      (4u * (ib % 2u))) &
                                     0xfu) |
                    (static_cast<int>((scales_h >> (2u * ib)) & 0x3u)
                     << 4);
            float dl = df * static_cast<float>(ls - 32);
            const uint8_t *qb = qsv + ib * 16u + l * 4u;
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t jc = l * 8u + j;
                const uint32_t byteIdx = jc & 15u;
                const uint32_t nib = (jc < 16u)
                                             ? (qb[byteIdx] & 0xfu)
                                             : (qb[byteIdx] >> 4);
                float wgt = dl * static_cast<float>(c_kvalues_iq4nl[nib]);
                const uint32_t col = base + e0 + j;
                if (col < cols) orow[col] = __float2half(wgt);
            }
        } else if (TYPE == kTypeIQ2S) {
            // IQ2_S block (82 B): d(2) + qs[64] + qh[8] + scales[8].
            // Per (ib32, l): dl = db[l/2]; gridIdx = qs[ib32*4+l] |
            // (qh[ib32] << (8-2l) & 0x300); sign = qs[32+ib32*4+l].
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            float df = __half2float(d);
            const uint8_t *qsv = rp + 2;
            const uint8_t *qh = rp + 66;
            const uint8_t *scales = rp + 74;
            const uint32_t ib32 = e0 >> 5u;
            const uint32_t l = (e0 >> 3u) & 3u;
            float db0 = df *
                        (0.5f + static_cast<float>(scales[ib32] & 0xf)) *
                        0.25f;
            float db1 = df *
                        (0.5f + static_cast<float>(scales[ib32] >> 4)) *
                        0.25f;
            float dl = (l >= 2u) ? db1 : db0;
            uint16_t gridIdx = static_cast<uint16_t>(
                    qsv[ib32 * 4u + l] |
                    ((qh[ib32] << (8 - 2u * l)) & 0x300));
            const uint8_t *grid =
                    reinterpret_cast<const uint8_t *>(&c_iq2s_grid[gridIdx]);
            const uint8_t signs = qsv[32u + ib32 * 4u + l];
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                float w = dl * static_cast<float>(grid[j]) *
                          ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t col = base + e0 + j;
                if (col < cols) orow[col] = __float2half(w);
            }
        } else if (TYPE == kTypeIQ3_XXS) {
            // IQ3_XXS block (98 B): d(2) + qs[96]; qs[64..95] is the
            // scales+signs area.  Per (ib32, l):
            //   aux32 = u32(qs+64+4*ib32); db = d*(0.5+(aux32>>28))*0.5
            //   signs = c_ksigns_iq2xs[(aux32>>7l)&127]
            //   grid1 = qs[8*ib32+2l+0], grid2 = qs[8*ib32+2l+1]
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            float df = __half2float(d);
            const uint8_t *qsv = rp + 2;
            const uint32_t ib32 = e0 >> 5u;
            const uint32_t l = (e0 >> 3u) & 3u;
            uint32_t aux32;
            std::memcpy(&aux32, qsv + 64u + 4u * ib32,
                        sizeof(uint32_t));
            float db = df *
                       (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
            const uint8_t signs =
                    c_ksigns_iq2xs[(aux32 >> (7u * l)) & 127u];
            const uint32_t g1 = c_iq3xxs_grid[qsv[8u * ib32 + 2u * l]];
            const uint32_t g2 =
                    c_iq3xxs_grid[qsv[8u * ib32 + 2u * l + 1u]];
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                uint32_t gval = (j < 4u) ? ((g1 >> (8u * j)) & 0xFFu)
                                         : ((g2 >> (8u * (j - 4u))) &
                                            0xFFu);
                float w = db * static_cast<float>(gval) *
                          ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t col = base + e0 + j;
                if (col < cols) orow[col] = __float2half(w);
            }
        } else if (TYPE == kTypeIQ3S) {
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            float df = __half2float(d);
            const uint8_t *qs = rp + 2;
            const uint8_t *qh = rp + 66;
            const uint8_t *signs = rp + 74;
            const uint8_t *scales = rp + 106;
            const uint32_t ib32 = e0 >> 5u;
            const uint32_t half = ib32 & 1u;
            const uint32_t pair = ib32 >> 1u;
            const uint32_t l = (e0 >> 3u) & 3u;
            const uint32_t qhIdx = pair * 2u + half;
            const uint32_t qsOff = pair * 16u + half * 8u;
            const uint32_t signOff = pair * 8u + half * 4u;
            float db = df *
                       (1.0f +
                        2.0f * static_cast<float>(
                                       (half == 0) ? (scales[pair] & 0xf)
                                                   : (scales[pair] >> 4)));
            const uint16_t gridIdx1 = static_cast<uint16_t>(
                    qs[qsOff + 2u * l] |
                    ((qh[qhIdx] << (8 - 2u * l)) & 256));
            const uint16_t gridIdx2 = static_cast<uint16_t>(
                    qs[qsOff + 2u * l + 1u] |
                    ((qh[qhIdx] << (7 - 2u * l)) & 256));
            const uint32_t g1 = c_iq3s_grid[gridIdx1];
            const uint32_t g2 = c_iq3s_grid[gridIdx2];
            const uint8_t sbyte = signs[signOff + l];
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                uint32_t gval = (j < 4u) ? ((g1 >> (8u * j)) & 0xFFu)
                                         : ((g2 >> (8u * (j - 4u))) &
                                            0xFFu);
                float w = db * static_cast<float>(gval) *
                          ((sbyte & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t col = base + e0 + j;
                if (col < cols) orow[col] = __float2half(w);
            }
        } else if (TYPE == kTypeQ5K) {
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            __half dmin = *reinterpret_cast<const __half *>(rp + 2);
            float df = __half2float(d), dmf = __half2float(dmin);
            const uint8_t *scales = rp + 4;
            const uint8_t *qh = rp + 16;
            const uint8_t *qs = rp + 48;
            const uint32_t g = e0 / 64u;
            const uint32_t sub = (e0 / 32u) & 1u;
            const uint32_t jj0 = e0 & 31u;
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
            uint8_t sc, m;
            getScaleMin(static_cast<int>(2u * g + sub), scales, &sc, &m);
            float dl = df * static_cast<float>(sc);
            float ml = dmf * static_cast<float>(m);
            const uint8_t u = (sub == 0) ? static_cast<uint8_t>(1u << (2u * g))
                                         : static_cast<uint8_t>(2u << (2u * g));
            const uint8_t *qsp = qs + g * 32u;
#pragma unroll
            for (uint32_t kk = 0; kk < 8; ++kk) {
                uint8_t q5 = (sub == 0)
                                     ? static_cast<uint8_t>(
                                               (qsp[jj0 + kk] & 0xF) |
                                               ((qh[jj0 + kk] & u) ? 16 : 0))
                                     : static_cast<uint8_t>(
                                               (qsp[jj0 + kk] >> 4) |
                                               ((qh[jj0 + kk] & u) ? 16 : 0));
                float wv = dl * static_cast<float>(q5) - ml;
                const uint32_t col = base + e0 + kk;
                if (col < cols) orow[col] = __float2half(wv);
            }
        } else {
            // Unknown type: zero the strip.
#pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                const uint32_t col = base + e0 + k;
                if (col < cols) orow[col] = __float2half(0.0f);
            }
        }
        static constexpr uint32_t SZ = (TYPE == kTypeQ5_0 || TYPE == kTypeQ8_0 ||
                                        TYPE == kTypeIQ4NL)
                                               ? (TYPE == kTypeQ5_0
                                                          ? kQ5_0_BYTES
                                                  : TYPE == kTypeQ8_0
                                                          ? kQ8_0_BYTES
                                                          : kIQ4NL_BYTES)
                                               : (TYPE == kTypeQ2K
                                                          ? kQ2K_BYTES
                                                  : TYPE == kTypeQ3K
                                                          ? kQ3K_BYTES
                                                  : TYPE == kTypeQ4K
                                                          ? kQ4K_BYTES
                                                  : TYPE == kTypeQ6K
                                                          ? kQ6K_BYTES
                                                  : TYPE == kTypeIQ2XS
                                                          ? kIQ2XS_BYTES
                                                  : TYPE == kTypeIQ3_XXS
                                                          ? kIQ3XXS_BYTES
                                                  : TYPE == kTypeIQ3S
                                                          ? kIQ3S_BYTES
                                                  : TYPE == kTypeIQ2S
                                                          ? kIQ2S_BYTES
                                                  : TYPE == kTypeIQ2XXS
                                                          ? kIQ2XXS_BYTES
                                                  : TYPE == kTypeIQ4XS
                                                          ? kIQ4XS_BYTES
                                                          : kQ5K_BYTES);
        rp += SZ;
    }
}

// Forward declaration: defined after launchQGemv (qwen35 section).
__global__ void kQ35F32Gemv(const float *x, const float *w, float *out,
                            uint32_t rows, uint32_t cols);

// Batched prefill GEMM: out[seqLen][rows] = X[seqLen][cols] @ W^T.
// Quantizes every token's activation to Q8_K once (into the caller's
// batched scratch) and runs kQGemmKxQ8K for the K-quant family, reading
// each weight block ONCE per 4-token group (vs once per token in the
// per-token launchQGemv loop).  Bit-exact with the per-token Q8_K
// launchQGemv path, so it is a drop-in for the qwen35 prefill that must
// stay numerically identical to decode.  Returns false (leaving `out`
// untouched) when the type/shape is not covered, so the caller can fall
// back to the exact per-token launchQGemv loop.
bool launchQGemm(int type, const void *wq, const float *x, float *out,
                 uint32_t seqLen, uint32_t rows, uint32_t cols,
                 uint32_t rowBytes, uint32_t blocksPerRow,
                 uint8_t *q8kBatch, uint64_t q8kBatchBytes) {
    if (seqLen == 0 || rows == 0 || cols == 0) return true;
    // ---- the int8 tensor-core INT8 path (int8 tensor cores, default ON) --------
    // A full shared-memory-tiled mul_mat_q: 128-token tiles on Turing
    // (stream-k), Q8_1 INT8-layout activations, mma.s8.m8n8k16 over the
    // RAW GGML weights.  This is the int8 tensor-core prefill design
    // (their INT8 measures ~33 TFLOPS on this card) and it replaces the
    // per-token GEMV loop for Q4_K/Q5_K/Q6_K **and Q8_0** (the dense
    // qwen35 ssm_out is Q8_0 -- without it that projection alone was
    // 512 launches/layer).
    // TINYCODER_INT8_GEMM=0 or TINYCODER_PREFILL_EXACT=1 falls back to the
    // bit-exact integer GEMM.
    if ((type == kTypeQ4K || type == kTypeQ5K || type == kTypeQ6K ||
         type == kTypeQ8_0 || type == kTypeIQ2S ||
         type == kTypeIQ3_XXS) &&
        seqLen >= 8 && (cols & 511u) == 0u &&
        int8Gemm(type, wq, x, out, seqLen, rows, cols, blocksPerRow,
                 reinterpret_cast<void *>(g_stream))) {
        return true;
    }
    // Which kernel template + block bytes for this type.
    const uint32_t kBlkBytes =
            (type == kTypeQ4K)     ? kQ4K_BYTES
            : (type == kTypeQ5K)   ? kQ5K_BYTES
            : (type == kTypeIQ4XS) ? kIQ4XS_BYTES
            : (type == kTypeQ6K)   ? kQ6K_BYTES
                                   : 0u;
    if (kBlkBytes == 0) return false;
    // The MMA-style kxQ8KBlockDot kernels write the Q6_K result on all
    // 8 sub-lanes (shuffle tree); Q4_K/Q5_K/IQ4_XS write lane 0 only.
    // Both are handled inside kQGemmKxQ8K, so no shape constraint here.
    const uint64_t actStride =
            static_cast<uint64_t>(blocksPerRow) * kQ8K_STRIDE;
    const uint64_t need = static_cast<uint64_t>(seqLen) * actStride;
    if (q8kBatch == nullptr || q8kBatchBytes < need) return false;
    // Quantize all tokens' activations (one block-row per token).
    dim3 qgrid(blocksPerRow, seqLen);
    kQuantizeQ8K_Batch<<<qgrid, 256, 0, g_stream>>>(
            x, q8kBatch, cols, seqLen,
            static_cast<uint32_t>(actStride));
    cudaError_t qe = cudaGetLastError();
    if (qe != cudaSuccess) {
        std::fprintf(stderr, "kQuantizeQ8K_Batch: %s\n",
                     cudaGetErrorString(qe));
        return false;
    }
    dim3 block(32, 8);
    // grid.x covers rows in 8-row blocks; grid.y steps 4 tokens/group.
    dim3 grid((rows + 7) / 8, (seqLen + 3) / 4);
    const uint32_t aStride = static_cast<uint32_t>(actStride);
    if (type == kTypeQ4K) {
        kQGemmKxQ8K<kTypeQ4K><<<grid, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), q8kBatch, out, rows,
                blocksPerRow, rowBytes, aStride, seqLen);
    } else if (type == kTypeQ5K) {
        kQGemmKxQ8K<kTypeQ5K><<<grid, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), q8kBatch, out, rows,
                blocksPerRow, rowBytes, aStride, seqLen);
    } else if (type == kTypeQ6K) {
        kQGemmKxQ8K<kTypeQ6K><<<grid, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), q8kBatch, out, rows,
                blocksPerRow, rowBytes, aStride, seqLen);
    } else {
        kQGemmKxQ8K<kTypeIQ4XS><<<grid, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), q8kBatch, out, rows,
                blocksPerRow, rowBytes, aStride, seqLen);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kQGemmKxQ8K: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

// ------------------------------------------------------------------
// Decode GEMV against a CALLER-SUPPLIED Q8_1 row (the 36-byte blocks
// kResidualNormQ81 emits).  These are the same gemv/Gemv.cu kernels,
// with the same launch geometry, that int8Gemv issues -- with its
// internal gemvLaunchQuantizeQ8F32 deleted, because the producer
// already wrote those exact bytes.  That is one graph node fewer per
// GEMV and one fewer pass over the activation.
// A false return means "no pre-Q8_1 route for this type/shape"; the
// caller then calls launchQGemv, which quantizes for itself.
extern "C" void gemvLaunchQ2KF32Plain(const void *, const void *, void *,
                                      int, int, int, int, int, void *);
extern "C" void gemvLaunchQ3KF32Plain(const void *, const void *, void *,
                                      int, int, int, int, int, void *);
extern "C" void gemvLaunchQ4KF32Plain(const void *, const void *, void *,
                                      int, int, int, int, int, void *);
extern "C" void gemvLaunchQ5KF32Plain(const void *, const void *, void *,
                                      int, int, int, int, int, void *);
extern "C" void gemvLaunchQ6KF32Plain(const void *, const void *, void *,
                                      int, int, int, int, int, void *);
extern "C" void gemvLaunchQ80F32Plain(const void *, const void *, void *,
                                      int, int, int, int, int, void *);
extern "C" void gemvLaunchIQ2SF32Plain(const void *, const void *, void *,
                                       int, int, int, int, int, void *);
extern "C" void gemvLaunchIQ2XSF32Plain(const void *, const void *, void *,
                                        int, int, int, int, int, void *);
extern "C" void gemvLaunchIQ3SF32Plain(const void *, const void *, void *,
                                       int, int, int, int, int, void *);
extern "C" void gemvLaunchIQ3XXSF32Plain(const void *, const void *,
                                         void *, int, int, int, int, int,
                                         void *);
// fused gate+up+silu, four rows per block, one warp per row: writes
// silu(gate.x)*up.x into gout and the raw up row into uout.
extern "C" __global__ void gemvIQ3XXSFusedGUF32PlainCuda1(
        const void *, const void *, const void *, float *, float *, int,
        int, int);
extern "C" __global__ void gemvIQ2XSFusedGUF32PlainCuda1(
        const void *, const void *, const void *, float *, float *, int,
        int, int);
extern "C" __global__ void gemvQ4KFusedGUF32PlainCuda1(
        const void *, const void *, const void *, float *, float *, int,
        int, int);

bool launchQGemvQ81(int type, const void *wq, const uint8_t *q81,
                    float *out, uint32_t rows, uint32_t cols) {
    if (q81 == nullptr || rows == 0 || cols == 0) return false;
    if ((cols & 255u) != 0) return false;
    const int c = static_cast<int>(cols);
    const int r = static_cast<int>(rows);
    // stride_col_y/stride_col_dst are only read for batch > 1; the
    // decode path is a single row, so they stay at cols/rows exactly
    // as int8Gemv passes them.
    void *st = reinterpret_cast<void *>(g_stream);
    switch (type) {
        case kTypeQ4K:
            gemvLaunchQ4KF32Plain(wq, q81, out, c, r, c, r, 1, st);
            break;
        case kTypeQ5K:
            gemvLaunchQ5KF32Plain(wq, q81, out, c, r, c, r, 1, st);
            break;
        case kTypeQ6K:
            gemvLaunchQ6KF32Plain(wq, q81, out, c, r, c, r, 1, st);
            break;
        case kTypeQ8_0:
            gemvLaunchQ80F32Plain(wq, q81, out, c, r, c, r, 1, st);
            break;
        case kTypeQ2K:
            gemvLaunchQ2KF32Plain(wq, q81, out, c, r, c, r, 1, st);
            break;
        case kTypeQ3K:
            gemvLaunchQ3KF32Plain(wq, q81, out, c, r, c, r, 1, st);
            break;
        case kTypeIQ2S:
            gemvLaunchIQ2SF32Plain(wq, q81, out, c, r, c, r, 1, st);
            break;
        case kTypeIQ2XS:
            gemvLaunchIQ2XSF32Plain(wq, q81, out, c, r, c, r, 1, st);
            break;
        case kTypeIQ3_XXS:
            gemvLaunchIQ3XXSF32Plain(wq, q81, out, c, r, c, r, 1, st);
            break;
        case kTypeIQ3S:
            gemvLaunchIQ3SF32Plain(wq, q81, out, c, r, c, r, 1, st);
            break;
        default:
            return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

bool launchQGemvFusedGUQ81(int type, const void *wg, const void *wu,
                           const uint8_t *q81, float *out, float *upOut,
                           uint32_t rows, uint32_t cols) {
    if (q81 == nullptr || rows == 0 || cols == 0) return false;
    if ((cols & 255u) != 0) return false;
    const int c = static_cast<int>(cols);
    const int r = static_cast<int>(rows);
    const int sy = c;
    const dim3 g((rows + 3u) / 4u), b(32, 4);
    if (type == kTypeIQ3_XXS) {
        gemvIQ3XXSFusedGUF32PlainCuda1<<<g, b, 0, g_stream>>>(
                wg, wu, q81, out, upOut, c, r, sy);
    } else if (type == kTypeIQ2XS) {
        gemvIQ2XSFusedGUF32PlainCuda1<<<g, b, 0, g_stream>>>(
                wg, wu, q81, out, upOut, c, r, sy);
    } else if (type == kTypeQ4K) {
        gemvQ4KFusedGUF32PlainCuda1<<<g, b, 0, g_stream>>>(
                wg, wu, q81, out, upOut, c, r, sy);
    } else {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

void launchQGemv(int type, const void *wq, const float *x, float *out,
                 uint32_t rows, uint32_t cols, uint32_t rowBytes,
                 uint32_t blocksPerRow, uint8_t *&q8k,
                 uint64_t &q8kBytes) {
    // ---- the int8 tensor-core GEMV decode path (default ON) -------------------
    // One warp per row, Q8_1 activations, the int8 tensor-core vec_dot: this is
    // the shape that reaches ~84 tok/s on the 9B.  Replaces the lean
    // but still scalar-ish quad-row kernels for K-quants/Q8_0 at
    // batch==1.  TINYCODER_INT8_GEMM=0 disables both INT8 and GEMV.
    if ((type == kTypeQ4K || type == kTypeQ5K || type == kTypeQ6K ||
         type == kTypeQ8_0 || type == kTypeQ2K || type == kTypeQ3K ||
         type == kTypeIQ2S || type == kTypeIQ3_XXS ||
         type == kTypeIQ2XS || type == kTypeIQ3S) &&
        int8Gemv(type, wq, x, out, 1, rows, cols,
                 reinterpret_cast<void *>(g_stream))) {
        return;
    }
    // Native EXL3 trellis: `wq` is the packed trellis device pointer
    // registered by the uploader.  Decode + transform happens inside
    // the fused kernel (no Q8_K activation quantization).
    if (type == kTypeEXL3) {
        std::string exl3Err;
        if (!exl3gpu::gemvFused(wq, x, out, g_stream, exl3Err)) {
            std::fprintf(stderr, "EXL3 fused GEMV failed: %s\n",
                         exl3Err.c_str());
            cudaMemsetAsync(out, 0, sizeof(float) * rows, g_stream);
        }
        return;
    }
    // x is read directly from global inside the kernel (coalesced, and
    // tiny enough to stay L1/L2-resident), so no shared memory is used.
    // cols must be a multiple of 32.  For this model hiddenSize=1536,
    // intermediateSize=8960, qLen=1536 and kvLen=256 are all multiples;
    // the Model layer pads any non-multiple up when building the desc.
    // 8 warps per block (256 threads) => 8 rows per block.
    dim3 block(32, 8);
    uint32_t nBlocks = (rows + 7) / 8;
    if (std::getenv("TINYCODER_TRACE_GEMV") != nullptr) {
        std::fprintf(stderr, "[gemv trace] type=%d q8k=%d rows=%u cols=%u "
                             "blocksPerRow=%u rowBytes=%u nBlocks=%u\n",
                     type, (type == kTypeIQ2S || type == kTypeIQ3_XXS || type == kTypeIQ3S),
                     rows, cols, blocksPerRow, rowBytes, nBlocks);
    }
    // Rows+nBlocks combined line (decode reads as (rows,nBlocks)).
    if (std::getenv("TINYCODER_TRACE_GEMV") != nullptr) {
        std::fprintf(stderr, "[gemv rnb] r=%u nB=%u c=%u\n", rows, nBlocks,
                     cols);
    }
    // P1 exp2 (2026-10-06): EXACT-columns fast path.  When
    // blocksPerRow*256 == cols the per-element `(cidx < cols)` guard
    // in the K-quant GEMVs is provably dead (all default-model
    // matrices: H=1536, I=8960, KV=256); the EXACT=true
    // instantiations drop it -- BIT-IDENTICAL arithmetic (max cidx
    // = bpr*256-1 = cols-1) with ~2 fewer SASS ops per weight.
    // MEASURED 2026-10-06 (Release, AVX2, interleaved + ABBA,
    // n-gen 8, reps 8): guarded 165.9/166.0/166.0/165.9/165.7/165.8
    // (mean 165.9) vs EXACT 170.1/170.2/170.1/169.2/170.2/170.3
    // (mean 170.0) -- a solid +2.5% tg8, position-independent.  The
    // gain is real (the guarded kernel's dead compare/select was NOT
    // free once the FFN is issue-bound) and costs nothing, so it is
    // DEFAULT ON.  TINYCODER_EXACT_COLS=0 restores the guarded
    // kernels for A/B.  Non-multiple-of-256 models keep the guard.
    const char *exactE = std::getenv("TINYCODER_EXACT_COLS");
    const bool exactCols =
            (exactE == nullptr || std::atoi(exactE) != 0) &&
            (static_cast<uint64_t>(blocksPerRow) * 256u ==
             static_cast<uint64_t>(cols));
    if (type == kTypeIQ2S || (type == kTypeIQ2XS && iq2xsInt()) ||
        type == kTypeIQ3_XXS || type == kTypeIQ3S ||
        type == kTypeQ5K || type == kTypeQ4K || type == kTypeQ6K ||
        type == kTypeIQ4XS) {
        // The float dequant-dot differs by up to ~5%, flipping near-tie
        // argmax at knife-edge rows -- so mirror the Q8_K integer path
        // here: quantize x once into the reusable q8k_ scratch, then
        // run the integer GEMV.
        const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                              static_cast<uint64_t>(kQ8K_STRIDE);
        if (q8k == nullptr || q8kBytes < need) {
            if (q8k) cudaFree(q8k);
            q8k = nullptr;
            q8kBytes = 0;
            cudaError_t e = cudaMalloc(&q8k, need);
            if (e != cudaSuccess) {
                std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                             cudaGetErrorString(e));
                cudaMemsetAsync(out, 0, sizeof(float) * rows, g_stream);
                return;
            }
            q8kBytes = need;
        }
        kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(x, q8k, cols);
        const bool d4 = iqDp4a();
        const bool grsm = iqGridSmem();
        const bool iq4 = iq4xw() && (rows & 3u) == 0u;
#define TC_IQ_GEMV(T, GB)                                               \
    do {                                                                \
        if (iq4) {                                                      \
            const uint32_t nB4 = (rows / 4u + 7u) / 8u;                 \
            if (d4)                                                     \
                kQGemvQ8K_4xW<T, true>                                  \
                        <<<nB4, block, grsm ? (GB) : 0u, g_stream>>>(   \
                                static_cast<const uint8_t *>(wq), q8k,  \
                                out, rows, blocksPerRow, rowBytes,      \
                                grsm);                                  \
            else                                                        \
                kQGemvQ8K_4xW<T, false>                                 \
                        <<<nB4, block, grsm ? (GB) : 0u, g_stream>>>(   \
                                static_cast<const uint8_t *>(wq), q8k,  \
                                out, rows, blocksPerRow, rowBytes,      \
                                grsm);                                  \
        } else if (d4)                                                  \
            kQGemvQ8K<T, true>                                          \
                    <<<nBlocks, block, grsm ? (GB) : 0u, g_stream>>>(   \
                            static_cast<const uint8_t *>(wq), q8k, out, \
                            rows, blocksPerRow, rowBytes, grsm);        \
        else                                                            \
            kQGemvQ8K<T, false>                                         \
                    <<<nBlocks, block, grsm ? (GB) : 0u, g_stream>>>(   \
                            static_cast<const uint8_t *>(wq), q8k, out, \
                            rows, blocksPerRow, rowBytes, grsm);        \
    } while (0)
        if (type == kTypeIQ2S) {
            TC_IQ_GEMV(kTypeIQ2S, 8192u);
        } else if (type == kTypeIQ2XS) {
            TC_IQ_GEMV(kTypeIQ2XS, 4096u);
        } else if (type == kTypeIQ3_XXS) {
            TC_IQ_GEMV(kTypeIQ3_XXS, 1024u);
        } else if (type == kTypeIQ3S) {
            TC_IQ_GEMV(kTypeIQ3S, 2048u);
        } else if (type == kTypeQ5K) {
            // Q5_K: the CPU reference (matMulVecBatchQ5K_SIMD ->
            // matMulVecBatchQ5K_Q8K_AVX2) quantizes the fp32 activation
            // to Q8_K and dots it against the 5-bit weights with exact
            // integer math; the float dequant-dot of kQGemv<kTypeQ5K>
            // diverged 0.5-5% per row (Q5KKernelParityWithCPU), which
            // compounds through the recurrent GDN state and flips the
            // logits.  Route through kQuantizeQ8K + the bit-exact
            // kQGemvKxQ8K<kTypeQ5K> (kxQ8KBlockDot's Q5K branch mirrors
            // the AVX2 float tree, including the volatile no-FMA
            // mul-then-add discipline).
            //
            // 4-ROWS-PER-WARP LM head (2026-10-12): the 8-active-lane
            // generic kernel ran the Q5_K LM head (all three IQ models,
            // 151936 rows) at 2.4-5.8 ms vs the ~0.36-0.85 ms floor.
            // kQGemvQ5KxQ8K_4xW is a bit-exact drop-in (all 32 lanes
            // issue weight loads, 4 rows/warp) for rows % 4 == 0;
            // TINYCODER_Q5K_LM_4XW=0 restores the generic kernel.
            const char *q5wideE = std::getenv("TINYCODER_Q5K_LM_4XW");
            const bool q5wide4 =
                    (q5wideE == nullptr || std::atoi(q5wideE) != 0);
            if (q5wide4 && (rows & 3u) == 0u) {
                const uint32_t gWarp = (rows / 4u + 7u) / 8u;
                kQGemvQ5KxQ8K_4xW<<<gWarp, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out,
                        rows, blocksPerRow, rowBytes);
            } else {
                kQGemvKxQ8K<kTypeQ5K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out,
                        rows, blocksPerRow, rowBytes);
            }
        } else if (type == kTypeQ4K) {
            // Q4_K: the CPU reference (matMulVecBatchQ4K_SIMD ->
            // matMulVecBatchQ4K_Q8K_AVX2) quantizes the activation to
            // Q8_K and does integer dots; the float dequant-dot
            // kQGemv<kTypeQ4K> diverged (Q4K_Q6KKernelParityWithCPU
            // proved the Q8_K path bit-exact).  Route through
            // kQuantizeQ8K + kQGemvKxQ8K<kTypeQ4K>.
            // 4-ROWS-PER-WARP (2026-10-29, TINYCODER_Q4K_4XW): the
            // kQGemvQ4KxQ8K_4xW drop-in for rows % 4 == 0 (all real
            // uses: dense qwen35 attnO/ffnDown/ffnGate/ffnUp and the
            // Q4_K LM head) is bit-exact and lifts the LARGEST quant
            // class of this model to the Q5_K/Q6_K 4xW bandwidth shape.
            if (q4k4xw() && (rows & 3u) == 0u) {
                const uint32_t w4 =
                        static_cast<uint32_t>(q4kBsy());
                const uint32_t gWarp = (rows / 4u + w4 - 1u) / w4;
                kQGemvQ4KxQ8K_4xW<<<gWarp, dim3(32, w4), 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out,
                        rows, blocksPerRow, rowBytes);
            } else {
                kQGemvKxQ8K<kTypeQ4K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out, rows,
                        blocksPerRow, rowBytes);
            }
        } else if (type == kTypeQ6K) {
            // Q6_K decode path: 4 rows per warp (all 32 lanes issue
            // weight loads; the 8-active-lane generic kernel ran the
            // LM head at ~103 GB/s vs the ~616 GB/s FFN peak because
            // 24 of 32 lanes idled per instruction).  rows%4==0 for
            // every real Q6_K use (LM head 151936, ffn_down 1536,
            // gate/up 8960); fall back to the generic path otherwise.
            if ((rows & 3u) == 0u) {
                const uint32_t gBlocks = rows / 4u;
                const uint32_t gWarp = (gBlocks + 7u) / 8u;
                const uint32_t nBlocks4 = gWarp;
                // WIDE (section 9.23) applied to the Q6_K LM head
                // (2026-10-11): 2x-U16 weight loads + int-deref Y
                // loads instead of the std::memcpy 4-byte-into-4x-U8
                // split.  Gated by TINYCODER_Q2K_WIDE (default on)
                // for A/B.
                if (gemvWide()) {
                    kQGemvQ6KxQ8K_4xW<true>
                            <<<nBlocks4, block, 0, g_stream>>>(
                                    static_cast<const uint8_t *>(wq),
                                    q8k, out, rows, blocksPerRow,
                                    rowBytes);
                } else {
                    kQGemvQ6KxQ8K_4xW<false>
                            <<<nBlocks4, block, 0, g_stream>>>(
                                    static_cast<const uint8_t *>(wq),
                                    q8k, out, rows, blocksPerRow,
                                    rowBytes);
                }
            } else {
                kQGemvKxQ8K<kTypeQ6K><<<nBlocks, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out,
                        rows, blocksPerRow, rowBytes);
            }
        } else if (type == kTypeIQ4XS) {
            kQGemvKxQ8K<kTypeIQ4XS><<<nBlocks, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), q8k, out, rows,
                    blocksPerRow, rowBytes);
        } else {
            kQGemvQ8K<kTypeIQ3S><<<nBlocks, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), q8k, out, rows,
                    blocksPerRow, rowBytes);
        }
        return;
    }
    if (type == kTypeQ2K) {
        // Q2_K x Q8_1 dp4a 4xW (2026-10-08, A/B TINYCODER_Q2K_Q81=1,
        // DEFAULT OFF until measured). Removes the float 4xW's
        // fp32-scalar activation load + fmaf per element (the
        // issue-bound cost).
        const char *q81E = std::getenv("TINYCODER_Q2K_Q81");
        const char *q81mE = std::getenv("TINYCODER_Q2K_Q81_GEMV");
        const bool q81On =
                (q81E != nullptr && std::atoi(q81E) != 0) ||
                (q81mE != nullptr && std::atoi(q81mE) != 0);
        if (q81On && (rows & 3u) == 0u) {
            const uint64_t need =
                    static_cast<uint64_t>(blocksPerRow) * 8u *
                    kQ8_1_STRIDE;
            if (q8k == nullptr || q8kBytes < need) {
                if (q8k) cudaFree(q8k);
                q8k = nullptr;
                q8kBytes = 0;
                cudaError_t e = cudaMalloc(&q8k, need);
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "cudaMalloc(q8_1): %s\n",
                                 cudaGetErrorString(e));
                    cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                    g_stream);
                    return;
                }
                q8kBytes = need;
            }
            const uint32_t nBlk32 = cols / 32u;
            kQuantizeQ8_1<<<nBlk32, 32, 0, g_stream>>>(x, q8k, cols);
            const char *fmm = std::getenv("TINYCODER_Q2K_Q81_GEMV");
            if (fmm != nullptr && std::atoi(fmm) != 0 &&
                blocksPerRow == 6u && rowBytes == 6u * 84u) {
                kQGemvQ2KxQ81_Gemv<<<rows, dim3(32, 2), 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out,
                        rows, blocksPerRow);
                return;
            }
            const char *skE = std::getenv("TINYCODER_Q2K_Q81_SPLITK");
            if (skE != nullptr && std::atoi(skE) != 0) {
                kQGemvQ2KxQ81_SplitK<<<rows, dim3(32, 2), 0,
                                       g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out,
                        rows, blocksPerRow, rowBytes);
            } else {
                const uint32_t nW4 = (rows / 4u + 7u) / 8u;
                kQGemvQ2KxQ81_4xW<<<nW4, block, 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out,
                        rows, blocksPerRow, rowBytes);
            }
            return;
        }
        // DEFAULT OFF until measured).  Row-granular 64-thread blocks
        // (one row per warp, grid = rows/2) cover the ENTIRE row range
        // so the machine stays full -- the 4xW/MQ4XW 280-block
        // launches only fill ~51% of the 68-SM thread slots and stream
        // the FFN at ~78-113 GB/s.  Requires the same Q8_K activation
        // quantization as MQ4XW; math identical (q2kXQ8KBlockDot).
        const char *mmv = std::getenv("TINYCODER_Q2K_GEMV");
        if (mmv != nullptr && std::atoi(mmv) != 0 && rows >= 2u) {
            const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                                  static_cast<uint64_t>(kQ8K_STRIDE);
            if (q8k == nullptr || q8kBytes < need) {
                if (q8k) cudaFree(q8k);
                q8k = nullptr;
                q8kBytes = 0;
                cudaError_t e = cudaMalloc(&q8k, need);
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                                 cudaGetErrorString(e));
                    cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                    g_stream);
                    return;
                }
                q8kBytes = need;
            }
            kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(x, q8k,
                                                             cols);
            const uint32_t nB = (rows + 1u) / 2u;
            kQGemvQ2KxQ8K_Gemv<<<nB, dim3(32, 2), 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), q8k, out, rows,
                    blocksPerRow, rowBytes);
            return;
        }
        // Q2_K x Q8_K int-dot 4xW (2026-09-30, A/B
        // TINYCODER_Q2K_MQ4XW=1, DEFAULT OFF until measured).
        // reference-int8 decode route: quantize the activation to Q8_K
        // once (kQuantizeQ8K) then run the integer 2-bit x Q8_K dot in
        // the Q6_K-proven 4-rows-per-warp shape -- every lane issues
        // u32 loads, 8 lanes per row, 4 rows per warp, no smem.  The
        // float kQGemvQ2KxW4 reads the fp32 activation SCALAR per
        // element, keeping the FFN at ~75 GB/s; this int path is the
        // fix for that (see kernel comment).  Requires rows%4==0
        // (all real Q2_K uses: gate/up 8960, attnO 1536).
        const char *mqx = std::getenv("TINYCODER_Q2K_MQ4XW");
        if (mqx != nullptr && std::atoi(mqx) != 0 &&
            (rows & 3u) == 0u) {
            const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                                  static_cast<uint64_t>(kQ8K_STRIDE);
            if (q8k == nullptr || q8kBytes < need) {
                if (q8k) cudaFree(q8k);
                q8k = nullptr;
                q8kBytes = 0;
                cudaError_t e = cudaMalloc(&q8k, need);
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                                 cudaGetErrorString(e));
                    cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                    g_stream);
                    return;
                }
                q8kBytes = need;
            }
            kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(x, q8k,
                                                             cols);
            // MQ4XW BLOCK-SHAPE A/B (2026-10-10, TINYCODER_Q2K_MQ4XW_BSY):
            // this int body has NO shared memory, so unlike the Q8_1
            // 4xW it can run at the winning BSY=2 block shape.  It was
            // only ever measured at the old 8-warp default.  Default
            // stays 8 (unchanged) until the A/B decides.
            const char *mbsyE = std::getenv("TINYCODER_Q2K_MQ4XW_BSY");
            const uint32_t mwarps = (mbsyE != nullptr)
                                            ? static_cast<uint32_t>(
                                                      std::atoi(mbsyE))
                                            : 8u;
            const uint32_t mWarps = rows / 4u;
            const uint32_t nW4 = (mWarps + mwarps - 1u) / mwarps;
            kQGemvQ2KxQ8K_4xW<<<nW4, dim3(32, mwarps), 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), q8k, out, rows,
                    blocksPerRow, rowBytes);
            return;
        }
        // 4-rows-per-warp Q2_K float kernel (2026-09-30,
        // TINYCODER_Q2K_4XW=0 disables; DEFAULT ON): replicates the
        // kQGemvQ6KxQ8K_4xW bandwidth shape (~616 GB/s) -- all 32
        // lanes issue u32 weight loads, 8 lanes per row, 4 rows per
        // warp, NO shared memory / barrier (both avoided because
        // kQGemvQ2K2xW's smem staging serialized the block).
        // rows%4==0 for all real Q2_K uses (gate/up 8960, attnO
        // 1536); per-element math identical to kQGemv<kTypeQ2K>,
        // reduce order differs (float-path ~1e-7 tolerance).
        // Measured +2.0-2.1% tg8 (165.1 -> 168.4 steady, interleaved
        // A/B, both rounds stable), parity 11/11 PASS.
        // 8-way block batching A/B (2026-09-30, lever-3,
        // TINYCODER_Q2K_8W=1, DEFAULT OFF): same 4xW geometry with
        // 8-block batch chains (see kernel comment).  Overrides the
        // 4xW path when enabled.
        const char *q2k8 = std::getenv("TINYCODER_Q2K_8W");
        const bool use8W = (q2k8 != nullptr && std::atoi(q2k8) != 0);
        const char *q2k4 = std::getenv("TINYCODER_Q2K_4XW");
        // MULTI-ROW-PER-WARP A/B (2026-09-30, TINYCODER_Q2K_NM=RPB,
        // DEFAULT OFF): RPB in {8,16,32}; overrides the 4xW path when
        // set.  See kQGemvQ2KxW4_Nm -- raises per-warp memory-level
        // parallelism on the tiny-grid Q2_K FFN decode GEMV.
        const char *q2knm = std::getenv("TINYCODER_Q2K_NM");
        const int nmv = (q2knm != nullptr) ? std::atoi(q2knm) : 0;
        // 4-WAY K-SPLIT A/B (2026-09-30, TINYCODER_Q2K_SPLIT4=1,
        // DEFAULT OFF): reference-gemv-style round-robin block split
        // across the warp's 4 lane groups.  See kQGemvQ2KxSplit4.
        const char *q2ks4 = std::getenv("TINYCODER_Q2K_SPLIT4");
        if (q2ks4 != nullptr && std::atoi(q2ks4) != 0) {
            kQGemvQ2KxSplit4<<<nBlocks, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), x, out, rows,
                    cols, rowBytes, blocksPerRow);
        } else if (nmv == 8 && (rows % 8u) == 0u) {
            kQGemvQ2KxW4_Nm<8>
                    <<<(rows / 8u + 7u) / 8u, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out,
                            rows, cols, rowBytes, blocksPerRow);
        } else if (nmv == 16 && (rows % 16u) == 0u) {
            kQGemvQ2KxW4_Nm<16>
                    <<<(rows / 16u + 7u) / 8u, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out,
                            rows, cols, rowBytes, blocksPerRow);
        } else if (nmv == 32 && (rows % 32u) == 0u) {
            kQGemvQ2KxW4_Nm<32>
                    <<<(rows / 32u + 7u) / 8u, block, 0, g_stream>>>(
                            static_cast<const uint8_t *>(wq), x, out,
                            rows, cols, rowBytes, blocksPerRow);
        } else if (use8W && (rows & 3u) == 0u) {
            const uint32_t nW4 = (rows / 4u + 7u) / 8u;
            kQGemvQ2KxW4b8<<<nW4, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), x, out, rows,
                    cols, rowBytes, blocksPerRow);
        } else if ((q2k4 == nullptr || std::atoi(q2k4) != 0) &&
                   (rows & 3u) == 0u) {
            // BLOCK-SHAPE A/B (2026-09-30, TINYCODER_Q2K_BSY=W): warps
            // per block for the 4xW kernel. 2-warp (64-thread) blocks
            // push the gate/up grid from 280 to 1120 blocks; the kernel
            // body and per-value math are unchanged (parity-identical).
            const char *bsyE = std::getenv("TINYCODER_Q2K_BSY");
            const uint32_t warps = (bsyE != nullptr)
                                           ? static_cast<uint32_t>(
                                                     std::atoi(bsyE))
                                           : 2u;
            const uint32_t nWarps = rows / 4u;
            const uint32_t nW4 =
                    (nWarps + warps - 1u) / warps;
            // REPACKED 16-byte-lane layout (2026-10-10, §9.19,
            // TINYCODER_Q2K_RPACK=1): repack once per matrix (cached by
            // weight pointer; the first eager decode repacks BEFORE any
            // graph capture, so the captured graph replays the RP
            // kernel), then run kQGemvQ2KxW4_RP -- 3 weight loads per
            // block instead of 12, each covering 64 contiguous bytes.
            // Byte-identical to MODE 1.
            const char *rpE = std::getenv("TINYCODER_Q2K_RPACK");
            if (rpE != nullptr && std::atoi(rpE) != 0 &&
                (cols & 255u) == 0u && (rows & 3u) == 0u) {
                static std::mutex rpMtx;
                static std::vector<std::pair<const void *, uint8_t *>>
                        rpCache;
                uint8_t *rpBuf = nullptr;
                {
                    std::lock_guard<std::mutex> lk(rpMtx);
                    for (auto &kv: rpCache)
                        if (kv.first == wq) {
                            rpBuf = kv.second;
                            break;
                        }
                    // cudaMalloc is illegal during stream capture, and
                    // launchQGemv is a free function (no member flag):
                    // query the stream directly.  The first eager
                    // decode repacks, so capture always hits the cache.
                    cudaStreamCaptureStatus capSt =
                            cudaStreamCaptureStatusNone;
                    cudaStreamIsCapturing(g_stream, &capSt);
                    const bool capturing =
                            (capSt != cudaStreamCaptureStatusNone);
                    if (rpBuf == nullptr && !capturing) {
                        void *p = nullptr;
                        const uint64_t rpBytes =
                                static_cast<uint64_t>(rows) *
                                blocksPerRow * 136u;
                        if (cudaMalloc(&p, rpBytes) == cudaSuccess) {
                            kRepackQ2K<<<dim3(rows, blocksPerRow), 32, 0,
                                         g_stream>>>(
                                    static_cast<const uint8_t *>(wq),
                                    static_cast<uint8_t *>(p), rowBytes,
                                    blocksPerRow);
                            rpBuf = static_cast<uint8_t *>(p);
                            rpCache.push_back({wq, rpBuf});
                        }
                    }
                }
                if (rpBuf != nullptr) {
                    if (exactCols)
                        kQGemvQ2KxW4_RP<true><<<nW4, dim3(32, warps), 0,
                                                g_stream>>>(
                                rpBuf, x, out, rows, cols, blocksPerRow);
                    else
                        kQGemvQ2KxW4_RP<false><<<nW4, dim3(32, warps), 0,
                                                 g_stream>>>(
                                rpBuf, x, out, rows, cols, blocksPerRow);
                    return;
                }
            }
            // MODE A/B (2026-10-10, TINYCODER_Q2K_MODE=0/1/2): 0 =
            // guarded scalar, 1 = EXACT (guard dropped when cols are
            // exact), 2 = EXACT + 128-bit activation load (LDG.128 per
            // 4 contiguous cols; attacks the issue-bound SM 97%/DRAM
            // 20% profile, plan ch.4).  Modes 1/2 are bit-identical to
            // each other; both require exactCols to be safe.
            // Default 1 (EXACT scalar): MODE 2 (LDG.128) measured
            // within noise / -0.2% over MODE 1 in 4 clean interleaved
            // rounds -> REJECTED (activation loads are not the Q2_K
            // gate/up limiter).  Kept env-selectable as an A/B record.
            const char *modeE = std::getenv("TINYCODER_Q2K_MODE");
            const int q2kMode = (modeE != nullptr) ? std::atoi(modeE) : 1;
            if (exactCols && q2kMode == 3) {
                kQGemvQ2KxW4<3><<<nW4, dim3(32, warps), 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows,
                        cols, rowBytes, blocksPerRow);
            } else if (exactCols && q2kMode == 6) {
                // MODE 6 (2026-10-10): MODE 1 with __ldcg (L2-only)
                // weight loads -- L1 pipe relief per the fresh ncu
                // (L1 73-75 % / DRAM 19 %).
                kQGemvQ2KxW4<6><<<nW4, dim3(32, warps), 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows,
                        cols, rowBytes, blocksPerRow);
            } else if (exactCols && q2kMode == 5) {
                // MODE 5 (2026-10-10): packed weight loads + LDG.128 x.
                kQGemvQ2KxW4<5><<<nW4, dim3(32, warps), 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows,
                        cols, rowBytes, blocksPerRow);
            } else if (exactCols && q2kMode == 4) {
                // MODE 4 (2026-10-10): packed weight loads only (u32
                // d|dm + u32 scale words instead of half/byte loads).
                kQGemvQ2KxW4<4><<<nW4, dim3(32, warps), 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows,
                        cols, rowBytes, blocksPerRow);
            } else if (exactCols && q2kMode >= 2) {
                kQGemvQ2KxW4<2><<<nW4, dim3(32, warps), 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows,
                        cols, rowBytes, blocksPerRow);
            } else if (exactCols) {
                kQGemvQ2KxW4<1><<<nW4, dim3(32, warps), 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows,
                        cols, rowBytes, blocksPerRow);
            } else {
                kQGemvQ2KxW4<0><<<nW4, dim3(32, warps), 0,
                                  g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows,
                        cols, rowBytes, blocksPerRow);
            }
        } else {
            kQGemv<kTypeQ2K><<<nBlocks, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), x, out, rows,
                    cols, rowBytes, blocksPerRow);
        }
    } else if (type == kTypeQ3K) {
        // Quantizes the fp32 activation to Q8_1 once (per-32 d+s), in
        // the (b*8+n) numbering the n-q3 row index (== cols/32/8)
        // requires. Targets the FFN down (the 130 GB/s stage): measured
        // down 1.40 -> 1.12 ms/token (~130 -> 163 GB/s), whole-decode
        // +5.0-5.4% (eager + graph), greedy parity byte-identical on
        // 4/4 prompts (plan 9.9).
        const char *fmm3 = std::getenv("TINYCODER_Q3K_GEMV");
        if ((fmm3 == nullptr || std::atoi(fmm3) != 0) &&
            (cols & 31u) == 0u) {
            const uint64_t need =
                    static_cast<uint64_t>(blocksPerRow) * 8u *
                    kQ8_1_STRIDE;
            if (q8k == nullptr || q8kBytes < need) {
                if (q8k) cudaFree(q8k);
                q8k = nullptr;
                q8kBytes = 0;
                cudaError_t e = cudaMalloc(&q8k, need);
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "cudaMalloc(q8_1): %s\n",
                                 cudaGetErrorString(e));
                    cudaMemsetAsync(out, 0, sizeof(float) * rows,
                                    g_stream);
                    return;
                }
                q8kBytes = need;
            }
            const uint32_t nBlk32 = cols / 32u;
            kQuantizeQ8_1<<<nBlk32, 32, 0, g_stream>>>(x, q8k, cols);
            // 4 warps only pays when every warp group has K work:
            // each group handles one 16-lane slot per iteration, so
            // with fewer than 8 blocks per row the extra warps idle.
            if (blocksPerRow >= 8u) {
                if (gemvWide())
                    kQGemvQ3KxQ81_Gemv<true, 4><<<rows, dim3(32, 4), 0,
                                                  g_stream>>>(
                            static_cast<const uint8_t *>(wq), q8k, out,
                            rows, blocksPerRow, rowBytes, nullptr);
                else
                    kQGemvQ3KxQ81_Gemv<false, 4><<<rows, dim3(32, 4), 0,
                                                   g_stream>>>(
                            static_cast<const uint8_t *>(wq), q8k, out,
                            rows, blocksPerRow, rowBytes, nullptr);
            } else if (gemvWide())
                kQGemvQ3KxQ81_Gemv<true, 2><<<rows, dim3(32, 2), 0,
                                              g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out,
                        rows, blocksPerRow, rowBytes, nullptr);
            else
                kQGemvQ3KxQ81_Gemv<false, 2><<<rows, dim3(32, 2), 0,
                                               g_stream>>>(
                        static_cast<const uint8_t *>(wq), q8k, out,
                        rows, blocksPerRow, rowBytes, nullptr);
            return;
        }
        // SPLIT-K occupancy A/B (2026-09-29, TINYCODER_SPLITK_DOWN=n):
        // the small-row Q3_K ffn down (rows=1536) runs at ~2.8
        // blocks/SM resident; split each row's block range across n
        // chunk-warps in ONE block to raise the concurrent-stream
        // count.  Per-chunk math is byte-identical; the cross-chunk
        // join is a 1-ulp-class reorder (see kQGemvSplitK).
        const char *sk = std::getenv("TINYCODER_SPLITK_DOWN");
        if (sk != nullptr) {
            const int skn = std::atoi(sk);
            if (skn == 2) {
                kQGemvSplitK<kTypeQ3K, 2><<<rows, dim3(32, 2), 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows,
                        cols, rowBytes, blocksPerRow);
                return;
            }
            if (skn == 4) {
                kQGemvSplitK<kTypeQ3K, 4><<<rows, dim3(32, 4), 0, g_stream>>>(
                        static_cast<const uint8_t *>(wq), x, out, rows,
                        cols, rowBytes, blocksPerRow);
                return;
            }
        }
        // 4-rows-per-warp Q3_K kernel (2026-10-06, P1 bandwidth
        // campaign, TINYCODER_Q3K_4XW=1 opts IN; DEFAULT OFF).
        // Mirrors the byte-shape-proven kQGemvQ2KxW4 geometry (all 32
        // lanes issue u32 weight loads, 8 lanes/row, 4 row-streams per
        // warp, NO smem/barrier), but the Q3_K hmask fold adds ~1 ALU
        // op per weight byte that the 1-row-per-warp kernel does NOT
        // pay.  MEASURED 2026-10-06 (Release, AVX2, interleaved,
        // n-gen 8, reps 8): baseline ~163 tok/s vs 4xW ~150 tok/s
        // (~-8%); block-size sweep BSY=1/2/4 also ~154-156, none
        // recovers it.  The ffn down (1536 rows) is ISSUE/LATENCY-bound
        // (SM ~96% busy, DRAM ~20%), so the extra per-byte ALU is not
        // hidden -- this REFUTES the "Q3_K down is bandwidth-limited by
        // its 1-row geometry" hypothesis and reconfirms the README
        // campaign verdict.  Kept opt-in as an A/B record; parity
        // verified (44 PASS / 0 FAIL with the kernel ON).
        const char *q3k4 = std::getenv("TINYCODER_Q3K_4XW");
        const bool use4W = (q3k4 != nullptr && std::atoi(q3k4) != 0);
        if (use4W && (rows & 3u) == 0u) {
            const char *bsyE3 = std::getenv("TINYCODER_Q3K_BSY");
            const uint32_t warps3 = (bsyE3 != nullptr)
                                            ? static_cast<uint32_t>(
                                                      std::atoi(bsyE3))
                                            : 8u;
            const uint32_t nW4 = (rows / 4u + warps3 - 1u) / warps3;
            kQGemvQ3KxW4<<<nW4, dim3(32, warps3), 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), x, out, rows,
                    cols, rowBytes, blocksPerRow);
            return;
        }
        if (exactCols) {
            kQGemv<kTypeQ3K, true><<<nBlocks, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), x, out, rows,
                    cols, rowBytes, blocksPerRow);
        } else {
            kQGemv<kTypeQ3K, false><<<nBlocks, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), x, out, rows,
                    cols, rowBytes, blocksPerRow);
        }
    } else if (type == kTypeIQ2XS) {
        kQGemv<kTypeIQ2XS><<<nBlocks, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), x, out, rows, cols,
                rowBytes, blocksPerRow);
    } else if (type == kTypeIQ2XXS) {
        kQGemv<kTypeIQ2XXS><<<nBlocks, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), x, out, rows, cols,
                rowBytes, blocksPerRow);
    } else if (type == kTypeIQ3S) {
        kQGemv<kTypeIQ3S><<<nBlocks, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), x, out, rows, cols,
                rowBytes, blocksPerRow);
    } else if (type == kTypeIQ2S) {
        kQGemv<kTypeIQ2S><<<nBlocks, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), x, out, rows, cols,
                rowBytes, blocksPerRow);
    } else if (type == kTypeIQ3_XXS) {
        kQGemv<kTypeIQ3_XXS><<<nBlocks, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), x, out, rows, cols,
                rowBytes, blocksPerRow);
    } else if (type == kTypeQ5_0) {
        kQGemvSmall32<kTypeQ5_0><<<nBlocks, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), x, out, rows, cols,
                rowBytes, blocksPerRow);
    } else if (type == kTypeIQ4NL) {
        // IQ4_NL (4.5 bpw) GEMV: 32-wide legacy blocks, weight =
        // d * kvalues_iq4nl[nibble] (float dequant-dot, like Q5_0).
        kQGemvSmall32<kTypeIQ4NL><<<nBlocks, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), x, out, rows, cols,
                rowBytes, blocksPerRow);
    } else if (type == kTypePTQ1_0) {
        launchQGemvPTQ1_0(static_cast<const uint8_t *>(wq), x, out, rows,
                          cols, rowBytes, blocksPerRow);
    } else if (type == kTypePQ2_0) {
        // Prism PQ2_0 (Bonsai) ternary GEMV: 128-wide blocks (34 B), four
        // output rows per 128-thread block.  The `x` here is already the
        // folded activation H S P x (the caller applies kPrismFold first).
        // The float4 activation load needs cols % 4 == 0 and a 16-byte
        // aligned x; every Prism width is a multiple of 128 and every
        // activation buffer comes from cudaMalloc, so this is the normal
        // path (the kernel still guards each access).
        launchQGemvPQ2_0(static_cast<const uint8_t *>(wq), x, out, rows,
                         cols, rowBytes, blocksPerRow);
    } else if (type == kTypeQ8_0) {
        // Q8_0: integer Q8_32 activation GEMV, bit-exact with the CPU
        // matMulVecBatchQ8_0_Q8K_AVX2 (quantize x per 32-block, exact
        // int8 madd + AVX float tree).  The fp32 dequant-dot
        // (kQGemvSmall32) rounds differently and diverges the logits
        // through a 40-layer decode, flipping near-tie argmax rows
        // (batch-vs-sequential and GPU-vs-CPU parity failures on
        // qwen35moe).  Every Q8_0 path — trunk GEMVs, routed experts,
        // shared expert, LM head — must run this identical math so the
        // GPU matches the CPU reference bit-for-bit.
        // The Q8_32 activation scratch REUSES the caller-supplied
        // q8k/q8kBytes (sized blocksPerRow * kQ8K_STRIDE = 304 B/block,
        // far larger than the 34 B/block this kernel needs), so no
        // separate allocation is required.  kQuantizeQ8_0x32 writes the
        // 32-wide Q8_0 activation blocks into the SAME buffer the
        // 256-wide kQuantizeQ8K would use — launchQGemv never has both
        // in flight on one call.
        // kQ8_0x32_STRIDE (64) < kQ8K_STRIDE (304): the reused buffer's
        // cap always covers the denser Q8_32 layout; the writer never
        // exceeds blocksPerRow*64 bytes.
        const uint64_t need = static_cast<uint64_t>(blocksPerRow) *
                              static_cast<uint64_t>(kQ8_0x32_STRIDE);
        const uint64_t have = static_cast<uint64_t>(blocksPerRow) *
                              static_cast<uint64_t>(kQ8K_STRIDE);
        if (q8k == nullptr || q8kBytes < have) {
            if (q8k) cudaFree(q8k);
            q8k = nullptr;
            q8kBytes = 0;
            cudaError_t e = cudaMalloc(&q8k, have);
            if (e != cudaSuccess) {
                std::fprintf(stderr, "cudaMalloc(q8k): %s\n",
                             cudaGetErrorString(e));
                cudaMemsetAsync(out, 0, sizeof(float) * rows, g_stream);
                return;
            }
            q8kBytes = have;
        }
        (void) need;
        kQuantizeQ8_0x32<<<blocksPerRow, 32, 0, g_stream>>>(x, q8k, cols);
        kQGemvQ8_0xQ8K<<<nBlocks, block, 0, g_stream>>>(
                static_cast<const uint8_t *>(wq), q8k, out, rows,
                blocksPerRow, rowBytes);
    } else if (type == kTypeF32) {
        // Plain F32 GEMV: weight blob is rows*cols floats row-major
        // (the qwen35 ssm_alpha / ssm_beta small matrices).  The device
        // buffer was uploaded with exactly this layout.
        kQ35F32Gemv<<<rows, 256, 0, g_stream>>>(
                x, static_cast<const float *>(wq), out, rows, cols);
    } else {
        cudaMemsetAsync(out, 0, sizeof(float) * rows, g_stream);
    }
}

// Placeholder q8_1-buffer references for launchQGemvResid's default
// arguments (only the ffnDown caller passes the real member buffers).
uint8_t *g_q8k_unused = nullptr;
uint64_t g_q8k_unused_bytes = 0;

// ---- launchQGemvResid (2026-10-09): the Q2_K (attnO) and Q3_K
// (ffnDown) decode GEMVs fused with the trailing residual
// (acc[row] += out[row]).  Deletes the kAddResidual launch after these
// two projections; the dot math and reduce tree are IDENTICAL to the
// base kernels, so out[] is written byte-for-byte as before and the
// residual stream stays byte-identical.  The activation is always fp32
// (these projections do not use the Q8_K int path by default), and the
// EXACT-columns flag mirrors launchQGemv.  Returns true when it
// launched; false for an unsupported type/shape, so the caller falls
// back to launchQGemv + kAddResidual.
bool launchQGemvResid(int type, const void *wq, const float *x,
                      float *out, float *acc, uint32_t rows,
                      uint32_t cols, uint32_t rowBytes,
                      uint32_t blocksPerRow, bool allowGemv = false,
                      uint8_t *&q8k = g_q8k_unused,
                      uint64_t &q8kBytes = g_q8k_unused_bytes,
                      const uint8_t *preQ81 = nullptr) {
    const char *exactE = std::getenv("TINYCODER_EXACT_COLS");
    const bool exactCols =
            (exactE == nullptr || std::atoi(exactE) != 0) &&
            (static_cast<uint64_t>(blocksPerRow) * 256u ==
             static_cast<uint64_t>(cols));
    if (type == kTypeQ2K) {
        // Match the DEFAULT 4xW path (BSY=2 warps/block, empty-env
        // enabled).  If the caller has switched off 4xW (or enabled a
        // different Q2_K geometry A/B), decline so the base
        // launchQGemv + kAddResidual runs instead -- keeps the fused
        // path exactly on top of whatever the default geometry is.
        const char *q2k4 = std::getenv("TINYCODER_Q2K_4XW");
        const bool use4W = (q2k4 == nullptr || std::atoi(q2k4) != 0);
        if (!use4W || (rows & 3u) != 0u) return false;
        const char *bsyE = std::getenv("TINYCODER_Q2K_BSY");
        const uint32_t warps =
                (bsyE != nullptr)
                        ? static_cast<uint32_t>(std::atoi(bsyE))
                        : 2u;
        const uint32_t nW4 = (rows / 4u + warps - 1u) / warps;
        if (exactCols) {
            kQGemvQ2KxW4Resid<true><<<nW4, dim3(32, warps), 0,
                                      g_stream>>>(
                    static_cast<const uint8_t *>(wq), x, out, acc, rows,
                    cols, rowBytes, blocksPerRow);
        } else {
            kQGemvQ2KxW4Resid<false><<<nW4, dim3(32, warps), 0,
                                       g_stream>>>(
                    static_cast<const uint8_t *>(wq), x, out, acc, rows,
                    cols, rowBytes, blocksPerRow);
        }
        return true;
    }
    if (type == kTypeQ3K && allowGemv) {
        // Used for ffnDown only (allowGemv); attnO keeps the float
        // fused-resid path.
        const char *fmm3 = std::getenv("TINYCODER_Q3K_GEMV");
        const bool gemv =
                (fmm3 == nullptr || std::atoi(fmm3) != 0) &&
                (cols & 31u) == 0u;
        if (gemv) {
            // preQ81 (2026-10-10, plan section 9.26 lever 1): the
            // caller's producer (combine W4 for attnO) already wrote
            // the Q8_1 activation blocks -- skip the allocation guard
            // and the separate kQuantizeQ8_1 launch entirely.
            if (preQ81 == nullptr) {
                const uint64_t need =
                        static_cast<uint64_t>(blocksPerRow) * 8u *
                        kQ8_1_STRIDE;
                if (q8k == nullptr || q8kBytes < need) {
                    if (q8k) cudaFree(q8k);
                    q8k = nullptr;
                    q8kBytes = 0;
                    cudaError_t e = cudaMalloc(&q8k, need);
                    if (e != cudaSuccess) {
                        std::fprintf(stderr, "cudaMalloc(q8_1): %s\n",
                                     cudaGetErrorString(e));
                        return false;
                    }
                    q8kBytes = need;
                }
                const uint32_t nBlk32 = cols / 32u;
                kQuantizeQ8_1<<<nBlk32, 32, 0, g_stream>>>(x, q8k, cols);
            }
            const uint8_t *y1 = (preQ81 != nullptr) ? preQ81 : q8k;
            if (blocksPerRow >= 8u) {
                if (gemvWide())
                    kQGemvQ3KxQ81_Gemv<true, 4><<<rows, dim3(32, 4), 0,
                                                  g_stream>>>(
                            static_cast<const uint8_t *>(wq), y1, out,
                            rows, blocksPerRow, rowBytes, acc);
                else
                    kQGemvQ3KxQ81_Gemv<false, 4><<<rows, dim3(32, 4), 0,
                                                   g_stream>>>(
                            static_cast<const uint8_t *>(wq), y1, out,
                            rows, blocksPerRow, rowBytes, acc);
            } else if (gemvWide())
                kQGemvQ3KxQ81_Gemv<true, 2><<<rows, dim3(32, 2), 0,
                                              g_stream>>>(
                        static_cast<const uint8_t *>(wq), y1, out, rows,
                        blocksPerRow, rowBytes, acc);
            else
                kQGemvQ3KxQ81_Gemv<false, 2><<<rows, dim3(32, 2), 0,
                                               g_stream>>>(
                        static_cast<const uint8_t *>(wq), y1, out, rows,
                        blocksPerRow, rowBytes, acc);
            return true;
        }
    }
    if (type == kTypeQ3K) {
        const uint32_t nBlocks = (rows + 7u) / 8u;
        dim3 block(32, 8);
        if (exactCols) {
            kQGemvQ3KResid<true><<<nBlocks, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), x, out, acc, rows,
                    cols, rowBytes, blocksPerRow);
        } else {
            kQGemvQ3KResid<false><<<nBlocks, block, 0, g_stream>>>(
                    static_cast<const uint8_t *>(wq), x, out, acc, rows,
                    cols, rowBytes, blocksPerRow);
        }
        return true;
    }
    return false;
}
