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

// GpuDequantKernels.hpp -- per-type block tables and IQ/legacy dequant + dot kernels.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// ------------------------------------------------------------------
// Quantized GEMV: y[rows] = x[cols] @ W^T, warp-per-row.
//
// x is staged in shared memory once per block by all 256 threads.
// Each warp owns one output row.  Each lane of the warp owns elements
// [lane*8, lane*8+8) of each 256-element block and precomputes the
// exact (scale, min, byte-base, shift) indices matching
// GGMLDequantize::dequantize{Q2_K,Q3_K,Q4_K}Block so the dequant loop
// is branch-free and bit-identical.
// ------------------------------------------------------------------

// cols must be a multiple of 32 (hiddenSize = 1536, intermediateSize =
// 8960, headDim*heads = 1536 are; the launcher pads cols up to the next
// 32 when they are not).  No shared memory is used: x is read directly
// from global (coalesced, and tiny enough to stay L1/L2-resident).
// P1 exp2 (2026-10-06): EXACT=true drops the per-element column
// guard `(cidx < cols)` when cols is an exact multiple of the
// 256-wide K-quant block (blocksPerRow*256 == cols).  That holds for
// every default-model matrix (H=1536, I=8960, KV=256), and the guard
// is provably dead there, so the arithmetic is BIT-IDENTICAL to the
// guarded path while shedding ~2 SASS ops per weight (the ISETP +
// predicated-zero select) -- the exact "reduce per-byte ALU" lever
// the P0 profile demanded.  The launcher only selects EXACT=true
// after satisfying the multiple-of-256 test, so non-conforming
// models keep the guarded instantiation.
template<int TYPE, bool EXACT = false>
__global__ void kQGemv(const uint8_t *__restrict__ w,
                       const float *__restrict__ x, float *__restrict__ out,
                       uint32_t rows, uint32_t cols, uint32_t rowBytes,
                       uint32_t blocksPerRow) {
    // x is read DIRECTLY from global (coalesced, and tiny enough to
    // stay L1/L2-resident); no shared-memory staging/barrier needed.
    // cols is padded up to a multiple of 32 (blocksPerRow*256 for the
    // K-quants); the final partial block's pad columns [cols, blk*256)
    // MUST contribute 0 (the CPU's SIMD kernels treat them as zero
    // dequant) and MUST NOT be read from x: the scratch x buffers are
    // allocated to the true H/I size (e.g. 896, 4864), so an unguarded
    // read at idx >= cols reads past the buffer into stale device
    // memory (prefill looked fine only because fresh cudaMalloc pages
    // are zero).  The helper masks those lanes to 0.
    uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;

    float acc = 0.0f;
    const uint32_t lane = threadIdx.x;
    const uint32_t e0 = lane * 8u;

    if (TYPE == kTypeQ2K) {
        // Coalesced byte scheme (bit-exact with dequantizeQ2_KBlock):
        // read x DIRECTLY from global (skip the shared-memory staging,
        // which added latency and occupancy pressure).  Reads of x are
        // coalesced (lanes read consecutive floats within each 32-group)
        // and x is tiny (cols <= 1536 floats), so it stays L1/L2-hot
        // across all rows.
        const uint32_t sub = lane / 16u;
        // 4-way block batching (mirrors kQGemv<kTypeIQ2XS>): 4
        // independent (load, dequant, fmaf-chain) streams hide the
        // serial inter-block dependency that kept the Q2_K gate+up at
        // ~23.4 us/layer (~5x the GB/s floor).  Each block's fmaf
        // chain is untouched and the fold (acc += a0; a1; a2; a3) is
        // in block order -- bit-exact with the serial loop.
        uint32_t b = 0;
        for (; b + 4u <= blocksPerRow; b += 4u) {
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
                                    EXACT ? x[cidx]
                                          : ((cidx < cols) ? x[cidx]
                                                           : 0.0f),
                                    slot);
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
                               EXACT ? x[cidx]
                                     : ((cidx < cols) ? x[cidx] : 0.0f),
                               acc);
                }
            }
            rp += kQ2K_BYTES;
        }
    } else if (TYPE == kTypeQ3K) {
        // Same coalesced-byte scheme as Q2_K.  hm (32 bytes, block+0)
        // holds the high-bit mask: bit 1<<jj for the same (sub,l)
        // position => hm[lane] with the 1<<jj mask.  qv = (qb>>2jj&3) -
        // (hm[lane]&(1<<jj) ? 0 : 4).  Scale is the repacked int8
        // sc16[half*8+jj*2+sub].  No dmin (Q3_K has none).
        const uint32_t sub = lane / 16u;
        // 4-way block batching (same rationale as the Q2_K branch):
        // independent (load, dequant, fmaf-chain) streams hide the
        // serial inter-block dependency; in-order fold keeps the
        // result bit-identical with the serial loop.
        uint32_t b = 0;
        for (; b + 4u <= blocksPerRow; b += 4u) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 32
            for (uint32_t bb = 0; bb < 4; ++bb) {
                const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kQ3K_BYTES;
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
                        // the hmask bit advances ACROSS halves too:
                        // half0 uses bits 0..3, half1 bits 4..7.
                        const uint32_t maskBit = 1u << (half * 4u + jj);
                        float qv = static_cast<float>(static_cast<int8_t>(
                                ((qb >> (jj * 2u)) & 3) -
                                ((hmb & maskBit) ? 0 : 4)));
                        const uint32_t cidx =
                                (b + bb) * 256u + half * 128u + jj * 32u + lane;
                        slot = fmaf(dl * qv,
                                    EXACT ? x[cidx]
                                          : ((cidx < cols) ? x[cidx]
                                                           : 0.0f),
                                    slot);
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
                    // the hmask bit advances ACROSS halves too:
                    // half0 uses bits 0..3, half1 bits 4..7.
                    const uint32_t maskBit = 1u << (half * 4u + jj);
                    float qv = static_cast<float>(static_cast<int8_t>(
                            ((qb >> (jj * 2u)) & 3) -
                            ((hmb & maskBit) ? 0 : 4)));
                    const uint32_t cidx =
                            b * 256u + half * 128u + jj * 32u + lane;
                    acc = fmaf(dl * qv,
                               EXACT ? x[cidx]
                                     : ((cidx < cols) ? x[cidx] : 0.0f),
                               acc);
                }
            }
            rp += kQ3K_BYTES;
        }
    } else if (TYPE == kTypeQ4K) {
        // Coalesced byte scheme (bit-exact with dequantizeQ4_KBlock):
        // each 64-weight group g reads 32 q bytes.  Byte
        // qs[g*32 + lane] supplies out[g*64 + lane] from its LOW
        // nibble (sub-scale is+0) and out[g*64 + 32 + lane] from its
        // HIGH nibble (sub-scale is+1).  32 lanes x 2 nibbles = 64
        // weights per group, 4 groups = 256 per block.
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
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            __half dmin = *reinterpret_cast<const __half *>(rp + 2);
            float df = __half2float(d), dmf = __half2float(dmin);
            const uint8_t *scales = rp + 4;
            const uint8_t *qs = rp + 16;
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
                const uint32_t c0 = b * 256u + g * 64u + lane;
                const uint32_t c1 = c0 + 32u;
                acc = fmaf(fmaf(dl0, qv0, -ml0),
                           (c0 < cols) ? x[c0] : 0.0f, acc);
                acc = fmaf(fmaf(dl1, qv1, -ml1),
                           (c1 < cols) ? x[c1] : 0.0f, acc);
            }
            rp += kQ4K_BYTES;
        }
    } else if (TYPE == kTypeQ6K) {
        // Q6_K block layout (210 B): ql(128) + qh(64) + sc(16) + d(2).
        // Each 128-weight half: qh[l] holds 2-bit fields for weights at
        // l, l+32, l+64, l+96; q1/q3 use ql[l] (low/high nibble), q2/q4
        // use ql[32+l]; 8 int8 scales, one per 16-weight group:
        //   sc[half*8 + sub*2 + (l/16)], value = d*scale*(q-32).
        const uint32_t half = e0 / 128u;
        const uint32_t sub = (e0 % 128u) / 32u;
        const uint32_t l = e0 % 32u;
        const uint32_t g16 = l / 16u;
        const uint32_t qlBase = half * 64u + ((sub & 1u) ? 32u : 0u);
        const uint32_t qhBase = half * 32u;
        const uint32_t shift = sub * 2u;            // 0,2,4,6
        const uint32_t hiNib = (sub & 2u) ? 4u : 0u;// q3/q4 use high nibble
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
            const uint8_t *rb = rp;
            const uint8_t *ql = rb;
            const uint8_t *qh = rb + 128;
            const int8_t *sc = reinterpret_cast<const int8_t *>(rb + 192);
            __half d = *reinterpret_cast<const __half *>(rb + 208);
            float dl = __half2float(d) *
                       static_cast<float>(sc[half * 8u + sub * 2u + g16]);
            const float *xs = x + b * 256u + e0;
#pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                uint32_t qb = (hiNib) ? (ql[qlBase + l + k] >> 4)
                                      : (ql[qlBase + l + k] & 0xF);
                qb |= ((qh[qhBase + l + k] >> shift) & 3u) << 4u;
                float qv = static_cast<float>(static_cast<int>(qb) - 32);
                const uint32_t ck = b * 256u + e0 + k;
                acc = fmaf(dl * qv, (ck < cols) ? xs[k] : 0.0f, acc);
            }
            rp += kQ6K_BYTES;
        }
    } else if (TYPE == kTypeIQ2XS) {
        // IQ2_XS block (74 B): d(2) + qs[64] (32 x uint16) +
        // scales[8].  Each uint16 holds a 9-bit grid index + 7-bit
        // sign index.  Each lane owns 8 consecutive columns = exactly
        // one (ib32, l) group's 8 weights (e0 is a multiple of 8), so
        // one (gridIdx, signIdx) lookup feeds all 8 of its columns.
        //
        // BLOCK BATCHING (2026-09-27): the per-lane float acc is
        // INDEPENDENT across blocks (the final 5-shuffle tree sums
        // lanes AFTER the whole loop), so processing 4 blocks into
        // separate accumulators a0..a3 and folding them at the end is
        // bit-exact -- the fmaf chain inside each block is unchanged,
        // and float addition is associative only in order, which we
        // preserve by folding a0+a1+a2+a3 into acc serially.  The
        // motive: 4 independent (load, decode, fmaf-chain) streams hide
        // the inter-block dependency that kept the 7B IQ2_S gate+up at
        // ~215 us/call (~5x the ~44 us GB/s floor).
        uint32_t b = 0;
        for (; b + 4u <= blocksPerRow; b += 4u) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 32
            for (uint32_t bb = 0; bb < 4; ++bb) {
                const uint8_t *crp = rp + static_cast<uint64_t>(bb) * kIQ2XS_BYTES;
                __half d = *reinterpret_cast<const __half *>(crp + 0);
                float df = __half2float(d);
                const uint16_t *qs16 =
                        reinterpret_cast<const uint16_t *>(crp + 2);
                const uint8_t *scales = crp + 66;
                const uint32_t ib32 = e0 >> 5u;    // sub-block 0..7
                const uint32_t l = (e0 >> 3u) & 3u;// 0..3
                float db0 = df * (0.5f + static_cast<float>(scales[ib32] & 0xf)) *
                            0.25f;
                float db1 = df * (0.5f + static_cast<float>(scales[ib32] >> 4)) *
                            0.25f;
                // CPU reference: dl = db[l/2] (l=0,1 -> db0; l=2,3 -> db1).
                float dl = (l >= 2u) ? db1 : db0;
                uint16_t qval = qs16[ib32 * 4u + l];
                uint16_t gridIdx = qval & 0x1FFu;
                uint8_t signIdx = static_cast<uint8_t>(qval >> 9);
                const uint64_t gv = c_iq2xs_grid[gridIdx];
                uint8_t signs = c_ksigns_iq2xs[signIdx];
                const float *xs = x + (b + bb) * 256u + e0;
                float &slot = (bb == 0u) ? a0 : (bb == 1u) ? a1
                                        : (bb == 2u)       ? a2
                                                           : a3;
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    float w =
                            dl * static_cast<float>((gv >> (8u * j)) & 0xFFu) *
                            ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    const uint32_t ck = (b + bb) * 256u + e0 + j;
                    slot = fmaf(w, (ck < cols) ? xs[j] : 0.0f, slot);
                }
            }
            acc += a0;
            acc += a1;
            acc += a2;
            acc += a3;
            rp += 4u * kIQ2XS_BYTES;
        }
        for (; b < blocksPerRow; ++b) {
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            float df = __half2float(d);
            const uint16_t *qs16 =
                    reinterpret_cast<const uint16_t *>(rp + 2);
            const uint8_t *scales = rp + 66;
            const uint32_t ib32 = e0 >> 5u;    // sub-block 0..7
            const uint32_t l = (e0 >> 3u) & 3u;// 0..3
            float db0 = df * (0.5f + static_cast<float>(scales[ib32] & 0xf)) *
                        0.25f;
            float db1 = df * (0.5f + static_cast<float>(scales[ib32] >> 4)) *
                        0.25f;
            float dl = (l >= 2u) ? db1 : db0;
            uint16_t qval = qs16[ib32 * 4u + l];
            uint16_t gridIdx = qval & 0x1FFu;
            uint8_t signIdx = static_cast<uint8_t>(qval >> 9);
            const uint64_t gv = c_iq2xs_grid[gridIdx];
            uint8_t signs = c_ksigns_iq2xs[signIdx];
            const float *xs = x + b * 256u + e0;
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                float w =
                        dl * static_cast<float>((gv >> (8u * j)) & 0xFFu) *
                        ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t ck = b * 256u + e0 + j;
                acc = fmaf(w, (ck < cols) ? xs[j] : 0.0f, acc);
            }
            rp += kIQ2XS_BYTES;
        }
    } else if (TYPE == kTypeIQ2XXS) {
        // IQ2_XXS block (66 B): d(2) + qs[32 x uint16] (64 B).
        // Per (ib32, l) group (ib32 = sub-block 0..7, l = 0..3) the
        // lane reads 8 bytes = qs[4*ib32 .. 4*ib32+7] as two LE
        // uint32 words.  Bytes 0..3 (word0) hold the four grid
        // indices aux8[l] into iq2xxs_grid[256]; word1 bits 7*l..7*l+6
        // hold the sign index and word1 bits 28..31 the scale:
        //   db = d*(0.5 + (word1>>28))*0.25
        //   w  = db * grid[j] * (signs & kmask[j] ? -1 : 1)
        // (bit-identical to GGMLDequantize::dequantizeIQ2_XXSBlock).
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
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
            // Native-word load + byte-extract (same fix as IQ2XS).
            const uint64_t gv = c_iq2xxs_grid[gridIdx];
            const float *xs = x + b * 256u + e0;
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                float wgt = db *
                            static_cast<float>((gv >> (8u * j)) & 0xFFu) *
                            ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t ck = b * 256u + e0 + j;
                acc = fmaf(wgt, (ck < cols) ? xs[j] : 0.0f, acc);
            }
            rp += kIQ2XXS_BYTES;
        }
    } else if (TYPE == kTypeIQ4XS) {
        // IQ4_XS block (136 B): d(2) + scales_h(2) + scales_l(4) +
        // qs[128].  Per 32-weight sub-block ib:
        //   ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf)
        //        | (((scales_h >> 2*ib) & 3) << 4)
        //   dl = d*(ls - 32); qs[ib*16 + j] byte -> col ib*32+j (low
        //   nibble) and col ib*32+16+j (high nibble) of kvalues_iq4nl.
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            float df = __half2float(d);
            uint16_t scales_h;
            std::memcpy(&scales_h, rp + 2, sizeof(uint16_t));
            const uint8_t *scales_l = rp + 4;
            const uint8_t *qsv = rp + 8;
            const uint32_t ib = e0 >> 5u;      // sub-block 0..7
            const uint32_t l = (e0 >> 3u) & 3u;// 4 bytes -> 8 weights
            const int ls =
                    static_cast<int>((scales_l[ib / 2u] >>
                                      (4u * (ib % 2u))) &
                                     0xfu) |
                    (static_cast<int>((scales_h >> (2u * ib)) & 0x3u)
                     << 4);
            float dl = df * static_cast<float>(ls - 32);
            const uint8_t *qb = qsv + ib * 16u + l * 4u;
            const float *xs = x + b * 256u + e0;
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t jc = l * 8u + j;// col within sub-block
                const uint32_t byteIdx = jc & 15u;
                const uint32_t nib = (jc < 16u)
                                             ? (qb[byteIdx] & 0xfu)
                                             : (qb[byteIdx] >> 4);
                float wgt = dl * static_cast<float>(c_kvalues_iq4nl[nib]);
                const uint32_t ck = b * 256u + e0 + j;
                acc = fmaf(wgt, (ck < cols) ? xs[j] : 0.0f, acc);
            }
            rp += kIQ4XS_BYTES;
        }
    } else if (TYPE == kTypeIQ2S) {
        // IQ2_S block (82 B): d(2) + qs[64] + qh[8] + scales[8].
        // Per (ib32, l) group (ib32 = sub-block 0..7, l = 0..3):
        //   dl  = db[l/2] where db0 = d*(0.5+(scales[ib32]&0xf))*0.25,
        //                     db1 = d*(0.5+(scales[ib32]>>4))*0.25
        //   gridIdx = qs[ib32*4 + l] | (qh[ib32] << (8-2l) & 0x300)
        //   signs   = qs[32 + ib32*4 + l]   (the sign byte lives in the
        //             qs tail -- CPU: signs = qs + QK_K/8)
        // The lane owns 8 consecutive columns => exactly one (ib32,l).
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            float df = __half2float(d);
            const uint8_t *qsv = rp + 2;    // 64 q bytes (grid idx + signs)
            const uint8_t *qh = rp + 66;    // 8 high bits
            const uint8_t *scales = rp + 74;// 8 scales
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
            const float *xs = x + b * 256u + e0;
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                float w =
                        dl * static_cast<float>(grid[j]) *
                        ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t ck = b * 256u + e0 + j;
                acc = fmaf(w, (ck < cols) ? xs[j] : 0.0f, acc);
            }
            rp += kIQ2S_BYTES;
        }
    } else if (TYPE == kTypeIQ3_XXS) {
        // IQ3_XXS block (98 B): d(2) + qs[96].  qs[64..95] doubles as
        // the scales+signs area (scales_and_signs = qs + QK_K/4 = qs+64).
        // Per (ib32, l):
        //   aux32 = u32(scales_and_signs + 4*ib32)
        //   db    = d*(0.5f + (aux32 >> 28))*0.5f
        //   signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]
        //   grid1 = qs[8*ib32 + 2*l + 0], grid2 = qs[8*ib32 + 2*l + 1]
        //   y[0..3] from grid1 bytes, y[4..7] from grid2 bytes.
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
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
            const float *xs = x + b * 256u + e0;
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                uint32_t gval = (j < 4u)
                                        ? ((g1 >> (8u * j)) & 0xFFu)
                                        : ((g2 >> (8u * (j - 4u))) &
                                           0xFFu);
                float w = db * static_cast<float>(gval) *
                          ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t ck = b * 256u + e0 + j;
                acc = fmaf(w, (ck < cols) ? xs[j] : 0.0f, acc);
            }
            rp += kIQ3XXS_BYTES;
        }
    } else if (TYPE == kTypeIQ3S) {
        // IQ3_S block (110 B): d(2) + qs[64] + qh[8] + signs[32] +
        // scales[4].  Sub-blocks pair up: pair=ib32>>1, half=ib32&1.
        // First half of a pair uses db1 = d*(1+2*(scales[pair]&0xf));
        // second half db2 = d*(1+2*(scales[pair]>>4)).  qh advance by
        // 2 per pair; qs by 16 per pair (8 per half); signs by 8 per
        // pair (4 per half).  Grid values are 4 x 4-bit bytes packed in
        // a uint32_t (grid1[j] = byte j).
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
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
            float db =
                    df *
                    (1.0f +
                     2.0f *
                             static_cast<float>((half == 0)
                                                        ? (scales[pair] & 0xf)
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
            const float *xs = x + b * 256u + e0;
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                uint32_t gval = (j < 4u)
                                        ? ((g1 >> (8u * j)) & 0xFFu)
                                        : ((g2 >> (8u * (j - 4u))) &
                                           0xFFu);
                float w = db * static_cast<float>(gval) *
                          ((sbyte & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                const uint32_t ck = b * 256u + e0 + j;
                acc = fmaf(w, (ck < cols) ? xs[j] : 0.0f, acc);
            }
            rp += kIQ3S_BYTES;
        }
    } else if (TYPE == kTypeQ5K) {
        // Q5_K block (176 B): d(2) + dmin(2) + scales[12] + qh[32] +
        // qs[128].  4 chunks of 64; sub-block 0 of a chunk uses qh bit
        // u1 = 1<<(2*g), sub-block 1 uses u2 = 2<<(2*g).  Lane owns 8
        // consecutive columns that never cross a 32-weight sub-block
        // boundary (e0 is a multiple of 8), so qs/qh byte = g*32 + jj.
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
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
            __half d = *reinterpret_cast<const __half *>(rp + 0);
            __half dmin = *reinterpret_cast<const __half *>(rp + 2);
            float df = __half2float(d), dmf = __half2float(dmin);
            const uint8_t *scales = rp + 4;
            const uint8_t *qh = rp + 16;
            const uint8_t *qs = rp + 48;
            const uint32_t g = e0 / 64u;         // chunk 0..3
            const uint32_t sub = (e0 / 32u) & 1u;// 0 or 1
            const uint32_t jj0 = e0 & 31u;       // first of 8
            uint8_t sc, m;
            getScaleMin(static_cast<int>(2 * g + sub), scales, &sc, &m);
            float dl = df * static_cast<float>(sc);
            float ml = dmf * static_cast<float>(m);
            const uint8_t u = (sub == 0)
                                      ? static_cast<uint8_t>(1u << (2u * g))
                                      : static_cast<uint8_t>(2u << (2u * g));
            const uint8_t *qsp = qs + g * 32u;
            // NOTE: qh does NOT advance per chunk -- the same qh[l]
            // bytes are reused for all 4 chunks with the u1/u2 bit
            // masks selecting the (chunk, sub-block) position (matches
            // dequantizeQ5_KBlock, which advances only qs by 32).
            const float *xs = x + b * 256u + e0;
#pragma unroll
            for (uint32_t kk = 0; kk < 8; ++kk) {
                const uint8_t qb = qsp[jj0 + kk];
                uint8_t q5 = (sub == 0)
                                     ? static_cast<uint8_t>(
                                               (qb & 0xF) |
                                               ((qh[jj0 + kk] & u) ? 16
                                                                   : 0))
                                     : static_cast<uint8_t>(
                                               (qb >> 4) |
                                               ((qh[jj0 + kk] & u) ? 16
                                                                   : 0));
                float w = dl * static_cast<float>(q5) - ml;
                const uint32_t ck = b * 256u + e0 + kk;
                acc = fmaf(w, (ck < cols) ? xs[kk] : 0.0f, acc);
            }
            rp += kQ5K_BYTES;
        }
    } else {
        acc = 0.0f;
    }
// Warp reduction: each lane holds the partial dot product over its
// disjoint 8-column-per-block subset; the full row result is the
// sum across all 32 lanes.  (5 shuffles + 5 adds per row.)
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) out[row] = acc;
}
