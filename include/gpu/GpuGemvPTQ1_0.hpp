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

// GpuGemvPTQ1_0.hpp -- Prism ptq1_0 GEMV (split-K, native trits, Q8 quantised input).
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// Split-K native PTQ1_0 GEMV: one warp per output row, one lane per
// 128-weight block, fp32 folded activation.  Thread kb reads the block
// as 28 contiguous bytes (7 words) and the matching 128 activation
// floats (float4-aligned groups), so the weight stream is a plain
// coalesced sweep with no transposed buffer and no warp exchange.
// Numerical order differs from the row-parallel oracle; the math is
// exact fp32 fma on the trits.
__device__ __forceinline__ void ptq1Trits5(const uint32_t P,
                                           uint32_t (&D)[5]) {
    const uint32_t vlo = __byte_perm(P, 0, 0x4140);
    const uint32_t vhi = __byte_perm(P, 0, 0x4342);
    uint32_t Fp = 0u;
    const uint32_t pow3[5] = {3u, 9u, 27u, 81u, 243u};
#pragma unroll
    for (uint32_t t = 0; t < 5u; ++t) {
        const uint32_t Fn =
                __byte_perm(vlo * pow3[t], vhi * pow3[t], 0x7531);
        D[t] = Fn - 3u * Fp;
        Fp = Fn;
    }
}

__device__ __forceinline__ float ptq1DigitDot4(const uint32_t d,
                                               const float4 xv) {
    float s = static_cast<float>(static_cast<int>(d & 0xFFu) - 1) *
              xv.x;
    s = fmaf(static_cast<float>(static_cast<int>((d >> 8) & 0xFFu) - 1),
             xv.y, s);
    s = fmaf(
            static_cast<float>(static_cast<int>((d >> 16) & 0xFFu) - 1),
            xv.z, s);
    s = fmaf(
            static_cast<float>(static_cast<int>((d >> 24) & 0xFFu) - 1),
            xv.w, s);
    return s;
}

template<uint32_t NW>
__global__ void __launch_bounds__(NW * 32u) kQGemvPTQ1_0SplitK(
        const uint8_t *__restrict__ w, const float *__restrict__ x,
        float *__restrict__ out, uint32_t rows, uint32_t cols,
        uint32_t rowBytes, uint32_t bpr) {
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * NW + warp;
    if (row >= rows) return;
    const uint8_t *rp =
            w + static_cast<uint64_t>(row) * rowBytes;
    float acc = 0.0f;
    for (uint32_t kb = lane; kb < bpr; kb += 32u) {
        const uint32_t *p = reinterpret_cast<const uint32_t *>(
                rp + static_cast<uint64_t>(kb) * 28u);
        const float *xb = x + static_cast<uint64_t>(kb) * 128u;
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (uint32_t g = 0; g < 6u; ++g) {
            uint32_t D[5];
            ptq1Trits5(p[g], D);
#pragma unroll
            for (uint32_t t = 0; t < 5u; ++t) {
                const uint32_t e = (g < 4u)
                                           ? (16u * t + 4u * g)
                                           : (80u + 8u * t + 4u * (g - 4u));
                a[e >> 5] += ptq1DigitDot4(
                        D[t], *reinterpret_cast<const float4 *>(xb + e));
            }
        }
        uint32_t v = __byte_perm(p[6], 0, 0x4140);
#pragma unroll
        for (uint32_t t = 0; t < 4u; t += 2u) {
            const uint32_t w0 = v * 3u;
            v = w0 & 0x00FF00FFu;
            const uint32_t w1 = v * 3u;
            v = w1 & 0x00FF00FFu;
            const uint32_t q = __byte_perm(w0, w1, 0x7531);
            a[3] += ptq1DigitDot4(
                    q, *reinterpret_cast<const float4 *>(xb + 120u + 2u * t));
        }
        const float d = __half2float(__ushort_as_half(
                static_cast<unsigned short>(p[6] >> 16)));
        acc += d * (a[0] + a[1] + a[2] + a[3]);
    }
#pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0u) out[row] = acc;
}

// Lossless 28 B -> 34 B PTQ1_0 -> PQ2_0 repack on the device, used only
// by the prefill path: the PQ2 int8 tile loader is 1.75x faster than
// the native one, so prefill repacks one matrix at a time into scratch
// and runs the PQ2 GEMM.  Layout is column order (code j at byte 2 +
// j/4, two bits at 2*(j%4)); d is copied verbatim.
__global__ void __launch_bounds__(128) kPTQ1_0ToPQ2(
        const uint8_t *__restrict__ src, uint8_t *__restrict__ dst,
        uint32_t nblocks) {
    const uint32_t nb = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (nb >= nblocks) return;
    const uint32_t t = threadIdx.x & 31u;
    const uint8_t *s = src + static_cast<uint64_t>(nb) * 28u;
    uint8_t *o = dst + static_cast<uint64_t>(nb) * 34u;
    if (t == 0u) {
        o[0] = s[26];
        o[1] = s[27];
    }
    const uint8_t pow3[5] = {1u, 3u, 9u, 27u, 81u};
    uint8_t byte = 0u;
#pragma unroll
    for (uint32_t jj = 0; jj < 4u; ++jj) {
        const uint32_t j = 4u * t + jj;
        uint8_t q;
        if (j < 80u) {
            q = static_cast<uint8_t>(s[j % 16u] * pow3[j / 16u]);
        } else if (j < 120u) {
            q = static_cast<uint8_t>(s[16u + (j - 80u) % 8u] *
                                     pow3[(j - 80u) / 8u]);
        } else {
            q = static_cast<uint8_t>(s[24u + (j - 120u) % 2u] *
                                     pow3[(j - 120u) / 2u]);
        }
        const uint8_t code = static_cast<uint8_t>(
                (static_cast<uint16_t>(q) * 3u) >> 8);
        byte |= static_cast<uint8_t>((code & 3u) << (2u * jj));
    }
    o[2u + t] = byte;
}

// Per-32 q8_1 quantisation of the folded activation, for the split-K
// GEMV: one warp per 32-element chunk, scale = max/127, integer sum
// exact.  ds holds (d8, isum) per 32-chunk, eight floats per 128-block.
__global__ void __launch_bounds__(128) kQuantizeQ8Blocks(
        const float *__restrict__ x, int8_t *__restrict__ q,
        float *__restrict__ ds, uint32_t nblocks) {
    const uint32_t b = blockIdx.x;
    const uint32_t t = threadIdx.x;
    const uint32_t ch = t >> 5;
    const uint32_t lane = t & 31u;
    const uint32_t base = b * 128u + ch * 32u;
    const float v = x[base + lane];
    float mx = fabsf(v);
#pragma unroll
    for (uint32_t o = 16u; o > 0u; o >>= 1u)
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
    const float d = mx > 0.0f ? mx / 127.0f : 0.0f;
    int qi = d > 0.0f ? __float2int_rn(v / d) : 0;
    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
    q[base + lane] = static_cast<int8_t>(qi);
    int is = qi;
#pragma unroll
    for (uint32_t o = 16u; o > 0u; o >>= 1u)
        is += __shfl_xor_sync(0xFFFFFFFFu, is, o);
    if (lane == 0u) {
        ds[b * 8u + ch * 2u] = d;
        ds[b * 8u + ch * 2u + 1u] = static_cast<float>(is);
    }
}

// Split-K native PTQ1_0 GEMV, the fork's geometry: thread (row group,
// 128-block) loads one q8 activation block and RPT weight blocks,
// dp4a-decodes trits with ptq1Trits5, and writes a partial to shared;
// a second phase reduces the K blocks per row.  Weight reads are 28
// contiguous bytes per block and the activation is int8, so neither
// side wastes a transaction.
template<uint32_t G, uint32_t RPT>
__global__ void __launch_bounds__(512) kQGemvPTQ1_0SplitQ8(
        const uint8_t *__restrict__ w, const int8_t *__restrict__ xq,
        const float *__restrict__ xd, float *__restrict__ out,
        uint32_t rows, uint32_t cols, uint32_t rowBytes, uint32_t bpr) {
    constexpr uint32_t R = G * RPT;
    extern __shared__ float part[];// [R][bpr]
    const uint32_t tid = threadIdx.x;
    const uint32_t bprP = (bpr + 31u) & ~31u;
    const uint32_t g = tid / bprP;
    const uint32_t kb = tid - g * bprP;
    const uint32_t row0 = blockIdx.x * R;
    if (g < G && kb < bpr) {
        int ua[32];
        const int4 *q4 = reinterpret_cast<const int4 *>(
                xq + static_cast<uint64_t>(kb) * 128u);
#pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) {
            const int4 v = q4[i];
            ua[i * 4u + 0u] = v.x;
            ua[i * 4u + 1u] = v.y;
            ua[i * 4u + 2u] = v.z;
            ua[i * 4u + 3u] = v.w;
        }
#pragma unroll
        for (uint32_t i = 0; i < RPT; ++i) {
            const uint32_t r = g + G * i;
            const uint32_t row = row0 + r;
            if (row >= rows) continue;
            const uint32_t *p = reinterpret_cast<const uint32_t *>(
                    w + static_cast<uint64_t>(row) * rowBytes +
                    static_cast<uint64_t>(kb) * 28u);
            int sumi[4] = {0, 0, 0, 0};
#pragma unroll
            for (uint32_t gg = 0; gg < 6u; ++gg) {
                uint32_t D[5];
                ptq1Trits5(p[gg], D);
#pragma unroll
                for (uint32_t t = 0; t < 5u; ++t) {
                    const uint32_t e =
                            (gg < 4u) ? (16u * t + 4u * gg)
                                      : (80u + 8u * t + 4u * (gg - 4u));
                    sumi[e >> 5] = __dp4a(
                            static_cast<int>(D[t]),
                            ua[(e >> 5) * 8u + ((e & 31u) >> 2u)],
                            sumi[e >> 5]);
                }
            }
            uint32_t v = __byte_perm(p[6], 0, 0x4140);
#pragma unroll
            for (uint32_t t = 0; t < 4u; t += 2u) {
                const uint32_t w0 = v * 3u;
                v = w0 & 0x00FF00FFu;
                const uint32_t w1 = v * 3u;
                v = w1 & 0x00FF00FFu;
                const uint32_t qq = __byte_perm(w0, w1, 0x7531);
                sumi[3] = __dp4a(static_cast<int>(qq),
                                 ua[3u * 8u + 6u + t / 2u], sumi[3]);
            }
            float aa = 0.0f;
#pragma unroll
            for (uint32_t k = 0; k < 4u; ++k) {
                const float d8 = xd[kb * 8u + k * 2u];
                const float is = xd[kb * 8u + k * 2u + 1u];
                aa = fmaf(d8, static_cast<float>(sumi[k] - static_cast<int>(is)),
                          aa);
            }
            const float dd = __half2float(__ushort_as_half(
                    static_cast<unsigned short>(p[6] >> 16)));
            part[r * bpr + kb] = dd * aa;
        }
    }
    __syncthreads();
    const uint32_t nw = blockDim.x >> 5;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    for (uint32_t r = warp; r < R; r += nw) {
        float sm = 0.0f;
        for (uint32_t k = lane; k < bpr; k += 32u)
            sm += part[r * bpr + k];
#pragma unroll
        for (uint32_t o = 16u; o > 0u; o >>= 1u)
            sm += __shfl_down_sync(0xFFFFFFFFu, sm, o);
        if (lane == 0u && row0 + r < rows) out[row0 + r] = sm;
    }
}

inline void launchQGemvPTQ1_0SplitQ8(const uint8_t *w, const float *x,
                                     int8_t *xq, float *xd, float *out,
                                     uint32_t rows, uint32_t cols,
                                     uint32_t rowBytes,
                                     uint32_t blocksPerRow) {
    const uint32_t nb = cols / 128u;
    kQuantizeQ8Blocks<<<nb, 128, 0, g_stream>>>(x, xq, xd, nb);
    static const uint32_t rptSel = [] {
        const char *e = std::getenv("TINYCODER_PT_SPLITRPT");
        return (e != nullptr && e[0] == '1')   ? 1u
               : (e != nullptr && e[0] == '2') ? 2u
               : (e != nullptr && e[0] == '8') ? 8u
                                               : 4u;
    }();
    const bool rpt8 = rptSel == 8u;
    const uint32_t G = blocksPerRow <= 42u   ? 6u
                       : blocksPerRow <= 48u ? 5u
                                             : 1u;
    const uint32_t RPT = rptSel;
    const uint32_t R = G * RPT;
    const uint32_t threads = G * ((blocksPerRow + 31u) & ~31u);
    const uint32_t shared = R * blocksPerRow * sizeof(float);
    const uint32_t grid = (rows + R - 1u) / R;
    if (rptSel == 1u) {
        switch (G) {
            case 6u:
                kQGemvPTQ1_0SplitQ8<6u, 1u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
            case 5u:
                kQGemvPTQ1_0SplitQ8<5u, 1u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
            default:
                kQGemvPTQ1_0SplitQ8<1u, 1u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
        }
    }
    if (rptSel == 2u) {
        switch (G) {
            case 6u:
                kQGemvPTQ1_0SplitQ8<6u, 2u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
            case 5u:
                kQGemvPTQ1_0SplitQ8<5u, 2u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
            default:
                kQGemvPTQ1_0SplitQ8<1u, 2u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
        }
    }
    if (rpt8) {
        switch (G) {
            case 8u:
                kQGemvPTQ1_0SplitQ8<8u, 8u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
            case 3u:
                kQGemvPTQ1_0SplitQ8<3u, 8u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
            case 5u:
                kQGemvPTQ1_0SplitQ8<5u, 8u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
            case 6u:
                kQGemvPTQ1_0SplitQ8<6u, 8u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
            default:
                kQGemvPTQ1_0SplitQ8<1u, 8u>
                        <<<grid, threads, shared, g_stream>>>(
                                w, xq, xd, out, rows, cols, rowBytes,
                                blocksPerRow);
                return;
        }
    }
    switch (G) {
        case 8u:
            kQGemvPTQ1_0SplitQ8<8u, 4u>
                    <<<grid, threads, shared, g_stream>>>(
                            w, xq, xd, out, rows, cols, rowBytes,
                            blocksPerRow);
            break;
        case 3u:
            kQGemvPTQ1_0SplitQ8<3u, 4u>
                    <<<grid, threads, shared, g_stream>>>(
                            w, xq, xd, out, rows, cols, rowBytes,
                            blocksPerRow);
            break;
        case 5u:
            kQGemvPTQ1_0SplitQ8<5u, 4u>
                    <<<grid, threads, shared, g_stream>>>(
                            w, xq, xd, out, rows, cols, rowBytes,
                            blocksPerRow);
            break;
        case 6u:
            kQGemvPTQ1_0SplitQ8<6u, 4u>
                    <<<grid, threads, shared, g_stream>>>(
                            w, xq, xd, out, rows, cols, rowBytes,
                            blocksPerRow);
            break;
        default:
            kQGemvPTQ1_0SplitQ8<1u, 4u>
                    <<<grid, threads, shared, g_stream>>>(
                            w, xq, xd, out, rows, cols, rowBytes,
                            blocksPerRow);
            break;
    }
}

inline void launchQGemvPTQ1_0SplitK(const uint8_t *w, const float *x,
                                    float *out, uint32_t rows,
                                    uint32_t cols, uint32_t rowBytes,
                                    uint32_t blocksPerRow) {
    kQGemvPTQ1_0SplitK<4u><<<(rows + 3u) / 4u, 128, 0, g_stream>>>(
            w, x, out, rows, cols, rowBytes, blocksPerRow);
}

// Native PTQ1_0 q8 GEMV: R=4 exchange kernel (probe best bit-exact
// variant, 428 GB/s on the 19.5 MB probe, vs 306 at R=4 contiguous).
inline void launchQGemvPTQ1_0Q8(const uint8_t *w, const int8_t *xqT,
                                const float *xd, float *out,
                                uint32_t rows, uint32_t cols,
                                uint32_t rowBytes,
                                uint32_t blocksPerRow) {
    static const uint32_t R = [] {
        const char *e = std::getenv("TINYCODER_PT_Q8R");
        return (e != nullptr && e[0] == '1') ? 1u : 4u;
    }();
    if (R == 1u) {
        kQGemvPTQ1_0Q8<1u><<<(rows + 3u) / 4u, 128, 0, g_stream>>>(
                w, xqT, xd, out, rows, cols, rowBytes, blocksPerRow);
    } else {
        kQGemvPTQ1_0Q8<4u><<<(rows + 15u) / 16u, 128, 0, g_stream>>>(
                w, xqT, xd, out, rows, cols, rowBytes, blocksPerRow);
    }
}

inline void launchQGemvPTQ1_0(const uint8_t *w, const float *x,
                              float *out, uint32_t rows, uint32_t cols,
                              uint32_t rowBytes,
                              uint32_t blocksPerRow) {
    const uint32_t blocks = (rows + 31u) / 32u;
    kQGemvPTQ1_0T<8u, 4u><<<blocks, 128, 0, g_stream>>>(
            w, x, out, rows, cols, rowBytes, blocksPerRow);
}

// Diagnostic: read-only twin of the q8 GEMV's weight stream (same bytes,
// no math, garbage output).  TINYCODER_PQ2_TWIN=1 swaps it in to price
// the compute against the stream in situ.  Never enabled by default.
__global__ void __launch_bounds__(128) kQGemvPQ2_0Twin(
        const uint8_t *__restrict__ w, float *__restrict__ out,
        uint32_t rows, uint32_t rowBytes, uint32_t blocksPerRow) {
    const uint32_t warp = blockIdx.x * 4u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = warp * 4u;
    if (row0 >= rows) return;
    const uint32_t nr = (rows - row0 < 4u) ? (rows - row0) : 4u;
    const uint8_t *rp[4];
#pragma unroll
    for (uint32_t r = 0; r < 4u; ++r)
        rp[r] = w + (uint64_t) (row0 + (r < nr ? r : nr - 1u)) * rowBytes;
    uint32_t s[4] = {0u, 0u, 0u, 0u};
    for (uint32_t b = 0; b < blocksPerRow; ++b) {
#pragma unroll
        for (uint32_t r = 0; r < 4u; ++r) {
            if (r >= nr) break;
            const uint8_t *bp = rp[r] + (uint64_t) b * 34u;
            s[r] += (uint32_t) __ldg(reinterpret_cast<const uint16_t *>(
                    bp + 2u + 2u * (lane & 15u)));
        }
    }
    if (lane == 0) {
#pragma unroll
        for (uint32_t r = 0; r < 4u; ++r)
            if (row0 + r < rows) out[row0 + r] = (float) s[r];
    }
}

// Rows per warp for the fork-style q8 GEMV.  The reference uses 4;
// TINYCODER_PQ2_Q8R selects the A/B geometry (1/2/4/8).
inline uint32_t pq2Q8Rows() {
    static const uint32_t r = [] {
        const char *e = std::getenv("TINYCODER_PQ2_Q8R");
        const long v = e ? std::strtol(e, nullptr, 10) : 4;
        return (v == 1 || v == 2 || v == 4 || v == 8)
                       ? static_cast<uint32_t>(v)
                       : static_cast<uint32_t>(4);
    }();
    return r;
}

// True when the q8 decode path should be used (default ON;
// TINYCODER_PQ2_Q8=0 restores the fp32-activation GEMV).
inline bool pq2Q8Enabled() {
    static const bool on = [] {
        const char *e = std::getenv("TINYCODER_PQ2_Q8");
        return e == nullptr || e[0] == '\0' || e[0] != '0';
    }();
    return on;
}

// Native PTQ1_0 q8 decode is opt-in: the int8 activation path is
// approximate, while the validated native path is the fp32 oracle.
// TINYCODER_PT_Q8=1 enables the dp4a GEMV for PTQ1_0 weights.
// ------------------------------------------------------------------
// K-block-stationary PTQ1_0 GEMV, ported verbatim from the Prism
// fork's mul_mat_vec_ptq1_0_ks (ggml-cuda/mmvq-ptq1_0.cuh).  Each
// thread owns one (row group, 128-weight block) pair, holds the whole
// q8_1 activation block in registers, walks RPT weight blocks, writes
// (w, dot) partials to shared, and a second phase reduces the K blocks
// per row.  The activation uses the fork's SOA_ISUM grouped layout:
// 32 K-blocks per group, word w of the group contiguous, so a warp
// load of one word is 32 consecutive words (one wavefront).
// ------------------------------------------------------------------
constexpr uint32_t kPtq1KsGroupKb = 32u;
constexpr uint32_t kPtq1KsWordsPerKb = 36u;
constexpr uint32_t kPtq1KsGroupWords = kPtq1KsGroupKb * kPtq1KsWordsPerKb;

// Word offset within one activation column of word w (0..7 qs words,
// 8 ds) of q8_1 block ib.
__host__ __device__ __forceinline__ uint32_t ptq1KsQ8Word(uint32_t ib,
                                                          uint32_t w) {
    const uint32_t kBlock = ib >> 2;
    const uint32_t sub = ib & 3u;
    const uint32_t group = kBlock >> 5;
    const uint32_t lane = kBlock & 31u;
    const uint32_t ww = w < 8u ? sub * 8u + w : 32u + sub;
    return group * kPtq1KsGroupWords + ww * kPtq1KsGroupKb + lane;
}

// PT ("piece of 16 bytes") activation layout, the fork's PTQ1_0
// planar-transposed SoA (ggml-cuda/mmvq-ptq1_0.cuh).  The 36 words of a
// 128-weight block are stored as nine 16-byte pieces, plane-major: piece
// t of block kb sits at (t*nblk + kb)*4 ints, so a warp whose lanes own
// consecutive blocks reads 32*16 contiguous bytes per load and a thread
// needs 9 int4 loads instead of 36 strided 4-byte ones.  Plane t of a
// block holds the 16 quants of columns t*16..t*16+15, which in the old
// SoA were words ww = 4t..4t+3; plane 8 holds the four (d, isum) words.
__host__ __device__ __forceinline__ uint32_t ptq1PtQ8Int(uint32_t chunk,
                                                         uint32_t lane,
                                                         uint32_t nblk) {
    const uint32_t kBlock = chunk >> 2;
    const uint32_t sub = chunk & 3u;
    const uint32_t t = 2u * sub + (lane >> 4);
    const uint32_t b = lane & 15u;
    return t * (nblk * 4u) + kBlock * 4u + (b >> 2);
}
__host__ __device__ __forceinline__ uint32_t ptq1PtDsInt(uint32_t chunk,
                                                         uint32_t nblk) {
    return 8u * (nblk * 4u) + (chunk >> 2) * 4u + (chunk & 3u);
}

// PDL: the fork overlaps the preceding quantize kernel with this one on
// Hopper via cudaGridDependencySynchronize.  sm_75 has no PDL, so the
// call compiles out; the launch order on g_stream is the fallback.
__device__ __forceinline__ void ptq1KsPdlSync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
}

// Quantize the folded activation into the SOA_ISUM grouped layout.  One
// warp per 32-column chunk; scale = max/127, exact integer sum stored
// in the high half of ds.  ncols must be a multiple of 32.
__global__ void __launch_bounds__(128) kQuantizePTQ1KsSoA(
        const float *__restrict__ act, int32_t *__restrict__ words,
        uint32_t nchunks) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t chunk = blockIdx.x * 4u + warp;
    if (chunk >= nchunks) return;
    const uint32_t col = chunk * 32u + lane;
    const float xi = act[col];
    float amax = fabsf(xi);
#pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    const int q = amax == 0.0f
                          ? 0
                          : static_cast<int>(roundf(xi / scale));
    int isum = q;
#pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        isum += __shfl_xor_sync(0xFFFFFFFFu, isum, off);
    int32_t *dstWord = words + ptq1KsQ8Word(chunk, lane >> 2u);
    reinterpret_cast<int8_t *>(dstWord)[lane & 3u] =
            static_cast<int8_t>(q);
    if (lane == 0u) {
        const half2 ds = make_half2(__float2half(scale),
                                    __short_as_half((short) isum));
        words[ptq1KsQ8Word(chunk, 8u)] =
                *reinterpret_cast<const int32_t *>(&ds);
    }
}

// Fused Prism fold + SOA quantize for the native decode: one kernel
// per projection instead of kPrismFold + kQuantizePTQ1KsSoA.  The
// fold math is kPrismFoldToQ8's (GDN value-head permutation, per
// feature signs, optional silu*mul), then the SoA quantizer's
// amax/scale/isum per 32 columns, written through ptq1KsQ8Word.
__global__ void __launch_bounds__(1024) kPrismFoldQuantSoA(
        const float *__restrict__ v, int32_t *__restrict__ words,
        uint32_t n, const float *__restrict__ signs, uint32_t gdnHeadV,
        uint32_t gdnNk, uint32_t gdnRep,
        const float *__restrict__ mul, uint32_t ptLayout) {
    __shared__ float tile[kPrismBlock];
    const uint32_t base = blockIdx.x * kPrismBlock;
    const uint32_t tid = threadIdx.x;
    if (base >= n) return;
    const uint32_t have =
            (n - base < kPrismBlock) ? (n - base) : kPrismBlock;
    const bool perm = (gdnRep > 1u && gdnHeadV > 0u && gdnNk > 0u);
    for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x) {
        float val = 0.0f;
        if (i < have) {
            const uint32_t g = base + i;
            const uint32_t si =
                    perm ? prismGdnSrcIndex(g, gdnHeadV, gdnNk, gdnRep)
                         : g;
            val = v[si];
            if (mul != nullptr)
                val = (val / (1.0f + __expf(-val))) * mul[si];
            if (signs != nullptr) val *= signs[g];
        }
        tile[i] = val;
    }
    __syncthreads();
    prismFwhtShared(tile, kPrismBlock, tid, blockDim.x);
    __syncthreads();
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t warps = blockDim.x >> 5u;
    const uint32_t chunks = n / 32u;
    for (uint32_t c = warp; c < (kPrismBlock / 32u); c += warps) {
        const uint32_t chunk =
                blockIdx.x * (kPrismBlock / 32u) + c;
        if (chunk >= chunks) break;
        const uint32_t colInTile = c * 32u + lane;
        const float xi = (colInTile < have) ? tile[colInTile] : 0.0f;
        float amax = fabsf(xi);
#pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        const int q = amax == 0.0f
                              ? 0
                              : static_cast<int>(roundf(xi / scale));
        int isum = q;
#pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            isum += __shfl_xor_sync(0xFFFFFFFFu, isum, off);
        const uint32_t nblk = n / 128u;
        int32_t *dstWord =
                words + (ptLayout != 0u
                                 ? ptq1PtQ8Int(chunk, lane, nblk)
                                 : ptq1KsQ8Word(chunk, lane >> 2u));
        reinterpret_cast<int8_t *>(dstWord)[lane & 3u] =
                static_cast<int8_t>(q);
        if (lane == 0u) {
            const half2 ds = make_half2(__float2half(scale),
                                        __short_as_half((short) isum));
            words[ptLayout != 0u ? ptq1PtDsInt(chunk, nblk)
                                 : ptq1KsQ8Word(chunk, 8u)] =
                    *reinterpret_cast<const int32_t *>(&ds);
        }
    }
}

// Fork's ptq1_0_ks_trits5: five base-3 digits of each byte of P, digit
// t of all four bytes packed into trit[t].
__device__ __forceinline__ void ptq1KsTrits5(const uint32_t P,
                                             uint32_t (&trit)[5]) {
    const uint32_t vlo = __byte_perm(P, 0, 0x4140);
    const uint32_t vhi = __byte_perm(P, 0, 0x4342);
    uint32_t carry = 0u;
    const uint32_t pow3[5] = {3u, 9u, 27u, 81u, 243u};
#pragma unroll
    for (uint32_t t = 0; t < 5u; ++t) {
        const uint32_t next =
                __byte_perm(vlo * pow3[t], vhi * pow3[t], 0x7531);
        trit[t] = next - 3u * carry;
        carry = next;
    }
}

// Fork's ptq1_0_ks_block_dot: ROWS weight blocks (7 words each) against
// one activation block in registers (act[0..31] = q8 words, act[32..35]
// = (scale, isum) per 32-column chunk).  wScale = fp16 group scale,
// actDot = sum_k scale_k * (chunkDot_k - isum_k).
template<uint32_t ROWS>
__device__ __forceinline__ void ptq1KsBlockDot(
        const uint32_t (&w)[ROWS][7], const int (&act)[36],
        float (&wScale)[ROWS], float (&actDot)[ROWS]) {
    int chunkDot[ROWS][4];
#pragma unroll
    for (uint32_t i = 0; i < ROWS; ++i)
#pragma unroll
        for (uint32_t k = 0; k < 4u; ++k) chunkDot[i][k] = 0;
#pragma unroll
    for (uint32_t grp = 0; grp < 6u; ++grp) {
#pragma unroll
        for (uint32_t i = 0; i < ROWS; ++i) {
            uint32_t trit[5];
            ptq1KsTrits5(w[i][grp], trit);
#pragma unroll
            for (uint32_t t = 0; t < 5u; ++t) {
                const uint32_t e = grp < 4u
                                           ? 16u * t + 4u * grp
                                           : 80u + 8u * t + 4u * (grp - 4u);
                chunkDot[i][e >> 5] = __dp4a(
                        static_cast<int>(trit[t]),
                        act[(e >> 5) * 8u + ((e & 31u) >> 2u)],
                        chunkDot[i][e >> 5]);
            }
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < ROWS; ++i) {
        uint32_t packed =
                __byte_perm(w[i][6], 0, 0x4140);
#pragma unroll
        for (uint32_t t = 0; t < 4u; t += 2u) {
            const uint32_t lo = packed * 3u;
            packed = lo & 0x00FF00FFu;
            const uint32_t hi = packed * 3u;
            packed = hi & 0x00FF00FFu;
            const uint32_t q = __byte_perm(lo, hi, 0x7531);
            chunkDot[i][3] = __dp4a(
                    static_cast<int>(q), act[3u * 8u + 6u + t / 2u],
                    chunkDot[i][3]);
        }
        float dot = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 4u; ++k) {
            const int32_t dsBits = act[32u + k];
            const half2 ds =
                    *reinterpret_cast<const half2 *>(&dsBits);
            const int isum = static_cast<int>(
                    __half_as_short(__high2half(ds)));
            dot += __low2float(ds) *
                   static_cast<float>(chunkDot[i][k] - isum);
        }
        wScale[i] = __half2float(__ushort_as_half(
                static_cast<unsigned short>(w[i][6] >> 16)));
        actDot[i] = dot;
    }
}

// Fork's mul_mat_vec_ptq1_0_ks, plain one-column path (no gate fusion,
// no biases, no scales -- the engine applies those outside the GEMV).
template<uint32_t G, uint32_t RPT, uint32_t ROWS, uint32_t LB,
         uint32_t MINB>
__global__ void __launch_bounds__(LB, MINB) kQGemvPTQ1_0Stationary(
        const uint8_t *__restrict__ weights,
        const int32_t *__restrict__ actWords, float *__restrict__ out,
        uint32_t ncols, uint32_t nrows, uint32_t rowBytes,
        uint32_t bpr, bool smallKOrder) {
    constexpr uint32_t R = G * RPT;
    extern __shared__ float2 partial[];// [R][bpr]
    const uint32_t row0 = R * blockIdx.x;
    const uint32_t nr = min(R, nrows - row0);
    const uint32_t rowGroup = threadIdx.x / bpr;
    const uint32_t kBlock = threadIdx.x - rowGroup * bpr;
    ptq1KsPdlSync();
    if (rowGroup < G) {
        int act[36];
        const int32_t *actCol = actWords + (kBlock >> 5) * kPtq1KsGroupWords +
                                (kBlock & 31u);
#pragma unroll
        for (uint32_t k = 0; k < 36u; ++k)
            act[k] = actCol[k * kPtq1KsGroupKb];
#pragma unroll
        for (uint32_t j = 0; j < RPT; j += ROWS) {
            uint32_t w[ROWS][7];
#pragma unroll
            for (uint32_t i = 0; i < ROWS; ++i) {
                const uint32_t r = min(rowGroup + G * (j + i), nr - 1u);
                const uint32_t *p = reinterpret_cast<const uint32_t *>(
                        weights + static_cast<uint64_t>(row0 + r) * rowBytes +
                        static_cast<uint64_t>(kBlock) * 28u);
#pragma unroll
                for (uint32_t k = 0; k < 7u; ++k) w[i][k] = p[k];
            }
            float wScale[ROWS];
            float actDot[ROWS];
            ptq1KsBlockDot<ROWS>(w, act, wScale, actDot);
#pragma unroll
            for (uint32_t i = 0; i < ROWS; ++i) {
                const uint32_t r = rowGroup + G * (j + i);
                if (r < nr)
                    partial[r * bpr + kBlock] =
                            make_float2(wScale[i], actDot[i]);
            }
        }
    }
    __syncthreads();
    const uint32_t span = smallKOrder ? 32u : 128u;
    const uint32_t warp = threadIdx.x / 32u;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t r = warp; r < nr; r += blockDim.x / 32u) {
        float sum = 0.0f;
        const float2 *p = partial + r * bpr;
        for (uint32_t k0 = 0; k0 < span; k0 += 32u) {
            float a = 0.0f;
            for (uint32_t k = k0 + lane; k < bpr; k += span) {
                const float2 v = p[k];
                a += v.x * v.y;
            }
            sum = k0 == 0u ? a : sum + a;
        }
#pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            sum += __shfl_down_sync(0xFFFFFFFFu, sum, off);
        if (lane == 0u) out[row0 + r] = sum;
    }
}

// ------------------------------------------------------------------
// Generic mul_mat_vec_q PTQ1_0 one-column path, ported from the
// Prism fork (ggml-cuda/mmvq.cu).  On sm_75 the fork runs this
// kernel, not the Hopper-only k-block-stationary one.  Two
// geometries, chosen by the fork's small-k rule
// (blocks_per_row < nwarps * 32, nwarps = 4):
//   bpr < 128  -> warp-per-row: warp w owns row row0+w, lanes stride
//                 the K blocks (no shared memory, shuffle reduce)
//   bpr >= 128 -> one row per CTA, 128 threads cover the blocks,
//                 shared tree reduce
// Both read the SOA_ISUM grouped activation written by
// kQuantizePTQ1KsSoA.
// ------------------------------------------------------------------
template<uint32_t NW>
__global__ void __launch_bounds__(NW * 32u) kQGemvPTQ1_0WarpRow(
        const uint8_t *__restrict__ weights,
        const int32_t *__restrict__ actWords, float *__restrict__ out,
        uint32_t ncols, uint32_t nrows, uint32_t rowBytes,
        uint32_t bpr) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t row = blockIdx.x * NW + warp;
    if (row >= nrows) return;
    float acc = 0.0f;
    const uint8_t *rowBase = weights + static_cast<uint64_t>(row) * rowBytes;
    for (uint32_t kb = lane; kb < bpr; kb += 32u) {
        int act[36];
        const int32_t *actCol = actWords + (kb >> 5) * kPtq1KsGroupWords +
                                (kb & 31u);
#pragma unroll
        for (uint32_t k = 0; k < 36u; ++k)
            act[k] = actCol[k * kPtq1KsGroupKb];
        uint32_t wr[1][7];
        const uint32_t *p = reinterpret_cast<const uint32_t *>(
                rowBase + static_cast<uint64_t>(kb) * 28u);
#pragma unroll
        for (uint32_t k = 0; k < 7u; ++k) wr[0][k] = p[k];
        float wScale[1];
        float actDot[1];
        ptq1KsBlockDot<1u>(wr, act, wScale, actDot);
        acc += wScale[0] * actDot[0];
    }
#pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0u) out[row] = acc;
}

__global__ void __launch_bounds__(128) kQGemvPTQ1_0CtaRow(
        const uint8_t *__restrict__ weights,
        const int32_t *__restrict__ actWords, float *__restrict__ out,
        uint32_t ncols, uint32_t nrows, uint32_t rowBytes,
        uint32_t bpr) {
    __shared__ float partial[128];
    const uint32_t tid = threadIdx.x;
    const uint32_t row = blockIdx.x;
    const uint8_t *rowBase = weights + static_cast<uint64_t>(row) * rowBytes;
    float acc = 0.0f;
    for (uint32_t kb = tid; kb < bpr; kb += 128u) {
        int act[36];
        const int32_t *actCol = actWords + (kb >> 5) * kPtq1KsGroupWords +
                                (kb & 31u);
#pragma unroll
        for (uint32_t k = 0; k < 36u; ++k)
            act[k] = actCol[k * kPtq1KsGroupKb];
        uint32_t wr[1][7];
        const uint32_t *p = reinterpret_cast<const uint32_t *>(
                rowBase + static_cast<uint64_t>(kb) * 28u);
#pragma unroll
        for (uint32_t k = 0; k < 7u; ++k) wr[0][k] = p[k];
        float wScale[1];
        float actDot[1];
        ptq1KsBlockDot<1u>(wr, act, wScale, actDot);
        acc += wScale[0] * actDot[0];
    }
    partial[tid] = acc;
    __syncthreads();
#pragma unroll
    for (uint32_t span = 64u; span > 0u; span >>= 1u) {
        if (tid < span) partial[tid] += partial[tid + span];
        __syncthreads();
    }
    if (tid == 0u) out[row] = partial[0];
}

// Warp-per-row against the PT activation layout: nine int4 loads per
// 128-weight block instead of 36 strided 4-byte words, with the warp's
// 32 lanes covering 512 contiguous bytes per load.  Same dot product
// as kQGemvPTQ1_0WarpRow, only the activation fetch differs.
template<uint32_t NW>
// (128, 8): the decoded 36-int activation window plus the weight words put this
// kernel at 90 registers, i.e. 5 blocks and 20 of 48 warps per SM -- 41% occupancy on
// a kernel that is pure weight streaming.  Eight blocks needs <= 64 registers.
__global__ void __launch_bounds__(NW * 32u, 8u) kQGemvPTQ1_0WarpRowPt(
        const uint8_t *__restrict__ weights,
        const int32_t *__restrict__ actWords, float *__restrict__ out,
        uint32_t ncols, uint32_t nrows, uint32_t rowBytes,
        uint32_t bpr) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x * NW + warp;
    if (row >= nrows) return;
    const uint8_t *rowBase =
            weights + static_cast<uint64_t>(row) * rowBytes;
    const int4 *act4 = reinterpret_cast<const int4 *>(actWords);
    float acc = 0.0f;
    for (uint32_t kb = lane; kb < bpr; kb += 32u) {
        int4 piece[9];
#pragma unroll
        for (uint32_t t = 0; t < 8u; ++t) piece[t] = act4[t * bpr + kb];
        piece[8] = act4[8u * bpr + kb];
        int act[36];
#pragma unroll
        for (uint32_t t = 0; t < 8u; ++t) {
            act[4u * t + 0u] = piece[t].x;
            act[4u * t + 1u] = piece[t].y;
            act[4u * t + 2u] = piece[t].z;
            act[4u * t + 3u] = piece[t].w;
        }
        act[32] = piece[8].x;
        act[33] = piece[8].y;
        act[34] = piece[8].z;
        act[35] = piece[8].w;
        uint32_t wr[1][7];
        const uint32_t *p = reinterpret_cast<const uint32_t *>(
                rowBase + static_cast<uint64_t>(kb) * 28u);
#pragma unroll
        for (uint32_t k = 0; k < 7u; ++k) wr[0][k] = p[k];
        float wScale[1];
        float actDot[1];
        ptq1KsBlockDot<1u>(wr, act, wScale, actDot);
        acc += wScale[0] * actDot[0];
    }
#pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0u) out[row] = acc;
}

// Warp-per-row-group variant: one warp owns RPT rows and each lane        // Warp-per-row-group variant: one warp owns RPT rows and each lane
// strides the K blocks, so the 36 activation words are loaded once for
// RPT rows (the fork's warp-per-row reloads them per row) and the
// reduction stays a shuffle, with no shared memory.
template<uint32_t NW, uint32_t RPT>
__global__ void __launch_bounds__(NW * 32u) kQGemvPTQ1_0WarpRows(
        const uint8_t *__restrict__ weights,
        const int32_t *__restrict__ actWords, float *__restrict__ out,
        uint32_t ncols, uint32_t nrows, uint32_t rowBytes,
        uint32_t bpr) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t row0 = (blockIdx.x * NW + warp) * RPT;
    float acc[RPT];
#pragma unroll
    for (uint32_t i = 0; i < RPT; ++i) acc[i] = 0.0f;
    for (uint32_t kb = lane; kb < bpr; kb += 32u) {
        int act[36];
        const int32_t *actCol = actWords + (kb >> 5) * kPtq1KsGroupWords +
                                (kb & 31u);
#pragma unroll
        for (uint32_t k = 0; k < 36u; ++k)
            act[k] = actCol[k * kPtq1KsGroupKb];
#pragma unroll
        for (uint32_t i = 0; i < RPT; ++i) {
            const uint32_t row = row0 + i;
            if (row >= nrows) continue;
            uint32_t wr[1][7];
            const uint32_t *p = reinterpret_cast<const uint32_t *>(
                    weights + static_cast<uint64_t>(row) * rowBytes +
                    static_cast<uint64_t>(kb) * 28u);
#pragma unroll
            for (uint32_t k = 0; k < 7u; ++k) wr[0][k] = p[k];
            float wScale[1];
            float actDot[1];
            ptq1KsBlockDot<1u>(wr, act, wScale, actDot);
            acc[i] += wScale[0] * actDot[0];
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < RPT; ++i) {
        float v = acc[i];
#pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            v += __shfl_down_sync(0xFFFFFFFFu, v, off);
        if (lane == 0u && row0 + i < nrows) out[row0 + i] = v;
    }
}

inline bool ptq1GenericEnabled() {
    static const bool on = [] {
        const char *e = std::getenv("TINYCODER_PT_GEN");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

// Fork's one-column geometry: small-k (bpr < nwarps*32) takes the
// warp-per-row kernel with 4 warps; otherwise one row per CTA with
// 128 threads.  Returns false only for shapes this table cannot run.
inline bool launchQGemvPTQ1_0GenericWords(
        const uint8_t *w, const int32_t *actWords, float *out,
        uint32_t rows, uint32_t cols, uint32_t rowBytes, uint32_t bpr) {
    if (bpr == 0u || (cols % 32u) != 0u || cols != bpr * 128u)
        return false;
    static const bool forceWarpRow = [] {
        const char *e = std::getenv("TINYCODER_PT_GEN_ALL");
        return e != nullptr && e[0] == '1';
    }();
    static const bool ptMode = [] {
        const char *e = std::getenv("TINYCODER_PT_PT");
        return e == nullptr || e[0] != '0';
    }();
    if (ptMode) {
        const uint32_t grid = (rows + 3u) / 4u;
        kQGemvPTQ1_0WarpRowPt<4u><<<grid, 128u, 0, g_stream>>>(
                w, actWords, out, cols, rows, rowBytes, bpr);
        return true;
    }
    if (bpr < 128u || forceWarpRow) {
        static const uint32_t rptSel = [] {
            const char *e = std::getenv("TINYCODER_PT_GEN_RPT");
            return e != nullptr && e[0] == '4'     ? 4u
                   : (e != nullptr && e[0] == '2') ? 2u
                                                   : 1u;
        }();
        if (rptSel == 4u) {
            const uint32_t grid = (rows + 15u) / 16u;
            kQGemvPTQ1_0WarpRows<4u, 4u><<<grid, 128u, 0, g_stream>>>(
                    w, actWords, out, cols, rows, rowBytes, bpr);
            return true;
        }
        if (rptSel == 2u) {
            const uint32_t grid = (rows + 7u) / 8u;
            kQGemvPTQ1_0WarpRows<4u, 2u><<<grid, 128u, 0, g_stream>>>(
                    w, actWords, out, cols, rows, rowBytes, bpr);
            return true;
        }
        static const uint32_t nwSel = [] {
            const char *e = std::getenv("TINYCODER_PT_GEN_NW");
            return e != nullptr && e[0] == '8' ? 8u : 4u;
        }();
        if (nwSel == 8u) {
            const uint32_t grid = (rows + 7u) / 8u;
            kQGemvPTQ1_0WarpRow<8u><<<grid, 256u, 0, g_stream>>>(
                    w, actWords, out, cols, rows, rowBytes, bpr);
        } else {
            const uint32_t grid = (rows + 3u) / 4u;
            kQGemvPTQ1_0WarpRow<4u><<<grid, 128u, 0, g_stream>>>(
                    w, actWords, out, cols, rows, rowBytes, bpr);
        }
    } else {
        kQGemvPTQ1_0CtaRow<<<rows, 128, 0, g_stream>>>(
                w, actWords, out, cols, rows, rowBytes, bpr);
    }
    return true;
}

// Quantize + GEMV (the original entry point): callers that already
// produced the SoA words (the fused fold+quantize kernel) use
// launchQGemvPTQ1_0GenericWords directly.
inline bool launchQGemvPTQ1_0Generic(
        const uint8_t *w, const float *act, int32_t *actWords,
        float *out, uint32_t rows, uint32_t cols, uint32_t rowBytes,
        uint32_t bpr) {
    if (bpr == 0u || (cols % 32u) != 0u || cols != bpr * 128u)
        return false;
    const uint32_t chunks = cols / 32u;
    kQuantizePTQ1KsSoA<<<(chunks + 3u) / 4u, 128, 0, g_stream>>>(
            act, actWords, chunks);
    return launchQGemvPTQ1_0GenericWords(w, actWords, out, rows, cols,
                                         rowBytes, bpr);
}

inline bool ptq1KsEnabled() {
    static const bool on = [] {
        const char *e = std::getenv("TINYCODER_PT_KS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

// Fork's per-shape launch table (plain 2D one-column calls):
//   bpr 33..63   -> G=4 RPT=2 ROWS=2 LB=256 MINB=4 small-k order
//   bpr 129..160 -> G=1 RPT=8 ROWS=2 LB=160 MINB=6, 4-warp order
// Returns false outside the table so the caller can fall back.
inline bool launchQGemvPTQ1_0Stationary(
        const uint8_t *w, const float *act, int32_t *actWords,
        float *out, uint32_t rows, uint32_t cols, uint32_t rowBytes,
        uint32_t bpr) {
    if (rows < 4096u || (cols % 32u) != 0u || bpr == 0u) return false;
    if (bpr <= 32u || bpr >= 64u) {
        if (bpr <= 128u || bpr > 160u) return false;
    }
    const uint32_t chunks = cols / 32u;
    kQuantizePTQ1KsSoA<<<(chunks + 3u) / 4u, 128, 0, g_stream>>>(
            act, actWords, chunks);
    const uint32_t R = (bpr > 128u) ? 8u : 8u;
    const uint32_t threads =
            (bpr > 128u) ? ((bpr + 31u) / 32u) * 32u
                         : ((4u * bpr + 31u) / 32u) * 32u;
    const uint32_t grid = (rows + R - 1u) / R;
    const size_t smem = static_cast<size_t>(R) * bpr * sizeof(float2);
    if (bpr > 128u) {
        kQGemvPTQ1_0Stationary<1u, 8u, 2u, 160u, 6u>
                <<<grid, threads, smem, g_stream>>>(
                        w, actWords, out, cols, rows, rowBytes, bpr,
                        false);
    } else {
        kQGemvPTQ1_0Stationary<4u, 2u, 2u, 256u, 4u>
                <<<grid, threads, smem, g_stream>>>(
                        w, actWords, out, cols, rows, rowBytes, bpr,
                        true);
    }
    return true;
}

inline bool ptq1SplitKEnabled() {
    static const bool on = [] {
        const char *e = std::getenv("TINYCODER_PT_SPLITK");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

inline bool ptq1Q8Enabled() {
    static const bool on = [] {
        const char *e = std::getenv("TINYCODER_PT_Q8");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

inline void launchQGemvPQ2_0Q8(const uint8_t *w, const int8_t *xq,
                               const float *xd, float *out,
                               uint32_t rows, uint32_t cols,
                               uint32_t rowBytes,
                               uint32_t blocksPerRow) {
    switch (pq2Q8Rows()) {
        case 1u:
            kQGemvPQ2_0Q8F<1u><<<(rows + 3u) / 4u, 128, 0, g_stream>>>(
                    w, xq, xd, out, rows, cols, rowBytes, blocksPerRow);
            break;
        case 2u:
            kQGemvPQ2_0Q8F<2u><<<(rows + 7u) / 8u, 128, 0, g_stream>>>(
                    w, xq, xd, out, rows, cols, rowBytes, blocksPerRow);
            break;
        case 8u:
            kQGemvPQ2_0Q8F<8u><<<(rows + 31u) / 32u, 128, 0, g_stream>>>(
                    w, xq, xd, out, rows, cols, rowBytes, blocksPerRow);
            break;
        default:
            kQGemvPQ2_0Q8F<4u><<<(rows + 15u) / 16u, 128, 0, g_stream>>>(
                    w, xq, xd, out, rows, cols, rowBytes, blocksPerRow);
            break;
    }
}

// Batched forward Prism transform, fp32 out: identical math to
// kPrismFold (sign flip, optional GDN value-head permutation, blockwise
// normalized Hadamard), one row per blockIdx.y.  Used by the int8
// tensor-core prefill spike, which quantises the activation itself.
__global__ void kPrismFoldB(const float *__restrict__ v,
                            float *__restrict__ dst, uint32_t n,
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
    float *dp = dst + static_cast<size_t>(s) * rowStride;
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
        dp[base + i] = tile[i];
    }
}

// PQ2_0 packed ternary -> fp16 twin, for the batched prefill GEMM.
// block = 256 (8 warps); each warp expands one 34-byte block: lane l owns
// columns [4l, 4l+4) of the 128-wide group, value = d * (code - 1), which
// is exact in fp16 (ternary times the fp16 group scale).  grid =
// (rows, ceil(blocksPerRow / 8)).
__global__ void kDequantPQ2_0ToF16(const uint8_t *__restrict__ w,
                                   __half *__restrict__ out, uint32_t rows,
                                   uint32_t cols, uint32_t rowBytes,
                                   uint32_t blocksPerRow) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps = blockDim.x >> 5;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    __half *op = out + static_cast<uint64_t>(row) * cols;
    // One warp per 128-block, 8-byte (four half) stores.  The previous
    // version launched one block per (row, 8 blocks) and had each warp
    // expand exactly one block: 87,040 blocks of 256 threads for a
    // 17408x5120 matrix, measured 150 GB/s of write+read traffic.  Here
    // one block owns a whole row (all blocks of it), which is the same
    // work with 1/5 the blocks and 8x the work per warp.
    for (uint32_t b = warp; b < blocksPerRow; b += warps) {
        const uint8_t *blk =
                rp + static_cast<uint64_t>(b) * kPQ2_0_BYTES;
        const float d =
                __half2float(*reinterpret_cast<const __half *>(blk));
        const uint8_t qb = blk[2u + lane];
        const uint32_t base = b * 128u + lane * 4u;
        const __half2 h01 = __floats2half2_rn(
                d * (static_cast<float>(qb & 3u) - 1.0f),
                d * (static_cast<float>((qb >> 2u) & 3u) - 1.0f));
        const __half2 h23 = __floats2half2_rn(
                d * (static_cast<float>((qb >> 4u) & 3u) - 1.0f),
                d * (static_cast<float>((qb >> 6u) & 3u) - 1.0f));
        if (base + 3u < cols) {
            uint2 v;
            v.x = *reinterpret_cast<const uint32_t *>(&h01);
            v.y = *reinterpret_cast<const uint32_t *>(&h23);
            *reinterpret_cast<uint2 *>(op + base) = v;
        } else {
            const float vv[4] = {d * (static_cast<float>(qb & 3u) - 1.0f),
                                 d * (static_cast<float>((qb >> 2u) & 3u) -
                                      1.0f),
                                 d * (static_cast<float>((qb >> 4u) & 3u) -
                                      1.0f),
                                 d * (static_cast<float>((qb >> 6u) & 3u) -
                                      1.0f)};
            for (uint32_t j = 0; j < 4u; ++j) {
                if (base + j < cols) {
                    op[base + j] = __float2half(vv[j]);
                }
            }
        }
    }
}

// PQ2_0 packed ternary -> GGML Q8_0 block layout (34 B per 32 weights:
// fp16 d + 32 int8).  Each PQ2_0 128-weight block becomes four Q8_0
// blocks that all carry the block's fp16 scale, and the int8 payload is
// the code itself minus one -- i.e. exactly the value the ternary block
// represents, so d32 * q8 == d * (code - 1) is exact.  This is the
// weight side of the P1 spike: it lets the engine's existing int8
// tensor-core GEMM (mma.s8.m8n8k16, stream-k) consume PQ2_0 weights, so
// the IMMA rate can be measured on the real shapes in situ.
// grid = rows, block = 256; one warp per 128-weight block.
__global__ void kDequantPQ2_0ToQ8_0(const uint8_t *__restrict__ w,
                                    uint8_t *__restrict__ out,
                                    uint32_t rows, uint32_t cols,
                                    uint32_t rowBytes,
                                    uint32_t blocksPerRow) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps = blockDim.x >> 5;
    const uint8_t *rp = w + static_cast<uint64_t>(row) * rowBytes;
    // one Q8_0 block per 32 columns
    uint8_t *op = out + static_cast<uint64_t>(row) *
                                (static_cast<uint64_t>(cols / 32u) * 34u);
    for (uint32_t b = warp; b < blocksPerRow; b += warps) {
        const uint8_t *blk =
                rp + static_cast<uint64_t>(b) * kPQ2_0_BYTES;
        const uint16_t d = *reinterpret_cast<const uint16_t *>(blk);
        const uint8_t qb = blk[2u + lane];
        // this lane's four codes are columns 4*lane..4*lane+3 of the
        // 128-column block, hence Q8_0 block (lane >> 3) at offset
        // (4*lane) & 31
        uint8_t *dst = op + static_cast<uint64_t>(b * 4u + (lane >> 3)) *
                                    34u;
        dst[0] = static_cast<uint8_t>(d & 0xFFu);
        dst[1] = static_cast<uint8_t>(d >> 8);
        int8_t *qs = reinterpret_cast<int8_t *>(dst + 2u);
        const uint32_t off = (lane & 7u) * 4u;
        qs[off + 0] = static_cast<int8_t>(static_cast<int>(qb & 3u) - 1);
        qs[off + 1] =
                static_cast<int8_t>(static_cast<int>((qb >> 2u) & 3u) - 1);
        qs[off + 2] =
                static_cast<int8_t>(static_cast<int>((qb >> 4u) & 3u) - 1);
        qs[off + 3] =
                static_cast<int8_t>(static_cast<int>((qb >> 6u) & 3u) - 1);
    }
}

// PQ2_0 token-embedding dequant + INVERSE fold (S then H -> rotate then
// sign).  grid = seqLen tokens; block = 256.  Dequantizes the ternary
// row into `hidden`, then applies the blockwise FWHT + signs in place so
// the latent (rotated) embedding row is restored to the primal basis.
__global__ void kEmbedDequantPQ2_0(
        const uint8_t *__restrict__ embed, const int32_t *__restrict__ tokens,
        float *__restrict__ hidden, uint32_t seqLen, uint32_t hiddenSize,
        uint32_t blocksPerRow, uint32_t rowBytes, uint32_t vocabSize,
        const float *__restrict__ signs) {
    const uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    const int32_t t = tokens[s];
    if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
    const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
    float *h = hidden + static_cast<size_t>(s) * hiddenSize;
    __shared__ float tile[kPrismBlock];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    for (uint32_t b = warp; b < blocksPerRow; b += (blockDim.x >> 5)) {
        const uint8_t *blk =
                row + static_cast<uint64_t>(b) * kPQ2_0_BYTES;
        const float d =
                __half2float(*reinterpret_cast<const __half *>(blk));
        const uint8_t qb = blk[2u + lane];
        const uint32_t base = b * 128u + lane * 4u;
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t c = base + j;
            if (c < hiddenSize)
                h[c] = d * (static_cast<float>((qb >> (j * 2u)) & 3u) -
                            1.0f);
        }
    }
    __syncthreads();
    for (uint32_t base = 0; base + kPrismBlock <= hiddenSize;
         base += kPrismBlock) {
        for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x)
            tile[i] = h[base + i];
        __syncthreads();
        prismFwhtShared(tile, kPrismBlock, tid, blockDim.x);
        for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x) {
            float val = tile[i];
            if (signs != nullptr) val *= signs[base + i];
            h[base + i] = val;
        }
        __syncthreads();
    }
}

// Native PTQ1_0 token-embedding dequant + inverse fold.  Lane p owns
// the four columns 4p..4p+3 of one 128-weight block, exactly the int
// that ptq1DecodeInt returns; the rest (FWHT, signs) is identical.
__global__ void kEmbedDequantPTQ1_0(
        const uint8_t *__restrict__ embed,
        const int32_t *__restrict__ tokens, float *__restrict__ hidden,
        uint32_t seqLen, uint32_t hiddenSize, uint32_t blocksPerRow,
        uint32_t rowBytes, uint32_t vocabSize,
        const float *__restrict__ signs) {
    const uint32_t s = blockIdx.x;
    if (s >= seqLen) return;
    const int32_t t = tokens[s];
    if (t < 0 || static_cast<uint32_t>(t) >= vocabSize) return;
    const uint8_t *row = embed + static_cast<uint64_t>(t) * rowBytes;
    float *h = hidden + static_cast<size_t>(s) * hiddenSize;
    __shared__ float tile[kPrismBlock];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t warps = blockDim.x >> 5;
    for (uint32_t b = warp; b < blocksPerRow; b += warps) {
        const uint8_t *blk =
                row + static_cast<uint64_t>(b) * kPTQ1_0_BYTES;
        const float d =
                __half2float(*reinterpret_cast<const __half *>(blk + 26u));
        const uint32_t q = ptq1DecodeInt(blk, lane);
        const uint32_t base = b * 128u + lane * 4u;
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t c = base + j;
            if (c < hiddenSize)
                h[c] = d * static_cast<float>(static_cast<int8_t>(
                                   (q >> (j * 8u)) & 0xFFu));
        }
    }
    __syncthreads();
    for (uint32_t base = 0; base + kPrismBlock <= hiddenSize;
         base += kPrismBlock) {
        for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x)
            tile[i] = h[base + i];
        __syncthreads();
        prismFwhtShared(tile, kPrismBlock, tid, blockDim.x);
        for (uint32_t i = tid; i < kPrismBlock; i += blockDim.x) {
            float val = tile[i];
            if (signs != nullptr) val *= signs[base + i];
            h[base + i] = val;
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------
// Q8_K activation quantization + integer vec-dot GEMV (decode path)
// for the IQ2_S / IQ3_XXS / IQ3_S weight types.
//
// ------------------------------------------------------------------

// Q8_K per-block stride: int8 qs[256] + int16 bsums[16] + float d,
// 16-byte aligned (float at 288).
static constexpr uint32_t kQ8K_Q = 0;
static constexpr uint32_t kQ8K_B = 256;
static constexpr uint32_t kQ8K_D = 256 + 32;
static constexpr uint32_t kQ8K_STRIDE = 256 + 32 + 16;

// Quantize one fp32 activation vector into Q8_K blocks in q8k.
// Bit-exact with GGMLDequantize::quantizeQ8K (nearest-int rintf,
// clamp to +-127, d = -max/vscale, bsums per 16-group).
// grid = ceil(cols/256), block = 256 threads.
__global__ void kQuantizeQ8K(const float *__restrict__ x,
                             uint8_t *__restrict__ q8k,
                             uint32_t cols) {
    const uint32_t b = blockIdx.x;
    const uint32_t e0 = b * 256u;
    const uint32_t n = (cols - e0) < 256u ? (cols - e0) : 256u;
    if (n == 0) return;
    uint8_t *blk = q8k + static_cast<uint64_t>(b) * kQ8K_STRIDE;
    float maxv = 0.0f;
    float amax = 0.0f;
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        float v = x[e0 + j];
        float av = fabsf(v);
        if (av > amax) {
            amax = av;
            maxv = v;
        }
    }
    // reduce the signed max (signed-ness travels with the abs-max).
    // NOTE: NO divergent early-exit here -- every thread must reach the
    // same __syncthreads() count or the block's barrier is undefined.
    // An all-zero block simply yields iscale/d = 0 (bsums stay 0).
    __shared__ float sMax[256];
    sMax[threadIdx.x] = (amax > 0.0f) ? maxv : 0.0f;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float a0 = sMax[threadIdx.x], a1 = sMax[threadIdx.x + s];
            if (fabsf(a1) > fabsf(a0)) sMax[threadIdx.x] = a1;
        }
        __syncthreads();
    }
    const float max = sMax[0];
    // __fdiv_rn forces the two IEEE correctly-rounded divisions
    // (-127/max then 1/iscale) that the CPU's quantizeQ8K emits as two
    // divss.  A plain '/' lets nvcc algebraically contract
    // 1/(-127/max) -> -max/127, which rounds differently (measured:
    // exact 1-ulp drift on the stored d, corrupting every downstream
    // Q4K/Q6K x Q8_K dot by 1 ulp even though qs/bsums stay bit-exact).
    const float iscale = (max != 0.0f) ? __fdiv_rn(-127.0f, max) : 0.0f;
    const float d = (max != 0.0f) ? __fdiv_rn(1.0f, iscale) : 0.0f;
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        if (max == 0.0f) {
            blk[kQ8K_Q + j] = 0;
            continue;
        }
        int v = static_cast<int>(rintf(iscale * x[e0 + j]));
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        blk[kQ8K_Q + j] = static_cast<uint8_t>(static_cast<int8_t>(v));
    }
    if (threadIdx.x == 0) {
        *reinterpret_cast<float *>(blk + kQ8K_D) = d;
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < 16; j += blockDim.x) {
        int sum = 0;
        for (uint32_t ii = 0; ii < 16; ++ii) {
            sum += static_cast<int>(
                    static_cast<int8_t>(blk[kQ8K_Q + j * 16u + ii]));
        }
        *reinterpret_cast<int16_t *>(blk + kQ8K_B + j * 2) =
                static_cast<int16_t>(sum);
    }
}

// ==================================================================
// int8 tensor-core INT8 (mma.s8.m8n8k16) for the Int8Pack format
// ==================================================================
// Target: Turing (sm_75) provides int8 tensor cores ONLY for the
// m8n8k16 / m8n8k32 shapes; mma.m16n8k16 for int8 requires sm_80+.  The
// pack (include/Int8Pack.hpp) therefore targets m8n8k16, whose B
// fragment (b_i = B[(lane%4)*4+i][lane>>2]) is IDENTICAL to the
// m16n8k16 B fragment, so the same packed weights feed both shapes.
//
// GEMM: y[seq][rows] = A[seq][cols] @ W[rows][cols]^T with the
// 256-wide Q8_K activation (kQuantizeQ8K) and the exact IQ3_XXS
// epilogue 4 * actd * sum_g db_g * sum_int_g (matches
// Int8Pack::dotBlockQ8K / dotProductIQ3_XXS_Q8K).  decode (seq==1) runs
// the SAME kernel -- the identical mma family, just one real token.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
#define TINYCODER_MMA_S8 1
#endif

// D += A @ B (m8n8k16, s8 x s8 -> s32).  a: 1 b32/lane (4 int8),
// b: 1 b32/lane (4 int8), c: 2 s32/lane.  Guarded so sm_80/sm_90 also
// compile (the shape is valid on every arch >= sm_75).
__device__ __forceinline__ void mmaS8_8x8x16(int32_t c[2], uint32_t a,
                                             uint32_t b) {
#ifdef TINYCODER_MMA_S8
    asm volatile(
            "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
            "{%0, %1}, {%2}, {%3}, {%0, %1};\n"
            : "+r"(c[0]), "+r"(c[1])
            : "r"(a), "r"(b));
#else
    (void) c;
    (void) a;
    (void) b;
#endif
}
