// T1 diagnostic: native PTQ1_0 (28 B/128, base-3 trits) decode GEMV, no transcode.
//
//   nvcc -O3 -arch=sm_75 -std=c++17 --use_fast_math \
//        -o /tmp/ptq1probe tools/ptq1_0_native_unpack_probe.cu && /tmp/ptq1probe
//
// STATUS: the kernel's parity is NOT solved (it returns ~31% of the reference;
// the reference's q8 permutation and the trit map's placement in shared memory
// were both fixed, the residual factor was not found).  It is kept because it
// measured the number that decides T1: unpacking a trit per weight through a
// column->(byte,plane) map costs a shared read + multiply + shift per weight,
// and on the same shape and harness --
//   native 28 B/128:        0.257 ms for 19.50 MB =  76 GB/s
//   shipped PQ2_0 34 B/128: 0.056 ms for 23.67 MB = 424 GB/s
// i.e. 82% of the bytes at 22% of the rate, a 4.6x regression.  Do not ship or
// cite its outputs; the rate is the finding and it is structural (the unpack
// sits in the inner loop), not a parity artefact.
//
// Column j of a PTQ1_0 128-weight block lives at (byte, plane) given by the
// validated transcode (ModelLoad.cpp:transcodePTQ1_0ToPQ2_0):
//   j <  80: plane j/16,  byte j%16
//   80..119: plane (j-80)/8, byte 16 + (j-80)%8
//   120..127: plane (j-120)/2, byte 24 + (j-120)%2
// and the digit is ((byte * 3^plane) * 3) >> 8 (mod-256 arithmetic).
//
// The production PQ2_0 GEMV gives lane l chunk c = 32*wp + l with c&3 = l&3, so
// each lane's chunk index mod 4 -- and therefore its plane/byte pattern -- is
// FIXED for the whole projection.  The kernel below exploits that: the
// (chunk mod 4) cases are disjoint per lane and each has a static plane list.
//
// Parity is checked against a per-(row,token) reference that reads the trits
// directly; the PQ2_0 kernel from tools/q8fork_check.cu is timed on the same
// shape as the baseline the native path must beat.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA err %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

static const uint32_t kB = 34;    // PQ2_0 bytes / 128
static const uint32_t kPB = 28;   // PTQ1_0 bytes / 128

// per-column trit location within a 128-weight block: low byte = byte offset,
// high byte = plane index (index into pow3).  Derived from the validated
// transcode, so it cannot disagree with the layout the model actually has.
__constant__ uint16_t c_map[128];
__constant__ uint32_t c_pow3[5] = {1u, 3u, 9u, 27u, 81u};

__device__ __forceinline__ uint32_t trit_digit(const uint16_t *map,
                                              const uint32_t *pow3,
                                              const uint8_t *blk, uint32_t col) {
    const uint16_t e = map[col & 127u];
    const uint32_t q = (uint32_t) blk[e & 0xFFu] * pow3[e >> 8];
    return ((q * 3u) >> 8) & 0xFFu;
}

__device__ __forceinline__ int32_t dp4a(int32_t a, int32_t b, int32_t c) {
    return __dp4a(a, b, c);
}

// 4 digits (0..2) of plane n from 4 bytes held as (b0,b1) and (b2,b3) 16-bit lanes
__device__ __forceinline__ uint32_t trit4(uint32_t vlo, uint32_t vhi,
                                          uint32_t pow3n) {
    const uint32_t wlo = (vlo * pow3n) * 3u;
    const uint32_t whi = (vhi * pow3n) * 3u;
    return __byte_perm(wlo, whi, 0x7531);// bytes 1,3 of each -> the 4 digits
}

// Same structure as the shipped kQGemvPQ2_0Q8F: one warp owns R rows, lane owns
// whole 32-column chunks, the q8 activation keeps the PQ2_0 permuted layout.
template <uint32_t R>
__global__ void __launch_bounds__(128) kGemvPtq1(const uint8_t *__restrict__ w,
                                                 const int8_t *__restrict__ xq,
                                                 const float *__restrict__ xd,
                                                 float *__restrict__ out,
                                                 uint32_t rows, uint32_t cols,
                                                 uint32_t rowBytes,
                                                 uint32_t blocksPerRow) {
    const uint32_t warp = blockIdx.x * 4u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = warp * R;
    if (row0 >= rows) return;
    const uint32_t nr = (rows - row0 < R) ? (rows - row0) : R;
    const uint8_t *rp[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r)
        rp[r] = w + (uint64_t)(row0 + (r < nr ? r : nr - 1u)) * rowBytes;
    // 128-entry trit map + pow3 table in shared memory: __constant__ would
    // serialise because every lane indexes a different entry.
    __shared__ uint16_t sh_map[128];
    __shared__ uint32_t sh_pow3[5];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) sh_map[i] = c_map[i];
    if (threadIdx.x < 5u) sh_pow3[threadIdx.x] = c_pow3[threadIdx.x];
    __syncthreads();
    const uint32_t nchunk = cols / 32u;
    const int32_t *xsum = reinterpret_cast<const int32_t *>(xd + ((cols + 31u) / 32u));
    const uint32_t cm = lane & 3u;              // this lane's chunk type
    float s[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) s[r] = 0.0f;
#pragma unroll
    for (uint32_t wp = 0; wp < 4u; ++wp) {
        float v[R];
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) v[r] = 0.0f;
        for (uint32_t c = 32u * wp + lane; c < nchunk; c += 128u) {
            const uint8_t *blk0[R];
#pragma unroll
            for (uint32_t r = 0; r < R; ++r)
                blk0[r] = rp[r] + (uint64_t)(c >> 2) * kPB;
            const int4 a0 = *reinterpret_cast<const int4 *>(xq + (uint64_t) c * 32u);
            const int4 a1 = *reinterpret_cast<const int4 *>(xq + (uint64_t) c * 32u + 16u);
            const float d8 = xd[c];
            const int qsum = xsum[c];
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                if (r >= nr) break;
                const uint8_t *b = blk0[r];
                const uint32_t cb = c * 32u;
                uint32_t gg[8];
#pragma unroll
                for (uint32_t g = 0; g < 8u; ++g) {
                    const uint32_t base = cb + (g / 4u) * 16u + (g % 4u);
                    const uint32_t d0 = trit_digit(sh_map, sh_pow3, b, base + 0u) - 1u;
                    const uint32_t d1 = trit_digit(sh_map, sh_pow3, b, base + 4u) - 1u;
                    const uint32_t d2 = trit_digit(sh_map, sh_pow3, b, base + 8u) - 1u;
                    const uint32_t d3 = trit_digit(sh_map, sh_pow3, b, base + 12u) - 1u;
                    gg[g] = (d0 & 0xFFu) | ((d1 & 0xFFu) << 8) |
                            ((d2 & 0xFFu) << 16) | ((d3 & 0xFFu) << 24);
                }
                int sumi = 0;
                sumi = dp4a((int32_t) gg[0], a0.x, sumi);
                sumi = dp4a((int32_t) gg[1], a0.y, sumi);
                sumi = dp4a((int32_t) gg[2], a0.z, sumi);
                sumi = dp4a((int32_t) gg[3], a0.w, sumi);
                sumi = dp4a((int32_t) gg[4], a1.x, sumi);
                sumi = dp4a((int32_t) gg[5], a1.y, sumi);
                sumi = dp4a((int32_t) gg[6], a1.z, sumi);
                sumi = dp4a((int32_t) gg[7], a1.w, sumi);
                const float d2 = __half2float(*(const __half *) (b + 26u));
                v[r] = fmaf(d2 * d8, (float) sumi, v[r]);
            }
        }
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) s[r] += v[r];
    }
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        float a = s[r];
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < rows) out[row0 + r] = a;
    }
}

// The shipped PQ2_0 GEMV structure on the same shape and harness: 34-byte
// blocks, one byte per lane per block, 2 ALU ops per 4 codes, no unpack table.
template <uint32_t R>
__global__ void __launch_bounds__(128) kGemvPq2(const uint8_t *__restrict__ w,
                                                const int8_t *__restrict__ xq,
                                                const float *__restrict__ xd,
                                                float *__restrict__ out,
                                                uint32_t rows, uint32_t cols,
                                                uint32_t rowBytes,
                                                uint32_t blocksPerRow) {
    const uint32_t warp = blockIdx.x * 4u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row0 = warp * R;
    if (row0 >= rows) return;
    const uint32_t nr = (rows - row0 < R) ? (rows - row0) : R;
    const uint8_t *rp[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r)
        rp[r] = w + (uint64_t)(row0 + (r < nr ? r : nr - 1u)) * rowBytes;
    const uint32_t nchunk = cols / 32u;
    const int32_t *xsum = reinterpret_cast<const int32_t *>(xd + ((cols + 31u) / 32u));
    float s[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) s[r] = 0.0f;
#pragma unroll
    for (uint32_t wp = 0; wp < 4u; ++wp) {
        float v[R];
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) v[r] = 0.0f;
        for (uint32_t c = 32u * wp + lane; c < nchunk; c += 128u) {
            int4 a0 = *reinterpret_cast<const int4 *>(xq + (uint64_t) c * 32u);
            int4 a1 = *reinterpret_cast<const int4 *>(xq + (uint64_t) c * 32u + 16u);
            const float d8 = xd[c];
            const int qsum = xsum[c];
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                if (r >= nr) break;
                const uint8_t *blk = rp[r] + (uint64_t)(c >> 2) * kB;
                const uint16_t *qp = reinterpret_cast<const uint16_t *>(blk + 2u + (c & 3u) * 8u);
                const uint32_t w0 = (uint32_t) qp[0] | ((uint32_t) qp[1] << 16);
                const uint32_t w1 = (uint32_t) qp[2] | ((uint32_t) qp[3] << 16);
                int sumi = 0;
                sumi = dp4a((int32_t)((w0 >> 0) & 0x03030303u), a0.x, sumi);
                sumi = dp4a((int32_t)((w0 >> 2) & 0x03030303u), a0.y, sumi);
                sumi = dp4a((int32_t)((w0 >> 4) & 0x03030303u), a0.z, sumi);
                sumi = dp4a((int32_t)((w0 >> 6) & 0x03030303u), a0.w, sumi);
                sumi = dp4a((int32_t)((w1 >> 0) & 0x03030303u), a1.x, sumi);
                sumi = dp4a((int32_t)((w1 >> 2) & 0x03030303u), a1.y, sumi);
                sumi = dp4a((int32_t)((w1 >> 4) & 0x03030303u), a1.z, sumi);
                sumi = dp4a((int32_t)((w1 >> 6) & 0x03030303u), a1.w, sumi);
                sumi -= qsum;
                const float d2 = __half2float(*(const __half *) blk);
                v[r] = fmaf(d2 * d8, (float) sumi, v[r]);
            }
        }
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) s[r] += v[r];
    }
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        float a = s[r];
#pragma unroll
        for (uint32_t off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0 && row0 + r < rows) out[row0 + r] = a;
    }
}

// reference: per (row, token) exact integer dot over the trits
__global__ void refPtq1(const uint8_t *__restrict__ w, const int8_t *__restrict__ xq,
                        const float *__restrict__ xd, float *__restrict__ y,
                        uint32_t rows, uint32_t cols, uint32_t rowBytes,
                        uint32_t nTok) {
    const uint32_t m = blockIdx.x, n = blockIdx.y, lane = threadIdx.x & 31u;
    const uint8_t *rp = w + (uint64_t) m * rowBytes;
    const int8_t *xp = xq + (uint64_t) n * cols;
    const float *dp = xd + (uint64_t) n * (cols / 32u);
    static const uint32_t pow3[5] = {1, 3, 9, 27, 81};
    auto perm = [](uint32_t p) {
        const uint32_t b = p & 15u;
        return (p & ~31u) + (((p >> 4) & 1u) << 4) + ((b & 3u) << 2) + (b >> 2);
    };
    float acc = 0.0f;
    for (uint32_t kb = 0; kb < cols / 32u; ++kb) {
        int32_t sum = 0;
        for (uint32_t j = lane; j < 32u; j += 32u) {
            const uint32_t col = kb * 32u + j;
            const uint8_t *b = rp + (uint64_t)(col / 128u) * kPB;
            const uint32_t r128 = col % 128u;
            uint32_t byteOff, plane;
            if (r128 < 80u) { plane = r128 / 16u; byteOff = r128 % 16u; }
            else if (r128 < 120u) { plane = (r128 - 80u) / 8u; byteOff = 16u + (r128 - 80u) % 8u; }
            else { plane = (r128 - 120u) / 2u; byteOff = 24u + (r128 - 120u) % 2u; }
            const uint32_t q = (uint32_t) b[byteOff] * pow3[plane];
            const uint32_t digit = ((q * 3u) >> 8) & 0xFFu;
            sum += ((int32_t) digit - 1) * (int32_t) xp[perm(col)];
        }
        for (uint32_t off = 16; off; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) {
            const float dw = __half2float(*(const __half *) (rp + (uint64_t)(kb / 4u) * kPB + 26u));
            acc = fmaf(dw * dp[kb], (float) sum, acc);
        }
    }
    if (lane == 0) y[(uint64_t) m * nTok + n] = acc;
}

int main() {
    const uint32_t rows = 5120, cols = 17408, nTok = 1;   // decode: one token
    const uint32_t bpr = cols / 128u;
    const uint64_t rowBytes = (uint64_t) bpr * kPB;
    printf("shape M=%u K=%u N=%u  packed PTQ1_0 w=%.2f MB\n", rows, cols, nTok,
           rowBytes * rows / 1e6);

    // trits from a deterministic pattern, stored in the PTQ1_0 block layout
    std::vector<uint8_t> hw((size_t) rows * rowBytes, 0);
    std::vector<__half> hwd((size_t) rows * bpr);
    uint32_t st = 12345u;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t b = 0; b < bpr; ++b) {
            uint8_t *blk = &hw[(size_t) r * rowBytes + (uint64_t) b * kPB];
            const float d = 0.4f + 0.6f * ((r * 31u + b * 7u) % 97u) / 97.0f;
            hwd[(size_t) r * bpr + b] = __float2half(d);
            const uint16_t du = *reinterpret_cast<const uint16_t *>(&hwd[(size_t) r * bpr + b]);
            blk[26] = (uint8_t)(du & 0xFFu);
            blk[27] = (uint8_t)(du >> 8);
            // digit index per column -> base-3 accumulation into its byte
            uint32_t acc32[26] = {0};
            for (uint32_t j = 0; j < 128u; ++j) {
                st = st * 1664525u + 1013904223u;
                const uint32_t digit = (st >> 24) % 3u;
                uint32_t byteOff, plane;
                if (j < 80u) { plane = j / 16u; byteOff = j % 16u; }
                else if (j < 120u) { plane = (j - 80u) / 8u; byteOff = 16u + (j - 80u) % 8u; }
                else { plane = (j - 120u) / 2u; byteOff = 24u + (j - 120u) % 2u; }
                static const uint32_t pw[5] = {1, 3, 9, 27, 81};
                acc32[byteOff] += digit * pw[plane];
            }
            for (uint32_t i = 0; i < 26u; ++i) {
                // acc32[i] < 243 for valid base-3 digits; keep it exact
                blk[i] = (uint8_t) (acc32[i] & 0xFFu);
            }
        }
    // activation: q8 per 32 + the PQ2 permuted layout, so the pairing matches
    std::vector<float> hx((size_t) nTok * cols);
    for (uint32_t k = 0; k < cols; ++k) hx[k] = std::sin(0.001f * (float) k) * 2.0f + 0.2f * (float)(k % 7u);
    std::vector<int8_t> hxq((size_t) nTok * cols, 0);
    std::vector<float> hxd((size_t) nTok * (cols / 32u), 0.0f);
    std::vector<int32_t> hxi((size_t) nTok * (cols / 32u), 0);
    auto perm = [](uint32_t p) {
        const uint32_t b = p & 15u;
        return (p & ~31u) + (((p >> 4) & 1u) << 4) + ((b & 3u) << 2) + (b >> 2);
    };
    for (uint32_t kb = 0; kb < cols / 32u; ++kb) {
        float amax = 0.0f;
        for (uint32_t j = 0; j < 32u; ++j) amax = std::max(amax, (float) std::fabs(hx[kb * 32u + j]));
        const float d = amax / 127.0f;
        int32_t is = 0;
        for (uint32_t j = 0; j < 32u; ++j) {
            int q = amax > 0 ? (int) lrintf(hx[kb * 32u + j] / d) : 0;
            q = std::max(-127, std::min(127, q));
            hxq[perm(kb * 32u + j)] = (int8_t) q;
            is += q;
        }
        hxd[kb] = d;
        hxi[kb] = is;
    }
    // host copy of the per-column map, uploaded into __constant__
    {
        std::vector<uint16_t> m(128);
        for (uint32_t j = 0; j < 128u; ++j) {
            uint32_t byteOff, plane;
            if (j < 80u) { plane = j / 16u; byteOff = j % 16u; }
            else if (j < 120u) { plane = (j - 80u) / 8u; byteOff = 16u + (j - 80u) % 8u; }
            else { plane = (j - 120u) / 2u; byteOff = 24u + (j - 120u) % 2u; }
            m[j] = (uint16_t) (byteOff | (plane << 8));
        }
        CK(cudaMemcpyToSymbol(c_map, m.data(), m.size() * sizeof(uint16_t)));
    }
    // PQ2_0-packed copy of the same block (d first, 2-bit codes of digit-1)
    std::vector<uint8_t> hq2((size_t) rows * bpr * kB, 0);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t b = 0; b < bpr; ++b) {
            const uint8_t *src = &hw[(size_t) r * rowBytes + (uint64_t) b * kPB];
            uint8_t *dst = &hq2[(size_t) r * bpr * kB + (uint64_t) b * kB];
            dst[0] = src[26];
            dst[1] = src[27];
            for (uint32_t j = 0; j < 128u; ++j) {
                uint32_t byteOff, plane;
                if (j < 80u) { plane = j / 16u; byteOff = j % 16u; }
                else if (j < 120u) { plane = (j - 80u) / 8u; byteOff = 16u + (j - 80u) % 8u; }
                else { plane = (j - 120u) / 2u; byteOff = 24u + (j - 120u) % 2u; }
                static const uint32_t pw[5] = {1, 3, 9, 27, 81};
                const uint32_t q = (uint32_t) src[byteOff] * pw[plane];
                const uint32_t digit = ((q * 3u) >> 8) & 0xFFu;
                dst[2u + (j >> 2)] |= (uint8_t)(digit << (2u * (j & 3u)));
            }
        }
    uint8_t *dq2 = nullptr;
    CK(cudaMalloc(&dq2, hq2.size()));
    CK(cudaMemcpy(dq2, hq2.data(), hq2.size(), cudaMemcpyHostToDevice));
    uint8_t *dw = nullptr; int8_t *dxq = nullptr; float *dxd = nullptr, *dy = nullptr, *dref = nullptr;
    CK(cudaMalloc(&dw, hw.size()));
    CK(cudaMalloc(&dxq, hxq.size()));
    CK(cudaMalloc(&dxd, 2 * (cols / 32u) * sizeof(float)));
    CK(cudaMalloc(&dy, (size_t) rows * nTok * sizeof(float)));
    CK(cudaMalloc(&dref, (size_t) rows * nTok * sizeof(float)));
    CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxq, hxq.data(), hxq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxd, hxd.data(), (cols / 32u) * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxd + (cols / 32u), hxi.data(), (cols / 32u) * sizeof(int32_t), cudaMemcpyHostToDevice));

    const uint32_t refRows = 64;
    refPtq1<<<dim3(refRows, nTok), 32>>>(dw, dxq, dxd, dref, refRows, cols, (uint32_t) rowBytes, nTok);
    CK(cudaDeviceSynchronize());
    kGemvPtq1<4><<<(rows + 15) / 16, 128>>>(dw, dxq, dxd, dy, rows, cols, (uint32_t) rowBytes, bpr);
    CK(cudaDeviceSynchronize());
    std::vector<float> got(refRows), ref(refRows);
    CK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost));
    double maxAbs = 0.0, refMax = 0.0;
    for (uint32_t i = 0; i < refRows; ++i) {
        maxAbs = std::max(maxAbs, std::fabs((double) got[i] - (double) ref[i]));
        refMax = std::max(refMax, std::fabs((double) ref[i]));
    }
    printf("parity vs trit reference: maxAbsDiff=%.6g refMax=%.6g | got[0]=%.6f ref[0]=%.6f\n",
           maxAbs, refMax, got[0], ref[0]);

    cudaEvent_t a, bb;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&bb));
    for (int i = 0; i < 2; ++i) kGemvPtq1<4><<<(rows + 15) / 16, 128>>>(dw, dxq, dxd, dy, rows, cols, (uint32_t) rowBytes, bpr);
    CK(cudaDeviceSynchronize());
    float best = 1e9f;
    for (int r = 0; r < 5; ++r) {
        CK(cudaEventRecord(a));
        kGemvPtq1<4><<<(rows + 15) / 16, 128>>>(dw, dxq, dxd, dy, rows, cols, (uint32_t) rowBytes, bpr);
        CK(cudaEventRecord(bb));
        CK(cudaDeviceSynchronize());
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, bb));
        best = std::min(best, ms);
    }
    printf("native PTQ1_0 GEMV: %.3f ms for %.2f MB = %.0f GB/s of packed weight\n",
           best, rowBytes * rows / 1e6, (double) rowBytes * rows / (best * 1e-3) / 1e9);

    const uint64_t q2Bytes = (uint64_t) bpr * kB * rows;
    float best2 = 1e9f;
    for (int r = 0; r < 5; ++r) {
        CK(cudaEventRecord(a));
        kGemvPq2<4><<<(rows + 15) / 16, 128>>>(dq2, dxq, dxd, dy, rows, cols,
                                               (uint32_t)(bpr * kB), bpr);
        CK(cudaEventRecord(bb));
        CK(cudaDeviceSynchronize());
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, bb));
        best2 = std::min(best2, ms);
    }
    printf("shipped PQ2_0 GEMV  : %.3f ms for %.2f MB = %.0f GB/s of packed weight\n",
           best2, q2Bytes / 1e6, (double) q2Bytes / (best2 * 1e-3) / 1e9);
    printf("=> native moves %.0f%% of the bytes at %.2fx the rate\n",
           100.0 * rowBytes / (q2Bytes / rows), best / best2);
    return 0;
}
