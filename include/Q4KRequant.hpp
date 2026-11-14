// TinyCoder — Q8_0 -> Q4_K requantisation helpers.
//
// The MoE expert blobs of a large checkpoint (Ornith-35B-A3B `q8_0`, 34.4 GB)
// cannot be held as 3.34 MB per-expert slots on an 11 GB card: the expert
// cache allocation failed and the whole model silently fell back to CPU
// experts.  Requantising the experts to Q4_K at load halves the bytes *and*
// routes them through the grouped int8 tensor-core prefill path (which needs
// Q4_K/Q6_K).  This is a deliberate quality trade (4-bit experts), the same
// one TINYCODER_Q8_Q4K already makes for dense weights.
#pragma once

#include "GGMLDequantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace q4k {

    // Quantize 256 floats into one compact Q4_K block (144 B): d, dmin,
    // 12 packed 6-bit scale/min bytes, 128 nibble bytes.  Affine per-32
    // groups with a non-positive min so the Q4_K min term stays >= 0.
    // (Same routine as the dense Q6_K/Q8_0 requantiser in ModelLoad.cpp.)
    inline void quantizeBlock(const float *x, uint8_t *out) {
        float scale[8], minv[8];
        uint8_t q[256];
        float maxScale = 0.0f, maxMin = 0.0f;
        for (uint32_t g = 0; g < 8; ++g) {
            const float *xp = x + g * 32;
            float mn = 0.0f, mx = 0.0f;
            for (uint32_t i = 0; i < 32; ++i) {
                mn = std::min(mn, xp[i]);
                mx = std::max(mx, xp[i]);
            }
            const float sc = (mx - mn) / 15.0f;
            scale[g] = sc;
            minv[g] = -mn;
            maxScale = std::max(maxScale, sc);
            maxMin = std::max(maxMin, minv[g]);
            for (uint32_t i = 0; i < 32; ++i) {
                int v = sc > 1e-30f
                                ? static_cast<int>(std::lrintf((xp[i] - mn) / sc))
                                : 0;
                q[g * 32 + i] =
                        static_cast<uint8_t>(std::min(15, std::max(0, v)));
            }
        }
        const uint16_t hd = maxScale > 0.0f
                                    ? tinycoder::GGMLDequantize::floatToHalf(maxScale / 63.0f)
                                    : 0;
        const uint16_t hdm = maxMin > 0.0f
                                     ? tinycoder::GGMLDequantize::floatToHalf(maxMin / 63.0f)
                                     : 0;
        const float d = tinycoder::GGMLDequantize::halfToFloat(hd);
        const float dm = tinycoder::GGMLDequantize::halfToFloat(hdm);
        uint8_t sc6[8], m6[8];
        for (uint32_t g = 0; g < 8; ++g) {
            int s = d > 1e-30f ? static_cast<int>(std::lrintf(scale[g] / d)) : 0;
            int m = dm > 1e-30f ? static_cast<int>(std::lrintf(minv[g] / dm)) : 0;
            sc6[g] = static_cast<uint8_t>(std::min(63, std::max(0, s)));
            m6[g] = static_cast<uint8_t>(std::min(63, std::max(0, m)));
        }
        std::memcpy(out + 0, &hd, 2);
        std::memcpy(out + 2, &hdm, 2);
        uint8_t *scales = out + 4;
        for (uint32_t j = 0; j < 4; ++j) {
            scales[j] = static_cast<uint8_t>(sc6[j] | ((sc6[j + 4] >> 4) << 6));
            scales[j + 4] = static_cast<uint8_t>(m6[j] | ((m6[j + 4] >> 4) << 6));
            scales[j + 8] = static_cast<uint8_t>((sc6[j + 4] & 0x0F) |
                                                 ((m6[j + 4] & 0x0F) << 4));
        }
        uint8_t *qs = out + 16;
        std::memset(qs, 0, 128);
        for (uint32_t g = 0; g < 8; ++g) {
            uint8_t *dst = qs + (g / 2) * 32;
            const uint32_t sh = (g & 1) * 4;
            for (uint32_t i = 0; i < 32; ++i)
                dst[i] |= static_cast<uint8_t>(q[g * 32 + i] << sh);
        }
    }

    // Page-aligned ownership for a requantised blob.  The CUDA pinning pass
    // rounds ranges outwards to 4 KB, so a plain std::vector's tail page would
    // overlap the next allocation and cudaHostRegister fails with "requested
    // memory range is already mapped" (seen with three adjacent vectors).
    struct AlignedBlob {
        std::unique_ptr<uint8_t[]> p;
        size_t bytes = 0;
        const uint8_t *data() const { return p.get(); }
    };

    inline bool allocAligned(AlignedBlob &b, size_t bytes) {
        if (bytes == 0) return false;
        bytes = ((bytes + 4095u) / 4096u) * 4096u;
        void *q = nullptr;
        if (::posix_memalign(&q, 4096, bytes) != 0) return false;
        std::memset(q, 0, bytes);
        b.p.reset(static_cast<uint8_t *>(q));
        b.bytes = bytes;
        return true;
    }

    // Requantise a densely packed Q8_0 matrix ([rows][cols], 34 B per 32
    // values) into Q4_K ([rows][cols/256 * 144 B]).  `cols` must be a multiple
    // of 256 (Q4_K block width); `nThreads` splits the rows.
    inline bool q8ToQ4kInto(const uint8_t *src, uint32_t rows, uint32_t cols,
                            uint8_t *dst, uint32_t nThreads = 1) {
        if (src == nullptr || dst == nullptr || rows == 0 || cols == 0 ||
            (cols & 255u) != 0)
            return false;
        const uint32_t q8Blocks = cols / 32u;   // 34 B each
        const uint32_t kBlocks = cols / 256u;   // 144 B each
        auto work = [&](uint32_t r0, uint32_t r1) {
            std::vector<float> x(256);
            for (uint32_t r = r0; r < r1; ++r) {
                const uint8_t *s =
                        src + static_cast<size_t>(r) * q8Blocks * 34u;
                uint8_t *d = dst + static_cast<size_t>(r) * kBlocks * 144u;
                for (uint32_t b = 0; b < kBlocks; ++b) {
                    for (uint32_t sub = 0; sub < 8; ++sub)
                        tinycoder::GGMLDequantize::dequantizeQ8_0Block(
                                s + (b * 8u + sub) * 34u, x.data() + sub * 32u);
                    quantizeBlock(x.data(), d + b * 144u);
                }
            }
        };
        if (nThreads <= 1 || rows < 8) {
            work(0, rows);
            return true;
        }
        const uint32_t n = std::min<uint32_t>(nThreads, rows);
        std::vector<std::thread> th;
        th.reserve(n);
        const uint32_t chunk = (rows + n - 1) / n;
        for (uint32_t t = 0; t < n; ++t) {
            const uint32_t r0 = t * chunk;
            const uint32_t r1 = std::min(rows, r0 + chunk);
            if (r0 >= r1) break;
            th.emplace_back(work, r0, r1);
        }
        for (auto &t : th) t.join();
        return true;
    }

    // Size in bytes of the Q4_K form of a [rows][cols] matrix.
    inline size_t q4kBytes(uint32_t rows, uint32_t cols) {
        return static_cast<size_t>(rows) * (cols / 256u) * 144u;
    }

}// namespace q4k
