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

#include "GGMLQuantize.hpp"

#include "GGMLDequantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tinycoder {

    namespace {
        // 6-bit packed scale/min fields, ggml get_scale_min_k4 order.
        void packQ4KScales(const uint8_t *sc6, const uint8_t *m6, uint8_t *out) {
            for (uint32_t j = 0; j < 4; ++j) {
                out[j] = static_cast<uint8_t>(sc6[j] | ((sc6[j + 4] >> 4) << 6));
                out[j + 4] = static_cast<uint8_t>(m6[j] | ((m6[j + 4] >> 4) << 6));
                out[j + 8] = static_cast<uint8_t>((sc6[j + 4] & 0x0F) |
                                                  ((m6[j + 4] & 0x0F) << 4));
            }
        }
    }// namespace

    void GGMLQuantize::quantizeQ4KBlock(const float *x, uint8_t *out) {
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
                int v = sc > 1e-30f ? static_cast<int>(std::lrintf((xp[i] - mn) / sc)) : 0;
                q[g * 32 + i] = static_cast<uint8_t>(std::min(15, std::max(0, v)));
            }
        }
        const uint16_t hd = maxScale > 0.0f
                                    ? GGMLDequantize::floatToHalf(maxScale / 63.0f)
                                    : 0;
        const uint16_t hdm = maxMin > 0.0f
                                     ? GGMLDequantize::floatToHalf(maxMin / 63.0f)
                                     : 0;
        const float d = GGMLDequantize::halfToFloat(hd);
        const float dm = GGMLDequantize::halfToFloat(hdm);
        uint8_t sc6[8], m6[8];
        for (uint32_t g = 0; g < 8; ++g) {
            int s = d > 1e-30f ? static_cast<int>(std::lrintf(scale[g] / d)) : 0;
            int m = dm > 1e-30f ? static_cast<int>(std::lrintf(minv[g] / dm)) : 0;
            sc6[g] = static_cast<uint8_t>(std::min(63, std::max(0, s)));
            m6[g] = static_cast<uint8_t>(std::min(63, std::max(0, m)));
        }
        std::memcpy(out + 0, &hd, 2);
        std::memcpy(out + 2, &hdm, 2);
        packQ4KScales(sc6, m6, out + 4);

        uint8_t *qs = out + 16;
        std::memset(qs, 0, 128);
        for (uint32_t g = 0; g < 8; ++g) {
            uint8_t *dst = qs + (g / 2) * 32;
            const uint32_t sh = (g & 1) * 4;
            for (uint32_t i = 0; i < 32; ++i)
                dst[i] |= static_cast<uint8_t>(q[g * 32 + i] << sh);
        }
    }

}// namespace tinycoder
