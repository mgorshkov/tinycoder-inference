// Tiny probe: compare every Q2_K prepacked dot-product variant against the
// dequantize-then-dot reference on the exact input used by
// DequantizeTest.Q2_K_PrePackedVsOriginal / Q2_K_Q8KernelVsReference.
#include "GGMLDequantize.hpp"
#include "SIMDMatMulVec.hpp"
#include "SIMDMatMulVecInternal.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace tinycoder;

namespace {
// Mirror of SIMDMatMulVec.cpp anonymous-namespace scalar float kernel.
float floatScalarProbe(const uint8_t *pb, const float *x) {
    auto halfToFloat = [](uint16_t h) -> float {
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f32;
        if (exp == 0) {
            if (mant == 0) {
                f32 = sign << 31;
            } else {
                int n = 0;
                while ((mant & 0x200) == 0 && n < 10) { mant <<= 1; n++; }
                mant &= 0x3FF;
                uint32_t mant_low = mant - 512;
                exp = 112 - n;
                f32 = (sign << 31) | (exp << 23) | (mant_low << 14);
            }
        } else if (exp == 31) {
            f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
        } else {
            exp = exp + (127 - 15);
            f32 = (sign << 31) | (exp << 23) | (mant << 13);
        }
        float result;
        std::memcpy(&result, &f32, sizeof(float));
        return result;
    };
    float d = halfToFloat(*(const uint16_t *)(pb + 16));
    float dmin = halfToFloat(*(const uint16_t *)(pb + 18));
    const uint8_t *scales = pb;
    const uint8_t *qs_expanded = pb + 20;
    double dot = 0.0;
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = scales[is++];
            float dl = d * (sc & 0xF);
            float ml = dmin * (sc >> 4);
            double sum_xq = 0.0, sum_x = 0.0;
            int base = n + j * 32;
            for (int l = 0; l < 16; ++l) {
                sum_xq += static_cast<double>(x[base + l]) * qs_expanded[l];
                sum_x += static_cast<double>(x[base + l]);
            }
            dot += static_cast<double>(dl) * sum_xq - static_cast<double>(ml) * sum_x;
            sc = scales[is++];
            dl = d * (sc & 0xF);
            ml = dmin * (sc >> 4);
            sum_xq = 0.0;
            sum_x = 0.0;
            for (int l = 0; l < 16; ++l) {
                sum_xq += static_cast<double>(x[base + 16 + l]) * qs_expanded[16 + l];
                sum_x += static_cast<double>(x[base + 16 + l]);
            }
            dot += static_cast<double>(dl) * sum_xq - static_cast<double>(ml) * sum_x;
        }
    }
    return static_cast<float>(dot);
}
} // namespace

int main() {
    uint8_t blockData[84] = {};
    blockData[80] = 0x00;
    blockData[81] = 0x40; // d = 2.0
    blockData[82] = 0x00;
    blockData[83] = 0x38; // dmin = 0.5
    for (int i = 0; i < 16; ++i) {
        blockData[i] = static_cast<uint8_t>((i << 4) | (15 - i));
    }
    for (int i = 0; i < 64; ++i) {
        uint8_t byteVal = 0;
        for (int b = 0; b < 4; ++b) {
            uint8_t val = static_cast<uint8_t>((i * 4 + b) % 4);
            byteVal |= (val << (b * 2));
        }
        blockData[16 + i] = byteVal;
    }

    float x[256];
    std::srand(42);
    for (int i = 0; i < 256; ++i) {
        x[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    float deqRef[256];
    GGMLDequantize::dequantizeQ2_KBlock(blockData, deqRef);
    double refDot = 0.0;
    for (int i = 0; i < 256; ++i) {
        refDot += static_cast<double>(x[i]) * deqRef[i];
    }
    float referenceResult = static_cast<float>(refDot);

    auto prepacked = GGMLDequantize::prepackQ2_K(blockData, 256);
    printf("prepacked size=%zu (expect 276)\n", prepacked.size());

    // 1. Dispatched float kernel (what the test calls)
    float dispFloat = dotProductQ2_K_PrePacked_SIMD(prepacked.data(), x);
    // 2. Probe copy of the anonymous scalar float kernel
    float scalarFloatProbe = floatScalarProbe(prepacked.data(), x);
    // 3. Direct AVX2 float
    float avx2Float = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
    avx2Float = simd::dotProductQ2_K_AVX2_PrePacked(prepacked.data(), x);
#endif
    // 4. Q8 path
    Q8KBlock q8;
    GGMLDequantize::quantizeQ8K(x, 256, &q8);
    float dispQ8 = dotProductQ2_K_PrePacked_Q8_SIMD(prepacked.data(), &q8);
#if defined(__AVX2__) && defined(__FMA__)
    float avx2Q8 = simd::dotProductQ2_K_PrePacked_Q8_AVX2(prepacked.data(), &q8);
#else
    float avx2Q8 = 0.0f;
#endif
#if defined(__AVX512F__) && defined(__FMA__)
    float avx512Q8 = simd::dotProductQ2_K_PrePacked_Q8_AVX512(prepacked.data(), &q8);
#else
    float avx512Q8 = 0.0f;
#endif
    // 5. Named scalar Q8 (anonymous in SIMDMatMulVec.cpp; reimplement equivalent
    //    via the dispatched default? just report the four above.)

    printf("reference            = %+.6f\n", referenceResult);
    printf("dispatch float       = %+.6f\n", dispFloat);
    printf("scalar float (probe) = %+.6f\n", scalarFloatProbe);
    printf("AVX2 float           = %+.6f\n", avx2Float);
    printf("dispatch Q8          = %+.6f\n", dispQ8);
    printf("AVX2 Q8              = %+.6f\n", avx2Q8);
    printf("AVX512 Q8            = %+.6f\n", avx512Q8);

    // Print first 32 qs_expanded values and first 32 dequantized refs.
    printf("qs_expanded[0..31]:");
    for (int i = 0; i < 32; ++i) printf(" %d", prepacked[20 + i]);
    printf("\nref[0..31]:");
    for (int i = 0; i < 32; ++i) printf(" %+.2f", deqRef[i]);
    printf("\n");
    return 0;
}
