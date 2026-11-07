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

// INT8 quantize kernel: f32 -> block_q8_1_int8
// fp32 -> Q8_1 INT8-layout activation quantizer.

#include "QuantBlocks.cuh"
#include "GemmKernels.cuh"


#ifdef USE_CUDA
#define CUDA_QUANTIZE_BLOCK_SIZE_INT8 128

static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_INT8) == 0, "Risk of out-of-bounds access.");

template <q8_1_ds_layout ds_layout>
static __global__ void quantizeQ8Blocks(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == Q8_1_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == Q8_1_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_int8 * y = (block_q8_1_int8 *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != Q8_1_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == Q8_1_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == Q8_1_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

// C-linkage quantize launchers

extern "C" void int8QuantizeQ8D4(
    const void *x_f32, const int32_t *ids, void *vy, int type_x,
    int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, void *stream) {
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_INT8 - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_INT8);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_INT8, 1, 1);
    quantizeQ8Blocks<Q8_1_LAYOUT_D4>
        <<<num_blocks, block_size, 0, (cudaStream_t)stream>>>(
            (const float *)x_f32, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

extern "C" void int8QuantizeQ8Ds4(
    const void *x_f32, const int32_t *ids, void *vy, int type_x,
    int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, void *stream) {
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_INT8 - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_INT8);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_INT8, 1, 1);
    quantizeQ8Blocks<Q8_1_LAYOUT_DS4>
        <<<num_blocks, block_size, 0, (cudaStream_t)stream>>>(
            (const float *)x_f32, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

extern "C" void int8QuantizeQ8D2s6(
    const void *x_f32, const int32_t *ids, void *vy, int type_x,
    int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, void *stream) {
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_INT8 - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_INT8);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_INT8, 1, 1);
    quantizeQ8Blocks<Q8_1_LAYOUT_D2S6>
        <<<num_blocks, block_size, 0, (cudaStream_t)stream>>>(
            (const float *)x_f32, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

#endif // USE_CUDA
