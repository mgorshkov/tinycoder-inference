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

// GemmEntry.cu — extern "C" __global__ entry points for the int8 tile /
// stream-k kernels and the Q8_1 int8-layout activation quantizers.
//
// The int8 tensor-core `int8Matmul` implementation lives in GemmKernels.cuh
// (tile loader + stream-k), Int8Mma.cuh (mma.s8.m8n8k16 fragments) and
// QuantDot.cuh (per-quant dot).  The engine never calls these wrappers
// directly: it goes through include/Int8Gemm.hpp -> GemmDispatch.cu, which owns
// the device query, scratch and tile selection.
//
// Kernel families:
//   tcInt8Dense_<q>_x<N>    tiled grid (kept for completeness / fallback)
//   tcInt8Sk_<q>_x<N>       stream-k grid (nSM blocks) + fixup
//   tcInt8Fixup_<q>_x<N>    folds partial tiles split across blocks
//   tcInt8Moe_<q>_x<N>      MoE gather variant (ids_dst + expert_bounds)
//   tcInt8Quant_<layout>    fp32 -> block_q8_1_int8 activation quantize
//   tcInt8Gather_<layout>   gathered-row variant of the quantizer

#include "QuantBlocks.cuh"
#include "GemmKernels.cuh"


#ifdef USE_CUDA
// ---------------------------------------------------------------------------
// Q8_1 quantization (INT8 layout) — __device__ template kept local so the
// host-side launchers are not pulled in.
// ---------------------------------------------------------------------------
template <q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void quantizeQ8Blocks_impl(
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

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1);
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;
    const int64_t iqs = i0 % (4*QK8_1);

    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != Q8_1_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

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

// ---------------------------------------------------------------------------
// DENSE Q8_1 quantization wrappers (ids=null, ne2=1, s02=s03=0).
// grid=(ne1, ceil(ne0/512), 1), block=(128). ne00=k, s01=k, ne0=pad(k,512).
// ---------------------------------------------------------------------------
#define DEFINE_QUANT(layout_const, name) \
    extern "C" __global__ void int8Quant##name( \
        const float * __restrict__ x, void * __restrict__ vy, \
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) { \
        quantizeQ8Blocks_impl<layout_const>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1); \
    }

DEFINE_QUANT(Q8_1_LAYOUT_D4,   D4)
DEFINE_QUANT(Q8_1_LAYOUT_DS4,  DS4)
DEFINE_QUANT(Q8_1_LAYOUT_D2S6, D2S6)

// ---------------------------------------------------------------------------
// DENSE int8Matmul wrappers: ids_dst=null, expert_bounds=null, tmp_fixup=null,
// channel/sample parameters hard-coded (nchannels=nsamples=1, all strides=0).
// ---------------------------------------------------------------------------
#define DEFINE_INT8_DENSE(tc_type_const, tag, INT8_X) \
    extern "C" __global__ void __launch_bounds__(256) \
    int8GemmDense##tag##Tile##INT8_X( \
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst, \
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, \
        const int ncols_y, const int stride_col_dst, const int ncols_max) { \
        int8MatmulImpl<tc_type_const, INT8_X, false>( \
            x, y, nullptr, nullptr, dst, nullptr, \
            ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, \
            1, 1, 0, 0, 0, \
            1, 1, 0, 0, 0, \
            ncols_max); \
    }

// K-quants + base types, tile_x in {32, 64, 128}
DEFINE_INT8_DENSE(TC_TYPE_Q4_K, Q4K, 32)
DEFINE_INT8_DENSE(TC_TYPE_Q4_K, Q4K, 64)
DEFINE_INT8_DENSE(TC_TYPE_Q4_K, Q4K, 128)
DEFINE_INT8_DENSE(TC_TYPE_Q5_K, Q5K, 32)
DEFINE_INT8_DENSE(TC_TYPE_Q5_K, Q5K, 64)
DEFINE_INT8_DENSE(TC_TYPE_Q5_K, Q5K, 128)
DEFINE_INT8_DENSE(TC_TYPE_Q6_K, Q6K, 32)
DEFINE_INT8_DENSE(TC_TYPE_Q6_K, Q6K, 64)
DEFINE_INT8_DENSE(TC_TYPE_Q6_K, Q6K, 128)
DEFINE_INT8_DENSE(TC_TYPE_Q8_0, Q80, 32)
DEFINE_INT8_DENSE(TC_TYPE_Q8_0, Q80, 64)
DEFINE_INT8_DENSE(TC_TYPE_Q8_0, Q80, 128)
DEFINE_INT8_DENSE(TC_TYPE_IQ2_S, IQ2S, 32)
DEFINE_INT8_DENSE(TC_TYPE_IQ2_S, IQ2S, 64)
DEFINE_INT8_DENSE(TC_TYPE_IQ2_S, IQ2S, 128)
DEFINE_INT8_DENSE(TC_TYPE_IQ3_XXS, IQ3XXS, 32)
DEFINE_INT8_DENSE(TC_TYPE_IQ3_XXS, IQ3XXS, 64)
DEFINE_INT8_DENSE(TC_TYPE_IQ3_XXS, IQ3XXS, 128)

// IQ quants (UD models): load_tiles_iq* dequantize a tile to q8 format in
// shared memory, then the same MMA path. Tile: iq2_xs/iq2_s -> Q3_K (84),
// others -> Q8_0 (76); all IQ types use the D4 Q8_1 activation layout.
// (int8_get_q8_1_ds_layout).


// ---------------------------------------------------------------------------
// DENSE stream-k: grid nblocks < ntiles (the sm_75+ schedule), work
// split continuously over (it, jt, k blocks); the tail partial sums go to
// tmp_fixup and are finished by tcInt8Fixup_<tag>_x<N>. ids/expert = null.
// ---------------------------------------------------------------------------
#define DEFINE_INT8_DENSE_SK(tc_type_const, tag, INT8_X) \
    extern "C" __global__ void __launch_bounds__(256) \
    int8GemmStreamK##tag##Tile##INT8_X( \
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst, \
        float * __restrict__ tmp_fixup, \
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, \
        const int ncols_y, const int stride_col_dst, const int ncols_max) { \
        int8MatmulImpl<tc_type_const, INT8_X, false>( \
            x, y, nullptr, nullptr, dst, tmp_fixup, \
            ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, \
            1, 1, 0, 0, 0, \
            1, 1, 0, 0, 0, \
            ncols_max); \
    }

#define DEFINE_INT8_FIXUP(tc_type_const, tag, INT8_X) \
    extern "C" __global__ void __launch_bounds__(256) \
    int8GemmFixup##tag##Tile##INT8_X( \
        float * __restrict__ dst, const float * __restrict__ tmp_fixup, \
        const int ncols_x, const int nrows_x, const int ncols_dst, \
        const int stride_col_dst, const int ncols_max) { \
        int8MatmulStreamKFixupImpl<tc_type_const, INT8_X, false>( \
            nullptr, nullptr, dst, tmp_fixup, \
            ncols_x, nrows_x, ncols_dst, (size_t) stride_col_dst, \
            1, 0, 1, 0, ncols_max); \
    }

#define DEFINE_INT8_SK_PAIR(tc_type_const, tag) \
    DEFINE_INT8_DENSE_SK(tc_type_const, tag, 32) \
    DEFINE_INT8_DENSE_SK(tc_type_const, tag, 64) \
    DEFINE_INT8_DENSE_SK(tc_type_const, tag, 128) \
    DEFINE_INT8_FIXUP(tc_type_const, tag, 32) \
    DEFINE_INT8_FIXUP(tc_type_const, tag, 64) \
    DEFINE_INT8_FIXUP(tc_type_const, tag, 128)

DEFINE_INT8_SK_PAIR(TC_TYPE_Q4_K, Q4K)
DEFINE_INT8_SK_PAIR(TC_TYPE_Q5_K, Q5K)
DEFINE_INT8_SK_PAIR(TC_TYPE_Q6_K, Q6K)
DEFINE_INT8_SK_PAIR(TC_TYPE_Q8_0, Q80)
DEFINE_INT8_SK_PAIR(TC_TYPE_IQ2_S, IQ2S)
DEFINE_INT8_SK_PAIR(TC_TYPE_IQ3_XXS, IQ3XXS)

// ---------------------------------------------------------------------------
// GATHER Q8_1 quantization (INT8 layout): output row i1 comes from input row
// ids[i1]. Needed for MoE: activations are laid out in expert-sorted order and
// the INT8 y matrix must use the same order as expert_bounds. ids=null means
// ordinary dense quantization.
// grid=(ne1, ceil(ne0/512), 1), block=(128).
// ---------------------------------------------------------------------------
#define DEFINE_QUANT_GATHER(layout_const, name) \
    extern "C" __global__ void tcInt8Quant_gather_##name( \
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy, \
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) { \
        quantizeQ8Blocks_impl<layout_const>(x, ids, vy, ne00, s01, 0, 0, ne0, ne1, 1); \
    }

DEFINE_QUANT_GATHER(Q8_1_LAYOUT_D4,   d4)
DEFINE_QUANT_GATHER(Q8_1_LAYOUT_DS4,  ds4)
DEFINE_QUANT_GATHER(Q8_1_LAYOUT_D2S6, d2s6)

// ---------------------------------------------------------------------------
// MoE wrappers for int8Matmul (MoE layout): ids_dst + expert_bounds.
//   x    — weights of ALL experts consecutively, [n_experts, nrows_x, ncols_x];
//          expert zt starts at block zt*stride_channel_x;
//   y    — activations in sorted order [m_total, ncols_x] in q8_1_int8;
//   dst  — [m_total, nrows_x], row jt*tile_x+j written to ids_dst[col_low+...];
// grid=(n_experts*ntx*nty, 1, 1) — experts are iterated inside over zt; tiles
// outside expert_bounds exit immediately.
// need_check=true: the expert's column count is not a multiple of tile_x.
// ---------------------------------------------------------------------------
#define DEFINE_INT8_MOE(tc_type_const, tag, INT8_X) \
    extern "C" __global__ void __launch_bounds__(256) \
    tcInt8Moe_##tag##_x##INT8_X( \
        const unsigned long long x_addr, const int * __restrict__ y, \
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds, \
        float * __restrict__ dst, \
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, \
        const int ncols_y, const int stride_col_dst, \
        const int n_experts, const int stride_channel_x, const int ncols_max) { \
        int8MatmulImpl<tc_type_const, INT8_X, true>( \
            (const char *) x_addr, y, ids_dst, expert_bounds, dst, nullptr, \
            ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, \
            /*channel_ratio=*/1, /*nchannels_y=*/n_experts, \
            /*stride_channel_x=*/stride_channel_x, /*stride_channel_y=*/0, /*stride_channel_dst=*/0, \
            /*sample_ratio=*/1, /*nsamples_y=*/1, \
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0, \
            ncols_max); \
    }


#endif // USE_CUDA
