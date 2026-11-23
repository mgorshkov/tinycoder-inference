// Expert-indexed Q8_1 GEMV for the two i-quant types the dense decode file
// (gemv/Gemv.cu) does not carry: IQ2_XXS and IQ1_S.
//
// Why a separate translation unit instead of two more GEMV_EXPERT_ENTRY lines
// in Gemv.cu: Gemv.cu is deliberately self-contained -- it re-declares its own
// block structs and grid tables -- while the Q8_1 dot products for these two
// types already exist exactly once, in gemm/QuantDot.cuh, next to the INT8 MMA
// path.  Including that header here reuses the same vec_dot code the prefill
// kernel uses rather than copying a 2048-entry grid table (IQ1_S) and a
// 256-entry one (IQ2_XXS) into a second file, where the two copies could
// drift.  The tables stay single-sourced; only the routing to them is new.
//
// The launch shape and the row indexing are the same as gemvExpertCoreImpl in
// Gemv.cu, so the only difference from the existing expert entries is the
// dot-product body:
//
//   output row r  ->  expert u = r / rowsPer, local row rr = r % rowsPer
//   weight row    ->  rowIdx[u] * rowsPer + rr
//   activation    ->  y + u * stride_col_y   (0 when all experts share a row)
//
// One CUDA block per output row, four warps per row, 32 lanes across the
// reduction.  The Q8_1 activation buffer is the one int8GemvExpert already
// quantizes in GemmDispatch.cu; these kernels only read it.

#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include <cstdint>

#include "../gemm/QuantDot.cuh"

#ifdef USE_CUDA
namespace {

    typedef float (*vec_dot_q_cuda_t)(const void *__restrict__ vbq,
                                      const block_q8_1 *__restrict__ bq8_1,
                                      const int &kbx, const int &iqs);

    /// Sum across the warp.  Same reduction tree as warp_reduce_sum_f32 in
    /// Gemv.cu (xor butterfly, all lanes get the total), written locally so
    /// this translation unit does not depend on Gemv.cu's statics.
    __device__ __forceinline__ float expertWarpSum(float x) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            x += __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE);
        return x;
    }

    /// The expert-indexed mat-vec body.  Verbatim shape of gemvExpertCoreImpl
    /// in Gemv.cu: `qi` weight-block sub-indices per row-block step, `vdr`
    /// quant blocks per iteration per thread.
    template<int qk, int qi, typename block_q_t, int vdr,
             vec_dot_q_cuda_t vec_dot_q_cuda>
    __device__ void gemvExpertIqCore(
            const void *__restrict__ vx, const block_q8_1 *__restrict__ y,
            float *__restrict__ dst, const int ncols_x, const int nrows_x,
            const int stride_col_y, const int *__restrict__ rowIdx,
            const int rowsPer) {
        constexpr int nwarps = 4;
        const int outRow = blockIdx.x;
        if (outRow >= nrows_x) return;
        const int u = outRow / rowsPer;
        const int rr = outRow - u * rowsPer;
        const int row = rowIdx[u] * rowsPer + rr;
        const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
        const int blocks_per_row_x = ncols_x / qk;
        constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
        const block_q8_1 *yv = y + (size_t) u * stride_col_y;

        float tmp = 0.0f;
        for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
             kbx += blocks_per_iter) {
            const int kby = kbx * (qk / QK8_1);
            const int kqs = vdr * (tid % (qi / vdr));
            tmp += vec_dot_q_cuda(vx, &yv[kby], row * blocks_per_row_x + kbx, kqs);
        }

        __shared__ float red[nwarps][WARP_SIZE];
        red[threadIdx.y][threadIdx.x] = tmp;
        __syncthreads();
        if (threadIdx.y != 0) return;
        float acc = red[0][threadIdx.x];
#pragma unroll
        for (int l = 1; l < nwarps; ++l)
            acc += red[l][threadIdx.x];
        acc = expertWarpSum(acc);
        if (threadIdx.x == 0) dst[outRow] = acc;
    }

    /// Four output rows per warp, one warp per row group (the decode shape).
    ///
    /// The one-row kernel above launches one CUDA block per output row, so a
    /// gate/up call is 18432 blocks of 128 threads that each read the SAME
    /// Q8_1 activation row (2592 B) and 4 warps' worth of reduction for a
    /// 650-740 B weight row.  Here a warp owns four consecutive rows: the
    /// (kbx, sub) coverage is identical -- 8 lanes cover a 256-weight block's
    /// eight 32-weight sub-blocks, four lane groups stride by four blocks -- so
    /// every 32-weight dot is the same vec_dot on the same operands, but the
    /// activation row is read once for four rows instead of four times, the
    /// launch drops 32x (8 warps x 4 rows per block), and the cross-warp
    /// reduction becomes one warp butterfly.
    ///
    /// Only the shared-activation shape is served (every expert reads the same
    /// row), so there is no stride_col_y: the row group is a contiguous slice
    /// of rowsPer, and rowsPer is a multiple of 4 on this model, so a group
    /// never straddles two experts.  A group that did straddle would still be
    /// correct per row; the u/rowsPer mapping below is applied per row.
    template<int qk, int qi, typename block_q_t, int vdr,
             vec_dot_q_cuda_t vec_dot_q_cuda>
    __device__ void gemvExpert4IqCore(
            const void *__restrict__ vx, const block_q8_1 *__restrict__ y,
            float *__restrict__ dst, const int ncols_x, const int nrows_x,
            const int *__restrict__ rowIdx, const int rowsPer) {
        constexpr int kRows = 4;
        constexpr int kWarps = 8;
        const int lane = threadIdx.x & 31u;
        const int warp = threadIdx.x >> 5u;
        const int o0 = (blockIdx.x * kWarps + warp) * kRows;
        if (o0 >= nrows_x) return;
        const int nr = min(kRows, nrows_x - o0);
        const int blocks_per_row = ncols_x / qk;
        const block_q_t *base = static_cast<const block_q_t *>(vx);

        const block_q_t *rp[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int o = o0 + (r < nr ? r : 0);
            const int u = o / rowsPer;
            const int rr = o - u * rowsPer;
            const int row = rowIdx[u] * rowsPer + rr;
            rp[r] = base + static_cast<size_t>(row) * blocks_per_row;
        }

        float acc[kRows] = {0.0f, 0.0f, 0.0f, 0.0f};
        // Same (quant block, 32-weight sub-block) mapping as the one-row
        // kernel's tid/8 and tid%8, on 32 lanes instead of 128.
        for (int kbx = lane >> 3; kbx < blocks_per_row; kbx += 4) {
            const int kby = kbx * (qk / QK8_1);
            const int kqs = vdr * (lane & 7);
            const block_q8_1 *yb = y + kby;
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                if (r < nr) acc[r] += vec_dot_q_cuda(rp[r], yb, kbx, kqs);
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                if (r < nr) dst[o0 + r] = acc[r];
        }
    }

}// namespace

// Host launchers.  The names match the DECL_GEMV_EXPERT / int8GemvExpert
// dispatch in gemm/GemmDispatch.cu; grid and block are the expert shape.

#define GEMV_EXPERT_IQ_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot)   \
    extern "C" __global__ void gemvExpert##tag##F32(                            \
            const void *__restrict__ vx, const void *__restrict__ vy,           \
            float *__restrict__ dst, const int ncols_x, const int nrows_x,      \
            const int stride_col_y, const int *__restrict__ rowIdx,             \
            const int rowsPer) {                                                \
        gemvExpertIqCore<qk_val, qi_val, block_q_t, vdr_val, vec_dot>(          \
                vx, (const block_q8_1 *) vy, dst, ncols_x, nrows_x,             \
                stride_col_y, rowIdx, rowsPer);                                 \
    }                                                                           \
    extern "C" void gemvLaunchExpert##tag(                                      \
            const void *vx, const void *vy, void *dst, int ncols_x,             \
            int nrows_x, int stride_col_y, const int *rowIdx, int rowsPer,      \
            void *stream) {                                                     \
        dim3 grid((unsigned int) nrows_x, 1, 1);                                \
        dim3 block(WARP_SIZE, 4, 1);                                            \
        gemvExpert##tag##F32<<<grid, block, 0,                                  \
                               static_cast<cudaStream_t>(stream)>>>(            \
                vx, vy, (float *) dst, ncols_x, nrows_x, stride_col_y, rowIdx,  \
                rowsPer);                                                       \
    }

// IQ2_XXS: QR2_XXS = 4, QI2_XXS = 16, VDR = 2 (same shape as IQ2_XS).
GEMV_EXPERT_IQ_ENTRY(IQ2XXS, block_iq2_xxs, QK_K, QI2_XXS, VDR_IQ2_XXS_Q8_1_GEMV,
                     vec_dot_iq2_xxs_q8_1)
// IQ1_S: QR1_S = 8, QI1_S = 8, VDR = 1.
GEMV_EXPERT_IQ_ENTRY(IQ1S, block_iq1_s, QK_K, QI1_S, VDR_IQ1_S_Q8_1_GEMV,
                     vec_dot_iq1_s_q8_1)

// Four rows per warp (decode).  One entry per gate/up type: IQ2_XS and IQ2_S
// already have their own one-row kernels in Gemv.cu, so only the 4-row entries
// are added for them here -- the one-row symbols would collide.

#define GEMV_EXPERT4_IQ_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot)  \
    extern "C" __global__ void gemvExpert4##tag##F32(                           \
            const void *__restrict__ vx, const void *__restrict__ vy,           \
            float *__restrict__ dst, const int ncols_x, const int nrows_x,      \
            const int *__restrict__ rowIdx, const int rowsPer) {                \
        gemvExpert4IqCore<qk_val, qi_val, block_q_t, vdr_val, vec_dot>(         \
                vx, (const block_q8_1 *) vy, dst, ncols_x, nrows_x, rowIdx,     \
                rowsPer);                                                       \
    }                                                                           \
    extern "C" void gemvLaunchExpert4##tag(                                     \
            const void *vx, const void *vy, void *dst, int ncols_x,             \
            int nrows_x, const int *rowIdx, int rowsPer, void *stream) {        \
        constexpr int kRowsPerBlock = 8 * 4;                                    \
        dim3 grid((unsigned int) ((nrows_x + kRowsPerBlock - 1) /               \
                                  kRowsPerBlock),                               \
                  1, 1);                                                        \
        dim3 block(32 * 8, 1, 1);                                               \
        gemvExpert4##tag##F32<<<grid, block, 0,                                 \
                                static_cast<cudaStream_t>(stream)>>>(           \
                vx, vy, (float *) dst, ncols_x, nrows_x, rowIdx, rowsPer);      \
    }

GEMV_EXPERT4_IQ_ENTRY(IQ2XXS, block_iq2_xxs, QK_K, QI2_XXS, VDR_IQ2_XXS_Q8_1_GEMV,
                      vec_dot_iq2_xxs_q8_1)
GEMV_EXPERT4_IQ_ENTRY(IQ2XS, block_iq2_xs, QK_K, QI2_XS, VDR_IQ2_XS_Q8_1_GEMV,
                      vec_dot_iq2_xs_q8_1)
GEMV_EXPERT4_IQ_ENTRY(IQ2S, block_iq2_s, QK_K, QI2_S, VDR_IQ2_S_Q8_1_GEMV,
                      vec_dot_iq2_s_q8_1)
GEMV_EXPERT4_IQ_ENTRY(IQ1S, block_iq1_s, QK_K, QI1_S, VDR_IQ1_S_Q8_1_GEMV,
                      vec_dot_iq1_s_q8_1)

#endif// USE_CUDA
