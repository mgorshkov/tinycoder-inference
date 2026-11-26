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

/*
 * CUDA GPU offload engine for TinyCoder.
 *
 * Two execution strategies (reference target: the Turing RTX 2080 Ti; also runs
 * on Ampere/Ada — RTX 3090/4090 — and newer GPUs via forward-compatible PTX):
 *
 * 1. Single-token generation (seqLen == 1) is DRAM-bandwidth-bound.
 *
 *   2. Batched prefill (seqLen > 1) is compute-bound.  Turing has no
 *      quantized tensor cores, so we materialize FP16 twins of all weights
 *      once at upload and run cublasGemmEx fp16 GEMMs (fp32 accumulate).
 *
 * Partial layer offload: upload() takes numGpuLayers (default = numLayers)
 * so only the first that many layers live on the GPU; the remainder can be
 * computed on the CPU by the caller (forward() copies the final hidden state
 * back instead of running the LM head when fewer than numLayers layers are
 * offloaded).
 *
 * Layer math (Qwen2 dense block), replicated vs tinycoder's CPU reference:
 *   hidden -> rmsNorm -> Q/K/V (+biases) -> RoPE(Q) + store KV(rotated K)
 *   -> warp flash attention -> attnO -> residual -> rmsNorm -> gate*up
 *   (SwiGLU silu fused) -> down -> residual; final rmsNorm -> LM head.
 */


// ============================================================================
// GPUComputeInternal.hpp -- INTERNAL shared implementation of the CUDA
// offload engine.  Created by splitting the former monolithic GPUCompute.cu
// into several translation units:
//   * GPUComputeGemm.cu      -- debug/parity probes (Q8_K / Q8_0 / INT8 rows)
//   * GPUComputeUpload.cu    -- scratch + weight upload / expert cache
//   * GPUComputeMoE.cu       -- resident-expert arenas + qwen35 prefix
//   * GPUComputeForward.cu   -- qwen35moe hybrid, forward(), CUDA graph
// This header holds the complete ANONYMOUS-NAMESPACE kernel + device-helper
// set.  Anonymous-namespace entities have INTERNAL linkage, so including this
// header in every .cu gives each TU its own copy -- no ODR/link collisions.
// The only cross-TU mutable state (the CUDA streams) lives OUTSIDE the
// anonymous namespace as `inline` variables (one shared instance).
// ============================================================================

#pragma once

#ifdef USE_CUDA

#include "EXL3Kernels.hpp"
#include "FFNPersistentKernel.hpp"
#include "GPUCompute.hpp"
#include "Int8Gemm.hpp"
#include "ThreadPool.hpp"
#if defined(__linux__)
#include <sys/mman.h>
#endif

// Device-side copies of the IQ2_XS / IQ3_S grid + sign lookup tables
// (byte-identical to the host tables in GridTables.cpp, GridTablesIQ3S.cpp
// and GGMLDequantize.hpp; auto-generated into this header).
#include "GridTablesDevice.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// Kernel set, split by designation.  The fragments are TEXT, included inside
// one anonymous namespace, so the preprocessed translation unit is identical to
// the former single file -- that equivalence is the contract, check it with
//     nvcc <flags> -E <any .cu> | grep -v '^#' | diff - <single-file build>
// Two things that look harmless and are not:
//   * The cross-TU `inline` state below (g_stream / g_cublas / g_stream2) MUST
//     stay OUTSIDE `namespace {`.  Moved inside, every TU gets its own stream;
//     launches from GemmDispatch/Upload then land on a stream the forward code
//     never ordered against.  Symptom: "[int8] launch failed" with err=no error,
//     prefill on a Q4_K_M checkpoint drops 2345 -> 189 tok/s and EXL3 loses 4.7%
//     (725 -> 690), while decode and every per-kernel resource figure stay
//     identical -- the kernels are fine, the ordering is not.
//   * The order below is load-bearing: helpers are used by the kernels after them.
namespace tinycoder::gpu {
    // ---- Cross-translation-unit GPU state ----
    // The CUDA stream / cuBLAS handle / init flag are SINGLE shared instances
    // (inline variables, external linkage) because the kernel + driver code is
    // now split across several translation units that all include this header.
    // A per-TU copy would give each TU a different stream and break kernel
    // ordering between them.
    inline cudaStream_t g_stream = nullptr;
    inline cublasHandle_t g_cublas = nullptr;
    // Secondary non-blocking stream (copy-engine only).  The CPU-expert
    // hybrid handoff (forwardQwen35MoePrefix) uses it for the norm D2H + FFN
    // H2D copies so they OVERLAP the GPU's g_stream compute.
    inline cudaStream_t g_stream2 = nullptr;
    inline std::once_flag g_cudaInit;
    // TINYCODER_GPU_VERBOSE=1: per-stage stream syncs + print (debug aid).
    inline const bool g_verbose = [] {
        const char *e = std::getenv("TINYCODER_GPU_VERBOSE");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();


    namespace {
        // DEPENDENCY ORDER, NOT ALPHABETICAL.  These are text fragments included into
        // one anonymous namespace: each relies on declarations the earlier ones
        // contribute (GpuCommon defines the shared helpers/macros the rest use,
        // GpuGemvIQInt8 defines kDequantF16 that GpuGemmDispatch launches).  Sorting
        // this list alphabetically breaks the build, so the block is fenced from
        // clang-format, which otherwise re-sorts includes.
        // clang-format off
#include "gpu/GpuCommon.hpp"
#include "gpu/GpuElementwiseKernels.hpp"
#include "gpu/GpuAttentionKernels.hpp"
#include "gpu/GpuDequantKernels.hpp"
#include "gpu/GpuGemvPQ2_0.hpp"
#include "gpu/GpuGemvPTQ1_0.hpp"
#include "gpu/GpuGemvQ8_1.hpp"
#include "gpu/GpuBatchGemmKernels.hpp"
#include "gpu/GpuGemvRowKQuant.hpp"
#include "gpu/GpuGemvQ2KFused.hpp"
#include "gpu/GpuGemvIQ.hpp"
#include "gpu/GpuGemvIQInt8.hpp"
#include "gpu/GpuGemmDispatch.hpp"
#include "gpu/GpuQwen35Kernels.hpp"
        // clang-format on
    }// namespace


    // ---- Cross-translation-unit device helpers ----
    // Defined in GPUComputeUpload.cu, called from GPUComputeMoE.cu.  Declared
    // at namespace scope (external linkage) so the call links across TUs.
    double bwSelfTestHostToPinned(const uint8_t *src, uint8_t *dst, size_t bytes,
                                  int iters);

}// namespace tinycoder::gpu

#endif// USE_CUDA
