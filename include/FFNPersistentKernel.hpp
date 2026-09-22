/*
⚡ TinyCoder AI

Copyright (c) 2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <cstdint>

#ifdef USE_CUDA

#include <cuda_runtime.h>

namespace tinycoder::gpu {

    // ---------------------------------------------------------------------
    // Persistent fused FFN mega-kernel (idea #21/A, 2026-10-08)
    //
    // ONE grid-barrier-sequenced launch replaces the entire dense-Qwen2 FFN
    // sub-block of the decode (seqLen==1) critical path:
    //
    //     FFN RMSNorm -> gate GEMV -> up GEMV -> silu-mul -> down GEMV
    //                 -> residual add
    //
    // The phases are separated by a device-side arrive/wait barrier instead of
    // six kernel-launch boundaries, so the per-layer inter-kernel dependency
    // bubbles inside the FFN disappear.  The grid is clamped to the exact
    // simultaneous residency (occupancy x SM count); the barrier is the classic
    // two-phase atomic counter/phase pair, NOT cooperative launch / grid_group,
    // so it compiles for any arch (the build targets sm_52 + PTX-JIT sm_75).
    //
    // ONLY runs for gate/up == Q2_K and down == Q3_K with
    // cols == blocksPerRow*256 (the 1.5B q2_k dense FFN).  Per-row arithmetic
    // is the kQGemv<kTypeQ2K>/<kTypeQ3K> 4-way-batched, in-order-fold,
    // 5-shuffle-tree float path -- the same value class as launchQGemv.
    //
    // Enabled by the caller only when TINYCODER_FUSE_FFN_PERSIST != 0.
    // ---------------------------------------------------------------------
    struct FfnPersistentArgs {
        const float *hiddenRes = nullptr;// [H] residual stream in
        float *hiddenOut = nullptr;      // [H] residual stream out
        const float *ffnNormW = nullptr; // [H] rmsNormFFN weight
        const uint8_t *wg = nullptr;     // gate Q2_K weight rows
        const uint8_t *wu = nullptr;     // up   Q2_K weight rows
        const uint8_t *wd = nullptr;     // down Q3_K weight rows
        float *productOut = nullptr;     // [I] = silu(gate)*up
        float *downOut = nullptr;        // [H] = down(product)
        uint32_t H = 0;                  // hidden size
        uint32_t I = 0;                  // intermediate size (== ffnRows)
        uint32_t ffnRows = 0;            // gate/up rows (8960 for the 1.5B)
        uint32_t downRows = 0;           // down rows (H)
        uint32_t gateRowBytes = 0;
        uint32_t downRowBytes = 0;
        uint32_t gateBlocksPerRow = 0;
        uint32_t downBlocksPerRow = 0;
        float eps = 1e-6f;
        // Q8_1 activation path (2026-10-08): when set, phase 1 additionally
        // quantizes the block-local FFN norm to llama.cpp's Q8_1 (per-32-block
        // d/s + int8, no sign forcing) into shared memory, and the gate/up dots
        // use the integer dp4a path instead of the fp32 4xW path.  The down
        // phase always consumes the fp32 silu product (productOut), so it is
        // unchanged.  Falls back to the float path if shared memory is
        // exhausted.
        bool q81 = false;
    };

    /// @brief Launch the persistent fused FFN mega-kernel on `stream`.
    /// @return true if the kernel was launched; false if the device is
    ///         unsupported / an allocation or occupancy query failed (the
    ///         caller must then run the default multi-launch FFN body).
    ///
    /// The barrier scratch is a permanently-owned device pair shared across
    /// launches (the decode is single-stream + serial, so one pair suffices);
    /// it is re-zeroed on `stream` before every launch.
    bool launchFfnPersistentLayer(const FfnPersistentArgs &args, cudaStream_t stream);

}// namespace tinycoder::gpu

#endif// USE_CUDA
