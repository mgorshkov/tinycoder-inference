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

// GpuCommon.hpp -- GGML type codes, env-gated launch helpers.
//
// Fragment of GPUComputeInternal.hpp, not a standalone header: it is
// #included there, in the order listed, INSIDE one anonymous namespace.
// Do not add `#pragma once` and do not include it directly -- the file is
// text, and every entity in it relies on the namespace and on the
// declarations that the fragments ahead of it contribute.


// GGML type codes (GGUFLoader.hpp).
constexpr uint32_t kTypeF32 = 0;
constexpr uint32_t kTypeF16 = 1;
constexpr uint32_t kTypeQ5_0 = 6;
constexpr uint32_t kTypeQ8_0 = 8;
constexpr uint32_t kTypeQ2K = 10;
constexpr uint32_t kTypeQ3K = 11;
constexpr uint32_t kTypeQ4K = 12;
constexpr uint32_t kTypeQ5K = 13;
constexpr uint32_t kTypeQ6K = 14;
constexpr uint32_t kTypeIQ2XS = 17;
constexpr uint32_t kTypeIQ3_XXS = 18;
constexpr uint32_t kTypeIQ2XXS = 16;
constexpr uint32_t kTypeIQ4NL = 20;
constexpr uint32_t kTypeIQ4XS = 23;
constexpr uint32_t kTypeIQ3S = 21;
constexpr uint32_t kTypeIQ2S = 22;
// Prism ternary (Bonsai) — 2 bits/weight, 34-byte / 128-weight blocks.
// Native EXL3 trellis linear (not a GGML block type; see
// IModelLoader.hpp / EXL3Kernels.hpp).
constexpr uint32_t kTypeEXL3 = 200;
constexpr uint32_t kTypePQ2_0 = 142;
constexpr uint32_t kPQ2_0_BYTES = 34; // d(fp16,2) + qs[32]
constexpr uint32_t kTypePTQ1_0 = 143; // Prism native ternary, 28 B/128
constexpr uint32_t kPTQ1_0_BYTES = 28;// qs[24] + qh[2] + d(fp16,2)
// Prism Hadamard-fold block size (prism.hadamard.block_size).
constexpr uint32_t kPrismBlock = 1024;

constexpr uint32_t kQ5_0_BYTES = 22;
constexpr uint32_t kQ8_0_BYTES = 34;
constexpr uint32_t kQ2K_BYTES = 84;
constexpr uint32_t kQ3K_BYTES = 110;
constexpr uint32_t kQ5K_BYTES = 176;
constexpr uint32_t kQ4K_BYTES = 144;
constexpr uint32_t kQ6K_BYTES = 210;
constexpr uint32_t kIQ2XS_BYTES = 74;
constexpr uint32_t kIQ3S_BYTES = 110;
constexpr uint32_t kIQ3XXS_BYTES = 98;
constexpr uint32_t kIQ2S_BYTES = 82;
constexpr uint32_t kIQ2XXS_BYTES = 66;
constexpr uint32_t kIQ4NL_BYTES = 18;// d(fp16,2) + qs[16]
constexpr uint32_t kIQ4XS_BYTES = 136;

constexpr float LOG2E = 1.4426950408889634f;


bool ensureCuda() {
    std::call_once(g_cudaInit, []() {
        if (cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking) !=
            cudaSuccess)
            return;
        if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) return;
        cublasSetStream(g_cublas, g_stream);
        if (cudaStreamCreateWithFlags(&g_stream2,
                                      cudaStreamNonBlocking) !=
            cudaSuccess)
            return;
    });
    return g_stream && g_cublas && g_stream2;
}
