/*
 * TinyCoder CUDA GPU offload engine -- debug & parity probes
 *
 * Part of the CUDA offload engine split out of the former GPUCompute.cu.  The
 * shared kernels/device helpers live in GPUComputeInternal.hpp (anonymous
 * namespace, so each translation unit gets its own internal-linkage copy).
 */

#ifdef USE_CUDA

#include "GPUComputeInternal.hpp"

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

namespace tinycoder::gpu {

    // ===== GPU debug / parity probes (Q8_K, Q8_0, MMQ row scans) =====
    // Host-side diagnostic: run ONE weight row through the GPU Q8_K GEMV
    // (kQuantizeQ8K + kQGemvQ8K) for the IQ2_S / IQ3_XXS / IQ3_S types, or
    // through the Q8_K-activation Q4_K / Q6_K GEMV (kQuantizeQ8K +
    // kQGemvKxQ8K), and return the scalar row output.  Used by unit tests to
    // verify the GPU integer-dot kernels reproduce the CPU Q8K path
    // bit-for-bit.
    float GPUModel::debugQ8KRow(uint32_t type, const uint8_t *wRow,
                                uint32_t blocksPerRow, const float *x,
                                uint32_t cols) {
        // Host->device: one weight row + one activation vector.
        uint32_t rowBytes = blocksPerRow *
                            (type == kTypeIQ2S      ? kIQ2S_BYTES
                             : type == kTypeIQ3_XXS ? kIQ3XXS_BYTES
                             : type == kTypeQ4K     ? static_cast<uint32_t>(
                                                          kQ4K_BYTES)
                             : type == kTypeQ5K ? static_cast<uint32_t>(
                                                          kQ5K_BYTES)
                             : type == kTypeQ6K ? static_cast<uint32_t>(
                                                          kQ6K_BYTES)
                                                : kIQ3S_BYTES);
        uint8_t *dW = nullptr;
        float *dX = nullptr, *dOut = nullptr;
        cudaMalloc(&dW, rowBytes);
        cudaMalloc(&dX, cols * sizeof(float));
        cudaMalloc(&dOut, sizeof(float));
        uint8_t *dQ8 = nullptr;
        cudaMalloc(&dQ8, static_cast<uint64_t>(blocksPerRow) * kQ8K_STRIDE);
        cudaMemcpy(dW, wRow, rowBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dX, x, cols * sizeof(float), cudaMemcpyHostToDevice);
        kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(dX, dQ8, cols);
        if (type == kTypeIQ2S) {
            kQGemvQ8K<kTypeIQ2S><<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        } else if (type == kTypeIQ3_XXS) {
            kQGemvQ8K<kTypeIQ3_XXS><<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        } else if (type == kTypeQ4K) {
            kQGemvKxQ8K<kTypeQ4K><<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        } else if (type == kTypeQ5K) {
            kQGemvKxQ8K<kTypeQ5K><<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        } else if (type == kTypeQ6K) {
            kQGemvKxQ8K<kTypeQ6K><<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        } else {
            kQGemvQ8K<kTypeIQ3S><<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        }
        // The kernels ran on the non-blocking g_stream; a synchronous memcpy on
        // the default stream would NOT wait for them, and the immediatelly
        // following cudaFree()s could hand the same addresses back to the next
        // call's cudaMalloc() while the previous kernels are still writing.
        // Sync the stream before reading the result (diagnostic-only path --
        // correctness over speed).
        cudaStreamSynchronize(g_stream);
        float result = 0.0f;
        cudaMemcpy(&result, dOut, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dW);
        cudaFree(dX);
        cudaFree(dOut);
        cudaFree(dQ8);
        return result;
    }

    // Host-side diagnostic: run ONE weight row through the GPU Q8_0 INTEGER
    // GEMV (kQuantizeQ8_0x32 + kQGemvQ8_0xQ8K) -- the exact path launchQGemv's
    // Q8_0 branch uses for the qwen35moe trunk projections and LM head -- and
    // return the scalar row output.  Used by unit tests to verify the GPU
    // reproduces the CPU matMulVecBatchQ8_0_Q8K_AVX2 reference bit-for-bit.
    float GPUModel::debugQ8_0Row(const uint8_t *wRow, uint32_t blocksPerRow,
                                 const float *x, uint32_t cols) {
        const uint32_t rowBytes = blocksPerRow * kQ8_0_BYTES;
        uint8_t *dW = nullptr;
        float *dX = nullptr, *dOut = nullptr;
        cudaMalloc(&dW, rowBytes);
        cudaMalloc(&dX, cols * sizeof(float));
        cudaMalloc(&dOut, sizeof(float));
        uint8_t *dQ8 = nullptr;
        cudaMalloc(&dQ8,
                   static_cast<uint64_t>(blocksPerRow) * kQ8_0x32_STRIDE);
        cudaMemcpy(dW, wRow, rowBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dX, x, cols * sizeof(float), cudaMemcpyHostToDevice);
        kQuantizeQ8_0x32<<<blocksPerRow, 32, 0, g_stream>>>(dX, dQ8, cols);
        kQGemvQ8_0xQ8K<<<1, dim3(32, 1), 0, g_stream>>>(
                dW, dQ8, dOut, 1, blocksPerRow, rowBytes);
        cudaStreamSynchronize(g_stream);
        float result = 0.0f;
        cudaMemcpy(&result, dOut, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dW);
        cudaFree(dX);
        cudaFree(dOut);
        cudaFree(dQ8);
        return result;
    }

    // Host-side diagnostic: run ONE weight row through the 32-wide legacy-block
    // float dequant-dot GEMV (kQGemvSmall32<Q5_0/Q8_0/IQ4_NL>) and return the
    // scalar row result.  Used by unit tests to verify the GPU GEMV for the
    // 32-wide legacy quant types reproduces the CPU float reference.
    float GPUModel::debugQGemvRow(uint32_t type, const uint8_t *wRow,
                                  uint32_t blocksPerRow, const float *x,
                                  uint32_t cols, uint32_t rowBytes) {
        // Route through the SAME launchQGemv dispatch the real forward uses
        // (32-wide legacy blocks via kQGemvSmall32, 256-wide K-quants via
        // kQGemv, integer Q8_32 / Q8_K paths where the CPU contract demands
        // them), so the unit-test row driver cannot diverge from production.
        // The nBlocks=1 grid covers the single requested row.
        uint8_t *dW = nullptr;
        float *dX = nullptr, *dOut = nullptr;
        cudaMalloc(&dW, rowBytes);
        cudaMalloc(&dX, cols * sizeof(float));
        cudaMalloc(&dOut, sizeof(float));
        cudaMemcpy(dW, wRow, rowBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dX, x, cols * sizeof(float), cudaMemcpyHostToDevice);
        float *dO = dOut;
        launchQGemv(type, dW, dX, dO, 1, cols, rowBytes, blocksPerRow, q8k_,
                    q8kBytes_);
        cudaStreamSynchronize(g_stream);
        float result = 0.0f;
        cudaMemcpy(&result, dOut, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dW);
        cudaFree(dX);
        cudaFree(dOut);
        return result;
    }

    // Whether THIS device exposes the int8 m8n8k16 tensor-core mma the MMQ
    // kernel needs (Turing sm_75+).  Pre-Turing (sm_70/sm_72) has no int8 mma
    // at all.  (The kernel itself is compiled only for archs in the gencode
    // list; this is the runtime half of the gate.)
    static bool mmqS8Supported() {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   dev) != cudaSuccess)
            return false;
        if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                   dev) != cudaSuccess)
            return false;
        return (major > 7) || (major == 7 && minor >= 5);
    }

    // Host-side diagnostic: the int8 tensor-core MMQ GEMM/GEMV.  Packs nothing
    // itself -- the caller supplies Int8Pack::packMma's B cells + scales -- so
    // the unit test can compare the mma kernel against a reference computed on
    // the SAME packed weights and the SAME GPU-quantized activations.
    bool GPUModel::debugMmqGemm(const int8_t *wPacked, const float *wScales,
                                uint32_t rows, uint32_t cols, const float *x,
                                uint32_t seq, float *out, uint8_t *outXq) {
        if (!mmqS8Supported()) return false;
        if (cols % 32u != 0u) return false;// a k16-tile must not straddle groups
        const uint32_t n8Tiles = (rows + 7u) / 8u;
        const uint32_t k16Tiles = (cols + 15u) / 16u;
        const uint32_t blocksPerRow = (cols + 255u) / 256u;
        const size_t bBytes = static_cast<size_t>(n8Tiles) * k16Tiles * 128;
        const size_t scBytes = static_cast<size_t>(n8Tiles) * k16Tiles * 8;
        const size_t xqBytes = static_cast<size_t>(seq) * blocksPerRow *
                               kQ8K_STRIDE;

        int8_t *dW = nullptr;
        float *dSc = nullptr, *dX = nullptr, *dY = nullptr;
        uint8_t *dXq = nullptr;
        cudaMalloc(&dW, bBytes);
        cudaMalloc(&dSc, scBytes * sizeof(float));
        cudaMalloc(&dX, static_cast<size_t>(seq) * cols * sizeof(float));
        cudaMalloc(&dXq, xqBytes);
        cudaMalloc(&dY, static_cast<size_t>(seq) * rows * sizeof(float));
        cudaMemcpy(dW, wPacked, bBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dSc, wScales, scBytes * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(dX, x, static_cast<size_t>(seq) * cols * sizeof(float),
                   cudaMemcpyHostToDevice);
        // Quantize each token's activation with the EXACT kQuantizeQ8K the
        // production paths use (bit-exact with GGMLDequantize::quantizeQ8K).
        for (uint32_t s = 0; s < seq; ++s) {
            kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(
                    dX + static_cast<size_t>(s) * cols,
                    dXq + static_cast<size_t>(s) * blocksPerRow * kQ8K_STRIDE,
                    cols);
        }
        const dim3 grid(n8Tiles, (seq + 63u) / 64u);
        kMmqGemm_8x8x16<<<grid, 256, 0, g_stream>>>(
                dW, dSc, reinterpret_cast<const int8_t *>(dXq), dY,
                static_cast<int>(seq), static_cast<int>(rows),
                static_cast<int>(cols), static_cast<int>(k16Tiles),
                static_cast<int>(blocksPerRow * kQ8K_STRIDE));
        cudaStreamSynchronize(g_stream);
        cudaMemcpy(out, dY, static_cast<size_t>(seq) * rows * sizeof(float),
                   cudaMemcpyDeviceToHost);
        if (outXq != nullptr)
            cudaMemcpy(outXq, dXq, xqBytes, cudaMemcpyDeviceToHost);
        cudaFree(dW);
        cudaFree(dSc);
        cudaFree(dX);
        cudaFree(dXq);
        cudaFree(dY);
        return true;
    }

    // Host-side diagnostic: run kQuantizeQ8K on x and copy the quantized
    // 256-wide Q8_K block bytes (kQ8K_STRIDE each) back to the host so a unit
    // test can diff the GPU activation quantization against the CPU's
    // GGMLDequantize::quantizeQ8K byte-for-byte (the Q4K/Q6K GEMV parity).
    // Also fills outBlk[b] with the per-block contribution the GPU Q4K/Q6K
    // GEMVs would accumulate, recomputed on the HOST from the GPU's OWN
    // quantized bytes using the SAME float trees as kxQ8KBlockDot (Q4K fma
    // chains for lane 0; Q6K per-lane offset+chunk FMAs accumulated in the
    // shfl hsum order).
    void GPUModel::debugQ8KBlocks(uint32_t type, const uint8_t *wRow,
                                  uint32_t blocksPerRow, const float *x,
                                  uint32_t cols, float *outBlk,
                                  uint8_t *outQuant) {
        const uint32_t blkBytes = type == kTypeQ4K ? kQ4K_BYTES : kQ6K_BYTES;
        const uint32_t rowBytes = blocksPerRow * blkBytes;
        float *dX = nullptr;
        uint8_t *dQ8 = nullptr, *dW = nullptr;
        cudaMalloc(&dX, cols * sizeof(float));
        cudaMalloc(&dQ8, static_cast<uint64_t>(blocksPerRow) * kQ8K_STRIDE);
        cudaMalloc(&dW, rowBytes);
        cudaMemcpy(dX, x, cols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dW, wRow, rowBytes, cudaMemcpyHostToDevice);
        kQuantizeQ8K<<<blocksPerRow, 256, 0, g_stream>>>(dX, dQ8, cols);
        cudaStreamSynchronize(g_stream);
        cudaMemcpy(outQuant, dQ8,
                   static_cast<uint64_t>(blocksPerRow) * kQ8K_STRIDE,
                   cudaMemcpyDeviceToHost);
        // Host recompute of the per-block float contributions from the GPU's
        // own quantized bytes (single token: q8 block b at outQuant+b*stride).
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
            const uint8_t *blkW = wRow + static_cast<uint64_t>(b) * blkBytes;
            const uint8_t *blkA =
                    outQuant + static_cast<uint64_t>(b) * kQ8K_STRIDE;
            const int8_t *y8 = reinterpret_cast<const int8_t *>(blkA);
            const int16_t *bsums =
                    reinterpret_cast<const int16_t *>(blkA + kQ8K_B);
            const float d8 = *reinterpret_cast<const float *>(blkA + kQ8K_D);
            if (type == kTypeQ4K) {
                // Q4K: exact 32-col integer dots + the 3-FMA scalar chain.
                float running = 0.0f;
                for (uint32_t i = 0; i < 8; ++i) {
                    const float d = __half2float(
                            *reinterpret_cast<const __half *>(blkW));
                    const float dmin = __half2float(
                            *reinterpret_cast<const __half *>(blkW + 2));
                    const uint8_t *scales = blkW + 4;
                    const uint8_t *qsv = blkW + 16;
                    uint8_t sc, m;
                    if (i < 4) {
                        sc = scales[i] & 63u;
                        m = scales[i + 4] & 63u;
                    } else {
                        sc = static_cast<uint8_t>(
                                (scales[i + 4] & 0xFu) |
                                ((scales[i - 4] >> 6) << 4));
                        m = static_cast<uint8_t>(
                                (scales[i + 4] >> 4) |
                                ((scales[i] >> 6) << 4));
                    }
                    const float dsv = d * static_cast<float>(sc);
                    const float mnv = dmin * static_cast<float>(m);
                    int s32q = 0;
                    const uint8_t *qsb = qsv + 32u * (i >> 1u);
                    for (uint32_t j = 0; j < 32u; ++j) {
                        const uint8_t qb = qsb[j];
                        const uint8_t wv = ((i & 1u) == 0) ? (qb & 0xFu)
                                                           : (qb >> 4);
                        s32q += static_cast<int>(wv) *
                                static_cast<int>(y8[i * 32u + j]);
                    }
                    const float bi = static_cast<float>(
                            static_cast<int>(bsums[2 * i]) +
                            static_cast<int>(bsums[2 * i + 1]));
                    running = fmaf(dsv, static_cast<float>(s32q), running);
                    // NOTE: the min term is SUBTRACTED (the GPU kernel computes
                    // d8*(ms - mns)); a +mnv here was a harness sign bug that
                    // made gpuBlkSerialSumMatchesGpuKernel a red herring.
                    running = fmaf(-mnv, bi, running);
                }
                outBlk[b] = fmaf(d8, running, 0.0f);
            } else {
                // Q6K: per-lane offset + chunk FMAs, then the shfl hsum tree.
                float running[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                const float d = __half2float(
                        *reinterpret_cast<const __half *>(blkW + 208));
                const int8_t *sc = reinterpret_cast<const int8_t *>(blkW + 192);
                const float kOff = (-32.0f * d) * d8;
                const float k2 = d * d8;
                for (uint32_t lane = 0; lane < 8; ++lane) {
                    const int bprod =
                            static_cast<int>(sc[2 * lane]) *
                                    static_cast<int>(bsums[2 * lane]) +
                            static_cast<int>(sc[2 * lane + 1]) *
                                    static_cast<int>(bsums[2 * lane + 1]);
                    float acc = kOff * static_cast<float>(bprod);
                    for (uint32_t c = 0; c < 2; ++c) {
                        int sumi = 0;
                        for (uint32_t s = 0; s < 4; ++s) {
                            const uint32_t idx = 4 * c + s;
                            const int sca = (lane < 4)
                                                    ? static_cast<int>(sc[2 * idx])
                                                    : static_cast<int>(sc[2 * idx + 1]);
                            int acc4 = 0;
                            for (uint32_t k = 0; k < 4; ++k) {
                                const uint32_t colp = 4 * lane + k;
                                const uint8_t qb =
                                        blkW[32u * (idx & 1u) +
                                             64u * (idx >> 2u) + colp];
                                const uint8_t low = ((idx & 2u) == 0)
                                                            ? (qb & 0xFu)
                                                            : (qb >> 4);
                                const uint8_t qhb =
                                        blkW[128u + 32u * (idx >> 2u) + colp];
                                const uint8_t hi = static_cast<uint8_t>(
                                        (qhb >> (2u * (idx & 3u))) & 3u);
                                const int wv = static_cast<int>(low | (hi << 4));
                                acc4 += wv * static_cast<int>(
                                                     y8[idx * 32u + colp]);
                            }
                            sumi += sca * acc4;
                        }
                        acc = fmaf(k2, static_cast<float>(sumi), acc);
                    }
                    running[lane] = acc;
                }
                const float p = running[0] + running[4];
                const float q = p + (running[1] + running[5]);
                outBlk[b] = q + ((running[2] + running[6]) +
                                 (running[3] + running[7]));
            }
        }
        cudaFree(dX);
        cudaFree(dQ8);
        cudaFree(dW);
    }

    // Host-side diagnostic: run kQuantizeQ8_0x32 on x and copy the quantized
    // block bytes (kQ8_0x32_STRIDE each) back to the host so a unit test can
    // diff the GPU activation quantization against a host-side scalar copy of
    // the CPU's quantize_row_q8_0 loop block by block.
    void GPUModel::debugQ8_0Blocks(const uint8_t *wRow, uint32_t blocksPerRow,
                                   const float *x, uint32_t cols,
                                   float *outBlk, uint8_t *outQuant,
                                   float *outTrace) {
        const uint32_t rowBytes = blocksPerRow * kQ8_0_BYTES;
        float *dX = nullptr;
        uint8_t *dQ8 = nullptr, *dW = nullptr;
        float *dTrace = nullptr, *dOut = nullptr;
        cudaMalloc(&dX, cols * sizeof(float));
        cudaMalloc(&dQ8,
                   static_cast<uint64_t>(blocksPerRow) * kQ8_0x32_STRIDE);
        cudaMalloc(&dW, rowBytes);
        cudaMemcpy(dX, x, cols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dW, wRow, rowBytes, cudaMemcpyHostToDevice);
        kQuantizeQ8_0x32<<<blocksPerRow, 32, 0, g_stream>>>(dX, dQ8, cols);
        if (outTrace != nullptr) {
            // Launch the tracing twin so the kernel's OWN per-block cumulative
            // sums (not the host recompute) can be diffed block-by-block.
            const uint32_t traceN = 2u * blocksPerRow + 16u;
            cudaMalloc(&dTrace, traceN * sizeof(float));
            cudaMalloc(&dOut, sizeof(float));
            kQGemvQ8_0xQ8KTrace<<<1, dim3(32, 1), 0, g_stream>>>(
                    dW, dQ8, dOut, dTrace, 1, blocksPerRow, rowBytes);
        }
        cudaStreamSynchronize(g_stream);
        // The trace kernel writes directly in the documented layout:
        // [0..blocksPerRow) cumulative sums, [blocksPerRow+g] block-0 t[g],
        // [blocksPerRow+8] block-0 blk; [2*blocksPerRow+g] block-1 s[g];
        // [2*blocksPerRow+8] block-1 blk.  Raw copy to the host, no reshaping.
        if (outTrace != nullptr) {
            const uint32_t traceN = 2u * blocksPerRow + 16u;
            cudaMemcpy(outTrace, dTrace, traceN * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
        if (dTrace != nullptr) cudaFree(dTrace);
        if (dOut != nullptr) cudaFree(dOut);
        if (dW != nullptr) cudaFree(dW);
        cudaMemcpy(outQuant, dQ8,
                   static_cast<uint64_t>(blocksPerRow) * kQ8_0x32_STRIDE,
                   cudaMemcpyDeviceToHost);
        // Recompute the per-block float contributions on the host using the
        // SAME float tree kQGemvQ8_0xQ8K runs (from the GPU's own quantized
        // bytes) so a test can compare against the CPU AVX2 per-block sums.
        for (uint32_t b = 0; b < blocksPerRow; ++b) {
            const uint8_t *blkW =
                    wRow + static_cast<uint64_t>(b) * kQ8_0_BYTES;
            const uint8_t *blkA =
                    outQuant + static_cast<uint64_t>(b) * kQ8_0x32_STRIDE;
            const float d = __half2float(*(const __half *) (blkW + 0));
            const float d8 = *reinterpret_cast<const float *>(blkA);
            if (d8 == 0.0f) {
                outBlk[b] = 0.0f;
                continue;
            }
            // Per-4-group integer sums (mirror of the kernel's shfl logic).
            int s[8];
            for (uint32_t g = 0; g < 8; ++g) {
                int c = 0;
                for (uint32_t j = 0; j < 4; ++j) {
                    const int wq = static_cast<int>(static_cast<int8_t>(
                            blkW[2u + 4u * g + j]));
                    const int xq = static_cast<int>(static_cast<int8_t>(
                            blkA[4u + 4u * g + j]));
                    c += xq * wq;
                }
                s[g] = c;
            }
            const float scale = d * d8;
            float t[8];
            for (uint32_t g = 0; g < 8; ++g) t[g] = scale * static_cast<float>(s[g]);
            const float u0 = t[0] + t[4];
            const float u1 = t[1] + t[5];
            const float u2 = t[2] + t[6];
            const float u3 = t[3] + t[7];
            const float v0 = u0 + u1;
            const float v1 = u2 + u3;
            outBlk[b] = v0 + v1;
        }
        cudaFree(dX);
        cudaFree(dQ8);
    }

}// namespace tinycoder::gpu

#endif// USE_CUDA
