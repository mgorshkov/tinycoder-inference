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

#include "Model.hpp"

#ifdef USE_CUDA

#include "GGUFLoader.hpp"
#include "GPUCompute.hpp"
#include "ModelInternal.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace tinycoder {

    using namespace gpu;

    namespace {

        uint32_t typeBlockBytes(uint32_t type) {
            switch (type) {
                case GGML_TYPE_Q5_0:
                    return 22;
                case GGML_TYPE_Q8_0:
                    return 34;
                case GGML_TYPE_Q2_K:
                    return 84;
                case GGML_TYPE_Q3_K:
                    return 110;
                case GGML_TYPE_Q4_K:
                    return 144;
                case GGML_TYPE_Q6_K:
                    return 210;
                case GGML_TYPE_Q8_K:
                    return 292;
                case GGML_TYPE_IQ2_XS:
                    return 74;// 256 weights in 74 bytes
                case GGML_TYPE_IQ3_XXS:
                    return 98;// 256 weights in 98 bytes
                case GGML_TYPE_IQ3_S:
                    return 110;// 256 weights in 110 bytes
                case GGML_TYPE_IQ2_S:
                    return 82;// 256 weights in 82 bytes
                case GGML_TYPE_IQ2_XXS:
                    return 66;// 256 weights in 66 bytes
                case GGML_TYPE_IQ4_NL:
                    return 18;// 32 weights in 18 bytes
                case GGML_TYPE_IQ4_XS:
                    return 136;// 256 weights in 136 bytes
                case GGML_TYPE_Q5_K:
                    return 176;// 256 weights in 176 bytes
                default:
                    return 0;
            }
        }

        // Blocks per row for a quant type and row width (mirrors the layout the
        // GPU kernels expect).  Q5_0/Q8_0 use 32-wide legacy blocks; the K-quants
        // use 256-wide blocks.
        uint32_t blocksPerRowFor(uint32_t type, uint32_t cols) {
            const uint32_t bs = ggmlBlockSize(type);
            return bs ? (cols + bs - 1) / bs : 0;
        }

        // The GPU offload engine's kernels are specialized per quant type:
        //   * kQGemv (weight GEMV) supports Q5_0/Q8_0 + the K-quants
        //     Q2_K/Q3_K/Q4_K/Q6_K.
        //   * kEmbedDequant (token embedding) supports Q5_0/Q8_0 + Q2_K.
        // Both Q5_0 and Q8_0 are 32-wide legacy blocks (22 B / 34 B) handled by
        // the kQGemvSmall32 / kEmbedDequantSmall32 kernels.
        bool supportedWeightType(uint32_t type) {
            switch (type) {
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ4_XS:
                case GGML_TYPE_IQ4_NL:
                    return true;
                default:
                    return false;
            }
        }

        bool supportedEmbeddingType(uint32_t type) {
            return type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q5_0 ||
                   type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ3_S ||
                   type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
                   type == GGML_TYPE_IQ4_NL;
        }

        DeviceMatrix toDeviceMatrix(const QuantizedMatrix &m) {
            DeviceMatrix dm;
            if (m.empty()) return dm;
            dm.rows = m.rows;
            dm.cols = m.cols;
            dm.type = m.type;
            // F32 weights (qwen35 ssm_alpha [nVHeads][hidden] and ssm_beta,
            // ssm_conv1d [channels][kernel]) are plain row-major floats.  There
            // is no dense-block encoding: rowBytes = cols*4 and upload() copies
            // rows*rowBytes bytes verbatim.  The GEMV kernel for kTypeF32 reads
            // them directly (kQ35F32Gemv / launchQGemv F32 branch).
            if (m.type == GGML_TYPE_F32) {
                dm.blocksPerRow = 0;
                dm.rowBytes = m.cols * sizeof(float);
                dm.q = const_cast<uint8_t *>(m.data.data());
                return dm;
            }
            dm.blocksPerRow = blocksPerRowFor(m.type, m.cols);
            dm.rowBytes = dm.blocksPerRow * typeBlockBytes(m.type);
            if (dm.rowBytes == 0) {
                dm.empty_ = true;
                return dm;
            }
            // Host pointer captured here; upload() copies the quantized bytes
            // to the device.  NO FP16 twin: the prefill path dequantizes
            // per-layer on-device into a reusable scratch (dequantMatrixF16),
            // so neither host RAM nor VRAM holds a persistent fp16 copy.
            dm.q = const_cast<uint8_t *>(m.data.data());
            return dm;
        }

    }// namespace

    /// @brief GPU offload adapter: mirrors the Model weights to the device.
    /// Owns the FP16 twins and the tinycoder::gpu::GPUModel runtime.
    class ModelGPUAdapter {
    public:
        ModelGPUAdapter() = default;
        ~ModelGPUAdapter() { destroy(); }

        // Worker that assumes mtx_ is ALREADY held (no locking inside).
        bool ensureUploadedLocked(Model &m, std::string &err) {
            if (uploaded_) return true;
            if (gpuUnavailable_) {
                // Latched: a previous upload attempt already failed (e.g. the
                // model does not fit in VRAM).  Retrying would re-run the
                // doomed full ~10 GB upload on EVERY fresh session (kvCache_.pos
                // == 0) and, since the first attempt used to leak its partial
                // allocations, steadily starve even the CPU fallback's host
                // RAM.  Stay on CPU for the rest of the model's lifetime.
                err = "GPU upload previously failed; staying on CPU (latched)";
                return false;
            }
            if (!gpuEnabled()) {
                err = "GPU disabled (opt out with TINYCODER_GPU=0)";
                return false;
            }
            const ModelConfig &cfg = m.config_;
            const auto &layers = m.layers_;
            const auto &emb = m.quantizedEmbeddings_;

            // ---- Quant-type support gate ----
            // Reject models whose tensors use quant types the GPU kernels do not
            // implement, so we never launch a kernel that misreads blocks or
            // writes out of bounds (which surfaces later as an
            // "illegal memory access" on an unrelated CUDA call).  Falling back
            // to CPU here is safe and correct.
            if (!supportedEmbeddingType(emb.type)) {
                err = "GPU offload unsupported: token embedding quant type " +
                      std::to_string(emb.type) +
                      " not implemented by GPU kernels (need Q2_K/Q4_K/Q5_K/"
                      "Q5_0/Q8_0/IQ3_S)";
                return false;
            }
            if (!m.lmHeadTied_ && !m.lmHead_.empty() &&
                !supportedWeightType(m.lmHead_.type)) {
                err = "GPU offload unsupported: LM head quant type " +
                      std::to_string(m.lmHead_.type) +
                      " not implemented by GPU kernels (need Q2_K/Q3_K/Q4_K/Q6_K)";
                return false;
            }
            for (uint32_t L = 0; L < cfg.numLayers; ++L) {
                const auto &w = layers[L];
                const uint32_t types[] = {w.attnQ.type, w.attnK.type,
                                          w.attnV.type, w.attnO.type,
                                          w.ffnGate.type, w.ffnUp.type,
                                          w.ffnDown.type,
                                          w.attnQKV.type, w.attnGate.type,
                                          w.ssmAlphaQ.type, w.ssmBetaQ.type,
                                          w.ssmOut.type,
                                          // qwen35moe MoE FFN tensors.
                                          w.ffnGateInpMoe.type,
                                          w.ffnGateExps.type, w.ffnUpExps.type,
                                          w.ffnDownExpsMoe.type,
                                          w.ffnGateShexp.type, w.ffnUpShexp.type,
                                          w.ffnDownShexp.type};
                for (uint32_t t: types) {
                    if (t == 0) continue;// F32 / empty
                    if (!supportedWeightType(t)) {
                        err = "GPU offload unsupported: layer " +
                              std::to_string(L) + " uses quant type " +
                              std::to_string(t) +
                              " not implemented by GPU kernels (need "
                              "Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ*)";
                        return false;
                    }
                }
            }

            // ---- Build device layer descriptors for the offloaded prefix ----
            // Default: offload ALL layers (full model on GPU). $TINYCODER_NGL
            // overrides with a partial offload count (llama.cpp -ngl style).
            uint32_t nGpu = cfg.numLayers;
            if (cfg.architecture == ARCH_QWEN35MOE) {
                // qwen35moe: FULL offload ONLY — the user's requirement ("make
                // it work without offloading" = no CPU continuation).  The Host
                // dispatch (Model::forward) has no partial-offload continuation
                // for qwen35moe (the CPU converges via computeQwen35MoE only in
                // the fully-CPU path), so TINYCODER_NGL is IGNORED here and all
                // layers stay on the GPU.  (The Auto-NGL estimator below is
                // qwen35-only; a too-large qwen35moe must OOM or fall back to
                // CPU wholesale, never silently drop layers.)
                nGpu = cfg.numLayers;
            } else if (const char *e = std::getenv("TINYCODER_NGL")) {
                int ngl = std::atoi(e);
                if (ngl > 0 && static_cast<uint32_t>(ngl) < nGpu) nGpu = static_cast<uint32_t>(ngl);
            } else if (cfg.architecture == ARCH_QWEN35) {
                // qwen35 is the only architecture whose host-side dispatch
                // (Model::forward) supports the partial-offload continuation,
                // so auto-fit is gated to it.  (qwen2 partial offload would
                // leave the hidden state on the device without a CPU resumer.)
                // No explicit -ngl: auto-fit the layer count to the GPU's free
                // VRAM (llama.cpp -ngl auto style).  Estimate per-layer device
                // residency (quantized weights + KV slots + recurrent conv/GDN
                // state) and pack as many layers as fit next to the embedding.
                // Partial offload skips the separate LM-head upload (the CPU
                // tail computes it), so it is not charged to the budget.
                uint64_t layerBytes = 0;
                const auto addMat = [&](const QuantizedMatrix &m) {
                    if (m.empty()) return;
                    if (m.type == GGML_TYPE_F32) {
                        // F32 weights (ssm_alpha/beta/conv1d) upload as raw
                        // row-major floats (rows*cols*4); K-quants use
                        // width-256 block encoding.
                        layerBytes += static_cast<uint64_t>(m.rows) * m.cols *
                                      sizeof(float);
                        return;
                    }
                    // Blocks per row follows the ACTUAL block width of the
                    // quant type (32-wide legacy blocks for Q5_0/Q8_0/IQ4_NL,
                    // 256-wide for the K-quants) — the same math
                    // toDeviceMatrix/blocksPerRowFor uses for the real upload.
                    // A hard-coded 256-wide estimate under-counts Q8_0 rows
                    // ~7.6x, making auto-fit compute hit>=numLayers and attempt
                    // a doomed full offload that OOMs at upload (seen with
                    // Qwen3.6-27B-Q8_0: "cudaMalloc(attnQ): out of memory").
                    const uint32_t blocks = blocksPerRowFor(m.type, m.cols);
                    layerBytes += static_cast<uint64_t>(m.rows) * blocks *
                                  typeBlockBytes(m.type);
                };
                const uint32_t kvLen = cfg.numKVHeads * cfg.headDim;
                if (cfg.architecture == ARCH_QWEN35) {
                    const uint32_t keyDim = cfg.ssmStateSize * cfg.ssmGroupCount;
                    const uint32_t valueDim = cfg.ssmInnerSize;
                    const uint32_t headV = cfg.ssmInnerSize /
                                           std::max(cfg.ssmTimeStepRank, 1u);
                    const uint32_t convChannels = 2 * keyDim + valueDim;
                    for (uint32_t L = 0; L < cfg.numLayers; ++L) {
                        const auto &w = layers[L];
                        addMat(w.attnQKV);
                        addMat(w.attnGate);
                        addMat(w.ssmAlphaQ);
                        addMat(w.ssmBetaQ);
                        addMat(w.ssmConv1d);
                        addMat(w.ssmOut);
                        addMat(w.ffnGate);
                        addMat(w.ffnUp);
                        addMat(w.ffnDown);
                    }
                    layerBytes = (layerBytes + cfg.numLayers - 1) / cfg.numLayers;
                    layerBytes += static_cast<uint64_t>(cfg.maxSeqLen) * kvLen *
                                  sizeof(float) * 2;
                    layerBytes += static_cast<uint64_t>(cfg.ssmConvKernel - 1) *
                                  convChannels * sizeof(float);
                    // GDN persistent state: [nVHeads][headV][headV] floats.
                    layerBytes += static_cast<uint64_t>(cfg.ssmTimeStepRank) *
                                  headV * headV * sizeof(float);
                } else {
                    for (uint32_t L = 0; L < cfg.numLayers; ++L) {
                        const auto &w = layers[L];
                        addMat(w.attnQ);
                        addMat(w.attnK);
                        addMat(w.attnV);
                        addMat(w.attnO);
                        addMat(w.ffnGate);
                        addMat(w.ffnUp);
                        addMat(w.ffnDown);
                    }
                    layerBytes = (layerBytes + cfg.numLayers - 1) / cfg.numLayers;
                    layerBytes += static_cast<uint64_t>(cfg.maxSeqLen) * kvLen *
                                  sizeof(float) * 2;
                }
                // Token embedding bytes (K-quants: width-256 blocks).
                const uint32_t embBlockBytes = typeBlockBytes(emb.type);
                const uint32_t embBlocksPerRow = blocksPerRowFor(emb.type, emb.hiddenSize);
                const uint64_t embedBytes =
                        static_cast<uint64_t>(cfg.vocabSize) * embBlockBytes * embBlocksPerRow;
                // Free VRAM: authoritative cudaMemGetInfo; conservative fallback.
                int64_t freeVram = 8ll << 30;
                size_t freeBytes = 0, totalBytes = 0;
                if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
                    freeVram = static_cast<int64_t>(freeBytes);
                }
                // Persistent compute scratch (RoPE/MRoPE tables, batched
                // qkv/gdnOut/q35QGate for the max prefill length, fp16 GEMM
                // twin) is NOT charged per layer by the weight/residency
                // estimate.  Reserve a 900 MB fixed overhead and pad the
                // per-layer residency 10% (the manual NGL=25 run succeeded at
                // ~292 MiB/layer vs the ~301 MiB padded estimate here), which
                // lands the fit at ~31 layers on the 11 GB card.  For big
                // prefills the wF16 twin (largest dense matrix, e.g. ffnGate
                // 17408x5120 fp16 = 178 MB) reuses this fixed reservation.
                const int64_t fixedOverhead = 900ll << 20;
                const int64_t budget =
                        freeVram - static_cast<int64_t>(embedBytes) - fixedOverhead;
                if (layerBytes > 0 && budget > 0) {
                    layerBytes = static_cast<uint64_t>(static_cast<double>(layerBytes) * 1.10);
                    uint32_t fit = static_cast<uint32_t>(budget / static_cast<int64_t>(layerBytes));
                    if (fit > 0 && fit < nGpu) {
                        nGpu = fit;
                        std::fprintf(stderr,
                                     "[gpu] auto-ngl: offloading %u of %u layers "
                                     "(~%.1f GB free, %llu B/layer)\n",
                                     nGpu, cfg.numLayers,
                                     static_cast<double>(freeVram) / (1ll << 30),
                                     static_cast<unsigned long long>(layerBytes));
                    }
                }
            }
            std::vector<DeviceLayer> deviceLayers;
            deviceLayers.reserve(nGpu);
            for (uint32_t L = 0; L < nGpu; ++L) {
                const auto &w = layers[L];
                DeviceLayer dl;
                dl.attnQ = toDeviceMatrix(w.attnQ);
                dl.attnK = toDeviceMatrix(w.attnK);
                dl.attnV = toDeviceMatrix(w.attnV);
                dl.attnO = toDeviceMatrix(w.attnO);
                dl.ffnGate = toDeviceMatrix(w.ffnGate);
                dl.ffnUp = toDeviceMatrix(w.ffnUp);
                dl.ffnDown = toDeviceMatrix(w.ffnDown);
                if (cfg.architecture == ARCH_QWEN2) {
                    dl.attnQBias = const_cast<float *>(
                            w.attnQBias.empty() ? nullptr : w.attnQBias.data());
                    dl.attnKBias = const_cast<float *>(
                            w.attnKBias.empty() ? nullptr : w.attnKBias.data());
                    dl.attnVBias = const_cast<float *>(
                            w.attnVBias.empty() ? nullptr : w.attnVBias.data());
                }
                // Qwen35 / Qwen35MoE recurrent/gated-delta-net extras (shared
                // GDN math; qwen35moe is a superset with the MoE FFN below).
                if (cfg.architecture == ARCH_QWEN35 ||
                    cfg.architecture == ARCH_QWEN35MOE) {
                    dl.attnQKV = toDeviceMatrix(w.attnQKV);
                    dl.attnGate = toDeviceMatrix(w.attnGate);
                    dl.ssmAlphaQ = toDeviceMatrix(w.ssmAlphaQ);
                    dl.ssmBetaQ = toDeviceMatrix(w.ssmBetaQ);
                    dl.ssmOut = toDeviceMatrix(w.ssmOut);
                    // ssm_conv1d is F32 [convKernel, convChannels].
                    dl.ssmConv1d = toDeviceMatrix(w.ssmConv1d);
                    dl.ssmABroadcast = const_cast<float *>(
                            w.ssmABroadcast.empty() ? nullptr
                                                    : w.ssmABroadcast.data());
                    dl.ssmDtBiasFull = const_cast<float *>(
                            w.ssmDtBiasFull.empty() ? nullptr
                                                    : w.ssmDtBiasFull.data());
                    dl.ssmNorm = const_cast<float *>(
                            w.ssmNorm.empty() ? nullptr : w.ssmNorm.data());
                    dl.attnQNorm = const_cast<float *>(
                            w.attnQNorm.empty() ? nullptr : w.attnQNorm.data());
                    dl.attnKNorm = const_cast<float *>(
                            w.attnKNorm.empty() ? nullptr : w.attnKNorm.data());
                    dl.postAttnNorm = const_cast<float *>(
                            w.postAttnNorm.empty() ? nullptr
                                                   : w.postAttnNorm.data());
                }
                // Qwen35MoE FFN extras.  The router (ffnGateInpMoe) and the
                // shared-expert router (ffnGateInpShexp) are F32 raw blobs;
                // the per-expert + shared matrices are quantized.  The GPU
                // driver supports the SEPARATE ffn_gate_exps + ffn_up_exps
                // layout only (the fused ffn_gate_up_exps variant is rejected
                // by the type gate above and falls back to CPU).
                if (cfg.architecture == ARCH_QWEN35MOE) {
                    dl.ffnGateInpMoe = toDeviceMatrix(w.ffnGateInpMoe);
                    dl.ffnGateExps = toDeviceMatrix(w.ffnGateExps);
                    dl.ffnUpExps = toDeviceMatrix(w.ffnUpExps);
                    dl.ffnDownExpsMoe = toDeviceMatrix(w.ffnDownExpsMoe);
                    dl.ffnGateShexp = toDeviceMatrix(w.ffnGateShexp);
                    dl.ffnUpShexp = toDeviceMatrix(w.ffnUpShexp);
                    dl.ffnDownShexp = toDeviceMatrix(w.ffnDownShexp);
                    dl.ffnGateInpShexp = toDeviceMatrix(w.ffnGateInpShexp);
                }
                dl.rmsNormAttn = const_cast<float *>(
                        w.rmsNormAttn.empty() ? nullptr : w.rmsNormAttn.data());
                dl.rmsNormFFN = const_cast<float *>(
                        w.rmsNormFFN.empty() ? nullptr : w.rmsNormFFN.data());
                deviceLayers.push_back(dl);
            }

            // ---- Geometry + embedding ----
            gpu::ModelGeometry geom;
            geom.hiddenSize = cfg.hiddenSize;
            geom.intermediateSize = cfg.intermediateSize;
            geom.numLayers = cfg.numLayers;
            geom.numGpuLayers = nGpu;
            geom.numAttentionHeads = cfg.numAttentionHeads;
            geom.numKVHeads = cfg.numKVHeads;
            geom.headDim = cfg.headDim;
            geom.maxSeqLen = cfg.maxSeqLen;
            geom.vocabSize = cfg.vocabSize;
            geom.ropeTheta = cfg.ropeTheta;
            geom.qwen2Bias = (cfg.architecture == ARCH_QWEN2) ? 1u : 0u;
            geom.finalNorm = m.finalNorm_.empty() ? nullptr : m.finalNorm_.data();
            if (cfg.architecture == ARCH_QWEN35 ||
                cfg.architecture == ARCH_QWEN35MOE) {
                geom.architecture = (cfg.architecture == ARCH_QWEN35MOE) ? 2u : 1u;
                geom.attentionKeyLength = cfg.attentionKeyLength;
                geom.attentionValueLength = cfg.attentionValueLength;
                geom.ropeDimensionCount = cfg.ropeDimensionCount;
                std::copy_n(cfg.ropeDimensionSections, 4,
                            geom.ropeDimensionSections);
                geom.fullAttentionInterval = cfg.fullAttentionInterval;
                geom.ssmInnerSize = cfg.ssmInnerSize;
                geom.ssmStateSize = cfg.ssmStateSize;
                geom.ssmGroupCount = cfg.ssmGroupCount;
                geom.ssmTimeStepRank = cfg.ssmTimeStepRank;
                geom.ssmConvKernel = cfg.ssmConvKernel;
                geom.nextnPredictLayers = cfg.nextnPredictLayers;
            }
            // Qwen35MoE expert geometry (ignored by qwen35/dense).
            geom.expertCount = cfg.expertCount;
            geom.expertUsedCount = cfg.expertUsedCount;
            geom.expertFF = cfg.expertFeedForwardLength;
            geom.expertSharedFF = cfg.expertSharedFeedForwardLength;

            // ---- Token embedding (always the embedding matrix) ----
            uint32_t embBlockBytes = typeBlockBytes(emb.type);
            uint32_t embBlocksPerRow = blocksPerRowFor(emb.type, emb.hiddenSize);
            uint32_t embRowBytes = embBlockBytes * embBlocksPerRow;

            // ---- Separate LM head (output.weight), if any ----
            // When the LM head is tied, geom.lmHeadTied == 1 and the engine's
            // logits GEMV uses the uploaded token-embedding matrix. When it is
            // separate, the adapter uploads the quantized output.weight matrix
            // (rows == vocabSize, cols == hiddenSize) into a distinct buffer.
            const void *lmHeadQ = nullptr;
            uint32_t lmHeadType = emb.type;
            uint32_t lmHeadRowBytes = embRowBytes;
            if (!m.lmHeadTied_ && !m.lmHead_.empty()) {
                lmHeadQ = m.lmHead_.data.data();
                lmHeadType = m.lmHead_.type;
                lmHeadRowBytes = typeBlockBytes(m.lmHead_.type) *
                                 blocksPerRowFor(m.lmHead_.type, m.lmHead_.cols);
            }
            geom.lmHeadTied = m.lmHeadTied_ ? 1u : 0u;

            if (!gpu_.upload(deviceLayers, geom, emb.data.data(), emb.type,
                             embRowBytes, lmHeadQ, lmHeadType, lmHeadRowBytes,
                             err)) {
                // Failed to fit.  For qwen35moe, retry ONCE in CPU-expert
                // hybrid mode (llama.cpp `--cpu-moe` style): the 256-expert
                // matrices (~32B of the model's 35B params) stay in host RAM
                // and only the top-8 of 256 experts per token are computed on
                // the CPU, so IQ2_M (~11.5 GB) and Q4_K_M (~22 GB) fit the
                // 11 GB card while the GPU still runs embedding, GDN/attention,
                // post-attn norm, KV cache and the LM head.  IQ1_M fits the
                // full-GPU upload on the first try and never reaches this path.
                if (cfg.architecture == ARCH_QWEN35MOE && !moeCpuEnabled_) {
                    // TINYCODER_DUMP_MOE=1: dump the CPU reference L0 MoE
                    // output (routed + shared) so the GPU fast path's tiny
                    // moeExpertOut can be compared directly.
                    {
                        const char *dnM = std::getenv("TINYCODER_DUMP_MOE");
                        if (dnM != nullptr && dnM[0] != '\0' && dnM[0] != '0') {
                            std::fprintf(stderr,
                                         "[gpu] CPU-expert hybrid retry armed "
                                         "(DUMP_MOE=1)\n");
                        }
                    }
                    std::string hybridErr = err + " -> retrying with MoE "
                                                  "experts offloaded to CPU";
                    std::fprintf(stderr, "[gpu] %s\n", hybridErr.c_str());
                    // Register the CPU-ROUTER-ONLY callback: per GPU layer, the
                    // GPU driver hands us the post-attn norm mirror; we run the
                    // ROUTER on the CPU (fp32 ffnGateInpMoe @ norm — reference
                    // math) + softmax + top-k + renormalization (NO expert
                    // FFNs — those run on the GPU through the per-layer expert
                    // cache built by buildExpertCache).  The per-(token, rank)
                    // (expertIdx, weight) tables are written into the shared
                    // host mirrors the GPU driver reads back.
                    //
                    // The CPU router is what makes batch and single-token
                    // prefills select the SAME experts: the GPU's fp16 prefill
                    // GEMM and its fp32 decode GEMV round the router logits
                    // slightly differently, which can flip a borderline expert
                    // in the top-8, break batch-vs-sequential parity and
                    // diverge from the CPU reference.  The fp32 CPU router is
                    // bit-identical for both paths (and to the pure-CPU path,
                    // which also uses w.ffnGateInpMoe.matMulVec).
                    //
                    // The kernel switch is atomic w.r.t. the adapter mutex, and
                    // the callback is invoked from the GPU driver on the main
                    // thread only, so no locking is needed inside.
                    gpu_.setMoeCpuFn([this, &m](uint32_t layer,
                                                const float *ffnNorm,
                                                float *ffnOut,
                                                uint32_t seqLen) -> bool {
                        const auto &w = m.layers_[layer];
                        // TINYCODER_DUMP_ROUTER=1: dump the first 3 layers'
                        // router selection (norm head) so a parity failure can
                        // be bisected (router tables vs expert FFN).
                        const char *dumpRouter = std::getenv("TINYCODER_DUMP_ROUTER");
                        if (dumpRouter != nullptr && dumpRouter[0] != '\0' &&
                            dumpRouter[0] != '0' && layer < 3 && seqLen >= 1) {
                            std::fprintf(stderr,
                                         "[router] L%u norm[0..4]={%.6f %.6f %.6f "
                                         "%.6f %.6f}\n",
                                         layer, ffnNorm[0], ffnNorm[1], ffnNorm[2],
                                         ffnNorm[3], ffnNorm[4]);
                        }
                        // The GPU driver (GPUCompute.cu qwen35moe hybrid block)
                        // SELECTS the fallback independently: it runs the routed
                        // experts from the device cache ONLY when
                        // `expertCacheBuilt_ && !forceMoeFallback`.  Keep this
                        // callback's contract EXACTLY in sync: when
                        // TINYCODER_MOE_FALLBACK=1 forces the CPU path, the
                        // driver will H2D moeFfnOutHost_ (the fallback branch in
                        // GPUCompute.cu) and therefore needs the FULL routed
                        // experts written here.  In the normal (non-fallback)
                        // mode, the driver consumes the router-only tables.
                        const bool forceMoeFallback =
                                (std::getenv("TINYCODER_MOE_FALLBACK") != nullptr);
                        // ---- Static resident-expert profiling (Option D) ----
                        // TINYCODER_MOE_PROFILE=1: count the routed expert
                        // selections per layer so the adapter can build the
                        // top-K resident policy (only ACTIVE during the profile
                        // phase; the counter vector is empty otherwise).  The
                        // router tables are computed HERE (regardless of which
                        // dispatch branch follows) so recording works in the
                        // fallback branch too (computeQwen35MoEFromLogits uses
                        // LOCAL tables and leaves moeRouterIdxHost_ stale).
                        if (!m.moeUsageCounts_.empty()) {
                            const size_t pn = static_cast<size_t>(seqLen) *
                                              m.config_.expertUsedCount;
                            gpu_.moeRouterIdxHost_.resize(pn);
                            gpu_.moeRouterWgtHost_.resize(pn);
                            m.computeQwen35MoERouterOnly(
                                    ffnNorm, seqLen, m.config_.hiddenSize, w,
                                    gpu_.moeRouterIdxHost_.data(),
                                    gpu_.moeRouterWgtHost_.data());
                            m.recordMoEUsage(layer, gpu_.moeRouterIdxHost_.data(),
                                             seqLen);
                        }
                        // ---- Option D dispatch (static resident-expert split):
                        // the driver set residentHostSlot_ >= 0 when this layer
                        // has resident experts AND Option D is active.  The CPU
                        // computes BOTH the router tables (all ranks, reference
                        // math) AND the NON-resident expert FFNs into the
                        // combined rank-major down table; resident ranks are the
                        // GPU's job (scheduleResidentExperts fills their slots in
                        // the same table on-device).
                        if (gpu_.residentHostSlot_ >= 0) {
                            const size_t n = static_cast<size_t>(seqLen) *
                                             m.config_.expertUsedCount;
                            gpu_.moeRouterIdxHost_.resize(n);
                            gpu_.moeRouterWgtHost_.resize(n);
                            m.computeQwen35MoERouterOnly(
                                    ffnNorm, seqLen, m.config_.hiddenSize, w,
                                    gpu_.moeRouterIdxHost_.data(),
                                    gpu_.moeRouterWgtHost_.data());
                            // Zero the WHOLE combined down table then compute
                            // only the non-resident ranks' outputs.
                            const int32_t slot = gpu_.residentHostSlot_;
                            auto &downTab = gpu_.moeDownTableHost_[slot];
                            std::fill(downTab.begin(), downTab.end(), 0.0f);
                            uint32_t maskWords = 0;
                            const uint64_t *mask = nullptr;
                            if (!m.moeResidentMask_.empty() &&
                                layer < m.moeResidentMask_.size()) {
                                const auto &mv = m.moeResidentMask_[layer];
                                mask = mv.data();
                                maskWords = static_cast<uint32_t>(mv.size());
                            }
                            m.computeQwen35MoENonResident(
                                    ffnNorm, downTab.data(), seqLen,
                                    m.config_.hiddenSize, w,
                                    gpu_.moeRouterIdxHost_.data(),
                                    gpu_.moeRouterWgtHost_.data(), mask,
                                    maskWords);
                            return true;
                        }
                        if (gpu_.expertCacheBuilt() && !forceMoeFallback) {
                            // ---- Router-only (GPU expert cache) ----
                            // The expert FFNs run on the GPU through the cache;
                            // write only the fp32 reference selection tables
                            // into the driver's host mirrors the cache path
                            // reads.  (~µs: a 256x2048 fp32 mat-vec per token.)
                            const size_t n = static_cast<size_t>(seqLen) *
                                             m.config_.expertUsedCount;
                            gpu_.moeRouterIdxHost_.resize(n);
                            gpu_.moeRouterWgtHost_.resize(n);
                            m.computeQwen35MoERouterOnly(
                                    ffnNorm, seqLen, m.config_.hiddenSize, w,
                                    gpu_.moeRouterIdxHost_.data(),
                                    gpu_.moeRouterWgtHost_.data());
                            // TINYCODER_DUMP_MOE=1 + L0: dump the same tables
                            // the GPU fast path consumes (expert id + weight /
                            // rank) so we can compare gate/up/down magnitudes.
                            const char *dnR = std::getenv("TINYCODER_DUMP_MOE");
                            if (dnR != nullptr && dnR[0] != '\0' &&
                                dnR[0] != '0' && layer < 4) {
                                const uint32_t eu = m.config_.expertUsedCount;
                                std::fprintf(stderr,
                                             "[router L%u] norm=%.6f experts=",
                                             layer, ffnNorm[0]);
                                for (uint32_t rr = 0; rr < eu; ++rr) {
                                    const int32_t e = gpu_.moeRouterIdxHost_[rr];
                                    const float wgt = gpu_.moeRouterWgtHost_[rr];
                                    std::fprintf(stderr, "(%d:%.5f)%s", e, wgt,
                                                 rr + 1 < eu ? " " : "");
                                }
                                std::fprintf(stderr, "\n");
                                // CPU reference per-rank (s=0) expert FFN for
                                // DIRECT comparison against the GPU cache
                                // path's `[gpu fast L0]` dump.  moeRouterIdxHost_
                                // IS populated here (router-only/cache mode —
                                // unlike the fallback callback, where it is
                                // empty and this would read garbage).  Repeats
                                // the reference math of computeQwen35MoEFromLogits:
                                //   gate/up rows of expert e (separate layout:
                                //   rows [e*ff, (e+1)*ff) in ffnGateExps /
                                //   ffnUpExps; fused: rows [e*ff*2, ...)), then
                                //   SwiGLU silu(gate)*up, then
                                //   down rows [e*H, (e+1)*H) of ffnDownExpsMoe.
                                // The print format matches `[gpu fast L0]`
                                // (r, e, wgt, post-silu gateUp, post-down) so
                                // the two can be diffed token-by-token.
                                {
                                    const uint32_t ff =
                                            m.config_.expertFeedForwardLength;
                                    const uint32_t H = m.config_.hiddenSize;
                                    const bool fusedE =
                                            !w.ffnGateUpExpsMoe.empty() &&
                                            w.ffnUpExps.empty();
                                    std::vector<float> gv(ff), uv(ff), gg(ff),
                                            dv(H);
                                    const float *x = ffnNorm;// token s=0
                                    for (uint32_t rr = 0; rr < eu; ++rr) {
                                        const int32_t e =
                                                gpu_.moeRouterIdxHost_[rr];
                                        const float wgt =
                                                gpu_.moeRouterWgtHost_[rr];
                                        if (e < 0) continue;
                                        const uint32_t ue =
                                                static_cast<uint32_t>(e);
                                        if (fusedE) {
                                            w.ffnGateUpExpsMoe.matMulVecRows(
                                                    x, ue * ff * 2, ff,
                                                    gv.data());
                                            w.ffnGateUpExpsMoe.matMulVecRows(
                                                    x, ue * ff * 2 + ff, ff,
                                                    uv.data());
                                        } else {
                                            w.ffnGateExps.matMulVecRows(
                                                    x, ue * ff, ff, gv.data());
                                            w.ffnUpExps.matMulVecRows(
                                                    x, ue * ff, ff, uv.data());
                                        }
                                        for (uint32_t i = 0; i < ff; ++i) {
                                            gg[i] = (gv[i] /
                                                     (1.0f + std::exp(-gv[i]))) *
                                                    uv[i];
                                        }
                                        w.ffnDownExpsMoe.matMulVecRows(
                                                gg.data(), ue * H, H,
                                                dv.data());
                                        std::fprintf(
                                                stderr,
                                                "[cpu ref L%u] r=%u e=%d "
                                                "wgt=%.6f "
                                                "gateUp[0..3]={%.6f %.6f %.6f "
                                                "%.6f} down[0..3]={%.6f %.6f "
                                                "%.6f %.6f}\n",
                                                layer, rr, e, wgt, gg[0],
                                                gg[1], gg[2], gg[3], dv[0],
                                                dv[1], dv[2], dv[3]);
                                    }
                                }
                            }
                            return true;
                        }
                        // ---- Fallback: full routed expert FFNs on the CPU ----
                        // buildExpertCache failed (device allocation) OR
                        // TINYCODER_MOE_FALLBACK=1 forced the old --cpu-moe
                        // path: reference router + the 8 routed experts into
                        // ffnOut (the GPU then adds the shared expert +
                        // residual).
                        m.computeQwen35MoEFromLogits(ffnNorm, ffnOut, seqLen,
                                                     m.config_.hiddenSize, w);
                        // TINYCODER_DUMP_MOE=1: dump the CPU reference
                        // per-rank (s=0) gate/up/down for expert 0 to compare
                        // against the GPU cache path dumps.
                        {
                            const char *dnP = std::getenv("TINYCODER_DUMP_MOE");
                            if (dnP != nullptr && dnP[0] != '\0' &&
                                dnP[0] != '0' && layer == 0 && seqLen >= 1) {
                                // NB: moeRouterIdxHost_ is NOT populated in
                                // fallback mode (computeQwen35MoEFromLogits uses
                                // local tables), so only the aggregated ffnOut
                                // dump below is available for comparison.
                            }
                        }
                        // TINYCODER_DUMP_MOE=1: dump the CPU reference L0
                        // routed-expert output (token 0) for direct comparison
                        // against the GPU cache path `[gpu L0 moeOut]`.
                        {
                            const char *dnF = std::getenv("TINYCODER_DUMP_MOE");
                            if (dnF != nullptr && dnF[0] != '\0' &&
                                dnF[0] != '0' && layer == 0 && seqLen >= 1) {
                                std::fprintf(stderr,
                                             "[cpu L0 routed] ffnOut[0..3]={%.6f "
                                             "%.6f %.6f %.6f} norm=%.6f\n",
                                             ffnOut[0], ffnOut[1], ffnOut[2],
                                             ffnOut[3], ffnNorm[0]);
                            }
                        }
                        return true;
                    });
                    moeCpuEnabled_ = true;
                    // upload() releases every partially-allocated device buffer
                    // on failure (GPUCompute.cu failCleanup), so this retry
                    // starts from a clean device.
                    if (gpu_.upload(deviceLayers, geom, emb.data.data(), emb.type,
                                    embRowBytes, lmHeadQ, lmHeadType,
                                    lmHeadRowBytes, err)) {
                        // ---- Expert placement policy (2026-09-15) ----
                        // DEFAULT-ON (2026-09-17): the per-layer expert LRU
                        // cache is BUILT BY DEFAULT now that the correctness
                        // defects are fixed (per-layer down-expert quant types,
                        // wrong down-slice type dispatch, and the device-slot
                        // WAR race — all 12 GPU-parity + keyword tests pass on
                        // Ornith-1.5-35B-Q4_K_M and Qwen3.6-35B-A3B-UD-Q4_K_M
                        // with the cache on).  Earlier the cache thrashed: at
                        // ~44-55% hit rate it streamed ~500-600 MB/token of
                        // expert weights over PCIe, strictly worse than pure
                        // CPU — that made Option B (GPU trunk + CPU experts)
                        // the default.  When the working set fits (hot expert
                        // set, Q4/Q6 experts at ~1.6-2x CPU throughput) the
                        // cache is faster; with the WAR fix it is deterministic.
                        // TINYCODER_MOE_CACHE=0 disables it (Option B);
                        // TINYCODER_MOE_CACHE=1 is a no-op that still enables
                        // it (for scripts/docs that already set it);
                        // TINYCODER_MOE_FALLBACK=1 forces the CPU expert path
                        // even with the cache built.
                        const char *moeCacheEnv =
                                std::getenv("TINYCODER_MOE_CACHE");
                        const bool moeCacheDisabled =
                                moeCacheEnv != nullptr && moeCacheEnv[0] != '\0' &&
                                moeCacheEnv[0] == '0';
                        const bool moeCache = !moeCacheDisabled;
                        if (moeCache) {
                            if (!gpu_.buildExpertCache(err)) {
                                std::fprintf(stderr,
                                             "[gpu] qwen35moe expert cache "
                                             "build failed: %s\n",
                                             err.c_str());
                            }
                        }
                        uploaded_ = true;
                        std::fprintf(stderr,
                                     "[gpu] qwen35moe hybrid upload OK (GPU "
                                     "trunk + CPU experts%s)\n",
                                     moeCache ? ", experts cached on GPU (default)"
                                              : "");
                        // ---- Option D (static resident-expert offload) ----
                        // TINYCODER_MOE_RESIDENT=1: after the hybrid upload
                        // succeeded, run the profiling pass + build the static
                        // top-K resident-expert arenas on the GPU.  Runs with
                        // the adapter mutex held; buildResidentExpertPolicy
                        // must not re-enter the adapter (it uses the CPU path
                        // via forceCpuForward_ + resetCpuKVState).  On any
                        // failure it just disables Option D (Option B stays).
                        buildResidentExpertPolicy(m, err);
                        return true;
                    }
                }
                // Latch the failure.  upload() now releases its partially
                // allocated device buffers (GPUCompute.cu failCleanup), and this
                // latch stops any later fresh session (kvCache_.pos == 0) from
                // re-attempting the doomed upload.  Without both, every
                // re-attempt re-allocates ~10 GB of VRAM and, before the leak
                // fix, leaked it, eventually exhausting host RAM for the CPU
                // fallback.
                gpuUnavailable_ = true;
                return false;
            }
            uploaded_ = true;
            return true;
        }

        bool forward(Model &m, const std::vector<int32_t> &tokens,
                     bool computeAllLogits, float *logitsOut, std::string &err) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!uploaded_) {
                if (!ensureUploadedLocked(m, err)) return false;
            }
            if (m.config_.architecture == ARCH_QWEN35MOE) {
                return gpu_.forwardQwen35MoePrefix(tokens, computeAllLogits,
                                                   logitsOut, err);
            }
            if (m.config_.architecture == ARCH_QWEN35) {
                return gpu_.forwardQwen35Prefix(tokens, computeAllLogits, logitsOut,
                                                err);
            }
            return gpu_.forward(tokens, computeAllLogits, logitsOut, err);
        }

        // ---- Option D: build the static resident-expert policy (all trunks
        // + top-K most-used experts per layer, GPU-resident).  Called ONCE
        // from ensureUploadedLocked AFTER the hybrid retry succeeded (the
        // expert host geometry + moeCpuFn_ callback are live).  Steps:
        //   1. profile: TINYCODER_MOE_PROFILE=1 -> enable m.moeUsageCounts_,
        //      run N CPU forwards over representative prompts, count the
        //      routed expert selections per layer;
        //   2. budget: measure free VRAM, subtract the trunk+shared+overhead,
        //      divide by the per-resident-expert arena bytes to get the max
        //      number of resident experts ACROSS all layers;
        //   3. pick the global top-K most-used experts per layer (the policy:
        //      every layer gets the SAME budget share; layers with highly
        //      concentrated routing keep their hottest experts);
        //   4. store the mask (m.moeResidentMask_) + the id lists, call
        //      gpu_.setResidentPolicy + gpu_.buildResidentArenas + enable
        //      gpu_.residentEnabled_.
        // The profile phase must run withOUT the GPU (the GPU is not yet
        // uploaded — this is right after the hybrid upload, and the callback
        // is armed), so it uses the pure-CPU forward path.
        void buildResidentExpertPolicy(Model &m, std::string &err) {
            if (m.config_.architecture != ARCH_QWEN35MOE) return;
            // Only when explicitly requested (Option D opt-in).
            const char *opt = std::getenv("TINYCODER_MOE_RESIDENT");
            if (opt == nullptr || opt[0] == '\0' || opt[0] == '0') return;
            const uint32_t numLayers = m.config_.numLayers;
            const uint32_t expertCount = m.config_.expertCount;
            if (numLayers == 0 || expertCount == 0) return;
            // ---- Type gate (2026-09-17): Option D is Q8_0-expert-only.
            // The Q4_K/Q6_K resident kernels are bit-exact (verified by
            // GPUCpuCompareTest.Q4K_Q6KKernelParityWithCPU), but measured on
            // Ornith-1.5-35B-Q4_K_M the split resident path is a 4.5x
            // REGRESSION vs Option B (0.32 vs 1.45 tg tok/s): the ~6-8
            // non-resident experts per layer run on the CPU with the
            // per-member scalar path instead of the grouped-batch path.
            // Q4_K_M therefore ALWAYS stays on Option B (grouped-batch CPU
            // experts); bailing here (before the expensive profiling pass)
            // makes TINYCODER_MOE_RESIDENT=1 on a Q4 model a clean, fast
            // no-op that merely prints this notice.
            if (numLayers > 0) {
                const uint32_t gateType = m.layers_[0].ffnGateExps.type;
                if (gateType != GGML_TYPE_Q8_0) {
                    std::fprintf(stderr,
                                 "[resident] expert type %u is not Q8_0: "
                                 "Option D is Q8_0-only (Q4_K/Q6_K residents "
                                 "measured as a 4.5x decode regression vs the "
                                 "grouped-batch Option B); keeping Option B "
                                 "permanently\n",
                                 gateType);
                    return;
                }
            }
            // ---- 1. Profiling pass ----
            // Enable the per-(layer, expert) usage counters.
            m.moeUsageCounts_.assign(numLayers, std::vector<uint64_t>(expertCount, 0));
            // Representative prompts to exercise the router (code + text mix).
            const std::vector<std::string> profilePrompts = {
                    "def fibonacci(n):\n    if n <= 1:\n        return n\n"
                    "    return fibonacci(n-1) + fibonacci(n-2)",
                    "The quadratic formula gives the roots of"
                    " ax^2+bx+c=0 as x = (-b +/- sqrt(b^2-4ac))/(2a).",
                    "class Graph:\n    def __init__(self):\n        "
                    "self.adj = {}\n    def add_edge(self, u, v):",
                    "Quantum entanglement is a physical phenomenon where"
                    " particles interact in ways such that the quantum state",
                    "static void quicksort(int *a, int lo, int hi) {\n"
                    "    if (lo >= hi) return;\n    int p = partition(a, lo, hi);",
            };
            // Run each prompt's first ~32 tokens through the pure-CPU path.
            // The CPU MoE path (forwardQwen35Layer -> computeQwen35MoE) records
            // the routed selections into moeUsageCounts_ directly whenever the
            // counters are armed — no GPU routing, no adapter re-entry.  We are
            // INSIDE ensureUploadedLocked (mtx_ HELD), so:
            //   * we must NOT call Model::clearKVCache() (it would call
            //     gpuClearKV -> adapter->clearKVCache -> re-lock mtx_ -> deadlock);
            //     instead poke the CPU-side cache only (resetCpuKVState);
            //   * we must NOT let Model::forward route into the GPU (adapter
            //     re-entry -> deadlock); forceCpuForward_ latches the CPU path.
            m.forceCpuForward_ = true;
            for (const auto &prompt: profilePrompts) {
                if (prompt.empty()) continue;
                std::vector<int32_t> toks = m.tokenize(prompt);
                if (toks.size() > 32) toks.resize(32);
                if (toks.empty()) continue;
                m.resetCpuKVState();
                (void) m.forward(toks, false);
            }
            m.forceCpuForward_ = false;
            m.resetCpuKVState();
            // ---- 2. VRAM budget ----
            // Ask the driver what the hybrid upload + arenas leave free.  The
            // GPUModel exposes no free-memory query; use cudaMemGetInfo here
            // (same context as the driver).
            size_t freeBytes = 0, totalBytes = 0;
            cudaError_t ce = cudaMemGetInfo(&freeBytes, &totalBytes);
            if (ce != cudaSuccess) {
                std::fprintf(stderr,
                             "[resident] cudaMemGetInfo failed: %s — "
                             "disabling Option D\n",
                             cudaGetErrorString(ce));
                m.moeUsageCounts_.clear();
                return;
            }
            // Per-resident-expert arena bytes = 2*gateSlice + downSlice
            // (gate/up [expertFF][H], down [H][expertFF]).
            const uint32_t expertFF = m.config_.expertFeedForwardLength;
            (void) expertFF;
            // Per-resident-expert arena bytes from the driver's recorded hybrid
            // geometry (expertRowBytesGate_/expertRowBytesDown_).
            const uint64_t perExpert = gpu_.residentPerExpertBytes();
            if (perExpert == 0) {
                m.moeUsageCounts_.clear();
                std::fprintf(stderr,
                             "[resident] no resident geometry — disabling\n");
                return;
            }
            // Keep 900 MB + 25% headroom for the trunk + shared expert + the
            // combine tables; the remainder funds the arenas.
            const uint64_t budget =
                    (freeBytes > static_cast<uint64_t>(1024ull * 1024 * 900))
                            ? (freeBytes - static_cast<uint64_t>(1024ull * 1024 * 900)) *
                                      3u / 4u
                            : 0;
            if (budget == 0) {
                m.moeUsageCounts_.clear();
                std::fprintf(stderr,
                             "[resident] VRAM budget exhausted — disabling\n");
                return;
            }
            const uint64_t maxResidentTotal = std::max<uint64_t>(budget / perExpert, 1);
            // ---- 3. Policy: distribute the budget across layers.  Each
            // layer gets floor(maxResidentTotal/numLayers) resident experts;
            // the per-layer top-K are the most-used experts from the profile.
            const uint32_t perLayerBudget = static_cast<uint32_t>(std::max<uint64_t>(
                    maxResidentTotal / numLayers, 1));
            std::vector<std::vector<uint32_t>> ids(numLayers);
            m.moeResidentMask_.assign(numLayers,
                                      std::vector<uint64_t>(0));
            const uint32_t maskWords = (expertCount + 63u) / 64u;
            uint64_t totalResident = 0;
            for (uint32_t L = 0; L < numLayers; ++L) {
                const auto &counts = m.moeUsageCounts_[L];
                // Rank experts by usage desc (stable by id for ties).
                std::vector<std::pair<uint64_t, uint32_t>> scored;
                scored.reserve(expertCount);
                for (uint32_t e = 0; e < expertCount; ++e) {
                    scored.emplace_back(counts[e], e);
                }
                std::partial_sort(scored.begin(),
                                  scored.begin() +
                                          std::min<uint32_t>(perLayerBudget,
                                                             expertCount),
                                  scored.end(),
                                  [](const auto &a, const auto &b) {
                                      if (a.first != b.first)
                                          return a.first > b.first;
                                      return a.second < b.second;
                                  });
                const uint32_t take = std::min<uint32_t>(perLayerBudget,
                                                         expertCount);
                auto &layerIds = ids[L];
                layerIds.reserve(take);
                m.moeResidentMask_[L].assign(maskWords, 0);
                for (uint32_t i = 0; i < take; ++i) {
                    const uint32_t e = scored[i].second;
                    layerIds.push_back(e);
                    m.moeResidentMask_[L][e / 64u] |=
                            (uint64_t(1) << (e % 64u));
                }
                totalResident += take;
            }
            // ---- 4. Store + build on the driver ----
            gpu_.setResidentPolicy(ids);
            std::string berr;
            if (!gpu_.buildResidentArenas(berr)) {
                std::fprintf(stderr, "[resident] buildResidentArenas: %s\n",
                             berr.c_str());
                m.moeUsageCounts_.clear();
                return;
            }
            gpu_.residentEnabled_ = true;
            std::fprintf(stderr,
                         "[resident] Option D active: %llu resident experts "
                         "across %u layers (budget %.0f MB, %llu bytes/expert, "
                         "per-layer cap %u)\n",
                         static_cast<unsigned long long>(totalResident), numLayers,
                         static_cast<double>(budget) / (1024.0 * 1024.0),
                         static_cast<unsigned long long>(perExpert),
                         perLayerBudget);
            m.moeUsageCounts_.clear();
            (void) err;
        }

        // Copy the GPU hidden state (partial offload) back to the host so the
        // CPU can continue the remaining layers.
        bool copyHiddenOut(float *hiddenOut, uint32_t seqLen, std::string &err) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!uploaded_) return false;
            return gpu_.copyHiddenOut(hiddenOut, seqLen, err);
        }

        void clearKVCache() {
            std::lock_guard<std::mutex> lk(mtx_);
            if (uploaded_) gpu_.clearKVCache();
        }

        // Public entry for explicit pre-upload (Model::gpuUploadIfEnabled).
        bool uploadIfEnabled(Model &m, std::string &err) {
            std::lock_guard<std::mutex> lk(mtx_);
            return ensureUploadedLocked(m, err);
        }

        void destroy() {
            std::lock_guard<std::mutex> lk(mtx_);
            if (uploaded_) {
                gpu_.~GPUModel();
                new (&gpu_) GPUModel();
                uploaded_ = false;
            }
        }

        std::mutex mtx_;
        bool uploaded_ = false;
        // Latched once an upload attempt fails (e.g. model exceeds VRAM).  The
        // adapter then stays on the CPU path for the model's lifetime instead of
        // re-attempting the doomed upload on every fresh session.
        bool gpuUnavailable_ = false;
        // Set once the qwen35moe CPU-expert hybrid (--cpu-moe style) retry is
        // armed: the callback is registered on gpu_ and the expert matrices are
        // NOT uploaded.  Guards the retry from firing more than once.
        bool moeCpuEnabled_ = false;
        gpu::GPUModel gpu_;
    };

    // ---- Model::GPU integration hooks (implemented here) ----

    // Opaque ModelGPUState definition (declared in Model.hpp).
    struct Model::ModelGPUState {
        ModelGPUAdapter *adapter = nullptr;
        ~ModelGPUState() {
            delete adapter;
            adapter = nullptr;
        }
    };

    // (ModelGPUState's own destructor is defined in class above.)

    bool Model::gpuForward(const std::vector<int32_t> &tokens, bool computeAllLogits,
                           float *logitsOut, std::string *errMsg) {
        if (!gpuState_) gpuState_ = new ModelGPUState();
        if (!gpuState_->adapter) gpuState_->adapter = new ModelGPUAdapter();
        std::string err;
        bool ok = gpuState_->adapter->forward(*this, tokens, computeAllLogits,
                                              logitsOut, err);
        if (errMsg && !err.empty()) *errMsg = std::move(err);
        return ok;
    }

    bool Model::gpuUploadIfEnabled(std::string *errMsg) {
        if (!gpu::gpuEnabled()) return false;
        if (!gpuState_) gpuState_ = new ModelGPUState();
        if (!gpuState_->adapter) gpuState_->adapter = new ModelGPUAdapter();
        std::string err;
        bool ok = gpuState_->adapter->uploadIfEnabled(*this, err);
        if (errMsg && !err.empty()) *errMsg = std::move(err);
        return ok;
    }

    void Model::gpuClearKV() {
        if (gpuState_ && gpuState_->adapter) {
            gpuState_->adapter->clearKVCache();
        }
    }

    uint32_t Model::gpuNGpuLayers() const {
        if (!gpuState_ || !gpuState_->adapter) return 0;
        std::lock_guard<std::mutex> lk(gpuState_->adapter->mtx_);
        return gpuState_->adapter->gpu_.numGpuLayers();
    }

    bool Model::gpuCopyHiddenOut(float *hiddenOut, uint32_t seqLen,
                                 std::string *errMsg) {
        if (!gpuState_ || !gpuState_->adapter) return false;
        std::string err;
        bool ok = gpuState_->adapter->copyHiddenOut(hiddenOut, seqLen, err);
        if (errMsg && !err.empty()) *errMsg = std::move(err);
        return ok;
    }

    void Model::gpuShutdown() {
        if (gpuState_) {
            delete gpuState_;
            gpuState_ = nullptr;
        }
    }

}// namespace tinycoder

#endif// USE_CUDA
