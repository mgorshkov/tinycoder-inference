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

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
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
                    return true;
                default:
                    return false;
            }
        }

        bool supportedEmbeddingType(uint32_t type) {
            return type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q5_0 ||
                   type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ3_S ||
                   type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
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
                    const uint32_t blocks = (m.cols + 255) / 256;
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
                    std::string hybridErr = err + " -> retrying with MoE "
                                                  "experts offloaded to CPU";
                    std::fprintf(stderr, "[gpu] %s\n", hybridErr.c_str());
                    // Register the CPU-expert callback: per GPU layer, the GPU
                    // driver hands us the post-attn norm mirror; we run the
                    // ROUTER on the CPU (fp32 ffnGateInpMoe @ norm — reference
                    // math) + softmax + top-k + renormalization + the routed
                    // expert FFNs (computeQwen35MoEFromLogits, ThreadPool-
                    // parallel over token x expert-rank pairs).
                    //
                    // The CPU router is what makes batch and single-token
                    // prefills select the SAME experts: the GPU's fp16 prefill
                    // GEMM and its fp32 decode GEMV round the router logits
                    // slightly differently, which can flip a borderline expert
                    // in the top-8, break batch-vs-sequential parity and
                    // diverge from the CPU reference.  The fp32 CPU router is
                    // bit-identical for both paths (and to the pure-CPU path,
                    // which also uses w.ffnGateInpMoe.matMulVec).
                    gpu_.setMoeCpuFn([&m](uint32_t layer, const float *ffnNorm,
                                          float *ffnOut, uint32_t seqLen) -> bool {
                        const auto &w = m.layers_[layer];
                        // TEMP-DIAG ($TINYCODER_MOE_VERIFY=1): per-layer sanify
                        // the norm mirror the GPU handoff gave us (finite, sane
                        // RMS scale) to catch D2H/race corruption deterministically.
                        //
                        // Compares top-8 router experts of the SAME position
                        // between the batch (seqLen=29) and sequential phases to
                        // isolate precision-induced routing flips (fp16 prefill
                        // GEMM vs fp32 decode GEMV feeding the CPU router):
                        //   batch: L0..L3, first+last rows (pos 0 and pos 28)
                        //   seq:   L0..L3 of fwd #1 (pos 0) and fwd #29 (pos 28)
                        static thread_local uint32_t tlsDiagBatchDc = 0;
                        static thread_local uint32_t tlsDiagSeqFwd = 0;
                        const bool tlsWantDiag =
                                std::getenv("TINYCODER_MOE_VERIFY") != nullptr &&
                                std::getenv("TINYCODER_MOE_VERIFY")[0] == '1';
                        if (seqLen == 1 && layer == 0) ++tlsDiagSeqFwd;
                        const bool tlsDiagBatch =
                                tlsWantDiag && seqLen > 1 && tlsDiagBatchDc++ < 4;
                        const bool tlsDiagSeq =
                                tlsWantDiag && seqLen == 1 && layer < 4 &&
                                (tlsDiagSeqFwd == 1 || tlsDiagSeqFwd == 29);
                        if (tlsDiagBatch || tlsDiagSeq) {
                            const uint32_t hs = m.config_.hiddenSize;
                            const uint32_t eu = m.config_.expertUsedCount;
                            const uint32_t ec = m.config_.expertCount;
                            auto diagRow = [&](const float *x, const char *lbl) {
                                double acc = 0.0;
                                float mx = -1e30f, mn = 1e30f;
                                for (uint32_t i = 0; i < hs; ++i) {
                                    float v = x[i];
                                    acc += double(v) * v;
                                    mx = std::max(mx, v);
                                    mn = std::min(mn, v);
                                }
                                const double rms = std::sqrt(acc / double(hs));
                                std::vector<float> lg(ec);
                                w.ffnGateInpMoe.matMulVec(x, lg.data());
                                std::vector<std::pair<float, uint32_t>> scored;
                                scored.reserve(ec);
                                for (uint32_t e = 0; e < ec; ++e) scored.push_back({lg[e], e});
                                std::partial_sort(scored.begin(), scored.begin() + eu,
                                                  scored.end(),
                                                  [](const auto &a, const auto &b) {
                                                      return a.first > b.first;
                                                  });
                                std::fprintf(stderr,
                                             "[moe-diag] L=%u %s seqLen=%u rms=%.5f "
                                             "min=%.5f max=%.5f\n",
                                             layer, lbl, seqLen, rms, mn, mx);
                                std::fprintf(stderr, "[moe-diag]   top-%u:",
                                             eu);
                                for (uint32_t r = 0; r < eu; ++r)
                                    std::fprintf(stderr, " %u(%.3f)", scored[r].second,
                                                 scored[r].first);
                                std::fprintf(stderr, "\n");
                            };
                            diagRow(ffnNorm, "firstRow");
                            if (seqLen > 1)
                                diagRow(ffnNorm + size_t(seqLen - 1) * hs, "lastRow");
                        }
                        m.computeQwen35MoEFromLogits(ffnNorm, ffnOut, seqLen,
                                                     m.config_.hiddenSize, w);
                        return true;
                    });
                    moeCpuEnabled_ = true;
                    // upload() releases every partially-allocated device buffer
                    // on failure (GPUCompute.cu failCleanup), so this retry
                    // starts from a clean device.
                    if (gpu_.upload(deviceLayers, geom, emb.data.data(), emb.type,
                                    embRowBytes, lmHeadQ, lmHeadType,
                                    lmHeadRowBytes, err)) {
                        uploaded_ = true;
                        std::fprintf(stderr,
                                     "[gpu] qwen35moe hybrid upload OK "
                                     "(experts on CPU)\n");
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
