/*
 * TinyCoder CUDA GPU offload engine -- forward & graph
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

    // ===== qwen35moe hybrid, dense forward(), CUDA-graph decode, lifecycle =====
    bool GPUModel::forwardQwen35MoePrefix(const std::vector<int32_t> &tokens,
                                          bool computeAllLogits, float *logitsOut,
                                          std::string &errMsg) {
        if (!allocated_) {
            errMsg = "GPU model not uploaded";
            return false;
        }
        if (tokens.empty()) return true;
        if (geom_.architecture != 2 || geom_.expertCount == 0 ||
            geom_.expertUsedCount == 0) {
            errMsg = "forwardQwen35MoePrefix called for non-qwen35moe geometry";
            return false;
        }
        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t nGpu = geom_.numGpuLayers;
        if (nGpu == 0) nGpu = geom_.numLayers;
        // qwen35moe is FULL-OFFLOAD ONLY (the CPU has no partial-offload
        // continuation for it; TINYCODER_NGL is ignored in the adapter).
        const bool fullOffload = (nGpu == geom_.numLayers);
        if (!fullOffload) {
            errMsg = "qwen35moe requires full offload (no partial)";
            return false;
        }
        const uint32_t H = geom_.hiddenSize;
        // (I = intermediateSize is unused here: the MoE FFN replaces the dense
        // FFN, so no dense ffnGate/ffnUp GEMVs run in this function.)
        const uint32_t nHeads = geom_.numAttentionHeads;
        const uint32_t nKV = geom_.numKVHeads;
        const uint32_t hd = geom_.headDim;
        const uint32_t qLen = nHeads * hd, kvLen = nKV * hd;
        const uint32_t pos = static_cast<uint32_t>(kvPos_);

        // Qwen35MoE dense-hybrid geometry (shared with qwen35).
        const uint32_t dInner = geom_.ssmInnerSize;           // 4096
        const uint32_t headK = geom_.ssmStateSize;            // 128
        const uint32_t nKHeads = geom_.ssmGroupCount;         // 16
        const uint32_t nVHeads = geom_.ssmTimeStepRank;       // 32
        const uint32_t headV = dInner / std::max(nVHeads, 1u);// 128
        const uint32_t keyDim = headK * nKHeads;              // 2048
        const uint32_t valueDim = headV * nVHeads;            // 4096
        const uint32_t convK = geom_.ssmConvKernel;           // 4
        const uint32_t convChannels = 2 * keyDim + valueDim;  // 8192
        const uint32_t qkvDim = convChannels;
        const uint32_t qgDims = nHeads * 2 * hd;

        // MoE geometry.
        const uint32_t expertCount = geom_.expertCount;   // 256
        const uint32_t expertUsed = geom_.expertUsedCount;// 8
        const uint32_t expertFF = geom_.expertFF;         // 512
        const uint32_t sharedFF = geom_.expertSharedFF;   // 512

        if (!ensureScratch(seqLen, errMsg)) return false;
        DeviceScratch &s = scratch_;
        cudaMemcpyAsync(s.tokens, tokens.data(), seqLen * sizeof(int32_t),
                        cudaMemcpyHostToDevice, g_stream);
        cudaStreamSynchronize(g_stream);

        // ---- MoE stats instrumentation.  Cheap steady_clock hooks only (no
        // per-layer stream syncs, no prints in the hot loop): enabled by
        // TINYCODER_MOE_STATS=1 OR the legacy TINYCODER_GPU_VERBOSE=1 (which
        // ALSO adds slower per-stage debug elsewhere -- never use it for
        // timing).  The [moe fwd] line is the ONLY print (once per forward).
        const bool moeTr = g_verbose ||
                           (std::getenv("TINYCODER_MOE_STATS") != nullptr);
        auto tF0 = std::chrono::steady_clock::now();
        if (moeTr) {
            if (!moeEvStart_) cudaEventCreate(&moeEvStart_);
            if (!moeEvEnd_) cudaEventCreate(&moeEvEnd_);
            cudaEventRecord(moeEvStart_, g_stream);
        }
        uint64_t cacheHits = 0, cacheMisses = 0, cacheMissBytes = 0;
        double cpuRouterMs = 0.0, layerGpuMs = 0.0, moeBlockMs = 0.0;
        double syncWaitMs = 0.0;
        double rankLoopMs = 0.0;// rank GEMV loop: fills + enqueues (host-side)
        double fillMs = 0.0;    // batched-path parallel fill (PHASE 1b/1c)
        bool moeFallbackWarned = false;

        // ---- Device-side phase profile (TINYCODER_PHASE_PROFILE=1) ----
        //
        // The [moe fwd] host timers (syncWaitMs / rankLoopMs / fillMs /
        // cpuRouterMs) measure HOST wall between enqueues, so a host-stall
        // (the pinned D2H handoff, the blocking sync) gets attributed to
        // "sync" even though the DEVICE may have already finished -- and the
        // device may still be draining the trunk while the host runs the
        // router.  To separate DEVICE time from HOST time inside the per-layer
        // wall, TINYCODER_PHASE_PROFILE=1 records stream-ordered CUDA events
        // at three boundaries per layer (layer start, post-trunk+shexp+D2H
        // enqueue, post-rank+addScaled enqueue), then accumulates the DEVICE
        // deltas:
        //   trunkDev  = ev1 - ev0  (true SM/engine time the trunk+D2H+shexp
        //                            consumed on the stream -- INCLUDES any
        //                            queue-wait on the previous layer's tail)
        //   gapDev    = ev2 - ev1  (device-idle stretch while the host ran the
        //                            CPU router + fills + enqueues -- if this
        //                            dominates, the WALL is host-side and the
        //                            fix is CPU/GPU overlap, not more kernels)
        // Events are local (created lazily, destroyed before return); nothing
        // is recorded unless the env flag is set.
        static thread_local const bool phaseProf = [] {
            const char *e = std::getenv("TINYCODER_PHASE_PROFILE");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        struct PhaseAcc {
            double trunkMs = 0.0;     // ev1-ev0 per layer (trunk+D2H+shexp dev)
            double gapMs = 0.0;       // ev2-ev1 per layer (router+fill host gap)
            double tailMs = 0.0;      // next-ev0 - ev2 (residual + addScaled dev)
            double lmHeadMs = 0.0;    // final norm + LM-head GEMV + D2H (dev)
            double hostRouterMs = 0.0;// cpuRouterMs (reported for the diff)
        };
        // Per-layer event ring [L][slot]: 3 stream-ordered events per layer.
        // Recorded with NO host syncs; all deltas are computed ONCE after the
        // final cudaStreamSynchronize(g_stream) (the true completion point),
        // so this probe adds zero extra syncs to the critical path.
        std::vector<std::array<cudaEvent_t, 3>> phaseEv;
        if (phaseProf) {
            // Sizing by nGpu (the loop caps at nGpu && nRealLayers, so the
            // worst case is nGpu entries; MTP layers are never executed).
            phaseEv.resize(nGpu);
            for (auto &row: phaseEv) {
                for (auto &ev: row) {
                    cudaEventCreate(&ev);
                }
            }
        }
        auto phaseMark = [&](uint32_t layerIdx, uint32_t slot) {
            if (!phaseProf) return;
            cudaEventRecord(phaseEv[layerIdx][slot], g_stream);
        };
        if (moeTr) {
            std::fprintf(stderr,
                         "[moe tr] entry seqLen=%u expertCacheBuilt_=%d "
                         "moeCpuFn_=%d nGpu=%u\n",
                         seqLen, expertCacheBuilt_ ? 1 : 0, moeCpuFn_ ? 1 : 0,
                         nGpu);
        }
        // Per-layer wall-clock hooks: layerGpuMs accumulates the enqueue-to-D2H
        // interval (GPU work + launch overhead); moeBlockMs accumulates the
        // hybrid-handoff interval (D2H sync .. residual) minus the CPU router.
        auto tLayer0 = std::chrono::steady_clock::now();
        auto startLayer = [&]() {
            tLayer0 = std::chrono::steady_clock::now();
        };
        auto endLayerGpu = [&]() {
            layerGpuMs += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - tLayer0)
                                  .count();
        };
        auto tMoe0 = std::chrono::steady_clock::now();
        auto startMoeBlock = [&]() {
            tMoe0 = std::chrono::steady_clock::now();
        };
        auto endMoeBlock = [&]() {
            moeBlockMs += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - tMoe0)
                                  .count();
        };
        // ---- PER-LAYER expert geometry accessors (2026-09-17) ----
        // The Ornith Q4_K_M mix changes the DOWN expert quant type between
        // layers (most layers down=Q6_K 210 B/block, layers 5-6 down=Q4_K
        // 144 B/block).  The member globals (expertType_/expertRowBytesGate_/
        // expertDownType_/expertRowBytesDown_) are overwritten every upload
        // step and hold only the LAST layer's values, so the cache decode must
        // derive gate/down geometry from THIS layer's ExpertSlot record
        // (falling back to the globals when the record is missing).
        auto layGateType = [&](uint32_t L) -> uint32_t {
            const uint32_t t = expertCache_[L].gateType;
            return t ? t : expertType_;
        };
        auto layGateRowBytes = [&](uint32_t L) -> uint32_t {
            const uint32_t rb = expertCache_[L].gateRowBytes;
            return rb ? rb : expertRowBytesGate_;
        };
        auto layDownType = [&](uint32_t L) -> uint32_t {
            const uint32_t t = expertCache_[L].downType;
            return t ? t : (expertDownType_ ? expertDownType_ : expertType_);
        };
        auto layDownRowBytes = [&](uint32_t L) -> uint32_t {
            const uint32_t rb = expertCache_[L].downRowBytes;
            return rb ? rb : expertRowBytesDown_;
        };
        auto layGateSliceB = [&](uint32_t L) -> uint64_t {
            return static_cast<uint64_t>(expertFF_) * layGateRowBytes(L);
        };
        auto layDownSliceB = [&](uint32_t L) -> uint64_t {
            return static_cast<uint64_t>(expertHM_) * layDownRowBytes(L);
        };

        // ---- Embedding (K-quant / legacy) — same dispatch as qwen35. ----
        {
            const uint32_t blocksPerRow = hostBlocksPerRow(embedType_, H);
            const uint8_t *eQ = reinterpret_cast<const uint8_t *>(embedQ_);
            if (embedType_ == kTypeQ5K || embedType_ == kTypeQ5_0) {
                kEmbedDequantQ5K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ4K) {
                kEmbedDequantQ4K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ8_0) {
                kEmbedDequantSmall32<kTypeQ8_0><<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeIQ4NL) {
                kEmbedDequantSmall32<kTypeIQ4NL><<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else {
                kEmbedDequant<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            }
        }
        // TINYCODER_DUMP_NORM=1: dump the GPU's embedding output (s.hidden)
        // before any layer, so a GPU-vs-CPU divergence can be localized to the
        // embedding vs the L0 recurrent/attention block.
        {
            const char *dn4 = std::getenv("TINYCODER_DUMP_NORM");
            if (dn4 != nullptr && dn4[0] != '\0' && dn4[0] != '0' && seqLen >= 1) {
                cudaStreamSynchronize(g_stream);
                float e0[8];
                cudaMemcpy(e0, s.hidden, 8 * sizeof(float), cudaMemcpyDeviceToHost);
                double esum = 0.0;
                for (int i = 0; i < 2048; ++i) {
                    float t;
                    cudaMemcpy(&t, s.hidden + i, sizeof(float),
                               cudaMemcpyDeviceToHost);
                    esum += static_cast<double>(t) * t;
                }
                std::fprintf(stderr,
                             "[gpu emb] hidden[0..7]={%.6f %.6f %.6f %.6f %.6f "
                             "%.6f %.6f %.6f} ||h||=%.3f\n",
                             e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                             std::sqrt(esum));
            }
        }

        // Dense projection helper.  qwen35moe runs the SAME math for prefill
        // (seqLen>1) and decode (seqLen==1): a per-token loop of the exact
        // quantized GEMV (launchQGemv), NEVER an fp16 cuBLAS GEMM.
        //
        // Exact per-token GEMVs are what make batch == sequential (and
        // GPU == CPU) here.  The hybrid CPU router (moeCpuFn_) consumes a D2H
        // mirror of the post-attn norm, and the prefill changed it: the old
        // fp16 cuBLAS GEMM rounded every projection differently from the
        // fp32 decode GEMV (visible as ~1e-3 router-logit deltas at layer 0),
        // and the MoE top-8 set is knife-edge (probabilities clustered within
        // ~1e-3), so a borderline expert flipped by layer 2 and diverged
        // chaotically over the 40 layers (maxDiff ~19, different top-1).  With
        // identical activations the CPU router selects bit-identical experts
        // and batch collapses to the (already verified) decode path.
        //
        // launchQGemv covers every type these projections use (K-quants via
        // kQGemv*, Q8_0 via kQGemvSmall32, F32 via kQ35F32Gemv), so no fp16
        // conversion path is needed at all.
        auto gemv = [&](const DeviceMatrix &m, const float *x, float *y,
                        uint32_t nRows, uint32_t nCols,
                        const char *what) -> bool {
            (void) what;
            if (m.q == nullptr || nRows == 0) return true;
            for (uint32_t st = 0; st < seqLen; ++st) {
                launchQGemv(m.type, m.q, x + static_cast<size_t>(st) * nCols,
                            y + static_cast<size_t>(st) * nRows, nRows, nCols,
                            m.rowBytes, m.blocksPerRow, q8k_, q8kBytes_);
            }
            return true;
        };

        const float invSqrtV = 1.0f / std::sqrt(static_cast<float>(headV));
        const uint32_t nDims = geom_.ropeDimensionCount > 0
                                       ? geom_.ropeDimensionCount
                                       : 64;

        // ---- Offloaded layer prefix (all layers; qwen35moe is full offload).
        // The MTP (NextN) block is never executed on the main decode path.
        const uint32_t nRealLayers =
                geom_.numLayers - geom_.nextnPredictLayers;
        for (uint32_t L = 0; L < nGpu && L < nRealLayers; ++L) {
            const DeviceLayer &w = layers_[L];
            if (moeTr) startLayer();
            phaseMark(L, 0);// ev0: layer start (trunk boundary)

            // ---- Attention / recurrent block (identical to qwen35) ----
            kRMSNormRow<<<seqLen, 1024, 1024 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.rmsNormAttn, H, seqLen, 1e-6f);

            // TINYCODER_DUMP_NORM=1: dump the GPU's L0 attention norm (the
            // recurrent block's input, BEFORE any conv/GDN math) so it can be
            // compared against the CPU's `[cpuattnnorm]` -- both are
            // rmsNormAttn(embedding), so any difference isolates
            // kRMSNormRow / the rmsNormAttn upload (embedding is already
            // verified byte-identical).
            {
                const char *dnA = std::getenv("TINYCODER_DUMP_NORM");
                if (L == 0 && dnA != nullptr && dnA[0] != '\0' &&
                    dnA[0] != '0') {
                    cudaStreamSynchronize(g_stream);
                    float a0[8];
                    cudaMemcpy(a0, s.norm, 8 * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    std::fprintf(stderr,
                                 "[gpuattnnorm] L0 attnNorm[0..4]={%.6f %.6f "
                                 "%.6f %.6f %.6f} (8th=%.6f)\n",
                                 a0[0], a0[1], a0[2], a0[3], a0[4], a0[7]);
                }
            }

            if ((L + 1) % geom_.fullAttentionInterval != 0) {
                // =================== Recurrent (gated delta net) ===========
                if (!gemv(w.attnQKV, s.norm, s.qkv, qkvDim, H, "attnQKV") ||
                    !gemv(w.attnGate, s.norm, s.q35z, dInner, H, "attnGate") ||
                    !gemv(w.ssmBetaQ, s.norm, s.q35Beta, nVHeads, H,
                          "ssmBetaQ") ||
                    !gemv(w.ssmAlphaQ, s.norm, s.q35GateV, nVHeads, H,
                          "ssmAlphaQ")) {
                    return false;
                }
                // beta=sigmoid(beta); gateV=softplus(alpha+dt.bias)*ssm_a
                kQ35BetaGate<<<seqLen, 64, 0, g_stream>>>(
                        s.q35Beta, s.q35GateV, w.ssmDtBiasFull, w.ssmABroadcast,
                        s.q35Beta, s.q35GateV, seqLen, nVHeads);

                // Sequential recurrence over tokens (per-layer persistent state).
                float *q35ConvState =
                        q35ConvState_ + static_cast<size_t>(L) * (convK - 1) * convChannels;
                float *q35GdnState =
                        q35GdnState_ + static_cast<size_t>(L) * nVHeads * headV * headV;
                const float *convW = static_cast<const float *>(w.ssmConv1d.q);
                const uint32_t convBlocks = (convChannels + 255) / 256;
                for (uint32_t st = 0; st < seqLen; ++st) {
                    const float *qkv = s.qkv + static_cast<size_t>(st) * qkvDim;
                    kQ35ConvStep<<<convBlocks, 256, 0, g_stream>>>(
                            convW, qkv, s.convIn, s.convOut, q35ConvState,
                            convChannels, convK);
                    // TINYCODER_DUMP_NORM=1: dump GPU L0 token-0 recurrent
                    // intermediates (qkv, convOut post-silu) to bisect against
                    // the CPU's [cpuqkv]/[cpuconv] (zero recurrent state at
                    // st==0, so convOut = silu(conv(qkv)) is a pure function).
                    const char *dnR = std::getenv("TINYCODER_DUMP_NORM");
                    if (L == 0 && st == 0 && dnR != nullptr && dnR[0] != '\0' &&
                        dnR[0] != '0') {
                        cudaStreamSynchronize(g_stream);
                        float q0[5], c0[5];
                        cudaMemcpy(q0, s.qkv, 5 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        cudaMemcpy(c0, s.convOut, 5 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        std::fprintf(stderr,
                                     "[gpuqkv] L0T0 qkv[0..4]={%.6f %.6f %.6f "
                                     "%.6f %.6f}\n",
                                     q0[0], q0[1], q0[2], q0[3], q0[4]);
                        std::fprintf(stderr,
                                     "[gpuconv] L0T0 convOut[0..4]={%.6f %.6f "
                                     "%.6f %.6f %.6f}\n",
                                     c0[0], c0[1], c0[2], c0[3], c0[4]);
                    }
                    kQ35L2NormRepeat<<<1, 128, 0, g_stream>>>(
                            s.convOut, s.qN, s.kN, s.vN, headK, headV, nKHeads,
                            nVHeads, keyDim, valueDim, 1);
                    const float *betaRow = s.q35Beta + static_cast<size_t>(st) * nVHeads;
                    const float *gateRow = s.q35GateV + static_cast<size_t>(st) * nVHeads;
                    float *gdnRow = s.gdnOut + static_cast<size_t>(st) * valueDim;
                    // TINYCODER_GDN_SEQ=1: restore the original sequential
                    // per-head launch loop (48 grid(1) launches) for parity
                    // bisection against the batched grid.x=nVHeads launch.
                    const bool gdnSeq = (std::getenv("TINYCODER_GDN_SEQ") != nullptr);
                    if (gdnSeq) {
                        for (uint32_t hv = 0; hv < nVHeads; ++hv) {
                            kQ35GdnStep<128><<<1, 128, 0, g_stream>>>(
                                    s.qN + static_cast<size_t>(hv) * headK,
                                    s.kN + static_cast<size_t>(hv) * headK,
                                    s.vN + static_cast<size_t>(hv) * headV,
                                    betaRow, gateRow,
                                    q35GdnState + static_cast<size_t>(hv) * headV * headV,
                                    gdnRow + static_cast<size_t>(hv) * headV, nVHeads,
                                    hv, invSqrtV);
                        }
                    } else {
                        kQ35GdnStepBatch<128><<<nVHeads, 128, 0, g_stream>>>(
                                s.qN, s.kN, s.vN, betaRow, gateRow, q35GdnState,
                                gdnRow, nVHeads, invSqrtV);
                    }
                }
                // Gated RMSNorm (per value-head) × silu(z) over the batch.
                kQ35GatedRmsNorm<<<seqLen, 256, 256 * sizeof(float), g_stream>>>(
                        s.q35z, s.gdnOut, w.ssmNorm, headV, nVHeads, seqLen);
                if (!gemv(w.ssmOut, s.gdnOut, s.attnProj, H, valueDim,
                          "ssmOut")) {
                    return false;
                }
                // TINYCODER_DUMP_NORM=1: dump GPU L0 token-0 gdnOut (post
                // gated-RMSNorm × silu(z)) and attnProj (post ssmOut) to bisect
                // against the CPU reference.  gdnOut[0..4] is token s's
                // [headV*nVHeads] slice; attnProj[0..4] is the ssmOut output.
                const char *dnO = std::getenv("TINYCODER_DUMP_NORM");
                if (L == 0 && dnO != nullptr && dnO[0] != '\0' &&
                    dnO[0] != '0') {
                    cudaStreamSynchronize(g_stream);
                    float g0[5], p0[5];
                    cudaMemcpy(g0, s.gdnOut, 5 * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(p0, s.attnProj, 5 * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    std::fprintf(stderr,
                                 "[gpuout] L0 gdnOut[0..4]={%.6f %.6f %.6f "
                                 "%.6f %.6f} attnProj[0..4]={%.6f %.6f %.6f "
                                 "%.6f %.6f}\n",
                                 g0[0], g0[1], g0[2], g0[3], g0[4], p0[0], p0[1],
                                 p0[2], p0[3], p0[4]);
                }
            } else {
                // =================== Full attention =======================
                if (!gemv(w.attnQ, s.norm, s.q35QGate, qgDims, H, "attnQ") ||
                    !gemv(w.attnK, s.norm, s.k, kvLen, H, "attnK") ||
                    !gemv(w.attnV, s.norm, s.v, kvLen, H, "attnV")) {
                    return false;
                }
                // Per-head Q RMSNorm (Q slice) + K RMSNorm.
                kQ35HeadRmsNorm2<<<seqLen * nHeads, 128, 128 * sizeof(float),
                                   g_stream>>>(s.q35QGate, s.q, w.attnQNorm, hd,
                                               2 * hd, seqLen * nHeads);
                kQ35HeadRmsNorm<<<seqLen * nKV, 128, 128 * sizeof(float),
                                  g_stream>>>(s.k, s.k, w.attnKNorm, hd,
                                              seqLen * nKV);
                // MRoPE on Q and K.
                kQ35MRoPE<<<dim3(seqLen, nHeads), 32, 0, g_stream>>>(
                        s.q, seqLen, nHeads, hd, nDims, pos, mropeCos_, mropeSin_);
                kQ35MRoPE<<<dim3(seqLen, nKV), 32, 0, g_stream>>>(
                        s.k, seqLen, nKV, hd, nDims, pos, mropeCos_, mropeSin_);
                // Store K/V into the layer's KV cache.
                // Full-attention layer; the KV cache is slotted by
                // floor(L/interval) (recurrent layers get no slot) — see
                // kvLayerOffset().  qwen35 stores fp32 KV (kvHalf_ is dense-only).
                float *kvK = reinterpret_cast<float *>(
                                     static_cast<uint8_t *>(kvK_) + kvLayerOffset(L));
                float *kvV = reinterpret_cast<float *>(
                                     static_cast<uint8_t *>(kvV_) + kvLayerOffset(L));
                kQ35StoreKV<<<seqLen, 256, 0, g_stream>>>(s.k, s.v, kvK, kvV,
                                                          seqLen, nKV, hd, pos);
                // Flash attention (warp per (token, q-head), HD=256).
                const float invSqrtA = 1.0f / std::sqrt(static_cast<float>(hd));
                dim3 blk(32, 4);
                kQ35WarpAttention<256><<<dim3(seqLen, (nHeads + 3) / 4), blk, 0,
                                         g_stream>>>(s.q, kvK, kvV, s.attnOut,
                                                     seqLen, nHeads, nKV, pos,
                                                     invSqrtA);
                // attnOut *= sigmoid(gate).
                kQ35SigmoidGate<<<seqLen, 256, 0, g_stream>>>(
                        s.attnOut, s.q35QGate, seqLen, nHeads, hd);
                if (!gemv(w.attnO, s.attnOut, s.attnProj, H, qLen, "attnO")) {
                    return false;
                }
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.attnProj, seqLen * H);

            // ---- Post-attention RMSNorm + MoE FFN + residual ----
            // TINYCODER_DUMP_NORM=1: dump the GPU's pre-post-norm hidden and
            // the resulting post-attn norm at L0 (seqLen==1) so a GPU-vs-CPU
            // layer-0 divergence can be localized (hidden vs norm scale).
            {
                const char *dn3 = std::getenv("TINYCODER_DUMP_NORM");
                if (dn3 != nullptr && dn3[0] != '\0' && dn3[0] != '0' &&
                    L == 0) {
                    cudaStreamSynchronize(g_stream);
                    float h0[8];
                    cudaMemcpy(h0, s.hidden, 8 * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    double hsum = 0.0;
                    for (int i = 0; i < 2048; ++i) {
                        float t;
                        cudaMemcpy(&t, s.hidden + i, sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        hsum += static_cast<double>(t) * t;
                    }
                    std::fprintf(stderr,
                                 "[gpu l0] pre-post-norm hidden[0..7]={%.6f "
                                 "%.6f %.6f %.6f %.6f %.6f %.6f %.6f} "
                                 "||h||=%.3f\n",
                                 h0[0], h0[1], h0[2], h0[3], h0[4], h0[5], h0[6],
                                 h0[7], std::sqrt(hsum));
                }
            }
            kRMSNormRow<<<seqLen, 1024, 1024 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.postAttnNorm, H, seqLen, 1e-6f);

            // ---------- MoE FFN: CPU-expert hybrid OR full-GPU ----------
            if (moeCpuFn_) {
                // When the per-layer expert cache was built (buildExpertCache
                // success), the 8-of-256 routed expert FFNs ALSO run on the GPU
                // from the cached slices (VRAM bandwidth): the CPU callback
                // only fills the router-selection tables, so NO routed-output
                // H2D crosses the PCIe bus and the ~32 MB/ layer of host-RAM
                // expert weights (the decode bottleneck on models bigger than
                // the page cache) is never streamed. The host does ~µs of
                // router work per layer (a 256x2048 fp32 mat-vec per token).
                // Without the cache (device-allocation failure fallback), the
                // CPU runs the full routed expert FFNs into moeFfnOutHost_ (the
                // old --cpu-moe path).
                //
                // SYNCHRONOUS handoff (no event ring).  Events proved racy (see
                // the historical note below); every layer hands the norm to the
                // host with a blocking in-order D2H and takes the routed output
                // back in-order on g_stream — no events, no cross-stream
                // ordering, no reuse hazard.  The CPU router is deterministic
                // on the exact device norm, so batch and single-token prefills
                // that produce bit-identical norms produce bit-identical MoE
                // outputs BY CONSTRUCTION, and the shared-expert kernels (step
                // 2) still run on-device while the CPU callback computes.
                const size_t normBytes =
                        static_cast<size_t>(seqLen) * H * sizeof(float);
                // Ensure the double-buffer mirrors exist (grow on demand).
                if (moeNormHost_.size() < 2) {
                    moeNormHost_.resize(2);
                    moeFfnOutHost_.resize(2);
                }
                if (moeNormHost_[0].size() < static_cast<size_t>(seqLen) * H) {
                    // Unpin the OLD ranges FIRST: the resize below may
                    // reallocate storage, which would leave the registration
                    // pointing at freed/repurposed memory (registering or
                    // freeing it later is UB).  moeNormPinned_[i] tracks each
                    // slot's current pinned range (nullptr = not pinned).
                    for (uint32_t pi = 0; pi < 2; ++pi) {
                        if (moeNormPinned_[pi] != nullptr) {
                            cudaHostUnregister(moeNormPinned_[pi]);
                            moeNormPinned_[pi] = nullptr;
                        }
                    }
                    for (auto &slot: moeNormHost_)
                        slot.resize(static_cast<size_t>(seqLen) * H);
                    for (auto &slot: moeFfnOutHost_)
                        slot.resize(static_cast<size_t>(seqLen) * H);
                    // PIN the D2H norm mirrors ONCE here (2026-09-14 root-cause
                    // fix): a pageable cudaMemcpyAsync D2H forces the driver to
                    // bounce 8 KB transfers through ITS internal pinned staging
                    // with a global lock, so the host-side enqueue BLOCKS until
                    // the copy completes — and the copy waits for the whole
                    // unflushed trunk burst, stranding the GPU at idle clocks.
                    // Per-layer profiling saw wall ~39 ms/layer with only ~4 ms
                    // of sync/router/rank/fill accounted — the unaccounted
                    // ~35 ms is exactly this hidden pageable-D2H serialization
                    // (invisible to the CUDA-event timeline: dev == total
                    // because the stream is EMPTY during the stall).
                    // cudaHostRegister pins the buffer, turning each per-layer
                    // transfer into a true async DMA.
                    // Opt-out for measurement: TINYCODER_NO_PIN_NORM=1.
                    const char *noPin = std::getenv("TINYCODER_NO_PIN_NORM");
                    const bool pinNorm =
                            (noPin == nullptr || noPin[0] == '\0' ||
                             noPin[0] == '0');
                    if (pinNorm) {
                        for (uint32_t pi = 0; pi < 2; ++pi) {
                            cudaError_t eP = cudaHostRegister(
                                    moeNormHost_[pi].data(),
                                    static_cast<size_t>(seqLen) * H *
                                            sizeof(float),
                                    cudaHostRegisterMapped |
                                            cudaHostRegisterPortable);
                            if (eP == cudaSuccess) {
                                moeNormPinned_[pi] =
                                        moeNormHost_[pi].data();
                            } else {
                                // Non-fatal: fall back to the pageable D2H (old
                                // behavior) rather than abort decode.
                                std::fprintf(stderr,
                                             "[gpu] cudaHostRegister(moeNormHost) "
                                             "failed: %s\n",
                                             cudaGetErrorString(eP));
                                moeNormPinned_[pi] = nullptr;
                            }
                        }
                    }
                }
                const uint32_t buf = moeBufFlip_ & 1u;
                if (moeTr) endLayerGpu();
                if (moeTr) startMoeBlock();
                // 1) D2H the post-attn norm on g_stream: the CPU router's exact
                //    input.  In-order AFTER this layer's RMSNorm (s.norm is
                //    still this layer's; L+1's RMSNorm is not enqueued until
                //    the next iteration), race-free by the sync in step 3.
                cudaMemcpyAsync(moeNormHost_[buf].data(), s.norm, normBytes,
                                cudaMemcpyDeviceToHost, g_stream);
                phaseMark(L, 1);// ev1: after trunk+D2H+shexp enqueues
                // 2) Shared-expert compute on g_stream (runs CONCURRENTLY with
                //    the CPU callback below — this keeps the GPU busy): gate
                //    sigmoid + SwiGLU + down into s.moeShexpDown.  It only
                //    reads s.norm (g_stream-ordered after the RMSNorm) and
                //    nothing here depends on the routed experts.
                if (w.ffnGateShexp.q != nullptr && w.ffnUpShexp.q != nullptr &&
                    w.ffnDownShexp.q != nullptr && sharedFF > 0) {
                    kQ35MoeShexpGate<<<seqLen, 256, 0, g_stream>>>(
                            s.norm, static_cast<const float *>(w.ffnGateInpShexp.q),
                            s.moeShexpGate, seqLen, H, H);
                    if (!gemv(w.ffnGateShexp, s.norm, s.moeShexpGateUp, sharedFF, H,
                              "ffnGateShexp") ||
                        !gemv(w.ffnUpShexp, s.norm, s.moeShexpGateUp2, sharedFF, H,
                              "ffnUpShexp")) {
                        return false;
                    }
                    kSiluMul<<<(seqLen * sharedFF + 255) / 256, 256, 0, g_stream>>>(
                            s.moeShexpGateUp, s.moeShexpGateUp2, seqLen * sharedFF);
                    if (!gemv(w.ffnDownShexp, s.moeShexpGateUp, s.moeShexpDown, H,
                              sharedFF, "ffnDownShexp")) {
                        return false;
                    }
                }
                // 3) Blocking sync: this layer's norm D2H is complete and the
                //    step-2 shared-expert kernels (reading s.norm) have
                //    finished.  The CPU callback now sees the exact norm.
                if (moeTr) {
                    auto tSync0 = std::chrono::steady_clock::now();
                    cudaStreamSynchronize(g_stream);
                    syncWaitMs += std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() -
                                          tSync0)
                                          .count();
                } else {
                    cudaStreamSynchronize(g_stream);
                }
                // TINYCODER_DUMP_NORM=1: verify the D2H mirror the CPU router
                // is about to read equals a FRESH device read of s.norm at the
                // SAME moment (post-sync).  A mismatch proves the mirror is
                // stale/raced (D2H ordering) vs an on-device s.norm clobber by
                // the MoE block (mirror matches but the post-block readback
                // differs from the pre-block value).
                {
                    const char *dnP = std::getenv("TINYCODER_DUMP_NORM");
                    if (dnP != nullptr && dnP[0] != '\0' && dnP[0] != '0' &&
                        seqLen >= 1) {
                        float dv[2];
                        cudaMemcpy(dv, s.norm, 2 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        const float *mv = moeNormHost_[buf].data();
                        std::fprintf(stderr,
                                     "[normprobe] L%u mirror[0..1]={%.6f %.6f} "
                                     "dev[0..1]={%.6f %.6f} %s\n",
                                     L, mv[0], mv[1], dv[0], dv[1],
                                     (mv[0] == dv[0] && mv[1] == dv[1])
                                             ? "MATCH"
                                             : "MISMATCH");
                    }
                }
                // ---- Option D (static resident-expert split) activation for
                // THIS layer.  residentEnabled_ (adapter policy) + residentActive_
                // (arenas built) + a non-empty resident set for L.  When active,
                // the adapter's callback (invoked below) computes BOTH the router
                // tables (all ranks) AND the NON-resident expert FFNs into the
                // rank-major down table moeDownTableHost_[buf]; the driver then
                // H2Ds it, runs the resident ranks on the GPU into the SAME
                // combined table, and accumulates rank-order.
                //
                // DECODE-ONLY (seqLen == 1): scheduleResidentExpertsBatch issues
                // per-token serial kernel launches, which at prefill batch sizes
                // (32+ tokens x up to 38 resident experts) is SLOWER than the
                // CPU grouped-batch path (computeQwen35MoEFromLogits) that
                // Option B uses.  Measured 2026-09-16: resident-on prefill was
                // 8.8 vs 9.3 pp tok/s for Option B.  Restricting the static
                // resident dispatch to decode keeps prefill on the proven
                // grouped path and delivers the decode speedup (7.58 vs
                // 7.23 tg tok/s).  The resident arenas stay uploaded either way.
                const bool residentOn =
                        residentEnabled_ && residentActive_ && seqLen == 1 &&
                        (std::getenv("TINYCODER_MOE_FALLBACK") == nullptr) &&
                        L < kResidentLayers && moeResidentCount_[L] > 0;
                if (residentOn) {
                    // Grow the double-buffered HOST down table + DEVICE mirror
                    // for the combined rank-major layout [eu][seqLen][H].
                    const size_t downHostN =
                            static_cast<size_t>(seqLen) * expertUsed * H;
                    if (moeDownTableHost_.size() < 2) moeDownTableHost_.resize(2);
                    for (auto &slot: moeDownTableHost_) {
                        if (slot.size() < downHostN) slot.resize(downHostN);
                    }
                    const uint64_t downBytes =
                            static_cast<uint64_t>(downHostN) * sizeof(float);
                    if (moeDownTableBytes_ < downBytes) {
                        if (moeDownTable_) cudaFree(moeDownTable_);
                        moeDownTable_ = nullptr;
                        moeDownTableBytes_ = 0;
                        cudaError_t eT = cudaMalloc(&moeDownTable_, downBytes);
                        if (eT != cudaSuccess) {
                            errMsg = std::string("cudaMalloc(moeDownTable): ") +
                                     cudaGetErrorString(eT);
                            return false;
                        }
                        moeDownTableBytes_ = downBytes;
                    }
                }
                // The adapter callback checks this to know whether to fill the
                // non-resident down table slot (Option D) for this layer.
                residentHostSlot_ = residentOn ? static_cast<int32_t>(buf) : -1;
                // 4) CPU callback: ROUTER-ONLY (expert cache built) or FULL
                //    routed experts (fallback), dispatched inside the callback
                //    on expertCacheBuilt().  While it runs, the GPU is idle (no
                //    L+1 work is enqueued yet), so nothing races device state.
                if (moeTr) {
                    auto tRouter0 = std::chrono::steady_clock::now();
                    const bool rOk = moeCpuFn_(L, moeNormHost_[buf].data(),
                                               moeFfnOutHost_[buf].data(),
                                               seqLen);
                    cpuRouterMs += std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() -
                                           tRouter0)
                                           .count();
                    if (!rOk) {
                        errMsg = "CPU MoE expert callback failed";
                        return false;
                    }
                } else if (!moeCpuFn_(L, moeNormHost_[buf].data(),
                                      moeFfnOutHost_[buf].data(), seqLen)) {
                    errMsg = "CPU MoE expert callback failed";
                    return false;
                }
                // TINYCODER_MOE_FALLBACK=1: force the old CPU-expert fallback
                // (adapter computes the FULL routed experts with
                // computeQwen35MoEFromLogits into moeFfnOutHost_, GPU only adds
                // the shared expert + residual) even when the device cache
                // exists.  Used to bisect the GPU-vs-CPU logits divergence:
                // if the fallback path AGREES with pure CPU but the
                // cache-driven GPU-expert path does NOT, the bug is in the
                // cache slices / expert GEMVs (not the trunk + router).
                const bool forceMoeFallback =
                        (std::getenv("TINYCODER_MOE_FALLBACK") != nullptr);
                // ---- Option D: static resident-expert split dispatch.  The
                // adapter callback (below) computed the router tables (all
                // ranks) AND the NON-resident expert FFNs into the combined
                // rank-major down table moeDownTableHost_[buf] (resident
                // slots zeroed).  H2D that table, then run the RESIDENT ranks
                // on the GPU (scheduleResidentExperts[/Batch]) into the SAME
                // combined table, then accumulate rank-order into
                // s.moeExpertOut (kQ35MoeAccumScaled over the combined table).
                if (residentOn) {
                    const size_t downBytes =
                            static_cast<size_t>(seqLen) * expertUsed * H *
                            sizeof(float);
                    if (moeDownTableBytes_ < downBytes) {
                        if (moeDownTable_) cudaFree(moeDownTable_);
                        moeDownTable_ = nullptr;
                        moeDownTableBytes_ = 0;
                        cudaError_t eT2 = cudaMalloc(&moeDownTable_, downBytes);
                        if (eT2 != cudaSuccess) {
                            errMsg = std::string("cudaMalloc(moeDownTable): ") +
                                     cudaGetErrorString(eT2);
                            return false;
                        }
                        moeDownTableBytes_ = downBytes;
                    }
                    // 5a) H2D the CPU-filled combined down table (rank-major).
                    cudaMemcpyAsync(moeDownTable_, moeDownTableHost_[buf].data(),
                                    downBytes, cudaMemcpyHostToDevice, g_stream);
                    // 5b) Zero the accumulator + H2D the weights (all ranks).
                    cudaMemsetAsync(s.moeExpertOut, 0,
                                    static_cast<uint64_t>(seqLen) * H *
                                            sizeof(float),
                                    g_stream);
                    cudaMemcpyAsync(s.moeWgt, moeRouterWgtHost_.data(),
                                    static_cast<size_t>(seqLen) * expertUsed *
                                            sizeof(float),
                                    cudaMemcpyHostToDevice, g_stream);
                    // 5c) GPU resident ranks write their down rows into the
                    // combined table's resident slots (rank-major offsets).
                    if (seqLen == 1) {
                        scheduleResidentExperts(L, moeDownTable_);
                    } else {
                        scheduleResidentExpertsBatch(L, moeDownTable_, seqLen);
                    }
                    // 5d) Accumulate in exact rank order over the COMBINED
                    // table: out[s] += wgt[s][r] * down[r][s] via the strict
                    // mul-then-add kernel (per-rank, rank-ascending — the CPU
                    // reference order).  The combined table is RANK-MAJOR
                    // [eu][seqLen][H], so rank r's token-slab base is
                    // moeDownTable_ + r*seqLen*H (kQ35MoeAccumScaled reads
                    // src[s*H]); grid = seqLen blocks, each rank adds its
                    // slice in ascending rank order.
                    for (uint32_t r = 0; r < expertUsed; ++r) {
                        kQ35MoeAccumScaled<<<seqLen, 256, 0, g_stream>>>(
                                s.moeWgt + r, expertUsed,
                                moeDownTable_ +
                                        static_cast<uint64_t>(r) * seqLen * H,
                                s.moeExpertOut, seqLen, H);
                    }
                    // The GPU path now produced the full routed-expert output
                    // (resident GPu + non-resident CPU ranks); the shared
                    // expert + residual are added below (shared steps 2/6).
                    goto moeAddShared;
                } else if (expertCacheBuilt_ && !forceMoeFallback) {
                    auto cacheStat = [&](uint32_t ee) {
                        const uint32_t cset = expertCacheSet(ee);
                        const bool hit =
                                (expertCacheTag_[L][cset][0] == static_cast<int32_t>(ee)) ||
                                (expertCacheTag_[L][cset][1] == static_cast<int32_t>(ee));
                        if (hit) {
                            ++cacheHits;
                        } else {
                            ++cacheMisses;
                            cacheMissBytes +=
                                    2 * layGateSliceB(L) + layDownSliceB(L);
                        }
                    };
                    // ---- Router-only: run the 8 routed experts on the GPU
                    // from the per-layer cache.  The callback wrote the
                    // per-(token, rank) (expertId, weight) tables into
                    // moeRouterIdxHost_/moeRouterWgtHost_.  Zero the
                    // accumulator and H2D the weights (the accumulation kernel
                    // reads them on-device) — both in-order on g_stream.
                    auto tRank0 = std::chrono::steady_clock::now();
                    cudaMemsetAsync(s.moeExpertOut, 0,
                                    static_cast<uint64_t>(seqLen) * H *
                                            sizeof(float),
                                    g_stream);
                    cudaMemcpyAsync(s.moeWgt, moeRouterWgtHost_.data(),
                                    static_cast<size_t>(seqLen) * expertUsed *
                                            sizeof(float),
                                    cudaMemcpyHostToDevice, g_stream);
                    // Cache slice geometry (PER-LAYER, 2026-09-17): Ornith's
                    // Q4_K_M mix changes the DOWN expert quant type between
                    // layers, so the block count is rebuilt from THIS layer's
                    // recorded gate/down types (cols == H for gate/up,
                    // cols == expertFF for down).
                    const uint32_t gTypeL = layGateType(L);
                    const uint32_t dTypeL = layDownType(L);
                    const uint32_t gBlocks =
                            (gTypeL == kTypeQ5_0 || gTypeL == kTypeQ8_0)
                                    ? (H + 31) / 32
                                    : (H + 255) / 256;
                    const uint32_t dBlocks =
                            (dTypeL == kTypeQ5_0 || dTypeL == kTypeQ8_0)
                                    ? (expertFF + 31) / 32
                                    : (expertFF + 255) / 256;
                    // Decode fast path: all 8 ranks' gate/up GEMVs share the
                    // SAME activation (s.norm) and the down GEMVs share the
                    // silu output.  Quantize the norm ONCE per layer into the
                    // DEDICATED q8kN_ scratch (Q8_32 layout) instead of
                    // per-(rank, matrix) re-quantize kernels (added
                    // 2026-09-13; measured ~1920 wasteful quantize
                    // launches/token before this split).  q8kN_ must NOT be
                    // q8k_: the per-rank down path re-quantizes s.moeGateUp
                    // into q8k_, and sharing one buffer made ranks 1..7 read
                    // rank 0's silu as the norm (correctness regression,
                    // measured 2026-09-13: garbage MoE output + cache misses
                    // 180->236, missMB 529->752, moeBlockMs 236.7->264.1).
                    // TINYCODER_MOE_SLOW=1 forces the generic per-(rank, matrix)
                    // launchQGemv path (identical to prefill) so a parity
                    // failure can be bisected: fast path vs the shared Q8_0
                    // GEMV branch.  Defaults OFF (fast path on).
                    const bool moeFastPath =
                            (std::getenv("TINYCODER_MOE_SLOW") == nullptr);
                    // Q8_0-expert decode fast path (batched Q8_0xQ8K kernels):
                    // entered only when THIS layer's gate/up experts are Q8_0.
                    if (seqLen == 1 && gTypeL == kTypeQ8_0 && moeFastPath) {
                        const uint64_t needN =
                                static_cast<uint64_t>(gBlocks) *
                                kQ8_0x32_STRIDE;
                        if (q8kN_ == nullptr || q8kNBytes_ < needN) {
                            if (q8kN_) cudaFree(q8kN_);
                            q8kN_ = nullptr;
                            q8kNBytes_ = 0;
                            cudaError_t eN = cudaMalloc(&q8kN_, needN);
                            if (eN != cudaSuccess) {
                                errMsg = std::string("cudaMalloc(q8kN): ") +
                                         cudaGetErrorString(eN);
                                return false;
                            }
                            q8kNBytes_ = needN;
                        }
                        launchQuantQ8_0x32(s.norm, q8kN_, H);
                        // ============ BATCHED decode fast path (2026-09-14) ====
                        // Replaces the per-rank serial 40-kernel chain with 4
                        // rank-indexed launches (grid.y = rank) so all expertUsed
                        // ranks' GEMVs co-schedule on the SMs.  The host gathers
                        // the per-rank cache bases (PHASE 1 — the miss fills
                        // already pipeline on g_stream2 via expertStageEv_) and
                        // publishes ONE tiny pointer-table H2D, then PHASE 2
                        // enqueues: GU GEMV (fused gate+up) -> silu -> per-rank
                        // down-input quantize -> down GEMV -> serialized weight
                        // accumulate (bit-exact CPU order, one thread per element
                        // looping ranks so there is no cross-rank race).
                        //
                        // SLOT-CONFLICT GUARD (2026-09-14): the batched path
                        // enqueues ALL 8 fills on g_stream2 BEFORE any GEMV, so
                        // a later rank's fill can EVICT an earlier rank's slot
                        // (same 2-way set, LRU way) and overwrite its bytes.  All
                        // batched GEMVs wait for all fills, so the earlier rank
                        // would read the WRONG expert (measured: norms diverge
                        // from L3 onward while L0-L2 matched — the ~23% of
                        // layers whose 8 routed experts collide in a set).  The
                        // interleaved per-rank loop below enqueues each GEMV
                        // right after its own fill (stream-ordered BEFORE any
                        // later eviction), so it is race-free: on a detected
                        // collision (two different experts mapping to the SAME
                        // slot base) we fall back to it — the old function
                        // refills each evicted slot right before its GEMV.
                        const uint32_t eu = expertUsed;
                        bool moeSlotConflict = false;
                        std::vector<const uint8_t *> moeSlotSeen;
                        moeSlotSeen.reserve(eu);
                        const uint64_t downStrideB =
                                static_cast<uint64_t>(dBlocks) * kQ8_0x32_STRIDE;
                        const uint64_t needDownP =
                                static_cast<uint64_t>(eu) * downStrideB;
                        if (moeQ8kDownP_ == nullptr ||
                            moeQ8kDownPBytes_ < needDownP) {
                            if (moeQ8kDownP_) cudaFree(moeQ8kDownP_);
                            moeQ8kDownP_ = nullptr;
                            moeQ8kDownPBytes_ = 0;
                            cudaError_t eD = cudaMalloc(&moeQ8kDownP_, needDownP);
                            if (eD != cudaSuccess) {
                                errMsg = std::string("cudaMalloc(moeQ8kDownP): ") +
                                         cudaGetErrorString(eD);
                                return false;
                            }
                            moeQ8kDownPBytes_ = needDownP;
                        }
                        const uint64_t needWq = static_cast<uint64_t>(eu) * 3 *
                                                sizeof(uint8_t *);
                        if (moeRankWq_ == nullptr || moeRankWqBytes_ < needWq) {
                            if (moeRankWq_) cudaFree(moeRankWq_);
                            moeRankWq_ = nullptr;
                            moeRankWqBytes_ = 0;
                            cudaError_t eW = cudaMalloc(&moeRankWq_, needWq);
                            if (eW != cudaSuccess) {
                                errMsg = std::string("cudaMalloc(moeRankWq): ") +
                                         cudaGetErrorString(eW);
                                return false;
                            }
                            moeRankWqBytes_ = needWq;
                        }
                        // PHASE 1 (LOCATE-ONLY): per-rank cache slot bases via
                        // the tag/LRU tables WITHOUT copying any bytes.  The
                        // (hit) bases + (miss) victim reservations are gathered
                        // here; the memcpy fills are dispatched to the
                        // ThreadPool in PARALLEL (PHASE 1b) so the g_stream2
                        // copy engine never idles behind the host's serial
                        // memcpy (measured 2026-09-15: 0.47 ms/miss single-
                        // threaded = ~61 ms/token of host fill on the critical
                        // path — the largest single item in moeBlockMs).
                        // SLOT-CONFLICT guard (unchanged): two DIFFERENT
                        // experts mapping to the SAME slot base means a later
                        // fill would evict an earlier rank's bytes before the
                        // batched GEMVs ran -> fall back to the interleaved
                        // per-rank loop, which refills each slot right before
                        // its own GEMVs and is therefore race-free.
                        struct MoeMissFill {
                            uint32_t expert;
                            const uint8_t *dest;// victim slot base (device)
                            uint32_t victim;    // way reserved in the set
                        };
                        // Slot geometry is derived from THIS layer's recorded
                        // gate/down row strides (per-layer down-expert types;
                        // see the accessors above the layer loop).
                        const uint64_t slotBytesB =
                                2 * layGateSliceB(L) + layDownSliceB(L);
                        // Grow the pinned stage ring to one slot if needed
                        // (mirrors ensureExpertCached's grow-on-demand).
                        if (expertStageBytes_ < slotBytesB) {
                            for (uint32_t b = 0; b < kExpertStageBufs; ++b) {
                                if (expertStage_[b]) {
                                    cudaHostUnregister(expertStage_[b]);
                                    std::free(expertStage_[b]);
                                }
                                expertStage_[b] = nullptr;
                                if (expertStageEv_[b]) {
                                    cudaEventDestroy(expertStageEv_[b]);
                                    expertStageEv_[b] = nullptr;
                                }
                            }
                            for (uint32_t b = 0; b < kExpertStageBufs; ++b) {
                                void *pp = nullptr;
                                if (posix_memalign(
                                            &pp, 4096,
                                            static_cast<size_t>(slotBytesB)) !=
                                    0) {
                                    errMsg = "posix_memalign(expertStage) failed";
                                    return false;
                                }
                                cudaError_t eP = cudaHostRegister(
                                        pp, static_cast<size_t>(slotBytesB),
                                        cudaHostRegisterMapped |
                                                cudaHostRegisterPortable);
                                if (eP != cudaSuccess) {
                                    std::free(pp);
                                    errMsg = std::string(
                                                     "cudaHostRegister(expertStage): ") +
                                             cudaGetErrorString(eP);
                                    return false;
                                }
                                expertStage_[b] = static_cast<uint8_t *>(pp);
                                cudaEventCreateWithFlags(
                                        &expertStageEv_[b],
                                        cudaEventDisableTiming);
                            }
                            expertStageBytes_ = slotBytesB;
                        }
                        std::vector<const uint8_t *> moeRankWqHost;
                        moeRankWqHost.reserve(3u * eu);
                        std::vector<MoeMissFill> moeMisses;
                        moeMisses.reserve(eu);
                        std::vector<uint32_t> moeMissStage;// stage idx per miss
                        moeMissStage.reserve(eu);
                        for (uint32_t r = 0; r < eu; ++r) {
                            const int32_t e0 = moeRouterIdxHost_.data()[r];
                            cacheStat(static_cast<uint32_t>(e0));
                            if (e0 < 0) {
                                errMsg = "qwen35moe expert cache miss: negative "
                                         "expert id in decode fast path";
                                return false;
                            }
                            const uint32_t ue = static_cast<uint32_t>(e0);
                            const uint32_t cset = expertCacheSet(ue);
                            const uint8_t *g0 =
                                    static_cast<const uint8_t *>(
                                            expertCache_[L].gateQ) +
                                    static_cast<size_t>(cset *
                                                        kExpertCacheWays) *
                                            static_cast<size_t>(slotBytesB);
                            const uint8_t *g1 =
                                    g0 + static_cast<size_t>(slotBytesB);
                            const uint8_t *cacheBase = nullptr;
                            const uint8_t *downQ = nullptr;
                            if (expertCacheTag_[L][cset][0] ==
                                static_cast<int32_t>(ue)) {
                                expertCacheLru_[L][cset] = 1;
                                cacheBase = g0;
                                downQ = g0 + 2 * layGateSliceB(L);
                            } else if (expertCacheTag_[L][cset][1] ==
                                       static_cast<int32_t>(ue)) {
                                expertCacheLru_[L][cset] = 0;
                                cacheBase = g1;
                                downQ = g1 + 2 * layGateSliceB(L);
                            } else {
                                // Miss: reserve the LRU way (bytes filled in
                                // PHASE 1b; tag committed after the H2D).
                                uint32_t victim = expertCacheLru_[L][cset];
                                const uint8_t *gBase =
                                        (victim == 0) ? g0 : g1;
                                if (expertCacheTag_[L][cset][0] < 0 &&
                                    expertCacheTag_[L][cset][1] < 0) {
                                    victim = 0;
                                    gBase = g0;
                                }
                                expertCacheLru_[L][cset] =
                                        (victim == 0) ? 1u : 0u;
                                cacheBase = gBase;
                                downQ = gBase + 2 * layGateSliceB(L);
                                moeMisses.push_back(
                                        {ue, gBase, victim});
                                moeMissStage.push_back(
                                        (expertStageFlag_++) %
                                        kExpertStageBufs);
                            }
                            for (size_t q = 0; q < moeSlotSeen.size(); ++q) {
                                if (moeSlotSeen[q] == cacheBase &&
                                    moeRouterIdxHost_.data()[q] != e0) {
                                    moeSlotConflict = true;
                                    break;
                                }
                            }
                            moeSlotSeen.push_back(cacheBase);
                            moeRankWqHost.push_back(cacheBase);
                            // PER-LAYER up-slice pointer (2026-09-17): gate row
                            // stride comes from THIS layer's slot record.
                            moeRankWqHost.push_back(
                                    cacheBase + static_cast<uint64_t>(expertFF_) *
                                                        layGateRowBytes(L));
                            moeRankWqHost.push_back(downQ);
                            if (moeSlotConflict) break;
                        }
                        // Conflict detected: the batched plan would read
                        // corrupted bytes.  Fall back to the interleaved
                        // per-rank loop (its ensureExpertCached refills each
                        // expert right before its own GEMVs on g_stream).
                        if (moeSlotConflict) goto moePerRank;
                        // PHASE 1b (SERIAL FILL — REVERTED 2026-09-16): the
                        // ThreadPool parallel-fill experiment (pool parallelFor
                        // gathering 3*misses zones into the stage ring, then
                        // back-to-back H2Ds) caused a catastrophic regression —
                        // 0.32-0.40 tok/s vs the 5.61 tok/s baseline, fillMs
                        // 91-2773 ms/token vs ~58, syncWaitMs 3.4 -> 90 ms.
                        // Root causes: (1) with ~130-180 misses/token and only
                        // kExpertStageBufs=8 stage buffers, the round-robin
                        // stage claim WRAPS within one token, so two pool
                        // workers concurrently memcpy DIFFERENT experts into
                        // the same stage while the enqueue loop H2Ds that
                        // stage twice (torn, non-bit-identical bytes — the
                        // "immutable source" safety argument is void); (2) the
                        // parallel gather fully serializes in phase 1b BEFORE
                        // phase 1c's H2Ds, so the copy engine idles during the
                        // gather while 8 workers hammer the same pinned pages
                        // (DMA snoop contention ~0.3 GB/s vs 7-11 GB/s
                        // serial); (3) fine-grained fetch_add + std::function
                        // dispatch per 1.1 MB job is the wrong tool for
                        // near-uniform slab copies.  The serial fill below
                        // preserves the original ring semantics (each stage's
                        // H2D drains before the same stage refills) and
                        // reproduces the 5.61 tok/s baseline.
                        if (!moeMisses.empty()) {
                            auto tFill0 = std::chrono::steady_clock::now();
                            const uint8_t *srcGate =
                                    expertCache_[L].hostGate
                                            ? expertCache_[L].hostGate
                                            : expertHostGate_;
                            const uint8_t *srcUp =
                                    expertCache_[L].hostUp
                                            ? expertCache_[L].hostUp
                                            : expertHostUp_;
                            const uint8_t *srcDown =
                                    expertCache_[L].hostDown
                                            ? expertCache_[L].hostDown
                                            : expertHostDown_;
                            for (size_t mi = 0; mi < moeMisses.size(); ++mi) {
                                const MoeMissFill &mf = moeMisses[mi];
                                const uint32_t st = moeMissStage[mi];
                                uint8_t *stage = expertStage_[st];
                                const uint64_t gateSlL = layGateSliceB(L);
                                const uint64_t downSlL = layDownSliceB(L);
                                const uint64_t eoff =
                                        static_cast<uint64_t>(mf.expert) *
                                        gateSlL;
                                std::memcpy(stage, srcGate + eoff, gateSlL);
                                std::memcpy(stage + gateSlL, srcUp + eoff,
                                            gateSlL);
                                std::memcpy(stage + 2 * gateSlL,
                                            srcDown +
                                                    static_cast<uint64_t>(
                                                            mf.expert) *
                                                            downSlL,
                                            downSlL);
                                cudaMemcpyAsync(
                                        const_cast<uint8_t *>(mf.dest), stage,
                                        static_cast<size_t>(slotBytesB),
                                        cudaMemcpyHostToDevice, g_stream2);
                                cudaEventRecord(expertStageEv_[st], g_stream2);
                                cudaStreamWaitEvent(g_stream,
                                                    expertStageEv_[st], 0);
                                expertCacheTag_[L][expertCacheSet(mf.expert)]
                                               [mf.victim] =
                                                       static_cast<int32_t>(mf.expert);
                            }
                            if (moeTr) {
                                fillMs += std::chrono::duration<double,
                                                                std::milli>(
                                                  std::chrono::steady_clock::now() -
                                                  tFill0)
                                                  .count();
                            }
                        }
                        cudaMemcpyAsync(moeRankWq_, moeRankWqHost.data(), needWq,
                                        cudaMemcpyHostToDevice, g_stream);
                        // PHASE 2: batched launches.
                        const uint8_t *const *wqTable =
                                reinterpret_cast<const uint8_t *const *>(
                                        moeRankWq_);
                        const uint32_t combinedRows = 2u * expertFF;
                        kQGemvQ8_0xQ8K_BatchGU<<<
                                dim3((combinedRows + 7) / 8, eu), dim3(32, 8), 0,
                                g_stream>>>(wqTable, q8kN_, s.moeGateUpP,
                                            s.moeGateUp2P, expertFF, gBlocks,
                                            layGateRowBytes(L), combinedRows);
                        kSiluMulBatch<<<dim3((expertFF + 255) / 256, eu), 256, 0,
                                        g_stream>>>(s.moeGateUpP, s.moeGateUp2P,
                                                    expertFF, eu);
                        kQuantizeQ8_0x32_Batch<<<dim3(dBlocks, eu), 32, 0,
                                                 g_stream>>>(
                                s.moeGateUpP, moeQ8kDownP_, expertFF, eu,
                                static_cast<uint32_t>(downStrideB));
                        kQGemvQ8_0xQ8K_BatchDown<<<dim3((H + 7) / 8, eu),
                                                   dim3(32, 8), 0, g_stream>>>(
                                wqTable, moeQ8kDownP_, s.moeDownP, H, dBlocks,
                                layDownRowBytes(L),
                                static_cast<uint32_t>(downStrideB));
                        kQ35MoeAccumScaledBatch<<<1, 256, 0, g_stream>>>(
                                s.moeWgt, expertUsed, s.moeDownP, s.moeExpertOut,
                                seqLen, H, eu);
                        // TINYCODER_DUMP_MOE=1 + L0: dump rank 0's batched
                        // gate/up + down products for bisection (replaces the
                        // old per-rank [gpu fast L0] dump).
                        {
                            const char *dnR = std::getenv("TINYCODER_DUMP_MOE");
                            if (L == 0 && dnR != nullptr && dnR[0] != '\0' &&
                                dnR[0] != '0') {
                                cudaStreamSynchronize(g_stream);
                                float gu[4], gd[4];
                                cudaMemcpy(gu, s.moeGateUpP, 4 * sizeof(float),
                                           cudaMemcpyDeviceToHost);
                                cudaMemcpy(gd, s.moeDownP, 4 * sizeof(float),
                                           cudaMemcpyDeviceToHost);
                                std::fprintf(stderr,
                                             "[gpu fast batch L0] rank0 e=%d "
                                             "gateUp[0..3]={%.6f %.6f %.6f %.6f} "
                                             "down[0..3]={%.6f %.6f %.6f %.6f}\n",
                                             moeRouterIdxHost_.data()[0], gu[0],
                                             gu[1], gu[2], gu[3], gd[0], gd[1],
                                             gd[2], gd[3]);
                            }
                        }
                    } else
                    moePerRank: {
                        for (uint32_t r = 0; r < expertUsed; ++r) {
                            const int32_t *row = moeRouterIdxHost_.data() + r;
                            // Same-expert fast path: the whole batch selected the
                            // same expert at rank r (the common prefill case).
                            bool sameExpert = true;
                            const int32_t e0 = row[0];
                            for (uint32_t st = 1; st < seqLen; ++st) {
                                if (row[st * expertUsed] != e0) {
                                    sameExpert = false;
                                    break;
                                }
                            }
                            if (seqLen == 1 || (sameExpert && e0 >= 0)) {
                                const uint32_t e = static_cast<uint32_t>(e0);
                                cacheStat(e);
                                std::string cErr;
                                const uint8_t *downQ = nullptr;
                                const uint8_t *cacheBase =
                                        ensureExpertCached(L, e, cErr, &downQ);
                                if (cacheBase == nullptr) {
                                    errMsg = "qwen35moe expert cache miss: " + cErr;
                                    return false;
                                }
                                // Slice the cached arena: gate at cacheBase, up
                                // one gateSlice later (both [expertFF][H]), down
                                // in the slot's separate region ([H][expertFF]).
                                // PER-LAYER geometry (2026-09-17): this layer's
                                // gate/down types and row strides come from the
                                // slot record — the Ornith Q4_K_M mix changes the
                                // DOWN expert quant type between layers (most
                                // layers down=Q6_K 210 B/block, layers 5-6
                                // down=Q4_K 144 B/block), and feeding the global
                                // (last-layer) type made kQGemv decode Q6K bytes
                                // as Q4K at the wrong row stride -> NaN logits.
                                DeviceMatrix gExps;
                                gExps.q = const_cast<void *>(
                                        static_cast<const void *>(cacheBase));
                                gExps.rows = expertFF;
                                gExps.cols = H;
                                gExps.type = gTypeL;
                                gExps.blocksPerRow = gBlocks;
                                gExps.rowBytes = layGateRowBytes(L);
                                DeviceMatrix uExps = gExps;
                                uExps.q = const_cast<void *>(
                                        static_cast<const void *>(
                                                cacheBase +
                                                static_cast<uint64_t>(expertFF_) *
                                                        layGateRowBytes(L)));
                                DeviceMatrix dExps;
                                dExps.q = const_cast<void *>(
                                        static_cast<const void *>(downQ));
                                dExps.rows = H;
                                dExps.cols = expertFF;
                                dExps.type = dTypeL;
                                dExps.blocksPerRow = dBlocks;
                                dExps.rowBytes = layDownRowBytes(L);
                                if (L < 4) {
                                    const char *dnG = std::getenv(
                                            "TINYCODER_DUMP_MOE");
                                    if (dnG != nullptr && dnG[0] != '\0' &&
                                        dnG[0] != '0') {
                                        cudaStreamSynchronize(g_stream);
                                        float nrm[4];
                                        cudaMemcpy(nrm, s.norm, 4 * sizeof(float),
                                                   cudaMemcpyDeviceToHost);
                                        std::fprintf(
                                                stderr,
                                                "[gpu gatein L%u] r=%u e=%u "
                                                "snorm[0..3]={%.6f %.6f %.6f %.6f} "
                                                "cacheBase=%p\n",
                                                L, r, static_cast<uint32_t>(e0),
                                                nrm[0], nrm[1], nrm[2], nrm[3],
                                                cacheBase);
                                    }
                                }
                                if (seqLen == 1 && gTypeL == kTypeQ8_0 &&
                                    moeFastPath) {
                                    // Pre-quantized decode fast path: s.norm is
                                    // already Q8_32-quantized once into q8kN_
                                    // above; the down input (s.moeGateUp, the silu
                                    // output) is fp32 and needs its own 16-block
                                    // quantize into the transient q8k_.
                                    launchQGemvQ8_0Pre(gExps.q, q8kN_, s.moeGateUp,
                                                       expertFF, gBlocks,
                                                       layGateRowBytes(L));
                                    launchQGemvQ8_0Pre(uExps.q, q8kN_, s.moeGateUp2,
                                                       expertFF, gBlocks,
                                                       layGateRowBytes(L));
                                    kSiluMul<<<(seqLen * expertFF + 255) / 256, 256,
                                               0, g_stream>>>(
                                            s.moeGateUp, s.moeGateUp2,
                                            seqLen * expertFF);
                                    launchQuantQ8_0x32(s.moeGateUp, q8k_, expertFF);
                                    launchQGemvQ8_0Pre(dExps.q, q8k_, s.moeDown, H,
                                                       dBlocks,
                                                       layDownRowBytes(L));
                                    // TINYCODER_DUMP_MOE=1 + L0: dump the fast
                                    // path's per-rank output values so the routed
                                    // contribution can be compared against the
                                    // slow path / CPU reference.
                                    {
                                        const char *dnR = std::getenv(
                                                "TINYCODER_DUMP_MOE");
                                        if (L < 4 && dnR != nullptr &&
                                            dnR[0] != '\0' && dnR[0] != '0') {
                                            cudaStreamSynchronize(g_stream);
                                            float gu[4], gd[4], gwt = moeRouterWgtHost_[r];
                                            cudaMemcpy(gu, s.moeGateUp,
                                                       4 * sizeof(float),
                                                       cudaMemcpyDeviceToHost);
                                            cudaMemcpy(gd, s.moeDown,
                                                       4 * sizeof(float),
                                                       cudaMemcpyDeviceToHost);
                                            std::fprintf(
                                                    stderr,
                                                    "[gpu fast L%u] r=%u e=%u wgt=%.6f "
                                                    "gateUp[0..3]={%.6f %.6f %.6f "
                                                    "%.6f} down[0..3]={%.6f %.6f "
                                                    "%.6f %.6f}\n",
                                                    L, r, static_cast<uint32_t>(e0),
                                                    gwt, gu[0], gu[1], gu[2], gu[3],
                                                    gd[0], gd[1], gd[2], gd[3]);
                                        }
                                    }
                                } else if (!gemv(gExps, s.norm, s.moeGateUp, expertFF,
                                                 H, "ffnGateExps") ||
                                           !gemv(uExps, s.norm, s.moeGateUp2,
                                                 expertFF, H, "ffnUpExps")) {
                                    return false;
                                } else {
                                    kSiluMul<<<(seqLen * expertFF + 255) / 256, 256,
                                               0, g_stream>>>(
                                            s.moeGateUp, s.moeGateUp2,
                                            seqLen * expertFF);
                                    if (!gemv(dExps, s.moeGateUp, s.moeDown, H,
                                              expertFF, "ffnDownExpsMoe")) {
                                        return false;
                                    }
                                }
                                // wgt[s*expertUsed + r] via base+offset (kernel
                                // reads the fixed-rank slice at s.moeWgt + r).
                                kQ35MoeAccumScaled<<<seqLen, 256, 0, g_stream>>>(
                                        s.moeWgt + r, expertUsed, s.moeDown,
                                        s.moeExpertOut, seqLen, H);
                            } else {
                                // Per-token fallback: each token has its own expert
                                // at this rank; run per-token GEMVs with the token's
                                // own cached slice + weight.
                                for (uint32_t st = 0; st < seqLen; ++st) {
                                    const int32_t e = row[st * expertUsed];
                                    if (e < 0) continue;
                                    const uint32_t ue = static_cast<uint32_t>(e);
                                    cacheStat(ue);
                                    std::string cErr;
                                    const uint8_t *downQ = nullptr;
                                    const uint8_t *cacheBase =
                                            ensureExpertCached(L, ue, cErr, &downQ);
                                    if (cacheBase == nullptr) {
                                        errMsg = "qwen35moe expert cache miss: " +
                                                 cErr;
                                        return false;
                                    }
                                    float *gateRow = s.moeGateUp +
                                                     static_cast<size_t>(st) * expertFF;
                                    float *upRow = s.moeGateUp2 +
                                                   static_cast<size_t>(st) * expertFF;
                                    const float *normRow =
                                            s.norm + static_cast<size_t>(st) * H;
                                    float *downRow =
                                            s.moeDown + static_cast<size_t>(st) * H;
                                    float *outRow = s.moeExpertOut +
                                                    static_cast<size_t>(st) * H;
                                    // PER-LAYER geometry (2026-09-17): the
                                    // gate/up GEMVs decode THIS layer's type
                                    // (gTypeL) at THIS layer's row stride; the
                                    // down GEMV decodes dTypeL (Q6_K for most
                                    // layers, Q4_K for layers 5-6 in the Ornith
                                    // Q4_K_M mix) using layDownRowBytes(L).
                                    launchQGemv(gTypeL, cacheBase, normRow,
                                                gateRow, expertFF, H,
                                                layGateRowBytes(L), gBlocks, q8k_,
                                                q8kBytes_);
                                    launchQGemv(gTypeL,
                                                cacheBase +
                                                        static_cast<uint64_t>(expertFF_) *
                                                                layGateRowBytes(L),
                                                normRow, upRow, expertFF, H,
                                                layGateRowBytes(L), gBlocks, q8k_,
                                                q8kBytes_);
                                    kSiluMul<<<(expertFF + 255) / 256, 256, 0,
                                               g_stream>>>(gateRow, upRow, expertFF);
                                    launchQGemv(dTypeL, downQ, gateRow, downRow,
                                                H, expertFF, layDownRowBytes(L),
                                                dBlocks, q8k_, q8kBytes_);
                                    const float g = moeRouterWgtHost_[static_cast<size_t>(st) * expertUsed + r];
                                    kAddScaled1<<<1, 256, 0, g_stream>>>(
                                            g, downRow, outRow, H);
                                }
                            }
                        }// end per-rank loop (generic prefill / slow path)
                    }
                        if (moeTr) {
                            rankLoopMs += std::chrono::duration<double, std::milli>(
                                                  std::chrono::steady_clock::now() -
                                                  tRank0)
                                                  .count();
                        }
                    // TINYCODER_PER_LAYER=1: per-layer MoE block breakdown —
                    // wall time since startMoeBlock split into the instrumented
                    // sub-stages vs the (un-instrumented) gap.  This isolates
                    // the ~1s/token unexplained budget outside fill/router/rank.
                    {
                        const char *perL = std::getenv("TINYCODER_PER_LAYER");
                        if (perL != nullptr && perL[0] != '\0' && perL[0] != '0') {
                            double moeWallMs = std::chrono::duration<double,
                                                                     std::milli>(
                                                       std::chrono::steady_clock::now() - tMoe0)
                                                       .count();
                            // Accumulated so far this forward (per-token
                            // callback runs once per layer, so these are
                            // deltas by construction).
                            std::fprintf(
                                    stderr,
                                    "[moe L%u] wall=%.2f sync=%.2f router=%.2f "
                                    "rank=%.2f fill=%.2f hits=%llu misses=%llu\n",
                                    L, moeWallMs, syncWaitMs, cpuRouterMs,
                                    rankLoopMs, fillMs,
                                    static_cast<unsigned long long>(cacheHits),
                                    static_cast<unsigned long long>(cacheMisses));
                        }
                    }
                } else {
                    // ---- Option B (no expert cache): the callback ran the
                    // full routed experts into moeFfnOutHost_; H2D the output.
                    // Reached when the expert cache was NOT built (disabled via
                    // TINYCODER_MOE_CACHE=0, build OOM'd, or
                    // TINYCODER_MOE_FALLBACK=1).  Cache-mode is the qwen35moe
                    // DEFAULT (2026-09-17).
                    if (!moeFallbackWarned) {
                        std::fprintf(stderr,
                                     "[gpu] L=%u: CPU-expert hybrid (GPU "
                                     "trunk + CPU experts; expert cache off — "
                                     "TINYCODER_MOE_CACHE=0 or fallback)\n",
                                     L);
                        moeFallbackWarned = true;
                    }
                    // on g_stream and sync so kAddScaled sees the copy.
                    cudaMemcpyAsync(s.moeExpertOut, moeFfnOutHost_[buf].data(),
                                    normBytes, cudaMemcpyHostToDevice, g_stream);
                    cudaStreamSynchronize(g_stream);
                }
                // ---- Option D branch completes to here (goto target): the
                // routed output (resident GPU + non-resident CPU ranks) is in
                // s.moeExpertOut; the shared expert + residual are added below
                // exactly as in every other path.
            moeAddShared:
                // 6) Add the shared expert into the routed output
                //    (moeShexpDown was computed in step 2).
                if (w.ffnGateShexp.q != nullptr && w.ffnUpShexp.q != nullptr &&
                    w.ffnDownShexp.q != nullptr && sharedFF > 0) {
                    kAddScaled<<<seqLen, 256, 0, g_stream>>>(
                            s.moeShexpGate, s.moeShexpDown, s.moeExpertOut, seqLen,
                            H);
                    // TINYCODER_DUMP_MOE=1: dump the shared-expert contribution
                    // (sigmoid * sharedDown) so routed vs shared can be isolated.
                    {
                        const char *dnS = std::getenv("TINYCODER_DUMP_MOE");
                        if (moeCpuFn_ != nullptr && L == 0 && seqLen >= 1 &&
                            dnS != nullptr && dnS[0] != '\0' && dnS[0] != '0') {
                            cudaStreamSynchronize(g_stream);
                            float sd[4], sg = 0.0f;
                            cudaMemcpy(sd, s.moeShexpDown, 4 * sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            cudaMemcpy(&sg, s.moeShexpGate, sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            std::fprintf(stderr,
                                         "[gpu L0 shared] sig=%.6f shexpDown"
                                         "[0..3]={%.6f %.6f %.6f %.6f}\n",
                                         sg, sd[0], sd[1], sd[2], sd[3]);
                        }
                    }
                }
                // TINYCODER_DUMP_MOE=1: dump L0's completed MoE output (routed
                // experts + shared expert, BEFORE the residual add) so the
                // cache-driven decode fast path can be bisected against the
                // TINYCODER_MOE_SLOW generic path and the pure-CPU fallback
                // (both verified passing the SequentialDecode gate).
                // TOKEN-0 ONLY (the batch token order is deterministic):  dump
                // moeExpertOut[0..3] so it can be compared against the CPU
                // reference L0 ffnOut for the same token.
                {
                    const char *dnT = std::getenv("TINYCODER_DUMP_MOE");
                    if (moeCpuFn_ != nullptr && L == 0 && dnT != nullptr &&
                        dnT[0] != '\0' && dnT[0] != '0' && seqLen >= 1) {
                        cudaStreamSynchronize(g_stream);
                        float mo[4];
                        cudaMemcpy(mo, s.moeExpertOut, 4 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        std::fprintf(stderr,
                                     "[gpu L0 moeOut] moeExpertOut[0..3]={%.6f "
                                     "%.6f %.6f %.6f}\n",
                                     mo[0], mo[1], mo[2], mo[3]);
                    }
                }
                {
                    const char *dnM = std::getenv("TINYCODER_DUMP_MOE");
                    if (dnM != nullptr && dnM[0] != '\0' && dnM[0] != '0' &&
                        moeCpuFn_ != nullptr) {
                        // Dump the norm for THIS layer (all layers, token 0)
                        // so the first divergence layer can be identified
                        // against the CPU `[cpunorm]` dumps.
                        cudaStreamSynchronize(g_stream);
                        float n0[4];
                        cudaMemcpy(n0, s.norm, 4 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        std::fprintf(stderr,
                                     "[gpu norm] L%u norm[0..3]={%.6f %.6f "
                                     "%.6f %.6f}\n",
                                     L, n0[0], n0[1], n0[2], n0[3]);
                    }
                }
                {
                    const char *dnM = std::getenv("TINYCODER_DUMP_MOE");
                    if (L == 0 && dnM != nullptr && dnM[0] != '\0' &&
                        dnM[0] != '0' && moeCpuFn_ != nullptr) {
                        cudaStreamSynchronize(g_stream);
                        float mo[8], norm0[4];
                        cudaMemcpy(mo, s.moeExpertOut, 8 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        cudaMemcpy(norm0, s.norm, 4 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        std::fprintf(stderr,
                                     "[gpu moe] L0 moeExpertOut[0..7]={%.6f "
                                     "%.6f %.6f %.6f %.6f %.6f %.6f %.6f} "
                                     "norm[0..3]={%.6f %.6f %.6f %.6f}\n",
                                     mo[0], mo[1], mo[2], mo[3], mo[4], mo[5],
                                     mo[6], mo[7], norm0[0], norm0[1], norm0[2],
                                     norm0[3]);
                    }
                }
                // Flip for the next layer.
                moeBufFlip_ ^= 1u;
            } else {
                // ---------- Full-GPU routed MoE FFN (softmax top-k + shared expert) ----------
                // 1) Router logits = ffn_gate_inp (F32 [expertCount x H]) @ norm.
                if (!gemv(w.ffnGateInpMoe, s.norm, s.moeLogits, expertCount, H,
                          "ffnGateInpMoe")) {
                    return false;
                }
                // 2) softmax over ALL experts + top-k + renormalize (norm_w clamp).
                const uint32_t moeShmem =
                        sizeof(float) * (expertCount + 2 * expertUsed);
                kQ35MoeRouter<<<seqLen, 256, moeShmem, g_stream>>>(
                        s.moeLogits, s.moeIdx, s.moeWgt, s.moeWsum, seqLen,
                        expertCount, expertUsed);
                // Copy the tiny top-k selection back to the host: the per-expert
                // GEMVs slice rows from the packed expert matrices and need the
                // SELECTED expert ids (moeIdx) as a host-side row offset per token.
                // The renormalized weights (moeWgt) are copied too: the per-token
                // fallback scales by the token's OWN rank-r weight, and reading
                // that from the device buffer (s.moeWgt) on the host would be UB.
                if (moeIdxHost_.size() < static_cast<size_t>(seqLen) * expertUsed) {
                    moeIdxHost_.resize(static_cast<size_t>(seqLen) * expertUsed);
                    moeWgtHost_.resize(static_cast<size_t>(seqLen) * expertUsed);
                }
                cudaMemcpyAsync(moeIdxHost_.data(), s.moeIdx,
                                static_cast<size_t>(seqLen) * expertUsed *
                                        sizeof(int32_t),
                                cudaMemcpyDeviceToHost, g_stream);
                cudaMemcpyAsync(moeWgtHost_.data(), s.moeWgt,
                                static_cast<size_t>(seqLen) * expertUsed *
                                        sizeof(float),
                                cudaMemcpyDeviceToHost, g_stream);
                cudaStreamSynchronize(g_stream);

                // 3) Per-expert GEMVs.  Expert e of the packed ffnGateExps /
                //    ffnUpExps spans rows [e*expertFF, (e+1)*expertFF) (cols == H);
                //    ffnDownExpsMoe spans rows [e*H, (e+1)*H) (cols == expertFF).
                //    The routing weight array is laid out [seqLen][expertUsed]
                //    (fixed rank r).  We iterate ranks ASCENDING and accumulate
                //    each rank's weighted expert output into moeExpertOut.  The
                //    per-expert gate/up/down buffers are reused across ranks.
                cudaMemsetAsync(s.moeExpertOut, 0,
                                static_cast<uint64_t>(seqLen) * H * sizeof(float),
                                g_stream);
                // 3) Per-expert GEMVs + SwiGLU + down + weighted accumulation.
                //
                // The per-expert matrices are PACKED: expert e occupies rows
                // [e*expertFF, (e+1)*expertFF) of ffnGateExps / ffnUpExps
                // (cols == H) and rows [e*H, (e+1)*H) of ffnDownExpsMoe
                // (cols == expertFF).  The GPU kernels read a contiguous row
                // range (rowStart = e*rowsPerExpert), so the driver launches ONE
                // GEMV per selected expert with the sliced row base; the whole
                // token batch shares the expert at rank r (the router kernel's
                // top-k builds the SAME rank ordering for every token only when
                // the probs are strictly ordered — but rank r may differ across
                // tokens!).  Exact parity with the CPU (matMulVecRows per token)
                // requires per-(token, rank) GEMVs: the CPU accumulates
                // routingWeight[e] * down(e, token).  We batch over tokens ONLY
                // when the whole batch selected the SAME expert at rank r (the
                // overwhelmingly common prefill case); otherwise fall back to a
                // per-token GEMV with the token's own expert slice + weight.
                //
                // moeWgt is laid out [seqLen][expertUsed] (rank r fixed); the
                // kQ35MoeAccumScaled kernel takes wgt[token][r] with
                // wgtStride == expertUsed and accumulates out += w * src.
                for (uint32_t r = 0; r < expertUsed; ++r) {
                    const int32_t *row = moeIdxHost_.data() + r;
                    const float *wgtRow = moeWgtHost_.data() + r;
                    // Detect the same-expert fast path: all tokens choose the
                    // same expert at this rank.
                    bool sameExpert = true;
                    const int32_t e0 = row[0];
                    for (uint32_t st = 1; st < seqLen; ++st) {
                        if (row[st * expertUsed] != e0) {
                            sameExpert = false;
                            break;
                        }
                    }
                    if (seqLen == 1 || (sameExpert && e0 >= 0)) {
                        const uint32_t e = static_cast<uint32_t>(e0);
                        // Row-sliced copies of the expert's matrices: the `gemv`
                        // lambda handles decode (launchQGemv) AND prefill (fp16
                        // cuBLAS GEMM) uniformly.  `gemv` dequantizes the slice
                        // into wF16_ on demand (cached by q pointer).
                        DeviceMatrix gExps = w.ffnGateExps;
                        gExps.q = static_cast<void *>(
                                static_cast<uint8_t *>(w.ffnGateExps.q) +
                                static_cast<size_t>(e) * expertFF *
                                        w.ffnGateExps.rowBytes);
                        gExps.rows = expertFF;
                        DeviceMatrix uExps = w.ffnUpExps;
                        uExps.q = static_cast<void *>(
                                static_cast<uint8_t *>(w.ffnUpExps.q) +
                                static_cast<size_t>(e) * expertFF *
                                        w.ffnUpExps.rowBytes);
                        uExps.rows = expertFF;
                        DeviceMatrix dExps = w.ffnDownExpsMoe;
                        dExps.q = static_cast<void *>(
                                static_cast<uint8_t *>(w.ffnDownExpsMoe.q) +
                                static_cast<size_t>(e) * H *
                                        w.ffnDownExpsMoe.rowBytes);
                        dExps.rows = H;
                        if (!gemv(gExps, s.norm, s.moeGateUp, expertFF, H,
                                  "ffnGateExps") ||
                            !gemv(uExps, s.norm, s.moeGateUp2, expertFF, H,
                                  "ffnUpExps")) {
                            return false;
                        }
                        // SwiGLU: gate = silu(gate) * up (in-place, [seqLen][ff]).
                        kSiluMul<<<(seqLen * expertFF + 255) / 256, 256, 0,
                                   g_stream>>>(s.moeGateUp, s.moeGateUp2,
                                               seqLen * expertFF);
                        if (!gemv(dExps, s.moeGateUp, s.moeDown, H, expertFF,
                                  "ffnDownExpsMoe")) {
                            return false;
                        }
                        // Accumulate: moeExpertOut[s] += wgt[s][r] * moeDown[s].
                        // kQ35MoeAccumScaled reads `wgt[s*stride]` from the BASE
                        // pointer passed in, so pass s.moeWgt + r to reach the
                        // per-(token, rank) weight moeWgt[s*expertUsed + r].
                        kQ35MoeAccumScaled<<<seqLen, 256, 0, g_stream>>>(
                                s.moeWgt + r, expertUsed, s.moeDown, s.moeExpertOut,
                                seqLen, H);
                    } else {
                        // Per-token fallback: each token has its own expert.
                        for (uint32_t st = 0; st < seqLen; ++st) {
                            const int32_t e = row[st * expertUsed];
                            if (e < 0) continue;
                            const uint32_t ue = static_cast<uint32_t>(e);
                            const uint32_t gRow0 = ue * expertFF;
                            const uint32_t dRow0 = ue * H;
                            float *gateRow = s.moeGateUp + static_cast<size_t>(st) * expertFF;
                            float *upRow = s.moeGateUp2 + static_cast<size_t>(st) * expertFF;
                            float *normRow = s.norm + static_cast<size_t>(st) * H;
                            float *downRow = s.moeDown + static_cast<size_t>(st) * H;
                            float *outRow = s.moeExpertOut + static_cast<size_t>(st) * H;
                            DeviceMatrix gExps = w.ffnGateExps;
                            gExps.q = static_cast<void *>(
                                    static_cast<uint8_t *>(w.ffnGateExps.q) +
                                    static_cast<size_t>(gRow0) * w.ffnGateExps.rowBytes);
                            gExps.rows = expertFF;
                            DeviceMatrix uExps = w.ffnUpExps;
                            uExps.q = static_cast<void *>(
                                    static_cast<uint8_t *>(w.ffnUpExps.q) +
                                    static_cast<size_t>(gRow0) * w.ffnUpExps.rowBytes);
                            uExps.rows = expertFF;
                            DeviceMatrix dExps = w.ffnDownExpsMoe;
                            dExps.q = static_cast<void *>(
                                    static_cast<uint8_t *>(w.ffnDownExpsMoe.q) +
                                    static_cast<size_t>(dRow0) * w.ffnDownExpsMoe.rowBytes);
                            dExps.rows = H;
                            launchQGemv(gExps.type, gExps.q, normRow, gateRow,
                                        gExps.rows, gExps.cols, gExps.rowBytes,
                                        gExps.blocksPerRow, q8k_, q8kBytes_);
                            launchQGemv(uExps.type, uExps.q, normRow, upRow,
                                        uExps.rows, uExps.cols, uExps.rowBytes,
                                        uExps.blocksPerRow, q8k_, q8kBytes_);
                            kSiluMul<<<(expertFF + 255) / 256, 256, 0, g_stream>>>(
                                    gateRow, upRow, expertFF);
                            launchQGemv(dExps.type, dExps.q, gateRow, downRow,
                                        dExps.rows, dExps.cols, dExps.rowBytes,
                                        dExps.blocksPerRow, q8k_, q8kBytes_);
                            const float g = wgtRow[st * expertUsed];
                            kAddScaled1<<<1, 256, 0, g_stream>>>(g, downRow, outRow,
                                                                 H);
                        }
                    }
                }

                // ---------- Shared expert (sigmoid-gated SwiGLU) ----------
                if (w.ffnGateShexp.q != nullptr && w.ffnUpShexp.q != nullptr &&
                    w.ffnDownShexp.q != nullptr && sharedFF > 0) {
                    // Gate: g[s] = sigmoid(ffn_gate_inp_shexp @ norm[s]).
                    kQ35MoeShexpGate<<<seqLen, 256, 0, g_stream>>>(
                            s.norm, static_cast<const float *>(w.ffnGateInpShexp.q),
                            s.moeShexpGate, seqLen, H, H);
                    // gate + up (shared-expert matrices, full [sharedFF x H]).
                    if (!gemv(w.ffnGateShexp, s.norm, s.moeShexpGateUp, sharedFF, H,
                              "ffnGateShexp") ||
                        !gemv(w.ffnUpShexp, s.norm, s.moeShexpGateUp2, sharedFF, H,
                              "ffnUpShexp")) {
                        return false;
                    }
                    // SwiGLU: [seqLen][sharedFF].
                    kSiluMul<<<(seqLen * sharedFF + 255) / 256, 256, 0, g_stream>>>(
                            s.moeShexpGateUp, s.moeShexpGateUp2, seqLen * sharedFF);
                    // Down: full [H x sharedFF] matrix.
                    if (!gemv(w.ffnDownShexp, s.moeShexpGateUp, s.moeShexpDown, H,
                              sharedFF, "ffnDownShexp")) {
                        return false;
                    }
                    // Accumulate: moeExpertOut[s] += gate[s] * shexpDown[s].
                    kAddScaled<<<seqLen, 256, 0, g_stream>>>(
                            s.moeShexpGate, s.moeShexpDown, s.moeExpertOut, seqLen,
                            H);
                }
            }// end full-GPU MoE (else of moeCpuFn_)

            // ---- Residual: hidden += moeExpertOut ----
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.moeExpertOut, seqLen * H);
            phaseMark(L, 2);// ev2: after rank + addScaled + residual enqueues
            if (moeTr) endMoeBlock();
        }

        cudaError_t llErr = cudaStreamSynchronize(g_stream);
        if (llErr != cudaSuccess) {
            errMsg = std::string("qwen35moe prefix sync: ") +
                     cudaGetErrorString(llErr);
            return false;
        }
        kvPos_ += seqLen;

        // ---- Full offload: final RMSNorm + LM head ----
        if (finalNorm_) {
            kRMSNormRow<<<seqLen, 1024, 1024 * sizeof(float), g_stream>>>(
                    s.hidden, s.hidden, finalNorm_, H, seqLen, 1e-6f);
        }
        const void *lmQ = lmHeadQ_ != nullptr ? lmHeadQ_ : embedQ_;
        uint32_t lmType = lmHeadQ_ != nullptr ? lmHeadType_ : embedType_;
        uint32_t lmRowBytes =
                lmHeadQ_ != nullptr ? lmHeadRowBytes_ : embedRowBytes_;
        uint32_t lmBlocks = hostBlocksPerRow(lmType, H);
        // logitsCapOneRow_: ONE-row logits scratch -> last token only.
        if (computeAllLogits && !logitsCapOneRow_) {
            for (uint32_t si = 0; si < seqLen; ++si) {
                launchQGemv(lmType, lmQ, s.hidden + si * H,
                            s.logits + si * geom_.vocabSize, geom_.vocabSize, H,
                            lmRowBytes, lmBlocks, q8k_, q8kBytes_);
            }
            cudaMemcpyAsync(logitsOut, s.logits,
                            seqLen * geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        } else {
            uint32_t si = seqLen - 1;
            launchQGemv(lmType, lmQ, s.hidden + si * H, s.logits,
                        geom_.vocabSize, H, lmRowBytes, lmBlocks, q8k_, q8kBytes_);
            cudaMemcpyAsync(logitsOut, s.logits, geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        }
        cudaError_t fin = cudaStreamSynchronize(g_stream);
        if (fin != cudaSuccess) {
            errMsg = std::string("qwen35moe lmhead sync: ") +
                     cudaGetErrorString(fin);
            return false;
        }
        // ---- Phase profile accumulation (TINYCODER_PHASE_PROFILE=1) ----
        // All events are complete (we just synced the stream).  Per layer:
        //   trunkDev[i]  = ev1-ev0  device time: trunk + D2H (stream-ordered)
        //   rankDev[i]   = ev2-ev1  device time: rank GEMVs + addScaled +
        //                           residual (the host gap while the router/
        //                           fills ran is NOT on the stream, so it does
        //                           not inflate this span)
        // Summed over layers vs devMs (= device span moeEvStart..moeEvEnd,
        // which includes the LM head) gives:
        //   deviceTrunkMs / deviceRankMs: true GPU work per phase
        //   deviceOtherMs  = devMs - trunk - rank: LM head + tails
        //   hostBubbleMs   = totalMs - devMs: HOST-side time the GPU was idle
        //                    (router + fills + sync waits + enqueue) -- the
        //                    CPU/GPU overlap lever.
        if (phaseProf && !phaseEv.empty()) {
            double devTrunk = 0.0, devRank = 0.0;
            for (size_t li = 0; li < phaseEv.size(); ++li) {
                float t0 = 0.0f, t1 = 0.0f;
                cudaEventElapsedTime(&t0, phaseEv[li][0], phaseEv[li][1]);
                cudaEventElapsedTime(&t1, phaseEv[li][1], phaseEv[li][2]);
                devTrunk += t0;
                devRank += t1;
            }
            float devMs2 = 0.0f;
            if (moeEvStart_ && moeEvEnd_) {
                cudaEventElapsedTime(&devMs2, moeEvStart_, moeEvEnd_);
            }
            double wallMs = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - tF0)
                                    .count();
            double devOther = static_cast<double>(devMs2) - devTrunk - devRank;
            if (devOther < 0.0) devOther = 0.0;
            std::fprintf(stderr,
                         "[phase] wall=%.1f dev=%.1f hostBubble=%.1f "
                         "devTrunk=%.1f(%u L) devRank=%.1f(%u L) devOther=%.1f "
                         "perLayerTrunk=%.2f perLayerRank=%.2f "
                         "perLayerBubble=%.2f\n",
                         wallMs, static_cast<double>(devMs2),
                         wallMs - static_cast<double>(devMs2), devTrunk,
                         static_cast<unsigned>(phaseEv.size()), devRank,
                         static_cast<unsigned>(phaseEv.size()), devOther,
                         devTrunk / static_cast<double>(phaseEv.size()),
                         devRank / static_cast<double>(phaseEv.size()),
                         (wallMs - static_cast<double>(devMs2)) /
                                 static_cast<double>(phaseEv.size()));
            // LM-head device span: devOther / (the final sync is not on the
            // timeline) approximates the finalNorm + LM-head GEMV + D2H.
        }
        if (moeTr) {
            double totalMs = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - tF0)
                                     .count();
            // Device elapsed (g_stream): everything the GPU actually ran from
            // the token H2D to the lmhead sync — the true GPU busy time.  The
            // difference (totalMs - devMs) is host-side enqueue/fill/CPU work
            // that the GPU was NOT busy with.
            float devMs = 0.0f;
            if (moeEvStart_ && moeEvEnd_) {
                cudaEventRecord(moeEvEnd_, g_stream);
                cudaEventSynchronize(moeEvEnd_);
                cudaEventElapsedTime(&devMs, moeEvStart_, moeEvEnd_);
            }
            std::fprintf(stderr,
                         "[moe fwd] seqLen=%u total=%.2f ms dev=%.1f "
                         "cacheHits=%llu cacheMisses=%llu missMB=%.0f "
                         "layerGpuMs=%.1f moeBlockMs=%.1f cpuRouterMs=%.1f "
                         "syncWaitMs=%.1f rankLoopMs=%.1f fillMs=%.1f%s\n",
                         seqLen, totalMs, devMs,
                         static_cast<unsigned long long>(cacheHits),
                         static_cast<unsigned long long>(cacheMisses),
                         cacheMissBytes / 1048576.0, layerGpuMs, moeBlockMs,
                         cpuRouterMs, syncWaitMs, rankLoopMs, fillMs,
                         moeFallbackWarned ? "  ** CPU-EXPERT FALLBACK ACTIVE **"
                                           : "");
        }
        // Destroy the phase-profile events (env-gated probe; no-op otherwise).
        for (auto &row: phaseEv) {
            for (auto &ev: row) {
                cudaEventDestroy(ev);
            }
        }
        return true;
    }

    bool GPUModel::forward(const std::vector<int32_t> &tokens, bool computeAllLogits,
                           float *logitsOut, std::string &errMsg) {
        if (!allocated_) {
            errMsg = "GPU model not uploaded";
            return false;
        }
        if (tokens.empty()) return true;
        auto t0 = std::chrono::steady_clock::now();
#define TIMING_TAG(what)                                               \
    do {                                                               \
        if (g_verbose) {                                               \
            double ms = std::chrono::duration<double, std::milli>(     \
                                std::chrono::steady_clock::now() - t0) \
                                .count();                              \
            std::fprintf(stderr, "[gpu fwd %6.2f ms] %s\n", ms, what); \
        }                                                              \
    } while (0)

        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t nGpu = geom_.numGpuLayers;
        if (nGpu == 0) nGpu = geom_.numLayers;
        uint32_t H = geom_.hiddenSize, I = geom_.intermediateSize;
        uint32_t nHeads = geom_.numAttentionHeads, nKV = geom_.numKVHeads;
        uint32_t hd = geom_.headDim;
        uint32_t qLen = nHeads * hd, kvLen = nKV * hd;
        bool fullOffload = (nGpu == geom_.numLayers);

        if (!ensureScratch(seqLen, errMsg)) return false;
        DeviceScratch &s = scratch_;
        cudaMemcpyAsync(s.tokens, tokens.data(), seqLen * sizeof(int32_t),
                        cudaMemcpyHostToDevice, g_stream);
        cudaStreamSynchronize(g_stream);

        uint32_t pos = static_cast<uint32_t>(kvPos_);

        // ---- CUDA-graph decode fast path ----
        // When the graph is compiled, seqLen==1 decode (greedy) replays it:
        // replayDecodeGraph updates the device pos scalar -> cudaGraphLaunch
        // -> sync -> copy logits to the caller's buffer.  The first eligible
        // decode BUILDS the graph: captureDecodeGraph begins recording, the
        // body below runs with graphCaptureActive_=true (device-pos kernel
        // variants, no host sync points) and records into graph_; the capture
        // run itself computes the first token's logits; then the end-of-forward
        // capture block calls cudaStreamEndCapture + instantiateGraph().  A
        // capture failure reroutes all subsequent decodes to the eager path.
        const bool graphWanted = (seqLen == 1) && !computeAllLogits &&
                                 fullOffload && (geom_.architecture == 0) &&
                                 !graphCaptureFailed_;
        if (graphWanted && !graphBuilt_ && !graphCaptureActive_) {
            // Build: begin the capture.  The body below runs with
            // graphCaptureActive_=true and records into graph_.
            const char *ge = std::getenv("TINYCODER_GPU_GRAPH");
            const bool gEnable = ge == nullptr || ge[0] == '\0' || ge[0] != '0';
            if (gEnable) {
                if (!captureDecodeGraph(pos)) {
                    graphCaptureFailed_ = true;
                    std::fprintf(stderr,
                                 "[gpu] CUDA-graph decode disabled "
                                 "(capture init failed)\n");
                } else {
                    std::fprintf(stderr,
                                 "[gpu] CUDA-graph decode: capturing "
                                 "(first decode token)\n");
                }
            } else {
                graphCaptureFailed_ = true;
            }
        }
        if (graphWanted && graphBuilt_ && !graphCaptureActive_) {
            // Steady-state replay.
            if (replayDecodeGraph(pos, logitsOut, errMsg)) {
                kvPos_ += seqLen;
                return true;
            }
            // Replay failure: fall through to the eager path once.
            std::fprintf(stderr, "[gpu] CUDA-graph replay failed: %s\n",
                         errMsg.c_str());
            graphCaptureFailed_ = true;
        }

        // ---- Embedding ----
        {
            uint32_t blocksPerRow = hostBlocksPerRow(embedType_, H);
            const uint8_t *eQ = reinterpret_cast<const uint8_t *>(embedQ_);
            if (embedType_ == kTypeQ5_0) {
                kEmbedDequantSmall32<kTypeQ5_0><<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ8_0) {
                kEmbedDequantSmall32<kTypeQ8_0><<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeIQ4NL) {
                kEmbedDequantSmall32<kTypeIQ4NL><<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeIQ3S) {
                kEmbedDequantIQ3S<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ4K) {
                kEmbedDequantQ4K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ5K) {
                kEmbedDequantQ5K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else {
                kEmbedDequant<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            }
        }
        TIMING_TAG("embed queued");

        // ---- Layer loop (only the offloaded prefix) ----
        cudaEvent_t evL0 = nullptr, evL1 = nullptr;
        // LM-head stage events (final RMSNorm + LM-head GEMV + D2H); only used
        // by the per-stage trace.
        cudaEvent_t evLmHeadS = nullptr, evLmHeadE = nullptr,
                    evLmMid = nullptr;
        std::vector<cudaEvent_t> evQKV, evAttn, evFFN;
        // per-matrix decode breakdown: attnO, gate(+up, up to silu),
        // down.  Only meaningful for seqLen == 1 (decode).
        std::vector<cudaEvent_t> evMtxAttnO, evMtxGateUp, evMtxDown;
        // Per-stage device-event trace.  TINYCODER_STAGE_TRACE=1 enables it
        // WITHOUT the verbose path's three cudaStreamSynchronize() per layer
        // (the "[gpu loop] L=.. " prints), which would inflate every measured
        // stage.  With STAGE_TRACE alone the events are recorded back-to-back
        // on the stream and read once after the loop-completion sync, so the
        // reported ms are the true kernel times.
        const bool stageTrace = std::getenv("TINYCODER_STAGE_TRACE") != nullptr;
        const bool traceStages = (g_verbose || stageTrace) && seqLen == 1;
        // During graph capture the body must not call ANY host-side CUDA API
        // (cudaEventRecord/cudaStreamSynchronize would be captured or fail);
        // the capture run suppresses the g_verbose event timeline.
        const bool hostEventTrace =
                (g_verbose || stageTrace) && !graphCaptureActive_;
        if (hostEventTrace) {
            cudaEventCreate(&evL0);
            cudaEventCreate(&evL1);
            cudaEventRecord(evL0, g_stream);
        }
        if (traceStages && !graphCaptureActive_) {
            evQKV.resize(nGpu);
            evAttn.resize(nGpu);
            evFFN.resize(nGpu);
            evMtxAttnO.resize(nGpu);
            evMtxGateUp.resize(nGpu);
            evMtxDown.resize(nGpu);
            for (uint32_t i = 0; i < nGpu; ++i) {
                cudaEventCreate(&evQKV[i]);
                cudaEventCreate(&evAttn[i]);
                cudaEventCreate(&evFFN[i]);
                cudaEventCreate(&evMtxAttnO[i]);
                cudaEventCreate(&evMtxGateUp[i]);
                cudaEventCreate(&evMtxDown[i]);
            }
            cudaEventCreate(&evLmHeadS);
            cudaEventCreate(&evLmHeadE);
            cudaEventCreate(&evLmMid);
        }
        for (uint32_t L = 0; L < nGpu; ++L) {
            if (L == 0) TIMING_TAG("layer loop start");
            const DeviceLayer &w = layers_[L];
            // Byte-offset the cache base: kvLayerStride_ bakes in the element
            // size selected at upload (fp16 when kvHalf_); the store/attention
            // kernels are templated on the cache element type and reinterpret
            // this pointer.  kvLayerOffset() maps L to its slot (identity for
            // the dense arch, floor(L/interval) for qwen35/MoE, whose recurrent
            // layers store no K/V).
            void *kvK = static_cast<uint8_t *>(kvK_) + kvLayerOffset(L);
            void *kvV = static_cast<uint8_t *>(kvV_) + kvLayerOffset(L);
            // Set by the fused-QKV branch (bias adds live in the GEMV
            // epilogues there); the seqLen==1 vs prefill split below reads it.
            bool biasesFolded = false;

            // Attention RMSNorm -> s.norm (OUT-OF-PLACE: s.hidden is the
            // residual stream and must survive until the residual add below).
            // Grid = seqLen: the kernel uses ONE BLOCK PER ROW (row ==
            // blockIdx.x), so `(seqLen + 255) / 256` would only normalize the
            // first row for seqLen <= 256.
            // Q8_1 fold for the QKV dp4a path (2026-10-10, plan section 9.26
            // lever 2): the attention rms_norm also emits the Q8_1 activation
            // the fused Q/K dp4a bodies consume (the V float path keeps
            // reading the fp32 row, which the kernel writes as before).
            // TINYCODER_QKV_DP4A=0 / TINYCODER_FUSE_Q81_PROD=0 restore the
            // separate-float path; q8k_ is sized by the combine fold guard
            // below (1920 B for this model's 1536-wide rows).
            bool rms1Q81Fold = false;
            if (seqLen == 1 && qkvDp4a() && fuseQ81Prod() && H == 1536) {
                const uint64_t needQ =
                        static_cast<uint64_t>(H / 32u) * kQ8_1_STRIDE;
                if (q8k_ != nullptr && q8kBytes_ >= needQ) {
                    kRMSNormRowQ81<<<seqLen, 1024, 1024 * sizeof(float),
                                     g_stream>>>(
                            s.hidden, s.norm, w.rmsNormAttn, q8k_, H, seqLen,
                            1e-6f);
                    rms1Q81Fold = true;
                }
            }
            if (!rms1Q81Fold) {
                kRMSNormRow<<<seqLen, 1024, 1024 * sizeof(float), g_stream>>>(
                        s.hidden, s.norm, w.rmsNormAttn, H, seqLen, 1e-6f);
            }

            // Q/K/V projections read the NORMED hidden (s.norm), not the
            // residual stream (s.hidden).
            if (seqLen == 1) {
                // Fused Q+K for the common dense q2_k case: attn_q AND attn_k
                // are both Q2_K against the SAME activation (s.norm).  One
                // launch computes both matrices (4-way block batching, shared
                // x reads), replacing two serial kQGemv launches.  The per-matrix
                // math is byte-identical to the separate kernels.  Only when
                // both types match Q2_K (this model's attn_q/attn_k); the bias
                // adds (below) still apply to both outputs.
                const bool fuseQK = (w.attnQ.type == kTypeQ2K &&
                                     w.attnK.type == kTypeQ2K);
                const uint32_t qRows = w.attnQ.rows;
                const uint32_t kRows = w.attnK.rows;
                // Fused Q+K+V (2026-10-10): one launch for all three matrices
                // when q/k are Q2_K and v is Q4_K (this model).  Replaces
                // fused-QK + the Q8_K quantize + the v GEMV, and folds the
                // three bias adds into the epilogues (bit-exact -- same float
                // add kAddBias performed).  TINYCODER_FUSE_QKV=0 opts out.
                const char *fqkvE = std::getenv("TINYCODER_FUSE_QKV");
                const bool fuseQKV = fuseQK && w.attnV.type == kTypeQ4K &&
                                     (fqkvE == nullptr || fqkvE[0] != '0');
                if (fuseQKV) {
                    const uint32_t nQBlocks =
                            (qRows + w.attnV.rows + 7) / 8;
                    kQGemvFusedQKV<<<nQBlocks, dim3(32, 8), 0, g_stream>>>(
                            static_cast<const uint8_t *>(w.attnQ.q),
                            static_cast<const uint8_t *>(w.attnK.q),
                            static_cast<const uint8_t *>(w.attnV.q), s.norm,
                            s.q, s.k, s.v, qRows, kRows, w.attnV.rows,
                            w.attnQ.cols, w.attnQ.rowBytes, w.attnK.rowBytes,
                            w.attnV.rowBytes, w.attnQ.blocksPerRow,
                            w.attnK.blocksPerRow, w.attnV.blocksPerRow,
                            w.attnQBias, w.attnKBias, w.attnVBias,
                            rms1Q81Fold ? q8k_ : nullptr);
                    biasesFolded = true;
                } else if (fuseQK) {
                    // Grid = ceil(qRows/8) warps; K rows are handled only by
                    // warps with row < kRows (kRows == 256 <= qRows == 1536).
                    const uint32_t nQBlocks = (qRows + 7) / 8;
                    kQGemvFusedQK_Q2K<<<nQBlocks, dim3(32, 8), 0, g_stream>>>(
                            static_cast<const uint8_t *>(w.attnQ.q),
                            static_cast<const uint8_t *>(w.attnK.q), s.norm, s.q,
                            s.k, qRows, kRows, w.attnQ.cols, w.attnQ.rowBytes,
                            w.attnK.rowBytes, w.attnQ.blocksPerRow,
                            w.attnK.blocksPerRow);
                } else {
                    launchQGemv(w.attnQ.type, w.attnQ.q, s.norm, s.q,
                                w.attnQ.rows, w.attnQ.cols, w.attnQ.rowBytes,
                                w.attnQ.blocksPerRow, q8k_, q8kBytes_);
                    launchQGemv(w.attnK.type, w.attnK.q, s.norm, s.k,
                                w.attnK.rows, w.attnK.cols, w.attnK.rowBytes,
                                w.attnK.blocksPerRow, q8k_, q8kBytes_);
                }
                if (!biasesFolded) {
                    launchQGemv(w.attnV.type, w.attnV.q, s.norm, s.v,
                                w.attnV.rows, w.attnV.cols, w.attnV.rowBytes,
                                w.attnV.blocksPerRow, q8k_, q8kBytes_);
                }
            } else {
                // cuBLAS fp16 GEMM (batch): cublasGemmEx tensor-op requires BOTH
                // A and B in fp16 (fp32 x fp16 returns CUBLAS_STATUS_NOT_SUPPORTED),
                // so keep an fp16 twin of the RMSNorm'd hidden here.
                kF32ToF16<<<(seqLen * H + 511) / 512, 256, 0, g_stream>>>(
                        s.norm, reinterpret_cast<__half2 *>(s.hiddenF16),
                        (seqLen * H) / 2);
                // out[seqLen][rows] = x[seqLen][cols] @ W[rows][cols]^T
                // CUBLAS is column-major: put W in the A slot (op=T gives
                // W^T) and x in the B slot (op=N), with C sized [rows][seqLen]
                // in column-major order == the row-major [seqLen][rows] output
                // buffer.  (The old transposed parameterization -- m=seqLen,
                // n=rows, ldc=seqLen -- wrote element (i,j) at flat[i+j*seqLen]
                // instead of flat[i*rows+j], scrambling every matrix output
                // whenever seqLen > 1.  Decode (seqLen==1) was unaffected,
                // which is why single-token generation looked sane while the
                // 40-token prefill produced NaN.)
                auto gemmXWt = [&](float *out, const void *x, const void *wF16,
                                   uint32_t rows, uint32_t cols) -> bool {
                    // computeType CUDA_R_32F requires alpha/beta as float*.
                    // (Passing __half* made cuBLAS read the half 1.0/0.0 bits
                    // plus the adjacent half as one float -- effectively a
                    // tiny alpha that underflowed the prefill GEMM results.)
                    float alpha = 1.0f, beta = 0.0f;
                    cublasStatus_t st = cublasGemmEx(
                            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(rows), static_cast<int>(seqLen),
                            static_cast<int>(cols), &alpha, wF16, CUDA_R_16F,
                            static_cast<int>(cols), x, CUDA_R_16F,
                            static_cast<int>(cols), &beta, out, CUDA_R_32F,
                            static_cast<int>(rows), CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    return st == CUBLAS_STATUS_SUCCESS;
                };
                // Stream each weight matrix's fp16 form on-device (no persistent
                // twins).  Q/K/V share the hiddenF16 X buffer; the three GEMMs
                // each re-dequant its own matrix into wF16_.
                void *wF = nullptr;
                if (!dequantMatrixF16(w.attnQ, &wF, errMsg) ||
                    !gemmXWt(s.q, s.hiddenF16, wF, w.attnQ.rows, w.attnQ.cols) ||
                    !dequantMatrixF16(w.attnK, &wF, errMsg) ||
                    !gemmXWt(s.k, s.hiddenF16, wF, w.attnK.rows, w.attnK.cols) ||
                    !dequantMatrixF16(w.attnV, &wF, errMsg) ||
                    !gemmXWt(s.v, s.hiddenF16, wF, w.attnV.rows, w.attnV.cols)) {
                    errMsg = errMsg.empty() ? "cublasGemmEx Q/K/V failed" : errMsg;
                    return false;
                }
                if (g_verbose && !graphCaptureActive_) {
                    cudaError_t e = cudaStreamSynchronize(g_stream);
                    std::fprintf(stderr, "[gpu loop] L=%u QKV: %s\n", L,
                                 e == cudaSuccess ? "ok"
                                                  : cudaGetErrorString(e));
                }
            }

            // Biases (Qwen2): bias is ONE row (qLen/kvLen floats) broadcast
            // across every token — kAddBias, never kAddResidual (which would
            // index the bias array up to seqLen*biasLen and overread it).
            if (!biasesFolded) {
                if (w.attnQBias) {
                    kAddBias<<<seqLen, 256, 0, g_stream>>>(
                            s.q, w.attnQBias, seqLen, qLen);
                }
                if (w.attnKBias) {
                    kAddBias<<<seqLen, 256, 0, g_stream>>>(
                            s.k, w.attnKBias, seqLen, kvLen);
                }
                if (w.attnVBias) {
                    kAddBias<<<seqLen, 256, 0, g_stream>>>(
                            s.v, w.attnVBias, seqLen, kvLen);
                }
            }
            if (traceStages && !graphCaptureActive_) {
                cudaEventRecord(evQKV[L], g_stream);
            }

            // RoPE(Q) + store K/V (K rotation fused).  During graph capture
            // the kernels read pos from the device scalar graphPos_ (frozen
            // pointer; value refreshed per replay); eagerly they take pos's
            // value as an argument (capture forbids freeze-framing host
            // values that change every token).  The store + attention kernels
            // are templated on the cache element type (fp16 via TINYCODER_KV16).
            if (graphCaptureActive_) {
                kRoPEQPos<<<seqLen, 128, 0, g_stream>>>(
                        s.q, ropeCos_, ropeSin_, seqLen, nHeads, hd, graphPos_);
            } else {
                kRoPEQ<<<seqLen, 128, 0, g_stream>>>(s.q, ropeCos_, ropeSin_, seqLen,
                                                     nHeads, hd, pos);
            }

            // Flash attention (warp per token-head).  Compile-time headDim
            // so the accumulators live in registers; grid.y covers exactly
            // nHeads warps (no redundant warps).
            float invSqrt = 1.0f / std::sqrt(static_cast<float>(hd));
            dim3 blk(32, 4);
            // Split-KV decode attention: batch==1 dense decode only.  The
            // one-warp-per-head kernel leaves 56/68 SMs idle and costs a hard
            // O(cachePos) wall; the split kernel tiles the KV range so every SM
            // stays busy.  TINYCODER_ATTN_SPLIT=0 restores the eager kernel.
            const char *asEnv = std::getenv("TINYCODER_ATTN_SPLIT");
            const bool splitAttn =
                    (seqLen == 1) && (geom_.architecture == 0) &&
                    (asEnv == nullptr || asEnv[0] == '\0' || asEnv[0] != '0') &&
                    attnPartialAcc_ != nullptr && attnChunks_ > 1;
            // Producer Q8_1 fold (2026-10-10, plan section 9.26 lever 1): the
            // combine W4 epilogue writes the attnO-input Q8_1 blocks straight
            // into q8k_ (shared with the GLU/down activations), deleting the
            // 48-block kQuantizeQ8_1 launch (2.5 us/layer).  TINYCODER_FUSE_Q81_PROD=0
            // or TINYCODER_ATTN_COMBINE4=0 restore the separate launch; the
            // allocation never happens during graph capture (the eager first
            // token sizes q8k_ and the guard then simply passes).
            uint8_t *combineQ81 = nullptr;
            if (seqLen == 1 && fuseQ81Prod() && attnCombine4()) {
                const uint64_t needA =
                        static_cast<uint64_t>(w.attnO.blocksPerRow) * 8u *
                        kQ8_1_STRIDE;
                if (q8k_ != nullptr && q8kBytes_ >= needA) {
                    combineQ81 = q8k_;
                } else if (!graphCaptureActive_) {
                    if (q8k_) cudaFree(q8k_);
                    q8k_ = nullptr;
                    q8kBytes_ = 0;
                    cudaError_t e = cudaMalloc(&q8k_, needA);
                    if (e == cudaSuccess) {
                        q8kBytes_ = needA;
                        combineQ81 = q8k_;
                    }
                } else if (q8k_ != nullptr) {
                    combineQ81 = q8k_;// stale-size buffer still >= attnO need
                }
            }
#define TC_KV_RUN(KVT)                                                        \
    do {                                                                      \
        KVT *kK = static_cast<KVT *>(kvK);                                    \
        KVT *kV = static_cast<KVT *>(kvV);                                    \
        if (graphCaptureActive_) {                                            \
            kStoreKVRopePos<KVT><<<seqLen, 256, 0, g_stream>>>(               \
                    s.k, s.v, kK, kV, seqLen, nKV, hd, graphPos_, ropeCos_,   \
                    ropeSin_);                                                \
        } else {                                                              \
            kStoreKVRope<KVT><<<seqLen, 256, 0, g_stream>>>(                  \
                    s.k, s.v, kK, kV, seqLen, nKV, hd, pos, pos, ropeCos_,    \
                    ropeSin_);                                                \
        }                                                                     \
        if (splitAttn) {                                                      \
            launchDecodeAttentionSplit(                                       \
                    s.q, kvK, kvV, s.attnOut, attnPartialAcc_, attnPartialM_, \
                    attnPartialL_, seqLen, nHeads, nKV, hd, pos,              \
                    graphCaptureActive_, graphPos_, attnChunks_, attnChunk_,  \
                    invSqrt, kvHalf_, combineQ81);                            \
        } else if (graphCaptureActive_) {                                     \
            if (hd == 128)                                                    \
                kWarpAttentionPos<128, KVT><<<dim3(seqLen, (nHeads + 3) / 4), \
                                              blk, 0, g_stream>>>(            \
                        s.q, kK, kV, s.attnOut, seqLen, nHeads, nKV,          \
                        graphPos_, invSqrt);                                  \
            else if (hd == 64)                                                \
                kWarpAttentionPos<64, KVT><<<dim3(seqLen, (nHeads + 3) / 4),  \
                                             blk, 0, g_stream>>>(             \
                        s.q, kK, kV, s.attnOut, seqLen, nHeads, nKV,          \
                        graphPos_, invSqrt);                                  \
            else                                                              \
                kWarpAttentionPos<32, KVT><<<dim3(seqLen, (nHeads + 3) / 4),  \
                                             blk, 0, g_stream>>>(             \
                        s.q, kK, kV, s.attnOut, seqLen, nHeads, nKV,          \
                        graphPos_, invSqrt);                                  \
        } else {                                                              \
            if (hd == 128)                                                    \
                kWarpAttention<128, KVT><<<dim3(seqLen, (nHeads + 3) / 4),    \
                                           blk, 0, g_stream>>>(               \
                        s.q, kK, kV, s.attnOut, seqLen, nHeads, nKV, pos,     \
                        invSqrt);                                             \
            else if (hd == 64)                                                \
                kWarpAttention<64, KVT><<<dim3(seqLen, (nHeads + 3) / 4),     \
                                          blk, 0, g_stream>>>(                \
                        s.q, kK, kV, s.attnOut, seqLen, nHeads, nKV, pos,     \
                        invSqrt);                                             \
            else                                                              \
                kWarpAttention<32, KVT><<<dim3(seqLen, (nHeads + 3) / 4),     \
                                          blk, 0, g_stream>>>(                \
                        s.q, kK, kV, s.attnOut, seqLen, nHeads, nKV, pos,     \
                        invSqrt);                                             \
        }                                                                     \
    } while (0)
            if (kvHalf_) TC_KV_RUN(__half);
            else
                TC_KV_RUN(float);
#undef TC_KV_RUN
            if (g_verbose && !graphCaptureActive_) {
                cudaError_t e = cudaStreamSynchronize(g_stream);
                std::fprintf(stderr, "[gpu loop] L=%u kv+attn: %s\n", L,
                             e == cudaSuccess ? "ok"
                                              : cudaGetErrorString(e));
            }
            if (traceStages && !graphCaptureActive_) {
                cudaEventRecord(evAttn[L], g_stream);
            }

            // attnO projection + residual
            if (traceStages && !graphCaptureActive_) {
                cudaEventRecord(evMtxAttnO[L], g_stream);
            }
            // Fused attnO+residual (2026-10-09, TINYCODER_FUSE_ATTNO_RESID,
            // DEFAULT ON): fold kAddResidual into the attnO GEMV epilogue and
            // delete the separate residual launch.  Decode-only, Q2_K
            // (w.attnO.type), fp32 activation.  Falls back cleanly.
            // MEASURED (interleaved, 8 reps, n-gen 16): eager 167.33 vs 166.22
            // (+0.67 %, 5/5 rounds), graph-on 166.11 vs 164.65 (+0.89 %,
            // 5/5 rounds); parity byte-identical (eager + graph).
            // TINYCODER_FUSE_ATTNO_RESID=0 opts out.
            bool attnOResidFused = false;
            if (seqLen == 1) {
                const char *ar = std::getenv("TINYCODER_FUSE_ATTNO_RESID");
                if (ar == nullptr || std::atoi(ar) != 0) {
                    attnOResidFused = launchQGemvResid(
                            w.attnO.type, w.attnO.q, s.attnOut, s.attnProj,
                            s.hidden, w.attnO.rows, w.attnO.cols,
                            w.attnO.rowBytes, w.attnO.blocksPerRow, true,
                            q8k_, q8kBytes_, combineQ81);
                }
                if (!attnOResidFused) {
                    launchQGemv(w.attnO.type, w.attnO.q, s.attnOut, s.attnProj,
                                w.attnO.rows, w.attnO.cols, w.attnO.rowBytes,
                                w.attnO.blocksPerRow, q8k_, q8kBytes_);
                }
            } else {
                // attnOut is the fp32 attention result; make an fp16 twin for
                // the fp16 tensor-core GEMM.
                kF32ToF16<<<(seqLen * qLen + 511) / 512, 256, 0, g_stream>>>(
                        s.attnOut, reinterpret_cast<__half2 *>(s.attnOutF16),
                        (seqLen * qLen) / 2);
                // CUDA_R_32F compute type -> alpha/beta must be float*.
                float alpha = 1.0f, beta = 0.0f;
                void *wF = nullptr;
                if (!dequantMatrixF16(w.attnO, &wF, errMsg)) return false;
                // attnProj[seqLen][rows] = attnOut[seqLen][cols] @ W[rows][cols]^T
                // (see the Q/K/V gemmXWt comment: CUBLAS column-major requires
                // W in the A slot, x in B, ldc=rows).
                cublasStatus_t st = cublasGemmEx(
                        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(w.attnO.rows), static_cast<int>(seqLen),
                        static_cast<int>(w.attnO.cols), &alpha, wF, CUDA_R_16F,
                        static_cast<int>(w.attnO.cols), s.attnOutF16, CUDA_R_16F,
                        static_cast<int>(w.attnO.cols), &beta, s.attnProj, CUDA_R_32F,
                        static_cast<int>(w.attnO.rows), CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) {
                    errMsg = "cublasGemmEx attnO failed";
                    return false;
                }
            }
            if (!attnOResidFused) {
                kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                        s.hidden, s.attnProj, seqLen * H);
            }

            // Persistent fused FFN mega-kernel (idea #21/A, DEFAULT OFF):
            // when enabled for a Q2_K gate/up + Q3_K down FFN, the whole FFN
            // sub-block (RMSNorm + gate + up + silu + down + residual) runs in
            // ONE grid-barrier-sequenced launch (see kFfnPersistentLayer).  The
            // default 6-launch body is wrapped in ffnDefault() and used
            // otherwise; both compute the same value class.
            const char *ffnPersE = std::getenv("TINYCODER_FUSE_FFN_PERSIST");
            const bool ffnPersist =
                    (ffnPersE != nullptr && std::atoi(ffnPersE) != 0) &&
                    (seqLen == 1) && (w.ffnGate.type == kTypeQ2K) &&
                    (w.ffnUp.type == kTypeQ2K) &&
                    (w.ffnDown.type == kTypeQ3K) &&
                    (w.ffnGate.cols == w.ffnGate.blocksPerRow * 256u) &&
                    (w.ffnDown.cols == w.ffnDown.blocksPerRow * 256u) &&
                    (w.ffnDown.rows == H);
            auto ffnDefault = [&]() -> bool {
                // FFN RMSNorm -> s.norm (OUT-OF-PLACE, one block per row).
                // Q8_1 fold (2026-10-10, plan section 9.26 lever 1): on the
                // dense decode (seqLen==1, H==1536) the block also emits the
                // Q8_1 activation blocks the GLU mmvq consumes, deleting the
                // 48-block kQuantizeQ8_1 launch (2.5 us/layer).  Bit-identical
                // bytes (same math over the same float values).
                bool rmsQ81Fold = false;
                if (seqLen == 1 && fuseQ81Prod() && H == 1536) {
                    const uint64_t needG =
                            static_cast<uint64_t>(H / 32u) * kQ8_1_STRIDE;
                    if (q8k_ != nullptr && q8kBytes_ >= needG) {
                        kRMSNormRowQ81<<<seqLen, 1024, 1024 * sizeof(float),
                                         g_stream>>>(
                                s.hidden, s.norm, w.rmsNormFFN, q8k_, H, seqLen,
                                1e-6f);
                        rmsQ81Fold = true;
                    }
                }
                if (!rmsQ81Fold) {
                    kRMSNormRow<<<seqLen, 1024, 1024 * sizeof(float),
                                  g_stream>>>(
                            s.hidden, s.norm, w.rmsNormFFN, H, seqLen, 1e-6f);
                }

                // gate / up + SwiGLU: inputs are the FFN RMSNorm'd vector (s.norm).
                if (traceStages && !graphCaptureActive_) {
                    cudaEventRecord(evMtxGateUp[L], g_stream);
                }
                // FUSED gate+up+silu (2026-09-28): when gate/up share the same
                // quant type (guaranteed for the dense Qwen2 FFN gate/up) and
                // it is one of the fused-supported types, ONE launch computes
                // both row dots (sharing the activation read / Q8_K quantize)
                // and writes silu(gate)*up into s.gate -- the same product the
                // three separate launches produced. The raw up values (s.up)
                // are then dead. A/B gate (2026-09-30,
                // TINYCODER_FUSE_GU_Q2K=n): the ORIGINAL float-dequant fused GU
                // path (=1) measured as a tg8 REGRESSION on both 2026-09-29
                // (see README GPU Comparison) and this session (fused ~149 vs
                // separate ~166 tg8 steady-state) -- it kept the ALU-heavy
                // float dequant per matrix. Interleaved A/B (5 rounds,
                // tg128@pp512): 193.2-194.0 -> 210.9-211.8 tok/s, +9.1..9.5 %,
                // 5/5 wins, 4/4 prompt parity. DEFAULT = 2 (dp4a fused GLU); =1
                // legacy float fused path; =0 separate per-matrix launches (the
                // old default).
                const char *fg = std::getenv("TINYCODER_FUSE_GU_Q2K");
                const int fuseGUQ2KMode = (fg != nullptr) ? std::atoi(fg) : 2;
                const bool fuseGUQ2K = fuseGUQ2KMode != 0;
                const bool fuseGU = (seqLen == 1) &&
                                    (w.ffnGate.type == w.ffnUp.type) &&
                                    ((w.ffnGate.type == kTypeQ2K && fuseGUQ2K) ||
                                     w.ffnGate.type == kTypeIQ2XS ||
                                     w.ffnGate.type == kTypeIQ2S ||
                                     w.ffnGate.type == kTypeIQ3_XXS ||
                                     w.ffnGate.type == kTypeIQ3S);
                // fp16/Tensor-Core decode FFN (2026-09-30, lever-2 A/B,
                // TINYCODER_TC_FFN=1, DEFAULT OFF): reuse the persistent prefill
                // fp16 twins for decode-time cublasGemmEx GEMVs (m=1).  The twin
                // reads ~6x the compact weight bytes (2 B/elem vs ~0.33 for Q2_K),
                // so this is EXPECTED to lose to the compact-stream 4xW path --
                // included as a measured A/B, see README campaign notes.
                // Decode-only (seqLen==1); prefill keeps the batched cuBLAS
                // path (n=seqLen) so the m=1 shape is never applied to batches.
                const char *tc = std::getenv("TINYCODER_TC_FFN");
                const bool tcFFN = (seqLen == 1) &&
                                   (tc != nullptr && std::atoi(tc) != 0) &&
                                   (w.ffnGate.f16Twin != nullptr) &&
                                   (w.ffnUp.f16Twin != nullptr);
                if (tcFFN) {
                    kF32ToF16<<<(H + 511) / 512, 256, 0, g_stream>>>(
                            s.norm, reinterpret_cast<__half2 *>(s.hiddenF16),
                            H / 2);
                    float alpha = 1.0f, beta = 0.0f;
                    void *wF = nullptr;
                    if (!dequantMatrixF16(w.ffnGate, &wF, errMsg)) return false;
                    // gate[1][rows] = norm[1][cols] @ W[rows][cols]^T (m=1:
                    // W in A slot op=T, x in B slot op=N -- see the gemmXWt
                    // comment for the column-major layout).
                    cublasStatus_t st = cublasGemmEx(
                            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(w.ffnGate.rows), 1,
                            static_cast<int>(w.ffnGate.cols), &alpha, wF,
                            CUDA_R_16F, static_cast<int>(w.ffnGate.cols),
                            s.hiddenF16, CUDA_R_16F,
                            static_cast<int>(w.ffnGate.cols), &beta, s.gate,
                            CUDA_R_32F, static_cast<int>(w.ffnGate.rows),
                            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (!dequantMatrixF16(w.ffnUp, &wF, errMsg)) return false;
                    cublasStatus_t st2 = cublasGemmEx(
                            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(w.ffnUp.rows), 1,
                            static_cast<int>(w.ffnUp.cols), &alpha, wF, CUDA_R_16F,
                            static_cast<int>(w.ffnUp.cols), s.hiddenF16, CUDA_R_16F,
                            static_cast<int>(w.ffnUp.cols), &beta, s.up, CUDA_R_32F,
                            static_cast<int>(w.ffnUp.rows), CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (st != CUBLAS_STATUS_SUCCESS || st2 != CUBLAS_STATUS_SUCCESS) {
                        errMsg = "cublasGemmEx decode TC gate/up failed";
                        return false;
                    }
                } else if (fuseGU) {
                    launchQGemvFusedGU(w.ffnGate.type, w.ffnGate.q, w.ffnUp.q,
                                       s.norm, s.gate, w.ffnGate.rows,
                                       w.ffnGate.cols, w.ffnGate.rowBytes,
                                       w.ffnGate.blocksPerRow, q8k_, q8kBytes_,
                                       rmsQ81Fold ? q8k_ : nullptr);
                } else if (seqLen == 1) {
                    launchQGemv(w.ffnGate.type, w.ffnGate.q, s.norm, s.gate,
                                w.ffnGate.rows, w.ffnGate.cols, w.ffnGate.rowBytes,
                                w.ffnGate.blocksPerRow, q8k_, q8kBytes_);
                    launchQGemv(w.ffnUp.type, w.ffnUp.q, s.norm, s.up, w.ffnUp.rows,
                                w.ffnUp.cols, w.ffnUp.rowBytes, w.ffnUp.blocksPerRow,
                                q8k_, q8kBytes_);
                } else {
                    // hiddenF16 was converted before Q/K/V, but s.norm has since
                    // been refreshed by the FFN RMSNorm, so rebuild the fp16 twin
                    // from s.norm for the FFN GEMMs.
                    kF32ToF16<<<(seqLen * H + 511) / 512, 256, 0, g_stream>>>(
                            s.norm, reinterpret_cast<__half2 *>(s.hiddenF16),
                            (seqLen * H) / 2);
                    // CUDA_R_32F compute type -> alpha/beta must be float*.
                    float alpha = 1.0f, beta = 0.0f;
                    void *wF = nullptr;
                    // gate/up[seqLen][rows] = norm[seqLen][cols] @ W[rows][cols]^T
                    // (CUBLAS column-major: W in A, x in B, ldc=rows).
                    if (!dequantMatrixF16(w.ffnGate, &wF, errMsg)) return false;
                    cublasStatus_t st = cublasGemmEx(
                            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(w.ffnGate.rows), static_cast<int>(seqLen),
                            static_cast<int>(w.ffnGate.cols), &alpha, wF, CUDA_R_16F,
                            static_cast<int>(w.ffnGate.cols), s.hiddenF16,
                            CUDA_R_16F, static_cast<int>(w.ffnGate.cols), &beta,
                            s.gate, CUDA_R_32F, static_cast<int>(w.ffnGate.rows),
                            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (!dequantMatrixF16(w.ffnUp, &wF, errMsg)) return false;
                    cublasStatus_t st2 = cublasGemmEx(
                            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(w.ffnUp.rows), static_cast<int>(seqLen),
                            static_cast<int>(w.ffnUp.cols), &alpha, wF, CUDA_R_16F,
                            static_cast<int>(w.ffnUp.cols), s.hiddenF16,
                            CUDA_R_16F, static_cast<int>(w.ffnUp.cols), &beta,
                            s.up, CUDA_R_32F, static_cast<int>(w.ffnUp.rows),
                            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (st != CUBLAS_STATUS_SUCCESS || st2 != CUBLAS_STATUS_SUCCESS) {
                        errMsg = "cublasGemmEx gate/up failed";
                        return false;
                    }
                }
                // The fused gate+up+silu kernel already folded the product into
                // s.gate; only the separate-launch path needs the kSiluMul pass.
                if (!fuseGU) {
                    kSiluMul<<<(seqLen * I + 255) / 256, 256, 0, g_stream>>>(
                            s.gate, s.up, seqLen * I);
                }

                // down projection + residual
                if (traceStages && !graphCaptureActive_) {
                    cudaEventRecord(evMtxDown[L], g_stream);
                }
                // Fused ffnDown+residual (2026-10-09, TINYCODER_FUSE_DOWN_RESID=1,
                // DEFAULT OFF): fold kAddResidual into the ffnDown GEMV epilogue and
                // delete the separate residual launch.  Decode-only, Q3_K
                // (w.ffnDown.type), fp32 activation.  Falls back cleanly.
                bool downResidFused = false;
                if (seqLen == 1) {
                    // 2026-10-10: fused residual is now DEFAULT ON for the down
                    // (mmvq epilogue fold, bit-exact); TINYCODER_FUSE_DOWN_RESID=0
                    // restores launchQGemv + kAddResidual.
                    const char *dr = std::getenv("TINYCODER_FUSE_DOWN_RESID");
                    if ((dr == nullptr || std::atoi(dr) != 0) &&
                        !(tcFFN && w.ffnDown.f16Twin != nullptr)) {
                        downResidFused = launchQGemvResid(
                                w.ffnDown.type, w.ffnDown.q, s.gate, s.ffnOut,
                                s.hidden, w.ffnDown.rows, w.ffnDown.cols,
                                w.ffnDown.rowBytes, w.ffnDown.blocksPerRow,
                                true, q8k_, q8kBytes_);
                    }
                    if (downResidFused) {
                        // already launched (GEMV + residual folded)
                    } else if (tcFFN && w.ffnDown.f16Twin != nullptr) {
                        kF32ToF16<<<(I + 511) / 512, 256, 0, g_stream>>>(
                                s.gate, reinterpret_cast<__half2 *>(s.gateF16),
                                I / 2);
                        float alpha = 1.0f, beta = 0.0f;
                        void *wF = nullptr;
                        if (!dequantMatrixF16(w.ffnDown, &wF, errMsg)) return false;
                        cublasStatus_t st = cublasGemmEx(
                                g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                static_cast<int>(w.ffnDown.rows), 1,
                                static_cast<int>(w.ffnDown.cols), &alpha, wF,
                                CUDA_R_16F, static_cast<int>(w.ffnDown.cols),
                                s.gateF16, CUDA_R_16F,
                                static_cast<int>(w.ffnDown.cols), &beta, s.ffnOut,
                                CUDA_R_32F, static_cast<int>(w.ffnDown.rows),
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                        if (st != CUBLAS_STATUS_SUCCESS) {
                            errMsg = "cublasGemmEx decode TC down failed";
                            return false;
                        }
                    } else {
                        launchQGemv(w.ffnDown.type, w.ffnDown.q, s.gate, s.ffnOut,
                                    w.ffnDown.rows, w.ffnDown.cols,
                                    w.ffnDown.rowBytes, w.ffnDown.blocksPerRow,
                                    q8k_, q8kBytes_);
                    }
                } else {
                    // gate holds silu(gate)*up in fp32; fp16 twin for the GEMM.
                    kF32ToF16<<<(seqLen * I + 511) / 512, 256, 0, g_stream>>>(
                            s.gate, reinterpret_cast<__half2 *>(s.gateF16),
                            (seqLen * I) / 2);
                    // CUDA_R_32F compute type -> alpha/beta must be float*.
                    float alpha = 1.0f, beta = 0.0f;
                    void *wF = nullptr;
                    if (!dequantMatrixF16(w.ffnDown, &wF, errMsg)) return false;
                    // ffnOut[seqLen][rows] = gate[seqLen][cols] @ W[rows][cols]^T
                    // (CUBLAS column-major: W in A, x in B, ldc=rows).
                    cublasStatus_t st = cublasGemmEx(
                            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(w.ffnDown.rows), static_cast<int>(seqLen),
                            static_cast<int>(w.ffnDown.cols), &alpha, wF, CUDA_R_16F,
                            static_cast<int>(w.ffnDown.cols), s.gateF16,
                            CUDA_R_16F, static_cast<int>(w.ffnDown.cols), &beta,
                            s.ffnOut, CUDA_R_32F, static_cast<int>(w.ffnDown.rows),
                            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (st != CUBLAS_STATUS_SUCCESS) {
                        errMsg = "cublasGemmEx ffnDown failed";
                        return false;
                    }
                }
                if (!downResidFused) {
                    kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                            s.hidden, s.ffnOut, seqLen * H);
                }
                return true;
            };// end ffnDefault
            if (ffnPersist) {
                // Persistent fused FFN mega-kernel (1 launch, device barriers).
                // productOut reuses s.gate (the default path's SwiGLU product
                // buffer), downOut reuses s.ffnOut (residual operand).  Falls
                // back to the default body if the device refuses the launch.
                FfnPersistentArgs pa;
                pa.hiddenRes = s.hidden;
                pa.hiddenOut = s.hidden;
                pa.ffnNormW = w.rmsNormFFN;
                pa.wg = static_cast<const uint8_t *>(w.ffnGate.q);
                pa.wu = static_cast<const uint8_t *>(w.ffnUp.q);
                pa.wd = static_cast<const uint8_t *>(w.ffnDown.q);
                pa.productOut = s.gate;
                pa.downOut = s.ffnOut;
                pa.H = H;
                pa.I = I;
                pa.ffnRows = w.ffnGate.rows;
                pa.downRows = w.ffnDown.rows;
                pa.gateRowBytes = w.ffnGate.rowBytes;
                pa.downRowBytes = w.ffnDown.rowBytes;
                pa.gateBlocksPerRow = w.ffnGate.blocksPerRow;
                pa.downBlocksPerRow = w.ffnDown.blocksPerRow;
                pa.eps = 1e-6f;
                // Q8_1 activation path (2026-10-08): opt-in with
                // TINYCODER_FFN_PERSIST_Q81=1.  Quantizes the block-local FFN
                // norm in-kernel (phase 1b) and runs gate/up as a Q2_K x Q8_1
                // dp4a dot; the launcher silently falls back to the fp32 path
                // when H is not a multiple of 32 or shared does not fit.
                const char *q81E = std::getenv("TINYCODER_FFN_PERSIST_Q81");
                pa.q81 = (q81E != nullptr && std::atoi(q81E) != 0);
                if (!launchFfnPersistentLayer(pa, g_stream)) {
                    if (!ffnDefault()) return false;
                }
            } else {
                if (!ffnDefault()) return false;
            }

            if (g_verbose && !graphCaptureActive_) {
                cudaError_t e = cudaStreamSynchronize(g_stream);
                std::fprintf(stderr, "[gpu loop] L=%u ffn: %s\n", L,
                             e == cudaSuccess ? "ok"
                                              : cudaGetErrorString(e));
            }
            if (traceStages && !graphCaptureActive_) {
                cudaEventRecord(evFFN[L], g_stream);
            }
        }
        // Host layer-loop sync: during capture the stream is being recorded
        // (a host sync would abort capture), so only run the eager check.
        cudaError_t llErr = cudaSuccess;
        if (!graphCaptureActive_) {
            llErr = cudaStreamSynchronize(g_stream);
        }
        if (llErr != cudaSuccess) {
            if (evL0) cudaEventDestroy(evL0);
            if (evL1) cudaEventDestroy(evL1);
            errMsg = std::string("GPU layer loop sync: ") + cudaGetErrorString(llErr);
            return false;
        }
        if (g_verbose && !graphCaptureActive_) {
            cudaEventRecord(evL1, g_stream);
            cudaEventSynchronize(evL1);
            float gpuMs = 0.0f;
            cudaEventElapsedTime(&gpuMs, evL0, evL1);
            std::fprintf(stderr, "[gpu fwd   gpu-ms] layer loop GPU elapsed = %.2f ms (%u layers)\n",
                         gpuMs, nGpu);
            // NOTE: evL0 is NOT destroyed here: the traceStages block below
            // re-uses it as the loop-start anchor ElapsedTime pair.  Destroying
            // first made cudaEventElapsedTime(evL0, evQKV[0]) return
            // cudaErrorInvalidResourceHandle, which poisons the thread's sticky
            // last-error and made the post-LM-head cudaGetLastError() check
            // report "GPU lmhead launch: invalid resource handle" under
            // TINYCODER_GPU_VERBOSE=1 (decode-only; prefill has no traceStages).
            // Both events are destroyed after the traceStages block below.
        }
        if (traceStages && !graphCaptureActive_) {
            double accQKV = 0.0, accAttn = 0.0, accFFN = 0.0;
            for (uint32_t i = 0; i < nGpu; ++i) {
                float a = 0.0f, b = 0.0f, c = 0.0f;
                if (i == 0) {
                    cudaEventElapsedTime(&a, evL0, evQKV[0]);
                } else {
                    cudaEventElapsedTime(&a, evFFN[i - 1], evQKV[i]);
                }
                cudaEventElapsedTime(&b, evQKV[i], evAttn[i]);
                cudaEventElapsedTime(&c, evAttn[i], evFFN[i]);
                accQKV += a;
                accAttn += b;
                accFFN += c;
            }
            std::fprintf(stderr,
                         "[gpu fwd   gpu-ms] decode stage sums: QKV/rms=%.2f ms, kv/attn/rope=%.2f ms, attnO/ffn=%.2f ms\n",
                         accQKV, accAttn, accFFN);
            // per-matrix sums inside the FFN-heavy stage (event stream order per
            // layer: evMtxAttnO, evMtxGateUp, evMtxDown, evFFN).  attnO span =
            // attnO GEMV + attnO residual + FFN RMSNorm; gate+up span = gate
            // GEMV + up GEMV + kSiluMul; down span = down GEMV + residual.
            double accAO = 0.0, accGU = 0.0, accDn = 0.0;
            for (uint32_t i = 0; i < nGpu; ++i) {
                float aa = 0.0f, gu = 0.0f, dn = 0.0f;
                cudaEventElapsedTime(&aa, evMtxAttnO[i], evMtxGateUp[i]);
                cudaEventElapsedTime(&gu, evMtxGateUp[i], evMtxDown[i]);
                cudaEventElapsedTime(&dn, evMtxDown[i], evFFN[i]);
                accAO += aa;
                accGU += gu;
                accDn += dn;
            }
            std::fprintf(stderr,
                         "[gpu fwd   gpu-ms] decode matrix sums: attnO=%.2f ms, gate+up=%.2f ms, down=%.2f ms\n",
                         accAO, accGU, accDn);
            for (uint32_t i = 0; i < nGpu; ++i) {
                cudaEventDestroy(evQKV[i]);
                cudaEventDestroy(evAttn[i]);
                cudaEventDestroy(evFFN[i]);
                cudaEventDestroy(evMtxAttnO[i]);
                cudaEventDestroy(evMtxGateUp[i]);
                cudaEventDestroy(evMtxDown[i]);
            }
        }
        // Anchor the LM-head stage right after the layer loop (final RMSNorm +
        // LM-head GEMV + D2H).  Recorded now, read after the LM head below.
        if (traceStages && !graphCaptureActive_ && evLmHeadS != nullptr) {
            cudaEventRecord(evLmHeadS, g_stream);
        }
        if (hostEventTrace) {
            cudaEventDestroy(evL0);
            cudaEventDestroy(evL1);
            evL0 = evL1 = nullptr;
        }

        kvPos_ += seqLen;

        if (!fullOffload) {
            // Partial offload: forward() stops after the offloaded layer prefix
            // and leaves the hidden state on the device.  The caller picks it up
            // via copyHiddenOut() and continues the remaining layers on the CPU
            // against the host KV cache.  (kvPos_ has already advanced.)
            cudaStreamSynchronize(g_stream);
            return true;
        }

        // ---- Full offload: final RMSNorm + LM head ----
        TIMING_TAG("layer loop done");
        // Final RMSNorm -> place the result INTO s.hidden (the residual stream
        // is no longer needed once the last layer's residual add has run).
        // One block per row.
        if (finalNorm_) {
            kRMSNormRow<<<seqLen, 1024, 1024 * sizeof(float), g_stream>>>(
                    s.hidden, s.hidden, finalNorm_, H, seqLen, 1e-6f);
        }

        // LM head: quantized GEMV. Tied head -> token-embedding matrix
        // (embedQ_). Separate output.weight -> lmHeadQ_ (when uploaded).
        TIMING_TAG("lmhead start");
        const void *lmQ = lmHeadQ_ != nullptr ? lmHeadQ_ : embedQ_;
        uint32_t lmType = lmHeadQ_ != nullptr ? lmHeadType_ : embedType_;
        uint32_t lmRowBytes = lmHeadQ_ != nullptr ? lmHeadRowBytes_ : embedRowBytes_;
        uint32_t blocksPerRow = hostBlocksPerRow(lmType, H);
        // The graph's D2H target is frozen at capture: replayDecodeGraph does
        // its own D2H from scratch_.logits into the caller's buffer, so the
        // captured LM head must write its logits to the DEVICE scratch_.logits
        // and skip the host D2H during capture (replay copies them out).
        // logitsCapOneRow_: ONE-row logits scratch -> last token only.
        if (computeAllLogits && !logitsCapOneRow_) {
            for (uint32_t si = 0; si < seqLen; ++si) {
                launchQGemv(lmType, lmQ, s.hidden + si * H,
                            s.logits + si * geom_.vocabSize, geom_.vocabSize, H,
                            lmRowBytes, blocksPerRow, q8k_, q8kBytes_);
            }
        } else {
            uint32_t si = seqLen - 1;
            launchQGemv(lmType, lmQ, s.hidden + si * H, s.logits,
                        geom_.vocabSize, H, lmRowBytes, blocksPerRow, q8k_,
                        q8kBytes_);
        }
        // Stage-trace midpoint: GEMV vs D2H split of the lmhead stage.
        if (traceStages && !graphCaptureActive_ && evLmMid != nullptr) {
            cudaEventRecord(evLmMid, g_stream);
        }
        if (!graphCaptureActive_) {
            // Only a ONE-row s.logits scratch (logitsCapOneRow_) or a decode
            // (seqLen==1) writes a single row, so D2H just that row.  The CALLER
            // buffer shape follows the computeAllLogits request: 1 row when
            // !computeAllLogits (always dst row 0) and seqLen rows otherwise (the
            // one-row fallback then lands at the last row, leaving the rest of
            // the caller's array untouched — matching the CPU path, which also
            // populates only the last row when computeAllLogits==false).
            const bool writeAll = computeAllLogits && !logitsCapOneRow_;
            const uint32_t rows = writeAll ? seqLen : 1u;
            const uint32_t dstRow =
                    (computeAllLogits && logitsCapOneRow_ && seqLen > 0u)
                            ? seqLen - 1u
                            : 0u;
            const uint64_t nL =
                    static_cast<uint64_t>(rows) * geom_.vocabSize;
            float *dst = logitsOut +
                         static_cast<uint64_t>(dstRow) * geom_.vocabSize;
            const char *pdE = std::getenv("TINYCODER_PINNED_D2H");
            const bool usePinned = (pdE == nullptr || pdE[0] != '0');
            if (usePinned && (lmPinned_ == nullptr ||
                              lmPinnedFloats_ < nL)) {
                if (lmPinned_) cudaFreeHost(lmPinned_);
                // Decode needs 1 x vocab; prefill's all-logits path grows to
                // seqLen x vocab on demand.
                cudaError_t pe = cudaHostAlloc(
                        &lmPinned_,
                        static_cast<uint64_t>(geom_.vocabSize) * sizeof(float) *
                                rows,
                        cudaHostAllocDefault);
                if (pe == cudaSuccess) {
                    lmPinnedFloats_ =
                            static_cast<uint64_t>(geom_.vocabSize) * rows;
                } else {
                    lmPinned_ = nullptr;
                    lmPinnedFloats_ = 0;
                }
            }
            if (usePinned && lmPinned_ != nullptr) {
                // Pinned staging: the caller's logits buffer is pageable, and a
                // direct async D2H into it runs the driver's slow staged path
                // (~4 GB/s, 0.15 ms measured).  Device -> pinned + host memcpy
                // is ~2x faster for the 0.6 MB decode copy.
                cudaMemcpyAsync(lmPinned_, s.logits, nL * sizeof(float),
                                cudaMemcpyDeviceToHost, g_stream);
                cudaStreamSynchronize(g_stream);
                std::memcpy(dst, lmPinned_, nL * sizeof(float));
            } else {
                cudaMemcpyAsync(dst, s.logits, nL * sizeof(float),
                                cudaMemcpyDeviceToHost, g_stream);
            }
        }
        cudaError_t lerr = cudaGetLastError();
        if (lerr != cudaSuccess) {
            errMsg = std::string("GPU lmhead launch: ") + cudaGetErrorString(lerr);
            return false;
        }
        if (traceStages && !graphCaptureActive_ && evLmHeadE != nullptr) {
            cudaEventRecord(evLmHeadE, g_stream);
            cudaStreamSynchronize(g_stream);
            float lmMs = 0.0f;
            if (evLmHeadS != nullptr) {
                cudaEventElapsedTime(&lmMs, evLmHeadS, evLmHeadE);
            }
            float gemvMs = 0.0f, d2hMs = 0.0f;
            if (evLmMid != nullptr && evLmHeadS != nullptr) {
                cudaEventElapsedTime(&gemvMs, evLmHeadS, evLmMid);
                cudaEventElapsedTime(&d2hMs, evLmMid, evLmHeadE);
            }
            std::fprintf(stderr,
                         "[gpu fwd   gpu-ms] decode lmhead stage: "
                         "finalnorm+lmhead=%.3f ms, d2h=%.3f ms (sum %.2f)\n",
                         gemvMs, d2hMs, lmMs);
            cudaEventDestroy(evLmHeadS);
            cudaEventDestroy(evLmHeadE);
            cudaEventDestroy(evLmMid);
            evLmHeadS = evLmHeadE = evLmMid = nullptr;
        }

        // EXPERT-CACHE WAR BARRIER (2026-09-17): the MoE expert-cache miss
        // fills run on g_stream2 (the copy engine) while the GEMVs that read
        // those slots run on g_stream.  cudaStreamSynchronize(g_stream) above
        // only waits for the COMPUTE stream — a g_stream2 H2D enqueued at the
        // end of the MoE block may STILL be in flight when forward() returns,
        // and a LATER forward's cache HIT reads that slot WITHOUT waiting
        // (hits wait on nothing).  That lets forward N's fills race forward
        // N+1's hit-reads -> run-to-run varying routed-expert outputs
        // (measured: ModelTest.KVCacheClear maxDiff 1.92-2.04 across runs while
        // default/Option-B mode is bit-identical at maxDiff 0; same signature
        // inflates ModelTest.CompareBatchVsSequentialPrefill to 16.28).
        // Draining g_stream2 here costs ~0 when the fills already completed
        // (the common steady state), and makes cache-mode deterministic:
        // by the time forward() returns, every slot byte is final.
        if (expertCacheBuilt_ && !graphCaptureActive_) {
            cudaError_t eS2 = cudaStreamSynchronize(g_stream2);
            if (eS2 != cudaSuccess) {
                errMsg = std::string("GPU forward g_stream2 drain: ") +
                         cudaGetErrorString(eS2);
                return false;
            }
        }
        if (graphCaptureActive_) {
            // End the capture and instantiate the graph.  The captured
            // stream contains: all decode kernels (device-pos variants), the
            // LM-head kernels, and NO host syncs / NO D2H (replay copies the
            // logits out).  captureDecodeGraph() wrote the first token's pos
            // into graphPos_ BEFORE BeginCapture, so the recorded kernels
            // read the right position.
            cudaGraph_t rawGraph = nullptr;
            cudaError_t capE =
                    cudaStreamEndCapture(g_stream, &rawGraph);
            graph_ = static_cast<void *>(rawGraph);
            graphCaptureActive_ = false;
            if (capE != cudaSuccess) {
                std::fprintf(stderr, "[gpu] CUDA-graph EndCapture failed: %s\n",
                             cudaGetErrorString(capE));
            }
            if (capE == cudaSuccess && rawGraph == nullptr) {
                std::fprintf(stderr, "[gpu] CUDA-graph EndCapture: null graph\n");
            }
            bool instOk = (capE == cudaSuccess && rawGraph != nullptr);
            if (instOk) instOk = instantiateGraph();
            if (instOk) {
                // Kernels are RECORDED, not executed during capture: launch
                // the freshly-instantiated graph once to compute the first
                // token's logits (and fill the KV cache for token 0).
                cudaError_t le = cudaGraphLaunch(
                        static_cast<cudaGraphExec_t>(graphExec_), g_stream);
                cudaStreamSynchronize(g_stream);
                if (le == cudaSuccess) {
                    cudaMemcpy(logitsOut, scratch_.logits,
                               geom_.vocabSize * sizeof(float),
                               cudaMemcpyDeviceToHost);
                } else {
                    std::fprintf(stderr, "[gpu] CUDA-graph first launch failed: %s\n",
                                 cudaGetErrorString(le));
                    graphCaptureFailed_ = true;
                }
            } else {
                std::fprintf(stderr, "[gpu] CUDA-graph instantiate failed\n");
                graphCaptureFailed_ = true;
            }
            TIMING_TAG("forward complete (graph captured)");
            return true;
        }
        cudaError_t fin = cudaStreamSynchronize(g_stream);
        if (fin != cudaSuccess) {
            errMsg = std::string("GPU forward final sync: ") +
                     cudaGetErrorString(fin);
            return false;
        }
        TIMING_TAG("forward complete");
        return true;
    }

    // CUDA-graph decode (TINYCODER_GPU_GRAPH=1). The dense decode path
    // dispatches ~15 kernels/layer x 28 layers + LM head every token; each
    // launch carries driver-side setup + inter-kernel dependency bubbles. The
    // capture records the SAME launch sequence the eager path issues (single
    // source of truth: forward() runs with graphCaptureActive_=true), then the
    // steady state replays it.
    //
    // Capture rules honored here:
    //  * cudaStreamBeginCapture(stream) must be called on an idle stream.
    //  * No host-side cudaStreamSynchronize / blocking cudaMemcpy may run
    //    during capture.  The decode forward's only host syncs are the
    //    g_verbose debug prints (skipped during capture) and the final sync
    //    (skipped during capture; the graph is instantiated from the captured
    //    stream, then the stream is synced once after).
    //  * All kernels must read position from DEVICE memory (graphPos_) --
    //    handled by the kRoPEQPos/kStoreKVRopePos/kWarpAttentionPos variants
    //    selected when graphCaptureActive_.
    //  * No cudaMalloc during capture: ensureGraphScratch() pre-allocates
    //    the decode scratch before the first capture.
    bool GPUModel::ensureGraphScratch() {
        if (graphPos_ == nullptr) {
            cudaError_t e = cudaMalloc(&graphPos_, sizeof(uint32_t));
            if (e != cudaSuccess) return false;
        }
        // Force the q8k_ scratch to its decode size so NO cudaMalloc runs
        // during capture (launchQGemv grows q8k_ lazily).  Issue one eager
        // Q8_K-quantized GEMV for the LM head (the widest decode matrix).
        // The Q6_K launch path is the only decode user of q8k_ on this dense
        // model (kQuantizeQ8K + kQGemvQ6KxQ8K_4xW); on IQ models the Q8_K
        // int path also sizes q8k_ to blocksPerRow.  Force it here.
        const uint32_t lmCols = geom_.hiddenSize;
        const uint32_t lmBlocks = hostBlocksPerRow(lmHeadQ_ != nullptr
                                                           ? lmHeadType_
                                                           : embedType_,
                                                   lmCols);
        uint8_t *scratch = q8k_;
        uint64_t scratchBytes = q8kBytes_;
        const void *lmQ = lmHeadQ_ != nullptr ? lmHeadQ_ : embedQ_;
        // Warm the q8k_ buffer to its widest layout (Q8_K int path would
        // allocate blocksPerRow*kQ8K_STRIDE); the Q6_K launch reuses it.
        const uint64_t need = static_cast<uint64_t>(lmBlocks) * kQ8K_STRIDE;
        if (scratch == nullptr || scratchBytes < need) {
            if (scratch) cudaFree(scratch);
            scratch = nullptr;
            if (cudaMalloc(&scratch, need) != cudaSuccess) return false;
            scratchBytes = need;
        }
        q8k_ = scratch;
        q8kBytes_ = scratchBytes;
        // Touch the LM-head logits output buffer: the graph captures the
        // D2H cudaMemcpyAsync to `logitsOut`'s FINAL ADDRESS; the caller's
        // logitsOut pointer is stable across calls but the D2H target must
        // have been allocated.  We capture the D2H INTO scratch_.logits and
        // copy to the caller's buffer AFTER the graph returns (the caller's
        // pointer is not graph-capturable without a memcpy shim).
        return lmQ != nullptr;
    }

    bool GPUModel::instantiateGraph() {
        if (graphExec_ != nullptr) return true;
        if (graph_ == nullptr) return false;
        cudaGraphExec_t rawExec = nullptr;
        cudaError_t e = cudaGraphInstantiate(&rawExec,
                                             static_cast<cudaGraph_t>(graph_),
                                             nullptr, nullptr, 0);
        graphExec_ = static_cast<void *>(rawExec);
        cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
        graph_ = nullptr;
        if (e != cudaSuccess) {
            graphExec_ = nullptr;
            return false;
        }
        graphBuilt_ = true;
        return true;
    }

    void GPUModel::destroyGraph() {
        // Teardown shared with ensureScratch() (scratch reallocation changes
        // the addresses the graph froze) and destroy().  graphPos_ is
        // deliberately kept: it is a tiny scratch-independent device scalar
        // reused across graph rebuilds.
        if (graphExec_) {
            cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graphExec_));
            graphExec_ = nullptr;
        }
        if (graph_) {
            cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
            graph_ = nullptr;
        }
        graphBuilt_ = false;
        graphCaptureActive_ = false;
        graphCaptureFailed_ = false;
    }

    // Capture the decode forward into a graph by running the eager body once
    // under cudaStreamBeginCapture.  The body (with graphCaptureActive_=true)
    // uses the device-pos kernel variants and skips host syncs; the graph is
    // then instantiated.  pos is the first token's position, written to
    // graphPos_ (blocking H2D) BEFORE capture begins so the recorded kernels
    // observe the right value on the capture run itself.  Returns true on
    // success; false (with graphCaptureFailed_) on any error so subsequent
    // forwards stay eager.
    bool GPUModel::captureDecodeGraph(uint32_t pos) {
        if (graphBuilt_ || graphCaptureFailed_) return graphBuilt_;
        if (!ensureGraphScratch()) {
            graphCaptureFailed_ = true;
            return false;
        }
        graphPosHost_ = pos;
        cudaError_t e = cudaMemcpy(graphPos_, &graphPosHost_, sizeof(uint32_t),
                                   cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            graphCaptureFailed_ = true;
            return false;
        }
        if (cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeRelaxed) !=
            cudaSuccess) {
            graphCaptureFailed_ = true;
            return false;
        }
        // Stream capture is now active: the body below (forward()'s decode
        // path) must use the device-pos kernel variants and skip host syncs.
        graphCaptureActive_ = true;
        // The eager body runs with the capture flag; it enqueues the kernels
        // (device-pos variants) and the LM head into scratch_.logits.  It must
        // NOT run any host cudaStreamSynchronize / blocking memcpy (capture
        // forbids them) -- forward() sees graphCaptureActive_ and skips the
        // sync points.  After forward() returns, the tail ends capture and
        // calls instantiateGraph().
        return true;
    }

    // Replay the captured decode graph.  The caller has already H2D'd the
    // tokens into s.tokens; this refreshes the device pos scalar, launches
    // the graph, syncs, and copies the logits from scratch_.logits to the
    // caller's buffer (the graph captured the LM head writing into
    // scratch_.logits -- the D2H target is frozen, and the caller's pointer
    // is not graph-capturable because it can change per call site).
    bool GPUModel::replayDecodeGraph(uint32_t pos, float *logitsOut,
                                     std::string &errMsg) {
        if (g_verbose) {
            std::fprintf(stderr, "[gpu] graph replay pos=%u\n", pos);
        }
        if (graphExec_ == nullptr) return false;
        graphPosHost_ = pos;
        cudaError_t e = cudaMemcpyAsync(graphPos_, &graphPosHost_,
                                        sizeof(uint32_t),
                                        cudaMemcpyHostToDevice, g_stream);
        if (e != cudaSuccess) {
            errMsg = std::string("GPU graph pos H2D: ") + cudaGetErrorString(e);
            return false;
        }
        e = cudaGraphLaunch(static_cast<cudaGraphExec_t>(graphExec_),
                            g_stream);
        if (e != cudaSuccess) {
            errMsg = std::string("GPU graph launch: ") + cudaGetErrorString(e);
            return false;
        }
        e = cudaStreamSynchronize(g_stream);
        if (e != cudaSuccess) {
            errMsg = std::string("GPU graph sync: ") + cudaGetErrorString(e);
            // A failed replay leaves the CUDA context with a STICKY async
            // error that poisons every subsequent launch/sync on this context:
            // without clearing it, the eager fallback path below fails
            // immediately with the same error and the whole token cascades to
            // the CPU path (whose KV cache lacks the GPU-owned prefix ->
            // garbage logits).  Consume the sticky error and drain the device
            // so the eager decode path can recompute THIS token correctly
            // against the (still valid) GPU KV cache.
            cudaGetLastError();
            cudaDeviceSynchronize();
            return false;
        }
        // Pinned staging (mirrors the eager path): a D2H into the caller's
        // PAGEABLE logits buffer runs the driver's slow staged path (~4 GB/s,
        // ~0.15 ms for the 0.6 MB decode copy); device->pinned + host memcpy
        // is ~2x faster.  TINYCODER_PINNED_D2H=0 opts out.
        const char *pdE = std::getenv("TINYCODER_PINNED_D2H");
        const bool usePinned = (pdE == nullptr || pdE[0] != '0');
        if (usePinned) {
            if (lmPinned_ == nullptr) {
                cudaError_t pe = cudaHostAlloc(
                        &lmPinned_,
                        static_cast<uint64_t>(geom_.vocabSize) * sizeof(float),
                        cudaHostAllocDefault);
                if (pe == cudaSuccess) {
                    lmPinnedFloats_ = geom_.vocabSize;
                } else {
                    lmPinned_ = nullptr;
                }
            }
            if (lmPinned_ != nullptr) {
                cudaMemcpy(lmPinned_, scratch_.logits,
                           geom_.vocabSize * sizeof(float),
                           cudaMemcpyDeviceToHost);
                std::memcpy(logitsOut, lmPinned_,
                            geom_.vocabSize * sizeof(float));
                return true;
            }
        }
        cudaMemcpy(logitsOut, scratch_.logits,
                   geom_.vocabSize * sizeof(float), cudaMemcpyDeviceToHost);
        return true;
    }

    bool GPUModel::copyHiddenOut(float *hiddenOut, uint32_t seqLen,
                                 std::string &errMsg) {
        if (!allocated_ || !scratchAlloc_) {
            errMsg = "GPU model not uploaded / no hidden state";
            return false;
        }
        if (seqLen > scratch_.seqCap) {
            errMsg = "copyHiddenOut: seqLen exceeds scratch capacity";
            return false;
        }
        cudaError_t e = cudaMemcpy(hiddenOut, scratch_.hidden,
                                   static_cast<size_t>(seqLen) * geom_.hiddenSize *
                                           sizeof(float),
                                   cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            errMsg = std::string("copyHiddenOut: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    void GPUModel::clearKVCache() {
        if (!allocated_) return;
        uint32_t nGpu = geom_.numGpuLayers;
        // Zero only the slots actually allocated (qwen35/MoE skip the recurrent
        // layers; kvSlots_ == nGpu for the dense arch and <= nGpu otherwise).
        const uint64_t kvAllocBytes =
                static_cast<uint64_t>(kvSlots_) * kvLayerStride_;
        cudaMemsetAsync(kvK_, 0, static_cast<size_t>(kvAllocBytes), g_stream);
        cudaMemsetAsync(kvV_, 0, static_cast<size_t>(kvAllocBytes), g_stream);
        // Qwen35 / Qwen35MoE: also reset the persistent gated-delta-net + conv1d
        // state so a fresh session starts from an empty recurrent state (matches
        // the CPU kvCache_.q35ConvState / q35GdnState zeroing on clearKVCache).
        // The MoE arch shares the same recurrent block, so both must reset.
        if (geom_.architecture == 1 || geom_.architecture == 2) {
            const uint32_t q35KeyDim = geom_.ssmStateSize * geom_.ssmGroupCount;
            const uint32_t q35QkvDim = 2 * q35KeyDim + geom_.ssmInnerSize;
            const uint32_t q35HeadV =
                    geom_.ssmInnerSize / std::max(geom_.ssmTimeStepRank, 1u);
            uint64_t q35ConvBytes =
                    static_cast<uint64_t>(nGpu) * (geom_.ssmConvKernel - 1) *
                    q35QkvDim * sizeof(float);
            uint64_t q35GdnBytes =
                    static_cast<uint64_t>(nGpu) * geom_.ssmTimeStepRank * q35HeadV *
                    q35HeadV * sizeof(float);
            if (q35ConvBytes == 0) q35ConvBytes = 1;
            if (q35GdnBytes == 0) q35GdnBytes = 1;
            cudaMemsetAsync(q35ConvState_, 0,
                            static_cast<size_t>(q35ConvBytes), g_stream);
            cudaMemsetAsync(q35GdnState_, 0,
                            static_cast<size_t>(q35GdnBytes), g_stream);
        }
        kvPos_ = 0;
        cudaStreamSynchronize(g_stream);
    }

    void GPUModel::destroy() {
        if (!allocated_) return;
        // CUDA-graph decode resources (TINYCODER_GPU_GRAPH).  destroyGraph()
        // tears down graphExec_/graph_ + capture state; graphPos_ is a
        // scratch-independent device scalar freed here (destroyGraph keeps it).
        destroyGraph();
        if (graphPos_) {
            cudaFree(graphPos_);
            graphPos_ = nullptr;
        }
        if (layers_) {
            uint32_t nGpu = geom_.numGpuLayers;
            for (uint32_t L = 0; L < nGpu; ++L) {
                DeviceLayer &l = layers_[L];
                auto freeMat = [](DeviceMatrix &m) {
                    cudaFree(m.q);
                    m.q = nullptr;
                    cudaFree(m.f16Twin);
                    m.f16Twin = nullptr;
                    m.f16TwinBytes = 0;
                };
                auto freeF32 = [](float *&p) {
                    cudaFree(p);
                    p = nullptr;
                };
                freeMat(l.attnQ);
                freeMat(l.attnK);
                freeMat(l.attnV);
                freeMat(l.attnO);
                freeMat(l.ffnGate);
                freeMat(l.ffnUp);
                freeMat(l.ffnDown);
                freeMat(l.attnQKV);
                freeMat(l.attnGate);
                freeMat(l.ssmAlphaQ);
                freeMat(l.ssmBetaQ);
                freeMat(l.ssmOut);
                freeMat(l.ssmConv1d);
                // Qwen35MoE (qwen35moe) MoE FFN matrices.
                freeMat(l.ffnGateInpMoe);
                freeMat(l.ffnGateExps);
                freeMat(l.ffnUpExps);
                freeMat(l.ffnDownExpsMoe);
                freeMat(l.ffnGateShexp);
                freeMat(l.ffnUpShexp);
                freeMat(l.ffnDownShexp);
                freeMat(l.ffnGateInpShexp);
                freeF32(l.attnQBias);
                freeF32(l.attnKBias);
                freeF32(l.attnVBias);
                freeF32(l.rmsNormAttn);
                freeF32(l.rmsNormFFN);
                freeF32(l.ssmABroadcast);
                freeF32(l.ssmDtBiasFull);
                freeF32(l.ssmNorm);
                freeF32(l.attnQNorm);
                freeF32(l.attnKNorm);
                freeF32(l.postAttnNorm);
            }
            delete[] layers_;
            layers_ = nullptr;
        }
        cudaFree(kvK_);
        cudaFree(kvV_);
        cudaFree(finalNorm_);
        cudaFree(embedQ_);
        cudaFree(lmHeadQ_);
        cudaFree(ropeCos_);
        cudaFree(ropeSin_);
        cudaFree(mropeCos_);
        cudaFree(mropeSin_);
        cudaFree(q35ConvState_);
        cudaFree(q35GdnState_);
        kvK_ = kvV_ = nullptr;
        finalNorm_ = embedQ_ = nullptr;
        lmHeadQ_ = nullptr;
        ropeCos_ = ropeSin_ = nullptr;
        mropeCos_ = mropeSin_ = nullptr;
        q35ConvState_ = q35GdnState_ = nullptr;
        if (wF16_) {
            cudaFree(wF16_);
            wF16_ = nullptr;
        }
        wF16Bytes_ = 0;
        wF16SrcQ_ = nullptr;
        if (q8k_) {
            cudaFree(q8k_);
            q8k_ = nullptr;
        }
        q8kBytes_ = 0;
        if (lmPinned_) {
            cudaFreeHost(lmPinned_);
            lmPinned_ = nullptr;
        }
        lmPinnedFloats_ = 0;
        if (q8kN_) {
            cudaFree(q8kN_);
            q8kN_ = nullptr;
        }
        q8kNBytes_ = 0;
        if (moeQ8kDownP_) {
            cudaFree(moeQ8kDownP_);
            moeQ8kDownP_ = nullptr;
        }
        moeQ8kDownPBytes_ = 0;
        if (moeRankWq_) {
            cudaFree(moeRankWq_);
            moeRankWq_ = nullptr;
        }
        moeRankWqBytes_ = 0;
        // CPU-expert hybrid handoff events (lazily created, freed at destroy).
        for (uint32_t i = 0; i < kMoeEventRing; ++i) {
            if (eNormReady[i]) {
                cudaEventDestroy(eNormReady[i]);
                eNormReady[i] = nullptr;
            }
            if (eCpuDone[i]) {
                cudaEventDestroy(eCpuDone[i]);
                eCpuDone[i] = nullptr;
            }
        }
        moeBufFlip_ = 0;
        // Unregister the pinned norm-mirror ranges BEFORE the vectors free
        // their storage (cudaHostUnregister on a freed pointer is UB).
        for (uint32_t pi = 0; pi < 2; ++pi) {
            if (moeNormPinned_[pi] != nullptr) {
                cudaHostUnregister(moeNormPinned_[pi]);
                moeNormPinned_[pi] = nullptr;
            }
        }
        moeNormHost_.clear();
        moeFfnOutHost_.clear();
        moeLogitsHost_.clear();
        moeRouterIdxHost_.clear();
        moeRouterWgtHost_.clear();
        expertCacheBuilt_ = false;
        expertType_ = 0;
        expertDownType_ = 0;
        expertCacheSets_ = kExpertCacheSetsDefault;
        // Per-layer expert cache arenas (hybrid qwen35moe mode).  Each layer
        // owns ONE contiguous arena anchored at ExpertSlot::gateQ (upQ/downQ
        // are interior pointers; only the anchor is freed).
        for (uint32_t L = 0; L < kExpertCacheLayers; ++L) {
            if (expertCache_[L].gateQ) {
                cudaFree(expertCache_[L].gateQ);
                expertCache_[L] = ExpertSlot();
            }
            for (uint32_t s2 = 0; s2 < kExpertCacheSets; ++s2) {
                expertCacheTag_[L][s2][0] = -1;
                expertCacheTag_[L][s2][1] = -1;
                expertCacheLru_[L][s2] = 0;
            }
        }
        expertCacheLayerBytes_ = 0;
        // Free the pinned H2D staging ring (registered malloc'd pages +
        // per-buffer events).
        for (uint32_t b = 0; b < kExpertStageBufs; ++b) {
            if (expertStage_[b]) {
                cudaHostUnregister(expertStage_[b]);
                std::free(expertStage_[b]);
                expertStage_[b] = nullptr;
            }
            if (expertStageEv_[b]) {
                cudaEventDestroy(expertStageEv_[b]);
                expertStageEv_[b] = nullptr;
            }
        }
        expertStageBytes_ = 0;
        expertStageFlag_ = 0;
        // ---- Static resident-expert arenas (Option D) ----
        // moeResidentCount_/moeResidentIds_ may never have been sized (no
        // policy built); guard all indexing by the vector being non-empty.
        for (uint32_t L = 0; L < kResidentLayers; ++L) {
            if (residentArena_[L]) {
                cudaFree(residentArena_[L]);
                residentArena_[L] = nullptr;
            }
            residentArenaBytes_[L] = 0;
            // moeResidentCount_/moeResidentIds_ may be sized to numLayers
            // (setResidentPolicy) OR empty (gate rejected before sizing) —
            // never index them past their real size (out-of-bounds .clear()
            // here corrupted the heap when the Q8_0 gate disabled Option D).
            if (L < moeResidentCount_.size()) moeResidentCount_[L] = 0;
            if (L < moeResidentIds_.size()) moeResidentIds_[L].clear();
            // residentIdxMap_ is indexed by EXPERT ID (up to 256), not by
            // resident slot — reset all 256 entries.
            for (uint32_t i = 0; i < 256; ++i) {
                residentIdxMap_[L][i] = -1;
            }
            for (uint32_t i = 0; i < kResidentMaxExperts; ++i) {
                residentGateBase_[L][i] = nullptr;
                residentUpBase_[L][i] = nullptr;
                residentDownBase_[L][i] = nullptr;
            }
        }
        if (residentRankWq_) {
            cudaFree(residentRankWq_);
            residentRankWq_ = nullptr;
        }
        residentRankWqBytes_ = 0;
        if (residentRankOf_) {
            cudaFree(residentRankOf_);
            residentRankOf_ = nullptr;
        }
        residentRankOfBytes_ = 0;
        if (residentWgt_) {
            cudaFree(residentWgt_);
            residentWgt_ = nullptr;
        }
        residentWgtBytes_ = 0;
        residentActive_ = false;
        residentMaskHost_.clear();
        moeDownTableHost_.clear();
        residentMaxPerLayer_ = 0;
        residentRowBytesGate_ = 0;
        residentRowBytesDown_ = 0;
        destroyScratch();
        allocated_ = false;
    }

    GPUModel::~GPUModel() { destroy(); }

}// namespace tinycoder::gpu

#endif// USE_CUDA
