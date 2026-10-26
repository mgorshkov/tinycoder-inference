/*
 * TinyCoder CUDA GPU offload engine -- upload & scratch
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

    // ===== Scratch allocation, weight upload (Prism), expert cache =====
    // GPU is enabled by default in any build compiled with USE_CUDA
    // (scripts/build.sh and npm run build:native produce such builds).
    // $TINYCODER_GPU=0 disables the offload engine (CPU-only), which is the
    // opt-out for systems without a usable CUDA device.
    bool gpuEnabled() {
        const char *e = std::getenv("TINYCODER_GPU");
        if (e == nullptr || e[0] == '\0') return true;
        return e[0] != '0';
    }

    bool GPUModel::ensureScratch(uint32_t seqLen, std::string &errMsg) {
        if (scratchAlloc_ && scratch_.seqCap >= seqLen) return true;
        // The CUDA decode graph froze pointers INTO the old scratch buffers
        // (scratch_.logits, s.hidden, s.norm, s.q/k/v, ...); the realloc below
        // changes those addresses.  Destroy the graph so the next decode
        // recaptures against the NEW buffers -- replaying a graph over freed
        // scratch hits an illegal memory access (observed on the first session
        // whose prefill exceeded the previous max prompt length, e.g. the 41
        // token Q4 after 36/33/35-token questions).
        destroyGraph();
        destroyScratch();
        // Allocate for the requested capacity (grow-on-demand; never shrink).
        uint32_t cap = std::max(seqLen, 1u);
        uint32_t H = geom_.hiddenSize, I = geom_.intermediateSize;
        uint32_t qLen = geom_.numAttentionHeads * geom_.headDim;
        uint32_t kvLen = geom_.numKVHeads * geom_.headDim;
        // Width of the "other" fp16 X-twin bucket (attnOutF16) used by the
        // prefill gemv helper for nCols that is neither H (hiddenF16) nor I
        // (gateF16).  Must cover the widest such input: the qwen35[MoE]
        // recurrent ssm_out GEMM reads valueDim = ssmInnerSize columns, and
        // the qwen35moe MoE down projections read expertFF / sharedFF
        // columns.  Allocating the bucket by qLen alone is too small when
        // qLen (nHeads*headDim) < ssmInnerSize (qwen35moe: 2048 < 4096).
        uint32_t f16OtherW = qLen;
        if (geom_.architecture == 1 || geom_.architecture == 2) {
            f16OtherW = std::max(f16OtherW, geom_.ssmInnerSize);
        }
        if (geom_.architecture == 2) {
            f16OtherW = std::max(f16OtherW, geom_.expertFF);
            f16OtherW = std::max(f16OtherW, geom_.expertSharedFF);
        }
        auto allocT = [&](auto *&p, uint64_t n, const char *what) -> bool {
            p = nullptr;
            cudaError_t e = cudaMalloc(&p, n);
            if (e != cudaSuccess) {
                p = nullptr;
                errMsg = std::string("cudaMalloc(scratch ") + what + "): " +
                         cudaGetErrorString(e);
                return false;
            }
            return true;
        };
        if (!allocT(scratch_.tokens, static_cast<uint64_t>(cap) * sizeof(int32_t),
                    "tokens") ||
            !allocT(scratch_.hidden, static_cast<uint64_t>(cap) * H * sizeof(float),
                    "hidden") ||
            !allocT(scratch_.norm, static_cast<uint64_t>(cap) * H * sizeof(float),
                    "norm") ||
            !allocT(scratch_.hiddenF16,
                    static_cast<uint64_t>(cap) * H * sizeof(uint16_t), "hiddenF16") ||
            !allocT(scratch_.q, static_cast<uint64_t>(cap) * qLen * sizeof(float),
                    "q") ||
            !allocT(scratch_.k, static_cast<uint64_t>(cap) * kvLen * sizeof(float),
                    "k") ||
            !allocT(scratch_.v, static_cast<uint64_t>(cap) * kvLen * sizeof(float),
                    "v") ||
            !allocT(scratch_.attnOut,
                    static_cast<uint64_t>(cap) * qLen * sizeof(float), "attnOut") ||
            !allocT(scratch_.attnOutF16,
                    static_cast<uint64_t>(cap) * f16OtherW * sizeof(uint16_t),
                    "attnOutF16") ||
            !allocT(scratch_.attnProj,
                    static_cast<uint64_t>(cap) * H * sizeof(float), "attnProj") ||
            !allocT(scratch_.gate, static_cast<uint64_t>(cap) * I * sizeof(float),
                    "gate") ||
            !allocT(scratch_.gateF16,
                    static_cast<uint64_t>(cap) * I * sizeof(uint16_t), "gateF16") ||
            !allocT(scratch_.up, static_cast<uint64_t>(cap) * I * sizeof(float),
                    "up") ||
            !allocT(scratch_.ffnOut,
                    static_cast<uint64_t>(cap) * H * sizeof(float), "ffnOut")) {
            destroyScratch();
            return false;
        }
        // ---- Qwen35 recurrent-layer scratch (dense gated delta net) ----
        // The dense projection outputs (qkv/z/beta/gateV) are BATCHED over
        // tokens ([seqLen][...]) because attnQKV/attnGate/ssmBetaQ/ssmAlphaQ are
        // plain dense GEMM/GEMVs; the conv + gated-delta-net recurrence is
        // inherently sequential, so those buffers are sized for ONE token
        // (the driver loops tokens feeding one token at a time).
        //   qkvDim = 2*keyDim + valueDim = 2*(headK*nKHeads) + dInner (10240)
        //   convChannels = qkvDim              convKernel = ssmConvKernel (4)
        //   headV = dInner / nVHeads (128)     qN/kN = headK*nVHeads (6144)
        //   vN/gdnOut/q35Norm = headV*nVHeads (6144)
        if (geom_.architecture == 1 || geom_.architecture == 2) {
            // Shared by the dense qwen35 and the qwen35moe recurrent layers
            // (qwen35moe uses the same gated-delta-net block; architecture 2
            // adds the MoE FFN scratch below).
            const uint32_t q35KeyDim = geom_.ssmStateSize * geom_.ssmGroupCount;
            const uint32_t q35QkvDim = 2 * q35KeyDim + geom_.ssmInnerSize;
            const uint32_t q35HeadV =
                    geom_.ssmInnerSize / std::max(geom_.ssmTimeStepRank, 1u);
            const uint32_t q35Kvn = geom_.ssmStateSize * geom_.ssmTimeStepRank;
            const uint32_t q35Vn = q35HeadV * geom_.ssmTimeStepRank;
            const uint32_t q35QGateDim =
                    geom_.numAttentionHeads * 2 * geom_.headDim;
            if (!allocT(scratch_.qkv,
                        static_cast<uint64_t>(cap) * q35QkvDim * sizeof(float),
                        "q35qkv") ||
                !allocT(scratch_.q35z,
                        static_cast<uint64_t>(cap) * geom_.ssmInnerSize *
                                sizeof(float),
                        "q35z") ||
                !allocT(scratch_.q35Beta,
                        static_cast<uint64_t>(cap) * geom_.ssmTimeStepRank *
                                sizeof(float),
                        "q35beta") ||
                !allocT(scratch_.q35GateV,
                        static_cast<uint64_t>(cap) * geom_.ssmTimeStepRank *
                                sizeof(float),
                        "q35gatev") ||
                !allocT(scratch_.q35QGate,
                        static_cast<uint64_t>(cap) * q35QGateDim * sizeof(float),
                        "q35qgate") ||
                !allocT(scratch_.convIn,
                        static_cast<uint64_t>(geom_.ssmConvKernel) *
                                q35QkvDim * sizeof(float),
                        "q35convin") ||
                !allocT(scratch_.convOut,
                        static_cast<uint64_t>(q35QkvDim) * sizeof(float),
                        "q35convout") ||
                !allocT(scratch_.qN,
                        static_cast<uint64_t>(q35Kvn) * sizeof(float), "q35qn") ||
                !allocT(scratch_.kN,
                        static_cast<uint64_t>(q35Kvn) * sizeof(float), "q35kn") ||
                !allocT(scratch_.vN,
                        static_cast<uint64_t>(q35Vn) * sizeof(float), "q35vn") ||
                !allocT(scratch_.gdnOut,
                        static_cast<uint64_t>(cap) * q35Vn * sizeof(float),
                        "q35gdn") ||
                !allocT(scratch_.q35Norm,
                        static_cast<uint64_t>(q35Vn) * sizeof(float),
                        "q35norm")) {
                destroyScratch();
                return false;
            }
        }
        // ---- Qwen35MoE (qwen35moe) MoE FFN scratch ----
        // Router logits [seqLen][expertCount], top-k routing [seqLen]x2*
        // expertUsed (int32 idx + float wgt), clamped-sum [seqLen].  Per-expert
        // gate/up + down are REUSED across ranks (one expert at a time, sized to
        // expertFF / hidden).  Shared-expert buffers are always present even when
        // the model disables the shared expert (geom fields then 0 -> 1-byte
        // allocs; the driver skips by geom.expertSharedFF > 0).
        if (geom_.architecture == 2 &&
            (geom_.expertCount > 0 && geom_.expertUsedCount > 0)) {
            const uint32_t ef = std::max(geom_.expertFF, 1u);
            const uint32_t sf = std::max(geom_.expertSharedFF, 1u);
            const uint32_t eu = geom_.expertUsedCount;
            if (!allocT(scratch_.moeLogits,
                        static_cast<uint64_t>(cap) * geom_.expertCount *
                                sizeof(float),
                        "moeLogits") ||
                !allocT(scratch_.moeIdx,
                        static_cast<uint64_t>(cap) * eu * sizeof(int32_t),
                        "moeIdx") ||
                !allocT(scratch_.moeWgt,
                        static_cast<uint64_t>(cap) * eu * sizeof(float),
                        "moeWgt") ||
                !allocT(scratch_.moeWsum,
                        static_cast<uint64_t>(cap) * sizeof(float), "moeWsum") ||
                !allocT(scratch_.moeGateUp,
                        static_cast<uint64_t>(cap) * ef * sizeof(float),
                        "moeGateUp") ||
                !allocT(scratch_.moeGateUp2,
                        static_cast<uint64_t>(cap) * ef * sizeof(float),
                        "moeGateUp2") ||
                !allocT(scratch_.moeDown,
                        static_cast<uint64_t>(cap) * H * sizeof(float),
                        "moeDown") ||
                // Per-rank slabs for the BATCHED decode fast path
                // (2026-09-14): all expertUsed ranks' gate/up/down single-
                // token buffers at once so the batched kernels can run the
                // ranks CONCURRENTLY (grid.y = rank) instead of the serialized
                // per-rank loop (was ~40 stream-serialized kernels/layer).
                !allocT(scratch_.moeGateUpP,
                        static_cast<uint64_t>(eu) * ef * sizeof(float),
                        "moeGateUpP") ||
                !allocT(scratch_.moeGateUp2P,
                        static_cast<uint64_t>(eu) * ef * sizeof(float),
                        "moeGateUp2P") ||
                !allocT(scratch_.moeDownP,
                        static_cast<uint64_t>(eu) * H * sizeof(float),
                        "moeDownP") ||
                !allocT(scratch_.moeExpertOut,
                        static_cast<uint64_t>(cap) * H * sizeof(float),
                        "moeExpertOut") ||
                !allocT(scratch_.moeShexpGate,
                        static_cast<uint64_t>(cap) * sizeof(float),
                        "moeShexpGate") ||
                !allocT(scratch_.moeShexpGateUp,
                        static_cast<uint64_t>(cap) * sf * sizeof(float),
                        "moeShexpGateUp") ||
                !allocT(scratch_.moeShexpGateUp2,
                        static_cast<uint64_t>(cap) * sf * sizeof(float),
                        "moeShexpGateUp2") ||
                !allocT(scratch_.moeShexpDown,
                        static_cast<uint64_t>(cap) * H * sizeof(float),
                        "moeShexpDown") ||
                // Double-buffered device snapshots of s.norm for the CPU-expert
                // hybrid handoff (see the moeNormSnap comment in the header).
                !allocT(scratch_.moeNormSnap[0],
                        static_cast<uint64_t>(cap) * H * sizeof(float),
                        "moeNormSnap0") ||
                !allocT(scratch_.moeNormSnap[1],
                        static_cast<uint64_t>(cap) * H * sizeof(float),
                        "moeNormSnap1")) {
                destroyScratch();
                return false;
            }
        }
        // ---- Split-KV decode attention partials (TINYCODER_ATTN_SPLIT) ----
        // Device-resident, pointer-stable across CUDA-graph capture.  Sized for
        // the full maxSeqLen chunk grid so the captured launch geometry never
        // changes; only the first ceil((cachePos+1)/chunk) chunks do work.
        {
            if (const char *ac = std::getenv("TINYCODER_ATTN_SPLIT_CHUNK")) {
                int v = std::atoi(ac);
                if (v > 0) attnChunk_ = static_cast<uint32_t>(v);
            }
            attnChunks_ = (geom_.maxSeqLen + attnChunk_ - 1) / attnChunk_;
            if (attnChunks_ == 0) attnChunks_ = 1;
            // The split-KV path is decode-only (seqLen == 1); both the split and
            // combine kernels index the partials by (token, head, chunk) with
            // token always 0 (kWarpAttentionSplit grid.x is seqLen).  Sizing the
            // partials by `cap` (up to maxSeqLen) multiplied this by up to 2048x
            // and, for wide-head models (Ternary-Bonsai: 96 q-heads), needed
            // ~3.2 GB — exhausting VRAM at upload.  Size for one token.
            const uint64_t nq = static_cast<uint64_t>(geom_.numAttentionHeads);
            const uint64_t accFloats = nq * attnChunks_ * geom_.headDim;
            const uint64_t mlFloats = nq * attnChunks_;
            if (!allocT(attnPartialAcc_, accFloats * sizeof(float),
                        "attnPartialAcc") ||
                !allocT(attnPartialM_, mlFloats * sizeof(float), "attnPartialM") ||
                !allocT(attnPartialL_, mlFloats * sizeof(float),
                        "attnPartialL")) {
                destroyScratch();
                return false;
            }
        }
        // ---- LM-head logits scratch (allocated LAST) ----
        // The generation path always requests ONLY the last token's logits, so
        // one [vocab] row suffices.  The historical [cap][vocab] buffer (2 GB
        // for cap=2048, vocab=248320) is wasteful: at Bonsai-2-27B's full
        // offload it is what starves the per-layer scratch (gateF16/q35qkv/...)
        // that ensureScratch allocates here, so the forward OOM'd at gateF16 and
        // dropped to the CPU ("out of memory").  Allocating it AFTER every other
        // scratch buffer — and only when the FULL buffer comfortably fits the
        // residual free VRAM — keeps those buffers safe.  Otherwise fall back to
        // a single row and flag it (logitsCapOneRow_) so the LM head writes just
        // the last token; computeAllLogits (all rows) is a debug/tool request
        // never made by the OpenAI generation path.
        logitsCapOneRow_ = false;
        {
            const uint64_t fullLogits =
                    static_cast<uint64_t>(cap) * geom_.vocabSize * sizeof(float);
            size_t freeB = 0, totB = 0;
            const bool memInfo = (cudaMemGetInfo(&freeB, &totB) == cudaSuccess);
            const uint64_t reserve = 512ull * 1024 * 1024;
            if (memInfo &&
                static_cast<uint64_t>(freeB) > fullLogits + reserve) {
                if (!allocT(scratch_.logits, fullLogits, "logits")) {
                    destroyScratch();
                    return false;
                }
            } else {
                if (!allocT(scratch_.logits,
                            static_cast<uint64_t>(geom_.vocabSize) * sizeof(float),
                            "logits")) {
                    destroyScratch();
                    return false;
                }
                logitsCapOneRow_ = true;
                std::fprintf(stderr,
                             "[gpu] logits scratch sized for ONE token "
                             "(%.1f MB) to conserve VRAM; full-sequence "
                             "logits unavailable\n",
                             static_cast<double>(geom_.vocabSize) *
                                     sizeof(float) / (1024.0 * 1024));
            }
        }
        scratch_.seqCap = cap;
        scratchAlloc_ = true;
        return true;
    }

    void GPUModel::destroyScratch() {
        if (!scratchAlloc_) return;
        cudaFree(attnPartialAcc_);
        attnPartialAcc_ = nullptr;
        cudaFree(attnPartialM_);
        attnPartialM_ = nullptr;
        cudaFree(attnPartialL_);
        attnPartialL_ = nullptr;
        cudaFree(scratch_.tokens);
        cudaFree(scratch_.hidden);
        cudaFree(scratch_.norm);
        cudaFree(scratch_.hiddenF16);
        scratch_.hiddenF16 = nullptr;
        cudaFree(scratch_.q);
        cudaFree(scratch_.k);
        cudaFree(scratch_.v);
        cudaFree(scratch_.attnOut);
        cudaFree(scratch_.attnOutF16);
        cudaFree(scratch_.attnProj);
        cudaFree(scratch_.gate);
        cudaFree(scratch_.gateF16);
        cudaFree(scratch_.up);
        cudaFree(scratch_.ffnOut);
        cudaFree(scratch_.logits);
        // Qwen35 recurrent + full-attention scratch.
        cudaFree(scratch_.qkv);
        cudaFree(scratch_.q35z);
        cudaFree(scratch_.q35Beta);
        cudaFree(scratch_.q35GateV);
        cudaFree(scratch_.q35QGate);
        cudaFree(scratch_.convIn);
        cudaFree(scratch_.convOut);
        cudaFree(scratch_.qN);
        cudaFree(scratch_.kN);
        cudaFree(scratch_.vN);
        cudaFree(scratch_.gdnOut);
        cudaFree(scratch_.q35Norm);
        // Qwen35MoE (qwen35moe) MoE FFN scratch.
        cudaFree(scratch_.moeLogits);
        cudaFree(scratch_.moeIdx);
        cudaFree(scratch_.moeWgt);
        cudaFree(scratch_.moeWsum);
        cudaFree(scratch_.moeGateUp);
        cudaFree(scratch_.moeGateUp2);
        cudaFree(scratch_.moeDown);
        cudaFree(scratch_.moeGateUpP);
        cudaFree(scratch_.moeGateUp2P);
        cudaFree(scratch_.moeDownP);
        cudaFree(scratch_.moeExpertOut);
        cudaFree(scratch_.moeShexpGate);
        cudaFree(scratch_.moeShexpGateUp);
        cudaFree(scratch_.moeShexpGateUp2);
        cudaFree(scratch_.moeShexpDown);
        cudaFree(scratch_.moeNormSnap[0]);
        cudaFree(scratch_.moeNormSnap[1]);
        // Value-initialize (zero all pointers) instead of std::memset: the
        // struct has non-trivial members, and std::memset on it trips
        // -Wclass-memaccess.
        scratch_ = DeviceScratch();
        scratchAlloc_ = false;
    }

    bool GPUModel::upload(const std::vector<DeviceLayer> &layers,
                          const ModelGeometry &geom, const void *embedQ,
                          uint32_t embedType, uint32_t embedRowBytes,
                          const void *lmHeadQ, uint32_t lmHeadType,
                          uint32_t lmHeadRowBytes, std::string &errMsg) {
        // Backward-compatible entry point: no Prism fold (plain models).
        return uploadWithPrism(layers, geom, embedQ, embedType, embedRowBytes,
                               lmHeadQ, lmHeadType, lmHeadRowBytes,
                               PrismGPUConfig{}, errMsg);
    }

    bool GPUModel::uploadWithPrism(const std::vector<DeviceLayer> &layers,
                                   const ModelGeometry &geom, const void *embedQ,
                                   uint32_t embedType, uint32_t embedRowBytes,
                                   const void *lmHeadQ, uint32_t lmHeadType,
                                   uint32_t lmHeadRowBytes,
                                   const PrismGPUConfig &prism,
                                   std::string &errMsg) {
        if (allocated_) destroy();
        // Clear any sticky CUDA error from a previous failed attempt (e.g. an
        // earlier upload's cudaMalloc OOM).  cudaGetLastError() consumes it;
        // leaving it set makes the NEXT forward's first cudaGetLastError()
        // (dequantMatrixF16) report the stale error and abort a healthy retry.
        if (cudaGetLastError() != cudaSuccess) {
            std::fprintf(stderr, "[gpu] upload: cleared sticky CUDA error\n");
        }
        if (!ensureCuda()) {
            errMsg = "CUDA runtime initialization failed";
            return false;
        }
        // numGpuLayers defaults to numLayers; caller may pass a smaller value in
        // geom (via the numGpuLayers field below) for partial offload.
        uint32_t nGpu = geom.numGpuLayers;
        if (nGpu == 0) nGpu = geom.numLayers;
        if (nGpu > geom.numLayers || nGpu > layers.size()) nGpu = geom.numLayers;

        geom_ = geom;
        geom_.numGpuLayers = nGpu;

        // ---- Prism Hadamard fold setup ----
        prismBlock_ = (prism.blockSize > 0) ? prism.blockSize : geom.prismBlock;
        if (prismBlock_ == 0) prismBlock_ = kPrismBlock;
        prismGdnVGrouped_ = prism.gdnVGrouped;
        // GDN value-head geometry (mirrors Model::applyPrismTransform):
        // headV = cols/nVHeads, rep = nVHeads/nKHeads (ssm_out input width =
        // ssmInnerSize = valueDim).
        prismHeadV_ = 0;
        prismNk_ = 0;
        prismRep_ = 0;
        if (prismGdnVGrouped_ && geom.ssmTimeStepRank > 0 &&
            geom.ssmGroupCount > 0 && geom.ssmInnerSize > 0 &&
            (geom.ssmTimeStepRank % geom.ssmGroupCount) == 0) {
            prismNk_ = geom.ssmGroupCount;
            prismHeadV_ = geom.ssmInnerSize / geom.ssmTimeStepRank;
            prismRep_ = geom.ssmTimeStepRank / geom.ssmGroupCount;
        }
        geom_.prismFold = prism.enabled ? 1u : 0u;
        geom_.prismBlock = prismBlock_;

        auto alloc = [&](void **p, size_t bytes, const char *what) -> bool {
            cudaError_t e = cudaMalloc(p, bytes);
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMalloc(") + what + "): " +
                         cudaGetErrorString(e);
                return false;
            }
            return true;
        };
        auto uploadBytes = [&](void *dst, const void *src, size_t bytes,
                               const char *what) -> bool {
            cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMemcpy(upload ") + what + "): " +
                         cudaGetErrorString(e);
                return false;
            }
            return true;
        };
        // On ANY allocation/upload failure, free everything already allocated
        // in this (or a previous failed) attempt.  Without this, each failed
        // upload leaks ~10.5 GB of VRAM (layer matrices + embed + KV slots are
        // allocated sequentially, and `allocated_` only flips true at the very
        // end), so every subsequent fresh session re-attempts the full upload
        // from an already-VRAM-starved device and eventually exhausts even the
        // CPU fallback's headroom.  The caller latches the failure (see
        // ModelGPUAdapter::gpuUnavailable_) so this cleanup runs at most once
        // per load.
        auto failCleanup = [&](const char *where) {
            std::fprintf(stderr,
                         "[gpu] upload failed at %s (%s); freeing %zu partial "
                         "device layers\n",
                         where, errMsg.c_str(),
                         static_cast<size_t>(layers_ ? nGpu : 0u));
            // destroy() starts with `if (!allocated_) return;`, and upload()
            // only flips allocated_ to true at the VERY END of a successful
            // upload.  Until then destroy() is a no-op, which previously leaked
            // every partially-allocated device buffer on a mid-upload OOM.
            // Bypass the guard so the partial allocations (layer matrices,
            // embedQ_, rope/MRoPE tables, q35 state, KV slots) are released.
            allocated_ = true;
            destroy();
            // Consume the sticky CUDA error left by the failed cudaMalloc.
            // cudaGetLastError() returns AND clears it; without this, the next
            // forward's first cudaGetLastError() (dequantMatrixF16) reports the
            // stale "out of memory" and aborts a healthy hybrid retry.
            (void) cudaGetLastError();
        };
        // Early-return sites below route through a common helper so the
        // partial allocations are always released on failure.
        auto uploadMat = [&](const DeviceMatrix &src, DeviceMatrix &dst,
                             const char *name) -> bool {
            if (src.q == nullptr) return true;
            dst = src;
            const uint64_t qBytes =
                    static_cast<uint64_t>(src.rows) * src.rowBytes;
            if (!alloc(&dst.q, qBytes, name)) {
                failCleanup(name);
                return false;
            }
            if (!uploadBytes(dst.q, src.q, qBytes, name)) {
                failCleanup(name);
                return false;
            }
            // NOTE: the persistent fp16 prefill twin is deliberately NOT
            // allocated here.  Doing it inline starved the quantized weights
            // themselves: on the 7B the twin pass consumed the whole free-VRAM
            // budget by layer ~14, so `cudaMalloc(ffnGate)` OOM'd and the
            // engine fell back to the CPU.  Twins are now allocated in a
            // dedicated second pass AFTER every quantized weight / KV / embed
            // buffer is resident (see "Persistent fp16 twins (partial set)"
            // below), sized to the residual free VRAM.
            return true;
        };
        auto uploadF32 = [&](float *&dst, const float *src, uint32_t n,
                             const char *name) -> bool {
            if (src == nullptr || n == 0) return true;
            if (!alloc(reinterpret_cast<void **>(&dst), sizeof(float) * n, name)) {
                failCleanup(name);
                return false;
            }
            if (!uploadBytes(dst, src, sizeof(float) * n, name)) {
                failCleanup(name);
                return false;
            }
            return true;
        };
        auto allocRaw = [&](void **p, size_t bytes, const char *what) -> bool {
            if (!alloc(p, bytes, what)) {
                failCleanup(what);
                return false;
            }
            return true;
        };
        auto uploadRaw = [&](void *dst, const void *src, size_t bytes,
                             const char *what) -> bool {
            if (!uploadBytes(dst, src, bytes, what)) {
                failCleanup(what);
                return false;
            }
            return true;
        };

        // ---- Persistent fp16 prefill twins (2026-09-29; partial-set 2026-10-01) ----
        // The prefill GEMM path dequantizes ONE matrix per layer into the
        // reusable wF16_ scratch (kDequantF16), then the fixed-space scratch is
        // overwritten by the next matrix.  nvprof showed this stream is the
        // prefill bottleneck: dequantizing gate+up+attnO+K/V reads ~4x the
        // quantized bytes (fp16 twin write + read-back) per layer (~1.6 ms of
        // the ~3 ms/layer prefill).  When the model FITS, allocate a persistent
        // fp16 twin per uploaded matrix ONCE (same kDequantF16 kernel, same
        // bytes — the GEMM input is bit-identical), then dequantMatrixF16()
        // returns the twin directly and the per-layer streaming dequant (and
        // its ~4x DRAM traffic) is skipped.
        //
        // PARTIAL SET (2026-10-01): the old code allocated twins INLINE during
        // the weight-upload loop with a budget measured BEFORE any of the
        // quantized weights were resident.  On the 7B (~2.4 GB of weights) the
        // twins consumed the whole free-VRAM budget by layer ~14, so the later
        // `cudaMalloc(ffnGate)` OOM'd and the engine fell back to the CPU
        // (0.4 tg tok/s).  The twin pass is now DEFERRED until every quantized
        // weight / KV / embed / scratch buffer is resident, and the budget is
        // the residual free VRAM minus a safety margin.  On a model too big for
        // full twins this leaves whatever fits: a partial twin set covering the
        // earliest (hottest) layers, with the streaming dequant as fallback for
        // the rest — no correctness impact, decode is untouched.
        // TINYCODER_PREFILL_TWINS=0 forces the streaming path.
        twinEnabled_ = false;
        twinBytesRemaining_ = 0;
        twinBytesAlloc_ = 0;
        twinMatrices_ = 0;
        {
            const char *tw = std::getenv("TINYCODER_PREFILL_TWINS");
            if (tw == nullptr || std::strcmp(tw, "0") != 0) {
                twinEnabled_ = true;// budget computed in the deferred pass
            }
        }

        // ---- Device LayerWeights (first nGpu layers only) ----
        layers_ = new DeviceLayer[nGpu];
        for (uint32_t L = 0; L < nGpu; ++L) {
            const DeviceLayer &hl = layers[L];
            DeviceLayer &dl = layers_[L];
            if (!uploadMat(hl.attnQ, dl.attnQ, "attnQ") ||
                !uploadMat(hl.attnK, dl.attnK, "attnK") ||
                !uploadMat(hl.attnV, dl.attnV, "attnV") ||
                !uploadMat(hl.attnO, dl.attnO, "attnO") ||
                !uploadMat(hl.ffnGate, dl.ffnGate, "ffnGate") ||
                !uploadMat(hl.ffnUp, dl.ffnUp, "ffnUp") ||
                !uploadMat(hl.ffnDown, dl.ffnDown, "ffnDown")) {
                return false;
            }
            // ---- Qwen35 extras (shared GDN block; qwen35moe adds the MoE FFN
            // extras below in its own architecture==2 block) ----
            if (geom_.architecture == 1 || geom_.architecture == 2) {
                // Recurrent / full-attention quantized matrices.
                if (!uploadMat(hl.attnQKV, dl.attnQKV, "attnQKV") ||
                    !uploadMat(hl.attnGate, dl.attnGate, "attnGate") ||
                    !uploadMat(hl.ssmAlphaQ, dl.ssmAlphaQ, "ssmAlphaQ") ||
                    !uploadMat(hl.ssmBetaQ, dl.ssmBetaQ, "ssmBetaQ") ||
                    !uploadMat(hl.ssmOut, dl.ssmOut, "ssmOut")) {
                    return false;
                }
                // ssm_conv1d: F32 [convChannels][convKernel]. The DeviceMatrix
                // descriptor came through toDeviceMatrix (rows/cols set, but
                // rowBytes==0 because type==F32 has no block encoding).  Upload
                // the raw rows*cols*4 bytes as an F32 blob for kQ35Conv1d.
                if (hl.ssmConv1d.q != nullptr && hl.ssmConv1d.rows > 0 &&
                    hl.ssmConv1d.cols > 0) {
                    dl.ssmConv1d.rows = hl.ssmConv1d.rows;
                    dl.ssmConv1d.cols = hl.ssmConv1d.cols;
                    uint64_t cb =
                            static_cast<uint64_t>(hl.ssmConv1d.rows) *
                            hl.ssmConv1d.cols * sizeof(float);
                    if (!alloc(&dl.ssmConv1d.q, static_cast<size_t>(cb),
                               "ssmConv1d"))
                        return false;
                    if (!uploadBytes(dl.ssmConv1d.q, hl.ssmConv1d.q, cb,
                                     "ssmConv1d"))
                        return false;
                }
                // F32 vectors.
                if (!uploadF32(dl.ssmABroadcast, hl.ssmABroadcast,
                               geom_.ssmTimeStepRank, "ssmA") ||
                    !uploadF32(dl.ssmDtBiasFull, hl.ssmDtBiasFull,
                               geom_.ssmTimeStepRank, "ssmDt") ||
                    !uploadF32(dl.ssmNorm, hl.ssmNorm,
                               geom_.ssmInnerSize / std::max(geom_.ssmTimeStepRank, 1u),
                               "ssmNorm") ||
                    !uploadF32(dl.attnQNorm, hl.attnQNorm, geom_.headDim, "attnQNorm") ||
                    !uploadF32(dl.attnKNorm, hl.attnKNorm, geom_.headDim, "attnKNorm") ||
                    !uploadF32(dl.postAttnNorm, hl.postAttnNorm, geom_.hiddenSize,
                               "postAttnNorm")) {
                    return false;
                }
            }
            // ---- Qwen35MoE (qwen35moe) MoE FFN extras ----
            // Full offload (`!moeCpuFn_`): upload the softmax router
            // (ffnGateInpMoe), the 256 per-expert matrices
            // (ffnGateExps/ffnUpExps/ffnDownExpsMoe, ~32B of the model's 35B
            // params — the VRAM hogs) and the shared expert (ffn*Shexp).
            //
            // CPU-expert hybrid (`moeCpuFn_` set): the 256 per-expert matrices
            // STAY OFF the device (~32B in host RAM; only 8/256 are touched
            // per token and those run on the CPU via the callback).  The
            // ROUTER and the SHARED EXPERT are small (router F32 [256x2048]
            // ~2 MB; ffn*Shexp ~12 MB) and are uploaded here so the GPU runs
            // them: the router kernel writes the top-k expert ids + weights
            // into the tiny host mirrors the callback consumes, and the shared
            // expert adds its contribution on the device.  That is what makes
            // IQ2_M (~11.5 GB) and Q4_K_M (~22 GB) fit an 11 GB card with a
            // busy GPU (shared expert runs there) while the CPU covers the
            // routed 8-of-256 experts.
            if (geom_.architecture == 2 &&
                hl.ffnGateInpMoe.q != nullptr) {
                // Router (F32 [expertCount x hidden]) is a raw float blob.
                if (!uploadMat(hl.ffnGateInpMoe, dl.ffnGateInpMoe,
                               "ffnGateInpMoe")) {
                    return false;
                }
                if (!moeCpuFn_) {
                    if (!uploadMat(hl.ffnGateExps, dl.ffnGateExps, "ffnGateExps") ||
                        !uploadMat(hl.ffnUpExps, dl.ffnUpExps, "ffnUpExps") ||
                        !uploadMat(hl.ffnDownExpsMoe, dl.ffnDownExpsMoe,
                                   "ffnDownExpsMoe")) {
                        return false;
                    }
                } else {
                    // ---- CPU-expert hybrid: record the HOST slice geometry for
                    // the per-expert device cache built later (buildExpertCache).
                    // The packed expert matrices stay in host RAM
                    // (ffnGateExps [expertCount*expertFF][H], ffnUpExps same,
                    // ffnDownExpsMoe [expertCount*H][expertFF]); the cache H2Ds
                    // ONE expert (3 slices) at a time.  Row strides come from
                    // the descriptors' block encoding.
                    //
                    // CRITICAL (2026-09-14): the H2D SOURCE must be recorded
                    // PER LAYER into expertCache_[L] (hostGate/hostUp/hostDown).
                    // Every layer has its OWN ffnGateExps/ffnUpExps/
                    // ffnDownExpsMoe host blob, and this loop runs once per
                    // layer; a single global expertHostGate_ would be
                    // overwritten every iteration and end up pointing at the
                    // LAST uploaded layer's matrices — making EVERY layer's
                    // cache execute the last layer's routed-expert weights
                    // (the L0 routed-output divergence root cause).
                    if (hl.ffnGateExps.q != nullptr) {
                        expertHostGate_ = static_cast<const uint8_t *>(hl.ffnGateExps.q);
                        expertRowBytesGate_ = hl.ffnGateExps.rowBytes;
                        expertType_ = hl.ffnGateExps.type;
                        expertCache_[L].hostGate = expertHostGate_;
                        // PER-LAYER geometry (2026-09-17): the globals above
                        // are overwritten once per layer and end up holding the
                        // LAST layer's values; the Ornith Q4_K_M mix changes
                        // the DOWN expert quant type between layers (most
                        // layers down=Q6_K, layers 5-6 down=Q4_K), so the cache
                        // must remember THIS layer's own type/rowBytes.
                        expertCache_[L].gateType = expertType_;
                        expertCache_[L].gateRowBytes = expertRowBytesGate_;
                    }
                    if (hl.ffnUpExps.q != nullptr) {
                        expertHostUp_ = static_cast<const uint8_t *>(hl.ffnUpExps.q);
                        expertCache_[L].hostUp = expertHostUp_;
                    }
                    if (hl.ffnDownExpsMoe.q != nullptr) {
                        expertHostDown_ = static_cast<const uint8_t *>(hl.ffnDownExpsMoe.q);
                        expertRowBytesDown_ = hl.ffnDownExpsMoe.rowBytes;
                        // The DOWN expert quant type is NOT necessarily the
                        // same as the gate/up type: the Ornith Q4_K_M mix has
                        // gate/up = Q4_K (12) but down = Q6_K (14).  The
                        // resident fast path needs BOTH recorded so each GEMV
                        // decodes the correct block layout (Q4K 144 B/block
                        // vs Q6K 210 B/block).
                        expertDownType_ = hl.ffnDownExpsMoe.type;
                        expertCache_[L].hostDown = expertHostDown_;
                        expertCache_[L].downType = expertDownType_;
                        expertCache_[L].downRowBytes = expertRowBytesDown_;
                    }
                    expertFF_ = geom_.expertFF;
                    expertCount_ = geom_.expertCount;
                    expertUsed_ = geom_.expertUsedCount;
                    expertHM_ = geom_.hiddenSize;
                }
                // Shared-expert matrices (gate/up/down quantized; the gate
                // vector ffn_gate_inp_shexp is F32 [1 x hidden]).
                if (hl.ffnGateShexp.q != nullptr) {
                    if (!uploadMat(hl.ffnGateShexp, dl.ffnGateShexp,
                                   "ffnGateShexp") ||
                        !uploadMat(hl.ffnUpShexp, dl.ffnUpShexp, "ffnUpShexp") ||
                        !uploadMat(hl.ffnDownShexp, dl.ffnDownShexp,
                                   "ffnDownShexp")) {
                        return false;
                    }
                }
                // ffn_gate_inp_shexp: F32 [1 x hidden] — ONE row blob.  The
                // DeviceMatrix came through toDeviceMatrix with type==F32,
                // which sets block encoding = raw floats (rowBytes=cols*4),
                // rows==1, cols==hidden.  uploadMat would need rows*rowBytes;
                // we handle the F32 blob directly (same as ssm_conv1d).
                if (hl.ffnGateInpShexp.q != nullptr &&
                    hl.ffnGateInpShexp.rows > 0) {
                    const uint64_t cb = static_cast<uint64_t>(hl.ffnGateInpShexp.rows) *
                                        hl.ffnGateInpShexp.cols * sizeof(float);
                    dl.ffnGateInpShexp.rows = hl.ffnGateInpShexp.rows;
                    dl.ffnGateInpShexp.cols = hl.ffnGateInpShexp.cols;
                    if (!allocRaw(&dl.ffnGateInpShexp.q, static_cast<size_t>(cb),
                                  "ffnGateInpShexp"))
                        return false;
                    if (!uploadRaw(dl.ffnGateInpShexp.q, hl.ffnGateInpShexp.q, cb,
                                   "ffnGateInpShexp"))
                        return false;
                }
            }
            uint32_t qLen = geom_.numAttentionHeads * geom_.headDim;
            uint32_t kvLen = geom_.numKVHeads * geom_.headDim;
            // For Qwen2, biases are [qLen]/[kvLen]; for Gemma/Qwen35MoE they are
            // absent (qwen2Bias == 0 routes them to null).
            uint32_t qb = geom_.qwen2Bias ? qLen : 0;
            uint32_t kb = geom_.qwen2Bias ? kvLen : 0;
            if (!uploadF32(dl.attnQBias, hl.attnQBias, qb, "attnQBias") ||
                !uploadF32(dl.attnKBias, hl.attnKBias, kb, "attnKBias") ||
                !uploadF32(dl.attnVBias, hl.attnVBias, kb, "attnVBias") ||
                !uploadF32(dl.rmsNormAttn, hl.rmsNormAttn, geom_.hiddenSize, "rmsAttn") ||
                !uploadF32(dl.rmsNormFFN, hl.rmsNormFFN, geom_.hiddenSize, "rmsFFN")) {
                return false;
            }
        }

        // ---- Final norm ----
        if (geom.finalNorm != nullptr) {
            if (!allocRaw(reinterpret_cast<void **>(&finalNorm_),
                          geom.hiddenSize * sizeof(float), "finalNorm"))
                return false;
            if (!uploadRaw(finalNorm_, geom.finalNorm,
                           geom.hiddenSize * sizeof(float), "finalNorm"))
                return false;
        }

        // ---- Embedding (quantized rows) ----
        if (embedQ != nullptr) {
            uint64_t embedBytes = static_cast<uint64_t>(geom.vocabSize) * embedRowBytes;
            if (!allocRaw(reinterpret_cast<void **>(&embedQ_), embedBytes, "embedQ"))
                return false;
            if (!uploadRaw(embedQ_, embedQ, embedBytes, "embedQ")) return false;
            embedType_ = embedType;
            embedRowBytes_ = embedRowBytes;
        }

        // ---- Prism Hadamard-fold sign vectors (Ternary-Bonsai PQ2_0) ----
        // Uploaded once; tiny (<= 17408 floats each).  Only the widths that
        // carry a sign vector are allocated; prismSignsFor() returns null for
        // the rest (identity signs).
        if (prism.enabled) {
            auto upSigns = [&](const float *src, uint32_t n, float *&dst,
                               uint32_t &nDst, const char *what) -> bool {
                if (src == nullptr || n == 0) return true;
                if (!allocRaw(reinterpret_cast<void **>(&dst),
                              sizeof(float) * n, what))
                    return false;
                if (!uploadRaw(dst, src, sizeof(float) * n, what)) return false;
                nDst = n;
                return true;
            };
            if (!upSigns(prism.signs5120, prism.n5120, signs5120_, nSigns5120_,
                         "prismSigns5120"))
                return false;
            if (!upSigns(prism.signs6144, prism.n6144, signs6144_, nSigns6144_,
                         "prismSigns6144"))
                return false;
            if (!upSigns(prism.signs17408, prism.n17408, signs17408_,
                         nSigns17408_, "prismSigns17408"))
                return false;
            // Device scratch for the folded activation.  Sized to the widest
            // folded projection input (= max hidden/intermediate/ssm-inner).
            const uint32_t widest = std::max(
                    std::max(geom.hiddenSize, geom.intermediateSize),
                    std::max(geom.ssmInnerSize, 1u));
            if (widest > 0) {
                const uint64_t fbBytes = sizeof(float) * widest;
                if (!allocRaw(reinterpret_cast<void **>(&prismFoldBuf_), fbBytes,
                              "prismFoldBuf"))
                    return false;
                prismFoldBufBytes_ = fbBytes;
            }
            std::fprintf(stderr,
                         "[gpu] prism fold enabled: block=%u gdnVGrouped=%d "
                         "signs[%u/%u/%u] foldBuf=%u floats\n",
                         prismBlock_, prismGdnVGrouped_ ? 1 : 0, nSigns5120_,
                         nSigns6144_, nSigns17408_,
                         static_cast<unsigned>(widest));
        }

        // ---- Separate LM head (quantized rows; nullable for tied heads) ----
        // In partial-offload mode (nGpu < numLayers) the shared tail runs the
        // final RMSNorm + LM head on the CPU (see Model::forward continuation),
        // so a separate output.weight matrix would NEVER be used on the GPU.
        // lmHeadQ_ stays null; the full-offload tail falls back to the tied
        // embedding matrix, which is correct because a full offload never has a
        // separate head without uploading it first.
        const bool useGpuLmHead = (lmHeadQ != nullptr) && (nGpu == geom.numLayers);
        if (useGpuLmHead) {
            uint64_t lmBytes = static_cast<uint64_t>(geom.vocabSize) * lmHeadRowBytes;
            if (!allocRaw(reinterpret_cast<void **>(&lmHeadQ_), lmBytes, "lmHeadQ"))
                return false;
            if (!uploadRaw(lmHeadQ_, lmHeadQ, lmBytes, "lmHeadQ")) return false;
            lmHeadType_ = lmHeadType;
            lmHeadRowBytes_ = lmHeadRowBytes;
        }

        // ---- RoPE tables [maxSeqLen][headDim/2] ----
        {
            uint32_t pairs = geom_.headDim / 2;
            std::vector<float> cosT(static_cast<size_t>(geom_.maxSeqLen) * pairs);
            std::vector<float> sinT(static_cast<size_t>(geom_.maxSeqLen) * pairs);
            std::vector<float> freq(pairs);
            for (uint32_t d2 = 0; d2 < pairs; ++d2) {
                freq[d2] = 1.0f / std::pow(geom_.ropeTheta,
                                           static_cast<float>(2 * d2) / geom_.headDim);
            }
            for (uint32_t p = 0; p < geom_.maxSeqLen; ++p) {
                for (uint32_t d2 = 0; d2 < pairs; ++d2) {
                    float ang = static_cast<float>(p) * freq[d2];
                    cosT[static_cast<size_t>(p) * pairs + d2] = std::cos(ang);
                    sinT[static_cast<size_t>(p) * pairs + d2] = std::sin(ang);
                }
            }
            if (!allocRaw(reinterpret_cast<void **>(&ropeCos_),
                          cosT.size() * sizeof(float), "ropeCos") ||
                !allocRaw(reinterpret_cast<void **>(&ropeSin_),
                          sinT.size() * sizeof(float), "ropeSin"))
                return false;
            if (!uploadRaw(ropeCos_, cosT.data(), cosT.size() * sizeof(float),
                           "ropeCos") ||
                !uploadRaw(ropeSin_, sinT.data(), sinT.size() * sizeof(float),
                           "ropeSin"))
                return false;
        }

        // ---- Qwen35 MRoPE cache [maxSeqLen][ropeDimensionCount/2] ----
        // Mirrors Model::applyMRoPE's theta_scale = thetaBase^(-2/nDims) with the
        // cache STARTING theta at 1.0f (not thetaBase: ggml seeds mrope cache
        // theta with the token POSITION, so freq_base enters only through
        // theta_scale).  The standard rope tables above are unused for qwen35.
        if (geom_.architecture == 1 || geom_.architecture == 2) {
            // Shared by qwen35 and qwen35moe (same gated-delta-net block; the
            // MoE arch reuses the MRoPE cache + persistent conv/GDN state).
            const uint32_t nDims = geom_.ropeDimensionCount > 0
                                           ? geom_.ropeDimensionCount
                                           : 64;
            const uint32_t nDimsHalf = nDims / 2;
            const float thetaBase = geom_.ropeTheta;
            const float thetaScale =
                    std::pow(thetaBase, -2.0f / static_cast<float>(nDims));
            std::vector<float> mcos(static_cast<size_t>(geom_.maxSeqLen) * nDimsHalf);
            std::vector<float> msin(static_cast<size_t>(geom_.maxSeqLen) * nDimsHalf);
            for (uint32_t p = 0; p < geom_.maxSeqLen; ++p) {
                float theta = 1.0f;
                for (uint32_t d2 = 0; d2 < nDimsHalf; ++d2) {
                    mcos[static_cast<size_t>(p) * nDimsHalf + d2] =
                            std::cos(static_cast<float>(p) * theta);
                    msin[static_cast<size_t>(p) * nDimsHalf + d2] =
                            std::sin(static_cast<float>(p) * theta);
                    theta *= thetaScale;
                }
            }
            if (!allocRaw(reinterpret_cast<void **>(&mropeCos_),
                          mcos.size() * sizeof(float), "mropeCos") ||
                !allocRaw(reinterpret_cast<void **>(&mropeSin_),
                          msin.size() * sizeof(float), "mropeSin"))
                return false;
            if (!uploadRaw(mropeCos_, mcos.data(), mcos.size() * sizeof(float),
                           "mropeCos") ||
                !uploadRaw(mropeSin_, msin.data(), msin.size() * sizeof(float),
                           "mropeSin"))
                return false;

            // Persistent recurrent-layer state (qwen35) — conv sliding window
            // [(convKernel-1)*convChannels] and gated-delta-net [nVHeads*headV^2]
            // per offloaded GPU layer.  Zero-initialized (fresh session).
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
            if (!allocRaw(reinterpret_cast<void **>(&q35ConvState_),
                          static_cast<size_t>(q35ConvBytes), "q35ConvState") ||
                !allocRaw(reinterpret_cast<void **>(&q35GdnState_),
                          static_cast<size_t>(q35GdnBytes), "q35GdnState"))
                return false;
            cudaMemsetAsync(q35ConvState_, 0,
                            static_cast<size_t>(q35ConvBytes), g_stream);
            cudaMemsetAsync(q35GdnState_, 0,
                            static_cast<size_t>(q35GdnBytes), g_stream);
        }

        // ---- KV cache (device, only nGpu layers) ----
        // fp16 KV cache (TINYCODER_KV16): dense qwen2 only; halves the O(context)
        // attention/RoPE DRAM traffic.  qwen35 keeps fp32 (its GDN kernels index
        // kvK_/kvV_ as float*).  TINYCODER_KV16=0 forces fp32.
        // Opt-in (default OFF): measured +0.4% tg64 at pp1024 but -6% prefill —
        // the 2026-10-01 split-KV kernel already removed attention as the decode
        // bottleneck (20.27 -> 1.74 ms at pp1024), so halving its bytes no longer
        // moves the wall.  Kept as a measured A/B record; TINYCODER_KV16=1 enables.
        {
            const char *kh = std::getenv("TINYCODER_KV16");
            kvHalf_ = (geom_.architecture == 0) && (kh != nullptr && kh[0] == '1');
        }
        const uint64_t kvElem = kvHalf_ ? sizeof(__half) : sizeof(float);
        // Per-layer KV slot bytes (ONE layer's K or V).
        kvLayerStride_ = static_cast<uint64_t>(geom_.maxSeqLen) *
                         geom_.numKVHeads * geom_.headDim * kvElem;
        // Slot COUNT: dense archs slot every offloaded layer; qwen35/qwen35moe
        // store K/V ONLY on the full-attention layers (every
        // fullAttentionInterval-th layer, at L ≡ interval-1 (mod interval)), so
        // the recurrent (gated-delta-net) layers get NO slot.  Slot index for
        // layer L is floor(L / interval) — see kvLayerOffset().  This frees
        // ~3/4 of the KV VRAM on the 48-recurrent/16-attention Bonsai-2-27B.
        kvSlots_ = nGpu;
        if ((geom_.architecture == 1u || geom_.architecture == 2u) &&
            geom_.fullAttentionInterval > 1u) {
            kvSlots_ = (nGpu + geom_.fullAttentionInterval - 1u) /
                       geom_.fullAttentionInterval;
        }
        const uint64_t kvAllocBytes =
                static_cast<uint64_t>(kvSlots_) * kvLayerStride_;
        if (!allocRaw(&kvK_, static_cast<size_t>(kvAllocBytes), "kvK") ||
            !allocRaw(&kvV_, static_cast<size_t>(kvAllocBytes), "kvV"))
            return false;
        cudaMemsetAsync(kvK_, 0, static_cast<size_t>(kvAllocBytes), g_stream);
        cudaMemsetAsync(kvV_, 0, static_cast<size_t>(kvAllocBytes), g_stream);

        // ---- Persistent fp16 twins: DEFERRED partial-set pass (2026-10-01) ----
        // Every quantized weight / KV / embed / scratch buffer is now resident,
        // so the residual free VRAM is the true twin budget (the old inline pass
        // measured the budget before the weights existed and starved the 7B).
        // Allocate as many per-matrix fp16 twins as fit, walking layers in
        // order — on a model too big for the full set this leaves a prefix
        // partial set, with the streaming dequant as the fallback for the rest.
        // Each twin is dequantized ONCE with the same kDequantF16 kernel the
        // streaming path uses, so the fp16 GEMM input is bit-identical.
        // Best-effort: an OOM consumes the sticky error and leaves f16Twin==NULL.
        twinBytesRemaining_ = 0;
        twinBytesAlloc_ = 0;
        twinMatrices_ = 0;
        if (twinEnabled_) {
            size_t freeB = 0, totB = 0;
            // Reserve enough for the streaming-dequant scratch that the
            // NON-twinned matrices still need: the reusable wF16_ buffer (the
            // largest matrix's fp16 form) plus ensureScratch's `maxSeqLen *
            // vocabSize` logits buffer and the per-layer activations.  A flat
            // 256 MB was too small — on the 7B it left no room for wF16_
            // (~136 MB) and the pp128 scratch (~93 MB), so the first prefill
            // OOM'd and fell back to the CPU.  Using the model's own geometry
            // makes the reservation correct for any model/context.
            uint64_t largestMat = 0;
            for (uint32_t L = 0; L < nGpu; ++L) {
                const DeviceLayer &dl = layers_[L];
                const DeviceMatrix *ms[] = {&dl.attnQ, &dl.attnK, &dl.attnV,
                                            &dl.attnO, &dl.ffnGate, &dl.ffnUp,
                                            &dl.ffnDown, &dl.attnQKV, &dl.attnGate,
                                            &dl.ssmAlphaQ, &dl.ssmBetaQ, &dl.ssmOut};
                for (const DeviceMatrix *m: ms) {
                    const uint64_t b =
                            static_cast<uint64_t>(m->rows) * m->cols * sizeof(__half);
                    if (b > largestMat) largestMat = b;
                }
            }
            // scratchEst mirrors ensureScratch's dominant terms.  The logits
            // term is ONE row when ensureScratch fell back to the one-row
            // buffer (logitsCapOneRow_); only a full [maxSeqLen][vocab] buffer
            // needs the cap*vocab reservation.  The rest is a small multiple of
            // cap*hidden plus a fragmentation allowance.
            const uint64_t logitsEst =
                    logitsCapOneRow_
                            ? static_cast<uint64_t>(geom_.vocabSize) * sizeof(float)
                            : static_cast<uint64_t>(geom_.maxSeqLen) *
                                      geom_.vocabSize * sizeof(float);
            const uint64_t scratchEst =
                    logitsEst +
                    static_cast<uint64_t>(geom_.maxSeqLen) * geom_.hiddenSize *
                            sizeof(float) * 24 +
                    128ull * 1024 * 1024;
            uint64_t margin = scratchEst + largestMat + 256ull * 1024 * 1024;
            if (const char *mb = std::getenv("TINYCODER_PREFILL_TWIN_MARGIN_MB")) {
                long v = std::atol(mb);
                if (v >= 0) margin = static_cast<uint64_t>(v) * 1024 * 1024;
            }
            if (cudaMemGetInfo(&freeB, &totB) == cudaSuccess && freeB > margin) {
                twinBytesRemaining_ = freeB - margin;
            } else {
                twinEnabled_ = false;
            }
            auto twinMat = [&](DeviceMatrix &m) {
                if (!twinEnabled_ || m.q == nullptr || m.rows == 0 || m.cols == 0)
                    return;
                const uint64_t need =
                        static_cast<uint64_t>(m.rows) * m.cols * sizeof(__half);
                if (need == 0 || need > twinBytesRemaining_) return;
                cudaError_t e = cudaMalloc(&m.f16Twin, need);
                if (e != cudaSuccess) {
                    // Best-effort: consume the sticky OOM and keep streaming.
                    (void) cudaGetLastError();
                    m.f16Twin = nullptr;
                    m.f16TwinBytes = 0;
                    return;
                }
                launchDequantF16(m.type, m.q, m.f16Twin, m.rows, m.cols,
                                 m.rowBytes, m.blocksPerRow);
                if (cudaGetLastError() != cudaSuccess) {
                    cudaFree(m.f16Twin);
                    m.f16Twin = nullptr;
                    m.f16TwinBytes = 0;
                    return;
                }
                m.f16TwinBytes = need;
                twinBytesRemaining_ -= need;
                twinBytesAlloc_ += need;
                ++twinMatrices_;
            };
            // Largest-matrices-first: each twin eliminates a fixed per-matrix
            // streaming dequant (~4x the quantized bytes of DRAM traffic), so
            // spending the residual budget on the biggest matrices (ffn
            // gate/up/down) removes the most prefill traffic per byte of VRAM.
            std::vector<DeviceMatrix *> twinCands;
            twinCands.reserve(static_cast<size_t>(nGpu) * 7 + 8);
            for (uint32_t L = 0; L < nGpu; ++L) {
                DeviceLayer &dl = layers_[L];
                twinCands.push_back(&dl.attnQ);
                twinCands.push_back(&dl.attnK);
                twinCands.push_back(&dl.attnV);
                twinCands.push_back(&dl.attnO);
                twinCands.push_back(&dl.ffnGate);
                twinCands.push_back(&dl.ffnUp);
                twinCands.push_back(&dl.ffnDown);
                if (geom_.architecture == 1 || geom_.architecture == 2) {
                    twinCands.push_back(&dl.attnQKV);
                    twinCands.push_back(&dl.attnGate);
                    twinCands.push_back(&dl.ssmAlphaQ);
                    twinCands.push_back(&dl.ssmBetaQ);
                    twinCands.push_back(&dl.ssmOut);
                }
            }
            std::stable_sort(twinCands.begin(), twinCands.end(),
                             [](const DeviceMatrix *a, const DeviceMatrix *b) {
                                 return static_cast<uint64_t>(a->rows) * a->cols >
                                        static_cast<uint64_t>(b->rows) * b->cols;
                             });
            for (DeviceMatrix *m: twinCands) twinMat(*m);
            if (twinBytesAlloc_ == 0) twinEnabled_ = false;
        }

        kvPos_ = 0;
        allocated_ = true;
        cudaStreamSynchronize(g_stream);
        if (twinEnabled_) {
            std::fprintf(stderr,
                         "[gpu] prefill fp16 twins: %u matrices, %.1f MiB "
                         "allocated (%.1f MiB budget remaining)\n",
                         twinMatrices_,
                         static_cast<double>(twinBytesAlloc_) / (1024.0 * 1024),
                         static_cast<double>(twinBytesRemaining_) /
                                 (1024.0 * 1024));
        } else {
            std::fprintf(stderr,
                         "[gpu] prefill fp16 twins DISABLED (streaming dequant "
                         "path)\n");
        }
        return true;
    }

    // 2-way set-associative expert cache set index for expert e: the
    // upper bits of the Knuth multiply-high hash, masked to the runtime
    // SET count (TINYCODER_MOE_WAYS = sets, default 16).  The set is a
    // power of 2 <= kExpertCacheSets so the static tag grid covers the
    // prefix of sets used.  Slot index within the arena =
    // set*kExpertCacheWays + way (way chosen by the tag/LRU lookup in
    // ensureExpertCached).
    // Set index for expert e (2-way set-associative): the upper bits of
    // the Knuth multiply-high hash, masked to the runtime SET count
    // (TINYCODER_MOE_WAYS = sets, default 16).  The set is a power of 2
    // <= kExpertCacheSets so the static tag grid covers the prefix used.
    // Slot index within the arena = set*kExpertCacheWays + way (way from
    // the tag/LRU lookup in ensureExpertCached).
    uint32_t GPUModel::expertCacheSet(uint32_t e) {
        const uint32_t sets = expertCacheSets_ ? expertCacheSets_
                                               : kExpertCacheSetsDefault;
        const uint32_t h = static_cast<uint32_t>(
                (static_cast<uint64_t>(e) * 2654435761u) >> 27u);
        // Power-of-2 -> cheap mask; otherwise modulo (harmless; only used
        // for the TINYCODER_MOE_WAYS capacity A/B with e.g. 24 sets).
        return (sets & (sets - 1u)) == 0u ? (h & (sets - 1u)) : (h % sets);
    }

    double bwSelfTestHostToPinned(const uint8_t *src, uint8_t *dst,
                                  size_t bytes, int iters) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            std::memcpy(dst, src, bytes);
        double sec = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
        return (double) bytes * iters / sec / (1ull << 30);
    }

}// namespace tinycoder::gpu

#endif// USE_CUDA
