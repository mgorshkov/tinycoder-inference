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

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#ifdef USE_CUDA

namespace tinycoder::gpu {

    /// @brief One quantized weight matrix uploaded to the device.
    ///
    /// `q` holds the raw GGUF quantized bytes (Q2_K/Q3_K/Q4_K/Q6_K/IQ2_XS/
    /// IQ3_S/Q5_K/...) so single-token generation streams the compact block
    /// layout straight from VRAM (the same memory traffic llama.cpp's CUDA
    /// GEMV uses).  There are NO persistent FP16 twins anymore: the cuBLAS
    /// tensor-core batch-GEMM prefill path dequantizes ONE matrix per layer
    /// on-device into the reusable wF16_ scratch (see dequantMatrixF16),
    /// which keeps 7B+ models inside 11 GB of VRAM.
    struct DeviceMatrix {
        void *q = nullptr;        // raw quantized bytes (row-major block layout)
        uint32_t rows = 0;        // output rows
        uint32_t cols = 0;        // input columns
        uint32_t type = 0;        // GGML_TYPE_* enum
        uint32_t blocksPerRow = 0;// number of 256-elem blocks per row
        uint32_t rowBytes = 0;    // q row stride in bytes
        bool empty_ = false;
        bool hasQ() const { return q != nullptr; }
    };

    /// @brief Qwen2 dense LayerWeight block on the device (host pointers BEFORE
    /// upload: upload() copies them to device and keeps the descriptors).
    /// Qwen35 (dense gated-delta-net) recurrent layers reuse the quantized
    /// matrix fields for their extra tensors (attnQKV/attnGate/ssmAlphaQ/
    /// ssmBetaQ/ssmOut via attnQ/attnK/attnV/attnO slots when the arch needs
    /// them); full-attention layers use attnQ/attnK/attnV/attnO directly.
    struct DeviceLayer {
        DeviceMatrix attnQ, attnK, attnV, attnO;
        DeviceMatrix ffnGate, ffnUp, ffnDown;
        float *attnQBias = nullptr, *attnKBias = nullptr, *attnVBias = nullptr;
        float *rmsNormAttn = nullptr, *rmsNormFFN = nullptr;

        // Qwen35 recurrent-layer extras (dense qwen35: gated delta net).
        // attnQKV (recurrent fused Q|K|V), attnGate (z), ssmBetaQ, ssmAlphaQ,
        // ssmOut are quantized matrices; ssmConv1d is F32 [convKernel][channels];
        // ssmABroadcast/ssmDtBiasFull/ssmNorm are F32 vectors.
        DeviceMatrix attnQKV, attnGate;
        DeviceMatrix ssmAlphaQ, ssmBetaQ, ssmOut;
        DeviceMatrix ssmConv1d;        // F32 [convKernel][convChannels]
        float *ssmABroadcast = nullptr;// nVHeads
        float *ssmDtBiasFull = nullptr;// nVHeads
        float *ssmNorm = nullptr;      // [headV] (per-value-head norm weight)

        // Qwen35 full-attention-layer extras.
        float *attnQNorm = nullptr, *attnKNorm = nullptr;// headDim each
        float *postAttnNorm = nullptr;                   // hiddenSize (FFN norm input)

        // ---- Qwen35MoE (qwen35moe) MoE FFN tensors ----
        // ffnGateInpMoe: F32 [expertCount x hiddenSize] router (rows ==
        // expertCount, one output per expert).  expert rows are packed
        // contiguously: expert e of ffnGateExps/ffnUpExps spans rows
        // [e*expertFF, (e+1)*expertFF); of ffnDownExpsMoe rows
        // [e*hiddenSize, (e+1)*hiddenSize).
        DeviceMatrix ffnGateInpMoe;  // F32 router
        DeviceMatrix ffnGateExps;    // [expertCount*expertFF, hidden]
        DeviceMatrix ffnUpExps;      // [expertCount*expertFF, hidden]
        DeviceMatrix ffnDownExpsMoe; // [expertCount*hidden, expertFF]
        DeviceMatrix ffnGateShexp;   // shared-expert gate [hidden x sharedFF]
        DeviceMatrix ffnUpShexp;     // shared-expert up   [hidden x sharedFF]
        DeviceMatrix ffnDownShexp;   // shared-expert down [sharedFF x hidden]
        DeviceMatrix ffnGateInpShexp;// F32 shared-expert router [1 x hidden]
    };

    /// @brief Geometry + final norm shared by all kernels.
    struct ModelGeometry {
        uint32_t hiddenSize = 1536;
        uint32_t intermediateSize = 8960;
        uint32_t numLayers = 28;
        uint32_t numGpuLayers = 0;// 0 = offload all layers; >0 = offload only the
                                  // first N layers (llama.cpp -ngl style partial
                                  // offload, CPU handles the rest)
        uint32_t numAttentionHeads = 12;
        uint32_t numKVHeads = 2;
        uint32_t headDim = 128;
        uint32_t maxSeqLen = 2048;
        uint32_t vocabSize = 151936;
        float ropeTheta = 1000000.0f;
        uint32_t qwen2Bias = 1;          // Qwen2 has Q/K/V biases; Gemma4/Qwen35MoE don't
        const float *finalNorm = nullptr;// hiddenSize floats (host)
        uint32_t lmHeadTied = 1;         // 1: LM head == token embeddings (embedQ). 0: the
                                         // adapter uploads the separate output.weight matrix
                                         // as embedQ (rows == vocabSize).

        // ---- Qwen35 / Qwen35MoE (gated-delta-net hybrid) geometry ----
        uint32_t architecture = 0;      // 0=qwen2-style, 1=qwen35, 2=qwen35moe
        uint32_t attentionKeyLength = 0;// explicit per-head dim (256 for 27B); 0 = headDim
        uint32_t attentionValueLength = 0;
        uint32_t ropeDimensionCount = 0;// MRoPE n_rot (64); 0 = use headDim
        uint32_t ropeDimensionSections[4] = {0, 0, 0, 0};
        uint32_t fullAttentionInterval = 0;// 0 = all layers full-attention
        uint32_t ssmInnerSize = 0;         // dInner (6144)
        uint32_t ssmStateSize = 0;         // headK (128)
        uint32_t ssmGroupCount = 0;        // nKHeads (16)
        uint32_t ssmTimeStepRank = 0;      // nVHeads (48)
        uint32_t ssmConvKernel = 0;        // (4)
        uint32_t nextnPredictLayers = 0;   // MTP layers (1 for 27B; never executed on GPU)

        // ---- Qwen35MoE (qwen35moe) expert geometry ----
        // 0 rows = the layer has no MoE FFN (dense qwen35: ignore these).
        uint32_t expertCount = 0;    // total experts (256)
        uint32_t expertUsedCount = 0;// routed top-k (8)
        uint32_t expertFF = 0;       // per-expert feed-forward width (512)
        uint32_t expertSharedFF = 0; // shared-expert FF width (512)
    };

    /// @brief GPU model runtime: owns device memory, hosts the persistent KV cache
    /// and drives a full forward pass (quantized GEMV generation or cuBLAS fp16
    /// GEMM prefill).
    class GPUModel {
    public:
        /// @brief Allocate device buffers and upload weights.
        /// @param layers Host layer weight descriptors (first geom.numGpuLayers of
        ///        them are uploaded; 0 in geom = all)
        /// @param geom Geometry + final norm
        /// @param embedQ Raw quantized token-embedding matrix (vocab x hidden)
        /// @param embedType GGML type of the embedding matrix
        /// @param embedRowBytes Byte stride of one embedding row
        /// @param lmHeadQ Raw quantized separate LM-head matrix (rows==vocabSize,
        ///        cols==hiddenSize); NULL when the LM head is tied to the embeddings
        /// @param lmHeadType GGML type of the separate LM head
        /// @param lmHeadRowBytes Byte stride of one separate-LM-head row
        /// @param errMsg Error message on failure
        bool upload(const std::vector<DeviceLayer> &layers, const ModelGeometry &geom,
                    const void *embedQ, uint32_t embedType, uint32_t embedRowBytes,
                    const void *lmHeadQ, uint32_t lmHeadType, uint32_t lmHeadRowBytes,
                    std::string &errMsg);

        /// @brief Copy the current GPU-side hidden state (after the offloaded layer
        /// prefix) back to the host.  Used by the partial-offload path: the caller
        /// feeds this vector into the CPU Model::forward for the remaining layers.
        /// @param hiddenOut Host buffer (seqLen * hiddenSize floats)
        /// @param seqLen Number of tokens whose hidden state to copy
        bool copyHiddenOut(float *hiddenOut, uint32_t seqLen, std::string &errMsg);

        /// @brief Run one forward pass entirely on GPU.
        /// @param tokens Input token IDs
        /// @param computeAllLogits Compute logits for ALL tokens (prefill debug
        ///        path); otherwise only the last token's logits.
        /// @param logitsOut Host buffer for the result.  When computeAllLogits is
        ///        false: vocabSize floats.  Otherwise: seqLen*vocabSize floats.
        bool forward(const std::vector<int32_t> &tokens, bool computeAllLogits,
                     float *logitsOut, std::string &errMsg);

        /// @brief Zero the GPU KV cache and reset the cache position.
        void clearKVCache();

        /// @brief Number of layers actually offloaded to the GPU (as resolved
        /// in upload() from geom.numGpuLayers, clamped to numLayers).
        uint32_t numGpuLayers() const { return geom_.numGpuLayers; }

        /// @brief CPU-expert hybrid callback (llama.cpp `--cpu-moe` style,
        /// ROUTED-EXPERTS-ONLY variant).
        ///
        /// When set (qwen35moe only), upload() SKIPS the 256 per-expert
        /// matrices (ffnGateExps/ffnUpExps/ffnDownExpsMoe — ~32B of the 35B
        /// params, the VRAM hogs) and the driver, per layer:
        ///   1. D2Hs the post-attn norm (s.norm) to the double-buffered host
        ///      mirror (the ONLY D2H; s.norm is already produced by the GPU
        ///      RMSNorm kernel),
        ///   2. invokes this callback to run the ROUTER on the CPU (fp32
        ///      ffnGateInpMoe @ norm — reference math) + softmax + top-k +
        ///      renormalization + the 8 routed expert FFNs
        ///      (computeQwen35MoEFromLogits — the exact reference math, so
        ///      batch and single-token prefills select the SAME experts),
        ///   3. H2Ds the routed-expert output; the GPU then adds the SHARED
        ///      EXPERT (ffn*Shexp, uploaded, computed on device) and the
        ///      residual on the GPU.
        ///
        /// Why the CPU runs the ROUTER + softmax/top-k instead of the GPU:
        /// the GPU's fp16 prefill GEMM and its fp32 decode GEMV round the
        /// router logits slightly differently, which can FLIP a borderline
        /// expert in the top-8 and swap in a completely different expert —
        /// breaking batch-vs-sequential parity.  Running the router on the
        /// CPU with fp32 reference math makes the logits bit-identical for
        /// batch, single-token and the pure-CPU reference, so the routed
        /// selection is stable BY CONSTRUCTION.  (The reference
        /// double-softmax + partial_sort is deterministic.)  The GPU still
        /// runs the whole attention/GDN trunk, the shared expert and the
        /// residual — the router is a tiny 256x2048 mat-vec per token, ~1% of
        /// the layer FLOPs, so the CPU cost is negligible.
        ///
        /// @param layer GPU layer index [0, numGpuLayers)
        /// @param ffnNorm post-attn RMSNorm input (seqLen * hiddenSize floats, host)
        /// @param ffnOut routed-expert FFN output (seqLen * hiddenSize floats,
        ///        host; the shared-expert contribution is added on the GPU)
        /// @param seqLen number of tokens
        /// @return false aborts the forward with an error
        using MoeCpuFn = std::function<bool(uint32_t layer, const float *ffnNorm,
                                            float *ffnOut, uint32_t seqLen)>;
        void setMoeCpuFn(MoeCpuFn fn) { moeCpuFn_ = std::move(fn); }
        bool moeOnCpu() const { return static_cast<bool>(moeCpuFn_); }

        /// @brief Qwen35-specific forward (embed + offloaded layer prefix).
        /// Full offload (numGpuLayers == numLayers): also runs the final norm +
        /// LM head and writes logits into logitsOut (computeAllLogits semantics
        /// identical to forward()).  Partial offload: returns after the prefix
        /// and leaves the hidden state on the device for the caller to pick up
        /// with copyHiddenOut().
        bool forwardQwen35Prefix(const std::vector<int32_t> &tokens,
                                 bool computeAllLogits, float *logitsOut,
                                 std::string &errMsg);

        /// @brief Qwen35MoE-specific forward (embed + full offload).
        /// Identical semantics to forwardQwen35Prefix but the FFN is the routed
        /// MoE (softmax top-k experts + sigmoid-gated shared expert).  qwen35moe
        /// is full-offload only (the CPU has no partial-offload continuation for
        /// it), so this always runs the final norm + LM head.
        bool forwardQwen35MoePrefix(const std::vector<int32_t> &tokens,
                                    bool computeAllLogits, float *logitsOut,
                                    std::string &errMsg);

        /// @brief Free all device resources.
        ~GPUModel();

        GPUModel() = default;
        GPUModel(const GPUModel &) = delete;
        GPUModel &operator=(const GPUModel &) = delete;

        /// @brief Debug: run the CUDA Q8_K quantization + integer GEMV kernels
        /// on ONE weight row and one activation vector, returning the row result.
        /// Only used by the GPU vs CPU parity diagnostics (GPUCpuCompareTest).
        /// @param type GGML_TYPE_* (IQ2_S / IQ3_XXS / IQ3_S only)
        /// @param wRow Host pointer to ONE quantized weight row
        /// @param blocksPerRow Number of 256-element blocks in the row
        /// @param x Host activation vector (cols floats)
        /// @param cols Row width
        /// @return The Q8K GEMV row result (host float).
        float debugQ8KRow(uint32_t type, const uint8_t *wRow, uint32_t blocksPerRow,
                          const float *x, uint32_t cols);

    private:
        // Grow-on-demand device scratch, sized for the current seqLen.
        // NOTE: fp16 scratch buffers are stored as uint16_t* (bit-compatible with
        // CUDA __half) so this header stays valid in host-only translation units;
        // GPUCompute.cu casts them to __half*/__half2* at the use sites.
        struct DeviceScratch {
            int32_t *tokens = nullptr;
            float *hidden = nullptr;      // [seqLen][hidden]
            uint16_t *hiddenF16 = nullptr;// [seqLen][hidden] fp16 twin for prefill GEMMs
            // ---- Qwen35 recurrent-layer scratch ----
            // The dense projections (attnQKV/attnGate/ssmBetaQ/ssmAlphaQ +
            // ssm_out) are BATCHED over tokens ([seqLen][...]).  The conv1d +
            // gated-delta-net recurrence is inherently sequential, so the
            // conv/L2-norm/gdn kernels process ONE token at a time (per-token
            // pointers into the row-strided buffers) inside a host loop.
            float *qkv = nullptr;     // [seqLen][qkvDim] (2*keyDim + valueDim)
            float *q35z = nullptr;    // [seqLen][dInner]
            float *q35Beta = nullptr; // [seqLen][nVHeads]
            float *q35GateV = nullptr;// [seqLen][nVHeads]
            float *convIn = nullptr;  // [convKernel*convChannels] one token
            float *convOut = nullptr; // [convChannels] one token
            float *qN = nullptr;      // [headK*nVHeads] one token
            float *kN = nullptr;      // [headK*nVHeads] one token
            float *vN = nullptr;      // [headV*nVHeads] one token
            float *gdnOut = nullptr;  // [seqLen][headV*nVHeads] (batched ssm_out)
            float *q35Norm = nullptr; // [headV*nVHeads] one token
            // Qwen35 full-attention: fused Q+gate projection output
            // [seqLen][nHeads*2*headDim].  The gate half feeds kQ35SigmoidGate
            // and the Q half feeds the per-head Q RMSNorm + attention.
            float *q35QGate = nullptr;
            // Q8_K-quantized activation scratch for the decode (seqLen==1) GEMV
            // of the IQ2_S/IQ3_XXS/IQ3_S matrices.  llama.cpp's CUDA backend
            // quantizes the fp32 activation to Q8_K (vec_dot_type) and uses
            // integer vec-dots (vec_dot_iq*_q8_K); the exact integer
            // accumulation is numerically different from a float dequant-dot
            // (up to ~5%), which flips near-tie argmax at knife-edge positions.
            // Layout per block (stride kQ8K_STRIDE in GPUCompute.cu):
            //   int8 qs[256] + int16 bsums[16] + float d.
            uint8_t *q8k = nullptr;// [blocksPerRow * kQ8K_STRIDE] Q8_K activation
            // [seqLen][hidden] RMSNorm output.  The norm results must NOT be
            // written back into `hidden`: `hidden` is the residual stream the
            // attention/FFN blocks add into (the CPU path norms into separate
            // attnNorm/ffnNorm buffers). Overwriting it with the normed vector
            // destroys the residual connections and produces garbage.
            float *norm = nullptr;
            float *q = nullptr;            // [seqLen][qHeads*headDim]
            float *k = nullptr;            // [seqLen][kHeads*headDim]
            float *v = nullptr;            // [seqLen][kHeads*headDim]
            float *attnOut = nullptr;      // [seqLen][qHeads*headDim]
            uint16_t *attnOutF16 = nullptr;// [seqLen][qHeads*headDim] fp16 twin
            float *attnProj = nullptr;     // [seqLen][hidden]
            float *gate = nullptr;         // [seqLen][intermediate]
            uint16_t *gateF16 = nullptr;   // [seqLen][intermediate] fp16 twin
            float *up = nullptr;           // [seqLen][intermediate]
            float *ffnOut = nullptr;       // [seqLen][hidden]
            // ---- Qwen35MoE (qwen35moe) MoE FFN scratch ----
            // DOUBLE-BUFFERED DEVICE SNAPSHOT of s.norm for the CPU-expert
            // hybrid handoff.  The CPU callback must read the post-attn norm of
            // layer L, but s.norm is REUSED (overwritten) by layer L+1's
            // RMSNorm while the D2H copy is still in flight.  D2H-ing s.norm
            // directly would race: batch prefills (237 KB copy at seqLen=29)
            // lose the race almost every layer -> garbage router/expert input.
            // Instead g_stream copies s.norm -> moeNormSnap[buf] BEFORE any
            // L+1 work (ordered on g_stream), then g_stream2 D2Hs THAT buffer
            // (which g_stream never touches again).  One per double-buffer slot.
            float *moeNormSnap[2] = {nullptr, nullptr};
            float *moeLogits = nullptr;// [seqLen][expertCount] router logits
            int32_t *moeIdx = nullptr; // [seqLen][expertUsed] top-k expert ids
            float *moeWgt = nullptr;   // [seqLen][expertUsed] renormalized weights
            float *moeWsum = nullptr;  // [seqLen] clamped top-k prob sum
            // Per-expert batch outputs during the routed FFN.  expertFF is the
            // per-expert feed-forward width; sharedFF the shared-expert width.
            // Buffer of rank r (r in [0, expertUsed)) holds the CURRENT token's
            // SwiGLU'd gate/up product (expertFF) and the expert's down output
            // (hidden).  They are reused across ranks to keep VRAM flat.
            float *moeGateUp = nullptr;   // [seqLen][expertFF] gate half
            float *moeGateUp2 = nullptr;  // [seqLen][expertFF] up half
            float *moeDown = nullptr;     // [seqLen][hidden]   one expert at a time
            float *moeExpertOut = nullptr;// [seqLen][hidden]   weighted accumulation
            // Shared-expert buffers: gate sigmoid, SwiGLU product, down output.
            float *moeShexpGate = nullptr;   // [seqLen] sigmoid gate
            float *moeShexpGateUp = nullptr; // [seqLen][sharedFF] SwiGLU gate half
            float *moeShexpGateUp2 = nullptr;// [seqLen][sharedFF] SwiGLU up half
            float *moeShexpDown = nullptr;   // [seqLen][hidden]   shared-expert down
            float *logits = nullptr;         // [seqLen*vocab] (max of the two)
            uint32_t seqCap = 0;
        };

        bool allocated_ = false;
        DeviceLayer *layers_ = nullptr;// device-backed descriptors
        float *kvK_ = nullptr;         // [nGpuLayers][maxSeqLen*kHeads*headDim]
        float *kvV_ = nullptr;
        // Qwen35 recurrent-state buffers: per GPU layer, conv state
        // [(convKernel-1) * convChannels] and gdn state [nVHeads*headV*headV]
        // (fp32; persistent across tokens, reset on clearKVCache).
        float *q35ConvState_ = nullptr;
        float *q35GdnState_ = nullptr;
        // MRoPE cache [maxSeqLen][ropeDimensionCount/2] (qwen35 only; uploaded
        // from the CPU-side table so the GPU kernels need not recompute cos/sin).
        float *mropeCos_ = nullptr;
        float *mropeSin_ = nullptr;
        float *finalNorm_ = nullptr;// device [hidden]
        float *embedQ_ = nullptr;   // raw quantized token-embedding bytes
        float *lmHeadQ_ = nullptr;  // raw quantized separate LM head (nullable)
        float *ropeCos_ = nullptr;  // [maxSeqLen][headDim/2]
        float *ropeSin_ = nullptr;
        // Reusable FP16 dequant scratch for the prefill GEMM path.  Sized to the
        // LARGEST single weight matrix (one layer's largest matrix, streamed
        // per-layer), so 7B+ models fit in VRAM without ~14 GB of persistent
        // FP16 twins.
        void *wF16_ = nullptr;
        uint64_t wF16Bytes_ = 0;
        const void *wF16SrcQ_ = nullptr;// matrix source whose dequant is in wF16_
        // Reusable Q8_K activation scratch for the decode GEMV of the IQ2_S /
        // IQ3_XXS / IQ3_S matrices (see the scratch_.q8k comment for the block
        // layout and stride).  Grown on demand to the largest blocksPerRow.
        uint8_t *q8k_ = nullptr;
        uint64_t q8kBytes_ = 0;
        ModelGeometry geom_{};
        DeviceScratch scratch_{};
        uint32_t embedType_ = 0;
        uint32_t embedRowBytes_ = 0;
        uint32_t lmHeadType_ = 0;
        uint32_t lmHeadRowBytes_ = 0;
        // Host mirror of the MoE router's top-k expert selection (qwen35moe).
        // The per-expert GEMVs need the SELECTED expert ids per token; the GPU
        // router kernel writes moeIdx on the device, so the driver copies the
        // tiny [seqLen*expertUsed] int32 block back each layer (expertUsed <= 8
        // => <= 32 ints for a typical prefill).  Reused across layers (grows on
        // demand, never shrinks).
        std::vector<int32_t> moeIdxHost_;
        // NO LONGER USED by the hybrid: since the router runs on the CPU (fp32
        // ffnGateInpMoe @ norm, reference math), no raw logits cross the PCIe
        // bus.  This mirror is retired; only moeIdxHost_/moeWgtHost_ remain for
        // the FULL-GPU path's top-k selection D2H.
        std::vector<float> moeLogitsHost_;
        // Host mirror of the per-(token, rank) renormalized routing weights
        // (qwen35moe).  The per-token MoE fallback needs the token's OWN
        // weight for the kAddScaled1 scaling; the same cudaMemcpyAsync that
        // fills moeIdxHost_ copies these bytes so the fallback never reads a
        // device pointer on the host (that would be undefined behavior).
        std::vector<float> moeWgtHost_;
        // CPU-expert hybrid (qwen35moe, llama.cpp `--cpu-moe` style): DOUBLE-
        // BUFFERED host mirrors of the per-layer post-attn RMSNorm input and
        // routed-expert FFN output used for the GPU<->CPU handoff when
        // moeCpuFn_ is set.  Ping-pong: while the CPU computes layer L's
        // experts into buffer A, the GPU's g_stream2 copy engine streams layer
        // L+1's norm into buffer B, so the per-layer handoff needs no hard
        // stream sync beyond the ping-pong event pair.  Grown on demand, reused
        // across layers, freed via the normal member lifetime.
        std::vector<std::vector<float>> moeNormHost_;  // [2][seqLen*H]
        std::vector<std::vector<float>> moeFfnOutHost_;// [2][seqLen*H]
        // Handoff event RINGS (created lazily on first hybrid forward), indexed
        // by layer modulo the ring size.  A ring (NOT a re-used ping-pong pair)
        // is REQUIRED: cudaEventRecord re-targets an event, so re-recording
        // `eNormReady[buf]` at layer L+2 while layer L's cudaStreamWaitEvent is
        // still pending on g_stream2 would silently re-point that wait at the
        // L+2 record — and by then s.norm was overwritten by layers L+1/L+2,
        // feeding the CPU callback garbage (non-deterministic batch output).
        // Ring size 16 covers the worst plausible in-flight depth (g_stream2 is
        // only as fast as its µs-scale copies while g_stream hammers 40 layers
        // of attention): a wait for index k is always enqueued before any
        // record of k + 16.
        // eNormReady[k]: g_stream finished the post-attn RMSNorm (s.norm) of
        //   layer k -> g_stream2 waits on it before its single D2H (norm).
        // eCpuDone[k]: CPU finished layer k's experts; g_stream2 recorded it
        //   AFTER the routed H2D -> g_stream waits on it before kAddScaled.
        static constexpr uint32_t kMoeEventRing = 16;
        cudaEvent_t eNormReady[kMoeEventRing] = {};
        cudaEvent_t eCpuDone[kMoeEventRing] = {};
        // Double-buffer flip state (0/1) for the norm/out host mirrors.
        uint32_t moeBufFlip_ = 0;
        // Optional CPU-expert callback (see setMoeCpuFn).  Null = full-GPU MoE
        // (all expert matrices uploaded).
        MoeCpuFn moeCpuFn_;
        size_t kvPos_ = 0;
        bool scratchAlloc_ = false;

        bool ensureScratch(uint32_t seqLen, std::string &errMsg);
        void destroyScratch();
        void destroy();
        /// @brief Upload the MRoPE (multi-section RoPE) cos/sin cache used by the
        /// qwen35 full-attention layers. Mirrors Model::applyMRoPE's theta_scale.
        bool uploadMRoPECache(std::string &errMsg);
        /// @brief Dequantize ONE weight matrix (m.q, quantized blocks) into the
        /// reusable per-layer FP16 scratch (wF16_), growing it as needed.
        /// @param m Device-layout matrix descriptor
        /// @param outBuf Receives the (cached) fp16 scratch pointer.
        bool dequantMatrixF16(const DeviceMatrix &m, void **outBuf,
                              std::string &errMsg);
    };

    /// @brief Global GPU enable flag (default ON in CUDA builds; $TINYCODER_GPU=0
    /// disables the offload engine, forcing the CPU path).
    bool gpuEnabled();

}// namespace tinycoder::gpu

#endif// USE_CUDA
