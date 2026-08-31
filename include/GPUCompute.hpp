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

        /// @brief CPU-expert hybrid callback (qwen35moe, llama.cpp `--cpu-moe`
        /// style).  The adapter registers ONE callback; the GPU driver
        /// dispatches on expertCacheBuilt():
        ///   * cache built -> the callback runs the fp32 reference ROUTER
        ///     (ffnGateInpMoe @ ffnNorm + softmax + top-k + renormalize) into
        ///     the public host mirrors moeRouterIdxHost_/moeRouterWgtHost_;
        ///     the per-expert FFNs run on the GPU through the per-layer expert
        ///     cache (buildExpertCache/ensureExpertCached), so the host does
        ///     ~µs of router work and the GPU does the entire expert FFN at
        ///     VRAM bandwidth (no more ~32 MB/layer host-RAM streaming).
        ///   * cache NOT built (device allocation failure fallback) -> the
        ///     callback runs the FULL routed expert FFNs on the CPU into
        ///     ffnOut (the old --cpu-moe path).
        /// The CPU router is what keeps batch == single-token expert selection
        /// bit-identical: the fp32 reference math (ffnGateInpMoe @ norm) is
        /// identical for batch, decode and the pure-CPU path, so a borderline
        /// expert cannot flip between prefills.
        /// @param layer GPU layer index [0, numGpuLayers)
        /// @param ffnNorm post-attn RMSNorm input (seqLen * hiddenSize floats,
        ///        host)
        /// @param ffnOut routed-expert FFN output (seqLen * hiddenSize floats,
        ///        host; UNUSED when the expert cache is built)
        /// @param seqLen number of tokens
        /// @return false aborts the forward with an error
        using MoeCpuFn = std::function<bool(uint32_t layer, const float *ffnNorm,
                                            float *ffnOut, uint32_t seqLen)>;
        void setMoeCpuFn(MoeCpuFn fn) { moeCpuFn_ = std::move(fn); }
        bool moeOnCpu() const { return static_cast<bool>(moeCpuFn_); }

        /// @brief True once the per-layer expert cache arenas exist (hybrid
        /// mode).  The driver uses this to pick the CPU-router + GPU-expert
        /// path over the full-CPU-expert fallback, and the adapter's callback
        /// uses it to decide which computation to run.
        bool expertCacheBuilt() const { return expertCacheBuilt_; }

        /// @brief Host mirror of the CPU router selection (hybrid + expert
        /// cache): per-(token, rank) expert ids + renormalized weights, written
        /// by the adapter's router-only callback and read back by the GPU
        /// driver to slice the cached expert matrices.  Grown on demand;
        /// indexed [seqLen * expertUsed].
        std::vector<int32_t> moeRouterIdxHost_;
        std::vector<float> moeRouterWgtHost_;

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

        /// @brief Build the per-layer expert LRU cache device arenas.
        ///
        /// Hybrid mode (moeCpuFn_ set) keeps the 256 per-expert matrices in host
        /// RAM and never calls uploadMat on them, so layers_[L].ffnGateExps.q
        /// stays NULL.  This method allocates the direct-mapped cache arenas
        /// (kExpertCacheWays experts per layer) and records the host slice
        /// geometry needed to H2D a single expert on a miss.  Called once from
        /// upload() AFTER the hybrid retry sets moeCpuFn_ — the cache is only
        /// used when moeCpuFn_ is set (full-GPU path uses the real uploads).
        /// @return false on a device allocation failure (caller stops decode).
        bool buildExpertCache(std::string &errMsg);

        /// @brief 2-way set-associative cache SET index for expert `e` (private
        /// helper).  Knuth multiply-high hash over the pow2 SET count (default
        /// 16 sets x 2 ways = 32 slots = same footprint as the old direct map,
        /// but a colliding hot pair coexists in the two ways).
        uint32_t expertCacheSet(uint32_t e);

        /// @brief Ensure expert `e` of layer `L` is resident in the device cache.
        /// H2Ds the WHOLE CONTIGUOUS slot on a miss (set-associative tag/LRU
        /// lookup, one pinned DMA per miss).  Used by the hybrid expert path
        /// before the per-expert GEMVs.
        /// @param downQOut [out] receives the slot's down-slice device base
        ///        (inside the contiguous [gate][up][down] slot range).
        /// @return gate-slice device base (up starts gateSlice bytes later) or
        ///         nullptr on failure.
        const uint8_t *ensureExpertCached(uint32_t L, uint32_t e,
                                          std::string &errMsg,
                                          const uint8_t **downQOut = nullptr);

        /// @brief Free all device resources.
        ~GPUModel();

        GPUModel() = default;
        GPUModel(const GPUModel &) = delete;
        GPUModel &operator=(const GPUModel &) = delete;

        /// @brief Debug: run the CUDA Q8_K quantization + integer GEMV kernels
        /// on ONE weight row and one activation vector, returning the row result.
        /// Only used by the GPU vs CPU parity diagnostics (GPUCpuCompareTest).
        /// @param type GGML_TYPE_* (IQ2_S / IQ3_XXS / IQ3_S, plus Q4_K and Q6_K
        ///        which dispatch to kQGemvKxQ8K)
        /// @param wRow Host pointer to ONE quantized weight row
        /// @param blocksPerRow Number of 256-element blocks in the row
        /// @param x Host activation vector (cols floats)
        /// @param cols Row width
        /// @return The Q8K GEMV row result (host float).
        float debugQ8KRow(uint32_t type, const uint8_t *wRow, uint32_t blocksPerRow,
                          const float *x, uint32_t cols);

        /// @brief Debug: run the CUDA Q8_0 integer GEMV kernels
        /// (kQuantizeQ8_0x32 + kQGemvQ8_0xQ8K) on ONE weight row and one
        /// activation vector, returning the row result.  Used by unit tests to
        /// verify the GPU Q8_0 integer path reproduces the CPU
        /// matMulVecBatchQ8_0_Q8K_AVX2 reference bit-for-bit (the qwen35moe
        /// trunk projections and experts all route through this exact path).
        /// @param wRow Host pointer to ONE quantized Q8_0 weight row
        /// @param blocksPerRow Number of 32-element blocks in the row
        /// @param x Host activation vector (cols floats)
        /// @param cols Row width
        /// @return The Q8_0 GEMV row result (host float).
        float debugQ8_0Row(const uint8_t *wRow, uint32_t blocksPerRow,
                           const float *x, uint32_t cols);

        /// @brief Debug: run the CUDA 32-wide legacy-block GEMV
        /// (kQGemvSmall32<Q5_0/Q8_0/IQ4_NL>) on ONE weight row and one
        /// activation vector, returning the scalar row result.  Used by unit
        /// tests to verify the GPU float dequant-dot path for the 32-wide
        /// legacy quant types reproduces the CPU reference.
        /// @param type GGML_TYPE_* (Q5_0 / Q8_0 / IQ4_NL)
        /// @param wRow Host pointer to ONE quantized weight row
        /// @param blocksPerRow Number of 32-element blocks in the row
        /// @param x Host activation vector (cols floats)
        /// @param cols Row width
        /// @return The GEMV row result (host float).
        float debugQGemvRow(uint32_t type, const uint8_t *wRow,
                            uint32_t blocksPerRow, const float *x,
                            uint32_t cols, uint32_t rowBytes);

        /// @brief Debug: quantize x with kQuantizeQ8_0x32 and emit the
        /// per-block float contributions kQGemvQ8_0xQ8K would accumulate.
        /// Fills outBlk[0..blocksPerRow) (host) and outQuant[0..
        /// blocksPerRow*kQ8_0x32_STRIDE) (host, raw quantized block bytes).
        /// Used to bisect the GPU Q8_0 integer kernel against the CPU AVX2
        /// reference block-by-block.
        /// If outTrace is non-null (size >= 2*blocksPerRow+8), fills it with
        /// the kQGemvQ8_0xQ8KTrace output: outTrace[0..blocksPerRow) are the
        /// kernel's per-block cumulative sums, outTrace[blocksPerRow + g] the
        /// block-0 t[g] intermediates, outTrace[blocksPerRow+8] block-0 blk.
        void debugQ8_0Blocks(const uint8_t *wRow, uint32_t blocksPerRow,
                             const float *x, uint32_t cols,
                             float *outBlk, uint8_t *outQuant,
                             float *outTrace);

        /// @brief Debug: quantize x with kQuantizeQ8K (256-wide Q8_K blocks,
        /// kQ8K_STRIDE layout) and emit the per-block float contributions the
        /// Q4_K/Q6_K GEMVs would accumulate.
        /// Fills outQuant[0..blocksPerRow*kQ8K_STRIDE) (host, raw quantized
        /// block bytes) and outBlk[0..blocksPerRow) with a host-side recompute
        /// of the per-block contribution using the EXACT kQGemvKxQ8K float
        /// tree on those bytes (Q4_K: fma chain; Q6_K: per-lane offset +
        /// chunk FMAs, horizontal sum).  Used by the Q4K/Q6K kernel parity
        /// unit test to localize whether a divergence is in the activation
        /// quantizer (outQuant vs CPU quantizeQ8K) or the GEMV float tree
        /// (outBlk vs the CPU kernel's per-block tree on the same bytes).
        /// @param type GGML_TYPE_* (Q4_K or Q6_K only)
        void debugQ8KBlocks(uint32_t type, const uint8_t *wRow,
                            uint32_t blocksPerRow, const float *x,
                            uint32_t cols, float *outBlk, uint8_t *outQuant);

        // ---- Static resident-expert policy (Option D, PUBLIC) ----
        // The adapter (ModelGPU.cpp) computes the top-K most-used experts per
        // layer during the profiling phase and stores the policy HERE before
        // calling buildResidentArenas().  The GPU driver then:
        //   * buildResidentArenas(): allocates the per-layer arenas holding the
        //     resident experts' packed gate/up/down slices (permanent, no LRU);
        //   * runs the resident ranks on the GPU from those arenas;
        //   * the adapter's MoeCpuFn computes the ROUTER + the NON-resident
        //     ranks on the CPU into moeDownTableHost_, which the driver H2Ds as
        //     moeDownTable_ and accumulates rank-order (bit-exact reference).
        static constexpr uint32_t kResidentLayers = 128;
        // Max resident experts per layer (upper bound for the static policy;
        // the runtime budget is set by the adapter from free VRAM).
        static constexpr uint32_t kResidentMaxExperts = 64;
        // Per-layer resident policy: [layer] = vector of expert ids, sorted by
        // usage descending (the first residentMaxPerLayer_ are resident).
        std::vector<std::vector<uint32_t>> moeResidentIds_;
        // Per-layer resident bitmask (expert id -> bit), used by the adapter's
        // CPU callback (computeQwen35MoENonResident): [kResidentLayers] vectors
        // of uint64 words (maskWords = ceil(expertCount/64)).
        std::vector<std::vector<uint64_t>> residentMaskHost_;
        // Resolved per-layer resident count (0 = none).
        std::vector<uint32_t> moeResidentCount_;
        // Double-buffered HOST down table (rank-major [r*seqLen*H]) the CPU
        // callback fills for NON-resident ranks; H2D'd once per layer as the
        // device down table moeDownTable_.
        std::vector<std::vector<float>> moeDownTableHost_;
        // Store the resident policy (adapter-side) + mark Option D active.
        void setResidentPolicy(std::vector<std::vector<uint32_t>> ids) {
            moeResidentIds_ = std::move(ids);
        }
        // Allocate the per-layer arenas + dispatch tables from the stored
        // policy + recorded hybrid geometry, then H2D the resident experts'
        // packed gate/up/down slices ONCE (public: the adapter calls this from
        // buildResidentExpertPolicy after setResidentPolicy).
        bool buildResidentArenas(std::string &errMsg);
        // The adapter's callback reads the host norm/weight mirrors.
        std::vector<float> moeWgtHost_;
        // Set by the adapter at build time (whether to run Option D dispatch).
        bool residentEnabled_ = false;
        // Set by the driver right before invoking the MoE callback: the flip
        // slot (0/1) of moeDownTableHost_ the CPU callback must fill with the
        // NON-resident rank-major down table (Option D only; -1 = disabled).
        int32_t residentHostSlot_ = -1;
        // Per-resident-expert arena bytes from the recorded hybrid geometry
        // (gate/up [expertFF][H], down [H][expertFF]); 0 = geometry not ready.
        // The adapter uses this to budget free VRAM across the arenas.
        uint64_t residentPerExpertBytes() const {
            if (expertFF_ == 0 || expertHM_ == 0 || expertRowBytesGate_ == 0 ||
                expertRowBytesDown_ == 0)
                return 0;
            return 2ull * static_cast<uint64_t>(expertFF_) * expertRowBytesGate_ +
                   static_cast<uint64_t>(expertHM_) * expertRowBytesDown_;
        }

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
            // Slab r (r in [0, expertUsed)) of the BATCHED buffers holds rank
            // r's gate/up product (expertFF) and down output (hidden).  The
            // distinct-slab layout (2026-09-14, decode fast path) breaks the
            // WAR-serialized rank loop: each rank's GEMVs write their OWN slab,
            // so all ranks' gate/up GEMVs run CONCURRENTLY (same q8kN_ act),
            // then all ranks' down GEMVs, then one batched accumulation.
            // The single-expert variants below remain for the prefill/fallback.
            float *moeGateUp = nullptr;   // [seqLen][expertFF] single expert
            float *moeGateUp2 = nullptr;  // [seqLen][expertFF] single expert
            float *moeDown = nullptr;     // [seqLen][hidden]   single expert
            float *moeExpertOut = nullptr;// [seqLen][hidden]   weighted accum
            float *moeGateUpP = nullptr;  // [expertUsed][expertFF] all ranks
            float *moeGateUp2P = nullptr; // [expertUsed][expertFF] all ranks
            float *moeDownP = nullptr;    // [expertUsed][hidden]   all ranks
            // Shared-expert buffers: gate sigmoid, SwiGLU product, down output.
            float *moeShexpGate = nullptr;   // [seqLen] sigmoid gate
            float *moeShexpGateUp = nullptr; // [seqLen][sharedFF] SwiGLU gate half
            float *moeShexpGateUp2 = nullptr;// [seqLen][sharedFF] SwiGLU up half
            float *moeShexpDown = nullptr;   // [seqLen][hidden]   shared-expert down
            float *logits = nullptr;         // [seqLen*vocab] (max of the two)
            uint32_t seqCap = 0;
        };

        // ---- Qwen35MoE (qwen35moe) GPU expert cache ----
        // When the 256-expert matrices cannot all fit in VRAM (hybrid mode,
        // llama.cpp `--cpu-moe` style), ONLY the router + shared expert are
        // uploaded.  The routed expert GEMVs are the decode hot path and MUST
        // NOT stream ~32 MB/layer from host RAM every token (that pages on
        // models larger than the page cache and leaves the GPU idle).  This
        // cache holds the RECENTLY-USED experts on the device so steady-state
        // decode hits VRAM bandwidth: direct-mapped by (layer, e % ways), each
        // slot stores the 3 expert slices (ffnGateExps/ffnUpExps rows
        // [e*expertFF, (e+1)*expertFF), ffnDownExpsMoe rows [e*H, (e+1)*H)).
        // Misses upload the ~4 MB slices synchronously (once per expert); with
        // stable routing the hit rate saturates after the first token.
        struct ExpertSlot {
            // SLOT-MAJOR arena layout (2026-09-14): slot s owns the CONTIGUOUS
            // range [s*slotBytes, (s+1)*slotBytes) = [gate][up][down] (slotBytes
            // = 2*gateSlice + downSlice).  gateQ is the arena base; upQ/downQ
            // are slot-0 interior pointers kept for clarity (per-slot offsets
            // are re-derived in ensureExpertCached).  The contiguous layout makes
            // a cache miss ONE pinned-memory DMA copy instead of three scattered
            // pageable bounces (was ~2.4 GB/s effective; see moe block profile).
            uint8_t *gateQ = nullptr;// arena base; slot s gate at +s*slotBytes
            uint8_t *upQ = nullptr;  // slot-0 interior: arena + gateSlice
            uint8_t *downQ = nullptr;// slot-0 interior: arena + 2*gateSlice
            // Per-layer HOST pointers to this layer's FULL packed expert
            // matrices (hybrid mode keeps them off-device; the per-slot cache
            // H2Ds slices from THESE).  Every layer has its OWN
            // ffnGateExps/ffnUpExps/ffnDownExpsMoe host blob, so the H2D
            // source MUST be selected per layer: a single global
            // expertHostGate_ would feed EVERY layer's cache with the last
            // uploaded layer's weights (the root cause of the L0 routed-expert
            // divergence — L0 executed layer-39 expert bytes while the router/
            // shared expert/trunk used layer-0 data).
            const uint8_t *hostGate = nullptr;
            const uint8_t *hostUp = nullptr;
            const uint8_t *hostDown = nullptr;
            // ---- Per-layer quant geometry (2026-09-17) ----
            // The GLOBALS below (expertType_ / expertRowBytesGate_ /
            // expertDownType_ / expertRowBytesDown_) are overwritten every
            // upload iteration and hold only the LAST layer's values.  The
            // Ornith Q4_K_M mix has PER-LAYER down-expert types (layers 0-4/7+
            // down=Q6_K 210 B/block, layers 5-6 down=Q4_K 144 B/block), so a
            // fixed global down stride feeds the wrong bytes into the fills /
            // GEMVs for the divergent layers -> NaN logits.  Record the real
            // per-layer type + rowBytes here and let every cache consumer
            // derive its geometry from THIS slot.
            uint32_t gateType = 0;    // expertType_ equivalent for this layer
            uint32_t gateRowBytes = 0;// expertRowBytesGate_ for this layer
            uint32_t downType = 0;    // expertDownType_ for this layer
            uint32_t downRowBytes = 0;// expertRowBytesDown_ for this layer
        };
        // 2-WAY SET-ASSOCIATIVE expert cache (2026-09-14) per layer:
        // kExpertCacheSets sets x kExpertCacheWays ways = 64 slots max — the
        // DEFAULT 16 sets x 2 ways = 32 slots is the SAME footprint as the old
        // 32-way direct map (~4.1 GB for 40 layers), but with a way-count of 2
        // so two experts that hash to the same set can COEXIST.  The direct
        // map thrashed: per token each layer routes ~8 experts, and ~2 of them
        // keep landing in the same slot -> the colliding pair evicted each
        // other EVERY token (~3.5 misses / layer / token sustained = the ~500
        // MB H2D / token / 225 ms PCIe floor).  A 2-way set lets a
        // repeatedly-colliding hot pair stay resident; the per-set LRU bit
        // evicts only the least-recently-used way on a genuine 3+ expert
        // conflict.
        static constexpr uint32_t kExpertCacheLayers = 128;// layers (40)
        static constexpr uint32_t kExpertCacheSets = 32;   // sets per layer (max)
        static constexpr uint32_t kExpertCacheWays = 2;    // ways per set
        static constexpr uint32_t kExpertCacheSlots =
                kExpertCacheSets * kExpertCacheWays;// 64 per layer
        // Runtime set count (env TINYCODER_MOE_WAYS = SETS, default 16; slots =
        // sets*2).  An A/B tool: more sets shrink conflicts (more VRAM), fewer
        // sets shrink VRAM (more conflicts).  16 sets = 32 slots = ~4.1 GB for
        // 40 layers (default footprint); 24 sets = 48 slots ≈ 6.1 GB; 32 sets
        // = 64 slots ≈ 8.2 GB (fits the 11 GB card after the hybrid retry
        // FREED the per-expert weight upload, so the expert cache can now
        // borrow that VRAM).
        static constexpr uint32_t kExpertCacheSetsDefault = 16;
        uint32_t expertCacheSets_ = kExpertCacheSetsDefault;
        ExpertSlot expertCache_[kExpertCacheLayers];
        int32_t expertCacheTag_[kExpertCacheLayers][kExpertCacheSets]
                               [kExpertCacheWays];
        // Per-set LRU bit: 0 -> way0 is LRU, 1 -> way1 is LRU.  On a 2-way
        // conflict the LRU way is evicted and the bit flips (so the surviving
        // way becomes LRU next — the natural 2-way pseudo-LRU).
        uint8_t expertCacheLru_[kExpertCacheLayers][kExpertCacheSets] = {};
        uint64_t expertCacheLayerBytes_ = 0;// bytes per layer arena (0 = not built)
        // Pinned H2D staging ring (2026-09-14): a miss H2Ds ONE CONTIGUOUS
        // 3.3 MB slot.  The host pages of the packed expert matrices are
        // pageable, and a pageable cudaMemcpyAsync H2D quietly BLOCKS the host
        // AND runs at ~2.4 GB/s (driver-staged); pinning a ring of staging
        // buffers turns each miss into a fast host memcpy (fill) + a truly
        // async single DMA copy, keeping the host AHEAD of the copy engine so
        // PCIe 3.0 x16 stays saturated (~6+ GB/s) across the whole layer
        // sequence.  Ring depth 8 lets the host run ~8 misses (~26 MB) ahead.
        static constexpr uint32_t kExpertStageBufs = 8;
        static constexpr uint32_t kExpertStageLayers = 128;
        uint8_t *expertStage_[kExpertStageBufs] = {nullptr, nullptr};
        uint64_t expertStageBytes_ = 0;
        // Device-elapsed for the [moe fwd] stats line (TINYCODER_MOE_STATS=1):
        // eventStart_ records on g_stream right after the token H2D, eventEnd_
        // right before the final sync; cudaEventElapsedTime then splits the
        // wall total into host-side enqueue/fill time vs GPU device time.
        // Created lazily on the first stats forward (NULL until then).
        cudaEvent_t moeEvStart_ = nullptr;
        cudaEvent_t moeEvEnd_ = nullptr;
        // Per-buffer "previous DMA finished" event (lazily created): the host
        // must not refill stage[b] while stage[b]'s previous cudaMemcpyAsync
        // is still being drained by the copy engine (WAR on the pinned bytes).
        // cudaEventSynchronize before the fill — ~costless when the engine is
        // already past the event (the common case: the host is ahead).
        cudaEvent_t expertStageEv_[kExpertStageBufs] = {};
        // Ring index for the NEXT miss's staging buffer (round-robin, NOT the
        // old double-buffer parity — 8 buffers survive deep per-layer pipelines).
        uint32_t expertStageFlag_ = 0;
        // Host-side row geometry of the packed expert matrices (from the
        // layer's DeviceMatrix at upload time; used to slice rows).
        uint32_t expertFF_ = 0;
        uint32_t expertCount_ = 0;
        uint32_t expertUsed_ = 0;
        uint32_t expertHM_ = 0;// hidden size
        // Host pointers of the FULL packed expert matrices (hybrid mode keeps
        // them off-device; the cache H2Ds slices from these).
        const uint8_t *expertHostGate_ = nullptr;
        const uint8_t *expertHostUp_ = nullptr;
        const uint8_t *expertHostDown_ = nullptr;
        uint32_t expertRowBytesGate_ = 0;
        uint32_t expertRowBytesDown_ = 0;
        uint32_t expertType_ = 0;// Q8_0 for Ornith; recorded from ffnGateExps
        // Quant type of the DOWN expert matrix (ffnDownExpsMoe).  For the
        // Q4_K_M Ornith it is Q6_K (type 14) while the gate/up are Q4_K
        // (type 12); the resident fast path needs BOTH so each GEMV decodes
        // the correct block layout (Q4K 144 B/block vs Q6K 210 B/block).
        uint32_t expertDownType_ = 0;
        bool expertCacheBuilt_ = false;

        // ---- Static resident-expert arenas (Option D) ----
        // Public-API members of the resident split are declared ABOVE (the
        // public policy block): moeResidentIds_ / residentMaskHost_ /
        // moeResidentCount_ / moeDownTableHost_ / setResidentPolicy /
        // moeWgtHost_ / residentEnabled_.  The PRIVATE members here own the
        // DEVICE-side arenas + dispatch tables and the internal helpers.
        // Per-layer arena: contiguous [resident][gate][up][down] slices.
        uint8_t *residentArena_[kResidentLayers] = {};
        uint64_t residentArenaBytes_[kResidentLayers] = {};
        // Host row geometry of the resident slices (same as expert cache, but
        // for the fixed resident set).
        uint32_t residentRowBytesGate_ = 0;
        uint32_t residentRowBytesDown_ = 0;
        // Per-(layer, resident slot) device row bases into residentArena_[L]
        // (gate base [L][i], up base [L][i], down base [L][i]).
        const uint8_t *residentGateBase_[kResidentLayers][kResidentMaxExperts] = {};
        const uint8_t *residentUpBase_[kResidentLayers][kResidentMaxExperts] = {};
        const uint8_t *residentDownBase_[kResidentLayers][kResidentMaxExperts] = {};
        // Map EXPERT ID -> resident slot (per layer); -1 = not resident.
        // Indexed by expert id (0..255) — expert ids exceed kResidentMaxExperts,
        // so this must be sized by the expert count, not the resident budget.
        int32_t residentIdxMap_[kResidentLayers][256];
        uint32_t residentMaxPerLayer_ = 0;
        // Device down table (rank-major [r*seqLen*H]); H2D'd once per layer
        // from moeDownTableHost_ (the CPU callback's non-resident output).
        float *moeDownTable_ = nullptr;
        uint64_t moeDownTableBytes_ = 0;
        // Device pointer tables for the batched GU/Down launches per layer
        // (rebuilt per layer from the resident slices).
        void **residentRankWq_ = nullptr;
        uint64_t residentRankWqBytes_ = 0;
        // Device flag per FULL rank (1 = resident, 0 = skip) consumed by
        // kQGemvQ8_0xQ8K_BatchDownResident: non-resident ranks must NOT
        // overwrite the CPU-filled slots of the combined down table.
        int32_t *residentRankOf_ = nullptr;
        uint64_t residentRankOfBytes_ = 0;
        // Device mirror of the per-rank weights for resident ranks only
        // (write the full router table; the kernel indexes resident ranks).
        float *residentWgt_ = nullptr;
        uint64_t residentWgtBytes_ = 0;
        // True when at least one layer has resident experts (Option D active).
        bool residentActive_ = false;
        // Run the resident experts of layer L (decode fast path: seqLen==1).
        void scheduleResidentExperts(uint32_t L, float *downOut);
        // Run the resident experts of layer L (prefill path: seqLen>1).
        void scheduleResidentExpertsBatch(uint32_t L, float *downOut,
                                          uint32_t seqLen);

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
        // IQ3_XXS / IQ3_S matrices AND the Q8_0 x Q8_32 integer GEMV
        // (kQuantizeQ8_0x32/kQGemvQ8_0xQ8K — bit-exact with the CPU's
        // matMulVecBatchQ8_0_Q8K_AVX2; the Q8_0 kernel reuses the same
        // caller-passed scratch, sized blocksPerRow * kQ8K_STRIDE which amply
        // covers its 64 B/block (kQ8_0x32_STRIDE) layout).  Grown on demand to
        // the largest blocksPerRow.
        uint8_t *q8k_ = nullptr;
        uint64_t q8kBytes_ = 0;
        // Dedicated Q8_32-quantized activation scratch for the DECODE MoE
        // expert fast path: holds Q8(s.norm) for the WHOLE per-layer rank loop
        // (all 8 ranks' gate/up pre-GEMVs read it).  The general q8k_ cannot
        // be used for this: the per-rank down path RE-quantizes s.moeGateUp
        // into q8k_, which would clobber the norm's quantized copy while ranks
        // r+1..7 still need it — measured correctness regression 2026-09-13
        // (ranks 1-7 computed gate/up against rank 0's silu -> garbage MoE
        // output -> random per-token expert routing -> cache misses 180->236,
        // moeBlockMs 236.7->264.1 ms/token).  Sized (H/32)*kQ8_0x32_STRIDE
        // (~4 KB for Ornith H=2048), grown on demand.
        uint8_t *q8kN_ = nullptr;
        uint64_t q8kNBytes_ = 0;
        // Batched-decode MoE scratch (2026-09-14): the per-rank DOWN-activation
        // Q8_32 scratch (moeQ8kDownP_, [expertUsed][dBlocks*kQ8_0x32_STRIDE])
        // and the per-rank cache-base pointer table (moeRankWq_, [3*expertUsed]
        // void*: slot base of the gate / up / down slice per rank) consumed by
        // kQGemvQ8_0xQ8K_BatchGU / kQuantizeQ8_0x32_Batch /
        // kQGemvQ8_0xQ8K_BatchDown.  Allocated on the first batched decode,
        // freed at destroy().
        uint8_t *moeQ8kDownP_ = nullptr;
        uint64_t moeQ8kDownPBytes_ = 0;
        uint8_t *moeRankWq_ = nullptr;
        uint64_t moeRankWqBytes_ = 0;
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
        // NOTE: moeWgtHost_ itself is declared in the PUBLIC Option-D block.
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
        // cudaHostRegister (pinned) ranges for the moeNormHost_ D2H mirrors
        // (2026-09-14: a pageable async-D2H is synchronous and serializes the
        // per-layer handoff).  Tracked so a grow or destroy can
        // cudaHostUnregister the OLD range BEFORE the vector reallocates /
        // frees it (unregistering a freed pointer is UB, and re-registering a
        // moved range leaves a stale mapping).  [2] mirrors the two
        // double-buffer slots; nullptr = that slot is not pinned.
        void *moeNormPinned_[2] = {nullptr, nullptr};
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
