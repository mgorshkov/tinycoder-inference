// Private state shared by InklingGPU.cu (weight cache, lifetime) and
// InklingForward.cu (the decode graph).  Not a public header: it exposes the
// device buffers and the cached per-layer device pointers only.
#pragma once

#include "gpu/InklingGPU.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinycoder::gpu {

    /// Tokens in one layer-major prefill pass.  Namespace-scope so the batched
    /// kernels in InklingBatched.cu can size shared arrays from it too (nested
    /// class names are not visible unqualified there).
    ///
    /// 512, not 128: the dense projections are [H, m] and the routed-MoE pairs
    /// are grouped once per pass, so the larger the outer batch the fewer times
    /// a per-pass constant is paid (block bank, message bank, K/V store,
    /// attention launch, one gather per expert).  512 is what fits: the arena is
    /// 1.2 GiB at 512 tokens, ~600 MiB at 128, and the checkpoint leaves ~1.8
    /// GiB free after weights + K/V cache.
    inline constexpr uint32_t kInklingBatchM = 512;

    /// Elements per quant block, mirrored in InklingGemv.cu.
    inline uint32_t inklingTypeQk(uint32_t type) {
        if (type == 40 || type == 42) return 64u;// NVFP4 / Q2_0
        if (type == 0 || type == 30 || type == 8) return 32u;// F32 / BF16 / Q8_0
        return 256u;// K-quants and IQ types
    }

    /// Bytes per quant block, mirrored in InklingGemv.cu.
    inline uint32_t inklingTypeBlockBytes(uint32_t type) {
        switch (type) {
            case 40: return 36;// NVFP4
            case 42: return 18;// Q2_0
            case 19: return 50;// IQ1_S
            case 16: return 66;// IQ2_XXS
            case 17: return 74;// IQ2_XS
            case 22: return 82;// IQ2_S
            case 23: return 136;// IQ4_XS
            case 0: return 128;// F32, 32 weights per block
            case 30: return 64;// BF16, 32 weights per block
            // The engine's int8Gemv types: only their row stride is needed here
            // (the GEMV itself is Q4_K/Q5_K/Q6_K/Q8_0/IQ2_XS/IQ2_S/IQ3_XXS/
            // IQ3_S via int8Gemv; IQ2_XS/IQ2_S can also reach inklingGemv).
            case 8: return 34;// Q8_0
            case 10: return 84;// Q2_K
            case 11: return 110;// Q3_K
            case 12: return 144;// Q4_K
            case 13: return 176;// Q5_K
            case 14: return 210;// Q6_K
            case 18: return 98;// IQ3_XXS
            case 21: return 110;// IQ3_S
            default: return 0;
        }
    }

    struct InklingGPU::Impl {
        struct DevMat {
            void *p = nullptr;
            uint64_t bytes = 0;
        };

        // ---- Lazily uploaded host weights / arrays, keyed by host pointer ----
        std::unordered_map<const void *, DevMat> mats;
        uint64_t uploadedBytes = 0;

        // ---- Legacy per-call host<->device scratch (InklingGPU::gemv) ----
        void *dx = nullptr;
        void *dy = nullptr;
        uint64_t dxBytes = 0;
        uint64_t dyBytes = 0;

        // ---- One cached device copy of every pointer the graph touches ----
        struct LayerDev {
            const void *attnQ = nullptr;
            const void *attnK = nullptr;
            const void *attnV = nullptr;
            const void *attnO = nullptr;
            const void *attnGate = nullptr;
            const void *attnRel = nullptr;
            const void *sinkProj = nullptr;
            const void *kConvProj = nullptr;// full-attention layers only
            const void *vConvProj = nullptr;
            const void *msgQuery = nullptr;// mainline layout only
            const void *ctrlDown[4] = {nullptr, nullptr, nullptr, nullptr};
            const void *ctrlGain[4] = {nullptr, nullptr, nullptr, nullptr};
            const void *ctrlForget[4] = {nullptr, nullptr, nullptr, nullptr};
            uint32_t ctrlRank = 0;
            const void *router = nullptr;
            const void *gateUp = nullptr;
            const void *down = nullptr;
            const void *kConvW = nullptr;
            const void *vConvW = nullptr;
            const void *attnPostConvW = nullptr;
            const void *ffnPostConvW = nullptr;
            const void *relBias = nullptr;
            const void *attnQClip = nullptr;
            const void *attnRelClip = nullptr;
            const void *rmsNormAttn = nullptr;
            const void *qInpNorm = nullptr;
            const void *kInpNorm = nullptr;
            const void *vInpNorm = nullptr;
            const void *rmsNormFFN = nullptr;
            const void *attnResQuery = nullptr;
            const void *ffnResQuery = nullptr;
            const void *probsBias = nullptr;
            const void *gateUpScale = nullptr;
            const void *downScale = nullptr;
        };

        // ---- Geometry (fixed for one model) ----
        uint32_t hidden = 0;
        uint32_t nHeads = 0;
        uint32_t nKvHeads = 0;
        uint32_t headDim = 0;
        uint32_t kvDim = 0;
        uint32_t nConv = 0;
        uint32_t histLen = 0;
        uint32_t nRel = 0;
        uint32_t extent = 0;
        uint32_t nExperts = 0;
        uint32_t nUsed = 0;
        uint32_t expertFf = 0;
        uint32_t phys = 0;
        uint32_t slots = 0;
        uint32_t core = 0;
        uint32_t blockSize = 0;
        uint32_t vocab = 0;
        uint32_t maxSeq = 0;
        uint32_t sconvChannels = 0;

        bool ready = false;
        cudaStream_t stream = nullptr;
        int kvPos = 0;

        // ---- CUDA graph of one decode token --------------------------------
        // `pos` and everything derived from it live in dPos, so one captured
        // graph replays for every token; a pos++ node advances it.  The host
        // mirrors the value in kvPos.
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphExec = nullptr;
        bool graphReady = false;
        bool graphDisabled = false;
        // Side stream for capture, so a failed capture cannot poison the
        // compute stream (a stream left capturing rejects ordinary copies).
        cudaStream_t captureStream = nullptr;
        // dPos mirrors kvPos only when this is true; multi-token direct calls
        // advance kvPos without touching dPos.
        bool dPosValid = false;
        int32_t *dPos = nullptr;
        // Pinned embedding staging ring (the graph reads one fixed device row).
        std::vector<float *> pinnedEmb;
        uint32_t pinnedEmbIdx = 0;

        // ---- TINYCODER_INKLING_GPU_PROF=1: per-section device timers ------
        // The graph is host-ordered, so two cudaEvents per section boundary
        // measure the device time of each section exactly.  A sync per
        // boundary drains the pipeline (so the sum is larger than the real
        // token time); the shares are what the profile is for.
        static constexpr int kProfSections = 13;
        static constexpr int kProfPool = 2048;
        bool prof = false;
        // One event per section boundary; the interval between boundary k and
        // k+1 is the wall-clock device time of the section that started at k.
        // Nothing is synchronised until the token ends, so the graph runs at
        // full speed and host launch bubbles are attributed to the section
        // that follows them (the sum equals the real token time).
        std::vector<cudaEvent_t> profPool;
        std::vector<int> profBoundarySection;
        uint32_t profIdx = 0;
        double profSec[kProfSections] = {};
        uint64_t profCalls[kProfSections] = {};
        uint64_t profLaunches = 0;
        uint64_t profGemmLaunches = 0;
        void profMark(int section, cudaStream_t s);
        void profReset();
        void profPrint(size_t tokens);

        // ---- Model-level device weights ----
        const void *tokNorm = nullptr;
        const void *finalNorm = nullptr;
        const void *outResQuery = nullptr;
        const void *lmHead = nullptr;
        uint32_t lmHeadType = 0;
        // Per-expert scale fallbacks (layers 20-23 ship no .scale tensors;
        // the CPU reference then uses 1.0f, and a missing bias uses 0.0f).
        std::vector<float> onesHost;
        std::vector<float> zerosHost;
        const void *onesDev = nullptr;
        const void *zerosDev = nullptr;
        // Expanded-layout message queries packed as [core][4][hidden].
        std::vector<float> hostLegacyMsg;
        const void *legacyMsg = nullptr;
        std::vector<LayerDev> layers;

        // ---- Recurrent state ----
        float *kCache = nullptr;// [slots][maxSeq][kvDim]
        float *vCache = nullptr;
        float *convState = nullptr;// [slots][histLen][sconvChannels]

        // ---- Scratch arena ----
        float *arena = nullptr;
        uint64_t arenaBytes = 0;
        float *embAll = nullptr;// [maxSeq][hidden] host-embedded rows

        float *dX = nullptr;
        float *dPrelude = nullptr;
        float *dPassOut = nullptr;
        float *dInpSA = nullptr;
        float *dFfnInp = nullptr;
        float *dMixOut = nullptr;
        float *dMsgMixed = nullptr;
        float *dBlockBank = nullptr;
        float *dBankSnap = nullptr;
        float *dPre0 = nullptr;
        float *dPre1 = nullptr;
        float *dPre2 = nullptr;
        float *dNq = nullptr;
        float *dNk = nullptr;
        float *dNv = nullptr;
        float *dNormed = nullptr;
        float *dPostO = nullptr;
        float *dConvH = nullptr;
        float *dO = nullptr;
        float *dMoeOut = nullptr;
        float *dQ = nullptr;
        float *dK = nullptr;
        float *dV = nullptr;
        float *dKProj = nullptr;
        float *dVProj = nullptr;
        float *dRel = nullptr;
        float *dSink = nullptr;
        float *dTable = nullptr;
        float *dGateProj = nullptr;
        float *dAttnOut = nullptr;
        float *dRouter = nullptr;
        float *dWgt = nullptr;
        float *dGateUp = nullptr;
        float *dExpertOut = nullptr;
        float *dDyn = nullptr;
        float *dCtrlIn = nullptr;
        float *dCtrlG = nullptr;
        float *dCtrlGv = nullptr;
        float *dCtrlFv = nullptr;
        float *dGateUpRank = nullptr;
        float *dDownRank = nullptr;
        int32_t *dIdx = nullptr;
        float *dLogits = nullptr;

        // ---- Batched-prefill scratch (see InklingBatched.cu) ----------------
        // One row per token in flight, m = kBatchM at most.  The block bank is
        // [nBlocks][m][hidden] (block-major) so banking a block is one flat
        // copy of m rows; every other buffer is [m][...] and therefore also
        // flat, which is what lets one projection see [H, m].
        //
        // 128, not 16: the routed experts are grouped by expert, and at m=128
        // the mean group is m*nUsed/nExperts = 8 tokens.  One 16-row MMA tile
        // therefore covers a whole group, so the 7.9 GB of expert weights is
        // streamed ONCE per 128-token chunk instead of once per 16-token chunk.
        // The dense projections use int8Gemm, which reuses its weight tile
        // across the whole M at tile_x = 64/128.  The growable limits are
        // (a) VRAM: every [m][...] buffer scales with m (the arena is ~6.1 MB
        // per token), so 512 tokens does not fit on an 11 GB card with the
        // IQ2_XXS checkpoint; (b) the grouped gather buffers, which are
        // [nExperts][m][...] because one expert may in the worst case receive
        // every token.
        static constexpr uint32_t kBatchM = kInklingBatchM;
        struct BatchDev {
            float *arena = nullptr;
            uint64_t bytes = 0;
            float *x = nullptr;
            float *prelude = nullptr;
            float *passOut = nullptr;
            float *inpSA = nullptr;
            float *ffnInp = nullptr;
            float *mixAttn = nullptr;// [m][3H] attention depth mix
            float *mixH = nullptr;   // [m][H]  ffn / output depth mix
            float *msgMixed = nullptr;// [m][4H]
            float *pre0 = nullptr;
            float *pre1 = nullptr;
            float *pre2 = nullptr;
            float *nq = nullptr;
            float *nk = nullptr;
            float *nv = nullptr;
            float *normed = nullptr;
            float *postO = nullptr;
            float *convH = nullptr;
            float *o = nullptr;
            float *moeOut = nullptr;
            float *blockBank = nullptr;// [bankRows][m][H]
            float *bankSnap = nullptr;// [bankRows][m][H]
            float *q = nullptr;
            float *k = nullptr;
            float *v = nullptr;
            float *kProj = nullptr;
            float *vProj = nullptr;
            float *rel = nullptr;
            float *sink = nullptr;
            float *table = nullptr;
            float *gateProj = nullptr;
            float *attnOut = nullptr;
            float *router = nullptr;
            int32_t *idx = nullptr;
            float *wgt = nullptr;
            float *gateUp = nullptr;
            float *expertOut = nullptr;
            float *dyn = nullptr;
            float *ctrlG = nullptr;
            float *ctrlInv = nullptr;// [2][m] per-token controller RMS inverses
            float *gateUpRank = nullptr;
            float *downRank = nullptr;
            float *logits = nullptr;
            float *emb = nullptr;
            // Pinned staging ring for the chunk's embedding rows: the upload
            // must be ordered on the compute stream (see forwardBatch), and a
            // pageable source would be staged through the driver.
            float *embPinned = nullptr;// [2][kBatchM][hidden]
            uint32_t embPinnedIdx = 0;
            // Routed-MoE grouping, compact: the m*nUsed (token, expert) pairs
            // are counting-sorted by expert, so expert e owns the slots
            // goff[e]..goff[e]+gcount[e] of every [nPairs][...] buffer below.
            // The padded [nExperts][capacity] layout this replaced had to be
            // sized for the worst case (all tokens on one expert) and cost
            // 1.5 GiB at m=512; the compact form is exactly m*nUsed rows
            // whatever the routing does.
            int32_t *gcount = nullptr;// [nExperts]   live rows per expert
            int32_t *goff = nullptr;  // [nExperts+1] slab start per expert
            int32_t *gexp = nullptr;  // [nPairs]     expert of each slot
            int32_t *gtok = nullptr;  // [nPairs]     token of each slot
            int32_t *gpos = nullptr;  // [m*nUsed]    slot of each pair
            float *gwgt = nullptr;    // [nPairs]     routing weight per slot
            // Pinned host mirrors for the exact per-expert int8 GEMV slices:
            // both the live-row count and the slab offset are device values and
            // the slice launches are host decisions.
            int32_t *countsPinned = nullptr;
            int32_t *goffPinned = nullptr;
            float *gx = nullptr;
            float *gout = nullptr;
            float *gact = nullptr;
            bool ready = false;
        } batch;
        bool ensureBatch(std::string &err);
        // Optional per-stage dump of the prelude hidden state (step-1 gate).
        void dumpBatchPrelude(const char *dir, const float *dev, uint32_t m,
                              cudaStream_t s);

        void release();
        void *getRaw(const void *host, uint64_t bytes, std::string &err,
                     const char *what);
        void *getMat(const void *host, uint32_t type, uint32_t rows,
                     uint32_t cols, std::string &err, const char *what);
    };

}// namespace tinycoder::gpu
