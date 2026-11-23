#pragma once
#include <cstdint>

namespace tinycoder::gpu {
    void kInklingRmsNorm(const float *x, const float *weight, float *out,
                         uint32_t n, float eps, void *stream);
    /// Batched RMSNorm: `rows` independent vectors of length `n` (per-KV-head
    /// value norm).  In-place (x == out) is safe.
    void kInklingRmsNormRows(const float *x, const float *weight, float *out,
                             uint32_t n, uint32_t rows, float eps, void *stream);
    void kInklingClip(float *x, uint32_t rows, uint32_t len, const float *scale,
                      float threshold, void *stream);
    void kInklingRelBias(const float *r, const float *bias, const float *sink,
                         float *table, uint32_t heads, uint32_t nRel,
                         uint32_t extent, void *stream);
    void kInklingAttention(const float *q, const float *kCache,
                           const float *vCache, const float *table,
                           uint32_t pos, uint32_t firstKey, uint32_t nKeys,
                           uint32_t headDim, uint32_t nHeads,
                           uint32_t nKvHeads, uint32_t extent, float qkScale,
                           float *out, void *stream);
    /// Same, but also appends the current token's k/v to the slot inside the
    /// attention launch (replaces the two separate device copies).
    void kInklingAttentionStore(const float *q, const float *kcur,
                                const float *vcur, float *kCache, float *vCache,
                                const float *table, uint32_t pos,
                                uint32_t firstKey, uint32_t nKeys,
                                uint32_t nHeads, uint32_t nKvHeads,
                                uint32_t extent, float qkScale, float *out,
                                void *stream);
    /// CUDA-graph form: `pos` and the SWA window/firstKey/nKeys derivation are
    /// read from device memory, so one captured graph replays for every token.
    void kInklingAttentionStorePos(const float *q, const float *kcur,
                                   const float *vcur, float *kCache,
                                   float *vCache, const float *table,
                                   const int32_t *posDev, bool isSwa,
                                   uint32_t window, uint32_t nHeads,
                                   uint32_t nKvHeads, uint32_t extent,
                                   float qkScale, float *out, void *stream);
    /// Full-attention log-scaled positions: q and rel *= tau(pos), with pos
    /// read from device memory (graph form).
    void kInklingTauScale(float *q, uint32_t qDim, float *rel, uint32_t relDim,
                          const int32_t *posDev, float alpha, float floor,
                          void *stream);
    /// Advance the device-side token position (one node at the end of the
    /// captured token graph).
    void kInklingPosStep(int32_t *posDev, void *stream);
    void kInklingMix(const float *states, uint32_t nStates, const float *query,
                     uint32_t nQueries, uint32_t hidden, float *out,
                     void *stream);
    /// Same, with `cur` appended as the (nStates+1)-th state so the caller does
    /// not need a device copy of the current hidden state into the block bank.
    void kInklingMixCurrent(const float *states, uint32_t nStates,
                            const float *cur, const float *query,
                            uint32_t nQueries, uint32_t hidden, float *out,
                            void *stream);
    void kInklingRouter(const float *logits, const float *bias, int32_t *idx,
                        float *wgt, uint32_t nExperts, uint32_t nUsed,
                        float scale, void *stream);
    void kInklingSitu(float *gateUp, uint32_t n, float beta, float lb,
                      void *stream);
    /// Loop control, fused: (a) the two RMS normalisations + the rank-64 down
    /// projection in one launch (g[rank] out); (b) gain/forget + the
    /// cur*(1-forget)+gain*msg mix in one launch.  Replaces six launches per
    /// control path.
    void kInklingControlDown(const float *cur, const float *msg,
                             const float *downW, float *g, uint32_t rank,
                             uint32_t hidden, void *stream);
    void kInklingControlMix(const float *cur, const float *msg,
                            const float *gainW, const float *forgetW,
                            const float *g, float *out, uint32_t hidden,
                            uint32_t rank, void *stream);

    /// One layer's loop controllers, decode form.
    ///
    /// A layer has four independent controllers: three for the attention depth
    /// mix (c = 0..2) and one for the FFN depth mix.  The token graph issued
    /// each one as a `controlDown` + `controlMix` pair, so the section cost was
    /// 96 launches per token and the down kernel recomputed the same two RMS
    /// norms in each of its `rank` blocks.  These three entries split that into
    /// inv -> down -> mix and batch every controller of the layer into one
    /// launch per stage: (a) the two RMS norms of `cur`/`msg`, 2 blocks per
    /// controller; (b) the rank-wide down projection, one block per
    /// (controller, rank); (c) gain/forget + mix, one thread per hidden element.
    ///
    /// The arithmetic is the reference `controlDownKernel`/`controlMixKernel2`
    /// arithmetic, not an approximation: the inv kernel is the same
    /// `rmsInvShared` reduction with the same 256-thread block, the down kernel
    /// sums the same 256 strided partials through the same shared tree, and the
    /// mix kernel walks gain/forget with the same ascending j (a float4 read
    /// only widens the load, the four FMAs stay in order).  Byte-identical
    /// output, so the captured graph's trajectory is unchanged.
    ///
    /// `scratch` must hold 2*kInklingMaxControls invs followed by
    /// kInklingMaxControls*rank g values.
    constexpr uint32_t kInklingMaxControls = 4;
    struct InklingControlArgs {
        const float *cur[kInklingMaxControls];
        const float *msg[kInklingMaxControls];
        const float *downW[kInklingMaxControls];
        const float *gainW[kInklingMaxControls];
        const float *forgetW[kInklingMaxControls];
        float *out[kInklingMaxControls];
        float *g;// [nCtl*rank]
        uint32_t nCtl;
        uint32_t rank;
        uint32_t hidden;
    };
    void kInklingControlDownFused(const InklingControlArgs &a, void *stream);
    void kInklingControlMixFused(const InklingControlArgs &a, void *stream);
    /// Fused per-expert scale + SITU over all routed experts: one block per
    /// expert, `scale[u]` applied to both gate and up before the activation.
    void kInklingSituAll(float *gateUp, uint32_t nUsed, uint32_t n,
                         const float *scale, float beta, float lb, void *stream);
    void kInklingMoeAccum(const float *expertOut, const float *wgt,
                          const float *scale, float *out, uint32_t nUsed,
                          uint32_t hidden, void *stream);
    void kInklingShortConv(const float *cur, const float *weight,
                           const float *dynamic, float *history, float *out,
                           uint32_t channels, uint32_t nConv, uint32_t nHeads,
                           bool residual, float scale, void *stream);

    // ---- batched prefill (m tokens in flight, every buffer [m][...]) --------
    //
    // The layouts are the decode graph's with a leading token index: one
    // activation row per token, so a projection becomes x[m][cols] @ W^T and
    // can run through inklingGemvBatched.  The block bank is [nBlocks][m][H]
    // instead (block-major), so one flat copy of m rows banks a block.
    //
    // Each kernel reproduces the arithmetic of its single-token counterpart in
    // the same order, so m=1 is bit-identical to the decode path.

    /// out = weighted RMS mix of `nStates` states (strided by token) plus the
    /// current row.  states is [nStates][m][hidden] (bank layout), cur/out are
    /// [m][*Stride].  One block per (token, query).
    void kInklingMixCurrentBatch(const float *states, uint32_t nStates,
                                 const float *cur, uint32_t curStride,
                                 const float *query, uint32_t nQueries,
                                 uint32_t hidden, float *out,
                                 uint32_t outStride, uint32_t m, void *stream);
    /// Same without the appended current row (cross-pass message bank).
    void kInklingMixBatch(const float *states, uint32_t nStates,
                          const float *query, uint32_t nQueries,
                          uint32_t hidden, float *out, uint32_t outStride,
                          uint32_t m, void *stream);
    /// Per-head L2 clip over [m][rows][len]; `scale` has `rows` rows shared by
    /// every token.
    void kInklingClipBatch(float *x, uint32_t rows, uint32_t len,
                           const float *scale, float threshold, uint32_t m,
                           void *stream);
    /// Relative-bias tables for all m tokens: table[t][h][d] from
    /// rel[t][h][:], sink[t][h].
    void kInklingRelBiasBatch(const float *r, const float *bias,
                              const float *sink, float *table, uint32_t heads,
                              uint32_t nRel, uint32_t extent, uint32_t pos0,
                              uint32_t m, void *stream);
    /// Append m tokens' K/V rows to the slot, then m-query causal attention.
    void kInklingStoreKvBatch(const float *kcur, const float *vcur,
                              float *kCache, float *vCache, uint32_t pos0,
                              uint32_t kvDim, uint32_t m, void *stream);
    void kInklingAttentionBatch(const float *q, const float *kCache,
                                const float *vCache, const float *table,
                                uint32_t pos0, uint32_t nHeads,
                                uint32_t nKvHeads, uint32_t extent,
                                float qkScale, float *out, uint32_t m,
                                uint32_t window, bool isSwa, void *stream);
    /// q/rel *= tau(pos0+t) for each token (full-attention layers).
    void kInklingTauScaleBatch(float *q, uint32_t qDim, float *rel,
                               uint32_t relDim, uint32_t pos0, float alpha,
                               float floor, uint32_t m, void *stream);
    /// Causal depthwise short convolution over m consecutive tokens.  The
    /// state is per slot and sequential, so the kernel walks the chunk in
    /// order for each channel: token t's taps see tokens t-1, t-2, ...
    void kInklingShortConvBatch(const float *cur, const float *weight,
                                const float *dynamic, float *history, float *out,
                                uint32_t channels, uint32_t nConv,
                                uint32_t nHeads, bool residual, float scale,
                                uint32_t m, void *stream);
    /// Top-k router for m tokens: idx/wgt are [m][nUsed].
    void kInklingRouterBatch(const float *logits, const float *bias, int32_t *idx,
                             float *wgt, uint32_t nExperts, uint32_t nUsed,
                             float scale, uint32_t m, void *stream);
    /// Loop controller over m tokens, bit-exact with the decode kernels
    /// (controlDownKernel / controlMixKernel2): their reduction order is
    /// preserved so the cross-pass control cannot shift the trajectory.
    /// g is [m][rank]; cur/msg are [m][*Stride].
    void kInklingControlDownBatch(const float *cur, uint32_t curStride,
                                  const float *msg, uint32_t msgStride,
                                  const float *downW, float *g, uint32_t rank,
                                  uint32_t hidden, uint32_t m, void *stream);
    /// out = cur - tanh(fv)*cur + tanh(gv)*msg, gv/fv from the rank dots.
    void kInklingControlMixBatch(const float *cur, uint32_t curStride,
                                 const float *msg, uint32_t msgStride,
                                 const float *gainW, const float *forgetW,
                                 const float *g, float *out, uint32_t hidden,
                                 uint32_t rank, uint32_t m, void *stream);
    /// SITU over a compact routed-pair gate/up buffer: one block per pair, pair
    /// p uses expert gexp[p]'s scale and lives at row p of gateUp/act.
    void kInklingSituGrouped(float *gateUp, float *act, const int32_t *gexp,
                             const float *scale, uint32_t nPairs, uint32_t n,
                             float beta, float lb, void *stream);
    /// Same activation, but from a per-token gate/up buffer (the exact path,
    /// where gate/up is a per-token int8 expert GEMV) into the compact layout:
    /// pair p lands at row gpos[p].
    void kInklingSituGather(const float *gateUp, const int32_t *idx,
                            const int32_t *gpos, float *act,
                            const float *scale, uint32_t n, uint32_t nUsed,
                            uint32_t m, float beta, float lb, void *stream);
    /// Counting-sort the m*nUsed routed (token, expert) pairs by expert into a
    /// compact pair array: counts[e] rows of expert e start at goff[e]
    /// (goff[nExperts] is the total).  gtok/gexp/gwgt are [nPairs], gpos is
    /// [m*nUsed].  No capacity bound, so one gather can cover the whole outer
    /// batch however skewed the routing is.
    void kInklingMoeGroupBuild(const int32_t *idx, const float *wgt,
                               int32_t *counts, int32_t *goff, int32_t *gtok,
                               int32_t *gpos, float *gwgt, int32_t *gexp,
                               uint32_t m, uint32_t nUsed, uint32_t nExperts,
                               void *stream);
    /// gx[slot] = x[gtok[slot]] for every compact slot, one block per slot.
    /// nPairs is the total row count (goff[nExperts]); the compact layout puts
    /// every slot in [0, nPairs) so no per-expert bounds are needed.
    void kInklingGatherExpertX(const float *x, const int32_t *gtok,
                               uint32_t nPairs, float *gx, uint32_t cols,
                               void *stream);
    /// out[t] = sum_u wgt[t][u]*downScale[idx[t][u]]*expertOut[gpos[t][u]].
    void kInklingMoeAccumGrouped(const float *expertOut, const int32_t *idx,
                                 const int32_t *gpos, const float *wgt,
                                 const float *downScale, float *out,
                                 uint32_t nUsed, uint32_t hidden, uint32_t m,
                                 void *stream);

    // ---- Q2_0 expert-down int8 MMQ -----------------------------------------
    //
    // Q2_0 is 18 B / 64 weights (fp16 d, 16 B of 2-bit codes, value
    // d*(code-1)).  These entry points quantize m activation rows to Q8_1
    // (one scale per 32 weights) and run mma.s8 over a shared-memory int8 tile
    // of the packed weights.  cols need only be a multiple of 64 (the Q2_0
    // block), not of 512: the Inkling down projection is cols=1152,
    // rows=2304, and 1152 = 18*64.
    //
    // Grouped form: w is [nExperts][rows][cols] contiguous Q2_0, x is
    // [nExperts][capacity][cols] fp32, out is [nExperts][capacity][rows].
    // counts[e] is the number of valid rows in that expert's group; rows
    // beyond counts[e] are not written.  capacity is the group *stride* (the
    // layer-major graph passes kBatchM = 128); the kernel walks the live rows
    // in 16-row MMA tiles, so a group of up to 128 tokens still streams the
    // expert's weights once.
    // `quantize=false` reuses the packed activations of the previous call on
    // the same x (timing/diagnostic use only).
    bool int8GemmQ2_0Grouped(const void *w, const float *x, float *out,
                             const int32_t *counts, uint32_t capacity,
                             uint32_t rows, uint32_t cols, uint32_t nExperts,
                             void *stream, bool quantize = true);
    /// Compact-layout form: counts[e] live rows of expert e start at row
    /// off[e] of x/out (nSlots rows in total), so no capacity stride exists.
    /// The packed Q8_1 scratch is nSlots rows.
    bool int8GemmQ2_0GroupedOffsets(const void *w, const float *x, float *out,
                                    const int32_t *counts, const int32_t *off,
                                    uint32_t nSlots, uint32_t rows,
                                    uint32_t cols, uint32_t nExperts,
                                    void *stream, bool quantize = true);
    /// Single-matrix form: out[seqLen][rows] = x[seqLen][cols] @ W^T.
    bool int8GemmQ2_0(const void *w, const float *x, float *out,
                      uint32_t seqLen, uint32_t rows, uint32_t cols,
                      void *stream, bool quantize = true);

    // Grouped int8 MMQ for the routed gate/up experts whose blocks are
    // IQ2_XXS (16), IQ2_XS (17) or IQ2_S (22): the Q2_0 down kernel's layout
    // with the i-quant block decoders.  Same contract as
    // int8GemmQ2_0Grouped: w is [nExperts][rows][cols], x is
    // [nExperts][capacity][cols] fp32, out is [nExperts][capacity][rows],
    // counts[e] live rows per expert, cols a multiple of 256 and rows a
    // multiple of 64.  The IQ family carries its scale per 16 weights, so the
    // k loop steps 16 weights per mma (four scale slots per 64-k chunk) rather
    // than Q2_0's one scale per 64.  Returns false for an unsupported
    // type/shape so the caller can keep the exact path.
    bool inklingExpertMmaSupported(uint32_t type);
    bool inklingExpertMmaGrouped(const void *w, const float *x, float *out,
                                 uint32_t rows, uint32_t cols, uint32_t capacity,
                                 const int32_t *counts, uint32_t nExperts,
                                 uint32_t type, void *stream,
                                 bool quantize = true);
    /// Compact-layout form: counts[e] live rows of expert e start at row off[e]
    /// of x/out (nSlots rows in total), so a group of any size still streams its
    /// expert's weights once.
    bool inklingExpertMmaGroupedOffsets(const void *w, const float *x, float *out,
                                        uint32_t rows, uint32_t cols,
                                        const int32_t *counts, const int32_t *off,
                                        uint32_t nSlots, uint32_t nExperts,
                                        uint32_t type, void *stream,
                                        bool quantize = true);
}// namespace tinycoder::gpu
