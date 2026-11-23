#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tinycoder {
    class Model;
}

namespace tinycoder::gpu {

    /// Decode one quantized row block and dot it with a FLOAT activation.
    /// Supported GGML types: 40 (NVFP4), 42 (Q2_0), 19 (IQ1_S), 16 (IQ2_XXS),
    /// 17 (IQ2_XS), 22 (IQ2_S), 23 (IQ4_XS), 0 (F32), 30 (BF16), 8 (Q8_0),
    /// 12 (Q4_K), 13 (Q5_K), 14 (Q6_K), 18 (IQ3_XXS).
    /// `stream` is a cudaStream_t passed as void* so this header stays
    /// host-only.  Returns false on an unsupported type or launch error.
    bool inklingGemv(uint32_t type, const void *w, const float *x, float *out,
                     uint32_t rows, uint32_t cols, void *stream);

    /// Same as inklingGemv but without the trailing cudaDeviceSynchronize.
    /// The caller must order the result on `stream`.
    bool inklingGemvAsync(uint32_t type, const void *w, const float *x, float *out,
                          uint32_t rows, uint32_t cols, void *stream);

    /// Batched prefill GEMV: `m` activations against one weight matrix.
    /// `x` is [m][cols] row-major, `out` is [m][rows] row-major, i.e. the
    /// transpose of the decode-time [rows] layout, so every token's row is
    /// contiguous.  Same blocks/warps shape as inklingGemv (one warp owns four
    /// output rows), and every weight block is decoded ONCE and multiplied by
    /// all `m` activations: that is the whole point, since decode is 3-8
    /// ALU ops per weight and the token batch amortises them.
    /// `m` must be 1..16.  The register ceiling is the accumulators: one fp32
    /// accumulator per (row, token) x four rows per warp = 4*m, i.e. 64 at
    /// m=16, on top of the 4*m/8 decoded weight registers.  Accumulation is
    /// fp32 and, for a given (row, token), the weights are summed in exactly
    /// the order inklingGemv uses, so the result is bit-identical to `m`
    /// single-token calls (the batched tool checks that as max|d| == 0).
    /// Same GGML block types as inklingGemv.  Syncs like inklingGemv.
    bool inklingGemvBatched(uint32_t type, const void *w, const float *x,
                            float *out, uint32_t rows, uint32_t cols, uint32_t m,
                            void *stream);

    /// Same, without the trailing device synchronisation: the batched prefill
    /// graph is one stream of launches and must not drain it per projection.
    bool inklingGemvBatchedAsync(uint32_t type, const void *w, const float *x,
                                 float *out, uint32_t rows, uint32_t cols,
                                 uint32_t m, void *stream);

    /// Small-row batched GEMV: one warp per (token, output row), for a
    /// projection with fewer rows than the batched kernel's 32-row block (the
    /// attention sink projection is 24 x 2304).  Same per-(row, token) lane
    /// order and shuffle tree as inklingGemvBatchedAsync, so it is
    /// bit-identical; it exists because the 32-row block turns a 24-row
    /// projection into ONE block however many tokens are in flight.
    bool inklingGemvBatchedSmallAsync(uint32_t type, const void *w,
                                      const float *x, float *out, uint32_t rows,
                                      uint32_t cols, uint32_t m, void *stream);

    /// Grouped batched GEMV: `nGroups` independent token groups against one
    /// expert-stacked weight tensor.  `x` is [nGroups][capacity][cols], `out`
    /// [nGroups][capacity][rows], `counts[g]` the live rows of group g, and
    /// group g's weights start at g*rows*rowBytes.  Used by the routed MoE: the
    /// m tokens' m*nUsed (token, expert) pairs are grouped by expert, so the
    /// decoder runs once per *distinct* expert.  Same numerics as
    /// inklingGemvBatched per (row, token).
    /// Token-parallel float decoder: same numbers as inklingGemvBatchedAsync
    /// (same per-output dot order) but the token groups go in grid.y, so a
    /// narrow matrix (the F32 router, the BF16 conv projections) fills the
    /// device instead of launching one block per 32 rows.
    bool inklingGemvBatchedTokenParallelAsync(uint32_t type, const void *w,
                                              const float *x, float *out,
                                              uint32_t rows, uint32_t cols,
                                              uint32_t m, void *stream);

    bool inklingGemvBatchedGroupedAsync(uint32_t type, const void *w,
                                        const float *x, float *out, uint32_t rows,
                                        uint32_t cols, uint32_t capacity,
                                        const int32_t *counts, uint32_t nGroups,
                                        void *stream);

    /// Compact-layout form: counts[g] live rows of group g start at row off[g]
    /// of x/out (no capacity stride), so one gather can cover the whole outer
    /// batch however skewed the routing is.
    bool inklingGemvBatchedGroupedOffsetsAsync(uint32_t type, const void *w,
                                               const float *x, float *out,
                                               uint32_t rows, uint32_t cols,
                                               const int32_t *counts,
                                               const int32_t *off,
                                               uint32_t nGroups, void *stream);

    /// Routed-expert GEMV: one launch, `nUsed` experts, one grid of
    /// nUsed*rowsPerExpert blocks.  The expert id for output row
    /// u*rowsPerExpert+r is read on the device from `idx[u]` (written by
    /// kInklingRouter on the same stream), so no host round trip is needed.
    /// `x` is advanced by u*xStride floats per expert (0 for a shared input,
    /// 2*expertFf when the activation is packed per expert).
    bool inklingGemvExpertAsync(uint32_t type, const void *w, const float *x,
                                uint32_t xStride, float *out,
                                uint32_t rowsPerExpert, uint32_t cols,
                                const int32_t *idx, uint32_t nUsed,
                                void *stream);

    /// Full-device Inkling-2-Mini decode graph.  Lazily uploaded host weights +
    /// 48 K/V slots + 48 short-conv histories + one scratch arena.  The CPU
    /// path in ModelInkling.cpp stays the reference and fallback.
    class InklingGPU {
    public:
        InklingGPU();
        ~InklingGPU();

        /// Hybrid path: one projection per call, host in / host out.
        bool gemv(const void *hostW, uint32_t type, uint32_t rows, uint32_t cols,
                  const float *x, float *out);
        bool gemvRows(const void *hostW, uint32_t type, uint32_t rowsTotal,
                      uint32_t cols, uint32_t rowStart, uint32_t numRows,
                      const float *x, float *out);
        bool available() const { return available_; }
        uint64_t uploadedBytes() const;

        /// Run the whole decode graph for `tokens` on the device.
        /// Logits are trimmed to config().vocabSizeUnpadded (200058).  When
        /// `allLogits` is false only the last token's row is written; when it
        /// is true every row is written ([seqLen][vocab]).  One stream, one
        /// host sync at the very end.
        bool forward(const Model &model, const std::vector<int32_t> &tokens,
                     bool allLogits, float *logits, std::string &err);

        /// Batched prefill: run 1..128 tokens as one [H, m] graph.  Same
        /// arithmetic as forward()'s token loop at every stage, with the
        /// projections reused across the whole m: a 128-token prompt pass reads
        /// each dense weight once per 4 tokens (the largest int8Gemv slice whose
        /// reduction tree matches the token path) and each routed expert's
        /// weights once per ~8-token group instead of once per token.
        /// Bit-exact with forward() at 128 and 512 tokens (every position's
        /// top-1 and top-5 set, max|d| == 0).  The tensor-core levers that give
        /// up that exactness are opt-in: TINYCODER_INKLING_GEMM=1 (dense int8
        /// GEMM) and TINYCODER_INKLING_Q2_0_MMA=1 (Q2_0 expert-down MMQ).
        /// Fall back to forward() when this returns false.
        bool forwardBatch(const Model &model, const std::vector<int32_t> &tokens,
                          bool allLogits, float *logits, std::string &err);

        /// Zero the 48 K/V slots and 48 short-conv histories and rewind the
        /// position to 0.  Safe to call before the first forward.
        void clearKVCache();

    private:
        bool ensureGraph(const Model &model, std::string &err);
        struct Impl;
        Impl *impl_ = nullptr;
        bool available_ = false;
    };

}// namespace tinycoder::gpu
