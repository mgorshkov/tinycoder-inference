#pragma once

// Context window planning: how many tokens may be cached, and at what K/V
// precision, given the model's own trained limit and the memory that is
// actually free.
//
// Pure arithmetic, no CUDA and no model: `planContext()` is unit-tested
// without a checkpoint.  The two memory probes are the only system calls.

#include <cstdint>
#include <string>

namespace tinycoder::detail {

    /// K/V storage precision.  I8 is a real store (prefill writes it, attention
    /// reads it), not a decode-only mirror.
    enum class KvPrecision { F32,
                             I8 };

    /// Everything the planner needs that comes from the checkpoint or the machine.
    struct ContextRequest {
        uint32_t requested = 0;         ///< caller's window, 0 = auto (2048)
        uint32_t modelMax = 0;          ///< GGUF *.context_length; 0 = unknown
        uint32_t numLayers = 0;         ///< all layers (for per-layer weight share)
        uint32_t numAttentionLayers = 0;///< layers that own K/V slots
        uint32_t numKVHeads = 0;
        uint32_t headDim = 0;
        uint32_t ropeDims = 0;       ///< headDim + mrope dims (cos/sin row width)
        bool int8KvSupported = false;///< kernels for an int8 store exist
        /// Pin the K/V precision instead of taking the first rung that fits.  The
        /// two-arm numerics check needs the SAME window served both ways; without
        /// this a 2048-token request always lands on f32 and there is nothing to
        /// compare the int8 store against.  Unset = normal ladder.
        KvPrecision forceKv = KvPrecision::F32;
        bool forceKvSet = false;
        uint64_t weightBytes = 0; ///< device-resident weight form
        uint64_t freeVram = 0;    ///< 0 = no usable GPU
        uint64_t availableRam = 0;///< free RAM *after* the weights are resident
        /// Non-K/V, non-weight device cost of a session: cuBLAS workspaces, the
        /// RoPE/MRoPE tables, attention/logits scratch, and the device form of the
        /// LM head (re-quantized to Q8_K for a separate head) plus the embedding
        /// upload.  It EXCLUDES the CUDA context (~600 MiB): the free-VRAM probe
        /// itself creates the context, so `freeVram` is already net of it.
        ///
        /// Measured on the reference card, as (used - weights - K/V):
        ///   9B at a 45k f32 window:  10426 - 5376 - 2872 = 2178 MiB
        ///   27B EXL3 at 2048:        10820 - 8187 - 324  = 2309 MiB
        /// and ~600 MiB of either is the context the probe already removed, so the
        /// scratch share is ~1.6-1.7 GiB.  1792 MiB is that, rounded up.
        ///
        /// Both directions have been wrong here: 900 MiB granted a 60672-token f32
        /// window on the 9B that OOM'd on its first prefill chunk, and 2560 MiB left
        /// the 27B with no f32 budget at all, which silently moved its DEFAULT 2048
        /// window onto the int8 store and shifted the reference logit by +0.028
        /// (19.336 instead of 19.308).  1408 MiB is the 27B's measured scratch share
        /// rounded up, and it is the number both ceilings are verified against: the
        /// 27B must keep its 2048-token f32+mirror default, and the 9B's granted f32
        /// ceiling must actually run (Task 7 gate).
        uint64_t gpuFixedOverhead = 1408ull << 20;
    };

    /// The decision, plus the sentence that reports it.  `reason` is what gets
    /// printed, so the message and the decision cannot diverge.
    struct ContextPlan {
        uint32_t seqLen = 0;    ///< the granted window; 0 = nothing fits
        uint32_t maxAllowed = 0;///< largest window the budget permits at this `kv`
        KvPrecision kv = KvPrecision::F32;
        bool kvMirror = false;///< f32 store + int8 decode mirror (arch-1 default)
        bool kvOnGpu = false;
        uint64_t kvBytes = 0;
        std::string reason;
    };

    /// Bytes of K/V plus RoPE tables for one cached token at `p` (and the int8
    /// decode mirror when the store is f32).
    uint64_t kvBytesPerToken(const ContextRequest &r, KvPrecision p, bool mirror);

    /// The K/V-only part of the above, for one *attention* layer: 2 KV heads' worth
    /// of store, plus scales for int8 or the mirror for f32.  The auto-fit charges
    /// this per layer; the RoPE tables are ctx-linear and charged once.
    uint64_t kvStoreBytesPerToken(const ContextRequest &r, KvPrecision p, bool mirror);

    /// Resolve the window.  Auto (requested == 0) is 2048: today's default, so
    /// published numbers stay comparable; the report line carries `maxAllowed` so
    /// the memory-derived ceiling is visible.  An explicit request is honored up
    /// to `maxAllowed`, escalated to int8 before it is reduced.
    ContextPlan planContext(const ContextRequest &r);

    /// Free device memory in bytes; 0 when there is no usable GPU (no USE_CUDA
    /// build, TINYCODER_GPU=0, or cudaMemGetInfo fails).
    uint64_t probeFreeVramBytes();

    /// @brief Total VRAM of the device that probeFreeVramBytes() reports, in bytes
    /// (0 when there is no usable device).  Used only to tell "this checkpoint can
    /// never fit this card" (run it on the host) from "the card is busy right now"
    /// (reduce the window or say so), never to size a grant.
    uint64_t probeTotalVramBytes();

    /// MemAvailable in bytes, falling back to free RAM from sysinfo().
    uint64_t probeAvailableRamBytes();

}// namespace tinycoder::detail
