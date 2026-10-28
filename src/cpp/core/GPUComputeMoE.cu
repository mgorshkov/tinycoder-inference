/*
 * TinyCoder CUDA GPU offload engine -- MoE / qwen35 prefix
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

    // ===== Expert cache build, resident arenas, qwen35 prefix driver =====
    bool GPUModel::buildExpertCache(std::string &errMsg) {
        // Hybrid-only: the 256 per-expert matrices stay in host RAM, so the
        // per-expert GEMVs cannot read layers_[L].ffnGateExps.q (host pointer).
        // Allocate a 2-way SET-ASSOCIATIVE device cache: kExpertCacheSets sets
        // x 2 ways = 64 slots per layer (TINYCODER_MOE_WAYS = sets, default
        // 32).  Each slot is ONE CONTIGUOUS [gate][up][down] block
        // (slotBytes = 2*gateSlice + downSlice) so a miss is a single pinned
        // DMA H2D.  With 40 layers this costs ~8.2 GB at 64 slots / 3.3 MB
        // per expert (see the VRAM budget in the header comment).
        if (!moeCpuFn_ || expertCacheBuilt_) return true;
        // Runtime set count: TINYCODER_MOE_WAYS (power of 2, clamped to
        // kExpertCacheSets).  A/B tool: more sets shrink conflicts (more
        // VRAM); fewer sets shrink VRAM (more conflicts).
        {
            const char *wv = std::getenv("TINYCODER_MOE_WAYS");
            if (wv != nullptr && wv[0] != '\0') {
                long v = std::strtol(wv, nullptr, 10);
                // Accept non-power-of-2 too (e.g. 24 sets = 48 slots for the
                // capacity A/B); the hash still masks to a power-of-2 PREFIX
                // (hash % sets via rejection-free mul-based range) so any set
                // count <= kExpertCacheSets works.
                if (v >= 2 && v <= static_cast<long>(kExpertCacheSets)) {
                    expertCacheSets_ = static_cast<uint32_t>(v);
                }
            } else {
                expertCacheSets_ = kExpertCacheSetsDefault;
            }
        }
        {
            const char *dnE = std::getenv("TINYCODER_DUMP_MOE");
            if (dnE != nullptr && dnE[0] != '\0' && dnE[0] != '0') {
                std::fprintf(stderr,
                             "[cache build] expertFF=%u expertHM=%u count=%u "
                             "gateRowBytes=%u downRowBytes=%u type=%u "
                             "sets=%u slots=%u\n",
                             expertFF_, expertHM_, expertCount_,
                             expertRowBytesGate_, expertRowBytesDown_,
                             expertType_, expertCacheSets_,
                             expertCacheSets_ * kExpertCacheWays);
            }
        }
        const uint32_t nGpu = geom_.numGpuLayers ? geom_.numGpuLayers
                                                 : geom_.numLayers;
        if (expertFF_ == 0 || expertHM_ == 0 || expertCount_ == 0 ||
            expertHostGate_ == nullptr || expertHostDown_ == nullptr) {
            errMsg = "buildExpertCache: missing host expert geometry (hybrid "
                     "upload did not record it)";
            return false;
        }
        // Per-layer geometry: gateSlice/downSlice vary per layer when the
        // down-expert quant type differs between layers (Ornith Q4_K_M layers
        // 5-6 down=Q4_K 144 B/block vs the rest down=Q6_K 210 B/block), so
        // EACH layer's arena is sized from ITS OWN slot record (recorded in
        // upload(); a missing record falls back to the last-layer globals).
        const uint32_t sets = expertCacheSets_ ? expertCacheSets_
                                               : kExpertCacheSetsDefault;
        uint64_t slotBytesL0 = 0;// layer-0 slot size (bw selftest below)
        for (uint32_t L = 0; L < nGpu; ++L) {
            const uint32_t layGateRowBytes =
                    expertCache_[L].gateRowBytes
                            ? expertCache_[L].gateRowBytes
                            : expertRowBytesGate_;
            const uint32_t layDownRowBytes =
                    expertCache_[L].downRowBytes
                            ? expertCache_[L].downRowBytes
                            : expertRowBytesDown_;
            const uint64_t gateSlice = static_cast<uint64_t>(expertFF_) *
                                       layGateRowBytes;
            const uint64_t downSlice = static_cast<uint64_t>(expertHM_) *
                                       layDownRowBytes;
            const uint64_t slotBytes = 2 * gateSlice + downSlice;
            if (L == 0) slotBytesL0 = slotBytes;
            // One contiguous slot-major arena per layer: slot s owns
            // [s*slotBytes, (s+1)*slotBytes) = [gate][up][down]; slot index =
            // set*2 + way.
            uint8_t *arena = nullptr;
            cudaError_t e = cudaMalloc(
                    &arena, static_cast<size_t>(slotBytes * sets *
                                                kExpertCacheWays));
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMalloc(expertCache): ") +
                         cudaGetErrorString(e);
                // Free what we already built, leave expertCacheBuilt_ false.
                for (uint32_t l2 = 0; l2 < L; ++l2) {
                    if (expertCache_[l2].gateQ) cudaFree(expertCache_[l2].gateQ);
                    expertCache_[l2] = ExpertSlot();
                }
                return false;
            }
            expertCache_[L].gateQ = arena;
            // Slot-0 interior pointers (per-slot offsets are re-derived in
            // ensureExpertCached from the slot-major stride, so these are
            // informational only).
            expertCache_[L].upQ = arena + static_cast<size_t>(gateSlice);
            expertCache_[L].downQ =
                    arena + static_cast<size_t>(2 * gateSlice);
            for (uint32_t s2 = 0; s2 < sets; ++s2) {
                expertCacheTag_[L][s2][0] = -1;
                expertCacheTag_[L][s2][1] = -1;
                expertCacheLru_[L][s2] = 0;
            }
            // ---- Keep the host expert blobs resident (2026-09-16).  The
            // weights are zero-copy mmap'd (GGUFLoader), so with a ~35 GB
            // model on a 31 GB host the kernel evicts expert pages while the
            // router touches other layers; the per-miss fill memcpy then
            // page-faults and re-reads 3.3 MB from the file (measured:
            // per-miss fillEnq 0.31-0.54 ms warm vs 3-18 ms cold — a 30x
            // swing that exactly tracks rankLoopMs/fillMs ballooning to
            // 2000-5000 ms/token).  Long-term (per-expert) residency is a
            // kExpertCacheWays-way per-layer working set; MADV_WILLNEED is a
            // pure hint that prefetches the pages without pinning them.
            // Opt-out with TINYCODER_NO_WILLNEED=1 (measurement A/B).
            {
                const char *noWn = std::getenv("TINYCODER_NO_WILLNEED");
                if (noWn == nullptr || noWn[0] == '\0' || noWn[0] == '0') {
#if defined(__linux__)
                    const uint8_t *hGate = expertCache_[L].hostGate
                                                   ? expertCache_[L].hostGate
                                                   : expertHostGate_;
                    const uint8_t *hUp = expertCache_[L].hostUp
                                                 ? expertCache_[L].hostUp
                                                 : expertHostUp_;
                    const uint8_t *hDown = expertCache_[L].hostDown
                                                   ? expertCache_[L].hostDown
                                                   : expertHostDown_;
                    if (hGate)
                        ::madvise(const_cast<void *>(static_cast<const void *>(hGate)),
                                  static_cast<size_t>(expertCount_) * gateSlice,
                                  MADV_WILLNEED);
                    if (hUp)
                        ::madvise(const_cast<void *>(static_cast<const void *>(hUp)),
                                  static_cast<size_t>(expertCount_) * gateSlice,
                                  MADV_WILLNEED);
                    if (hDown)
                        ::madvise(const_cast<void *>(static_cast<const void *>(hDown)),
                                  static_cast<size_t>(expertCount_) * downSlice,
                                  MADV_WILLNEED);
#endif
                }
            }
        }
        expertCacheBuilt_ = true;
        // TINYCODER_BW_SELFTEST=1: measure the two H2D legs (host->pinned
        // memcpy fill and pinned->device DMA) so a bandwidth ceiling can be
        // attributed to the right one.  The MoE block remains ~225 ms/token
        // even with pinned staging, so this is the decisive measurement.
        if (std::getenv("TINYCODER_BW_SELFTEST") != nullptr) {
            const uint8_t *src = expertHostGate_;
            // Use the same registration scheme as the real path so the
            // selftest measures the ACTUAL staging buffer behavior.
            void *pp = nullptr;
            if (posix_memalign(&pp, 4096,
                               static_cast<size_t>(slotBytesL0)) != 0) {
                errMsg = "posix_memalign(bw selftest) failed";
                return false;
            }
            cudaHostRegister(pp, static_cast<size_t>(slotBytesL0),
                             cudaHostRegisterMapped | cudaHostRegisterPortable);
            expertStage_[0] = static_cast<uint8_t *>(pp);
            double fillGB = bwSelfTestHostToPinned(src, expertStage_[0],
                                                   static_cast<size_t>(
                                                           slotBytesL0),
                                                   64);
            // Pinned -> device DMA (one slot, repeated).
            uint8_t *dst = expertCache_[0].gateQ;
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < 64; ++i) {
                cudaMemcpyAsync(dst, expertStage_[0],
                                static_cast<size_t>(slotBytesL0),
                                cudaMemcpyHostToDevice, g_stream);
            }
            cudaStreamSynchronize(g_stream);
            double dmaSec = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - t1)
                                    .count();
            double dmaGB = (double) slotBytesL0 * 64 / dmaSec / (1ull << 30);
            std::fprintf(stderr,
                         "[bw selftest] slotBytes=%llu hostFill=%.1f GB/s "
                         "pinnedDMA=%.1f GB/s\n",
                         static_cast<unsigned long long>(slotBytesL0),
                         fillGB, dmaGB);
        }
        return true;
    }

    const uint8_t *GPUModel::ensureExpertCached(uint32_t L, uint32_t e,
                                                std::string &errMsg,
                                                const uint8_t **downQOut) {
        if (!expertCacheBuilt_ || e >= expertCount_) return nullptr;
        const uint32_t set = expertCacheSet(e);
        ExpertSlot &s = expertCache_[L];
        // PER-LAYER geometry (2026-09-17): the globals are overwritten each
        // upload iteration and hold only the last layer's values; the cache
        // slot record holds THIS layer's real gate/down types and row strides.
        const uint32_t layGateRowBytes = s.gateRowBytes ? s.gateRowBytes
                                                        : expertRowBytesGate_;
        const uint32_t layDownRowBytes = s.downRowBytes ? s.downRowBytes
                                                        : expertRowBytesDown_;
        const uint64_t gateSlice = static_cast<uint64_t>(expertFF_) *
                                   layGateRowBytes;
        const uint64_t downSlice = static_cast<uint64_t>(expertHM_) *
                                   layDownRowBytes;
        const uint64_t slotBytes = 2 * gateSlice + downSlice;
        const uint32_t sets = expertCacheSets_ ? expertCacheSets_
                                               : kExpertCacheSetsDefault;
        if (set >= sets) {
            errMsg = "ensureExpertCached: set out of runtime set count";
            return nullptr;
        }
        // ---- 2-way set-associative lookup.  The set is set; the two ways
        // (0 and 1) hold their own tags.  Arena slot = set*2 + way.
        // ---- SLOT-MAJOR arena (2026-09-14): slot s owns the CONTIGUOUS range
        // [s*slotBytes, (s+1)*slotBytes) = [gate][up][down].  gateQ is the
        // layer arena base; up = base + gateSlice; down = base + 2*gateSlice.
        const uint8_t *g0 = static_cast<const uint8_t *>(s.gateQ) +
                            static_cast<size_t>(set * kExpertCacheWays) *
                                    static_cast<size_t>(slotBytes);
        const uint8_t *g1 = g0 + static_cast<size_t>(slotBytes);
        // Hit: one of the two ways holds expert e.
        if (expertCacheTag_[L][set][0] == static_cast<int32_t>(e)) {
            // Touch makeLRU = way1 (so way0 is now the MRU of the pair).
            expertCacheLru_[L][set] = 1;
            if (downQOut) *downQOut = g0 + 2 * gateSlice;
            return g0;
        }
        if (expertCacheTag_[L][set][1] == static_cast<int32_t>(e)) {
            expertCacheLru_[L][set] = 0;
            if (downQOut) *downQOut = g1 + 2 * gateSlice;
            return g1;
        }
        // ---- Miss: choose the LRU way (expertCacheLru_ bit; initialize to
        // way0 if both are empty so the first miss fills way0).
        uint32_t victim = expertCacheLru_[L][set];
        const uint8_t *gBase = (victim == 0) ? g0 : g1;
        if (expertCacheTag_[L][set][0] < 0 && expertCacheTag_[L][set][1] < 0)
            victim = 0, gBase = g0;
        // Flip the LRU bit so the OTHER way is now LRU (2-way pseudo-LRU:
        // after filling victim, the survivor becomes LRU next).
        expertCacheLru_[L][set] = (victim == 0) ? 1u : 0u;
        if (downQOut) *downQOut = gBase + 2 * gateSlice;
        // ---- Miss path: ONE contiguous pinned H2D of the whole slot on
        // g_stream.  The host expert bytes are PINNED into a per-miss ring
        // buffer (host-side memcpy is synchronous, THEN one cudaMemcpyAsync
        // from PINNED memory reaches the PCIe 3.0 x16 DMA ceiling ~6 GB/s —
        // the old path did 3 pageable bounces at ~2.4 GB/s; measured 500
        // MB/token ≈ 225 ms/token in moeBlockMs).  Copies stay on g_stream
        // (NOT g_stream2): the GEMVs that read the slices run on g_stream,
        // and a g_stream2 H2D without a matching stream-wait lets the GEMVs
        // read stale bytes (measured: broke SequentialDecodeArgmaxAgrees with
        // the 2026-09-14 prefetch attempt).  A torn read between the two
        // stage copies is harmless because both copies are bit-identical
        // (the HOST expert blobs are immutable constants, so every fill
        // writes the exact same bytes).
        // ----
        // Per-layer H2D source (recorded in upload(): every layer has its OWN
        // expert host blobs — see expertCache_[L].hostGate/Up/Down).
        const uint8_t *srcGate = s.hostGate ? s.hostGate : expertHostGate_;
        const uint8_t *srcUp = s.hostUp ? s.hostUp : expertHostUp_;
        const uint8_t *srcDown = s.hostDown ? s.hostDown : expertHostDown_;
        // Allocate (grow-on-demand) the pinned stage ring.  MEASURED
        // 2026-09-14 (TINYCODER_BW_SELFTEST): cudaHostAlloc (write-combined
        // mapped pages) drops the HOST-SIDE fill to 0.8 GB/s — the memcpy into
        // WC memory is the bottleneck, not the DMA (11.5 GB/s to the device).
        // malloc + cudaHostRegister keeps the pages CACHEABLE so the host fill
        // runs at DRAM speed, and the bits then DMA over at the same wire
        // speed.  Registered pages must be page-aligned (posix_memalign).
        if (expertStageBytes_ < slotBytes) {
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
                if (posix_memalign(&pp, 4096, static_cast<size_t>(slotBytes)) !=
                    0) {
                    errMsg = "posix_memalign(expertStage) failed";
                    return nullptr;
                }
                cudaError_t eP = cudaHostRegister(
                        pp, static_cast<size_t>(slotBytes),
                        cudaHostRegisterMapped | cudaHostRegisterPortable);
                if (eP != cudaSuccess) {
                    std::free(pp);
                    errMsg = std::string("cudaHostRegister(expertStage): ") +
                             cudaGetErrorString(eP);
                    return nullptr;
                }
                expertStage_[b] = static_cast<uint8_t *>(pp);
                cudaEventCreateWithFlags(&expertStageEv_[b],
                                         cudaEventDisableTiming);
            }
            expertStageBytes_ = slotBytes;
        }
        const uint32_t st = (expertStageFlag_++) % kExpertStageBufs;
        // TINYCODER_TRACE_MISS=1: per-miss wall time split (host fill + memcpy
        // enqueue) to attribute the moeBlockMs budget.
        bool trMiss = false;
        auto tMiss0 = std::chrono::steady_clock::now();
        if (std::getenv("TINYCODER_TRACE_MISS") != nullptr) trMiss = true;
        // ---- DRAIN-BEFORE-REFILL (2026-09-15 root-cause, parity): the stage
        // ring is shared by ALL fills, and with >kExpertStageBufs misses per
        // forward the round-robin claim wraps — miss k+8 refills stage st=a
        // with a DIFFERENT expert's bytes while the copy engine may still be
        // DMA-ing miss k's H2D OUT of that stage.  That tears the pinned
        // source mid-transfer and lands blended expert bytes in the device
        // slot, deterministically corrupting the routed-expert weights (both
        // the batched fast path AND the TINYCODER_MOE_SLOW per-rank path
        // failed SequentialDecodeArgmaxAgrees with the SAME corrupted tokens
        // while the CPU-expert fallback — which never touches this ring —
        // passed).  The old "the host blobs are immutable so re-fill writes
        // the same bytes" argument only holds when the SAME expert refills
        // the stage; a wrap refills with DIFFERENT bytes.  Wait for the
        // previous owner's DMA to drain before overwriting (cheap when the
        // copy already completed — the common case when fills run ahead).
        if (cudaEventQuery(expertStageEv_[st]) != cudaSuccess) {
            cudaEventSynchronize(expertStageEv_[st]);
        }
        // Synchronous host fill into PINNED memory (memory-bandwidth bound;
        // ~3.3 MB ≈ tens of µs) — then ONE DMA H2D per miss.
        std::memcpy(expertStage_[st],
                    srcGate + static_cast<uint64_t>(e) * gateSlice,
                    static_cast<size_t>(gateSlice));
        std::memcpy(expertStage_[st] + gateSlice,
                    srcUp + static_cast<uint64_t>(e) * gateSlice,
                    static_cast<size_t>(gateSlice));
        std::memcpy(expertStage_[st] + 2 * gateSlice,
                    srcDown + static_cast<uint64_t>(e) * downSlice,
                    static_cast<size_t>(downSlice));
        // DEVICE-SLOT WAR FIX (2026-09-17): the H2D fill MUST run in-order on
        // g_stream, NOT pipelined on g_stream2.  The per-rank decode path
        // (moePerRank -> ensureExpertCached -> launchQGemv) enqueues rank r's
        // GEMVs on g_stream right after this fill; if the fill raced ahead on
        // the copy engine, a LATER rank's miss could H2D the SAME device slot
        // (two experts mapping to the same 2-way set + LRU victim) while rank
        // r's GEMVs were still reading it — a timing-dependent torn read.
        // Measured on Ornith-1.5-35B-Q4_K_M with TINYCODER_MOE_CACHE=1:
        // ModelTest.KVCacheClear maxDiff 1.92-2.04 VARYING across runs (fwd1
        // all-misses tears vs fwd3 all-hits clean) while default/Option-B mode
        // is bit-identical (maxDiff 0.0); ModelTest.CompareBatchVsSequential
        // inflated to 16.28 with top-1 still matching.  The g_stream2 + event-
        // wait design only guarantees RAW (this rank's GEMV after its own
        // fill); it does NOT order a later rank's eviction AFTER an earlier
        // rank's GEMVs.  One stream (g_stream) gives the total order
        // fill_r -> GEMV_r -> fill_{r+1} -> GEMV_{r+1}, so any eviction of a
        // slot read by rank r happens after rank r's GEMVs complete.  The
        // BATCHED fast path (Q8_0 experts) keeps its g_stream2 pipeline: its
        // fills are all enqueued BEFORE any GEMV and the slot-conflict guard
        // falls back here whenever two ranks would share a slot base, so it
        // cannot tear.
        cudaMemcpyAsync(const_cast<uint8_t *>(gBase), expertStage_[st],
                        static_cast<size_t>(slotBytes), cudaMemcpyHostToDevice,
                        g_stream);
        // Record on g_stream so the ring's drain-before-refill (above) waits
        // for THIS copy before the stage is reused.
        cudaEventRecord(expertStageEv_[st], g_stream);
        expertCacheTag_[L][set][victim] = static_cast<int32_t>(e);
        if (trMiss) {
            double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - tMiss0)
                                .count();
            std::fprintf(stderr,
                         "[miss] L=%u e=%u set=%u way=%u fillEnq=%.3f ms "
                         "slotBytes=%llu\n",
                         L, e, set, victim, ms,
                         static_cast<unsigned long long>(slotBytes));
        }
        // TINYCODER_DUMP_MOE=1: byte-compare the JUST-COPIED gate slice
        // against the host rows for the same expert (L==0 only).  This is the
        // decisive check for "cache slice content wrong" vs "kernel wrong":
        // if the first bytes differ, the H2D/slot layout feeds wrong weight
        // bytes into the expert GEMVs.
        {
            const char *dnC = std::getenv("TINYCODER_DUMP_MOE");
            if (L == 0 && dnC != nullptr && dnC[0] != '\0' && dnC[0] != '0') {
                cudaStreamSynchronize(g_stream);
                const size_t cmpN = 2 * 34;// 2 Q8_0 blocks (row 0, col 0..63)
                std::vector<uint8_t> dev(cmpN), host(cmpN);
                cudaMemcpy(dev.data(), gBase, cmpN, cudaMemcpyDeviceToHost);
                // Compare against the PER-LAYER host blob (layer 0) — the
                // global expertHostGate_ is only a fallback now.
                const uint8_t *hsrc = srcGate +
                                      static_cast<uint64_t>(e) * expertFF_ *
                                              expertRowBytesGate_;
                std::memcpy(host.data(), hsrc, cmpN);
                size_t ndiff = 0;
                size_t firstDiff = cmpN;
                for (size_t i = 0; i < cmpN; ++i) {
                    if (dev[i] != host[i]) {
                        if (ndiff == 0) firstDiff = i;
                        ++ndiff;
                    }
                }
                std::fprintf(stderr,
                             "[cache cmp] L=0 set=%u way=%u e=%u dev[0..3]=%02x%02x"
                             "%02x%02x host[0..3]=%02x%02x%02x%02x "
                             "ndiff=%zu firstDiff=%zu\n",
                             set, victim, e, dev[0], dev[1], dev[2], dev[3],
                             host[0], host[1], host[2], host[3], ndiff,
                             firstDiff);
            }
        }
        return gBase;
    }

    // ==========================================================================
    // ---- Static resident-expert offload (Option D) ----
    //
    // All trunks on the GPU + a FIXED top-K most-used experts per layer
    // permanently resident in per-layer arenas.  The adapter (ModelGPU.cpp)
    // computes the policy from a TINYCODER_MOE_PROFILE=1 profiling pass and
    // stores it via setResidentPolicy; buildResidentArenas() then allocates
    // the arenas and H2Ds the resident experts' packed gate/up/down slices
    // ONCE (no LRU, no per-token PCIe streaming of resident weights).  At
    // inference time the MoE block splits the 8 routed ranks: resident ranks
    // run on the GPU from the arenas (scheduleResidentExperts[/Batch]), the
    // NON-resident ranks run on the CPU (the adapter callback computes them
    // into moeDownTableHost_ rank-major, H2D'd once per layer), and the
    // combined table accumulates in the exact reference rank order.
    // ==========================================================================

    bool GPUModel::buildResidentArenas(std::string &errMsg) {
        // Geometry is recorded by the hybrid upload (same fields as the expert
        // cache); without it there is nothing to build.
        if (expertFF_ == 0 || expertHM_ == 0 || expertCount_ == 0 ||
            expertHostGate_ == nullptr || expertHostDown_ == nullptr) {
            errMsg = "buildResidentArenas: missing host expert geometry "
                     "(hybrid upload did not record it)";
            return false;
        }
        // The resident fast path runs the Q8_0 integer pipeline
        // (launchQuantQ8_0x32 + kQGemvQ8_0xQ8K_BatchGU/DownR, bit-exact with
        // the CPU Q8_0 reference).
        //
        // NOTE (2026-09-17): the Q4_K/Q6_K pipeline (kQuantizeQ8K +
        // kQGemvKxQ8K_BatchGU<kTypeQ4K> + kSiluMulBatch + kQuantizeQ8K_Batch +
        // kQGemvKxQ8K_BatchDownR<kTypeQ6K>) is bit-exact (verified by
        // GPUCpuCompareTest.Q4K_Q6KKernelParityWithCPU) but is DISABLED here:
        // measured on Ornith-1.5-35B-Q4_K_M the Option-D resident split is a
        // 4.5x REGRESSION vs Option B (0.32 vs 1.45 tg tok/s) because the
        // ~6-8 non-resident experts per layer run on the CPU with the
        // per-member scalar path (no grouped-batch batching), so decode is
        // slower than the fully-grouped Option-B path.  Q4_K_M stays on
        // Option B permanently.  Other expert quant types (IQ2_XXS/IQ4_XS/
        // Q5_K etc.) would be misread as Q8_0 blocks (silent garbage), so
        // Option D stays disabled for them too; the caller keeps the
        // CPU-expert (Option B) placement in all of those cases.
        const bool residentTypesOk = expertType_ == kTypeQ8_0;
        if (!residentTypesOk) {
            errMsg = "buildResidentArenas: expert type " +
                     std::to_string(expertType_) + " / down type " +
                     std::to_string(expertDownType_) +
                     " not supported by the resident fast path (Q8_0 only); "
                     "disabling Option D";
            return false;
        }
        // The policy vectors (adapter-side, public) may not be sized yet;
        // force them to the canonical [kResidentLayers] shape so every
        // downstream [L] index is safe.
        if (moeResidentIds_.size() < kResidentLayers) {
            moeResidentIds_.resize(kResidentLayers);
        }
        if (moeResidentCount_.size() < kResidentLayers) {
            moeResidentCount_.assign(kResidentLayers, 0);
        }
        if (moeDownTableHost_.size() < 2) moeDownTableHost_.resize(2);
        const uint32_t nGpu = geom_.numGpuLayers ? geom_.numGpuLayers
                                                 : geom_.numLayers;
        // Per-resident-expert slice geometry (identical to the expert cache
        // slot layout: gate [expertFF][H], up [expertFF][H], down [H][expertFF]).
        const uint64_t gateSlice = static_cast<uint64_t>(expertFF_) *
                                   expertRowBytesGate_;
        const uint64_t downSlice = static_cast<uint64_t>(expertHM_) *
                                   expertRowBytesDown_;
        if (gateSlice == 0 || downSlice == 0) {
            errMsg = "buildResidentArenas: zero row geometry";
            return false;
        }
        residentRowBytesGate_ = expertRowBytesGate_;
        residentRowBytesDown_ = expertRowBytesDown_;
        // Pre-clear the maps.
        for (uint32_t L = 0; L < kResidentLayers; ++L) {
            for (uint32_t i = 0; i < 256; ++i) residentIdxMap_[L][i] = -1;
        }
        residentActive_ = false;
        // Resolve the per-layer resident counts from the stored policy, then
        // allocate + H2D.  The policy vectors are indexed [layer]; a layer with
        // no entry or an empty id list has no resident experts.
        for (uint32_t L = 0; L < nGpu && L < moeResidentIds_.size(); ++L) {
            const auto &ids = moeResidentIds_[L];
            const uint32_t cnt = std::min<uint32_t>(
                    static_cast<uint32_t>(ids.size()), kResidentMaxExperts);
            if (cnt == 0) {
                moeResidentCount_[L] = 0;
                continue;
            }
            // Sanity: expert ids must be in range.
            bool ok = true;
            for (uint32_t i = 0; i < cnt; ++i) {
                if (ids[i] >= expertCount_) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                errMsg = "buildResidentArenas: resident expert id out of range";
                return false;
            }
            const uint64_t arenaBytes = static_cast<uint64_t>(cnt) *
                                        (2 * gateSlice + downSlice);
            uint8_t *arena = nullptr;
            cudaError_t e = cudaMalloc(&arena, static_cast<size_t>(arenaBytes));
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMalloc(residentArena): ") +
                         cudaGetErrorString(e);
                // Free what we already built.
                for (uint32_t l2 = 0; l2 < L; ++l2) {
                    if (residentArena_[l2]) {
                        cudaFree(residentArena_[l2]);
                        residentArena_[l2] = nullptr;
                    }
                    residentArenaBytes_[l2] = 0;
                    moeResidentCount_[l2] = 0;
                }
                return false;
            }
            residentArena_[L] = arena;
            residentArenaBytes_[L] = arenaBytes;
            moeResidentCount_[L] = cnt;
            // Per-layer H2D source (mirror the expert cache's per-layer blob
            // selection — the hard-won 2026-09-14 lesson).
            const uint8_t *srcGate = expertCache_[L].hostGate
                                             ? expertCache_[L].hostGate
                                             : expertHostGate_;
            const uint8_t *srcUp = expertCache_[L].hostUp
                                           ? expertCache_[L].hostUp
                                           : expertHostUp_;
            const uint8_t *srcDown = expertCache_[L].hostDown
                                             ? expertCache_[L].hostDown
                                             : expertHostDown_;
            for (uint32_t i = 0; i < cnt; ++i) {
                const uint32_t e = ids[i];
                // Arena layout (slot-major, same as the cache):
                //   slot i: [gate][up][down]
                const uint8_t *slotBase =
                        arena + static_cast<size_t>(i) * (2 * gateSlice + downSlice);
                const uint8_t *gateBase = slotBase;
                const uint8_t *upBase = slotBase + gateSlice;
                const uint8_t *downBase = slotBase + 2 * gateSlice;
                cudaMemcpyAsync(const_cast<uint8_t *>(gateBase),
                                srcGate + static_cast<uint64_t>(e) * gateSlice,
                                static_cast<size_t>(gateSlice),
                                cudaMemcpyHostToDevice, g_stream);
                cudaMemcpyAsync(const_cast<uint8_t *>(upBase),
                                srcUp + static_cast<uint64_t>(e) * gateSlice,
                                static_cast<size_t>(gateSlice),
                                cudaMemcpyHostToDevice, g_stream);
                cudaMemcpyAsync(const_cast<uint8_t *>(downBase),
                                srcDown + static_cast<uint64_t>(e) * downSlice,
                                static_cast<size_t>(downSlice),
                                cudaMemcpyHostToDevice, g_stream);
                residentGateBase_[L][i] = gateBase;
                residentUpBase_[L][i] = upBase;
                residentDownBase_[L][i] = downBase;
                residentIdxMap_[L][e] = static_cast<int32_t>(i);
            }
            residentActive_ = true;
        }
        // Build the host bitmask mirrors the CPU callback consumes:
        // residentMaskHost_[L] = [ceil(expertCount/64)] words, bit e set ==
        // resident.  The adapter also keeps its own copy (m.moeResidentMask_);
        // this is the DRIVER's mirror, indexed [L].
        const uint32_t maskWords = (expertCount_ + 63u) / 64u;
        residentMaskHost_.assign(kResidentLayers, std::vector<uint64_t>(maskWords, 0));
        for (uint32_t L = 0; L < nGpu && L < moeResidentIds_.size(); ++L) {
            const auto &ids = moeResidentIds_[L];
            const uint32_t cnt = moeResidentCount_[L];
            for (uint32_t i = 0; i < cnt; ++i) {
                const uint32_t e = ids[i];
                residentMaskHost_[L][e / 64u] |= (uint64_t(1) << (e % 64u));
            }
        }
        // Grow the dispatch tables ONCE (worst-case capacity): the wq table
        // holds 3*maxResident pointers; the rank-flag array is [expertUsed].
        const uint32_t eu = std::max(expertUsed_, 1u);
        residentMaxPerLayer_ = eu;
        const uint64_t needWq = static_cast<uint64_t>(std::max(kResidentMaxExperts, eu)) * 3u *
                                sizeof(uint8_t *);
        if (residentRankWq_ == nullptr || residentRankWqBytes_ < needWq) {
            if (residentRankWq_) cudaFree(residentRankWq_);
            residentRankWq_ = nullptr;
            residentRankWqBytes_ = 0;
            cudaError_t eW = cudaMalloc(&residentRankWq_, static_cast<size_t>(needWq));
            if (eW != cudaSuccess) {
                errMsg = std::string("cudaMalloc(residentRankWq): ") +
                         cudaGetErrorString(eW);
                return false;
            }
            residentRankWqBytes_ = needWq;
        }
        const uint64_t needRankOf = static_cast<uint64_t>(eu) * sizeof(int32_t);
        if (residentRankOf_ == nullptr || residentRankOfBytes_ < needRankOf) {
            if (residentRankOf_) cudaFree(residentRankOf_);
            residentRankOf_ = nullptr;
            residentRankOfBytes_ = 0;
            cudaError_t eR = cudaMalloc(&residentRankOf_, static_cast<size_t>(needRankOf));
            if (eR != cudaSuccess) {
                errMsg = std::string("cudaMalloc(residentRankOf): ") +
                         cudaGetErrorString(eR);
                return false;
            }
            residentRankOfBytes_ = needRankOf;
        }
        const uint64_t needWgt = static_cast<uint64_t>(std::max(static_cast<uint32_t>(
                                                                        kResidentMaxExperts),
                                                                eu)) *
                                 sizeof(float);
        if (residentWgt_ == nullptr || residentWgtBytes_ < needWgt) {
            if (residentWgt_) cudaFree(residentWgt_);
            residentWgt_ = nullptr;
            residentWgtBytes_ = 0;
            cudaError_t eG = cudaMalloc(&residentWgt_, static_cast<size_t>(needWgt));
            if (eG != cudaSuccess) {
                errMsg = std::string("cudaMalloc(residentWgt): ") +
                         cudaGetErrorString(eG);
                return false;
            }
            residentWgtBytes_ = needWgt;
        }
        // Drain the upload stream so the arenas are fully populated before the
        // first inference forward touches them.
        cudaStreamSynchronize(g_stream);
        // Debug: report the policy footprint.
        {
            const char *dnR = std::getenv("TINYCODER_DUMP_MOE");
            if (dnR != nullptr && dnR[0] != '\0' && dnR[0] != '0') {
                uint64_t totalBytes = 0;
                for (uint32_t L = 0; L < kResidentLayers; ++L) {
                    totalBytes += residentArenaBytes_[L];
                }
                std::fprintf(stderr,
                             "[resident] built %u layers, %u resident max per "
                             "layer, arena bytes/layer=%llu, total=%.2f MB, "
                             "expertCount=%u expertFF=%u\n",
                             nGpu, kResidentMaxExperts,
                             static_cast<unsigned long long>(
                                     2 * gateSlice + downSlice),
                             static_cast<double>(totalBytes) / (1024.0 * 1024.0),
                             expertCount_, expertFF_);
            }
        }
        return true;
    }

    // ---- Decode fast path (seqLen == 1): gather the resident slots the
    // CURRENT layer routed (via moeRouterIdxHost_ + residentIdxMap_), build
    // the compact slot table + rank-flag table, then run the same 4 kernels as
    // the batched decode path but with the ranks COMPACTED (residentCount) and
    // the down rows written to their FULL-rank offsets via
    // kQGemvQ8_0xQ8K_BatchDownR.  downOut is the COMBINED table
    // [rank * seqLen * H] — zeroed already, non-resident slots filled by the
    // CPU.  Resident ranks overwrite their own slots; accumulation then reads
    // the whole table in rank order.
    void GPUModel::scheduleResidentExperts(uint32_t L, float *downOut) {
        if (!residentActive_) return;
        if (L >= kResidentLayers) return;
        const uint32_t cnt = moeResidentCount_[L];
        if (cnt == 0) return;
        const uint32_t eu = expertUsed_;
        if (eu == 0) return;
        DeviceScratch &s = scratch_;
        const uint32_t H = expertHM_;
        // Compact per-rank tables: resident slots only (in routed-rank order).
        std::vector<const uint8_t *> slotWqHost;
        slotWqHost.reserve(3u * cnt);
        std::vector<int32_t> rankOfHost;
        rankOfHost.reserve(cnt);
        for (uint32_t r = 0; r < eu; ++r) {
            const int32_t e0 = moeRouterIdxHost_.data()[r];
            if (e0 < 0) continue;
            const uint32_t ue = static_cast<uint32_t>(e0);
            if (ue >= expertCount_) continue;
            const int32_t slot = residentIdxMap_[L][ue];
            if (slot < 0) continue;// non-resident: CPU already handled it
            const uint32_t us = static_cast<uint32_t>(slot);
            slotWqHost.push_back(residentGateBase_[L][us]);
            slotWqHost.push_back(residentUpBase_[L][us]);
            slotWqHost.push_back(residentDownBase_[L][us]);
            rankOfHost.push_back(static_cast<int32_t>(r));
        }
        const uint32_t jn = static_cast<uint32_t>(rankOfHost.size());
        if (jn == 0) return;// all ranks non-resident: nothing to do
        // Upload the compact tables (grow-on-demand).
        const uint64_t wqBytes = static_cast<uint64_t>(jn) * 3u * sizeof(uint8_t *);
        if (residentRankWqBytes_ < wqBytes) {
            if (residentRankWq_) cudaFree(residentRankWq_);
            residentRankWq_ = nullptr;
            residentRankWqBytes_ = 0;
            cudaMalloc(&residentRankWq_, static_cast<size_t>(wqBytes));
            residentRankWqBytes_ = wqBytes;
        }
        cudaMemcpyAsync(residentRankWq_, slotWqHost.data(), wqBytes,
                        cudaMemcpyHostToDevice, g_stream);
        const uint64_t rankOfBytes = static_cast<uint64_t>(jn) * sizeof(int32_t);
        if (residentRankOfBytes_ < rankOfBytes) {
            if (residentRankOf_) cudaFree(residentRankOf_);
            residentRankOf_ = nullptr;
            residentRankOfBytes_ = 0;
            cudaMalloc(&residentRankOf_, static_cast<size_t>(rankOfBytes));
            residentRankOfBytes_ = rankOfBytes;
        }
        cudaMemcpyAsync(residentRankOf_, rankOfHost.data(), rankOfBytes,
                        cudaMemcpyHostToDevice, g_stream);
        // Quantize the norm once into q8kN_ (shared activation for ALL
        // resident GU GEMVs).  Two activation layouts:
        //   * Q8_0 experts: 32-wide Q8_32 blocks (kQ8_0x32_STRIDE=64) via
        //     launchQuantQ8_0x32 ;
        //   * Q4_K gate/up (+ Q6_K down): 256-wide Q8_K blocks
        //     (kQ8K_STRIDE=304) via kQuantizeQ8K .
        const uint32_t gBlocks = (expertType_ == kTypeQ5_0 || expertType_ == kTypeQ8_0)
                                         ? (H + 31) / 32
                                         : (H + 255) / 256;
        const uint32_t dBlocks = (expertType_ == kTypeQ5_0 || expertType_ == kTypeQ8_0)
                                         ? (expertFF_ + 31) / 32
                                         : (expertFF_ + 255) / 256;
        // Gate/up are the type recorded in expertType_ (Q8_0 or Q4_K); the
        // DOWN matrix is expertDownType_ (Q8_0 or Q6_K).  buildResidentArenas
        // gates the supported pairs, so only these two pipelines run here.
        const bool isQ4 = (expertType_ == kTypeQ4K);
        const uint32_t actStride = isQ4 ? kQ8K_STRIDE : kQ8_0x32_STRIDE;
        const uint64_t needN = static_cast<uint64_t>(gBlocks) * actStride;
        if (q8kN_ == nullptr || q8kNBytes_ < needN) {
            if (q8kN_) cudaFree(q8kN_);
            q8kN_ = nullptr;
            q8kNBytes_ = 0;
            cudaMalloc(&q8kN_, static_cast<size_t>(needN));
            q8kNBytes_ = needN;
        }
        const uint8_t *const *wqTable =
                reinterpret_cast<const uint8_t *const *>(residentRankWq_);
        const uint32_t combinedRows = 2u * expertFF_;
        const uint32_t downStrideB =
                static_cast<uint32_t>(dBlocks) * actStride;
        const uint64_t needDownP = static_cast<uint64_t>(jn) * downStrideB;
        if (moeQ8kDownP_ == nullptr || moeQ8kDownPBytes_ < needDownP) {
            if (moeQ8kDownP_) cudaFree(moeQ8kDownP_);
            moeQ8kDownP_ = nullptr;
            moeQ8kDownPBytes_ = 0;
            cudaMalloc(&moeQ8kDownP_, static_cast<size_t>(needDownP));
            moeQ8kDownPBytes_ = needDownP;
        }
        if (isQ4) {
            // ---- Q4_K gate/up (144 B/block) + Q6_K down (210 B/block) with
            // a 256-wide Q8_K activation; bit-exact with the CPU
            // matMulVecBatchQ4K/Q6K_Q8K_AVX2 kernels. ----
            kQuantizeQ8K<<<gBlocks, 256, 0, g_stream>>>(s.norm, q8kN_, H);
            kQGemvKxQ8K_BatchGU<kTypeQ4K><<<dim3((combinedRows + 7) / 8, jn),
                                            dim3(32, 8), 0, g_stream>>>(
                    wqTable, q8kN_, s.moeGateUpP, s.moeGateUp2P, expertFF_,
                    gBlocks, expertRowBytesGate_, combinedRows);
            kSiluMulBatch<<<dim3((expertFF_ + 255) / 256, jn), 256, 0,
                            g_stream>>>(s.moeGateUpP, s.moeGateUp2P, expertFF_,
                                        jn);
            kQuantizeQ8K_Batch<<<dim3(dBlocks, jn), 256, 0, g_stream>>>(
                    s.moeGateUpP, moeQ8kDownP_, expertFF_, jn, downStrideB);
            kQGemvKxQ8K_BatchDownR<kTypeQ6K><<<dim3((H + 7) / 8, jn),
                                               dim3(32, 8), 0, g_stream>>>(
                    wqTable, moeQ8kDownP_, downOut,
                    reinterpret_cast<const int32_t *>(residentRankOf_), H,
                    dBlocks, expertRowBytesDown_, downStrideB, H);
        } else {
            // ---- Q8_0 gate/up/down (34 B/32-wide block) with a 32-wide
            // Q8_32 activation (the original Option D pipeline). ----
            launchQuantQ8_0x32(s.norm, q8kN_, H);
            kQGemvQ8_0xQ8K_BatchGU<<<dim3((combinedRows + 7) / 8, jn),
                                     dim3(32, 8), 0, g_stream>>>(
                    wqTable, q8kN_, s.moeGateUpP, s.moeGateUp2P, expertFF_,
                    gBlocks, expertRowBytesGate_, combinedRows);
            kSiluMulBatch<<<dim3((expertFF_ + 255) / 256, jn), 256, 0,
                            g_stream>>>(s.moeGateUpP, s.moeGateUp2P, expertFF_,
                                        jn);
            kQuantizeQ8_0x32_Batch<<<dim3(dBlocks, jn), 32, 0, g_stream>>>(
                    s.moeGateUpP, moeQ8kDownP_, expertFF_, jn, downStrideB);
            kQGemvQ8_0xQ8K_BatchDownR<<<dim3((H + 7) / 8, jn), dim3(32, 8), 0,
                                        g_stream>>>(
                    wqTable, moeQ8kDownP_, downOut,
                    reinterpret_cast<const int32_t *>(residentRankOf_), H,
                    dBlocks, expertRowBytesDown_, downStrideB, H);
        }
    }

    // ---- Prefill path (seqLen > 1).  Each (token, rank) may route a
    // DIFFERENT (resident vs non-resident) expert, so the resident GPU work is
    // per-(rank, token): a token that routed to a resident expert computes its
    // down row into [rankOf][seqLen][H]; a non-resident token is skipped (the
    // CPU callback filled its rows AND rankOf tells the accumulation kernel
    // the full rank order).  We gather the resident ranks only and reuse the
    // SAME compact 4-kernel pipeline per (rank, token-group); the Q8_0 fast
    // path matches the CPU reference bit-for-bit by construction.
    void GPUModel::scheduleResidentExpertsBatch(uint32_t L, float *downOut,
                                                uint32_t seqLen) {
        if (!residentActive_) return;
        if (L >= kResidentLayers) return;
        const uint32_t cnt = moeResidentCount_[L];
        if (cnt == 0) return;
        const uint32_t eu = expertUsed_;
        if (eu == 0) return;
        // NOTE: named `scr` — the token loop below uses `s` for the token index.
        DeviceScratch &scr = scratch_;
        const uint32_t H = expertHM_;
        const uint32_t seqH = static_cast<uint32_t>(seqLen) * H;
        const uint32_t gBlocks = (expertType_ == kTypeQ5_0 || expertType_ == kTypeQ8_0)
                                         ? (H + 31) / 32
                                         : (H + 255) / 256;
        const uint32_t dBlocks = (expertType_ == kTypeQ5_0 || expertType_ == kTypeQ8_0)
                                         ? (expertFF_ + 31) / 32
                                         : (expertFF_ + 255) / 256;
        const uint8_t *const *wqTable =
                reinterpret_cast<const uint8_t *const *>(residentRankWq_);
        // Lower the per-(token, rank) routed expert ids to their RESIDENT slot
        // (-1 = non-resident: the CPU callback already filled that down row).
        std::vector<int32_t> tokSlot(static_cast<size_t>(seqLen) * eu);
        for (uint32_t s = 0; s < seqLen; ++s) {
            const int32_t *row = moeRouterIdxHost_.data() +
                                 static_cast<size_t>(s) * eu;
            for (uint32_t r = 0; r < eu; ++r) {
                const int32_t e0 = row[r];
                const uint32_t ue = (e0 >= 0) ? static_cast<uint32_t>(e0) : 0u;
                tokSlot[static_cast<size_t>(s) * eu + r] =
                        (e0 >= 0 && ue < expertCount_)
                                ? residentIdxMap_[L][ue]
                                : -1;
            }
        }
        // Per token: quantize that token's norm ONCE into q8kN_, then for each
        // resident rank that routed to a resident slot run the exact same
        // compact 4-kernel pipeline as the decode path (one activation per
        // launch set).  The down row is written at [rank][s*H] in the combined
        // rank-major table; the accumulation kernel later reads all ranks in
        // the exact reference order.  This is CORRECTNESS-first (the decode
        // path has the fully-batched version; prefill is bounded by the trunk
        // GEMMs anyway).
        // Gate/up are expertType_ (Q8_0 or Q4_K), down is expertDownType_;
        // the 256-wide Q8_K activation layout for the Q4_K pipeline is
        // kQ8K_STRIDE vs the 32-wide Q8_32 (kQ8_0x32_STRIDE) for Q8_0.
        const bool isQ4 = (expertType_ == kTypeQ4K);
        const uint32_t actStride = isQ4 ? kQ8K_STRIDE : kQ8_0x32_STRIDE;
        for (uint32_t s = 0; s < seqLen; ++s) {
            const float *normRow = scr.norm + static_cast<size_t>(s) * H;
            const uint64_t needN = static_cast<uint64_t>(gBlocks) * actStride;
            if (q8kN_ == nullptr || q8kNBytes_ < needN) {
                if (q8kN_) cudaFree(q8kN_);
                q8kN_ = nullptr;
                q8kNBytes_ = 0;
                cudaMalloc(&q8kN_, static_cast<size_t>(needN));
                q8kNBytes_ = needN;
            }
            if (isQ4) {
                kQuantizeQ8K<<<gBlocks, 256, 0, g_stream>>>(normRow, q8kN_, H);
            } else {
                launchQuantQ8_0x32(normRow, q8kN_, H);
            }
            // Per resident rank, if token s routed to its slot, compute.
            for (uint32_t r = 0; r < eu; ++r) {
                const int32_t slot = tokSlot[static_cast<size_t>(s) * eu + r];
                if (slot < 0) continue;
                const uint32_t us = static_cast<uint32_t>(slot);
                const uint8_t *wq[3] = {residentGateBase_[L][us],
                                        residentUpBase_[L][us],
                                        residentDownBase_[L][us]};
                // ONE-entry wq table (rank j=0 compact) + rankOf [r].
                const uint8_t *const wqPtr[3] = {wq[0], wq[1], wq[2]};
                cudaMemcpyAsync(residentRankWq_, wqPtr, 3 * sizeof(uint8_t *),
                                cudaMemcpyHostToDevice, g_stream);
                const int32_t rankOfOne[1] = {static_cast<int32_t>(r)};
                cudaMemcpyAsync(residentRankOf_, rankOfOne, sizeof(int32_t),
                                cudaMemcpyHostToDevice, g_stream);
                const uint32_t combinedRows = 2u * expertFF_;
                if (isQ4) {
                    kQGemvKxQ8K_BatchGU<kTypeQ4K><<<dim3((combinedRows + 7) / 8, 1),
                                                    dim3(32, 8), 0, g_stream>>>(
                            wqTable, q8kN_, scr.moeGateUpP, scr.moeGateUp2P,
                            expertFF_, gBlocks, expertRowBytesGate_,
                            combinedRows);
                } else {
                    kQGemvQ8_0xQ8K_BatchGU<<<dim3((combinedRows + 7) / 8, 1),
                                             dim3(32, 8), 0, g_stream>>>(
                            wqTable, q8kN_, scr.moeGateUpP, scr.moeGateUp2P,
                            expertFF_, gBlocks, expertRowBytesGate_,
                            combinedRows);
                }
                kSiluMulBatch<<<dim3((expertFF_ + 255) / 256, 1), 256, 0, g_stream>>>(
                        scr.moeGateUpP, scr.moeGateUp2P, expertFF_, 1);
                const uint64_t needP = static_cast<uint64_t>(dBlocks) * actStride;
                if (moeQ8kDownPBytes_ < needP) {
                    if (moeQ8kDownP_) cudaFree(moeQ8kDownP_);
                    moeQ8kDownP_ = nullptr;
                    moeQ8kDownPBytes_ = 0;
                    cudaMalloc(&moeQ8kDownP_, static_cast<size_t>(needP));
                    moeQ8kDownPBytes_ = needP;
                }
                if (isQ4) {
                    kQuantizeQ8K_Batch<<<dim3(dBlocks, 1), 256, 0, g_stream>>>(
                            scr.moeGateUpP, moeQ8kDownP_, expertFF_, 1,
                            static_cast<uint32_t>(dBlocks * kQ8K_STRIDE));
                } else {
                    kQuantizeQ8_0x32_Batch<<<dim3(dBlocks, 1), 32, 0, g_stream>>>(
                            scr.moeGateUpP, moeQ8kDownP_, expertFF_, 1,
                            static_cast<uint32_t>(dBlocks * kQ8_0x32_STRIDE));
                }
                // The down kernel writes out[rankOf[j]*outStride + row]; pass
                // the FULL table base + outStride = seqH so the rank row lands
                // at [r*seqH + row].  The per-token row within the rank slab
                // is NOT explicit here — this path computes ONE token per
                // launch set, and the kernel's row index (blockIdx.x*8+ty)
                // already IS the token's H offset, so out[r*seqH + row] is the
                // exact token row.  (Passing downRow with outStride=H would
                // double-add the rank.)
                // Rank-slab base = downOut + this token's in-rank offset; the
                // kernel adds rankOf[j]*outStride (= r*seqH) + row, so the row
                // lands at [r*seqH + s*H + row] — the exact combined-table
                // slot the rank-order accumulation reads for token s.  Passing
                // downOut alone with outStride=seqH would write token 0's rows
                // for EVERY token (later tokens' resident contributions would
                // be lost from the accumulate).
                if (isQ4) {
                    kQGemvKxQ8K_BatchDownR<kTypeQ6K><<<dim3((H + 7) / 8, 1),
                                                       dim3(32, 8), 0, g_stream>>>(
                            wqTable, moeQ8kDownP_,
                            downOut + static_cast<uint64_t>(s) * H,
                            reinterpret_cast<const int32_t *>(residentRankOf_), H,
                            dBlocks, expertRowBytesDown_,
                            static_cast<uint32_t>(dBlocks * kQ8K_STRIDE), seqH);
                } else {
                    kQGemvQ8_0xQ8K_BatchDownR<<<dim3((H + 7) / 8, 1), dim3(32, 8),
                                                0, g_stream>>>(
                            wqTable, moeQ8kDownP_,
                            downOut + static_cast<uint64_t>(s) * H,
                            reinterpret_cast<const int32_t *>(residentRankOf_), H,
                            dBlocks, expertRowBytesDown_,
                            static_cast<uint32_t>(dBlocks * kQ8_0x32_STRIDE),
                            seqH);
                }
            }
        }
    }

    bool GPUModel::dequantMatrixF16(const DeviceMatrix &m, void **outBuf,
                                    std::string &errMsg) {
        if (m.q == nullptr || m.rows == 0 || m.cols == 0) {
            *outBuf = nullptr;
            return true;
        }
        // Persistent fp16 twin fast path (2026-09-29): the uploader built a
        // one-time dequantized copy of this matrix (see uploadMat); return it
        // directly and skip the per-layer streaming dequant entirely.
        if (m.f16Twin != nullptr) {
            *outBuf = m.f16Twin;
            return true;
        }
        // Streaming fallback (no persistent twin — model too big for the VRAM
        // budget, or TINYCODER_PREFILL_TWINS=0).  Cache by source matrix:
        // re-dequantizing the same matrix (e.g. two GEMM calls sharing one
        // layer's matrix) is wasted work.
        if (wF16SrcQ_ == m.q) {
            *outBuf = wF16_;
            return true;
        }
        const uint64_t need = static_cast<uint64_t>(m.rows) * m.cols * sizeof(__half);
        if (need > wF16Bytes_) {
            if (wF16_) cudaFree(wF16_);
            wF16_ = nullptr;
            wF16Bytes_ = 0;
            cudaError_t e = cudaMalloc(&wF16_, need);
            if (e != cudaSuccess) {
                errMsg = std::string("cudaMalloc(wF16): ") + cudaGetErrorString(e);
                return false;
            }
            wF16Bytes_ = need;
        }
        launchDequantF16(m.type, m.q, wF16_, m.rows, m.cols, m.rowBytes,
                         m.blocksPerRow);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            errMsg = std::string("launchDequantF16: ") + cudaGetErrorString(e);
            return false;
        }
        wF16SrcQ_ = m.q;
        *outBuf = wF16_;
        return true;
    }

    // Prism forward activation transform: GDN value-head permutation (ssm_out
    // only) -> per-feature sign flip -> blockwise normalized Hadamard, writing
    // the folded activation into the shared device scratch and returning it.
    // Returns `x` unchanged when the model is not Prism-folded or the transform
    // is disabled.  No-op for widths with no sign vector and no permutation.
    const float *GPUModel::prismFoldForward(const float *x, uint32_t cols,
                                            bool gdnPerm) {
        if (x == nullptr) return x;
        if (geom_.prismFold == 0 || prismFoldBuf_ == nullptr) return x;
        if (prismBlock_ == 0 || cols == 0) return x;
        const float *signs = prismSignsFor(cols);
        const bool perm = gdnPerm && prismGdnVGrouped_ && prismRep_ > 1u &&
                          prismHeadV_ > 0u && prismNk_ > 0u;
        if (signs == nullptr && !perm) return x;// identity transform
        const uint32_t grid = (cols + prismBlock_ - 1u) / prismBlock_;
        kPrismFold<<<grid, 256, 0, g_stream>>>(
                x, prismFoldBuf_, cols, signs, perm ? prismHeadV_ : 0u,
                perm ? prismNk_ : 0u, perm ? prismRep_ : 0u);
        return prismFoldBuf_;
    }

    bool GPUModel::forwardQwen35Prefix(const std::vector<int32_t> &tokens,
                                       bool computeAllLogits, float *logitsOut,
                                       std::string &errMsg) {
        if (!allocated_) {
            errMsg = "GPU model not uploaded";
            return false;
        }
        if (tokens.empty()) return true;
        if (geom_.architecture != 1) {
            errMsg = "forwardQwen35Prefix called for non-qwen35 geometry";
            return false;
        }
        uint32_t seqLen = static_cast<uint32_t>(tokens.size());
        uint32_t nGpu = geom_.numGpuLayers;
        if (nGpu == 0) nGpu = geom_.numLayers;
        bool fullOffload = (nGpu == geom_.numLayers);
        const uint32_t H = geom_.hiddenSize, I = geom_.intermediateSize;
        const uint32_t nHeads = geom_.numAttentionHeads;
        const uint32_t nKV = geom_.numKVHeads;
        const uint32_t hd = geom_.headDim;
        const uint32_t qLen = nHeads * hd, kvLen = nKV * hd;
        const uint32_t pos = static_cast<uint32_t>(kvPos_);

        // Qwen35 dense-hybrid geometry.
        const uint32_t dInner = geom_.ssmInnerSize;           // 6144
        const uint32_t headK = geom_.ssmStateSize;            // 128
        const uint32_t nKHeads = geom_.ssmGroupCount;         // 16
        const uint32_t nVHeads = geom_.ssmTimeStepRank;       // 48
        const uint32_t headV = dInner / std::max(nVHeads, 1u);// 128
        const uint32_t keyDim = headK * nKHeads;              // 2048
        const uint32_t valueDim = headV * nVHeads;            // 6144
        const uint32_t convChannels = 2 * keyDim + valueDim;  // 10240
        const uint32_t convK = geom_.ssmConvKernel;           // 4
        const uint32_t qkvDim = 2 * keyDim + valueDim;
        const uint32_t qgDims = nHeads * 2 * hd;

        if (!ensureScratch(seqLen, errMsg)) return false;
        DeviceScratch &s = scratch_;
        cudaMemcpyAsync(s.tokens, tokens.data(), seqLen * sizeof(int32_t),
                        cudaMemcpyHostToDevice, g_stream);
        cudaStreamSynchronize(g_stream);

        // ---- Embedding (K-quant / legacy).  Mirror forward()'s dispatch. ----
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
            } else if (embedType_ == kTypePQ2_0) {
                // Prism ternary (Bonsai): dequantize + INVERSE fold (rotate, then
                // sign) so the latent (rotated) embedding row is restored to the
                // primal basis.  The forward transform is applied to activations
                // later (prismFoldForward).  Signs are per-feature at the hidden
                // width; prismSignsFor(H) is null in identity-sign mode.
                kEmbedDequantPQ2_0<<<seqLen, 256, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize, prismSignsFor(H));
            } else {
                kEmbedDequant<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            }
        }
        // TINYCODER_DUMP_NORM=1: dump the GPU's embedding output (s.hidden)
        // before any layer so the first diverging stage is identifiable against
        // the CPU's [cpuemb] (dense qwen35; mirror of the MoE-side hook).
        {
            const char *dn4 = std::getenv("TINYCODER_DUMP_NORM");
            if (dn4 != nullptr && dn4[0] != '\0' && dn4[0] != '0' && seqLen >= 1) {
                cudaStreamSynchronize(g_stream);
                float e0[8];
                cudaMemcpy(e0, s.hidden, 8 * sizeof(float), cudaMemcpyDeviceToHost);
                double esum = 0.0;
                for (uint32_t i = 0; i < H; ++i) {
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

        // Dense projection helper.  qwen35 projections run quantized kernels —
        // NEVER an fp16 cuBLAS GEMM.
        //
        //   * Prefill (seqLen > 1): the batched Q8_K GEMM
        //     (launchQGemm -> kQuantizeQ8K_Batch + kQGemmKxQ8K) reads each
        //     weight block ONCE per 4-token group and dots it against 4 token
        //     activations, so the weight stream is no longer repeated per
        //     prompt token.  It is bit-identical to the per-token Q8_K
        //     launchQGemv path (same kxQ8KBlockDot per (row, token)), which is
        //     the property that keeps batch == sequential == CPU.  Prism-folded
        //     (PQ2_0) prefill takes the per-token loop (the activation must be
        //     Hadamard-folded first).
        //   * Decode (seqLen == 1) and every type/shape the GEMM does not
        //     cover (Q8_0, IQ4_NL, F32, Q5_0, Prism) fall back to the
        //     per-token launchQGemv loop.
        //   * launchQGemv covers every type these projections use with the
        //     exact CPU-matching float tree (K-quants, Q5_0/Q8_0/IQ4_NL via
        //     kQGemvSmall32, Q8_0 via the integer Q8_32 path, F32 via
        //     kQ35F32Gemv).  The IQ4NLKernelParityWithCPU row scan proves
        //     kQGemvSmall32<IQ4NL> reproduces matMulVecBatchIQ4NL_SIMD to
        //     ~1e-6 abs error on every IQ4_NL matrix in the model.
        //   * The old fp16 batched path (dequantMatrixF16 -> kF32ToF16 ->
        //     cublasGemmEx) fp16-rounded the X-side AND the weights.  fp16
        //     mantissa noise (~2^-11 relative per element) is NOT tolerable in
        //     the recurrent gated-delta-net: it compounds through the
        //     persistent conv/GDN state and the hidden norm diverges ~2% by
        //     layer 0, exploding past layer 8 (ParisPromptLogitsAgree NGL>=16
        //     fails: GPU top-1 248046 vs CPU 248068).  qwen35moe learned the
        //     same lesson (see forwardQwen35MoE's gemv comment) — dense qwen35
        //     must behave identically, which is why the new batched GEMM keeps
        //     the integer Q8_K math instead of fp16 tensor cores.
        const bool gemmPrefill = (seqLen > 1) && (geom_.prismFold == 0);
        auto gemv = [&](const DeviceMatrix &m, const float *x, float *y,
                        uint32_t nRows, uint32_t nCols,
                        const char *what) -> bool {
            if (m.q == nullptr || nRows == 0) return true;
            // Prism: the folded (PQ2_0) weights expect the ACTIVATION in the
            // Hadamard-folded basis H S P x, so transform it first.  Only
            // PQ2_0 tensors are folded (the F32 ssm_alpha/ssm_beta stay primal),
            // and the ssm_out projection additionally needs the GDN value-head
            // permutation.  All launches are on g_stream, so the single shared
            // prismFoldBuf_ is safe: the GPU executes transform -> gemv -> next
            // transform in stream order.
            const bool folded = (geom_.prismFold != 0) && (m.type == kTypePQ2_0);
            const bool gdnPerm = folded && (std::strcmp(what, "ssmOut") == 0);
            // Batched prefill GEMM (bit-exact Q8_K): one weight read per
            // 4-token group.  Grown once and reused across every projection.
            if (gemmPrefill && !folded && nCols == m.blocksPerRow * 256u) {
                const uint64_t actStride =
                        static_cast<uint64_t>(m.blocksPerRow) * kQ8K_STRIDE;
                const uint64_t need = static_cast<uint64_t>(seqLen) * actStride;
                if (q8kPrefill_ == nullptr || q8kPrefillBytes_ < need) {
                    if (q8kPrefill_) cudaFree(q8kPrefill_);
                    q8kPrefill_ = nullptr;
                    q8kPrefillBytes_ = 0;
                    if (cudaMalloc(&q8kPrefill_, need) == cudaSuccess) {
                        q8kPrefillBytes_ = need;
                    }
                }
                if (launchQGemm(m.type, m.q, x, y, seqLen, nRows, nCols,
                                m.rowBytes, m.blocksPerRow, q8kPrefill_,
                                q8kPrefillBytes_)) {
                    return true;
                }
            }
            for (uint32_t st = 0; st < seqLen; ++st) {
                const float *xp = x + static_cast<size_t>(st) * nCols;
                if (folded) xp = prismFoldForward(xp, nCols, gdnPerm);
                launchQGemv(m.type, m.q, xp,
                            y + static_cast<size_t>(st) * nRows, nRows, nCols,
                            m.rowBytes, m.blocksPerRow, q8k_, q8kBytes_);
            }
            return true;
        };

        const float invSqrtV = 1.0f / std::sqrt(static_cast<float>(headV));
        const uint32_t nDims = geom_.ropeDimensionCount > 0
                                       ? geom_.ropeDimensionCount
                                       : 64;

        const uint32_t nRealLayers =
                geom_.numLayers - geom_.nextnPredictLayers;
        for (uint32_t L = 0; L < nGpu && L < nRealLayers; ++L) {
            const DeviceLayer &w = layers_[L];

            // ---- Attention / recurrent block ----
            kRMSNormRow<<<seqLen, 1024, 1024 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.rmsNormAttn, H, seqLen, 1e-6f);

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
                // TINYCODER_DUMP_NORM=1: dump GPU L0 token-0 qkv (post
                // attnQKV GEMM/GEMV) to bisect against the CPU's [cpuqkv].
                {
                    const char *dnR = std::getenv("TINYCODER_DUMP_NORM");
                    if (L == 0 && dnR != nullptr && dnR[0] != '\0' &&
                        dnR[0] != '0') {
                        cudaStreamSynchronize(g_stream);
                        float q0[5];
                        cudaMemcpy(q0, s.qkv, 5 * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        std::fprintf(stderr,
                                     "[gpuqkv] L0T0 qkv[0..4]={%.6f %.6f %.6f "
                                     "%.6f %.6f}\n",
                                     q0[0], q0[1], q0[2], q0[3], q0[4]);
                    }
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
                    // TINYCODER_DUMP_NORM=1: dump GPU L0 token-0 convOut
                    // (post silu(conv1d(qkv))) to bisect against [cpuconv].
                    {
                        const char *dnC = std::getenv("TINYCODER_DUMP_NORM");
                        if (L == 0 && st == 0 && dnC != nullptr &&
                            dnC[0] != '\0' && dnC[0] != '0') {
                            cudaStreamSynchronize(g_stream);
                            float c0[5];
                            cudaMemcpy(c0, s.convOut, 5 * sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            std::fprintf(stderr,
                                         "[gpuconv] L0T0 convOut[0..4]="
                                         "{%.6f %.6f %.6f %.6f %.6f}\n",
                                         c0[0], c0[1], c0[2], c0[3], c0[4]);
                        }
                    }
                    kQ35L2NormRepeat<<<1, 128, 0, g_stream>>>(
                            s.convOut, s.qN, s.kN, s.vN, headK, headV, nKHeads,
                            nVHeads, keyDim, valueDim, 1,
                            l2NormSmem() ? 1u : 0u);
                    const float *betaRow = s.q35Beta + static_cast<size_t>(st) * nVHeads;
                    const float *gateRow = s.q35GateV + static_cast<size_t>(st) * nVHeads;
                    float *gdnRow = s.gdnOut + static_cast<size_t>(st) * valueDim;
                    // TINYCODER_GDN_SEQ=1: sequential per-head loop (parity
                    // bisection, same as the decode site below).
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
            } else {
                // =================== Full attention =======================
                if (!gemv(w.attnQ, s.norm, s.q35QGate, qgDims, H, "attnQ") ||
                    !gemv(w.attnK, s.norm, s.k, kvLen, H, "attnK") ||
                    !gemv(w.attnV, s.norm, s.v, kvLen, H, "attnV")) {
                    return false;
                }
                // Per-head Q RMSNorm (Q slice = first headDim of each 2*headDim
                // fused Q+gate row) and K RMSNorm (dense, in place).
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
                // Only FULL-ATTENTION layers reach here (the recurrent branch
                // above is the alternative), and the cache is slotted by
                // floor(L/interval) — NOT by L (see kvLayerOffset()).
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
                // attnOut *= sigmoid(gate) (gate = second headDim of the fused
                // Q+gate projection).
                kQ35SigmoidGate<<<seqLen, 256, 0, g_stream>>>(
                        s.attnOut, s.q35QGate, seqLen, nHeads, hd);
                if (!gemv(w.attnO, s.attnOut, s.attnProj, H, qLen, "attnO")) {
                    return false;
                }
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.attnProj, seqLen * H);

            // ---- Post-attention RMSNorm + SwiGLU FFN + residual ----
            kRMSNormRow<<<seqLen, 1024, 1024 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.postAttnNorm, H, seqLen, 1e-6f);
            if (!gemv(w.ffnGate, s.norm, s.gate, I, H, "ffnGate") ||
                !gemv(w.ffnUp, s.norm, s.up, I, H, "ffnUp")) {
                return false;
            }
            kSiluMul<<<(seqLen * I + 255) / 256, 256, 0, g_stream>>>(s.gate,
                                                                     s.up,
                                                                     seqLen * I);
            if (!gemv(w.ffnDown, s.gate, s.ffnOut, H, I, "ffnDown")) {
                return false;
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.ffnOut, seqLen * H);
            // TINYCODER_DUMP_LAYERS=1: dump the GPU post-FFN hidden (last
            // token) per offloaded layer, mirroring the CPU's [cpuL n] dump in
            // Model::forward so a GPU-vs-CPU divergence can be localized to the
            // exact layer.  Will NOT be async-safe with the eager store below:
            // the D2H reads for (sq, fnv) are done as a batch via one
            // cudaMemcpy of the last token's row (hiddenSize floats).
            const char *dnL = std::getenv("TINYCODER_DUMP_LAYERS");
            if (dnL != nullptr && dnL[0] != '\0' && dnL[0] != '0') {
                cudaStreamSynchronize(g_stream);
                std::vector<float> hrow(H);
                cudaMemcpy(hrow.data(), s.hidden + (seqLen - 1) * H,
                           H * sizeof(float), cudaMemcpyDeviceToHost);
                double sq = 0.0;
                uint64_t fnv = 1469598103934665603ULL;
                for (uint32_t i = 0; i < H; ++i) {
                    sq += static_cast<double>(hrow[i]) * hrow[i];
                    uint32_t bits;
                    std::memcpy(&bits, &hrow[i], sizeof(bits));
                    fnv ^= bits;
                    fnv *= 1099511628211ULL;
                }
                std::fprintf(stderr,
                             "[gpuL %3u] post-ffn norm=%.6f fnv=%016llx "
                             "first8=[%.6f %.6f %.6f %.6f %.6f %.6f "
                             "%.6f %.6f]\n",
                             L, std::sqrt(sq),
                             static_cast<unsigned long long>(fnv), hrow[0],
                             hrow[1], hrow[2], hrow[3], hrow[4], hrow[5],
                             hrow[6], hrow[7]);
            }
        }

        cudaError_t llErr = cudaStreamSynchronize(g_stream);
        if (llErr != cudaSuccess) {
            errMsg = std::string("qwen35 prefix sync: ") +
                     cudaGetErrorString(llErr);
            return false;
        }
        kvPos_ += seqLen;

        if (!fullOffload) {
            // Partial offload: the caller copies the hidden state (copyHiddenOut)
            // and continues the remaining layers on the CPU against the host KV
            // cache.  kvPos_ has already advanced on the GPU side (the host KV
            // cache position tracks it via kvCache_.pos in Model::forward).
            return true;
        }

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
        // Prism: the (folded) LM head also consumes H S P x.
        const bool lmFolded = (geom_.prismFold != 0) && (lmType == kTypePQ2_0);
        // logitsCapOneRow_: the s.logits scratch holds ONE row; write only the
        // last token (the generation path never requests all rows anyway).
        if (computeAllLogits && !logitsCapOneRow_) {
            for (uint32_t si = 0; si < seqLen; ++si) {
                const float *lh = s.hidden + si * H;
                if (lmFolded) lh = prismFoldForward(lh, H, false);
                launchQGemv(lmType, lmQ, lh,
                            s.logits + si * geom_.vocabSize, geom_.vocabSize, H,
                            lmRowBytes, lmBlocks, q8k_, q8kBytes_);
            }
            cudaMemcpyAsync(logitsOut, s.logits,
                            seqLen * geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        } else {
            uint32_t si = seqLen - 1;
            const float *lh = s.hidden + si * H;
            if (lmFolded) lh = prismFoldForward(lh, H, false);
            launchQGemv(lmType, lmQ, lh, s.logits,
                        geom_.vocabSize, H, lmRowBytes, lmBlocks, q8k_, q8kBytes_);
            cudaMemcpyAsync(logitsOut, s.logits, geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        }
        cudaError_t fin = cudaStreamSynchronize(g_stream);
        if (fin != cudaSuccess) {
            errMsg = std::string("qwen35 lmhead sync: ") +
                     cudaGetErrorString(fin);
            return false;
        }
        return true;
    }

}// namespace tinycoder::gpu

#endif// USE_CUDA
