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

/*
 * TinyCoder CUDA GPU offload engine -- MoE / qwen35 prefix
 *
 * Part of the CUDA offload engine split out of the former GPUCompute.cu.  The
 * shared kernels/device helpers live in GPUComputeInternal.hpp (anonymous
 * namespace, so each translation unit gets its own internal-linkage copy).
 */

#ifdef USE_CUDA

#include "GPUComputeInternal.hpp"
#include "Int8Gemm.hpp"

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
        uint32_t sets = expertCacheSets_ ? expertCacheSets_
                                         : kExpertCacheSetsDefault;
        // ---- AUTO-SHRINK ON OOM (2026-11-10) ------------------------------
        // A large-expert model (Ornith-35B-A3B Q8_0: 3.33 MB per slot) cannot
        // afford 32 sets x 2 ways on every layer of an 11 GB card, and the old
        // path gave up entirely -> the ENTIRE model fell back to CPU experts
        // (26 tok/s prefill, measured).  Halving the set count on allocation
        // failure keeps the GPU expert path alive with a smaller working set;
        // misses then stream from the pinned GGUF blobs at PCIe speed, which
        // is an order of magnitude better than running the experts on the CPU.
        uint64_t slotBytesL0 = 0;// layer-0 slot size (bw selftest below)
        for (;;) {
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
                    for (uint32_t l2 = 0; l2 <= L; ++l2) {
                        if (l2 < kExpertCacheLayers && expertCache_[l2].gateQ)
                            cudaFree(expertCache_[l2].gateQ);
                        if (l2 < kExpertCacheLayers) expertCache_[l2] = ExpertSlot();
                    }
                    cudaGetLastError();
                    // Default ON: a smaller GPU expert cache is always better
                    // than silently degrading the whole model to CPU experts.
                    // TINYCODER_MOE_SHRINK=0 restores the old fail-hard behaviour.
                    static const bool shrinkOk = [] {
                        const char *e = std::getenv("TINYCODER_MOE_SHRINK");
                        return e == nullptr || e[0] != '0';
                    }();
                    if (sets > 2 && shrinkOk) {
                        const uint32_t next = (sets / 2 >= 2) ? (sets / 2) : 2;
                        if (std::getenv("TINYCODER_GPU_VERBOSE") != nullptr)
                            std::fprintf(stderr,
                                         "[cache build] OOM at %u sets -> retry with "
                                         "%u sets\n",
                                         sets, next);
                        sets = next;
                        slotBytesL0 = 0;
                        break;// restart the whole layer loop with fewer sets
                    }
                    errMsg = std::string("cudaMalloc(expertCache): ") +
                             cudaGetErrorString(e);
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
            expertCacheSets_ = sets;
            break;// allocation succeeded at this set count
        }// end auto-shrink retry loop
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

    void GPUModel::beginExpertArenaEpoch() { ++expertArenaEpoch_; }

    void GPUModel::waitExpertArenaFill(uint32_t e) {
        if (e < expertArenaEv_.size() && expertArenaEv_[e] != nullptr)
            cudaStreamWaitEvent(g_stream, expertArenaEv_[e], 0);
    }

    // ---- Full-layer prefill arena (2026-11-09) ------------------------------
    // Direct-mapped resident set: slot e lives at expertArena_ + e*slotBytes,
    // where slotBytes is THIS layer's [gate][up][down] size.  A slot is valid
    // iff its stamp equals the current epoch, and the epoch is bumped once per
    // layer pass, so every layer fills its own experts exactly once no matter
    // how the token loop below interleaves them.  Geometry (row strides, host
    // blob pointers) comes from the same per-layer ExpertSlot record the 2-way
    // cache uses, so both paths slice identical bytes.
    const uint8_t *GPUModel::ensureExpertArenaCached(uint32_t L, uint32_t e,
                                                     std::string &errMsg,
                                                     const uint8_t **downQOut) {
        if (e >= expertCount_) return nullptr;
        ExpertSlot &s = expertCache_[L];
        const uint32_t layGateRowBytes = s.gateRowBytes ? s.gateRowBytes
                                                        : expertRowBytesGate_;
        const uint32_t layDownRowBytes = s.downRowBytes ? s.downRowBytes
                                                        : expertRowBytesDown_;
        const uint64_t gateSlice = static_cast<uint64_t>(expertFF_) *
                                   layGateRowBytes;
        const uint64_t downSlice = static_cast<uint64_t>(expertHM_) *
                                   layDownRowBytes;
        const uint64_t slotBytes = 2 * gateSlice + downSlice;
        const uint64_t needBytes = slotBytes * expertCount_;
        if (expertArena_ == nullptr || expertArenaBytes_ < needBytes) {
            if (expertArena_) cudaFree(expertArena_);
            expertArena_ = nullptr;
            expertArenaBytes_ = 0;
            cudaError_t eA = cudaMalloc(&expertArena_,
                                        static_cast<size_t>(needBytes));
            if (eA != cudaSuccess) {
                errMsg = std::string("cudaMalloc(expertArena): ") +
                         cudaGetErrorString(eA);
                return nullptr;
            }
            expertArenaBytes_ = needBytes;
            expertArenaStamp_.assign(expertCount_, -1);
            if (expertArenaEv_.size() != expertCount_) {
                for (cudaEvent_t ev: expertArenaEv_)
                    if (ev) cudaEventDestroy(ev);
                expertArenaEv_.assign(expertCount_, nullptr);
                for (uint32_t q = 0; q < expertCount_; ++q)
                    cudaEventCreateWithFlags(&expertArenaEv_[q],
                                             cudaEventDisableTiming);
            }
            ++expertArenaEpoch_;// every previously resident slot is gone
        }
        if (expertArenaStamp_.size() != expertCount_)
            expertArenaStamp_.assign(expertCount_, -1);
        uint8_t *slot = expertArena_ + static_cast<uint64_t>(e) * slotBytes;
        if (downQOut) *downQOut = slot + 2 * gateSlice;
        if (expertArenaStamp_[e] == expertArenaEpoch_) return slot;
        // Miss: grow the pinned staging ring if this layer needs more room.
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
                if (posix_memalign(&pp, 4096,
                                   static_cast<size_t>(slotBytes)) != 0) {
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
        const uint8_t *srcGate = s.hostGate ? s.hostGate : expertHostGate_;
        const uint8_t *srcUp = s.hostUp ? s.hostUp : expertHostUp_;
        const uint8_t *srcDown = s.hostDown ? s.hostDown : expertHostDown_;
        // ---- Page-lock this layer's expert blobs ONCE (2026-11-09) --------
        // The generic path stages every miss through a small pinned ring: a
        // host memcpy of the 1.8 MB slot (~0.4 ms at DRAM speed) plus a DMA.
        // Locking the mmap'd blob instead lets the copy engine read the file
        // pages directly, removing the host memcpy from the critical path
        // entirely (measured 10.9-11.7 GB/s H2D from registered pages, vs
        // ~2 GB/s effective with the staging copy).
        if (!s.hostPinned) {
            auto pinRange = [&](const uint8_t *p, uint64_t bytes) -> bool {
                if (p == nullptr || bytes == 0) return false;
                const uintptr_t a =
                        reinterpret_cast<uintptr_t>(p) &
                        ~static_cast<uintptr_t>(4095);
                const uint64_t off = reinterpret_cast<uintptr_t>(p) - a;
                const uint64_t len =
                        ((bytes + off + 4095) / 4096) * static_cast<uint64_t>(4096);
                cudaError_t eR = cudaHostRegister(
                        reinterpret_cast<void *>(a), static_cast<size_t>(len),
                        cudaHostRegisterPortable);
                if (eR != cudaSuccess) {
                    if (std::getenv("TINYCODER_GPU_VERBOSE") != nullptr)
                        std::fprintf(stderr,
                                     "[arena] pin FAILED bytes=%.1fMB: %s\n",
                                     static_cast<double>(len) / 1048576.0,
                                     cudaGetErrorString(eR));
                    cudaGetLastError();
                    return false;
                }
                expertPinnedRegions_.push_back(
                        {reinterpret_cast<void *>(a), static_cast<size_t>(len)});
                return true;
            };
            const bool okP = pinRange(srcGate, expertCount_ * gateSlice) &&
                             pinRange(srcUp, expertCount_ * gateSlice) &&
                             pinRange(srcDown, expertCount_ * downSlice);
            s.hostPinned = okP;
            if (std::getenv("TINYCODER_GPU_VERBOSE") != nullptr)
                std::fprintf(stderr,
                             "[arena] L=%u pinned=%d blobMB=%.1f slotBytes=%llu\n",
                             L, static_cast<int>(okP),
                             static_cast<double>(expertCount_ * (2 * gateSlice +
                                                                 downSlice)) /
                                     1048576.0,
                             static_cast<unsigned long long>(2 * gateSlice +
                                                             downSlice));
        }
        if (s.hostPinned) {
            cudaMemcpyAsync(slot, srcGate + static_cast<uint64_t>(e) * gateSlice,
                            static_cast<size_t>(gateSlice),
                            cudaMemcpyHostToDevice, g_stream2);
            cudaMemcpyAsync(slot + gateSlice,
                            srcUp + static_cast<uint64_t>(e) * gateSlice,
                            static_cast<size_t>(gateSlice),
                            cudaMemcpyHostToDevice, g_stream2);
            // The down slice goes out on a SECOND copy stream so the down
            // transfer can proceed while gate+up is still in flight (Turing has
            // several copy engines; one stream serialises all three).  A
            // stream2->aux wait keeps the single per-expert ready event valid.
            static cudaStream_t auxCopy = nullptr;
            static cudaEvent_t auxEv = nullptr;
            if (auxCopy == nullptr) {
                cudaStreamCreateWithFlags(&auxCopy, cudaStreamNonBlocking);
                cudaEventCreateWithFlags(&auxEv, cudaEventDisableTiming);
            }
            if (auxCopy != nullptr) {
                cudaMemcpyAsync(slot + 2 * gateSlice,
                                srcDown + static_cast<uint64_t>(e) * downSlice,
                                static_cast<size_t>(downSlice),
                                cudaMemcpyHostToDevice, auxCopy);
                cudaEventRecord(auxEv, auxCopy);
                cudaStreamWaitEvent(g_stream2, auxEv, 0);
            } else {
                cudaMemcpyAsync(slot + 2 * gateSlice,
                                srcDown + static_cast<uint64_t>(e) * downSlice,
                                static_cast<size_t>(downSlice),
                                cudaMemcpyHostToDevice, g_stream2);
            }
            if (e < expertArenaEv_.size() && expertArenaEv_[e] != nullptr)
                cudaEventRecord(expertArenaEv_[e], g_stream2);
            expertArenaStamp_[e] = expertArenaEpoch_;
            return slot;
        }
        const uint32_t st = (expertStageFlag_++) % kExpertStageBufs;
        if (cudaEventQuery(expertStageEv_[st]) != cudaSuccess)
            cudaEventSynchronize(expertStageEv_[st]);
        std::memcpy(expertStage_[st],
                    srcGate + static_cast<uint64_t>(e) * gateSlice,
                    static_cast<size_t>(gateSlice));
        std::memcpy(expertStage_[st] + gateSlice,
                    srcUp + static_cast<uint64_t>(e) * gateSlice,
                    static_cast<size_t>(gateSlice));
        std::memcpy(expertStage_[st] + 2 * gateSlice,
                    srcDown + static_cast<uint64_t>(e) * downSlice,
                    static_cast<size_t>(downSlice));
        cudaMemcpyAsync(slot, expertStage_[st],
                        static_cast<size_t>(slotBytes), cudaMemcpyHostToDevice,
                        g_stream2);
        cudaEventRecord(expertStageEv_[st], g_stream2);
        if (e < expertArenaEv_.size() && expertArenaEv_[e] != nullptr)
            cudaEventRecord(expertArenaEv_[e], g_stream2);
        expertArenaStamp_[e] = expertArenaEpoch_;
        return slot;
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
        std::memcpy(expertStage_[st],
                    srcGate + static_cast<uint64_t>(e) * gateSlice,
                    static_cast<size_t>(gateSlice));
        std::memcpy(expertStage_[st] + gateSlice,
                    srcUp + static_cast<uint64_t>(e) * gateSlice,
                    static_cast<size_t>(gateSlice));
        std::memcpy(expertStage_[st] + 2 * gateSlice,
                    srcDown + static_cast<uint64_t>(e) * downSlice,
                    static_cast<size_t>(downSlice));
        cudaMemcpyAsync(const_cast<uint8_t *>(gBase), expertStage_[st],
                        static_cast<size_t>(slotBytes), cudaMemcpyHostToDevice,
                        g_stream);
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

    bool GPUModel::dequantMatrixF16Into(const DeviceMatrix &m, void *dst,
                                        void *stream, std::string &errMsg) {
        if (m.q == nullptr || m.rows == 0 || m.cols == 0) return false;
        launchDequantF16(m.type, m.q, dst, m.rows, m.cols, m.rowBytes,
                         m.blocksPerRow, reinterpret_cast<cudaStream_t>(stream));
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            errMsg = std::string("dequantMatrixF16Into: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
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

    // q8_1 variant of the H S P transform for the dp4a decode GEMV: identical
    // math, quantised to int8 per 32 columns in the PrismML fork's PQ2 q8
    // layout (kPrismFoldToQ8).  Requires the q8 scratch from uploadWithPrism.
    void GPUModel::prismFoldToQ8(const float *x, uint32_t cols, bool gdnPerm,
                                 const float *mul) {
        if (x == nullptr || prismXq_ == nullptr || prismXd_ == nullptr) return;
        if (geom_.prismFold == 0 || prismBlock_ == 0 || cols == 0) return;
        const float *signs = prismSignsFor(cols);
        const bool perm = gdnPerm && prismGdnVGrouped_ && prismRep_ > 1u &&
                          prismHeadV_ > 0u && prismNk_ > 0u;
        const uint32_t grid = (cols + prismBlock_ - 1u) / prismBlock_;
        kPrismFoldToQ8<<<grid, 256, 0, g_stream>>>(
                x, prismXq_, prismXd_, cols, signs, perm ? prismHeadV_ : 0u,
                perm ? prismNk_ : 0u, perm ? prismRep_ : 0u, mul, prismXqT_);
    }

    namespace {
        // TINYCODER_PQ2_BATCH=0 restores the validated per-token PQ2_0 GEMV loop
        // for prefill (A/B switch; the batched fp16-twin GEMM is the default).
        bool pq2BatchDisabled() {
            static const bool off = [] {
                const char *e = std::getenv("TINYCODER_PQ2_BATCH");
                return e != nullptr && e[0] == '0';
            }();
            return off;
        }
    }// namespace

    // Prism PQ2_0 batched prefill.  The per-token GEMV loop re-reads every packed
    // weight for every prompt token, which is why Bonsai's pp512 was 16.8 tok/s
    // (the same 6.85 GB/token as decode) while the reference runtime, which runs
    // a batched GEMM, measures 879.7 pp512 on this card and this file.  This
    // expands the packed ternary to an fp16 twin once per matrix and issues one
    // fp16 tensor-core GEMM; the fold (signs + blockwise Hadamard + optional GDN
    // value-head permutation) moves to the batched activation, once per token per
    // projection, exactly as the format intends.  Numerics: the ternary weight
    // times its fp16 group scale is EXACT in fp16, so the only new rounding is the
    // activation's fp32 -> fp16 store, the same rounding the validated EXL3 batch
    // path uses (logits match the fused path to 0.001 there).
    // P1 spike: the packed ternary -> Q8_0 block layout, then the engine's own
    // int8 tensor-core GEMM (mma.s8.m8n8k16, stream-k).  This exists to measure
    // the IMMA rate on the real shapes; it keeps the fp16-twin path intact as
    // the fallback and as the numeric reference.
    bool GPUModel::gemmPQ2Int8(const DeviceMatrix &m, const float *x, float *y,
                               uint32_t seqLen, bool gdnPerm,
                               std::string &errMsg, bool preFolded) {
        if (m.q == nullptr || m.rows == 0 || m.cols == 0 || seqLen < 8) {
            return false;
        }
        if (geom_.prismFold == 0 || prismBlock_ == 0) return false;
        if ((m.cols % 512u) != 0u || (m.cols % 128u) != 0u) return false;
        if (!int8GemmAvailable()) return false;
        const uint64_t wBytes = (m.type == kTypePTQ1_0)
                                        ? static_cast<uint64_t>(m.rows) *
                                                  m.blocksPerRow * 34u
                                        : static_cast<uint64_t>(m.rows) * m.rowBytes;
        if (pq2Wq8_ == nullptr || pq2Wq8Bytes_ < wBytes) {
            if (pq2Wq8_) cudaFree(pq2Wq8_);
            pq2Wq8_ = nullptr;
            pq2Wq8Bytes_ = 0;
            if (cudaMalloc(&pq2Wq8_, static_cast<size_t>(wBytes)) != cudaSuccess) {
                return false;
            }
            pq2Wq8Bytes_ = wBytes;
        }
        const uint64_t xBytes =
                static_cast<uint64_t>(seqLen) * m.cols * sizeof(float);
        if (pq2Xf_ == nullptr || pq2XfBytes_ < xBytes) {
            if (pq2Xf_) cudaFree(pq2Xf_);
            pq2Xf_ = nullptr;
            pq2XfBytes_ = 0;
            if (cudaMalloc(&pq2Xf_, static_cast<size_t>(xBytes)) != cudaSuccess) {
                return false;
            }
            pq2XfBytes_ = xBytes;
        }
        static const bool i8T0On = std::getenv("TINYCODER_PQ2_PROF") != nullptr;
        cudaEvent_t i8Start = nullptr;
        if (i8T0On) {
            cudaEventCreate(&i8Start);
            cudaEventRecord(i8Start, g_stream);
        }
        // fold the whole activation batch to fp32 (int8Gemm quantises it itself)
        const float *signs = prismSignsFor(m.cols);
        const bool perm = gdnPerm && prismGdnVGrouped_ && prismRep_ > 1u &&
                          prismHeadV_ > 0u && prismNk_ > 0u;
        // gate/up and Q/K/V share one activation, so the caller folds it once
        // (pq2PreFold) and both projections reuse pq2Xf_.  The flag is explicit
        // because prefill reuses addresses across layers with different data:
        // caching on the pointer would produce wrong logits, not a crash.
        if (!preFolded) {
            const uint32_t foldGrid = (m.cols + prismBlock_ - 1u) / prismBlock_;
            kPrismFoldB<<<dim3(foldGrid, seqLen), 256, 0, g_stream>>>(
                    x, pq2Xf_, m.cols, signs, perm ? prismHeadV_ : 0u,
                    perm ? prismNk_ : 0u, perm ? prismRep_ : 0u, m.cols);
        }
        // Native PQ2_0: int8Gemm decodes the packed 2-bit weights inside its
        // tile loader, so no int8/fp16 twin is materialised at all.  The k unit
        // for this type is one 34-byte block (128 weights), i.e. m.blocksPerRow.
        static const bool nativePq2 = [] {
            const char *e = std::getenv("TINYCODER_PQ2_NATIVE");
            return e == nullptr || e[0] != '0';
        }();
        const uint32_t kWarps = 8u;
        if (!nativePq2 && m.type == kTypePQ2_0) {
            kDequantPQ2_0ToQ8_0<<<m.rows, kWarps * 32u, 0, g_stream>>>(
                    static_cast<const uint8_t *>(m.q),
                    static_cast<uint8_t *>(pq2Wq8_), m.rows, m.cols,
                    m.rowBytes, m.blocksPerRow);
        }
        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            errMsg = std::string("PQ2_0 int8 twin launch: ") +
                     cudaGetErrorString(le);
            cudaGetLastError();
            return false;
        }
        // P1 measurement: split expand (fold + twin) from the int8 GEMM, same
        // event discipline as the fp16 path's TINYCODER_PQ2_PROF block.
        static const bool i8Prof = std::getenv("TINYCODER_PQ2_PROF") != nullptr;
        static double i8T[2] = {0.0, 0.0};
        static uint64_t i8N = 0, i8Flops = 0;
        cudaEvent_t i8Ev[3];
        if (i8Prof) {
            for (auto &e: i8Ev) cudaEventCreate(&e);
            i8Ev[0] = i8Start;
            cudaEventRecord(i8Ev[1], g_stream);
        }
        // Native PTQ1_0 prefill: int8Gemm has its own 28-byte tile loader
        // (int8GemmStreamKPTQ10Tile{32,64,128}), so the matrix can be fed
        // straight in and the per-call transcode into pq2Wq8_ disappears --
        // that transcode re-reads every weight (28 B) and re-writes it as 34 B
        // on every layer of every prompt.
        // DEFAULT OFF by measurement: the direct route avoids that transcode but
        // its tile loader (load_tiles_ptq1_0) costs ~1.77x the PQ2_0 loader on
        // identical work, so pp512 measured 498.8 / 496.5 against 728.4 / 726.4
        // for transcode + PQ2_0.  The transcode stays the cheaper of the two
        // until that loader is fixed; TINYCODER_PT_PREFILL=1 selects the direct
        // path (correct: logits match the transcode route bit-for-bit, because
        // ternary -> PQ2_0 is lossless).
        static const bool ptq1PrefillNative = [] {
            const char *e = std::getenv("TINYCODER_PT_PREFILL");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        const bool ptq1NativeGemm = m.type == kTypePTQ1_0 && ptq1PrefillNative;
        if (m.type == kTypePTQ1_0 && !ptq1NativeGemm) {
            const uint32_t nb = m.rows * m.blocksPerRow;
            kPTQ1_0ToPQ2<<<(nb + 3u) / 4u, 128, 0, g_stream>>>(
                    static_cast<const uint8_t *>(m.q),
                    static_cast<uint8_t *>(pq2Wq8_), nb);
        }
        const bool ok = ptq1NativeGemm
                                ? int8Gemm(static_cast<int>(kTypePTQ1_0), m.q,
                                           pq2Xf_, y, seqLen, m.rows, m.cols,
                                           m.blocksPerRow, g_stream)
                        : m.type == kTypePTQ1_0
                                ? int8Gemm(static_cast<int>(kTypePQ2_0),
                                           pq2Wq8_, pq2Xf_, y, seqLen,
                                           m.rows, m.cols, m.blocksPerRow,
                                           g_stream)
                        : nativePq2
                                ? int8Gemm(static_cast<int>(m.type), m.q, pq2Xf_,
                                           y, seqLen, m.rows, m.cols,
                                           m.blocksPerRow, g_stream)
                                : int8Gemm(static_cast<int>(kTypeQ8_0), pq2Wq8_,
                                           pq2Xf_, y, seqLen, m.rows, m.cols,
                                           m.cols / 32u, g_stream);
        if (i8Prof) {
            cudaEventRecord(i8Ev[2], g_stream);
            cudaEventSynchronize(i8Ev[2]);
            float a = 0.0f, b2 = 0.0f;
            cudaEventElapsedTime(&a, i8Ev[0], i8Ev[1]);
            cudaEventElapsedTime(&b2, i8Ev[1], i8Ev[2]);
            i8T[0] += a;
            i8T[1] += b2;
            i8Flops += 2ull * m.rows * m.cols * seqLen;
            for (auto &e: i8Ev) cudaEventDestroy(e);
            if (++i8N % 64u == 0u) {
                std::fprintf(stderr,
                             "[pq2int8] %llu proj | expand %.1f ms | gemm "
                             "%.1f ms (%.1f TFLOPS)\n",
                             (unsigned long long) i8N, i8T[0], i8T[1],
                             static_cast<double>(i8Flops) / (i8T[1] * 1e9));
                i8T[0] = i8T[1] = 0.0;
                i8Flops = 0;
            }
        }
        if (!ok) {
            errMsg = "int8Gemm declined the PQ2_0 Q8_0 twin";
            return false;
        }
        return true;
    }

    bool GPUModel::gemmPQ2Cublas(const DeviceMatrix &m, const float *x, float *y,
                                 uint32_t seqLen, bool gdnPerm,
                                 std::string &errMsg) {
        if (m.q == nullptr || m.rows == 0 || m.cols == 0 || seqLen == 0) {
            return false;
        }
        if (geom_.prismFold == 0 || prismBlock_ == 0) return false;
        if ((m.cols % 128u) != 0) return false;
        const uint64_t wBytes =
                static_cast<uint64_t>(m.rows) * m.cols * sizeof(__half);
        if (pq2W16_ == nullptr || pq2W16Bytes_ < wBytes) {
            if (pq2W16_) cudaFree(pq2W16_);
            pq2W16_ = nullptr;
            pq2W16Bytes_ = 0;
            if (cudaMalloc(&pq2W16_, static_cast<size_t>(wBytes)) != cudaSuccess) {
                return false;
            }
            pq2W16Bytes_ = wBytes;
        }
        const uint64_t xBytes =
                static_cast<uint64_t>(seqLen) * m.cols * sizeof(__half);
        if (pq2X16_ == nullptr || pq2X16Bytes_ < xBytes) {
            if (pq2X16_) cudaFree(pq2X16_);
            pq2X16_ = nullptr;
            pq2X16Bytes_ = 0;
            if (cudaMalloc(&pq2X16_, static_cast<size_t>(xBytes)) != cudaSuccess) {
                return false;
            }
            pq2X16Bytes_ = xBytes;
        }
        const float *signs = prismSignsFor(m.cols);
        const bool perm = gdnPerm && prismGdnVGrouped_ && prismRep_ > 1u &&
                          prismHeadV_ > 0u && prismNk_ > 0u;
        const uint32_t foldGrid = (m.cols + prismBlock_ - 1u) / prismBlock_;
        kPrismFoldToF16<<<dim3(foldGrid, seqLen), 256, 0, g_stream>>>(
                x, reinterpret_cast<__half *>(pq2X16_), m.cols, signs,
                perm ? prismHeadV_ : 0u, perm ? prismNk_ : 0u,
                perm ? prismRep_ : 0u, m.cols);
        // TINYCODER_PQ2_PROF=1: split the batched prefill into fold, weight
        // expansion and GEMM.  Synced, so it is a diagnostic only.
        static const bool pq2Prof = std::getenv("TINYCODER_PQ2_PROF") != nullptr;
        static double pq2T[3] = {0.0, 0.0, 0.0};
        static uint64_t pq2N = 0;
        static double pq2MB = 0.0;
        cudaEvent_t pq2Ev[4];
        if (pq2Prof) {
            for (auto &e: pq2Ev) cudaEventCreate(&e);
            cudaEventRecord(pq2Ev[0], g_stream);
        }
        // ---- overlap the twin expansion with the GEMM -------------------
        // The expansion writes 8x the packed bytes and is pure DRAM (measured
        // ~163 ms per 512-token prefill); the cuBLAS GEMM that consumes it is
        // compute-bound at 90% of the fp32-accumulate peak.  They use opposite
        // resources, so the expansion is split into NCH row chunks and chunk
        // i+1 is expanded on g_stream2 WHILE chunk i is multiplied on g_stream.
        // Row ranges are disjoint, so one twin buffer still serves both, and
        // an event per chunk orders the GEMM behind its own expansion.
        constexpr uint32_t kWarps = 8u;
        static const uint32_t nchCfg = [] {
            const char *e = std::getenv("TINYCODER_PQ2_OVERLAP");
            const long v = e != nullptr ? std::strtol(e, nullptr, 10) : 1;
            return (v >= 1 && v <= 8) ? static_cast<uint32_t>(v)
                                      : static_cast<uint32_t>(1);
        }();
        const uint32_t nch = (m.rows >= 1024u * nchCfg) ? nchCfg : 1u;
        const uint32_t chunkRows = (m.rows + nch - 1u) / nch;
        static cudaEvent_t ovEv[8];
        static bool ovEvInit = false;
        if (nch > 1u && !ovEvInit) {
            for (auto &e: ovEv) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
            ovEvInit = true;
        }
        auto expandChunk = [&](uint32_t i, cudaStream_t st) {
            const uint32_t r0 = i * chunkRows;
            const uint32_t rn = (m.rows - r0 < chunkRows) ? (m.rows - r0)
                                                          : chunkRows;
            if (rn == 0u) return;
            kDequantPQ2_0ToF16<<<rn, kWarps * 32u, 0, st>>>(
                    static_cast<const uint8_t *>(m.q) +
                            (uint64_t) r0 * m.rowBytes,
                    reinterpret_cast<__half *>(pq2W16_) +
                            (uint64_t) r0 * m.cols,
                    rn, m.cols, m.rowBytes, m.blocksPerRow);
        };
        expandChunk(0, g_stream);
        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            errMsg = std::string("PQ2_0 batched prefill launch: ") +
                     cudaGetErrorString(le);
            cudaGetLastError();
            return false;
        }
        if (pq2Prof) cudaEventRecord(pq2Ev[1], g_stream);
        cublasSetStream(g_cublas, g_stream);
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
        for (uint32_t i = 0; i < nch; ++i) {
            const uint32_t r0 = i * chunkRows;
            const uint32_t rn = (m.rows - r0 < chunkRows) ? (m.rows - r0)
                                                          : chunkRows;
            if (rn == 0u) break;
            // kick the next chunk's expansion on the other stream; the GEMM
            // below reads only this chunk's rows, so they do not collide
            if (nch > 1u && i + 1u < nch) {
                expandChunk(i + 1u, g_stream2);
                cudaEventRecord(ovEv[i + 1u], g_stream2);
            }
            if (nch > 1u && i > 0u) {
                cudaStreamWaitEvent(g_stream, ovEv[i], 0);
            }
            st = cublasGemmEx(
                    g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rn),
                    static_cast<int>(seqLen), static_cast<int>(m.cols), &alpha,
                    reinterpret_cast<const __half *>(pq2W16_) +
                            (uint64_t) r0 * m.cols,
                    CUDA_R_16F, static_cast<int>(m.cols), pq2X16_, CUDA_R_16F,
                    static_cast<int>(m.cols), &beta,
                    // C is column-major [seqLen][rows]: a row chunk starts at
                    // +r0 inside each token column, and ldc stays m.rows.
                    y + (uint64_t) r0, CUDA_R_32F, static_cast<int>(m.rows),
                    CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) break;
        }
        if (nch > 1u) {
            // the next projection's expansion must not race this one's GEMM
            cudaEventRecord(ovEv[7], g_stream);
            cudaStreamWaitEvent(g_stream2, ovEv[7], 0);
        }
        if (st != CUBLAS_STATUS_SUCCESS) {
            errMsg = "cublasGemmEx PQ2_0 prefill failed";
            return false;
        }
        if (pq2Prof) {
            cudaEventRecord(pq2Ev[2], g_stream);
            cudaEventSynchronize(pq2Ev[2]);
            float a0 = 0, a1 = 0;
            cudaEventElapsedTime(&a0, pq2Ev[0], pq2Ev[1]);// fold + expand
            cudaEventElapsedTime(&a1, pq2Ev[1], pq2Ev[2]);// gemm
            pq2T[0] += a0;
            pq2T[1] += a1;
            pq2MB += static_cast<double>(m.rows) * m.cols * 2.0 / 1e6;
            if (++pq2N % 64u == 0u) {
                std::fprintf(stderr,
                             "[pq2prof] %llu proj | expand %.1f ms | gemm %.1f ms "
                             "| gemm %.0f MB in %.1f ms = %.0f GB/s\n",
                             (unsigned long long) pq2N, pq2T[0], pq2T[1],
                             pq2MB / 1e6 * 64512.0 / 64.0,
                             pq2T[1] / 64.0,
                             (pq2MB / 64.0) / (pq2T[1] / 64.0));
                pq2T[0] = pq2T[1] = 0.0;
                pq2MB = 0.0;
            }
        }
        return true;
    }

    // Batched Q8_0 GEMV for narrow matrices (rows < 128).
    // The int8 stream-k engine needs a 128-row weight tile, so projections like the
    // dense qwen35 ssm_alpha/ssm_beta (32x4096) fell back to one GEMV launch per
    // token -- 512 launches per layer, measured at 20% of 9B q8_0 prefill. One
    // block per token computes all `rows` outputs in a single launch.
    // Batched Q8_0 GEMV for narrow matrices (rows < 128): one warp per output row,
    // so the 32 row dot-products run concurrently and reduce by warp shuffle (no
    // block-wide barriers).  One block per token.
    // Batched Q8_0 GEMV for narrow matrices (rows < 128), token-blocked.
    // The 32x4096 ssm_alpha/ssm_beta projections are tiny, so the cost is entirely
    // re-reading their 131 KB of weights once per token.  Blocking 16 tokens per
    // block (one warp per output row) cuts that traffic 16x; a plain one-token-per-
    // block version was measured slower than the per-token GEMV fallback.

    template<int TP>
    __global__ void q8SmallRowsGemm(const uint8_t *__restrict__ w,
                                    const float *__restrict__ x,
                                    float *__restrict__ y,
                                    uint32_t rows, uint32_t cols, uint32_t seqLen) {
        const uint32_t warp = threadIdx.x >> 5;
        const uint32_t lane = threadIdx.x & 31u;
        if (warp >= rows) return;
        const uint32_t tok0 = blockIdx.x * TP;
        const uint32_t nt = (seqLen - tok0 < TP) ? (seqLen - tok0) : TP;
        const uint32_t nb = cols / 32u;
        const uint8_t *wr = w + static_cast<size_t>(warp) * nb * 34u;
        float acc[TP];
        for (int t = 0; t < TP; ++t) acc[t] = 0.0f;
        for (uint32_t b = lane; b < nb; b += 32u) {
            const uint8_t *p = wr + static_cast<size_t>(b) * 34u;
            const float d = __half2float(*reinterpret_cast<const __half *>(p));
            const int8_t *q = reinterpret_cast<const int8_t *>(p + 2);
            for (uint32_t t = 0; t < nt; ++t) {
                const float *xb = x + static_cast<size_t>(tok0 + t) * cols + b * 32u;
                float sc = 0.0f;
                for (int i = 0; i < 32; ++i) sc += static_cast<float>(q[i]) * xb[i];
                acc[t] += d * sc;
            }
        }
        for (uint32_t t = 0; t < nt; ++t) {
            float a = acc[t];
            for (uint32_t st = 16u; st > 0u; st >>= 1)
                a += __shfl_down_sync(0xffffffffu, a, st);
            if (lane == 0)
                y[static_cast<size_t>(tok0 + t) * rows + warp] = a;
        }
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

        // ---- CUDA-graph decode (TINYCODER_Q35_GRAPH) ---------------------
        // A single-token pass launches ~700 small kernels.  With the INT8/GEMV
        // ports and split-KV attention the GPU work is ~12 ms/token; the host
        // dispatch that feeds it is several ms of the wall.  Capture the whole
        // pass once and replay it, refreshing only the token and position
        // device scalars.  A capture failure falls back to the eager path.
        const char *qgEnv = std::getenv("TINYCODER_Q35_GRAPH");
        // Opt-in: capture works and replays correctly, but it measured +0.3%
        // (74.4 -> 74.6 tok/s) -- the decode is GPU-kernel bound, not host
        // launch bound, and the first decode pays a ~1.3 s capture.  Enable
        // with TINYCODER_Q35_GRAPH=1 when the host is the bottleneck.
        const bool graphOn = qgEnv != nullptr && qgEnv[0] != '\0' && qgEnv[0] != '0';
        const bool graphWanted = graphOn && seqLen == 1 && fullOffload &&
                                 !computeAllLogits && !graphCaptureFailed_;
        if (graphWanted && graphBuilt_) {
            return replayQwen35DecodeGraph(tokens, logitsOut, errMsg);
        }
        if (graphWanted && !graphBuilt_ && !graphCaptureActive_) {
            if (ensureGraphScratch()) {
                // cudaMalloc is not capturable: reserve both GEMV scratch
                // widths (hidden and intermediate) before BeginCapture.
                int8GemvReserve(H, 1);
                int8GemvReserve(I, 1);
                cudaMemcpy(s.tokens, tokens.data(), sizeof(int32_t),
                           cudaMemcpyHostToDevice);
                graphPosHost_ = pos;
                cudaMemcpy(graphPos_, &graphPosHost_, sizeof(uint32_t),
                           cudaMemcpyHostToDevice);
                if (cudaStreamBeginCapture(g_stream,
                                           cudaStreamCaptureModeRelaxed) ==
                    cudaSuccess) {
                    graphCaptureActive_ = true;
                } else {
                    graphCaptureFailed_ = true;
                }
            } else {
                graphCaptureFailed_ = true;
            }
        }
        const uint32_t *dpos = graphCaptureActive_ ? graphPos_ : nullptr;
        if (!graphCaptureActive_) {
            cudaMemcpyAsync(s.tokens, tokens.data(), seqLen * sizeof(int32_t),
                            cudaMemcpyHostToDevice, g_stream);
            cudaStreamSynchronize(g_stream);
        }

        // ---- per-phase decode trace (TINYCODER_STAGE_TRACE=1) --------------
        // The arch-1 path had no accounting between "projection" and
        // "everything else".  Three events per layer split a token into the
        // attention/SSM projections, the recurrent/attention kernels that
        // follow them, and the FFN; the events are recorded back-to-back on the
        // stream and read once at the end, so the sums are GPU times.
        const char *q35TraceEnv = std::getenv("TINYCODER_STAGE_TRACE");
        const bool q35Trace = q35TraceEnv != nullptr && q35TraceEnv[0] != '\0' &&
                              q35TraceEnv[0] != '0' && !graphCaptureActive_ &&
                              seqLen == 1;
        static cudaEvent_t q35Ev[8 + 4 * 256];
        static bool q35EvInit = false;
        if (q35Trace && !q35EvInit) {
            for (auto &e: q35Ev) cudaEventCreate(&e);
            q35EvInit = true;
        }
        auto q35Mark = [&](uint32_t slot) {
            if (q35Trace) cudaEventRecord(q35Ev[slot], g_stream);
        };
        auto q35T0 = std::chrono::steady_clock::now();
        q35Mark(0);
        // ---- Embedding (K-quant / legacy).  Mirror forward()'s dispatch. ----
        {
            const uint32_t blocksPerRow = hostBlocksPerRow(embedType_, H);
            const uint8_t *eQ = reinterpret_cast<const uint8_t *>(embedQ_);
            if (embedType_ == kTypeF16) {
                kEmbedF16<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, embedRowBytes_,
                        geom_.vocabSize);
            } else if (embedType_ == kTypeQ5K || embedType_ == kTypeQ5_0) {
                kEmbedDequantQ5K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ4K) {
                kEmbedDequantQ4K<<<seqLen, 32, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize);
            } else if (embedType_ == kTypeQ6K) {
                kEmbedDequantQ6K<<<seqLen, 32, 0, g_stream>>>(
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
            } else if (embedType_ == kTypePTQ1_0) {
                kEmbedDequantPTQ1_0<<<seqLen, 256, 0, g_stream>>>(
                        eQ, s.tokens, s.hidden, seqLen, H, blocksPerRow,
                        embedRowBytes_, geom_.vocabSize, prismSignsFor(H));
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
        q35Mark(7);
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
        // Batched-prefill gate for weights that are NOT Hadamard-folded.  On a
        // Prism model (prismFold != 0) the folded PQ2_0 weights stay on the
        // per-token loop, but the F32 ssm_alpha / ssm_beta projections are NOT
        // folded and must still batch: leaving them on the per-token loop costs
        // 2 launches x seqLen x 48 GDN layers (49,248 launches on a 512-token
        // prefill, measured 0.28 s of the 1.07 s prefill before this gate).
        // Every branch below that uses it is additionally guarded by !folded, so
        // for a non-Prism model this is simply "seqLen > 1".
        const bool batchPrefill = (seqLen > 1);
        // ---- Per-projection profiler (TINYCODER_PROF=1) ----
        // Accumulates wall time per `what` label (projection name) so the prefill
        // split is measured, not guessed.  Reports the slowest six every 512 calls.
        static const bool g_prof = std::getenv("TINYCODER_PROF") != nullptr;
        static const char *g_pName[32] = {nullptr};
        static double g_pMs[32] = {0};
        static long g_pN[32] = {0};
        static int g_pCnt = 0;
        static long g_pTotal = 0;
        static long g_pEnter = 0;
        // Event pairs are RECORDED only and reduced in one batch after a single
        // stream sync: synchronising per call (the first version) serialised the
        // stream and inflated the small projections' measured share.
        struct ProfSample {
            cudaEvent_t a, b;
            const char *nm;
        };
        static ProfSample g_pS[4096];
        static int g_pSN = 0;
        struct ProfScope {
            const char *nm;
            cudaEvent_t a, b;
            explicit ProfScope(const char *n) : nm(n) {
                if (g_prof) {
                    cudaEventCreate(&a);
                    cudaEventCreate(&b);
                    cudaEventRecord(a, g_stream);
                }
            }
            ~ProfScope() {
                if (!g_prof) return;
                cudaEventRecord(b, g_stream);
                if (g_pSN < 4000) {
                    g_pS[g_pSN].a = a;
                    g_pS[g_pSN].b = b;
                    g_pS[g_pSN].nm = nm;
                    ++g_pSN;
                }
                if (++g_pTotal % 4000 == 0 && g_pSN > 0) {
                    cudaStreamSynchronize(g_stream);
                    g_pCnt = 0;
                    for (int i = 0; i < g_pSN; ++i) {
                        float ms = 0.0f;
                        cudaEventElapsedTime(&ms, g_pS[i].a, g_pS[i].b);
                        int sl = -1;
                        for (int k = 0; k < g_pCnt; ++k)
                            if (std::strcmp(g_pName[k], g_pS[i].nm) == 0) sl = k;
                        if (sl < 0 && g_pCnt < 32) {
                            sl = g_pCnt++;
                            g_pName[sl] = g_pS[i].nm;
                            g_pMs[sl] = 0;
                            g_pN[sl] = 0;
                        }
                        if (sl >= 0) {
                            g_pMs[sl] += ms;
                            g_pN[sl] += 1;
                        }
                        cudaEventDestroy(g_pS[i].a);
                        cudaEventDestroy(g_pS[i].b);
                    }
                    g_pSN = 0;
                    double tot = 0.0;
                    for (int i = 0; i < g_pCnt; ++i) tot += g_pMs[i];
                    bool used[32] = {false};
                    for (int k = 0; k < 8; ++k) {
                        int best = -1;
                        for (int i = 0; i < g_pCnt; ++i)
                            if (!used[i] && (best < 0 || g_pMs[i] > g_pMs[best])) best = i;
                        if (best < 0 || g_pMs[best] <= 0.0) break;
                        used[best] = true;
                        std::fprintf(stderr, "[prof] %-14s %8.1f ms %6ld calls %5.1f%%\n",
                                     g_pName[best], g_pMs[best], g_pN[best],
                                     100.0 * g_pMs[best] / (tot + 1e-9));
                    }
                    std::fprintf(stderr, "[prof] --- total %.1f ms ---\n", tot);
                    for (int i = 0; i < g_pCnt; ++i) {
                        g_pMs[i] = 0;
                        g_pN[i] = 0;
                    }
                }
            }
        };
        // siluMul/siluDone: request that the FFN activation be fused into this
        // projection's input Hadamard (used for ffnDown).  *siluDone reports
        // whether it actually happened; if not, the caller must run kSiluMul
        // and issue the projection again.
        // Prologue for a GROUP of folded PQ2_0 projections sharing one
        // activation vector (gate+up, q/k/v, qkv+gate): fold+quantise once and
        // let each gemv() below reuse it.  The q8 scratch is consumed by the
        // next GEMV on the same stream and no other fold runs in between.
        // Returns false when the q8 path does not apply, in which case every
        // projection folds its own activation as before.
        auto pq2PreFold = [&](const DeviceMatrix &m, const float *x,
                              uint32_t nCols, bool gdnPerm = false) -> bool {
            if (geom_.prismFold == 0 ||
                (m.type != kTypePQ2_0 && m.type != kTypePTQ1_0))
                return false;
            if ((nCols % 128u) != 0u || m.blocksPerRow * 128u != m.cols)
                return false;
            // Native PTQ1_0 decode folds and quantizes inside its own kernel
            // (and reuses that fold across q/k/v and gate/up), so the shared
            // PQ2 pre-fold here would be pure duplicate work -- it was 128
            // kPrismFoldToQ8 launches per token.
            if (m.type == kTypePTQ1_0 && ptq1GenericEnabled()) return false;
            if (seqLen == 1) {
                if (!pq2Q8Enabled() || prismXq_ == nullptr || prismXd_ == nullptr)
                    return false;
                prismFoldToQ8(x, nCols, false);
                return true;
            }
            // Prefill: fold the whole batch to fp32 into pq2Xf_, the buffer
            // gemmPQ2Int8's own fold would fill, so both projections of a pair
            // (gate/up, Q/K/V) can skip it.
            if (m.cols % 512u != 0u || !int8GemmAvailable()) return false;
            const uint64_t xBytes =
                    static_cast<uint64_t>(seqLen) * m.cols * sizeof(float);
            if (pq2Xf_ == nullptr || pq2XfBytes_ < xBytes) {
                if (pq2Xf_) cudaFree(pq2Xf_);
                pq2Xf_ = nullptr;
                pq2XfBytes_ = 0;
                if (cudaMalloc(&pq2Xf_, static_cast<size_t>(xBytes)) !=
                    cudaSuccess) {
                    return false;
                }
                pq2XfBytes_ = xBytes;
            }
            const float *pSigns = prismSignsFor(m.cols);
            // `gdnPerm` MUST be part of this predicate: gemmPQ2Int8 applies the
            // GDN value-head permutation only when its own gdnPerm argument is
            // set, so omitting it here folds with the permutation at call sites
            // that must not have it (that bug failed the ptq1_0 text gate).
            const bool pPerm = gdnPerm && prismGdnVGrouped_ && prismRep_ > 1u &&
                               prismHeadV_ > 0u && prismNk_ > 0u;
            const uint32_t pGrid = (m.cols + prismBlock_ - 1u) / prismBlock_;
            kPrismFoldB<<<dim3(pGrid, seqLen), 256, 0, g_stream>>>(
                    x, pq2Xf_, m.cols, pSigns, pPerm ? prismHeadV_ : 0u,
                    pPerm ? prismNk_ : 0u, pPerm ? prismRep_ : 0u, m.cols);
            return cudaGetLastError() == cudaSuccess;
        };
        auto pq2Q8Applies = [&](const DeviceMatrix &m, uint32_t nCols) -> bool {
            return geom_.prismFold != 0 &&
                   (m.type == kTypePQ2_0 || m.type == kTypePTQ1_0) &&
                   seqLen == 1 && pq2Q8Enabled() && prismXq_ != nullptr &&
                   prismXd_ != nullptr && (nCols % 128u) == 0u &&
                   m.blocksPerRow * 128u == m.cols;
        };
        auto foldThreads = [] {
            static const uint32_t v = [] {
                const char *e = std::getenv("TINYCODER_PT_FOLD_T");
                const int n = e != nullptr ? std::atoi(e) : 0;
                return n == 128 ? 128u : n == 256 ? 256u
                                 : n == 512       ? 512u
                                                  : 1024u;
            }();
            return v;
        };
        auto ptLayoutSel = [] {
            static const uint32_t v = [] {
                const char *e = std::getenv("TINYCODER_PT_PT");
                return (e == nullptr || e[0] != '0') ? 1u : 0u;
            }();
            return v;
        };
        auto gemv = [&](const DeviceMatrix &m, const float *x, float *y,
                        uint32_t nRows, uint32_t nCols, const char *what,
                        const float *siluMul = nullptr,
                        bool *siluDone = nullptr,
                        bool preFolded = false) -> bool {
            if (m.q == nullptr || nRows == 0) return true;
            ProfScope ps(what);
            if (g_prof && g_pEnter < 80) {
                std::fprintf(stderr, "[proj] %-12s rows=%u cols=%u type=%d seq=%u\n",
                             what, nRows, nCols, static_cast<int>(m.type), seqLen);
                ++g_pEnter;
            }
            // Prism: the folded (PQ2_0) weights expect the ACTIVATION in the
            // Hadamard-folded basis H S P x, so transform it first.  Only
            // PQ2_0 tensors are folded (the F32 ssm_alpha/ssm_beta stay primal),
            // and the ssm_out projection additionally needs the GDN value-head
            // permutation.  All launches are on g_stream, so the single shared
            // prismFoldBuf_ is safe: the GPU executes transform -> gemv -> next
            // transform in stream order.
            const bool folded =
                    (geom_.prismFold != 0) &&
                    (m.type == kTypePQ2_0 || m.type == kTypePTQ1_0);
            const bool gdnPerm = folded && (std::strcmp(what, "ssmOut") == 0);
            // F32 projections (ssm_alpha / ssm_beta [nVHeads][hidden]) have no
            // quantized batch kernel, so the generic path would launch one
            // tiny GEMV PER TOKEN (49248 launches / 512-token prefill
            // measured, 0.28 s).  kQ35F32GemmBatch runs the identical per-row
            // dot + reduction as the decode GEMV, so prefill and sequential
            // stay bit-identical (a cuBLAS Sgemm version was faster but broke
            // the recurrent state: maxDiff 7.9).
            if (batchPrefill && !folded && m.type == kTypeF32) {
                std::string f32Err;
                // Prism (folded) models take the bit-identical per-row kernel:
                // this is the first time these projections batch at all on a
                // folded trunk, and the recorded cuBLAS-Sgemm experiment above
                // moved the recurrent state (maxDiff 7.9).  Non-Prism models keep
                // their existing cuBLAS path, unchanged.
                if (geom_.prismFold == 0 &&
                    gemmF32Cublas(m, x, y, seqLen, f32Err)) {
                    return true;
                }
                kQ35F32GemmBatch<<<dim3(nRows, (seqLen + 7u) / 8u), 256, 0,
                                   g_stream>>>(
                        x, reinterpret_cast<const float *>(m.q), y, nRows, nCols,
                        seqLen);
                cudaError_t pe = cudaGetLastError();
                if (pe == cudaSuccess) return true;
            }
            // Batched prefill GEMM (INT8 int8 tensor cores for the K-quants and
            // Q8_0; bit-exact integer Q8_K GEMM otherwise).  The block-size
            // gate must follow the quant's own block width: 256 for K-quants,
            // 32 for Q8_0/Q5_0/IQ4_NL (the dense qwen35 ssm_out is Q8_0).
            const uint32_t qkBlock = (m.type == kTypeQ8_0 || m.type == kTypeQ5_0 ||
                                      m.type == kTypeIQ4NL)
                                             ? 32u
                                             : 256u;
            // Optional Q4_K/Q6_K prefill via F16 dequant + cuBLAS; measured
            // against the int8 stream-K path with TINYCODER_Q4_CUBLAS=1.
            if (batchPrefill && !folded &&
                (m.type == kTypeQ4K || m.type == kTypeQ6K) &&
                std::getenv("TINYCODER_Q4_CUBLAS") != nullptr) {
                std::string q4Err;
                if (gemmQ8Cublas(m, x, y, seqLen, q4Err)) return true;
            }
            if (batchPrefill && !folded &&
                (m.type == kTypeQ6K ||
                 (m.type == kTypeQ5K &&
                  std::getenv("TINYCODER_Q5_CUBLAS") != nullptr))) {
                std::string q6Err;
                if (gemmQ8Cublas(m, x, y, seqLen, q6Err)) return true;
            }
            if (batchPrefill && !folded && m.type == kTypeQ4K &&
                m.cols > 8192u &&
                std::getenv("TINYCODER_Q4_CUBLAS_DOWN") != nullptr) {
                std::string q4Err;
                if (gemmQ8Cublas(m, x, y, seqLen, q4Err)) return true;
            }
            // RESCOPE (task 3): this lambda had NO int8Gemm call, so a loader
            // fix would have been unreachable. Probe first, gated, then decide.
            // Default ON: measured 2,233 -> 2,449 pp512 on Ornith-9B-Q8_0
            // (+9.7%), top-3 logits identical to the fp16 path.
            static const bool q8Int8 = [] {
                const char *e = std::getenv("TINYCODER_Q8_INT8");
                return e == nullptr || e[0] != '0';
            }();
            // rows < 128 (one warp-tile of weight rows, e.g. the dense qwen35
            // ssm_out at 32 rows) faults in the int8 stream-k tile loader: the
            // tile is kInt8Y=128 rows tall and the tail is not clamped for a
            // sub-tile matrix.  Keep those on cuBLAS; everything with >= 128
            // rows goes through int8.
            // Narrow Q8_0 projections (rows < 128) cannot use the 128-row int8
            // tile, so batch them here instead of falling back to per-token GEMV.
            // ON: 2449 -> 2497 pp512 on Ornith-9B-Q8_0, logits identical
            // (21.191521 both ways), neutral on 9B q4_k_m (2470->2474) and 7B
            // iq3_xxs (2742->2737).  TINYCODER_Q8_SMALL=0 disables.
            static const bool q8Small = [] {
                const char *e = std::getenv("TINYCODER_Q8_SMALL");
                return e == nullptr || e[0] != '0';
            }();
            if (q8Small && batchPrefill && !folded && m.type == kTypeQ8_0 &&
                nRows < 128u &&
                nRows > 0u && (nCols & 31u) == 0u && seqLen >= 4u) {
                const uint32_t threads = 32u * nRows;
                // Token blocking factor is a template parameter: a runtime array
                // index forces the accumulators into local memory and cost 20%.
                static const int tp = [] {
                    const char *e = std::getenv("TINYCODER_Q8_SMALL_TOK");
                    const long v = e ? std::strtol(e, nullptr, 10) : 16;
                    return (v == 8 || v == 16 || v == 32) ? static_cast<int>(v) : 16;
                }();
                auto launch = [&](auto kern, int bt) {
                    const uint32_t tb = (seqLen + bt - 1u) / bt;
                    kern<<<tb, threads, 0, g_stream>>>(
                            reinterpret_cast<const uint8_t *>(m.q), x, y, nRows, nCols,
                            seqLen);
                };
                if (tp == 8) launch(q8SmallRowsGemm<8>, 8);
                else if (tp == 32)
                    launch(q8SmallRowsGemm<32>, 32);
                else
                    launch(q8SmallRowsGemm<16>, 16);
                return true;
            }
            if (q8Int8 && batchPrefill && !folded && m.type == kTypeQ8_0 &&
                seqLen >= 8 && (nCols & 511u) == 0 && nRows >= 128) {
                const bool ok8 = int8Gemm(static_cast<int>(m.type), m.q, x, y, seqLen,
                                          nRows, nCols, m.blocksPerRow,
                                          reinterpret_cast<void *>(g_stream));
                // The fault is asynchronous: sync before trusting the result.
                const cudaError_t e8 = cudaStreamSynchronize(g_stream);
                if (e8 != cudaSuccess) {
                    std::fprintf(stderr, "[q8int8] rows=%u cols=%u seq=%u: %s\n",
                                 nRows, nCols, seqLen, cudaGetErrorString(e8));
                    cudaGetLastError();
                } else if (ok8) {
                    return true;
                }
            }
            // Q8_0 prefill: dequantize to F16 and use a cuBLAS tensor-core
            // GEMM instead of the per-token GEMV loop (73 tok/s).  This is
            // the batch path used by other GGUF engines for Q8.
            if (batchPrefill && !folded && m.type == kTypeQ8_0) {
                // NOTE: routing Q8_0 through launchQGemm was measured at 2.1
                // tok/s (vs 2,214 with the cuBLAS fp16 path) -- the integer
                // Q8_K batched kernel is not viable for Q8_0 weights.  The
                // int8 tensor-core path (int8Gemm -> load_tiles_q8_0, which
                // wants 2-byte quant pairs) was re-probed on 2026-11-11 and
                // still takes an illegal memory access on real Q8_0 weights
                // (it falls back to CPU and the logits diverge: 15.939 vs the
                // 15.610 CPU oracle).  The F16 dequant + cuBLAS GEMM below is
                // therefore the correct path, and this model's prefill gap is
                // the price of the 8.9 GB -> 17.8 GB F16 expansion (~35 GB of
                // extra traffic per 512-token prefill), not a scheduling bug.
                std::string q8Err;
                if (gemmQ8Cublas(m, x, y, seqLen, q8Err)) return true;
            }
            if (batchPrefill && !folded && nCols == m.blocksPerRow * qkBlock) {
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
            // EXL3 prefill: decode the trellis to F16 ONCE per matrix, then a
            // single fp16 tensor-core GEMM.  The per-token loop below re-reads
            // the packed trellis for every token (5.3 GB x seqLen), which is
            // why batch-512 prefill measured 5.8 tok/s.
            if (seqLen > 1 && m.type == kTypeEXL3 && !folded) {
                std::string exl3BErr;
                if (gemmExl3Batch(m, x, y, seqLen, exl3BErr, siluMul)) {
                    if (siluDone != nullptr) *siluDone = (siluMul != nullptr);
                    return true;
                }
            }
            // Prism PQ2_0 (Bonsai) prefill: fold the whole activation batch once
            // and run one fp16 tensor-core GEMM over the fp16-expanded ternary
            // weights.  The per-token loop below streams all 6.85 GB of packed
            // weights per prompt token, which is what made pp512 (16.8) no faster
            // than tg128 (15.3) on this model.  A silu fusion request is not
            // handled here: returning without setting siluDone leaves *siluDone
            // == false so the caller runs kSiluMul and calls again, exactly as for
            // the other batch paths that cannot fuse it.
            if (seqLen > 1 && folded && siluMul == nullptr && !pq2BatchDisabled()) {
                // PQ2_0 prefill: the packed 2-bit weights are decoded inside
                // the int8 tensor-core GEMM's tile loader (load_tiles_pq2_0),
                // so no fp16/int8 twin is materialised.  Measured pp512 869.5
                // against 680.6 for the fp16-twin + cuBLAS path, decode
                // unchanged, logits bit-identical (198 @ 17.122244, same top-5).
                // The win is the 163 ms of twin writes, not a faster GEMM: the
                // GEMM runs at ~43 TFLOPS against cuBLAS's 48.4.  TINYCODER_PQ2_INT8=0
                // restores the fp16 twin as the A/B reference.
                static const bool pq2Int8On = [] {
                    const char *e = std::getenv("TINYCODER_PQ2_INT8");
                    return e == nullptr || e[0] != '0';
                }();
                std::string pq2Err;
                if (pq2Int8On) {
                    if (gemmPQ2Int8(m, x, y, seqLen, gdnPerm, pq2Err,
                                    preFolded)) {
                        return true;
                    }
                    static bool warnedInt8 = false;
                    if (!warnedInt8) {
                        warnedInt8 = true;
                        std::fprintf(stderr,
                                     "[pq2int8] declined rows=%u cols=%u "
                                     "seq=%u: %s\n",
                                     nRows, nCols, seqLen, pq2Err.c_str());
                    }
                }
                // The fp16-twin expansion hardcodes 34-byte PQ2_0 blocks; the
                // native PTQ1_0 path has no twin, so a short batch (seq < 8,
                // where int8Gemm declines) falls through to the per-token GEMV.
                if (m.type == kTypePQ2_0 &&
                    gemmPQ2Cublas(m, x, y, seqLen, gdnPerm, pq2Err)) {
                    return true;
                }
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    std::fprintf(stderr,
                                 "[pq2batch] declined rows=%u cols=%u seq=%u: %s\n",
                                 nRows, nCols, seqLen, pq2Err.c_str());
                }
            }
            // Decode on folded PQ2_0 weights: quantise the folded activation
            // to q8 (or reuse pq2PreFold's) and consume it with the dp4a GEMV
            // (kQGemvPQ2_0Q8F, the PrismML fork's mapping).  The fold writes and
            // the GEMV reads the same buffers on g_stream in that order.
            // The gdnPerm case (ssm_out) stays on the fp32-activation path: the
            // permuted fold produces a non-finite q8 buffer for that shape
            // (measured: every projection after it inherits NaN), so the q8
            // path covers the other folded projections -- 92% of the decode
            // weight bytes.  TINYCODER_PQ2_Q8=0 restores fp32-activation decode
            // for every projection (A/B oracle).
            const bool pq2Q8 = folded && pq2Q8Enabled() &&
                               (m.type != kTypePTQ1_0 || ptq1Q8Enabled()) &&
                               prismXq_ != nullptr && prismXd_ != nullptr &&
                               (nCols % 128u) == 0u &&
                               m.blocksPerRow * 128u == m.cols;
            for (uint32_t st = 0; st < seqLen; ++st) {
                const float *xp = x + static_cast<size_t>(st) * nCols;
                if (pq2Q8) {
                    static int dbgPQ2 = 0;
                    static const bool dbgOn = [] {
                        const char *e = std::getenv("TINYCODER_PT_DBG");
                        return e != nullptr && e[0] == '1';
                    }();
                    if (dbgOn && dbgPQ2 < 8) {
                        std::fprintf(stderr,
                                     "[pq2dbg] type=%d rows=%u cols=%u folded=%d "
                                     "gdnPerm=%d\n",
                                     (int) m.type, nRows, nCols, (int) folded,
                                     (int) gdnPerm);
                        ++dbgPQ2;
                    }
                    // siluMul != nullptr means the caller left x in its pre-
                    // silu state for us to absorb (ffn_down decode).
                    if (!preFolded) prismFoldToQ8(xp, nCols, gdnPerm, siluMul);
                    static const bool twin = [] {
                        const char *e = std::getenv("TINYCODER_PQ2_TWIN");
                        return e != nullptr && e[0] == '1';
                    }();
                    if (twin) {
                        kQGemvPQ2_0Twin<<<(nRows + 15u) / 16u, 128, 0, g_stream>>>(
                                static_cast<const uint8_t *>(m.q),
                                y + static_cast<size_t>(st) * nRows, nRows,
                                m.rowBytes, m.blocksPerRow);
                    } else if (m.type == kTypePTQ1_0) {
                        launchQGemvPTQ1_0Q8(static_cast<const uint8_t *>(m.q),
                                            prismXqT_, prismXd_,
                                            y + static_cast<size_t>(st) * nRows,
                                            nRows, nCols, m.rowBytes,
                                            m.blocksPerRow);
                    } else {
                        launchQGemvPQ2_0Q8(static_cast<const uint8_t *>(m.q),
                                           prismXq_, prismXd_,
                                           y + static_cast<size_t>(st) * nRows,
                                           nRows, nCols, m.rowBytes,
                                           m.blocksPerRow);
                    }
                    continue;
                }
                if (m.type == kTypePTQ1_0 && ptq1GenericEnabled()) {
                    // Fused fold+quantize (A2): one launch instead of fold +
                    // quantize, writing the SoA words the GEMV reads.
                    static const bool fuseFold = [] {
                        const char *e = std::getenv("TINYCODER_PT_FUSE");
                        return e != nullptr && e[0] == '1';
                    }();
                    const float *fsigns = folded ? prismSignsFor(nCols) : nullptr;
                    const bool fperm = folded && gdnPerm &&
                                       prismGdnVGrouped_ && prismRep_ > 1u &&
                                       prismHeadV_ > 0u && prismNk_ > 0u;
                    const bool fusable =
                            folded && fuseFold && (fsigns != nullptr || fperm);
                    const float *xf =
                            (folded && !fusable) ? prismFoldForward(xp, nCols, gdnPerm)
                                                 : xp;
                    int32_t *genWords = ptq1Words_;
                    const uint32_t groups = (m.blocksPerRow + 31u) / 32u;
                    const size_t need = static_cast<size_t>(groups) * 1152u *
                                        sizeof(int32_t);
                    if (ptq1Words_ == nullptr || ptq1WordBytes_ < need) {
                        if (ptq1Words_ != nullptr) cudaFree(ptq1Words_);
                        ptq1Words_ = nullptr;
                        if (cudaMalloc(reinterpret_cast<void **>(&ptq1Words_),
                                       need) == cudaSuccess)
                            ptq1WordBytes_ = need;
                        genWords = ptq1Words_;
                    }
                    if (genWords != nullptr && fusable) {
                        // Reuse the folded+quantized words across projections
                        // that share the activation within one layer (q/k/v,
                        // gate/up).  Keyed on (epoch, source, width, perm):
                        // the epoch changes per layer, so an address reused by
                        // the next layer never serves a stale fold.
                        const bool foldHit =
                                foldCacheSrc_ == xp && foldCacheCols_ == nCols &&
                                foldCacheEpoch_ == foldEpoch_ &&
                                foldCachePerm_ == fperm &&
                                foldCacheSilu_ == (siluMul != nullptr);
                        static const bool cacheDbg = [] {
                            const char *e =
                                    std::getenv("TINYCODER_PT_CACHE_DBG");
                            return e != nullptr && e[0] == '1';
                        }();
                        if (cacheDbg) {
                            static uint32_t nHit = 0, nMiss = 0;
                            if (foldHit) ++nHit;
                            else
                                ++nMiss;
                            if ((nHit + nMiss) % 401u == 0u) {
                                std::fprintf(stderr,
                                             "[foldcache] hit=%u miss=%u\n",
                                             nHit, nMiss);
                                nHit = nMiss = 0;
                            }
                        }
                        if (foldHit) {
                            if (launchQGemvPTQ1_0GenericWords(
                                        static_cast<const uint8_t *>(m.q),
                                        genWords,
                                        y + static_cast<size_t>(st) * nRows,
                                        nRows, nCols, m.rowBytes,
                                        m.blocksPerRow))
                                continue;
                        }
                        const uint32_t fgrid =
                                (nCols + kPrismBlock - 1u) / kPrismBlock;
                        // siluMul MUST be forwarded: for ffn_down the caller
                        // leaves x in its pre-silu state and expects the fold to
                        // absorb silu(x)*mul.  Dropping it (nullptr) fed the
                        // un-gated activation into the down projection, which
                        // is invisible in a single-step logits probe only if the
                        // step is prefill; every decode token was wrong.
                        kPrismFoldQuantSoA<<<fgrid, foldThreads(), 0,
                                             g_stream>>>(
                                xp, genWords, nCols, fsigns,
                                fperm ? prismHeadV_ : 0u,
                                fperm ? prismNk_ : 0u,
                                fperm ? prismRep_ : 0u, siluMul,
                                ptLayoutSel());
                        foldCacheSrc_ = xp;
                        foldCacheCols_ = nCols;
                        foldCacheEpoch_ = foldEpoch_;
                        foldCachePerm_ = fperm;
                        foldCacheSilu_ = (siluMul != nullptr);
                        if (launchQGemvPTQ1_0GenericWords(
                                    static_cast<const uint8_t *>(m.q), genWords,
                                    y + static_cast<size_t>(st) * nRows, nRows,
                                    nCols, m.rowBytes, m.blocksPerRow))
                            continue;
                    }
                    if (genWords != nullptr &&
                        launchQGemvPTQ1_0Generic(
                                static_cast<const uint8_t *>(m.q), xf, genWords,
                                y + static_cast<size_t>(st) * nRows, nRows,
                                nCols, m.rowBytes, m.blocksPerRow)) {
                        // Kernel-only rate report (TINYCODER_PT_RATE=1): times
                        // 20 back-to-back launches per shape, folding the
                        // quantize in, so the frame can be compared with the
                        // stage trace to separate kernel from fold/launch.
                        static const int ptRateMode = [] {
                            const char *e = std::getenv("TINYCODER_PT_RATE");
                            return e == nullptr ? 0 : (e[0] == '2' ? 2 : (e[0] == '1' ? 1 : 0));
                        }();
                        if (ptRateMode == 2) {
                            // In-situ attribution: time every real GEMV call
                            // with its own event pair and report the summed
                            // kernel time against the wall clock, so the
                            // inter-kernel idle can be separated from the
                            // kernel work.
                            static double accMs = 0.0;
                            static long calls = 0;
                            static cudaEvent_t ia = nullptr, ib = nullptr;
                            if (ia == nullptr) {
                                cudaEventCreate(&ia);
                                cudaEventCreate(&ib);
                            }
                            cudaEventRecord(ia, g_stream);
                            const bool ok2 = launchQGemvPTQ1_0Generic(
                                    static_cast<const uint8_t *>(m.q), xf,
                                    genWords, y + static_cast<size_t>(st) * nRows,
                                    nRows, nCols, m.rowBytes, m.blocksPerRow);
                            cudaEventRecord(ib, g_stream);
                            cudaEventSynchronize(ib);
                            float ms2 = 0.0f;
                            cudaEventElapsedTime(&ms2, ia, ib);
                            accMs += ms2;
                            ++calls;
                            if (calls % 768 == 0) {
                                std::fprintf(stderr,
                                             "[ptreal] calls=%ld sum=%.3f ms "
                                             "avg=%.4f ms/call\n",
                                             calls, accMs, accMs / 768.0);
                                accMs = 0.0;
                                calls = 0;
                            }
                            if (ok2) continue;
                        }
                        if (ptRateMode == 1) {
                            static uint64_t seen[16];
                            static int nseen = 0;
                            const uint64_t key =
                                    (static_cast<uint64_t>(nRows) << 32) | nCols;
                            bool known = false;
                            for (int i = 0; i < nseen; ++i)
                                if (seen[i] == key) known = true;
                            if (!known && nseen < 16) {
                                seen[nseen++] = key;
                                const uint8_t *wq =
                                        static_cast<const uint8_t *>(m.q);
                                float *outp = y + static_cast<size_t>(st) * nRows;
                                cudaEvent_t ea, eb;
                                cudaEventCreate(&ea);
                                cudaEventCreate(&eb);
                                for (int r = 0; r < 5; ++r)
                                    launchQGemvPTQ1_0Generic(
                                            wq, xf, genWords, outp, nRows, nCols,
                                            m.rowBytes, m.blocksPerRow);
                                cudaEventRecord(ea, g_stream);
                                for (int r = 0; r < 20; ++r)
                                    launchQGemvPTQ1_0Generic(
                                            wq, xf, genWords, outp, nRows, nCols,
                                            m.rowBytes, m.blocksPerRow);
                                cudaEventRecord(eb, g_stream);
                                cudaEventSynchronize(eb);
                                float ms = 0.0f;
                                cudaEventElapsedTime(&ms, ea, eb);
                                const double per = ms / 20.0;
                                const double gb =
                                        static_cast<double>(nRows) * m.rowBytes /
                                        1e9;
                                std::fprintf(stderr,
                                             "[ptrate] rows=%u cols=%u bpr=%u "
                                             "ms=%.4f GB=%.3f GBps=%.1f\n",
                                             nRows, nCols, m.blocksPerRow, per,
                                             gb, gb / (per * 1e-3));
                                cudaEventDestroy(ea);
                                cudaEventDestroy(eb);
                            }
                        }
                        continue;
                    }
                }
                if (m.type == kTypePTQ1_0 && ptq1KsEnabled()) {
                    const float *xf =
                            folded ? prismFoldForward(xp, nCols, gdnPerm) : xp;
                    static int32_t *ksWords = nullptr;
                    static size_t ksWordBytes = 0;
                    const uint32_t groups = (m.blocksPerRow + 31u) / 32u;
                    const size_t need =
                            static_cast<size_t>(groups) * 1152u * sizeof(int32_t);
                    if (ksWords == nullptr || ksWordBytes < need) {
                        if (ksWords != nullptr) cudaFree(ksWords);
                        ksWords = nullptr;
                        if (cudaMalloc(reinterpret_cast<void **>(&ksWords),
                                       need) == cudaSuccess)
                            ksWordBytes = need;
                    }
                    if (ksWords != nullptr &&
                        launchQGemvPTQ1_0Stationary(
                                static_cast<const uint8_t *>(m.q), xf, ksWords,
                                y + static_cast<size_t>(st) * nRows, nRows,
                                nCols, m.rowBytes, m.blocksPerRow))
                        continue;
                }
                if (m.type == kTypePTQ1_0 && ptq1SplitKEnabled()) {
                    const float *xf =
                            folded ? prismFoldForward(xp, nCols, gdnPerm) : xp;
                    launchQGemvPTQ1_0SplitQ8(
                            static_cast<const uint8_t *>(m.q), xf, prismXq_,
                            prismXd_, y + static_cast<size_t>(st) * nRows, nRows,
                            nCols, m.rowBytes, m.blocksPerRow);
                    continue;
                }
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
            ++foldEpoch_;
            const DeviceLayer &w = layers_[L];

            // ---- Attention / recurrent block ----
            kRMSNormRow<<<seqLen, 1024, 1024 * sizeof(float), g_stream>>>(
                    s.hidden, s.norm, w.rmsNormAttn, H, seqLen, 1e-6f);

            static bool tprof = std::getenv("TINYCODER_TRUNK_PROF") != nullptr;
            static cudaEvent_t tev0 = nullptr, tev1 = nullptr;
            if (tprof) {
                if (tev0 == nullptr) {
                    cudaEventCreateWithFlags(&tev0, cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&tev1, cudaEventDisableTiming);
                }
                cudaEventRecord(tev0, g_stream);
                if (L == 0)
                    std::fprintf(stderr, "[trunk] marks ON (L=0)\n");
            }
            if ((L + 1) % geom_.fullAttentionInterval != 0) {
                // =================== Recurrent (gated delta net) ===========
                const bool pfQkv = pq2PreFold(w.attnQKV, s.norm, H);
                if (!gemv(w.attnQKV, s.norm, s.qkv, qkvDim, H, "attnQKV",
                          nullptr, nullptr, pfQkv) ||
                    !gemv(w.attnGate, s.norm, s.q35z, dInner, H, "attnGate",
                          nullptr, nullptr, pfQkv)) {
                    return false;
                }
                // The two 48-row F32 GDN projections (ssm_alpha, ssm_beta)
                // plus their per-head epilogue are ONE launch in decode: one
                // block per (matrix, head) computes the same per-row dot as
                // kQ35F32Gemv and applies the same expression as
                // kQ35BetaGate, so this is a pure launch fusion (3 -> 1) and
                // the recurrent state is bit-identical.
                const bool fusedAlphaBeta =
                        seqLen == 1 && w.ssmBetaQ.type == kTypeF32 &&
                        w.ssmAlphaQ.type == kTypeF32 &&
                        w.ssmBetaQ.q != nullptr && w.ssmAlphaQ.q != nullptr &&
                        w.ssmDtBiasFull != nullptr && w.ssmABroadcast != nullptr;
                // Opt-in only: the merged norm+beta kernel measured 60.87 tg
                // against 62.9 for the two separate launches -- coupling a
                // 40-block stage (norm) with an 80-block one (the dots) halves
                // the dot parallelism, which costs more than the launch saves.
                static const bool gdnMergeOn = [] {
                    const char *e = std::getenv("TINYCODER_GDN_MERGE");
                    return e != nullptr && e[0] == '1';
                }();
                if (fusedAlphaBeta && !(gdnMergeOn && seqLen == 1)) {
                    // 1024 threads/block instead of 256: the shape is 2*nVHeads
                    // blocks reading 40 KB each, which measured 8.9 us at 225
                    // GB/s (latency-bound with only 80 blocks x 256 threads).
                    // The kernel body is unchanged; only the per-thread trip
                    // count and the cross-warp reduction width change.
                    // 1024 threads/block instead of 256: this shape is 2*nVHeads
                    // blocks reading 40 KB each, which measured 8.9 us at 225 GB/s
                    // (latency-bound with only 80 blocks x 256 threads).  The
                    // kernel body is unchanged; only the per-thread trip count and
                    // the cross-warp reduction width change.
                    kQ35F32GemvDualBeta<<<2u * nVHeads, 1024, 0, g_stream>>>(
                            s.norm,
                            reinterpret_cast<const float *>(w.ssmAlphaQ.q),
                            reinterpret_cast<const float *>(w.ssmBetaQ.q),
                            s.q35GateV, s.q35Beta, w.ssmDtBiasFull,
                            w.ssmABroadcast, nVHeads, H);
                } else if (!gemv(w.ssmBetaQ, s.norm, s.q35Beta, nVHeads, H,
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
                q35Mark(8 + 4 * L);
                // beta=sigmoid(beta); gateV=softplus(alpha+dt.bias)*ssm_a.
                // Fused into kQ35F32GemvDualBeta's epilogue when that ran.
                if (!fusedAlphaBeta) {
                    kQ35BetaGate<<<seqLen, 64, 0, g_stream>>>(
                            s.q35Beta, s.q35GateV, w.ssmDtBiasFull,
                            w.ssmABroadcast, s.q35Beta, s.q35GateV, seqLen,
                            nVHeads);
                }

                // Sequential recurrence over tokens (per-layer persistent state).
                float *q35ConvState =
                        q35ConvState_ + static_cast<size_t>(L) * (convK - 1) * convChannels;
                float *q35GdnState =
                        q35GdnState_ + static_cast<size_t>(L) * nVHeads * headV * headV;
                const float *convW = static_cast<const float *>(w.ssmConv1d.q);
                const uint32_t convBlocks = (convChannels + 255) / 256;
                if (seqLen > 1) {
                    // CHUNK SCAN (2026-11-04): one conv kernel over all prompt
                    // channels, one batched L2-norm launch, one GDN scan kernel
                    // per value head.  Bit-exact with the per-token launches
                    // below (same expressions/order, same persistent state
                    // layout) but executes the whole chunk with three launches
                    // per layer instead of 3*seqLen.
                    kQ35ConvParallel<<<dim3(convBlocks, seqLen), 256, 0,
                                       g_stream>>>(
                            convW, s.qkv, s.convOut, q35ConvState, convChannels,
                            convK, seqLen);
                    kQ35ConvStateUpdate<<<convBlocks, 256, 0, g_stream>>>(
                            s.qkv, q35ConvState, convChannels, convK, seqLen);
                    kQ35L2NormHeads<<<dim3(seqLen, nVHeads), 128, 0, g_stream>>>(
                            s.convOut, s.qN, s.kN, s.vN, headK, headV, nKHeads,
                            nVHeads, keyDim, valueDim, seqLen,
                            geom_.gdnVHeadsTiled);
                    // TINYCODER_DUMP_NORM=1: dump GPU L0 token-0 convOut
                    // (post silu(conv1d(qkv))) to bisect against [cpuconv].
                    {
                        const char *dnC = std::getenv("TINYCODER_DUMP_NORM");
                        if (L == 0 && dnC != nullptr && dnC[0] != '\0' &&
                            dnC[0] != '0') {
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
                    // Warp-per-row scan: a warp owns whole state rows (4
                    // columns per lane for 32 lanes = one 128-wide row), and
                    // those rows' state lives in registers for the whole chunk.
                    //
                    // 4 rows per warp (RPW=4) with 4 warps per block = 16 rows
                    // per block, so nVHeads*8 blocks; the per-(token, head)
                    // (k,q) slice was otherwise re-fetched once per block, i.e.
                    // 32x per token per head.  Measured on the real geometry
                    // (tools/gdn_scan_probe.cu, warm clocks, min of 5 rounds):
                    // 1 row/warp 1.646 ms per layer -> 4 rows/warp 1.234 ms
                    // (-25%), and the result is BIT-IDENTICAL (every row keeps
                    // its own 32-lane dot and its own summation order; only the
                    // per-token scalar loads are hoisted one token earlier).
                    // 8 lanes per row / 4 rows per warp: the two lowest tree
                    // levels collapse into register adds and only 3 shuffles
                    // remain per dot.  Same partial products and same addition
                    // order as the 32-lane tree, verified bitwise in
                    // tools/gdn_scan_probe.cu.  Probe, warm clocks, min of 5:
                    // 32-lane 1 row/warp 1.655 ms per layer -> this 1.015 ms
                    // (-39%); in the engine 1.101 -> 0.90 ms (53.0 -> ~43 ms).
                    constexpr uint32_t kGdnWarps = 4u;
                    constexpr uint32_t kGdnRowsPerBlock = kGdnWarps * 4u;
                    // 128 = the HEADV the kernel is instantiated with.
                    const uint32_t gdnBlocks =
                            nVHeads * ((128u + kGdnRowsPerBlock - 1u) /
                                       kGdnRowsPerBlock);
                    kQ35GdnWarpScan8<128, kGdnWarps>
                            <<<gdnBlocks, kGdnWarps * 32u, 0, g_stream>>>(
                                    s.qN, s.kN, s.vN, s.q35Beta, s.q35GateV,
                                    q35GdnState, s.gdnOut, nVHeads, valueDim,
                                    seqLen, invSqrtV);
                } else {
                    for (uint32_t st = 0; st < seqLen; ++st) {
                        const float *qkv = s.qkv + static_cast<size_t>(st) * qkvDim;
                        static const bool convSplit = [] {
                            const char *e = std::getenv("TINYCODER_GDN_CONV4");
                            return e == nullptr || e[0] != '0';
                        }();
                        if (convSplit && convK == 4u) {
                            kQ35ConvStep4<<<(convChannels * 4u + 255u) / 256u,
                                            256, 0, g_stream>>>(
                                    convW, qkv, s.convIn, s.convOut,
                                    q35ConvState, convChannels);
                        } else {
                            kQ35ConvStep<<<convBlocks, 256, 0, g_stream>>>(
                                    convW, qkv, s.convIn, s.convOut,
                                    q35ConvState, convChannels, convK);
                        }
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
                        if (gdnMergeOn) {
                            kQ35NormBetaAlpha<<<dim3(1, nVHeads), 1024, 0,
                                                g_stream>>>(
                                    s.convOut, s.qN, s.kN, s.vN, s.norm,
                                    reinterpret_cast<const float *>(w.ssmAlphaQ.q),
                                    reinterpret_cast<const float *>(w.ssmBetaQ.q),
                                    s.q35GateV, s.q35Beta, w.ssmDtBiasFull,
                                    w.ssmABroadcast, headK, headV, nKHeads,
                                    nVHeads, keyDim, valueDim,
                                    geom_.gdnVHeadsTiled, H);
                        } else {
                            kQ35L2NormHeads<<<dim3(1, nVHeads), 128, 0, g_stream>>>(
                                    s.convOut, s.qN, s.kN, s.vN, headK, headV,
                                    nKHeads, nVHeads, keyDim, valueDim, 1,
                                    geom_.gdnVHeadsTiled);
                        }
                        // Same warp-per-row kernel as the chunk prefill (seqLen=1
                        // iteration) so batch and sequential are identical.
                        kQ35GdnWarpScan<128><<<nVHeads * 32u, 128, 0, g_stream>>>(
                                s.qN, s.kN, s.vN,
                                s.q35Beta + static_cast<size_t>(st) * nVHeads,
                                s.q35GateV + static_cast<size_t>(st) * nVHeads,
                                q35GdnState,
                                s.gdnOut + static_cast<size_t>(st) * valueDim, nVHeads,
                                valueDim, 1, invSqrtV);
                    }
                }
                {
                    const char *dg0 = std::getenv("TINYCODER_DUMP_GDN");
                    if (dg0 != nullptr && dg0[0] != '\0' && dg0[0] != '0') {
                        cudaStreamSynchronize(g_stream);
                        const size_t lastT = static_cast<size_t>(seqLen - 1);
                        std::vector<float> raw(valueDim), wnorm(headV), zrow(8);
                        cudaMemcpy(raw.data(), s.gdnOut + lastT * valueDim,
                                   valueDim * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        cudaMemcpy(wnorm.data(), w.ssmNorm,
                                   headV * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(zrow.data(), s.q35z + lastT * dInner,
                                   8 * sizeof(float), cudaMemcpyDeviceToHost);
                        double rs = 0.0, ws = 0.0;
                        float wmn = 1e30f, wmx = -1e30f;
                        uint32_t bad = 0;
                        for (float x: raw) {
                            if (!std::isfinite(x)) ++bad;
                            else
                                rs += static_cast<double>(x) * x;
                        }
                        for (float x: wnorm) {
                            ws += static_cast<double>(x) * x;
                            wmn = std::min(wmn, x);
                            wmx = std::max(wmx, x);
                        }
                        std::fprintf(stderr,
                                     "[gdnp %3u] preNormOut=%.4f nonfinite=%u | "
                                     "ssmNorm[0..4]={%.5f %.5f %.5f %.5f %.5f} "
                                     "min=%.5f max=%.5f nrm=%.4f | "
                                     "z[0..7]={%.4f %.4f %.4f %.4f %.4f %.4f "
                                     "%.4f %.4f} raw[0..3]={%.5f %.5f %.5f %.5f}\n",
                                     L, std::sqrt(rs), bad, wnorm[0], wnorm[1],
                                     wnorm[2], wnorm[3], wnorm[4], wmn, wmx,
                                     std::sqrt(ws), zrow[0], zrow[1], zrow[2],
                                     zrow[3], zrow[4], zrow[5], zrow[6], zrow[7],
                                     raw[0], raw[1], raw[2], raw[3]);
                    }
                }
                // Gated RMSNorm (per value-head) × silu(z) over the batch.
                q35Mark(9 + 4 * L);// after the scan / attention, before the gated norm
                kQ35GatedRmsNormHeads<<<dim3(seqLen, nVHeads), 128, 0, g_stream>>>(
                        s.q35z, s.gdnOut, w.ssmNorm, headV, nVHeads, seqLen);
                if (tprof) {
                    cudaEventRecord(tev1, g_stream);
                    cudaEventSynchronize(tev1);
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, tev0, tev1);
                    std::fprintf(stderr,
                                 "[trunk] L=%u recurrent=%.2fms seqLen=%u\n",
                                 L, ms, seqLen);
                }
                // TINYCODER_DUMP_GDN=1: census of every stage of the
                // recurrent branch (last token) plus the persistent delta-net
                // state, so an overflow inside the GDN can be attributed to a
                // stage instead of guessed at.
                {
                    const char *dg = std::getenv("TINYCODER_DUMP_GDN");
                    if (dg != nullptr && dg[0] != '\0' && dg[0] != '0') {
                        cudaStreamSynchronize(g_stream);
                        auto census = [](const float *dev, size_t n, uint32_t &bad,
                                         float &mx, double *sq = nullptr) {
                            std::vector<float> v(n);
                            cudaMemcpy(v.data(), dev, n * sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            double acc = 0.0;
                            for (size_t i = 0; i < n; ++i) {
                                if (!std::isfinite(v[i])) ++bad;
                                else {
                                    mx = std::max(mx, std::fabs(v[i]));
                                    acc += static_cast<double>(v[i]) * v[i];
                                }
                            }
                            if (sq != nullptr) *sq = acc;
                        };
                        uint32_t qb = 0, cb = 0, ob = 0, gb = 0, sb = 0, zb = 0;
                        float qm = 0, cm = 0, om = 0, gm = 0, sm = 0, zm = 0;
                        const size_t last = static_cast<size_t>(seqLen - 1);
                        double qSq = 0, cSq = 0, oSq = 0, zSq = 0;
                        census(s.qkv + last * qkvDim, qkvDim, qb, qm, &qSq);
                        census(s.convOut + last * (2 * keyDim + valueDim),
                               2 * keyDim + valueDim, cb, cm, &cSq);
                        census(s.gdnOut + last * valueDim, valueDim, ob, om, &oSq);
                        census(s.q35z + last * dInner, dInner, zb, zm, &zSq);
                        census(s.q35GateV + last * nVHeads, nVHeads, gb, gm);
                        // Gate sign: g = softplus(a+dt)*ssm_a must be NEGATIVE
                        // (ssm_a = -exp(A_log)), i.e. decay = exp(g) < 1.  A
                        // positive g makes the delta-net state grow
                        // geometrically and overflow within a few dozen tokens.
                        float gMin = 1e30f, gMax = -1e30f;
                        {
                            // All tokens, not just the last: a single positive
                            // gate at any position makes the delta-net state
                            // grow geometrically from then on.
                            std::vector<float> v(static_cast<size_t>(seqLen) *
                                                 nVHeads);
                            cudaMemcpy(v.data(),
                                       s.q35GateV,
                                       v.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            for (float x: v) {
                                if (!std::isfinite(x)) continue;
                                gMin = std::min(gMin, x);
                                gMax = std::max(gMax, x);
                            }
                        }
                        float vMax = 0.0f, kMax = 0.0f;
                        {
                            std::vector<float> v(static_cast<size_t>(seqLen) *
                                                 valueDim);
                            cudaMemcpy(v.data(), s.vN, v.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            for (float x: v) vMax = std::max(vMax, std::fabs(x));
                        }
                        {
                            std::vector<float> v(static_cast<size_t>(seqLen) *
                                                 nVHeads * headK);
                            cudaMemcpy(v.data(), s.kN, v.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            for (float x: v) kMax = std::max(kMax, std::fabs(x));
                        }
                        census(q35GdnState,
                               static_cast<size_t>(nVHeads) * headV * headV, sb,
                               sm);
                        // Per-head state maxima: distinguishes "every head
                        // drifts" from "one head's decay is on the wrong side
                        // of 1".
                        float hTop = 0.0f, hSecond = 0.0f;
                        uint32_t hTopIdx = 0;
                        {
                            std::vector<float> v(static_cast<size_t>(nVHeads) *
                                                 headV * headV);
                            cudaMemcpy(v.data(), q35GdnState,
                                       v.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost);
                            const size_t per = static_cast<size_t>(headV) * headV;
                            for (uint32_t hv = 0; hv < nVHeads; ++hv) {
                                float hm = 0.0f;
                                for (size_t i = 0; i < per; ++i) {
                                    const float a = std::fabs(v[hv * per + i]);
                                    if (std::isfinite(a)) hm = std::max(hm, a);
                                }
                                if (hm > hTop) {
                                    hSecond = hTop;
                                    hTop = hm;
                                    hTopIdx = hv;
                                } else if (hm > hSecond) {
                                    hSecond = hm;
                                }
                            }
                        }
                        std::fprintf(stderr,
                                     "[gdn %3u] qkv=%.4g/%u conv=%.4g/%u "
                                     "out=%.4g/%u z=%.4g/%u "
                                     "gate=[%.4g,%.4g] state=%.6g/%u "
                                     "head_top=%u:%.4g next=%.4g vmax=%.4g "
                                     "kmax=%.4g nrm(qkv)=%.4f nrm(conv)=%.4f "
                                     "nrm(out)=%.4f nrm(z)=%.4f\n",
                                     L, qm, qb, cm, cb, om, ob, zm, zb, gMin,
                                     gMax, sm, sb, hTopIdx, hTop, hSecond, vMax,
                                     kMax, std::sqrt(qSq), std::sqrt(cSq),
                                     std::sqrt(oSq), std::sqrt(zSq));
                        cudaStreamSynchronize(g_stream);
                    }
                }
                if (!gemv(w.ssmOut, s.gdnOut, s.attnProj, H, valueDim,
                          "ssmOut")) {
                    return false;
                }
            } else {
                // =================== Full attention =======================
                const bool pfQ = pq2PreFold(w.attnQ, s.norm, H);
                if (!gemv(w.attnQ, s.norm, s.q35QGate, qgDims, H, "attnQ",
                          nullptr, nullptr, pfQ) ||
                    !gemv(w.attnK, s.norm, s.k, kvLen, H, "attnK", nullptr,
                          nullptr, pfQ) ||
                    !gemv(w.attnV, s.norm, s.v, kvLen, H, "attnV", nullptr,
                          nullptr, pfQ)) {
                    return false;
                }
                q35Mark(8 + 4 * L);
                // Per-head Q RMSNorm (Q slice = first headDim of each 2*headDim
                // fused Q+gate row) and K RMSNorm (dense, in place).
                // Fused q/k RMSNorm + partial MRoPE (one kernel each).
                kQ35QNormRoPE<<<dim3(seqLen, nHeads), 128,
                                128 * sizeof(float), g_stream>>>(
                        s.q35QGate, s.q, w.attnQNorm, hd, nDims, nHeads, seqLen,
                        pos, dpos, mropeCos_, mropeSin_);
                kQ35KNormRoPE<<<dim3(seqLen, nKV), 128, 128 * sizeof(float),
                                g_stream>>>(s.k, w.attnKNorm, hd, nDims, nKV,
                                            seqLen, pos, dpos, mropeCos_,
                                            mropeSin_);
                // Store K/V into the layer's KV cache.
                // Only FULL-ATTENTION layers reach here (the recurrent branch
                // above is the alternative), and the cache is slotted by
                // floor(L/interval) — NOT by L (see kvLayerOffset()).
                const uint32_t kvSlot = static_cast<uint32_t>(
                        kvLayerOffset(L) / kvLayerStride_);
                // In int8-store mode there is no f32 cache to point at.
                float *kvK = nullptr;
                float *kvV = nullptr;
                if (!kvStoreI8_) {
                    kvK = reinterpret_cast<float *>(
                            static_cast<uint8_t *>(kvK_) + kvLayerOffset(L));
                    kvV = reinterpret_cast<float *>(
                            static_cast<uint8_t *>(kvV_) + kvLayerOffset(L));
                }
                const bool kv8Decode = kv8_ && seqLen == 1 &&
                                       !graphCaptureActive_ && kvK8_ != nullptr;
                const float invSqrtA = 1.0f / std::sqrt(static_cast<float>(hd));
                if (kv8Decode) {
                    int8_t *k8 = static_cast<int8_t *>(kvK8_) +
                                 static_cast<size_t>(kvSlot) * kv8LayerStride_;
                    int8_t *v8 = static_cast<int8_t *>(kvV8_) +
                                 static_cast<size_t>(kvSlot) * kv8LayerStride_;
                    const size_t sCount = kv8ScaleStride_ / sizeof(float);
                    float *ks = kvK8Scale_ + static_cast<size_t>(kvSlot) * sCount;
                    float *vs = kvV8Scale_ + static_cast<size_t>(kvSlot) * sCount;
                    // Cache-hygiene fix (2026-11-15): rebuild the int8 pool for
                    // THIS layer from the f32 cache before attending over it.
                    //
                    // This block used to be guarded by one process-wide flag
                    // (`kv8Ready_`) that was set by the first full-attention
                    // layer to execute it, so every later layer skipped its own
                    // rebuild.  A prefill writes the f32 KV only, so those
                    // layers' int8 pools still held the rows of whatever
                    // generation had run before: after a truncated
                    // generate("Hello") the next question attended "Hello" as
                    // its context (measured: the answer to "What is the capital
                    // of France?" came back as "Hello! I'm a student in the
                    // field of computer science"), and after a Paris question
                    // the next question answered "Paris".  `kv8Rows_[slot]`
                    // makes the rebuild per-layer: rows [rows8, pos) are the
                    // ones a prefill left in the f32 cache but not in the pool.
                    uint32_t &rows8 = kv8Rows_[kvSlot];
                    if (!kvStoreI8_ && rows8 < pos) {
                        const uint32_t n = pos - rows8;
                        dim3 qg(n, nKV);
                        kQ35QuantKV8<<<qg, 128, 0, g_stream>>>(
                                kvK, kvV, k8, v8, ks, vs, n, nKV, hd,
                                /*dstTokenBase=*/rows8, /*srcTokenBase=*/rows8);
                        rows8 = pos;
                    }
                    kQ35QuantKV8<<<dim3(1, nKV), 128, 0, g_stream>>>(
                            s.k, s.v, k8, v8, ks, vs, 1, nKV, hd,
                            static_cast<uint32_t>(pos), /*srcTokenBase=*/0);
                    rows8 = static_cast<uint32_t>(pos) + 1u;
                    const uint32_t len8 = static_cast<uint32_t>(pos) + 1u;
                    uint32_t chunks8 = attnChunks_;
                    if (attnChunk_ != 0) {
                        const uint32_t need = (len8 + attnChunk_ - 1u) / attnChunk_;
                        chunks8 = need < chunks8 ? need : chunks8;
                        if (chunks8 == 0) chunks8 = 1;
                    }
                    if (chunks8 > 1 && attnPartialAcc_ != nullptr) {
                        kQ35WarpAttention8Split<<<dim3(1, chunks8, nHeads),
                                                  dim3(32, 1), 0, g_stream>>>(
                                s.q, k8, v8, ks, vs, attnPartialAcc_,
                                attnPartialM_, attnPartialL_, nHeads, nKV,
                                static_cast<uint32_t>(pos), chunks8,
                                attnChunk_, hd, invSqrtA);
                        kAttnCombinePartial<256>
                                <<<dim3(1, nHeads), 32, 0, g_stream>>>(
                                        attnPartialAcc_, attnPartialM_,
                                        attnPartialL_, s.attnOut, 1, nHeads,
                                        chunks8);
                    } else {
                        dim3 blk8(32, 4);
                        kQ35WarpAttention8<<<dim3((nHeads + 3) / 4), blk8, 0,
                                             g_stream>>>(
                                s.q, k8, v8, ks, vs, s.attnOut, nHeads, nKV,
                                static_cast<uint32_t>(pos), hd, invSqrtA);
                    }
                } else if (kvStoreI8_) {
                    // int8 as the store: quantize the chunk straight into the
                    // pool (one block per (token, kv-head), 128 threads) and
                    // attend over the quantized rows.  The +16-row capacity
                    // covers a decode token past a full window.
                    int8_t *k8 = static_cast<int8_t *>(kvK8_) +
                                 static_cast<size_t>(kvSlot) * kv8LayerStride_;
                    int8_t *v8 = static_cast<int8_t *>(kvV8_) +
                                 static_cast<size_t>(kvSlot) * kv8LayerStride_;
                    const size_t sCount = kv8ScaleStride_ / sizeof(float);
                    float *ks = kvK8Scale_ + static_cast<size_t>(kvSlot) * sCount;
                    float *vs = kvV8Scale_ + static_cast<size_t>(kvSlot) * sCount;
                    kQ35QuantKV8<<<dim3(seqLen, nKV), 128, 0, g_stream>>>(
                            s.k, s.v, k8, v8, ks, vs, seqLen, nKV, hd,
                            static_cast<uint32_t>(pos), /*srcTokenBase=*/0);
                    kv8Rows_[kvSlot] = static_cast<uint32_t>(pos) + seqLen;
                    // Opt-in like the f32 shared-KV variant: the warp kernel is
                    // the proven default, and the shared one needs a block per
                    // kv head (see the kernel comment).
                    static const bool attnShared8 = [] {
                        const char *e = std::getenv("TINYCODER_ATTN_MMA");
                        return e != nullptr && e[0] != '0';
                    }();
                    const uint32_t headGroup = nHeads / nKV;
                    if (attnShared8 && seqLen > 1 && headGroup >= 1 &&
                        headGroup <= 32) {
                        kQ35PrefillAttnSharedKV8<256>
                                <<<dim3(seqLen, nKV),
                                   dim3(32, headGroup), 2 * 256 * sizeof(float),
                                   g_stream>>>(s.q, k8, v8, ks, vs, s.attnOut,
                                               seqLen, nHeads, nKV, pos, dpos,
                                               invSqrtA);
                    } else {
                        constexpr uint32_t kW = 4;
                        kQ35WarpAttention8Prefill<256, kW>
                                <<<dim3(seqLen, (nHeads + kW - 1) / kW),
                                   dim3(32, kW), 0, g_stream>>>(
                                        s.q, k8, v8, ks, vs, s.attnOut, seqLen,
                                        nHeads, nKV, pos, dpos, invSqrtA);
                    }
                } else {
                    kQ35StoreKV<<<seqLen, 256, 0, g_stream>>>(
                            s.k, s.v, kvK, kvV, seqLen, nKV, hd, pos, dpos);
                    // No int8-pool bookkeeping is needed here: a prefill only
                    // adds f32 rows, and the next decode's rebuild converts
                    // exactly the rows the pool is missing (kv8Rows_[slot]).
                    dim3 blk(32, 4);
                    const char *asEnv = std::getenv("TINYCODER_ATTN_SPLIT");
                    const bool splitAttn =
                            seqLen == 1 && attnPartialAcc_ != nullptr &&
                            attnChunks_ > 1 &&
                            (asEnv == nullptr || asEnv[0] == '\0' || asEnv[0] != '0');
                    {
                        static const bool tprof3 =
                                std::getenv("TINYCODER_TRUNK_PROF") != nullptr;
                        if (tprof3) {
                            if (moeAttnEvStart_ == nullptr)
                                cudaEventCreateWithFlags(&moeAttnEvStart_,
                                                         cudaEventDisableTiming);
                            cudaEventRecord(moeAttnEvStart_, g_stream);
                        }
                    }
                    if (splitAttn) {
                        if (dpos != nullptr) {
                            kWarpAttentionSplitPos<256, float, 1>
                                    <<<dim3(1, attnChunks_, nHeads), dim3(32, 1), 0,
                                       g_stream>>>(
                                            s.q, kvK, kvV, attnPartialAcc_,
                                            attnPartialM_, attnPartialL_, 1, nHeads,
                                            nKV, dpos, attnChunks_, attnChunk_,
                                            invSqrtA);
                        } else {
                            kWarpAttentionSplit<256, float, 1>
                                    <<<dim3(1, attnChunks_, nHeads), dim3(32, 1), 0,
                                       g_stream>>>(
                                            s.q, kvK, kvV, attnPartialAcc_,
                                            attnPartialM_, attnPartialL_, 1, nHeads,
                                            nKV, pos, attnChunks_, attnChunk_,
                                            invSqrtA);
                        }
                        kAttnCombinePartial<256>
                                <<<dim3(1, nHeads), 32, 0, g_stream>>>(
                                        attnPartialAcc_, attnPartialM_,
                                        attnPartialL_, s.attnOut, 1, nHeads,
                                        attnChunks_);
                    } else {
                        // TINYCODER_ATTN_MMA=1: block owns all heads of one
                        // token; the K/V row is staged once in shared memory
                        // and read by every head warp (8x less global K/V
                        // traffic).  Same per-head math, so the result is
                        // bit-identical to kQ35WarpAttention.
                        static const bool attnShared = [] {
                            const char *e = std::getenv("TINYCODER_ATTN_MMA");
                            return e != nullptr && e[0] != '0';
                        }();
                        if (attnShared && seqLen > 1) {
                            constexpr uint32_t kW = 8;
                            kQ35PrefillAttnSharedKV<256, kW>
                                    <<<dim3(seqLen, (nHeads + kW - 1) / kW),
                                       dim3(32, kW), 2 * 256 * sizeof(float),
                                       g_stream>>>(s.q, kvK, kvV, s.attnOut,
                                                   seqLen, nHeads, nKV, pos,
                                                   nullptr, invSqrtA);
                        } else {
                            kQ35WarpAttention<256><<<dim3(seqLen, (nHeads + 3) / 4),
                                                     blk, 0, g_stream>>>(
                                    s.q, kvK, kvV, s.attnOut, seqLen, nHeads, nKV,
                                    pos, nullptr, invSqrtA);
                        }
                    }
                }
                // attnOut *= sigmoid(gate) (gate = second headDim of the fused
                // Q+gate projection).
                q35Mark(9 + 4 * L);
                kQ35SigmoidGate<<<seqLen, 256, 0, g_stream>>>(
                        s.attnOut, s.q35QGate, seqLen, nHeads, hd);
                if (!gemv(w.attnO, s.attnOut, s.attnProj, H, qLen, "attnO")) {
                    return false;
                }
                {
                    static const bool tprof2 =
                            std::getenv("TINYCODER_TRUNK_PROF") != nullptr;
                    if (tprof2) {
                        if (moeAttnEv_ == nullptr) {
                            cudaEventCreateWithFlags(&moeAttnEv_,
                                                     cudaEventDisableTiming);
                        }
                        static double acc2 = 0.0;
                        cudaEventRecord(moeAttnEv_, g_stream);
                        cudaEventSynchronize(moeAttnEv_);
                        float ms2 = 0.0f;
                        cudaEventElapsedTime(&ms2, moeAttnEvStart_, moeAttnEv_);
                        acc2 += ms2;
                        std::fprintf(stderr,
                                     "[trunk] L=%u attn+attnO=%.2fms seqLen=%u\n",
                                     L, ms2, seqLen);
                    }
                }
            }
            q35Mark(10 + 4 * L);
            // ---- Post-attention residual + RMSNorm (one fused kernel) ----
            kAddResidualRmsNorm<<<seqLen, 1024, 1024 * sizeof(float),
                                  g_stream>>>(s.hidden, s.attnProj, s.norm,
                                              w.postAttnNorm, H, seqLen, 1e-6f);
            bool fusedGU = false;
            if (seqLen == 1 && w.ffnGate.type == kTypeQ4K &&
                w.ffnUp.type == kTypeQ4K) {
                fusedGU = int8GemvDualQ4K(
                        w.ffnGate.q, w.ffnUp.q, s.norm, s.gate, s.up,
                        w.ffnGate.rows, w.ffnGate.cols,
                        reinterpret_cast<void *>(g_stream));
            }
            if (!fusedGU) {
                const bool pfGu = pq2PreFold(w.ffnGate, s.norm, H);
                if (!gemv(w.ffnGate, s.norm, s.gate, I, H, "ffnGate", nullptr,
                          nullptr, pfGu) ||
                    !gemv(w.ffnUp, s.norm, s.up, I, H, "ffnUp", nullptr,
                          nullptr, pfGu)) {
                    return false;
                }
            }
            // Dense FFN activation: silu(gate)*up.  In prefill the EXL3 down
            // projection absorbs it into its own input Hadamard
            // (kExl3FwhtToF16's `mul` argument), which deletes a whole
            // read+read+write pass over [seqLen][I] -- measured 196 us per layer,
            // 12.6 ms per 512-token prefill.  Every other route keeps the
            // standalone kernel and the original call order.
            const bool wantFusedSilu =
                    (seqLen > 1 && w.ffnDown.type == kTypeEXL3);
            // Decode: the q8 fold of ffn_down absorbs silu(gate)*up (same
            // expression), which removes one launch and one 17408-element pass
            // per layer.  The predicate is the same one the GEMV uses, so the
            // two can never disagree about whether the activation is still
            // pre-silu.
            const bool decodeSiluFold =
                    !wantFusedSilu && pq2Q8Applies(w.ffnDown, I);
            if (!wantFusedSilu) {
                if (!fusedGU && !decodeSiluFold) {
                    kSiluMul<<<(seqLen * I + 255) / 256, 256, 0, g_stream>>>(
                            s.gate, s.up, seqLen * I);
                }
                if (!gemv(w.ffnDown, s.gate, s.ffnOut, H, I, "ffnDown",
                          decodeSiluFold ? s.up : nullptr)) {
                    return false;
                }
            } else {
                bool siluDone = false;
                if (!gemv(w.ffnDown, s.gate, s.ffnOut, H, I, "ffnDown", s.up,
                          &siluDone)) {
                    return false;
                }
                if (!siluDone) {
                    // Declined (EXL3 batch unavailable): activate and redo, so
                    // the values are identical to the unfused order.
                    static bool warnedSilu = false;
                    if (!warnedSilu) {
                        warnedSilu = true;
                        std::fprintf(stderr,
                                     "[silu] down projection declined the fused "
                                     "activation; running kSiluMul + a second "
                                     "projection\n");
                    }
                    kSiluMul<<<(seqLen * I + 255) / 256, 256, 0, g_stream>>>(
                            s.gate, s.up, seqLen * I);
                    if (!gemv(w.ffnDown, s.gate, s.ffnOut, H, I, "ffnDown")) {
                        return false;
                    }
                }
            }
            kAddResidual<<<(seqLen * H + 255) / 256, 256, 0, g_stream>>>(
                    s.hidden, s.ffnOut, seqLen * H);
            q35Mark(11 + 4 * L);
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
                double apSq = 0.0;
                std::vector<float> arow(H);
                cudaMemcpy(arow.data(), s.attnProj + (seqLen - 1) * H,
                           H * sizeof(float), cudaMemcpyDeviceToHost);
                float apMax = 0.0f;
                for (uint32_t i = 0; i < H; ++i) {
                    apSq += static_cast<double>(arow[i]) * arow[i];
                    apMax = std::max(apMax, std::fabs(arow[i]));
                }
                // Magnitude census: the prefill GEMM path feeds the ACTIVATION
                // through fp16 (gemmExl3Batch / cuBLAS), so any |x| above the
                // fp16 max 65504 turns the operand into inf.  Print the peaks
                // of every tensor that becomes a GEMM operand.
                float hMax = 0.0f, nMax = 0.0f, fMax = 0.0f, gMax = 0.0f;
                float fNorm = 0.0f;
                uint32_t hBad = 0, nBad = 0, fBad = 0, gBad = 0, apBad = 0;
                for (uint32_t i = 0; i < H; ++i) {
                    if (!std::isfinite(hrow[i])) ++hBad;
                    else
                        hMax = std::max(hMax, std::fabs(hrow[i]));
                    if (!std::isfinite(arow[i])) ++apBad;
                }
                {
                    std::vector<float> v(H);
                    cudaMemcpy(v.data(), s.norm + (seqLen - 1) * H, H * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    for (uint32_t i = 0; i < H; ++i) {
                        if (!std::isfinite(v[i])) ++nBad;
                        else
                            nMax = std::max(nMax, std::fabs(v[i]));
                    }
                    cudaMemcpy(v.data(), s.ffnOut + (seqLen - 1) * H, H * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    double fSq = 0.0;
                    for (uint32_t i = 0; i < H; ++i) {
                        if (!std::isfinite(v[i])) ++fBad;
                        else {
                            fMax = std::max(fMax, std::fabs(v[i]));
                            fSq += static_cast<double>(v[i]) * v[i];
                        }
                    }
                    fNorm = static_cast<float>(std::sqrt(fSq));
                }
                {
                    std::vector<float> v(I);
                    cudaMemcpy(v.data(), s.gate + (seqLen - 1) * I, I * sizeof(float),
                               cudaMemcpyDeviceToHost);
                    for (uint32_t i = 0; i < I; ++i) {
                        if (!std::isfinite(v[i])) ++gBad;
                        else
                            gMax = std::max(gMax, std::fabs(v[i]));
                    }
                }
                std::fprintf(stderr,
                             "[gpuM %3u] maxAbs hidden=%.6g norm=%.6g "
                             "ffnOut=%.6g gate(silu)=%.6g attnProj=%.6g | "
                             "nonfinite hidden=%u norm=%u ffnOut=%u gate=%u "
                             "attnProj=%u | ffnNorm=%.4f apNorm=%.4f\n",
                             L, hMax, nMax, fMax, gMax, apMax, hBad, nBad, fBad,
                             gBad, apBad, fNorm,
                             static_cast<float>(std::sqrt(apSq)));
                std::fprintf(stderr,
                             "[gpuL %3u] post-ffn norm=%.6f fnv=%016llx "
                             "attnProjNorm=%.6f "
                             "first8=[%.6f %.6f %.6f %.6f %.6f %.6f "
                             "%.6f %.6f]\n",
                             L, std::sqrt(sq),
                             static_cast<unsigned long long>(fnv), std::sqrt(apSq),
                             hrow[0], hrow[1], hrow[2], hrow[3], hrow[4],
                             hrow[5], hrow[6], hrow[7]);
            }
        }

        if (!graphCaptureActive_) {
            cudaError_t llErr = cudaStreamSynchronize(g_stream);
            if (llErr != cudaSuccess) {
                errMsg = std::string("qwen35 prefix sync: ") +
                         cudaGetErrorString(llErr);
                return false;
            }
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
        if (std::getenv("TINYCODER_EXL3_DBG") != nullptr) {
            // Step 1 probe: is the hidden state already NaN BEFORE the LM head?
            cudaStreamSynchronize(g_stream);
            std::vector<float> hv(static_cast<size_t>(H) * seqLen);
            cudaMemcpy(hv.data(), s.hidden, hv.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            size_t bad = 0;
            float mn = 1e30f, mx = -1e30f;
            for (float v: hv) {
                if (!std::isfinite(v)) ++bad;
                else {
                    mn = std::fmin(mn, v);
                    mx = std::fmax(mx, v);
                }
            }
            std::fprintf(stderr, "[exl3hidden] H=%u seq=%u n=%zu nonfinite=%zu min=%g max=%g\n",
                         H, seqLen, hv.size(), bad, mn, mx);
        }
        const void *lmQ = lmHeadQ_ != nullptr ? lmHeadQ_ : embedQ_;
        uint32_t lmType = lmHeadQ_ != nullptr ? lmHeadType_ : embedType_;
        uint32_t lmRowBytes =
                lmHeadQ_ != nullptr ? lmHeadRowBytes_ : embedRowBytes_;
        uint32_t lmBlocks = hostBlocksPerRow(lmType, H);
        // Prism: the (folded) LM head also consumes H S P x.
        const bool lmFolded =
                (geom_.prismFold != 0) &&
                (lmType == kTypePQ2_0 || lmType == kTypePTQ1_0);
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
            static const bool lmFuse = [] {
                const char *e = std::getenv("TINYCODER_PT_FUSE");
                return e != nullptr && e[0] == '1';
            }();
            const float *lmSigns = lmFolded ? prismSignsFor(H) : nullptr;
            const bool lmFusable = lmFolded && lmFuse && lmSigns != nullptr;
            if (lmFolded && !lmFusable) lh = prismFoldForward(lh, H, false);
            // The fp32 oracle is the fallback.  A validated native 28 B kernel
            // exists in this tree now (the GEN SOA path, gated by
            // TINYCODER_PT_LMH_GEN here): the older q8 LM-head attempt that
            // collapsed the logits (17.12 -> 1.26) was a different, broken
            // variant, not this one.
            static const bool lmGen = [] {
                const char *e = std::getenv("TINYCODER_PT_LMH_GEN");
                return e != nullptr && e[0] == '1';
            }();
            bool lmDone = false;
            if (lmGen && lmType == kTypePTQ1_0) {
                static int32_t *lmWords = nullptr;
                static size_t lmWordBytes = 0;
                const uint32_t groups = (lmBlocks + 31u) / 32u;
                const size_t need = static_cast<size_t>(groups) * 1152u *
                                    sizeof(int32_t);
                if (lmWords == nullptr || lmWordBytes < need) {
                    if (lmWords != nullptr) cudaFree(lmWords);
                    lmWords = nullptr;
                    if (cudaMalloc(reinterpret_cast<void **>(&lmWords), need) ==
                        cudaSuccess)
                        lmWordBytes = need;
                }
                if (lmWords != nullptr && lmFusable) {
                    const uint32_t fgrid = (H + kPrismBlock - 1u) / kPrismBlock;
                    kPrismFoldQuantSoA<<<fgrid, foldThreads(), 0, g_stream>>>(
                            lh, lmWords, H, lmSigns, 0u, 0u, 0u, nullptr,
                            ptLayoutSel());
                    lmDone = launchQGemvPTQ1_0GenericWords(
                            static_cast<const uint8_t *>(lmQ), lmWords,
                            s.logits, geom_.vocabSize, H, lmRowBytes, lmBlocks);
                }
                if (lmWords != nullptr && !lmDone)
                    lmDone = launchQGemvPTQ1_0Generic(
                            static_cast<const uint8_t *>(lmQ), lh, lmWords,
                            s.logits, geom_.vocabSize, H, lmRowBytes, lmBlocks);
            }
            if (!lmDone)
                launchQGemv(lmType, lmQ, lh, s.logits,
                            geom_.vocabSize, H, lmRowBytes, lmBlocks, q8k_,
                            q8kBytes_);
            q35Mark(1);
            if (!graphCaptureActive_) {
                cudaMemcpyAsync(logitsOut, s.logits,
                                geom_.vocabSize * sizeof(float),
                                cudaMemcpyDeviceToHost, g_stream);
            }
        }
        if (graphCaptureActive_) {
            cudaGraph_t g = nullptr;
            if (cudaStreamEndCapture(g_stream, &g) == cudaSuccess && g != nullptr) {
                cudaGraphExec_t ex = nullptr;
                if (cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0) ==
                    cudaSuccess) {
                    graph_ = g;
                    graphExec_ = ex;
                    graphBuilt_ = true;
                    // Node count is the receipt that the whole single-token
                    // pass (folds, dp4a GEMVs, GDN scan, attention, LM head)
                    // went into the graph rather than escaping it.
                    size_t nNodes = 0;
                    cudaGraphGetNodes(g, nullptr, &nNodes);
                    std::fprintf(stderr, "[q35graph] captured %zu nodes\n",
                                 nNodes);
                } else {
                    cudaGraphDestroy(g);
                    graphCaptureFailed_ = true;
                }
            } else {
                graphCaptureFailed_ = true;
            }
            graphCaptureActive_ = false;
            // Stream capture records without executing: run the instantiated
            // graph once to produce this token's logits (kvPos_ was advanced
            // by the host-side bookkeeping above).
            if (graphBuilt_) {
                cudaGraphLaunch(static_cast<cudaGraphExec_t>(graphExec_), g_stream);
            }
            cudaStreamSynchronize(g_stream);
            cudaMemcpyAsync(logitsOut, s.logits,
                            geom_.vocabSize * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream);
        }
        cudaError_t fin = cudaStreamSynchronize(g_stream);
        if (fin != cudaSuccess) {
            errMsg = std::string("qwen35 lmhead sync: ") +
                     cudaGetErrorString(fin);
            return false;
        }
        if (q35Trace && nRealLayers > 0) {
            const double hostMs = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() -
                                          q35T0)
                                          .count();
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, q35Ev[0], q35Ev[7]);
            const double head = ms;
            double proj = 0.0, midG = 0.0, postG = 0.0, midA = 0.0,
                   postA = 0.0, ffn = 0.0;
            for (uint32_t L = 0; L < nGpu && L < nRealLayers; ++L) {
                const cudaEvent_t prev =
                        (L == 0) ? q35Ev[7] : q35Ev[11 + 4 * (L - 1)];
                cudaEventElapsedTime(&ms, prev, q35Ev[8 + 4 * L]);
                proj += ms;
                const bool gdn = (geom_.fullAttentionInterval == 0) ||
                                 ((L + 1) % geom_.fullAttentionInterval != 0);
                cudaEventElapsedTime(&ms, q35Ev[8 + 4 * L], q35Ev[9 + 4 * L]);
                if (gdn) midG += ms;
                else
                    midA += ms;
                cudaEventElapsedTime(&ms, q35Ev[9 + 4 * L], q35Ev[10 + 4 * L]);
                if (gdn) postG += ms;
                else
                    postA += ms;
                cudaEventElapsedTime(&ms, q35Ev[10 + 4 * L], q35Ev[11 + 4 * L]);
                ffn += ms;
            }
            const cudaEvent_t last = q35Ev[11 + 4 * (nRealLayers - 1)];
            cudaEventElapsedTime(&ms, last, q35Ev[1]);
            const double tail = ms;
            const double state = midG + postG + midA + postA;
            const double gpu = head + proj + state + ffn + tail;
            std::fprintf(stderr,
                         "[q35trace] head=%.2f proj=%.2f (attn-only=%.2f) "
                         "state=%.2f ffn=%.2f tail=%.2f | gpu=%.2f host=%.2f "
                         "gap=%.2f ms/token\n",
                         head, proj, midA, state, ffn, tail, gpu, hostMs,
                         hostMs - gpu);
            std::fprintf(stderr,
                         "[q35trace] state split: gdn scan=%.2f rest=%.2f | "
                         "attn kernels=%.2f attnO=%.2f\n",
                         midG, postG, midA, postA);
        }
        return true;
    }

    bool GPUModel::replayQwen35DecodeGraph(const std::vector<int32_t> &tokens,
                                           float *logitsOut,
                                           std::string &errMsg) {
        DeviceScratch &s = scratch_;
        const uint32_t pos = static_cast<uint32_t>(kvPos_);
        cudaMemcpyAsync(s.tokens, tokens.data(), sizeof(int32_t),
                        cudaMemcpyHostToDevice, g_stream);
        graphPosHost_ = pos;
        cudaMemcpyAsync(graphPos_, &graphPosHost_, sizeof(uint32_t),
                        cudaMemcpyHostToDevice, g_stream);
        if (cudaGraphLaunch(static_cast<cudaGraphExec_t>(graphExec_), g_stream) !=
            cudaSuccess) {
            errMsg = std::string("qwen35 graph launch: ") +
                     cudaGetErrorString(cudaGetLastError());
            graphCaptureFailed_ = true;
            return false;
        }
        if (cudaStreamSynchronize(g_stream) != cudaSuccess) {
            errMsg = "qwen35 graph sync failed";
            return false;
        }
        cudaMemcpyAsync(logitsOut, s.logits,
                        geom_.vocabSize * sizeof(float),
                        cudaMemcpyDeviceToHost, g_stream);
        if (cudaStreamSynchronize(g_stream) != cudaSuccess) {
            errMsg = "qwen35 graph logits copy failed";
            return false;
        }
        kvPos_ += 1;
        return true;
    }

}// namespace tinycoder::gpu

#endif// USE_CUDA
