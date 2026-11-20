#include "ContextPlan.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <sys/sysinfo.h>

namespace tinycoder::detail {
    namespace {

        constexpr uint64_t kMiB = 1024ull * 1024ull;
        constexpr uint64_t kGiB = 1024ull * kMiB;

        /// Left free on top of gpuFixedOverhead.  A window that "just fits" is not a
        /// window: the fp16 prefill twins and the logits scratch are best-effort and
        /// the driver may hold a little more than it reports.
        constexpr uint64_t kSafetyMargin = 256ull * kMiB;

        /// Below this nothing is granted.  A 512-token window is not a usable model.
        constexpr uint32_t kFloorSeqLen = 1024;

        /// Auto, when the caller does not ask.  Today's default, kept so every
        /// published benchmark stays comparable.
        constexpr uint32_t kAutoSeqLen = 2048;

        /// Ceiling when the metadata carries no context length at all.
        constexpr uint32_t kFallbackModelMax = 4096;

        /// Surplus RAM kept for the runtime when the K/V lives in host memory.
        constexpr uint64_t kRamSlack = 2 * kGiB;

        uint64_t satSub(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }

        uint32_t floor256(uint64_t v, uint32_t ceiling) {
            uint64_t n = (v / 256ull) * 256ull;
            if (n > ceiling) n = static_cast<uint64_t>(ceiling);
            return static_cast<uint32_t>(n);
        }

        /// One rung of the precision ladder.
        struct Rung {
            KvPrecision kv;
            bool mirror;
            bool minResidency;///< only the largest single layer's weights are charged
            uint32_t fit;     ///< tokens this rung can hold
        };

    }// namespace

    uint64_t kvStoreBytesPerToken(const ContextRequest &r, KvPrecision p, bool mirror) {
        const uint64_t kvHeads = r.numKVHeads;
        const uint64_t hd = r.headDim;
        const uint64_t elem = (p == KvPrecision::I8) ? 1ull : 4ull;
        // K and V.
        uint64_t perLayer = 2ull * kvHeads * hd * elem;
        // The int8 store carries one f32 scale per (token, kv head), for K and V.
        if (p == KvPrecision::I8) perLayer += 2ull * kvHeads * 4ull;
        // The f32 store may additionally keep an int8 decode mirror.
        if (p == KvPrecision::F32 && mirror) perLayer += 2ull * kvHeads * (hd + 4ull);
        return perLayer;
    }

    uint64_t kvBytesPerToken(const ContextRequest &r, KvPrecision p, bool mirror) {
        // RoPE tables are ctx-linear: cos+sin f32 rows of `ropeDims` per token, one
        // copy for the whole model rather than one per layer.
        return static_cast<uint64_t>(r.numAttentionLayers) *
                       kvStoreBytesPerToken(r, p, mirror) +
               4ull * static_cast<uint64_t>(r.ropeDims);
    }

    ContextPlan planContext(const ContextRequest &r) {
        ContextPlan plan;

        const uint32_t ceiling = r.modelMax ? r.modelMax : kFallbackModelMax;
        const bool gpu = r.freeVram != 0;
        const bool modelMaxUnknown = (r.modelMax == 0);

        // ---- Budget -----------------------------------------------------------
        // Every rung is charged the K/V bytes it would really keep resident; the
        // rung's weight residency is the whole model, or one layer for the
        // minimum-residency rung (an explicit long request may leave layers on the
        // CPU).
        uint64_t baseBudget = 0;
        if (gpu) {
            baseBudget = satSub(satSub(r.freeVram, r.gpuFixedOverhead), kSafetyMargin);
        } else {
            // The RAM probe is taken after the weights are resident, so the
            // host weights are already excluded from `availableRam` — subtracting
            // `weightBytes` here would count them twice.
            baseBudget = satSub(r.availableRam, kRamSlack);
        }

        std::vector<Rung> rungs;
        auto addRung = [&](KvPrecision kv, bool mirror, bool minResidency) {
            if (r.forceKvSet && kv != r.forceKv) return;
            // Weights are host-side at plan time, so a GPU rung must pay for them
            // out of the VRAM budget before the K/V does.
            const uint64_t weights =
                    minResidency && r.numLayers
                            ? r.weightBytes / r.numLayers
                            : r.weightBytes;
            const uint64_t budget = satSub(baseBudget, gpu ? weights : 0);
            const uint64_t bytes = kvBytesPerToken(r, kv, mirror);
            Rung rung{kv, mirror, minResidency, 0};
            rung.fit = bytes == 0 ? ceiling
                                  : floor256(budget / bytes, ceiling);
            rungs.push_back(rung);
        };

        if (gpu) {
            // The int8 mirror is a qwen35-family capability (`kv8_`), so it shares
            // the flag with the int8 store: a qwen2 session pays f32 only.
            if (r.int8KvSupported) {
                addRung(KvPrecision::F32, /*mirror=*/true, /*minResidency=*/false);
            }
            addRung(KvPrecision::F32, /*mirror=*/false, /*minResidency=*/false);
            if (r.int8KvSupported) {
                addRung(KvPrecision::I8, /*mirror=*/false, /*minResidency=*/false);
            }
            // No minimum-residency rung: it assumed ONE layer's weights resident, so
            // a request that only it could satisfy was granted and then ran with
            // almost the whole model on the CPU.  Measured on Bonsai-2-27B at a 64k
            // request: granted 65536 through this rung, prefill 586 s for 63983
            // tokens (109 tok/s), and the needle answer was wrong (`<<< Record
            // 000000>>>`).  The full-residency fit is ~47600 tokens there, so the
            // honest grant is a reduction with "limited by VRAM", not a 109 tok/s
            // session that cannot answer.  A user who wants layer shedding can still
            // pin TINYCODER_NGL.
        } else {
            // The host cache is f32 only, and there is no mirror to speak of.
            addRung(KvPrecision::F32, /*mirror=*/false, /*minResidency=*/false);
        }

        uint32_t bestFit = 0;
        for (const auto &rung: rungs) bestFit = std::max(bestFit, rung.fit);

        // ---- What to ask for -------------------------------------------------
        const bool clampedByModel = r.requested > ceiling;
        const uint32_t want = r.requested ? std::min(r.requested, ceiling)
                                          : std::min(kAutoSeqLen, bestFit);

        // ---- Grant -----------------------------------------------------------
        std::string reason;
        if (want < kFloorSeqLen || bestFit < kFloorSeqLen) {
            plan.seqLen = 0;
            plan.maxAllowed = 0;
            plan.kvOnGpu = gpu;
            // An auto request has no size to compare against the floor: report the
            // fit, not "requested 0" (which is what the 27B printed while the fixed
            // overhead was wrong).
            const bool requestedBelowFloor =
                    r.requested != 0 && r.requested < kFloorSeqLen;
            reason = requestedBelowFloor
                             ? "requested " + std::to_string(r.requested) +
                                       ", below the " + std::to_string(kFloorSeqLen) +
                                       "-token floor"
                             : "not even " + std::to_string(kFloorSeqLen) +
                                       " tokens fit";
            if (!gpu) reason += "; no GPU, RAM path";
            if (modelMaxUnknown) reason += "; model max unknown, assuming 4096";
            if (r.forceKvSet && !gpu) {
                reason += r.forceKv == KvPrecision::I8
                                  ? "; an int8 K/V store needs the GPU"
                                  : "; f32 K/V pinned by the caller";
            }
            plan.reason = reason;
            return plan;
        }

        // First rung that can hold the whole request wins: a short request keeps
        // today's numerics instead of being served from int8.
        const Rung *chosen = nullptr;
        for (const auto &rung: rungs) {
            if (rung.fit >= want) {
                chosen = &rung;
                break;
            }
        }
        const bool reduced = (chosen == nullptr);
        if (reduced) {
            for (const auto &rung: rungs) {
                if (rung.fit == bestFit) {
                    chosen = &rung;
                    break;
                }
            }
        }

        plan.kv = chosen->kv;
        plan.kvMirror = chosen->mirror;
        plan.kvOnGpu = gpu;
        plan.seqLen = std::min(want, chosen->fit);
        // The ladder's largest fit, NOT the granted rung's: callers print this as
        // "how big can this checkpoint go" (ctx_probe, context_gate.sh, the release
        // notes), and answering that with the rung the *current* request happened to
        // land on made the same model report 41984 for an 8k request and 196352 for
        // a 64k one.  The precision that achieves it is named in `reason`.
        plan.maxAllowed = bestFit;
        plan.kvBytes = static_cast<uint64_t>(plan.seqLen) *
                       kvBytesPerToken(r, plan.kv, plan.kvMirror);

        // ---- The sentence that reports it ------------------------------------
        if (r.requested) {
            reason = "requested " + std::to_string(r.requested);
            if (clampedByModel) {
                reason += ", clamped to model max " + std::to_string(ceiling);
            }
        } else {
            reason = "auto";
            if (bestFit > plan.seqLen) {
                // Name the precision that ACHIEVES bestFit, not the one the granted
                // window uses: the two differ whenever a cheaper rung serves the
                // request (27B: 2048 at f32+mirror, 12288 at int8), and printing
                // bestFit with the granted window's precision claimed "up to 12288
                // at f32 K/V + mirror", which that rung cannot hold.
                const Rung *widest = nullptr;
                for (const auto &rung: rungs) {
                    if (rung.fit == bestFit) {
                        widest = &rung;
                        break;
                    }
                }
                if (widest == nullptr) widest = chosen;
                reason += "; memory allows up to " + std::to_string(bestFit) + " at " +
                          (widest->kv == KvPrecision::I8 ? "int8 K/V" : "f32 K/V") +
                          (widest->mirror ? " + mirror" : "");
            }
        }
        if (reduced) {
            reason += std::string(", limited by ") + (gpu ? "VRAM" : "RAM") + ": " +
                      std::to_string(chosen->fit) + " tokens fit" +
                      (chosen->minResidency ? " with one layer resident" : "");
        }
        if (!gpu) reason += "; no GPU, RAM path";
        if (modelMaxUnknown) reason += "; model max unknown, assuming 4096";
        plan.reason = reason;
        return plan;
    }

    uint64_t probeFreeVramBytes() {
#ifdef USE_CUDA
        if (const char *env = std::getenv("TINYCODER_GPU")) {
            if (std::string(env) == "0") return 0;
        }
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) return 0;
        return static_cast<uint64_t>(freeBytes);
#else
        return 0;
#endif
    }

    uint64_t probeTotalVramBytes() {
#ifdef USE_CUDA
        if (const char *env = std::getenv("TINYCODER_GPU")) {
            if (std::string(env) == "0") return 0;
        }
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) return 0;
        return static_cast<uint64_t>(totalBytes);
#else
        return 0;
#endif
    }

    uint64_t probeAvailableRamBytes() {
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open()) {
            std::string key;
            uint64_t value = 0;
            std::string unit;
            while (meminfo >> key >> value >> unit) {
                if (key == "MemAvailable:") {
                    // The kernel reports kB for this field.
                    return value * 1024ull;
                }
                meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
        }
        struct sysinfo si{};
        if (sysinfo(&si) == 0) {
            return static_cast<uint64_t>(si.freeram) * si.mem_unit;
        }
        return 0;
    }

}// namespace tinycoder::detail
