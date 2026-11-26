// Full-device Inkling-2-Mini decode graph (Task 5).
//
// Mirrors ModelInkling.cpp's forwardInkling exactly, step for step:
//   embed + token norm, prelude blocks, three loop passes over
//   [loopStart, loopEnd) with slot = pass==0 ? ip : phys + (pass-1)*core +
//   (ip-loopStart), the cross-pass message bank, postlude blocks, output
//   depth-mix + final norm + LM head + softcap + vocab trim.
//
// One stream, no host synchronisation inside the token loop; only the final
// logits are copied back (plus one copy per token when allLogits is set).
//
// Everything that already has a kernel is reused: inklingGemv (NVFP4/Q2_0/
// IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ4_XS/F32/BF16), int8Gemv (Q4_K/Q5_K/Q6_K/
// Q8_0), kInklingRmsNorm(Rows), kInklingClip, kInklingShortConv,
// kInklingRelBias, kInklingAttention, kInklingMix, kInklingRouter,
// kInklingSitu, kInklingMoeAccum.  The only new device code here is the
// elementwise glue the CPU did inline (residual add, sigmoid gate multiply,
// tau scale, control mix, per-expert scale gather, softcap).
#include "Int8Gemm.hpp"
#include "gpu/InklingGPU.hpp"
#include "gpu/InklingKernels.hpp"

#include "InklingGPUImpl.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// Model.hpp pulls in np/Array.hpp -> <immintrin.h>, and GCC 13's
// amxtileintrin.h emits two AMX builtins that nvcc's cudafe++ front end does
// not know.  They only appear inside unused inline wrappers, so neutralise the
// two names before the header is parsed; that keeps the host-only Model
// interface out of the device pass without touching the third-party np tree.
#if defined(__CUDACC__) || defined(__NVCC__)
#define __builtin_ia32_ldtilecfg(x) ((void) 0)
#define __builtin_ia32_sttilecfg(x) ((void) 0)
#endif
#ifndef __CUDA_ARCH__
// np (pulled in by Model.hpp) carries `#pragma omp` inside its templates.  The
// host pass of a CUDA translation unit has no -fopenmp, so gcc reports every
// one of them as an unknown pragma -- 44 sites, none of them ours and none of
// them reachable from this file.  Suppress that class for the include only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#include "Model.hpp"
#pragma GCC diagnostic pop
#endif

namespace tinycoder::gpu {
    namespace {

        [[maybe_unused]] constexpr float kRmsEps = 1e-6f;

        // ---- elementwise glue (the CPU did these inline) --------------------

        [[maybe_unused]]
        __global__ void combineKernel(float *out, const float *a, const float *b,
                                      uint32_t n, float sa, float sb) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += gridDim.x * blockDim.x)
                out[i] = sa * a[i] + sb * b[i];
        }

        [[maybe_unused]]
        __global__ void sigmoidMulKernel(float *x, const float *g, uint32_t n) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += gridDim.x * blockDim.x)
                x[i] *= 1.0f / (1.0f + expf(-g[i]));
        }

        [[maybe_unused]]
        __global__ void scaleKernel(float *x, uint32_t n, float s) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += gridDim.x * blockDim.x)
                x[i] *= s;
        }

        [[maybe_unused]]
        __global__ void scaleByKernel(float *x, uint32_t n, const float *sp) {
            const float s = *sp;
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += gridDim.x * blockDim.x)
                x[i] *= s;
        }

        // out = cur - tanh(forget)*cur + tanh(gain)*message
        [[maybe_unused]]
        __global__ void controlMixKernel(const float *cur, const float *msg,
                                         const float *gv, const float *fv,
                                         float *out, uint32_t n) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += gridDim.x * blockDim.x)
                out[i] = cur[i] - tanhf(fv[i]) * cur[i] + tanhf(gv[i]) * msg[i];
        }

        [[maybe_unused]]
        __global__ void gatherScalesKernel(float *gateUp, float *down,
                                           const int32_t *idx,
                                           const float *gateUpScale,
                                           const float *downScale, uint32_t n) {
            const uint32_t u = threadIdx.x;
            if (u >= n) return;
            const int32_t e = idx[u];
            gateUp[u] = gateUpScale[e];
            down[u] = downScale[e];
        }

        [[maybe_unused]]
        __global__ void softcapKernel(float *x, uint32_t n, float cap) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
                 i += gridDim.x * blockDim.x)
                x[i] = cap * tanhf(x[i] / cap);
        }

    }// namespace

#ifndef __CUDA_ARCH__
    namespace {
        inline void combine(float *out, const float *a, const float *b, uint32_t n,
                            float sa, float sb, cudaStream_t s) {
            combineKernel<<<(n + 255u) / 256u, 256, 0, s>>>(out, a, b, n, sa, sb);
        }
        inline void sigmoidMul(float *x, const float *g, uint32_t n, cudaStream_t s) {
            sigmoidMulKernel<<<(n + 255u) / 256u, 256, 0, s>>>(x, g, n);
        }
        inline void scaleVec(float *x, uint32_t n, float v, cudaStream_t s) {
            scaleKernel<<<(n + 255u) / 256u, 256, 0, s>>>(x, n, v);
        }
        inline void scaleBy(float *x, uint32_t n, const float *v, cudaStream_t s) {
            scaleByKernel<<<(n + 255u) / 256u, 256, 0, s>>>(x, n, v);
        }
        inline void softcap(float *x, uint32_t n, float cap, cudaStream_t s) {
            softcapKernel<<<(n + 255u) / 256u, 256, 0, s>>>(x, n, cap);
        }

        /// Dispatch one projection.  The engine's int8 GEMV (Q8_1 activation +
        /// dp4a) is tried first because it is materially faster than the float
        /// decoders on every type it covers: Q4_K/Q5_K/Q6_K/Q8_0 and the
        /// i-quants IQ2_XS/IQ2_S/IQ3_S/IQ3_XXS.  It pads the activation to a
        /// 512-wide row, so it handles the partial 256-block tail.  It refuses
        /// anything else (notably Q2_0, IQ2_XXS, IQ4_XS, NVFP4, IQ1_S, F32 and
        /// BF16), and those fall back to the Inkling scalar decoders.
        ///
        /// Numerics: the int8 route quantises the activation to Q8_1, the same
        /// class as the engine's shipped default CPU path, so the GPU logits
        /// are gated against that path (top-1/top-5 exact, per-logit band 0.5,
        /// measured 0.29) rather than against the exact scalar path.  The
        /// exact-path delta (~5.4) is the activation quantisation itself: the
        /// default CPU path's own exact-vs-default RMS is 4.57.
        bool gemvAny(uint32_t type, const void *w, const float *x, float *out,
                     uint32_t rows, uint32_t cols, cudaStream_t s) {
            if (int8Gemv(type, w, x, out, 1, rows, cols, s)) return true;
            return inklingGemvAsync(type, w, x, out, rows, cols, s);
        }

        /// Routed-expert GEMV: the engine's expert-indexed int8 kernel reads the
        /// expert ids on the device, so the routing decision never reaches the
        /// host.  It covers every type this checkpoint's gate/up uses (IQ2_XS,
        /// IQ2_S, IQ2_XXS, IQ1_S).
        ///
        /// Gate/up (xStride == 0, every expert reads the same activation row) is
        /// served by the four-rows-per-warp kernel first: one warp owns four
        /// rows, so the Q8_1 row is read once for four dots and the launch is
        /// 32x smaller.  The batched prefill path keeps calling the one-row
        /// int8GemvExpert, so its arithmetic is unchanged.
        ///
        /// What neither covers is the expert down projection here: its
        /// activation rows are strided by 2*ff while the reduction is ff, so the
        /// shared-row/k-contiguous strides the int8 entries accept (0 or cols)
        /// do not apply, and down stays on the Inkling float decoders, which use
        /// the same device-side index.
        bool gemvExpertAny(uint32_t type, const void *w, const float *x,
                           uint32_t xStride, float *out, uint32_t rowsPer,
                           uint32_t cols, const int32_t *idx, uint32_t nUsed,
                           cudaStream_t s) {
            // IQ1_S/IQ2_XXS int8 moves the official logit band from 0.419 to
            // 0.606 (limit 0.5) for +1.6% decode; keep those two on the float
            // decoder until the band is re-baselined.  The 4-row kernel stays
            // compiled for the re-baseline experiment.
            const bool iqFloat = type == 16u || type == 19u;
            // The 4-row variant is kept compiled but not used: measured
            // 2026-11-25 with the IQ1_S/IQ2_XXS float skip in place (so it
            // covers this checkpoint's 11 IQ2_XS/IQ2_S gate-up layers), it is
            // +0.11 tok/s (44.72 -> 44.83) and moves the official band from
            // 0.41927 to 0.44451.  That breaks the 0.419 threshold for a 0.25%
            // gain, so decode keeps the 1-row kernel.  Re-test if the band is
            // re-baselined.
            if (false && !iqFloat && xStride == 0 &&
                int8GemvExpert4(type, w, x, out, rowsPer, cols, idx, nUsed, s))
                return true;
            if (!iqFloat && int8GemvExpert(type, w, x, out, rowsPer, cols, idx,
                                           nUsed, xStride, s))
                return true;
            return inklingGemvExpertAsync(type, w, x, xStride, out, rowsPer,
                                          cols, idx, nUsed, s);
        }

        bool copyD2D(void *dst, const void *src, uint64_t bytes, cudaStream_t s) {
            return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s) ==
                   cudaSuccess;
        }

        /// Per-stage parity dump, same file names the CPU reference writes when
        /// TINYCODER_INKLING_DUMP points at a directory.  Debug only: this is
        /// the one place the graph may synchronise, and only when asked.
        /// Per-layer hidden norm, the same diagnostic the CPU reference prints
        /// under TINYCODER_INKLING_TRACE.  Debug only: this synchronises the
        /// graph stream, which is why it is gated on an env var.
        void traceNorm(uint32_t tok, uint32_t il, const float *d, uint32_t n,
                       cudaStream_t s) {
            if (std::getenv("TINYCODER_INKLING_TRACE") == nullptr) return;
            std::vector<float> h(n);
            if (cudaMemcpyAsync(h.data(), d, static_cast<size_t>(n) * sizeof(float),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess)
                return;
            if (cudaStreamSynchronize(s) != cudaSuccess) return;
            double ss = 0.0;
            for (uint32_t i = 0; i < n; ++i) ss += double(h[i]) * h[i];
            std::fprintf(stderr, "G%u L%u:%.3g ", tok, il, std::sqrt(ss));
        }

        void dumpDev(const char *dir, const char *tag, const float *d, uint32_t n,
                     uint32_t slot, cudaStream_t s) {
            if (dir == nullptr) return;
            char path[1024];
            // Optional prefix keeps a GPU dump from overwriting the CPU
            // reference's files when both run against one directory.
            const char *pre = std::getenv("TINYCODER_INKLING_DUMP_PREFIX");
            std::snprintf(path, sizeof(path), "%s/%s%s-%u.f32", dir,
                          pre != nullptr ? pre : "", tag, slot);
            std::vector<float> h(n);
            // The graph stream is non-blocking, so a plain cudaMemcpy (legacy
            // stream) would NOT wait for it.  Order the read on the graph
            // stream and sync that stream only.
            if (cudaMemcpyAsync(h.data(), d, static_cast<size_t>(n) * sizeof(float),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess ||
                cudaStreamSynchronize(s) != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            FILE *f = std::fopen(path, "wb");
            if (f != nullptr) {
                std::fwrite(h.data(), sizeof(float), n, f);
                std::fclose(f);
            }
        }


        /// Tagged stage dump for the batched-prefill bisect: writes
        /// "<prefix>-<stage>-t<tok>-s<slot>.f32" when TINYCODER_INKLING_BATCHED_DUMP
        /// names a directory and TINYCODER_INKLING_STAGE_SLOT matches `slot`.
        void dumpStage(const char *dir, const char *prefix, const char *stage,
                       uint32_t tok, uint32_t slot, const float *d, uint32_t n,
                       cudaStream_t s) {
            if (dir == nullptr) return;
            const char *only = std::getenv("TINYCODER_INKLING_STAGE_SLOT");
            if (only != nullptr && static_cast<uint32_t>(std::atoi(only)) != slot)
                return;
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/%s-%s-t%u-s%u.f32", dir, prefix,
                          stage, tok, slot);
            std::vector<float> h(n);
            if (cudaMemcpyAsync(h.data(), d, static_cast<size_t>(n) * sizeof(float),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess ||
                cudaStreamSynchronize(s) != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            FILE *f = std::fopen(path, "wb");
            if (f != nullptr) {
                std::fwrite(h.data(), sizeof(float), n, f);
                std::fclose(f);
            }
        }

        /// Parity-dump hook for the batched-prefill tool: the per-token path
        /// writes `tm-t<t>-prelude.f32` and `tm-t<t>-s<slot>.f32` under the same
        /// TINYCODER_INKLING_BATCHED_DUMP directory the batched path writes
        /// `bt-*` into, so the two trajectories can be compared stage by stage.
        /// Debug only: it syncs the stream.
        void dumpParity(const char *dir, const char *fmt, uint32_t a, uint32_t b,
                        const float *d, uint32_t n, cudaStream_t s) {
            if (dir == nullptr) return;
            char name[128];
            std::snprintf(name, sizeof(name), fmt, a, b);
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
            std::vector<float> h(n);
            if (cudaMemcpyAsync(h.data(), d, static_cast<size_t>(n) * sizeof(float),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess ||
                cudaStreamSynchronize(s) != cudaSuccess) {
                cudaGetLastError();
                return;
            }
            FILE *f = std::fopen(path, "wb");
            if (f != nullptr) {
                std::fwrite(h.data(), sizeof(float), n, f);
                std::fclose(f);
            }
        }

    }// namespace

    namespace {
        enum ProfSec {
            P_EMBED = 0, P_ATTN_MIX, P_ATTN_PROJ, P_ATTN_GLUE, P_ATTN_CORE,
            P_ATTN_OUT, P_FFN_MIX, P_ROUTER, P_EXP_GU, P_EXP_DOWN, P_FFN_POST,
            P_OUT_MIX, P_LMHEAD
        };
        constexpr int kProfSectionCount = 13;
        constexpr const char *kProfNames[kProfSectionCount] = {
                "embed", "attn_mix", "attn_proj", "attn_glue", "attn_core",
                "attn_out", "ffn_mix", "router", "expert_gate_up",
                "expert_down", "ffn_post", "output_mix", "lm_head"};
    }

    void InklingGPU::Impl::profReset() {
        for (int i = 0; i < kProfSectionCount; ++i) {
            profSec[i] = 0.0;
            profCalls[i] = 0;
        }
        profLaunches = 0;
        profGemmLaunches = 0;
        profIdx = 0;
        if (profPool.empty()) {
            profPool.resize(kProfPool);
            profBoundarySection.resize(kProfPool);
            for (auto &e: profPool) cudaEventCreateWithFlags(&e, cudaEventDefault);
        }
    }

    void InklingGPU::Impl::profMark(int section, cudaStream_t s) {
        if (!prof || profIdx >= profPool.size()) return;
        cudaEventRecord(profPool[profIdx], s);
        profBoundarySection[profIdx] = section;
        ++profIdx;
    }

    void InklingGPU::Impl::profPrint(size_t tokens) {
        // The caller has stream-synchronised, so every event has a timestamp.
        for (uint32_t k = 0; k + 1 < profIdx; ++k) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, profPool[k], profPool[k + 1]) !=
                cudaSuccess) {
                cudaGetLastError();
                continue;
            }
            const int sec = profBoundarySection[k];
            if (sec < 0 || sec >= kProfSectionCount) continue;
            profSec[sec] += ms;
            ++profCalls[sec];
        }
        const double n = static_cast<double>(tokens ? tokens : 1);
        double sum = 0.0;
        std::printf("[inkling-prof] per token: launches=%llu (gemv=%llu)\n",
                    static_cast<unsigned long long>(profLaunches / n),
                    static_cast<unsigned long long>(profGemmLaunches / n));
        for (int i = 0; i < kProfSectionCount; ++i) {
            sum += profSec[i] / n;
            std::printf("  %-15s %8.3f ms  calls/token=%llu\n", kProfNames[i],
                        profSec[i] / n,
                        static_cast<unsigned long long>(profCalls[i] / n));
        }
        std::printf("  %-15s %8.3f ms  (sum of sections)\n", "TOTAL", sum);
        std::fflush(stdout);
    }

    // -------------------------------------------------------------------------
    // One-time setup: allocate the recurrent state + scratch arena and upload
    // every physical-layer projection exactly once (the lazy cache is keyed by
    // the host pointer, so a second forward reuses it).
    // -------------------------------------------------------------------------
    bool InklingGPU::ensureGraph(const Model &model, std::string &err) {
        Impl *im = impl_;
        if (im->ready) return true;
        const ModelConfig &cfg = model.config();

        im->hidden = cfg.hiddenSize;
        im->nHeads = cfg.numAttentionHeads;
        im->nKvHeads = cfg.numKVHeads;
        im->headDim = cfg.headDim;
        im->kvDim = im->nKvHeads * im->headDim;
        im->nConv = cfg.shortconvLCache;
        im->histLen = im->nConv > 1 ? im->nConv - 1 : 0;
        im->nRel = cfg.relativeLength;
        im->extent = cfg.relativeExtent;
        im->nExperts = cfg.expertCount;
        im->nUsed = cfg.expertUsedCount;
        im->expertFf = cfg.expertFeedForwardLength;
        im->phys = static_cast<uint32_t>(model.layers_.size());
        im->slots = cfg.numKvSlots ? cfg.numKvSlots : im->phys;
        im->core = cfg.loopEnd > cfg.loopStart ? cfg.loopEnd - cfg.loopStart : 0;
        im->blockSize = cfg.attnResBlockSize;
        im->vocab = cfg.vocabSizeUnpadded ? cfg.vocabSizeUnpadded : cfg.vocabSize;
        im->maxSeq = cfg.maxSeqLen;
        im->sconvChannels = 2 * im->kvDim + 2 * im->hidden;

        auto bad = [&](const std::string &m) {
            err = m;
            return false;
        };
        if (im->phys == 0 || im->slots == 0 || im->core == 0 || im->blockSize == 0 ||
            im->nConv < 2 || im->histLen == 0 || im->nRel == 0 || im->extent == 0 ||
            im->nExperts == 0 || im->nUsed == 0 || im->expertFf == 0 ||
            im->vocab == 0 || im->maxSeq == 0)
            return bad("incomplete Inkling geometry");
        if (im->nHeads > 32 || im->nHeads % im->nKvHeads != 0 ||
            im->nKvHeads > im->nHeads)
            return bad("unsupported attention head geometry");
        if (im->nExperts > 256) return bad("router supports at most 256 experts");
        if (im->nUsed > im->nExperts) return bad("expertUsedCount > expertCount");
        if (cfg.loopStart >= cfg.loopEnd || cfg.loopEnd > im->phys)
            return bad("bad loop range");
        if (im->kvDim == 0 || im->hidden == 0 || im->headDim == 0)
            return bad("bad hidden/head dimensions");
        if (cfg.loopGateRank > 64) return bad("loopGateRank > 64");
        if (!int8GemmAvailable())
            return bad("int8 GEMV unavailable (TINYCODER_INT8_GEMM=0?)");

        // ---- recurrent state --------------------------------------------------
        const uint64_t kvFloats = static_cast<uint64_t>(im->slots) *
                                  im->maxSeq * im->kvDim;
        const uint64_t convFloats = static_cast<uint64_t>(im->slots) *
                                    im->histLen * im->sconvChannels;
        cudaError_t ce = cudaStreamCreateWithFlags(&im->stream,
                                                   cudaStreamNonBlocking);
        if (ce != cudaSuccess) {
            cudaGetLastError();
            return bad(std::string("cudaStreamCreate failed: ") +
                       cudaGetErrorString(ce));
        }
        // The captured token graph calls the engine's int8 GEMV, whose Q8_1
        // scratch is allocated lazily.  cudaMalloc inside a stream capture is
        // not permitted, so reserve the worst-case shape before any capture.
        int8GemvReserve(4608u, 8u);
        if (cudaStreamCreateWithFlags(&im->captureStream,
                                      cudaStreamNonBlocking) != cudaSuccess) {
            cudaGetLastError();
            return bad("cudaStreamCreate(capture) failed");
        }

        auto alloc = [&](void **p, uint64_t bytes, const char *what) {
            const cudaError_t e = cudaMalloc(p, bytes);
            if (e != cudaSuccess) {
                cudaGetLastError();
                size_t mf = 0, mt = 0;
                cudaMemGetInfo(&mf, &mt);
                err = std::string("cudaMalloc(") + what + ", " +
                      std::to_string(bytes) + " bytes) failed: " +
                      cudaGetErrorString(e) + " [free " +
                      std::to_string(mf >> 20) + " MiB of " +
                      std::to_string(mt >> 20) + " MiB]";
                *p = nullptr;
                return false;
            }
            return true;
        };
        if (!alloc(reinterpret_cast<void **>(&im->kCache), kvFloats * sizeof(float),
                   "kv_slots_k"))
            return false;
        if (!alloc(reinterpret_cast<void **>(&im->vCache), kvFloats * sizeof(float),
                   "kv_slots_v"))
            return false;
        if (!alloc(reinterpret_cast<void **>(&im->convState),
                   convFloats * sizeof(float), "shortconv_history"))
            return false;
        cudaMemsetAsync(im->kCache, 0, kvFloats * sizeof(float), im->stream);
        cudaMemsetAsync(im->vCache, 0, kvFloats * sizeof(float), im->stream);
        cudaMemsetAsync(im->convState, 0, convFloats * sizeof(float), im->stream);

        // ---- scratch arena ----------------------------------------------------
        const uint64_t H = im->hidden;
        const uint64_t qDim = static_cast<uint64_t>(im->nHeads) * im->headDim;
        // The block bank holds one hidden row per attention-residual block
        // boundary.  A looped layout visits those boundaries once per pass, so
        // the row count is bounded by the cache slots (48 for Inkling-2-Mini)
        // plus the appended current state, with two rows of slack.  That upper
        // bound, not phys / attnResBlockSize + 1 (which is only 5 for the
        // 24-block loop at blockSize 6), is what the take() calls below
        // advance by; budgeting 8 rows here while taking slots + 2 (= 50) put
        // the logits and the embedded-row staging past the end of the arena.
        const uint64_t bankRows = static_cast<uint64_t>(im->slots) + 2;
        uint64_t need = 0;
        need += 11 * H;                       // x prelude passOut inpSA ffnInp pre0..2 nq nk nv
        need += 5 * H;                        // normed postO convH o moeOut
        need += 3 * H;                        // mixOut (3 queries)
        need += 4 * H;                        // msgMixed (4 queries)
        need += 2 * bankRows * H;             // blockBank + bankSnap
        need += 2 * H;                        // ctrlIn
        need += 2 * H;                        // ctrlGv + ctrlFv
        need += 3 * qDim;                     // q gateProj attnOut
        need += 4 * static_cast<uint64_t>(im->kvDim);// k v kProj vProj
        need += static_cast<uint64_t>(im->nHeads) * im->nRel;
        need += im->nHeads;                   // sink
        need += static_cast<uint64_t>(im->nHeads) * (im->extent + 1);// table
        need += im->nExperts;                 // router
        need += 3 * static_cast<uint64_t>(im->nUsed);// wgt + gateUpRank + downRank
        need += static_cast<uint64_t>(im->nUsed) * 2 * im->expertFf;// gateUp
        need += static_cast<uint64_t>(im->nUsed) * H;// expertOut
        need += static_cast<uint64_t>(im->nConv) * im->nKvHeads;// dynamic taps
        need += 64;                           // ctrlG
        need += im->vocab;                    // logits
        need += im->nUsed;                    // router index buffer (int32)
        need += 1;                            // device token position (int32)
        need += static_cast<uint64_t>(im->maxSeq) * H;// embedded rows
        im->arenaBytes = need * sizeof(float);
        if (!alloc(reinterpret_cast<void **>(&im->arena), im->arenaBytes,
                   "forward_scratch_arena"))
            return false;
        float *cur = im->arena;
        auto take = [&](uint64_t n) {
            float *p = cur;
            cur += n;
            return p;
        };
        im->dX = take(H);
        im->dPrelude = take(H);
        im->dPassOut = take(H);
        im->dInpSA = take(H);
        im->dFfnInp = take(H);
        im->dPre0 = take(H);
        im->dPre1 = take(H);
        im->dPre2 = take(H);
        im->dNq = take(H);
        im->dNk = take(H);
        im->dNv = take(H);
        im->dNormed = take(H);
        im->dPostO = take(H);
        im->dConvH = take(H);
        im->dO = take(H);
        im->dMoeOut = take(H);
        im->dMixOut = take(3 * H);
        im->dMsgMixed = take(4 * H);
        im->dBlockBank = take(bankRows * H);
        im->dBankSnap = take(bankRows * H);
        im->dCtrlIn = take(2 * H);
        im->dCtrlGv = take(H);
        im->dCtrlFv = take(H);
        im->dQ = take(qDim);
        im->dGateProj = take(qDim);
        im->dAttnOut = take(qDim);
        im->dK = take(im->kvDim);
        im->dV = take(im->kvDim);
        im->dKProj = take(im->kvDim);
        im->dVProj = take(im->kvDim);
        im->dRel = take(static_cast<uint64_t>(im->nHeads) * im->nRel);
        im->dSink = take(im->nHeads);
        im->dTable = take(static_cast<uint64_t>(im->nHeads) * (im->extent + 1));
        im->dRouter = take(im->nExperts);
        im->dWgt = take(im->nUsed);
        im->dGateUpRank = take(im->nUsed);
        im->dDownRank = take(im->nUsed);
        im->dGateUp = take(static_cast<uint64_t>(im->nUsed) * 2 * im->expertFf);
        im->dExpertOut = take(static_cast<uint64_t>(im->nUsed) * H);
        im->dDyn = take(static_cast<uint64_t>(im->nConv) * im->nKvHeads);
        // The loop-controller scratch (2*4 invs + 4*rank g values) lives at the
        // head of dCtrlIn; dCtrlG aliases the g block so the parity dump and the
        // batched reference's comment keep naming it.
        im->dCtrlG = im->dCtrlIn + 2u * kInklingMaxControls;
        im->dLogits = take(im->vocab);
        im->embAll = take(static_cast<uint64_t>(im->maxSeq) * H);
        im->dIdx = reinterpret_cast<int32_t *>(take(im->nUsed));
        im->dPos = reinterpret_cast<int32_t *>(take(1));
        // The take() calls above must never exceed the budget in `need`: the
        // 64 floats reserved for ctrlG stay unused because dCtrlG aliases the
        // g block inside dCtrlIn.  An overrun silently places the tail of the
        // arena outside its allocation, which is what the 8-row bank budget
        // did (by 193,472 floats on Inkling-2-Mini).
        const uint64_t taken = static_cast<uint64_t>(cur - im->arena);
        if (taken > need) {
            return bad("scratch arena overrun: " + std::to_string(taken) +
                       " floats taken, " + std::to_string(need) + " budgeted");
        }

        // Pinned staging for the per-token embedding row the graph reads.
        im->pinnedEmb.resize(4);
        for (auto &pe: im->pinnedEmb) {
            if (cudaHostAlloc(reinterpret_cast<void **>(&pe), H * sizeof(float),
                              cudaHostAllocDefault) != cudaSuccess) {
                cudaGetLastError();
                return bad("cudaHostAlloc(embedding staging) failed");
            }
        }

        // ---- model-level weights ---------------------------------------------
        auto raw = [&](const np::Array<float> &a, uint64_t want,
                       const char *what) -> const void * {
            if (a.size() < want) {
                err = std::string("bad size for ") + what + ": " +
                      std::to_string(a.size()) + " < " + std::to_string(want);
                return nullptr;
            }
            return im->getRaw(a.data(), want * sizeof(float), err, what);
        };
        im->tokNorm = raw(model.inklingTokNorm_, H, "token_embd_norm");
        if (!im->tokNorm) return false;
        im->finalNorm = raw(model.finalNorm_, H, "output_norm");
        if (!im->finalNorm) return false;
        im->outResQuery = raw(model.inklingOutputResQuery_, H, "output_res_query");
        if (!im->outResQuery) return false;

        // Cross-pass message queries, packed as [core][4][hidden] in the same
        // (res, kv, kv, res) order the CPU builds in forwardInkling.
        if (cfg.inklingExpandedLayout) {
            const uint64_t per = static_cast<uint64_t>(2) * im->core * H;
            if (model.inklingLoopResQuery_.size() < per ||
                model.inklingLoopKvResQuery_.size() < per)
                return bad("missing loop_res_query / loop_kv_res_query");
            im->hostLegacyMsg.resize(static_cast<size_t>(im->core) * 4 * H);
            for (uint32_t i = 0; i < im->core; ++i) {
                float *dst = im->hostLegacyMsg.data() +
                             static_cast<size_t>(i) * 4 * H;
                const float *res = model.inklingLoopResQuery_.data();
                const float *kv = model.inklingLoopKvResQuery_.data();
                std::memcpy(dst + 0 * H, res + static_cast<size_t>(2 * i) * H,
                            H * sizeof(float));
                std::memcpy(dst + 1 * H, kv + static_cast<size_t>(2 * i) * H,
                            H * sizeof(float));
                std::memcpy(dst + 2 * H, kv + static_cast<size_t>(2 * i + 1) * H,
                            H * sizeof(float));
                std::memcpy(dst + 3 * H, res + static_cast<size_t>(2 * i + 1) * H,
                            H * sizeof(float));
            }
            im->legacyMsg = im->getRaw(im->hostLegacyMsg.data(),
                                       im->hostLegacyMsg.size() * sizeof(float),
                                       err, "loop_msg_query");
            if (!im->legacyMsg) return false;
        }

        // ---- LM head ----------------------------------------------------------
        if (model.lmHead_.empty())
            return bad("Inkling GPU forward needs a separate (non-tied) LM head");
        if (model.lmHead_.rows < im->vocab || model.lmHead_.cols != im->hidden)
            return bad("LM head shape does not match the model config");
        im->lmHeadType = model.lmHead_.type;
        im->lmHead = im->getMat(model.lmHead_.data.data(), im->lmHeadType,
                                im->vocab, im->hidden, err, "output.weight");
        if (!im->lmHead) return false;

        // ---- per-layer weights ------------------------------------------------
        auto str = [](const char *p, uint32_t i) {
            return std::string("blk.") + std::to_string(i) + "." + p;
        };
        auto mat = [&](const QuantizedMatrix &m, uint32_t il,
                       const char *name) -> const void * {
            if (m.empty() || m.data.empty()) {
                err = "missing tensor " + str(name, il);
                return nullptr;
            }
            return im->getMat(m.data.data(), m.type, m.rows, m.cols, err,
                              str(name, il).c_str());
        };
        auto optMat = [&](const QuantizedMatrix &m, uint32_t il,
                          const char *name) -> const void * {
            if (m.empty() || m.data.empty()) return nullptr;
            return im->getMat(m.data.data(), m.type, m.rows, m.cols, err,
                              str(name, il).c_str());
        };

        // Fallbacks for the optional per-expert scale / bias tensors (layers
        // 20-23 carry none; the CPU reference substitutes 1.0 / 0.0).
        im->onesHost.assign(im->nExperts, 1.0f);
        im->zerosHost.assign(im->nExperts, 0.0f);
        im->onesDev = im->getRaw(im->onesHost.data(),
                                 im->onesHost.size() * sizeof(float), err,
                                 "expert_scale_ones");
        im->zerosDev = im->getRaw(im->zerosHost.data(),
                                  im->zerosHost.size() * sizeof(float), err,
                                  "expert_bias_zeros");
        if (!im->onesDev || !im->zerosDev) return false;

        im->layers.assign(im->phys, Impl::LayerDev{});
        for (uint32_t il = 0; il < im->phys; ++il) {
            const auto &w = model.layers_[il];
            const auto &ink = w.inkling;
            auto &dev = im->layers[il];
            const bool isSwa = il < cfg.slidingWindowPattern.size()
                                       ? cfg.slidingWindowPattern[il] != 0
                                       : true;

            dev.attnQ = mat(w.attnQ, il, "attn_q.weight");
            dev.attnK = mat(w.attnK, il, "attn_k.weight");
            dev.attnV = mat(w.attnV, il, "attn_v.weight");
            dev.attnO = mat(w.attnO, il, "attn_output.weight");
            dev.attnGate = mat(ink.attnGate, il, "attn_gate.weight");
            dev.attnRel = mat(ink.attnRel, il, "attn_rel.weight");
            dev.sinkProj = mat(ink.attnSinkProj, il, "attn_sink_proj.weight");
            dev.router = mat(w.ffnGateInp, il, "ffn_gate_inp.weight");
            dev.gateUp = mat(w.ffnGateUpExps, il, "ffn_gate_up_exps.weight");
            dev.down = mat(w.ffnDownExps, il, "ffn_down_exps.weight");
            if (!dev.attnQ || !dev.attnK || !dev.attnV || !dev.attnO ||
                !dev.attnGate || !dev.attnRel || !dev.sinkProj || !dev.router ||
                !dev.gateUp || !dev.down)
                return false;

            // attn_rel_bias is a QuantizedMatrix holding raw F32 (nRel x extent).
            if (ink.attnRelBias.type != GGML_TYPE_F32 ||
                ink.attnRelBias.data.size() <
                        static_cast<uint64_t>(im->nRel) * im->extent * sizeof(float)) {
                err = "bad attn_rel_bias for blk." + std::to_string(il);
                return false;
            }
            dev.relBias = im->getRaw(ink.attnRelBias.data.data(),
                                     static_cast<uint64_t>(im->nRel) * im->extent *
                                             sizeof(float),
                                     err, str("attn_rel_bias.weight", il).c_str());
            if (!dev.relBias) return false;

            auto rawQm = [&](const QuantizedMatrix &m, uint32_t il,
                             const char *name, uint64_t wantBytes) -> const void * {
                if (m.data.size() < wantBytes) {
                    err = "bad size for " + str(name, il) + ": " +
                          std::to_string(m.data.size()) + " < " +
                          std::to_string(wantBytes);
                    return nullptr;
                }
                return im->getRaw(m.data.data(), wantBytes, err,
                                  str(name, il).c_str());
            };
            dev.kConvW = rawQm(ink.attnKConv, il, "attn_k_conv.weight",
                               static_cast<uint64_t>(im->nConv) * im->kvDim * 4);
            dev.vConvW = rawQm(ink.attnVConv, il, "attn_v_conv.weight",
                               static_cast<uint64_t>(im->nConv) * im->kvDim * 4);
            dev.attnPostConvW = rawQm(ink.attnPostConv, il,
                                      "attn_post_conv.weight",
                                      static_cast<uint64_t>(im->nConv) * H * 4);
            dev.ffnPostConvW = rawQm(ink.ffnPostConv, il, "ffn_post_conv.weight",
                                     static_cast<uint64_t>(im->nConv) * H * 4);
            if (!dev.kConvW || !dev.vConvW || !dev.attnPostConvW ||
                !dev.ffnPostConvW)
                return false;

            dev.attnQClip = raw(ink.attnQClip, static_cast<uint64_t>(im->nHeads) * im->headDim,
                                str("attn_q_clip.weight", il).c_str());
            dev.attnRelClip = raw(ink.attnRelClip,
                                  static_cast<uint64_t>(im->nHeads) * im->nRel,
                                  str("attn_rel_clip.weight", il).c_str());
            dev.rmsNormFFN = raw(w.rmsNormFFN, H, str("ffn_norm.weight", il).c_str());
            dev.probsBias = ink.expertProbsBias.size() >= im->nExperts
                                    ? raw(ink.expertProbsBias, im->nExperts,
                                          str("exp_probs_b.bias", il).c_str())
                                    : im->zerosDev;
            dev.gateUpScale = ink.expertGateUpScale.size() >= im->nExperts
                                      ? raw(ink.expertGateUpScale, im->nExperts,
                                            str("ffn_gate_up_exps.scale", il).c_str())
                                      : im->onesDev;
            dev.downScale = ink.expertDownScale.size() >= im->nExperts
                                    ? raw(ink.expertDownScale, im->nExperts,
                                          str("ffn_down_exps.scale", il).c_str())
                                    : im->onesDev;
            if (!dev.attnQClip || !dev.attnRelClip || !dev.rmsNormFFN ||
                !dev.probsBias || !dev.gateUpScale || !dev.downScale)
                return false;

            if (isSwa) {
                dev.rmsNormAttn = raw(w.rmsNormAttn, H,
                                      str("attn_norm.weight", il).c_str());
                if (!dev.rmsNormAttn) return false;
            } else {
                dev.qInpNorm = raw(ink.attnQInpNorm, H,
                                   str("attn_q_inp_norm.weight", il).c_str());
                dev.kInpNorm = raw(ink.attnKInpNorm, H,
                                   str("attn_k_inp_norm.weight", il).c_str());
                dev.vInpNorm = raw(ink.attnVInpNorm, H,
                                   str("attn_v_inp_norm.weight", il).c_str());
                dev.kConvProj = optMat(ink.attnKConvProj, il,
                                       "attn_k_conv_proj.weight");
                dev.vConvProj = optMat(ink.attnVConvProj, il,
                                       "attn_v_conv_proj.weight");
                if (!dev.qInpNorm || !dev.kInpNorm || !dev.vInpNorm ||
                    !dev.kConvProj || !dev.vConvProj) {
                    if (err.empty()) err = "missing full-attention tensors for blk." +
                                           std::to_string(il);
                    return false;
                }
            }

            if (!ink.attnResQuery.empty()) {
                const uint32_t nq = isSwa ? 1 : 3;
                dev.attnResQuery = raw(ink.attnResQuery,
                                       static_cast<uint64_t>(nq) * H,
                                       str("attn_res_query.weight", il).c_str());
                if (!dev.attnResQuery) return false;
            }
            dev.ffnResQuery = raw(ink.ffnResQuery, H,
                                  str("ffn_res_query.weight", il).c_str());
            if (!dev.ffnResQuery) return false;

            if (il >= cfg.loopStart && il < cfg.loopEnd) {
                dev.msgQuery = optMat(ink.msgQuery, il, "loop_msg_query.weight");
                if (dev.msgQuery == nullptr && !cfg.inklingExpandedLayout) {
                    if (err.empty()) err = "missing loop_msg_query for blk." +
                                           std::to_string(il);
                    return false;
                }
                uint32_t rank = 0;
                for (int c = 0; c < 4; ++c) {
                    const uint32_t r = ink.ctrlDown[c].rows;
                    if (c == 0) rank = r;
                    if (r != rank || ink.ctrlGain[c].rows != im->hidden ||
                        ink.ctrlForget[c].rows != im->hidden ||
                        ink.ctrlDown[c].cols != 2 * H ||
                        ink.ctrlGain[c].cols != rank ||
                        ink.ctrlForget[c].cols != rank) {
                        err = "bad loop controller shape for blk." +
                              std::to_string(il) + " ctrl " + std::to_string(c);
                        return false;
                    }
                    dev.ctrlDown[c] = mat(ink.ctrlDown[c], il,
                                          ("loop_ctrl_down." + std::to_string(c)).c_str());
                    dev.ctrlGain[c] = mat(ink.ctrlGain[c], il,
                                          ("loop_ctrl_gain." + std::to_string(c)).c_str());
                    dev.ctrlForget[c] = mat(ink.ctrlForget[c], il,
                                            ("loop_ctrl_forget." + std::to_string(c)).c_str());
                    if (!dev.ctrlDown[c] || !dev.ctrlGain[c] || !dev.ctrlForget[c])
                        return false;
                }
                if (rank == 0 || rank > 64)
                    return bad("bad loop controller rank for blk." +
                               std::to_string(il));
                dev.ctrlRank = rank;
            }
        }
        im->ready = true;
        return true;
    }

    // -------------------------------------------------------------------------
    // The decode graph.
    // -------------------------------------------------------------------------
    bool InklingGPU::forward(const Model &model, const std::vector<int32_t> &tokens,
                             bool allLogits, float *logits, std::string &err) {
        err.clear();
        if (!available_ || impl_ == nullptr) {
            err = "CUDA device unavailable";
            return false;
        }
        if (tokens.empty()) {
            err = "empty token list";
            return false;
        }
        if (logits == nullptr) {
            err = "null logits buffer";
            return false;
        }
        if (!ensureGraph(model, err)) return false;
        Impl *im = impl_;
        const ModelConfig &cfg = model.config();
        const uint32_t H = im->hidden;
        const uint32_t qDim = im->nHeads * im->headDim;
        const float qkScale = 1.0f / std::sqrt(float(im->headDim));
        const float postScale = 1.0f / std::sqrt(float(im->nConv + 1));
        cudaStream_t s = im->stream;
        const char *dumpDir = std::getenv("TINYCODER_INKLING_GPU_DUMP");
        if (dumpDir == nullptr) dumpDir = std::getenv("TINYCODER_INKLING_DUMP");
        // Parity dumps for the batched-prefill tool: this path is the reference
        // trajectory, so it writes tm-* with the same layout the batched graph
        // writes bt-* in.  See tools/inkling_gpu_batched_prefill_test.cpp.
        const char *parityDir = std::getenv("TINYCODER_INKLING_BATCHED_DUMP");
        if (!im->prof) {
            const char *pe = std::getenv("TINYCODER_INKLING_GPU_PROF");
            if (pe != nullptr && pe[0] != '\0' && pe[0] != '0') {
                im->prof = true;
            }
        }
        if (im->prof) im->profReset();
        auto profMark = [&](int section) { im->profMark(section, s); };
        if (im->kvPos < 0 ||
            static_cast<uint64_t>(im->kvPos) + tokens.size() > im->maxSeq) {
            err = "context window exceeded: pos " + std::to_string(im->kvPos) +
                  " + " + std::to_string(tokens.size()) + " > " +
                  std::to_string(im->maxSeq);
            return false;
        }

        auto K = [&](uint64_t n) { if (im->prof) im->profLaunches += n; };
        auto CK = [&]() {
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                err = std::string("CUDA error: ") + cudaGetErrorString(e);
                return false;
            }
            return true;
        };

        // Enqueue one token's whole graph.  `devicePos` selects the graph form
        // (pos read from dPos, so the captured graph replays for every token)
        // over the direct form (pos as a host constant).
        auto enqueueToken = [&](size_t si, bool devicePos,
                                bool copyLogits) -> bool {
            const uint32_t pos =
                    devicePos ? 0u
                              : static_cast<uint32_t>(im->kvPos) +
                                        static_cast<uint32_t>(si);
            const uint32_t embOff =
                    devicePos ? 0u : static_cast<uint32_t>(si) * H;
            profMark(P_EMBED);
            kInklingRmsNorm(im->embAll + embOff,
                            static_cast<const float *>(im->tokNorm), im->dX, H,
                            kRmsEps, s);
            if (dumpDir != nullptr) {
                char tag[64];
                std::snprintf(tag, sizeof(tag), "emb-%u",
                              static_cast<unsigned>(tokens[si]));
                dumpDev(dumpDir, tag, im->dX, H, 0, s);
            }
            K(1);// token norm
            uint32_t nBlocks = 0;

            auto GE = [&](uint32_t type, const void *dw, const float *x, float *out,
                          uint32_t rows, uint32_t cols) {
                if (im->prof) ++im->profGemmLaunches;
                if (!gemvAny(type, dw, x, out, rows, cols, s)) {
                    err = "GEMV launch failed (type " + std::to_string(type) +
                          ", rows " + std::to_string(rows) + ", cols " +
                          std::to_string(cols) + ")";
                    return false;
                }
                return true;
            };

            // Block-bank message mix: states = blocks[0..nBlocks) + current x.
            auto mixCurrent = [&](float *out, const void *dQuery,
                                  uint32_t nq) -> bool {
                kInklingMixCurrent(im->dBlockBank, nBlocks, im->dX,
                                   static_cast<const float *>(dQuery), nq, H,
                                   out, s);
                return true;
            };


            auto runLayer = [&](uint32_t physIdx, uint32_t slot,
                                const float *dMessage) -> bool {
                const auto &w = model.layers_[physIdx];
                const auto &ink = w.inkling;
                const auto &dev = im->layers[physIdx];
                const bool isSwa = physIdx < cfg.slidingWindowPattern.size()
                                           ? cfg.slidingWindowPattern[physIdx] != 0
                                           : true;
                profMark(P_ATTN_MIX);
                float *hist = im->convState +
                              static_cast<size_t>(slot) * im->histLen *
                                      im->sconvChannels;

                // inpSA = x (before attention)
                if (!copyD2D(im->dInpSA, im->dX, H * sizeof(float), s)) return false;

                // ---- attention inputs: depth mix, then optional message control
                float *pre0, *pre1, *pre2;
                if (dev.attnResQuery != nullptr) {
                    const uint32_t nq = isSwa ? 1 : 3;
                    if (!mixCurrent(im->dMixOut, dev.attnResQuery, nq)) return false;
                    pre0 = im->dMixOut;
                    pre1 = isSwa ? im->dMixOut : im->dMixOut + H;
                    pre2 = isSwa ? im->dMixOut : im->dMixOut + 2 * H;
                } else {
                    pre0 = pre1 = pre2 = im->dX;
                }
                dumpStage(parityDir, "tm", "mix", static_cast<uint32_t>(si), slot,
                          pre0, H, s);
                if (dMessage != nullptr) {
                    // All three attention controllers of the layer in one
                    // launch per stage: the rank-wide down (one block per
                    // controller and rank, each with the reference kernel's own
                    // two RMS norms) and the mix (one thread per hidden
                    // element, weights read four at a time).  Bit-exact with
                    // the reference controlDown/controlMix pair.
                    float *outs[3] = {im->dPre0, im->dPre1, im->dPre2};
                    const float *curs[3] = {pre0, pre1, pre2};
                    InklingControlArgs ctl{};
                    ctl.nCtl = 3;
                    ctl.rank = dev.ctrlRank;
                    ctl.hidden = H;
                    for (uint32_t c = 0; c < 3; ++c) {
                        ctl.cur[c] = curs[c];
                        ctl.msg[c] = dMessage + c * H;
                        ctl.downW[c] = static_cast<const float *>(dev.ctrlDown[c]);
                        ctl.gainW[c] = static_cast<const float *>(dev.ctrlGain[c]);
                        ctl.forgetW[c] =
                                static_cast<const float *>(dev.ctrlForget[c]);
                        ctl.out[c] = outs[c];
                    }
                    ctl.g = im->dCtrlIn + 2u * kInklingMaxControls;
                    kInklingControlDownFused(ctl, s);
                    kInklingControlMixFused(ctl, s);
                    pre0 = im->dPre0;
                    pre1 = im->dPre1;
                    pre2 = im->dPre2;
                }
                dumpStage(parityDir, "tm", "ctg", static_cast<uint32_t>(si), slot,
                          im->dCtrlIn + 2u * kInklingMaxControls, dev.ctrlRank, s);
                dumpStage(parityDir, "tm", "msg", static_cast<uint32_t>(si), slot,
                          dMessage != nullptr ? dMessage : im->dX, H, s);
                if (dMessage != nullptr)
                    dumpStage(parityDir, "tm", "msgall",
                              static_cast<uint32_t>(si), slot, dMessage, 4 * H,
                              s);

                dumpStage(parityDir, "tm", "pre", static_cast<uint32_t>(si), slot,
                          pre0, H, s);
                // A sliding-window layer normalises one shared row with one
                // shared weight: pre0 == pre1 == pre2 (the depth mix writes a
                // single vector) and all three use rmsNormAttn, so the three
                // launches are the same computation.  Keep Q's result and let K
                // and V read it -- identical values by construction, two fewer
                // launches.  Layer-message controllers write three distinct
                // rows, so the check is `pre1 == pre0 && pre2 == pre0`.
                const float *nQ = im->dNq;
                const float *nK = im->dNk;
                const float *nV = im->dNv;
                if (isSwa) {
                    kInklingRmsNorm(pre0, static_cast<const float *>(dev.rmsNormAttn),
                                    im->dNq, H, kRmsEps, s);
                    if (pre1 == pre0 && pre2 == pre0) {
                        nK = im->dNq;
                        nV = im->dNq;
                    } else {
                        kInklingRmsNorm(pre1,
                                        static_cast<const float *>(dev.rmsNormAttn),
                                        im->dNk, H, kRmsEps, s);
                        kInklingRmsNorm(pre2,
                                        static_cast<const float *>(dev.rmsNormAttn),
                                        im->dNv, H, kRmsEps, s);
                    }
                } else {
                    kInklingRmsNorm(pre0, static_cast<const float *>(dev.qInpNorm),
                                    im->dNq, H, kRmsEps, s);
                    kInklingRmsNorm(pre1, static_cast<const float *>(dev.kInpNorm),
                                    im->dNk, H, kRmsEps, s);
                    kInklingRmsNorm(pre2, static_cast<const float *>(dev.vInpNorm),
                                    im->dNv, H, kRmsEps, s);
                }

                dumpDev(dumpDir, "nq", im->dNq, H, slot, s);

                // attn-input norms (one when Q/K/V share the row and weight)
                // + the two controller stages.
                K((isSwa && pre1 == pre0 && pre2 == pre0 ? 1u : 3u) +
                  (dMessage != nullptr ? 2u : 0u));
                profMark(P_ATTN_PROJ);

                // ---- Q/K/V/rel/sink projections ------------------------------
                if (!GE(w.attnQ.type, dev.attnQ, nQ, im->dQ, qDim, H))
                    return false;
                if (!GE(w.attnK.type, dev.attnK, nK, im->dKProj, im->kvDim, H))
                    return false;
                if (!GE(w.attnV.type, dev.attnV, nV, im->dVProj, im->kvDim, H))
                    return false;
                if (!GE(ink.attnRel.type, dev.attnRel, nQ, im->dRel,
                        im->nHeads * im->nRel, H))
                    return false;

                profMark(P_ATTN_GLUE);

                // ---- short convolutions (K at 0, V at kvDim in the history) ---
                const float convScale =
                        isSwa ? 1.0f / std::sqrt(float(im->nConv + 1))
                              : 1.0f / std::sqrt(float(2 * im->nConv + 1));
                const float *dyn = nullptr;
                if (!isSwa && dev.kConvProj != nullptr) {
                    if (!GE(ink.attnKConvProj.type, dev.kConvProj, nK, im->dDyn,
                            im->nConv * im->nKvHeads, H))
                        return false;
                    dyn = im->dDyn;
                }
                kInklingShortConv(im->dKProj,
                                  static_cast<const float *>(dev.kConvW), dyn, hist,
                                  im->dK, im->kvDim, im->nConv, im->nKvHeads,
                                  true, convScale, s);
                dyn = nullptr;
                if (!isSwa && dev.vConvProj != nullptr) {
                    if (!GE(ink.attnVConvProj.type, dev.vConvProj, nV, im->dDyn,
                            im->nConv * im->nKvHeads, H))
                        return false;
                    dyn = im->dDyn;
                }
                kInklingShortConv(im->dVProj,
                                  static_cast<const float *>(dev.vConvW), dyn,
                                  hist + static_cast<size_t>(im->histLen) * im->kvDim,
                                  im->dV, im->kvDim, im->nConv, im->nKvHeads,
                                  true, convScale, s);

                // ---- clip / per-head value norm / rel clip / sink ------------
                kInklingClip(im->dQ, im->nHeads, im->headDim,
                             static_cast<const float *>(dev.attnQClip), cfg.qkClip, s);
                kInklingClip(im->dK, im->nKvHeads, im->headDim, nullptr,
                             cfg.qkClip, s);
                kInklingRmsNormRows(im->dV, nullptr, im->dV, im->headDim,
                                    im->nKvHeads, kRmsEps, s);
                kInklingClip(im->dRel, im->nHeads, im->nRel,
                             static_cast<const float *>(dev.attnRelClip),
                             cfg.relativeClip, s);
                if (!GE(ink.attnSinkProj.type, dev.sinkProj, nQ, im->dSink,
                        im->nHeads, H))
                    return false;

                // The CPU reference dumps Qcur/Kcur/Vcur/Rcur before the
                // log-scaled-position tau is applied.
                dumpDev(dumpDir, "Qcur", im->dQ, qDim, slot, s);
                dumpDev(dumpDir, "Kcur", im->dK, im->kvDim, slot, s);
                dumpDev(dumpDir, "Vcur", im->dV, im->kvDim, slot, s);
                dumpDev(dumpDir, "Rcur", im->dRel, im->nHeads * im->nRel, slot, s);

                // ---- log-scaled positions (full attention only) --------------
                if (!isSwa) {
                    if (devicePos) {
                        kInklingTauScale(im->dQ, qDim, im->dRel,
                                         im->nHeads * im->nRel, im->dPos,
                                         cfg.logScaleAlpha, cfg.logScaleFloor, s);
                    } else {
                        const float arg =
                                std::max(1.0f, float(pos + 1) /
                                                       std::max(1.0f, cfg.logScaleFloor));
                        const float tau = 1.0f + cfg.logScaleAlpha * std::log(arg);
                        scaleVec(im->dQ, qDim, tau, s);
                        scaleVec(im->dRel, im->nHeads * im->nRel, tau, s);
                    }
                }

                K(7);// 2 shortconv + 3 clip + 1 rmsnormRows + 1 relbias
                profMark(P_ATTN_CORE);

                // ---- relative bias table, K/V store, attention ---------------
                kInklingRelBias(im->dRel, static_cast<const float *>(dev.relBias),
                                im->dSink, im->dTable, im->nHeads, im->nRel,
                                im->extent, s);
                float *kSlot = im->kCache +
                               static_cast<size_t>(slot) * im->maxSeq * im->kvDim;
                float *vSlot = im->vCache +
                               static_cast<size_t>(slot) * im->maxSeq * im->kvDim;
                if (devicePos) {
                    kInklingAttentionStorePos(im->dQ, im->dK, im->dV, kSlot,
                                              vSlot, im->dTable, im->dPos, isSwa,
                                              cfg.slidingWindow, im->nHeads,
                                              im->nKvHeads, im->extent, qkScale,
                                              im->dAttnOut, s);
                } else {
                    const uint32_t firstKey =
                            isSwa && cfg.slidingWindow > 0 &&
                                            pos + 1 > cfg.slidingWindow
                                    ? pos + 1 - cfg.slidingWindow
                                    : 0;
                    const uint32_t nKeys = pos + 1 - firstKey;
                    kInklingAttentionStore(im->dQ, im->dK, im->dV, kSlot, vSlot,
                                           im->dTable, pos, firstKey, nKeys,
                                           im->nHeads, im->nKvHeads, im->extent,
                                           qkScale, im->dAttnOut, s);
                }

                dumpDev(dumpDir, "kqv", im->dAttnOut, qDim, slot, s);
                dumpStage(parityDir, "tm", "ao", static_cast<uint32_t>(si), slot,
                          im->dAttnOut, qDim, s);

                K(1);// attention
                profMark(P_ATTN_OUT);

                // ---- attention gate + output projection + post conv ----------
                if (!GE(ink.attnGate.type, dev.attnGate, im->dNq, im->dGateProj,
                        qDim, H))
                    return false;
                dumpDev(dumpDir, "gateProj", im->dGateProj, qDim, slot, s);
                sigmoidMul(im->dAttnOut, im->dGateProj, qDim, s);
                if (!GE(w.attnO.type, dev.attnO, im->dAttnOut, im->dO, H, qDim))
                    return false;
                dumpDev(dumpDir, "wo_out", im->dO, H, slot, s);
                kInklingShortConv(im->dO,
                                  static_cast<const float *>(dev.attnPostConvW),
                                  nullptr,
                                  hist + static_cast<size_t>(im->histLen) *
                                                 (2 * im->kvDim),
                                  im->dPostO, H, im->nConv, 1, true, postScale, s);
                dumpDev(dumpDir, "attn_out", im->dPostO, H, slot, s);

                // ---- block bank / residual -----------------------------------
                if (physIdx % im->blockSize == 0) {
                    if (!copyD2D(im->dBlockBank +
                                         static_cast<size_t>(nBlocks) * H,
                                 im->dInpSA, H * sizeof(float), s))
                        return false;
                    ++nBlocks;
                    if (!copyD2D(im->dX, im->dPostO, H * sizeof(float), s))
                        return false;
                } else {
                    combine(im->dX, im->dPostO, im->dInpSA, H, 1.0f, 1.0f, s);
                }

                K(3);// sigmoid gate, attn post-conv, residual
                profMark(P_FFN_MIX);

                // ---- FFN: depth mix, optional message, norm, routed MoE ------
                if (!copyD2D(im->dFfnInp, im->dX, H * sizeof(float), s)) return false;
                if (!mixCurrent(im->dMixOut, dev.ffnResQuery, 1)) return false;
                if (dMessage != nullptr) {
                    InklingControlArgs ctl{};
                    ctl.nCtl = 1;
                    ctl.rank = dev.ctrlRank;
                    ctl.hidden = H;
                    ctl.cur[0] = im->dMixOut;
                    ctl.msg[0] = dMessage + 3 * H;
                    ctl.downW[0] = static_cast<const float *>(dev.ctrlDown[3]);
                    ctl.gainW[0] = static_cast<const float *>(dev.ctrlGain[3]);
                    ctl.forgetW[0] = static_cast<const float *>(dev.ctrlForget[3]);
                    ctl.out[0] = im->dNormed;
                    ctl.g = im->dCtrlIn + 2u * kInklingMaxControls;
                    kInklingControlDownFused(ctl, s);
                    kInklingControlMixFused(ctl, s);
                    kInklingRmsNorm(im->dNormed,
                                    static_cast<const float *>(dev.rmsNormFFN),
                                    im->dNormed, H, kRmsEps, s);
                } else {
                    kInklingRmsNorm(im->dMixOut,
                                    static_cast<const float *>(dev.rmsNormFFN),
                                    im->dNormed, H, kRmsEps, s);
                }

                K(dMessage != nullptr ? 4u : 1u);// mix + norm (+ control down/mix)
                profMark(P_ROUTER);

                dumpStage(parityDir, "tm", "nrm", static_cast<uint32_t>(si), slot,
                          im->dNormed, H, s);
                // ---- router ---------------------------------------------------
                if (!GE(w.ffnGateInp.type, dev.router, im->dNormed, im->dRouter,
                        im->nExperts, H))
                    return false;
                dumpStage(parityDir, "tm", "rou", static_cast<uint32_t>(si), slot,
                          im->dRouter, im->nExperts, s);
                kInklingRouter(im->dRouter, static_cast<const float *>(dev.probsBias),
                               im->dIdx, im->dWgt, im->nExperts, im->nUsed,
                               cfg.expertWeightsScale, s);
                dumpParity(parityDir, "tm-idx-t%u-s%u", static_cast<uint32_t>(si),
                           slot, reinterpret_cast<const float *>(im->dIdx),
                           im->nUsed, s);
                gatherScalesKernel<<<1, 32, 0, s>>>(
                        im->dGateUpRank, im->dDownRank, im->dIdx,
                        static_cast<const float *>(dev.gateUpScale),
                        static_cast<const float *>(dev.downScale), im->nUsed);

                K(2);// router + expert-scale gather
                profMark(P_EXP_GU);

                // ---- routed experts (device-side expert ids, no host sync) ---
                if (!gemvExpertAny(w.ffnGateUpExps.type, dev.gateUp,
                                   im->dNormed, 0, im->dGateUp,
                                   2 * im->expertFf, H, im->dIdx, im->nUsed,
                                   s)) {
                    err = "expert gate_up GEMV launch failed";
                    return false;
                }
                kInklingSituAll(im->dGateUp, im->nUsed, im->expertFf,
                                im->dGateUpRank, cfg.situBeta,
                                cfg.situLinearBeta, s);
                K(1);// fused per-expert scale + SITU
                profMark(P_EXP_DOWN);

                // rowsPer is the OUTPUT rows per expert (H) and cols the
                // reduction length (expertFf): the down projection maps
                // expertFf -> H.  Gate/up is 2*expertFf -> H, where both are
                // 2304 on this model, so only down distinguishes the two.
                if (!gemvExpertAny(w.ffnDownExps.type, dev.down, im->dGateUp,
                                   2 * im->expertFf, im->dExpertOut, H,
                                   im->expertFf, im->dIdx, im->nUsed, s)) {
                    err = "expert down GEMV launch failed";
                    return false;
                }
                kInklingMoeAccum(im->dExpertOut, im->dWgt, im->dDownRank,
                                 im->dMoeOut, im->nUsed, H, s);
                dumpStage(parityDir, "tm", "moe", static_cast<uint32_t>(si), slot,
                          im->dMoeOut, H, s);

                K(1);// moe accumulation
                profMark(P_FFN_POST);

                // ---- FFN post conv + residual --------------------------------
                kInklingShortConv(im->dMoeOut,
                                  static_cast<const float *>(dev.ffnPostConvW),
                                  nullptr,
                                  hist + static_cast<size_t>(im->histLen) *
                                                 (2 * im->kvDim + H),
                                  im->dConvH, H, im->nConv, 1, true, postScale, s);
                dumpDev(dumpDir, "ffn_out", im->dConvH, H, slot, s);
                combine(im->dX, im->dFfnInp, im->dConvH, H, 1.0f, 1.0f, s);
                K(2);// ffn post-conv + residual
                dumpDev(dumpDir, "l_out", im->dX, H, slot, s);
                traceNorm(static_cast<uint32_t>(si), physIdx, im->dX, H, s);
                dumpParity(parityDir, "tm-t%u-s%u", static_cast<uint32_t>(si),
                           slot, im->dX, H, s);
                return true;
            };

            // ---- prelude ------------------------------------------------------
            for (uint32_t il = 0; il < cfg.loopStart; ++il)
                if (!runLayer(il, il, nullptr)) return false;
            const uint32_t preludeBlocks = nBlocks;
            if (!copyD2D(im->dPrelude, im->dX, H * sizeof(float), s)) {
                err = "prelude copy failed";
                return false;
            }
            dumpParity(parityDir, "tm-t%u-prelude", static_cast<uint32_t>(si), 0,
                       im->dX, H, s);

            // ---- three loop passes -------------------------------------------
            for (uint32_t pass = 0; pass < cfg.loopCount; ++pass) {
                uint32_t msgStates = 0;
                if (pass > 0) {
                    const uint32_t skip = preludeBlocks + 1;
                    const uint32_t nBank = nBlocks > skip ? nBlocks - skip : 0;
                    if (!copyD2D(im->dBankSnap,
                                 im->dBlockBank + static_cast<size_t>(skip) * H,
                                 static_cast<size_t>(nBank) * H * sizeof(float), s) ||
                        !copyD2D(im->dPassOut, im->dX, H * sizeof(float), s)) {
                        err = "message bank copy failed";
                        return false;
                    }
                    nBlocks = preludeBlocks;
                    if (!copyD2D(im->dX, im->dPrelude, H * sizeof(float), s) ||
                        !copyD2D(im->dBankSnap + static_cast<size_t>(nBank) * H,
                                 im->dPassOut, H * sizeof(float), s)) {
                        err = "message bank copy failed";
                        return false;
                    }
                    msgStates = nBank + 1;
                }
                for (uint32_t ip = cfg.loopStart; ip < cfg.loopEnd; ++ip) {
                    const uint32_t slot =
                            pass == 0
                                    ? ip
                                    : im->phys + (pass - 1) * im->core +
                                              (ip - cfg.loopStart);
                    if (pass == 0) {
                        if (!runLayer(ip, slot, nullptr)) return false;
                        continue;
                    }
                    const void *mq = im->layers[ip].msgQuery;
                    if (mq == nullptr) {
                        mq = static_cast<const float *>(im->legacyMsg) +
                             static_cast<size_t>(ip - cfg.loopStart) * 4 * H;
                    }
                    kInklingMix(im->dBankSnap, msgStates,
                                static_cast<const float *>(mq), 4, H, im->dMsgMixed, s);
                    if (!runLayer(ip, slot, im->dMsgMixed)) return false;
                }
            }

            // ---- postlude -----------------------------------------------------
            for (uint32_t il = cfg.loopEnd; il < im->phys; ++il)
                if (!runLayer(il, il, nullptr)) return false;

            K(2);// ffn post-conv + residual of the last layer
            profMark(P_OUT_MIX);

            // ---- output depth mix + norm + LM head + softcap + trim ----------
            if (!mixCurrent(im->dMixOut, im->outResQuery, 1)) return false;
            kInklingRmsNorm(im->dMixOut,
                            static_cast<const float *>(im->finalNorm), im->dNormed,
                            H, kRmsEps, s);
            K(2);// output depth mix + final norm
            profMark(P_LMHEAD);
            if (!gemvAny(im->lmHeadType, im->lmHead, im->dNormed, im->dLogits,
                         im->vocab, H, s)) {
                err = "LM head GEMV launch failed";
                return false;
            }
            if (cfg.finalLogitSoftcapping > 0.0f)
                softcap(im->dLogits, im->vocab, cfg.finalLogitSoftcapping, s);
            K(cfg.finalLogitSoftcapping > 0.0f ? 1u : 0u);
            if (devicePos) kInklingPosStep(im->dPos, s);
            if (copyLogits && (allLogits || si + 1 == tokens.size())) {
                float *dst = allLogits ? logits + si * im->vocab : logits;
                if (cudaMemcpyAsync(dst, im->dLogits,
                                    im->vocab * sizeof(float),
                                    cudaMemcpyDeviceToHost, s) != cudaSuccess) {
                    cudaGetLastError();
                    err = "logits copy failed";
                    return false;
                }
            }
            return CK();
        };

        // Upload one embedding row into the fixed device row the graph reads.
        auto uploadEmb = [&](size_t si) -> bool {
            std::vector<float> row = model.quantizedEmbeddings_.getRow(
                    static_cast<uint32_t>(tokens[si]));
            if (row.size() != H) {
                err = "bad embedding row for token " +
                      std::to_string(tokens[si]);
                return false;
            }
            if (im->pinnedEmb.empty()) {
                err = "embedding upload failed: pinned ring is empty";
                return false;
            }
            float *pin = im->pinnedEmb[im->pinnedEmbIdx %
                                       static_cast<uint32_t>(im->pinnedEmb.size())];
            ++im->pinnedEmbIdx;
            std::memcpy(pin, row.data(), H * sizeof(float));
            const cudaError_t ue = cudaMemcpyAsync(
                    im->embAll, pin, H * sizeof(float),
                    cudaMemcpyHostToDevice, s);
            if (ue != cudaSuccess) {
                cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
                cudaStreamIsCapturing(s, &cs);
                err = std::string("embedding upload failed: ") +
                      cudaGetErrorString(ue) + " (pinned->device, bytes=" +
                      std::to_string(H * sizeof(float)) + ", dst=" +
                      std::to_string(reinterpret_cast<uintptr_t>(im->embAll)) +
                      ", src=" +
                      std::to_string(reinterpret_cast<uintptr_t>(pin)) +
                      ", capture=" + std::to_string(static_cast<int>(cs)) +
                      ")";
                cudaGetLastError();
                return false;
            }
            return true;
        };

        // The graph is captured for a single-token decode call only.  A
        // multi-token call (prefill, or the 1000-token context test) is not a
        // capture candidate: it has variable shape and per-token copies, and a
        // failed capture must never take the whole model off the GPU.
        const bool graphWanted =
                tokens.size() == 1 && dumpDir == nullptr && !im->prof &&
                !im->graphDisabled && im->headDim == 128u &&
                im->extent <= 4096u &&
                std::getenv("TINYCODER_INKLING_GPU_NOGRAPH") == nullptr;

        auto runDirect = [&]() -> bool {
            std::vector<float> hostEmb(tokens.size() * H);
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::vector<float> row = model.quantizedEmbeddings_.getRow(
                        static_cast<uint32_t>(tokens[i]));
                if (row.size() != H) {
                    err = "bad embedding row for token " +
                          std::to_string(tokens[i]);
                    return false;
                }
                std::memcpy(hostEmb.data() + i * H, row.data(),
                            H * sizeof(float));
            }
            // Async copy ON THE COMPUTE STREAM, then wait for it: a plain
            // cudaMemcpy runs on the legacy default stream, and a non-blocking
            // compute stream has no implicit ordering with it, so the copy's
            // device DMA could still be in flight when the first token reads
            // this buffer.  The stream sync also keeps the pageable source
            // buffer alive for the copy's lifetime.  This is the multi-token
            // prefill path, so one wait per call costs nothing.
            const cudaError_t ue = cudaMemcpyAsync(
                    im->embAll, hostEmb.data(), hostEmb.size() * sizeof(float),
                    cudaMemcpyHostToDevice, s);
            if (ue == cudaSuccess &&
                cudaStreamSynchronize(s) != cudaSuccess)
                cudaGetLastError();
            if (ue != cudaSuccess) {
                err = std::string("embedding upload failed (direct): ") +
                      cudaGetErrorString(ue) + " (host->device, bytes=" +
                      std::to_string(hostEmb.size() * sizeof(float)) +
                      ", dst=" +
                      std::to_string(reinterpret_cast<uintptr_t>(im->embAll)) +
                      ")";
                cudaGetLastError();
                return false;
            }
            for (size_t si = 0; si < tokens.size(); ++si)
                if (!enqueueToken(si, false, true)) return false;
            return true;
        };

        // ---- batched prefill -------------------------------------------------
        // A multi-token call is a prompt chunk: run the [H, m] graph for kBatchM
        // (128) tokens, so a 512-token prompt is four layer-major passes instead
        // of the old 32 sixteen-token chunks.  Each pass reads a dense weight
        // matrix once per 4 tokens (int8Gemv's largest exact slice) and each
        // routed expert's weights once per ~8-token group; the opt-in tensor-core
        // levers read them once per pass.  The per-token path stays the reference
        // (TINYCODER_INKLING_BATCHED=0), and this call never falls back
        // mid-flight: a failure here is an error, because the KV/conv state has
        // already advanced.
        const char *batchEnv = std::getenv("TINYCODER_INKLING_BATCHED");
        // "2" forces the batched graph even for one token: that is how the
        // parity tool isolates a batched kernel from the batching itself.
        const bool batchForced = batchEnv != nullptr && batchEnv[0] == '2';
        // Geometry the [H, m] graph assumes (its attention kernel is the
        // warp-per-head form).  Anything else keeps the token loop rather than
        // failing the whole forward.
        const bool batchGeometry = cfg.headDim == 128u &&
                                   cfg.numAttentionHeads <= 32u &&
                                   cfg.numAttentionHeads % cfg.numKVHeads == 0 &&
                                   cfg.relativeExtent <= 4096u;
        const bool batchWanted = (tokens.size() > 1 || batchForced) &&
                                 batchGeometry && dumpDir == nullptr &&
                                 (batchEnv == nullptr || batchEnv[0] != '0') &&
                                 std::getenv("TINYCODER_INKLING_GPU_NOBATCH") ==
                                         nullptr;
        if (batchWanted) {
            for (size_t off = 0; off < tokens.size(); off += Impl::kBatchM) {
                const size_t n =
                        std::min<size_t>(Impl::kBatchM, tokens.size() - off);
                const std::vector<int32_t> chunk(tokens.begin() + off,
                                                 tokens.begin() + off + n);
                float *dst = allLogits ? logits + off * im->vocab : logits;
                if (!forwardBatch(model, chunk, allLogits, dst, err)) return false;
            }
            const cudaError_t be = cudaStreamSynchronize(s);
            if (be != cudaSuccess) {
                cudaGetLastError();
                err = std::string("cudaStreamSynchronize failed: ") +
                      cudaGetErrorString(be);
                return false;
            }
            im->dPosValid = false;// the batched path advances kvPos only
            return true;
        }

        if (graphWanted && !im->graphReady) {
            // Capture on a side stream: the compute stream never enters capture
            // mode, so a failure here cannot make later copies fail.
            if (cudaStreamSynchronize(s) != cudaSuccess) {
                cudaGetLastError();
                im->graphDisabled = true;
            } else {
                const int32_t startPos = im->kvPos;
                if (cudaMemcpyAsync(im->dPos, &startPos, sizeof(int32_t),
                                    cudaMemcpyHostToDevice, s) != cudaSuccess ||
                    cudaStreamSynchronize(s) != cudaSuccess) {
                    cudaGetLastError();
                    im->graphDisabled = true;
                    std::fprintf(stderr, "[inkling] device-pos upload failed; "
                                         "direct launches\n");
                } else {
                    im->dPosValid = true;
                    const cudaStream_t mainStream = s;
                    s = im->captureStream;
                    const cudaError_t ceb = cudaStreamBeginCapture(
                            s, cudaStreamCaptureModeThreadLocal);
                    if (ceb != cudaSuccess) {
                        cudaGetLastError();
                        s = mainStream;
                        im->graphDisabled = true;
                        std::fprintf(stderr,
                                     "[inkling] cudaStreamBeginCapture: %s; "
                                     "direct launches\n",
                                     cudaGetErrorString(ceb));
                    } else {
                        const bool ok = enqueueToken(0, true, false);
                        cudaGraph_t g = nullptr;
                        const cudaError_t cee = cudaStreamEndCapture(s, &g);
                        s = mainStream;
                        if (!ok || cee != cudaSuccess || g == nullptr) {
                            if (g != nullptr) cudaGraphDestroy(g);
                            cudaGetLastError();
                            im->graphDisabled = true;
                            std::fprintf(stderr,
                                         "[inkling] cudaStreamEndCapture: %s; "
                                         "direct launches\n",
                                         cudaGetErrorString(cee));
                        } else {
                            cudaGraphExec_t exec = nullptr;
                            const cudaError_t cei =
                                    cudaGraphInstantiate(&exec, g, 0);
                            if (cei != cudaSuccess || exec == nullptr) {
                                cudaGraphDestroy(g);
                                cudaGetLastError();
                                im->graphDisabled = true;
                                std::fprintf(stderr,
                                             "[inkling] cudaGraphInstantiate: "
                                             "%s; direct launches\n",
                                             cudaGetErrorString(cei));
                            } else {
                                im->graph = g;
                                im->graphExec = exec;
                                im->graphReady = true;
                                std::fprintf(stderr,
                                             "[inkling] CUDA graph ready "
                                             "(single-token decode)\n");
                            }
                        }
                    }
                }
            }
        }
        if (graphWanted && im->graphReady && !im->dPosValid) {
            // A multi-token direct call advanced kvPos without dPos.
            const int32_t p = im->kvPos;
            if (cudaMemcpyAsync(im->dPos, &p, sizeof(int32_t),
                                cudaMemcpyHostToDevice, s) == cudaSuccess)
                im->dPosValid = true;
        }
        if (!graphWanted || !im->graphReady) {
            const bool ok = runDirect();
            if (ok && !graphWanted) im->dPosValid = false;
            if (!ok) return false;
        } else {
            for (size_t si = 0; si < tokens.size(); ++si) {
                if (!uploadEmb(si)) return false;
                if (cudaGraphLaunch(im->graphExec, s) != cudaSuccess) {
                    cudaGetLastError();
                    err = "cudaGraphLaunch failed";
                    return false;
                }
                if (allLogits || si + 1 == tokens.size()) {
                    float *dst = allLogits ? logits + si * im->vocab : logits;
                    if (cudaMemcpyAsync(dst, im->dLogits,
                                        im->vocab * sizeof(float),
                                        cudaMemcpyDeviceToHost, s) != cudaSuccess) {
                        cudaGetLastError();
                        err = "logits copy failed";
                        return false;
                    }
                }
            }
        }

        const cudaError_t e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) {
            cudaGetLastError();
            err = std::string("cudaStreamSynchronize failed: ") +
                  cudaGetErrorString(e);
            return false;
        }
        im->kvPos += static_cast<int>(tokens.size());
        if (im->prof) {
            im->profMark(-1, s);
            im->profPrint(tokens.size());
        }
        return true;
    }

#endif// __CUDA_ARCH__

}// namespace tinycoder::gpu
