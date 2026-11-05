// gdn_scan_probe -- standalone measurement of the prefill GDN recurrence
// (kQ35GdnWarpScan) for the 27B's real geometry: 48 value heads, 128x128
// state per head, 512 tokens.
//
// It includes the ENGINE header and launches the shipped kernel (no copy), so
// a number measured here is a number of the shipped code.  The two ablation
// kernels below are probe-local copies of the same token loop with one
// ingredient removed; they exist only to attribute the time and are never
// launched by the engine.
//
// Usage: gdn_scan_probe [seqLen=512] [nVHeads=48] [reps=20]

#include "GPUComputeInternal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

using namespace tinycoder::gpu;

// Link-time stubs.  The engine header references these three entry points from
// inline code the probe never calls; defining them here keeps this probe free
// of libtinycoder_core.a (whose separable-compilation device link is slow and
// would pull in the whole engine for one kernel).
namespace tinycoder::gpu {
    bool int8Gemm(int, const void *, const float *, float *, uint32_t, uint32_t,
                  uint32_t, uint32_t, void *) {
        return false;
    }
    bool int8Gemv(int, const void *, const float *, float *, uint32_t, uint32_t,
                  uint32_t, void *) {
        return false;
    }
}// namespace tinycoder::gpu
namespace tinycoder::exl3gpu {
    bool gemvFused(const void *, const float *, float *, cudaStream_t,
                   std::string &) {
        return false;
    }
}// namespace tinycoder::exl3gpu

namespace gdnprobe {

    // ------------------------------------------------------------------
    // Ablations: copies of the (RPW=1) token loop with one thing removed.
    // ABL 0 = full reference (must match the shipped kernel bit for bit),
    // ABL 1 = no cross-lane reduction (both dots stay lane-local),
    // ABL 2 = no k/q loads (lane-derived constants instead).
    // ------------------------------------------------------------------
    template<uint32_t HEADV, int ABL>
    __global__ void kGdnAblation(const float *__restrict__ qN,
                                 const float *__restrict__ kN,
                                 const float *__restrict__ vN,
                                 const float *__restrict__ beta,
                                 const float *__restrict__ decayV,
                                 float *__restrict__ state,
                                 float *__restrict__ gdnOut,
                                 uint32_t nVHeads, uint32_t valueDim,
                                 uint32_t seqLen, float invSqrtHeadV) {
        constexpr uint32_t RPB = 4u;
        constexpr uint32_t NBLK = HEADV / RPB;
        const uint32_t hv = blockIdx.x / NBLK;
        const uint32_t rowBase = (blockIdx.x % NBLK) * RPB;
        if (hv >= nVHeads) return;
        const uint32_t warp = threadIdx.x >> 5;
        const uint32_t lane = threadIdx.x & 31u;
        const uint32_t row = rowBase + warp;
        if (row >= HEADV) return;
        const uint32_t col = lane * 4u;
        const uint32_t kvn = static_cast<uint32_t>(HEADV) * nVHeads;
        float4 s = *reinterpret_cast<const float4 *>(
                state + (static_cast<size_t>(hv) * HEADV + row) * HEADV + col);
        float4 k4, q4;
        k4 = *reinterpret_cast<const float4 *>(kN + hv * HEADV + col);
        q4 = *reinterpret_cast<const float4 *>(qN + hv * HEADV + col);
        for (uint32_t t = 0; t < seqLen; ++t) {
            if (ABL != 2) {
                k4 = *reinterpret_cast<const float4 *>(
                        kN + static_cast<size_t>(t) * kvn + hv * HEADV + col);
                q4 = *reinterpret_cast<const float4 *>(
                        qN + static_cast<size_t>(t) * kvn + hv * HEADV + col);
            } else {
                k4.x += 1e-7f;
                q4.y += 1e-7f;
            }
            const float dec = __expf(decayV[static_cast<size_t>(t) * nVHeads + hv]);
            s.x *= dec;
            s.y *= dec;
            s.z *= dec;
            s.w *= dec;
            float part = s.x * k4.x;
            part = fmaf(s.y, k4.y, part);
            part = fmaf(s.z, k4.z, part);
            part = fmaf(s.w, k4.w, part);
            if (ABL == 1) {
                part = __shfl_sync(0xffffffffu, part, 0);
            } else {
                part += __shfl_down_sync(0xffffffffu, part, 16);
                part += __shfl_down_sync(0xffffffffu, part, 8);
                part += __shfl_down_sync(0xffffffffu, part, 4);
                part += __shfl_down_sync(0xffffffffu, part, 2);
                part += __shfl_down_sync(0xffffffffu, part, 1);
                part = __shfl_sync(0xffffffffu, part, 0);
            }
            const float del =
                    (vN[static_cast<size_t>(t) * kvn + hv * HEADV + row] - part) *
                    beta[static_cast<size_t>(t) * nVHeads + hv];
            s.x = fmaf(del, k4.x, s.x);
            s.y = fmaf(del, k4.y, s.y);
            s.z = fmaf(del, k4.z, s.z);
            s.w = fmaf(del, k4.w, s.w);
            float po = s.x * q4.x;
            po = fmaf(s.y, q4.y, po);
            po = fmaf(s.z, q4.z, po);
            po = fmaf(s.w, q4.w, po);
            if (ABL != 1) {
                po += __shfl_down_sync(0xffffffffu, po, 16);
                po += __shfl_down_sync(0xffffffffu, po, 8);
                po += __shfl_down_sync(0xffffffffu, po, 4);
                po += __shfl_down_sync(0xffffffffu, po, 2);
                po += __shfl_down_sync(0xffffffffu, po, 1);
            }
            if (lane == 0u)
                gdnOut[static_cast<size_t>(t) * valueDim + hv * HEADV + row] =
                        po * invSqrtHeadV;
        }
        *reinterpret_cast<float4 *>(
                state + (static_cast<size_t>(hv) * HEADV + row) * HEADV + col) = s;
    }

    struct Shapes {
        uint32_t seqLen, nVHeads, headV, valueDim, kvn;
        float invSqrtHeadV;
    };

    template<uint32_t HEADV, uint32_t RPB>
    void launchScan8(const float *qN, const float *kN, const float *vN,
                     const float *beta, const float *decay, float *state,
                     float *out, const Shapes &sh) {
        constexpr uint32_t ROWS = RPB * 4u;
        const uint32_t blocks = sh.nVHeads * (HEADV / ROWS);
        kQ35GdnWarpScan8<HEADV, RPB><<<blocks, RPB * 32u, 0, nullptr>>>(
                qN, kN, vN, beta, decay, state, out, sh.nVHeads, sh.valueDim,
                sh.seqLen, sh.invSqrtHeadV);
    }

    template<uint32_t HEADV, uint32_t RPB, uint32_t RPW, bool PF = true,
             bool PFS = false, uint32_t MAXBLK = 1u>
    void launchScan(const float *qN, const float *kN, const float *vN,
                    const float *beta, const float *decay, float *state,
                    float *out, const Shapes &sh) {
        constexpr uint32_t ROWS = RPB * RPW;
        const uint32_t blocks = sh.nVHeads * (HEADV / ROWS);
        kQ35GdnWarpScan<HEADV, RPB, RPW, PF, PFS, MAXBLK>
                <<<blocks, RPB * 32u, 0, nullptr>>>(
                qN, kN, vN, beta, decay, state, out, sh.nVHeads, sh.valueDim,
                sh.seqLen, sh.invSqrtHeadV);
    }

    template<typename F>
    float bench(F &&launch, int reps) {
        launch();// warm
        cudaDeviceSynchronize();
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) launch();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return ms / static_cast<float>(reps);
    }

    // One (head, row) of the scan, exactly the shipped expression order, for
    // an independent check of the kernel's semantics.
    void cpuRow(const std::vector<float> &qN, const std::vector<float> &kN,
                const std::vector<float> &vN, const std::vector<float> &beta,
                const std::vector<float> &decay, std::vector<float> &state,
                std::vector<float> &out, uint32_t hv, uint32_t row,
                uint32_t headV, uint32_t nVHeads, uint32_t valueDim,
                uint32_t seqLen, float invSqrtHeadV, bool decayUnity) {
        const uint32_t kvn = headV * nVHeads;
        std::vector<float> s(headV);
        for (uint32_t c = 0; c < headV; ++c)
            s[c] = state[(static_cast<size_t>(hv) * headV + row) * headV + c];
        auto warpSum = [](float *p) {
            for (int d = 16; d > 0; d >>= 1)
                for (int i = 0; i < d; ++i) p[i] += p[i + d];
            return p[0];
        };
        float part[32], po[32];
        for (uint32_t t = 0; t < seqLen; ++t) {
            const float dec = decayUnity
                                      ? 1.0f
                                      : std::exp(decay[static_cast<size_t>(t) * nVHeads + hv]);
            for (uint32_t c = 0; c < headV; ++c) s[c] *= dec;
            const float *k = &kN[static_cast<size_t>(t) * kvn + hv * headV];
            const float *q = &qN[static_cast<size_t>(t) * kvn + hv * headV];
            for (uint32_t g = 0; g < 32; ++g) {
                float p = s[g * 4] * k[g * 4];
                p = std::fma(s[g * 4 + 1], k[g * 4 + 1], p);
                p = std::fma(s[g * 4 + 2], k[g * 4 + 2], p);
                p = std::fma(s[g * 4 + 3], k[g * 4 + 3], p);
                part[g] = p;
            }
            const float del =
                    (vN[static_cast<size_t>(t) * kvn + hv * headV + row] -
                     warpSum(part)) *
                    beta[static_cast<size_t>(t) * nVHeads + hv];
            for (uint32_t c = 0; c < headV; ++c) s[c] = std::fma(del, k[c], s[c]);
            for (uint32_t g = 0; g < 32; ++g) {
                float p = s[g * 4] * q[g * 4];
                p = std::fma(s[g * 4 + 1], q[g * 4 + 1], p);
                p = std::fma(s[g * 4 + 2], q[g * 4 + 2], p);
                p = std::fma(s[g * 4 + 3], q[g * 4 + 3], p);
                po[g] = p;
            }
            out[static_cast<size_t>(t) * valueDim + hv * headV + row] =
                    warpSum(po) * invSqrtHeadV;
        }
        for (uint32_t c = 0; c < headV; ++c)
            state[(static_cast<size_t>(hv) * headV + row) * headV + c] = s[c];
    }

}// namespace gdnprobe
using namespace gdnprobe;

int main(int argc, char **argv) {
    uint32_t seqLen = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 512u;
    uint32_t nVHeads = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 48u;
    int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    float decayMag = argc > 4 ? static_cast<float>(std::atof(argv[4])) : 0.02f;
    constexpr uint32_t HEADV = 128u;
    Shapes sh{seqLen, nVHeads, HEADV, HEADV * nVHeads, HEADV * nVHeads,
              1.0f / std::sqrt(static_cast<float>(HEADV))};

    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> qN(static_cast<size_t>(seqLen) * sh.kvn);
    std::vector<float> kN(qN.size()), vN(qN.size());
    std::vector<float> beta(static_cast<size_t>(seqLen) * nVHeads);
    std::vector<float> decay(beta.size(), 0.0f);
    std::vector<float> state(static_cast<size_t>(nVHeads) * HEADV * HEADV);
    std::vector<float> out(static_cast<size_t>(seqLen) * sh.valueDim);
    for (auto &x : qN) x = nd(rng);
    for (auto &x : kN) x = nd(rng);
    for (auto &x : vN) x = nd(rng);
    for (auto &x : beta) x = 0.5f + 0.1f * nd(rng);
    for (auto &x : decay) x = -decayMag - 0.001f * std::fabs(nd(rng));
    for (auto &x : state) x = 0.01f * nd(rng);
    const std::vector<float> state0 = state;

    float *dq, *dk, *dv, *db, *dd, *ds, *dOut;
    cudaMalloc(&dq, qN.size() * 4);
    cudaMalloc(&dk, kN.size() * 4);
    cudaMalloc(&dv, vN.size() * 4);
    cudaMalloc(&db, beta.size() * 4);
    cudaMalloc(&dd, decay.size() * 4);
    cudaMalloc(&ds, state.size() * 4);
    cudaMalloc(&dOut, out.size() * 4);
    cudaMemcpy(dq, qN.data(), qN.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, kN.data(), kN.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, vN.data(), vN.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dd, decay.data(), decay.size() * 4, cudaMemcpyHostToDevice);

    auto reset = [&](const std::vector<float> &st) {
        cudaMemcpy(ds, st.data(), st.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(dOut, 0, out.size() * 4);
    };
    auto fetch = [&](std::vector<float> &st, std::vector<float> &o) {
        cudaMemcpy(st.data(), ds, st.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(o.data(), dOut, o.size() * 4, cudaMemcpyDeviceToHost);
    };

    // ---- semantics: shipped kernel vs an independent CPU row ---------------
    {
        std::vector<float> st = state0;
        reset(st);
        kQ35GdnWarpScan<HEADV><<<32u, 128, 0, nullptr>>>(
                dq, dk, dv, db, dd, ds, dOut, 1u, HEADV, seqLen, sh.invSqrtHeadV);
        std::vector<float> gS(static_cast<size_t>(HEADV) * HEADV), gO(
                static_cast<size_t>(seqLen) * HEADV);
        cudaMemcpy(gS.data(), ds, gS.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gO.data(), dOut, gO.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<float> cS = st, cO(out.size(), 0.0f);
        cpuRow(qN, kN, vN, beta, decay, cS, cO, 0u, 7u, HEADV, nVHeads, HEADV,
               seqLen, sh.invSqrtHeadV, /*decayUnity=*/true);
        double dmax = 0, omax = 0;
        for (uint32_t c = 0; c < HEADV; ++c)
            dmax = std::max(dmax,
                            std::fabs(static_cast<double>(gS[7 * HEADV + c]) -
                                      cS[7 * HEADV + c]));
        for (uint32_t t = 0; t < seqLen; ++t)
            omax = std::max(omax,
                            std::fabs(static_cast<double>(gO[t * HEADV + 7]) -
                                      cO[t * HEADV + 7]));
        std::printf("cpu-vs-gpu (head0 row7): maxAbsDiff state %.3e out %.3e\n",
                    dmax, omax);
    }

    // ---- timing sweep ----------------------------------------------------
    // The GPU is at 300 MHz when idle and only boosts after a second or two of
    // load, so every variant is measured round-robin after a common warm-up and
    // the MINIMUM of the rounds is reported (a cold first sample is otherwise
    // indistinguishable from a slow variant).
    std::vector<float> refS, refO;
    {
        reset(state0);
        launchScan<HEADV, 4, 1>(dq, dk, dv, db, dd, ds, dOut, sh);
        cudaDeviceSynchronize();
        fetch(refS, refO);
    }

    struct Cand {
        const char *nm;
        uint32_t rowsPerBlock, threads, blocks;
        std::function<void()> run;
        bool bitwise = false;
    };
    std::vector<Cand> cands;
    auto add = [&](const char *nm, uint32_t blocks, uint32_t threads,
                   auto &&fn) {
        cands.push_back(Cand{nm, 0, threads, blocks,
                             [&, fn]() {
                                 reset(state0);
                                 fn();
                             },
                             false});
    };
    add("v0  1row/warp  4w", 1536, 128, [&] { launchScan<HEADV, 4, 1>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v8L 8lane 4row 4w", 384, 128, [&] { launchScan8<HEADV, 4>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v8M 8lane 4row 8w", 192, 256, [&] { launchScan8<HEADV, 8>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v1  1row/warp  4w nopf", 1536, 128, [&] { launchScan<HEADV, 4, 1, false>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v2  1row/warp  4w pfs", 1536, 128, [&] { launchScan<HEADV, 4, 1, true, true>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v3  2row/warp  4w nopf", 768, 128, [&] { launchScan<HEADV, 4, 2, false>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v4  4row/warp  4w nopf", 384, 128, [&] { launchScan<HEADV, 4, 4, false>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v5  4row/warp  4w pfs", 384, 128, [&] { launchScan<HEADV, 4, 4, true, true>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v6  1row/warp  4w lb12", 1536, 128, [&] { launchScan<HEADV, 4, 1, true, false, 12>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v7  1row/warp  4w nopf lb12", 1536, 128, [&] { launchScan<HEADV, 4, 1, false, false, 12>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v8  4row/warp  4w nopf lb10", 384, 128, [&] { launchScan<HEADV, 4, 4, false, false, 10>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("v9  4row/warp  4w pfs lb10", 384, 128, [&] { launchScan<HEADV, 4, 4, true, true, 10>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("vA  2row/warp  4w pfs lb12", 768, 128, [&] { launchScan<HEADV, 4, 2, true, true, 12>(dq, dk, dv, db, dd, ds, dOut, sh); });
    add("vB  8row/warp  4w nopf lb8", 192, 128, [&] { launchScan<HEADV, 4, 8, false, false, 8>(dq, dk, dv, db, dd, ds, dOut, sh); });

    // correctness: every candidate must be bit-identical to v0
    for (auto &c : cands) {
        reset(state0);
        c.run();
        cudaDeviceSynchronize();
        std::vector<float> s1, o1;
        fetch(s1, o1);
        c.bitwise = (std::memcmp(s1.data(), refS.data(), s1.size() * 4) == 0) &&
                    (std::memcmp(o1.data(), refO.data(), o1.size() * 4) == 0);
    }
    // warm-up: 2 s of the baseline so the clocks are settled before any sample
    {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < 400; ++i) cands[0].run();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("warmup %.0f ms\n", ms);
    }

    std::vector<float> best(cands.size(), 1e9f);
    for (int round = 0; round < 5; ++round)
        for (size_t i = 0; i < cands.size(); ++i)
            best[i] = std::min(best[i], bench(cands[i].run, reps));
    const float baseMs = best[0];
    for (size_t i = 0; i < cands.size(); ++i)
        std::printf("%-28s grid %5u x %4u : %8.3f ms  (%5.1f%% of v0) %s\n",
                    cands[i].nm, cands[i].blocks, cands[i].threads, best[i],
                    100.0 * best[i] / best[0],
                    cands[i].bitwise ? "bitwise" : "DIFFERS");

    // ---- ablations -------------------------------------------------------
    auto abl = [&](int which, const char *nm) {
        float ms = bench([&]() {
            reset(state0);
            if (which == 0)
                kGdnAblation<HEADV, 0><<<nVHeads * 32u, 128, 0, nullptr>>>(
                        dq, dk, dv, db, dd, ds, dOut, nVHeads, sh.valueDim,
                        seqLen, sh.invSqrtHeadV);
            else if (which == 1)
                kGdnAblation<HEADV, 1><<<nVHeads * 32u, 128, 0, nullptr>>>(
                        dq, dk, dv, db, dd, ds, dOut, nVHeads, sh.valueDim,
                        seqLen, sh.invSqrtHeadV);
            else
                kGdnAblation<HEADV, 2><<<nVHeads * 32u, 128, 0, nullptr>>>(
                        dq, dk, dv, db, dd, ds, dOut, nVHeads, sh.valueDim,
                        seqLen, sh.invSqrtHeadV);
        }, reps);
        std::printf("%-22s %8.3f ms  (%5.1f%% of v0)\n", nm, ms,
                    100.0 * ms / baseMs);
    };
    abl(0, "ablate: reference");
    abl(1, "ablate: no warp reduce");
    abl(2, "ablate: no k/q loads");
    return 0;
}
