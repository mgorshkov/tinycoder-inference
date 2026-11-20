// ContextPlanTest.cpp — the memory arithmetic behind the context window.
//
// Every expected value here is derived from the geometries in the plan's
// Global Constraints (G27: 64 layers / 16 attention / 4 KV / 256 dim;
// G9: 32 / 8 / 4 / 256) and from the budget constants (900 MiB fixed
// overhead, 256 MiB safety margin, multiples of 256, floor 1024).  If one of
// these fails, either the planner changed or the arithmetic did — both are
// worth stopping for.

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "ContextPlan.hpp"

using tinycoder::detail::ContextPlan;
using tinycoder::detail::ContextRequest;
using tinycoder::detail::kvBytesPerToken;
using tinycoder::detail::KvPrecision;
using tinycoder::detail::planContext;

namespace {

    constexpr uint64_t MiB = 1024ull * 1024ull;
    constexpr uint64_t GiB = 1024ull * MiB;

    // G27 — Ternary-Bonsai-2-27B / Qwen3.8-27B: 64 layers, 16 full-attention
    // layers (interval 4), 4 K/V heads of 256, and 64 MRoPE dims on top of the
    // 256 text dims (the cos/sin row width is 320, i.e. 1280 B/token).
    ContextRequest g27() {
        ContextRequest r;
        r.modelMax = 262144;
        r.numLayers = 64;
        r.numAttentionLayers = 16;
        r.numKVHeads = 4;
        r.headDim = 256;
        r.ropeDims = 320;
        r.int8KvSupported = true;
        return r;
    }

    // G9 — Ornith-1.5-9B: 32 layers, 8 full-attention layers, same K/V shape.
    ContextRequest g9() {
        ContextRequest r;
        r.modelMax = 262144;
        r.numLayers = 32;
        r.numAttentionLayers = 8;
        r.numKVHeads = 4;
        r.headDim = 256;
        r.ropeDims = 320;
        r.int8KvSupported = true;
        return r;
    }

}// namespace

// ---------------------------------------------------------------------------
// Byte accounting.  These four numbers are what every window decision in this
// file is built on, so they are asserted exactly rather than approximately.
// ---------------------------------------------------------------------------

TEST(ContextPlan, KvBytesG27) {
    const auto r = g27();
    EXPECT_EQ(kvBytesPerToken(r, KvPrecision::F32, /*mirror=*/true), 165632u);
    EXPECT_EQ(kvBytesPerToken(r, KvPrecision::F32, /*mirror=*/false), 132352u);
    EXPECT_EQ(kvBytesPerToken(r, KvPrecision::I8, /*mirror=*/false), 34560u);
}

TEST(ContextPlan, KvBytesG9) {
    const auto r = g9();
    EXPECT_EQ(kvBytesPerToken(r, KvPrecision::F32, /*mirror=*/true), 83456u);
    EXPECT_EQ(kvBytesPerToken(r, KvPrecision::F32, /*mirror=*/false), 66816u);
    EXPECT_EQ(kvBytesPerToken(r, KvPrecision::I8, /*mirror=*/false), 17920u);
}

// ---------------------------------------------------------------------------
// Auto: 2048, at today's precision, while printing what memory would allow.
// ---------------------------------------------------------------------------

TEST(ContextPlan, AutoIs2048WithRoomToSpare) {
    auto r = g27();
    r.gpuFixedOverhead = 900ull << 20;// the ladder, pinned
    r.weightBytes = 4 * GiB;
    r.freeVram = 12 * GiB;
    const auto p = planContext(r);
    EXPECT_EQ(p.seqLen, 2048u);
    EXPECT_EQ(p.kv, KvPrecision::F32);
    EXPECT_TRUE(p.kvMirror);
    EXPECT_TRUE(p.kvOnGpu);
    // maxAllowed is the LADDER's best fit, not the rung this request landed on:
    // 44288 is what f32+mirror holds, 213248 is what the int8 rung holds, and a
    // caller asking "how big can this checkpoint go" means the latter.  The
    // reason line names the precision that achieves it.
    EXPECT_EQ(p.maxAllowed, 213248u);
    EXPECT_NE(p.reason.find("auto"), std::string::npos);
    EXPECT_NE(p.reason.find("int8"), std::string::npos);
}

// 12 GiB free / 4 GiB weights: f32+mirror holds 44288 tokens, so a request
// inside that is served at f32+mirror — no escalation for a window that fits.
TEST(ContextPlan, RequestedHonoredUpToBudget) {
    auto r = g27();
    r.gpuFixedOverhead = 900ull << 20;// the ladder, pinned
    r.weightBytes = 4 * GiB;
    r.freeVram = 12 * GiB;
    r.requested = 40960;
    const auto p = planContext(r);
    EXPECT_EQ(p.seqLen, 40960u);
    EXPECT_EQ(p.kv, KvPrecision::F32);
    EXPECT_TRUE(p.kvMirror);
    EXPECT_EQ(p.kvBytes, 40960ull * 165632ull);
}

// 8 GiB free, 4 GiB of weights: f32+mirror reaches 18432 tokens, f32 alone
// 23040, int8 89088 — so a 65536 request must escalate to int8, not shrink.
TEST(ContextPlan, RequestEscalatesToInt8) {
    auto r = g27();
    r.gpuFixedOverhead = 900ull << 20;// the ladder, pinned
    r.weightBytes = 4 * GiB;
    r.freeVram = 8 * GiB;
    r.requested = 65536;
    const auto p = planContext(r);
    EXPECT_EQ(p.seqLen, 65536u);
    EXPECT_EQ(p.kv, KvPrecision::I8);
    EXPECT_FALSE(p.kvMirror);
    EXPECT_EQ(p.maxAllowed, 89088u);
}

TEST(ContextPlan, RequestReducedByMemory) {
    auto r = g27();
    r.gpuFixedOverhead = 900ull << 20;// the ladder, pinned
    r.weightBytes = 4 * GiB;
    r.freeVram = 8 * GiB;
    r.requested = 524288;
    const auto p = planContext(r);
    // The largest fit in the ladder wins, and that is the full-residency int8
    // rung (89088); there is no shedding rung to inflate it any more.
    EXPECT_EQ(p.seqLen, p.maxAllowed);
    EXPECT_EQ(p.seqLen % 256u, 0u);
    EXPECT_LT(p.seqLen, 524288u);
    EXPECT_EQ(p.seqLen, 89088u);
    EXPECT_EQ(p.kv, KvPrecision::I8);
    EXPECT_NE(p.reason.find("VRAM"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Boundaries: the model's own limit, the floor, and nothing-fits.
// ---------------------------------------------------------------------------

TEST(ContextPlan, RequestAtModelMaxIsGranted) {
    auto r = g27();
    r.weightBytes = 1 * GiB;
    r.freeVram = 40 * GiB;
    r.requested = 262144;
    const auto p = planContext(r);
    EXPECT_EQ(p.seqLen, 262144u);
    EXPECT_EQ(p.reason.find("model max"), std::string::npos)
            << "a request equal to the model max must not be reported as clamped";
}

TEST(ContextPlan, RequestAboveModelMaxIsClamped) {
    auto r = g27();
    r.gpuFixedOverhead = 900ull << 20;// the ladder, pinned
    r.weightBytes = 1 * GiB;
    r.freeVram = 40 * GiB;
    r.requested = 300000;
    const auto p = planContext(r);
    EXPECT_EQ(p.seqLen, 262144u);
    EXPECT_NE(p.reason.find("model max"), std::string::npos);
}

TEST(ContextPlan, Int8UnsupportedFallsBackToF32MirrorOff) {
    auto r = g27();
    r.gpuFixedOverhead = 900ull << 20;// the ladder, pinned
    r.weightBytes = 4 * GiB;
    r.freeVram = 8 * GiB;
    r.requested = 65536;
    r.int8KvSupported = false;
    const auto p = planContext(r);
    EXPECT_EQ(p.kv, KvPrecision::F32);
    EXPECT_FALSE(p.kvMirror);
    EXPECT_EQ(p.seqLen, 23040u);
}

// The whole model does not fit (7 GiB of weights in an 8 GiB budget), so the
// fit is small; the request must be reduced to it.
// There is deliberately no minimum-residency rung: a grant that assumed one
// layer's weights let a 64k request through on Bonsai and the session ran at
// 109 tok/s with a wrong answer.  A request beyond the full-residency fit is
// reduced to that fit instead.
TEST(ContextPlan, LongRequestIsReducedToTheFullResidencyFit) {
    auto r = g27();
    r.gpuFixedOverhead = 900ull << 20;// the ladder, pinned
    r.freeVram = 11ull << 30;
    r.weightBytes = 7ull << 30;
    r.requested = r.modelMax;
    ContextPlan p = planContext(r);
    ASSERT_GT(p.seqLen, 0u);
    EXPECT_LT(p.seqLen, r.modelMax);// reduced, not the whole 262144
    EXPECT_EQ(p.seqLen, p.maxAllowed);
    EXPECT_EQ(p.kv, KvPrecision::I8);
}

TEST(ContextPlan, NoGpuUsesRam) {
    auto r = g27();
    r.weightBytes = 5 * GiB;
    r.availableRam = 40 * GiB;
    r.requested = 65536;// freeVram stays 0
    const auto p = planContext(r);
    EXPECT_FALSE(p.kvOnGpu);
    EXPECT_EQ(p.kv, KvPrecision::F32);
    EXPECT_FALSE(p.kvMirror);
    EXPECT_EQ(p.seqLen, 65536u);
    EXPECT_NE(p.reason.find("RAM"), std::string::npos);
}

TEST(ContextPlan, BelowFloorIsRejected) {
    auto r = g9();
    r.weightBytes = 5 * GiB;
    r.freeVram = 12 * GiB;
    r.requested = 512;
    const auto p = planContext(r);
    EXPECT_EQ(p.seqLen, 0u);
    EXPECT_FALSE(p.reason.empty());
}

TEST(ContextPlan, FloorAccepted) {
    auto r = g9();
    r.weightBytes = 5 * GiB;
    r.freeVram = 12 * GiB;
    r.requested = 1024;
    const auto p = planContext(r);
    EXPECT_EQ(p.seqLen, 1024u);
}

TEST(ContextPlan, NothingFitsAtAll) {
    auto r = g27();
    r.weightBytes = 1 * GiB;
    r.freeVram = 700 * MiB;// below gpuFixedOverhead + safety margin
    const auto p = planContext(r);
    EXPECT_EQ(p.seqLen, 0u);
    EXPECT_EQ(p.maxAllowed, 0u);
    EXPECT_FALSE(p.reason.empty());
}

// A checkpoint whose metadata has no *.context_length must not make every
// request "clamped to 0": the ceiling falls back to 4096 and says so.
// The default overhead is a measurement, not a preference: a 9B f32 session at
// a 45k window used ~2.3 GiB beyond weights + K/V, and a 60k f32 window granted
// with a 900 MiB overhead OOM'd on its first prefill chunk.
TEST(ContextPlan, DefaultOverheadCoversMeasuredSessionCost) {
    const ContextRequest r;
    // ~1.3 GiB of scratch for the 27B and ~1.6 GiB for the 9B, both measured
    // with the CUDA context already excluded by the probe; 900 MiB under-granted
    // (an OOM at 60672 on the 9B) and 2560 MiB over-charged (it moved the 27B's
    // default 2048 window onto int8 and shifted the reference logit).
    EXPECT_GE(r.gpuFixedOverhead, 1280ull << 20);
    EXPECT_LE(r.gpuFixedOverhead, 1536ull << 20);
}

// TINYCODER_CONTEXT_KV8 pins the store so the same window can be measured both
// ways; a pinned precision must be honoured even when a cheaper rung fits.
TEST(ContextPlan, ForcedPrecisionSkipsThePreferredRung) {
    auto r = g27();
    r.requested = 4096;
    r.freeVram = 12ull << 30;
    r.weightBytes = 4ull << 30;
    ContextPlan p = planContext(r);
    EXPECT_EQ(p.kv, KvPrecision::F32);// unforced: the f32 mirror wins
    r.forceKv = KvPrecision::I8;
    r.forceKvSet = true;
    ContextPlan q = planContext(r);
    EXPECT_EQ(q.kv, KvPrecision::I8);
    EXPECT_EQ(q.seqLen, 4096u);
    EXPECT_FALSE(q.kvMirror);
    r.forceKv = KvPrecision::F32;
    ContextPlan f = planContext(r);
    EXPECT_EQ(f.kv, KvPrecision::F32);
}

TEST(ContextPlan, MissingModelMaxFallsBack) {
    auto r = g9();
    r.modelMax = 0;
    r.weightBytes = 1 * GiB;
    r.freeVram = 40 * GiB;
    r.requested = 65536;
    const auto p = planContext(r);
    EXPECT_EQ(p.seqLen, 4096u);
    EXPECT_NE(p.reason.find("4096"), std::string::npos);
}

// Invariants that must hold for every input, not just the cases above.
TEST(ContextPlan, ResultIsAlwaysRoundedAndBounded) {
    auto check = [](const ContextRequest &r) {
        const auto p = planContext(r);
        if (r.requested == 0) {
            EXPECT_EQ(p.seqLen % 256u, 0u) << "an auto window is memory-derived";
        } else {
            EXPECT_LE(p.seqLen, std::min(r.requested,
                                         r.modelMax ? r.modelMax : 4096u));
        }
        EXPECT_LE(p.seqLen, r.modelMax ? r.modelMax : 4096u);
        EXPECT_GE(p.maxAllowed, p.seqLen);
        EXPECT_NO_THROW({
            (void) kvBytesPerToken(r, p.kv, p.kvMirror);
        });
    };
    const auto r = g27();
    check(r);
    auto a = r;
    a.requested = 3000;
    a.weightBytes = 4 * GiB;
    a.freeVram = 12 * GiB;
    check(a);
    auto b = r;
    b.requested = 262144;
    b.weightBytes = 4 * GiB;
    b.freeVram = 12 * GiB;
    check(b);
    auto c = r;
    c.weightBytes = 4 * GiB;
    c.freeVram = 8 * GiB;
    c.requested = 524288;
    check(c);
    auto d = r;
    d.availableRam = 40 * GiB;
    d.requested = 65536;
    check(d);
    auto e = r;
    e.modelMax = 0;
    e.weightBytes = 1 * GiB;
    e.freeVram = 40 * GiB;
    check(e);
}
