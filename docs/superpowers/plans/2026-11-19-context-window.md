# Context Window Sized by Memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the context window a function of the model's trained limit and available VRAM/RAM instead of a hardcoded 2048, and make an 8-bit K/V store the mechanism that lets long windows fit an 11 GB card.

**Architecture:** A pure `planContext()` module decides the window and K/V precision from a memory budget; `Model::load` stops overwriting the caller's request and stores the granted window; the host K/V cache is allocated lazily for attention layers only; int8 K/V becomes a real store (prefill writes it, prefill attention reads it) for the qwen35 paths. Two new runnable probes (`--ctx`, `tools/ctx_probe`) are the evidence vehicle.

**Tech Stack:** C++20, CUDA (sm_75), CMake, GoogleTest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-19-context-window-design.md` — the executor reads both; the spec wins on behaviour, this plan wins on names and numbers.

## Global Constraints

- Build with the core target named explicitly, and check the exit status, never a grep window:
  `cmake --build build-1.0 --target tinycoder_core -j 20 || { echo BUILD FAILED; exit 1; }`
  Then rebuild the tool you run, in the same command.
- **Do not commit.** The user commits. Every task ends by leaving the tree dirty and listing the changed files in its report.
- Work in place on branch `1.0.2` (user's choice); no worktree. If the branch or tree state is not what this plan assumes, stop and ask.
- **One GPU session at a time**; never `pkill -f` (it matches your own shell) — `pgrep -x tinycoder_bench` then kill the PID.
- Correctness baselines that must not move: `exl3_gpu_parity` = **29 `[FAIL]`** lines; `tiny_logits_probe` on the 27B EXL3 `"def hello():"` = **id 198**, top-5 `198,1133,271,460,317`, logit **19.308 ± 2e-3**; `tinycoder_test` = **90 tests, 73 passed, 0 failed**; `scripts/text_gate.sh` exit 0.
- Default behaviour must not change: with no request the window is 2048 tokens, K/V precision is f32 + int8 mirror, and every published number stays comparable.
- No date-stamped README paragraphs. Numbers go in the ledger (`.superpowers/sdd/2026-11-19-context-window/progress.md`) and in `docs/`.
- Design docs live in `docs/`; this plan lives in `docs/superpowers/plans/`.
- Geometry constants used by every test below:
  - **G27** (Bonsai / 27B EXL3): 64 layers, 16 attention layers, 4 K/V heads, head dim 256, rope dims 256, model max 262144.
    Bytes per token: f32 store `132352`, f32 store + mirror `165632`, int8 store `34560`.
  - **G9** (Ornith-9B): 32 layers, 8 attention layers, 4 K/V heads, head dim 256.
    Bytes per token: f32 `66816`, f32 + mirror `83456`, int8 `17920`.
  - Budget constants: `gpuFixedOverhead` = 900 MiB, safety margin = 256 MiB, rounding = multiples of 256, floor 1024.

## Review Focus

Inputs and conditions the spec implies but does not enumerate; each line names the behaviour a reasonable person expects, and the task that pins it.

1. **A request exactly at the model's advertised limit** (262144). Expect: granted as-is, not rejected by an off-by-one (`> modelMax` vs `>=`). Pinned in Task 1 (`RequestAtModelMaxIsGranted`) and Task 7 (9B at 262144).
2. **A request larger than any possible memory** (e.g. 1M tokens on an 11 GB card). Expect: reduced to what memory allows, reported, engine runs; no OOM, no silent CPU fallback, no wrong numbers. Pinned in Task 1 (`RequestReducedByMemory`) and Task 6.
3. **A second forward that overflows the window** (cache already holds N tokens, request needs M more, N+M > window). Expect: a diagnosed error naming the arithmetic, cache untouched; today this silently writes past the end. Pinned in Task 4 (`ContextFullMidConversation`).
4. **A partially offloaded model** (nGpu between 1 and numLayers−1): the host K/V slot map must agree with the GPU slot map or attention reads another layer's history. Expect: identical logits at NGL=0 and NGL=2. Pinned in Task 3 (slot-map parity run).
5. **A checkpoint whose metadata has no `*.context_length`.** Expect: a sane fallback ceiling (4096) rather than 0, which would make every request "clamped to 0". Pinned in Task 1 (`MissingModelMaxFallsBack`) and Task 2.

---

### Task 1: Planner module and its tests

**Files:**
- Create: `include/ContextPlan.hpp`
- Create: `src/cpp/core/ContextPlan.cpp`
- Create: `unit_tests/ContextPlanTest.cpp`
- Modify: `unit_tests/CMakeLists.txt` (add `ContextPlanTest.cpp` to the `tinycoder_test` source list, next to `DequantizeTest.cpp`)
- Note: the core library globs `src/cpp/core/*.cpp`, so no root-CMake change is needed.

**Interfaces:**
- Consumes: nothing.
- Produces (exact, later tasks rely on these names):
```cpp
namespace tinycoder::detail {
enum class KvPrecision { F32, I8 };

struct ContextRequest {
    uint32_t requested = 0;          // 0 = auto
    uint32_t modelMax = 0;           // 0 = unknown -> 4096 fallback
    uint32_t numLayers = 0;
    uint32_t numAttentionLayers = 0; // full-attention layers that own K/V
    uint32_t numKVHeads = 0;
    uint32_t headDim = 0;
    uint32_t ropeDims = 0;
    bool     int8KvSupported = false;
    uint64_t weightBytes = 0;        // device-resident weight form
    uint64_t freeVram = 0;           // 0 = no usable GPU
    uint64_t availableRam = 0;
    uint64_t gpuFixedOverhead = 900ull << 20;
};

struct ContextPlan {
    uint32_t seqLen = 0;             // 0 = nothing could be granted
    uint32_t maxAllowed = 0;         // largest the budget permits
    KvPrecision kv = KvPrecision::F32;
    bool kvMirror = false;
    bool kvOnGpu = false;
    uint64_t kvBytes = 0;
    std::string reason;
};

ContextPlan planContext(const ContextRequest &r);
uint64_t kvBytesPerToken(const ContextRequest &r, KvPrecision p, bool mirror);
uint64_t probeFreeVramBytes();       // 0 without USE_CUDA or with TINYCODER_GPU=0
uint64_t probeAvailableRamBytes();   // MemAvailable, sysinfo fallback
}
```

- [ ] **Step 1: Write the failing tests**

`unit_tests/ContextPlanTest.cpp`, `TEST(ContextPlan, ...)` cases, table-driven where the case is a number:

| test | input | expected |
|---|---|---|
| `KvBytesG27` | G27 | `kvBytesPerToken(F32,true)==165632`, `(F32,false)==132352`, `(I8,false)==34560` |
| `KvBytesG9` | G9 | `83456`, `66816`, `17920` |
| `AutoIs2048WithRoomToSpare` | G27, weights 4 GiB, freeVram 12 GiB, requested 0, int8 on | `seqLen==2048`, `kv==F32`, `kvMirror==true`, `kvOnGpu==true`, `maxAllowed==44288` |
| `RequestedHonoredUpToBudget` | same, requested 65536 | `seqLen==65536`, `kv==F32`, `kvBytes==65536ull*165632ull` |
| `RequestEscalatesToInt8` | G27, weights 4 GiB, freeVram 8 GiB, requested 65536, int8 on | `seqLen==65536`, `kv==I8`, `kvMirror==false`, `maxAllowed==89088` |
| `RequestReducedByMemory` | G27, weights 4 GiB, freeVram 8 GiB, requested 524288 | `seqLen==89088`, `reason` contains `VRAM` |
| `RequestAtModelMaxIsGranted` | G27, weights 1 GiB, freeVram 40 GiB, requested 262144, modelMax 262144 | `seqLen==262144`, `reason` does **not** contain `model max` |
| `RequestAboveModelMaxIsClamped` | requested 300000, modelMax 262144, weights 1 GiB, freeVram 40 GiB | `seqLen==262144`, `reason` contains `model max` |
| `Int8UnsupportedFallsBackToF32MirrorOff` | G27, weights 4 GiB, freeVram 8 GiB, requested 65536, `int8KvSupported=false` | `kv==F32`, `kvMirror==false`, `seqLen==23040` |
| `MinimumResidencyRung` | G27, weights 7 GiB (does not fit whole), freeVram 8 GiB, requested 65536, int8 on | `seqLen==65536` or larger, `kv==I8` (the ladder's partial-residency rung) |
| `NoGpuUsesRam` | freeVram 0, availableRam 40 GiB, weights 5 GiB, requested 65536 | `kvOnGpu==false`, `kv==F32`, `kvMirror==false`, `seqLen==65536` |
| `BelowFloorIsRejected` | requested 512 | `seqLen==0`, `reason` non-empty |
| `FloorAccepted` | requested 1024, plenty of memory | `seqLen==1024` |
| `NothingFitsAtAll` | G27, weights 20 GiB, freeVram 8 GiB, requested 0 | `seqLen==0`, `maxAllowed==0`, `reason` non-empty |
| `ResultIsAlwaysRoundedAndBounded` | loop over every case above | `seqLen%256==0 \|\| seqLen==0`, `maxAllowed>=seqLen`, `seqLen<=modelMax` |
| `MissingModelMaxFallsBack` | `modelMax==0`, plenty of memory, requested 65536 | `seqLen==4096`, `reason` contains `4096` |

- [ ] **Step 2: Run to verify it fails**

```bash
cmake --build build-1.0 --target tinycoder_core -j 20 || { echo BUILD FAILED; exit 1; }
cmake --build build-1.0 --target tinycoder_test -j 20 || { echo BUILD FAILED; exit 1; }
./build-1.0/unit_tests/tinycoder_test --gtest_filter='ContextPlan*'
```
Expected: FAIL — `ContextPlan.hpp` not found / undefined reference to `planContext`. A compile failure is the expected first red.

- [ ] **Step 3: Implement `include/ContextPlan.hpp` and `src/cpp/core/ContextPlan.cpp`**

`kvBytesPerToken` = `numAttentionLayers × (storeBytes + storeScales + mirrorBytes) + 4·ropeDims`, with
`storeBytes = 2·numKVHeads·headDim·(4|1)`, `storeScales = 2·numKVHeads·4` for the int8 store and 0 for f32,
`mirrorBytes = 2·numKVHeads·(headDim + 4)` when the store is f32 and the mirror is on (K and V as int8 plus one
f32 scale per (token, k/v head)) and 0 otherwise, and `ropeDims = headDim + mropeDims` (the cos/sin row width:
320 for a 256-dim qwen35 with 64 mrope dims, which is the 1280 B/token in the geometry constants).

`planContext`, in this order (the order is the specification — it decides which precision a granted window reports):
1. `ceiling = modelMax ? modelMax : 4096`; when `modelMax == 0`, append `model max unknown, assuming 4096` to `reason`.
2. Build the ladder as a list of `(precision, mirror, residency)` rungs: `(F32,true,full)`, `(F32,false,full)`, then — only when `int8KvSupported` — `(I8,false,full)`, `(I8,false,min)`. For each rung, `budget = freeVram − gpuFixedOverhead − 268435456 − weightsAtResidency` (`weightsAtResidency` = `weightBytes` for `full`, `weightBytes/numLayers` for `min`), and `fit = floor256(budget / kvBytesPerToken(rung))` clamped to `[0, ceiling]`.
3. `want = requested ? min(requested, ceiling) : min(2048, maxAllowedDraft)` where the auto form is simply `min(2048, bestFit)` with `bestFit` = the largest `fit` in the ladder.
4. Grant: the **first** rung whose `fit >= want` supplies `seqLen = want`, that rung's precision, and `maxAllowed = fit`. If no rung reaches `want`, take the rung with the largest `fit`, grant `fit`, and say which resource bound it (`limited by VRAM: …` / `limited by RAM: …`). First rung wins ties, so a short request is served at today's precision.
5. `kvBytes = seqLen × kvBytesPerToken(granted rung)`; `kvOnGpu = freeVram != 0`.
6. No GPU (`freeVram == 0`): budget = `availableRam − weightBytes − 2147483648`, ladder = `(F32,false,full)` only, `kvOnGpu=false`, `reason` gets `; no GPU, RAM path`.
7. `seqLen == 0` when `want < 1024` or nothing fits: reason names the floor (`not even 1024 tokens fit`).
8. `reason` shapes: `requested 65536, limited by VRAM: …`, `requested 300000, clamped to model max 262144`, `auto`, `auto; memory allows up to <maxAllowed>`, `model max unknown, assuming 4096`.

`probeFreeVramBytes`: `#ifdef USE_CUDA` → return 0 when `TINYCODER_GPU=0`, else `cudaMemGetInfo` free bytes (0 on error); `#else` → 0. `probeAvailableRamBytes`: `/proc/meminfo` `MemAvailable:` in kB ×1024, else `sysinfo().freeram`.

- [ ] **Step 4: Run to verify it passes**

```bash
cmake --build build-1.0 --target tinycoder_core tinycoder_test -j 20 || { echo BUILD FAILED; exit 1; }
./build-1.0/unit_tests/tinycoder_test --gtest_filter='ContextPlan*'
```
Expected: 16 tests, all PASS, 0 failed.

- [ ] **Step 5: Leave the tree dirty** and report: `include/ContextPlan.hpp`, `src/cpp/core/ContextPlan.cpp`, `unit_tests/ContextPlanTest.cpp`, `unit_tests/CMakeLists.txt`.

---

### Task 2: `Model::load` honours the request and reports the window

**Files:**
- Modify: `include/ModelConfig.hpp:92` (`maxSeqLen = 0` with a comment "0 = auto (2048 unless memory has less)"), and `maxSeqLen` inside `estimatedMemoryBytes()` (line ~166) to use `(maxSeqLen ? maxSeqLen : 2048)`
- Modify: `include/Model.hpp` (public `bool setContextLength(uint32_t)`, `uint32_t contextLength() const`, `const detail::ContextPlan &contextPlan() const`; members `uint32_t requestedSeqLen_ = 0`, `detail::ContextPlan contextPlan_{}`; include `ContextPlan.hpp`)
- Modify: `src/cpp/core/ModelLoad.cpp` — the only file with the clamps

**Interfaces:**
- Consumes: `tinycoder::detail::planContext`, `ContextRequest`, `probeFreeVramBytes`, `probeAvailableRamBytes` (Task 1); `ModelConfig` fields `numLayers`, `numKVHeads`, `headDim`, `maxSeqLen`.
- Produces: `Model::setContextLength`, `Model::contextLength`, `Model::contextPlan`; `config_.maxSeqLen` = granted window everywhere downstream (host K/V, GPU upload, auto-fit).

- [ ] **Step 1: Write the failing test**

In `unit_tests/ModelTest.cpp`, next to the existing load tests (skip the same way `ModelTest::SetUp` does when no model path is set):
```cpp
TEST_F(ModelTest, ContextLengthRequestIsHonored) {
    tinycoder::Model m;
    ASSERT_TRUE(m.setContextLength(4096));
    ASSERT_TRUE(m.load(SharedTestEnv::modelPath, nullptr));
    EXPECT_EQ(m.contextLength(), 4096u);
    EXPECT_EQ(m.contextPlan().seqLen, 4096u);
    EXPECT_GE(m.contextPlan().maxAllowed, 4096u);
    EXPECT_FALSE(m.setContextLength(8192));  // too late, load already fixed it
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cmake --build build-1.0 --target tinycoder_core tinycoder_test -j 20 || { echo BUILD FAILED; exit 1; }
TINYCODER_MODEL_PATH=/data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf \
  ./build-1.0/unit_tests/tinycoder_test --gtest_filter='ModelTest.ContextLengthRequestIsHonored'
```
Expected: FAIL — no member `setContextLength`.

- [ ] **Step 3: Implement**

`src/cpp/core/ModelLoad.cpp`:
1. At the top of `Model::load`, before anything overwrites `config_`: `requestedSeqLen_ = config_.maxSeqLen ? config_.maxSeqLen : requestedSeqLen_;` and remember it in a local `const uint32_t requested = requestedSeqLen_;`.
2. Delete **both** clamps: the block after the first `config_ = metaLoader->config();` (the `Reducing maxSeqLen from … to 2048` message) and the `Re-apply maxSeqLen cap` block after `config_ = loader.config();`. After the second assignment restore the request (`config_.maxSeqLen = requested;`) and build the `ContextRequest` from the loaded metadata (`fileSize` is the weight-byte estimate; `numAttentionLayers` = `fullAttentionInterval ? ceil(numLayers/interval) : numLayers`, interval from the loader config; `ropeDims` = `mrope ? mropeSectionTotal/2*2 : headDim` — use `headDim` when in doubt), call `planContext`, store it in `contextPlan_`, and set `config_.maxSeqLen = contextPlan_.seqLen`.
3. If `contextPlan_.seqLen == 0`, fail the load: `setError("cannot grant a context window: " + plan.reason)`.
4. Print exactly one line from `plan.reason` in the spec's §8.6 shape, replacing the removed message. No second context line anywhere (`initKVCache`'s `KV cache: …` line is removed in Task 3).
5. `setContextLength(n)` returns false once `loaded_`; otherwise sets `requestedSeqLen_` and `config_.maxSeqLen`.

- [ ] **Step 4: Verify**

```bash
cmake --build build-1.0 --target tinycoder_core tiny_logits_probe tinycoder_test -j 20 || { echo BUILD FAILED; exit 1; }
TINYCODER_MODEL_PATH=/data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf \
  ./build-1.0/unit_tests/tinycoder_test --gtest_filter='ModelTest.ContextLengthRequestIsHonored'
./build-1.0/unit_tests/tiny_logits_probe /data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw "def hello():"
```
Expected: test PASS. The probe prints `[TinyCoder] context window: 2048 tokens (auto; memory allows up to N …)` and **no** `Reducing maxSeqLen` line; logits unchanged: `id=198`, logit in `19.308 ± 2e-3`, top-5 `198,1133,271,460,317`. A missing `context_length` in a checkpoint still yields `4096` and a reason line, not a clamp to 0.

- [ ] **Step 5: Document the knob** in `ENVIRONMENT.md` (the `TINYCODER_*` section): context comes from `Model::setContextLength` / `tc_engine_load(max_seq_len)` / `--ctx`; `0` = auto 2048; the report line prints the memory-derived maximum.

---

### Task 3: Lazy host K/V, attention layers only, one slot map

**Files:**
- Modify: `include/Model.hpp` (add `size_t hostKvSlot(uint32_t layer) const`, `bool ensureHostKv(uint32_t firstCpuLayer)`, members `uint32_t hostKvFirst_ = 0`, `uint32_t hostKvCount_ = 0`, `bool hostKvReady_ = false`)
- Modify: `src/cpp/core/ModelLoad.cpp` (`initKVCache` no longer allocates; it only resets state; keep the SSM/recurrent state allocation there)
- Modify: `src/cpp/core/ModelForward.cpp` (call `ensureHostKv` at the top of the CPU path; replace every `layer * maxSeqLen * nKVHeads * headDim` with `hostKvSlot(layer)`)
- Modify: `src/cpp/core/ModelQwen35.cpp:245,248`, `src/cpp/core/ModelDebug.cpp:231,233,602,604`, `src/cpp/core/ModelForwardDebug.cpp` (the other 16 sites)

**Interfaces:**
- Consumes: `config_.maxSeqLen` (granted), `nGpu`/`firstCpuLayer` from `ModelGPU`.
- Produces: `Model::hostKvSlot(layer)` — index of a layer's slot within the allocated host cache, valid only for `layer >= hostKvFirst_`; the CPU path's only way to address K/V.

- [ ] **Step 1: Record the before state**

```bash
TINYCODER_GPU=0 ./build-1.0/unit_tests/tiny_logits_probe /data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf "def hello():" | tee /tmp/hostkv_before.txt
/usr/bin/time -v ./build-1.0/unit_tests/tiny_logits_probe /data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf "def hello():" 2>&1 | grep -E "Maximum resident|KV cache"
```
Expected: `id=198` and a `KV cache: 28 layers x 2048 seq x 2 heads x 128 dim = 112 MB` line; record both the logits and the RSS.

- [ ] **Step 2: Implement**

`initKVCache`: keep the recurrent/SSM state, drop the `np::Array<float>` K/V allocation and its log line. `ensureHostKv(firstCpuLayer)`: allocate `k`/`v` as `[count][maxSeqLen][numKVHeads][headDim]` and be idempotent, where the slot formula mirrors the GPU exactly (`kvLayerOffset` in `include/GPUCompute.hpp:874`):
```cpp
// slot = layer / interval (interval = 1 for dense archs); recurrent layers
// never address K/V, so a colliding slot number is harmless.
hostKvBaseSlot_ = interval > 1 ? firstCpuLayer / interval : firstCpuLayer;
const uint32_t lastSlot = interval > 1 ? (config_.numLayers + interval - 1) / interval
                                      : config_.numLayers;
hostKvCount_ = lastSlot - hostKvBaseSlot_;
// hostKvSlot(layer) -> (interval > 1 ? layer / interval : layer) - hostKvBaseSlot_
```
Log one line `host K/V: <count> slots x <maxSeqLen> tokens = <N> MB (CPU path only)`. If the GPU session later falls back to the CPU, call `ensureHostKv(0)` again first (freeing and re-allocating for all layers) and log the rebuild — the lazy path must not hand a CPU layer a slot the GPU owns.

- [ ] **Step 3: Verify host K/V is identical and smaller**

```bash
cmake --build build-1.0 --target tinycoder_core tiny_logits_probe tinycoder_test -j 20 || { echo BUILD FAILED; exit 1; }
TINYCODER_GPU=0 ./build-1.0/unit_tests/tiny_logits_probe /data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf "def hello():" | tee /tmp/hostkv_after.txt
diff <(grep -E "id=|logit" /tmp/hostkv_before.txt) <(grep -E "id=|logit" /tmp/hostkv_after.txt)
./build-1.0/unit_tests/tinycoder_test
```
Expected: no diff (identical `id`/logit); the KV line is gone; RSS drops by the unallocated share; `90 tests, 73 passed, 0 failed`.

- [ ] **Step 4: Slot-map parity under partial offload** (Review Focus 4)

```bash
./build-1.0/unit_tests/tiny_logits_probe /data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw "def hello():" | tee /tmp/ngl0.txt
TINYCODER_NGL=2 ./build-1.0/unit_tests/tiny_logits_probe /data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw "def hello():" | tee /tmp/ngl2.txt
diff <(grep -E "id=" /tmp/ngl0.txt) <(grep -E "id=" /tmp/ngl2.txt)
```
Expected: same top-1 id and top-5 order in both (a layer-slot disagreement changes the tokens, which is the point of the check). Logit may differ within the ±2e-3 cublasLt band.

---

### Task 4: Context-full is a diagnosed error, never a silent overwrite

**Files:**
- Modify: `include/Model.hpp` (`bool contextFull_ = false;` + accessor `bool contextFull() const`)
- Modify: `include/GPUCompute.hpp` (`bool contextFull_` on the GPU model, cleared on upload)
- Modify: `src/cpp/core/ModelForward.cpp` (check at the top of `forward`; no CPU fallback when `contextFull_`)
- Modify: `src/cpp/core/ModelGeneration.cpp:299` (stop cleanly at the window)
- Modify: `src/cpp/core/cuda/forward/GPUComputeForward.cu` (`GPUModel::forward` rejects `kvPos_ + seqLen > maxSeqLen` before any kernel)

**Interfaces:**
- Consumes: `config_.maxSeqLen`, `kvCache_.pos`, `kvPos_`.
- Produces: `Model::contextFull()`; a `forward` that returns an empty array (and leaves the cache untouched) when the request does not fit.

- [ ] **Step 1: Write the failing test**

```cpp
TEST_F(ModelTest, ContextFullMidConversation) {
    tinycoder::Model m;
    ASSERT_TRUE(m.setContextLength(1024));
    ASSERT_TRUE(m.load(SharedTestEnv::modelPath, nullptr));
    // Prefill 1000 tokens, then ask for 64 more: 1064 > 1024.
    std::vector<int32_t> first(1000, 1), second(64, 1);
    ASSERT_NE(m.forward(first).size(), 0u);
    const auto out = m.forward(second);
    EXPECT_EQ(out.size(), 0u);
    EXPECT_TRUE(m.contextFull());
    EXPECT_EQ(m.kvCacheSize(), 1000u);   // cache untouched
}
```

- [ ] **Step 2: Run to verify it fails**

Build + filter `ModelTest.ContextFullMidConversation` with `TINYCODER_MODEL_PATH` on the 1.5B.
Expected: FAIL — the second forward returns non-empty and `kvCacheSize()` grew (today it writes past the buffer).

- [ ] **Step 3: Implement**

`Model::forward`: `if (kvCache_.pos + tokens.size() > config_.maxSeqLen) { contextFull_ = true; std::cerr << "[TinyCoder] context full: " << kvCache_.pos << " cached + " << tokens.size() << " new > " << config_.maxSeqLen << "-token window\n"; return {}; }` — before the GPU dispatch and before any host K/V write, and make the existing GPU→CPU fallback skip when `contextFull_`. `GPUModel::forward`: the same check on `kvPos_`, setting the same flag. `generate`: `if (model.kvCacheSize() + 1 > maxSeqLen) break;` before the decode step, and stop normally (no error) — a full window ends generation the way `n_ctx` does.

- [ ] **Step 4: Verify**

```bash
cmake --build build-1.0 --target tinycoder_core tinycoder_test -j 20 || { echo BUILD FAILED; exit 1; }
TINYCODER_MODEL_PATH=/data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf \
  ./build-1.0/unit_tests/tinycoder_test --gtest_filter='ModelTest.ContextFull*'
```
Expected: PASS, the stderr line printed once, cache position unchanged. Then the adversarial case that produced silent corruption: `tiny_logits_probe` with `--ctx 2048` (arrives in Task 5) and a 3000-token prompt must print the context-full line and exit non-zero instead of writing 4 MB past the cache.

---

### Task 5: `--ctx` on the bench, `tools/ctx_probe`, and the gate script

**Files:**
- Modify: `benchmarks/BenchMain.cpp` (`--ctx N`, default 0; call `model.setContextLength(opt.ctx)` before `load`)
- Create: `tools/ctx_probe.cpp`
- Modify: `unit_tests/CMakeLists.txt` (add the `ctx_probe` executable next to `tiny_logits_probe`, linked against `tinycoder_core`)
- Create: `scripts/context_gate.sh`

**Interfaces:**
- Consumes: `setContextLength`, `contextPlan`, `generate`.
- Produces: `./build-1.0/tools/ctx_probe <model> --ctx N --needle-tokens T [--gen G]` — exit 0 iff the generated text contains `pineapple-7731`; prints `granted=… maxAllowed=… kv=…`.

- [ ] **Step 1: Implement `tools/ctx_probe.cpp`**

Prompt: first line `The access code is pineapple-7731.`, then filler lines `Record 000123: routine maintenance log entry.` until the token count is within 48 tokens of `T`, then `Question: what is the access code? Answer with the code only.`. Print `granted=<plan.seqLen> maxAllowed=<plan.maxAllowed> kv=<f32|int8>`, generate greedily `G` tokens (default 16), print the text, and return 0 only if it contains `pineapple-7731`.

- [ ] **Step 2: Verify the probing works at a short window**

```bash
cmake --build build-1.0 --target tinycoder_core tiny_logits_probe ctx_probe tinycoder_bench -j 20 || { echo BUILD FAILED; exit 1; }
./build-1.0/tools/ctx_probe /data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf --ctx 4096 --needle-tokens 3500
```
Expected: `granted=4096 maxAllowed=4096 kv=f32`, the answer contains `pineapple-7731`, exit 0.

- [ ] **Step 3: Write `scripts/context_gate.sh`**

Runs `ctx_probe` at 8192/32768/65536 tokens on Ornith-9B (f32) and Ternary-Bonsai (int8 where f32 does not fit), any failure exits non-zero, and it prints the granted window and `maxAllowed` for each run. `set -uo pipefail` and a `MODELS=(path|ctx|label)` table, mirroring `scripts/text_gate.sh`'s shape. A SKIP is a failure (same rule as the text gate).

- [ ] **Step 4: Verify**

```bash
./scripts/context_gate.sh 2>&1 | tail -20
```
Expected: at this stage the 8192 rung passes and 32768/65536 may still be limited by the sizing in Task 6 — record which and why; the script must exit non-zero, not lie.

---

### Task 6: Device sizing and split-KV chunks for long windows

**Files:**
- Modify: `src/cpp/core/cuda/forward/GPUComputeUpload.cu` (RoPE/MRoPE tables, `attnChunks_`/`attnChunk_`, int8 pool, logits scratch — all already `maxSeqLen`-driven, verify each)
- Modify: `include/GPUCompute.hpp` (`attnChunk_` floor, `maxAttnChunks_ = 512` invariant)
- Modify: `src/cpp/core/ModelGPU.cpp` (charge K/V per layer at `plan.kv`/`plan.kvMirror`, pass `geom.int8Kv`)

**Interfaces:**
- Consumes: `contextPlan()` (Task 2), `maxSeqLen` = granted window.
- Produces: `attnChunks_ = ceil(maxSeqLen/attnChunk_) <= 512` as a hard invariant; a GPU session that allocates exactly the granted window.

- [ ] **Step 1: Add the invariant check**

In `GPUComputeUpload.cu`, after computing `attnChunk_`: raise it to `((maxSeqLen + 511) / 512 + 31) / 32 * 32` when that is larger, and log one line `split-KV: chunk=<n> chunks=<n>` at 32768+ tokens only. Assert `attnChunks_ <= 512`.

- [ ] **Step 2: Pinned test — a split prefill agrees with a single prefill**

Add to `unit_tests/ModelTest.cpp` (this is spec §10 item 5, the check that the cache-carrying path and the fresh-cache path agree at a long window):

```cpp
TEST_F(ModelTest, SplitPrefillMatchesSinglePrefill) {
    tinycoder::Model m;
    ASSERT_TRUE(m.setContextLength(8192));
    ASSERT_TRUE(m.load(SharedTestEnv::modelPath, nullptr));
    std::vector<int32_t> a(3000, 100), b(3000, 101), all = a;
    all.insert(all.end(), b.begin(), b.end());
    const auto one = m.forward(all);                 // 6000 tokens in one call
    m.clearKVCache();
    m.forward(a);                                    // 3000 ...
    const auto two = m.forward(b);                   // ... then 3000 more
    const uint32_t V = SharedTestEnv::config.vocabSize;
    ASSERT_EQ(one.size(), all.size() * V);
    ASSERT_EQ(two.size(), b.size() * V);
    float maxDiff = 0.0f;
    for (uint32_t i = 0; i < V; ++i) {
        maxDiff = std::max(maxDiff, std::fabs(one.get(one.size() - V + i) -
                                              two.get(two.size() - V + i)));
    }
    EXPECT_LT(maxDiff, 1e-3f) << "split prefill diverged from a single prefill";
}
```

Run it on the 1.5B (CPU and GPU). Expected: `maxDiff < 1e-3`. If it is larger, record the measured number in the ledger and investigate the cache path before widening the bound — the bound is the requirement, not a knob.

- [ ] **Step 3: Verify a 64k f32 window on the 9B** (Review Focus 2)

```bash
cmake --build build-1.0 --target tinycoder_core ctx_probe -j 20 || { echo BUILD FAILED; exit 1; }
./build-1.0/tools/ctx_probe /data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf --ctx 65536 --needle-tokens 65000
```
Expected: `granted=65536` (or a smaller granted window with a reason naming VRAM), the answer contains the code, `split-KV` invariant holds, and no CUDA error. If 64k f32 does not fit, record the actual `maxAllowed` — Task 7 exists to raise it.

- [ ] **Step 4: Verify a request above every limit is reduced, not fatal**

```bash
./build-1.0/tools/ctx_probe /data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf --ctx 1048576 --needle-tokens 60000
```
Expected: `granted=<maxAllowed> maxAllowed=<same>`, a reason line naming the resource, exit 0 with the right answer — never a crash, never an OOM, never a silent CPU run.

---

### Task 7: int8 K/V as the store (arch 1)

**Files:**
- Modify: `include/GPUComputeInternal.hpp` (two new kernels: `kQ35WarpAttention8Prefill`, `kQ35PrefillAttnSharedKV8`)
- Modify: `src/cpp/core/cuda/forward/GPUComputeMoE.cu` (`forwardQwen35Prefix`: int8 store on prefill, int8 prefill attention dispatch, f32 K/V not allocated in int8-store mode)
- Modify: `src/cpp/core/cuda/forward/GPUComputeUpload.cu` (skip the f32 K/V allocation when `kvStoreI8_`; size the int8 pool `maxSeqLen + 16`)
- Modify: `include/GPUCompute.hpp` (`bool kvStoreI8_`)

**Interfaces:**
- Consumes: `plan.kv`, `plan.kvMirror`, `maxSeqLen`.
- Produces: `kvStoreI8_` sessions where prefill writes int8 K/V with `kQ35QuantKV8` (`dstTokenBase = pos`, `srcTokenBase = 0`, one block per (token, kv-head), blockDim 128) and both prefill variants read it dequantized as `int8 × scale[token][kvHead]`; decode keeps the existing `kQ35WarpAttention8*` kernels and `kv8Rows_[slot] == pos` after a prefill.

- [ ] **Step 1: Write the failing check first** (there is no unit-test entry point for these kernels; the probe is the test)

```bash
./build-1.0/tools/ctx_probe /data/models/bonsai/Ternary-Bonsai-2-27B-PQ2_0.gguf --ctx 65536 --needle-tokens 64000
```
Expected before the change: `granted` well below 65536 (f32 does not fit), reason names VRAM. That is the red.

- [ ] **Step 2: Implement** the two kernels with these signatures (mirroring the f32 kernels at
  `include/GPUComputeInternal.hpp:15511` and `:15581`; the only addition is the scale arrays and int8 K/V):

```cpp
template <uint32_t HD, uint32_t W>
__global__ void kQ35WarpAttention8Prefill(const float *q, const int8_t *k8, const int8_t *v8,
                                          const float *ks, const float *vs, float *out,
                                          uint32_t seqLen, uint32_t nHeads, uint32_t nKVHeads,
                                          uint32_t cachePos, const uint32_t *dpos, float invSqrt);
template <uint32_t HD, uint32_t W>
__global__ void kQ35PrefillAttnSharedKV8(const float *q, const int8_t *kCache8,
                                         const int8_t *vCache8, const float *ks, const float *vs,
                                         float *out, uint32_t seqLen, uint32_t nHeads,
                                         uint32_t nKVHeads, uint32_t cachePos,
                                         const uint32_t *dpos, float invSqrt);
```

Bodies: copy the f32 kernels and replace every `kCache[...]` load with `float(kCache8[...]) * ks[token*nKVHeads + kvHead]` (likewise `v` with `vs`), keeping `csEnd = cachePos + s`, the `static_assert(HD == 256)`, and the shared-KV variant's per-row staging so the 8× traffic saving survives. Then wire the store — `kQ35QuantKV8(tokens=seqLen, nKV, hd, dstTokenBase=pos, srcTokenBase=0)` writing straight from the pre-attention K/V scratch, one block per (token, kv-head), blockDim 128 — dispatch the `*8Prefill` variants when `kvStoreI8_`, and skip the `kvK_`/`kvV_` allocation entirely in that mode.

- [ ] **Step 3: Verify numerics, then the window**

```bash
cmake --build build-1.0 --target tinycoder_core ctx_probe tiny_logits_probe tinycoder_bench exl3_gpu_parity -j 20 || { echo BUILD FAILED; exit 1; }
./build-1.0/unit_tests/tiny_logits_probe /data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw "def hello():"   # default path: unchanged, ±2e-3
./build-1.0/unit_tests/exl3_gpu_parity 2>&1 | grep -c '\[FAIL\]'   # must print 29
./scripts/text_gate.sh                                            # exit 0
./build-1.0/tools/ctx_probe /data/models/bonsai/Ternary-Bonsai-2-27B-PQ2_0.gguf --ctx 65536 --needle-tokens 64000
./build-1.0/tools/ctx_probe /data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf --ctx 262144 --needle-tokens 261000
```
Expected: default logits in band, 29 FAILs, text gate 0, Bonsai `granted=65536 kv=int8` with the right answer, 9B `granted=262144 kv=int8` (or the measured `maxAllowed`, reported). Record the int8-vs-f32 logit delta at 2048 in the ledger; a top-1 change there is a bug, not noise.

---

### Task 8: int8 decode for qwen35moe (arch 2) — droppable

**Files:**
- Modify: `src/cpp/core/cuda/forward/GPUComputeForward.cu` (`forwardQwen35MoePrefix`: port the int8 decode block from `forwardQwen35Prefix`)
- Modify: `src/cpp/core/ContextPlan.cpp` / `ModelLoad.cpp` (`int8KvSupported` true for arch 2 only once the port is verified)

**Interfaces:**
- Consumes: the arch-1 block as the template.
- Produces: arch-2 sessions can select `KvPrecision::I8`.

- [ ] **Step 1:** Run the 35B-A3B at 2048 with the port absent and record top-1/logit + tg.
- [ ] **Step 2:** Port the block; verify top-1 and top-5 unchanged and the logit within the int8 arm's stated delta, then flip `int8KvSupported` for arch 2.
- [ ] **Step 3:** If the port cannot be verified in this session (model not on the box, load too slow), **leave `int8KvSupported == false` for arch 2** and record why. Do not grant a window the kernels cannot serve — arch 1 covers every acceptance model.

---

### Task 9: Whole-branch verification and the numbers

**Files:**
- Create/extend: `.superpowers/sdd/2026-11-19-context-window/progress.md` (what was measured, what was rejected, why)
- Modify: `AGENTS.md` only if the test count changed (it will: Task 1 adds 16 tests) — `90 tests, 73 passed` → the measured line.

- [ ] **Step 1: Run the full gate list in order**

```bash
cmake --build build-1.0 --target tinycoder_core tinycoder_test tiny_logits_probe ctx_probe tinycoder_bench exl3_gpu_parity -j 20 || { echo BUILD FAILED; exit 1; }
./build-1.0/unit_tests/exl3_gpu_parity 2>&1 | grep -c '\[FAIL\]'                       # 29
./build-1.0/unit_tests/tiny_logits_probe /data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw "def hello():"   # id=198, ±2e-3
./build-1.0/unit_tests/tinycoder_test                                                   # 0 failed
./scripts/text_gate.sh                                                                  # exit 0
./scripts/context_gate.sh                                                               # exit 0
```
Expected: every line as annotated. Any deviation is a task to do, not a note to write.

- [ ] **Step 2: Confirm the default did not regress**

```bash
./build-1.0/benchmarks/tinycoder_bench --model /data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw --n-prompts 512 --n-gen 128 --reps 1 --threads 36
```
Expected: `pp512`/`tg128` within run-to-run spread of the pre-change numbers (window still 2048, precision still f32+mirror). A regression means the plan is charging or allocating something it should not — fix before recording anything.

- [ ] **Step 3: Record the granted windows and throughput at 8k/64k** for Ornith-9B and Bonsai in the ledger, with the protocol line (`--n-gen 128`, threads 36, one process at a time) and `maxAllowed` beside each. Do not touch `BENCHMARKS.md`: its rows are the 2048 default and that default has not changed.

- [ ] **Step 4: Report** the changed files, the measured windows, and the two honest limitations: 256k needs ≥16 GiB of VRAM for these 27B-class checkpoints, and 1M needs RoPE scaling this engine does not have.
