# Context window limited by VRAM/RAM, not by a constant

Date: 2026-11-19
Status: approved approach (A+B); awaiting design sign-off

## 1. Intent

The engine ignores the caller's context request and caps every model at 2048
tokens. A model that advertises 262144 positions (`Qwen3.8-27B` EXL3,
`Ternary-Bonsai-2-27B`) gets a 2048-token window, and nothing the caller passes
changes it — so an editor/server built on this engine reports `2048` and
truncates its prompts to it. The goal: **the context window is a function of the
model's trained limit and the memory actually available (VRAM, then RAM), and
nothing else.** A caller that asks for 65536 gets 65536 when the machine can
hold it, and a precise, non-misleading explanation when it cannot.

## 2. What is broken today (evidence)

1. `Model::load` copies `loader.config()` over the caller's `ModelConfig`, so a
   `maxSeqLen` set before `load()` (which is exactly what
   `tinycoder_shim.cpp:tc_engine_load` does) is discarded
   ([ModelLoad.cpp:400](../../../src/cpp/core/ModelLoad.cpp)).
2. Two hard clamps then force 2048
   ([ModelLoad.cpp:348-355](../../../src/cpp/core/ModelLoad.cpp) and
   [ModelLoad.cpp:402-405](../../../src/cpp/core/ModelLoad.cpp)), printing
   `Reducing maxSeqLen from 262144 to 2048 to fit available memory`.
   Measured on this box: 27B EXL3 prints `262144 → 2048`; Qwen2.5-1.5B prints
   `32768 → 2048`.
3. `ModelConfig::maxSeqLen` defaults to 2048
   ([ModelConfig.hpp:92](../../../include/ModelConfig.hpp)).
4. There is **no guard** on the KV position: `kvPos_` / `kvCache_.pos` advance
   past a KV array sized `maxSeqLen`. Measured: a 3000-token prefill on the
   1.5B ran at 4490 pp tok/s while writing ~4 MB past `kvK_` — silent
   corruption, not an error.
5. The host KV cache is allocated for **all** layers at load, even when the GPU
   owns the session: 1024 MB at 2048 for the 27B, of which only 16 of 64 layers
   use K/V (4× waste), and it would be 32 GiB at 65536 for a cache the GPU never
   reads.
6. Two context-linear device tables are sized from `maxSeqLen` and are fine, but
   one is not: the split-KV partials are sized `nHeads × ceil(maxSeqLen/16)` —
   201 MB at 262144 for 24 heads, 805 MB for Bonsai's 96 heads, and 3.2 GB at
   1M.

## 3. Goals

- `maxSeqLen` requested by the caller is honored, up to the model's advertised
  limit, and only reduced when memory genuinely cannot hold it.
- The granted window is derived from free VRAM (GPU path) or available RAM
  (CPU path), with the K/V bytes/token actually charged for the model's
  geometry and the chosen K/V precision.
- When the request does not fit at f32 K/V, the engine escalates to int8 K/V
  before reducing the window, and reports which step it took.
- A request that cannot be satisfied is reported in one honest line naming the
  limiting resource; the misreporting `Reducing … to 2048` line is removed.
- Reaching the window boundary stops generation cleanly and rejects an
  oversized prefill; no write past the end of any KV buffer.
- int8 K/V is a real KV store (prefill and decode), not a decode-only mirror, so
  it halves/quarters the residency that the window is budgeted against.
- Measured acceptance, on this 2080 Ti (11.2 GiB) with the checkpoint's own
  advertised limit (262144), each run **asking** for its window (auto stays
  2048 by rule 4, so the gate passes `--ctx`):
  - **Ornith-1.5-9B-Q4_K_M** (arch 1, 32 layers, 8 attention layers):
    **65536 at f32** (mirror off, 4 GiB K/V; 5.5 GB weights → ~10.5 GiB) and
    **262144 at int8** (4.25 GiB K/V → ~10.7 GiB).
  - **Ternary-Bonsai-2-27B-PQ2_0** (arch 1, 64 layers, 16 attention layers):
    **65536 at int8** (2.1 GiB K/V; measured 2882 MiB free at 2048 makes this
    ~1.4 GiB of headroom), and as far above 64k as the card allows (~90k).
    256k is **not** reachable on 11 GiB for this checkpoint: 8.6 GiB K/V +
    6.6 GB weights + 0.9 GiB = 16.3 GiB. It needs a ≥16 GiB card; on a 24 GiB
    card it fits with 8 GiB to spare.
  - Qwen3.8-27B EXL3 (arch 1, 8.4 GB trellis): 65536 at int8 is marginal
    (8.4 + 2.1 + 0.9 = 11.4 GiB); expect ~49152 there, and say so rather than
    pretending.
  All must also answer a needle question at the granted window.

  **Measured, and these projections were optimistic by 20-80%:** the planner's
  fit is not the K/V arithmetic above, because the session's fixed cost is
  1,408 MiB (measured: 900 MiB of over-grant put a 60,672-token 9B session into
  OOM), the KV slots are one per full-attention layer and the mirror is charged
  per slot. What the card actually grants, each confirmed by a load with the
  window requested:
  - Ornith-1.5-9B `q4_k_m`: **52,480** f32 (not 65,536), 41,984 f32+mirror,
    **196,352** int8 (not 262,144).
  - Ternary-Bonsai-2-27B `pq2_0`: **16,640** f32, **64,256** int8 (not ~90k).
  - Qwen3.8-27B EXL3 `1.60bpw`: **2,560** f32+mirror, **12,288** int8 (not
    ~49k -- the trellis weights plus the fixed cost leave less than the
    projection assumed).
  The measured table, the throughput at these windows and what 256k/1M would
  need are in `BENCHMARKS.md` under "Context window"; the fit ladder and its
  rungs are in `ContextPlan.cpp`.

## 4. Non-goals

- Multi-GPU / tensor-parallel sharding. The engine is single-device; "1M on four
  3090s" is a different project.
- RoPE scaling (YaRN / NTK) beyond the model's trained positions. The window is
  clamped to the advertised maximum; `1M` on a 262144-trained model is not a
  memory question and is out of scope.
- q4 K/V. int8 is the precision step for this change; q4 needs kernel work with
  no gate yet.
- Reworking the CPU K/V layout for the recurrent layers of the *CPU* path
  beyond the slot map described below.
- Changing `InferenceParams::maxTokens` (a generation cap, unrelated).

## 5. Constraints

- 2080 Ti, 11 GB, sm_75; 96 GB RAM available; no inference libraries.
- One session on the GPU, one editor per file. Build with
  `--target tinycoder_core`, then the tools, and check the exit status.
- The logits gate (top-1 id 198, top-5 198/1133/271/460/317, logit
  `19.308 ± 2e-3`) and `scripts/text_gate.sh` must stay green. Auto mode must
  therefore keep f32 K/V whenever it fits — short-context numerics must not
  change.
- Everything published needs a runnable script in the repo.

## 6. Memory model

K/V bytes per token (K and V) for a 27B/Bonsai-class hybrid — 64 layers, 16
full-attention layers, 4 K/V heads × 256:

| mode | bytes/token/layer | ×16 layers | 8k | 64k | 256k |
|---|---|---|---|---|---|
| f32 store + int8 decode mirror (today's arch-1 default) | 8192 + 2128 | 161 KiB | 1.26 GiB | 10 GiB | 40 GiB |
| f32 store, mirror off | 8192 | 128 KiB | 1 GiB | 8 GiB | 32 GiB |
| int8 store (mirror == store) | 2096 | 34.2 KiB | 267 MiB | 2.1 GiB | 8.6 GiB |

The mirror line is not a technicality: the mirror is what `kv8_` allocates today
in addition to f32, and it is charged as such (see 8.2), so a f32 session's
window is 21% smaller than a naive f32-only count would predict.

RoPE/MRoPE tables add 1280 B/token (f32; 335 MB at 256k). Split-KV partials are
bounded by the chunk cap (8.4): ≤ 64 MB even at 256k with 96 q-heads, so they
live in the fixed overhead rather than the closed form. Weights on device:
6.6 GB (Bonsai PQ2_0), 8.4 GB (27B EXL3), 5.5 GB (Ornith-9B Q4_K_M).

Ceilings on the 11.2 GiB card (weights + K/V + 0.9 GiB fixed overhead):

| model | int8 window | f32 window | 256k possible? |
|---|---|---|---|
| Bonsai PQ2_0 (6.6 GB) | ~64k, up to ~90k | ~24k | no — 8.6 GiB K/V puts it at 16.3 GiB |
| Ornith-9B Q4 (5.5 GB) | 262144 (4.25 GiB) | ~64k (mirror off) | yes, int8 |
| 27B EXL3 (8.4 GB) | ~49k | ~16k | no |
| any model, 24 GB card | 256k (Bonsai: 8.6 + 6.6 + 0.9 = 16.1 GiB) | — | yes |

Measured baselines today (11.2 GiB card, all at the clamped 2048 window):
Bonsai PQ2_0 upload ends at 7938 MiB used / 2882 MiB free, tg2 51.9 tok/s;
Ornith-9B fully offloads, pp512 2573 tok/s, tg8 83.6 tok/s; 27B EXL3 does not
fully offload. These are the "before" numbers for the window work.

## 7. Approaches considered

- **A — remove the cap, budget f32 K/V.** Small. Honors the request and sizes
  it from memory, but f32 K/V costs 128 KiB/token, so the 11 GB card tops out
  around 16k and a 24 GB card around 128k. Misses the 64k requirement.
- **B — A + int8 K/V as the store** (prefill and decode), attention-layers-only
  host K/V, split-KV chunk scaling, context guard. Reaches 64k on 11 GB and
  256k on 24 GB. Larger, touches kernels, needs a numerics gate.
- **C — B + q4 K/V and/or host-RAM K/V spill.** Reaches 256k on 11 GB / 1M on
  RAM, but decode over PCIe-resident K/V, or new q4 kernels with no gate. Out
  of scope.

Decision: **A+B**. C stays documented as the next step if the user ever needs a
window on hardware that B cannot reach.

## 8. Design

### 8.1 Resolution of the window (pure, unit-testable)

New `include/ContextPlan.hpp` (+ `.cpp`), `namespace tinycoder::detail`, no CUDA
and no model dependency, so it is testable without a checkpoint:

```cpp
enum class KvPrecision { F32, I8 };
struct ContextRequest {
    uint32_t requested;      // 0 = auto
    uint32_t modelMax;       // GGUF *.context_length / EXL3 max_position_embeddings
    uint32_t numLayers, numAttentionLayers, numKVHeads, headDim;
    uint32_t ropeDims;       // headDim + mrope dims (the cos/sin row width)
    bool int8KvSupported;    // qwen35/qwen35moe GPU path only (see 8.2)
    uint64_t weightBytes;    // device-resident weight form (trellis / quantized file size)
    uint64_t freeVram;       // 0 = no usable GPU
    uint64_t availableRam;   // MemAvailable
    uint64_t gpuFixedOverhead = 900ull << 20;
};
struct ContextPlan {
    uint32_t seqLen;         // the granted window
    uint32_t maxAllowed;     // the largest window memory permits (for reporting)
    KvPrecision kv;          // F32 or I8 store
    bool kvMirror;           // F32 store + int8 decode mirror (arch-1 default)
    bool kvOnGpu;
    uint64_t kvBytes;
    std::string reason;      // one line, names the limiting resource
};
ContextPlan planContext(const ContextRequest &);
uint64_t kvBytesPerToken(const ContextRequest &r, KvPrecision, bool mirror); // nAttn*(K+V+scales) + 4*ropeDims
uint64_t probeFreeVramBytes();      // 0 when no usable GPU
uint64_t probeAvailableRamBytes();
```

Rules, in order:

1. Charge bytes per token: `kvBytesPerToken()` = per-attention-layer K/V at the
   chosen precision (plus the mirror when the store is f32 and the mirror is on)
   + the RoPE/MRoPE tables. Linear in the window, so every fit is closed-form.
2. `maxAllowed` = the largest window the memory budget permits at the best
   precision available (the ladder in rule 3), rounded down to a multiple of
   256, clamped to `[1024, modelMax]`; 0 means "not even 1024 fits".
3. GPU budget = `freeVram − gpuFixedOverhead − 256 MB` safety margin. Try, in
   order: f32 store + int8 mirror with the full model resident; f32 store with
   the mirror off; int8 store with the full model; int8 store with the minimum
   residency (the largest single layer). The best rung that fits 1024 tokens
   defines the precision; `maxAllowed` is then that rung's fit. int8 enters the
   ladder only when `int8KvSupported`; otherwise the last rungs do not exist.
4. `want = requested ? min(requested, modelMax) : min(2048, maxAllowed)`.
   **Auto is 2048 on purpose**: it is today's default, so every published
   benchmark stays comparable and no caller silently trades decode speed for a
   window nobody asked for. The report line always prints `maxAllowed`, so the
   memory-derived ceiling is visible and one `--ctx` away. A request is honored
   up to `maxAllowed`; only memory shrinks it, and only the model's trained
   limit clamps it (`reason` says which).
5. **Ownership: the plan owns the window; the auto-fit owns the layer count.**
   `ContextPlan` decides `seqLen`/precision/`kvOnGpu` and is final before any
   device allocation. `ModelGPU.cpp`'s auto-fit consumes the resulting K/V bytes
   (charged per layer, as it already does) and reports `nGpu`. "Minimum
   residency" is exactly one layer's weights: an explicit long request may leave
   63 of 64 layers on the CPU, and it is honored rather than refused, because
   the request is the user's and the cost is printed.
6. **Never switch from GPU to CPU because of context size.** A GPU run that
   cannot hold the request reports the reduced window instead; a 10× slower CPU
   fallback triggered by a context request would be a surprise. CPU-only is
   entered explicitly (`TINYCODER_GPU=0`) or when the GPU cannot host the model
   at all (existing auto-fit behaviour). A K/V allocation failure despite the
   plan is an error for that GPU session, not a silent CPU run: it is reported
   with the numbers (free MiB, requested bytes) and the user picks a smaller
   `--ctx`.
7. RAM path (CPU-only build, or no GPU): budget = `MemAvailable − hostWeights −
   slack`; the host cache is f32 only, so the ladder is f32 with the mirror off.
8. Floor 1024, ceiling `modelMax`. Below 1024 nothing is granted: the load
   fails with a clear message instead of running a window it cannot honor.

Auto reserves a further 256 MB safety margin on top of `gpuFixedOverhead`. The
plan is **immutable once device allocation starts**: there is no halve-and-retry
inside `upload()` (it would desync RoPE tables, the int8 pool, `attnChunk_` and
the host slot map, which are sized from the window in different files). If a K/V
allocation fails anyway, see rule 5.

`Model::load` keeps the caller's pre-set `config_.maxSeqLen` as the request
across the `loader.config()` overwrite, resolves the plan after metadata is
available, and stores the granted window in `config_.maxSeqLen`. The
`tc_engine_load` shim therefore needs no change.

New public surface: `bool Model::setContextLength(uint32_t)`, valid before
`load()` (returns false after), and `Model::contextPlan()` returning the granted
plan for reporting. `ModelConfig::maxSeqLen` defaults to `0` (auto).

### 8.2 int8 K/V as the store (qwen35, qwen35moe)

Today, and only for the **dense qwen35 (arch 1)** path that lives in
`GPUComputeMoE.cu::forwardQwen35Prefix` (the file names are crossed — the MoE
forward is in `GPUComputeForward.cu::forwardQwen35MoePrefix`): prefill always
writes f32 (`kQ35StoreKV`), and decode rebuilds an int8 mirror from f32
(`kQ35QuantKV8`) and attends over it (`kQ35WarpAttention8`,
`kQ35WarpAttention8Split`). The MoE forward has **no** int8 decode path at all,
and the upload enables `kv8_` only for `architecture == 1`. The f32 array is the
real store, and the int8 pool adds a further 27% of that residency
(34.2 KiB/token on top of 128) while saving nothing.

Two distinct modes must stay distinct:

- `kv8_` (existing): f32 store + int8 **decode mirror**, default ON for arch 1.
  This is today's behaviour at short context and must not change. Its bytes
  (2128 B/token/layer: 2096 K/V + 32 scale) are **charged on top of** the f32
  store in `kvBytes()`, because the mirror really is allocated. That is the
  `161 KiB/token` row of §6, and it is why the first rung of the ladder (8.1) is
  "f32 + mirror" rather than "f32".
- `kvStoreI8_` (new): int8 is the **store**; prefill writes int8, prefill
  attention reads int8, no f32 array is allocated, and the mirror is the store
  (so `kvBytes` = `34.2 KiB/token`, not `128 + 34`). Only ever selected by the
  plan for `ARCH_QWEN35`/`ARCH_QWEN35MOE`, and it implies int8 decode.
- The middle rung, f32 store with the **mirror off**, is a supported mode with
  no new kernels: `kv8Decode` is already false when `kv8_` is off, and decode
  then uses the f32 split/eager attention path that prefill already uses.

Changes for `kvStoreI8_`:

- Prefill stores K/V straight into the int8 pool with `kQ35QuantKV8`
  (`tokens = seqLen`, `dstTokenBase = pos`, `srcTokenBase = 0` — the kernel
  already has the bases needed, one block per (token, kv-head), blockDim 128 =
  its shared-array width), and the f32 `kvK_`/`kvV_` arrays are **not**
  allocated (their bytes go to the window budget instead).
- Prefill attention reads int8: new `kQ35WarpAttention8Prefill` and
  `kQ35PrefillAttnSharedKV8` variants that dequantize on load
  (`int8 × ks[token][kvHead]`), same causal range `csEnd = cachePos + s`, same
  per-head math. The shared-KV variant converts once per staged row, keeping its
  8× traffic saving; its `static_assert(HD == 256)` carries over.
- Decode keeps the existing int8 kernels; `kv8Rows_[slot]` becomes exactly `pos`
  after a prefill (no rebuild step).
- The int8 pool is sized `maxSeqLen + 16` (the existing +16 slack stays).
- Arch 2 (qwen35moe) needs the int8 decode block ported from
  `forwardQwen35Prefix` first (it is a copy of that full-attention block);
  until that lands, `int8KvSupported` is false for arch 2 and the planner
  budgets f32 for it. Do not grant an int8-sized window the kernels cannot
  serve. **Both acceptance models in §3 are arch 1 (Bonsai PQ2_0 is
  `general.architecture = qwen35`, Ornith-9B likewise), so the 64k/256k targets
  do not depend on this port**; the port is what gives a qwen35moe checkpoint
  (35B-A3B) an int8 window too.
- `TINYCODER_KV8=1` forces int8, `=0` forces f32 (existing knob, semantics
  extended to prefill); absent = the plan decides (f32 when it fits).

qwen2 dense keeps f32 (or `TINYCODER_KV16`) K/V: its prefill attention is a
materialized fp16 GEMM path (`kAttnGemmMaxSeq` = 512), and its K/V is small
(2 K/V heads). The planner still budgets its window from memory, just never with
int8.

### 8.3 Host K/V cache

- Allocate **lazily** on the first CPU-path forward, not at load.
- Allocate slots only for the layers the CPU will run: layers
  `[firstCpuLayer, numLayers)`; and, for qwen35 hybrids, only the
  full-attention layers among them. `firstCpuLayer` = `nGpu` when the GPU owns a
  prefix, else 0.
- One private helper, `hostKvSlot(layer)`, replaces every
  `layer * maxSeqLen * nKVHeads * headDim` index in `ModelForward.cpp`,
  `ModelQwen35.cpp`, and the debug duplicates (`ModelDebug.cpp`,
  `ModelForwardDebug.cpp`), so allocation and use cannot drift.
- If the GPU session later falls back to the CPU (a rare failure path), the host
  cache is rebuilt for all layers first, and the rebuild is logged.
- For a **CPU-only** run, after the weights are loaded, re-check the RAM plan
  against measured `MemAvailable` and reduce the window (re-printing the same
  line) if the estimate was optimistic. This re-check is only legal when no
  device K/V was sized from the window (nothing device-side is context-linear
  except the K/V, RoPE tables and partials); with any device residency the plan
  is final (§8.1 rule 3/6).

### 8.4 Split-KV chunks

`attnChunks_ = ceil(maxSeqLen / attnChunk_) ≤ 512` is a **hard invariant**, not a
floor: it bounds the partial buffers, keeps `grid.y` sane, and lets the planner
treat partials as a constant. `attnChunk_` keeps its current default as a
minimum and is raised to satisfy the invariant. At 65536 that is chunk 128
(512 chunks, 12.5 MB of partials for 24 heads, 50 MB for Bonsai's 96); at 262144
chunk 512 (same bytes). `attnPartialM/L` are `nHeads × chunks` floats (≤ 200 KB).
All of it sits inside `gpuFixedOverhead`; `ContextPlan` needs no chunk field,
only the invariant.

### 8.5 Context-full behaviour

- GPU forward entry: `if (kvPos_ + seqLen > maxSeqLen)` → set an explicit
  `contextFull_` flag, return false with a message. `Model::forward` checks the
  flag **before** its CPU fallback and returns an empty array with a loud line
  (`prompt N tokens + cache P exceeds the M-token window`), instead of falling
  back to a CPU path that would corrupt its own cache.
- CPU forward: the same check against `kvCache_.pos`.
- `generate`: before each decode step, stop when `kvCacheSize() + 1 >
  maxSeqLen` (a full window ends generation normally, like `n_ctx` in
  llama.cpp); never calls `forward` with a token it cannot place.
- Nothing writes past a KV buffer; the 3000-token-on-2048 corruption becomes a
  diagnosed error.

### 8.6 Reporting

One line, printed once at load, replacing the removed clamp message. Three
shapes, all from `reason`, so the message and the decision cannot diverge:

```
# nobody asked (auto): say what memory would have allowed
[TinyCoder] context window: 2048 tokens (auto; memory allows up to 56832 at
            f32 K/V + mirror — request more with --ctx or the model's context_length)

# asked and granted
[TinyCoder] context window: 65536 tokens (requested 65536; K/V int8 2.11 GiB,
            34.6 KiB/token; model max 262144)

# asked, memory said less
[TinyCoder] context window: 49152 tokens (requested 65536, limited by VRAM:
            2882 MiB free after 6.6 GiB weights; K/V int8 34.6 KiB/token)
```

`maxAllowed` is printed in every case, including the granted one when it is
larger than `seqLen` (it is equal then). `tc_engine_info` reports the granted
window, and the tool/app that asked can now tell "granted what I asked" from
"granted less".

### 8.7 Bench/probe surface

`tinycoder_bench` gains `--ctx N` (0 = auto) so long-window numbers can be
taken with the standard protocol. A new `tools/ctx_probe.cpp` loads a checkpoint
at a requested window, builds a needle-in-the-tail prompt of N tokens with the
model's own tokenizer, generates, and prints the granted window, the answer and
the VRAM/RAM used — the runnable method the long-window gate and any published
number are based on. No new environment variables in the library.

## 9. Touch points

| file | change |
|---|---|
| `include/ModelConfig.hpp` | `maxSeqLen = 0` (auto); docs |
| `include/ContextPlan.hpp`, `src/cpp/core/ContextPlan.cpp` | new planner + memory probes |
| `src/cpp/core/ModelLoad.cpp` | keep the request across `loader.config()`; resolve plan; remove both clamps; new report line; lazy host K/V; `setContextLength`/`contextPlan` |
| `include/Model.hpp` | new public methods + `contextFull_`/host-KV state |
| `src/cpp/core/ModelForward.cpp` | context guard, host-KV slot helper, no CPU fallback on context-full |
| `src/cpp/core/ModelQwen35.cpp`, `ModelDebug.cpp`, `ModelForwardDebug.cpp` | host-KV slot helper (18 index sites in total) |
| `src/cpp/core/ModelGeneration.cpp` | stop at the window |
| `src/cpp/core/ModelGPU.cpp` | auto-fit charges the plan's K/V bytes per layer and honours `plan.kv`/`plan.kvMirror`; `geom.int8Kv` |
| `include/GPUComputeInternal.hpp` | int8 prefill attention kernels (2) |
| `src/cpp/core/cuda/forward/GPUComputeMoE.cu` | dense qwen35 (`forwardQwen35Prefix`): int8 store/prefill dispatch |
| `src/cpp/core/cuda/forward/GPUComputeForward.cu` | moe qwen35 (`forwardQwen35MoePrefix`): port the int8 decode block, then the store/prefill dispatch; qwen2 guard |
| `src/cpp/core/cuda/forward/GPUComputeUpload.cu` | KV/partials/RoPE sizing from the granted window, chunk scaling, f32 KV not allocated in int8-store mode |
| `include/GPUCompute.hpp` | `kvStoreI8_`, `contextFull_`, chunk cap |
| `benchmarks/BenchMain.cpp` | `--ctx` |
| `tools/ctx_probe.cpp` + `CMakeLists.txt` | long-window needle probe |
| `unit_tests/ContextPlanTest.cpp`, `unit_tests/CMakeLists.txt` | planner tests |
| `scripts/context_gate.sh` | long-window gate (build + run ctx_probe) |
| `docs/…`, `ENVIRONMENT.md` | document the knob and the precision escalation |

## 10. Numerics and verification

Auto mode keeps f32 K/V whenever it fits, so the existing gates must stay
bit-identical. int8 K/V is a real numeric change and needs its own evidence:

1. `exl3_gpu_parity` — 29 baseline `[FAIL]` lines, unchanged count.
2. `tiny_logits_probe` — two arms, two different bars:
   - **default path** (auto at 2048, i.e. f32 store + the existing int8 decode
     mirror): the published bar, `id=198`, top-5 `198/1133/271/460/317`, logit
     `19.308 ± 2e-3`. Nothing about the default may move.
   - **forced int8 store** (`TINYCODER_KV8=1` with `--ctx` large enough to
     select `kvStoreI8_`): same top-1 id and same top-5 ordering, with the logit
     delta *measured and reported* rather than held to 2e-3. Prefill attention
     over 8-bit K/V is a real numeric step, so requiring the f32 band would make
     the feature untestable; requiring the token to change would make it
     worthless. A top-1 change at 4 tokens, or a relative logit move beyond a
     few 1e-2, means a bug (bad scale, wrong scale index), not "quantization
     noise". Record the measured delta in the ledger.
3. `scripts/text_gate.sh` — unchanged pass on the default path, plus one run with
   `TINYCODER_KV8=1` to confirm int8 K/V does not break generation.
4. **Long-window gate, new** (`scripts/context_gate.sh`): prompt with a unique
   fact at the head, question at the tail, at 8192/32768/65536 tokens; the answer
   must appear. Runs f32 where it fits and int8 where it does not; a mismatch is
   a failure. This is the gate that says a granted window actually works.
5. Parity at long window: prefill `[A(3000)] + [B(3000)]` in two calls vs one
   6000-token prefill → same next token and logits within 1e-3 (fresh cache vs
   cache-carrying path). CPU on the 1.5B, GPU on the 1.5B.
6. `tinycoder_test` — 0 failures; the new `ContextPlanTest` cases count toward
   the total, so update `AGENTS.md`'s "90 tests, 73 passed" when it lands.
7. Numbers to publish (with `scripts/context_gate.sh` green on the same tree):
   granted window per model, VRAM/RAM used, pp at 8k/64k, tg at 2k/8k/32k/64k,
   and the f32-vs-int8 logit delta at 2048.

## 11. Risks

- **int8 K/V prefill kernels are new code on the correctness path.** Mitigation:
  bit-comparable formulation (same dequantization expression as the decode
  kernel), parity test 5, and the gate in 4. If int8 prefill attention cannot be
  made to pass the needle gate at a window that f32 cannot hold, the honest
  outcome is to keep f32 and grant the smaller window, saying so — not to ship a
  long window whose answers are wrong. Report it as a measured limitation.
- **Auto mode is a policy, and policies surprise people.** Mitigation: it is
  printed, it never reduces below the full-offload-preserving window unless the
  model is larger than the card, and an explicit request always wins.
- **Host RAM estimates.** Mitigation: re-check against `MemAvailable` after the
  weights are loaded, and let a failed allocation produce one clear error rather
  than a silent smaller window.
- **Bonsai at 64k on 11 GB is close to the edge** (9.7 of 11.2 GB). If the
  auto-fit's fixed overhead is off, this may land at 48k; the number will be
  measured and reported, not assumed.

## 12. Integration note (not in this repo)

`tinycoder-openai` already passes its per-model `context_length` to
`tc_engine_load`, and `tc_engine_info` will now report the granted window. To go
past 8192 end-to-end its registry entries must ask for it
(`models[].context_length`, falling back to `max_seq_len`). That is a config
change in that repo, not an engine change; it is listed here so the "still 8192"
outcome is not mistaken for a remaining cap in this tree.

## 13. Rollout

One feature commit on branch `1.0.2` (this repo's convention), amended while
unreleased; the user commits. Leave the tree dirty, describe the change.
