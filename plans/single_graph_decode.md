# Structural Single-Graph Decode — Findings & Plan

Status: 2026-10-01. Author: engine perf campaign.

## 1. What the task asked for

Reach llama.cpp-parity decode by turning the per-token forward into "one
fused multi-op graph with device-resident intermediates", eliminating the
~218 launches/token and the per-stage host round-trips.

## 2. Ground truth measured this session (RTX 2080 Ti, CUDA 12.0, arch 75)

Reference model: `qwen2.5-coder-1.5b-instruct-q2_k.gguf` (dense qwen2, 28
layers, H=1536, I=8960, vocab 151936). Harness: `tinycoder_bench`
(llama-bench parity, decode-only, greedy, warm reps).

### 2.1 The existing CUDA graph already captures the whole token

`TINYCODER_GPU_GRAPH=1` (default) captures ALL 28 layers' GEMVs + attention +
RoPE + KV store + LM head into a **single** `cudaGraphExec_t` and replays it
with one `cudaGraphLaunch` per token (see
[`captureDecodeGraph`](src/cpp/core/GPUCompute.cu:12165),
[`replayDecodeGraph`](src/cpp/core/GPUCompute.cu:12201)). So the headline
"single-graph decode" already exists.

### 2.2 …and it buys ~nothing

| Path | tg64 tok/s | ms/token |
|---|---|---|
| Graph ON (default) | 135.6 | 7.38 |
| Graph OFF (eager) | 136.2 | 7.34 |

Launch overhead is **not** the bottleneck. Per-token device elapsed
(`moeEvStart_/moeEvEnd_`, `TINYCODER_MOE_STATS=1`) is ~7.1 ms of a 7.36 ms
wall — the GPU is ~97 % busy inside the graph. Cutting launches cannot help.

### 2.3 The real bottleneck: decode attention scales O(context)

`TINYCODER_GPU_VERBOSE=1` stage sums (28 layers, eager):

| Context (pp) | QKV/rms | **kv/attn/rope** | attnO/ffn | layer loop |
|---|---|---|---|---|
| 16 | 1.21 | 0.76 | 4.21 | 6.19 |
| 128 | 1.08 | 2.88 | 3.70 | 7.66 |
| 512 | 1.07 | **10.31** | 3.65 | 15.03 |
| 1024 | 1.06 | **20.27** | 3.65 | 24.99 |

The GEMVs are flat (~3.6 ms) — they are already at the DRAM bandwidth wall.
Attention, by contrast, grows linearly and at 1 K context is **81 %** of the
layer loop.

Root cause: [`kWarpAttention`](src/cpp/core/GPUCompute.cu:306) /
[`kWarpAttentionPos`](src/cpp/core/GPUCompute.cu:430) launch **one warp per
(token, q-head)**. At batch==1 that is `nHeads = 12` warps for the entire
68-SM GPU, each looping over the whole `cachePos`. 56 of 68 SMs idle and the
critical path is O(cachePos).

## 3. Plan

1. **Split-KV decode attention (highest value, this change).** Tile the KV
   range into `attnChunk_`-sized windows; launch `nHeads * numChunks` warps
   (numChunks = ceil(maxSeqLen/chunkSize)); each warp folds one window into a
   `(m, l, acc)` online-softmax partial; a tiny combine kernel merges the
   partials with the standard flash-attention rescale. Same math, grouped
   reduction (fp rounding differs; top-k identity preserved). Works for both
   eager and captured graphs (value vs device-pos template).
2. Subsequent (out of scope here): fuse the residual/RMSNorm elementwise
   kernels into the GEMV epilogues and fold the LM head into the same graph
   node stream. These are marginal next to (1).

## 4. Implementation notes

- Scratch: `attnPartialAcc/M/L` sized `nHeads * attnChunks * HD` floats
  (`attnChunks = ceil(maxSeqLen/attnChunk_)`), allocated once in
  `ensureScratch`, freed in `destroyScratch`. Pointer-stable across capture.
- Env: `TINYCODER_ATTN_SPLIT` (default ON; `=0` disables),
  `TINYCODER_ATTN_SPLIT_CHUNK` (default 64).
- Kernels: [`kWarpAttentionSplit`](src/cpp/core/GPUCompute.cu:510) /
  [`kWarpAttentionSplitPos`](src/cpp/core/GPUCompute.cu:578) (value vs
  device-pos variants), combine
  [`kAttnCombinePartial`](src/cpp/core/GPUCompute.cu:650), host dispatcher
  [`launchDecodeAttentionSplit`](src/cpp/core/GPUCompute.cu:687).
- Only the dense arch (`geom_.architecture == 0`) and `seqLen == 1` use the
  split path; prefill and qwen35 keep the existing kernels.

## 5. Result (measured 2026-10-01, RTX 2080 Ti, arch 75, Release)

`tinycoder_bench --n-prompts <pp> --n-gen 64 --reps 5` (warm), split vs eager:

| Context (pp) | SPLIT=1 tok/s | SPLIT=0 tok/s | speedup |
|---|---|---|---|
| 64   | 145.59 | 134.60 | 1.08× |
| 256  | 144.17 |  89.03 | 1.62× |
| 512  | 143.12 |  61.50 | 2.33× |
| 1024 | 142.19 |  37.97 | 3.74× |

- **Attention stage** at pp=1024: **20.27 ms → 1.74 ms** (11.7× faster).
- Decode is now **flat at ~142–145 tok/s** across the whole context range;
  the eager path collapses as O(context) exactly as §2.3 predicted.
- **Prefill unaffected**: pp64 5,176 vs 5,174 tok/s.
- **Parity**: in-process A/B (eager vs split in the same process) IDENTICAL
  at every context length and chunk size; all deterministic GPU-vs-CPU gtest
  parity tests PASS with split on and off. The two full-suite failures
  (`AnswersQuestion/0`, `/3`) are pre-existing sampling flakes — they fail on
  the pre-change `build-rel` baseline too.

### 5.1 Why "single-graph decode" wasn't the win

The requested single-graph decode already existed (§2.1) and, per §2.2, buys
~0 % throughput because the graph is ~97 % GPU-busy. The measured wall was
**attention occupancy**, not launch count — so the deliverable is split-KV
decode attention, which parallelizes the O(context) scan across
`nHeads * numChunks` warps instead of `nHeads` warps.
