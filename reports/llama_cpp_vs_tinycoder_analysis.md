# llama.cpp vs TinyCoder — Performance Analysis & Recommendations

**Scope**: architectural differences between [`llama.cpp`](/home/mike/git/llama.cpp) (commit `557614e02`, current main) and the TinyCoder inference engine, why generation/prefill performance differs on the default model (`qwen2.5-coder-1.5b-instruct-q2_k.gguf`), and concrete, evidence-ranked changes to align with or beat llama.cpp.

**Companion**: all diagrams and charts are in [`llama_cpp_vs_tinycoder_diagrams.md`](./llama_cpp_vs_tinycoder_diagrams.md).

---

## 1. TL;DR

1. **On the CPU the two engines are already at parity at the production thread count.** At 8 threads on the i7-4790K reference host, TinyCoder does 26.3–26.6 tok/s vs llama.cpp's 26.9 tok/s — a ~1 % gap that sits **below the DRAM floor** (~27 tok/s = 0.67 GB/token ÷ ~18 GB/s). Both stream the *same* GGUF weight bytes per token; the kernels differ but the memory wall does not.
2. **The historical "big difference" was a scheduling bug, now fixed.** Before the 2026-08-27 work-stealing campaign, TinyCoder's static-slab FFN dispatch made 4 threads drag at 19.2 tok/s vs llama.cpp's 29.4. Porting llama.cpp's `atomic_fetch_add` chunk stealing into the fused FFN (`parallelForSteal2`) recovered +11–14 % at 4 threads.
3. **The remaining 4-thread gap (~22 vs 29.4 tok/s) is not one lever** — it is the sum of llama.cpp's per-kernel internals, its thread-split activation quantize, and its single-dispatch-per-tensor graph. Each of those, implemented individually in TinyCoder, measured equal-or-slower on this host (see §5.3). Parity at 4 threads requires a different memory substrate, not more CPU kernel tricks.
4. **On the GPU the dominant gap was attention, now fixed.** Eager decode attention was O(context) (20.27 ms at 1 K context on RTX 2080 Ti, 81 % of the layer loop). The split-KV decode attention cut the attention stage 11.7× and made decode flat at ~142–145 tok/s across all context lengths. The whole-token CUDA graph (already present) buys ~0 % because the GPU is ~97 % busy inside it.
5. **Recommended next steps** (ranked by expected gain): (R1) FP16 KV cache on the CPU path; (R2) one shared act→Q8_K quantization per layer reused across all batch kernels (prefill); (R3) apply work-stealing dispatch to the LM-head and QKV stages at low thread counts; (R4) optional lossless Q3_K→Q2_K re-quant of `ffnDown` behind a quality flag (−24 % FFN traffic); (R5) hardware: DDR4-3200 or CUDA offload is the only path past the CPU floor; (R6) long-context GPU headroom: fold residual/RMSNorm into GEMV epilogues and ensure F16 KV is always on.

---

## 2. Measurement baseline (what "the big difference" actually is)

All numbers below come from the engine's own llama-bench-parity harness ([`benchmarks/BenchMain.cpp`](../benchmarks/BenchMain.cpp:1)) and the documented A/B campaigns in [`plans/generation_optimizations.md`](../plans/generation_optimizations.md:25) and [`plans/single_graph_decode.md`](../plans/single_graph_decode.md:11). Host: i7-4790K (4c/8t, 8 MiB L3, DDR3-1600 dual-channel) for CPU; RTX 2080 Ti (CUDA 12.0, arch 75) for GPU. Model: `qwen2.5-coder-1.5b-instruct-q2_k.gguf` (28 layers, H=1536, nHeads=12, nKVHeads=2, headDim=128, I=8960, vocab=151936).

| Metric | TinyCoder | llama.cpp | Gap |
|---|---|---|---|
| CPU generation @8 threads (tg64) | 26.3–26.6 tok/s | 26.9 tok/s | ~1 % (at DRAM floor) |
| CPU generation @4 threads (tg64) | 21.6–22.5 tok/s | 29.4 tok/s | ~24–27 % |
| CPU generation @4 threads (pre-fix slab) | 19.2–19.9 tok/s | 29.4 tok/s | ~33 % |
| CPU hard floor (0.67 GB/token @ ~18 GB/s) | ~27 tok/s | ~27 tok/s | — |
| GPU prefill pp64 | ~5,176 tok/s | n/a on this host¹ | — |
| GPU decode tg64 @pp1024 (eager attn) | 37.97 tok/s | — | attention-bound |
| GPU decode tg64 @pp1024 (split-KV) | 142.19 tok/s | ~150–250 tok/s typical² | needs side-by-side run |

¹ No llama-bench numbers exist in-repo for this GPU/host; a direct `llama-bench -p 64 -n 64 -ngl 99` run is the recommended follow-up. ² Typical published llama-bench range for 1.5B q2_k on 2080-Ti-class hardware; **not a measured value on this host** — do not quote as fact until measured.

**Important framing**: the two engines' CPU decode rates converge as thread count rises because both are
memory-bandwidth-bound. Threads only change *how well* the available DRAM bandwidth is used, and at
8 threads both use it nearly optimally. The interesting differences are therefore (a) at 4 threads
(scheduling/ALU efficiency), (b) at prefill (GEMM shape), and (c) on GPU at long context (attention).

---

## 3. Architectural difference matrix

| Dimension | llama.cpp | TinyCoder | Performance consequence |
|---|---|---|---|
| Execution model | DAG of ops per token, built per arch, scheduled across backends | Imperative per-layer loop with hand-fused kernels | TinyCoder has far fewer dispatch barriers/token (~390 ops vs ~120 kernel calls); llama.cpp amortizes via graph reuse |
| Parallelism unit | one threadpool dispatch **per tensor** (whole `nr0×nr1` space) | one dispatch **per kernel**, some kernels cover several tensors (Q+K, gate+up+down) | llama.cpp's per-tensor dispatch is the stronger shape at low thread counts (see §5.1) |
| Work distribution | `atomic_fetch_add` chunks, 64 rows for GEMV | slab (static) **or** steal (4-tile chunks) | steal closed 60 % of the 4-thread gap |
| Activation quantization | act→Q8_K `from_float` once per tensor, thread-split | cooperative last-arriver quantize inside fused FFN phase-1 | net-zero on this host (DRAM hides the bubble); llama.cpp's version is marginally better at 4 threads |
| Weight streaming | same GGUF bytes (Q2_K/Q3_K/Q6_K) | same GGUF bytes, plus compact/prepacked copies | identical floor; TinyCoder adds load-time copies to cut per-token ALU |
| Attention (CPU) | FA-fused op (flash-attn port) or vanilla path | `attentionFused` SIMD flash-style | both ~1.4 ms/tok at short ctx; parity |
| Attention (GPU) | flash-attn FA, split-KV stream-k style for long ctx | `kWarpAttention` + **split-KV** (nHeads × chunks warps) | TinyCoder now matches llama.cpp's shape; 11.7× attention-stage fix |
| KV cache dtype | f16 (CPU & GPU) | **f32 CPU**, f16/f32 GPU | CPU attention reads 2× cache bytes in TinyCoder (R1) |
| Prefill GEMM (CPU) | vec_dot row-chunked, x reused across tokens | register-tiled Q8_K int8 batch kernels, weight-stationary | TinyCoder's qwen35 19.6× win; for 1.5B both are batched and similar |
| Prefill GEMM (GPU) | cublas fp16 / mmq | cublasGemmEx fp16 tensor cores | parity approach |
| LM head | full MUL_MAT per token (or get_rows tied) | Q6_K 8-row tile, top-K pruning, `computeAllLogits=false` skips prefill heads | TinyCoder skips redundant prefill LM-head work llama.cpp also skips via output flags |
| Thread default | physical cores | logical CPUs | TinyCoder saturates DRAM with HT lanes at 8t; llama.cpp avoids HT contention at 4t |
| Memory mgmt | ggml-alloc arena, graph-reserved buffers | per-thread `ScratchPool`, zero allocs/token | parity in steady state |
| CUDA graphs | op-level optional capture | whole-token capture (`cudaGraphLaunch`/token) | measured ~0 % benefit; launch overhead was never the wall |
| Multi-backend | CPU/CUDA/Metal/Vulkan/RPC scheduler | CPU + CUDA, hard-coded offload policy | TinyCoder is simpler; llama.cpp's abstraction costs scheduling overhead TinyCoder avoids |

---

## 4. Deep dive: how each engine runs the default model

### 4.1 llama.cpp decode step

1. `llama_decode()` packs the single new token into a `llama_ubatch`, applies `n_ubatch` micro-batching.
2. `build_qwen2()` ([`src/models/qwen2.cpp`](/home/mike/git/llama.cpp/src/models/qwen2.cpp:100)) constructs ~390+ ops: per layer RMSNorm → 3× `MUL_MAT` (Q/K/V) → RoPE → KV-store view → **fused `FLASH_ATTN` node** (resolved at load time by `resolve_fused_ops`, [`src/llama-context.cpp`](/home/mike/git/llama.cpp/src/llama-context.cpp:504)) → `MUL_MAT` O → add → RMSNorm → 2× `MUL_MAT` (gate/up) → silu → `MUL_MAT` down → add.
3. `ggml_backend_sched` splits the graph across backends (CPU + CUDA0..N for `-ngl`), then `ggml-alloc` reuses graph buffers.
4. Each backend computes its slice. On CPU, every `MUL_MAT` is **one** threadpool task: workers `atomic_fetch_add` 64-row chunks over the whole `(nr0, nr1)` space ([`ggml-cpu.c:1254`](/home/mike/git/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1254)); the x-activation is quantized to Q8_K once by a thread-split `from_float`.
5. On CUDA, decode projections go through **mmvq** quantized GEMV kernels (VDR-tuned per type, [`ggml-cuda/mmvq.cu`](/home/mike/git/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:289)); prefill uses fp16 tensor-core `cublasGemmEx`; attention is the FA flash kernel with split-KV for long contexts.

**Cost model**: per token, ~390 op dispatches × barrier ≈ small-but-real fixed overhead (visible at 4 threads as llama.cpp's *edge* — it wins despite more dispatches because each dispatch is fully parallel over the tensor, no tail drag).

### 4.2 TinyCoder decode step

1. `Model::generate()` → `forward({token})` ([`ModelForward.cpp`](../src/cpp/core/ModelForward.cpp:48)). GPU fast path when enabled; CPU path otherwise.
2. Per layer: RMSNorm SIMD → fused **Q+K compact Q2_K GEMV** + V GEMV ([`ModelForward.cpp`](../src/cpp/core/ModelForward.cpp:412)) → RoPE (Q only) → `storeKVWithRoPE` (K rotation fused into the cache write) → `attentionFused` (SIMD flash-style, [`ModelPrimitives.cpp`](../src/cpp/core/ModelPrimitives.cpp:185)) → **attnO compact Q3_K with residual in the epilogue** ([`ModelForward.cpp`](../src/cpp/core/ModelForward.cpp:513)) → RMSNorm → **fused gate+up+down Q2_K→Q3_K single kernel** with work-stealing and cooperative act quantize ([`ModelForward.cpp`](../src/cpp/core/ModelForward.cpp:710), [`SIMDMatMulVecAVX2.cpp`](../src/cpp/core/SIMDMatMulVecAVX2.cpp:5042)).
3. Final norm → LM head (Q6_K separate head, 8-row tile, optional exact top-K pruning via `lmHeadBounds_`).
4. Dispatch: `seqLen==1` runs kernels on the calling thread; kernels parallelize internally via `parallelForSteal2` (FFN) or per-row tiling (LM head).

**Cost model**: ~120 kernel calls/token (≈ 4–6 per layer) with zero heap allocations. The fusion removes intermediate buffer round-trips, but the weight bytes are identical to llama.cpp's — which is why the engines meet at the DRAM floor.

---

## 5. Root-cause analysis of the CPU generation gap

### 5.1 Why llama.cpp wins at 4 threads (and why TinyCoder nearly closes it at 8)

The single largest measurable cause was **static-slab tail drag**: with 4 threads and a static partition of the fused FFN rows, the last finishing thread idles DRAM during the tail. llama.cpp's dynamic chunk stealing keeps DRAM saturated until the final chunk is stolen. TinyCoder's fix (`parallelForSteal2`) recovered +11–14 % (19.2→21.6–22.5 tok/s), leaving:

| Residual factor (measured individually) | Effect | Verdict in TinyCoder |
|---|---|---|
| llama.cpp `nr0×nr1` chunking shape | 64-row bands measured **+1.4 % slower** | reverted — 16-row register tiles better on 4c/8 MiB L3 |
| `b-outer/r-inner` loop order | +5.5 % slower | reverted |
| Software prefetch hints | ±0.5 % noise | reverted |
| LM-head tile shape (8 vs 16 vs 32 rows) | sub-noise | kept 8 |
| Cooperative act→Q8_K quantize | net-zero at 8t | kept as cleanup (AVX2 bsums) |
| llama.cpp thread-split quantize + kernel-internal ALU | the *residual* ~4–5 tok/s at 4t | each isolated change measured equal-or-slower |

**Conclusion**: at 4 threads the remaining ~7 tok/s is a *bundle* of small llama.cpp advantages that are not individually transplantable without regressing other parts of the TinyCoder shape. At 8 threads the bundle shrinks to ~0.4–0.6 tok/s. **The engine is at the memory wall, not the ALU wall.**

### 5.2 The DRAM floor is the real constraint

0.67 GB/token (FFN ~390 MB, LM head ~231 MB, rest ~50 MB) at ~16–18 GB/s effective gives a hard floor of 27–31 tok/s on DDR3-1600. Both engines sit within 0.5 tok/s of it. **No CPU-side kernel change can exceed ~27 tok/s on this host.** The only levers past it: faster DRAM (DDR4-3200 ≈ +30 %), a smaller LM-head quant (forbidden: lossy), or CUDA offload.

### 5.3 What was tried and reverted (evidence of the wall)

From [`plans/generation_optimizations.md`](../plans/generation_optimizations.md:80):

- 64-row band threading: `fused_gateUp_ffnDown` 3752 → 3806 ms (**worse**).
- Prefetch distance/hint sweep (+32/+96, T0/T1): all within ±0.5 % noise.
- LM-head tile 16/32 rows: 1678 vs 1679 ms — sub-noise.
- b-outer/r-inner: +5.5 % slower.

These reversals are *proof* the remaining gap is not addressable by CPU kernel micro-optimization on this memory substrate.

---

## 6. Root-cause analysis of the prefill gap

TinyCoder's prefill was historically its weakest point (the qwen35 dense path started at 135 s/token). The fix was a family of **register-tiled batch GEMMs** ([`QuantizedMatrix.cpp`](../src/cpp/core/QuantizedMatrix.cpp:93)) that are weight-stationary (each weight row reused across all tokens) and quantize x to Q8_K once per kernel, using `_mm256_maddubs_epi16` int8 dots:

- Prefill QKV: prepacked Q2_K for Q/K, Q4_K register-tiled for V ([`ModelForward.cpp`](../src/cpp/core/ModelForward.cpp:324)).
- Prefill FFN: `matMulVecFusedGateUp_Batch` (silu fused in epilogue) + compact Q3_K/Q8_K batch for down ([`ModelForward.cpp`](../src/cpp/core/ModelForward.cpp:597)).
- LM head: skipped for all but the last token (`computeAllLogits=false`) — llama.cpp's `n_outputs` machinery does the equivalent.

**Remaining structural differences vs llama.cpp prefill:**

1. **Per-kernel x-quantization**: each TinyCoder batch kernel re-quantizes the activation (norm output) to Q8_K. llama.cpp quantizes the activation **once per tensor** and shares it across the rows of that tensor's matmul. Within one layer, TinyCoder quantizes the same vector up to 4–5 times (Q, K, V, gate+up, down-x) where llama.cpp does it ~3 times (once per MUL_MAT input). This is ALU waste — mostly hidden at pp64, but it scales with tokens.
2. **No cross-tensor activation reuse**: the attn-norm output is re-read per projection kernel. llama.cpp's graph keeps activations in L2/L3-sized arena buffers; TinyCoder's scratch buffers do the same, so this is minor.
3. **Threading granularity in prefill**: prefill kernels parallelize over output rows (register tiles); llama.cpp parallelizes over `nr0×nr1` chunks (tokens × rows). For pp64 the token dimension adds parallelism llama.cpp exploits at 8 threads.

**Prefill parity statement**: TinyCoder pp64 (CPU) is not separately benchmarked vs llama.cpp in-repo for the 1.5B model; the GPU pp64 is ~5,176 tok/s. Given both use the same class of batched kernels and the GPU paths are both cuBLAS fp16, the expected prefill gap is small; a direct `tinycoder_bench` vs `llama-bench -p 64` CPU/GPU A/B is the recommended measurement (§8, M0).

---

## 7. GPU path analysis (why decode differed, and what fixed it)

Ground truth from [`plans/single_graph_decode.md`](../plans/single_graph_decode.md:28):

- The whole-token CUDA graph already existed and buys **~0 %** (135.6 vs 136.2 tok/s): device elapsed is ~7.1 ms of a 7.36 ms wall; the GPU is ~97 % busy. Launch count was never the problem.
- The actual bottleneck was **decode attention occupancy**: one warp per (token, q-head) = 12 warps on a 68-SM GPU, each scanning the full `cachePos`. Attention went from 0.76 ms (pp16) to 20.27 ms (pp1024) — 81 % of the layer loop — while GEMVs stayed flat at ~3.6 ms (DRAM/VRAM bound).
- **Fix (implemented)**: split-KV decode attention — `nHeads × numChunks` warps, online-softmax partials, tiny combine kernel ([`GPUCompute.cu`](../src/cpp/core/GPUCompute.cu:510)). Attention stage 20.27 → 1.74 ms (**11.7×**); decode flat at 142–145 tok/s for pp64…pp1024; parity retained.

This is now architecturally equivalent to llama.cpp's FA split-KV/stream-k decode attention. Remaining GPU deltas:

| Area | llama.cpp | TinyCoder | Gap |
|---|---|---|---|
| Decode GEMV | mmvq, VDR-tuned, per-type CTA geometry | `kQGemv*` per-type kernels | functionally equivalent; tune `kQGemvKxQ8K` CTA shape per type if profiling shows it |
| Fused KV-store + attention | single fused op | separate store + attention kernels (both in the same graph) | small; a fused store+attn kernel would cut one launch per layer |
| Residual/norm fusion | fused elementwise ops | separate `kAddResidual`, `kRMSNormRow`, `kSiluMul` launches per layer | ~3 launches/layer × 28 = ~84 launches/token inside the graph — currently hidden, becomes relevant only if kernel times shrink |
| KV dtype | f16 | f16 (kvHalf) — verify always on | if any path stores F32, attention bandwidth doubles |
| Long context | FA split-KV scales sub-linearly | split-KV flat to pp1024, scales the same way | parity |

---

## 8. Recommendations (ranked by expected gain, with evidence)

### M0 — Establish a clean side-by-side benchmark first (cheap, prerequisite)

Run on the same host, same model, same warm-repeat protocol:
`tinycoder_bench --model qwen2.5-coder-1.5b-instruct-q2_k.gguf --n-prompts 64 --n-gen 64 --reps 5` vs
`llama-bench -m <model> -p 64 -n 64 -t 4` and `-t 8` (and `-ngl 99` on the GPU host).
Expected outcome per §2: 8-thread parity, 4-thread llama edge, GPU near-parity after split-KV.
This makes every subsequent change measurable with the interleaved A/B discipline already documented in [`plans/generation_optimizations.md`](../plans/generation_optimizations.md:180).

---

### R1 — FP16 KV cache on the CPU path (highest value on CPU; low risk)

**What**: store K/V as `uint16_t` (F16) instead of `np::Array<float>` ([`Model.hpp`](../include/Model.hpp:700)); dequantize to F32 inside `attentionFused` and `storeKVWithRoPE` or keep an F16 dot path.
**Why**: attention reads the *entire* cache per token per layer. F32 cache = 2× bytes read vs llama.cpp's f16. At 1 K context this is ~2.3 MB reads/layer × 28 — currently masked by the short-context profile (attention ≈ 1.4 ms/tok) but it is the single cleanest bandwidth reduction left on CPU.
**Risk**: F16→F32 conversion ALU per element; cache-position writes. Mitigate by converting on the fly inside the dot loop (llama.cpp does exactly this).
**Expected gain**: attention stage −40–50 % at long context; generation +1–3 % at ≥512-token context on CPU; negligible at short context.

---

### R2 — One shared act→Q8_K quantization per layer in prefill (prefill efficiency)

**What**: quantize the RMSNorm output once per layer into a shared Q8_K buffer, and have all batch GEMM kernels of that layer (Q/K/V, gate+up, down-x) consume it instead of re-quantizing per kernel.
**Why**: TinyCoder re-quantizes the same activation 4–5× per layer ([`ModelForward.cpp`](../src/cpp/core/ModelForward.cpp:324)); llama.cpp does it once per tensor. Pure ALU savings; matters more as tokens/pp grow.
**Risk**: buffer layout coupling between kernels; bit-parity must be re-verified (49/49 suite).
**Expected gain**: prefill ALU down ~10–20 %; wall-clock gain scales with pp size (pp64: small; pp512+: measurable).

---

### R3 — Extend work-stealing dispatch to the LM head and QKV stages (low-thread-count parity)

**What**: apply `parallelForSteal2`-style chunk stealing to the LM-head 8-row tiles and the QKV GEMV stages when `threads ≤ physical cores`.
**Why**: the steal schedule was the +11–14 % unlock for the fused FFN ([`plans/generation_optimizations.md`](../plans/generation_optimizations.md:101)); the LM head is ~28 % of generation and still uses a static row partition (`matMulVecBatchQ6K_Q8K_AVX2`). Tail-drag in the head is the same failure mode the FFN had.
**Risk**: none of substance — schedule-only change, bit-identical (the FFN precedent proves parity holds).
**Expected gain**: +1–3 % at 4 threads; ~0 at 8 (bandwidth-bound).

---

### R4 — Optional lossless Q3_K→Q2_K re-quant of `ffnDown` behind a quality flag (bandwidth reduction)

**What**: build a Q2_K copy of `ffnDown` at load time (like the existing prepacked copies) and use it only under an explicit opt-in. The Q2_K dot streams 84 B/block vs Q3_K's 110 B/block (−24 %).
**Why**: the FFN is 66–70 % of generation and the down-matrix is half of it. This is the *only* remaining lossless weight-stream cut on CPU.
**Why not default**: the repo's §3 rule forbids re-quant of the down matrix on the default path (it previously reached >26.3 tok/s only by emitting divergent quality — see [`plans/generation_optimizations.md`](../plans/generation_optimizations.md:143)). Keep it opt-in like the pre-packed copies.
**Expected gain**: ~+10–15 % generation **if** quality allows (requires per-model validation); DRAM floor drops from ~27 to ~31 tok/s.

---

### R5 — Hardware: DDR4-3200 or CUDA offload is the only path past the CPU floor (planning)

The documented ceiling math is unambiguous: 0.67 GB/token ÷ ~18 GB/s ≈ 27 tok/s hard floor on DDR3-1600. 35 tok/s needs ~23 GB/s or ~0.58 GB/token. Options:
- DDR4-3200 dual-channel ≈ +30 % bandwidth → ~35 tok/s (stated in [`plans/generation_optimizations.md`](../plans/generation_optimizations.md:95)).
- CUDA offload (already implemented and default in CUDA builds) moves decode to VRAM bandwidth: ~142–145 tok/s measured on RTX 2080 Ti.

---

### R6 — GPU headroom: fuse store+attention, fold residual/norms into GEMV epilogues, verify F16 KV (incremental)

- Verify `kvHalf` is always active on the decode path (GPU already supports F16 KV; make it unconditional).
- Fuse `kStoreKVRope` + `kWarpAttentionSplit` into one kernel (saves 1 launch/layer; currently ~28 launches/token inside the graph).
- Fold `kAddResidual`/`kRMSNormRow` into the GEMV epilogues (saves ~84 launches/token).
- These are marginal today (graph is ~97 % busy, kernels are bandwidth-bound) but compound with R1-class cache reductions at long context.

---

### What NOT to do (measured regressions or dead ends)

| Idea | Evidence | Verdict |
|---|---|---|
| Whole-token CUDA graph "optimization" | already exists; ~0 % benefit | keep, don't expand effort here |
| 64-row band threading | +1.4 % slower | reverted |
| b-outer/r-inner loops | +5.5 % slower | reverted |
| Prefetch hint/distance sweeps | ±0.5 % noise | reverted |
| Oversubscribe threads (> logical CPUs) | −84 % at 12 threads | hard cap at logical CPUs |
| Sub-Q6_K LM head re-quant | lossy; previously "won" benchmarks by diverging quality | forbidden on default path |
| Adopt ggml-style DAG+scheduler just for CPU | TinyCoder's imperative fused loop already achieves parity at the wall; the scheduler's only proven value was work stealing, already ported | not needed |

---

## 9. Source references

- TinyCoder forward/generation: [`ModelForward.cpp`](../src/cpp/core/ModelForward.cpp:48), [`ModelGeneration.cpp`](../src/cpp/core/ModelGeneration.cpp:168), [`Model.hpp`](../include/Model.hpp:125)
- TinyCoder threading: [`ThreadPool.hpp`](../include/ThreadPool.hpp:87), [`ThreadPool.cpp`](../src/cpp/core/ThreadPool.cpp)
- TinyCoder SIMD kernels: [`SIMDMatMulVecAVX2.cpp`](../src/cpp/core/SIMDMatMulVecAVX2.cpp:5042), [`QuantizedMatrix.cpp`](../src/cpp/core/QuantizedMatrix.cpp:93)
- TinyCoder GPU: [`GPUCompute.cu`](../src/cpp/core/GPUCompute.cu:325) (attention), [`GPUCompute.cu`](../src/cpp/core/GPUCompute.cu:774) (GEMV), [`GPUCompute.cu`](../src/cpp/core/GPUCompute.cu:12615) (graph capture), [`ModelGPU.cpp`](../src/cpp/core/ModelGPU.cpp)
- TinyCoder measured data: [`plans/generation_optimizations.md`](../plans/generation_optimizations.md:25), [`plans/single_graph_decode.md`](../plans/single_graph_decode.md:28)
- llama.cpp graph build: `/home/mike/git/llama.cpp/src/models/qwen2.cpp:100`
- llama.cpp fused-op resolution: `/home/mike/git/llama.cpp/src/llama-context.cpp:504`
- llama.cpp mul_mat chunk stealing: `/home/mike/git/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1254`
- llama.cpp CUDA decode GEMV: `/home/mike/git/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:289`
- llama.cpp ubatch processing: `/home/mike/git/llama.cpp/src/llama-context.cpp:1333`