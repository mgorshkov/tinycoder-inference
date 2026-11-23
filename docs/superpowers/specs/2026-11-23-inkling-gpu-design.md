# Inkling-2-Mini CUDA forward: design

## Intent

Implement a CUDA forward for Inkling-2-Mini so decode beats llama.cpp on this
box. The current CPU path is correct but measures **0.32 tok/s** (tg16) against
llama.cpp CPU **4.17 tok/s** and **20.83 tok/s** prefill. The 2080 Ti is idle
during Inkling runs. The primary contract is to beat **llama.cpp CPU on IQ2_XXS**
(the same 8.49 GB file, which fits 11 GB with the K/V cache); the baseline must
be measured before the claim. The stretch target is to beat **llama.cpp CUDA on
IQ2_XXS**, whose baseline is also measured first. Measured on this box: llama.cpp CPU IQ2
**23.33 pp / 4.77 tg**; llama.cpp CUDA IQ2 **32.20 tg** (pp16 not produced).
Design point: >4.77 tok/s is the contract, 32+ is the stretch. 5 tok/s is the
floor, not the goal.

## Why the GPU, and why now

Decode is memory-bound: ~5.4B effective active parameters at ~2.3 bits is
~1.6 GB of weight traffic per token. At the card's 616 GB/s that is a ~360
tok/s ceiling; even 20% efficiency is 70 tok/s. The CPU path is at 0.32 because
2/4-bit expert blocks are decoded on the CPU and every small projection pays a
thread-pool dispatch. The GPU removes both costs. The CPU path stays as the
reference and the fallback.

## Goals

- IQ2_XXS decode at > 4.77 tok/s (llama.cpp CPU), 2048-token window, on the
  2080 Ti (sm_75); stretch > 32.20 tok/s (llama.cpp CUDA).
- Logits and generated text agree with the CPU path: same top-1 and top-5
  ordering on the fixed prompts, and the text gate answers.
- No inference library linked. cuBLAS is allowed as a plain GEMM.

## Non-goals

- Vision tower, DSpark speculator, 1M context.
- Q3_K_XL on the GPU (13.6 GB does not fit 11 GB).
- Beating llama.cpp prefill in the first phase. Prefill is phase 4; the
  decode target is the contract for this design.
- A general GPU framework: the kernels are Inkling-specific and may assume the
  published geometry (hidden 2304, 24/8 heads, 1152 expert FF, 128 experts).

## Approaches considered

1. **CPU int8 prepack** (rejected as the path to a win). Dequantize experts once
   to int8 + per-32 scales, use `maddubs`/`dpbusd`. The measured ceiling is
   parity with llama.cpp (it uses the same technique), not a win, and it costs
   ~26 GB of RAM.
2. **Hybrid: GPU matvecs, CPU glue.** Reuses the existing GEMV launchers, but
   the remaining CPU glue (norms, depth mix, conv, attention) measures ~1.2 s
   per token even with all matvecs removed. It cannot reach 4.17 without a
   second optimization project.
3. **Full GPU forward** (chosen). Every operation runs on the device; the host
   only embeds the prompt, launches the graph, and reads logits. Highest
   ceiling, largest kernel inventory, but the inventory is finite and each
   kernel is small.

## Architecture

New module `src/cpp/core/cuda/inkling/` with a class `InklingGPU` owned by
`Model` (like `ModelGPUState`). It owns:

- one device buffer per quantized weight tensor (physical block weights are
  uploaded once and shared across loop slots);
- 48 K/V slots (`48 x maxSeqLen x 8 x 128`, f16 or f32) and 48 short-conv
  state buffers;
- scratch for hidden, Q/K/V, relative table, attention scores, expert outputs;
- one stream and pinned host staging for the token id and final logits.

The forward is a device-side graph of small kernels, launched in the same order
as `forwardInkling`:

1. embedding lookup + token norm;
2. prelude blocks `[0, loopStart)`;
3. three loop passes over `[loopStart, loopEnd)` with per-slot state, message
   controls and the block bank;
4. postlude blocks `[loopEnd, numPhysicalLayers)`;
5. output depth mix + norm + LM head + softcap + vocab trim.

### Kernels

| kernel | notes |
|---|---|
| `kInklingEmbed` | existing embedding kernels; padded vocab |
| `kInklingRmsNorm` | hidden and per-head variants |
| `kInklingClip` | per-head L2 clip with optional scale |
| `kInklingConv` | depthwise causal 4-tap, per-slot history, optional dynamic taps |
| `kInklingRelBias` | table `[extent][heads]` + gather by distance + sink |
| `kInklingAttention` | GQA, SWA/full, sink logit, bias, log-scaled tau |
| `kInklingMix` | softmax over <= 5 RMS-normalized states, n queries 1/3 |
| `kInklingControl` | concat RMS, rank-64 down, tanh gain/forget |
| `kInklingRouter` | F32 router, sigmoid, bias, top-8, renormalize x8 |
| `kInklingSitu` | beta/lb SITU on gate/up |
| `kInklingPostConv` | residual + short conv + scale |
| `kInklingGemv` | one templated decoder for NVFP4, Q2_0, IQ1_S, IQ2_XXS |

The four quantized GEMV types are the only kernels that do not already exist in
a reusable form. `launchQGemv` covers Q4_K/Q5_K/Q6_K/Q8_0/IQ2_XS/IQ2_S/
IQ3_XXS/IQ3_S; the IQ2_XXS file also uses NVFP4, Q2_0, IQ1_S and IQ2_XXS. Each
decoder follows the existing `gemvLaunchIQ2XSF32Plain` shape: one warp per row,
Q8_1 activation, block decode into registers/shared, int8 dot. The block layouts
are already in `cuda/gemm/QuantBlocks.cuh` and are shared with the int8 GEMM.

## Memory budget (IQ2_XXS, 11 GB)

| item | size |
|---|---|
| weights | 8.49 GB |
| K/V, 48 slots x 2048 x 8 x 128 x 2 B x 2 (f16) | 0.40 GB |
| short-conv state | 0.004 GB |
| scratch (Q/K/V, table, experts, logits staging) | < 0.3 GB |
| CUDA context / cuBLAS workspaces | < 0.5 GB |
| **total** | **~9.7 GB** |

The window is planned before allocation; if 2048 does not fit the planner
reduces it, exactly as the CPU path does. No fp16 weight twins.

## Correctness

- Per-stage parity: dump `Qcur/Kcur/Vcur/Rcur`, `attn_out`, `ffn_out`, `l_out`
  from the GPU path and compare with the CPU path (which is already verified
  against llama.cpp) at cosine > 0.999, then exact top-5.
- Final-logits gate on `9707,11,1879,323,358,279,2779,338` and on the Paris
  chat prompt: top-1 exact, top-5 set, each logit within 0.15 absolute. This is
  a per-logit band; it is deliberately not "same order", which the CPU IQ2 path
  does not itself have.
- The text gate (`text_gate.sh`) must pass on IQ2_XXS before a speed number is
  reported.

## Phases and acceptance

1. **Skeleton + four GEMVs.** Upload IQ2 and implement `kInklingGemv` for
   **NVFP4, Q2_0, IQ1_S and IQ2_XXS** (block 0 is IQ1_S gate-up + Q2_0 down, so
   NVFP4 alone would block phase 2). Compare one expert projection per type.
   Accept: `max|d| / max|ref| < 1e-2` (a norm-relative gate; a per-element
   relative error is undefined near zero).
2. **Decode graph, one block.** Embed, norm, Q/K/V/rel/sink, conv, attention,
   gate/Wo, FFN for block 0. Accept: block-0 `l_out` cosine > 0.999 vs CPU.
3. **Loop + postlude + LM head.** Full token. Accept: top-5 parity and the
   Paris answer.
4. **Prefill (not a release gate).** `int8Gemm` cannot run this model: it
   requires 512-aligned columns, and the projections are 2304/1152 wide. Batch
   the decode GEMV at m=8 and add a tiled attention kernel instead. Accept:
   pp16 >= 5 tok/s (stretch 10) and the same text; decode remains the contract.
5. **Tuning.** CUDA graph, stream overlap, vectorized attention. Accept:
   tg16 >= 5 tok/s (stretch 15) on IQ2.

## Baselines to measure before claiming a win

1. llama.cpp CPU on IQ2_XXS: `llama-bench -m <IQ2> -p 16 -n 16 -t 36 -ngl 0 -r 1`
   -> **23.33 pp / 4.77 tg**.
2. llama.cpp CUDA on IQ2_XXS: same command with `-ngl 99` and
   `LLAMA_ARG_CTX_SIZE=2048` -> **32.20 tg** (pp16 not produced; record that).
3. TinyCoder CPU on IQ2_XXS (current reference): `tinycoder_bench --n-prompts 16
   --n-gen 16 --threads 36`.

The published row is "faster than baseline 1"; beating baseline 2 is the
stretch, reported separately.

## Risks

- **VRAM.** 9.7 GB of 11 GB leaves little headroom; the context planner must
  fail closed, and the 2048 window is the design point. Q3 is out of scope.
- **Kernel count.** ~1000 launches/token. At 5 us each that is 5 ms, tolerable;
  CUDA graph capture is the mitigation and is phase 5.
- **Attention at 2048.** The CPU kernel is O(pos); the GPU kernel must tile
  keys and keep the relative-bias gather in shared memory.
- **sm_75.** No `cp.async`; use synchronous staging and 4-byte/8-byte loads,
  the same constraint the existing kernels already live with.

## Open decisions left to the builder

- f16 vs f32 K/V (f16 halves the budget; correctness must be checked against
  the CPU f32 path).
- Whether to keep the CPU path as the default until phase 3 passes, and how the
  model selects the GPU path (`TINYCODER_GPU=1` + arch check).
- CUDA graph scope: whole token or per layer.
