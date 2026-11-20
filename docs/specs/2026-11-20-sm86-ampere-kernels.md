# Ampere (sm_86) kernels for every supported model — design

Status: draft for approval, 2026-11-20.
Scope owner: this design covers the GPU compute paths of every row published in
`BENCHMARKS-2080TI.md`, and produces the first column of `BENCHMARKS-3090.md`. It is
written to be planned from without further questions to the author; the decisions that
were open are recorded in §12.

## 0. Why now

An external user reproduced our `GPU comparison: tinycoder vs llama.cpp / exllamav3`
table on an RTX 3090 and reported three things that do not fit "the numbers scale
with bandwidth":

| row | our 2080 Ti (`BENCHMARKS-2080TI.md`) | their 3090 | ratio |
|---|---|---|---|
| Qwen3.8-27B EXL3 decode | 23.60 tok/s | 26.6 tok/s | **1.13x** (bandwidth ratio 1.52x) |
| Qwen3.8-27B EXL3 decode, exllamav3 | 23.76 tok/s | 54.0 tok/s | 2.27x |
| Ternary-Bonsai-2-27B `pq2_0` prefill | 880.6 tok/s | 27.7 tok/s | **0.03x** |
| Ternary-Bonsai-2-27B `pq2_0` decode | 51.73 tok/s | 24.6 tok/s | 0.48x |
| Ornith-35B-A3B `q8_0`, 2x3090 full offload | 53.6 tok/s (hybrid) | 125.6 tok/s (llama.cpp) | 2.3x |

Two readings, both of which the design must act on:

1. **Decode barely scales with the wider card, and that is the real defect.** Their
   EXL3 decode number implies 8.6 GB / 37.6 ms = **229 GB/s, 24% of 936**. On our own
   2080 Ti the same path measures 472-495 GB/s, i.e. **77-80% of 616**
   (`.superpowers/sdd/2026-11-18-decode-budget-attack/progress.md`, STAGE_TRACE
   section). So the same kernel that is near the memory roof on Turing lands at a
   quarter of the roof on Ampere. A path that loses efficiency when the card gets
   faster is latency- and occupancy-limited, not bandwidth-limited.
2. **The Bonsai `pq2_0` row is not a tuning gap, it is a path that was not taken.**
   pp == tg on their card is the signature of the batched int8 GEMM declining and the
   engine falling back to per-token GEMV: `gemmPQ2Int8` returns `false` on any of
   several failure conditions (`src/cpp/core/cuda/gemm/GemmDispatch.cu:620-700`) and
   the caller silently takes the slow route. On our card the same dispatch measures
   880 tok/s, so whatever declined on their box is (a) silent by construction and
   (b) arch- or build-dependent. This is instrumentable today.

The common cause is that the entire tree is tuned for one card: **sm_75, 68 SMs,
616 GB/s, 64 KB smem, no `cp.async`, no `m16n8k16`, 32 warps/SM**. Ampere does not
make that card 1.5x faster; it makes four structural options available (async copy,
k32 int8 MMA, 16 KB more smem per block, 48 warps/SM) and moves the bottleneck from
"bytes/sec per SM" to "warps in flight per SM". The work below is to take those
options per kernel family, not to rescale tile sizes.

## 1. The list — scope is exactly the `BENCHMARKS-2080TI.md` table

The supported set is the nine rows of the GPU comparison table, and nothing else. No
other checkpoint in `/data/models` is in scope; models that the engine supports but the
table does not publish stay on their current path and only have to not regress.

| # | row (`BENCHMARKS-2080TI.md`) | quant type(s) | GPU mode | prefill kernel | decode kernel |
|---|---|---|---|---|---|
| 1 | Qwen2.5-Coder-1.5B | `q2_k` | full offload | `xW4` int8 GEMM | `kQGemvQ2KxW4_Nm` / `GpuGemvQ2KFused` |
| 2 | Qwen2.5-Coder-1.5B | `iq3_xxs` | full offload | IQ int8 GEMM | `GpuGemvIQInt8` / `GpuGemvIQ` |
| 3 | Qwen2.5-Coder-7B | `iq2_s` | full offload | IQ int8 GEMM | `GpuGemvIQInt8` / `GpuGemvIQ` |
| 4 | Qwen2.5-Coder-7B | `iq3_xxs` | full offload | IQ int8 GEMM | `GpuGemvIQInt8` / `GpuGemvIQ` |
| 5 | Ternary-Bonsai-2-27B | `pq2_0` | full offload | PQ2_0 int8 GEMM | `kQGemvPQ2_0Q8F` |
| 6 | Ternary-Bonsai-2-27B | `ptq1_0` | full offload | PQ2_0 tile loader (transcoded) | `kQGemvPTQ1_0*` |
| 7 | Ornith-1.5-9B | `q4_k_m` -> `q4_k` | full offload | `xW4` int8 GEMM | `xW4` / `GpuGemvRowKQuant` |
| 8 | Ornith-1.5-35B-A3B | `q8_0` + `q4_k` experts | hybrid, layer arena | `xW4` int8 GEMM, grouped | `FFNPersistent` + MoE |
| 9 | Qwen3.8-27B EXL3 | `1.60bpw` EXL3 | full offload | materialise + cuBLAS (fused HMMA alt.) | `exl3FusedMatVec` / `exl3GemvV2` |

Seven GGUF quant types plus EXL3. In code but **out of scope** (must not regress, no new
work): `q3_k`, `q5_k`, `q6_k`, `iq2_xs`, `iq3_s`, `q4_0`/`q5_0`/`q8_1` dp4a, MXFP4/NVFP4,
and every checkpoint not listed above.

Two properties of the list drive the order of work:

- **Rows 1-4 and 7 are already latency rows on the 2080 Ti.** Row 1 decodes 0.7 GB at
  3.07 ms/token = 233 GB/s, i.e. 38% of the card's bandwidth while its weights are a
  quarter of the cache-sized regime. Their 3090 numbers will not improve on bandwidth;
  they improve only on per-token fixed cost (§7.4) and grid parallelism (§7.2.1).
- **Rows 5-6 and 9 are the ones the external report breaks**, in that order of severity
  (rows 5-6: prefill fell to decode speed; row 9: decode at 24% of the roof while the
  reference engine reaches 50%).

## 2. Intent

Ship a second, explicitly Ampere-shaped implementation of every hot kernel family for
those nine rows, so that on a GA102-class card (RTX 3090 first) each published row is
within a stated fraction of the memory or tensor roof, with the text gates passing, and
so that the first rented 3090 hour produces a published column instead of a debugging
session.

## 3. Goals / non-goals / anti-goals

**Goals**

1. Every family in §7 has a kernel that is *structurally* correct for Ampere: async
   staging where the loop is staging-bound, occupancy chosen for 48 warps/SM, int8
   MMA in the `m16n8k32` form, and no more than 2 barriers per k-step.
2. The dispatch that selects a path is driven by **measured device properties**
   (SM count, opt-in smem, cc) and is **loud on fallback**: one `stderr` line per
   process when a batched path declines, and a counter in the bench output.
3. Validation that does not need a 3090 (§8): every change is verifiable from the
   cubin's SASS plus bit-exact parity on the sm_75 card.
4. One rented-hour script (§8.4) that builds, gates, benchmarks and profiles on the
   target card and emits a table column.

**Non-goals**

- Ada (`sm_89`) and Blackwell (`sm_120`) tuning. The build already emits `sm_80` and
  `sm_86`; anything gated on `>= 800` also serves `sm_80` for free, and that is
  desirable, but no `sm_89`-specific instruction will be used.
- New quantization formats, new models, or changes to numerics that the gates have
  not licensed.
- The CPU path, MoE expert offload heuristics, and the sampler.

**Anti-goals** (a result that technically hits its number and is still a failure)

- A speedup on a path that fails `scripts/text_gate.sh`, or a row added to
  benchmark file whose shape is not on the `AnswersQuestion` allow-list
  (`unit_tests/ModelTest.cpp`) — an ungated row can silently measure garbage.
- A second implementation of one loop that drifts from the first. The fused-vs-cuBLAS
  history is the model: two code paths, one of which nobody re-checks.
- Numbers from a build whose arch list is unknown. `CMAKE_CUDA_ARCHITECTURES` is
  already a cache variable and a stale value produces a plausible, wrong binary.
- "Faster" measured with a knob set that the shipped default does not set.

## 4. Constraints

1. **No 3090 is available.** Only an RTX 2080 Ti (sm_75, 11 GB) is present. Any claim
   of a 3090 win is a prediction until §8.4 runs.
2. `AGENTS.md` protocol binds: build the core target explicitly and check the exit
   code; the parity probe's baseline is 29 `[FAIL]` lines; logits gate is token id and
   top-5 order, not the value; `tinycoder_test` must be 113/96/0 without a model;
   benchmarks are `--n-prompts 512 --n-gen 128 --threads 36`, serial, three runs, no
   env knobs unless labelled; do not commit.
3. **The ledger binds.** These are already measured on this tree and must not be
   re-proposed as new ideas without a new mechanism:
   - shared-staged activation GEMV: 2x slower (occupancy loss + barrier), reverted;
   - split-K on the decode GEMV: no gain; prefetch depth: neutral;
   - stream-K `blocksPerSm` > 1: monotonic worse for PQ2_0 (823/788/771/757/707/631
     at B=1/2/4/8/16/32); `tileX=64` worse; the tile axis is flat-to-worse;
   - fp16 accumulate on the fused EXL3 path: unreachable on sm_75, 2.401 ms against
     1.859 ms for materialise+cuBLAS;
   - per-shape `cublasLt` heuristic pinning: the heuristic returns two candidates with
     identical attributes 35% apart in time.
4. Numerics: the fused EXL3 prefill, the decode GEMV and the MoE path are covered by
   different probes (§7.5), and the decode GEMV currently has **no bit-exact probe**
   reaching it — only the gated question tests. Any work in §7.3 must add one.

## 5. The hardware delta that matters

| property | 2080 Ti (sm_75) | 3090 (sm_86) | what it changes |
|---|---|---|---|
| SMs | 68 | 82 | grid parallelism; per-SM work per launch falls 17% |
| bandwidth | 616 GB/s | 936 GB/s | needs 1.52x the bytes-in-flight per SM to stay at the same efficiency |
| smem / SM | 64 KB | 100 KB | a 2-stage pipeline can hold a 128x128 int8 tile pair; today's single buffer is 57.9 KB |
| smem opt-in / block | 64 KB | 99 KB | `gemmSmemBytes(128,...)` = 57 856 B fits with 41 KB to spare for a second stage |
| warps / SM | 32 | 48 | 8 warps/block at 255 regs = 17% occupancy; the same binary cannot fill an Ampere SM |
| `cp.async` | absent | `cp.async` (`sm_80+`) | removes LDG->register->STS from the staging loop |
| int8 MMA shape | `m8n8k16` | `m16n8k32` (`TC_CUDA_CC_AMPERE`, = 800) | already selected (`Int8Mma.cuh:904`); IMMA count in the sm_86 cubin is 6720 vs 21504 |
| fp16 MMA | `m16n8k8`, fp16-acc 2x | `m16n8k16`, fp32-acc 71 / fp16-acc 142 TFLOPS | the 2080 Ti's fp16-acc is unreachable for other reasons; on Ampere it is the full-rate mode |
| L1 | 96 KB | 128 KB | decoders that read L1-resident activation bytes get more headroom, smem carveout eats it |
| int8 tensor (dense) | 215 TOPS | 284 TOPS | +32% total, **+9% per SM** (both ~2 K MAC/clk/SM): the tensor roof barely moves, so a kernel at 47% of it on the 2080 Ti is not rescued by the 3090 |

Four consequences the design is built on:

- **Latency, not bytes.** 8 warps/block at 255 registers with 5 `__syncthreads()` per
  k-step cannot keep 936 GB/s fed. Ampere's 48 warps/SM is the resource we are not
  using; cp.async is how we use it without spending registers.
- **The int8 prefill GEMM is capped at ~50% of the tensor roof by its own epilogue.**
  From the sm_86 SASS: per 16 k-steps the body issues 256 `IMMA.16832.S8.S8` but
  **1024 `I2FP.F32.S32` + 1024 `FMUL` + 1024 `FFMA`** — three ALU ops per accumulator per
  32-weight k-block to apply the per-block scales (`GemmKernels.cuh:1288`). At the
  documented CC-8.6 conversion rate of 16/clk/SM that is ~1 024 conversion clocks
  against ~1 026 MMA clocks per k-step: a 1:1 ratio, so no amount of dual-issue gets
  this kernel past ~50% of the roof. The staging term adds a second ceiling at ~61%
  (5 170 DRAM clocks against 8 192 MMA clocks per 256-k stage). Because the int8 roof is
  nearly flat per SM across the two cards (+9%), both ceilings transfer to the 3090
  unchanged — which is the honest explanation for a 3090 prefill column that gains
  ~1.3x, not 1.5x.
- **The loader's ALU is the second ceiling.** Measured on sm_75: PTQ1_0's tile loader
  with its base-3 decode removed runs 62.7 TFLOPS, with it 29.5; the PQ2_0 LUT loader
  47.9. This is the same class of defect as the epilogue (arithmetic on the critical
  path of a kernel whose tensor pipes are idle meanwhile), and it is why the prefill
  plan below is ordered "epilogue, then staging, then loader".
- **Decode must not be re-plumbed through smem.** The measured 2x loss stands, and
  Ampere does not change why it lost (`__syncthreads` + occupancy). Decode work is
  grid parallelism, register budget, and per-token fixed cost.
- **The fused EXL3 prefill is worth revisiting.** `docs/specs/exl3-hmma-fused-gemm.md`
  §11a lists five reasons it lost on sm_75: (a) A streamed from L2 and latency-bound at
  16 warps/SM, (b) no `cp.async`, (c) the 64-accumulator cap, (d) 2-way `ldmatrix`
  conflicts from a 16-half pitch, (e) no `m16n8k16`. Ampere removes (b), (e), and with
  100 KB of smem, largely (a) and (c). That is the one Turing negative result whose
  premises this card invalidates.

## 6. Approaches considered

| | approach | trade-off | verdict |
|---|---|---|---|
| A | one shared Ampere staging layer, all quant types at once, validated by SASS invariants | cheapest, covers 12 types in one change, but leaves per-family decode untouched | rejected as the whole answer; its staging primitive is kept as `AsyncStage` (§7.1) |
| B | per-family Ampere kernels, dispatched on device properties | highest ceiling, largest surface, most work that only the rented hour can score | **chosen** |
| C | rent the card first, optimise from ncu stall reasons | best information per dollar, but an hour spent profiling a path that is known-wrong (5 barriers, 1 block/SM) buys little | rejected as the *first* step; becomes the acceptance run of §8.4 |

B is chosen with one piece of A folded in: the staging primitive is written once and
shared by the prefill GEMM and the EXL3 fused kernel (§7.1). That is a code-sharing
decision, not a scope decision — each family still gets its own tuned loop.

## 7. Design

### 7.0 Structure

Three rules apply to every family.

1. **One staging primitive.** `AsyncStage<Stages>` in a new
   `src/cpp/core/cuda/gemm/AsyncStage.cuh`: `prologue/commit/wait` with
   `cp.async.ca.shared.global` on Ampere and a synchronous `LDG + STS` fallback with
   identical addressing below that. Single loop body, `if constexpr` on the staging
   mode — never two copies of the tile loop.
   The arch test itself must be spelled with the tree's named constants
   (`TC_CUDA_CC_AMPERE` = 800, `QuantBlocks.cuh:289-297`; there is no literal
   `>= 800` anywhere in the tree), and it must be an `#if` that defines a
   `constexpr bool`, not an `#if` around a body: `__CUDA_ARCH__` is undefined during
   the host pass, so the token cannot appear inside a construct the host compiler also
   parses. Concretely

   ```cpp
   #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= TC_CUDA_CC_AMPERE
   constexpr bool TC_ASYNC_COPY = true;
   #else
   constexpr bool TC_ASYNC_COPY = false;
   #endif
   ```

   and the loop says `if constexpr (TC_ASYNC_COPY) ... else ...` exactly once. The
   tree has 51 `__CUDA_ARCH__` sites today, all instruction selection inside inline
   helpers (`Int8Mma.cuh`, `QuantBlocks.cuh`, `Gemv.cu`); none of them selects a whole
   staging strategy, which is why this rule is new rather than a precedent to follow.
2. **Occupancy is a compile-time budget, not an accident.** Each kernel carries
   `__launch_bounds__(threads, blocksPerSM)` chosen from a table in the design, and
   `scripts/arch_check.py` fails when the compiled register count breaks the budget
   for either arch.
3. **Dispatch reads the device, not the source.** `Int8Ctx` already queries
   `nsm`/`smpbo`/`cc` (`GemmDispatch.cu:203-231`); new heuristics key on those and
   never on `#if __CUDA_ARCH__` alone, because the host cannot see the arch the kernel
   was compiled for.

### 7.1 Prefill int8 GEMM — the seven GGUF types in §1

Files: `gemm/GemmKernels.cuh` (the loop), `gemm/Int8Mma.cuh` (MMA + fragment copy),
`gemm/QuantBlocks.cuh` (loaders), `gemm/GemmDispatch.cu` (tiles, stream-K, grid).

Types in scope: `q2_k`, `iq2_s`, `iq3_xxs`, `q4_k`, `q8_0`, `pq2_0`, `ptq1_0`. One loop
serves all seven (`int8MatmulProcessTile` is instantiated per type through the macro in
`GemmDispatch.cu:52-95`), which is why this is one design item and not seven.

Current state, verified from source and from the sm_86 cubin:

- One 256-k stage (`INT8_ITER_K=256`, 8 quant blocks) is
  `load_tiles(X) -> stage Y half 1 -> sync -> mma -> sync -> stage Y half 2 -> sync ->
  mma -> sync`: **four barriers per stage**, one buffer, no overlap.
- Shared layout per stage: X 38 912 B + Y 18 432 B + ids 512 B = **57 856 B**;
  254 registers x 256 threads = 65 024 of 65 536. Both limits independently give
  **one block and 8 warps per SM**, i.e. 17% of the 48 warps an Ampere SM can hold.
- The MMA is already the Ampere shape (`mma.m16n8k32.s8` via `Int8Mma.cuh:905`), and
  the A-side already uses `ldmatrix.x4`; the B-side deliberately uses `load_generic`.
- `int8GemmDense*` is **never launched** (only `StreamK` + `Fixup` are), and
  `int8_get_nbytes_shared` is dead code. Both are removal candidates.

Changes, ranked by the ceiling each one moves:

1. **The scale epilogue — the largest term, and the only one that needs a numerics
   decision.** The kernel applies `dW*dA` to every accumulator after every 32-weight
   k-block, which costs 3 ALU ops per accumulator per block (1024 `I2FP` + 1024 `FMUL` +
   1024 `FFMA` per 16 k-steps against 256 `IMMA`). That is a 1:1 clock ratio and a hard
   ~50% ceiling. The fix is to apply the product once per 256-k stage: either requantize
   each weight 32-block to one scale per stage at load time, or fold the per-block
   product into a per-stage scale on the activation side. Both change the arithmetic:
   the requantization error is ~0.4% RMS, far outside the logits gate's ±2e-3 band, so
   **this cannot be done under the existing gate** — it needs the tolerance probe in
   §7.5 and an explicit decision that a GGUF prefill number computed with per-stage
   scales may be published (it would no longer be bit-comparable with libggml's mmq,
   which applies per-block scales). Until that decision, the cheaper non-numeric
   variant is item 4.
2. **Two-stage `cp.async` staging.** Removes the 5 170 DRAM clocks that serialise
   against 8 192 MMA clocks per stage (ceiling 61%). It does **not** fit at today's
   stage size: 2 x 57 344 + 512 = 115 712 B > 101 376 B opt-in. It fits if the stage is
   halved to 128 k (2 x (20 480 + 18 432) + 512 = 78 336 B) or `tile_y` drops to 64
   (2 x (19 456 + 18 432) + 512 = 76 288 B). Both are stride changes: the X pitch of
   76 ints (304 B) and the Y pitch of 36 ints are what make `ldmatrix.x4` and
   `load_generic` conflict-free, so the new pitches must be re-derived and re-checked —
   this is the single most parity-visible item in the design.
3. **The 2-byte weight loads.** `block_q8_0.qs` sits at byte offset 2
   (`QuantBlocks.cuh:196`), so the loader issues **68 `LDG.E.U16` per tile** where 34
   32-bit loads would do: twice the LSU wavefronts per byte. Two options, both parity-
   visible: a load-time padded 36-byte layout, or stage raw 34-byte blocks and
   de-interleave in smem. This is also what makes `cp.async` awkward (a 16-byte chunk
   cannot start at block+2), so items 2 and 3 interact and should land together.
4. **Occupancy, without touching numerics.** `tile_y=64` halves the accumulator to 32
   registers, which fits 2 blocks/SM at 128 registers, and it was measured *worse* on
   sm_75 (B=2 788 vs B=1 823) — so it is re-measured, never assumed. Note the bookkeeping
   the card change forces: `nblocks = nsm * blocksPerSm` becomes 82x1 (or 82x8 for Q8_0)
   instead of 68x1, so the stream-K k-split deepens 1.2x and its fixup traffic
   (`fixBytes = nblocks*tileX*kInt8Y*4`) grows with it, without any measurement yet.
5. **The loader ALU (PTQ1_0/PQ2_0).** (a) mate the packed weights to `cp.async` and
   decode in the fragment path, or (b) materialise int8 once per matrix into a
   no-arithmetic loader — measured 62.7 TFLOPS against 29.5 for PTQ1_0 on sm_75, which
   is the strongest single number in the ledger. The blocker for (b) is a 128-weight
   int8 block with its own stride: a dispatcher change, not a kernel tweak.
6. **Remove the dead paths** (`int8GemmDense*`, `int8_get_nbytes_shared`) so that
   `arch_check.py` audits what actually runs.
7. **Fix the silent-fallback hole.** `gemmPQ2Int8` and friends return `false` into a
   caller that takes a per-token route. Every such site gets a one-line
   `std::fprintf(stderr, "[int8] <reason> ...")` on first occurrence per process and a
   counter surfaced by `--n-gen`/`--n-prompt` output, so "prefill ran at decode speed"
   is visible in a log instead of inferred from a table. This is the only item here that
   could explain the external report's 32x Bonsai prefill collapse, and it costs a day
   less than any of the others.

Acceptance (owned by §8): 2 blocks/SM at 128 registers or 1 block at ≤ 168 registers
with ≥ 4 stages; `LDGSTS` present in the sm_86 SASS and absent in sm_75; the parity
probe's 29-failure baseline unchanged; the logits gate green; and for item 1, the
tolerance probe of §7.5 in place *before* any numeric change lands.

### 7.2 Decode GEMV — what is actually on the default path

The first version of this section assumed the per-type GEMV headers were the hot path.
They are not, and the correction changes the work. Verified dispatch order
(`launchQGemv`, `GpuGemvIQInt8.hpp:962`):

1. `int8Gemv(...)` (`:975`) for `q4_k`/`q5_k`/`q6_k`/`q8_0`/`q2_k`/`q3_k`/`iq2_s`/
   `iq2_xs`/`iq3_s`/`iq3_xxs`, gated only on compute capability >= 7.5
   (`GemmDispatch.cu:207-238`, `:500`) — so it is the path on a 3090 as well, and the
   per-type headers plus every `TINYCODER_Q2K_*`/`Q4K_4XW`/`IQ_4XW`/`Q8K_4XW` knob are
   the `TINYCODER_INT8_GEMM=0` fallback.
2. That path is the vendored `gemvCoreImpl` (`gemv/Gemv.cu:3238`): **one row per
   128-thread block, four warps splitting K, one fp32 accumulator per thread, 36-48
   registers, 384 B of smem, 12 blocks/SM**. Grid = `nrows`, so a 5120-row projection
   launches 5120 blocks: **48 warps/SM = 100% warp occupancy, 62 blocks/SM of grid.**
   For rows 1-4, 7 and the 35B row's projections there is nothing to fix in geometry.
3. The exceptions, i.e. the rows the external report is about, are the ternary and
   EXL3 families, plus one fused kernel:
   - PQ2_0 default `kQGemvPQ2_0T<2,8>` (62 regs, 8 rows/block, 640 blocks, 31 warps/SM);
   - PTQ1_0 default `kQGemvPTQ1_0WarpRowPt<4>` (90 regs, 4 rows/block, 1280 blocks,
     **20 warps/SM = 41%**, register-capped);
   - `kQGemvFusedGU_Q2K` at **243 registers = 1 block and 8 warps/SM**, on row 1's
     gate/up.

So the decode work is not "rewrite the GEMV". It is:

1. **Leave `gemvCoreImpl` alone.** It is already at the card's warp cap on sm_86; its
   ceiling is bytes and per-token fixed cost. Instrument it (item 4), do not touch it.
   The arithmetic that says so: row 3, 2.9 GB at 93.5 tok/s is 10.7 ms/token, while the
   weights alone at 616 GB/s are 4.7 ms — more than half of that token is not weight
   streaming, and none of it is in this kernel.
2. **Close the two register holes.** PTQ1_0 `WarpRowPt<4>` from 90 to <=64 registers
   (`__launch_bounds__(128, 8)`) buys 8 blocks and 32 warps per SM; `FusedGU_Q2K` from
   243 to <=128 buys 2-4 blocks. Both are statically verifiable, both move a published
   row, and the sm_75 measurement of the same change is the cheapest experiment in this
   design.
3. **Grid parallelism matters only for the fallback family.** Its 32-rows/block shape
   gives 160 blocks at rows=5120 (1.95 blocks/SM, 15.6 warps/SM) and 48 blocks at
   rows=1536, i.e. 34 idle SMs. Since that family is only reachable with
   `TINYCODER_INT8_GEMM=0`, the fix is split-K for rows < 8192 behind the same knob, not
   a rewrite of the default path. (The ledger's split-K negative was measured on
   row-parallel shapes; this is the shape class it did not cover.)
4. **Prove which kernel runs, per projection.** `TINYCODER_TRACE_GEMV=1` already exists
   (`GpuGemvIQInt8.hpp:999`). The receipt prints the resolved kernel for every
   projection shape of every row; a published number without that line is a number
   whose kernel is unknown.
5. **Per-token fixed cost is the row.** On sm_75 decode spends 3.4 ms/token outside the
   weight streams (GDN scan 0.85, GDN rest ~1.2, attention 0.76, tail+head 0.54, of a
   15.4 ms token). That cost does not shrink on a 3090 — it is launch- and latency-bound —
   so it becomes the entire budget at 70+ tok/s. §7.4 is not garnish; for rows 1-4 and 7
   it is the whole result.

### 7.3 EXL3 — decode GEMV and fused prefill

Files: `exl3/EXL3Kernels.cu`, the `kTypeEXL3` arm of `launchQGemv`
(`GpuGemvIQInt8.hpp:984`), `docs/specs/exl3-hmma-fused-gemm.md`.

**The decode GEMV is integer-ALU bound, not bandwidth bound, and that inverts the
priority of this section.** The arithmetic, per token on the 27B: trellis traffic is
**5.342 GB** (not 8.6 GB — the `embed_tokens` bf16 table is 3 GB of the checkpoint and
only one row of it is read), so the bandwidth term is 5.71 ms at 936 GB/s. The decode
arithmetic is 215.8 G integer ops at 8.97 Tops/s = **24.0 ms**. Measured 23.6 tok/s on
the 2080 Ti is 76% of the same model's ALU ceiling (32.0 ms) and the external 26.6 tok/s
on a 3090 is 64% of its own (37.6 ms) — a 1.52x-bandwidth card buying 1.13x is the
signature of an ALU-bound kernel.

1. **Fold the `mul1` byte sum into dp4a.** Every tensor in the shipped checkpoints is
   `mul1` (`quantization_config.json`, `EXL3Loader.cpp:687-691`), not the `3INST`
   arithmetic recorded in older notes. `decodeMul1` costs four byte extracts, three
   shifts and two adds per weight on the K=2/K=3 kernel (`exl3GemvV3`), while the K=1
   kernel already folds the identical arithmetic to one `IMAD` + one `IDP.4A` + one
   `HADD2`. The fold is bit-exact — verified exhaustively over all 65536 windows, because
   `0x6400+S` is *rounded* to fp16 rather than exact at that magnitude (25600..26620, 16-ulp
   spacing) and both paths round the same payload; the unsigned `dp4a` overload is required,
   since the signed one changes 94% of the windows. An earlier revision of this document gave
   "exact in fp16" as the reason, which was wrong even though the fold was right (
   fp16 payload) and takes the 3090 ceiling from 41.6 to ~88 tok/s.
   **Implemented and measured**: on the 2080 Ti, `tg128` went **23.53 -> 30.85 tok/s**
   (three runs: 30.90 / 30.85 / 30.75, spread 0.15) with `pp512` unchanged at
   726.8-728.2 against a 726.7 baseline, and the kernel now sits at ~98% of the same
   model's integer-ALU ceiling (32.0 ms predicted, 32.4 ms measured) instead of 76%.
   That is the whole Turing headroom for this kernel: what remains on either card is
   bytes (8.7 ms here, 5.7 ms on a 3090), which is now 3.7x from binding.
2. **Residency.** `exl3GemvV3` sits at 67 registers at KSPLIT=2, which is 3 blocks/SM
   instead of 4 — 24 of 48 warps. Either get it back under 64 or accept 3, but stop
   paying a slot for 3 registers.
3. **`pickKsplit` was calibrated for 108 SMs** (a 272-block "4 per SM" cap and a
   192-block threshold). On 82 SMs that leaves 17% of the block slots unused on the
   shapes it was tuned for. Scale both by `cudaDevAttrMultiProcessorCount`, which
   evaluates to the old integers on the card. **Implemented in this change.**
4. **Barriers.** Two `__syncthreads()` per 128-k chunk (272 of them on `k=17408`)
   restart the prefetch window. Rolling the `wc→wn` load across the chunk boundary is
   worth 10-20% at k-heavy shapes but the K=1 note at `:2205` records this class of
   change producing a rare load-dependent corruption that every short probe passed —
   it needs a real generation as the gate, not the 4-token probe.
5. **Probe.** The claim that no probe reaches the decode GEMV is **true of the gated
   suite only**: `exl3_gpu_parity:217` runs the shipped `gemvFused` at 1e-2 and
   `exl3_k1_probe:533` compares per output element against a double reference at 4e-5.
   What is missing is a *bit-exact* gtest. Add one, and point `exl3_k1_probe` at a K=2
   base so the V3 path — the one the 27B uses — is covered too.
6. **Fused prefill**: keep materialise+cuBLAS as the default and the Turing measurement
   (27.6 vs 34.4 TFLOPS) as the control. On Ampere the bigger lever than `cp.async` is
   `m16n8k16`, which halves both the `HMMA` count and the `ldmatrix` count; the kernel
   is at 128 registers and 2 blocks/SM with `bT` at 8 KB per stage, so a 3-4 stage ring
   fits the 100 KB budget.

### 7.4 Attention, GDN, MoE

Files: `GpuAttentionKernels.hpp`, `GpuQwen35Kernels.hpp`, `forward/GPUComputeMoE.cu`
(arch-1 Qwen3.8-27B prefix), `forward/GPUComputeForward.cu` (arch-2 35B-A3B prefix and
the dense Qwen2.5 loop), `ffn/FFNPersistentKernel.cu`.
`forward/FlashAttentionPrefill.cu` **does not exist** — only a stale unlinked object
survives in the build tree — so nothing in it is reachable and it is not in scope.

1. **Decode attention is not the problem, and the geometry is not what the first draft
   said.** For the 27B, K/V are stored as int8 per (token, kv-head) at 34 560 B/token,
   so a decode step reads `ctx x 33.8 KiB`: 16.8 MB at 512 context (18 us at 936 GB/s)
   and 134 MB at 4096 (143 us) against a ~42 ms step — 0.04% and 0.34%. What the kernel
   loses is *reuse*: the 6 query heads of a kv head each re-fetch the same row, and
   `kWarpAttentionSplit<HD, float, 1>` runs one warp per block (`blk (32,1)`) even
   though `launchDecodeAttentionSplit` already implements a `WSUB` 2/4 merge. That is a
   real 4x warps-per-block change, but the non-bit-identical softmax reduction means it
   needs its own gate.
2. **Prefill attention has no tensor cores on the default path.** The cuBLAS
   `QK^T`/softmax/`PV` path is gated to `architecture==0 && hd==128`
   (`GPUComputeForward.cu:3828-3835`), which is dead for the 27B's `hd=256`; the 27B runs
   the FMA `kQ35WarpAttention<256>` (46 registers, no smem), and the in-tree comment
   records ~25 ms/layer against llama.cpp's ~7. `TINYCODER_ATTN_MMA=1` already selects
   `kQ35PrefillAttnSharedKV<256,8>`; making it the default after a gate is the cheapest
   prefill win in this section.
3. **GDN**: `kQ35GdnStepBatch<128>` at 255 registers is the only 255-register kernel in
   any sm_86 cubin (not an attention kernel), and it is on the **35B-A3B** path, where it
   places 2 blocks/SM = 8 warps and only `nVHeads` blocks. `kQ35GdnWarpScan<128>` (61
   registers, 8 blocks, 32 warps) is the 27B's kernel and is bit-exact with it; promoting
   the warp-scan shape for the MoE decode is a 4x occupancy gain at low risk. Every
   timing in the tree for this is a 2080 Ti number.
4. **MoE**: full-GPU expert placement is already the primary path (`gpu_.upload()` is
   tried first; the LRU cache arms only on failure, `ModelGPU.cpp:683-690`), so there is
   no flag to add. What is missing for "fits on two 3090s" is multi-device support, which
   this design does not include. The published row stays hybrid-only (§12.3).

### 7.5 Probes and gates per family

| family | existing probe | gap |
|---|---|---|
| prefill int8 GEMM | `tools/int8_gemm_probe` is **timing only** (`:33-40`, its `--compare` compares time against the cuBLAS fp16 twin); `unit_tests/Int8PackParityTest` covers the decode `Int8Pack` layout, not this GEMM; end-to-end only via the gated questions | **no numeric test reaches an int8 prefill tile.** Required: a host reference that dequantizes the same rows to fp32, quantizes the activation with the same Q8_1 layout (`ActivationQuant.cu:137-147`), dots in fp32/fp64, and compares per element with a stated tolerance, over shapes that reach every dimension: seqLen spanning a full `tile_x` (512) and a partial one (130), rows not a multiple of 128, K not a multiple of 256, `ntiles % nblocks` both 0 and non-zero (fixup path), and `tileX` in {32,64,128}. Item 1 of §7.1 also needs the probe to *pin the k-order*, because a reordered scale application is not bit-reproducible |
| decode GEMV | gated question tests only | **bit-exact probe missing** (§7.3.1) |
| EXL3 fused prefill | `exl3_gpu_parity` identity probes | Ampere variant needs the same two identity probes |
| attention | `unit_tests/tinycoder_attn_parity`, `tools/attn_split_parity.cpp` | none |
| MoE | gated 35B-A3B questions | n/a |

## 8. Validation without a 3090

This is a first-class part of the design: it is how "very high performance" is kept
honest when the only card in the box is the one the code was already tuned for.

**7.1 Tier 0 — static, per arch, in CI.** New `scripts/arch_check.py` (built on
`scripts/sass_hist.py`) asserts, for every kernel in a named list, for each of
`sm_75/sm_80/sm_86`:

- `LOCAL:0 STACK:0` (no spills) and registers within the budget the kernel declares;
- computed blocks/SM from registers and dynamic smem ≥ the declared minimum;
- on `>= sm_80`: `LDGSTS` count > 0 for kernels that use `AsyncStage`, and
  `cp.async.wait_group` present; on `sm_75`: `LDGSTS` = 0;
- the MMA shape census matches the arch (IMMA 16832 on ≥80, 8816 on 75; HMMA 16816 on
  ≥80);
- dynamic smem request ≤ 99 KB, static ≤ 48 KB.

A kernel that silently stops using async copy, or drifts past the occupancy budget,
fails the build. This is the check that makes the rest of the plan safe.

**7.2 Tier 1 — numerics on the card we have.** The async and sync staging paths share
one loop body, so sm_75 exercises the *same addressing, the same stage indices and the
same barriers*, only the copy instruction differs. On top of that: the 29-failure parity
baseline, the logits gate (`id=198`, top-5 order), a ≥45-token prompt for finite logits,
`scripts/text_gate.sh` per checkpoint, and `tinycoder_bench` 113/96/0.

**7.3 Tier 2 — a cost model, to rank and to pre-register expectations.** From `ncu` on
sm_75 take per kernel: achieved DRAM bytes/s, issued instructions, `smsp__inst_executed`,
stall reasons, warps resident. Predict the 3090 time as
`max(bytes / 936 GB/s / efficiency, instructions / (82 SM * 1.7 GHz * issue rate))`
with the efficiency and issue rate carried over from the sm_75 measurement. Two uses:
rank the work (§10), and **write the acceptance thresholds down before renting**, so the
hour confirms or refutes a prediction rather than generating one. The reporter's numbers
already fit this model (229 GB/s implied by 26.6 tok/s), which is evidence the model has
signal.

**7.4 Tier 3 — one scripted hour.** `scripts/sm86_receipt.sh`, runnable by a stranger.
**Written and dry-run on the sm_75 box** (`SM86_ALLOW_NON_AMPERE=1 SM86_STATIC_ONLY=1`), which
exercises everything except the benchmarks; the script refuses to produce numbers on the wrong
architecture, and its static half already earns its keep — see the two notes below. It automates
items 1-4, 6-8; item 5 (ncu) is left to the operator because `ncu` is not installed here and its
section set is a per-kernel judgement, and the script records whether the instrument existed
rather than silently omitting the pass.

1. assert `CMAKE_CUDA_ARCHITECTURES` contains 86 and `cuobjdump --list-elf` shows
   `sm_86` in the built objects (a stale cache value is a known failure mode);
2. `tools/int8_mma_peak` and an 8 GB copy probe: the card's actual roofs, so every
   ratio later in the log is against measured, not nominal, peak;
3. parity baseline, logits gate, `scripts/text_gate.sh` for every checkpoint;
   **note** — the list of checkpoints is *read from `text_gate.sh`*, not duplicated: a row and
   its gate must live in the same file, or a new row gets published without one;
4. `tinycoder_bench` for every row, 3 runs, interleaved with the reference engine as
   the existing protocol requires; report spread and the memory roof ratio per row;
5. `ncu --set full` on the four kernels that matter (prefill GEMM, decode GEMV, GDN
   scan, attention), plus `--print-summary per-kernel`, to see whether the predicted
   stall reason is the real one;
6. dump the sm_86 SASS census next to the sm_75 one, and the resolved dispatch
   decisions (`tileX`, `blocksPerSm`, `nblocks`, fallback counters, resolved GEMV kernel
   per projection);
7. run the staged trace (`TINYCODER_STAGE_TRACE=1`) on one model **first**, before any
   tuning conclusion is drawn: if the external report's 4x decode gap is real, this names
   the stage it lives in, and it is the cheapest thing on the card to run;
8. emit a markdown fragment in `BENCHMARKS-3090.md`'s column format, with the clock and
   memory temperature recorded.

Two things the dry run already established, worth stating before the card arrives:

- The int8 tensor kernels are **defined in `GemmEntry.cu`**, not in `GemmDispatch.cu` (which only
  declares them and selects tiles). A SASS census taken from `GemmDispatch.o` reads "0 IMMA" on a
  perfectly good build; the receipt points at `GemmEntry.o` and reports the **shape split**,
  because that is the signal: the sm_86 cubin carries 4032 `IMMA.16832` (m16n8k32, Ampere) and
  2688 `IMMA.16816` (the Turing form), so the new shape is selected and the old one is still
  emitted for the paths that keep it.
- `LDGSTS` count on sm_86 is **0** today, i.e. no `cp.async` anywhere in the tree. That is the
  baseline the staging rewrite (§7.1) has to move, and the receipt records it either way so a
  before/after diff is one line.

## 9. Acceptance criteria

| tier | must hold |
|---|---|
| 0 | every listed kernel passes `arch_check.py` on 75/80/86; build exit 0 |
| 1 | parity 29 failures; logits `198` + same top-5; text gate green on every checkpoint; suite 113/96/0 |
| 2 | predicted 3090 numbers written into the design's table before the hour, per row |
| 3 | on the 3090: decode ≥ 70% of the measured memory roof per row, prefill ≥ 55% of the measured int8 roof, gates green, no silent fallback counter; the numbers land in `BENCHMARKS-3090.md` |

## 10. Order of work

Each stage is independently shippable and independently gated.

1. **Instrumentation first** (no speed claims): fallback counters + `arch_check.py` +
   the missing decode-GEMV parity probe. Everything after this is measurable.
2. **Decode register holes and dispatch trace** (§7.2.2, §7.2.4). Cheapest, measurable on
   sm_75 today, and two of the three items sit on rows the external report is about.
3. **Prefill `AsyncStage` + barriers + double buffer** (§7.1.1-2), then the loader-ALU
   work (§7.1.3).
4. **EXL3 Ampere fused prefill** (§7.3.2), last because it is the only item that needs
   the card to judge.
5. **GDN/attention occupancy** (§7.4), which becomes the decode row's budget at 70+ tok/s.

## 11. Risks and their detectors

| risk | detector |
|---|---|
| async copy issued then immediately waited (correct, not async) | Tier 0 `LDGSTS`/`wait_group` counts + Tier 3 stall reasons |
| register blowup from deeper pipelining collapses occupancy | Tier 0 blocks/SM, before and after |
| two copies of one loop drift apart | single body + `if constexpr`; review rule: no `#if __CUDA_ARCH__` around a loop body |
| sm_86 SASS is right but the *dispatch* picks the old path | Tier 0 dispatch print + Tier 3 fallback counters |
| rented-hour variance (clock, memory temperature) | report clock; 3 runs; interleaved reference engine |
| a probe passes while generation is garbage | the ≥45-token prompt and the text gate, which exist for exactly this |

## 12. Decisions taken

1. **Approach B** — per-family Ampere kernels — with the staging primitive from A shared
   between prefill GEMM and the EXL3 fused kernel. (§6)
2. **Benchmarks are per card.** The existing table was renamed
   `BENCHMARKS-2080TI.md` and a sibling `BENCHMARKS-3090.md` was created, empty by
   construction: no row is published until §8.4 has run and its gate has passed. A row's
   baseline column is measured on the same card as its TinyCoder column.
3. **The Ornith-1.5-35B-A3B row is hybrid-only** in both files, and says so in the table
   and in a note under it. No fully GPU-resident number is published: the model is
   34.4 GB, it does not fit on either card, and the row's two columns run the same
   expert-offload mode.
4. **The numerics trade in §7.1 item 1 is the one thing not yet decided.** Applying the
   quant scales once per 256-k stage instead of once per 32-weight block is the only
   change that moves the ~50% epilogue ceiling, and it changes the arithmetic
   (~0.4% RMS requantization error). It is therefore held out of the first
   implementation pass: everything else in §7.1 lands first, and this one waits for a
   tolerance probe and an explicit yes, because it makes a GGUF prefill number
   non-comparable with libggml's mmq.
5. **One rented session, after implementation**, not a mid-work exploratory hour. The
   card is rented by the partner; the agent cannot rent one. What makes that acceptable
   without risking a wasted session is §8 tiers 0-2, and a cost-model prediction written
   down before the session so the session confirms or refutes it.

## 13. What is left to the builder

- The exact stage count and smem accounting for §7.1.1: the budget is 99 KB and today's
  tile is 57 856 B; how many stages fit for each of the seven types is an
  implementation measurement, not a design constraint.
- The rows-per-block crank values in §7.2.1 per family, and which families need split-K
  at 82 SMs (the design says "only where rows cannot fill the grid"; the builder derives
  the numbers from the projection shapes).
- Whether the EXL3 fused Ampere kernel beats materialise+cuBLAS: the design keeps both
  and gates the default on the rented hour.
