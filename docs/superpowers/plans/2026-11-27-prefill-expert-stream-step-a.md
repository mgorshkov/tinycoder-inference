# Prefill expert stream — Step A: cross-task prefetch

> **For agentic workers:** implement task by task. Steps use `- [ ]`.

**Goal:** Raise the Q8_0 expert prefill stream from 9.5 GB/s to >= 10.9 GB/s
(1.15x, pp512 >= 146) by prefetching the *next* task's weight rows, without
touching the arithmetic.

**Architecture:** `matMulVecBatchQ8_0Groups_AVX2` hands each worker one
(group, 4-row tile) task at a time; while that task's member loop runs, the
DRAM stream for it is the only one in flight. The change issues
`_mm_prefetch` for the next task's rows and scale tile at the top of a task so
the next stream is already moving. The 4x2 tile, its 8 accumulators, its
pre-converted scales and its fp32 accumulation order are untouched.

**Tech Stack:** C++20, AVX2/FMA/F16C intrinsics, CMake, gtest.

**Spec:** `docs/superpowers/specs/2026-11-27-prefill-expert-stream-design.md`

## Global Constraints

- Experts stay native Q8_0 on the CPU; no requantisation.
- Bit-exactness: integer dot unchanged, fp32 accumulation order unchanged.
- Decode floor: tg128 must stay >= 21.3 tok/s.
- Acceptance bar: pp512 >= 146 with `text_gate.sh` PASS, suite 0 failed,
  `moe_parity` 0 outside tolerance.
- Do not commit; leave the tree dirty.
- Benchmarks serially; A/B via `scripts/ornith_ab.sh` (3 interleaved pairs).

## Review Focus

- A task at the end of the work queue has no "next task": the prefetch must
  clamp, not read past `numTasks`.
- The next task may belong to a different group (different `base`), so the
  prefetched address must be computed from that task's own base, not the
  current one.
- The prefetch must not extend the loop's live range enough to spill the
  8 accumulators (the reason an earlier tile-widening attempt lost 3%).

---

### Task 1: Cross-task prefetch with a control arm

**Files:**
- Modify: `src/cpp/core/SIMDMatMulVecAVX2.cpp` (groups kernel, phase-2 lambda)
- Test: existing gates (`text_gate.sh`, `moe_parity`, suite)

**Interfaces:**
- Consumes: `q8DotTile<RN,SN,PRECONV>(rowBase, xqP, xdP, blocksPerRow, outPtr, dRows)`
- Produces: no new API; behaviour gated by `TINYCODER_MOE_XPREFETCH` (0 = off).

- [ ] **Step 1: Add the env control arm**

`TINYCODER_MOE_XPREFETCH=0` disables the cross-task prefetch (default on), read
once into a function-local `static const bool`.

- [ ] **Step 2: Prefetch the next task at task entry**

In the phase-2 lambda, after `task` is known, compute the next task index
(`min(task + chunk, numTasks - 1)`) and prefetch, for each of its rows, the
first block plus a second block one cache line deeper, and its `dTile` rows
when `blocksPerRow <= 64`.

- [ ] **Step 3: Build**

`cmake --build build --target tinycoder_core tinycoder_bench -j 20 || exit 1`

- [ ] **Step 4: Measure both arms, 2 runs each**

Default and `TINYCODER_MOE_XPREFETCH=0`, `--n-prompts 512 --n-gen 16`.

- [ ] **Step 5: Gates**

`./scripts/text_gate.sh ornith/Ornith-1.5-35B-A3B`, `tinycoder_test`,
`moe_parity`.

- [ ] **Step 6: A/B against llama**

`scripts/ornith_ab.sh ... -ncmoe 99`; record pp512/tg128 means and spread.

### Task 2: Decide B

- [ ] If pp512 >= 146 with gates green: B is funded; write its plan.
- [ ] Otherwise: record the negative result in the ledger and stop.
