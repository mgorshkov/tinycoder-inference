# Q8_0 expert prefill: per-thread memory parallelism, then dual-socket locality

Date: 2026-11-27
Status: awaiting approval

## Intent and why

Prefill is CPU-expert-stream bound. `pp512` is 4.04 s, of which ~3.6 s is
streaming 34.2 GB of native Q8_0 expert weights at **9.5 GB/s** on 18
node-0 threads; the GPU trunk is 0.18 s and the rest ~0.2 s. The same
access pattern reads at **19.4 GB/s** at 18 threads on node 0 and
**35.8 GB/s** interleaved across both sockets (measured with a pure-read
harness), so the expert kernel runs at ~49% of the machine's own ceiling
for that pattern.

The goal is to close that gap, in two ordered steps so that the cheap one
either delivers or is falsified before the structural one is funded.

## Goals

- **A** — reduce the *dependency distance* between a thread's weight loads
  so the stream stops stalling, measured against a like-for-like read
  probe (see "Decisions from the builder check"). The stream sits at
  9.5 GB/s against a **19.4 GB/s** single-socket ceiling for the same
  pattern, so A's own ceiling is ~239 pp512; the acceptance bar is a
  **1.15x (>=146 pp512)** with the gates green — confirmed 2026-11-27;
  **B is funded only if A passes this bar**.
- **B** — make both sockets memory-local: half the expert bytes resident
  in node-1 RAM, two 18-thread groups, each streaming its own node. Its
  ceiling is the **both-socket** read rate (35.8 GB/s, ~400 pp512), so B
  keeps headroom even if A lands near its own 19.4 GB/s ceiling; target
  2x (250 pp512).

## Non-goals

- No requantisation: the experts stay native Q8_0 on the CPU.
- No new GPU expert kernels. The GPU-offload lever is measured dead
  (36 GFLOPS effective at m~16 vs 290 GFLOPS on the CPU; an 8x grouped
  GEMM would be needed just to break even).
- No decode regression: `tg128` must stay at or above 21.3 tok/s.

## Anti-goals

- A change that looks faster because work was dropped (the resident path
  taught this: 64.4 pp was real, 75.3 pp was real, both still losses).
- Pinning threads across sockets without node-local data: measured -20%.

## Constraints

- 2x Xeon E5-2686 v4. node0 = CPUs 0-17 (SMT 36-53), node1 = 18-35
  (SMT 54-71). 125 GB RAM, weights in page cache, THP = madvise.
- Build stays host-agnostic: fixed per-ISA flags plus runtime dispatch.
- Verification for any published number: `scripts/text_gate.sh`, the
  gtest suite, `moe_parity`, and `scripts/ornith_ab.sh` (3 interleaved
  pairs). Benchmarks run serially.
- Repository rule: the user commits; the tree is left dirty.

## Approaches considered

| # | Approach | Upside | Cost | Verdict |
|---|---|---|---|---|
| A | Per-thread MLP: two independent expert-tile streams per thread | 10-25% | one kernel change, one build | **chosen first** |
| B | Dual-socket node-local expert duplication | up to 2x | second pool, split store, +17 GB RAM | **chosen second** |
| C | Grouped int8 GEMM on the GPU (llama's `mul_mat_id`/MMQ shape) | ~10% of bytes | 8x kernel improvement to break even | rejected, measured |
| D | VRAM-resident expert slice | ~10% of bytes | measured 75.3 pp vs 126.7 | rejected, measured |
| E | Bigger tiles / deeper prefetch / more threads / interleaved copies | 0 | measured neutral or negative | rejected, measured |

A is chosen first because it is contained and falsifiable in one build,
and because it answers the question B depends on: is the stream
latency-starved (then B's extra sockets help) or already at a shared
saturation point (then B would not)?

## Decisions from the builder check

The reviewer's four own-able gaps are settled here as the builder's call;
the fifth is now confirmed by the human partner: A must reach 1.15x
(>=146 pp512) with the gates green, and B is funded only if it does.

1. **Stream distance, not stream count.** Counting streams has been
   measured: `TILE_ROWS=8` is the same trade (more streams, less 2-token
   weight reuse) and came out 3% slower, and the 4x2 tile already spends
   8+2+2 of 16 ymm. So A changes *distance*, not count: keep `q8DotTile<4,2>`
   exactly as it is (same arithmetic, same reuse, same instructions) and
   prefetch the **next task's** rows while the current task runs, so the
   DRAM sees a second stream from a distant address. A control arm with
   the cross-task prefetch disabled separates the effect.
2. **B is prefill-only.** The interleaved anonymous store was already
   measured at tg128 15.8 (from 24.2), because an expert-alternating store
   makes ~half of every decode token's reads cross-socket. B therefore
   keeps the current node-0 page-cache mapping for `seqLen == 1` and uses
   the duplicated store only for prefill. Both copies coexist: 34.2 GB of
   page cache + 34.2 GB anonymous = 68 GB of the 125 GB.

## The how — B

- Build one anonymous 34.2 GB expert store, `mbind`-ed in **2 MiB-aligned
  chunks** rather than per expert: one expert's slice is 512 rows x 2176 B
  = 1.06 MiB, so a 2 MB huge page spans ~1.9 experts and per-expert
  `mbind` would split the mapping into sub-2 MB VMAs that khugepaged
  cannot collapse. Chunks of 64 experts (68 MB per matrix) are assigned
  alternately to node 0 and node 1, so every expert is entirely local and
  every page is a huge page. The kernel takes a per-chunk base table
  instead of `W_q8_0 + (expert*rowsPerExpert + rowStart)*rowBytes`.
- Two pools of 18 threads, one pinned per node. Groups whose expert is
  node-0 local go to pool A, the rest to pool B; routing is uniform, so
  the split is balanced by construction. Both pools write disjoint pair
  indices of the same `partial` buffer.
- The second pool runs from a helper `std::thread` while the calling
  thread runs the first pool's region, so the two regions overlap.

## The how — A

Today the 4x2 tile holds 8 accumulators and issues one dependent weight
load chain per thread; the same tile is replayed once per member pair, so
the stream goes quiet between passes. A keeps that tile untouched and adds
a **cross-task prefetch**: when a worker starts task *i*, it issues
`_mm_prefetch` for task *i+1*'s first rows and its scale tile, so the next
task's DRAM stream is already in flight while the current one computes.
Nothing in the accumulator structure, the 2-token reuse, or the fp32
accumulation order changes. Control arm: the same build with the
cross-task prefetch compiled out.

## The comparator must be committed

The 19.4 GB/s figure came from a throwaway `/tmp` harness, and the repo
rule is that no published number comes from `/tmp`; worse, that harness
read each tile **once**, while the kernel replays a tile once per member
pair, so it was never a like-for-like ceiling. A therefore starts by
committing `tools/membw_probe.cu` with two modes — read-once and
replay-per-member — and reports A against the replay ceiling. Without
that, a null result would be uninterpretable.

## What is left to the builder

- A: prefetch distance (rows ahead) and whether the next task is the same
  group's next tile or the next worker chunk; both are measurements.
- B: chunk size (64 experts proposed), pool startup order, and placement
  verification with `move_pages` before any number is reported.
- Ordering: A's measured result against the replay ceiling decides
  whether B is started at all.
