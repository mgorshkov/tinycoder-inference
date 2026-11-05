# Spec: HMMA.16816 rewrite of the fused EXL3 prefill GEMM

Status: ready to implement. Owner: next session.
Target: `src/cpp/core/cuda/exl3/EXL3Kernels.cu`.

## 1. Problem

The EXL3 prefill path materialises each trellis into an fp16 `[k][n]` matrix and calls
cuBLAS: **649 tok/s** on Qwen3.8-27B EXL3 1.60 bpw (pp512), 0.90x of exllamav3's 716.6.
That path moves 3x the necessary bytes per forward — it reads the 5.3 GB trellis, writes
5.3 GB of fp16, then reads those 5.3 GB back in the GEMM.

The fused kernel exists (`gemmFusedRawT`) and removes the round trip, but it tops out at
**418.7 tok/s**. Root cause is the MMA stage: `wmma::mma_sync` with `m16n16k16` and fp32
accumulate on sm_75 does not reach what cuBLAS achieves with hand-scheduled `mma.sync`.

A `wmma` fragment is opaque; the compiler picks register allocation and scheduling, and it
cannot interleave the integer decode work with the tensor-core issue stream the way the
tuned kernels do. Closing the 1.55x gap requires the native instruction and manual control.

## 2. Goal

A fused kernel that decodes the trellis and issues `mma.sync.aligned.m16n8k16` directly,
with the decode overlapped against MMA issue, beating **649 tok/s** at pp512 on the 27B and
keeping bit-exact output versus the materialised raw matrix.

Non-goal: decode (tg128). That path is separate and untouched.

## 3. Instruction set

**CORRECTION (found during implementation):** `mma.sync.aligned.m16n8k16` requires
sm_80+. ptxas rejects it for sm_75 — "Feature '.m16n8k16' requires .target sm_80 or
higher". Turing's native fp16 tensor MMA is **`m16n8k8`**, which is exactly what
`wmma::m16n16k16` already decomposed into (4x `HMMA.1688`). The rewrite therefore keeps the
instruction mix and fixes the *scheduling*: no opaque `wmma`, no uncoalesced
`load_matrix_sync` global A loads, no smem epilogue bounce.

sm_75 provides, and this spec uses:

- `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` — the native fp16 tensor-core op.
- `ldmatrix.sync.aligned.m8n8.x4.b16` (A), `.x4.trans` / `.x2` (B) — shared-memory to
  fragment loads.

Do not use `wmma::` anywhere in the new MMA stage.

## 4. Fragment layouts (lane t in 0..31)

These are the fragile part. A wrong mapping yields a kernel that runs fast and computes
garbage — see section 9 for the gate that catches it.

A, 16x16 fp16 row-major, 4 x `.b32` = 8 halfs per lane:

```
a0,a1 -> row t/4,       cols (t%4)*2 + {0,1}
a2,a3 -> row t/4 + 8,   cols (t%4)*2 + {0,1}
a4,a5 -> row t/4,       cols (t%4)*2 + 8 + {0,1}
a6,a7 -> row t/4 + 8,   cols (t%4)*2 + 8 + {0,1}
```

B, 16x8 fp16, 2 x `.b32` = 4 halfs per lane:

```
b0,b1 -> rows (t%4)*2 + {0,1}, col t/4
b2,b3 -> rows (t%4)*2 + 8 + {0,1}, col t/4
```

C/D, 16x8 fp32, 4 registers per lane:

```
c0,c1 -> row t/4,      cols (t%4)*2 + {0,1}
c2,c3 -> row t/4 + 8,  cols (t%4)*2 + {0,1}
```

An `m16n16` output is two `m16n8` accumulators side by side (n-halves 0..7 and 8..15).

## 5. What to keep, what to replace

Keep `decodeChunk`. It is validated: the parity harness reports
`fused gemm vs raw matrix (maxAbsDiff=0.000000)` on every case, and it writes the tile in
`[k][n]` layout as 16x16 halfs into the double-buffered `bT`.

Replace only the MMA stage (`mmaChunk`) and the epilogue store.

Mapping: one decoded `[k][n]` tile (16 k-rows x 16 n-cols) feeds two `m16n8k16` calls —
columns 0..7 and 8..15. A is the activation tile, B is the decoded weight tile in the
`row.col` convention (A row-major over m, B col-major over k).

## 6. Pipeline

```
for each k-chunk c:
    decode chunk c+1 into bT[1-cur]      (integer ALU, all warps)
    ldmatrix A fragments for chunk c
    issue mma for chunk c against bT[cur] (tensor cores)
    __syncthreads()
```

This is the existing `DECODE_FIRST` ordering; preserve it and preserve the double buffer.
The intent is that the decode of chunk c+1 issues while chunk c's MMAs are still draining,
so the ALU and tensor-core units overlap.

Staging A through shared memory is required for `ldmatrix`; A is reused across the block's
n-tiles, so stage it in smem once per chunk rather than reloading per n-tile.

## 7. Register budget — the hard constraint

Each `m16n8` accumulator costs 4 registers. Two wmma configs already failed here:

| config | accumulator regs | result |
|---|---:|---|
| MLOOP=2, NT=4 | 64 | correct, 418.7 |
| MLOOP=4, NT=4 | 128 | **garbage** |
| MLOOP=2, NT=8 | 128 | **garbage** |
| 1024-thread w32/w16 | n/a | cannot launch (64K regs/SM) |

Rules for the new kernel:

- Keep total accumulators at or below **64 registers/thread** (4 x 4 m-strips x 4 n-tiles x 2 halves
  is already 128 — do not repeat that).
- Suggested shape: 8 warps, each warp owns 2 m-strips x 2 n-tiles x 2 n-halves = 8 accumulators
  = 32 registers, leaving room for A/B fragments and the decode's working set.
- Verify with `cuobjdump -res-usage` that `LOCAL:0 STACK:0` (no spills) and that the register
  count leaves at least one block resident per SM.

## 8. Interface

Add a new kernel and keep the wmma version as the fallback:

- `TINYCODER_EXL3_TILE=hmma` (or a dedicated `TINYCODER_EXL3_MMA=hmma`) selects it.
- Correctness fallback: if the launch fails for any reason, return failure so
  `gemmExl3Batch` disables the fused path and uses materialise+cuBLAS. A fast-but-wrong
  projection must never reach the model — the existing fail-safe in `gemmExl3Batch` is what
  makes this safe.
- Default stays materialise+cuBLAS until the new kernel beats 649 tok/s.

## 9. Verification gates, in order

1. **Parity, bit-exact**: `./build-1.0/unit_tests/exl3_gpu_parity` must print
   `[PASS] fused gemm vs raw matrix (maxAbsDiff=0.000000)` on all four cases. Run this after
   *every* layout change.
2. **Logits**: `TINYCODER_EXL3_FUSED=1 ./build-1.0/unit_tests/tiny_logits_probe <27B> "def hello():"`
   must give `top1: id=198 logit=16.66-16.67` (reference: `16.665943` from the materialised path,
   `16.667089` from the cuBLAS batch path).
3. **FLOP sanity**: implied TFLOPS = `2 * 27e9 * tok_s / 1e12`. Compare against the measured
   ceiling for the chosen accumulate type: ~46.6-68 TFLOPS for f32 accumulate on this card
   (~27 for fp16 accumulate). Anything above the ceiling is skipping work. This check caught a
   bogus 2,085 tok/s (112 TFLOPS) from the MLOOP=4 config, and a mid-rewrite ~2081 tok/s
   artifact. NOTE: the original formula in this spec was dimensionally wrong (it divided by
   tok_s, which yields 5.4e10 rather than a rate).
4. **Speed**: `tinycoder_bench --n-prompts 512 --n-gen 128 --reps 1 --threads 36`, three runs.
   Target: beat 649.2.
5. **Regression**: `./build-1.0/unit_tests/tinycoder_test` stays 72 OK / 0 failed, and the
   default path (no env vars) still measures 649 +- 3 tok/s and logs `16.667089`.

## 10. Build and measurement notes

- `.cu` edits require `cmake --build build-1.0 --target tinycoder_core -j 20` **explicitly**;
  building only a tool target leaves the core object stale and produces false readings.
- Never `pkill -f tinycoder_bench` (the pattern matches the shell itself); use `pgrep -x`.
- `compute-sanitizer` hangs on the 27B model — do not use it.
- Model: `/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw`.

## 11. Risks

- **Layout errors are silent and fast.** Mitigation is gate 1 plus gate 3.
- **Register pressure.** The n-halves double the accumulator count versus the wmma version;
  the strip/tile shape in section 7 is chosen to stay under 64.
- **Diminishing returns.** If after a correct implementation the kernel lands between 419 and
  649, the round-trip removal was still not worth it on this hardware and the default should
  stay on materialise+cuBLAS; report the number rather than shipping a slower path.
- **Estimate**: one session for implementation, one for tuning, 3-6 build/measure iterations
  on the layouts alone.

## 11a. OUTCOME (measured, this hardware)

Implemented as `gemmFusedHmmaT<8, 2, 4, 4, true, 0, FULL>` (256x64 C tile, 64 acc regs,
REG:128 LOCAL:0 STACK:0, 2 blocks/SM). Results, 3 runs back-to-back, protocol `--n-gen 128`:

| path | mean pp512 | implied TFLOPS |
|---|---:|---:|
| materialise + cuBLAS (default) | 636.6 (645.2 verified later) | 34.4 |
| fused native HMMA | 510.8 | 27.6 |

Fused is **1.25x slower**. The §11 fallback branch is therefore taken: on sm_75 the round-trip
removal is not worth it, and the default stays on materialise+cuBLAS.

Why it loses: eliminating the materialise recovers at most 21% of the default's GEMM time, so a
fused kernel would need ~80% of peak; it reaches 51% because (a) A is streamed from L2 once per
64-column block (~179 GB/forward, more than the B traffic saved) and is latency-bound at 16
warps/SM; (b) sm_75 has no `cp.async`; (c) the 64-accumulator cap forces a choice between 2x
decode amplification or 2x A traffic; (d) B's 16-half smem pitch gives 2-way ldmatrix conflicts;
(e) no m16n8k16 on Turing. A perfect fused GEMM would still be only ~18% ahead of cuBLAS.

Still-open ideas, by expected value: register-level trellis prefetch 1-2 chunks ahead (erases the
remaining 11% decode interference); 128-column blocks with 2 warps per row-strip (halves A
traffic at the same accumulator budget).

## 12. Reference values

| quantity | value |
|---|---|
| materialise + cuBLAS (default) | 649.2 pp512 |
| best fused wmma (MLOOP=2, NT=4) | 418.7 pp512 |
| previous fused wmma (MLOOP=1) | 318.5 pp512 |
| exllamav3 baseline | 716.6 pp512 |
| FP16 tensor-core peak, sm_75 (2080 Ti) | ~27 TFLOPS |
