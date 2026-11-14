# AGENTS.md

Working notes for anyone (human or agent) touching this repo. Everything here was learned the
expensive way; each rule exists because ignoring it cost a day or produced a false result.

## What this is

A from-scratch C++20/CUDA inference engine. No inference library is linked: GGUF and EXL3 dequant,
the int8 MMA GEMM, the attention and recurrent (gated-delta-net) kernels, and the samplers are all
implemented in this tree. The exception is cuBLAS, used as a plain GEMM.

When adding a kernel, the question to ask is not "does it work" but "how do I prove it agrees with
the reference bit-for-bit or within a stated tolerance, and what would a wrong answer look like?".

## Environment

| | |
|---|---|
| GPU | RTX 2080 Ti, 11 GB, **sm_75**, ~616 GB/s |
| CPU | 2x Xeon E5-2686 v4, 36 threads used by the benchmarks |
| Build dir | `build-1.0` |
| Reference model | `/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw` |

sm_75 constraints that repeatedly matter:

- Native fp16 tensor MMA is **`mma.sync.aligned.m16n8k8`**. `m16n8k16` is sm_80+ and ptxas rejects
  it. `wmma::m16n16k16` already lowers to four `HMMA.1688`.
- fp32-accumulate HMMA runs at **half** rate versus fp16-accumulate. Published tensor peaks for this
  card: **53.8 TFLOPS** with fp32 accumulate, **~107.6 TFLOPS** with fp16 accumulate (nominal 1.55 GHz;
  clocks boost, so a back-to-back microbenchmark measures 46.6-68 TFLOPS for fp32-acc).
- No `cp.async`. Global -> shared staging is synchronous.
- 64K registers per SM. Budget accordingly: an `m16n8` fp32 accumulator is 4 registers, so 16
  tiles x 4 = 64 registers leaves little room. **1024-thread blocks cannot launch** with a
  register-hungry kernel; prefer more work per warp over more warps.
- No `ldmatrix` bank-conflict-free path from a 16-half shared pitch; expect 2-way conflicts unless
  the staging layout is swizzled.

## Build rules

1. **Always name the core target explicitly** when a `.cu` changed:

   ```bash
   cmake --build build-1.0 --target tinycoder_core -j 20
   ```

   Building only a tool target (`--target tiny_logits_probe`, `--target tinycoder_bench`) leaves the
   core object stale and silently measures the *old* kernel. This produced repeated false
   regressions and one false "improvement" before it was understood.

2. **Check the exit status, never a grep window over the build output.** Piping a failing build
   through `grep " error" | head -5` shows five unrelated warnings, and the error lines sit beyond
   the cut. A stale binary then produces believable numbers. If you must filter, keep the exit code:

   ```bash
   cmake --build build-1.0 --target tinycoder_core -j 20 || { echo BUILD FAILED; exit 1; }
   ```

3. Rebuild the tool you intend to run after the core, in the same command:

   ```bash
   cmake --build build-1.0 --target tinycoder_core tiny_logits_probe tinycoder_bench -j 20
   ```

## Benchmarking

**The only comparable protocol** (default environment, no env vars):

```bash
./build-1.0/benchmarks/tinycoder_bench \
  --model /data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw \
  --n-prompts 512 --n-gen 128 --reps 1 --threads 36
```

- `pp512` is prefill, `tg128` is decode. Report both, three runs, spread included.
- **`--n-gen 8` is not comparable** with any number taken at `--n-gen 128`. Mixing them caused two
  false regression alarms.
- **Run benchmarks serially.** Two processes on this GPU contend for bandwidth and produce numbers
  that describe neither. If another session is measuring, wait; do not "just do a quick run".
- Report the environment you ran in. Numbers with env knobs set are not the shipped default and must
  be labelled as such.

### Physically impossible results are bugs

Before believing any speedup, check it against the hardware:

```
implied TFLOPS  = 2 * params * tok_s / 1e12       # prefill
implied GB/s    = weight_bytes * tok_s / 1e9      # decode
```

For this card the published tensor peaks are ~53.8 TFLOPS (fp32 accumulate) and ~107.6 TFLOPS (fp16
accumulate); memory peak is ~616 GB/s. A fused EXL3 config once reported **2,085 tok/s** — 112
TFLOPS, above even the fp16-accumulate peak — because it was silently skipping work.
The arithmetic caught it; enthusiasm would not have. The same check later caught a mid-session
artefact at ~2,081 tok/s.

## Correctness verification

Run these in order after every change to a compute path. They are cheap and each catches a
different class of mistake.

1. **Kernel parity** — `./build-1.0/unit_tests/exl3_gpu_parity`

   Baseline is **29 `[FAIL]` lines**, all in the pre-existing `b=4.0` / K=4 "unsupported geometry"
   class, and the count is identical with and without the fused path. Compare the *count*, not a
   feeling. The valuable checks are the identity-probe ones:

   ```
   [PASS] fused gemm vs raw matrix (maxAbsDiff=0.000000)
   [PASS] fused gemm multi-strip (256 rows) vs raw matrix (maxAbsDiff=0.000000)
   ```

   The multi-strip case was added because a `seqLen=16` probe only ever exercises the first
   m-strip, so a broken strip index in a multi-strip kernel passes it. If you add a tiling
   dimension, add a probe that reaches the new dimension.

2. **Logits** — `./build-1.0/unit_tests/tiny_logits_probe <model> "def hello():`

   Default path must print `top1: id=198 logit= 17.612242`. The earlier
   `16.667089` / fused `16.666790` pair was measured with a corrupted
   gated-delta-net state (`kQ35GdnWarpScan` dropped the broadcast of the
   `<state_row, k>` reduction, so the persistent scan state grew without bound);
   with the state fixed the value moves to 17.612242, and exllamav3 1.6.6 on the
   same three token ids prints `id=198 logit=19.2969` with the same top-5
   ordering, i.e. the new value is the closer of the two. Do not "restore" it.

3. **Long-prompt logits — do not skip this.** The 4-token probe above stayed green for an entire
   day while the engine emitted `NaN` for prompts of 39 tokens or more and generated nothing but
   `!!!!`. A short prompt is not a smoke test for a 27B model. Always also check a **>=45-token
   chat-style prompt** for finite logits, and check that generation answers a factual question.

4. **Suite** — `./build-1.0/unit_tests/tinycoder_test` must report **90 tests, 73 passed, 0 failed**
   without a model (the rest skip because they need a checkpoint). With `TINYCODER_MODEL_PATH` set, the gated question tests run instead of skipping; they
   are the ones that catch real inference bugs:

   ```bash
   TINYCODER_MODEL_PATH=/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw \
     ./build-1.0/unit_tests/tinycoder_test --gtest_filter='FullQuestions*'
   ```

   They print the generated text, so read it — a passing keyword check and a plausible-looking
   completion are not the same thing.

## Measuring the baseline engine (exllamav3)

The comparison column must be measured on this box with this checkpoint, not quoted. Wheels are
enough; no source build:

```bash
python3 -m venv /tmp/exl3venv
/tmp/exl3venv/bin/pip install torch exllamav3==1.6.6
PATH=/tmp/exl3venv/bin:$PATH /tmp/exl3venv/bin/python scripts/exl3_baseline_bench.py <checkpoint>
PATH=/tmp/exl3venv/bin:$PATH /tmp/exl3venv/bin/python scripts/exl3_ask.py <checkpoint> "What is the capital of France?"
```

- `ninja` must be on `PATH` for their JIT build (the wheel depends on it but does not add it to PATH).
- `model.load(max_chunk_size=N)` and `Cache(model, max_num_tokens=N)` must agree, and `N` must be a
  multiple of 256, or it asserts / demands more VRAM than the card has.
- `scripts/exl3_ask.py` is the sanity check that the baseline engine is really inferring, and a way
  to prove a suspect checkpoint is fine.

## EXL3 specifics worth knowing before touching the kernels

- Layout is `(k/16, n/16, 16*K)`; **one trellis tile is one 16x16 (k,n) weight tile**, which is one
  GEMM k-step. Decode is arithmetic, not a table gather:
  `x*89226354+64248484 -> (x & 0x8fff8fff) ^ 0x3b603b60 -> f(lo)+f(hi)`.
- The decomposition is `W^T x = Dsv Hn What^T Hk Dsu x`. **RAW mode (default) keeps both Hadamards
  and the scales off the weights**, applied to activations instead. That is what makes the batched
  path cheap, and it also means the decoded weights are large: measured O(4,000) to O(60,000).
- Consequence: **fp16 accumulation is unsafe here without a power-of-two pre-scale.** At magnitude
  2^12 the fp16 ulp is 3.9, i.e. the size of the answer. exllamav3 can use fp16 accumulate because it
  applies `svh` to the weights before the MMA (and its path is guarded to sm_86 anyway — Turing is
  unvalidated there). If you want that 2x, pre-scale by `2^-s` per chunk and compensate exactly in
  the fold.
- The fused prefill kernel and the decode GEMV are different code paths with different coverage:
  prefill has bit-exact parity probes; **the decode GEMV has none reaching it** except the gated
  tests. That asymmetry is where a bug can live for weeks.

## Operational rules

- **Never `pkill -f tinycoder_bench`.** The pattern matches the shell running the command and kills
  your own session. Use `pgrep -x tinycoder_bench` and kill the PID.
- **`compute-sanitizer` hangs on the 27B model.** Do not use it there; use a small model or a
  single-projection harness for memory checking.
- **Do not commit.** The user commits; leave the tree dirty and describe what changed.
- **Do not add date-stamped or changelog paragraphs to README.md.** No "Re-measured <date>", no "an
  earlier edition recorded X". Numbers belong in the table; the procedure belongs in the method
  section; history belongs in the ledger.
- Keep the method for any published number in the repo as a runnable script, not in `/tmp`.
- One session at a time on the GPU, and one editor at a time per file. Two agents editing the same
  `.cu` produce builds and measurements of unknown provenance.

## Evidence ledger

`.superpowers/sdd/2026-11-11-prefill-parity/progress.md` records what was measured, what was tried
and rejected, and why. Before repeating an experiment, grep it — several levers here (tile sizes,
block counts, fused tile configs, fp16 accumulate without pre-scaling) have already been measured
and falsified, and the negative results include the reason.

Design docs live in `docs/`. The most recent is `docs/specs/exl3-hmma-fused-gemm.md`, including a
section on why the fused kernel loses to materialise+cuBLAS on this hardware and what a rewrite would
need.

## A closing note on reporting

A truthful "slower, and here is the measurement that shows why" is worth more than an unverified
win. Most of the durable knowledge in this repo came from negative results that were reproduced and
written down: fused tiling ceilings, dead int8 block-count levers, the round-trip removal that costs
more than it saves. Report the number, the protocol, the spread, and what limits it.
