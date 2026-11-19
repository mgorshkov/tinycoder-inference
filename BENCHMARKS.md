# Benchmarking: GPU Performance (TinyCoder vs llama.cpp)

## GPU Comparison (TinyCoder vs llama.cpp / exllamav3)

Columns follow the benchmark defaults: `tinycoder_bench --n-prompts 512 --n-gen 128 --threads 36` (**pp512 / tg128**). Every row below is a **CUDA build** (`cmake -B build -DCMAKE_BUILD_TYPE=Release`, which enables the offload engine whenever `nvcc` is installed; `-DENABLE_CUDA=OFF` produces a CPU-only binary that these numbers do not describe). `--gpu` forces the offload path and `--cpu` takes the CPU baseline of the same run; a CPU-only binary answers `--gpu` with "needs a CUDA build" and exit 2 rather than an unknown-option error. GGUF baselines are `llama-bench -p 512 -n 128 -t 36 -r 1`, run **interleaved** with the TinyCoder run (three alternating pairs, so clock drift affects both columns equally, means reported; the Ornith-35B row adds `-ncmoe 99`, the same expert-offload mode the TinyCoder column uses); the EXL3 baseline is a local exllamav3 1.6.6 run measured with the same pp512/tg128 protocol — see [How the exllamav3 EXL3 baseline was measured](#how-the-exllamav3-exl3-baseline-was-measured); the two Ternary-Bonsai rows instead use PrismML's llama.cpp fork, because stock llama.cpp cannot load either Bonsai packing — see [How the Prism Bonsai baseline was measured](#how-the-prism-bonsai-baseline-was-measured). Warm mean-of-reps, RTX 2080 Ti. Bold marks the faster engine in a column. Figures are the mean of three alternating interleaved pairs (spread = (max - min)/2); the
dense-prefill attention path computes QK^T, causal softmax and PV in fp16 via cuBLAS, which moves the
attention output by maxAbsDiff 0.0146 against the previous fp32 warp kernel -- a numerics change
licensed by the gates, not a refactor, and measured as byte-identical greedy text on a 512-token prompt
with the old and new path in the same binary.

| Model | File size | GPU mode TinyCoder | **TinyCoder pp512** | **TinyCoder tg128** | Baseline engine | **Baseline pp512** | **Baseline tg128** |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B `q2_k` | 0.7 GB | full offload | **12,239.8 tok/s** | **325.9 tok/s** | llama.cpp | 7,679.2 tok/s | 321.6 tok/s |
| Qwen2.5-Coder-1.5B `iq3_xxs` | 0.9 GB | full offload | **12,243.4 tok/s** | **273.8 tok/s** | llama.cpp | 10,007.0 tok/s | 262.2 tok/s |
| Qwen2.5-Coder-7B `iq2_s` | 2.4 GB | full offload | **2,765.7 tok/s** | 87.7 tok/s | llama.cpp | 2,482.0 tok/s | **96.5 tok/s** |
| Qwen2.5-Coder-7B `iq3_xxs` | 2.9 GB | full offload | **3,060.5 tok/s** | 93.5 tok/s | llama.cpp | 2,938.0 tok/s | **97.4 tok/s** |
| Ternary-Bonsai-2-27B `pq2_0` (Prism) | 6.7 GB | full offload | **880.6 tok/s** | **51.73 tok/s** | PrismML llama.cpp (fork) | 878.5 tok/s | 50.36 tok/s |
| Ternary-Bonsai-2-27B `ptq1_0` (Prism) | 5.5 GB | full offload | **880.7 tok/s** | 51.73 tok/s | PrismML llama.cpp (fork) | 848.1 tok/s | **65.29 tok/s** |
| Ornith-1.5-9B `q4_k_m` | 5.9 GB | full offload | 2,560.4 tok/s | 82.3 tok/s | llama.cpp | **2,666.4 tok/s** | **84.1 tok/s** |
| Ornith-1.5-35B-A3B `q8_0` → `q4_k` experts | 34.4 GB | hybrid (layer arena, grouped int8) | **176.3 tok/s** | **53.6 tok/s** | llama.cpp `-ncmoe 99` | 132.2 tok/s | 17.1 tok/s |
| Qwen3.8-27B EXL3 `1.60bpw` | 8.6 GB | full offload | **727.5 tok/s** | 23.60 tok/s | exllamav3 1.6.6 | 719.5 tok/s | **23.76 tok/s** |

## How the exllamav3 EXL3 baseline was measured

The baseline is a real run on the reference box (RTX 2080 Ti, 11 GB, driver 595.91), on the **same
checkpoint** as the TinyCoder column: `Qwen3.8-27B EXL3 1.60bpw`. No source build is needed —
exllamav3 publishes a `py3-none-any` wheel that JIT-compiles its CUDA extension on first import:

```bash
python3 -m venv /tmp/exl3venv
/tmp/exl3venv/bin/pip install torch exllamav3==1.6.6   # torch 2.15.0, CUDA 13 wheels
PATH=/tmp/exl3venv/bin:$PATH \
  /tmp/exl3venv/bin/python scripts/exl3_baseline_bench.py <checkpoint-dir>
```

`ninja` must be on `PATH` for the JIT build (it ships with the wheel's dependencies, but is not on
the default `PATH`). The first import compiles the extension for the local architecture.

Loading the model needs two constraints satisfied at once, or it either asserts or asks for more
VRAM than an 11 GB card has — `max_chunk_size` at load and `Cache(max_num_tokens=...)` must be set
together, and the cache size must be a multiple of the page size (256):

```python
config = Config.from_directory(path)
model  = Model.from_config(config)
cache  = Cache(model, max_num_tokens=1024)          # multiple of 256
model.load(progressbar=False, max_chunk_size=1024)  # must match the cache shape
```

The harness mirrors `tinycoder_bench`'s protocol rather than using exllamav3's own `eval/perf.py`,
so the two columns are comparable: a 512-token prefill timed with `model.prefill`, then 128 greedy
decode steps timed individually (`model.forward` + `GreedySampler`), one process per run.

Measured over three runs:

| run | pp512 | tg128 |
|---|---:|---:|
| 1 | 720.3 tok/s | 23.79 tok/s |
| 2 | 718.2 tok/s | 23.74 tok/s |
| 3 | 720.1 tok/s | 23.75 tok/s |
| **mean** | **719.5 tok/s** | **23.76 tok/s** |

The tg128 figure includes Python per-token overhead (sampler call plus tensor concatenation), so it
is a floor rather than a best case. These runs are interleaved with the TinyCoder runs
(`scripts/exl3_ab_bench.sh`, three alternating pairs), not measured engine-by-engine: separate passes
let clock drift bias the comparison -- measuring each engine in its own pass moved the baseline by
0.6% prefill and 0.3% decode here, which is larger than the difference being claimed.
On the interleaved protocol TinyCoder leads prefill, 727.5 against 719.5 (+1.1%, both spreads 1.0),
and **exllamav3 leads decode**, 23.76 against 23.60 (+0.7%, spreads 0.03-0.05). The decode lead is
small but outside the noise, so it is not reported as a tie.


## How the Prism Bonsai baseline was measured

Stock llama.cpp cannot load either Bonsai packing: `PQ2_0` is PrismML's own ggml type 142 and
`PTQ1_0` its type 143, and both need the model's Hadamard-activation runtime. The canonical runtime
is PrismML's fork of llama.cpp, which carries both types and builds unmodified for sm_75. It was
measured **here, on this card, on the same checkpoint**, interleaved with the TinyCoder run (three
alternating pairs, means, spread = (max - min)/2, GPU verified empty before every run):

```bash
git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp ~/llama-prism
cmake -B ~/llama-prism/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build ~/llama-prism/build --target llama-bench -j 36
~/llama-prism/build/bin/llama-bench -m <file> -p 512 -n 128 -t 36 -r 1

./scripts/bonsai_baseline_bench.sh          # both files, three interleaved pairs, raw logs per run
```

`-p 512 -n 128 -t 36 -r 1` mirrors every other row's `llama-bench` column, with the defaults
(f16 KV cache, no speculative drafter). **The 96-100 tok/s figures published for this family are
MTP-drafting numbers**; the fork's own plain-decode receipt is 68.3 tok/s on an RTX 4070 at ~90% of
that card's bandwidth (`bonsai-ada-surgery`, `docs/speed/decode-efficiency.md`), and this card
measures 50.36 on `pq2_0` = 345 GB/s of packed weight, 56% of the 616 GB/s roof. The gap to Ada is
sm_75's missing async copy, not a kernel defect: the reference's own mat-vec is at 90% of peak there.
TinyCoder's `pq2_0` decode is 2.7% ahead of that (51.73 against 50.36) on a q8_1 activation and
`dp4a` over the same packed weight -- 6.42 GB of folded weight in 13.2 ms of GEMV, i.e. ~460 GB/s in
the kernel -- and its remaining per-token cost is not the mat-vec: 4.4 ms of it is the GDN scan, the
per-layer norms, the folds and the LM head. Prefill runs int8 tensor cores straight off the packed
trits, as the fork does: the 2-bit codes are decoded inside the GEMM's tile loader, so no fp16 twin
is materialised, and one fold serves each shared activation (gate/up, Q/K/V), so both prefill rows
lead or match the fork: 880.7 against 848.1 on `ptq1_0`, and `pq2_0` leads by 2.1 tok/s (880.6
against 878.5, a margin that is inside the spread rather than a win).

`PTQ1_0` is 1.75 bpw against `PQ2_0`'s 2.125, so the reference decodes it faster (65.3 vs 51.7) on
14% fewer bytes. TinyCoder's `ptq1_0` row reads the same ternary weights after a **lossless
load-time transcode to the PQ2_0 block layout** (both packs hold identical trits and group scale --
the transcoded file reproduces the native `pq2_0` file's top-5 logits to six decimals), so its
per-token traffic equals the `pq2_0` row's and the two TinyCoder rows agree -- including prefill, which
runs the same native int8 path on the transcoded blocks. The `ptq1_0` decode gap is therefore the
transcode's own 1.19 GB/step: 5.66 GB native against 6.85 transcoded, which is 0.83, and the measured
ratio is 0.79. A native 28 B/128 decode would land near 62 tok/s at the current per-byte rate; the
first attempt at one measured 4.6x slower (a per-code map lookup in the inner loop) and was not
shipped, so the transcode stays and is also why the `ptq1_0` column exists at all.

## Context window: what each checkpoint is granted, and what a long window costs

The default is still **2048 tokens** (`--ctx` unset): a request is only honoured when one is given, so
every row in the table above is unchanged and the three long-context rows were re-measured at the
default window to confirm it (9B 2554.9 / 83.44, Bonsai `pq2_0` 878.7 / 51.28, 27B EXL3 727.4 / 23.52
against the published 2560.4 / 82.3, 880.6 / 51.73, 727.5 / 23.60 -- all inside the run-to-run spread).
Passing `--ctx N` asks the planner for N; it grants the largest window that fits the measured free
VRAM/RAM minus the weights, the session's fixed ~1.4 GiB of scratch, and the K/V cost of the precision
it picks (f32 K/V + an int8 decode mirror, then f32, then int8), and it refuses the load below 1024
tokens. What it grants on an 11 GB card:

| Checkpoint | K/V at the grant | Largest window granted (`maxAllowed`) | What leaves the ladder first |
|---|---|---|---|
| Ornith-1.5-9B `q4_k_m` | int8 | **196,352** | f32 K/V (52,480), then f32+mirror (41,984) |
| Ternary-Bonsai-2-27B `pq2_0` | int8 | **64,256** | f32 K/V (16,640), then f32+mirror |
| Qwen3.8-27B EXL3 `1.60bpw` | int8 | **12,288** (default 2048 stays f32+mirror) | f32 K/V (2,560), then f32+mirror |

Throughput at a granted window, same protocol as the table above (`--n-prompts 512 --n-gen 128 --reps 1
--threads 36`, one process at a time; `maxAllowed` alongside each grant):

| Checkpoint | `--ctx` | Granted | `maxAllowed` | K/V | pp512 | tg128 |
|---|---|---|---|---|---|---|
| Ornith-1.5-9B | 8192 | 8192 | 196,352 | f32+mirror, 652 MB | 2535.9 tok/s | 83.36 tok/s |
| Ornith-1.5-9B | 65536 | 65536 | 196,352 | int8, 1120 MB | 2531.2 tok/s | 79.88 tok/s |
| Ternary-Bonsai-2-27B | 8192 | 8192 | 64,256 | f32+mirror, 1294 MB | 878.8 tok/s | 51.34 tok/s |
| Ternary-Bonsai-2-27B | 64000 | 64000 | 64,256 | int8, 2109 MB | 873.2 tok/s | 48.49 tok/s |

A 64k window costs **2-6% of decode and nothing measurable in prefill** at these sizes (prefill here
processes 512 tokens, so its per-token cost does not see the window; the long-prompt prefill rate does
-- the f32 store without the mirror (`--ctx 45056`, a rung only that request can take) prefills a
44,892-token prompt at 533 tok/s because the causal attention over the window is quadratic, against
2535 tok/s at pp512). The int8 store is what makes the window fit, and its cost is one
quantisation pass per K/V row: at the same 2048 window on Bonsai, f32 gives `id=198 logit=17.122244`
and int8 `id=198 logit=17.109360` -- same top-1, same top-3 order, a 0.0129 absolute (7.5e-4 relative)
shift, i.e. quantisation noise rather than a different distribution.

Sizes this does **not** reach on an 11 GB card, computed from the measured per-token costs (9B
17,920 B/token int8, Bonsai and the 27B 34,560 B/token) plus the weights and the fixed overhead:

| Window | 9B (5.6 GB weights) | Bonsai (6.7 GB) | 27B EXL3 (8.6 GB) |
|---|---|---|---|
| 262,144 (256k) | 11.3 GiB | 16.4 GiB | 18.2 GiB |
| 1,048,576 (1M) | 24.4 GiB | 41.7 GiB | 44.6 GiB |

so 256k needs roughly a 12-24 GiB card for these checkpoints and 1M needs 24-48 GiB (or four cards
with layer sharding, which this engine does not implement). Positions past the 262,144 tokens the
checkpoints were trained for also need RoPE scaling, which the engine does not have: the grant is a
capacity, not an accuracy claim, and retrieval accuracy beyond the trained length is untested here.

Gates for these rows: `./scripts/context_gate.sh` (six rows, exit 0 -- the 9B answers a needle from
token 0 at 64k, Bonsai is grant-checked because it does not answer a needle at any length, the 27B
grant row confirms its default 2048 is untouched), and `./scripts/text_gate.sh`, where every checkpoint
these rows use answers (1.5B, 7B, 9B, Bonsai `pq2_0`/`ptq1_0`, 27B EXL3). The 35B-A3B row is the one
failure in that script and it is not this feature's: it emits `endahendah...` from the first token at
BASE `d79d73f` too, byte-identical logits, so its published 176.3 / 53.6 row still has no working text
gate.

---

Back to the [README](README.md).
