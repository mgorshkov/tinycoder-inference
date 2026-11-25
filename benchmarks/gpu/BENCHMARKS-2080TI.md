# Benchmarking on RTX 2080 Ti: GPU Performance (TinyCoder vs llama.cpp)

Per-card benchmark files: this one is the **RTX 2080 Ti** (11 GB, sm_75, ~616 GB/s), the
reference box every number below was measured on. The Ampere sibling is
[BENCHMARKS-3090.md](BENCHMARKS-3090.md). A number is only comparable within the file for
the card it was measured on, and each file's baseline column must be measured on that
same card.

## GPU Comparison (TinyCoder vs llama.cpp / exllamav3)

Columns follow the benchmark defaults: `tinycoder_bench --n-prompts 512 --n-gen 128 --threads 36` (**pp512 / tg128**). Every row below is a **CUDA build** (`cmake -B build -DCMAKE_BUILD_TYPE=Release`, which enables the offload engine whenever `nvcc` is installed; `-DENABLE_CUDA=OFF` produces a CPU-only binary that these numbers do not describe). The rows were measured with the AVX2 matvec and sampler kernels compiled in (`ENABLE_AVX2=ON`, `ENABLE_AVX512=OFF` on the reference box); a tree whose cache holds `ENABLE_AVX2=OFF` compiles `SIMDMatMulVecAVX2.cpp` to an empty translation unit and runs scalar CPU kernels, so its CPU-path numbers are not comparable — `scripts/build.sh` prints the resolved set after configure for exactly this reason. `--gpu` forces the offload path and `--cpu` takes the CPU baseline of the same run; a CPU-only binary answers `--gpu` with "needs a CUDA build" and exit 2 rather than an unknown-option error. GGUF baselines are `llama-bench -p 512 -n 128 -t 36 -r 1`, run **interleaved** with the TinyCoder run (three alternating pairs, so clock drift affects both columns equally, means reported; the Ornith-35B row adds `-ncmoe 99`, the same expert-offload mode the TinyCoder column uses; the Inkling-2-Mini row's baseline is llama.cpp CUDA (`-ngl 99`, 2048-token context) from the Unsloth `inkling2-loop-legacy-layout` tree (`dd330cc` plus the looped-layer fix), because stock llama.cpp mainline cannot load that expanded-layout checkpoint, and `scripts/inkling_ab.sh` runs those pairs); the EXL3 baseline is a local exllamav3 1.6.6 run measured with the same pp512/tg128 protocol — see [How the exllamav3 EXL3 baseline was measured](#how-the-exllamav3-exl3-baseline-was-measured); the two Ternary-Bonsai rows instead use PrismML's llama.cpp fork, because stock llama.cpp cannot load either Bonsai packing — see [How the Prism Bonsai baseline was measured](#how-the-prism-bonsai-baseline-was-measured). Warm mean-of-reps, RTX 2080 Ti. Bold marks the faster engine in a column. Figures are the mean of three alternating interleaved pairs (spread = (max - min)/2); the
dense-prefill attention path computes QK^T, causal softmax and PV in fp16 via cuBLAS, which moves the
attention output by maxAbsDiff 0.0146 against the previous fp32 warp kernel -- a numerics change
licensed by the gates, not a refactor, and measured as byte-identical greedy text on a 512-token prompt
with the old and new path in the same binary.

| Model | File size | GPU mode TinyCoder | **TinyCoder pp512** | **TinyCoder tg128** | Baseline engine | **Baseline pp512** | **Baseline tg128** |
|---|---|---|---|---|---|---|---|
| [Qwen2.5-Coder-1.5B `q2_k`](https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q2_K.gguf) | 0.7 GB | full offload | **12,274.6 tok/s** | **347.4 tok/s** | llama.cpp | 7,708.6 tok/s | 321.5 tok/s |
| [Qwen2.5-Coder-1.5B `iq3_xxs`](https://huggingface.co/ggml-org/Qwen2.5-Coder-1.5B-IQ3_XXS-GGUF/resolve/main/qwen2.5-coder-1.5b-iq3_xxs-imat.gguf) | 0.9 GB | full offload | **12,243.4 tok/s** | **273.8 tok/s** | llama.cpp | 10,007.0 tok/s | 262.2 tok/s |
| [Qwen3.5-4B `Q4_K_M`](https://huggingface.co/LocalScribe/Qwen3.5-4B-Q4_K_M.gguf/resolve/main/Qwen3.5-4B-Q4_K_M.gguf) | 2.6 GB | full offload | **4,212.5 tok/s** | **135.6 tok/s** | llama.cpp | 4,123.2 tok/s | 133.6 tok/s |
| [Qwen2.5-Coder-7B `iq2_s`](https://huggingface.co/ThomasBaruzier/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf) | 2.4 GB | full offload | **2,765.7 tok/s** | 87.7 tok/s | llama.cpp | 2,482.0 tok/s | **96.5 tok/s** |
| [Qwen2.5-Coder-7B `iq3_xxs`](https://huggingface.co/ThomasBaruzier/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-IQ3_XXS.gguf) | 2.9 GB | full offload | **3,060.5 tok/s** | 93.5 tok/s | llama.cpp | 2,938.0 tok/s | **97.4 tok/s** |
| [Ternary-Bonsai-2-27B `pq2_0` (Prism)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf) | 6.7 GB | full offload | **880.6 tok/s** | **51.73 tok/s** | PrismML llama.cpp (fork) | 878.5 tok/s | 50.36 tok/s |
| [Ternary-Bonsai-2-27B `ptq1_0` (Prism)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf) | 5.5 GB | full offload | **880.7 tok/s** | 51.73 tok/s | PrismML llama.cpp (fork) | 848.1 tok/s | **65.29 tok/s** |
| [Ornith-1.5-9B `q4_k_m`](https://huggingface.co/bartowski/Ornith-1.5-9B-GGUF/resolve/main/Ornith-1.5-9B-Q4_K_M.gguf) | 5.9 GB | full offload | 2,575.7 tok/s | **85.8 tok/s** | llama.cpp | **2,652.0 tok/s** | 84.2 tok/s |
| [Ornith-1.5-35B-A3B `q8_0` → `q4_k` experts](https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-A3B-Q8_0.gguf) | 34.4 GB | **hybrid only** (experts in RAM, layer arena) | **133.4 tok/s** | **24.10 tok/s** | llama.cpp `-ncmoe 99` | 132.2 tok/s | 17.1 tok/s |
| [Inkling-2-Mini `iq2_xxs`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-IQ2_XXS.gguf) | 8.5 GB | full offload | **1200.0 tok/s** | **47.0 tok/s** | llama.cpp CUDA (`-ngl 99`, ctx 2048) | 883.6 tok/s | 44.3 tok/s |
| [Qwen3.8-27B EXL3 `1.60bpw`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/SC_1.60bpw_H3) | 8.6 GB | full offload | **725.8 tok/s** | **29.83 tok/s** | exllamav3 1.6.6 | 716.8 tok/s | 23.67 tok/s |

The Qwen3.5-4B row and the re-measured 1.5B / 9B rows were taken on the same reference
box (driver 595.91.07) with three alternating pairs of `llama-bench -p 512 -n 128 -t 36 -r 1
-ngl 99` and `tinycoder_bench --n-prompts 512 --n-gen 128 --threads 36`; means over the
three pairs, spread (max - min)/2. The 27B EXL3 TinyCoder cells were re-checked the same
way (three serial runs: pp512 725.1 +/- 1.3, tg128 29.79 +/- 0.04) and unchanged.

The Ornith-1.5-35B-A3B row describes **one configuration only**: the hybrid mode in which
the expert FFNs live in system RAM and both columns run the same expert-offload mode
(`-ncmoe 99` on the llama.cpp side). It is not a statement about the model on a fully
GPU-resident path, and it must not be compared against one; at 34.4 GB the model does not
fit on this card, so no fully resident number exists for it here.

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
On the interleaved protocol TinyCoder leads both columns: prefill 725.8 against 716.8 (+1.3%,
spreads 1.2 and 1.6), and decode **29.83 against 23.67 (+26.0%**, spreads 0.06 and 0.07). The decode
margin is the `mul1` dp4a fold in `exl3GemvV3` (see the EXL3 section of AGENTS.md): the same
arithmetic as one `IMAD` + one `dp4a` + one `HADD2` instead of four byte extracts, three shifts and
two adds, bit-identical. A repeat session on the same binary measured 30.77 / 23.63, so the
between-session spread on this row is about 3%, larger than the within-session spread of 0.06.


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
-- all inside the run-to-run spread of the numbers published for them at the time; the 27B decode
figure is 23.52 because that run predates the `mul1` dp4a fold in `exl3GemvV3`).
Passing `--ctx N` asks the planner for N; it grants the largest window that fits the measured free
VRAM/RAM minus the weights, the session's fixed ~1.4 GiB of scratch, and the K/V cost of the precision
it picks (f32 K/V + an int8 decode mirror, then f32, then int8), and it refuses the load below 1024
tokens. What it grants on an 11 GB card:

| Checkpoint | K/V at the grant | Largest window granted (`maxAllowed`) | What leaves the ladder first |
|---|---|---|---|
| [Ornith-1.5-9B `q4_k_m`](https://huggingface.co/bartowski/Ornith-1.5-9B-GGUF/resolve/main/Ornith-1.5-9B-Q4_K_M.gguf) | int8 | **196,352** | f32 K/V (52,480), then f32+mirror (41,984) |
| Ternary-Bonsai-2-27B `pq2_0` | int8 | **64,256** | f32 K/V (16,640), then f32+mirror |
| [Qwen3.8-27B EXL3 `1.60bpw`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/SC_1.60bpw_H3) | int8 | **12,288** (default 2048 stays f32+mirror) | f32 K/V (2,560), then f32+mirror |

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
these rows use answers (1.5B, 7B, 9B, the 35B-A3B, Bonsai `pq2_0`/`ptq1_0`, 27B EXL3): the whole
script passes.

---

## Inkling-2-Mini (IQ2_XXS, canonical and diagnostic protocols)

IQ2_XXS (8.49 GB) is the only published Inkling quant that fits the 11 GB card
with the K/V cache. `TINYCODER_INKLING_GPU=1` selects the full-device CUDA
graph; the rest is the default environment, 36 threads. The canonical protocol
(`--n-prompts 512 --n-gen 128`) and the short-run diagnostic both run here:

| engine | pp512 | tg128 | pp16 | tg16 |
|---|---|---|---|---|
| TinyCoder GPU graph | **1200.0** | **47.0** | **233.3** | **51.65** |
| TinyCoder CPU (36 threads, `TINYCODER_NUMA=off`) | - | - | 1.90 | 1.92 |
| llama.cpp CPU (`-ngl 0`) | 333.8 | 4.41 | 23.7 | 4.91 |
| llama.cpp CUDA (`-ngl 99`, ctx 2048) | 883.6 | 44.3 | 177.2 | 35.52 |

All tok/s, one session: the mean of three alternating interleaved pairs, with
the range of the three (`scripts/inkling_ab.sh`, text gate first). TinyCoder GPU 1199.10–1201.80 / 46.97–47.01 / 232.90–233.60 / 51.62–51.71;
llama.cpp CUDA 883.52–883.76 / 44.27–44.35 / 176.52–178.53 / 34.77–35.91;
llama.cpp CPU 330.94–336.36 / 3.97–4.92 / 23.34–24.16 / 4.56–5.10; TinyCoder CPU
1.90–1.90 / 1.91–1.93. TinyCoder CPU was not run at pp512/tg128; its 16-token
diagnostic is already 2.6x behind llama.cpp CPU decode and 12x behind on
prefill, and the prefill path is token-at-a-time.

The GPU graph is now **faster than llama.cpp CUDA on both canonical columns**:
prefill 1200.0 against 883.6 (1.36x) and decode **47.0 against 44.3 (1.06x)**. At tg16 it is
ahead of llama.cpp CUDA (51.65 vs 35.52); at pp16 it beats llama.cpp CPU 9.9x
(233.3 vs 23.7) and beats llama.cpp CUDA (233.3 vs 177.2). Prefill at the
canonical 512 is 1.36x **ahead** (1200.0 vs 883.6): the m=512 outer batch removed three of the four
weight re-reads, and the remaining cost is the skewed expert MMA
(the expert path is ~57% of the 426.7 ms pass). The pp16/tg16 columns are
diagnostics: a 16-token prompt is not a prefill benchmark, and tg16 is measured
after a 16-token prefill while tg128 follows 512 tokens. They must not be
compared across prompt lengths.

Physics check on the decode number: the graph uploads 7.66 GiB (8.22e9 bytes) of
weights, so 47.0 tok/s moves 386 GB/s, 63% of this card's ~616 GB/s. The figure
is inside the bandwidth budget, not beyond it, which is what a decode graph with
no skipped work should look like. Batched prefill reads a weight block once for
512 tokens in one pass, so 1200.0 tok/s is ~19 GB/s of weight traffic; prefill is
ahead of the CUDA reference, and the remaining headroom is expert-MMA/attention
occupancy, not bandwidth.

The decode path gained three changes. The long-context attention kernel was the
previous bottleneck (35.6 ms of a 65 ms token at tg128): it now runs one block
per query head with 2–8 warps splitting the key range (was one block for all 24
heads), computes the QK dot once per key with an online softmax (was three
passes), and stages the relative-bias row in shared memory; the depth-mix kernel
runs one warp per state and accumulates in float instead of fp64. That moved
tg128 16.34 → 31.86 (+95%) and pp512 23.2 → 33.0 (+42%). Second, the dense
projections whose type the engine's int8 Q8_1 GEMV covers (Q4_K/Q5_K/Q6_K/Q8_0
and IQ2_XS/IQ2_S/IQ3_S/IQ3_XXS) now use that kernel instead of the Inkling float
decoder. On this checkpoint that is attention Q/K/V plus relative bias
(Q4_K/Q5_K) and attention output (Q8_0): attn_proj 4.6 → 1.93 ms, attn_out
3.4 → 1.94 ms. It moved tg128 31.86 → 36.7 (+15%), pp512 33.0 → 39.1 (+18%),
tg16 33.93 → 39.0 (+15%) and pp16 34.7 → 41.1 (+18%).

Third, the routed experts now use an expert-indexed int8 GEMV. The float
decoder read expert rows through a device-side index, so the existing int8
kernel did not apply; the new `gemvExpertCoreImpl` keeps that indexing (one
block per output row, four warps, weight row
`rowIdx[outRow/rowsPer]*rowsPer + outRow%rowsPer`, activation row
`outRow/rowsPer`) and swaps the arithmetic underneath for the Q8_1 dot. It
covers Q4_K/Q5_K/Q6_K/Q8_0/IQ2_XS/IQ2_S/IQ3_S/IQ3_XXS, plus a Q2_0 dot written
for this path (`block_q2_0` is 18 B per 64 weights — fp16 scale and 2-bit codes,
value `d*(code-1)` — so the block is read as two virtual 32-weight halves and
quantised to Q8_1). On this checkpoint that is 11 of the 24 blocks' gate/up
(IQ2_XS, IQ2_S): `expert_gate_up` 6.23 → 4.00 ms (-36%). It moved tg128
36.7 → 40.3 (+10%), pp512 39.1 → 41.8 (+7%), tg16 39.0 → 43.6 (+12%) and
pp16 41.1 → 44.0 (+7%). The later 16-token batched prefill path raised pp512 to
68.7 and pp16 to 67.6 without changing decode. The 128-token layer-major pass
then raised pp512 to 108.3 and pp16 to 81.8, still bit-exact. The expert-MMA
decode, batched-GEMV spill fix and one-warp sink projection raised pp512 to
168.2 and pp16 to 106.8, still bit-exact at 512 tokens. The Q8 tensor-core
default then raised pp512 to 247.5 and pp16 to 131.7; the race fixes keep it
deterministic across fresh processes. The m=512 outer batch (one layer-major
pass, compact counting-sort expert gather, fixed controller/router kernels)
raised pp512 to 406.2 and pp16 to 155.2; the expert weight-stationary retile then raised pp512 to 423.4; IQ1_S/NVFP4 in the MMA raised the MoE to 500 ms and pp512 to 557.9. The deterministic 32-token k-tile MMA and aux fixes then raised pp512 to 604.5 and pp16 to 218.8; the expert MMA is now at llama.cpp's level (~333 ms) and the remaining 1.45x was the non-expert half. The flash attention, batched relbias, register-stationary controller and tiled short-conv then raised pp512 to 834.0. The final MoE-auxiliary shape fixes (gather one block per compact slot, parallel group build, warp-per-chunk Q8_1; md5 unchanged) cut the pass 613.9→428.6 ms and raised pp512 to 1194.7, ahead of the CUDA reference on prefill for the first time, with decode 40.4; the decode-fused controller/attention then raised it to 44.7, and the NVFP4/IQ grid and short-matrix GEMV fixes raised it to **47.0**, so both canonical columns are ahead of llama.cpp CUDA (prefill 1.36x, decode 1.06x). The
token-major exact path remains bit-identical; moving 23 layer executions to the
IQ1_S/NVFP4 Q8_1 MMA changes the fast path's 512-token top-1 on 41/512 positions
and top-5 set on 119/512 (was 27/111), min cosine 0.9916. That is the accepted
Q8 floor; the official final-logit band stays 0.41927 and 10/10 fresh processes
answer Paris.

The default path is **llama.cpp-class quantised numerics**: Q8_1 activations in
the dense/expert MMQ. The final-logit gate passes (top-1, top-5 set, band
0.41927), and the text gate is deterministic: the test now samples greedily by
default (`--sample` restores stochastic, `--seed` pins it), and **10/10 fresh
processes generate 40 tokens and answer Paris**. The earlier Paris 1/3 run was
the test's unseeded sampler (`seed = 0` → random_device), not the engine. The
racecheck audit did find and fix a real batched `rmsInvShared` barrier race and
the four mix softmax warp-wide shared RMWs; after that, 20 fresh fast-path
prefill processes are bit-identical. The exact path remains available via
`TINYCODER_INKLING_GEMM=0`, `_EXPERT_MMA=0`, `_Q2_0_MMA=0` (224.9 pp512).

`expert_down` went 4.23 → 4.30 ms, which is neutral: it is Q2_0 on 18 blocks
and NVFP4 on 6, and Q2_0's int8 form only works if the *bytes* were the cost.
At 0.28 B/weight they are not — the loop is decode-ALU bound like the rest of
this path, so moving from the float decoder to int8 changes little. The
remaining gate/up blocks are IQ2_XXS (7) and IQ1_S (6), both still refused by
the int8 kernel; adding those decoders is the next lever on this column.
Q2_0 (down), IQ2_XXS, IQ4_XS (LM head), NVFP4, IQ1_S, F32 and BF16 stay on the
float decoders.

Profile at tg128 (`TINYCODER_INKLING_GPU_PROF=1`), per token over 48
layer-slots: `expert_down` 4.30 ms, `attn_mix` (depth mix, control, three RMS
norms) 4.04, `expert_gate_up` 4.00, `attn_core` 2.99, `attn_out` 2.00, `ffn_mix`
2.00, `attn_proj` (Q/K/V/relative-bias GEMVs) 1.99, `attn_glue` (short
convolutions) 1.57, `router` 1.38, `lm_head` 0.81, `ffn_post` 0.67 — 25.7 ms
total across 1302 launches per token (no CUDA graph capture yet). The two
expert GEMVs are still 8.3 ms of the 25.7 ms. `attn_mix` at 4.04 ms across 48
calls is now the second-largest single item and is launch-latency bound (the
three RMS norms and the control channels are ~430 tiny launches per token), so
it responds to fusion or CUDA-graph capture rather than to a faster kernel. A
Debug CUDA build is ~10x slower; these numbers are Release.

Method: `./scripts/inkling_ab.sh` (text gate first, then interleaved runs of
both protocols against both llama.cpp backends). Baseline provenance: llama.cpp
Unsloth `inkling2-loop-legacy-layout` (dd330cc) with the looped-layer
`layer_phys` fix needed to run the expanded layout; stock mainline cannot load
the file. That tree also carries step-time `[dbg]` prints, which cost well under
1% and can only make llama.cpp slower.

Parity: on a 1-token prompt top-1 and top-5 set match both the exact scalar CPU
path and the shipped default CPU path. Against the exact path the worst top-5
delta is 5.43 (all-vocab RMS 4.75); against the default path it is 0.29 (RMS
0.30). The 5.43 is the Q8_1 activation quantisation of the int8 GEMV, the same
class as the default CPU path, whose own exact-vs-default RMS is 4.57 — not a
defect, which is why the default path is the gate reference for the GPU graph.

The gate tool (`build-inkling/unit_tests/inkling_gpu_forward_test`) now also
runs multi-token prompts, because a 1-token forward does not exercise the
per-slot K/V and short-conv state at all. That gap was hiding a real defect: at
HEAD the graph was coherent for one token and emitted word salad from the
second, a `dBlockBank`/`dBankSnap` scratch buffer sized `8*H` while the depth
mix indexes it by slot (`slots+2` rows). `FullQuestions` failed on the GPU path
and passed on CPU before the fix, on the same binary. With the sizing corrected
the 2-token gate passes on the documented default-path criterion (top-1 and
top-5 set match, per-logit band 0.164, all-vocab RMS 0.333) and the exact path
agrees on top-1 with band 2.70. The gate itself is now stated as that criterion:
the default path's top-1, top-5 ordering and ≤0.5 band are the gate, and the
exact path's top-1 is required while its band is reported as a diagnostic,
because the reference's own default-vs-exact RMS is 2.64 on that 2-token prompt
(4.57 on one token) — the Q8_1 rounding flips a near-tied router expert
(selection gap ~3e-4 at block 0) and that is a discrete trajectory change, not
an arithmetic error.

Kernel parity is unchanged (attention maxdiff 1.192e-07, depth-mix 1.192e-07),
and `inkling_gpu_parity` gained a short-convolution case with non-zero history
(step0/step1 maxdiff 1.490e-08) plus expert-GEMV cases for IQ2_XS (shared
activation, 2.190e-03) and Q2_0 per-row (2.284e-03). The GPU path answers
"The capital of France is Paris." on the chat prompt and passes all four gated
questions. VRAM: 8828 MiB of 11264 MiB (weights 7.66 GiB, K/V 768 MiB,
scratch 19.4 MiB).

Checkpoints (`unsloth/Inkling-2-Mini-GGUF`):

- [Inkling-2-Mini-UD-IQ2_XXS.gguf](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-IQ2_XXS.gguf)
- [Inkling-2-Mini-UD-Q3_K_XL.gguf](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-Q3_K_XL.gguf)

The Q3_K_XL file (13.6 GB) is CPU-only on this card; it does not fit 11 GB.

---

Back to the [README](../README.md).
