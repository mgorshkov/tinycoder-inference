# Benchmarking: GPU Performance (TinyCoder vs llama.cpp)

## GPU Comparison (TinyCoder vs llama.cpp / exllamav3)

Columns follow the benchmark defaults: `tinycoder_bench --n-prompts 512 --n-gen 128 --threads 36` (**pp512 / tg128**). GGUF baselines are `llama-bench -p 512 -n 128 -t 36 -r 1`, run **interleaved** with the TinyCoder run (three alternating pairs, so clock drift affects both columns equally, means reported; the Ornith-35B row adds `-ncmoe 99`, the same expert-offload mode the TinyCoder column uses); the EXL3 baseline is a local exllamav3 1.6.6 run measured with the same pp512/tg128 protocol — see [How the exllamav3 EXL3 baseline was measured](#how-the-exllamav3-exl3-baseline-was-measured). Warm mean-of-reps, RTX 2080 Ti. Bold marks the faster engine in a column. **A TinyCoder figure is only printed if that checkpoint passes the text gate** (`./scripts/text_gate.sh`, which answers a factual question through the real forward and treats a skipped test as a failure): a throughput number from a path that cannot answer a question measures wasted work, not performance. Rows without a TinyCoder figure are withheld on exactly that basis and say so below the table.

| Model | File size | GPU mode TinyCoder | **TinyCoder pp512** | **TinyCoder tg128** | Baseline engine | **Baseline pp512** | **Baseline tg128** |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B `q2_k` | 0.7 GB | full offload | **10,149.8 tok/s** | **327.91 tok/s** | llama.cpp | 7,769.8 tok/s | 323.19 tok/s |
| Qwen2.5-Coder-1.5B `iq3_xxs` | 0.9 GB | full offload | **10,192.3 tok/s** | **276.77 tok/s** | llama.cpp | 10,143.5 tok/s | 261.47 tok/s |
| Qwen2.5-Coder-7B `iq2_s` | 2.4 GB | full offload | **2,518.2 tok/s** | 87.40 tok/s | llama.cpp | 2,500.9 tok/s | **96.59 tok/s** |
| Qwen2.5-Coder-7B `iq3_xxs` | 2.9 GB | full offload | 2,777.3 tok/s | 93.14 tok/s | llama.cpp | **2,929.1 tok/s** | **97.38 tok/s** |
| Ternary-Bonsai-2-27B `pq2_0` (Prism) | 6.7 GB | full offload | 16.7 tok/s | 15.24 tok/s | llama.cpp | — | — |
| Ornith-1.5-9B `q4_k_m` | 5.9 GB | full offload | 2,560.6 tok/s | 82.25 tok/s | llama.cpp | **2,671.7 tok/s** | **84.29 tok/s** |
| Ornith-1.5-35B-A3B `q8_0` → `q4_k` experts | 34.4 GB | hybrid (layer arena, grouped int8) | **176.3 tok/s** | **53.67 tok/s** | llama.cpp `-ncmoe 99` | 130.7 tok/s | 16.42 tok/s |
| Qwen3.8-27B EXL3 `1.60bpw` | 8.6 GB | full offload | **723.0 tok/s** | **23.26 tok/s** | exllamav3 1.6.6 | 707.5 tok/s | 22.9 tok/s |

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
| 1 | 703.8 tok/s | 14.11 tok/s |
| 2 | 711.5 tok/s | 22.90 tok/s |
| 3 | 707.3 tok/s | 22.61 tok/s |
| **mean** | **707.5 tok/s** | **22.9 tok/s** |

Run 1's tg128 is a cold-cache outlier; runs 2-3 are the steady state. The tg128 figure also includes
Python per-token overhead (sampler call plus tensor concatenation), so it is a floor rather than a
best case.

---

Back to the [README](README.md).
