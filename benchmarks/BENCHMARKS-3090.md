# Benchmarking on RTX 3090: GPU Performance (TinyCoder vs llama.cpp)

**Status: no row is published yet.** Every cell below is empty on purpose. This file
exists so that the Ampere numbers have a home with the same protocol, gates and
vocabulary as the Turing file, and so that the first session on the card fills a table
rather than writing a report. Nothing here may be quoted until §2 is satisfied.

Per-card benchmark files: this one is the **RTX 3090** (24 GB, sm_86, GA102, 82 SMs,
936 GB/s, 100 KB smem/SM, 99 KB opt-in per block, `cp.async`, `mma.m16n8k32` int8).
The reference box is [BENCHMARKS-2080TI.md](BENCHMARKS-2080TI.md); a number is only
comparable within the file for the card it was measured on, and a row's baseline column
must be measured on the same card in the same session.

## 1. GPU comparison (TinyCoder vs llama.cpp / exllamav3)

Columns follow the same benchmark defaults as the 2080 Ti file:
`tinycoder_bench --n-prompts 512 --n-gen 128 --threads 36` (**pp512 / tg128**).

| Model | File size | GPU mode TinyCoder | **TinyCoder pp512** | **TinyCoder tg128** | Baseline engine | **Baseline pp512** | **Baseline tg128** | decode % of memory roof | prefill % of int8 roof |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B `q2_k` | 0.7 GB | full offload | — | — | llama.cpp | — | — | — | — |
| Qwen2.5-Coder-1.5B `iq3_xxs` | 0.9 GB | full offload | — | — | llama.cpp | — | — | — | — |
| Qwen2.5-Coder-7B `iq2_s` | 2.4 GB | full offload | — | — | llama.cpp | — | — | — | — |
| Qwen2.5-Coder-7B `iq3_xxs` | 2.9 GB | full offload | — | — | llama.cpp | — | — | — | — |
| Ternary-Bonsai-2-27B `pq2_0` (Prism) | 6.7 GB | full offload | — | — | PrismML llama.cpp (fork) | — | — | — | — |
| Ternary-Bonsai-2-27B `ptq1_0` (Prism) | 5.5 GB | full offload | — | — | PrismML llama.cpp (fork) | — | — | — | — |
| Ornith-1.5-9B `q4_k_m` | 5.9 GB | full offload | — | — | llama.cpp | — | — | — | — |
| Ornith-1.5-35B-A3B `q8_0` → `q4_k` experts | 34.4 GB | **hybrid only** | — | — | llama.cpp `-ncmoe 99` | — | — | — | — |
| Qwen3.8-27B EXL3 `1.60bpw` | 8.6 GB | full offload | — | — | exllamav3 1.6.6 | — | — | — | — |

The last two columns are the reason this file exists. Absolute tok/s says nothing about
whether the kernels are good on this card; the ratio against the **measured** roof does.
The roof is measured in the same session by `tools/int8_mma_peak` (int8 tensor) and a
copy/stream probe (memory), never taken from the specification, because a 3090's
delivered clock and memory temperature move both numbers.

The Ornith-1.5-35B-A3B row is **hybrid only**, exactly as in the 2080 Ti file: the
expert FFNs live in system RAM and both columns run the same expert-offload mode
(`-ncmoe 99` on the llama.cpp side). It is not a statement about a fully GPU-resident
path; at 34.4 GB the model does not fit on this card, so no fully resident number can
exist here.

## 2. What a row must satisfy before it is published

Same rules as the 2080 Ti file, plus one that is specific to this card:

1. **The text gate passes on the same tree and the same checkpoint**:
   `./scripts/text_gate.sh`. A skip is a failure; a row whose shape is not on the
   `AnswersQuestion` allow-list in `unit_tests/ModelTest.cpp` has no gate and cannot be
   published.
2. **The build's arch list is known.** The configure line is recorded, and
   `cuobjdump --list-elf` on the built `tinycoder_core.a` shows `sm_86`. A stale
   `CMAKE_CUDA_ARCHITECTURES` cache value produces a plausible, slow, wrong binary.
3. **The physics check passes**: `2 * params * tok_s / 1e12` (prefill, against the
   measured int8 roof) and `weight_bytes * tok_s / 1e9` (decode, against the measured
   memory roof). A number above the card's roof is a bug, not a win.
4. **Interleaved, three runs, spread reported**, and the card's clock and memory
   temperature recorded in the session log.
5. **No silent fallback.** If a batched path declined and the engine took a per-token
   route, the fallback counter must read zero for a published row.

## 3. Protocol and build recipe

Build for the card's own SASS, and check that it happened:

```bash
cmake -B build-3090 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-3090 --target tinycoder_core tinycoder_bench -j 20 || exit 1
cuobjdump --list-elf build-3090/libtinycoder_core.a | grep -c sm_86   # must be > 0
```

`ENABLE_AVX2=ON` and `ENABLE_AVX512=OFF` as on the reference box, so the CPU-path
fallbacks are the same shape. Benchmark protocol is byte-identical to the 2080 Ti file:
`--n-prompts 512 --n-gen 128 --threads 36`, serial, one process at a time, no other
tenant on the GPU, warm mean of three alternating interleaved pairs.

## 4. Baselines on this card

Each baseline must be built for `sm_86` and run on this card, in the same session:

- **llama.cpp**: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`, `llama-bench -p 512
  -n 128 -t 36 -r 1`.
- **PrismML fork** (the two Bonsai rows; stock llama.cpp cannot load either packing):
  same build flags, `scripts/bonsai_baseline_bench.sh`.
- **exllamav3 1.6.6** (EXL3 row): `scripts/exl3_baseline_bench.py`, with
  `max_chunk_size` and `Cache(max_num_tokens=...)` equal and a multiple of 256. The
  first run of a fresh process is cold on this card; the published figure is the warm
  mean, and the cold first run is recorded in the session log rather than dropped.

## 5. How this table gets filled

One session, scripted end to end (`scripts/sm86_receipt.sh`, added by the sm_86 work —
see `docs/specs/2026-11-20-sm86-ampere-kernels.md` §8.4): roofs, parity, logits gate,
text gate, interleaved benchmark per row, `ncu` on the four kernels that matter, SASS
census, resolved dispatch decisions, then this table and the two roof percentages.
The session costs about an hour of rental; the script exists so that the hour is a
measurement and not a debugging session, and so that anyone can re-run it.
