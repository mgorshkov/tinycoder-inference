# Benchmarking on RTX 3090: GPU Performance (TinyCoder vs llama.cpp)

**Status: no row is *published* yet; §1.1 records one rented session as *reported*.**
A third-party stand ran `tinycoder_bench` and the OpenAI server on a single 3090 at
commit `6c578a5` and sent the numbers. That report is **not reproduced in this repository**; what it
said is summarised in §1.1, so the figures keep their provenance without republishing
the sender's document. They are real measurements on the right
card, but they do not satisfy §2 — no text gate through `scripts/text_gate.sh`, no roof
measured in-session, and the Q6_K rows logged thousands of int8 fallbacks — so they live
in §1.1 as **reported**, not in the published table. Nothing here may be quoted as a
published row until §2 is satisfied.

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
| [Qwen2.5-Coder-1.5B `q2_k`](https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q2_K.gguf) | 0.7 GB | full offload | — | — | llama.cpp | — | — | — | — |
| [Qwen2.5-Coder-1.5B `iq3_xxs`](https://huggingface.co/ggml-org/Qwen2.5-Coder-1.5B-IQ3_XXS-GGUF/resolve/main/qwen2.5-coder-1.5b-iq3_xxs-imat.gguf) | 0.9 GB | full offload | — | — | llama.cpp | — | — | — | — |
| [Qwen2.5-Coder-7B `iq2_s`](https://huggingface.co/ThomasBaruzier/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf) | 2.4 GB | full offload | — | — | llama.cpp | — | — | — | — |
| [Qwen2.5-Coder-7B `iq3_xxs`](https://huggingface.co/ThomasBaruzier/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-IQ3_XXS.gguf) | 2.9 GB | full offload | — | — | llama.cpp | — | — | — | — |
| [Ternary-Bonsai-2-27B `pq2_0` (Prism)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf) | 6.7 GB | full offload | — | — | PrismML llama.cpp (fork) | — | — | — | — |
| [Ternary-Bonsai-2-27B `ptq1_0` (Prism)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf) | 5.5 GB | full offload | — | — | PrismML llama.cpp (fork) | — | — | — | — |
| [Ornith-1.5-9B `q4_k_m`](https://huggingface.co/bartowski/Ornith-1.5-9B-GGUF/resolve/main/Ornith-1.5-9B-Q4_K_M.gguf) | 5.9 GB | full offload | — | — | llama.cpp | — | — | — | — |
| [Ornith-1.5-35B-A3B `q8_0` → `q4_k` experts](https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-A3B-Q8_0.gguf) | 34.4 GB | **hybrid only** | — | — | llama.cpp `-ncmoe 99` | — | — | — | — |
| [Inkling-2-Mini `iq2_xxs`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-IQ2_XXS.gguf) | 8.5 GB | full offload | — | — | llama.cpp CUDA (`-ngl 99`, ctx 2048) | — | — | — | — |
| [Qwen3.8-27B EXL3 `1.60bpw`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/SC_1.60bpw_H3) | 8.6 GB | full offload | — | — | exllamav3 1.6.6 | — | — | — | — |

The last two columns are the reason this file exists. Absolute tok/s says nothing about
whether the kernels are good on this card; the ratio against the **measured** roof does.
The roof is measured in the same session by `tools/int8_mma_peak` (int8 tensor) and a
copy/stream probe (memory), never taken from the specification, because a 3090's
delivered clock and memory temperature move both numbers.

### 1.1 Reported numbers, rented stand (2026-11-20)

Box: Xeon E5-2678 v3 (12c/24t, AVX2), 30 GiB RAM, **one** RTX 3090 on PCIe 3.0 x16,
driver 595.99.02, nvcc 13.1. Build: `-DENABLE_CUDA=ON -DENABLE_AVX2=ON
-DENABLE_AVX512=OFF -DENABLE_AMX=OFF`, arches 75;80;86, commit `6c578a5` — i.e.
**before** the sm_86 Q6_K int8 branch and before the fp16 subnormal fix, both of which
change the rows marked †. Protocol: `tinycoder_bench --n-prompts 512 --n-gen 128
--reps 3 --threads 24 --gpu`, three consecutive runs, spread < 1 %. Note the two
deviations from §3: **24 threads** (the box has 24) and no interleaved baseline columns.

| Model (checkpoint) | VRAM GiB | **pp512** | **tg128** | server tg, 512-token prompt | notes |
|---|---|---|---|---|---|
| [Qwen2.5-Coder-1.5B `q2_k` (unsloth)](https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q2_K.gguf) | 3.3 | **22 343** | **415.8** | 234.5 | int8 prefill healthy |
| [Qwen2.5-Coder-1.5B `iq3_xxs` (ggml-org)](https://huggingface.co/ggml-org/Qwen2.5-Coder-1.5B-IQ3_XXS-GGUF/resolve/main/qwen2.5-coder-1.5b-iq3_xxs-imat.gguf) | 3.2 | 22 598 | 403.0 | 224.2 | |
| [Qwen2.5-Coder-7B `iq2_s`](https://huggingface.co/ThomasBaruzier/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf) | 7.0 | 5 337 | 165.4 | 123.2 | |
| [Qwen2.5-Coder-7B `iq3_xxs`](https://huggingface.co/ThomasBaruzier/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-IQ3_XXS.gguf) | 7.4 | 5 374 | 159.7 | 122.4 | |
| [Ternary-Bonsai-2-27B `pq2_0` (Prism)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf) | 13.4 | 1 368 | 69.0 | 57.2 | 17.11: 27.7 / 24.6 |
| Ornith-1.5-35B-A3B `Q4_K_M` † | 22.8 | **129** | 62.2 | 52.2 | ~3000 `[int8] launch failed (type=14 …)` per run; 17.11: 130 / 61.5 |
| Qwen3.6-35B-A3B `UD-Q4_K_M` (unsloth) † | 22.9 | 102 | 59.1 | 48.0 | 5 180 `[int8] launch failed`; window 26 880 (VRAM-limited) |
| Qwen3.6-35B-A3B `UD-IQ1_M` (unsloth) † | 11.8 | **81** | 51.7 | 45.1 | same fallback lines |
| Qwen3.6-27B `UD-Q4_K_XL` (unsloth) | 23.2 | 360 | 33.4 | 19.9 | no fallback lines in its log |
| [Qwen3.8-27B EXL3 `1.60bpw`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/SC_1.60bpw_H3) | 15.0 | 1 172 | 27.0 | 25.1 | 17.11: 1173 / 26.6 |
| Qwen3.8-27B `UD-Q4_K_M` (unsloth) † | 22.7 | **32** | 12.8 | 12.4 | `kQuantizeQ8K_Batch: invalid configuration argument` (510×); aborted by the stand's 15 tok/s rule |
| Qwen2.5-Coder-1.5B `Q2_K` (bartowski) | 1.1 | — | ~20 | — | **CPU only**: `token embedding quant type 14 not implemented by GPU kernels` |
| Gemma 4 26B-A4B `UD-Q4_K_XL` (unsloth) | — | — | — | — | does not load: `Missing attention weights for layer 5` (global layers share V) |

The rented stand's 35B row is the **`Q4_K_M`** quantization from
`ornith-ai/Ornith-1.5-35B-A3B-GGUF`, not the `q8_0` file linked in §1; the two are
different artefacts of the same model and their speeds are not comparable cell-for-cell.

† expected to move once the two engine fixes in
`docs/superpowers/plans/2026-11-22-engine-defects-from-3090-report.md` land: the Q6_K
int8 branch already exists in the current tree, and the fp16 subnormal fix changes what
the 35B's Q4_K expert requant computes.

**Server vs `tinycoder_bench`:** the gateway loses 3–44 % of decode, monotonically with
model speed (1.5B 415.8 → 234.5, −44 %; 7B 165.4 → 123.2, −26 %; Bonsai 69.0 → 57.2,
−17 %; 35B 62.2 → 52.2, −16 %; EXL3 27.0 → 25.1, −7 %). That is a fixed per-token cost in
the gateway/SSE path, not an engine number; report §4.7 has the measurement.

**Where we stand on this card.** No llama.cpp or exllamav3 run on a 3090 exists in that
session, so this file cannot yet answer "faster than llama.cpp on Ampere" — §4 requires
the baseline column to be measured on the same card in the same session. The one
same-card competitor measured is NInfer on the same Qwen3.8-27B (different
quantization, so an engine comparison, not a quantization one): 97–110 tok/s decode with
DFlash2 speculation, ~49 tok/s without it, against our 25.1 — report §5.4. Against the
**published** card specification (936 GB/s, 284 int8 TOPS, not measured in that session)
the reported rows sit at 31–49 % of the memory roof on dense models (1.5B 291 GB/s, 7B
397 GB/s, Bonsai 462 GB/s), 25 % on the ALU-bound EXL3 row (232 GB/s), and **0.6–3 % of
the int8 roof on the 27B/35B prefill rows** — that gap is the fallback, not the kernels.
Long-context prefill is the other limit: 60k tokens run at 167 tok/s where 512 tokens run
at 1206 (report §4.1, §5.3), and 182k takes 51 minutes to first token.

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

A row measured elsewhere may be listed in §1.1 as **reported** when at least the card,
commit, flags and protocol are known; it moves into §1 only after a session in this
repository satisfies all five rules on that card. The distinction is not pedantry: the
per-token fallback in rule 5 is exactly what the rented Q6_K rows hit, and those numbers
describe the fallback path, not the kernels.

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

---

## Inkling-2-Mini

No Ampere numbers are published for Inkling-2-Mini; the row is listed with
dashes so the checkpoint and the intended baseline are tracked. The engine's full-device CUDA
graph exists and is verified on sm_75, but it has not been measured on a 3090;
the 2080 Ti numbers are the only measurement, and on that card the graph is
slower than llama.cpp CUDA on both canonical columns. The parity and speed
results are in [BENCHMARKS-2080TI.md](BENCHMARKS-2080TI.md).

Back to the [README](../README.md).
