# Models

## Getting a Model

TinyCoder loads two checkpoint formats:

- **GGUF** v3 — the llama.cpp/GGML container. Quantized K/IQ blocks, F32/F16 tensors, tokenizer and architecture metadata are read directly; `general.architecture` selects the model family.
- **EXL3** — the ExLlamaV3 `trellis`/`mul1` checkpoint format (HF `safetensors` + `config.json` + `tokenizer.json`).

Download the `.gguf` file and point the engine at it via `TINYCODER_MODEL_PATH` or the `Model::load()` API. For an **EXL3** checkpoint, download the HF checkpoint **directory** (e.g. `Qwen/Qwen3.5-27B-exl3-1.60bpw` — `config.json` + `*.safetensors` + `tokenizer.json`) and point the engine at the directory path; `Model::load` auto-selects [`EXL3Loader`](src/cpp/core/EXL3Loader.cpp).

[`Model::load`](src/cpp/core/ModelLoad.cpp) detects the format automatically, and both formats use the same CUDA/CPU kernels, KV cache, and offload path.

### Checkpoints behind the published benchmarks

Every row in [BENCHMARKS-2080TI.md](benchmarks/gpu/BENCHMARKS-2080TI.md) and [BENCHMARKS-3090.md](benchmarks/gpu/BENCHMARKS-3090.md) was measured on one of these files; the link is the exact artefact, so a row can be reproduced rather than approximated:

| Model | Quantization | Download |
|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | `q2_k` | [Qwen2.5-Coder-1.5B-Instruct-Q2_K.gguf](https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q2_K.gguf) |
| Qwen2.5-Coder-1.5B-Instruct | `iq3_xxs` | [qwen2.5-coder-1.5b-iq3_xxs-imat.gguf](https://huggingface.co/ggml-org/Qwen2.5-Coder-1.5B-IQ3_XXS-GGUF/resolve/main/qwen2.5-coder-1.5b-iq3_xxs-imat.gguf) |
| Qwen2.5-Coder-7B-Instruct | `iq2_s` | [Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf](https://huggingface.co/ThomasBaruzier/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf) |
| Qwen2.5-Coder-7B-Instruct | `iq3_xxs` | [Qwen2.5-Coder-7B-Instruct-IQ3_XXS.gguf](https://huggingface.co/ThomasBaruzier/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-IQ3_XXS.gguf) |
| Ternary-Bonsai-2-27B (Prism) | `PQ2_0` | [Ternary-Bonsai-2-27B-PQ2_0.gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf) |
| Ternary-Bonsai-2-27B (Prism) | `PTQ1_0` | [Ternary-Bonsai-2-27B-PTQ1_0.gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf) |
| Ornith-1.5-9B | `q4_k_m` | [Ornith-1.5-9B-Q4_K_M.gguf](https://huggingface.co/bartowski/Ornith-1.5-9B-GGUF/resolve/main/Ornith-1.5-9B-Q4_K_M.gguf) |
| Ornith-1.5-35B-A3B | `q8_0` | [Ornith-1.5-35B-A3B-Q8_0.gguf](https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-A3B-Q8_0.gguf) |
| Qwen3.8-27B (EXL3, directory) | `1.60bpw` | [turboderp/Qwen3.8-27B-exl3 — branch `SC_1.60bpw_H3`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/SC_1.60bpw_H3) |
| Inkling-2-Mini | `iq2_xxs` | [Inkling-2-Mini-UD-IQ2_XXS.gguf](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-IQ2_XXS.gguf) |
| Inkling-2-Mini | `q3_k_xl` | [Inkling-2-Mini-UD-Q3_K_XL.gguf](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-Q3_K_XL.gguf) |

The EXL3 row is a **branch of a multi-checkpoint repository**, so it has no single `resolve/main` file: download the branch as a directory (`config.json`, `*.safetensors`, tokenizer files) and point `Model::load` at the directory, per the paragraph above.

The two Inkling rows are the **Unsloth expanded layout** (`general.architecture = inkling2`, 24 physical blocks and 48 logical cache slots). The llama.cpp mainline `inkling2_loop` layout (24 blocks plus `loop_msg_query` per core block) is also accepted; the loader maps both onto the same 24 physical blocks and 48 KV slots. See [Inkling-2-Mini](#inkling-2-mini) for the verified parity and the measured performance. The published row is in [BENCHMARKS-2080TI.md](benchmarks/gpu/BENCHMARKS-2080TI.md#L27).

## Inkling-2-Mini

Inkling-2-Mini is a 26.4B-parameter looped Mixture-of-Experts model: 24 physical
blocks, of which blocks 6-17 run three times with shared weights. Each pass
keeps its own K/V and short-convolution state, passes exchange gated messages,
and attention uses a learned relative-position bias with a sink logit instead
of RoPE. The engine implements the text decoder in `src/cpp/core/ModelInkling.cpp`.

Supported now:

- **Both GGUF layouts**: the llama.cpp mainline `inkling2_loop` layout (24
  blocks, `loop.start/end/count`, per-block `loop_msg_query`) and the Unsloth
  expanded `inkling2` layout (24 physical blocks, 48 logical slots,
  `loop.core_start/core_end`, global `loop_res_query`/`loop_kv_res_query`).
- **All quantization types used by the published Unsloth quants**: Q2_K, Q3_K,
  Q4_K, Q5_K, Q6_K, Q8_0, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
  IQ3_S, IQ4_NL, IQ4_XS, NVFP4, Q2_0, BF16 and F32.
- **Text only.** The vision tower, the DSpark speculator and the 1M-token
  context are not implemented; the granted window is whatever the existing
  RAM planner can hold (2048 by default).
- **SIMD coverage.** NVFP4, Q2_0, IQ1_S and the remaining 256-weight i-quants
  have AVX2 batch kernels; `ENABLE_AVX512=ON` compiles the 16-wide twins.  AMX
  is deliberately not used for these types: AMX is an int8/BF16 tile engine, so
  a 2/4-bit code with a UE4M3/fp16 scale that changes every 16 or 64 weights
  must first be expanded to int8 and scaled exactly as the AVX2 path already
  does, which adds tile setup and saves no arithmetic.  The reference host (and
  the Xeon E5-2686 v4 in this repo's environment table) has no AMX either, so
  there is no AMX path to measure here.

Canonical protocol (`tinycoder_bench --n-prompts 512 --n-gen 128 --threads 36`,
`TINYCODER_INKLING_GPU=1`, default environment) on the IQ2_XXS checkpoint: **1200.0
tok/s** prefill and **47.0 tok/s** decode, against llama.cpp CUDA (Unsloth
`inkling2-loop-legacy-layout`, `-ngl 99`, 2048-token context) at 883.6 / 44.3 —
three alternating interleaved pairs from `scripts/inkling_ab.sh`, text gate
first. The full row, its 16-token diagnostic and the spread are in
[BENCHMARKS-2080TI.md](benchmarks/gpu/BENCHMARKS-2080TI.md).

---

Back to the [README](README.md).
