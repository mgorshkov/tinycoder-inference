# TinyCoder CPU benchmarks — 2x Intel Xeon E5-2686 v4

Reference box: **2x Intel Xeon E5-2686 v4**, 18 cores / 36 threads per socket-pair
(36 threads used), 2.30 GHz base / 3.0 GHz max, 90 MiB L3, 125 GiB RAM.
No GPU is used: `TINYCODER_GPU=0`.

The GPU tables live in [`benchmarks/gpu/`](../gpu/). This file uses the same
checkpoints as the GPU table, but a **16-token prompt / 1 generated token**
diagnostic rather than the canonical `pp512`/`tg128`.

```bash
TINYCODER_GPU=0 TINYCODER_NUMA=off ./build/benchmarks/tinycoder_bench \
  --model <checkpoint> --n-prompts 16 --n-gen 1 --reps 1 --threads 36
```

All tok/s. `pp16` is 16 prompt tokens; `tg1` is one decoded token.

| Model | File size | TinyCoder pp16 | TinyCoder tg1 | llama.cpp CPU pp16 | llama.cpp CPU tg1 |
|---|---|---|---|---|---|
| [Qwen2.5-Coder-1.5B `q2_k`](https://huggingface.co/unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q2_K.gguf) | 0.7 GB | 84.1 | **35.23** | **187.1** | 30.23 |
| [Qwen2.5-Coder-1.5B `iq3_xxs`](https://huggingface.co/ggml-org/Qwen2.5-Coder-1.5B-IQ3_XXS-GGUF/resolve/main/qwen2.5-coder-1.5b-iq3_xxs-imat.gguf) | 0.9 GB | 17.0 | 7.36 | **199.9** | **31.57** |
| [Qwen3.5-4B `Q4_K_M`](https://huggingface.co/LocalScribe/Qwen3.5-4B-Q4_K_M.gguf/resolve/main/Qwen3.5-4B-Q4_K_M.gguf) | 2.6 GB | 8.7 | **5.99** | **58.8** | 4.83 |
| [Qwen2.5-Coder-7B `iq2_s`](https://huggingface.co/ThomasBaruzier/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf) | 2.4 GB | 2.8 | 1.26 | **57.1** | **8.34** |
| [Qwen2.5-Coder-7B `iq3_xxs`](https://huggingface.co/ThomasBaruzier/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-IQ3_XXS.gguf) | 2.9 GB | 4.0 | 2.87 | **58.8** | **6.90** |
| [Ternary-Bonsai-2-27B `pq2_0` (Prism)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PQ2_0.gguf) | 6.7 GB | 0.6 | 0.58 | **13.9 †** | **1.47 †** |
| [Ternary-Bonsai-2-27B `ptq1_0` (Prism)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf) | 5.5 GB | 0.6 | 0.45 | **7.27 †** | **1.99 †** |
| [Ornith-1.5-9B `q4_k_m`](https://huggingface.co/bartowski/Ornith-1.5-9B-GGUF/resolve/main/Ornith-1.5-9B-Q4_K_M.gguf) | 5.9 GB | 5.9 | 3.34 | **45.4** | **4.23** |
| [Inkling-2-Mini `iq2_xxs`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-IQ2_XXS.gguf) | 8.5 GB | 2.0 | 2.00 | **19.2 ‡** | **3.69 ‡** |
| [Ornith-1.5-35B-A3B `q8_0`](https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-A3B-Q8_0.gguf) | 34.4 GB | 6.5 | 3.51 | **27.7** | **3.76** |
| [Qwen3.8-27B EXL3 `1.60bpw`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/SC_1.60bpw_H3) | 8.6 GB | n/r | n/r | n/a | n/a |

The TinyCoder columns were re-measured with a CPU-only TinyCoder build
(`-DENABLE_CUDA=OFF`, `build-cpu/`) at 36 threads; they are not a CUDA-build
artifact. The llama.cpp columns are a **CPU-only llama.cpp build**
(`-DGGML_CUDA=OFF`) with `llama-bench -p 16 -n 1 -t 36 -ngl 0 -r 1`, same files
and protocol. The earlier CUDA-build `-ngl 0` numbers differ by up to 24% on
decode (e.g. 1.5B q2_k 27.0 vs 30.5), so only this table is the CPU reference.
The two Prism rows and the Inkling row could not be re-measured with mainline
llama.cpp (fork-only loaders, † and ‡); their baseline columns are the fork
measurements. † PrismML's llama.cpp fork, the only loader for the
Prism ternary format. ‡ Unsloth's `inkling2-loop-legacy-layout` fork
(`dd330cc`), because mainline cannot load the expanded layout. `n/a` = no
llama.cpp loader for EXL3; exllamav3 is a GPU engine and has no CPU path here.
`n/r` = not run: the EXL3 CPU loader wants ~132 GB of f32 weights on a
125 GiB box and starts swapping before it can measure; that checkpoint is
GPU-only here. All other rows completed. The Qwen2.5/Ornith/Ternary rows use the same files as
[`BENCHMARKS-2080TI.md`](../gpu/BENCHMARKS-2080TI.md); the last row is the EXL3
checkpoint. `pp16`/`tg1` must not be compared with `pp512`/`tg128` rows in the
GPU table.

---

Back to the [README](../../README.md).
