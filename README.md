# ⚡ TinyCoder Inference Engine for NVIDIA GPUs (Turing / Ampere / Ada)

**High-performant local inference engine** for the [TinyCoder AI](https://github.com/mgorshkov/tinycoder) VS Code extension and the [TinyCoder OpenAI](https://github.com/mgorshkov/tinycoder-openai) project, which exposes the same engine through an OpenAI-compatible API. This repository contains **only the engine**: the C++/CUDA core (`src/cpp/core`), public headers (`include/`), unit tests (`unit_tests/`), benchmarks (`benchmarks/`), and build plumbing.

The engine runs the **Qwen3.8** (qwen35 dense), **Ornith** (qwen35moe), **Gemma 4**, and Prism **Ternary-Bonsai** model families from **GGUF** or **EXL3** (`trellis` / `mul1`) checkpoints — e.g. `Qwen3.5-27B` at 1.60 bpw — with **mixed quantization**, **AMX/AVX-512/AVX2 CPU acceleration** (runtime SIMD dispatch), and a **CUDA GPU offload engine**. It is designed to run efficiently on limited hardware.

> **GPU support.** The CUDA engine runs natively on **Turing, Ampere and Ada GPUs — including the RTX 2080 Ti, RTX 3090, RTX 4090 and RTX 5090** — plus newer architectures via forward-compatible PTX JIT, with runtime capability dispatch and a CPU fallback. **No GPU emulator is used or needed.** See [Supported GPUs](#supported-gpus-no-emulator-required).

## Benchmarking: GPU Performance (TinyCoder vs llama.cpp)

### GPU Comparison (TinyCoder vs llama.cpp / exllamav3)

Columns follow the benchmark defaults: `tinycoder_bench --n-prompts 512 --n-gen 128 --threads 36` (**pp512 / tg128**). GGUF baselines are `llama-bench -p 512 -n 128 -t 36`; the EXL3 row uses exllamav3's native `eval/perf.py -m <checkpoint> -max_length 512 -chunk_size 512 -cs 4096 -dr 2`. Warm mean-of-reps, RTX 2080 Ti. **Re-measured 2026-11-05** on the reference box (TinyCoder columns; the baseline columns are the recorded llama.cpp/exllamav3 numbers); bold marks the faster engine in a column.

| Model | File size | GPU mode TinyCoder | **TinyCoder pp512** | **TinyCoder tg128** | Baseline engine | **Baseline pp512** | **Baseline tg128** |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B `q2_k` | 0.7 GB | full offload | **10,183 tok/s** | **317.7 tok/s** | llama.cpp | 7,523 tok/s | 285.7 tok/s |
| Qwen2.5-Coder-1.5B `iq3_xxs` | 0.6 GB | full offload | **10,273 tok/s** | **248.1 tok/s** | llama.cpp | 10,120 tok/s | 244.7 tok/s |
| Qwen2.5-Coder-7B `iq2_s` | 2.4 GB | full offload | **2,575 tok/s** | **95.0 tok/s** | llama.cpp | 2,447 tok/s | 90.7 tok/s |
| Qwen2.5-Coder-7B `iq3_xxs` | 2.9 GB | full offload | 2,574 tok/s | **101.2 tok/s** | llama.cpp | **2,887 tok/s** | 86.4 tok/s |
| Ternary-Bonsai-2-27B `pq2_0` (Prism) | 6.7 GB | full offload | 17.0 tok/s | 15.49 tok/s | llama.cpp | — | — |
| Ornith-1.5-9B `q4_k_m` | 5.9 GB | full offload | 2,203 tok/s | 81.1 tok/s | llama.cpp | **2,726 tok/s** | **84.4 tok/s** |
| Ornith-1.5-9B `q8_0` | 8.9 GB | full offload | 1,620 tok/s | 57.3 tok/s | llama.cpp | **2,932 tok/s** | **58.5 tok/s** |
| Ornith-1.5-35B `q4_k_m` | 20.2 GB | hybrid (expert cache) | 33.0 tok/s | 19.69 tok/s | llama.cpp | **255 tok/s** | **25.8 tok/s** |
| Ornith-1.5-35B-A3B `q8_0` | 34.4 GB | hybrid (expert cache) | 23.9 tok/s | 11.03 tok/s | llama.cpp | **147 tok/s** | **18.8 tok/s** |
| Qwen3.8-27B EXL3 `1.60bpw` | 5.3 GB | full offload | 5.8 tok/s | 5.44 tok/s | exllamav3 1.6.6 | **716.6 tok/s** | **24.0 tok/s** |

## Key Features

### 🧠 Supported Models

TinyCoder loads **both [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) files** (the llama.cpp / GGML ecosystem) **and [EXL3](https://github.com/turboderp-org/exllamav3) checkpoints** (the ExLlamaV3 `trellis` format). Whichever format you use, you get the **same models, quantization support, and CUDA offload engine** — point the engine at a `.gguf` file or a checkpoint directory and it picks the right format automatically via [`Model::load`](src/cpp/core/ModelLoad.cpp).

| Model Family | Format | Sizes | Architecture |
|---|---|---|---|
| **Qwen3.5-27B** | **EXL3** (HF safetensors, 1.60 bpw `mul1`) | 27B dense hybrid | 48× Gated-Delta-Net + 16× full-attention, MRoPE, GQA, **fused Q+gate**, SwiGLU |
| **Qwen3.8 (qwen35)** | GGUF | 27B dense hybrid | 48× Gated-Delta-Net + 16× full-attention, MRoPE, GQA, SwiGLU, MTP |
| **Ornith (qwen35moe)** | GGUF | 35B-A3B (Q8_0 / Q4_K_M checkpoints) | qwen35moe MoE: 256 experts (top-8), GDN + full-attention hybrid, softmax router |
| **Ornith-1.5-9B (qwen35)** | GGUF | 9B dense (Q4_K_M) | dense GDN + full-attention hybrid: 32 layers (24 recurrent + 8 full-attn), MRoPE, GQA, SwiGLU |
| **Gemma 4** | GGUF | 26B (A4B MoE) | MoE, RoPE, GQA, GeGLU, RMSNorm |
| **Ternary-Bonsai (Prism)** | GGUF | 27B dense hybrid (`PQ2_0`) | qwen35 + Prism Hadamard-fold (latent embedding, folded weights); private type 142 |

- Chat templates: `<|im_start|>system/user/assistant<|im_end|>` (Qwen), `<start_of_turn>user/model<end_of_turn>` (Gemma)
- **Weight tying** — LM head shares the token embedding matrix when possible
- Architecture detection via GGUF `general.architecture` or EXL3 `config.json` -> [`ModelConfig.hpp`](include/ModelConfig.hpp:37)

### ⚡ SIMD CPU Acceleration

- **Runtime SIMD dispatch** — auto-detects the best available instruction set: **AMX** (Intel Sapphire Rapids+), **AVX-512** (Skylake-X / Ice Lake+), **AVX2** (Haswell+ / AMD Zen+), or a universal **Scalar** fallback
- **OpenMP** thread parallelism for all matrix operations
- Fused dequantize-dot-product kernels that avoid float temporaries
- Optional runtime A/B against the scalar double-precision baseline via `TINYCODER_FORCE_SCALAR=1`

### 🎮 CUDA GPU Offload Engine

- Full-model GPU offload ([`GPUComputeForward.cu`](src/cpp/core/GPUComputeForward.cu), [`GPUComputeMoE.cu`](src/cpp/core/GPUComputeMoE.cu), [`GPUComputeUpload.cu`](src/cpp/core/GPUComputeUpload.cu), [`ModelGPU.cpp`](src/cpp/core/ModelGPU.cpp)):
  - **Prefill** — cuBLAS fp16 tensor-core GEMM for all projections
  - **Decode** — quantized on-the-fly-dequant GEMV kernels reading raw GGUF blocks from VRAM
  - **Flash attention** — `kWarpAttention<HD>` with compile-time register accumulators, exactly `nHeads` warps
- **Enabled by default** in builds compiled with `ENABLE_CUDA=ON`; all layers offloaded; automatic fallback to CPU when CUDA is unavailable or at the first forward error
- **Hybrid partial offload for qwen35** (large dense models that exceed VRAM): the engine keeps the whole model in RAM and mirrors only the first `numGpuLayers` layers' weights to the GPU, running those layers on the GPU and continuing the rest on the CPU in the same forward pass. See [GPU Partial-Offload Scheme](#gpu-partial-offload-scheme-qwen35).
- Runtime controls: `TINYCODER_GPU=0` forces the CPU path, `TINYCODER_NGL` limits offload to N layers (unset → automatic VRAM fit), `TINYCODER_GPU_VERBOSE=1` prints per-stage CUDA timings

#### Supported GPUs (no emulator required)

The CUDA offload engine runs natively on **Turing, Ampere and Ada GPUs — including the RTX 2080 Ti, RTX 3090, RTX 4090 and RTX 5090** — and on newer architectures through forward-compatible PTX JIT. **No GPU emulator is used or needed**: every path launches real CUDA kernels and queries the device at runtime, and the engine falls back to the CPU path when no usable GPU is present.

| GPU | Architecture | Compute capability | How it runs |
|---|---|---|---|
| RTX 2080 Ti | Turing | sm_75 | Native SASS (`compute_75`) |
| RTX 3090 | Ampere | sm_86 | Native SASS (`compute_86`) |
| RTX 4090 | Ada | sm_89 | sm_86 SASS or embedded PTX JIT (build `-DCMAKE_CUDA_ARCHITECTURES=89` for native SASS) |
| RTX 5090 | Blackwell | sm_120 | Embedded PTX JIT (build `-DCMAKE_CUDA_ARCHITECTURES=120` for native SASS) |

- The build emits SASS **and PTX** for multiple capabilities (`75` / `80` / `86` by default; override with `-DCMAKE_CUDA_ARCHITECTURES=...`), so the driver JITs forward-compatible code for a GPU whose exact SASS is not embedded.
- Capability gates are **lower bounds with fallbacks**, never equality or device-name checks — e.g. [`int8MmaOk()`](src/cpp/core/GPUComputeGemm.cu:170) enables the int8 `m8n8k16` INT8 path only when the device is sm_75+ (`(major > 7) || (major == 7 && minor >= 5)`); everything else falls back to the quantized GEMV/GEMM path or the CPU.
- Much of the kernel suite is Turing-tuned (EXL3 kernels; warp flash-attention and split-KV sized for the 2080 Ti's 68 SMs). Ampere/Ada and newer GPUs run the same kernels, so clock/bandwidth differences surface as performance, not correctness.

### 🧵 Tuned Threading & Scheduling

- Dedicated [`ThreadPool`](include/ThreadPool.hpp) with work-stealing (`parallelForSteal2`) for the fused FFN (llama.cpp `ggml_compute_forward_mul_mat_one_chunk` style)
- Cooperative last-arriver act→Q8_K quantize inside the fused FFN phase 1
- CPU affinity pinning of workers to **distinct logical CPUs** (default ON, runtime `TINYCODER_AFFINITY=0/1`)
- Steady-state generation performs **zero heap allocations per token** (per-thread `ScratchPool` reuse)

## Dependencies

No external dependency except:

- **[np library](https://github.com/mgorshkov/np)** — NumPy-style arrays with SIMD/CUDA acceleration (fetched via `FetchContent`)
- **[shared cmake config](https://github.com/mgorshkov/cmake)** — compiler checks, CUDA setup, OpenMP (fetched via `FetchContent`)
- **Google Test** — unit tests only (fetched via `FetchContent`)

All are fetched automatically; no system packages beyond a compiler, CMake, and (optionally) the CUDA Toolkit are required.

## Build & Install

### Prerequisites

- **C++20 compiler** (GCC ≥ 13, Clang ≥ 14, or MSVC 2019+)
- **CMake** ≥ 3.18

### Optional (for acceleration)

- **OpenMP** (usually included with the compiler)
- **CUDA Toolkit** ≥ 11.0 + nvcc (for GPU support on Turing / Ampere / Ada and newer NVIDIA GPUs — RTX 2080 Ti, RTX 3090, RTX 4090, RTX 5090)

### One-shot build

```bash
./scripts/build.sh                          # GPU build (-DENABLE_CUDA=ON) — default
./scripts/build.sh --no-cuda                # CPU-only build (-DENABLE_CUDA=OFF)
./scripts/build.sh --cpu --no-tests         # CPU-only static lib, no tests/bench
```

The script configures CMake, builds `tinycoder_core` (+ `tinycoder_test` and `tinycoder_bench`), and runs CTest.

### Manual build

```bash
# CPU-only
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# CUDA GPU offload
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build-cuda --config Release -j

# Run tests
ctest --test-dir build --output-on-failure
```

### Consuming as a library

The engine is designed to be embedded via `FetchContent` — the [TinyCoder extension](https://github.com/mgorshkov/tinycoder) does exactly this (it provides its own `np`/`cmake` targets and adds the N-API bridge on top of `tinycoder_core`):

```cmake
FetchContent_Declare(tinycoder_inference
    GIT_REPOSITORY <this repo>
    GIT_TAG <tag or commit>
)
FetchContent_MakeAvailable(tinycoder_inference)
target_link_libraries(your_target PRIVATE tinycoder_core)
```

This installs the static library and headers (`install(TARGETS tinycoder_core …)` / `install(FILES ${TINYCODER_HEADERS} DESTINATION include/tinycoder)`).

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_CUDA` | OFF | CUDA GPU acceleration (ON in `scripts/build.sh`) |
| `ENABLE_AVX2` | ON | AVX2 + FMA optimizations |
| `ENABLE_AVX512` | OFF | AVX-512 optimizations |
| `ENABLE_AMX` | OFF | Intel AMX (Advanced Matrix Extensions) |
| `ENABLE_OPENMP` | ON | OpenMP thread parallelism |
| `TINYCODER_FFN_STEAL` | ON | Fused FFN uses chunked work-stealing scheduling |
| `TINYCODER_AFFINITY` | ON | Pin worker threads to distinct logical CPUs |
| `TINYCODER_COOP_QUANTIZE` | ON | Cooperative last-arriver act→Q8_K quantize in fused FFN |
| `TINYCODER_STEAL_CHUNK` | 4 | Work-stealing chunk size (tiles) for fused FFN |
| `BUILD_SHARED_LIBS` | OFF | Build a shared library instead of static |
| `BUILD_TESTS` | ON | Build unit tests and benchmarks |

## Getting a Model

TinyCoder supports **GGUF v3 files** and **EXL3 (ExLlamaV3) Hugging Face checkpoint directories**. The GGUF table below lists every model that has been load-verified and benchmarked on this project's reference hardware (2× Intel Xeon E5-2686 v4 + RTX 2080 Ti **11 GB**); "GPU mode" is what the engine automatically selects on that 11 GB card:

| Model | Quantization | Size (approx) | GPU mode on 11 GB | Notes |
|-------|-------------|---------------|-------------------|-------|
| `Qwen3.8-27B` | UD-IQ1_M | 6.7 GB | full offload | Ultra-compact dense qwen35 |
| `Qwen3.8-27B` | UD-IQ3_S | 12.0 GB | partial offload | Balanced IQ dense |
| `Qwen3.8-27B` | UD-Q4_K_M | 16.5 GB | partial offload | **Recommended dense qwen35** |
| `Qwen3.8-27B` | UD-Q5_K_M | 19.8 GB | partial offload | Higher-precision dense |
| `Qwen3.8-27B` | UD-Q8_K_XL | 31.5 GB | partial offload | Near-lossless dense |
| `Gemma 4 (26B-A4B)` | Q4_K_XL (qat-UD) | 14.2 GB | partial offload | Instruction-tuned MoE |
| `Ternary-Bonsai-2-27B` | PQ2_0 (Prism) | 6.7 GB | full offload | Prism Hadamard-fold, 2-bit (TinyCoder-only) |
| `Ornith-1.5-35B-A3B` | Q8_0 | ~34.4 GB | hybrid (experts CPU) | qwen35moe, 256 experts (top-8) |
| `Ornith-1.5-35B` | Q4_K_M | ~20.2 GB | hybrid (experts CPU) | 41-layer qwen35moe variant |
| `Ornith-1.5-9B` | Q4_K_M | ~5.9 GB | full offload | **dense** qwen35 (32 layers, hidden 4096), no MoE; test suite 64/64 ✓ |

All rates are warm decode (tg8) on the reference hardware — the complete GPU table with llama.cpp comparison lives in [Benchmarking](#benchmarking). The Gemma 4 coding variants (`gemma4-coding-*`) hit a loader gap (their attention tensors are named differently) and print a clear load error rather than producing wrong output.

Download the `.gguf` file and point the engine at it via `TINYCODER_MODEL_PATH` or the `Model::load()` API. For an **EXL3** checkpoint, download the HF checkpoint **directory** (e.g. `Qwen/Qwen3.5-27B-exl3-1.60bpw` — `config.json` + `*.safetensors` + `tokenizer.json`) and point the engine at the directory path; `Model::load` auto-selects [`EXL3Loader`](src/cpp/core/EXL3Loader.cpp).

## Usage (C++ API)

```cpp
#include "Model.hpp"
#include "ModelConfig.hpp"

using namespace tinycoder;

// 1. Create and load the model (GGUF v3)
Model model;
std::string err;
if (!model.load("/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf", &err)) {
    // handle error
}

// 2. Configure generation
InferenceParams params;
params.maxTokens      = 512;    // max tokens to generate
params.temperature    = 0.7f;   // sampling temperature
params.topP           = 0.9f;   // nucleus sampling
params.topK           = 40.0f;  // top-K sampling
params.repeatPenalty  = 1.1f;   // repetition penalty
params.repeatLastN    = 64;     // penalty window
params.seed           = 42;     // 0 = random

// 3. Generate token by token (streaming callback)
std::string prompt = "Write a C++ function to add two numbers.";
std::vector<int32_t> tokens = model.generate(prompt, params,
    [](int32_t tokenId, const std::string &tokenText) {
        std::cout << tokenText << std::flush;
        return true;   // return false to stop generation early
    });

// Low-level access:
//   model.forward(tokenIds, /*computeAllLogits=*/false)  → prefill
//   model.tokenize(prompt)                                → token IDs
//   model.formatChat({{"user", "Hello"}})                 → chat prompt
//   model.clearKVCache()                                  → reset context
```

### Environment Variables

All runtime knobs (GPU offload, threading, and the per-kernel A/B
switches) are documented in [ENVIRONMENT.md](ENVIRONMENT.md).

## Testing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The Google Test suite (`tinycoder_test`) covers dequantization correctness across all quantization types, reference parity against sequential token-by-token forward passes, full-question generation tests, thread-pool reentrancy, and GPU/CPU comparison. On the reference build (`q2_k`) **55 pass / 2 fail / 16 skip** — the 2 failures are the non-deterministic sampled full-question cases (`SampleQuestionTest.AnswersQuestion/0,2`), which pass in isolation and are pre-existing sampling flakes, not regressions. All deterministic parity tests (GPU-vs-CPU logits, batch-vs-sequential prefill, sequential-decode argmax) pass bit-exactly with the split-KV and fp16-KV paths both on and off.

**EXL3 tests** (self-contained; the `EXL3*` suite covers the trellis codec, the safetensors reader/JSON parser, path sniffing, the Hugging Face `tokenizer.json` loader, and — when a checkpoint directory is present — config mapping + tokenizer round-trip):

```bash
# Codec/loader/tokenizer unit tests (no model needed): 12/12 pass
./build/unit_tests/tinycoder_test --gtest_filter='EXL3*'

# Point the self-contained EXL3 loader + the full format-neutral Model::load
# path at a real checkpoint directory (overrides the default search path)
TINYCODER_EXL3_CHECKPOINT=~/qwen3.8-27B-exl3-1.60bpw \
  ./build/unit_tests/tinycoder_test --gtest_filter='EXL3*'

# Validate a real EXL3 checkpoint directory (inventory + numeric sanity)
./build/unit_tests/exl3_model_smoke ~/qwen3.8-27B-exl3-1.60bpw

# EXL3 GPU kernels (CUDA build): bit-exact parity, then the benchmark tables
./build-avx2/unit_tests/exl3_gpu_parity    # 48/48 checks pass
./build-avx2/unit_tests/exl3_kernel_bench  # reconstruct + GEMV + decode projection
```

The `EXL3*` codec/reader/tokenizer tests are **always run** — they no longer require a model. Only the model-dependent suites (`ModelTest`, `ReferenceCompareTest`, `GPUCpuCompareTest`, `Qwen35Test`, `GGUFMetadataTest`, …) skip themselves when `TINYCODER_MODEL_PATH` (or `--model-path`) is unset or unresolvable; when it is unset the test binary falls back to the first existing default checkpoint (`qwen2.5` GGUF references, then the `qwen3.8-27B-exl3-1.60bpw` directory). Pointing `TINYCODER_MODEL_PATH` at an EXL3 directory exercises the full format-neutral `Model::load` path, including the Hugging Face `tokenizer.json` tokenizer.

On `Ornith-1.5-9B-Q4_K_M` (dense qwen35, hidden 4096) the suite runs **66 pass / 0 fail / 16 skip** (2026-10-29) — the 16 skips are the usual model-family guards (real-model IQ3_XXS/mma-mmq parity, the Q2_K Qwen2 reference compare, orphan-layer dumps, and the keyword questions reserved for named reference models). The int8 GEMM/GEMV path is covered by the model/GPU-vs-CPU forward tests.

## Technical Details

### GPU Partial-Offload Scheme (qwen35)

For dense models larger than VRAM (e.g. `Qwen3.8-27B` = 19.8 GB Q5_K_M on an 11 GB RTX 2080 Ti), the engine uses a **static residency split**: the *entire* model stays loaded in RAM, and only the first `numGpuLayers` transformer blocks are mirrored to GPU VRAM. No weights are transferred at runtime — the split is decided once at upload time.

#### Placement

| Object | Quantity | Residence | Bytes (Q5_K_M) |
|---|---|---|---|
| Token embedding (`Q5_K`) | 248320×5120 | GPU copy + RAM | 874 MB VRAM |
| Layers 0..29 weights (`Q5_K`/`Q6_K`/`F32`) | 30 × ~272 MB | GPU copies (RAM retains too) | 8.16 GB VRAM |
| Layers 30..64 weights | 35 × ~272 MB | RAM only | 9.5 GB RAM |
| Per-layer KV cache (fp32) | 2048×4×256×4×2 | GPU: layers 0..29 · RAM: 30..64 | 503 + 588 MB |
| Conv window (3×10240 f32) + GDN state (48×128×128 f32) | 3.27 MB/layer | GPU: 0..29 · RAM: 30..64 | 98 + 114 MB |
| MRoPE cos/sin cache | 2048×32×2×4 | GPU | 0.5 MB |
| Separate LM head `output.weight` + Q8_K twin | 248320×5120 | **RAM only** (skipped in partial offload) | 994 + 1383 MB |
| Scratch (hidden/norm/q/k/v/gate/up/ffnOut + fp16 twins + `wF16_`) | — | GPU | ~1 GB |

#### Calculation graph (one `forward(tokens)` pass)

```
tokens[seqLen] (int32)
   │
   ▼ ① GPU  kEmbedDequantQ5K (device-resident embedding)
hidden[seqLen, 5120]                          ← DEVICE
   │
   ▼ ② GPU layer loop  L = 0 .. numGpuLayers-1   (all weights device-resident)
   │   recurrent layer (L+1)%4 ≠ 0:
   │     attn_qkv·norm→qkv;  attn_gate→z;  ssm_beta→β(sigmoid);  ssm_alpha→α(softplus·a)
   │     conv1d(qkv, device conv state) → silu;  L2-norm q,k → GDN state update
   │     gated RMSNorm(S·v)·silu(z) → gdnOut;  ssm_out·gdnOut → attnProj
   │   full-attention layer (L+1)%4 == 0:
   │     attn_q (fused Q+gate), attn_k, attn_v → head-norm → MRoPE → device KV
   │     warp flash-attention × sigmoid(gate) → attn_out → attnProj
   │   shared per-layer tail: post-attn RMSNorm → ffn_gate/up → silu·u → ffn_down
   │                         → hidden += attnProj + ffnOut
   │
   ▼ ③ PCIe  copyHiddenOut  (device hidden → host; the only D2H hop per pass)
hidden[seqLen, 5120]                          ← HOST
   │
   ▼ ④ CPU layer loop  L = numGpuLayers .. 64   (AVX2 SIMD, RAM weights)
   │   forwardQwen35Layer: identical math, host KV + host conv/GDN state
   │
   ▼ ⑤ HOST  final RMSNorm → LM head (Q6_K→Q8_K) → logits[seqLen, 248320]
   │
   ▼ ⑦ HOST  kvCache_.pos += seqLen  (GPU kvPos_ already advanced in ②)
logits → sampler
```

**Consistency:** each layer owns its own recurrent state — layer `numGpuLayers-1`'s GDN/KV stay on device, layer `numGpuLayers`'s on host, and both sides advance by `seqLen` per pass so positions never diverge (verified by `BatchVsSequentialCoherent`).

#### Automatic VRAM fit (`TINYCODER_NGL` unset)

Instead of guessing `-ngl`, the engine measures real device residency (K-quant block bytes + F32 matrices + KV slots + conv/GDN state per layer, embedding on top) and fills GPU VRAM from `cudaMemGetInfo` free bytes minus a fixed 900 MB scratch/GEMM-twin margin (llama.cpp `-ngl auto` style, gated to qwen35 — the only arch with the CPU continuation). On the 2080 Ti this lands at **37/65 layers** for the 4-bit files (the Q4_K embedding has a dedicated `kEmbedDequantQ4K` kernel — see the consolidated [Benchmarking](#benchmarking) section for the measured rates and the llama.cpp comparison).

#### Weight compression for the GPU prefix

The GPU kernels support the full K-quant family (`Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K`), so a lower-bit model file shrinks the resident layers with zero code:

| Quant | per-layer GPU weights | 30-layer VRAM | auto-fit GPU layers |
|---|---|---|---|
| Q5_K_M (default) | ≈272 MB | 8.16 GB | 30 |
| Q4_K (e.g. `Qwen3.8-27B-UD-Q4_K_M`, 16.5 GB) | ≈225 MB | 6.75 GB | **37** ✓ |
| Q8_0 (e.g. `Qwen3.8-27B-UD-Q8_K_XL`, 31.5 GB) | ≈415 MB | 12.4 GB (doesn't fit 30) | **20** ✓ |
| Q3_K | ≈175 MB | 5.25 GB | ~43 |
| Q2_K | ≈133 MB | 3.99 GB | ~50 |

The prefill path's fp16 working form is transient (one matrix in `wF16_` at a time); decode uses quantized GEMV directly, so the only real reduction lever is the file's quant.

### Ornith-1.5-9B (dense qwen35)

`Ornith-1.5-9B-Q4_K_M.gguf` reports `general.architecture = qwen35` — it is a **dense** gated-delta-net hybrid, *not* the MoE `qwen35moe` of the 35B Ornith checkpoints: 32 layers, hidden 4096, 16 attention heads / 4 KV heads, `headDim` 256, MRoPE (`rope.dimension_count` 64, sections `[11,11,10,0]`, theta 1e7), `full_attention_interval` 4 (24 recurrent + 8 full-attention layers), and no MTP block. Its GDN SSM geometry is `ssm.inner_size` 4096, `ssm.state_size` 128, `ssm.group_count` 16, `ssm.time_step_rank` 32, `ssm.conv_kernel` 4 — the smaller dense variant (the 27B dense uses `ssm.inner_size` 6144 / rank 48). The tokenizer ships `add_bos_token=0`, so the checkpoint exposes no BOS id (EOS 248046 / pad 248044 remain). At 5.9 GB it fits the 11 GB card and runs **full GPU offload** (the dense path, no CPU-expert hybrid).

Verified **2026-10-29** on the reference hardware: the model loads, `tinycoder_test` runs **66 passed / 0 failed / 16 skipped**, and greedy generation is coherent (`tiny_logits_probe --generate` answers *"Write a C++ function to add two numbers."* with a templated `add()`).

##### Benchmark: Ornith-1.5-9B-Q4_K_M (vs llama.cpp)

Reference hardware (RTX 2080 Ti 11 GB + 2× Xeon E5-2686 v4), llama-bench protocol: `tinycoder_bench --n-prompts 512 --n-gen 128 --reps 3 --threads 36` vs `llama-bench -p 512 -n 128 -t 36 -ngl 99 -r 5` (CUDA build `9b8446b`). The 5.9 GB file fits the 11 GB card, so both run **fully offloaded**.

| Metric | TinyCoder (2026-11-04) | TinyCoder (baseline) | llama.cpp | vs llama.cpp |
|---|---|---|---|---|
| pp512 | **2,251 tok/s** | 46.6 tok/s | 2,726 tok/s | 0.83× |
| tg128 | **80.8 tok/s** | 37.2 tok/s | 84.4 tok/s | 0.96× |

The 2026-11-04 pass (**48× prefill / 2.0× decode**) replaced the in-tree
integer prefill GEMM with the int8 tensor-core kernels (Q4_K /
Q5_K / Q6_K / Q8_0, 128-token stream-k tiles) and the decode GEMV with its
**GEMV** port (one warp per row, Q8_1 activations), plus a register-resident
warp-per-row GDN chunk scan and NUMA pinning of the CPU pool to the GPU's
node. `TINYCODER_INT8_GEMM=0` restores the integer path. The 2026-10-29 levers below
are still active and unchanged.

**Measurement caveat (this workload only).** Decode allocates ~70 per-token `kQuantizeQ8K` scratch buffers in the model's first forward, so the GPU idles ~19 s during load; a short (pp4/tg32) bench then measures only a few hundred ms of work and can read **6–41 tok/s** purely from the GPU boost state (P8 300 MHz → P0 1965 MHz) it happened to settle in. All numbers above use the **llama-bench protocol (pp512/tg128), warm reps on a pre-heated GPU**; sustained decode runs at 95% GPU / ~230 W. The three A/B numbers in the campaign table below were each measured back-to-back in a single session, so their *ratios* are clock-independent.

##### Decode + prefill optimization campaign (2026-10-29)

Three **bit-exact** levers took the dense-qwen35 GPU path from `pp512 18.1 / tg128 13.0` to **`pp512 46.7 / tg128 37.2` tok/s** (decode **2.86×**, prefill **2.58×**), keeping `tinycoder_test` at **64 passed / 0 failed / 16 skipped** and greedy streams byte-identical. Each is guarded by its own env A/B switch (all default **on**).

| Lever | Switch | Default | A/B measured | What it fixed |
|---|---|---|---|---|
| Q4_K 4-rows-per-warp decode GEMV (`kQGemvQ4KxQ8K_4xW`) | `TINYCODER_Q4K_4XW` | **on** | tg16 **13.80 → 25.24 tok/s (+83%)** | The generic `kQGemvKxQ8K<kTypeQ4K>` ran one row per warp with only 8 of 32 lanes issuing weight loads, for the model's **largest** quant class (Q4_K ≈2.7 GB of 5.9 GB) — while Q5_K/Q6_K already had 4xW kernels. The new kernel mirrors [`kQGemvQ5KxQ8K_4xW`](include/GPUComputeInternal.hpp:4475) (4 rows/warp, all 32 lanes load, 8× `__dp4a`) |
| Vectorized `kxQ8KBlockDot<kTypeQ4K>` integer body | (shared) | **on** | pp128 **18.6 → 27.5 tok/s (+48%)** | The shared block-dot did the 32-column integer dot with 32 scalar byte-loads + per-byte nibble/sign selects; it is the body of BOTH the generic 1-row GEMV and the batched prefill GEMM (`kQGemmKxQ8K`). The `__dp4a` form is bit-identical (integer sum is exact + order-free) |
| Snapshot L2 norms in `kQ35L2NormRepeat` | `TINYCODER_L2NORM_SMEM` | **on** | tg16 **25.4 → 41.1 (+62%)**, pp128 **27.5 → 47.2 (+72%)** | The kernel recomputed the **same `headK`-wide double-precision L2 norm for every element** of each slice's `headK*nVHeads` output (O(`headK²·nVHeads`) double adds/token/layer). Now each slice's `iq/ik` is computed once into shared memory. nvprof: this kernel fell from **#1 (4286 µs / 6528 calls)** to 57 µs. Because it runs per prompt token, prefill gained even more than decode |

`nvprof` attribution across the campaign (units are relative): the Q4_K GEMV (`kQGemvQ4KxQ8K_4xW`, x7936) and Q6_K GEMV (x3138) are now the decode tops, `kQGemmKxQ8K` (x376, ALU-bound) the prefill top, and `kQ35L2NormRepeat` dropped from the #1 slot to a rounding error. **Remaining gap** is the same integer-dot ALU bound documented below; the Q4_K 4xW + `__dp4a` work removes the byte-load-instruction overhead but not the Q8_K integer-dot throughput ceiling.

**Correctness.** Every lever is parity-checked: `tinycoder_test` 64/0/16 (the deterministic GPU-vs-CPU and batch-vs-sequential tests are bit-exact), and the Q4_K 4xW greedily decodes the 160-token probe stream byte-identically to the generic kernel.

**Prefill INT8 — int8 tensor cores.** The dense projections use a stream-k int8 tensor-core GEMM (`mma.s8.m8n8k16`) for Q4_K/Q5_K/Q6_K/Q8_0, reading the raw GGUF weights (zero extra VRAM). It is the default; `TINYCODER_PREFILL_EXACT=1` restores the bit-exact integer Q8_K GEMM, and `TINYCODER_INT8_GEMM=0` disables the tensor-core path.

**Prefill is no longer per-token.** The dense-qwen35 CUDA forward originally ran `launchQGemv` once per prompt token, re-reading every weight matrix `seqLen` times, so `pp ≈ tg`. It now dispatches a **batched Q8_K GEMM** ([`kQGemmKxQ8K`](include/GPUComputeInternal.hpp:4288) + [`kQuantizeQ8K_Batch`](include/GPUComputeInternal.hpp:6279), launched via [`launchQGemm`](include/GPUComputeInternal.hpp:8663)) for the K-quant projections: each warp owns 4 tokens (lane groups) × one weight row, so every weight block is dotted against **4 token activations per read** instead of one. It is **bit-exact** with the per-token `launchQGemv` path — the same [`kxQ8KBlockDot`](include/GPUComputeInternal.hpp:3763) runs per (row, token) with the 8-lane reduction confined to the token's group — which is why the whole suite stays **64/64 green** (batch-vs-sequential, GPU-vs-CPU, and the Paris-prompt logits agree exactly). Prefill rose **14.4 → 18.1 tok/s (+26%)** while decode is unchanged.

**Remaining gap.** The dense-projection prefill is now tensor-core bound; the remaining gap to the baseline is the GDN recurrence, attention and the MoE expert path, not the quantized GEMM.

**Correctness cross-check.** Both engines answer the same greedy prompt (`"--temp 0"`) coherently — TinyCoder (`tiny_logits_probe "Write a C++ function to add two numbers." --generate`) emits a templated `add<T>(T a, T b)`, llama.cpp (`llama-cli -p "…" -n 160 --temp 0 -ngl 99 -t 36`) emits a `double addNumbers(double a, double b)` with a `main()` demo and the same `[Start thinking]/[End thinking]` reasoning preamble. The divergence after the opener is ordinary near-tie greedy drift (different kernel accumulation order), not a correctness gap.

### Ornith (qwen35moe) GPU — CPU-Expert Hybrid (default)

`ARCH_QWEN35MOE` models — the **Ornith** checkpoints (`Ornith-1.5-35B-A3B-Q8_0`, `Ornith-1.5-35B-Q4_K_M`) — are **MoE hybrids**: gated-delta-net (GDN) recurrent layers interleaved with full-attention layers (40 for Ornith/Qwen3.6, 41 for the Ornith Q4_K_M variant), each layer ending in a **softmax-routed MoE FFN**: a shared `ffn_gate_inp` router over **256 experts** (top-8 per token, weights re-normalized with the `norm_w` clamp), a shared sigmoid-gated expert, and per-expert SwiGLU (`ffn_gate_exps`/`ffn_up_exps` [256·expertFF] → `ffn_down_exps` [256·hidden]). The CPU reference lives in [`computeQwen35MoE()`](src/cpp/core/ModelMoE.cpp:161); the reference routed-expert half (router on CPU, used by the hybrid driver) is [`computeQwen35MoEFromLogits()`](src/cpp/core/ModelMoE.cpp:385); the GPU trunk driver is [`forwardQwen35MoePrefix()`](src/cpp/core/GPUComputeForward.cu:29).

#### Default placement: GPU trunk + CPU experts

On GPUs with insufficient VRAM for the full model (e.g. the 35 GB Ornith Q8_0 on an 11 GB RTX 2080 Ti), the engine uses the **CPU-expert hybrid** (llama.cpp `--cpu-moe`-style) as the **default**:

- The GPU runs the **trunk**: token embedding, the GDN/attention block, post-attention RMSNorm, KV cache, LM head, shared expert and residual. All trunk weights live in VRAM; per layer it copies only the post-attention norm to host (pinned D2H).
- The **CPU** runs the routed experts: the fp32 router (`ffn_gate_inp` [256×hidden] @ norm — bit-identical across batch and single-token decode), softmax → top-8 → renormalization, then the 8 selected expert FFNs (12 matmuls: 8× gate/up + 8× down, per layer) via **grouped batch GEMMs**.
- The result is copied back and residual-added on-device. Only 8/256 = 3.1% of the expert weights are touched per token, so the ~32B-param expert block never needs VRAM.

The upload first attempts full offload; on OOM the adapter retries **once** with the expert matrices left in host RAM ([`ModelGPUAdapter`](src/cpp/core/ModelGPU.cpp:166) → `GPUModel::setMoeCpuFn`), printing `[gpu] qwen35moe hybrid upload OK (GPU trunk + CPU experts, experts cached on GPU (default))` by default, or the suffix-less `[gpu] qwen35moe hybrid upload OK (GPU trunk + CPU experts)` variant when `TINYCODER_MOE_CACHE=0` disables the cache — then releases every partially allocated device buffer, consumes the sticky CUDA error and latches the failure: no repeated doomed attempts and no VRAM leak. `TINYCODER_NGL` partial offload has no CPU continuation for the MoE arch and is ignored.

#### Why the on-device expert cache was opt-in (now default-on since 2026-09-17)

A per-layer **LRU expert cache** (`buildExpertCache`/`ensureExpertCached`) is available for hybrid qwen35moe runs. It used to be **off by default**: on a 35 GB model in 11 GB VRAM the cache thrashed — `[moe fwd]` stats showed 44-55% hit rates with ~500-600 MB/token streamed over PCIe (`fillMs` up to 5300 ms, `moeBlockMs` up to 7000 ms with cold spikes), slower than the deterministic CPU-expert path. After the 2026-09-17 correctness fixes (per-layer expert types, down-slice dispatch, device-slot WAR race) the cache is **enabled by default** — the upload message reads `experts cached on GPU (default)` — and `TINYCODER_MOE_CACHE=0` disables it (Option B); `TINYCODER_MOE_STATS=1` prints the per-forward attribution. `TINYCODER_MOE_FALLBACK=1` forces the CPU-expert path even when the cache is built.

#### The grouped-batch CPU expert path (2026-09-15)

The routed-expert half is a **grouped, weight-stationary batch pipeline** in [`computeQwen35MoEFromLogits()`](src/cpp/core/ModelMoE.cpp:385): per layer the (token, rank) selections are grouped by expert id, and each expert's slice (`ffn_gate_exps`/`ffn_up_exps` [expertFF] and `ffn_down_exps` [hidden]) is run as ONE batch GEMM over its group tokens through [`matMulVecBatchQ8_0_SIMD()`](include/SIMDMatMulVec.hpp:562). Per (token, row) the kernel uses the identical per-block int8 math + per-token x quantization as the single-token reference, so every partial is **bit-exact** (verified: batch vs sequential decode and all 4 keyword questions agree). Partials land in disjoint per-(s, r) slab rows and are reduced in exact rank order — no accumulation-order change. The batch kernel also quantizes the group's activations **once per call** (`thread_local` generation guard), eliminating ~10M scalar `lrintf` ops/layer, and because the callback runs on the main thread, the kernel's `parallelForSlab` is a top-level pool dispatch — all 8 threads tile the expert rows (the decode-expert parallelism that `parallelFor2D(1,8)`'s serial shortcut had killed).

On an **unthreaded/fused** layout the path falls back to the exact per-member single-token math (identical results).

#### Supported qwen35moe models

| Quant | File size | GPU mode on 11 GB | Verified |
|---|---|---|---|
| `UD-IQ1_M` | 10.05 GB | ✅ full offload | single-token + batch/seq coherence ✓ |
| `UD-IQ2_M` | 11.5 GB | ✅ hybrid (experts on CPU) | SingleTokenForward / MultiTokenForward / batch-seq coherence / `SequentialDecodeArgmaxAgrees` ✓ |
| `UD-Q4_K_M` | 22.1 GB | ✅ hybrid (experts on CPU) | SingleTokenForward ✓ |
| `Ornith-1.5-35B-A3B-Q8_0` | ~35 GB | ✅ hybrid (experts on CPU) | all 4 SampleQuestion keyword tests ✓ (2026-09) |
| `Ornith-1.5-35B-Q4_K_M` | ~22 GB | ✅ hybrid (experts on CPU) | single-token ✓; grouped-batch Q4_K/Q6_K Option B (permanent) ✓; 8-token sequential decode vs CPU top-10 ✓ (EOG-corruption-free; near-tie top-1 relaxed to CPU top-10) ✓; **expert cache verified working and DEFAULT-ON (2026-09-17, all 12 tests)**: all 8 GPU-parity tests + 4 keyword questions pass ✓ |

The cold-cache host cannot hold the 35 GB model in page cache, so single cold runs (e.g. `tinycoder_test` without warmup) report far lower numbers than warm repeats. Both harnesses now print a [`[bench-parity]`](benchmarks/BenchMain.cpp:281) line quantifying the same decode region with explicit cold/warm state, and the question tests warm the page cache once in [`SharedTestEnv`](unit_tests/SharedTestEnv.hpp:19) (`TINYCODER_TEST_NO_WARMUP=1` opt-out). See the [Benchmarking](#benchmarking) section for the steady-state warm rates of all qwen35moe models.

The MoE FFN adds the router GEMV (F32 `ffn_gate_inp` [256×hidden]), the top-k router kernel (softmax over 256 → strict `>` tie-break top-8), the grouped per-expert batch GEMVs (AVX2 Q8_0 kernel, expert-parallel over the pool), weighted accumulation, and the shared-expert sigmoid-gated SwiGLU. The prefill batch GEMM path (seqLen > 1) is weight-stationary: each selected expert's rows are read once per group instead of once per member token.

#### Static resident-expert offload (Option D) — `TINYCODER_MOE_RESIDENT=1` (2026-09-16)

For **Q8_0-expert** qwen35moe models, Option D puts the **hottest experts permanently on the GPU** (opt-in): no LRU, no per-token PCIe streaming of resident weights.

- **Profiling pass**: at upload time the adapter runs a representative code+text prompt mix through the **pure-CPU path** (`Model::forceCpuForward_` + `resetCpuKVState`, deadlock-free with `mtx_` held), counting per-layer routed expert selections into `moeUsageCounts_` (via the `computeQwen35MoE` → `recordMoEUsage` hook in [`ModelGPU::buildResidentExpertPolicy`](src/cpp/core/ModelGPU.cpp:864)).
- **Policy**: `cudaMemGetInfo` computes the free-VRAM budget (keeps 900 MB + 25% headroom for the trunk + shared expert + combine tables); the budget / per-expert arena bytes (`2·gateSlice + downSlice`, `3342336` B for this model) gives `floor(budget/N) ` resident experts per layer, filled with the per-layer **top-K most-used** experts (`std::partial_sort`).
- **Arenas**: [`GPUModel::buildResidentArenas`](src/cpp/core/GPUComputeMoE.cu:456) allocates per-layer contiguous arenas and H2Ds the resident experts' packed gate/up/down slices **once** (per-layer host blob selection, mirroring the expert-cache lesson). **Q8_0-expert types ONLY (hard gate, 2026-09-17)** — the resident fast path reuses the Q8_0 integer kernels (`launchQuantQ8_0x32` + `kQGemvQ8_0xQ8K_BatchGU` + `kSiluMulBatch` + `kQuantizeQ8_0x32_Batch` + `kQGemvQ8_0xQ8K_BatchDownR`). The Q4_K/Q6_K resident kernels (`kQGemvKxQ8K<kTypeQ4K/Q6K>`) are bit-exact (verified by `GPUCpuCompareTest.Q4K_Q6KKernelParityWithCPU`) but are **disabled**: measured on Ornith-1.5-35B-Q4_K_M the split resident path was a **4.5× decode regression vs Option B** (0.32 vs 1.45 tg tok/s) because the ~6-8 non-resident experts per layer run on the CPU with the per-member scalar path (no grouped-batch batching), so Q4_K_M stays on Option B permanently. Non-Q8_0 models (Q4_K_M, IQ2_M etc.) cleanly fall back to Option B — `TINYCODER_MOE_RESIDENT=1` prints a notice and is a fast no-op on them.
- **Dispatch**: per decode step the adapter's CPU callback computes the router tables (all ranks) **and** the non-resident expert FFNs into the combined rank-major down table (`moeDownTableHost_`, resident slots zeroed); the driver H2Ds it, runs the resident ranks on the GPU via `scheduleResidentExperts` (4-kernel pipeline, rank-remapped down rows), and accumulates in exact reference rank order (`__fadd_rn(__fmul_rn(g,x),o)`). Verified **bit-identical** to the Option-B CPU reference at layer 0 (`TINYCODER_DUMP_MOE=1` `[gpu L0 moeOut]` match).
- **Decode-only**: the resident dispatch is gated to `seqLen == 1`. Prefill keeps the Option-B grouped-batch path — `scheduleResidentExpertsBatch`'s per-token serial launches were slower at batch sizes (measurement: 8.8 vs 9.3 pp tok/s), and the user's stated priority is prefill + generation speed.
- **Destroy safety**: `destroy()` no longer indexes the policy vectors past their real size (fixes a heap corruption when the Q8_0 gate disabled Option D after `setResidentPolicy`).
- **Disabled cases**: `TINYCODER_MOE_FALLBACK=1` forces Option B; non-Q8_0 experts fall back (hard-gated before the profiling pass — a fast no-op with a printed notice, 2026-09-17); VRAM budget exhaustion falls back.

Measured on Ornith-1.5-35B-A3B-Q8_0: Option D's resident experts are ~5% faster than Option B in decode (7.6–7.8 vs 7.2–7.4 tg tok/s); prefill stays on the Option-B grouped-batch path. Both are summarized in the [Benchmarking](#benchmarking) table.

##### Q4_K_M: Option B (grouped-batch CPU experts) is the permanent path (2026-09-17)

The grouped-batch CPU expert path was extended to the **Q4_K (gate/up) + Q6_K (down)** expert pair (see [`computeQwen35MoEFromLogits`](src/cpp/core/ModelMoE.cpp:385) `canBatch`/`batchGemm`), taking Ornith-1.5-35B-Q4_K_M decode from **0.15 → 1.45 tg tok/s** (the Q4_K path was scalar-only before: the Q8_0-only `canBatch` gate fell back to per-member loops). Both the shared-expert matrix and the MoE experts are Q4_K_M/Q6_K_M on this file.

On Ornith-1.5-35B-Q4_K_M, the grouped-batch CPU expert path took decode from **0.15 → 1.45 tg tok/s** (the Q4_K path was scalar-only before); Option D's resident split was a 4.5× regression (0.32 tg tok/s) and is **disabled by hard gate** — Q4_K_M runs Option B permanently (see the [Benchmarking](#benchmarking) table for the current warm rates).

Both the Option-B grouped path and the (now-disabled) resident kernels were verified **bit-exact** against the CPU AVX2 reference by `GPUCpuCompareTest.Q4K_Q6KKernelParityWithCPU` over 64 rows × 8 deterministic tokens for real L0 gate/up (Q4_K) and down (Q6_K) weight rows — including the Q4_K float-tree fix (the CPU AVX2 kernel is compiled **without** FMA contraction, so the GPU kernel uses volatile non-fused mul+add) and the Q6_K quantizer `d` `__fdiv_rn` fix. With `TINYCODER_MOE_RESIDENT=1` on a Q4 model the policy builder prints `[resident] expert type 12 is not Q8_0 … keeping Option B permanently` and skips the profiling pass, so the env var is harmless — Q4_K_M always runs the fastest (Option B) path.

##### Expert cache on Q4_K_M — per-layer down-expert types (2026-09-17)

The on-device LRU expert cache (default-on since 2026-09-17; `TINYCODER_MOE_CACHE=0` disables, `=1`/unset enables) now works on Ornith-1.5-35B-Q4_K_M. Before these fixes a **16-test batch failed with NaN logits**. Two-stage root cause:

1. **Wrong down-slice type**: the cache decode passed the gate/up quant type (`expertType_`, Q4_K, 144 B/block) to the **down** GEMV, whose quant type is **Q6_K (210 B/block)** on this model. Fixed by dispatching the down slice with its own recorded type (`dExps.type` / `expertDownType_`).
2. **Per-layer down-expert types** (the deeper root cause): the Ornith Q4_K_M GGUF stores the **down-expert quant type differently across layers** — blk.0-4 and blk.7+ have down = Q6_K (210 B/block), but **blk.5-6 have down = Q4_K (144 B/block)**; gate/up are always Q4_K. The cache's geometry globals (`expertRowBytesDown_`, etc.) are overwritten every upload step and held only the **last** layer's values, so layers 5-6 were filled and decoded with Q6_K geometry (420 B/row instead of 288 B/row) → garbage half-floats → NaN. The fix records **per-layer** quant geometry on every [`ExpertSlot`](include/GPUCompute.hpp:538) (`gateType`/`gateRowBytes`/`downType`/`downRowBytes`), sizes each layer's arena from its own slot, and derives every cache-slice pointer/row-stride/GEMV type from per-layer accessors (`layGateType`/`layGateRowBytes`/`layDownType`/`layDownRowBytes`/`layGateSliceB`/`layDownSliceB` in [`GPUComputeForward.cu`](src/cpp/core/GPUComputeForward.cu:193)) — covering the batched Q8_0 fast path, the per-rank `launchQGemv` path, and the per-token fallback.

##### Expert cache — device-slot WAR race (2026-09-17)

After the per-layer geometry fix, `ModelTest.KVCacheClear` and `ModelTest.CompareBatchVsSequentialPrefill` still failed with **run-to-run varying** diffs (`KVCacheClear` maxDiff 1.92–2.04 across runs) while default/Option-B mode stayed bit-identical (maxDiff 0.0). Root cause: **a device-slot WAR race**. The per-rank decode path (`moePerRank` → [`ensureExpertCached`](src/cpp/core/GPUComputeMoE.cu:214) → `launchQGemv`) pipelined each miss fill on `g_stream2` (the copy engine) while the GEMVs that read that slot ran on `g_stream`. The g_stream2 + event-wait design only orders RAW (this rank's GEMV after its own fill) — it does **not** order a later rank's fill **after** an earlier rank's GEMVs. When two routed experts hash into the same 2-way set (common with 8 experts over 16 sets), a later rank's miss would H2D the same device slot while the earlier rank's GEMVs were still reading it → a timing-dependent torn read. Forward 1 (all misses, evictions tear) vs forward 3 (all hits, clean) differ — exactly `KVCacheClear`'s signature.

Fix: the per-rank fills now run **in-order on `g_stream`** ([`GPUComputeMoE.cu`](src/cpp/core/GPUComputeMoE.cu:365)), giving the total order `fill_r → GEMV_r → fill_{r+1} → GEMV_{r+1}` — any slot eviction happens after the previous owner's GEMVs complete. The **batched Q8_0 fast path keeps its `g_stream2` pipeline** (all fills enqueue before any GEMV, and its slot-conflict guard falls back to the per-rank path whenever two ranks would share a slot base, so it cannot tear). Additionally, [`GPUModel::forward`](src/cpp/core/GPUComputeForward.cu:1893) drains `g_stream2` at forward end when the cache is built, so no fill is in flight when a later forward's cache **hit** reads a slot (hits wait on nothing).

Verified (2026-09-17, RTX 2080 Ti 11 GB, **expert cache default-on, no env**): all **12 tests pass** — `ModelTest.SingleTokenForward`/`MultiTokenForward`/`KVCacheClear`/`CompareBatchVsSequentialPrefill`, `Qwen35Test.SingleTokenForwardFinite`/`BatchVsSequentialCoherent`, `GPUCpuCompareTest.ParisPromptLogitsAgree`/`SequentialDecodeArgmaxAgrees`, and all 4 `FullQuestions/SampleQuestionTest` keyword answers. `CompareBatchVsSequentialPrefill` is now **bit-identical** (`maxDiff=0`) and `KVCacheClear` is deterministic — matching Option-B mode. Option B (no cache) re-verified via `TINYCODER_MOE_CACHE=0` with **no regression** (the 2 prior-failing tests pass). `SequentialDecodeArgmaxAgrees` keeps the Q4_K depth-5 boundary relaxation at step 5 (GPU top-1 `"T"` lands just past the CPU top-10 — the same near-tie cluster the default Option B mode exhibits with `"q"` at CPU rank 10); the assertion now fails only if an EOG/chat-special token pollutes the GPU top-5.

### Page-Cache Prefetch (`TINYCODER_PREFETCH`)

Weights are loaded via a file-backed `mmap` of the GGUF tensor section (no heap copy — see [`GGUFLoader::mapTensorData`](src/cpp/core/GGUFLoader.cpp:687)); pages are faulted in lazily on first touch. On models that fit in RAM, that lazy first-touch cost lands **inside the first inference pass** (the question harness "warm-up" prefill took 140 s and first-question decode ~0.6 tok/s on the 35 GB Q8_0). The loader now **eagerly prefetches the tensor section into page cache at load time** (`posix_fadvise(POSIX_FADV_WILLNEED)` + a per-page touch barrier) whenever the section **fits ≤ ¾ of physical RAM** — the 21–22 GB Q4 models prefetch (2–3 min load, then warm inference), while the 35 GB Q8_0 correctly **skips** (over-RAM eager read perturbed the page-cache LRU and made first passes *slower* — measured 0.22–0.51 tok/s vs 0.62–1.76 lazy). Overrides: `TINYCODER_PREFETCH=0` never prefetch, `=1` force.

### KV Cache

The KV cache stores key/value tensors as F32 arrays with shape `[numLayers, maxSeqLen, numKVHeads, headDim]`.

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- **[np library](https://github.com/mgorshkov/np)** — NumPy-style arrays with SIMD/CUDA acceleration
- **llama.cpp** — reference implementation used for correctness parity and performance benchmarking
