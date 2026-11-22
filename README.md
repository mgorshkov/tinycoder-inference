# ⚡ TinyCoder Inference Engine for NVIDIA GPUs (Turing / Ampere / Ada)

**High-performant local inference engine** for the [TinyCoder AI](https://github.com/mgorshkov/tinycoder) VS Code extension and the [TinyCoder OpenAI](https://github.com/mgorshkov/tinycoder-openai) project, which exposes the same engine through an OpenAI-compatible API. This repository contains **only the engine**: the C++/CUDA core (`src/cpp/core`), public headers (`include/`), unit tests (`unit_tests/`), benchmarks (`benchmarks/`), and build plumbing.

The engine runs the **Qwen2.5** (`qwen2`), **Qwen3.6** and **Qwen3.8** (`qwen35` dense, `qwen35moe` mixture-of-experts), **Ornith**, **Gemma 4**, and Prism **Ternary-Bonsai** model families from **GGUF** or **EXL3** (`trellis` / `mul1`) checkpoints — e.g. `Qwen3.5-27B` at 1.60 bpw — with **mixed quantization**, **AMX/AVX-512/AVX2 CPU acceleration** (runtime SIMD dispatch), and a **CUDA GPU offload engine**. It is designed to run efficiently on limited hardware.

> **Written from scratch — no inference libraries or engines.** Every part of TinyCoder is original code in this repository: the C++/CUDA kernels, GEMV/GEMM and MMA schedules, GGUF and EXL3 loaders, quantizers/dequantizers, tokenizer, scheduler, KV cache, and CPU SIMD paths. It does **not** link, embed, vendor, or call llama.cpp, ggml, Candle, PyTorch, ExLlamaV3, or any other inference runtime; there is **no inference-library dependency** at build or run time. The only third-party code is non-inference plumbing: the [np](https://github.com/mgorshkov/np) array utility, a shared CMake config, and GoogleTest for unit tests (the CUDA Toolkit/cuBLAS is the system GPU runtime). All performance work is TinyCoder's own kernels.

> **GPU support.** The CUDA engine runs natively on **Turing, Ampere and Ada GPUs — including the RTX 2080 Ti, RTX 3090, RTX 4090 and RTX 5090** — plus newer architectures via forward-compatible PTX JIT, with runtime capability dispatch and a CPU fallback. See [Supported GPUs](#supported-gpus).

## Build & Install

### Prerequisites

- **C++20 compiler** (GCC ≥ 13, Clang ≥ 14, or MSVC 2019+)
- **CMake** ≥ 3.18

### Strongly recommended (large speed-up)

TinyCoder builds and runs with only the base toolchain, but **both items below are strongly recommended** — without them you get the unoptimised fallback, not the numbers in [BENCHMARKS-2080TI.md](benchmarks/BENCHMARKS-2080TI.md):

- **CUDA Toolkit** ≥ 11.0 + nvcc — **install this if you have any NVIDIA GPU.** CMake enables the GPU offload automatically when nvcc is on the `PATH` (`-DENABLE_CUDA=OFF` forces a CPU-only build); a CPU-only `tinycoder_bench` rejects `--gpu` with an explicit "needs a CUDA build" message rather than an unknown-option error. It enables the GPU offload engine on Turing / Ampere / Ada and newer (RTX 2080 Ti, RTX 3090, RTX 4090, RTX 5090). Without it the engine runs on the CPU only, which is several times to orders of magnitude slower depending on model size; on a model that does not fit in RAM it will not run at all.
- **OpenMP** — **usually already present with the compiler**, so this normally costs nothing. It parallelises the CPU path across all cores; without it CPU inference is single-threaded.

### One-shot build

```bash
./scripts/build.sh                          # GPU build (-DENABLE_CUDA=ON) — default
./scripts/build.sh --no-cuda                # CPU-only build (-DENABLE_CUDA=OFF)
./scripts/build.sh --cpu --no-tests         # CPU-only static lib, no tests/bench
```

The script configures CMake, builds `tinycoder_core` (+ `tinycoder_test` and `tinycoder_bench`), and runs CTest.

### Manual build

```bash
# CUDA GPU offload — the default whenever nvcc is found
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# CPU-only: force it explicitly (a box without nvcc gets this automatically)
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build-cpu --config Release -j

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

All build-time cache variables (`ENABLE_CUDA`, `ENABLE_AVX512`, `BUILD_TESTS`, …) with their
defaults are listed in [ENVIRONMENT.md](ENVIRONMENT.md#cmake-options).

This installs the static library and headers (`install(TARGETS tinycoder_core …)` / `install(FILES ${TINYCODER_HEADERS} DESTINATION include/tinycoder)`).

## Benchmarking

GPU performance against llama.cpp and exllamav3 — the comparison table, the measurement
protocol, and the procedure used for the exllamav3 baseline — lives in
[BENCHMARKS-2080TI.md](benchmarks/BENCHMARKS-2080TI.md), with the Ampere table in
[BENCHMARKS-3090.md](benchmarks/BENCHMARKS-3090.md).

## Key Features

### ⚡ SIMD CPU Acceleration

- **Runtime SIMD dispatch** — auto-detects the best available instruction set: **AMX** (Intel Sapphire Rapids+), **AVX-512** (Skylake-X / Ice Lake+), **AVX2** (Haswell+ / AMD Zen+), or a universal **Scalar** fallback. The configure step reads the build host's CPU flags too and enables AVX-512 only when that CPU has it, so a default build runs on the machine that produced it
- **OpenMP** thread parallelism for all matrix operations

### 🎮 CUDA GPU Offload Engine

- Full-model GPU offload:
  - **Prefill** — two paths, selected per projection: cuBLAS fp16 tensor-core GEMM, and a from-scratch int8 `mma.s8.m8n8k16` stream-K GEMM that quantizes activations to Q8_0 and reads GGUF blocks straight from VRAM
  - **Decode** — quantized on-the-fly-dequant GEMV kernels reading raw GGUF blocks from VRAM, with an int8 KV store (or an f32 store plus an int8 decode mirror, or plain f32 — whichever the planner's fit chose) for qwen35 full-attention layers, with per-slot row tracking so repeated `generate()` calls cannot reuse a previous conversation's rows
  - **Attention** — warp flash attention (`kWarpAttention<HD>`, compile-time register accumulators, exactly `nHeads` warps) and the gated-delta-net recurrent kernels for qwen35's linear layers
  - **Decode capture** — the decode graph is captured into a CUDA graph (`TINYCODER_GPU_GRAPH`)
- **EXL3 native kernels** — trellis reconstruct, GEMV and a fused prefill GEMM under [`cuda/exl3/`](src/cpp/core/cuda/exl3), with the cublasLt autotuned path as default and `TINYCODER_EXL3_LT=0` reproducing the older fixed-order k summation bit-for-bit
- **Enabled by default** in builds compiled with `ENABLE_CUDA=ON`; all layers offloaded; automatic fallback to CPU when CUDA is unavailable or at the first forward error
- **Two offload modes for models that exceed VRAM**: hybrid partial offload keeps the whole model in RAM and mirrors the first `numGpuLayers` layers to the GPU, and the MoE path additionally streams experts for the 35B-A3B checkpoint through a layer arena while the dense body stays resident
- **Context window sized by memory, not by a constant** — `model.setContextLength(N)` (or `tinycoder_bench --ctx N`) asks for N tokens before the load; the planner measures free VRAM/RAM and grants the largest window that still fits the weights, the session's fixed ~1.4 GiB of scratch, and the K/V it is about to allocate. It sheds the int8 decode mirror first, then switches the K/V store to int8, so an 11 GB card holds **52,480** tokens for a 9B `q4_k_m` at f32 and **196,352** at int8 (27B EXL3: 2,560 f32 / **12,288** int8). A request that does not fit is reduced to the fit and the reason is printed; a window below 1024 tokens refuses the load. The default is unchanged at 2048 tokens, so every published throughput number still describes the shipped default. `TINYCODER_CONTEXT_DEBUG=1` prints the planner's inputs
- Runtime controls: `TINYCODER_GPU=0` forces the CPU path, `TINYCODER_NGL` limits offload to N layers (unset → automatic VRAM fit), `TINYCODER_INT8_GEMM=0` disables the int8 prefill path, `TINYCODER_KV8=0` disables the int8 decode mirror (it follows the context plan by default), `TINYCODER_CONTEXT_KV8` pins the precision the planner may grant, `TINYCODER_GPU_VERBOSE=1` prints per-stage CUDA timings. Full list in [ENVIRONMENT.md](ENVIRONMENT.md).

#### Supported GPUs

The CUDA offload engine runs natively on **Turing, Ampere and Ada GPUs — including the RTX 2080 Ti, RTX 3090, RTX 4090 and RTX 5090** — and on newer architectures through forward-compatible PTX JIT. Every path launches real CUDA kernels and queries the device at runtime, and the engine falls back to the CPU path when no usable GPU is present.

| GPU | Architecture | Compute capability | How it runs |
|---|---|---|---|
| RTX 2080 Ti | Turing | sm_75 | Native SASS (`compute_75`) |
| RTX 3090 | Ampere | sm_86 | Native SASS (`compute_86`) |
| RTX 4090 | Ada | sm_89 | sm_86 SASS or embedded PTX JIT (build `-DCMAKE_CUDA_ARCHITECTURES=89` for native SASS) |
| RTX 5090 | Blackwell | sm_120 | Embedded PTX JIT (build `-DCMAKE_CUDA_ARCHITECTURES=120` for native SASS) |

## Dependencies

**No inference-library dependency.** The engine, kernels, loaders, and tokenizers are written from scratch in this repository. The only external packages are non-inference plumbing:

- **[np library](https://github.com/mgorshkov/np)** — NumPy-style arrays with SIMD/CUDA acceleration (fetched via `FetchContent`)
- **[shared cmake config](https://github.com/mgorshkov/cmake)** — compiler checks, CUDA setup, OpenMP (fetched via `FetchContent`)
- **Google Test** — unit tests only (fetched via `FetchContent`)

All are fetched automatically; no system packages beyond a compiler, CMake, and (optionally) the CUDA Toolkit are required. No llama.cpp, ggml, Candle, PyTorch, or ExLlamaV3 code is used anywhere in the build.

## Getting a Model

TinyCoder loads two checkpoint formats:

- **GGUF** v3 — the llama.cpp/GGML container. Quantized K/IQ blocks, F32/F16 tensors, tokenizer and architecture metadata are read directly; `general.architecture` selects the model family.
- **EXL3** — the ExLlamaV3 `trellis`/`mul1` checkpoint format (HF `safetensors` + `config.json` + `tokenizer.json`).

Download the `.gguf` file and point the engine at it via `TINYCODER_MODEL_PATH` or the `Model::load()` API. For an **EXL3** checkpoint, download the HF checkpoint **directory** (e.g. `Qwen/Qwen3.5-27B-exl3-1.60bpw` — `config.json` + `*.safetensors` + `tokenizer.json`) and point the engine at the directory path; `Model::load` auto-selects [`EXL3Loader`](src/cpp/core/EXL3Loader.cpp).

[`Model::load`](src/cpp/core/ModelLoad.cpp) detects the format automatically, and both formats use the same CUDA/CPU kernels, KV cache, and offload path.

### Checkpoints behind the published benchmarks

Every row in [BENCHMARKS-2080TI.md](benchmarks/BENCHMARKS-2080TI.md) and [BENCHMARKS-3090.md](benchmarks/BENCHMARKS-3090.md) was measured on one of these files; the link is the exact artefact, so a row can be reproduced rather than approximated:

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

The EXL3 row is a **branch of a multi-checkpoint repository**, so it has no single `resolve/main` file: download the branch as a directory (`config.json`, `*.safetensors`, tokenizer files) and point `Model::load` at the directory, per the paragraph above.

## Usage (C++ API)

```cpp
#include "Model.hpp"
#include "ModelConfig.hpp"

using namespace tinycoder;

// 1. Create and load the model (GGUF v3)
Model model;
std::string err;
// Optional, and only before load(): ask for a window and let the planner grant
// the largest that fits (unset = 2048; see model.contextLength() / contextPlan()).
model.setContextLength(65536);
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

The Google Test suite (`tinycoder_test`) covers dequantization correctness across all quantization types, the context planner and its window guards, reference parity against sequential token-by-token forward passes, split-vs-single prefill, full-question generation tests, thread-pool reentrancy, and GPU/CPU comparison. Without a model it reports **114 tests, 97 passed, 0 failed** (the rest skip for want of a checkpoint); with one loaded the gated inference tests run too — `ModelTest.*` on a 1.5B `q2_k` passes 14/14 and the four `FullQuestions` cases pass on the 9B `q4_k_m`. The sampled full-question cases (temperature 0.7, `TOP_K`/`TOP_P` set) remain the flaky part: on the 27B EXL3 `1.60bpw` reference two of the four stop right after the opening code fence (`"```cpp"`, `"```python"`) because that checkpoint emits EOG there under sampling, while the same prompts answer correctly under greedy decoding — a checkpoint/sampler interaction, not a forward-path defect. All deterministic parity tests (GPU-vs-CPU logits, batch-vs-sequential prefill, sequential-decode argmax) pass bit-exactly with the split-KV and fp16-KV paths both on and off.

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

## License

MIT License — see [LICENSE](LICENSE).
