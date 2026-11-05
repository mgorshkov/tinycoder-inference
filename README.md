# ⚡ TinyCoder Inference Engine for NVIDIA GPUs (Turing / Ampere / Ada)

**High-performant local inference engine** for the [TinyCoder AI](https://github.com/mgorshkov/tinycoder) VS Code extension and the [TinyCoder OpenAI](https://github.com/mgorshkov/tinycoder-openai) project, which exposes the same engine through an OpenAI-compatible API. This repository contains **only the engine**: the C++/CUDA core (`src/cpp/core`), public headers (`include/`), unit tests (`unit_tests/`), benchmarks (`benchmarks/`), and build plumbing.

The engine runs the **Qwen3.8** (qwen35 dense), **Ornith** (qwen35moe), **Gemma 4**, and Prism **Ternary-Bonsai** model families from **GGUF** or **EXL3** (`trellis` / `mul1`) checkpoints — e.g. `Qwen3.5-27B` at 1.60 bpw — with **mixed quantization**, **AMX/AVX-512/AVX2 CPU acceleration** (runtime SIMD dispatch), and a **CUDA GPU offload engine**. It is designed to run efficiently on limited hardware.

> **Written from scratch — no inference libraries or engines.** Every part of TinyCoder is original code in this repository: the C++/CUDA kernels, GEMV/GEMM and MMA schedules, GGUF and EXL3 loaders, quantizers/dequantizers, tokenizer, scheduler, KV cache, and CPU SIMD paths. It does **not** link, embed, vendor, or call llama.cpp, ggml, Candle, PyTorch, ExLlamaV3, or any other inference runtime; there is **no inference-library dependency** at build or run time. The only third-party code is non-inference plumbing: the [np](https://github.com/mgorshkov/np) array utility, a shared CMake config, and GoogleTest for unit tests (the CUDA Toolkit/cuBLAS is the system GPU runtime). All performance work is TinyCoder's own kernels.

> **GPU support.** The CUDA engine runs natively on **Turing, Ampere and Ada GPUs — including the RTX 2080 Ti, RTX 3090, RTX 4090 and RTX 5090** — plus newer architectures via forward-compatible PTX JIT, with runtime capability dispatch and a CPU fallback. **No GPU emulator is used or needed.** See [Supported GPUs](#supported-gpus-no-emulator-required).

## Benchmarking: GPU Performance (TinyCoder vs llama.cpp)

### GPU Comparison (TinyCoder vs llama.cpp / exllamav3)

Columns follow the benchmark defaults: `tinycoder_bench --n-prompts 512 --n-gen 128 --threads 36` (**pp512 / tg128**). GGUF baselines are `llama-bench -p 512 -n 128 -t 36`; the EXL3 baseline is a local exllamav3 1.6.6 run measured with the same pp512/tg128 protocol — see [How the exllamav3 EXL3 baseline was measured](#how-the-exllamav3-exl3-baseline-was-measured). Warm mean-of-reps, RTX 2080 Ti. Bold marks the faster engine in a column.

| Model | File size | GPU mode TinyCoder | **TinyCoder pp512** | **TinyCoder tg128** | Baseline engine | **Baseline pp512** | **Baseline tg128** |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B `q2_k` | 0.7 GB | full offload | **10,151.4 tok/s** | **317.9 tok/s** | llama.cpp | 7,523 tok/s | 285.7 tok/s |
| Qwen2.5-Coder-1.5B `iq3_xxs` | 0.6 GB | full offload | **10,231.3 tok/s** | **247.9 tok/s** | llama.cpp | 10,120 tok/s | 244.7 tok/s |
| Qwen2.5-Coder-7B `iq2_s` | 2.4 GB | full offload | **2,707.0 tok/s** | **95.3 tok/s** | llama.cpp | 2,447 tok/s | 90.7 tok/s |
| Qwen2.5-Coder-7B `iq3_xxs` | 2.9 GB | full offload | 2,741.6 tok/s | **101.5 tok/s** | llama.cpp | **2,872 tok/s** | 95.1 tok/s |
| Ternary-Bonsai-2-27B `pq2_0` (Prism) | 6.7 GB | full offload | 16.90 tok/s | 15.42 tok/s | llama.cpp | — | — |
| Ornith-1.5-9B `q4_k_m` | 5.9 GB | full offload | 2,483.4 tok/s | **88.7 tok/s** | llama.cpp | **2,606 tok/s** | 83.8 tok/s |
| Ornith-1.5-9B `q8_0` | 8.9 GB | full offload | 2,843.4 tok/s | 57.36 tok/s | llama.cpp | **2,902 tok/s** | **58.4 tok/s** |
| Ornith-1.5-35B `q4_k_m` | 20.2 GB | hybrid (grouped int8) | 217.3 tok/s | 23.39 tok/s | llama.cpp `-ncmoe 99` | **233.8 tok/s** | **25.8 tok/s** |
| Ornith-1.5-35B-A3B `q8_0` → `q4_k` experts | 34.4 GB | hybrid (layer arena, grouped int8) | **175.8 tok/s** | **53.90 tok/s** | llama.cpp `-ncmoe 99` | 136.8 tok/s | 18.8 tok/s |
| Qwen3.8-27B EXL3 `1.60bpw` | 8.6 GB | full offload | 672.7 tok/s | 17.49 tok/s | exllamav3 1.6.6 | **707.5 tok/s** | **22.9 tok/s** |

### How the exllamav3 EXL3 baseline was measured

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


## Key Features

### 🧠 Supported Models

TinyCoder loads two checkpoint formats:

- **[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)** — the llama.cpp/GGML container. Quantized K/IQ blocks, F32/F16 tensors, tokenizer and architecture metadata are read directly; `general.architecture` selects the model family.
- **[EXL3](https://github.com/turboderp-org/exllamav3)** — the ExLlamaV3 `trellis`/`mul1` checkpoint format (HF `safetensors` + `config.json` + `tokenizer.json`).

Point the engine at a `.gguf` file or an EXL3 checkpoint directory; [`Model::load`](src/cpp/core/ModelLoad.cpp) detects the format automatically and both use the same CUDA/CPU kernels, KV cache, and offload path.

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
- **Hybrid partial offload for qwen35** (large dense models that exceed VRAM): the engine keeps the whole model in RAM and mirrors only the first `numGpuLayers` layers' weights to the GPU, running those layers on the GPU and continuing the rest on the CPU in the same forward pass. See GPU Partial-Offload Scheme.
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

**No inference-library dependency.** The engine, kernels, loaders, and tokenizers are written from scratch in this repository. The only external packages are non-inference plumbing:

- **[np library](https://github.com/mgorshkov/np)** — NumPy-style arrays with SIMD/CUDA acceleration (fetched via `FetchContent`)
- **[shared cmake config](https://github.com/mgorshkov/cmake)** — compiler checks, CUDA setup, OpenMP (fetched via `FetchContent`)
- **Google Test** — unit tests only (fetched via `FetchContent`)

All are fetched automatically; no system packages beyond a compiler, CMake, and (optionally) the CUDA Toolkit are required. No llama.cpp, ggml, Candle, PyTorch, or ExLlamaV3 code is used anywhere in the build.

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
| `Ornith-1.5-35B-A3B` | Q8_0 (experts → Q4_K) | ~34.4 GB | hybrid (layer arena) | qwen35moe, 256 experts (top-8) |
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

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- **[np library](https://github.com/mgorshkov/np)** — NumPy-style arrays with SIMD/CUDA acceleration
- **llama.cpp** — reference implementation used for correctness parity and performance benchmarking
