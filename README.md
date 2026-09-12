# ⚡ TinyCoder Inference Engine

**High-performant local inference engine** for the [TinyCoder AI](https://github.com/mgorshkov/tinycoder) VS Code extension. This repository contains **only the engine**: the C++/CUDA core (`src/cpp/core`), public headers (`include/`), unit tests (`unit_tests/`), benchmarks (`benchmarks/`), and build plumbing.

The engine runs **Qwen2.5-Coder**, **Gemma 4**, and **Qwen3.6/Qwen35MoE** model families with **mixed quantization** (IQ3_XXS, IQ3_S, IQ3_XS, IQ2_S, IQ2_M, Q2_K, Q4_K, Q4_K_XL, Q4_K_M, Q5_K, Q5_1, Q6_K, F32), **AMX/AVX-512/AVX2 CPU acceleration** with runtime SIMD dispatch, and a **CUDA GPU offload engine**. It is designed to run efficiently on limited hardware.

> The N-API bridge (`src/cpp/bridge`), TypeScript layer, and the VS Code panel live in the [extension repository](https://github.com/mgorshkov/tinycoder), which consumes this project's `tinycoder_core` static library via `FetchContent`.

## Key Features

### 🧠 Supported Models

| Model Family | Sizes | Architecture |
|---|---|---|
| **Qwen2.5-Coder** | 0.5B, 1.5B, 7B | RoPE, GQA, SwiGLU, RMSNorm |
| **Gemma 4** | 26B (A4B MoE), coding variants | MoE, RoPE, GQA, GeGLU, RMSNorm |
| **Qwen3.6** | 35B (A3B MoE, incl. SSM layers) | MoE, SSM, RoPE, GQA, SwiGLU, RMSNorm |
| **Qwen3.8 (qwen35)** | 27B dense hybrid | 48× Gated-Delta-Net + 16× full-attention, MRoPE, GQA, SwiGLU, MTP |

- Chat templates: `<|im_start|>system/user/assistant<|im_end|>` (Qwen), `<start_of_turn>user/model<end_of_turn>` (Gemma)
- **Weight tying** — LM head shares the token embedding matrix when possible
- Architecture detection via GGUF `general.architecture` ([`ModelConfig.hpp`](include/ModelConfig.hpp:37))

### 📦 Mixed Quantization (Multiple GGML Types)

Unlike many inference engines that use a single quantization type, TinyCoder supports **mixed-quantization GGUF files** where different tensors use different quantization formats:

| Type | Bits/Weight | Block Size | Block Bytes | Used For |
|------|-------------|------------|-------------|----------|
| **IQ3_XXS** | 3.0625 | 256 | 98 | Primary weight quantization |
| **IQ3_S** | 3.4375 | 256 | 110 | Higher-precision layers |
| **IQ3_XS** | 3.3125 | 256 | 106 | Balanced MoE compression |
| **IQ2_S** | 2.5625 | 256 | 82 | Aggressive compression |
| **IQ2_M** | 2.6875 | 256 | 86 | Moderate aggressive compression |
| **IQ4_NL** | 4.5 | 32 | 18 | Qwen3.8 attn_qkv/attn_gate/ssm_out (non-linear grid) |
| **IQ4_XS** | 4.25 | 256 | 136 | Qwen3.8 ffn_gate/ffn_down |
| **Q2_K** | 2.5625 | 256 | 82 | Small model quantization |
| **Q4_K** | 4.0625 | 256 | 144 | Balanced precision |
| **Q4_K_M** | 4.5 | 256 | 144 | Medium-balanced precision |
| **Q4_K_XL** | 4.5 | 256 | 160 | Extended balanced precision |
| **Q5_K** | 5.0625 | 256 | 176 | Embeddings, high-precision layers |
| **Q5_1** | 5.0625 | 32 | 32 | Legacy format support |
| **Q6_K** | 6.5625 | 256 | 210 | High-precision layers |
| **F32** | 32 | 1 | 4 | Norms, biases |

All weights are stored in their **native quantized format** in memory and dequantized **on-the-fly** during matrix-vector multiplication. This keeps memory usage close to the compressed file size (~637 MB for 1.5B) rather than the F32 dequantized size (~5.2 GB).

### ⚡ SIMD CPU Acceleration

- **Runtime SIMD dispatch** via `np::internal::max_simd_level()` — auto-detects the best available instruction set:
  - **AMX** — Intel Sapphire Rapids+ (tile matrix multiply, 8-bit)
  - **AVX-512** — Intel Skylake-X / Ice Lake+ (16-wide FMA)
  - **AVX2** — Intel Haswell+ / AMD Zen+ (8-wide FMA)
  - **Scalar** — Universal fallback
- **OpenMP** thread parallelism for all matrix operations
- Per-file SIMD compile flags: only [`SIMDMatMulVecAVX2.cpp`](src/cpp/core/SIMDMatMulVecAVX2.cpp) is compiled with `-mavx2 -mfma -mf16c`, [`SIMDMatMulVecAVX512.cpp`](src/cpp/core/SIMDMatMulVecAVX512.cpp) with `-mavx512f -mavx512bw -mfma` — no AVX code is generated in other translation units
- Fused dequantize-dot-product kernels (K-quants) avoid float temporaries entirely
- **Pre-packed Q2_K** data for `ffnGate`/`ffnUp` eliminates 2-bit extraction in the SIMD kernel (see [`Model.hpp`](include/Model.hpp:55))
- **FP16 weight copies** of `attnO`/`ffnDown` halve LM-head/FFN memory bandwidth; **Q8_K prepacked copies** power the prefill batch GEMM (`_mm256_maddubs_epi16` int8 kernels)
- **Batch SIMD kernels** for qwen35's unrolled prefill path over all 64 layers: register-tiled Q6_K/Q4_K/Q3_K/Q8_K compacts + IQ4_XS/IQ4_NL/Q5_K AVX2 kernels pumped the 27B dense forward from **135 s/token → 6.9 s/token (~19.6×)** on an 8-core CPU (see [`QuantizedMatrix.cpp`](src/cpp/core/QuantizedMatrix.cpp:93))
- Runtime `TINYCODER_FORCE_SCALAR=1` A/B switches between the SIMD batch kernels and the scalar double-precision baseline (for correctness differentials)

### 🎮 CUDA GPU Offload Engine

- Full-model GPU offload ([`GPUCompute.cu`](src/cpp/core/GPUCompute.cu), [`ModelGPU.cpp`](src/cpp/core/ModelGPU.cpp)):
  - **Prefill** — cuBLAS fp16 tensor-core GEMM for all projections
  - **Decode** — quantized on-the-fly-dequant GEMV kernels reading raw GGUF blocks from VRAM
  - **Flash attention** — `kWarpAttention<HD>` with compile-time register accumulators, exactly `nHeads` warps
- **Enabled by default** in builds compiled with `ENABLE_CUDA=ON`; all layers offloaded; automatic fallback to CPU when CUDA is unavailable or at the first forward error
- **Hybrid partial offload for qwen35** (large dense models that exceed VRAM): the engine keeps the whole model in RAM and mirrors only the first `numGpuLayers` layers' weights to the GPU, running those layers on the GPU and continuing the rest on the CPU in the same forward pass. See [GPU Partial-Offload Scheme](#gpu-partial-offload-scheme-qwen35).
- Runtime controls: `TINYCODER_GPU=0` forces the CPU path, `TINYCODER_NGL` limits offload to N layers (unset → automatic VRAM fit), `TINYCODER_GPU_VERBOSE=1` prints per-stage CUDA timings

### 🧵 Tuned Threading & Scheduling

- Dedicated [`ThreadPool`](include/ThreadPool.hpp) with work-stealing (`parallelForSteal2`) for the fused FFN (llama.cpp `ggml_compute_forward_mul_mat_one_chunk` style)
- Cooperative last-arriver act→Q8_K quantize inside the fused FFN phase 1
- CPU affinity pinning of workers to **distinct logical CPUs** (default ON, runtime `TINYCODER_AFFINITY=0/1`)
- Steady-state generation performs **zero heap allocations per token** (per-thread `ScratchPool` reuse)

## Project Structure

```
tinycoder-inference/
├── include/                        # Public engine headers
│   ├── Model.hpp                   # Transformer model (forward, generate, KV cache)
│   ├── ModelConfig.hpp             # Model & inference configuration
│   ├── GGUFLoader.hpp              # GGUF v3 file format loader
│   ├── GGMLDequantize.hpp          # Multi-type dequantization (Q5_K, IQ3_XXS, …)
│   ├── IQ3XXS.hpp                  # Legacy IQ3_XXS block-level dequantization
│   ├── LMHead.hpp                  # LM head computation (CPU OpenMP path)
│   ├── LMHeadCUDA.hpp              # LM head CUDA (cublasSgemv) interface
│   ├── GPUCompute.hpp              # CUDA GPU offload engine interface
│   ├── SIMDMatMulVec.hpp           # SIMD-accelerated dot product & accumulate
│   ├── AlignedVector.hpp           # 64-byte aligned vector
│   ├── ChatTemplateRenderer.hpp    # Chat template formatting
│   ├── MemHints.hpp                # Memory-hint helpers
│   ├── ThreadPool.hpp              # Thread pool + work-stealing schedulers
│   └── Tokenizer.hpp               # BPE tokenizer (Qwen2.5 / Gemma / Qwen35MoE)
├── src/cpp/core/                   # Engine core (compiled once into tinycoder_core)
│   ├── Model.cpp                   # Transformer model (forward, generate)
│   ├── ModelForward.cpp            # Forward pass implementation
│   ├── ModelGeneration.cpp         # Generation loop & sampling orchestration
│   ├── ModelSampling.cpp           # Token sampling (AVX2-optimized)
│   ├── ModelLoad.cpp               # Model loading & weight prep
│   ├── ModelMoE.cpp                # Mixture-of-Experts layers
│   ├── ModelGPU.cpp                # GPU offload adapter
│   ├── GPUCompute.cu               # CUDA kernels (GEMV, flash attention)
│   ├── LMHeadCUDA.cu               # CUDA LM head (cublasSgemv)
│   ├── GGUFLoader.cpp              # GGUF v3 reader (metadata + tensor data)
│   ├── GridTables.cpp              # IQ2_S grid lookup table (1024 entries)
│   ├── GridTablesIQ3S.cpp          # IQ3_S grid lookup table (512 entries)
│   ├── QuantizedMatrix.cpp         # Quantized matrix-vector multiply (CPU)
│   ├── QuantizedEmbedding.cpp      # Quantized token embedding dequantization
│   ├── SIMDMatMulVec.cpp           # SIMD dispatch (AVX2/AVX-512/scalar)
│   ├── SIMDMatMulVecAVX2.cpp       # AVX2 SIMD kernels
│   ├── SIMDMatMulVecAVX512.cpp     # AVX-512 SIMD kernels
│   ├── ThreadPool.cpp              # Thread pool implementation
│   ├── Tokenizer.cpp               # BPE tokenizer (GGUF embedded + file loading)
│   └── …                           # (ModelDebug / ModelPrimitives / …)
├── unit_tests/                     # Google Test suite + shared test env
├── benchmarks/                     # llama-bench-parity generation benchmark
├── scripts/
│   ├── build.sh                    # One-shot build + test script
│   └── perf-threading.sh           # Thread-count + perf-counter harness
├── plans/                          # Optimization plans/documents
├── CMakeLists.txt                  # CMake build (FetchContent np + cmake-common)
├── BENCHMARK_REPORT.md             # Detailed TinyCoder vs llama.cpp benchmark report
└── LICENSE                         # MIT License
```

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
- **CUDA Toolkit** ≥ 11.0 + nvcc (for GPU support)

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

TinyCoder supports GGUF v3 files from multiple model families:

| Model | Quantization | Size (approx) | Notes |
|-------|-------------|---------------|-------|
| `Qwen2.5-Coder-0.5B` | Q4_K_M | ~350 MB | Lightweight coding |
| `Qwen2.5-Coder-1.5B` | Q2_K | ~550 MB | Aggressive compression |
| `Qwen2.5-Coder-1.5B` | IQ3_XXS (imat) | ~700 MB | Recommended balanced |
| `Qwen2.5-Coder-7B` | IQ2_S | ~2.0 GB | Ultra-compact 7B |
| `Qwen2.5-Coder-7B` | IQ3_XXS (imat) | ~3.2 GB | Recommended 7B |
| `Gemma 4 Coding` | Q2_K | ~8.5 GB | Lightweight MoE coding |
| `Gemma 4 Coding` | Q4_K_M | ~13 GB | Balanced MoE coding |
| `Gemma 4 Coding` | Q6_K | ~19 GB | High-precision MoE coding |
| `Gemma 4 (26B-A4B)` | Q4_K_XL (qat-UD) | ~14 GB | Instruction-tuned MoE |
| `Qwen3.6 (35B-A3B)` | IQ3_XS (distilled) | ~12 GB | Reasoning-distilled MoE |
| `Qwen3.6 (35B-A3B)` | IQ2_M (UD) | ~8.5 GB | Ultra-compact MoE |

Download the `.gguf` file and point the engine at it via `TINYCODER_MODEL_PATH` or the `Model::load()` API.

## Usage (C++ API)

```cpp
#include "Model.hpp"
#include "ModelConfig.hpp"

using namespace tinycoder;

// 1. Create and load the model (GGUF v3)
Model model;
std::string err;
if (!model.load("/path/to/qwen2.5-coder-1.5b-instruct.gguf", &err)) {
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

| Variable | Description |
|----------|-------------|
| `TINYCODER_MODEL_PATH` | Default GGUF path used by tests/benchmarks |
| `TINYCODER_THREADS` | Thread-pool size override (default: logical CPU count) |
| `TINYCODER_AFFINITY` | `0`/`1` — override CPU-affinity pinning at runtime |
| `TINYCODER_GPU` | `0` forces the CPU path in CUDA builds |
| `TINYCODER_NGL` | Limits GPU offload to N layers (llama.cpp `-ngl` style) |
| `TINYCODER_GPU_VERBOSE` | `1` prints per-stage CUDA timings |
| `TINYCODER_PROFILE` | `1` enables per-stage profiling output |

## Testing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The Google Test suite (`tinycoder_test`) covers dequantization correctness across all quantization types, reference parity against sequential token-by-token forward passes, full-question generation tests, thread-pool reentrancy, and GPU/CPU comparison (49/49 tests pass, bit-exact parity on the reference build).

## Benchmarking

A llama-bench-parity harness (`tinycoder_bench`) reproduces `llama-bench`'s `pp64`/`tg64` protocol: fixed-token prompt, greedy decode, decode-loop-only timing, warmup + warm repeats (mean ± stdev).

```bash
# CPU-only build
cmake --build build --target tinycoder_bench -j
TINYCODER_MODEL_PATH=/path/to/model.gguf \
  ./build/benchmarks/tinycoder_bench --n-prompts 64 --n-gen 64 --reps 5

# CUDA build (GPU offload engine, default ON in CUDA builds)
TINYCODER_MODEL_PATH=/path/to/model.gguf \
  ./build-cuda/benchmarks/tinycoder_bench --gpu --reps 5 --n-gen 64
```

See [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) for the full TinyCoder vs llama.cpp comparison, per-stage generation profile, and the optimization history.

### Performance Highlights

Measured on i7-4790K (AVX2) / RTX 2080 Ti with `qwen2.5-coder-1.5b-instruct-q2_k.gguf` (see [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) for methodology):

| Path | Prefill | Generation |
|------|---------|-----------|
| **CPU (8 threads, AVX2)** | ~85.8 tok/s | ~26.3–26.6 tok/s (~38 ms/token) |
| **CPU vs llama.cpp (8t)** | ~1.2× ahead | within ~0.4–0.6 tok/s (par) |
| **CUDA (RTX 2080 Ti) pp64** | **4,773 tok/s** (13.4 ms) | — |
| **CUDA (RTX 2080 Ti) tg64** | — | **128.9 tok/s** |

Generation is memory-bound (weight streaming at ~17–19 GB/s effective on DDR3-1600); the fused gate+up+down FFN (~56% of token time) and the LM head (~28%) dominate. MoE models show lower tok/s due to expert routing overhead.

## Technical Details

### Prefill vs. Generation

Like most decoder-only transformers, inference is split into two phases orchestrated by [`Model::generate()`](src/cpp/core/ModelGeneration.cpp):

1. **Prefill** — the whole prompt is processed at once with batched GEMM kernels (weight matrix read once instead of `seqLen` times); causal masking enforces attention locality; the LM head is computed only for the last token; the KV cache is populated for every prompt position.
2. **Generation** — token-by-token decode with `seqLen = 1`: fused per-token GEMV kernels (`matMulVecFusedQKV`, `matMulVecFusedGateUp`, `deqMatMulVecF16`), KV-cache reuse so per-token work stays constant, and the full LM head computed every step (the largest single memory read per token).

### Fused Dequantize-Dot Product

Instead of dequantizing entire matrices to F32, [`QuantizedMatrix::matMulVec()`](src/cpp/core/QuantizedMatrix.cpp) uses block-level fused strategies:

- **Dequantize-then-dot** (legacy, non-K-quant types): one quantized block → small stack buffer → SIMD dot product
- **Fully fused dot product** (K-quants): `dot += dl × sum_xq - ml × sum_x` per group of 16 weights — no float temporary at all
- **Pre-packed Q2_K** for `ffnGate`/`ffnUp` (97.7% of Q2_K dot-product calls): 2-bit values expanded to bytes at load time, single `_mm_loadu_si128` per group at runtime

### SIMD Runtime Dispatch

[`SIMDMatMulVec`](src/cpp/core/SIMDMatMulVec.cpp) provides five key operations with runtime dispatch (`SCALAR < SSE2 < SSE3 < AVX < AVX2 < AVX512 < AMX`), initialized once via `np::internal::max_simd_level()`:

- `dotProductFMA()` — dot product of float vectors
- `dotProductFMA_F16()` — dot product with FP16-stored weights (`_mm256_cvtph_ps`)
- `accumulateFMA()` — fused multiply-add accumulation
- `dotProductQ2_K_SIMD()` — fused Q2_K dequantize-dot for native blocks
- `dotProductQ2_K_PrePacked_SIMD()` — fused Q2_K dot for pre-packed blocks

### Mixed Quantization Dequantization

[`GGMLDequantize`](include/GGMLDequantize.hpp) provides block-level dequantization and fused dot products for all 12 supported quantization formats, plus `dequantizeToF16()` (one block at a time into a stack buffer, converted inline to FP16 — no large intermediate F32 allocation) and `prepackQ2_K()`.

### GGUF Format Support

- **Version**: GGUF v3
- **Tensor types**: F32, Q5_K, Q5_1, Q4_K, Q4_K_XL, Q4_K_M, Q2_K, Q6_K, IQ3_XXS, IQ3_S, IQ3_XS, IQ2_S, IQ2_M, Q8_0, Q8_1, Q8_K, IQ4_NL, IQ4_XS (dequantized on-the-fly)
- **Metadata**: Architecture, layer count, head count, RoPE config, chat template, `rope.dimension_sections` (INT32/UINT32 arrays, e.g. qwen35 `[11,11,10,0]`)
- **Tokenizer**: Embedded BPE vocab + merges (tiktoken-compatible)

### Qwen3.8 (qwen35) Hybrid Architecture

`Qwen3.8-27B-UD-Q4_K_M.gguf` is a **dense hybrid** model: 48 gated-delta-net (GDN) recurrent layers interleaved with 16 full-attention layers (layer *i* is full-attention when `(i+1) % 4 == 0`), plus one MTP (multi-token-prediction) block that is *not* executed in the main decode pass. The full forward lives in [`ModelQwen35.cpp`](src/cpp/core/ModelQwen35.cpp):

- **Recurrent (GDN) layer** — `attn_qkv` (Q/K/V), `attn_gate` (z), `ssm_beta`, `ssm_alpha` (softplus + `ssm_dt.bias` → decay `exp(gate·a)`), conv1d (kernel 4, silu), per-head L2-norm q/k, then the gated-delta-net recurrence over 48 value heads with a persistent per-layer state matrix and gated RMSNorm (`ssm_norm` × `silu(z)`), projected by `ssm_out`.
- **Full-attention layer** — fused Q+gate projection (`attn_q` outputs `[nHeads·2·headDim]`; Q is the first `headDim` of each head, the gate is the second), per-head Q/K RMSNorm (`attn_q_norm`/`attn_k_norm`), **MRoPE** (`rope.dimension_count`=64, sections `[11,11,10,0]`, theta=1e7), GQA attention over 4 KV heads with `attn_factor` = 1/√headDim, elementwise `sigmoid(gate)`, `attn_output` projection.
- **MTP block** — loaded but dispatched via the separate MTP draft graph only (llama.cpp parity).

**MRoPE gotcha (two layers deep):** `ggml_mrope_cache_init()` seeds its theta ramp with the **token position**, so the angular frequency of pair *k* is `angle = p · freq_base^(-2k/n_dims)` — `freq_base` (1e7) enters **only** through the inter-pair scaling, never as a multiplier of the position. Seeding the cache with `freq_base` itself (matching the standard RoPE table convention) injected an extra factor of 1e7 into every angle and scrambled every full-attention layer — the debug per-layer bisection caught it: recurrent layers 0-2 matched llama exactly while layer 3 (the first full-attention layer) diverged first. The MRoPE cache is computed inline in [`applyMRoPE()`](src/cpp/core/ModelQwen35.cpp:115).

**Correctness vs llama.cpp (AR decode of the 12-token reference prompt, same `argmax` family):**

| Signal | TinyCoder | llama.cpp (AR) | Match |
|---|---|---|---|
| 12 token embeddings (fnv fingerprints) | 12/12 | — | ✅ bit-exact |
| Final post-norm hidden norm | 125.87 | 125.29 | 0.47% |
| Final hidden first8 max diff | 0.103 | — | < 0.5 ✅ |
| Layer 3 (first full-attn) norm | 23.70 | 23.68 | 0.999× |
| Layer 63 norm | 650.2 | 651.4 | ~0.2% |
| Logits argmax (SIMD & scalar) | **248068 " thinking"** | 248068 | ✅ |
| Logits top 2/3 | 760 "The", 57590 "Paris" | same | ✅ |

The residual ~0.5% is Q8_K activation-quantization noise from the SIMD batch kernels (verified: `TINYCODER_FORCE_SCALAR=1` scalar baseline stays within it, and SIMD/scalar disagree by < 3% of the hidden norm through all 64 layers).

### GPU Partial-Offload Scheme (qwen35)

For dense models larger than VRAM (e.g. `Qwen3.6-27B` = 19.8 GB Q5_K_M on an 11 GB RTX 2080 Ti), the engine uses a **static residency split**: the *entire* model stays loaded in RAM, and only the first `numGpuLayers` transformer blocks are mirrored to GPU VRAM. No weights are transferred at runtime — the split is decided once at upload time.

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

Instead of guessing `-ngl`, the engine measures real device residency (K-quant block bytes + F32 matrices + KV slots + conv/GDN state per layer, embedding on top) and fills GPU VRAM from `cudaMemGetInfo` free bytes minus a fixed 900 MB scratch/GEMM-twin margin (llama.cpp `-ngl auto` style, gated to qwen35 — the only arch with the CPU continuation). On the 2080 Ti this lands at **30/65 layers**.

#### Measured (Qwen3.6-27B-Q5_K_M, RTX 2080 Ti 11 GB)

| Config | GPU layers | Decode | Single-token forward | Parity |
|---|---|---|---|---|
| CPU only | 0 | ~0.045 tok/s | 116 s | reference |
| `TINYCODER_NGL=25` | 25 | 0.0675 tok/s | 46 s | top-1 id=353 ✓ |
| **auto-fit** | **30** | **0.0817 tok/s** | 46 s | top-1 id=353 ✓ |

#### Measured (Qwen3.6-27B-UD-Q4_K_XL, RTX 2080 Ti 11 GB)

Same scheme, smaller K-quants → more layers land on device (the `Q4_K` embedding now has a dedicated `kEmbedDequantQ4K` kernel). Batch/seq coherence and keyword answers all pass:

| Config | GPU layers | Prefill | Decode | Sample answers |
|---|---|---|---|---|
| **auto-fit** | **37** | 0.135‑0.199 tok/s | **0.153‑0.167 tok/s** | Paris / pointer / C++ / Python ✓ |

Decode is ~2× the Q5_K_M auto-fit rate (0.0817 tok/s) from the combination of more resident layers (37 vs 30) and the cheaper 4-bit weights.

#### Weight compression for the GPU prefix

The GPU kernels support the full K-quant family (`Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K`), so a lower-bit model file shrinks the resident layers with zero code:

| Quant | per-layer GPU weights | 30-layer VRAM | auto-fit GPU layers |
|---|---|---|---|
| Q5_K_M (default) | ≈272 MB | 8.16 GB | 30 |
| Q4_K (e.g. `Qwen3.6-27B-UD-Q4_K_XL`, 17.9 GB) | ≈225 MB | 6.75 GB | **37** ✓ |
| Q3_K | ≈175 MB | 5.25 GB | ~43 |
| Q2_K | ≈133 MB | 3.99 GB | ~50 |

The prefill path's fp16 working form is transient (one matrix in `wF16_` at a time); decode uses quantized GEMV directly, so the only real reduction lever is the file's quant.

### Qwen3.6-35B-A3B (qwen35moe) GPU — Full Offload + CPU-Expert Hybrid

`Qwen3.6-35B-A3B` (`ARCH_QWEN35MOE`) is a **MoE hybrid**: 30 gated-delta-net (GDN) recurrent layers + 10 full-attention layers (40 total), each with a **softmax-routed MoE FFN** — a shared `ffn_gate_inp` router over **256 experts** (top-8 per token, weights re-normalized with the `norm_w=1e-4` clamp), a shared sigmoid-gated expert, and per-expert SwiGLU (`ffn_gate_exps`/`ffn_up_exps` [256·512] → `ffn_down_exps` [256·hidden]). The CPU reference lives in [`computeQwen35MoE()`](src/cpp/core/ModelMoE.cpp:161); the GPU driver is [`forwardQwen35MoePrefix()`](src/cpp/core/GPUCompute.cu:3827).

The GPU engine requires **full offload** (all 40 layers; partial offload has no CPU continuation for the MoE arch, so `TINYCODER_NGL` is ignored). On the 11 GB RTX 2080 Ti:

- **IQ1_M (10.05 GB)** — full offload: every tensor (embedding, GDN/attention, all 256-expert matrices, router, shared expert, LM head) is uploaded; the GPU runs the entire decode.
- **IQ2_M (11.5 GB) / Q4_K_M (22.1 GB)** — **CPU-expert hybrid** (llama.cpp `--cpu-moe` style, automatic): the upload first tries the full-GPU layout; when it OOMs, the adapter retries ONCE with only the **256-expert matrices staying in host RAM** ([`ModelGPUAdapter`](src/cpp/core/ModelGPU.cpp:166) → `GPUModel::setMoeCpuFn`). Per layer the GPU still runs embedding, the GDN/attention block, the post-attention RMSNorm, the KV cache, the LM head, and the residual; it copies only the post-attention norm to CPU, computes the routed MoE FFN (softmax over 256 → top-8 → per-expert SwiGLU → shared expert) with the **exact reference math** [`computeQwen35MoE()`](src/cpp/core/ModelMoE.cpp:161), copies the result back and residual-adds on-device. Only 8/256 = 3.1% of the expert weights are touched per token, so the ~32B-param expert block never needs to fit VRAM.

A failed upload releases every partially allocated device buffer (`GPUModel::upload` cleanup), consumes the sticky CUDA error, and the adapter latches the failure — no repeated doomed attempts and no VRAM leak.

| Quant | File size | GPU mode on 11 GB | Verified |
|---|---|---|---|
| `UD-IQ1_M` | 10.05 GB | ✅ full offload | single-token + batch/seq coherence ✓ |
| `UD-IQ2_M` | 11.5 GB | ✅ hybrid (experts on CPU) | SingleTokenForward / MultiTokenForward / batch-seq coherence / `SequentialDecodeArgmaxAgrees` ✓ |
| `UD-Q4_K_M` | 22.1 GB | ✅ hybrid (experts on CPU) | SingleTokenForward ✓ |

**Measured (Qwen3.6-35B-A3B-UD-IQ1_M, full offload, RTX 2080 Ti 11 GB):**

| Path | Runtime | Parity |
|---|---|---|
| Single-token forward, GPU | **2.5 s** | top-1 id=25 ":" == CPU |
| Single-token forward, CPU | ~11 s | reference |
| Batch prefill + sequential decode, GPU | **~10 s** | batch/seq top-1 id=198 ✅ |
| Batch prefill + sequential decode, CPU-fallback | ~149 s | reference |

Decode is **~4.4×**, prefill+batch **~15×** faster than the CPU path. GPU top-10 logits track the CPU reference within quantization noise at every sequential-decoding step (e.g. step 7 `>` ), confirming the router + shared-expert math.

The MoE FFN adds the router GEMV (F32 `ffn_gate_inp` [256×hidden]), the top-k router kernel (softmax over 256 → serial top-8 insert with strict `>` tie-break), per-expert row-sliced GEMVs (batched over tokens via the fp16 GEMM path in prefill, warp-per-row in decode), weighted accumulation (`kQ35MoeAccumScaled`/`kAddScaled1`), and the shared-expert sigmoid-gated SwiGLU.

### KV Cache

The KV cache stores key/value tensors as F32 arrays with shape `[numLayers, maxSeqLen, numKVHeads, headDim]`. For the 1.5B model with 2048 context: 28 layers × 2048 × 2 KV heads × 128 dim × 4 bytes × 2 (K+V) ≈ **112 MB**.

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- **[np library](https://github.com/mgorshkov/np)** — NumPy-style arrays with SIMD/CUDA acceleration
- **llama.cpp** — reference implementation used for correctness parity and performance benchmarking
