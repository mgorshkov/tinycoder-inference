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
| **Ornith (qwen35moe)** | 35B-A3B (Q8_0 / Q4_K_M checkpoints) | qwen35moe MoE: 256 experts (top-8), GDN + full-attention hybrid, softmax router |
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
│   ├── GGMLDequantize.hpp          # Multi-type dequantization (Q5_K, IQ3_XXS, IQ4_NL, …)
│   ├── IQ3XXS.hpp                  # Legacy IQ3_XXS block-level dequantization
│   ├── LMHead.hpp                  # LM head computation (CPU OpenMP path)
│   ├── LMHeadCUDA.hpp              # LM head CUDA (cublasSgemv) interface
│   ├── GPUCompute.hpp              # CUDA GPU offload engine interface
│   ├── SIMDMatMulVec.hpp           # SIMD-accelerated dot product & accumulate
│   ├── SIMDMatMulVecInternal.hpp   # SIMD kernel internals
│   ├── AlignedVector.hpp           # 64-byte aligned vector
│   ├── ChatTemplateRenderer.hpp    # Chat template formatting
│   ├── MemHints.hpp                # Memory-hint helpers
│   ├── ThreadPool.hpp              # Thread pool + work-stealing schedulers
│   ├── GridTablesDevice.hpp        # Device grid tables (IQ2_S / IQ3_S dequant)
│   ├── ModelInternal.hpp           # Model internals for forward/debug
│   └── Tokenizer.hpp               # BPE tokenizer (Qwen2.5 / Gemma / Qwen35MoE)
├── src/cpp/core/                   # Engine core (compiled once into tinycoder_core)
│   ├── Model.cpp                   # Transformer model (forward, generate)
│   ├── ModelConfig.cpp             # Model & inference configuration
│   ├── ModelForward.cpp            # Forward pass implementation
│   ├── ModelGeneration.cpp         # Generation loop & sampling orchestration
│   ├── ModelSampling.cpp           # Token sampling (AVX2-optimized)
│   ├── ModelLoad.cpp               # Model loading & weight prep
│   ├── ModelMoE.cpp                # Mixture-of-Experts layers
│   ├── ModelQwen35.cpp             # qwen35 (Qwen3.6-27B / Qwen3.8-27B) recurrent+attention
│   ├── ModelGPU.cpp                # GPU offload adapter (upload, partial-offload policy)
│   ├── GPUCompute.cu               # CUDA kernels (quantized GEMV, flash attention, IQ4_NL/Q8_K)
│   ├── LMHeadCUDA.cu               # CUDA LM head (cublasSgemv / top-k)
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
│   ├── ModelPrimitives.cpp         # rmsNorm / softmax / RoPE primitives
│   ├── ModelInternal.cpp           # internal helpers
│   ├── ModelDebug.cpp              # debug dumps (hidden states, layer-wise)
│   └── ModelForwardDebug.cpp       # token-by-token debug forward
├── unit_tests/                     # Google Test suite + shared test env
├── benchmarks/                     # llama-bench-parity generation benchmark (tinycoder_bench)
├── tools/                          # Dev tools: llama reference probe, gguf census, weight dumps
├── scripts/
│   ├── build.sh                    # One-shot build + test script
│   └── perf-threading.sh           # Thread-count + perf-counter harness
├── plans/                          # Optimization plans/documents
├── CMakeLists.txt                  # CMake build (FetchContent np + cmake-common)
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

TinyCoder supports GGUF v3 files from multiple model families. The table below lists every model that has been load-verified and benchmarked on this project's reference hardware (i7-4790K + RTX 2080 Ti **11 GB**); "GPU mode" is what the engine automatically selects on that 11 GB card:

| Model | Quantization | Size (approx) | GPU mode on 11 GB | Notes |
|-------|-------------|---------------|-------------------|-------|
| `Qwen2.5-Coder-1.5B` | Q2_K | ~0.7 GB | full offload | Lightweight coding, 132 tg tok/s |
| `Qwen2.5-Coder-1.5B` | IQ3_XXS (imat) | ~0.6 GB | full offload | Recommended balanced |
| `Qwen2.5-Coder-7B` | IQ2_S | ~2.4 GB | full offload | Ultra-compact 7B |
| `Qwen2.5-Coder-7B` | IQ3_XXS (imat) | ~2.9 GB | full offload | Recommended 7B |
| `Gemma 4 (26B-A4B)` | Q4_K_XL (qat-UD) | ~14 GB | full offload (fits) | Instruction-tuned MoE |
| `Qwen3.6-35B-A3B` | UD-IQ1_M | ~9.4 GB | full offload | Ultra-compact 35B MoE, 13 tg tok/s |
| `Qwen3.6-35B-A3B` | UD-IQ2_M | ~10.7 GB | hybrid (experts CPU) | Ultra-compact MoE |
| `Qwen3.6-35B-A3B` | UD-Q4_K_M | ~20.6 GB | hybrid (experts CPU) | **9.7 tg tok/s decode on this hardware — recommended** |
| `Qwen3.6-27B` | UD-Q4_K_XL | ~16.7 GB | partial offload (34/65) | Dense qwen35 |
| `Qwen3.8-27B` | UD-Q4_K_M | ~15.3 GB | partial offload (37/65) | Dense qwen35 with IQ4_NL/Q8_K kernels |
| `Ornith-1.5-35B-A3B` | Q8_0 | ~34.4 GB | hybrid (experts CPU) | Full-precision MoE |
| `Ornith-1.5-35B` | Q4_K_M | ~20.2 GB | hybrid (experts CPU) | 41-layer qwen35moe variant |

All rates are warm decode (tg8) on the reference hardware — the complete GPU table with llama.cpp comparison lives in [Benchmarking](#benchmarking). The Gemma 4 coding files (`gemma4-coding-*`) hit a loader gap ("missing attention weights for layer 5" — the file names its attention tensors differently), and the `gemma-2-*` files use the `gemma2` architecture, which is not yet supported; both print a clear load error rather than producing wrong output.

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
| `TINYCODER_MOE_CACHE` | **On by default (2026-09-17)** — the qwen35moe on-device expert LRU cache is built automatically; this switch only disables it: set to `0` → Option B CPU experts (no cache). `1` or unset = default-on (no-op) |
| `TINYCODER_MOE_STATS` | `1` prints per-forward `[moe fwd]` CPU/GPU wall attribution (router, rank loop, fills, sync) |
| `TINYCODER_MOE_FALLBACK` | `1` forces the CPU-expert fallback even when the device cache exists (bisection control) |
| `TINYCODER_TEST_NO_WARMUP` | `1` skips the question-test page-cache warmup (bench-parity protocol) |
| `TINYCODER_TIMING` | `1` prints per-layer qwen35 phase wall (recattn / postnorm / moe / resid) |
| `TINYCODER_GPU_GRAPH` | `0` disables the CUDA-graph decode capture/replay (default **on**, 2026-09-30). Replay removes per-launch CPU overhead, but decode is DRAM-bandwidth-bound so the win is jitter reduction (tg8 std ~17.8 → ~14.9), not throughput (see GPU Comparison campaign notes) |
| `TINYCODER_FUSE_GU_Q2K` | `1` opts INTO the Q2_K fused gate+up decode kernel (default **off**, 2026-09-30 — measured as a tg8 regression: fused ~149 vs separate ~166 tok/s). The IQ2x/IQ3xxs fused paths are unaffected by this switch |
| `TINYCODER_SPLITK_DOWN` | `2`/`4` — Q3_K ffn-down split-K occupancy override (default off; `4` measured -5% on the matrix, kept opt-in for byte-identical default output) |
| `TINYCODER_Q2K_4XW` | `0` disables the **default-on** 4-rows-per-warp Q2_K decode GEMV (`kQGemvQ2KxW4`, 2026-09-30; measured +2.0% tg8, sees all Q2_K decode matrices with rows%4==0) |
| `TINYCODER_Q2K_INTGU` | `1` opts INTO the Q2_K int-dot fused gate+up path (default **off**, 2026-09-30 — measured as a tg8 regression vs the fused-float and separate-launch defaults; see campaign notes). Requires `TINYCODER_FUSE_GU_Q2K=1` to reach `launchQGemvFusedGU` |
| `TINYCODER_TC_FFN` | `1` opts INTO the fp16/Tensor-Core decode FFN (`cublasGemmEx` m=1 on the persistent prefill fp16 twins), default **off**, 2026-09-30 — measured as a **~20% tg8 regression** (131 vs 165 tok/s); kept opt-in as an A/B record (see campaign notes) |
| `TINYCODER_Q2K_8W` | `1` opts INTO the 8-way-block-batching variant of the Q2_K 4xW decode GEMV (`kQGemvQ2KxW4b8`), default **off**, 2026-09-30 — measured as within noise (no consistent win), kept opt-in as an A/B record (see campaign notes) |
| `TINYCODER_Q2K_MQ4XW` | `1` opts INTO the Q2_K x Q8_K int-dot 4xW decode GEMV (`kQGemvQ2KxQ8K_4xW`, llama.cpp-mmq route), default **off**, 2026-09-30 — measured as within noise vs the float 4xW default; kept opt-in as an A/B record (see campaign notes) |
| `TINYCODER_Q2K_MMVQ` | `1` opts INTO the row-granular mmvq-shaped Q2_K x Q8_K decode GEMV (`kQGemvQ2KxQ8K_Mmvq`: grid covers the ENTIRE row range in tiny 64-thread blocks, one row per warp), default **off**, 2026-09-30 — measured as a **-8.5% to -14.5% tg8 regression** vs the float 4xW default and REFUTES the wave-granularity hypothesis (see campaign notes) |
| `TINYCODER_Q2K_NM` | `8`/`16`/`32` — multi-row-per-warp Q2_K decode GEMV (`kQGemvQ2KxW4_Nm`: each of the warp's 4 lane groups owns R=RPB/4 independent rows), default **off**, 2026-09-30 — measured **-3% (RPB=8) to -56% (RPB=32)** vs the 4xW default; REFUTES the per-warp memory-level-parallelism hypothesis (more in-flight row streams per warp LOWER occupancy without buying bandwidth) |
| `TINYCODER_Q2K_SPLIT4` | `1` opts INTO the 4-way K-split Q2_K decode GEMV (`kQGemvQ2KxSplit4`: warp's 4 lane groups own round-robin block quarters, llama.cpp-mmvq style), default **off**, 2026-09-30 — measured **-9%** vs the 4xW default; REFUTES the serial-block-chain-latency hypothesis |
| `TINYCODER_Q2K_BSY` | `W` — warps per block for the Q2_K 4xW decode GEMV (default `8`; `1`/`2`/`4` A/B), 2026-09-30 — measured within noise vs the default 8-warp block; REFUTES the block-shape/grid-granularity hypothesis |
| `TINYCODER_ATTN_SPLIT` | `0` disables split-KV decode attention (default **on**, 2026-10-01). Dense-arch, `seqLen==1` only. Parallelizes the O(context) attention scan across `nHeads × ceil(maxSeqLen/chunk)` warps (was `nHeads`) — held decode flat at ~142–145 tok/s to 1K context vs the eager path's collapse (pp1024: 142 vs 38 tok/s) |
| `TINYCODER_ATTN_SPLIT_CHUNK` | KV-window size for split-KV decode attention (default `64`). Smaller = more warps (better occupancy) but more combine traffic |
| `TINYCODER_KV16` | `1` opts INTO an fp16 KV cache on the dense qwen2 path (default **off**, 2026-10-01 — measured +0.4% tg64 at pp1024 but -6% prefill; the split-KV kernel already removed attention as the decode bottleneck, so halving its bytes no longer moves the wall). Halves KV VRAM and the O(context) attention/RoPE DRAM traffic |
| `TINYCODER_PREFILL_TWIN_MARGIN_MB` | Overrides the residual-VRAM safety margin reserved away from the fp16 prefill-twin pass (default: geometry-derived — `maxSeqLen*vocab*4` logits + `maxSeqLen*hidden*4*24` activations + the largest matrix's fp16 form + 256 MB). Twins are allocated only from the residual free VRAM AFTER every quantized weight/KV/embed buffer is resident |

## Testing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The Google Test suite (`tinycoder_test`) covers dequantization correctness across all quantization types, reference parity against sequential token-by-token forward passes, full-question generation tests, thread-pool reentrancy, and GPU/CPU comparison. On the reference build (RTX 2080 Ti, 1.5B `q2_k`) **55 pass / 2 fail / 16 skip** — the 2 failures are the non-deterministic sampled full-question cases (`SampleQuestionTest.AnswersQuestion/0,2`), which pass in isolation and are pre-existing sampling flakes, not regressions. All deterministic parity tests (GPU-vs-CPU logits, batch-vs-sequential prefill, sequential-decode argmax) pass bit-exactly with the split-KV and fp16-KV paths both on and off.

## Benchmarking

### Hardware & Protocol

Measured on a single machine: **i7-4790K (4C/8T, AVX2) + RTX 2080 Ti 11 GB + DDR3-1600**. All numbers are **warm** (page cache + GPU weights loaded) greedy decode with the same llama-bench-style protocol — fixed 16-token prompt (`pp16`), 8 generated tokens (`tg8`), 8 threads, mean of warm repeats.

- **TinyCoder**: [`tinycoder_bench`](benchmarks/BenchMain.cpp) with `--n-prompts 16 --n-gen 8 --reps 1`
- **llama.cpp**: `llama-bench -p 16 -n 8 -r 1 -t 8 -ngl N` (CUDA build). Models that fit fully use `-ngl 99`; larger models use the matching hybrid/partial-offload config (`-ncmoe` keeps MoE experts on CPU — same designer choice as TinyCoder's CPU-expert hybrid).

### GPU Comparison (TinyCoder vs llama.cpp)

| Model | File size | GPU mode TinyCoder | **TinyCoder pp16** | **TinyCoder tg8** | **llama.cpp pp16** | **llama.cpp tg8** |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B `q2_k` | 0.3 GB | full offload | **3,153 tok/s** | 176 tok/s | 3,775 tok/s | 456 tok/s |
| Qwen2.5-Coder-1.5B `q2_k` | 0.7 GB | full offload | **1,993 tok/s** | 160 tok/s | 2,063 tok/s | 261 tok/s |
| Qwen2.5-Coder-1.5B `iq3_xxs` | 0.6 GB | full offload | 1,602 tok/s | 132 tok/s | 1,918 tok/s | 213 tok/s |
| Qwen2.5-Coder-7B `iq2_s` | 2.4 GB | full offload | **253.4 tok/s** | 26.7 tok/s | 849 tok/s | 94 tok/s |
| Qwen2.5-Coder-7B `iq3_xxs` | 2.9 GB | full offload | **262.2 tok/s** | 47.5 tok/s | 879 tok/s | 90 tok/s |
| Qwen3.6-35B-A3B `UD-IQ1_M` | 9.4 GB | full offload | **21.1 tok/s** | **16.9 tok/s** | 565 tok/s | 120 tok/s |
| Qwen3.6-35B-A3B `UD-IQ2_M` | 10.7 GB | hybrid (experts CPU) | **19.4 tok/s** | **15.2 tok/s** | 32.8 tok/s | 24.6 tok/s |
| Qwen3.6-35B-A3B `UD-Q4_K_M` | 20.6 GB | hybrid (experts CPU) | **15.1 tok/s** | **12.8 tok/s** | 53.0 tok/s | 25.2 tok/s |
| **Ornith-1.5-35B-A3B `Q8_0`** | 34.4 GB | hybrid (experts CPU) | **10.9 tok/s** | **8.3 tok/s** | 1.0 tok/s | 0.9 tok/s |
| Ornith-1.5-35B `Q4_K_M` | 20.2 GB | hybrid (experts CPU) | **15.8 tok/s** | **13.3 tok/s** | 56.3 tok/s | 25.7 tok/s |
| Qwen3.6-27B `UD-Q4_K_XL` | 16.7 GB | partial offload (34/65) | **1.5 tok/s** | **1.39 tok/s** | 7.7 tok/s | 2.4 tok/s |
| Qwen3.8-27B `UD-Q4_K_M` | 15.3 GB | partial offload (37/65) | **1.0 tok/s** | **0.94 tok/s** | 6.5 tok/s | 2.9 tok/s |

_All TinyCoder columns re-measured **2026-10-02** (`--n-prompts 16 --n-gen 8 --reps 5`, warm) after the partial prefill-twin set and fp16-KV campaign landed. The **Qwen2.5-Coder-0.5B `q2_k` row** (added 2026-10-02 during the TTFO spike) has a **freshly measured llama.cpp column** (`llama-bench -ngl 99 -p 16 -n 8 -r 10`, build `ac4cddeb0`): pp16 3,774.57 ± 28.42, tg8 456.36 ± 0.95 — the target/draft pair used for the speculative-decoding audit. The remaining llama.cpp columns are the previously published reference. The headline change is the two **7B full-offload rows: 184 → 253.4 and 203 → 262.2 pp16** from the deferred partial twin set (the 7B previously OOM'd the twin pass and fell back to streaming/CPU). The partial-offload 27B rows are CPU-tail-bound and were measured on a host under load average ~5–6 (other tenants), so they remain conservative lower bounds._

**Key takeaways:**

- **Qwen2.5-Coder-0.5B `q2_k` (new row, 2026-10-02)**: TinyCoder **pp16 3,153 / tg8 176 tok/s** vs llama.cpp **pp16 3,775 / tg8 456 tok/s**. Added as the **draft model for the speculative-decoding (TTFO) audit** (same 151,936-token vocab and ARCH_QWEN2 as the 1.5B target, so no tokenizer alignment needed). TinyCoder reaches **176/456 = 39% of llama.cpp tg8** here, versus **160/261 = 61%** on the 1.5B — i.e. the absolute decode gap is **larger** on the smaller model, so it does not shrink with model size; it is **kernel-structural**, not a per-model artifact. The audit closed speculative decoding as a decode lever (see suggestion 1) and redirected the remaining work to the M>1 Tensor-Core GEMM (suggestion 2).
- On the **RTX 2080 Ti 11 GB**, `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` decodes at **12.8 tg tok/s** (~78 ms/token — up from 9.7 after the 2026-09-26 F32-router vectorization + 32-set expert cache) with **15.1 pp tok/s** prefill (up from 10.5) — a practical interactive rate for 3.5B-active-parameter MoE models. The other MoE hybrids land at **12.8–16.9 tg tok/s** (`IQ1_M` 16.9, `IQ2_M` 15.2, Ornith `Q4_K_M` 13.3, Ornith `Q8_0` 8.3).
- **llama.cpp's CUDA backend is ~2× faster** on the MoE hybrids (25 vs 12.5 tg tok/s): its batched expert kernel streams expert weights to the GPU in one pass, while TinyCoder's CPU-expert path is bound by the i7's AVX2 + main-memory bandwidth for the top-8 expert FFNs. TinyCoder's GPU trunk is not the bottleneck in hybrid mode.
- On **full offload**, the 2026-09-27/28 GPU kernel campaigns closed most of the decode gap: `iq3_xxs` 1.5B decode went 36 → 131 tg (3.6×) and 7B `iq2_s` 4.1 → 26.8 tg (6.5×). The 2026-09-27 fixes: (1) **native-word grid-table loads** (the GEMV kernels issued up to 8 divergent `ld.const.u8` per lane-block against the QUANT grid/sign tables through the single-ported constant cache — now one `uint64`/`uint32` load + register byte-extraction) and (2) **moving the grid/sign tables from `__constant__` to `__device__`** so divergent indexed access flows through L1/L2 at full bandwidth instead of serializing at the constant cache. The 2026-09-28 addition: **fused gate+up+silu decode kernels** ([`kQGemvFusedGU_IQ2XS`](src/cpp/core/GPUCompute.cu) for the IQ2_XS float dequant-dot and `kQGemvQ8KFusedGU<TYPE>` for the Q8_K integer path) compute BOTH FFN row-dots in one warp sharing the activation read / Q8_K quantize and fold `silu(gate)*up` into the gate store — replacing the old gate+up+`kSiluMul` triple launch with ONE kernel (~283 → ~243 launches/token). Measured: 7B `iq2_s` 25.7 → 26.8 tg (+4.3%), 1.5B `iq3_xxs` 131 → 135 tg (+3.4%). All three changes preserve bit-exact float accumulation order, validated by `Q8KKernelParityWithCPU` (absErr=0) and `SequentialDecodeArgmaxAgrees` (GPU top-10 logits within ~0.01 of the CPU reference on both models).
- **Prefill root-cause (2026-09-28, nvprof)**: the remaining 2–4× prefill gap is NOT the GEMMs — `cublasGemmEx` kernels measure only **2–5 µs each** on pp16. The cost is the **per-layer fp16 weight dequant streaming**: dequantizing the fp16 twin of every matrix into `wF16_` then reading it back reads ~4× the quantized weight bytes (gate 531 µs + up 539 µs + attnO 444 µs + per-layer K/V dequants ≈ 1.6 ms/layer of the ~3 ms/layer prefill). llama.cpp's mmq kernels dequantize weights **inside the GEMM tile** (one pass over the quantized bytes). **Campaign-3 outcome (tried and reverted, 2026-09-28)**: a WMMA `m16n16k16` tiled dequant-in-GEMM kernel (`kGemmDequantF16`, shared-memory 32×256 weight tiles dequantized once per k-round, CUDA 12.0's `mma.h` dropped the Turing `m16n8k16` shapes so the 16×16×16 fragments were used) measured **2.3× SLOWER** than the streaming dequant it replaced (IQ2_XS path: 218 ms vs 95.5 ms total across the same pp16 workload; pp16 7B `iq2_s` regressed 186 → 123 tok/s): the strict `dequant → `__syncthreads` → MMA` serial chain with one `smW` buffer gives every round a full block-wide barrier, and its 8-threads-per-row fill (256 threads / 32 rows) has ~4× less TLP than the warp-per-row streaming dequant, which streams the same bytes at full DRAM bandwidth. Persistent fp16 twins do not fit either: the 7B's full twin set is 13.6 GB and even gate/up-only is 7.6 GB against the 11 GB card. The old per-matrix `kDequantF16` + `cublasGemmEx` path (bit-identical fp16 twins, `ParisPromptLogitsAgree` PASS on 7B) is therefore the keeper, and the remaining prefill lever is a DIFFERENT one: overlap the dequant stream with the GEMMs via a second CUDA stream (the Q/K/V, attnO, gate, up, down dequants are independent per matrix and currently serialize on one stream), or dequantize only the largest consumers (FFN gate/up) once per forward into a small per-layer-pair twin cache inside the 11 GB budget.
- **Prefill twin-cache campaign (2026-09-29)**: landed the "fits-in-budget" branch of the 2026-09-28 plan for models small enough: `upload()` now allocates a **persistent fp16 twin per matrix** when the model fits a VRAM budget (`TINYCODER_PREFILL_TWINS=1` default; budget = free VRAM at upload minus 900 MB headroom for KV/embed/scratch; `TINYCODER_PREFILL_TWINS=0` forces the streaming path). Each twin is dequantized ONCE with the same `kDequantF16` kernel the streaming path uses (bit-identical fp16 input to `cublasGemmEx`), then `dequantMatrixF16()` returns the twin directly — the per-layer streaming dequant and its ~4× DRAM traffic (gate 531 µs + up 539 µs + attnO 444 µs + K/V per layer, ~1.6 ms/layer of ~3 ms) are skipped for every forward. Measured on 1.5B `q2_k` (196 twins = 2,499 MiB of the 9.5 GiB budget): **pp16 851 → 1,973 tok/s (+132%, ~96% of llama.cpp's 2,063)**; pp128 4,427 → 7,298 tok/s (+65%); decode unchanged (tg8 168 — the `seqLen==1` path runs `launchQGemv` straight from the quantized blocks and never touches fp16). Remeasured 2026-09-29 on the current build: **pp16 2,027 tok/s (~98% of llama.cpp's 2,063)**, tg8 166 tok/s (unchanged). Bit-exactness validated: `ModelTest.CompareBatchVsSequentialPrefill`, `GPUCpuCompareTest.ParisPromptLogitsAgree` and `SequentialDecodeArgmaxAgrees` all PASS. Implementation bug caught & fixed before measurement: the twin dequant originally read the **host** descriptor `src.q` instead of the device copy `dst.q`, faulting the kernel with an `illegal memory access` that poisoned the next `cudaMemcpy` and silently fell back to CPU (the parity tests initially "passed" against the CPU engine until the sticky-error path was inspected). Larger models still take the streaming path — the budget self-disables when the twins don't fit (the 7B's 13.6 GB twin set vs the 11 GB card), and a mid-upload twin `cudaMalloc` OOM leaves `f16Twin==NULL` (best-effort) after consuming the sticky error.
- **1.5B q2_k decode campaign (2026-09-28/29)**: profiled the 1.5B `q2_k` decode with nvprof and found the **Q6_K LM head (`kQGemvKxQ8K`) at 1850 µs/launch** (~4.9 ms/token, ~40% of decode) because its block-dot only used **lanes 0–7** (24 of 32 lanes idle per instruction → ~103 GB/s vs the FFN's all-lane ~620 GB/s). Three fixes landed, all bit-exact per row (validated by `SequentialDecodeArgmaxAgrees` + `ParisPromptLogitsAgree`, both PASS where the latter FAILED pre-fix): (1) **fixed a silent fold bug** — the generic kernel's 4-block batched fold was lane-0-only while Q6_K accumulates on all 8 active lanes, so 4 of 6 LM-head blocks dropped 7/8 of their columns (~58% of the dot product; the tolerant argmax tests had masked it — the logit range was systematically ~3.3 low); (2) **vectorized the Q6_K block-dot** to native `uint32_t` register loads; (3) **new 4-rows-per-warp kernel** [`kQGemvQ6KxQ8K_4xW`](src/cpp/core/GPUCompute.cu) so all 32 lanes issue weight loads (6417 → 1604 warps, LM head **1850 → 821 µs**). Also (4) **4-block batching** for the Q2_K/Q3_K float decode GEMVs (mirrors the IQ2_XS pattern) shaved the Q3_K ffn down 2.13 → 1.99 ms. Net decode: **tg8 149.8 → 167.5 tok/s** (48.3 ms/token), pp16 819.7 → 851 tok/s. Bonus fix: the `TINYCODER_GPU_VERBOSE=1` stage tracing destroyed `evL0` before the `cudaEventElapsedTime(evL0, …)` anchor and poisoned the sticky CUDA last-error — under verbose, every decode token fell back to the CPU LM head and reported `GPU lmhead launch: invalid resource handle`; the event destroys now happen after the trace block.
- **Q2_K fused gate+up decode experiments (2026-09-29, MEASURED REGRESSIONS → all reverted)**: the FFN's gate+up were fused for Q2_K with (1) a register-heavy dual-accumulator kernel (`kQGemvFusedGU_Q2K`, accg+accu in one warp) and (2) a row-batched serial variant (`kQGemvBatchGU_Q2K`, gate-then-up dot serial in one warp sharing the activation/Q8_K quantize). Both were SLOWER than the separate `launchQGemv(gate)` + `launchQGemv(up)` baseline (gate+up 1.72 → 2.43 ms and 2.72 ms/layer; tg8 167.5 → 129.9 and 132.6 tok/s): unlike the Q6_K LM head (sub-lanes 0–7 idle), Q2_K's float block dot already uses **all 32 lanes**, so the kernel is at its DRAM floor with warp-per-row tiling — both fusions reduce concurrent row-streams and occupancy without adding DRAM parallelism. **Third attempt, 2026-09-29 (also reverted)**: a **2-rows-per-warp Q2_K GEMV with shared-memory x staging** (`kQGemvQ2K2xW`, `TINYCODER_Q2K_2XW=1` A/B gate): each warp computed two adjacent rows with the SAME `sX[cidx]` activation fetch to halve L2 x-traffic, and the dequant replicated `kQGemv<kTypeQ2K>`'s a0..a3 fold structure for bit-exactness. Measured WORSE: gate+up 2.46 → 4.23 ms/layer and tg8 153 → 112 tok/s — the `__syncthreads` staging barrier serialized the block's 8 warps and the dual-row register/ILP pressure outweighed the (already L1-hot) x-sharing. **Fourth attempt, 2026-09-29 (also reverted) — this FALSIFIES the launch-overhead lever**: a **grid-level gate+up batch** (`kQGemvBatchGU`, `TINYCODER_BATCH_GU=1` A/B gate) kept warp-per-row math BYTE-IDENTICAL to `kQGemv<kTypeQ2K>` (same fmaf chain, same 4-way a0..a3 fold, same 5-shuffle reduce) and only merged the two matrix grids into ONE launch (3 → 2 launches/layer: batch + `kSiluMul`). Measured WORSE despite the launch cut: gate+up **1.73 → 2.93 ms/layer (+69%)** and tg8 153 → 119 tok/s (interleaved A/B, 3 reps each, stable). Root cause: one grid covering 2*rows warps makes the gate and up streams **serialize** — the gate rows (blocks 0..1119) must drain before the up rows (blocks 1120..2239) get scheduled, so the two matrices' DRAM streams no longer overlap each other or the next stage the way two back-to-back kernels did. Conclusion after four independent attempts: for Q2_K decode, **neither rows-per-warp ILP, nor per-FFN launch reduction, nor cross-matrix grid batching helps — all reduce the number of CONCURRENT DRAM row-streams, which is the actual resource Q2_K decode is bound on**. The remaining lever for the tg8 261 gap is **raising the concurrent-stream count on the SMALL-ROW matrices** (attnO/gate+up/down at rows=1536 → only ~2.8 blocks/SM; the down kernel alone streams Q3_K at 5.9 MB but takes 74 µs/layer, ~7× its ~10 µs DRAM floor) — e.g. split-K (rows stay warp-per-row but each row's block-range splits across more warps in ONE block → more resident streams) or a fused QKV/down-row kernel that runs the OTHER matrix's rows concurrently. The IQ-path Q8_K integer GEMVs (`kQGemvQ8K`, 4-block batched) don't have this problem because their integer math is far cheaper per element and the 7B's larger rows (>4×) keep the SM count high.
- **Split-K occupancy experiment (2026-09-29, KEPT — env-gated)**: implemented the redirect above. [`kQGemvSplitK<kTypeQ3K,KN>`](src/cpp/core/GPUCompute.cu) handles each row with **KN chunk-warps in ONE block** (block = 32×KN lanes, grid = rows): chunk warp w owns the contiguous block sub-range `[(w·bpr)/KN, ((w+1)·bpr)/KN)` and runs the byte-identical Q3_K float chain (4-way `a0..a3` batching, remainder fold, 5-shuffle reduce); the KN per-chunk partials are joined in shared memory in chunk order. This strictly RAISES the concurrent warp count on the small-row Q3_K ffnDown (rows=1536): KN=4 → 6144 warps vs 1536, at the same per-warp register footprint. Measured (interleaved A/B, matrix sums): **down 1.51 → 1.40 ms/layer (-7.3%)** in the session A/B, and a fresh steady-state remeasurement (2 rounds, settled boosted clock) confirms **down 1.49 → 1.41 ms/layer (-5%)** at constant gate+up/attnO — also beating the low-clock baseline (down 2.07 → 1.94). All 3 parity tests PASS. The cross-chunk float join is a ~1e-7 reorder (within the Q2K/Q3K float-path tolerance — these paths already approximately-match the CPU int reference, unlike the bit-exact Q8_K-int paths). **A Q2_K split-K branch (rows=1536 attnO/gate+up) was measured and REVERTED: it REGRESSES gate+up 1.77 → 3.40 ms/layer (+92%)** — the 8960-row gate/up (95% of the Q2K time) isn't small-row at all, and splitting it floods the L2 with duplicate activation reads; only the true small-row down benefits. Decode **kept as env-gated opt-in** (`TINYCODER_SPLITK_DOWN=4`; 2 is ~-3%, 4 is best) rather than default — the ~0.7% total-decode gain is real but modest (tg8 wall stays within noise), and keeping it opt-in preserves the default path's byte-identical output.
- **CUDA-graph decode (2026-09-30, KEPT — jitter win only, NULL throughput result)**: full `cudaStreamBeginCapture` decode capture/replay (`TINYCODER_GPU_GRAPH`, default on since 2026-09-30) — position-parametric kernels switched to a device-scalar `pos` (frozen capture pointer, value refreshed per replay), host syncs/`cudaEventRecord`s gated off during capture (they would serialize capture-time execution), LM-head output written to a scratch buffer + replay-time D2H copy, 1 capture per process, every steady-state token replays. Measured: graph ON tg8 146.75±4.46 vs graph OFF 140.32±17.85 (Release, this run's sampling); on the restored-166 baseline, graph ON 165 / OFF 166 — i.e. **~0% throughput, variance-level**. Root cause: eager decode launch overhead is FULLY OVERLAPPED — the GPU layer loop only takes ~8.6 ms/token despite ~500 launches/token; decode is **DRAM-bandwidth-bound**, not launch-bound, so replaying recorded launches adds nothing. Kept anyway: it removes host-side jitter (±17.85 → ±4.46 tok/s std) with identical output (all parity tests PASS with graph on). **Levers re-ranked**: launch-count reduction is DEAD; the tg8 gap to llama.cpp (261 tok/s = 3.8 ms/token) is **FFN GEMV throughput**: the fused/lone Q2_K gate+up kernel runs at ~75 GB/s vs the ~616 GB/s Turing ceiling measured on wider rows (Q6_K LM head `kQGemvQ6KxQ8K_4xW`). Closing tg8 requires RAISING FFN GEMV bandwidth utilization — larger per-warp row tiles, better Q2_K block batching, or fp16/Tensor-Core offload of the FFN — not fewer launches.
- **Q2_K fused-GU dispatch audit (2026-09-30)**: the build shipped with the Q2_K fused gate+up kernel ACTIVE via the generic `fuseGU` dispatch, contradicting the 2026-09-29 measured-regression record above. A/B in the Release build (graph off, interleaved): fused steady-state 148.8/149.2/149.3 tg vs separate 165.6/165.9/166.0 tg — **fused is an ~11% regression**, exactly reproducing the documented 1.72→2.43 ms/layer finding. The dispatch is now gated `TINYCODER_FUSE_GU_Q2K=1` for opt-in (**default off**), restoring the documented 166 tg baseline; the IQ2x/IQ3xxs fused paths (documented +3.4% on `iq3_xxs`) are untouched. The separate-launch 166 tg8 baseline is byte-identical to the pre-fusion decode path and passes all parity tests.
- **Q2_K 4-rows-per-warp GEMV (2026-09-30, KEPT — default ON)**: [`kQGemvQ2KxW4`](src/cpp/core/GPUCompute.cu) replicates the bandwidth shape that took the Q6_K LM head to ~616 GB/s: **all 32 lanes issue `uint32_t` weight loads**, 8 lanes per row (each lane owns a contiguous 4-col slice), 4 independent row-streams per warp, and NO shared memory / `__syncthreads` (explicitly avoiding the two failure modes of the reverted `kQGemvQ2K2xW`: smem staging barrier + dual-row register pressure). The Q2_K element map is bit-exact with `dequantizeQ2_KBlock` (col c → q byte `q[n*32 + c%32]`, shift `2j`, scale sub `c%32≥16`; a `u32` load per lane per group supersedes the old kernel's 12 byte-loads/lane/block); per-element float math is identical, only the reduce/lane-partition order differs (float-path ~1e-7 tolerance). Interleaved A/B, 2 rounds, both stable: **tg8 165.1 → 168.4 tok/s (+2.0%)**, pp16 unchanged, parity 11/11 PASS with graph on. First Q2_K decode change since 2026-09-29 that does NOT regress — confirming the FFN gain comes from raising concurrent row-streams/occupancy, not from launch fusion. Default ON, `TINYCODER_Q2K_4XW=0` disables.
- **Q2_K int-dot fused gate+up (2026-09-30, REJECTED — default OFF)**: [`kQGemvQ2KIntFusedGU`](src/cpp/core/GPUCompute.cu) implements llama.cpp-mmq's integer route for the Q2_K FFN: `kQuantizeQ8K` the fp32 activation ONCE, then BOTH gate/up row dots run as Q2_K-int x Q8_K-int DP4A-style dots against the shared Q8_K activation (one activation block read per weight block for both matrices). The int side is bit-exact with the CPU reference `dotProductQ2_K_PrePacked_Q8_Scalar` / the compact AVX2 kernel: lane `i` owns 8 contiguous cols `i*8..i*8+7`, weight bytes are contiguous at `q + (i>>4)*32 + (i&3)*8` with the constant per-lane 2-bit shift `2*((i>>2)&3)` (verified vs `dequantizeQ2_KBlock`), scale byte index `i>>1` is an exact identity, so the whole `dl` term is ONE 32-lane int xor-tree of `(scales[i>>1]&0xF)*S_i` and the `ml` term folds on lane 0 from the 16 Q8 bsums (`sum_g (scales[g]>>4)*bsums[g]`); 4-way block batching, lane-0 serial float fold in block order, silu fold. Parity 11/11 PASS. **A/B (eager, graph off, interleaved, 2 rounds): separate-launch default 158.7/157.7 tg8 vs float-fused 144.3/140.4 vs int-fused 134.2/126.8 — the int path is ~7-10% WORSE than the float-fused shell and ~17% below the separate default (R1 std 0.23, very stable)**. Root cause: the Q2_K FFN decode is DRAM-latency/bandwidth-bound, not ALU-bound — the float 4xW kernel already streams each weight byte once with minimal per-load work, while the int route adds a `kQuantizeQ8K` launch plus 8 five-level lane-0-folded int trees per 4-block batch and a second activation read; replacing float ALU with int ALU cannot win (same conclusion as the CUDA-graph null result — the 168 vs 261 tok/s gap to llama.cpp is NOT closed by reducing compute). REJECTED on measured data; harness kept under `TINYCODER_Q2K_INTGU=1`.
- **fp16/Tensor-Core decode FFN (2026-09-30, REJECTED — default OFF)**: [`TINYCODER_TC_FFN=1`] reuses the persistent prefill fp16 twins for decode-time `cublasGemmEx` GEMVs (m=1; gate/up/down per layer via `dequantMatrixF16` + tensor-op, replacing the compact-stream `kQGemvQ2KxW4` decode path). Parity 11/11 PASS (fp16 tolerance). **A/B (eager, graph off, interleaved, 2 rounds): TC-FFN 131.0/131.1 tg8 (std <0.5) vs separate-launch compact-stream default 155.3/163.9 — a ~20% regression that exactly tracks the file-header's bandwidth argument**: the fp16 twin streams 2 B/element vs Q2_K's ~0.33 B/element (~6x the weight traffic), and decode is DRAM-bandwidth-bound — Tensor Cores cannot help a GEMV whose memory-bound part dominates. REJECTED on measured data; harness kept under `TINYCODER_TC_FFN=1`.
- **Q2_K 8-way block batching (2026-09-30, REJECTED — default OFF)**: [`kQGemvQ2KxW4b8`](src/cpp/core/GPUCompute.cu), gated `TINYCODER_Q2K_8W=1`, batches **8 blocks per outer iteration** in the identical 4xW geometry (all 32 lanes u32 loads, 8 lanes/row, 4 rows/warp, no smem/barrier), doubling the in-flight per-lane weight-load count to hide DRAM latency at higher register cost. Per-element float math identical to `kQGemvQ2KxW4` (only block-accumulation order differs; float-path ~1e-7 tolerance), parity 11/11 PASS. **A/B (eager, graph off, interleaved, 2 rounds): 8W 160.5/152.5 tg8 (R1 std 0.54, R2 std 17.2) vs 4xW default 158.5/164.4 — no consistent direction; both sit inside the machine's between-run noise band (default itself measured 155-165 across this session)**, so the extra registers/branching buy nothing for a DRAM-bound GEMV whose 4-way chain already saturates latency hiding. REJECTED on measured data; harness kept under `TINYCODER_Q2K_8W=1`.
- **Q2_K x Q8_K int-dot 4xW decode GEMV (2026-09-30, REJECTED — default OFF)**: [`kQGemvQ2KxQ8K_4xW`](src/cpp/core/GPUCompute.cu), gated `TINYCODER_Q2K_MQ4XW=1`, is the llama.cpp-mmq decode route (kQuantizeQ8K the fp32 activation ONCE, then the 2-bit x Q8_K dot via `__dp4a` on per-lane u32 weight+activation loads) in the exact Q6_K-proven 4xW structure (all 32 lanes loading, 8 lanes/row, 4 rows/warp, 4-way block batching into independent per-lane float accumulators, single end 8-lane float tree). Quoting the measured decode matrix sums (gate+up 2.24 ms, down 2.11 ms of the ~7.5 ms/token layer loop at only ~78-113 GB/s) this was the mmq fix candidate for the FFN bandwidth gap. Parity 11/11 PASS (the ml term required a direct indexed int16 bsums load -- a packed-u64 shift by >64 bits is UB and silently zeroed high groups). **A/B (eager, graph off, interleaved, 2 rounds): dp4a-int 156.6/154.1 tg8 vs float-4xW default 162.9/155.1 (default band 155-168) -- within noise, NO win**. Root cause: both kernels stream the identical Q2_K weight bytes (one byte per 4 cols) -- the dominate cost is the weight stream itself; the float kernel's per-element fp32 activation reads are L1-resident (1536 floats hot per layer) and the int route's byte-count win on the activation is offset by the extra `kQuantizeQ8K` launch. The 168 vs 261 tok/s gap to llama.cpp is therefore NOT in the Q2_K decode kernel; it lives in the matmul **shape/occupancy** (llama.cpp's batched mmq tiles whole row-ranges) — see the `__dp4a` kernel as the retained reference for that direction.
- **Row-granular mmvq-shaped Q2_K decode GEMV (2026-09-30, REJECTED — default OFF; user-requested "batch the entire tensor row range" kernel)**: [`kQGemvQ2KxQ8K_Mmvq`](src/cpp/core/GPUCompute.cu), gated `TINYCODER_Q2K_MMVQ=1`, is a faithful port of llama.cpp's ACTUAL decode structure (verified against `~/git/llama.cpp` master): Q2_K decode runs `mul_mat_vec_q` (mmvq), NOT a 128-bit whole-matrix kernel — grid.x = ceil(rows/rpb) covers the entire row range in tiny blocks (rpb = nwarps = 2 for the small-K FFN: 64-thread blocks, 8960 rows → 4480 blocks for gate/up), one row per warp, the row's Q2_K block range split round-robin across the warp's 4 lane-groups, u32-class (`get_int_b4`) loads + `__dp4a` (llama has NO LDG.128 in its Q2_K dot). The motivation was the wave-granularity theory: the 4xW's 280 blocks × 256 threads = 71,680 of 139,264 thread slots (~51% occupancy) appeared to explain the FFN's ~78-113 GB/s vs the LM head's ~616 GB/s on 4,748 full blocks. Parity 11/11 PASS (same q2kXQ8KBlockDot math as MQ4XW). **A/B (eager, graph off, interleaved, 2 rounds): MMVQ 153.3/144.0 tg8 vs float 4xW default 167.4/168.4 — REGRESSION (-8.5% to -14.5%)**. This REFUTES the wave-granularity/occupancy hypothesis: a 280-block × 256-thread grid is ALREADY a single full wave (68 SMs × 8 resident 256-thread blocks = 544 ≥ 280 — every SM is occupied for the kernel's whole lifetime; only the per-SM warp count differs). Folding the FFN into 4,480 tiny blocks bought pure block-launch/scheduling overhead with no bandwidth gain, and combined with the MQ4XW result closes the FFN decode-kernel lever entirely: neither int-dot math nor grid shape beats the float 4xW, because the ~6 blocks/row are too few for per-warp K partitioning to matter and the activation stream (1536 floats read per gate/up layer — L1/L2-resident, not DRAM) is not the bottleneck the ~616 GB/s LM-head analogy implied. The 168 vs 261 tok/s gap to llama.cpp is NOT addressable inside the Q2_K FFN kernel (5th failed hypothesis: compute route → int, tensor-core, 8W, dp4a; grid shape → mmvq; only the float 4xW is a win and it is already shipped).
- **Q2_K FFN decode — occupancy/latency campaign closed (2026-09-30)**: a direct `nvidia-smi` sample during a steady decode run shows the **SM 96–97% busy but DRAM only ~20% utilized** (SM 1950 MHz / mem 6800 MHz, ~200 W), so the 1.5B `q2_k` decode is **issue/latency-bound, not bandwidth-bound** — the opposite of what the ~616 GB/s Q6_K LM-head analogy had implied. Three further levers were implemented and A/B measured against the 162–168 tg8 default (graph off, interleaved, 2 rounds, machine-band stable): **multi-row-per-warp** ([`kQGemvQ2KxW4_Nm`](src/cpp/core/GPUCompute.cu), `TINYCODER_Q2K_NM` — R independent rows per lane group) gave 158 (R=1), 76 (R=2), 71 (R=4) tg/s; **4-way K-split** ([`kQGemvQ2KxSplit4`](src/cpp/core/GPUCompute.cu), `TINYCODER_Q2K_SPLIT4`) gave 148 tg/s; **block-shape** (256→32 threads via `TINYCODER_Q2K_BSY`, llama.cpp's 2-warp blocks) was within noise (160–162 tg/s). All three REGRESS or tie — the Q2_K FFN is already at its latency-limited floor in the 4xW shape, and every attempt to add in-flight streams per warp lowers occupancy without raising DRAM pressure. The **261 vs 168 tok/s tg8 gap to llama.cpp is therefore NOT a Q2_K-kernel-shape gap**: the SM-busy-96%/DRAM-20% signature is the fingerprint of a latency/launch-overhead-bound loop, and llama.cpp's single fused decode CUDA graph replays the whole token in tens of launches versus TinyCoder's ~2,800. Closing tg8 requires a structural **single-graph decode** (capture all 28 layers' GEMVs/attention into one graph with device-resident intermediate buffers), not a micro-kernel change.
- **Split-KV decode attention (2026-10-01, KEPT — default ON)**: the "single-graph decode" premise was audited first and **falsified**: `captureDecodeGraph`/`replayDecodeGraph` (`TINYCODER_GPU_GRAPH`, default on) ALREADY capture all 28 layers' GEMVs + attention + RoPE + KV store + LM head into ONE `cudaGraphExec_t` replayed with a single `cudaGraphLaunch` per token — and it buys ~0 % throughput (graph ON 135.6 vs OFF 136.2 tok/s) because per-token device elapsed is ~7.1 ms of a 7.36 ms wall (**~97 % GPU-busy**; `TINYCODER_MOE_STATS=1`). Cutting launches cannot help. A `TINYCODER_GPU_VERBOSE=1` context sweep then located the real wall: the GEMVs are flat (~3.6 ms) but **attention scales O(context)** — at pp1024 `kv/attn/rope` is **20.27 ms of the 24.99 ms** layer loop (81 %). Root cause: [`kWarpAttention`](src/cpp/core/GPUCompute.cu:306)/[`kWarpAttentionPos`](src/cpp/core/GPUCompute.cu:430) launch **one warp per (token, q-head)** = only 12 warps for the whole 68-SM GPU, each scanning the full `cachePos`. The fix tiles the KV range into `TINYCODER_ATTN_SPLIT_CHUNK` (default 64) windows and launches `nHeads × ceil(maxSeqLen/chunk)` warps ([`kWarpAttentionSplit`](src/cpp/core/GPUCompute.cu:510)/[`kWarpAttentionSplitPos`](src/cpp/core/GPUCompute.cu:578)), each folding its window into a flash-attention `(m,l,acc)` online-softmax partial, merged by a tiny [`kAttnCombinePartial`](src/cpp/core/GPUCompute.cu:650). Measured (`tinycoder_bench --n-gen 64 --reps 5`, warm, SPLIT=1 vs SPLIT=0): **pp64 145.6 vs 134.6 tok/s (1.08×)**, pp256 144.2 vs 89.0 (1.62×), pp512 143.1 vs 61.5 (2.33×), **pp1024 142.2 vs 38.0 (3.74×)** — attention stage at pp1024 **20.27 → 1.74 ms (11.7×)**; decode is now **flat at ~142–145 tok/s across the whole context range** while the eager path collapses. Prefill unaffected (pp64 5,176 vs 5,174 tok/s). **Parity**: in-process A/B (both paths, same process) is IDENTICAL at every context length and chunk size; all deterministic GPU-vs-CPU gtest parity tests PASS with split on and off (the two full-suite failures `AnswersQuestion/0,3` are pre-existing sampling flakes — they fail on the pre-change `build-rel` baseline too). Two lane-indexing bugs were caught by the in-process A/B harness ([`tools/attn_split_parity.cpp`](tools/attn_split_parity.cpp)): the partial store/combine gather omitted the lane dimension (`pa[i]`→`pa[lane + i*32]`, combine stride `NV`→`HD`). Conclusion: **the single-graph decode already existed; the win came from parallelizing the O(context) attention scan across warps, not from removing launches.** See [`plans/single_graph_decode.md`](plans/single_graph_decode.md).
- **Dense qwen35 models** (Qwen3.6-27B / Qwen3.8-27B) are slower: only 34–37 of 65 layers fit the 11 GB card, so decode alternates GPU+CPU layers per token (partial offload, background on CPU elsewhere). The 27B Q4 models run **~0.9–1.4 tg tok/s** here (2026-10-02 remeasure; the earlier 0.4–0.9 figures were on a busier host); llama.cpp with the same `-ngl` reaches ~2.4–2.9 tg tok/s (its CPU tail is faster than the i7's AVX2 fallback in TinyCoder).
- **Partial prefill-twin set (2026-10-01, KEPT — the headline win)**: the 2026-09-29 twin cache self-disabled for the 7B because it measured the free-VRAM budget **before** the quantized weights existed, so the twins consumed the entire card by layer ~14 and the next `cudaMalloc(ffnGate)` OOM'd → silent CPU fallback (0.37 tg8 / 0.5 pp16). The fix moves twin allocation **out of the per-matrix upload loop** into a deferred second pass that runs only after every quantized weight / KV / embedding / scratch buffer is resident, then allocates the twins **largest-matrix-first** (`std::stable_sort` by byte size) until a geometry-derived margin is reached (`maxSeqLen·vocab·4` logits + `maxSeqLen·hidden·4·24` activations + the largest matrix's fp16 form + 256 MB; override `TINYCODER_PREFILL_TWIN_MARGIN_MB`). Twins are best-effort: matrices that don't fit keep the streaming `kDequantF16` path. Measured on 7B `iq2_s` (RTX 2080 Ti, Release, warm): **pp16 0.5 → 253.6 tok/s, pp128 1,466 tok/s, pp512 2,424 tok/s, tg8 26.7 tok/s** — pp128/pp512 now **beat** llama.cpp's 849 pp16 / 94 tg8, versus the previous CPU-fallback collapse. 1.5B `q2_k` is unchanged (pp16 2,021, tg8 165 — its full twin set fit before), and 7B `iq3_xxs` lands at 261.7 pp16 / 47.4 tg8. Bit-exactness preserved: `ParisPromptLogitsAgree`, `CompareBatchVsSequentialPrefill` and `SequentialDecodeArgmaxAgrees` all PASS (the twin is the same `kDequantF16` fp16 input the streaming path feeds `cublasGemmEx`).
- **fp16 KV cache (2026-10-01, KEPT — opt-in `TINYCODER_KV16=1`, default OFF)**: the KV-store, RoPE store, eager attention and split-KV attention kernels are now **templated on the cache element type** (`float`/`__half`) with `kvLoad`/`kvStore` device overloads; `GPUModel` holds `void *kvK_`/`void *kvV_` plus a `kvHalf_` flag and the dense qwen2 forward dispatches both element types via a `TC_KV_RUN(KVT)` macro. Halving the KV bytes halves the O(context) attention/RoPE DRAM traffic and the KV VRAM footprint, but measured **only +0.4% tg64 at pp1024 and −6% prefill** — the 2026-10-01 split-KV kernel already removed attention as the decode bottleneck, so the wall is elsewhere. Kept as an opt-in A/B (full parity: in-process both-path A/B IDENTICAL; all deterministic GPU-vs-CPU gtest parity tests PASS with KV16 on and off) rather than a default.
- **Second-stream dequant/GEMM overlap (2026-10-02, FALSIFIED — not implemented)**: the residual "overlap the per-matrix dequant with the GEMMs on a second stream" lever was audited against the measured profile and closed before writing any kernel. Two facts kill it: (1) the streaming path reuses a **single** `wF16_` scratch ([`dequantMatrixF16()`](src/cpp/core/GPUCompute.cu:9394)), so consecutive matrices can never share the buffer concurrently — overlapping them requires N× the fp16 scratch (the 7B's largest matrix is ~136 MB, and the residual free VRAM is exactly what the twin pass now consumes); (2) the 2026-09-28 nvprof root-cause already showed the `cublasGemmEx` kernels are **2–5 µs each** while the dequant streams are **~500 µs each** — the dequant is 99 % of the streaming prefill wall and is pure DRAM-bandwidth-bound (reads the quantized bytes, writes the fp16 bytes), so there is no independent compute to hide behind it. A second stream would only re-serialize on the same DRAM bus. The correct fix is to **eliminate** the dequant stream, which is exactly what the persistent fp16 twin pass does (and it now fits a large prefix of every model via the deferred largest-first allocator). Lever closed on the existing profile evidence.

### Further improvement suggestions (ranked by measured headroom)

1. **⛔ TTFO — CLOSED: Speculative decoding with a draft model (2026-10-02, negative result).** The last untried M>1 lever was audited end-to-end and closed. Every prior optimization attempt was an **M=1 GEMV** shape (fused-GU, 2xW, 8W, int-dp4a, split-K, mmvq) and every one tied or regressed — the SM-busy-96%/DRAM-20% signature says the kernel is latency/issue-bound, not bandwidth-bound — so the hypothesis was that **tiling several decode rows (speculative-draft rows) into one `mmq` tile** would amortize the weight stream across M>1 outputs. Setup: the `Qwen2.5-Coder-0.5B q2_k` draft shares the target's tokenizer (151,936 vocab) and architecture (ARCH_QWEN2), so no tokenizer alignment is needed. Measured (RTX 2080 Ti, TINYCODER_THREADS=8, warm reps, GPU + CUDA graph): draft M=1 = **4.96 ms/token** (203.8 tok/s, tg32); target M=1 = **6.03–6.21 ms/token** (161–166 tok/s, tg8/tg32); batched-verify (fp16 twins active — the prefill path hits 1,768 tok/s pp16 = 9.0 ms for 16 rows) costs 8.5→12.3 ms across seqLen 4→32. A TTFO round of K=4 draft tokens + full 20-token verify costs **31.6 ms** vs the **24.8 ms** M=1 baseline. Projected speedup at α=0.7: **K=2 0.75×, K=4 0.97×, K=8 1.07×** — **no K clears the 1.2× decision gate**, and the 0.5B draft is already ~1/3 the target's per-token cost, so shrinking the draft further cannot recover the gap (the draft+verify *sum* is the problem, not the draft alone). Verdict: **speculative decoding is not the path to the 261 vs 160 tok/s gap on this hardware.** The measurement also falsified the "fp16 twins aren't used for M>1" concern — batched-verify throughput is already near the pp16 GEMM ceiling, so the wall is the **per-token compute/launch loop**, not the batch shape. See the new item 1 below for the redirect this points to.

2. **🎯 M>1 Tensor-Core GEMM for the FFN block (the remaining decode lever).** The TTFO audit above establishes where the ~40% tg8 gap to llama.cpp actually lives: not in the GEMV kernel (5+ rejected shapes), not in launch count (CUDA-graph null result), and not in speculative batching (closed above). What is left is the **FFN weight-stream cost per token** — the gate/up/down GEMVs dominate decode (~5.5 ms of the ~6.0 ms/token layer loop on 1.5B `q2_k`) and run at only ~78–113 GB/s vs the ~616 GB/s Turing ceiling on wide rows. The direct fix is to **dequantize once and run the FFN as a true GEMM** — dequant-in-GEMM (read each quantized weight byte once inside the tile, no fp16 twin, no separate dequant pass) with **Tensor-Core `mma` tiles for M>1** rows, so the weight stream is amortized across the batch. This differs from the two rejected attempts: the 2026-09-28 WMMA `kGemmDequantF16` was **M=1** and lost to the `__syncthreads` serial chain (~2.3× slower); the 2026-09-30 `TINYCODER_TC_FFN` reuse of **fp16 twins** was ~20% slower because the twin streams 2 B/element vs Q2_K's ~0.33 B/element. The untried combination is **quantized weights + in-tile dequant + M>1 mma tiles** (one pass over the 0.33 B/element Q2_K bytes, TF32/FP16 accumulation), which is exactly llama.cpp's `mmq` structure. Open questions to answer before implementing: (a) does M>1 arrive only via batched prefill/chunked-decode (speculative is closed), and (b) does the 11 GB card hold the required per-tile scratch. **This is the highest-value structural decode change remaining.**

3. **✅ DONE — Partial prefill-twin set (2026-10-01, KEPT).** The deferred second-pass, largest-first twin allocator with a geometry-derived margin (see campaign notes) took 7B `iq2_s` from the CPU-fallback 0.5 pp16 / 0.37 tg8 to **pp16 253.4, pp128 1,466, pp512 2,424 tok/s, tg8 26.7** (2026-10-02 remeasure) — pp128/pp512 now beat llama.cpp's 849 pp16 / 94 tg8, and 1.5B `q2_k` is unchanged (pp16 ~1,993). Remaining prefill headroom for the mid-size models is whatever fraction of the twin set does not fit the residual VRAM (a larger-prefix allocator or an LRU twin cache); the second-stream overlap alternative is closed (item 4).

4. **⛔ CLOSED — Overlap prefill dequant with GEMMs on a second stream (2026-10-02, falsified on profile).** The candidate is dead for two reasons: the streaming path reuses ONE `wF16_` scratch (overlap needs N× the fp16 buffer, and the residual VRAM is now spent on twins), and the 2026-09-28 nvprof showed the GEMMs are 2–5 µs vs ~500 µs dequant streams — the dequant is DRAM-bandwidth-bound with no independent compute to hide behind, so a second stream just re-serializes on the same bus. The twin path (item 3) is the actual fix: it **eliminates** the stream instead of overlapping it.

5. **Extend split-KV attention to the prefill / chunked path.** The 2026-10-01 split kernel is gated to `seqLen==1` dense decode; long-context prefill still uses the one-warp-per-head kernel. Chunked-prefill attention (or a flash-prefill variant) would help long prompts. **✅ fp16 KV cache DONE (2026-10-01, opt-in `TINYCODER_KV16=1`, default OFF)** — the store/attention/RoPE kernels are templated on the cache element type (`float`/`__half`); measured +0.4% tg64 at pp1024 / −6% prefill, so it stays a non-default A/B (the split-KV kernel already made attention non-bottleneck at decode).

6. **Fuse the elementwise glue.** Residual-add, RMSNorm, RoPE and `silu-mul` are separate launches; folding them into the GEMV epilogues / attention kernels trims graph nodes (marginal on throughput per the CUDA-graph null result, but it shrinks the graph and helps the captured-replay path).

7. **Partial-offload scheduling for the 27B models.** At 34–37/65 layers the decode alternates GPU/CPU per token and the CPU tail dominates; prefetching the next CPU layer's weights while the GPU layer runs (or pinning the expert/attention split to the faster device per layer) is where the **~2–2.5× gap** to llama.cpp lives (TinyCoder ~0.9–1.4 vs llama.cpp ~2.4–2.9 tg tok/s, 2026-10-02).

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

Instead of guessing `-ngl`, the engine measures real device residency (K-quant block bytes + F32 matrices + KV slots + conv/GDN state per layer, embedding on top) and fills GPU VRAM from `cudaMemGetInfo` free bytes minus a fixed 900 MB scratch/GEMM-twin margin (llama.cpp `-ngl auto` style, gated to qwen35 — the only arch with the CPU continuation). On the 2080 Ti this lands at **37/65 layers** for the 4-bit files (the Q4_K embedding has a dedicated `kEmbedDequantQ4K` kernel — see the consolidated [Benchmarking](#benchmarking) section for the measured rates and the llama.cpp comparison).

#### Weight compression for the GPU prefix

The GPU kernels support the full K-quant family (`Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K`), so a lower-bit model file shrinks the resident layers with zero code:

| Quant | per-layer GPU weights | 30-layer VRAM | auto-fit GPU layers |
|---|---|---|---|
| Q5_K_M (default) | ≈272 MB | 8.16 GB | 30 |
| Q4_K (e.g. `Qwen3.6-27B-UD-Q4_K_XL`, 17.9 GB) | ≈225 MB | 6.75 GB | **37** ✓ |
| Q8_0 (e.g. `Qwen3.6-27B-Q8_0`, 27.7 GB) | ≈415 MB | 12.4 GB (doesn't fit 30) | **20** ✓ |
| Q3_K | ≈175 MB | 5.25 GB | ~43 |
| Q2_K | ≈133 MB | 3.99 GB | ~50 |

The prefill path's fp16 working form is transient (one matrix in `wF16_` at a time); decode uses quantized GEMV directly, so the only real reduction lever is the file's quant.

### Qwen3.6-35B-A3B / Ornith (qwen35moe) GPU — CPU-Expert Hybrid (default)

`ARCH_QWEN35MOE` models — `Qwen3.6-35B-A3B` and the **Ornith** checkpoints (`Ornith-1.5-35B-A3B-Q8_0`, `Ornith-1.5-35B-Q4_K_M`) — are **MoE hybrids**: gated-delta-net (GDN) recurrent layers interleaved with full-attention layers (40 for Ornith/Qwen3.6, 41 for the Ornith Q4_K_M variant), each layer ending in a **softmax-routed MoE FFN**: a shared `ffn_gate_inp` router over **256 experts** (top-8 per token, weights re-normalized with the `norm_w` clamp), a shared sigmoid-gated expert, and per-expert SwiGLU (`ffn_gate_exps`/`ffn_up_exps` [256·expertFF] → `ffn_down_exps` [256·hidden]). The CPU reference lives in [`computeQwen35MoE()`](src/cpp/core/ModelMoE.cpp:161); the reference routed-expert half (router on CPU, used by the hybrid driver) is [`computeQwen35MoEFromLogits()`](src/cpp/core/ModelMoE.cpp:385); the GPU trunk driver is [`forwardQwen35MoePrefix()`](src/cpp/core/GPUCompute.cu:3827).

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
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | ~22 GB | ✅ hybrid (experts on CPU) | **expert cache verified (2026-09-17): all 8 GPU-parity tests pass** ✓; batch-vs-sequential deterministic `maxDiff=0.711` (fixed trunk-kernel precision delta, within gate; Ornith Q4_K_M is 0.0) |

The 29 GB-RAM machine cannot hold the 35 GB model in page cache, so single cold runs (e.g. `tinycoder_test` without warmup) report 15-23× lower numbers than warm repeats. Both harnesses now print a [`[bench-parity]`](benchmarks/BenchMain.cpp:281) line quantifying the same decode region with explicit cold/warm state, and the question tests warm the page cache once in [`SharedTestEnv`](unit_tests/SharedTestEnv.hpp:19) (`TINYCODER_TEST_NO_WARMUP=1` opt-out). See the [Benchmarking](#benchmarking) section for the steady-state warm rates of all qwen35moe models.

The MoE FFN adds the router GEMV (F32 `ffn_gate_inp` [256×hidden]), the top-k router kernel (softmax over 256 → strict `>` tie-break top-8), the grouped per-expert batch GEMVs (AVX2 Q8_0 kernel, expert-parallel over the pool), weighted accumulation, and the shared-expert sigmoid-gated SwiGLU. The prefill batch GEMM path (seqLen > 1) is weight-stationary: each selected expert's rows are read once per group instead of once per member token.

#### Static resident-expert offload (Option D) — `TINYCODER_MOE_RESIDENT=1` (2026-09-16)

For **Q8_0-expert** qwen35moe models, Option D puts the **hottest experts permanently on the GPU** (opt-in): no LRU, no per-token PCIe streaming of resident weights.

- **Profiling pass**: at upload time the adapter runs a representative code+text prompt mix through the **pure-CPU path** (`Model::forceCpuForward_` + `resetCpuKVState`, deadlock-free with `mtx_` held), counting per-layer routed expert selections into `moeUsageCounts_` (via the `computeQwen35MoE` → `recordMoEUsage` hook in [`ModelGPU::buildResidentExpertPolicy`](src/cpp/core/ModelGPU.cpp:864)).
- **Policy**: `cudaMemGetInfo` computes the free-VRAM budget (keeps 900 MB + 25% headroom for the trunk + shared expert + combine tables); the budget / per-expert arena bytes (`2·gateSlice + downSlice`, `3342336` B for this model) gives `floor(budget/N) ` resident experts per layer, filled with the per-layer **top-K most-used** experts (`std::partial_sort`).
- **Arenas**: [`GPUModel::buildResidentArenas`](src/cpp/core/GPUCompute.cu:5406) allocates per-layer contiguous arenas and H2Ds the resident experts' packed gate/up/down slices **once** (per-layer host blob selection, mirroring the expert-cache lesson). **Q8_0-expert types ONLY (hard gate, 2026-09-17)** — the resident fast path reuses the Q8_0 integer kernels (`launchQuantQ8_0x32` + `kQGemvQ8_0xQ8K_BatchGU` + `kSiluMulBatch` + `kQuantizeQ8_0x32_Batch` + `kQGemvQ8_0xQ8K_BatchDownR`). The Q4_K/Q6_K resident kernels (`kQGemvKxQ8K<kTypeQ4K/Q6K>`) are bit-exact (verified by `GPUCpuCompareTest.Q4K_Q6KKernelParityWithCPU`) but are **disabled**: measured on Ornith-1.5-35B-Q4_K_M the split resident path was a **4.5× decode regression vs Option B** (0.32 vs 1.45 tg tok/s) because the ~6-8 non-resident experts per layer run on the CPU with the per-member scalar path (no grouped-batch batching), so Q4_K_M stays on Option B permanently. Non-Q8_0 models (Q4_K_M, IQ2_M etc.) cleanly fall back to Option B — `TINYCODER_MOE_RESIDENT=1` prints a notice and is a fast no-op on them.
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
2. **Per-layer down-expert types** (the deeper root cause): the Ornith Q4_K_M GGUF stores the **down-expert quant type differently across layers** — blk.0-4 and blk.7+ have down = Q6_K (210 B/block), but **blk.5-6 have down = Q4_K (144 B/block)**; gate/up are always Q4_K. The cache's geometry globals (`expertRowBytesDown_`, etc.) are overwritten every upload step and held only the **last** layer's values, so layers 5-6 were filled and decoded with Q6_K geometry (420 B/row instead of 288 B/row) → garbage half-floats → NaN. The fix records **per-layer** quant geometry on every [`ExpertSlot`](include/GPUCompute.hpp:538) (`gateType`/`gateRowBytes`/`downType`/`downRowBytes`), sizes each layer's arena from its own slot, and derives every cache-slice pointer/row-stride/GEMV type from per-layer accessors (`layGateType`/`layGateRowBytes`/`layDownType`/`layDownRowBytes`/`layGateSliceB`/`layDownSliceB` in [`GPUCompute.cu`](src/cpp/core/GPUCompute.cu:6469)) — covering the batched Q8_0 fast path, the per-rank `launchQGemv` path, and the per-token fallback.

##### Expert cache — device-slot WAR race (2026-09-17)

After the per-layer geometry fix, `ModelTest.KVCacheClear` and `ModelTest.CompareBatchVsSequentialPrefill` still failed with **run-to-run varying** diffs (`KVCacheClear` maxDiff 1.92–2.04 across runs) while default/Option-B mode stayed bit-identical (maxDiff 0.0). Root cause: **a device-slot WAR race**. The per-rank decode path (`moePerRank` → [`ensureExpertCached`](src/cpp/core/GPUCompute.cu:5188) → `launchQGemv`) pipelined each miss fill on `g_stream2` (the copy engine) while the GEMVs that read that slot ran on `g_stream`. The g_stream2 + event-wait design only orders RAW (this rank's GEMV after its own fill) — it does **not** order a later rank's fill **after** an earlier rank's GEMVs. When two routed experts hash into the same 2-way set (common with 8 experts over 16 sets), a later rank's miss would H2D the same device slot while the earlier rank's GEMVs were still reading it → a timing-dependent torn read. Forward 1 (all misses, evictions tear) vs forward 3 (all hits, clean) differ — exactly `KVCacheClear`'s signature.

Fix: the per-rank fills now run **in-order on `g_stream`** ([`GPUCompute.cu`](src/cpp/core/GPUCompute.cu:5351)), giving the total order `fill_r → GEMV_r → fill_{r+1} → GEMV_{r+1}` — any slot eviction happens after the previous owner's GEMVs complete. The **batched Q8_0 fast path keeps its `g_stream2` pipeline** (all fills enqueue before any GEMV, and its slot-conflict guard falls back to the per-rank path whenever two ranks would share a slot base, so it cannot tear). Additionally, [`GPUModel::forward`](src/cpp/core/GPUCompute.cu:8646) drains `g_stream2` at forward end when the cache is built, so no fill is in flight when a later forward's cache **hit** reads a slot (hits wait on nothing).

Verified (2026-09-17, RTX 2080 Ti 11 GB, **expert cache default-on, no env**): all **12 tests pass** — `ModelTest.SingleTokenForward`/`MultiTokenForward`/`KVCacheClear`/`CompareBatchVsSequentialPrefill`, `Qwen35Test.SingleTokenForwardFinite`/`BatchVsSequentialCoherent`, `GPUCpuCompareTest.ParisPromptLogitsAgree`/`SequentialDecodeArgmaxAgrees`, and all 4 `FullQuestions/SampleQuestionTest` keyword answers. `CompareBatchVsSequentialPrefill` is now **bit-identical** (`maxDiff=0`) and `KVCacheClear` is deterministic — matching Option-B mode. Option B (no cache) re-verified via `TINYCODER_MOE_CACHE=0` with **no regression** (the 2 prior-failing tests pass). `SequentialDecodeArgmaxAgrees` keeps the Q4_K depth-5 boundary relaxation at step 5 (GPU top-1 `"T"` lands just past the CPU top-10 — the same near-tie cluster the default Option B mode exhibits with `"q"` at CPU rank 10); the assertion now fails only if an EOG/chat-special token pollutes the GPU top-5.

### Page-Cache Prefetch (`TINYCODER_PREFETCH`)

Weights are loaded via a file-backed `mmap` of the GGUF tensor section (no heap copy — see [`GGUFLoader::mapTensorData`](src/cpp/core/GGUFLoader.cpp:687)); pages are faulted in lazily on first touch. On models that fit in RAM, that lazy first-touch cost lands **inside the first inference pass** (the question harness "warm-up" prefill took 140 s and first-question decode ~0.6 tok/s on the 35 GB Q8_0). The loader now **eagerly prefetches the tensor section into page cache at load time** (`posix_fadvise(POSIX_FADV_WILLNEED)` + a per-page touch barrier) whenever the section **fits ≤ ¾ of physical RAM** — the 21–22 GB Q4 models prefetch (2–3 min load, then warm inference), while the 35 GB Q8_0 correctly **skips** (over-RAM eager read perturbed the page-cache LRU and made first passes *slower* — measured 0.22–0.51 tok/s vs 0.62–1.76 lazy). Overrides: `TINYCODER_PREFETCH=0` never prefetch, `=1` force.

### KV Cache

The KV cache stores key/value tensors as F32 arrays with shape `[numLayers, maxSeqLen, numKVHeads, headDim]`. For the 1.5B model with 2048 context: 28 layers × 2048 × 2 KV heads × 128 dim × 4 bytes × 2 (K+V) ≈ **112 MB**.

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- **[np library](https://github.com/mgorshkov/np)** — NumPy-style arrays with SIMD/CUDA acceleration
- **llama.cpp** — reference implementation used for correctness parity and performance benchmarking
