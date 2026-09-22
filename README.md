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
| `TINYCODER_FUSE_QKV` | `0` opts OUT of the fused Q+K+V decode kernel (default **on**, 2026-10-10). One launch computes attn_q (Q2_K) + attn_k (Q2_K) + attn_v (Q4_K float dequant-dot) against the same fp32 activation and folds the three bias adds into the epilogues (Q/K bit-exact, V in the fp32-x domain = closer to the CPU reference) — replaces fused-QK + Q8_K quantize + v GEMV + 3 bias launches (7 launches → 1). Measured QKV stage 1.45 → 0.75 ms/token, tg128@pp512 **178.8 → 191.7 tok/s (+7.8%)**, greedy streams byte-identical on 4/4 prompts (eager + graph) |
| `TINYCODER_FUSE_DOWN_RESID` | `0` restores GEMV + separate `kAddResidual` for the FFN-down (default **on**, 2026-10-10): when the down takes the Q3_K mmvq path, the residual add `hidden[row] = hidden[row] + out[row]` is folded bit-exactly into the mmvq epilogue — one launch fewer per layer. Measured **+0.7%** tg128@pp512 (4/4 interleaved wins), streams byte-identical on 4/4 prompts |
| `TINYCODER_MMVQ_WIDE` | `0` restores the `std::memcpy`-from-`uint8_t*` load idiom in the dp4a mmvq-family kernels (default **on**, 2026-10-10): nvcc cannot prove 4-B alignment through a char pointer, so every int read was split into 4 byte LDGs — 32 of the ~70 static loop instructions — saturating the LSU queue (LG-throttle 57–66% of stall cycles) and costing ~40% issue rate vs llama.cpp's SASS for the same algorithm. The wide idiom (llama.cpp's int-cast deref for the 84-B-stride Q2_K GLU; int deref for the 4-aligned y reads and 2×`LDG.E.U16` for the 110-B-stride Q3_K weight ints, which are 2-mod-4 on odd rows) halves the load-instruction count. Measured tg128@pp512 interleaved A/B (5 rounds): **211.5–212.1 → 249.9–251.7 tok/s (+18.0–19.0%), 5/5 wins**, 4/4 prompt stream parity. **Extended 2026-10-11 to the Q6_K LM head** (`kQGemvQ6KxQ8K_4xW`: 2×`LDG.E.U16` weight ints — the 210-B blocks are 2-mod-4 on odd blocks — and int-deref activation loads, replacing the per-int 4×byte-LDG `std::memcpy` split): LM-head stage **0.85 → 0.42 ms/token**, decode **281.0 → 297.4 tok/s (+5.8%)**, interleaved 5/5, full suite 57/57 parity |
| `TINYCODER_PINNED_D2H` | `0` restores the direct logits D2H (default **on**, 2026-10-10): the 0.6 MB logits copy into the caller's pageable buffer ran the driver's staged path (~4 GB/s); now device → pinned staging + host memcpy (~2× faster), applied to both the eager and graph-replay D2H. Measured **+0.2%** tg128@pp512, streams byte-identical on 4/4 prompts |
| `TINYCODER_FUSE_GU_Q2K` | Q2_K gate+up decode mode (default **`2`**, 2026-10-10): **`2`** = the llama.cpp-style fused GLU kernel `kQGemvQ2KxQ81_GLU` — ONE launch quantizes the fp32 activation to Q8_1, computes both Q2_K weight dots with the shared Q8_1 operand (dp4a, one row per block, 2 warps K-split) and writes `silu(gate)*up` straight to the down-GEMV input (replaces gate GEMV + up GEMV + `kSiluMul`, 3 launches → 2). Replicates the structure measured on llama.cpp (plan §9.20: one fused launch at 237 GB/s effective vs our two launches at 124 GB/s — the win is the second weight stream per thread, not a different inner loop). Interleaved A/B tg128@pp512 (5 rounds): **193.2–194.0 → 210.9–211.8 tok/s (+9.1–9.5%), 5/5 wins**; greedy streams match on 4/4 prompts and the final top-10 logits are digit-identical to the separate-launch path. **`1`** = the legacy float-dequant fused path (measured tg8 regression, kept for A/B). **`0`** = separate per-matrix launches (the pre-2026-10-10 default). The kernel requires 1536-wide Q2_K rows (504-B row stride); other shapes fall back to the float path. The IQ2x/IQ3xxs fused paths are unaffected by this switch |
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
| `TINYCODER_Q3K_4XW` | `1` opts INTO a 4-rows-per-warp Q3_K decode GEMV (`kQGemvQ3KxW4`, the Q2_K-named 4xW geometry applied to the ffn-down), default **off**, 2026-10-06 — measured **~-8%** tg8 (baseline ~163 vs ~150 tok/s, interleaved, reps 8); the Q3_K high-bit-mask fold adds ~1 ALU op per weight byte on an issue/latency-bound matrix (SM ~96% busy, DRAM ~20%) so the extra work is not hidden. REFUTES the "Q3_K down is bandwidth-limited by its 1-row geometry" hypothesis; kept opt-in as an A/B record. See [`plans/outperform_llama_architecture.md`](plans/outperform_llama_architecture.md) §7.2 |
| `TINYCODER_Q3K_BSY` | `W` — warps per block for the opt-in Q3_K 4xW kernel (default `8`; `1`/`2`/`4` A/B), 2026-10-06 — no block shape recovers the ~-8% (all ~154-156 tok/s) |
| `TINYCODER_EXACT_COLS` | `0` disables the **default-on** EXACT-columns GEMV specialization (2026-10-06): when `blocksPerRow*256 == cols` — true for every default-model matrix (H=1536, I=8960, KV=256) — the per-element `(cidx < cols)` guard is provably dead, so the `EXACT=true` kernel drops it. **Bit-identical** output, ~2 fewer SASS ops/weight; measured **+2.5% tg8** (165.9 → 170.0, interleaved + ABBA, reps 8) with 57/57 parity. Non-multiple-of-256 models keep the guarded path. See [`plans/outperform_llama_architecture.md`](plans/outperform_llama_architecture.md) §7.5 |
| `TINYCODER_ATTN_SPLIT` | `0` disables split-KV decode attention (default **on**, 2026-10-01). Dense-arch, `seqLen==1` only. Parallelizes the O(context) attention scan across `nHeads × ceil(maxSeqLen/chunk)` warps (was `nHeads`) — held decode flat at ~142–145 tok/s to 1K context vs the eager path's collapse (pp1024: 142 vs 38 tok/s) |
| `TINYCODER_ATTN_SPLIT_CHUNK` | KV-window size for split-KV decode attention (default `16`, 2026-10-10 — was `64`). Smaller = more warps (better occupancy) but more combine traffic. **Measured (pp512/tg32, reps 3): 128→130.8, 64→153.9, 32→168.1, **16→173.1**, 12→172.6, 8→170.5, 4→159.4, 2→128.1, 1→92.3 tok/s — 16 rows is the peak (12 heads × 32 chunks = 384 warps on 68 SMs) and wins at every context length (pp16 171.7→175.6, pp128 155.7→174.7, pp512 154.4→173.1, pp1024 153.2→170.9), making decode essentially context-flat. At the llama-bench pp512/tg128 protocol this is **153.5 → 173.2 tok/s (+12.8%)**, parity byte-identical; `kv/attn/rope` 1.54 → 0.81 ms/token (−47%) |
| `TINYCODER_ATTN_SUBWARP` | Sub-warps cooperating per split-KV (head, chunk): `2` (default, 2026-10-10) splits each chunk into two contiguous 8-position sub-spans merged in shared memory with the standard softmax rescale — 2x the active warps at unchanged combine cost (8/8 interleaved pair wins, +0.35% tg128@pp512); `4` = one 4-position load-batch group per warp (measured -0.2%, slower — merge overhead eats the chain shortening); `0`/`1` = legacy one-warp-per-(head, chunk) layout |
| `TINYCODER_ATTN_COMBINE4` | `0` restores the 32-thread single-warp split-KV attention combine (default **on**, 2026-10-10). The combine ran 12 blocks × 32 threads (384 threads on 68 SMs), each thread serially folding 128 flash partials — 14.5 µs at DRAM 0.4%, pure latency; the W4 kernel (`kAttnCombinePartialW4`) launches 128 threads, each warp folds every 4th chunk into its own (m, l, acc) partial and the four merge flash-style in shared memory (fold order changes, same float tolerance class as split-KV itself). Measured (plan §9.24): A/B interleaved 5 rounds **+1.05–2.42%, 5/5 wins** (263.4–264.4 vs 257.8–261.3 tok/s A-side) as the last item of the attention-combine/attnO/rmsnorm bundle; 4/4 prompt stream parity |
| `TINYCODER_ATTN_COMBINE_W8` | `0` pins the 4-warp combine layout (default **on**, 2026-10-10): with numChunks ≥ 64 the combine launches 256-thread blocks (NW=8 warps fold every 8th chunk, 3072 threads across the 12 head-blocks vs 1536), halving the per-warp serial fold again; NW=4 stays for short contexts. Measured (plan §9.27): interleaved 5 rounds **4/5 wins, +0.3–1.5%** (277.9–281.1 vs 276.2–278.1 tok/s); 4/4 prompt stream parity |
| `TINYCODER_FUSE_Q81_PROD` | `0` restores the separate 48-block `kQuantizeQ8_1` launches for the attnO and GLU inputs (default **on**, 2026-10-10). The producers already hold the whole row: [`kRMSNormRowQ81`](src/cpp/core/GPUCompute.cu:250) quantizes per-warp (warp w covers Q8_1 block w from element t and block 32+w from element t+1024) and the combine warp-0 epilogue quantizes its NV per-head blocks — the quantize math is verbatim `kQuantizeQ8_1` over the same floats, so the Q8_1 bytes (and GEMV outputs) are bit-identical. All folds share one stream-ordered q8k_ buffer per layer. Measured (plan §9.26): A/B interleaved 5 rounds **+1.4–2.4%, 5/5 wins** (267.0–269.2 vs 262.7–263.9 tok/s); 4/4 prompt stream parity |
| `TINYCODER_QKV_DP4A` | `0` restores the byte-load float-dequant Q/K bodies in the fused QKV kernel (default **on**, 2026-10-10). The Q/K loops now run llama's `vec_dot_q2_K_q8_1` structure (`__dp4a`, int-cast weight derefs — the 504-B/84-B strides are 4-aligned) against the Q8_1 activation the attention rms_norm emits, replacing the §9.23 byte-load disease at its purest (8+ byte LDGs per block per lane); V keeps the Q4_K float path on fp32. Measured (plan §9.26): A/B interleaved 5 rounds **+4.0–4.1%, 5/5 wins** (276.9–277.5 vs 265.4–266.7 tok/s); 4/4 prompt stream parity |
| `TINYCODER_KV16` | `1` opts INTO an fp16 KV cache on the dense qwen2 path (default **off**, 2026-10-01 — measured +0.4% tg64 at pp1024 but -6% prefill; the split-KV kernel already removed attention as the decode bottleneck, so halving its bytes no longer moves the wall). Halves KV VRAM and the O(context) attention/RoPE DRAM traffic |
| `TINYCODER_PREFILL_TWIN_MARGIN_MB` | Overrides the residual-VRAM safety margin reserved away from the fp16 prefill-twin pass (default: geometry-derived — `maxSeqLen*vocab*4` logits + `maxSeqLen*hidden*4*24` activations + the largest matrix's fp16 form + 256 MB). Twins are allocated only from the residual free VRAM AFTER every quantized weight/KV/embed buffer is resident |
| `TINYCODER_IQ_DP4A` | `0` restores the scalar grid-lookup dot in the IQ2_S/IQ3_XXS/IQ3_S Q8_K GEMVs (default **on**, 2026-10-11). The grid dot now uses `__dp4a` + the branchless XOR-subtract sign application from llama's `vec_dot_iq*_q8_1` (`__vcmpne4`/`__vsub4`), replacing 8 scalar IMAD + 8 sign selects per (ib32,l) with 2 `dp4a`; bit-identical integer sum. Measured: interleaved A/B **+5.6% (1.5B iq3_xxs, 3/3)** and **+7.6% (7B iq3_xxs, 2/2)**, greedy streams byte-identical |
| `TINYCODER_IQ2XS_INT` | `0` restores the float dequant-dot path for IQ2_XS (GGML type 17) matrices (default **on**, 2026-10-11). IQ2_XS previously ran only the float kernels `kQGemvFusedGU_IQ2XS` (869 µs/layer at 25% occupancy) and `kQGemv` (499 µs at 83% LG-throttle); it now uses `kQGemvQ8KFusedGU<kTypeIQ2XS>`/`kQGemvQ8K<kTypeIQ2XS>` (ls=2·nib+1, final scale 0.125). Measured: interleaved A/B **+93% (7B iq2_s, 2/2)**, greedy stream identical to the float path |
| `TINYCODER_IQ_GRID_SMEM` | `0` keeps the IQ codebook grid (`c_iq2s_grid`/`c_iq2xs_grid`/`c_iq3s_grid`/`c_iq3xxs_grid`, 1–8 KB) in global/L1 (default **on**, 2026-10-11). The IQ decode GEMVs are LG-throttle bound (42–53% of stall cycles) on the divergent codebook lookups; staging the grid in shared memory moves those reads to the MIO path. Measured: interleaved A/B **+13% (7B iq2_s, 2/2)**, **+4.3% (7B iq3_xxs, 2/2)**, +0.5% (1.5B iq3_xxs); greedy streams byte-identical |

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

Columns follow the benchmark defaults: `tinycoder_bench --n-prompts 512 --n-gen 128` (**pp512 / tg128**) and `llama-bench -p 512 -n 128` (same shapes), warm mean-of-reps, RTX 2080 Ti.

| Model | File size | GPU mode TinyCoder | **TinyCoder pp512** | **TinyCoder tg128** | **llama.cpp pp512** | **llama.cpp tg128** |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B `q2_k` | 0.7 GB | full offload | **9,838 tok/s** | **304.5 tok/s** | 7,523 tok/s | 285.7 tok/s |
| Qwen2.5-Coder-1.5B `iq3_xxs` | 0.6 GB | full offload | 9,819 tok/s | **194.0 tok/s** | 10,120 tok/s | 244.7 tok/s |
| Qwen2.5-Coder-7B `iq2_s` | 2.4 GB | full offload | 2,419 tok/s | **77.4 tok/s** | 2,447 tok/s | 90.7 tok/s |
| Qwen2.5-Coder-7B `iq3_xxs` | 2.9 GB | full offload | 2,463 tok/s | **66.2 tok/s** | 2,887 tok/s | 86.4 tok/s |

> **IQ decode update (2026-10-12):** the IQ models' tied/separate **Q5_K LM head** was
> the single largest decode stage (2.4–5.8 ms of a 7–17 ms token) because it ran the
> 8-active-lane generic [`kQGemvKxQ8K<kTypeQ5K>`](src/cpp/core/GPUCompute.cu:3905) GEMV.
> It now runs a **4-rows-per-warp, `__dp4a`-vectorized, aligned-load** kernel
> ([`kQGemvQ5KxQ8K_4xW`](src/cpp/core/GPUCompute.cu:4175)) that is **bit-exact** with the
> generic path: the LM head dropped to 0.49 ms (1.5B) / 1.13 ms (7B), lifting IQ `tg128`
> by **+39 % / +32 % / +37 %** (1.5B `iq3_xxs` / 7B `iq3_xxs` / 7B `iq2_s`). Default on;
> `TINYCODER_Q5K_LM_4XW=0` restores the generic kernel.

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
