# TinyCoder OOM Fix — Full Context Handoff

> **Purpose:** This file captures the complete diagnosis and plan for fixing the OOM crash
> when running the 35B-A3B Q8_0 model through `tinycoder_test`. It was written to be loaded
> into a fresh session in the **tinycoder-inference** project (repo root:
> `/home/mike/git/tinycoder-inference`). All paths below are relative to that repo unless
> noted.

---

## 1. The Task (verbatim from the user)

> "Please fix `TINYCODER_MODEL_PATH=/data/models/ornith/Ornith-1.5-35B-A3B-Q8_0.gguf ./build/unit_tests/tinycoder_test`.
> Now we have OOM. The tests must pass, the model must work fine, inference speed must be 30 tok/sec."

Three requirements:
1. **Tests must pass** (68 tests / 8 suites in `unit_tests/`).
2. **The model must work fine** (35B-A3B Q8_0 MoE — Qwen3.6 architecture `qwen35moe`).
3. **Inference speed must be ~30 tok/s** on this host.

**Command to run (from `/home/mike/git/tinycoder-inference`, NOT the extension repo):**
```bash
TINYCODER_MODEL_PATH=/data/models/ornith/Ornith-1.5-35B-A3B-Q8_0.gguf ./build/unit_tests/tinycoder_test
```
Note: the extension repo (`/home/mike/git/tinycoder`) consumes this engine via FetchContent with
`BUILD_TESTS=OFF`; `./build/` there has no `unit_tests`. The real test binary lives in the
engine repo's build dir (`/home/mike/git/tinycoder-inference/build/unit_tests/tinycoder_test`).

---

## 2. Reference Host Facts (measured)

| Item | Value |
|---|---|
| CPU | Intel i7-4790K (Haswell), 4 physical cores / 8 logical CPUs |
| Cache | 8 MiB L3 |
| RAM | 31 GiB total, ~29 GiB available; 8 GiB swap |
| SIMD | AVX2 only (no AVX-512, no AMX) |
| GPU | NVIDIA RTX 2080 Ti 11 GB present, but THIS build is CPU-only (`ENABLE_CUDA=OFF` in this tree) |

Memory-bandwidth reality: DDR3-1600 dual channel ≈ ~18–20 GB/s achievable.

---

## 3. The Model File

- Path: `/data/models/ornith/Ornith-1.5-35B-A3B-Q8_0.gguf`
- Size: **36,903,139,520 bytes (~36.9 GB)**
- Qwen3.6 35B-A3B, architecture **`qwen35moe`** (MoE = Mixture of Experts)
- Quantization: **Q8_0** (32 weights/block, 34 bytes/block: fp16 `d` + 32×int8 `qs`)
- 40 layers, hidden 2048, vocab 248,320
- Layer types (llama.cpp `qwen35moe.cpp`):
  - **Recurrent** (gated delta net): `(i+1) % fullAttentionInterval != 0` → tensors `attn_qkv`,
    `attn_gate`, `ssm_conv1d`, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_dt.bias`, `ssm_norm`, `ssm_out`
  - **Full attention**: rest → `attn_q` (Q+gate fused), `attn_k`, `attn_v`, `attn_output`,
    `attn_q_norm`, `attn_k_norm`
- Routed experts: `ffn_gate_inp` (router), `ffn_gate_exps`/`ffn_up_exps` (or fused
  `ffn_gate_up_exps`) + `ffn_down_exps`; gated shared expert `ffn_*_shexp` + `ffn_gate_inp_shexp`
- MTP tensors: `nextn.*` (may be absent; loaded as optional)
- Loader log per layer (attention/recurrent + experts):
  - `ffn_gate_exps` ≈ **272 MB**, `ffn_up_exps` ≈ **272 MB**, `ffn_down_exps` ≈ **272 MB**
  - ≈ **1.1 GB per layer** for the layer's tensors → 40 layers is the bulk of the 36.9 GB file

---

## 4. Reproduced OOM (measured)

- `RSS` climbed monotonically while loading:
  - 17.5 GB @ 45 s
  - 30.8 GB @ 105 s (layer 16 of 40)
  - **Killed (exit 137, OOM killer)**

### Root cause (CONFIRMED)
`Model::load()` (`src/cpp/core/ModelLoad.cpp`) creates a local `GGUFLoader loader` that
**mmaps** the tensor data section (file-backed, reclaimable — see comment at
`include/GGUFLoader.hpp` line 168: *"The loader itself holds NO heap copy of the weights"*),
BUT then **heap-copies every tensor** into `QuantizedMatrix.data` via
`qm.data.assign(data, data + dataBytes)` at [`src/cpp/core/ModelLoad.cpp:349`](src/cpp/core/ModelLoad.cpp:349).
At end of `Model::load()` the loader goes out of scope (`ModelLoad.cpp:126` local), munmaps,
and the model keeps only the heap copies.

For a 36.9 GB model on a 31 GB machine this cannot fit:
- ~37 GB heap quantized copies + touched mmap pages (charged to RSS during the copy)
- ~0.9 GB F32 dequantized embeddings + FP16 copy + Q8_K copy + LM head Q8_K copy + KV cache
- → OOM at ~layer 16/40.

So the mmap design intent (see `GGUFLoader.hpp:168–178`) is **not honored** by `Model::load`.

### Memory consumers at load time, in detail
| Consumer | Size |
|---|---|
| Heap copies of all quantized tensors (~36.9 GB file) | ~37 GB |
| `dequantizedEmbeddings_` F32 copy of 248,320×2048 | ~2.0 GB |
| `dequantizedEmbeddings_.dataF16` (FP16 copy) | ~1.0 GB |
| `dequantizedEmbeddings_.dataQ8K` (Q8_K copy) | ~0.5 GB |
| `lmHead_` (separate output.weight, 515 MB quantized) + `lmHeadQ8K_` (Q8_K copy) + `lmHeadBounds_` (top-K table) | ~1 GB+ |
| KV cache: 40 layers × 2048 seq × 2 KV heads × 128 headDim × 4 B × 2 | ~160 MB |
| Qwen35 recurrent state (ssmConvBuf/ssmState/q35ConvState/q35GdnState) | small |
| **Total** | **far exceeds 31 GB → OOM ~layer 16** |

---

## 5. Key Code Facts (verified by reading the code)

### 5.1 `Model::load()` — `src/cpp/core/ModelLoad.cpp`
- Line 43: `bool Model::load(modelPath, outError, progressCb)`
- Line 82–87: `GGUFLoader metaLoader; metaLoader.loadMetadata(path)` → `config_`
- Line 92–96: caps `config_.maxSeqLen` to 2048
- Line 126: `GGUFLoader loader; loader.load(modelPath)` — **this is the mmap loader; it is local**
- Line 197: `if (!loadWeights(loader))` — the heap-copy hotspot
- Line 208: `buildDequantizedEmbeddings()`
- Line 216: `initKVCache()`
- Line 221: `loaded_ = true`; loader destroyed at scope exit → **mmap gone; only heap copies remain**

### 5.2 `loadQuantized` lambda — `src/cpp/core/ModelLoad.cpp:285–351`
- `info = loader.getTensorInfo(name)`; optional tensors return `QuantizedMatrix{}` quietly
- `data = loader.getTensor(name)` — **direct pointer into the mmap**
- Shape decoding: GGUF "numpy-reversed"; 2D → rows=shape[1], cols=shape[0]; 3D (experts) → rows *= shape[2]
- `dataBytes` computed from `ggmlBlockSize`/`ggmlTypeSize`
- **Line 349: `qm.data.assign(data, data + dataBytes)` — the OOM hot spot (heap copy)**

### 5.3 Qwen35MoE weight loading — `src/cpp/core/ModelLoad.cpp:636–712`
- `layers_[i].ffnGateInpMoe = loadQuantized("blk.N.ffn_gate_inp.weight")`
- Fused-or-separate experts: tries `ffn_gate_up_exps`, falls back to `ffn_gate_exps` + `ffn_up_exps`
- `ffnDownExpsMoe`, `ffnGateInpShexp`, `ffnGateShexp`, `ffnUpShexp`, `ffnDownShexp`
- Recurrent layer: `attnQKV`, `attnGate`, `ssmConv1d` (F32), `ssmABroadcast` (1D F32), `ssmAlphaQ`,
  `ssmBetaQ`, `ssmDtBiasFull` (1D F32), `ssmNorm` (1D F32), `ssmOut`
- Full-attn layer: `attnQ`, `attnK`, `attnV`, `attnO`, `attnQNorm`, `attnKNorm`
- MTP: `nextnEhProj`, `nextnEnorm`, `nextnHnorm`, `nextnSharedHeadNorm` (optional)

### 5.4 GGUFLoader mmap design — `include/GGUFLoader.hpp:168–179` (Linux)
- `const uint8_t *tensorData_` (section base, may be offset within mapped range)
- `void *mmapPtr_`, `uint64_t mmapLen_`, `int mmapFd_`
- `mapTensorData()` in `src/cpp/core/GGUFLoader.cpp` (~line 729): `open(O_RDONLY)`,
  page-align offset, `mmap(PROT_READ, MAP_PRIVATE)`
- Destructor → `unmapTensorData()` (munmap)
- `getTensor(name)` returns `tensorData_ + offset` — direct mmap pointer
- Non-Linux fallback: `readTensorData()` heap read
- **Key insight: if the loader outlives the copies, weights can stay file-backed; pages are
  faulted on demand and are reclaimable → a 36.9 GB model fits in 31 GB RAM.**

### 5.5 `QuantizedMatrix` — `include/Model.hpp:49–110`
- `AlignedVector<uint8_t> data` (64-byte aligned, `posix_memalign`)
- `AlignedVector<uint8_t> prepackedData` (Q2_K prepack only)
- `rows`, `cols`, `type`
- `empty()` = `data.empty()`
- Methods: `matMulVec(x)`, `matMulVec(x,out)`, `matMulVecRows(x,rowStart,numRows[,out])`,
  `matMulVecFusedGateUp(other,x,gateOut,upOut,applySwish)`

### 5.6 Consumers (all read via `data.data()`)
- `src/cpp/core/QuantizedMatrix.cpp` — `matMulVec`, `matMulVecRows` (line 297), dispatch
- `src/cpp/core/ModelMoE.cpp` — `computeQwen35MoEFromLogits` (line 328) and `computeQwen35MoE`
  (line 254 area): per-expert reads via `matMulVecRows(x, expertIdx*expertFF, expertFF, buf)` —
  **only active experts' rows are read per token**
- `src/cpp/core/ModelForward.cpp` — forward pass reads `w.attn*`, `w.ffn*`

### 5.7 Per-token weight traffic (why 30 tok/s is achievable despite 36.9 GB model)
- MoE: only **~3–8 of 128 experts** are active per token (top-k routing)
- Per token the engine reads only: active expert rows + shared expert + attn/SSM weights
- So per-token weight stream is **~1–2 GB (not 36.9 GB)** — bandwidth ~18 GB/s ⇒ ~30 tok/s is
  a plausible hard floor; llama.cpp CPU reaches ~29 tok/s on this host (per the plan doc).

---

## 6. Q8_0 Compute-Path Facts (relevant to the 30 tok/s requirement)

- **No AVX2 batch SIMD kernel for Q8_0.** `matMulVecBatchSIMD` in
  `src/cpp/core/QuantizedMatrix.cpp` handles ONLY: Q6_K, Q5_K, Q4_K, IQ4_XS, IQ4_NL, Q3_K, Q8_K.
- Q8_0 falls through to the **generic fused scalar path**: `GGMLDequantize::matMulVecFused`
  using `dequantizeQ8_0Block` / `dotProductQ8_0` (`include/GGMLDequantize.hpp` line 244 /
  line 1653) — scalar dequant-dot ≈ **0.3 GFLOP/s** vs Q8_K int8 kernels ≈ **36 GFLOP/s**.
- The Q8_K integer dot path (`matMulVecFusedQ8K`, `matMulVecBatchQ8K`) quantizes activations to
  Q8_K once and reuses across rows with `_mm256_maddubs_epi16`.
- `supportsQ8KDot(type)` at `include/GGMLDequantize.hpp:1901` (gated by `TINYCODER_DISABLE_Q8K_DOT`)
  lists which types use the Q8_K int8 path — need to verify whether Q8_0 is in the CPU list
  (grep earlier showed Q8_0 in a list, but it may have been the GPU offload list).
- **Risk:** even after the memory fix, per-token Q8_0 scalar dequant-dot over the active
  weights (attention + MoE experts) might be far slower than 30 tok/s. Mitigation options:
  1. Add a Q8_0 AVX2 batch kernel to `matMulVecBatchSIMD` (or a fused Q8_0 dequant-dot SIMD path).
  2. Route Q8_0 through the Q8_K integer-dot path when available.
  3. First verify with `TINYCODER_PROFILE=1` where the time actually goes; MoE sparsity may
     already make the scalar path fast enough for the active-expert subset.

---

## 7. Env Vars / Knobs / Build

```bash
TINYCODER_MODEL_PATH   # model path (the test suite reads it; default = qwen2.5-coder-1.5b q2_k)
TINYCODER_THREADS      # overrides recommendedThreadCount() (logical CPUs = 8 here)
TINYCODER_AFFINITY     # thread pinning
TINYCODER_GPU / TINYCODER_NGL  # GPU offload (this build: CPU-only)
TINYCODER_FORCE_SCALAR # force scalar kernels
TINYCODER_HUGE_PAGES   # MADV_HUGEPAGE hints (see include/MemHints.hpp:adviseHugePages)
TINYCODER_PROFILE      # per-phase timing
TINYCODER_DISABLE_Q8K_DOT
TINYCODER_MOE_VERIFY
TINYCODER_DUMP_TOKEN
```

Build (engine repo root):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
Existing binary: `build/unit_tests/tinycoder_test`. CMake options seen: `TINYCODER_AFFINITY`,
`TINYCODER_FFN_STEAL`, `TINYCODER_COOP_QUANTIZE`, `TINYCODER_STEAL_CHUNK`, `BUILD_TESTS`.

---

## 8. Test Suite Facts

- 68 tests, 8 suites. `unit_tests/Main.cpp` initializes `ThreadPool::instance()` with
  `recommendedThreadCount()`, reads `TINYCODER_MODEL_PATH`, registers `SharedTestEnv`
  (`unit_tests/SharedTestEnv.hpp`) which loads the model once globally (gtest environment).
- tok/s is **printed, not asserted** (e.g. `AnswersQuestion` at `unit_tests/ModelTest.cpp:299`,
  `GenerateTokens` at line 260, `Qwen35Test` group gated to qwen35/qwen35moe arch).
- So 30 tok/s is a user-perf gate, not a test assertion — passing tests ≠ 30 tok/s.

---

## 9. Planned Fix (Option A — mmap-backed zero-copy weights)

**Goal:** Stop heap-copying the 36.9 GB of quantized weights; keep them file-backed in the mmap
so (a) load doesn't OOM and (b) RSS stays ≈ working set (page-cache-reclaimable).

1. **Make the loader persistent:** add a member (e.g. `std::unique_ptr<GGUFLoader>
   mmapLoader_` or keep the mmap state) to `Model` (private section, `include/Model.hpp`,
   after line 578 area). Create it in `Model::load()` (replacing the local `loader` at
   `ModelLoad.cpp:126`) and keep it alive for the lifetime of the Model.
2. **Add an external-buffer mode to `QuantizedMatrix`:** a `const uint8_t *externalData`
   pointer (plus owning-buffer bool) alongside `AlignedVector<uint8_t> data`. Shim
   `data()`/`size()`/`empty()` so every existing kernel call site keeps working:
   - If `externalData` set → `data()` returns it, `size()` returns `dataBytes`, `empty()` false.
   - `prepackedData` (Q2_K) unchanged (still heap, only for Q2_K).
   - `loadQuantized` (`ModelLoad.cpp:285–351`): instead of `qm.data.assign(...)`, record
     `qm.setExternal(info->offset-within-tensor-data, dataBytes)` — i.e. just the pointer into
     the mmap + length. **No copy.**
   - `matMulVec*` in `src/cpp/core/QuantizedMatrix.cpp` must read through `data()` (already do
     via `data.data()` → change to a `data()` accessor or make `data` behave transparently).
3. **Keep small/F32 loads as-is:** `loadF32_1D` (norms, ssm params — KB-scale), embeddings
   dequantization, LM-head prepacks (`buildDequantizedEmbeddings()` at `ModelLoad.cpp:858`),
   KV cache. These total ~4–5 GB and are fine.
4. **AlignedVector::assign on external data** must be avoided — no heap copy.
5. Be careful with:
   - `QuantizedMatrix` copy/move semantics (it's returned by value from `loadQuantized` and
     stored in `std::vector<LayerWeights> layers_` — if the struct gets copied, the external
     pointer must stay valid; it will, as long as the loader outlives the Model).
   - `empty()` semantics used in `loadWeights` (`if (layers_[i].ffnGateUpExpsMoe.empty())`).
   - The loader must NOT be destroyed before the Model (it's the source of the pointers).

### Alternative considered (NOT chosen)
- Skipping/pinning weights: not viable — tests need the full model.
- Building the whole engine with CUDA: the GPU (11 GB) can't hold a 36.9 GB model either.

---

## 10. Performance Plan for 30 tok/s (after the memory fix)

1. Rebuild and confirm load completes without OOM (watch RSS — should stay near working set,
   not 37 GB).
2. Run with `TINYCODER_PROFILE=1` and count thread scaling:
   - 8 threads (current default) vs 4 (llama.cpp parity per `plans/generation_optimizations.md`).
3. If Q8_0 scalar dequant-dot dominates per-token time: add a Q8_0 AVX2 fused
   dequantize-dot batch kernel (`matMulVecBatchSIMD`) or route through the Q8_K int8 dot path.
4. MoE expert sparsity should keep per-token bytes read low; measure actual per-token traffic.

Reference: `plans/generation_optimizations.md` — measured llama.cpp CPU ≈ 29.4 tok/s (4 threads)
vs TinyCoder ≈ 26.3–26.5 tok/s (8 threads) on the 1.5B Q2_K model; host DDR3 ~18 GB/s.

---

## 11. File/Line Quick Reference

| File | Lines | What |
|---|---|---|
| `src/cpp/core/ModelLoad.cpp` | 43–225 | `Model::load()` — local mmap loader, copy path |
| `src/cpp/core/ModelLoad.cpp` | 285–351 | `loadQuantized` — `qm.data.assign` at **349** (OOM hot spot) |
| `src/cpp/core/ModelLoad.cpp` | 636–712 | qwen35moe weight loading |
| `src/cpp/core/ModelLoad.cpp` | 858–1035 | `buildDequantizedEmbeddings()` (LM head prepacks) |
| `src/cpp/core/ModelLoad.cpp` | 793+ | `initKVCache()` |
| `include/GGUFLoader.hpp` | 82–185 | GGUFLoader class; mmap state 168–179 |
| `src/cpp/core/GGUFLoader.cpp` | ~729 | `mapTensorData()` (mmap) |
| `include/Model.hpp` | 49–110 | `QuantizedMatrix` |
| `include/Model.hpp` | 553+ | private members (layers_, lmHead_, dequantizedEmbeddings_, kvCache_) |
| `include/AlignedVector.hpp` | 46–164 | aligned heap buffer; `assign()` allocates+memcpy |
| `src/cpp/core/QuantizedMatrix.cpp` | 93–130 | `matMulVecBatchSIMD` — **no Q8_0 case** |
| `src/cpp/core/QuantizedMatrix.cpp` | 297+ | `matMulVecRows` (MoE expert reads) |
| `src/cpp/core/ModelMoE.cpp` | 254–288 / 328–494 | MoE compute (expert row reads per token) |
| `include/GGMLDequantize.hpp` | 244 / 1653 / 1901 / ~2134 | Q8_0 dequant, Q8_0 dot, supportsQ8KDot, Q8_K fused |
| `unit_tests/Main.cpp` | 45–77 | main; reads TINYCODER_MODEL_PATH |
| `unit_tests/SharedTestEnv.hpp` | 26–70 | global model load |
| `unit_tests/ModelTest.cpp` | 260 / 299 | GenerateTokens / AnswersQuestion (tok/s printed, not asserted) |
| `plans/generation_optimizations.md` | — | perf plan; 30–35 tok/s target; 0.67 GB/token stream notes |

---

## 12. Todo State

```
[x] Explore tinycoder-inference repo structure and unit test setup
[x] Reproduce the OOM with TINYCODER_MODEL_PATH=... ./build/unit_tests/tinycoder_test
[x] Diagnose the memory over-allocation root cause      (root cause CONFIRMED: heap copy at ModelLoad.cpp:349)
[ ] Implement the fix (mmap-backed zero-copy weights / QuantizedMatrix external buffer)
[ ] Rebuild and verify tests pass with model working at 30 tok/sec
[ ] Verify final state end-to-end