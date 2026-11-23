# Engine defects from the 2026-11-20 RTX 3090 recheck — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the three engine-side defects in the rented 3090 report reproducible in this tree and fix them: int8 prefill refuses Q6_K shapes it should run, Q6_K token embeddings silently disable GPU offload, and Gemma 4 cannot load because global layers share V.

**Architecture:** Two of the three are "a supported quant type is missing from one dispatch table" (int8 GEMM shapes, embedding gather) and are fixed by adding the type plus a parity test that would have caught it. The third is real geometry: Gemma 4 mixes per-layer KV head counts and shares V on its global layers, so the loader must read the per-layer array and alias the shared tensor instead of requiring it.

**Tech Stack:** C++20, CUDA (sm_75 reference box, sm_86 target), CMake, GoogleTest, `tools/` probes, `scripts/text_gate.sh` as the model-level gate.

**Spec:** the 2026-11-20 RTX 3090 recheck report from the eva-brain stand
(received 2026-11-22; **not published in this repository**). Sections 4 and 6
are the defect list; §5.5 is the affected measurements. The numbers it
reported are transcribed in `benchmarks/gpu/BENCHMARKS-3090.md` §1.1, which is
the in-repo record.

## Global Constraints

- Reference box: RTX 2080 Ti, sm_75, build dir `build-1.0`. Build the core explicitly: `cmake --build build-1.0 --target tinycoder_core <tools> -j 20`; check the exit code, never a grep window.
- Every change to a compute path ends with the gates in `AGENTS.md`: `tinycoder_test` (0 failed), `exl3_gpu_parity` (0 `[FAIL]`, 10 `[SKIP]`), `int8_gemm_parity` and `gemv_decode_parity` on `Ornith-1.5-9B-Q4_K_M`, the 27B logits probe (`id=198`, 19.3090 ±2e-3), and `./scripts/text_gate.sh` (all rows pass).
- A published benchmark row must have a passing text gate on the same tree and checkpoint (`benchmarks/gpu/BENCHMARKS-3090.md` §2). No row may be published from a run with a non-zero fallback counter.
- `-lineinfo` stays off (default). New `.cu` files compile for 75;80;86 unless a task says otherwise.
- Commit messages: short, change-oriented, no measured numbers (see `AGENTS.md` → Versioning).

## Review Focus

- **`m` below 8.** `int8Gemm` documents `seqLen >= 8`; a probe that passes `m=4` must get `false` and a clean log line, not a launch attempt.
- **Tied embeddings.** When `token_embd` is the LM head (`lmHeadTied_`), the Q6_K embedding fix must also cover the GPU head path, or the model loads on GPU and answers garbage.
- **Aliased V is weights, not state.** Reusing the previous layer's `attn_v` must not reuse its KV cache slots; the shared tensor is a weight matrix only.
- **A refused prompt stays refused.** Nothing in these tasks may turn the engine's window refusal into truncation.
- **Mixed per-layer KV heads.** `head_count_kv = [8,8,8,8,8,2,...]` must not collapse to `[0]`; a layer whose KV head count differs must reshape K/V with its own count.

---

### Task 1: A shape probe for `int8Gemm`, and the Q6_K shapes the 3090 report says fail

The report quotes `[int8] launch failed (type=14 k=2048 n=8192 m=495 x=128 tiles=256 blocks=82)` ~3000 times per run at `6c578a5`. Type 14 is Q6_K and that branch was added in the sm_86 commit, so the first job is to prove the current tree runs these shapes at all — no checkpoint we own has `k=2048` Q6_K, so the probe must build the tensor itself.

**Files:**
- Create: `tools/int8_gemm_shape_probe.cpp`
- Modify: `unit_tests/CMakeLists.txt` (add the target next to `int8_gemm_parity`, lines ~140-157)
- Test: the tool is its own gate (exit 0 / exit 1)

**Interfaces:**
- Produces: `int main(int argc, char **argv)` accepting `--shape k,n,m,type` repeated, `--tolerance <rel>`; exit 0 when every shape verified inside tolerance, 1 otherwise, 2 on bad usage.
- Consumes: `tinycoder::gpu::int8Gemm(int type, const void *w, const float *x, float *out, uint32_t seqLen, uint32_t rows, uint32_t cols, uint32_t blocksPerRow, void *stream)` and `GGMLDequantize::dequantizeBlock(type, block, out, n)`.

- [ ] **Step 1: Write the tool with a failing-by-design shape list**

```cpp
// Q6_K, the three shapes the 3090 report logged (k, n, m), plus m=8 as the floor.
const Shape shapes[] = {{2048, 8192, 495}, {6144, 5120, 512}, {2048, 512, 495}, {6144, 5120, 8}};
// Weights: rows x cols Q6_K blocks (210 B / 256 values), small deterministic
// values (|w| <= 0.05, i.e. the magnitude range that exposed the fp16 flush).
// Activation: random int8-valued floats so the Q8_1 quantisation is exact.
// Reference: fp64 dot over GGMLDequantize::dequantizeBlock rows.
```

- [ ] **Step 2: Run it on the current tree**

Run: `./build-1.0/unit_tests/int8_gemm_shape_probe --shape 2048,8192,495,14 --shape 6144,5120,512,14`
Expected: PASS for both. If `int8Gemm` returns false, the tool prints `refused (type=14 k=... n=... m=...)`; if it returns true with `relRMS > 2e-3`, it prints `FAIL`. Record whichever happens — that decides Step 3.

- [ ] **Step 3: Fix only if Step 2 fails**

If the launch is refused, add the missing type/tile arm in `src/cpp/core/cuda/gemm/GemmDispatch.cu` (`launchSk`'s type switch, the same place the Q6_K stream-k branch lives) and extend `int8GemmAvailable()`. If the values are wrong, the defect is in the kernel, not the dispatch: dump the first mismatching block and compare against `GGMLDequantize` before touching the kernel.

- [ ] **Step 4: Verify the shapes and the floor**

Run: `./build-1.0/unit_tests/int8_gemm_shape_probe --shape 2048,8192,495,14 --shape 6144,5120,512,14 --shape 2048,512,495,14 --shape 6144,5120,7,14`
Expected: three PASS lines, then `refused (m=7 < 8)` and exit 1 — the tool treats a refusal of `m<8` as expected, so it prints `ok (m<8 refused by design)` and exits 0.

- [ ] **Step 5: Run the existing int8 gates and commit**

Run: `./build-1.0/unit_tests/int8_gemm_parity /data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf --seqlen 1,8,64,128 --max-shapes 4`
Expected: `12 verified, 4 skipped, 0 outside tolerance`.

```bash
git add tools/int8_gemm_shape_probe.cpp unit_tests/CMakeLists.txt src/cpp/core/cuda/gemm/GemmDispatch.cu
git commit -m "int8 GEMM: probe and fix the Q6_K prefill shapes the 3090 stand logged"
```

### Task 2: Q6_K token embeddings must not silently drop the model to CPU

The report's defect 2: `bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF` Q2_K stores `token_embd` as Q6_K; `supportedEmbeddingType` (`src/cpp/core/ModelGPU.cpp:141-147`) has no Q6_K, so the engine logs `GPU offload inactive (CPU path): token embedding quant type 14 not implemented by GPU kernels` and runs at ~20 tok/s while the server reports success.

**Files:**
- Modify: `src/cpp/core/ModelGPU.cpp:141-147` (`supportedEmbeddingType`)
- Modify: the GPU embedding gather kernel that switches on the quant type (locate with `grep -rn "case GGML_TYPE_Q5_K" src/cpp/core/cuda/` — the same switch needs a Q6_K arm decoding 210-byte blocks, mirroring `GGMLDequantize::dequantizeQ6KBlock`)
- Test: `unit_tests/GPUCpuCompareTest.cpp` (new test case)

**Interfaces:**
- Consumes: the existing GPU embedding lookup used by `ModelGPU` for `quantizedEmbeddings_`; its row gather must accept `GGML_TYPE_Q6_K`.
- Produces: `supportedEmbeddingType` returns true for Q6_K; a gtest `GPUEmbeddingQ6KMatchesCpu` comparing GPU-gathered rows against `GGMLDequantize` over 8 row indices.

- [ ] **Step 1: Write the failing test** in `unit_tests/GPUCpuCompareTest.cpp`: build a `[8 rows x 512 cols]` Q6_K table from deterministic floats, gather rows `{0,1,3,7}` through the GPU path, compare each against `GGMLDequantize::dequantize(type, ...)` at `1e-6` relative (Q6_K is exact: FP16 scale + int8 payload).
- [ ] **Step 2: Run it** — `./build-1.0/unit_tests/tinycoder_test --gtest_filter='*Q6K*'`. Expected: FAIL, the GPU path refuses type 14 or returns zeros.
- [ ] **Step 3: Implement** — add `GGML_TYPE_Q6_K` to `supportedEmbeddingType` and the Q6_K arm to the gather switch (block size 210 B, `d` fp16 at bytes 208-209, 16 six-bit groups: 2 bits from `ql`, 4 from `qh`, offsets `-32`; mirror `GGMLDequantize::dequantizeBlock` case 14 exactly, do not re-derive).
- [ ] **Step 4: Run the test** — expected PASS; then `./scripts/text_gate.sh qwen2.5-coder-1.5b` to confirm Q2_K-embedding models still pass.
- [ ] **Step 5: Report the offload decision** — `ModelGPU::ensureUploaded` already writes the reasoning into `err`; make the *server-visible* half explicit by logging one line at load: `[gpu] offload=full|cpu reason=<...>`. This is the "показывать в /v1/models" half of the report's defect: the engine's job is to make the reason a single machine-readable line.
- [ ] **Step 6: Commit** — `git add src/cpp/core/ModelGPU.cpp src/cpp/core/cuda unit_tests/GPUCpuCompareTest.cpp` + `git commit -m "GPU embeddings: Q6_K support and an explicit offload decision line"`.

### Task 3: Gemma 4 loads — per-layer KV heads and shared V

The report's defect 3. Verified in this tree against `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`: `gemma4.attention.head_count_kv` is a **30-element array** `[8,8,8,8,8,2, 8,8,8,8,8,2, …]`, and layers `5,11,17,23,29` (the global ones) have `attn_q`, `attn_k`, `attn_output` but **no `attn_v`** — V is shared with the nearest previous layer. `GGUFLoader.cpp` parses the array with `std::stoul` (first element only) and `ModelLoad.cpp:1226` rejects the layer for the missing V.

**Files:**
- Modify: `include/ModelConfig.hpp` (add `std::vector<uint32_t> numKVHeadsPerLayer`)
- Modify: `src/cpp/core/GGUFLoader.cpp` (~line 622: array form of `attention.head_count_kv`)
- Modify: `src/cpp/core/ModelLoad.cpp` (~line 1200-1230: accept a missing `attn_v` when a previous layer has one; record the source layer index)
- Modify: `src/cpp/core/ModelForward.cpp` and the GPU geometry (`src/cpp/core/ModelGPU.cpp:590-630`) to use the per-layer KV head count
- Test: `unit_tests/GGUFMetadataTest.cpp` (array parsing) and a load test on the checkpoint above

**Interfaces:**
- Produces: `ModelConfig::numKVHeadsPerLayer` (empty means "uniform `numKVHeads`"); `LayerWeights::attnVSharedFrom` (`uint32_t`, `0xFFFFFFFF` when not shared).
- Consumes: `loadQuantized` / `loadF32_1D` already in `ModelLoad`; `QuantizedMatrix` is copyable by value, so aliasing the source layer's matrix is a copy of the descriptor, not of the weights.

- [ ] **Step 1: Write the failing test** — in `GGUFMetadataTest.cpp`, parse the real header of `/data/models/gemma/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` and assert `numKVHeadsPerLayer.size() == 30`, `[5] == 2`, `[0] == 8`. Run it; expect FAIL (`size() == 0`).
- [ ] **Step 2: Parse the array** — in `GGUFLoader`'s `attention.head_count_kv` arm, accept both scalar and array; fill `numKVHeadsPerLayer`; keep `numKVHeads = max_element` so untouched code paths keep their current geometry. Run the test; expect PASS.
- [ ] **Step 3: Write the failing load test** — `TINYCODER_MODEL_PATH=<the gemma file> ./build-1.0/unit_tests/tinycoder_test --gtest_filter='ReferenceCompareTest.EmbeddingStats'` currently fails with `Missing attention weights for layer 5`. That is the reproduction.
- [ ] **Step 4: Implement the alias** — when `attnV` is empty, search backwards for the nearest layer with a non-empty `attnV`, require the same `attnV.rows/cols/type`, set `attnVSharedFrom`, and keep the forward's V projection reading the aliased matrix. Do the same for a missing `attn_k` (some Gemma conversions share K too) — if the search fails for either, keep the current hard failure with the layer number.
- [ ] **Step 5: Use the per-layer KV count** — replace `config_.numKVHeads` with a per-layer accessor (`detail::kvHeadsForLayer(config_, i)`) at every reshape: `ModelForward.cpp` attention block, `ModelGPU.cpp` geometry, and the KV-cache sizing (size from `max` over layers).
- [ ] **Step 6: Verify end to end** — `TINYCODER_MODEL_PATH=<gemma file> ./build-1.0/unit_tests/tiny_logits_probe`-style probe prints finite logits and a plausible token; add the checkpoint to `scripts/text_gate.sh` (label `gemma4 moe`) and add its shape to `AnswersQuestion` in `unit_tests/ModelTest.cpp`; run `./scripts/text_gate.sh gemma4` → PASS.
- [ ] **Step 7: Commit** — `git commit -m "Gemma 4: per-layer KV head counts and shared V on global layers"`.

### Task 4: Release host weight copies after a full GPU upload

The report's improvement 4: Ornith Q4_K_M with all 41 layers on the GPU holds **22.5 GiB of host RAM** and logs `Insufficient memory: need ~42881 MB but only 6963 MB available`; the 7B models hold 11.5–11.9 GiB for 7 GiB of VRAM. Their host has 30 GiB total. The engine keeps dequantized embeddings, fp16 twins and host expert blobs after a successful full upload.

**Files:**
- Modify: `src/cpp/core/ModelGPU.cpp` (after `uploaded_ = true` in the full-offload branch: drop the host-side copies that no CPU path can reach any more)
- Modify: `src/cpp/core/ModelLoad.cpp` (do not build pre-dequantized embeddings when the plan is "everything on GPU", or free them after upload)
- Test: `scripts/sm86_receipt.sh` already records RSS; add a `--rss` probe line to it, or measure `/proc/<pid>/status VmHWM` in the new test below

**Interfaces:**
- Produces: a log line `[gpu] host weights released: <MB>` with the freed bytes, and `Model::hostWeightsReleased()` for tests.
- Constrains: nothing may free a matrix that `cpuFallback_` can still reach — guard on `gpuUnavailable_ == false && uploaded_ && !partialOffload`.

- [ ] **Step 1: Measure the baseline** — run the 9B fully offloaded with `/usr/bin/time -v ./build-1.0/unit_tests/tiny_logits_probe …` and record `Maximum resident set size`.
- [ ] **Step 2: Free after upload** — after a successful full upload, clear `dequantizedEmbeddings_`, the fp16 twin buffers and `expertRequant_` (keeping the quantized originals that the CPU fallback would need); print the freed total.
- [ ] **Step 3: Measure again** — same command; expect at least 1 GiB less RSS on the 9B and no change in the logits gate.
- [ ] **Step 4: Prove the fallback still works** — `TINYCODER_GPU=0` on the same model answers `Paris` (the CPU path must not read a freed buffer).
- [ ] **Step 5: Commit** — `git commit -m "GPU: release host weight copies after a full upload"`.

## Not in this plan (separate plans, with the report's numbers as the target)

- **Long-context prefill** (report §4.1): Bonsai 1206 → 167 tok/s at 60k, TTFT 51 min at 182k; the suspicion is quadratic attention over the full window. Acceptance: prefill at 60k within 2× of the 16k rate, or a measured explanation of why not.
- **Request concurrency** (§4.2): four 4k requests take turns (Ornith 2.1 tok/s summed against 52 single). Needs continuous batching or multiple KV slots.
- **EXL3 27B against exllamav3/NInfer** (§4.3, §5.4): 25 vs 49 tok/s without speculation; NInfer's MTP/DFlash2 reaches 97–110. Speculative decoding does not exist in this tree (MTP weights are loaded and never executed) and is its own plan.
- **Server defects 4-6** — see `2026-11-22-openai-server-defects-from-3090-report.md` (different repository).
