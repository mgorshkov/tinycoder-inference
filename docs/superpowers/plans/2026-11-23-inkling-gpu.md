# Inkling-2-Mini CUDA Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Inkling-2-Mini IQ2_XXS decode entirely on the 2080 Ti and beat
llama.cpp CPU (4.77 tg16), with llama.cpp CUDA (32.20 tg16) as the stretch.

**Architecture:** A new `InklingGPU` device module owns the uploaded weights,
48 K/V slots and 48 short-conv state buffers. One forward launches a fixed
sequence of small kernels (embedding, projections via quantized GEMV,
conv/attention/depth-mix/control glue, routed MoE, LM head) on one stream. The
CPU path stays the reference and fallback.

**Tech Stack:** C++20, CUDA 12.4, sm_75, cuBLAS as a plain GEMM, the engine's
existing `cuda/gemm` int8 machinery and `QuantBlocks.cuh` block decoders.

**Spec:** `docs/superpowers/specs/2026-11-23-inkling-gpu-design.md`

## Global Constraints

- GPU: RTX 2080 Ti, 11 GB, sm_75, 616 GB/s. No `cp.async`; use synchronous staging.
- Target checkpoint: `/data/models/inkling/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf` (8.49 GB).
- Window: 2048 tokens; the planner may reduce, never overcommit VRAM.
- Geometry is fixed for this model: hidden 2304, 24/8 heads, head_dim 128,
  24 physical blocks, loop [6,18) x 3, 48 slots, 128 experts, expert FF 1152,
  relative 16 x 1024, shortconv 4, vocab 200058 (padded 201024).
- GPU parity: top-1 exact, top-5 set, each top-5 logit within 0.15 absolute.
  Per-stage tensors: cosine >= 0.999.
- Do not commit; the user commits. A task ends with a build + test checkpoint and a diff summary.
- Rebuild `tinycoder_core` before any tool: `cmake --build build --target tinycoder_core <tool> -j 20`.

## Review Focus

- **VRAM overcommit**: 8.49 GB weights + 0.4 GB K/V + scratch must fit 11 GB;
  a failed allocation must fall back to the CPU path, not crash.
- **SWA/full boundary**: block 5, 11, 17, 23 are full attention; a wrong mask
  or window silently changes text.
- **Loop-slot state aliasing**: pass 1/2 must use slots 24-35/36-47, never
  overwrite pass 0's K/V or conv history.
- **Near-tie logits**: IQ2's 0.1-spread entries; the 0.15 band is the gate, not
  exact order.
- **Bad/unknown tensor types**: an upload must reject a type it cannot decode
  rather than produce zeros.

---

### Task 1: Upload skeleton and the four GEMV decoders

**Files:**
- Create: `include/gpu/InklingGPU.hpp`
- Create: `src/cpp/core/cuda/inkling/InklingGPU.cu`
- Create: `src/cpp/core/cuda/inkling/InklingGemv.cu`
- Create: `tools/inkling_gpu_parity.cpp`
- Modify: `include/Model.hpp` (friend `InklingGPU`; `inklingGpuState_`)
- Modify: `tools/CMakeLists.txt` (add `inkling_gpu_parity`)

**Interfaces:**
- Consumes: `Model::layers_` (`QuantizedMatrix` raw pointers/type/rows/cols),
  `cuda/gemm/QuantBlocks.cuh` block structs and `tc_type_traits`.
- Produces:
  - `bool inklingGemv(uint32_t type, const void *w, const float *x, float *out, uint32_t rows, uint32_t cols, cudaStream_t stream);`
    supported types: 40 NVFP4, 42 Q2_0, 19 IQ1_S, 16 IQ2_XXS.
  - `class InklingGPU { bool upload(const Model&, std::string&); bool available() const; ... };`
  - `bool inklingUploadTensor(void*& dst, const QuantizedMatrix&, std::string&);`

- [ ] **Step 1: Write the failing parity test.** `tools/inkling_gpu_parity.cpp`
  loads IQ2, allocates one expert row set for each type, computes the CPU
  reference with `GGMLDequantize::dequantize`, runs `inklingGemv`, and asserts
  `max|gpu-cpu| / max|cpu| < 1e-2` for each type. Print the per-type ratio.
- [ ] **Step 2: Run it and confirm it fails** with `inklingGemv` not defined.
- [ ] **Step 3: Implement upload + one warp-per-row decoder per type.**
  Decode each block with the same block layout as `QuantBlocks.cuh`; for
  IQ1_S/IQ2_XXS reuse `iq1s_grid_gpu`/grid lookups already in the tree. Q8_1
  activation quantisation can call the existing `gemvLaunchQuantizeQ8F32`.
- [ ] **Step 4: Run the parity test.** Expected: all four types print a ratio
  below 1e-2, exit 0.
- [ ] **Step 5: Checkpoint.** `cmake --build build --target tinycoder_core
  inkling_gpu_parity -j 20`; record the ratios. Do not commit.

### Task 2: Elementwise and short-convolution kernels

**Files:**
- Create: `src/cpp/core/cuda/inkling/InklingElementwise.cu`
- Modify: `include/gpu/InklingKernels.hpp`
- Modify: `tools/inkling_gpu_parity.cpp`

**Interfaces:**
- Consumes: Task 1's device buffers and stream.
- Produces:
  - `void kInklingRmsNorm(const float*, const float*, float*, uint32_t n, float eps, cudaStream_t);`
  - `void kInklingClip(float*, uint32_t rows, uint32_t len, const float* scale, float threshold, cudaStream_t);`
  - `void kInklingShortConv(const float* cur, const float* weight, const float* dynamic, float* history, float* out, uint32_t channels, uint32_t nConv, uint32_t nHeads, bool residual, float scale, cudaStream_t);`

- [ ] **Step 1: Add failing tests** for RMSNorm (vs CPU), clip (a row whose
  norm exceeds the threshold), and a 4-tap conv with a 2-token history.
- [ ] **Step 2: Run and confirm failures.**
- [ ] **Step 3: Implement the kernels**, one block per row for norm/clip; one
  block per channel group for conv, updating history in place.
- [ ] **Step 4: Run the tests.** Expected: max diff < 1e-5 for norm/clip,
  < 1e-4 for conv.
- [ ] **Step 5: Checkpoint.**

### Task 3: Relative bias, attention and depth mix

**Files:**
- Create: `src/cpp/core/cuda/inkling/InklingAttention.cu`
- Modify: `tools/inkling_gpu_parity.cpp`

**Interfaces:**
- Consumes: `Q/K/V/R` device buffers, slot K/V cache, relative-bias table.
- Produces:
  - `void kInklingRelBias(const float* r, const float* bias, const float* sink, float* table, uint32_t heads, uint32_t nRel, uint32_t extent, cudaStream_t);`
  - `void kInklingAttention(const float* q, const float* kCache, const float* vCache, const float* table, uint32_t pos, uint32_t headDim, uint32_t nHeads, uint32_t nKvHeads, uint32_t window, bool isSwa, float scale, float* out, cudaStream_t);`
  - `void kInklingMix(const float* const* states, uint32_t nStates, const float* query, uint32_t nQueries, uint32_t hidden, float* out, cudaStream_t);`

- [ ] **Step 1: Write failing tests**: table row 0 vs CPU; attention with 1
  key and sink; attention with 3 keys at distances 0-2; mix over 2 states.
- [ ] **Step 2: Run and confirm failures.**
- [ ] **Step 3: Implement.** Attention: one block per (head, query), stage K/V
  tiles and the bias row in shared memory; softmax includes the zero sink
  logit; SWA starts at `max(0, pos+1-window)`.
- [ ] **Step 4: Run tests.** Expected cosine 1.0 for the table/mix, max diff
  < 1e-4 for attention on the small cases.
- [ ] **Step 5: Checkpoint.**

### Task 4: Router, SITU and routed experts

**Files:**
- Create: `src/cpp/core/cuda/inkling/InklingMoe.cu`

**Interfaces:**
- Produces:
  - `void kInklingRouter(const float* logits, const float* bias, int32_t* idx, float* wgt, uint32_t nExperts, uint32_t nUsed, float scale, cudaStream_t);`
  - `void kInklingSitu(float* gateUp, uint32_t n, float beta, float lb, cudaStream_t);`
  - `void kInklingMoeAccum(const float* expertOut, const float* wgt, float* out, uint32_t nUsed, uint32_t hidden, cudaStream_t);`

- [ ] **Step 1: Write failing tests**: router top-8 on a known logits vector
  (including a tie and a bias flip); SITU vs CPU; weighted sum of 2 experts.
- [ ] **Step 2: Run and confirm failures.**
- [ ] **Step 3: Implement.** Router is a single block (128 experts) with a
  stable top-k; experts call Task 1's `inklingGemv` on the expert row slice,
  then SITU, then the down GEMV.
- [ ] **Step 4: Run tests.** Expected exact ids/weights and max diff < 1e-5.
- [ ] **Step 5: Checkpoint.**

### Task 5: Full decode graph

**Files:**
- Create: `src/cpp/core/cuda/inkling/InklingForward.cu`
- Modify: `include/gpu/InklingGPU.hpp`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: `bool InklingGPU::forward(const std::vector<int32_t>& tokens, bool allLogits, float* logits, std::string& err);`
  and `void InklingGPU::clearKVCache();`.

- [ ] **Step 1: Write a failing end-to-end test** in
  `tools/inkling_gpu_parity.cpp`: token `9707`, compare the last-token logits
  with `Model::forward` (CPU), top-5 set and 0.15 band.
- [ ] **Step 2: Run and confirm the GPU path is missing.**
- [ ] **Step 3: Implement the order**: embed+norm, prelude, three passes with
  `slot = pass==0 ? ip : 24 + (pass-1)*12 + (ip-6)`, bank/message mix,
  postlude, output mix/norm, LM head, softcap, vocab trim. One stream; no host
  sync until logits.
- [ ] **Step 4: Run the test.** Expected: top-1 matches and the band passes.
- [ ] **Step 5: Checkpoint.**

### Task 6: Model dispatch, memory guard and reset

**Files:**
- Modify: `include/Model.hpp`, `src/cpp/core/ModelLoad.cpp`,
  `src/cpp/core/ModelForward.cpp`, `src/cpp/core/ModelGPU.cpp`

**Interfaces:**
- Consumes: `InklingGPU::upload/forward/clearKVCache`.
- Produces: `Model::inklingGpuForward(...)`, `Model::gpuUploadIfEnabled`
  branch for `inkling2`, and `clearKVCache()` zeroing the device conv/KV state.

- [ ] **Step 1: Write a failing test**: `TINYCODER_GPU=1` + IQ2, assert the
  model takes the GPU path and the logits match the CPU path; assert
  `TINYCODER_GPU=0` still uses CPU.
- [ ] **Step 2: Run and confirm the branch is missing.**
- [ ] **Step 3: Implement the branch.** If upload fails for any reason, print
  the reason and fall back to CPU; never leave a half-uploaded session.
- [ ] **Step 4: Run the test.**
- [ ] **Step 5: Checkpoint.**

### Task 7: Correctness gate

**Files:**
- Modify: `scripts/text_gate.sh` (already has the IQ2 row)
- Modify: `README.md`, `benchmarks/gpu/BENCHMARKS-2080TI.md` only after the gate passes.

- [ ] **Step 1: Run** `TINYCODER_GPU=1 TINYCODER_MODEL_PATH=<IQ2> ./build/unit_tests/tinycoder_test --gtest_filter='FullQuestions/SampleQuestionTest.AnswersQuestion/1'`.
- [ ] **Step 2:** Expected: PASS with "Paris".
- [ ] **Step 3: Run the fixed-id logits gate** on
  `9707,11,1879,323,358,279,2779,338`; top-1 exact and top-5 within 0.15.
- [ ] **Step 4: Checkpoint.**

### Task 8: Performance to the contract

**Files:**
- Modify: `src/cpp/core/cuda/inkling/InklingForward.cu`
- Modify: `benchmarks/gpu/BENCHMARKS-2080TI.md`

- [ ] **Step 1: Measure** `tinycoder_bench --model <IQ2> --n-prompts 16
  --n-gen 16 --threads 36` with `TINYCODER_GPU=1`. Record pp16/tg16 and spread.
- [ ] **Step 2:** If tg16 <= 4.77, profile: per-kernel time with
  `TINYCODER_GPU_VERBOSE=1` or nsys; identify the top three kernels.
- [ ] **Step 3:** Apply one lever at a time: CUDA graph capture, batch the
  eight expert GEMVs into one launch, vectorize attention, reduce syncs.
- [ ] **Step 4:** Re-measure. The task is done only when tg16 > 4.77; report
  the number even if it is slower, with the limiting kernel.
- [ ] **Step 5: Update the benchmark table** only with a gate-passing run.

### Task 9: Prefill (stretch)

**Files:**
- Modify: `src/cpp/core/cuda/inkling/InklingForward.cu`

- [ ] **Step 1:** Extend `inklingGemv` to m=8 tokens per launch (the four
  Inkling types are not in `int8Gemv`) and tile attention over the batch.
- [ ] **Step 2:** Measure pp16; target >= 5 (stretch 10).
- [ ] **Step 3:** Re-run the text gate; checkpoint.

---

## Self-review

- Spec coverage: intent/goals (Tasks 1-8), parity (1,2,3,5,7), memory (6),
  prefill (9), risks (2,3,6).
- Types: `inklingGemv` is used by Tasks 1, 4, 5; `InklingGPU` by 5, 6.
- Review Focus tests: VRAM fallback (Task 6), SWA/full (Task 3), slot aliasing
  (Task 5), near-tie band (Task 5), unknown type rejection (Task 1).
