cmake_minimum_required(VERSION 3.18.0)

set(PROJECT_NAME tinycoder_unit_tests)

project(${PROJECT_NAME} VERSION 0.5.0 DESCRIPTION "TinyCoder Inference Unit Tests")

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# tinycoder_core static library is built in the root CMakeLists.txt
# and is available as a target in this subdirectory scope.

# ---- Google Test executable ----
add_executable(tinycoder_test
    Main.cpp
    ModelTest.cpp
    DequantizeTest.cpp
    ContextPlanTest.cpp
    EXL3Test.cpp
    ReferenceCompareTest.cpp
    ThreadPoolReentrancyTest.cpp
    ThreadPoolOwnershipTest.cpp
    EmbeddingTableTest.cpp
    GPUCpuCompareTest.cpp
    Int8PackParityTest.cpp
    SharedTestEnv.cpp
)

target_link_libraries(tinycoder_test
    PRIVATE
    gtest
    tinycoder_core
)

# ---- Toolchain workaround (CUDA + GCC) ----
# nvcc's device-link step injects `-L/usr/lib/gcc/x86_64-linux-gnu/<N>` ahead of
# the host C++ runtime, so the driver's implicit `-lstdc++` resolves to the
# gcc-<N> libstdc++ (a build that predates CXXABI_1.3.15).  googletest's
# `main` references `__cxa_call_terminate`, which only the newer SYSTEM
# libstdc++ defines, so the test fails to link with:
#   undefined reference to `__cxa_call_terminate'
# Appending the system libstdc++ by full path (after the other libs) restores
# the symbol without changing the CPU-only build.  Guarded by an EXISTS check
# so a non-standard layout is simply left to the default toolchain.
if(ENABLE_CUDA AND NOT APPLE AND EXISTS "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    target_link_libraries(tinycoder_test PRIVATE /usr/lib/x86_64-linux-gnu/libstdc++.so.6)
endif()

# ---- Standalone logits probe (diagnostic; not a gtest) ----
# Links the same way as tinycoder_test so CUDA separable compilation and
# tinycoder_core are all wired correctly for tools/tiny_logits_probe.cpp.
add_executable(tiny_logits_probe
    ../tools/tiny_logits_probe.cpp
)

target_link_libraries(tiny_logits_probe
    PRIVATE
    tinycoder_core
)

# Context-window probe: needle-in-a-haystack at a requested window.  The gate
# script (scripts/context_gate.sh) drives it, so it lives next to the logits
# probe rather than in tests/.
add_executable(ctx_probe
    ../tools/ctx_probe.cpp
)

target_link_libraries(ctx_probe
    PRIVATE
    tinycoder_core
)

# ---- EXL3 checkpoint smoke probe (diagnostic; not a gtest) ----
# Loads a real EXL3 HF directory through EXL3Loader and validates
# the materialized GGUF-named tensor inventory + numeric sanity, without a
# tokenizer.  Runs on CPU (no CUDA needed).
add_executable(exl3_model_smoke
    ../tools/exl3_model_smoke.cpp
)

target_link_libraries(exl3_model_smoke
    PRIVATE
    tinycoder_core
)

# ---- EXL3 GPU kernel parity probe (diagnostic; not a gtest) ----
# Cross-checks the EXL3 decode/reconstruct/GEMV GPU kernels against the CPU
# reference codec.  Self-contained (synthetic trellis), so it runs without a
# model file.  Only meaningful in a CUDA build.
if(ENABLE_CUDA)
    add_executable(exl3_gpu_parity
        ../tools/exl3_gpu_parity.cpp
    )

    target_link_libraries(exl3_gpu_parity
        PRIVATE
        tinycoder_core
        CUDA::cudart
    )

    set_target_properties(exl3_gpu_parity PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- Q2_0 int8 MMQ parity + microbench (Inkling expert down) ----------
    # Runs the standalone kernel on the real checkpoint's Q2_0 down tensor,
    # checks it against an fp64 dot over the dequantized rows (2e-3, the Q8_1
    # activation floor) and reports TFLOPS for the real cols=1152/rows=2304
    # shape.  This is the gate for src/cpp/core/cuda/gemm/Q2_0Mma.cu.
    add_executable(q2_0_mmq_parity
        ../tools/q2_0_mmq_parity.cpp
    )

    target_link_libraries(q2_0_mmq_parity
        PRIVATE
        tinycoder_core
        CUDA::cudart
    )

    set_target_properties(q2_0_mmq_parity PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- GGUF prefill int8 GEMM per-shape probe (diagnostic; not a gtest) ----
    # Times int8Gemm for the exact (type, rows, cols) of the 7B/9B projections,
    # so the prefill deficit can be attributed to a shape instead of an average.
    add_executable(int8_gemm_probe
        ../tools/int8_gemm_probe.cpp
    )

    target_link_libraries(int8_gemm_probe
        PRIVATE
        tinycoder_core
        CUDA::cudart
        CUDA::cublas
        nvidia-ml
    )

    set_target_properties(int8_gemm_probe PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- int8 GEMM numeric gate (diagnostic; not a gtest) ----
    # Compares int8Gemm against a float64 dot over the dequantized weights of a
    # real checkpoint, per (type, rows, cols).  Gates the sm_86 prefill work.
    add_executable(int8_gemm_parity
        ../tools/int8_gemm_parity.cpp
    )

    target_link_libraries(int8_gemm_parity
        PRIVATE
        tinycoder_core
        CUDA::cudart
    )

    set_target_properties(int8_gemm_parity PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- Synthetic int8 GEMM shape probe (diagnostic; not a gtest) ----
    # Builds Q6_K shapes no checkpoint in this tree has (k=2048/6144, the ones
    # the rented 3090 run logged as `int8 launch failed`), from a real block run
    # lifted out of a checkpoint, and compares the int8 path against fp64.
    add_executable(int8_gemm_shape_probe
        ../tools/int8_gemm_shape_probe.cpp
    )

    target_link_libraries(int8_gemm_shape_probe
        PRIVATE
        tinycoder_core
        CUDA::cudart
    )

    set_target_properties(int8_gemm_shape_probe PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- MoE token-block throughput probe (diagnostic; not a gtest) ----
    # Measures the int8 GEMM rate at m = 8/16/32/64, n=512, k=2048, i.e. one
    # routed expert's gate matrix at prefill.  Decides whether a grouped
    # (grid.z = expert) MoE kernel could keep up with a PCIe weight stream.
    add_executable(int8_m16_probe
        ../tools/int8_m16_probe.cu
    )

    target_link_libraries(int8_m16_probe
        PRIVATE
        tinycoder_core
        CUDA::cudart
    )

    set_target_properties(int8_m16_probe PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    # ---- GGUF decode GEMV numeric gate (diagnostic; not a gtest) ----
    # Per (type, rows, cols) against an fp64 reference on real weights.  The
    # decode path had no tool coverage at all before this.
    add_executable(gemv_decode_parity
        ../tools/gemv_decode_parity.cpp
    )

    target_link_libraries(gemv_decode_parity
        PRIVATE
        tinycoder_core
        CUDA::cudart
    )

    set_target_properties(gemv_decode_parity PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- Expert-down int8 timing probe (diagnostic; not a gtest) ----
    # Answers "would the down projection profit from an int8 dot?" by timing the
    # float expert GEMV against int8GemvExpert on the real 8-expert down shape.
    # The int8 path cannot serve the real call's 2*ff activation stride, so the
    # probe gives it the contiguous stride it accepts -- its best case.
    add_executable(gemv_down_probe
        ../tools/gemv_down_probe.cpp
    )

    target_link_libraries(gemv_down_probe
        PRIVATE
        tinycoder_core
        CUDA::cudart
    )

    set_target_properties(gemv_down_probe PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- MoE expert-path numeric gate (diagnostic; not a gtest) ----
    # The 3-D expert stacks and their row-window GEMV had no coverage: both
    # other parity tools only walk 2-D tensors.  This one checks the router and
    # one slice per expert against float64 on real checkpoint weights.
    add_executable(moe_parity
        ../tools/moe_parity.cpp
    )

    target_link_libraries(moe_parity
        PRIVATE
        tinycoder_core
    )

    set_target_properties(moe_parity PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- EXL3 GPU kernel benchmark (diagnostic; not a gtest) ----
    # Times the reconstruct pipeline + steady-state GEMV over the real layer
    # shapes of a checkpoint and projects a decode step.  Needs the CUDA runtime
    # headers directly (cudaGetDeviceProperties), hence CUDA::cudart.
    add_executable(exl3_kernel_bench
        ../tools/exl3_kernel_bench.cpp
    )

    target_link_libraries(exl3_kernel_bench
        PRIVATE
        tinycoder_core
    )

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(exl3_kernel_bench PRIVATE CUDA::cudart)

    # ---- Decode-vs-prefill logits agreement probe (diagnostic) ----
    add_executable(exl3_decode_agree
        ../tools/exl3_decode_agree.cpp
    )

    target_link_libraries(exl3_decode_agree
        PRIVATE
        tinycoder_core
    )

    set_target_properties(exl3_decode_agree PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    set_target_properties(exl3_kernel_bench PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- Per-shape fused-GEMV probe (diagnostic; not a gtest) ----
    # Registers a synthetic linear of one real decode geometry through the
    # shipping registry (uploadFusedLinearDevice) and times gemvFused, so a
    # kernel-selection sweep costs seconds instead of a 40 s model load.  It
    # enters through the production dispatch, so every selection knob
    # (TINYCODER_EXL3_GEMV_K1, _K1TT, _GEMV_SPLIT, TINYCODER_EXL3_K1DIAG)
    # still applies.
    add_executable(exl3_gemv_probe
        ../tools/exl3_gemv_probe.cpp
    )

    target_link_libraries(exl3_gemv_probe
        PRIVATE
        tinycoder_core
    )

    target_link_libraries(exl3_gemv_probe PRIVATE CUDA::cudart)

    set_target_properties(exl3_gemv_probe PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- Per-output-element decode-GEMV probe (diagnostic; not a gtest) ----
    # Decodes ONE real 1-bit projection with the CPU reference codec, computes
    # What^T x in double, and compares the shipped `gemvFused` per output
    # element, so the decode GEMV finally has a probe that reaches it: the
    # 4-token logits probe never runs a decode GEMV, and exl3_gpu_parity's K=1
    # cases are cb=0/cb=1, which the codebook gate routes away from the K=1
    # fast path.  Needs the model directory as argv[1]; OpenMP is speed only.
    add_executable(exl3_k1_probe
        ../tools/exl3_k1_probe.cu
    )

    target_link_libraries(exl3_k1_probe
        PRIVATE
        tinycoder_core
    )

    target_link_libraries(exl3_k1_probe PRIVATE CUDA::cudart)

    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(exl3_k1_probe PRIVATE OpenMP::OpenMP_CXX)
    endif()

    set_target_properties(exl3_k1_probe PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # ---- fp16-accumulate pre-scale decision probe (diagnostic; not a gtest) --
    # Standalone measurement of the EXL3 prefill GEMM for one real projection:
    # fp32-accumulate HMMA vs fp16-accumulate with an exact 2^-s pre-scale,
    # both against an fp64 cuBLAS reference, against materialise+cuBLAS.  It
    # copies the decode/MMA code instead of including EXL3Kernels.cu, so it
    # cannot inherit a stale core object or perturb the engine.  Nothing in the
    # shipped path changes; this only adds an executable.
    add_executable(exl3_hacc_probe
        ../tools/exl3_hacc_probe.cu
    )

    target_link_libraries(exl3_hacc_probe
        PRIVATE
        tinycoder_core
        CUDA::cudart
        CUDA::cublas
    )

    set_target_properties(exl3_hacc_probe PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
endif()

target_include_directories(tinycoder_test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Testing
)

# Set C++ standard for the test executable
set_target_properties(tinycoder_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Register the test with CTest
add_test(NAME tinycoder_test COMMAND tinycoder_test)

if(ENABLE_CUDA)

add_executable(inkling_gpu_parity
    ../tools/inkling_gpu_parity.cpp
)
target_link_libraries(inkling_gpu_parity PRIVATE tinycoder_core CUDA::cudart)

# ---- Inkling-2-Mini full CUDA decode graph parity gate (diagnostic) ----
# Runs token 9707 through Model::forward (CPU reference) and
# gpu::InklingGPU::forward, then gates on top-1 exact, top-5 set and each
# top-5 logit within 0.15 absolute.  Needs the IQ2_XXS checkpoint as argv[1].
add_executable(inkling_gpu_forward_test
    ../tools/inkling_gpu_forward_test.cpp
)
target_link_libraries(inkling_gpu_forward_test PRIVATE tinycoder_core CUDA::cudart)
set_target_properties(inkling_gpu_forward_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ---- Inkling-2-Mini batched-prefill GEMV gate (diagnostic) ----
# inklingGemvBatched decodes every weight block once for an m-token batch.
# This tool compares it, per output element, against m single-token inklingGemv
# calls on one real tensor per hot type (NVFP4, Q2_0, IQ1_S, IQ2_XXS, IQ2_XS,
# IQ2_S) at m=16, checks a partial-row/partial-M tail, and prints the m=16
# speedup on an 8-expert ffn slice.  Needs the IQ2_XXS checkpoint as argv[1].
add_executable(inkling_gpu_batched_test
    ../tools/inkling_gpu_batched_test.cpp
)
target_link_libraries(inkling_gpu_batched_test PRIVATE tinycoder_core CUDA::cudart)
set_target_properties(inkling_gpu_batched_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ---- Inkling-2-Mini batched-prefill gate (parity + pp16/pp512) ----
# Runs one prompt twice on one engine: the token-major decode graph
# (TINYCODER_INKLING_BATCHED=0) and the [H, m] batched prefill graph, then
# requires per-position top-1/top-5 parity, logit cosine >= 0.9999, the same
# cosine on the prelude hidden state and on every block's hidden state, and
# prints both prefill throughputs.  Needs the IQ2_XXS checkpoint.
#   inkling_gpu_batched_prefill_test <model.gguf> --m 16 --bench 512 --reps 1
add_executable(inkling_gpu_batched_prefill_test
    ../tools/inkling_gpu_batched_prefill_test.cpp
)
target_link_libraries(inkling_gpu_batched_prefill_test PRIVATE tinycoder_core CUDA::cudart)
set_target_properties(inkling_gpu_batched_prefill_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ---- Grouped gate/up expert MMQ parity + microbench (IQ2_XXS/XS/S) ----
# Standalone float64 gate on the real blk.*.ffn_gate_up_exps tensor at the 2e-3
# int8 floor, plus per-path timing of one grouped layer (mma / float / int8
# 4-row slices) so a pp delta can be attributed to the kernel.
#   inkling_expert_mma_parity <model.gguf> --type 22 --oneshot 0
add_executable(inkling_expert_mma_parity
    ../tools/inkling_expert_mma_parity.cpp
)
target_link_libraries(inkling_expert_mma_parity PRIVATE tinycoder_core CUDA::cudart)
set_target_properties(inkling_expert_mma_parity PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ---- FP32 dequant + cuBLAS SGEMM prototype for the routed experts ---------
# Measures the hypothesis "dequantize the experts to FP32 once, then SGEMM the
# token columns" against the current grouped float GEMV on the real blk.0
# tensors: dequant exactness vs GGMLDequantize, dequant GB/s, SGEMM TFLOPS at
# m=8..512, one real MoE block replayed from the engine's dumped hidden states
# (parity per token), the real routed row counts, and a projected pp512.
# Nothing in the shipped graph is touched.
#   inkling_fp32_gemm_probe <model.gguf> [--m 128] [--dump <dir>]
add_executable(inkling_fp32_gemm_probe
    ../tools/inkling_fp32_gemm_probe.cpp
)
target_link_libraries(inkling_fp32_gemm_probe PRIVATE tinycoder_core CUDA::cudart CUDA::cublas)
set_target_properties(inkling_fp32_gemm_probe PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ---- Batched attention kernel isolation probe ----
# Times the attention kernel alone on the real geometry and reports cycles per
# (token, key), so a section number can be attributed to the kernel instead of
# guessed at.  --tile compares the 8-query-token tiled form.
add_executable(inkling_attn_probe
    ../tools/inkling_attn_probe.cpp
)
target_link_libraries(inkling_attn_probe PRIVATE tinycoder_core CUDA::cudart)
set_target_properties(inkling_attn_probe PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

endif() # ENABLE_CUDA
