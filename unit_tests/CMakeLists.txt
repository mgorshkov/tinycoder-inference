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
    EXL3Test.cpp
    ReferenceCompareTest.cpp
    ThreadPoolReentrancyTest.cpp
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
# Links the same way as tinycoder_test so CUDA separable compilation,
# np, and tinycoder_core are all wired correctly for tools/tiny_logits_probe.cpp.
add_executable(tiny_logits_probe
    ../tools/tiny_logits_probe.cpp
)

target_link_libraries(tiny_logits_probe
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
