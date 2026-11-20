cmake_minimum_required(VERSION 3.18.0)

# TinyCoder Inference Engine for NVIDIA GPUs (Turing / Ampere / Ada)
# ---------------------------------------------------------------------------
# Standalone C++/CUDA inference core used by the TinyCoder VS Code extension.
# This repository contains ONLY the engine: src/cpp/core (C++/CUDA), include/
# (public engine headers), unit_tests/, benchmarks/ and the build plumbing.
# The N-API bridge (src/cpp/bridge) lives in the extension repository
# (https://github.com/mgorshkov/tinycoder) and consumes this project's tinycoder_core static library
# via FetchContent.
#
# Standalone build (static lib + tests + bench):
#   cmake -B build -DCMAKE_BUILD_TYPE=Release [-DENABLE_CUDA=OFF] [-DBUILD_TESTS=...]
#   cmake --build build --config Release
#   ctest --test-dir build            # when BUILD_TESTS=ON
# ---------------------------------------------------------------------------
project(tinycoder_inference VERSION 0.5.0 DESCRIPTION "High-Performant local inference engine" LANGUAGES CXX)
enable_testing()

# ---- Options ----
# CUDA defaults to ON when nvcc is present: a plain `cmake -B build` on a GPU box
# should produce the offload build, and a CPU-only default was a trap -- it built
# a `tinycoder_bench` that answers `--gpu` with "unknown option: --gpu", which
# reads like a typo rather than a build choice.  -DENABLE_CUDA=OFF still forces
# the CPU build, and a box without nvcc falls back to it automatically.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(TINYCODER_CUDA_DEFAULT ON)
else()
    set(TINYCODER_CUDA_DEFAULT OFF)
endif()
option(ENABLE_CUDA "Enable CUDA GPU acceleration (default: ON when nvcc is available)"
        ${TINYCODER_CUDA_DEFAULT})
# Both SIMD defaults follow the BUILD HOST, from /proc/cpuinfo.  A default build
# has to run where it was built, and it also has to be the configuration the
# benchmarks describe.  Both defaults have been wrong in opposite directions:
#   * ENABLE_AVX512 used to default ON unconditionally, so `cmake -B build` on a
#     Broadwell-class Xeon (the reference box: E5-2686 v4, AVX2 but no AVX-512)
#     produced a binary that died with SIGILL at startup -- including
#     `tinycoder_bench --gpu`, which looked like a GPU problem.
#   * ENABLE_AVX2 used to default OFF, so that same default build compiled
#     SIMDMatMulVecAVX2.cpp into an empty TU (its body is one __AVX2__ guard) and
#     ran scalar matvec: correct, but several times slower on the CPU path, and
#     not the configuration any published row was measured with.
# -DENABLE_AVX2=... / -DENABLE_AVX512=... still force either way, for a
# cross-build or a container whose /proc/cpuinfo describes a different machine.
set(TINYCODER_HOST_AVX2 OFF)
set(TINYCODER_HOST_AVX512 OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" TINYCODER_CPUINFO LIMIT 131072)
    string(FIND "${TINYCODER_CPUINFO}" "avx2" TINYCODER_AVX2_FOUND)
    if(NOT TINYCODER_AVX2_FOUND EQUAL -1)
        set(TINYCODER_HOST_AVX2 ON)
    endif()
    string(FIND "${TINYCODER_CPUINFO}" "avx512f" TINYCODER_AVX512_FOUND)
    if(NOT TINYCODER_AVX512_FOUND EQUAL -1)
        set(TINYCODER_HOST_AVX512 ON)
    endif()
endif()
option(ENABLE_AVX2 "Enable AVX2 optimizations (default: ON when the host CPU has it)"
        ${TINYCODER_HOST_AVX2})
option(ENABLE_AVX512 "Enable AVX-512 optimizations (default: ON when the host CPU has it)"
        ${TINYCODER_HOST_AVX512})
# AVX-512 sub-features (Ice Lake+ only; OPT-IN because the runtime dispatch
# cannot protect code compiled with these flags — a CPU without them will
# SIGILL). ENABLE_AVX512 alone enables only the base set (F/DQ/BW/VL) which
# runs on all AVX-512 CPUs including Cascade Lake.
option(ENABLE_AVX512_VBMI "Enable AVX-512 VBMI (byte permutes; Ice Lake+) optimizations" OFF)
option(ENABLE_AVX512_VBMI2 "Enable AVX-512 VBMI2 (byte compress/expand; Ice Lake+) optimizations" OFF)
option(ENABLE_AMX "Enable AMX (Advanced Matrix Extensions) optimizations" OFF)
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(BUILD_SHARED_LIBS "Build shared library" OFF)
option(TINYCODER_FFN_STEAL "Fused FFN kernel uses chunked work-stealing scheduling" ON)
# CPU affinity for worker threads.
option(TINYCODER_AFFINITY "Pin worker threads to distinct logical CPUs" ON)
# Fused FFN act->Q8_K quantize scheduling.
option(TINYCODER_COOP_QUANTIZE "Fused FFN act quantize uses cooperative last-arriver scheduling" ON)
# Work-stealing chunk size (in tiles) for the fused FFN kernel. 8-12 at 4
# threads reduces that overhead.
set(TINYCODER_STEAL_CHUNK "4" CACHE STRING "Fused FFN work-stealing chunk size (tiles)")
option(BUILD_TESTS "Build unit tests and benchmarks" ON)

# ---- PIC ----
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Fetch shared cmake configuration ----
include(FetchContent)

# Only fetch when a parent scope (e.g. the TinyCoder extension repo consuming
# this project via FetchContent) has not already made it available.
if(NOT cmake_SOURCE_DIR)
    FetchContent_Declare(
        cmake
        GIT_REPOSITORY https://github.com/mgorshkov/cmake.git
        GIT_TAG main
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(cmake)
endif()

# Apply global cmake options (compiler checks, CUDA setup, OpenMP, etc.)
include(${cmake_SOURCE_DIR}/Config.cmake)

# ---- Fetch np library ----
# Only fetch when not already provided by an embedding parent project.
if(NOT TARGET np)
    FetchContent_Declare(
        np
        GIT_REPOSITORY https://github.com/mgorshkov/np.git
        GIT_TAG main
        GIT_SHALLOW TRUE
        CMAKE_ARGS
        -DENABLE_CUDA=${ENABLE_CUDA}
        -DENABLE_OPENMP=${ENABLE_OPENMP}
        -DENABLE_SSE2=${ENABLE_SSE2}
        -DENABLE_SSE3=${ENABLE_SSE3}
        -DENABLE_AVX=${ENABLE_AVX}
        -DENABLE_AVX2=${ENABLE_AVX2}
        -DENABLE_AVX512=${ENABLE_AVX512}
        -DENABLE_AVX512_VBMI=${ENABLE_AVX512_VBMI}
        -DENABLE_AVX512_VBMI2=${ENABLE_AVX512_VBMI2}
        -DENABLE_AMX=${ENABLE_AMX}
        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
        -DDEBUG_INFO=${DEBUG_INFO}
    )
    FetchContent_MakeAvailable(np)
endif()

# ---- Core engine library ----
# All engine sources live in src/cpp/core and are compiled exactly once into a
# static library (tinycoder_core). The extension addon and the unit tests link
# against it, so the engine sources are never duplicated across targets.
# NOTE: CMAKE_CURRENT_SOURCE_DIR is used (not CMAKE_SOURCE_DIR) so this
# project also configures correctly when consumed via FetchContent.
file(GLOB TINYCODER_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core/*.cpp
)

# CUDA source files (only compiled when CUDA is enabled).
# GPU support: the shared cmake Config.cmake sets CMAKE_CUDA_ARCHITECTURES to
# 75 80 86 (RTX 20xx/30xx native SASS + forward-compatible PTX for newer GPUs),
# so the engine runs on Turing / Ampere / Ada and newer NVIDIA GPUs.  Override
# per build for fully native code on other targets, e.g.
#   -DCMAKE_CUDA_ARCHITECTURES=89     # Ada (RTX 40xx) native SASS
#   -DCMAKE_CUDA_ARCHITECTURES=120    # Blackwell (RTX 50xx) native SASS
if(ENABLE_CUDA)
    file(GLOB_RECURSE TINYCODER_CUDA_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core/cuda/*.cu
    )
    list(APPEND TINYCODER_CORE_SOURCES ${TINYCODER_CUDA_SOURCES})
    # The CUDA runtime needs explicit cuBLAS (already found below) plus the
    # shared library path for nvcc device-linking.
    #
    # -lineinfo gives line numbers in profiler traces.  It is not free: nvcc
    # keeps every translation unit's PTX text in the cubin to map SASS back to
    # source, so a fatbin built with it is roughly 2-3x larger.  Measured on this
    # tree with 2077 kernels and arches 75;80;86, one arch's cubin is 164 MB and
    # 90 MB of that is the `.nv_debug_ptx_txt` sections.  Neither the SASS nor
    # its speed changes (-O3 code is identical), so it is OFF by default and
    # opt-in for profiling, where ncu needs the source lines:
    #   cmake -B build -DTINYCODER_LINEINFO=ON
    #
    # Debug builds are excluded regardless: the shared cmake Config.cmake
    # already adds --device-debug (-G), and nvcc warns that -G overrides
    # -lineinfo for every .cu TU.
    option(TINYCODER_LINEINFO
        "CUDA -lineinfo: profiler source lines, but each cubin keeps its TU's PTX text (2-3x larger fatbin)" OFF)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND TINYCODER_LINEINFO)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
        message(STATUS "tinycoder: CUDA -lineinfo enabled (TINYCODER_LINEINFO=ON): "
                       "ncu source lines at 2-3x fatbin size")
    endif()
    # Benign nvlink noise: CMake's separable-compilation device-link step
    # passes the host-only static archives (-ldl/-lrt/-lpthread pulled in via
    # the np/Threads link interface) to nvlink, which cannot use host static
    # archives and prints "Skipping incompatible ... when searching for -l..."
    # for every (arch, target).  They are a toolchain artifact, not code
    # warnings, and cannot be removed from the link line without breaking the
    # host link.  nvcc forwards -Xnvlink options to nvlink only during the
    # -dlink stage (ignored at plain compile), so carrying -Xnvlink -w in
    # CMAKE_CUDA_FLAGS suppresses the nvlink warnings while device-link
    # ERRORS still fail the build.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xnvlink -w")
endif()

set(TINYCODER_HEADERS
    include/AlignedVector.hpp
    include/ChatTemplateRenderer.hpp
    include/EXL3Codec.hpp
    include/EXL3Kernels.hpp
    include/EXL3Loader.hpp
    include/GGMLDequantize.hpp
    include/GGUFLoader.hpp
    include/GPUCompute.hpp
    # GPUComputeInternal.hpp is a PRIVATE implementation detail of the CUDA
    # engine (kernels/helpers shared by the split GPUCompute*.cu translation
    # units).  It is intentionally NOT listed here so it is not installed as a
    # public header.
    include/IModelLoader.hpp
    include/Int8Pack.hpp
    include/IQ3XXS.hpp
    include/LMHead.hpp
    include/LMHeadCUDA.hpp
    include/MemHints.hpp
    include/Model.hpp
    include/ModelConfig.hpp
    include/ModelInternal.hpp
    include/SafetensorsReader.hpp
    include/SIMDMatMulVec.hpp
    include/SIMDMatMulVecInternal.hpp
    include/ThreadPool.hpp
    include/Tokenizer.hpp
)

# ---- SIMD compile flags for the SIMD mat-vec kernels ----
# The scalar implementations and runtime dispatch live in SIMDMatMulVec.cpp,
# which is compiled WITHOUT SIMD ISA extensions. The AVX2 and AVX-512 kernels
# live in their own translation units (SIMDMatMulVecAVX2.cpp /
# SIMDMatMulVecAVX512.cpp) so that per-file flags can enable the required ISA
# without forcing AVX/AVX-512/AMX code generation onto the rest of the codebase.
# Each kernel file uses #ifdef guards so it compiles to an empty TU when the
# corresponding ISA is disabled.
set(SIMD_MATMUL_AVX2_FLAGS "")
if(ENABLE_AVX2)
    set(SIMD_MATMUL_AVX2_FLAGS "${SIMD_MATMUL_AVX2_FLAGS} -mavx2 -mfma -mf16c")
endif()
if(TINYCODER_FFN_STEAL)
    # Lever 1 A/B: route the fused gate+up+down FFN through the chunked
    # work-stealing scheduler (parallelForSteal2) instead of the static slabs.
    set(SIMD_MATMUL_AVX2_FLAGS "${SIMD_MATMUL_AVX2_FLAGS} -DTINYCODER_USE_FFN_STEAL")
    # Work-stealing chunk size (tiles) for both phases of the fused FFN.
    set(SIMD_MATMUL_AVX2_FLAGS "${SIMD_MATMUL_AVX2_FLAGS} -DTINYCODER_STEAL_CHUNK=${TINYCODER_STEAL_CHUNK}")
endif()
if(TINYCODER_COOP_QUANTIZE)
    set(SIMD_MATMUL_AVX2_FLAGS "${SIMD_MATMUL_AVX2_FLAGS} -DTINYCODER_USE_COOP_QUANTIZE")
endif()
if(SIMD_MATMUL_AVX2_FLAGS)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core/SIMDMatMulVecAVX2.cpp
        PROPERTIES
        COMPILE_FLAGS "${SIMD_MATMUL_AVX2_FLAGS}"
    )
endif()

# The sampling pipeline (temperature/top-K/top-P softmax filtering + the
# cumulative-scan draw) runs a full-vocab pass a few times per token; give it
# the same AVX2/FMA flags as the SIMD kernel TU (guarded by __AVX2__/__FMA__
# inside the source, so it degrades to scalar otherwise).
set(SAMPLING_AVX2_FLAGS "")
if(ENABLE_AVX2)
    set(SAMPLING_AVX2_FLAGS "${SAMPLING_AVX2_FLAGS} -mavx2 -mfma")
endif()
if(SAMPLING_AVX2_FLAGS)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core/ModelSampling.cpp
        PROPERTIES
        COMPILE_FLAGS "${SAMPLING_AVX2_FLAGS}"
    )
endif()

# AVX-512 kernels TU. The base flags (-mavx512f -mavx512bw -mfma) cover all
# intrinsics used here (the _mm512_maddubs_epi16/_mm512_madd_epi16 byte/word
# kernels need -mavx512bw) and run on any AVX-512 CPU including Cascade Lake.
# VBMI/VBMI2 are NOT added here automatically: enabling them would let the
# compiler emit vpermi2b/vpermt2b that Cascade Lake cannot execute (SIGILL).
# Projects that KNOW their target supports Ice Lake+ can opt in by enabling
# ENABLE_AVX512_VBMI / ENABLE_AVX512_VBMI2, which also emits the -mavx512vbmi
# / -mavx512vbmi2 flags for this TU.
set(SIMD_MATMUL_AVX512_FLAGS "")
if(ENABLE_AVX512)
    set(SIMD_MATMUL_AVX512_FLAGS "${SIMD_MATMUL_AVX512_FLAGS} -mavx512f -mavx512bw -mfma")
endif()
if(ENABLE_AVX512 AND ENABLE_AVX512_VBMI)
    set(SIMD_MATMUL_AVX512_FLAGS "${SIMD_MATMUL_AVX512_FLAGS} -mavx512vbmi")
endif()
if(ENABLE_AVX512 AND ENABLE_AVX512_VBMI2)
    set(SIMD_MATMUL_AVX512_FLAGS "${SIMD_MATMUL_AVX512_FLAGS} -mavx512vbmi2")
endif()
if(SIMD_MATMUL_AVX512_FLAGS)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core/SIMDMatMulVecAVX512.cpp
        PROPERTIES
        COMPILE_FLAGS "${SIMD_MATMUL_AVX512_FLAGS}"
    )
endif()

# Build the engine core once; tests and the extension addon both link this
# target.
add_library(tinycoder_core STATIC ${TINYCODER_CORE_SOURCES})

target_include_directories(tinycoder_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${np_SOURCE_DIR}/include
)

target_link_libraries(tinycoder_core
    PUBLIC
    np
)

set_target_properties(tinycoder_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Compile-time configuration for the thread pool + fused FFN:
#   TINYCODER_AFFINITY    - 1 pins workers to distinct physical cores, 0 lets
#                           the OS schedule on all logical CPUs (default; faster
#                           at low thread counts on memory-bound generation).
#   TINYCODER_STEAL_CHUNK - work-stealing chunk size (tiles) for the fused FFN.
target_compile_definitions(tinycoder_core PRIVATE
    TINYCODER_AFFINITY=$<IF:$<BOOL:${TINYCODER_AFFINITY}>,1,0>
    TINYCODER_STEAL_CHUNK=${TINYCODER_STEAL_CHUNK}
)
# CUDA configuration for the core library
if(ENABLE_CUDA)
    target_compile_definitions(tinycoder_core PUBLIC USE_CUDA)
    set_target_properties(tinycoder_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    find_library(CUBLAS_LIBRARY cublas)
    # cublasLt is used for the EXL3 prefill GEMM: cublasGemmEx's default tile
    # quantises badly on this model's shapes (1.25x slower on ffn_down).  It
    # ships in the same package as cublas.
    find_library(CUBLASLT_LIBRARY cublasLt)
    if(CUBLAS_LIBRARY)
        target_link_libraries(tinycoder_core PRIVATE ${CUBLAS_LIBRARY})
        target_compile_definitions(tinycoder_core PRIVATE USE_CUBLAS)
        message(STATUS "cuBLAS found: ${CUBLAS_LIBRARY}")
        if(CUBLASLT_LIBRARY)
            target_link_libraries(tinycoder_core PRIVATE ${CUBLASLT_LIBRARY})
            message(STATUS "cuBLASLt found: ${CUBLASLT_LIBRARY}")
        else()
            message(WARNING "cuBLASLt not found - EXL3 prefill GEMM uses cublasGemmEx")
        endif()
    else()
        message(WARNING "cuBLAS not found - LM head CUDA acceleration disabled")
    endif()
endif()

# ---- Unit Tests (Google Test) + benchmarks ----
if(BUILD_TESTS)
    add_subdirectory(unit_tests)
    # reference-benchmark parity warm generation benchmark (benchmarks/BenchMain.cpp).
    # Kept intentionally OUT of unit_tests/: it is a measurement harness, not a
    # parity/correctness test - it would add a gtest-incompatible generate loop.
    add_subdirectory(benchmarks)
endif()

# ---- Install ----
install(TARGETS tinycoder_core ARCHIVE DESTINATION lib)
install(FILES ${TINYCODER_HEADERS} DESTINATION include/tinycoder)

# ---- Print configuration summary ----
message(STATUS "")
message(STATUS "TinyCoder Inference Engine for NVIDIA GPUs (Turing / Ampere / Ada) Build Configuration:")
message(STATUS "  AVX2:          ${ENABLE_AVX2}")
message(STATUS "  AVX-512:       ${ENABLE_AVX512}")
message(STATUS "  AVX-512 VBMI:  ${ENABLE_AVX512_VBMI}")
message(STATUS "  AVX-512 VBMI2: ${ENABLE_AVX512_VBMI2}")
message(STATUS "  AMX:           ${ENABLE_AMX}")
message(STATUS "  OpenMP:        ${ENABLE_OPENMP}")
if(ENABLE_CUDA)
    message(STATUS "  CUDA:          ON (${CMAKE_CUDA_COMPILER})")
elseif(CMAKE_CUDA_COMPILER)
    message(STATUS "  CUDA:          OFF by request (-DENABLE_CUDA=OFF; nvcc is available)")
else()
    message(STATUS "  CUDA:          OFF (no nvcc found)")
endif()
message(STATUS "  Tests:         ${BUILD_TESTS}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  np library:    ${np_SOURCE_DIR}")
message(STATUS "  Output:        ${CMAKE_BINARY_DIR}/libtinycoder_core.a")
message(STATUS "")
