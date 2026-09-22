#!/bin/bash
#
# build.sh — Build the TinyCoder inference engine as a static library
#            (plus the unit tests and benchmarks when BUILD_TESTS=ON).
#
# This script:
#   1. Configures the CMake build tree.
#   2. Builds tinycoder_core (plus tinycoder_test and tinycoder_bench).
#   3. Runs CTest when the tests are enabled.
#
# CUDA (GPU offload) is the DEFAULT build path — the engine is developed and
# benchmarked against the llama.cpp GPU baseline with -DENABLE_CUDA=ON.  On a
# machine without nvcc/CUDA, pass --no-cuda (or --cpu) to request a CPU-only
# build instead of failing the configure.
#
# SIMD flags: each SIMD feature is a separate --feature/--no-feature pair that
# is translated into a -DENABLE_* CMake cache entry and appended to
# EXTRA_CMAKE_ARGS, so you can select the instruction sets without editing
# CMakeLists.txt.  The AVX-512 sub-features (VBMI / VBMI2) are Ice Lake+ only
# and default OFF — turning them on for a CPU that lacks them (e.g. Cascade
# Lake) produces code that crashes with SIGILL at runtime.
#
# Prerequisites:
#   - C++20 compiler (gcc >= 13, clang >= 14, or MSVC 2019+)
#   - CMake >= 3.18
#   - CUDA Toolkit >= 11.0 + nvcc (default GPU build only)
#
# Usage:
#   ./scripts/build.sh [options]
#
# Generic:
#   --cuda           GPU build: -DENABLE_CUDA=ON (default)
#   --no-cuda | --cpu   CPU-only build: -DENABLE_CUDA=OFF
#   --no-tests       Skip unit tests and benchmarks: -DBUILD_TESTS=OFF
#
# SIMD (each can be --X to enable or --no-X to disable):
#   --sse2 / --no-sse2
#   --avx  / --no-avx
#   --avx2 / --no-avx2
#   --avx512 / --no-avx512
#   --avx512-vbmi / --no-avx512-vbmi   (Ice Lake+ only; default OFF)
#   --avx512-vbmi2 / --no-avx512-vbmi2 (Ice Lake+ only; default OFF)
#
# Examples:
#   ./scripts/build.sh                          # GPU offload build (default)
#   ./scripts/build.sh --no-cuda --no-tests     # CPU-only static lib
#   ./scripts/build.sh --no-cuda --avx2 --no-avx512   # AVX2 only
#   ./scripts/build.sh --no-cuda --avx512        # base AVX-512 (Cascade Lake safe)
#   ./scripts/build.sh --no-cuda --avx512 --avx512-vbmi --avx512-vbmi2  # Ice Lake+
#   ./scripts/build.sh 2>&1 | tee build.log
#

set -euo pipefail

# ---- CLI parsing ----
CUDA_FLAGS="-DENABLE_CUDA=ON"      # GPU build is the default
TESTS_FLAGS="-DBUILD_TESTS=ON"     # tests + bench are the default
BUILD_DIR="build"
EXTRA_CMAKE_ARGS=()

usage() {
    sed -n '2,100p' "$0" | sed 's/^# \{0,1\}//'
}

# Map a "--feature/--no-feature" pair to "ENABLE_<NAME>".
# CLI flags are lowercase (--avx512, --avx512-vbmi2); the CMake cache
# entries use uppercase (ENABLE_AVX512, ENABLE_AVX512_VBMI2).
simd_opt() {
    local flag="$1"; shift
    local name="$1"
    local lower
    lower="$(printf '%s' "${name}" | tr '[:upper:]' '[:lower:]')"
    case "${flag}" in
        --"${lower}")    EXTRA_CMAKE_ARGS+=("-DENABLE_${name}=ON") ;;
        --no-"${lower}") EXTRA_CMAKE_ARGS+=("-DENABLE_${name}=OFF") ;;
        *) return 1 ;;
    esac
}

for arg in "$@"; do
    case "${arg}" in
        --cuda)
            CUDA_FLAGS="-DENABLE_CUDA=ON"
            ;;
        --no-cuda|--cpu)
            CUDA_FLAGS="-DENABLE_CUDA=OFF"
            ;;
        --no-tests)
            TESTS_FLAGS="-DBUILD_TESTS=OFF"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --sse2|--no-sse2)
            simd_opt "${arg}" "SSE2"
            ;;
        --avx|--no-avx)
            simd_opt "${arg}" "AVX"
            ;;
        --avx2|--no-avx2)
            simd_opt "${arg}" "AVX2"
            ;;
        --avx512|--no-avx512)
            simd_opt "${arg}" "AVX512"
            ;;
        --avx512-vbmi|--no-avx512-vbmi)
            simd_opt "${arg}" "AVX512_VBMI"
            ;;
        --avx512-vbmi2|--no-avx512-vbmi2)
            simd_opt "${arg}" "AVX512_VBMI2"
            ;;
        *)
            echo "ERROR: unknown argument '${arg}'" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

ROOT_DIR="$(readlink -f "$(dirname "$BASH_SOURCE")/..")"
cd "${ROOT_DIR}"

echo "=== TinyCoder Inference Engine Build ==="
echo "Root: ${ROOT_DIR}"
if [[ "${CUDA_FLAGS}" == "-DENABLE_CUDA=ON" ]]; then
    echo "Mode: GPU offload build (-DENABLE_CUDA=ON)"
else
    echo "Mode: CPU-only build (-DENABLE_CUDA=OFF)"
fi
if [[ ${#EXTRA_CMAKE_ARGS[@]} -gt 0 ]]; then
    echo "Extra CMake args:"
    printf '  %s\n' "${EXTRA_CMAKE_ARGS[@]}"
fi

# ---- 1. Configure ----
echo ""
echo "--- Step 1/3: Configuring CMake ---"
cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release ${CUDA_FLAGS} ${TESTS_FLAGS} \
    "${EXTRA_CMAKE_ARGS[@]}"

# ---- 2. Build ----
echo ""
echo "--- Step 2/3: Building ---"
cmake --build "${BUILD_DIR}" --config Release

# ---- 3. Run tests (when enabled) ----
if [[ "${TESTS_FLAGS}" == "-DBUILD_TESTS=ON" ]]; then
    echo ""
    echo "--- Step 3/3: Running unit tests ---"
    ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

echo ""
echo "=== Build complete ==="
