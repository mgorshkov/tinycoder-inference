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
# Prerequisites:
#   - C++20 compiler (gcc >= 13, clang >= 14, or MSVC 2019+)
#   - CMake >= 3.18
#   - CUDA Toolkit >= 11.0 + nvcc (default GPU build only)
#
# Usage:
#   ./scripts/build.sh [--cuda | --no-cuda | --cpu | --no-tests | --help]
#
#   (default)  GPU build: -DENABLE_CUDA=ON
#   --cuda     Explicitly request the GPU build (same as default)
#   --no-cuda  CPU-only build: -DENABLE_CUDA=OFF
#   --cpu      Alias for --no-cuda
#   --no-tests Skip unit tests and benchmarks: -DBUILD_TESTS=OFF
#   --help     Show this help
#
# Examples:
#   ./scripts/build.sh                          # GPU offload build (default)
#   ./scripts/build.sh --no-cuda --no-tests     # CPU-only static lib
#   ./scripts/build.sh 2>&1 | tee build.log
#

set -euo pipefail

# ---- CLI parsing ----
CUDA_FLAGS="-DENABLE_CUDA=ON"      # GPU build is the default
TESTS_FLAGS="-DBUILD_TESTS=ON"     # tests + bench are the default
BUILD_DIR="build"

usage() {
    sed -n '2,41p' "$0" | sed 's/^# \{0,1\}//'
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
        *)
            echo "ERROR: unknown argument '${arg}'" >&2
            echo "Usage: ./scripts/build.sh [--cuda | --no-cuda | --cpu | --no-tests | --help]" >&2
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

# ---- 1. Configure ----
echo ""
echo "--- Step 1/3: Configuring CMake ---"
cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release ${CUDA_FLAGS} ${TESTS_FLAGS}

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
echo "Static library: ${BUILD_DIR}/libtinycoder_core.a"
