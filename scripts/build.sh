#!/bin/bash
#
# build.sh — Build the TinyCoder inference engine as a static library for
#            NVIDIA GPUs (Turing / Ampere / Ada, incl. RTX 2080 Ti / 3090 / 4090 /
#            5090) via CUDA, plus the unit tests and benchmarks when BUILD_TESTS=ON.
#
# This script:
#   0. Removes the existing build tree for a from-scratch build (default; pass
#      --no-clean / --incremental or TINYCODER_CLEAN=0 to reuse it).
#   1. Configures the CMake build tree.
#   2. Builds tinycoder_core (plus tinycoder_test and tinycoder_bench).
#   3. Runs CTest when the tests are enabled.
#
# CUDA (GPU offload) is the DEFAULT build path — the engine is developed and
# benchmarked against the reference GPU baseline with -DENABLE_CUDA=ON.  On a
# machine without nvcc/CUDA, pass --no-cuda (or --cpu) to request a CPU-only
# build instead of failing the configure.
#
# ENABLE_AVX2 and ENABLE_AVX512 both default to ON only when the host CPU reports
# the feature (CMakeLists.txt reads /proc/cpuinfo).  Both directions matter: a
# build that enables kernels the build machine cannot execute dies with SIGILL at
# startup, and a build that silently leaves AVX2 off compiles the AVX2 matvec TU
# to nothing and runs scalar kernels -- correct, but not comparable with the
# published numbers.  --avx2 / --no-avx2 and --avx512 / --no-avx512 still force
# either way for a cross-build; the resolved set is printed after configure.
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
#   --clean          Delete the build tree first (default)
#   --no-clean | --incremental   Reuse the existing build tree (fast rebuilds;
#                    a stale CMakeCache.txt survives, so use --clean after
#                    pulling changes to the build configuration)
#
# Parallelism:
#   TINYCODER_BUILD_DIR   Build tree (default: build)
#   TINYCODER_JOBS        Parallel compile jobs (default: all logical CPUs)
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
# Build tree: override with TINYCODER_BUILD_DIR (e.g. to keep a CPU and a CUDA
# tree side by side).
BUILD_DIR="${TINYCODER_BUILD_DIR:-build}"
# Clean by default: a reused tree keeps a stale CMakeCache.txt, which is how a
# cached CMAKE_CUDA_ARCHITECTURES=52 once turned a supported build into a wall
# of __dp4a errors.  --no-clean / --incremental / TINYCODER_CLEAN=0 opt out.
CLEAN="${TINYCODER_CLEAN:-1}"
EXTRA_CMAKE_ARGS=()

# ---- Parallelism ----
# Use as many cores as possible: default to the number of logical CPUs the OS
# reports (nproc), falling back to getconf/_SC_NPROCESSORS_ONLN.  Override with
# TINYCODER_JOBS=<n> (or JOBS) when you want to cap the parallelism.
detect_jobs() {
    local n=""
    if command -v nproc >/dev/null 2>&1; then
        n="$(nproc)"
    elif command -v getconf >/dev/null 2>&1; then
        n="$(getconf _NPROCESSORS_ONLN)"
    fi
    [[ "${n}" =~ ^[1-9][0-9]*$ ]] || n=1
    printf '%s' "${n}"
}
JOBS="${TINYCODER_JOBS:-${JOBS:-$(detect_jobs)}}"
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || JOBS=1

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
        --clean)
            CLEAN=1
            ;;
        --no-clean|--incremental)
            CLEAN=0
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

# Delete the build tree, but only something that is recognisably a build tree:
# a CMake cache, or a build*/ directory directly inside this repository.  A
# typo in TINYCODER_BUILD_DIR must never turn into an rm -rf of source.
clean_build_dir() {
    [[ -e "${BUILD_DIR}" ]] || return 0
    local abs
    abs="$(readlink -f -- "${BUILD_DIR}" 2>/dev/null || true)"
    if [[ -z "${abs}" || "${abs}" == "/" || "${abs}" == "${HOME}" || \
          "${abs}" == "${ROOT_DIR}" ]]; then
        echo "ERROR: refusing to delete '${BUILD_DIR}'" >&2
        exit 1
    fi
    if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        : # a CMake build tree, whatever it is called
    elif [[ "${abs}" == "${ROOT_DIR}/"* && "$(basename "${abs}")" == build* ]]; then
        : # a build*/ directory inside the repo (e.g. after a failed configure)
    else
        echo "ERROR: refusing to delete '${BUILD_DIR}': it is not a CMake" >&2
        echo "       build tree. Delete it manually if that is intended." >&2
        exit 1
    fi
    echo "Removing '${BUILD_DIR}' for a from-scratch build"
    rm -rf -- "${BUILD_DIR:?}"
}

echo "=== TinyCoder Inference Engine for NVIDIA GPUs (Turing / Ampere / Ada) Build ==="
echo "Root: ${ROOT_DIR}"
echo "Build dir: ${BUILD_DIR}"
echo "Clean build: $([[ "${CLEAN}" == 1 ]] && echo yes || echo "no (incremental)")"
echo "Parallel jobs: ${JOBS}"
if [[ "${CUDA_FLAGS}" == "-DENABLE_CUDA=ON" ]]; then
    echo "Mode: GPU offload build (-DENABLE_CUDA=ON)"
else
    echo "Mode: CPU-only build (-DENABLE_CUDA=OFF)"
fi
if [[ ${#EXTRA_CMAKE_ARGS[@]} -gt 0 ]]; then
    echo "Extra CMake args:"
    printf '  %s\n' "${EXTRA_CMAKE_ARGS[@]}"
fi

if [[ "${CLEAN}" == "1" ]]; then
    echo ""
    echo "--- Step 0/3: Removing the existing build tree ---"
    clean_build_dir
fi

# ---- 1. Configure ----
echo ""
echo "--- Step 1/3: Configuring CMake ---"
cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release ${CUDA_FLAGS} ${TESTS_FLAGS} \
    "${EXTRA_CMAKE_ARGS[@]}"

# Report the SIMD set the configure actually resolved.  The defaults follow the
# host, but a reused build tree keeps whatever its cache already holds, so a
# build log that does not print this cannot explain its own benchmark numbers.
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo ""
    echo "SIMD configuration (from ${BUILD_DIR}/CMakeCache.txt):"
    grep -E '^ENABLE_(AVX2|AVX512|AVX512_VBMI|AVX512_VBMI2):' \
        "${BUILD_DIR}/CMakeCache.txt" | sed 's/^/  /' || true
    if grep -q '^ENABLE_AVX2:BOOL=OFF' "${BUILD_DIR}/CMakeCache.txt"; then
        echo "  WARNING: AVX2 kernels are OFF -- CPU matvec runs scalar;"
        echo "           this build's CPU-path numbers are not comparable with"
        echo "           the published ones (pass --avx2 to enable them)."
    fi
fi

# ---- 2. Build ----
echo ""
echo "--- Step 2/3: Building (${JOBS} jobs) ---"
cmake --build "${BUILD_DIR}" --config Release --parallel "${JOBS}"

# ---- 3. Run tests (when enabled) ----
if [[ "${TESTS_FLAGS}" == "-DBUILD_TESTS=ON" ]]; then
    echo ""
    echo "--- Step 3/3: Running unit tests ---"
    ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

echo ""
echo "=== Build complete ==="
