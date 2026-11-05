#!/usr/bin/env bash
# A/B the cublasLt EXL3 prefill GEMM against the exact-order cublasGemmEx path.
#
# The knob is TINYCODER_EXL3_LT: ON is the shipped default (2026-11-14), 0 is the
# escape hatch that reproduces the bit-exact 19.308743 logit. See AGENTS.md gate 2.
#
#   bash scripts/exl3_lt_ab.sh              # 3 bench runs per arm + both probes
#   RUNS=1 bash scripts/exl3_lt_ab.sh       # quick check
#
# Two things this script exists to get right, both of which produced false
# numbers before:
#
#  1. SERIAL GPU USE. Two 27B processes on an 11 GB card do not merely contend,
#     the second one fails `cudaMalloc` during upload and falls back to the CPU
#     path, which then segfaults. So every step waits for the card to be idle.
#     Detection uses `pgrep -x` on the binary names, never `pgrep -f` on a
#     pattern that also matches this script.
#
#  2. ATTRIBUTION. Whole-prefill tok/s cannot tell the GEMM from the
#     materialiser that overlaps it. TINYCODER_EXL3_PHASE=1 records five events
#     per projection and prints one line at exit; PHASE_SKIP drops the cold
#     warmup prefill (which is also where the cublasLt autotune runs), leaving
#     exactly one 512-token prefill. 409 projections is that prefill.
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1

MODEL="${TINYCODER_MODEL_PATH:-/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw}"
BENCH=./build-1.0/benchmarks/tinycoder_bench
PROBE=./build-1.0/unit_tests/tiny_logits_probe
RUNS="${RUNS:-3}"
SKIP="${PHASE_SKIP:-410}"

[ -x "$BENCH" ] || { echo "missing $BENCH -- build tinycoder_core + tinycoder_bench"; exit 1; }

idle() {
    for _ in $(seq 1 120); do
        if ! pgrep -x tinycoder_bench >/dev/null &&
           ! pgrep -x tiny_logits_probe >/dev/null &&
           ! pgrep -x tinycoder_test >/dev/null &&
           ! pgrep -x exl3_gpu_parity >/dev/null; then
            sleep 5
            return 0
        fi
        sleep 5
    done
    echo "!! GPU never went idle; a contended run describes neither arm" >&2
    return 1
}

echo "== logits, both paths =="
for lt in 0 1; do
    idle || exit 1
    echo "-- TINYCODER_EXL3_LT=$lt"
    TINYCODER_EXL3_LT=$lt "$PROBE" "$MODEL" "def hello():" 2>/dev/null |
        grep -E "top[1-5]:" | sed "s/^/   LT=$lt/"
done

echo
echo "== per-prefill attribution, 409 projections each =="
for lt in 0 1; do
    idle || exit 1
    TINYCODER_EXL3_LT=$lt TINYCODER_EXL3_PHASE=1 TINYCODER_EXL3_PHASE_SKIP=$SKIP \
        "$BENCH" --model "$MODEL" --n-prompts 512 --n-gen 128 --reps 1 --threads 36 \
        2>&1 | grep -E "exl3phase|pp512 \|" | sed "s/^/   LT=$lt  /"
done

echo
echo "== $RUNS bench runs per arm (the only comparable protocol) =="
for lt in 0 1; do
    for r in $(seq 1 "$RUNS"); do
        idle || exit 1
        out=$(TINYCODER_EXL3_LT=$lt "$BENCH" --model "$MODEL" \
                  --n-prompts 512 --n-gen 128 --reps 1 --threads 36 2>&1 |
                  grep -oE "pp512 \| [0-9.]+ .*" | tail -1)
        echo "   LT=$lt run$r  $out"
    done
done

echo
echo "Reminder: LT=1 is the default. Its top-1 logit is 19.3090 +-2e-3 with id=198"
echo "and the top-5 unchanged; LT=0 is bit-exact at 19.308743. Gate on the id and"
echo "the ordering, not on the value."
