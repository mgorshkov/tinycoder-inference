#!/usr/bin/env bash
# Deep ncu profile (stall/occupancy/instruction sections) of ONLY the two
# dp4a mmvq-family decode GEMVs, on either side of the head-to-head.
# Usage:
#   sudo bash -c 'nohup bash scripts/ncu_deep.sh tiny > /tmp/ncu_deep_tiny.log 2>&1 &'
#   sudo bash -c 'nohup bash scripts/ncu_deep.sh llama > /tmp/ncu_deep_llama.log 2>&1 &'
# Outputs: /tmp/ncu_deep_tiny.txt / /tmp/ncu_deep_llama.txt
set -u

MODE=${1:-tiny}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1

NCU="$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)"
[ -x "$NCU" ] || { echo "ncu not found"; exit 1; }

SECTIONS=(--section SpeedOfLight --section Occupancy --section SchedulerStats
          --section WarpStateStats --section InstructionStats)

MODEL=/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf

if [ "$MODE" = "tiny" ]; then
    OUT=/tmp/ncu_deep_tiny.txt
    export TINYCODER_GPU=1
    export TINYCODER_GPU_GRAPH=0
    "$NCU" "${SECTIONS[@]}" \
        --kernel-name 'regex:kQGemvQ3KxQ81_Mmvq|kQGemvQ2KxQ81_GLU' \
        --launch-skip 160 \
        --launch-count 8 \
        --target-processes all \
        ./build-p1/unit_tests/tiny_logits_probe "$MODEL" "The capital of France is" \
            --generate --max-tokens 8 > "$OUT" 2>&1
else
    OUT=/tmp/ncu_deep_llama.txt
    BENCH=/home/mike/git/llama.cpp/build-cuda/bin/llama-bench
    "$NCU" "${SECTIONS[@]}" \
        --kernel-name 'regex:mul_mat_vec_q' \
        --launch-skip 6 \
        --launch-count 12 \
        --target-processes all \
        "$BENCH" -m "$MODEL" -p 0 -n 128 > "$OUT" 2>&1
fi

RC=$?
echo "mode=$MODE ncu exit=$RC wrote $OUT ($(wc -l < "$OUT") lines)"
exit $RC
