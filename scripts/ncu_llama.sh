#!/usr/bin/env bash
# Profile llama-bench decode kernels with ncu (requires root for GPU counters).
# Usage:  sudo bash scripts/ncu_llama.sh
# Output: /tmp/ncu_llama.txt
set -u

NCU="$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)"
if [ -z "$NCU" ]; then
    echo "ERROR: ncu not found under /opt/nvidia/nsight-compute" >&2
    exit 1
fi

BENCH=/home/mike/git/llama.cpp/build-cuda/bin/llama-bench
MODEL=/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf

"$NCU" \
    --section SpeedOfLight \
    --kernel-name "regex:mul_mat|quantize_row_q8_1|rms_norm|rope|silu" \
    --launch-count 32 \
    --target-processes all \
    "$BENCH" -m "$MODEL" -p 0 -n 128 \
    > /tmp/ncu_llama.txt 2>&1

RC=$?
echo "ncu exit=$RC"
ls -la /tmp/ncu_llama.txt
exit $RC
