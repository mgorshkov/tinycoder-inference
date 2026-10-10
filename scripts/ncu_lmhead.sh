#!/usr/bin/env bash
# LM-head head-to-head ncu capture (requires root for GPU counters).
# Usage:  sudo bash -c 'nohup bash scripts/ncu_lmhead.sh > /tmp/ncu_lmhead.log 2>&1 &'
# Output: /tmp/ncu_lmhead_tiny.txt (kQGemvQ6KxQ8K_4xW, eager decode)
#         /tmp/ncu_lmhead_llama.txt (llama.cpp's lmhead kernel, biggest mul_mat_vec)
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1

NCU="$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)"
if [ -z "$NCU" ]; then
    echo "ERROR: ncu not found under /opt/nvidia/nsight-compute" >&2
    exit 1
fi

MODEL=/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf
BENCH=/home/mike/git/llama.cpp/build-cuda/bin/llama-bench

# --- TinyCoder side: eager decode (graph replay would still profile, but the
# eager path matches the census protocol), Q6_K LM head only.  Skip the
# prefill-final + first-token launches, capture 4 steady-state decode launches.
export TINYCODER_GPU=1
export TINYCODER_GPU_GRAPH=0
"$NCU" --section SpeedOfLight \
    --kernel-name regex:kQGemvQ6K \
    --launch-skip 2 --launch-count 4 --target-processes all \
    ./build-p1/unit_tests/tiny_logits_probe "$MODEL" "The capital of France is" --generate --max-tokens 8 \
    > /tmp/ncu_lmhead_tiny.txt 2>&1
echo "tiny side exit=$?"

# --- llama.cpp side: capture 60 mul_mat_vec-family launches of tg decode; the
# LM head is the (much) longest-duration kernel in the window.
"$NCU" --section SpeedOfLight \
    --kernel-name "regex:mul_mat_vec" \
    --launch-count 60 --target-processes all \
    "$BENCH" -m "$MODEL" -p 0 -n 64 \
    > /tmp/ncu_lmhead_llama.txt 2>&1
echo "llama side exit=$?"

ls -la /tmp/ncu_lmhead_tiny.txt /tmp/ncu_lmhead_llama.txt
