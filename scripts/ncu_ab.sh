#!/usr/bin/env bash
# Apples-to-apples ncu head-to-head: TinyCoder vs the reference engine, EAGER
# decode, same GPU, same model, SpeedOfLight + per-kernel summary on both sides.
#
# GPU performance counters are root-only (driver >= 441), so run with sudo:
#     sudo bash scripts/ncu_ab.sh
#
# Outputs (readable by the user after the run):
#   /tmp/ncu_ab_tiny.txt         TinyCoder decode census  (per-kernel summary)
#   /tmp/ncu_ab_ref.txt          reference decode census  (per-kernel summary)
#   /tmp/ncu_ab_tiny_lmhead.txt  TinyCoder LM-head detail (per-launch)
#   /tmp/ncu_ab_ref_lmhead.txt   reference LM-head detail (per-launch)
#   /tmp/ncu_ab_tiny_deep.txt    TinyCoder stall/occupancy census (per-kernel)
#   /tmp/ncu_ab_ref_deep.txt     reference stall/occupancy census (per-kernel)
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1

NCU="$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)"
[ -x "$NCU" ] || { echo "ERROR: ncu not found under /opt/nvidia/nsight-compute" >&2; exit 1; }

MODEL=/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf
BENCH=/home/mike/git/llama.cpp/build-cuda/bin/llama-bench
[ -x "$BENCH" ] || BENCH=/home/mike/git/llama.cpp/build/bin/llama-bench
[ -x "$BENCH" ] || { echo "ERROR: reference benchmark not found" >&2; exit 1; }
PROBE="$HERE/build-p1/unit_tests/tiny_logits_probe"
[ -x "$PROBE" ] || { echo "ERROR: $PROBE not found (build build-p1)" >&2; exit 1; }

# Eager path (CUDA-graph replay hides the individual launches from ncu).
export TINYCODER_GPU=1
export TINYCODER_GPU_GRAPH=0

echo "[1/6] TinyCoder decode census (per-launch rows, eager)..."
"$NCU" --section SpeedOfLight \
    --kernel-name 'regex:kQGemv|kWarpAttentionSplit|kWarpAttention|kAttnCombine|kQuantizeQ8|kRMSNormRow|kRoPEQ|kSiluMul|kAddResidual|kEmbed' \
    --launch-skip 160 --launch-count 120 --target-processes all \
    "$PROBE" "$MODEL" "The capital of France is" --generate --max-tokens 12 \
    > /tmp/ncu_ab_tiny.txt 2>&1
echo "  -> /tmp/ncu_ab_tiny.txt ($(wc -l < /tmp/ncu_ab_tiny.txt) lines)"

echo "[2/6] reference decode census (per-kernel summary, eager)..."
"$NCU" --section SpeedOfLight --print-summary per-kernel \
    --kernel-name 'regex:mul_mat|quantize_row|rms_norm|rope|silu|soft_max' \
    --launch-count 120 --target-processes all \
    "$BENCH" -m "$MODEL" -p 0 -n 8 \
    > /tmp/ncu_ab_ref.txt 2>&1
echo "  -> /tmp/ncu_ab_ref.txt ($(wc -l < /tmp/ncu_ab_ref.txt) lines)"

echo "[3/6] TinyCoder LM head detail (per-launch)..."
"$NCU" --section SpeedOfLight \
    --kernel-name regex:kQGemvQ6K \
    --launch-skip 2 --launch-count 4 --target-processes all \
    "$PROBE" "$MODEL" "The capital of France is" --generate --max-tokens 8 \
    > /tmp/ncu_ab_tiny_lmhead.txt 2>&1
echo "  -> /tmp/ncu_ab_tiny_lmhead.txt ($(wc -l < /tmp/ncu_ab_tiny_lmhead.txt) lines)"

echo "[4/6] reference LM head detail (per-launch, widest mul_mat window)..."
"$NCU" --section SpeedOfLight \
    --kernel-name 'regex:mul_mat' \
    --launch-count 260 --target-processes all \
    "$BENCH" -m "$MODEL" -p 0 -n 16 \
    > /tmp/ncu_ab_ref_lmhead.txt 2>&1
echo "  -> /tmp/ncu_ab_ref_lmhead.txt ($(wc -l < /tmp/ncu_ab_ref_lmhead.txt) lines)"

# Deep stall/occupancy census on the decode body kernels.  The 2026-10-10
# LG-throttle campaign fixed the byte-load disease, so this is the first fresh
# stall breakdown on the wide-load build -- it decides whether the remaining
# per-stage residue is issue-, latency- or LG-throttle-bound (and where to aim
# the next lever).  Sections: SoL + WarpStateStats + SchedulerStats + Occupancy.
DEEP_SECTIONS="--section SpeedOfLight --section WarpStateStats --section SchedulerStats --section Occupancy"
echo "[5/6] TinyCoder deep stall census (body GEMVs + LM head + glue)..."
"$NCU" $DEEP_SECTIONS --print-summary per-kernel \
    --kernel-name 'regex:kQGemvFusedQKV|kQGemvQ2KxQ81_GLU|kQGemvQ3KxQ81_Mmvq|kQGemvQ6KxQ8K_4xW|kAttnCombinePartialW4|kRMSNormRowQ81|kRoPEQ|kQuantizeQ8' \
    --launch-skip 160 --launch-count 60 --target-processes all \
    "$PROBE" "$MODEL" "The capital of France is" --generate --max-tokens 8 \
    > /tmp/ncu_ab_tiny_deep.txt 2>&1
echo "  -> /tmp/ncu_ab_tiny_deep.txt ($(wc -l < /tmp/ncu_ab_tiny_deep.txt) lines)"

echo "[6/6] reference deep stall census (mmvq GEMVs)..."
"$NCU" $DEEP_SECTIONS --print-summary per-kernel \
    --kernel-name 'regex:mul_mat_vec_q|mul_mat_vec_f' \
    --launch-skip 4 --launch-count 48 --target-processes all \
    "$BENCH" -m "$MODEL" -p 0 -n 64 \
    > /tmp/ncu_ab_ref_deep.txt 2>&1
echo "  -> /tmp/ncu_ab_ref_deep.txt ($(wc -l < /tmp/ncu_ab_ref_deep.txt) lines)"

echo "done."
ls -la /tmp/ncu_ab_*.txt
