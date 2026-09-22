#!/usr/bin/env sh
# Finish the apples-to-apples ncu run.  The main ncu_ab.sh run had already
# produced a valid reference census (/tmp/ncu_ab_ref.txt), a valid TinyCoder
# LM-head SpeedOfLight window (/tmp/ncu_ab_tiny_lmhead.txt) and a valid
# reference LM-head window (/tmp/ncu_ab_ref_lmhead.txt), then died on a
# bash-only array under `sh`.  This script produces the three remaining
# captures needed to pick the next lever:
#
#   sudo sh scripts/ncu_ab_finish.sh
#
#   /tmp/ncu_ab_tiny.txt        TinyCoder decode census, per-launch (HEAD probe)
#   /tmp/ncu_ab_tiny_deep.txt   TinyCoder LM-head stall/occupancy (the lever)
#   /tmp/ncu_ab_tiny_body.txt   TinyCoder body-GEMV stall/occupancy census
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1

NCU="$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)"
[ -x "$NCU" ] || { echo "ERROR: ncu not found under /opt/nvidia/nsight-compute" >&2; exit 1; }

MODEL=/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf
PROBE="$HERE/build-p1/unit_tests/tiny_logits_probe"
[ -x "$PROBE" ] || { echo "ERROR: $PROBE not found (build build-p1)" >&2; exit 1; }

export TINYCODER_GPU=1
export TINYCODER_GPU_GRAPH=0

DEEP_SECTIONS="--section SpeedOfLight --section WarpStateStats --section SchedulerStats --section Occupancy"

echo "[1/3] TinyCoder decode census (per-launch rows, eager, HEAD probe)..."
"$NCU" --section SpeedOfLight \
    --kernel-name 'regex:kQGemv|kWarpAttentionSplit|kWarpAttention|kAttnCombine|kQuantizeQ8|kRMSNormRow|kRoPEQ|kSiluMul|kAddResidual|kEmbed' \
    --launch-skip 160 --launch-count 120 --target-processes all \
    "$PROBE" "$MODEL" "The capital of France is" --generate --max-tokens 12 \
    > /tmp/ncu_ab_tiny.txt 2>&1
echo "  -> /tmp/ncu_ab_tiny.txt ($(wc -l < /tmp/ncu_ab_tiny.txt) lines)"

# The Q6_K LM head is the only stage still behind llama (60.1% vs 74.5% DRAM).
# Skip past the prefill-final + first decode launches (their LM-head GEMVs are
# cold / partly prefilled) and profile 6 steady-state ones with the stall
# sections.
echo "[2/3] TinyCoder LM-head deep stall census (kQGemvQ6KxQ8K_4xW)..."
"$NCU" $DEEP_SECTIONS \
    --kernel-name regex:kQGemvQ6K \
    --launch-skip 2 --launch-count 6 --target-processes all \
    "$PROBE" "$MODEL" "The capital of France is" --generate --max-tokens 8 \
    > /tmp/ncu_ab_tiny_deep.txt 2>&1
echo "  -> /tmp/ncu_ab_tiny_deep.txt ($(wc -l < /tmp/ncu_ab_tiny_deep.txt) lines)"

echo "[3/3] TinyCoder body-GEMV deep stall census (QKV/GLU/down/combine/glue)..."
"$NCU" $DEEP_SECTIONS --print-summary per-kernel \
    --kernel-name 'regex:kQGemvFusedQKV|kQGemvQ2KxQ81_GLU|kQGemvQ3KxQ81_Mmvq|kAttnCombinePartialW4|kRMSNormRowQ81|kRoPEQ|kQuantizeQ8' \
    --launch-skip 160 --launch-count 60 --target-processes all \
    "$PROBE" "$MODEL" "The capital of France is" --generate --max-tokens 8 \
    > /tmp/ncu_ab_tiny_body.txt 2>&1
echo "  -> /tmp/ncu_ab_tiny_body.txt ($(wc -l < /tmp/ncu_ab_tiny_body.txt) lines)"

echo "done."
ls -la /tmp/ncu_ab_*.txt
