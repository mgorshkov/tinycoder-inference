#!/usr/bin/env sh
# Full nsight-compute profile of the IQ decode path, mirroring the q2_k
# ncu_ab census: per-launch SpeedOfLight census + per-launch stall/occupancy
# (WarpStateStats/SchedulerStats/Occupancy) on the dominant GEMV kernels.
# Requires root for GPU counters:
#
#     sudo sh scripts/ncu_iq.sh
#
# Outputs:
#   /tmp/ncu_iq_<tag>.txt        per-launch SoL decode census
#   /tmp/ncu_iq_<tag>_deep.txt   per-launch stall/occupancy on the IQ GEMVs
# with <tag> in {1p5b_xxs, 7b_iq2s, 7b_xxs}.
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1

NCU="$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)"
[ -x "$NCU" ] || { echo "ERROR: ncu not found under /opt/nvidia/nsight-compute" >&2; exit 1; }

PROBE="$HERE/build-p1/unit_tests/tiny_logits_probe"
[ -x "$PROBE" ] || { echo "ERROR: $PROBE not found (build build-p1)" >&2; exit 1; }

export TINYCODER_GPU=1
export TINYCODER_GPU_GRAPH=0

CENSUS_NAME='regex:kQGemv|kWarpAttentionSplit|kWarpAttention|kAttnCombine|kQuantizeQ8|kRMSNormRow|kRoPEQ|kSiluMul|kAddResidual|kEmbed'
# The kernels that carry the IQ decode: the Q8_K int family (IQ2_S/IQ3_XXS/
# IQ3_S), the IQ2_XS float family, and the tied/separate Q5_K LM head.
DEEP_NAME='regex:kQGemvQ8KFusedGU|kQGemvQ8K|kQGemvFusedGU_IQ2XS|kQGemvKxQ8K|kQGemv$|kEmbedDequantQ5K'
DEEP_SECTIONS="--section SpeedOfLight --section WarpStateStats --section SchedulerStats --section Occupancy"

capture() {
    model="$1"; tag="$2"
    echo "== $tag : $model"
    "$NCU" --section SpeedOfLight --kernel-name "$CENSUS_NAME" \
        --launch-skip 160 --launch-count 120 --target-processes all \
        "$PROBE" "$model" "The capital of France is" --generate --max-tokens 12 \
        > "/tmp/ncu_iq_${tag}.txt" 2>&1
    echo "   /tmp/ncu_iq_${tag}.txt ($(wc -l < "/tmp/ncu_iq_${tag}.txt") lines)"
    "$NCU" $DEEP_SECTIONS --kernel-name "$DEEP_NAME" \
        --launch-skip 160 --launch-count 24 --target-processes all \
        "$PROBE" "$model" "The capital of France is" --generate --max-tokens 12 \
        > "/tmp/ncu_iq_${tag}_deep.txt" 2>&1
    echo "   /tmp/ncu_iq_${tag}_deep.txt ($(wc -l < "/tmp/ncu_iq_${tag}_deep.txt") lines)"
}

capture /data/models/qwen/qwen2.5-coder-1.5b-iq3_xxs-imat.gguf       1p5b_xxs
capture /data/models/qwen/Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf        7b_iq2s
capture /data/models/qwen/qwen2.5-coder-7b-instruct-iq3_xxs-imat.gguf 7b_xxs

echo "done."
ls -la /tmp/ncu_iq_*.txt
