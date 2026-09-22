#!/usr/bin/env sh
# nsight-compute profile of the IQ PREFILL path (seqLen=512), mirroring the
# decode census.  Requires root:
#
#     sudo sh scripts/ncu_iq_prefill.sh
#
# The probe is run with a long prompt and --max-tokens 1 so the prefill
# dominates; the window skips the upload-time fp16-twin dequants.
# Outputs: /tmp/ncu_iqpf_<tag>.txt  (per-kernel SpeedOfLight summary)
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1

NCU="$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)"
[ -x "$NCU" ] || { echo "ERROR: ncu not found under /opt/nvidia/nsight-compute" >&2; exit 1; }

PROBE="$HERE/build-p1/unit_tests/tiny_logits_probe"
[ -x "$PROBE" ] || { echo "ERROR: $PROBE not found (build build-p1)" >&2; exit 1; }

export TINYCODER_GPU=1
export TINYCODER_GPU_GRAPH=0

# ~512-token prompt.
P=""
i=0
while [ $i -lt 26 ]; do P="$P The quick brown fox jumps over the lazy dog and writes concise C plus plus code."; i=$((i+1)); done

capture() {
    model="$1"; tag="$2"
    echo "== prefill $tag : $model"
    "$NCU" --section SpeedOfLight --print-summary per-kernel \
        --kernel-name 'regex:kDequant|gemm|xmma|kQuantize|kRMSNormRow|kWarpAttention|kEmbed|kSiluMul|kAddResidual|kQGemv|kRoPEQ|kAttnCombine' \
        --launch-skip 40 --launch-count 300 --target-processes all \
        "$PROBE" "$model" "$P" --generate --max-tokens 1 \
        > "/tmp/ncu_iqpf_${tag}.txt" 2>&1
    echo "   /tmp/ncu_iqpf_${tag}.txt ($(wc -l < "/tmp/ncu_iqpf_${tag}.txt") lines)"
}

capture /data/models/qwen/qwen2.5-coder-7b-instruct-iq3_xxs-imat.gguf 7b_xxs
capture /data/models/qwen/Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf        7b_iq2s
capture /data/models/qwen/qwen2.5-coder-1.5b-iq3_xxs-imat.gguf       1p5b_xxs

echo "done."
ls -la /tmp/ncu_iqpf_*.txt
