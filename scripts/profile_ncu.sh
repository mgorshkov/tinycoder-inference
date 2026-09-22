#!/bin/sh
# Nsight Compute profile of the TinyCoder decode GEMV + attention kernels.
#
# GPU performance counters are restricted to root by default (driver >= 441),
# so run this with sudo:
#     sudo bash scripts/profile_ncu.sh
# Output lands in /tmp/ncu_decode.txt (readable by the user).
#
# Sections:
#   SpeedOfLight     SM %, Memory %, DRAM/L1/L2 throughput  (is it BW-bound?)
#   SchedulerStats   issue-slot utilization, warps per scheduler
#   WarpStateStats   the stall breakdown (long scoreboard = DRAM latency,
#                    lg_throttle = LSU, mio = shared, selected = issue-bound)
#   Occupancy        achieved vs theoretical warps per SM

NCU=/opt/nvidia/nsight-compute/2025.4.1/ncu
[ -x "$NCU" ] || NCU=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
[ -x "$NCU" ] || { echo "ncu not found"; exit 1; }

MODEL=${1:-/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf}
OUT=${2:-/tmp/ncu_decode.txt}
CMD=${3:-}
HERE=$(cd "$(dirname "$0")/.." && pwd)

# Eager path (graph replay hides the individual launches from ncu).
export TINYCODER_GPU=1
export TINYCODER_GPU_GRAPH=0

cd "$HERE" || exit 1

if [ -n "$CMD" ]; then
    # Arbitrary command mode, e.g. profiling llama-bench for a head-to-head:
    #   sudo bash scripts/profile_ncu.sh <model> /tmp/ncu_llama.txt \
    #        "/path/llama-bench -m <model> -p 512 -n 128"
    eval "set -- $CMD"
    "$NCU" \
        --section SpeedOfLight \
        --kernel-name 'regex:mul_mat|quantize_row_q8_1|rms_norm|rope|silu' \
        --launch-count 32 \
        --target-processes all \
        "$@" > "$OUT" 2>&1
else
    # Full per-layer decode census (GEMVs + attention + elementwise + the
    # activation quantizes) so the llama.cpp head-to-head table is
    # apples-to-apples: SpeedOfLight ONLY, same as the llama capture.
    # The 6-token prompt's PREFILL consumes ~1(embed) + 28*6(elementwise)
    # + 2(LM-head quantize+GEMV) matching launches before the first decode
    # step, so skip that window.
    "$NCU" \
        --section SpeedOfLight \
        --kernel-name 'regex:kQGemv|kWarpAttentionSplit|kAttentionSplit|kAttnCombine|kQuantizeQ8|kRMSNormRow|kRoPEQ|kSiluMul|kAddResidual|kEmbed' \
        --launch-skip 160 \
        --launch-count 60 \
        --target-processes all \
        ./build-p1/unit_tests/tiny_logits_probe "$MODEL" "The capital of France is" \
            --generate --max-tokens 8 > "$OUT" 2>&1
fi

echo "wrote $OUT ($(wc -l < "$OUT") lines)"
grep -c "kQGemv\|kWarpAttention" "$OUT" | sed 's/^/matched kernel reports: /'
