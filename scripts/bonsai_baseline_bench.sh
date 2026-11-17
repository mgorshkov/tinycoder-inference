#!/usr/bin/env bash
# bonsai_baseline_bench.sh — Interleaved A/B of TinyCoder vs the PrismML llama.cpp fork on the Bonsai checkpoints (the canonical runtime for this family; stock llama.cpp loads neither format).
# Three alternating interleaved pairs per checkpoint: TinyCoder vs the PrismML
# llama.cpp fork (the canonical runtime for this family -- stock llama.cpp
# refuses both formats), same file, serial, GPU verified empty before each run.
#
#   scripts/bonsai_baseline_bench.sh [outdir]
#
# Raw stdout of every run is kept in <outdir>/ so the parse can be re-derived
# without repeating the measurements; the means are printed at the end.
set -uo pipefail
OUT="${1:-/tmp/bonsai_ab}"
mkdir -p "$OUT"
TC=./build-1.0/benchmarks/tinycoder_bench
# The reference runtime is the PrismML fork (stock llama.cpp loads neither Bonsai
# packing).  Override LLAMA_BENCH if it lives elsewhere; the default assumes the
# usual side-by-side checkout next to this repo.
LB="${LLAMA_BENCH:-$HOME/git/llama-prism/build/bin/llama-bench}"
PQ2=/data/models/bonsai/Ternary-Bonsai-2-27B-PQ2_0.gguf
PTQ1=/data/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf

check_gpu() {
  # A process can linger for a few seconds after its run; wait it out, then
  # refuse to measure if something is still resident (never share the GPU).
  for _ in $(seq 1 15); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
    sleep 2
  done
  local apps; apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [ -n "$apps" ]; then echo "GPU BUSY ($apps) - refusing to measure" >&2; exit 1; fi
}

for spec in "PQ2:$PQ2" "PTQ1:$PTQ1"; do
  tag="${spec%%:*}"; m="${spec#*:}"
  for pair in 1 2 3; do
    check_gpu
    timeout 1800 "$TC" --model "$m" --n-prompts 512 --n-gen 128 --reps 1 \
        --threads 36 > "$OUT/tc_${tag}_${pair}.log" 2>&1
    check_gpu
    timeout 1800 "$LB" -m "$m" -p 512 -n 128 -t 36 -r 1 \
        > "$OUT/lb_${tag}_${pair}.log" 2>&1
    echo "pair $pair $tag done"
  done
done
python3 scripts/bonsai_ab_summary.py "$OUT"
