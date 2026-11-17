#!/usr/bin/env bash
# Steady-state per-token budget for the qwen35 decode path.
# Usage: decode_budget.sh <model.gguf> ["ENV=VAL ..."] [--assert-tail<MS>
#                          | --assert-shape-ms<MS]
set -euo pipefail
model=${1:?usage: decode_budget.sh <model.gguf> [env-string] [assert]}
envs=${2:-}
assert=${3:-}
out=$(mktemp)
trap 'rm -f "$out"' EXIT
env $envs TINYCODER_STAGE_TRACE=1 ./build-1.0/benchmarks/tinycoder_bench \
  --model "$model" --n-prompts 1 --n-gen 16 --reps 1 --threads 36 >"$out" 2>&1 \
  || { echo "bench failed"; tail -5 "$out"; exit 1; }
# the last trace line is the steady state (the first decode token is cold)
grep '^\[q35trace\]' "$out" | tail -2 | sed -n '1p;2p'
case "$assert" in
  --assert-tail\<*)
    lim=${assert#--assert-tail<}
    tail_ms=$(grep 'ms/token' "$out" | tail -1 | sed -E 's/.*tail=([0-9.]+).*/\1/')
    awk -v a="$tail_ms" -v b="$lim" 'BEGIN{exit !(a<b)}' \
      && echo "PASS tail=${tail_ms} < ${lim}" \
      || { echo "FAIL tail=${tail_ms} >= ${lim}"; exit 1; }
    ;;
  --assert-shape-ms\<*)
    lim=${assert#--assert-shape-ms<}
    ms=$(grep 'ms/token' "$out" | tail -1 | sed -E 's/.*proj=([0-9.]+).*ffn=([0-9.]+).*/\1 \2/' | awk '{print $1+$2}')
    awk -v a="$ms" -v b="$lim" 'BEGIN{exit !(a<b)}' \
      && echo "PASS proj+ffn=${ms} < ${lim}" \
      || { echo "FAIL proj+ffn=${ms} >= ${lim}"; exit 1; }
    ;;
esac
