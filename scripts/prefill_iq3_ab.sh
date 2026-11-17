#!/usr/bin/env sh
# prefill_iq3_ab.sh — Interleaved prefill A/B of TinyCoder vs llama.cpp on IQ3 checkpoints, one pair at a time.
# Paths are relative to this checkout; override with TINYCODER_ROOT / LLAMA_DIR.
ROOT="${TINYCODER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/../llama.cpp}"
PRISM_DIR="${PRISM_DIR:-$ROOT/../llama-prism}"
# Interleaved prefill A/B: TinyCoder vs llama.cpp, one pair at a time.
#
#   sh scripts/prefill_iq3_ab.sh <model.gguf> [pairs]
#
# The comparable protocol:
#   TinyCoder  --n-prompts 512 --n-gen 128 --reps 1 --threads 36
#   llama.cpp  llama-bench -p 512 -n 128 -t 36 -r 1  (plus $LLAMA_EXTRA)
#
# LLAMA_EXTRA is empty by default.  An earlier version of this script passed
# "-b 2048 -ub 2048" because a single default-flag llama run read 8,742 against
# 10,169 with those flags on the 1.5B IQ3_XXS.  That was NOT a flag effect: the
# slow run was the first process after an idle period and sampled the GPU at
# 300 MHz (nvml), the flag runs at 1,965 MHz.  Repeating the default invocation
# three times in a row gives 8,246 (300 MHz) then 10,022 and 10,047 (1,965 MHz),
# so a WARM-UP pass is what matters.  Both engines are now warmed before pair 1
# (see warm) and the alternation keeps the clock common to both columns.
#
# The 2026-11-16 baseline rows were measured this way: llama pp512 ~10,02x-10,19x,
# matching the published 10,138.6.
#
# The engines alternate inside every pair (ours, llama, llama, ours, ...) so a
# clock drift over the batch hits both columns equally, and the batch refuses to
# start if a benchmark process is already resident: two processes on this GPU
# produce numbers that describe neither.
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE" || exit 1

MODEL="${1:?usage: $0 <model.gguf> [pairs]}"
PAIRS="${2:-3}"
LLAMA="${LLAMA_BENCH:-${LLAMA_DIR}/build-cuda/bin/llama-bench}"
LLAMA_EXTRA="${LLAMA_EXTRA-}"
BENCH="$HERE/build-1.0/benchmarks/tinycoder_bench"

[ -x "$BENCH" ] || { echo "ERROR: $BENCH missing" >&2; exit 1; }
[ -x "$LLAMA" ] || { echo "ERROR: $LLAMA missing" >&2; exit 1; }

log=/tmp/prefill_ab_$(date +%H%M%S).log
: > "$log"
T=$(mktemp -d)

norm() { sed -n 's/^ *\([0-9][0-9]*\.[0-9]*\).*/\1/p'; }

ours() {  # $1 = label
    pgrep -x tinycoder_bench >/dev/null && { echo "BUSY: tinycoder_bench resident" >&2; return 1; }
    out=$("$BENCH" --model "$MODEL" --n-prompts 512 --n-gen 128 --reps 1 --threads 36 2>/dev/null \
          | grep -E 'tg128 \|' | tail -1)
    pp=$(printf '%s' "$out" | sed -n 's/.*|| pp512 | \([0-9.]*\).*/\1/p')
    tg=$(printf '%s' "$out" | sed -n 's/^tg128 | [0-9]* threads | \([0-9.]*\).*/\1/p')
    printf 'ours  %-6s pp512=%s  tg128=%s\n' "$1" "$pp" "$tg" | tee -a "$log"
    printf '%s\n' "$pp" >> "$T/ours_pp"
    printf '%s\n' "$tg" >> "$T/ours_tg"
}

llama() {  # $1 = label
    # shellcheck disable=SC2086
    out=$("$LLAMA" -m "$MODEL" -p 512 -n 128 -t 36 -r 1 $LLAMA_EXTRA 2>/dev/null \
          | grep -E '\| *(pp512|tg128) *\|')
    pp=$(printf '%s' "$out" | grep pp512 | sed 's/.*| *\([0-9][0-9]*\.[0-9]*\).*/\1/')
    tg=$(printf '%s' "$out" | grep tg128 | sed 's/.*| *\([0-9][0-9]*\.[0-9]*\).*/\1/')
    printf 'llama %-6s pp512=%s  tg128=%s\n' "$1" "$pp" "$tg" | tee -a "$log"
    printf '%s\n' "$pp" >> "$T/llama_pp"
    printf '%s\n' "$tg" >> "$T/llama_tg"
}

pgrep -x tinycoder_bench >/dev/null && { echo "ERROR: tinycoder_bench already running" >&2; exit 1; }

# Warm-up: leave the idle DVFS state before the first measured pair, or pair 1
# reads ~16% low on the engine that runs first (idle is 300 MHz, boost 1,965).
clk() { nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1; }
warm() {
    "$BENCH" --model "$MODEL" --n-prompts 512 --n-gen 8 --reps 1 --threads 36 >/dev/null 2>&1
    # shellcheck disable=SC2086
    "$LLAMA" -m "$MODEL" -p 512 -n 8 -t 36 -r 1 $LLAMA_EXTRA >/dev/null 2>&1
}
warm
warm
echo "warm clock: $(clk) MHz"

i=1
while [ "$i" -le "$PAIRS" ]; do
    if [ $((i % 2)) -eq 1 ]; then
        ours "$i" || exit 1
        llama "$i"
    else
        llama "$i"
        ours "$i" || exit 1
    fi
    i=$((i + 1))
done

echo "----"
awk -v tag="ours  pp512" '{s+=$1; if(n==0||$1>mx)mx=$1; if(n==0||$1<mn)mn=$1; n++}
     END{printf "%-12s mean %.1f  (%s)  spread %.2f%%\n", tag, s/n, "", (mx-mn)/((mx+mn)/2)*100}' "$T/ours_pp"
awk -v tag="llama pp512" '{s+=$1; if(n==0||$1>mx)mx=$1; if(n==0||$1<mn)mn=$1; n++}
     END{printf "%-12s mean %.1f  (%s)  spread %.2f%%\n", tag, s/n, "", (mx-mn)/((mx+mn)/2)*100}' "$T/llama_pp"
awk -v tag="ours  tg128" '{s+=$1; if(n==0||$1>mx)mx=$1; if(n==0||$1<mn)mn=$1; n++}
     END{printf "%-12s mean %.2f  spread %.2f%%\n", tag, s/n, (mx-mn)/((mx+mn)/2)*100}' "$T/ours_tg"
awk -v tag="llama tg128" '{s+=$1; if(n==0||$1>mx)mx=$1; if(n==0||$1<mn)mn=$1; n++}
     END{printf "%-12s mean %.2f  spread %.2f%%\n", tag, s/n, (mx-mn)/((mx+mn)/2)*100}' "$T/llama_tg"
paste "$T/ours_pp" "$T/llama_pp" | awk '{r=$1/$2; s+=r; n++}
     END{if(n) printf "ours/llama pp512 = %.4f  (%+.2f%%)\n", s/n, (s/n-1)*100}'
echo "log: $log  (raw: $T)"
