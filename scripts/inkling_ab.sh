#!/usr/bin/env bash
# inkling_ab.sh — interleaved TinyCoder vs llama.cpp on Inkling-2-Mini IQ2_XXS.
#
# Runs the canonical pp512/tg128 row and the pp16/tg16 diagnostic in alternating
# pairs, so clock drift affects both engines equally.  The text gate runs first:
# a throughput number only exists if the checkpoint answers on the same tree.
#
# Baseline provenance: llama.cpp Unsloth `inkling2-loop-legacy-layout` (dd330cc)
# with the looped-layer `layer_phys` fix needed to run the Unsloth expanded
# layout.  Stock llama.cpp mainline cannot load this file.  Point LLAMA_BENCH at
# that build (default: llama-bench on PATH).
#
# The CPU row uses TINYCODER_NUMA=off and --threads 36.  The default NUMA policy
# reduces 36 to the 18 physical cores of the GPU's node, which is not the
# 36-thread comparison the table labels.
#
# Usage:
#   ./scripts/inkling_ab.sh [<IQ2_XXS.gguf>] [pairs]
#
# Environment:
#   TINYCODER_BENCH  tinycoder_bench path (default: ./build/benchmarks/...)
#   TEST_BIN         tinycoder_test path  (default: ./build/unit_tests/...)
#   LLAMA_BENCH      llama-bench path     (default: llama-bench on PATH)
#   THREADS          thread count         (default: 36)
#   SKIP_GATE=1      skip the text gate (diagnostic only; not a publishable run)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-/data/models/inkling/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf}"
PAIRS="${2:-3}"
BENCH="${TINYCODER_BENCH:-$ROOT/build/benchmarks/tinycoder_bench}"
TEST_BIN="${TEST_BIN:-$ROOT/build/unit_tests/tinycoder_test}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"
THREADS="${THREADS:-36}"

[ -x "$BENCH" ] || { echo "missing $BENCH (Release build)" >&2; exit 2; }
[ -x "$TEST_BIN" ] || { echo "missing $TEST_BIN" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "missing checkpoint $MODEL" >&2; exit 2; }
if ! command -v "$LLAMA_BENCH" >/dev/null 2>&1 && [ ! -x "$LLAMA_BENCH" ]; then
    echo "llama-bench not found: set LLAMA_BENCH to the Unsloth build" >&2
    exit 2
fi

# Never inherit an Inkling or GPU opt-in from the caller; each row sets exactly
# the switch it needs.
unset TINYCODER_INKLING_GPU TINYCODER_GPU

if [ "${SKIP_GATE:-0}" != "1" ]; then
    echo "== text gate (Inkling rows) =="
    if ! TEST_BIN="$TEST_BIN" "$ROOT/scripts/text_gate.sh" inkling; then
        echo "text gate failed; refusing to report throughput" >&2
        exit 1
    fi
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >>"$TMP"; }

# tinycoder_bench summary:
#   tg128 | 36 threads | 16.34 +/- 0.00 tok/s | ... || pp512 | 23.2 +/- 0.0 tok/s | ...
tc_run() { # $1 = env string, $2 = n-prompts, $3 = n-gen
    local envs="$1" pp="$2" tg="$3"
    # shellcheck disable=SC2086
    env $envs "$BENCH" --model "$MODEL" --n-prompts "$pp" --n-gen "$tg" \
        --threads "$THREADS" --reps 1 2>/dev/null |
        awk '
            $1 ~ /^tg[0-9]+$/ && $2 == "|" {
                k = 0
                for (i = 1; i <= NF; ++i) {
                    if ($i == "tok/s") {
                        if (++k == 1) tgv = $(i - 3)
                        else          ppv = $(i - 3)
                    }
                }
                if (ppv == "" || tgv == "") print "nan nan"
                else                        printf "%.2f %.2f\n", ppv, tgv
                exit
            }'
}

# llama-bench row:
#   | inkling2_loop ... | CUDA | 0 | pp512 | 322.72 ± 0.00 |
llama_run() { # $1 = -ngl, $2 = n-prompts, $3 = n-gen, $4 = env string
    local ngl="$1" pp="$2" tg="$3" envs="$4"
    # shellcheck disable=SC2086
    env $envs "$LLAMA_BENCH" -m "$MODEL" -p "$pp" -n "$tg" -t "$THREADS" \
        -ngl "$ngl" -r 1 2>/dev/null |
        awk -v pp="pp$pp" -v tg="tg$tg" '
            {
                for (i = 1; i <= NF; ++i) {
                    if ($i == pp) ppv = $(i + 2)
                    if ($i == tg) tgv = $(i + 2)
                }
            }
            END {
                if (ppv == "" || tgv == "") print "nan nan"
                else                        printf "%.2f %.2f\n", ppv, tgv
            }'
}

for p in $(seq 1 "$PAIRS"); do
    echo "== pair $p/$PAIRS =="

    read -r pp tg <<<"$(tc_run "TINYCODER_INKLING_GPU=1" 512 128)"
    record "TinyCoder GPU" "pp512/tg128" "$pp" "$tg"

    read -r pp tg <<<"$(llama_run 0 512 128 "")"
    record "llama.cpp CPU" "pp512/tg128" "$pp" "$tg"

    read -r pp tg <<<"$(llama_run 99 512 128 "LLAMA_ARG_CTX_SIZE=2048")"
    record "llama.cpp CUDA" "pp512/tg128" "$pp" "$tg"

    read -r pp tg <<<"$(tc_run "TINYCODER_INKLING_GPU=1" 16 16)"
    record "TinyCoder GPU" "pp16/tg16" "$pp" "$tg"

    read -r pp tg <<<"$(tc_run "TINYCODER_NUMA=off" 16 16)"
    record "TinyCoder CPU" "pp16/tg16" "$pp" "$tg"

    read -r pp tg <<<"$(llama_run 0 16 16 "")"
    record "llama.cpp CPU" "pp16/tg16" "$pp" "$tg"

    read -r pp tg <<<"$(llama_run 99 16 16 "LLAMA_ARG_CTX_SIZE=2048")"
    record "llama.cpp CUDA" "pp16/tg16" "$pp" "$tg"
done

echo
echo "mean [min-max] over $PAIRS interleaved pair(s), tok/s"
awk -F'\t' '
    {
        key = $1 "|" $2
        if (!(key in seen)) { seen[key] = 1; order[++n] = key }
        k = n_of[key]++
        spp[key] += $3; stg[key] += $4
        if (k == 0 || $3 < minpp[key]) minpp[key] = $3
        if (k == 0 || $3 > maxpp[key]) maxpp[key] = $3
        if (k == 0 || $4 < mintg[key]) mintg[key] = $4
        if (k == 0 || $4 > maxtg[key]) maxtg[key] = $4
    }
    END {
        printf "%-16s %-14s %10s %10s\n", "engine", "protocol", "pp", "tg"
        for (i = 1; i <= n; ++i) {
            key = order[i]; split(key, a, "|")
            c = n_of[key]
            printf "%-16s %-14s %7.2f [%6.2f-%6.2f] %7.2f [%6.2f-%6.2f]\n",
                   a[1], a[2], spp[key] / c, minpp[key], maxpp[key],
                   stg[key] / c, mintg[key], maxtg[key]
        }
    }' "$TMP"
