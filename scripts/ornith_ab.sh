#!/usr/bin/env bash
# ornith_ab.sh — Interleaved A/B of TinyCoder vs llama-bench on one checkpoint (3 alternating pairs, means, spread).
# Paths are relative to this checkout; override with TINYCODER_ROOT / LLAMA_DIR.
ROOT="${TINYCODER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/../llama.cpp}"
PRISM_DIR="${PRISM_DIR:-$ROOT/../llama-prism}"
# Interleaved A/B of TinyCoder vs llama-bench on one checkpoint.
#
# Protocol (AGENTS.md): default environment, no env knobs, 3 alternating
# interleaved pairs, `--n-prompts 512 --n-gen 128 --reps 1 --threads 36` on the
# TinyCoder side and `-p 512 -n 128 -t 36 -r 1` on the llama side, means and
# spread reported.  One GPU owner at a time: before every single run the script
# checks that `nvidia-smi --query-compute-apps=pid` is empty and aborts
# otherwise -- two processes on this GPU produce numbers that describe neither.
#
#   scripts/ornith_ab.sh <checkpoint> [llama extra args...]
#   scripts/ornith_ab.sh /data/models/ornith/Ornith-1.5-35B-A3B-Q8_0.gguf -ncmoe 99
#
# Raw output of every run is kept (TINYCODER_LOG / LLAMA_LOG, default under
# /tmp) so the spread can be re-derived without re-measuring.
set -uo pipefail

MODEL="${1:?usage: $0 <checkpoint> [llama extra args...]}"
shift || true
LLAMA_ARGS=("$@")

BENCH="${TINYCODER_BENCH:-$ROOT/build/benchmarks/tinycoder_bench}"
LLAMA_BENCH="${LLAMA_BENCH:-${LLAMA_DIR}/build-cuda/bin/llama-bench}"
LLAMA_LIB="${LLAMA_LIB:-${LLAMA_DIR}/build-cuda/bin}"
PAIRS="${PAIRS:-3}"
TAG="$(basename "$MODEL")"
TINYCODER_LOG="${TINYCODER_LOG:-/tmp/ab_tiny_${TAG}.log}"
LLAMA_LOG="${LLAMA_LOG:-/tmp/ab_llama_${TAG}.log}"

[ -x "$BENCH" ] || { echo "missing $BENCH" >&2; exit 2; }
[ -x "$LLAMA_BENCH" ] || { echo "missing $LLAMA_BENCH" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "missing model $MODEL" >&2; exit 2; }
: >"$TINYCODER_LOG"
: >"$LLAMA_LOG"

require_idle_gpu() {
    local pids
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
    if [ -n "$pids" ]; then
        echo "[ab] waiting for GPU to go idle (held by: $pids)" >&2
        for _ in $(seq 1 120); do
            sleep 10
            pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
            [ -z "$pids" ] && return 0
        done
        echo "[ab] ABORT: GPU still busy: $pids" >&2
        exit 3
    fi
}

SIZE_BYTES=$(stat -c%s "$MODEL")
echo "[ab] model=$MODEL size=$((SIZE_BYTES / 1024 / 1024)) MiB pairs=$PAIRS llama_args=${LLAMA_ARGS[*]:-none}"

for i in $(seq 1 "$PAIRS"); do
    require_idle_gpu
    echo "[ab] === pair $i/$PAIRS: tinycoder ==="
    "$BENCH" --model "$MODEL" --n-prompts 512 --n-gen 128 --reps 1 --threads 36 \
        >>"$TINYCODER_LOG" 2>&1
    grep -E "^(pp512|tg128) \|" "$TINYCODER_LOG" | tail -2 | sed 's/^/    /'
    require_idle_gpu
    echo "[ab] === pair $i/$PAIRS: llama-bench ==="
    ( cd "$LLAMA_LIB" && LD_LIBRARY_PATH="$LLAMA_LIB" "$LLAMA_BENCH" -m "$MODEL" \
        -p 512 -n 128 -t 36 -r 1 "${LLAMA_ARGS[@]}" >>"$LLAMA_LOG" 2>&1 )
    grep -E "\| *(pp512|tg128) *\|" "$LLAMA_LOG" | tail -2 | sed 's/^/    /'
done

echo
echo "[ab] ---- summary: 3-run means over $PAIRS interleaved pairs ----"
python3 - "$TINYCODER_LOG" "$LLAMA_LOG" "$SIZE_BYTES" <<'PY'
import re, statistics, sys
tlog, llog, size = sys.argv[1], sys.argv[2], int(sys.argv[3])

def tiny(path):
    out = {'pp512': [], 'tg128': []}
    # "tg128 | 36 threads | 93.14 +/- 0.00 tok/s | ... || pp512 | 2777.3 +/- 0.0 tok/s | ..."
    for line in open(path, errors='ignore'):
        # "tg128 | 36 threads | 13.74 +/- 0.00 tok/s | ...": the MEAN is the
        # number before "+/-"; the bare "([\d.]+) tok/s" form matched nothing
        # (or the +/- stdev) and silently produced 0 runs.
        for key, val in re.findall(r'\b(pp512|tg128) \| (?:[^|]*\| )?([\d.]+) \+/-', line):
            out[key].append(float(val))
    return out

def llama(path):
    out = {'pp512': [], 'tg128': []}
    # "| model | size | params | backend | ngl | pp512 | 2673.7 ± 0.00 |"
    for line in open(path, errors='ignore'):
        cells = [c.strip() for c in line.split('|')]
        for key in out:
            if key in cells:
                i = cells.index(key)
                m = re.search(r'([\d.]+)', cells[i + 1])
                if m:
                    out[key].append(float(m.group(1)))
    return out

tv, lv = tiny(tlog), llama(llog)
def stat(v):
    if not v:
        return 'n/a', float('nan')
    m = statistics.mean(v)
    return f'{m:.1f}', ((max(v) - min(v)) / m * 100.0 if m else 0.0)

for key in ('pp512', 'tg128'):
    tm, tsp = stat(tv[key])
    lm, lsp = stat(lv[key])
    delta = ''
    if tv[key] and lv[key]:
        t, l = statistics.mean(tv[key]), statistics.mean(lv[key])
        delta = f'  tiny/llama {(t / l - 1.0) * 100.0:+.1f}%'
    print(f'  {key:5s} tinycoder {tm:>9s} tok/s ({len(tv[key])} runs, spread {tsp:.2f}%)'
          f'   llama {lm:>9s} tok/s ({len(lv[key])} runs, spread {lsp:.2f}%){delta}')

if tv['tg128']:
    tg = statistics.mean(tv['tg128'])
    print(f'  implied decode bandwidth (all weights every token): '
          f'{size * tg / 1e9:.1f} GB/s  (peak ~616 GB/s)')
print(f'  raw logs: {tlog} , {llog}')
PY
