#!/usr/bin/env bash
# exl3_ab_bench.sh — Interleaved A/B of TinyCoder vs exllamav3 1.6.6 on the
# Qwen3.8-27B EXL3 checkpoint.  Three alternating pairs, serial, GPU verified
# empty before each run, means and spreads reported.
#
# The Bonsai family is measured interleaved (scripts/bonsai_baseline_bench.sh)
# because separate runs let clock drift bias the comparison.  The EXL3 row was
# previously measured engine-by-engine, so this makes it the same protocol.
#
#   scripts/exl3_ab_bench.sh [outdir]
#
# Raw stdout of every run is kept in <outdir>/ so the parse can be re-derived
# without repeating the measurements.
set -uo pipefail
OUT="${1:-/tmp/exl3_ab}"
mkdir -p "$OUT"
TC=./build-1.0/benchmarks/tinycoder_bench
MODEL=/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw
PYTHON="${EXL3_PYTHON:-/tmp/exl3venv/bin/python}"

check_gpu() {
  for _ in $(seq 1 15); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
    sleep 2
  done
  local apps; apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [ -n "$apps" ]; then echo "GPU BUSY ($apps) - refusing to measure" >&2; exit 1; fi
}

[ -x "$PYTHON" ] || { echo "exllamav3 venv missing: $PYTHON" >&2; exit 1; }

for pair in 1 2 3; do
  check_gpu
  timeout 1800 "$TC" --model "$MODEL" --n-prompts 512 --n-gen 128 --reps 1 \
      --threads 36 > "$OUT/tc_exl3_${pair}.log" 2>&1
  check_gpu
  PATH="$(dirname "$PYTHON"):$PATH" timeout 1800 "$PYTHON" \
      scripts/exl3_baseline_bench.py "$MODEL" > "$OUT/ev_exl3_${pair}.log" 2>&1
  echo "pair $pair done"
done

python3 - "$OUT" <<'PY'
import glob, os, re, statistics as st, sys
out = sys.argv[1]
rows = {}
for path in sorted(glob.glob(os.path.join(out, "*.log"))):
    engine = os.path.basename(path)[:2]
    txt = open(path, "rb").read().decode("utf-8", "replace")
    if engine == "tc":
        m = re.search(r"pp \d+ tok @ [^|]*\|\| ([0-9.]+) \+/- [0-9.]+ tok/s \(warm reps\)"
                      r" \|\| tg \d+ tok @ [^|]*\|\| ([0-9.]+) \+/-", txt)
        v = (float(m.group(1)), float(m.group(2))) if m else None
    else:
        pp = re.search(r"RESULT pp512 \|\s*([0-9.]+) tok/s", txt)
        tg = re.search(r"RESULT tg128 \|\s*([0-9.]+) tok/s", txt)
        v = (float(pp.group(1)), float(tg.group(1))) if (pp and tg) else None
    if v is None:
        print("PARSE FAIL:", path)
        continue
    rows.setdefault(engine, []).append(v)
names = {"tc": "TinyCoder", "ev": "exllamav3"}
print("\n%-12s %9s %9s %10s %10s %3s" % ("engine", "pp512", "tg128", "pp spread", "tg spread", "n"))
for eng in ("tc", "ev"):
    v = rows.get(eng, [])
    if not v: continue
    pp = [x[0] for x in v]; tg = [x[1] for x in v]
    print("%-12s %9.1f %9.2f %10.1f %10.2f %3d" %
          (names[eng], st.mean(pp), st.mean(tg),
           (max(pp) - min(pp)) / 2, (max(tg) - min(tg)) / 2, len(v)))
PY
