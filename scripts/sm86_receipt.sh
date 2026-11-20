#!/usr/bin/env bash
# sm86_receipt.sh — the whole verification of an Ampere (sm_86) build, on one
# machine, in one run, with a receipt for every claim.
#
# Why this script exists: none of the sm_86 work can be measured on the sm_75
# reference box (RTX 2080 Ti).  Everything up to the point of a rented 3090 is
# compile-time verification and a cost model, and both are only as good as the
# checks that the card then runs.  This is that run: it refuses to proceed on the
# wrong architecture, it refuses to publish a number whose text gate failed, and
# it writes an artefact per claim instead of a summary someone has to trust.
#
#   scripts/sm86_receipt.sh [outdir]
#   SM86_STATIC_ONLY=1 scripts/sm86_receipt.sh [outdir]   # device + cubin +
#       SASS + arch_check + gates, no benchmarks.  Usable on the sm_75 box with
#       SM86_ALLOW_NON_AMPERE=1 to check the script and the SASS receipts; the
#       sm_86 guard correctly refuses the numbers without the card.
#
# What it produces in <outdir> (default /tmp/sm86_receipt):
#   00_device.txt            the device as reported by the driver, and the SM count
#   01_elf.txt               which cubins the binary actually carries (sm_86 must be there)
#   02_res_usage.txt         registers/spills per kernel on sm_86, from the real cubin
#   03_sass_imma.txt         IMMA shape counts on sm_86 (m16n8k32 expected, not m8n8k16)
#   04_sass_ldgsts.txt       LDGSTS count: 0 before the staging rewrite, >0 after
#   05_arch_check.txt        scripts/arch_check.py against the built tree, both archs
#   06_gates.txt             exl3 parity count, logits id/order, both parity probes, suite
#   07_text_gate.txt         scripts/text_gate.sh, every checkpoint
#   08_bench_<model>.log     pp512/tg128 per checkpoint, default environment
#   09_ab_llama_<model>.log  interleaved TinyCoder vs llama-bench, if llama-bench is on PATH
#   10_ab_exl3.log           interleaved TinyCoder vs exllamav3 (EXL3 row only)
#   RECEIPT.md               the summary, with the physics check applied to every row
#
# Rules this script enforces rather than documents:
#   * It refuses to run if the visible device is not sm_86 (override with
#     SM86_ALLOW_NON_AMPERE=1 only to test the script itself).
#   * A row is only reported if the text gate for that checkpoint passed in this
#     same run.  A fast number from a path that cannot answer is not a result.
#   * Every throughput is checked against the card's published peaks
#     (2*params*tok_s for prefill, weight_bytes*tok_s for decode) and a value
#     above the roof is reported as IMPOSSIBLE rather than published.
#   * Benchmarks run serially, GPU idle before each, three runs, spread reported.
set -uo pipefail

OUT="${1:-/tmp/sm86_receipt}"
mkdir -p "$OUT"
TC="./build-1.0/benchmarks/tinycoder_bench"
MODEL_EXL3="/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw"
EXL3_PYTHON="${EXL3_PYTHON:-/tmp/exl3venv/bin/python}"
THREADS="${SM86_THREADS:-36}"

say() { printf '\n=== %s\n' "$*"; }
die() { printf 'sm86_receipt: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight: the three ways a rented hour dies before any number is taken.
say "preflight"
PREFLIGHT_FAIL=0
# Always returns 0: as the last command of a `pf ... || pf ...` chain, a
# non-zero return here fired the FAIL branch on a passing line.
pf() { printf '  %-46s %s\n' "$1" "$2"; [ "$2" = FAIL ] && PREFLIGHT_FAIL=1; return 0; }

# 1. The build needs three git repos (mgorshkov/cmake, np, googletest) via
#    FetchContent. On a box with no network that clone fails and nothing builds;
#    FETCHCONTENT_FULLY_DISCONNECTED plus the three *_SOURCE_DIR vars make the
#    build offline. Tested on the reference box: configures and builds clean.
CACHED_DEPS=0
for d in ${FETCHCONTENT_SOURCE_DIR_CMAKE:+/}${FETCHCONTENT_SOURCE_DIR_CMAKE:-x} \
         ${FETCHCONTENT_SOURCE_DIR_NP:+x}${FETCHCONTENT_SOURCE_DIR_NP:-x} \
         ${FETCHCONTENT_SOURCE_DIR_GOOGLETEST:+x}${FETCHCONTENT_SOURCE_DIR_GOOGLETEST:-x}; do
  [ -d "$d" ] && CACHED_DEPS=$((CACHED_DEPS + 1))
done
if [ "${FETCHCONTENT_FULLY_DISCONNECTED:-OFF}" = "ON" ]; then
  [ "$CACHED_DEPS" -eq 3 ] && pf "offline deps (3 source dirs)" ok \
                           || pf "offline deps (need all 3 source dirs)" FAIL
elif [ -d "build-1.0/_deps/cmake-src" ]; then
  pf "dependency sources present" ok
else
  pf "dependency sources present (needs network)" "?"
fi

# 2. The toolchain that has to exist for the gates, not just the benchmarks.
command -v nvcc >/dev/null 2>&1 && pf "nvcc" ok || pf "nvcc" FAIL
command -v cuobjdump >/dev/null 2>&1 && pf "cuobjdump (SASS receipts)" ok || pf "cuobjdump" FAIL
[ -x "$TC" ] && pf "tinycoder_bench" ok || pf "tinycoder_bench (build first)" FAIL
for t in tiny_logits_probe int8_gemm_parity gemv_decode_parity exl3_gpu_parity tinycoder_test; do
  [ -x "build-1.0/unit_tests/$t" ] && pf "unit_tests/$t" ok || pf "unit_tests/$t" FAIL
done

# 3. Every checkpoint the script is about to name. A missing model used to show
#    up as a text-gate failure 40 minutes in.
missing=0
while IFS='|' read -r path _a _l; do
  # -e, not -f: an EXL3 checkpoint is a directory.
  [ -e "$path" ] || { printf '  %-46s MISSING\n' "$(basename "$path")"; missing=1; }
done < <(grep -E '^  "' scripts/text_gate.sh | sed -E 's/^  "//; s/"$//')
[ "$missing" -eq 0 ] && pf "all checkpoints present" ok || pf "checkpoint set" FAIL

[ "$PREFLIGHT_FAIL" -eq 1 ] && die "preflight failed: fix the FAIL lines above before measuring"

# ---------------------------------------------------------------------------
say "device"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem \
           --format=csv > "$OUT/00_device.txt" 2>&1
nvidia-smi -q -d CLOCK >> "$OUT/00_device.txt" 2>&1
cat "$OUT/00_device.txt" | head -4

# The one guard that matters: every number below is a claim about sm_86.
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo "")
SM=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "compute_cap=${CC:-unknown} name=${SM}" | tee -a "$OUT/00_device.txt"
case "${CC}" in
  8.6) : ;;
  "")
     echo "sm86_receipt: this driver does not report compute_cap; check name=${SM}" >&2
     [ "${SM86_ALLOW_NON_AMPERE:-0}" = "1" ] || die "cannot confirm sm_86" ;;
  *)
     [ "${SM86_ALLOW_NON_AMPERE:-0}" = "1" ] || \
       die "wrong device: compute_cap=${CC} (${SM}); this receipt is only valid on sm_86. \
Set SM86_ALLOW_NON_AMPERE=1 only to exercise the script itself." ;;
esac

[ -x "$TC" ] || die "$TC missing; build first: cmake --build build-1.0 --target tinycoder_core tinycoder_bench -j 20"

# ---------------------------------------------------------------------------
# Instrument availability, recorded before anything is measured: a receipt that
# silently omits the ncu pass or the measured roof reads like a complete one.
say "instruments"
{
  for t in nvcc nvbandwidth ncu llama-bench nvidia-smi; do
    printf '%-14s %s\n' "$t" "$(command -v "$t" || echo 'not installed')"
  done
  printf '%-14s %s\n' "exllamav3" \
    "$([ -x "$EXL3_PYTHON" ] && echo "$EXL3_PYTHON" || echo 'venv missing')"
} | tee "$OUT/00b_instruments.txt"

# ---------------------------------------------------------------------------
say "measured roofs (design item 2: ratios must be against the card's real peak)"
if command -v nvcc >/dev/null 2>&1; then
  # int8_mma_peak measures the fp32-accumulate IMMA roof with register-only
  # operands, so it is an upper bound no kernel here can exceed.
  if nvcc -arch="sm_${CC/./}" -O3 -o "$OUT/int8_mma_peak" tools/int8_mma_peak.cu \
        > "$OUT/02b_mma_peak_build.log" 2>&1; then
    "$OUT/int8_mma_peak" 2>&1 | tee "$OUT/02b_mma_peak.txt" | tail -4
  else
    echo "int8_mma_peak build failed (see 02b_mma_peak_build.log): roof unchecked" | \
        tee "$OUT/02b_mma_peak.txt"
  fi
else
  echo "nvcc not on PATH: the IMMA roof was not measured, ratios below use the nominal 284 TOPS" | \
      tee "$OUT/02b_mma_peak.txt"
fi
if command -v nvbandwidth >/dev/null 2>&1; then
  nvbandwidth -t device_to_device_memcpy_read_ce 2>&1 | tail -8 | tee "$OUT/02c_memory_roof.txt"
else
  echo "nvbandwidth not installed: memory roof is the nominal 936 GB/s, not measured" | \
      tee "$OUT/02c_memory_roof.txt"
fi

# ---------------------------------------------------------------------------
say "what the binary carries (fatbin must contain sm_86 SASS, not just PTX)"
find build-1.0 -name "*.o" -newer CMakeLists.txt 2>/dev/null | head -1 >/dev/null
ELF_SRC="build-1.0/CMakeFiles/tinycoder_core.dir/src/cpp/core/cuda/exl3/EXL3Kernels.cu.o"
# The int8 tensor-core kernels are DEFINED in GemmEntry.cu (GemmDispatch.cu only
# declares them and picks tiles), so the SASS receipts have to come from there:
# GemmDispatch.o contains no IMMA at all and a receipt taken from it would read
# "0 IMMA" on a perfectly good build.  Found by running this script.
ELF_GEMM="build-1.0/CMakeFiles/tinycoder_core.dir/src/cpp/core/cuda/gemm/GemmEntry.cu.o"
{
  for f in "$ELF_SRC" "$ELF_GEMM"; do
    echo "--- $f"
    cuobjdump --list-elf "$f" 2>&1 | sed -n '1,40p'
  done
} | tee "$OUT/01_elf.txt" | grep -c "sm_86" | xargs -I{} echo "sm_86 ELF entries: {}"
grep -q "sm_86" "$OUT/01_elf.txt" || die "no sm_86 cubin in the built objects: the numbers would be sm_75 code"

# ---------------------------------------------------------------------------
# The static receipts.  These are the checks that stand in for hardware when
# there is none; they are cheap enough to run first, and a failure here means
# the benchmarks below would measure the wrong kernel.
say "static receipts (sm_86)"
cuobjdump -res-usage -arch sm_86 "$ELF_SRC" > "$OUT/02_res_usage.txt" 2>&1
cuobjdump -res-usage -arch sm_86 "$ELF_GEMM" >> "$OUT/02_res_usage.txt" 2>&1
# IMMA shape: m16n8k32 has k=32, so the SASS mnemonic carries 16832; the Turing
# form is 1688/16816.  Counting both is how a silent downgrade shows up.
# The split matters, not the total: 16832 is m16n8k32 (Ampere's int8 shape) and
# 16816 is the Turing form.  An sm_86 cubin that carries only 16816 has silently
# kept the old shape.
cuobjdump -sass -arch sm_86 "$ELF_GEMM" 2>/dev/null | grep -oE "IMMA\.[A-Z0-9.]+" | sort | uniq -c \
    > "$OUT/03_sass_imma.txt"
cat "$OUT/03_sass_imma.txt"
grep -q "16832" "$OUT/03_sass_imma.txt" || \
    echo "WARNING: no IMMA.16832 in the sm_86 cubin: the int8 MMA is still on the Turing shape" | \
    tee -a "$OUT/03_sass_imma.txt"
# cp.async: absent before the staging rewrite, present after.  Recorded either
# way so a receipt from before and after can be diffed.
cuobjdump -sass -arch sm_86 "$ELF_GEMM" 2>/dev/null | grep -c "LDGSTS" \
    > "$OUT/04_sass_ldgsts.txt"
echo "LDGSTS on sm_86: $(cat "$OUT/04_sass_ldgsts.txt")"

python3 scripts/arch_check.py --arch 75 86 > "$OUT/05_arch_check.txt" 2>&1
echo "arch_check exit: $? (see $OUT/05_arch_check.txt)"
tail -2 "$OUT/05_arch_check.txt"

# ---------------------------------------------------------------------------
say "correctness gates"
{
  echo "--- exl3_gpu_parity (baseline count is 29 FAIL)"
  ./build-1.0/unit_tests/exl3_gpu_parity 2>&1 | grep -c "\[FAIL\]"
  echo "--- logits (gate is the id and the top-5 order, not the value)"
  ./build-1.0/unit_tests/tiny_logits_probe "$MODEL_EXL3" "def hello():" 2>&1 |
      grep -E "top[1-5]:" | sed -n '1,5p'
  echo "--- int8 GEMM parity"
  ./build-1.0/unit_tests/int8_gemm_parity /data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf \
      --seqlen 8 --rows 32 --max-shapes 8 2>&1 | grep -E "^\[parity\] (PASS|FAIL)|verified"
  echo "--- decode GEMV parity"
  ./build-1.0/unit_tests/gemv_decode_parity /data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf \
      --seqlen 1 --rows 32 --max-shapes 8 2>&1 | grep -E "^\[parity\] (PASS|FAIL)|verified"
  echo "--- suite"
  ./build-1.0/unit_tests/tinycoder_test 2>&1 | grep -E "^\[==========\]|^\[  PASSED  \]|^\[  FAILED  \]"
} > "$OUT/06_gates.txt" 2>&1
cat "$OUT/06_gates.txt"

# ---------------------------------------------------------------------------
say "text gates (a throughput without one is not a result)"
./scripts/text_gate.sh > "$OUT/07_text_gate.txt" 2>&1
TEXT_GATE_EXIT=$?
grep -E "PASS|FAIL" "$OUT/07_text_gate.txt"
# Every checkpoint with a row must answer; a throughput without a passing gate is
# not a result, so record the failing set explicitly instead of trusting the exit.
GATE_FAILS=$(grep -c "FAIL" "$OUT/07_text_gate.txt")
echo "text_gate exit=${TEXT_GATE_EXIT} failing rows=${GATE_FAILS} (expected 0)"

if [ "${SM86_STATIC_ONLY:-0}" = "1" ]; then
  say "static-only run: benchmarks skipped (SM86_STATIC_ONLY=1)"
  exit 0
fi

# ---------------------------------------------------------------------------
check_gpu() {
  for _ in $(seq 1 30); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0
    sleep 2
  done
  die "GPU stayed busy for 60s; refusing to measure"
}

# A row is published only if its text gate PASSED in this run.  Requiring the
# PASS line rather than the absence of a FAIL one matters: if the gate script
# dies before printing anything, "no FAIL" would be true for every row and the
# whole receipt would look gated.
gate_ok() {
  local label="$1"
  [ -f "$OUT/07_text_gate.txt" ] && grep -qE "^${label} +PASS" "$OUT/07_text_gate.txt"
}

bench_row() {   # bench_row <label> <model>
  local label="$1" model="$2"
  if ! gate_ok "$label"; then
    echo "SKIP $label: text gate failed in this run ($OUT/07_text_gate.txt)" |
        tee -a "$OUT/08_bench_summary.txt"
    return
  fi
  local log="$OUT/08_bench_$(basename "$model").log"
  : > "$log"
  for r in 1 2 3; do
    check_gpu
    "$TC" --model "$model" --n-prompts 512 --n-gen 128 --reps 1 \
        --threads "$THREADS" >> "$log" 2>&1
  done
  grep -E "^tg128" "$log" | tee -a "$OUT/08_bench_summary.txt"
}

say "benchmarks (serial, GPU idle before each, three runs)"
: > "$OUT/08_bench_summary.txt"
# The checkpoint list is READ FROM text_gate.sh rather than duplicated here: a
# row and its gate must live in the same place, or a new row gets published with
# no gate (which is exactly how the 9B row spent weeks measuring noise).
while IFS='|' read -r path _answer label; do
  case "$label" in
    *35B-A3B*)
      # Documented pre-existing failure: it emits endahendah from the first token
      # at BASE d79d73f too, so the row has no working gate. Benchmarked anyway,
      # because its published number needs a 3090 counterpart, but flagged.
      echo "NOTE 35B-A3B: text gate is a known failure; its numbers are recorded, not published" |
          tee -a "$OUT/08_bench_summary.txt" ;;
  esac
  bench_row "$label" "$path"
done < <(grep -E '^  "' scripts/text_gate.sh | sed -E 's/^  "//; s/"$//')

# ---------------------------------------------------------------------------
# Design item 7, and deliberately before any tuning conclusion: if a whole-stage
# gap exists against the reference engine, this names the stage it lives in.  It
# is the cheapest measurement on the card.
say "stage trace (per-stage device events; TINYCODER_STAGE_TRACE=1)"
for spec in "9B q4_k_m:/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf" \
            "qwen3.8-27b exl3:$MODEL_EXL3"; do
  label="${spec%%:*}"; model="${spec#*:}"
  gate_ok "$label" || { echo "SKIP $label stage trace: text gate failed"; continue; }
  check_gpu
  TINYCODER_STAGE_TRACE=1 "$TC" --model "$model" --n-prompts 512 --n-gen 16 \
      --reps 1 --threads "$THREADS" > "$OUT/07b_stage_trace_$(basename "$model").log" 2>&1
  grep -E "^(head|proj|state|ffn|tail|attn|gpu|host|budget|\[stage)" \
      "$OUT/07b_stage_trace_$(basename "$model").log" | tail -12
done

# ---------------------------------------------------------------------------
say "baseline columns (interleaved, so clock drift hits both sides equally)"
if command -v llama-bench >/dev/null 2>&1; then
  for spec in "7B iq2_s:/data/models/qwen/qwen2.5/Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf" \
              "9B q4_k_m:/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf"; do
    label="${spec%%:*}"; model="${spec#*:}"
    gate_ok "$label" || { echo "SKIP $label baseline: text gate failed"; continue; }
    for pair in 1 2 3; do
      check_gpu
      "$TC" --model "$model" --n-prompts 512 --n-gen 128 --reps 1 --threads "$THREADS" \
           > "$OUT/09_ab_llama_$(basename "$model")_tc${pair}.log" 2>&1
      check_gpu
      llama-bench -m "$model" -p 512 -n 128 -t "$THREADS" -r 2 \
           > "$OUT/09_ab_llama_$(basename "$model")_lb${pair}.log" 2>&1
    done
    echo "interleaved pair set done: $label"
  done
else
  echo "llama-bench not on PATH: GGUF baseline columns not measured in this receipt" |
      tee -a "$OUT/09_ab_llama_missing.txt"
fi

if [ -x "$EXL3_PYTHON" ] && gate_ok "qwen3.8-27b exl3"; then
  OUT_AB="$OUT/10_ab_exl3"
  scripts/exl3_ab_bench.sh "$OUT_AB" > "$OUT/10_ab_exl3.log" 2>&1
  tail -6 "$OUT/10_ab_exl3.log"
else
  echo "exllamav3 venv missing ($EXL3_PYTHON) or its text gate failed: EXL3 baseline not measured" \
      | tee "$OUT/10_ab_exl3.log"
fi

# ---------------------------------------------------------------------------
say "receipt"
python3 - "$OUT" <<'RECEIPT_PY'
import glob, os, re, statistics as st, sys

out = sys.argv[1]
# Published peaks for the sm_86 reference card (RTX 3090, GA102):
#   int8 tensor 284 TOPS (fp32-accumulate path), 936 GB/s.
PEAK_TOPS_INT8 = 284.0
PEAK_GBPS = 936.0
# params (B) and resident weight bytes (GB) per checkpoint: the physics check
# needs both, and a row whose size is unknown is reported as unchecked rather
# than assumed fine.
PARAMS_B = {"1.5b": 1.5, "7b": 6.7, "9b": 8.8, "27b": 27.0}
WEIGHT_GB = {"1.5b": 0.7, "7b": 2.4, "9b": 5.9, "27b": 8.6}

def size_of(name):
    low = name.lower()
    for k in ("1.5b", "7b", "9b", "27b"):
        if k in low:
            return k
    return None

# One row per checkpoint, not per run: three interleaved runs of the same model
# are three measurements of one row, and the spread between them is part of the
# number (AGENTS.md: report both values and the spread).
runs = {}
for path in sorted(glob.glob(os.path.join(out, "08_bench_*.log"))):
    model = os.path.basename(path)[len("08_bench_"):-len(".log")]
    txt = open(path, "rb").read().decode("utf-8", "replace")
    for m in re.finditer(r"^tg128 \| \d+ threads \| ([0-9.]+) \+/- [0-9.]+ tok/s \| "
                         r"[0-9.]+ ms/decode \(mean\) \|\| pp512 \| ([0-9.]+)", txt, re.M):
        runs.setdefault(model, []).append((float(m.group(2)), float(m.group(1))))
if not runs:
    print("no benchmark rows parsed: nothing was measured")
    sys.exit(0)

lines = ["| model | runs | pp512 tok/s | pp spread | tg128 tok/s | tg spread | "
         "implied prefill TOPS | implied decode GB/s | verdict |",
         "|---|---|---|---|---|---|---|---|---|"]
for model, v in runs.items():
    pp = [x[0] for x in v]
    tg = [x[1] for x in v]
    pp_m, tg_m = st.mean(pp), st.mean(tg)
    pp_s = (max(pp) - min(pp)) / 2 if len(pp) > 1 else 0.0
    tg_s = (max(tg) - min(tg)) / 2 if len(tg) > 1 else 0.0
    k = size_of(model)
    if k is None:
        lines.append("| %s | %d | %.1f | %.1f | %.2f | %.2f | - | - | unchecked |"
                     % (model, len(v), pp_m, pp_s, tg_m, tg_s))
        continue
    tops = 2.0 * PARAMS_B[k] * 1e9 * pp_m / 1e12
    gbps = WEIGHT_GB[k] * tg_m
    bad = []
    if tops > PEAK_TOPS_INT8:
        bad.append("prefill above tensor roof")
    if gbps > PEAK_GBPS:
        bad.append("decode above memory roof")
    verdict = "ok" if not bad else "IMPOSSIBLE: " + ", ".join(bad)
    lines.append("| %s | %d | %.1f | %.1f | %.2f | %.2f | %.1f | %.0f | %s |"
                 % (model, len(v), pp_m, pp_s, tg_m, tg_s, tops, gbps, verdict))

summary = os.path.join(out, "RECEIPT.md")
with open(summary, "w") as f:
    f.write("# sm_86 receipt\n\n")
    dev = open(os.path.join(out, "00_device.txt")).read().split("\n")[0]
    f.write(dev + "\n\n")
    f.write("```\n")
    for f2 in ("01_elf.txt", "04_sass_ldgsts.txt", "02b_mma_peak.txt"):
        p = os.path.join(out, f2)
        if os.path.exists(p):
            f.write("--- %s\n" % f2 + open(p).read()[-800:] + "\n")
    f.write("```\n\n")
    f.write("## Rows\n\n" + "\n".join(lines) + "\n\n")
    f.write("Spread is (max - min) / 2 over the runs of that row. A row whose `verdict` is "
            "IMPOSSIBLE is a bug in the measurement, not a win: the columns are "
            "2*params*tok_s and weight_bytes*tok_s against this card's published peaks.\n\n")
    f.write("Rows exist only if their text gate passed in this run (`07_text_gate.txt`); "
            "a throughput without one is not a result.\n")
print("\n".join(lines))
print("\nreceipt written to", summary)
RECEIPT_PY

say "done: $OUT"
