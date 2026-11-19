#!/usr/bin/env bash
# context_gate.sh — does a long window actually work?
#
# For each row: request a window, put an access code at the start of a prompt
# that fills it, ask for the code at the far end, and require the code back.
# A window that is merely *reported* granted but is not really cached fails
# here, which is the whole point: the engine used to clamp every model to 2048
# and nobody noticed, because nothing ever asked it to attend across more.
#
# A row FAILS when:
#   * the request was not granted (the probe prints granted < requested and the
#     limiting resource), or
#   * the prompt did not fit (exit 2), or
#   * the answer does not contain the code (exit 1), or
#   * the checkpoint is missing (a SKIP is a FAILURE, same rule as the text gate).
#
#   ./scripts/context_gate.sh                # every row
#   ./scripts/context_gate.sh bonsai         # only rows whose path matches
#
# Exit status: 0 only if every row answers at its requested window.
set -uo pipefail

PROBE_BIN="${PROBE_BIN:-./build-1.0/unit_tests/ctx_probe}"
FILTER="${1:-}"

# model path | requested ctx | label | needle position (0 = grant only)
#
# Ternary-Bonsai-2-27B is grant-only: it answers the needle at 600 tokens and
# emits EOS at 3600 in BOTH the f32 and the int8 store, so a failing needle row
# there would report a model limitation as a window result.  Its rows assert the
# grant (f32 and int8 ceilings) and nothing more.
#
# The requested windows are the ceilings the planner actually grants on the
# reference card (11 GB), measured: 9B f32 52480 / int8 196352, Bonsai f32 16640
# / int8 64256, 27B f32+mirror 12288.  Asking for more than the ceiling is a
# legitimate failure mode ("limited by VRAM") and is checked by the reduction
# tests, not by this gate.
MODELS=(
  "/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf|8192|9B 8k|7680"
  "/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf|32768|9B 32k|32256"
  "/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf|65536|9B 64k int8|65024"
  "/data/models/bonsai/Ternary-Bonsai-2-27B-PQ2_0.gguf|32768|bonsai 32k|0"
  "/data/models/bonsai/Ternary-Bonsai-2-27B-PQ2_0.gguf|64000|bonsai 64k|0"
  "/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw|8192|27b exl3 8k|7680"
)

fail=0
ran=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r model ctx label needle <<<"$entry"
  if [ -n "$FILTER" ] && [[ "$model" != *"$FILTER"* ]]; then
    continue
  fi
  if [ ! -e "$model" ]; then
    printf '%-14s SKIP (missing file: %s)\n' "$label" "$model"
    fail=1
    continue
  fi
  ran=$((ran + 1))
  if [ "$needle" = "0" ]; then
    gen=0
  else
    gen=256
  fi
  out="$("$PROBE_BIN" "$model" --ctx "$ctx" --needle-tokens "$needle" \
      --gen "$gen" 2>&1 | tr -d '\000')"
  rc=$?
  granted="$(sed -n 's/^granted=\([0-9]*\) maxAllowed=\([0-9]*\).*/\1/p' <<<"$out" | head -1)"
  allowed="$(sed -n 's/^granted=\([0-9]*\) maxAllowed=\([0-9]*\).*/\2/p' <<<"$out" | head -1)"
  kv="$(sed -n 's/^granted=[0-9]* maxAllowed=[0-9]* kv=\([a-z0-9]*\).*/\1/p' <<<"$out" | head -1)"
  if [ -z "$granted" ]; then
    printf '%-14s FAIL (no plan: model did not load)\n' "$label"
    grep -a -m2 -E "FATAL|failed" <<<"$out" | sed 's/^/               /'
    fail=1
    continue
  fi
  # A grant-only row demands the full requested window; a needle row also has to
  # answer it, so a reduced grant is a failure there for the same reason.
  if [ "$rc" -eq 4 ] || [ "$granted" -lt "$ctx" ]; then
    reason="$(sed -n 's/^reason=//p' <<<"$out" | head -1)"
    printf '%-14s FAIL (requested %s, granted %s, maxAllowed %s, kv=%s)\n' \
      "$label" "$ctx" "$granted" "$allowed" "$kv"
    printf '               %s\n' "$reason"
    fail=1
    continue
  fi
  if [ "$rc" -ne 0 ]; then
    ans="$(sed -n 's/^answer=<<<\(.*\)>>>$/\1/p' <<<"$out" | head -1 | cut -c1-96)"
    printf '%-14s FAIL (granted %s, maxAllowed %s, kv=%s, exit %s) %s\n' \
      "$label" "$granted" "$allowed" "$kv" "$rc" "$ans"
    fail=1
    continue
  fi
  printf '%-14s PASS (granted %s, maxAllowed %s, kv=%s)\n' \
    "$label" "$granted" "$allowed" "$kv"
done

if [ "$ran" -eq 0 ]; then
  echo "no checkpoint matched filter '$FILTER'" >&2
  exit 2
fi
exit "$fail"
