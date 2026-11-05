#!/usr/bin/env bash
# Text gate for every checkpoint that has a row in BENCHMARKS.md.
#
# A throughput number only exists if the engine can answer a question on that
# checkpoint. This runs the gated factual question through the real forward and
# requires the answer to contain the expected keyword.
#
# Exit status: 0 only if every checkpoint answers correctly. Any skip counts as
# a FAILURE — `FullQuestions` skips shapes that are missing from its allow-list
# in unit_tests/ModelTest.cpp, and a skip reads like a pass in a summary. That
# is exactly how three published rows once measured an incoherent path.
#
#   ./scripts/text_gate.sh              # every checkpoint
#   ./scripts/text_gate.sh qwen2.5      # only models whose path matches
#
set -uo pipefail

TEST_BIN="${TEST_BIN:-./build-1.0/unit_tests/tinycoder_test}"
FILTER="${1:-}"

# model path | expected substring | label
MODELS=(
  "/data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf|Paris|1.5B q2_k"
  "/data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-iq3_xxs-imat.gguf|Paris|1.5B iq3_xxs"
  "/data/models/qwen/qwen2.5/qwen2.5-coder-7b-instruct-iq3_xxs-imat.gguf|Paris|7B iq3_xxs"
  "/data/models/qwen/qwen2.5/Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf|Paris|7B iq2_s"
  "/data/models/ornith/Ornith-1.5-9B-Q4_K_M.gguf|Paris|9B q4_k_m"
  "/data/models/ornith/Ornith-1.5-35B-A3B-Q8_0.gguf|Paris|35B-A3B"
)

fail=0
ran=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r model want label <<<"$entry"
  if [ -n "$FILTER" ] && [[ "$model" != *"$FILTER"* ]]; then
    continue
  fi
  if [ ! -f "$model" ]; then
    printf '%-14s SKIP (missing file: %s)\n' "$label" "$model"
    fail=1
    continue
  fi
  ran=$((ran + 1))
  out="$(TINYCODER_MODEL_PATH="$model" timeout 2400 "$TEST_BIN" \
      --gtest_filter='FullQuestions/SampleQuestionTest.AnswersQuestion/1' 2>&1 | tr -d '\000')"
  if grep -qa "Skipped" <<<"$out"; then
    printf '%-14s FAIL (no text gate: shape missing from the FullQuestions allow-list)\n' "$label"
    fail=1
  elif grep -qa "\[  FAILED  \]" <<<"$out"; then
    text="$(grep -a -m1 'Output:' <<<"$out" | cut -c1-96)"
    printf '%-14s FAIL %s\n' "$label" "$text"
    fail=1
  elif grep -qa "$want" <<<"$out"; then
    printf '%-14s PASS (answers "%s")\n' "$label" "$want"
  else
    text="$(grep -a -m1 'Output:' <<<"$out" | cut -c1-96)"
    printf '%-14s FAIL (answer lacks "%s") %s\n' "$label" "$want" "$text"
    fail=1
  fi
done

if [ "$ran" -eq 0 ]; then
  echo "no checkpoint matched filter '$FILTER'" >&2
  exit 2
fi
exit "$fail"
