#!/usr/bin/env python3
"""Aggregate an `nvprof --print-gpu-trace --csv` dump into kernel classes.

Usage:
    python3 scripts/prefill_attribution.py /tmp/trace.csv [--top 15] [--scale 0.5]

Input is the CSV that nvprof writes to stdout (NOT `-o`, which is the binary
.nvvp profile).  Each row is
    start_ms, duration_ms, gx, gy, gz, bx, by, bz, smem, dsmem, regs, ...,
    "<kernel name>", <pid>
and the kernel name is the second-to-last field.

`--scale` divides every duration by a constant: a bench run does a COLD warmup
plus N timed reps, so a trace of a `--reps 2` run contains 3 identical prefills
and `--scale 3` yields the per-prefill cost.

The point of this script is attribution, not benchmarking: it answers "which
kernel class owns the prefill" for BOTH engines with the same classifier, so a
gap can be assigned to a class instead of guessed at.  It is deliberately
engine-neutral: llama.cpp's `mul_mat_q`/`mul_mat_vec_q` and TinyCoder's
`int8Gemm*`/`gemv*` land in the same bucket.
"""
import csv
import re
import sys
from collections import defaultdict

# (class, regex) -- first match wins, so put the specific ones first.
CLASSES = [
    ("attn", r"flash_attn|attn|\bkv_?store|softmax"),
    ("gemm", r"mul_mat|gemm|Gemm|wmma|hmma|imma"),
    ("gemv", r"gemv|Gemv|GEMV|matvec|dot"),
    ("dequant", r"dequant|Dequant|to_f16|f32_to_f16|convert"),
    ("norm", r"norm|Norm"),
    ("rope", r"rope|Rope"),
    ("quantize", r"quantize"),
    ("copy", r"memcpy|Memcpy|copy|Copy|fill|Fill|set_rows|get_rows"),
    ("misc", r".*"),
]


def classify(name):
    for cls, pat in CLASSES:
        if re.search(pat, name):
            return cls
    return "misc"


def main():
    path = sys.argv[1]
    top = 15
    scale = 1.0
    for i, a in enumerate(sys.argv):
        if a == "--top":
            top = int(sys.argv[i + 1])
        if a == "--scale":
            scale = float(sys.argv[i + 1])
    per_class = defaultdict(float)
    per_kernel = defaultdict(float)
    rows = 0
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("==") or not line.strip():
                continue
            try:
                fields = next(csv.reader([line]))
            except Exception:
                continue
            if len(fields) < 3:
                continue
            try:
                dur = float(fields[1])
            except ValueError:
                continue
            # The kernel name is the last quoted field; the CSV writer leaves
            # it at len-2 (a pid follows).
            name = fields[-2] if not fields[-2].isdigit() else fields[-3]
            if not name or name in ("Device", "Pinned", "Pageable"):
                # a memcpy row: fields[-2] is the direction
                name = "memcpy:" + (fields[-2] if name else "?")
            per_class[classify(name)] += dur
            per_kernel[name] += dur
            rows += 1
    total = sum(per_class.values())
    print(f"{path}: {rows} rows, total {total / scale:.2f} ms"
          f"{' / %.2f ms per unit' % (total / scale) if scale == 1 else ''}")
    print("\nby class (ms):")
    for cls, ms in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:10s} {ms / scale:9.2f}  {100.0 * ms / total:5.1f}%")
    print(f"\ntop {top} kernels by total ms:")
    for name, ms in sorted(per_kernel.items(), key=lambda kv: -kv[1])[:top]:
        short = re.sub(r"<.*", "", name)
        short = re.sub(r"void |const |__half", "", short)[:90]
        print(f"  {ms / scale:9.2f}  {short}")


if __name__ == "__main__":
    main()
