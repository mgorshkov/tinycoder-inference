#!/usr/bin/env python3
"""Means and spreads from the raw logs written by bonsai_baseline_bench.sh."""
import glob, os, re, statistics as st, sys

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bonsai_ab"
rows = {}
for path in sorted(glob.glob(os.path.join(out, "*.log"))):
    base = os.path.basename(path)[:-4]
    engine, tag, pair = base.split("_")
    txt = open(path, "rb").read().decode("utf-8", "replace")
    if engine == "tc":
        m = re.search(r"pp \d+ tok @ [^|]*\|\| ([0-9.]+) \+/- [0-9.]+ tok/s \(warm reps\)"
                      r" \|\| tg \d+ tok @ [^|]*\|\| ([0-9.]+) \+/-", txt)
    else:
        m = None
        pp = tg = None
        for line in txt.splitlines():
            if "pp512" in line:
                pp = re.findall(r"\|\s*([0-9.]+) [±+-]", line)
            if "tg128" in line:
                tg = re.findall(r"\|\s*([0-9.]+) [±+-]", line)
        if pp and tg:
            m = None
            rows.setdefault((engine, tag), []).append((float(pp[0]), float(tg[0])))
            continue
    if m:
        rows.setdefault((engine, tag), []).append((float(m.group(1)), float(m.group(2))))
    else:
        print("PARSE FAIL:", path)

names = {"tc": "TinyCoder", "lb": "PrismML-llama.cpp"}
print("\n%-18s %-6s %9s %9s %10s %10s %5s" %
      ("engine", "file", "pp512", "tg128", "pp spread", "tg spread", "n"))
for (eng, tag), v in sorted(rows.items()):
    pp = [x[0] for x in v]
    tg = [x[1] for x in v]
    print("%-18s %-6s %9.1f %9.2f %10.1f %10.2f %5d" %
          (names.get(eng, eng), tag, st.mean(pp), st.mean(tg),
           (max(pp) - min(pp)) / 2, (max(tg) - min(tg)) / 2, len(v)))
