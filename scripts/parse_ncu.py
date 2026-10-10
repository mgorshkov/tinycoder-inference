#!/usr/bin/env python3
"""Parse ncu SpeedOfLight text output: one line per kernel launch (robust to
nested-namespace template headers on both llama.cpp and TinyCoder kernels)."""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ncu_tiny.txt"
text = open(path, errors="replace").read()
lines = text.splitlines()

hdr = re.compile(r"^\s+\S.*\((\d+), (\d+), (\d+)\)x\((\d+), (\d+), (\d+)\), Context")
metrics = ["Duration", "Memory Throughput", "DRAM Throughput",
           "Compute (SM) Throughput", "Achieved Occupancy"]

rows = []
cur = None
for ln in lines:
    m = hdr.match(ln)
    if m:
        if cur:
            rows.append(cur)
        name = ln.strip()
        name = re.sub(r"\(.*$", "", name).strip()
        # keep the last name component for ours, first for llama
        short = name.split("::")[-1] if "::" in name else name
        short = re.sub(r"<.*", "", short)
        cur = {"name": short,
               "grid": f"({m.group(1)},{m.group(2)})x({m.group(4)},{m.group(5)})",
               "vals": {}}
    elif cur is not None:
        for metric in metrics:
            if ln.strip().startswith(metric):
                mm = re.search(r"([\d.]+)\s*$", ln)
                if mm:
                    cur["vals"][metric] = mm.group(1)
if cur:
    rows.append(cur)

for r in rows:
    v = r["vals"]
    print(f'{r["name"]:34s} {r["grid"]:16s} dur={v.get("Duration","?"):>8s} '
          f'dram%={v.get("DRAM Throughput","?"):>6s} sm%={v.get("Compute (SM) Throughput","?"):>6s} '
          f'occ={v.get("Achieved Occupancy","?"):>6s}')
