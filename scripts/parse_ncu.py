#!/usr/bin/env python3
"""Parse ncu text output: one line per kernel launch (robust to nested-namespace
template headers on both the reference and TinyCoder kernels).

Works on both the SpeedOfLight per-kernel census and the deeper
SpeedOfLight+WarpStateStats+SchedulerStats+Occupancy capture: metrics that are
absent from a given capture are simply printed as "-".

Usage: parse_ncu.py [capture.txt] [--sort dur|name]
"""
import re
import sys

path = "/tmp/ncu_tiny.txt"
sort_key = "name"
args = [a for a in sys.argv[1:] if a != "--sort"]
for i, a in enumerate(sys.argv[1:]):
    if a == "--sort" and i + 2 <= len(sys.argv[1:]):
        sort_key = sys.argv[i + 2]
if args:
    path = args[0]

text = open(path, errors="replace").read()
lines = text.splitlines()

hdr = re.compile(r"^\s+\S.*\((\d+)(?:, (\d+))?(?:, (\d+))?\)x"
                 r"\((\d+)(?:, (\d+))?(?:, (\d+))?\).*"
                 r"(?:Context \d+|Device \d+, CC [\d.]+)")

# Metric name -> (column header, printed width).  Order = print order.
METRICS = [
    ("Duration", "dur(us)", 8),
    ("DRAM Throughput", "dram%", 6),
    ("Memory Throughput", "mem%", 6),
    ("Compute (SM) Throughput", "sm%", 6),
    ("Achieved Occupancy", "occ%", 6),
    ("Theoretical Occupancy", "thocc%", 6),
    ("One Or More Eligible", "elig%", 6),
    ("Issued Warp Per Scheduler", "issue/sc", 8),
    ("Active Warps Per Scheduler", "actw/sc", 7),
    ("Eligible Warps Per Scheduler", "eligw/sc", 8),
    ("Warp Cycles Per Issued Instruction", "wp/inst", 7),
    ("Executed Instructions", "execinst", 8),
]

def last_num(ln):
    m = re.search(r"([\d][\d,]*\.?\d*)\s*$", ln)
    return m.group(1).replace(",", "") if m else None

rows = []
cur = None
for ln in lines:
    m = hdr.match(ln)
    if m:
        if cur:
            rows.append(cur)
        name = re.sub(r"\(.*$", "", ln.strip()).strip()
        # keep the last template-qualified component for ours, and the function
        # name for the reference (their lines are `mul_mat_vec_q<...>`)
        short = name.split("::")[-1] if "::" in name else name
        short = re.sub(r"<.*", "", short).strip()
        short = re.sub(r"^void\s+", "", short)
        gx = m.group(1); gy = m.group(2) or "1"; gz = m.group(3) or "1"
        bx = m.group(4); by = m.group(5) or "1"; bz = m.group(6) or "1"
        cur = {"name": short,
               "grid": f"({gx},{gy})x({bx},{by})",
               "vals": {}}
    elif cur is not None:
        s = ln.strip()
        for metric, _, _ in METRICS:
            if s.startswith(metric + " ") or s == metric:
                v = last_num(ln)
                if v is not None:
                    cur["vals"][metric] = v
if cur:
    rows.append(cur)

if sort_key == "dur":
    rows.sort(key=lambda r: -float(r["vals"].get("Duration", 0) or 0))

hdr_cols = "".join(f'{h:>{w}s} ' for _, h, w in METRICS)
print(f'{"kernel":34s} {"grid":14s} {hdr_cols}')
for r in rows:
    cols = ""
    for metric, _, w in METRICS:
        v = r["vals"].get(metric, "-")
        cols += f'{v:>{w}s} '
    print(f'{r["name"]:34s} {r["grid"]:14s} {cols}')

# Aggregate duration per kernel name (helps read the census at a glance).
dur_rows = [r for r in rows if "Duration" in r["vals"]]
if dur_rows:
    print("\n-- duration totals (us) --")
    agg = {}
    for r in dur_rows:
        agg.setdefault(r["name"], []).append(float(r["vals"]["Duration"]))
    for n in sorted(agg, key=lambda k: -sum(agg[k])):
        d = agg[n]
        print(f'{n:34s} n={len(d):3d}  sum={sum(d):9.2f}  mean={sum(d)/len(d):8.2f}')
