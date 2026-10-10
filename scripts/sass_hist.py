#!/usr/bin/env python3
"""SASS opcode histogram per kernel function, filtered by a regex.
Usage: cuobjdump -sass FILE | python3 scripts/sass_hist.py <regex>"""
import re
import sys
from collections import Counter

pat = re.compile(sys.argv[1])
cur = None
cnt = Counter()
active = False
for ln in sys.stdin:
    m = re.match(r"\s*Function : (\S+)", ln)
    if m:
        cur = m.group(1)
        active = bool(pat.search(cur))
        continue
    if active:
        im = re.search(r"/\*[0-9a-f]+\*/\s+@?!?P?\d?\s*([A-Z][A-Z0-9._]+)", ln)
        if im:
            cnt[im.group(1)] += 1
tot = sum(cnt.values())
print(f"# {sys.argv[1]}: {tot} instructions")
for op, c in cnt.most_common(24):
    print(f"{op:14s} {c:6d}  {100.0*c/max(tot,1):5.1f}%")
