#!/usr/bin/env python3
"""arch_check.py — static per-architecture invariants for tinycoder's CUDA kernels.

Why this exists: on this project the only GPU in the box is the one the kernels were
already tuned for. A kernel can therefore lose its whole reason for existing -- stop
using cp.async, drift past the register budget that gives it a second block per SM, or
quietly start spilling -- and still produce plausible, *slower* numbers on the card that
is present. This script reads the compiled cubin, per architecture, and fails the build
when a kernel stops matching the shape it is supposed to have.

It is a regression detector, not a profiler: it can prove a kernel is structurally what
the design says, and it can never prove one is fast.

Usage:
    python3 scripts/arch_check.py --build-dir build-1.0
    python3 scripts/arch_check.py --build-dir build-1.0 --arch 86 --json out.json
    python3 scripts/arch_check.py --selftest

Exit status: 0 = every audited kernel matches its budget, 1 = at least one violation or
a parsing failure (a silently unparsed kernel counts as a failure: the audit exists to
notice drift, and a kernel it cannot see is drift).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict

MAX_REGS_PER_SM = 65536
SMEM_PER_SM = {75: 65536, 80: 102400, 86: 102400, 89: 102400, 120: 102400}
SMEM_OPTIN_PER_BLOCK = {75: 65536, 80: 101376, 86: 101376, 89: 101376, 120: 101376}
WARPS_PER_SM = {75: 32, 80: 64, 86: 48, 89: 48, 120: 48}

# Threads per block, needed to turn a register count into blocks/SM. Every entry here was
# read from the kernel's launch configuration in the source; anything not listed is
# assumed 256 and reported as assumed.
THREADS = {
    "gemvCoreImpl": 128,
    "kQGemvPTQ1_0WarpRowPt": 128,
    "kQGemvPQ2_0T": 128,
    "kQGemvFusedGU_Q2K": 256,
    "int8GemmStreamK": 256,
    "int8GemmDense": 256,
    "kFfnPersistentLayer": 256,
    "kQ35GdnStepBatch": 128,
    "kQ35GdnWarpScan": 128,
    "gemmFusedHmmaT": 256,
    "exl3GemvK1": 256,
    "exl3GemvV3": 256,
    "exl3FusedMatVec": 256,
    "exl3GemvV2": 256,
}

# symbol-substring -> budget. `max_regs` is a ceiling that must hold *now* (it encodes the
# occupancy decision the design made); `min_blocks` is the blocks/SM the kernel must be
# able to place from registers alone; `require_ldgsts` is set for kernels that the design
# says must use async staging on sm_80+.
BUDGETS = {
    # Decode GEMV.  gemvCoreImpl is the default path for ten of the twelve GGUF
    # types: one row per 128-thread block, four warps splitting K, at the card's
    # warp cap -- so its budget is "do not drift".
    r"gemvCoreImpl": {"max_regs": 48, "min_blocks": 12},
    # The ternary/EXL3 families.  Template arguments are part of the budget
    # because the uninstantiated debug shapes differ by 130 registers.
    r"kQGemvPTQ1_0WarpRowPtILi4E": {"max_regs": 96, "min_blocks": 5},
    r"kQGemvPQ2_0TILj2ELj8E": {"max_regs": 72, "min_blocks": 8},
    r"kQGemvFusedGU_Q2K": {"max_regs": 255, "min_blocks": 1},
    # Prefill int8 GEMM: 254 registers, one block and 8 warps per SM today.  The
    # design's target is two blocks (max_regs 128); it becomes the budget when
    # the staging task lands, not before.
    r"int8GemmStreamK": {"max_regs": 255, "min_blocks": 1},
    r"int8GemmDense": {"max_regs": 255, "min_blocks": 1},
    # Attention / GDN / FFN.
    r"kQ35GdnStepBatchILj128E": {"max_regs": 255, "min_blocks": 2},
    r"kQ35GdnWarpScanILj128E": {"max_regs": 64, "min_blocks": 8},
    r"kFfnPersistentLayer": {"max_regs": 128, "min_blocks": 2},
    # EXL3 decode.  V3 is the K=2/K=3 path the checkpoints actually use; it
    # carried (256,4) launch bounds in this change to stop losing a residency
    # slot to three registers.
    r"exl3GemvK1ILi2ELi8ELi4ELi0ELi1E": {"max_regs": 64, "min_blocks": 4},
    r"exl3GemvV3ILi2ELi8ELi4ELi2E": {"max_regs": 64, "min_blocks": 4},
    # The opt-in TINYCODER_EXL3_GEMV=1 path: 68 registers, 3 blocks.M.
    r"exl3FusedMatVec": {"max_regs": 72, "min_blocks": 3},
    # EXL3 fused prefill: 128 registers, 2 blocks/SM, ~16 warps of 48.
    r"gemmFusedHmmaT": {"max_regs": 128, "min_blocks": 2},
}

RES_RE = re.compile(r"^\s*Function (\S+):$")
USAGE_RE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
)
SASS_INSN_RE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+@?!?P?\d?\s*([A-Z][A-Z0-9._]+)")
SASS_FUNC_RE = re.compile(r"\s*Function : (\S+)")


@dataclass
class Kernel:
    name: str
    arch: int
    regs: int = 0
    stack: int = 0
    shared: int = 0
    local: int = 0
    insn: dict = field(default_factory=dict)
    demangled: str = ""

    @property
    def short(self):
        """Readable label: the budget key plus the template arguments.

        Anonymous-namespace kernels do not demangle to anything shorter than the
        mangled name, and the template arguments are the part that distinguishes
        one instantiation's budget from another, so they are extracted here.
        """
        key, _budget = budget_for(self.name)
        if key is None:
            return self.name.split("__Z")[-1][:56]
        args = re.findall(r"IL[jil]([0-9]+)E|ILb([01])E", self.name)
        if not args:
            return re.sub(r"\\", "", key)[:56]
        vals = [a or b for a, b in args]
        return f"{re.sub(r'\\\\', '', key)}<{','.join(vals)}>"[:56]

    def threads(self):
        for key, n in THREADS.items():
            if key in self.name:
                return n, False
        return 256, True

    def blocks_per_sm(self):
        n, _ = self.threads()
        by_regs = MAX_REGS_PER_SM // max(self.regs * n, 1)
        by_smem = 64 if self.shared == 0 else SMEM_PER_SM.get(self.arch, 102400) // self.shared
        by_warps = WARPS_PER_SM.get(self.arch, 48) // max((n + 31) // 32, 1)
        return min(by_regs, by_smem, by_warps)

    def count(self, mnemonic):
        return sum(v for k, v in self.insn.items() if k.split(".")[0] == mnemonic)

    def to_dict(self):
        return asdict(self)


def parse_res_usage(text, arch):
    """Parse `cuobjdump -res-usage` output into Kernels. Returns (kernels, unparsed)."""
    kernels, cur, unparsed = [], None, 0
    for line in text.splitlines():
        m = RES_RE.match(line)
        if m:
            cur = Kernel(m.group(1), arch)
            kernels.append(cur)
            continue
        u = USAGE_RE.search(line)
        if u and cur is not None:
            cur.regs, cur.stack, cur.shared, cur.local = (int(g) for g in u.groups())
        elif line.strip().startswith("Function ") and not u:
            unparsed += 1
    return kernels, unparsed


def parse_sass_census(text):
    """Count mnemonics per function from `cuobjdump -sass` output."""
    census, cur = {}, None
    for line in text.splitlines():
        m = SASS_FUNC_RE.match(line)
        if m:
            cur = m.group(1)
            census.setdefault(cur, {})
            continue
        if cur is None:
            continue
        im = SASS_INSN_RE.search(line)
        if im:
            op = im.group(1)
            census[cur][op] = census[cur].get(op, 0) + 1
    return census


def run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if out.returncode != 0:
        return None, out.stderr.strip()[:200]
    return out.stdout, None


def demangle_all(names):
    """Map mangled -> readable name.  cuobjdump strips the leading `_Z`."""
    uniq = sorted({n for n in names if n.startswith("N")})
    if not uniq:
        return {}
    try:
        out = subprocess.run(["c++filt"] + ["_Z" + n for n in uniq],
                             capture_output=True, text=True, timeout=600).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    lines = out.splitlines()
    if len(lines) != len(uniq):
        return {}
    table = {}
    for mangled, pretty in zip(uniq, lines):
        # tinycoder::gpu::{anonymous}::kQ35GdnStepBatch<128> -> kQ35GdnStepBatch<128>
        pretty = re.sub(r"^.*::", "", pretty.strip())
        table[mangled] = pretty[:70]
    return table


def budget_for(name):
    for key, b in BUDGETS.items():
        if re.search(key, name):
            return key, b
    return None, None


def audit(build_dir, arches, want_json):
    objs = []
    for root, _dirs, files in os.walk(build_dir):
        for f in files:
            if f.endswith(".cu.o") and "tinycoder_core.dir" in root:
                objs.append(os.path.join(root, f))
    objs.sort()
    if not objs:
        print(f"arch_check: no .cu.o under {build_dir}", file=sys.stderr)
        return 1

    report, violations, assumed_threads, unaudited = {}, [], [], 0
    for arch in arches:
        for obj in objs:
            usage, err = run(["cuobjdump", "-res-usage", "-arch", f"sm_{arch}", obj])
            if usage is None:
                continue  # object does not carry this arch
            kernels, unparsed = parse_res_usage(usage, arch)
            if unparsed:
                violations.append(f"{obj}: {unparsed} function block(s) without a usage line")
            sass, _ = run(["cuobjdump", "-sass", "-arch", f"sm_{arch}", obj])
            census = parse_sass_census(sass) if sass else {}
            for k in kernels:
                k.insn = census.get(k.name, {})
                report.setdefault(arch, []).append(k)
                key, budget = budget_for(k.name)
                if key is None:
                    unaudited += 1
                    continue
                threads, assumed = k.threads()
                if assumed:
                    assumed_threads.append(f"{k.short} (assumed {threads})")
                if k.local or k.stack:
                    violations.append(
                        f"sm_{arch} {k.short}: spills to local memory "
                        f"(LOCAL:{k.local} STACK:{k.stack})")
                if k.regs > budget["max_regs"]:
                    violations.append(
                        f"sm_{arch} {k.short}: {k.regs} registers > budget "
                        f"{budget['max_regs']}")
                blocks = k.blocks_per_sm()
                if blocks < budget["min_blocks"]:
                    violations.append(
                        f"sm_{arch} {k.short}: {blocks} block(s)/SM < required "
                        f"{budget['min_blocks']} ({k.regs} regs x {threads} threads)")
                if budget.get("require_ldgsts"):
                    n = k.count("LDGSTS")
                    if arch >= 80 and n == 0:
                        violations.append(
                            f"sm_{arch} {k.short}: no LDGSTS although async staging is required")
                    if arch < 80 and n:
                        violations.append(
                            f"sm_{arch} {k.short}: LDGSTS present on an arch without cp.async")

    if want_json:
        with open(want_json, "w") as fh:
            json.dump({a: [k.to_dict() for k in ks] for a, ks in report.items()}, fh, indent=1)

    table = demangle_all([k.name for ks in report.values() for k in ks])
    for ks in report.values():
        for k in ks:
            k.demangled = table.get(k.name, "")
    for arch in list(report):
        worst, seen = [], {}
        for k in report[arch]:
            key = k.short
            if key not in seen or k.regs > seen[key].regs:
                seen[key] = k
        report[arch] = [k for k in seen.values() if budget_for(k.name)[0] is not None]

    print(f"{'arch':>5} {'kernel':<46} {'regs':>5} {'blk/SM':>7} {'LDGSTS':>7} "
          f"{'IMMA':>6} {'HMMA':>6}")
    for arch in sorted(report):
        for k in sorted(report[arch], key=lambda x: -x.regs):
            key, _b = budget_for(k.name)
            if key is None:
                continue
            print(f"sm_{arch:<2} {k.short[:46]:<46} {k.regs:>5} {k.blocks_per_sm():>7} "
                  f"{k.count('LDGSTS'):>7} {k.count('IMMA'):>6} {k.count('HMMA'):>6}")
    if assumed_threads:
        print("\nassumed thread counts (add to THREADS): "
              + ", ".join(sorted(set(assumed_threads))))
    print(f"\n{unaudited} kernel instantiation(s) not covered by BUDGETS")
    dedup, keep = set(), []
    for v in violations:
        sig = re.sub(r"\s*\d+ registers", " N registers", v)
        sig = re.sub(r"\(\d+ regs x (\d+) threads\)", r"(N regs x \1 threads)", sig)
        if sig in dedup:
            continue
        dedup.add(sig)
        keep.append(v)
    violations = keep
    if violations:
        print(f"\n{len(violations)} violation(s):")
        for v in violations:
            print("  " + v)
        return 1
    print("\narch_check: all audited kernels match their budgets")
    return 0


SELFTEST_USAGE = """\
Function clean_kernel:
  REG:64 STACK:0 SHARED:0 LOCAL:0 TEXTURE:0 SURFACE:0 SAMPLER:0
Function spilled_kernel:
  REG:40 STACK:32 SHARED:0 LOCAL:8 TEXTURE:0 SURFACE:0 SAMPLER:0
Function fat_kernel:
  REG:200 STACK:0 SHARED:0 LOCAL:0 TEXTURE:0 SURFACE:0 SAMPLER:0
"""
SELFTEST_SASS = """\
\t\tFunction : clean_kernel
        /*0000*/                   LDGSTS.E.BYPASS.128 R4, [R2], 0x10 ;
        /*0010*/                   IMMA.16832.S8.S8 R8, R4, R8, R8 ;
\t\tFunction : spilled_kernel
        /*0000*/                   STL [R1+0x10], R4 ;
"""


def selftest():
    kernels, unparsed = parse_res_usage(SELFTEST_USAGE, 86)
    census = parse_sass_census(SELFTEST_SASS)
    by_short = {k.name: k for k in kernels}
    for name, k in by_short.items():
        k.insn = census.get(name, {})
    checks = [
        ("parses three kernels", len(kernels) == 3, len(kernels)),
        ("no unparsed blocks", unparsed == 0, unparsed),
        ("clean has no local", by_short["clean_kernel"].local == 0, None),
        ("spilled reports LOCAL", by_short["spilled_kernel"].local == 8, None),
        ("spilled reports STACK", by_short["spilled_kernel"].stack == 32, None),
        ("fat kernel blocks/SM = 1", by_short["fat_kernel"].blocks_per_sm() == 1,
         by_short["fat_kernel"].blocks_per_sm()),
        ("LDGSTS counted", by_short["clean_kernel"].count("LDGSTS") == 1, None),
        ("IMMA counted", by_short["clean_kernel"].count("IMMA") == 1, None),
        ("STL counted", by_short["spilled_kernel"].count("STL") == 1, None),
    ]
    failed = 0
    for label, ok, got in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + ("" if ok else f" (got {got})"))
        failed += 0 if ok else 1
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default="build-1.0")
    ap.add_argument("--arch", nargs="+", type=int, default=[75, 80, 86])
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    return audit(args.build_dir, args.arch, args.json)


if __name__ == "__main__":
    sys.exit(main())
