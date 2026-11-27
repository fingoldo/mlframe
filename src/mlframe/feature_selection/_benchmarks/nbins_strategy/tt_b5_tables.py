"""Print the Q2 tables (markdown) from tt_b5_bench.jsonl, tt_b5_mrmr_baseline.jsonl.   python tt_b5_tables.py BENCH.jsonl [MRMR.jsonl]"""
import json
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
by = defaultdict(dict)
for r in rows:
    by[(r["kind"], r["n"], r["p"])][r.get("variant", "")] = r


def cell(r, key="cpu_user_s"):
    if r is None:
        return "-"
    if "timeout_s" in r:
        return "timeout"
    return f"{r[key]:.1f}"


def table(kind, variants, title, key="cpu_user_s"):
    print(f"\n### {title}  ({key})")
    print("| n \\ p | " + " | ".join(f"{p} {v}" for p in (20, 200, 1000) for v in variants if not (kind == "target" and v == "cap64" and p == 20)) + " |")
    print("|---|" + "---|" * sum(1 for p in (20, 200, 1000) for v in variants if not (kind == "target" and v == "cap64" and p == 20)))
    for n in (20000, 100000, 1000000):
        out = []
        for p in (20, 200, 1000):
            for v in variants:
                if kind == "target" and v == "cap64" and p == 20:
                    continue
                out.append(cell(by.get((kind, n, p), {}).get(v), key))
        print(f"| {n} | " + " | ".join(out) + " |")


for key in ("cpu_user_s", "compute_s", "peak_wset_MB"):
    table("target", ["full", "cap64"], "TARGET criterion, K=9, B=40, design = all p columns (full) or the 64 best-correlated (cap64)", key)
    table("feature", ["obs_only", "null_sub50k", "null_all"], "FEATURE criterion (numba kernels), K=9; obs_only = no null, null_sub50k = null on 50k rows, null_all = exact null on all rows", key)
    table("feature_np", ["obs_only", "null_sub50k", "null_all"], "FEATURE criterion (numpy SIMD blocks), same variants", key)
print("\n### baselines (one fit)")
print("| kind | n | p | cpu_user_s | wall s | peak MB |")
print("|---|---|---|---|---|---|")
for (kind, n, p), v in sorted(by.items()):
    if kind in ("ridge", "hgb"):
        for _var, r in v.items():
            if "timeout_s" in r:
                print(f"| {kind} | {n} | {p} | timeout | - | - |")
            else:
                print(f"| {kind} | {n} | {p} | {r['cpu_user_s']} | {r['compute_s']} | {r['peak_wset_MB']} |")
if len(sys.argv) > 2:
    print("\n### MRMR fits")
    for l in open(sys.argv[2], encoding="utf-8"):
        print(l.strip()[:300])
print("\n### stages of the target criterion (compute s; wall minus data generation)")
for (kind, n, p), v in sorted(by.items()):
    if kind == "target":
        for var, r in v.items():
            if "stages" in r:
                print(f"target n={n} p={p} {var}: {r['stages']}  cpu={r['cpu_user_s']}")
