"""Tables for question 1 from the JSONL of tt_b5_invariance.py.  python tt_b5_inv_report.py FILE.jsonl [--steps 0|1] [--detail]

A variant fit is compared with the raw fit of the same fixture / scheme / tag / data seed (random_state 0 in both).  Two levels:
  final = the set of names get_feature_names_out() returns;
  core  = the provenance rows in the output with a finite MI gain (the MI greedy's selection) and the largest |gain difference| over the shared rows.
Families: target (log, cbrt of y), feature (exp, cube, tanh of every column; ``lin`` = the one column replaced by the transform that linearises the link), jitter (raw refit with random_state 1).
"""
import collections
import json
import sys

path = sys.argv[1]
steps_f = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else None
detail = "--detail" in sys.argv
rows = [json.loads(l) for l in open(path) if l.strip()]
base = {}
for r in rows:
    if r["variant"] == "raw" and r["fitseed"] == 0 and not r.get("error"):
        base[(r["fixture"], r["scheme"], r["steps"], r["seed"], r.get("tag", ""))] = r
agg = collections.defaultdict(lambda: [0, 0, 0, 0.0])
per = collections.defaultdict(lambda: [0, 0, 0])
bad = []
for r in rows:
    if steps_f is not None and r["steps"] != steps_f:
        continue
    b = base.get((r["fixture"], r["scheme"], r["steps"], r["seed"], r.get("tag", "")))
    if b is None or r is b or r.get("error"):
        continue
    fam = "jitter" if r["fitseed"] == 1 else ("target" if r["variant"] in ("log", "cbrt") else "feature")
    sub = fam if fam == "jitter" else f"{fam}:{r['variant']}"
    final = set(r["names"]) == set(b["names"])
    rc, bc = dict(r.get("core", [])), dict(b.get("core", []))
    core = set(rc) == set(bc) if "core" in r else None
    gd = max((abs(rc[k] - bc[k]) for k in set(rc) & set(bc)), default=0.0)
    sc = r["scheme"] + ("+" + r["tag"] if r.get("tag") else "")
    k1 = ("steps%d" % r["steps"], sc, fam)
    k2 = ("steps%d" % r["steps"], sc, r["fixture"], fam)
    for d, k in ((agg, k1), (per, k2)):
        d[k][0] += 1
        d[k][1] += final
        d[k][2] += bool(core)
    agg[k1][3] = max(agg[k1][3], gd)
    if not final:
        bad.append((r["steps"], sc, r["fixture"], r["seed"], sub, sorted(set(b["names"]) ^ set(r["names"]))[:6]))
print("| run | scheme | family | cases | identical final names | identical MI core | max abs gain diff on shared core rows |")
print("|---|---|---|---|---|---|---|")
for k in sorted(agg):
    a = agg[k]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {a[0]} | {a[1]} | {a[2]} | {a[3]:.1e} |")
if detail:
    print("\nper fixture: (run, scheme, fixture, family): cases, final identical, core identical")
    for k in sorted(per):
        print(" ", k, per[k])
    print("\nnon-identical final sets (run, scheme, fixture, data seed, variant, symmetric difference with raw):")
    for b in sorted(bad):
        print(" ", b)
