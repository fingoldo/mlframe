"""Aggregate the JSONL written by tt_b4_run_detect.py into tables (research B4).

Usage: python tt_b4_analyze_detect.py FILE1.jsonl [FILE2.jsonl ...]
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

STRONG = 0.01  # a column counts as "transform available" when the best judge gain exceeds this (test R2 units)


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def score(o, method, rule):
    cands = o["cands"]
    jd = {int(k): v for k, v in o["judge"].items()}
    acc = {int(j): f for j, f in o["methods"][method]["accepted"][rule].items()}
    inf = {c: (lk, mg) for c, lk, mg in o["inf"]}
    partners = {c for c, _, _ in o["partners"]}
    out = {"nl": [], "lin": [], "partner_acc": 0, "noise_acc": 0, "useless": 0, "cap_real": 0.0, "cap_best": 0.0, "n_acc": len(acc), "partner_real": 0.0, "noise_real": 0.0}
    for c, (lk, mg) in inf.items():
        g = jd.get(c)
        if g is None:
            continue
        best_gain = max(v for v in g if v is not None)
        best_f = cands[max(range(len(g)), key=lambda k: -1e9 if g[k] is None else g[k])]
        pick = acc.get(c)
        real = 0.0 if pick is None else (g[cands.index(pick)] or 0.0)
        if lk == "id":
            out["lin"].append({"accepted": pick is not None, "pick": pick, "real": real})
            continue
        if best_gain < STRONG:
            status = "weak"
        elif pick is None:
            status = "miss"
        elif real >= 0.8 * best_gain:
            status = "hit"
        elif real >= 0.3 * best_gain:
            status = "partial"
        else:
            status = "miss"
        out["nl"].append({"col": c, "link": lk, "marg": mg, "best_f": best_f, "best_gain": best_gain, "pick": pick, "real": real, "status": status})
        out["cap_real"] += max(real, 0.0)
        out["cap_best"] += best_gain
    for j, f in acc.items():
        if j in inf:
            continue
        g = jd.get(j)
        real = 0.0 if g is None or g[cands.index(f)] is None else g[cands.index(f)]
        if j in partners:
            out["partner_acc"] += 1; out["partner_real"] += real
        else:
            out["noise_acc"] += 1; out["noise_real"] += real
        if real < 0.001:
            out["useless"] += 1
    for c in acc:
        if c in inf:
            g = jd.get(c)
            real = 0.0 if g is None or g[cands.index(acc[c])] is None else g[cands.index(acc[c])]
            if real < 0.001 and inf[c][0] == "id":
                out["useless"] += 1
    return out


def table(rows, combos):
    print("method/rule | datasets | nl hit | partial | miss | weak | linear-col transform accepted | partner accepted/ds | other noise accepted/ds | useless adds/ds | captured gain / oracle gain | seconds/ds")
    print("--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---")
    for method, rule in combos:
        st = collections.Counter(); lin_acc = lin_n = 0; pa = na = us = 0; cr = cb = 0.0; secs = []
        for o in rows:
            s = score(o, method, rule)
            for r in s["nl"]:
                st[r["status"]] += 1
            lin_n += len(s["lin"]); lin_acc += sum(r["accepted"] for r in s["lin"])
            pa += s["partner_acc"]; na += s["noise_acc"]; us += s["useless"]; cr += s["cap_real"]; cb += s["cap_best"]
            secs.append(o["methods"][method]["seconds"])
        n = len(rows)
        tot = sum(st.values())
        print(f"{method}/{rule} | {n} | {st['hit']}/{tot} | {st['partial']} | {st['miss']} | {st['weak']} | {lin_acc}/{lin_n} | {pa / n:.2f} | {na / n:.2f} | {us / n:.2f} | "
              f"{cr / max(cb, 1e-12):.2f} | {np.mean(secs):.1f}")


def by_link(rows, method, rule):
    agg = collections.defaultdict(collections.Counter)
    gains = collections.defaultdict(list)
    for o in rows:
        for r in score(o, method, rule)["nl"]:
            agg[r["link"]][r["status"]] += 1
            gains[r["link"]].append(r["best_gain"])
    print(f"per link, {method}/{rule}: link | hit | partial | miss | weak | mean oracle gain (test R2)")
    for lk in sorted(agg):
        c = agg[lk]
        print(f"  {lk:8s} | {c['hit']} | {c['partial']} | {c['miss']} | {c['weak']} | {np.mean(gains[lk]):.3f}")


def picks(rows, method, rule):
    cnt = collections.Counter()
    for o in rows:
        for r in score(o, method, rule)["nl"]:
            cnt[(r["link"], r["pick"] or "-", r["best_f"])] += 1
    print(f"picks (link, picked, oracle-best) for {method}/{rule}:")
    for k, v in sorted(cnt.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    rows = load(sys.argv[1:])
    print(f"datasets: {len(rows)}; p = {sorted({o['p'] for o in rows})}; n_train = {sorted({o['n'] for o in rows})}")
    table(rows, [("M1", r) for r in ("column", "maxT", "pooled", "gated")] + [("M2", r) for r in ("column", "maxT", "pooled", "gated")] + [("M3", "stepwise")])
    for m, r in (("M1", "gated"), ("M2", "gated"), ("M3", "stepwise")):
        by_link(rows, m, r)
    picks(rows, "M3", "stepwise")
