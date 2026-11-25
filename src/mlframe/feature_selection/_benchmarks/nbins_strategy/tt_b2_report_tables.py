"""B2: compact markdown tables for research_tt_b2.md from the JSONL result files (all numbers come from the files; nothing is typed by hand).

python tt_b2_report_tables.py zoo|naxis|paxis|fzoo
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

import tt_b2_analyze as A

D = "D:/Temp/wt2/audits/2026-11-23/"


def zoo():
    rows = A.load(D + "tt_b2_zoo2_s0.jsonl," + D + "tt_b2_zoo2_s1.jsonl")
    g = defaultdict(list)
    for r in rows:
        g[(r["fam"], r["var"])].append(r)
    print("| target / noise | helps (RMSE) | oracle RMSE ratio | W1 hit, FP, regret | CV hit, FP, regret | helps (MAE) | W1 hit (MAE) | CV hit (MAE) |")
    print("|---|---|---|---|---|---|---|---|")
    for key in sorted(g):
        rr = g[key]
        w, c = A.summarize(rr, "W1"), A.summarize(rr, "CV")
        wm, cm = A.summarize(rr, "W1", "mae"), A.summarize(rr, "CV", "mae")
        print(f"| {key[0]} / {key[1]} | {w['helps']}/{w['n']} | {w['oracle']:.3f} | {w['hit']}/{w['helps']}, {w['fp']}/{w['nohelp']}, {w['regret']:.3f} | {c['hit']}/{c['helps']}, {c['fp']}/{c['nohelp']}, {c['regret']:.3f} | "
              f"{wm['helps']}/{wm['n']} | {wm['hit']}/{wm['helps']} | {cm['hit']}/{cm['helps']} |")
    w, c = A.summarize(rows, "W1"), A.summarize(rows, "CV")
    wm, cm = A.summarize(rows, "W1", "mae"), A.summarize(rows, "CV", "mae")
    print(f"| **all 28 cells** | {w['helps']}/{w['n']} | {w['oracle']:.3f} | {w['hit']}/{w['helps']}, {w['fp']}/{w['nohelp']}, {w['regret']:.3f} | {c['hit']}/{c['helps']}, {c['fp']}/{c['nohelp']}, {c['regret']:.3f} | {wm['helps']}/{wm['n']} | {wm['hit']}/{wm['helps']} | {cm['hit']}/{cm['helps']} |")


def stages():
    rows = A.load(D + "tt_b2_zoo2_s0.jsonl," + D + "tt_b2_zoo2_s1.jsonl")
    print("| stage | hit/helps | non-identity picks when nothing helps | of which worse than identity by >0.5% | worse by >2% | mean pick/identity RMSE | worst | regret |")
    print("|---|---|---|---|---|---|---|---|")
    for c, label in (("LL", "LL argmax (Jacobian, YJ near 1 collapsed)"), ("LLnull", "LL + sign-flip null"), ("W1", "LL + null + held-out gate (wave-1 recipe)"), ("G2p", "same with paired sign-flip gate"),
                     ("RG", "Huber-robust LL + null + paired gate"), ("EG", "extended set (+log1p, sqrt) + paired gate"), ("S", "W1 + support abstention"), ("CV", "cross-fitted original-scale max-T (CV)"), ("H", "CV, else W1")):
        s = A.summarize(rows, c)
        cost = sum(1 for r in rows if A.classify(r, c, "rmse")[0] in ("FP", "HARM") and A.classify(r, c, "rmse")[1] > 1.005)
        print(f"| {label} | {s['hit']}/{s['helps']} | {s['fp']}/{s['nohelp']} | {cost} | {s['harm']} | {s['ratio']:.4f} | {s['worst']:.3f} | {s['regret']:.4f} |")
    s = A.summarize(rows, "W1")
    print(f"(oracle mean ratio {s['oracle']:.4f}; 196 runs = 28 cells x 7 seeds)")


def naxis():
    rows = A.load(D + "tt_b2_tn_s0.jsonl," + D + "tt_b2_tn_s1.jsonl")
    print("| n | runs | helps | LL null + gate (W1): hit, FP/nohelp, regret | + unclipped null (N_W1) | CV: hit, FP/nohelp, regret | oracle ratio |")
    print("|---|---|---|---|---|---|---|")
    for n in (300, 1000, 5000, 20000):
        rr = [r for r in rows if r["n"] == n]
        out = []
        for c in ("W1", "N_W1", "CV"):
            s = A.summarize(rr, c)
            out.append(f"{s['hit']}/{s['helps']}, {s['fp']}/{s['nohelp']}, {s['regret']:.3f}")
        s = A.summarize(rr, "W1")
        print(f"| {n} | {s['n']} | {s['helps']} | " + " | ".join(out) + f" | {s['oracle']:.3f} |")


def fzoo():
    rows = A.load(D + "tt_b2_feat1.jsonl")
    from tt_b2_core import outcome
    from collections import Counter
    oc, reg = Counter(), []
    for r in rows:
        ra = A._ratios(r["rmse"], "id")
        nonid = {k: v for k, v in ra.items() if k != "id"}
        b = min(nonid, key=nonid.get)
        pk = r["pick"]
        rp = ra[pk] if pk != "id" else 1.0
        oc[outcome(rp, nonid[b], picked_id=(pk == "id"))] += 1
        reg.append(rp / min(nonid[b], 1.0))
    print(dict(oc), "mean regret", round(float(np.mean(reg)), 4), "runs", len(rows))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
