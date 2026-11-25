"""Markdown tables of the B3 report from the jsonl results (agent B3, 2026-11-24).

``python tt_b3_tables.py RESULT_DIR`` expects e1a/e1b (n 2000, p 5), e2a/e2b/e2c (n 20000, p 5), e3a/e3b (n 2000, p 40), e4a/e4b (n 20000, p 40) in RESULT_DIR; blocks that are missing are skipped.
Tables: headline per block (ridge, smeared RMSE, full oracle) ; per fixture ; other models and metrics ; time per call.
"""

from __future__ import annotations

import os
import sys

import numpy as np

import tt_b3_analyze as A
import tt_b3_fixtures as F

CORE = ("lin", "lin_hetero", "lin_mixed", "lin_t3", "lin_pos")  # controls with a linear mean and symmetric noise: the null of the test holds exactly
BLOCKS = {
    "n=2000 p=5": ["e1a", "e1b"], "n=20000 p=5": ["e2a", "e2b", "e2c"], "n=2000 p=40": ["e3a", "e3b"], "n=20000 p=40": ["e4a", "e4b"],
}


def load_block(d: str, names):
    files = [os.path.join(d, f"{n}.jsonl") for n in names if os.path.exists(os.path.join(d, f"{n}.jsonl"))]
    return A.load(files) if files else []


def stats(rows, m):
    rs = [(r, r[m]) for r in rows if m in r]
    if not rs:
        return None
    good = np.mean([d["regret"] <= 1.02 if r["gain"] <= 0.98 else d["ratio"] <= 1.01 for r, d in rs])
    reg = np.array([d["regret"] for r, d in rs])
    rat = np.array([d["ratio"] for r, d in rs])
    gain = [d for r, d in rs if r["gain"] <= 0.98]
    miss = np.mean([d["name"] == "identity" for d in gain]) if gain else float("nan")
    ctrl = [d for r, d in rs if F.KIND.get(r["kind"]) == "control"]
    fpc = np.mean([d["name"] != "identity" for d in ctrl]) if ctrl else float("nan")
    harm = np.mean([d["ratio"] > 1.02 for r, d in rs])
    ab = np.mean([d["abstain"] for r, d in rs])
    return len(rs), good, reg.mean(), np.median(reg), rat.mean(), rat.max(), miss, fpc, harm, ab


def fmt(x, p=0):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{100 * x:.{p}f}%" if p is not None else str(x))


def main(d: str) -> None:
    methods = ["ref", "old_ext_search", "old_search", "old_mi", "old_ext_mi"]
    print("## Headline: ridge, smeared RMSE, full oracle (good = within 2 percent of the best candidate on gain cases, no loss on no-gain cases)\n")
    print("| block | slice | method | cases | good | mean regret | median regret | mean ratio | worst ratio | miss on gain | FP on controls | harm >2% | abstain |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for blk, names in BLOCKS.items():
        recs = load_block(d, names)
        if not recs:
            continue
        rows = A.case_rows(recs, "ridge", "rmse_smear", methods, "full")
        for sl, pred in (("transform", lambda r: F.KIND[r["kind"]] == "transform"), ("transform (old set can express the answer)", lambda r: F.KIND[r["kind"]] == "transform" and r["kind"] not in ("recip", "logit_bnd")), ("ambiguous", lambda r: F.KIND[r["kind"]] == "ambiguous"), ("control", lambda r: F.KIND[r["kind"]] == "control"), ("core control", lambda r: r["kind"] in CORE), ("all", lambda r: True)):
            sub = [r for r in rows if pred(r)]
            for m in methods:
                s = stats(sub, m)
                if s is None:
                    continue
                n, good, reg, med, rat, worst, miss, fpc, harm, ab = s
                print(f"| {blk} | {sl} | {m} | {n} | {fmt(good)} | {reg:.3f} | {med:.3f} | {rat:.3f} | {worst:.2f} | {fmt(miss)} | {fmt(fpc)} | {fmt(harm)} | {fmt(ab)} |")
    print("\n## Per fixture: ridge RMSE, full oracle: good / mean regret (ref vs old_search), oracle best/identity\n")
    print("| fixture | kind | " + " | ".join(f"{b} ref | {b} old_search | {b} oracle" for b in BLOCKS) + " |\n|---|---|" + "---|---|---|" * len(BLOCKS))
    cache = {}
    for blk, names in BLOCKS.items():
        recs = load_block(d, names)
        cache[blk] = A.case_rows(recs, "ridge", "rmse_smear", ["ref", "old_ext_search", "old_search"], "full") if recs else []
    for fx in F.FIXTURES:
        cells = []
        for blk in BLOCKS:
            rs = [r for r in cache[blk] if r["kind"] == fx]
            if not rs:
                cells += ["-", "-", "-", "-"]
                continue
            a, b, c = stats(rs, "ref"), stats(rs, "old_ext_search"), stats(rs, "old_search")
            cells += [f"{fmt(a[1])} / {a[2]:.3f}", "-" if b is None else f"{fmt(b[1])} / {b[2]:.3f}", f"{fmt(c[1])} / {c[2]:.3f}", f"{np.mean([r['gain'] for r in rs]):.3f}"]
        print(f"| {fx} | {F.KIND[fx]} | " + " | ".join(cells) + " |")
    print("\n## Other models and metrics (all fixtures of the block; good / mean regret / worst ratio / FP on controls)\n")
    print("| block | model, metric | ref | old_search | old_mi |\n|---|---|---|---|---|")
    for blk, names in BLOCKS.items():
        recs = load_block(d, names)
        if not recs:
            continue
        for model, metric in (("ridge", "mae_clip"), ("ridge_spline", "rmse_smear"), ("ridge_spline", "mae_clip"), ("hgb", "rmse_smear"), ("hgb", "mae_clip")):
            rows = A.case_rows(recs, model, metric, ["ref", "old_ext_search", "old_search", "old_mi"], "full")
            if not rows:
                continue
            cells = []
            for m in ("ref", "old_ext_search", "old_search", "old_mi"):
                s = stats(rows, m)
                cells.append("-" if s is None else f"{fmt(s[1])} / {s[2]:.3f} / {s[5]:.2f} / {fmt(s[7])}")
            print(f"| {blk} | {model}, {metric} | " + " | ".join(cells) + " |")
    print("\n## Wall time per call (seconds, median over cases; measured while 4 other agents and 2 own workers ran, so an upper bound)\n")
    print("| block | reference linear head | reference spline head | old search (MI top-3 + ridge gate) |\n|---|---|---|---|")
    for blk, names in BLOCKS.items():
        recs = load_block(d, names)
        if not recs:
            continue
        tl = [r["time"].get("linear_f1") for r in recs if r["time"].get("linear_f1")]
        ts = [r["time"].get("spline_f1") for r in recs if r["time"].get("spline_f1")]
        to = [r["time"].get("old") for r in recs if r["time"].get("old")]
        print(f"| {blk} | {np.median(tl):.2f} (p90 {np.percentile(tl, 90):.2f}) | {f'{np.median(ts):.2f} (p90 {np.percentile(ts, 90):.2f})' if ts else '-'} | {np.median(to):.2f} (p90 {np.percentile(to, 90):.2f}) |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
