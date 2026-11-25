"""Tables for the B3 report from the jsonl files written by tt_b3_run.py (agent B3, 2026-11-24).

Definitions (per case, model, metric; score = original-scale RMSE of the smeared mean prediction, or MAE of the median prediction, on the untouched test block):
* ``best``: lowest score over every candidate scored by the judge (tt_a4 set + signed_square, reciprocal, regression-lambda Yeo-Johnson + the reference picks as fitted).
* ``ratio`` = score(pick) / score(identity); ``regret`` = score(pick) / score(best) (>= 1).
* ``gain case``: best / identity <= 0.98 (some offered transform helps by at least 2 percent). ``no-gain case``: otherwise.
* ``good``: gain case -> regret <= 1.02 (the pick keeps all but 2 percent of what the oracle can win); no-gain case -> ratio <= 1.01 (nothing lost vs identity).
* ``FP``: a non-identity pick on a no-gain case; ``harm``: ratio > 1.02 whatever the case. ``miss``: identity picked on a gain case. ``abstain``: the method returned identity.

Usage: ``python tt_b3_analyze.py FILE.jsonl [FILE2.jsonl ...] [--by fixture|all] [--model ridge] [--metric rmse_smear]``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import List

import numpy as np

import tt_b3_fixtures as F

PAIR = {"ridge": "linear", "hgb": "linear", "ridge_spline": "spline"}
METRIC_GATE = {"rmse_smear": "rmse", "mae_clip": "mae"}
OLD = ("old_mi", "old_mi_yj", "old_search")


def load(paths: List[str]) -> List[dict]:
    """Records of the given jsonl files; picks of the control experiment ``tt_b3_oldext`` (``oldext_<file>``) are merged into ``rec['old']`` when that file exists."""
    import os

    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        side = os.path.join(os.path.dirname(p), "oldext_" + os.path.basename(p))
        if os.path.exists(side):
            with open(side, encoding="utf-8") as fh:
                ext = {(d["kind"], d["n"], d["p"], d["seed"]): d for d in (json.loads(line) for line in fh if line.strip())}
            for r in recs:
                d = ext.get((r["kind"], r["n"], r["p"], r["seed"]))
                if d:
                    r["old"]["old_ext_mi"] = d["old_ext_mi"]
                    r["old"]["old_ext_search"] = d["old_ext_search"]
        out += recs
    return out


def pick_score(rec: dict, model: str, metric: str, method: str):
    """(pick name, score of the pick under the judge, abstained) for ``method`` in 'ref', 'ref2' (2-fold), 'old_*'."""
    j = rec["judge"][model]
    if method.startswith("old_"):
        name = rec["old"].get(method)
        if name is None:
            return None, np.nan, None
        return name, j.get(name, {}).get(metric, np.nan), name == "identity"
    folds = 2 if method == "ref2" else 1
    suffix = method[3:] if method.startswith("ref_") else ""
    key = f"{PAIR[model]}_f{folds}_{METRIC_GATE[metric]}{suffix}"
    if key not in rec["ref"]:
        return None, np.nan, None
    r = rec["ref"][key]
    jk = rec["ref_keys"][key]
    return r["name"], j.get(jk, {}).get(metric, np.nan), r["abstained"]


RESTRICTED = {"identity", "cbrt", "signed_sqrt", "signed_square", "asinh", "log", "log1p", "reciprocal", "logit", "yj_reg"}


def case_rows(recs: List[dict], model: str, metric: str, methods: List[str], oracle: str = "full") -> List[dict]:
    """``oracle='restricted'``: best over the reference's own offered set only (so a better Box-Cox / marginal-lambda Yeo-Johnson of tt_a4 does not count against the pick)."""
    rows = []
    for rec in recs:
        j = rec["judge"].get(model)
        if not j or "identity" not in j:
            continue
        ident = j["identity"][metric]
        pool = j if oracle == "full" else {k: v for k, v in j.items() if k in RESTRICTED or k.startswith("ref:")}
        best_name = min(pool, key=lambda k: pool[k][metric])
        best = j[best_name][metric]
        if model == "hgb":
            pass  # restricted oracle
        row = {"kind": rec["kind"], "n": rec["n"], "p": rec["p"], "seed": rec["seed"], "best": best_name, "gain": best / ident, "ident": ident}
        for m in methods:
            name, sc, ab = pick_score(rec, model, metric, m)
            if name is None or not np.isfinite(sc):
                continue
            row[m] = {"name": name, "ratio": sc / ident, "regret": sc / best, "abstain": bool(ab)}
        rows.append(row)
    return rows


def summarise(rows: List[dict], methods: List[str], label: str = "") -> List[str]:
    lines = []
    for m in methods:
        rs = [(r, r[m]) for r in rows if m in r]
        if not rs:
            continue
        gain = [(r, d) for r, d in rs if r["gain"] <= 0.98]
        nog = [(r, d) for r, d in rs if r["gain"] > 0.98]
        good = [d["regret"] <= 1.02 if r["gain"] <= 0.98 else d["ratio"] <= 1.01 for r, d in rs]
        fp = [d["name"] != "identity" and not d["abstain"] for r, d in nog]
        harm = [d["ratio"] > 1.02 for r, d in rs]
        miss = [d["name"] == "identity" for r, d in gain]
        ctrl = [(r, d) for r, d in rs if F.KIND.get(r["kind"]) == "control"]
        fpc = [d["name"] != "identity" for r, d in ctrl]
        regret = np.array([d["regret"] for r, d in rs])
        ratio = np.array([d["ratio"] for r, d in rs])
        lines.append(
            f"| {label} | {m} | {len(rs)} | {100 * np.mean(good):.0f}% | {regret.mean():.3f} | {np.median(regret):.3f} | {ratio.mean():.3f} | {ratio.max():.2f} | "
            f"{100 * np.mean(fp) if fp else float('nan'):.0f}% ({sum(fp)}/{len(fp)}) | {100 * np.mean(fpc) if fpc else float('nan'):.0f}% ({sum(fpc)}/{len(fpc)}) | {100 * np.mean(harm):.0f}% | "
            f"{100 * np.mean(miss) if miss else float('nan'):.0f}% ({sum(miss)}/{len(miss)}) | {100 * np.mean([d['abstain'] for r, d in rs]):.0f}% |"
        )
    return lines


HEADER = "| slice | method | cases | good | mean regret | median regret | mean ratio | worst ratio | FP on no-gain | FP on controls | harm >2% | miss on gain | abstain |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default="ridge")
    ap.add_argument("--metric", default="rmse_smear")
    ap.add_argument("--by", default="all")
    ap.add_argument("--methods", default="ref,ref2,old_mi,old_mi_yj,old_search")
    ap.add_argument("--filter", default="")
    ap.add_argument("--oracle", default="full", choices=["full", "restricted"])
    a = ap.parse_args()
    recs = load(a.files)
    if a.filter:
        recs = [r for r in recs if r["kind"] in a.filter.split(",")]
    methods = a.methods.split(",")
    rows = case_rows(recs, a.model, a.metric, methods, a.oracle)
    print(f"cases={len(rows)} model={a.model} metric={a.metric} oracle={a.oracle}")
    print(HEADER)
    if a.by == "all":
        print("\n".join(summarise(rows, methods, "all")))
        for grp in ("transform", "ambiguous", "control"):
            print("\n".join(summarise([r for r in rows if F.KIND[r["kind"]] == grp], methods, grp)))
    else:
        for k in sorted({r["kind"] for r in rows}, key=lambda s: list(F.KIND).index(s)):
            print("\n".join(summarise([r for r in rows if r["kind"] == k], methods, k)))
    ids = defaultdict(list)
    for r in rows:
        ids[r["kind"]].append(r["gain"])
    print("oracle best/identity per fixture (mean over seeds):", {k: round(float(np.mean(v)), 3) for k, v in ids.items()})


if __name__ == "__main__":
    main()
