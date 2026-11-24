"""Gate-level table for the B3 report (agent B3, 2026-11-24): per fixture, how often the LL null test fires, how often the second gate vetoes, how often a non-identity transform is returned, and the time per call.

Columns: ``LLsig`` = share of cases where the best candidate beat the null threshold (p <= alpha); ``veto`` = share where a significant candidate failed the held-out original-scale gate (among LLsig);
``final`` = share returning a non-identity transform; ``abstain_reason`` counts; ``t_lin`` / ``t_spl`` = median seconds of one analysis (linear / spline head, 40 null draws, 1 fold) inside the run.

Usage: ``python tt_b3_calib.py FILE.jsonl [...] [--head linear] [--metric rmse]``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

import tt_b3_analyze as A
import tt_b3_fixtures as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--head", default="linear")
    ap.add_argument("--metric", default="rmse")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    recs = A.load(a.files)
    by = defaultdict(list)
    for r in recs:
        by[(r["kind"], r["n"], r["p"])].append(r)
    print("| fixture | n | p | cases | LLsig | veto | final non-identity | median p | t_lin s | t_spl s |\n|---|---|---|---|---|---|---|---|---|---|")
    for (kind, n, p), rs in sorted(by.items(), key=lambda kv: (list(F.KIND).index(kv[0][0]), kv[0][1], kv[0][2])):
        key = f"{a.head}_f1_{a.metric}"
        ds = [r["ref"][key] for r in rs if key in r["ref"]]
        if not ds:
            continue
        sig = [d["p"] <= a.alpha and d["gain"] is not None for d in ds]
        veto = [("failed the held-out" in d["reason"]) for d, s in zip(ds, sig) if s]
        fin = [not d["abstained"] for d in ds]
        tl = [r["time"].get(f"{a.head}_f1") for r in rs if r["time"].get(f"{a.head}_f1")]
        ts = [r["time"].get("spline_f1") for r in rs if r["time"].get("spline_f1")]
        print(
            f"| {kind} | {n} | {p} | {len(ds)} | {100 * np.mean(sig):.0f}% | {100 * np.mean(veto) if veto else float('nan'):.0f}% ({sum(veto)}/{len(veto)}) | {100 * np.mean(fin):.0f}% | "
            f"{np.median([d['p'] for d in ds]):.2f} | {np.median(tl) if tl else float('nan'):.2f} | {np.median(ts) if ts else float('nan'):.2f} |"
        )


if __name__ == "__main__":
    main()
