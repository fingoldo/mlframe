"""List the failure cases of the reference in B3 result files (agent B3, 2026-11-24).

A failure = ratio to identity above ``--harm`` (default 1.02: the pick is worse than doing nothing) or, on a gain case, regret above ``--regret`` (default 1.05: the pick leaves more than 5 percent on the table).
Prints per case the pick, its parameters, the LL gains of the top candidates, the null threshold and the judge ratios of the main candidates.

Usage: ``python tt_b3_failures.py FILE.jsonl [...] [--model ridge] [--metric rmse_smear] [--oracle restricted] [--method ref]``.
"""

from __future__ import annotations

import argparse


import tt_b3_analyze as A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default="ridge")
    ap.add_argument("--metric", default="rmse_smear")
    ap.add_argument("--oracle", default="restricted")
    ap.add_argument("--method", default="ref")
    ap.add_argument("--harm", type=float, default=1.02)
    ap.add_argument("--regret", type=float, default=1.05)
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    recs = A.load(a.files)
    rows = A.case_rows(recs, a.model, a.metric, [a.method], a.oracle)
    by = {(r["kind"], r["n"], r["p"], r["seed"]): r for r in recs}
    shown = 0
    for r in rows:
        d = r.get(a.method)
        if d is None:
            continue
        gain_case = r["gain"] <= 0.98
        if not (d["ratio"] > a.harm or (gain_case and d["regret"] > a.regret)):
            continue
        rec = by[(r["kind"], r["n"], r["p"], r["seed"])]
        key = f"{A.PAIR[a.model]}_f{2 if a.method == 'ref2' else 1}_{A.METRIC_GATE[a.metric]}"
        pk = rec["ref"][key]
        g = {k: round(v, 3) for k, v in pk["gains"].items() if v is not None}
        top = sorted(g, key=lambda k: -g[k])[:4]
        j = rec["judge"][a.model]
        idv = j["identity"][a.metric]
        jr = {k: round(j[k][a.metric] / idv, 3) for k in j if k in ("cbrt", "asinh", "log", "signed_sqrt", "yj_reg", "box_cox", "log1p", "reciprocal", "logit", "signed_square")}
        print(
            f"{r['kind']:13s} n{r['n']} p{r['p']} s{r['seed']:<3d} pick={d['name']:13s} {pk['params']} ratio={d['ratio']:.3f} regret={d['regret']:.3f} best={r['best']}({r['gain']:.3f}) "
            f"tau={pk['tau']:.3f} p={pk['p']:.3f} topLL={[(t, g[t]) for t in top]} judge={jr} reason={pk['reason'][:110]}"
        )
        shown += 1
        if shown >= a.limit:
            break
    print(f"shown {shown}")


if __name__ == "__main__":
    main()
