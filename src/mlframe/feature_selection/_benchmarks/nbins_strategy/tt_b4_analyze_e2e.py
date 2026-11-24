"""Aggregate tt_b4_run_e2e.py JSONL files: mean held-out R2 (and RMSE) per feature set and model over the seeds of each (p, n) setting.

Usage: python tt_b4_analyze_e2e.py FILE1.jsonl [FILE2.jsonl ...]
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

ORDER = ["none", "blind", "onepass", "onepass_sel", "stepwise", "stepwise_rep", "oracle"]
MODELS = ["ridge", "gam", "hgb"]


def main(paths):
    for path in paths:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if not rows:
            continue
        p, n = rows[0]["p"], rows[0]["n"]
        print(f"\n### p={p} n_train={n} seeds={[r['seed'] for r in rows]}  (mean test R2 / RMSE; columns = mean #columns fed to the model)")
        print("feature set | cols | ridge R2 | ridge RMSE | gam R2 | gam RMSE | hgb R2 | hgb RMSE")
        print("--- | --- | --- | --- | --- | --- | --- | ---")
        for v in ORDER:
            cols = np.mean([r["ncols"][v] for r in rows])
            cells = []
            for m in MODELS:
                vals = [r["res"][v][m] for r in rows if v in r["res"] and m in r["res"][v]]
                if len(vals) == len(rows):
                    cells += [f"{np.mean([x['r2'] for x in vals]):.4f}", f"{np.mean([x['rmse'] for x in vals]):.4f}"]
                else:
                    cells += ["not run", "not run"]
            print(f"{v} | {cols:.0f} | " + " | ".join(cells))
        # per-seed ridge R2 differences vs none, recommended and oracle
        d = collections.defaultdict(list)
        for r in rows:
            for v in ("blind", "onepass", "onepass_sel", "stepwise", "oracle"):
                d[v].append(r["res"][v]["ridge"]["r2"] - r["res"]["none"]["ridge"]["r2"])
        print("ridge R2 gain over none per seed: " + "; ".join(f"{v}: " + ",".join(f"{x:+.3f}" for x in xs) for v, xs in d.items()))
        sec = collections.defaultdict(list)
        for r in rows:
            for k, v in r["seconds"].items():
                sec[k].append(v)
        print("seconds per dataset (contended machine): " + ", ".join(f"{k} {np.mean(v):.1f}" for k, v in sec.items()))
        print("recommended picks (stepwise):", [r["chosen"]["stepwise"] for r in rows])
        print("truth links:", [r["inf"] for r in rows])


if __name__ == "__main__":
    main(sys.argv[1:])
