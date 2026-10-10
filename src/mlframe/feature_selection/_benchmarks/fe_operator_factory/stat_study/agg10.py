"""Aggregate the downstream runs of ``exp10_downstream``: relative MAE / RMSE improvement of each feature set over the raw-columns baseline, per model and target.

Positive = lower error than the baseline. Records written before the metric change (key ``r2`` instead of ``mae`` / ``rmse``) are legacy, are skipped and are counted in the header.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.agg10``; reads the committed ``results/``, or the scratch outputs of ``exp10_downstream`` with ``--fresh``.
"""

import json

import numpy as np

from ..common._paths import data_dir
from ..common.downstream import METRICS, REFERENCE_METRICS, rel_improvement

SETS = ["+preset", "+p1", "+p2", "+p1+p2"]


def main() -> None:
    """Print one table per model, error metric and sample size."""
    recs = []
    for f in sorted(data_dir("stat_study").glob("ds_*.json")):
        recs += json.loads(f.read_text())
    legacy = [r for r in recs if "mae" not in r]
    recs = [r for r in recs if "mae" in r]
    print(f"{len(recs)} records with MAE/RMSE, {len(legacy)} legacy R^2-only records skipped (not a decision metric)")
    for model in ("ridge", "hgb"):
        for metric in METRICS:
            for n in sorted({r["n"] for r in recs}):
                print(f"\n## {model}  n={n}   relative {metric.upper()} improvement over raw columns (mean over seeds; positive = better)")
                print("target".ljust(28) + "".join(s.rjust(10) for s in SETS))
                tot = {s: [] for s in SETS}
                for tn in dict.fromkeys(r["target"] for r in recs):
                    rs = [r for r in recs if r["target"] == tn and r["n"] == n]
                    if not rs:
                        continue
                    v = {s: np.mean([rel_improvement(r[metric][f"{model}|{s}"], r[metric][f"{model}|raw"]) for r in rs]) for s in SETS}
                    for s in SETS:
                        tot[s].append(v[s])
                    print(tn.ljust(28) + "".join(f"{v[s]:10.3f}" for s in SETS) + f"   (seeds={len(rs)})")
                if tot[SETS[0]]:
                    print("MEAN".ljust(28) + "".join(f"{np.mean(tot[s]):10.3f}" for s in SETS))
        for ref in REFERENCE_METRICS:
            for n in sorted({r["n"] for r in recs if ref in r}):
                print(f"\n## {model}  n={n}   {ref.upper()} difference to raw columns (REFERENCE ONLY, not a decision metric; positive = higher)")
                print("target".ljust(28) + "".join(s.rjust(10) for s in SETS))
                for tn in dict.fromkeys(r["target"] for r in recs):
                    rs = [r for r in recs if r["target"] == tn and r["n"] == n and ref in r]
                    if rs:
                        v = {s: np.mean([r[ref][f"{model}|{s}"] - r[ref][f"{model}|raw"] for r in rs]) for s in SETS}
                        print(tn.ljust(28) + "".join(f"{v[s]:10.3f}" for s in SETS) + f"   (seeds={len(rs)})")


if __name__ == "__main__":
    main()
