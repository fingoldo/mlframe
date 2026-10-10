"""Aggregate the exp7 two-parameter runs (``p2_30000_*.json``): MI of the 1- and 2-parameter shift families and their Miller-Madow ratio pass rate against the 2-D joint MI.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.agg7``."""

import json

import numpy as np

from ..common._paths import data_dir

recs = []
for f in sorted(data_dir("stat_study").glob("p2_30000_*.json")):
    recs += json.loads(f.read_text())
print("seeds", len({r["seed"] for r in recs}))
cols = ["truth", "preset", "ols1", "grid1", "ols2", "ols2rob", "grid2"]
print("target".ljust(26) + "".join(c.rjust(14) for c in cols) + "   joint  MM-ratio(best of 1p / 2p ols2 / preset) pass>=.9")
for tn in dict.fromkeys(r["target"] for r in recs):
    rs = [r for r in recs if r["target"] == tn]
    line = tn.ljust(26)
    for c in cols:
        v = np.array([r[c] for r in rs])
        line += f"{v.mean():.3f}+-{v.std():.3f}".rjust(14)
    j = np.mean([r["joint"] for r in rs])

    def q(f):
        """Share of records whose MI under ``f`` reaches 0.9 x the Miller-Madow joint MI."""
        return np.mean([f(r) >= 0.9 * r["joint_mm"] for r in rs])

    print(
        line
        + f"  {j:.3f}  "
        + "/".join(f"{q(f):.2f}" for f in (lambda r: max(r["ols1_mm"], r["preset_mm"]), lambda r: max(r["ols2_mm"], r["preset_mm"]), lambda r: r["preset_mm"]))
    )
