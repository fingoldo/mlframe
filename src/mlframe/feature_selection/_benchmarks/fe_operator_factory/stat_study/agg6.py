"""Aggregate the exp6 tables (``t_<preset>_<seed>.json``): mean and sd of the MI of every shift estimator per target and the share of seeds reaching 0.9 x the 2-D joint MI.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.agg6 <minimal|medium>``."""

import json
import sys

import numpy as np

from ..common._paths import data_dir

preset = sys.argv[1]
recs = []
for f in sorted(data_dir("stat_study").glob(f"t_{preset}_*.json")):
    recs += json.loads(f.read_text())
seeds = sorted({r["seed"] for r in recs})
print(preset, "seeds", len(seeds))
cols = ["truth", "preset_best", "median", "zc", "zc_sig_fb", "ols", "ols_rob", "grid", "both_centred", "als"]
print("target".ljust(24) + "".join(c[:9].rjust(14) for c in cols) + "   joint  pass.9(preset/ols/grid)")
for tn in dict.fromkeys(r["target"] for r in recs):
    rs = [r for r in recs if r["target"] == tn]
    line = tn.ljust(24)
    for c in cols:
        v = np.array([r[c] for r in rs])
        line += f"{v.mean():.3f}+-{v.std():.3f}".rjust(14)
    j = np.array([r["joint"] for r in rs])
    ps = [np.mean([r[c] >= 0.9 * r["joint"] for r in rs]) for c in ("preset_best", "ols", "grid")]
    print(line + f"  {j.mean():.3f}  " + "/".join(f"{p:.2f}" for p in ps))
