"""Recall of the planted pair when only the top-K near-miss pairs by headroom (joint minus preset-best MI) are kept; reads the exp11 runs.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.pf_headroom``."""

import json

import numpy as np

from ..common._paths import data_dir

recs = []
for f in sorted(data_dir("stat_study").glob("pf_*.json")):
    nn = int(f.name.split("pf_")[1].split("_")[0])
    for r in json.loads(f.read_text()):
        r["n"] = nn
        recs.append(r)
for nn in (5000, 30000):
    for ds in dict.fromkeys(r["ds"] for r in recs):
        rs = [r for r in recs if r["ds"] == ds and r["n"] == nn]
        if not rs or not any(r["planted"] for r in rs):
            continue
        out = {}
        for K in (2, 3):
            hit = []
            for s in range(len(rs) // 10):
                blk = rs[s * 10 : (s + 1) * 10]
                ok = [i for i, r in enumerate(blk) if 0.3 <= r["preset"] / max(r["joint"], 1e-9) < 0.9 and r["joint"] > 0.02]
                top = sorted(ok, key=lambda i: -(blk[i]["joint"] - blk[i]["preset"]))[:K]
                hit.append(any(blk[i]["planted"] for i in top))
            out[K] = np.mean(hit)
        print(nn, ds[:40].ljust(42), "recall of planted pair, near-miss window then top-K by headroom (joint - preset_best):", out)
