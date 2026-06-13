"""cProfile harness for importance_agg='dispatched' RFECV fit.

Confirms the new aggregation is NOT a hot path (it runs once per outer iter on a small feature x fold table)
and that the dispatched fit wall is not a regression vs legacy. Run:
  python -m mlframe.feature_selection._benchmarks.profile_rfecv_importance_agg
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
os.environ.setdefault("MLFRAME_KEEP_BROKEN_CUPY", "1")

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlframe.feature_selection.wrappers.rfecv import RFECV


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(1500, 30)), columns=[f"c{i}" for i in range(30)])
    y = (X["c0"] * 1.5 + X["c1"] - X["c2"] + rng.normal(scale=0.5, size=1500) > 0).astype(int)
    return X, y


def _fit(agg):
    X, y = _data()
    sel = RFECV(estimator=RandomForestClassifier(n_estimators=40, random_state=0),
                cv=3, max_refits=12, importance_agg=agg,
                early_stopping_val_nsplits=None, random_state=0)
    sel.fit(X, y)
    return sel


def main():
    # Wall-time A/B (3 runs each, warm).
    for agg in ("legacy", "dispatched"):
        _fit(agg)  # warm
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            _fit(agg)
            ts.append(time.perf_counter() - t)
        print(f"{agg:<11} wall: {min(ts):.3f}s (min of 3)")

    # cProfile the dispatched fit; show top cumulative + the agg helper's share.
    pr = cProfile.Profile()
    pr.enable()
    _fit("dispatched")
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    out = s.getvalue()
    print(out)
    agg_lines = [ln for ln in out.splitlines() if "importance_agg" in ln or "aggregate_" in ln]
    print("=== aggregation-helper attribution (should be tiny) ===")
    print("\n".join(agg_lines) or "(no aggregate_* frames in top 25 -> not a hot path)")


if __name__ == "__main__":
    main()
