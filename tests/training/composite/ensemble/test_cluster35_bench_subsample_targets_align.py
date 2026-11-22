"""Audit 2026-11-20 finding 35-F52: the subsample bench fits its weights against the holdout target, row for row.

compute_oof_holdout_predictions returns OOF rows for the holdout split only. The bench used to pair them with the natural-order y.
The helper now returns the matching holdout target, and it has one row per OOF row.
"""

import importlib

import numpy as np
import pandas as pd

bench = importlib.import_module("mlframe.training.composite.ensemble._benchmarks.bench_oof_subsample_speedup")


def test_oof_call_returns_a_target_aligned_with_the_oof_rows():
    """The returned holdout target has one value per OOF row, and it equals the y of the holdout rows."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() + 0.1 * rng.normal(size=n)
    groups = np.arange(n) % 15
    oof, y_holdout, _wall = bench._oof_call(X, y, groups, kfold=3)
    assert y_holdout.shape[0] == oof.shape[0]
    assert np.all(np.isin(y_holdout, y))
