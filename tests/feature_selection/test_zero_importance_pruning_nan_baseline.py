"""Regression: a NaN baseline CV score must not freeze the full set as the winner."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mlframe.feature_selection.zero_importance_pruning import iterative_zero_importance_pruning


def test_nan_baseline_score_does_not_block_later_rounds():
    """When the first scoring call returns NaN, a later finite round can still become the best set."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
    y = X["a"].to_numpy()
    calls = {"n": 0}

    def scoring(t, p):
        """NaN for the first fold only, finite afterwards."""
        calls["n"] += 1
        return float("nan") if calls["n"] <= 5 else -float(np.mean((t - p) ** 2))

    out = iterative_zero_importance_pruning(RandomForestRegressor(n_estimators=3, random_state=0), X, y, scoring,
                                            importance_fn=lambda m, Xr, yy: np.array([1.0, 0.0, 0.0][: Xr.shape[1]] + [1.0] * max(0, Xr.shape[1] - 3)))
    assert out == ["a"]
