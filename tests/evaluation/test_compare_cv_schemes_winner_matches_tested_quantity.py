"""The winning scheme is the one with the smallest per-fold distance to the out-of-order score, the quantity the significance test compares."""

from __future__ import annotations

import numpy as np

from mlframe.evaluation import compare_cv_schemes


class _Const:
    """Model predicting a fixed value so the metric below returns a chosen number per fold."""

    def fit(self, X, y):
        """No-op fit."""
        return self

    def predict(self, X):
        """Return the fold's wanted score through the target (see metric)."""
        return np.zeros(len(X))


def test_scheme_with_cancelling_folds_does_not_win_on_a_zero_mean_gap():
    """Scheme A's folds sit at ooo +/- 0.2 (mean gap 0, fold gap 0.2); scheme B's at ooo + 0.05 (fold gap 0.05): B wins."""
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.zeros(40)
    targets = iter([1.0, 0.8, 1.2, 1.05, 1.05])

    def metric(_y, _p):
        """Return the next scripted score."""
        return next(targets)

    idx = np.arange(20)
    ooo = (idx, np.arange(20, 40))
    res = compare_cv_schemes(X, y, {"A": [(idx, idx + 20), (idx, idx + 20)], "B": [(idx, idx + 20), (idx, idx + 20)]}, ooo, _Const, metric)
    assert res["best_scheme"] == "B"
