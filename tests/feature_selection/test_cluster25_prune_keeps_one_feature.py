"""Audit 2026-11-20 finding 25-F23: the unanimous permutation prune never returns an empty selection.

When every feature looks harmful in every fold, the prune keeps the one whose permutation helped least, instead of dropping all.
The permutation step is patched so that every feature is unanimously harmful, which is the case the audit describes.
"""

import numpy as np
import pandas as pd

import importlib

prune_module = importlib.import_module("mlframe.feature_selection.unanimous_permutation_prune")


class _Result:
    """Stand-in for sklearn's permutation_importance result: every feature improves when permuted (negative delta)."""

    def __init__(self, n):
        """Store one negative mean importance per feature."""
        self.importances_mean = -np.arange(1, n + 1, dtype=float)


class _Model:
    """A fitted-nothing estimator, since the permutation step is patched."""

    def fit(self, X, y):
        """Return self; nothing to fit."""
        return self


def test_all_harmful_features_leave_one_not_none(monkeypatch):
    """Three features all harmful in both folds: one survives, never zero."""

    def fake_permutation_importance(model, X, y, **kwargs):
        """Report every feature as unanimously harmful."""
        return _Result(X.shape[1])

    import sklearn.inspection

    monkeypatch.setattr(sklearn.inspection, "permutation_importance", fake_permutation_importance)
    X = pd.DataFrame(np.random.default_rng(0).standard_normal((40, 3)), columns=["a", "b", "c"])
    y = np.arange(40) % 2
    cv_splits = [(np.arange(0, 20), np.arange(20, 40)), (np.arange(20, 40), np.arange(0, 20))]
    kept = prune_module.unanimous_permutation_prune(X, y, _Model, cv_splits, scoring=lambda a, b: 0.0, n_repeats=2, max_iterations=5)
    assert len(kept) >= 1
