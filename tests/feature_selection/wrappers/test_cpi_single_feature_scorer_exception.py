"""FS_WRAPPERS-6: _conditional_permutation_importance's single-feature (p==1, no-conditioning-set)
fallback branch called model.score() with no try/except, unlike the general (p>1) branch a few lines
below which explicitly wraps the identical call so a scorer crash degrades to NaN instead of aborting
the whole per-fold FI computation."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.wrappers._helpers_importance import _conditional_permutation_importance


class _FlakyScoreModel:
    """A model whose score() succeeds once (the baseline call) then raises on every subsequent call
    (the permuted-X calls inside the importance loop) -- simulates a scorer that crashes specifically
    on out-of-distribution permuted input."""

    def __init__(self):
        self._n_calls = 0

    def fit(self, X, y):
        """No-op fit; returns self."""
        return self

    def score(self, X, y):
        """First call (the baseline) returns a fixed value; every later call raises."""
        self._n_calls += 1
        if self._n_calls == 1:
            return 0.5
        raise RuntimeError("synthetic scorer crash on permuted input")


def test_single_feature_branch_survives_scorer_crash():
    """With a single feature (p==1, the no-conditioning-set fallback path), a model.score() crash must not propagate:
    the importance is recorded as NaN (unmeasured), matching the general (p>1) branch. 0.0 used to stand in for it, and
    since real importances are baseline - score and routinely negative, an unmeasured feature outranked harmful ones."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.random((n, 1))
    y = rng.random(n)
    model = _FlakyScoreModel()
    model.fit(X, y)

    importances = _conditional_permutation_importance(model, X, y, n_repeats=3)

    assert importances.shape == (1,)
    assert np.isnan(importances[0])


def test_general_branch_records_nan_when_every_repeat_fails():
    """The p>1 branch follows the same rule: every permuted score() raised, so the importance is NaN, not 0.0."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.random((n, 3))
    y = X[:, 0] + rng.random(n) * 0.1
    model = _FlakyScoreModel()
    model.fit(X, y)

    importances = _conditional_permutation_importance(model, X, y, n_repeats=3)

    assert importances.shape == (3,)
    assert np.isnan(importances).all(), f"every repeat failed on every feature, expected all NaN, got {importances}"
