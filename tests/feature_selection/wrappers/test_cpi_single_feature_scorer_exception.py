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
    """With a single feature (p==1, the no-conditioning-set fallback path), a model.score() crash
    must not propagate -- it must degrade to a 0.0 importance (all-NaN score_losses), matching the
    general (p>1) branch's documented failure mode."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.random((n, 1))
    y = rng.random(n)
    model = _FlakyScoreModel()
    model.fit(X, y)

    importances = _conditional_permutation_importance(model, X, y, n_repeats=3)

    assert importances.shape == (1,)
    assert np.isfinite(importances).all()
    assert importances[0] == 0.0
