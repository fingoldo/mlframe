"""Audit 2026-11-20 finding 34-F11: the auto block search scores candidate partitions with the same sample weights as the final fit.

Before the fix the search fitted its CV folds without ``sample_weight`` and pooled an unweighted RMSE, so the chosen partition could
optimise a different objective from the one the final block models minimise.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.direct_multi_horizon import DirectMultiHorizonEnsemble


def _data():
    """Four horizons driven by two features, with a row-wise weight that is very uneven."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 2))
    Y = np.column_stack([X[:, 0] * (h + 1) + X[:, 1] * 0.5 * h + rng.normal(scale=0.3, size=n) for h in range(4)])
    w = np.where(np.arange(n) < n // 2, 10.0, 0.1)
    return X, Y, w


def test_weighted_cv_score_differs_from_unweighted():
    """Passing sample_weight changes the candidate's CV score, so the search is no longer the unweighted objective."""
    X, Y, w = _data()
    ens = DirectMultiHorizonEnsemble(LinearRegression, auto_block_search=True, cv_splits=3, random_state=0)
    unweighted = ens._cv_score_block_size(X, Y, 2)
    weighted = ens._cv_score_block_size(X, Y, 2, sample_weight=w)
    assert np.isfinite(weighted)
    assert abs(weighted - unweighted) > 1e-6


def test_search_fits_fold_models_with_the_weights(monkeypatch):
    """During the auto search every fold fit receives the sample weights, not None."""
    X, Y, w = _data()
    ens = DirectMultiHorizonEnsemble(LinearRegression, auto_block_search=True, cv_splits=3, random_state=0)
    seen = []
    real = ens._fit_blocks

    def spy(Xs, Ys, blocks, sample_weight=None):
        """Record whether each fit received weights."""
        seen.append(sample_weight is not None)
        return real(Xs, Ys, blocks, sample_weight=sample_weight)

    monkeypatch.setattr(ens, "_fit_blocks", spy)
    ens._search_horizon_blocks(X, Y, sample_weight=w)
    assert seen and all(seen)
