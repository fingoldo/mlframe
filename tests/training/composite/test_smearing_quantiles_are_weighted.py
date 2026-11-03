"""Smearing residual quantiles follow the fit's sample weights."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.estimator._smearing import N_SMEAR_QUANTILES, residual_quantiles


class _ZeroModel:
    """Predicts zero, so the residuals equal the target."""

    def predict(self, X):
        """Return zeros for every row."""
        return np.zeros(len(X))


def test_weighted_residual_quantiles_shift_towards_heavy_rows():
    """Rows with large residuals carrying most of the weight pull the quantiles up versus the unweighted ones."""
    n = 4000
    t = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]) + np.random.default_rng(0).normal(0, 0.01, n)
    X = np.zeros((n, 1))
    w = np.concatenate([np.full(n // 2, 0.01), np.full(n // 2, 1.0)])
    plain = residual_quantiles(_ZeroModel(), X, t)
    weighted = residual_quantiles(_ZeroModel(), X, t, sample_weight=w)
    assert plain is not None and weighted is not None and len(weighted) == N_SMEAR_QUANTILES
    assert float(np.mean(weighted)) > float(np.mean(plain)) + 0.3
    assert float(np.mean(weighted > 0.5)) > 0.9


def test_unit_weights_match_unweighted_quantiles():
    """Equal weights reproduce the unweighted quantiles closely."""
    n = 4000
    t = np.random.default_rng(1).normal(0, 1, n)
    X = np.zeros((n, 1))
    a = residual_quantiles(_ZeroModel(), X, t)
    b = residual_quantiles(_ZeroModel(), X, t, sample_weight=np.ones(n))
    assert a is not None and b is not None
    np.testing.assert_allclose(a, b, atol=0.02)
