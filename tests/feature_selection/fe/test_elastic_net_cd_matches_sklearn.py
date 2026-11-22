"""The coordinate-descent elastic net / lasso solves scikit-learn's problem: the same coefficients (scikit-learn run to a tight tolerance) and the same support."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._elastic_net_cd import elastic_net_coef


def _data(n, p, seed):
    """Correlated columns, a sparse truth, a constant column and unequal scales."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, p))
    X = base + 0.4 * base[:, [0]]
    X[:, -1] = 2.0
    X *= rng.uniform(0.5, 3.0, size=p)
    truth = np.zeros(p)
    truth[:3] = [1.5, -1.0, 0.6]
    return X, X @ truth + 0.7 * rng.standard_normal(n)


@pytest.mark.parametrize("alpha,l1", [(0.01, 1.0), (0.1, 1.0), (0.05, 0.5), (0.2, 0.2)])
def test_matches_sklearn(alpha, l1):
    """Coefficients equal sklearn's (run to 1e-12 tolerance) to 1e-6."""
    from sklearn.linear_model import ElasticNet, Lasso

    X, y = _data(600, 12, int(alpha * 100) + int(l1 * 10))
    ref = (Lasso(alpha=alpha, tol=1e-12, max_iter=200000) if l1 == 1.0 else ElasticNet(alpha=alpha, l1_ratio=l1, tol=1e-12, max_iter=200000)).fit(X, y)
    got, converged = elastic_net_coef(X, y, alpha, l1)
    assert converged
    np.testing.assert_allclose(got, ref.coef_, rtol=0, atol=1e-6)
    assert list(np.flatnonzero(got)) == list(np.flatnonzero(ref.coef_))


def test_a_large_penalty_zeroes_everything_and_a_non_converging_budget_is_reported():
    """alpha above the largest correlation gives all zeros; a budget of one sweep on a hard problem reports non-convergence."""
    X, y = _data(300, 8, 1)
    zeros, converged = elastic_net_coef(X, y, 1e3, 1.0)
    assert converged and not zeros.any()
    _, converged = elastic_net_coef(X, y, 1e-4, 1.0, max_iter=1)
    assert not converged


def test_gram_form_gives_the_residual_form_coefficients(monkeypatch):
    """The Gram-matrix sweeps (used when there are more rows than columns) and the residual sweeps reach the same optimum, lasso and elastic net, with a zero-variance column in the design."""
    from mlframe.feature_selection.filters import _elastic_net_cd as E

    rng = np.random.default_rng(11)
    X = rng.standard_normal((3000, 25)) * rng.uniform(0.3, 4.0, size=25)
    X[:, 7] = 1.5
    y = X[:, 0] - 0.5 * X[:, 3] + 0.3 * rng.standard_normal(3000)
    for l1_ratio in (1.0, 0.4):
        gram, ok_g = E.elastic_net_coef(X, y, 0.05, l1_ratio)
        monkeypatch.setattr(E, "_GRAM_MAX_COLUMNS", 0)  # force the residual form
        resid, ok_r = E.elastic_net_coef(X, y, 0.05, l1_ratio)
        monkeypatch.undo()
        assert ok_g and ok_r
        np.testing.assert_allclose(gram, resid, rtol=1e-6, atol=1e-8)
