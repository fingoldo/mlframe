"""The closed-form ridge probe fit equals scikit-learn's ``Ridge`` and the gate's regression probe uses it."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_probe_ridge import fit_ridge_closed_form


@pytest.mark.parametrize("n,p", [(50, 3), (400, 6), (2000, 12)])
def test_matches_sklearn_ridge(n, p):
    """Weights and intercept equal ``Ridge(alpha=1.0)``'s to rounding."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(n)
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 3.0, size=p) + rng.normal(size=p)
    y = X @ rng.normal(size=p) + 0.4 * rng.normal(size=n) + 2.0
    w, b = fit_ridge_closed_form(X, y, 1.0)
    ref = Ridge(alpha=1.0).fit(X, y)
    np.testing.assert_allclose(w, ref.coef_, rtol=1e-9, atol=1e-10)
    assert b == pytest.approx(float(ref.intercept_), rel=1e-9, abs=1e-10)


def test_non_finite_input_returns_none():
    """A NaN column makes the system non-finite: the caller then keeps the scikit-learn fit."""
    X = np.ones((10, 2))
    X[3, 0] = np.nan
    assert fit_ridge_closed_form(X, np.arange(10.0), 1.0) is None


def test_gate_regression_probe_scores_like_the_sklearn_fit():
    """``_held_out_probe_score`` for a regression target equals the R^2 of the scikit-learn ridge on the same fold."""
    from sklearn.linear_model import Ridge

    from mlframe.feature_selection.filters._fe_accuracy_gate import _held_out_probe_score
    from mlframe.metrics.core import fast_r2_score

    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1] + 0.2 * rng.normal(size=300)).astype(np.float64)
    tr, va = np.arange(0, 200), np.arange(200, 300)
    got = _held_out_probe_score(X, tr=tr, va=va, y_enc=y, classification=False, binary=False)
    Xtr = X[tr].astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    m = Ridge(alpha=1.0).fit(((Xtr - mu) / sd).astype(np.float32), y[tr])
    want = fast_r2_score(y[va], m.predict(((X[va] - mu) / sd).astype(np.float32)))
    assert got == pytest.approx(float(want), abs=1e-5)
