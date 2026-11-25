"""``fit_ridge_closed_form`` solves the normal equations by Cholesky from ``CHOLESKY_MIN_COLUMNS`` columns up: the same weights as the general solver, and the same refusal of a non-finite system."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_probe_ridge as ridge


def _reference(X, y, alpha):
    """The pre-Cholesky body: LU solve of the same normal equations."""
    A = np.asarray(X, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64).ravel()
    x_mean, y_mean = A.mean(axis=0), float(t.mean())
    Ac = A - x_mean
    gram = Ac.T @ Ac
    gram[np.diag_indices_from(gram)] += alpha
    w = np.linalg.solve(gram, Ac.T @ (t - y_mean))
    return w, y_mean - float(x_mean @ w)


@pytest.mark.parametrize("k", [1, 8, 15, 16, 17, 64, 400])
@pytest.mark.parametrize("alpha", [1e-6, 1.0, 100.0])
def test_weights_equal_the_general_solver_below_at_and_above_the_threshold(k, alpha):
    """Weights and intercept agree with the LU solution to 1e-9 relative, whatever side of the threshold the width falls on."""
    rng = np.random.default_rng(k)
    X = rng.standard_normal((1500, k)) * rng.uniform(0.1, 10.0, k)
    y = X @ rng.standard_normal(k) + rng.standard_normal(1500)
    got = ridge.fit_ridge_closed_form(X, y, alpha)
    want = _reference(X, y, alpha)
    assert got is not None
    assert np.allclose(got[0], want[0], rtol=1e-9, atol=1e-9) and abs(got[1] - want[1]) <= 1e-9 * (1 + abs(want[1]))


def test_collinear_columns_are_solved_through_the_penalty():
    """A repeated column makes the Gram matrix singular; the penalty on the diagonal keeps it positive definite and both solvers return the same finite weights."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((800, 20))
    X = np.hstack([base, base[:, :3]])
    y = base[:, 0] + rng.standard_normal(800)
    got, want = ridge.fit_ridge_closed_form(X, y, 1.0), _reference(X, y, 1.0)
    assert got is not None and np.allclose(got[0], want[0], rtol=1e-8, atol=1e-8)


def test_non_finite_input_is_refused_on_both_sides_of_the_threshold():
    """A NaN in the design gives ``None`` (the caller keeps the scikit-learn fit), not NaN weights."""
    rng = np.random.default_rng(1)
    for k in (4, 40):
        X = rng.standard_normal((300, k))
        X[3, 1] = np.nan
        assert ridge.fit_ridge_closed_form(X, rng.standard_normal(300), 1.0) is None
