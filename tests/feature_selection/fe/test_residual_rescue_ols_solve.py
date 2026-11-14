"""``_ols_solve`` equals ``np.linalg.lstsq`` on well-conditioned designs and hands near-collinear ones to ``lstsq``."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _residual_rescue_fe as R


@pytest.mark.parametrize("p", [1, 8, 40])
def test_well_conditioned_design_equals_lstsq(p):
    """Standardised random columns plus an intercept: one and several right-hand sides agree to 1e-10."""
    rng = np.random.default_rng(p)
    n = 6000
    A = np.column_stack([rng.standard_normal((n, p)), np.ones(n)])
    B = np.column_stack([A[:, 0] + rng.standard_normal(n), rng.standard_normal(n)])
    for rhs in (B[:, 0], B):
        want = np.linalg.lstsq(A, rhs, rcond=None)[0]
        assert np.allclose(R._ols_solve(A, rhs), want, rtol=0, atol=1e-10)


def test_float32_columns_are_solved_in_double():
    """A float32 design (the criterion dtype of some runs) still returns the double-precision solution."""
    rng = np.random.default_rng(0)
    A = np.column_stack([rng.standard_normal((4000, 5)), np.ones(4000)]).astype(np.float32)
    b = (A[:, 0] + 0.1 * rng.standard_normal(4000)).astype(np.float32)
    want = np.linalg.lstsq(A.astype(np.float64), b.astype(np.float64), rcond=None)[0]
    assert np.allclose(R._ols_solve(A, b), want, rtol=0, atol=1e-8)


@pytest.mark.parametrize("kind", ["duplicate", "constant"])
def test_rank_deficient_design_gives_the_lstsq_fitted_values(kind):
    """Exactly rank-deficient designs (a repeated column, a second constant): the coefficients may differ (minimum norm), the fitted values equal lstsq's."""
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    second = x.copy() if kind == "duplicate" else np.ones(n)
    A = np.column_stack([x, second, z, np.ones(n)])
    b = 2 * x + z + 0.1 * rng.standard_normal(n)
    beta = R._ols_solve(A, b)
    assert np.isfinite(beta).all()
    want = A @ np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.abs(A @ beta - want).max() < 1e-6


def test_near_duplicate_column_is_fitted_as_the_identifiable_model():
    """A column that is a copy of another plus 1e-9 noise lies below the resolution of A'A: the fit is that of the design without the copy (the identifiable model), and stays finite and
    stable, where an SVD solve fits the noise direction with coefficients near 1e9."""
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    A = np.column_stack([x, x + 1e-9 * rng.standard_normal(n), z, np.ones(n)])
    b = 2 * x + z + 0.1 * rng.standard_normal(n)
    beta = R._ols_solve(A, b)
    assert np.isfinite(beta).all() and np.abs(beta).max() < 1e3
    identifiable = np.column_stack([x, z, np.ones(n)])
    want = identifiable @ np.linalg.lstsq(identifiable, b, rcond=None)[0]
    assert np.abs(A @ beta - want).max() < 1e-6


def test_degenerate_design_does_not_reach_lstsq(monkeypatch):
    """A rank-deficient design is solved from the eigen-decomposition of A'A: the SVD of the whole matrix is not called."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2000, 5))
    A = np.column_stack([x, x[:, 0] + x[:, 1], np.ones(2000)])
    called = []
    monkeypatch.setattr(np.linalg, "lstsq", lambda *a, **k: called.append(1))
    R._ols_solve(A, x[:, 2] + rng.standard_normal(2000))
    assert not called


def test_matrix_target_keeps_one_column_per_target_on_a_degenerate_design():
    """Several right-hand sides on a rank-deficient design: one coefficient column per target, each with lstsq's fitted values."""
    rng = np.random.default_rng(2)
    n = 2000
    x = rng.standard_normal(n)
    A = np.column_stack([x, x, rng.standard_normal(n), np.ones(n)])
    B = np.column_stack([A[:, 0] + rng.standard_normal(n), rng.standard_normal(n)])
    beta = R._ols_solve(A, B)
    assert beta.shape == (4, 2)
    assert np.abs(A @ beta - A @ np.linalg.lstsq(A, B, rcond=None)[0]).max() < 1e-6
