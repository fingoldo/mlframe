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


def test_near_collinear_design_takes_lstsq():
    """A column that is a copy of another (plus 1e-9 noise) fails the conditioning check; the result is lstsq's, finite, and fits the target."""
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.standard_normal(n)
    A = np.column_stack([x, x + 1e-9 * rng.standard_normal(n), rng.standard_normal(n), np.ones(n)])
    b = 2 * x + A[:, 2]
    beta = R._ols_solve(A, b)
    assert np.isfinite(beta).all()
    assert np.allclose(np.linalg.lstsq(A, b, rcond=None)[0], beta, rtol=0, atol=1e-6)
    assert np.abs(A @ beta - b).max() < 1e-6
