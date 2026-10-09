"""The hinge held-out scorer's OLS solve equals lstsq and keeps lstsq's answer on singular designs."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _hinge_basis_fe as hb


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e-3])
def test_matches_lstsq_on_well_conditioned_designs(scale):
    """[1, x, relu(x - tau)] at several scales."""
    rng = np.random.default_rng(1)
    n = 50_000
    x = rng.normal(size=n) * scale
    A = np.column_stack([np.ones(n), x, np.maximum(x - 0.3 * scale, 0.0)])
    y = 2.0 * x + 3.0 * A[:, 2] + rng.normal(size=n)
    np.testing.assert_allclose(hb._ols_coef(A, y), np.linalg.lstsq(A, y, rcond=None)[0], rtol=1e-6, atol=1e-8)


def test_a_collinear_design_returns_the_minimum_norm_solution():
    """A leg equal to x makes A'A singular: the answer must still be lstsq's minimum-norm fit."""
    rng = np.random.default_rng(2)
    n = 5_000
    x = rng.normal(size=n)
    A = np.column_stack([np.ones(n), x, x])
    y = x + rng.normal(size=n)
    np.testing.assert_allclose(hb._ols_coef(A, y), np.linalg.lstsq(A, y, rcond=None)[0], rtol=1e-8, atol=1e-10)


def test_a_constant_leg_falls_back():
    """An all-zero leg (a hinge beyond the data) gives a zero coefficient like lstsq, not NaN."""
    rng = np.random.default_rng(3)
    n = 2_000
    x = rng.normal(size=n)
    A = np.column_stack([np.ones(n), x, np.zeros(n)])
    y = x + rng.normal(size=n)
    coef = hb._ols_coef(A, y)
    assert np.all(np.isfinite(coef)) and abs(coef[2]) < 1e-9


def test_the_incremental_gain_is_unchanged():
    """End to end: the held-out R^2 gain of a real hinge leg equals the one computed with lstsq coefficients."""
    rng = np.random.default_rng(4)
    n = 30_000
    x = rng.normal(size=n)
    y = np.maximum(x - 0.2, 0.0) * 2 + 0.3 * x + 0.2 * rng.normal(size=n)
    leg = np.maximum(x - 0.2, 0.0)
    gain = hb._heldout_incremental_r2(x, leg, y)
    va = (np.arange(n) % 3) == 0
    tr = ~va

    def r2(cols):
        """Held-out R^2 of an lstsq fit on the design made of ``cols``."""
        A = np.column_stack(cols)
        c = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        return 1.0 - float(np.sum((y[va] - A[va] @ c) ** 2)) / float(np.sum((y[va] - y[va].mean()) ** 2))

    one = np.ones(n)
    assert gain == pytest.approx(r2([one, x, leg]) - r2([one, x]), abs=1e-9)
