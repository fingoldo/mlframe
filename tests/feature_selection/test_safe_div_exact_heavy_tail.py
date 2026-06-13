"""``_safe_div`` must be EXACT on every nonzero denominator -- heavy-tail regression pin.

The binary FE registry's ``div`` (and the hermite-FE sibling) guard against ``x/0`` blowup.
The guard must NOT perturb nonzero denominators: a heavy-tailed ratio target such as
``y = 0.2*a**2/b`` (with ``b`` uniform on (0,1), so ``b`` can be ~1e-6) concentrates almost
all of its magnitude on the smallest-``b`` points, and a relative error of even ``2e-9/b``
there inflates a linear downstream's test MAE by ~0.05 (the whole irreducible ``f/5`` floor).
The prior ``x / (y + sign(y)*eps + eps)`` form perturbed every positive denominator by
``2*eps``; the fix divides exactly wherever ``y != 0`` and substitutes ``eps`` only at exact
zero. These pins assert the exactness + the finite-at-zero behaviour for both implementations.
"""
from __future__ import annotations

import numpy as np
import pytest


def _impls():
    from mlframe.feature_selection.filters.feature_engineering import _safe_div as fe_div
    from mlframe.feature_selection.filters.hermite_fe import _safe_div as h_div
    return [("feature_engineering", fe_div), ("hermite_fe", h_div)]


@pytest.mark.parametrize("modname,div", _impls())
def test_safe_div_is_exact_on_nonzero_denominators(modname, div):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10_000)
    # span both signs and many magnitudes, incl. the heavy-tail near-zero denominators.
    y = np.concatenate([
        rng.uniform(1e-7, 1.0, 4000),     # small positive (the heavy-tail regime)
        -rng.uniform(1e-7, 1.0, 4000),    # small negative
        rng.standard_normal(2000) * 5.0,  # ordinary magnitudes (may include ~0, never exact 0)
    ])
    y = y[y != 0.0]
    x = x[: y.size]
    got = np.asarray(div(x, y), dtype=np.float64)
    exact = x / y
    # EXACT (to f64 round-off), not merely close: the heavy-tail point must not be perturbed.
    assert np.allclose(got, exact, rtol=1e-12, atol=0.0), (
        f"{modname}._safe_div perturbs nonzero denominators "
        f"(max rel err {np.max(np.abs(got - exact) / np.abs(exact)):.2e})"
    )


@pytest.mark.parametrize("modname,div", _impls())
def test_safe_div_is_finite_at_exact_zero(modname, div):
    x = np.array([1.0, -2.0, 0.0, 3.0])
    y = np.array([0.0, 0.0, 0.0, 0.0])
    got = np.asarray(div(x, y), dtype=np.float64)
    assert np.all(np.isfinite(got)), f"{modname}._safe_div produced non-finite output at y==0: {got}"


@pytest.mark.parametrize("modname,div", _impls())
def test_safe_div_heavy_tail_ratio_target_recovers_floor(modname, div):
    """End-to-end guard: a linear fit of ``0.2*a**2/b`` on the ``div``-built feature reaches the
    noise floor. With the old perturbing eps the smallest-``b`` point alone inflated MAE by ~0.05."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    rng = np.random.default_rng(1)
    n = 80_000
    a = rng.random(n)
    b = rng.random(n)
    noise = rng.random(n)
    y = 0.2 * a**2 / b + noise / 5.0
    feat = np.asarray(div(a**2, b), dtype=np.float64).reshape(-1, 1)
    tr, te = slice(0, int(0.9 * n)), slice(int(0.9 * n), n)
    mdl = LinearRegression().fit(feat[tr], y[tr])
    mae = mean_absolute_error(y[te], mdl.predict(feat[te]))
    # irreducible noise/5 floor is ~0.05; the exact division keeps us within a small margin of it.
    assert mae <= 0.06, f"{modname}._safe_div heavy-tail linear MAE {mae:.4f} exceeds the noise floor"
