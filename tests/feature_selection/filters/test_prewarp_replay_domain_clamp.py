"""Regression: a fitted operand pre-warp must not extrapolate its polynomial past the training range at replay.

A degree-4 Chebyshev warp evaluated outside its fit-time [-1, 1] axis grows like z**4, so a single unseen operand value above the
training max turned ``prewarp(x)`` - and every engineered column built on it - into an extreme outlier for the downstream model.
The replay domain is now pinned to the fit-time axis range (constant beyond the data), with fit-time values unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import apply_operand_prewarp, fit_operand_prewarp, fit_pair_prewarp_als


def _data(seed=0, n=1500):
    """Return operands a, b and a noisy polynomial-interaction target y."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 12.0, n)
    b = rng.uniform(1.0, 5.0, n)
    y = (a**3 - 18 * a) * (b**2 - 3 * b) / 100 + rng.normal(0, 0.5, n)
    return a, b, y


@pytest.mark.parametrize("basis", ["chebyshev", "legendre", "hermite", "laguerre"])
def test_pair_prewarp_is_constant_beyond_fit_range(basis):
    """A fitted operand prewarp holds a bounded plateau beyond the fit range instead of extrapolating the polynomial."""
    a, b, y = _data()
    spec_a, spec_b = fit_pair_prewarp_als(a, b, y, basis=basis, max_degree=4)
    assert spec_a is not None and spec_b is not None
    fit_vals = apply_operand_prewarp(a, spec_a)
    # Hermite's z-score clip is symmetric at the larger fit-time |z|, so its plateau can start marginally past a.max(); probe well beyond.
    edge = apply_operand_prewarp(np.array([a.max() * 2.0]), spec_a)[0]
    far = apply_operand_prewarp(np.array([a.max() * 10.0]), spec_a)[0]
    assert np.all(np.isfinite(fit_vals))
    # Beyond the data the warp holds a boundary value instead of growing like a degree-4 polynomial.
    assert abs(far - edge) <= 1e-6 * max(1.0, abs(edge)), f"{basis}: prewarp extrapolates past the fit range ({edge} -> {far})"
    assert abs(far) <= 2.0 * float(np.max(np.abs(fit_vals))), f"{basis}: prewarp plateau {far} far outside the fit-time range"


@pytest.mark.parametrize("basis", ["chebyshev", "legendre", "hermite", "laguerre"])
def test_operand_prewarp_fit_time_values_unchanged_by_the_clamp(basis):
    """The clamp bounds contain every fit-time axis value, so the training column equals the unclamped evaluation."""
    a, _, y = _data(1)
    spec = fit_operand_prewarp(a, a**2 + y, basis=basis, max_degree=4)
    assert spec is not None and "clip" in spec["preprocess"], "the fitted spec carries no clamp, so the comparison below is trivial"
    unclamped = dict(spec, preprocess={k: v for k, v in spec["preprocess"].items() if k != "clip"})
    np.testing.assert_allclose(apply_operand_prewarp(a, spec), apply_operand_prewarp(a, unclamped), rtol=0, atol=1e-12)
