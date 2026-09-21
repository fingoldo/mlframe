"""Basis preprocessors scale a column by its own spread, not by an absolute 1e-12 pad.

``std + 1e-12`` and ``(hi - lo) + 1e-12`` are negligible only while the column's magnitude sits far above 1e-12. A column whose spread is
genuinely ~1e-13 (a normalised residual, a difference of two nearly-equal engineered columns, a small-unit measurement) was divided by the pad
rather than by its own spread, so the Hermite/Legendre axis collapsed, the basis read as uninformative, and the column was dropped. Scaling
that same column up by 1e13 is a pure change of units, so every preprocessor must return the same z-axis for both.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import (
    _apply_minmax,
    _apply_zscore,
    _preprocess_minmax_neg1_1,
    _preprocess_zscore,
)

_TINY = 1e-13


def _column(seed: int = 0, n: int = 512) -> np.ndarray:
    """A well-behaved Gaussian column at unit scale; multiply by ``_TINY`` for the regime the pad corrupts."""
    return np.random.default_rng(seed).normal(size=n)


def test_zscore_is_unit_scale_on_a_tiny_spread_column():
    """A ~1e-13 spread standardises to unit variance; the padded form shrank it by roughly an order of magnitude."""
    z, params = _preprocess_zscore(_column() * _TINY)
    assert abs(float(z.std()) - 1.0) < 1e-6, f"z-axis collapsed: std={float(z.std())}"
    assert params["std"] > 0.0


def test_zscore_is_invariant_to_a_pure_change_of_units():
    """Scaling the column by 1e13 is a change of units, so the z-axis must be the same to floating-point noise."""
    x = _column(1)
    z_unit, _ = _preprocess_zscore(x)
    z_tiny, _ = _preprocess_zscore(x * _TINY)
    assert np.allclose(z_tiny, z_unit, rtol=1e-9, atol=1e-9), f"max |diff| = {float(np.abs(z_tiny - z_unit).max())}"


def test_minmax_spans_the_full_interval_on_a_tiny_spread_column():
    """A ~1e-13 range maps onto the full [-1, 1] basis domain rather than a sliver of it."""
    z, _ = _preprocess_minmax_neg1_1(_column(2) * _TINY)
    assert abs(float(z.min()) + 1.0) < 1e-9 and abs(float(z.max()) - 1.0) < 1e-9, f"range [{float(z.min())}, {float(z.max())}]"


def test_minmax_is_invariant_to_a_pure_change_of_units():
    """The min-max axis, too, must not depend on the column's unit."""
    x = _column(3)
    z_unit, _ = _preprocess_minmax_neg1_1(x)
    z_tiny, _ = _preprocess_minmax_neg1_1(x * _TINY)
    assert np.allclose(z_tiny, z_unit, rtol=1e-9, atol=1e-9), f"max |diff| = {float(np.abs(z_tiny - z_unit).max())}"


@pytest.mark.parametrize("scale", [1.0, _TINY], ids=["unit", "tiny"])
def test_replay_reproduces_the_fitted_axis(scale):
    """Replaying the stored params on the fit data must return the fitted axis at either scale: fit and transform stay in lockstep."""
    x = _column(4) * scale
    z_fit, zp = _preprocess_zscore(x)
    assert np.allclose(_apply_zscore(x, zp), z_fit, rtol=1e-12, atol=0.0)
    m_fit, mp = _preprocess_minmax_neg1_1(x)
    assert np.allclose(_apply_minmax(x, mp), m_fit, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("value", [0.0, 3.5, -2e6], ids=["zero", "small", "large"])
def test_a_constant_column_still_maps_to_a_degenerate_axis(value):
    """No spread means no axis: the z-score is all zeros and the min-max axis pins to the domain edge, as before."""
    x = np.full(256, value)
    z, _ = _preprocess_zscore(x)
    assert np.array_equal(z, np.zeros_like(x)), f"constant column produced a non-zero z-axis: {z[:4]}"
    m, _ = _preprocess_minmax_neg1_1(x)
    assert np.array_equal(m, np.full_like(x, -1.0)), f"constant column produced {m[:4]}"
