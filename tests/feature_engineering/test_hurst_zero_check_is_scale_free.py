"""The Hurst estimate of a series does not depend on its units.

The degenerate-window check compared the spread with an absolute 1e-12, so a series in tiny units (standard deviation about 1e-13) gave NaN
while the same series scaled up gave a value. A constant window is still NaN.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering import hurst as h


def test_rs_estimate_is_the_same_for_a_series_in_tiny_units():
    """A series multiplied by 1e-13 gives the same R/S statistic as the original."""
    x = np.random.default_rng(0).normal(size=500)
    assert np.isfinite(h.compute_hurst_rs(x * 1e-13))
    np.testing.assert_allclose(h.compute_hurst_rs(x * 1e-13), h.compute_hurst_rs(x), rtol=1e-9)


def test_single_window_estimate_is_the_same_for_a_series_in_tiny_units():
    """The rolling-kernel primitive is scale-free the same way."""
    x = np.random.default_rng(0).normal(size=500)
    np.testing.assert_allclose(h._hurst_rs_single(x * 1e-13), h._hurst_rs_single(x), rtol=1e-9)


def test_constant_window_is_still_nan():
    """A window with no spread has no Hurst estimate."""
    assert np.isnan(h.compute_hurst_rs(np.ones(100)))
    assert np.isnan(h._hurst_rs_single(np.ones(100)))
