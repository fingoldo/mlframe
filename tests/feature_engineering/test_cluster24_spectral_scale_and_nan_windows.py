"""Cluster 24 regressions: spectral windows are imputed per window, the spectral floors are scale-free, and the Hurst family reports NaN for
windows that held a NaN (audit 2026-11-20, findings 24-F32, 24-F33 and 24-F34).

Spectral NaN fill used the mean of the whole segment, so an early window took values from its future. The entropy and flatness floors were absolute
(1e-9, 1e-12 on squared amplitude), so a small-amplitude series changed its result. The rolling Hurst-family wrappers zero-filled NaN and then returned a
finite value for the window that held it.
"""

import numpy as np
import pytest

from mlframe.feature_engineering.hurst import rolling_hurst
from mlframe.feature_engineering.spectral import rolling_spectral_centroid, rolling_spectral_entropy, rolling_spectral_flatness


def test_spectral_window_does_not_see_future_values_through_nan_fill():
    """Changing values after index 5 leaves every window ending before index 6 unchanged, with a NaN inside an early window (24-F32)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    x[2] = np.nan
    gid = np.zeros(x.size, dtype=np.int64)
    base = rolling_spectral_centroid(x, gid, window_K=4)
    y = x.copy()
    y[6:] = 1e6 * rng.normal(size=y.size - 6)
    changed = rolling_spectral_centroid(y, gid, window_K=4)
    np.testing.assert_array_equal(base[:6], changed[:6])


@pytest.mark.parametrize("func", [rolling_spectral_entropy, rolling_spectral_flatness])
def test_spectral_floors_are_scale_free(func):
    """Scaling the series by a constant leaves entropy and flatness unchanged (24-F33)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    gid = np.zeros(x.size, dtype=np.int64)
    a = func(x, gid, window_K=32)
    b = func(1e-6 * x, gid, window_K=32)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_hurst_window_with_nan_is_nan_and_next_window_is_clean():
    """Windows that contain the NaN are NaN; the first window that excludes it equals the same series with that value zero-filled (24-F34)."""
    rng = np.random.default_rng(2)
    K = 20
    x = rng.normal(size=120)
    x[50] = np.nan
    out = rolling_hurst(x, window_K=K)
    assert np.isnan(out[50 : 50 + K]).all()
    filled = x.copy()
    filled[50] = 0.0
    reference = rolling_hurst(filled, window_K=K)
    assert np.isfinite(out[50 + K])
    np.testing.assert_allclose(out[50 + K :], reference[50 + K :], rtol=1e-9, equal_nan=True)
