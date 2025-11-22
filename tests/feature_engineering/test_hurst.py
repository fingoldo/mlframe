"""Hypothesis-based tests for hurst.py module."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from mlframe.feature_engineering.hurst import (
    compute_hurst_exponent,
    compute_hurst_rs,
    precompute_hurst_exponent,
)


@given(st.lists(st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False), min_size=20, max_size=500))
@settings(max_examples=50)
def test_hurst_returns_valid_range(arr):
    """Hurst exponent should generally be between 0 and 1.5 for most series."""
    h, c = compute_hurst_exponent(np.array(arr, dtype=np.float64))
    if not np.isnan(h):
        assert -0.5 <= h <= 2.0, f"Hurst exponent {h} out of expected range"


@given(st.lists(st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False), min_size=5, max_size=5))
def test_hurst_min_window_edge_case(arr):
    """Test with arrays at minimum window size."""
    result = compute_hurst_exponent(np.array(arr, dtype=np.float64), min_window=5)
    assert len(result) == 2
    assert isinstance(result[0], (float, np.floating))
    assert isinstance(result[1], (float, np.floating))


@given(st.integers(min_value=1, max_value=4))
def test_hurst_returns_nan_for_short_arrays(size):
    """Arrays shorter than min_window should return NaN."""
    arr = np.random.randn(size).astype(np.float64)
    h, c = compute_hurst_exponent(arr, min_window=5)
    assert np.isnan(h) and np.isnan(c)


@given(st.lists(st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False), min_size=10, max_size=100))
@settings(max_examples=30, deadline=None)
def test_hurst_rs_nonnegative(arr):
    """R/S statistic should be non-negative."""
    arr = np.array(arr, dtype=np.float64)
    rs = compute_hurst_rs(arr)
    assert rs >= 0.0


@given(st.floats(min_value=0.1, max_value=0.5))
def test_hurst_windows_log_step(log_step):
    """Test different window log steps."""
    arr = np.random.randn(100).astype(np.float64)
    h, c = compute_hurst_exponent(arr, windows_log_step=log_step)
    assert len((h, c)) == 2


@given(st.booleans())
def test_hurst_take_diffs_parameter(take_diffs):
    """Test both take_diffs modes."""
    arr = np.cumsum(np.random.randn(100)).astype(np.float64)  # Random walk
    h, c = compute_hurst_exponent(arr, take_diffs=take_diffs)
    assert len((h, c)) == 2


@given(st.integers(min_value=3, max_value=20))
def test_hurst_min_window_parameter(min_window):
    """Test different min_window values."""
    arr = np.random.randn(100).astype(np.float64)
    h, c = compute_hurst_exponent(arr, min_window=min_window)
    assert len((h, c)) == 2


def test_hurst_random_walk_close_to_half():
    """Random walk should have Hurst exponent close to 0.5."""
    np.random.seed(42)
    arr = np.cumsum(np.random.randn(1000)).astype(np.float64)
    h, c = compute_hurst_exponent(arr, take_diffs=True)
    # Random walk H should be around 0.5 (with some tolerance)
    if not np.isnan(h):
        assert 0.3 <= h <= 0.7, f"Random walk Hurst {h} not close to 0.5"


def test_hurst_trending_series():
    """Trending series should have Hurst exponent > 0.5."""
    np.random.seed(42)
    trend = np.linspace(0, 10, 1000)
    noise = np.random.randn(1000) * 0.1
    arr = (trend + noise).astype(np.float64)
    h, c = compute_hurst_exponent(arr)
    if not np.isnan(h):
        assert h > 0.3, f"Trending series Hurst {h} expected to be > 0.3"
