"""Hypothesis-based tests for timeseries.py module."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume

from mlframe.feature_engineering.timeseries import (
    find_next_cumsum_left_index,
    find_next_cumsum_right_index,
    get_nwindows_expected,
    get_ts_window_name,
)


@given(
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=50)
def test_find_next_cumsum_left_index(amount, right_index):
    """Test cumsum index finder returns valid indices."""
    arr = np.random.rand(right_index + 10).astype(np.float64) * 10
    left, total = find_next_cumsum_left_index(arr, amount, right_index)
    assert 0 <= left <= right_index
    assert total >= 0


@given(
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0, max_value=50)
)
@settings(max_examples=50)
def test_find_next_cumsum_right_index(amount, left_index):
    """Test cumsum right index finder returns valid indices."""
    arr = np.random.rand(left_index + 50).astype(np.float64) * 10
    right, total = find_next_cumsum_right_index(arr, amount, left_index)
    assert left_index <= right < len(arr)
    assert total >= 0


@given(st.integers(min_value=1, max_value=10))
def test_get_nwindows_expected(n_windows):
    """Test window count calculation."""
    windows = {'': list(range(n_windows))}
    result = get_nwindows_expected(windows)
    assert result == n_windows


@given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=5))
def test_get_nwindows_expected_multiple_vars(n_windows1, n_windows2):
    """Test window count with multiple variables."""
    windows = {
        '': list(range(n_windows1)),
        'var2': list(range(n_windows2))
    }
    result = get_nwindows_expected(windows)
    assert result == n_windows1 + n_windows2


def test_get_nwindows_expected_empty():
    """Test with empty windows dict."""
    result = get_nwindows_expected({})
    assert result == 0


@given(st.integers(min_value=1, max_value=1000))
def test_get_ts_window_name_index(window_size):
    """Test window naming for index-based windows."""
    name = get_ts_window_name("", window_size, "D")
    assert str(window_size) in name
    assert "D" in name


@given(st.text(min_size=1, max_size=10, alphabet='abcdefghij'))
def test_get_ts_window_name_var(var_name):
    """Test window naming for variable-based windows."""
    name = get_ts_window_name(var_name, 1000.0)
    assert var_name in name


def test_find_cumsum_left_zero_index():
    """Test with right_index = 0."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    left, total = find_next_cumsum_left_index(arr, 1.0, 0)
    assert left == 0
    assert total == 0.0


def test_find_cumsum_right_end_index():
    """Test with left_index at end."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    right, total = find_next_cumsum_right_index(arr, 1.0, len(arr) - 1)
    assert right == len(arr) - 1


@given(st.booleans())
@settings(deadline=None)
def test_find_cumsum_use_abs(use_abs):
    """Test absolute value mode."""
    arr = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float64)
    left, total = find_next_cumsum_left_index(arr, 5.0, 4, use_abs=use_abs)
    assert 0 <= left <= 4


@given(st.integers(min_value=1, max_value=10))
def test_find_cumsum_min_samples(min_samples):
    """Test min_samples constraint."""
    arr = np.ones(20, dtype=np.float64)
    left, total = find_next_cumsum_left_index(arr, 1.0, 15, min_samples=min_samples)
    # Should respect min_samples
    assert 15 - left >= min_samples or left == 0


def test_find_cumsum_with_nans():
    """Test handling of NaN values in array."""
    arr = np.array([1.0, np.nan, 2.0, np.nan, 3.0], dtype=np.float64)
    left, total = find_next_cumsum_left_index(arr, 3.0, 4)
    # Should skip NaN values
    assert 0 <= left <= 4


def test_find_cumsum_none_index():
    """Test with None index (should use array length)."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    left, total = find_next_cumsum_left_index(arr, 10.0, None)
    assert 0 <= left <= len(arr)
