"""Hypothesis-based tests for numerical.py module."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from mlframe.feature_engineering.numerical import (
    compute_numaggs,
    get_numaggs_names,
    compute_simple_stats_numba,
    compute_nunique_modes_quantiles_numpy,
    rolling_moving_average,
)


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=100))
@settings(max_examples=50, deadline=None)
def test_compute_numaggs_output_length(arr):
    """Test that numaggs returns consistent length."""
    arr = np.array(arr, dtype=np.float32)
    result = compute_numaggs(arr)
    expected_names = get_numaggs_names()
    assert len(result) == len(expected_names)


@given(st.lists(st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False), min_size=5, max_size=100))
@settings(max_examples=50, deadline=None)
def test_simple_stats_min_max_correct(arr):
    """Test min/max calculation."""
    arr = np.array(arr, dtype=np.float64)
    min_val, max_val, argmin, argmax, mean_val, std_val = compute_simple_stats_numba(arr)
    assert np.isclose(min_val, arr.min()), f"Min mismatch: {min_val} vs {arr.min()}"
    assert np.isclose(max_val, arr.max()), f"Max mismatch: {max_val} vs {arr.max()}"


@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=10, max_size=100))
@settings(max_examples=30)
def test_quantiles_ordered(values):
    """Test that quantiles are properly ordered."""
    arr = np.array(values, dtype=np.float64)
    result = compute_nunique_modes_quantiles_numpy(arr)
    # Extract quantiles (after nunique, modes_min, modes_max, modes_mean, modes_qty)
    quantiles = result[5:10]  # Default 5 quantiles
    for i in range(len(quantiles) - 1):
        assert quantiles[i] <= quantiles[i + 1], f"Quantiles not ordered: {quantiles}"


@given(st.integers(min_value=2, max_value=50))
@settings(deadline=None)
def test_rolling_moving_average_length(n):
    """Test rolling MA output length."""
    arr = np.random.rand(100).astype(np.float64)
    result = rolling_moving_average(arr, n)
    expected_len = len(arr) - n + 1
    assert len(result) == expected_len


@given(st.booleans())
def test_numaggs_directional_only(directional_only):
    """Test directional_only parameter."""
    arr = np.random.randn(50).astype(np.float32)
    result = compute_numaggs(arr, directional_only=directional_only)
    names = get_numaggs_names(directional_only=directional_only)
    assert len(result) == len(names)


@given(st.booleans())
def test_numaggs_return_entropy(return_entropy):
    """Test return_entropy parameter."""
    arr = np.random.randn(50).astype(np.float32)
    result = compute_numaggs(arr, return_entropy=return_entropy)
    names = get_numaggs_names(return_entropy=return_entropy)
    assert len(result) == len(names)


@given(st.booleans())
def test_numaggs_return_hurst(return_hurst):
    """Test return_hurst parameter."""
    arr = np.random.randn(50).astype(np.float32)
    result = compute_numaggs(arr, return_hurst=return_hurst)
    names = get_numaggs_names(return_hurst=return_hurst)
    assert len(result) == len(names)


def test_numaggs_single_element():
    """Test with single element array - should return NaNs."""
    arr = np.array([1.0], dtype=np.float32)
    result = compute_numaggs(arr)
    assert all(np.isnan(v) for v in result)


def test_numaggs_two_elements():
    """Test with two element array."""
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = compute_numaggs(arr)
    names = get_numaggs_names()
    assert len(result) == len(names)


@given(st.lists(st.floats(min_value=0.01, max_value=0.99), min_size=1, max_size=5, unique=True))
@settings(max_examples=30)
def test_numaggs_custom_quantiles(quantiles):
    """Test with custom quantile values."""
    quantiles = sorted(quantiles)
    arr = np.random.randn(50).astype(np.float32)
    result = compute_numaggs(arr, q=quantiles)
    names = get_numaggs_names(q=quantiles)
    assert len(result) == len(names)


def test_rolling_ma_basic():
    """Test basic rolling MA calculation."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = rolling_moving_average(arr, 3)
    expected = np.array([2.0, 3.0, 4.0])
    assert np.allclose(result, expected)


def test_rolling_ma_window_too_large():
    """Test that window > array length raises error."""
    arr = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        rolling_moving_average(arr, 5)


def test_simple_stats_all_nan():
    """Test with array containing NaN values."""
    arr = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    min_val, max_val, argmin, argmax, mean_val, std_val = compute_simple_stats_numba(arr)
    # Should handle gracefully
    assert isinstance(mean_val, float)


def test_simple_stats_mixed_finite():
    """Test with mixed finite and non-finite values."""
    arr = np.array([1.0, np.nan, 3.0, np.inf, 2.0], dtype=np.float64)
    min_val, max_val, argmin, argmax, mean_val, std_val = compute_simple_stats_numba(arr)
    # Should only consider finite values
    assert min_val == 1.0
    assert max_val == 3.0
