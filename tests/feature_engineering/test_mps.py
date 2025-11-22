"""Hypothesis-based tests for mps.py module."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from mlframe.feature_engineering.mps import (
    find_maximum_profit_system,
    compute_area_profits,
    backfill_zeros,
)


@given(st.lists(st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False), min_size=3, max_size=100))
@settings(max_examples=50, deadline=None)
def test_mps_positions_valid(prices):
    """Test that positions are always -1, 0, or 1."""
    prices = np.array(prices, dtype=np.float64)
    result = find_maximum_profit_system(prices)
    positions = result['positions']
    assert all(p in [-1, 0, 1] for p in positions), f"Invalid positions: {set(positions)}"


@given(st.lists(st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False), min_size=5, max_size=100))
@settings(max_examples=50)
def test_mps_profits_finite(prices):
    """Test that profits don't contain NaN or Inf."""
    prices = np.array(prices, dtype=np.float64)
    result = find_maximum_profit_system(prices)
    profits = result['profits']
    assert np.all(np.isfinite(profits)), f"Non-finite profits: {profits[~np.isfinite(profits)]}"


@given(st.floats(min_value=0, max_value=0.01))
def test_mps_tc_parameter(tc):
    """Test with various transaction costs."""
    prices = np.array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0], dtype=np.float64)
    result = find_maximum_profit_system(prices, tc=tc)
    assert len(result['positions']) == len(prices) - 1


@given(st.sampled_from(['fraction', 'fixed']))
def test_mps_tc_mode(tc_mode):
    """Test both transaction cost modes."""
    prices = np.array([100.0, 105.0, 95.0, 110.0], dtype=np.float64)
    result = find_maximum_profit_system(prices, tc=0.001, tc_mode=tc_mode)
    assert 'positions' in result
    assert 'profits' in result


def test_mps_output_length():
    """Test that output length is correct (n-1)."""
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64)
    result = find_maximum_profit_system(prices)
    assert len(result['positions']) == len(prices) - 1
    assert len(result['profits']) == len(prices)


def test_mps_two_prices():
    """Test with minimum array (2 prices)."""
    prices = np.array([100.0, 105.0], dtype=np.float64)
    result = find_maximum_profit_system(prices)
    assert len(result['positions']) == 1


def test_mps_single_price():
    """Test with single price - should return empty."""
    prices = np.array([100.0], dtype=np.float64)
    result = find_maximum_profit_system(prices)
    assert len(result['positions']) == 0


@given(st.integers(min_value=0, max_value=5))
def test_mps_shift_parameter(shift):
    """Test shift parameter."""
    prices = np.random.rand(20).astype(np.float64) * 100 + 50
    result = find_maximum_profit_system(prices, shift=shift)
    assert len(result['positions']) == len(prices) - 1


@given(st.booleans())
def test_mps_optimize_consecutive_regions(optimize):
    """Test optimize_consecutive_regions parameter."""
    prices = np.array([100.0, 101.0, 100.5, 102.0, 101.5, 103.0], dtype=np.float64)
    result = find_maximum_profit_system(prices, optimize_consecutive_regions=optimize)
    assert len(result['positions']) == len(prices) - 1


def test_backfill_zeros_right():
    """Test backfill from right direction."""
    arr = np.array([0, 0, 1, 0, 0, -1, 0, 0], dtype=np.int8)
    result = backfill_zeros(arr, direction='right')
    expected = np.array([1, 1, 1, -1, -1, -1, 0, 0], dtype=np.int8)
    assert np.array_equal(result, expected)


def test_backfill_zeros_left():
    """Test backfill from left direction."""
    arr = np.array([0, 0, 1, 0, 0, -1, 0, 0], dtype=np.int8)
    result = backfill_zeros(arr, direction='left')
    expected = np.array([0, 0, 1, 1, 1, -1, -1, -1], dtype=np.int8)
    assert np.array_equal(result, expected)


def test_backfill_zeros_all_zeros():
    """Test backfill with all zeros."""
    arr = np.zeros(5, dtype=np.int8)
    result = backfill_zeros(arr, direction='right')
    assert np.array_equal(result, arr)


def test_backfill_zeros_no_zeros():
    """Test backfill with no zeros."""
    arr = np.array([1, -1, 1, -1], dtype=np.int8)
    result = backfill_zeros(arr, direction='right')
    assert np.array_equal(result, arr)


def test_mps_uptrend():
    """Test with clear uptrend - should be mostly long."""
    prices = np.linspace(100, 200, 50).astype(np.float64)
    result = find_maximum_profit_system(prices, tc=0)
    # Should have mostly positive positions for uptrend
    assert sum(result['positions'] == 1) >= sum(result['positions'] == -1)


def test_mps_downtrend():
    """Test with clear downtrend - should be mostly short."""
    prices = np.linspace(200, 100, 50).astype(np.float64)
    result = find_maximum_profit_system(prices, tc=0)
    # Should have mostly negative positions for downtrend
    assert sum(result['positions'] == -1) >= sum(result['positions'] == 1)


def test_compute_area_profits_basic():
    """Test area profits calculation."""
    prices = np.array([100.0, 110.0, 105.0, 115.0], dtype=np.float64)
    positions = np.array([1, 1, 1], dtype=np.int8)
    profits = compute_area_profits(prices, positions)
    assert len(profits) == len(prices)
    assert profits[-1] == 0.0  # Last element should be 0
