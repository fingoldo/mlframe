"""Hypothesis-based tests for categorical.py module."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume

from mlframe.feature_engineering.categorical import (
    compute_countaggs,
    get_countaggs_names,
)


@given(st.lists(st.text(alphabet='abcdefghij', min_size=1, max_size=5), min_size=1, max_size=100))
@settings(max_examples=50, deadline=None)
def test_countaggs_returns_correct_length(values):
    """Test that countaggs returns expected number of features."""
    arr = pd.Series(values)
    result = compute_countaggs(arr)
    expected_names = get_countaggs_names()
    assert len(result) == len(expected_names)


@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50))
@settings(max_examples=50, deadline=None)
def test_countaggs_numeric_values(values):
    """Test countaggs with numeric values."""
    arr = pd.Series(values)
    result = compute_countaggs(arr, counts_compute_values_numaggs=True)
    assert all(isinstance(v, (int, float, np.number)) or pd.isna(v) for v in result)


@given(st.integers(min_value=1, max_value=5))
def test_countaggs_top_n_parameter(top_n):
    """Test different top_n values produce correct output length."""
    arr = pd.Series(['a', 'b', 'c', 'a', 'b', 'a', 'd', 'e', 'f'])
    result = compute_countaggs(arr, counts_top_n=top_n)
    names = get_countaggs_names(counts_top_n=top_n)
    assert len(result) == len(names)


@given(st.booleans())
def test_countaggs_normalize_parameter(normalize):
    """Test both normalize modes."""
    arr = pd.Series(['a', 'b', 'a', 'c', 'a'])
    result = compute_countaggs(arr, counts_normalize=normalize)
    names = get_countaggs_names(counts_normalize=normalize)
    assert len(result) == len(names)


@given(st.booleans(), st.booleans())
def test_countaggs_return_options(return_counts, return_values):
    """Test different return options."""
    arr = pd.Series(['x', 'y', 'x', 'z'])
    result = compute_countaggs(
        arr,
        counts_return_top_counts=return_counts,
        counts_return_top_values=return_values
    )
    names = get_countaggs_names(
        counts_return_top_counts=return_counts,
        counts_return_top_values=return_values
    )
    assert len(result) == len(names)


def test_countaggs_single_value():
    """Test with single repeated value."""
    arr = pd.Series(['a'] * 100)
    result = compute_countaggs(arr)
    names = get_countaggs_names()
    assert len(result) == len(names)


def test_countaggs_all_unique():
    """Test with all unique values."""
    arr = pd.Series(list('abcdefghij'))
    result = compute_countaggs(arr)
    names = get_countaggs_names()
    assert len(result) == len(names)


@given(st.booleans())
def test_countaggs_compute_numaggs_option(compute_numaggs):
    """Test toggling numaggs computation."""
    arr = pd.Series(['a', 'b', 'a', 'c'])
    result = compute_countaggs(arr, counts_compute_numaggs=compute_numaggs)
    names = get_countaggs_names(counts_compute_numaggs=compute_numaggs)
    assert len(result) == len(names)


def test_countaggs_empty_series():
    """Test with empty series - should handle gracefully or raise appropriate error."""
    arr = pd.Series([], dtype=object)
    try:
        result = compute_countaggs(arr)
        # If it doesn't raise, result should still match names
        names = get_countaggs_names()
        assert len(result) == len(names)
    except (ValueError, IndexError):
        # Empty series may raise an error - that's acceptable
        pass
