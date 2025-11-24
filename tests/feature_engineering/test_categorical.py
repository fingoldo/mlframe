"""Comprehensive tests for categorical.py module.

Includes:
- Regression tests against manual calculations
- Parameter coverage tests for all options
- Edge case tests
- Type handling tests
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume

from mlframe.feature_engineering.categorical import (
    compute_countaggs,
    get_countaggs_names,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def simple_categorical():
    """Simple categorical series with known distribution."""
    # 'a': 3, 'b': 2, 'c': 1
    return pd.Series(['a', 'b', 'a', 'c', 'a', 'b'])


@pytest.fixture
def numeric_categorical():
    """Numeric series treated as categorical."""
    # 1: 3, 2: 2, 3: 1
    return pd.Series([1, 2, 1, 3, 1, 2])


@pytest.fixture
def many_unique():
    """Series with many unique values."""
    return pd.Series(list('abcdefghij'))


@pytest.fixture
def single_value():
    """Series with single repeated value."""
    return pd.Series(['a'] * 100)


@pytest.fixture
def two_values():
    """Series with exactly two unique values."""
    return pd.Series(['a', 'b', 'a', 'b', 'a'])


# =============================================================================
# REGRESSION TESTS: COUNT VALUES
# =============================================================================

class TestCountValuesRegression:
    """Regression tests verifying count calculations match expected values."""

    def test_normalized_counts_sum_to_one(self, simple_categorical):
        """Test that normalized counts sum to 1.0."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=False,
                                   counts_top_n=3, counts_return_top_values=False)
        names = get_countaggs_names(counts_normalize=True, counts_compute_numaggs=False,
                                    counts_top_n=3, counts_return_top_values=False)

        # Top 3 + bottom 3 counts
        top_counts = result[:3]
        # Top counts should sum close to 1 if we have 3 unique values
        assert np.isclose(sum(top_counts), 1.0, rtol=1e-6)

    def test_absolute_counts_sum_to_length(self, simple_categorical):
        """Test that absolute counts sum to array length."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_normalize=False, counts_compute_numaggs=False,
                                   counts_top_n=3, counts_return_top_values=False)

        # Top 3 counts (all 3 unique values)
        top_counts = result[:3]
        assert sum(top_counts) == len(arr)

    def test_top_count_is_maximum(self, simple_categorical):
        """Test that top_1_vcnt is the maximum count."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=False,
                                   counts_top_n=1, counts_return_top_values=False)

        # First value should be top count (a: 3/6 = 0.5)
        top_count = result[0]
        expected = 3.0 / 6.0  # 'a' appears 3 times out of 6
        assert np.isclose(top_count, expected, rtol=1e-6)

    def test_bottom_count_is_minimum(self, simple_categorical):
        """Test that btm_1_vcnt is the minimum count."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=False,
                                   counts_top_n=1, counts_return_top_values=False)

        # Second value should be bottom count (c: 1/6)
        bottom_count = result[1]
        expected = 1.0 / 6.0  # 'c' appears 1 time out of 6
        assert np.isclose(bottom_count, expected, rtol=1e-6)

    def test_top_value_is_most_frequent(self, numeric_categorical):
        """Test that top_1_vval is the most frequent value."""
        arr = numeric_categorical
        result = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=False,
                                   counts_top_n=1, counts_return_top_counts=False)

        # First value should be the most frequent (1 appears 3 times)
        top_value = result[0]
        assert top_value == 1

    def test_counts_order_descending(self, simple_categorical):
        """Test that top counts are in descending order."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=False,
                                   counts_top_n=3, counts_return_top_values=False)

        top_counts = result[:3]
        # Should be sorted descending
        assert top_counts[0] >= top_counts[1] >= top_counts[2]

    def test_mean_of_normalized_counts(self, simple_categorical):
        """Test mean of normalized counts matches expected value."""
        arr = simple_categorical
        # With 3 unique values, mean of normalized counts = 1/3
        result = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=True,
                                   counts_top_n=0)
        names = get_countaggs_names(counts_normalize=True, counts_compute_numaggs=True,
                                    counts_top_n=0)

        # Find arithmetic mean
        arimean_idx = names.index('arimean_cntnrm')
        computed_mean = result[arimean_idx]
        expected_mean = 1.0 / 3.0  # 3 unique values, each normalized count averages to 1/3
        assert np.isclose(computed_mean, expected_mean, rtol=1e-6)


# =============================================================================
# PARAMETER COVERAGE TESTS
# =============================================================================

class TestParameterCoverage:
    """Test all parameter combinations produce consistent output."""

    @pytest.mark.parametrize("normalize", [True, False])
    def test_normalize_output_length(self, simple_categorical, normalize):
        """Test counts_normalize parameter affects output correctly."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_normalize=normalize)
        names = get_countaggs_names(counts_normalize=normalize)
        assert len(result) == len(names)

    @pytest.mark.parametrize("compute_numaggs", [True, False])
    def test_compute_numaggs_output_length(self, simple_categorical, compute_numaggs):
        """Test counts_compute_numaggs parameter."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_compute_numaggs=compute_numaggs)
        names = get_countaggs_names(counts_compute_numaggs=compute_numaggs)
        assert len(result) == len(names)

    @pytest.mark.parametrize("top_n", [0, 1, 2, 3, 5])
    def test_top_n_output_length(self, simple_categorical, top_n):
        """Test counts_top_n parameter."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_top_n=top_n)
        names = get_countaggs_names(counts_top_n=top_n)
        assert len(result) == len(names)

    @pytest.mark.parametrize("return_counts", [True, False])
    def test_return_top_counts_output_length(self, simple_categorical, return_counts):
        """Test counts_return_top_counts parameter."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_return_top_counts=return_counts)
        names = get_countaggs_names(counts_return_top_counts=return_counts)
        assert len(result) == len(names)

    @pytest.mark.parametrize("return_values", [True, False])
    def test_return_top_values_output_length(self, simple_categorical, return_values):
        """Test counts_return_top_values parameter."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_return_top_values=return_values)
        names = get_countaggs_names(counts_return_top_values=return_values)
        assert len(result) == len(names)

    @pytest.mark.parametrize("compute_values_numaggs", [True, False])
    def test_compute_values_numaggs_with_numeric(self, numeric_categorical, compute_values_numaggs):
        """Test counts_compute_values_numaggs with numeric data."""
        arr = numeric_categorical
        result = compute_countaggs(arr, counts_compute_values_numaggs=compute_values_numaggs)
        names = get_countaggs_names(counts_compute_values_numaggs=compute_values_numaggs)
        assert len(result) == len(names)

    def test_all_options_disabled(self, simple_categorical):
        """Test with all options disabled returns minimal output."""
        arr = simple_categorical
        result = compute_countaggs(
            arr,
            counts_compute_numaggs=False,
            counts_top_n=0,
            counts_compute_values_numaggs=False
        )
        names = get_countaggs_names(
            counts_compute_numaggs=False,
            counts_top_n=0,
            counts_compute_values_numaggs=False
        )
        assert len(result) == len(names)
        assert len(result) == 0  # Should be empty

    def test_numerical_kwargs_propagation(self, simple_categorical):
        """Test that numerical_kwargs are properly passed through."""
        arr = simple_categorical

        # With exotic means
        result1 = compute_countaggs(
            arr,
            numerical_kwargs={'return_exotic_means': True, 'return_unsorted_stats': False}
        )
        names1 = get_countaggs_names(
            numerical_kwargs={'return_exotic_means': True, 'return_unsorted_stats': False}
        )

        # Without exotic means
        result2 = compute_countaggs(
            arr,
            numerical_kwargs={'return_exotic_means': False, 'return_unsorted_stats': False}
        )
        names2 = get_countaggs_names(
            numerical_kwargs={'return_exotic_means': False, 'return_unsorted_stats': False}
        )

        # Different lengths due to exotic means inclusion
        assert len(result1) == len(names1)
        assert len(result2) == len(names2)
        assert len(result1) != len(result2)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_repeated_value(self, single_value):
        """Test with single repeated value (one unique)."""
        arr = single_value
        result = compute_countaggs(arr, counts_top_n=3)
        names = get_countaggs_names(counts_top_n=3)
        assert len(result) == len(names)

        # With only 1 unique value, positions 2 and 3 should be NaN
        # Find top_2_vcnt and top_3_vcnt positions
        top_2_idx = names.index('top_2_vcnt')
        top_3_idx = names.index('top_3_vcnt')
        assert np.isnan(result[top_2_idx])
        assert np.isnan(result[top_3_idx])

    def test_padding_when_top_n_exceeds_unique(self, two_values):
        """Test behavior when top_n exceeds number of unique values."""
        arr = two_values  # Only 2 unique values
        result = compute_countaggs(arr, counts_compute_numaggs=False,
                                   counts_top_n=3, counts_return_top_values=False)
        names = get_countaggs_names(counts_compute_numaggs=False,
                                    counts_top_n=3, counts_return_top_values=False)

        # Should have 3 top + 3 bottom = 6 values
        assert len(result) == 6
        assert len(names) == len(result)

        # First two positions should have valid counts, extras may be NaN
        assert np.isfinite(result[0])  # top_1
        assert np.isfinite(result[1])  # top_2

    def test_all_unique_values(self, many_unique):
        """Test with all unique values."""
        arr = many_unique  # 10 unique values
        result = compute_countaggs(arr, counts_normalize=True, counts_top_n=5)
        names = get_countaggs_names(counts_normalize=True, counts_top_n=5)
        assert len(result) == len(names)

        # All counts should be equal (1/10 each)
        # Find top_1_vcnt
        idx = names.index('top_1_vcnt') if 'top_1_vcnt' in names else None
        if idx is not None:
            assert np.isclose(result[idx], 0.1, rtol=1e-6)

    def test_empty_series(self):
        """Test with empty series."""
        arr = pd.Series([], dtype=object)
        try:
            result = compute_countaggs(arr)
            names = get_countaggs_names()
            assert len(result) == len(names)
        except (ValueError, IndexError, ZeroDivisionError):
            # Empty series may raise an error - that's acceptable
            pass

    def test_series_with_nans(self):
        """Test series containing NaN values."""
        arr = pd.Series(['a', 'b', np.nan, 'a', np.nan, 'b'])
        result = compute_countaggs(arr)
        names = get_countaggs_names()
        assert len(result) == len(names)
        # NaN values should be filtered out by value_counts

    def test_top_n_zero(self, simple_categorical):
        """Test with counts_top_n=0 returns no top/bottom features."""
        arr = simple_categorical
        result = compute_countaggs(arr, counts_top_n=0)
        names = get_countaggs_names(counts_top_n=0)

        # Should have no top/bottom features
        assert 'top_1_vcnt' not in names
        assert 'btm_1_vcnt' not in names
        assert len(result) == len(names)

    def test_top_n_exceeds_unique_count(self, two_values):
        """Test when top_n exceeds number of unique values."""
        arr = two_values  # Only 2 unique
        result = compute_countaggs(arr, counts_top_n=10)
        names = get_countaggs_names(counts_top_n=10)

        assert len(result) == len(names)
        # Most positions should be NaN
        nan_count = sum(1 for v in result if pd.isna(v))
        assert nan_count > 0


# =============================================================================
# TYPE HANDLING TESTS
# =============================================================================

class TestTypeHandling:
    """Test handling of different data types."""

    def test_string_values(self):
        """Test with string values."""
        arr = pd.Series(['apple', 'banana', 'apple', 'cherry'])
        result = compute_countaggs(arr)
        names = get_countaggs_names()
        assert len(result) == len(names)

    def test_integer_values(self):
        """Test with integer values."""
        arr = pd.Series([1, 2, 1, 3, 1, 2])
        result = compute_countaggs(arr)
        names = get_countaggs_names()
        assert len(result) == len(names)

    def test_float_values(self):
        """Test with float values."""
        arr = pd.Series([1.5, 2.5, 1.5, 3.5])
        result = compute_countaggs(arr)
        names = get_countaggs_names()
        assert len(result) == len(names)

    def test_mixed_numeric_types(self):
        """Test with mixed int and float."""
        arr = pd.Series([1, 2.5, 1, 3])
        result = compute_countaggs(arr)
        names = get_countaggs_names()
        assert len(result) == len(names)

    def test_values_numaggs_with_numeric(self, numeric_categorical):
        """Test counts_compute_values_numaggs=True with numeric values."""
        arr = numeric_categorical
        result = compute_countaggs(arr, counts_compute_values_numaggs=True)
        names = get_countaggs_names(counts_compute_values_numaggs=True)

        # Should have directional numaggs features
        assert 'arimean_vvls' in names
        assert 'ratio_vvls' in names

        # Find the arimean_vvls value - should be finite
        idx = names.index('arimean_vvls')
        assert np.isfinite(result[idx])

    def test_values_numaggs_with_non_numeric(self, simple_categorical):
        """Test counts_compute_values_numaggs=True with non-numeric values returns NaN."""
        arr = simple_categorical  # String values
        result = compute_countaggs(arr, counts_compute_values_numaggs=True)
        names = get_countaggs_names(counts_compute_values_numaggs=True)

        # Should have directional numaggs features but with NaN values
        assert 'arimean_vvls' in names
        idx = names.index('arimean_vvls')
        assert np.isnan(result[idx])

    def test_boolean_values(self):
        """Test with boolean values."""
        arr = pd.Series([True, False, True, True, False])
        result = compute_countaggs(arr)
        names = get_countaggs_names()
        assert len(result) == len(names)


# =============================================================================
# FEATURE NAME CONSISTENCY TESTS
# =============================================================================

class TestFeatureNameConsistency:
    """Test that feature names are consistent with output."""

    def test_names_match_result_length(self, simple_categorical):
        """Test that names length always matches result length."""
        arr = simple_categorical

        # Test multiple parameter combinations
        param_combos = [
            {},
            {'counts_normalize': False},
            {'counts_top_n': 5},
            {'counts_compute_numaggs': False},
            {'counts_return_top_counts': False},
            {'counts_return_top_values': False},
            {'counts_compute_values_numaggs': True},
        ]

        for params in param_combos:
            result = compute_countaggs(arr, **params)
            names = get_countaggs_names(**params)
            assert len(result) == len(names), f"Mismatch with params: {params}"

    def test_name_suffix_reflects_normalize(self):
        """Test that feature name suffix reflects normalization setting."""
        arr = pd.Series(['a', 'b', 'a'])

        # Normalized
        names_norm = get_countaggs_names(counts_normalize=True, counts_compute_numaggs=True)
        assert any('_cntnrm' in name for name in names_norm)
        assert not any('_cnt' in name and '_cntnrm' not in name for name in names_norm)

        # Not normalized
        names_abs = get_countaggs_names(counts_normalize=False, counts_compute_numaggs=True)
        assert any('_cnt' in name and '_cntnrm' not in name for name in names_abs)

    def test_top_n_feature_names(self):
        """Test that top_n features have correct naming pattern."""
        names = get_countaggs_names(counts_top_n=3, counts_compute_numaggs=False)

        # Should have top_1, top_2, top_3 and btm_3, btm_2, btm_1
        expected_patterns = [
            'top_1_vcnt', 'top_2_vcnt', 'top_3_vcnt',
            'btm_3_vcnt', 'btm_2_vcnt', 'btm_1_vcnt',
            'top_1_vval', 'top_2_vval', 'top_3_vval',
            'btm_3_vval', 'btm_2_vval', 'btm_1_vval',
        ]

        for pattern in expected_patterns:
            assert pattern in names, f"Missing expected name: {pattern}"

    def test_values_numaggs_feature_names(self):
        """Test that values numaggs features have correct naming."""
        names = get_countaggs_names(counts_compute_values_numaggs=True)

        # Should have _vvls suffix for directional features
        assert 'arimean_vvls' in names
        assert 'ratio_vvls' in names


# =============================================================================
# HYPOTHESIS PROPERTY-BASED TESTS
# =============================================================================

class TestHypothesisProperties:
    """Property-based tests using Hypothesis."""

    @given(st.lists(st.text(alphabet='abcdefghij', min_size=1, max_size=5),
                    min_size=1, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_output_length_matches_names(self, values):
        """Test that output length always matches names length."""
        arr = pd.Series(values)
        result = compute_countaggs(arr)
        names = get_countaggs_names()
        assert len(result) == len(names)

    @given(st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50))
    @settings(max_examples=50, deadline=None)
    def test_numeric_values_produce_valid_output(self, values):
        """Test countaggs with numeric values produces valid output."""
        arr = pd.Series(values)
        result = compute_countaggs(arr, counts_compute_values_numaggs=True)
        assert all(isinstance(v, (int, float, np.number)) or pd.isna(v) for v in result)

    @given(st.integers(min_value=1, max_value=10))
    @settings(deadline=None)
    def test_top_n_produces_correct_length(self, top_n):
        """Test different top_n values produce correct output length."""
        arr = pd.Series(['a', 'b', 'c', 'a', 'b', 'a', 'd', 'e', 'f'])
        result = compute_countaggs(arr, counts_top_n=top_n)
        names = get_countaggs_names(counts_top_n=top_n)
        assert len(result) == len(names)

    @given(st.booleans())
    @settings(deadline=None)
    def test_normalize_produces_correct_length(self, normalize):
        """Test both normalize modes produce correct length."""
        arr = pd.Series(['a', 'b', 'a', 'c', 'a'])
        result = compute_countaggs(arr, counts_normalize=normalize)
        names = get_countaggs_names(counts_normalize=normalize)
        assert len(result) == len(names)

    @given(st.booleans(), st.booleans())
    @settings(deadline=None)
    def test_return_options_produce_correct_length(self, return_counts, return_values):
        """Test different return options produce correct length."""
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

    @given(st.booleans())
    @settings(deadline=None)
    def test_compute_numaggs_toggle(self, compute_numaggs):
        """Test toggling numaggs computation."""
        arr = pd.Series(['a', 'b', 'a', 'c'])
        result = compute_countaggs(arr, counts_compute_numaggs=compute_numaggs)
        names = get_countaggs_names(counts_compute_numaggs=compute_numaggs)
        assert len(result) == len(names)


# =============================================================================
# FUNCTIONAL CORRECTNESS TESTS
# =============================================================================

class TestFunctionalCorrectness:
    """Test functional correctness with known inputs."""

    def test_known_distribution_counts(self):
        """Test with known distribution produces expected counts."""
        # Create distribution: a=5, b=3, c=2
        arr = pd.Series(['a'] * 5 + ['b'] * 3 + ['c'] * 2)
        result = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=False,
                                   counts_top_n=3, counts_return_top_values=False)

        # Expected normalized counts: [0.5, 0.3, 0.2]
        expected_top = [0.5, 0.3, 0.2]

        for i, expected in enumerate(expected_top):
            assert np.isclose(result[i], expected, rtol=1e-6), \
                f"Position {i}: {result[i]} vs {expected}"

    def test_known_distribution_values(self):
        """Test with known distribution produces expected top values."""
        # Create distribution: 10=5, 20=3, 30=2
        arr = pd.Series([10] * 5 + [20] * 3 + [30] * 2)
        result = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=False,
                                   counts_top_n=3, counts_return_top_counts=False)

        # Top values should be [10, 20, 30]
        top_values = result[:3]
        assert top_values[0] == 10  # Most frequent
        assert top_values[1] == 20
        assert top_values[2] == 30  # Least frequent

    def test_uniform_distribution(self):
        """Test with uniform distribution (all same count)."""
        arr = pd.Series(['a', 'b', 'c', 'd', 'e'])  # All unique
        result = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=False,
                                   counts_top_n=3, counts_return_top_values=False)

        # All counts should be 0.2
        for i in range(3):
            assert np.isclose(result[i], 0.2, rtol=1e-6)

    def test_single_dominant_value(self):
        """Test with one dominant value."""
        arr = pd.Series(['a'] * 99 + ['b'])
        result = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=False,
                                   counts_top_n=1, counts_return_top_values=False)

        # Top count should be 0.99
        assert np.isclose(result[0], 0.99, rtol=1e-6)
        # Bottom count should be 0.01
        assert np.isclose(result[1], 0.01, rtol=1e-6)

    def test_absolute_vs_normalized_consistency(self):
        """Test that absolute and normalized counts are consistent."""
        arr = pd.Series(['a'] * 6 + ['b'] * 4)  # Total = 10

        # Absolute counts
        result_abs = compute_countaggs(arr, counts_normalize=False, counts_compute_numaggs=False,
                                       counts_top_n=2, counts_return_top_values=False)

        # Normalized counts
        result_norm = compute_countaggs(arr, counts_normalize=True, counts_compute_numaggs=False,
                                        counts_top_n=2, counts_return_top_values=False)

        # Absolute: [6, 4] and Normalized: [0.6, 0.4]
        assert result_abs[0] == 6
        assert result_abs[1] == 4
        assert np.isclose(result_norm[0], 0.6, rtol=1e-6)
        assert np.isclose(result_norm[1], 0.4, rtol=1e-6)

    def test_min_max_of_counts(self):
        """Test min/max of counts are correct."""
        # Distribution: a=5, b=3, c=2, d=1
        arr = pd.Series(['a'] * 5 + ['b'] * 3 + ['c'] * 2 + ['d'] * 1)
        result = compute_countaggs(arr, counts_normalize=False, counts_compute_numaggs=True,
                                   counts_top_n=0)
        names = get_countaggs_names(counts_normalize=False, counts_compute_numaggs=True,
                                    counts_top_n=0)

        # Find min and max
        min_idx = names.index('min_cnt')
        max_idx = names.index('max_cnt')

        assert result[min_idx] == 1  # 'd' count
        assert result[max_idx] == 5  # 'a' count
