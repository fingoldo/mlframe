"""Comprehensive tests for numerical.py module.

Includes:
- Regression tests against numpy/scipy/sklearn reference implementations
- Parameter coverage tests for all options
- Performance benchmarks
- Edge case tests
"""

import pytest
import numpy as np
from scipy import stats
from hypothesis import given, strategies as st, settings, assume

from mlframe.feature_engineering.numerical import (
    compute_numaggs,
    get_numaggs_names,
    compute_simple_stats_numba,
    compute_nunique_modes_quantiles_numpy,
    compute_numerical_aggregates_numba,
    compute_moments_slope_mi,
    compute_ncrossings,
    rolling_moving_average,
    get_basic_feature_names,
    get_moments_slope_mi_feature_names,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def random_array_100():
    """Random array with 100 elements."""
    np.random.seed(42)
    return np.random.randn(100).astype(np.float64)


@pytest.fixture
def random_array_1000():
    """Random array with 1000 elements."""
    np.random.seed(42)
    return np.random.randn(1000).astype(np.float64)


@pytest.fixture
def positive_array():
    """Array with only positive values (for geometric/harmonic mean)."""
    np.random.seed(42)
    return (np.abs(np.random.randn(100)) + 0.1).astype(np.float64)


@pytest.fixture
def known_signal():
    """Deterministic sinusoidal signal for exact value tests."""
    return np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float64)


@pytest.fixture
def weights_array():
    """Weights array for weighted statistics tests."""
    np.random.seed(123)
    w = np.abs(np.random.randn(100)) + 0.1
    return w.astype(np.float64)


# =============================================================================
# REGRESSION TESTS: BASIC STATISTICS VS NUMPY
# =============================================================================

class TestBasicStatsRegression:
    """Regression tests for basic statistics against numpy."""

    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                    min_size=5, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_min_vs_numpy(self, arr):
        """Test minimum matches numpy."""
        arr = np.array(arr, dtype=np.float64)
        min_val, _, _, _, _, _ = compute_simple_stats_numba(arr)
        assert np.isclose(min_val, np.min(arr), rtol=1e-10)

    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                    min_size=5, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_max_vs_numpy(self, arr):
        """Test maximum matches numpy."""
        arr = np.array(arr, dtype=np.float64)
        _, max_val, _, _, _, _ = compute_simple_stats_numba(arr)
        assert np.isclose(max_val, np.max(arr), rtol=1e-10)

    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                    min_size=5, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_argmin_vs_numpy(self, arr):
        """Test argmin matches numpy."""
        arr = np.array(arr, dtype=np.float64)
        _, _, argmin, _, _, _ = compute_simple_stats_numba(arr)
        assert int(argmin) == np.argmin(arr)

    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                    min_size=5, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_argmax_vs_numpy(self, arr):
        """Test argmax matches numpy."""
        arr = np.array(arr, dtype=np.float64)
        _, _, _, argmax, _, _ = compute_simple_stats_numba(arr)
        assert int(argmax) == np.argmax(arr)

    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                    min_size=5, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_mean_vs_numpy(self, arr):
        """Test mean matches numpy."""
        arr = np.array(arr, dtype=np.float64)
        _, _, _, _, mean_val, _ = compute_simple_stats_numba(arr)
        assert np.isclose(mean_val, np.mean(arr), rtol=1e-10)

    @given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                    min_size=5, max_size=200))
    @settings(max_examples=50, deadline=None)
    def test_std_vs_numpy(self, arr):
        """Test std matches numpy (ddof=0)."""
        arr = np.array(arr, dtype=np.float64)
        _, _, _, _, _, std_val = compute_simple_stats_numba(arr)
        assert np.isclose(std_val, np.std(arr, ddof=0), rtol=1e-10)


# =============================================================================
# REGRESSION TESTS: EXOTIC MEANS VS SCIPY/MANUAL
# =============================================================================

class TestExoticMeansRegression:
    """Regression tests for exotic means (quadratic, cubic, geometric, harmonic)."""

    def test_quadratic_mean_vs_reference(self, positive_array):
        """Test quadratic mean (RMS) matches manual calculation."""
        arr = positive_array
        result = compute_numerical_aggregates_numba(arr, whiten_means=False, return_exotic_means=True)
        names = get_basic_feature_names(whiten_means=False, return_exotic_means=True)

        # Find quadratic mean in results
        quad_idx = names.index('quadmean')
        computed = result[quad_idx]
        expected = np.sqrt(np.mean(arr ** 2))
        assert np.isclose(computed, expected, rtol=1e-6), f"Quadratic mean: {computed} vs {expected}"

    def test_cubic_mean_vs_reference(self, positive_array):
        """Test cubic mean matches manual calculation."""
        arr = positive_array  # Use positive array for valid cubic mean
        result = compute_numerical_aggregates_numba(arr, whiten_means=False, return_exotic_means=True)
        names = get_basic_feature_names(whiten_means=False, return_exotic_means=True)

        cubic_idx = names.index('qubmean')
        computed = result[cubic_idx]
        expected = np.cbrt(np.mean(arr ** 3))
        assert np.isclose(computed, expected, rtol=1e-6), f"Cubic mean: {computed} vs {expected}"

    def test_geometric_mean_vs_scipy(self, positive_array):
        """Test geometric mean matches scipy.stats.gmean."""
        arr = positive_array
        result = compute_numerical_aggregates_numba(arr, whiten_means=False, return_exotic_means=True)
        names = get_basic_feature_names(whiten_means=False, return_exotic_means=True)

        geo_idx = names.index('geomean')
        computed = result[geo_idx]
        expected = stats.gmean(arr)
        assert np.isclose(computed, expected, rtol=1e-6), f"Geometric mean: {computed} vs {expected}"

    def test_harmonic_mean_vs_scipy(self, positive_array):
        """Test harmonic mean matches scipy.stats.hmean."""
        arr = positive_array
        result = compute_numerical_aggregates_numba(arr, whiten_means=False, return_exotic_means=True)
        names = get_basic_feature_names(whiten_means=False, return_exotic_means=True)

        harm_idx = names.index('harmmean')
        computed = result[harm_idx]
        expected = stats.hmean(arr)
        assert np.isclose(computed, expected, rtol=1e-6), f"Harmonic mean: {computed} vs {expected}"

    def test_whitened_means_subtract_arithmetic(self, positive_array):
        """Test that whitened means properly subtract arithmetic mean."""
        arr = positive_array
        arith_mean = np.mean(arr)

        # Non-whitened
        result_raw = compute_numerical_aggregates_numba(arr, whiten_means=False, return_exotic_means=True)
        names = get_basic_feature_names(whiten_means=False, return_exotic_means=True)
        quad_raw = result_raw[names.index('quadmean')]

        # Whitened (note: whitened names have 'w' suffix)
        result_white = compute_numerical_aggregates_numba(arr, whiten_means=True, return_exotic_means=True)
        names_white = get_basic_feature_names(whiten_means=True, return_exotic_means=True)
        quad_white = result_white[names_white.index('quadmeanw')]

        assert np.isclose(quad_white, quad_raw - arith_mean, rtol=1e-6)


# =============================================================================
# REGRESSION TESTS: MOMENTS VS SCIPY
# =============================================================================

class TestMomentsRegression:
    """Regression tests for statistical moments against scipy."""

    def test_mad_vs_numpy(self, random_array_100):
        """Test MAD matches numpy calculation."""
        arr = random_array_100
        mean_val = np.mean(arr)
        result, _ = compute_moments_slope_mi(arr, mean_val, directional_only=False)
        names = get_moments_slope_mi_feature_names(directional_only=False)

        mad_idx = names.index('mad')
        computed = result[mad_idx]
        expected = np.mean(np.abs(arr - mean_val))
        assert np.isclose(computed, expected, rtol=1e-6), f"MAD: {computed} vs {expected}"

    def test_std_vs_numpy_in_moments(self, random_array_100):
        """Test std in moments matches numpy."""
        arr = random_array_100
        mean_val = np.mean(arr)
        result, _ = compute_moments_slope_mi(arr, mean_val, directional_only=False)
        names = get_moments_slope_mi_feature_names(directional_only=False)

        std_idx = names.index('std')
        computed = result[std_idx]
        expected = np.std(arr, ddof=0)
        assert np.isclose(computed, expected, rtol=1e-6), f"Std: {computed} vs {expected}"

    def test_skewness_vs_scipy(self, random_array_100):
        """Test skewness matches scipy.stats.skew."""
        arr = random_array_100
        mean_val = np.mean(arr)
        result, _ = compute_moments_slope_mi(arr, mean_val, directional_only=False)
        names = get_moments_slope_mi_feature_names(directional_only=False)

        skew_idx = names.index('skew')
        computed = result[skew_idx]
        expected = stats.skew(arr, bias=True)
        assert np.isclose(computed, expected, rtol=1e-4), f"Skewness: {computed} vs {expected}"

    def test_kurtosis_vs_scipy(self, random_array_100):
        """Test kurtosis matches scipy.stats.kurtosis (Fisher=True)."""
        arr = random_array_100
        mean_val = np.mean(arr)
        result, _ = compute_moments_slope_mi(arr, mean_val, directional_only=False)
        names = get_moments_slope_mi_feature_names(directional_only=False)

        kurt_idx = names.index('kurt')
        computed = result[kurt_idx]
        expected = stats.kurtosis(arr, fisher=True, bias=True)
        assert np.isclose(computed, expected, rtol=1e-4), f"Kurtosis: {computed} vs {expected}"


# =============================================================================
# REGRESSION TESTS: QUANTILES AND TREND
# =============================================================================

class TestQuantilesAndTrendRegression:
    """Regression tests for quantiles and linear trend."""

    @pytest.mark.parametrize("q", [
        [0.1, 0.5, 0.9],
        [0.25, 0.5, 0.75],
        [0.05, 0.25, 0.5, 0.75, 0.95],
    ])
    def test_quantiles_vs_numpy(self, random_array_100, q):
        """Test quantiles match numpy.quantile."""
        arr = random_array_100
        # Use same method as the function default
        result = compute_nunique_modes_quantiles_numpy(arr, q=q, quantile_method="median_unbiased")

        # Quantiles start after nunique, modes_min, modes_max, modes_mean, modes_qty
        quantiles = result[5:5 + len(q)]
        # Use same method in numpy
        expected = np.quantile(arr, q, method="median_unbiased")

        for i, (comp, exp) in enumerate(zip(quantiles, expected)):
            assert np.isclose(comp, exp, rtol=1e-6), f"Quantile {q[i]}: {comp} vs {exp}"

    def test_slope_intercept_vs_polyfit(self, random_array_100):
        """Test slope/intercept match numpy.polyfit."""
        arr = random_array_100
        mean_val = np.mean(arr)
        # Don't pass xvals, let the function generate it internally to match types
        result, _ = compute_moments_slope_mi(arr, mean_val)
        names = get_moments_slope_mi_feature_names()

        slope_idx = names.index('slope')
        intercept_idx = names.index('intercept')

        slope_computed = result[slope_idx]
        intercept_computed = result[intercept_idx]

        # numpy polyfit returns [slope, intercept] for degree 1
        xvals = np.arange(len(arr), dtype=np.float64)
        slope_expected, intercept_expected = np.polyfit(xvals, arr, 1)

        assert np.isclose(slope_computed, slope_expected, rtol=1e-6), \
            f"Slope: {slope_computed} vs {slope_expected}"
        assert np.isclose(intercept_computed, intercept_expected, rtol=1e-6), \
            f"Intercept: {intercept_computed} vs {intercept_expected}"

    def test_r_value_vs_pearsonr(self, random_array_100):
        """Test correlation coefficient matches scipy.stats.pearsonr."""
        arr = random_array_100
        mean_val = np.mean(arr)
        # Don't pass xvals, let the function generate it internally
        result, _ = compute_moments_slope_mi(arr, mean_val)
        names = get_moments_slope_mi_feature_names()

        r_idx = names.index('r')
        r_computed = result[r_idx]
        xvals = np.arange(len(arr), dtype=np.float64)
        r_expected, _ = stats.pearsonr(xvals, arr)

        assert np.isclose(r_computed, r_expected, rtol=1e-6), f"R-value: {r_computed} vs {r_expected}"


# =============================================================================
# REGRESSION TESTS: WEIGHTED STATISTICS
# =============================================================================

class TestWeightedStatsRegression:
    """Regression tests for weighted statistics."""

    def test_weighted_mean_vs_numpy(self, random_array_100, weights_array):
        """Test weighted mean matches numpy.average."""
        arr = random_array_100
        weights = weights_array

        result = compute_numerical_aggregates_numba(arr, weights=weights)
        names = get_basic_feature_names(weights=weights)

        wmean_idx = names.index('warimean')
        computed = result[wmean_idx]
        expected = np.average(arr, weights=weights)

        assert np.isclose(computed, expected, rtol=1e-6), \
            f"Weighted mean: {computed} vs {expected}"

    def test_weighted_std_vs_manual(self, random_array_100, weights_array):
        """Test weighted std matches manual calculation."""
        arr = random_array_100
        weights = weights_array
        mean_val = np.mean(arr)
        wmean_val = np.average(arr, weights=weights)

        result, _ = compute_moments_slope_mi(arr, mean_val, weights=weights,
                                             weighted_mean_value=wmean_val)
        names = get_moments_slope_mi_feature_names(weights=weights)

        wstd_idx = names.index('wstd')
        computed = result[wstd_idx]

        # Manual weighted std
        expected = np.sqrt(np.average((arr - wmean_val) ** 2, weights=weights))

        assert np.isclose(computed, expected, rtol=1e-6), \
            f"Weighted std: {computed} vs {expected}"


# =============================================================================
# PARAMETER COVERAGE TESTS
# =============================================================================

class TestParameterCoverage:
    """Test all parameter combinations produce consistent output."""

    @pytest.mark.parametrize("directional_only", [True, False])
    def test_directional_only_output_length(self, random_array_100, directional_only):
        """Test directional_only parameter affects output length correctly."""
        arr = random_array_100.astype(np.float32)
        result = compute_numaggs(arr, directional_only=directional_only)
        names = get_numaggs_names(directional_only=directional_only)
        assert len(result) == len(names)

    @pytest.mark.parametrize("return_entropy", [True, False])
    def test_return_entropy_output_length(self, random_array_100, return_entropy):
        """Test return_entropy parameter affects output length correctly."""
        arr = random_array_100.astype(np.float32)
        result = compute_numaggs(arr, return_entropy=return_entropy)
        names = get_numaggs_names(return_entropy=return_entropy)
        assert len(result) == len(names)

    @pytest.mark.parametrize("return_hurst", [True, False])
    def test_return_hurst_output_length(self, random_array_100, return_hurst):
        """Test return_hurst parameter affects output length correctly."""
        arr = random_array_100.astype(np.float32)
        result = compute_numaggs(arr, return_hurst=return_hurst)
        names = get_numaggs_names(return_hurst=return_hurst)
        assert len(result) == len(names)

    @pytest.mark.parametrize("return_profit_factor", [True, False])
    def test_return_profit_factor_output_length(self, random_array_100, return_profit_factor):
        """Test return_profit_factor parameter."""
        arr = random_array_100.astype(np.float32)
        result = compute_numaggs(arr, return_profit_factor=return_profit_factor)
        names = get_numaggs_names(return_profit_factor=return_profit_factor)
        assert len(result) == len(names)

    @pytest.mark.parametrize("return_exotic_means", [True, False])
    def test_return_exotic_means_output_length(self, random_array_100, return_exotic_means):
        """Test return_exotic_means parameter."""
        arr = random_array_100.astype(np.float32)
        result = compute_numaggs(arr, return_exotic_means=return_exotic_means)
        names = get_numaggs_names(return_exotic_means=return_exotic_means)
        assert len(result) == len(names)

    @pytest.mark.parametrize("return_unsorted_stats", [True, False])
    def test_return_unsorted_stats_output_length(self, random_array_100, return_unsorted_stats):
        """Test return_unsorted_stats parameter."""
        arr = random_array_100.astype(np.float32)
        result = compute_numaggs(arr, return_unsorted_stats=return_unsorted_stats)
        names = get_numaggs_names(return_unsorted_stats=return_unsorted_stats)
        assert len(result) == len(names)

    @pytest.mark.parametrize("return_n_zer_pos_int", [True, False])
    def test_return_n_zer_pos_int_output_length(self, random_array_100, return_n_zer_pos_int):
        """Test return_n_zer_pos_int parameter."""
        arr = random_array_100.astype(np.float32)
        result = compute_numaggs(arr, return_n_zer_pos_int=return_n_zer_pos_int)
        names = get_numaggs_names(return_n_zer_pos_int=return_n_zer_pos_int)
        assert len(result) == len(names)

    def test_return_drawdown_stats_output_length(self, random_array_100):
        """Test return_drawdown_stats parameter."""
        arr = random_array_100.astype(np.float32)
        result = compute_numaggs(arr, return_drawdown_stats=True)
        names = get_numaggs_names(return_drawdown_stats=True)
        assert len(result) == len(names)

    def test_return_distributional_output_length(self, positive_array):
        """Test return_distributional parameter."""
        arr = positive_array.astype(np.float32)
        result = compute_numaggs(arr, return_distributional=True)
        names = get_numaggs_names(return_distributional=True)
        assert len(result) == len(names)


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks comparing numba to numpy/scipy."""

    @pytest.mark.benchmark(group="simple_stats")
    def test_benchmark_simple_stats_numba(self, benchmark, random_array_1000):
        """Benchmark numba simple stats."""
        arr = random_array_1000
        benchmark(compute_simple_stats_numba, arr)

    @pytest.mark.benchmark(group="simple_stats")
    def test_benchmark_simple_stats_numpy(self, benchmark, random_array_1000):
        """Benchmark numpy equivalent of simple stats."""
        arr = random_array_1000

        def numpy_simple_stats(arr):
            return (np.min(arr), np.max(arr), np.argmin(arr), np.argmax(arr),
                    np.mean(arr), np.std(arr))

        benchmark(numpy_simple_stats, arr)

    @pytest.mark.benchmark(group="moments")
    def test_benchmark_moments_numba(self, benchmark, random_array_1000):
        """Benchmark numba moments computation."""
        arr = random_array_1000
        mean_val = np.mean(arr)
        benchmark(compute_moments_slope_mi, arr, mean_val)

    @pytest.mark.benchmark(group="moments")
    def test_benchmark_moments_scipy(self, benchmark, random_array_1000):
        """Benchmark scipy moments computation."""
        arr = random_array_1000

        def scipy_moments(arr):
            mean_val = np.mean(arr)
            return (np.mean(np.abs(arr - mean_val)),
                    np.std(arr),
                    stats.skew(arr),
                    stats.kurtosis(arr))

        benchmark(scipy_moments, arr)

    @pytest.mark.benchmark(group="rolling_ma")
    def test_benchmark_rolling_ma_numba(self, benchmark, random_array_1000):
        """Benchmark numba rolling moving average."""
        arr = random_array_1000
        benchmark(rolling_moving_average, arr, 10)

    @pytest.mark.benchmark(group="rolling_ma")
    def test_benchmark_rolling_ma_numpy(self, benchmark, random_array_1000):
        """Benchmark numpy convolve for moving average."""
        arr = random_array_1000

        def numpy_rolling_ma(arr, n):
            return np.convolve(arr, np.ones(n) / n, mode='valid')

        benchmark(numpy_rolling_ma, arr, 10)

    @pytest.mark.benchmark(group="crossings")
    def test_benchmark_crossings_numba(self, benchmark, random_array_1000):
        """Benchmark numba crossings computation."""
        arr = random_array_1000
        marks = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        benchmark(compute_ncrossings, arr, marks)

    @pytest.mark.benchmark(group="crossings")
    def test_benchmark_crossings_numpy(self, benchmark, random_array_1000):
        """Benchmark numpy equivalent crossings."""
        arr = random_array_1000
        marks = [0.0, 0.5, 1.0]

        def numpy_crossings(arr, marks):
            results = []
            for mark in marks:
                diff = arr - mark
                crossings = np.sum(np.diff(np.sign(diff)) != 0)
                results.append(crossings)
            return results

        benchmark(numpy_crossings, arr, marks)

    @pytest.mark.benchmark(group="full_numaggs")
    def test_benchmark_full_numaggs(self, benchmark, random_array_1000):
        """Benchmark full compute_numaggs."""
        arr = random_array_1000.astype(np.float32)
        benchmark(compute_numaggs, arr)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_element_array(self):
        """Test with single element array."""
        arr = np.array([1.0], dtype=np.float32)
        result = compute_numaggs(arr)
        # Should return NaNs for most statistics
        assert all(np.isnan(v) for v in result)

    def test_two_element_array(self):
        """Test with two element array."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = compute_numaggs(arr)
        names = get_numaggs_names()
        assert len(result) == len(names)

    def test_all_zeros(self):
        """Test with all-zero array."""
        arr = np.zeros(100, dtype=np.float32)
        result = compute_numaggs(arr, return_entropy=False)  # Entropy may fail on constant
        names = get_numaggs_names(return_entropy=False)
        assert len(result) == len(names)

    def test_all_same_value(self):
        """Test with constant non-zero array."""
        arr = np.ones(100, dtype=np.float32) * 5.0
        result = compute_numaggs(arr, return_entropy=False)
        names = get_numaggs_names(return_entropy=False)
        assert len(result) == len(names)

    def test_array_with_nans(self):
        """Test proper NaN handling in mixed array."""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        min_val, max_val, argmin, argmax, mean_val, std_val = compute_simple_stats_numba(arr)

        # Should only consider finite values: [1.0, 3.0, 5.0]
        assert min_val == 1.0
        assert max_val == 5.0
        assert np.isclose(mean_val, 3.0)

    def test_all_nan_array(self):
        """Test with all NaN array."""
        arr = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        min_val, max_val, argmin, argmax, mean_val, std_val = compute_simple_stats_numba(arr)
        # Should handle gracefully
        assert isinstance(mean_val, float)

    def test_mixed_finite_infinite(self):
        """Test with mixed finite and infinite values."""
        arr = np.array([1.0, np.nan, 3.0, np.inf, 2.0], dtype=np.float64)
        min_val, max_val, argmin, argmax, mean_val, std_val = compute_simple_stats_numba(arr)

        # Should only consider finite values: [1.0, 3.0, 2.0]
        assert min_val == 1.0
        assert max_val == 3.0

    def test_negative_values_geometric_mean(self, random_array_100):
        """Test geometric mean handling with negative values."""
        # Geometric mean is typically undefined for negative values
        arr = random_array_100  # Contains negative values
        result = compute_numerical_aggregates_numba(arr, return_exotic_means=True)
        names = get_basic_feature_names(return_exotic_means=True)

        # Should still return a result (may be NaN for geometric mean)
        assert len(result) == len(names)

    def test_very_large_values(self):
        """Test with very large values for numerical stability."""
        arr = np.array([1e100, 2e100, 3e100], dtype=np.float64)
        result = compute_numerical_aggregates_numba(arr, geomean_log_mode=True)

        # Should handle without overflow
        assert np.all(np.isfinite(result) | np.isnan(result))

    def test_very_small_values(self):
        """Test with very small values for numerical stability."""
        arr = np.array([1e-100, 2e-100, 3e-100], dtype=np.float64)
        result = compute_numerical_aggregates_numba(arr)

        # Should handle without underflow issues
        assert len(result) > 0


# =============================================================================
# FUNCTIONAL CORRECTNESS TESTS
# =============================================================================

class TestFunctionalCorrectness:
    """Test functional correctness with known signals."""

    def test_ncrossings_known_signal(self):
        """Test crossing count with known signal."""
        # Signal that crosses 0 exactly 4 times
        arr = np.array([-1, 1, -1, 1, -1], dtype=np.float64)
        marks = np.array([0.0], dtype=np.float64)
        result = compute_ncrossings(arr, marks)
        assert result[0] == 4

    def test_nunique_known_values(self):
        """Test nunique with known values."""
        arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=np.float64)
        result = compute_nunique_modes_quantiles_numpy(arr)
        nuniques = result[0]
        assert nuniques == 4  # 4 unique values

    def test_mode_known_distribution(self):
        """Test mode detection with known distribution."""
        arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=np.float64)
        result = compute_nunique_modes_quantiles_numpy(arr)
        # modes_max should be 4 (most frequent)
        modes_max = result[2]
        assert modes_max == 4.0

    def test_profit_factor_known_trades(self):
        """Test profit factor with known gains/losses."""
        # Gains: 10, 20 = 30
        # Losses: -5, -10 = -15
        # Profit factor = 30 / 15 = 2.0
        arr = np.array([10.0, -5.0, 20.0, -10.0], dtype=np.float64)
        result = compute_numerical_aggregates_numba(arr, return_profit_factor=True)
        names = get_basic_feature_names(return_profit_factor=True)

        pf_idx = names.index('profit_factor')
        computed = result[pf_idx]
        assert np.isclose(computed, 2.0, rtol=1e-6)

    def test_rolling_ma_basic_values(self):
        """Test rolling MA produces correct values."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = rolling_moving_average(arr, 3)
        expected = np.array([2.0, 3.0, 4.0])  # (1+2+3)/3, (2+3+4)/3, (3+4+5)/3
        assert np.allclose(result, expected)

    def test_rolling_ma_window_too_large(self):
        """Test that window > array length raises error."""
        arr = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError):
            rolling_moving_average(arr, 5)


# =============================================================================
# HYPOTHESIS PROPERTY-BASED TESTS
# =============================================================================

class TestHypothesisProperties:
    """Property-based tests using Hypothesis."""

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                              min_value=-1e6, max_value=1e6),
                    min_size=2, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_compute_numaggs_output_length(self, arr):
        """Test that numaggs returns consistent length."""
        arr = np.array(arr, dtype=np.float32)
        result = compute_numaggs(arr)
        expected_names = get_numaggs_names()
        assert len(result) == len(expected_names)

    @given(st.lists(st.floats(min_value=1, max_value=1000,
                              allow_nan=False, allow_infinity=False),
                    min_size=5, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_simple_stats_min_max_correct(self, arr):
        """Test min/max calculation."""
        arr = np.array(arr, dtype=np.float64)
        min_val, max_val, argmin, argmax, mean_val, std_val = compute_simple_stats_numba(arr)
        assert np.isclose(min_val, arr.min())
        assert np.isclose(max_val, arr.max())

    @given(st.lists(st.integers(min_value=-100, max_value=100),
                    min_size=10, max_size=100))
    @settings(max_examples=30, deadline=None)
    def test_quantiles_ordered(self, values):
        """Test that quantiles are properly ordered."""
        arr = np.array(values, dtype=np.float64)
        result = compute_nunique_modes_quantiles_numpy(arr)
        quantiles = result[5:10]  # Default 5 quantiles
        for i in range(len(quantiles) - 1):
            assert quantiles[i] <= quantiles[i + 1]

    @given(st.integers(min_value=2, max_value=50))
    @settings(deadline=None)
    def test_rolling_moving_average_length(self, n):
        """Test rolling MA output length."""
        arr = np.random.rand(100).astype(np.float64)
        result = rolling_moving_average(arr, n)
        expected_len = len(arr) - n + 1
        assert len(result) == expected_len

    @given(st.lists(st.floats(min_value=0.01, max_value=0.99),
                    min_size=1, max_size=5, unique=True))
    @settings(max_examples=30, deadline=None)
    def test_numaggs_custom_quantiles(self, quantiles):
        """Test with custom quantile values."""
        quantiles = sorted(quantiles)
        arr = np.random.randn(50).astype(np.float32)
        result = compute_numaggs(arr, q=quantiles)
        names = get_numaggs_names(q=quantiles)
        assert len(result) == len(names)

    @given(st.booleans())
    @settings(deadline=None)
    def test_numaggs_directional_only(self, directional_only):
        """Test directional_only parameter."""
        arr = np.random.randn(50).astype(np.float32)
        result = compute_numaggs(arr, directional_only=directional_only)
        names = get_numaggs_names(directional_only=directional_only)
        assert len(result) == len(names)

    @given(st.booleans())
    @settings(deadline=None)
    def test_numaggs_return_entropy(self, return_entropy):
        """Test return_entropy parameter."""
        arr = np.random.randn(50).astype(np.float32)
        result = compute_numaggs(arr, return_entropy=return_entropy)
        names = get_numaggs_names(return_entropy=return_entropy)
        assert len(result) == len(names)

    @given(st.booleans())
    @settings(deadline=None)
    def test_numaggs_return_hurst(self, return_hurst):
        """Test return_hurst parameter."""
        arr = np.random.randn(50).astype(np.float32)
        result = compute_numaggs(arr, return_hurst=return_hurst)
        names = get_numaggs_names(return_hurst=return_hurst)
        assert len(result) == len(names)
