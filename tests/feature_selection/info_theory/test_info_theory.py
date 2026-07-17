"""
Comprehensive tests for feature_selection/filters.py

Tests include:
- Property-based tests for helper functions using hypothesis
- MRMR feature selection tests for classification and regression
- Feature engineering capability tests
- Edge cases and integration tests
"""

import pytest
import numpy as np

from hypothesis import given, settings, strategies as st, HealthCheck


# Import the module under test
from mlframe.feature_selection.filters import (
    entropy,
    compute_mi_from_classes,
)


class TestEntropyProperties:
    """Property-based tests for entropy function."""

    @given(st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=20))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_entropy_nonnegative(self, freqs):
        """Entropy should always be non-negative."""
        freqs_arr = np.array(freqs, dtype=np.float64)
        freqs_arr = freqs_arr / freqs_arr.sum()  # Normalize
        result = entropy(freqs_arr)
        assert result >= 0, f"Entropy should be >= 0, got {result}"

    def test_entropy_uniform_distribution(self):
        """Test entropy of uniform distribution equals ln(n)."""
        for n in [2, 4, 8, 16]:
            freqs = np.ones(n) / n
            result = entropy(freqs)
            expected = np.log(n)  # Natural log
            assert np.isclose(result, expected, rtol=1e-5), f"Uniform entropy for n={n}: expected {expected}, got {result}"

    def test_entropy_near_deterministic_distribution(self):
        """Test entropy of near-deterministic distribution is close to 0."""
        # Use very small but non-zero values to avoid 0*log(0)=nan
        freqs = np.array([0.9999, 0.0001 / 3, 0.0001 / 3, 0.0001 / 3])
        result = entropy(freqs)
        assert result < 0.01, f"Near-deterministic entropy should be ~0, got {result}"


class TestMIProperties:
    """Property-based tests for mutual information computation."""

    def test_mi_identical_variables(self):
        """MI(X, X) should equal H(X)."""
        rng = np.random.default_rng(42)
        classes_x = rng.integers(0, 5, 1000).astype(np.int32)

        # Compute frequencies
        unique, counts = np.unique(classes_x, return_counts=True)
        freqs_x = np.zeros(5, dtype=np.float64)
        freqs_x[unique] = counts / counts.sum()

        result = compute_mi_from_classes(classes_x, freqs_x, classes_x, freqs_x)

        # MI(X,X) = H(X) for identical variables
        expected_h = entropy(freqs_x[freqs_x > 0])

        assert np.isclose(result, expected_h, rtol=0.1), f"MI(X,X) should equal H(X): expected {expected_h}, got {result}"

    def test_mi_independent_variables(self):
        """MI of independent variables should be close to 0."""
        rng = np.random.default_rng(42)
        n = 5000
        classes_x = rng.integers(0, 5, n).astype(np.int32)
        classes_y = rng.integers(0, 5, n).astype(np.int32)

        # Compute frequencies
        unique_x, counts_x = np.unique(classes_x, return_counts=True)
        freqs_x = np.zeros(5, dtype=np.float64)
        freqs_x[unique_x] = counts_x / counts_x.sum()

        unique_y, counts_y = np.unique(classes_y, return_counts=True)
        freqs_y = np.zeros(5, dtype=np.float64)
        freqs_y[unique_y] = counts_y / counts_y.sum()

        result = compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y)

        # Should be close to 0 for independent variables
        assert result < 0.05, f"MI of independent vars should be ~0, got {result}"

    def test_mi_symmetry(self):
        """MI(X, Y) should equal MI(Y, X)."""
        rng = np.random.default_rng(42)
        classes_x = rng.integers(0, 5, 1000).astype(np.int32)
        classes_y = ((classes_x + rng.integers(0, 2, 1000)) % 5).astype(np.int32)

        # Compute frequencies
        unique_x, counts_x = np.unique(classes_x, return_counts=True)
        freqs_x = np.zeros(5, dtype=np.float64)
        freqs_x[unique_x] = counts_x / counts_x.sum()

        unique_y, counts_y = np.unique(classes_y, return_counts=True)
        freqs_y = np.zeros(5, dtype=np.float64)
        freqs_y[unique_y] = counts_y / counts_y.sum()

        mi_xy = compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y)
        mi_yx = compute_mi_from_classes(classes_y, freqs_y, classes_x, freqs_x)

        assert np.isclose(mi_xy, mi_yx, rtol=0.01), f"MI should be symmetric: MI(X,Y)={mi_xy}, MI(Y,X)={mi_yx}"

    def test_mi_bounded_by_entropy(self):
        """MI(X, Y) should be bounded by min(H(X), H(Y))."""
        rng = np.random.default_rng(42)
        nbins = 5
        n = 1000
        classes_x = rng.integers(0, nbins, n).astype(np.int32)
        classes_y = rng.integers(0, nbins, n).astype(np.int32)

        # Compute frequencies
        unique_x, counts_x = np.unique(classes_x, return_counts=True)
        freqs_x = np.zeros(nbins, dtype=np.float64)
        freqs_x[unique_x] = counts_x / counts_x.sum()

        unique_y, counts_y = np.unique(classes_y, return_counts=True)
        freqs_y = np.zeros(nbins, dtype=np.float64)
        freqs_y[unique_y] = counts_y / counts_y.sum()

        result = compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y)

        # Compute H(X) and H(Y)
        h_x = entropy(freqs_x[freqs_x > 0])
        h_y = entropy(freqs_y[freqs_y > 0])

        max_mi = min(h_x, h_y)
        assert result <= max_mi + 0.01, f"MI should be <= min(H(X), H(Y)): got {result}, max={max_mi}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
