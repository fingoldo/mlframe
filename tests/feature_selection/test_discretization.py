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
import pandas as pd
import warnings

from hypothesis import given, settings, strategies as st, assume, HealthCheck
from hypothesis.extra.numpy import arrays

from sklearn.datasets import make_classification, make_regression

# Import the module under test
from mlframe.feature_selection.filters import (
    MRMR,
    entropy,
    categorize_dataset,
    discretize_array,
    compute_mi_from_classes,
)

class TestDiscretization:
    """Tests for discretization functions."""

    def test_discretize_array_shape_preservation(self):
        """Discretized array should have same length as input."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)

        result = discretize_array(x, n_bins=10)

        assert len(result) == len(x)

    def test_discretize_array_bin_count(self):
        """Discretized values should be in valid bin range."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        n_bins = 10

        result = discretize_array(x, n_bins=n_bins)

        assert result.min() >= 0
        assert result.max() < n_bins

    @given(arrays(dtype=np.float64, shape=st.integers(100, 500),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_discretize_array_properties(self, x):
        """Property-based test for discretization."""
        n_bins = 10
        result = discretize_array(x, n_bins=n_bins)

        assert len(result) == len(x)
        assert result.min() >= 0
        assert result.max() < n_bins

    def test_categorize_dataset_shape(self):
        """Test categorize_dataset preserves shape."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((100, 5)), columns=['a', 'b', 'c', 'd', 'e'])

        result, nbins_arr, categorical_vars = categorize_dataset(X, n_bins=10)

        assert result.shape == X.shape
        assert len(nbins_arr) == X.shape[1]


# ================================================================================================
# MRMR Basic Functionality Tests
# ================================================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
