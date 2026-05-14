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

class TestMRMRParameters:
    """Test MRMR parameter variations."""

    @pytest.mark.parametrize("nbins", [5, 10, 20])
    def test_quantization_nbins(self, simple_classification_data, nbins):
        """Test different quantization_nbins values."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            quantization_nbins=nbins,
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)
        assert hasattr(mrmr, 'n_features_')

    @pytest.mark.parametrize("method", ["quantile", "uniform"])
    def test_quantization_method(self, simple_classification_data, method):
        """Test different quantization methods."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            quantization_method=method,
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)
        assert hasattr(mrmr, 'n_features_')

    @pytest.mark.parametrize("algo", ["fleuret", "pld"])
    def test_mrmr_relevance_algo(self, simple_classification_data, algo):
        """Test different mRMR relevance algorithms."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            mrmr_relevance_algo=algo,
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)
        assert hasattr(mrmr, 'n_features_')

    @pytest.mark.parametrize("min_gain", [0.0001, 0.001, 0.01])
    def test_min_relevance_gain(self, simple_classification_data, min_gain):
        """Test different min_relevance_gain values."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            min_relevance_gain=min_gain,
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)
        assert hasattr(mrmr, 'n_features_')

    def test_verbose_levels(self, simple_classification_data):
        """Test different verbosity levels."""
        X, y, _ = simple_classification_data

        for verbose in [0, 1, 2]:
            mrmr = MRMR(
                full_npermutations=2,
                baseline_npermutations=2,
                verbose=verbose,
                n_jobs=1
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mrmr.fit(X, y)

            assert hasattr(mrmr, 'n_features_')


# ================================================================================================
# Integration Tests
# ================================================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
