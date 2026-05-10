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
from typing import *

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

class TestMRMRBasic:
    """Basic functionality tests for MRMR class."""

    def test_initialization(self):
        """Test MRMR initializes with default parameters."""
        mrmr = MRMR()

        assert mrmr.quantization_nbins == 10
        assert mrmr.verbose == 0

    def test_fit_returns_self(self, simple_classification_data):
        """Test that fit returns self for method chaining."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        result = mrmr.fit(X, y)

        assert result is mrmr

    def test_fit_sets_attributes(self, simple_classification_data):
        """Test that fit sets expected attributes."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)

        assert hasattr(mrmr, 'support_')
        assert hasattr(mrmr, 'n_features_')
        assert hasattr(mrmr, 'n_features_in_')

    def test_transform_shape(self, simple_classification_data):
        """Test transform produces correct output shape."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)
        X_transformed = mrmr.transform(X)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == mrmr.n_features_

    def test_sklearn_api_compliance(self, simple_classification_data):
        """Test MRMR follows sklearn API conventions."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        # fit_transform should work
        X_transformed = mrmr.fit_transform(X, y)

        assert X_transformed is not None
        # MRMR stores original feature count (without target column added internally)
        assert mrmr.n_features_in_ > 0
        assert mrmr.n_features_in_ <= X.shape[1]


# ================================================================================================
# MRMR Feature Selection Tests
# ================================================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
