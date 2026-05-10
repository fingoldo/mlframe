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

class TestMRMREdgeCases:
    """Test MRMR edge cases and error handling."""

    def test_single_feature(self):
        """Test MRMR with single feature."""
        np.random.seed(42)
        X = pd.DataFrame({'a': np.random.randn(200)})
        y = (X['a'] > 0).astype(int)

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)
        assert mrmr.n_features_ == 1

    def test_all_noise_features(self):
        """Test MRMR when all features are noise."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 5), columns=['a', 'b', 'c', 'd', 'e'])
        y = np.random.randint(0, 2, 200)

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            min_relevance_gain=0.1,  # High threshold
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        # Should complete without error
        assert hasattr(mrmr, 'n_features_')

    def test_constant_feature(self):
        """Test MRMR with constant feature."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'informative': np.random.randn(n),
            'constant': np.ones(n),
        })
        y = (X['informative'] > 0).astype(int)

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        # Should handle constant feature gracefully
        assert hasattr(mrmr, 'n_features_')

    def test_highly_correlated_features(self, correlated_features_data):
        """Test MRMR with highly correlated features."""
        X, y, _ = correlated_features_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        # Should complete and select features
        assert mrmr.n_features_ > 0

    def test_ndarray_input(self, simple_classification_data):
        """Test MRMR with numpy array input instead of DataFrame."""
        X_df, y, _ = simple_classification_data
        X = X_df.values

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)
        assert hasattr(mrmr, 'n_features_')

    def test_skip_retraining_parameter_exists(self, simple_classification_data):
        """Test skip_retraining_on_same_shape parameter can be set."""
        X, y, _ = simple_classification_data

        # Just test that the parameter can be set without error
        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            skip_retraining_on_same_shape=True,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)
        assert hasattr(mrmr, 'n_features_')

    def test_no_features_selected_transform(self):
        """Test MRMR transform when no features are selected (empty selection).

        This tests the edge case where MRMR finds no useful features due to
        very strict thresholds. The transform should still work without errors.
        Regression test for IndexError with empty numpy array indexing.
        """
        np.random.seed(42)
        # Create data where features have zero correlation with target
        X = pd.DataFrame(np.random.randn(200, 5), columns=['a', 'b', 'c', 'd', 'e'])
        y = np.random.randint(0, 2, 200)

        mrmr = MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            min_relevance_gain=10.0,  # Extremely high threshold - nothing will pass
            min_nonzero_confidence=0.99,  # Very strict confidence requirement
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        # Transform should work even with empty selection
        X_transformed = mrmr.transform(X)

        # Verify it completed without IndexError
        assert X_transformed is not None
        assert hasattr(mrmr, 'n_features_')

    def test_perfect_feature_detection(self):
        """Test MRMR detects a feature with perfect correlation to target.
        
        When one feature is perfectly correlated with the target, MRMR
        should identify and select it.
        """
        np.random.seed(42)
        n = 500
        # Create noise features
        X = pd.DataFrame({
            'noise1': np.random.randn(n),
            'noise2': np.random.randn(n),
            'noise3': np.random.randn(n),
        })
        # Perfect feature: target is directly derived from it
        X['perfect'] = np.random.randn(n)
        y = (X['perfect'] > 0).astype(int)  # Binary classification from perfect feature
        
        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )
        
        mrmr.fit(X, y)
        
        # The perfect feature should be selected - check via support_ mask
        selected_features = X.columns[mrmr.support_].tolist() if hasattr(mrmr, 'support_') else []
        assert 'perfect' in selected_features, f"Perfect feature not selected. Selected: {selected_features}"
        # It should be among the top features
        assert mrmr.n_features_ >= 1


# ================================================================================================
# MRMR Parameter Coverage Tests
# ================================================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
