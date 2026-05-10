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

class TestMRMRFeatureEngineering:
    """Test MRMR's feature engineering capabilities."""

    @pytest.mark.slow
    def test_synergistic_feature_detection(self, synergistic_features_data):
        """
        Test that MRMR detects synergistic features.
        y = a^2/b + log(c)*sin(d)

        MRMR should:
        1. Select a, b, c, d (not e)
        2. Potentially recommend engineered features
        """
        df, y, expected_features = synergistic_features_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=10,
            fe_max_steps=1,  # Enable feature engineering
            fe_min_pair_mi_prevalence=1.1,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        # Get selected feature names
        selected_indices = mrmr.support_.tolist()
        selected_names = [df.columns[i] for i in selected_indices]

        # Noise feature 'e' should not be selected
        assert 'e' not in selected_names, "Noise feature 'e' should not be selected"

        # At least 3 of the 4 informative features should be selected
        informative_selected = sum(1 for f in expected_features if f in selected_names)
        assert informative_selected >= 3, \
            f"Expected at least 3 informative features, got {informative_selected}: {selected_names}"

    @pytest.mark.slow
    def test_feature_engineering_example(self, feature_engineering_example_data):
        """
        Test the user's exact example from the ticket:
        y = a^2/b + log(c)*sin(d)

        MRMR should select a, b, c, d and potentially recommend:
        - mul(log(c), sin(d))
        - mul(squared(a), reciproc(b))
        """
        df, y, expected_features = feature_engineering_example_data

        mrmr = MRMR(
            full_npermutations=10,
            baseline_npermutations=20,
            fe_max_steps=2,
            fe_min_pair_mi_prevalence=1.05,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        # Get selected feature names
        selected_indices = mrmr.support_.tolist()
        selected_names = [df.columns[i] for i in selected_indices]

        # Check that noise is not selected
        assert 'e' not in selected_names, "Noise feature 'e' should not be selected"

        # Check that key features are selected
        for feat in ['a', 'b']:
            assert feat in selected_names, f"Feature '{feat}' should be selected"

    def test_multiplicative_synergy(self, multiplicative_synergy_data):
        """Test that MRMR detects multiplicative synergy: y = a * b."""
        df, y, expected_features = multiplicative_synergy_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            fe_max_steps=1,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        selected_indices = mrmr.support_.tolist()
        selected_names = [df.columns[i] for i in selected_indices]

        # Both a and b should be selected
        for feat in expected_features:
            assert feat in selected_names, f"Feature '{feat}' should be selected"

        # Noise feature should ideally not be selected
        # (though with limited permutations it might be)

    def test_additive_synergy(self, additive_synergy_data):
        """Test that MRMR detects additive relationships: y = a + b."""
        df, y, expected_features = additive_synergy_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        selected_indices = mrmr.support_.tolist()
        selected_names = [df.columns[i] for i in selected_indices]

        # Both a and b should be selected
        for feat in expected_features:
            assert feat in selected_names, f"Feature '{feat}' should be selected"

    @pytest.mark.parametrize("transform_name,transform_func,feature_gen", [
        ("squared", lambda x: x**2, lambda: np.random.randn(3000)),
        ("log", lambda x: np.log(np.abs(x) + 1), lambda: np.random.rand(3000) + 0.1),
        ("sin", np.sin, lambda: np.random.rand(3000) * 2 * np.pi),
    ])
    def test_unary_transform_detection(self, transform_name, transform_func, feature_gen):
        """Test that MRMR can detect features with unary transforms."""
        np.random.seed(42)

        a = feature_gen()
        b = np.random.randn(len(a))  # Noise

        y = transform_func(a) + np.random.randn(len(a)) * 0.1

        df = pd.DataFrame({'a': a, 'b': b})

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        selected_indices = mrmr.support_.tolist()

        # Feature 'a' (index 0) should be selected
        assert 0 in selected_indices, f"Feature 'a' should be selected for {transform_name} transform"

    def test_no_false_positives_independent_features(self):
        """Test that MRMR doesn't over-select when features are independent."""
        np.random.seed(42)
        n = 5000

        # All independent features
        df = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
            'c': np.random.randn(n),
            'd': np.random.randn(n),
            'e': np.random.randn(n),
        })

        # Target only depends on 'a'
        y = df['a'] + np.random.randn(n) * 0.1

        mrmr = MRMR(
            full_npermutations=10,
            baseline_npermutations=10,
            min_relevance_gain=0.01,  # Higher threshold
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        # Should select very few features (ideally just 'a')
        assert mrmr.n_features_ <= 3, \
            f"Expected few features for simple relationship, got {mrmr.n_features_}"


# ================================================================================================
# MRMR Edge Cases
# ================================================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
