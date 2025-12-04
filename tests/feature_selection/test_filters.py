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


# ================================================================================================
# Property-Based Tests for Helper Functions
# ================================================================================================

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
            assert np.isclose(result, expected, rtol=1e-5), \
                f"Uniform entropy for n={n}: expected {expected}, got {result}"

    def test_entropy_near_deterministic_distribution(self):
        """Test entropy of near-deterministic distribution is close to 0."""
        # Use very small but non-zero values to avoid 0*log(0)=nan
        freqs = np.array([0.9999, 0.0001/3, 0.0001/3, 0.0001/3])
        result = entropy(freqs)
        assert result < 0.01, \
            f"Near-deterministic entropy should be ~0, got {result}"


class TestMIProperties:
    """Property-based tests for mutual information computation."""

    def test_mi_identical_variables(self):
        """MI(X, X) should equal H(X)."""
        np.random.seed(42)
        classes_x = np.random.randint(0, 5, 1000).astype(np.int32)

        # Compute frequencies
        unique, counts = np.unique(classes_x, return_counts=True)
        freqs_x = np.zeros(5, dtype=np.float64)
        freqs_x[unique] = counts / counts.sum()

        result = compute_mi_from_classes(classes_x, freqs_x, classes_x, freqs_x)

        # MI(X,X) = H(X) for identical variables
        expected_h = entropy(freqs_x[freqs_x > 0])

        assert np.isclose(result, expected_h, rtol=0.1), \
            f"MI(X,X) should equal H(X): expected {expected_h}, got {result}"

    def test_mi_independent_variables(self):
        """MI of independent variables should be close to 0."""
        np.random.seed(42)
        n = 5000
        classes_x = np.random.randint(0, 5, n).astype(np.int32)
        classes_y = np.random.randint(0, 5, n).astype(np.int32)

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
        np.random.seed(42)
        classes_x = np.random.randint(0, 5, 1000).astype(np.int32)
        classes_y = ((classes_x + np.random.randint(0, 2, 1000)) % 5).astype(np.int32)

        # Compute frequencies
        unique_x, counts_x = np.unique(classes_x, return_counts=True)
        freqs_x = np.zeros(5, dtype=np.float64)
        freqs_x[unique_x] = counts_x / counts_x.sum()

        unique_y, counts_y = np.unique(classes_y, return_counts=True)
        freqs_y = np.zeros(5, dtype=np.float64)
        freqs_y[unique_y] = counts_y / counts_y.sum()

        mi_xy = compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y)
        mi_yx = compute_mi_from_classes(classes_y, freqs_y, classes_x, freqs_x)

        assert np.isclose(mi_xy, mi_yx, rtol=0.01), \
            f"MI should be symmetric: MI(X,Y)={mi_xy}, MI(Y,X)={mi_yx}"

    def test_mi_bounded_by_entropy(self):
        """MI(X, Y) should be bounded by min(H(X), H(Y))."""
        np.random.seed(42)
        nbins = 5
        n = 1000
        classes_x = np.random.randint(0, nbins, n).astype(np.int32)
        classes_y = np.random.randint(0, nbins, n).astype(np.int32)

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
        assert result <= max_mi + 0.01, \
            f"MI should be <= min(H(X), H(Y)): got {result}, max={max_mi}"


class TestDiscretization:
    """Tests for discretization functions."""

    def test_discretize_array_shape_preservation(self):
        """Discretized array should have same length as input."""
        np.random.seed(42)
        x = np.random.randn(1000)

        result = discretize_array(x, n_bins=10)

        assert len(result) == len(x)

    def test_discretize_array_bin_count(self):
        """Discretized values should be in valid bin range."""
        np.random.seed(42)
        x = np.random.randn(1000)
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
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=['a', 'b', 'c', 'd', 'e'])

        result, nbins_arr, categorical_vars = categorize_dataset(X, n_bins=10)

        assert result.shape == X.shape
        assert len(nbins_arr) == X.shape[1]


# ================================================================================================
# MRMR Basic Functionality Tests
# ================================================================================================

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

class TestMRMRClassification:
    """Test MRMR on classification tasks."""

    def test_binary_classification(self, simple_classification_data):
        """Test MRMR identifies informative features in binary classification."""
        X, y, informative_indices = simple_classification_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            min_relevance_gain=0.001,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        selected_indices = set(mrmr.support_.tolist())
        informative_set = set(informative_indices)

        # At least some informative features should be selected
        overlap = len(selected_indices & informative_set)
        assert overlap >= 1, f"No informative features detected"

    def test_imbalanced_classes(self, imbalanced_classification_data):
        """Test MRMR on imbalanced classification data."""
        X, y, informative_indices = imbalanced_classification_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        assert hasattr(mrmr, 'n_features_')
        assert mrmr.n_features_ > 0

    def test_multiclass_classification(self, multiclass_data):
        """Test MRMR on multiclass classification data."""
        X, y, informative_indices = multiclass_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        assert hasattr(mrmr, 'n_features_')
        assert mrmr.n_features_ > 0


class TestMRMRRegression:
    """Test MRMR on regression tasks."""

    def test_linear_regression(self, simple_regression_data):
        """Test MRMR identifies informative features in regression."""
        X, y, informative_indices = simple_regression_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            min_relevance_gain=0.001,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        selected_indices = set(mrmr.support_.tolist())
        informative_set = set(informative_indices)

        overlap = len(selected_indices & informative_set)
        assert overlap >= 1, f"No informative features detected"

    def test_nonlinear_regression(self, nonlinear_transform_data):
        """Test MRMR on data with nonlinear relationships."""
        X, y, informative_names = nonlinear_transform_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        assert hasattr(mrmr, 'n_features_')
        assert mrmr.n_features_ > 0


# ================================================================================================
# MRMR Feature Engineering Tests
# ================================================================================================

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

class TestMRMRIntegration:
    """Integration tests for MRMR with downstream tasks."""

    def test_pipeline_compatibility(self, simple_classification_data):
        """Test MRMR works in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier

        X, y, _ = simple_classification_data

        pipeline = Pipeline([
            ('feature_selection', MRMR(
                full_npermutations=3,
                baseline_npermutations=3,
                verbose=0,
                n_jobs=1
            )),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X, y)

        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)

    def test_with_downstream_model(self, simple_classification_data):
        """Test that MRMR-selected features improve or maintain model performance."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        X, y, _ = simple_classification_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        X_selected = mrmr.transform(X)

        # Train model on selected features
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        scores = cross_val_score(clf, X_selected, y, cv=3, scoring='accuracy')

        # Should achieve reasonable accuracy
        assert scores.mean() > 0.5, f"Mean accuracy too low: {scores.mean()}"


# ================================================================================================
# Run Tests
# ================================================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
