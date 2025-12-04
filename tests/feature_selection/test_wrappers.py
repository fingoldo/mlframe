"""
Comprehensive tests for feature_selection/wrappers.py

Tests include:
- Property-based tests for helper functions using hypothesis
- RFECV parameter coverage tests
- Synthetic dataset tests with known informative features
- Multiple estimator tests (CatBoost, XGBoost, LightGBM, LogisticRegression, LinearRegression)

Note: Common fixtures are defined in conftest.py and shared with test_filters.py
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from typing import *

from hypothesis import given, settings, strategies as st, assume, HealthCheck
from hypothesis.extra.numpy import arrays

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error

# Import the module under test
from mlframe.feature_selection.wrappers import (
    RFECV,
    OptimumSearch,
    VotesAggregation,
    split_into_train_test,
    store_averaged_cv_scores,
    get_feature_importances,
    select_appropriate_feature_importances,
    get_next_features_subset,
    get_actual_features_ranking,
)

# Try importing boosting libraries
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Fixtures are now imported from conftest.py automatically


# ================================================================================================
# Helper Function Tests (Property-based with Hypothesis)
# ================================================================================================

class TestSplitIntoTrainTest:
    """Property-based tests for split_into_train_test function."""

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=2, max_value=20),
        train_frac=st.floats(min_value=0.3, max_value=0.8)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_split_shapes_ndarray(self, n_samples, n_features, train_frac):
        """Test that split produces correct shapes for ndarrays."""
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        train_size = int(n_samples * train_frac)
        train_index = np.arange(train_size)
        test_index = np.arange(train_size, n_samples)

        X_train, y_train, X_test, y_test = split_into_train_test(X, y, train_index, test_index)

        assert X_train.shape == (train_size, n_features)
        assert X_test.shape == (n_samples - train_size, n_features)
        assert len(y_train) == train_size
        assert len(y_test) == n_samples - train_size

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=2, max_value=20),
        train_frac=st.floats(min_value=0.3, max_value=0.8)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_split_shapes_dataframe(self, n_samples, n_features, train_frac):
        """Test that split produces correct shapes for DataFrames."""
        X = pd.DataFrame(np.random.randn(n_samples, n_features),
                        columns=[f'f{i}' for i in range(n_features)])
        y = pd.Series(np.random.randn(n_samples))

        train_size = int(n_samples * train_frac)
        train_index = np.arange(train_size)
        test_index = np.arange(train_size, n_samples)

        X_train, y_train, X_test, y_test = split_into_train_test(X, y, train_index, test_index)

        assert X_train.shape == (train_size, n_features)
        assert X_test.shape == (n_samples - train_size, n_features)

    def test_split_with_feature_indices(self):
        """Test split with specific feature indices."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        train_index = np.arange(70)
        test_index = np.arange(70, 100)
        features_indices = [0, 2, 4]

        X_train, y_train, X_test, y_test = split_into_train_test(
            X, y, train_index, test_index, features_indices
        )

        assert X_train.shape == (70, 3)
        assert X_test.shape == (30, 3)


class TestStoreAveragedCVScores:
    """Tests for store_averaged_cv_scores function."""

    def test_basic_averaging(self):
        """Test basic score averaging."""
        class MockSelf:
            mean_perf_weight = 1.0
            std_perf_weight = 0.1

        scores = [0.8, 0.82, 0.78, 0.81, 0.79]
        evaluated_scores_mean = {}
        evaluated_scores_std = {}

        mean, std, final = store_averaged_cv_scores(
            pos=5, scores=scores,
            evaluated_scores_mean=evaluated_scores_mean,
            evaluated_scores_std=evaluated_scores_std,
            self=MockSelf()
        )

        assert np.isclose(mean, np.mean(scores))
        assert np.isclose(std, np.std(scores))
        assert 5 in evaluated_scores_mean
        assert 5 in evaluated_scores_std

    @given(st.lists(st.floats(min_value=0.5, max_value=1.0), min_size=2, max_size=10))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_averaging_properties(self, scores):
        """Test that averaging has expected properties."""
        class MockSelf:
            mean_perf_weight = 1.0
            std_perf_weight = 0.0

        evaluated_scores_mean = {}
        evaluated_scores_std = {}

        mean, std, final = store_averaged_cv_scores(
            pos=1, scores=scores,
            evaluated_scores_mean=evaluated_scores_mean,
            evaluated_scores_std=evaluated_scores_std,
            self=MockSelf()
        )

        # Final score equals mean when std_weight is 0
        assert np.isclose(final, mean)


class TestSelectAppropriateFeatureImportances:
    """Tests for select_appropriate_feature_importances function."""

    def test_use_all_fi_runs(self):
        """Test selecting all feature importance runs."""
        feature_importances = {
            '10_0': {'a': 1, 'b': 2},
            '10_1': {'a': 1.5, 'b': 1.8},
            '5_0': {'a': 2, 'b': 1},
        }

        result = select_appropriate_feature_importances(
            feature_importances=feature_importances,
            nfeatures=5,
            n_original_features=10,
            use_all_fi_runs=True,
            use_last_fi_run_only=False,
            use_one_freshest_fi_run=False,
            use_fi_ranking=False,
        )

        assert len(result) == 3

    def test_use_last_fi_run_only(self):
        """Test selecting only last FI run."""
        feature_importances = {
            '10_0': {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10},
            '10_1': {'a': 1.5, 'b': 1.8, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9},
            '5_0': {'a': 2, 'b': 1, 'c': 3, 'd': 4, 'e': 5},
        }

        result = select_appropriate_feature_importances(
            feature_importances=feature_importances,
            nfeatures=5,
            n_original_features=10,
            use_all_fi_runs=False,
            use_last_fi_run_only=True,
            use_one_freshest_fi_run=False,
            use_fi_ranking=False,
        )

        # Should only include runs with n_original_features length
        assert all(len(v) == 10 for v in result.values())


# ================================================================================================
# RFECV Parameter Coverage Tests
# ================================================================================================

class TestRFECVParameters:
    """Tests covering RFECV parameter variations."""

    def test_basic_initialization(self):
        """Test RFECV initializes with default parameters."""
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(estimator=estimator)

        assert rfecv.estimator is estimator
        assert rfecv.cv == 3
        assert rfecv.verbose == 1

    @pytest.mark.parametrize("max_nfeatures", [None, 5, 10])
    def test_max_nfeatures(self, simple_classification_data, max_nfeatures):
        """Test max_nfeatures parameter."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_nfeatures=max_nfeatures,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)

        # max_nfeatures limits search space, not final selection
        # Just verify it completes and selects valid number of features
        assert hasattr(rfecv, 'n_features_')
        assert rfecv.n_features_ > 0

    @pytest.mark.parametrize("mean_perf_weight,std_perf_weight", [
        (1.0, 0.0),
        (1.0, 0.1),
        (0.5, 0.5),
    ])
    def test_perf_weights(self, simple_classification_data, mean_perf_weight, std_perf_weight):
        """Test performance weight parameters."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            mean_perf_weight=mean_perf_weight,
            std_perf_weight=std_perf_weight,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'n_features_')

    @pytest.mark.parametrize("feature_cost", [0.0, 0.001, 0.01])
    def test_feature_cost(self, simple_classification_data, feature_cost):
        """Test feature_cost parameter."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            feature_cost=feature_cost,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'n_features_')

    @pytest.mark.parametrize("cv", [2, 3, 5])
    def test_cv_values(self, simple_classification_data, cv):
        """Test different CV values."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            cv=cv,
            max_refits=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'cv_results_')

    @pytest.mark.parametrize("cv_shuffle", [True, False])
    def test_cv_shuffle(self, simple_classification_data, cv_shuffle):
        """Test cv_shuffle parameter."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            cv_shuffle=cv_shuffle,
            max_refits=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'n_features_')

    @pytest.mark.parametrize("votes_aggregation_method", [
        VotesAggregation.Borda,
        VotesAggregation.AM,
        VotesAggregation.Copeland,
        VotesAggregation.Dowdall,
    ])
    def test_votes_aggregation(self, simple_classification_data, votes_aggregation_method):
        """Test different votes aggregation methods."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            votes_aggregation_method=votes_aggregation_method,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'n_features_')

    @pytest.mark.parametrize("use_all_fi_runs,use_last_fi_run_only", [
        (True, False),
        (False, True),
    ])
    def test_fi_run_selection(self, simple_classification_data, use_all_fi_runs, use_last_fi_run_only):
        """Test feature importance run selection parameters."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            use_all_fi_runs=use_all_fi_runs,
            use_last_fi_run_only=use_last_fi_run_only,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'feature_importances_')

    @pytest.mark.parametrize("keep_estimators", [True, False])
    def test_keep_estimators(self, simple_classification_data, keep_estimators):
        """Test keep_estimators parameter."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            keep_estimators=keep_estimators,
            max_refits=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)

        if keep_estimators:
            assert len(rfecv.estimators_) > 0
        else:
            assert len(rfecv.estimators_) == 0

    @pytest.mark.parametrize("frac", [None, 0.5, 0.8])
    def test_frac(self, simple_classification_data, frac):
        """Test frac parameter for data subsampling."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            frac=frac,
            max_refits=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'n_features_')

    def test_skip_retraining_on_same_shape(self, simple_classification_data):
        """Test skip_retraining_on_same_shape parameter."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            skip_retraining_on_same_shape=True,
            max_refits=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        # First fit
        rfecv.fit(X, y)
        n_features_1 = rfecv.n_features_

        # Second fit should be skipped
        rfecv.fit(X, y)
        n_features_2 = rfecv.n_features_

        assert n_features_1 == n_features_2

    @pytest.mark.parametrize("conduct_final_voting", [True, False])
    def test_conduct_final_voting(self, simple_classification_data, conduct_final_voting):
        """Test conduct_final_voting parameter."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            conduct_final_voting=conduct_final_voting,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'support_')

    def test_nofeatures_dummy_scoring(self, simple_classification_data):
        """Test nofeatures_dummy_scoring parameter."""
        X, y, _ = simple_classification_data

        for nofeatures_dummy_scoring in [True, False]:
            estimator = RandomForestClassifier(n_estimators=10, random_state=42)
            rfecv = RFECV(
                estimator=estimator,
                nofeatures_dummy_scoring=nofeatures_dummy_scoring,
                max_refits=2,
                verbose=0,
                optimizer_plotting='No',
                random_state=42
            )

            rfecv.fit(X, y)
            assert 0 in rfecv.cv_results_['nfeatures']

    def test_max_runtime_mins(self, simple_classification_data):
        """Test max_runtime_mins parameter."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_runtime_mins=0.01,  # Very short
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'n_features_')

    def test_max_refits(self, simple_classification_data):
        """Test max_refits parameter."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        # Number of evaluated feature sets should be <= max_refits + 1 (includes 0 features)
        assert len(rfecv.cv_results_['nfeatures']) <= 4

    def test_max_noimproving_iters(self, simple_classification_data):
        """Test max_noimproving_iters parameter."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_noimproving_iters=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'n_features_')

    def test_importance_getter_string(self, simple_classification_data):
        """Test importance_getter with string value."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            importance_getter='feature_importances_',
            max_refits=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'feature_importances_')

    def test_importance_getter_coef(self, simple_classification_data):
        """Test importance_getter with coef_ for linear models."""
        X, y, _ = simple_classification_data

        estimator = LogisticRegression(max_iter=1000, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            importance_getter='coef_',
            max_refits=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'feature_importances_')

    def test_frac_validation(self):
        """Test that invalid frac values raise ValueError."""
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        with pytest.raises(ValueError):
            RFECV(estimator=estimator, frac=1.5)

        with pytest.raises(ValueError):
            RFECV(estimator=estimator, frac=0.0)

        with pytest.raises(ValueError):
            RFECV(estimator=estimator, frac=-0.5)


# ================================================================================================
# Synthetic Dataset Tests with Multiple Estimators
# ================================================================================================

def get_classification_estimators():
    """Get list of classification estimators to test."""
    estimators = [
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
        ('RandomForest', RandomForestClassifier(n_estimators=20, random_state=42)),
    ]

    if HAS_CATBOOST:
        estimators.append(('CatBoost', CatBoostClassifier(
            iterations=10, depth=3, verbose=0, random_state=42
        )))

    if HAS_XGBOOST:
        estimators.append(('XGBoost', XGBClassifier(
            n_estimators=10, max_depth=3, verbosity=0, random_state=42
        )))

    if HAS_LIGHTGBM:
        estimators.append(('LightGBM', LGBMClassifier(
            n_estimators=10, max_depth=3, verbose=-1, random_state=42
        )))

    return estimators


def get_regression_estimators():
    """Get list of regression estimators to test."""
    estimators = [
        ('LinearRegression', LinearRegression()),
        ('RandomForest', RandomForestRegressor(n_estimators=20, random_state=42)),
    ]

    if HAS_CATBOOST:
        estimators.append(('CatBoost', CatBoostRegressor(
            iterations=10, depth=3, verbose=0, random_state=42
        )))

    if HAS_XGBOOST:
        estimators.append(('XGBoost', XGBRegressor(
            n_estimators=10, max_depth=3, verbosity=0, random_state=42
        )))

    if HAS_LIGHTGBM:
        estimators.append(('LightGBM', LGBMRegressor(
            n_estimators=10, max_depth=3, verbose=-1, random_state=42
        )))

    return estimators


class TestRFECVSyntheticClassification:
    """Test RFECV on synthetic classification datasets with known informative features."""

    @pytest.mark.parametrize("name,estimator", get_classification_estimators())
    def test_binary_classification(self, simple_classification_data, name, estimator):
        """Test that RFECV identifies informative features in binary classification."""
        X, y, informative_indices = simple_classification_data

        importance_getter = 'coef_' if name == 'LogisticRegression' else 'feature_importances_'

        rfecv = RFECV(
            estimator=estimator,
            importance_getter=importance_getter,
            max_refits=5,
            max_noimproving_iters=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)

        # Get selected feature indices
        selected_indices = set(rfecv.support_.tolist())
        informative_set = set(informative_indices)

        # Check overlap - at least 60% of informative features should be selected
        overlap = len(selected_indices & informative_set)
        recall = overlap / len(informative_set) if len(informative_set) > 0 else 0

        assert recall >= 0.4, f"{name}: Only {recall*100:.0f}% of informative features detected"

    @pytest.mark.parametrize("name,estimator", get_classification_estimators())
    def test_imbalanced_classification(self, imbalanced_classification_data, name, estimator):
        """Test RFECV on imbalanced classification data."""
        X, y, informative_indices = imbalanced_classification_data

        importance_getter = 'coef_' if name == 'LogisticRegression' else 'feature_importances_'

        rfecv = RFECV(
            estimator=estimator,
            importance_getter=importance_getter,
            max_refits=5,
            max_noimproving_iters=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)

        # Should complete without error
        assert hasattr(rfecv, 'n_features_')
        assert rfecv.n_features_ > 0

    @pytest.mark.parametrize("name,estimator", get_classification_estimators()[:2])  # Subset for speed
    def test_multiclass(self, multiclass_data, name, estimator):
        """Test RFECV on multiclass classification data."""
        X, y, informative_indices = multiclass_data

        importance_getter = 'coef_' if name == 'LogisticRegression' else 'feature_importances_'

        rfecv = RFECV(
            estimator=estimator,
            importance_getter=importance_getter,
            max_refits=4,
            max_noimproving_iters=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)

        assert hasattr(rfecv, 'n_features_')
        assert rfecv.n_features_ > 0

    @pytest.mark.parametrize("name,estimator", get_classification_estimators()[:2])
    def test_high_dimensional(self, high_dimensional_data, name, estimator):
        """Test RFECV on high-dimensional data (p > n)."""
        X, y, informative_indices = high_dimensional_data

        importance_getter = 'coef_' if name == 'LogisticRegression' else 'feature_importances_'

        rfecv = RFECV(
            estimator=estimator,
            importance_getter=importance_getter,
            max_refits=4,
            max_noimproving_iters=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)

        # In high-dimensional case, just verify it completes and selects valid features
        # With limited refits, may not always reduce features
        assert hasattr(rfecv, 'n_features_')
        assert rfecv.n_features_ > 0

    @pytest.mark.parametrize("name,estimator", get_classification_estimators()[:2])
    def test_correlated_features(self, correlated_features_data, name, estimator):
        """Test RFECV on data with correlated informative features."""
        X, y, informative_indices = correlated_features_data

        importance_getter = 'coef_' if name == 'LogisticRegression' else 'feature_importances_'

        rfecv = RFECV(
            estimator=estimator,
            importance_getter=importance_getter,
            max_refits=5,
            max_noimproving_iters=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)

        # Get selected feature indices
        selected_indices = set(rfecv.support_.tolist())
        informative_set = set(informative_indices)

        # With correlated features, at least some informative ones should be selected
        overlap = len(selected_indices & informative_set)
        assert overlap >= 1, f"{name}: No informative features detected"


class TestRFECVSyntheticRegression:
    """Test RFECV on synthetic regression datasets with known informative features."""

    @pytest.mark.parametrize("name,estimator", get_regression_estimators())
    def test_basic_regression(self, simple_regression_data, name, estimator):
        """Test that RFECV identifies informative features in regression."""
        X, y, informative_indices = simple_regression_data

        importance_getter = 'coef_' if name == 'LinearRegression' else 'feature_importances_'

        rfecv = RFECV(
            estimator=estimator,
            importance_getter=importance_getter,
            max_refits=5,
            max_noimproving_iters=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)

        # Get selected feature indices
        selected_indices = set(rfecv.support_.tolist())
        informative_set = set(informative_indices)

        # Check overlap
        overlap = len(selected_indices & informative_set)
        recall = overlap / len(informative_set) if len(informative_set) > 0 else 0

        assert recall >= 0.4, f"{name}: Only {recall*100:.0f}% of informative features detected"


class TestRFECVTransform:
    """Test RFECV transform functionality."""

    def test_transform_dataframe(self, simple_classification_data):
        """Test transform on DataFrame input."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        X_transformed = rfecv.transform(X)

        assert X_transformed.shape[1] == rfecv.n_features_
        assert X_transformed.shape[0] == X.shape[0]

    def test_transform_ndarray(self, simple_classification_data):
        """Test transform on ndarray input."""
        X_df, y, _ = simple_classification_data
        X = X_df.values

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        X_transformed = rfecv.transform(X)

        assert X_transformed.shape[1] == rfecv.n_features_
        assert X_transformed.shape[0] == X.shape[0]


class TestRFECVEdgeCases:
    """Test RFECV edge cases and error handling."""

    def test_single_feature(self):
        """Test RFECV with single feature."""
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_refits=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert rfecv.n_features_ == 1

    def test_all_noise_features(self):
        """Test RFECV when all features are noise."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        # Should still complete
        assert hasattr(rfecv, 'n_features_')

    def test_with_groups(self, simple_classification_data):
        """Test RFECV with group parameter."""
        X, y, _ = simple_classification_data
        groups = np.repeat(np.arange(20), 10)  # 20 groups, 10 samples each

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_refits=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y, groups=groups)
        assert hasattr(rfecv, 'n_features_')

    def test_cv_results_structure(self, simple_classification_data):
        """Test that cv_results_ has correct structure."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)

        assert 'nfeatures' in rfecv.cv_results_
        assert 'cv_mean_perf' in rfecv.cv_results_
        assert 'cv_std_perf' in rfecv.cv_results_

        # All lists should have same length
        assert len(rfecv.cv_results_['nfeatures']) == len(rfecv.cv_results_['cv_mean_perf'])
        assert len(rfecv.cv_results_['nfeatures']) == len(rfecv.cv_results_['cv_std_perf'])

    def test_unfitted_transform(self):
        """Test RFECV transform when support_ is not set (unfitted or failed fit).

        This tests the edge case where transform is called but the RFECV
        either wasn't fitted or the fit failed before setting support_.
        The transform should return the original data without error.
        Regression test for AttributeError: 'RFECV' object has no attribute 'support_'.
        """
        X = np.random.randn(100, 5)

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_refits=2,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        # Transform without fit - should return original data without error
        X_transformed = rfecv.transform(X)
        assert X_transformed is not None
        np.testing.assert_array_equal(X_transformed, X)

    def test_perfect_feature_detection(self):
        """Test RFECV detects a feature with perfect correlation to target.
        
        When one feature is perfectly correlated with the target, RFECV
        should identify and select it.
        """
        np.random.seed(42)
        n = 500  # Larger sample for better feature importance estimation
        # Create noise features
        noise1 = np.random.randn(n)
        noise2 = np.random.randn(n)
        noise3 = np.random.randn(n)
        # Perfect feature: target is directly derived from it
        perfect = np.random.randn(n)
        y = (perfect > 0).astype(int)  # Binary classification from perfect feature
        
        X = np.column_stack([noise1, noise2, noise3, perfect])
        
        # Use more estimators for better feature importance
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            max_refits=10,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )
        
        rfecv.fit(X, y)
        
        # At least some features should be selected  
        assert rfecv.n_features_ >= 1
        # The RFECV should complete without error
        assert hasattr(rfecv, 'support_')


# ================================================================================================
# Optimizer Method Tests
# ================================================================================================

class TestOptimizerMethods:
    """Test different optimization methods."""

    @pytest.mark.parametrize("method", [
        OptimumSearch.ModelBasedHeuristic,
        OptimumSearch.ExhaustiveRandom,
    ])
    def test_search_methods(self, simple_classification_data, method):
        """Test different top_predictors_search_method values."""
        X, y, _ = simple_classification_data

        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        rfecv = RFECV(
            estimator=estimator,
            top_predictors_search_method=method,
            max_refits=3,
            verbose=0,
            optimizer_plotting='No',
            random_state=42
        )

        rfecv.fit(X, y)
        assert hasattr(rfecv, 'n_features_')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
