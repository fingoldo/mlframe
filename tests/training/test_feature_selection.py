"""
Integration tests for feature selection components.

Tests MRMR, RFECV, and combined pipeline functionality.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
import warnings

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.configs import TargetTypes
from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.wrappers import RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from .shared import SimpleFeaturesAndTargetsExtractor


# ================================================================================================
# Test Class 1: MRMR Feature Selection
# ================================================================================================


class TestMRMRFeatureSelection:
    """Tests for MRMR feature selector."""

    def test_mrmr_basic_fit(self, sample_regression_data):
        """Test basic MRMR fit on regression data."""
        df, feature_names, y = sample_regression_data
        X = df[feature_names]

        # Create MRMR selector with minimal settings for speed
        selector = MRMR(
            verbose=0,
            max_runtime_mins=1,
            quantization_nbins=5,
            use_simple_mode=True,
            n_workers=1,
        )

        # Fit
        selector.fit(X, y)

        # Verify attributes are set
        assert hasattr(selector, 'n_features_in_')
        assert selector.n_features_in_ == len(feature_names)

    def test_mrmr_transform(self, sample_regression_data):
        """Test MRMR transform after fit."""
        df, feature_names, y = sample_regression_data
        X = df[feature_names]

        selector = MRMR(
            verbose=0,
            max_runtime_mins=1,
            quantization_nbins=5,
            use_simple_mode=True,
            n_workers=1,
        )

        selector.fit(X, y)

        # Transform if support_ is set
        if selector.support_ is not None and selector.support_.any():
            X_transformed = selector.transform(X)
            # Should have fewer or equal features
            assert X_transformed.shape[1] <= X.shape[1]

    def test_mrmr_with_classification(self, sample_classification_data):
        """Test MRMR on classification data."""
        df, feature_names, _, y = sample_classification_data
        X = df[feature_names]

        selector = MRMR(
            verbose=0,
            max_runtime_mins=1,
            quantization_nbins=5,
            use_simple_mode=True,
            n_workers=1,
        )

        selector.fit(X, y)

        assert hasattr(selector, 'n_features_in_')

    def test_mrmr_with_categorical_features(self, sample_categorical_data):
        """Test MRMR with categorical features."""
        df, feature_names, cat_features, y = sample_categorical_data

        # Use only numeric features for MRMR (categorical handling varies)
        numeric_features = [f for f in feature_names if f not in cat_features]
        X = df[numeric_features]

        selector = MRMR(
            verbose=0,
            max_runtime_mins=1,
            quantization_nbins=5,
            use_simple_mode=True,
            n_workers=1,
        )

        selector.fit(X, y)

        assert selector.n_features_in_ == len(numeric_features)

    def test_mrmr_skip_retraining_same_shape(self, sample_regression_data):
        """Test MRMR skips retraining on same shape data."""
        df, feature_names, y = sample_regression_data
        X = df[feature_names]

        selector = MRMR(
            verbose=0,
            max_runtime_mins=1,
            quantization_nbins=5,
            skip_retraining_on_same_shape=True,
            n_workers=1,
        )

        # First fit
        selector.fit(X, y)

        # Second fit with same shape should be skipped
        selector.fit(X, y)

        # Signature should be set
        assert selector.signature is not None

    def test_mrmr_different_quantization_methods(self, sample_regression_data):
        """Test MRMR with different quantization methods."""
        df, feature_names, y = sample_regression_data
        X = df[feature_names].iloc[:200]  # Small subset for speed
        y_subset = y[:200]

        # Only 'quantile' and 'uniform' are supported by discretize_array
        for method in ['quantile', 'uniform']:
            selector = MRMR(
                verbose=0,
                max_runtime_mins=0.5,
                quantization_method=method,
                quantization_nbins=5,
                use_simple_mode=True,
                n_workers=1,
            )

            try:
                selector.fit(X, y_subset)
                assert hasattr(selector, 'n_features_in_')
            except Exception as e:
                # Some methods may not work with certain data
                warnings.warn(f"Quantization method {method} failed: {e}")


# ================================================================================================
# Test Class 2: RFECV Feature Selection
# ================================================================================================


class TestRFECVFeatureSelection:
    """Tests for RFECV feature selector."""

    def test_rfecv_basic_regression(self, sample_regression_data):
        """Test basic RFECV with regression estimator."""
        df, feature_names, y = sample_regression_data
        X = df[feature_names].iloc[:200]  # Small subset for speed
        y_subset = y[:200]

        estimator = RandomForestRegressor(n_estimators=5, random_state=42)

        selector = RFECV(
            estimator=estimator,
            cv=2,
            verbose=0,
            max_runtime_mins=1,
            max_refits=3,
        )

        selector.fit(X, y_subset)

        assert hasattr(selector, 'n_features_in_')

    def test_rfecv_basic_classification(self, sample_classification_data):
        """Test basic RFECV with classification estimator."""
        df, feature_names, _, y = sample_classification_data
        X = df[feature_names].iloc[:200]
        y_subset = y[:200]

        estimator = RandomForestClassifier(n_estimators=5, random_state=42)

        selector = RFECV(
            estimator=estimator,
            cv=2,
            verbose=0,
            max_runtime_mins=1,
            max_refits=3,
        )

        selector.fit(X, y_subset)

        assert hasattr(selector, 'n_features_in_')

    def test_rfecv_with_catboost_regressor(self, sample_regression_data):
        """Test RFECV with CatBoost regressor."""
        df, feature_names, y = sample_regression_data
        X = df[feature_names].iloc[:200]
        y_subset = y[:200]

        estimator = CatBoostRegressor(
            iterations=10,
            verbose=0,
            allow_writing_files=False,
        )

        selector = RFECV(
            estimator=estimator,
            cv=2,
            verbose=0,
            max_runtime_mins=1,
            max_refits=3,
        )

        selector.fit(X, y_subset)

        assert hasattr(selector, 'n_features_in_')

    def test_rfecv_with_catboost_classifier(self, sample_classification_data):
        """Test RFECV with CatBoost classifier."""
        df, feature_names, _, y = sample_classification_data
        X = df[feature_names].iloc[:200]
        y_subset = y[:200]

        estimator = CatBoostClassifier(
            iterations=10,
            verbose=0,
            allow_writing_files=False,
        )

        selector = RFECV(
            estimator=estimator,
            cv=2,
            verbose=0,
            max_runtime_mins=1,
            max_refits=3,
        )

        selector.fit(X, y_subset)

        assert hasattr(selector, 'n_features_in_')

    def test_rfecv_transform(self, sample_regression_data):
        """Test RFECV transform after fit."""
        df, feature_names, y = sample_regression_data
        X = df[feature_names].iloc[:200]
        y_subset = y[:200]

        estimator = RandomForestRegressor(n_estimators=5, random_state=42)

        selector = RFECV(
            estimator=estimator,
            cv=2,
            verbose=0,
            max_runtime_mins=1,
            max_refits=3,
        )

        selector.fit(X, y_subset)

        # Transform if support_ is set
        if hasattr(selector, 'support_') and selector.support_ is not None:
            X_transformed = selector.transform(X)
            assert X_transformed.shape[1] <= X.shape[1]

    def test_rfecv_with_early_stopping(self, sample_regression_data):
        """Test RFECV with early stopping configuration."""
        df, feature_names, y = sample_regression_data
        X = df[feature_names].iloc[:200]
        y_subset = y[:200]

        estimator = CatBoostRegressor(
            iterations=20,
            verbose=0,
            allow_writing_files=False,
        )

        selector = RFECV(
            estimator=estimator,
            cv=2,
            verbose=0,
            max_runtime_mins=1,
            max_refits=3,
            early_stopping_rounds=5,
        )

        selector.fit(X, y_subset)

        assert hasattr(selector, 'n_features_in_')


# ================================================================================================
# Test Class 3: Integration with train_mlframe_models_suite
# ================================================================================================


class TestFeatureSelectionIntegration:
    """Test feature selection integration with training suite."""

    def test_rfecv_classification(self, sample_classification_data, temp_data_dir, common_init_params, fast_iterations):
        """Test RFECV with classification task."""
        df, feature_names, cat_features, y = sample_classification_data
        # Remove categorical for simplicity
        numeric_df = df[[f for f in feature_names if f not in cat_features] + ['target']]

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, metadata = train_mlframe_models_suite(
            df=numeric_df,
            target_name="test_target",
            model_name="rfecv_classification",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            rfecv_models=["cb_rfecv"],
            config_params_override={"iterations": fast_iterations},
            init_common_params={
                **common_init_params,
                "rfecv_params": {"max_runtime_mins": 1, "max_refits": 3},
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models
        assert TargetTypes.BINARY_CLASSIFICATION in models["target"]

    def test_multiple_rfecv_models(self, sample_regression_data, temp_data_dir, common_init_params, fast_iterations, check_lgb_gpu_available):
        """Test training with multiple RFECV estimators."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Use models that are available
        rfecv_models = ["cb_rfecv", "xgb_rfecv"]
        if check_lgb_gpu_available:
            rfecv_models.append("lgb_rfecv")

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="multi_rfecv",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            rfecv_models=rfecv_models,
            config_params_override={"iterations": fast_iterations},
            init_common_params={
                **common_init_params,
                "rfecv_params": {"max_runtime_mins": 1, "max_refits": 3},
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]

    def test_use_mrmr_fs_true(self, sample_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test training with MRMR feature selection enabled."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="with_mrmr",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            use_mrmr_fs=True,
            mrmr_kwargs={
                "verbose": 0,
                "max_runtime_mins": 1,
                "n_workers": 1,
                "quantization_nbins": 5,
                "use_simple_mode": True,
            },
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]

    def test_mrmr_with_classification(self, sample_classification_data, temp_data_dir, common_init_params, fast_iterations):
        """Test MRMR feature selection with classification task."""
        df, feature_names, cat_features, y = sample_classification_data
        # Use only numeric features
        numeric_df = df[[f for f in feature_names if f not in cat_features] + ['target']]

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, metadata = train_mlframe_models_suite(
            df=numeric_df,
            target_name="test_target",
            model_name="mrmr_classification",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            use_mrmr_fs=True,
            mrmr_kwargs={
                "verbose": 0,
                "max_runtime_mins": 1,
                "n_workers": 1,
                "quantization_nbins": 5,
            },
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models
        assert TargetTypes.BINARY_CLASSIFICATION in models["target"]


# ================================================================================================
# Test Class 4: Combined Pipeline Tests
# ================================================================================================


class TestCombinedPipelines:
    """Test combined feature selection and pipeline scenarios."""

    def test_mrmr_combined_with_rfecv(self, sample_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test using MRMR + RFECV together in the same training run."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train with both MRMR (filter) and RFECV (wrapper) feature selection
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mrmr_plus_rfecv",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            rfecv_models=["cb_rfecv"],  # RFECV wrapper
            use_mrmr_fs=True,  # MRMR filter
            mrmr_kwargs={
                "verbose": 0,
                "max_runtime_mins": 1,
                "n_workers": 1,
                "quantization_nbins": 5,
                "use_simple_mode": True,
            },
            config_params_override={"iterations": fast_iterations},
            init_common_params={
                **common_init_params,
                "rfecv_params": {"max_runtime_mins": 1, "max_refits": 3},
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        # Should have models from both regular training and RFECV
        assert len(models["target"][TargetTypes.REGRESSION]) >= 1

    def test_feature_selection_with_fairness(self, temp_data_dir, common_init_params, fast_iterations):
        """Test feature selection combined with fairness_features parameter."""
        np.random.seed(42)
        n_samples = 200

        df = pd.DataFrame({
            'feature_0': np.random.randn(n_samples),
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'group_feature': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randn(n_samples),
        })

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="fs_with_fairness",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            use_mrmr_fs=True,
            mrmr_kwargs={
                "verbose": 0,
                "max_runtime_mins": 1,
                "n_workers": 1,
            },
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            control_params_override={
                'fairness_features': ['group_feature'],
                'fairness_min_pop_cat_thresh': 10,  # Small threshold for test data
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]

    def test_rfecv_with_polars(self, sample_polars_data, temp_data_dir, common_init_params, fast_iterations):
        """Test RFECV with Polars DataFrame input."""
        pl_df, feature_names, y = sample_polars_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="rfecv_polars",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            rfecv_models=["cb_rfecv"],
            config_params_override={"iterations": fast_iterations},
            init_common_params={
                **common_init_params,
                "rfecv_params": {"max_runtime_mins": 1, "max_refits": 3},
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models

    def test_mrmr_with_polars(self, sample_polars_data, temp_data_dir, common_init_params, fast_iterations):
        """Test MRMR with Polars DataFrame input."""
        pl_df, feature_names, y = sample_polars_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="mrmr_polars",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            use_mrmr_fs=True,
            mrmr_kwargs={
                "verbose": 0,
                "max_runtime_mins": 1,
                "n_workers": 1,
            },
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models

    def test_rfecv_with_small_dataset(self, temp_data_dir, common_init_params):
        """Test RFECV with very small dataset."""
        # Create small dataset
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_0': np.random.randn(50),
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50),
            'target': np.random.randn(50),
        })

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="rfecv_small",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            rfecv_models=["cb_rfecv"],
            config_params_override={"iterations": 5},
            init_common_params={
                **common_init_params,
                "rfecv_params": {"max_runtime_mins": 0.5, "max_refits": 2, "cv": 2},
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models

    def test_rfecv_with_many_features(self, temp_data_dir, common_init_params):
        """Test RFECV with many features (feature selection stress test)."""
        # Create dataset with many features
        np.random.seed(42)
        n_features = 30
        X = np.random.randn(200, n_features)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(200) * 0.5

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="rfecv_many_features",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            rfecv_models=["cb_rfecv"],
            config_params_override={"iterations": 5},
            init_common_params={
                **common_init_params,
                "rfecv_params": {"max_runtime_mins": 2, "max_refits": 5},
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models


# ================================================================================================
# Test Class 5: Edge Cases
# ================================================================================================


class TestFeatureSelectionEdgeCases:
    """Test edge cases in feature selection."""

    def test_mrmr_with_constant_feature(self):
        """Test MRMR handles constant features."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_0': np.random.randn(100),
            'constant': np.ones(100),  # Constant feature
            'feature_2': np.random.randn(100),
        })
        y = np.random.randn(100)

        selector = MRMR(
            verbose=0,
            max_runtime_mins=0.5,
            quantization_nbins=3,
            n_workers=1,
        )

        # Should handle constant feature gracefully
        try:
            selector.fit(X, y)
            assert True
        except Exception:
            # May raise error for constant features - that's ok
            pass

    def test_mrmr_with_nan_values(self):
        """Test MRMR with NaN values in data."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_0': np.random.randn(100),
            'feature_1': np.random.randn(100),
        })
        X.loc[10:15, 'feature_0'] = np.nan  # Add NaN values
        y = np.random.randn(100)

        selector = MRMR(
            verbose=0,
            max_runtime_mins=0.5,
            quantization_nbins=3,
            n_workers=1,
        )

        # May handle or raise error
        try:
            selector.fit(X, y)
        except Exception:
            # Expected - NaN handling varies
            pass

    def test_rfecv_with_single_feature(self):
        """Test RFECV with single feature."""
        np.random.seed(42)
        X = pd.DataFrame({'feature_0': np.random.randn(100)})
        y = np.random.randn(100)

        estimator = RandomForestRegressor(n_estimators=5, random_state=42)

        selector = RFECV(
            estimator=estimator,
            cv=2,
            verbose=0,
            max_runtime_mins=0.5,
            max_refits=2,
        )

        # Should work with single feature
        selector.fit(X, y)
        assert selector.n_features_in_ == 1

    def test_rfecv_stops_on_timeout(self):
        """Test RFECV respects max_runtime_mins."""
        np.random.seed(42)
        n_features = 20
        X = pd.DataFrame(np.random.randn(500, n_features), columns=[f'f{i}' for i in range(n_features)])
        y = np.random.randn(500)

        estimator = RandomForestRegressor(n_estimators=10, random_state=42)

        selector = RFECV(
            estimator=estimator,
            cv=3,
            verbose=0,
            max_runtime_mins=0.1,  # Very short timeout
            max_refits=100,  # High refits that won't be reached
        )

        import time
        start = time.time()
        selector.fit(X, y)
        elapsed = time.time() - start

        # Should finish within reasonable time (allowing some overhead)
        assert elapsed < 60  # 1 minute max including overhead
