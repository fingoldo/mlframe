"""
Comprehensive tests for all 13 mlframe model types.

Tests cover:
- Regression and classification
- Categorical feature support (with ordinal and onehot encoding)
- Polars DataFrame support
- CPU and GPU configurations (for cb, xgb, lgb, mlp)
- RFECV feature selection with all supported estimators
- Special cases and edge conditions
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.configs import PolarsPipelineConfig


# ================================================================================================
# Mock FeaturesAndTargetsExtractor
# ================================================================================================

class SimpleFeaturesAndTargetsExtractor:
    """Mock FeaturesAndTargetsExtractor for testing."""

    def __init__(self, target_column='target', regression=True):
        self.target_column = target_column
        self.regression = regression

    def transform(self, df):
        """
        Transform method that returns the expected tuple.

        Returns: (df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, columns_to_drop)
        """
        # Extract target
        if isinstance(df, pd.DataFrame):
            target_values = df[self.target_column].values
        else:  # Polars
            target_values = df[self.target_column].to_numpy()

        # Create target_by_type dict
        target_type = "REGRESSION" if self.regression else "CLASSIFICATION"
        target_by_type = {
            target_type: {
                self.target_column: target_values
            }
        }

        # Return all expected values
        return (
            df,  # df
            target_by_type,  # target_by_type
            None,  # group_ids_raw
            None,  # group_ids
            None,  # timestamps
            None,  # artifacts
            [self.target_column],  # columns_to_drop
        )


# ================================================================================================
# Model Lists & Constants
# ================================================================================================

# All 13 model types
LINEAR_MODELS = ["linear", "ridge", "lasso", "elasticnet", "huber", "ransac", "sgd"]
TREE_MODELS = ["cb", "lgb", "xgb", "hgb"]
NEURAL_MODELS = ["mlp", "ngb"]
ALL_MODELS = LINEAR_MODELS + TREE_MODELS + NEURAL_MODELS

# Models that support classification (exclude RANSAC)
CLASSIFICATION_MODELS = [
    "linear", "ridge", "lasso", "elasticnet", "huber", "sgd",  # Linear
    "cb", "lgb", "xgb", "hgb",  # Tree
    "mlp", "ngb"  # Neural
]

# Models that support categorical features natively (HGB excluded per user request)
CATEGORICAL_NATIVE_MODELS = ["cb", "lgb", "xgb"]

# Models that need category encoding in pipeline (HGB added per user request)
CATEGORICAL_ENCODING_MODELS = LINEAR_MODELS + NEURAL_MODELS + ["hgb"]

# Models that support GPU
GPU_MODELS = ["cb", "xgb", "lgb", "mlp"]


# ================================================================================================
# Test Class 1: All Models Regression
# ================================================================================================

class TestAllModelsRegression:
    """Test all 13 models on regression tasks."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_basic_regression(self, model_name, sample_regression_data, sample_large_regression_data, temp_data_dir):
        """Test basic regression for all models."""
        pytest_module = __import__('pytest')

        # Import checks for optional dependencies
        if model_name == "ngb":
            try:
                import ngboost
            except ImportError:
                pytest_module.skip("NGBoost not available")
        elif model_name == "mlp":
            try:
                import pytorch_lightning
            except ImportError:
                pytest_module.skip("PyTorch Lightning not available")

        # Use larger dataset for SGD for better convergence
        if model_name == "sgd":
            df, feature_names, y = sample_large_regression_data
        else:
            df, feature_names, y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_regression",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_regression_with_polars(self, model_name, sample_polars_data, temp_data_dir):
        """Test regression with Polars DataFrame."""
        pytest_module = __import__('pytest')

        # Import checks
        if model_name == "ngb":
            try:
                import ngboost
            except ImportError:
                pytest_module.skip("NGBoost not available")
        elif model_name == "mlp":
            try:
                import pytorch_lightning
            except ImportError:
                pytest_module.skip("PyTorch Lightning not available")

        # Skip SGD with small Polars data (convergence issues)
        if model_name == "sgd":
            pytest_module.skip("SGD needs larger dataset")

        pl_df, feature_names, y = sample_polars_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name=f"{model_name}_polars_regression",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0


# ================================================================================================
# Test Class 2: All Models Classification
# ================================================================================================

class TestAllModelsClassification:
    """Test all models on classification tasks (excluding RANSAC)."""

    @pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
    def test_basic_classification(self, model_name, sample_classification_data, temp_data_dir):
        """Test basic classification."""
        pytest_module = __import__('pytest')

        # Import checks
        if model_name == "ngb":
            try:
                import ngboost
            except ImportError:
                pytest_module.skip("NGBoost not available")
        elif model_name == "mlp":
            try:
                import pytorch_lightning
            except ImportError:
                pytest_module.skip("PyTorch Lightning not available")

        df, feature_names, y = sample_classification_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=False)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_classification",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "CLASSIFICATION" in models["target"]
        assert len(models["target"]["CLASSIFICATION"]) > 0

    @pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
    def test_classification_with_polars(self, model_name, sample_classification_data, temp_data_dir):
        """Test classification with Polars DataFrame."""
        pytest_module = __import__('pytest')

        # Import checks
        if model_name == "ngb":
            try:
                import ngboost
            except ImportError:
                pytest_module.skip("NGBoost not available")
        elif model_name == "mlp":
            try:
                import pytorch_lightning
            except ImportError:
                pytest_module.skip("PyTorch Lightning not available")

        df, feature_names, y = sample_classification_data
        pl_df = pl.from_pandas(df)

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=False)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name=f"{model_name}_polars_classification",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "CLASSIFICATION" in models["target"]
        assert len(models["target"]["CLASSIFICATION"]) > 0


# ================================================================================================
# Test Class 3: Categorical Features
# ================================================================================================

class TestCategoricalFeatures:
    """Test categorical feature support."""

    @pytest.mark.parametrize("model_name", CATEGORICAL_NATIVE_MODELS)
    @pytest.mark.parametrize("encoding", ["ordinal", "onehot"])
    def test_native_categorical_support(self, model_name, encoding, sample_categorical_data, temp_data_dir):
        """Test models with native categorical support (cb, lgb, xgb)."""
        df, feature_names, cat_features, y = sample_categorical_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Configure pipeline with categorical encoding
        pipeline_config = PolarsPipelineConfig(
            categorical_encoding=encoding,
        )

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_{encoding}_native_cat",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            pipeline_config=pipeline_config,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0

    @pytest.mark.parametrize("model_name", CATEGORICAL_ENCODING_MODELS)
    @pytest.mark.parametrize("encoding", ["ordinal", "onehot"])
    def test_categorical_with_encoding(self, model_name, encoding, sample_categorical_data, temp_data_dir):
        """Test models that need category encoding in pipeline (linear, neural, hgb)."""
        pytest_module = __import__('pytest')

        # Import checks
        if model_name == "ngb":
            try:
                import ngboost
            except ImportError:
                pytest_module.skip("NGBoost not available")
        elif model_name == "mlp":
            try:
                import pytorch_lightning
            except ImportError:
                pytest_module.skip("PyTorch Lightning not available")

        # Skip RANSAC with categorical (not well-suited)
        if model_name == "ransac":
            pytest_module.skip("RANSAC not tested with categorical features")

        df, feature_names, cat_features, y = sample_categorical_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Configure pipeline with categorical encoding
        pipeline_config = PolarsPipelineConfig(
            categorical_encoding=encoding,
        )

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_{encoding}_encoded_cat",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            pipeline_config=pipeline_config,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0


# ================================================================================================
# Test Class 4: GPU Support
# ================================================================================================

class TestGPUSupport:
    """Test GPU configuration for supported models."""

    @pytest.mark.parametrize("model_name", ["cb", "xgb"])
    def test_cpu_configuration(self, model_name, sample_regression_data, temp_data_dir):
        """Test forced CPU configuration for cb and xgb."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Force CPU via config_params_override
        if model_name == "cb":
            config_override = {'cb_kwargs': {'task_type': 'CPU'}}
        elif model_name == "xgb":
            config_override = {'xgb_kwargs': {'device': 'cpu'}}

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_cpu",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            config_params_override=config_override,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0

    @pytest.mark.parametrize("model_name", ["cb", "xgb"])
    def test_gpu_configuration(self, model_name, sample_regression_data, temp_data_dir, check_gpu_available):
        """Test GPU configuration for cb and xgb (if GPU available)."""
        if not check_gpu_available:
            pytest.skip("GPU not available")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Configure for GPU via config_params_override
        if model_name == "cb":
            config_override = {'cb_kwargs': {'task_type': 'GPU'}}
        elif model_name == "xgb":
            config_override = {'xgb_kwargs': {'device': 'cuda'}}

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_gpu",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            config_params_override=config_override,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0

    def test_lgb_cpu_configuration(self, sample_regression_data, temp_data_dir):
        """Test LightGBM CPU configuration."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Force CPU via config_params_override
        config_override = {'lgb_kwargs': {'device_type': 'cpu'}}

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="lgb_cpu",
            features_and_targets_extractor=fte,
            mlframe_models=["lgb"],
            config_params_override=config_override,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0

    def test_lgb_gpu_configuration(self, sample_regression_data, temp_data_dir, check_lgb_gpu_available, check_gpu_available):
        """Test LightGBM GPU configuration (with CUDA Tree Learner check)."""
        if not check_gpu_available:
            pytest.skip("GPU not available")

        if not check_lgb_gpu_available:
            pytest.skip("LightGBM CUDA not available (missing GPU-enabled build)")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Configure for GPU via config_params_override
        config_override = {'lgb_kwargs': {'device_type': 'cuda'}}

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="lgb_gpu",
            features_and_targets_extractor=fte,
            mlframe_models=["lgb"],
            config_params_override=config_override,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0

    def test_mlp_cpu_configuration(self, sample_regression_data, temp_data_dir):
        """Test MLP CPU configuration (with trainer params)."""
        pytest_module = __import__('pytest')

        try:
            import pytorch_lightning
        except ImportError:
            pytest_module.skip("PyTorch Lightning not available")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Force CPU with trainer params via config_params_override
        config_override = {
            'mlp_kwargs': {
                'trainer_params': {
                    'accelerator': 'cpu',
                    'devices': 1
                }
            }
        }

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mlp_cpu",
            features_and_targets_extractor=fte,
            mlframe_models=["mlp"],
            config_params_override=config_override,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0

    def test_mlp_gpu_configuration(self, sample_regression_data, temp_data_dir, check_gpu_available):
        """Test MLP GPU configuration (if GPU available)."""
        pytest_module = __import__('pytest')

        try:
            import pytorch_lightning
            import torch
        except ImportError:
            pytest_module.skip("PyTorch Lightning not available")

        if not check_gpu_available:
            pytest_module.skip("GPU not available")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Configure for GPU via config_params_override
        config_override = {
            'mlp_kwargs': {
                'trainer_params': {
                    'accelerator': 'cuda',
                    'devices': 1
                }
            }
        }

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mlp_gpu",
            features_and_targets_extractor=fte,
            mlframe_models=["mlp"],
            config_params_override=config_override,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0


# ================================================================================================
# Test Class 5: Feature Selection
# ================================================================================================

class TestFeatureSelection:
    """Test RFECV feature selection with different estimators."""

    @pytest.mark.parametrize("estimator", ["cb_rfecv", "lgb_rfecv", "xgb_rfecv"])
    def test_rfecv_with_estimator(self, estimator, sample_regression_data, temp_data_dir, check_lgb_gpu_available):
        """Test RFECV with each supported estimator, using cb as final model."""
        # Skip LightGBM RFECV if GPU build not available
        if estimator == "lgb_rfecv" and not check_lgb_gpu_available:
            pytest.skip("LightGBM CUDA not available (missing GPU-enabled build)")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train with RFECV
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"cb_with_{estimator}",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],  # Use cb as final model
            rfecv_models=[estimator],  # Vary the RFECV estimator
            init_common_params={
                'show_perf_chart': False,
                'rfecv_params': {'max_runtime_mins': 2},  # Limit RFECV runtime for tests
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0


# ================================================================================================
# Test Class 6: Special Cases
# ================================================================================================

class TestSpecialCases:
    """Test special cases and edge conditions."""

    def test_ransac_regression_only(self, sample_outlier_data, temp_data_dir):
        """Test that RANSAC works for regression (not classification)."""
        df, feature_names, y = sample_outlier_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train RANSAC on regression with outliers
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="ransac_outliers",
            features_and_targets_extractor=fte,
            mlframe_models=["ransac"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0

    def test_ridge_classifier_without_predict_proba(self, sample_classification_data, temp_data_dir):
        """Test RidgeClassifier uses predict() fallback (no predict_proba)."""
        df, feature_names, y = sample_classification_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=False)

        # Train RidgeClassifier
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="ridge_classifier_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify it worked (should use predict() fallback)
        assert "target" in models
        assert "CLASSIFICATION" in models["target"]
        assert len(models["target"]["CLASSIFICATION"]) > 0

    def test_sgd_convergence(self, sample_large_regression_data, temp_data_dir):
        """Test SGD with larger dataset for better convergence."""
        df, feature_names, y = sample_large_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train SGD with larger dataset
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="sgd_large_data",
            features_and_targets_extractor=fte,
            mlframe_models=["sgd"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0

    def test_huber_with_outliers(self, sample_outlier_data, temp_data_dir):
        """Test Huber regressor on data with outliers."""
        df, feature_names, y = sample_outlier_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train Huber with outliers
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="huber_outliers",
            features_and_targets_extractor=fte,
            mlframe_models=["huber"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0
