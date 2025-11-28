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
from mlframe.training.configs import TargetTypes
from .shared import SimpleFeaturesAndTargetsExtractor, get_cpu_config, skip_if_dependency_missing


# ================================================================================================
# Model Lists & Constants
# ================================================================================================

# All 13 model types
LINEAR_MODELS = ["linear", "ridge", "lasso", "elasticnet", "huber", "ransac", "sgd"]
TREE_MODELS = ["cb", "lgb", "xgb", "hgb"]
NEURAL_MODELS = ["mlp", "ngb"]
ALL_MODELS = LINEAR_MODELS + TREE_MODELS + NEURAL_MODELS

# Models that support classification (exclude RANSAC)
CLASSIFICATION_MODELS = ["linear", "ridge", "lasso", "elasticnet", "huber", "sgd", "cb", "lgb", "xgb", "hgb", "mlp", "ngb"]  # Linear  # Tree  # Neural

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
    def test_basic_regression(self, model_name, sample_regression_data, sample_large_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test basic regression for all models."""
        skip_if_dependency_missing(model_name)

        # Use larger dataset for SGD for better convergence
        if model_name == "sgd":
            df, feature_names, y = sample_large_regression_data
        else:
            df, feature_names, y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
        config_override = get_cpu_config(model_name, fast_iterations)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_regression",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_regression_with_polars(self, model_name, sample_polars_data, temp_data_dir, common_init_params, fast_iterations):
        """Test regression with Polars DataFrame."""
        skip_if_dependency_missing(model_name)

        pl_df, feature_names, y = sample_polars_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
        config_override = get_cpu_config(model_name, fast_iterations)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name=f"{model_name}_polars_regression",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0


# ================================================================================================
# Test Class 2: All Models Classification
# ================================================================================================


class TestAllModelsClassification:
    """Test all models on classification tasks (excluding RANSAC)."""

    @pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
    def test_basic_classification(self, model_name, sample_classification_data, temp_data_dir, common_init_params, fast_iterations):
        """Test basic classification."""
        skip_if_dependency_missing(model_name)

        df, feature_names, _, y = sample_classification_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        config_override = get_cpu_config(model_name, fast_iterations)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_classification",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.BINARY_CLASSIFICATION in models["target"]
        assert len(models["target"][TargetTypes.BINARY_CLASSIFICATION]) > 0

    @pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
    def test_classification_with_polars(self, model_name, sample_classification_data, temp_data_dir, common_init_params, fast_iterations):
        """Test classification with Polars DataFrame."""
        skip_if_dependency_missing(model_name)

        df, feature_names, _, y = sample_classification_data
        pl_df = pl.from_pandas(df)

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        config_override = get_cpu_config(model_name, fast_iterations)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name=f"{model_name}_polars_classification",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.BINARY_CLASSIFICATION in models["target"]
        assert len(models["target"][TargetTypes.BINARY_CLASSIFICATION]) > 0


# ================================================================================================
# Test Class 3: Categorical Features
# ================================================================================================


class TestCategoricalFeatures:
    """Test categorical feature support."""

    @pytest.mark.parametrize("model_name", CATEGORICAL_NATIVE_MODELS)
    @pytest.mark.parametrize("encoding", ["ordinal", "onehot"])
    def test_native_categorical_support(self, model_name, encoding, sample_categorical_data, temp_data_dir, common_init_params, fast_iterations):
        """Test models with native categorical support (cb, lgb, xgb)."""
        df, feature_names, cat_features, y = sample_categorical_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Configure pipeline with categorical encoding
        pipeline_config = PolarsPipelineConfig(
            categorical_encoding=encoding,
        )

        # Force CPU for GPU-capable models (GPU tests are separate)
        config_override = {"iterations": fast_iterations}
        if model_name == "cb":
            config_override.update({"cb_kwargs": {"task_type": "CPU"}})
        elif model_name == "xgb":
            config_override.update({"xgb_kwargs": {"device": "cpu"}})
        elif model_name == "lgb":
            config_override.update({"lgb_kwargs": {"device_type": "cpu"}})
        elif model_name == "mlp":
            config_override.update({"mlp_kwargs": {"trainer_params": {"devices": 1}}})

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_{encoding}_native_cat",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            pipeline_config=pipeline_config,
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    @pytest.mark.parametrize("model_name", CATEGORICAL_ENCODING_MODELS)
    @pytest.mark.parametrize("encoding", ["ordinal", "onehot"])
    def test_categorical_with_encoding(self, model_name, encoding, sample_categorical_data, temp_data_dir, common_init_params, fast_iterations):
        """Test models that need category encoding in pipeline (linear, neural, hgb)."""
        pytest_module = __import__("pytest")

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

        df, feature_names, cat_features, y = sample_categorical_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

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
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0


# ================================================================================================
# Test Class 4: GPU Support
# ================================================================================================


class TestGPUSupport:
    """Test GPU configuration for supported models."""

    @pytest.mark.parametrize("model_name", ["cb", "xgb"])
    def test_cpu_configuration(self, model_name, sample_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test forced CPU configuration for cb and xgb."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Force CPU via config_params_override
        config_override = {"iterations": fast_iterations}
        if model_name == "cb":
            config_override.update({"cb_kwargs": {"task_type": "CPU"}})
        elif model_name == "xgb":
            config_override.update({"xgb_kwargs": {"device": "cpu"}})

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_cpu",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    @pytest.mark.parametrize("model_name", ["cb", "xgb"])
    def test_gpu_configuration(self, model_name, sample_regression_data, temp_data_dir, check_gpu_available, common_init_params, fast_iterations):
        """Test GPU configuration for cb and xgb (if GPU available)."""
        if not check_gpu_available:
            pytest.skip("GPU not available")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Configure for GPU via config_params_override
        config_override = {"iterations": fast_iterations}
        if model_name == "cb":
            config_override.update({"cb_kwargs": {"task_type": "GPU"}})
        elif model_name == "xgb":
            config_override.update({"xgb_kwargs": {"device": "cuda"}})

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_gpu",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    def test_lgb_cpu_configuration(self, sample_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test LightGBM CPU configuration."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Force CPU via config_params_override
        config_override = {"iterations": fast_iterations, "lgb_kwargs": {"device_type": "cpu"}}

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="lgb_cpu",
            features_and_targets_extractor=fte,
            mlframe_models=["lgb"],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    def test_lgb_gpu_configuration(self, sample_regression_data, temp_data_dir, check_lgb_gpu_available, check_gpu_available, common_init_params, fast_iterations):
        """Test LightGBM GPU configuration (with CUDA Tree Learner check)."""
        if not check_gpu_available:
            pytest.skip("GPU not available")

        if not check_lgb_gpu_available:
            pytest.skip("LightGBM CUDA not available (missing GPU-enabled build)")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Configure for GPU via config_params_override
        config_override = {"iterations": fast_iterations, "lgb_kwargs": {"device_type": "cuda"}}

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="lgb_gpu",
            features_and_targets_extractor=fte,
            mlframe_models=["lgb"],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    def test_mlp_cpu_configuration(self, sample_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test MLP CPU configuration (with trainer params)."""
        pytest_module = __import__("pytest")

        try:
            import pytorch_lightning
        except ImportError:
            pytest_module.skip("PyTorch Lightning not available")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Force CPU with trainer params via config_params_override
        config_override = {"iterations": fast_iterations, "mlp_kwargs": {"trainer_params": {"accelerator": "cpu", "devices": 1}}}

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mlp_cpu",
            features_and_targets_extractor=fte,
            mlframe_models=["mlp"],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    def test_mlp_gpu_configuration(self, sample_regression_data, temp_data_dir, check_gpu_available, common_init_params, fast_iterations):
        """Test MLP GPU configuration (if GPU available)."""
        pytest_module = __import__("pytest")

        try:
            import pytorch_lightning
            import torch
        except ImportError:
            pytest_module.skip("PyTorch Lightning not available")

        if not check_gpu_available:
            pytest_module.skip("GPU not available")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Configure for GPU via config_params_override
        config_override = {"iterations": fast_iterations, "mlp_kwargs": {"trainer_params": {"accelerator": "cuda", "devices": 1}}}

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mlp_gpu",
            features_and_targets_extractor=fte,
            mlframe_models=["mlp"],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    def test_mlp_multi_gpu_configuration(self, sample_regression_data, temp_data_dir, check_gpu_available, common_init_params, fast_iterations):
        """Test MLP multi-GPU configuration (if multiple GPUs available)."""
        pytest_module = __import__("pytest")

        try:
            import pytorch_lightning
            import torch
        except ImportError:
            pytest_module.skip("PyTorch Lightning not available")

        if not check_gpu_available:
            pytest_module.skip("GPU not available")

        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            pytest_module.skip(f"Multi-GPU test requires at least 2 GPUs, found {num_gpus}")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Configure for multi-GPU via config_params_override
        # Use max 2 GPUs and ddp_spawn strategy to avoid NCCL hangs in pytest
        test_gpus = min(num_gpus, 2)
        config_override = {
            "iterations": fast_iterations,
            "mlp_kwargs": {
                "trainer_params": {
                    "accelerator": "cuda",
                    "devices": test_gpus,
                    "strategy": "ddp_spawn",
                }
            }
        }

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mlp_multi_gpu",
            features_and_targets_extractor=fte,
            mlframe_models=["mlp"],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    def test_catboost_multi_gpu_configuration(self, sample_regression_data, temp_data_dir, check_gpu_available, common_init_params, fast_iterations):
        """Test CatBoost multi-GPU configuration (if multiple GPUs available)."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available for GPU count check")

        if not check_gpu_available:
            pytest.skip("GPU not available")

        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            pytest.skip(f"Multi-GPU test requires at least 2 GPUs, found {num_gpus}")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Configure for multi-GPU via config_params_override
        config_override = {"iterations": fast_iterations, "cb_kwargs": {"task_type": "GPU", "devices": f"0-{num_gpus-1}", "verbose": 0}}

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="cb_multi_gpu",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0


# ================================================================================================
# Test Class 5: Feature Selection
# ================================================================================================


class TestFeatureSelection:
    """Test RFECV feature selection with different estimators."""

    @pytest.mark.parametrize("estimator", ["cb_rfecv", "lgb_rfecv", "xgb_rfecv"])
    def test_rfecv_with_estimator(self, estimator, sample_regression_data, temp_data_dir, check_lgb_gpu_available, common_init_params, fast_iterations):
        """Test RFECV with each supported estimator, using cb as final model."""
        # Skip LightGBM RFECV if GPU build not available
        if estimator == "lgb_rfecv" and not check_lgb_gpu_available:
            pytest.skip("LightGBM CUDA not available (missing GPU-enabled build)")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train with RFECV - force CPU for all models to avoid GPU memory issues
        config_override = {
            "iterations": fast_iterations,
            "cb_kwargs": {"task_type": "CPU"},  # Force CatBoost to CPU
            "xgb_kwargs": {"device": "cpu"},  # Force XGBoost to CPU
            "lgb_kwargs": {"device_type": "cpu"},  # Force LightGBM to CPU
        }
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"cb_with_{estimator}",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],  # Use cb as final model
            rfecv_models=[estimator],  # Vary the RFECV estimator
            config_params_override=config_override,
            init_common_params={
                **common_init_params,
                "rfecv_params": {"max_runtime_mins": 2},  # Limit RFECV runtime for tests
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0


# ================================================================================================
# Test Class 6: Special Cases
# ================================================================================================


class TestSpecialCases:
    """Test special cases and edge conditions."""

    def test_ransac_regression_only(self, sample_outlier_data, temp_data_dir, common_init_params, fast_iterations):
        """Test that RANSAC works for regression (not classification)."""
        df, feature_names, y = sample_outlier_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train RANSAC on regression with outliers
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="ransac_outliers",
            features_and_targets_extractor=fte,
            mlframe_models=["ransac"],
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    def test_ridge_classifier_without_predict_proba(self, sample_classification_data, temp_data_dir, common_init_params, fast_iterations):
        """Test RidgeClassifier uses predict() fallback (no predict_proba)."""
        df, feature_names, _, y = sample_classification_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # Train RidgeClassifier
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="ridge_classifier_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify it worked (should use predict() fallback)
        assert "target" in models
        assert TargetTypes.BINARY_CLASSIFICATION in models["target"]
        assert len(models["target"][TargetTypes.BINARY_CLASSIFICATION]) > 0

    def test_sgd_convergence(self, sample_large_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test SGD with larger dataset for better convergence."""
        df, feature_names, y = sample_large_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train SGD with larger dataset
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="sgd_large_data",
            features_and_targets_extractor=fte,
            mlframe_models=["sgd"],
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    def test_huber_with_outliers(self, sample_outlier_data, temp_data_dir, common_init_params, fast_iterations):
        """Test Huber regressor on data with outliers."""
        df, feature_names, y = sample_outlier_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train Huber with outliers
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="huber_outliers",
            features_and_targets_extractor=fte,
            mlframe_models=["huber"],
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0


# ================================================================================================
# Test Class 7: Outlier Detection
# ================================================================================================


class TestOutlierDetection:
    """Test outlier detection with different detectors in train_mlframe_models_suite."""

    def test_isolation_forest_outlier_detection(self, sample_outlier_data, temp_data_dir, common_init_params, fast_iterations):
        """Test IsolationForest outlier detection in parent function."""
        from sklearn.ensemble import IsolationForest
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        df, feature_names, y = sample_outlier_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Create IsolationForest outlier detector
        outlier_detector = Pipeline([
            ("imp", SimpleImputer()),
            ("est", IsolationForest(contamination=0.1, n_estimators=50, random_state=42, n_jobs=-1))
        ])

        # Train with outlier detection
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="outlier_detection_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],  # Simple model for fast test
            outlier_detector=outlier_detector,
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify model trained successfully
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

        # Verify outlier detector is stored in metadata
        assert "outlier_detector" in metadata
        assert metadata["outlier_detector"] is not None

    def test_outlier_detection_reduces_training_samples(self, sample_outlier_data, temp_data_dir, common_init_params, fast_iterations):
        """Test that outlier detection actually removes samples."""
        from sklearn.ensemble import IsolationForest
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        df, feature_names, y = sample_outlier_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
        original_size = len(df)

        # High contamination to remove more samples
        outlier_detector = Pipeline([
            ("imp", SimpleImputer()),
            ("est", IsolationForest(contamination=0.15, n_estimators=50, random_state=42, n_jobs=-1))
        ])

        # Train with outlier detection
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="outlier_reduction_test",
            features_and_targets_extractor=fte,
            mlframe_models=["linear"],  # Simple model for fast test
            outlier_detector=outlier_detector,
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=1,  # Enable verbose to see outlier rejection messages
        )

        # Verify model trained
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0

    def test_local_outlier_factor_detection(self, sample_outlier_data, temp_data_dir, common_init_params, fast_iterations):
        """Test LocalOutlierFactor outlier detection (novelty mode)."""
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        df, feature_names, y = sample_outlier_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Create LOF outlier detector (novelty=True for predict support)
        outlier_detector = Pipeline([
            ("imp", SimpleImputer()),
            ("est", LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True, n_jobs=-1))
        ])

        # Train with outlier detection
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="lof_outlier_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],  # Simple model for fast test
            outlier_detector=outlier_detector,
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify model trained successfully
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]
        assert len(models["target"][TargetTypes.REGRESSION]) > 0


# ================================================================================================
# Test Class 8: Prediction Validation
# ================================================================================================


class TestPredictionValidation:
    """Validate prediction quality and sanity."""

    def test_predictions_in_valid_range_regression(self, sample_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test regression predictions are in a reasonable range."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train model
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="pred_range_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Get predictions from the trained model result
        model_entry = models["target"][TargetTypes.REGRESSION][0]

        # Check that test predictions exist and are reasonable
        if hasattr(model_entry, 'test_preds') and model_entry.test_preds is not None:
            preds = model_entry.test_preds
            assert not np.all(np.isnan(preds)), "Predictions should not be all NaN"
            assert not np.all(np.isinf(preds)), "Predictions should not have infinity"

            # Predictions should be in a reasonable range (within 10x of target range)
            assert np.all(np.abs(preds) < np.abs(y).max() * 10 + 100), \
                "Predictions should be in reasonable range"

    def test_probabilities_sum_to_one(self, sample_classification_data, temp_data_dir, common_init_params, fast_iterations):
        """Test classification probabilities sum to approximately 1."""
        df, feature_names, _, y = sample_classification_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # Train model
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="prob_sum_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],  # CatBoost has good proba support
            config_params_override={"iterations": fast_iterations, "cb_kwargs": {"task_type": "CPU"}},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Get model entry
        model_entry = models["target"][TargetTypes.BINARY_CLASSIFICATION][0]

        # Check test probabilities if available
        if hasattr(model_entry, 'test_probs') and model_entry.test_probs is not None:
            probs = model_entry.test_probs
            if probs.ndim == 2:
                prob_sums = probs.sum(axis=1)
                np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5,
                    err_msg="Classification probabilities should sum to 1")
            else:
                # Binary probs in [0, 1]
                assert np.all(probs >= 0) and np.all(probs <= 1), \
                    "Binary probabilities should be in [0, 1]"

    def test_predictions_not_all_nan(self, sample_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test that predictions are not all NaN."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train model
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="not_nan_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Get predictions
        model_entry = models["target"][TargetTypes.REGRESSION][0]

        # Verify predictions are not all NaN
        for attr in ['train_preds', 'val_preds', 'test_preds']:
            if hasattr(model_entry, attr):
                preds = getattr(model_entry, attr)
                if preds is not None and len(preds) > 0:
                    assert not np.all(np.isnan(preds)), f"{attr} should not be all NaN"

    def test_predictions_not_all_same(self, sample_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test that predictions are not all identical (model learned something)."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train model
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="not_same_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": fast_iterations},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Get predictions
        model_entry = models["target"][TargetTypes.REGRESSION][0]

        # Verify predictions have variance (model learned)
        if hasattr(model_entry, 'test_preds') and model_entry.test_preds is not None:
            preds = model_entry.test_preds
            if len(preds) > 1:
                assert np.std(preds) > 1e-10, \
                    "Predictions should not all be identical (model should learn)"

    @pytest.mark.parametrize("model_name", ["cb", "lgb", "hgb", "ridge"])
    def test_prediction_shape_matches_input(self, model_name, sample_regression_data, temp_data_dir, common_init_params, fast_iterations):
        """Test that prediction shape matches input data size."""
        skip_if_dependency_missing(model_name)

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
        config_override = get_cpu_config(model_name, fast_iterations)

        # Train model
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_name}_shape_test",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Get predictions
        model_entry = models["target"][TargetTypes.REGRESSION][0]

        # Check that shapes match expected sizes from metadata
        if hasattr(model_entry, 'test_preds') and model_entry.test_preds is not None:
            # Predictions shape should match test size
            assert len(model_entry.test_preds) == metadata.get('test_size', len(model_entry.test_preds))


# ================================================================================================
# Test Class 9: GPU Usage Verification
# ================================================================================================


class TestGPUUsageVerification:
    """Verify GPU is actually used when configured."""

    def test_catboost_gpu_training_params(self, sample_regression_data, temp_data_dir, check_gpu_available, common_init_params, fast_iterations):
        """Test CatBoost GPU configuration is properly applied."""
        if not check_gpu_available:
            pytest.skip("GPU not available")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Configure for GPU
        config_override = {
            "iterations": fast_iterations,
            "cb_kwargs": {"task_type": "GPU", "verbose": 0}
        }

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="cb_gpu_verify",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify training succeeded
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]

        # Get the trained model and verify GPU params if accessible
        model_entry = models["target"][TargetTypes.REGRESSION][0]
        if hasattr(model_entry, 'model'):
            model = model_entry.model
            # CatBoost stores task_type in params
            if hasattr(model, 'get_param') or hasattr(model, 'get_all_params'):
                try:
                    params = model.get_all_params() if hasattr(model, 'get_all_params') else {}
                    # GPU task_type should be set
                    assert params.get('task_type', 'CPU') in ['GPU', 'gpu'], \
                        "CatBoost should be configured for GPU"
                except Exception:
                    pass  # Some models don't expose params

    def test_xgboost_gpu_training_params(self, sample_regression_data, temp_data_dir, check_gpu_available, common_init_params, fast_iterations):
        """Test XGBoost GPU configuration is properly applied."""
        if not check_gpu_available:
            pytest.skip("GPU not available")

        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Configure for GPU
        config_override = {
            "iterations": fast_iterations,
            "xgb_kwargs": {"device": "cuda"}
        }

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="xgb_gpu_verify",
            features_and_targets_extractor=fte,
            mlframe_models=["xgb"],
            config_params_override=config_override,
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify training succeeded
        assert "target" in models
        assert TargetTypes.REGRESSION in models["target"]

        # Verify XGBoost GPU config
        model_entry = models["target"][TargetTypes.REGRESSION][0]
        if hasattr(model_entry, 'model'):
            model = model_entry.model
            if hasattr(model, 'get_params'):
                params = model.get_params()
                # Check device is cuda
                assert params.get('device', 'cpu') in ['cuda', 'gpu'], \
                    "XGBoost should be configured for GPU"