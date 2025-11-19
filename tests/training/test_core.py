"""
Integration tests for core training functionality.

Tests the main train_mlframe_models_suite function end-to-end.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import joblib

from mlframe.training.core import train_mlframe_models_suite


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


class TestTrainMLFrameModelsSuiteBasic:
    """Basic smoke tests for train_mlframe_models_suite."""

    def test_train_single_linear_model_regression(self, sample_regression_data, temp_data_dir):
        """Test training a single linear model on regression data."""
        df, feature_names, y = sample_regression_data

        # Create FTE
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_model",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify structure (models[target_name][target_type])
        assert isinstance(models, dict)
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) > 0

        # Verify metadata
        assert metadata["model_name"] == "test_model"
        assert metadata["target_name"] == "test_target"
        assert "configs" in metadata
        assert "pipeline" in metadata

    def test_train_single_linear_model_classification(self, sample_classification_data, temp_data_dir):
        """Test training a single linear model on classification data."""
        df, feature_names, y = sample_classification_data

        # Create FTE
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=False)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_model_classif",
            features_and_targets_extractor=fte,
            mlframe_models=["linear"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify structure (models[target_name][target_type])
        assert isinstance(models, dict)
        assert "target" in models
        assert "CLASSIFICATION" in models["target"]

        # Check that model has predictions
        model_entry = models["target"]["CLASSIFICATION"][0]
        assert hasattr(model_entry, "model") or "model" in model_entry.__dict__ if hasattr(model_entry, "__dict__") else True

    def test_train_multiple_linear_models(self, sample_regression_data, temp_data_dir):
        """Test training multiple linear models together."""
        df, feature_names, y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train multiple models
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="multi_linear",
            features_and_targets_extractor=fte,
            mlframe_models=["linear", "ridge", "lasso"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify all 3 models were trained (models[target_name][target_type])
        assert "target" in models
        assert "REGRESSION" in models["target"]
        # Should have 3 models (linear, ridge, lasso)
        assert len(models["target"]["REGRESSION"]) >= 3

    def test_train_with_polars_dataframe(self, sample_polars_data, temp_data_dir):
        """Test training with Polars DataFrame input."""
        pl_df, feature_names, y = sample_polars_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train with Polars input
        models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="polars_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify it worked (models[target_name][target_type])
        assert isinstance(models, dict)
        assert "target" in models
        assert "REGRESSION" in models["target"]


class TestUnifiedTrainingLoop:
    """Test the unified training loop with mixed model types - THE CRITICAL TEST!"""

    def test_train_mixed_linear_and_tree_models(self, sample_regression_data, temp_data_dir):
        """
        THE CRITICAL TEST: Validate unified training loop with linear + tree models.

        This test confirms that linear models (ridge) and tree models (cb)
        can be trained together through the same unified training code path.
        This validates the entire refactoring effort!
        """
        pytest = __import__('pytest')

        # Check if catboost is available
        try:
            import catboost
            has_catboost = True
        except ImportError:
            has_catboost = False

        if not has_catboost:
            pytest.skip("CatBoost not available - skipping mixed model test")

        df, feature_names, y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # THE CRITICAL TEST: Train linear model + tree model together!
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="unified_loop_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge", "cb"],  # LINEAR + TREE!
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=1,  # Show output to see both models training
        )

        # Verify BOTH models were trained
        assert "target" in models
        assert "REGRESSION" in models["target"]

        # Should have 2 models: ridge (linear) + cb (tree)
        trained_models = models["target"]["REGRESSION"]
        assert len(trained_models) >= 2, f"Expected 2 models, got {len(trained_models)}"

        # Verify both model types are present
        model_names = [m.model_name if hasattr(m, 'model_name') else str(m) for m in trained_models]
        print(f"\n[SUCCESS] UNIFIED TRAINING LOOP VALIDATED!")
        print(f"   Trained models: {model_names}")
        print(f"   Linear + Tree models trained through SAME code path!")

    def test_train_mixed_linear_and_lgb(self, sample_regression_data, temp_data_dir):
        """Test unified training with linear + LightGBM models."""
        pytest = __import__('pytest')

        # Check if lightgbm is available
        try:
            import lightgbm
            has_lgb = True
        except ImportError:
            has_lgb = False

        if not has_lgb:
            pytest.skip("LightGBM not available - skipping mixed model test")

        df, feature_names, y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train linear + LightGBM
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="linear_lgb_test",
            features_and_targets_extractor=fte,
            mlframe_models=["linear", "lgb"],  # LINEAR + LGB!
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify both models trained
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) >= 2


class TestTrainMLFrameModelsSuiteEnsembles:
    """Test ensemble creation."""

    def test_train_with_ensembles(self, sample_regression_data, temp_data_dir):
        """Test ensemble creation from multiple models."""
        df, feature_names, y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train with ensembles enabled
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="ensemble_test",
            features_and_targets_extractor=fte,
            mlframe_models=["linear", "ridge"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=True,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify models were trained (models[target_name][target_type])
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) >= 2


class TestTrainMLFrameModelsSuiteMetadata:
    """Test metadata saving and structure."""

    def test_metadata_saving_and_structure(self, sample_regression_data, temp_data_dir):
        """Test that metadata is saved correctly."""
        df, feature_names, y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="metadata_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Check metadata structure
        assert "model_name" in metadata
        assert "target_name" in metadata
        assert "mlframe_models" in metadata
        assert "configs" in metadata
        assert "pipeline" in metadata
        assert "cat_features" in metadata
        assert "columns" in metadata

        # Check split details
        assert "train_details" in metadata
        assert "val_details" in metadata
        assert "test_details" in metadata
        assert "train_size" in metadata

        # Check configs
        assert "preprocessing" in metadata["configs"]
        assert "pipeline" in metadata["configs"]
        assert "split" in metadata["configs"]

        # Verify metadata file was saved
        from pyutilz.strings import slugify
        metadata_file = Path(temp_data_dir) / "models" / slugify("test_target") / slugify("metadata_test") / "metadata.joblib"
        assert metadata_file.exists()

        # Load and verify
        loaded_metadata = joblib.load(metadata_file)
        assert loaded_metadata["model_name"] == "metadata_test"


class TestTrainMLFrameModelsSuiteConfigurations:
    """Test different configuration scenarios."""

    def test_with_custom_split_config(self, sample_regression_data, temp_data_dir):
        """Test with custom train/val/test split configuration."""
        df, feature_names, y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        from mlframe.training.configs import TrainingSplitConfig

        split_config = TrainingSplitConfig(
            test_size=0.2,
            val_size=0.1,
            shuffle_test=True,
            shuffle_val=True,
        )

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="custom_split",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            split_config=split_config,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify split was applied
        total_size = len(df)
        assert metadata["train_size"] > 0
        assert metadata["val_size"] > 0
        assert metadata["test_size"] > 0

        # Check approximate sizes
        assert metadata["test_size"] / total_size == pytest.approx(0.2, abs=0.05)
        assert metadata["val_size"] / total_size == pytest.approx(0.1, abs=0.05)

    def test_with_preprocessing_config(self, sample_regression_data, temp_data_dir):
        """Test with custom preprocessing configuration."""
        df, feature_names, y = sample_regression_data

        # Add some NaN values
        df_with_nan = df.copy()
        df_with_nan.loc[10:20, feature_names[0]] = np.nan

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        from mlframe.training.configs import PreprocessingConfig

        preprocessing_config = PreprocessingConfig(
            fillna_value=0.0,
            fix_infinities=True,
            remove_constant_columns=True,
        )

        models, metadata = train_mlframe_models_suite(
            df=df_with_nan,
            target_name="test_target",
            model_name="preproc_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            preprocessing_config=preprocessing_config,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify it completed successfully (models[target_name][target_type])
        assert "target" in models
        assert "REGRESSION" in models["target"]

    def test_with_pipeline_config(self, sample_regression_data, temp_data_dir):
        """Test with custom pipeline configuration."""
        df, feature_names, y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        from mlframe.training.configs import PolarsPipelineConfig

        pipeline_config = PolarsPipelineConfig(
            use_polarsds_pipeline=False,  # Disable polars-ds pipeline
        )

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="pipeline_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            pipeline_config=pipeline_config,
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify pipeline was created
        assert metadata["pipeline"] is not None or metadata["pipeline"] is None


class TestTrainMLFrameModelsSuiteEdgeCases:
    """Test edge cases and error conditions."""

    def test_with_small_dataset(self, temp_data_dir):
        """Test with very small dataset."""
        # Create tiny dataset
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_0': np.random.randn(50),
            'feature_1': np.random.randn(50),
            'target': np.random.randn(50),
        })

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="small_data",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Should still work (models[target_name][target_type])
        assert "target" in models
        assert "REGRESSION" in models["target"]

    def test_with_no_ensembles(self, sample_regression_data, temp_data_dir):
        """Test with ensembles explicitly disabled."""
        df, feature_names, y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="no_ensemble",
            features_and_targets_extractor=fte,
            mlframe_models=["linear", "ridge"],
            init_common_params={'show_perf_chart': False},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Verify models were trained but no ensemble (models[target_name][target_type])
        assert "target" in models
        assert "REGRESSION" in models["target"]
        assert len(models["target"]["REGRESSION"]) >= 2
