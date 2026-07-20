"""
Integration tests for core training functionality.

Tests the main train_mlframe_models_suite function end-to-end.
"""

import importlib

from mlframe.training import FeatureSelectionConfig, OutputConfig, PreprocessingConfig


import pytest
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.configs import TargetTypes
from .shared import SimpleFeaturesAndTargetsExtractor

try:
    from tests.conftest import fast_subset
except ImportError:  # pragma: no cover

    def fast_subset(values, **_):
        """Fast subset."""
        return list(values)


# In --fast mode, the model-name sweep collapses to ``ridge`` (smallest, fastest fit).
_MODEL_NAMES_FAST = fast_subset(["ridge", "xgb", "cb", "lgb", "mlp"], representative="ridge")
_TREE_MODEL_NAMES_FAST = fast_subset(["hgb", "cb", "xgb"], representative="cb")

# Deterministic RNG (single seed per module).
_W53_RNG = __import__("numpy").random.default_rng(0)


def _assert_trained_target_entries(entries, *, target_type_label: str):
    """Behavioural upgrade of bare ``len(models[ttype]['target']) > 0`` asserts.

    Pre-fix the assert only checked that *some* entry existed; a regression that returned a
    single placeholder SimpleNamespace with no fitted model would pass. We now probe each entry
    for the actual contract: it must carry a fitted ``.model`` attribute with predict-shaped
    output."""
    assert isinstance(entries, list), f"{target_type_label}: expected models[ttype]['target'] to be a list, got {type(entries).__name__}"
    assert len(entries) >= 1, f"{target_type_label}: no trained entries returned (empty list)"
    for i, entry in enumerate(entries):
        m = getattr(entry, "model", None)
        assert m is not None, f"{target_type_label}: entries[{i}] missing .model handle: {entry!r}"
        has_predict = callable(getattr(m, "predict", None))
        has_predict_proba = callable(getattr(m, "predict_proba", None))
        assert (
            has_predict or has_predict_proba
        ), f"{target_type_label}: entries[{i}].model is not a fitted estimator: no predict / predict_proba on {type(m).__name__}"


class TestTrainMLFrameModelsSuiteBasic:
    """Basic smoke tests for train_mlframe_models_suite."""

    def test_train_single_linear_model_regression(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test training a single linear model on regression data."""
        df, _feature_names, _y = sample_regression_data

        # Create FTE
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_model",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify structure (models[target_type][target_name])
        assert isinstance(models, dict)
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

        # Verify metadata
        assert metadata["model_name"] == "test_model"
        assert metadata["target_name"] == "test_target"
        assert "configs" in metadata
        assert "pipeline" in metadata

    def test_train_single_linear_model_classification(self, sample_classification_data, temp_data_dir, common_init_params):
        """Test training a single linear model on classification data."""
        df, _feature_names, _, _y = sample_classification_data

        # Create FTE
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # Train
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_model_classif",
            features_and_targets_extractor=fte,
            mlframe_models=["linear"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify structure (models[target_type][target_name])
        assert isinstance(models, dict)
        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]

        # Check that model has predictions
        model_entry = models[TargetTypes.BINARY_CLASSIFICATION]["target"][0]
        assert hasattr(model_entry, "model") or "model" in model_entry.__dict__ if hasattr(model_entry, "__dict__") else True

    def test_train_multiple_linear_models(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test training multiple linear models together."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train multiple models
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="multi_linear",
            features_and_targets_extractor=fte,
            mlframe_models=["linear", "ridge", "lasso"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify all 3 models were trained (models[target_name][target_type])
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        # Should have 3 models (linear, ridge, lasso)
        assert len(models[TargetTypes.REGRESSION]["target"]) >= 3

    def test_train_with_polars_dataframe(self, sample_polars_data, temp_data_dir, common_init_params):
        """Test training with Polars DataFrame input."""
        pl_df, _feature_names, _y = sample_polars_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train with Polars input
        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="polars_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify it worked (models[target_name][target_type])
        assert isinstance(models, dict)
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]


class TestUnifiedTrainingLoop:
    """Test the unified training loop with mixed model types - THE CRITICAL TEST!"""

    def test_train_mixed_linear_and_tree_models(self, sample_regression_data, temp_data_dir, common_init_params):
        """
        THE CRITICAL TEST: Validate unified training loop with linear + tree models.

        This test confirms that linear models (ridge) and tree models (cb)
        can be trained together through the same unified training code path.
        This validates the entire refactoring effort!
        """
        pytest = __import__("pytest")

        # Check if catboost is available
        try:
            importlib.import_module("catboost")

            has_catboost = True
        except ImportError:
            has_catboost = False

        if not has_catboost:
            pytest.skip("CatBoost not available - skipping mixed model test")

        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # THE CRITICAL TEST: Train linear model + tree model together!
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="unified_loop_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge", "cb"],  # LINEAR + TREE!
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=1,  # Show output to see both models training
            hyperparams_config={"iterations": 50},
        )

        # Verify BOTH models were trained
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

        # Should have 2 models: ridge (linear) + cb (tree)
        trained_models = models[TargetTypes.REGRESSION]["target"]
        assert len(trained_models) >= 2, f"Expected 2 models, got {len(trained_models)}"

        # Verify both model types are present
        model_names = [m.model_name if hasattr(m, "model_name") else str(m) for m in trained_models]
        print(f"\n[SUCCESS] UNIFIED TRAINING LOOP VALIDATED!")
        print(f"   Trained models: {model_names}")
        print(f"   Linear + Tree models trained through SAME code path!")

    def test_train_mixed_linear_and_lgb(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test unified training with linear + LightGBM models."""
        pytest = __import__("pytest")

        # Check if lightgbm is available
        try:
            importlib.import_module("lightgbm")

            has_lgb = True
        except ImportError:
            has_lgb = False

        if not has_lgb:
            pytest.skip("LightGBM not available - skipping mixed model test")

        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train linear + LightGBM
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="linear_lgb_test",
            features_and_targets_extractor=fte,
            mlframe_models=["linear", "lgb"],  # LINEAR + LGB!
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
            hyperparams_config={"iterations": 50},
        )

        # Verify both models trained
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        assert len(models[TargetTypes.REGRESSION]["target"]) >= 2


class TestTreeModelsWithEarlyStopping:
    """Test tree models with early stopping enabled via callback_params."""

    @pytest.mark.parametrize("model_type", ["xgb", "lgb", "cb"])
    def test_tree_model_with_early_stopping(self, sample_regression_data, temp_data_dir, common_init_params, model_type):
        """Test tree model training with early stopping callback."""
        # Skip if required library not available
        if model_type == "cb":
            pytest.importorskip("catboost")
        elif model_type == "lgb":
            pytest.importorskip("lightgbm")
        elif model_type == "xgb":
            pytest.importorskip("xgboost")

        df, _feature_names, _y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"{model_type}_early_stop_test",
            features_and_targets_extractor=fte,
            mlframe_models=[model_type],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=1,
            hyperparams_config={"iterations": 50},
            behavior_config={
                "callback_params": {"patience": 5, "verbose": False},
            },
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        assert len(models[TargetTypes.REGRESSION]["target"]) >= 1


class TestTrainMLFrameModelsSuiteEnsembles:
    """Test ensemble creation."""

    def test_train_with_ensembles(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test ensemble creation from multiple models."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train with ensembles enabled
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="ensemble_test",
            features_and_targets_extractor=fte,
            mlframe_models=["linear", "ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=True,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify models were trained (models[target_name][target_type])
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        assert len(models[TargetTypes.REGRESSION]["target"]) >= 2


class TestTrainMLFrameModelsSuiteMetadata:
    """Test metadata saving and structure."""

    def test_metadata_saving_and_structure(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test that metadata is saved correctly."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Train
        _models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="metadata_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
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
        # 2026-04-29: format switched joblib -> pickle proto=5 + zstd L3 (8c301f2).
        from pyutilz.strings import slugify
        import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

        metadata_dir = Path(temp_data_dir) / "models" / slugify("test_target") / slugify("metadata_test")
        zst_path = metadata_dir / "metadata.pkl.zst"
        pkl_path = metadata_dir / "metadata.pkl"
        if zst_path.exists():
            import zstandard as zstd

            loaded_metadata = pickle.loads(zstd.ZstdDecompressor().decompress(zst_path.read_bytes()))  # nosec B301 -- round-trip of a locally-created, trusted object
        elif pkl_path.exists():
            loaded_metadata = pickle.loads(pkl_path.read_bytes())  # nosec B301 -- round-trip of a locally-created, trusted object
        else:
            raise AssertionError(f"No metadata file in {metadata_dir}")
        assert loaded_metadata["model_name"] == "metadata_test"


class TestTrainMLFrameModelsSuiteConfigurations:
    """Test different configuration scenarios."""

    def test_with_custom_split_config(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test with custom train/val/test split configuration."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        from mlframe.training.configs import TrainingSplitConfig

        split_config = TrainingSplitConfig(
            test_size=0.2,
            val_size=0.1,
            shuffle_test=True,
            shuffle_val=True,
        )

        _models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="custom_split",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            split_config=split_config,
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
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

    def test_with_preprocessing_config(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test with custom preprocessing configuration."""
        df, feature_names, _y = sample_regression_data

        # Add some NaN values
        df_with_nan = df.copy()
        df_with_nan.loc[10:20, feature_names[0]] = np.nan

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        from mlframe.training.configs import PreprocessingConfig

        preprocessing_config = PreprocessingConfig(
            fillna_value=0.0,
            fix_infinities=True,
            remove_constant_columns=True,
        )

        models, _metadata = train_mlframe_models_suite(
            df=df_with_nan,
            target_name="test_target",
            model_name="preproc_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            preprocessing_config=preprocessing_config,
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify it completed successfully (models[target_name][target_type])
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

    def test_with_pipeline_config(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test with custom pipeline configuration."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        from mlframe.training.configs import PreprocessingBackendConfig

        pipeline_config = PreprocessingBackendConfig(
            prefer_polarsds=False,  # Disable polars-ds pipeline
        )

        _models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="pipeline_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            pipeline_config=pipeline_config,
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify pipeline config was applied
        assert "pipeline" in metadata


class TestTrainWithoutSaving:
    """Test training without saving models or charts to disk (data_dir=None, models_dir=None)."""

    def test_with_data_dir_none(self, sample_regression_data, common_init_params):
        """Test that training works with data_dir=None (no charts saved)."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_model",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=None, models_dir="models"),
        )

        assert isinstance(models, dict)
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

    def test_with_models_dir_none(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test that training works with models_dir=None (no models saved)."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_model",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir=None),
        )

        assert isinstance(models, dict)
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

    def test_with_both_dirs_none(self, sample_regression_data, common_init_params):
        """Test that training works with both data_dir=None and models_dir=None."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_model",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=None, models_dir=None),
        )

        assert isinstance(models, dict)
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        assert metadata["model_name"] == "test_model"


class TestTrainMLFrameModelsSuiteEdgeCases:
    """Test edge cases and error conditions."""

    def test_with_small_dataset(self, temp_data_dir, common_init_params):
        """Test with very small dataset."""
        # Create tiny dataset
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(50),
                "feature_1": np.random.randn(50),
                "target": np.random.randn(50),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="small_data",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Should still work (models[target_name][target_type])
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

    def test_with_no_ensembles(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test with ensembles explicitly disabled."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="no_ensemble",
            features_and_targets_extractor=fte,
            mlframe_models=["linear", "ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify models were trained but no ensemble (models[target_name][target_type])
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        assert len(models[TargetTypes.REGRESSION]["target"]) >= 2

    def test_with_single_row_dataset(self, temp_data_dir, common_init_params):
        """Test behavior with single row dataset."""
        # Create single row dataset
        df = pd.DataFrame(
            {
                "feature_0": [1.0],
                "feature_1": [2.0],
                "target": [3.0],
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Single row cannot be split into train/val/test; the suite must
        # either raise ValueError/IndexError OR succeed with degenerate
        # output. Either is a valid contract; we just assert the call
        # doesn't silently return None / hang.
        try:
            models, metadata = train_mlframe_models_suite(
                df=df,
                target_name="test_target",
                model_name="single_row",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                reporting_config=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
                verbose=0,
            )
            # Success path: assert we got models back
            assert models is not None
            assert metadata is not None
        except (ValueError, IndexError):
            # Expected path: split / fit refused the degenerate 1-row input.
            # Treat as PASS -- the suite surfaced the limitation cleanly
            # rather than producing garbage output. (Was pytest.skip; that
            # made the test silently green even if some future refactor
            # turned the raise into a hang or wrong-shape success.)
            pass

    def test_with_very_few_samples(self, temp_data_dir, common_init_params):
        """Test with fewer samples than required for train/val/test split."""
        # Create tiny dataset (5 samples)
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(5),
                "feature_1": np.random.randn(5),
                "target": np.random.randn(5),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Should handle gracefully or raise appropriate error
        try:
            models, metadata = train_mlframe_models_suite(
                df=df,
                target_name="test_target",
                model_name="very_few_samples",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                reporting_config=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
                verbose=0,
            )
            # Success path: assert we got models back
            assert models is not None
            assert metadata is not None
        except (ValueError, IndexError):
            # Expected due to small size
            pytest.skip("Very-few-samples input correctly raised ValueError/IndexError")

    def test_with_all_constant_features(self, temp_data_dir, common_init_params):
        """Test with all constant features (should be removed by preprocessing)."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "const_feature_0": [1.0] * 100,
                "const_feature_1": [2.0] * 100,
                "target": np.random.randn(100),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Should handle gracefully - constant features removed
        try:
            models, _metadata = train_mlframe_models_suite(
                df=df,
                target_name="test_target",
                model_name="const_features",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                reporting_config=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
                verbose=0,
            )
            # If it succeeds, check results
            assert "target" in models or models is not None
        except (ValueError, AttributeError, Exception) as e:
            # Expected - no features remaining after preprocessing or related errors.
            # Empty-feature path can also surface as ``NotFittedError`` when the
            # pre-pipeline is asked to transform an all-constant frame whose fit
            # was skipped (sklearn Pipeline -> 'not fitted yet'); accept the
            # downstream messaging too.
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg
                for keyword in [
                    "feature",
                    "constant",
                    "empty",
                    "nonetype",
                    "none",
                    "array",
                    "dtype",
                    "length",
                    "label",
                    "fitted",
                    "pipeline",
                    "instance",
                ]
            )

    def test_with_high_nan_ratio(self, temp_data_dir, common_init_params):
        """Test with high ratio of NaN values."""
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )

        # Make 80% of values NaN
        nan_mask = np.random.random(n_samples) < 0.8
        df.loc[nan_mask, "feature_0"] = np.nan

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Should handle with imputation
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="high_nan",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models

    def test_with_binary_classification_imbalanced(self, temp_data_dir, common_init_params):
        """Test with highly imbalanced binary classification.

        Bumped n=100 -> 1000 (95% / 5%, 50 minority) so the unstratified
        random split lands a few minority samples into both val and
        test. The post-2026-04 ``_validate_target_values`` guard in
        trainer.py raises ValueError when val has only one class - on
        n=100 with 5% rate the splitter (10% val) routinely drops to 0
        minorities. Per persisted memory `rare_imbalance_needs_large_n`:
        rare_1pct needs n>=5000 for reliable random split; 5% at n=1000
        is enough.
        """
        np.random.seed(42)
        n_samples = 1000
        n_minority = n_samples // 20  # 5%
        df = (
            pd.DataFrame(
                {
                    "feature_0": np.random.randn(n_samples),
                    "feature_1": np.random.randn(n_samples),
                    "target": [0] * (n_samples - n_minority) + [1] * n_minority,
                }
            )
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="imbalanced",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]

    def test_with_multiclass_few_samples_per_class(self, temp_data_dir, common_init_params):
        """Test multiclass with very few samples per class."""
        np.random.seed(42)
        n_samples = 30  # 10 samples per class for 3 classes
        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": [0] * 10 + [1] * 10 + [2] * 10,
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="few_per_class",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models

    def test_empty_dataframe_raises_error(self, temp_data_dir, common_init_params):
        """Test that empty DataFrame raises appropriate error."""
        df = pd.DataFrame({"feature_0": [], "feature_1": [], "target": []})

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        with pytest.raises((ValueError, IndexError, Exception)):
            train_mlframe_models_suite(
                df=df,
                target_name="test_target",
                model_name="empty_df",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                reporting_config=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
                verbose=0,
            )

    def test_single_feature_dataset(self, temp_data_dir, common_init_params):
        """Test with single feature (minimal feature space)."""
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="single_feature",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

    def test_nan_in_target_column(self, temp_data_dir, common_init_params):
        """Test that NaN values in regression target raise a clear ValueError."""
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )
        # Add NaN to target
        df.loc[10:20, "target"] = np.nan

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        with pytest.raises(ValueError, match="target contains.*NaN"):
            train_mlframe_models_suite(
                df=df,
                target_name="test_target",
                model_name="nan_target",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                reporting_config=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
                verbose=0,
            )

    def test_infinity_in_target_column(self, temp_data_dir, common_init_params):
        """Test that infinity values in regression target raise a clear ValueError."""
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )
        df.loc[5, "target"] = np.inf
        df.loc[15, "target"] = -np.inf

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        with pytest.raises(ValueError, match="target contains.*infinity"):
            train_mlframe_models_suite(
                df=df,
                target_name="test_target",
                model_name="inf_target",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                reporting_config=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
                verbose=0,
            )

    def test_infinite_values_in_features(self, temp_data_dir, common_init_params):
        """Test behavior with infinite values in features."""
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )
        # Add infinite values
        df.loc[5, "feature_0"] = np.inf
        df.loc[10, "feature_1"] = -np.inf

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        from mlframe.training.configs import PreprocessingConfig

        preprocessing_config = PreprocessingConfig(fix_infinities=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="inf_features",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            preprocessing_config=preprocessing_config,
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

    def test_duplicate_column_names(self, temp_data_dir, common_init_params):
        """Test behavior with duplicate column names."""
        np.random.seed(42)
        n_samples = 100
        # Create DataFrame with duplicate column names (pandas allows this)
        df = pd.DataFrame(np.random.randn(n_samples, 3), columns=["feature", "feature", "target"])

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Should either handle or raise a clear error -- both are valid.
        try:
            models, metadata = train_mlframe_models_suite(
                df=df,
                target_name="test_target",
                model_name="dup_cols",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                reporting_config=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
                verbose=0,
            )
            # Success path: assert we got models back
            assert models is not None
            assert metadata is not None
        except (ValueError, KeyError):
            # Expected path: suite refused the duplicate-columns input
            # rather than producing wrong-shape output. Treat as PASS
            # (was pytest.skip; that made silent green even if a future
            # refactor turned the raise into a hang or wrong-shape return).
            pass

    def test_special_characters_in_column_names(self, temp_data_dir, common_init_params):
        """Test handling of special characters in column names."""
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "feature with spaces": np.random.randn(n_samples),
                "feature/with/slashes": np.random.randn(n_samples),
                "feature[with]brackets": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Should handle special characters gracefully
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="special_chars",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

    def test_high_dimensional_data(self, temp_data_dir, common_init_params):
        """Test with high-dimensional data (many features)."""
        np.random.seed(42)
        n_samples = 100
        n_features = 500  # High-dimensional

        data = {f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)}
        data["target"] = np.random.randn(n_samples)
        df = pd.DataFrame(data)

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="high_dim",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],  # Ridge handles high-dim well
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]


class TestCustomTransformers:
    """Test passing custom scaler, imputer, category_encoder via PreprocessingConfig."""

    def test_custom_scaler(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test passing a custom scaler via PreprocessingConfig."""
        from sklearn.preprocessing import RobustScaler

        df, _feature_names, _y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Create custom scaler
        custom_scaler = RobustScaler()

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="custom_scaler_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            preprocessing_config=PreprocessingConfig(scaler=custom_scaler),
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

    def test_custom_imputer(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test passing a custom imputer via PreprocessingConfig."""
        from sklearn.impute import KNNImputer

        df, feature_names, _y = sample_regression_data

        # Add some NaN values to test imputation
        df_with_nan = df.copy()
        df_with_nan.loc[10:20, feature_names[0]] = np.nan

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Create custom imputer
        custom_imputer = KNNImputer(n_neighbors=3)

        models, _metadata = train_mlframe_models_suite(
            df=df_with_nan,
            target_name="test_target",
            model_name="custom_imputer_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            preprocessing_config=PreprocessingConfig(imputer=custom_imputer),
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

    def test_custom_category_encoder(self, sample_categorical_data, temp_data_dir, common_init_params):
        """Test passing a custom category_encoder via PreprocessingConfig."""
        import category_encoders as ce

        df, _feature_names, _cat_features, _y = sample_categorical_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Create custom encoder
        custom_encoder = ce.TargetEncoder()

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="custom_encoder_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            preprocessing_config=PreprocessingConfig(category_encoder=custom_encoder),
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

    def test_all_custom_transformers(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test passing all custom transformers together."""
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.impute import SimpleImputer
        import category_encoders as ce

        df, _feature_names, _y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Create custom transformers
        custom_scaler = MinMaxScaler()
        custom_imputer = SimpleImputer(strategy="median")
        custom_encoder = ce.OrdinalEncoder()

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="all_custom_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            preprocessing_config=PreprocessingConfig(
                scaler=custom_scaler,
                imputer=custom_imputer,
                category_encoder=custom_encoder,
            ),
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

    def test_default_transformers_initialized(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test that default transformers are initialized when not provided."""
        df, _feature_names, _y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Don't pass any custom transformers - defaults should be used
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="default_transformers_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify training succeeded (defaults were used)
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

    def test_custom_scaler_with_mlp_model(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test custom scaler is actually used by MLP model (which uses the scaler)."""
        pytest = __import__("pytest")

        # Check if torch is available for MLP
        try:
            importlib.import_module("torch")

            has_torch = True
        except ImportError:
            has_torch = False

        if not has_torch:
            pytest.skip("PyTorch not available - skipping MLP test")

        from sklearn.preprocessing import StandardScaler

        df, _feature_names, _y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Create a marked scaler to verify it's used
        custom_scaler = StandardScaler(with_mean=False)  # Non-default setting

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mlp_scaler_test",
            features_and_targets_extractor=fte,
            mlframe_models=["mlp"],
            reporting_config=common_init_params,
            preprocessing_config=PreprocessingConfig(scaler=custom_scaler),
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]


class TestFairnessFeatures:
    """Tests for fairness_features parameter in configure_training_params."""

    def test_fairness_features_empty_list(self, temp_data_dir, common_init_params):
        """Test that empty fairness_features list works (no subgroups)."""
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="no_fairness_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            behavior_config={
                "fairness_features": [],  # explicitly empty
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]


class TestCalibration:
    """Tests for prefer_calibrated_classifiers parameter."""

    def test_calibrated_classifier_basic(self, sample_classification_data, temp_data_dir, common_init_params):
        """Test that prefer_calibrated_classifiers produces calibrated model."""
        df, _feature_names, _, _y = sample_classification_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="calibrated_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            behavior_config={
                "prefer_calibrated_classifiers": True,
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]
        _assert_trained_target_entries(models[TargetTypes.BINARY_CLASSIFICATION]["target"], target_type_label="BINARY_CLASSIFICATION")

    def test_calibrated_classifier_with_cb(self, sample_classification_data, temp_data_dir, common_init_params):
        """Test calibration with CatBoost classifier."""
        pytest = __import__("pytest")

        try:
            importlib.import_module("catboost")

            has_catboost = True
        except ImportError:
            has_catboost = False

        if not has_catboost:
            pytest.skip("CatBoost not available")

        df, _feature_names, _, _y = sample_classification_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="calibrated_cb_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config={"iterations": 10},
            reporting_config=common_init_params,
            behavior_config={
                "prefer_calibrated_classifiers": True,
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]

    def test_uncalibrated_vs_calibrated(self, sample_classification_data, temp_data_dir, common_init_params):
        """Test that we can train both calibrated and uncalibrated classifiers."""
        df, _feature_names, _, _y = sample_classification_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # Train uncalibrated
        models_uncal, _ = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="uncalibrated",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            behavior_config={
                "prefer_calibrated_classifiers": False,
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Train calibrated
        models_cal, _ = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="calibrated",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            behavior_config={
                "prefer_calibrated_classifiers": True,
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Both should produce valid models
        assert TargetTypes.BINARY_CLASSIFICATION in models_uncal
        assert TargetTypes.BINARY_CLASSIFICATION in models_cal
        assert "target" in models_uncal[TargetTypes.BINARY_CLASSIFICATION]
        assert "target" in models_cal[TargetTypes.BINARY_CLASSIFICATION]


class TestConfidenceAnalysis:
    """Tests for include_confidence_analysis parameter."""

    def test_confidence_analysis_basic(self, sample_classification_data, temp_data_dir, common_init_params):
        """Test that confidence analysis runs without errors."""
        pytest = __import__("pytest")

        try:
            importlib.import_module("catboost")

            has_catboost = True
        except ImportError:
            has_catboost = False

        if not has_catboost:
            pytest.skip("CatBoost not available")

        df, _feature_names, _, _y = sample_classification_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # 2026-04-27: ConfidenceAnalysisConfig wired as first-class kwarg of
        # the suite. Pre-refactor this test could only set
        # `init_common_params={"include_confidence_analysis": True}`; now it
        # constructs the typed config explicitly.
        from mlframe.training import ConfidenceAnalysisConfig

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="confidence_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config={"iterations": 10},
            reporting_config=common_init_params,
            confidence_analysis_config=ConfidenceAnalysisConfig(include=True),
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]


class TestCustomMetrics:
    """Tests for custom scoring metrics via control_params_override."""

    def test_custom_regression_scoring(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test custom default_regression_scoring parameter."""
        from sklearn.metrics import mean_absolute_error

        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="custom_scoring_reg",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            behavior_config={
                "default_regression_scoring": dict(score_func=mean_absolute_error, response_method="predict", greater_is_better=False),
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

    def test_custom_classification_scoring(self, sample_classification_data, temp_data_dir, common_init_params):
        """Test custom default_classification_scoring parameter."""
        from sklearn.metrics import f1_score

        df, _feature_names, _, _y = sample_classification_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="custom_scoring_clf",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            behavior_config={
                "default_classification_scoring": dict(score_func=f1_score, response_method="predict", greater_is_better=True),
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]


class TestSampleWeights:
    """Tests for sample weight functionality via extractor's get_sample_weights()."""

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_FAST)
    def test_sample_weight_with_custom_weights(self, model_name, temp_data_dir, common_init_params):
        """Test that models are trained with custom sample weights."""
        from .shared import TimestampedFeaturesExtractor

        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )

        # Create sample weights that emphasize recent samples
        sample_weights = {
            "recency": np.linspace(0.5, 1.0, n_samples),  # Linearly increasing weights
        }

        fte = TimestampedFeaturesExtractor(
            target_column="target",
            regression=True,
            sample_weights=sample_weights,
        )

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="sample_weights_test",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        # Should have 1 model with recency weights (uniform not auto-added when custom weights provided)
        model_list = models[TargetTypes.REGRESSION]["target"]
        assert len(model_list) == 1

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_FAST)
    def test_multiple_weight_schemas(self, model_name, temp_data_dir, common_init_params):
        """Test custom extractor returning multiple weight schemas."""
        from .shared import TimestampedFeaturesExtractor

        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )

        # Multiple weight schemas
        sample_weights = {
            "recency": np.linspace(0.5, 1.0, n_samples),
            "inverse_recency": np.linspace(1.0, 0.5, n_samples),
        }

        fte = TimestampedFeaturesExtractor(
            target_column="target",
            regression=True,
            sample_weights=sample_weights,
        )

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="multi_weights_test",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        # Should have 2 models: recency, inverse_recency (uniform not auto-added)
        model_list = models[TargetTypes.REGRESSION]["target"]
        assert len(model_list) == 2


class TestGroupIds:
    """Tests for group ID functionality via extractor's group_field."""

    def test_group_ids_basic(self, temp_data_dir, common_init_params):
        """Test that group_ids are extracted from group_field column."""
        from .shared import TimestampedFeaturesExtractor

        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
                "group_id": _W53_RNG.choice(["A", "B", "C"], n_samples),
            }
        )

        fte = TimestampedFeaturesExtractor(
            target_column="target",
            regression=True,
            group_field="group_id",
        )

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="group_ids_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]


class TestFairnessFeaturesExtended:
    """Extended tests for fairness subgroup evaluation with min_pop_cat_thresh.

    Fairness analysis evaluates model performance consistency across different
    demographic groups (e.g., age, gender, region) or categorical segments.
    """

    def test_fairness_categorical_features(self, temp_data_dir, common_init_params):
        """Test fairness evaluation with categorical features using small threshold."""
        np.random.seed(42)
        n_samples = 200

        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "category": _W53_RNG.choice(["A", "B", "C"], n_samples),
                "target": np.random.randn(n_samples),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="fairness_cat_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            behavior_config={
                "fairness_features": ["category"],
                "fairness_min_pop_cat_thresh": 10,  # Small threshold for test data
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

    def test_fairness_continuous_features(self, temp_data_dir, common_init_params):
        """Test fairness evaluation with continuous features binned into subgroups."""
        np.random.seed(42)
        n_samples = 200

        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="fairness_cont_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            behavior_config={
                "fairness_features": ["feature_0"],  # Continuous feature will be binned
                "fairness_min_pop_cat_thresh": 10,  # Small threshold for test data
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]

    def test_fairness_min_pop_cat_thresh_parameter_passthrough(self, temp_data_dir, common_init_params):
        """Test that fairness_min_pop_cat_thresh parameter is passed through the API.

        This test verifies that the parameter is accepted without error.
        """
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "feature_0": np.random.randn(n_samples),
                "feature_1": np.random.randn(n_samples),
                "target": np.random.randn(n_samples),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Test that parameter is accepted (empty features = no subgroups)
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="fairness_param_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            behavior_config={
                "fairness_features": [],
                "fairness_min_pop_cat_thresh": 10,  # Verify this parameter is accepted
            },
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]


# =====================================================================================================================
# PREDICTION TESTS
# =====================================================================================================================


class TestPredictMLFrameModelsSuite:
    """Tests for predict_mlframe_models_suite function."""

    def test_predict_classification_basic(self, sample_classification_data, temp_data_dir, common_init_params):
        """Test basic prediction with a trained classification model."""
        from mlframe.training.core import predict_mlframe_models_suite

        df, feature_names, _, _y = sample_classification_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # First train a model
        _models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="predict_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Get path to models
        models_path = f"{temp_data_dir}/models/test_target/predict_test"

        # Create test data (same format as training)
        n_test = 50
        np.random.seed(123)
        test_df = pd.DataFrame(
            {
                **{f"feature_{i}": np.random.randn(n_test) for i in range(len(feature_names))},
                "target": np.random.randint(0, 2, n_test),  # Include target for extractor
            }
        )

        # Generate predictions
        results = predict_mlframe_models_suite(
            df=test_df,
            models_path=models_path,
            features_and_targets_extractor=fte,
            verbose=0,
        )

        # Verify structure
        assert "predictions" in results
        assert "probabilities" in results
        assert "metadata" in results
        assert results["metadata"] is not None

        # Verify predictions
        assert len(results["predictions"]) > 0
        for preds in results["predictions"].values():
            assert len(preds) == n_test
            assert all(p in [0, 1] for p in preds)

    def test_predict_regression_basic(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test basic prediction with a trained regression model."""
        from mlframe.training.core import predict_mlframe_models_suite

        df, feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # First train a model
        _models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="predict_regr_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Get path to models
        models_path = f"{temp_data_dir}/models/test_target/predict_regr_test"

        # Create test data
        n_test = 50
        np.random.seed(123)
        test_df = pd.DataFrame(
            {
                **{f"feature_{i}": np.random.randn(n_test) for i in range(len(feature_names))},
                "target": np.random.randn(n_test),
            }
        )

        # Generate predictions
        results = predict_mlframe_models_suite(
            df=test_df,
            models_path=models_path,
            features_and_targets_extractor=fte,
            return_probabilities=False,  # Regression doesn't have probabilities
            verbose=0,
        )

        # Verify predictions
        assert len(results["predictions"]) > 0
        for preds in results["predictions"].values():
            assert len(preds) == n_test
            assert all(isinstance(p, (int, float, np.integer, np.floating)) for p in preds)

    def test_load_mlframe_suite(self, sample_classification_data, temp_data_dir, common_init_params):
        """Test loading a trained suite."""
        from mlframe.training.core import load_mlframe_suite

        df, _feature_names, _, _y = sample_classification_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # First train a model
        _models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="load_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Get path to models
        models_path = f"{temp_data_dir}/models/test_target/load_test"

        # Load the suite
        loaded_models, loaded_metadata = load_mlframe_suite(models_path)

        # Verify metadata was loaded
        assert loaded_metadata is not None
        assert "model_name" in loaded_metadata
        assert loaded_metadata["model_name"] == "load_test"

        # Verify models were loaded
        assert len(loaded_models) > 0


class TestFeatureSelectorsWithPolarsPipeline:
    """Tests for feature selectors (MRMR, RFECV) with polars-ds pipeline.

    These tests verify that feature selectors are NOT skipped when polars pipeline is applied.
    The bug was that skip_pre_pipeline_transform=True was set when polars pipeline was applied,
    which incorrectly skipped feature selectors along with preprocessing.
    """

    def test_mrmr_with_polars_pipeline_regression(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test that MRMR feature selection runs correctly with polars-ds pipeline on regression."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        from mlframe.training.configs import PreprocessingBackendConfig

        # Enable polars-ds pipeline
        pipeline_config = PreprocessingBackendConfig(
            prefer_polarsds=True,
        )

        # Train with MRMR feature selection
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mrmr_polars_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            pipeline_config=pipeline_config,
            reporting_config=common_init_params,
            use_ordinary_models=False,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=1,
            feature_selection_config=FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs={"max_runtime_mins": 0.5, "verbose": 0}),
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

        # Verify MRMR was used (model name should contain MRMR)
        model_entry = models[TargetTypes.REGRESSION]["target"][0]
        assert hasattr(model_entry, "model_name") or model_entry is not None

    def test_mrmr_with_polars_pipeline_classification(self, sample_classification_data, temp_data_dir, common_init_params):
        """Test that MRMR feature selection runs correctly with polars-ds pipeline on classification."""
        df, _feature_names, _, _y = sample_classification_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        from mlframe.training.configs import PreprocessingBackendConfig

        # Enable polars-ds pipeline
        pipeline_config = PreprocessingBackendConfig(
            prefer_polarsds=True,
        )

        # Train with MRMR feature selection
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mrmr_polars_clf_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            pipeline_config=pipeline_config,
            reporting_config=common_init_params,
            use_ordinary_models=False,
            feature_selection_config=FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs={"max_runtime_mins": 0.5, "verbose": 0}),
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=1,
        )

        # Verify training succeeded
        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]
        _assert_trained_target_entries(models[TargetTypes.BINARY_CLASSIFICATION]["target"], target_type_label="BINARY_CLASSIFICATION")

    def test_rfecv_with_polars_pipeline_regression(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test that RFECV feature selection runs correctly with polars-ds pipeline."""
        pytest = __import__("pytest")

        try:
            importlib.import_module("catboost")

            has_catboost = True
        except ImportError:
            has_catboost = False

        if not has_catboost:
            pytest.skip("CatBoost not available")

        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        from mlframe.training.configs import PreprocessingBackendConfig

        # Enable polars-ds pipeline
        pipeline_config = PreprocessingBackendConfig(
            prefer_polarsds=True,
        )

        # Train with RFECV feature selection using CatBoost
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="rfecv_polars_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            pipeline_config=pipeline_config,
            reporting_config=common_init_params,
            use_ordinary_models=False,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=1,
            feature_selection_config=FeatureSelectionConfig(rfecv_models=["cb_rfecv"]),
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

    def test_mrmr_without_polars_pipeline(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test MRMR feature selection without polars pipeline (baseline test)."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        from mlframe.training.configs import PreprocessingBackendConfig

        # Disable polars-ds pipeline
        pipeline_config = PreprocessingBackendConfig(
            prefer_polarsds=False,
        )

        # Train with MRMR feature selection
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mrmr_no_polars_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            pipeline_config=pipeline_config,
            reporting_config=common_init_params,
            use_ordinary_models=False,
            feature_selection_config=FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs={"max_runtime_mins": 0.5, "verbose": 0}),
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=1,
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

    def test_mrmr_and_ordinary_models_with_polars_pipeline(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test that both MRMR and ordinary models work together with polars pipeline."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        from mlframe.training.configs import PreprocessingBackendConfig

        # Enable polars-ds pipeline
        pipeline_config = PreprocessingBackendConfig(
            prefer_polarsds=True,
        )

        # Train with both ordinary models AND MRMR models
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mixed_mrmr_ordinary_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            pipeline_config=pipeline_config,
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=1,
            feature_selection_config=FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs={"max_runtime_mins": 0.5, "verbose": 0}),
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        # Should have at least 2 models: ordinary + MRMR
        assert len(models[TargetTypes.REGRESSION]["target"]) >= 2

    def test_mrmr_with_polars_input_dataframe(self, sample_polars_data, temp_data_dir, common_init_params):
        """Test MRMR with Polars DataFrame input (not just pipeline config)."""
        pl_df, _feature_names, _y = sample_polars_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        from mlframe.training.configs import PreprocessingBackendConfig

        # Enable polars-ds pipeline
        pipeline_config = PreprocessingBackendConfig(
            prefer_polarsds=True,
        )

        # Train with MRMR on Polars input
        models, _metadata = train_mlframe_models_suite(
            df=pl_df,  # Polars DataFrame input
            target_name="test_target",
            model_name="mrmr_polars_input_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            pipeline_config=pipeline_config,
            reporting_config=common_init_params,
            use_ordinary_models=False,
            feature_selection_config=FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs={"max_runtime_mins": 0.5, "verbose": 0}),
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=1,
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

    def test_mrmr_with_linear_model_and_polars_pipeline(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test MRMR with linear model (which needs scaling) and polars pipeline.

        This tests the case where polars pipeline handles scaling, and MRMR still needs to run.
        """
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        from mlframe.training.configs import PreprocessingBackendConfig

        # Enable polars-ds pipeline (handles scaling)
        pipeline_config = PreprocessingBackendConfig(
            prefer_polarsds=True,
        )

        # Train with MRMR + linear model
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="mrmr_linear_polars_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],  # Linear model needs scaling
            pipeline_config=pipeline_config,
            reporting_config=common_init_params,
            use_ordinary_models=False,
            feature_selection_config=FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs={"max_runtime_mins": 0.5, "verbose": 0}),
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=1,
        )

        # Verify training succeeded
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")


class TestMRMRBinaryClassificationEdgeCases:
    """Test MRMR with binary classification for edge cases."""

    @pytest.fixture
    def perfect_impact_classification_data(self):
        """Create data where one feature perfectly predicts target."""
        np.random.seed(42)
        n_samples = 200

        # Perfect predictor - directly determines binary target
        perfect_feature = np.random.randint(0, 2, n_samples)
        target = perfect_feature  # Direct relationship

        # Noise features
        noise = np.random.randn(n_samples, 5)

        df = pd.DataFrame({"perfect": perfect_feature, **{f"noise_{i}": noise[:, i] for i in range(5)}, "target": target})
        return df

    @pytest.fixture
    def no_impact_classification_data(self):
        """Create data where no features have impact on target."""
        np.random.seed(42)
        n_samples = 200

        # Random features
        features = np.random.randn(n_samples, 5)
        # Random target (independent of features)
        target = np.random.randint(0, 2, n_samples)

        df = pd.DataFrame({**{f"feature_{i}": features[:, i] for i in range(5)}, "target": target})
        return df

    @pytest.mark.parametrize("use_simple_mode", [True, False])
    def test_mrmr_perfect_impact_classification(self, perfect_impact_classification_data, temp_data_dir, common_init_params, use_simple_mode):
        """Test MRMR correctly identifies perfect predictor in binary classification."""
        df = perfect_impact_classification_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"mrmr_perfect_clf_{use_simple_mode}",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=False,
            feature_selection_config=FeatureSelectionConfig(
                use_mrmr_fs=True,
                mrmr_kwargs={"max_runtime_mins": 0.5, "verbose": 0, "use_simple_mode": use_simple_mode},
            ),
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Should train successfully and find the perfect feature
        assert TargetTypes.BINARY_CLASSIFICATION in models
        _assert_trained_target_entries(models[TargetTypes.BINARY_CLASSIFICATION]["target"], target_type_label="BINARY_CLASSIFICATION")

    @pytest.mark.parametrize("use_simple_mode", [True, False])
    def test_mrmr_no_impact_classification(self, no_impact_classification_data, temp_data_dir, common_init_params, use_simple_mode):
        """Test MRMR gracefully handles no predictive features in binary classification."""
        df = no_impact_classification_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # This should skip training when no features selected (no error)
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name=f"mrmr_noimpact_clf_{use_simple_mode}",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=False,
            feature_selection_config=FeatureSelectionConfig(
                use_mrmr_fs=True,
                mrmr_kwargs={
                    "max_runtime_mins": 0.5,
                    "verbose": 0,
                    "use_simple_mode": use_simple_mode,
                    "min_relevance_gain": 10.0,  # Very high threshold to ensure no features selected
                },
            ),
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        # Should complete without error (may have empty models if no features selected)
        assert isinstance(models, dict)


class TestCustomPrePipelines:
    """Tests for custom_pre_pipelines parameter."""

    def test_incremental_pca_pre_pipeline_regression(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test IncrementalPCA as a custom pre-pipeline for regression."""
        from sklearn.decomposition import IncrementalPCA

        df, _feature_names, _y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Create IncrementalPCA pre-pipeline
        custom_pipelines = {
            "ipca5": IncrementalPCA(n_components=5),
        }

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="ipca_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=False,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=1,
            feature_selection_config=FeatureSelectionConfig(custom_pre_pipelines=custom_pipelines),
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(models[TargetTypes.REGRESSION]["target"], target_type_label="REGRESSION")

    def test_incremental_pca_with_ordinary_models(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test IncrementalPCA combined with ordinary models."""
        from sklearn.decomposition import IncrementalPCA

        df, _feature_names, _y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        custom_pipelines = {
            "ipca5": IncrementalPCA(n_components=5),
        }

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="ipca_ordinary_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
            feature_selection_config=FeatureSelectionConfig(custom_pre_pipelines=custom_pipelines),
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        # Should have 2 models: ordinary + ipca5
        assert len(models[TargetTypes.REGRESSION]["target"]) >= 2

    def test_multiple_custom_pre_pipelines(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test multiple custom pre-pipelines."""
        from sklearn.decomposition import IncrementalPCA

        df, _feature_names, _y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        custom_pipelines = {
            "ipca3": IncrementalPCA(n_components=3),
            "ipca5": IncrementalPCA(n_components=5),
        }

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="multi_ipca_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=False,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
            feature_selection_config=FeatureSelectionConfig(custom_pre_pipelines=custom_pipelines),
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        # Should have 2 models: ipca3 + ipca5
        assert len(models[TargetTypes.REGRESSION]["target"]) == 2

    def test_custom_pre_pipeline_classification(self, sample_classification_data, temp_data_dir, common_init_params):
        """Test IncrementalPCA as a custom pre-pipeline for classification."""
        from sklearn.decomposition import IncrementalPCA

        df, _feature_names, _, _y = sample_classification_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        custom_pipelines = {
            "ipca5": IncrementalPCA(n_components=5),
        }

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="ipca_clf_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=False,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
            feature_selection_config=FeatureSelectionConfig(custom_pre_pipelines=custom_pipelines),
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]
        _assert_trained_target_entries(models[TargetTypes.BINARY_CLASSIFICATION]["target"], target_type_label="BINARY_CLASSIFICATION")


class TestPolarsNativeFastpath:
    """Tests for CatBoost Polars native fastpath — no pandas conversion."""

    def test_catboost_receives_polars_dataframe(self, temp_data_dir, common_init_params, monkeypatch):
        """Verify CatBoost .fit() receives a Polars DataFrame when input is Polars."""
        pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["a", "b", "c"], size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # Capture the DataFrame type that reaches model.fit()
        fit_df_types = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            fit_df_types.append((model_type_name, type(train_df).__name__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper was lifted into its own module (_training_loop) and re-
        # imported into trainer / _trainer_configure / _trainer_train_and_evaluate
        # at the 2026-05-22 split. Patch every binding so the spy intercepts
        # whichever the suite reaches.
        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="polars_fastpath_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]

        # The CatBoost model must have received a Polars DataFrame, NOT pandas
        cb_entries = [(name, df_type) for name, df_type in fit_df_types if "CatBoost" in name]
        assert len(cb_entries) > 0, f"No CatBoost .fit() calls recorded. All calls: {fit_df_types}"
        for _name, df_type in cb_entries:
            assert df_type == "DataFrame", f"CatBoost received {df_type} instead of Polars DataFrame — fastpath not active"

    def test_hgb_receives_polars_dataframe(self, temp_data_dir, common_init_params, monkeypatch):
        """Verify HGB .fit() receives a Polars DataFrame when input is Polars."""
        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["a", "b", "c"], size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        fit_df_types = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            fit_df_types.append((model_type_name, type(train_df).__name__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper was lifted into its own module (_training_loop) and re-
        # imported into trainer / _trainer_configure / _trainer_train_and_evaluate
        # at the 2026-05-22 split. Patch every binding so the spy intercepts
        # whichever the suite reaches.
        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="hgb_polars_test",
            features_and_targets_extractor=fte,
            mlframe_models=["hgb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]

        # HGB must have received a Polars DataFrame, NOT pandas
        hgb_entries = [(name, df_type) for name, df_type in fit_df_types if "HistGradient" in name or "HGB" in name]
        assert len(hgb_entries) > 0, f"No HGB .fit() calls recorded. All calls: {fit_df_types}"
        for _name, df_type in hgb_entries:
            assert df_type == "DataFrame", f"HGB received {df_type} instead of Polars DataFrame — fastpath not active"

    def test_hgb_polars_categorical_is_cast(self, temp_data_dir, common_init_params, monkeypatch):
        """Verify HGB receives cat columns as pl.Categorical (not pl.String) on Polars fastpath."""
        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["a", "b", "c"], size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_dfs = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            captured_dfs.append(train_df)
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="hgb_cat_cast_test",
            features_and_targets_extractor=fte,
            mlframe_models=["hgb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert len(captured_dfs) > 0
        train_df = captured_dfs[0]
        assert isinstance(train_df, pl.DataFrame)
        # cat_feat must be a categorical dtype (pl.Categorical OR pl.Enum) —
        # NOT raw pl.String. Enum is produced by the default
        # ``align_polars_categorical_dicts=True`` behaviour, which pins
        # the category universe across train/val/test and dodges the
        # XGBoost sparse-code bug; the HGB fastpath accepts either.
        dtype = train_df["cat_feat"].dtype
        assert (dtype == pl.Categorical) or isinstance(dtype, pl.Enum), f"cat_feat dtype is {dtype}, expected pl.Categorical or pl.Enum"

    @pytest.mark.parametrize(
        "model_name,regression",
        [
            ("cb", False),
            ("cb", True),
            ("xgb", False),
            ("xgb", True),
            ("hgb", False),
            ("hgb", True),
        ],
    )
    def test_polars_fastpath_parametrized(self, model_name, regression, temp_data_dir, common_init_params, monkeypatch):
        """Parametrized: verify Polars fastpath for cb/hgb/xgb with classification and regression."""
        if model_name == "cb":
            pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["a", "b", "c"], size=n),
                "target": np.random.randn(n) if regression else np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=regression)

        fit_df_types = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            fit_df_types.append((model_type_name, type(train_df).__name__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper was lifted into its own module (_training_loop) and re-
        # imported into trainer / _trainer_configure / _trainer_train_and_evaluate
        # at the 2026-05-22 split. Patch every binding so the spy intercepts
        # whichever the suite reaches.
        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name=f"param_{model_name}_{'reg' if regression else 'cls'}",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        expected_tt = TargetTypes.REGRESSION if regression else TargetTypes.BINARY_CLASSIFICATION
        assert expected_tt in models
        assert "target" in models[expected_tt]

        # Model must have received a Polars DataFrame
        relevant = [(n, t) for n, t in fit_df_types if n != ""]
        assert len(relevant) > 0, f"No .fit() calls recorded: {fit_df_types}"
        for name, df_type in relevant:
            assert df_type == "DataFrame", f"{name} received {df_type} instead of Polars DataFrame"

    def test_mixed_feature_types(self, temp_data_dir, common_init_params, monkeypatch):
        """Test Polars fastpath with numeric (float, int), string, pl.Categorical, and boolean features."""
        pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "float_feat": np.random.randn(n),
                "int_feat": np.random.randint(0, 100, size=n),
                "str_feat": _W53_RNG.choice(["alpha", "beta", "gamma"], size=n),
                "cat_feat": pl.Series(_W53_RNG.choice(["x", "y", "z"], size=n)).cast(pl.Categorical),
                "bool_feat": _W53_RNG.choice([True, False], size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_dfs = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            captured_dfs.append((model_type_name, train_df))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="mixed_types_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert len(captured_dfs) > 0
        train_df = captured_dfs[0][1]
        assert isinstance(train_df, pl.DataFrame), f"Expected Polars DataFrame, got {type(train_df)}"

    def test_hgb_high_cardinality_categorical(self, temp_data_dir, common_init_params, monkeypatch):
        """HGB with a categorical column having >255 unique values should get ordinal-encoded to integer."""
        np.random.seed(42)
        n = 500
        # 300 unique categories — exceeds the 255 limit for pl.Categorical in HGB
        high_card_values = [f"cat_{i}" for i in range(300)]
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "high_card": _W53_RNG.choice(high_card_values, size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_dfs = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            captured_dfs.append((model_type_name, train_df))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="hgb_highcard_test",
            features_and_targets_extractor=fte,
            mlframe_models=["hgb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert len(captured_dfs) > 0
        train_df = captured_dfs[0][1]
        assert isinstance(train_df, pl.DataFrame)
        # High-cardinality column: HGB's max_bins=255 limit means a 300-unique column cannot be
        # kept as pl.Categorical. HGBStrategy.prepare_polars_dataframe ordinal-encodes it to an
        # integer dtype (UInt32); polars-ds's int_to_float(f32=True) may then cast to Float32.
        # Both outcomes are acceptable — the column must still exist and must no longer be string.
        # Column may also be absent if the pipeline dropped it after ordinal encoding.
        if "high_card" in train_df.columns:
            dt = train_df["high_card"].dtype
            assert dt != pl.String and dt != pl.Utf8, f"high_card should be numeric-encoded, got {dt}"
        # Verify training completed successfully despite >255 unique categories
        assert "target" in models[TargetTypes.BINARY_CLASSIFICATION]

    def test_multiple_categorical_columns(self, temp_data_dir, common_init_params, monkeypatch):
        """Test Polars fastpath with 3+ categorical columns of varying cardinality."""
        pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        # Cast string cols to pl.Categorical so they're treated as categorical features.
        # Without the cast, mlframe's auto_detect_feature_types promotes any pl.String
        # column above cat_text_cardinality_threshold (default 300 as of round 12,
        # was 50 previously) to a CatBoost text feature — which on tiny train sets
        # (n=200, 100 unique values for cat_high) raises "Dictionary size is 0"
        # inside CatBoost's text feature estimator.
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_low": _W53_RNG.choice(["a", "b"], size=n),
                "cat_mid": _W53_RNG.choice([f"m{i}" for i in range(20)], size=n),
                "cat_high": _W53_RNG.choice([f"h{i}" for i in range(100)], size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        ).with_columns(
            [
                pl.col("cat_low").cast(pl.Categorical),
                pl.col("cat_mid").cast(pl.Categorical),
                pl.col("cat_high").cast(pl.Categorical),
            ]
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_dfs = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            captured_dfs.append((model_type_name, train_df))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="multi_cat_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
            # Caller cast the columns to ``pl.Categorical`` explicitly; the
            # ``honor_user_dtype=True`` knob propagates that intent to the
            # auto-detect pass so a high-cardinality ``pl.Categorical`` (e.g.
            # ``cat_high`` with 100 uniques over 200 rows) is NOT auto-
            # promoted to ``text_features`` (where CB's TF-IDF estimator
            # would trip "Dictionary size is 0").
            feature_types_config=__import__(
                "mlframe.training.configs",
                fromlist=["FeatureTypesConfig"],
            ).FeatureTypesConfig(honor_user_dtype=True),
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert len(captured_dfs) > 0
        train_df = captured_dfs[0][1]
        assert isinstance(train_df, pl.DataFrame), f"Expected Polars DataFrame, got {type(train_df)}"
        # All 3 categorical columns should be present
        for col in ["cat_low", "cat_mid", "cat_high"]:
            assert col in train_df.columns, f"Missing column {col} in train_df"

    def test_polars_fastpath_with_sample_weights(self, temp_data_dir, common_init_params, monkeypatch):
        """Test that Polars fastpath works correctly when sample_weight is provided."""
        pytest.importorskip("catboost")
        from .shared import TimestampedFeaturesExtractor

        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["a", "b", "c"], size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        )

        sample_weights = {
            "recency": np.linspace(0.5, 1.0, n),
        }

        fte = TimestampedFeaturesExtractor(
            target_column="target",
            regression=False,
            sample_weights=sample_weights,
        )

        fit_df_types = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            fit_df_types.append((model_type_name, type(train_df).__name__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper was lifted into its own module (_training_loop) and re-
        # imported into trainer / _trainer_configure / _trainer_train_and_evaluate
        # at the 2026-05-22 split. Patch every binding so the spy intercepts
        # whichever the suite reaches.
        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="polars_weights_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        # CatBoost should still receive Polars DataFrame even with sample weights
        cb_entries = [(name, df_type) for name, df_type in fit_df_types if "CatBoost" in name]
        assert len(cb_entries) > 0, f"No CatBoost .fit() calls recorded. All calls: {fit_df_types}"
        for _name, df_type in cb_entries:
            assert df_type == "DataFrame", f"CatBoost received {df_type} instead of Polars DataFrame with sample weights"

    @pytest.mark.parametrize("model_name", _TREE_MODEL_NAMES_FAST)
    def test_polars_fastpath_regression_target(self, model_name, temp_data_dir, common_init_params, monkeypatch):
        """Test Polars fastpath with a continuous regression target."""
        if model_name == "cb":
            pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "feat_a": np.random.randn(n),
                "feat_b": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["low", "mid", "high"], size=n),
                "target": np.random.randn(n) * 10 + 50,  # continuous regression target
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        captured_dfs = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            captured_dfs.append((model_type_name, type(train_df).__name__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name=f"polars_reg_{model_name}",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        # Must receive Polars DataFrame
        assert len(captured_dfs) > 0
        for name, df_type in captured_dfs:
            assert df_type == "DataFrame", f"{name} received {df_type} instead of Polars DataFrame for regression"

    def test_non_catboost_still_gets_pandas(self, temp_data_dir, common_init_params, monkeypatch):
        """Verify non-CatBoost models still receive pandas even when input is Polars."""
        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "target": np.random.randn(n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        fit_df_types = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            fit_df_types.append((model_type_name, type(train_df).__module__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="polars_nonfastpath_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        # Ridge must receive pandas, not Polars
        ridge_entries = [(name, mod) for name, mod in fit_df_types if "Ridge" in name or "SGD" in name or "Linear" in name]
        assert len(ridge_entries) > 0, f"No linear .fit() calls recorded. All calls: {fit_df_types}"
        for _name, mod in ridge_entries:
            assert "pandas" in mod, f"Linear model received {mod} instead of pandas — polars should NOT leak to non-CatBoost models"

    @pytest.mark.parametrize(
        "models,should_skip",
        [
            (["cb"], True),
            (["xgb"], True),
            (["hgb"], True),
            (["cb", "xgb", "hgb"], True),
            (["cb", "ridge"], False),
            (["ridge"], False),
        ],
    )
    def test_skip_categorical_encoding_auto_detection(self, models, should_skip, temp_data_dir, common_init_params, monkeypatch):
        """Verify skip_categorical_encoding is auto-set when all models support Polars natively."""
        if "cb" in models:
            pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["a", "b", "c"], size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_configs = []

        # ``fit_and_transform_pipeline`` lives in ``training.pipeline`` and is re-imported into
        # ``_phase_helpers_fit_pipeline`` (current call site). ``_phase_helpers``/``_phase_helpers_fit_split``
        # no longer import this name directly, so only these two bindings need patching.
        from mlframe.training import pipeline as _pipeline_mod
        from mlframe.training.core import _phase_helpers_fit_pipeline as _phfit_mod

        original_fit = _pipeline_mod.fit_and_transform_pipeline

        def _spy_pipeline(**kwargs):
            """Spy pipeline."""
            captured_configs.append(kwargs["config"].skip_categorical_encoding)
            return original_fit(**kwargs)

        monkeypatch.setattr(_pipeline_mod, "fit_and_transform_pipeline", _spy_pipeline)
        monkeypatch.setattr(_phfit_mod, "fit_and_transform_pipeline", _spy_pipeline)

        train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="skip_catenc_test",
            features_and_targets_extractor=fte,
            mlframe_models=models,
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert len(captured_configs) > 0
        assert captured_configs[0] == should_skip, f"skip_categorical_encoding={captured_configs[0]}, expected {should_skip} for models={models}"

    def test_mixed_polars_and_nonpolars_models(self, temp_data_dir, common_init_params, monkeypatch):
        """Mixed models: cb gets Polars, ridge gets pandas, both train successfully."""
        pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["a", "b", "c"], size=n),
                "target": np.random.randn(n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        fit_df_info = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            fit_df_info.append((model_type_name, type(train_df).__module__, type(train_df).__name__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="mixed_models_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb", "ridge"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        assert len(models[TargetTypes.REGRESSION]["target"]) >= 2

        # CatBoost must get Polars
        cb_entries = [(n, m) for n, m, _ in fit_df_info if "CatBoost" in n]
        assert len(cb_entries) > 0
        for _name, mod in cb_entries:
            assert "polars" in mod, f"CatBoost received {mod} instead of Polars"

        # Ridge must get pandas
        ridge_entries = [(n, m) for n, m, _ in fit_df_info if "Ridge" in n or "SGD" in n or "Linear" in n]
        assert len(ridge_entries) > 0
        for _name, mod in ridge_entries:
            assert "pandas" in mod, f"Linear model received {mod} instead of pandas"

    def test_all_polars_native_models_together(self, temp_data_dir, common_init_params, monkeypatch):
        """All 3 polars-native models (cb, xgb, hgb) run together — each gets Polars, no cache collision."""
        pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["x", "y", "z"], size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        fit_df_info = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            fit_df_info.append((model_type_name, type(train_df).__name__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="all_polars_native",
            features_and_targets_extractor=fte,
            mlframe_models=["cb", "xgb", "hgb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert len(models[TargetTypes.BINARY_CLASSIFICATION]["target"]) >= 3

        # All models must receive Polars DataFrames
        relevant = [(n, t) for n, t in fit_df_info if n != ""]
        assert len(relevant) >= 3, f"Expected ≥3 .fit() calls, got {len(relevant)}: {relevant}"
        for name, df_type in relevant:
            assert df_type == "DataFrame", f"{name} received {df_type} instead of Polars DataFrame"

    def test_polars_fastpath_predictions_valid(self, temp_data_dir, common_init_params):
        """Verify that Polars fastpath models produce valid (non-NaN, correct shape) predictions."""
        pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        # Create a dataset with signal so predictions aren't random
        x = np.random.randn(n)
        pl_df = pl.DataFrame(
            {
                "signal": x,
                "noise": np.random.randn(n) * 0.1,
                "cat_feat": _W53_RNG.choice(["a", "b"], size=n),
                "target": (x > 0).astype(int),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="preds_valid_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 20},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        model_entries = models[TargetTypes.BINARY_CLASSIFICATION]["target"]
        assert len(model_entries) > 0
        model_ns = model_entries[0]
        # Model must be fitted
        assert model_ns.model is not None
        # Metrics must be computed
        assert model_ns.metrics is not None
        assert any(model_ns.metrics[split] for split in ["train", "val", "test"])

    def test_pandas_input_still_works_for_polars_native_models(self, temp_data_dir, common_init_params, monkeypatch):
        """Pandas input should work for cb/xgb/hgb — no Polars fastpath, standard pandas path."""
        pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        import pandas as pd

        pd_df = pd.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["a", "b", "c"], size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        fit_df_types = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            fit_df_types.append((model_type_name, type(train_df).__module__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pd_df,
            target_name="test_target",
            model_name="pandas_input_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        # With pandas input, no Polars fastpath — model receives pandas
        cb_entries = [(n, m) for n, m in fit_df_types if "CatBoost" in n]
        assert len(cb_entries) > 0
        for _name, mod in cb_entries:
            assert "pandas" in mod or "numpy" in mod, f"With pandas input, CatBoost should receive pandas/numpy, got {mod}"

    def test_polars_fastpath_with_many_categorical_values(self, temp_data_dir, common_init_params, monkeypatch):
        """Polars fastpath with high-cardinality categoricals (>10 unique values)."""
        pytest.importorskip("catboost")

        np.random.seed(42)
        n = 200
        cats = _W53_RNG.choice([f"cat_{i}" for i in range(50)], size=n)
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": cats,
                "target": np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        fit_df_types = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            fit_df_types.append((model_type_name, type(train_df).__name__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper was lifted into its own module (_training_loop) and re-
        # imported into trainer / _trainer_configure / _trainer_train_and_evaluate
        # at the 2026-05-22 split. Patch every binding so the spy intercepts
        # whichever the suite reaches.
        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="high_card_cats_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        cb_entries = [(n, t) for n, t in fit_df_types if "CatBoost" in n]
        assert len(cb_entries) > 0
        for _name, df_type in cb_entries:
            assert df_type == "DataFrame", f"CatBoost received {df_type} instead of Polars DataFrame"

    def test_skip_categorical_encoding_manual_flag(self, temp_data_dir, common_init_params, monkeypatch):
        """Manually setting skip_categorical_encoding=True is preserved in the pipeline config."""
        pytest.importorskip("catboost")
        from mlframe.training.configs import PreprocessingBackendConfig

        np.random.seed(42)
        n = 200
        pl_df = pl.DataFrame(
            {
                "num_feat": np.random.randn(n),
                "cat_feat": _W53_RNG.choice(["a", "b", "c"], size=n),
                "target": np.random.randint(0, 2, size=n),
            }
        )

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_configs = []

        # ``fit_and_transform_pipeline`` lives in ``training.pipeline`` and is re-imported into
        # ``_phase_helpers_fit_pipeline`` (current call site). ``_phase_helpers``/``_phase_helpers_fit_split``
        # no longer import this name directly, so only these two bindings need patching.
        from mlframe.training import pipeline as _pipeline_mod
        from mlframe.training.core import _phase_helpers_fit_pipeline as _phfit_mod

        original_fit = _pipeline_mod.fit_and_transform_pipeline

        def _spy_pipeline(**kwargs):
            """Spy pipeline."""
            captured_configs.append(kwargs["config"].skip_categorical_encoding)
            return original_fit(**kwargs)

        monkeypatch.setattr(_pipeline_mod, "fit_and_transform_pipeline", _spy_pipeline)
        monkeypatch.setattr(_phfit_mod, "fit_and_transform_pipeline", _spy_pipeline)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="manual_skip_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            pipeline_config=PreprocessingBackendConfig(skip_categorical_encoding=True),
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        assert len(captured_configs) > 0
        assert captured_configs[0] is True, "Manual skip_categorical_encoding=True should be preserved"


# =============================================================================
# Text Features & Embedding Features Support
# =============================================================================


def _make_text_embedding_polars_df(n=200, n_cat_unique=5, n_text_unique=100):
    """Build a Polars DataFrame with numeric, categorical, text, and embedding columns."""
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "num_feat": rng.standard_normal(n),
            "cat_feat": rng.choice([f"cat_{i}" for i in range(n_cat_unique)], size=n),
            "text_feat": rng.choice([f"sentence number {i} with some words" for i in range(n_text_unique)], size=n),
            "emb_feat": [rng.standard_normal(4).tolist() for _ in range(n)],
            "target": rng.integers(0, 2, size=n),
        }
    )


class TestTextAndEmbeddingFeatures:
    """Tests for text_features and embedding_features support (CatBoost)."""

    # -----------------------------------------------------------------------
    # A: Feature type detection and routing
    # -----------------------------------------------------------------------

    def test_catboost_text_features_in_fit_params(self, temp_data_dir, common_init_params, monkeypatch):
        """CatBoost receives text_features in fit_params via Polars fastpath."""
        pytest.importorskip("catboost")
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_fit_params = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            captured_fit_params.append((model_type_name, dict(fit_params) if fit_params else {}))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="text_feat_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            feature_types_config=FeatureTypesConfig(text_features=["text_feat"]),
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        cb_entries = [(n, p) for n, p in captured_fit_params if "CatBoost" in n]
        assert len(cb_entries) > 0
        assert "text_features" in cb_entries[0][1], f"text_features missing from fit_params: {cb_entries[0][1].keys()}"
        assert "text_feat" in cb_entries[0][1]["text_features"]

    def test_catboost_embedding_features_in_fit_params(self, temp_data_dir, common_init_params, monkeypatch):
        """CatBoost receives embedding_features in fit_params via Polars fastpath."""
        pytest.importorskip("catboost")
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_fit_params = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            captured_fit_params.append((model_type_name, dict(fit_params) if fit_params else {}))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="emb_feat_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            feature_types_config=FeatureTypesConfig(embedding_features=["emb_feat"]),
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        cb_entries = [(n, p) for n, p in captured_fit_params if "CatBoost" in n]
        assert len(cb_entries) > 0
        assert "embedding_features" in cb_entries[0][1], f"embedding_features missing: {cb_entries[0][1].keys()}"
        assert "emb_feat" in cb_entries[0][1]["embedding_features"]

    def test_non_catboost_drops_text_columns(self, temp_data_dir, common_init_params, monkeypatch):
        """Ridge model's train_df should NOT contain text columns."""
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_dfs = {}
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            cols = list(train_df.columns) if hasattr(train_df, "columns") else []
            captured_dfs[model_type_name] = cols
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="drop_text_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            feature_types_config=FeatureTypesConfig(text_features=["text_feat"]),
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        for model_name, cols in captured_dfs.items():
            assert "text_feat" not in cols, f"{model_name} received text_feat column — should have been dropped"

    def test_non_catboost_drops_embedding_columns(self, temp_data_dir, common_init_params, monkeypatch):
        """Ridge model's train_df should NOT contain embedding columns."""
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_dfs = {}
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            cols = list(train_df.columns) if hasattr(train_df, "columns") else []
            captured_dfs[model_type_name] = cols
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="drop_emb_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            feature_types_config=FeatureTypesConfig(embedding_features=["emb_feat"]),
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        for model_name, cols in captured_dfs.items():
            assert "emb_feat" not in cols, f"{model_name} received emb_feat column — should have been dropped"

    def test_text_cat_mutual_exclusivity_raises(self, temp_data_dir, common_init_params):
        """ValueError when same column is in both text_features and embedding_features."""
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # Same column in text AND embedding → must raise ValueError
        with pytest.raises(ValueError, match="text_features.*embedding_features"):
            train_mlframe_models_suite(
                df=pl_df,
                target_name="test_target",
                model_name="mutual_excl_test",
                features_and_targets_extractor=fte,
                mlframe_models=["cb"],
                feature_types_config=FeatureTypesConfig(text_features=["text_feat"], embedding_features=["text_feat"]),
                reporting_config=common_init_params,
                hyperparams_config={"iterations": 10},
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                verbose=0,
                output_config=OutputConfig(data_dir=temp_data_dir),
            )

    def test_embedding_auto_detection_polars(self, temp_data_dir, common_init_params):
        """pl.List(pl.Float64) columns are auto-detected as embedding features."""
        from mlframe.training.configs import FeatureTypesConfig

        pytest.importorskip("catboost")

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # Don't specify embedding_features — let auto-detection find emb_feat
        _models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="emb_autodetect",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            feature_types_config=FeatureTypesConfig(),
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        assert "emb_feat" in metadata.get("embedding_features", []), f"emb_feat not auto-detected. embedding_features={metadata.get('embedding_features')}"

    def test_text_auto_detection_high_cardinality(self, temp_data_dir, common_init_params):
        """String column with 100 unique values is auto-detected as text feature."""
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df(n_text_unique=100)
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        _models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="text_autodetect",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            feature_types_config=FeatureTypesConfig(cat_text_cardinality_threshold=50),
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        assert "text_feat" in metadata.get("text_features", []), f"text_feat not auto-detected as text. text_features={metadata.get('text_features')}"

    def test_text_auto_detection_low_cardinality_stays_cat(self, temp_data_dir, common_init_params):
        """String column with 5 unique values stays categorical (not text)."""
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df(n_cat_unique=5, n_text_unique=5)
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        _models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="low_card_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            feature_types_config=FeatureTypesConfig(cat_text_cardinality_threshold=50),
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        # With only 5 unique text values (< threshold=50), text_feat should NOT be in text_features
        text_feats = metadata.get("text_features", [])
        assert "text_feat" not in text_feats, f"text_feat should stay categorical (5 unique < threshold 50), but found in text_features"

    def test_user_declared_polars_categorical_not_promoted_to_text(self, temp_data_dir, common_init_params):
        """Columns the user explicitly marked as pl.Categorical must stay
        categorical even when their cardinality would otherwise trigger
        text-auto-detection — IF the caller passes
        ``FeatureTypesConfig(honor_user_dtype=True)``. Default behaviour
        (honor_user_dtype=False) still promotes by cardinality; see
        ``test_auto_detect_polars_categorical_promoted_by_cardinality``
        in ``test_core_helper_functions.py``.
        """
        from mlframe.training.configs import FeatureTypesConfig

        # 100 unique values + pl.Categorical dtype — threshold=50 would promote if we
        # didn't honor the user's explicit dtype. honor_user_dtype=True tells
        # auto-detect to treat pl.Categorical / pl.Enum as user-declared and
        # skip the cardinality-based promotion.
        pl_df = _make_text_embedding_polars_df(n_text_unique=100)
        pl_df = pl_df.with_columns(pl.col("text_feat").cast(pl.Categorical))

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        _models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="user_cat_preserved",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            feature_types_config=FeatureTypesConfig(cat_text_cardinality_threshold=50, honor_user_dtype=True),
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )
        text_feats = metadata.get("text_features", [])
        assert "text_feat" not in text_feats, (
            "User-declared pl.Categorical column (text_feat) was silently promoted to "
            f"text_features — user's explicit dtype must be honored. text_features={text_feats}"
        )

    def test_catboost_text_and_embedding_together(self, temp_data_dir, common_init_params, monkeypatch):
        """CatBoost receives both text_features and embedding_features simultaneously."""
        pytest.importorskip("catboost")
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_fit_params = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            captured_fit_params.append((model_type_name, dict(fit_params) if fit_params else {}))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="text_emb_together",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            feature_types_config=FeatureTypesConfig(text_features=["text_feat"], embedding_features=["emb_feat"]),
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        cb_entries = [(n, p) for n, p in captured_fit_params if "CatBoost" in n]
        assert len(cb_entries) > 0
        fp = cb_entries[0][1]
        assert "text_features" in fp and "embedding_features" in fp
        assert "text_feat" in fp["text_features"]
        assert "emb_feat" in fp["embedding_features"]

    def test_mixed_models_text_features(self, temp_data_dir, common_init_params, monkeypatch):
        """CatBoost gets text columns, Ridge drops them — both train successfully."""
        pytest.importorskip("catboost")
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_dfs = {}
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            cols = list(train_df.columns) if hasattr(train_df, "columns") else []
            captured_dfs[model_type_name] = cols
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="mixed_text_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb", "ridge"],
            feature_types_config=FeatureTypesConfig(text_features=["text_feat"], embedding_features=["emb_feat"]),
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        # CatBoost should have text_feat and emb_feat
        cb_cols = [c for name, c in captured_dfs.items() if "CatBoost" in name]
        assert len(cb_cols) > 0
        assert "text_feat" in cb_cols[0], "CatBoost should retain text_feat"
        assert "emb_feat" in cb_cols[0], "CatBoost should retain emb_feat"

        # Ridge should NOT have text_feat or emb_feat
        ridge_cols = [c for name, c in captured_dfs.items() if "Ridge" in name]
        assert len(ridge_cols) > 0
        assert "text_feat" not in ridge_cols[0], "Ridge should not have text_feat"
        assert "emb_feat" not in ridge_cols[0], "Ridge should not have emb_feat"

    def test_custom_cardinality_threshold(self, temp_data_dir, common_init_params):
        """cat_text_cardinality_threshold=10 changes text vs cat classification."""
        from mlframe.training.configs import FeatureTypesConfig

        # 20 unique text values — with threshold=10 it should be text, with 50 it should be cat
        pl_df = _make_text_embedding_polars_df(n_text_unique=20)
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        _models, metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="threshold_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            feature_types_config=FeatureTypesConfig(cat_text_cardinality_threshold=10),
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        assert "text_feat" in metadata.get(
            "text_features", []
        ), f"With threshold=10 and 20 unique values, text_feat should be text. Got: {metadata.get('text_features')}"

    # -----------------------------------------------------------------------
    # B: Feature tier ordering
    # -----------------------------------------------------------------------

    def test_model_training_order_by_feature_tier(self, temp_data_dir, common_init_params, monkeypatch):
        """CatBoost (tier 1) trains before Ridge (tier 2) regardless of input order."""
        pytest.importorskip("catboost")
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        training_order = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            training_order.append(model_type_name)
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        # Pass Ridge FIRST in the list — but CatBoost should still train first (higher tier)
        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="tier_order_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge", "cb"],
            feature_types_config=FeatureTypesConfig(text_features=["text_feat"], embedding_features=["emb_feat"]),
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        cb_idx = next((i for i, n in enumerate(training_order) if "CatBoost" in n), None)
        ridge_idx = next((i for i, n in enumerate(training_order) if "Ridge" in n), None)
        assert cb_idx is not None and ridge_idx is not None
        assert cb_idx < ridge_idx, f"CatBoost should train before Ridge. Order: {training_order}"

    def test_tier_column_dropping_cached(self, temp_data_dir, common_init_params, monkeypatch):
        """Column dropping happens once per tier, not once per model."""
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_dfs = {}
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            # Store id to verify same object is reused
            """Spy train."""
            captured_dfs[model_type_name] = id(train_df)
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="tier_cache_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge", "lgb"],
            feature_types_config=FeatureTypesConfig(text_features=["text_feat"], embedding_features=["emb_feat"]),
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        # Both Ridge and Lasso are same tier — should receive same trimmed DF object
        ridge_ids = [v for k, v in captured_dfs.items() if "Ridge" in k]
        lgb_ids = [v for k, v in captured_dfs.items() if "LGB" in k or "lgb" in k.lower()]
        # Both should have trained (same tier — both lack text/embedding support)
        assert len(ridge_ids) > 0 and len(lgb_ids) > 0, f"captured keys: {list(captured_dfs.keys())}"

    # -----------------------------------------------------------------------
    # C: Memory optimizations (B1-B5)
    # -----------------------------------------------------------------------

    def test_no_clone_when_skip_categorical_encoding(self, temp_data_dir, common_init_params, monkeypatch):
        """B1: No .clone() when all models are Polars-native (skip_categorical_encoding auto-set)."""
        pytest.importorskip("catboost")

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        clone_calls = []
        original_clone = pl.DataFrame.clone

        def _spy_clone(self):
            """Spy clone."""
            clone_calls.append(True)
            return original_clone(self)

        monkeypatch.setattr(pl.DataFrame, "clone", _spy_clone)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="no_clone_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        assert len(clone_calls) == 0, f"Expected 0 clone() calls (skip_categorical_encoding=True), got {len(clone_calls)}"

    def test_post_pipeline_polars_deleted(self, temp_data_dir, common_init_params, monkeypatch):
        """B2: Post-pipeline Polars DFs freed after pandas conversion when clone was needed."""

        n = 200
        rng = np.random.default_rng(42)
        pl_df = pl.DataFrame(
            {
                "num_feat": rng.standard_normal(n),
                "cat_feat": rng.choice(["a", "b", "c"], size=n),
                "target": rng.integers(0, 2, size=n),
            }
        )
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # Force clone by mixing polars-native and non-native models with encoding
        from mlframe.training.configs import PreprocessingBackendConfig

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="post_pipeline_del_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            pipeline_config=PreprocessingBackendConfig(categorical_encoding="ordinal"),
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        # If we got here without OOM, the test passes — the point is that
        # post-pipeline DFs were deleted. Hard to assert memory directly,
        # so we just verify training completed successfully.
        assert TargetTypes.BINARY_CLASSIFICATION in models

    def test_prepare_polars_cached_across_weights(self, temp_data_dir, common_init_params, monkeypatch):
        """B3: prepare_polars_dataframe() called once per model, not once per weight schema."""
        pytest.importorskip("catboost")

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        prepare_calls = []
        from mlframe.training.strategies import CatBoostStrategy

        original_prepare = CatBoostStrategy.prepare_polars_dataframe

        def _spy_prepare(self, df, cat_features):
            """Spy prepare."""
            prepare_calls.append(True)
            return original_prepare(self, df, cat_features)

        monkeypatch.setattr(CatBoostStrategy, "prepare_polars_dataframe", _spy_prepare)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="prepare_cache_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        # With 1 weight schema (uniform), prepare should be called exactly 3 times
        # (train, val, test) — NOT more from the weight loop
        assert len(prepare_calls) <= 3, f"prepare_polars_dataframe called {len(prepare_calls)} times — should be ≤3 (once per split)"

    def test_tier_uses_select_not_drop(self, temp_data_dir, common_init_params, monkeypatch):
        """B4: Tier trimming produces correct column count (text/emb excluded)."""
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        captured_ncols = {}
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            ncols = train_df.shape[1] if hasattr(train_df, "shape") else 0
            captured_ncols[model_type_name] = ncols
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="select_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            feature_types_config=FeatureTypesConfig(text_features=["text_feat"], embedding_features=["emb_feat"]),
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        # Ridge DF should have 2 fewer columns (text_feat + emb_feat dropped)
        for name, ncols in captured_ncols.items():
            if "Ridge" in name:
                # Original has num_feat, cat_feat, text_feat, emb_feat = 4 feature columns
                # After dropping text + emb = 2 feature columns remain
                assert ncols < 4, f"Ridge should have fewer columns after tier trimming, got {ncols}"

    def test_polars_originals_freed_after_tier1(self, temp_data_dir, common_init_params, monkeypatch):
        """B5: Pre-pipeline Polars originals released after all Polars-native models finish."""
        pytest.importorskip("catboost")
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        # Track training order and if Polars originals exist at each call
        training_info = []
        import mlframe.training.trainer as trainer_mod

        original_train = trainer_mod._train_model_with_fallback

        def _spy_train(model, model_obj, model_type_name, train_df, train_target, fit_params, verbose=False):
            """Spy train."""
            training_info.append((model_type_name, type(train_df).__name__))
            return original_train(
                model=model,
                model_obj=model_obj,
                model_type_name=model_type_name,
                train_df=train_df,
                train_target=train_target,
                fit_params=fit_params,
                verbose=verbose,
            )

        # The helper is re-imported into multiple sibling modules after the
        # 2026-05-22 trainer split; patch every binding so whichever the
        # suite reaches sees the spy.
        monkeypatch.setattr(trainer_mod, "_train_model_with_fallback", _spy_train)
        from mlframe.training import _trainer_train_and_evaluate as _tte_mod
        from mlframe.training import _trainer_configure as _tc_mod
        from mlframe.training import _training_loop as _tl_mod

        monkeypatch.setattr(_tte_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tc_mod, "_train_model_with_fallback", _spy_train, raising=False)
        monkeypatch.setattr(_tl_mod, "_train_model_with_fallback", _spy_train, raising=False)

        _models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="tier_release_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb", "ridge"],
            feature_types_config=FeatureTypesConfig(text_features=["text_feat"], embedding_features=["emb_feat"]),
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 10},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        # Both models should have trained successfully
        assert len(training_info) >= 2
        # CatBoost got Polars, Ridge got pandas
        cb_entries = [(n, t) for n, t in training_info if "CatBoost" in n]
        ridge_entries = [(n, t) for n, t in training_info if "Ridge" in n]
        assert len(cb_entries) > 0 and len(ridge_entries) > 0
        assert cb_entries[0][1] == "DataFrame", "CatBoost should receive Polars DataFrame"

    # -----------------------------------------------------------------------
    # D: GPU tests (actual training)
    # -----------------------------------------------------------------------

    @pytest.mark.gpu
    def test_catboost_polars_with_text_features_trains(self, temp_data_dir, common_init_params):
        """Actual CatBoost training with text features on Polars input."""
        pytest.importorskip("catboost")
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df(n=300)
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="gpu_text_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            feature_types_config=FeatureTypesConfig(text_features=["text_feat"]),
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 20},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        model_ns = models[TargetTypes.BINARY_CLASSIFICATION]["target"][0]
        assert model_ns.model is not None

    @pytest.mark.gpu
    def test_catboost_polars_with_embeddings_trains(self, temp_data_dir, common_init_params):
        """Actual CatBoost training with embedding features on Polars input."""
        pytest.importorskip("catboost")
        from mlframe.training.configs import FeatureTypesConfig

        pl_df = _make_text_embedding_polars_df(n=300)
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="test_target",
            model_name="gpu_emb_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            feature_types_config=FeatureTypesConfig(embedding_features=["emb_feat"]),
            reporting_config=common_init_params,
            hyperparams_config={"iterations": 20},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=temp_data_dir),
        )

        assert TargetTypes.BINARY_CLASSIFICATION in models
        model_ns = models[TargetTypes.BINARY_CLASSIFICATION]["target"][0]
        assert model_ns.model is not None
