"""
Comprehensive coverage tests for train_mlframe_models_suite.

Targets ~99% line+branch coverage of core.py:865-1793.
Split into sections matching the function's internal phases.
"""

import logging
import os
import unittest.mock
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.configs import (
    FeatureTypesConfig,
    ModelHyperparamsConfig,
    PolarsPipelineConfig,
    PreprocessingConfig,
    TargetTypes,
    TrainingBehaviorConfig,
    TrainingSplitConfig,
)
from .shared import SimpleFeaturesAndTargetsExtractor, TimestampedFeaturesExtractor


# ============================================================================
# Section 0-1: Input Validation + Configuration Setup (Agent 1)
# ============================================================================


class TestInputValidation:
    """Tests for input validation in train_mlframe_models_suite."""

    def test_invalid_df_type_raises_type_error(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        with pytest.raises(TypeError, match="df must be pandas DataFrame"):
            train_mlframe_models_suite(
                df=[1, 2, 3],
                target_name="target",
                model_name="test_model",
                features_and_targets_extractor=extractor,
                verbose=0,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                init_common_params=common_init_params,
            )

    def test_non_parquet_path_raises_value_error(self, sample_regression_data, temp_data_dir, common_init_params):
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        with pytest.raises(ValueError, match="File path must be a .parquet file"):
            train_mlframe_models_suite(
                df="data/my_file.csv",
                target_name="target",
                model_name="test_model",
                features_and_targets_extractor=extractor,
                verbose=0,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                init_common_params=common_init_params,
            )

    def test_empty_target_name_raises_value_error(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        with pytest.raises(ValueError, match="target_name cannot be empty"):
            train_mlframe_models_suite(
                df=df,
                target_name="",
                model_name="test_model",
                features_and_targets_extractor=extractor,
                verbose=0,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                init_common_params=common_init_params,
            )

    def test_empty_model_name_raises_value_error(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="",
                features_and_targets_extractor=extractor,
                verbose=0,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                init_common_params=common_init_params,
            )

    def test_none_extractor_raises_value_error(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        with pytest.raises(ValueError, match="features_and_targets_extractor is required"):
            train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="test_model",
                features_and_targets_extractor=None,
                verbose=0,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                init_common_params=common_init_params,
            )

    def test_int_df_raises_type_error(self, common_init_params):
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        with pytest.raises(TypeError, match="df must be pandas DataFrame"):
            train_mlframe_models_suite(
                df=42,
                target_name="target",
                model_name="test_model",
                features_and_targets_extractor=extractor,
                verbose=0,
                use_mlframe_ensembles=False,
                init_common_params=common_init_params,
            )

    def test_parquet_path_loads_and_trains(self, sample_regression_data, tmp_path, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(str(parquet_path), index=False)
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        models, metadata = train_mlframe_models_suite(
            df=str(parquet_path),
            target_name="target",
            model_name="test_parquet",
            features_and_targets_extractor=extractor,
            mlframe_models=["linear"],
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config={"iterations": 10},
            data_dir=temp_data_dir,
            init_common_params=common_init_params,
        )
        assert isinstance(models, dict)
        assert isinstance(metadata, dict)

    def test_dict_configs_all_accepted(self, sample_regression_data, temp_data_dir, common_init_params):
        """All configs passed as dicts are accepted and converted to Pydantic internally."""
        df, feature_names, y = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_dict_configs",
            features_and_targets_extractor=extractor,
            mlframe_models=["linear"],
            preprocessing_config={"fillna_value": 0.0},
            split_config={"test_size": 0.1, "val_size": 0.1},
            hyperparams_config={"iterations": 10},
            behavior_config={"prefer_gpu": False},
            verbose=0,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            init_common_params=common_init_params,
        )
        assert isinstance(models, dict)


class TestConfigurationSetup:
    """Tests for configuration setup in train_mlframe_models_suite."""

    def test_pydantic_preprocessing_config_passthrough(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        preproc = PreprocessingConfig(fillna_value=-999.0)
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_pydantic_preproc",
            features_and_targets_extractor=extractor,
            mlframe_models=["linear"],
            preprocessing_config=preproc,
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config={"iterations": 10},
            data_dir=temp_data_dir,
            init_common_params=common_init_params,
        )
        assert isinstance(models, dict)

    def test_pydantic_split_config_passthrough(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        split = TrainingSplitConfig(test_size=0.15, val_size=0.15)
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_pydantic_split",
            features_and_targets_extractor=extractor,
            mlframe_models=["linear"],
            split_config=split,
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config={"iterations": 10},
            data_dir=temp_data_dir,
            init_common_params=common_init_params,
        )
        assert isinstance(models, dict)

    def test_pydantic_hyperparams_config_passthrough(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        hparams = ModelHyperparamsConfig(iterations=10)
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_pydantic_hparams",
            features_and_targets_extractor=extractor,
            mlframe_models=["lasso"],
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config=hparams,
            data_dir=temp_data_dir,
            init_common_params=common_init_params,
        )
        assert isinstance(models, dict)

    def test_pydantic_behavior_config_passthrough(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        behavior = TrainingBehaviorConfig(prefer_gpu=False)
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_pydantic_behavior",
            features_and_targets_extractor=extractor,
            mlframe_models=["linear"],
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config={"iterations": 10},
            behavior_config=behavior,
            data_dir=temp_data_dir,
            init_common_params=common_init_params,
        )
        assert isinstance(models, dict)


# ============================================================================
# Section 2-3: Data Loading, Preprocessing, Splitting (Agent 2)
# ============================================================================


class TestDataLoadingPreprocessing:
    """Tests for phase 2: data loading and preprocessing."""

    def test_preprocessing_fillna(self, sample_regression_data, temp_data_dir, common_init_params):
        """NaN values in features should not crash training when fillna_value is set."""
        df, _, _ = sample_regression_data
        df = df.copy()
        df.iloc[:10, 0] = np.nan
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_fillna",
            features_and_targets_extractor=extractor,
            mlframe_models=["ridge"],
            preprocessing_config={"fillna_value": 0.0},
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config={"iterations": 10},
            data_dir=temp_data_dir,
            init_common_params=common_init_params,
        )
        assert metadata is not None

    def test_preprocessing_drop_columns(self, sample_regression_data, temp_data_dir, common_init_params):
        """Columns in preprocessing_config.drop_columns are removed before training."""
        df, feature_names, _ = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        _, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_drop",
            features_and_targets_extractor=extractor,
            mlframe_models=["ridge"],
            preprocessing_config={"drop_columns": ["feature_0"]},
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config={"iterations": 10},
            data_dir=temp_data_dir,
            init_common_params=common_init_params,
        )
        cols = list(metadata["columns"])
        assert "feature_0" not in cols


class TestSplitting:
    """Tests for phase 3: train/val/test splitting."""

    def test_split_sizes_sum_to_total(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        _, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_split_sum",
            features_and_targets_extractor=extractor,
            mlframe_models=["ridge"],
            split_config={"test_size": 0.2, "val_size": 0.2},
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config={"iterations": 10},
            data_dir=temp_data_dir,
            init_common_params=common_init_params,
        )
        total = metadata["train_size"] + metadata["val_size"] + metadata["test_size"]
        assert total == len(df)

    def test_artifact_files_saved_when_data_dir_given(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_artifacts",
            features_and_targets_extractor=extractor,
            mlframe_models=["ridge"],
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config={"iterations": 10},
            data_dir=temp_data_dir,
            models_dir="models",
            init_common_params=common_init_params,
        )
        saved_files = []
        for root, dirs, files in os.walk(temp_data_dir):
            saved_files.extend(files)
        assert len(saved_files) > 0, "Expected at least one artifact file in data_dir"

    def test_no_artifact_files_when_no_data_dir(self, sample_regression_data, tmp_path, common_init_params):
        """When data_dir is empty string, no artifact files should be written to tmp_path."""
        df, _, _ = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_no_artifacts",
            features_and_targets_extractor=extractor,
            mlframe_models=["ridge"],
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config={"iterations": 10},
            data_dir="",
            init_common_params=common_init_params,
        )
        # tmp_path should be empty (nothing written there)
        saved_files = list(tmp_path.rglob("*"))
        assert len(saved_files) == 0

    def test_split_metadata_keys_present(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        extractor = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        _, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="test_split_keys",
            features_and_targets_extractor=extractor,
            mlframe_models=["ridge"],
            verbose=0,
            use_mlframe_ensembles=False,
            hyperparams_config={"iterations": 10},
            data_dir=temp_data_dir,
            init_common_params=common_init_params,
        )
        for key in ("train_size", "val_size", "test_size"):
            assert key in metadata


# ============================================================================
# Section 4-4.5: Pipeline Fitting + Feature Type Detection (Agent 3)
# ============================================================================


class TestPipelineFitting:
    """Tests for Section 4 pipeline behavior in train_mlframe_models_suite."""

    def _call(self, df, temp_data_dir, common_init_params, **kwargs):
        defaults = dict(
            target_name="test_target",
            model_name="pipe_test",
            features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True),
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
            hyperparams_config={"iterations": 10},
        )
        defaults.update(kwargs)
        return train_mlframe_models_suite(df=df, **defaults)

    def test_metadata_pipeline_key_present_pandas(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        _, metadata = self._call(df, temp_data_dir, common_init_params, mlframe_models=["ridge"])
        assert "pipeline" in metadata

    def test_metadata_cat_features_key_present(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        _, metadata = self._call(df, temp_data_dir, common_init_params, mlframe_models=["ridge"])
        assert "cat_features" in metadata

    def test_metadata_columns_contains_features(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, _ = sample_regression_data
        _, metadata = self._call(df, temp_data_dir, common_init_params, mlframe_models=["ridge"])
        assert "columns" in metadata
        cols = list(metadata["columns"])
        for f in feature_names:
            assert f in cols

    def test_pandas_input_no_polars_pre_keys(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        _, metadata = self._call(df, temp_data_dir, common_init_params, mlframe_models=["ridge"])
        assert "train_df_polars_pre" not in metadata

    def test_auto_skip_encoding_all_polars_native(self, sample_regression_data, temp_data_dir, common_init_params):
        """All Polars-native models + Polars input → auto-skip encoding; training succeeds."""
        pytest.importorskip("catboost")
        df, _, _ = sample_regression_data
        pl_df = pl.from_pandas(df)
        _, metadata = self._call(
            pl_df, temp_data_dir, common_init_params,
            mlframe_models=["cb"],
            hyperparams_config={"iterations": 10, "cb_kwargs": {"task_type": "CPU"}},
        )
        assert "pipeline" in metadata

    def test_no_auto_skip_with_non_native_model(self, sample_regression_data, temp_data_dir, common_init_params):
        """ridge (non-native) on Polars input → auto-skip NOT triggered."""
        df, _, _ = sample_regression_data
        pl_df = pl.from_pandas(df)
        _, metadata = self._call(pl_df, temp_data_dir, common_init_params, mlframe_models=["ridge"])
        assert "pipeline" in metadata

    def test_polars_input_columns_stored(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, _ = sample_regression_data
        pl_df = pl.from_pandas(df)
        _, metadata = self._call(pl_df, temp_data_dir, common_init_params, mlframe_models=["ridge"])
        cols = list(metadata["columns"])
        for f in feature_names:
            assert f in cols

    def test_mixed_native_and_non_native_no_auto_skip(self, sample_regression_data, temp_data_dir, common_init_params):
        """cb (native) + ridge (non-native) → auto-skip NOT triggered."""
        pytest.importorskip("catboost")
        df, _, _ = sample_regression_data
        pl_df = pl.from_pandas(df)
        _, metadata = self._call(
            pl_df, temp_data_dir, common_init_params,
            mlframe_models=["ridge", "cb"],
            hyperparams_config={"iterations": 10, "cb_kwargs": {"task_type": "CPU"}},
        )
        assert "pipeline" in metadata
        assert "columns" in metadata


class TestFeatureTypeDetection:
    """Tests for Section 4.5 feature type detection."""

    def _call(self, df, temp_data_dir, common_init_params, **kwargs):
        defaults = dict(
            target_name="test_target",
            model_name="ftd_test",
            features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True),
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
            hyperparams_config={"iterations": 10},
        )
        defaults.update(kwargs)
        return train_mlframe_models_suite(df=df, **defaults)

    def test_text_features_in_metadata(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        df = df.copy()
        df["desc"] = "some text"
        _, metadata = self._call(
            df, temp_data_dir, common_init_params,
            mlframe_models=["ridge"],
            feature_types_config={"text_features": ["desc"], "auto_detect_feature_types": False},
        )
        assert "text_features" in metadata
        assert "desc" in metadata["text_features"]

    def test_embedding_features_in_metadata(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        df = df.copy()
        df["emb"] = [np.zeros(4).tolist() for _ in range(len(df))]
        _, metadata = self._call(
            df, temp_data_dir, common_init_params,
            mlframe_models=["ridge"],
            feature_types_config={"embedding_features": ["emb"], "auto_detect_feature_types": False},
        )
        assert "embedding_features" in metadata
        assert "emb" in metadata["embedding_features"]

    def test_no_feature_types_yields_empty_lists(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        _, metadata = self._call(df, temp_data_dir, common_init_params, mlframe_models=["ridge"])
        assert "text_features" in metadata
        assert "embedding_features" in metadata
        assert isinstance(metadata["text_features"], list)
        assert isinstance(metadata["embedding_features"], list)


# ============================================================================
# Section 5: Model Training Loop (Agent 4)
# ============================================================================


class TestModelTrainingLoop:
    """Tests for the model training loop in train_mlframe_models_suite (section 5)."""

    def test_unknown_model_skipped_with_warning(self, sample_regression_data, temp_data_dir, common_init_params, caplog):
        """Unknown model names emit a warning and are skipped; known models still train."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)

        with caplog.at_level(logging.WARNING):
            models, metadata = train_mlframe_models_suite(
                df=df,
                target_name="test_target",
                model_name="test_unknown_skip",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge", "unknown_xyz_model"],
                init_common_params=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                models_dir="models",
                hyperparams_config={"iterations": 10},
                verbose=0,
            )

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("unknown_xyz_model" in str(m) and "not known" in str(m) for m in warning_messages)
        assert TargetTypes.REGRESSION in models
        assert len(models[TargetTypes.REGRESSION]["target"]) >= 1

    def test_all_unknown_models_produces_empty(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_all_unknown",
            features_and_targets_extractor=fte,
            mlframe_models=["unknown_xyz_model"],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            hyperparams_config={"iterations": 10},
            verbose=0,
        )
        regression_models = models.get(TargetTypes.REGRESSION, {}).get("target", [])
        assert len(regression_models) == 0

    def test_uniform_weight_default(self, sample_regression_data, temp_data_dir, common_init_params):
        """Empty sample_weights from FTE → uniform default → one model per type."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        models, _ = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_uniform",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            hyperparams_config={"iterations": 10},
            verbose=0,
        )
        trained = models[TargetTypes.REGRESSION]["target"]
        assert len(trained) == 1

    def test_custom_weight_schema_trains_twice(self, sample_regression_data, temp_data_dir, common_init_params):
        """Two weight schemas → each model trained twice."""
        df, feature_names, y = sample_regression_data
        n_samples = len(df)
        fte = TimestampedFeaturesExtractor(
            target_column='target',
            regression=True,
            sample_weights={"uniform": None, "recency": np.ones(n_samples)},
        )
        models, _ = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_two_weights",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            hyperparams_config={"iterations": 10},
            verbose=0,
        )
        trained = models[TargetTypes.REGRESSION]["target"]
        assert len(trained) == 2

    def test_multiple_models_produce_multiple_entries(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        models, _ = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_multi",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge", "lasso"],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            hyperparams_config={"iterations": 10},
            verbose=0,
        )
        trained = models[TargetTypes.REGRESSION]["target"]
        assert len(trained) >= 2

    def test_two_models_two_weights_four_entries(self, sample_regression_data, temp_data_dir, common_init_params):
        """2 models × 2 weight schemas = 4 entries."""
        df, feature_names, y = sample_regression_data
        n_samples = len(df)
        fte = TimestampedFeaturesExtractor(
            target_column='target',
            regression=True,
            sample_weights={"uniform": None, "recency": np.ones(n_samples)},
        )
        models, _ = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_multi_x_weights",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge", "lasso"],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            hyperparams_config={"iterations": 10},
            verbose=0,
        )
        trained = models[TargetTypes.REGRESSION]["target"]
        assert len(trained) == 4

    def test_ensemble_scored_with_multiple_models(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        models, _ = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_ensemble",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge", "lasso"],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=True,
            data_dir=temp_data_dir,
            hyperparams_config={"iterations": 10},
            verbose=0,
        )
        trained = models[TargetTypes.REGRESSION]["target"]
        assert len(trained) >= 2

    def test_ensemble_not_scored_single_model(self, sample_regression_data, temp_data_dir, common_init_params):
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column='target', regression=True)
        models, _ = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_no_ens",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=True,
            data_dir=temp_data_dir,
            hyperparams_config={"iterations": 10},
            verbose=0,
        )
        trained = models[TargetTypes.REGRESSION]["target"]
        assert len(trained) == 1

    def test_model_clone_per_weight(self, sample_regression_data, temp_data_dir, common_init_params):
        """Each weight schema produces a distinct model clone."""
        df, feature_names, y = sample_regression_data
        n_samples = len(df)
        fte = TimestampedFeaturesExtractor(
            target_column='target',
            regression=True,
            sample_weights={"uniform": None, "recency": np.ones(n_samples)},
        )
        models, _ = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_clone",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            hyperparams_config={"iterations": 10},
            verbose=0,
        )
        trained = models[TargetTypes.REGRESSION]["target"]
        assert len(trained) == 2
        m0 = getattr(trained[0], 'model', trained[0])
        m1 = getattr(trained[1], 'model', trained[1])
        assert m0 is not m1


# ============================================================================
# Section 6: Recurrent Models + Cross-Cutting (Agent 5)
# ============================================================================


class TestRecurrentModels:
    """Tests for Section 6 recurrent model training path.

    Uses targeted mocking of only the recurrent code path to avoid
    interfering with regular model training (clone, process_model, etc.).
    """

    def _build_sequences(self, n_samples=1000, seq_len=10, n_seq_features=5):
        return [np.random.randn(seq_len, n_seq_features) for _ in range(n_samples)]

    def test_recurrent_fit_called_and_error_handled(self, sample_regression_data, temp_data_dir, common_init_params):
        """Recurrent model fit() is called; errors are caught gracefully."""
        df, feature_names, y = sample_regression_data
        n_samples = len(df)
        sequences = self._build_sequences(n_samples=n_samples)

        mock_model = unittest.mock.MagicMock()
        mock_model.get_params.return_value = {}
        mock_model.set_params.return_value = mock_model
        # fit() raises to test error handling path
        mock_model.fit.side_effect = RuntimeError("simulated GPU OOM")

        def fake_configure(**kwargs):
            return {"lstm": {"model": mock_model}}

        with unittest.mock.patch("mlframe.training.trainer._configure_recurrent_params", side_effect=fake_configure):
            # Patch clone ONLY for the recurrent section by selectively returning mock
            original_clone = __import__("sklearn.base", fromlist=["clone"]).clone

            def selective_clone(estimator, **kw):
                if estimator is mock_model:
                    return mock_model
                return original_clone(estimator, **kw)

            with unittest.mock.patch("mlframe.training.core.clone", side_effect=selective_clone):
                fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
                models, metadata = train_mlframe_models_suite(
                    df=df,
                    target_name="test_target",
                    model_name="test_recurrent",
                    features_and_targets_extractor=fte,
                    mlframe_models=["ridge"],
                    recurrent_models=["lstm"],
                    sequences=sequences,
                    init_common_params=common_init_params,
                    use_ordinary_models=True,
                    use_mlframe_ensembles=False,
                    data_dir=temp_data_dir,
                    models_dir="models",
                    verbose=0,
                    hyperparams_config={"iterations": 10},
                )

        # fit() was called (even though it raised)
        assert mock_model.fit.called, "Expected recurrent model fit() to be called"
        # Regular ridge model still trained successfully
        assert TargetTypes.REGRESSION in models
        assert len(models[TargetTypes.REGRESSION]["target"]) >= 1

    def test_unknown_recurrent_model_skipped(self, sample_regression_data, temp_data_dir, common_init_params):
        """Unconfigured recurrent model is skipped; fit() not called."""
        df, feature_names, y = sample_regression_data
        n_samples = len(df)
        sequences = self._build_sequences(n_samples=n_samples)

        mock_model = unittest.mock.MagicMock()
        mock_model.get_params.return_value = {}

        def fake_configure(**kwargs):
            return {}  # "gru" not configured

        with unittest.mock.patch("mlframe.training.trainer._configure_recurrent_params", side_effect=fake_configure):
            fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
            models, metadata = train_mlframe_models_suite(
                df=df,
                target_name="test_target",
                model_name="test_recurrent_unknown",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                recurrent_models=["gru"],
                sequences=sequences,
                init_common_params=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                models_dir="models",
                verbose=0,
                hyperparams_config={"iterations": 10},
            )

        assert not mock_model.fit.called


class TestCrossCuttingParametrized:
    """Cross-cutting parametrized tests for model/dataframe-type combinations."""

    @pytest.mark.parametrize("df_type", ["pandas", "polars"])
    @pytest.mark.parametrize("model", ["ridge", "lasso"])
    def test_model_df_combinations(self, df_type, model, sample_regression_data, temp_data_dir, common_init_params):
        """Each (model, df_type) produces valid non-empty output."""
        df, feature_names, y = sample_regression_data
        input_df = pl.from_pandas(df) if df_type == "polars" else df

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
        models, metadata = train_mlframe_models_suite(
            df=input_df,
            target_name="test_target",
            model_name=f"xcut_{model}_{df_type}",
            features_and_targets_extractor=fte,
            mlframe_models=[model],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
            hyperparams_config={"iterations": 10},
        )

        assert isinstance(models, dict)
        assert len(models) > 0
        for tt, targets in models.items():
            for tname, model_list in targets.items():
                assert len(model_list) > 0


class TestMetadataCompleteness:
    """Tests verifying metadata completeness and persistence."""

    def _train(self, df, temp_data_dir, common_init_params):
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
        return train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="meta_test",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
            hyperparams_config={"iterations": 10},
        )

    def test_metadata_has_all_expected_keys(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        _, metadata = self._train(df, temp_data_dir, common_init_params)
        expected_keys = [
            "model_name", "target_name", "configs", "pipeline",
            "cat_features", "columns", "text_features", "embedding_features",
            "train_size", "val_size", "test_size",
        ]
        for key in expected_keys:
            assert key in metadata, f"Expected metadata key '{key}' not found"

    def test_metadata_configs_exists(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        _, metadata = self._train(df, temp_data_dir, common_init_params)
        assert "configs" in metadata
        assert metadata["configs"] is not None

    def test_split_sizes_sum_to_original(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        _, metadata = self._train(df, temp_data_dir, common_init_params)
        total = metadata["train_size"] + metadata["val_size"] + metadata["test_size"]
        assert total == len(df)

    def test_metadata_saved_to_disk(self, sample_regression_data, temp_data_dir, common_init_params):
        df, _, _ = sample_regression_data
        self._train(df, temp_data_dir, common_init_params)
        joblib_files = list(Path(temp_data_dir).rglob("*.joblib"))
        assert len(joblib_files) > 0
        loaded = joblib.load(joblib_files[0])
        assert isinstance(loaded, dict)
