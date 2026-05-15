"""Smoke + early-return contract tests for the 7 publicly-exported core/ functions that have no direct unit-test coverage.

These tests exercise the cheapest reliable entry-points: disabled-feature paths, empty-input handling, input-validation guards. They are NOT a substitute for the end-to-end ``train_mlframe_models_suite`` integration tests but pin down the early-return contracts that
silent refactors would otherwise break (the contracts most likely to regress under further function-level decomposition).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.core._phase_composite_discovery import run_composite_target_discovery
from mlframe.training.core._phase_composite_post import run_composite_post_processing
from mlframe.training.core._phase_config_setup import setup_configuration
from mlframe.training.core._phase_dummy_baselines import run_dummy_baselines
from mlframe.training.core._phase_polars_fixes import apply_polars_categorical_fixes
from mlframe.training.core._phase_temporal_audit import run_temporal_audit_batch
from mlframe.training.core._training_context import TrainingContext
from mlframe.training.core.predict import predict_from_models


# ---------------------------------------------------------------------------
# apply_polars_categorical_fixes
# ---------------------------------------------------------------------------

def test_apply_polars_categorical_fixes_no_polars_inputs_returns_pandas_branch_unchanged():
    """When train_df_polars is None the function bypasses the entire Polars-categorical fix block and returns the pandas frames unchanged."""
    train_df_pd = pd.DataFrame({"a": [1, 2, 3]})
    val_df_pd = pd.DataFrame({"a": [4, 5]})
    test_df_pd = pd.DataFrame({"a": [6]})

    result = apply_polars_categorical_fixes(
        train_df_polars=None,
        val_df_polars=None,
        test_df_polars=None,
        train_df_pd=train_df_pd,
        val_df_pd=val_df_pd,
        test_df_pd=test_df_pd,
        filtered_train_df=train_df_pd,
        filtered_val_df=val_df_pd,
        cat_features=None,
        align_polars_categorical_dicts=False,
        defer_pandas_conv=False,
        was_polars_input=False,
        verbose=False,
    )
    assert isinstance(result, tuple)
    assert len(result) >= 3  # train/val/test slots at minimum


def test_apply_polars_categorical_fixes_no_cat_features_is_noop():
    """With no cat_features declared the null-fill and dict-alignment passes should both short-circuit."""
    train_pl = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = apply_polars_categorical_fixes(
        train_df_polars=train_pl,
        val_df_polars=None,
        test_df_polars=None,
        train_df_pd=None,
        val_df_pd=None,
        test_df_pd=None,
        filtered_train_df=train_pl,
        filtered_val_df=None,
        cat_features=[],
        align_polars_categorical_dicts=True,
        defer_pandas_conv=False,
        was_polars_input=True,
        verbose=False,
    )
    assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# run_temporal_audit_batch
# ---------------------------------------------------------------------------

def test_run_temporal_audit_batch_no_timestamp_returns_empty():
    """With explicit opt-out (column='') AND no FTE ts_field, the audit silently produces an empty dict."""
    behavior_config = SimpleNamespace(target_temporal_audit_column="")
    fte = SimpleNamespace(ts_field=None)

    result = run_temporal_audit_batch(
        behavior_config=behavior_config,
        features_and_targets_extractor=fte,
        df=None,
        timestamps=None,
        target_by_type={},
        verbose=False,
    )
    assert result == {}


def test_run_temporal_audit_batch_no_behavior_config_returns_empty():
    """``behavior_config=None`` triggers the FTE-fallback path; with no FTE ts_field the audit is silent."""
    fte = SimpleNamespace(ts_field=None)

    result = run_temporal_audit_batch(
        behavior_config=None,
        features_and_targets_extractor=fte,
        df=None,
        timestamps=None,
        target_by_type={},
        verbose=False,
    )
    assert result == {}


# ---------------------------------------------------------------------------
# run_dummy_baselines
# ---------------------------------------------------------------------------

def _dummy_baselines_minimal_kwargs(cfg, metadata):
    """Build the 25-kwarg call dict for ``run_dummy_baselines`` with all-None inputs; only ``cfg`` and ``metadata`` carry signal."""
    return dict(
        target_type="REGRESSION",
        cur_target_name="y",
        target_name="y",
        model_name="m",
        current_train_target=None,
        current_val_target=None,
        current_test_target=None,
        filtered_train_df=None,
        filtered_val_df=None,
        test_df_pd=None,
        filtered_train_idx=None,
        filtered_val_idx=None,
        test_idx=None,
        timestamps=None,
        cat_features=[],
        dummy_baselines_config=cfg,
        quantile_regression_config=None,
        reporting_config=None,
        _dropped_high_card_data={},
        train_od_idx=None,
        val_od_idx=None,
        plot_file=None,
        metadata=metadata,
        target_by_type={},
        _split_preds_probs=lambda raw: (raw, None),  # never called on early-return path
    )


def test_run_dummy_baselines_disabled_returns_metadata_unchanged():
    """``dummy_baselines_config.enabled=False`` triggers the wrapped-try early return at the top of the function."""
    cfg = SimpleNamespace(enabled=False, apply_to_target_types=[])
    metadata = {"sentinel": 42}
    result = run_dummy_baselines(**_dummy_baselines_minimal_kwargs(cfg, metadata))
    assert result is metadata
    assert result == {"sentinel": 42}


def test_run_dummy_baselines_wrong_target_type_returns_metadata_unchanged():
    """``enabled=True`` but the current ``target_type`` is not in ``apply_to_target_types`` still triggers the early return."""
    cfg = SimpleNamespace(enabled=True, apply_to_target_types=["BINARY_CLASSIFICATION"])
    metadata = {"sentinel": 7}
    result = run_dummy_baselines(**_dummy_baselines_minimal_kwargs(cfg, metadata))
    assert result is metadata


# ---------------------------------------------------------------------------
# run_composite_target_discovery
# ---------------------------------------------------------------------------

def test_run_composite_target_discovery_disabled_returns_inputs_unchanged():
    """``composite_target_discovery_config.enabled=False`` returns the (target_by_type, metadata) tuple without touching it."""
    cfg = SimpleNamespace(enabled=False)
    target_by_type = {"REGRESSION": {"y": np.array([1.0, 2.0])}}
    metadata = {"sentinel": "x"}

    new_targets, new_metadata = run_composite_target_discovery(
        composite_target_discovery_config=cfg,
        target_by_type=target_by_type,
        mlframe_models=[],
        metadata=metadata,
        filtered_train_df=None,
        filtered_train_idx=None,
        train_df_pd=None,
        val_df_pd=None,
        test_df_pd=None,
        train_idx=None,
        val_idx=None,
        test_idx=None,
        baseline_diagnostics_config=None,
        cat_features=[],
        verbose=False,
    )
    assert new_targets is target_by_type
    assert new_metadata is metadata


def test_run_composite_target_discovery_no_regression_target_returns_inputs_unchanged():
    """Enabled but ``TargetTypes.REGRESSION not in target_by_type`` -> early return per the gate at the top of the function body."""
    cfg = SimpleNamespace(enabled=True)
    target_by_type = {"BINARY_CLASSIFICATION": {"y": np.array([0, 1])}}
    metadata = {}

    new_targets, new_metadata = run_composite_target_discovery(
        composite_target_discovery_config=cfg,
        target_by_type=target_by_type,
        mlframe_models=[],
        metadata=metadata,
        filtered_train_df=None,
        filtered_train_idx=None,
        train_df_pd=None,
        val_df_pd=None,
        test_df_pd=None,
        train_idx=None,
        val_idx=None,
        test_idx=None,
        baseline_diagnostics_config=None,
        cat_features=[],
        verbose=False,
    )
    assert new_targets is target_by_type
    assert new_metadata is metadata


# ---------------------------------------------------------------------------
# run_composite_post_processing
# ---------------------------------------------------------------------------

def test_run_composite_post_processing_no_specs_returns_inputs_unchanged():
    """With empty ``metadata["composite_target_specs"]`` and disabled discovery all 3 internal blocks (wrap / cross-ensemble / suite-end summary)
    skip and the function returns ``(models, metadata)`` as its signature promises. Regression guard: prior version fell off the end and
    implicitly returned ``None`` (caller in main.py compensated with a ``is not None`` check)."""
    models: dict = {}
    metadata = {"composite_target_specs": {}}
    discovery_cfg = SimpleNamespace(enabled=False, cross_target_ensemble_strategy="off")
    dummy_cfg = SimpleNamespace(enabled=False, apply_to_target_types=[], best_model_min_lift=None)

    new_models, new_metadata = run_composite_post_processing(
        models=models,
        metadata=metadata,
        target_by_type={},
        composite_target_discovery_config=discovery_cfg,
        target_name="y",
        model_name="m",
        filtered_train_df=None,
        filtered_val_df=None,
        test_df_pd=None,
        filtered_train_idx=None,
        filtered_val_idx=None,
        test_idx=None,
        train_df_pd=None,
        val_df_pd=None,
        train_idx=None,
        val_idx=None,
        dummy_baselines_config=dummy_cfg,
        reporting_config=None,
        plot_file=None,
        verbose=False,
    )
    assert new_models is models
    assert new_metadata is metadata
    assert metadata == {"composite_target_specs": {}}


# ---------------------------------------------------------------------------
# setup_configuration
# ---------------------------------------------------------------------------

def test_setup_configuration_with_all_none_configs_builds_training_context():
    """``setup_configuration`` materializes default Pydantic configs from None and returns a populated ``TrainingContext``."""
    ctx = setup_configuration(
        preprocessing_config=None,
        pipeline_config=None,
        feature_types_config=None,
        split_config=None,
        hyperparams_config=None,
        behavior_config=None,
        reporting_config=None,
        output_config=None,
        outlier_detection_config=None,
        feature_selection_config=None,
        confidence_analysis_config=None,
        baseline_diagnostics_config=None,
        dummy_baselines_config=None,
        quantile_regression_config=None,
        composite_target_discovery_config=None,
        feature_handling_config=None,
        model_name="test_model",
        target_name="y",
        mlframe_models=["cb"],
        verbose=False,
    )
    assert isinstance(ctx, TrainingContext)
    assert ctx.model_name == "test_model"
    assert ctx.target_name == "y"
    # All config slots should be non-None Pydantic objects (or whatever _ensure_config materialises)
    assert ctx.preprocessing_config is not None
    assert ctx.behavior_config is not None
    assert ctx.split_config is not None


# ---------------------------------------------------------------------------
# predict_from_models
# ---------------------------------------------------------------------------

def test_predict_from_models_invalid_df_raises_type_error():
    """Input-validation guard: ``df`` that is neither pandas nor polars raises TypeError before any model work."""
    with pytest.raises(TypeError, match="pandas or polars DataFrame"):
        predict_from_models(
            df="not a dataframe",
            models={},
            metadata={},
            features_and_targets_extractor=None,
            return_probabilities=False,
            verbose=0,
        )


def test_predict_from_models_empty_models_dict_returns_empty_predictions():
    """An empty ``models`` mapping is a valid input: the function should return its result-shaped dict without raising, having processed no models."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = predict_from_models(
        df=df,
        models={},
        metadata={},
        features_and_targets_extractor=None,
        return_probabilities=False,
        verbose=0,
    )
    assert isinstance(result, dict)
    assert "predictions" in result
    assert result["predictions"] == {} or len(result["predictions"]) == 0
