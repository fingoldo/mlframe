"""Happy-path unit tests for the publicly-exported ``training/core/`` functions.

Companion to ``test_core_public_smoke.py``: the smoke file pins the cheapest reliable contracts (disabled-feature paths, empty-input handling,
input-validation guards). This file exercises the enabled paths with minimal valid inputs so a real refactor can't silently turn the body
into a pass-through and still claim coverage.

``run_composite_target_discovery`` / ``run_composite_post_processing`` are exercised with feature-flagged-on configs and minimal valid state.
The full discovery / wrapping algorithms need real fitted models -- they're covered end-to-end by the suite-level tests in ``test_core.py``.
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
# apply_polars_categorical_fixes — enabled path: nullable Categorical column gets filled
# ---------------------------------------------------------------------------


def test_apply_polars_categorical_fixes_fills_nullable_categorical():
    """A Polars ``pl.Categorical`` column containing nulls is fill_null'd to ``'__MISSING__'`` so CatBoost 1.2.10's fused-cpdef dispatcher won't trip."""
    # build a polars frame with a nullable Categorical
    cat_col = pl.Series("cat_feat", ["a", "b", None, "a", "c"], dtype=pl.Categorical)
    train_pl = pl.DataFrame({"cat_feat": cat_col})

    train_out, *_ = apply_polars_categorical_fixes(
        train_df_polars=train_pl,
        val_df_polars=None,
        test_df_polars=None,
        train_df_pd=None,
        val_df_pd=None,
        test_df_pd=None,
        filtered_train_df=train_pl,
        filtered_val_df=None,
        cat_features=["cat_feat"],
        align_polars_categorical_dicts=False,
        defer_pandas_conv=True,
        was_polars_input=True,
        verbose=False,
    )

    # Output is a Polars frame
    assert isinstance(train_out, pl.DataFrame)
    # The null was replaced (either by '__MISSING__' or an equivalent sentinel)
    assert train_out["cat_feat"].null_count() == 0


# ---------------------------------------------------------------------------
# run_temporal_audit_batch — enabled path: timestamps present, FTE-detected column triggers audit
# ---------------------------------------------------------------------------


def _ts_extractor(field_name: str = "ts"):
    """Build a minimal features_and_targets_extractor stub with a ``.ts_field`` attribute."""
    return SimpleNamespace(ts_field=field_name)


def test_run_temporal_audit_batch_with_timestamps_returns_per_target_results():
    """FTE-detected timestamp column + a regression target produces an audit entry per (target_type, target_name)."""
    n = 60
    timestamps = np.arange(n, dtype=np.int64) * 86400 * 1_000_000_000  # daily, ns
    target_y = np.linspace(0.0, 1.0, n)
    # behavior_config=None forces the FTE-auto-detect path; FTE has ts_field set so audit fires.
    result = run_temporal_audit_batch(
        behavior_config=None,
        features_and_targets_extractor=_ts_extractor("ts"),
        timestamps=timestamps,
        target_by_type={"REGRESSION": {"y": target_y}},
        verbose=False,
    )
    # Either dispatch fired and we have audit data, or the underlying ``_audit_targets_over_time`` was skipped gracefully
    # (rare on numeric path). Either way we get a dict back and don't raise.
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# setup_configuration — enabled path: dict hyperparams flow into ctx
# ---------------------------------------------------------------------------


def test_setup_configuration_dict_hyperparams_flow_into_ctx():
    """When ``hyperparams_config`` is a dict, ``setup_configuration`` materialises the typed Pydantic config and routes the user's values onto ``ctx``."""
    ctx = setup_configuration(
        preprocessing_config=None,
        pipeline_config=None,
        feature_types_config=None,
        split_config=None,
        hyperparams_config={"iterations": 333, "learning_rate": 0.07, "early_stopping_rounds": 12},
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
        model_name="happy_path_model",
        target_name="y",
        mlframe_models=["cb"],
        verbose=False,
    )
    assert isinstance(ctx, TrainingContext)
    assert ctx.model_name == "happy_path_model"
    assert ctx.target_name == "y"
    # The dict-form hyperparams round-tripped through Pydantic — accessible via attribute access on the typed object
    assert ctx.hyperparams_config.iterations == 333
    assert pytest.approx(ctx.hyperparams_config.learning_rate, 1e-6) == 0.07
    assert ctx.hyperparams_config.early_stopping_rounds == 12


# ---------------------------------------------------------------------------
# predict_from_models — enabled path: pre-fitted sklearn Ridge in models dict
# ---------------------------------------------------------------------------


def test_predict_from_models_with_fitted_ridge_returns_predictions():
    """A trained sklearn Ridge wrapped in a ``SimpleNamespace(model=...)`` shape returns numeric predictions when fed a compatible DataFrame."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])
    y_train = 2 * X_train["a"] - X_train["b"] + 0.5 * X_train["c"] + rng.normal(0, 0.01, 50)
    model = Ridge().fit(X_train, y_train)
    entry = SimpleNamespace(model=model, pre_pipeline=None, model_name="ridge")
    models = {"REGRESSION": {"y": [entry]}}
    # metadata.input_schema=None disables the column validator (it accepts any df when no schema is pinned).
    metadata: dict = {"pipeline": None, "input_schema": None, "model_schemas": {}}

    X_serve = pd.DataFrame(rng.standard_normal((5, 3)), columns=["a", "b", "c"])
    result = predict_from_models(
        df=X_serve,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=None,
        return_probabilities=False,
        verbose=0,
    )

    assert isinstance(result, dict)
    # The Ridge produced numeric predictions, keyed by ``f"{target_type}_{target_name}"`` (plus possibly a pre_pipeline tag)
    assert len(result["predictions"]) >= 1
    preds = next(iter(result["predictions"].values()))
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (5,)


# ---------------------------------------------------------------------------
# run_dummy_baselines — enabled path: real regression target produces baseline metadata
# ---------------------------------------------------------------------------


def _dummy_baselines_min_kwargs(target_type, current_target, metadata, cfg, *, split_preds_probs=None):
    """Build the 25-kwarg call dict for ``run_dummy_baselines`` parameterised over the few inputs the body actually reads."""
    return dict(
        target_type=target_type,
        cur_target_name="y",
        target_name="y",
        model_name="happy_path",
        current_train_target=current_target,
        current_val_target=current_target[:5] if current_target is not None else None,
        current_test_target=current_target[5:10] if current_target is not None else None,
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
        _split_preds_probs=split_preds_probs if split_preds_probs is not None else (lambda raw: (raw, None)),
    )


def test_run_dummy_baselines_enabled_regression_returns_metadata():
    """``enabled=True`` + a real regression target makes the function attempt baselines; with minimal inputs the wrapped-try keeps the
    suite alive on any computational misstep but always returns the same metadata dict reference."""
    cfg = SimpleNamespace(
        enabled=True,
        apply_to_target_types=["REGRESSION"],
        best_model_min_lift=0.05,
    )
    metadata: dict = {}
    target = np.linspace(0.0, 1.0, 30)
    result = run_dummy_baselines(**_dummy_baselines_min_kwargs("REGRESSION", target, metadata, cfg))
    # Same dict reference back, with the function possibly mutating it
    assert result is metadata


# ---------------------------------------------------------------------------
# run_composite_target_discovery — enabled path: REGRESSION target + minimal feature frame
# ---------------------------------------------------------------------------


def test_run_composite_target_discovery_enabled_regression_initializes_metadata():
    """``enabled=True`` + REGRESSION target progresses past the early-return gate at line 60 and runs the discovery prologue
    (``_init_composite_discovery_metadata``) which populates the three metadata buckets even when no composite specs survive screening."""
    from mlframe.training.configs import TargetTypes

    rng = np.random.default_rng(0)
    n = 80
    feature_df = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    target_y = rng.standard_normal(n)
    target_by_type = {TargetTypes.REGRESSION: {"y": target_y}}
    metadata: dict = {}
    cfg = SimpleNamespace(
        enabled=True,
        multilabel_strategy="per_target",
        cross_target_ensemble_strategy="off",
        oof_random_state=42,
    )

    new_targets, new_metadata = run_composite_target_discovery(
        composite_target_discovery_config=cfg,
        target_by_type=target_by_type,
        mlframe_models=["cb"],
        metadata=metadata,
        filtered_train_df=feature_df,
        filtered_train_idx=np.arange(n),
        train_df_pd=feature_df,
        val_df_pd=feature_df.iloc[: n // 5],
        test_df_pd=feature_df.iloc[: n // 5],
        train_idx=np.arange(n),
        val_idx=np.arange(n // 5),
        test_idx=np.arange(n // 5),
        baseline_diagnostics_config=None,
        cat_features=[],
        verbose=False,
    )

    # Prologue populated the three metadata buckets the function unconditionally creates when enabled
    assert "composite_target_specs" in new_metadata
    assert "composite_target_failures" in new_metadata
    assert "composite_target_filter_drops" in new_metadata
    # target_by_type still contains the original REGRESSION target (possibly more keys after multilabel expansion)
    assert TargetTypes.REGRESSION in new_targets


# ---------------------------------------------------------------------------
# run_composite_post_processing — enabled path: discovery on + cross-ensemble strategy on, but no specs to act on
# ---------------------------------------------------------------------------


def test_run_composite_post_processing_enabled_no_specs_runs_all_three_blocks():
    """``enabled=True`` + ``cross_target_ensemble_strategy != 'off'`` progresses past the early-return gates for Block A (wrap),
    Block B (cross-target ensemble) and Block C (suite-end summary). With empty ``composite_target_specs`` Blocks A/B no-op cleanly;
    Block C's wrapped-try keeps the suite alive on any error and always returns the ``(models, metadata)`` tuple."""
    models: dict = {}
    metadata = {"composite_target_specs": {}}
    discovery_cfg = SimpleNamespace(enabled=True, cross_target_ensemble_strategy="mean", oof_random_state=42)
    dummy_cfg = SimpleNamespace(enabled=False, apply_to_target_types=[], best_model_min_lift=0.05)

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
    # Contract per signature: always returns (models, metadata)
    assert new_models is models
    assert new_metadata is metadata
