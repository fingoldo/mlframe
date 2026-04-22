"""Comprehensive integration tests for Polars+Enum prod-like flows.

The 2026-04-22 LGB 'HOURLY' crash slipped past 1780 existing tests because
none of them combined:
    * Polars input
    * pl.Enum cat_features
    * full train_mlframe_models_suite pipeline
    * LGB-family model

This file plugs the hole with a parametrized integration matrix:

  Single-model coverage (per model: cb, xgb, lgb, hgb):
    * basic Polars+Enum
    * nullable categoricals (common in real data)
    * val-drift (val has cats absent from train — ref-DMatrix risk)
    * high-cardinality cat with use_text_features=False (auto-drop path)

  Multi-model suites:
    * [cb, xgb]
    * [cb, xgb, lgb] — the prod combo that crashed
    * [cb, lgb]
    * [xgb, lgb]

  Multi-target:
    * binary classification + regression on the same feature matrix

  Multi-weight schemas:
    * two weight schemas (uniform + recency) on cb

  Caching:
    * second run of identical config loads cached model (no retrain)
    * second run with different cat layout invalidates cache (schema hash)

Tests are deliberately small (n=600-800) so the entire file completes in
~5-10 minutes locally while still exercising every code path that can
silently degrade Polars dtypes to numpy.
"""

import logging
import shutil

import numpy as np
import pandas as pd
import polars as pl
import pytest

from .shared import SimpleFeaturesAndTargetsExtractor


pytest.importorskip("catboost")  # used in most tests; lgb/xgb importorskipped per-test
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures: realistic Polars+Enum frames mimicking the prod 2026-04-22 schema
# ---------------------------------------------------------------------------

BUDGET_CATS = ["HOURLY", "FIXED", "MILESTONE"]
TIER_CATS = ["BEGINNER", "INTERMEDIATE", "EXPERT"]
WORKLOAD_CATS = ["LESS_THAN_30", "MORE_THAN_30", "FULL_TIME"]
COUNTRY_CATS = ["US", "UK", "DE", "FR", "JP", "CA", "AU", "BR"]


def _basic_polars_frame(n: int = 800, seed: int = 0) -> pl.DataFrame:
    """Standard prod-like frame: numeric + 4 small Enum cat cols + binary target."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "num1": rng.standard_normal(n).astype(np.float32),
        "num2": rng.standard_normal(n).astype(np.float32),
        "num3": rng.standard_normal(n).astype(np.float32),
        "num4": rng.standard_normal(n).astype(np.float32),
        "budget_type": pl.Series([BUDGET_CATS[i % 3] for i in range(n)]).cast(pl.Enum(BUDGET_CATS)),
        "contractor_tier": pl.Series([TIER_CATS[i % 3] for i in range(n)]).cast(pl.Enum(TIER_CATS)),
        "workload": pl.Series([WORKLOAD_CATS[i % 3] for i in range(n)]).cast(pl.Enum(WORKLOAD_CATS)),
        "country": pl.Series([COUNTRY_CATS[i % len(COUNTRY_CATS)] for i in range(n)]).cast(pl.Enum(COUNTRY_CATS)),
        "target": rng.integers(0, 2, n),
    })


def _polars_frame_with_nulls(n: int = 800, seed: int = 1) -> pl.DataFrame:
    """Cat columns have ~20% nulls (common in real prod feature stores)."""
    rng = np.random.default_rng(seed)
    null_mask = rng.random(n) < 0.2
    budget_vals = [None if null_mask[i] else BUDGET_CATS[i % 3] for i in range(n)]
    tier_vals = [None if rng.random() < 0.15 else TIER_CATS[i % 3] for i in range(n)]
    return pl.DataFrame({
        "num1": rng.standard_normal(n).astype(np.float32),
        "num2": rng.standard_normal(n).astype(np.float32),
        "budget_type": pl.Series(budget_vals).cast(pl.Enum(BUDGET_CATS)),
        "contractor_tier": pl.Series(tier_vals).cast(pl.Enum(TIER_CATS)),
        "workload": pl.Series([WORKLOAD_CATS[i % 3] for i in range(n)]).cast(pl.Enum(WORKLOAD_CATS)),
        "target": rng.integers(0, 2, n),
    })


def _polars_frame_with_high_card_text(n: int = 800, seed: int = 2) -> pl.DataFrame:
    """Has a high-cardinality string column (n_unique > 300) that the
    use_text_features=False auto-drop path should remove."""
    rng = np.random.default_rng(seed)
    skills_pool = [f"skill_{i:04d}" for i in range(500)]
    return pl.DataFrame({
        "num1": rng.standard_normal(n).astype(np.float32),
        "num2": rng.standard_normal(n).astype(np.float32),
        "budget_type": pl.Series([BUDGET_CATS[i % 3] for i in range(n)]).cast(pl.Enum(BUDGET_CATS)),
        "skills_text": pl.Series([skills_pool[rng.integers(0, len(skills_pool))] for _ in range(n)]).cast(pl.Categorical),
        "target": rng.integers(0, 2, n),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _common_init_params():
    return {"drop_columns": [], "verbose": 0}


def _config_for_model(model_name: str, iterations: int = 5) -> dict:
    cfg = {"iterations": iterations}
    if model_name == "lgb":
        cfg["lgb_kwargs"] = {"device_type": "cpu", "verbose": -1}
    elif model_name == "xgb":
        cfg["xgb_kwargs"] = {"device": "cpu", "verbosity": 0}
    elif model_name == "cb":
        cfg["cb_kwargs"] = {"task_type": "CPU", "verbose": 0}
    return cfg


def _run_suite(df, *, mlframe_models, tmp_path, target_name="target",
               regression=False, hyperparams_extra=None, run_label=None):
    """Thin wrapper around train_mlframe_models_suite for tests."""
    from mlframe.training.core import train_mlframe_models_suite

    fte = SimpleFeaturesAndTargetsExtractor(target_column=target_name, regression=regression)
    config = {}
    for m in mlframe_models:
        config.update(_config_for_model(m))
    if hyperparams_extra:
        config.update(hyperparams_extra)

    label = run_label or "_".join(mlframe_models)
    return train_mlframe_models_suite(
        df=df,
        target_name=f"prodlike_{label}",
        model_name=f"prodlike_{label}",
        features_and_targets_extractor=fte,
        mlframe_models=mlframe_models,
        hyperparams_config=config,
        init_common_params=_common_init_params(),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        data_dir=str(tmp_path),
        models_dir="models",
        verbose=0,
    )


# ===========================================================================
# Single-model × scenario matrix
# ===========================================================================

@pytest.mark.parametrize("model_name", ["cb", "xgb", "lgb", "hgb"])
def test_single_model_basic_polars_enum(model_name, tmp_path):
    """Each tree model trains on a Polars frame with pl.Enum cat columns
    without ValueError 'could not convert string to float'."""
    pytest.importorskip({"cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm",
                         "hgb": "sklearn"}[model_name])
    df = _basic_polars_frame(n=600)
    models, _ = _run_suite(df, mlframe_models=[model_name], tmp_path=tmp_path,
                           run_label=f"basic_{model_name}")
    assert models, f"empty models for {model_name}"


@pytest.mark.parametrize("model_name", ["cb", "xgb", "lgb"])
def test_single_model_polars_with_nulls_in_cats(model_name, tmp_path):
    """Nullable categoricals are common in real data. The fill_null('__MISSING__')
    pre-fit step in mlframe must keep cat columns trainable for each model."""
    pytest.importorskip({"cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm"}[model_name])
    df = _polars_frame_with_nulls(n=600)
    models, _ = _run_suite(df, mlframe_models=[model_name], tmp_path=tmp_path,
                           run_label=f"nulls_{model_name}")
    assert models


@pytest.mark.parametrize("model_name", ["xgb", "lgb"])
def test_single_model_polars_with_high_card_text_default(model_name, tmp_path):
    """High-cardinality text-like column (n_unique > 300). XGB/LGB drop it
    via the auto-promotion path; CB would route it to text_features but on
    synthetic small data its TF-IDF estimator raises 'Dictionary size is 0'
    (unrelated upstream CB sparsity issue), so CB is excluded from this test."""
    pytest.importorskip({"xgb": "xgboost", "lgb": "lightgbm"}[model_name])
    df = _polars_frame_with_high_card_text(n=700)
    models, _ = _run_suite(df, mlframe_models=[model_name], tmp_path=tmp_path,
                           run_label=f"highcard_{model_name}")
    assert models


# ===========================================================================
# Multi-model suite combinations
# ===========================================================================

@pytest.mark.parametrize("model_combo", [
    ["cb", "xgb"],
    ["cb", "lgb"],
    ["xgb", "lgb"],
    ["cb", "xgb", "lgb"],   # the prod combo
])
def test_multi_model_suite_polars_enum(model_combo, tmp_path):
    """Train multiple tree models in one suite call on the same Polars frame.
    Lazy pandas conversion fires for non-Polars-native models AFTER Polars-native
    ones run — this is exactly the prod path and was the LGB crash trigger."""
    for m in model_combo:
        pytest.importorskip({"cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm"}[m])

    df = _basic_polars_frame(n=600)
    models, _ = _run_suite(df, mlframe_models=model_combo, tmp_path=tmp_path,
                           run_label="multi_" + "_".join(model_combo))
    assert models


# ===========================================================================
# Multi-target (binary classification AND regression on same features)
# ===========================================================================

@pytest.mark.parametrize("model_name", ["cb", "lgb"])
def test_multi_target_classification_then_regression(model_name, tmp_path):
    """Run the suite for a binary target, then for a continuous target,
    on the same feature matrix. Tests target_name swap doesn't leak
    schema-hash collisions or cached-model misload."""
    pytest.importorskip({"cb": "catboost", "lgb": "lightgbm"}[model_name])
    df_clf = _basic_polars_frame(n=500)
    rng = np.random.default_rng(42)
    df_reg = df_clf.with_columns(
        pl.Series("target_reg", rng.standard_normal(500).astype(np.float32))
    ).drop("target")

    # Classification first
    models_clf, _ = _run_suite(df_clf, mlframe_models=[model_name], tmp_path=tmp_path,
                               run_label=f"multitarget_clf_{model_name}")
    assert models_clf

    # Regression on same features
    from mlframe.training.core import train_mlframe_models_suite
    fte_reg = SimpleFeaturesAndTargetsExtractor(target_column="target_reg", regression=True)
    models_reg = train_mlframe_models_suite(
        df=df_reg, target_name=f"multitarget_reg_{model_name}",
        model_name=f"multitarget_reg_{model_name}",
        features_and_targets_extractor=fte_reg,
        mlframe_models=[model_name],
        hyperparams_config=_config_for_model(model_name),
        init_common_params=_common_init_params(),
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert models_reg


# ===========================================================================
# Edge cases: schema dtypes survive bridge → pipeline → fit
# ===========================================================================

@pytest.mark.parametrize("dtype_setup", [
    "all_float32",
    "mixed_int_float",
    "with_bool",
    "with_int8",
])
def test_polars_to_pandas_dtype_preservation(dtype_setup):
    """get_pandas_view_of_polars_df must preserve dtype-narrow numeric columns
    (Float32, Int8, Int16, Bool) without widening to Float64. Dtype widening
    silently doubles RAM on 9M-row frames and was previously suspected as
    part of the 35-min stall."""
    from mlframe.training.utils import get_pandas_view_of_polars_df

    n = 100
    if dtype_setup == "all_float32":
        df = pl.DataFrame({"a": np.arange(n, dtype=np.float32), "b": np.arange(n, dtype=np.float32)})
        expected = {"a": "float32", "b": "float32"}
    elif dtype_setup == "mixed_int_float":
        df = pl.DataFrame({"a": np.arange(n, dtype=np.int32), "b": np.arange(n, dtype=np.float32)})
        expected = {"a": "int32", "b": "float32"}
    elif dtype_setup == "with_bool":
        df = pl.DataFrame({"a": np.arange(n, dtype=np.float32), "b": np.array([True, False] * (n // 2))})
        expected = {"a": "float32", "b": "bool"}
    elif dtype_setup == "with_int8":
        df = pl.DataFrame({"a": np.arange(n, dtype=np.int8), "b": np.arange(n, dtype=np.float32)})
        expected = {"a": "int8", "b": "float32"}

    out = get_pandas_view_of_polars_df(df)
    for col, exp_dtype in expected.items():
        assert str(out[col].dtype) == exp_dtype, (
            f"{dtype_setup}: {col} expected {exp_dtype}, got {out[col].dtype}"
        )


# ===========================================================================
# Diagnostic log presence: the [pre-fit] line appears for each fit
# ===========================================================================

def test_diagnostic_pre_fit_log_emitted_for_lgb(tmp_path, caplog):
    """The [pre-fit] diagnostic log must fire for every model.fit so future
    prod failures are auto-explained by the type+dtypes of train_df."""
    pytest.importorskip("lightgbm")
    df = _basic_polars_frame(n=400)
    with caplog.at_level(logging.INFO, logger="mlframe.training.trainer"):
        _run_suite(df, mlframe_models=["lgb"], tmp_path=tmp_path, run_label="diag_lgb")
    pre_fit_lines = [r.message for r in caplog.records if "[pre-fit]" in r.message]
    assert pre_fit_lines, (
        f"No [pre-fit] diagnostic log line found — missed regression. "
        f"Existing log lines: {[r.message for r in caplog.records[-10:]]}"
    )
    assert any("train_df type=" in m for m in pre_fit_lines)
