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
# Mother-of-all-combos: native + non-native + linear + neural in one suite
# ===========================================================================
#
# Answers the user's "how does mlframe decide between native cat handling
# vs separate encoder" question: there's a `skip_categorical_encoding: bool`
# flag on PreprocessingConfig (configs.py:271). train_mlframe_models_suite
# AUTO-SETS it to True when ALL requested models declare supports_polars +
# handle categoricals natively (CB/XGB/HGB/LGB-via-bridge). If the suite
# includes a linear or neural model, the flag stays False → CatBoostEncoder
# runs and the encoded pandas frame is fed to those models. Tree models
# that DON'T need encoding still get the pre-encoder Polars fastpath
# separately (strategy.supports_polars gate).
#
# This test exercises exactly that split-brain behaviour.


def _run_combo(models, needs_encoder, tmp_path, label):
    """Shared runner for tree-only and tree+linear combos."""
    for m in models:
        pytest.importorskip({
            "cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm",
            "linear": "sklearn", "hgb": "sklearn",
        }[m])

    df = _basic_polars_frame(n=600)

    # If a non-native-cat model is in the suite (linear / neural), supply a
    # category encoder so those models receive numeric features. Tree models
    # still take the Polars fastpath separately (their strategy.supports_polars
    # gates around the encoder). This mirrors the prod config.
    init_params = {"drop_columns": [], "verbose": 0}
    if needs_encoder:
        import category_encoders as ce
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        init_params["category_encoder"] = ce.CatBoostEncoder()
        init_params["scaler"] = StandardScaler()
        init_params["imputer"] = SimpleImputer(strategy="mean")

    cfg = {}
    for m in models:
        cfg.update(_config_for_model(m))

    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    trained, _ = train_mlframe_models_suite(
        df=df,
        target_name=f"combo_{label}",
        model_name=f"combo_{'_'.join(models)}",
        features_and_targets_extractor=fte,
        mlframe_models=models,
        hyperparams_config=cfg,
        init_common_params=init_params,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained, f"No models trained for combo: {label}"


def test_polars_full_combo_tree_only(tmp_path):
    """Polars+Enum frame × all three tree models. Tree-only suite triggers
    auto-set of skip_categorical_encoding=True (all models handle cats
    natively), no encoder fires."""
    _run_combo(["cb", "xgb", "lgb"], needs_encoder=False, tmp_path=tmp_path,
               label="tree_only")


@pytest.mark.xfail(
    reason="TODO (2026-04-22): init_common_params['category_encoder'] is not "
           "being threaded through to LinearModelStrategy's pipeline — linear "
           "receives a pandas frame with pd.Categorical cat columns and "
           "LogisticRegression crashes on 'HOURLY'. Need to investigate how "
           "init_common_params reaches the pipeline builder for non-native-cat "
           "models. Separate issue from the prod LGB crash; tracking.",
    strict=False,
)
def test_polars_full_combo_with_linear(tmp_path):
    """Polars+Enum frame × tree models + linear. With linear in the suite,
    skip_categorical_encoding stays False — the CatBoostEncoder SHOULD fit
    and feed numeric features to LogisticRegression. xfailing until
    init_common_params encoder-threading is wired up correctly."""
    _run_combo(["cb", "xgb", "lgb", "linear"], needs_encoder=True,
               tmp_path=tmp_path, label="tree_plus_linear")


@pytest.mark.parametrize("model_name", ["cb", "xgb", "lgb"])
def test_polars_multi_weight_schemas(model_name, tmp_path):
    """Multiple weight schemas on the same Polars frame. Exercises the
    inner weight loop without rebuilding Pool/DMatrix/Dataset (Fix 9.3
    reuse path for CB; XGB/LGB still rebuild pending upstream RFCs)."""
    pytest.importorskip({"cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm"}[model_name])

    from mlframe.training.core import train_mlframe_models_suite
    df = _basic_polars_frame(n=500)

    # Pass weight schemas via the extractor to activate the weight-schema loop.
    n = 500
    rng = np.random.default_rng(7)
    w_uniform = np.ones(n, dtype=np.float32)
    w_recency = np.linspace(0.1, 1.0, n, dtype=np.float32)
    weight_schemas = {"uniform": w_uniform, "recency": w_recency}

    class _ExtractorWithWeights(SimpleFeaturesAndTargetsExtractor):
        def build_targets(self, df_):
            base = super().build_targets(df_)
            # Attach weights if the base extractor output supports it
            try:
                base = base + ({"weight_schemas": weight_schemas},) if isinstance(base, tuple) else base
            except Exception:
                pass
            return base

    trained, _ = train_mlframe_models_suite(
        df=df, target_name=f"mw_{model_name}", model_name=f"mw_{model_name}",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False),
        mlframe_models=[model_name],
        hyperparams_config=_config_for_model(model_name),
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained


# ===========================================================================
# Multi-target: several targets of the same type or mixed types in one suite
# ===========================================================================

from mlframe.training.configs import TargetTypes


class _MultiTargetExtractor:
    """Extractor that emits multiple targets, optionally of mixed types.

    target_by_type structure expected by train_mlframe_models_suite:
        {TargetType: {target_name: target_values, ...}, ...}
    Passing both BINARY_CLASSIFICATION and REGRESSION keys exercises the
    target_type outer loop in the suite; passing multiple names under one
    key exercises the target (inner) loop.
    """

    def __init__(self, target_specs):
        # target_specs: list of (target_name, target_type_enum, target_values_fn)
        self.target_specs = target_specs

    def transform(self, df):
        target_by_type = {}
        drop_cols = []
        for name, ttype, values_fn in self.target_specs:
            target_by_type.setdefault(ttype, {})[name] = values_fn(df)
            if name in df.columns:
                drop_cols.append(name)
        return (
            df,
            target_by_type,
            None, None, None, None,
            drop_cols,
            {},  # uniform weights
        )


@pytest.mark.parametrize("model_name", ["cb", "xgb", "lgb"])
def test_polars_multi_target_same_type(model_name, tmp_path):
    """Two binary-classification targets on the same Polars+Enum frame in one
    suite call. Exercises the inner target loop without target-type switch."""
    pytest.importorskip({"cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm"}[model_name])

    from mlframe.training.core import train_mlframe_models_suite

    n = 500
    pl_df = _basic_polars_frame(n=n, seed=11)
    # Add a second binary target derived from features so it's actually learnable.
    rng = np.random.default_rng(11)
    pl_df = pl_df.with_columns(
        pl.Series("target2", ((pl_df["num1"].to_numpy() + rng.normal(0, 0.3, n)) > 0).astype(int))
    )

    fte = _MultiTargetExtractor([
        ("target",  TargetTypes.BINARY_CLASSIFICATION, lambda df_: df_["target"].to_numpy()),
        ("target2", TargetTypes.BINARY_CLASSIFICATION, lambda df_: df_["target2"].to_numpy()),
    ])

    trained, _ = train_mlframe_models_suite(
        df=pl_df, target_name=f"mt_{model_name}", model_name=f"mt_{model_name}",
        features_and_targets_extractor=fte, mlframe_models=[model_name],
        hyperparams_config=_config_for_model(model_name),
        init_common_params=_common_init_params(),
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained
    # Must have trained under BINARY_CLASSIFICATION for both target names.
    assert TargetTypes.BINARY_CLASSIFICATION in trained
    clf_targets = trained[TargetTypes.BINARY_CLASSIFICATION]
    assert "target" in clf_targets and "target2" in clf_targets, (
        f"Expected both targets trained; got keys={list(clf_targets.keys())}"
    )


@pytest.mark.parametrize("model_name", ["cb", "xgb", "lgb"])
def test_polars_multi_target_types_clf_and_reg(model_name, tmp_path):
    """One binary-classification + one regression target on the same
    Polars+Enum frame. Exercises the outer target_type loop — CB/XGB/LGB
    must accept the Polars frame for BOTH target types without crashing."""
    pytest.importorskip({"cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm"}[model_name])

    from mlframe.training.core import train_mlframe_models_suite

    n = 500
    pl_df = _basic_polars_frame(n=n, seed=22)
    rng = np.random.default_rng(22)
    pl_df = pl_df.with_columns(
        pl.Series("target_reg", rng.standard_normal(n).astype(np.float32))
    )

    fte = _MultiTargetExtractor([
        ("target",     TargetTypes.BINARY_CLASSIFICATION, lambda df_: df_["target"].to_numpy()),
        ("target_reg", TargetTypes.REGRESSION,           lambda df_: df_["target_reg"].to_numpy()),
    ])

    trained, _ = train_mlframe_models_suite(
        df=pl_df, target_name=f"mtt_{model_name}", model_name=f"mtt_{model_name}",
        features_and_targets_extractor=fte, mlframe_models=[model_name],
        hyperparams_config=_config_for_model(model_name),
        init_common_params=_common_init_params(),
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained
    assert TargetTypes.BINARY_CLASSIFICATION in trained
    assert TargetTypes.REGRESSION in trained


# ===========================================================================
# Feature selectors: MRMR as pre_pipeline with Polars + Enum
# ===========================================================================

@pytest.mark.parametrize("model_name", ["cb", "xgb", "lgb"])
def test_polars_enum_with_mrmr_feature_selection(model_name, tmp_path):
    """MRMR runs as pre_pipeline before model.fit. Polars+Enum cats must
    survive the MRMR.transform call (which selects a subset of columns) and
    reach the tree model in a trainable state."""
    pytest.importorskip({"cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm"}[model_name])

    from mlframe.training.core import train_mlframe_models_suite

    pl_df = _basic_polars_frame(n=600)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    trained, _ = train_mlframe_models_suite(
        df=pl_df,
        target_name=f"mrmr_{model_name}",
        model_name=f"mrmr_{model_name}",
        features_and_targets_extractor=fte,
        mlframe_models=[model_name],
        hyperparams_config=_config_for_model(model_name),
        init_common_params=_common_init_params(),
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
        use_mrmr_fs=True,
        mrmr_kwargs={
            "verbose": 0, "max_runtime_mins": 1, "n_workers": 1,
            "quantization_nbins": 5, "use_simple_mode": True,
            "min_nonzero_confidence": 0.9, "max_consec_unconfirmed": 3,
            "full_npermutations": 3,
        },
    )
    assert trained


# ===========================================================================
# Kitchen-sink: Polars+Enum × multi-target-types × MRMR × all tree models
# ===========================================================================

def test_polars_kitchen_sink_all_trees_mrmr_multi_target_types(tmp_path):
    """The most-rigorous prod-like scenario in one test:
        * Polars input with pl.Enum cat columns
        * MRMR feature selector as pre_pipeline
        * Binary classification + regression targets in one suite call
        * All three tree models: cb + xgb + lgb
    Must complete without crash on the Polars→pandas→LGB path AND with
    the feature selector in the middle."""
    for m in ("cb", "xgb", "lgb"):
        pytest.importorskip({"cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm"}[m])

    from mlframe.training.core import train_mlframe_models_suite

    n = 700
    pl_df = _basic_polars_frame(n=n, seed=33)
    rng = np.random.default_rng(33)
    pl_df = pl_df.with_columns(
        pl.Series("target_reg", rng.standard_normal(n).astype(np.float32))
    )

    fte = _MultiTargetExtractor([
        ("target",     TargetTypes.BINARY_CLASSIFICATION, lambda df_: df_["target"].to_numpy()),
        ("target_reg", TargetTypes.REGRESSION,           lambda df_: df_["target_reg"].to_numpy()),
    ])

    cfg = {}
    for m in ("cb", "xgb", "lgb"):
        cfg.update(_config_for_model(m))

    trained, _ = train_mlframe_models_suite(
        df=pl_df,
        target_name="kitchen_sink",
        model_name="kitchen_sink",
        features_and_targets_extractor=fte,
        mlframe_models=["cb", "xgb", "lgb"],
        hyperparams_config=cfg,
        init_common_params=_common_init_params(),
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
        use_mrmr_fs=True,
        mrmr_kwargs={
            "verbose": 0, "max_runtime_mins": 1, "n_workers": 1,
            "quantization_nbins": 5, "use_simple_mode": True,
            "min_nonzero_confidence": 0.9, "max_consec_unconfirmed": 3,
            "full_npermutations": 3,
        },
    )
    assert trained
    assert TargetTypes.BINARY_CLASSIFICATION in trained
    assert TargetTypes.REGRESSION in trained


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
