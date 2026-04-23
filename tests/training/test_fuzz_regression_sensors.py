"""Permanent regression sensors for bugs first caught by the fuzz suite.

Each entry here is a **named, deterministic** test pinned to the exact
``FuzzCombo`` seed that first reproduced the bug. After a bug is fixed,
its sensor loses the ``@pytest.mark.xfail`` marker and becomes a
permanent guard. If the sensor ever goes red in the future, the
developer knows *which* combo / combination of features regressed.

Workflow:
  1. Fuzz suite catches a new failure class → add predicate to
     ``KNOWN_XFAIL_RULES`` in ``_fuzz_combo.py``.
  2. Pin the smallest-n failing combo as a sensor here. Add
     ``@pytest.mark.xfail(strict=True, reason="tracked by ...")``.
  3. When upstream fix lands, the xfail becomes XPASS → strict=True
     flips that to a hard fail. The PR that fixes the bug must delete
     both the xfail line AND the predicate in the rule table.
  4. The bare-named sensor then sits in the suite forever as a regression
     tripwire.

All sensors share ``_sensor_cleanup`` teardown with the fuzz suite to
avoid native-segfault-on-state-accumulation seen on Windows.
"""
from __future__ import annotations

import gc
import pytest

from ._fuzz_combo import FuzzCombo, build_frame_for_combo
from .shared import SimpleFeaturesAndTargetsExtractor


@pytest.fixture(autouse=True)
def _sensor_cleanup():
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    try:
        from mlframe.training import trainer as _tr
        for attr in ("_CB_POOL_CACHE", "_CB_VAL_POOL_CACHE"):
            cache = getattr(_tr, attr, None)
            if hasattr(cache, "clear"):
                cache.clear()
    except Exception:
        pass
    gc.collect()
    gc.collect()


def _skip_if_deps_missing(*models):
    pkg = {"cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm",
           "hgb": "sklearn", "linear": "sklearn"}
    for m in models:
        pytest.importorskip(pkg[m])


def _run_sensor_combo(combo: FuzzCombo, tmp_path):
    from mlframe.training.core import train_mlframe_models_suite

    df, target_col, _ = build_frame_for_combo(combo)
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
    )

    cfg: dict = {"iterations": 3 if combo.n_rows <= 300 else 5}
    if "lgb" in combo.models:
        cfg["lgb_kwargs"] = {"device_type": "cpu", "verbose": -1}
    if "xgb" in combo.models:
        cfg["xgb_kwargs"] = {"device": "cpu", "verbosity": 0}
    if "cb" in combo.models:
        cfg["cb_kwargs"] = {"task_type": "CPU", "verbose": 0}

    init_params: dict = {"drop_columns": [], "verbose": 0}
    if "linear" in combo.models and combo.cat_feature_count > 0:
        try:
            import category_encoders as ce
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            init_params["category_encoder"] = ce.CatBoostEncoder()
            init_params["scaler"] = StandardScaler()
            init_params["imputer"] = SimpleImputer(strategy="mean")
        except ImportError:
            pass

    mrmr_kwargs = None
    if combo.use_mrmr_fs:
        mrmr_kwargs = {
            "verbose": 0, "max_runtime_mins": 1, "n_workers": 1,
            "quantization_nbins": 5, "use_simple_mode": True,
            "min_nonzero_confidence": 0.9, "max_consec_unconfirmed": 3,
            "full_npermutations": 3,
        }

    trained, _meta = train_mlframe_models_suite(
        df=df,
        target_name=combo.short_id(),
        model_name=combo.short_id(),
        features_and_targets_extractor=fte,
        mlframe_models=list(combo.models),
        hyperparams_config=cfg,
        init_common_params=init_params,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        data_dir=str(tmp_path),
        models_dir="models",
        verbose=0,
        use_mrmr_fs=combo.use_mrmr_fs,
        mrmr_kwargs=mrmr_kwargs,
    )
    assert trained, "train_mlframe_models_suite returned empty models dict"


# ---------------------------------------------------------------------------
# Sensor #1 — tier_dfs_cache Polars/pandas collision (FIXED 2026-04-22).
# ---------------------------------------------------------------------------

def test_sensor_tier_cache_polars_pandas_collision(tmp_path):
    """Regression guard for the tier_dfs_cache bug that first surfaced as
    ``AttributeError: 'DataFrame' object has no attribute 'schema'``.

    Fuzz caught this combo as c0011 (linear + xgb on polars_nullable,
    n=300, cat_feature_count=3). Root cause: ``_build_tier_dfs`` in
    ``mlframe/training/core.py`` keyed its cache on just ``strategy.feature_tier()``
    — but Linear (non-polars-native) iteration stashed *pandas* tier-DFs
    under tier=(False,False); the subsequent XGB (polars-native) iteration
    requested the same tier and got pandas back, then
    ``XGBoostStrategy.prepare_polars_dataframe`` tried ``df.schema.items()``
    on a pandas DataFrame and blew up.

    Fixed 2026-04-22 by composing the cache key with the container kind
    (pl / pd) sampled from the first non-None DF in the passed dict.
    If this sensor ever goes red again, the most likely regression is
    someone simplifying the cache key back to just ``tier``.
    """
    _skip_if_deps_missing("linear", "xgb")
    combo = FuzzCombo(
        models=("linear", "xgb"),
        input_type="polars_nullable",
        n_rows=300,
        cat_feature_count=3,
        null_fraction_cats=0.0,
        use_mrmr_fs=False,
        weight_schemas=("uniform",),
        target_type="binary_classification",
        auto_detect_cats=True,
        align_polars_categorical_dicts=False,
        seed=11,
    )
    _run_sensor_combo(combo, tmp_path)


# ---------------------------------------------------------------------------
# Sensor — MRMR.transform raised AttributeError on un-set support_ (FIXED 2026-04-22).
# ---------------------------------------------------------------------------

def test_sensor_mrmr_transform_handles_missing_support_(tmp_path):
    """Regression guard for MRMR.transform crashing on un-set support_.

    Fuzz c0109: single Linear + MRMR on pandas with 1 cat feature + n=1200
    synthetic-data. MRMR.fit() exited early (no positive MI signal on the
    noise-level target) without ever assigning ``self.support_``. The
    sklearn Pipeline then called MRMR.transform() as part of a fit→transform
    flow, and filters.py:3435 ``if self.support_ is None`` raised
    AttributeError because the attribute didn't exist at all.

    Fixed 2026-04-22 by replacing direct attribute access with
    ``getattr(self, 'support_', None)`` in MRMR.transform — preserving the
    "no selection yet" pass-through semantics without requiring the
    attribute to exist. Regressing this = direct dict lookup on support_.
    """
    _skip_if_deps_missing("linear")
    combo = FuzzCombo(
        models=("linear",),
        input_type="pandas",
        n_rows=1200,
        cat_feature_count=1,
        null_fraction_cats=0.0,
        use_mrmr_fs=True,
        weight_schemas=("uniform",),
        target_type="binary_classification",
        auto_detect_cats=True,
        align_polars_categorical_dicts=False,
        seed=109,
    )
    _run_sensor_combo(combo, tmp_path)


# ---------------------------------------------------------------------------
# Sensor #2 — _SafeUnpickler blocked category_encoders (FIXED 2026-04-22).
# ---------------------------------------------------------------------------

def test_sensor_safeunpickler_allows_category_encoders(tmp_path):
    """Regression guard for the cached-model-load WARN that defeated the
    schema-hash cache for every suite including a Linear model.

    Fuzz suite + manual repro showed:
    ``Could not load model from file ... linear__sch_....dump:
      Unsafe class blocked by _SafeUnpickler allowlist:
      category_encoders.cat_boost.CatBoostEncoder``
    → ``load_mlframe_model`` returned None → ``train_eval.py:637`` then
    crashed on ``loaded_model.model`` (NoneType). Double bug: (a) the
    allowlist missed a widely-used safe package, (b) no None-guard on the
    loader result.

    Fixed 2026-04-22:
      * Added ``category_encoders`` prefix to ``_SAFE_MODULE_PREFIXES`` in
        ``mlframe/training/io.py``.
      * Added None-guard in ``mlframe/training/train_eval.py`` — falls
        back to retraining with an actionable WARN.

    Test strategy: run the suite twice. First run trains + saves. Second
    run must load the cached artifact cleanly (no WARN, no fallback).
    """
    _skip_if_deps_missing("linear", "xgb")
    import logging
    combo = FuzzCombo(
        models=("linear", "xgb"),
        input_type="polars_nullable",
        n_rows=300,
        cat_feature_count=3,
        null_fraction_cats=0.0,
        use_mrmr_fs=False,
        weight_schemas=("uniform",),
        target_type="binary_classification",
        auto_detect_cats=True,
        align_polars_categorical_dicts=False,
        seed=11,
    )

    # First run: trains + saves.
    _run_sensor_combo(combo, tmp_path)

    # Second run (identical config, same tmp_path) must load from cache.
    # Capture log output; if any "Unsafe class blocked" WARN fires, the
    # allowlist regression has returned.
    import io
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("mlframe")
    logger.addHandler(handler)
    try:
        _run_sensor_combo(combo, tmp_path)
    finally:
        logger.removeHandler(handler)
    log_text = buf.getvalue()
    assert "Unsafe class blocked" not in log_text, (
        "Regression: _SafeUnpickler blocked a category_encoders class. "
        f"Full WARN log: {log_text[:500]}"
    )
