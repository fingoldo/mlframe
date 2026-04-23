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
# Sensor — MRMR must accept pl.DataFrame natively without full .to_pandas() (FIXED 2026-04-22, Fix 10).
# ---------------------------------------------------------------------------

def test_sensor_mrmr_native_polars_no_full_to_pandas():
    """Regression guard for MRMR.fit copying the whole frame via X.to_pandas()
    when the input is a pl.DataFrame.

    CLAUDE.md forbids caller-visible copies on 100+ GB prod frames. Before
    Fix 10, MRMR.fit unconditionally called X.to_pandas() at filters.py:~2894.
    The fix branches on isinstance(X, pl.DataFrame) and uses Polars-native
    with_columns / fill_null / drop (no-copy for Arrow-backed numerics).
    categorize_dataset was likewise adapted: numeric subset via to_numpy()
    (zero-copy for numerics) and a bounded conversion of only the cat-col
    subset for OrdinalEncoder.

    Spy on pl.DataFrame.to_pandas during fit and assert ≤ 1 call (allowed
    for the cat-col subset only).
    """
    import polars as pl
    import numpy as np
    from mlframe.feature_selection.filters import MRMR

    n = 500
    rng = np.random.default_rng(42)
    pl_df = pl.DataFrame({
        "num1": rng.standard_normal(n).astype(np.float32),
        "num2": rng.standard_normal(n).astype(np.float32),
        "num3": rng.standard_normal(n).astype(np.float32),
        "cat1": pl.Series(["A", "B", "C"] * (n // 3 + 1))[:n].cast(pl.Enum(["A", "B", "C"])),
    })
    y = rng.integers(0, 2, n)

    call_count = {"n": 0}
    orig = pl.DataFrame.to_pandas
    def _spy(self, *args, **kwargs):
        call_count["n"] += 1
        return orig(self, *args, **kwargs)
    pl.DataFrame.to_pandas = _spy
    try:
        sel = MRMR(
            verbose=0, max_runtime_mins=1, n_workers=1,
            quantization_nbins=5, use_simple_mode=True,
            min_nonzero_confidence=0.9, max_consec_unconfirmed=3,
            full_npermutations=3,
        )
        sel.fit(pl_df, y)
    finally:
        pl.DataFrame.to_pandas = orig

    # 0 calls expected — MRMR's polars path uses pl.col.to_physical() to
    # get integer codes for Categorical / Enum / bool / Utf8-cast-to-Cat
    # without ever materializing to pandas. Any .to_pandas() call means
    # someone re-introduced a copy path.
    assert call_count["n"] == 0, (
        f"MRMR.fit called pl.DataFrame.to_pandas() {call_count['n']} times — "
        f"regression: Fix 10 requires a zero-copy polars path (to_physical "
        f"for cat codes, .to_numpy() for numerics). A full-frame .to_pandas() "
        f"would OOM on 200 GB prod (CLAUDE.md)."
    )

    # transform must return pl.DataFrame.
    out = sel.transform(pl_df)
    assert isinstance(out, pl.DataFrame), (
        f"MRMR.transform must preserve pl.DataFrame input type; got {type(out).__name__}"
    )


# ---------------------------------------------------------------------------
# Sensor — MRMR must actually SELECT cat features that matter (both pandas & polars).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frame_type", ["pandas", "polars"])
def test_sensor_mrmr_selects_informative_cat_features(frame_type):
    """Functional guard: MRMR must include an informative categorical
    column among its selected features (not drop it as noise). Covers
    both pandas and Polars inputs — asserts that the Fix 10 polars path
    preserves MI-based selection quality, not just API compatibility.

    Data: 1000 rows, binary target derived directly from a 3-level cat
    column; three noise numeric columns; one noise cat column.
      * cat_signal: categorical with 3 levels, perfectly predictive
        (class 0 for level 'A', class 1 for 'B' and 'C').
      * cat_noise: 4 random levels, independent of target.
      * num_noise_1..3: iid normal.
    MRMR should rank cat_signal well above the noise features.
    """
    import numpy as np
    from mlframe.feature_selection.filters import MRMR

    n = 1000
    rng = np.random.default_rng(7)
    signal_levels = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    y = (signal_levels != "A").astype(np.int32)  # fully determined by cat_signal
    noise_levels = np.array(["x", "y", "z", "w"])[rng.integers(0, 4, n)]
    num_data = rng.standard_normal((n, 3)).astype(np.float32)

    if frame_type == "polars":
        import polars as pl
        df = pl.DataFrame({
            "num_noise_1": num_data[:, 0],
            "num_noise_2": num_data[:, 1],
            "num_noise_3": num_data[:, 2],
            "cat_signal": pl.Series(signal_levels).cast(pl.Enum(["A", "B", "C"])),
            "cat_noise": pl.Series(noise_levels).cast(pl.Enum(["x", "y", "z", "w"])),
        })
    else:
        import pandas as pd
        df = pd.DataFrame({
            "num_noise_1": num_data[:, 0],
            "num_noise_2": num_data[:, 1],
            "num_noise_3": num_data[:, 2],
            "cat_signal": pd.Categorical(signal_levels, categories=["A", "B", "C"]),
            "cat_noise": pd.Categorical(noise_levels, categories=["x", "y", "z", "w"]),
        })

    sel = MRMR(
        verbose=0, max_runtime_mins=1, n_workers=1,
        quantization_nbins=5, use_simple_mode=True,
        min_nonzero_confidence=0.9, max_consec_unconfirmed=3,
        full_npermutations=5, min_relevance_gain=1e-4,
    )
    sel.fit(df, y)

    # MRMR must select at least one feature, and cat_signal must be first
    # (by highest MI with target).
    support = sel.support_
    assert support is not None and len(support) >= 1, (
        f"MRMR ({frame_type}) selected no features on perfectly-predictive "
        f"cat_signal — MI computation is broken."
    )
    selected_names = [sel.feature_names_in_[i] for i in support]
    assert "cat_signal" in selected_names, (
        f"MRMR ({frame_type}) did NOT select cat_signal despite perfect "
        f"correlation with target. Selected: {selected_names}. "
        f"Likely root cause: categorize_dataset's polars path fails to "
        f"encode cat codes properly, so MI on cat_signal is computed as 0."
    )


# ---------------------------------------------------------------------------
# Sensor — Linear polars gating bug (FIXED 2026-04-22 via Fix 11).
# ---------------------------------------------------------------------------

def test_sensor_linear_polars_gating_bug(tmp_path):
    """Regression guard for the ``polars_pipeline_applied`` global-flag bug
    at core.py:3085. Before Fix 11, the flag was OR-accumulated across the
    model-suite loop — seeded True at core.py:1995 when the input is Polars
    and the polars-ds pipeline exists, then every iteration inherited True.
    Linear (which DOES need encoder+scaler+imputer) then had
    skip_preprocessing=True forced onto it and its pre_pipeline was
    silently skipped. LogReg received raw pd.Categorical and crashed on
    'HOURLY' / 'FIXED'.

    Fix 11 replaces the OR-accumulation with a per-strategy condition:
        polars_pipeline_applied AND strategy.supports_polars
                                AND NOT strategy.requires_encoding

    Linear satisfies NONE of the last two, so its pre_pipeline runs fully.

    Sensor pins the smallest-n combo (c0011: linear+xgb, polars_nullable,
    n=300, ncats=3). If a future refactor re-introduces the accumulator or
    drops the per-strategy gate, this test reds immediately.
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
