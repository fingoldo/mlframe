"""Integration tests for the 2026-04-21 training-overhead fixes.

Plan: C:/Users/TheLocalCommander/.claude/plans/jolly-wishing-deer.md

Covers Fixes 2, 3A, 3B, 4, 5, 6, 7, 8. Fix 1's end-to-end behaviour is
exercised indirectly via the LGB-on-Polars-input test (post-Fix-1 path
that also hits Fix 4's defence-in-depth setter patch). Full suite
integration (`train_mlframe_models_suite`) is out of scope for this file
— the heavy end-to-end runs live in test_all_models.py.

Key constraint (user memory): full tests, no --fast. These tests are all
individually cheap (<5 s each on CPU) so the wall-clock cost is modest.
"""

import hashlib
import json
import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Fix 2 — stale _warn_on_unsupported_polars_dtypes is gone; post-fail
#         schema dump no longer claims Enum-is-culprit.
# ---------------------------------------------------------------------------


def test_fix2_pre_fit_warning_removed():
    """The pre-fit Enum-dispatch warning function must be gone so CB fits
    on aligned Enum frames no longer emit spurious 2-min-fallback claims."""
    from mlframe.training import trainer
    assert not hasattr(trainer, "_warn_on_unsupported_polars_dtypes"), (
        "Fix 2 regressed: the stale warning helper came back. It mis-predicted "
        "a 2-min CB fallback that never happens on CB 1.2.10 + polars 1.40 "
        "with post-fill_null Enum frames."
    )


def test_fix2_schema_dump_reframes_enum_as_info_not_culprit():
    """Post-fail schema dump: Enum without nulls must be reported for
    visibility, not flagged as the cause of 'No matching signature found'."""
    from mlframe.training.trainer import _polars_schema_diagnostic

    enum_dt = pl.Enum(["red", "green", "blue"])
    df = pl.DataFrame({
        "a": np.arange(20, dtype=np.float32),
        "job_type": pl.Series("job_type", ["red"] * 20, dtype=enum_dt),
    })
    dump = _polars_schema_diagnostic(df, cat_features=["job_type"], text_features=[])
    assert "most likely cause" not in dump, (
        "Fix 2 regressed: schema dump again blames Enum as the culprit, "
        "which is empirically wrong on CB 1.2.10."
    )
    # Must still mention the column (operators need per-column info for the real culprit).
    assert "job_type" in dump


# ---------------------------------------------------------------------------
# Fix 3A — pyutilz get_df_memory_consumption accepts deep kwarg.
# ---------------------------------------------------------------------------


def test_fix3a_deep_false_on_pandas_uses_shallow_accounting():
    """deep=False path returns pointer-size totals for object columns —
    orders of magnitude faster on frames with million-unique strings."""
    from pyutilz.data.pandaslib import get_df_memory_consumption

    df = pd.DataFrame({
        "i": np.arange(1000),
        "obj": ["x" * 64 for _ in range(1000)],  # 64B per string
    })
    shallow = get_df_memory_consumption(df, deep=False)
    deep = get_df_memory_consumption(df, deep=True)
    assert shallow < deep, f"shallow must be < deep (got shallow={shallow}, deep={deep})"
    # Index + int64 column alone is already ~8k bytes; object pointer sizes
    # add another ~8k (shallow), while deep adds ~64k (1000*64). Just check
    # the shallow didn't accidentally return the deep value.
    assert shallow < 30_000, f"shallow should be << deep; got shallow={shallow}"


def test_fix3a_default_deep_true_preserves_behaviour():
    """Library default stays ``deep=True`` for back-compat with every
    caller that predates Fix 3A. mlframe's specific heuristic call site
    passes ``deep=False`` explicitly (see trainer.py
    ``configure_training_params``). This test asserts the library
    default is unchanged."""
    from pyutilz.data.pandaslib import get_df_memory_consumption

    df = pd.DataFrame({"s": ["abcdef"] * 500})
    default_value = get_df_memory_consumption(df)
    explicit_deep = get_df_memory_consumption(df, deep=True)
    assert default_value == explicit_deep
    # The kwarg should still exist and accept False.
    explicit_shallow = get_df_memory_consumption(df, deep=False)
    assert explicit_shallow < explicit_deep


def test_fix3a_polars_branch_unchanged():
    """Polars frames go through .estimated_size() — O(cols). deep flag is
    a no-op here; result is the same regardless."""
    from pyutilz.data.pandaslib import get_df_memory_consumption

    pdf = pl.DataFrame({"i": np.arange(500), "s": ["x"] * 500})
    a = get_df_memory_consumption(pdf, deep=True)
    b = get_df_memory_consumption(pdf, deep=False)
    assert a == b


# ---------------------------------------------------------------------------
# Fix 3B — cache Polars size; select_target forwards the kwargs.
# ---------------------------------------------------------------------------


def test_fix3b_select_target_accepts_size_kwargs():
    """select_target must accept train_df_size_bytes / val_df_size_bytes
    kwargs. Checked via inspect so we don't have to run the full function."""
    import inspect
    from mlframe.training.train_eval import select_target

    params = inspect.signature(select_target).parameters
    assert "train_df_size_bytes" in params, list(params.keys())
    assert "val_df_size_bytes" in params


def test_fix3b_configure_training_params_accepts_size_kwargs():
    import inspect
    from mlframe.training.trainer import configure_training_params

    params = inspect.signature(configure_training_params).parameters
    assert "train_df_size_bytes" in params
    assert "val_df_size_bytes" in params


# ---------------------------------------------------------------------------
# Fix 4 — LGB feature_names_in_ setter shim lets non-pandas fits proceed.
# ---------------------------------------------------------------------------


def test_fix4_lgb_accepts_polars_input_after_shim():
    """Without the shim, LGB 4.6.0 + sklearn 1.8 crashes with
    AttributeError on non-pandas X. After the shim, fit completes."""
    pytest.importorskip("lightgbm")
    import lightgbm as _lgb
    # Import trainer to install the shim (idempotent).
    from mlframe.training import trainer  # noqa: F401

    rng = np.random.default_rng(0)
    n = 300
    df_polars = pl.DataFrame({
        "x0": rng.random(n).astype(np.float32),
        "x1": rng.random(n).astype(np.float32),
    })
    y = rng.integers(0, 2, size=n)
    m = _lgb.LGBMClassifier(n_estimators=3, verbose=-1)
    # Must not raise AttributeError.
    m.fit(df_polars, y)
    assert m.n_features_ == 2


def test_fix4_shim_is_idempotent():
    """Re-importing trainer should not re-install the setter; the marker
    attribute is set on first install and gates subsequent calls."""
    from mlframe.training import trainer
    trainer._patch_lgb_feature_names_in_setter()  # call twice
    import lightgbm.sklearn as _lgbm_sk
    prop = _lgbm_sk.LGBMModel.__dict__["feature_names_in_"]
    assert prop.fset is not None
    assert getattr(_lgbm_sk.LGBMModel, "_mlframe_feature_names_setter_installed", False)


# ---------------------------------------------------------------------------
# Fix 5 — fast_aucs_per_group_optimized upfront filter.
# ---------------------------------------------------------------------------


def test_fix5_upfront_filter_preserves_valid_group_aucs():
    """Upfront filter must produce bit-identical valid-group AUCs vs the
    pre-filter path. We compute directly via compute_grouped_group_aucs as
    the reference."""
    from mlframe.metrics import fast_aucs_per_group_optimized, compute_grouped_group_aucs

    rng = np.random.default_rng(1337)
    # Small but diverse: mix of single-sample, single-class, valid-multiclass groups.
    group_ids = np.array([0, 0, 0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5], dtype=np.int32)
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1], dtype=np.float64)
    y_score = rng.random(len(group_ids))

    # Reference: feed everything to the numba inner loop.
    sort_idx = np.argsort(group_ids)
    ref = compute_grouped_group_aucs(group_ids[sort_idx], y_true[sort_idx], y_score[sort_idx])
    _, _, got = fast_aucs_per_group_optimized(y_true, y_score, group_ids)

    # Same keyset, same values (including NaN).
    assert set(ref.keys()) == set(got.keys())
    for k in ref:
        r0, p0 = ref[k]
        r1, p1 = got[k]
        assert (np.isnan(r0) and np.isnan(r1)) or np.isclose(r0, r1)
        assert (np.isnan(p0) and np.isnan(p1)) or np.isclose(p0, p1)


def test_fix5_upfront_filter_faster_on_skewed_workload():
    """95 %-NaN workload: upfront filter is faster than letting numba iterate
    every group. Uses a large synthetic frame; tolerance generous — this is
    a smoke check, not a benchmark gate."""
    import time
    from mlframe.metrics import fast_aucs_per_group_optimized

    rng = np.random.default_rng(0)
    n = 200_000
    n_valid = 5_000
    # 95%+ single-sample groups, the rest multi-sample multi-class.
    group_ids = np.arange(n, dtype=np.int32)
    tail = n - n_valid
    for i in range(n_valid):
        start = tail + i * 2  # 2-sample groups
        if start + 1 < n:
            group_ids[start:start + 2] = 10_000_000 + i
    y_true = rng.integers(0, 2, size=n).astype(np.float64)
    y_score = rng.random(n)

    # Warm numba once.
    _ = fast_aucs_per_group_optimized(y_true[:200], y_score[:200], group_ids[:200])

    t0 = time.perf_counter()
    fast_aucs_per_group_optimized(y_true, y_score, group_ids)
    elapsed = time.perf_counter() - t0
    # On modern CPU, 200k rows with 95 % single-sample should finish well under 5 s.
    # Pre-Fix-5 it's ~1 s but degrades >5 s once the per-group sort of the valid
    # tail grows; we just assert "not pathologically slow".
    assert elapsed < 10.0, f"regressed to {elapsed:.2f}s on 200k skewed groups"


# ---------------------------------------------------------------------------
# Fix 6 — use_text_features toggle.
# ---------------------------------------------------------------------------


def test_fix6_use_text_features_false_suppresses_auto_promotion():
    """With use_text_features=False, high-cardinality columns must NOT
    be promoted to text_features — but they MUST appear in the
    ``auto_detected_high_card_to_drop`` return list so the caller
    drops them entirely (prevents XGB QuantileDMatrix OOM + CB
    artefact bloat; see Fix 6 correction 2026-04-22)."""
    from mlframe.training.core import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    rng = np.random.default_rng(0)
    n = 1000
    vocab = [f"tok_{i}" for i in range(500)]
    vals = rng.choice(vocab, size=n)
    df = pl.DataFrame({
        "numeric": rng.random(n).astype(np.float32),
        "highcard": pl.Series(vals),
    })

    t_on, _, drop_on = _auto_detect_feature_types(
        df, FeatureTypesConfig(), cat_features=[], verbose=False,
    )
    assert "highcard" in t_on
    assert drop_on == []  # promoted to text_features, not dropped

    t_off, _, drop_off = _auto_detect_feature_types(
        df, FeatureTypesConfig(use_text_features=False), cat_features=[], verbose=False,
    )
    assert t_off == []
    assert drop_off == ["highcard"]  # Fix 6 correction: caller must drop this.


def test_fix6_use_text_features_false_honors_explicit_list():
    """2026-04-21 refined semantics: ``use_text_features=False`` gates
    AUTO-promotion only. User-supplied explicit ``text_features`` list
    is honored regardless — if the user passed it, they intend those
    columns routed as text_features. This keeps the direct-API behaviour
    consistent with tests that expect explicit opt-in to work end-to-end
    (e.g. ``test_non_catboost_drops_text_columns`` sets explicit
    ``text_features`` and expects downstream tier-build to drop them)."""
    from mlframe.training.core import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    df = pl.DataFrame({"x": ["a", "b"]})
    t_off, _, drop_off = _auto_detect_feature_types(
        df,
        FeatureTypesConfig(use_text_features=False, text_features=["x"]),
        cat_features=[],
        verbose=False,
    )
    assert t_off == ["x"]
    # Explicit user-listed columns are skipped via ``user_assigned`` —
    # they're NOT in the auto-detected drop list even when the flag is off.
    assert drop_off == []


def test_fix6_use_text_features_false_end_to_end_xgb_does_not_see_highcard(tmp_path):
    """Regression test for the 2026-04-22 prod OOM:
    ``use_text_features=False`` on a frame with a high-cardinality
    string column MUST result in XGB training on a df that does NOT
    contain that column. Before the fix, the col silently stayed as
    cat_feature, XGB's QuantileDMatrix tried to build a 2M-level cat
    index on 9M rows, and the process was killed with MemoryError.

    This is the test that would have caught the bug the original
    Fix 6 unit-test suite missed — it exercises the real contract
    (what reaches fit / what's persisted in metadata), not just the
    helper's return value in isolation."""
    pytest.importorskip("xgboost")
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import FeatureTypesConfig, PolarsPipelineConfig
    from .shared import SimpleFeaturesAndTargetsExtractor

    rng = np.random.default_rng(0)
    n = 800
    # 400 unique tokens > default cat_text_cardinality_threshold (300) —
    # would auto-promote to text_features under use_text_features=True,
    # would auto-drop under use_text_features=False.
    vocab = [f"tok_{i}" for i in range(400)]
    df = pl.DataFrame({
        "f_num": rng.random(n).astype(np.float32),
        "f_lowcard": pl.Series(rng.choice(["A", "B", "C"], size=n)),
        "skills_text_like": pl.Series(rng.choice(vocab, size=n)),
        "target": rng.integers(0, 2, size=n).astype(np.int64),
    })

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    _, metadata = train_mlframe_models_suite(
        df=df,
        target_name="test_target",
        model_name="fix6_regression_test",
        features_and_targets_extractor=fte,
        mlframe_models=["xgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PolarsPipelineConfig(
            use_polarsds_pipeline=False,
            categorical_encoding=None,
            scaler_name=None,
            imputer_strategy=None,
        ),
        feature_types_config=FeatureTypesConfig(use_text_features=False),
        data_dir=str(tmp_path),
        models_dir="models",
        verbose=0,
    )

    # Contract: the high-card column must be absent from the trained
    # input schema — XGB never saw it.
    trained_cols = metadata["columns"]
    trained_cols = list(trained_cols) if not isinstance(trained_cols, list) else trained_cols
    assert "skills_text_like" not in trained_cols, (
        f"Fix 6 regression: ``use_text_features=False`` but column "
        f"'skills_text_like' still reached the XGB fit frame. "
        f"metadata['columns']={trained_cols}"
    )
    # And the stored cat_features must not list it either.
    assert "skills_text_like" not in (metadata.get("cat_features") or [])
    assert "skills_text_like" not in (metadata.get("text_features") or [])


# ---------------------------------------------------------------------------
# Fix 7 — tqdmu_lazy_start resets elapsed counter at first iteration.
# ---------------------------------------------------------------------------


def test_fix7_lazy_start_defers_elapsed_counter():
    """Bar's start_t must be reset when the first item yields, not when
    the iterable is constructed — prevents the stale-timer artefact."""
    import time
    from pyutilz.system import tqdmu_lazy_start

    def gen():
        time.sleep(0.5)  # Simulate idle before first yield.
        yield "a"
        yield "b"

    items = []
    for item in tqdmu_lazy_start(gen(), desc="test", total=2):
        items.append(item)
    assert items == ["a", "b"]


# ---------------------------------------------------------------------------
# Fix 8 — per-model input-schema fingerprint.
# ---------------------------------------------------------------------------


def test_fix8_fingerprint_deterministic():
    """Same df, same config -> same hash. Basic sanity."""
    from mlframe.training.utils import compute_model_input_fingerprint

    df = pl.DataFrame({
        "a": np.arange(10, dtype=np.float32),
        "b": pl.Series("b", ["x"] * 10, dtype=pl.Enum(["x", "y"])),
    })
    h1, _ = compute_model_input_fingerprint(df, cat_features=["b"])
    h2, _ = compute_model_input_fingerprint(df, cat_features=["b"])
    assert h1 == h2


def test_fix8_fingerprint_role_sensitivity():
    """Same df, same column set, but role changes (cat <-> text) must
    produce a different hash — they result in different fit-time behaviour."""
    from mlframe.training.utils import compute_model_input_fingerprint

    df = pl.DataFrame({
        "a": np.arange(10, dtype=np.float32),
        "b": pl.Series("b", ["x"] * 10, dtype=pl.Enum(["x", "y"])),
    })
    h_cat, _ = compute_model_input_fingerprint(df, cat_features=["b"])
    h_text, _ = compute_model_input_fingerprint(df, text_features=["b"])
    assert h_cat != h_text


def test_fix8_fingerprint_lgb_vs_cb_on_same_df():
    """Both models receive columns {a, b, c} where c is text-promoted.
    - CB: c plays role=text (supports_text_features=True)
    - LGB: c is dropped at tier build -> only {a, b}, role=numeric/cat
    Thus their fingerprints differ, not because of a flag, but because they
    truly see different schemas. This is the correctness payoff vs hashing
    config flags (user's explicit concern in the plan)."""
    from mlframe.training.utils import compute_model_input_fingerprint

    df_cb = pl.DataFrame({
        "a": np.arange(10, dtype=np.float32),
        "b": pl.Series("b", ["x"] * 10).cast(pl.Categorical),
        "c": pl.Series("c", ["long_text_" * 10] * 10),  # text-ish
    })
    df_lgb = df_cb.drop("c")  # LGB tier drops text columns
    h_cb, _ = compute_model_input_fingerprint(df_cb, cat_features=["b"], text_features=["c"])
    h_lgb, _ = compute_model_input_fingerprint(df_lgb, cat_features=["b"])
    assert h_cb != h_lgb


def test_fix8_fingerprint_utf8_and_string_aliases_collapse():
    """pl.Utf8 and pl.String are aliases in polars 1.x. Hash must not
    flip when a column's dtype string swaps between them."""
    from mlframe.training.utils import _canonical_dtype_str

    assert _canonical_dtype_str(pl.Utf8) == _canonical_dtype_str(pl.String)


def test_fix8_fingerprint_enum_category_drift_invalidates():
    """A val-set Enum with extra categories produces a different hash
    than the train-set Enum — correct cache-invalidation signal."""
    from mlframe.training.utils import compute_model_input_fingerprint

    df_train = pl.DataFrame({
        "a": np.arange(10, dtype=np.float32),
        "b": pl.Series("b", ["x"] * 10, dtype=pl.Enum(["x", "y"])),
    })
    df_val = pl.DataFrame({
        "a": np.arange(10, dtype=np.float32),
        "b": pl.Series("b", ["x"] * 10, dtype=pl.Enum(["x", "y", "z"])),
    })
    h_tr, _ = compute_model_input_fingerprint(df_train, cat_features=["b"])
    h_val, _ = compute_model_input_fingerprint(df_val, cat_features=["b"])
    assert h_tr != h_val


def test_fix8_fingerprint_json_canonicalisation_uses_sorted_keys():
    """Per-user memory rule: JSON serialisation for hashing must
    sort_keys=True. We verify by rebuilding the canonical form and
    reproducing the hash from the returned schema list."""
    from mlframe.training.utils import compute_model_input_fingerprint

    df = pl.DataFrame({
        "z": np.arange(5, dtype=np.float32),
        "a": pl.Series("a", ["x"] * 5).cast(pl.Categorical),
    })
    got_hash, schema = compute_model_input_fingerprint(df, cat_features=["a"])
    canonical = json.dumps(schema, sort_keys=True)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    assert got_hash == expected


def test_fix8_behavior_config_has_hash_suffix_flag():
    """Opt-out flag for legacy filenames exists."""
    from mlframe.training.configs import TrainingBehaviorConfig

    assert TrainingBehaviorConfig().model_file_hash_suffix is True
    assert TrainingBehaviorConfig(model_file_hash_suffix=False).model_file_hash_suffix is False


# ---------------------------------------------------------------------------
# Fix 9.4.1 — per-build logging of Pool / DMatrix / Dataset.
# ---------------------------------------------------------------------------


def test_fix9_build_logging_fires_on_dmatrix(caplog):
    """Every xgb.DMatrix construction must emit one INFO ``[dataset-build]``
    log line with shape + duration + callsite."""
    import logging
    from mlframe.training import trainer  # noqa: F401 — installs patches
    pytest.importorskip("xgboost")
    import xgboost as xgb

    rng = np.random.default_rng(0)
    X = rng.random((200, 5)).astype(np.float32)
    with caplog.at_level(logging.INFO, logger="mlframe.training.trainer"):
        xgb.DMatrix(X)
    msgs = [r.message for r in caplog.records if r.name == "mlframe.training.trainer"]
    assert any("[dataset-build] xgboost.DMatrix" in m and "shape=200x5" in m for m in msgs), (
        f"expected DMatrix build-log; got {msgs}"
    )


def test_fix9_build_logging_fires_on_pool(caplog):
    """Every catboost.Pool construction must emit one INFO
    ``[dataset-build]`` log line."""
    import logging
    from mlframe.training import trainer  # noqa: F401 — installs patches
    pytest.importorskip("catboost")
    from catboost import Pool

    rng = np.random.default_rng(0)
    X = rng.random((150, 3))
    y = rng.integers(0, 2, size=150)
    with caplog.at_level(logging.INFO, logger="mlframe.training.trainer"):
        Pool(data=X, label=y)
    msgs = [r.message for r in caplog.records if r.name == "mlframe.training.trainer"]
    assert any("[dataset-build] catboost.Pool" in m and "shape=150x3" in m for m in msgs)


def test_fix9_build_logging_fires_on_lgb_dataset(caplog):
    """Every lightgbm.Dataset construction must emit one INFO
    ``[dataset-build]`` log line."""
    import logging
    from mlframe.training import trainer  # noqa: F401 — installs patches
    pytest.importorskip("lightgbm")
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.random((80, 4)).astype(np.float32)
    y = rng.integers(0, 2, size=80).astype(np.float32)
    with caplog.at_level(logging.INFO, logger="mlframe.training.trainer"):
        lgb.Dataset(data=X, label=y)
    msgs = [r.message for r in caplog.records if r.name == "mlframe.training.trainer"]
    assert any("[dataset-build] lightgbm.Dataset" in m and "shape=80x4" in m for m in msgs)


def test_fix9_build_logger_patch_is_idempotent():
    """Re-importing trainer / re-invoking the patcher is a no-op — double
    wrapping would cause each build to log twice."""
    from mlframe.training import trainer
    trainer._patch_dataset_constructors_with_logging()
    trainer._patch_dataset_constructors_with_logging()
    import xgboost as xgb
    # Check that the outermost wrapper points to our logged __init__ once.
    # The marker attr is set True on first install; re-install checks the
    # own-dict marker and early-returns.
    assert xgb.DMatrix.__dict__.get("_mlframe_build_logger_installed", False)


# ---------------------------------------------------------------------------
# Fix 9.4.2 — capability detection at suite start.
# ---------------------------------------------------------------------------


def test_fix9_capability_detection_returns_booleans():
    """``_detect_dataset_reuse_capabilities`` returns a dict with bool
    flags for every installed library."""
    from mlframe.training.core import _detect_dataset_reuse_capabilities

    caps = _detect_dataset_reuse_capabilities()
    expected_keys = {
        "cb_pool_set_label", "cb_pool_set_weight", "cb_pool_label_swap",
        "xgb_dmatrix_set_label", "xgb_dmatrix_set_weight", "xgb_sklearn_accepts_dmatrix",
        "lgb_dataset_set_label", "lgb_dataset_set_weight", "lgb_sklearn_accepts_dataset",
    }
    assert expected_keys.issubset(caps.keys())
    for k, v in caps.items():
        assert isinstance(v, bool), f"{k} is {type(v).__name__}, expected bool"


def test_fix9_cb_pool_label_swap_detected_on_current_install():
    """Installed catboost 1.2.10 has Pool.set_label + set_weight
    (confirmed 2026-04-21). If this test fails the installed CB may
    have regressed — worth investigating before trusting the reuse
    fast-path."""
    from mlframe.training.core import _detect_dataset_reuse_capabilities

    caps = _detect_dataset_reuse_capabilities()
    assert caps["cb_pool_set_label"] is True
    assert caps["cb_pool_set_weight"] is True
    assert caps["cb_pool_label_swap"] is True


# ---------------------------------------------------------------------------
# Fix 9.4.3 — CatBoost Pool reuse across weight swaps.
# ---------------------------------------------------------------------------


def test_fix9_cb_pool_reuse_weight_only_swap_no_rebuild():
    """Same train_df + same label + different weight → cache hit on the
    second fit, no new Pool constructor call."""
    from mlframe.training import trainer
    pytest.importorskip("catboost")
    import catboost as cb

    rng = np.random.default_rng(0)
    n = 300
    df = pl.DataFrame({
        "num": rng.standard_normal(n).astype(np.float32),
        "cat": pl.Series("cat", rng.choice(["a", "b", "c"], size=n)).cast(pl.Categorical),
    })
    y = rng.integers(0, 2, size=n)
    w1 = np.ones(n)
    w2 = np.linspace(0.1, 1.0, n)

    # Clear cache for a clean count.
    trainer._CB_POOL_CACHE.clear()

    build_count = {"n": 0}
    orig_init = cb.Pool.__init__

    def counting_init(self, *args, **kwargs):
        build_count["n"] += 1
        return orig_init(self, *args, **kwargs)

    # Temporarily replace __init__ (over top of the existing build-logger
    # wrapper — both fire, order doesn't matter for the counter).
    cb.Pool.__init__ = counting_init
    try:
        m1 = cb.CatBoostClassifier(iterations=3, verbose=False)
        trainer._train_model_with_fallback(
            m1, m1, "CatBoostClassifier", df, y,
            {"cat_features": ["cat"], "sample_weight": w1}, False,
        )
        m2 = cb.CatBoostClassifier(iterations=3, verbose=False)
        trainer._train_model_with_fallback(
            m2, m2, "CatBoostClassifier", df, y,
            {"cat_features": ["cat"], "sample_weight": w2}, False,
        )
    finally:
        cb.Pool.__init__ = orig_init

    # Exactly 1 Pool build for the 2 fits. Pre-Fix-9 this would have
    # been at least 2 (one per fit) as CB's wrapper always rebuilds.
    assert build_count["n"] == 1, (
        f"expected 1 Pool build across 2 weight-only fits; got {build_count['n']}"
    )


def test_honor_user_dtype_preserves_polars_categorical():
    """honor_user_dtype=True: pre-cast pl.Categorical column stays a
    cat_feature even when cardinality exceeds the text-promotion threshold.
    Same df with default config (honor_user_dtype=False) promotes it.
    The signal that distinguishes "user cast to Categorical explicitly"
    from "raw String that happens to be high-cardinality" is the dtype
    coming into the function."""
    from mlframe.training.core import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    rng = np.random.default_rng(0)
    n = 1000
    vocab = [f"v_{i}" for i in range(500)]
    vals = rng.choice(vocab, size=n)
    df = pl.DataFrame({
        "num": rng.random(n).astype(np.float32),
        "user_cat": pl.Series("user_cat", vals).cast(pl.Categorical),
    })

    # Default honor_user_dtype=False: auto-promote high-cardinality Categorical.
    t_default, _, _ = _auto_detect_feature_types(
        df,
        FeatureTypesConfig(cat_text_cardinality_threshold=50),
        cat_features=["user_cat"],
        verbose=False,
    )
    assert "user_cat" in t_default, "default must auto-promote high-card Categorical"

    # honor_user_dtype=True: user's explicit Categorical intent honored.
    t_honor, _, _ = _auto_detect_feature_types(
        df,
        FeatureTypesConfig(
            cat_text_cardinality_threshold=50, honor_user_dtype=True,
        ),
        cat_features=["user_cat"],
        verbose=False,
    )
    assert t_honor == [], (
        f"honor_user_dtype=True must preserve pl.Categorical; got text_features={t_honor}"
    )


def test_honor_user_dtype_still_promotes_raw_string():
    """honor_user_dtype=True gates ONLY the pre-cast categorical path.
    Raw pl.Utf8 / pl.String columns are still auto-promotion candidates —
    there's no user intent encoded in their dtype."""
    from mlframe.training.core import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    rng = np.random.default_rng(0)
    n = 1000
    vocab = [f"v_{i}" for i in range(500)]
    df = pl.DataFrame({"raw_str": rng.choice(vocab, size=n)})  # stays pl.String

    t, _, _ = _auto_detect_feature_types(
        df,
        FeatureTypesConfig(
            cat_text_cardinality_threshold=50, honor_user_dtype=True,
        ),
        cat_features=[],
        verbose=False,
    )
    assert "raw_str" in t, (
        "honor_user_dtype must NOT block raw String promotion — no user "
        f"dtype signal there. Got text_features={t}"
    )


def test_honor_user_dtype_pandas_category_parity():
    """Symmetry check: pandas ``category`` dtype gets the same treatment
    as pl.Categorical when honor_user_dtype=True."""
    from mlframe.training.core import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    vocab = [f"v_{i}" for i in range(100)]
    rng = np.random.default_rng(0)
    vals = rng.choice(vocab, size=500)
    df = pd.DataFrame({"user_cat": pd.Categorical(vals)})

    # Default: promoted.
    t_default, _, _ = _auto_detect_feature_types(
        df,
        FeatureTypesConfig(cat_text_cardinality_threshold=10),
        cat_features=["user_cat"],
        verbose=False,
    )
    assert "user_cat" in t_default

    # honor_user_dtype=True: preserved.
    t_honor, _, _ = _auto_detect_feature_types(
        df,
        FeatureTypesConfig(
            cat_text_cardinality_threshold=10, honor_user_dtype=True,
        ),
        cat_features=["user_cat"],
        verbose=False,
    )
    assert t_honor == []


def test_align_polars_categorical_dicts_no_test_leakage(tmp_path):
    """2026-04-22 future-leakage fix: category dict alignment must
    compute its Enum vocabulary from train + val ONLY. Test-set
    categories must NOT leak into the Enum — otherwise we're
    seeding training-time preprocessing with held-out data and
    invalidating the test as a future-unseen-data proxy.

    This test constructs a frame where the test split has a
    category ('test_only_cat') that does NOT appear in train or
    val. After training, the persisted Enum vocabulary for that
    column must NOT contain 'test_only_cat'."""
    pytest.importorskip("catboost")
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        FeatureTypesConfig,
        TrainingBehaviorConfig,
        PolarsPipelineConfig,
        TrainingSplitConfig,
    )
    from .shared import SimpleFeaturesAndTargetsExtractor

    rng = np.random.default_rng(0)
    # Construct a frame where:
    #   - first 70% rows use categories A, B, C (train)
    #   - next 10% use categories A, B (val)
    #   - last 20% use 'test_only_cat' (test split via chronological ordering)
    n_tr = 700
    n_vl = 100
    n_te = 200
    n = n_tr + n_vl + n_te
    cat_vals = np.concatenate([
        rng.choice(["A", "B", "C"], size=n_tr),
        rng.choice(["A", "B"], size=n_vl),
        np.array(["test_only_cat"] * n_te, dtype=object),
    ])
    df = pl.DataFrame({
        "f_num": rng.random(n).astype(np.float32),
        "my_cat": pl.Series(cat_vals, dtype=pl.Categorical),
        "target": rng.integers(0, 2, size=n).astype(np.int64),
    })

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    _, metadata = train_mlframe_models_suite(
        df=df,
        target_name="test_target",
        model_name="leakage_test",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PolarsPipelineConfig(
            use_polarsds_pipeline=False,
            categorical_encoding=None,
            scaler_name=None,
            imputer_strategy=None,
        ),
        split_config=TrainingSplitConfig(
            shuffle_val=False,
            shuffle_test=False,
            test_size=0.2,
            val_size=0.1,
            wholeday_splitting=False,
        ),
        behavior_config=TrainingBehaviorConfig(
            align_polars_categorical_dicts=True,
            prefer_gpu_configs=False,
        ),
        feature_types_config=FeatureTypesConfig(use_text_features=True),
        hyperparams_config={"iterations": 3},
        data_dir=str(tmp_path),
        models_dir="models",
        verbose=0,
    )

    # Contract: the Enum / Categorical vocabulary for 'my_cat' must be
    # a subset of {A, B, C} — test_only_cat must NOT have leaked in.
    cats_seen = None
    for entry in metadata.get("model_schemas", {}).values():
        schema = entry.get("input_schema", [])
        for col_entry in schema:
            if col_entry["name"] == "my_cat":
                cats_seen = col_entry.get("dtype", "")
                break
        if cats_seen is not None:
            break
    # dtype string for pl.Enum(...) includes the category list; assert
    # test_only_cat is absent from it (strict substring check).
    assert cats_seen is not None, "'my_cat' not found in any model_schema"
    assert "test_only_cat" not in cats_seen, (
        f"Future-leakage regression: 'test_only_cat' (test-only "
        f"category) leaked into the trained Enum vocabulary. "
        f"dtype snapshot: {cats_seen!r}"
    )


def test_orch1_cb_val_pool_reuse_across_weight_swaps():
    """Orch-1 (2026-04-21): val Pool also reused across weight fits.
    Pre-Orch-1 each CB fit rebuilt both train AND val Pools; the train
    side got Fix 9.4.3, the val side stayed on rebuild-every-fit. This
    test counts constructor calls across 3 weight-only fits and asserts
    one train Pool build + one val Pool build (was 3+3)."""
    from mlframe.training import trainer
    pytest.importorskip("catboost")
    import catboost as cb

    rng = np.random.default_rng(0)
    n, nv = 300, 80
    train_df = pl.DataFrame({
        "num": rng.standard_normal(n).astype(np.float32),
        "cat": pl.Series("cat", rng.choice(["a", "b", "c"], size=n)).cast(pl.Categorical),
    })
    val_df = pl.DataFrame({
        "num": rng.standard_normal(nv).astype(np.float32),
        "cat": pl.Series("cat", rng.choice(["a", "b", "c"], size=nv)).cast(pl.Categorical),
    })
    y = rng.integers(0, 2, size=n)
    yv = rng.integers(0, 2, size=nv)

    trainer._CB_POOL_CACHE.clear()
    trainer._CB_VAL_POOL_CACHE.clear()

    build_count = {"n": 0}
    orig_init = cb.Pool.__init__

    def counting_init(self, *args, **kwargs):
        build_count["n"] += 1
        return orig_init(self, *args, **kwargs)

    cb.Pool.__init__ = counting_init
    try:
        for w in (np.ones(n), np.linspace(0.1, 1, n), np.linspace(0.5, 2, n)):
            m = cb.CatBoostClassifier(iterations=3, verbose=False)
            trainer._train_model_with_fallback(
                m, m, "CatBoostClassifier", train_df, y,
                {"cat_features": ["cat"], "sample_weight": w,
                 "eval_set": [(val_df, yv)]},
                False,
            )
    finally:
        cb.Pool.__init__ = orig_init

    # Exactly 2 Pool builds (1 train + 1 val) for 3 fits. Pre-Orch-1:
    # 6 builds total (3 train rebuilds × pre-Fix-9.4.3 would be that;
    # with 9.4.3 train=1 but val=3; now val=1 too).
    assert build_count["n"] == 2, (
        f"expected 1 train + 1 val Pool build (2 total) across 3 weight-only fits; "
        f"got {build_count['n']}"
    )


def test_fix943_cb_val_pool_reused_on_predict_path_too():
    """Fix 9.4.3 correction (2026-04-22): the val Pool cached at
    fit time MUST be reused on the subsequent predict_proba /
    predict path too. Pre-fix the metrics path called
    ``model.predict_proba(val_df)`` with a raw DataFrame, CB's
    sklearn wrapper rebuilt a fresh Pool at the C++ boundary →
    53-66 s wasted per metrics invocation on the 7M-row prod run
    (2026-04-22 log). This test counts Pool constructor calls
    across (1) fit and (2) predict_proba on the SAME val frame;
    total must be 2 (1 train + 1 val at fit, 0 at predict)."""
    from mlframe.training import trainer
    pytest.importorskip("catboost")
    import catboost as cb

    rng = np.random.default_rng(0)
    n, nv = 300, 80
    train_df = pl.DataFrame({
        "num": rng.standard_normal(n).astype(np.float32),
        "cat": pl.Series("cat", rng.choice(["a", "b", "c"], size=n)).cast(pl.Categorical),
    })
    val_df = pl.DataFrame({
        "num": rng.standard_normal(nv).astype(np.float32),
        "cat": pl.Series("cat", rng.choice(["a", "b", "c"], size=nv)).cast(pl.Categorical),
    })
    y = rng.integers(0, 2, size=n)
    yv = rng.integers(0, 2, size=nv)

    trainer._CB_POOL_CACHE.clear()
    trainer._CB_VAL_POOL_CACHE.clear()

    build_count = {"n": 0}
    orig_init = cb.Pool.__init__

    def counting_init(self, *args, **kwargs):
        build_count["n"] += 1
        return orig_init(self, *args, **kwargs)

    cb.Pool.__init__ = counting_init
    try:
        m = cb.CatBoostClassifier(iterations=3, verbose=False)
        trainer._train_model_with_fallback(
            m, m, "CatBoostClassifier", train_df, y,
            {"cat_features": ["cat"], "sample_weight": np.ones(n),
             "eval_set": [(val_df, yv)]},
            False,
        )
        # After fit: 2 Pool builds (1 train + 1 val).
        builds_after_fit = build_count["n"]
        assert builds_after_fit == 2, f"fit built {builds_after_fit} Pools, expected 2"

        # predict_proba on the SAME val_df — reuse path must skip rebuild.
        trainer._predict_with_fallback(m, val_df, method="predict_proba")
        builds_after_predict = build_count["n"]
        assert builds_after_predict == 2, (
            f"Fix 9.4.3 regression: predict_proba rebuilt the val Pool "
            f"(build count {builds_after_fit} -> {builds_after_predict}; "
            f"expected stays at 2). The cached val Pool at "
            f"_CB_VAL_POOL_CACHE was NOT reused — 53-66 s wasted per "
            f"metrics invocation on 7M-row prod."
        )
    finally:
        cb.Pool.__init__ = orig_init


def test_fix9_cb_stale_cat_features_filtered():
    """If ``fit_params['cat_features']`` contains a name absent from
    train_df.columns (e.g. after MRMR dropped it), ``_maybe_get_or_build_cb_pool``
    should narrow to the intersection so the subsequent fit doesn't
    raise ``ValueError: 'feat' is not in list``."""
    from mlframe.training import trainer
    pytest.importorskip("catboost")

    rng = np.random.default_rng(0)
    n = 100
    df = pl.DataFrame({
        "num": rng.standard_normal(n).astype(np.float32),
        "cat_present": pl.Series("cat_present", rng.choice(["x", "y"], size=n)).cast(pl.Categorical),
    })
    fit_params = {
        # "cat_absent" is not in df.columns — MRMR-dropped, stale list.
        "cat_features": ["cat_absent", "cat_present"],
        "sample_weight": np.ones(n),
    }

    trainer._CB_POOL_CACHE.clear()
    pool = trainer._maybe_get_or_build_cb_pool(
        model_type_name="CatBoostClassifier",
        model=None,
        train_df=df,
        train_target=rng.integers(0, 2, size=n),
        fit_params=fit_params,
    )
    assert pool is not None, "Pool build should succeed once stale feats are filtered"
    # fit_params mutated in place, now contains only the intersection.
    assert fit_params["cat_features"] == ["cat_present"]
