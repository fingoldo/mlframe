"""Regression tests for audits/full_audit_2026-07-21/training_misc_small.md findings F1-F8.

Also covers a bug DISCOVERED while fixing F3/F4 (not in the audit report): the id-reuse-safe
content-hash helpers ``_full_x_content_hash``/``_full_target_content_hash`` in
``mlframe.training.pipeline._pipeline_cache`` had their OWN (id, shape)-keyed memo return a STALE
digest when CPython reuses a just-freed array's memory address for a new, differently-content
array of matching shape -- the exact bug class F3/F4 fix, one layer down. Fixed with a
weakref-validated cache entry.
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd
import pytest

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# F1 (P0): build_slice_eval_sets(source="both") bypassed GroupKFold for the random-shard half
# ---------------------------------------------------------------------------


def test_f1_source_both_with_group_ids_uses_groupkfold(caplog):
    """source='both' + group_ids now switches the random-shard half to GroupKFold (query-safe),
    same as source='random' always did, and still appends the fairness shards."""
    from mlframe.training.slicing._slice_helpers import build_slice_eval_sets

    rng = np.random.default_rng(0)
    n = 1000
    val_X = rng.normal(size=(n, 3))
    val_y = rng.normal(size=n)
    group_ids = np.repeat(np.arange(50), 20)
    subgroups = {"seg_a": np.arange(0, 200)}

    with caplog.at_level(logging.WARNING, logger="mlframe.training.slicing._slice_helpers"):
        result = build_slice_eval_sets(
            val_X, val_y, source="both", k=5, min_rows_per_shard=10,
            group_ids=group_ids, indexed_subgroups=subgroups,
        )
    assert any("switching the random-shard" in r.getMessage() for r in caplog.records)

    random_shards = [s for s in result if s.name.startswith("valid_shard_r")]
    assert random_shards
    for shard in random_shards:
        for g in set(group_ids[shard.row_indices]):
            group_rows = np.where(group_ids == g)[0]
            assert set(group_rows).issubset(set(shard.row_indices)), f"group {g} split across shards"

    fairness_shards = [s for s in result if s.name.startswith("valid_shard_f")]
    assert fairness_shards, "source='both' must still produce the fairness half"


def test_f1_source_random_alone_still_works():
    """source='random' + group_ids (the pre-existing, already-fixed path) is unaffected by the F1 restructure."""
    from mlframe.training.slicing._slice_helpers import build_slice_eval_sets

    rng = np.random.default_rng(0)
    n = 500
    val_X = rng.normal(size=(n, 2))
    val_y = rng.normal(size=n)
    group_ids = np.repeat(np.arange(25), 20)

    result = build_slice_eval_sets(val_X, val_y, source="random", k=5, min_rows_per_shard=10, group_ids=group_ids)
    assert result
    assert all(s.name.startswith("valid_shard_r") for s in result)
    for shard in result:
        for g in set(group_ids[shard.row_indices]):
            group_rows = np.where(group_ids == g)[0]
            assert set(group_rows).issubset(set(shard.row_indices))


# ---------------------------------------------------------------------------
# F2 (P1): slice_min_delta_in_se silently no-op'd unless slice_persist_history=True
# ---------------------------------------------------------------------------


def test_f2_slice_min_delta_in_se_works_without_persist_history():
    """The SE-scaled min_delta now actually applies with slice_persist_history=False (its default)."""
    import time

    from mlframe.training.callbacks._callbacks import UniversalCallback

    cb = UniversalCallback(
        patience=10, min_delta=0.5, monitor_dataset="valid_0", monitor_metric="loss",
        mode="min", slice_k=3, slice_aggregate_mode="mean", slice_aggregate_confidence=0.9,
        slice_min_delta_in_se=1.0, slice_persist_history=False, verbose=0,
    )
    cb.start_time = time.time()
    cb.last_reporting_ts = time.time()
    cb.iter = 0

    for name, v in {"valid_0": 1.0, "shard_1": 1.0, "shard_2": 1.2, "shard_3": 0.8}.items():
        cb.metric_history.setdefault(name, {}).setdefault("loss", []).append(float(v))
    cb.should_stop()

    assert cb._last_slice_shard_values == [1.0, 1.2, 0.8]
    effective = cb._effective_min_delta(cb._last_slice_shard_values)
    assert effective != cb.min_delta
    assert cb.slice_shard_score_history == [], "memory-opt behavior (no full history) must be preserved"


# ---------------------------------------------------------------------------
# F3 / F4 (P1 x2): id(target) reused as a "did the label change" cache check
# ---------------------------------------------------------------------------


def test_f3_f4_content_fingerprint_distinguishes_id_colliding_targets():
    """The train/val Pool caches now key label-swap detection off content, not id() -- verified by
    forcing an id() collision between two DIFFERENT target arrays via del + same-size reallocation."""
    from mlframe.training.pipeline._pipeline_cache import _full_target_content_hash

    gc.collect()
    arr1 = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    id1 = id(arr1)
    sig1 = _full_target_content_hash(arr1)
    del arr1
    gc.collect()

    arr2 = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    id2 = id(arr2)
    sig2 = _full_target_content_hash(arr2)

    assert id1 == id2, "test setup: expected the id() collision to reproduce on this platform"
    assert sig1 != sig2, "F3/F4 REGRESSION: content fingerprint must differ for different content"


def test_f3_f4_cb_pool_attrs_use_content_sig_not_id():
    """_cb_pool_build.py / _cb_pool.py no longer set/read the id()-based _mlframe_last_target_id attribute."""
    import inspect

    from mlframe.training.cb import _cb_pool, _cb_pool_build

    for mod in (_cb_pool_build, _cb_pool):
        src = inspect.getsource(mod)
        assert "_mlframe_last_target_id" not in src
        assert "_mlframe_last_target_sig" in src


def test_f3_f4_cb_pool_reuse_hit_with_label_swap_does_not_crash(caplog, monkeypatch):
    """Discovered while re-verifying F3/F4: the content-sig rename left the cache-hit log line's
    ternary still reading the pre-rename local ``last_target_id`` name (distinct from the
    ``_mlframe_last_target_id`` attribute the source-inspection test above checks for), which raised
    ``NameError`` on every reused-Pool hit that swapped the label. Two calls with the SAME train_df
    but DIFFERENT train_target content force the cache-hit + label-swap branch.

    The installed catboost==1.2.10 in this environment has no ``Pool.set_label`` (only
    ``set_weight``/``get_label``), so ``_cb_reuse_capable()`` is False here and the whole
    reuse fast-path normally no-ops (by design -- see ``_phase_config_setup.py``'s
    "Pool.set_label/set_weight not available" log line). The crash this test guards against only
    manifests on a catboost build that DOES expose ``set_label``, so the capability gate and the
    method itself are monkeypatched to exercise that code path deterministically regardless of the
    catboost version actually installed."""
    pytest.importorskip("catboost")
    from catboost import Pool as _Pool

    from mlframe.training.cb import _cb_pool
    from mlframe.training.cb._cb_pool import _CB_POOL_CACHE
    from mlframe.training.cb._cb_pool_build import _maybe_get_or_build_cb_pool

    monkeypatch.setattr(_cb_pool, "_cb_reuse_capable", lambda: True)
    if not callable(getattr(_Pool, "set_label", None)):
        monkeypatch.setattr(_Pool, "set_label", lambda self, y: None, raising=False)

    _CB_POOL_CACHE.clear()
    try:
        rng = np.random.default_rng(0)
        train_df = pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"])
        target1 = rng.normal(size=20).astype(np.float64)
        target2 = target1 + 1.0  # different content, same shape/dtype -- forces a label swap on hit

        fit_params: dict = {}
        pool1 = _maybe_get_or_build_cb_pool("CatBoostRegressor", None, train_df, target1, dict(fit_params))
        assert pool1 is not None, "test setup: Pool-reuse fast path did not activate despite the capability gate being forced True"

        with caplog.at_level(logging.INFO, logger="mlframe.training.cb._cb_pool_build"):
            pool2 = _maybe_get_or_build_cb_pool("CatBoostRegressor", None, train_df, target2, dict(fit_params))
        assert pool2 is pool1, "expected a cache hit (same train_df) reusing the same Pool object"
        assert any(
            "swapped weight + label" in r.getMessage() for r in caplog.records
        ), f"F3/F4 REGRESSION: cache-hit label-swap log line must not raise/silently vanish; records={[r.getMessage() for r in caplog.records]}"

        # Weight-only swap (same target content again): must also log cleanly, without "+ label".
        with caplog.at_level(logging.INFO, logger="mlframe.training.cb._cb_pool_build"):
            caplog.clear()
            fit_params["sample_weight"] = np.ones(20)
            pool3 = _maybe_get_or_build_cb_pool("CatBoostRegressor", None, train_df, target2, dict(fit_params))
        assert pool3 is pool1
        hit_messages = [r.getMessage() for r in caplog.records if "[cb-pool-reuse] hit" in r.getMessage()]
        assert hit_messages, f"expected a cache-hit log line; records={[r.getMessage() for r in caplog.records]}"
        assert not hit_messages[0].endswith("+ label"), f"weight-only swap must not report a label swap: {hit_messages[0]!r}"
    finally:
        _CB_POOL_CACHE.clear()


# ---------------------------------------------------------------------------
# Discovered while fixing F3/F4: _pipeline_cache's own (id, shape) memo had the same vulnerability
# ---------------------------------------------------------------------------


def test_discovered_pipeline_cache_hash_memo_survives_id_reuse():
    """_full_target_content_hash / _full_x_content_hash's internal LRU memo is now weakref-validated,
    so a stale digest from a freed array is never returned for a new array reusing its address."""
    from mlframe.training.pipeline._pipeline_cache import _full_target_content_hash, _full_x_content_hash

    gc.collect()
    t1 = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    ts1 = _full_target_content_hash(t1)
    del t1
    gc.collect()
    t2 = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    ts2 = _full_target_content_hash(t2)
    assert ts1 != ts2

    gc.collect()
    x1 = np.zeros((3, 2), dtype=np.float64)
    xs1 = _full_x_content_hash(x1)
    del x1
    gc.collect()
    x2 = np.ones((3, 2), dtype=np.float64)
    xs2 = _full_x_content_hash(x2)
    assert xs1 != xs2

    # Memo must still hit correctly for GENUINE repeats on the SAME object.
    arr = np.array([9, 9, 9], dtype=np.int64)
    assert _full_target_content_hash(arr) == _full_target_content_hash(arr)


# ---------------------------------------------------------------------------
# F5 (P1): compute_learning_curve defaulted to process-based joblib parallelism (defeats the
# module's own zero-copy RAM-safety design for its default parallel path)
# ---------------------------------------------------------------------------


def test_f5_learning_curve_default_parallel_backend_is_threads(monkeypatch):
    """compute_learning_curve's default parallel_backend is 'threads', not the old hardcoded 'processes'."""
    import joblib

    from sklearn.linear_model import Ridge
    from sklearn.metrics import get_scorer

    import mlframe.training.diagnostics.learning_curve as lc_mod

    captured = []
    orig_parallel = joblib.Parallel

    class _Spy(orig_parallel):
        """Spy."""
        def __init__(self, *a, **kw):
            captured.append(kw.get("prefer"))
            super().__init__(*a, **kw)

    monkeypatch.setattr(joblib, "Parallel", _Spy)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(scale=0.1, size=200)
    scorer = get_scorer("r2")

    lc_mod.compute_learning_curve(lambda: Ridge(), X, y, scorer=scorer, n_jobs=2)
    assert captured[-1] == "threads"

    lc_mod.compute_learning_curve(lambda: Ridge(), X, y, scorer=scorer, n_jobs=2, parallel_backend="processes")
    assert captured[-1] == "processes"


# ---------------------------------------------------------------------------
# F6 (P2): CatBoostRanker null-fill guard only handled object-dtype cat columns, not pandas
# CategoricalDtype columns with nulls
# ---------------------------------------------------------------------------


def test_f6_cb_ranker_fits_with_nullable_categorical_dtype_column():
    """A pandas CategoricalDtype cat_feature column with null cells no longer crashes _fit_cb_ranker."""
    from mlframe.training.ranking.ranking import _fit_cb_ranker

    rng = np.random.default_rng(0)
    n = 200
    group_ids = np.repeat(np.arange(20), n // 20)
    y = rng.integers(0, 4, size=n)
    cat_series = pd.Series(pd.Categorical(rng.choice(["a", "b", "c"], size=n)))
    cat_series[rng.choice(n, size=15, replace=False)] = np.nan
    X = pd.DataFrame({"num_feat": rng.normal(size=n), "cat_col": cat_series})
    assert isinstance(X["cat_col"].dtype, pd.CategoricalDtype)
    assert X["cat_col"].isna().any()

    result = _fit_cb_ranker(
        X, y, group_ids, None, None, None,
        obj_kwargs={}, model_kwargs={"iterations": 5, "verbose": False},
        cat_features=["cat_col"],
        early_stopping_rounds=None, verbose=False,
    )
    assert result["model"] is not None


def test_f6_unfilled_categorical_null_genuinely_crashes_catboost():
    """Sanity: confirms the pre-fix failure signature is real (a CatBoostError on unfilled category-dtype NaN)."""
    from catboost import CatBoostRanker, Pool

    rng = np.random.default_rng(0)
    n = 200
    group_ids = np.repeat(np.arange(20), n // 20)
    y = rng.integers(0, 4, size=n)
    cat_series = pd.Series(pd.Categorical(rng.choice(["a", "b", "c"], size=n)))
    cat_series[rng.choice(n, size=15, replace=False)] = np.nan
    X = pd.DataFrame({"num_feat": rng.normal(size=n), "cat_col": cat_series})

    with pytest.raises(Exception, match="NaN"):
        pool = Pool(X, label=y, group_id=group_ids, cat_features=["cat_col"])
        CatBoostRanker(iterations=5, verbose=False).fit(pool)


# ---------------------------------------------------------------------------
# F7 (P2): stale comment claimed a units switch (days->seconds) that was never implemented
# ---------------------------------------------------------------------------


def test_f7_recency_weight_subday_span_math_matches_corrected_comment():
    """Pins the ACTUAL mechanism (ratio-cancellation via _log_min_age) the corrected comment now describes."""
    span_days = 0.3
    min_age_days = 1.0 / 86400.0
    log_min_age = np.log(min_age_days)
    max_drop_factor = np.log(span_days) - log_min_age
    assert max_drop_factor > 0, "sub-day span must still yield a non-negative max_drop factor"
    assert np.isclose(max_drop_factor, np.log(span_days / min_age_days))
    assert np.isclose(max_drop_factor, np.log(span_days * 86400))  # == log(span in seconds)


# ---------------------------------------------------------------------------
# F8 (P2): HGBStrategy duplicated strategies/__init__.py's polars-categorical-detection helpers
# ---------------------------------------------------------------------------


def test_f8_hgb_strategy_uses_shared_polars_categorical_helpers():
    """hgb.py no longer defines its own copy of _polars_categorical_dtypes/_is_polars_categorical/_get_polars_cat_columns."""
    import inspect

    from mlframe.training.strategies import hgb as hgb_mod

    src = inspect.getsource(hgb_mod)
    assert "def _polars_categorical_dtypes" not in src
    assert "def _is_polars_categorical" not in src
    assert "def _get_polars_cat_columns" not in src
    assert "from . import get_polars_cat_columns" in src


def test_f8_hgb_polars_categorical_detection_still_works():
    """End-to-end: HGBStrategy's cat-column detection (now delegated) still finds Categorical/Utf8/String columns."""
    import polars as pl

    from mlframe.training.strategies import get_polars_cat_columns

    df = pl.DataFrame({
        "num_col": [1.0, 2.0, 3.0],
        "cat_col": pl.Series(["a", "b", "c"], dtype=pl.Categorical),
        "str_col": ["x", "y", "z"],
    })
    detected = set(get_polars_cat_columns(df))
    assert detected == {"cat_col", "str_col"}
