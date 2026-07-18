"""Regression tests for the F4 (efficiency / hoisting) audit wave.

Covers:
    - VIEW-CACHE-NOT-WIPED (``_release_ctx_polars_frames`` clears _pandas_view_cache entries).
    - VIEW-CACHE-NO-EVICT (cache bounded at 4 entries).
    - FP-KEY-OMITS-CONTENT (cache key folds id(train_df) + ncols).
    - CACHE-KEY-CONTENT-OMITTED-POLARS-SCHEMA (dtype suffix on polars frames).
    - CAT-DRIFT-FULL-IMPLODE (drift snapshot caches train-side implode on ctx).
    - SLUGIFY-PER-TGT (memoized slugify).
    - POLARS-PANDAS-CHURN (recurrent numpy-coercion cache).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pl = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# CACHE-KEY-CONTENT-OMITTED-POLARS-SCHEMA
# ---------------------------------------------------------------------------


def test_compute_pipeline_cache_key_folds_polars_dtype():
    """Compute pipeline cache key folds polars dtype."""
    from mlframe.training.core._phase_train_one_target import _compute_pipeline_cache_key

    df_i32 = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int32)})
    df_i64 = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int64)})
    k_i32 = _compute_pipeline_cache_key("tree", "ord", (False, False), True, [], [], [], train_df=df_i32)
    k_i64 = _compute_pipeline_cache_key("tree", "ord", (False, False), True, [], [], [], train_df=df_i64)
    assert k_i32 != k_i64, "Int32 vs Int64 schema must produce distinct cache keys"


def test_compute_pipeline_cache_key_stable_without_train_df():
    """Compute pipeline cache key stable without train df."""
    from mlframe.training.core._phase_train_one_target import _compute_pipeline_cache_key

    k1 = _compute_pipeline_cache_key("tree", "ord", (False, False), True, [], [], [])
    k2 = _compute_pipeline_cache_key("tree", "ord", (False, False), True, [], [], [])
    assert k1 == k2


# ---------------------------------------------------------------------------
# CAT-DRIFT-FULL-IMPLODE
# ---------------------------------------------------------------------------


def test_drift_snapshot_uses_ctx_cache():
    """Drift snapshot uses ctx cache."""
    from mlframe.training.core._phase_helpers import _log_cardinality_and_drift_snapshot

    df = pl.DataFrame({"cat": ["a", "b", "c"] * 50})
    val_df = pl.DataFrame({"cat": ["a", "b", "z"] * 50})
    test_df = pl.DataFrame({"cat": ["a", "q", "c"] * 50})
    ctx = SimpleNamespace(_cat_drift_implode_cache={})
    _log_cardinality_and_drift_snapshot(
        train_df=df,
        val_df=val_df,
        test_df=test_df,
        cat_features=["cat"],
        text_features=[],
        embedding_features=[],
        ctx=ctx,
    )
    # The cache must now hold one entry keyed by (id(df), ('cat',)).
    assert len(ctx._cat_drift_implode_cache) == 1
    assert next(iter(ctx._cat_drift_implode_cache.values()))["cat"] == {"a", "b", "c"}


def test_drift_snapshot_without_ctx_does_not_crash():
    """Drift snapshot without ctx does not crash."""
    from mlframe.training.core._phase_helpers import _log_cardinality_and_drift_snapshot

    df = pl.DataFrame({"cat": ["a", "b", "c"] * 50})
    val_df = pl.DataFrame({"cat": ["a", "b", "z"] * 50})
    test_df = pl.DataFrame({"cat": ["a", "q", "c"] * 50})
    # ctx=None falls back to the pre-fix recompute path; behaviour must stay identical.
    _log_cardinality_and_drift_snapshot(
        train_df=df,
        val_df=val_df,
        test_df=test_df,
        cat_features=["cat"],
        text_features=[],
        embedding_features=[],
        ctx=None,
    )


# ---------------------------------------------------------------------------
# SLUGIFY-PER-TGT
# ---------------------------------------------------------------------------


def test_cached_slugify_is_memoized():
    """Cached slugify is memoized."""
    from mlframe.training.core._phase_train_one_target import _cached_slugify

    s1 = _cached_slugify("Some Target Name 42")
    s2 = _cached_slugify("Some Target Name 42")
    assert s1 == s2
    # functools.lru_cache exposes the cache_info introspection.
    info = _cached_slugify.cache_info()
    assert info.hits >= 1


# ---------------------------------------------------------------------------
# POLARS-PANDAS-CHURN
# ---------------------------------------------------------------------------


def test_recurrent_numpy_cache_returns_same_array():
    """Recurrent numpy cache returns same array."""
    import pandas as pd
    from mlframe.training.core._phase_recurrent import _coerce_features_to_float32

    df = pd.DataFrame({"x": np.arange(100, dtype=np.float32), "y": np.arange(100, dtype=np.float32)})
    cache: dict = {}
    arr1 = _coerce_features_to_float32(df, cache=cache, cache_key=("train", id(df)))
    arr2 = _coerce_features_to_float32(df, cache=cache, cache_key=("train", id(df)))
    assert arr1 is arr2, "second call must hit the cache and return the same ndarray"
    assert arr1.dtype == np.float32


def test_recurrent_coerce_returns_float32_for_int_input():
    """Recurrent coerce returns float32 for int input."""
    import pandas as pd
    from mlframe.training.core._phase_recurrent import _coerce_features_to_float32

    df = pd.DataFrame({"x": np.arange(10, dtype=np.int64)})
    arr = _coerce_features_to_float32(df)
    assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# Cached init signature (SIG-IN-EXCEPT)
# ---------------------------------------------------------------------------


def test_cached_init_params_returns_consistent_set():
    """Cached init params returns consistent set."""
    from mlframe.training.core._phase_train_one_target import _cached_init_params

    class _Foo:
        """Dummy class with a 3-arg constructor, used to check _cached_init_params' return set is stable across calls."""

        def __init__(self, a, b, c=3):
            pass

    p1 = _cached_init_params(_Foo)
    p2 = _cached_init_params(_Foo)
    assert p1 == {"a", "b", "c"}
    assert p1 is p2, "second call must return cached set"
