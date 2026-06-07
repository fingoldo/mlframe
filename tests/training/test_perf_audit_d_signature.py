"""Regression tests for Audit D efficiency / caching fixes.

Audit D 2026-05-18:
- P0-1 / P0-2: data_signature no longer full-materialises every column.
- P0-3: ``_content_fingerprint_for_cache`` returns a fresh-per-call sentinel for nested dtypes
  (List / Array / Struct) so two consecutive cache lookups with the same id() but different
  content NEVER collide.
- P1-2: ``_row_order_fingerprint`` polars path catches inner reorders inside the first 256 rows.
- P2-1: ``_PRE_PIPELINE_CACHE_MAX`` default bumped 4 → 8.

These tests are sub-1s; marked ``fast`` for the inner-loop test runs.
"""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")
import numpy as np

from mlframe.training.composite.cache import data_signature, _row_order_fingerprint
from mlframe.training._pipeline_helpers import (
    _PRE_PIPELINE_CACHE_MAX,
    _content_fingerprint_for_cache,
    _fresh_uncachable,
)


@pytest.mark.fast
def test_data_signature_is_stable_for_same_frame() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
    sig1 = data_signature(df, "target", ["a"])
    sig2 = data_signature(df, "target", ["a"])
    assert sig1 == sig2


@pytest.mark.fast
def test_data_signature_inner_row_reorder_changes_signature() -> None:
    """P1-2: an inner shuffle that does not touch first/last n_edge rows must burst the cache.

    Build a frame >> 256 rows so the hash_rows().slice(0, 256) prefix catches reorders inside the
    sampled prefix; a shuffle that touches ANY of the first 256 rows changes the row hashes.
    """
    n = 1000
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"a": np.arange(n), "target": rng.integers(0, 2, size=n)})

    # Shuffle the middle range: rows [50:500] reordered, head [0:50] and tail [-50:] untouched.
    idx = np.arange(n)
    inner = idx[50:500].copy()
    rng.shuffle(inner)
    idx[50:500] = inner
    df_inner = df[idx.tolist()]

    sig_orig = data_signature(df, "target", ["a"])
    sig_inner = data_signature(df_inner, "target", ["a"])
    assert sig_orig != sig_inner, (
        "Inner reorder must produce a different signature; otherwise stale cache hits occur."
    )


@pytest.mark.fast
def test_row_order_fingerprint_polars_inner_reorder_changes_fingerprint() -> None:
    """Direct unit test for ``_row_order_fingerprint`` polars path."""
    n = 500
    df = pl.DataFrame({"a": np.arange(n), "b": np.arange(n) * 2})
    fp_orig = _row_order_fingerprint(df)
    # Swap rows 10 and 200 -- both inside the 256-row prefix.
    idx = np.arange(n)
    idx[10], idx[200] = idx[200], idx[10]
    df_swap = df[idx.tolist()]
    fp_swap = _row_order_fingerprint(df_swap)
    assert fp_orig != fp_swap


@pytest.mark.fast
def test_uncachable_sentinel_cross_target_isolation() -> None:
    """P0-3: two frames containing List dtype columns must NEVER reuse each other's cache key.

    Construct two polars frames with embedding columns (List(Float64)) but different content.
    Both fingerprint calls should return SENTINEL tuples that compare UNEQUAL to each other --
    so the cache key (which folds the fingerprint) cannot collide cross-target.
    """
    df1 = pl.DataFrame({"emb": [[1.0, 2.0, 3.0]], "x": [1]})
    df2 = pl.DataFrame({"emb": [[4.0, 5.0, 6.0]], "x": [2]})
    fp1 = _content_fingerprint_for_cache(df1)
    fp2 = _content_fingerprint_for_cache(df2)
    # Both flagged uncachable...
    assert fp1[0] == "uncached"
    assert fp2[0] == "uncached"
    # ...but the sentinels are distinct instances so the tuples are NOT equal.
    assert fp1 != fp2, (
        "Two uncachable fingerprints must compare unequal so cross-target cache hits "
        "cannot occur on embedding-bearing frames (P0-3)."
    )


@pytest.mark.fast
def test_uncachable_sentinel_is_fresh_per_call() -> None:
    """``_fresh_uncachable`` must yield non-equal tuples on each call."""
    a = _fresh_uncachable()
    b = _fresh_uncachable()
    assert a != b
    assert a[0] == "uncached" == b[0]


@pytest.mark.fast
def test_pre_pipeline_cache_default_max_is_at_least_eight() -> None:
    """P2-1: typical suite has 5 distinct models; default must fit without eviction."""
    assert _PRE_PIPELINE_CACHE_MAX >= 8, (
        f"_PRE_PIPELINE_CACHE_MAX={_PRE_PIPELINE_CACHE_MAX}; "
        "audit D P2-1 bumped default to 8 to avoid thrashing on cb+lgb+xgb+mlp+linear."
    )
