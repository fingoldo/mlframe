"""Cluster 27 regression: the CatBoost val-Pool cache key sees every numeric value (audit 2026-11-20, finding 27-F11).

The reader matched ``compute_signature(X)[:4]`` = (columns, rows, cols, dtypes), which stops before the content hash, and the sampled hash
alone misses a change between sampled rows. Now both sides use ``exact_numeric=True`` and compare through the content hash.
"""

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training import _predict_guards as pg
from mlframe.training._dataset_cache_fingerprint import CONTENT_KEY_LEN, compute_signature


def _frame(n=5000):
    """A numeric frame plus a string column, larger than the sampled-row cap."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 9, n), "s": rng.choice(["x", "y"], n)})


def test_single_unsampled_numeric_change_changes_the_exact_key():
    """Editing one value at a row the sample skips leaves the sampled key equal and changes the exact key."""
    df = _frame()
    sampled_rows = {int(i) for i in np.linspace(0, len(df) - 1, 64)}
    row = next(i for i in range(1, len(df)) if i not in sampled_rows)
    other = df.copy()
    other.loc[row, "a"] += 1.0
    assert compute_signature(df) == compute_signature(other), "the sampled key cannot see this edit"
    assert compute_signature(df, exact_numeric=True) != compute_signature(other, exact_numeric=True)


def test_identical_content_keeps_the_exact_key_and_works_for_polars():
    """A copy keeps the key; a polars frame builds an exact key too."""
    df = _frame()
    assert compute_signature(df, exact_numeric=True) == compute_signature(df.copy(), exact_numeric=True)
    pl_df = pl.from_pandas(df)
    assert compute_signature(pl_df, exact_numeric=True) == compute_signature(pl_df.clone(), exact_numeric=True)


def test_reader_does_not_return_a_pool_cached_for_other_content():
    """The predict-side lookup misses when only an unsampled numeric value differs from the cached val frame."""
    df = _frame()
    pool = object()
    pg._CB_VAL_POOL_CACHE.clear()
    pg._CB_VAL_POOL_CACHE[compute_signature(df, extra=((), (), ()), exact_numeric=True)] = pool
    try:
        assert pg._cb_val_pool_cache_lookup(df.copy(), "predict") is pool
        sampled_rows = {int(i) for i in np.linspace(0, len(df) - 1, 64)}
        other = df.copy()
        other.loc[next(i for i in range(1, len(df)) if i not in sampled_rows), "b"] += 1
        assert pg._cb_val_pool_cache_lookup(other, "predict") is None
    finally:
        pg._CB_VAL_POOL_CACHE.clear()


def test_content_key_prefix_includes_the_content_hash():
    """CONTENT_KEY_LEN covers columns, shape, dtypes and the content hash."""
    assert CONTENT_KEY_LEN == 5
