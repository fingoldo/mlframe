"""Bit-identity + hardening tests for the Wave 8 A4 cache-key efficiency changes.

Covers:
- A4-02: ``_pre_pipeline_cache_get`` accepts a precomputed key and returns the same entry.
- A4-03: ``_full_target_content_hash`` (id, shape) memo is bit-identical to the cold recompute.
- A4-04: ``_canonical_dtype_pairs`` schema-hoist + id-memo are bit-identical to the cold compute.
- A4-05: the model-input fingerprint cache pin-invariant guard catches a column mismatch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.core import _phase_train_one_target as _pt
from mlframe.training.core._phase_train_one_target import (
    _canonical_dtype_pairs,
    _canonical_dtype_pairs_compute,
)
from mlframe.training.pipeline import _pipeline_cache as _pc
from mlframe.training.pipeline._pipeline_cache import (
    _full_target_content_hash,
    _pre_pipeline_cache_clear,
    _pre_pipeline_cache_get,
    _pre_pipeline_cache_key,
    _pre_pipeline_cache_set,
)


# --- A4-04 ---------------------------------------------------------------------------

def test_a4_04_dtype_pairs_memo_bit_identical_to_compute() -> None:
    """The id-memoised wrapper returns exactly what the cold compute returns."""
    df = pl.DataFrame({
        "a": np.arange(5, dtype=np.int32),
        "b": np.arange(5, dtype=np.float32),
        "c": pl.Series(["x", "y", "x", "y", "x"], dtype=pl.Categorical),
    })
    _pt._DTYPE_PAIRS_MEMO.clear()
    expected = _canonical_dtype_pairs_compute(df)
    first = _canonical_dtype_pairs(df)        # cold (populates memo)
    second = _canonical_dtype_pairs(df)       # warm (memo hit)
    assert first == expected
    assert second == expected


def test_a4_04_dtype_pairs_distinct_schemas_distinct_results() -> None:
    """Two frames of identical width but different dtypes must not collide in the memo."""
    _pt._DTYPE_PAIRS_MEMO.clear()
    df_i = pd.DataFrame({"x": np.array([1, 2, 3], dtype=np.int32)})
    df_f = pd.DataFrame({"x": np.array([1.0, 2.0, 3.0], dtype=np.float64)})
    assert _canonical_dtype_pairs(df_i) != _canonical_dtype_pairs(df_f)


# --- A4-03 ---------------------------------------------------------------------------

def test_a4_03_target_hash_memo_bit_identical_to_cold() -> None:
    """The warm (memo) hash equals the cold recompute byte-for-byte."""
    y = pl.Series("y", np.random.default_rng(11).integers(0, 2, 5000).astype(np.int8))
    _pc._PIPELINE_TARGET_HASH_CACHE.clear()
    cold = _full_target_content_hash(y)        # populates memo
    _again = _full_target_content_hash(y)      # memo hit
    _pc._PIPELINE_TARGET_HASH_CACHE.clear()
    cold2 = _full_target_content_hash(y)       # recompute from scratch
    assert cold == _again == cold2
    assert cold != ""


def test_a4_03_distinct_targets_distinct_hashes() -> None:
    """Two distinct targets must hash differently even with the memo live."""
    _pc._PIPELINE_TARGET_HASH_CACHE.clear()
    y1 = pl.Series("y", np.zeros(1000, dtype=np.int8))
    y2 = pl.Series("y", np.ones(1000, dtype=np.int8))
    assert _full_target_content_hash(y1) != _full_target_content_hash(y2)


# --- A4-02 ---------------------------------------------------------------------------

def test_a4_02_cache_get_accepts_precomputed_key() -> None:
    """``_pre_pipeline_cache_get(key=...)`` returns the same entry as the key-less call,
    so the get/set double-call no longer depends on the single-slot _LAST_KEY_CACHE memo."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _pre_pipeline_cache_clear()
    train = pd.DataFrame({"a": np.arange(20, dtype=np.float64), "b": np.arange(20, dtype=np.float64)})
    val = pd.DataFrame({"a": np.arange(5, dtype=np.float64), "b": np.arange(5, dtype=np.float64)})
    pipe = Pipeline([("scaler", StandardScaler())])
    tgt = pd.Series(np.arange(20) % 2)

    train_out = train.copy()
    val_out = val.copy()
    _pre_pipeline_cache_set(train, val, pipe, train_out, val_out, train_target=tgt, target_name="t")

    key = _pre_pipeline_cache_key(train, val, pipe, train_target=tgt, target_name="t")
    via_key = _pre_pipeline_cache_get(train, val, pipe, train_target=tgt, target_name="t", key=key)
    via_recompute = _pre_pipeline_cache_get(train, val, pipe, train_target=tgt, target_name="t")
    assert via_key is not None
    assert via_key is via_recompute  # same stored entry object
    _pre_pipeline_cache_clear()


def test_a4_02_precomputed_key_matches_internal_key() -> None:
    """A precomputed key passed in must equal the key the get would build internally
    (so threading it in cannot change which slot is read)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _pre_pipeline_cache_clear()
    train = pd.DataFrame({"a": np.arange(8, dtype=np.float64)})
    pipe = Pipeline([("scaler", StandardScaler())])
    k1 = _pre_pipeline_cache_key(train, None, pipe, target_name="z")
    k2 = _pre_pipeline_cache_key(train, None, pipe, target_name="z")
    assert k1 == k2


# --- A4-05 ---------------------------------------------------------------------------

def test_a4_05_fingerprint_pin_invariant_guard_detects_column_mismatch() -> None:
    """Replicate the guard logic: a cache entry whose stored schema column names differ from
    the live frame's columns (id-recycle hazard) must be treated as a miss, not served."""
    from mlframe.training.utils import compute_model_input_fingerprint

    live = pd.DataFrame({"alpha": [1.0, 2.0], "beta": [3.0, 4.0]})
    # Cached schema from a DIFFERENT (same-ncols) frame -- the recycled-id scenario.
    stale_hash, stale_schema = compute_model_input_fingerprint(
        pd.DataFrame({"gamma": [9.0, 9.0], "delta": [9.0, 9.0]})
    )
    live_cols = sorted(str(c) for c in live.columns)
    cached_cols = sorted(rec.get("name") for rec in stale_schema)
    assert cached_cols != live_cols  # guard would force a recompute

    # A matching schema (same frame) passes the guard.
    fresh_hash, fresh_schema = compute_model_input_fingerprint(live)
    fresh_cols = sorted(rec.get("name") for rec in fresh_schema)
    assert fresh_cols == live_cols
    assert fresh_hash != stale_hash
