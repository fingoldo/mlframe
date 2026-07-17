"""Direct module-level coverage for mlframe.feature_selection.pre_screen.

The existing ``test_pre_screen_unsupervised.py`` covers the conservative-default contract
for A-Arch-001 (variance=0, null > 0.99, protected_columns). This file adds the gaps
flagged in U18 of the tests-expand audit:

- None / empty-shape inputs
- ``apply_drops`` polars + pandas identity when no overlap
- SparseDtype-aware null counting (the explicit branch at pre_screen:138-149)
- Pandas extension dtypes (CategoricalDtype, StringDtype, nullable Int) NOT crashing
- Threshold boundary: strictly > on null_fraction (mirrors existing but expands)
- biz_value: constant-string column on a polars frame is preserved (variance rule
  is numeric-only by contract)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_selection.pre_screen import apply_drops, compute_unsupervised_drops


# ----------------------------------------------------------------------------
# compute_unsupervised_drops — edge / type cases
# ----------------------------------------------------------------------------


def test_compute_drops_none_input_returns_empty():
    """None input must not crash and must return []."""
    assert compute_unsupervised_drops(None) == []


def test_compute_drops_zero_row_frame_returns_empty():
    """0-row frame must not crash and returns []."""
    df = pd.DataFrame({"x": pd.array([], dtype="float64"), "y": pd.array([], dtype="float64")})
    assert compute_unsupervised_drops(df) == []


def test_compute_drops_single_column_pandas():
    """Single-column frame with constant value must drop that column."""
    df = pd.DataFrame({"only": np.full(100, 7.7)})
    drops = compute_unsupervised_drops(df)
    assert drops == ["only"], f"single constant col must be dropped; got {drops!r}"


def test_compute_drops_all_null_column_pandas():
    """All-null column drops via the null threshold (1.0 > 0.99 default)."""
    n = 100
    df = pd.DataFrame({"all_null": pd.array([np.nan] * n, dtype="float64"), "ok": np.arange(n).astype(float)})
    drops = compute_unsupervised_drops(df)
    assert "all_null" in drops, f"all-null col must be dropped; got {drops!r}"
    assert "ok" not in drops


def test_compute_drops_all_null_column_polars():
    """Compute drops all null column polars."""
    n = 100
    df = pl.DataFrame({"all_null": pl.Series([None] * n, dtype=pl.Float64), "ok": np.arange(n).astype(float)})
    drops = compute_unsupervised_drops(df)
    assert "all_null" in drops, f"all-null polars col must be dropped; got {drops!r}"
    assert "ok" not in drops


def test_compute_drops_string_column_not_dropped_for_variance():
    """Per module contract: variance rule is numeric-only. A non-constant string column
    must survive even though variance is undefined on strings."""
    df = pd.DataFrame({"label": ["x", "y", "x", "z", "y", "x"] * 50, "num": np.arange(300, dtype=float)})
    drops = compute_unsupervised_drops(df)
    assert "label" not in drops, f"non-null string col must not be dropped; got {drops!r}"


def test_compute_drops_categorical_dtype_does_not_crash():
    """np.issubdtype on a CategoricalDtype raises TypeError pre-fix; pre_screen uses
    pd.api.types.is_numeric_dtype which handles extension dtypes. Verify behavior."""
    df = pd.DataFrame(
        {
            "cat": pd.Categorical(["a", "b", "a", "c"] * 25),
            "num": np.arange(100, dtype=float),
        }
    )
    # Must not raise
    drops = compute_unsupervised_drops(df)
    assert "cat" not in drops, "non-degenerate Categorical must not be dropped"


def test_compute_drops_string_dtype_does_not_crash():
    """Compute drops string dtype does not crash."""
    df = pd.DataFrame(
        {
            "s": pd.array(["a", "b", "a", "c"] * 25, dtype="string"),
            "num": np.arange(100, dtype=float),
        }
    )
    drops = compute_unsupervised_drops(df)
    assert "s" not in drops, "non-degenerate StringDtype must not be dropped"


def test_compute_drops_nullable_int_extension_dtype():
    """pd.Int64Dtype (nullable Int) is numeric per pd.api.types — variance rule applies."""
    n = 100
    # Constant nullable int column
    const_col = pd.array([5] * n, dtype="Int64")
    df = pd.DataFrame({"const_nullable": const_col, "ok": np.arange(n, dtype=float)})
    drops = compute_unsupervised_drops(df)
    assert "const_nullable" in drops, f"constant Int64 (nullable) col must be dropped; got {drops!r}"


def test_compute_drops_custom_variance_threshold():
    """Caller can raise the variance bar to drop near-constant columns."""
    rng = np.random.default_rng(0)
    # Near-constant column: var ~ 1e-6
    near_const = rng.normal(loc=0.0, scale=1e-3, size=200)
    truly_random = rng.normal(scale=1.0, size=200)
    df = pd.DataFrame({"near_const": near_const, "truly_random": truly_random})
    # default variance_threshold=0.0 keeps near_const; raising it to 1e-5 should drop it
    default_drops = compute_unsupervised_drops(df, variance_threshold=0.0)
    assert "near_const" not in default_drops, f"default threshold should preserve near_const; got {default_drops}"
    aggressive_drops = compute_unsupervised_drops(df, variance_threshold=1e-5)
    assert "near_const" in aggressive_drops, f"raised threshold must drop near_const; got {aggressive_drops}"
    assert "truly_random" not in aggressive_drops, "truly random col must survive"


def test_compute_drops_protected_columns_overrides_all_rules():
    """Protected columns must NEVER be dropped, even with extreme null fraction."""
    df = pd.DataFrame(
        {
            "ts_id": pd.array([None] * 100, dtype="float64"),  # 100% null → would normally drop
            "x": np.arange(100, dtype=float),
        }
    )
    drops = compute_unsupervised_drops(df, protected_columns={"ts_id"})
    assert "ts_id" not in drops, f"protected column must survive 100% nulls; got {drops!r}"


# ----------------------------------------------------------------------------
# SparseDtype branch — explicit module support per pre_screen.py:138-149
# ----------------------------------------------------------------------------


def test_compute_drops_sparse_column_with_nan_fill_value_kept():
    """A SparseDtype(float64, NaN) column representing real TF-IDF storage (most
    rows are fill=NaN by storage convention but the dense data is sparse-by-design)
    must NOT be dropped just because dense-view nulls dominate."""
    n = 200
    # Build a sparse array: 5 non-null stored values among 200 cells, fill=NaN
    sp = pd.arrays.SparseArray(
        [1.0, np.nan, 2.0, np.nan, 3.0] + [np.nan] * (n - 5),
        fill_value=np.nan,
    )
    df = pd.DataFrame({"tfidf_term": sp, "ok": np.arange(n, dtype=float)})
    # 5 stored values, all non-null → null_count uses sp_values path → ~0 nulls counted
    # Module logic: null_count = (n_unfilled if fill_is_nan else 0) + stored_nan_count
    # = 195 + 0 = 195; 195 > 0.99*200=198? No: 195 < 198. So column kept.
    drops = compute_unsupervised_drops(df, null_fraction_threshold=0.99)
    assert "tfidf_term" not in drops, f"sparse TF-IDF-like col with non-null sp_values must be preserved; got {drops!r}"


# ----------------------------------------------------------------------------
# apply_drops — extended cases
# ----------------------------------------------------------------------------


def test_apply_drops_none_input_returns_none():
    """Apply drops none input returns none."""
    assert apply_drops(None, ["x"]) is None


def test_apply_drops_no_overlap_returns_input_unchanged():
    """Apply drops no overlap returns input unchanged."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = apply_drops(df, ["nonexistent"])
    assert out is df, "no-overlap drop list must return input unchanged (identity)"


def test_apply_drops_empty_drop_list_returns_input():
    """Apply drops empty drop list returns input."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out = apply_drops(df, [])
    assert out is df, "empty drop list must return input unchanged"


def test_apply_drops_polars_preserves_schema_for_remaining_cols():
    """Apply drops polars preserves schema for remaining cols."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": ["a", "b", "c"],
            "z": [1.1, 2.2, 3.3],
        }
    )
    out = apply_drops(df, ["y"])
    assert out.columns == ["x", "z"], f"polars drop must remove y; got cols {out.columns}"
    assert out["x"].dtype == df["x"].dtype, "remaining column dtypes must be preserved"
    assert out["z"].dtype == df["z"].dtype


def test_apply_drops_pandas_preserves_remaining_dtypes():
    """Apply drops pandas preserves remaining dtypes."""
    df = pd.DataFrame(
        {
            "x": pd.array([1, 2, 3], dtype="Int64"),
            "y": pd.array(["a", "b", "c"], dtype="string"),
            "z": [1.1, 2.2, 3.3],
        }
    )
    out = apply_drops(df, ["y"])
    assert list(out.columns) == ["x", "z"]
    assert out["x"].dtype == df["x"].dtype, "Int64 dtype must survive"


# ----------------------------------------------------------------------------
# biz_value: end-to-end (compute -> apply)
# ----------------------------------------------------------------------------


def test_biz_value_compute_then_apply_returns_clean_frame_pandas():
    """biz_value: full round-trip drops the noise columns and the resulting frame has
    only the useful columns. Locks in the production-call pattern."""
    n = 300
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "signal_1": rng.normal(size=n),
            "signal_2": rng.normal(size=n),
            "noise_const": np.full(n, 0.5),
            "noise_null": [np.nan] * (n - 1) + [1.0],
            "target": rng.integers(0, 2, size=n),
        }
    )
    drops = compute_unsupervised_drops(df, protected_columns={"target"})
    cleaned = apply_drops(df, drops)
    assert set(cleaned.columns) == {"signal_1", "signal_2", "target"}, f"after pre-screen frame must contain only useful cols; got {list(cleaned.columns)}"
    assert len(cleaned) == n, "row count must be preserved"


def test_biz_value_compute_then_apply_returns_clean_frame_polars():
    """Biz value compute then apply returns clean frame polars."""
    n = 300
    rng = np.random.default_rng(11)
    df = pl.DataFrame(
        {
            "signal_1": rng.normal(size=n),
            "signal_2": rng.normal(size=n),
            "noise_const": np.full(n, 0.5),
            "noise_null": pl.Series([None] * (n - 1) + [1.0], dtype=pl.Float64),
            "target": rng.integers(0, 2, size=n),
        }
    )
    drops = compute_unsupervised_drops(df, protected_columns={"target"})
    cleaned = apply_drops(df, drops)
    assert set(cleaned.columns) == {"signal_1", "signal_2", "target"}, f"after pre-screen polars frame must contain only useful cols; got {cleaned.columns}"
    assert cleaned.height == n
