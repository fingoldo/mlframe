"""Tests for mlframe.preprocessing.prepare_df_for_catboost.

Focus: dtype preservation when converting nullable/extension columns to
CatBoost-friendly numpy floats. Historical bug: everything was widened to
float64 via bare `astype(float)` (pandas) or `cast(pl.Float64)` (polars),
which silently cost memory/GPU bandwidth on users who had deliberately
chosen narrow precision.

Rules we enforce:
- Non-nullable numpy floats pass through unchanged.
- `pd.Float32Dtype`/`pd.Float64Dtype` → preserve precision (32→32, 64→64).
- `pd.Int8..Int32` / `pd.UInt8..UInt32` / `pd.BooleanDtype` → float32
  (values fit exactly, saves memory).
- `pd.Int64Dtype` / `pd.UInt64Dtype` → float64 (>~2**24 loses precision).
- Polars Float32/Float64 — untouched.
- Polars small ints with nulls → Float32; only Int64/UInt64 with nulls → Float64.
- Polars int columns WITHOUT nulls are not cast at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.preprocessing import prepare_df_for_catboost


# ---------------------------------------------------------------------------
# pandas dtype preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col_dtype, expected_out", [
    (np.float32,                   np.float32),
    (np.float64,                   np.float64),
    (pd.Float32Dtype(),            np.dtype("float32")),
    (pd.Float64Dtype(),            np.dtype("float64")),
    (pd.Int8Dtype(),               np.dtype("float32")),
    (pd.Int16Dtype(),              np.dtype("float32")),
    (pd.Int32Dtype(),              np.dtype("float32")),
    (pd.UInt8Dtype(),              np.dtype("float32")),
    (pd.UInt16Dtype(),             np.dtype("float32")),
    (pd.UInt32Dtype(),             np.dtype("float32")),
    (pd.Int64Dtype(),              np.dtype("float64")),
    (pd.UInt64Dtype(),             np.dtype("float64")),
    (pd.BooleanDtype(),            np.dtype("float32")),
])
def test_pandas_dtype_preserved_or_narrowed(col_dtype, expected_out):
    if isinstance(col_dtype, type) and issubclass(col_dtype, np.floating):
        # Non-nullable numpy floats — should pass through untouched.
        arr = np.array([1.0, 2.0, 3.0], dtype=col_dtype)
    elif isinstance(col_dtype, pd.BooleanDtype):
        arr = pd.array([True, False, None], dtype=col_dtype)
    else:
        arr = pd.array([1, 2, None], dtype=col_dtype)
    df = pd.DataFrame({"c": arr})
    out = prepare_df_for_catboost(df.copy(), cat_features=[])
    assert out.dtypes["c"] == expected_out, f"{col_dtype} → {out.dtypes['c']} (expected {expected_out})"


def test_pandas_float32_non_nullable_is_noop():
    """Sanity: bare numpy float32 must not be touched even if a bug regressed
    the extension-dtype branch.
    """
    df = pd.DataFrame({"f": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
    out = prepare_df_for_catboost(df.copy(), cat_features=[])
    assert out.dtypes["f"] == np.float32


def test_pandas_nullable_float32_fills_na_but_keeps_precision():
    """End-to-end: pd.Float32Dtype column with a null must survive as
    numpy float32 with np.nan in place of the null.
    """
    df = pd.DataFrame({"f": pd.array([1.0, 2.0, None], dtype=pd.Float32Dtype())})
    out = prepare_df_for_catboost(df.copy(), cat_features=[])
    assert out.dtypes["f"] == np.float32
    # Null became np.nan (not pd.NA — CatBoost cannot handle the latter).
    assert np.isnan(out["f"].iloc[-1])


# ---------------------------------------------------------------------------
# polars dtype preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src_dtype, values, expected_out", [
    (pl.Float32,   [1.0, 2.0, None],       pl.Float32),
    (pl.Float64,   [1.0, 2.0, None],       pl.Float64),
    (pl.Int8,      [1, 2, None],           pl.Float32),
    (pl.Int16,     [1, 2, None],           pl.Float32),
    (pl.Int32,     [1, 2, None],           pl.Float32),
    (pl.UInt8,     [1, 2, None],           pl.Float32),
    (pl.UInt16,    [1, 2, None],           pl.Float32),
    (pl.UInt32,    [1, 2, None],           pl.Float32),
    (pl.Int64,     [1, 2, None],           pl.Float64),
    (pl.UInt64,    [1, 2, None],           pl.Float64),
    (pl.Boolean,   [True, False, None],    pl.Float32),
    # No-null columns: left alone entirely (no unnecessary cast).
    (pl.Int32,     [1, 2, 3],              pl.Int32),
    (pl.Int64,     [1, 2, 3],              pl.Int64),
    (pl.Float32,   [1.0, 2.0, 3.0],        pl.Float32),
])
def test_polars_dtype_preserved_or_narrowed(src_dtype, values, expected_out):
    df = pl.DataFrame({"c": pl.Series("c", values, dtype=src_dtype)})
    out = prepare_df_for_catboost(df.clone(), cat_features=[])
    assert out.dtypes[0] == expected_out, f"{src_dtype} → {out.dtypes[0]} (expected {expected_out})"


def test_polars_no_nulls_skips_cast_entirely():
    """Micro-optimisation guard: when a non-float column has no nulls, the
    function shouldn't waste a cast. This test also catches an accidental
    regression that would cast every int column to Float*.
    """
    df = pl.DataFrame({
        "i32": pl.Series("i32", [1, 2, 3], dtype=pl.Int32),
        "i64": pl.Series("i64", [1, 2, 3], dtype=pl.Int64),
        "u16": pl.Series("u16", [1, 2, 3], dtype=pl.UInt16),
    })
    out = prepare_df_for_catboost(df.clone(), cat_features=[])
    assert out.dtypes == [pl.Int32, pl.Int64, pl.UInt16]
