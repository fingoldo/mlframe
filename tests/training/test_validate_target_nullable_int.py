"""Sensor: _validate_target_values must detect NaN / pd.NA / polars-null on
nullable-integer targets, not silently skip the check.

Pre-fix shape (agent finding #5 of pandas/polars asymmetry audit):

- pandas nullable Int64 target with pd.NA values: ``target.values`` returned a
  pandas ExtensionArray, ``np.isfinite(ext_arr)`` raised TypeError, caught at the
  outer except, nan_count = inf_count = 0 -- "no NaN detected" silently -> training
  proceeds on <NA> rows -> CatBoost crashes opaquely in C++.

- polars Int64 with nulls: same TypeError path on the conversion, same silent skip.

Both sides defeated the function's stated purpose ("Check target for NaN /
infinity values before training"). Post-fix the helper coerces via to_numpy
with na_value=nan for pandas and cast(Float64) for polars-Int-with-null, so
np.isfinite sees a real numeric NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training._data_helpers import _validate_target_values


# ---- pandas paths ---------------------------------------------------------


def test_pandas_float_nan_detected():
    """Baseline: legacy float NaN was always detected."""
    t = pd.Series([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="NaN"):
        _validate_target_values(t, subset_name="train")


def test_pandas_nullable_int64_with_na_detected_post_fix():
    """REGRESSION: pre-fix this slipped through silently."""
    t = pd.Series([1, 2, pd.NA, 4], dtype="Int64")
    with pytest.raises(ValueError, match="NaN"):
        _validate_target_values(t, subset_name="train")


def test_pandas_nullable_int32_with_na_detected_post_fix():
    t = pd.Series([1, pd.NA, 3], dtype="Int32")
    with pytest.raises(ValueError, match="NaN"):
        _validate_target_values(t, subset_name="train")


def test_pandas_nullable_float64_with_na_detected():
    t = pd.Series([1.0, pd.NA, 2.0], dtype=pd.Float64Dtype())
    with pytest.raises(ValueError, match="NaN"):
        _validate_target_values(t, subset_name="train")


def test_pandas_nullable_int_without_na_passes():
    """No NA -> no error. Pre-fix this also worked, just by accident (no NA to detect)."""
    t = pd.Series([1, 2, 3, 4], dtype="Int64")
    # Should not raise
    _validate_target_values(t, subset_name="train", is_classification=True)


def test_pandas_float_with_inf_detected():
    t = pd.Series([1.0, np.inf, 2.0])
    with pytest.raises(ValueError, match="infinity"):
        _validate_target_values(t, subset_name="train")


# ---- polars paths ---------------------------------------------------------


def test_polars_int64_with_null_detected_post_fix():
    """REGRESSION: pre-fix polars Int64 nulls also slipped through (different code path,
    same defeat-the-check shape)."""
    t = pl.Series([1, 2, None, 4], dtype=pl.Int64)
    with pytest.raises(ValueError, match="NaN"):
        _validate_target_values(t, subset_name="train")


def test_polars_int32_with_null_detected():
    t = pl.Series([1, None, 3], dtype=pl.Int32)
    with pytest.raises(ValueError, match="NaN"):
        _validate_target_values(t, subset_name="train")


def test_polars_float_with_nan_detected():
    """Baseline: polars Float NaN was already detected (the to_numpy() conversion
    surfaces it as np.nan natively)."""
    t = pl.Series([1.0, float("nan"), 2.0], dtype=pl.Float64)
    with pytest.raises(ValueError, match="NaN"):
        _validate_target_values(t, subset_name="train")


def test_polars_int_without_null_passes():
    t = pl.Series([1, 2, 3], dtype=pl.Int64)
    _validate_target_values(t, subset_name="train", is_classification=True)


def test_polars_clean_target_passes():
    t = pl.Series([0, 1, 0, 1], dtype=pl.Int8)
    _validate_target_values(t, subset_name="train", is_classification=True)


# ---- classification single-class detection (post-fix path still works) ----


def test_single_class_target_raises_classification_error():
    t = pd.Series([1, 1, 1, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="one unique value"):
        _validate_target_values(t, subset_name="train", is_classification=True)


def test_polars_single_class_target_raises_classification_error():
    t = pl.Series([1, 1, 1], dtype=pl.Int8)
    with pytest.raises(ValueError, match="one unique value"):
        _validate_target_values(t, subset_name="train", is_classification=True)
