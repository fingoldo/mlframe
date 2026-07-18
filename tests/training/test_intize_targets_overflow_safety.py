"""Sensor tests for intize_targets overflow promotion.

Pre-fix bug (2026-05-20 silent-coercion audit): targets[name].astype(np.int8) silently
wrapped multiclass labels >127 on pandas (e.g. label 200 -> -56). Polars Series.cast(pl.Int8)
raised InvalidOperationError -- asymmetric trap.

Post-fix: intize_targets promotes int8 -> int16 -> int32 -> int64 based on the actual value
range. All paths share one promotion table via _safe_int_cast_numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.extractors import (
    intize_targets,
    _safe_int_cast_numpy,
    _smallest_safe_int_dtype,
)


# ----- _smallest_safe_int_dtype --------------------------------------------------


@pytest.mark.parametrize(
    "min_v,max_v,expected",
    [
        (0, 1, np.int8),  # binary
        (0, 127, np.int8),  # int8 upper bound
        (-128, 127, np.int8),  # full int8 range
        (0, 128, np.int16),  # int8 overflow by 1
        (-129, 0, np.int16),  # int8 underflow by 1
        (0, 32_767, np.int16),  # int16 upper bound
        (0, 32_768, np.int32),  # int16 overflow by 1
        (0, 2_147_483_647, np.int32),  # int32 upper bound
        (0, 2_147_483_648, np.int64),  # int32 overflow by 1
    ],
)
def test_smallest_safe_int_dtype_promotion_table(min_v, max_v, expected):
    """Smallest safe int dtype promotion table."""
    assert _smallest_safe_int_dtype(min_v, max_v) == np.dtype(expected)


# ----- _safe_int_cast_numpy -----------------------------------------------------


def test_safe_cast_binary_labels_stays_int8():
    """Safe cast binary labels stays int8."""
    arr = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    out = _safe_int_cast_numpy(arr, "y")
    assert out.dtype == np.int8


def test_safe_cast_multiclass_200_labels_promotes_to_int16():
    """Regression: label 200 used to silently wrap to -56 under astype(int8)."""
    arr = np.arange(200, dtype=np.int64)
    out = _safe_int_cast_numpy(arr, "y")
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, arr)  # no value lost


def test_safe_cast_multiclass_50k_labels_promotes_to_int32():
    """50_000 classes (high-card label-encoded target)."""
    arr = np.arange(50_000, dtype=np.int64)
    out = _safe_int_cast_numpy(arr, "y")
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, arr)


def test_safe_cast_negative_values_promotes():
    """Negative class IDs (rare but legal) should also promote."""
    arr = np.array([-200, -1, 0, 200], dtype=np.int64)
    out = _safe_int_cast_numpy(arr, "y")
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, arr)


def test_safe_cast_integer_valued_float_allowed():
    """float input with integer-valued contents should still convert."""
    arr = np.array([0.0, 1.0, 2.0, 200.0], dtype=np.float64)
    out = _safe_int_cast_numpy(arr, "y")
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, [0, 1, 2, 200])


def test_safe_cast_fractional_float_raises():
    """Safe cast fractional float raises."""
    arr = np.array([0.5, 1.5], dtype=np.float64)
    with pytest.raises(ValueError, match="non-integer"):
        _safe_int_cast_numpy(arr, "y")


def test_safe_cast_nan_raises():
    """Safe cast nan raises."""
    arr = np.array([0.0, np.nan, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="non-integer or non-finite"):
        _safe_int_cast_numpy(arr, "y")


def test_safe_cast_bool_array_to_int8():
    """Safe cast bool array to int8."""
    arr = np.array([True, False, True], dtype=np.bool_)
    out = _safe_int_cast_numpy(arr, "y")
    assert out.dtype == np.int8


# ----- intize_targets (the public API) ------------------------------------------


def test_intize_pandas_binary_no_overflow():
    """Intize pandas binary no overflow."""
    targets = {"y": pd.Series([0, 1, 1, 0], dtype=np.int64)}
    intize_targets(targets)
    assert targets["y"].dtype == np.int8


def test_intize_pandas_multiclass_200_no_silent_wrap():
    """Regression: this exact path used to silently wrap label 200 -> -56."""
    targets = {"y": pd.Series(np.arange(200), dtype=np.int64)}
    intize_targets(targets)
    out = targets["y"]
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, np.arange(200))


def test_intize_polars_multiclass_200_no_invalid_op():
    """Regression: pl.Series.cast(pl.Int8) used to raise InvalidOperationError on 200 classes."""
    targets = {"y": pl.Series([i for i in range(200)], dtype=pl.Int64)}
    intize_targets(targets)
    out = targets["y"]
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, np.arange(200))


def test_intize_numpy_already_int8_passes_through():
    """Intize numpy already int8 passes through."""
    arr = np.array([0, 1, 2], dtype=np.int8)
    targets = {"y": arr}
    intize_targets(targets)
    assert targets["y"].dtype == np.int8


def test_intize_mixed_pandas_polars_numpy_in_one_dict():
    """All three input types share the same promotion table."""
    targets = {
        "pd_y": pd.Series(np.arange(150), dtype=np.int64),  # > int8 -> int16
        "pl_y": pl.Series(list(range(150)), dtype=pl.Int64),
        "np_y": np.arange(150, dtype=np.int64),
    }
    intize_targets(targets)
    assert targets["pd_y"].dtype == np.int16
    assert targets["pl_y"].dtype == np.int16
    assert targets["np_y"].dtype == np.int16
    for k in ("pd_y", "pl_y", "np_y"):
        np.testing.assert_array_equal(targets[k], np.arange(150))


def test_intize_unsupported_type_raises():
    """Intize unsupported type raises."""
    targets = {"y": [0, 1, 1]}  # list, not array/series
    with pytest.raises(TypeError, match="Unsupported target type"):
        intize_targets(targets)
