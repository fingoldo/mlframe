"""Sensor: pandas nullable Float dtype (pd.Float32Dtype / Float64Dtype) inf values
must be scrubbed by _process_special_values and detected by _frame_contains_inf,
matching the polars cs.float() behavior.

Pre-fix shape: both functions used ``select_dtypes(include="floating")`` which
silently SKIPPED pandas nullable Float ExtensionDtype columns (those are
ExtensionDtype, not legacy np.floating). Inf values in pd.Float64 columns slipped
past the diagnostic + the scrub + the loud-fail check, and crashed XGB / HGB
downstream in C++ with no log line pointing back.

Post-fix: _pandas_float_like_columns helper covers both legacy numpy and pandas
nullable extension Float dtypes, mirroring polars cs.float().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.preprocessing import (
    _frame_contains_inf,
    _pandas_float_like_columns,
    _process_special_values,
)


def test_helper_covers_both_legacy_and_extension_float():
    df = pd.DataFrame(
        {
            "legacy32": pd.array([1.0, 2.0], dtype="float32"),
            "legacy64": pd.array([1.0, 2.0], dtype="float64"),
            "nullable32": pd.array([1.0, None], dtype=pd.Float32Dtype()),
            "nullable64": pd.array([1.0, None], dtype=pd.Float64Dtype()),
            "int_col": pd.array([1, 2], dtype="int64"),
            "str_col": ["a", "b"],
        }
    )
    cols = _pandas_float_like_columns(df)
    assert set(cols) == {"legacy32", "legacy64", "nullable32", "nullable64"}, f"expected all 4 float-like columns, got {cols}"


def test_legacy_float64_inf_scrubbed():
    """Pre-fix baseline: legacy np.float64 inf was scrubbed (this branch was always OK)."""
    df = pd.DataFrame({"x": np.array([1.0, np.inf, -np.inf, 2.0])})
    out = _process_special_values(df, verbose=False)
    arr = out["x"].to_numpy()
    assert np.isnan(arr[1]) and np.isnan(arr[2]), f"legacy float inf not scrubbed: {arr}"


def test_nullable_float64_inf_scrubbed_post_fix():
    """REGRESSION: pre-fix this silently kept inf in pd.Float64 columns.
    Post-fix the helper covers extension dtypes too -> inf -> nan."""
    df = pd.DataFrame(
        {
            "x": pd.array([1.0, float("inf"), float("-inf"), 2.0], dtype=pd.Float64Dtype()),
        }
    )
    out = _process_special_values(df, verbose=False)
    # Pull out values; nullable Float NaN reads as pd.NA OR np.nan depending on flow.
    vals = out["x"].to_numpy(dtype=np.float64, na_value=np.nan)
    assert np.isnan(vals[1]) and np.isnan(vals[2]), f"nullable Float64 inf not scrubbed (pre-fix bug): values = {vals}"


def test_nullable_float32_inf_also_scrubbed():
    df = pd.DataFrame(
        {
            "x": pd.array([1.0, float("inf"), 2.0], dtype=pd.Float32Dtype()),
        }
    )
    out = _process_special_values(df, verbose=False)
    vals = out["x"].to_numpy(dtype=np.float64, na_value=np.nan)
    assert np.isnan(vals[1])


def test_frame_contains_inf_nullable_float_detected():
    """REGRESSION: pre-fix _frame_contains_inf returned False for inf-in-nullable-Float
    frames, defeating the fix_infinities=False loud-fail safeguard."""
    df = pd.DataFrame(
        {
            "x": pd.array([1.0, float("inf")], dtype=pd.Float64Dtype()),
        }
    )
    assert _frame_contains_inf(df) is True, "pre-fix bug: nullable Float inf not detected by _frame_contains_inf"


def test_frame_contains_inf_clean_nullable_float_returns_false():
    df = pd.DataFrame(
        {
            "x": pd.array([1.0, 2.0, None], dtype=pd.Float64Dtype()),
        }
    )
    assert _frame_contains_inf(df) is False, "no inf present; should be False"


def test_frame_contains_inf_int_only_returns_false():
    """Int-only frames have no float-like columns; should be False even if integers max out."""
    df = pd.DataFrame({"x": np.arange(100, dtype=np.int64)})
    assert _frame_contains_inf(df) is False
