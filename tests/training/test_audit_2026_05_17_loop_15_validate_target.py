"""Regression: _validate_target_values single-pass isfinite fastpath.

Old: two full-array scans (`isnan().sum()` + `isinf().sum()`).
New: one `isfinite()` pass; only re-scan the non-finite subset when
counting is needed for the error message.

Behaviour MUST be identical across:
- all-finite targets (the all-clear short-circuit)
- targets with only NaN
- targets with only inf
- targets with both
- object-dtype targets (TypeError fall-through)
- pd.Series wrapper unwrap

Profile attribution under cProfile inflated the function's apparent
cost to 67ms/call on n=5000; the actual work is ~25us/call with the
fastpath (1.58x faster than the prior 37us baseline on the same
shape — micro-bench in tests/perf/results/_loop_iter_log.md iter 15).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training._data_helpers import _validate_target_values


def test_all_finite_passes() -> None:
    arr = np.arange(5000, dtype=np.float64)
    _validate_target_values(arr, subset_name="train", is_classification=False)


def test_all_finite_classification_passes() -> None:
    arr = np.random.default_rng(0).integers(0, 2, size=5000).astype(np.float64)
    _validate_target_values(arr, subset_name="train", is_classification=True)


def test_nan_raises_with_count() -> None:
    arr = np.array([1.0, np.nan, 3.0, np.nan, np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="3 NaN"):
        _validate_target_values(arr, subset_name="train", is_classification=False)


def test_inf_raises_with_count() -> None:
    arr = np.array([1.0, np.inf, 3.0, -np.inf], dtype=np.float64)
    with pytest.raises(ValueError, match="2 infinity"):
        _validate_target_values(arr, subset_name="train", is_classification=False)


def test_nan_and_inf_raises_with_both_counts() -> None:
    arr = np.array([np.nan, np.inf, 1.0, -np.inf, np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="2 NaN and 2 infinity"):
        _validate_target_values(arr, subset_name="val", is_classification=False)


def test_pd_series_wrapper_unwraps() -> None:
    s = pd.Series([1.0, 2.0, np.nan])
    with pytest.raises(ValueError, match="1 NaN"):
        _validate_target_values(s, subset_name="test", is_classification=False)


def test_object_dtype_falls_through_to_classification_branch() -> None:
    """Categorical / object targets cannot be isfinite'd; the TypeError
    fall-through must still let the single-class detector run."""
    arr = np.array(["a", "b", "a", "b"], dtype=object)
    _validate_target_values(arr, subset_name="train", is_classification=True)


def test_single_class_classification_raises() -> None:
    arr = np.zeros(100, dtype=np.float64)
    with pytest.raises(ValueError, match="only one unique value"):
        _validate_target_values(arr, subset_name="train", is_classification=True)


def test_empty_target_does_not_crash() -> None:
    arr = np.array([], dtype=np.float64)
    _validate_target_values(arr, subset_name="train", is_classification=False)
