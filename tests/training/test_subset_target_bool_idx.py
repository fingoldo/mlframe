"""Sensor: _subset_target accepts boolean masks for BOTH pandas and polars.

Pre-fix shape (agent finding #3 of pandas/polars asymmetry audit):
- pandas: ``target.values[bool_mask]`` works -- numpy-style bool indexing
  succeeds silently.
- polars: ``target.gather(bool_mask)`` raises ``InvalidOperationError`` ("gather
  expects integer indices, got bool").

Same caller code with the same logical input produced opposite outcomes per
backend. The polars branch errored loudly; the pandas branch silently corrupted
the downstream index (the implicit row-position semantics differed because
pandas .values[bool_mask] respects positional bool but the wrapped pd.Series
index slot used ``target.index[bool_mask]`` which silently picked the WRONG
rows when target.index was a sorted RangeIndex of length != mask length).

Post-fix: bool masks are normalised to integer positions via np.flatnonzero
at the top of the function so both branches take the integer path uniformly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training._data_helpers import _extract_target_subset as _subset_target


def test_pandas_bool_mask_works_and_returns_correct_rows():
    t = pd.Series([10, 20, 30, 40, 50])
    mask = np.array([True, False, True, False, True])
    out = _subset_target(t, mask)
    np.testing.assert_array_equal(out.to_numpy(), [10, 30, 50])


def test_polars_bool_mask_works_post_fix():
    """REGRESSION: pre-fix raised InvalidOperationError."""
    t = pl.Series("y", [10, 20, 30, 40, 50])
    mask = np.array([True, False, True, False, True])
    out = _subset_target(t, mask)
    np.testing.assert_array_equal(out.to_numpy(), [10, 30, 50])


def test_numpy_bool_mask_works():
    t = np.array([10, 20, 30, 40, 50])
    mask = np.array([True, False, True, False, True])
    out = _subset_target(t, mask)
    np.testing.assert_array_equal(out, [10, 30, 50])


def test_pandas_int_idx_unchanged():
    """Sanity: integer indexing still works after the bool normalisation hook."""
    t = pd.Series([10, 20, 30, 40, 50])
    idx = np.array([0, 2, 4])
    out = _subset_target(t, idx)
    np.testing.assert_array_equal(out.to_numpy(), [10, 30, 50])


def test_polars_int_idx_unchanged():
    t = pl.Series("y", [10, 20, 30, 40, 50])
    idx = np.array([0, 2, 4])
    out = _subset_target(t, idx)
    np.testing.assert_array_equal(out.to_numpy(), [10, 30, 50])


def test_numpy_int_idx_unchanged():
    t = np.array([10, 20, 30, 40, 50])
    idx = np.array([0, 2, 4])
    out = _subset_target(t, idx)
    np.testing.assert_array_equal(out, [10, 30, 50])


def test_idx_none_returns_target_unchanged():
    t = pd.Series([1, 2, 3])
    out = _subset_target(t, None)
    assert out is t


def test_pandas_list_bool_mask_normalised():
    """Plain Python list of bool should also work (gets coerced to np.bool_ array)."""
    t = pd.Series([10, 20, 30])
    out = _subset_target(t, [True, False, True])
    np.testing.assert_array_equal(out.to_numpy(), [10, 30])


def test_polars_list_bool_mask_normalised():
    t = pl.Series("y", [10, 20, 30])
    out = _subset_target(t, [True, False, True])
    np.testing.assert_array_equal(out.to_numpy(), [10, 30])
