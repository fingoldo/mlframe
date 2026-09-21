"""The multi-target OOF row slice of a polars frame must gather by the index array, and equal the list-based slice.

``_slice_rows_by_idx`` turned each fold's indices into a Python list before indexing a polars frame (about 160k ints per
fold and component slice), and its ``filter`` fallback could never run. Polars gathers from an integer ndarray directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.core._phase_composite_post_xt_ensemble._phase_composite_post_xt_mtr_oof import _slice_rows_by_idx

pl = pytest.importorskip("polars")


def _frame(n: int = 1000):
    """A small polars frame with mixed dtypes."""
    rng = np.random.default_rng(0)
    return pl.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 5, size=n), "c": [f"s{i % 7}" for i in range(n)]})


def test_the_slice_equals_the_list_based_one():
    """Same rows, same order, same dtypes as indexing with a Python list."""
    df = _frame()
    idx = np.sort(np.random.default_rng(1).choice(len(df), size=300, replace=False))
    got = _slice_rows_by_idx(df, idx)
    assert got.equals(df[idx.tolist()])


def test_unsorted_and_repeated_indices_keep_their_order():
    """Fold indices need not be sorted; the gather must follow them exactly."""
    df = _frame()
    idx = np.array([5, 3, 3, 999, 0])
    got = _slice_rows_by_idx(df, idx)
    assert got["a"].to_list() == [df["a"][int(i)] for i in idx]


def test_int32_indices_are_accepted():
    """KFold hands back int64 but other splitters may not; any integer array gathers."""
    df = _frame()
    idx = np.arange(0, 1000, 7, dtype=np.int32)
    assert _slice_rows_by_idx(df, idx).height == idx.size
