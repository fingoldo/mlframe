"""``_codes_with_extra_columns``: the code matrix with the survivors' codes appended keeps the dtype of the matrix (``np.column_stack`` promoted an int8 matrix to the int64 of the extras)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._assign_support_tail import _codes_with_extra_columns


def _extras(n: int, k: int, high: int = 10, seed: int = 1) -> "list[np.ndarray]":
    """``k`` int64 code vectors of ``n`` rows below ``high``."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, high, n).astype(np.int64) for _ in range(k)]


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32])
def test_the_result_keeps_the_dtype_and_the_columns_of_the_stack(dtype):
    """Same values and column order as ``np.column_stack`` of the matrix and the vectors, in the dtype of the matrix."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 20, (500, 6)).astype(dtype)
    extra = _extras(500, 4)
    got = _codes_with_extra_columns(data, extra)
    expected = np.column_stack([data] + [e.reshape(-1, 1) for e in extra])
    assert got.dtype == np.dtype(dtype)
    assert got.shape == (500, 10)
    assert np.array_equal(got, expected)


def test_a_code_that_does_not_fit_takes_the_wider_dtype():
    """A value outside the range of an int8 matrix is not wrapped: the result is wide enough to hold it."""
    data = np.zeros((100, 3), dtype=np.int8)
    extra = [np.full(100, 300, dtype=np.int64)]
    got = _codes_with_extra_columns(data, extra)
    assert got.dtype.itemsize > 1
    assert np.array_equal(got[:, 3], np.full(100, 300))


def test_a_float_matrix_stays_float():
    """A float matrix takes the extras as floats (the stack would give the same dtype)."""
    data = np.linspace(0, 1, 60, dtype=np.float32).reshape(20, 3)
    got = _codes_with_extra_columns(data, _extras(20, 2))
    assert got.dtype == np.float32
    assert np.array_equal(got[:, :3], data)


def test_no_extra_columns_copies_the_matrix():
    """An empty list of extras gives an equal matrix of the same shape."""
    data = np.arange(12, dtype=np.int16).reshape(4, 3)
    got = _codes_with_extra_columns(data, [])
    assert np.array_equal(got, data) and got is not data
