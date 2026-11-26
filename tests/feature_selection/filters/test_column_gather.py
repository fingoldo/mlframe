"""The parallel column gather of a row-major code matrix equals numpy's gather and transpose, padded."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._column_gather import gather_columns_cm


def _reference(data: np.ndarray, cols: np.ndarray, pad_to: int, dtype) -> np.ndarray:
    """numpy's gather, transpose and zero padding."""
    n = data.shape[0]
    out = np.zeros((len(cols), -(-n // pad_to) * pad_to), dtype=dtype)
    out[:, :n] = data[:, cols].T
    return out


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.int32])
@pytest.mark.parametrize("n", [1, 63, 64, 65, 8191, 8193, 20_001])
def test_the_block_equals_numpys_gather_for_every_row_count_around_the_tiles(dtype, n):
    """Row counts at, below and above the tile (64) and the row block (8192): equal values, zero padding to a multiple of four."""
    rng = np.random.default_rng(n)
    data = rng.integers(0, 120, (n, 37)).astype(dtype)
    cols = np.array([36, 0, 5, 5, 17], dtype=np.int64)
    out = gather_columns_cm(data, cols, pad_to=4)
    assert out.dtype == np.uint8 and out.shape == (5, -(-n // 4) * 4)
    assert np.array_equal(out, _reference(data, cols, 4, np.uint8))


def test_a_column_view_of_a_larger_matrix_and_an_fortran_matrix_give_the_same_block():
    """Non-contiguous inputs (a slice of columns, a column-major matrix) are read through their own strides."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 50, (5000, 60)).astype(np.uint8)
    cols = np.array([3, 59, 10], dtype=np.int64)
    expected = _reference(data, cols, 1, np.uint8)
    assert np.array_equal(gather_columns_cm(np.asfortranarray(data), cols), expected)
    assert np.array_equal(gather_columns_cm(data[:, ::2], np.array([1, 4]), pad_to=1), _reference(data[:, ::2], np.array([1, 4]), 1, np.uint8))


def test_no_columns_and_no_rows_give_empty_blocks_of_the_right_shape():
    """An empty column list gives ``(0, n_pad)``; no rows give ``(m, 0)``."""
    data = np.zeros((10, 4), dtype=np.uint8)
    assert gather_columns_cm(data, np.empty(0, dtype=np.int64), pad_to=4).shape == (0, 12)
    assert gather_columns_cm(np.zeros((0, 4), dtype=np.uint8), np.array([1, 2]), pad_to=4).shape == (2, 0)


@pytest.mark.parametrize("dtype", [np.int8, np.int32])
@pytest.mark.parametrize("n", [1, 63, 65, 8193, 20_001])
def test_the_scatter_writes_each_block_row_into_its_matrix_column(dtype, n):
    """``dst[:, cols[j]] = src[j]`` for scattered (not contiguous, not sorted-adjacent) destination columns, cast to the matrix dtype; the other columns stay as they were."""
    from mlframe.feature_selection.filters._column_gather import scatter_columns_rm

    rng = np.random.default_rng(n)
    dst = np.full((n, 11), -1, dtype=dtype)
    src = rng.integers(0, 100, (4, n)).astype(np.int16)
    cols = np.array([9, 0, 4, 5], dtype=np.int64)
    scatter_columns_rm(dst, cols, src)
    expected = np.full((n, 11), -1, dtype=dtype)
    for j, c in enumerate(cols):
        expected[:, c] = src[j]
    assert np.array_equal(dst, expected)


def test_the_scatter_refuses_a_block_that_does_not_fit():
    """A block with another row count, or more rows than destination columns, is an error and writes nothing."""
    from mlframe.feature_selection.filters._column_gather import scatter_columns_rm

    dst = np.zeros((10, 3), dtype=np.int8)
    with pytest.raises(ValueError, match="does not fit"):
        scatter_columns_rm(dst, np.array([0, 1]), np.zeros((2, 9), dtype=np.int16))
    with pytest.raises(ValueError, match="does not fit"):
        scatter_columns_rm(dst, np.array([0]), np.zeros((2, 10), dtype=np.int16))
    assert not dst.any()
