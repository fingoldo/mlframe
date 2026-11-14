"""``unique_with_counts`` gives ``np.unique(values, return_counts=True)`` for a large float32 vector through a radix sort and one scan, and numpy for everything else.

The adaptive binner computed the unique values and counts of every column it edged with ``np.unique`` (a comparison sort, 6.8 ms for 100000 float32 values; the radix route is 1.1 ms). The values and counts must be the same numbers for continuous
columns, tied columns, a column full of zeros of both signs, one value, two values and sizes around the switch-over size; other dtypes and small vectors keep numpy; threads running it at once do not disturb each other.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization import _quantile_radix as qr


def _same(values: np.ndarray) -> np.ndarray:
    """Values and counts equal numpy's (``array_equal`` compares -0.0 and +0.0 as equal, as the binner does); returns the counts."""
    got_v, got_c = qr.unique_with_counts(values)
    want_v, want_c = np.unique(values, return_counts=True)
    np.testing.assert_array_equal(got_v, want_v)
    np.testing.assert_array_equal(got_c, want_c)
    assert got_v.dtype == want_v.dtype and got_c.sum() == values.size
    return got_c


@pytest.mark.parametrize("n", [4095, 4096, 4097, 20_000, 100_003])
def test_continuous_ties_and_zero_columns_match_numpy(n):
    """A continuous column, one with a few thousand levels, one with ten levels and one with zeros of both signs, at sizes around the switch-over."""
    rng = np.random.default_rng(n)
    assert _same(rng.standard_normal(n).astype(np.float32)).sum() == n
    assert _same(rng.integers(0, 3000, n).astype(np.float32)).size > 2000
    assert _same(rng.integers(0, 10, n).astype(np.float32)).size == 10
    zeros = np.where(rng.random(n) < 0.4, np.float32(0.0), rng.standard_normal(n).astype(np.float32))
    zeros[rng.random(n) < 0.1] = np.float32(-0.0)
    zero_counts = _same(zeros)
    assert zero_counts.max() == np.count_nonzero(zeros == 0.0), "all zeros, of both signs, collapse into one level"


def test_one_value_and_two_values_and_extremes():
    """A constant column, two levels, and the largest and smallest finite float32 values."""
    n = 8192
    assert _same(np.full(n, 3.5, dtype=np.float32)).tolist() == [n]
    assert _same(np.where(np.arange(n) % 2 == 0, np.float32(1.0), np.float32(2.0))).tolist() == [n // 2, n // 2]
    info = np.finfo(np.float32)
    assert _same(np.where(np.arange(n) % 3 == 0, info.max, np.where(np.arange(n) % 3 == 1, info.min, np.float32(0.0))).astype(np.float32)).size == 3


def test_other_dtypes_and_small_vectors_take_numpy():
    """float64, int and 2-D inputs and vectors under the switch-over size are numpy's own result."""
    rng = np.random.default_rng(1)
    _same(rng.standard_normal(10_000))
    _same(rng.integers(0, 50, 10_000).astype(np.int32))
    _same(rng.standard_normal(100).astype(np.float32))
    two_d = rng.standard_normal((200, 50)).astype(np.float32)
    got_v, got_c = qr.unique_with_counts(two_d)
    want_v, want_c = np.unique(two_d, return_counts=True)
    np.testing.assert_array_equal(got_v, want_v)
    np.testing.assert_array_equal(got_c, want_c)


def test_the_input_is_not_modified_and_non_contiguous_views_work():
    """The radix sort works on a copy; a strided column view gives the result of its copy."""
    rng = np.random.default_rng(2)
    base = rng.standard_normal((20_000, 3)).astype(np.float32)
    column = base[:, 1]
    before = column.copy()
    got_v, got_c = qr.unique_with_counts(column)
    np.testing.assert_array_equal(column, before)
    want_v, want_c = np.unique(before, return_counts=True)
    np.testing.assert_array_equal(got_v, want_v)
    np.testing.assert_array_equal(got_c, want_c)


def test_threads_running_it_at_once_agree_with_the_serial_result():
    """The binner calls it from a thread pool: eight threads over different columns give each column's own result."""
    rng = np.random.default_rng(3)
    columns = [rng.integers(0, 2000 + 500 * j, 30_000).astype(np.float32) for j in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(qr.unique_with_counts, columns * 3))
    for column, (v, c) in zip(columns * 3, results):
        want_v, want_c = np.unique(column, return_counts=True)
        np.testing.assert_array_equal(v, want_v)
        np.testing.assert_array_equal(c, want_c)
