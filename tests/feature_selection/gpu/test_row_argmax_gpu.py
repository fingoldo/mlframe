"""The one-launch row argmax equals cp.argmax over the interleaved stack, ties included; the finiteness memo never goes stale."""

from __future__ import annotations

import gc

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._row_argmax_gpu import all_finite_cached, row_argmax_cm


@pytest.mark.parametrize("m", [2, 3, 4, 7])
def test_matches_cupy_argmax(m):
    """First-max index as float64, for random data."""
    rng = np.random.default_rng(m)
    cols = [cp.asarray(rng.normal(size=50_001)) for _ in range(m)]
    want = cp.argmax(cp.stack(cols, axis=1), axis=1).astype(cp.float64)
    got = row_argmax_cm(cp, cols)
    assert got.dtype == cp.float64 and got.shape == want.shape
    np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(want))


def test_ties_resolve_to_the_lowest_index():
    """Equal maxima give the first operand, as argmax does; all-equal rows give 0."""
    a = cp.asarray(np.array([1.0, 2.0, 2.0, 3.0, 5.0, 5.0]))
    b = cp.asarray(np.array([1.0, 2.0, 3.0, 3.0, 4.0, 5.0]))
    c = cp.asarray(np.array([1.0, 1.0, 3.0, 2.0, 5.0, 5.0]))
    np.testing.assert_array_equal(cp.asnumpy(row_argmax_cm(cp, [a, b, c])), cp.asnumpy(cp.argmax(cp.stack([a, b, c], axis=1), axis=1).astype(cp.float64)))


def test_signed_zero_and_negative_values():
    """-0.0 versus 0.0 compare equal (first wins) and negatives order normally."""
    a = cp.asarray(np.array([-0.0, -2.0, -1.0]))
    b = cp.asarray(np.array([0.0, -1.0, -1.0]))
    np.testing.assert_array_equal(cp.asnumpy(row_argmax_cm(cp, [a, b])), cp.asnumpy(cp.argmax(cp.stack([a, b], axis=1), axis=1).astype(cp.float64)))


def test_finiteness_memo_is_per_array_object():
    """A finite array is True, one with NaN or inf is False, repeated calls agree, and a new array reusing a freed id is not served the old answer."""
    good = np.arange(10.0)
    bad = np.array([1.0, np.nan])
    assert all_finite_cached(good) and all_finite_cached(good)
    assert not all_finite_cached(bad) and not all_finite_cached(bad)
    for _ in range(20):
        tmp = np.arange(5.0)
        assert all_finite_cached(tmp)
        del tmp
        gc.collect()
        tmp2 = np.array([np.inf, 1.0])
        assert not all_finite_cached(tmp2)


def test_a_copy_is_rescanned():
    """A new array object is always rescanned (the memo assumes a column is not edited in place between triples; FE operand columns are read-only inputs)."""
    x = np.arange(4.0)
    assert all_finite_cached(x)
    y = x.copy()
    y[0] = np.nan
    assert not all_finite_cached(y)
