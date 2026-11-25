"""The label-carrying radix sort of ``sort_xy`` gives the order and the values of the index-carrying kernels, for float32 columns and for float32 columns held in double precision."""

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization import _quantile_radix as qr


def _column(n=6000, seed=0):
    """A float32 column with heavy ties, both zeros, an infinity pair and a wide range."""
    rng = np.random.default_rng(seed)
    x = np.round(rng.standard_normal(n) * 50.0, 1).astype(np.float32)
    x[::11] = np.float32(0.0)
    x[1::11] = np.float32(-0.0)
    x[5] = np.float32(np.inf)
    x[6] = np.float32(-np.inf)
    x[7] = np.float32(1e-38)
    return x


def _same(a, b):
    """Equal values bit for bit (the sign of a zero included) and equal labels."""
    return a[0].dtype == b[0].dtype and np.array_equal(a[0].view(np.int64), b[0].view(np.int64)) and a[1].dtype == b[1].dtype and np.array_equal(a[1], b[1])


@pytest.mark.parametrize("label_dtype", [np.int8, np.uint8, np.int16, np.int32, np.int64])
def test_a_float32_column_sorts_as_the_index_kernel_sorts_it(label_dtype):
    """Same widened values (zeros of both signs in the same places), same stable order of the labels."""
    x = _column()
    y = (np.random.default_rng(1).random(x.size) * 5).astype(label_dtype)
    assert _same(qr.sort_xy(x, y), qr._sort_xy_f32_njit(x, y))


@pytest.mark.parametrize("label_dtype", [np.int8, np.int32])
def test_a_float32_column_held_in_double_takes_the_narrow_route_and_equals_the_double_kernel(label_dtype, monkeypatch):
    """The values widened from float32 are sorted on 4-byte keys; the result equals the double-precision kernel's."""
    x = _column().astype(np.float64)
    y = (np.random.default_rng(2).random(x.size) < 0.4).astype(label_dtype)
    reference = qr._sort_xy_f64_njit(x, y)
    called = []
    real = qr._sort_xy_pairs_f32_njit
    monkeypatch.setattr(qr, "_sort_xy_pairs_f32_njit", lambda *a: (called.append(1), real(*a))[1])
    assert _same(qr.sort_xy(x, y), reference) and called == [1]


def test_a_double_column_that_is_not_float32_keeps_the_double_kernel(monkeypatch):
    """One value that float32 cannot hold sends the whole column to the double-precision kernel."""
    x = _column().astype(np.float64)
    x[100] = 0.1  # not a float32 value
    y = (np.random.default_rng(3).random(x.size) < 0.5).astype(np.int8)
    monkeypatch.setattr(qr, "_sort_xy_pairs_f32_njit", lambda *a: pytest.fail("the narrow kernel must not run"))
    assert _same(qr.sort_xy(x, y), qr._sort_xy_f64_njit(x, y))


def test_nan_short_vectors_and_the_switch_behave_as_before(monkeypatch):
    """A double column with NaN and a short column give None; ``MLFRAME_SORT_XY_PAIRS=0`` runs the index kernels."""
    x = _column().astype(np.float64)
    y = np.zeros(x.size, dtype=np.int8)
    nan = x.copy()
    nan[3] = np.nan
    assert qr.sort_xy(nan, y) is None and qr.sort_xy(x[:100], y[:100]) is None
    monkeypatch.setenv("MLFRAME_SORT_XY_PAIRS", "0")
    monkeypatch.setattr(qr, "_sort_xy_pairs_f32_njit", lambda *a: pytest.fail("the pairs kernel is switched off"))
    x32 = _column()
    assert _same(qr.sort_xy(x32, y), qr._sort_xy_f32_njit(x32, y)) and _same(qr.sort_xy(x, y), qr._sort_xy_f64_njit(x, y))


def test_the_float64_exactness_probe():
    """True with the float32 copy for widened float32 values, False for a value float32 cannot hold, for an overflow to infinity and for a denormal underflow."""
    ok, narrow = qr._f64_as_f32_njit(_column(100).astype(np.float64))
    assert ok and narrow.dtype == np.float32 and narrow.size == 100
    for bad in (0.1, 1e300, 1e-60):
        values = np.array([1.5, bad, 2.0])
        assert qr._f64_as_f32_njit(values)[0] is False
