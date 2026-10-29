"""``LazyHostCodes`` copies wide integer codes in one narrow transfer and returns the same array as the dense copy."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit
from mlframe.feature_selection.filters._lazy_host_codes import LazyHostCodes


@pytest.mark.parametrize("lo,hi", [(0, 12), (-128, 127), (0, 127), (-5, 5)])
def test_codes_that_fit_int8_cross_in_one_small_transfer(lo, hi):
    """Codes inside the int8 range travel as one transfer of about one byte per code, and the host array equals the dense copy in value and dtype."""
    rng = np.random.default_rng(0)
    codes = rng.integers(lo, hi + 1, 6000).astype(np.int64)
    codes[0], codes[1] = lo, hi
    lazy = LazyHostCodes(cp.asarray(codes), np.int64)
    with residency_audit() as rep:
        host = np.asarray(lazy)
    np.testing.assert_array_equal(host, codes)
    assert host.dtype == np.int64
    assert len(rep.d2h) == 1 and rep.d2h[0] < codes.nbytes // 4


@pytest.mark.parametrize("hi", [128, 1000, 70000])
def test_codes_outside_int8_fall_back_to_the_dense_copy(hi):
    """A value that does not fit int8 gives the exact dense array (the narrow attempt is discarded)."""
    codes = (np.arange(5000, dtype=np.int64) * 7) % 13
    codes[17] = hi
    codes[18] = -3
    lazy = LazyHostCodes(cp.asarray(codes), np.int64)
    np.testing.assert_array_equal(np.asarray(lazy), codes)


def test_two_dimensional_and_float_arrays_keep_their_layout():
    """A 2-D code block keeps its shape and values; a float device array takes the dense path untouched."""
    codes = (np.arange(3 * 4000, dtype=np.int64).reshape(4000, 3) * 5) % 9
    np.testing.assert_array_equal(np.asarray(LazyHostCodes(cp.asarray(codes), np.int64)), codes)
    vals = np.linspace(-3.0, 3.0, 5000)
    np.testing.assert_array_equal(np.asarray(LazyHostCodes(cp.asarray(vals), np.float64)), vals)


def test_small_arrays_use_the_dense_copy():
    """Below the size threshold the range read is not worth it: one dense transfer."""
    codes = np.arange(100, dtype=np.int64) % 7
    lazy = LazyHostCodes(cp.asarray(codes), np.int64)
    with residency_audit() as rep:
        np.testing.assert_array_equal(np.asarray(lazy), codes)
    assert len(rep.d2h) == 1 and rep.d2h[0] == codes.nbytes


def test_host_as_float32_equals_the_host_cast_and_crosses_once_at_half_the_bytes():
    """A lazy float64 array read as float32 equals ``astype(float32)`` of the host values (NaN, infinities, overflow, ties included), crosses as half the bytes, and a second read
    of the same dtype, by any reader, costs no further transfer."""
    from mlframe.feature_selection.filters._lazy_host_codes import host_as

    rng = np.random.default_rng(5)
    vals = np.concatenate([rng.standard_normal(5000) * 10.0 ** rng.integers(-30, 30, 5000), [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0 + 2.0**-24, 3.4e38, 1e39]])
    lazy = LazyHostCodes(cp.asarray(vals), np.float64)
    with residency_audit() as rep:
        first = host_as(lazy, np.float32)
        second = lazy.host_as(np.float32)
    with np.errstate(over="ignore"):  # 1e39 overflows float32 on purpose: both conversions give infinity
        want = vals.astype(np.float32)
    assert first.dtype == np.float32 and second is first
    np.testing.assert_array_equal(first.view(np.uint32), want.view(np.uint32))
    assert rep.bulk_d2h == [want.nbytes]


def test_host_as_flushes_float32_subnormals_to_zero_and_nothing_else_changes():
    """The device conversion turns results below the smallest normal float32 into zero; every other value equals the host conversion."""
    from mlframe.feature_selection.filters._lazy_host_codes import host_as

    vals = np.concatenate([np.linspace(-3.0, 3.0, 4000), [-8.019314252534474e-39, 1.0970639932180818e-41, 1e-45, 0.0]])
    got = host_as(LazyHostCodes(cp.asarray(vals), np.float64), np.float32)
    want = vals.astype(np.float32)
    small = np.abs(want) < np.finfo(np.float32).tiny
    np.testing.assert_array_equal(got[~small], want[~small])
    assert not got[small].any()


def test_host_as_leaves_host_arrays_equal_width_and_already_copied_arrays_alone():
    """Plain arrays, an equal-width request and an already-copied lazy array take the ordinary conversion."""
    from mlframe.feature_selection.filters._lazy_host_codes import host_as

    a = np.linspace(0.0, 1.0, 10)
    np.testing.assert_array_equal(host_as(a, np.float32), a.astype(np.float32))
    lazy = LazyHostCodes(cp.asarray(a), np.float64)
    np.testing.assert_array_equal(host_as(lazy, np.float64), a)
    _ = np.asarray(lazy)
    np.testing.assert_array_equal(host_as(lazy, np.float32), a.astype(np.float32))
