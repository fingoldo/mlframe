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
