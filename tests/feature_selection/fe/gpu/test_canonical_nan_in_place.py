"""The radix select's NaN canonicalisation rewrites NaN payloads in place and builds no second candidate matrix."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_resident_select import _canonical_nan


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sign_bit_nan_becomes_positive_nan_in_the_same_array(dtype):
    """A negative-signed NaN comes back as the positive quiet NaN, the other values are untouched, and the returned object is the input."""
    x = np.array([[1.0, -2.0, np.nan], [np.inf, -np.inf, 0.0]], dtype=dtype)
    bits = x.view(np.uint32 if dtype == np.float32 else np.uint64).copy()
    bits[0, 2] = bits.dtype.type((1 << (31 if dtype == np.float32 else 63)) | int(bits[0, 2]))  # sign-bit NaN
    xg = cp.asarray(bits.view(dtype))
    out = _canonical_nan(xg)
    assert out is xg
    got = cp.asnumpy(out)
    want = np.where(np.isnan(x), np.nan, x).astype(dtype)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[~np.isnan(got)], want[~np.isnan(want)])
    gbits = got.view(np.uint32 if dtype == np.float32 else np.uint64)
    assert int(gbits[0, 2]) >> (31 if dtype == np.float32 else 63) == 0


def test_integer_input_is_returned_unchanged_and_no_new_array_is_made():
    """Integer input is a no-op; a float input of a non-contiguous view is rewritten through the view."""
    xi = cp.arange(6, dtype=cp.int32)
    assert _canonical_nan(xi) is xi
    base = cp.asarray(np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float32))
    view = base[:, ::1]
    assert _canonical_nan(view) is view
