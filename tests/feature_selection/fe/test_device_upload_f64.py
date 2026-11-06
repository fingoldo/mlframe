"""The float64 device upload widens a float32 host matrix on the device and gives the matrix the host-side widening gave."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._device_upload_f64 import upload_as_float64


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_upload_equals_host_widening(dtype):
    """Same values, float64, C-contiguous, for float32 / float64 / integer input and for a non-contiguous view."""
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(1000, 7)) * 100).astype(dtype)
    for host in (X, X[:, ::2]):
        got = upload_as_float64(cp, host)
        want = np.asarray(host, dtype=np.float64)
        assert got.dtype == cp.float64 and got.flags.c_contiguous
        np.testing.assert_array_equal(cp.asnumpy(got), want)


def test_list_input_and_non_finite_values():
    """A nested list is accepted and NaN / inf survive the float32 round trip."""
    X = np.array([[1.5, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    np.testing.assert_array_equal(cp.asnumpy(upload_as_float64(cp, X)), np.asarray(X, dtype=np.float64))
    assert cp.asnumpy(upload_as_float64(cp, [[1.0, 2.0], [3.0, 4.0]])).tolist() == [[1.0, 2.0], [3.0, 4.0]]
