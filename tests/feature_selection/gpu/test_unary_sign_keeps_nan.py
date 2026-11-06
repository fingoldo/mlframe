"""The device sign of a NaN is NaN, as the NumPy and njit twins give; it used to be 0 (audit PARG3-6, wave 3)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")

_SIGN_CODE = 8  # the sign operation in the kernel's op table


def test_device_sign_of_nan_is_nan_and_other_values_keep_their_sign():
    """The sign op maps NaN to NaN and +/- values to +1 / -1 on the device."""
    from mlframe.feature_selection.filters import _unary_stack_cuda as us

    x = cp.asarray(np.array([np.nan, -2.0, 0.0, 3.5]))
    codes = cp.asarray(np.array([_SIGN_CODE], dtype=np.int32))
    out = cp.asnumpy(us.unary_stack_cuda(x, codes, 0.0))
    assert np.isnan(out[0, 0])
    np.testing.assert_array_equal(out[0, 1:], np.array([-1.0, 0.0, 1.0]))
