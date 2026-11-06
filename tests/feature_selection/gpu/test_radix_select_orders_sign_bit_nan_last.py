"""The radix order statistics put a sign-bit NaN last, as the sort path does.

A NaN from ``inf - inf`` carries the sign bit, which keyed below -inf in the radix transform, so the rank-0 statistic came out NaN while the sorted
rank 0 is the smallest finite value (audit PARG1-4, wave 3).
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def test_sign_bit_nan_is_ordered_last_by_the_radix_select():
    """Column [inf - inf, 0, 1, 2]: rank 0 is 0.0 and rank 3 is NaN, as np.sort gives."""
    from mlframe.feature_selection.filters import _gpu_resident_select as rs

    x = np.array([np.inf - np.inf, 0.0, 1.0, 2.0])
    assert np.signbit(x[0])
    out = cp.asnumpy(rs.radix_order_statistics(cp.asarray(x[:, None]), [0, 3])).ravel()
    sorted_x = np.sort(x)
    assert out[0] == sorted_x[0] == 0.0
    assert np.isnan(out[1]) and np.isnan(sorted_x[3])
