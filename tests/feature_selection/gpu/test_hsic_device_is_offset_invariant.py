"""The device HSIC does not change when a column is shifted by a large offset.

The kernel uses only pairwise differences, but the rows were narrowed to float32 before the subtraction: an offset of 1e6 has a step of about 0.06
and snapped a 0.01-spread column (audit PARG3-4, wave 3).
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


def test_device_hsic_is_invariant_to_a_large_row_offset():
    """Rows shifted by 1e6 give the same biased HSIC as the unshifted rows."""
    from mlframe.feature_selection.filters import _hsic_gpu as hg

    rng = np.random.default_rng(0)
    m, k = 200, 3
    xs = 0.01 * rng.normal(size=(k, m))
    A = rng.normal(size=(m, m))
    lc = A - A.mean(axis=0) - A.mean(axis=1, keepdims=True) + A.mean()
    lc = (lc + lc.T) / 2.0
    sigmas = np.full(k, 0.02)
    base = hg.hsic_biased_columns_gpu(xs, lc, sigmas, 1.0)
    shifted = hg.hsic_biased_columns_gpu(xs + 1e6, lc, sigmas, 1.0)
    np.testing.assert_allclose(shifted, base, rtol=1e-4, atol=1e-9)
