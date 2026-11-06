"""The cupy polynomial basis stays finite for a large operand: the float32 build is rebuilt in float64 when it overflows.

A hermite basis of degree 25 with a |z|=40 outlier overflowed to inf in float32, so the candidate was dropped while the float64 twin scores it
(audit PARG2-3, wave 3).
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


def test_hermite_basis_of_an_outlier_is_finite_and_matches_numpy():
    """He_25 at an outlier of 40 is finite on the device and equals numpy's hermite_e value."""
    from numpy.polynomial import hermite_e
    from mlframe.feature_selection.filters import _cupy_polynom_optimizer as po

    x = np.random.default_rng(0).normal(size=500)
    x[7] = 40.0
    B = cp.asnumpy(po._basis_matrix_gpu(cp, cp.asarray(x), 25, 0))
    assert np.isfinite(B).all()
    expected = hermite_e.hermeval(40.0, np.eye(26)[25])
    assert B[7, 25] == pytest.approx(expected, rel=1e-5)
