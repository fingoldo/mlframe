"""The GPU DTW distance matches the CPU distance when the costs exceed 1e18.

The unreachable cells held a finite 1e18, which can undercut a real cost above that value; the distance then came out too small (audit PARG4-3, wave 3).
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


def test_cupy_dtw_matches_cpu_when_costs_exceed_1e18():
    """x = [0, 0, 0] against y = [2e9] * 3: every cost is 4e18, and the CPU distance is 1.2e19."""
    from mlframe.signal import dtw as d

    x = np.zeros(3)
    y = np.full(3, 2e9)
    cpu = d.dtw_cpu(x, y)
    gpu = float(d.dtw_cupy_full(x, y))
    assert gpu == pytest.approx(cpu, rel=1e-9)
    assert gpu > 1e18
