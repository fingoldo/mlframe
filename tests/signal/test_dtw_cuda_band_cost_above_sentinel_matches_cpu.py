"""The banded numba.cuda DTW keeps a true path cost above 1e18 instead of letting an out-of-band sentinel undercut it.

Out-of-band cells held a finite 1e18. For ``x = [2e9, 0, 0]`` and ``y = [0, 0, 0]`` at window 1 the only in-band path costs 4e18 (the
diagonal). The cell beside it reads the 1e18 sentinel as its best predecessor, so the endpoint came out at 1e18 and the distance at
1e9, while the CPU reference returns 2e9.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.signal import dtw as D

nb_cuda = pytest.importorskip("numba.cuda")

try:
    _HAS_DEVICE = bool(nb_cuda.is_available())
except Exception:
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def test_numba_cuda_banded_dtw_matches_cpu_when_path_cost_exceeds_sentinel():
    """The numba.cuda distance equals the CPU distance when the in-band path cost is above 1e18."""
    x = np.array([2e9, 0.0, 0.0])
    y = np.array([0.0, 0.0, 0.0])

    cpu = D.dtw_cpu(x, y, window=1)
    cpu_distance = cpu[0] if isinstance(cpu, tuple) else cpu
    gpu_distance = D.dtw_cuda(x, y, window=1)[0]

    assert gpu_distance == pytest.approx(float(cpu_distance), rel=1e-9)
