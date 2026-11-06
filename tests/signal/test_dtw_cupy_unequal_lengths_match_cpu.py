"""The GPU DTW distance for series of unequal length matches the CPU distance.

The device sweep returned the unreachable sentinel (1e9) for unequal lengths while the CPU path returns a finite distance (audit PARG4-11, wave 3).
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


def test_cupy_dtw_matches_cpu_for_unequal_lengths():
    """Lengths 1000 and 1300: the GPU banded and full distances both equal the CPU distance."""
    from mlframe.signal import dtw as d

    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    y = rng.normal(size=1300)
    cpu = d.dtw_cpu(x, y)[0]
    assert float(d.dtw_cupy_banded(x, y)[0]) == pytest.approx(cpu, rel=1e-9)
    assert float(d.dtw_cupy_full(x, y)[0]) == pytest.approx(cpu, rel=1e-9)
