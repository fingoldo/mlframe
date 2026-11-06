"""The on-device segment entropies are double precision and agree with the host entropy to 1e-9.

Float32 entropies of a 64-cell joint left about 3e-7 nats of error, above the documented 1e-9 agreement (audit PARG3-5, wave 3).
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


def test_segment_entropy_matches_the_host_double_entropy():
    """Two independent 8-bin columns: the joint entropy from the device equals the host double entropy to 1e-9."""
    from mlframe.feature_selection.filters import _friend_graph_gpu_entropy_resident as fe

    rng = np.random.default_rng(0)
    n = 20000
    x = rng.integers(0, 8, n)
    y = rng.integers(0, 8, n)
    counts = np.bincount(x * 8 + y, minlength=64).astype(np.int64)
    got = fe.entropy_segments_gpu(cp, cp.asarray(counts), n, np.array([0, 64]))
    p = counts / n
    p = p[p > 0]
    expected = -(p * np.log(p)).sum()
    assert got.dtype == np.float64
    assert float(got[0]) == pytest.approx(expected, abs=1e-9)
