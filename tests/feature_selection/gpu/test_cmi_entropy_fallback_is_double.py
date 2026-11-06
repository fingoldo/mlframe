"""The four-entropy CMI fallback reduces in float64, so it agrees with the host entropy to the 1e-9 gate.

With float32 entropies of size log(n) the cancellation error was about 1e-6 nats (audit PARG1-6, wave 3).
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


def test_fallback_entropy_matches_the_host_double_entropy():
    """A marginal of a large joint count tensor gives the double-precision entropy to 1e-9."""
    from mlframe.feature_selection.filters.info_theory import _cmi_cuda as cc

    rng = np.random.default_rng(0)
    counts = rng.integers(0, 40000, size=(1, 16, 16, 16)).astype(np.int64)
    n = int(counts.sum())
    got = cp.asnumpy(cc._entropy_from_counts_axis(cp.asarray(counts), (1, 2), n))
    marg = counts.sum(axis=(1, 2)).astype(np.float64) / n
    expected = -(marg * np.log(marg)).sum(axis=1) if marg.ndim == 2 else -(marg * np.log(marg)).sum()
    np.testing.assert_allclose(np.asarray(got).reshape(-1), np.asarray(expected).reshape(-1), atol=1e-9)
