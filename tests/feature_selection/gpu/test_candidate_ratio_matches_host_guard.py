"""The device candidate ratio is u / max(|v|, 1e-6), the host rule, not u / (|v| + 1e-6).

With |v| near 1e-7 the additive guard changes the ratio by about 10 percent, so the device and host candidate orders differed (audit PARG1-2, wave 3).
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


def test_ratio_candidate_uses_the_host_guard_for_small_denominators():
    """The ratio column of the first pair equals u / max(|v|, 1e-6) for a denominator near 1e-7."""
    from mlframe.feature_selection.filters import _resident_candidate_mi as rc

    u = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([1e-7, -2e-7, 0.5, 1e-7])
    out = cp.asnumpy(rc._build_best_existing_op_candidates_gpu([cp.asarray(u), cp.asarray(v)], cp))
    m = 2
    expected = u / np.maximum(np.abs(v), 1e-6)
    np.testing.assert_allclose(out[:, m + 2], expected, rtol=1e-5)
