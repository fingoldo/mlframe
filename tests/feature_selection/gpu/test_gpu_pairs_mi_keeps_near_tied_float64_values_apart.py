"""The GPU pair MI bins a float64 candidate without merging values that are distinct only in float64.

A column 1 + i * 1e-12 narrowed to float32 is one value, so every row shared one bin and the MI was zero (audit PARG2-4, wave 3).
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


def test_near_tied_float64_candidate_keeps_its_mutual_information():
    """y follows the row index of a column 1 + i * 1e-12: the observed MI is ln 2 on the device."""
    from mlframe.feature_selection.filters import _gpu_resident_pair_mi as gp

    n = 40000
    i = np.arange(n)
    cand = (1.0 + i * 1e-12).reshape(-1, 1)
    y = (i >= n // 2).astype(np.int64)
    freqs = np.bincount(y, minlength=2) / n
    out = gp.gpu_pairs_fe_mi(cand, 20, y, y, freqs, 12, 0.95, False)
    assert out is not None
    assert float(np.asarray(out).ravel()[0]) > 0.6
