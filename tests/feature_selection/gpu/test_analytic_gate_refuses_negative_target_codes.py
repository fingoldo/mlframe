"""The fused analytic gate declines a target with negative codes and leaves them to the generic path.

A -1 sentinel in the target indexed the shared histogram below zero (audit PARG2-7, wave 3).
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


def test_negative_target_code_returns_none_for_the_generic_path():
    """With a -1 in the target the fused analytic path returns None."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_analytic_gpu as pa

    rng = np.random.default_rng(0)
    n, k = 300, 3
    codes = cp.asarray(rng.integers(0, 4, (n, k)).astype(np.int8))
    yc = rng.integers(0, 2, n).astype(np.int64)
    yc[::17] = -1
    assert pa._fused_observed_mi_and_bins(codes, yc, 2) is None
