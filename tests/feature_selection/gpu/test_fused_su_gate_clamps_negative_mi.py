"""The fused noise-gate statistic is never negative: rounding on a constant column gave a tiny negative MI, and the host clamps it to 0.

With class frequencies perturbed by rounding-scale amounts the fused SU came out at about -6e-16 (audit PARG2-6, wave 3). The fused path is enabled
through its environment switch for this test.
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


def test_fused_su_of_a_constant_column_is_not_negative(monkeypatch):
    """A single-bin column against a binary target gives SU >= 0 on the fused path."""
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSED_GATE", "1")
    from mlframe.feature_selection.filters import _batch_mi_noise_gate_fused as fg

    n = 400
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n).astype(np.int32)
    freqs_y = np.bincount(y, minlength=2) / n * (1.0 + 3e-16)
    out = fg.fused_gate_mi(
        cp.asarray(np.zeros((n, 1), dtype=np.int32)),
        np.array([1]),
        cp.asarray(np.array([1], dtype=np.int32)),
        cp.asarray(y[None, :]),
        cp.asarray(freqs_y.astype(np.float64)),
        freqs_y,
        n, 1, 2, 1, 32, True,
    )
    assert out is not None
    assert out[0, 0] >= 0.0
