"""The batched joint entropy takes the fused kernel for large joints by default: the same occupied cells as count + reduce and entropies within a few ULP
(the two kernels sum the cells in a different order)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_batched_mi_cmi as C
from mlframe.feature_selection.filters._fe_batched_mi import _rows_entropy_and_k
from mlframe.feature_selection.filters._fe_cmi_fused_backend import _FUSED_ENTROPY_MIN_M, fused_entropy_backend


@pytest.mark.parametrize("kb", [4, 40, 100, 400, 1000])
def test_fused_equals_count_reduce(monkeypatch, kb):
    """Same occupied cells and entropies to round-off with the fused kernel forced on and off, skewed codes included."""
    rng = np.random.default_rng(0)
    n, k, kx = 7001, 37, 10
    X = cp.asarray(np.minimum(rng.geometric(0.3, (n, k)) - 1, kx - 1).astype(np.int64))
    b = cp.asarray(rng.integers(0, kb, n).astype(np.int64))
    want_h, want_k = _rows_entropy_and_k(C._batched_joint_counts2(X, b, kx, kb), 1.0 / n)
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSE_CMI_ENTROPY", "1")
    got = C._batched_joint_entropy_and_k2(X, b, kx, kb, 1.0 / n)
    assert got is not None
    assert cp.array_equal(got[1].astype(cp.int64), want_k.astype(cp.int64))
    assert float(cp.max(cp.abs(got[0] - want_h) / cp.abs(want_h))) <= 1e-14


def test_default_dispatch(monkeypatch):
    """Without the switch, small joints keep count + reduce and large ones take the fused kernel; ``0`` turns it off."""
    monkeypatch.delenv("MLFRAME_FE_GPU_FUSE_CMI_ENTROPY", raising=False)
    rng = np.random.default_rng(1)
    X = cp.asarray(rng.integers(0, 10, (3000, 5)).astype(np.int64))
    small = cp.asarray(rng.integers(0, 4, 3000).astype(np.int64))
    large = cp.asarray(rng.integers(0, 400, 3000).astype(np.int64))
    if fused_entropy_backend(3000, 40) == "count":
        assert C._batched_joint_entropy_and_k2(X, small, 10, 4, 1 / 3000) is None
    if fused_entropy_backend(3000, 4000) == "fused":
        assert C._batched_joint_entropy_and_k2(X, large, 10, 400, 1 / 3000) is not None
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSE_CMI_ENTROPY", "0")
    assert C._batched_joint_entropy_and_k2(X, large, 10, 400, 1 / 3000) is None
    assert _FUSED_ENTROPY_MIN_M > 40
