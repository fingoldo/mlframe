"""A cupy ``OutOfMemoryError`` in the resident order-2 pair-maxT floor must not trip the process-wide GPU circuit breaker (AUD23-02).

The breaker exists for a poisoned CUDA context (a WDDM-TDR launch failure), where every later attempt re-faults. An out-of-memory error is recoverable: the pool can be emptied and the next call may fit. Tripping on it
sent every later floor of the process to the CPU njit floor (measured 2026-11-23 in a profiled 100000 x 100 fit: pool limit 2.58 GB, 693 MB allocation refused, breaker tripped).
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _permutation_null_pair_resident as RES


def _inputs(n: int = 400, n_vars: int = 6, k_bins: int = 4):
    """A small discretised pool: every pair of ``n_vars`` columns, a binary target."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, k_bins, size=(n, n_vars)).astype(np.int32)
    pa, pb = np.triu_indices(n_vars, 1)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    return data, pa.astype(np.int64), pb.astype(np.int64), np.full(n_vars, k_bins, dtype=np.int64), y, np.full(2, 0.5)


def _call(monkeypatch, exc: Exception):
    """Run the floor with the tiled kernel replaced by one that raises ``exc``; returns the result."""
    import mlframe.feature_selection.filters._permutation_null_pair_tiles as tiles

    def boom(*a, **k):
        """Stand in for the tiled device computation and fail."""
        raise exc

    monkeypatch.setattr(tiles, "_pooled_pair_null_perm_tiles", boom)
    data, pa, pb, nb, y, fy = _inputs()
    return RES.pooled_pair_permutation_null_joint_mi_floor_cupy(data, pa, pb, nb, y, fy, n_permutations=8)


def test_out_of_memory_returns_none_without_tripping_the_breaker(monkeypatch):
    """The call falls back to the CPU floor for THIS call (``None``) and leaves the GPU path armed for the next one."""
    RES.reset_pair_maxt_gpu_circuit_breaker()
    try:
        result = _call(monkeypatch, cp.cuda.memory.OutOfMemoryError(1 << 30, 1 << 20))
        assert result is None
        assert RES._PAIR_MAXT_GPU_FAILED is False, "an out-of-memory error is recoverable and must not trip the breaker"
    finally:
        RES.reset_pair_maxt_gpu_circuit_breaker()


def test_a_launch_fault_still_trips_the_breaker(monkeypatch):
    """Contrast: any other exception (a poisoned context) keeps tripping, so the fix does not remove the protection."""
    RES.reset_pair_maxt_gpu_circuit_breaker()
    try:
        result = _call(monkeypatch, RuntimeError("synthetic launch failure"))
        assert result is None
        assert RES._PAIR_MAXT_GPU_FAILED is True
    finally:
        RES.reset_pair_maxt_gpu_circuit_breaker()
