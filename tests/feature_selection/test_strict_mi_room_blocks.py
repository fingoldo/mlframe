"""The resident MI is scored in column blocks sized from the thread's pool room, with the values of one call; a matrix that fits is one call."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as M


def _require_device():
    """Skip only when there is no CUDA device at all; checked before the work, so the skip cannot follow a real run."""
    try:
        import cupy as _cp
        if _cp.cuda.runtime.getDeviceCount() > 0:
            return
    except Exception:
        pass
    pytest.skip("no CUDA device on this host")


def _case(k=40):
    """A float32 candidate matrix and binary labels."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, k)).astype(np.float32)
    y = (X[:, 0] + rng.standard_normal(4000) > 0).astype(np.int64)
    return X, y


def test_blocks_equal_one_call_and_a_fitting_matrix_is_one_call(monkeypatch):
    """Under a tiny slot pool the matrix is split (width 16) and equals the single call; with room it is a single call."""
    _require_device()
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    X, y = _case()
    whole = M._strict_resident_mi(X, y, 10, False)
    assert whole is not None, "strict resident MI must run on a host with a CUDA device"
    calls = []
    real = M._strict_resident_mi
    monkeypatch.setattr(M, "_strict_resident_mi", lambda Xc, *a: (calls.append(int(Xc.shape[1])), real(Xc, *a))[1])
    assert np.array_equal(M._strict_mi_in_room_blocks(X, y, 10, False), np.asarray(whole, dtype=np.float64))
    assert calls == [40]
    calls.clear()
    slot = cp.cuda.MemoryPool()
    slot.set_limit(size=4 * 1024 * 1024)  # room 4 MB: 0.5 * 4 MB / (24 B * 4000 rows) = 21 columns
    with cp.cuda.using_allocator(slot.malloc):
        out = M._strict_mi_in_room_blocks(X, y, 10, False)
    assert len(calls) > 1 and sum(calls) == 40
    assert np.array_equal(out, np.asarray(whole, dtype=np.float64))
