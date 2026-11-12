"""GPU memory and pinned staging are released when a GPU step raises, not only when it succeeds (tracker 2026-11-02, GPU5-1, GPU5-2, GPU5-3)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import _kernels_cupy as kc
from mlframe.feature_engineering.transformer import random_features as rf
from mlframe.feature_engineering.transformer import row_attention as ra


def _spy(monkeypatch):
    """Replace the two release calls in both modules with recorders and return the list they append to."""
    calls: list[str] = []
    monkeypatch.setattr(rf, "free_gpu_memory_pool", lambda: calls.append("pool"))
    monkeypatch.setattr(ra, "free_gpu_memory_pool", lambda: calls.append("pool"))
    monkeypatch.setattr(kc, "clear_pinned_buffers", lambda: calls.append("pinned") or 0)
    monkeypatch.setattr(ra, "clear_pinned_buffers", lambda: calls.append("pinned") or 0)
    # The release sits inside the GPU branch, which use_gpu=True only enters when a device exists. The kernels are
    # mocked here, so force that branch and exercise the release contract without requiring hardware.
    monkeypatch.setattr(rf, "_resolve_use_gpu", lambda *a, **k: True)
    return calls


def _boom(*args, **kwargs):
    """Stand in for a GPU kernel that fails part-way."""
    raise RuntimeError("simulated out-of-memory")


def test_rff_failure_still_releases_pool_and_pinned_buffers(monkeypatch):
    """When rff_matmul_cupy raises, the CPU fallback answers and both the device pool and the pinned buffers are still released."""
    calls = _spy(monkeypatch)
    monkeypatch.setattr(kc, "rff_matmul_cupy", _boom)
    state = rf.RFFState(None, np.ones((3, 4), dtype=np.float32), np.zeros(4, dtype=np.float32), 0.5, 8, "rff", np.float32)
    out = rf._rff_project(state, np.ones((10, 3), dtype=np.float32), use_gpu=True, gpu_threshold=0, batch_rows=100, release_memory_after=True)
    assert out.shape == (10, 8)
    assert "pool" in calls and "pinned" in calls


def test_rff_success_path_releases_too(monkeypatch):
    """On success the release still happens exactly as before."""
    calls = _spy(monkeypatch)
    monkeypatch.setattr(kc, "rff_matmul_cupy", lambda X, W, b, out, scale, batch_rows=0: None)
    state = rf.RFFState(None, np.ones((3, 4), dtype=np.float32), np.zeros(4, dtype=np.float32), 0.5, 8, "rff", np.float32)
    rf._rff_project(state, np.ones((10, 3), dtype=np.float32), use_gpu=True, gpu_threshold=0, batch_rows=100, release_memory_after=True)
    assert "pool" in calls and "pinned" in calls


def test_row_attention_failure_still_releases(monkeypatch):
    """When a scale of the attention pipeline raises, the exception reaches the caller and the pool and pinned buffers are released first."""
    calls = _spy(monkeypatch)
    monkeypatch.setattr(ra, "_attend_one_scale", _boom)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    y = rng.normal(size=60).astype(np.float32)
    with pytest.raises(RuntimeError, match="simulated"):
        ra.compute_row_attention(X, y, None, KFold(n_splits=3, shuffle=True, random_state=0), seed=0, n_heads=1, head_dim=2, k=4, gpu_stage4=False)
    assert "pool" in calls and "pinned" in calls
