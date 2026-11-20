"""Friend-graph GPU: tile sizing counts every live buffer, and each CPU fallback is logged at WARNING."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_gpu_vram, _gpu_policy, friend_graph_gpu as fgg


def test_pairs_per_tile_counts_all_live_buffers():
    """A tile's live int64 buffers (several rows of n) must fit the 35% budget."""
    n, free_b, jc = 1_000_000, 4 * 1024**3, 64
    p = fgg._pairs_per_tile(free_b, n, jc, 10**6)
    assert p >= 1
    assert p * 4 * n * 8 <= int(free_b * 0.35)
    assert fgg._pairs_per_tile(free_b, n, jc, 3) == 3


@pytest.fixture
def _gpu_ready(monkeypatch):
    """Pretend a GPU is usable so dispatch reaches the backend call."""
    monkeypatch.setattr(_gpu_policy, "gpu_globally_disabled", lambda: False)
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda b: True)
    monkeypatch.setattr(fgg, "_CUPY_AVAIL", True)
    monkeypatch.setattr(fgg, "_CUDA_AVAIL", True)

    def _boom(*a, **k):
        """Raise a fake device fault."""
        raise RuntimeError("device fault")

    monkeypatch.setattr(fgg, "friend_graph_stats_cupy", _boom)
    monkeypatch.setattr(fgg, "friend_graph_stats_cuda", _boom)


def _args():
    """Tiny valid dispatch inputs."""
    data = np.zeros((10, 3), dtype=np.int32)
    return [0, 1], data, np.array([2, 2, 2]), np.array([2])


def test_forced_backend_failure_logs_warning(_gpu_ready, caplog):
    """A forced backend that faults falls back to CPU with a WARNING."""
    with caplog.at_level(logging.WARNING, logger=fgg.logger.name):
        assert fgg.dispatch_friend_graph_stats(*_args(), force_backend="cupy") is None
    assert any("forced cupy backend failed" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("choice", ["cupy", "cuda"])
def test_chosen_backend_failure_logs_warning(_gpu_ready, monkeypatch, caplog, choice):
    """A tuned backend that faults falls back to CPU with a WARNING."""
    monkeypatch.setattr(fgg, "_friend_graph_backend_choice", lambda n, k: choice)
    with caplog.at_level(logging.WARNING, logger=fgg.logger.name):
        assert fgg.dispatch_friend_graph_stats(*_args()) is None
    assert any(f"{choice} backend failed" in r.getMessage() for r in caplog.records)


def test_vram_cushion_decline_logs_warning(_gpu_ready, monkeypatch, caplog):
    """Declining the GPU for lack of free VRAM is logged."""
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda b: False)
    with caplog.at_level(logging.WARNING, logger=fgg.logger.name):
        assert fgg.dispatch_friend_graph_stats(*_args()) is None
    assert any("lack of free VRAM" in r.getMessage() for r in caplog.records)
