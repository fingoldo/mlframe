"""The usability prescan scans every pair in blocks sized from the free host RAM."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fe_step import _step_pairs_rank as sp


@pytest.fixture
def host_only(monkeypatch):
    """Hide cupy so the block size follows the host budget alone (the VRAM bound is tested separately)."""
    monkeypatch.setitem(sys.modules, "cupy", None)


def test_block_size_follows_free_memory(monkeypatch, host_only):
    """Twice the free RAM gives twice the pairs per block; a block is never empty."""
    monkeypatch.setattr("mlframe.feature_selection.filters._orthogonal_univariate_fe._streamed_gpu_mi._host_free_bytes", lambda: 4 * 2**30)
    small = sp._prescan_block_pairs(100_000, np.float32)
    monkeypatch.setattr("mlframe.feature_selection.filters._orthogonal_univariate_fe._streamed_gpu_mi._host_free_bytes", lambda: 8 * 2**30)
    large = sp._prescan_block_pairs(100_000, np.float32)
    assert large >= 2 * small - 1 and small >= 1


def test_block_size_is_at_least_one_pair_when_memory_is_exhausted(monkeypatch, host_only):
    """No free memory still scores one pair at a time instead of refusing the scan."""
    monkeypatch.setattr("mlframe.feature_selection.filters._orthogonal_univariate_fe._streamed_gpu_mi._host_free_bytes", lambda: 0)
    assert sp._prescan_block_pairs(10_000_000, np.float64) == 1


def test_block_size_shrinks_with_rows(monkeypatch, host_only):
    """More rows per pair means fewer pairs per block under the same budget."""
    monkeypatch.setattr("mlframe.feature_selection.filters._orthogonal_univariate_fe._streamed_gpu_mi._host_free_bytes", lambda: 2**30)
    assert sp._prescan_block_pairs(1_000_000, np.float32) < sp._prescan_block_pairs(100_000, np.float32)


def test_block_size_is_capped_by_free_vram(monkeypatch):
    """With a device present, a block never asks the dispatcher for more than the VRAM cushion admits."""
    import cupy as cp

    from mlframe.feature_selection.filters._fe_gpu_vram import _cushion_bytes

    monkeypatch.setattr("mlframe.feature_selection.filters._orthogonal_univariate_fe._streamed_gpu_mi._host_free_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (2 * 2**30, 4 * 2**30))
    pairs = sp._prescan_block_pairs(30_000, np.float32)
    required = 2 * 30_000 * 4 * pairs + 8 * sp._N_PAIR_FORMS * pairs
    assert required <= (2 * 2**30 - _cushion_bytes(4 * 2**30)) * 0.9 + 2 * 30_000 * 4
