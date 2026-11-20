"""``_float32_unsafe`` reads the operand matrix in its (operands, rows) layout, and the dispatcher's CUDA fallback is logged."""

from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as bpu


def test_float32_unsafe_sees_per_operand_offsets_in_operand_major_layout():
    """Two operands at offsets 1e8 and 2e8 with a small spread are unsafe for float32; the mixed-up reshape averaged them into a wide column."""
    n = 100
    i = np.arange(n, dtype=np.float64)
    y = np.sin(i)
    operands = np.stack([1e8 + 0.1 * i, 2e8 + 0.1 * i])
    assert bpu._float32_unsafe(y, operands) is True


def test_float32_unsafe_false_for_well_scaled_operands():
    """Standard-normal operands stay float32-safe."""
    rng = np.random.default_rng(0)
    assert bpu._float32_unsafe(rng.standard_normal(200), rng.standard_normal((3, 200))) is False


def test_cuda_failure_in_usability_dispatch_warns(monkeypatch, caplog):
    """A CUDA failure falls back to the CPU table and logs a WARNING."""
    import mlframe.feature_selection.filters._fe_gpu_vram as vram
    import mlframe.feature_selection.filters._gpu_policy as policy

    def boom(*a, **k):
        """Raise a driver fault."""
        raise RuntimeError("launch failed")

    monkeypatch.setattr(bpu, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bpu, "batch_pair_usability_corr_cuda_warp", boom)
    monkeypatch.setattr(bpu, "_device_wins", lambda n: True)
    monkeypatch.setattr(policy, "gpu_globally_disabled", lambda: False)
    monkeypatch.setattr(vram, "fe_gpu_has_vram_cushion", lambda nbytes: True)
    rng = np.random.default_rng(1)
    y = rng.standard_normal(60).astype(np.float32)
    ops = rng.standard_normal((3, 60)).astype(np.float32)
    with caplog.at_level(logging.WARNING, logger=bpu.logger.name):
        table, backend = bpu._dispatch_batch_pair_usability_corr_uncached(y, ops, np.array([0, 1]), np.array([1, 2]), None, None)
    assert backend == "cpu" and table.shape == (2, 9)
    assert any(r.levelno == logging.WARNING and "launch failed" in r.getMessage() for r in caplog.records)
