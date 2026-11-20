"""A GPU backend failure inside the noise-gate dispatcher is logged at WARNING (it used to be debug-only), driven on the CPU by monkeypatching."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import batch_mi_noise_gate_gpu as bng


def _args():
    """Minimal gate arguments; the patched backends never read them."""
    disc = np.zeros((10, 2), dtype=np.int32)
    return disc, np.array([2, 2]), np.zeros(10, dtype=np.int32), np.zeros(10, dtype=np.int32), np.array([1.0]), 3, np.uint64(1), 0.5, False


@pytest.mark.parametrize("backend", ["cupy", "cuda"])
def test_backend_failure_in_noise_gate_dispatch_warns(monkeypatch, caplog, backend):
    """The dispatcher returns None (CPU fallback) and emits a WARNING naming the failed backend."""
    import mlframe.feature_selection.filters._fe_gpu_vram as vram
    import mlframe.feature_selection.filters._gpu_policy as policy

    def boom(*a, **k):
        """Raise like a driver fault."""
        raise RuntimeError("driver fault")

    monkeypatch.setattr(policy, "gpu_globally_disabled", lambda: False)
    monkeypatch.setattr(vram, "fe_gpu_has_vram_cushion", lambda nbytes: True)
    monkeypatch.setattr(bng, "_batch_mi_noise_gate_backend_choice", lambda n, k: backend)
    monkeypatch.setattr(bng, "cupy_available", lambda: True)
    monkeypatch.setattr(bng, "cuda_available", lambda: True)
    monkeypatch.setattr(bng, "batch_mi_with_noise_gate_cupy", boom)
    monkeypatch.setattr(bng, "batch_mi_with_noise_gate_cuda", boom)
    with caplog.at_level(logging.WARNING, logger=bng.logger.name):
        assert bng.dispatch_batch_mi_with_noise_gate_gpu(*_args()) is None
    assert any(r.levelno == logging.WARNING and backend in r.getMessage() and "driver fault" in r.getMessage() for r in caplog.records)
