"""Pair-MI dispatcher fallbacks are logged through ``log_throttle`` at WARNING, including a cupy runtime fault that surfaces as a ``RuntimeError`` subclass."""

from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_selection.filters import batch_pair_mi_gpu as bpm


def _dispatch():
    """A tiny ``_PairMiDispatch`` over a 4-column frame."""
    rng = np.random.default_rng(0)
    fd = rng.integers(0, 3, size=(50, 4)).astype(np.int32)
    y = rng.integers(0, 2, size=50).astype(np.int32)
    freqs = np.bincount(y, minlength=2).astype(np.float64) / 50
    return bpm._PairMiDispatch(fd, np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64), np.full(4, 3, dtype=np.int32), y, freqs)


class _CudaFault(RuntimeError):
    """Stand-in for ``cupy.cuda.runtime.CUDARuntimeError``."""


_CudaFault.__module__ = "cupy.cuda.runtime"


def test_feature_major_cuda_runtime_fault_is_warned_not_debugged(monkeypatch, caplog):
    """A cupy-origin RuntimeError is a fault, not a shape decline: WARNING and the row-major kernels run."""
    import mlframe.feature_selection.filters._batch_pair_mi_cuda_fm as fm

    def boom(*a, **k):
        """Raise the fake cupy fault."""
        raise _CudaFault("illegal address")

    monkeypatch.setattr(bpm, "_CUPY_AVAIL", True)
    monkeypatch.setattr(fm, "batch_pair_mi_cuda_feature_major", boom)
    run = _dispatch()
    with caplog.at_level(logging.WARNING, logger=bpm.logger.name):
        assert run.try_feature_major() is None
    assert any(r.levelno == logging.WARNING and "illegal address" in r.getMessage() for r in caplog.records)


def test_row_chunked_failure_is_warned(monkeypatch, caplog):
    """try_row_chunked logs the fallback to CPU at WARNING and returns None."""

    def boom(*a, **k):
        """Raise a kernel fault."""
        raise RuntimeError("chunk fault")

    monkeypatch.setattr(bpm, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bpm, "batch_pair_mi_cuda_row_chunked", boom)
    with caplog.at_level(logging.WARNING, logger=bpm.logger.name):
        assert _dispatch().try_row_chunked("test") is None
    assert any(r.levelno == logging.WARNING and "chunk fault" in r.getMessage() for r in caplog.records)
