"""The numba.cuda pair-MI entry points reject codes at or above their column's cardinality before any device work, and the CUDA probe resolves without pyutilz's probe."""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.feature_selection.filters._batch_pair_mi_cuda_kernels as k


def _inputs(bad_code: int):
    """Three-column byte codes with column 0 holding ``bad_code`` in one row, nbins 3, two classes."""
    fd = np.zeros((6, 3), dtype=np.int32)
    fd[:, 1] = [0, 1, 2, 0, 1, 2]
    fd[2, 0] = bad_code
    nbins = np.array([3, 3, 3], dtype=np.int32)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    freqs = np.full(2, 0.5)
    return fd, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), nbins, y, freqs


def test_batch_pair_mi_cuda_rejects_code_above_column_cardinality(monkeypatch):
    """A code >= nbins[col] would index outside the shared histogram, so it raises ValueError on the host."""
    monkeypatch.setattr(k, "_CUDA_AVAIL", True)
    monkeypatch.setattr(k, "_CUDA_KERNEL", object())
    with pytest.raises(ValueError, match="outside"):
        k.batch_pair_mi_cuda(*_inputs(5))


def test_row_chunked_rejects_code_above_column_cardinality(monkeypatch):
    """The row-chunked accumulator would write into another pair's slice for an over-range code, so it raises ValueError on the host."""
    monkeypatch.setattr(k, "_CUDA_AVAIL", True)
    monkeypatch.setattr(k, "_CUDA_HIST_KERNEL", object())
    with pytest.raises(ValueError, match="outside"):
        k.batch_pair_mi_cuda_row_chunked(*_inputs(5))


def test_resolve_cuda_available_falls_back_when_pyutilz_probe_is_missing(monkeypatch):
    """A pyutilz without ``is_cuda_available`` falls back to numba's probe instead of leaving the flag undefined."""
    import pyutilz.core.pythonlib as pl

    class _Cuda:
        """Stand-in numba.cuda reporting a device."""

        @staticmethod
        def is_available() -> bool:
            """Report a device."""
            return True

    monkeypatch.delattr(pl, "is_cuda_available", raising=False)
    monkeypatch.setattr(k, "_nb_cuda", _Cuda)
    assert k._resolve_cuda_available() is True
