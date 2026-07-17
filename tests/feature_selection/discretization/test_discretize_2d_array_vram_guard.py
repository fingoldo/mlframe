"""Regression test for the VRAM safety gate in ``discretize_2d_array``'s CUDA dispatch.

Same bug class as the ``batch_pair_mi_gpu`` fix (2026-07-10 wellbore production run): the CUDA path
(``discretize_2d_array_cuda``) uploads the WHOLE input array unconditionally (``d_arr = cp.asarray(arr)``),
then ``cp.percentile`` needs a comparably-sized internal sort/partition scratch buffer on top -- at
production scale (millions of rows) this can consume a small card's entire VRAM. This dispatch site is
reached from ``categorize_dataset`` (MRMR.fit's discretization step), which per a real 2.4M-row production
trace runs BEFORE the pair-MI stage and left a trivial 4-byte allocation OOMing moments later -- strong
evidence VRAM was already near-exhausted by this exact upload.

Pins the fix: ``discretize_2d_array`` must consult ``_fe_gpu_vram.fe_gpu_has_vram_cushion`` before ever
calling ``discretize_2d_array_cuda``, and fall back to the CPU prange path when the upload would not
safely fit -- mirroring every other GPU-FE dispatch site's guard.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import mlframe.feature_selection.filters.discretization as disc_mod


def _big_enough_array():
    """Shape large enough that ``_DISCRETIZE_SPEC.choose`` would naturally pick "cuda" for real, so the
    test exercises the actual dispatch condition rather than a mocked bypass."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(size=(2000, 300)).astype(np.float32)  # 600_000 cells >= the 500k crossover


def test_cuda_path_skipped_when_vram_insufficient(monkeypatch, caplog):
    """When the full-upload VRAM guard rejects, the row-chunked GPU path is tried next (see
    ``test_discretize_2d_array_row_chunked.py``); this test isolates just the full-upload REJECT ->
    detailed-log behavior by also stubbing the row-chunked path to a fast no-op."""
    arr = _big_enough_array()

    monkeypatch.setattr(disc_mod._DISCRETIZE_SPEC, "choose", lambda **kw: "cuda")
    monkeypatch.setattr("pyutilz.core.pythonlib.is_cuda_available", lambda: True)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_vram.fe_gpu_has_vram_cushion", lambda bytes_needed: False)

    def _boom(*a, **kw):
        raise AssertionError("discretize_2d_array_cuda (full-upload) must NOT be invoked when the VRAM guard fails")

    def _fake_row_chunked(arr, n_bins, method, dtype):
        return np.zeros(arr.shape, dtype=dtype)

    monkeypatch.setattr(disc_mod, "discretize_2d_array_cuda", _boom)
    monkeypatch.setattr(disc_mod, "discretize_2d_array_cuda_row_chunked", _fake_row_chunked)

    with caplog.at_level(logging.WARNING):
        out = disc_mod.discretize_2d_array(arr, n_bins=10, method="quantile", dtype=np.int8)

    assert out.shape == arr.shape
    assert any("GPU upload REJECTED" in r.message for r in caplog.records)
    assert any("free=" in r.message and "total=" in r.message for r in caplog.records)


def test_cuda_path_used_when_vram_fits(monkeypatch):
    """Sanity check the guard is not a blanket disable: when VRAM fits, the cuda path still fires."""
    arr = _big_enough_array()

    monkeypatch.setattr(disc_mod._DISCRETIZE_SPEC, "choose", lambda **kw: "cuda")
    monkeypatch.setattr("pyutilz.core.pythonlib.is_cuda_available", lambda: True)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_vram.fe_gpu_has_vram_cushion", lambda bytes_needed: True)

    calls = {"n": 0}

    def _fake_cuda(arr, n_bins, method, dtype):
        calls["n"] += 1
        return np.zeros(arr.shape, dtype=dtype)

    monkeypatch.setattr(disc_mod, "discretize_2d_array_cuda", _fake_cuda)

    out = disc_mod.discretize_2d_array(arr, n_bins=10, method="quantile", dtype=np.int8)

    assert calls["n"] == 1
    assert out.shape == arr.shape


def test_cuda_path_untouched_when_below_size_threshold(monkeypatch):
    """A small array (below the kernel_tuning_cache size crossover) must never reach the VRAM guard or
    the CUDA path at all -- CPU prange handles it directly, matching pre-fix behavior for the common case."""
    rng = np.random.default_rng(1)
    small = rng.standard_normal(size=(50, 5)).astype(np.float32)

    def _boom_cuda(*a, **kw):
        raise AssertionError("discretize_2d_array_cuda must not fire for a small array")

    def _boom_cushion(*a, **kw):
        raise AssertionError("VRAM cushion probe must not fire when the size dispatcher already picked CPU")

    monkeypatch.setattr(disc_mod, "discretize_2d_array_cuda", _boom_cuda)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_vram.fe_gpu_has_vram_cushion", _boom_cushion)

    out = disc_mod.discretize_2d_array(small, n_bins=5, method="quantile", dtype=np.int8)
    assert out.shape == small.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
