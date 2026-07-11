"""Tests for the row-chunked GPU fallback in ``discretize_2d_array``'s CUDA dispatch.

Companion to ``test_discretize_2d_array_vram_guard.py`` (which pins the VRAM pre-flight check). Per
explicit user feedback, a VRAM-insufficient rejection should not just fall back to slow CPU when a
row-chunked GPU pass could still deliver most of the GPU speed win. Two methods, two different
correctness stories:

* ``uniform``: EXACT. Column min/max are genuinely reducible across row-chunks (running min/max), so the
  row-chunked result must be BIT-IDENTICAL to the full-upload CUDA kernel.
* ``quantile``: APPROXIMATE by construction (exact quantiles need the full column's order statistics,
  which isn't reducible across row-chunks). Bin edges come from a GPU-resident random subsample instead.
  Per the project's documented FE/MRMR exception, the bar here is closeness/selection-equivalence, not
  bit-identity -- validated via a high match-fraction + small max-code-drift assertion, not ``==``.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

import mlframe.feature_selection.filters.discretization as disc_mod

pytestmark = pytest.mark.skipif(
    not (getattr(disc_mod, "_DISCRETIZE_SPEC", None) is not None),
    reason="discretization module unavailable",
)


def _cuda_available() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return bool(is_cuda_available())
    except Exception:
        return False


def test_dispatch_tries_row_chunked_before_cpu_when_vram_insufficient(monkeypatch, caplog):
    """When the VRAM guard rejects the full upload, ``discretize_2d_array`` must try the row-chunked GPU
    path BEFORE giving up on the GPU -- not go straight to CPU prange."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(size=(2000, 300)).astype(np.float32)  # 600_000 cells >= the 500k crossover

    monkeypatch.setattr(disc_mod._DISCRETIZE_SPEC, "choose", lambda **kw: "cuda")
    monkeypatch.setattr("pyutilz.core.pythonlib.is_cuda_available", lambda: True)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_vram.fe_gpu_has_vram_cushion", lambda bytes_needed: False)

    def _boom_full(*a, **kw):
        raise AssertionError("discretize_2d_array_cuda (full-upload) must NOT be invoked when the VRAM guard fails")

    calls = {"row_chunked": 0}

    def _fake_row_chunked(arr, n_bins, method, dtype):
        calls["row_chunked"] += 1
        return np.zeros(arr.shape, dtype=dtype)

    monkeypatch.setattr(disc_mod, "discretize_2d_array_cuda", _boom_full)
    monkeypatch.setattr(disc_mod, "discretize_2d_array_cuda_row_chunked", _fake_row_chunked)

    with caplog.at_level(logging.WARNING):
        out = disc_mod.discretize_2d_array(arr, n_bins=10, method="quantile", dtype=np.int8)

    assert calls["row_chunked"] == 1
    assert out.shape == arr.shape
    assert any("GPU upload REJECTED" in r.message for r in caplog.records)
    assert any("free=" in r.message for r in caplog.records)


def test_dispatch_falls_back_to_cpu_when_row_chunked_also_fails(monkeypatch, caplog):
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(size=(2000, 300)).astype(np.float32)

    monkeypatch.setattr(disc_mod._DISCRETIZE_SPEC, "choose", lambda **kw: "cuda")
    monkeypatch.setattr("pyutilz.core.pythonlib.is_cuda_available", lambda: True)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_vram.fe_gpu_has_vram_cushion", lambda bytes_needed: False)

    def _boom(*a, **kw):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(disc_mod, "discretize_2d_array_cuda", _boom)
    monkeypatch.setattr(disc_mod, "discretize_2d_array_cuda_row_chunked", _boom)

    with caplog.at_level(logging.WARNING):
        out = disc_mod.discretize_2d_array(arr, n_bins=10, method="quantile", dtype=np.int8)

    assert out.shape == arr.shape
    assert any("row-chunked CUDA also failed" in r.message for r in caplog.records)


@pytest.mark.skipif(not _cuda_available(), reason="numba.cuda/cupy not available on this host")
class TestRealHardware:
    def test_uniform_row_chunked_is_bit_identical_to_full(self):
        rng = np.random.default_rng(3)
        arr = rng.standard_normal(size=(20_000, 6)).astype(np.float32)
        full = disc_mod.discretize_2d_array_cuda(arr, n_bins=10, method="uniform", dtype=np.int32)
        chunked = disc_mod.discretize_2d_array_cuda_row_chunked(arr, n_bins=10, method="uniform", dtype=np.int32)
        np.testing.assert_array_equal(full, chunked)

    def test_uniform_row_chunked_bit_identical_with_forced_tiny_chunks(self, monkeypatch):
        rng = np.random.default_rng(3)
        arr = rng.standard_normal(size=(20_000, 6)).astype(np.float32)
        full = disc_mod.discretize_2d_array_cuda(arr, n_bins=10, method="uniform", dtype=np.int32)
        monkeypatch.setattr(disc_mod, "_choose_discretize_row_chunk_rows", lambda *a, **kw: 777)
        chunked = disc_mod.discretize_2d_array_cuda_row_chunked(arr, n_bins=10, method="uniform", dtype=np.int32)
        np.testing.assert_array_equal(full, chunked, err_msg="uniform min/max reduction must stay exact across many tiny row-chunks")

    def test_quantile_row_chunked_closely_matches_full_subsample_approximation(self):
        """Approximate by design: assert a high match fraction + small max-drift, not bit-identity."""
        rng = np.random.default_rng(7)
        arr = rng.standard_normal(size=(50_000, 8)).astype(np.float32)
        full = disc_mod.discretize_2d_array_cuda(arr, n_bins=10, method="quantile", dtype=np.int32)
        chunked = disc_mod.discretize_2d_array_cuda_row_chunked(arr, n_bins=10, method="quantile", dtype=np.int32, quantile_subsample_rows=20_000)
        match_frac = float(np.mean(full == chunked))
        max_diff = int(np.max(np.abs(full.astype(int) - chunked.astype(int))))
        assert match_frac >= 0.90, f"subsample-based quantile edges diverged too much: match_frac={match_frac:.4f}"
        assert max_diff <= 1, f"quantile row-chunked drift must stay within one bin (boundary-adjacent only), got max_diff={max_diff}"

    def test_quantile_row_chunked_shape_correct_when_subsample_exceeds_n_rows(self):
        """quantile_subsample_rows > n_rows must use the whole array (no crash, no truncation)."""
        rng = np.random.default_rng(1)
        arr = rng.standard_normal(size=(500, 5)).astype(np.float32)
        out = disc_mod.discretize_2d_array_cuda_row_chunked(arr, n_bins=5, method="quantile", dtype=np.int32, quantile_subsample_rows=1_000_000)
        assert out.shape == arr.shape
        assert np.all(out >= 0) and np.all(out < 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
