"""The resident noise gate hands the fit to the CPU gate when its (P, total_size) counts buffer does not fit free VRAM.

Before the fix the buffer was allocated unconditionally. On a 4 GB card a counts buffer of several GB oversubscribed VRAM
and the kernels stalled for minutes. The gate now returns None when the VRAM cushion check fails, and the caller falls back.
"""

import logging

import numpy as np
import pytest

import mlframe.feature_selection.filters.batch_mi_noise_gate_gpu as bg
from mlframe.feature_selection.filters.info_theory import merge_vars

pytestmark = pytest.mark.skipif(not bg._CUDA_AVAIL, reason="numba.cuda not available on this host")


def _inputs(n=3000, K=45, nbins=8, ky=3, seed=3):
    """A small resident-gate workload: K columns of nbins bins against a ky-class target."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, ky, size=n).astype(np.int32)
    cols = rng.integers(0, nbins, size=(n, K)).astype(np.int32)
    cy, fy, _ = merge_vars(factors_data=y.reshape(-1, 1), vars_indices=np.array([0], dtype=np.int64), var_is_nominal=None, factors_nbins=np.array([ky], dtype=np.int64), dtype=np.int32)
    nb = np.full(K, nbins, dtype=np.int64)
    return dict(disc_2d=cols, factors_nbins=nb, classes_y=cy, classes_y_safe=cy.copy(), freqs_y=fy, npermutations=10, base_seed=np.uint64(0), min_nonzero_confidence=0.99, use_su=False, dtype=np.int32)


def test_resident_gate_returns_none_when_counts_buffer_does_not_fit_vram(monkeypatch, caplog):
    """With no VRAM cushion the resident gate returns None (CPU fallback) and logs why, instead of allocating the counts buffer."""
    from mlframe.feature_selection.filters import _fe_gpu_vram

    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *_a, **_k: False)
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSED_GATE", "0")
    kwargs = _inputs()
    kwargs["use_su"] = True  # the MI bit-set gate and the fused gate both skip the counts buffer; the SU gate reaches it
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.batch_mi_noise_gate_gpu"):
        got = bg.batch_mi_with_noise_gate_cuda_resident(**kwargs)
    assert got is None
    assert any("does not fit the free VRAM cushion" in r.getMessage() for r in caplog.records)


def test_resident_gate_still_runs_on_device_when_the_counts_buffer_fits(monkeypatch):
    """With the cushion present the resident gate runs on the device and returns the (K,) gated vector."""
    from mlframe.feature_selection.filters import _fe_gpu_vram

    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *_a, **_k: True)
    monkeypatch.setenv("MLFRAME_FE_GPU_FUSED_GATE", "0")
    kwargs = _inputs()
    kwargs["use_su"] = True
    got = bg.batch_mi_with_noise_gate_cuda_resident(**kwargs)
    assert got is not None
    assert np.asarray(got).shape[0] == 45


def test_column_tiled_gate_matches_the_untiled_gate(monkeypatch):
    """When the full counts buffer does not fit, the columns are gated in tiles and the result equals the untiled gate (per-column decision)."""
    from mlframe.feature_selection.filters import _fe_gpu_vram

    kwargs = _inputs()
    kwargs["use_su"] = True
    K_y = int(kwargs["freqs_y"].shape[0])
    total = int(np.sum(kwargs["factors_nbins"])) * K_y
    full_bytes = (kwargs["npermutations"] + 1) * total * 8
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *_a, **_k: True)
    untiled = bg.batch_mi_with_noise_gate_cuda_resident(**kwargs)
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda b, **_k: b <= full_bytes // 2)
    tiled = bg.batch_mi_with_noise_gate_cuda_resident(**kwargs)
    assert tiled is not None
    assert np.allclose(np.asarray(tiled), np.asarray(untiled), atol=1e-12, rtol=1e-12, equal_nan=True)
