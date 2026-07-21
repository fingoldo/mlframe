"""Regression: GPU FE must DEGRADE to CPU under memory pressure, never drop the engineered feature.

On a host that is itself out of RAM (paging), cupy cannot allocate the pinned H2D staging buffer and raises
``cudaErrorMemoryAllocation``. Pre-fix, ``polyeval_dispatch`` picked the CUDA backend at n>=500k and the raised error
propagated to ``generate_univariate_basis_features``, which caught it and SKIPPED the whole column -- silently losing the
feature. These tests pin the fallback: a failing CUDA polyeval returns the CPU result, and the resident builder bails
early (routing to the host/CPU path) when free VRAM is insufficient.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.feature_selection.filters.hermite_fe as hf


def test_polyeval_dispatch_falls_back_to_cpu_on_cuda_error(monkeypatch):
    """A CUDA polyeval that raises (OOM/driver) must fall back to the CPU backend and return the correct result, not
    propagate the error (which upstream turns into a dropped feature)."""
    x = np.random.default_rng(0).normal(size=600_000)
    c = np.zeros(3)
    c[2] = 1.0
    cpu_expected = hf._NJIT_PAR_FUNCS["hermite"](x, c)

    monkeypatch.setattr(hf, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(hf, "_warn_polyeval_cuda_fallback_once", lambda exc: None)

    def _boom(basis, xx, cc, device=None):
        """Always raises ``RuntimeError('cudaErrorMemoryAllocation: out of memory')``."""
        raise RuntimeError("cudaErrorMemoryAllocation: out of memory")

    monkeypatch.setattr(hf, "_polyeval_cuda", _boom)
    monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "cuda")  # force the CUDA branch so we hit _boom then fall back

    got = hf.polyeval_dispatch("hermite", x, c)
    assert np.allclose(got, cpu_expected), "polyeval must return the CPU result after the CUDA path fails"


def test_polyeval_default_never_uploads_to_gpu(monkeypatch):
    """Host-in/host-out contract: the DEFAULT (unforced) path computes on the CPU and never attempts a GPU upload, even
    at large n and with CUDA available -- the result lands on the host, so the H2D/D2H transfer is pure waste."""
    x = np.random.default_rng(1).normal(size=600_000)  # > the legacy n>=500k CUDA threshold
    c = np.zeros(3)
    c[2] = 1.0
    cpu_expected = hf._NJIT_PAR_FUNCS["hermite"](x, c)

    monkeypatch.setattr(hf, "_CUDA_AVAILABLE", True)

    def _should_not_run(basis, xx, cc, device=None):
        """Should not run."""
        raise AssertionError("GPU polyeval must NOT be attempted on the default host-in/host-out path")

    monkeypatch.setattr(hf, "_polyeval_cuda", _should_not_run)
    monkeypatch.delenv("MLFRAME_POLYEVAL_BACKEND", raising=False)

    got = hf.polyeval_dispatch("hermite", x, c)
    assert np.allclose(got, cpu_expected)


def test_resident_vram_gate_raises_when_matrix_exceeds_budget(monkeypatch):
    """The resident univariate builder's VRAM gate must raise (-> caller's host fallback) when the estimated device
    footprint exceeds the free-VRAM budget, instead of attempting the poisoning multi-GB pinned upload."""
    import mlframe.feature_selection.filters._orthogonal_univariate_fe as ofe

    # Patch cupy.cuda.runtime.memGetInfo to report a tiny free-VRAM so the estimate blows the budget.
    cp = pytest.importorskip("cupy")
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (256 * 2**20, 4 * 2**30))  # 256MB free
    with pytest.raises(RuntimeError, match="host/CPU FE path"):
        ofe._raise_if_vram_insufficient(5_000_000, 80)


def test_resident_vram_gate_noop_on_query_failure(monkeypatch):
    """A memGetInfo failure must NOT raise (the try/except OOM fallback is the backstop) -- assume OK."""
    import mlframe.feature_selection.filters._orthogonal_univariate_fe as ofe

    cp = pytest.importorskip("cupy")

    def _raise():
        """Always raises ``RuntimeError('no device')``."""
        raise RuntimeError("no device")

    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", _raise)
    ofe._raise_if_vram_insufficient(5_000_000, 80)  # must not raise
