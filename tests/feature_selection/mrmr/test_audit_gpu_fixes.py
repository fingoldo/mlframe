"""Regression tests for the GPU-resident / GPU-infra findings of the mrmr audit fix wave.

CPU-testable via monkeypatch (fake cupy / module globals) -- none require a real CUDA device. Pins that the
order-2 pair-maxT resident floor trips its circuit breaker on a device fault (instead of silently swallowing
it so the breaker never arms), that the extval materialiser rejects a too-narrow code dtype up front, and
that the batch-pair usability-corr dispatcher honors the GPU opt-out even over an explicit force_backend.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest


class _BoomCupy:
    """Fake cupy whose every attribute access raises, simulating a poisoned CUDA context."""

    def __getattr__(self, _name):
        raise RuntimeError("simulated cudaErrorLaunchFailure")


def test_pair_maxt_resident_fault_trips_circuit_breaker(monkeypatch):
    """A device fault inside the resident order-2 pair-maxT floor must trip the process circuit breaker (so later
    FE steps skip the poisoned GPU) and return None, not silently swallow the fault leaving the breaker unarmed."""
    import mlframe.feature_selection.filters._permutation_null_pair_resident as m

    m.reset_pair_maxt_gpu_circuit_breaker()
    monkeypatch.setitem(sys.modules, "cupy", _BoomCupy())
    n = 16
    out = m.pooled_pair_permutation_null_joint_mi_floor_cupy(
        factors_data=np.zeros((n, 4), dtype=np.int32),
        pair_a=np.array([0, 1], dtype=np.int64),
        pair_b=np.array([2, 3], dtype=np.int64),
        nbins=np.full(4, 4, dtype=np.int64),
        classes_y=np.zeros(n, dtype=np.int64),
        freqs_y=np.array([0.5, 0.5], dtype=np.float64),
        n_permutations=8,
        quantile=0.95,
    )
    assert out is None
    assert m._PAIR_MAXT_GPU_FAILED is True, "device fault must arm the circuit breaker"
    m.reset_pair_maxt_gpu_circuit_breaker()


def test_extval_codes_host_rejects_narrow_dtype():
    """gpu_materialise_extval_codes_host must raise (not silently return None) when the requested code dtype cannot
    represent nbins-1 -- the int8@nbins>128 wrap class. The guard runs before any cupy use, so it is CPU-testable."""
    from mlframe.feature_selection.filters._gpu_resident_extval import gpu_materialise_extval_codes_host

    with pytest.raises(ValueError, match="cannot represent codes"):
        gpu_materialise_extval_codes_host(
            np.zeros(16, dtype=np.float64), [np.zeros(16, dtype=np.float64)], np.array([0], dtype=np.int64),
            nbins=200, dtype=np.int8,
        )


def test_usability_corr_dispatch_optout_beats_force_backend(monkeypatch):
    """dispatch_batch_pair_usability_corr must honor MLFRAME_DISABLE_GPU even when force_backend requests a GPU
    kernel on a CUDA-capable host -- the opt-out wins over an explicit force."""
    import mlframe.feature_selection.filters.batch_pair_usability_corr_gpu as bpu

    monkeypatch.setattr(bpu, "_CUDA_AVAIL", True, raising=False)
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    monkeypatch.setattr(bpu, "batch_pair_usability_corr_cuda_warp", lambda *a, **k: pytest.fail("GPU kernel entered despite opt-out"))
    monkeypatch.setattr(bpu, "batch_pair_usability_corr_cuda", lambda *a, **k: pytest.fail("GPU kernel entered despite opt-out"))

    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(size=n).astype(np.float64)
    operand_matrix = rng.normal(size=(4, n)).astype(np.float64)
    pair_a = np.array([0, 1], dtype=np.int64)
    pair_b = np.array([2, 3], dtype=np.int64)
    _result, backend = bpu.dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b, force_backend="cuda_warp")
    assert backend not in ("cuda", "cuda_warp"), f"opt-out must beat force_backend; got backend={backend}"


def test_noise_gate_fallback_choice_crossover(monkeypatch):
    """_batch_mi_noise_gate_fallback_choice must gate GPU on large n AND large K, prefer cupy over cuda, and fall
    back to cpu when no GPU backend is available -- the pre-sweep heuristic covering the previously untested fn."""
    import mlframe.feature_selection.filters._batch_mi_noise_gate_tuning as t

    monkeypatch.setattr(t, "_CUPY_AVAIL", True)
    monkeypatch.setattr(t, "_CUDA_AVAIL", True)
    assert t._batch_mi_noise_gate_fallback_choice(500, 64) == "cpu"  # below both thresholds
    assert t._batch_mi_noise_gate_fallback_choice(5000, 512) == "cupy"  # large n AND K -> cupy preferred
    monkeypatch.setattr(t, "_CUPY_AVAIL", False)
    assert t._batch_mi_noise_gate_fallback_choice(5000, 512) == "cuda"  # cupy absent -> cuda
    monkeypatch.setattr(t, "_CUDA_AVAIL", False)
    assert t._batch_mi_noise_gate_fallback_choice(5000, 512) == "cpu"  # no GPU -> cpu
