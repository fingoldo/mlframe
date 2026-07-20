"""Regression test for mrmr_audit_2026-07-20 B-10: ``ksg_mi_dispatch``'s GPU path had no circuit breaker.

Pre-fix, the only guard around the ``mixed_ksg_mi_gpu`` call was ``except ImportError: pass`` -- a real
runtime/driver fault (CUDA OOM, context corruption) raised something OTHER than ImportError and propagated
straight out of ``ksg_mi_dispatch`` instead of falling back to CPU, and nothing prevented every subsequent
call from re-attempting the same doomed launch (unlike every other GPU dispatch site in this package,
e.g. ``info_theory/_cmi_cuda.py``, ``permutation.py``'s ``mi_direct``, ``_permutation_null_pair_resident.py``).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_breaker():
    """Reset the module-level circuit breaker before and after every test so tests don't leak state."""
    from mlframe.feature_selection.filters._ksg import reset_ksg_gpu_circuit_breaker

    reset_ksg_gpu_circuit_breaker()
    yield
    reset_ksg_gpu_circuit_breaker()


def test_gpu_runtime_fault_falls_back_to_cpu_not_raises(monkeypatch):
    """A non-ImportError exception from the GPU path (e.g. a CUDA OOM) must be caught and the call must
    fall back to CPU mixed_ksg_mi -- pre-fix this propagated out of ksg_mi_dispatch uncaught."""
    import mlframe.feature_selection.filters._ksg as ksg_mod

    monkeypatch.setattr(ksg_mod, "_KSG_GPU_THRESHOLD", 10)  # force the >= threshold branch on a small fixture

    def _boom(*args, **kwargs):
        """Simulate a real GPU runtime fault (not ImportError)."""
        raise RuntimeError("simulated CUDA driver fault")

    monkeypatch.setattr(ksg_mod, "mixed_ksg_mi_gpu", _boom)

    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)

    result = ksg_mod.ksg_mi_dispatch(x, y, prefer_gpu=True)
    assert np.isfinite(result), f"expected a finite CPU-computed MI after GPU fallback, got {result!r}"


def test_gpu_fault_trips_breaker_so_next_call_skips_gpu_path(monkeypatch):
    """After one GPU-path fault, the circuit breaker must be tripped so a SECOND call does not even
    attempt the GPU path (pre-fix: no breaker existed, so every call re-attempted the doomed launch)."""
    import mlframe.feature_selection.filters._ksg as ksg_mod

    monkeypatch.setattr(ksg_mod, "_KSG_GPU_THRESHOLD", 10)

    call_count = {"n": 0}

    def _boom(*args, **kwargs):
        """Simulate a real GPU runtime fault (not ImportError), counting calls."""
        call_count["n"] += 1
        raise RuntimeError("simulated CUDA driver fault")

    monkeypatch.setattr(ksg_mod, "mixed_ksg_mi_gpu", _boom)

    rng = np.random.default_rng(1)
    n = 50
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)

    ksg_mod.ksg_mi_dispatch(x, y, prefer_gpu=True)
    assert call_count["n"] == 1
    assert ksg_mod._KSG_GPU_FAILED is True, "circuit breaker must be tripped after a GPU runtime fault"

    ksg_mod.ksg_mi_dispatch(x, y, prefer_gpu=True)
    assert call_count["n"] == 1, "a tripped circuit breaker must prevent a second GPU launch attempt entirely"


def test_import_error_does_not_trip_breaker(monkeypatch):
    """Cupy genuinely absent (ImportError) is the expected 'no GPU here' case -- it must NOT trip the
    breaker (which is reserved for real runtime faults) and must still fall back to CPU cleanly."""
    import mlframe.feature_selection.filters._ksg as ksg_mod

    monkeypatch.setattr(ksg_mod, "_KSG_GPU_THRESHOLD", 10)

    def _no_cupy(*args, **kwargs):
        """Simulate cupy genuinely absent (the expected no-GPU-here case)."""
        raise ImportError("no module named cupy")

    monkeypatch.setattr(ksg_mod, "mixed_ksg_mi_gpu", _no_cupy)

    rng = np.random.default_rng(2)
    n = 50
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)

    result = ksg_mod.ksg_mi_dispatch(x, y, prefer_gpu=True)
    assert np.isfinite(result)
    assert ksg_mod._KSG_GPU_FAILED is False, "an ImportError (cupy absent) must not trip the runtime-fault circuit breaker"
