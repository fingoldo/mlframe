"""GPU exception-handling + circuit-breaker regression tests (2026-07-09 fix).

Two independent GPU-reliability gaps closed together:

1. ``evaluate_candidate``'s ``use_gpu=True`` branch called ``mi_direct_gpu`` with NO exception
   handling -- unlike every sibling GPU dispatch point in this codebase. A single CUDA fault
   (driver hiccup, transient OOM, a context poisoned by an earlier unrelated call) crashed the
   WHOLE ``MRMR.fit()`` instead of degrading to CPU for that one candidate.
2. ``permutation.mi_direct``'s own internal GPU fastpath had no persistent circuit breaker: a
   context-poisoning fault (see ``info_theory._cmi_cuda``'s own breaker for the mechanism) meant
   every SUBSEQUENT call re-attempted the GPU and failed identically -- a retry storm.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import evaluation as evaluation_mod
from mlframe.feature_selection.filters import permutation as permutation_mod
from tests.feature_selection.contracts.test_evaluation import _build_xor_factors, _make_eval_kwargs


@pytest.fixture
def xor_factors():
    return _build_xor_factors()


# ================================================================================================
# evaluate_candidate: GPU exception -> CPU fallback, never crashes
# ================================================================================================


def test_evaluate_candidate_gpu_failure_falls_back_to_cpu_not_crash(xor_factors, monkeypatch):
    factors_data, factors_nbins, factors_names = xor_factors

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated cudaErrorLaunchFailure")

    monkeypatch.setattr(evaluation_mod, "mi_direct_gpu", _boom)

    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    kwargs["use_gpu"] = True
    # Must NOT raise -- this is the actual regression (pre-fix, this crashed the whole call).
    current_gain, sink_reasons = evaluate_candidate_safe_call(kwargs)
    assert isinstance(current_gain, float)
    assert isinstance(sink_reasons, set)


def test_evaluate_candidate_gpu_failure_result_matches_pure_cpu_path(xor_factors, monkeypatch):
    """The CPU fallback must compute the SAME relevance gain a pure ``use_gpu=False`` call would --
    it is not a degraded/approximate result, just a different backend for the identical computation."""
    factors_data, factors_nbins, factors_names = xor_factors

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated GPU fault")

    kwargs_gpu = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    kwargs_gpu["use_gpu"] = True
    monkeypatch.setattr(evaluation_mod, "mi_direct_gpu", _boom)
    gain_fallback, _ = evaluation_mod.evaluate_candidate(**kwargs_gpu)
    monkeypatch.undo()

    kwargs_cpu = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    kwargs_cpu["use_gpu"] = False
    gain_cpu, _ = evaluation_mod.evaluate_candidate(**kwargs_cpu)

    assert gain_fallback == pytest.approx(gain_cpu, abs=1e-9)


def evaluate_candidate_safe_call(kwargs):
    return evaluation_mod.evaluate_candidate(**kwargs)


def test_evaluate_candidate_gpu_success_path_unaffected(xor_factors):
    """Sanity: when the GPU call is NOT monkeypatched to fail, cupy-absent hosts still exercise the
    real dispatch (mi_direct_gpu internally falls back to CPU when cupy is unavailable) without our
    new try/except changing that pre-existing behavior."""
    factors_data, factors_nbins, factors_names = xor_factors
    kwargs = _make_eval_kwargs(factors_data, factors_nbins, factors_names)
    kwargs["use_gpu"] = True
    current_gain, _ = evaluation_mod.evaluate_candidate(**kwargs)
    assert current_gain >= 0.0


# ================================================================================================
# permutation.mi_direct: GPU circuit breaker
# ================================================================================================


@pytest.fixture(autouse=True)
def _reset_mi_direct_breaker():
    permutation_mod.reset_mi_direct_gpu_circuit_breaker()
    yield
    permutation_mod.reset_mi_direct_gpu_circuit_breaker()


def _build_binary_factors(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=n).astype(np.int32)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    factors_data = np.column_stack([x, y]).astype(np.int32)
    factors_nbins = np.array([2, 2], dtype=np.int64)
    return factors_data, factors_nbins


def test_mi_direct_gpu_fault_trips_breaker(monkeypatch):
    factors_data, factors_nbins = _build_binary_factors()

    monkeypatch.setattr(permutation_mod, "_MI_DIRECT_GPU_FAILED", False)

    def _fake_is_cuda_available():
        return True

    def _boom(**kwargs):
        raise RuntimeError("simulated CUDA launch failure")

    monkeypatch.setattr("pyutilz.core.pythonlib.is_cuda_available", _fake_is_cuda_available)
    # Patch the gpu module's mi_direct_gpu (imported lazily inside mi_direct).
    import mlframe.feature_selection.filters.gpu as gpu_mod
    monkeypatch.setattr(gpu_mod, "mi_direct_gpu", _boom)

    assert permutation_mod._MI_DIRECT_GPU_FAILED is False
    # npermutations>=32 + parallelism="outer" + return_null_mean=False are required to even reach the
    # GPU fastpath gate (see mi_direct's docstring).
    result = permutation_mod.mi_direct(
        factors_data, x=(0,), y=(1,), factors_nbins=factors_nbins,
        npermutations=32, parallelism="outer", prefer_gpu=True, return_null_mean=False,
    )
    assert isinstance(result, tuple)
    assert permutation_mod._MI_DIRECT_GPU_FAILED is True, "circuit breaker must trip on the first GPU fault"


def test_mi_direct_breaker_skips_gpu_probe_on_subsequent_calls(monkeypatch):
    """Once tripped, later calls must not even ATTEMPT the GPU path (no is_cuda_available probe)."""
    factors_data, factors_nbins = _build_binary_factors()
    permutation_mod._MI_DIRECT_GPU_FAILED = True

    probe_calls = []

    def _tracking_probe():
        probe_calls.append(1)
        return True

    monkeypatch.setattr("pyutilz.core.pythonlib.is_cuda_available", _tracking_probe)

    permutation_mod.mi_direct(
        factors_data, x=(0,), y=(1,), factors_nbins=factors_nbins,
        npermutations=32, parallelism="outer", prefer_gpu=True, return_null_mean=False,
    )
    assert probe_calls == [], "tripped breaker must short-circuit before even probing CUDA availability"


def test_mi_direct_breaker_reset_rearms_gpu_path():
    permutation_mod._MI_DIRECT_GPU_FAILED = True
    permutation_mod.reset_mi_direct_gpu_circuit_breaker()
    assert permutation_mod._MI_DIRECT_GPU_FAILED is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
