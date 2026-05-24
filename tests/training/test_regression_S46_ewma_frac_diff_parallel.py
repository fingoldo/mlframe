"""Regression sensor S46 (Wave 4b): verify the EWMA / frac-diff-inverse
backend ladder for ``_ewma_kernel`` / ``_frac_diff_inverse_kernel``.

Three contracts:

1. Numerical identity vs the production v1 njit kernel at every routed
   size class. The parallel-batched njit kernel MUST produce bit-identical
   results to running v1 K times.
2. The size-based default dispatcher returns ``"njit"`` for K=1 and
   ``"njit_par"`` for K>=2 AND N>=50k (EWMA) / N>=10k (frac-diff-inv);
   ``MLFRAME_EWMA_BACKEND`` / ``MLFRAME_FRAC_DIFF_INV_BACKEND`` env-var
   force-overrides take precedence.
3. The dispatcher gracefully degrades to single-spec path when
   kernel_tuning_cache is unavailable (pyutilz / cupy missing).
"""
from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clean_ewma_env(monkeypatch):
    """Strip env-vars so dispatcher tests start from a known default state."""
    monkeypatch.delenv("MLFRAME_EWMA_BACKEND", raising=False)
    monkeypatch.delenv("MLFRAME_FRAC_DIFF_INV_BACKEND", raising=False)
    yield


# ---------- EWMA ----------

@pytest.mark.parametrize("K,N", [(1, 1_000), (3, 50_000), (10, 100_000)])
def test_ewma_batched_matches_per_spec_v1(K, N):
    """Batched-parallel EWMA == K applications of the single-spec v1 kernel."""
    from mlframe.training._composite_transforms_nonlinear import (
        _ewma_kernel, _ewma_kernel_njit_par_batched,
    )
    rng = np.random.default_rng(13)
    base_batch = rng.standard_normal((K, N)).astype(np.float64)
    alphas = rng.uniform(0.1, 0.4, size=K).astype(np.float64)
    anchors = rng.standard_normal(K).astype(np.float64)
    ref = np.empty((K, N), dtype=np.float64)
    for s in range(K):
        ref[s] = _ewma_kernel(
            np.ascontiguousarray(base_batch[s]), float(alphas[s]), float(anchors[s]),
        )
    got = _ewma_kernel_njit_par_batched(
        np.ascontiguousarray(base_batch), alphas, anchors,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("N", [100, 10_000, 100_000])
def test_ewma_compute_default_path_matches_v1(N):
    """Public ``_ewma_compute`` (1-D) matches direct v1 call. Routes through dispatcher; default selects ``njit`` for K=1."""
    from mlframe.training._composite_transforms_nonlinear import (
        _ewma_compute, _ewma_kernel,
    )
    rng = np.random.default_rng(0)
    base = rng.standard_normal(N).astype(np.float64)
    k = 7
    anchor = 0.5
    expected = _ewma_kernel(np.ascontiguousarray(base), 2.0 / (k + 1), anchor)
    got = _ewma_compute(base, k, anchor)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


def test_ewma_default_dispatcher_selects_njit_for_k1():
    """K=1 -> ``njit`` regardless of N (the parallel kernel has prange overhead that loses on a single spec). When KTC is unavailable the size-based fallback applies."""
    from mlframe.training._composite_transforms_nonlinear import _lookup_ewma_backend
    assert _lookup_ewma_backend(1, 100) == "njit"
    assert _lookup_ewma_backend(1, 10_000_000) == "njit"


def test_ewma_default_dispatcher_selects_njit_par_above_threshold():
    """K>=2 AND N>=50k -> ``njit_par`` per measured crossover."""
    from mlframe.training._composite_transforms_nonlinear import _lookup_ewma_backend
    assert _lookup_ewma_backend(10, 100_000) == "njit_par"
    assert _lookup_ewma_backend(2, 100_000) == "njit_par"
    # Below either threshold the par-batched cost dominates.
    assert _lookup_ewma_backend(2, 10_000) == "njit"
    assert _lookup_ewma_backend(1, 100_000) == "njit"


def test_ewma_env_force_override_takes_precedence(monkeypatch):
    """``MLFRAME_EWMA_BACKEND=njit_par`` forces the parallel-batched path even on K=1 N=100 (where the size-based default returns ``njit``)."""
    from mlframe.training._composite_transforms_nonlinear import _lookup_ewma_backend
    monkeypatch.setenv("MLFRAME_EWMA_BACKEND", "njit_par")
    assert _lookup_ewma_backend(1, 100) == "njit_par"
    monkeypatch.setenv("MLFRAME_EWMA_BACKEND", "njit")
    assert _lookup_ewma_backend(10, 10_000_000) == "njit"
    monkeypatch.setenv("MLFRAME_EWMA_BACKEND", "garbage")
    assert _lookup_ewma_backend(10, 100_000) == "njit_par"


@pytest.mark.parametrize("force_backend", ["njit", "njit_par"])
def test_ewma_compute_batched_force_backend_numerical_identity(monkeypatch, force_backend):
    """Forcing either backend yields the same result as the v1-per-spec reference."""
    from mlframe.training._composite_transforms_nonlinear import (
        _ewma_compute_batched, _ewma_kernel,
    )
    monkeypatch.setenv("MLFRAME_EWMA_BACKEND", force_backend)
    rng = np.random.default_rng(11)
    K, N = 6, 5_000
    base = rng.standard_normal((K, N)).astype(np.float64)
    ks = rng.integers(3, 15, size=K).astype(np.float64)
    anchors = rng.standard_normal(K).astype(np.float64)
    ref = np.empty((K, N), dtype=np.float64)
    for s in range(K):
        ref[s] = _ewma_kernel(
            np.ascontiguousarray(base[s]),
            2.0 / (float(ks[s]) + 1.0),
            float(anchors[s]),
        )
    got = _ewma_compute_batched(base, ks, anchors)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_ewma_compute_batched_raises_on_shape_mismatch():
    """K rows of ``base`` must match ks shape and anchors shape exactly."""
    from mlframe.training._composite_transforms_nonlinear import _ewma_compute_batched
    base = np.zeros((3, 100), dtype=np.float64)
    with pytest.raises(ValueError, match="must each equal"):
        _ewma_compute_batched(base, np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="must each equal"):
        _ewma_compute_batched(base, np.zeros(3), np.zeros(2))


# ---------- frac_diff_inverse ----------

@pytest.mark.parametrize("K,N", [(1, 1_000), (3, 10_000), (5, 50_000)])
def test_frac_diff_inv_batched_matches_per_spec_v1(K, N):
    """Batched-parallel frac-diff-inverse == K applications of the single-spec v1 kernel."""
    from mlframe.training._composite_transforms_nonlinear import (
        _frac_diff_inverse_kernel,
        _frac_diff_inverse_kernel_njit_par_batched,
        _frac_diff_weights,
    )
    rng = np.random.default_rng(7)
    lags = 30
    weights = _frac_diff_weights(0.5, lags)
    weights_batch = np.tile(weights, (K, 1))
    anchors = rng.standard_normal(K).astype(np.float64)
    t_batch = rng.standard_normal((K, N)).astype(np.float64)
    ref = np.empty((K, N), dtype=np.float64)
    for s in range(K):
        ref[s] = _frac_diff_inverse_kernel(
            np.ascontiguousarray(t_batch[s]), lags,
            np.ascontiguousarray(weights_batch[s]),
            float(anchors[s]),
        )
    got = _frac_diff_inverse_kernel_njit_par_batched(
        np.ascontiguousarray(t_batch), lags,
        np.ascontiguousarray(weights_batch), anchors,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_frac_diff_inv_default_dispatcher_thresholds():
    from mlframe.training._composite_transforms_nonlinear import _lookup_frac_diff_inv_backend
    assert _lookup_frac_diff_inv_backend(1, 100_000) == "njit"
    assert _lookup_frac_diff_inv_backend(5, 10_000) == "njit_par"
    assert _lookup_frac_diff_inv_backend(5, 1_000) == "njit"


def test_frac_diff_inv_env_force_override(monkeypatch):
    from mlframe.training._composite_transforms_nonlinear import _lookup_frac_diff_inv_backend
    monkeypatch.setenv("MLFRAME_FRAC_DIFF_INV_BACKEND", "njit_par")
    assert _lookup_frac_diff_inv_backend(1, 1) == "njit_par"
    monkeypatch.setenv("MLFRAME_FRAC_DIFF_INV_BACKEND", "njit")
    assert _lookup_frac_diff_inv_backend(1000, 1_000_000) == "njit"


def test_frac_diff_inv_force_par_numerical_identity(monkeypatch):
    """Forcing ``njit_par`` on a K=1 input produces the same result as the scalar v1 kernel (the par-batched kernel runs with K=1 and unwraps the single row)."""
    from mlframe.training._composite_transforms_nonlinear import (
        _frac_diff_inverse_compute, _frac_diff_inverse_kernel, _frac_diff_weights,
    )
    monkeypatch.setenv("MLFRAME_FRAC_DIFF_INV_BACKEND", "njit_par")
    rng = np.random.default_rng(3)
    lags = 30
    weights = _frac_diff_weights(0.5, lags)
    t = rng.standard_normal(2_000).astype(np.float64)
    anchor = 0.25
    got = _frac_diff_inverse_compute(t, lags, weights, anchor)
    ref = _frac_diff_inverse_kernel(np.ascontiguousarray(t), lags, weights, anchor)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_frac_diff_inv_compute_batched_force_backend_identity(monkeypatch):
    from mlframe.training._composite_transforms_nonlinear import (
        _frac_diff_inverse_compute_batched, _frac_diff_inverse_kernel, _frac_diff_weights,
    )
    monkeypatch.setenv("MLFRAME_FRAC_DIFF_INV_BACKEND", "njit_par")
    rng = np.random.default_rng(21)
    K, N, lags = 4, 5_000, 30
    weights = _frac_diff_weights(0.5, lags)
    weights_batch = np.tile(weights, (K, 1))
    anchors = rng.standard_normal(K).astype(np.float64)
    t_batch = rng.standard_normal((K, N)).astype(np.float64)
    ref = np.empty((K, N), dtype=np.float64)
    for s in range(K):
        ref[s] = _frac_diff_inverse_kernel(
            np.ascontiguousarray(t_batch[s]), lags,
            np.ascontiguousarray(weights_batch[s]),
            float(anchors[s]),
        )
    got = _frac_diff_inverse_compute_batched(t_batch, lags, weights_batch, anchors)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_kernel_tuning_cache_lookup_does_not_crash_on_missing_pyutilz():
    """Even if pyutilz / KernelTuningCache is unavailable, the dispatcher MUST fall back to the size-based default rather than raising. Verified by snapshotting sys.modules around a force-None of the import path so the test is xdist-safe (no global state escapes)."""
    import sys as _sys
    from mlframe.training import _composite_transforms_nonlinear as M
    _key = "mlframe.feature_selection.filters._kernel_tuning"
    _orig_snapshot = dict(_sys.modules)
    try:
        _sys.modules[_key] = None
        bc = M._lookup_ewma_backend(10, 100_000)
        assert bc == "njit_par"
    finally:
        # Full snapshot/restore protects against cross-test pollution per CLAUDE.md Test-pollution rule.
        _sys.modules.clear()
        _sys.modules.update(_orig_snapshot)


# ---------- End-to-end via ewma_residual / frac_diff registry ----------

def test_ewma_residual_forward_inverse_roundtrip_unchanged():
    """Production ``ewma_residual`` transform pair still round-trips (forward then inverse recovers y) after dispatcher wiring."""
    from mlframe.training.composite_transforms import get_transform
    rng = np.random.default_rng(99)
    t = get_transform("ewma_residual")
    n = 5_000
    base = np.cumsum(rng.standard_normal(n)) / 10.0
    y = base + 0.1 * rng.standard_normal(n)
    params = t.fit(y, base)
    T = t.forward(y, base, params)
    y_back = t.inverse(T, base, params)
    np.testing.assert_allclose(y_back, y, rtol=1e-10, atol=1e-10)


def test_frac_diff_forward_inverse_roundtrip_unchanged():
    from mlframe.training.composite_transforms import get_transform
    rng = np.random.default_rng(100)
    t = get_transform("frac_diff")
    n = 1_000
    y = np.cumsum(rng.standard_normal(n)) / 5.0
    params = t.fit(y, np.zeros(n))
    T = t.forward(y, np.zeros(n), params)
    y_back = t.inverse(T, np.zeros(n), params)
    np.testing.assert_allclose(y_back, y, rtol=1e-10, atol=1e-10)
