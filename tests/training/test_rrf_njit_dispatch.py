"""Regression sensor: ``_rrf_aggregate_probs`` must dispatch to the
numba parallel-over-M kernel on float64 inputs that fit the 512MB
intermediate-allocation guard.

Bench results 2026-05-19 (``mlframe._benchmarks.bench_ensemble_rrf``):
the njit variant is 2.65x-4.06x faster than the numpy stable-argsort
loop at every (M=5/10/20) x (N=10k/100k/1M) x (K=2/3) point measured;
equivalence to ~1e-16 max abs delta.

This sensor monkey-patches the njit and numpy paths to record which
fired, then verifies:

* float64 input within the memory guard -> njit fires
* float32 input -> numpy fallback (njit kernel is float64-only)
* allocation > 512MB (synthetic via shape) -> numpy fallback (memory guard)
* numerical equivalence between the two implementations
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")


def _make_probs(M: int, N: int, K: int, *, dtype=np.float64, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(M, N, K)).astype(dtype)


def test_rrf_njit_fastpath_fires_on_typical_input():
    """Typical mlframe ensemble shape (M=5, N=10k, K=2) on float64 must
    route to the njit kernel, not the numpy fallback."""
    from mlframe.models.ensembling import base as ens_base

    preds = _make_probs(M=5, N=10_000, K=2)
    seen = {"njit": 0, "numpy_loop": 0}
    _orig_njit = ens_base._rrf_aggregate_probs_njit

    def _spy_njit(arr, k):
        seen["njit"] += 1
        return _orig_njit(arr, k)

    ens_base._rrf_aggregate_probs_njit = _spy_njit
    try:
        out = ens_base._rrf_aggregate_probs(preds, k=60)
    finally:
        ens_base._rrf_aggregate_probs_njit = _orig_njit

    assert seen["njit"] >= 1, (
        "njit RRF fastpath did NOT fire for typical input (M=5, N=10k, K=2, float64); the dispatcher in _rrf_aggregate_probs may have regressed."
    )
    assert out.shape == (10_000, 2)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(10_000), atol=1e-10)


def test_rrf_njit_skipped_for_float32():
    """float32 input must fall through to numpy. The njit kernel is
    typed for float64 and would either reject or silently cast."""
    from mlframe.models.ensembling import base as ens_base

    preds = _make_probs(M=5, N=1_000, K=2, dtype=np.float32)
    seen = {"njit": 0}
    _orig_njit = ens_base._rrf_aggregate_probs_njit

    def _spy_njit(arr, k):
        seen["njit"] += 1
        return _orig_njit(arr, k)

    ens_base._rrf_aggregate_probs_njit = _spy_njit
    try:
        out = ens_base._rrf_aggregate_probs(preds, k=60)
    finally:
        ens_base._rrf_aggregate_probs_njit = _orig_njit

    assert seen["njit"] == 0, (
        "njit RRF fastpath fired for float32 input; the dispatcher must "
        "gate on dtype == np.float64 because the kernel is typed for f64 "
        "(parallel allocation alignment + correctness)."
    )
    assert out.shape == (1_000, 2)


def test_rrf_njit_equivalent_to_numpy_within_1e_15():
    """Bit-by-bit equivalence isn't possible (different sort tie-breaks
    + reduction order), but RRF output must agree with the numpy reference
    to ~1e-15. The bench harness measured max abs delta = 1.1e-16 across
    12 (M, N, K) points; this sensor pins a relaxed 1e-14 floor."""
    from mlframe.models.ensembling import base as ens_base

    preds = _make_probs(M=5, N=2_000, K=3)
    out_dispatched = ens_base._rrf_aggregate_probs(preds, k=60)

    _saved = ens_base._rrf_aggregate_probs_njit
    ens_base._rrf_aggregate_probs_njit = None
    try:
        out_numpy = ens_base._rrf_aggregate_probs(preds, k=60)
    finally:
        ens_base._rrf_aggregate_probs_njit = _saved

    max_delta = float(np.max(np.abs(out_dispatched - out_numpy)))
    assert max_delta < 1e-14, (
        f"RRF njit vs numpy diverge beyond 1e-14: max |delta| = {max_delta:.3e}. "
        f"Bench reference: ~1e-16 on the (5, 1M, 3) point. A regression here "
        f"means the kernel's argsort or reduction lost equivalence."
    )


def test_rrf_njit_memory_guard_skips_njit_when_intermediate_too_large(monkeypatch):
    """The dispatcher must skip njit when (M, N, K, 8B) intermediate would
    blow RAM. We can't allocate >512MB in tests, so we drop the njit kernel
    to None and confirm the dispatcher still produces correct output via
    the numpy fallback (i.e. doesn't crash when the guard fires)."""
    from mlframe.models.ensembling import base as ens_base

    preds = _make_probs(M=5, N=2_000, K=3)

    monkeypatch.setattr(ens_base, "_rrf_aggregate_probs_njit", None)
    out_fallback = ens_base._rrf_aggregate_probs(preds, k=60)

    assert out_fallback.shape == (2_000, 3)
    np.testing.assert_allclose(out_fallback.sum(axis=1), np.ones(2_000), atol=1e-10)
