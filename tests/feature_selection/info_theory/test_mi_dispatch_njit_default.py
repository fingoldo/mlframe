"""MI-dispatch policy regression (2026-06-19).

Pins the ground-truth decision that the batched / single plug-in FE-MI dispatch
defaults to the CPU njit backend, NOT cuda, on this many-small-call multi-thread path.

Background: the per-call kernel-tuner microbenchmark over-rated the GPU because it
measured MI in isolation with a warm GPU buffer -- it never saw the production reality
where the FE pair-search / orth-FE / conditional-gate scans fire many small MI calls from
joblib worker threads contending on ONE GPU, each paying a fresh H2D/D2H + cudaMalloc +
sync (~700ms/call measured; 82 calls = 61s of cupy argsort). End-to-end A/B on the
canonical 5-feature/n=100k fit: GPU 318-368s / 5.0 GB vs njit 114s / 1.6 GB, byte-identical
selection. So the dispatch trusts the end-to-end measurement and defaults to njit; the
sweep was upgraded to measure contention (it exposes the 5-7x penalty) and an explicit
``MLFRAME_MI_BACKEND=cuda`` still forces the GPU for a caller that benefits.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_fallback_mi_backend_is_njit_for_all_shapes():
    """The no-measurement fallback must be the conservative njit, at every (n, k)."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import (
        _fallback_mi_backend,
    )

    for n in (1_000, 10_000, 75_000, 100_000, 1_000_000):
        for k in (1, 5, 20, 200):
            assert _fallback_mi_backend(n, k) == "njit", (n, k)


def test_batch_dispatch_default_matches_forced_njit(monkeypatch):
    """With no override, the batch dispatch returns the njit result bit-for-bit
    (i.e. it routes to njit by default rather than cuda)."""
    import mlframe.feature_selection.filters.hermite_fe  # noqa: F401 -- full-init parent first (the direct

    # ``import _hermite_fe_mi`` cycle: hermite_fe/__init__ imports _ensure_cuda_kernels back from it).
    from mlframe.feature_selection.filters import _hermite_fe_mi as H

    monkeypatch.delenv("MLFRAME_MI_BACKEND", raising=False)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 6))
    y = rng.integers(0, 3, size=4000).astype(np.int64)

    default = H.plugin_mi_classif_batch_dispatch(X, y, 20)
    monkeypatch.setenv("MLFRAME_MI_BACKEND", "njit")
    forced_njit = H.plugin_mi_classif_batch_dispatch(X, y, 20)
    np.testing.assert_array_equal(default, forced_njit)


def test_single_dispatch_default_matches_forced_njit(monkeypatch):
    """Single dispatch default matches forced njit."""
    import mlframe.feature_selection.filters.hermite_fe  # noqa: F401 -- full-init parent first (the direct

    # ``import _hermite_fe_mi`` cycle: hermite_fe/__init__ imports _ensure_cuda_kernels back from it).
    from mlframe.feature_selection.filters import _hermite_fe_mi as H

    monkeypatch.delenv("MLFRAME_MI_BACKEND", raising=False)
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = rng.integers(0, 4, size=5000).astype(np.int64)

    default = H.plugin_mi_classif_dispatch(x, y, 20)
    monkeypatch.setenv("MLFRAME_MI_BACKEND", "njit")
    forced_njit = H.plugin_mi_classif_dispatch(x, y, 20)
    assert default == pytest.approx(forced_njit, abs=1e-12)


def test_sweep_records_concurrency_dimension():
    """The tuner sweep must expose the concurrency it measured under, so the contention
    penalty it now models is auditable (and a future GPU-resident path can re-tune)."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import _auto_tune_sweeps_a as S

    assert S._physical_concurrency() >= 1
    # The per-call timer degrades to the solo loop at concurrency=1 and returns a finite ms.
    rng = np.random.default_rng(2)
    x = rng.normal(size=2000)
    y = rng.integers(0, 3, size=2000).astype(np.int64)
    from mlframe.feature_selection.filters.hermite_fe import _plugin_mi_classif_njit

    t = S._median_per_call(_plugin_mi_classif_njit, (x, y, 20), concurrency=2, n_iters=2)
    assert np.isfinite(t) and t >= 0.0
