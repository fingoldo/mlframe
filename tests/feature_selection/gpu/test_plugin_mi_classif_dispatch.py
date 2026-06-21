"""Equivalence + dispatcher tests for the plug-in MI estimator.

The njit (CPU) and cupy (GPU) implementations must produce numerically
identical results up to fp64 round-off because they both compute the
same plug-in MI formula on the same quantile-binned columns. Bit-for-bit
matching is the bar -- any larger drift indicates a binning or scatter
bug, not a tolerable numerical difference.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import (
    _CUDA_AVAILABLE,
    _plugin_mi_classif_batch_njit,
    _plugin_mi_classif_njit,
    plugin_mi_classif_batch_dispatch,
    plugin_mi_classif_dispatch,
)

cupy = pytest.importorskip("cupy") if _CUDA_AVAILABLE else None
if not _CUDA_AVAILABLE:
    pytest.skip("cupy not available", allow_module_level=True)

# Re-import the cuda-only functions after the importorskip gate.
from mlframe.feature_selection.filters.hermite_fe import (  # noqa: E402
    _plugin_mi_classif_batch_cuda,
    _plugin_mi_classif_cuda,
)


class TestPluginMIClassifEquivalence:
    """cupy batch MI matches njit batch MI bit-for-bit on the same inputs."""

    @pytest.mark.parametrize("n", [1_000, 50_000, 100_000])
    @pytest.mark.parametrize("k", [1, 5, 20])
    @pytest.mark.parametrize("n_classes", [2, 3, 5])
    def test_batch_cuda_matches_njit(
        self, n: int, k: int, n_classes: int,
    ) -> None:
        rng = np.random.default_rng(seed=11 + n + k + n_classes)
        X = rng.normal(size=(n, k))
        y = rng.integers(0, n_classes, size=n).astype(np.int64)
        njit_arr = _plugin_mi_classif_batch_njit(X, y, 20)
        cuda_arr = _plugin_mi_classif_batch_cuda(X, y, 20)
        np.testing.assert_allclose(
            cuda_arr, njit_arr, atol=1e-12,
            err_msg=(
                f"batch CUDA MI diverged from njit at n={n}, k={k}, "
                f"n_classes={n_classes}: cuda={cuda_arr}, njit={njit_arr}"
            ),
        )

    @pytest.mark.parametrize("n", [1_000, 50_000, 100_000])
    def test_single_cuda_matches_njit(self, n: int) -> None:
        rng = np.random.default_rng(seed=11 + n)
        x = rng.normal(size=n)
        y = rng.integers(0, 4, size=n).astype(np.int64)
        njit_mi = float(_plugin_mi_classif_njit(x, y, 20))
        cuda_mi = _plugin_mi_classif_cuda(x, y, 20)
        assert abs(cuda_mi - njit_mi) < 1e-12, (
            f"single-col CUDA MI diverged from njit at n={n}: "
            f"cuda={cuda_mi}, njit={njit_mi}"
        )


class TestPluginMIClassifDispatcher:
    """Dispatcher routes to the right backend based on n and env override."""

    def test_dispatcher_routes_to_njit_below_threshold(self) -> None:
        rng = np.random.default_rng(11)
        # n=5_000, k=10: under the kernel_tuning_cache fallback this is
        # below the batch (k>1) crossover (~10k on cc 6.1) so the
        # dispatcher must route to njit. Per-host KTC may refine this
        # boundary but the worst-case fallback still keeps n=5k on njit.
        n = 5_000
        X = rng.normal(size=(n, 10))
        y = rng.integers(0, 3, size=n).astype(np.int64)
        out = plugin_mi_classif_batch_dispatch(X, y, 20)
        expected = _plugin_mi_classif_batch_njit(X, y, 20)
        np.testing.assert_array_equal(out, expected)

    def test_dispatcher_env_override_cuda(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MLFRAME_MI_BACKEND", "cuda")
        rng = np.random.default_rng(11)
        n = 5_000
        X = rng.normal(size=(n, 8))
        y = rng.integers(0, 3, size=n).astype(np.int64)
        # The dispatcher should now route to CUDA even though n < threshold.
        # We can't observe the routing directly, but the result must match
        # _plugin_mi_classif_batch_cuda exactly (bit-for-bit since same input).
        out = plugin_mi_classif_batch_dispatch(X, y, 20)
        expected = _plugin_mi_classif_batch_cuda(X, y, 20)
        np.testing.assert_array_equal(out, expected)

    def test_dispatcher_env_override_njit(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MLFRAME_MI_BACKEND", "njit")
        rng = np.random.default_rng(11)
        n = 500_000  # well ABOVE the batch CUDA threshold
        X = rng.normal(size=(n, 5))
        y = rng.integers(0, 3, size=n).astype(np.int64)
        out = plugin_mi_classif_batch_dispatch(X, y, 20)
        expected = _plugin_mi_classif_batch_njit(X, y, 20)
        np.testing.assert_array_equal(out, expected)


class TestPluginMIDispatcherConsultsKTC:
    """The 2026-05-20 fix routes plugin_mi_classif_(batch_)dispatch
    through ``kernel_tuning_cache.dispatch.lookup_mi_classif_backend``
    instead of hardcoded thresholds. A silent regression to hardcoded
    constants would defeat the per-host calibration win (2-4x at
    production sizes on most HW).

    These tests monkey-patch ``lookup_mi_classif_backend`` and assert
    the dispatcher actually CALLS it (with the right (n, k) shape) on
    every dispatcher invocation that has cupy available.
    """

    def test_batch_dispatcher_consults_lookup(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache import (
            dispatch as _ktc_dispatch,
        )

        rng = np.random.default_rng(11)
        n, k = 50_000, 8
        X = rng.normal(size=(n, k))
        y = rng.integers(0, 3, size=n).astype(np.int64)

        calls: list[tuple[int, int]] = []
        original = _ktc_dispatch.lookup_mi_classif_backend

        def _spy(n_samples, k_arg, **kwargs):
            calls.append((n_samples, k_arg))
            return original(n_samples, k_arg, **kwargs)

        monkeypatch.setattr(
            _ktc_dispatch, "lookup_mi_classif_backend", _spy,
        )
        # Drop any cached MLFRAME_MI_BACKEND override so the dispatcher
        # actually walks the cache path (env-var force-override would
        # short-circuit the lookup).
        monkeypatch.delenv("MLFRAME_MI_BACKEND", raising=False)

        plugin_mi_classif_batch_dispatch(X, y, 20)
        assert calls, (
            "plugin_mi_classif_batch_dispatch did NOT consult "
            "lookup_mi_classif_backend; the KTC integration regressed "
            "to hardcoded thresholds."
        )
        assert calls[-1] == (n, k), (
            f"KTC lookup called with wrong shape: got {calls[-1]}, "
            f"expected ({n}, {k})"
        )

    def test_single_dispatcher_consults_lookup(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache import (
            dispatch as _ktc_dispatch,
        )

        rng = np.random.default_rng(11)
        n = 100_000
        x = rng.normal(size=n)
        y = rng.integers(0, 3, size=n).astype(np.int64)

        calls: list[tuple[int, int]] = []
        original = _ktc_dispatch.lookup_mi_classif_backend

        def _spy(n_samples, k_arg, **kwargs):
            calls.append((n_samples, k_arg))
            return original(n_samples, k_arg, **kwargs)

        monkeypatch.setattr(
            _ktc_dispatch, "lookup_mi_classif_backend", _spy,
        )
        monkeypatch.delenv("MLFRAME_MI_BACKEND", raising=False)

        plugin_mi_classif_dispatch(x, y, 20)
        assert calls, (
            "plugin_mi_classif_dispatch did NOT consult "
            "lookup_mi_classif_backend; the KTC integration regressed."
        )
        assert calls[-1] == (n, 1), (
            f"single-col KTC lookup called with wrong shape: "
            f"got {calls[-1]}, expected ({n}, 1)"
        )

    def test_env_override_bypasses_ktc(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``MLFRAME_MI_BACKEND=njit`` / =cuda is the documented escape
        hatch; it MUST short-circuit the KTC lookup so operators
        debugging dispatcher behaviour aren't fighting the cache."""
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache import (
            dispatch as _ktc_dispatch,
        )

        rng = np.random.default_rng(11)
        n = 100_000
        X = rng.normal(size=(n, 5))
        y = rng.integers(0, 3, size=n).astype(np.int64)

        calls: list = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return "cuda"  # would route to cuda if called

        monkeypatch.setattr(
            _ktc_dispatch, "lookup_mi_classif_backend", _spy,
        )
        monkeypatch.setenv("MLFRAME_MI_BACKEND", "njit")
        plugin_mi_classif_batch_dispatch(X, y, 20)
        assert not calls, (
            f"MLFRAME_MI_BACKEND=njit did NOT short-circuit KTC: "
            f"lookup got {len(calls)} call(s). The env-var escape "
            f"hatch is the documented way to debug; it must bypass "
            f"the cache."
        )


class TestPluginMIPerHostRegionOverridesFallback:
    """A persisted per-host KTC region MUST override the hardcoded fallback.

    The hardcoded fallback (75k single / 10k batch) was measured on a
    GTX 1050 Ti cc 6.1. On other GPUs the real njit-vs-cuda crossover
    differs -- e.g. on an RTX 500 Ada the batch crossover is ~50k, so the
    10k fallback mis-routes the 10k-50k band to GPU where CPU is faster.
    The fix is purely data: ``ensure_mi_classif_dispatch_tuning`` populates
    a per-host region the lookup consults. This test pins that a region
    saying "njit at n=20k k=5" beats the fallback's cuda, so a regression
    that ignores the persisted crossover (reverting to the hardcoded
    constant) is caught.
    """

    def test_persisted_region_njit_at_20k_batch_beats_fallback_cuda(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache import (
            dispatch as _ktc_dispatch,
        )

        n, k = 20_000, 5
        # Sanity: the hardcoded fallback would (wrongly, on this HW) route
        # batch n=20k to cuda -- that is exactly the band the per-host
        # region corrects.
        assert _ktc_dispatch._fallback_mi_backend(n, k) == "cuda"

        class _FakeCache:
            def get_or_tune(self, name, *, dims, tuner, axes, fallback, **kw):
                # Emulate a persisted per-host region whose measured
                # crossover routes batch n<=20k to njit (RTX 500 Ada shape).
                if dims["n_samples"] <= 20_000:
                    return {"backend_choice": "njit"}
                return {"backend_choice": "cuda"}

        monkeypatch.setattr(_ktc_dispatch, "_get_cache", lambda: _FakeCache())
        choice = _ktc_dispatch.lookup_mi_classif_backend(n, k)
        assert choice == "njit", (
            "per-host KTC region was ignored; dispatcher fell back to the "
            "hardcoded GTX-1050-Ti crossover (10k batch), which mis-routes "
            "the 10k-50k band to GPU on faster-atomic GPUs."
        )


class TestPluginMIClassifFastSplitArgsort:
    """``plugin_mi_classif_fast`` / ``..._batch_fast`` hoist the
    ``np.argsort`` step OUT of numba into pure numpy (where it dispatches
    to a SIMD-optimised C sort that numba's argsort wrapper does not
    match at n=1500). Must produce bit-for-bit identical results to
    ``_plugin_mi_classif_njit`` because both compute the same plug-in
    MI on the same quantile-binned columns.
    """

    @pytest.mark.parametrize("n", [500, 1_500, 50_000])
    @pytest.mark.parametrize("n_classes", [2, 3, 5])
    def test_fast_single_col_matches_njit(self, n: int, n_classes: int) -> None:
        from mlframe.feature_selection.filters.hermite_fe import (
            _plugin_mi_classif_njit,
            plugin_mi_classif_fast,
        )
        rng = np.random.default_rng(seed=11 + n + n_classes)
        x = rng.normal(size=n)
        y = rng.integers(0, n_classes, size=n).astype(np.int64)
        njit_mi = float(_plugin_mi_classif_njit(x, y, 20))
        fast_mi = plugin_mi_classif_fast(x, y, 20)
        assert abs(fast_mi - njit_mi) < 1e-12, (
            f"plugin_mi_classif_fast diverged from njit at n={n}, "
            f"n_classes={n_classes}: fast={fast_mi}, njit={njit_mi}"
        )

    @pytest.mark.parametrize("n", [500, 1_500, 50_000])
    @pytest.mark.parametrize("k", [1, 5, 20])
    def test_fast_batch_matches_njit(self, n: int, k: int) -> None:
        from mlframe.feature_selection.filters.hermite_fe import (
            _plugin_mi_classif_batch_njit,
            plugin_mi_classif_batch_fast,
        )
        rng = np.random.default_rng(seed=11 + n + k)
        X = rng.normal(size=(n, k))
        y = rng.integers(0, 3, size=n).astype(np.int64)
        njit_arr = _plugin_mi_classif_batch_njit(X, y, 20)
        fast_arr = plugin_mi_classif_batch_fast(X, y, 20)
        np.testing.assert_allclose(fast_arr, njit_arr, atol=1e-12)


class TestPluginMIClassifBizValue:
    """biz_value: cupy must be measurably faster than njit at n >= 1M.

    Anything slower indicates a regression in the cupy port (extra
    H2D copies, missing batch fusion, etc.).
    """

    def test_cuda_faster_than_njit_at_production_scale(self) -> None:
        import time
        rng = np.random.default_rng(11)
        n, k = 1_000_000, 20
        X = rng.normal(size=(n, k))
        y = rng.integers(0, 3, size=n).astype(np.int64)
        # Warmup both paths so JIT + cupy compilation are out of timing.
        _plugin_mi_classif_batch_njit(X[:1000], y[:1000], 20)
        _plugin_mi_classif_batch_cuda(X[:1000], y[:1000], 20)
        t_njit = []
        t_cuda = []
        for _ in range(3):
            t0 = time.perf_counter()
            _plugin_mi_classif_batch_njit(X, y, 20)
            t_njit.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            _plugin_mi_classif_batch_cuda(X, y, 20)
            t_cuda.append(time.perf_counter() - t0)
        med_njit = float(np.median(t_njit))
        med_cuda = float(np.median(t_cuda))
        assert med_cuda < med_njit, (
            f"CUDA batch MI should be faster than njit at n=1M, k=20 -- "
            f"got cuda={med_cuda:.3f}s, njit={med_njit:.3f}s "
            f"(ratio={med_njit / max(med_cuda, 1e-9):.2f}x)"
        )


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required")
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_resident_batch_cuda_matches_host_input(seed):
    """MATRIX-NATIVE: _plugin_mi_classif_batch_cuda_resident (cupy in, NO H2D) must equal
    the host-input _plugin_mi_classif_batch_cuda bit-for-bit -- the resident entry only
    skips the per-call cp.asarray, the MI math is identical. Guards the H2D-free core the
    resident-candidate path uses."""
    import cupy as cp
    from mlframe.feature_selection.filters.hermite_fe import (
        _plugin_mi_classif_batch_cuda,
        _plugin_mi_classif_batch_cuda_resident,
    )
    rng = np.random.default_rng(seed)
    n, k = 4000, 11
    X = rng.standard_normal((n, k)).astype(np.float64)
    y = (rng.random(n) > 0.5).astype(np.int64)
    mi_host = _plugin_mi_classif_batch_cuda(X, y, 20)
    mi_res = _plugin_mi_classif_batch_cuda_resident(cp.asarray(X), cp.asarray(y), 20)
    assert mi_host.shape == mi_res.shape == (k,)
    assert float(np.max(np.abs(mi_host - mi_res))) == 0.0, (
        f"seed={seed}: resident MI must equal host-input MI bit-for-bit"
    )
