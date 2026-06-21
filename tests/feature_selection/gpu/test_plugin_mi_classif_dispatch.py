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


class TestPluginMIDispatchGroundTruthNjitOverride:
    """The host-input MI dispatch (``plugin_mi_classif_(batch_)dispatch``)
    deliberately defaults to njit via a GROUND-TRUTH OVERRIDE and does NOT
    consult ``lookup_mi_classif_backend``.

    The earlier (2026-05-20) KTC-consulting path favored cuda from a SOLO
    microbenchmark; the contention-aware end-to-end A/B then showed njit is
    3x faster on the real fit (GPU 318-368s/5.0GB vs njit 115s/1.6GB,
    identical selection), because the host-input cuda path pays a ~700ms
    per-call H2D/D2H + serialised launch/sync penalty the microbench never
    saw. So the dispatch now short-circuits to njit BEFORE any KTC lookup;
    the GPU win moved to the *resident* path (``_plugin_mi_classif_batch_
    cuda_resident``), which eliminates the per-call H2D the dispatch can't.

    These tests pin that the host-input dispatch does NOT pay the KTC lookup
    and returns the njit result; ``MLFRAME_MI_BACKEND=cuda`` still forces GPU
    for a caller whose own end-to-end profile justifies it (covered above).
    """

    def test_batch_dispatcher_does_not_consult_ktc(
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
        monkeypatch.delenv("MLFRAME_MI_BACKEND", raising=False)

        out = plugin_mi_classif_batch_dispatch(X, y, 20)
        assert not calls, (
            "plugin_mi_classif_batch_dispatch consulted the KTC lookup; the "
            "ground-truth njit override must short-circuit BEFORE it (the "
            "host-input cuda path is 3x slower end-to-end under contention)."
        )
        np.testing.assert_array_equal(out, _plugin_mi_classif_batch_njit(X, y, 20))

    def test_single_dispatcher_does_not_consult_ktc(
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

        out = plugin_mi_classif_dispatch(x, y, 20)
        assert not calls, (
            "plugin_mi_classif_dispatch consulted the KTC lookup; the "
            "ground-truth njit override must short-circuit BEFORE it."
        )
        assert out == float(_plugin_mi_classif_njit(x, y, 20))

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
    """A persisted per-host KTC region MUST override the conservative fallback.

    ``_fallback_mi_backend`` is now ``njit`` unconditionally (the earlier
    GPU-favoring constants came from a solo microbench; the contention-aware
    end-to-end fit is njit-faster -- see ``dispatch._fallback_mi_backend``).
    The per-host concurrency-aware sweep (``_run_sweep_mi_classif_dispatch``)
    is the ONLY thing allowed to route a region to cuda, where it genuinely
    wins under contention. This test pins that a persisted region saying
    "cuda at n=20k k=5" overrides the njit fallback, so a regression that
    ignores the persisted crossover (hardcoding njit even when the host was
    measured cuda-faster) is caught.
    """

    def test_persisted_region_cuda_at_20k_batch_overrides_njit_fallback(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache import (
            dispatch as _ktc_dispatch,
        )

        n, k = 20_000, 5
        # Sanity: the conservative fallback now routes batch n=20k to njit --
        # that is the safe default a measured per-host region may override.
        assert _ktc_dispatch._fallback_mi_backend(n, k) == "njit"

        class _FakeCache:
            def get_or_tune(self, name, *, dims, tuner, axes, fallback, **kw):
                # Emulate a persisted per-host region whose contention-aware
                # measurement found cuda faster at batch n<=20k on this host.
                if dims["n_samples"] <= 20_000:
                    return {"backend_choice": "cuda"}
                return {"backend_choice": "njit"}

        monkeypatch.setattr(_ktc_dispatch, "_get_cache", lambda: _FakeCache())
        choice = _ktc_dispatch.lookup_mi_classif_backend(n, k)
        assert choice == "cuda", (
            "per-host KTC region was ignored; lookup fell back to the "
            "conservative njit default instead of honoring the measured "
            "cuda crossover persisted for this host."
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
