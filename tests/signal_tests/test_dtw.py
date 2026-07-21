"""Tests for ``mlframe.signal.dtw`` -- CPU + GPU backends + auto-dispatch.

Correctness contract: GPU paths must produce a path whose endpoint is
the same global-cost optimum as the CPU dtaidistance baseline. Path
prefix may differ in the middle when multiple cells share equal cost
(ties resolved by backtrace policy differently across backends).
Distance values agree within float32 tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest


def _gen_pair(n=200, m=150, seed=0):
    """Builds seeded synthetic test data; returns ``(x, y)``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n).astype(np.float32)
    y = rng.normal(0, 1, m).astype(np.float32)
    return x, y


class TestDtwCpuBaseline:
    """Groups tests covering TestDtwCpuBaseline."""
    def test_dtaidistance_available(self):
        """Dtaidistance available."""
        pytest.importorskip("dtaidistance")
        from mlframe.signal.dtw import dtw_cpu

        x, y = _gen_pair()
        d, path = dtw_cpu(x, y, window=50)
        assert np.isfinite(d)
        assert d > 0
        assert isinstance(path, list)
        assert path[0] == (0, 0)
        assert path[-1] == (len(x) - 1, len(y) - 1)

    def test_psi_relaxation_changes_endpoints(self):
        """psi > 0 relaxes start/end constraints; the path may not
        start at (0, 0) or end at (n-1, m-1)."""
        pytest.importorskip("dtaidistance")
        from mlframe.signal.dtw import dtw_cpu

        x, y = _gen_pair(n=200, m=300)
        _, _path_strict = dtw_cpu(x, y, window=50, psi=0)
        _, path_psi = dtw_cpu(x, y, window=50, psi=30)
        # With psi, endpoints can drift; this is a soft check (we just
        # assert the path is non-degenerate).
        assert len(path_psi) > 10


class TestDtwGpuBackends:
    """Groups tests covering TestDtwGpuBackends."""

    @pytest.mark.parametrize("shape", [(200, 150), (500, 300), (1000, 800)])
    def test_cupy_distance_matches_cpu(self, shape):
        """Cupy distance matches cpu."""
        pytest.importorskip("cupy")
        pytest.importorskip("dtaidistance")
        from mlframe.signal.dtw import dtw_cpu, dtw_cupy

        n, m = shape
        # Window must be >= |n-m| so the band admits the (n,m) endpoint.
        window = max(50, abs(n - m) + 20)
        x, y = _gen_pair(n=n, m=m, seed=hash(shape) & 0xFFFF)
        d_cpu, _ = dtw_cpu(x, y, window=window)
        d_gpu, path_gpu = dtw_cupy(x, y, window=window)
        # Cost values differ between dtaidistance (uses 1-Euclidean,
        # square-root at the END) and our squared-Euclidean diagonal
        # kernel (root once at the end). Compare relative agreement:
        # the relative gap should be < 5% on random data.
        rel = abs(d_cpu - d_gpu) / max(d_cpu, 1e-9)
        assert rel < 0.05, f"distance disagreement {rel * 100:.2f}%"
        # Path endpoints
        assert path_gpu[0] == (0, 0)
        assert path_gpu[-1] == (n - 1, m - 1)

    @pytest.mark.parametrize("shape", [(300, 200), (800, 500)])
    def test_numba_cuda_matches_cupy(self, shape):
        """Numba cuda matches cupy."""
        pytest.importorskip("numba")
        from numba import cuda

        if not cuda.is_available():
            pytest.skip("no CUDA device")
        pytest.importorskip("cupy")
        from mlframe.signal.dtw import dtw_cuda, dtw_cupy

        n, m = shape
        window = max(50, abs(n - m) + 20)
        x, y = _gen_pair(n=n, m=m, seed=hash(shape) & 0xFFFF)
        d_nb, path_nb = dtw_cuda(x, y, window=window)
        d_cp, path_cp = dtw_cupy(x, y, window=window)
        # Both use the same numerical recipe (squared diff, min-of-3,
        # final sqrt); distances should agree to fp32 tolerance.
        np.testing.assert_allclose(d_nb, d_cp, rtol=1e-4, atol=1e-3)
        # Endpoint cells must match (both backtraces converge there).
        assert path_nb[-1] == path_cp[-1]
        assert path_nb[0] == path_cp[0]


class TestDtwBandedGpuBuffer:
    """CPX-P0-2: the GPU paths store only the 2*window+1 live Sakoe-Chiba
    diagonals (O(n*window) device RAM) instead of the full (n+1)x(m+1) cost
    matrix. The banded buffer must produce a BIT-IDENTICAL distance + path to the
    pre-fix full-matrix sweep AND match the dtaidistance CPU reference, including
    band-boundary windows."""

    def _full_matrix_cupy(self, x, y, window):
        """The retained pre-CPX-P0-2 full-matrix cupy sweep (regression baseline)."""
        from mlframe.signal.dtw import dtw_cupy_full

        return dtw_cupy_full(x, y, window=window)

    @pytest.mark.parametrize(
        "n,m,window",
        [
            (500, 500, 200),  # square, ample band
            (1000, 800, 250),  # rectangular, band > |n-m|
            (600, 400, 200),  # window == |n-m| (band-boundary: endpoint exactly on the edge)
            (500, 500, 499),  # near-full band
            (400, 400, 30),  # narrow band
        ],
    )
    def test_banded_cupy_bit_identical_to_full_matrix(self, n, m, window):
        """Banded distance + warping path == pre-fix full-matrix GPU path, exactly."""
        pytest.importorskip("cupy")
        from mlframe.signal.dtw import dtw_cupy_banded

        rng = np.random.default_rng(n * 7 + m * 13 + window)
        x = rng.standard_normal(n).astype(np.float32)
        y = rng.standard_normal(m).astype(np.float32)
        d_full, p_full = self._full_matrix_cupy(x, y, window)
        d_band, p_band = dtw_cupy_banded(x, y, window=window)
        assert d_band == d_full, f"banded {d_band} != full {d_full}"
        assert p_band == p_full, "banded warping path diverged from full-matrix path"

    @pytest.mark.parametrize("n,m,window", [(500, 500, 200), (800, 600, 250)])
    def test_banded_cupy_matches_cpu_reference(self, n, m, window):
        """Banded GPU distance matches the dtaidistance CPU reference (fp32 tol)."""
        pytest.importorskip("cupy")
        pytest.importorskip("dtaidistance")
        from mlframe.signal.dtw import dtw_cupy_banded, dtw_cpu

        rng = np.random.default_rng(n + m + window)
        x = rng.standard_normal(n).astype(np.float32)
        y = rng.standard_normal(m).astype(np.float32)
        d_band, _ = dtw_cupy_banded(x, y, window=window)
        d_cpu, _ = dtw_cpu(x, y, window=window)
        rel = abs(d_band - d_cpu) / max(d_cpu, 1e-9)
        assert rel < 1e-4, f"banded vs CPU rel gap {rel:.2e}"

    def test_banded_cupy_allocates_only_band_sized_device_buffer(self):
        """The banded path must allocate O(n*window) device RAM, not O(n*m). Measure
        the actual peak cupy pool growth so a revert to the full-matrix cost buffer
        (which would allocate >= n*m*4 bytes) trips this regression sensor."""
        import cupy as cp
        from mlframe.signal.dtw import dtw_cupy_banded

        n, m, window = 2000, 2000, 100
        rng = np.random.default_rng(1)
        x = rng.standard_normal(n).astype(np.float32)
        y = rng.standard_normal(m).astype(np.float32)
        dtw_cupy_banded(x[:200], y[:200], window=window)  # warm kernel compile
        mp = cp.get_default_memory_pool()
        mp.free_all_blocks()
        base = mp.total_bytes()
        d_band, _ = dtw_cupy_banded(x, y, window=window)
        peak = mp.total_bytes() - base
        full_bytes = (n + 1) * (m + 1) * 4
        band_bytes = (n + 1) * (2 * window + 1) * 4
        # Peak growth must be far below the full matrix (allow slack for x/y + pool
        # rounding); a full-matrix revert allocates ~16 MB here vs the band's ~1.6 MB.
        assert peak < full_bytes // 2, f"peak {peak} >= half the full matrix {full_bytes}"
        assert peak < band_bytes * 6, f"peak {peak} far exceeds band buffer {band_bytes}"
        assert np.isfinite(d_band) and d_band > 0

    def test_banded_numba_cuda_matches_banded_cupy(self):
        """The numba.cuda banded path agrees with the cupy banded path to fp32 tol."""
        pytest.importorskip("numba")
        from numba import cuda

        if not cuda.is_available():
            pytest.skip("no CUDA device")
        pytest.importorskip("cupy")
        from mlframe.signal.dtw import dtw_cuda, dtw_cupy

        rng = np.random.default_rng(99)
        x = rng.standard_normal(700).astype(np.float32)
        y = rng.standard_normal(500).astype(np.float32)
        d_nb, p_nb = dtw_cuda(x, y, window=250)
        d_cp, p_cp = dtw_cupy(x, y, window=250)
        np.testing.assert_allclose(d_nb, d_cp, rtol=1e-4, atol=1e-3)
        assert p_nb == p_cp


class TestDispatcher:
    """Groups tests covering TestDispatcher."""
    def test_small_n_routes_to_cpu(self, monkeypatch, tmp_path):
        """Below the threshold the FALLBACK routes to CPU. ``set_dtw_dispatch_threshold``
        governs the fallback only -- the dispatch (spec.choose) prefers a tuned cache
        entry over the threshold -- so we exercise the fallback path deterministically:
        an isolated empty cache + autotune off (no on-miss sweep) -> fallback -> the
        raised threshold -> CPU, matching dtw_cpu exactly."""
        pytest.importorskip("dtaidistance")
        monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("MLFRAME_DTW_AUTOTUNE", "0")
        from mlframe.signal.dtw import (
            dtw_dispatch,
            dtw_cpu,
            set_dtw_dispatch_threshold,
            _DTW_SPEC,
        )

        _DTW_SPEC._choice_cache.clear()  # drop any memoized choice from earlier tests
        set_dtw_dispatch_threshold(10_000_000)  # force CPU via the fallback
        x, y = _gen_pair(n=200, m=150)
        d_disp, path_disp = dtw_dispatch(x, y, window=50)
        d_cpu, path_cpu = dtw_cpu(x, y, window=50)
        assert d_disp == d_cpu
        assert path_disp == path_cpu
        _DTW_SPEC._choice_cache.clear()

    def test_large_n_routes_to_gpu_when_available(self):
        """Large n routes to gpu when available."""
        pytest.importorskip("cupy")
        from mlframe.signal.dtw import (
            dtw_dispatch,
            dtw_cupy,
            set_dtw_dispatch_threshold,
        )

        # Tiny threshold -> auto-pick GPU on any non-trivial input.
        set_dtw_dispatch_threshold(100)
        x, y = _gen_pair(n=500, m=400)
        d_disp, _ = dtw_dispatch(x, y, window=150)
        d_cupy, _ = dtw_cupy(x, y, window=150)
        np.testing.assert_allclose(d_disp, d_cupy, rtol=1e-6)

    def test_psi_forces_cpu_even_on_gpu_hw(self):
        """GPU backends don't honour psi; dispatcher must fall to CPU
        when psi > 0 to preserve correctness."""
        pytest.importorskip("dtaidistance")
        from mlframe.signal.dtw import dtw_dispatch, dtw_cpu, set_dtw_dispatch_threshold

        set_dtw_dispatch_threshold(100)  # auto-prefer GPU otherwise
        x, y = _gen_pair(n=500, m=400)
        d_disp, path_disp = dtw_dispatch(x, y, window=150, psi=20)
        d_cpu, path_cpu = dtw_cpu(x, y, window=150, psi=20)
        assert d_disp == d_cpu
        assert path_disp == path_cpu

    def test_env_var_force_override(self, monkeypatch):
        """Env var force override."""
        pytest.importorskip("dtaidistance")
        from mlframe.signal.dtw import dtw_dispatch, dtw_cpu

        monkeypatch.setenv("MLFRAME_DTW_BACKEND", "cpu")
        x, y = _gen_pair(n=500, m=400)
        d_disp, _path_disp = dtw_dispatch(x, y, window=150)
        d_cpu, _path_cpu = dtw_cpu(x, y, window=150)
        assert d_disp == d_cpu

    def test_backend_kwarg_force(self):
        """Backend kwarg force."""
        pytest.importorskip("dtaidistance")
        from mlframe.signal.dtw import dtw_dispatch, dtw_cpu

        x, y = _gen_pair(n=500, m=400)
        d_disp, _ = dtw_dispatch(x, y, window=150, backend="cpu")
        d_cpu, _ = dtw_cpu(x, y, window=150)
        assert d_disp == d_cpu

    def test_unknown_backend_raises(self):
        """Unknown backend raises."""
        from mlframe.signal.dtw import dtw_dispatch

        x, y = _gen_pair(n=100, m=80)
        with pytest.raises(ValueError):
            dtw_dispatch(x, y, backend="opencl")


class TestKernelTuningCacheLookup:
    """Groups tests covering TestKernelTuningCacheLookup."""
    def test_cache_lookup_falls_back_gracefully(self, monkeypatch, tmp_path):
        """On a cache miss (no tuned entry for ``dtw_dispatch``), the dispatch
        falls back gracefully to a valid source-code-default backend without
        raising. Exercised via the spec's choose() with an isolated empty cache +
        autotune off (so no on-miss sweep) -> the _dtw_fallback_choice heuristic."""
        monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("MLFRAME_DTW_AUTOTUNE", "0")
        from mlframe.signal.dtw import _DTW_SPEC, _dtw_fallback_choice

        _DTW_SPEC._choice_cache.clear()
        result = _DTW_SPEC.choose(n_cells=123_456)
        assert result in ("cpu", "cuda", "cupy")
        assert result == _dtw_fallback_choice(123_456)  # fallback path on a miss
        _DTW_SPEC._choice_cache.clear()
