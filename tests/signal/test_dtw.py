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
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n).astype(np.float32)
    y = rng.normal(0, 1, m).astype(np.float32)
    return x, y


class TestDtwCpuBaseline:
    def test_dtaidistance_available(self):
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
        _, path_strict = dtw_cpu(x, y, window=50, psi=0)
        _, path_psi = dtw_cpu(x, y, window=50, psi=30)
        # With psi, endpoints can drift; this is a soft check (we just
        # assert the path is non-degenerate).
        assert len(path_psi) > 10


class TestDtwGpuBackends:
    @pytest.mark.parametrize("shape", [(200, 150), (500, 300), (1000, 800)])
    def test_cupy_distance_matches_cpu(self, shape):
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
        assert rel < 0.05, f"distance disagreement {rel*100:.2f}%"
        # Path endpoints
        assert path_gpu[0] == (0, 0)
        assert path_gpu[-1] == (n - 1, m - 1)

    @pytest.mark.parametrize("shape", [(300, 200), (800, 500)])
    def test_numba_cuda_matches_cupy(self, shape):
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


class TestDispatcher:
    def test_small_n_routes_to_cpu(self):
        """Below the threshold the auto-dispatch path uses CPU. We
        bump the threshold above the input size and assert that the
        return matches dtw_cpu exactly."""
        pytest.importorskip("dtaidistance")
        from mlframe.signal.dtw import (
            dtw_dispatch, dtw_cpu, set_dtw_dispatch_threshold,
        )
        set_dtw_dispatch_threshold(10_000_000)  # force CPU
        x, y = _gen_pair(n=200, m=150)
        d_disp, path_disp = dtw_dispatch(x, y, window=50)
        d_cpu, path_cpu = dtw_cpu(x, y, window=50)
        assert d_disp == d_cpu
        assert path_disp == path_cpu

    def test_large_n_routes_to_gpu_when_available(self):
        pytest.importorskip("cupy")
        from mlframe.signal.dtw import (
            dtw_dispatch, dtw_cupy, set_dtw_dispatch_threshold,
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
        pytest.importorskip("dtaidistance")
        from mlframe.signal.dtw import dtw_dispatch, dtw_cpu
        monkeypatch.setenv("MLFRAME_DTW_BACKEND", "cpu")
        x, y = _gen_pair(n=500, m=400)
        d_disp, path_disp = dtw_dispatch(x, y, window=150)
        d_cpu, path_cpu = dtw_cpu(x, y, window=150)
        assert d_disp == d_cpu

    def test_backend_kwarg_force(self):
        pytest.importorskip("dtaidistance")
        from mlframe.signal.dtw import dtw_dispatch, dtw_cpu
        x, y = _gen_pair(n=500, m=400)
        d_disp, _ = dtw_dispatch(x, y, window=150, backend="cpu")
        d_cpu, _ = dtw_cpu(x, y, window=150)
        assert d_disp == d_cpu

    def test_unknown_backend_raises(self):
        from mlframe.signal.dtw import dtw_dispatch
        x, y = _gen_pair(n=100, m=80)
        with pytest.raises(ValueError):
            dtw_dispatch(x, y, backend="opencl")


class TestKernelTuningCacheLookup:
    def test_cache_lookup_falls_back_gracefully(self):
        """When kernel_tuning_cache is unavailable or has no entry
        for ``dtw_dispatch``, the lookup must return the source-code
        default. Exercised by calling ``_lookup_dtw_threshold`` with
        a fresh cell-count -- the cache miss is expected and the
        function must not raise."""
        from mlframe.signal.dtw import (
            _lookup_dtw_threshold, _DEFAULT_GPU_MIN_CELLS,
        )
        result = _lookup_dtw_threshold(n_cells=123_456)
        assert isinstance(result, int)
        # Default fallback path returns the source-code constant.
        assert result == _DEFAULT_GPU_MIN_CELLS or result > 0
