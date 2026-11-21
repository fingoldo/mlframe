"""A host MemoryError in the CUDA path of the usability |corr| is retried on the device with smaller operand blocks instead of falling to the CPU backend; the values are the CPU backend's."""

import numpy as np
import pytest

pytest.importorskip("cupy")
numba_cuda = pytest.importorskip("numba.cuda")

from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as G


def _inputs(n=4000, p=24, seed=0):
    """A float32 target, a float32 operand matrix (one row per operand) and all pairs of its first ten rows."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).astype(np.float32)
    ops = rng.standard_normal((p, n)).astype(np.float32)
    ii, jj = np.triu_indices(10, 1)
    return y, ops, ii.astype(np.int64), jj.astype(np.int64)


def test_memoryerror_is_retried_on_the_device_with_smaller_blocks(monkeypatch):
    """The whole-matrix call raises MemoryError once; the result comes back from the blocked device path and equals the CPU one."""
    if not numba_cuda.is_available():
        pytest.skip("no CUDA device")
    y, ops, pa, pb = _inputs()
    forms = np.asarray(G.ALL_FORM_IDS, dtype=np.int64)
    cpu = G.batch_pair_usability_corr_njit_parallel(y, ops, pa, pb, forms)
    real = G.batch_pair_usability_corr_cuda_warp
    calls = {"n": 0}

    def flaky(*a, **k):
        """Raises a host MemoryError on the first (whole matrix) call only."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise MemoryError("Unable to allocate 240. MiB")
        return real(*a, **k)

    monkeypatch.setattr(G, "batch_pair_usability_corr_cuda_warp", flaky)
    monkeypatch.setattr(G, "_device_wins", lambda n: True)
    got, backend = G._dispatch_batch_pair_usability_corr_uncached(y, ops, pa, pb, forms)
    assert backend == "cuda_warp_blocked", backend
    np.testing.assert_allclose(got, cpu, rtol=0, atol=2e-5)
    assert calls["n"] >= 2


def test_other_errors_still_fall_back_to_the_cpu(monkeypatch):
    """A failure that is not a MemoryError keeps the old route to the CPU backend."""
    if not numba_cuda.is_available():
        pytest.skip("no CUDA device")
    y, ops, pa, pb = _inputs(seed=1)
    forms = np.asarray(G.ALL_FORM_IDS, dtype=np.int64)

    def broken(*a, **k):
        """A driver-style failure."""
        raise RuntimeError("driver fault")

    monkeypatch.setattr(G, "batch_pair_usability_corr_cuda_warp", broken)
    monkeypatch.setattr(G, "_device_wins", lambda n: True)
    _got, backend = G._dispatch_batch_pair_usability_corr_uncached(y, ops, pa, pb, forms)
    assert backend == "cpu"
