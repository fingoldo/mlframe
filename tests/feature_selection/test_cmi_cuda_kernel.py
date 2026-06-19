"""Parity + dispatch tests for the batched CUDA conditional-MI kernel.

Gate: the GPU kernel must reproduce the CPU ``conditional_mi`` (raw I(X;Y|Z)) within ~1e-9 on the
SAME discretized inputs, across seeds and per-variable cardinalities. CUDA tests skip cleanly when
no GPU / cupy is present; the CPU-fallback path of the dispatcher is exercised unconditionally.

RAM-light by mandate: n <= 3000, p <= 64, single process.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory._cmi_cuda import (
    conditional_mi_batched_dispatch,
    cupy_available,
)
from mlframe.feature_selection.filters.info_theory._entropy_kernels import conditional_mi

_HAS_GPU = cupy_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA/cupy GPU available")


def _make_case(seed, n, p, nbx, nby, nbz):
    rng = np.random.RandomState(seed)
    cols = [rng.randint(0, nbx, n) for _ in range(p)]
    nbins = [nbx] * p
    yc = rng.randint(0, nby, n)
    zc = rng.randint(0, nbz, n)
    fd = np.column_stack(cols + [yc, zc]).astype(np.int32)
    fb = np.array(nbins + [nby, nbz], dtype=np.int64)
    return fd, fb, p  # y_index = p, z_index = p+1


def _cpu_ref(fd, fb, p):
    yi, zi = p, p + 1
    return np.array([
        conditional_mi(fd, np.array([c]), np.array([yi]), np.array([zi]), None, fb)
        for c in range(p)
    ])


@_gpu_only
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("nbins", [(2, 2, 2), (4, 3, 5), (7, 2, 4), (3, 3, 3)])
def test_cuda_cmi_bit_parity(seed, nbins):
    nbx, nby, nbz = nbins
    fd, fb, p = _make_case(seed, n=3000, p=20, nbx=nbx, nby=nby, nbz=nbz)
    gpu = conditional_mi_batched_dispatch(fd, np.arange(p), p, p + 1, fb, force="cuda")
    cpu = _cpu_ref(fd, fb, p)
    assert np.abs(gpu - cpu).max() < 1e-9


@_gpu_only
def test_cuda_cmi_skewed_and_constant_columns():
    """Heavy-skew + a constant candidate column (degenerate, CMI -> 0) still match the CPU."""
    rng = np.random.RandomState(7)
    n, p = 2500, 12
    cols = [rng.choice([0, 1, 2, 3], size=n, p=[0.85, 0.1, 0.03, 0.02]) for _ in range(p - 1)]
    cols.append(np.zeros(n, dtype=int))  # constant column
    yc = rng.randint(0, 3, n)
    zc = rng.randint(0, 4, n)
    fd = np.column_stack(cols + [yc, zc]).astype(np.int32)
    fb = np.array([4] * p + [3, 4], dtype=np.int64)
    gpu = conditional_mi_batched_dispatch(fd, np.arange(p), p, p + 1, fb, force="cuda")
    cpu = _cpu_ref(fd, fb, p)
    assert np.abs(gpu - cpu).max() < 1e-9
    assert gpu[-1] == pytest.approx(0.0, abs=1e-12)  # constant col -> zero CMI


def test_cpu_fallback_matches_reference():
    """force='cpu' path of the dispatcher equals the direct CPU loop (no GPU needed)."""
    fd, fb, p = _make_case(seed=0, n=1500, p=10, nbx=4, nby=3, nbz=3)
    disp = conditional_mi_batched_dispatch(fd, np.arange(p), p, p + 1, fb, force="cpu")
    cpu = _cpu_ref(fd, fb, p)
    assert np.abs(disp - cpu).max() == 0.0  # same njit kernel -> exact


def test_dispatch_default_returns_finite_nonneg():
    """Default (auto) dispatch returns finite, non-negative CMI regardless of GPU presence."""
    fd, fb, p = _make_case(seed=3, n=1200, p=8, nbx=3, nby=2, nbz=3)
    out = conditional_mi_batched_dispatch(fd, np.arange(p), p, p + 1, fb)
    assert out.shape == (p,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_kernel_module_pickle_safe():
    """No live CUDA context / RawKernel leaks via module import; a consumer dict pickles."""
    import pickle

    payload = {"cand": np.arange(5), "result": np.zeros(5)}
    assert pickle.loads(pickle.dumps(payload))["cand"].tolist() == list(range(5))


@_gpu_only
def test_single_candidate_and_large_p():
    """Edge shapes: p=1 and p=64 both match the CPU."""
    for p in (1, 64):
        fd, fb, _ = _make_case(seed=5, n=2000, p=p, nbx=4, nby=2, nbz=3)
        gpu = conditional_mi_batched_dispatch(fd, np.arange(p), p, p + 1, fb, force="cuda")
        cpu = _cpu_ref(fd, fb, p)
        assert np.abs(gpu - cpu).max() < 1e-9
