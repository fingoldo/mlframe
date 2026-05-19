"""Numerical-equivalence test for ``batch_pair_mi_cuda`` and
``batch_pair_mi_cupy`` vs the CPU njit baseline.

Both GPU backends must produce MIs that match the CPU kernel to fp tolerance
across the same parametrize matrix as ``test_batch_pair_mi_prange.py``. If
CUDA / CuPy is unavailable on the host (CI without GPU), the relevant test
auto-skips.

The dispatcher (:func:`dispatch_batch_pair_mi`) is also covered: with no
``force_backend`` argument it must return MIs identical to the CPU path
regardless of which backend it picked, and with ``force_backend`` it must
honour the override.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from mlframe.feature_selection.filters.batch_pair_mi_gpu import (
    _CUDA_AVAIL,
    _CUPY_AVAIL,
    batch_pair_mi_cuda,
    batch_pair_mi_cupy,
    batch_pair_mi_njit_prange,
    dispatch_batch_pair_mi,
)


def _build_factor_data(n_samples: int, nbins_per_col, seed: int):
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, nb, size=n_samples) for nb in nbins_per_col]
    data = np.column_stack(cols).astype(np.int32)
    return data, np.asarray(nbins_per_col, dtype=np.int32)


def _build_pair_inputs(n_samples, nbins_per_col, n_classes_y, seed):
    data, nbins = _build_factor_data(n_samples, nbins_per_col, seed)
    rng = np.random.default_rng(seed + 100)
    y_raw = rng.integers(0, n_classes_y, size=n_samples).astype(np.int32)
    freqs_y = np.bincount(y_raw, minlength=n_classes_y).astype(np.float64) / n_samples
    n_cols = len(nbins_per_col)
    pairs = list(itertools.combinations(range(n_cols), 2))
    pair_a = np.array([p[0] for p in pairs], dtype=np.int64)
    pair_b = np.array([p[1] for p in pairs], dtype=np.int64)
    return data, nbins, y_raw, freqs_y, pair_a, pair_b


PARAM_MATRIX = [
    (500, [4, 4, 4, 4], 2, 1),
    (500, [3, 5, 7, 4], 3, 2),
    (2000, [5, 5, 5, 5, 5, 5], 4, 3),
    (1000, [2, 3, 4, 5], 2, 4),
]


@pytest.mark.skipif(not _CUDA_AVAIL, reason="numba.cuda not available on this host")
@pytest.mark.parametrize("n_samples,nbins_per_col,n_classes_y,seed", PARAM_MATRIX)
def test_batch_pair_mi_cuda_matches_cpu(n_samples, nbins_per_col, n_classes_y, seed):
    data, nbins, y, freqs_y, pa, pb = _build_pair_inputs(
        n_samples, nbins_per_col, n_classes_y, seed,
    )
    mi_cpu = batch_pair_mi_njit_prange(data, pa, pb, nbins, y, freqs_y)
    mi_cuda = batch_pair_mi_cuda(data, pa, pb, nbins, y, freqs_y)
    np.testing.assert_allclose(
        mi_cuda, mi_cpu, atol=1e-9, rtol=1e-9,
        err_msg=(
            f"batch_pair_mi_cuda diverged from CPU njit baseline: "
            f"shapes=(n={n_samples}, nbins={nbins_per_col}, ny={n_classes_y}). "
            f"cuda[:5]={mi_cuda[:5]}, cpu[:5]={mi_cpu[:5]}"
        ),
    )


@pytest.mark.skipif(not _CUPY_AVAIL, reason="cupy not available on this host")
@pytest.mark.parametrize("n_samples,nbins_per_col,n_classes_y,seed", PARAM_MATRIX)
def test_batch_pair_mi_cupy_matches_cpu(n_samples, nbins_per_col, n_classes_y, seed):
    data, nbins, y, freqs_y, pa, pb = _build_pair_inputs(
        n_samples, nbins_per_col, n_classes_y, seed,
    )
    mi_cpu = batch_pair_mi_njit_prange(data, pa, pb, nbins, y, freqs_y)
    mi_cupy = batch_pair_mi_cupy(data, pa, pb, nbins, y, freqs_y)
    np.testing.assert_allclose(
        mi_cupy, mi_cpu, atol=1e-9, rtol=1e-9,
        err_msg=(
            f"batch_pair_mi_cupy diverged from CPU njit baseline: "
            f"shapes=(n={n_samples}, nbins={nbins_per_col}, ny={n_classes_y}). "
            f"cupy[:5]={mi_cupy[:5]}, cpu[:5]={mi_cpu[:5]}"
        ),
    )


def test_dispatch_falls_back_to_cpu_below_thresholds():
    """Small inputs must hit the CPU path regardless of GPU availability."""
    data, nbins, y, freqs_y, pa, pb = _build_pair_inputs(500, [4, 4, 4, 4], 2, 1)
    mi, backend = dispatch_batch_pair_mi(data, pa, pb, nbins, y, freqs_y)
    assert backend == "njit"
    mi_ref = batch_pair_mi_njit_prange(data, pa, pb, nbins, y, freqs_y)
    np.testing.assert_allclose(mi, mi_ref, atol=1e-12, rtol=1e-12)


def test_dispatch_force_backend_njit():
    data, nbins, y, freqs_y, pa, pb = _build_pair_inputs(500, [4, 4, 4, 4], 2, 1)
    mi, backend = dispatch_batch_pair_mi(
        data, pa, pb, nbins, y, freqs_y, force_backend="njit",
    )
    assert backend == "njit"


@pytest.mark.skipif(not _CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_dispatch_force_backend_cuda_matches_cpu():
    data, nbins, y, freqs_y, pa, pb = _build_pair_inputs(2000, [5, 5, 5, 5, 5, 5], 4, 3)
    mi_cpu = batch_pair_mi_njit_prange(data, pa, pb, nbins, y, freqs_y)
    mi_force, backend = dispatch_batch_pair_mi(
        data, pa, pb, nbins, y, freqs_y, force_backend="cuda",
    )
    assert backend == "cuda"
    np.testing.assert_allclose(mi_force, mi_cpu, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(not _CUPY_AVAIL, reason="cupy not available on this host")
def test_dispatch_force_backend_cupy_matches_cpu():
    data, nbins, y, freqs_y, pa, pb = _build_pair_inputs(2000, [5, 5, 5, 5, 5, 5], 4, 3)
    mi_cpu = batch_pair_mi_njit_prange(data, pa, pb, nbins, y, freqs_y)
    mi_force, backend = dispatch_batch_pair_mi(
        data, pa, pb, nbins, y, freqs_y, force_backend="cupy",
    )
    assert backend == "cupy"
    np.testing.assert_allclose(mi_force, mi_cpu, atol=1e-9, rtol=1e-9)


def test_dispatch_force_unavailable_falls_back_to_cpu(monkeypatch):
    """When the user forces a GPU backend that isn't installed, dispatcher
    must silently fall through to the CPU path (not raise)."""
    import mlframe.feature_selection.filters.batch_pair_mi_gpu as mod

    monkeypatch.setattr(mod, "_CUDA_AVAIL", False)
    monkeypatch.setattr(mod, "_CUPY_AVAIL", False)
    data, nbins, y, freqs_y, pa, pb = _build_pair_inputs(500, [4, 4, 4, 4], 2, 1)
    mi, backend = mod.dispatch_batch_pair_mi(
        data, pa, pb, nbins, y, freqs_y, force_backend="cuda",
    )
    assert backend == "njit"
