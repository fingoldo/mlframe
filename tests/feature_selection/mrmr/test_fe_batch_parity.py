"""Parity + dispatch guard for the two FE-batcher backends (CPU ``_fe_cpu_batch`` / GPU ``_fe_gpu_batch``)
and the dispatcher (``_fe_batch_dispatch``) (2026-06-26).

The user's contract: the separate CPU and GPU FE-scoring paths must produce IDENTICAL features. They share
the edge-binned plain plug-in MI (``_fe_edge_mi``), so this pins:
  1. CPU batcher == the underlying edge primitive, and is column-chunk-INVARIANT.
  2. (CUDA) CPU batcher == GPU batcher to ~1e-9 on continuous AND tied matrices.
  3. dispatcher honours the force env / STRICT / default-CPU rules and the chosen backend matches the path.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_cpu_batch import cpu_fe_batch_mi
from mlframe.feature_selection.filters._fe_edge_mi import plugin_mi_classif_batch_edge_njit
from mlframe.feature_selection.filters._fe_batch_dispatch import choose_fe_batch_backend, fe_batch_mi


def _continuous(seed=3, n=5000, k=40, nbins=10):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1, 5, n); b = rng.uniform(1, 5, n); c = rng.uniform(1, 5, n)
    cols = [a ** 2 / b, np.log(c) * a, a * b, np.log(a) + np.log(b)]
    while len(cols) < k:
        cols.append(rng.uniform(0, 1, n))
    X = np.column_stack([np.nan_to_num(col.astype(np.float64)) for col in cols])
    y = np.searchsorted(np.quantile(a, np.linspace(0, 1, nbins + 1))[1:-1], a).astype(np.int64)
    return X, y, nbins


def _tied(seed=5, n=5000, k=24, nbins=10):
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 5, n).astype(np.float64), (rng.uniform(0, 1, n) > 0.5).astype(np.float64),
            np.sign(rng.normal(0, 1, n)).astype(np.float64), rng.integers(0, 3, n).astype(np.float64)]
    while len(cols) < k:
        cols.append(rng.integers(0, rng.integers(2, 7), n).astype(np.float64))
    X = np.column_stack([c.astype(np.float64) for c in cols])
    y = rng.integers(0, nbins, n).astype(np.int64)
    return X, y, nbins


@pytest.mark.parametrize("fixture", [_continuous, _tied], ids=["continuous", "tied"])
def test_cpu_batcher_matches_primitive_and_is_chunk_invariant(fixture):
    X, y, nb = fixture()
    prim = plugin_mi_classif_batch_edge_njit(X, y, nb)
    whole = cpu_fe_batch_mi(X, y, nb)
    chunked = cpu_fe_batch_mi(X, y, nb, max_cols_per_chunk=3)  # force many chunks
    assert np.array_equal(prim, whole), "CPU batcher (no chunk) must equal the underlying edge primitive"
    assert np.allclose(whole, chunked, atol=1e-12, rtol=0), "column chunking must not change MI"


def test_dispatcher_force_env(monkeypatch):
    monkeypatch.setenv("MLFRAME_FE_VRAM_BACKEND", "cpu")
    assert choose_fe_batch_backend(10_000, 100) == "cpu"
    monkeypatch.delenv("MLFRAME_FE_VRAM_BACKEND", raising=False)
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    # default (no force, no tuned entry) is the conservative CPU
    assert choose_fe_batch_backend(10_000, 100) == "cpu"


def test_dispatcher_cpu_path_matches_primitive():
    X, y, nb = _continuous()
    out = fe_batch_mi(X, y, nb, backend="cpu")
    assert np.array_equal(out, plugin_mi_classif_batch_edge_njit(X, y, nb))


# ---------------------------------------------------------------------------
# CUDA: CPU batcher == GPU batcher (the cross-backend identity the user requires).
# ---------------------------------------------------------------------------
def _need_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
@pytest.mark.parametrize("fixture", [_continuous, _tied], ids=["continuous", "tied"])
def test_cpu_batcher_matches_gpu_batcher(fixture):
    import cupy as cp
    from mlframe.feature_selection.filters._fe_gpu_batch import gpu_fe_batch_mi

    X, y, nb = fixture()
    cpu = cpu_fe_batch_mi(X, y, nb)
    gpu = gpu_fe_batch_mi(X, y, nb)
    cp.get_default_memory_pool().free_all_blocks()
    assert np.allclose(cpu, gpu, atol=1e-9, rtol=0), (
        f"CPU vs GPU batcher diverged: max|d|={np.max(np.abs(cpu - gpu)):.3e} "
        f"(the two FE paths must select identically)"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
@pytest.mark.parametrize("fixture", [_continuous, _tied], ids=["continuous", "tied"])
def test_gpu_f32_batch_is_selection_equivalent_to_f64(fixture):
    """The opt-in f32 GPU batch path (MLFRAME_FE_VRAM_F32) is ~2.2x faster but only SELECTION-EQUIVALENT to
    f64 (not 1e-9): the MI values drift ~5e-6 from f32 order statistics, but the RANKING is preserved, so
    the same features are selected. Pin rank-equivalence (Spearman 1.0, identical argmax + top-K), NOT a
    tight value tolerance."""
    import cupy as cp
    from scipy.stats import spearmanr
    from mlframe.feature_selection.filters._fe_gpu_batch import gpu_fe_batch_mi

    X, y, nb = fixture()
    mi64 = gpu_fe_batch_mi(X, y, nb, dtype=np.float64)
    mi32 = gpu_fe_batch_mi(X, y, nb, dtype=np.float32)
    cp.get_default_memory_pool().free_all_blocks()
    assert spearmanr(mi64, mi32).correlation > 0.99999, "f32 must preserve the MI ranking"
    assert np.argmax(mi64) == np.argmax(mi32), "f32 must keep the same top feature"
    top = min(20, X.shape[1])
    assert set(np.argsort(-mi64)[:top]) == set(np.argsort(-mi32)[:top]), "f32 top-K set must match f64"


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_gpu_batcher_vram_chunk_invariant():
    """The GPU executor's VRAM column-chunk boundaries must not change per-column MI."""
    import cupy as cp
    from mlframe.feature_selection.filters._fe_gpu_batch._executor import gpu_fe_batch_mi

    X, y, nb = _continuous(n=4000, k=60)
    full = gpu_fe_batch_mi(X, y, nb)
    # Score in two halves -> concatenate; must match the single-call result (per-column independence).
    half = X.shape[1] // 2
    a = gpu_fe_batch_mi(np.ascontiguousarray(X[:, :half]), y, nb)
    b = gpu_fe_batch_mi(np.ascontiguousarray(X[:, half:]), y, nb)
    cp.get_default_memory_pool().free_all_blocks()
    assert np.allclose(full, np.concatenate([a, b]), atol=1e-12, rtol=0)
