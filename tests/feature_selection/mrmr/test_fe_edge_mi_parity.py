"""Parity guard for the edge-binning plain plug-in MI (``_fe_edge_mi``), the CPU twin of the GPU resident
edge plug-in MI used by the FE batcher (2026-06-26).

The two FE-batcher backends (CPU njit, GPU cupy) must score every candidate column IDENTICALLY so they
select the same features. The orthogonal/basis families score by PLAIN plug-in MI; the legacy CPU kernel
bins x by RANK while the GPU bins by PERCENTILE EDGE. On continuous data these agree bit-for-bit; on TIED
columns they diverge. This suite pins:

  1. CPU edge MI == a numpy edge reference (~1e-9) on continuous AND tied fixtures.
  2. CPU edge MI == legacy CPU RANK MI on continuous data (selection-neutral unification), and DIFFERS on
     tied data (documents the exact divergence the edge unification removes -- so the test is meaningful).
  3. (CUDA only) CPU edge MI == GPU resident edge MI (~1e-9) on continuous AND tied -- the real
     cross-backend identity guarantee.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_edge_mi import plugin_mi_classif_batch_edge_njit
from mlframe.feature_selection.filters.hermite_fe import _plugin_mi_classif_batch_njit


def _np_edge_mi_one(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Reference: equi-frequency percentile-edge binning matching the GPU orth twin (FIXED n_bins-1
    interior edges = np.quantile lerp [1:-1], NO dedup) + searchsorted-right + plain plug-in MI,
    mirroring ``_fe_edge_mi._plugin_mi_classif_edge_njit`` exactly (hist sized over n_bins)."""
    n = x.shape[0]
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, qs)            # NO np.unique: keep all n_bins-1 interior edges (GPU convention)
    interior = edges[1:-1]
    if interior.shape[0] == 0:
        codes = np.zeros(n, np.int64)
    else:
        codes = np.searchsorted(interior, x, side="right").astype(np.int64)
    y_min = int(y.min())
    yc = y.astype(np.int64) - y_min
    n_classes = int(yc.max()) + 1
    hxy = np.zeros((n_bins, n_classes), dtype=np.int64)
    np.add.at(hxy, (codes, yc), 1)
    hist_x = hxy.sum(axis=1)
    hist_y = hxy.sum(axis=0)
    log_n = np.log(n)
    mi = 0.0
    for b in range(n_bins):
        if hist_x[b] == 0:
            continue
        log_hx = np.log(hist_x[b])
        for c in range(n_classes):
            n_xy = hxy[b, c]
            if n_xy == 0 or hist_y[c] == 0:
                continue
            mi += (n_xy / n) * (np.log(n_xy) + log_n - log_hx - np.log(hist_y[c]))
    return max(mi, 0.0)


def _np_edge_mi_batch(X, y, n_bins):
    return np.array([_np_edge_mi_one(np.ascontiguousarray(X[:, j]), y, n_bins) for j in range(X.shape[1])])


def _continuous_fixture(seed=7, n=6000, k=24, n_bins=10):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1, 5, n); b = rng.uniform(1, 5, n); c = rng.uniform(1, 5, n)
    cols = [a ** 2 / b, np.log(c) * a, a * b, np.log(a) + np.log(b), 1.0 / (a ** 2 * b)]
    while len(cols) < k:
        cols.append(rng.uniform(0, 1, n))
    X = np.column_stack([np.nan_to_num(col.astype(np.float64)) for col in cols])
    y = np.searchsorted(np.quantile(a ** 2 / b, np.linspace(0, 1, n_bins + 1))[1:-1], a ** 2 / b).astype(np.int64)
    return X, y, n_bins


def _tied_fixture(seed=11, n=6000, k=20, n_bins=10):
    """Low-cardinality / tied columns: integers, indicators, sign columns, coarse quantiles -- exactly the
    columns at which rank and edge binning disagree."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 1, n)
    cols = [
        rng.integers(0, 5, n).astype(np.float64),            # 5 distinct values, heavy ties
        (base > 0.5).astype(np.float64),                     # binary indicator
        np.sign(rng.normal(0, 1, n)).astype(np.float64),     # {-1,0,1}
        np.floor(base * 4).astype(np.float64),               # 4 coarse bins
        rng.integers(0, 3, n).astype(np.float64),            # 3 distinct
    ]
    while len(cols) < k:
        cols.append(rng.integers(0, rng.integers(2, 8), n).astype(np.float64))
    X = np.column_stack([np.nan_to_num(col.astype(np.float64)) for col in cols])
    y = rng.integers(0, n_bins, n).astype(np.int64)
    return X, y, n_bins


@pytest.mark.parametrize("fixture", [_continuous_fixture, _tied_fixture], ids=["continuous", "tied"])
def test_cpu_edge_mi_matches_numpy_reference(fixture):
    X, y, nb = fixture()
    cpu_edge = plugin_mi_classif_batch_edge_njit(X, y, nb)
    ref = _np_edge_mi_batch(X, y, nb)
    assert np.allclose(cpu_edge, ref, atol=1e-9, rtol=0), (
        f"CPU edge MI diverged from numpy edge reference: max|d|={np.max(np.abs(cpu_edge - ref)):.3e}"
    )


def test_edge_equals_rank_on_continuous():
    """On continuous (tie-free) data edge binning == rank binning bit-for-bit -> selection-neutral."""
    X, y, nb = _continuous_fixture()
    cpu_edge = plugin_mi_classif_batch_edge_njit(X, y, nb)
    cpu_rank = _plugin_mi_classif_batch_njit(X, y, nb)
    assert np.allclose(cpu_edge, cpu_rank, atol=1e-12, rtol=0), (
        f"edge vs rank must be bit-identical on continuous data: max|d|={np.max(np.abs(cpu_edge - cpu_rank)):.3e}"
    )


def test_edge_differs_from_rank_on_tied():
    """On tied/low-cardinality columns edge and rank DISAGREE -- this is the exact cross-backend
    divergence the edge unification removes; if they ever stop differing here the test is no longer
    exercising the tied case."""
    X, y, nb = _tied_fixture()
    cpu_edge = plugin_mi_classif_batch_edge_njit(X, y, nb)
    cpu_rank = _plugin_mi_classif_batch_njit(X, y, nb)
    assert np.max(np.abs(cpu_edge - cpu_rank)) > 1e-6, (
        "expected rank vs edge to differ on tied columns; fixture no longer ties at bin boundaries"
    )


# ---------------------------------------------------------------------------
# CUDA: CPU edge MI == GPU resident edge MI (the real cross-backend identity).
# ---------------------------------------------------------------------------
def _need_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
@pytest.mark.parametrize("fixture", [_continuous_fixture, _tied_fixture], ids=["continuous", "tied"])
def test_cpu_edge_mi_matches_gpu_resident_edge(fixture):
    import cupy as cp
    from mlframe.feature_selection.filters._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    X, y, nb = fixture()
    cpu_edge = plugin_mi_classif_batch_edge_njit(X, y, nb)
    Xg = cp.asarray(X, dtype=cp.float64)
    yg = cp.asarray(y, dtype=cp.int64)
    gpu_edge = np.asarray(_plugin_mi_classif_batch_cuda_resident(Xg, yg, nb))
    cp.get_default_memory_pool().free_all_blocks()
    assert np.allclose(cpu_edge, gpu_edge, atol=1e-9, rtol=0), (
        f"CPU edge MI != GPU resident edge MI: max|d|={np.max(np.abs(cpu_edge - gpu_edge)):.3e} "
        f"(the two FE-batcher backends must score every column identically)"
    )
