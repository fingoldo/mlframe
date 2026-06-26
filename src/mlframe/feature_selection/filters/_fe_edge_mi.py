"""Edge-binning plain plug-in MI for the CPU FE-batcher path -- the bit-faithful CPU twin of the GPU
edge plug-in MI (2026-06-26).

The orthogonal/basis FE families score candidates by PLAIN plug-in MI(col; y). The legacy CPU kernel
(``_hermite_fe_mi._plugin_mi_classif_njit``) bins x by RANK (argsort -> n/nbins rows per bin); the GPU
kernel (``_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident``) bins by PERCENTILE EDGE
(``_radix_select_interior_edges`` / cp.percentile + searchsorted). On continuous data the two are
bit-identical, but on TIED / low-cardinality columns they disagree (rank splits equal values across a bin
boundary, edge keeps them together) -- which, under full GPU residency, would let the two backends select
different forms. This module is the CPU edge-binning twin so the CPU batcher path and the GPU batcher path
score candidates IDENTICALLY on EVERY column, tied or not. See ``_fe_mi_contract`` for the full contract.

Self-contained leaf (numpy + numba only): it reproduces the two canonical, separately-tested algorithms --
the equi-frequency percentile-edge binning of ``_usability_njit_pool._qbin_into`` (np.quantile lerp edges
at virtual index ``q*(n-1)`` + np.unique dedup + ``searchsorted(interior, side='right')``) and the plain
plug-in MI summation of ``_hermite_fe_mi._plugin_mi_from_binned_njit`` -- so it carries no import-cycle
risk. The ``test_fe_edge_mi_parity`` suite pins it bit-equal (to ~1e-9) to a numpy reference AND, on a CUDA
host, to the GPU resident edge MI, on continuous AND tied fixtures.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, inline="always")
def _edge_bin_codes(val: np.ndarray, n_bins: int, codes_out: np.ndarray) -> int:
    """Equi-frequency PERCENTILE-EDGE bin ``val`` (finite float64) into ``codes_out`` (int32, in
    [0, n_bins)). Matches the GPU orth twin ``_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident``
    EXACTLY: the FIXED ``n_bins-1`` interior edges ``cp.percentile(X, qs)[1:-1]`` (np.quantile linear
    interpolation at virtual index ``q*(n-1)``) with NO dedup, then ``searchsorted(interior, side='right')``.

    NOTE the deliberate difference from ``_usability_njit_pool._qbin_into`` (the usability-pool binner),
    which np.unique-DEDUPS the edges and so collapses tied bins to a variable cardinality. The orth/basis
    family's GPU kernel does NOT dedup -- it keeps all n_bins-1 interior edges -- so on tied / duplicate
    values the two conventions assign different codes. This twin follows the GPU (no dedup) so the CPU and
    GPU batcher backends score the orth family identically on tied columns (verified ~1e-9, the
    test_fe_edge_mi_parity 'tied' case). Returns ``n_bins`` (the fixed bin count)."""
    n = val.shape[0]
    nq = n_bins + 1
    # Edges need only the order statistics at the lerp anchors (lo, lo+1) of each of the nq quantile
    # points -- ~2*nq positions, not the whole sorted array. np.partition places EXACTLY those positions
    # at their sorted values in O(n) (introselect) vs the O(n log n) full sort; part[lo]/part[hi] are then
    # the identical order statistics np.sort would give, so the edges (and MI) are BIT-IDENTICAL.
    los = np.empty(nq, dtype=np.int64)
    fracs = np.empty(nq, dtype=np.float64)
    kths = np.empty(2 * nq, dtype=np.int64)
    m = 0
    for k in range(nq):
        pos = (k / (nq - 1)) * (n - 1)  # quantile fraction k/n_bins -> virtual index q*(n-1)
        lo = int(np.floor(pos))
        hi = lo + 1 if lo < n - 1 else lo
        los[k] = lo
        fracs[k] = pos - lo
        kths[m] = lo; kths[m + 1] = hi; m += 2
    part = np.partition(val, kths[:m])
    edges = np.empty(nq, dtype=np.float64)
    for k in range(nq):
        lo = los[k]
        hi = lo + 1 if lo < n - 1 else lo
        edges[k] = part[lo] + (part[hi] - part[lo]) * fracs[k]
    ni = n_bins - 1  # interior edges live at edges[1 .. n_bins-1]; searchsorted-right via bisect
    if ni <= 0:
        for i in range(n):
            codes_out[i] = 0
        return n_bins
    for i in range(n):
        v = val[i]
        lo = 0
        hi = ni
        while lo < hi:
            mid = (lo + hi) // 2
            if v < edges[1 + mid]:
                hi = mid
            else:
                lo = mid + 1
        codes_out[i] = lo
    return n_bins


@njit(cache=True, fastmath=True)
def _plugin_mi_classif_edge_njit(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Plain plug-in MI of continuous x (1-D float64) with discrete y (1-D int64), EDGE-binning x.
    Same plug-in estimator (nats) and occupied-cell reduction order as
    ``_hermite_fe_mi._plugin_mi_classif_njit`` -- only the x binning is percentile-edge, not rank."""
    n = x.shape[0]
    if n == 0:
        return 0.0
    y_min = y[0]
    y_max = y[0]
    for i in range(1, n):
        if y[i] < y_min:
            y_min = y[i]
        if y[i] > y_max:
            y_max = y[i]
    n_classes = (y_max - y_min) + 1

    codes = np.empty(n, dtype=np.int32)
    _edge_bin_codes(x, n_bins, codes)

    hist_xy = np.zeros((n_bins, n_classes), dtype=np.int64)
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        b = codes[i]
        c = y[i] - y_min
        hist_xy[b, c] += 1
        hist_x[b] += 1
        hist_y[c] += 1

    log_n = np.log(n)
    mi = 0.0
    for b in range(n_bins):
        if hist_x[b] == 0:
            continue
        log_hx = np.log(hist_x[b])
        for c in range(n_classes):
            n_xy = hist_xy[b, c]
            if n_xy == 0 or hist_y[c] == 0:
                continue
            mi += (n_xy / n) * (np.log(n_xy) + log_n - log_hx - np.log(hist_y[c]))
    if mi < 0.0:
        mi = 0.0
    return mi


@njit(cache=True, fastmath=True, parallel=True)
def plugin_mi_classif_batch_edge_njit(X_cols: np.ndarray, y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Edge-binning plain plug-in MI of each column of ``X_cols`` (continuous, (n,k) float64) with discrete
    y. Parallel over columns. The CPU batcher's per-column MI core; identical (to ~fp reduction order) to
    the GPU resident edge plug-in MI, so the two batcher backends select the same forms on every column."""
    k = X_cols.shape[1]
    out = np.zeros(k, dtype=np.float64)
    for j in prange(k):
        out[j] = _plugin_mi_classif_edge_njit(np.ascontiguousarray(X_cols[:, j]), y, n_bins)
    return out
