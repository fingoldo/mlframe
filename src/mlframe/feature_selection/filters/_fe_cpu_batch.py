"""CPU FE-batcher path: score a resident (n, K) candidate matrix by edge-binned plug-in MI (2026-06-26).

The CPU half of the two separate, independently-optimised FE-scoring backends (the GPU half is
``_fe_gpu_batch``). It is deliberately simple: the host has ONE compute unit and large RAM, so there is
no device loop and no VRAM packer -- cores are exploited by the njit ``parallel=True`` prange INSIDE the
MI kernel (``_fe_edge_mi.plugin_mi_classif_batch_edge_njit``), and the only budgeting needed is an
optional column-chunk so a very wide candidate matrix does not spike the transient MI working set beyond
the host RAM envelope (``feature_engineering._fe_effective_buffer_budget_bytes``).

It uses the SAME percentile-edge binning + plain plug-in MI as the GPU resident path, so the two backends
score every column identically (``test_fe_batch_parity``). Partial-NaN columns are masked to their finite
subset per column, matching ``_orth_mi_backends._mi_classif_batch`` semantics; an all-NaN or single-bin
column returns 0.0.
"""
from __future__ import annotations

import numpy as np

from ._fe_edge_mi import _plugin_mi_classif_edge_njit, plugin_mi_classif_batch_edge_njit


def _available_ram_bytes() -> int:
    """Host available RAM in bytes, or -1 when psutil is absent (callers treat -1 as 'no cap')."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return -1


def _cpu_col_chunk(n: int, n_cols: int, *, n_workers: int = 1) -> int:
    """How many candidate columns to score in ONE prange call so the transient MI working set stays inside
    the host RAM budget. The resident matrix is already in RAM; this only bounds the per-call scratch
    (per-thread code buffer + histograms ~ a small multiple of n*8). Returns ``n_cols`` (no chunk) when no
    budget is resolvable -- the common case, since the working set is tiny relative to RAM."""
    from .feature_engineering import _fe_effective_buffer_budget_bytes

    budget = _fe_effective_buffer_budget_bytes(_available_ram_bytes(), n_workers=n_workers)
    if budget is None or budget < 0:
        return int(n_cols)
    # Per-column transient: the contiguous f64 column copy (n*8) plus the codes/hist scratch (~n*8). Two
    # f64-column-equivalents per column is a safe envelope for the prange scratch held concurrently.
    per_col = max(1, n) * 16
    fit = int(budget // per_col)
    if fit < 1:
        fit = 1
    return min(int(n_cols), fit)


def cpu_fe_batch_mi(
    X_cands: np.ndarray,
    y_codes: np.ndarray,
    nbins: int = 10,
    *,
    n_workers: int = 1,
    max_cols_per_chunk: int | None = None,
) -> np.ndarray:
    """Edge-binned plain plug-in MI of every column of ``X_cands`` (n, K) against discrete ``y_codes`` (n,).

    Identical math to the GPU resident edge MI -> the two FE-batcher backends select the same forms. Cores
    are used by the prange inside the kernel; ``max_cols_per_chunk`` (default: RAM-budget-resolved) caps the
    transient scratch for very wide matrices. Returns a host (K,) float64 MI array.
    """
    X = np.ascontiguousarray(X_cands, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    out = np.zeros(k, dtype=np.float64)
    if n == 0 or k == 0:
        return out
    y = np.ascontiguousarray(y_codes, dtype=np.int64)

    # Per-column finite partition (mirrors _mi_classif_batch): dense columns go through the batch kernel;
    # partial-NaN columns are scored on their finite subset; all-NaN -> 0.0.
    finite_all = np.isfinite(X).all(axis=0)
    dense = np.where(finite_all)[0]
    partial = np.where(~finite_all)[0]

    if dense.size:
        Xd = X if dense.size == k else np.ascontiguousarray(X[:, dense])
        chunk = int(max_cols_per_chunk) if max_cols_per_chunk else _cpu_col_chunk(n, dense.size, n_workers=n_workers)
        if chunk >= dense.size:
            out[dense] = plugin_mi_classif_batch_edge_njit(Xd, y, nbins)
        else:
            for s in range(0, dense.size, chunk):
                sl = slice(s, min(s + chunk, dense.size))
                out[dense[sl]] = plugin_mi_classif_batch_edge_njit(np.ascontiguousarray(Xd[:, sl]), y, nbins)

    for j in partial:
        col = X[:, j]
        mask = np.isfinite(col)
        if not mask.any():
            out[j] = 0.0
            continue
        out[j] = _plugin_mi_classif_edge_njit(np.ascontiguousarray(col[mask]), y[mask], nbins)
    return out
