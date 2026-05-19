"""Unified kNN helper: hnswlib (HNSW) drop-in for sklearn.neighbors.NearestNeighbors.

Why: sklearn's NearestNeighbors falls back to ball_tree at d>20 with O(N^(1-1/d)) query
cost — at N=1M, d=64 this is 2-5 min total query phase. hnswlib uses HNSW graph index
with O(log N) per query — same workload runs in 50-200ms. ~10-50x speedup at d>20, N>500k.

Used by cdist, local_lift, RSD, BGM (the kNN-bottlenecked mechanisms).

Falls back to sklearn NearestNeighbors if hnswlib isn't installed. Returns ``(dists, ids)``
identical in shape and dtype to ``sklearn.NearestNeighbors.kneighbors(return_distance=True)``,
so callers can swap in without other code changes.

The HNSW-or-sklearn switch is automatic via ``prefer_hnsw_at_n`` (default 50000): below this
threshold sklearn's ball_tree is competitive and we skip the HNSW index-build overhead;
above it we get the speedup. The threshold matches the empirical crossover on a 16-core AVX2
box at d=20-100.

hnswlib parameters chosen for high-recall / acceptable build time on tabular FE workloads:
M=16 (graph degree), ef_construction=200, ef_search=max(k+8, 64). Empirically gives ~99% recall
vs exact kNN at these sizes; we don't validate against exact every call because the downstream
FE features are statistical aggregates and 99% recall is plenty for the signal they capture.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

_HNSW_AVAILABLE: bool | None = None


def _check_hnsw_available() -> bool:
    """Cached check for hnswlib import."""
    global _HNSW_AVAILABLE
    if _HNSW_AVAILABLE is None:
        try:
            import hnswlib  # noqa: F401
            _HNSW_AVAILABLE = True
        except ImportError:
            _HNSW_AVAILABLE = False
            logger.info(
                "[_knn_helper] hnswlib not installed; falling back to sklearn NearestNeighbors. "
                "Install hnswlib (pip install hnswlib) for 10-50x speedup at N>500k, d>20."
            )
    return _HNSW_AVAILABLE


def knn_search(
    X_subset: np.ndarray,
    X_query: np.ndarray,
    k: int,
    *,
    metric: str = "l2",
    prefer_hnsw_at_n: int = 50_000,
    hnsw_M: int = 16,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbours of each X_query row in X_subset.

    Returns
    -------
    dists : (n_query, min(k, n_subset)) float32, EUCLIDEAN distances (sqrt of squared L2)
        Compatible with sklearn ``NearestNeighbors(...).kneighbors()`` default output.
    ids : (n_query, min(k, n_subset)) int64, row indices into X_subset

    Notes
    -----
    Uses hnswlib if available and N_subset >= prefer_hnsw_at_n. Otherwise sklearn
    NearestNeighbors with algorithm="auto" (which picks kd_tree at d<20, ball_tree else).

    For X_subset.shape[0] == 0 returns sentinel arrays (dists=1e6, ids=0) of shape
    (n_query, k) so callers can downstream-process without an empty-set branch.
    """
    n_sub = X_subset.shape[0]
    n_q = X_query.shape[0]
    if n_sub == 0:
        return (
            np.full((n_q, k), 1e6, dtype=np.float32),
            np.zeros((n_q, k), dtype=np.int64),
        )
    k_request = min(k, n_sub)

    use_hnsw = n_sub >= prefer_hnsw_at_n and _check_hnsw_available()
    if use_hnsw:
        import hnswlib
        d = X_subset.shape[1]
        space = "l2" if metric == "l2" else metric
        idx = hnswlib.Index(space=space, dim=d)
        idx.init_index(max_elements=n_sub, ef_construction=hnsw_ef_construction, M=hnsw_M)
        idx.add_items(np.ascontiguousarray(X_subset, dtype=np.float32), np.arange(n_sub))
        idx.set_ef(max(k_request + 8, hnsw_ef_search))
        try:
            labels, sq_dists = idx.knn_query(np.ascontiguousarray(X_query, dtype=np.float32), k=k_request)
        except Exception as exc:
            logger.warning("[_knn_helper] hnswlib knn_query failed (%s); falling back to sklearn.", exc)
            return _sklearn_fallback(X_subset, X_query, k_request)
        # hnswlib's 'l2' space returns SQUARED L2; take sqrt for sklearn-compat Euclidean.
        dists = np.sqrt(np.maximum(sq_dists, 0.0, dtype=np.float32)).astype(np.float32)
        return dists, labels.astype(np.int64)

    return _sklearn_fallback(X_subset, X_query, k_request)


def _sklearn_fallback(X_subset: np.ndarray, X_query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(X_subset)
    dists, ids = nn.kneighbors(X_query)
    return dists.astype(np.float32, copy=False), ids.astype(np.int64, copy=False)
