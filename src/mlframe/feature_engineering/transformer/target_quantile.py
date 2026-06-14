"""Target-quantile attention — soft cluster-membership features over target-defined clusters.

Algorithm:
1. Bucket training y into K quantiles (K=10 default). Bucket b contains rows with y in [Q_{b/K}, Q_{(b+1)/K}].
2. Compute the X-space centroid μ_b = mean(X_train[y_in_b]) for each bucket.
3. For each query row x, compute similarity(x, μ_b) for b=1..K (cosine or RBF kernel).
4. Output K features per row — soft assignment to each target-quantile cluster.

Why this is "transformer-like": this is a frozen mixture-of-experts dispatch. Each "expert" is a quantile-defined cluster; the row's features expose which experts
(target ranges) its X-pattern most resembles.

Why this should help boostings:
- Plain row-attention asks "which neighbours have similar X to me?" → aggregates their y.
- Target-quantile attention asks "which target-ranges does my X resemble?" → directly maps X-pattern to target-distribution. The output features are the per-bucket similarities, which boostings can split on (e.g. "is this row similar to high-y cluster?").

Different from kNN-target-encoding: kNN-TE picks k closest neighbours by X-distance and averages their y. Quantile attention picks K predefined target ranges and tells how much each row "belongs" to each range. They capture different aspects of the X-y manifold.

Leakage discipline (Mode A OOF):
- Quantile boundaries are computed PER FOLD on the train subset only.
- Centroids are computed PER FOLD on the train subset's X bucketed by y.
- Validation rows of a fold receive their similarity features from THAT fold's quantile boundaries and centroids — leak-free.

Mode B (X_query!=None):
- Quantile boundaries and centroids computed once on full train.
- X_query receives similarities to those centroids.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numba
import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


@numba.njit(parallel=True, fastmath=False, nogil=True)
def _bucket_sums_counts(X_pool: np.ndarray, y_pool: np.ndarray, edges: np.ndarray):  # pragma: no cover
    """Single fused pass: assign each row to its target-quantile bucket via ``searchsorted`` and accumulate per-bucket X sums + row counts.

    Replaces the K-pass ``(y>=lo)&(y<hi)`` boolean-mask + fancy-index-gather loop in ``_compute_centroids``. The old path walked all N rows K times (K full-length
    boolean temporaries) and gathered ``X_pool[mask]`` K times (K large fancy-index copies). This walks N once, branchlessly bucketing each row, with per-thread
    private accumulators (float64) reduced at the end. Bucket semantics are identical: ``searchsorted(edges, y, side='right')-1`` reproduces the half-open
    ``[lo, hi)`` membership the old code used, with the first/last bucket clamped (matching the old ``b==K-1`` inclusive upper edge and below-min rows landing in
    bucket 0). float64 accumulation makes the centroid means at least as accurate as numpy's float32 reduction; observed divergence ~5.8e-8 (single float32 ULP).
    """
    n, d = X_pool.shape
    K = edges.shape[0] - 1
    nthreads = numba.get_num_threads()
    sums = np.zeros((nthreads, K, d), dtype=np.float64)
    counts = np.zeros((nthreads, K), dtype=np.int64)
    for i in numba.prange(n):
        t = numba.get_thread_id()
        b = np.searchsorted(edges, y_pool[i], side="right") - 1
        if b < 0:
            b = 0
        elif b >= K:
            b = K - 1
        counts[t, b] += 1
        for j in range(d):
            sums[t, b, j] += X_pool[i, j]
    return sums.sum(axis=0), counts.sum(axis=0)


def compute_target_quantile_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Any,
    *,
    seed: int,
    n_quantiles: int = 10,
    similarity: Literal["cosine", "rbf"] = "cosine",
    rbf_gamma: Optional[float] = None,
    standardize: bool = True,
    column_prefix: str = "tq",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Soft cluster-membership features for target-quantile clusters.

    Output shape: ``(N, n_quantiles)``. Each column is the similarity to one target-quantile cluster's X-centroid.

    Parameters:
        ``n_quantiles`` - number of buckets to split y into. 10 is standard; can be lower (e.g. 4) for small N where each bucket needs enough rows for a stable centroid.
        ``similarity`` - "cosine" (default, requires L2-normalised X) or "rbf" (Gaussian kernel; needs ``rbf_gamma`` or it defaults to ``1 / d``).
        ``standardize`` - RobustScaler on X before similarity computation.

    For binary classification, y has only 2 unique values → we still bucket via quantiles, but effectively get 2 clusters (one per class). Use ``n_quantiles=2`` or
    leave default; the function still works but with fewer effective clusters.
    """
    from sklearn.preprocessing import RobustScaler
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2; got {n_quantiles}.")

    if standardize:
        scaler = RobustScaler().fit(X_train)
        X_tr_s = scaler.transform(X_train).astype(dtype, copy=False)
        X_q_s = scaler.transform(X_query).astype(dtype, copy=False) if X_query is not None else None
    else:
        X_tr_s = X_train.astype(dtype, copy=False)
        X_q_s = X_query.astype(dtype, copy=False) if X_query is not None else None

    if similarity == "rbf" and rbf_gamma is None:
        rbf_gamma = 1.0 / max(X_tr_s.shape[1], 1)

    def _compute_centroids(X_pool: np.ndarray, y_pool: np.ndarray) -> np.ndarray:
        """Return (n_quantiles, d) centroids: bucket y_pool into quantiles, take mean X for rows in each bucket."""
        d = X_pool.shape[1]
        # Quantile edges (n_quantiles+1 boundaries including min/max).
        edges = np.quantile(y_pool, np.linspace(0, 1, n_quantiles + 1))
        # Make edges strictly increasing in case of ties (e.g. binary y or skewed continuous).
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9
        sums, counts = _bucket_sums_counts(np.ascontiguousarray(X_pool), np.ascontiguousarray(y_pool), edges.astype(np.float64))
        centroids = np.zeros((n_quantiles, d), dtype=dtype)
        pool_mean = None
        for b in range(n_quantiles):
            if counts[b] == 0:
                # Empty bucket (can happen with ties on binary y); fall back to the previous bucket's centroid or the pool mean.
                if b > 0:
                    centroids[b] = centroids[b - 1]
                else:
                    if pool_mean is None:
                        pool_mean = X_pool.mean(axis=0)
                    centroids[b] = pool_mean
            else:
                centroids[b] = (sums[b] / counts[b]).astype(dtype, copy=False)
        return centroids

    def _similarity_matrix(X_anchor: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Return (n_anchor, n_quantiles) similarity. Cosine or RBF as configured."""
        if similarity == "cosine":
            a_norms = np.linalg.norm(X_anchor, axis=1, keepdims=True)
            c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            a_safe = np.maximum(a_norms, 1e-12)
            c_safe = np.maximum(c_norms, 1e-12)
            return (X_anchor / a_safe) @ (centroids / c_safe).T
        # RBF: exp(-gamma * ||x - c||^2). Vectorise via the identity ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c.
        a_sq = np.einsum("ij,ij->i", X_anchor, X_anchor)
        c_sq = np.einsum("ij,ij->i", centroids, centroids)
        dist_sq = a_sq[:, None] + c_sq[None, :] - 2.0 * (X_anchor @ centroids.T)
        return np.exp(-rbf_gamma * np.maximum(dist_sq, 0.0))

    if X_query is None:
        # Mode A: OOF.
        out = np.zeros((X_tr_s.shape[0], n_quantiles), dtype=dtype)
        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_tr_s)):
            centroids = _compute_centroids(X_tr_s[tr_idx], y_train[tr_idx])
            out[va_idx] = _similarity_matrix(X_tr_s[va_idx], centroids).astype(dtype, copy=False)
    else:
        # Mode B: single-pass with full-train centroids.
        centroids = _compute_centroids(X_tr_s, y_train)
        out = _similarity_matrix(X_q_s, centroids).astype(dtype, copy=False)

    names = [f"{column_prefix}_q{b}" for b in range(n_quantiles)]
    return pl.DataFrame({name: out[:, i] for i, name in enumerate(names)})
