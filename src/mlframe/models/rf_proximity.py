"""Random-forest proximity as an exposed similarity/distance metric + Breiman's outlier measure (PZAD rf lecture).

The random-forest lecture (Дьяконов 2020, slides 2-3, 34) frames RF as a near-universal method that also yields a
LEARNED SIMILARITY: "чем чаще 2 объекта попадают в один лист, тем они ближе" — the Breiman (2001) proximity
``prox(i, j) = (1/n_trees) * #{trees where i and j land in the same leaf}``. Unlike a fixed metric it is supervised
(driven by whatever the forest split on) and needs no feature scaling, so it powers RF-based clustering, MDS,
missing-value imputation and outlier detection — the "+/- кластеризация: можно построить метрику" cell of slide 2.

This is DISTINCT from ``feature_engineering.transformer.rf_proximity`` (which uses boosting-leaf co-occurrence to
weight a per-row TARGET AGGREGATE as a feature): here we expose the full N x N proximity MATRIX as a reusable metric
plus ``rf_outlier_measure`` — for the unsupervised clustering/anomaly use case, not feature creation. ``sklearn`` has
neither (only the raw ``estimator.apply``). The proximity is O(n^2) in memory, so ``rf_proximity_matrix`` guards on a
``max_n`` cap (pass a subsample for large data — the standard practice).
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import numba

    _HAS_NUMBA = True
except Exception:  # numba is an optional accelerator; fall back to numpy
    _HAS_NUMBA = False

logger = logging.getLogger(__name__)

__all__ = ["rf_proximity_matrix", "proximity_to_distance", "rf_outlier_measure"]

_DEFAULT_MAX_N = 20000  # n^2 float64 proximity = ~3.2 GB at 20k; refuse larger by default (subsample instead)


def _leaves_from(estimator, X) -> np.ndarray:
    """Return the (n_samples, n_trees) leaf-index matrix, accepting a fitted forest (``.apply``) or a precomputed array."""
    if hasattr(estimator, "apply") and X is not None:
        leaves = np.asarray(estimator.apply(X))
    else:
        leaves = np.asarray(estimator)  # already a leaf-index matrix
    if leaves.ndim != 2:
        raise ValueError("rf_proximity_matrix: expected a 2D (n_samples, n_trees) leaf-index matrix.")
    return np.ascontiguousarray(leaves.astype(np.int64))


if _HAS_NUMBA:

    @numba.njit(cache=True, parallel=True, nogil=True)
    def _proximity_kernel(leaves: np.ndarray) -> np.ndarray:
        n, t = leaves.shape
        out = np.empty((n, n), dtype=np.float64)
        inv_t = 1.0 / t
        for i in numba.prange(n):
            out[i, i] = 1.0
            for j in range(i + 1, n):
                same = 0
                for k in range(t):
                    if leaves[i, k] == leaves[j, k]:
                        same += 1
                v = same * inv_t
                out[i, j] = v
                out[j, i] = v
        return out
else:

    def _proximity_kernel(leaves: np.ndarray) -> np.ndarray:
        n, t = leaves.shape
        out = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                v = np.count_nonzero(leaves[i] == leaves[j]) / t
                out[i, j] = out[j, i] = v
        return out


def rf_proximity_matrix(estimator, X=None, *, max_n: int = _DEFAULT_MAX_N) -> np.ndarray:
    """Breiman random-forest proximity: the ``n x n`` fraction of trees in which each pair of rows shares a leaf.

    Parameters
    ----------
    estimator : fitted forest with ``.apply`` (RandomForest/ExtraTrees/any bagged tree ensemble) OR a precomputed
        ``(n_samples, n_trees)`` leaf-index matrix.
    X : array-like, optional
        Samples to embed; required when ``estimator`` is a fitted forest, ignored when a leaf matrix is passed.
    max_n : int
        Refuse to allocate the ``n x n`` matrix above this many rows (O(n^2) memory). Subsample for larger data.

    Returns
    -------
    np.ndarray
        Symmetric ``(n, n)`` proximity in [0, 1] with unit diagonal. Convert to a distance with :func:`proximity_to_distance`.
    """
    leaves = _leaves_from(estimator, X)
    n = leaves.shape[0]
    if n > max_n:
        raise ValueError(
            f"rf_proximity_matrix: n={n} exceeds max_n={max_n}; the n x n proximity is O(n^2) memory. " f"Subsample rows or raise max_n explicitly."
        )
    return np.asarray(_proximity_kernel(leaves))


def proximity_to_distance(proximity: np.ndarray) -> np.ndarray:
    """Convert a proximity matrix to a Euclidean-embeddable distance ``sqrt(1 - prox)`` (0 on the diagonal)."""
    prox = np.asarray(proximity, dtype=np.float64)
    dist = np.sqrt(np.clip(1.0 - prox, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return dist


def rf_outlier_measure(proximity: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Breiman's proximity-based outlier score: rows with low total proximity to their own class are outliers.

    For each row ``j`` the raw measure is ``n_class / sum_{k in class(j)} prox(j, k)^2`` (large when ``j`` is far from
    its class in forest space). Raw scores are then standardized WITHIN each class by the class median and MAD, so the
    score is comparable across classes. With ``y=None`` all rows are treated as one class (unsupervised proximity).

    Returns
    -------
    np.ndarray
        Per-row outlier score; higher means more outlying. Breiman flags scores gtrsim 10 as outliers.
    """
    prox = np.asarray(proximity, dtype=np.float64)
    n = prox.shape[0]
    yy = np.zeros(n, dtype=np.int64) if y is None else np.ascontiguousarray(y).astype(np.int64).ravel()
    if yy.shape[0] != n:
        raise ValueError("rf_outlier_measure: y length must match the proximity matrix size.")

    raw = np.empty(n, dtype=np.float64)
    p2 = prox * prox
    for c in np.unique(yy):
        idx = np.where(yy == c)[0]
        sub = p2[np.ix_(idx, idx)]
        s = sub.sum(axis=1) - np.diag(sub)  # exclude self-proximity (=1)
        # Floor the denominator: a row sharing NO leaves with its class (s=0) is maximally outlying; without a floor
        # n/tiny overflows to inf and poisons the class median/MAD. 1e-8 keeps it large-but-finite (raw <= n*1e8).
        raw[idx] = idx.shape[0] / np.maximum(s, 1e-8)

    out = np.empty(n, dtype=np.float64)
    for c in np.unique(yy):
        idx = np.where(yy == c)[0]
        r = raw[idx]
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        scale = mad if mad > 0 else (np.finfo(np.float64).tiny if r.std() == 0 else r.std())
        out[idx] = (r - med) / scale
    return out
