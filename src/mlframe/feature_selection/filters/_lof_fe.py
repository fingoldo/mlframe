"""Local Outlier Factor / k-NN local density-ratio feature (mrmr_audit_2026-07-20
fe_expansion.md "Local Outlier Factor / k-NN local density-ratio feature").

Breunig et al. (2000, "LOF: Identifying Density-Based Local Outliers"): for each row, compute the
ratio of its local k-NN density to the AVERAGE local density of its k neighbors (reachability-
distance based); rows in locally sparse regions relative to their neighborhood score high even
when the neighborhood itself is not globally extreme.

Why this catches a shape the catalog misses: distinct from a global elliptical/Gaussian anomaly
score (Mahalanobis distance, which assumes a single global covariance shape), LOF is LOCAL and
non-parametric -- it catches anomalies in a MULTI-MODAL joint distribution (e.g. several
well-separated Gaussian clusters of normal behavior, where a row is anomalous for sitting in a
locally-sparse gap BETWEEN clusters even though its raw distance to the GLOBAL mean/covariance is
unremarkable, since the global Mahalanobis ellipsoid straddles all clusters and a between-cluster
point can have a perfectly ordinary global Mahalanobis distance).
"""

from __future__ import annotations

import numpy as np

__all__ = ["lof_scores"]


def _pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """(n, n) squared Euclidean distance matrix via the matmul trick ``||a-b||^2 = ||a||^2 +
    ||b||^2 - 2*a.b``; the diagonal is forced to +inf so a row never counts itself as a neighbor."""
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)
    return np.asarray(d2)


def lof_scores(X: np.ndarray, *, k: int = 20) -> np.ndarray:
    """Local Outlier Factor score per row of ``X``.

    Parameters
    ----------
    X : (n, p) array
        Numeric columns to jointly compute the local density ratio over.
    k : int
        Neighborhood size. Clamped to ``n - 1`` when ``n`` is small (fewer than ``k + 1`` rows) so
        the function degrades gracefully rather than requesting more neighbors than exist.

    Returns
    -------
    (n,) float64 array of LOF scores. ``LOF ~= 1`` means a row's local density matches its
    neighbors' (ordinary); ``LOF >> 1`` means a row sits in a locally sparse region relative to its
    neighborhood (a local outlier). Degenerate input (n < 3, non-finite X) returns an all-NaN
    ``(n,)`` array rather than raising.

    Cost: brute-force O(n^2) pairwise distances (the matmul trick) -- fine for moderate n; very
    large n (>~200k) would need chunking or an approximate k-NN library, per the audit's own note.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    if n < 3:
        return np.full(n, np.nan, dtype=np.float64)
    if not np.isfinite(X).all():
        return np.full(n, np.nan, dtype=np.float64)

    k_eff = min(k, n - 1)
    if k_eff < 1:
        return np.full(n, np.nan, dtype=np.float64)

    d2 = _pairwise_sq_dist(X)
    dist = np.sqrt(d2)

    # k nearest neighbor indices per row (unordered within the k-set; LOF's own formula only needs
    # the SET of k-NN and each pairwise distance within it, not a strict rank ordering).
    nn_idx = np.argpartition(dist, k_eff - 1, axis=1)[:, :k_eff]
    nn_dist = np.take_along_axis(dist, nn_idx, axis=1)
    k_distance = nn_dist.max(axis=1)  # (n,) distance to the k-th nearest neighbor of each row

    # reach-dist_k(p, o) = max(k-distance(o), dist(p, o)) for each of p's k neighbors o.
    reach_dist = np.maximum(k_distance[nn_idx], nn_dist)  # (n, k_eff)
    lrd = 1.0 / np.maximum(reach_dist.mean(axis=1), 1e-12)  # local reachability density per row

    # LOF(p) = mean over p's k neighbors o of lrd(o) / lrd(p).
    lof = lrd[nn_idx].mean(axis=1) / np.maximum(lrd, 1e-12)
    return np.asarray(lof)
