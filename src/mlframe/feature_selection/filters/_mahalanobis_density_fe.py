"""Multivariate Mahalanobis / Gaussian-copula joint density anomaly score (mrmr_audit_2026-07-20
fe_expansion.md "Multivariate Mahalanobis / Gaussian-copula joint density anomaly score").

Computes a single new feature ``d(row) = sqrt((x-mu)^T Sigma^-1 (x-mu))`` over a correlated
cluster (or all) of numeric raw columns jointly -- the classical multivariate-normal quadratic
form, with the mean/covariance Ledoit-Wolf shrunk (reused from ``sklearn.covariance.LedoitWolf``,
not reimplemented) to avoid p-close-to-n ill-conditioning.

Why this catches a shape the catalog misses: y can depend on whether a row sits inside or outside
an ELLIPSOIDAL level-set of the joint distribution of p=15-30 correlated numeric columns (e.g. a
multivariate process-control / fraud "jointly-typical vs jointly-atypical" target) where NO single
column, pair, triplet, or even a quadruplet arity-4 cross-basis is individually extreme -- each
column can sit comfortably within its own marginal range while the JOINT combination is far in
Mahalanobis distance (the classic multivariate-outlier-invisible-to-univariate-checks scenario).
The existing group_distance / conditional-dispersion families condition one column's deviation on
ONE other (binned) column; this is the p-way generalization using the FULL covariance structure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf

__all__ = ["mahalanobis_density_feature"]


def mahalanobis_density_feature(
    X: np.ndarray,
    *,
    X_fit: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mahalanobis distance of every row of ``X`` to a Ledoit-Wolf shrunk mean/covariance.

    Parameters
    ----------
    X : (n, p) array
        Rows to score.
    X_fit : (n_fit, p) array, optional
        The rows to fit ``mu``/``Sigma`` on. ``None`` (default) fits on ``X`` itself. Pass the
        TRAIN rows explicitly and ``X`` as the full (train+test) set for a leak-safe fit-once/
        apply-to-all-rows contract, mirroring the existing K-fold-fit-then-apply discipline used
        elsewhere in this codebase.

    Returns
    -------
    (n,) float64 array of Mahalanobis distances (``>= 0``). Degenerate input (n_fit < p+1, p < 1,
    non-finite X or X_fit) returns an all-NaN ``(n,)`` array rather than raising -- Ledoit-Wolf
    itself needs at least a modest sample-to-dimension ratio to shrink meaningfully.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    fit_arr = np.asarray(X_fit, dtype=np.float64) if X_fit is not None else X
    if fit_arr.ndim == 1:
        fit_arr = fit_arr[:, None]
    n_fit = fit_arr.shape[0]

    if p < 1 or n_fit < p + 1:
        return np.full(n, np.nan, dtype=np.float64)
    if not (np.isfinite(X).all() and np.isfinite(fit_arr).all()):
        return np.full(n, np.nan, dtype=np.float64)

    lw = LedoitWolf().fit(fit_arr)
    mu = lw.location_
    Sigma_inv = lw.get_precision()

    delta = X - mu
    d2 = np.einsum("ni,ij,nj->n", delta, Sigma_inv, delta)
    return np.asarray(np.sqrt(np.maximum(d2, 0.0)))
