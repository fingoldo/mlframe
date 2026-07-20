"""Sliced Inverse Regression (SIR) oblique-direction projection feature (mrmr_audit_2026-07-20
fe_expansion.md "Sliced Inverse Regression (SIR) oblique-direction projection feature").

Li (1991, "Sliced Inverse Regression for Dimension Reduction"): slice ``y`` into ``H`` bins,
compute the per-slice mean of ``X``, form the between-slice-mean covariance matrix
``M = Cov(E[X | slice])``, then solve the generalized eigenproblem ``Sigma^{-1} M v = lambda v``
(``Sigma`` = overall covariance of ``X``); the top eigenvector(s) ``v`` give the LINEAR COMBINATION
direction(s) ``w.x`` along which ``y`` varies most -- an effective dimension-reduction direction,
not restricted to any 2 or 3 named columns.

Why this catches a shape the catalog misses: ``y = 1{0.6*x1 + 0.5*x2 + 0.4*x3 + 0.3*x4 + 0.4*x5 >
c}`` -- a genuinely OBLIQUE (rotated) threshold spread thinly across 5 correlated columns, where
EVERY individual weight is too small for that column's marginal MI to clear the screening floor,
and no pairwise/triplet/quadruplet product of any 2-4 of the 5 columns reconstructs the linear
combination (axis-aligned bases multiplied together cannot represent a rotated hyperplane
economically). SIR recovers the direction ``w = (0.6, 0.5, 0.4, 0.3, 0.4)`` directly as a single
new feature ``w.x``, after which the existing argmax/gate/threshold machinery (or a plain MI
screen) picks it up trivially.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from ._mi_greedy_cmi_fe import _quantile_bin

__all__ = ["sir_direction_features"]


def sir_direction_features(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_slices: int = 10,
    n_directions: int = 2,
) -> np.ndarray:
    """Sliced Inverse Regression projection features: the top ``n_directions`` SIR eigenvectors'
    projections of ``X``, as new ``(n, n_directions)`` columns.

    Parameters
    ----------
    X : (n, p) array
        Candidate numeric columns to jointly project (correlated columns are exactly where SIR's
        oblique direction beats any per-column or product-of-per-column basis).
    y : (n,) array
        Continuous or discrete target; sliced into ``n_slices`` equi-frequency bins (reusing the
        existing ``_quantile_bin`` helper) regardless of whether it is a classification or
        regression target -- SIR's own construction only needs a slicing, not a native
        classification/regression distinction.
    n_slices : int
        Number of equi-frequency slices of ``y``. Li (1991)'s own guidance: more slices resolve
        finer structure but each slice needs enough rows for a stable per-slice mean; the default
        10 is the standard textbook choice.
    n_directions : int
        Number of top eigenvector directions to project onto and return as columns.

    Returns
    -------
    (n, n_directions) float64 array of projections ``X @ v_1, ..., X @ v_{n_directions}``.
    Degenerate input (n < 2, p < 1, fewer than 2 realized slices, or a singular/near-singular
    ``Sigma``) returns an ``(n, 0)`` array rather than raising -- callers treat zero emitted
    directions as "nothing to add".
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    y = np.asarray(y, dtype=np.float64).ravel()
    if n < 2 or p < 1 or y.size != n or n_directions < 1:
        return np.empty((n, 0), dtype=np.float64)
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        return np.empty((n, 0), dtype=np.float64)

    slice_ids = _quantile_bin(y, nbins=n_slices)
    uniq_slices = np.unique(slice_ids)
    if uniq_slices.size < 2:
        return np.empty((n, 0), dtype=np.float64)

    x_mean = X.mean(axis=0)
    Xc = X - x_mean
    Sigma = (Xc.T @ Xc) / n

    # Between-slice-mean covariance M = sum_h (n_h/n) * (mean_h - x_mean) @ (mean_h - x_mean)^T.
    M = np.zeros((p, p), dtype=np.float64)
    for s in uniq_slices:
        mask = slice_ids == s
        n_h = int(mask.sum())
        if n_h < 1:
            continue
        slice_mean_dev = X[mask].mean(axis=0) - x_mean
        M += (n_h / n) * np.outer(slice_mean_dev, slice_mean_dev)

    # Ridge-stabilize Sigma so the generalized eigenproblem stays solvable even when X's columns
    # are exactly collinear (a genuinely singular Sigma) -- a tiny trace-scaled shift that does not
    # move a well-conditioned Sigma's eigenvectors materially, mirroring the same trace-scaled-ridge
    # convention used elsewhere in this codebase (e.g. _fe_pure_form_retention_gpu_resident.py).
    trace = float(np.trace(Sigma))
    if trace <= 1e-12:
        return np.empty((n, 0), dtype=np.float64)
    Sigma_ridge = Sigma + (1e-8 * trace / p) * np.eye(p)

    try:
        eigvals, eigvecs = scipy.linalg.eigh(M, Sigma_ridge)
    except (scipy.linalg.LinAlgError, ValueError):
        return np.empty((n, 0), dtype=np.float64)

    # scipy.linalg.eigh returns ascending eigenvalues; SIR wants the LARGEST (most y-variation).
    order = np.argsort(eigvals)[::-1]
    k = min(n_directions, p)
    top_dirs = eigvecs[:, order[:k]]
    return np.asarray(Xc @ top_dirs)
