"""Layer 72 (mrmr_audit_2026-07-20 fe_expansion.md): Chatterjee's Xi rank-correlation dependence
scorer for the auto-scorer pool (``_orth_auto_scorer_fe.py`` / ``_orthogonal_scorer_auto_fe.py``).

Chatterjee (2021, "A New Coefficient of Correlation", JASA) defines::

    xi(X, Y) = 1 - 3 * sum_i |r_{i+1} - r_i| / (n^2 - 1)

where ``r_i`` is the rank of ``y`` reordered by ascending ``x`` (ties in ``x`` broken by a random
permutation, per Chatterjee's own construction, so the estimator stays well-defined on discrete/
low-cardinality ``x``). Xi is asymptotically 0 iff X and Y are independent and 1 iff Y is a
MEASURABLE FUNCTION of X (not merely monotone, unlike Spearman) -- a genuinely different
construction from every scorer already in the pool (plug-in MI is quantile-binning-based, KSG is
kNN-distance-based, copula-MI is rank-uniformized-MI, dCor is U-centred-distance-matrix-based,
HSIC is RKHS-kernel-based).

Why this catches a shape the catalog misses: on a smooth but highly-OSCILLATORY target such as
``y = sin(20*x) + noise``, plug-in MI's fixed quantile bins average many oscillation cycles into
each bin (near-zero MI), KSG's kNN balls at moderate k similarly smear across cycles, and dCor/HSIC
(calibrated to a GLOBAL scale) underweight the fine local structure. Xi's sort-then-walk
construction is scale-free and directly sees every local up/down rank flip in y, so it stays high
at oscillation frequencies where every distance/kernel/binning scorer decays toward the null floor.

Cost: O(n log n) (one argsort), not O(n^2) like dCor/HSIC -- no subsampling needed even at large n.
"""

from __future__ import annotations

import numpy as np

__all__ = ["xi_correlation", "xi_correlation_batch"]


def xi_correlation(x: np.ndarray, y: np.ndarray, *, random_state: int = 0) -> float:
    """Chatterjee's Xi correlation coefficient between two 1-D arrays.

    Ties in ``x`` are broken by a random permutation (Chatterjee's own construction) so the
    estimator stays well-defined on discrete/low-cardinality ``x`` rather than depending on
    argsort's tie-breaking order. Returns 0.0 on degenerate input (n < 2, or ``y`` constant, since
    a constant y has zero variation to detect regardless of x).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n < 2 or y.size != n:
        return 0.0
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return 0.0
    if float(np.std(y)) <= 1e-12:
        return 0.0
    rng = np.random.default_rng(random_state)
    # Break x-ties randomly (Chatterjee 2021, Section 2): a stable/deterministic tie-break would
    # systematically bias the walk direction on tied runs; a random tie-break makes the estimator
    # correct in expectation on discrete/repeated-value x.
    perm = rng.permutation(n)
    order = np.lexsort((perm, x))
    y_ordered = y[order]
    ranks = np.argsort(np.argsort(y_ordered, kind="stable"), kind="stable").astype(np.float64) + 1.0
    xi = 1.0 - 3.0 * float(np.sum(np.abs(np.diff(ranks)))) / (n**2 - 1)
    return float(xi)


def xi_correlation_batch(X: np.ndarray, y: np.ndarray, *, random_state: int = 0) -> np.ndarray:
    """Vectorized ``xi_correlation`` for every column of ``X`` (shape ``(n, K)``) against the same
    ``y`` -- avoids re-deriving ``y``'s own rank array K times (the walk depends only on the
    ORDER x induces on y, so the ``y``-rank computation per column is the dominant repeated cost
    otherwise). Returns a ``(K,)`` float64 array."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    y = np.asarray(y, dtype=np.float64).ravel()
    out = np.zeros(k, dtype=np.float64)
    if n < 2 or y.size != n or float(np.std(y)) <= 1e-12:
        return out
    if not np.isfinite(y).all():
        return out
    rng = np.random.default_rng(random_state)
    for j in range(k):
        col = X[:, j]
        if not np.isfinite(col).all():
            continue
        perm = rng.permutation(n)
        order = np.lexsort((perm, col))
        y_ordered = y[order]
        ranks = np.argsort(np.argsort(y_ordered, kind="stable"), kind="stable").astype(np.float64) + 1.0
        out[j] = 1.0 - 3.0 * float(np.sum(np.abs(np.diff(ranks)))) / (n**2 - 1)
    return out
