"""Layer 72: Chatterjee's Xi rank-correlation dependence
scorer for the auto-scorer pool (``_orth_auto_scorer_fe.py`` / ``_orthogonal_scorer_auto_fe.py``).

Chatterjee (2021, "A New Coefficient of Correlation", JASA) defines the TIE-CORRECTED estimator::

    xi(X, Y) = 1 - n * sum_i |r_{i+1} - r_i| / (2 * sum_i l_i (n - l_i))

with ``r_i = #{j: y_j <= y_(i)}`` and ``l_i = #{j: y_j >= y_(i)}`` (both counting ties), which reduces to
the familiar no-ties form ``1 - 3 * sum_i |r_{i+1} - r_i| / (n^2 - 1)`` when ``y`` has no ties but stays
unbiased on tied/discrete ``y`` (e.g. any classification target),
where ``r_i`` is the rank of ``y`` reordered by ascending ``x`` (ties in ``x`` broken by a random
permutation, per Chatterjee's own construction, so the estimator stays well-defined on discrete/
low-cardinality ``x``). Xi is asymptotically 0 iff X and Y are independent and 1 iff Y is a
MEASURABLE FUNCTION of X (not merely monotone, unlike Spearman) - a genuinely different
construction from every scorer already in the pool (plug-in MI is quantile-binning-based, KSG is
kNN-distance-based, copula-MI is rank-uniformized-MI, dCor is U-centred-distance-matrix-based,
HSIC is RKHS-kernel-based).

Why this catches a shape the catalog misses: on a smooth but highly-OSCILLATORY target such as
``y = sin(20*x) + noise``, plug-in MI's fixed quantile bins average many oscillation cycles into
each bin (near-zero MI), KSG's kNN balls at moderate k similarly smear across cycles, and dCor/HSIC
(calibrated to a GLOBAL scale) underweight the fine local structure. Xi's sort-then-walk
construction is scale-free and directly sees every local up/down rank flip in y, so it stays high
at oscillation frequencies where every distance/kernel/binning scorer decays toward the null floor.

Cost: O(n log n) (one argsort), not O(n^2) like dCor/HSIC - no subsampling needed even at large n.
"""

from __future__ import annotations

import numpy as np

__all__ = ["xi_correlation", "xi_correlation_batch"]


def _xi_from_order(y_ordered: np.ndarray) -> float:
    """Chatterjee's Xi from ``y`` already reordered by ascending ``x``, using the TIE-CORRECTED estimator.

    ``xi = 1 - n * sum_i |r_{i+1} - r_i| / (2 * sum_i l_i (n - l_i))`` where ``r_i = #{j: y_j <= y_(i)}``
    (right-rank, counts ties) and ``l_i = #{j: y_j >= y_(i)}``. This reduces to the no-ties form
    ``1 - 3*sum|dr|/(n^2-1)`` when y has no ties, but on tied/discrete y (any classification target) the
    no-ties form's forced-distinct ranks and ``n^2-1`` denominator biased Xi toward 0.
    """
    n = y_ordered.size
    if n < 2:
        return 0.0
    _uniq, inv, counts = np.unique(y_ordered, return_counts=True, return_inverse=True)
    le = np.cumsum(counts)  # per unique-group: #{y <= group value}
    ge = np.cumsum(counts[::-1])[::-1]  # per unique-group: #{y >= group value}
    r = le[inv].astype(np.float64)  # r_i, already in x-induced order (y_ordered is x-sorted)
    ll = ge[inv].astype(np.float64)  # l_i
    num = float(n) * float(np.sum(np.abs(np.diff(r))))
    den = 2.0 * float(np.sum(ll * (n - ll)))
    return float(1.0 - num / den) if den > 0.0 else 0.0


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
    return _xi_from_order(y_ordered)


def xi_correlation_batch(X: np.ndarray, y: np.ndarray, *, random_state: int = 0) -> np.ndarray:
    """Vectorized ``xi_correlation`` for every column of ``X`` (shape ``(n, K)``) against the same
    ``y`` - avoids re-deriving ``y``'s own rank array K times (the walk depends only on the
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
        out[j] = _xi_from_order(y_ordered)
    return out
