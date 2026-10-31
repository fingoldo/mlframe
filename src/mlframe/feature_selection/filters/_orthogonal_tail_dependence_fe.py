"""Layer 73: upper/lower tail-dependence coefficient scorer
for the auto-scorer pool - a dedicated co-exceedance statistic, distinct from the Layer 66
copula-MI scorer's FULL-distribution average.

From extreme-value / copula theory (Longin & Solnik 2001; McNeil, Frey & Embrechts 2005), the upper
tail-dependence coefficient::

    lambda_U = lim_{q->1} P(Y > F_Y^{-1}(q) | X > F_X^{-1}(q))

measures how often ``X`` and ``Y`` are BOTH in their extreme tails simultaneously, estimated
empirically as the co-exceedance rate at a fixed high quantile threshold ``q`` divided by
``(1 - q)`` (symmetrically, ``lambda_L`` at the lower tail via ``1 - q``).

Why this catches a shape the catalog misses: the Layer 66 copula-MI scorer computes MI on the FULL
rank-uniformized joint distribution - an AVERAGE dependence measure over the whole ``[0,1]^2``
unit square. On a target where two columns are co-dependent ONLY in their joint extreme tail and
essentially independent in the bulk (a Gumbel-copula-like structure - financial contagion,
extreme-weather co-occurrence, catastrophic-failure data), the bulk's near-independence DILUTES the
average copula-MI toward a middling, easy-to-miss value, while this dedicated statistic isolates
exactly the co-exceedance signal that matters for a target like ``y = 1{catastrophic joint extreme
event}``. This reuses the SAME rank-uniformization the copula-MI family already computes - a
different summary of the same transformed data, not a duplicate.

The raw co-exceedance rate is floored against a permutation-null co-exceedance rate (independently
shuffling ``v``) to distinguish genuine tail dependence from finite-sample chance co-exceedance at
small ``n * (1 - q)`` - with few points in the tail, a handful of coincidental co-exceedances can
otherwise look like strong dependence.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._fe_usability_signal import _crit_np_dtype
from ._orthogonal_copula_mi_fe import _rank_to_uniform

__all__ = ["tail_dependence_score"]


@njit(cache=True, parallel=True)
def _tail_dep_columns_njit(cols_rows: np.ndarray, v: np.ndarray, q: float, out: np.ndarray) -> None:
    """``out[j]`` = ``max(upper, lower)`` floored tail-dependence of row ``j`` of ``cols_rows`` ``(k, n)`` against the rank-uniformised target ``v``.

    The column is rank-transformed once (average ranks for ties, divided by ``n + 1``) and both tails are read from the same ranks: the co-exceedance rate past the ``q`` quantile (upper) and
    below ``1 - q`` (lower), each minus its closed-form shuffle expectation ``P(u in tail) * P(v in tail) / (1 - q)``, clamped at 0. One parallel task per column."""
    kk = cols_rows.shape[0]
    n = cols_rows.shape[1]
    hi = q
    lo = 1.0 - q
    denom = 1.0 - q
    n_v_hi = 0
    n_v_lo = 0
    for i in range(n):
        if v[i] > hi:
            n_v_hi += 1
        if v[i] < lo:
            n_v_lo += 1
    for j in prange(kk):
        col = cols_rows[j]
        order = np.argsort(col, kind="mergesort")
        n_u_hi = 0
        n_u_lo = 0
        both_hi = 0
        both_lo = 0
        i = 0
        while i < n:
            e = i + 1
            while e < n and col[order[e]] == col[order[i]]:
                e += 1
            u = 0.5 * (i + 1 + e) / (n + 1.0)  # mean of the 1-based ranks i + 1 .. e
            u_hi = u > hi
            u_lo = u < lo
            for t in range(i, e):
                if u_hi:
                    n_u_hi += 1
                    if v[order[t]] > hi:
                        both_hi += 1
                if u_lo:
                    n_u_lo += 1
                    if v[order[t]] < lo:
                        both_lo += 1
            i = e
        up = both_hi / n / denom - (n_u_hi / n) * (n_v_hi / n) / denom
        low = both_lo / n / denom - (n_u_lo / n) * (n_v_lo / n) / denom
        best = up if up > low else low
        out[j] = best if best > 0.0 else 0.0


def tail_dependence_columns(cols_rows: np.ndarray, y: np.ndarray, *, q: float = 0.95) -> np.ndarray:
    """Floored tail dependence (max of the two tails) of every row of ``cols_rows`` ``(k, n)`` against ``y``, in one parallel kernel; all values must be finite.

    ``y`` is rank-uniformised once here. Zero for every column when ``n < 2`` or ``q`` leaves no tail."""
    k, n = cols_rows.shape
    out = np.zeros(k, dtype=np.float64)  # Intentional float64: statistic output kept f64
    if n < 2 or (1.0 - q) <= 0.0 or k == 0:
        return out
    v = _rank_to_uniform(np.asarray(y).ravel())
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    with parallel_kernel_entry():
        _tail_dep_columns_njit(np.ascontiguousarray(cols_rows), np.ascontiguousarray(v, dtype=np.float64), float(q), out)  # Intentional float64: ranks in double precision
    return out


def tail_dependence_score(
    x: np.ndarray,
    y: np.ndarray,
    *,
    q: float = 0.95,
    tail: str = "upper",
    n_perm: int = 100,
    random_state: int = 0,
) -> float:  # unused-ok: random_state is kept for call-site compatibility, the floor is the exact shuffle expectation and needs no draws
    """Permutation-floored tail-dependence coefficient between two 1-D arrays.

    Parameters
    ----------
    x, y : array-like (n,)
        Any numeric arrays of identical length.
    q : float
        Quantile threshold defining the tail (0.95 = top/bottom 5%).
    tail : {"upper", "lower"}
        ``"upper"``: co-exceedance rate of BOTH series above their ``q``-quantile, divided by
        ``1 - q`` (Longin-Solnik ``lambda_U``). ``"lower"``: co-exceedance rate of BOTH series
        below their ``1-q``-quantile, divided by ``1 - q`` (``lambda_L``).
    n_perm : int
        Number of independent shuffles of the rank-uniformized ``v`` used to estimate the
        chance co-exceedance floor. The permutation-null MEAN is subtracted from the raw rate
        (clamped at 0) so a genuinely independent pair reads ~0, not a spurious positive value
        from the raw rate's own finite-sample bias.
    random_state : int
        Seed for the permutation draws (deterministic).

    Returns
    -------
    float
        The floored tail-dependence coefficient, clamped to ``[0, +inf)``. Returns 0.0 on
        degenerate input (n < 2, or too few tail observations for ``q`` to be meaningful).
    """
    if tail not in ("upper", "lower"):
        raise ValueError(f"tail_dependence_score: tail must be 'upper' or 'lower', got {tail!r}")
    x = np.asarray(x, dtype=_crit_np_dtype()).ravel()
    y = np.asarray(y, dtype=_crit_np_dtype()).ravel()
    n = x.size
    if n < 2 or y.size != n:
        return 0.0
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return 0.0

    u = _rank_to_uniform(x)
    v = _rank_to_uniform(y)
    thresh = q if tail == "upper" else 1.0 - q
    denom = 1.0 - q
    if denom <= 0.0:
        return 0.0

    def _rate(u_arr: np.ndarray, v_arr: np.ndarray) -> float:
        """Co-exceedance rate of both rank-uniformized arrays past the tail threshold, normalized by (1-q)."""
        if tail == "upper":
            hit = (u_arr > thresh) & (v_arr > thresh)
        else:
            hit = (u_arr < thresh) & (v_arr < thresh)
        return float(np.mean(hit)) / denom

    raw_rate = _rate(u, v)
    if n_perm <= 0:
        return max(0.0, raw_rate)
    # Under an independent shuffle of v the expected co-exceedance rate is exactly P(u in tail) * P(v in tail) / (1-q); the permutation mean estimated that quantity with Monte-Carlo noise
    # (and cost n_perm shuffles of n rows), so the floor uses the closed form. n_perm only switches the floor on; random_state is kept for the signature.
    if tail == "upper":
        p_u, p_v = float(np.mean(u > thresh)), float(np.mean(v > thresh))
    else:
        p_u, p_v = float(np.mean(u < thresh)), float(np.mean(v < thresh))
    null_mean = p_u * p_v / denom
    return max(0.0, raw_rate - null_mean)
