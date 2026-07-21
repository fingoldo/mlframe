"""Layer 73 (mrmr_audit_2026-07-20 fe_expansion.md): upper/lower tail-dependence coefficient scorer
for the auto-scorer pool -- a dedicated co-exceedance statistic, distinct from the Layer 66
copula-MI scorer's FULL-distribution average.

From extreme-value / copula theory (Longin & Solnik 2001; McNeil, Frey & Embrechts 2005), the upper
tail-dependence coefficient::

    lambda_U = lim_{q->1} P(Y > F_Y^{-1}(q) | X > F_X^{-1}(q))

measures how often ``X`` and ``Y`` are BOTH in their extreme tails simultaneously, estimated
empirically as the co-exceedance rate at a fixed high quantile threshold ``q`` divided by
``(1 - q)`` (symmetrically, ``lambda_L`` at the lower tail via ``1 - q``).

Why this catches a shape the catalog misses: the Layer 66 copula-MI scorer computes MI on the FULL
rank-uniformized joint distribution -- an AVERAGE dependence measure over the whole ``[0,1]^2``
unit square. On a target where two columns are co-dependent ONLY in their joint extreme tail and
essentially independent in the bulk (a Gumbel-copula-like structure -- financial contagion,
extreme-weather co-occurrence, catastrophic-failure data), the bulk's near-independence DILUTES the
average copula-MI toward a middling, easy-to-miss value, while this dedicated statistic isolates
exactly the co-exceedance signal that matters for a target like ``y = 1{catastrophic joint extreme
event}``. This reuses the SAME rank-uniformization the copula-MI family already computes -- a
different summary of the same transformed data, not a duplicate.

The raw co-exceedance rate is floored against a permutation-null co-exceedance rate (independently
shuffling ``v``) to distinguish genuine tail dependence from finite-sample chance co-exceedance at
small ``n * (1 - q)`` -- with few points in the tail, a handful of coincidental co-exceedances can
otherwise look like strong dependence.
"""

from __future__ import annotations

import numpy as np

from ._orthogonal_copula_mi_fe import _rank_to_uniform

__all__ = ["tail_dependence_score"]


def tail_dependence_score(
    x: np.ndarray,
    y: np.ndarray,
    *,
    q: float = 0.95,
    tail: str = "upper",
    n_perm: int = 100,
    random_state: int = 0,
) -> float:
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
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
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
    rng = np.random.default_rng(random_state)
    null_rates = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        v_perm = v[rng.permutation(n)]
        null_rates[i] = _rate(u, v_perm)
    null_mean = float(np.mean(null_rates))
    return max(0.0, raw_rate - null_mean)
