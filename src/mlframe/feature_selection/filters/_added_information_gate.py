"""Does an engineered column carry information about the target BEYOND its own source column? A G-test of the conditional information, for the group-statistic families.

The group-statistic families (grouped quantile / aggregate / distance, composite aggregate) built their gate on ``MI(candidate; y) - MI(source; y) >= 0`` and a fixed floor of ``1e-4`` nats. Both are
blind to the estimation noise: a column that only re-encodes its source sits at an uplift of zero, and the sign of the estimate decides whether it passes; a column of pure noise has a plug-in MI of
``(Bx - 1)(By - 1) / (2 n)`` (1.5e-4 nats at 30000 rows, 10 bins, a binary target), above the floor, and the sign of its uplift is a coin flip. On a frame without any signal the gate admitted the
maximum number of columns in every seed.

The test here is the one that answers the question: ``G = 2 n CMI(candidate; y | source)`` is chi-square distributed with ``occupied_source_bins * (Bc - 1) * (By - 1)`` degrees of freedom when the
candidate adds nothing to its source. A copy of the source has ``G`` near 0 whatever the sign of its MI difference; a noise column has ``G`` of the order of its degrees of freedom; a column that carries
what the source lacks has a ``G`` many times larger. The gate keeps a candidate when its p-value is below ``alpha`` divided by the number of candidates scored (Bonferroni).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numba import njit

from ._analytic_mi_null import _chi2_sf

#: Expected count per cell of the (source, candidate, target) table below which the chi-square approximation is not trusted and the candidate is left to the other gates.
_MIN_EXPECTED_CELL = 5.0


@njit(cache=True, nogil=True)
def _dense_small_ints(flat: np.ndarray, limit: int) -> "tuple[np.ndarray, int]":
    """``(codes, K)`` of the non-negative integer vector ``flat`` renumbered to ``0..K-1`` over its occupied values in one counting pass and one gather; ``K = -1`` (nothing else valid) when a value is negative or not below ``limit``."""
    n = flat.shape[0]
    mx = 0
    for i in range(n):
        v = flat[i]
        if v < 0:
            return np.empty(0, dtype=np.int64), -1
        if v > mx:
            mx = v
    if mx >= limit:
        return np.empty(0, dtype=np.int64), -1
    lut = np.full(mx + 1, -1, dtype=np.int64)
    for i in range(n):
        lut[flat[i]] = 0
    k = 0
    for v in range(mx + 1):
        if lut[v] == 0:
            lut[v] = k
            k += 1
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = lut[flat[i]]
    return out, k


def _dense_codes(a: np.ndarray) -> "tuple[np.ndarray, int]":
    """``a`` renumbered to 0..K-1 over its occupied values, and K. Small non-negative integer codes (the usual input) are renumbered through a count table, not a sort."""
    a = np.asarray(a)
    if a.size and a.dtype.kind in "iu":
        out, k = _dense_small_ints(np.ascontiguousarray(a).reshape(-1), 1 << 22)
        if k >= 0:
            return out, k
    _, inverse = np.unique(a, return_inverse=True)
    inverse = inverse.reshape(-1).astype(np.int64)
    return inverse, int(inverse.max()) + 1 if inverse.size else 0


def added_information_p(candidate: np.ndarray, source: "np.ndarray | None", y: np.ndarray) -> float:
    """p-value of ``H0: the candidate adds nothing about y beyond its source`` (G-test of ``CMI(candidate; y | source)``); ``source=None`` tests ``MI(candidate; y) = 0``.

    The arguments are integer codes (any labels, renumbered here). ``NaN`` when the table is too sparse for the chi-square approximation (expected count per cell under ``_MIN_EXPECTED_CELL``);
    ``1.0`` when the table has no degrees of freedom (a constant candidate or target)."""
    n = int(np.asarray(y).shape[0])
    c, bc = _dense_codes(candidate)
    t, by = _dense_codes(y)
    if source is None:
        s, bs = np.zeros(n, dtype=np.int64), 1
    else:
        s, bs = _dense_codes(source)
    df = bs * (bc - 1) * (by - 1)
    if n <= 0 or df <= 0:
        return 1.0
    if n / float(bs * bc * by) < _MIN_EXPECTED_CELL:
        return float("nan")
    joint = np.bincount((s * bc + c) * by + t, minlength=bs * bc * by).reshape(bs, bc, by).astype(np.float64)  # Intentional float64: counts and their logarithms in the G statistic
    n_sc = joint.sum(axis=2)
    n_sy = joint.sum(axis=1)
    n_s = joint.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        term = joint * (np.log(joint) + np.log(n_s)[:, None, None] - np.log(n_sc)[:, :, None] - np.log(n_sy)[:, None, :])
    g_stat = 2.0 * float(np.nansum(term[joint > 0]))
    if g_stat <= 0.0:
        return 1.0
    return float(min(1.0, max(0.0, _chi2_sf(g_stat, df))))


def passes_added_information(scores: pd.DataFrame, alpha: Optional[float]) -> pd.Series:
    """Boolean mask over the rows of ``scores`` (column ``p_added``): the candidates whose p-value is below ``alpha / len(scores)``.

    ``alpha=None`` keeps every row (the gate is off), and a row without a p-value (``NaN``: the table was too sparse to test) is kept, so the other gates of the family decide it as before."""
    if alpha is None or "p_added" not in scores.columns:
        return pd.Series(True, index=scores.index)
    p = scores["p_added"].astype(float)
    return p.isna() | (p < float(alpha) / max(1, len(scores)))


__all__ = ["added_information_p", "passes_added_information"]
