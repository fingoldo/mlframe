"""Per-entity recency-weighted aggregation of ordered histories.

Implements Dyakonov's weighted probability / weighted mean estimator
(``p_j = sum_i w_i * v_ij``) as a reusable per-group feature primitive: for
each entity (group), order its observations oldest -> newest and take the
recency-weighted mean of a value column, where the weights come from
:func:`mlframe.core.recency_weights.recency_weights` (poly / exp / power).

This subsumes two lecture use-cases in one primitive:
- binary ``v`` (visited / event happened) -> recency-weighted EVENT RATE
- continuous ``v`` (purchase amount)       -> recency-weighted MEAN VALUE

At the identity parameter (poly delta=0 / exp lam=1 / power gamma=0) the result
is the plain unweighted per-group mean, so opting into recency weighting never
changes behaviour until a non-identity parameter is chosen.

The weights are recomputed inline per group inside the njit kernel (no per-group
allocation), so a frame with millions of small entities stays allocation-light.
"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit

from mlframe.core.recency_weights import SCHEMES

logger = logging.getLogger(__name__)

__all__ = ["per_group_recency_weighted_mean"]


@njit(fastmath=False, cache=True)
def _group_recency_weighted_mean_sorted(v_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, scheme_code: int, param: float) -> np.ndarray:
    """Per-group recency-weighted mean over group-contiguous, within-group oldest-first sorted values.

    Returns one value per group (aligned with ``starts``/``ends``). Weights are computed inline (oldest -> newest)
    to avoid a per-group temporary; both numerator and denominator accumulate in the same pass, so an unnormalized
    weight family still yields the correct weighted mean (we divide by the realized weight sum).
    """
    n_groups = starts.shape[0]
    out = np.empty(n_groups, dtype=np.float64)
    for g in range(n_groups):
        s = starts[g]
        e = ends[g]
        m = e - s
        if m <= 0:
            out[g] = np.nan
            continue
        num = 0.0
        den = 0.0
        for pos in range(m):
            i = m - pos  # 1-based recency index, oldest (i=m) -> newest (i=1)
            if scheme_code == 0:
                w = ((m - i + 1) / m) ** param
            elif scheme_code == 1:
                w = param**i
            else:
                w = 1.0 / (i**param)
            num += w * v_sorted[s + pos]
            den += w
        out[g] = num / den if den > 0.0 else np.nan
    return out


def per_group_recency_weighted_mean(
    values: np.ndarray,
    group_ids: np.ndarray,
    *,
    order: np.ndarray | None = None,
    scheme: str = "poly",
    param: float = 1.0,
    broadcast: bool = True,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Recency-weighted per-entity mean of ``values``.

    Parameters
    ----------
    values : np.ndarray
        1-D value column (binary event indicator or continuous amount).
    group_ids : np.ndarray
        1-D entity id per row, aligned with ``values``.
    order : np.ndarray, optional
        1-D sort key giving within-entity chronological order (e.g. timestamp). Rows are ordered
        ASCENDING (oldest first) within each entity. If None, the existing row order within each
        entity is treated as chronological (stable).
    scheme : {'poly', 'exp', 'power'}
        Recency weight family (see :func:`mlframe.core.recency_weights.recency_weights`).
    param : float
        Weight-family parameter. Identity (poly 0 / exp 1 / power 0) -> plain unweighted mean.
    broadcast : bool
        If True (default) the per-entity value is scattered back to every original row (a feature column,
        length ``n``). If False, returns one value per unique entity in first-appearance order.
    fill_value : float
        Value used for empty groups (only reachable via broadcast=False bookkeeping; empty groups don't occur otherwise).

    Returns
    -------
    np.ndarray
        float64. Shape ``(n,)`` when ``broadcast`` else ``(n_groups,)``.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"per_group_recency_weighted_mean: scheme must be one of {SCHEMES}, got {scheme!r}.")
    values = np.ascontiguousarray(values, dtype=np.float64)
    group_ids = np.ascontiguousarray(group_ids)
    n = values.shape[0]
    if group_ids.shape[0] != n:
        raise ValueError("per_group_recency_weighted_mean: values and group_ids length mismatch.")
    param = float(param)

    if n == 0:
        return np.empty(0, dtype=np.float64)

    # Primary sort by group (stable), secondary by order within group so each group is contiguous AND oldest-first.
    if order is not None:
        order = np.ascontiguousarray(order)
        if order.shape[0] != n:
            raise ValueError("per_group_recency_weighted_mean: order length mismatch.")
        # lexsort: last key is primary -> (order within, then group as primary).
        sort_idx = np.lexsort((order, group_ids))
    else:
        sort_idx = np.argsort(group_ids, kind="stable")

    g_sorted = group_ids[sort_idx]
    v_sorted = values[sort_idx]
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), bnd.astype(np.int64)))
    ends = np.concatenate((bnd.astype(np.int64), np.array([n], dtype=np.int64)))

    scheme_code = SCHEMES.index(scheme)
    per_group = _group_recency_weighted_mean_sorted(v_sorted, starts, ends, scheme_code, param)

    if not broadcast:
        # Return per-group values in first-appearance order of the ORIGINAL array.
        _, first_pos = np.unique(group_ids, return_index=True)
        appearance_order = np.argsort(first_pos)
        # per_group is indexed by sorted-unique-group order; map to appearance order.
        sorted_unique = g_sorted[starts]
        # Build a lookup from group id -> per_group value.
        out = np.empty(sorted_unique.shape[0], dtype=np.float64)
        lut = {gid: per_group[k] for k, gid in enumerate(sorted_unique)}
        unique_appearance = group_ids[np.sort(first_pos)]
        for k, gid in enumerate(unique_appearance):
            out[k] = lut.get(gid, fill_value)
        return out

    # Broadcast each group's value back to its original rows (vectorized: repeat to sorted order, invert the sort).
    out_sorted = np.repeat(per_group, ends - starts)
    out = np.empty(n, dtype=np.float64)
    out[sort_idx] = out_sorted
    return out
