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
from typing import Sequence

import numpy as np
from numba import njit

from mlframe.core.recency_weights import SCHEMES

logger = logging.getLogger(__name__)

__all__ = ["per_group_recency_weighted_mean", "per_group_recency_weighted_agg"]

AGGS = ("mean", "sum", "min", "max", "std", "var")


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


@njit(fastmath=False, cache=True)
def _group_recency_weighted_agg_sorted(v_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, scheme_code: int, param: float, agg_code: int) -> np.ndarray:
    """Per-group aggregation (mean/sum/min/max/std/var) of ``weight * value`` over group-contiguous, oldest-first values.

    ``agg_code``: 0=mean (weighted, matches ``_group_recency_weighted_mean_sorted``), 1=sum, 2=min, 3=max,
    4=std, 5=var. min/max aggregate the WEIGHTED value itself (weight in (0, 1] at the identity normalization
    shrinks older observations toward 0), so unlike mean this genuinely changes what min/max select, not just
    how they're scaled. std/var compute the recency-weighted (biased, population) variance of the RAW values
    around the recency-weighted mean -- i.e. dispersion, not dispersion-of-weighted-values -- via a first pass
    accumulating the weighted mean and a second pass accumulating the weighted sum-of-squared-deviations; both
    passes recompute weights inline (no per-group weight array), so this stays allocation-free like the others.
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
        if agg_code == 0 or agg_code == 4 or agg_code == 5:
            num = 0.0
            den = 0.0
            for pos in range(m):
                i = m - pos
                if scheme_code == 0:
                    w = ((m - i + 1) / m) ** param
                elif scheme_code == 1:
                    w = param**i
                else:
                    w = 1.0 / (i**param)
                num += w * v_sorted[s + pos]
                den += w
            mean = num / den if den > 0.0 else np.nan
            if agg_code == 0:
                out[g] = mean
                continue
            if den <= 0.0 or m < 2:
                out[g] = np.nan
                continue
            sq = 0.0
            for pos in range(m):
                i = m - pos
                if scheme_code == 0:
                    w = ((m - i + 1) / m) ** param
                elif scheme_code == 1:
                    w = param**i
                else:
                    w = 1.0 / (i**param)
                dv = v_sorted[s + pos] - mean
                sq += w * dv * dv
            var = sq / den
            out[g] = np.sqrt(var) if agg_code == 4 else var
        else:
            acc = 0.0 if agg_code == 1 else np.nan
            for pos in range(m):
                i = m - pos
                if scheme_code == 0:
                    w = ((m - i + 1) / m) ** param
                elif scheme_code == 1:
                    w = param**i
                else:
                    w = 1.0 / (i**param)
                wv = w * v_sorted[s + pos]
                if agg_code == 1:
                    acc += wv
                elif agg_code == 2:
                    acc = wv if (pos == 0 or wv < acc) else acc
                else:
                    acc = wv if (pos == 0 or wv > acc) else acc
            out[g] = acc
    return out


@njit(fastmath=False, cache=True)
def _group_recency_weighted_agg_multi_sorted(
    v_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, scheme_code: int, params: np.ndarray, agg_code: int
) -> np.ndarray:
    """Same aggregation as :func:`_group_recency_weighted_agg_sorted` but for MULTIPLE decay ``params`` in one pass.

    The group-sort and boundary computation (the expensive, non-embarrassingly-parallel part of the parent
    function -- an O(n log n) argsort/lexsort) happens exactly once in the caller regardless of how many
    ``params`` are requested; only the O(sum(m)) per-group weight loop is repeated per param here, so this is
    strictly cheaper than calling the single-param path once per decay value (which would re-sort every time).

    Returns shape ``(n_groups, n_params)``.
    """
    n_groups = starts.shape[0]
    n_params = params.shape[0]
    out = np.empty((n_groups, n_params), dtype=np.float64)
    for g in range(n_groups):
        s = starts[g]
        e = ends[g]
        m = e - s
        if m <= 0:
            for k in range(n_params):
                out[g, k] = np.nan
            continue
        for k in range(n_params):
            param = params[k]
            if agg_code == 0 or agg_code == 4 or agg_code == 5:
                num = 0.0
                den = 0.0
                for pos in range(m):
                    i = m - pos
                    if scheme_code == 0:
                        w = ((m - i + 1) / m) ** param
                    elif scheme_code == 1:
                        w = param**i
                    else:
                        w = 1.0 / (i**param)
                    num += w * v_sorted[s + pos]
                    den += w
                mean = num / den if den > 0.0 else np.nan
                if agg_code == 0:
                    out[g, k] = mean
                    continue
                if den <= 0.0 or m < 2:
                    out[g, k] = np.nan
                    continue
                sq = 0.0
                for pos in range(m):
                    i = m - pos
                    if scheme_code == 0:
                        w = ((m - i + 1) / m) ** param
                    elif scheme_code == 1:
                        w = param**i
                    else:
                        w = 1.0 / (i**param)
                    dv = v_sorted[s + pos] - mean
                    sq += w * dv * dv
                var = sq / den
                out[g, k] = np.sqrt(var) if agg_code == 4 else var
            else:
                acc = 0.0 if agg_code == 1 else np.nan
                for pos in range(m):
                    i = m - pos
                    if scheme_code == 0:
                        w = ((m - i + 1) / m) ** param
                    elif scheme_code == 1:
                        w = param**i
                    else:
                        w = 1.0 / (i**param)
                    wv = w * v_sorted[s + pos]
                    if agg_code == 1:
                        acc += wv
                    elif agg_code == 2:
                        acc = wv if (pos == 0 or wv < acc) else acc
                    else:
                        acc = wv if (pos == 0 or wv > acc) else acc
                out[g, k] = acc
    return out


def per_group_recency_weighted_agg(
    values: np.ndarray,
    group_ids: np.ndarray,
    *,
    agg: str = "mean",
    order: np.ndarray | None = None,
    scheme: str = "poly",
    param: float = 1.0,
    params: Sequence[float] | None = None,
    broadcast: bool = True,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Recency-weighted per-entity aggregation of ``values`` (mean/sum/min/max/std/var).

    Multiplies each observation by a recency weight (0 at the oldest, 1 at the most recent -- see
    :func:`mlframe.core.recency_weights.recency_weights`) before aggregating, generalizing
    :func:`per_group_recency_weighted_mean` beyond mean to sum/min/max/std/var. For ``agg='mean'`` this is
    identical to :func:`per_group_recency_weighted_mean` (same weighted-mean formula). For sum/min/max
    there is no equivalent existing primitive: weighting-then-min/max genuinely changes which observation
    is selected (recent extremes are preferred over older, larger-magnitude ones), unlike weighting-then-mean.
    For ``std``/``var`` the RAW values are aggregated (not pre-multiplied by weight like sum/min/max): weights
    instead control how much each observation contributes to the dispersion estimate, so recent observations
    dominate the variance the same way they dominate the mean -- this surfaces a recent change in volatility
    regime that a plain unweighted std/var (which spreads its attention evenly across the whole history) misses.

    Parameters
    ----------
    values, group_ids, order, scheme, param, broadcast, fill_value
        See :func:`per_group_recency_weighted_mean`.
    agg : {'mean', 'sum', 'min', 'max', 'std', 'var'}
        Aggregation applied to the weighted values (std/var use raw values weighted around the recency-weighted
        mean; groups with fewer than 2 observations yield NaN, matching pandas' ddof=0-adjacent "undefined" convention).
    params : Sequence[float], optional
        Opt-in multi-decay mode: when given (non-empty), computes the SAME aggregation at every decay strength in
        ``params`` in one call, reusing the single group sort/boundary pass (the O(n log n) part) instead of
        re-sorting per decay value the way calling this function once per ``param`` would. Typical use: a fast
        decay (short memory, reacts to a regime shift) and a slow decay (long memory, stable signal) as two
        columns of the same feature family. When ``params`` is given, ``param`` is ignored and the return gains
        a trailing axis of length ``len(params)`` (shape ``(n, len(params))`` broadcast, ``(n_groups, len(params))``
        otherwise) ordered exactly as ``params``. Leaving ``params`` as None reproduces the prior single-``param``
        behavior bit-for-bit.

    Returns
    -------
    np.ndarray
        float64. Shape ``(n,)`` / ``(n_groups,)`` for single ``param``; ``(n, len(params))`` / ``(n_groups, len(params))``
        when ``params`` is given.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"per_group_recency_weighted_agg: scheme must be one of {SCHEMES}, got {scheme!r}.")
    if agg not in AGGS:
        raise ValueError(f"per_group_recency_weighted_agg: agg must be one of {AGGS}, got {agg!r}.")
    values = np.ascontiguousarray(values, dtype=np.float64)
    group_ids = np.ascontiguousarray(group_ids)
    n = values.shape[0]
    if group_ids.shape[0] != n:
        raise ValueError("per_group_recency_weighted_agg: values and group_ids length mismatch.")

    multi = params is not None and len(params) > 0
    if multi:
        params_arr = np.ascontiguousarray(params, dtype=np.float64)
        if params_arr.shape[0] == 0:
            raise ValueError("per_group_recency_weighted_agg: params must be non-empty when given.")
    else:
        param = float(param)

    if n == 0:
        return np.empty((0, params_arr.shape[0]), dtype=np.float64) if multi else np.empty(0, dtype=np.float64)

    if order is not None:
        order = np.ascontiguousarray(order)
        if order.shape[0] != n:
            raise ValueError("per_group_recency_weighted_agg: order length mismatch.")
        sort_idx = np.lexsort((order, group_ids))
    else:
        sort_idx = np.argsort(group_ids, kind="stable")

    g_sorted = group_ids[sort_idx]
    v_sorted = values[sort_idx]
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), bnd.astype(np.int64)))
    ends = np.concatenate((bnd.astype(np.int64), np.array([n], dtype=np.int64)))

    scheme_code = SCHEMES.index(scheme)
    agg_code = AGGS.index(agg)

    if multi:
        per_group = _group_recency_weighted_agg_multi_sorted(v_sorted, starts, ends, scheme_code, params_arr, agg_code)
    else:
        per_group = _group_recency_weighted_agg_sorted(v_sorted, starts, ends, scheme_code, param, agg_code)

    if not broadcast:
        _, first_pos = np.unique(group_ids, return_index=True)
        sorted_unique = g_sorted[starts]
        out_shape = (sorted_unique.shape[0], params_arr.shape[0]) if multi else (sorted_unique.shape[0],)
        out = np.empty(out_shape, dtype=np.float64)
        lut = {gid: per_group[k] for k, gid in enumerate(sorted_unique)}
        unique_appearance = group_ids[np.sort(first_pos)]
        fill = np.full(params_arr.shape[0], fill_value) if multi else fill_value
        for k, gid in enumerate(unique_appearance):
            out[k] = lut.get(gid, fill)
        return out

    out_sorted = np.repeat(per_group, ends - starts, axis=0) if multi else np.repeat(per_group, ends - starts)
    out = np.empty_like(out_sorted)
    out[sort_idx] = out_sorted
    return out
