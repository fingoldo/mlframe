"""Per-group sliding-window iterator primitives.

Generic shape: take a 1-D array ``values`` aligned with a 1-D
``group_ids`` array, sort by group_ids (so all rows of a group are
contiguous), iterate per-group with `numpy` slicing, apply a function,
write results back to the ORIGINAL row order.

Why this exists: every per-group rolling-feature implementation in
mlframe + downstream projects reinvents the same boilerplate:

    sort_idx = np.argsort(group_ids, kind="stable")
    g_sorted = group_ids[sort_idx]
    v_sorted = values[sort_idx]
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate(([0], bnd))
    ends = np.concatenate((bnd, [n]))
    for s, e in zip(starts, ends):
        seg = v_sorted[s:e]
        # ... compute on seg
        out[sort_idx[s:e]] = result

The four well-known footguns:

1. forget ``kind="stable"`` -> non-deterministic when group_ids has ties
2. forget to scatter via ``sort_idx[s:e]`` -> output rows in WRONG slots
3. assume groups appear contiguously in input -> SILENT data leak when
   they don't (the sort is mandatory; ``stable=True`` is mandatory too
   so within-group order is preserved)
4. off-by-one in ``starts``/``ends`` when last group ends at n

This module ships ONE primitive ``per_group_apply`` that handles all
four, plus a thin convenience wrapper ``per_group_sliding_window``
that yields fixed-K window slices for the common rolling-feature case.
"""

from __future__ import annotations

__all__ = [
    "per_group_apply",
    "per_group_sliding_window",
    "iter_group_segments",
    "per_group_shift",
    "per_group_cum_reduce",
    "per_group_rolling_reduce",
    "per_group_nth",
    "per_group_rank",
]

import logging
from typing import Any, Callable, Iterator, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def iter_group_segments(
    group_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort + segment by group_id.

    Returns ``(sort_idx, starts, ends)`` such that for each group
    ``i``, the rows in the sorted order are ``sort_idx[starts[i]:ends[i]]``.
    Within each group the ORIGINAL row order is preserved (stable sort).

    O(n log n) time, O(n) memory. Standalone (callers that need the
    raw indices for custom iteration use this directly without
    paying for the ``per_group_apply`` callback machinery).
    """
    g = np.ascontiguousarray(group_ids)
    n = g.size
    if n == 0:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
        )
    sort_idx = np.argsort(g, kind="stable")
    g_sorted = g[sort_idx]
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate(([0], bnd)).astype(np.intp)
    ends = np.concatenate((bnd, [n])).astype(np.intp)
    return sort_idx, starts, ends


def per_group_apply(
    values: np.ndarray,
    group_ids: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    *,
    fill_value: float = np.nan,
    min_group_size: int = 1,
    output_dtype: Any = np.float64,
    output_shape_extra: Tuple[int, ...] = (),
) -> np.ndarray:
    """Apply ``fn(segment_values)`` per group; scatter back to original row order.

    Parameters
    ----------
    values
        1-D array of input values aligned to ``group_ids``.
    group_ids
        1-D array of group identifiers. Any dtype that ``np.argsort``
        can compare (int / str / bytes / pl-categorical).
    fn
        Callable invoked once per group with the SORTED segment's
        values. Must return an array of the same length as the segment
        (per-row outputs) OR ``None`` to mean "skip this group, fill
        with ``fill_value``". For per-row outputs with extra trailing
        dimensions (e.g. K spectral bands per row), pass
        ``output_shape_extra=(K,)`` and return a ``(seg_len, K)`` array.
    fill_value
        Used when ``fn`` returns ``None`` or the group is below
        ``min_group_size``.
    min_group_size
        Groups with fewer rows skip the callback and emit ``fill_value``
        for every row. Catches the common "K-window doesn't fit"
        degenerate case at the boundary.
    output_dtype
        Numpy dtype of the output array.
    output_shape_extra
        Extra trailing dimensions when ``fn`` returns ``(seg_len, *extra)``.
        Defaults to ``()`` for the common scalar-per-row case.

    Returns
    -------
    out
        Array of shape ``(len(values),) + output_shape_extra`` with
        results scattered back to the ORIGINAL row order.
    """
    values_arr = np.ascontiguousarray(values)
    n = values_arr.size
    if n != len(group_ids):
        raise ValueError(
            f"per_group_apply: values length {n} != group_ids length "
            f"{len(group_ids)}"
        )
    out_shape = (n,) + tuple(output_shape_extra)
    out = np.full(out_shape, fill_value, dtype=output_dtype)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    values_sorted = values_arr[sort_idx]
    for s, e in zip(starts, ends):
        seg = values_sorted[s:e]
        if seg.size < min_group_size:
            continue
        try:
            res = fn(seg)
        except Exception as err:
            logger.warning(
                "per_group_apply: fn raised on group of size %d: %s; "
                "filling with %s.",
                int(e - s), err, fill_value,
            )
            continue
        if res is None:
            continue
        res = np.asarray(res)
        if res.shape[0] != (e - s):
            raise ValueError(
                f"per_group_apply: fn returned shape {res.shape} but "
                f"segment length is {e - s}; per-row output expected."
            )
        # Scatter into output array. Trailing dims (if any) flow naturally.
        out[sort_idx[s:e]] = res
    return out


def per_group_shift(
    values: np.ndarray,
    group_ids: np.ndarray,
    n: int = 1,
    *,
    fill_value: float = np.nan,
    output_dtype: Any = np.float64,
) -> np.ndarray:
    """Lag/lead values within each group; positions out of bounds get ``fill_value``.

    ``n > 0`` = lag (shift forward in time: row i gets value at i-n).
    ``n < 0`` = lead (look ahead: row i gets value at i+|n|).

    Boundary contract: shifts NEVER bleed across group boundaries. The
    first ``|n|`` rows of each group (for n>0) get ``fill_value``;
    the last ``|n|`` rows (for n<0) get ``fill_value``. The naive
    ``np.roll`` or ``pd.Series.shift`` on a concatenated panel produces
    a silent leak at every group boundary -- the canonical "lag feature
    leaks across entities" bug.
    """
    values_arr = np.ascontiguousarray(values)
    out = np.full(values_arr.size, fill_value, dtype=output_dtype)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    for s, e in zip(starts, ends):
        seg_idx = sort_idx[s:e]
        seg_len = seg_idx.size
        if n > 0:
            if seg_len <= n:
                continue
            out[seg_idx[n:]] = values_arr[seg_idx[:-n]]
        elif n < 0:
            k = -n
            if seg_len <= k:
                continue
            out[seg_idx[:-k]] = values_arr[seg_idx[k:]]
        else:  # n == 0
            out[seg_idx] = values_arr[seg_idx]
    return out


def per_group_cum_reduce(
    values: np.ndarray,
    group_ids: np.ndarray,
    op: str = "sum",
    *,
    reverse: bool = False,
    output_dtype: Any = np.float64,
) -> np.ndarray:
    """Running aggregate per group with reset at each group boundary.

    ``op`` in {"sum", "prod", "max", "min", "count"}. ``count`` returns
    1-indexed within-group row count (cumsum of ones).

    ``reverse=True`` runs the cumulative reduction RIGHT-TO-LEFT, useful
    for "remaining-budget" / "time-to-end" features. The reduce is
    applied per-group, never bleeds across boundaries (the silent-leak
    failure of ``np.cumsum`` on a concatenated panel).
    """
    _accum = {
        "sum": np.add.accumulate,
        "prod": np.multiply.accumulate,
        "max": np.maximum.accumulate,
        "min": np.minimum.accumulate,
    }
    if op == "count":
        # Ignore values; emit 1-indexed within-group row count.
        n = group_ids.size if hasattr(group_ids, "size") else len(group_ids)
        out = np.empty(n, dtype=output_dtype)
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            seg_idx = sort_idx[s:e]
            arange = np.arange(1, seg_idx.size + 1, dtype=output_dtype)
            if reverse:
                arange = arange[::-1]
            out[seg_idx] = arange
        return out
    if op not in _accum:
        raise ValueError(
            f"op={op!r} not in {{'sum', 'prod', 'max', 'min', 'count'}}"
        )
    fn = _accum[op]
    values_arr = np.ascontiguousarray(values, dtype=output_dtype)
    out = np.empty_like(values_arr, dtype=output_dtype)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    for s, e in zip(starts, ends):
        seg_idx = sort_idx[s:e]
        seg = values_arr[seg_idx]
        if reverse:
            seg = seg[::-1]
        cum = fn(seg)
        if reverse:
            cum = cum[::-1]
        out[seg_idx] = cum
    return out


def per_group_rolling_reduce(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int,
    op: str = "mean",
    *,
    min_periods: int | None = None,
    fill_value: float = np.nan,
    output_dtype: Any = np.float64,
) -> np.ndarray:
    """Trailing-K-window built-in reduction per group.

    ``op`` in {"mean", "sum", "std", "var", "min", "max", "median"}.

    For ``mean``/``sum`` uses prefix-sum O(n) per group; for
    ``min``/``max`` uses sliding-window via stride_tricks O(n*K) but
    cache-friendly; for the rest uses sliding_window_view + axis=1
    reduction. The first ``min_periods - 1`` rows of each group emit
    ``fill_value`` (defaults to ``window_K`` -- i.e. only emit when the
    full window is available).
    """
    if window_K < 1:
        raise ValueError(f"window_K must be >= 1, got {window_K}")
    if min_periods is None:
        min_periods = window_K
    if min_periods < 1 or min_periods > window_K:
        raise ValueError(
            f"min_periods must be in [1, window_K], got {min_periods}"
        )

    from numpy.lib.stride_tricks import sliding_window_view

    values_arr = np.ascontiguousarray(values, dtype=output_dtype)
    out = np.full(values_arr.size, fill_value, dtype=output_dtype)
    sort_idx, starts, ends = iter_group_segments(group_ids)

    for s, e in zip(starts, ends):
        seg_idx = sort_idx[s:e]
        seg = values_arr[seg_idx]
        seg_len = seg.size
        if seg_len < min_periods:
            continue
        if op in ("sum", "mean"):
            # Prefix-sum O(n).
            seg_f = np.where(np.isfinite(seg), seg, 0.0)
            cs = np.concatenate(([0.0], np.cumsum(seg_f)))
            window_sums = cs[window_K:] - cs[:-window_K]
            if op == "mean":
                window_sums = window_sums / window_K
            # Write into out at last-position-anchor of each window.
            out[seg_idx[window_K - 1:]] = window_sums
            # min_periods shorter prefix
            if min_periods < window_K:
                for k in range(min_periods - 1, window_K - 1):
                    if k >= seg_len:
                        break
                    s_partial = float(cs[k + 1])
                    out[seg_idx[k]] = s_partial / (k + 1) if op == "mean" else s_partial
        elif op in ("std", "var", "median", "min", "max"):
            wins = sliding_window_view(seg, window_K)
            if op == "std":
                vals = wins.std(axis=1, ddof=1) if window_K > 1 else np.zeros(wins.shape[0])
            elif op == "var":
                vals = wins.var(axis=1, ddof=1) if window_K > 1 else np.zeros(wins.shape[0])
            elif op == "median":
                vals = np.median(wins, axis=1)
            elif op == "min":
                vals = wins.min(axis=1)
            elif op == "max":
                vals = wins.max(axis=1)
            out[seg_idx[window_K - 1:]] = vals
        else:
            raise ValueError(
                f"op={op!r} not in {{'mean','sum','std','var','min','max','median'}}"
            )
    return out


def per_group_nth(
    values: np.ndarray,
    group_ids: np.ndarray,
    n: int = 0,
    *,
    from_end: bool = False,
    broadcast: bool = False,
    fill_value: float = np.nan,
) -> tuple:
    """First / last / Nth value per group.

    When ``broadcast=False`` (default): returns
    ``(unique_group_ids, per_group_value)`` -- two arrays of length
    ``n_groups``. When ``broadcast=True``: returns a single array of
    length ``len(values)`` with each group's value replicated to every
    row of that group (ready for delta features: e.g.
    ``price - per_group_nth(price, session, n=0, broadcast=True)[1]``
    gives session-entry-relative price).

    Negative ``n`` indexes from end (e.g. ``n=-1, from_end=False`` is
    same as ``n=0, from_end=True`` and gives the LAST value per group).
    """
    values_arr = np.ascontiguousarray(values)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    n_groups = starts.size

    unique_ids = np.empty(n_groups, dtype=np.asarray(group_ids).dtype)
    per_group = np.full(n_groups, fill_value, dtype=np.float64)

    for g, (s, e) in enumerate(zip(starts, ends)):
        seg_idx = sort_idx[s:e]
        seg_len = seg_idx.size
        unique_ids[g] = group_ids[seg_idx[0]]
        if from_end:
            idx_within = seg_len - 1 - n if n >= 0 else -n - 1
        else:
            idx_within = n if n >= 0 else seg_len + n
        if 0 <= idx_within < seg_len:
            per_group[g] = values_arr[seg_idx[idx_within]]

    if not broadcast:
        return (unique_ids, per_group)

    # Broadcast back to row-level.
    out = np.full(values_arr.size, fill_value, dtype=np.float64)
    for g, (s, e) in enumerate(zip(starts, ends)):
        out[sort_idx[s:e]] = per_group[g]
    return (unique_ids, out)


def per_group_rank(
    values: np.ndarray,
    group_ids: np.ndarray,
    *,
    method: str = "average",
    pct: bool = False,
    ascending: bool = True,
) -> np.ndarray:
    """Within-group rank of each value.

    ``method`` in {"average", "min", "max", "dense", "ordinal"}; matches
    ``scipy.stats.rankdata`` semantics. ``pct=True`` returns
    normalised rank in ``[0, 1]`` (rank / n_group_rows).

    Stable sort within each group guarantees deterministic tie-break
    on identical input (no train/serve skew). The naive
    ``pd.groupby().rank()`` on >10M rows is the canonical "why is my
    FE step 40 min" hotspot; this version vectorises per segment.
    """
    if method not in {"average", "min", "max", "dense", "ordinal"}:
        raise ValueError(
            f"method={method!r} not in {{'average','min','max','dense','ordinal'}}"
        )
    values_arr = np.ascontiguousarray(values, dtype=np.float64)
    out = np.full(values_arr.size, np.nan, dtype=np.float64)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    for s, e in zip(starts, ends):
        seg_idx = sort_idx[s:e]
        seg = values_arr[seg_idx]
        if not ascending:
            seg = -seg
        # Rank only the FINITE entries; NaN/inf stay NaN. scipy.rankdata's
        # default nan_policy='propagate' otherwise poisons the WHOLE group
        # to NaN on a single missing value (silent: a rank-based feature
        # over any NaN-bearing column collapsed to an all-NaN -> "constant"
        # column, dropped downstream). pct normalises over the finite count
        # so observed values span (0, 1] regardless of how many are missing.
        finite = np.isfinite(seg)
        n_fin = int(finite.sum())
        if n_fin == 0:
            continue
        seg_fin = seg[finite]
        try:
            from scipy.stats import rankdata
            ranks = rankdata(seg_fin, method=method).astype(np.float64)
        except Exception:
            # average-method fallback
            order = np.argsort(seg_fin, kind="stable")
            ranks = np.empty(seg_fin.size, dtype=np.float64)
            ranks[order] = np.arange(1, seg_fin.size + 1, dtype=np.float64)
        if pct:
            ranks = ranks / n_fin
        seg_out = np.full(seg.size, np.nan, dtype=np.float64)
        seg_out[finite] = ranks
        out[seg_idx] = seg_out
    return out


def per_group_sliding_window(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int,
    *,
    min_group_size: int | None = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(sort_idx_segment, window_view, fill_indices)`` per group.

    Convenience wrapper around ``iter_group_segments`` for the
    fixed-K rolling-feature case. Skips groups whose segment is
    shorter than ``min_group_size`` (defaults to ``window_K``).

    Each yielded tuple:

    * ``sort_idx_segment``: indices into the ORIGINAL ``values`` array
      for the rows belonging to this group, in sorted (within-group
      original) order. Length = seg_len.
    * ``window_view``: ``np.lib.stride_tricks.sliding_window_view(seg,
      window_K)`` -- shape ``(seg_len - window_K + 1, window_K)``.
    * ``write_indices``: original-row indices for the LAST-POSITION
      anchor of each window (canonical convention: rolling stat at
      row ``r`` summarises rows ``[r - K + 1, r]``). Length =
      seg_len - K + 1.

    Caller writes per-window results to ``out[write_indices] = ...``.

    Skipping behaviour: if ``seg_len < min_group_size`` the group is
    skipped entirely (no yield). The first ``K - 1`` rows of a group
    naturally have no full-K window; those output slots stay at the
    caller's chosen fill value.

    Usage::

        from mlframe.feature_engineering.grouped import per_group_sliding_window
        out_mean = np.full(n, np.nan)
        for sort_idx_seg, wins, write_idx in per_group_sliding_window(
            x, well_id, window_K=100,
        ):
            out_mean[write_idx] = wins.mean(axis=1)

    The yield interface is deliberately raw (rather than
    ``per_group_apply`` style) so the caller can build the per-window
    feature with a SINGLE vectorised ``axis=1`` numpy call across all
    windows in the group, rather than a Python-loop over rows.
    """
    if window_K < 1:
        raise ValueError(f"window_K must be >= 1, got {window_K}")
    if min_group_size is None:
        min_group_size = window_K
    from numpy.lib.stride_tricks import sliding_window_view

    values_arr = np.ascontiguousarray(values)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    for s, e in zip(starts, ends):
        seg_len = int(e - s)
        if seg_len < min_group_size:
            continue
        sort_idx_seg = sort_idx[s:e]
        seg = values_arr[sort_idx_seg]
        wins = sliding_window_view(seg, window_K)
        # write_indices: original row indices of the LAST-POSITION
        # anchor (rows K-1 ... seg_len-1 of the segment).
        write_indices = sort_idx_seg[window_K - 1:]
        yield sort_idx_seg, wins, write_indices
