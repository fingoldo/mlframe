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
import os
from typing import Any, Callable, Optional, Tuple

import numpy as np

# per_group_cum_reduce(op="count"): use the vectorized within-group-rank path only
# when the average group size is at or below this (i.e. many small groups). Above it,
# the trivial Python loop beats the full-length repeat/rank temporaries. Crossover
# measured at avg~=64-100 (bench_per_group_cum_count_vectorized_iter135.py): avg<=50
# wins 1.2-1.7x, avg=20 -> 1.9x, avg=10 -> 2.9x; avg>=100 ties/loses. Default 64 sits
# on the safe side of the crossover. Env-overridable per host.
_COUNT_VECTORIZE_MAX_AVG = int(os.environ.get("MLFRAME_GROUPED_COUNT_VECTORIZE_MAX_AVG", "64"))

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

from ._grouped_segments import iter_group_segments


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
        raise ValueError(f"per_group_apply: values length {n} != group_ids length " f"{len(group_ids)}")
    out_shape = (n, *tuple(output_shape_extra))
    out = np.full(out_shape, fill_value, dtype=output_dtype)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    values_sorted = values_arr[sort_idx]
    n_groups_attempted = 0
    n_groups_failed = 0
    last_err: Optional[Exception] = None
    for s, e in zip(starts, ends):
        seg = values_sorted[s:e]
        if seg.size < min_group_size:
            continue
        n_groups_attempted += 1
        try:
            res = fn(seg)
        except Exception as err:
            # A per-group failure (e.g. a numerically degenerate segment fn can't handle) is
            # tolerated -- that's the documented "skip and fill" contract. But if EVERY attempted
            # group fails, that's a systematic bug in the caller-supplied fn (wrong signature,
            # unconditional exception), not a per-group edge case; silently returning an
            # all-fill_value array in that case is the "error silently swallowed into wrong
            # downstream behavior" pattern, so we escalate below instead.
            n_groups_failed += 1
            last_err = err
            # A wide panel can have thousands of groups; an fn bug that fails on many of them
            # would otherwise flood the log one line per group. log_throttle caps the per-group
            # detail and lets the systematic-failure check below (or a future summary) carry the rest.
            log_throttle(
                logger,
                "per_group_apply_fn_raised",
                logging.WARNING,
                "per_group_apply: fn raised on group of size %d: %s; filling with %s",
                int(e - s), err, fill_value,
            )
            continue
        if res is None:
            continue
        res = np.asarray(res)
        if res.shape[0] != (e - s):
            raise ValueError(f"per_group_apply: fn returned shape {res.shape} but " f"segment length is {e - s}; per-row output expected.")
        # Scatter into output array. Trailing dims (if any) flow naturally.
        out[sort_idx[s:e]] = res
    if n_groups_attempted > 0 and n_groups_failed == n_groups_attempted:
        raise RuntimeError(
            f"per_group_apply: fn raised on ALL {n_groups_attempted} attempted group(s) -- this "
            f"looks like a systematic bug in fn (not a per-group edge case), so returning an "
            f"all-{fill_value} array would silently hide it. Last error: {last_err!r}"
        ) from last_err
    if n_groups_failed > 5:
        logger.warning("per_group_apply: %d of %d group(s) raised in total (last: %r).", n_groups_failed, n_groups_attempted, last_err)
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
    # bench-attempt-rejected (2026-06-23): a fully-vectorized rewrite (within-group
    # rank via np.repeat(starts, seg_lens), masked gather/scatter, no per-group loop)
    # is bit-identical but NOT faster -- it allocates several full-length intp arrays
    # (arange + repeat + rank + valid mask + masked index sets) so it is memory-
    # bandwidth bound, while the dominant cost (iter_group_segments' sort) is shared.
    # Measured best-of-5 (bench_per_group_shift_vectorized_iter135.py): 10M/200k groups
    # 1.10x (lag) / 0.96x (lead); 1M/20k 1.02x / 0.96x; 1M/5-groups 0.43x / 0.37x. The
    # few-groups case regresses 2.3x. Net no-win + uglier; loop kept.
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
        nseg = starts.size
        # Gated vectorization: the within-group 0-based rank of every sorted row is
        # ``arange(n) - repeat(starts, seg_lens)`` in one pass, so count = rank+1
        # (or seg_len-rank when reverse) needs NO Python per-group loop and NO
        # per-group arange allocation. Bit-identical. But the full-length repeat/rank
        # temporaries make it ~2x SLOWER than the trivial loop when groups are large;
        # it wins (1.2-1.9x, measured @ many-small-groups incl. the 10M/200k prof
        # shape) only when the average group is small. Gate on avg group size
        # (bench_per_group_cum_count_vectorized_iter135.py: avg<=64 wins, avg>=100
        # ties/loses) via ``n <= nseg * _COUNT_VECTORIZE_MAX_AVG``.
        if nseg and n <= nseg * _COUNT_VECTORIZE_MAX_AVG:
            seg_lens = (ends - starts).astype(np.intp)
            rank = np.arange(n, dtype=np.intp) - np.repeat(starts, seg_lens)
            if reverse:
                seg_len_per_pos = np.repeat(seg_lens, seg_lens)
                vals = (seg_len_per_pos - rank).astype(output_dtype)
            else:
                vals = (rank + 1).astype(output_dtype)
            out[sort_idx] = vals
            return out
        for s, e in zip(starts, ends):
            seg_idx = sort_idx[s:e]
            arange = np.arange(1, seg_idx.size + 1, dtype=output_dtype)
            if reverse:
                arange = arange[::-1]
            out[seg_idx] = arange
        return out
    if op not in _accum:
        raise ValueError(f"op={op!r} not in {{'sum', 'prod', 'max', 'min', 'count'}}")
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
        raise ValueError(f"min_periods must be in [1, window_K], got {min_periods}")

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
            out[seg_idx[window_K - 1 :]] = window_sums
            # min_periods shorter prefix
            if min_periods < window_K:
                for k in range(min_periods - 1, window_K - 1):
                    if k >= seg_len:
                        break
                    s_partial = float(cs[k + 1])
                    out[seg_idx[k]] = s_partial / (k + 1) if op == "mean" else s_partial
        elif op in ("std", "var", "median", "min", "max"):
            if seg_len >= window_K:
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
                out[seg_idx[window_K - 1 :]] = vals
            # min_periods shorter prefix: rows with fewer than window_K observations available (either because
            # the segment itself is shorter than window_K, or -- when seg_len >= window_K -- the leading rows
            # before the first full window) mirror the sum/mean branch's partial-prefix handling above via an
            # EXPANDING (not sliding) window from the start of the group. sliding_window_view cannot express a
            # window wider than the segment, so a full lstsq-style call would crash here whenever seg_len falls
            # in [min_periods, window_K) -- this loop is the fix.
            if min_periods < window_K:
                for k in range(min_periods - 1, min(window_K - 1, seg_len)):
                    partial = seg[: k + 1]
                    if op == "std":
                        out[seg_idx[k]] = float(partial.std(ddof=1)) if partial.size > 1 else 0.0
                    elif op == "var":
                        out[seg_idx[k]] = float(partial.var(ddof=1)) if partial.size > 1 else 0.0
                    elif op == "median":
                        out[seg_idx[k]] = float(np.median(partial))
                    elif op == "min":
                        out[seg_idx[k]] = float(partial.min())
                    elif op == "max":
                        out[seg_idx[k]] = float(partial.max())
        else:
            raise ValueError(f"op={op!r} not in {{'mean','sum','std','var','min','max','median'}}")
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


# Carved into sibling modules (monolith split, CLAUDE.md "sibling re-export" convention) to keep
# this file under the 1000 LOC budget: iter_group_segments' own dependency-free primitives live in
# _grouped_segments.py (imported above), and the rank-kernel family (per_group_rank,
# per_group_sliding_window, plus their njit helpers) lives in grouped_rank.py. Re-exported here
# unchanged so existing `from mlframe.feature_engineering.grouped import per_group_rank` call sites
# are unaffected. Neither sibling imports FROM this module, so this is not a cycle.
from .grouped_rank import per_group_rank, per_group_sliding_window
