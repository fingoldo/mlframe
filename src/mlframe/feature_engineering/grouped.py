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
from typing import Any, Callable, Iterator, Tuple, cast

import numpy as np

# per_group_cum_reduce(op="count"): use the vectorized within-group-rank path only
# when the average group size is at or below this (i.e. many small groups). Above it,
# the trivial Python loop beats the full-length repeat/rank temporaries. Crossover
# measured at avg~=64-100 (bench_per_group_cum_count_vectorized_iter135.py): avg<=50
# wins 1.2-1.7x, avg=20 -> 1.9x, avg=10 -> 2.9x; avg>=100 ties/loses. Default 64 sits
# on the safe side of the crossover. Env-overridable per host.
_COUNT_VECTORIZE_MAX_AVG = int(os.environ.get("MLFRAME_GROUPED_COUNT_VECTORIZE_MAX_AVG", "64"))

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # no-op fallback so the module imports without numba
        """No-op decorator stand-in for ``numba.njit`` when numba is unavailable, so the module still imports and runs (slower, pure Python)."""
        def wrap(fn):
            """Identity decorator applied when ``njit`` is called with arguments (e.g. ``@njit(cache=True)``)."""
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap


logger = logging.getLogger(__name__)


_RANK_METHOD_CODES = {"average": 0, "min": 1, "max": 2, "dense": 3, "ordinal": 4}


@njit(cache=True)
def _stable_counting_segments_int(g, gmin, span):
    """Stable counting sort of integer group ids in O(n + span).

    Returns ``(sort_idx, starts, ends)`` identical to
    ``np.argsort(g, kind="stable")`` + boundary detection: rows are ordered by
    ``(group_id, original_index)``, so within-group original order is preserved.
    Only valid for integer keys with a bounded ``span`` (the caller gates on RAM).
    """
    n = g.shape[0]
    counts = np.zeros(span + 1, dtype=np.int64)
    for i in range(n):
        counts[g[i] - gmin] += 1
    offsets = np.empty(span + 1, dtype=np.int64)
    acc = 0
    nonempty = 0
    for b in range(span + 1):
        offsets[b] = acc
        if counts[b] > 0:
            nonempty += 1
        acc += counts[b]
    sort_idx = np.empty(n, dtype=np.intp)
    cursor = offsets.copy()
    for i in range(n):
        b = g[i] - gmin
        sort_idx[cursor[b]] = i
        cursor[b] += 1
    starts = np.empty(nonempty, dtype=np.intp)
    ends = np.empty(nonempty, dtype=np.intp)
    k = 0
    for b in range(span + 1):
        if counts[b] > 0:
            starts[k] = offsets[b]
            ends[k] = offsets[b] + counts[b]
            k += 1
    return sort_idx, starts, ends


@njit(cache=True)
def _per_group_rank_sorted_njit(seg_vals, starts, ends, method_code, pct):
    """Within-group rank of finite values, one pass over group-sorted segments.

    ``seg_vals`` holds the finite values laid out group-contiguously (each
    ``[starts[g]:ends[g]]`` slice is one group, in within-group original order).
    Returns ranks in the SAME layout. Replaces the per-group ``scipy.stats.rankdata``
    Python loop; tie semantics match ``rankdata`` for all five methods, with the
    ordinal tie-break following within-group original order (stable argsort), matching
    the legacy path which fed ``rankdata`` the original-order segment.
    """
    m = seg_vals.shape[0]
    out = np.empty(m, dtype=np.float64)
    n_groups = starts.shape[0]
    for g in range(n_groups):
        s = starts[g]
        e = ends[g]
        seg_n = e - s
        if seg_n == 0:
            continue
        order = np.argsort(seg_vals[s:e], kind="mergesort")  # stable: ordinal ties by original order
        if method_code == 4:  # ordinal
            for k in range(seg_n):
                r = k + 1.0
                if pct:
                    r = r / seg_n
                out[s + order[k]] = r
            continue
        i = 0
        while i < seg_n:
            j = i + 1
            v = seg_vals[s + order[i]]
            while j < seg_n and seg_vals[s + order[j]] == v:
                j += 1
            # tie block covers ranks (i+1 .. j) in 1-based ordinal terms
            if method_code == 0:  # average
                rank = (i + 1 + j) / 2.0
            elif method_code == 1:  # min
                rank = i + 1.0
            elif method_code == 2:  # max
                rank = float(j)
            else:  # dense -- assigned below
                rank = 0.0
            for k in range(i, j):
                out[s + order[k]] = rank
            i = j
        if method_code == 3:  # dense: recompute as count of distinct values seen
            dense = 0.0
            prev_set = False
            prev = 0.0
            for k in range(seg_n):
                v = seg_vals[s + order[k]]
                if (not prev_set) or v != prev:
                    dense += 1.0
                    prev = v
                    prev_set = True
                out[s + order[k]] = dense
        if pct:
            for k in range(seg_n):
                out[s + k] = out[s + k] / seg_n
    return out


@njit(cache=True)
def _per_group_rank_ordinal_tiebreak_njit(seg_vals, seg_tb, starts, ends, pct):
    """Ordinal rank per group-contiguous segment, ties broken by ``seg_tb`` not row order.

    Composes two small per-group stable mergesorts (tiebreak, then primary) instead of one
    global multi-key ``np.lexsort`` over the whole array: a global lexsort re-sorts ALL n
    rows per key (measured 29s @10M/3-keys, i.e. slower than the group loop it replaced),
    while this sorts only ``seg_n`` rows per group per pass -- same total-work shape as the
    existing single-key ``_per_group_rank_sorted_njit`` kernel above.
    """
    m = seg_vals.shape[0]
    out = np.empty(m, dtype=np.float64)
    n_groups = starts.shape[0]
    for g in range(n_groups):
        s = starts[g]
        e = ends[g]
        seg_n = e - s
        if seg_n == 0:
            continue
        order_tb = np.argsort(seg_tb[s:e], kind="mergesort")
        composed_vals = seg_vals[s:e][order_tb]
        order_primary = np.argsort(composed_vals, kind="mergesort")
        order = order_tb[order_primary]
        for k in range(seg_n):
            r = k + 1.0
            if pct:
                r = r / seg_n
            out[s + order[k]] = r
    return out


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
    # Integer keys with a bounded value span use an O(n) stable counting sort instead of the
    # O(n log n) ``argsort(kind="stable")``; this is the shared bottleneck of every per-group helper
    # here (47-100x on the segmentation step @10M, see _benchmarks/bench_group_sort.py). The output is
    # bit-identical (rows ordered by (group_id, original_index)). Gated on ``span <= 4n + 1M`` so the
    # ``span+1`` counts array stays RAM-safe; sparse / huge-span / non-integer keys keep the argsort path.
    if _HAS_NUMBA and np.issubdtype(g.dtype, np.integer) and n > 1:
        gmin = int(g.min())
        span = int(g.max()) - gmin
        if 0 <= span <= 4 * n + 1_000_000:
            return cast(Tuple[np.ndarray, np.ndarray, np.ndarray], _stable_counting_segments_int(g, gmin, span))

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
        raise ValueError(f"per_group_apply: values length {n} != group_ids length " f"{len(group_ids)}")
    out_shape = (n, *tuple(output_shape_extra))
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
            raise ValueError(f"per_group_apply: fn returned shape {res.shape} but " f"segment length is {e - s}; per-row output expected.")
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


def _per_group_rank_ordinal_tiebreak(
    values_arr: np.ndarray,
    group_ids: np.ndarray,
    tiebreak_values: np.ndarray,
    *,
    pct: bool,
    ascending: bool,
    tiebreak_ascending: bool,
) -> np.ndarray:
    """Ordinal within-group rank with ties broken by a secondary column instead of row order.

    Plain ``method="ordinal"`` breaks ties by whatever order rows happen to sit in (the
    stable-sort within-group original order) -- deterministic, but semantically arbitrary:
    two rows sharing the exact same primary value get an arbitrary 1-apart rank split with
    no information content. When the caller has a meaningful secondary criterion (e.g. rank
    by score, tie-break by more-recent timestamp / larger volume), this resolves ties by
    that column via ``np.lexsort`` instead, so the split direction actually means something.

    One global ``np.lexsort`` (keys: group, then primary value, then tiebreak -- lexsort's
    LAST key is the primary sort criterion) replaces a Python per-group loop: a first cut at
    this used ``iter_group_segments`` + a per-group ``np.lexsort``/``np.flatnonzero`` loop and
    profiled at ~6x SLOWER than the plain-ordinal path at 10M rows / 200k groups
    (prof_per_group_rank_10m.py) -- Python call overhead dominates at that group count. This
    version is a single sort + a group-boundary/run-length pass, matching the plain path's cost.
    """
    tb_arr = np.ascontiguousarray(tiebreak_values, dtype=np.float64)
    if tb_arr.size != values_arr.size:
        raise ValueError(f"tiebreak_values length {tb_arr.size} != values length {values_arr.size}")
    out = np.full(values_arr.size, np.nan, dtype=np.float64)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    if sort_idx.size == 0:
        return out

    seg_vals = values_arr[sort_idx]
    seg_tb = tb_arr[sort_idx]
    if not ascending:
        seg_vals = -seg_vals
    if not tiebreak_ascending:
        seg_tb = -seg_tb
    finite_mask = np.isfinite(seg_vals)

    if _HAS_NUMBA:
        # Same group-contiguous-layout trick as the primary rank kernel: compact the
        # finite rows into per-group segments once, then a single njit pass does two
        # small per-group stable sorts per group instead of a Python loop of np.lexsort
        # calls (measured ~6x faster @10M/200k groups, prof_per_group_rank_10m.py).
        seg_finite_idx = np.flatnonzero(finite_mask)
        fin_vals = seg_vals[seg_finite_idx]
        fin_tb = seg_tb[seg_finite_idx]
        finite_cum = np.concatenate(([0], np.cumsum(finite_mask.astype(np.intp))))
        fstarts = finite_cum[starts].astype(np.intp)
        fends = finite_cum[ends].astype(np.intp)
        ranks_fin = _per_group_rank_ordinal_tiebreak_njit(fin_vals, fin_tb, fstarts, fends, bool(pct))
        out[sort_idx[seg_finite_idx]] = ranks_fin
        return out

    for s, e in zip(starts, ends):
        finite = finite_mask[s:e]
        n_fin = int(finite.sum())
        if n_fin == 0:
            continue
        idx_fin = np.flatnonzero(finite)
        primary = seg_vals[s:e][idx_fin]
        secondary = seg_tb[s:e][idx_fin]
        # lexsort's last key is primary; NaN in the tiebreak column sorts last within a
        # tied block (np.lexsort places NaN at the end), so a missing tiebreak degrades
        # gracefully to "after all rows with a real tiebreak value" rather than raising.
        order = np.lexsort((secondary, primary))
        ranks = np.empty(n_fin, dtype=np.float64)
        ranks[order] = np.arange(1, n_fin + 1, dtype=np.float64)
        if pct:
            ranks = ranks / n_fin
        seg_out = np.full(e - s, np.nan, dtype=np.float64)
        seg_out[idx_fin] = ranks
        out[sort_idx[s:e]] = seg_out
    return out


@njit(cache=True)
def _per_group_rank_causal_njit(seg_vals, starts, ends, pct, exclude_self):
    """Expanding-window average-tie rank per group-contiguous segment.

    For each row (in within-group original/time order), ranks it against only the rows at
    or before its own position (``exclude_self=False``) or strictly before it
    (``exclude_self=True``). Uses a Fenwick tree over the group's dense value buckets so the
    whole segment is O(seg_n log seg_n) rather than the O(seg_n^2) of re-sorting the
    prefix at every row -- the naive approach that would make this unusable past a few
    thousand rows/group.
    """
    m = seg_vals.shape[0]
    out = np.empty(m, dtype=np.float64)
    n_groups = starts.shape[0]
    for g in range(n_groups):
        s = starts[g]
        e = ends[g]
        n = e - s
        if n == 0:
            continue
        seg = seg_vals[s:e]
        uniq = np.unique(seg)
        k_buckets = uniq.shape[0]
        buckets = np.searchsorted(uniq, seg)  # 0-based, ties share a bucket
        bit = np.zeros(k_buckets + 1, dtype=np.int64)
        for i in range(n):
            b = buckets[i]
            if exclude_self:
                # Query BEFORE inserting: window is strictly the rows before i.
                idx = b + 1
                count_leq = 0
                while idx > 0:
                    count_leq += bit[idx]
                    idx -= idx & (-idx)
                idx = b
                count_less = 0
                while idx > 0:
                    count_less += bit[idx]
                    idx -= idx & (-idx)
                denom = i
                if denom == 0:
                    out[s + i] = np.nan
                else:
                    count_equal = count_leq - count_less
                    avg_rank = count_less + (count_equal + 1) / 2.0
                    out[s + i] = avg_rank / denom if pct else avg_rank
                idx = b + 1
                while idx <= k_buckets:
                    bit[idx] += 1
                    idx += idx & (-idx)
            else:
                # Insert THEN query: window includes the row's own value.
                idx = b + 1
                while idx <= k_buckets:
                    bit[idx] += 1
                    idx += idx & (-idx)
                idx = b + 1
                count_leq = 0
                while idx > 0:
                    count_leq += bit[idx]
                    idx -= idx & (-idx)
                idx = b
                count_less = 0
                while idx > 0:
                    count_less += bit[idx]
                    idx -= idx & (-idx)
                denom = i + 1
                count_equal = count_leq - count_less
                avg_rank = count_less + (count_equal + 1) / 2.0
                out[s + i] = avg_rank / denom if pct else avg_rank
    return out


def _per_group_rank_causal(
    values_arr: np.ndarray,
    group_ids: np.ndarray,
    *,
    pct: bool,
    ascending: bool,
    exclude_self: bool,
) -> np.ndarray:
    """Dispatch helper backing ``per_group_rank(..., causal=True)`` -- see its docstring."""
    out = np.full(values_arr.size, np.nan, dtype=np.float64)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    if sort_idx.size == 0:
        return out

    seg_vals = values_arr[sort_idx]
    if not ascending:
        seg_vals = -seg_vals
    finite_mask = np.isfinite(seg_vals)

    if _HAS_NUMBA:
        seg_finite_idx = np.flatnonzero(finite_mask)
        fin_vals = seg_vals[seg_finite_idx]
        finite_cum = np.concatenate(([0], np.cumsum(finite_mask.astype(np.intp))))
        fstarts = finite_cum[starts].astype(np.intp)
        fends = finite_cum[ends].astype(np.intp)
        ranks_fin = _per_group_rank_causal_njit(fin_vals, fstarts, fends, bool(pct), bool(exclude_self))
        out[sort_idx[seg_finite_idx]] = ranks_fin
        return out

    import bisect

    for s, e in zip(starts, ends):
        finite = finite_mask[s:e]
        idx_fin = np.flatnonzero(finite)
        seg_idx = sort_idx[s:e]
        seen: list = []
        for k, pos in enumerate(idx_fin):
            v = float(seg_vals[s:e][pos])
            if exclude_self:
                if k == 0:
                    out[seg_idx[pos]] = np.nan
                    bisect.insort(seen, v)
                    continue
                lo = bisect.bisect_left(seen, v)
                hi = bisect.bisect_right(seen, v)
                denom = len(seen)
                avg_rank = lo + (hi - lo + 1) / 2.0
                out[seg_idx[pos]] = avg_rank / denom if pct else avg_rank
                bisect.insort(seen, v)
            else:
                bisect.insort(seen, v)
                lo = bisect.bisect_left(seen, v)
                hi = bisect.bisect_right(seen, v)
                denom = len(seen)
                avg_rank = lo + (hi - lo + 1) / 2.0
                out[seg_idx[pos]] = avg_rank / denom if pct else avg_rank
    return out


def per_group_rank(
    values: np.ndarray,
    group_ids: np.ndarray,
    *,
    method: str = "average",
    pct: bool = False,
    ascending: bool = True,
    tiebreak_values: np.ndarray | None = None,
    tiebreak_ascending: bool = True,
    causal: bool = False,
    causal_exclude_self: bool = False,
) -> np.ndarray:
    """Within-group rank of each value.

    ``method`` in {"average", "min", "max", "dense", "ordinal"}; matches
    ``scipy.stats.rankdata`` semantics. ``pct=True`` returns
    normalised rank in ``[0, 1]`` (rank / n_group_rows).

    Stable sort within each group guarantees deterministic tie-break
    on identical input (no train/serve skew). The naive
    ``pd.groupby().rank()`` on >10M rows is the canonical "why is my
    FE step 40 min" hotspot; this version vectorises per segment.

    ``tiebreak_values`` (opt-in, ``method="ordinal"`` only): a secondary column, aligned
    to ``values``, that breaks ties deterministically by ITS ordering instead of arbitrary
    original-row order (e.g. rank bids by price, tie-break by submission time). Ignored /
    forbidden for the other methods, since average/min/max/dense assign the SAME rank to
    every row in a tied block by definition -- the tie-break order can't change their output,
    so silently accepting it there would be a no-op that looks like it did something.
    Omitting it (the default) leaves this function's output bit-identical to before.

    ``causal`` (opt-in, ``method="average"`` only): rank each row against only the rows
    of its group SEEN SO FAR in the input's within-group order, instead of the whole group
    (including rows that come after it in time). Callers must pre-sort each group by
    timestamp before calling -- the function has no separate time column, it uses the same
    "within-group original order" convention as the rest of this module (see
    ``iter_group_segments``). This is the leak-safe variant for online/causal scoring: a
    static full-group percentile computed once and reused at serve time silently uses
    future rows a real online scorer never has access to, which inflates any backtest that
    consumes it as a feature. ``causal_exclude_self`` (default ``False``) controls whether a
    row's own value is included in its own window; ``True`` gives a strictly-prior-only rank
    (the first row of every group has no prior data and is NaN), ``False`` includes the
    row's own value (the first row of a causal group always ranks 1.0 / pct 1.0, since it's
    alone in its own window so far). Forbidden with ``tiebreak_values`` (orthogonal opt-ins,
    combining them isn't implemented). Omitting ``causal`` (the default) leaves this
    function's output bit-identical to before.
    """
    if method not in {"average", "min", "max", "dense", "ordinal"}:
        raise ValueError(f"method={method!r} not in {{'average','min','max','dense','ordinal'}}")
    if causal and tiebreak_values is not None:
        raise ValueError("causal and tiebreak_values cannot be combined")
    if causal_exclude_self and not causal:
        raise ValueError("causal_exclude_self=True has no effect unless causal=True")
    if causal:
        if method != "average":
            raise ValueError("causal is only supported with method='average' (the expanding-window rank is a running average-tie percentile)")
        values_arr_causal = np.ascontiguousarray(values, dtype=np.float64)
        return _per_group_rank_causal(values_arr_causal, group_ids, pct=pct, ascending=ascending, exclude_self=causal_exclude_self)
    if tiebreak_values is not None:
        if method != "ordinal":
            raise ValueError("tiebreak_values is only supported with method='ordinal' (average/min/max/dense give tied rows the same rank regardless of tie-break order)")
        values_arr_tb = np.ascontiguousarray(values, dtype=np.float64)
        return _per_group_rank_ordinal_tiebreak(values_arr_tb, group_ids, tiebreak_values, pct=pct, ascending=ascending, tiebreak_ascending=tiebreak_ascending)
    values_arr = np.ascontiguousarray(values, dtype=np.float64)
    out = np.full(values_arr.size, np.nan, dtype=np.float64)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    if sort_idx.size == 0:
        return out

    # Rank only the FINITE entries; NaN/inf stay NaN. scipy.rankdata's default
    # nan_policy='propagate' otherwise poisons the WHOLE group to NaN on a single
    # missing value (silent: a rank-based feature over any NaN-bearing column collapses
    # to all-NaN -> "constant" -> dropped). pct normalises over the finite count so
    # observed values span (0, 1] regardless of how many are missing.
    seg_vals = values_arr[sort_idx]
    if not ascending:
        seg_vals = -seg_vals
    finite_mask = np.isfinite(seg_vals)

    if _HAS_NUMBA:
        # Whole-batch path: lay finite values out group-contiguously, rank every group in a
        # single njit pass. Replaces the per-group scipy.rankdata Python loop (100k+ calls,
        # each with its own argsort + dispatch) that dominated per_group_rank at large group counts.
        seg_finite_idx = np.flatnonzero(finite_mask)
        fin_vals = seg_vals[seg_finite_idx]
        # Per-group finite counts -> compact starts/ends in the finite layout (groups stay contiguous
        # because the source layout is already group-sorted and we drop within-group rows in order).
        n_groups = starts.shape[0]
        seg_len = (ends - starts).astype(np.intp)
        finite_cum = np.concatenate(([0], np.cumsum(finite_mask.astype(np.intp))))
        fstarts = finite_cum[starts]
        fends = finite_cum[ends]
        del seg_len
        ranks_fin = _per_group_rank_sorted_njit(fin_vals, fstarts.astype(np.intp), fends.astype(np.intp), _RANK_METHOD_CODES[method], bool(pct))
        out[sort_idx[seg_finite_idx]] = ranks_fin
        del n_groups
        return out

    from scipy.stats import rankdata

    for s, e in zip(starts, ends):
        seg_idx = sort_idx[s:e]
        seg = seg_vals[s:e]
        finite = finite_mask[s:e]
        n_fin = int(finite.sum())
        if n_fin == 0:
            continue
        seg_fin = seg[finite]
        ranks = rankdata(seg_fin, method=method).astype(np.float64)
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
        write_indices = sort_idx_seg[window_K - 1 :]
        yield sort_idx_seg, wins, write_indices
