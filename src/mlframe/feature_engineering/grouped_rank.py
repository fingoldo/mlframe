"""``per_group_rank`` / ``per_group_sliding_window`` and their rank-kernel family: carved out of
``grouped.py`` (monolith split, CLAUDE.md "sibling re-export" convention) to keep the parent module
under the 1000 LOC budget. Re-exported from ``grouped.py`` unchanged.

Self-contained (imports only from ``._grouped_segments``, never from ``.grouped``) so ``grouped.py``
can re-export ``per_group_rank``/``per_group_sliding_window`` from here without creating an import
cycle.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from ._grouped_segments import HAS_NUMBA as _HAS_NUMBA
from ._grouped_segments import iter_group_segments, njit

__all__ = ["per_group_rank", "per_group_sliding_window"]


_RANK_METHOD_CODES = {"average": 0, "min": 1, "max": 2, "dense": 3, "ordinal": 4}


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
