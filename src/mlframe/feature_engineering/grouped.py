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
