"""Inter-event (time-gap) and group-aggregate features from an entity grouping key.

Composes ``feature_engineering.grouped`` primitives — ``per_group_shift`` (lag/lead within a group) and
``iter_group_segments`` (sort-once group segmentation) — into the "inter-event" feature family that turns a
(possibly heuristically reconstructed, e.g. probabilistic entity-resolution) join key into predictive signal:
time since the previous/next event for the same entity, and that entity's mean/std/median inter-event gap
and value, broadcast to every one of its rows. This is the exact feature set IEEE-CIS's 2nd place team built
from their reconstructed transaction ``uid``.

Backend: a fused ``numba.njit`` group-reduce kernel computing mean/std/median for every group in ONE pass,
with a ``per_group_apply``-based numpy fallback. Profiled (``_benchmarks/bench_entity_inter_event.py``):
the natural implementation via ``per_group_apply`` calls ``np.nanmean``/``np.nanstd``/``np.nanmedian`` once
per group — at real-world entity cardinality (many small groups, e.g. 100k+ pseudo-entities) each call pays
full numpy dispatch/masking overhead on a ~10-element array, dominating wall time (26.9s at 100k entities /
1M rows). The fused njit kernel computes all three stats for every group in one contiguous pass with no
per-group numpy dispatch, at any group count/cardinality — see the class docstring for the measured speedup.

Opt-in windowed variant (``window_size=`` / ``window_time=``): the whole-history group stats above are a
single number per entity, constant across all of that entity's rows — a customer whose transaction tempo
recently sped up or slowed down gets the SAME ``group_mean_time_delta`` at every row, diluted by however
much history preceded the regime change. Passing ``window_size`` (trailing K gaps) or ``window_time``
(trailing T time-units) restricts the mean/std to a causal, per-row trailing window ending at that row —
strictly no future leakage, since rows within a group are assumed already in chronological order (the same
assumption ``per_group_shift`` relies on). Median is intentionally not windowed (a sliding-window median
needs an order-statistics structure; mean/std cover the regime-drift use case at O(window) per row).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import numba

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a core mlframe dependency; exercised only if absent
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _median_sorted_njit(a: np.ndarray) -> float:
        """Median via full sort + midpoint, bit-identical to ``np.median`` (numba's ``np.median`` support is
        version-fragile, ``np.sort`` is universally supported)."""
        s = np.sort(a)
        n = s.size
        m = n // 2
        if n & 1:
            return float(s[m])
        return float(0.5 * (s[m - 1] + s[m]))

    @numba.njit(cache=True)
    def _group_mean_std_median_njit(values_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple:
        n_groups = starts.shape[0]
        means = np.empty(n_groups, dtype=np.float64)
        stds = np.empty(n_groups, dtype=np.float64)
        medians = np.empty(n_groups, dtype=np.float64)
        for g in range(n_groups):
            seg = values_sorted[starts[g] : ends[g]]
            finite = seg[np.isfinite(seg)]
            if finite.size == 0:
                means[g] = np.nan
                stds[g] = np.nan
                medians[g] = np.nan
                continue
            means[g] = finite.mean()
            stds[g] = finite.std() if finite.size > 1 else 0.0
            medians[g] = _median_sorted_njit(finite)
        return means, stds, medians

    @numba.njit(cache=True)
    def _windowed_stats_by_count_njit(values_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, window_size: int) -> tuple:
        n = values_sorted.shape[0]
        means = np.full(n, np.nan, dtype=np.float64)
        stds = np.full(n, np.nan, dtype=np.float64)
        for g in range(starts.shape[0]):
            s = starts[g]
            e = ends[g]
            for i in range(s, e):
                lo = i - window_size + 1
                if lo < s:
                    lo = s
                total = 0.0
                cnt = 0
                for j in range(lo, i + 1):
                    v = values_sorted[j]
                    if not np.isnan(v):
                        total += v
                        cnt += 1
                if cnt == 0:
                    continue
                m = total / cnt
                means[i] = m
                if cnt == 1:
                    stds[i] = 0.0
                    continue
                var_sum = 0.0
                for j in range(lo, i + 1):
                    v = values_sorted[j]
                    if not np.isnan(v):
                        var_sum += (v - m) * (v - m)
                stds[i] = np.sqrt(var_sum / cnt)
        return means, stds

    @numba.njit(cache=True)
    def _windowed_stats_by_time_njit(
        values_sorted: np.ndarray, timestamps_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, window_time: float
    ) -> tuple:
        # two-pointer sliding window ended by row `i` -- amortized O(n) per group, since `lo` only advances.
        n = values_sorted.shape[0]
        means = np.full(n, np.nan, dtype=np.float64)
        stds = np.full(n, np.nan, dtype=np.float64)
        for g in range(starts.shape[0]):
            s = starts[g]
            e = ends[g]
            lo = s
            total = 0.0
            total_sq = 0.0
            cnt = 0
            for i in range(s, e):
                while lo < i and timestamps_sorted[lo] <= timestamps_sorted[i] - window_time:
                    v_lo = values_sorted[lo]
                    if not np.isnan(v_lo):
                        total -= v_lo
                        total_sq -= v_lo * v_lo
                        cnt -= 1
                    lo += 1
                v_i = values_sorted[i]
                if not np.isnan(v_i):
                    total += v_i
                    total_sq += v_i * v_i
                    cnt += 1
                if cnt == 0:
                    continue
                m = total / cnt
                means[i] = m
                if cnt == 1:
                    stds[i] = 0.0
                    continue
                var = total_sq / cnt - m * m
                if var < 0.0:
                    var = 0.0
                stds[i] = np.sqrt(var)
        return means, stds


def _windowed_stats_by_count_py(values_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, window_size: int) -> tuple:
    """Pure-numpy fallback for ``_windowed_stats_by_count_njit`` (used only when numba is absent)."""
    n = values_sorted.shape[0]
    means = np.full(n, np.nan, dtype=np.float64)
    stds = np.full(n, np.nan, dtype=np.float64)
    for s, e in zip(starts, ends):
        for i in range(s, e):
            lo = max(s, i - window_size + 1)
            seg = values_sorted[lo : i + 1]
            finite = seg[np.isfinite(seg)]
            if finite.size == 0:
                continue
            means[i] = finite.mean()
            stds[i] = finite.std() if finite.size > 1 else 0.0
    return means, stds


def _windowed_stats_by_time_py(values_sorted: np.ndarray, timestamps_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, window_time: float) -> tuple:
    """Pure-numpy fallback for ``_windowed_stats_by_time_njit`` (used only when numba is absent)."""
    n = values_sorted.shape[0]
    means = np.full(n, np.nan, dtype=np.float64)
    stds = np.full(n, np.nan, dtype=np.float64)
    for s, e in zip(starts, ends):
        for i in range(s, e):
            ts_i = timestamps_sorted[i]
            lo = s
            while lo < i and timestamps_sorted[lo] <= ts_i - window_time:
                lo += 1
            seg = values_sorted[lo : i + 1]
            finite = seg[np.isfinite(seg)]
            if finite.size == 0:
                continue
            means[i] = finite.mean()
            stds[i] = finite.std() if finite.size > 1 else 0.0
    return means, stds


def _windowed_group_stats(
    values: np.ndarray,
    timestamps: np.ndarray,
    group_ids: np.ndarray,
    *,
    window_size: Optional[int] = None,
    window_time: Optional[float] = None,
) -> tuple:
    """Causal per-row trailing-window mean/std, ending at each row (row's own value included).

    Exactly one of ``window_size`` (trailing K samples) / ``window_time`` (trailing T time-units) must be
    given. Rows within a group are assumed already in chronological order (same contract as
    ``per_group_shift``/``iter_group_segments``) so the window never reaches into the future.
    """
    from mlframe.feature_engineering.grouped import iter_group_segments

    values_arr = np.ascontiguousarray(values, dtype=np.float64)
    ts_arr = np.ascontiguousarray(timestamps, dtype=np.float64)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    n = values_arr.shape[0]
    values_sorted = values_arr[sort_idx]
    ts_sorted = ts_arr[sort_idx]
    starts64 = starts.astype(np.int64)
    ends64 = ends.astype(np.int64)

    if window_size is not None:
        if _NUMBA_AVAILABLE:
            means_sorted, stds_sorted = _windowed_stats_by_count_njit(values_sorted, starts64, ends64, np.int64(window_size))
        else:
            means_sorted, stds_sorted = _windowed_stats_by_count_py(values_sorted, starts64, ends64, window_size)
    else:
        assert window_time is not None
        if _NUMBA_AVAILABLE:
            means_sorted, stds_sorted = _windowed_stats_by_time_njit(values_sorted, ts_sorted, starts64, ends64, np.float64(window_time))
        else:
            means_sorted, stds_sorted = _windowed_stats_by_time_py(values_sorted, ts_sorted, starts64, ends64, window_time)

    mean_out = np.empty(n, dtype=np.float64)
    std_out = np.empty(n, dtype=np.float64)
    mean_out[sort_idx] = means_sorted
    std_out[sort_idx] = stds_sorted
    return mean_out, std_out


def _broadcast_group_stats(values: np.ndarray, group_ids: np.ndarray) -> tuple:
    """Per-group mean/std/median, broadcast back to every row in original order."""
    from mlframe.feature_engineering.grouped import iter_group_segments

    values_arr = np.ascontiguousarray(values, dtype=np.float64)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    n = values_arr.shape[0]

    if _NUMBA_AVAILABLE:
        values_sorted = values_arr[sort_idx]
        means, stds, medians = _group_mean_std_median_njit(values_sorted, starts.astype(np.int64), ends.astype(np.int64))
        group_lengths = ends - starts
        group_of_sorted_row = np.repeat(np.arange(starts.shape[0]), group_lengths)
        mean_out = np.empty(n, dtype=np.float64)
        std_out = np.empty(n, dtype=np.float64)
        median_out = np.empty(n, dtype=np.float64)
        mean_out[sort_idx] = means[group_of_sorted_row]
        std_out[sort_idx] = stds[group_of_sorted_row]
        median_out[sort_idx] = medians[group_of_sorted_row]
        return mean_out, std_out, median_out

    from mlframe.feature_engineering.grouped import per_group_apply

    def _mean_fn(seg: np.ndarray) -> np.ndarray:
        return np.full(seg.shape, np.nanmean(seg))

    def _std_fn(seg: np.ndarray) -> np.ndarray:
        return np.full(seg.shape, np.nanstd(seg))

    def _median_fn(seg: np.ndarray) -> np.ndarray:
        return np.full(seg.shape, np.nanmedian(seg))

    return (
        per_group_apply(values_arr, group_ids, _mean_fn),
        per_group_apply(values_arr, group_ids, _std_fn),
        per_group_apply(values_arr, group_ids, _median_fn),
    )


def entity_inter_event_features(
    entity_ids: np.ndarray,
    timestamps: np.ndarray,
    value_col: Optional[np.ndarray] = None,
    *,
    window_size: Optional[int] = None,
    window_time: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Time-gap + group-aggregate features keyed by ``entity_ids``.

    Parameters
    ----------
    entity_ids
        ``(n,)`` grouping key (an actual id, or a heuristically reconstructed pseudo-entity key).
    timestamps
        ``(n,)`` numeric timestamp/ordering column aligned to ``entity_ids``. Rows are assumed already
        sorted chronologically within each entity (same contract as ``per_group_shift``).
    value_col
        Optional ``(n,)`` numeric column (e.g. transaction amount) to also aggregate per entity.
    window_size
        Opt-in: restrict the group stats to a causal trailing window of the last K gaps/values ending at
        each row (regime-drift signal a whole-history mean/std washes out — see module docstring). Mutually
        exclusive with ``window_time``. Omit both to keep the default whole-history-to-date behavior.
    window_time
        Opt-in: like ``window_size`` but the window is the trailing T time-units ending at each row's
        timestamp, rather than a fixed event count. Mutually exclusive with ``window_size``.

    Returns
    -------
    dict[str, np.ndarray]
        ``time_since_prev_event`` / ``time_to_next_event`` (NaN at each entity's first/last row —
        boundaries never bleed across entities, per ``per_group_shift``'s contract), plus
        ``group_mean_time_delta`` / ``group_std_time_delta`` / ``group_median_time_delta`` (that entity's
        overall inter-event-gap statistics, same value repeated for every row of the entity). When
        ``value_col`` is given, the same three group-aggregate stats are added for it as
        ``group_mean_value`` / ``group_std_value`` / ``group_median_value``. When ``window_size`` or
        ``window_time`` is given, adds causal windowed counterparts ``group_mean_time_delta_windowed`` /
        ``group_std_time_delta_windowed`` (and, with ``value_col``, ``group_mean_value_windowed`` /
        ``group_std_value_windowed``) — a per-row trailing-window mean/std instead of a whole-history one.
    """
    from mlframe.feature_engineering.grouped import per_group_shift

    if window_size is not None and window_time is not None:
        raise ValueError("entity_inter_event_features: pass at most one of window_size / window_time, not both")
    if window_size is not None and window_size < 1:
        raise ValueError(f"entity_inter_event_features: window_size must be >= 1, got {window_size}")
    if window_time is not None and window_time <= 0:
        raise ValueError(f"entity_inter_event_features: window_time must be > 0, got {window_time}")

    ts = np.ascontiguousarray(timestamps, dtype=np.float64)

    prev_ts = per_group_shift(ts, entity_ids, n=1)
    next_ts = per_group_shift(ts, entity_ids, n=-1)
    time_since_prev = ts - prev_ts
    time_to_next = next_ts - ts

    mean_dt, std_dt, median_dt = _broadcast_group_stats(time_since_prev, entity_ids)
    out = {
        "time_since_prev_event": time_since_prev,
        "time_to_next_event": time_to_next,
        "group_mean_time_delta": mean_dt,
        "group_std_time_delta": std_dt,
        "group_median_time_delta": median_dt,
    }

    if value_col is not None:
        mean_v, std_v, median_v = _broadcast_group_stats(value_col, entity_ids)
        out["group_mean_value"] = mean_v
        out["group_std_value"] = std_v
        out["group_median_value"] = median_v

    if window_size is not None or window_time is not None:
        mean_dt_w, std_dt_w = _windowed_group_stats(time_since_prev, ts, entity_ids, window_size=window_size, window_time=window_time)
        out["group_mean_time_delta_windowed"] = mean_dt_w
        out["group_std_time_delta_windowed"] = std_dt_w

        if value_col is not None:
            mean_v_w, std_v_w = _windowed_group_stats(value_col, ts, entity_ids, window_size=window_size, window_time=window_time)
            out["group_mean_value_windowed"] = mean_v_w
            out["group_std_value_windowed"] = std_v_w

    return out


__all__ = ["entity_inter_event_features"]
