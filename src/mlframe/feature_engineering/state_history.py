"""``last_k_distinct_states_with_durations``: per-row history of the last K distinct PRIOR state segments.

Source: dd_3rd_nasa-airport-config.md -- "a loop walking backward through configuration-change events,
recording each of the last 10 distinct configurations and how long (in minutes) each was active before the
next change." Distinct from :func:`mlframe.feature_engineering.state_duration.time_since_state_change` (a
binary-state, CURRENT-run-only feature): this handles an arbitrary categorical state alphabet and produces,
per row, the K most recently COMPLETED distinct state segments (state code + dwell duration each) preceding
the current position -- a compact "what were the last few regimes and how long did each last" feature block
for any sticky-state/config-change time series.

Backend: a single fused ``numba.njit`` pass per group using an O(1)-update circular buffer of the last K
completed segments (no growing list, no per-group Python callback), reusing the existing
:func:`mlframe.feature_engineering.grouped.iter_group_segments` scan machinery.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    import numba

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a core mlframe dependency; exercised only if absent
    _NUMBA_AVAILABLE = False


def _state_history_numpy(codes_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = codes_sorted.shape[0]
    out_states = np.full((n, k), -1, dtype=np.int64)
    out_durations = np.full((n, k), np.nan, dtype=np.float64)

    for g in range(starts.shape[0]):
        s, e = starts[g], ends[g]
        buf_states = np.full(k, -1, dtype=np.int64)
        buf_durations = np.zeros(k, dtype=np.float64)
        head = 0
        count = 0

        run_length = 1
        prev_state = codes_sorted[s]
        for i in range(s, e):
            if i > s:
                if codes_sorted[i] != prev_state:
                    buf_states[head] = prev_state
                    buf_durations[head] = run_length
                    head = (head + 1) % k
                    count = min(count + 1, k)
                    run_length = 1
                    prev_state = codes_sorted[i]
                else:
                    run_length += 1

            for kk in range(k):
                idx = (head - 1 - kk) % k
                if kk < count:
                    out_states[i, kk] = buf_states[idx]
                    out_durations[i, kk] = buf_durations[idx]

    return out_states, out_durations


if _NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _state_history_njit(codes_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        n = codes_sorted.shape[0]
        out_states = np.full((n, k), -1, dtype=np.int64)
        out_durations = np.full((n, k), np.nan, dtype=np.float64)

        for g in range(starts.shape[0]):
            s, e = starts[g], ends[g]
            buf_states = np.full(k, -1, dtype=np.int64)
            buf_durations = np.zeros(k, dtype=np.float64)
            head = 0
            count = 0

            run_length = 1
            prev_state = codes_sorted[s]
            for i in range(s, e):
                if i > s:
                    if codes_sorted[i] != prev_state:
                        buf_states[head] = prev_state
                        buf_durations[head] = run_length
                        head = (head + 1) % k
                        count = min(count + 1, k)
                        run_length = 1
                        prev_state = codes_sorted[i]
                    else:
                        run_length += 1

                for kk in range(k):
                    idx = (head - 1 - kk) % k
                    if kk < count:
                        out_states[i, kk] = buf_states[idx]
                        out_durations[i, kk] = buf_durations[idx]

        return out_states, out_durations


def last_k_distinct_states_with_durations(state_codes: np.ndarray, group_ids: np.ndarray, k: int = 10) -> Dict[str, np.ndarray]:
    """Per-row history of the last ``k`` distinct COMPLETED state segments preceding each row.

    Parameters
    ----------
    state_codes
        ``(n,)`` integer-coded categorical state per (entity, period) row (e.g. via ``pd.factorize``), in the
        row order reflecting the entity's true chronological sequence (rows are grouped and re-sorted by
        ``group_ids`` internally, sort your frame by time within each group before calling).
    group_ids
        ``(n,)`` entity/group key aligned to ``state_codes``.
    k
        Number of most-recent distinct prior segments to retain per row.

    Returns
    -------
    dict[str, np.ndarray]
        ``state_lag_1..k`` (int, ``-1`` for unavailable) -- the integer state code of each of the last ``k``
        distinct COMPLETED segments before the current row, ``state_lag_1`` most recent.
        ``duration_lag_1..k`` (float, ``NaN`` for unavailable) -- how many consecutive rows that segment
        lasted before transitioning away.
    """
    from mlframe.feature_engineering.grouped import iter_group_segments

    codes_arr = np.ascontiguousarray(state_codes).astype(np.int64)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    codes_sorted = codes_arr[sort_idx]

    if _NUMBA_AVAILABLE:
        out_states_sorted, out_durations_sorted = _state_history_njit(codes_sorted, starts.astype(np.int64), ends.astype(np.int64), k)
    else:
        out_states_sorted, out_durations_sorted = _state_history_numpy(codes_sorted, starts, ends, k)

    n = codes_arr.shape[0]
    out_states = np.empty((n, k), dtype=np.int64)
    out_durations = np.empty((n, k), dtype=np.float64)
    out_states[sort_idx] = out_states_sorted
    out_durations[sort_idx] = out_durations_sorted

    result: Dict[str, np.ndarray] = {}
    for kk in range(k):
        result[f"state_lag_{kk + 1}"] = out_states[:, kk]
        result[f"duration_lag_{kk + 1}"] = out_durations[:, kk]
    return result


__all__ = ["last_k_distinct_states_with_durations"]
