"""Duration-in-state features: condense a whole family of raw lag-N binary flags into 2 columns.

A common panel-data pattern (product possession, subscription/churn status, account state) is a per-entity-
per-period binary "state" series. The naive feature set is a pile of raw lag flags (state at t-1, t-2, ...,
t-N); the more informative and far more compact pair is "how long has the entity been in its CURRENT state":
``possession_duration`` (consecutive periods the state has been True, NaN while False) and
``cancellation_duration`` (consecutive periods False SINCE the entity was last True at least once, NaN
before the entity's first-ever True and while currently True) — exactly the two features Santander Product
Recommendation's 9th place team built instead of "a whole set" of raw lag flags.

Backend: a single fused ``numba.njit`` pass over the group-sorted state array (one linear scan, no per-group
Python callback — the "many small groups + per-group Python dispatch" trap that bit two earlier features
this session), with a numpy fallback.
"""
from __future__ import annotations

import numpy as np

try:
    import numba

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a core mlframe dependency; exercised only if absent
    _NUMBA_AVAILABLE = False


def _state_duration_numpy(state_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple:
    """Pure-numpy computation of per-row possession/cancellation run lengths and activation counts within each group."""
    n = state_sorted.shape[0]
    possession = np.full(n, np.nan, dtype=np.float64)
    cancellation = np.full(n, np.nan, dtype=np.float64)
    activation_count = np.zeros(n, dtype=np.int64)
    for g in range(starts.shape[0]):
        s, e = starts[g], ends[g]
        run_length = 0
        ever_true = False
        prev_state = None
        n_activations = 0
        for i in range(s, e):
            cur = bool(state_sorted[i])
            if prev_state is None or cur != prev_state:
                run_length = 0
                if cur:
                    # a fresh True run starting (including the entity's very first row) is one activation
                    n_activations += 1
            run_length += 1
            if cur:
                ever_true = True
                possession[i] = run_length
            elif ever_true:
                cancellation[i] = run_length
            activation_count[i] = n_activations
            prev_state = cur
    return possession, cancellation, activation_count


if _NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _state_duration_njit(state_sorted: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple:
        """Numba-accelerated computation of per-row possession/cancellation run lengths and activation counts within each group."""
        n = state_sorted.shape[0]
        possession = np.full(n, np.nan, dtype=np.float64)
        cancellation = np.full(n, np.nan, dtype=np.float64)
        activation_count = np.zeros(n, dtype=np.int64)
        for g in range(starts.shape[0]):
            s, e = starts[g], ends[g]
            run_length = 0
            ever_true = False
            prev_state = -1  # sentinel: no previous row yet
            n_activations = 0
            for i in range(s, e):
                cur = 1 if state_sorted[i] else 0
                if prev_state == -1 or cur != prev_state:
                    run_length = 0
                    if cur == 1:
                        # a fresh True run starting (including the entity's very first row) is one activation
                        n_activations += 1
                run_length += 1
                if cur == 1:
                    ever_true = True
                    possession[i] = run_length
                elif ever_true:
                    cancellation[i] = run_length
                activation_count[i] = n_activations
                prev_state = cur
        return possession, cancellation, activation_count


def time_since_state_change(state: np.ndarray, group_ids: np.ndarray, include_activation_count: bool = False) -> dict[str, np.ndarray]:
    """Duration-in-current-state features for a per-entity binary state series.

    Parameters
    ----------
    state
        ``(n,)`` boolean/binary array, one value per (entity, period) row, in the row order that reflects
        the entity's true chronological sequence (rows are grouped and re-sorted by ``group_ids`` internally
        using a stable sort, so pre-sort your frame by time within each group before calling).
    group_ids
        ``(n,)`` entity/group key aligned to ``state``.
    include_activation_count
        Opt-in (default ``False``, output unchanged when omitted). When ``True``, also returns
        ``activation_count`` — the number of times the entity has transitioned False->True up to and
        including the current row (0 before the entity's first-ever True). Two rows can share an identical
        ``cancellation_duration`` (e.g. both "cancelled 1 period ago") yet differ sharply in churn history —
        one a first-time cancellation, the other the entity's 5th acquire/cancel cycle — a distinction pure
        duration-since-change cannot express but that predicts recidivism-driven outcomes (e.g. permanent
        churn) on its own.

    Returns
    -------
    dict[str, np.ndarray]
        ``possession_duration`` — consecutive periods the state has been True, counting the row itself (1 at
        the very row it became True); ``NaN`` while the state is False.
        ``cancellation_duration`` — consecutive periods False SINCE the entity was last True; ``NaN`` while
        the state is True, and ``NaN`` before the entity's first-ever True (a state that has literally never
        been True cannot be "cancelled").
        ``activation_count`` (only when ``include_activation_count=True``) — integer count of prior
        acquire/reacquire events (state transitions into True) up to and including the current row.
    """
    from mlframe.feature_engineering.grouped import iter_group_segments

    state_arr = np.ascontiguousarray(state).astype(np.bool_)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    state_sorted = state_arr[sort_idx]

    if _NUMBA_AVAILABLE:
        possession_sorted, cancellation_sorted, activation_count_sorted = _state_duration_njit(state_sorted, starts.astype(np.int64), ends.astype(np.int64))
    else:
        possession_sorted, cancellation_sorted, activation_count_sorted = _state_duration_numpy(state_sorted, starts, ends)

    n = state_arr.shape[0]
    possession = np.empty(n, dtype=np.float64)
    cancellation = np.empty(n, dtype=np.float64)
    possession[sort_idx] = possession_sorted
    cancellation[sort_idx] = cancellation_sorted

    result: dict[str, np.ndarray] = {"possession_duration": possession, "cancellation_duration": cancellation}
    if include_activation_count:
        activation_count = np.empty(n, dtype=np.int64)
        activation_count[sort_idx] = activation_count_sorted
        result["activation_count"] = activation_count
    return result


__all__ = ["time_since_state_change"]
