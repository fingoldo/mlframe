"""Regression: per_group_rolling_reduce must not crash when a group's length falls short of window_K.

FE_ROOT_A-2 (2026-08-05 audit): the sum/mean branch already had a partial-prefix fallback for rows with
fewer than ``window_K`` observations available (an expanding window from the group's own start), but the
std/var/median/min/max branch called ``sliding_window_view(seg, window_K)`` unconditionally -- which
raises ``ValueError: window shape cannot be larger than input array shape`` whenever a group's length
falls in ``[min_periods, window_K)``. Fixed by mirroring the sum/mean branch's expanding-window fallback
for these five ops too.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.grouped import per_group_rolling_reduce


def _ref(op: str, arr: np.ndarray) -> float:
    """Reference reduction matching per_group_rolling_reduce's own per-window convention."""
    if op == "std":
        return float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if op == "var":
        return float(arr.var(ddof=1)) if arr.size > 1 else 0.0
    if op == "median":
        return float(np.median(arr))
    if op == "min":
        return float(arr.min())
    return float(arr.max())


@pytest.mark.parametrize("op", ["std", "var", "median", "min", "max"])
def test_short_group_below_window_k_does_not_crash(op):
    """A group shorter than window_K but at/above min_periods must not raise ValueError."""
    values = np.array([1.0, 3.0, 2.0, 10.0, 8.0, 6.0, 9.0, 7.0, 5.0], dtype=np.float64)
    group_ids = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    window_K = 5
    min_periods = 2

    out = per_group_rolling_reduce(values, group_ids, window_K, op=op, min_periods=min_periods)
    assert out.shape == values.shape

    g0 = values[0:3]  # group 0: length 3 < window_K -- entirely partial (expanding) windows.
    assert np.isnan(out[0])
    assert out[1] == _ref(op, g0[:2])
    assert out[2] == _ref(op, g0[:3])

    g1 = values[3:9]  # group 1: length 6 >= window_K -- partial prefix then two full windows.
    assert np.isnan(out[3])
    assert out[4] == _ref(op, g1[:2])
    assert out[5] == _ref(op, g1[:3])
    assert out[6] == _ref(op, g1[:4])
    assert out[7] == _ref(op, g1[:5])
    assert out[8] == _ref(op, g1[1:6])


@pytest.mark.parametrize("op", ["std", "var", "median", "min", "max"])
def test_all_groups_shorter_than_window_k(op):
    """Every group in the input is shorter than window_K: must not crash, all rows below min_periods
    stay fill_value, the rest are expanding-window reductions."""
    values = np.array([5.0, 2.0, 4.0, 1.0], dtype=np.float64)
    group_ids = np.array([0, 0, 1, 1])
    window_K = 10
    min_periods = 1

    out = per_group_rolling_reduce(values, group_ids, window_K, op=op, min_periods=min_periods)
    assert out[0] == _ref(op, values[0:1])
    assert out[1] == _ref(op, values[0:2])
    assert out[2] == _ref(op, values[2:3])
    assert out[3] == _ref(op, values[2:4])


def test_matches_full_window_path_when_min_periods_equals_window_k():
    """Sanity: when min_periods == window_K (the default), behaviour is unchanged -- only full windows
    are emitted, no expanding-window fallback rows."""
    values = np.arange(12, dtype=np.float64)
    group_ids = np.zeros(12, dtype=np.int64)
    out = per_group_rolling_reduce(values, group_ids, window_K=4, op="max")
    assert np.isnan(out[:3]).all()
    assert out[3] == 3.0
    assert out[11] == 11.0
