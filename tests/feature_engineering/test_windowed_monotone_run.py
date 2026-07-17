"""Regression for rolling_longest_monotone_run after njit vectorization.

The per-window-row longest-sign-run scan moved from a pure-Python double loop
to an @njit(parallel) kernel (~150x at 200k windows). These tests pin the
kernel against (a) hand-computed values on monotone signals and (b) an
independent reimplementation of the prior Python loop across all directions
and multiple groups.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering import rolling_longest_monotone_run
from mlframe.feature_engineering.grouped import per_group_sliding_window


def _ref(values, group_ids, window_K, direction):
    """Independent reimplementation of the pre-njit per-row Python loop."""
    out = np.full(values.size, np.nan, dtype=np.float64)
    for _si, wins, wi in per_group_sliding_window(values, group_ids, window_K=window_K):
        d = np.diff(wins, axis=1)
        mr = np.zeros(d.shape[0], dtype=np.int32)
        for r in range(d.shape[0]):
            lu = ld = cu = cd = 0
            for v in d[r]:
                if v > 0:
                    cu += 1
                    cd = 0
                    if cu > lu:
                        lu = cu
                elif v < 0:
                    cd += 1
                    cu = 0
                    if cd > ld:
                        ld = cd
                else:
                    cu = cd = 0
            run = lu if direction == "up" else (ld if direction == "down" else max(lu, ld))
            mr[r] = run + 1 if run > 0 else 1
        out[wi] = mr.astype(np.float64)
    return out


def test_strictly_increasing_signal_full_run():
    """Strictly increasing signal full run."""
    values = np.arange(50, dtype=np.float64)
    groups = np.zeros(50, dtype=int)
    up = rolling_longest_monotone_run(values, groups, window_K=10, direction="up")
    any_ = rolling_longest_monotone_run(values, groups, window_K=10, direction="any")
    dn = rolling_longest_monotone_run(values, groups, window_K=10, direction="down")
    fin_up = up[~np.isnan(up)]
    assert fin_up.size == 41  # positions 9..49
    np.testing.assert_array_equal(fin_up, 10.0)  # whole K-window monotone up
    np.testing.assert_array_equal(any_[~np.isnan(any_)], 10.0)
    np.testing.assert_array_equal(dn[~np.isnan(dn)], 1.0)  # no down-run -> length 1


@pytest.mark.parametrize("direction", ["up", "down", "any"])
def test_matches_python_reference_multigroup(direction):
    """Matches python reference multigroup."""
    rng = np.random.default_rng(123)
    n = 600
    values = rng.normal(size=n).cumsum()  # random walk -> mixed up/down runs
    group_ids = np.repeat(np.arange(4), n // 4)
    got = rolling_longest_monotone_run(values, group_ids, window_K=15, direction=direction)
    exp = _ref(values, group_ids, window_K=15, direction=direction)
    np.testing.assert_array_equal(np.nan_to_num(got, nan=-1.0), np.nan_to_num(exp, nan=-1.0))


def test_flat_segments_reset_runs():
    # A plateau (zero diffs) must break both up and down runs.
    """Flat segments reset runs."""
    values = np.array([0, 1, 2, 2, 2, 3, 4, 5, 6, 7], dtype=np.float64)
    groups = np.zeros(values.size, dtype=int)
    got = rolling_longest_monotone_run(values, groups, window_K=10, direction="up")
    exp = _ref(values, groups, window_K=10, direction="up")
    np.testing.assert_array_equal(np.nan_to_num(got, nan=-1.0), np.nan_to_num(exp, nan=-1.0))
