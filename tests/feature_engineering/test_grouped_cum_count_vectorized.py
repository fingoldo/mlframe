"""Regression: gated vectorized path for per_group_cum_reduce(op="count").

The vectorized within-group-rank path (engaged when avg group size <=
_COUNT_VECTORIZE_MAX_AVG) must be BIT-IDENTICAL to the per-group Python loop on
both the gated-in (many small groups) and gated-out (few large groups) regimes,
for forward and reverse counts, across output dtypes, ties, negatives, and the
single-group / empty edge cases.

Pins the gate so a future "just always vectorize" (which regresses ~2x on few
large groups) cannot slip through: the small-group case must take the vectorized
branch and the large-group case must take the loop, and both must match the
reference loop.
"""

import numpy as np
import pytest

from mlframe.feature_engineering import grouped
from mlframe.feature_engineering.grouped import per_group_cum_reduce, iter_group_segments


def _loop_count(group_ids, reverse, output_dtype):
    """The pre-optimization reference: per-group arange + scatter."""
    n = len(group_ids)
    out = np.empty(n, dtype=output_dtype)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    for s, e in zip(starts, ends):
        seg_idx = sort_idx[s:e]
        ar = np.arange(1, seg_idx.size + 1, dtype=output_dtype)
        if reverse:
            ar = ar[::-1]
        out[seg_idx] = ar
    return out


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("output_dtype", [np.float64, np.float32, np.int64])
def test_small_groups_vectorized_matches_loop(reverse, output_dtype):
    rng = np.random.default_rng(1)
    n = 50_000
    n_groups = 5_000  # avg 10 rows -> well under the gate -> vectorized path
    gids = rng.integers(0, n_groups, size=n).astype(np.int64)
    nseg = iter_group_segments(gids)[1].size
    assert n <= nseg * grouped._COUNT_VECTORIZE_MAX_AVG, "expected the gated-IN regime"
    got = per_group_cum_reduce(np.empty(n), gids, "count", reverse=reverse, output_dtype=output_dtype)
    ref = _loop_count(gids, reverse, output_dtype)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("reverse", [False, True])
def test_large_groups_loop_matches_reference(reverse):
    rng = np.random.default_rng(2)
    n = 50_000
    n_groups = 3  # avg ~16k rows -> above the gate -> Python loop path
    gids = rng.integers(0, n_groups, size=n).astype(np.int64)
    nseg = iter_group_segments(gids)[1].size
    assert n > nseg * grouped._COUNT_VECTORIZE_MAX_AVG, "expected the gated-OUT regime"
    got = per_group_cum_reduce(np.empty(n), gids, "count", reverse=reverse)
    ref = _loop_count(gids, reverse, np.float64)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("reverse", [False, True])
def test_negative_and_tied_keys(reverse):
    gids = np.array([-3, 5, -3, -3, 5, 0, 0, 0, 0], dtype=np.int64)
    got = per_group_cum_reduce(np.empty(gids.size), gids, "count", reverse=reverse)
    ref = _loop_count(gids, reverse, np.float64)
    np.testing.assert_array_equal(got, ref)


def test_single_group_and_empty():
    g1 = np.zeros(7, dtype=np.int64)
    np.testing.assert_array_equal(
        per_group_cum_reduce(np.empty(7), g1, "count"),
        np.arange(1, 8, dtype=np.float64),
    )
    g0 = np.empty(0, dtype=np.int64)
    assert per_group_cum_reduce(np.empty(0), g0, "count").size == 0


def test_env_threshold_forces_loop(monkeypatch):
    """avg=10 group set is vectorized by default but the loop with threshold=0
    must yield the identical result (gate is purely a perf dispatch)."""
    rng = np.random.default_rng(3)
    n = 20_000
    gids = rng.integers(0, 2_000, size=n).astype(np.int64)
    default = per_group_cum_reduce(np.empty(n), gids, "count")
    monkeypatch.setattr(grouped, "_COUNT_VECTORIZE_MAX_AVG", 0)  # force loop
    forced_loop = per_group_cum_reduce(np.empty(n), gids, "count")
    np.testing.assert_array_equal(default, forced_loop)
