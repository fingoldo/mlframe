"""The per-query index groups agree with the live group-slice helper in the ranking metrics, on random group ids."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.ranking import _iter_group_slices
from mlframe.reporting.charts.ltr import _per_query_groups


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_index_groups_match_sorted_group_slices(seed: int) -> None:
    """Each query's index array from the chart helper equals the rows the ranking helper slices out for that query."""
    rng = np.random.default_rng(seed)
    n = 500
    group_ids = rng.integers(0, 40, n)
    # arange as y_true makes the sorted y_true the permutation itself, so the slices are row indices.
    sorted_idx, _, starts = _iter_group_slices(np.arange(n), np.zeros(n), group_ids)
    expected = [sorted_idx[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)]
    got = _per_query_groups(group_ids)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        np.testing.assert_array_equal(np.sort(g), np.sort(e))
