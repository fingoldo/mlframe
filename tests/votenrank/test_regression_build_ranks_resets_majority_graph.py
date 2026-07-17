"""Regression: build_ranks() did not reset self.majority_graph, so after a graph-using method
(copeland/condorcet/minimax) materialised it once, a subsequent build_ranks() on changed table
data reused the STALE majority graph and produced rankings for the old data."""

import pandas as pd

from mlframe.votenrank import Leaderboard


def _make_table(values):
    return pd.DataFrame(values, index=["A", "B", "C"], columns=["t1", "t2", "t3"])


def test_build_ranks_invalidates_stale_majority_graph():
    # First data: A dominates.
    lb = Leaderboard(_make_table([[9, 9, 9], [5, 5, 5], [1, 1, 1]]))
    first = lb.copeland_ranking()  # materialises majority_graph
    assert first.index[0] == "A"

    # Reuse the same instance with reversed data: C should now dominate.
    lb.table = _make_table([[1, 1, 1], [5, 5, 5], [9, 9, 9]])
    lb.build_ranks()
    assert lb.majority_graph is None, "build_ranks must invalidate the stale majority graph"

    second = lb.copeland_ranking()
    assert second.index[0] == "C", "ranking must reflect the new data, not the stale graph"
