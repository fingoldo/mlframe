"""Cluster 42 regression: the majority graph is unchanged by the single-pass rewrite, and the table is not copied needlessly (audit 2026-11-20, findings 42-F8, 42-F9)."""

import numpy as np
import pandas as pd

from mlframe.votenrank.leaderboard.leaderboard_impl import Leaderboard


def test_majority_graph_values_match_pairwise_outcomes():
    """A beats B on both tasks: entry (A,B) is 1 (a win), (B,A) is 0; the diagonal is a self-tie (0.5) plus the identity term (0.5) = 1.0 (42-F9)."""
    table = pd.DataFrame({"t1": [3.0, 1.0], "t2": [3.0, 1.0]}, index=["A", "B"])
    lb = Leaderboard(table)
    lb._ensure_majority_graph()
    g = lb.majority_graph
    assert g.loc["A", "B"] == 1.0
    assert g.loc["B", "A"] == 0.0
    assert g.loc["A", "A"] == 1.0


def test_tied_pair_scores_half_each_way():
    """Identical scores give a tie: each direction gets 0.5 (42-F9)."""
    table = pd.DataFrame({"t1": [2.0, 2.0], "t2": [5.0, 5.0]}, index=["A", "B"])
    lb = Leaderboard(table)
    lb._ensure_majority_graph()
    assert lb.majority_graph.loc["A", "B"] == 0.5
    assert lb.majority_graph.loc["B", "A"] == 0.5


def test_table_values_unchanged_after_construction():
    """Construction keeps the scores as given (42-F8: the no-op reindex is gone)."""
    table = pd.DataFrame({"t1": [1.0, 2.0]}, index=["A", "B"])
    lb = Leaderboard(table)
    np.testing.assert_array_equal(lb.table.to_numpy(), table.to_numpy())
