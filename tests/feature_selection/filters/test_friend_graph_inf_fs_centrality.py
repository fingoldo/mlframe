"""Direct unit coverage for ``friend_graph._inf_fs_centrality`` (mrmr_audit_2026-07-20
edge_cases.md #19/#21) -- the Roffo et al. 2017 Infinite Feature Selection centrality score,
previously only exercised transitively via full ``build_friend_graph`` calls."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.friend_graph import FriendGraphEdge, _inf_fs_centrality


class TestDegenerateInputsReturnEmptyOrZero:
    """Fewer than 2 nodes, no edges, or an all-zero adjacency have nothing to rank."""

    def test_single_node_returns_empty_dict(self):
        """A single-node selection has no pair to rank, hitting the explicit `n < 2` guard."""
        assert _inf_fs_centrality([1], []) == {}

    def test_empty_selection_returns_empty_dict(self):
        """An empty selection is also `n < 2` and must not raise on the empty array path."""
        assert _inf_fs_centrality([], []) == {}

    def test_no_edges_returns_empty_dict_even_with_multiple_nodes(self):
        """`n >= 2` but zero edges hits the `not edges` guard before any adjacency is built."""
        assert _inf_fs_centrality([1, 2, 3], []) == {}

    def test_all_zero_weight_edges_returns_zero_scores_not_empty(self):
        """Edges exist (so the graph isn't trivially empty) but every MI weight is 0 -- lambda_max
        of an all-zero adjacency is 0, hitting the explicit `lambda_max <= 0.0` early-return."""
        edges = [FriendGraphEdge(a=1, b=2, mi=0.0), FriendGraphEdge(a=2, b=3, mi=0.0)]
        scores = _inf_fs_centrality([1, 2, 3], edges)
        assert scores == {1: 0.0, 2: 0.0, 3: 0.0}

    def test_two_node_single_edge_returns_finite_scores_in_unit_range(self):
        """The minimal non-trivial (2-node, 1-edge) graph must not crash the eigendecomposition/
        matrix-inverse path and must produce finite scores in [0, 1]."""
        edges = [FriendGraphEdge(a=1, b=2, mi=0.7)]
        scores = _inf_fs_centrality([1, 2], edges)
        assert set(scores.keys()) == {1, 2}
        assert all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in scores.values())


class TestCentralityRanking:
    """A genuinely asymmetric graph must rank the more-connected/higher-weight node higher."""

    def test_hub_node_scores_highest_in_star_graph(self):
        """Node 1 is the hub connected to 3 leaves; it must score strictly higher than any leaf."""
        edges = [
            FriendGraphEdge(a=1, b=2, mi=0.8),
            FriendGraphEdge(a=1, b=3, mi=0.5),
            FriendGraphEdge(a=1, b=4, mi=0.3),
        ]
        scores = _inf_fs_centrality([1, 2, 3, 4], edges)
        assert scores[1] == max(scores.values())
        assert scores[1] == 1.0, "min-max normalization pins the top-scoring node to exactly 1.0"

    def test_scores_are_within_unit_interval(self):
        """A generic random weighted graph must produce finite, min-max normalized [0, 1] scores."""
        rng = np.random.default_rng(0)
        sel = [1, 2, 3, 4, 5]
        edges = []
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                w = float(rng.uniform(0.01, 1.0))
                edges.append(FriendGraphEdge(a=sel[i], b=sel[j], mi=w))
        scores = _inf_fs_centrality(sel, edges)
        assert all(0.0 <= v <= 1.0 for v in scores.values())
        assert min(scores.values()) == 0.0 and max(scores.values()) == 1.0

    def test_returned_keys_match_selection_exactly(self):
        """The output dict's keys must be exactly the (non-contiguous, non-zero-based) selection IDs."""
        edges = [FriendGraphEdge(a=10, b=20, mi=0.5), FriendGraphEdge(a=20, b=30, mi=0.9)]
        scores = _inf_fs_centrality([10, 20, 30], edges)
        assert set(scores.keys()) == {10, 20, 30}

    def test_isolated_node_with_no_incident_edges_still_scores_but_lowest(self):
        """Node 4 is in ``sel`` but has zero incident edges (isolated in the adjacency) -- its row of
        A is all-zero, so it must not dominate the ranking; it should score at or near the bottom."""
        edges = [FriendGraphEdge(a=1, b=2, mi=0.9), FriendGraphEdge(a=2, b=3, mi=0.9)]
        scores = _inf_fs_centrality([1, 2, 3, 4], edges)
        assert scores[4] == min(scores.values())
