"""Cluster 24 regressions: hybrid vote fallbacks (audit 2026-11-20, findings 24-F41 and 24-F42).

24-F41: when MRMR selects no relevant column, its member used to vote for every relevant column, so a cluster another member picked got two votes.
The member now abstains (no "mrmr" key) and the degradation is logged.

24-F42: when no cluster reached the vote, every cluster any member picked was kept, the opposite of the strict rule. Now only the clusters with
the most votes are kept.
"""

from mlframe.feature_selection.hybrid_selector import HybridSelector


def _selector_with_clusters(vote):
    """A HybridSelector with three clusters, bypassing fit: ``A`` holds a1, ``B`` holds b1, ``C`` holds c1."""
    sel = HybridSelector(vote=vote)
    sel.cluster_of_ = {"a1": "A", "b1": "B", "c1": "C"}
    sel.members_ = {"A": ["a1"], "B": ["b1"], "C": ["c1"]}
    sel.fi_ = {"a1": 0.1, "b1": 0.2, "c1": 0.3}
    sel.anchor_fe = False
    sel.expand_clusters = False
    sel.fi_guard = False
    return sel


def test_no_consensus_keeps_best_supported_cluster_not_the_union():
    """With vote=3 and votes A:1, B:2, C:1 nothing reaches 3; only B, the most supported, is kept (24-F42)."""
    sel = _selector_with_clusters(vote=3)
    member_sel = {"mrmr": ["a1"], "shap": ["b1"], "boruta": ["b1", "c1"]}
    selected = sel._combine(member_sel, ["a1", "b1", "c1"])
    assert list(selected) == ["b1"]


def test_no_consensus_with_a_tie_keeps_every_top_cluster():
    """Two clusters tied for the most votes are both kept, and nothing else (24-F42)."""
    sel = _selector_with_clusters(vote=3)
    member_sel = {"mrmr": ["a1", "b1"], "shap": ["b1", "a1"], "boruta": []}
    selected = sel._combine(member_sel, ["a1", "b1", "c1"])
    assert sorted(selected) == ["a1", "b1"]


def test_missing_mrmr_member_is_a_clean_abstention():
    """With the mrmr key omitted the other members vote alone; a cluster backed by two of them is kept at vote=2 (24-F41)."""
    sel = _selector_with_clusters(vote=2)
    member_sel = {"shap": ["b1"], "boruta": ["b1"]}
    selected = sel._combine(member_sel, ["a1", "b1", "c1"])
    assert list(selected) == ["b1"]
