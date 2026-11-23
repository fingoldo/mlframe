"""Cluster 12 regression: the golden fingerprint refuses a selected column with no FI score (audit 2026-11-20, finding 12-F89).

A missing score used to default to 0.0, so a column could be ranked wrongly and the hard gate still pass. The check uses a stub fitted
object, so no real hybrid fit is run.
"""

from types import SimpleNamespace

import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid import hybrid_opt_baseline as hb


def _stub(selected, fi):
    """A minimal fitted-hybrid stand-in with the attributes _fingerprint reads."""
    return SimpleNamespace(raw_selected_=list(selected), fi_=dict(fi), n_engineered_=0, member_selections_={}, members_={})


def test_missing_fi_score_is_refused():
    """A selected column absent from fi_ raises naming it, instead of being ranked at 0.0 (12-F89)."""
    with pytest.raises(ValueError, match="no FI score"):
        hb._fingerprint(_stub(["a", "b"], {"a": 1.0}))


def test_complete_fi_gives_the_ranking_by_score():
    """Control: with every selected column scored, the ranking is by descending score, ties by name (12-F89)."""
    fp = hb._fingerprint(_stub(["a", "b", "c"], {"a": 0.2, "b": 0.9, "c": 0.9}))
    assert fp["combined_ranking"] == ["b", "c", "a"]
    assert fp["n_selected"] == 3
