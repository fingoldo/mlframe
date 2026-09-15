"""The raw monotone-twin drop keeps the higher-relevance twin; a twin with no screening MI must not lose by default.

``cached_MIs`` is filled by the greedy screen, so a raw column that a later rescue pass re-added can have no entry. Reading that miss as
0.0 made the rescued twin lose to any screen-selected twin, discarding exactly the column the rescue had recovered.
"""

from __future__ import annotations

from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._monotone_twin import monotone_twin_to_drop


def test_unknown_relevance_keeps_both_twins():
    """When either twin's relevance is unknown the drop is skipped: nothing is dropped."""
    assert monotone_twin_to_drop(candidate=7, kept=3, cached_mis={(3,): 0.4}) is None
    assert monotone_twin_to_drop(candidate=7, kept=3, cached_mis={(7,): 0.4}) is None


def test_known_relevance_drops_the_lower_twin():
    """Controls: with both relevances known the lower one is dropped, and a tie keeps the earlier-selected (kept) twin."""
    assert monotone_twin_to_drop(candidate=7, kept=3, cached_mis={(3,): 0.4, (7,): 0.6}) == 3
    assert monotone_twin_to_drop(candidate=7, kept=3, cached_mis={(3,): 0.6, (7,): 0.4}) == 7
    assert monotone_twin_to_drop(candidate=7, kept=3, cached_mis={(3,): 0.5, (7,): 0.5}) == 7
