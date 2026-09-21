"""Prospective FE pairs with equal operand-reuse counts are searched strongest-MI first.

The pair dict maps ``(raw_vars_pair, pair_mi)`` to a reuse counter and was sorted by the counter alone with a stable sort. Pairs are inserted
in ascending-MI order, so every tie group (all pairs early in the walk, when every counter is 0) put its weakest pair first.
"""

from __future__ import annotations

from mlframe.feature_selection.filters._mrmr_fe_step._step_pair_order import order_prospective_pairs


def _ascending_mi_insertion(entries):
    """Build the dict in ascending pair-MI order, as the ranking walk does."""
    return {key: counter for key, counter in sorted(entries, key=lambda e: e[0][1])}


def test_ties_on_counter_order_by_descending_mi():
    """All counters equal: the order is exactly descending pair MI."""
    entries = [(((1, 2), 0.05), 0), (((3, 4), 0.40), 0), (((5, 6), 0.12), 0), (((7, 8), 0.33), 0)]
    ordered = list(order_prospective_pairs(_ascending_mi_insertion(entries)))
    assert [k[1] for k in ordered] == [0.40, 0.33, 0.12, 0.05]


def test_counter_still_dominates_mi():
    """A higher reuse counter precedes a higher MI, so the cache-locality ordering is kept."""
    entries = [(((1, 2), 0.90), 0), (((3, 4), 0.10), 3), (((5, 6), 0.20), 3), (((7, 8), 0.50), 1)]
    ordered = list(order_prospective_pairs(_ascending_mi_insertion(entries)))
    assert ordered == [((5, 6), 0.20), ((3, 4), 0.10), ((7, 8), 0.50), ((1, 2), 0.90)]


def test_exact_ties_keep_insertion_order_and_values_are_preserved():
    """Equal (counter, MI) keep their insertion order; every key keeps its counter value."""
    entries = [(((1, 2), 0.2), 2), (((3, 4), 0.2), 2), (((5, 6), 0.7), 0)]
    src = {key: counter for key, counter in entries}
    ordered = order_prospective_pairs(src)
    assert list(ordered) == [((1, 2), 0.2), ((3, 4), 0.2), ((5, 6), 0.7)]
    assert ordered == src


def test_empty_pool():
    """No pairs: an empty dict back."""
    assert order_prospective_pairs({}) == {}
