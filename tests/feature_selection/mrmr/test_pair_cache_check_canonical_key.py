"""The all-pairs-cached check reads the pair cache by its canonical sorted key, whatever order the candidate set iterates in."""

from __future__ import annotations

from mlframe.feature_selection.filters._mrmr_fe_step._step_pairmi import all_pairs_cached


def _ids_not_ascending() -> list:
    """Candidate ids in descending order: every pair ``combinations`` yields is (high, low), the order a set of mixed raw and engineered indices can give."""
    return [1000, 40, 7, 3, 2]


def test_filled_cache_under_sorted_keys_counts_as_complete():
    """Every pair stored as (low, high) is found although the ids yield every pair as (high, low)."""
    ids = _ids_not_ascending()
    ordered = sorted(ids)
    cache = {(a, b): 0.1 for i, a in enumerate(ordered) for b in ordered[i + 1 :]}
    assert all_pairs_cached(ids, cache, {}) is True


def test_a_missing_pair_is_reported():
    """One pair absent from both caches makes the check False."""
    ids = _ids_not_ascending()
    ordered = sorted(ids)
    cache = {(a, b): 0.1 for i, a in enumerate(ordered) for b in ordered[i + 1 :]}
    del cache[(2, 3)]
    assert all_pairs_cached(ids, cache, {}) is False
    assert all_pairs_cached(ids, cache, {(2, 3): 0.2}) is True
