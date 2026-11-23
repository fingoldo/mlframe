"""Cluster 26 regression: a tuple and a list with the same items canonicalise to different bytes (audit 2026-11-20, finding 26-F33)."""

from mlframe.training._canonical_json import canonical_json_bytes


def test_tuple_and_list_differ():
    """(1, 2) and [1, 2] no longer produce the same canonical bytes (26-F33)."""
    assert canonical_json_bytes((1, 2)) != canonical_json_bytes([1, 2])


def test_identical_lists_still_match():
    """Control: equal lists canonicalise identically (26-F33)."""
    assert canonical_json_bytes([1, 2]) == canonical_json_bytes([1, 2])
