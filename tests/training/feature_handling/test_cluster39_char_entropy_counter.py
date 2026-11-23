"""Cluster 39 regression: character entropy keeps its value while counting in one pass (audit 2026-11-20, finding 39-F45)."""

import math

from mlframe.training.feature_handling.text_detection import _shannon_entropy_of_chars


def test_uniform_two_symbol_alphabet_has_one_bit():
    """Two equally frequent characters give exactly 1 bit (39-F45)."""
    assert math.isclose(_shannon_entropy_of_chars(["ab", "ba"]), 1.0)


def test_none_entries_are_skipped_and_empty_is_zero():
    """None entries are ignored and empty input returns 0.0 (39-F45)."""
    assert _shannon_entropy_of_chars([None, None]) == 0.0
    assert math.isclose(_shannon_entropy_of_chars(["ab", None, "ba"]), 1.0)
