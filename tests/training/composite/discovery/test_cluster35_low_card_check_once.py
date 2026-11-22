"""Audit 2026-11-20 finding 35-F44: the grouping test checks integer-valuedness once, not per column.

The loop already checks ``int_valued[j]``; the call site uses the level-count part alone. The wrapper keeps the full test for callers
that pass a raw column.
"""

import numpy as np

from mlframe.training.composite.discovery import _structural_hints as sh


def test_level_check_matches_full_check_on_integer_columns():
    """On an integer-valued column the level-only check agrees with the full check."""
    rng = np.random.default_rng(0)
    col = rng.integers(0, 6, size=400).astype(np.float64)
    assert sh._low_card_levels_ok(col) == sh._is_low_card_integer(col)


def test_full_check_still_rejects_non_integer_columns():
    """The wrapper refuses a continuous column; the level-only helper has no integer test."""
    col = np.linspace(0.0, 1.0, 400)
    assert sh._is_low_card_integer(col) is False
