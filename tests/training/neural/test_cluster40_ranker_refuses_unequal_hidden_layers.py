"""Audit 2026-11-20 finding 40-F10: MLPRanker refuses hidden_layers with differing sizes instead of silently using the first width."""

import pytest

pytest.importorskip("lightning")
from mlframe.training.neural.ranker import MLPRanker


def test_unequal_hidden_layers_are_refused_at_construction():
    """(64, 32) would build every layer at 64 units; the constructor says so."""
    with pytest.raises(ValueError, match="differing sizes"):
        MLPRanker(hidden_layers=(64, 32))


def test_equal_hidden_layers_are_accepted():
    """(64, 64) builds exactly what it says."""
    assert MLPRanker(hidden_layers=(64, 64)).hidden_layers == (64, 64)
