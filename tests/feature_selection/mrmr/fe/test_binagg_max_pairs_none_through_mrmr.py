"""MRMR passes ``fe_binned_numeric_agg_max_pairs=None`` (score every pair) down to the binned-aggregate stage unchanged."""

from __future__ import annotations

from mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_cascade_early_b import _optional_int
from mlframe.feature_selection.filters.mrmr import MRMR


def test_optional_int_keeps_none_and_converts_numbers():
    """None means no cap and stays None; a number (or a numeric string from a config) becomes the integer cap."""
    assert _optional_int(None) is None
    assert _optional_int(64) == 64
    assert _optional_int("12") == 12


def test_the_constructor_scores_every_pair_by_default():
    """Every pair is scored by default (None, in memory-sized batches); an integer cap is still accepted and round-trips through get_params."""
    assert MRMR().fe_binned_numeric_agg_max_pairs is None
    est = MRMR(fe_binned_numeric_agg_max_pairs=32)
    assert est.get_params()["fe_binned_numeric_agg_max_pairs"] == 32
