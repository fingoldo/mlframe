"""Coverage for mlframe.testing.parametric's eager (non-Hypothesis) helpers.

X_TEST_SUITE_ARCHITECTURE-5: these were added so plain (non-property-based) tests -- the
majority of hand-rolled edge-case fixtures across the suite -- can reuse the same
building blocks the Hypothesis strategies use, instead of re-deriving them inline.
"""

from __future__ import annotations

import math

import polars as pl

from mlframe.testing.parametric import (
    categorical_series,
    constant_series,
    high_card_text_series,
    inf_heavy_float_series,
    sparse_null_series,
)


def test_constant_series_all_rows_equal():
    """constant_series produces a fixed-length column with every row equal to value."""
    s = constant_series("c", pl.Int32, 7, length=5)
    assert s.len() == 5
    assert s.to_list() == [7] * 5


def test_categorical_series_uses_only_given_categories_and_nulls():
    """categorical_series cycles through categories and injects nulls at the requested rate."""
    cats = ["a", "b", "c"]
    s = categorical_series("cat", cats, length=9, null_rate=1 / 3, use_enum=True)
    assert s.len() == 9
    non_null = [v for v in s.to_list() if v is not None]
    assert set(non_null) <= set(cats)
    assert s.null_count() > 0


def test_inf_heavy_float_series_contains_specials():
    """inf_heavy_float_series actually contains +inf/-inf/NaN, not just finite values."""
    s = inf_heavy_float_series("f", length=20, specials_rate=0.5)
    vals = s.to_list()
    assert any(v is not None and math.isinf(v) for v in vals)
    assert any(v is not None and math.isnan(v) for v in vals)


def test_high_card_text_series_all_unique():
    """high_card_text_series produces length distinct strings."""
    s = high_card_text_series("t", length=50)
    assert s.len() == 50
    assert s.n_unique() == 50


def test_sparse_null_series_mostly_null():
    """sparse_null_series is dominated by nulls at the configured non_null_rate."""
    s = sparse_null_series("s", pl.Utf8, length=1000, non_null_rate=0.001)
    assert s.null_count() > 900
