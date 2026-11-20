"""Audit 2026-11-20 finding 39-F32: the literal string "NaN" in a Polars string column is data; only float NaN cells are missing."""

import polars as pl

from mlframe.training.feature_handling._target_encoders_canon import _NULL_SENTINEL, _series_tokens_polars


def test_string_nan_is_kept_as_its_own_token():
    """A string column whose value is the word 'NaN' keeps that token instead of the missing-value sentinel."""
    tokens = _series_tokens_polars(pl.Series(["NaN", "x"]))
    assert tokens[0] == "NaN"
    assert tokens[0] != _NULL_SENTINEL


def test_float_nan_still_becomes_the_null_sentinel():
    """A float NaN cell is missing and maps to the sentinel, as the pandas path does."""
    tokens = _series_tokens_polars(pl.Series([float("nan"), 1.0]))
    assert tokens[0] == _NULL_SENTINEL
    assert tokens[1] != _NULL_SENTINEL
