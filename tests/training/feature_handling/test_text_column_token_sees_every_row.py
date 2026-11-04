"""The text-column cache token changes when any row changes, not only the five sampled ones."""

from __future__ import annotations

import pandas as pd
import polars as pl

from mlframe.training.feature_handling.apply import _text_column_content_token


def _texts(n=100):
    """A text column of n distinct strings."""
    return [f"doc number {i}" for i in range(n)]


def test_interior_row_change_changes_the_token_pandas():
    """Row 10 is not among the sampled positions 0, n//4, n//2, 3n//4, n-1, yet the token must differ."""
    a, b = _texts(), _texts()
    b[10] = "something else entirely"
    assert _text_column_content_token(pd.DataFrame({"t": a}), "t") != _text_column_content_token(pd.DataFrame({"t": b}), "t")


def test_interior_row_change_changes_the_token_polars():
    """Same check for a polars frame."""
    a, b = _texts(), _texts()
    b[10] = "something else entirely"
    assert _text_column_content_token(pl.DataFrame({"t": a}), "t") != _text_column_content_token(pl.DataFrame({"t": b}), "t")


def test_identical_content_gives_the_same_token():
    """Equal columns still share a token, so the cache keeps working."""
    assert _text_column_content_token(pd.DataFrame({"t": _texts()}), "t") == _text_column_content_token(pd.DataFrame({"t": _texts()}), "t")
