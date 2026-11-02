"""The shortlist adapter pairs columns by name and keeps the caller's index (tracker 2026-11-02 wave 10, ALIGN10-1, ALIGN10-2)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.transformer._suite_adapter import ShortlistTransformerAdapter


def _compute(X, X_query=None, *, seed=0):
    """RFF-style stand-in: one feature per query row, the query's first column minus the train mean of that column."""
    import polars as pl

    q = X if X_query is None else X_query
    return pl.DataFrame({"first_minus_mean": q[:, 0] - X[:, 0].mean()})


def _data():
    """A frame with a non-default index and two columns of very different scale."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=40), "b": 1000 + rng.normal(size=40)}, index=np.arange(500, 540))


def test_reordered_columns_give_the_same_features():
    """transform(df[['b','a']]) must equal transform(df[['a','b']]); it used to compare train column a with query column b."""
    df = _data()
    ad = ShortlistTransformerAdapter(_compute, passthrough=False, needs_y=False).fit(df[["a", "b"]])
    straight = ad.transform(df[["a", "b"]])
    swapped = ad.transform(df[["b", "a"]])
    np.testing.assert_allclose(straight.to_numpy(), swapped.to_numpy())


def test_missing_column_raises():
    """A frame that lacks a fit column is refused with a message naming it."""
    df = _data()
    ad = ShortlistTransformerAdapter(_compute, passthrough=False, needs_y=False).fit(df[["a", "b"]])
    with pytest.raises(ValueError, match="missing"):
        ad.transform(df[["a"]])


@pytest.mark.parametrize("passthrough", [True, False])
def test_index_is_kept(passthrough):
    """The output carries the input index, so a join back onto the input lines up."""
    df = _data()
    out = ShortlistTransformerAdapter(_compute, passthrough=passthrough, needs_y=False).fit(df).transform(df)
    assert list(out.index) == list(df.index)
