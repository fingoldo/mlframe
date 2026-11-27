"""Cluster 28 regression: ``_DatasetReuseMixin._realign_val_columns`` gives the validation frame the train frame's categorical dtypes (audit 2026-11-20, 28-F76, part)."""

import pandas as pd

from mlframe.training.lgb_shim import _DatasetReuseMixin

realign = _DatasetReuseMixin._realign_val_columns


def test_categorical_val_column_takes_the_train_categories():
    """A val column that is a plain object column becomes the train frame's categorical dtype, categories included (28-F76)."""
    X = pd.DataFrame({"c": pd.Categorical(["a", "b", "c"]), "x": [1.0, 2.0, 3.0]})
    X_val = pd.DataFrame({"c": ["a", "c"], "x": [4.0, 5.0]})
    out = realign(X, X_val)
    assert out["c"].dtype == X["c"].dtype
    assert list(out["c"].cat.categories) == ["a", "b", "c"]
    assert out["x"].dtype == X_val["x"].dtype


def test_matching_or_missing_columns_leave_the_frame_untouched():
    """Same dtype, a column absent from val, and non-frame inputs are returned as they came (28-F76)."""
    X = pd.DataFrame({"c": pd.Categorical(["a", "b"]), "d": pd.Categorical(["u", "v"])})
    same = pd.DataFrame({"c": pd.Categorical(["a", "b"])})
    assert realign(X, same) is same
    assert realign(X.to_numpy(), same) is same


def test_unseen_value_becomes_missing_not_an_error():
    """A val value outside the train categories casts to NaN instead of raising (28-F76)."""
    X = pd.DataFrame({"c": pd.Categorical(["a", "b"])})
    out = realign(X, pd.DataFrame({"c": ["a", "z"]}))
    assert out["c"].isna().tolist() == [False, True]
