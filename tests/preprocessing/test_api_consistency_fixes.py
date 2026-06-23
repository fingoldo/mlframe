"""Regression tests for public-API consistency fixes in preprocessing.

Covers:
  API-P0-2 -- apply_features_cleaning default does NOT mutate the caller's df.
  API35    -- clusterize uses ``true_labels is not None`` (no ValueError on ndarray input).
  API36    -- prepare_df_for_xgboost is non-mutating by default (matches prepare_df_for_catboost).
"""

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- API-P0-2
def test_apixp0_apply_features_cleaning_default_does_not_mutate_caller_df():
    from mlframe.preprocessing.cleaning import apply_features_cleaning

    df = pd.DataFrame({"a": [1, 2, 3], "b": [9, 9, 9]})
    features_cleaning = {
        "features_transforms": {"a": {2: 20}},
        "constant_features": ["b"],
    }
    before = df.copy(deep=True)

    out = apply_features_cleaning(df, features_cleaning)

    # Caller's df untouched: same columns, same values.
    pd.testing.assert_frame_equal(df, before)
    # Returned frame carries the replacement + dropped constant column.
    assert out["a"].tolist() == [1, 20, 3]
    assert "b" not in out.columns


def test_apixp0_apply_features_cleaning_update_data_true_mutates():
    from mlframe.preprocessing.cleaning import apply_features_cleaning

    df = pd.DataFrame({"a": [1, 2, 3], "b": [9, 9, 9]})
    features_cleaning = {
        "features_transforms": {"a": {2: 20}},
        "constant_features": ["b"],
    }
    out = apply_features_cleaning(df, features_cleaning, update_data=True)
    assert out is df
    assert df["a"].tolist() == [1, 20, 3]
    assert "b" not in df.columns


# --------------------------------------------------------------------------- API35
def test_api35_clusterize_accepts_ndarray_true_labels(monkeypatch):
    """Pre-fix ``if true_labels:`` raised 'truth value of an array is ambiguous' on the natural ndarray input."""
    import mlframe.preprocessing.cluster as cluster

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    true_labels = rng.integers(0, 3, size=30)  # ndarray -> ambiguous truth value pre-fix
    assert isinstance(true_labels, np.ndarray)

    labels = cluster.clusterize(
        X=X, true_labels=true_labels,
        show_plot=False, show_metrics=True, list_members=False,
    )
    assert labels is not None
    assert len(labels) == 30


# --------------------------------------------------------------------------- API36
def test_api36_prepare_df_for_xgboost_non_mutating_by_default():
    from mlframe.preprocessing.transforms import prepare_df_for_xgboost

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    cat_features = ["b"]
    before_dtype = df["b"].dtype

    out = prepare_df_for_xgboost(df, cat_features=cat_features)

    # Returned frame has 'b' as Categorical.
    assert isinstance(out["b"].dtype, pd.CategoricalDtype)
    # Caller's df dtype untouched (non-mutating default, matching prepare_df_for_catboost contract).
    assert df["b"].dtype == before_dtype
    assert not isinstance(df["b"].dtype, pd.CategoricalDtype)
    # Caller's cat_features list also untouched.
    assert cat_features == ["b"]


def test_api36_prepare_df_for_xgboost_inplace_true_mutates():
    from mlframe.preprocessing.transforms import prepare_df_for_xgboost

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    out = prepare_df_for_xgboost(df, cat_features=["b"], inplace=True)
    assert out is df
    assert isinstance(df["b"].dtype, pd.CategoricalDtype)
