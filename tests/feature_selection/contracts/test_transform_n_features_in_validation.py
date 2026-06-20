"""Wave 9.1 loop-iter-19 regression: ``MRMR.transform`` must enforce
sklearn's ``n_features_in_`` shape contract.

Pre-fix at ``_mrmr_validate_transform.py``: when ``X`` was a non-pandas
array (e.g. ``np.ndarray``), ``transform()`` accepted ANY column count
and silently positional-sliced ``X[:, support_]`` from whatever slots
the support array pointed at. Confirmed live: fit on a 4-col DataFrame
then transform a 3-col, 5-col, 7-col ndarray returned shape (n, k)
with no error or warning.

Effect: silent column-position corruption. Concrete production
scenario: an ETL step prepended an ID column before predict-time;
``MRMR.transform`` happily indexed the ID column as if it were
feature 0. The downstream model received wrong-position features
without any signal that something was off.

Fix: early ``n_features_in_`` shape check in ``transform()``, mirroring
sklearn's canonical ``BaseEstimator._check_n_features`` behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _fit():
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] > 0).astype(np.int64), name="y")
    return MRMR(verbose=0).fit(X, y), X


@pytest.mark.parametrize("wrong_cols", [3, 5, 7])
def test_transform_ndarray_wrong_cols_raises(wrong_cols):
    """ndarray with wrong column count must raise ValueError, not
    silently return positional-sliced garbage.
    """
    sel, _ = _fit()
    X_wrong = np.random.standard_normal((200, wrong_cols))
    with pytest.raises(ValueError, match="features"):
        sel.transform(X_wrong)


def test_transform_ndarray_correct_cols_succeeds():
    """ndarray with the correct column count works as before."""
    sel, _ = _fit()
    X_ok = np.random.standard_normal((200, 4))
    out = sel.transform(X_ok)
    assert out.shape[0] == 200
    assert out.shape[1] >= 1


def test_transform_dataframe_correct_cols_succeeds():
    """DataFrame transform unaffected by the fix."""
    sel, X = _fit()
    out = sel.transform(X)
    assert out.shape[0] == 200
    assert out.shape[1] >= 1


def test_transform_dataframe_extra_columns_realigned():
    """sklearn-canonical: DataFrame with EXTRA columns is allowed
    (realigned by name; fit-time columns must all be present).
    Pre-iter-19 the test asserted this raises - that was wrong per
    sklearn ``_check_feature_names(reset=False)`` semantics. The
    iter-19 strict shape check applies only to unnamed arrays.
    """
    sel, _ = _fit()
    rng = np.random.default_rng(1)
    X_extra = pd.DataFrame(
        rng.standard_normal((200, 5)),
        columns=["a", "b", "c", "d", "e"],  # all fit-time cols + extra
    )
    # Must succeed (extra cols dropped by name realignment).
    out = sel.transform(X_extra)
    assert out.shape[0] == 200
    # Output width is the full selected set INCLUDING engineered features (MRMR does FE by default),
    # so it equals len(get_feature_names_out()) -- not len(support_), which counts only raw survivors.
    assert out.shape[1] == len(sel.get_feature_names_out())


def test_transform_dataframe_missing_column_raises():
    """DataFrame missing a fit-time selected column MUST raise."""
    sel, _ = _fit()
    rng = np.random.default_rng(1)
    # Drop 'a' (likely selected by fit)
    X_missing = pd.DataFrame(
        rng.standard_normal((200, 3)),
        columns=["b", "c", "d"],
    )
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        sel.transform(X_missing)


def test_n_features_in_attribute_set_after_fit():
    """sklearn convention: ``n_features_in_`` MUST be an attribute on
    fitted estimators per BaseEstimator._check_n_features.
    """
    sel, _ = _fit()
    assert hasattr(sel, "n_features_in_")
    assert sel.n_features_in_ == 4
