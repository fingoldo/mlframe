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


def test_transform_dataframe_wrong_cols_also_raises():
    """The shape check also fires for DataFrame inputs with wrong column count.
    The pre-existing missing-name check would have caught some cases but
    not all (e.g. wrong column count with overlapping names).
    """
    sel, _ = _fit()
    # 5-col DataFrame with a, b, c, d, e (all selected names present)
    rng = np.random.default_rng(1)
    X_wrong = pd.DataFrame(
        rng.standard_normal((200, 5)),
        columns=["a", "b", "c", "d", "e"],
    )
    with pytest.raises(ValueError, match="features"):
        sel.transform(X_wrong)


def test_n_features_in_attribute_set_after_fit():
    """sklearn convention: ``n_features_in_`` MUST be an attribute on
    fitted estimators per BaseEstimator._check_n_features.
    """
    sel, _ = _fit()
    assert hasattr(sel, "n_features_in_")
    assert sel.n_features_in_ == 4
