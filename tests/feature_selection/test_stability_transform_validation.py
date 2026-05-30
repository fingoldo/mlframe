"""Wave 9.1 loop-iter-42 regression: ``StabilityMRMR.transform`` must
validate column semantics at transform time.

Pre-fix at ``stability.py:145-148``::

    def transform(self, X, y=None):
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]
        return X[:, self.support_]

The function positional-indexed via ``support_`` with NO check that the
columns at transform time match the columns at fit time. Three
silent-failure modes:

1. Column reorder: ``support_=[1,2,3]`` (for fit-time cols b,c,d) on a
   transform-time frame ordered ``[d,c,b,a]`` returns columns 1,2,3 of
   the reordered frame = ``c,b,a`` - but the support_ indices were
   computed assuming the FIT-time order.
2. Column rename: rename ``b`` to ``b_renamed`` -> transform returns
   the column at position 1 silently labelled ``b_renamed``.
3. Column drop: drop ``a`` -> ``X[:, [1,2,3]]`` returns positions
   1,2,3 of the dropped frame = ``c,d,...`` - wrong slice with no error.

Severity: high. Downstream models silently see features under wrong
names; columns dropped or reordered upstream of transform produce
incorrect predictions with no warning.

Fix at stability.py:145 (sklearn ``_check_feature_names(reset=False)``
semantics):
- DataFrame with column names: reorder by name if all fit columns
  present, raise on missing.
- ndarray (no names): width check vs ``n_features_in_``, raise on
  mismatch.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _fit():
    from mlframe.feature_selection.filters.stability import StabilityMRMR
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
        "d": rng.standard_normal(n),
    })
    y = pd.Series((X["b"] + X["c"] > 0).astype(np.int64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = StabilityMRMR(
            estimator=MRMR(verbose=0), n_bootstraps=3,
            sample_fraction=0.5,
        ).fit(X, y)
    return sel, X


def test_transform_reordered_columns_realigned():
    """Reordering columns must NOT change the output (sklearn
    convention - realign by name).
    """
    sel, X = _fit()
    out_ordered = sel.transform(X)
    # Same data, columns shuffled.
    X_shuffled = X[["d", "c", "b", "a"]]
    out_shuffled = sel.transform(X_shuffled)
    # Both outputs must reference the SAME underlying columns by name.
    if hasattr(out_ordered, "columns"):
        assert list(out_ordered.columns) == list(out_shuffled.columns), (
            f"transform on reordered columns gave different output names: "
            f"{list(out_ordered.columns)} vs {list(out_shuffled.columns)}"
        )
        # And the values per column must match.
        for col in out_ordered.columns:
            np.testing.assert_allclose(
                out_ordered[col].values,
                out_shuffled[col].values,
            )


def test_transform_missing_fit_column_raises():
    """Dropping a fit-time column that's in support_ must raise."""
    sel, X = _fit()
    # Drop a column that was likely selected ('b' or 'c'); choose one
    # by inspecting the fitted feature names.
    selected_names = [
        sel.feature_names_in_[i] for i in sel.support_
    ]
    if not selected_names:
        pytest.skip("Empty support_; cannot test missing-column drop")
    drop_col = selected_names[0]
    X_dropped = X.drop(columns=[drop_col])
    with pytest.raises(ValueError, match="missing"):
        sel.transform(X_dropped)


def test_transform_extra_columns_allowed_if_fit_set_present():
    """Extra columns AT transform time are fine as long as all fit-time
    columns are present (sklearn realigns by name).
    """
    sel, X = _fit()
    X_extra = X.copy()
    X_extra["extra_col"] = np.zeros(len(X))
    # Must not raise.
    sel.transform(X_extra)


def test_transform_ndarray_wrong_width_raises():
    """ndarray with different column count must raise (no names to
    realign).
    """
    sel, _ = _fit()
    rng = np.random.default_rng(1)
    X_wrong = rng.standard_normal((200, 3))  # fit saw 4 cols
    with pytest.raises(ValueError, match="features"):
        sel.transform(X_wrong)
