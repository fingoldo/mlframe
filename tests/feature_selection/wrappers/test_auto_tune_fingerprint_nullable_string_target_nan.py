"""Regression tests: DataFingerprint must handle nullable columns, string labels and report NaN in sparse columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.wrappers._auto_tune import DataFingerprint


def _frame(n=200, seed=0):
    """Numeric frame with a binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"].to_numpy() > 0).astype(int)
    return X, y


def test_nullable_int_column_with_na_does_not_crash():
    """A pandas Int64 column holding pd.NA must be read as float with NaN, and flag has_nan."""
    X, y = _frame()
    X["n"] = pd.array(np.arange(len(X)), dtype="Int64")
    X.loc[3, "n"] = pd.NA
    fp = DataFingerprint.from_xy(X, y)
    assert fp.has_nan is True
    assert fp.n_features == 3


def test_string_class_labels_do_not_crash():
    """String class labels must be factorised for the correlation signals rather than raising on astype(float)."""
    X, y = _frame()
    labels = np.where(y == 1, "pos", "neg")
    fp = DataFingerprint.from_xy(X, labels)
    assert fp.target_type == "binary"
    assert fp.max_abs_corr_to_y > 0.5


def test_mostly_nan_column_still_sets_has_nan():
    """A column with fewer than 10 finite rows used to be skipped before the NaN check, hiding its NaN."""
    X, y = _frame()
    sparse = np.full(len(X), np.nan)
    sparse[:4] = 1.0
    X["sparse"] = sparse
    assert DataFingerprint.from_xy(X, y).has_nan is True


def test_all_nan_column_sets_has_nan():
    """An all-NaN column counts as constant and used to be skipped before the NaN check."""
    X, y = _frame()
    X["allnan"] = np.nan
    assert DataFingerprint.from_xy(X, y).has_nan is True
