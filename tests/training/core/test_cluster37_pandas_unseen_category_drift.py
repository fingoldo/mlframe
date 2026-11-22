"""Cluster 37 regression: unseen-category drift is computed for pandas frames too (audit 2026-11-20, finding 37-F19)."""

import pandas as pd

from mlframe.training.core._phase_drift_snapshot import _compute_drift_rows_pandas


def test_pandas_val_only_category_is_counted():
    """A category present only in val is reported as val-only drift on pandas frames (37-F19)."""
    train = pd.DataFrame({"c": ["a", "b", "a"]})
    val = pd.DataFrame({"c": ["a", "z"]})
    test = pd.DataFrame({"c": ["b", "y", "y"]})
    rows = _compute_drift_rows_pandas(train, val, test, [("c", 2)])
    assert rows == [("c", 2, 1, 1)]


def test_pandas_no_drift_when_all_categories_seen_in_train():
    """Control: categories all present in train produce zero unseen counts."""
    train = pd.DataFrame({"c": ["a", "b"]})
    val = pd.DataFrame({"c": ["a"]})
    test = pd.DataFrame({"c": ["b", "a"]})
    assert _compute_drift_rows_pandas(train, val, test, [("c", 2)]) == [("c", 2, 0, 0)]
