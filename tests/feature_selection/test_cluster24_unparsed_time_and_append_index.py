"""Cluster 24 regressions: unparseable time labels are reported, and appended columns must share the frame's index (audit 2026-11-20, findings 24-F46 and 24-F49).

Object timestamps that fail to parse became NaT and sorted last without a word, so the folds silently stopped being chronological. The append helper
aligned new columns on the index, so a mismatched index appended NaN columns silently.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._append_columns import append_columns
from mlframe.feature_selection._cv_splitters import chronological_order


def test_unparseable_time_labels_warn_and_sort_last():
    """A label such as 'Q1-2024' is reported, and its row sorts after every parseable timestamp (24-F46)."""
    labels = pd.Series(["2024-01-02", "Q1-2024", "2024-01-01"], dtype=object)
    with pytest.warns(UserWarning, match="could not be parsed"):
        order = chronological_order(labels)
    assert list(order) == [2, 0, 1]


def test_parseable_time_labels_do_not_warn(recwarn):
    """Fully parseable object timestamps keep their order and raise no warning (24-F46)."""
    labels = pd.Series(["2024-01-02", "2024-01-01"], dtype=object)
    order = chronological_order(labels)
    assert list(order) == [1, 0]
    assert not [w for w in recwarn if "could not be parsed" in str(w.message)]


def test_append_rejects_mismatched_index_and_accepts_shared_index():
    """A new_cols frame with a different index raises instead of appending NaN; the matching index appends the columns (24-F49)."""
    X = pd.DataFrame({"a": np.arange(4.0)}, index=[10, 11, 12, 13])
    shifted = pd.DataFrame({"f": np.arange(4.0)}, index=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="share X's index"):
        append_columns(X, shifted)

    same = pd.DataFrame({"f": np.arange(4.0) * 2}, index=X.index)
    out = append_columns(X, same)
    assert list(out.columns) == ["a", "f"]
    np.testing.assert_array_equal(out["f"].to_numpy(), same["f"].to_numpy())
