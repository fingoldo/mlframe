"""Regression: reports.compare_estimators_by_test_set indexed y_test positionally via
y_test[train_indices], which is LABEL-based for a pandas Series with a non-default index ->
silent misalignment vs X_test.iloc[train_indices]. The fix uses .iloc for positional access.

This sensor pins the indexing SEMANTICS used by the fixed code path: with a non-default-index
Series, positional .iloc must align with X.iloc, whereas raw []-indexing would misalign."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _positional_select(y, indices):
    # Mirrors the fixed line in reports.py.
    """Helper that positional select."""
    return y.iloc[indices] if hasattr(y, "iloc") else y[indices]


def test_positional_indexing_aligns_with_iloc_on_nondefault_index():
    """Positional indexing aligns with iloc on nondefault index."""
    n = 10
    X = pd.DataFrame({"f": np.arange(n) * 10.0}, index=[100 + i for i in range(n)])
    # y constructed so label-based [] would pick the WRONG rows on this non-default index.
    y = pd.Series(np.arange(n), index=[100 + i for i in range(n)])

    train_indices = [0, 2, 4, 6]
    X_sel = X.iloc[train_indices, :]
    y_sel = _positional_select(y, train_indices)

    # Positional alignment: y values must equal the positions selected from X, not the labels.
    assert list(y_sel.values) == train_indices
    # And they line up with the X rows positionally (f == position*10).
    np.testing.assert_array_equal(X_sel["f"].values, np.array(train_indices) * 10.0)


def test_label_based_indexing_would_misalign_proving_bug():
    """Label based indexing would misalign proving bug."""
    n = 10
    y = pd.Series(np.arange(n), index=[100 + i for i in range(n)])
    train_indices = [0, 2, 4, 6]
    # Pre-fix code did y[train_indices] -> label lookup. Labels 0/2/4/6 are absent here
    # (index starts at 100), so the buggy form raises (or silently mismatches when labels
    # happen to exist). Either way it is NOT positional and differs from the .iloc result.
    try:
        bad = y[train_indices]
        misaligned = list(bad.values) != train_indices
    except KeyError:
        misaligned = True
    assert misaligned
