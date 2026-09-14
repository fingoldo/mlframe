"""A frame with non-string column labels must fit (IMPL-9, found while fixing mrmr_audit_2026-09-14 CORE-2).

The greedy screen renders each candidate's name with ``"-".join(factors_names[i] ...)``, which raises ``TypeError`` for any non-string label.
``pd.DataFrame(ndarray)`` produces exactly such labels (0..p-1), so any caller that wraps an array without naming its columns hit it; it
is also why every stability-selection replicate on ndarray input failed and the fit silently fell back to classic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(labels, seed=0, n=300):
    """Six columns under the given labels, with signal on the first two."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(labels))), columns=labels)
    y = ((X.iloc[:, 0] + X.iloc[:, 1] + 0.3 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def _fit(X, y):
    """A light, deterministic classic fit."""
    return MRMR(verbose=0, random_seed=0, fe_max_steps=0, fe_hinge_enable=False, dcd_enable=False, build_friend_graph=False).fit(X, y)


def test_integer_column_labels_fit_and_transform():
    """pandas' default integer labels must fit, select the signal columns, and transform by the same labels."""
    X, y = _frame(list(range(6)))
    m = _fit(X, y)
    selected = [m.feature_names_in_[int(i)] for i in m.support_]
    assert {0, 1} <= set(selected), f"selected {selected}"
    assert list(m.transform(X).columns) == selected


def test_string_labels_select_the_same_columns():
    """Control: the same data under string labels selects the same positions, so the label type does not change the result."""
    X_int, y = _frame(list(range(6)))
    X_str, _ = _frame([f"c{i}" for i in range(6)])
    m_int, m_str = _fit(X_int, y), _fit(X_str, y)
    assert sorted(int(i) for i in m_int.support_) == sorted(int(i) for i in m_str.support_)
