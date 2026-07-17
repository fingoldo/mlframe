"""Duplicate column-name input-validation guards across the feature selectors.

Generalises the GroupAwareMRMR duplicate-name fix: a selector that iterates columns by
label (``X[name]``) returns a DataFrame instead of a Series on duplicate names, and the
downstream ``.dtype`` access raises ``AttributeError`` (or the wrapped booster raises a
cryptic error). Each selector below now either handles duplicate names cleanly
(``pre_screen``) or surfaces a clear, actionable ``ValueError`` at fit entry (RFECV,
BorutaShap, ShapProxiedFS, HybridSelector) -- never an uncaught low-level crash.

Verified pre-fix (via ``git stash`` of the prod file in this worktree): each test below
raised ``AttributeError: 'DataFrame' object has no attribute 'dtype'`` / ``LightGBMError``
before the guard landed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _dup_frame(n: int = 80):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"x": rng.randn(n), "x2": rng.randn(n), "c": rng.randn(n)})
    X.columns = ["x", "x", "c"]  # duplicate label
    y = pd.Series((X.iloc[:, 0].to_numpy() > 0).astype(int), name="y")
    return X, y


def test_pre_screen_handles_duplicate_column_names():
    """``compute_unsupervised_drops`` iterates positionally; duplicate names no longer crash."""
    from mlframe.feature_selection.pre_screen import compute_unsupervised_drops

    X, _ = _dup_frame()
    # No raise; a non-degenerate frame yields no drops.
    drops = compute_unsupervised_drops(X)
    assert drops == []

    # A constant duplicate-named block is still detected (dropped by name).
    Xc = X.copy()
    Xc.iloc[:, 0] = 1.0
    Xc.iloc[:, 1] = 1.0
    assert "x" in compute_unsupervised_drops(Xc)


def test_rfecv_raises_clear_error_on_duplicate_column_names():
    from sklearn.tree import DecisionTreeClassifier

    from mlframe.feature_selection.wrappers.rfecv import RFECV

    X, y = _dup_frame()
    sel = RFECV(estimator=DecisionTreeClassifier(max_depth=2, random_state=0), cv=2)
    with pytest.raises(ValueError, match="duplicate column names not supported"):
        sel.fit(X, y)


def test_boruta_shap_raises_clear_error_on_duplicate_column_names():
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = _dup_frame()
    sel = BorutaShap(n_trials=5, classification=True, verbose=False)
    with pytest.raises(ValueError, match="duplicate column names not supported"):
        sel.fit(X, y)


def test_shap_proxied_fs_raises_clear_error_on_duplicate_column_names():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _dup_frame()
    sel = ShapProxiedFS(verbose=False)
    with pytest.raises(ValueError, match="duplicate column names not supported"):
        sel.fit_transform(X, y)


def test_hybrid_selector_raises_clear_error_on_duplicate_column_names():
    from mlframe.feature_selection.hybrid_selector import HybridSelector

    X, y = _dup_frame()
    sel = HybridSelector(prescreen=True)
    with pytest.raises(ValueError, match="duplicate column names not supported"):
        sel.fit(X, y)
