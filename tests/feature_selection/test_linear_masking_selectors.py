"""Regression: relevance-only selectors kept linearly redundant columns (singular / near-singular selected Gram).

ACE accepted all of an exact identity ``x3 = 2*x1 - x2`` (every member clears the contrast bar), and forward_select
added the second member of a corr~0.999 pair because its CV gain looked like any other fold jitter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlframe.feature_selection._linear_masking import drop_linearly_masked, linear_r2
from mlframe.feature_selection.ace import ace_select
from mlframe.feature_selection.forward_select import forward_select


def _frame(seed=0, n=600):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": 2.0 * x1 - x2})
    y = ((x1 + x2 + 0.25 * rng.normal(size=n)) > 0).astype(np.int64)
    return X, y


def test_linear_r2_detects_identity_and_ignores_independent():
    X, _ = _frame()
    a = X.to_numpy()
    assert linear_r2(a[:, :2], a[:, 2]) > 0.999999
    assert linear_r2(a[:, :1], a[:, 1]) < 0.05
    mask = drop_linearly_masked(a, np.ones(3, bool), np.array([0.3, 0.2, 0.5]))
    assert mask.sum() == 2 and mask[2]  # most important member kept, one of the others masked


def test_ace_does_not_keep_rank_deficient_triple():
    X, y = _frame()
    est = RandomForestClassifier(n_estimators=40, random_state=0, n_jobs=1)
    res = ace_select(X, y, estimator=est, n_replicates=5, n_masking_rounds=1)
    sub = X[res.selected_features].to_numpy()
    assert len(res.selected_features) >= 2
    assert np.linalg.matrix_rank(sub - sub.mean(0), tol=1e-8) == sub.shape[1], res.selected_features


def test_forward_select_skips_near_duplicate_of_selected():
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "a_dup": a + 0.01 * rng.normal(size=n)})
    y = (a > 0).astype(int)
    from sklearn.linear_model import LogisticRegression

    sel = forward_select(X, y, lambda: LogisticRegression(), cv=3, min_improvement=-1.0)
    assert len(sel) == 1, sel
    assert len(forward_select(X, y, lambda: LogisticRegression(), cv=3, min_improvement=-1.0, mask_redundant=False)) == 2
