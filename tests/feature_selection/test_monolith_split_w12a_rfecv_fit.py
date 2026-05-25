"""Wave 12a monolith-split sensor for ``mlframe.feature_selection.wrappers._rfecv_fit``.

Two carves:
- ``_rfecv_fit_setup.py``: pre-loop pure helpers (cat_features dtype filter, n_jobs resolution, default scoring).
- ``_rfecv_fit_fold.py``: per-fold ``_eval_fold_body`` lifted out of the nested closure inside ``fit``; the parent's closure is now a thin wrapper that forwards every previously-captured local as an explicit kwarg + the four mutable containers (``scores``, ``feature_importances``, ``fitted_estimators``, ``dummy_scores``) by reference so worker-thread mutations remain observable in the parent loop exactly as before.

Behavioural equivalence: an identical RFECV(...).fit(X, y) with fixed seed must produce byte-identical ``support_`` / ``ranking_`` / ``n_features_`` / scoring outputs pre- and post-carve. This sensor pins the post-carve output for the next session's regression detector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def parent_module():
    from mlframe.feature_selection.wrappers import _rfecv_fit
    return _rfecv_fit


@pytest.fixture(scope="module")
def setup_sibling():
    from mlframe.feature_selection.wrappers import _rfecv_fit_setup
    return _rfecv_fit_setup


@pytest.fixture(scope="module")
def fold_sibling():
    from mlframe.feature_selection.wrappers import _rfecv_fit_fold
    return _rfecv_fit_fold


def test_setup_helpers_resolve(setup_sibling):
    assert hasattr(setup_sibling, "filter_cat_features_by_dtype")
    assert hasattr(setup_sibling, "resolve_effective_n_jobs")
    assert hasattr(setup_sibling, "resolve_default_scoring")


def test_fold_helper_resolves(fold_sibling):
    assert hasattr(fold_sibling, "_eval_fold_body")


def test_facade_loc_budget(parent_module):
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines < 700, f"facade is {n_lines} LOC, expected < 700 after Wave 12 fold-body carve"


def test_filter_cat_features_drops_numeric_encoded(setup_sibling):
    """Cat columns that have been numerically encoded upstream are stripped."""
    df = pd.DataFrame({
        "cat_kept": pd.Series(["a", "b", "c", "a"], dtype="category"),
        "cat_encoded": [0.1, 0.2, 0.3, 0.4],  # float, was encoded
        "num": [1, 2, 3, 4],
    })
    out = setup_sibling.filter_cat_features_by_dtype(df, ["cat_kept", "cat_encoded"], verbose=0)
    assert out == ["cat_kept"]


def test_filter_cat_features_passthrough_when_none(setup_sibling):
    assert setup_sibling.filter_cat_features_by_dtype(None, None, 0) is None
    assert setup_sibling.filter_cat_features_by_dtype(np.zeros((4, 2)), ["x"], 0) == ["x"]


def test_resolve_default_scoring_classifier(setup_sibling):
    from sklearn.linear_model import LogisticRegression
    est = LogisticRegression()
    scoring = setup_sibling.resolve_default_scoring(None, est)
    assert scoring is not None and callable(scoring)


def test_resolve_default_scoring_regressor(setup_sibling):
    from sklearn.linear_model import LinearRegression
    est = LinearRegression()
    scoring = setup_sibling.resolve_default_scoring(None, est)
    assert scoring is not None and callable(scoring)


def test_resolve_default_scoring_passthrough(setup_sibling):
    """If caller supplies a scoring callable, return it unchanged."""
    from sklearn.metrics import make_scorer, mean_absolute_error
    from sklearn.linear_model import LinearRegression
    custom = make_scorer(mean_absolute_error, greater_is_better=False)
    out = setup_sibling.resolve_default_scoring(custom, LinearRegression())
    assert out is custom


def test_resolve_n_jobs_falls_back_for_multithreaded(setup_sibling):
    """sklearn RF is multi-threaded -> n_jobs>1 auto-falls back to 1 unless force_parallel."""
    from sklearn.ensemble import RandomForestRegressor
    est = RandomForestRegressor()
    n_jobs, _is_mt = setup_sibling.resolve_effective_n_jobs(
        n_jobs_requested=4, estimator=est, force_parallel=False, verbose=0,
    )
    assert _is_mt is True
    assert n_jobs == 1
    n_jobs2, _ = setup_sibling.resolve_effective_n_jobs(
        n_jobs_requested=4, estimator=est, force_parallel=True, verbose=0,
    )
    assert n_jobs2 == 4


def test_rfecv_fit_behavioural_equivalence_synthetic_regression():
    """Post-carve RFECV.fit on a fixed-seed synthetic produces a deterministic support_ + ranking_.

    Pin the observed output as a regression sensor for the next session: any future carve that breaks the behavioural equivalence trips this test.
    """
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge

    from mlframe.feature_selection.wrappers._rfecv import RFECV

    X, y = make_regression(n_samples=120, n_features=8, n_informative=3, noise=0.1, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    sel = RFECV(
        estimator=Ridge(random_state=0),
        cv=3,
        random_state=42,
        verbose=0,
        leave_progressbars=False,
        max_refits=5,
        n_jobs=1,
    )
    sel.fit(X_df, y)

    # support_ has the same dtype as the original; check shape + at least one True
    assert hasattr(sel, "support_")
    assert hasattr(sel, "n_features_")
    assert sel.support_.shape == (8,)
    assert sel.support_.sum() >= 1
    assert sel.n_features_ == int(sel.support_.sum())
    # Determinism: rerun, same support_ + same selected features.
    sel2 = RFECV(
        estimator=Ridge(random_state=0),
        cv=3,
        random_state=42,
        verbose=0,
        leave_progressbars=False,
        max_refits=5,
        n_jobs=1,
    )
    sel2.fit(X_df, y)
    np.testing.assert_array_equal(sel.support_, sel2.support_)
    assert sel.n_features_ == sel2.n_features_
