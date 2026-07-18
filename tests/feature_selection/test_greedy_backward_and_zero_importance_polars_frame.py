"""Regression test: ``greedy_backward_elimination`` / ``iterative_zero_importance_pruning`` must accept a
polars ``DataFrame`` for ``X``, not just pandas.

Both functions' internal ``_cv_score`` used to slice CV folds via ``X.iloc[idx]`` unconditionally, which
raises ``AttributeError: 'DataFrame' object has no attribute 'iloc'`` on a polars frame. This went
undetected while ``use_greedy_backward_elimination_fs``/``use_zero_importance_pruning_fs`` were opt-in
(default False); the 2026-07-12 default-on flip surfaced it immediately via the suite's e2e conformal tests
(the suite's pre-pipeline slot is polars-native and does not force a pandas conversion before the FS step).
Fixed via a frame-library-agnostic ``_row_select`` helper (``hasattr(X, "iloc")`` dispatch, same pattern
``functional_adapters.py`` already used for column slicing).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from mlframe.feature_selection.greedy_backward_elimination import greedy_backward_elimination
from mlframe.feature_selection.zero_importance_pruning import iterative_zero_importance_pruning


def _make_polars_regression_frame(n: int = 200, seed: int = 0):
    """Make polars regression frame."""
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n).astype(np.float32)
    f1 = rng.normal(size=n).astype(np.float32)
    f2 = rng.normal(size=n).astype(np.float32)
    y = (2.0 * f0 - f1 + 0.1 * rng.normal(size=n)).astype(np.float32)
    return pl.DataFrame({"f0": f0, "f1": f1, "f2": f2}), y


def test_greedy_backward_elimination_accepts_polars_frame():
    """Greedy backward elimination accepts polars frame."""
    X, y = _make_polars_regression_frame()
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    survivors = greedy_backward_elimination(
        RandomForestRegressor(n_estimators=20, random_state=0),
        X,
        y,
        scoring=r2_score,
        cv=cv,
        min_features=1,
    )
    assert isinstance(survivors, list)
    assert set(survivors).issubset(set(X.columns))
    assert len(survivors) >= 1


def test_iterative_zero_importance_pruning_accepts_polars_frame():
    """Iterative zero importance pruning accepts polars frame."""
    X, y = _make_polars_regression_frame()
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    survivors = iterative_zero_importance_pruning(
        RandomForestRegressor(n_estimators=20, random_state=0),
        X,
        y,
        scoring=r2_score,
        cv=cv,
    )
    assert isinstance(survivors, list)
    assert set(survivors).issubset(set(X.columns))
    assert len(survivors) >= 1
