"""biz_value test for ``feature_selection.greedy_backward_elimination.greedy_backward_elimination``.

Source: dd_1st_pover-t-tests.md -- permutation importance used not just to rank but to actually decide
removal: "removed the ones for which we registered a score improvement" when shuffled/dropped. On a small,
overparameterized regression with many pure-noise columns, a model fit on ALL features overfits the noise;
greedily removing whichever single feature most improves mean CV R2, repeated until no removal helps, should
recover a materially better held-out R2 than the full feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

from mlframe.feature_selection.greedy_backward_elimination import greedy_backward_elimination


def _make_overparameterized_regression(n: int, n_signal: int, n_noise: int, seed: int):
    """Make overparameterized regression."""
    rng = np.random.default_rng(seed)
    X_signal = rng.normal(size=(n, n_signal))
    beta = rng.normal(size=n_signal)
    y = X_signal @ beta + rng.normal(scale=0.5, size=n)
    X_noise = rng.normal(size=(n, n_noise))
    columns = [f"s{i}" for i in range(n_signal)] + [f"n{i}" for i in range(n_noise)]
    X = pd.DataFrame(np.hstack([X_signal, X_noise]), columns=columns)
    return X, y


def test_biz_val_greedy_backward_elimination_beats_full_feature_set():
    """Biz val greedy backward elimination beats full feature set."""
    X, y = _make_overparameterized_regression(n=80, n_signal=3, n_noise=40, seed=2)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)

    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    survivors = greedy_backward_elimination(Ridge(alpha=0.1), Xtr, ytr, scoring=r2_score, cv=cv, min_features=1)

    assert len(survivors) < X.shape[1], "expected the greedy search to drop at least one noise feature"
    assert all(s.startswith("s") for s in survivors[:3]) or set(f"s{i}" for i in range(3)).issubset(survivors)

    model_full = Ridge(alpha=0.1).fit(Xtr, ytr)
    model_selected = Ridge(alpha=0.1).fit(Xtr[survivors], ytr)
    r2_full = float(r2_score(yte, model_full.predict(Xte)))
    r2_selected = float(r2_score(yte, model_selected.predict(Xte[survivors])))

    assert (
        r2_selected > r2_full + 0.05
    ), f"expected greedy backward elimination to improve held-out R2 by >=0.05, got selected={r2_selected:.4f} full={r2_full:.4f}"


def test_greedy_backward_elimination_respects_min_features():
    """Greedy backward elimination respects min features."""
    X, y = _make_overparameterized_regression(n=60, n_signal=2, n_noise=15, seed=3)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    survivors = greedy_backward_elimination(Ridge(alpha=0.1), X, y, scoring=r2_score, cv=cv, min_features=10)
    assert len(survivors) >= 10


def test_greedy_backward_elimination_no_removal_helps_returns_all_features():
    # every column is genuinely informative -> no single removal should improve CV score.
    """Greedy backward elimination no removal helps returns all features."""
    rng = np.random.default_rng(4)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (X["a"] * 2 + X["b"] * 2 + X["c"] * 2 + rng.normal(scale=0.05, size=n)).to_numpy()
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    survivors = greedy_backward_elimination(Ridge(alpha=0.01), X, y, scoring=r2_score, cv=cv, min_features=1)
    assert set(survivors) == {"a", "b", "c"}


def test_greedy_backward_elimination_matches_fresh_per_call_kfold_reference():
    """``_cv_score`` used to call ``cv.split(X)`` fresh on every column-drop candidate; the fold indices only ever
    depend on row count (never on which columns remain), so re-deriving them per candidate was pure wasted
    shuffling -- the fix precomputes the fold indices ONCE per ``greedy_backward_elimination`` call and reuses
    them. Pin bit-identical selection against a reference that re-derives a KFold(shuffle=True, random_state=0)
    split fresh for every single candidate (the pre-fix behavior), proving the hoist changed no result."""
    from sklearn.base import clone

    def _reference_score(estimator, frame, y_arr, n_splits, scoring):
        """Reference score."""
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        row_select = (lambda idx: frame.iloc[idx]) if hasattr(frame, "iloc") else (lambda idx: frame[idx])
        scores = []
        for train_idx, test_idx in cv.split(frame):
            model = clone(estimator)
            model.fit(row_select(train_idx), y_arr[train_idx])
            preds = model.predict(row_select(test_idx))
            scores.append(scoring(y_arr[test_idx], preds))
        return float(np.mean(scores))

    def _reference_greedy_backward_elimination(estimator, X, y, scoring, n_splits, min_features=1, tol=0.0):
        """Reference greedy backward elimination."""
        y_arr = np.asarray(y)
        remaining = list(X.columns)
        current_score = _reference_score(estimator, X[remaining], y_arr, n_splits, scoring)
        while len(remaining) > min_features:
            best_candidate, best_score = None, current_score
            for col in remaining:
                candidate_cols = [c for c in remaining if c != col]
                score = _reference_score(estimator, X[candidate_cols], y_arr, n_splits, scoring)
                if score > best_score + tol:
                    best_score, best_candidate = score, col
            if best_candidate is None:
                break
            remaining.remove(best_candidate)
            current_score = best_score
        return remaining

    X, y = _make_overparameterized_regression(n=80, n_signal=3, n_noise=12, seed=5)
    reference = _reference_greedy_backward_elimination(Ridge(alpha=0.1), X, y, r2_score, n_splits=4)
    actual = greedy_backward_elimination(Ridge(alpha=0.1), X, y, scoring=r2_score, cv=KFold(n_splits=4, shuffle=True, random_state=0), min_features=1)
    assert actual == reference, f"hoisted-fold-precompute changed the selection: {actual} != {reference}"
