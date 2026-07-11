"""biz_value test for ``feature_selection.stochastic_bandit_selection.stochastic_bandit_selection``.

Source: 2nd_mercedes-benz-greener-manufacturing.md -- adaptive-weighted random-subset feature search, features
that consistently beat a moving-average CV baseline get their sampling weight increased (and eventually
locked in), features that consistently lose get down-weighted. This is a STOCHASTIC search, so a single-seed
comparison against a plain (non-adaptive) random-subset search at the same epoch budget is noisy -- the
correct, honest validation is a MULTI-SEED AVERAGE (does the adaptive weighting help on average, not does it
win every single run).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

from mlframe.feature_selection.stochastic_bandit_selection import stochastic_bandit_selection


def _make_overparameterized_regression(n: int, n_signal: int, n_noise: int, seed: int):
    rng = np.random.default_rng(seed)
    X_signal = rng.normal(size=(n, n_signal))
    beta = rng.normal(size=n_signal) * 3.0
    y = X_signal @ beta + rng.normal(scale=0.5, size=n)
    X_noise = rng.normal(size=(n, n_noise))
    columns = [f"s{i}" for i in range(n_signal)] + [f"n{i}" for i in range(n_noise)]
    X = pd.DataFrame(np.hstack([X_signal, X_noise]), columns=columns)
    return X, y


def _plain_random_subset_search(X, y, cv, subset_size, n_epochs, seed):
    rng = np.random.default_rng(seed)
    best_score, best_subset = -np.inf, None
    for _ in range(n_epochs):
        cols = list(rng.choice(X.columns, size=subset_size, replace=False))
        scores = []
        for train_idx, test_idx in cv.split(X):
            model = Ridge(alpha=0.1).fit(X[cols].iloc[train_idx], y[train_idx])
            scores.append(r2_score(y[test_idx], model.predict(X[cols].iloc[test_idx])))
        score = float(np.mean(scores))
        if score > best_score:
            best_score, best_subset = score, cols
    return best_subset


def test_biz_val_bandit_selection_beats_plain_random_search_on_average():
    n_seeds = 10
    bandit_scores, random_scores = [], []

    for seed in range(n_seeds):
        X, y = _make_overparameterized_regression(n=150, n_signal=3, n_noise=40, seed=seed)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
        cv = KFold(n_splits=3, shuffle=True, random_state=0)

        bandit_subset = stochastic_bandit_selection(Ridge(alpha=0.1), Xtr, ytr, scoring=r2_score, subset_size=8, n_epochs=150, cv=cv, random_state=seed)
        random_subset = _plain_random_subset_search(Xtr, ytr, cv, subset_size=8, n_epochs=150, seed=seed)

        model_bandit = Ridge(alpha=0.1).fit(Xtr[bandit_subset], ytr)
        model_random = Ridge(alpha=0.1).fit(Xtr[random_subset], ytr)
        bandit_scores.append(r2_score(yte, model_bandit.predict(Xte[bandit_subset])))
        random_scores.append(r2_score(yte, model_random.predict(Xte[random_subset])))

    mean_bandit = float(np.mean(bandit_scores))
    mean_random = float(np.mean(random_scores))

    assert mean_bandit > mean_random, f"expected adaptive-weighted search to beat plain random search on average across {n_seeds} seeds, got bandit={mean_bandit:.4f} random={mean_random:.4f}"
    assert mean_bandit >= 0.96, f"expected the bandit selector to recover a strong held-out R2, got {mean_bandit:.4f}"


def test_stochastic_bandit_selection_returns_requested_subset_size():
    X, y = _make_overparameterized_regression(n=100, n_signal=2, n_noise=10, seed=0)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    subset = stochastic_bandit_selection(Ridge(alpha=0.1), X, y, scoring=r2_score, subset_size=5, n_epochs=20, cv=cv, random_state=0)
    assert len(subset) == 5
    assert len(set(subset)) == 5
