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
    rng = np.random.default_rng(seed)
    X_signal = rng.normal(size=(n, n_signal))
    beta = rng.normal(size=n_signal)
    y = X_signal @ beta + rng.normal(scale=0.5, size=n)
    X_noise = rng.normal(size=(n, n_noise))
    columns = [f"s{i}" for i in range(n_signal)] + [f"n{i}" for i in range(n_noise)]
    X = pd.DataFrame(np.hstack([X_signal, X_noise]), columns=columns)
    return X, y


def test_biz_val_greedy_backward_elimination_beats_full_feature_set():
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

    assert r2_selected > r2_full + 0.05, f"expected greedy backward elimination to improve held-out R2 by >=0.05, got selected={r2_selected:.4f} full={r2_full:.4f}"


def test_greedy_backward_elimination_respects_min_features():
    X, y = _make_overparameterized_regression(n=60, n_signal=2, n_noise=15, seed=3)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    survivors = greedy_backward_elimination(Ridge(alpha=0.1), X, y, scoring=r2_score, cv=cv, min_features=10)
    assert len(survivors) >= 10


def test_greedy_backward_elimination_no_removal_helps_returns_all_features():
    # every column is genuinely informative -> no single removal should improve CV score.
    rng = np.random.default_rng(4)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (X["a"] * 2 + X["b"] * 2 + X["c"] * 2 + rng.normal(scale=0.05, size=n)).to_numpy()
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    survivors = greedy_backward_elimination(Ridge(alpha=0.01), X, y, scoring=r2_score, cv=cv, min_features=1)
    assert set(survivors) == {"a", "b", "c"}
