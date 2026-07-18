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
from mlframe.feature_selection.stochastic_bandit_selection_ensemble import stochastic_bandit_selection_ensemble


def _make_overparameterized_regression(n: int, n_signal: int, n_noise: int, seed: int):
    """Make overparameterized regression."""
    rng = np.random.default_rng(seed)
    X_signal = rng.normal(size=(n, n_signal))
    beta = rng.normal(size=n_signal) * 3.0
    y = X_signal @ beta + rng.normal(scale=0.5, size=n)
    X_noise = rng.normal(size=(n, n_noise))
    columns = [f"s{i}" for i in range(n_signal)] + [f"n{i}" for i in range(n_noise)]
    X = pd.DataFrame(np.hstack([X_signal, X_noise]), columns=columns)
    return X, y


def _plain_random_subset_search(X, y, cv, subset_size, n_epochs, seed):
    """Plain random subset search."""
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
    """Biz val bandit selection beats plain random search on average."""
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

    assert (
        mean_bandit > mean_random
    ), f"expected adaptive-weighted search to beat plain random search on average across {n_seeds} seeds, got bandit={mean_bandit:.4f} random={mean_random:.4f}"
    assert mean_bandit >= 0.96, f"expected the bandit selector to recover a strong held-out R2, got {mean_bandit:.4f}"


def _make_mixed_signal_regression(n: int, n_strong: int, n_weak: int, n_noise: int, seed: int):
    """A signal pool with a few strong features (lock in almost every seed) and several weak-but-real ones
    (lock in only on some seeds within a short epoch budget), among a large noise pool -- the exact regime
    where a single bandit run's locked-in "top_feats" pool is itself a noisy sample of the true signal set.
    """
    rng = np.random.default_rng(seed)
    n_signal = n_strong + n_weak
    X_signal = rng.normal(size=(n, n_signal))
    beta = np.concatenate([rng.normal(size=n_strong) * 4.0, rng.normal(size=n_weak) * 0.8])
    y = X_signal @ beta + rng.normal(scale=1.5, size=n)
    X_noise = rng.normal(size=(n, n_noise))
    columns = [f"strong{i}" for i in range(n_strong)] + [f"weak{i}" for i in range(n_weak)] + [f"n{i}" for i in range(n_noise)]
    X = pd.DataFrame(np.hstack([X_signal, X_noise]), columns=columns)
    true_signal = set(columns[:n_signal])
    return X, y, true_signal


def test_biz_val_bandit_selection_ensemble_recovers_more_signal_than_single_seed():
    """Biz val bandit selection ensemble recovers more signal than single seed."""
    X, y, true_signal = _make_mixed_signal_regression(n=200, n_strong=3, n_weak=5, n_noise=40, seed=7)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    seeds = list(range(8))
    common_kwargs = dict(subset_size=6, n_epochs=50, cv=cv, lock_in_threshold=2.5)

    single_seed_recalls = []
    for seed in seeds:
        result = stochastic_bandit_selection_ensemble(Ridge(alpha=0.1), X, y, scoring=r2_score, seeds=[seed], **common_kwargs)
        selected = set(result.union_top_feats)
        single_seed_recalls.append(len(selected & true_signal) / len(true_signal))

    mean_single_seed_recall = float(np.mean(single_seed_recalls))

    ensemble_result = stochastic_bandit_selection_ensemble(Ridge(alpha=0.1), X, y, scoring=r2_score, seeds=seeds, **common_kwargs)
    ensemble_selected = set(ensemble_result.union_top_feats)
    ensemble_recall = len(ensemble_selected & true_signal) / len(true_signal)

    assert ensemble_recall > mean_single_seed_recall, (
        f"expected the {len(seeds)}-seed union to recover more of the true signal set than a lone seed on "
        f"average, got ensemble={ensemble_recall:.4f} mean_single_seed={mean_single_seed_recall:.4f}"
    )
    assert ensemble_recall >= 0.55, f"expected the ensemble union to recover most of the true signal set, got {ensemble_recall:.4f}"

    # the strong (easy) signal features should have near-unanimous cross-seed agreement, the weak ones lower --
    # the stability diagnostic must actually distinguish reliable from lucky-guess selections.
    strong_feats = [f for f in true_signal if f.startswith("strong")]
    strong_stability = float(np.mean([ensemble_result.stability.get(f, 0.0) for f in strong_feats]))
    noise_feats = [f for f in ensemble_result.union_top_feats if f.startswith("n")]
    noise_stability = float(np.mean([ensemble_result.stability[f] for f in noise_feats])) if noise_feats else 0.0
    assert strong_stability > noise_stability, (
        f"expected strong-signal features to have higher cross-seed selection agreement than noise features "
        f"that got selected by chance, got strong={strong_stability:.4f} noise={noise_stability:.4f}"
    )


def test_stochastic_bandit_selection_returns_requested_subset_size():
    """Stochastic bandit selection returns requested subset size."""
    X, y = _make_overparameterized_regression(n=100, n_signal=2, n_noise=10, seed=0)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    subset = stochastic_bandit_selection(Ridge(alpha=0.1), X, y, scoring=r2_score, subset_size=5, n_epochs=20, cv=cv, random_state=0)
    assert len(subset) == 5
    assert len(set(subset)) == 5
