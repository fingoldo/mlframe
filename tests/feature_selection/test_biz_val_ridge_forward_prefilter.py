"""biz_value test for ``feature_selection.ridge_forward_prefilter.ridge_coefficient_prefilter``.

The win (Bojan's 1st_home-credit-default-risk.md pattern): with many raw candidate features, most of them
noise, a cheap Ridge-coefficient ranking + a small CV sweep over log-spaced pool sizes should prune down to a
SMALL fraction of the original feature count while keeping CV score close to (not meaningfully worse than)
the full-feature-set score -- "almost no CV loss" with a materially smaller downstream MRMR/RFECV candidate
pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from mlframe.feature_selection.ridge_forward_prefilter import ridge_coefficient_prefilter


def _make_noisy_dataset(n: int, d_informative: int, d_noise: int, seed: int):
    rng = np.random.default_rng(seed)
    X_info = rng.normal(size=(n, d_informative))
    X_noise = rng.normal(size=(n, d_noise))
    w = rng.normal(size=d_informative)
    y = X_info @ w + rng.normal(scale=0.5, size=n)
    cols = [f"info{i}" for i in range(d_informative)] + [f"noise{i}" for i in range(d_noise)]
    X = pd.DataFrame(np.concatenate([X_info, X_noise], axis=1), columns=cols)
    return X, y


def test_biz_val_ridge_prefilter_prunes_features_with_minimal_cv_loss():
    X, y = _make_noisy_dataset(n=400, d_informative=8, d_noise=792, seed=0)  # 800 total features

    full_score = float(np.mean(cross_val_score(Ridge(alpha=1.0), X.to_numpy(), y, cv=3, scoring="r2")))

    selected = ridge_coefficient_prefilter(X.to_numpy(), y, list(X.columns), cv=3, tol=0.02, alpha=1.0)
    reduction = 1.0 - len(selected) / X.shape[1]

    pruned_score = float(np.mean(cross_val_score(Ridge(alpha=1.0), X[selected].to_numpy(), y, cv=3, scoring="r2")))

    assert reduction > 0.8, f"expected >80% feature-count reduction (800 raw features, mostly noise), got {reduction:.4f} ({len(selected)} kept)"
    assert pruned_score >= full_score - 0.03, f"expected pruned-pool CV score close to full-feature-set score, got pruned={pruned_score:.4f} vs full={full_score:.4f}"

    n_informative_kept = sum(1 for f in selected if f.startswith("info"))
    assert n_informative_kept >= 7, f"expected nearly all 8 informative features to survive the prefilter, kept {n_informative_kept}"


def test_ridge_prefilter_smallest_pool_within_tolerance_is_selected():
    X, y = _make_noisy_dataset(n=300, d_informative=4, d_noise=124, seed=1)  # 128 total features
    selected_tight = ridge_coefficient_prefilter(X.to_numpy(), y, list(X.columns), cv=3, tol=0.005, alpha=1.0)
    selected_loose = ridge_coefficient_prefilter(X.to_numpy(), y, list(X.columns), cv=3, tol=0.2, alpha=1.0)
    assert len(selected_loose) <= len(selected_tight), f"a looser tolerance should select an equal-or-smaller pool, got loose={len(selected_loose)} tight={len(selected_tight)}"


def test_ridge_prefilter_classification_returns_valid_feature_names():
    rng = np.random.default_rng(2)
    n, d = 300, 60
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    names = [f"f{i}" for i in range(d)]
    selected = ridge_coefficient_prefilter(X, y, names, cv=3, tol=0.05, is_classifier=True, alpha=1.0)
    assert 0 < len(selected) <= d
    assert set(selected).issubset(set(names))
