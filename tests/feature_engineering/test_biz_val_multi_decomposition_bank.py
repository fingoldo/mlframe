"""biz_value test for ``feature_engineering.multi_decomposition_bank.multi_decomposition_feature_bank``.

The win (3rd_mercedes-benz-greener-manufacturing.md): when the true signal lies on a low-rank manifold buried
inside many noisy, mutually-correlated raw features, a linear model fit on the raw high-dimensional feature
set is diluted by per-feature noise and near-collinearity (ill-conditioned design matrix). Concatenating
low-dimensional decomposition projections (which recover an approximation of the underlying manifold)
alongside the raw features gives the model a cleaner, denoised signal to draw on directly, improving
held-out performance over raw features alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.multi_decomposition_bank import multi_decomposition_feature_bank


def _make_low_rank_manifold_dataset(n: int, n_raw_features: int, seed: int):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    loadings = rng.normal(size=(3, n_raw_features))
    X_raw = latent @ loadings + rng.normal(scale=4.0, size=(n, n_raw_features))  # heavy per-feature noise
    y = latent[:, 0] * 2.0 - latent[:, 1] * 1.5 + latent[:, 2] * 1.0 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame(X_raw, columns=[f"f{i}" for i in range(n_raw_features)])
    return df, y


def test_biz_val_decomposition_bank_improves_fit_on_noisy_low_rank_manifold():
    # RandomForest (not a linear model): a decomposition projection is a LINEAR recombination of the raw
    # features, so it adds zero new information to a linear model (already in the raw features' span) -- the
    # real win is for models like trees that split axis-aligned and struggle to reconstruct a rotated/mixed
    # low-rank signal from many individually-noisy raw columns, but can directly exploit a pre-computed
    # low-rank projection column once it's simply another input feature.
    df, y = _make_low_rank_manifold_dataset(n=500, n_raw_features=100, seed=0)

    def _rf():
        return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)

    auc_raw = cross_val_score(_rf(), df.to_numpy(), y, cv=5, scoring="r2").mean()

    bank = multi_decomposition_feature_bank(df, n_components=5, methods=("svd", "pca", "ica"), random_state=0)
    df_augmented = pd.concat([df, bank], axis=1)
    auc_augmented = cross_val_score(_rf(), df_augmented.to_numpy(), y, cv=5, scoring="r2").mean()

    assert auc_augmented > auc_raw + 0.15, f"expected the decomposition-bank-augmented feature set to materially beat raw features alone on a noisy low-rank-manifold target, got augmented={auc_augmented:.4f} raw={auc_raw:.4f}"


def test_multi_decomposition_feature_bank_output_shape():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(200, 20)), columns=[f"f{i}" for i in range(20)])
    bank = multi_decomposition_feature_bank(df, n_components=4, methods=("svd", "pca", "grp", "srp"))
    assert bank.shape == (200, 16)  # 4 methods x 4 components
    assert list(bank.columns)[:4] == [f"decomp_svd_{i}" for i in range(4)]


def test_multi_decomposition_feature_bank_invalid_method_raises():
    import pytest

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        multi_decomposition_feature_bank(df, methods=("svd", "not_a_method"))
