"""biz_value test for ``feature_engineering.compute_auxiliary_feature_prediction_features``.

The win: an important observed feature (``ext_source``) is a NOISY proxy for the true latent factor driving
the target, while several other correlated columns (each also noisy) collectively let a model reconstruct a
cleaner estimate of that latent factor than any single raw column carries. At small training size with many
irrelevant noise columns mixed in, a tree model struggles to implicitly re-derive that reconstruction from
the raw columns alone; adding the explicit OOF-predicted-value feature (already a denoised reconstruction)
should measurably help -- mirroring the Home Credit 3rd place's ext_source submodel technique.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from mlframe.feature_engineering import compute_auxiliary_feature_prediction_features


def _make_noisy_proxy_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    ext_source = z + rng.normal(scale=1.8, size=n)
    others = {f"f{i}": z + rng.normal(scale=1.2, size=n) for i in range(10)}
    noise_cols = {f"noise{i}": rng.normal(size=n) for i in range(10)}
    y = z + rng.normal(scale=0.2, size=n)
    X = pd.DataFrame({"ext_source": ext_source, **others, **noise_cols})
    return X, y


def test_biz_val_auxiliary_feature_prediction_beats_raw_features_alone_mse():
    X, y = _make_noisy_proxy_dataset(n=150, seed=2)
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(y))
    train_idx, test_idx = perm[:100], perm[100:]
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]

    baseline = GradientBoostingRegressor(random_state=0, n_estimators=150, max_depth=4).fit(X_train, y_train)
    mse_baseline = mean_squared_error(y_test, baseline.predict(X_test))

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    aux_train = compute_auxiliary_feature_prediction_features(X_train, ["ext_source"], splitter=kf, seed=0).to_pandas()
    aux_test = compute_auxiliary_feature_prediction_features(X_train, ["ext_source"], X_query=X_test, seed=0).to_pandas()

    X_train_aug = pd.concat([X_train, aux_train], axis=1)
    X_test_aug = pd.concat([X_test, aux_test], axis=1)
    augmented = GradientBoostingRegressor(random_state=0, n_estimators=150, max_depth=4).fit(X_train_aug, y_train)
    mse_augmented = mean_squared_error(y_test, augmented.predict(X_test_aug))

    improvement = 1.0 - mse_augmented / mse_baseline
    assert improvement > 0.1, f"expected >10% MSE reduction from the auxiliary feature-prediction columns, got {improvement:.4f} (baseline={mse_baseline:.4f}, augmented={mse_augmented:.4f})"


def test_auxiliary_feature_prediction_output_columns():
    X, y = _make_noisy_proxy_dataset(n=80, seed=4)
    kf = KFold(n_splits=4, shuffle=True, random_state=0)
    result = compute_auxiliary_feature_prediction_features(X, ["ext_source", "f0"], splitter=kf, seed=0)
    assert set(result.columns) == {"auxfeat_ext_source_pred", "auxfeat_ext_source_resid", "auxfeat_f0_pred", "auxfeat_f0_resid"}
    assert result.shape[0] == 80


def test_auxiliary_feature_prediction_rejects_unknown_target_feature():
    X, y = _make_noisy_proxy_dataset(n=50, seed=5)
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    try:
        compute_auxiliary_feature_prediction_features(X, ["not_a_real_column"], splitter=kf, seed=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
