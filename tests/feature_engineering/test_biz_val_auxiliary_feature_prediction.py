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
    assert improvement > 0.1, (
        f"expected >10% MSE reduction from the auxiliary feature-prediction columns, got {improvement:.4f} (baseline={mse_baseline:.4f}, augmented={mse_augmented:.4f})"
    )


def test_auxiliary_feature_prediction_output_columns():
    X, _y = _make_noisy_proxy_dataset(n=80, seed=4)
    kf = KFold(n_splits=4, shuffle=True, random_state=0)
    result = compute_auxiliary_feature_prediction_features(X, ["ext_source", "f0"], splitter=kf, seed=0)
    assert set(result.columns) == {"auxfeat_ext_source_pred", "auxfeat_ext_source_resid", "auxfeat_f0_pred", "auxfeat_f0_resid"}
    assert result.shape[0] == 80


def test_biz_val_auxiliary_feature_prediction_uncertainty_distinguishes_reliable_rows():
    """The win: ``n_uncertainty_repeats > 1`` emits an ``_uncertainty`` column (across-bootstrap-repeat
    prediction std) that should be HIGHER on rows where the auxiliary-feature reconstruction is actually less
    reliable. Half the rows here are built from a stable, low-noise proxy relationship (predictors tightly
    constrain the target -> reconstruction should be easy and consistent across resamples) and half from an
    unstable, high-noise one (predictors barely constrain the target -> different bootstrap resamples should
    disagree). A per-row comparison against the *observed* residual is confounded by ``ext_source``'s own
    irreducible label noise (constant across both halves), so this test instead checks the group-level split
    the uncertainty column is meant to expose directly.
    """
    rng = np.random.default_rng(7)
    n_half = 150
    z = rng.normal(size=2 * n_half)
    ext_source = z + rng.normal(scale=1.8, size=2 * n_half)
    stable_noise_scale = np.concatenate([np.full(n_half, 0.3), np.full(n_half, 4.0)])
    others = {f"f{i}": z + stable_noise_scale * rng.normal(size=2 * n_half) for i in range(8)}
    X = pd.DataFrame({"ext_source": ext_source, **others})
    is_unstable = np.concatenate([np.zeros(n_half, dtype=bool), np.ones(n_half, dtype=bool)])

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    result = compute_auxiliary_feature_prediction_features(X, ["ext_source"], splitter=kf, seed=0, n_uncertainty_repeats=8).to_pandas()

    assert "auxfeat_ext_source_uncertainty" in result.columns
    uncertainty = result["auxfeat_ext_source_uncertainty"].to_numpy()
    mean_unc_stable = uncertainty[~is_unstable].mean()
    mean_unc_unstable = uncertainty[is_unstable].mean()
    ratio = mean_unc_unstable / mean_unc_stable
    assert ratio > 1.3, (
        f"expected uncertainty on the unstable-proxy rows to be >1.3x the stable-proxy rows' uncertainty, "
        f"got {ratio:.4f} (stable={mean_unc_stable:.4f}, unstable={mean_unc_unstable:.4f})"
    )


def test_auxiliary_feature_prediction_default_unchanged_when_uncertainty_unused():
    """n_uncertainty_repeats defaults to 1 -- output must be bit-identical to the pre-extension code path,
    with no ``_uncertainty`` column emitted."""
    X, _y = _make_noisy_proxy_dataset(n=80, seed=4)
    kf = KFold(n_splits=4, shuffle=True, random_state=0)
    default_result = compute_auxiliary_feature_prediction_features(X, ["ext_source"], splitter=kf, seed=0)
    explicit_result = compute_auxiliary_feature_prediction_features(X, ["ext_source"], splitter=kf, seed=0, n_uncertainty_repeats=1)
    assert set(default_result.columns) == {"auxfeat_ext_source_pred", "auxfeat_ext_source_resid"}
    assert default_result.equals(explicit_result)


def test_auxiliary_feature_prediction_rejects_unknown_target_feature():
    X, _y = _make_noisy_proxy_dataset(n=50, seed=5)
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    try:
        compute_auxiliary_feature_prediction_features(X, ["not_a_real_column"], splitter=kf, seed=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
