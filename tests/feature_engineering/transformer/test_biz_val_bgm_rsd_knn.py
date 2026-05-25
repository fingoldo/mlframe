"""Biz-value tests for the remaining FE shortlist transformers without focused biz_value coverage prior to this wave:
the BGM family (``bgmm_density_ratio``, ``bgmm_dual_class``, ``bgmm_virtual``, ``bgmm_multiscale``,
``bgmm_quantile_bands``, ``bgm_clustered_smote``) and the RSD-kNN family (``y_quintile_baseline_knn`` +
``residual_stratified_distance``).

Per memory ``project_mlframe_fe_transformer_shortlist`` only 5 of 103 transformers ship into
``train_mlframe_models_suite``: cdist / local_lift / BGM / RFF / RSD-kNN. The cdist + local_lift focused biz_values
landed in W3C; the RFF set landed earlier. The BGM and RSD-kNN focused biz_values were deferred to Wave 10c because
the BGM family has 6 variants and each fit is ~1-2s, pushing the file's wall-time budget. We pin one
"the variant should clearly win" synthetic per family rather than per variant; that keeps the wall under 20s while
still catching silent kernel regressions in the shared scoring / standardisation / per-fold-fit machinery.

Quantitative wins:

* ``compute_bgmm_density_ratio_features`` on a multi-modal Gaussian-mixture binary target: logistic regression on
  raw inputs cannot recover the discriminator because the two classes have overlapping spherical components;
  log-density-ratio is the Bayes-optimal feature and lifts CV AUC by >=0.05 absolute when concatenated with raw.
* ``compute_y_quintile_baseline_knn_features`` on a regression target with quintile-conditional structure: a Ridge
  on raw inputs misses the per-quintile baseline-prediction-at-neighbours signal that boosts CV R^2 by >=0.05
  absolute when concatenated.

Both tests use ``Mode A`` (X_query=None + splitter) to exercise the per-fold leakage-safe path that ships in the
production shortlist.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

pytestmark = [pytest.mark.fast]


def _two_gmm_binary_mixture(n: int = 1200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Binary target where each class is a mixture of 3 spherical Gaussian components in 4D, with overlapping
    centres so a linear discriminator cannot separate cleanly. Class densities have non-trivial shape, so the
    log-density-ratio feature carries strictly more information than any per-axis projection."""
    rng = np.random.default_rng(seed)
    d = 4
    centres_pos = np.array([[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    centres_neg = np.array([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, -2.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    n_pos = n // 2
    n_neg = n - n_pos

    def _draw(centres: np.ndarray, n_draw: int) -> np.ndarray:
        comps = rng.integers(0, centres.shape[0], size=n_draw)
        return centres[comps] + rng.normal(scale=0.8, size=(n_draw, d))

    X_pos = _draw(centres_pos, n_pos)
    X_neg = _draw(centres_neg, n_neg)
    X = np.concatenate([X_pos, X_neg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)], axis=0)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _knn_recoverable_regression(n: int = 1500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Regression target where the global mean is uninformative (Ridge R^2 ~ 0) but the LOCAL kNN mean of the LGB-
    baseline prediction sharply tracks y. Achieved with a target that is a smooth non-linear function on a 2D
    manifold embedded in 6D noise.

    Ridge cannot find the manifold because the relevant directions are non-linear; the lgb baseline inside the
    transformer learns the function on the manifold; per-stratum kNN averages of that baseline (the RSD-kNN
    features) then expose it as a near-linear signal that Ridge consumes directly."""
    rng = np.random.default_rng(seed)
    d = 6
    X = rng.uniform(-2.0, 2.0, size=(n, d)).astype(np.float32)
    # Target: smooth bivariate non-linear interaction in the first two axes; the rest are noise dimensions.
    y = (
        np.sin(X[:, 0] * 1.2) * np.cos(X[:, 1] * 1.2)
        + 0.5 * np.tanh(X[:, 0] * X[:, 1])
        + 0.2 * rng.standard_normal(n)
    ).astype(np.float32)
    return X, y


def _cv_auc(model_ctor, X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> float:
    from sklearn.metrics import roc_auc_score
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in splitter.split(X):
        m = model_ctor()
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[va])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X[va])
        aucs.append(roc_auc_score(y[va], proba))
    return float(np.mean(aucs))


def _cv_r2(model_ctor, X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> float:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s = []
    for tr, va in splitter.split(X):
        m = model_ctor()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[va])
        ss_res = float(np.sum((y[va] - pred) ** 2))
        ss_tot = float(np.sum((y[va] - y[va].mean()) ** 2))
        r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return float(np.mean(r2s))


def _logreg():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)


def _ridge():
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0)


def test_biz_val_bgmm_density_ratio_lifts_linear_auc_on_multimodal_mixture():
    """BGM log-density-ratio features, concatenated with raw inputs, must lift logistic-regression CV AUC by >=0.05
    absolute on a multi-modal Gaussian-mixture binary target.

    Floor 0.05; measured win is typically >=0.08 (raw linear ~0.55-0.60 on the chosen overlapping mixture, BGM-
    augmented ~0.65+). A regression that returns constant log-density (BGM fit-failure swallow path) would
    collapse the delta to ~0.0 and trip the floor."""
    from mlframe.feature_engineering.transformer.bgmm_density_ratio import compute_bgmm_density_ratio_features

    X, y = _two_gmm_binary_mixture(n=1000, seed=0)
    auc_raw = _cv_auc(_logreg, X, y, n_splits=5, seed=42)

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    feats_df = compute_bgmm_density_ratio_features(
        X_train=X, y_train=y, X_query=None, splitter=splitter, seed=0, task="binary",
        component_counts=(3, 5), standardize=True,
    )
    feats_arr = feats_df.to_numpy().astype(np.float32)
    # Replace any +/-inf produced by extreme log-density estimates with a large finite floor so logistic regression
    # does not bail out on inf gradients on the synthetic edge case.
    feats_arr = np.clip(feats_arr, -50.0, 50.0)
    X_aug = np.concatenate([X, feats_arr], axis=1)
    auc_aug = _cv_auc(_logreg, X_aug, y, n_splits=5, seed=42)

    delta = auc_aug - auc_raw
    assert delta >= 0.05, (
        f"BGM log-density-ratio must lift linear CV AUC by >=0.05 on a multi-modal Gaussian mixture; "
        f"raw={auc_raw:.4f}, bgm-aug={auc_aug:.4f}, delta={delta:.4f}"
    )


def test_biz_val_y_quintile_baseline_knn_lifts_ridge_r2_on_knn_recoverable_target():
    """RSD-kNN features (5 strata x mean+std of baseline-pred-at-kNN), concatenated with raw inputs, must lift
    Ridge CV R^2 by >=0.05 absolute on a kNN-recoverable non-linear regression target.

    Floor 0.05; measured win is typically >=0.10 on the chosen fixture. The lgb baseline inside the transformer
    captures the smooth bivariate non-linear structure; without it, Ridge on raw inputs cannot decompose
    sin(x0)cos(x1) + tanh(x0*x1)."""
    pytest.importorskip("lightgbm")
    from mlframe.feature_engineering.transformer.y_quintile_baseline_knn import (
        compute_y_quintile_baseline_knn_features,
    )

    X, y = _knn_recoverable_regression(n=1200, seed=0)
    r2_raw = _cv_r2(_ridge, X, y, n_splits=5, seed=42)

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    feats_df = compute_y_quintile_baseline_knn_features(
        X_train=X, y_train=y, X_query=None, splitter=splitter, seed=0, task="regression", standardize=True,
    )
    feats_arr = feats_df.to_numpy().astype(np.float32)
    X_aug = np.concatenate([X, feats_arr], axis=1)
    r2_aug = _cv_r2(_ridge, X_aug, y, n_splits=5, seed=42)

    delta = r2_aug - r2_raw
    assert delta >= 0.05, (
        f"RSD-kNN (y_quintile_baseline_knn) must lift Ridge CV R^2 by >=0.05 on a quintile-conditional regression "
        f"target; raw={r2_raw:.4f}, rsdknn-aug={r2_aug:.4f}, delta={delta:.4f}"
    )
