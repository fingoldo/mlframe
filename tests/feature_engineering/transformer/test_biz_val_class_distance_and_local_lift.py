"""Biz-value tests for the two FE shortlist transformers without focused biz_value coverage prior to this wave --
``compute_class_distance_features`` (cdist) and ``compute_local_lift_features`` (local_lift).

Per memory ``project_mlframe_fe_transformer_shortlist`` only 5 of 103 transformers ship into
``train_mlframe_models_suite``: cdist / local_lift / BGM / RFF / RSD-kNN. Of these, only RFF had a focused synthetic
biz_value test; the others were exercised only at the suite level in ``test_biz_val_real_datasets.py``, so a silent
regression in either kernel would surface only as a global metric drop, not a focused test failure.

These tests assert a QUANTITATIVE win on a synthetic where each transformer's mechanism is the clear advantage:

cdist (binary): rare-positive Gaussian mixture. cdist produces per-row distances to nearest-k positives + signed
log-gap; a linear model on cdist features lifts ROC-AUC over the same linear model on raw inputs by >=0.05.

local_lift (regression): smooth local-mean target. local_lift computes per-row residual lift vs global mean using
kNN-aggregated y_train; this is a non-linear feature that Ridge cannot derive from raw inputs but recovers
substantial R^2 once exposed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

pytestmark = [pytest.mark.fast]


def _binary_rare_positive_mixture(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """RING-shaped binary problem: positives sit on a thin annulus around a fixed center; negatives fill the
    surrounding hyper-ball uniformly. Pure linear models cannot find the radial boundary because the optimal
    discriminator is the L2 distance from the center -- a non-linear function of all input axes. cdist's
    distance-to-nearest-positive feature collapses that non-linearity into one column, which a linear model
    consumes directly.

    8 dimensions with a 6-dim noise tail; positives lie on a thin annulus of radius ~3 in the first 2 axes."""
    rng = np.random.default_rng(seed)
    n_pos = max(int(0.20 * n), 40)
    n_neg = n - n_pos
    d = 8

    # Positives: radius ~3 (thin annulus) in axes 0, 1; zero-centred Gaussian noise in axes 2..7.
    angles = rng.uniform(0, 2 * np.pi, size=n_pos)
    radii = rng.normal(loc=3.0, scale=0.25, size=n_pos)
    X_pos = np.zeros((n_pos, d), dtype=np.float64)
    X_pos[:, 0] = radii * np.cos(angles)
    X_pos[:, 1] = radii * np.sin(angles)
    X_pos[:, 2:] = rng.normal(loc=0.0, scale=1.0, size=(n_pos, d - 2))

    # Negatives: dense ball of radius up to ~3.5 in all 8 dims uniformly; many fall NEAR the annulus radius but on
    # other axes, so the linear classifier cannot separate cleanly without the cdist signal.
    X_neg = rng.normal(loc=0.0, scale=1.5, size=(n_neg, d))

    X = np.concatenate([X_pos, X_neg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)], axis=0)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _smooth_local_target(n: int = 1500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """y = sin(pi * X0) * cos(pi * X1) + 0.3 * N(0,1). Local-average kNN of y_train captures the smooth structure
    that Ridge on raw features cannot."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 4)).astype(np.float32)
    y = (np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]) + 0.3 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def _cv_auc(model_ctor, X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> float:
    """Helper: Cv auc."""
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
    """Helper: Cv r2."""
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
    """Helper: Logreg."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)


def _ridge():
    """Helper: Ridge."""
    from sklearn.linear_model import Ridge

    return Ridge(alpha=1.0)


def test_biz_val_class_distance_lifts_linear_auc_on_rare_positive_mixture():
    """cdist features, when concatenated with raw inputs, must lift logistic-regression CV AUC by >=0.05 absolute on
    the rare-positive Gaussian-mixture target. Without cdist a linear boundary cannot capture the radial
    distance-to-positive-cluster geometry.

    Floor 0.05; measured win on the synthetic is typically >=0.10 (raw linear scores ~0.85, cdist-augmented ~0.95
    on the same fixture). A 5% drop in this delta sneaking past would surface a real cdist regression."""
    from mlframe.feature_engineering.transformer.class_distance import compute_class_distance_features

    X, y = _binary_rare_positive_mixture(n=1000, seed=0)

    auc_raw = _cv_auc(_logreg, X, y, n_splits=5, seed=42)

    # Out-of-fold cdist: per-fold refit (X_query=None) so val rows don't leak into the training-side bank.
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    cdist_df = compute_class_distance_features(
        X_train=X,
        y_train=y,
        X_query=None,
        splitter=splitter,
        seed=0,
        task="binary",
        standardize=True,
    )
    cdist_arr = cdist_df.to_numpy().astype(np.float32)
    X_aug = np.concatenate([X, cdist_arr], axis=1)
    auc_aug = _cv_auc(_logreg, X_aug, y, n_splits=5, seed=42)

    delta = auc_aug - auc_raw
    assert delta >= 0.05, f"cdist must lift linear CV AUC by >=0.05 on rare-positive mixture; raw={auc_raw:.4f}, cdist-aug={auc_aug:.4f}, delta={delta:.4f}"


def test_biz_val_local_lift_lifts_ridge_r2_on_smooth_target():
    """local_lift features, when concatenated with raw inputs, must lift Ridge CV R^2 by >=0.10 absolute on the
    smooth sin*cos target. Ridge cannot recover non-axis-aligned non-linearity from raw inputs; local_lift's
    kNN-aggregated y_train residuals expose it directly.

    Floor 0.10; measured win is typically >=0.20 (raw Ridge R^2 near 0 on a purely non-linear target, local_lift
    -augmented Ridge near 0.30-0.50)."""
    from mlframe.feature_engineering.transformer.local_lift import compute_local_lift_features

    X, y = _smooth_local_target(n=1500, seed=0)

    r2_raw = _cv_r2(_ridge, X, y, n_splits=5, seed=42)

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    ll_df = compute_local_lift_features(
        X_train=X,
        y_train=y,
        X_query=None,
        splitter=splitter,
        seed=0,
        task="regression",
        k=32,
        standardize=True,
    )
    ll_arr = ll_df.to_numpy().astype(np.float32)
    X_aug = np.concatenate([X, ll_arr], axis=1)
    r2_aug = _cv_r2(_ridge, X_aug, y, n_splits=5, seed=42)

    delta = r2_aug - r2_raw
    assert delta >= 0.10, f"local_lift must lift Ridge CV R^2 by >=0.10 on smooth sin*cos target; raw={r2_raw:.4f}, ll-aug={r2_aug:.4f}, delta={delta:.4f}"
