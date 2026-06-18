"""Biz-value floor gates for supervised / local-model FE transformers that previously had no quantitative floor (they
were only exercised in the informational ``test_biz_val_real_datasets`` matrix, which always passes).

Each test pins a DELTA-vs-OFF win on a synthetic engineered so the operator's mechanism is the Bayes-relevant feature a
LINEAR model (logistic regression) cannot otherwise recover, concatenated with raw inputs via the leakage-safe Mode-A
(splitter) path. The non-linearity of the synthetic is what creates the headroom: a linear-projection operator (LDA /
NCA) is deliberately NOT pinned here because logreg already finds any linear direction, so those operators show no
linear-downstream headroom (an honest finding -- their win needs a tree downstream, deferred to a heavier bench).

* ``compute_aux_mlp_features`` - OOF MLP proba/logit recovers a non-linear (XOR-of-blobs) binary boundary.
* ``compute_class_mahalanobis_features`` - class-conditional covariance differs between classes, so the per-class
  Mahalanobis gap separates classes whose MEANS coincide (linear discriminant at chance).
* ``compute_local_classifier_features`` - locally-weighted logistic fit recovers a curved (XOR) boundary as proba.
* ``compute_rf_proximity_attention`` - tree-proximity softmax-kNN target encoding recovers the XOR partition.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

pytestmark = [pytest.mark.fast]


def _xor_blobs_binary(n: int = 1200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """XOR-of-blobs binary target in 2 informative dims + 4 noise dims. Class = sign(x0)*sign(x1)>0; means coincide at
    origin so a linear discriminator is at chance, but a non-linear model (MLP / local fit) recovers the boundary."""
    rng = np.random.default_rng(seed)
    n_inf = 2
    n_noise = 4
    Xi = rng.normal(scale=1.0, size=(n, n_inf)).astype(np.float32)
    y = ((Xi[:, 0] > 0) ^ (Xi[:, 1] > 0)).astype(np.int8)
    Xn = rng.normal(scale=1.0, size=(n, n_noise)).astype(np.float32)
    X = np.concatenate([Xi, Xn], axis=1)
    return X, y


def _diff_covariance_binary(n: int = 1200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Binary target where both classes share the same mean (origin) but have differently-oriented covariance: the
    positive class is elongated along the x0=x1 diagonal, the negative along x0=-x1. Class means are uninformative,
    so a mean-difference (LDA / logreg) is at chance, but the class-conditional Mahalanobis gap separates them."""
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    n_neg = n - n_pos
    base = rng.normal(size=(n, 2)).astype(np.float32)
    a = np.array([[3.0, 0.0], [0.0, 0.3]], dtype=np.float32)
    rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
    cov_pos = rot(np.pi / 4) @ a
    cov_neg = rot(-np.pi / 4) @ a
    X_pos = base[:n_pos] @ cov_pos.T
    X_neg = base[n_pos:] @ cov_neg.T
    Xi = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)], axis=0)
    Xn = rng.normal(scale=1.0, size=(n, 4)).astype(np.float32)
    X = np.concatenate([Xi, Xn], axis=1).astype(np.float32)
    perm = rng.permutation(n)
    return X[perm], y[perm]


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


def _logreg():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)


def _augment_auc(compute_fn, X, y, *, floor: float, name: str, **kwargs) -> None:
    """Compute Mode-A leakage-safe features on a fresh splitter, concat with raw, and assert CV-AUC delta >= floor."""
    auc_raw = _cv_auc(_logreg, X, y, n_splits=5, seed=42)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    feats = compute_fn(X_train=X, y_train=y, X_query=None, splitter=splitter, seed=0, **kwargs)
    feats_arr = np.clip(feats.to_numpy().astype(np.float32), -50.0, 50.0)
    feats_arr = np.nan_to_num(feats_arr, nan=0.0, posinf=50.0, neginf=-50.0)
    X_aug = np.concatenate([X, feats_arr], axis=1)
    auc_aug = _cv_auc(_logreg, X_aug, y, n_splits=5, seed=42)
    delta = auc_aug - auc_raw
    assert delta >= floor, f"{name} must lift linear CV AUC by >={floor}; raw={auc_raw:.4f}, aug={auc_aug:.4f}, delta={delta:.4f}"


def test_biz_val_aux_mlp_lifts_linear_auc_on_xor_blobs():
    from mlframe.feature_engineering.transformer.aux_mlp import compute_aux_mlp_features

    X, y = _xor_blobs_binary(n=1200, seed=0)
    _augment_auc(compute_aux_mlp_features, X, y, floor=0.20, name="aux_mlp", task="binary", hidden_size=16, max_iter=400)


def test_biz_val_class_mahalanobis_lifts_linear_auc_on_diff_covariance():
    from mlframe.feature_engineering.transformer.class_mahalanobis import compute_class_mahalanobis_features

    X, y = _diff_covariance_binary(n=1200, seed=0)
    _augment_auc(compute_class_mahalanobis_features, X, y, floor=0.15, name="class_mahalanobis", standardize=True)


def test_biz_val_local_classifier_lifts_linear_auc_on_xor_blobs():
    from mlframe.feature_engineering.transformer.local_classifier import compute_local_classifier_features

    X, y = _xor_blobs_binary(n=1200, seed=0)
    _augment_auc(compute_local_classifier_features, X, y, floor=0.15, name="local_classifier", task="binary", k=32, standardize=True)


def test_biz_val_rf_proximity_lifts_linear_auc_on_xor_blobs():
    from mlframe.feature_engineering.transformer.rf_proximity import compute_rf_proximity_attention

    X, y = _xor_blobs_binary(n=1200, seed=0)
    _augment_auc(compute_rf_proximity_attention, X, y, floor=0.15, name="rf_proximity", task="binary", n_aux_trees=200, k=32)
