"""biz_value test for ``votenrank.knn_fallback_predictor.KNNFallbackPredictor``, composed with the existing
``confidence_gated_blend``.

The win (3rd_mechanisms-of-action-moa-prediction.md): a primary (linear) model has a "blind spot" region with
a genuinely nonlinear local decision boundary it can't represent, while a k-NN target-average recovers strong
local signal there simply by averaging nearby labeled neighbors -- no global model assumption needed. This
test confirms the composition ``KNNFallbackPredictor`` + ``confidence_gated_blend`` beats the primary model
alone specifically ON the blind-spot region, validating the source's own "similar known-label" fallback
pattern using mlframe's existing blend-gating plumbing plus this session's new predictor wrapper.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.votenrank.confidence_gated_blend import confidence_gated_blend
from mlframe.votenrank.knn_fallback_predictor import KNNFallbackPredictor


def _make_blind_spot_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 2))

    # Globally linearly separable EXCEPT inside a small checkerboard "blind spot" patch (a local XOR-style
    # pattern no linear model can represent), which a linear model gets essentially at chance.
    linear_part = (X[:, 0] > 0).astype(int)
    in_blind_spot = (np.abs(X[:, 0]) < 1.0) & (np.abs(X[:, 1]) < 1.0)
    checkerboard = ((np.floor(X[:, 0] * 2) + np.floor(X[:, 1] * 2)) % 2 == 0).astype(int)
    y = np.where(in_blind_spot, checkerboard, linear_part)
    return X, y, in_blind_spot


def test_biz_val_knn_fallback_beats_primary_model_in_blind_spot_region():
    X_train, y_train, _ = _make_blind_spot_dataset(n=4000, seed=0)
    X_test, y_test, in_blind_spot_test = _make_blind_spot_dataset(n=2000, seed=1)

    primary = LogisticRegression(max_iter=500).fit(X_train, y_train)
    primary_proba = primary.predict_proba(X_test)[:, 1]

    knn = KNNFallbackPredictor(k=15, metric="l2").fit(X_train, y_train.astype(np.float64))
    knn_pred, knn_confidence = knn.predict(X_test)

    blended = confidence_gated_blend(primary_proba, knn_pred, knn_confidence, confidence_threshold=np.median(knn_confidence), gated_weight=1.0, default_weight=0.0)

    auc_primary_blind_spot = roc_auc_score(y_test[in_blind_spot_test], primary_proba[in_blind_spot_test])
    auc_blended_blind_spot = roc_auc_score(y_test[in_blind_spot_test], blended[in_blind_spot_test])

    assert auc_primary_blind_spot < 0.65, f"expected the primary linear model to be near-chance in its blind spot, got AUC={auc_primary_blind_spot:.4f}"
    assert auc_blended_blind_spot > 0.75, f"expected the KNN-fallback-blended prediction to strongly recover signal in the blind spot, got AUC={auc_blended_blind_spot:.4f}"
    assert auc_blended_blind_spot > auc_primary_blind_spot + 0.15, f"expected a material improvement over the primary model alone in the blind spot, got blended={auc_blended_blind_spot:.4f} primary={auc_primary_blind_spot:.4f}"


def test_knn_fallback_predictor_confidence_high_when_dense_low_when_sparse():
    rng = np.random.default_rng(2)
    dense_cluster = rng.normal(loc=0.0, scale=0.1, size=(500, 2))
    y_train = rng.integers(0, 2, 500).astype(np.float64)
    knn = KNNFallbackPredictor(k=5).fit(dense_cluster, y_train)

    near_query = np.array([[0.0, 0.0]])
    far_query = np.array([[50.0, 50.0]])
    _, near_confidence = knn.predict(near_query)
    _, far_confidence = knn.predict(far_query)

    assert near_confidence[0] > far_confidence[0], f"expected higher confidence for a query near the training data, got near={near_confidence[0]:.4f} far={far_confidence[0]:.4f}"
