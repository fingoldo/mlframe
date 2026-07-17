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

    blended = confidence_gated_blend(
        primary_proba, knn_pred, knn_confidence, confidence_threshold=np.median(knn_confidence), gated_weight=1.0, default_weight=0.0
    )

    auc_primary_blind_spot = roc_auc_score(y_test[in_blind_spot_test], primary_proba[in_blind_spot_test])
    auc_blended_blind_spot = roc_auc_score(y_test[in_blind_spot_test], blended[in_blind_spot_test])

    assert auc_primary_blind_spot < 0.65, f"expected the primary linear model to be near-chance in its blind spot, got AUC={auc_primary_blind_spot:.4f}"
    assert auc_blended_blind_spot > 0.75, (
        f"expected the KNN-fallback-blended prediction to strongly recover signal in the blind spot, got AUC={auc_blended_blind_spot:.4f}"
    )
    assert auc_blended_blind_spot > auc_primary_blind_spot + 0.15, (
        f"expected a material improvement over the primary model alone in the blind spot, got blended={auc_blended_blind_spot:.4f} primary={auc_primary_blind_spot:.4f}"
    )


def test_knn_fallback_predictor_confidence_high_when_dense_low_when_sparse():
    rng = np.random.default_rng(2)
    dense_cluster = rng.normal(loc=0.0, scale=0.1, size=(500, 2))
    y_train = rng.integers(0, 2, 500).astype(np.float64)
    knn = KNNFallbackPredictor(k=5).fit(dense_cluster, y_train)

    near_query = np.array([[0.0, 0.0]])
    far_query = np.array([[50.0, 50.0]])
    _, near_confidence = knn.predict(near_query)
    _, far_confidence = knn.predict(far_query)

    assert near_confidence[0] > far_confidence[0], (
        f"expected higher confidence for a query near the training data, got near={near_confidence[0]:.4f} far={far_confidence[0]:.4f}"
    )


def _make_coldstart_dataset(n: int, seed: int, dense_frac: float, label_noise: float):
    """Globally linearly separable EXCEPT inside a densely-sampled central "cold-start" patch (many similar
    labeled neighbors packed close together -- e.g. a popular product/user segment with lots of history) with
    a local XOR-style checkerboard pattern no linear model can represent. Uniform label noise everywhere
    degrades a pure kNN average's GLOBAL accuracy (each neighborhood mixes in some flipped labels) while a
    linear main model smooths noise out via its global fit -- except inside the patch, where the main model's
    structural blind spot dominates and the kNN's dense, locally-consistent neighborhood still averages out
    the noise well. So each predictor is best in a different region, and neither is uniformly best."""
    rng = np.random.default_rng(seed)
    n_dense = int(n * dense_frac)
    X_dense = rng.uniform(-1, 1, size=(n_dense, 2))
    X_rest = rng.uniform(-3, 3, size=(n - n_dense, 2))
    X = np.vstack([X_dense, X_rest])
    linear_part = (X[:, 0] > 0).astype(int)
    in_coldstart = (np.abs(X[:, 0]) < 1.0) & (np.abs(X[:, 1]) < 1.0)
    checkerboard = ((np.floor(X[:, 0] * 2) + np.floor(X[:, 1] * 2)) % 2 == 0).astype(int)
    y = np.where(in_coldstart, checkerboard, linear_part)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y)
    order = rng.permutation(n)
    return X[order], y[order], in_coldstart[order]


def test_biz_val_knn_fallback_predictor_predict_blend_beats_either_predictor_alone():
    X_train, y_train, _ = _make_coldstart_dataset(n=4000, seed=10, dense_frac=0.5, label_noise=0.30)
    X_test, y_test, _ = _make_coldstart_dataset(n=2000, seed=11, dense_frac=0.25, label_noise=0.0)

    main = LogisticRegression(max_iter=500).fit(X_train, y_train)
    main_pred = main.predict_proba(X_test)[:, 1]

    knn = KNNFallbackPredictor(k=15, metric="l2").fit(X_train, y_train.astype(np.float64))
    blended = knn.predict_blend(X_test, main_pred, gated_weight=1.0, default_weight=0.0)
    knn_pred_only, _ = knn.predict(X_test)

    auc_main_only = roc_auc_score(y_test, main_pred)
    auc_knn_only = roc_auc_score(y_test, knn_pred_only)
    auc_blended = roc_auc_score(y_test, blended)

    assert auc_blended > auc_main_only + 0.03, f"expected predict_blend to beat the main model alone, got blended={auc_blended:.4f} main={auc_main_only:.4f}"
    assert auc_blended > auc_knn_only + 0.005, (
        f"expected predict_blend to beat the kNN fallback alone, got blended={auc_blended:.4f} knn_only={auc_knn_only:.4f}"
    )
    assert auc_blended > 0.95, f"expected predict_blend to recover strong overall AUC, got {auc_blended:.4f}"


def test_knn_fallback_predictor_predict_blend_does_not_change_predict_behavior():
    """predict_blend is additive/opt-in -- plain predict() must stay bit-identical when it's never called."""
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(300, 4))
    y_train = rng.integers(0, 2, 300).astype(np.float64)
    X_query = rng.normal(size=(50, 4))

    knn_a = KNNFallbackPredictor(k=5).fit(X_train, y_train)
    pred_a, confidence_a = knn_a.predict(X_query)

    knn_b = KNNFallbackPredictor(k=5).fit(X_train, y_train)
    main_pred = rng.uniform(size=50)
    knn_b.predict_blend(X_query, main_pred)  # exercise the new path on a second instance
    pred_b, confidence_b = knn_b.predict(X_query)

    np.testing.assert_array_equal(pred_a, pred_b)
    np.testing.assert_array_equal(confidence_a, confidence_b)
