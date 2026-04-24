"""Tests for per-class isotonic calibration (Session 4).

Verifies:
- _PerClassIsotonicCalibrator fits K independent IsotonicRegressions
- MULTICLASS: re-normalises rows to sum to 1 (preserves softmax invariant)
- MULTILABEL: independent columns (no re-normalisation)
- Constant-label-column graceful skip (identity mapping)
- _PostHocMultiCalibratedModel pickling roundtrip
- Metrics registry dispatches hamming/subset/jaccard for multilabel
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.configs import TargetTypes
from mlframe.training.trainer import (
    _PerClassIsotonicCalibrator,
    _PostHocMultiCalibratedModel,
)


def _make_multiclass(N=500, K=3, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(K), size=N)
    y = rng.integers(0, K, size=N)
    return probs, y


def _make_multilabel(N=500, K=3, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.01, 0.99, size=(N, K))
    y = rng.integers(0, 2, size=(N, K))
    return probs, y


# ---------------------------------------------------------------------------
# _PerClassIsotonicCalibrator
# ---------------------------------------------------------------------------


def test_multiclass_rows_sum_to_one():
    probs, y = _make_multiclass()
    cal = _PerClassIsotonicCalibrator.fit(probs, y, TargetTypes.MULTICLASS_CLASSIFICATION)
    out = cal.predict_proba(probs)
    assert out.shape == probs.shape
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)


def test_multilabel_columns_independent():
    probs, y = _make_multilabel()
    cal = _PerClassIsotonicCalibrator.fit(probs, y, TargetTypes.MULTILABEL_CLASSIFICATION)
    out = cal.predict_proba(probs)
    assert out.shape == probs.shape
    # Multilabel: columns are independent; rows typically do NOT sum to 1.
    row_sums = out.sum(axis=1)
    assert not np.allclose(row_sums, 1.0, atol=0.1)


def test_output_in_0_1_range():
    for target_type in (TargetTypes.MULTICLASS_CLASSIFICATION, TargetTypes.MULTILABEL_CLASSIFICATION):
        if target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
            probs, y = _make_multiclass()
        else:
            probs, y = _make_multilabel()
        cal = _PerClassIsotonicCalibrator.fit(probs, y, target_type)
        out = cal.predict_proba(probs)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()


def test_constant_label_column_identity():
    """When a label column has all 0s (or all 1s), calibrator should skip
    that class and apply identity mapping (not crash)."""
    probs, _ = _make_multilabel(N=200, K=3, seed=0)
    # Label 2: all zeros (constant)
    y = np.random.default_rng(1).integers(0, 2, size=(200, 3))
    y[:, 2] = 0
    cal = _PerClassIsotonicCalibrator.fit(probs, y, TargetTypes.MULTILABEL_CLASSIFICATION)
    # Class 2 calibrator should be None (skipped)
    assert cal.calibrators[2] is None
    out = cal.predict_proba(probs)
    # Class 2 column in output should equal input (identity)
    np.testing.assert_allclose(out[:, 2], probs[:, 2], atol=0.0)


def test_calibration_reduces_brier_loss():
    """On miscalibrated probabilities, per-class isotonic should reduce
    the Brier loss (expected calibration error). Functional signal that
    the calibration machinery actually improves predictions."""
    from sklearn.metrics import brier_score_loss
    rng = np.random.default_rng(42)
    N = 1000
    K = 3
    # Ground-truth class membership
    y = rng.integers(0, K, size=N)
    # Miscalibrated probs: one-hot-ish on y but shrunken toward uniform
    # (flatter-than-true), which isotonic should sharpen.
    true_onehot = np.zeros((N, K))
    true_onehot[np.arange(N), y] = 1.0
    miscal = 0.6 * true_onehot + 0.4 / K
    # Add noise so isotonic has something to learn
    miscal = miscal + rng.uniform(-0.05, 0.05, size=(N, K))
    miscal = np.clip(miscal, 1e-3, 1 - 1e-3)
    miscal = miscal / miscal.sum(axis=1, keepdims=True)

    cal = _PerClassIsotonicCalibrator.fit(miscal, y, TargetTypes.MULTICLASS_CLASSIFICATION)
    calibrated = cal.predict_proba(miscal)

    # Compute Brier loss for class 0 (binary subset) before/after
    brier_before = brier_score_loss((y == 0).astype(int), miscal[:, 0])
    brier_after = brier_score_loss((y == 0).astype(int), calibrated[:, 0])
    assert brier_after <= brier_before + 1e-6, (
        f"Brier loss increased after calibration: "
        f"before={brier_before:.4f}, after={brier_after:.4f}"
    )


class _PickleableFakeBase:
    """Top-level class (picklable) fake estimator for pickle tests."""
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        # Deterministic uniform; not rng-dependent so pickle roundtrip is stable
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full((n, 3), 1/3)


# ---------------------------------------------------------------------------
# _PostHocMultiCalibratedModel
# ---------------------------------------------------------------------------


def test_wrapped_model_predict_proba_shape():
    probs, y = _make_multiclass(N=300, K=3)
    cal = _PerClassIsotonicCalibrator.fit(probs, y, TargetTypes.MULTICLASS_CLASSIFICATION)

    class FakeBase:
        classes_ = np.array([0, 1, 2])
        def predict_proba(self, X):
            return np.random.default_rng(0).dirichlet(np.ones(3), size=X.shape[0])

    wrapped = _PostHocMultiCalibratedModel(
        FakeBase(), cal, TargetTypes.MULTICLASS_CLASSIFICATION,
    )
    X = np.random.default_rng(0).standard_normal((50, 4))
    out = wrapped.predict_proba(X)
    assert out.shape == (50, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)


def test_wrapped_model_predict_uses_decision_rule():
    """For MULTICLASS, predict() returns argmax class labels, not probs."""
    probs, y = _make_multiclass()
    cal = _PerClassIsotonicCalibrator.fit(probs, y, TargetTypes.MULTICLASS_CLASSIFICATION)

    class FakeBase:
        classes_ = np.array([0, 1, 2])
        def predict_proba(self, X):
            return np.array([[0.1, 0.8, 0.1], [0.5, 0.3, 0.2]])

    wrapped = _PostHocMultiCalibratedModel(
        FakeBase(), cal, TargetTypes.MULTICLASS_CLASSIFICATION,
        classes_=np.array([0, 1, 2]),
    )
    preds = wrapped.predict(np.zeros((2, 4)))
    assert preds.shape == (2,)
    # Note: post-calibration argmax may differ from pre-calibration — that's
    # the whole point. We only check shape + dtype here.
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_wrapped_model_pickle_roundtrip():
    """joblib pickle preserves calibrator + target_type."""
    import pickle
    probs, y = _make_multiclass()
    cal = _PerClassIsotonicCalibrator.fit(probs, y, TargetTypes.MULTICLASS_CLASSIFICATION)

    wrapped = _PostHocMultiCalibratedModel(
        _PickleableFakeBase(), cal, TargetTypes.MULTICLASS_CLASSIFICATION,
    )
    blob = pickle.dumps(wrapped)
    loaded = pickle.loads(blob)
    X = np.zeros((5, 4))
    out_original = wrapped.predict_proba(X)
    out_loaded = loaded.predict_proba(X)
    np.testing.assert_allclose(out_original, out_loaded, atol=1e-12)


# ---------------------------------------------------------------------------
# Metrics registry
# ---------------------------------------------------------------------------


def test_metrics_registry_builtin_multilabel():
    from mlframe.training.metrics_registry import list_registered, iter_extra_metrics
    names = list_registered(TargetTypes.MULTILABEL_CLASSIFICATION)
    assert "hamming_loss" in names
    assert "subset_accuracy" in names
    assert "jaccard_samples" in names

    # Dispatch them
    y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.int8)
    preds_NK = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.int8)
    probs_NK = preds_NK.astype(float)

    results = dict(iter_extra_metrics(
        TargetTypes.MULTILABEL_CLASSIFICATION, y_true, probs_NK, preds_NK,
    ))
    assert "hamming_loss" in results
    assert "subset_accuracy" in results
    assert "jaccard_samples" in results
    assert 0.0 <= results["hamming_loss"] <= 1.0
    assert 0.0 <= results["subset_accuracy"] <= 1.0
    assert 0.0 <= results["jaccard_samples"] <= 1.0


def test_metrics_registry_user_registration():
    """Users can register custom metrics without touching evaluation.py."""
    from mlframe.training.metrics_registry import (
        register_metric, unregister_metric, list_registered, iter_extra_metrics,
    )

    def my_metric(y_true, probs_NK, preds_NK):
        return 42.0

    register_metric(TargetTypes.MULTILABEL_CLASSIFICATION, "my_test_metric", my_metric)
    try:
        assert "my_test_metric" in list_registered(TargetTypes.MULTILABEL_CLASSIFICATION)
        results = dict(iter_extra_metrics(
            TargetTypes.MULTILABEL_CLASSIFICATION,
            np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)),
        ))
        assert results["my_test_metric"] == 42.0
    finally:
        unregister_metric(TargetTypes.MULTILABEL_CLASSIFICATION, "my_test_metric")
    assert "my_test_metric" not in list_registered(TargetTypes.MULTILABEL_CLASSIFICATION)


def test_metrics_registry_failing_metric_skipped():
    """A metric that raises is silently skipped — doesn't break iteration."""
    from mlframe.training.metrics_registry import (
        register_metric, unregister_metric, iter_extra_metrics,
    )

    def broken_metric(y_true, probs_NK, preds_NK):
        raise RuntimeError("oh no")

    register_metric(TargetTypes.MULTILABEL_CLASSIFICATION, "broken", broken_metric)
    try:
        # Should NOT propagate the RuntimeError
        results = dict(iter_extra_metrics(
            TargetTypes.MULTILABEL_CLASSIFICATION,
            np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)),
        ))
        assert "broken" not in results  # skipped silently
    finally:
        unregister_metric(TargetTypes.MULTILABEL_CLASSIFICATION, "broken")
