"""Regression test: CBIterationMetricsCallback._predict must route learning_to_rank (and other
non-classification target types) to plain predict, not predict_proba.

Bug: the dispatch used ``"regression" in target_type``, which is False for "learning_to_rank", so a
ranker wired with the genuine target_type string would hit ``model.predict_proba`` -- rankers don't
expose that method, raising AttributeError, silently swallowed by the surrounding broad except (debug-only
log), leaving ``iteration_metrics_`` permanently empty.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.callbacks.iteration_metrics import CBIterationMetricsCallback


class _RankerModel:
    """Stub mimicking a CatBoost ranker: has predict, no predict_proba (matches the real contract)."""

    def predict(self, pool, ntree_end):
        """Predict."""
        return np.array([0.1, 0.2, 0.3])


class _ClassifierModel:
    """Stub mimicking a CatBoost classifier: has predict_proba."""

    def predict_proba(self, pool, ntree_end):
        """Predict proba."""
        return np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])


@pytest.fixture
def y_val():
    """Y val."""
    return np.array([0, 1, 1])


def test_predict_routes_learning_to_rank_to_plain_predict(y_val):
    """learning_to_rank must use model.predict, never model.predict_proba (which rankers lack)."""
    cb = CBIterationMetricsCallback(val_pool=None, y_val=y_val, target_type="learning_to_rank")
    score = cb._predict(_RankerModel(), ntree_end=5)
    assert score is not None
    np.testing.assert_array_equal(score, [0.1, 0.2, 0.3])


def test_predict_routes_binary_classification_to_predict_proba(y_val):
    """binary_classification must still use predict_proba, collapsed to the positive-class column."""
    cb = CBIterationMetricsCallback(val_pool=None, y_val=y_val, target_type="binary_classification")
    score = cb._predict(_ClassifierModel(), ntree_end=5)
    assert score is not None
    np.testing.assert_allclose(score, [0.1, 0.6, 0.8])


def test_predict_routes_regression_to_plain_predict(y_val):
    """regression must still use plain predict (pre-existing behavior, not regressed by the fix)."""
    cb = CBIterationMetricsCallback(val_pool=None, y_val=y_val, target_type="regression")
    score = cb._predict(_RankerModel(), ntree_end=5)
    assert score is not None
    np.testing.assert_array_equal(score, [0.1, 0.2, 0.3])
