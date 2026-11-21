"""Cluster 36 regression: the proba-as-predict wrapper refuses a non-binary classifier (audit 2026-11-20, finding 36-F4)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlframe.training.composite.stacking_multi_stage import _ProbaAsPredictWrapper


def test_proba_wrapper_refuses_a_multiclass_classifier():
    """Column 1 of predict_proba is not a positive-class probability for three classes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = rng.integers(0, 3, 120)
    with pytest.raises(ValueError, match="binary"):
        _ProbaAsPredictWrapper(LogisticRegression(max_iter=200)).fit(X, y)


def test_proba_wrapper_accepts_a_binary_classifier():
    """Binary targets are unchanged."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    wrapper = _ProbaAsPredictWrapper(LogisticRegression(max_iter=200)).fit(X, y)
    assert wrapper.predict(X).shape == (120,)
