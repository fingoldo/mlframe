"""Regression tests for the audit findings in this module."""
import pytest
"""Audit 2026-11-20 finding 40-F4: a multilabel recurrent classifier predicts an indicator matrix, not a NotFittedError."""

import numpy as np

pytest.importorskip("lightning")
from mlframe.training.neural._recurrent_wrappers import RecurrentClassifierWrapper


def test_multilabel_predict_returns_thresholded_indicators(monkeypatch):
    """Per-label sigmoid probabilities are thresholded at 0.5 per column; no classes_ attribute is needed."""
    wrapper = RecurrentClassifierWrapper.__new__(RecurrentClassifierWrapper)
    wrapper._is_multilabel = True
    probs = np.array([[0.9, 0.1, 0.6], [0.2, 0.8, 0.4]])
    monkeypatch.setattr(wrapper, "predict_proba", lambda *a, **k: probs)
    out = wrapper.predict(np.zeros((2, 3)))
    np.testing.assert_array_equal(out, [[1, 0, 1], [0, 1, 0]])
