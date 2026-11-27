"""Cluster 33 regression: binary and multiclass base margins clip probabilities with one epsilon (audit 2026-11-20, 33-F4)."""

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite import classification as cls
from mlframe.training.composite.classification import CompositeClassificationEstimator

margin = CompositeClassificationEstimator._margin_from_estimator


class _ProbaOnly:
    """An estimator exposing only predict_proba."""

    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        """Return the stored probabilities."""
        return self._proba


def test_saturated_binary_probability_is_capped_at_the_shared_epsilon():
    """p=0 and p=1 give margins of -/+ log((1-eps)/eps), about 13.8 (33-F4)."""
    est = _ProbaOnly([[1.0, 0.0], [0.0, 1.0]])
    out = margin(SimpleNamespace(n_classes_=2), est, None)
    cap = np.log((1 - cls._PROBA_MARGIN_EPS) / cls._PROBA_MARGIN_EPS)
    np.testing.assert_allclose(out, [-cap, cap])


def test_saturated_multiclass_probability_uses_the_same_epsilon():
    """A zero class probability becomes log(eps) = -13.8, not log(1e-12) = -27.6 (33-F4)."""
    est = _ProbaOnly([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])
    out = margin(SimpleNamespace(n_classes_=3), est, None)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0, 1:], np.log(cls._PROBA_MARGIN_EPS))
    np.testing.assert_allclose(out[1], np.log([0.2, 0.3, 0.5]))
    assert cls._PROBA_MARGIN_EPS == 1e-6
