"""Regression tests for the audit findings in this module."""
import pytest
"""Classifier score() without sample weights returns accuracy, not the CAP accuracy ratio (audit 2026-11-20, finding 40-F2)."""
import numpy as np
from sklearn.base import ClassifierMixin

pytest.importorskip("lightning")
from mlframe.training.neural.base._base_predict import _PredictMixin


class _FixedPredictions(_PredictMixin, ClassifierMixin):
    """Classifier whose predict() returns given labels, so score() can be checked without training a network."""

    def __init__(self, labels):
        """Init."""
        self._labels = np.asarray(labels)

    def predict(self, X):
        """Return the fixed labels for any input."""
        return self._labels


def test_unweighted_score_is_the_fraction_of_correct_labels():
    """Three of four labels match: accuracy is 0.75 (the accuracy ratio of these scores is a different number)."""
    y = np.array([0, 1, 1, 0])
    model = _FixedPredictions([0, 1, 1, 1])
    assert model.score(np.zeros((4, 1)), y) == 0.75


def test_weighted_score_agrees_with_the_unweighted_one_for_equal_weights():
    """With equal sample weights the weighted branch reports the same accuracy as the unweighted one."""
    y = np.array([0, 1, 1, 0])
    model = _FixedPredictions([0, 1, 1, 1])
    unweighted = model.score(np.zeros((4, 1)), y)
    weighted = model.score(np.zeros((4, 1)), y, sample_weight=np.ones(4))
    assert unweighted == weighted == 0.75
