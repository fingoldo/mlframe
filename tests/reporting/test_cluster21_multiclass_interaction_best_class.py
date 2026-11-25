"""Cluster 21 regression: multiclass SHAP interactions rank each pair by its strongest class (audit 2026-11-20, finding 21-F6).

A pair that separates only the early classes used to score zero, because the reduction read the last class alone. Each pair now takes the class
it moves most, as the main-effects panel does.
"""

import numpy as np

import mlframe.inference.tree_shap as tree_shap
from mlframe.reporting.charts import shap_interactions as si


class _FakeExplainer:
    """Stands in for the TreeExplainer and returns a fixed (rows, F, F, K) interaction tensor."""

    tensor = None

    def __init__(self, model):
        """Ignore the model: the tensor is set by the test."""

    def shap_interaction_values(self, X):
        """Return the configured interaction tensor."""
        return _FakeExplainer.tensor


def _tensor(k: int) -> np.ndarray:
    """Two rows, two features, K classes; pair (0, 1) is strong only in class 0."""
    t = np.zeros((2, 2, 2, k))
    t[:, 0, 1, 0] = 1.0
    t[:, 1, 0, 0] = 1.0
    return t


def test_multiclass_pair_is_ranked_by_its_strongest_class(monkeypatch):
    """A pair that separates only class 0 of three is ranked by class 0, not by the last class (21-F6)."""
    monkeypatch.setattr(tree_shap, "TreeExplainer", _FakeExplainer)
    _FakeExplainer.tensor = _tensor(3)
    out = si._mean_abs_interaction(object(), None)
    assert np.isclose(out[0, 1], 1.0)
    assert np.isclose(out[1, 0], 1.0)


def test_binary_keeps_the_positive_class_convention(monkeypatch):
    """Control: with two classes the last class is still the one ranked (21-F6)."""
    monkeypatch.setattr(tree_shap, "TreeExplainer", _FakeExplainer)
    t = np.zeros((2, 2, 2, 2))
    t[:, 0, 1, 1] = 0.7
    _FakeExplainer.tensor = t
    out = si._mean_abs_interaction(object(), None)
    assert np.isclose(out[0, 1], 0.7)
