"""Regression test for fuzz c0103: MultiOutputClassifier.predict_proba (a Python list of per-label
(N, 2) arrays) getting silently np.asarray()-stacked into (n_labels, N, 2) by _predict_with_fallback,
which downstream code (_canonical_predict_proba_shape) couldn't recognise as the list form and crashed
with "unsupported probs shape (n_labels, N, 2)".
"""

import numpy as np

from mlframe.training.cb._cb_pool import _predict_with_fallback, _wrap_predict_result
from mlframe.training._classif_helpers import _canonical_predict_proba_shape


class _FakeMultiOutputClassifier:
    """Mimics sklearn's MultiOutputClassifier.predict_proba: list[(N, 2)], one per label."""

    def __init__(self, n_labels=3, n_rows=5):
        self.n_labels = n_labels
        self.n_rows = n_rows
        self.classes_ = [np.array([0, 1]) for _ in range(n_labels)]

    def predict_proba(self, X):
        """Predict proba."""
        rng = np.random.default_rng(0)
        return [np.column_stack([1 - p, p]) for p in (rng.uniform(size=(self.n_labels, self.n_rows)))]

    def predict(self, X):
        """Predict."""
        return np.zeros((self.n_rows, self.n_labels), dtype=int)


def test_wrap_predict_result_canonicalizes_list_proba():
    """Wrap predict result canonicalizes list proba."""
    lst = [np.column_stack([1 - np.linspace(0, 1, 5), np.linspace(0, 1, 5)]) for _ in range(3)]
    out = _wrap_predict_result(lst, method="predict_proba", classes_=None)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape == (5, 3)


def test_predict_with_fallback_multioutput_proba_is_2d():
    """Predict with fallback multioutput proba is 2d."""
    model = _FakeMultiOutputClassifier(n_labels=3, n_rows=7)
    X = np.zeros((7, 2))
    probs = _predict_with_fallback(model, X, method="predict_proba")
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape == (7, 3)
    # Downstream canonicalization must be a no-op on an already-canonical (N, K) array.
    canon = _canonical_predict_proba_shape(probs)
    assert canon.shape == probs.shape
