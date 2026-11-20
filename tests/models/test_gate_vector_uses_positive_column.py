"""A binary predict_proba enters the stacking gate as its positive column, not as 2n flattened values (audit 2026-11-20, finding 18-F3)."""
import numpy as np

from mlframe.models.ensembling.score_gate import _as_gate_vector


def test_binary_proba_gives_the_positive_column():
    """An (n, 2) probability matrix must yield the length-n positive column."""
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    assert np.allclose(_as_gate_vector(proba, 3), [0.1, 0.8, 0.5])


def test_one_dimensional_predictions_are_unchanged():
    """A plain vector keeps its values."""
    assert np.allclose(_as_gate_vector(np.array([1.0, 2.0, 3.0]), 3), [1.0, 2.0, 3.0])
