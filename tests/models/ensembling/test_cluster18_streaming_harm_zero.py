"""Cluster 18 regression: the streaming harm ensemble returns exactly 0 in a cell where a member predicts 0 (audit 2026-11-20, finding 18-F15).

The materialised path forces the harmonic mean to 0 for a cell with any zero member. The streaming path used to return about 3e-150 there (a
1e150 surrogate for 1/0). The zero is now tracked per cell, so the streamed value is exactly 0, and the other cells keep their harmonic mean.
"""

import numpy as np

from mlframe.models.ensembling.predict import ensemble_probabilistic_predictions_streaming


def _members():
    """Three (N, 2) probability members. Cell (row 0, column 0) has a zero in member ``a``; the rest are non-zero."""
    a = np.array([[0.0, 1.0], [0.4, 0.6], [0.2, 0.8]])
    b = np.array([[0.5, 0.5], [0.3, 0.7], [0.6, 0.4]])
    c = np.array([[0.7, 0.3], [0.5, 0.5], [0.9, 0.1]])
    return a, b, c


def test_zero_cell_is_exactly_zero():
    """The cell with a zero member is exactly 0.0 in the streamed harm, not a 1e-150 residue (18-F15)."""
    streamed = ensemble_probabilistic_predictions_streaming(*_members(), ensemble_method="harm", ensure_prob_limits=False, verbose=False)[0]
    assert streamed[0, 0] == 0.0


def test_nonzero_cell_keeps_its_harmonic_mean():
    """A cell without a zero keeps the plain harmonic mean: column 1 of row 0 is 3 / (1/1.0 + 1/0.5 + 1/0.3) (18-F15)."""
    streamed = ensemble_probabilistic_predictions_streaming(*_members(), ensemble_method="harm", ensure_prob_limits=False, verbose=False)[0]
    expected = 3.0 / (1.0 / 1.0 + 1.0 / 0.5 + 1.0 / 0.3)
    assert np.isclose(streamed[0, 1], expected, rtol=0, atol=1e-12)
