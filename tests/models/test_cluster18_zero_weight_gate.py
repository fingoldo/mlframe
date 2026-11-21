"""The quality gate's weighted means do not raise on all-zero sample weights (audit 2026-11-20, finding 18-F9)."""

import numpy as np

from mlframe.models.ensembling.quality_gate import _weighted_mean_or_nan


def test_all_zero_weights_give_nan_not_an_exception():
    """A zero-weight group used to raise ZeroDivisionError inside the gate."""
    assert np.isnan(_weighted_mean_or_nan(np.array([1.0, 2.0]), np.array([0.0, 0.0])))


def test_positive_weights_match_numpy_average():
    """Ordinary weights give the usual weighted mean."""
    values = np.array([1.0, 3.0])
    weights = np.array([1.0, 3.0])
    assert _weighted_mean_or_nan(values, weights) == np.average(values, weights=weights)
