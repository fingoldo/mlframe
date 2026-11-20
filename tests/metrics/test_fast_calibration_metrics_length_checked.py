"""fast_calibration_metrics refuses mismatched lengths instead of reading past y_pred (audit 2026-11-20, finding 17-F8)."""
import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_metrics import fast_calibration_metrics


def test_empty_or_mismatched_inputs_are_refused():
    """An empty y_pred against a non-empty y_true used to be read out of bounds by the serial kernel."""
    with pytest.raises(ValueError):
        fast_calibration_metrics(np.array([0.0, 1.0]), np.array([]), 10)
