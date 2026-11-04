"""zero_division is honoured by fast_precision and fast_classification_report, and empty input gives NaN accuracy."""
import numpy as np

from mlframe.metrics._core_precision_mape import fast_classification_report, fast_precision


def test_precision_of_a_class_nobody_predicted_uses_zero_division():
    """No prediction of class 1: sklearn gives zero_division (1), the kernel returned 0.0 whatever was passed."""
    y = np.array([1, 0])
    p = np.array([0, 0])
    assert fast_precision(y, p, 2, 1) == 1.0
    assert fast_precision(y, p, 2, 0) == 0.0


def test_report_empty_denominators_use_zero_division():
    """y_true = y_pred = [0, 0] with 2 classes: class 1 has no support and no predictions, so P and R take zero_division=1."""
    out = fast_classification_report(np.array([0, 0]), np.array([0, 0]), 2, 1)
    precisions, recalls = out[5], out[6]
    assert precisions[1] == 1.0 and recalls[1] == 1.0


def test_empty_input_gives_nan_accuracy_and_balanced_accuracy():
    """An empty slice has no accuracy; 0.0 read as a real worst score."""
    out = fast_classification_report(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert np.isnan(out[2]) and np.isnan(out[3])
