"""The multiclass MCC refuses a label outside [0, K) instead of dropping the row (audit 2026-11-20, finding 17-F7)."""
import numpy as np
import pytest

from mlframe.metrics.classification._classification_extras import matthews_corrcoef_multiclass


def test_label_outside_the_range_is_refused():
    """A label equal to K used to be skipped by the kernel and the total shrank silently."""
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2])
    with pytest.raises(ValueError, match="outside"):
        matthews_corrcoef_multiclass(y_true, y_pred, n_classes=3)


def test_labels_in_range_still_give_the_mcc():
    """In-range labels keep the MCC; a perfect prediction scores 1."""
    y = np.array([0, 1, 2, 0, 1, 2])
    assert abs(matthews_corrcoef_multiclass(y, y, n_classes=3) - 1.0) < 1e-9
