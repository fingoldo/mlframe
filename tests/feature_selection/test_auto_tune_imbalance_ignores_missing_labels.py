"""The imbalance of a float target ignores its missing labels (audit 2026-11-20, finding 14-F2)."""
import numpy as np

from mlframe.feature_selection.wrappers._auto_tune import _target_type_and_imbalance


def test_nan_labels_are_not_counted_as_a_class():
    """Three zeros, two ones and two NaN: the minority share is 2/5 over the labelled rows, not 2/7 with NaN counted as a class."""
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, np.nan, np.nan])
    target_type, imbalance = _target_type_and_imbalance(y, n=7)
    assert target_type == "binary"
    assert abs(imbalance - 2.0 / 5.0) < 1e-9
