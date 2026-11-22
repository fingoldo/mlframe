"""Audit 2026-11-20 finding 28-F17: set_label writes the same encoded labels fit trained on, not the raw strings."""

import numpy as np
import pytest

pytest.importorskip("lightning")
pytest.importorskip("lightgbm")

from sklearn.preprocessing import LabelEncoder

from mlframe.training.lgb_shim import LGBMClassifierWithDatasetReuse


class _RecordingDataset:
    """Stands in for the cached training Dataset and records the label it is given."""

    def __init__(self):
        """Start with no recorded label."""
        self.label = None

    def set_label(self, y):
        """Record the label array the shim writes."""
        self.label = np.asarray(y)


def test_set_label_applies_the_fitted_label_encoding():
    """With classes ['a', 'b'] fitted, set_label(['b', 'a']) writes [1, 0] to the cached Dataset."""
    model = LGBMClassifierWithDatasetReuse(verbose=-1)
    model._le = LabelEncoder().fit(np.array(["a", "b"]))
    model._cached_train_dataset = _RecordingDataset()
    model.set_label(np.array(["b", "a"]))
    np.testing.assert_array_equal(model._cached_train_dataset.label, [1, 0])
