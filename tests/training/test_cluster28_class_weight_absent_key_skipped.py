"""Audit 2026-11-20 finding 28-F12: a class_weight key with no training label is skipped, not a KeyError at fit."""

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.training.lgb_shim import LGBMClassifierWithDatasetReuse


def test_class_weight_key_absent_from_the_training_labels_is_skipped():
    """Weights for labels 0 and 9 with training labels {0, 1}: label 9 has no class index, so only label 0 keeps its weight."""
    model = LGBMClassifierWithDatasetReuse(class_weight={0: 2.0, 9: 5.0}, verbose=-1)
    model._pre_fit_bookkeeping(np.array([0, 1, 0, 1]))
    assert model._class_weight == {0: 2.0}
