"""sklearn-contract regression test for KerasCompatibleMLP (SK5).

sklearn constructs estimators freely (clone), so __init__ must NOT raise when
TensorFlow is absent and must NOT set learned attrs (model_). The _HAS_TF check
belongs in fit().
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from mlframe.training.neural import keras_compat
from mlframe.training.neural.keras_compat import KerasCompatibleMLP


def test_sk5_constructs_without_tf_and_no_learned_attr_in_init():
    # Must not raise even when TF is unavailable.
    est = KerasCompatibleMLP()
    # __init__ must not set the learned attr (trailing-underscore convention).
    assert not hasattr(est, "model_")
    # clone() round-trips (only reads constructor params).
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()


def test_sk5_fit_raises_clear_error_when_tf_absent():
    if keras_compat._HAS_TF:
        pytest.skip("tensorflow installed; the missing-TF fit error path is not exercised here")
    est = KerasCompatibleMLP()
    X = np.zeros((4, 3), dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)
    with pytest.raises(ImportError):
        est.fit(X, y)
