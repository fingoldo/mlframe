"""The binary positive-class column is documented as label 1; a model whose second class is something else is reported."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.inference.predict import get_models_raw_predictions


class _Binary:
    """Two-class model exposing ``classes_`` and fixed probabilities."""

    def __init__(self, classes):
        """Store the class labels."""
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        """Return constant two-column probabilities."""
        return np.tile([0.3, 0.7], (len(X), 1))


def _run(classes, caplog, name):
    """Predict with a model whose ``classes_`` is ``classes`` and return the captured warnings."""
    X = pd.DataFrame({"a": [1.0, 2.0]})
    with caplog.at_level(logging.WARNING, logger="mlframe.inference.predict"):
        out = get_models_raw_predictions({name: _Binary(classes)}, X, allow_unvalidatable_models=True)
    return out, [r.getMessage() for r in caplog.records if "not of label 1" in r.getMessage()]


def test_non_unit_second_class_is_reported(caplog):
    """classes_ [0, 2] makes the returned column the probability of 2, which the warning names."""
    out, msgs = _run([0, 2], caplog, "m_two")
    assert np.allclose(out["m_two"], 0.7)
    assert msgs and "2" in msgs[0]


def test_zero_one_classes_are_silent(caplog):
    """The conventional [0, 1] labelling raises no warning."""
    _, msgs = _run([0, 1], caplog, "m_std")
    assert not msgs
