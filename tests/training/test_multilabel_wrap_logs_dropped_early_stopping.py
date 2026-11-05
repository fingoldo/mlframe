"""Wrapping a model for a multilabel target says when it switches off the model's early stopping.

The per-label fits cannot early-stop, so the wrapper turns the setting off and each label trains for its full budget. That changes the fit
and used to happen without any log line. It now logs a warning naming the parameters it dropped, and a failure to drop them is a warning too.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlframe.training._training_loop_objectives import _maybe_wrap_for_2d_target


class _EarlyStoppingClassifier:
    """A classifier advertising early stopping, so the wrapper has something to switch off."""

    def __init__(self, early_stopping=True, **kwargs):
        self.early_stopping = early_stopping

    def get_params(self, deep=True):
        """Parameters the wrapper reads."""
        return {"early_stopping": self.early_stopping}

    def set_params(self, **kwargs):
        """Apply parameters in place."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


def test_dropping_early_stopping_for_a_multilabel_wrap_is_logged(caplog):
    """A model with early stopping on is wrapped; the warning names the parameter that was dropped."""
    model = _EarlyStoppingClassifier(early_stopping=True)
    y_2d = np.array([[0, 1], [1, 0], [0, 1], [1, 1]])
    with caplog.at_level(logging.WARNING, logger="mlframe.training._training_loop_objectives"):
        _maybe_wrap_for_2d_target(model, y_2d)
    assert any("early_stopping" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_a_model_without_early_stopping_logs_nothing_about_it(caplog):
    """Nothing to drop means no warning about early stopping."""
    model = LogisticRegression()
    y_2d = np.array([[0, 1], [1, 0], [0, 1], [1, 1]])
    with caplog.at_level(logging.WARNING, logger="mlframe.training._training_loop_objectives"):
        _maybe_wrap_for_2d_target(model, y_2d)
    assert not any("early" in r.getMessage() for r in caplog.records)
