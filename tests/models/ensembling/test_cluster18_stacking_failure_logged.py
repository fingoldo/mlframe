"""Cluster 18 regression: a failed member test-prediction stacking is logged with its cause (audit 2026-11-20, finding 18-F19).

When the stacking fails the ensemble has no member_test_preds. The failure is logged as a WARNING that carries the exception, so the missing
attribute has a visible reason. A successful stacking still stamps the attribute.
"""

import logging
from types import SimpleNamespace

import numpy as np

from mlframe.models.ensembling.process_method import _stack_member_predictions


def test_failure_is_logged_with_its_cause(caplog):
    """An empty result tuple makes the stamp fail, and the WARNING names the stacking failure (18-F19)."""
    preds = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
    with caplog.at_level(logging.WARNING):
        _stack_member_predictions(preds, ())
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("member test-prediction stacking failed" in m for m in messages)


def test_success_stamps_member_test_preds():
    """Control: a successful stacking sets member_test_preds as a column matrix (18-F19)."""
    entry = SimpleNamespace()
    preds = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
    _stack_member_predictions(preds, (entry,))
    assert entry.member_test_preds.shape == (3, 2)
