"""Cluster 21 regressions: off-{0,1} labels are named in a log line, and a failing SHAP score proxy is a WARNING (audit 2026-11-20, 21-F7, 21-F15)."""

import logging

import numpy as np

from mlframe.reporting.charts import risk_coverage as rc
from mlframe.reporting.charts import shap_panels as sp


def test_off_binary_labels_are_logged_and_named(caplog):
    """Labels coded {1, 2} are dropped with a WARNING that names them, instead of silently (21-F7)."""
    y = np.array([1, 2, 1, 2, 1, 2])
    p = np.linspace(0.1, 0.9, 6)
    with caplog.at_level(logging.WARNING, logger=rc.logger.name):
        rc._prepare_classification(y, p, task="binary")
    assert any("outside {0, 1}" in r.getMessage() for r in caplog.records)


class _BrokenModel:
    """A model whose every output method raises."""

    def decision_function(self, X):
        """Always fail."""
        raise RuntimeError("no decision function for this frame")


def test_failing_score_proxy_logs_a_warning(caplog):
    """A raising decision_function is reported at WARNING and the proxy returns None (21-F15)."""
    with caplog.at_level(logging.WARNING, logger=sp.logger.name):
        out = sp._score_proxy(_BrokenModel(), np.zeros((4, 2)), 4)
    assert out is None
    assert any("decision_function() failed" in r.getMessage() for r in caplog.records)
