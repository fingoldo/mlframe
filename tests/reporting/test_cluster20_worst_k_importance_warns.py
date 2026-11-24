"""Cluster 20 regression: a mismatched feature-importance vector is reported, not silently replaced (audit 2026-11-20, finding 20-F35).

worst_k_table ignores an importance vector whose length differs from the feature count and falls back to the first columns. It now logs a
WARNING saying so; a matching vector is used silently and sorts as before.
"""

import logging

import numpy as np
import pandas as pd

from mlframe.reporting.charts import error_analysis as ea


def _frame(n=40, p=4):
    """A small regression frame with distinct residuals."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = rng.normal(size=n)
    pred = y + rng.normal(size=n)
    return X, y, pred


def test_mismatched_importance_vector_logs_a_warning(caplog):
    """An importance vector with the wrong length is ignored with a WARNING (20-F35)."""
    X, y, pred = _frame()
    with caplog.at_level(logging.WARNING, logger=ea.logger.name):
        ea.worst_k_table(X, y, pred, task="regression", k=5, feature_importances=[0.1, 0.2])
    assert any(r.levelno == logging.WARNING and "ignoring them" in r.getMessage() for r in caplog.records)


def test_matching_importance_vector_is_silent(caplog):
    """Control: a matching importance vector is used with no warning (20-F35)."""
    X, y, pred = _frame()
    with caplog.at_level(logging.WARNING, logger=ea.logger.name):
        ea.worst_k_table(X, y, pred, task="regression", k=5, feature_importances=[0.4, 0.1, 0.3, 0.2])
    assert not any("ignoring them" in r.getMessage() for r in caplog.records)
