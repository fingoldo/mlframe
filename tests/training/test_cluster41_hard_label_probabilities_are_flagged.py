"""Cluster 41 regression: the predict()-fallback probabilities are labelled as hard 0/1 indicators in the log (audit 2026-11-20, finding 41-F14)."""

import logging

import numpy as np

from mlframe.training.reporting._reporting_probabilistic_helpers import _predict_probs_if_missing


class _LabelsOnly:
    """A classifier with predict() but no predict_proba."""

    classes_ = np.array([0, 1])

    def predict(self, X):
        """Alternate the two classes."""
        return np.arange(len(X)) % 2


def test_one_hot_fallback_warns_that_the_probability_metrics_are_label_metrics(caplog):
    """The fallback yields a one-hot matrix and a WARNING naming ROC AUC / Brier as label metrics."""
    import pandas as pd

    with caplog.at_level(logging.WARNING):
        probs = _predict_probs_if_missing(None, _LabelsOnly(), pd.DataFrame({"a": range(6)}))
    assert probs.shape == (6, 2) and set(np.unique(probs)) == {0.0, 1.0}
    assert any("hard 0/1 class indicators" in r.getMessage() and "ROC AUC" in r.getMessage() for r in caplog.records)
