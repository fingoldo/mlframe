"""GroundTruth.mi_reference memoises on the data, not only on the column names (audit 2026-11-20, finding 04-F9)."""
import numpy as np

import mlframe.data.datasets._oracle as oracle
from mlframe.data.datasets.ground_truth import GroundTruth


def test_same_columns_with_other_labels_recompute(monkeypatch):
    """Two calls with identical column names but different labels must both reach the estimator."""
    calls = []

    def fake_reference_mi(columns, labels, true_prob=None):
        """Record the call and return a marker dict."""
        calls.append(float(np.sum(labels)))
        return {"marker": calls[-1]}

    monkeypatch.setattr(oracle, "reference_mi", fake_reference_mi)
    truth = GroundTruth(features={})
    cols = {"x1": np.arange(10.0)}
    first = truth.mi_reference(cols, np.zeros(10))
    second = truth.mi_reference(cols, np.ones(10))
    assert len(calls) == 2
    assert first != second
