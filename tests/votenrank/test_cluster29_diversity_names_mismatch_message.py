"""Cluster 29 regression: a names mismatch in the diversity ablation raises a ValueError that shows both lists (audit 2026-11-20, finding 29-F34)."""

import numpy as np
import pytest

from mlframe.votenrank import correlation_diversity_ablation as cda


def test_names_mismatch_names_both_lists(monkeypatch):
    """When the correlation matrix returns other names, the error lists them rather than raising an empty AssertionError (29-F34)."""
    monkeypatch.setattr(cda, "residual_correlation_matrix", lambda _p: (np.eye(2), ["b", "a"]))
    oof = {"a": np.array([0.1, 0.2, 0.3]), "b": np.array([0.2, 0.1, 0.4])}
    scores = {"a": 0.5, "b": 0.6}
    with pytest.raises(ValueError, match="do not match"):
        cda.diversity_ablation_report(oof, scores, np.array([0, 1, 1]), lambda y, p: 0.0)
