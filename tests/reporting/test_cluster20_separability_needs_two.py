"""Cluster 20 regression: the separability panel refuses fewer than two features with a clear error (audit 2026-11-20, finding 20-F36).

A single feature used to raise a raw IndexError from features[1]. It now raises ValueError naming the count; two features still draw.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts import engineered_separability as es


def _frame(n=300):
    """Two numeric features and a binary target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + X["b"] > 0).astype(int).to_numpy()
    return X, y


def test_single_feature_gives_a_clear_error():
    """One feature is refused with a ValueError naming the count, not an IndexError (20-F36)."""
    X, y = _frame()
    with pytest.raises(ValueError, match="at least 2 features; got 1"):
        es.separability_panel(X, y, ["a"], sample=100)


def test_two_features_still_build_a_panel():
    """Control: two features build the panel as before (20-F36)."""
    X, y = _frame()
    panel = es.separability_panel(X, y, ["a", "b"], sample=200)
    assert panel is not None
