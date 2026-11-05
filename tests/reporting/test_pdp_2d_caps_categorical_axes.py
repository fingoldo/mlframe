"""The two-feature PDP caps a categorical axis to the same number of labels as the one-feature PDP.

A categorical axis with k labels costs k predictions over (sample x k) rows. The one-feature sweep capped the labels to
``CATEGORICAL_PDP_MAX_CATEGORIES``; the two-feature sweep swept every declared label, so a 2000-category column made a 2000-row surface and
a 2000 x grid predict over sample x 2000 rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.reporting.charts import pdp_ice


class _CountingModel:
    """A predictor that records how many rows it was asked to score, and returns a fixed score per row."""

    def __init__(self):
        self.rows_scored = 0
        self.calls = 0

    def predict(self, X):
        """Score ``X`` (rows are counted) with the mean of the numeric column."""
        self.calls += 1
        self.rows_scored += len(X)
        return np.asarray(X["num"], dtype=np.float64)


def _frame(n_categories: int, n: int = 600, seed: int = 0):
    """A categorical column with ``n_categories`` labels and a numeric column."""
    rng = np.random.default_rng(seed)
    cat = pd.Categorical(rng.integers(0, n_categories, size=n).astype(str))
    return pd.DataFrame({"cat": cat, "num": rng.normal(size=n)})


def test_two_feature_pdp_caps_a_high_cardinality_categorical_axis():
    """A 2000-label categorical axis is swept over at most CATEGORICAL_PDP_MAX_CATEGORIES labels, and predicts over that many."""
    X = _frame(2000)
    model = _CountingModel()
    out = pdp_ice.compute_pdp_2d(model, X, ("cat", "num"), grid=10, sample=40, seed=0)
    cap = pdp_ice.CATEGORICAL_PDP_MAX_CATEGORIES
    assert out["surface"].shape[0] <= cap
    assert model.rows_scored <= out["surface"].shape[0] * 40 * out["surface"].shape[1]
    assert model.rows_scored < 40 * 2000


def test_low_cardinality_categorical_axis_is_not_truncated():
    """A categorical axis with fewer labels than the cap keeps every label."""
    X = _frame(4, n=200)
    out = pdp_ice.compute_pdp_2d(_CountingModel(), X, ("cat", "num"), grid=10, sample=40, seed=0)
    assert out["surface"].shape[0] == len(X["cat"].cat.categories)
