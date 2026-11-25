"""Audit 2026-11-20 finding 40-F22: MLPRanker.predict checks the column names and order, not only the count (behavioural test)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightning")
from mlframe.training.neural.ranker import MLPRanker


def _frame(n=40, seed=0):
    """Small ranking frame with named columns; the label is driven by column ``a``, 4 rows per query."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = (X["a"].to_numpy() > 0).astype(float)
    group = np.repeat(np.arange(n // 4), 4)
    return X, y, group


def test_predict_refuses_reordered_columns_of_equal_width():
    """Same width, columns in a different order: predict raises instead of scoring with the wrong columns (40-F22)."""
    X, y, group = _frame()
    m = MLPRanker(n_epochs=1, random_state=0).fit(X, y, group_ids=group)
    with pytest.raises(ValueError, match="name or order"):
        m.predict(X[["b", "a", "c"]])


def test_predict_accepts_the_fitted_columns():
    """Control: the same columns in the same order are scored (40-F22)."""
    X, y, group = _frame()
    m = MLPRanker(n_epochs=1, random_state=0).fit(X, y, group_ids=group)
    scores = m.predict(X)
    assert scores.shape == (len(X),)
