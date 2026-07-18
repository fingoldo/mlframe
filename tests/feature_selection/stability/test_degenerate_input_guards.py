"""Regression tests for degenerate-input guards in feature_selection.

Each guard converts a previously-opaque downstream failure (np.column_stack([]),
LogisticRegression single-class, Jaccard-of-empty == 1.0, partial-length column
indexing, duplicate-name category drop) into a clear early error / verdict.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.hetero_vote import heterogeneous_relevance_vote
from mlframe.feature_selection.compare_selectors import compare_selectors
from mlframe.feature_selection.importance import plot_feature_importance
from mlframe.feature_selection.optbinning import get_binningprocess_featureselectors


def test_hetero_vote_empty_feature_set_returns_empty_verdict():
    """Hetero vote empty feature set returns empty verdict."""
    X = pd.DataFrame(index=range(20))  # zero columns
    y = np.array([0, 1] * 10)
    accepted, info = heterogeneous_relevance_vote(X, y)
    assert accepted == []
    assert info["vote_fraction"] == {}
    assert info["n_models"] == 0


def test_hetero_vote_single_class_y_raises_clear_error():
    """Hetero vote single class y raises clear error."""
    X = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0)[::-1]})
    y = np.zeros(20, dtype=int)  # single class
    with pytest.raises(ValueError, match=r">= 2 classes"):
        heterogeneous_relevance_vote(X, y, classification=True)


def test_compare_selectors_empty_columns_raises():
    """Compare selectors empty columns raises."""
    X = pd.DataFrame(index=range(10))  # zero columns
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match=r">= 1 column"):
        compare_selectors(X, y, selectors=[object()])


def test_plot_feature_importance_partial_columns_raises():
    """Plot feature importance partial columns raises."""
    fi = np.array([0.3, 0.5, 0.2])
    with pytest.raises(ValueError, match=r"must be 0 or len"):
        plot_feature_importance(fi, columns=["a", "b"], kind="x", show_plots=False, log_fi=False)


def test_optbinning_duplicate_column_names_raises():
    # Category dtype on duplicated names: pre-fix ``nocat_cols.remove(col)`` runs once per detected
    # category column but the list holds only one "dup" entry -> opaque ValueError on the second remove.
    """Optbinning duplicate column names raises."""
    df = pd.DataFrame({"a": ["x", "y"] * 4, "b": ["p", "q"] * 4}).astype("category")
    df.columns = ["dup", "dup"]
    with pytest.raises(ValueError, match=r"unique column names"):
        get_binningprocess_featureselectors(df)
