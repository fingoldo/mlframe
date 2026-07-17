"""Regression tests for degenerate-input / resource-leak fixes in preprocessing.

Covers:
- is_variable_truly_continuous: empty + all-NaN columns short-circuit to discrete+0 (EDGE3).
- compute_naive_outlier_score: train/test feature-count mismatch raises (EDGE5).
- compute_naive_outlier_score: empty train fold raises instead of silently reporting no outliers anywhere.
- clusterize / list_cluster_members: no leaked figures + empty-labels guard (LEAK-P2).
- prepare_df_for_catboost: skipped_columns surfaces columns left unprocessed after a cast failure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlframe.preprocessing.cleaning import is_variable_truly_continuous
from mlframe.preprocessing.outliers import compute_naive_outlier_score
from mlframe.preprocessing.transforms import prepare_df_for_catboost
from mlframe.preprocessing import cluster as cluster_mod


def test_is_variable_truly_continuous_empty_column():
    # Pre-fix: n_outliers/len(values) -> ZeroDivisionError on the empty column.
    """Is variable truly continuous empty column."""
    is_cont, pct = is_variable_truly_continuous(values=np.array([], dtype=float))
    assert is_cont is False
    assert pct == 0.0


def test_is_variable_truly_continuous_all_nan_column():
    # Pre-fix: np.nanmin(all-NaN) raised / NaN poisoned the continuity decision.
    """Is variable truly continuous all nan column."""
    is_cont, pct = is_variable_truly_continuous(values=np.array([np.nan, np.nan, np.nan]))
    assert is_cont is False
    assert pct == 0.0


def test_compute_naive_outlier_score_feature_count_mismatch_raises():
    # Pre-fix: njit indexed train mins/maxs by X_test column count -> silent OOB garbage.
    """Compute naive outlier score feature count mismatch raises."""
    X_train = np.zeros((10, 3), dtype=float)
    X_test = np.zeros((5, 4), dtype=float)
    with pytest.raises(ValueError, match="feature counts must match"):
        compute_naive_outlier_score(X_train, X_test)


def test_compute_naive_outlier_score_empty_train_fold_raises():
    # Pre-fix: nanmin/nanmax on an empty X_train silently collapsed to NaN, and every X_test comparison
    # against NaN evaluated False -> reported "no outliers anywhere" instead of raising on a degenerate fold.
    """Compute naive outlier score empty train fold raises."""
    X_train = np.zeros((0, 3), dtype=float)
    X_test = np.ones((5, 3), dtype=float) * 1000.0  # would be flagged out-of-range against any real bounds
    with pytest.raises(ValueError, match="0 rows"):
        compute_naive_outlier_score(X_train, X_test)


def test_prepare_df_for_catboost_surfaces_skipped_columns():
    # A column of unhashable values (lists) can't be cast to "category" (pandas requires hashable values
    # for the categories index) -- the cast raises, gets caught+logged, and must be surfaced via skipped_columns.
    """Prepare df for catboost surfaces skipped columns."""
    df = pd.DataFrame({"a": pd.Categorical(["x", "y", "x"]), "b": [[1], [2], [3]]})
    skipped: list = []
    out = prepare_df_for_catboost(df, cat_features=["b"], ensure_categorical=True, skipped_columns=skipped)
    assert "b" in skipped
    assert out["b"].dtype != "category"


def test_list_cluster_members_empty_labels_no_raise():
    # Pre-fix: max([]) -> ValueError on empty labels.
    """List cluster members empty labels no raise."""
    cluster_mod.list_cluster_members(np.array([], dtype=int), [])


def test_clusterize_does_not_leak_figures():
    """Clusterize does not leak figures."""
    before = len(plt.get_fignums())
    X = np.random.RandomState(0).randn(60, 2)
    labels = cluster_mod.clusterize(X=X, true_labels=None, show_plot=True, show_metrics=False, list_members=False)
    after = len(plt.get_fignums())
    assert after == before, f"clusterize leaked {after - before} figure(s)"
    assert len(labels) == 60
