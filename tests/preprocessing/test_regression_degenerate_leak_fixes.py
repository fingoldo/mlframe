"""Regression tests for degenerate-input / resource-leak fixes in preprocessing.

Covers:
- is_variable_truly_continuous: empty + all-NaN columns short-circuit to discrete+0 (EDGE3).
- compute_naive_outlier_score: train/test feature-count mismatch raises (EDGE5).
- clusterize / list_cluster_members: no leaked figures + empty-labels guard (LEAK-P2).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlframe.preprocessing.cleaning import is_variable_truly_continuous
from mlframe.preprocessing.outliers import compute_naive_outlier_score
from mlframe.preprocessing import cluster as cluster_mod


def test_is_variable_truly_continuous_empty_column():
    # Pre-fix: n_outliers/len(values) -> ZeroDivisionError on the empty column.
    is_cont, pct = is_variable_truly_continuous(values=np.array([], dtype=float))
    assert is_cont is False
    assert pct == 0.0


def test_is_variable_truly_continuous_all_nan_column():
    # Pre-fix: np.nanmin(all-NaN) raised / NaN poisoned the continuity decision.
    is_cont, pct = is_variable_truly_continuous(values=np.array([np.nan, np.nan, np.nan]))
    assert is_cont is False
    assert pct == 0.0


def test_compute_naive_outlier_score_feature_count_mismatch_raises():
    # Pre-fix: njit indexed train mins/maxs by X_test column count -> silent OOB garbage.
    X_train = np.zeros((10, 3), dtype=float)
    X_test = np.zeros((5, 4), dtype=float)
    with pytest.raises(ValueError, match="feature counts must match"):
        compute_naive_outlier_score(X_train, X_test)


def test_list_cluster_members_empty_labels_no_raise():
    # Pre-fix: max([]) -> ValueError on empty labels.
    cluster_mod.list_cluster_members(np.array([], dtype=int), [])


def test_clusterize_does_not_leak_figures():
    before = len(plt.get_fignums())
    X = np.random.RandomState(0).randn(60, 2)
    labels = cluster_mod.clusterize(X=X, true_labels=None, show_plot=True, show_metrics=False, list_members=False)
    after = len(plt.get_fignums())
    assert after == before, f"clusterize leaked {after - before} figure(s)"
    assert len(labels) == 60
