"""Cluster 21 regressions: an all-missing slice column takes the missingness code, and a ranking metric with a cutoff keeps its direction (audit 2026-11-20, findings 21-F5, 21-F16)."""

import numpy as np

from mlframe.reporting.charts.slice_finder import _bin_matrix
from mlframe.reporting.charts.training_curve import _metric_panel


def test_all_missing_column_gets_the_missingness_code_not_bin_zero():
    """A column that is entirely NaN used to land in bin 0 with a numeric range (21-F5)."""
    mat = np.column_stack([np.full(6, np.nan), np.arange(6.0)])
    codes, _edges, has_missing = _bin_matrix(mat, nbins=3)
    assert has_missing[0] is True
    assert set(codes[:, 0].tolist()) == {1}


def test_ranking_metric_with_cutoff_is_treated_as_higher_is_better():
    """'ndcg@10' must be recognised as a higher-is-better metric, so its optimum is the maximum (21-F16)."""
    splits = {"train": np.linspace(0.1, 0.9, 6), "val": np.array([0.2, 0.5, 0.9, 0.8, 0.6, 0.4])}
    spec = _metric_panel("ndcg@10", splits, es_iteration=None)
    assert spec.point_markers and spec.point_markers[0][0] == 2.0  # the maximum of the validation curve
