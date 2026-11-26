"""Cluster 20 regressions: time-axis flag, feature-name contract, length guard order, and the relevance-grade probe (audit 2026-11-20, findings 20-F26, 20-F16, 20-F23, 20-F19).

metric_over_time flagged any x as a time axis even when the index was not a datetime, so the renderers drew bucket ordinals as 1970 dates. The ndarray path of
the feature-matrix resolver accepted any number of feature names. The category length guard ran after sampling, so a longer X passed it. The relevance
violin judged the number of grades from the first 50k rows, so data sorted with few early grades took the per-grade path.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts import category_discriminability as cd
from mlframe.reporting.charts import drift
from mlframe.reporting.charts.error_analysis import _resolve_feature_matrix
from mlframe.reporting.charts.ltr import _score_by_rel_panel


def test_non_datetime_index_is_not_marked_as_a_time_axis(monkeypatch):
    """With a RangeIndex the line is a bucket axis: x_is_time is False and the label says bucket (20-F26)."""
    import mlframe.training.evaluation as evaluation

    perf = pd.DataFrame({"roc_auc": [0.7, 0.8, 0.75]}, index=pd.RangeIndex(3))
    monkeypatch.setattr(evaluation, "compute_ml_perf_by_time", lambda *a, **k: perf)
    fig = drift.metric_over_time(np.zeros(3), np.zeros(3), np.arange(3), metric="roc_auc", min_samples=1)
    panel = fig.panels[0][0]
    assert panel.x_is_time is False
    assert panel.xlabel == "bucket"


def test_ndarray_feature_names_must_match_the_column_count():
    """A short or long name list for an ndarray X raises ValueError, as the DataFrame path does; a matching list works (20-F16)."""
    X = np.zeros((5, 3))
    _, names = _resolve_feature_matrix(X, ["a", "b", "c"])
    assert names == ["a", "b", "c"]
    for bad in (["a", "b"], ["a", "b", "c", "d"]):
        with pytest.raises(ValueError, match="one-to-one"):
            _resolve_feature_matrix(X, bad)


def test_longer_x_than_y_is_refused_even_when_y_is_sampled(monkeypatch):
    """When y is above the sampling cap, an X longer than y still raises the length-mismatch error (20-F23)."""
    monkeypatch.setattr(cd, "_COUNT_SUBSAMPLE_CAP", 10)
    n_y = 30
    X = pd.DataFrame({"cat": np.array(["a", "b", "c"])[np.arange(50) % 3]})
    y = (np.arange(n_y) % 2).astype(float)
    with pytest.raises(ValueError, match="length mismatch"):
        cd.category_discriminability_table(X, y, ["cat"], min_support=1)


def test_grades_that_widen_after_the_first_50k_rows_take_the_binned_path():
    """Relevance grades 0-2 in the first 50k rows and 0-20 later still count as many grades, so the violin bins them (20-F19)."""
    n = 120_000
    y_true = np.concatenate([np.arange(60_000) % 3, np.arange(60_000) % 21]).astype(np.int64)
    y_score = np.random.default_rng(0).uniform(size=n)
    panel = _score_by_rel_panel(y_true, y_score, np.arange(n) // 10)
    assert len(panel.group_labels) <= 6, "grades are quantile-binned, not one violin per grade"
