"""Regression: the fairness ROC-AUC path must use the positive-class indicator,
not the raw target labels.

``fast_roc_auc`` assumes ``y_true`` is a 0/1 indicator (it accumulates
``tps += y_true[i]``). ``report_probabilistic_model_perf`` previously threaded
the RAW ``targets`` into ``compute_fairness_metrics`` for the binary ROC-AUC
metric, so non-0/1 binary labels (e.g. {1, 2}) silently produced NaN / wrong
per-group AUC. The fix maps targets to the positive-class indicator
(``targets == classes[1]``) before the fairness/calibration consumers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import roc_auc_score

from mlframe.training.reporting._reporting_probabilistic import (
    report_probabilistic_model_perf,
)


def _run(target_labels):
    rng = np.random.default_rng(1)
    n = 2000
    g = rng.integers(0, 2, n)
    y01 = rng.integers(0, 2, n)
    # weak signal so AUC is clearly distinct from 1.0 / 0.5
    p1 = np.clip(0.18 * y01 + rng.random(n) * 0.82, 0.0, 1.0)
    probs = np.column_stack([1.0 - p1, p1])

    classes = list(target_labels)
    # map 0/1 -> the requested two-label set (label_for_0, label_for_1)
    y = np.where(y01 == 1, classes[1], classes[0])

    feat = pd.Series(g, name="grp")
    subgroups = {"grp": dict(bins=feat, bins_names=[0, 1])}

    metrics: dict = {}
    report_probabilistic_model_perf(
        targets=y, columns=["f0"], model_name="m", model=None,
        classes=classes, probs=probs, preds=None,
        subgroups=subgroups, subset_index=np.arange(n),
        print_report=False, show_perf_chart=False,
        fairness_calibration_charts=False, calibration_by_feature_charts=False,
        calibration_heatmap_2d_charts=False,
        metrics=metrics,
        custom_ice_metric=lambda y_true, y_score, **kw: 0.0,
    )
    fr = metrics["fairness_report"]
    got = float(fr.xs("ROC AUC", level="metric")["metric_mean"].iloc[0])
    expected = float(np.mean([roc_auc_score(y01[g == b], p1[g == b]) for b in (0, 1)]))
    return got, expected


@pytest.mark.parametrize("labels", [[0, 1], [1, 2], ["neg", "pos"]])
def test_fairness_roc_auc_uses_positive_class_indicator(labels):
    got, expected = _run(labels)
    assert np.isfinite(got), f"fairness ROC AUC is non-finite for labels={labels}"
    # pre-fix: labels {1,2} / strings fed raw into fast_roc_auc -> NaN / wrong.
    assert got == pytest.approx(expected, abs=1e-9), (
        f"fairness ROC AUC {got} != sklearn per-group mean {expected} for labels={labels}"
    )
    # sanity: the synthetic must be discriminating (not a trivial 1.0 pass).
    assert 0.6 < expected < 0.85
