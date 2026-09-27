"""BaselineDiagnostics on a target with missing labels runs on the labelled rows instead of skipping or going NaN."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from mlframe.training.baselines.diagnostics import BaselineDiagnostics
from mlframe.training.configs import BaselineDiagnosticsConfig

from .test_baseline_diagnostics import _make_dominant_binary, _make_dominant_regression


@pytest.mark.parametrize("target_type", ["regression", "binary_classification"])
def test_a_target_with_gaps_is_diagnosed_on_its_labelled_rows(target_type):
    """Kept, the unlabelled rows made the regression RMSE NaN (a "degenerate target" skip) and NaN a binary class."""
    df, feats, y = _make_dominant_regression(n=600) if target_type == "regression" else _make_dominant_binary(n=600)
    y = np.asarray(y, dtype=np.float64).copy()
    y[np.random.default_rng(4).random(len(y)) < 0.3] = np.nan
    report = BaselineDiagnostics(BaselineDiagnosticsConfig(quick_model_n_estimators=40, sample_n=None)).fit_and_report(
        train_df=df.drop(columns=["y"]), train_target=y, feature_cols=feats, target_type=target_type, target_name="y"
    )
    assert report.skipped is False, report.skip_reason
    assert math.isfinite(report.headline_metric_value)
