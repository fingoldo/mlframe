"""Regression report on quantile-regression (N, K) predictions scored against a single-column target.

The dummy-baseline report for a quantile target handed (N, K) per-alpha predictions with a 1-D target to
``report_regression_model_perf``, which fed them to the 1-D MAE kernel and died with a numba TypingError
(``abs(array(float64, 1d, C))``), so every quantile run lost its pre-training floor report.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.core import fast_mean_absolute_error


def _quantile_fixture(n: int = 400):
    """A 1-D target plus (N, 3) per-alpha predictions straddling it by a fixed offset, so MAE against the median column is known exactly."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    preds = np.column_stack([y - 1.0, y + 0.25, y + 1.0])
    return y, preds


def test_quantile_preds_with_1d_target_report_scores_median_column():
    """Headline metrics come from the alpha closest to 0.5; the full (N, K) matrix is still returned."""
    from mlframe.training.evaluation import report_model_perf

    y, preds = _quantile_fixture()
    metrics: dict = {}
    out_preds, out_probs = report_model_perf(
        targets=y, columns=[], df=None, model=None, model_name="q", preds=preds, probs=None, show_fi=False,
        print_report=False, target_type="quantile_regression", quantile_alphas=[0.1, 0.5, 0.9], metrics=metrics,
    )
    assert metrics["MAE"] == pytest.approx(0.25, abs=1e-12)
    assert out_probs is None
    assert np.asarray(out_preds).shape == preds.shape


def test_quantile_preds_without_alphas_fall_back_to_middle_column():
    """With no alphas given, the middle prediction column stands in for the median."""
    from mlframe.training.reporting._reporting_regression import report_regression_model_perf

    y, preds = _quantile_fixture()
    metrics: dict = {}
    report_regression_model_perf(targets=y, columns=[], model_name="q", model=None, preds=preds, print_report=False, metrics=metrics)
    assert metrics["MAE"] == pytest.approx(0.25, abs=1e-12)


def test_fast_mae_rejects_1d_target_with_2d_preds_with_value_error_not_numba_typing_error():
    """A genuine ndim mismatch fails closed with sklearn's ValueError, not an opaque numba TypingError."""
    y, preds = _quantile_fixture()
    with pytest.raises(ValueError, match="different number of output"):
        fast_mean_absolute_error(y, preds)
