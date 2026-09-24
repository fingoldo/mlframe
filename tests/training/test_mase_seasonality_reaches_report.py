"""ReportingConfig.mase_seasonality sets the lag of the train naive-MAE scale, and the report computes MASE with it."""

import numpy as np

from mlframe.training._prediction_envelope_clip import TrainEnvelopeStats, train_naive_mae


def test_train_naive_mae_uses_the_lag():
    y = np.tile([0.0, 10.0], 10)
    assert train_naive_mae(y, 1) == 10.0
    assert train_naive_mae(y, 2) is None  # the lag-2 naive forecast is exact: no scale
    assert train_naive_mae(np.ones(3), 5) is None


def test_report_computes_mase_from_the_threaded_scale():
    from mlframe.training.reporting._reporting_regression import report_regression_model_perf

    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    preds = y + 0.5
    stats = TrainEnvelopeStats(float(y.min()), float(y.max()), float(y.std()), naive_mae=2.0)
    metrics = {}
    report_regression_model_perf(targets=y, columns=[], model=None, model_name="m", preds=preds, metrics=metrics, print_report=False,
                                 show_perf_chart=False, mase_naive_mae=stats.naive_mae, mase_seasonality=7)
    flat = metrics if "MASE" in metrics else next(v for v in metrics.values() if isinstance(v, dict) and "MASE" in v)
    assert abs(flat["MASE"] - 0.25) < 1e-9 and flat["MASE_seasonality"] == 7
