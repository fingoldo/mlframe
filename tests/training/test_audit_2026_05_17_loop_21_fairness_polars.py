"""Regression: compute_fairness_metrics must accept polars Series and
pandas inputs interchangeably.

Pre-fix path (fuzz c0143_276cf2b5):
1. Suite trains a CB regressor on a polars-nullable frame; the native
   model.predict returns a polars Series.
2. report_regression_model_perf threads `preds` straight through to
   compute_fairness_metrics(..., y_pred=preds) without coercing.
3. Inside compute_fairness_metrics, the bin-loop builds
   ``idx = bins == bin_name`` where `bins` is a pandas Series sliced
   by subset_index. `idx` is then a pandas boolean Series.
4. ``y_pred[idx]`` invokes polars Series.__getitem__ which rejects
   pandas-Series keys with
   ``TypeError: cannot select elements using key of type
   'pandas.core.series.Series': ...``.

Post-fix:
- y_true / y_pred coerced to numpy at the function boundary (uniform
  indexable surface regardless of caller-side carrier type).
- idx coerced to numpy at the indexing site for symmetry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.metrics.core import compute_fairness_metrics
from mlframe.metrics.core import fast_mean_absolute_error, fast_root_mean_squared_error


def _make_inputs(n: int = 200, n_bins: int = 4):
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(n).astype(np.float64)
    y_pred = y_true + rng.standard_normal(n).astype(np.float64) * 0.1
    bins_arr = (rng.integers(0, n_bins, size=n)).astype(object)
    bins_arr = np.array([f"bin_{int(b)}" for b in bins_arr], dtype=object)
    bins_series = pd.Series(bins_arr, index=np.arange(n))
    subgroups = {
        "categorical_group": {
            "bins": bins_series,
            "bins_names": None,
        },
    }
    subset_index = np.arange(n)
    metrics_dict = {"MAE": fast_mean_absolute_error, "RMSE": fast_root_mean_squared_error}
    higher_is_better = {"MAE": False, "RMSE": False}
    return y_true, y_pred, subgroups, subset_index, metrics_dict, higher_is_better


def test_compute_fairness_metrics_with_polars_series_y_pred() -> None:
    """Polars Series y_pred (model native output) must not crash the
    bin-loop indexing."""
    y_true, y_pred, subgroups, subset_index, metrics_dict, higher_is_better = _make_inputs()
    y_pred_pl = pl.Series("y_pred", y_pred)
    out = compute_fairness_metrics(
        metrics=metrics_dict,
        metrics_higher_is_better=higher_is_better,
        subgroups=subgroups,
        subset_index=subset_index,
        y_true=y_true,
        y_pred=y_pred_pl,
    )
    assert out is not None
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0


def test_compute_fairness_metrics_with_polars_series_y_true() -> None:
    """Same coverage for y_true on the polars side."""
    y_true, y_pred, subgroups, subset_index, metrics_dict, higher_is_better = _make_inputs()
    y_true_pl = pl.Series("y_true", y_true)
    out = compute_fairness_metrics(
        metrics=metrics_dict,
        metrics_higher_is_better=higher_is_better,
        subgroups=subgroups,
        subset_index=subset_index,
        y_true=y_true_pl,
        y_pred=y_pred,
    )
    assert isinstance(out, pd.DataFrame)


def test_compute_fairness_metrics_with_pandas_series() -> None:
    """Baseline: pandas Series y_pred (legacy carrier) still works."""
    y_true, y_pred, subgroups, subset_index, metrics_dict, higher_is_better = _make_inputs()
    y_pred_pd = pd.Series(y_pred, name="y_pred")
    out = compute_fairness_metrics(
        metrics=metrics_dict,
        metrics_higher_is_better=higher_is_better,
        subgroups=subgroups,
        subset_index=subset_index,
        y_true=y_true,
        y_pred=y_pred_pd,
    )
    assert isinstance(out, pd.DataFrame)


def test_compute_fairness_metrics_with_plain_numpy() -> None:
    """Baseline: plain ndarray input (declared signature) still works."""
    y_true, y_pred, subgroups, subset_index, metrics_dict, higher_is_better = _make_inputs()
    out = compute_fairness_metrics(
        metrics=metrics_dict,
        metrics_higher_is_better=higher_is_better,
        subgroups=subgroups,
        subset_index=subset_index,
        y_true=y_true,
        y_pred=y_pred,
    )
    assert isinstance(out, pd.DataFrame)
