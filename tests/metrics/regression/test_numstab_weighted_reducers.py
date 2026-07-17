"""Numerical-stability regression tests for the njit weighted reducers.

Pre-fix the weighted MAE/MSE/R2 kernels divided the accumulated sum by ``wsum`` with no
``wsum <= 0`` guard, so an all-zero weight vector produced 0/0 == nan (MAE/MSE) or a
nan mean propagated into R2. The guard now returns nan explicitly for an empty weight mass.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.metrics.regression._regression_metrics import (
    _fast_mae_weighted_seq,
    _fast_mae_weighted_par,
    _fast_mse_weighted_seq,
    _fast_mse_weighted_par,
    _fast_r2_score_weighted_seq,
    _fast_r2_score_weighted_par,
)


@pytest.mark.parametrize(
    "fn",
    [
        _fast_mae_weighted_seq,
        _fast_mae_weighted_par,
        _fast_mse_weighted_seq,
        _fast_mse_weighted_par,
        _fast_r2_score_weighted_seq,
        _fast_r2_score_weighted_par,
    ],
)
def test_weighted_reducer_zero_weights_returns_nan_not_inf(fn):
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    w = np.zeros(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fn(y_true, y_pred, w)
    assert not np.isinf(res), f"{fn.__name__} returned inf on all-zero weights"
    assert np.isnan(res), f"{fn.__name__} must return nan on all-zero weights, got {res}"


@pytest.mark.parametrize(
    "fn",
    [_fast_mae_weighted_seq, _fast_mse_weighted_seq, _fast_r2_score_weighted_seq],
)
def test_weighted_reducer_normal_weights_unchanged(fn):
    # Guard must not perturb the normal (positive-weight) result.
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    w = np.array([1.0, 2.0, 1.0, 1.0])
    res = fn(y_true, y_pred, w)
    assert np.isfinite(res)
