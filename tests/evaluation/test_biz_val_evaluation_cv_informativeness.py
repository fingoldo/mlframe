"""biz_value test for ``evaluation.cv_informativeness_check``.

The win: when group-CV genuinely carries cross-group signal (y depends on X consistently across groups), the
real model should beat the leaked-stats dummy in most folds (``informative=True``). When groups are mutually
unrelated (each group's y is an independent random offset with no relation to X or other groups -- the exact
failure mode the source writeup hit), the real model should NOT reliably beat the leaked dummy
(``informative=False``), correctly flagging that CV-driven decisions are untrustworthy there.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.evaluation.cv_informativeness import cv_informativeness_check


def _neg_rmse(y_true, y_pred):
    return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _make_group_splits(group_ids: np.ndarray):
    for g in np.unique(group_ids):
        test_idx = np.flatnonzero(group_ids == g)
        train_idx = np.flatnonzero(group_ids != g)
        yield train_idx, test_idx


def test_biz_val_cv_informativeness_flags_uninformative_groups():
    rng = np.random.default_rng(0)
    n_groups = 8
    rows_per_group = 200
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    X = rng.normal(0, 1, (n_groups * rows_per_group, 3))
    # each group's y is an independent random offset unrelated to X or other groups -- no cross-group signal.
    group_offsets = rng.normal(0, 5, n_groups)
    y = group_offsets[group_ids] + rng.normal(0, 0.1, n_groups * rows_per_group)

    result = cv_informativeness_check(X, y, _make_group_splits(group_ids), model_factory=lambda: LinearRegression(), metric_fn=_neg_rmse, maximize=True)
    assert result["informative"] is False, result


def test_biz_val_cv_informativeness_confirms_informative_groups():
    rng = np.random.default_rng(1)
    n_groups = 8
    rows_per_group = 200
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    X = rng.normal(0, 1, (n_groups * rows_per_group, 3))
    # y depends consistently on X across ALL groups -- real cross-group signal.
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 0.3, n_groups * rows_per_group)

    result = cv_informativeness_check(X, y, _make_group_splits(group_ids), model_factory=lambda: LinearRegression(), metric_fn=_neg_rmse, maximize=True)
    assert result["informative"] is True, result
    assert result["fraction_folds_informative"] > 0.7


def _make_variable_size_group_splits(group_ids: np.ndarray):
    for g in np.unique(group_ids):
        test_idx = np.flatnonzero(group_ids == g)
        train_idx = np.flatnonzero(group_ids != g)
        yield train_idx, test_idx


def test_biz_val_cv_informativeness_trend_uniformly_uninformative():
    # every group's y is an independent random offset unrelated to X, REGARDLESS of group size -- there is no
    # size-informativeness relationship to find; the diagnostic must not manufacture a spurious sparsity trend.
    rng = np.random.default_rng(2)
    n_groups = 10
    group_sizes = rng.integers(50, 400, n_groups)
    group_ids = np.repeat(np.arange(n_groups), group_sizes)
    n = int(group_sizes.sum())
    X = rng.normal(0, 1, (n, 3))
    group_offsets = rng.normal(0, 5, n_groups)
    y = group_offsets[group_ids] + rng.normal(0, 0.1, n)

    result = cv_informativeness_check(
        X,
        y,
        _make_variable_size_group_splits(group_ids),
        model_factory=lambda: LinearRegression(),
        metric_fn=_neg_rmse,
        maximize=True,
        check_trend=True,
    )
    assert result["informative"] is False, result
    trend = result["trend_diagnostic"]
    assert trend["classification"] == "uniformly_uninformative", trend
    assert abs(trend["correlation"]) < 0.5, trend


def test_biz_val_cv_informativeness_trend_decaying_with_group_sparsity():
    # small held-out groups behave like the fully-uninformative scenario (y unrelated to X, an independent
    # per-group offset); large held-out groups carry the genuine X->y relationship. This directly encodes the
    # remediation story: re-folding into fewer, larger groups would recover informativeness here, unlike the
    # uniformly-uninformative case above where no re-folding could help.
    rng = np.random.default_rng(3)
    group_sizes = np.array([15, 15, 15, 15, 300, 300, 300, 300])
    n_groups = len(group_sizes)
    group_ids = np.repeat(np.arange(n_groups), group_sizes)
    n = int(group_sizes.sum())
    X = rng.normal(0, 1, (n, 3))
    y = np.empty(n)
    start = 0
    for _g, size in enumerate(group_sizes):
        end = start + size
        if size < 100:
            y[start:end] = rng.normal(0, 5) + rng.normal(0, 0.1, size)
        else:
            y[start:end] = 3.0 * X[start:end, 0] - 2.0 * X[start:end, 1] + rng.normal(0, 0.3, size)
        start = end

    result = cv_informativeness_check(
        X,
        y,
        _make_variable_size_group_splits(group_ids),
        model_factory=lambda: LinearRegression(),
        metric_fn=_neg_rmse,
        maximize=True,
        check_trend=True,
    )
    trend = result["trend_diagnostic"]
    assert trend["classification"] == "decaying_with_group_sparsity", trend
    assert trend["correlation"] > 0.5, trend
    # small-fold margins measurably worse than large-fold margins.
    small_margins = trend["margins"][:4]
    large_margins = trend["margins"][4:]
    assert np.mean(small_margins) < np.mean(large_margins), trend


def test_cv_informativeness_check_trend_default_off_output_unchanged():
    # opt-in contract: omitting check_trend must not add a trend_diagnostic key or otherwise change the result.
    rng = np.random.default_rng(4)
    n_groups = 6
    rows_per_group = 100
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    X = rng.normal(0, 1, (n_groups * rows_per_group, 3))
    y = 3.0 * X[:, 0] + rng.normal(0, 0.3, n_groups * rows_per_group)

    result_default = cv_informativeness_check(X, y, _make_group_splits(group_ids), model_factory=lambda: LinearRegression(), metric_fn=_neg_rmse, maximize=True)
    result_explicit_off = cv_informativeness_check(
        X,
        y,
        _make_group_splits(group_ids),
        model_factory=lambda: LinearRegression(),
        metric_fn=_neg_rmse,
        maximize=True,
        check_trend=False,
    )
    assert "trend_diagnostic" not in result_default
    assert result_default == result_explicit_off


def test_cv_informativeness_invalid_stat_raises():
    import pytest

    with pytest.raises(ValueError):
        cv_informativeness_check(
            np.zeros((4, 1)),
            np.zeros(4),
            [(np.array([0, 1]), np.array([2, 3]))],
            model_factory=lambda: LinearRegression(),
            metric_fn=_neg_rmse,
            leaked_dummy_stat="bogus",
        )
