"""Unit tests for ``mlframe.metrics.compute_all_metrics`` -- the per-iteration full-suite aggregator."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics import compute_all_metrics
from mlframe.metrics.iteration_metrics import (
    REGRESSION_METRIC_KEYS,
    _BINARY_METRIC_KEYS,
    _MULTICLASS_METRIC_KEYS,
)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def test_binary_returns_full_key_set(rng):
    n = 2000
    y = rng.integers(0, 2, n)
    score = np.clip(0.35 + 0.3 * y + rng.normal(0, 0.3, n), 0, 1)
    d = compute_all_metrics(y, score, "binary_classification")
    assert set(d) == set(_BINARY_METRIC_KEYS)
    # An informative model on a separable target must have AUC well above chance.
    assert d["ROC_AUC"] > 0.6
    assert np.isfinite(d["ECE"]) and np.isfinite(d["log_loss"]) and np.isfinite(d["MCC"])


def test_binary_accepts_2col_proba(rng):
    n = 800
    y = rng.integers(0, 2, n)
    p1 = np.clip(0.4 + 0.25 * y + rng.normal(0, 0.2, n), 1e-3, 1 - 1e-3)
    proba = np.column_stack([1 - p1, p1])
    d = compute_all_metrics(y, proba, "binary_classification")
    d1 = compute_all_metrics(y, p1, "binary_classification")
    assert d["ROC_AUC"] == d1["ROC_AUC"]


def test_multiclass_returns_full_key_set(rng):
    n = 2000
    y = rng.integers(0, 3, n)
    sc = rng.random((n, 3))
    sc[np.arange(n), y] += 1.2
    d = compute_all_metrics(y, sc, "multiclass_classification")
    assert set(d) == set(_MULTICLASS_METRIC_KEYS)
    assert d["accuracy"] > 0.5
    assert np.isfinite(d["macro_ROC_AUC"]) and np.isfinite(d["log_loss"])


def test_multilabel_macro_averages_binary_keys(rng):
    n = 1500
    y = rng.integers(0, 2, (n, 4))
    score = np.clip(y * 0.5 + rng.random((n, 4)) * 0.4 + 0.1, 0, 1)
    d = compute_all_metrics(y, score, "multilabel_classification")
    assert set(d) == set(_BINARY_METRIC_KEYS)
    assert d["ROC_AUC"] > 0.6


def test_regression_returns_full_key_set(rng):
    n = 2000
    yt = rng.normal(0, 1, n)
    ys = yt + rng.normal(0, 0.3, n)
    d = compute_all_metrics(yt, ys, "regression")
    assert set(d) == set(REGRESSION_METRIC_KEYS)
    assert d["R2"] > 0.8
    assert d["MAE"] > 0 and d["RMSE"] >= d["MAE"]


def test_single_class_val_degrades_gracefully(rng):
    n = 1000
    y = np.zeros(n, dtype=int)  # no positives -> ranking metrics undefined
    score = rng.random(n)
    d = compute_all_metrics(y, score, "binary_classification")
    # Ranking metrics NaN, calibration metrics still finite.
    assert np.isnan(d["ROC_AUC"]) and np.isnan(d["PR_AUC"]) and np.isnan(d["KS"])
    assert np.isfinite(d["ECE"]) and np.isfinite(d["brier_loss"])


def test_nan_scores_dropped_not_propagated(rng):
    n = 1000
    y = rng.integers(0, 2, n)
    score = np.clip(0.35 + 0.3 * y + rng.normal(0, 0.3, n), 0, 1)
    score[:10] = np.nan
    d = compute_all_metrics(y, score, "binary_classification")
    assert np.isfinite(d["ROC_AUC"]) and np.isfinite(d["log_loss"])


def test_all_nan_scores_yields_nan_dict(rng):
    n = 500
    y = rng.integers(0, 2, n)
    score = np.full(n, np.nan)
    d = compute_all_metrics(y, score, "binary_classification")
    assert all(np.isnan(v) for v in d.values())


def test_empty_input_yields_nan_dict():
    d = compute_all_metrics(np.array([], dtype=int), np.array([], dtype=float), "binary_classification")
    assert set(d) == set(_BINARY_METRIC_KEYS)
    assert all(np.isnan(v) for v in d.values())


def test_targettypes_enum_member_accepted(rng):
    from mlframe.training import TargetTypes

    n = 800
    y = rng.integers(0, 2, n)
    score = np.clip(0.4 + 0.2 * y + rng.normal(0, 0.2, n), 0, 1)
    d = compute_all_metrics(y, score, TargetTypes.BINARY_CLASSIFICATION)
    assert set(d) == set(_BINARY_METRIC_KEYS)


def test_unknown_type_1d_falls_back_to_regression(rng):
    n = 500
    yt = rng.normal(0, 1, n)
    ys = yt + rng.normal(0, 0.2, n)
    d = compute_all_metrics(yt, ys, "learning_to_rank")
    assert set(d) == set(REGRESSION_METRIC_KEYS)
