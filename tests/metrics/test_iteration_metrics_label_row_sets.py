"""Iteration metrics use one row set: rows with a NaN or out-of-range label leave every metric, not only log-loss."""

from __future__ import annotations

import warnings

import numpy as np

import mlframe.metrics.iteration_metrics as mod
from mlframe.metrics.iteration_metrics import _binary_metrics, _multiclass_metrics


def _mc(n=300, k=3, seed=0):
    """Labels and a score matrix that is right most of the time."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    sc = rng.random((n, k)) * 0.2
    sc[np.arange(n), y] += 0.8
    return y, sc


def test_nan_binary_label_is_dropped_not_counted_as_a_class():
    """A NaN label used to cast to INT_MIN, give a hugely negative positive count and enter every kernel."""
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.4).astype(float)
    p = np.clip(y * 0.6 + rng.random(400) * 0.4, 0, 1)
    clean = _binary_metrics(y, p, 10)
    y_nan, p_nan = np.append(y, np.nan), np.append(p, 0.5)
    got = _binary_metrics(y_nan, p_nan, 10)
    for key in ("accuracy", "ROC_AUC", "log_loss", "MCC"):
        assert np.isfinite(got[key]) and got[key] == clean[key], key


def test_multiclass_out_of_range_label_leaves_every_metric_unchanged():
    """One label 99 appended to a clean set must not move accuracy, MCC or the macro averages: the row is excluded everywhere, not only from log-loss."""
    y, sc = _mc()
    mod._MULTICLASS_LOGLOSS_OOB_WARN_SEEN.clear()
    clean = _multiclass_metrics(y, sc, 3, 10)
    mod._MULTICLASS_LOGLOSS_OOB_WARN_SEEN.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dirty = _multiclass_metrics(np.append(y, 99), np.vstack([sc, [0.1, 0.1, 0.8]]), 3, 10)
    for key, val in clean.items():
        assert dirty[key] == val or (np.isnan(val) and np.isnan(dirty[key])), key


def test_multiclass_nan_label_is_dropped():
    """A NaN label is excluded like an out-of-range one."""
    y, sc = _mc()
    mod._MULTICLASS_LOGLOSS_OOB_WARN_SEEN.clear()
    clean = _multiclass_metrics(y, sc, 3, 10)
    mod._MULTICLASS_LOGLOSS_OOB_WARN_SEEN.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dirty = _multiclass_metrics(np.append(y.astype(float), np.nan), np.vstack([sc, [0.1, 0.1, 0.8]]), 3, 10)
    assert dirty["accuracy"] == clean["accuracy"] and dirty["MCC"] == clean["MCC"]


def test_n_classes_mismatch_follows_the_score_columns():
    """n_classes=3 with four score columns: predictions are argmaxed over four columns, so the class count is four and column-3 predictions are scored, not left outside [0, k)."""
    y, sc = _mc(k=4)
    out = _multiclass_metrics(y, sc, 3, 10)
    ref = _multiclass_metrics(y, sc, 4, 10)
    assert out["accuracy"] == ref["accuracy"] and out["MCC"] == ref["MCC"]
