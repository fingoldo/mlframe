"""Tests for label_distribution_drift module.

Covers:
- binary classification: detection of prior-shift (the user's core use case)
- multiclass: per-class rate drift
- multilabel: per-label drift
- regression: mean drift in train-sigma units
- pandas / polars / numpy input compatibility
- val_target=None / test_target=None
- threshold tuning
- format_drift_report rendering
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.drift_report import (
    DEFAULT_BINARY_DRIFT_WARN_THRESHOLD_PP,
    compute_label_distribution_drift,
    format_drift_report,
)


# -----------------------------------------------------------------------------
# Binary classification — the user's main use case
# -----------------------------------------------------------------------------


def test_binary_no_drift_no_warnings():
    rng = np.random.default_rng(0)
    train = (rng.uniform(size=10_000) < 0.30).astype(np.int8)
    val = (rng.uniform(size=2_000) < 0.30).astype(np.int8)
    test = (rng.uniform(size=2_000) < 0.30).astype(np.int8)

    report = compute_label_distribution_drift(train, val, test, "binary_classification")

    assert report["target_type"] == "binary_classification"
    assert abs(report["splits"]["train"]["p_positive"] - 0.30) < 0.02
    # ~0% drift for an iid sample
    assert report["drifts"]["max_abs_drift_pp"] < 3.0
    assert report["warnings"] == []


def test_binary_severe_drift_warns():
    """User's exact scenario: train P(y=1)=0.30, val/test=0.80 — selection bias."""
    rng = np.random.default_rng(42)
    train = (rng.uniform(size=10_000) < 0.30).astype(np.int8)
    val = (rng.uniform(size=2_000) < 0.80).astype(np.int8)
    test = (rng.uniform(size=2_000) < 0.80).astype(np.int8)

    report = compute_label_distribution_drift(train, val, test, "binary_classification")

    # Drift well above threshold
    assert report["drifts"]["max_abs_drift_pp"] > 40.0
    # Both val and test trigger warns
    assert len(report["warnings"]) == 2
    assert any("VAL" in w for w in report["warnings"])
    assert any("TEST" in w for w in report["warnings"])
    assert any("selection-bias" in w or "prior-shift" in w
               for w in report["warnings"])


def test_binary_user_production_pattern():
    """Mirror the user's production training log: train P=0.74, val=0.86, test=0.83.

    Should fire 2 warnings (val ~+12pp, test ~+9pp) — both above the 5pp default.
    """
    rng = np.random.default_rng(0)
    train = (rng.uniform(size=100_000) < 0.74).astype(np.int8)
    val = (rng.uniform(size=10_000) < 0.86).astype(np.int8)
    test = (rng.uniform(size=10_000) < 0.83).astype(np.int8)

    report = compute_label_distribution_drift(train, val, test, "binary_classification")

    val_pp = report["drifts"]["val_minus_train_pp"]
    test_pp = report["drifts"]["test_minus_train_pp"]
    assert 10.0 < val_pp < 14.0, f"VAL drift {val_pp} outside expected ~+12pp"
    assert 7.0 < test_pp < 11.0, f"TEST drift {test_pp} outside expected ~+9pp"
    assert len(report["warnings"]) == 2


def test_binary_below_threshold_no_warn():
    rng = np.random.default_rng(0)
    train = (rng.uniform(size=10_000) < 0.50).astype(np.int8)
    # +3pp drift, under default 5pp threshold
    val = (rng.uniform(size=2_000) < 0.53).astype(np.int8)
    test = (rng.uniform(size=2_000) < 0.52).astype(np.int8)

    report = compute_label_distribution_drift(train, val, test, "binary_classification")
    assert report["warnings"] == []


def test_binary_threshold_tunable():
    """Drift of +3pp: silent at default 5pp, fires at custom 1pp threshold."""
    rng = np.random.default_rng(0)
    train = (rng.uniform(size=20_000) < 0.50).astype(np.int8)
    val = (rng.uniform(size=10_000) < 0.535).astype(np.int8)

    r_default = compute_label_distribution_drift(
        train, val, None, "binary_classification",
    )
    assert r_default["warnings"] == []

    r_strict = compute_label_distribution_drift(
        train, val, None, "binary_classification",
        warn_threshold_pp=1.0,
    )
    assert len(r_strict["warnings"]) == 1


def test_binary_only_train_no_other_splits():
    train = np.array([0, 1, 1, 0, 1])
    report = compute_label_distribution_drift(
        train, None, None, "binary_classification",
    )
    assert report["splits"]["train"]["n"] == 5
    assert report["splits"]["val"] is None
    assert report["splits"]["test"] is None


def test_binary_pandas_series_input():
    train = pd.Series(np.array([0, 1, 1, 0, 1, 0, 0, 0]))
    val = pd.Series(np.array([1, 1, 1, 1, 1]))  # 100% positive
    report = compute_label_distribution_drift(train, val, None, "binary_classification")
    assert report["splits"]["train"]["p_positive"] == 0.375
    assert report["splits"]["val"]["p_positive"] == 1.0
    # Drift +62.5pp, fires warn
    assert len(report["warnings"]) == 1


def test_binary_polars_series_input():
    pl = pytest.importorskip("polars")
    train = pl.Series([0, 1, 1, 0, 1, 0, 0, 0])
    val = pl.Series([1, 1, 1, 1, 1])
    report = compute_label_distribution_drift(train, val, None, "binary_classification")
    assert report["splits"]["train"]["p_positive"] == 0.375
    assert report["splits"]["val"]["p_positive"] == 1.0


# -----------------------------------------------------------------------------
# Multiclass classification
# -----------------------------------------------------------------------------


def test_multiclass_no_drift():
    rng = np.random.default_rng(0)
    train = rng.choice([0, 1, 2], size=10_000, p=[0.5, 0.3, 0.2])
    val = rng.choice([0, 1, 2], size=2_000, p=[0.5, 0.3, 0.2])
    report = compute_label_distribution_drift(train, val, None, "multiclass_classification")
    assert report["target_type"] == "multiclass_classification"
    assert set(report["splits"]["train"]["counts"].keys()) == {0, 1, 2}
    assert report["warnings"] == []


def test_multiclass_class_prior_shift_warns():
    rng = np.random.default_rng(0)
    train = rng.choice([0, 1, 2], size=10_000, p=[0.7, 0.2, 0.1])
    # val/test concentrate on the rare class — should fire warn for class 2
    val = rng.choice([0, 1, 2], size=2_000, p=[0.4, 0.2, 0.4])
    report = compute_label_distribution_drift(train, val, None, "multiclass_classification")
    assert len(report["warnings"]) >= 1
    assert any("class 2" in w or "P(y=2)" in w for w in report["warnings"])


# -----------------------------------------------------------------------------
# Multilabel classification
# -----------------------------------------------------------------------------


def test_multilabel_no_drift():
    rng = np.random.default_rng(0)
    train = (rng.uniform(size=(10_000, 3)) < [0.3, 0.5, 0.7]).astype(np.int8)
    val = (rng.uniform(size=(2_000, 3)) < [0.3, 0.5, 0.7]).astype(np.int8)
    report = compute_label_distribution_drift(train, val, None, "multilabel_classification")
    assert report["splits"]["train"]["n_labels"] == 3
    assert report["warnings"] == []


def test_multilabel_per_label_drift_warns():
    rng = np.random.default_rng(0)
    train = (rng.uniform(size=(10_000, 3)) < [0.3, 0.5, 0.7]).astype(np.int8)
    # Label 0 shifts from 0.3 → 0.5 (+20pp), label 2 stays
    val = (rng.uniform(size=(2_000, 3)) < [0.5, 0.5, 0.7]).astype(np.int8)
    report = compute_label_distribution_drift(train, val, None, "multilabel_classification")
    deltas = report["drifts"]["val_minus_train_pp_per_label"]
    assert deltas[0] > 15.0  # label 0 shift
    assert abs(deltas[1]) < 5.0
    assert abs(deltas[2]) < 5.0
    assert any("label 0" in w for w in report["warnings"])


# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------


def test_regression_no_drift():
    rng = np.random.default_rng(0)
    train = rng.normal(loc=10.0, scale=2.0, size=10_000)
    val = rng.normal(loc=10.0, scale=2.0, size=2_000)
    report = compute_label_distribution_drift(train, val, None, "regression")
    assert report["target_type"] == "regression"
    assert abs(report["splits"]["train"]["mean"] - 10.0) < 0.1
    assert report["warnings"] == []


def test_regression_mean_drift_warns():
    rng = np.random.default_rng(0)
    train = rng.normal(loc=10.0, scale=2.0, size=10_000)
    # val mean shifted by 2σ — fires the 0.5σ default
    val = rng.normal(loc=14.0, scale=2.0, size=2_000)
    report = compute_label_distribution_drift(train, val, None, "regression")
    assert report["drifts"]["val_mean_z_vs_train"] > 0.5
    assert len(report["warnings"]) == 1
    assert "regression target shift" in report["warnings"][0]


# -----------------------------------------------------------------------------
# format_drift_report rendering
# -----------------------------------------------------------------------------


def test_format_binary_includes_p_positive_and_warns():
    rng = np.random.default_rng(0)
    train = (rng.uniform(size=1_000) < 0.30).astype(np.int8)
    val = (rng.uniform(size=200) < 0.80).astype(np.int8)
    report = compute_label_distribution_drift(train, val, None, "binary_classification")
    rendered = format_drift_report(report, target_name="cl_act_total_hired_above_1")

    assert "label_distribution_drift report" in rendered
    assert "binary_classification" in rendered
    assert "cl_act_total_hired_above_1" in rendered
    assert "P(y=1)" in rendered
    assert "WARN" in rendered  # has warning


def test_format_no_warnings_says_so():
    rng = np.random.default_rng(0)
    train = (rng.uniform(size=10_000) < 0.30).astype(np.int8)
    val = (rng.uniform(size=2_000) < 0.30).astype(np.int8)
    report = compute_label_distribution_drift(train, val, None, "binary_classification")
    rendered = format_drift_report(report)
    assert "no drift warnings" in rendered


def test_format_regression():
    rng = np.random.default_rng(0)
    train = rng.normal(loc=10.0, scale=2.0, size=1000)
    val = rng.normal(loc=10.5, scale=2.0, size=200)
    report = compute_label_distribution_drift(train, val, None, "regression")
    rendered = format_drift_report(report)
    assert "regression" in rendered
    assert "mean=" in rendered
    assert "std=" in rendered


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


def test_train_target_none_returns_safe_report():
    report = compute_label_distribution_drift(None, None, None, "binary_classification")
    assert report["splits"] == {}
    assert "train_target is None" in report["warnings"][0]


def test_default_threshold_is_5pp():
    assert DEFAULT_BINARY_DRIFT_WARN_THRESHOLD_PP == 5.0


def test_metadata_round_trip_format():
    """The report dict must be JSON-serialisable (joblib-friendly)."""
    import json
    rng = np.random.default_rng(0)
    train = (rng.uniform(size=1_000) < 0.30).astype(np.int8)
    report = compute_label_distribution_drift(train, None, None, "binary_classification")
    # Must serialise without error (no numpy scalars leaking through)
    s = json.dumps(report)
    parsed = json.loads(s)
    assert parsed["target_type"] == "binary_classification"
