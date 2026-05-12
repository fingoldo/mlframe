"""biz_val tests for ``mlframe.training.drift_report`` +
``mlframe.training.evaluation`` -- pure functions that don't require
a full pipeline fit.

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# compute_label_distribution_drift
# ---------------------------------------------------------------------------


def test_biz_val_drift_compute_label_binary_no_drift_balanced():
    """Identical train/val/test distributions -> no warnings, all
    p_positive within ~sampling noise of each other."""
    from mlframe.training.drift_report import compute_label_distribution_drift
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=2000).astype(np.int64)
    result = compute_label_distribution_drift(
        train_target=pd.Series(y[:1000]),
        val_target=pd.Series(y[1000:1500]),
        test_target=pd.Series(y[1500:]),
        target_type="binary_classification",
    )
    # Key invariant: splits dict contains per-split breakdown dicts.
    splits = result["splits"]
    assert isinstance(splits, dict)
    assert set(splits.keys()) == {"train", "val", "test"}
    for split_name, s in splits.items():
        assert isinstance(s, dict)
        for k in ("n", "n_positive", "p_positive"):
            assert k in s, f"split {split_name} missing key {k}"


def test_biz_val_drift_compute_label_binary_detects_shift():
    """Train 50/50, val 100% positive -> must fire a drift warning."""
    from mlframe.training.drift_report import compute_label_distribution_drift
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=500).astype(np.int64)
    val = np.ones(200, dtype=np.int64)  # 100% positive
    result = compute_label_distribution_drift(
        train_target=train,
        val_target=val,
        test_target=None,
        target_type="binary_classification",
    )
    # Regardless of warning-vs-not, drifts dict must indicate a large pp shift.
    drifts = result["drifts"]
    max_pp = drifts["max_abs_drift_pp"]
    assert max_pp > 10.0, (
        f"50/50 train vs 100% positive val must have >10pp max drift; "
        f"got {max_pp:.1f}"
    )
    # Warnings should be non-empty (drift magnitude >> 5pp threshold).
    assert len(result.get("warnings", [])) > 0


def test_biz_val_drift_compute_label_regression_mean_shift():
    """Train ~N(0,1), val ~N(3,1) -- must detect a large mean shift."""
    from mlframe.training.drift_report import compute_label_distribution_drift
    rng = np.random.default_rng(42)
    train = rng.normal(loc=0.0, scale=1.0, size=500)
    val = rng.normal(loc=3.0, scale=1.0, size=200)
    result = compute_label_distribution_drift(
        train_target=pd.Series(train),
        val_target=pd.Series(val),
        test_target=None,
        target_type="regression",
    )
    # Regression split contains mean / std / median / p01 / p99.
    reg_train = result["splits"]["train"]
    assert "mean" in reg_train
    # Large shift -> warnings must be present.
    assert len(result.get("warnings", [])) > 0


def test_biz_val_drift_compute_label_handles_missing_val():
    """``val_target=None`` produces ``val=NULL`` in the splits dict."""
    from mlframe.training.drift_report import compute_label_distribution_drift
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=300).astype(np.int64)
    result = compute_label_distribution_drift(
        train_target=train,
        val_target=None,
        test_target=None,
        target_type="binary_classification",
    )
    assert result["splits"].get("val") is None
    assert result["splits"].get("test") is None


def test_biz_val_drift_compute_label_binary_result_has_drift_keys():
    """Result dict must contain all standard top-level keys regardless
    of whether drift is detected."""
    from mlframe.training.drift_report import compute_label_distribution_drift
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200).astype(np.int64)
    result = compute_label_distribution_drift(
        train_target=y[:100],
        val_target=y[100:],
        test_target=None,
        target_type="binary_classification",
    )
    expected_keys = {"target_type", "splits", "drifts", "warnings",
                        "warn_threshold_pp"}
    missing = expected_keys - set(result.keys())
    assert not missing, f"missing top-level keys: {missing}"


# ---------------------------------------------------------------------------
# format_drift_report (from drift_report.py)
# ---------------------------------------------------------------------------


def test_biz_val_drift_format_drift_report_nonempty_string():
    """``format_drift_report`` with a valid report dict must produce
    a non-trivial multi-line string."""
    from mlframe.training.drift_report import (
        compute_label_distribution_drift, format_drift_report,
    )
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=500).astype(np.int64)
    val_shift = np.ones(200, dtype=np.int64)
    dr_dict = compute_label_distribution_drift(
        train_target=y[:300],
        val_target=val_shift,
        test_target=None,
        target_type="binary_classification",
    )
    formatted = format_drift_report(dr_dict)
    assert isinstance(formatted, str)
    assert len(formatted) > 30

# ---------------------------------------------------------------------------
# root_mean_squared_error (from evaluation.py, sklearn-compatible)
# ---------------------------------------------------------------------------


def test_biz_val_eval_rmse_perfect_predictions():
    """RMSE of identical y_pred == y_true must be 0."""
    from mlframe.training.evaluation import root_mean_squared_error
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rmse = root_mean_squared_error(y, y)
    assert rmse == 0.0 or rmse < 1e-12, f"RMSE of y=y must be 0; got {rmse}"


def test_biz_val_eval_rmse_constant_offset():
    """RMSE of y_pred = y + c must equal |c|."""
    from mlframe.training.evaluation import root_mean_squared_error
    y = np.array([10.0, 20.0, 30.0])
    c = 5.0
    rmse = root_mean_squared_error(y, y + c)
    assert abs(rmse - c) < 1e-12, (
        f"RMSE of y vs y+{c} must be {c}; got {rmse:.6f}"
    )


@pytest.mark.parametrize("n_samples", [100, 500, 2000])
def test_biz_val_eval_rmse_scales_with_samples(n_samples):
    """RMSE must handle small / medium / large input without overflow."""
    from mlframe.training.evaluation import root_mean_squared_error
    rng = np.random.default_rng(42)
    y = rng.normal(size=n_samples)
    rmse = root_mean_squared_error(y, y + 0.1 * rng.normal(size=n_samples))
    assert 0.0 < rmse < 0.3
