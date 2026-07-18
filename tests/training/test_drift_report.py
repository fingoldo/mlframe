"""Unit tests for `training.drift_report.compute_label_distribution_drift`
and `format_drift_report`.

The pre-existing `tests/reporting/test_drift_report.py` exercises a different
reporting module. The public API here (binary / multiclass / multilabel /
regression branches + the format helper) had only indirect coverage. This
file pins each branch independently.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.drift_report import (
    DEFAULT_BINARY_DRIFT_WARN_THRESHOLD_PP,
    DEFAULT_REGRESSION_MEAN_Z_WARN_THRESHOLD,
    compute_label_distribution_drift,
    format_drift_report,
)

# ----- binary classification ---------------------------------------------


def test_binary_no_drift_when_splits_match():
    """Binary no drift when splits match."""
    rng = np.random.default_rng(0)
    train = rng.integers(0, 2, size=2000)
    val = rng.integers(0, 2, size=500)
    test = rng.integers(0, 2, size=500)
    out = compute_label_distribution_drift(train, val, test, target_type="binary_classification")
    assert out["target_type"] == "binary_classification"
    assert out["splits"]["train"]["n"] == 2000
    assert "p_positive" in out["splits"]["train"]
    # All p_positive within 5pp -> warnings list must be empty.
    assert out["warnings"] == []
    assert out["drifts"]["max_abs_drift_pp"] < DEFAULT_BINARY_DRIFT_WARN_THRESHOLD_PP


def test_binary_prior_shift_50_50_to_90_10_warns():
    """Binary prior shift 50 50 to 90 10 warns."""
    rng = np.random.default_rng(1)
    train = rng.integers(0, 2, size=4000)
    # Shifted val/test: 90% positive vs train's ~50%.
    val = (rng.random(size=1000) < 0.9).astype(int)
    test = (rng.random(size=1000) < 0.9).astype(int)
    out = compute_label_distribution_drift(train, val, test, target_type="binary_classification")
    train_p = out["splits"]["train"]["p_positive"]
    val_p = out["splits"]["val"]["p_positive"]
    test_p = out["splits"]["test"]["p_positive"]
    assert abs(val_p - train_p) > 0.30  # at least 30pp
    assert abs(test_p - train_p) > 0.30
    # Warning must fire for both val and test.
    assert any("VAL" in w for w in out["warnings"])
    assert any("TEST" in w for w in out["warnings"])
    # Drift score on shifted distribution must be at least 2x larger
    # than the no-shift case (where it was <5pp -> here >=30pp).
    assert out["drifts"]["max_abs_drift_pp"] >= 30.0


def test_binary_format_drift_report_contains_target_name_and_summary():
    """Binary format drift report contains target name and summary."""
    rng = np.random.default_rng(2)
    train = rng.integers(0, 2, size=100)
    val = rng.integers(0, 2, size=50)
    test = rng.integers(0, 2, size=50)
    rep = compute_label_distribution_drift(train, val, test, target_type="binary_classification")
    out = format_drift_report(rep, target_name="churn")
    assert isinstance(out, str)
    assert out  # non-empty
    assert "churn" in out
    assert "target_type=binary_classification" in out
    # Each split row must be rendered.
    assert "train" in out
    assert "val" in out
    assert "test" in out


# ----- regression --------------------------------------------------------


def test_regression_no_drift_within_threshold():
    """Regression no drift within threshold."""
    rng = np.random.default_rng(3)
    train = rng.normal(loc=0.0, scale=1.0, size=2000)
    val = rng.normal(loc=0.0, scale=1.0, size=500)
    test = rng.normal(loc=0.0, scale=1.0, size=500)
    out = compute_label_distribution_drift(train, val, test, target_type="regression")
    assert out["target_type"] == "regression"
    assert "mean" in out["splits"]["train"]
    assert "p01" in out["splits"]["train"]
    assert "p99" in out["splits"]["train"]
    # Same DGP -> |z| < 0.5 sigma -> no warnings.
    assert out["warnings"] == []


def test_regression_mean_shift_warns_when_above_threshold():
    """Regression mean shift warns when above threshold."""
    rng = np.random.default_rng(4)
    train = rng.normal(loc=0.0, scale=1.0, size=2000)
    # 5-sigma shifted val/test (way above the 0.5 sigma threshold).
    val = rng.normal(loc=5.0, scale=1.0, size=500)
    test = rng.normal(loc=5.0, scale=1.0, size=500)
    out = compute_label_distribution_drift(train, val, test, target_type="regression")
    # z-scores recorded.
    assert abs(out["drifts"]["val_mean_z_vs_train"]) >= DEFAULT_REGRESSION_MEAN_Z_WARN_THRESHOLD
    assert abs(out["drifts"]["test_mean_z_vs_train"]) >= DEFAULT_REGRESSION_MEAN_Z_WARN_THRESHOLD
    assert any("VAL" in w for w in out["warnings"])
    assert any("TEST" in w for w in out["warnings"])


# ----- multiclass --------------------------------------------------------


def test_multiclass_no_drift_when_class_priors_match():
    """Multiclass no drift when class priors match."""
    rng = np.random.default_rng(5)
    train = rng.integers(0, 4, size=4000)
    val = rng.integers(0, 4, size=1000)
    test = rng.integers(0, 4, size=1000)
    out = compute_label_distribution_drift(train, val, test, target_type="multiclass_classification")
    assert out["target_type"] == "multiclass_classification"
    assert "rates" in out["splits"]["train"]
    # All classes within 5pp -> no warnings on random uniform priors.
    assert out["warnings"] == []


def test_multiclass_warns_when_one_class_shifts():
    # Train: uniform over 3 classes. Val/test: heavily skewed to class 0.
    """Multiclass warns when one class shifts."""
    rng = np.random.default_rng(6)
    train = rng.integers(0, 3, size=3000)
    val = np.full(1000, 0)  # all class 0
    test = np.full(1000, 0)
    out = compute_label_distribution_drift(train, val, test, target_type="multiclass_classification")
    assert len(out["warnings"]) >= 1
    # Drift per-class dict has entries for every observed class.
    assert "val_minus_train_pp_per_class" in out["drifts"]
    assert "test_minus_train_pp_per_class" in out["drifts"]


# ----- multilabel --------------------------------------------------------


def test_multilabel_2d_ndarray_no_drift():
    """Multilabel 2d ndarray no drift."""
    rng = np.random.default_rng(7)
    train = (rng.random(size=(2000, 4)) < 0.3).astype(int)
    val = (rng.random(size=(500, 4)) < 0.3).astype(int)
    test = (rng.random(size=(500, 4)) < 0.3).astype(int)
    out = compute_label_distribution_drift(train, val, test, target_type="multilabel_classification")
    assert out["target_type"] == "multilabel_classification"
    assert len(out["splits"]["train"]["p_positive_per_label"]) == 4
    # Matched DGP -> empty warnings.
    assert out["warnings"] == []


def test_multilabel_warns_when_one_label_shifts():
    """Multilabel warns when one label shifts."""
    rng = np.random.default_rng(8)
    train = (rng.random(size=(2000, 3)) < 0.3).astype(int)
    val = train[:500].copy()
    val[:, 0] = 1  # force label 0 to 100% in val
    test = train[500:1000].copy()
    test[:, 0] = 1
    out = compute_label_distribution_drift(train, val, test, target_type="multilabel_classification")
    assert any("label 0" in w for w in out["warnings"])


# ----- edge cases --------------------------------------------------------


def test_train_target_none_returns_safe_report():
    """Train target none returns safe report."""
    out = compute_label_distribution_drift(None, np.array([0, 1]), np.array([0, 1]), target_type="binary_classification")
    assert out["splits"] == {}
    assert any("train_target is None" in w for w in out["warnings"])


def test_val_and_test_none_handled():
    """Val and test none handled."""
    rng = np.random.default_rng(9)
    train = rng.integers(0, 2, size=200)
    out = compute_label_distribution_drift(train, None, None, target_type="binary_classification")
    assert out["splits"]["train"]["n"] == 200
    assert out["splits"]["val"] is None
    assert out["splits"]["test"] is None


def test_empty_test_handled_without_crash():
    """Empty test handled without crash."""
    rng = np.random.default_rng(10)
    train = rng.integers(0, 2, size=200)
    val = rng.integers(0, 2, size=50)
    test = np.array([], dtype=int)
    out = compute_label_distribution_drift(train, val, test, target_type="binary_classification")
    # n=0 split summary: p_positive is NaN.
    assert out["splits"]["test"]["n"] == 0
    assert np.isnan(out["splits"]["test"]["p_positive"])


def test_format_drift_report_no_target_name():
    """Format drift report no target name."""
    rep = compute_label_distribution_drift(np.array([0, 1, 1]), None, None, target_type="binary_classification")
    out = format_drift_report(rep)
    assert isinstance(out, str)
    assert "target_type" in out
    # When no target_name passed, the "target=..." segment is absent.
    assert "target=" not in out


def test_format_drift_report_regression_branch_has_mean_std():
    """Format drift report regression branch has mean std."""
    rng = np.random.default_rng(11)
    rep = compute_label_distribution_drift(
        rng.normal(size=300),
        rng.normal(size=100),
        rng.normal(size=100),
        target_type="regression",
    )
    out = format_drift_report(rep, target_name="price")
    assert "mean=" in out
    assert "std=" in out
    assert "price" in out


def test_format_drift_report_emits_no_warn_marker_when_clean():
    """Format drift report emits no warn marker when clean."""
    rng = np.random.default_rng(12)
    rep = compute_label_distribution_drift(
        rng.integers(0, 2, size=1000),
        rng.integers(0, 2, size=200),
        rng.integers(0, 2, size=200),
        target_type="binary_classification",
    )
    out = format_drift_report(rep, target_name="x")
    if not rep["warnings"]:
        assert "(no drift warnings" in out


def test_multiclass_split_summary_single_pass_via_unique(monkeypatch):
    """``_multiclass_split_summary`` must count via one ``np.unique`` pass, not one ``(arr == c).sum()`` scan per class.

    Pre-fix code did K full equality scans and never called ``np.unique``; the spy below would record 0 calls and
    fail. Output identity is pinned alongside so the single-pass rewrite stays bit-identical to per-class counting.
    """
    import mlframe.training.drift_report as dr

    rng = np.random.default_rng(7)
    arr = rng.integers(0, 12, size=50_000)
    classes = list(np.unique(arr).tolist())

    calls = {"n": 0}
    real_unique = np.unique

    def _spy_unique(a, *args, **kwargs):
        """Spy unique."""
        calls["n"] += 1
        return real_unique(a, *args, **kwargs)

    monkeypatch.setattr(dr.np, "unique", _spy_unique)
    out = dr._multiclass_split_summary(arr, classes)

    assert calls["n"] >= 1, "single-pass path must use np.unique(return_counts) (pre-fix per-class scan did not)"

    expected_counts = {int(c): int((arr == c).sum()) for c in classes}
    assert out["counts"] == expected_counts
    assert out["n"] == arr.shape[0]
    assert sum(out["counts"].values()) == arr.shape[0]


def test_multiclass_split_summary_handles_missing_class():
    """A class present in ``classes`` but absent from ``arr`` reports count 0 (lookup miss path)."""
    arr = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    out = _ms = __import__("mlframe.training.drift_report", fromlist=["_multiclass_split_summary"])._multiclass_split_summary(arr, [0, 1, 2])
    assert out["counts"] == {0: 2, 1: 3, 2: 0}
    assert out["rates"][2] == 0.0
