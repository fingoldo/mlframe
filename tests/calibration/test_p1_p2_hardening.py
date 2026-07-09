"""Regression tests for the calibration/post.py P1/P2 audit findings (multi-class/degenerate guard,
hardcoded metric-column KeyError, metric-key-drift warning, full_name collision handling).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def _synth_binary(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    p1 = rng.random(n)
    probs = np.column_stack([1 - p1, p1])
    target = (p1 + rng.normal(0, 0.1, n) > 0.5).astype(int)
    return probs, target


def test_multiclass_calib_target_raises_instead_of_silently_misfitting():
    """P1-1: BinaryPostCalibrator/get_postcalibrators is binary-only; a 3-class calib_target must
    raise a clear error rather than silently producing garbage via calibrators that don't validate y."""
    from mlframe.calibration.post import compare_postcalibrators, named_calibrator
    from sklearn.isotonic import IsotonicRegression

    n = 300
    rng = np.random.default_rng(0)
    p1 = rng.random(n)
    probs = np.column_stack([1 - p1, p1])
    target = rng.integers(0, 3, size=n)  # 3 classes

    fake_calibrators = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]
    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        with pytest.raises(ValueError, match="exactly 2 distinct classes"):
            compare_postcalibrators(
                model_name="m", columns=["y"], calib_probs=probs, calib_target=target,
                oos_probs=None, oos_target=None, include_patterns=["sklearn"],
            )


def test_degenerate_single_class_calib_target_raises_clear_error():
    """P1-2: a single-class calib_target must raise our own clear diagnostic, not a third-party
    stack trace deep inside sklearn/netcal fit()."""
    from mlframe.calibration.post import compare_postcalibrators, named_calibrator
    from sklearn.isotonic import IsotonicRegression

    n = 200
    rng = np.random.default_rng(0)
    p1 = rng.random(n)
    probs = np.column_stack([1 - p1, p1])
    target = np.zeros(n, dtype=int)  # single class

    fake_calibrators = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]
    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        with pytest.raises(ValueError, match="exactly 2 distinct classes"):
            compare_postcalibrators(
                model_name="m", columns=["y"], calib_probs=probs, calib_target=target,
                oos_probs=None, oos_target=None, include_patterns=["sklearn"],
            )


def test_missing_ice_column_does_not_raise_keyerror():
    """P1-3: if report_model_perf's metric dict ever omits 'ice' or 'feature_importances', the final
    formatting step must not KeyError and discard every already-fitted calibrator's results."""
    from mlframe.calibration.post import compare_postcalibrators, named_calibrator
    from sklearn.isotonic import IsotonicRegression

    probs, target = _synth_binary()
    fake_calibrators = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]

    def _fake_report_model_perf(*args, metrics=None, **kwargs):
        # The real report_model_perf stores the per-class metric dict at metrics[1] (class_id 1);
        # compare_postcalibrators's PERF_DICT_COL=1 flattens that. No "ice"/"feature_importances" here.
        if metrics is not None:
            metrics[1] = {"auc": 0.9}
        return None, None

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators), patch(
        "mlframe.calibration.post.report_model_perf", side_effect=_fake_report_model_perf
    ):
        metrics_df, calibrators, failed = compare_postcalibrators(
            model_name="m", columns=["y"], calib_probs=probs, calib_target=target,
            oos_probs=None, oos_target=None, include_patterns=["sklearn"], selection="self_eval",
        )

    assert metrics_df is not None
    assert "auc" in metrics_df.columns
    assert "ice" not in metrics_df.columns
    assert calibrators, "the fitted calibrator must still be returned despite the missing 'ice' column"


def test_metric_key_mismatch_logs_warning(caplog):
    """P1-4: when one calibrator's metric dict has a narrower key set than the rest (union-joined,
    NaN-filled), a warning must be logged rather than silently ranking it via NaN sort placement."""
    from mlframe.calibration.post import compare_postcalibrators, named_calibrator
    from sklearn.isotonic import IsotonicRegression

    probs, target = _synth_binary()
    fake_calibrators = [
        named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso1", lib="sklearn"),
        named_calibrator(IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0), name="Iso2", lib="sklearn"),
    ]

    def _fake_report_model_perf(*args, model_name="", metrics=None, **kwargs):
        if metrics is None:
            return None, None
        if "Iso2" in model_name:
            metrics[1] = {"ice": 0.1}  # narrower key set: no "auc"
        else:
            metrics[1] = {"ice": 0.1, "auc": 0.9}
        return None, None

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators), patch(
        "mlframe.calibration.post.report_model_perf", side_effect=_fake_report_model_perf
    ), caplog.at_level("WARNING", logger="mlframe.calibration.post"):
        metrics_df, calibrators, failed = compare_postcalibrators(
            model_name="m", columns=["y"], calib_probs=probs, calib_target=target,
            oos_probs=None, oos_target=None, include_patterns=["sklearn"], selection="self_eval",
        )

    assert any("narrower metric-key set" in rec.message for rec in caplog.records)


def test_full_name_collision_disambiguated_not_overwritten():
    """P1-5: two zoo entries resolving to the same full_name() must not silently overwrite one
    another in fit_calibrators/metrics; the second gets a disambiguating suffix and a warning."""
    from mlframe.calibration.post import compare_postcalibrators, named_calibrator
    from sklearn.isotonic import IsotonicRegression

    probs, target = _synth_binary()
    # Two DIFFERENT calibrator instances that both resolve to the exact same full_name().
    fake_calibrators = [
        named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Dup", lib="x"),
        named_calibrator(IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0), name="Dup", lib="x"),
    ]

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        metrics_df, calibrators, failed = compare_postcalibrators(
            model_name="m", columns=["y"], calib_probs=probs, calib_target=target,
            oos_probs=None, oos_target=None, include_patterns=["x"], selection="self_eval",
        )

    assert len(calibrators) == 2, "both calibrators must survive despite the name collision"
    assert "x.Dup" in calibrators
    assert any(name.startswith("x.Dup#") for name in calibrators), "the colliding entry must get a disambiguated key"
