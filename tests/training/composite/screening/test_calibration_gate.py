"""Unit + biz_value tests for the calibration-aware spec scorer.

Covers: penalty detects bias, penalty detects variance miscalibration, a
well-calibrated spec gets ~0 penalty, no-harm pass-through on empty residuals,
the default-disabled flag, and the biz_value ranking flip (calibrated spec A
outranks lucky-but-miscalibrated overfit spec B even when raw RMSE favours B).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._calibration_gate import (
    CALIBRATION_GATE_DEFAULT_ENABLED,
    CalibrationScore,
    calibration_adjusted_score,
    calibration_penalty,
)


def test_default_flag_is_no_harm_off():
    assert CALIBRATION_GATE_DEFAULT_ENABLED is False


def test_well_calibrated_spec_near_zero_penalty():
    rng = np.random.default_rng(0)
    infold = rng.normal(0.0, 1.0, size=4000)
    oof = rng.normal(0.0, 1.0, size=4000)  # unbiased + same spread
    penalty, bias, var_miscal = calibration_penalty(oof, infold)
    assert bias < 0.05, f"unbiased residuals should have tiny bias, got {bias}"
    assert var_miscal < 0.1, f"matched spread -> small var_miscal, got {var_miscal}"
    assert penalty < 0.15


def test_penalty_detects_bias():
    rng = np.random.default_rng(1)
    infold = rng.normal(0.0, 1.0, size=4000)
    unbiased = rng.normal(0.0, 1.0, size=4000)
    biased = unbiased + 1.5  # systematic offset -> mean residual ~1.5
    p_unb, b_unb, _ = calibration_penalty(unbiased, infold)
    p_bias, b_bias, _ = calibration_penalty(biased, infold)
    assert b_bias > 1.0, f"shifted residuals should show large bias, got {b_bias}"
    assert p_bias > p_unb + 1.0


def test_penalty_detects_variance_miscalibration():
    rng = np.random.default_rng(2)
    infold = rng.normal(0.0, 1.0, size=4000)  # tight in-fold (overfit)
    oof_wide = rng.normal(0.0, 4.0, size=4000)  # inflated OOF spread
    _, _, vm_wide = calibration_penalty(oof_wide, infold)
    # IQR ratio ~4 -> var_miscal ~3.
    assert vm_wide > 2.0, f"4x spread inflation should show large var_miscal, got {vm_wide}"
    # Matched spread is small for contrast.
    oof_matched = rng.normal(0.0, 1.0, size=4000)
    _, _, vm_matched = calibration_penalty(oof_matched, infold)
    assert vm_matched < 0.1
    assert vm_wide > vm_matched + 2.0


def test_empty_residuals_pass_gain_through_no_harm():
    res = calibration_adjusted_score(0.7, np.array([np.nan, np.nan]))
    assert isinstance(res, CalibrationScore)
    assert res.adjusted_score == 0.7
    assert np.isnan(res.calibration_penalty)


def test_var_miscal_skipped_without_infold():
    rng = np.random.default_rng(3)
    oof = rng.normal(0.0, 1.0, size=2000)
    penalty, bias, var_miscal = calibration_penalty(oof, None)
    assert np.isnan(var_miscal)  # no in-fold reference -> term skipped
    assert np.isfinite(penalty)
    assert penalty == pytest.approx(bias, rel=1e-9)


def test_biz_val_calibration_ranks_calibrated_above_overfit():
    """Biz_value: spec A generalises-and-calibrates; spec B overfits (lower
    in-sample/raw RMSE but biased + inflated OOF residuals). Raw RMSE favours B;
    the calibration-adjusted score MUST rank A above B.

    Measured: A penalty ~0.0, B penalty ~2.x (bias ~0.9 + var_miscal ~1.x). With
    raw gain favouring B by 0.10, the adjustment flips the order by a wide margin.
    """
    rng = np.random.default_rng(7)

    # Spec A: honest. OOF residuals unbiased, spread ~ in-fold spread.
    a_infold = rng.normal(0.0, 1.0, size=5000)
    a_oof = rng.normal(0.0, 1.0, size=5000)
    a_gain = 0.50  # weaker raw merit

    # Spec B: lucky overfit. Tight in-fold residuals (memorised train), but OOF
    # residuals are biased AND inflated -> classic overfit signature.
    b_infold = rng.normal(0.0, 0.5, size=5000)
    b_oof = rng.normal(0.9, 1.8, size=5000)
    b_gain = 0.60  # B looks BETTER on raw in-sample merit

    # Raw ranking would pick B (higher gain).
    assert b_gain > a_gain

    sa = calibration_adjusted_score(a_gain, a_oof, a_infold)
    sb = calibration_adjusted_score(b_gain, b_oof, b_infold)

    # A stays well-calibrated, B is heavily penalised.
    assert sa.calibration_penalty < 0.2, sa
    assert sb.calibration_penalty > 1.5, sb

    # The flip: calibration-adjusted score ranks A above B by a clear margin
    # despite raw RMSE favouring B.
    assert sa.adjusted_score > sb.adjusted_score + 0.5, (sa, sb)


def test_smoke_import_dotted_module():
    import importlib

    mod = importlib.import_module("mlframe.training.composite.discovery._calibration_gate")
    assert hasattr(mod, "calibration_adjusted_score")
    assert hasattr(mod, "calibration_penalty")
    assert hasattr(mod, "CalibrationScore")
