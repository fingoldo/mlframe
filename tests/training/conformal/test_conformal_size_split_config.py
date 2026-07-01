"""Unit tests for the `conformal_size` field on `TrainingSplitConfig`.

`conformal_size` is the second disjoint holdout reserved for conformal residuals of the
recalibrated shipped predictor (distinct from `calib_size`, which fits the recalibration map).
These tests pin the field's bounds and the extended sum-of-fractions validator.
"""
from __future__ import annotations

import pytest

from mlframe.training._preprocessing_configs import TrainingSplitConfig


def test_conformal_size_defaults_none_no_behavior_change():
    cfg = TrainingSplitConfig()
    assert cfg.conformal_size is None


def test_conformal_size_accepted_alongside_calib():
    cfg = TrainingSplitConfig(test_size=0.1, val_size=0.1, calib_size=0.05, conformal_size=0.05)
    assert cfg.conformal_size == 0.05
    assert cfg.calib_size == 0.05


def test_sum_of_fractions_validator_includes_conformal_size():
    # 0.5 + 0.3 + 0.15 + 0.1 = 1.05 > 1.0 -> must reject, and the message must name conformal_size.
    with pytest.raises(ValueError, match="conformal_size"):
        TrainingSplitConfig(test_size=0.5, val_size=0.3, calib_size=0.15, conformal_size=0.1)


def test_sum_of_fractions_validator_passes_at_boundary():
    cfg = TrainingSplitConfig(test_size=0.4, val_size=0.2, calib_size=0.2, conformal_size=0.2)
    assert cfg.test_size + cfg.val_size + cfg.calib_size + cfg.conformal_size == pytest.approx(1.0)


def test_conformal_size_rejects_out_of_range():
    with pytest.raises(ValueError):
        TrainingSplitConfig(conformal_size=1.0)
    with pytest.raises(ValueError):
        TrainingSplitConfig(conformal_size=-0.1)
