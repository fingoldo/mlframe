"""Unit tests for the `conformal_size` field on `TrainingSplitConfig`.

`conformal_size` is the second disjoint holdout reserved for conformal residuals of the
recalibrated shipped predictor (distinct from `calib_size`, which fits the recalibration map).
These tests pin the field's bounds and the extended sum-of-fractions validator.
"""

from __future__ import annotations

import pytest

from mlframe.training._preprocessing_configs import TrainingSplitConfig


def test_conformal_size_defaults_none_no_behavior_change():
    """Conformal size defaults none no behavior change."""
    cfg = TrainingSplitConfig()
    assert cfg.conformal_size is None


def test_conformal_size_is_refused_while_the_carve_is_unwired():
    """A non-zero `conformal_size` is REFUSED, because no production code carves the slice it promises.

    This previously asserted the field was accepted alongside `calib_size`. Accepting it was the defect: the
    split path never carves a conformal slice, so finalize scored residuals on the very calib slice the
    recalibration map was fitted on -- in-sample residuals, and prediction intervals narrower than they should
    be, with nothing raised. The config now fails closed until the carve is wired, and says how to proceed.
    """
    with pytest.raises(ValueError, match="not wired into the split path yet"):
        TrainingSplitConfig(test_size=0.1, val_size=0.1, calib_size=0.05, conformal_size=0.05)

    # calib_size alone is the documented regression-safe configuration and must still be accepted.
    cfg = TrainingSplitConfig(test_size=0.1, val_size=0.1, calib_size=0.05)
    assert cfg.calib_size == 0.05
    assert cfg.conformal_size is None

    # An explicit 0.0 means "no conformal slice" and is not the un-wired case, so it must pass too.
    assert TrainingSplitConfig(test_size=0.1, val_size=0.1, calib_size=0.05, conformal_size=0.0).conformal_size == 0.0


def test_sum_of_fractions_validator_includes_conformal_size():
    # 0.5 + 0.3 + 0.15 + 0.1 = 1.05 > 1.0 -> must reject, and the message must name conformal_size.
    """Sum of fractions validator includes conformal size."""
    # Matched on the SUM message specifically: the un-wired guard below it also names conformal_size, so a
    # bare "conformal_size" match would pass even if the sum check were removed entirely.
    with pytest.raises(ValueError, match=r"must be <= 1\.0"):
        TrainingSplitConfig(test_size=0.5, val_size=0.3, calib_size=0.15, conformal_size=0.1)


def test_sum_of_fractions_validator_passes_at_boundary():
    """Fractions summing to exactly 1.0 are accepted; the bound is `> 1.0`, not `>= 1.0`.

    The boundary is now reached without `conformal_size`, which is refused outright while its carve is
    un-wired -- reaching 1.0 through it would make this test pass or fail for that reason instead of the
    sum-validator boundary it is named for.
    """
    cfg = TrainingSplitConfig(test_size=0.4, val_size=0.2, calib_size=0.4)
    assert cfg.test_size + cfg.val_size + cfg.calib_size == pytest.approx(1.0)


def test_conformal_size_rejects_out_of_range():
    """Conformal size rejects out of range."""
    with pytest.raises(ValueError):
        TrainingSplitConfig(conformal_size=1.0)
    with pytest.raises(ValueError):
        TrainingSplitConfig(conformal_size=-0.1)
