"""Regression tests for the training-log audit 2026-09-20, discovery group.

- ``DSC-02`` the honest-RMSE gate discarded the raw baseline's prediction vector, so no paired significance test on
  a spec's gain was possible downstream.
- ``DSC-01`` / ``DSC-03`` ``min_honest_gain_to_train`` was a constant with no relation to the noise of the gain it
  filtered; a production run shipped 9 specs at gains of +0.002..+0.011 and warned about GPU non-determinism across
  those same extra fits in the very next log line.
- ``DSC-04`` a zero-inflated target was routed to log / cbrt y-compressors whose convex inverse cannot reconstruct
  spread, and the collapse was only caught after full fits (2.6 and 5.6 minutes of GPU).
"""
from __future__ import annotations

import numpy as np

from mlframe.training._composite_target_discovery_config_base import CompositeTargetDiscoveryConfigBase
from mlframe.training.composite.discovery._honest_rmse_gate import _paired_rmse_gain_se
from mlframe.training.composite.discovery._point_mass_gate import (
    CURVED_Y_COMPRESSORS,
    POINT_MASS_FRACTION_THRESHOLD,
    point_mass_curved_inverse_skips,
    point_mass_fraction,
)
from mlframe.training.composite.spec import CompositeSpec

N = 20_000


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _paired_predictions(n=N, spec_edge=0.0, seed=0):
    """``(y, y_hat_raw, y_hat_spec)`` where the spec shrinks the shared error by ``spec_edge``."""
    rng = np.random.default_rng(seed)
    y = rng.normal(100.0, 20.0, n)
    err = rng.normal(0.0, 10.0, n)
    return y, y + err, y + err * (1.0 - spec_edge)


# --------------------------------------------------------------------------------------------------
# DSC-02: the paired standard error of the gain
# --------------------------------------------------------------------------------------------------


def test_paired_gain_se_makes_a_real_edge_significant():
    """A spec that genuinely cuts the error must clear a 2-sigma bar on the paired statistic."""
    y, raw, spec = _paired_predictions(spec_edge=0.05)
    se = _paired_rmse_gain_se(y, raw, spec, _rmse(y, raw), _rmse(y, spec))
    gain = _rmse(y, raw) - _rmse(y, spec)
    assert np.isfinite(se) and se > 0
    assert gain > 2.0 * se, (gain, se)


def test_paired_gain_se_keeps_a_noise_edge_below_the_bar():
    """Two models with the same expected error must NOT clear it, which is what the old constant floor could not tell."""
    rng = np.random.default_rng(1)
    y, raw, _spec = _paired_predictions(seed=1)
    spec = y + rng.normal(0.0, 10.0, y.size)  # independent error of identical scale
    se = _paired_rmse_gain_se(y, raw, spec, _rmse(y, raw), _rmse(y, spec))
    gain = _rmse(y, raw) - _rmse(y, spec)
    assert np.isfinite(se) and se > 0
    assert abs(gain) < 2.0 * se, (gain, se)


def test_paired_gain_se_is_undefined_on_too_few_rows():
    y, raw, spec = _paired_predictions(n=20)
    assert not np.isfinite(_paired_rmse_gain_se(y, raw, spec, _rmse(y, raw), _rmse(y, spec)))


def test_paired_gain_se_is_undefined_when_both_models_are_perfect():
    y = np.linspace(0.0, 1.0, N)
    assert not np.isfinite(_paired_rmse_gain_se(y, y, y, 0.0, 0.0))


def test_spec_carries_the_gain_standard_error_field():
    """The field must exist on the frozen spec, or the downstream floor has nothing to read."""
    assert "honest_holdout_rmse_gain_se" in CompositeSpec.__dataclass_fields__
    assert CompositeSpec.__dataclass_fields__["honest_holdout_rmse_gain_se"].default is None


def test_config_exposes_the_noise_aware_floor_multiplier():
    cfg = CompositeTargetDiscoveryConfigBase()
    assert cfg.min_honest_gain_z == 2.0
    assert cfg.min_honest_gain_to_train == 0.001


# --------------------------------------------------------------------------------------------------
# DSC-04: point-mass targets vs curved y-compressors
# --------------------------------------------------------------------------------------------------


def _zero_inflated(zero_frac: float, n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < zero_frac, 0.0, rng.pareto(1.2, n) * 60.0)


def test_zero_inflated_target_skips_the_curved_compressors():
    """The production shape: ``target_total_charge`` with median 0 and ~71% of rows exactly zero."""
    y = _zero_inflated(0.71)
    assert point_mass_fraction(y) > POINT_MASS_FRACTION_THRESHOLD
    assert point_mass_curved_inverse_skips(y) == CURVED_Y_COMPRESSORS


def test_the_skip_list_covers_the_two_transforms_that_collapsed_in_production():
    assert {"log_y", "cbrt_y"} <= CURVED_Y_COMPRESSORS


def test_clipping_transforms_are_not_skipped():
    """``y_quantile_clip`` keeps a piecewise-linear inverse and was the one production composite that beat raw y."""
    assert "y_quantile_clip" not in CURVED_Y_COMPRESSORS


def test_a_continuous_target_is_untouched():
    rng = np.random.default_rng(3)
    y = rng.gamma(2.0, 50.0, 100_000)
    assert point_mass_fraction(y) < 0.01
    assert point_mass_curved_inverse_skips(y) == frozenset()


def test_a_discrete_target_without_a_dominant_mass_is_untouched():
    """Discreteness alone is not the problem -- concentration is."""
    rng = np.random.default_rng(4)
    y = rng.integers(0, 5, 100_000).astype(float)
    assert point_mass_fraction(y) < POINT_MASS_FRACTION_THRESHOLD
    assert point_mass_curved_inverse_skips(y) == frozenset()


def test_the_threshold_is_the_contract():
    below = point_mass_curved_inverse_skips(_zero_inflated(POINT_MASS_FRACTION_THRESHOLD - 0.1, seed=5))
    above = point_mass_curved_inverse_skips(_zero_inflated(POINT_MASS_FRACTION_THRESHOLD + 0.1, seed=5))
    assert below == frozenset()
    assert above == CURVED_Y_COMPRESSORS


def test_point_mass_fraction_is_undecidable_on_a_tiny_sample():
    assert point_mass_fraction(np.zeros(10)) == 0.0
