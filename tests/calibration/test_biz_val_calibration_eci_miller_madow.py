"""biz_value: Miller-Madow correction on the Entropy-Based Calibration Index (ECI).

Ground truth: a perfectly-calibrated model produces a UNIFORM PIT distribution, true entropy log(bins), true ECI exactly 0. The plug-in entropy estimator is negatively biased at finite n, so the legacy plug-in ECI is positively biased (>0) -- spuriously reports miscalibration on a calibrated model. Miller-Madow (default) cancels the leading 1/N bias and pulls ECI back toward 0.

Measured at n=500 (bench bench_eci_miller_madow.py): MM closer to true 0 in 12/14 cells across bins=10/20 x 7 seeds; mean |ECI| 0.0132 (plugin) -> 0.0039 (MM). Floors below are set ~15-25% below the measured aggregate to absorb seed noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.calibration.quality import entropy_calibration_index


@pytest.mark.parametrize("bins", [10, 20])
def test_biz_val_eci_miller_madow_closer_to_zero_on_calibrated(bins):
    """On uniform (perfectly-calibrated) PIT, MM-corrected ECI is closer to the true 0 than plug-in on the majority of seeds."""
    n = 500
    seeds = range(7)
    mm_wins = 0
    plugin_errs, mm_errs = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pit = rng.uniform(0.0, 1.0, size=n)
        e_plugin = abs(entropy_calibration_index(pit, bins=bins, miller_madow=False))
        e_mm = abs(entropy_calibration_index(pit, bins=bins, miller_madow=True))
        plugin_errs.append(e_plugin)
        mm_errs.append(e_mm)
        if e_mm < e_plugin:
            mm_wins += 1
    assert mm_wins >= 5, f"MM should beat plug-in on majority of 7 seeds (bins={bins}); got {mm_wins}"
    # Aggregate: measured ~3.4x reduction; require at least ~2x closer to 0.
    assert np.mean(mm_errs) <= 0.55 * np.mean(plugin_errs), (
        f"MM mean |ECI| {np.mean(mm_errs):.5f} should be <= 0.55x plug-in {np.mean(plugin_errs):.5f} (bins={bins})"
    )


def test_biz_val_eci_default_is_miller_madow():
    """The corrective mechanism is ON by default: default call equals miller_madow=True, differs from legacy plug-in."""
    rng = np.random.default_rng(0)
    pit = rng.uniform(0.0, 1.0, size=500)
    default = entropy_calibration_index(pit, bins=10)
    mm = entropy_calibration_index(pit, bins=10, miller_madow=True)
    plugin = entropy_calibration_index(pit, bins=10, miller_madow=False)
    assert default == mm
    assert default != plugin
    # Correction is positive (raises entropy estimate => lowers ECI) on a non-degenerate histogram.
    assert default < plugin


def test_biz_val_eci_correction_magnitude_matches_formula():
    """MM correction equals (K_obs - 1)/(2N): plug-in ECI minus MM ECI must match the closed form exactly.

    Uses a MISCALIBRATED (skewed) PIT so both ECI values sit comfortably above the index's ``max(eci, 0)`` floor.
    On a near-uniform PIT the MM-corrected ECI underflows that floor and the clamp turns ``mm`` into 0, which
    legitimately breaks the raw ``plugin - mm`` identity -- the clamp is the correct calibration-index behaviour
    (ECI cannot be negative), so the identity is only meaningful where neither side is clamped.
    """
    rng = np.random.default_rng(3)
    pit = rng.beta(2.0, 5.0, size=2000)
    bins = 10
    counts, _ = np.histogram(pit, bins=bins, range=(0, 1))
    k_obs = int(np.count_nonzero(counts))
    expected_delta = (k_obs - 1) / (2.0 * counts.sum())
    plugin = entropy_calibration_index(pit, bins=bins, miller_madow=False)
    mm = entropy_calibration_index(pit, bins=bins, miller_madow=True)
    # Both ECI values above the 0 floor => neither is clamped => the closed-form correction identity holds exactly.
    assert plugin > expected_delta and mm > 0.0, f"input not miscalibrated enough to clear the ECI clamp; plugin={plugin}, mm={mm}"
    assert plugin - mm == pytest.approx(expected_delta, rel=1e-9)
