"""PROP2: entropy_calibration_index (Miller-Madow default) must never go negative.

A calibration index below its perfect-calibration floor of 0 is meaningless; the
MM bias correction could push observed entropy above log(bins) on a genuinely
calibrated (near-uniform) PIT, driving eci negative pre-fix. A truly miscalibrated
input must still yield eci > 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.calibration.quality import entropy_calibration_index


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 8, 13, 21, 34, 55])
def test_eci_nonnegative_on_calibrated_uniform_pit(seed):
    rng = np.random.default_rng(seed)
    pit = rng.uniform(0.0, 1.0, size=500)
    eci = entropy_calibration_index(pit, bins=10, miller_madow=True)
    assert eci >= 0.0, f"eci negative on calibrated input: {eci}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5])
def test_eci_positive_on_miscalibrated_input(seed):
    rng = np.random.default_rng(seed)
    # PIT concentrated near 0 -> far from uniform -> genuine miscalibration.
    pit = rng.beta(0.5, 5.0, size=500)
    eci = entropy_calibration_index(pit, bins=10, miller_madow=True)
    assert eci > 0.0, f"miscalibrated input should give eci > 0: {eci}"
