"""Regression: ``_monotonic_residual_fit`` orient sign matches ``scipy.stats.spearmanr``.

The fit's monotone orientation is decided by the SIGN of the Spearman correlation between
``base`` and ``y``. The perf change replaced the full ``scipy.stats.spearmanr`` with
``_spearman_sign`` (ordinal-rank covariance sign). These tests pin sign-equivalence across
continuous, tied, and negative-slope data so a future "just use scipy again" or a broken
rank computation that flips the sign (and thus the fitted spline) is caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.transforms.nonlinear import (
    _monotonic_residual_fit,
    _spearman_sign,
)


def _scipy_dir(base: np.ndarray, y: np.ndarray) -> int:
    spearmanr = pytest.importorskip("scipy.stats").spearmanr
    rho, _ = spearmanr(base, y)
    return 1 if (rho is None or not np.isfinite(rho) or rho >= 0) else -1


@pytest.mark.parametrize("seed", range(40))
def test_spearman_sign_matches_scipy_across_random(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(500, 4000))
    base = rng.normal(size=n)
    if seed % 3 == 0:  # inject ties (discrete base)
        base = np.round(base)
    sign = rng.choice([1, -1])
    y = sign * np.tanh(base) + rng.normal(size=n) * rng.uniform(0.1, 1.5)
    assert _spearman_sign(base, y) == _scipy_dir(base, y)


def test_spearman_sign_increasing_and_decreasing() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=3000)
    assert _spearman_sign(base, 3.0 * base + rng.normal(size=3000) * 0.1) == 1
    assert _spearman_sign(base, -3.0 * base + rng.normal(size=3000) * 0.1) == -1


def test_spearman_sign_constant_input_is_increasing() -> None:
    # Degenerate (zero covariance) maps to +1, matching the legacy rho>=0 / rho-is-None rule.
    base = np.zeros(100)
    y = np.arange(100, dtype=float)
    assert _spearman_sign(base, y) == 1


def test_monotonic_fit_direction_matches_scipy() -> None:
    rng = np.random.default_rng(11)
    n = 5000
    base = rng.normal(size=n)
    y = -2.0 * np.tanh(base) + rng.normal(size=n) * 0.3  # decreasing
    params = _monotonic_residual_fit(y, base)
    assert params["monotone_direction"] == _scipy_dir(base, y) == -1
