"""The pair-prewarp ALS target is winsorized only when its variance is carried by a thin tail."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_setup import winsorize_heavy_tailed_target
from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp import apply_operand_prewarp, fit_pair_prewarp_als


def _case2(n=30000):
    rng = np.random.default_rng(0)
    a, b, c, d, f = (rng.random(n) for _ in range(5))
    return c, d, 0.2 * a**2 / b + f / 5.0 + np.log(c * 2) * np.sin(d / 3)


def test_light_tailed_target_is_returned_unchanged():
    """Includes smoothly heavy-tailed targets (lognormal, Student-t(3)) whose tail is part of the signal."""
    rng = np.random.default_rng(1)
    for y in (rng.standard_normal(5000), rng.lognormal(0.0, 1.0, 5000), rng.standard_t(3, 5000)):
        assert winsorize_heavy_tailed_target(y) is y


def test_heavy_tailed_target_is_clipped_to_its_1_99_quantiles():
    _, _, y = _case2()
    out = winsorize_heavy_tailed_target(y)
    lo, hi = np.quantile(y, [0.01, 0.99])
    assert out.min() == lo and out.max() == hi


def test_prewarp_recovers_log_factor_under_heavy_tailed_target():
    """``y = 0.2 a**2/b + f/5 + log(2c) sin(d/3)``: on the raw target the ALS warp of ``c`` chases the a**2/b tail."""
    c, d, y = _case2()
    raw_a, _ = fit_pair_prewarp_als(c, d, y)
    win_a, _ = fit_pair_prewarp_als(c, d, winsorize_heavy_tailed_target(y))
    truth = np.log(2 * c)
    raw_corr = abs(np.corrcoef(apply_operand_prewarp(c, raw_a), truth)[0, 1])
    win_corr = abs(np.corrcoef(apply_operand_prewarp(c, win_a), truth)[0, 1])
    assert win_corr > raw_corr + 0.03 and win_corr > 0.95, (raw_corr, win_corr)
