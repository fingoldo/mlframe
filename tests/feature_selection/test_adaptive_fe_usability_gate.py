"""Regression: the adaptive Fourier/chirp auto-gate fires only where the raw
column is NOT already a strong smooth predictor of y.

The default-ON adaptive Fourier/chirp operators were hijacking selection on
linear / monotone / heavy-tailed / leak columns -- emitting ``__sin``/``__cos``/
``__qcos`` columns that replaced the raw signal in ``support_``. The gate keys
on a rank-based held-out cubic R^2 so heavy-tailed monotone signals (Cauchy,
lognormal) read as usable (gate blocks Fourier) while genuine oscillations stay
below the cap (gate lets Fourier fire).
"""
import numpy as np

from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
    _ADAPTIVE_FE_RAW_USABILITY_CAP,
    _heldout_smooth_r2,
)


def test_smooth_r2_high_on_monotone_low_on_oscillation():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    y_linear = 1.5 * x + 0.05 * rng.standard_normal(n)
    assert _heldout_smooth_r2(x, y_linear) >= _ADAPTIVE_FE_RAW_USABILITY_CAP

    # Pure high-frequency oscillation: raw x cannot predict y with a cubic.
    xu = rng.uniform(-1.0, 1.0, n)
    y_osc = np.sin(2.0 * np.pi * 5.3 * xu) + 0.05 * rng.standard_normal(n)
    assert _heldout_smooth_r2(xu, y_osc) < _ADAPTIVE_FE_RAW_USABILITY_CAP


def test_smooth_r2_robust_to_heavy_tails():
    # A Cauchy-distributed monotone signal: extreme outliers must NOT understate
    # usability (the bug that let Fourier hijack ``cauchy_sig`` into support).
    rng = np.random.default_rng(1)
    n = 2500
    x = rng.standard_cauchy(n)
    y = (x > np.median(x)).astype(float) + 0.01 * rng.standard_normal(n)
    assert _heldout_smooth_r2(x, y) >= _ADAPTIVE_FE_RAW_USABILITY_CAP


def test_smooth_r2_degenerate_inputs_return_zero():
    assert _heldout_smooth_r2(np.zeros(100), np.arange(100.0)) == 0.0   # constant x
    assert _heldout_smooth_r2(np.arange(10.0), np.arange(10.0)) == 0.0  # too few rows
