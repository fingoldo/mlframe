"""biz_value + unit tests for the COMPETITION-ONLY FloatPrecisionDenoiser (mlframe.competition).

Mirrors the Amex/BNP "denoise deliberately injected float noise" writeups: a low-cardinality
integer/rounded true value is multiplicatively scaled and has small noise injected on top,
and the denoiser must recover the true value far more accurately than the raw noisy series.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.competition.float_precision_denoise import FloatPrecisionDenoiser


def _make_amex_like(n: int, scale: float, noise_scale: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True integer values in [0, 99], scaled by `scale`, with small proportional noise injected.

    Noise is kept well inside the quantization step (fraction `noise_scale` of it, default caller
    passes 1.0 meaning "up to one full step" — actual writeups inject noise much smaller than the
    step so the true value is still recoverable) so floor-based recovery is unambiguous.
    """
    rng = np.random.default_rng(seed)
    true_int = rng.integers(0, 100, size=n)
    true_value = true_int / scale
    noise = rng.uniform(0.0, 0.2 / scale, size=n)  # small noise, well within one quantization step
    noisy = true_value + noise
    return true_int.astype(np.float64), true_value, noisy


def test_float_precision_denoiser_recovers_known_scale_amex_style() -> None:
    true_int, true_value, noisy = _make_amex_like(n=5000, scale=100.0, noise_scale=1.0, seed=0)

    denoiser = FloatPrecisionDenoiser(max_decimal_pow=6, max_denominator=1000, use_floor=True)
    result = denoiser.fit_transform(noisy)

    assert denoiser.denominator_ == pytest.approx(100.0)
    np.testing.assert_allclose(result.denoised * 100.0, true_int, atol=1e-6)


def test_float_precision_denoiser_finds_unknown_small_integer_denominator() -> None:
    # BNP-style: denominator is an arbitrary small integer, not a power of 10.
    rng = np.random.default_rng(1)
    denominator = 37.0
    true_int = rng.integers(0, 50, size=3000)
    true_value = true_int / denominator
    noise = rng.uniform(0.0, 0.2 / denominator, size=3000)
    noisy = true_value + noise

    denoiser = FloatPrecisionDenoiser(max_decimal_pow=6, max_denominator=1000, use_floor=True)
    denoiser.fit(noisy)

    assert denoiser.denominator_ == pytest.approx(denominator)


def test_float_precision_denoiser_transform_matches_fit_transform() -> None:
    _, _, noisy = _make_amex_like(n=1000, scale=100.0, noise_scale=1.0, seed=2)
    denoiser = FloatPrecisionDenoiser()
    denoiser.fit(noisy)
    manual = denoiser.transform(noisy)
    combined = FloatPrecisionDenoiser().fit_transform(noisy)
    np.testing.assert_allclose(manual, combined.denoised)


def test_float_precision_denoiser_rejects_empty_series() -> None:
    with pytest.raises(ValueError):
        FloatPrecisionDenoiser().fit(np.array([]))


def test_float_precision_denoiser_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        FloatPrecisionDenoiser(max_decimal_pow=-1)
    with pytest.raises(ValueError):
        FloatPrecisionDenoiser(max_denominator=0)


def test_biz_val_float_precision_denoiser_recovery_accuracy_beats_raw() -> None:
    """Denoiser recovers the true coarse value far more accurately than the raw noisy series.

    Synthetic mirrors the Amex writeup: true low-cardinality integer values scaled by 100 with
    small proportional noise injected on top (as in x[i] = np.floor(x[i] * 100) preprocessing).
    """
    true_int, true_value, noisy = _make_amex_like(n=20000, scale=100.0, noise_scale=1.0, seed=42)

    denoiser = FloatPrecisionDenoiser(max_decimal_pow=6, max_denominator=1000, use_floor=True)
    result = denoiser.fit_transform(noisy)

    tol = 1e-6
    denoised_hit_rate = float(np.mean(np.isclose(result.denoised, true_value, atol=tol)))
    raw_hit_rate = float(np.mean(np.isclose(noisy, true_value, atol=tol)))

    # measured: denoised_hit_rate == 1.0, raw_hit_rate == 0.0 on this synthetic; thresholds set
    # comfortably below the measured value per project convention (5-15% margin)
    assert denoised_hit_rate >= 0.95
    assert raw_hit_rate <= 0.05
    assert denoised_hit_rate > raw_hit_rate
