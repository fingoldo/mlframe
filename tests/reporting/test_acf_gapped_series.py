"""The ACF must not close the gaps a holed series leaves.

CHARTS_A-18 / C-12: four builders read ROW ORDER as time. Dropping a non-finite row closed the gap it left, so lag
k stopped meaning k steps of the caller's grid and started meaning "k surviving observations" -- biasing the
autocorrelation toward zero, increasingly so at higher lags. The estimator is pairwise-complete now: holes stay
holes and the autocovariance is normalised by the pairs actually observed at each lag.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts._acf import acf_fft, gap_fraction, pacf_levinson


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    """An AR(1) series whose true ACF is exactly ``phi**k``."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.mark.parametrize("hole_rate", [1 / 7, 1 / 3])
def test_known_ar1_survives_punched_holes(hole_rate):
    """The audit's own proposed meta-test: punch holes in a known AR(1), the coefficient must still come back."""
    phi, n = 0.7, 40_000
    x = _ar1(n, phi, seed=0)
    rng = np.random.default_rng(11)
    holed = x.copy()
    holed[rng.random(n) < hole_rate] = np.nan

    est, n_used = acf_fft(holed, nlags=5)
    truth = phi ** np.arange(1, 6)
    assert n_used < n, "n_used must count observations, not the grid they sit on"
    # Every lag within 0.02 of the truth, INCLUDING the high lags where gap-closing bias accumulated worst.
    assert np.max(np.abs(est - truth)) < 0.02, f"{est} vs {truth}"


def test_gap_closing_bias_is_actually_gone():
    """Pin the direction of the defect: the old estimator was biased toward zero, and worse at higher lags."""
    phi, n = 0.7, 40_000
    x = _ar1(n, phi, seed=0)
    rng = np.random.default_rng(11)
    holed = x.copy()
    holed[rng.random(n) < 1 / 7] = np.nan

    est, _ = acf_fft(holed, nlags=5)
    dropped, _ = acf_fft(holed[np.isfinite(holed)], nlags=5)  # what closing the gaps produces
    truth = phi ** np.arange(1, 6)
    assert np.all(dropped < truth - 0.005), "the gap-closing form should be biased low at every lag"
    assert np.max(np.abs(est - truth)) < np.max(np.abs(dropped - truth)) / 2


def test_gap_free_series_is_bit_identical_to_the_biased_estimator():
    """No holes -> the original 1/n normalisation, so nothing about a clean series moved."""
    x = _ar1(2_000, 0.5, seed=4)
    est, n_used = acf_fft(x, nlags=20)
    arr = x - x.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * arr.size - 1)))
    f = np.fft.rfft(arr, n=nfft)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft)[:21] / arr.size
    assert n_used == x.size
    assert np.array_equal(est, acov[1:] / acov[0])


def test_pacf_identifies_ar1_through_the_holes():
    """PACF is the panel that names the ORDER; with the bias it understated the lag-1 reflection coefficient."""
    x = _ar1(40_000, 0.7, seed=0)
    rng = np.random.default_rng(11)
    holed = x.copy()
    holed[rng.random(x.size) < 1 / 7] = np.nan
    pac, _ = pacf_levinson(holed, nlags=5)
    assert abs(pac[0] - 0.7) < 0.02
    assert np.all(np.abs(pac[1:]) < 0.05), "an AR(1) has no partial autocorrelation past lag 1"


def test_all_nan_and_constant_series_return_nothing():
    """Degenerate input must yield an empty vector so the panel annotates instead of drawing spurious bars."""
    assert acf_fft(np.full(50, np.nan), nlags=5)[0].size == 0
    assert acf_fft(np.full(50, 3.0), nlags=5)[0].size == 0
    assert gap_fraction(np.full(50, np.nan)) == 1.0
