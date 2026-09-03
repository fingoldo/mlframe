"""The worm plot's normality verdict must not depend on how many rows the model was scored on.

The panel decimates to at most 2000 plotting positions and deliberately over-represents the tails (100 head +
100 tail), which raises a fair question: is the "% of points outside the 95% band" -- and therefore the
HEAVY TAILS / normal verdict printed in the title -- biased by that decimation, and does it flip at the n=2000
boundary where decimation switches on?

Measured here rather than reasoned about. Across the boundary and four orders of magnitude, a Gaussian
residual scores 0.0002-0.0143 and a t(3) scores 0.61-0.99, so the 0.05 threshold sits between two populations
that are three to four orders of magnitude apart in separation, not on its own null. The reason the observed
Gaussian rate is far below the nominal 5% is that standardising by the SAMPLE sd absorbs most of the
variability the asymptotic quantile SE budgets for -- the same mechanism that makes a heavy tail pull BOTH
worm tails inside the band, which the shape table in `regression.py` already accounts for.

This file pins that separation. A future change to the decimation, the plotting positions, or the standardiser
that genuinely does bias the verdict shows up as the Gaussian rate climbing toward the threshold or the two
populations overlapping.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from mlframe.reporting.charts.regression import _WORM_PLOT_CAP, _WORM_TAIL_KEEP, _decimate_keep_tails

# The title flips to "HEAVY TAILS -- a few errors far larger than Gaussian (RMSE understates worst-case)" here.
_HEAVY_TAIL_THRESHOLD = 0.05


def _frac_outside_band(resid: np.ndarray) -> float:
    """The panel's own `_frac_out`: the share of plotted worm points falling outside the pointwise 95% band."""
    n = resid.size
    mu, sd = float(resid.mean()), float(resid.std())
    order_stats = np.sort(resid)
    keep = _decimate_keep_tails(n, _WORM_PLOT_CAP, _WORM_TAIL_KEEP)
    z_sample = (order_stats[keep] - mu) / sd
    p_k = (keep.astype(np.float64) + 1.0 - 0.375) / (n + 0.25)  # Blom plotting positions
    zt = norm.ppf(p_k)
    detrended = z_sample - zt
    phi = np.maximum(norm.pdf(zt), 1e-12)
    ci = 1.959963984540054 * np.sqrt(p_k * (1.0 - p_k) / n) / phi
    return float(np.mean(np.abs(detrended) > ci)) if ci.size else 0.0


def _mean_frac(dist: str, n: int, reps: int = 3, seed: int = 0) -> float:
    """Average `_frac_out` over a few draws, so one unlucky sample does not decide the assertion."""
    rng = np.random.default_rng(seed)
    draw = (lambda: rng.normal(0, 1, n)) if dist == "gauss" else (lambda: rng.standard_t(3, n))
    return float(np.mean([_frac_outside_band(draw()) for _ in range(reps)]))


class TestTheVerdictIsStableAcrossTheDecimationBoundary:
    """Below 2000 rows the decimation is a no-op; above it, 10% of the plotted points are tail order statistics."""

    @pytest.mark.parametrize("n", [500, 1999, 2001, 10_000])
    def test_gaussian_residuals_never_read_as_heavy_tailed(self, n):
        """Measured 0.0087 / 0.0143 / 0.0002 / 0.0068 at these sizes."""
        assert _mean_frac("gauss", n) < _HEAVY_TAIL_THRESHOLD

    @pytest.mark.parametrize("n", [500, 1999, 2001, 10_000])
    def test_heavy_tailed_residuals_always_read_as_heavy_tailed(self, n):
        """Measured 0.61 / 0.86 / 0.84 / 0.94 at these sizes."""
        assert _mean_frac("t3", n) >= _HEAVY_TAIL_THRESHOLD

    def test_the_boundary_itself_does_not_flip_the_verdict(self):
        """The sharpest form: 1999 rows and 2001 rows are the same distribution on opposite sides of the switch."""
        below, above = _mean_frac("gauss", 1999), _mean_frac("gauss", 2001)
        assert (below < _HEAVY_TAIL_THRESHOLD) == (above < _HEAVY_TAIL_THRESHOLD), (below, above)


class TestTheThresholdHasRealMargin:
    """A pointwise 95% band excludes ~5% of points under normality in THEORY, which would put 0.05 on its own null."""

    def test_the_observed_gaussian_rate_is_far_below_the_nominal_five_percent(self):
        """It is not on its own null: standardising by the sample sd absorbs most of the budgeted variability."""
        assert _mean_frac("gauss", 100_000) < _HEAVY_TAIL_THRESHOLD / 3.0

    def test_the_two_populations_do_not_overlap(self):
        """The separation is what makes the verdict trustworthy, not the exact constant."""
        assert _mean_frac("t3", 100_000) > 10 * _mean_frac("gauss", 100_000)

    def test_a_mildly_heavy_tail_still_separates(self):
        """t(8) is much closer to Normal than t(3); the verdict must still be able to tell them apart."""
        rng = np.random.default_rng(7)
        t8 = float(np.mean([_frac_outside_band(rng.standard_t(8, 20_000)) for _ in range(3)]))
        assert t8 > _mean_frac("gauss", 20_000)
