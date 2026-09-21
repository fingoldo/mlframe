"""Regression: the adaptive/chirp Fourier detector must not admit a frequency whose oscillation lives only in a sparse tail.

On a skewed column with a few extreme rows the chirp axis ``u = sign(z)*z**2`` puts ~90% of the rows into a sliver (~0.005) of
[0, 1]. The detector then locked frequencies ~20-24 that complete ~0.1 cycles over the bulk (a smooth ramp there) and oscillate
only across the unsampled tail gaps - california_housing ``AveRooms__qsin/qcos23.5`` etc., which extrapolate as arbitrary +-1
on unseen tail rows and dragged the downstream holdout R^2 from 0.39 to 0.13.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._orthogonal_univariate_fe._fourier_core_cycles import core_span
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe import (
    _chirp_axis,
    _detect_fourier_freqs_for_col,
    _fit_chirp_warp_for_col,
)

_CHIRP_GRID = tuple(0.5 * k for k in range(1, 49))


def _skewed_tail_column(seed: int, n: int = 1125):
    """Lognormal bulk + 12 extreme rows; y is a smooth (log) function of x plus noise, NOT an oscillation."""
    rng = np.random.default_rng(seed)
    x = rng.lognormal(1.6, 0.25, n)
    k = rng.choice(n, 12, replace=False)
    x[k] = rng.uniform(15, 50, 12)
    y = np.log(x) + rng.normal(0, 0.3, n)
    y[k] += rng.normal(0, 3, 12)
    return x, y


def test_chirp_detector_rejects_tail_aliased_frequencies_on_skewed_column():
    """Before the guard, seeds 0/3/4/5 admitted freqs 10-23 on the chirp axis; each completes < 0.25 cycles over the core."""
    for seed in (0, 3, 4, 5):
        x, y = _skewed_tail_column(seed)
        u = _chirp_axis(x, *_fit_chirp_warp_for_col(x))
        assert core_span(u) < 0.01  # the scenario: bulk squeezed into a sliver of the axis
        freqs = _detect_fourier_freqs_for_col(u, y, f_grid=_CHIRP_GRID, min_rows=800, max_freqs=6)
        assert freqs == [], f"seed={seed}: tail-aliased chirp frequencies admitted: {freqs}"


def test_chirp_detector_still_recovers_genuine_chirp():
    """Positive control: a genuine chirp on a well-behaved column keeps its detected frequency."""
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(0, 1, n)
    y = np.sin(2 * np.pi * 0.5 * x * x) + rng.normal(0, 0.2, n)
    u = _chirp_axis(x, *_fit_chirp_warp_for_col(x))
    freqs = _detect_fourier_freqs_for_col(u, y, f_grid=_CHIRP_GRID, min_rows=800, max_freqs=6)
    assert freqs, "genuine chirp no longer detected"
