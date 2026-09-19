"""Tail-aliasing guard for the adaptive / chirp Fourier frequency detector.

A detected frequency is only an OSCILLATION if the sinusoid actually completes a meaningful fraction of a cycle where the
data lives. On a skewed column (and especially on the chirp axis ``u = sign(z)*z**2``, which squares the tail) the [0, 1]
working axis is dominated by a few extreme rows: on california_housing ``AveRooms`` the 5%-95% core of the chirp axis spans
only ~0.004 of it, so a "frequency 23.4" completes 0.09 cycles over 90% of the rows. In the core such a leg is a smooth
sub-cycle ramp (a trend the polynomial basis already expresses), while its oscillation lives entirely in the sparse tail,
where the sin/cos value of a row is fixed by which of ~20 unsampled cycles it lands in. The held-out gate can pass it
(tail rows are high-leverage in the periodogram power), but on unseen data every new tail row gets an arbitrary +-1 and
the linear downstream model extrapolates garbage (measured: holdout R^2 0.13 with the eight ``AveRooms__qsin/qcos`` legs
at freqs 22-24 admitted vs 0.39 without any one of them).

The guard rejects a frequency whose cycle count over the inner-quantile core is below ``MIN_CORE_CYCLES``. A genuine
oscillation / chirp spans many cycles over the core (a standard-normal chirp at the detector's u-frequency ~2 already covers
~0.5 cycles; typical fixtures 1-10), so it is untouched.
"""

from __future__ import annotations

import numpy as np

# Inner-quantile core of the working axis over which cycles are counted.
CORE_Q_LO = 0.05
CORE_Q_HI = 0.95
# Minimum number of sinusoid cycles over that core. The coarse grid's lowest frequency (0.5) on a non-skewed axis (core ~0.9
# of the span) gives ~0.45 cycles, well above this bar, so undistorted columns keep every frequency they had.
MIN_CORE_CYCLES = 0.25


def core_span(z01: np.ndarray) -> float:
    """Width of the ``[CORE_Q_LO, CORE_Q_HI]`` inner-quantile core of the (finite) working axis ``z01``."""
    z = np.asarray(z01, dtype=np.float64).ravel()
    z = z[np.isfinite(z)]
    if z.size < 2:
        return 0.0
    lo, hi = np.quantile(z, [CORE_Q_LO, CORE_Q_HI])
    return float(hi - lo)


def freq_is_tail_aliased(freq: float, span: float) -> bool:
    """True iff ``freq`` completes fewer than ``MIN_CORE_CYCLES`` cycles over a core of width ``span``."""
    return float(freq) * float(span) < MIN_CORE_CYCLES
