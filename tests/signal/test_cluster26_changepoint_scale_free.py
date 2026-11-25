"""Cluster 26 regression: the changepoint penalty is scale-free (audit 2026-11-20, finding 26-F12).

The penalty multiplies a robust variance of ``y``, so rescaling the series does not change the breakpoints it finds. Before the fix a series
scaled by 1000 gave many more breakpoints at the same penalty.
"""

import numpy as np

from mlframe.signal.changepoint_detection import detect_regime_changepoints


def _series(seed: int = 0) -> np.ndarray:
    """Two regimes with a clear mean shift at row 100, unit-scale noise."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, 1.0, 100), rng.normal(5.0, 1.0, 100)])


def test_breakpoints_do_not_change_under_rescaling():
    """The same series times 1000 finds the same breakpoints as the original (26-F12)."""
    y = _series()
    base = detect_regime_changepoints(y, backend="njit")["breakpoints"]
    scaled = detect_regime_changepoints(y * 1000.0, backend="njit")["breakpoints"]
    assert base == scaled


def test_the_real_shift_is_found_at_unit_scale():
    """Control: the mean shift at row 100 is detected on the original series (26-F12)."""
    bps = detect_regime_changepoints(_series(), backend="njit")["breakpoints"]
    assert any(abs(b - 100) <= 5 for b in bps)
