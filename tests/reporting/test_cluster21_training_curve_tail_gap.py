"""Cluster 21 regression: a series that never logged the final iteration leaves a gap after its last sample, not a held value (audit 2026-11-20, finding 21-F17).

Interpolation onto the shared x must not extrapolate flat past the last logged position. Series that reach the final iteration still
interpolate normally.
"""

import numpy as np

from mlframe.reporting.charts.training_curve import _interp_with_gaps


def test_points_past_the_last_sample_are_nan():
    """Iterations beyond the last logged position are NaN, not the last value (21-F17)."""
    x = np.arange(10, dtype=np.float64)
    pos = np.array([0.0, 2.0, 4.0, 6.0])  # final iteration 9 was never logged
    s = np.array([1.0, 0.8, 0.6, 0.4])
    out = _interp_with_gaps(x, pos, s)
    assert np.isnan(out[7:]).all()
    assert np.isclose(out[2], 0.8)


def test_in_range_points_still_interpolate():
    """Control: points inside the sampled span are interpolated, not gapped (21-F17)."""
    x = np.arange(7, dtype=np.float64)
    pos = np.array([0.0, 2.0, 4.0, 6.0])
    s = np.array([1.0, 0.8, 0.6, 0.4])
    out = _interp_with_gaps(x, pos, s)
    assert np.isfinite(out).all()
    assert np.isclose(out[1], 0.9)
