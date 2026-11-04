"""The prediction-interval band plots only finite rows, so a NaN row never reaches the line or the band."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.quantile import _interval_band_panel


def test_nan_rows_are_not_drawn():
    """Rows with a NaN target or NaN quantile are excluded from the band, the median line and the markers."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.normal(size=n)
    P = np.sort(rng.normal(size=(n, 3)), axis=1)
    y[:20] = np.nan
    P[20:50, 1] = np.nan
    spec = _interval_band_panel(y, P, [0.1, 0.5, 0.9])
    assert len(spec.x) == n - 50
    lo, hi = spec.band
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert len(spec.y) > 0
    for series in spec.y:
        assert np.all(np.isfinite(np.asarray(series, dtype=np.float64)))
