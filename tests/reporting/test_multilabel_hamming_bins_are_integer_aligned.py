"""Hamming-distance histogram bins are one unit wide and centred on the integers 0..L."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.multilabel import _hamming_dist_panel


def test_bins_centre_on_integers_when_the_largest_distance_is_below_the_label_count():
    """Ten labels but a worst row of only 3 disagreements: the old [0, 3] range gave 11 bins of width 0.27 straddling the integers."""
    rng = np.random.default_rng(0)
    y_true = (rng.random((500, 10)) < 0.5).astype(np.int8)
    y_proba = y_true.astype(np.float64).copy()
    flips = rng.integers(0, 4, size=500)
    for i, k in enumerate(flips):
        cols = rng.choice(10, size=k, replace=False)
        y_proba[i, cols] = 1.0 - y_proba[i, cols]
    spec = _hamming_dist_panel(y_true, y_proba, None)
    assert spec.bin_width == 1.0
    np.testing.assert_array_equal(spec.bin_centers, np.arange(11, dtype=np.float64))
    np.testing.assert_allclose(np.asarray(spec.values).sum(), 1.0)
    assert np.asarray(spec.values)[4:].sum() == 0.0
    expected = np.bincount(flips, minlength=11) / 500
    np.testing.assert_allclose(spec.values, expected)
