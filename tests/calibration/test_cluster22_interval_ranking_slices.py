"""Cluster 22 regression: a full-length ranking is cut to each interval's rows before binning (audit 2026-11-20, finding 22-F12).

``show_classifier_calibration`` passed the same full-length ``indices`` to every interval, but each interval is binned on its own slice, so the
unchecked njit kernel read past the slice or binned the wrong rows. With the ranking cut down to the slice, each interval must score exactly as it does
when it is sorted by itself.
"""

import numpy as np

from mlframe.calibration.quality import estimate_calibration_quality_binned, show_classifier_calibration


def test_intervals_with_a_shared_ranking_match_intervals_sorted_alone():
    """Two intervals scored with the full argsort give the same metrics as each slice binned alone (22-F12)."""
    rng = np.random.default_rng(0)
    n = 400
    y_pred = rng.permutation(n) / n + 1e-6  # distinct values: no ties, so the slice ranking is unambiguous
    y_true = (rng.uniform(size=n) < y_pred).astype(float)
    indices = np.argsort(y_pred)

    got = show_classifier_calibration(y_true, y_pred, "t", indices=indices, nbins=5, nintervals=2, skip_plotting=True)
    half = n // 2
    expected = [
        estimate_calibration_quality_binned(y_true[:half], y_pred[:half], nbins=5)[3],
        estimate_calibration_quality_binned(y_true[half:], y_pred[half:], nbins=5)[3],
    ]
    assert isinstance(got, list) and len(got) == 2
    for g, e in zip(got, expected):
        assert set(g) == set(e)
        for key in e:
            assert np.isclose(float(g[key]), float(e[key]), equal_nan=True), key
