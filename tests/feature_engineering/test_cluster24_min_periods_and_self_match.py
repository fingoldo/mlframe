"""Cluster 24 regressions: rolling min_periods never exceeds a small window, and a self-query's k-th distance is to another point (audit 2026-11-20, findings 24-F23, 24-F26)."""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.rolling_target_correlation import rolling_target_correlation_tracker
from mlframe.feature_engineering.spatial import local_density_features


def test_small_window_does_not_crash_on_the_default_min_periods():
    """A window of 5 used to ask pandas for min_periods 10 (24-F26)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=40)})
    y = rng.normal(size=40)
    out = rolling_target_correlation_tracker(X, y, window=5)
    assert len(out) == 40


def test_in_sample_kth_distance_excludes_the_point_itself():
    """Querying the reference with itself: dist_to_kth must be the k-th OTHER point (24-F23)."""
    pts = np.array([[0.0], [1.0], [2.0], [4.0], [8.0], [16.0]])
    feats = local_density_features(pts, pts, k=2)
    # For point 0 the two nearest OTHER points are at 1 and 2, so the 2nd is 2.0.
    assert np.isclose(feats["dist_to_kth"][0], 2.0)
