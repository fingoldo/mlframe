"""Cluster 20 regression: with tied scores the AP interval is centred on the AP the panel title shows (audit 2026-11-20, 20-F12)."""

import numpy as np
from sklearn.metrics import average_precision_score

from mlframe.reporting.charts.binary import bootstrap_ap_ci


def test_point_estimate_is_sklearns_ap_under_heavy_ties():
    """Quantised scores: the returned AP equals sklearn's distinct-threshold AP, and the interval brackets it (20-F12)."""
    rng = np.random.default_rng(0)
    n = 6000
    y = (rng.uniform(size=n) < 0.3).astype(int)
    s = np.round(np.clip(0.3 + 0.25 * y + 0.2 * rng.normal(size=n), 0, 1), 1)
    ap, lo, hi = bootstrap_ap_ci(y, s, seed=0)
    assert abs(ap - average_precision_score(y, s)) < 1e-9
    assert lo < ap < hi
