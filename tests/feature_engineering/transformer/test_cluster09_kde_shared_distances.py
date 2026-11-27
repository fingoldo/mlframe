"""Cluster 09 regression: the KDE log-ratio features build the pairwise distances once per class and reuse them across bandwidths (audit 2026-11-20, 09-F27)."""

import numpy as np

from mlframe.feature_engineering.transformer import density_ratio as dr


def test_multi_bandwidth_rows_equal_single_bandwidth_calls():
    """Each row of the multi-bandwidth result is exactly the single-bandwidth log-KDE at that bandwidth (09-F27)."""
    rng = np.random.default_rng(0)
    Xt = rng.normal(size=(120, 5)).astype(np.float32)
    Xq = rng.normal(size=(2500, 5)).astype(np.float32)
    hs = (0.3, 0.7, 1.4, 2.5)
    multi = dr._gaussian_kde_log_multi(Xq, Xt, hs, chunk=700)
    assert multi.shape == (4, 2500)
    for j, h in enumerate(hs):
        np.testing.assert_array_equal(multi[j], dr._gaussian_kde_log(Xq, Xt, h, chunk=700))


def test_empty_training_subset_gives_the_floor_for_every_bandwidth():
    """With no training rows every bandwidth row is the -30 floor (09-F27)."""
    out = dr._gaussian_kde_log_multi(np.zeros((3, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), (1.0, 2.0))
    assert out.shape == (2, 3) and (out == -30.0).all()
