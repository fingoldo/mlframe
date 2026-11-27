"""Cluster 17 regression: the quantile-binning kernel finds each pocket by binary search and agrees with the old linear scan (audit 2026-11-20, 17-F12)."""

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_plot import _quantile_binning_kernel


def _linear_reference(y_true, y_pred, edges):
    """The pre-fix linear scan, written in plain Python."""
    nbins = len(edges) - 1
    cnt = np.zeros(nbins, dtype=np.int64)
    pos = np.zeros(nbins, dtype=np.int64)
    psum = np.zeros(nbins)
    for yt, p in zip(y_true, y_pred):
        b = 0
        for e in range(1, nbins):
            if p >= edges[e]:
                b = e
            else:
                break
        cnt[b] += 1
        pos[b] += yt
        psum[b] += p
    keep = cnt > 0
    return psum[keep] / cnt[keep], pos[keep] / cnt[keep], cnt[keep]


@pytest.mark.parametrize("nbins", [1, 2, 7, 100])
def test_matches_linear_scan_including_ties_edges_and_nan(nbins):
    """Random scores, scores exactly on edges, out-of-range scores and a NaN land in the same pockets as the linear scan (17-F12)."""
    rng = np.random.default_rng(nbins)
    n = 3000
    p = rng.uniform(-0.1, 1.1, n)
    edges = np.sort(np.concatenate([[0.0, 1.0], rng.uniform(0, 1, nbins - 1)])) if nbins > 1 else np.array([0.0, 1.0])
    if nbins > 2:
        edges[2] = edges[1]
    k = min(5, len(edges))
    p[:k] = edges[:k]
    p[10] = np.nan
    y = (rng.uniform(size=n) < 0.4).astype(np.int8)
    got = _quantile_binning_kernel(y, p, edges)
    want = _linear_reference(y, p, edges)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, equal_nan=True)
