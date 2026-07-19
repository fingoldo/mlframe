"""Business-value tests for the closed-form adaptive-nbins strategies (sturges / freedman_diaconis
/ knuth): each must actually RESOLVE genuine structure at a known number of natural clusters, not
just "not crash" on synthetic data (``assert res is not None`` style checks would have missed the
silent-degenerate-fallback bug class documented in
``tests/feature_selection/fe/adaptive/test_adaptive_nbins_degenerate_fallback.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import (
    edges_sturges,
    edges_freedman_diaconis,
    edges_knuth,
    _plug_in_mi,
)

# ln(4) - the maximum achievable MI when x perfectly separates 4 equiprobable clusters.
_MAX_MI_4_CLUSTERS = float(np.log(4))


def _four_cluster_data(seed: int):
    """4 well-separated Gaussian clusters on x, y = cluster id. Any binner that places at
    least one edge between each pair of adjacent clusters recovers ~ln(4) nats of MI."""
    rng = np.random.default_rng(seed)
    n_per = 5000
    centers = [0.0, 10.0, 20.0, 30.0]
    x = np.concatenate([rng.normal(c, 0.5, n_per) for c in centers])
    y = np.concatenate([np.full(n_per, i, dtype=np.int64) for i in range(len(centers))])
    idx = rng.permutation(x.size)
    return x[idx], y[idx]


@pytest.mark.parametrize("name,fn", [("sturges", edges_sturges), ("freedman_diaconis", edges_freedman_diaconis), ("knuth", edges_knuth)])
def test_biz_val_adaptive_nbins_resolves_four_natural_clusters(name, fn):
    """Threshold set ~10% below the measured MI (~1.374-1.385 nats across all three methods,
    seed 0) and ~ln(4)=1.386 theoretical max -- a method that under-bins (merges clusters) or
    silently collapses (the MDLP-class bug) would score far below this."""
    x, y = _four_cluster_data(seed=0)
    edges = fn(x)
    assert edges.size > 0, f"{name}: 4-cluster column silently produced ZERO edges"
    x_binned = np.searchsorted(edges, x, side="right").astype(np.int64)
    mi = _plug_in_mi(x_binned, y, miller_madow=True)
    threshold = 0.90 * _MAX_MI_4_CLUSTERS  # ~1.247
    assert mi >= threshold, f"{name}: MI={mi:.4f} below {threshold:.4f} (max={_MAX_MI_4_CLUSTERS:.4f}) -- clusters not resolved"


@pytest.mark.parametrize("name,fn", [("sturges", edges_sturges), ("freedman_diaconis", edges_freedman_diaconis), ("knuth", edges_knuth)])
def test_biz_val_adaptive_nbins_no_signal_stays_low(name, fn):
    """Counterpart to the cluster-resolution test: pure noise (x independent of y) must score
    NEAR zero MI after Miller-Madow correction -- confirms the above isn't just "MI is always
    high regardless of binning", it's genuinely conditional on real structure."""
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.standard_normal(n)
    y = rng.integers(0, 4, size=n).astype(np.int64)
    edges = fn(x)
    x_binned = np.searchsorted(edges, x, side="right").astype(np.int64) if edges.size else np.zeros(n, dtype=np.int64)
    mi = _plug_in_mi(x_binned, y, miller_madow=True)
    assert mi < 0.05, f"{name}: no-signal column scored MI={mi:.4f} (expected near 0)"
