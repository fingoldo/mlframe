"""Cluster 17 regression: the weighted ICE's bin_index covers its quantile strategy and both fallbacks (audit 2026-11-20, finding 17-F20)."""

import numpy as np

from mlframe.metrics.classification._ice_kernel_weighted import bin_index


def test_quantile_bins_hold_equal_weight_mass_and_follow_the_weights():
    """With unit weights the equal-mass bins are equally populated; heavy rows pull the edges so each bin holds a similar WEIGHT, not a similar count."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(float)
    counts = np.bincount(bin_index(y, p, np.ones(1000), 10, "quantile"), minlength=10)
    assert counts.min() >= 90 and counts.max() <= 110
    w = np.where(p > 0.9, 20.0, 1.0)
    idx = bin_index(y, p, w, 10, "quantile")
    mass = np.bincount(idx, weights=w, minlength=10)
    assert mass.max() / mass.min() < 3.0, "weight mass per bin stays comparable even though 10 percent of rows carry 20x weight"


def test_uniform_strategy_and_the_single_distinct_edge_fallback():
    """Uniform bins are equal-width and clamped to [0, nbins); a constant prediction (one distinct edge) falls back to bin 0 for every row."""
    p = np.linspace(0.0, 1.0, 101)
    idx = bin_index(np.zeros(101), p, np.ones(101), 5, "uniform")
    assert idx.min() == 0 and idx.max() == 4 and np.all(np.diff(idx) >= 0)
    const = bin_index(np.zeros(8), np.full(8, 0.3), np.ones(8), 4, "quantile")
    assert (const == 0).all()
