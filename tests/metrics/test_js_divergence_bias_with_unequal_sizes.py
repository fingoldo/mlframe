"""The JS bias correction uses (K-1)/8 * (1/np + 1/nq), exact for unequal sample sizes (tracker 2026-11-02 wave 13, FORM13-1)."""

import numpy as np

from mlframe.metrics._drift import js_divergence


def test_same_distribution_with_unequal_sizes_scores_near_zero():
    """100 vs 10000 draws of one distribution, 10 bins: the plug-in JS averages ~0.011; the old (K-1)/(2N) floor removed only 0.0004 of it."""
    rng = np.random.default_rng(0)
    vals = [js_divergence(rng.normal(size=100), rng.normal(size=10000), nbins=10) for _ in range(60)]
    assert np.mean(vals) < 0.004
