"""The Miller-Madow correction removes the plug-in MI bias instead of doubling it (tracker 2026-11-02 wave 11, NJ11-1)."""

import numpy as np

from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import _bin_mi_uniform_pair


def test_copula_pair_mi_of_independent_columns_is_near_zero():
    """20x20 bins, n=2000: the plug-in bias is ~0.09 nats. The corrected estimate averages ~0; the old code added the bias again and averaged ~0.18."""
    rng = np.random.default_rng(0)
    vals = [_bin_mi_uniform_pair(rng.uniform(size=2000), rng.uniform(size=2000), n_bins=20) for _ in range(30)]
    assert np.mean(vals) < 0.03
