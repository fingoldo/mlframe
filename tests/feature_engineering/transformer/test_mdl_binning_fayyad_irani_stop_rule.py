"""The MDL split is accepted only when it beats the Fayyad-Irani cost with k, k1, k2 the classes present (tracker 2026-11-02 wave 12, NJ12-4)."""

import numpy as np

from mlframe.feature_engineering.transformer.mdl_binning_pairwise import _mdl_bin_edges


def test_split_on_a_weak_five_class_signal_is_rejected():
    """n=488, five classes, a mix of noise and a threshold signal: the hard-coded 2 and the size-weighted entropies understated the cost (delta 0.9 against 18.9), so a split was accepted."""
    rng = np.random.default_rng(10)
    n, k = int(rng.integers(100, 600)), int(rng.integers(2, 6))
    x = rng.normal(size=n)
    p = rng.uniform(0.1, 0.9)
    y = np.where(rng.uniform(size=n) < p, (x > 0).astype(int) * (k - 1), rng.integers(0, k, n))
    assert (n, k) == (488, 5)
    assert _mdl_bin_edges(x, y, k) == []
