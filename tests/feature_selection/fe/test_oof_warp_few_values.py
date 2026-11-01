"""The few-values screen of the OOF warp equals counting the distinct values of the whole column."""

import numpy as np

from mlframe.feature_selection.filters._oof_warp_fe import _has_few_values


def test_equals_unique_count():
    """Discrete, continuous, and a column whose distinct values only appear after the prefix."""
    rng = np.random.default_rng(0)
    late = np.concatenate([np.zeros(5000), rng.normal(size=5000)])
    for v in (rng.integers(0, 5, 9000).astype(float), rng.normal(size=9000), late, np.arange(10.0)):
        for cap in (3, 5, 20):
            assert _has_few_values(v, cap) == (np.unique(v).size <= cap)
