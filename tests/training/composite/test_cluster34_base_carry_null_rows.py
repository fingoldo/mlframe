"""Audit 2026-11-20 finding 34-F19: the base-carrying null is computed on the same rows as the real MI.

Before the fix the base was permuted over all rows and then masked with the unpermuted finite mask, so NaN base values could
enter the scored rows and the null used a smaller set. The test makes a column that carries the base exactly, with NaN base
entries, and checks it is dropped (its MI is far above any shuffled null) while an independent column is kept.
"""

import numpy as np

from mlframe.training.composite.discovery._eval_stats import base_carrying_keep_mask


def test_carrying_column_is_dropped_with_nan_base_values():
    """A column equal to the base (with NaN base entries) is dropped; an independent column is kept."""
    rng = np.random.default_rng(0)
    n = 400
    base = rng.normal(size=n)
    base[::17] = np.nan
    carrier = base.copy()
    independent = rng.normal(size=n)
    x = np.column_stack([carrier, independent])
    keep = base_carrying_keep_mask(x, base, nbins=10, n_permutations=10, seed=1)
    assert keep.tolist() == [False, True]
