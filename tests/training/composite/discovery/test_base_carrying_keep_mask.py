"""L-22: a column that carries the base (binned MI above a calibrated permutation null) is dropped; an independent column is kept."""

import numpy as np

from mlframe.training.composite.discovery._eval_stats import base_carrying_keep_mask
from mlframe.training.composite.discovery._eval_stats import compose_keep as _compose_keep


def test_carrier_is_dropped_and_independent_column_kept():
    """A copy of the base with noise is dropped; a column independent of the base is kept."""
    rng = np.random.default_rng(0)
    n = 2000
    base = rng.standard_normal(n)
    carrier = base + 0.1 * rng.standard_normal(n)
    independent = rng.standard_normal(n)
    X = np.column_stack([carrier, independent])
    keep = base_carrying_keep_mask(X, base, nbins=16)
    assert keep.tolist() == [False, True]


def test_small_sample_keeps_everything():
    """Fewer than 50 rows give no evidence, so every column is kept."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 3))
    assert base_carrying_keep_mask(X, rng.standard_normal(30), nbins=8).all()


def test_compose_keep_maps_a_second_mask_onto_the_original_columns():
    """Second mask over the already-kept columns lands on the right original positions."""
    prev = np.array([True, False, True, True])
    new = np.array([False, True, True])
    assert _compose_keep(prev, new).tolist() == [False, False, True, True]
