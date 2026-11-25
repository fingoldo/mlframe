"""Cluster 34 regression: the finite-row mask from the auto-base prep equals the row-wise definition (audit 2026-11-20, finding 34-F21).

The mask is computed once after the impute and reused for the all-row mask and the per-pair counts. It must still mean exactly "row has a finite
target and every kept feature is finite".
"""

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.discovery._auto_base_prep import finite_mask_prep


def test_finite_mask_is_the_row_wise_definition():
    """A row is kept iff its target and all its kept features are finite (34-F21)."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 3))
    y = rng.normal(size=n)
    x[3, 1] = np.nan
    x[17, 0] = np.inf
    y[40] = np.nan
    self = SimpleNamespace(config=SimpleNamespace(auto_base_mi_per_pair_mask=True, auto_base_mnar_per_pair_threshold=0.5))
    out = finite_mask_prep(self, ["f0", "f1", "f2"], x.copy(), y, [], 10)
    finite = out[5]
    expected = np.isfinite(y) & np.isfinite(x).all(axis=1)
    assert np.array_equal(finite, expected)


def test_all_finite_input_keeps_every_row():
    """Control: with fully finite input every row is kept (34-F21)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 2))
    y = rng.normal(size=120)
    self = SimpleNamespace(config=SimpleNamespace(auto_base_mi_per_pair_mask=True, auto_base_mnar_per_pair_threshold=0.5))
    out = finite_mask_prep(self, ["a", "b"], x.copy(), y, [], 10)
    assert out[5].all()
