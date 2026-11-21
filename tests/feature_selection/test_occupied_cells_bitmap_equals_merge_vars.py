"""The bit-map count of occupied cells equals the ``merge_vars`` count of the same joint, and signals a code outside its bins (MRMR-CONFIRM-CELLS, 2026-11-19)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._confirm_predictor_engineered import _occupied_cells_over
from mlframe.feature_selection.filters.info_theory import merge_vars


def _merge_count(data: np.ndarray, z_idx: list, xy_idx: list, nbins: np.ndarray) -> tuple:
    """The reference: fold Z into a dense class array, then fold the other variables on top, as the confirm path did."""
    z_classes, _, z_n = merge_vars(factors_data=data, vars_indices=z_idx, var_is_nominal=None, factors_nbins=nbins, dtype=np.int32, current_nclasses=1, final_classes=None, verbose=False)
    _, _, cells = merge_vars(factors_data=data, vars_indices=xy_idx, var_is_nominal=None, factors_nbins=nbins, dtype=np.int32, current_nclasses=z_n, final_classes=z_classes.copy(), verbose=False)
    return z_classes, int(z_n), int(cells)


@pytest.mark.parametrize("n, z_vars, xy_vars, nb", [(20000, 4, 3, 10), (50000, 6, 2, 12), (5000, 2, 3, 10), (30000, 5, 3, 3)])
def test_bitmap_count_equals_merge_vars(n: int, z_vars: int, xy_vars: int, nb: int):
    """Random codes with a constant column and a skewed one: both counts agree for the joint of the other variables folded over the Z classes."""
    rng = np.random.default_rng(n + nb)
    p = z_vars + xy_vars
    data = rng.integers(0, nb, size=(n, p)).astype(np.int8)
    data[:, 0] = 1
    data[:, 1] = np.where(rng.random(n) < 0.9, 0, rng.integers(0, nb, n))
    nbins = np.full(p, nb, dtype=np.int64)
    z_idx = list(range(z_vars))
    xy_idx = list(range(z_vars, p))
    z_classes, z_n, ref = _merge_count(data, z_idx, xy_idx, nbins)
    span = z_n
    for c in xy_idx:
        span *= int(nbins[c])
    got = _occupied_cells_over(np.ascontiguousarray(z_classes, dtype=np.int32), data, np.asarray(xy_idx, dtype=np.int64), nbins, span)
    assert got == ref, (got, ref)


def test_code_outside_its_bins_is_signalled():
    """A code at or above its bin count returns -1, so the caller falls back to ``merge_vars``."""
    data = np.array([[0, 1], [0, 5], [1, 0]], dtype=np.int8)
    nbins = np.array([2, 3], dtype=np.int64)
    got = _occupied_cells_over(np.zeros(3, dtype=np.int32), data, np.array([0, 1], dtype=np.int64), nbins, 6)
    assert got == -1
