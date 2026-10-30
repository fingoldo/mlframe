"""The held-out MDLP cut search reads the train part of a column without ties off the column's sort order; the cuts are those of the search that sorts the train rows."""

from __future__ import annotations

import numpy as np
import pytest
from numba import njit

from mlframe.feature_selection.filters._mdlp_oos_honest_njit import oos_mdlp_cuts_njit, oos_mdlp_cuts_presorted_njit


@njit(cache=False)
def _sorting(x, y, seed, n_classes):
    """The cuts of the search that de-duplicates and sorts the train rows, with the generator seeded."""
    np.random.seed(seed)
    return oos_mdlp_cuts_njit(x, y, n_classes, 5, 6, 0.5, 3, 0.3)


@njit(cache=False)
def _presorted(x, y, order, seed, n_classes):
    """The cuts of the search that reads the train rows off ``order``, with the generator seeded."""
    np.random.seed(seed)
    return oos_mdlp_cuts_presorted_njit(x, y, order, n_classes, 5, 6, 0.5, 3, 0.3)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("signal", [0.0, 0.7])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_presorted_cuts_equal_the_sorting_search(n_classes, signal, dtype):
    """Noise and a planted dependence, several seeds: identical cut points."""
    rng = np.random.default_rng(0)
    n = 4000
    for seed in range(6):
        x = rng.standard_normal(n).astype(dtype)
        y = np.where(rng.random(n) < signal, (x > 0.3).astype(np.int64) * (n_classes - 1), rng.integers(0, n_classes, n)).astype(np.int64)
        order = np.argsort(x, kind="mergesort").astype(np.int32)
        a = _sorting(x, y, seed, n_classes)
        b = _presorted(x, y, order, seed, n_classes)
        assert np.array_equal(a, b), (seed, a, b)
        if signal:
            assert a.size > 0
