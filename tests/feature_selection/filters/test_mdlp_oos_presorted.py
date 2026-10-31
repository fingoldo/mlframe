"""The held-out MDLP cut search reads the distinct pairs and the train part of a column off the column's sort order; the cuts are those of the search that sorts the rows itself."""

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
def test_presorted_cuts_equal_the_sorting_search_without_ties(n_classes, signal):
    """Noise and a planted dependence on a column without tied values, several seeds: identical cut points to the sorting search."""
    rng = np.random.default_rng(0)
    n = 4000
    for seed in range(6):
        x = rng.standard_normal(n)
        y = np.where(rng.random(n) < signal, (x > 0.3).astype(np.int64) * (n_classes - 1), rng.integers(0, n_classes, n)).astype(np.int64)
        order = np.argsort(x, kind="mergesort").astype(np.int32)
        a = _sorting(x[order], y[order], seed, n_classes)  # the sorted rows are the de-duplicated pairs of a column without ties
        b = _presorted(x, y, order, seed, n_classes)
        assert np.array_equal(a, b), (seed, a, b)
        if signal:
            assert a.size > 0


@pytest.mark.parametrize("n_classes", [2, 3])
def test_presorted_cuts_do_not_depend_on_how_tied_rows_are_ordered(n_classes):
    """With tied values (rounded and repeated ones) the distinct sorted pairs are canonical: any ascending order of the rows gives the same cuts, and they lie inside the data."""
    rng = np.random.default_rng(1)
    n = 4000
    for seed in range(6):
        x = np.round(rng.standard_normal(n), 1) if seed % 2 else np.where(rng.random(n) < 0.03, 0.5, rng.standard_normal(n))
        x = x.astype(np.float32)
        y = np.where(rng.random(n) < 0.7, (x > 0.3).astype(np.int64) * (n_classes - 1), rng.integers(0, n_classes, n)).astype(np.int64)
        a = _presorted(x, y, np.argsort(x, kind="mergesort").astype(np.int32), seed, n_classes)
        shuffled = rng.permutation(n)
        alt = shuffled[np.argsort(x[shuffled], kind="mergesort")].astype(np.int32)  # ties in a different row order
        b = _presorted(x, y, alt, seed, n_classes)
        assert np.array_equal(a, b), seed
        assert a.size == 0 or (np.all(np.diff(a) > 0) and x.min() <= a.min() and a.max() <= x.max())
