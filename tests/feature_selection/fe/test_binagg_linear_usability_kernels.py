"""The compiled row passes of the binned-aggregate linear-usability check equal their numpy forms."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _binned_agg_linear_usability as lu


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_standardised_equals_the_numpy_form(dtype):
    """Finite entries centred and scaled, non-finite ones 0, a constant column and an all-missing column all zeros; double precision out."""
    rng = np.random.default_rng(0)
    col = (3.0 + 2.0 * rng.standard_normal(5000)).astype(dtype)
    col[::17] = np.nan
    col[5] = np.inf
    got = lu._standardised(col)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, lu._standardised_numpy(col), rtol=2e-5 if dtype is np.float32 else 1e-12, atol=2e-6 if dtype is np.float32 else 1e-12)
    assert not lu._standardised(np.full(100, 2.5, dtype=dtype)).any()
    assert not lu._standardised(np.full(100, np.nan, dtype=dtype)).any()
    strided = np.ascontiguousarray(np.stack([col, col], axis=1))[:, 0]
    np.testing.assert_array_equal(lu._standardised(strided), got)


def _design(n: int = 3000) -> tuple:
    """A fold-sorted design (intercept and two columns), three class indicators, the fold starts and a cell per row."""
    rng = np.random.default_rng(1)
    folds = np.sort(rng.integers(0, 5, n))
    starts = np.concatenate([[0], np.cumsum(np.bincount(folds))])[:-1].astype(np.int64)
    base = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    y = rng.integers(0, 3, n)
    Y = (y[:, None] == np.arange(3)[None, :]).astype(np.float64)
    return base, Y, starts, rng.integers(0, 7, n).astype(np.int64)


def test_fold_grams_equal_the_numpy_form():
    """The per-fold normal-equation blocks, the squared sums and the totals about the fold means."""
    base, Y, starts, _cells = _design()
    for got, want in zip(lu._fold_grams(base, Y, starts), lu._fold_grams_numpy(base, Y, starts)):
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-11)


def test_cell_sums_equal_the_bincount_form():
    """Counts and per-(fold, cell) sums of every design and indicator column, to the bit (the same row order of additions)."""
    base, Y, starts, cells = _design()
    prep = {"base": base, "Y": Y, "starts": starts}
    got, want = lu._cell_sums(prep, cells, 7), lu._cell_sums_numpy(prep, cells, 7)
    assert set(got) == set(want)
    for key in want:
        np.testing.assert_array_equal(got[key], want[key])


def test_a_column_that_carries_the_cell_effect_is_kept_and_a_random_one_is_not():
    """End to end through the kernels: the cell-spread column of a target driven by the cell is kept, a random per-cell column is not."""
    rng = np.random.default_rng(2)
    n = 6000
    g = rng.integers(0, 12, n)
    effect = rng.standard_normal(12)
    y = (effect[g] + 0.4 * rng.standard_normal(n) > 0).astype(np.int64)
    X = pd.DataFrame({"g": g.astype(np.float32), "a": rng.standard_normal(n).astype(np.float32)})
    cands = {"signal": effect[g].astype(np.float32), "noise": rng.standard_normal(12)[g].astype(np.float32)}
    assert lu.linear_usability_keeps(X, ["g", "a"], cands, y, g, 0.01, "pair") == {"signal"}
