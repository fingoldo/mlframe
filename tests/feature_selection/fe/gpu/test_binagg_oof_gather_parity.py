"""The one-pass OOF assembly of the binned-aggregate candidates equals the per-fold loop it replaced, value for value."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _binned_numeric_agg_resident as res


def _tables(rng, n_folds, n_cells, empty_frac):
    """Per-fold cell stat tables with some empty (NaN) cells, as the moment pass yields."""
    out = []
    for _ in range(n_folds):
        t = rng.normal(size=n_cells)
        t[rng.random(n_cells) < empty_frac] = np.nan
        out.append(cp.asarray(t))
    return out


@pytest.mark.parametrize("n,n_folds,n_cells,empty_frac", [(10_000, 5, 12, 0.0), (50_000, 5, 40, 0.3), (7, 3, 4, 0.5), (1_000, 2, 1, 1.0)])
def test_gather_matches_the_fold_loop(n, n_folds, n_cells, empty_frac):
    """Same column for ordinary, sparse (many empty train cells), tiny and all-empty tables."""
    rng = np.random.default_rng(3)
    codes = cp.asarray(rng.integers(0, n_cells, size=n).astype(np.int64))
    fold_ids = cp.asarray(rng.permutation(n).astype(np.int64) % n_folds)
    tables = _tables(rng, n_folds, n_cells, empty_frac)
    fallback = 0.37
    expected = res._oof_column_foldloop(cp, tables, fold_ids, codes, n, fallback)
    got = res._oof_column_gather(cp, tables, fold_ids * n_cells + codes, fallback)
    assert np.array_equal(cp.asnumpy(expected), cp.asnumpy(got))


def test_inf_in_a_stat_table_falls_back_like_nan():
    """A non-finite stat (inf from a degenerate cell) is replaced by the fallback exactly as in the loop."""
    n, n_folds, n_cells = 100, 2, 3
    codes = cp.asarray(np.arange(n, dtype=np.int64) % n_cells)
    fold_ids = cp.asarray(np.arange(n, dtype=np.int64) % n_folds)
    tables = [cp.asarray([1.0, np.inf, np.nan]), cp.asarray([-np.inf, 2.0, 3.0])]
    expected = res._oof_column_foldloop(cp, tables, fold_ids, codes, n, -5.0)
    got = res._oof_column_gather(cp, tables, fold_ids * n_cells + codes, -5.0)
    assert np.array_equal(cp.asnumpy(expected), cp.asnumpy(got))
