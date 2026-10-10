"""The all-folds moment kernels of the binned-aggregate OOF build equal the per-fold computation, and decline when the tables do not fit in shared memory."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _binned_numeric_agg_resident as res
from mlframe.feature_selection.filters._cell_moments_kernel import fold_cell_moments


def _inputs(n=20000, n_cells=12, n_folds=5, seed=0):
    """Cell codes, a value column with non-finite entries (zeroed, with the 0/1 finite weight), and fold ids on the device."""
    rng = np.random.default_rng(seed)
    codes = cp.asarray(rng.integers(0, n_cells, n), dtype=cp.int64)
    v = rng.standard_normal(n) * 3.0 + 100.0
    v[::97] = np.nan
    v_g = cp.asarray(v)
    finite = cp.isfinite(v_g)
    return codes, cp.where(finite, v_g, 0.0), finite.astype(cp.float64), cp.asarray(rng.integers(0, n_folds, n), dtype=cp.int64)


def test_all_folds_tables_equal_the_per_fold_moments():
    """Row f of every table holds the moments of the train rows of fold f, as the per-fold masked kernels compute them."""
    codes, v_safe, finite_f, folds = _inputs()
    tabs = fold_cell_moments(cp, codes, v_safe, finite_f, folds, 5, 12)
    assert tabs is not None
    for f in range(5):
        w = cp.where(folds != f, finite_f, 0.0)
        ref = res._per_cell_moments_stable_masked_gpu(cp, codes, v_safe, w, 12)
        for got, want in zip(tabs, ref):
            np.testing.assert_allclose(cp.asnumpy(got[f]), cp.asnumpy(want), rtol=1e-9, atol=1e-9)


def test_declines_when_the_tables_do_not_fit_in_shared_memory():
    """3 * n_folds * n_cells doubles above the 48 KB block budget returns None so the caller takes the per-fold form."""
    codes, v_safe, finite_f, folds = _inputs(n=500, n_cells=2000)
    assert fold_cell_moments(cp, codes, v_safe, finite_f, folds, 5, 2000) is None


@pytest.mark.parametrize("stat", ["mean", "std", "skew", "kurt"])
def test_stat_tables_agree_between_the_fused_and_the_per_fold_form(monkeypatch, stat):
    """The pair's stat tables are the same whether the moments come from the all-folds kernels or from the per-fold fallback."""
    codes, v_safe, finite_f, folds = _inputs(seed=3)
    fused = res._pair_fold_stat_tables(cp, codes, v_safe, finite_f, folds, 5, 12, [stat])[stat]
    monkeypatch.setattr("mlframe.feature_selection.filters._cell_moments_kernel.fold_cell_moments", lambda *a, **k: None)
    fallback = res._pair_fold_stat_tables(cp, codes, v_safe, finite_f, folds, 5, 12, [stat])[stat]
    assert fused.shape == (5, 12)
    np.testing.assert_allclose(cp.asnumpy(fused), cp.asnumpy(fallback), rtol=1e-8, atol=1e-9, equal_nan=True)
