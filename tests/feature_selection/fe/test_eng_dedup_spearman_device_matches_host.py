"""The device masked-Spearman kernel gives the host kernel's |rho| on pairs with NaN rows, ties, constant columns and too-short overlaps."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("numba")

from numba.typed import List as TypedList

from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_batch_corr import finite_sort_order, masked_abs_spearman_presorted
from mlframe.feature_selection.filters._mrmr_fit_impl import _eng_dedup_spearman_gpu as gpu


def _frame(n, seed):
    """A candidate with NaN rows and kept columns: copies, ties, independent, constant, and one with fewer than 8 finite rows."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n).astype(np.float32)
    a[rng.random(n) < 0.06] = np.nan
    cols = [
        (a * 2.0 + 1.0).astype(np.float32),  # monotone copy: |rho| 1
        (-a).astype(np.float32),  # reversed copy: |rho| 1
        rng.normal(size=n).astype(np.float32),  # independent
        np.round(a, 1).astype(np.float32),  # heavy ties, near copy
        np.full(n, 3.0, dtype=np.float32),  # constant
        np.where(rng.random(n) < 0.999, np.nan, 1.0).astype(np.float32),  # < 8 finite rows
        (np.round(a * 3) + rng.integers(0, 2, n)).astype(np.float32),
    ]
    cols[2][rng.random(n) < 0.1] = np.nan
    cols[6][rng.random(n) < 0.2] = np.nan
    return a, cols


def _host(a, cols, rel_tol):
    """The host kernel's |rho|."""
    others, orders = TypedList(), TypedList()
    for c in cols:
        others.append(c)
        orders.append(finite_sort_order(c, cache=False))
    out = np.empty(len(cols))
    masked_abs_spearman_presorted(a, np.isfinite(a), finite_sort_order(a, cache=False), others, orders, rel_tol, out)
    return out


@pytest.mark.parametrize("n,seed", [(3000, 0), (20011, 1)])
def test_device_matches_host(n, seed):
    """Same |rho| as the host kernel, skipped pairs included."""
    a, cols = _frame(n, seed)
    want = _host(a, cols, 1e-12)
    got = np.full(len(cols), 7.0)
    assert gpu.masked_abs_spearman_device(a, finite_sort_order(a, cache=False), cols, [finite_sort_order(c, cache=False) for c in cols], 1e-12, got)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    assert (want == -1.0).sum() >= 2 and (want > 0.99).sum() >= 2  # the frame exercises both the skip rules and the collisions
    gpu.clear_device_column_cache()


def test_second_candidate_reuses_cached_columns():
    """A second candidate finds the kept columns on the device already, and the cache can be cleared."""
    a, cols = _frame(5000, 3)
    orders = [finite_sort_order(c, cache=False) for c in cols]
    for s in (0, 1):
        cand = np.roll(a, s * 17)
        got = np.empty(len(cols))
        assert gpu.masked_abs_spearman_device(cand, finite_sort_order(cand, cache=False), cols, orders, 1e-12, got)
        np.testing.assert_allclose(got, _host(cand, cols, 1e-12), rtol=1e-9, atol=1e-9)
    assert len(gpu._COLUMN_CACHE) == len(cols)
    gpu.clear_device_column_cache()
    assert not gpu._COLUMN_CACHE
