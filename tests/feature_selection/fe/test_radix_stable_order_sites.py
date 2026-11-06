"""The two stable-sort sites that now take the radix order return the permutation of the stable comparison sort, signed zeros, ties and NaN included."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_batch_corr import finite_sort_order
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_helpers import cached_stable_argsort


def _column(dtype, with_nan):
    """A column with ties, both zeros and (optionally) NaN / inf."""
    rng = np.random.default_rng(5)
    x = np.round(rng.normal(size=4000) * 3, 1).astype(dtype)
    x[::17] = 0.0
    x[3::29] = -0.0
    if with_nan:
        x[::101] = np.nan
        x[7::211] = np.inf
        x[9::223] = -np.inf
    return x


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("with_nan", [False, True])
def test_finite_sort_order_equals_the_stable_comparison_sort(dtype, with_nan):
    """Row ids of the finite values in ascending order, ties in row order."""
    x = _column(dtype, with_nan)
    rows = np.flatnonzero(np.isfinite(x))
    ref = rows[np.argsort(x[rows], kind="mergesort")]
    assert np.array_equal(finite_sort_order(x, cache=False), ref)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("with_nan", [False, True])
def test_cached_stable_argsort_equals_np_argsort_stable(dtype, with_nan):
    """With the cache off (the default) the order is the stable argsort for finite floats (radix) and for a vector with NaN or an integer type (comparison sort)."""
    if dtype == np.int64 and with_nan:
        pytest.skip("no NaN in an integer vector")
    x = _column(dtype, with_nan) if dtype != np.int64 else (np.arange(4000) % 37).astype(np.int64)
    assert np.array_equal(cached_stable_argsort(x), np.argsort(x, kind="stable"))
