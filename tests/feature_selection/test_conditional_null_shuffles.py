"""The conditional permutation null builds its within-stratum shuffles in one parallel kernel per block; every shuffle and the null it gives are those of the per-permutation loop."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_cmi_redundancy_null import _conditional_cpu_null, _NullInputs, _StratifiedSupport
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import cmi_from_binned_fixed_yz
from mlframe.feature_selection.filters._numba_utils import within_stratum_shuffles


def test_kernel_rows_equal_the_stable_argsort_shuffle():
    """Each row is the candidate permuted by ``np.argsort(z_rank + keys, kind="stable")`` inside its stratum."""
    rng = np.random.default_rng(0)
    n = 5000
    z = rng.integers(0, 12, n)
    order = np.argsort(z, kind="stable")
    sorted_z = z[order]
    z_rank = np.zeros(n, dtype=np.float32)
    z_rank[1:] = np.cumsum(sorted_z[1:] != sorted_z[:-1])
    x = rng.integers(0, 7, n).astype(np.int64)
    x_sorted = x[order]
    keys = rng.random((9, n))
    out = np.empty((9, n), dtype=np.int64)
    within_stratum_shuffles(keys, z_rank, order, x_sorted, out)
    for i in range(9):
        want = np.empty_like(x)
        want[order] = x_sorted[np.argsort(z_rank + keys[i], kind="stable")]
        assert np.array_equal(out[i], want), i


@pytest.mark.parametrize("nperm", [5, 64, 70])
def test_cpu_null_equals_the_per_permutation_loop(nperm):
    """The null floor and mean from the block kernel equal those of the loop that draws the keys and shuffles one permutation at a time."""
    rng = np.random.default_rng(1)
    n = 3000
    cand = rng.integers(0, 6, n).astype(np.int64)
    y = rng.integers(0, 3, n).astype(np.int64)
    z = rng.integers(0, 10, n).astype(np.int64)

    def fresh():
        """A null input and its stratified support on the same data and seed."""
        inp = _NullInputs(cand, y, 7, 3)
        return inp, _StratifiedSupport(inp, z)

    inp, sup = fresh()
    got = _conditional_cpu_null(inp, sup, nperm, 0.95)
    inp2, sup2 = fresh()
    xh = inp2.host_x()
    x_sorted = xh[sup2.order]
    nulls = np.empty(nperm)
    for i in range(nperm):
        keys = inp2.rng.random(inp2.n_size)
        within = np.argsort(sup2.z_rank + keys, kind="stable")
        x_perm = np.empty_like(xh)
        x_perm[sup2.order] = x_sorted[within]
        nulls[i] = float(cmi_from_binned_fixed_yz(x_perm, sup2.y_i, sup2.z_i, sup2.h_yz, sup2.h_z, sup2.k_yz, sup2.k_z, sup2.n_f))
    assert got == (float(np.quantile(nulls, 0.95)), float(np.mean(nulls)))
