"""The occupied joint-bin count of the pair pool (the Miller-Madow bias of the order-2 maxT floor) is exact in its fast form and is not computed twice by the floor."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _permutation_null as pn


def _pool(n=4000, p=14, seed=0):
    """A binned matrix with mixed cardinalities, nested columns (tables that cannot fill) and constants; the pair pool of all its pairs."""
    rng = np.random.default_rng(seed)
    nbins = rng.integers(2, 13, size=p).astype(np.int64)
    data = np.column_stack([rng.integers(0, int(k), size=n) for k in nbins]).astype(np.int8)
    data[:, 1] = data[:, 0] // 2  # nested: column 1 is a coarsening of column 0, so most joint cells stay empty
    nbins[1] = int(nbins[0] // 2 + 1)
    data[:, 2] = 0  # a constant column: one occupied row of the table
    nbins[2] = 3
    ia, ib = np.triu_indices(p, 1)
    return data, nbins, ia.astype(np.int64), ib.astype(np.int64)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64])
def test_the_fast_count_equals_the_serial_kernel_and_a_set_count(dtype):
    """Integer counts: the parallel column-major kernel with the early exit gives the serial row-major kernel's counts and the distinct-pair count of numpy, for nested, constant and dense pairs."""
    data, nbins, ia, ib = _pool()
    data = data.astype(dtype)
    fast = pn._pairwise_occupied_joint_k(data, ia, ib, nbins)
    serial = np.asarray(pn._pairwise_occupied_joint_k_njit(data, ia, ib, nbins))
    reference = np.array([np.unique(data[:, a].astype(np.int64) * int(nbins[b]) + data[:, b]).size for a, b in zip(ia, ib)])
    assert np.array_equal(fast, serial)
    assert np.array_equal(fast, reference)
    constant = (ia == 2) | (ib == 2)
    assert np.array_equal(fast[constant], np.array([np.unique(data[:, b if a == 2 else a]).size for a, b in zip(ia[constant], ib[constant])]))  # a constant column adds no cell
    assert (fast < nbins[ia] * nbins[ib]).any()  # tables that cannot fill (nested pair) are counted to the end of the rows, not cut short


def test_an_empty_pool_has_no_counts():
    """No pairs: an empty count vector, no kernel launch."""
    data, nbins, _ia, _ib = _pool()
    empty = np.empty(0, dtype=np.int64)
    assert pn._pairwise_occupied_joint_k(data, empty, empty, nbins).shape == (0,)


def test_the_floor_takes_the_callers_bias_without_counting_again(monkeypatch):
    """With ``mm_debias`` the floor reads the per-pair bias the caller passes: the same floor as when it counts the cells itself, and no second count of the pool."""
    data, nbins, ia, ib = _pool(n=6000, p=12, seed=3)
    rng = np.random.default_rng(1)
    y = (data[:, 3] + rng.integers(0, 3, size=data.shape[0]) > 6).astype(np.int8)
    fy = np.bincount(y, minlength=2) / y.size
    kwargs = dict(factors_data=data, nbins=nbins, pair_a=ia, pair_b=ib, classes_y=y, freqs_y=fy, n_permutations=10, quantile=0.95, random_seed=0, mm_debias=True)
    bias = pn.pairwise_mm_joint_bias(data, ia, ib, nbins, 2)
    own = pn.pooled_pair_permutation_null_joint_mi_floor(**kwargs)
    pn._PAIR_FLOOR_MEMO.clear()
    calls = []
    original = pn._pairwise_occupied_joint_k
    monkeypatch.setattr(pn, "_pairwise_occupied_joint_k", lambda *a, **k: calls.append(1) or original(*a, **k))
    given = pn.pooled_pair_permutation_null_joint_mi_floor(**kwargs, mm_bias=bias)
    assert given == own
    assert calls == []
    assert pn.pooled_pair_permutation_null_joint_mi_floor(**kwargs, mm_bias=bias[:-1]) == own  # a vector of the wrong length is ignored, the cells are counted
    assert calls == [1]
