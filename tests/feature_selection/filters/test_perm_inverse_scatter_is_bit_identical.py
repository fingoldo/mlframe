"""Permutation inverses built by scatter must be bit-identical to ``np.argsort`` (mrmr_audit_2026-09-14 PERF-3).

Three FE permutation-null loops inverted each ``rng.permutation(n)`` with ``np.argsort`` -- an O(n log n) sort run
12 times per candidate -- where an O(n) scatter gives exactly the same array for a duplicate-free permutation.
These pin the invariant itself and one real call site end to end, so a later "simplification" that reintroduces
a tie-sensitive inverse, or reorders the RNG draws, is caught.
"""

import numpy as np
import pytest


@pytest.mark.parametrize("n", [1, 2, 7, 1000])
def test_scatter_inverse_equals_argsort(n):
    """``inv[perm] = arange`` must equal ``np.argsort(perm)`` exactly, element for element and in dtype."""
    arange = np.arange(n, dtype=np.int64)
    for seed in range(200):
        perm = np.random.default_rng(seed).permutation(n)
        inv = np.empty(n, dtype=np.int64)
        inv[perm] = arange
        ref = np.argsort(perm)
        assert inv.dtype == ref.dtype
        assert np.array_equal(inv, ref)


@pytest.mark.parametrize("n", [7, 1000])
def test_collapsed_gather_scatter_equals_argsort_gather(n):
    """``out[perm] = feat`` must equal ``feat[np.argsort(perm)]`` exactly -- the form the lattice/gate loops use."""
    for seed in range(200):
        rng = np.random.default_rng(seed)
        feat = rng.normal(size=n)
        perm = rng.permutation(n)
        out = np.empty(n, dtype=np.float64)
        out[perm] = feat
        assert np.array_equal(out, feat[np.argsort(perm)])


def test_inverse_index_matrix_built_column_by_column_equals_argsort():
    """The resident modular path fills an ``(n, n_perm)`` int64 inverse-index matrix; pin that exact 2-D form."""
    n, n_perm = 500, 12
    rng = np.random.default_rng(3)
    perms = [rng.permutation(n) for _ in range(n_perm)]
    arange = np.arange(n, dtype=np.int64)
    got = np.empty((n, n_perm), dtype=np.int64)
    ref = np.empty((n, n_perm), dtype=np.int64)
    for i, perm in enumerate(perms):
        got[np.asarray(perm), i] = arange
        ref[:, i] = np.argsort(np.asarray(perm))
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_integer_lattice_perm_null_hi_is_bit_identical_to_the_argsort_form(seed):
    """End to end on a real call site: the null band must match an argsort reference exactly, not approximately.

    The reference replays the original loop verbatim -- same RNG, same draw order, same batched MI call -- so the
    only thing that differs is how the permutation is inverted.
    """
    from mlframe.feature_selection.filters import _integer_lattice_fe as IL
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    rng = np.random.default_rng(100 + seed)
    n = 600
    feat = rng.normal(size=n)
    y = (feat + rng.normal(scale=0.8, size=n) > 0).astype(np.int64)
    nbins, n_perm, z = 8, 12, 3.0

    got = IL._perm_null_hi(feat, y, nbins, n_perm=n_perm, seed=seed, z=z)

    ref_rng = np.random.default_rng(seed)
    yi = IL.encode_y_for_classif_mi(y)
    mat = np.empty((yi.size, n_perm), dtype=np.float64)
    for i in range(n_perm):
        perm = ref_rng.permutation(yi.size)
        mat[:, i] = feat[np.argsort(perm)]
    vals = np.asarray(_mi_classif_batch(mat, yi, nbins=nbins), dtype=np.float64)
    ref = float(vals.mean() + z * vals.std())

    assert got == ref


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_conditional_gate_perm_null_hi_is_bit_identical_to_the_argsort_form(seed):
    """The same end-to-end pin for the conditional-gate sibling's host-ndarray branch, which is the batched one.

    A resident cupy handle takes a different, unedited per-perm branch; a host ndarray is what reaches the scatter.
    """
    from mlframe.feature_selection.filters import _conditional_gate_fe as CG
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    rng = np.random.default_rng(200 + seed)
    n = 600
    feat = rng.normal(size=n)
    y = (feat + rng.normal(scale=0.8, size=n) > 0).astype(np.int64)
    nbins, n_perm, z = 8, 12, 3.0

    got = CG._perm_null_hi(feat, y, nbins, n_perm=n_perm, seed=seed, z=z)

    ref_rng = np.random.default_rng(seed)
    yi = CG.encode_y_for_classif_mi(y)
    n_y = int(yi.size)
    feat_host = np.ascontiguousarray(feat, dtype=np.float64).ravel()
    mat = np.empty((n_y, n_perm), dtype=np.float64)
    for i in range(n_perm):
        perm = ref_rng.permutation(n_y)
        mat[:, i] = feat_host[np.argsort(perm)]
    vals = np.asarray(_mi_classif_batch(mat, yi, nbins=nbins), dtype=np.float64)
    ref = float(vals.mean() + z * vals.std())

    assert got == ref
