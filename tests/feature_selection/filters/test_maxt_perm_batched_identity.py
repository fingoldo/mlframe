"""The permutation-batched maxT pair-MI kernel is bit-identical to the per-shuffle batch_pair_mi_prange.

``pooled_pair_permutation_null_joint_mi_floor`` reuses each pair's joint encoding + x-marginal across all K
shuffles (only the (joint, y_perm) contingency re-runs). This must produce EXACTLY the same MI per (shuffle, pair)
as calling ``batch_pair_mi_prange`` once per shuffle, so the maxT floor is unchanged.
"""

import numpy as np

from mlframe.feature_selection.filters.info_theory import (
    batch_pair_mi_perm_batched,
    batch_pair_mi_prange,
    batch_triple_mi_perm_batched,
    batch_triple_mi_prange,
)
from mlframe.feature_selection.filters._permutation_null import (
    pooled_pair_permutation_null_joint_mi_floor,
    pooled_triple_permutation_null_joint_mi_floor,
)


def test_perm_batched_bit_identical_to_per_shuffle():
    rng = np.random.default_rng(0)
    n, ncols, nbins, K = 8000, 12, 12, 25
    fd = rng.integers(0, nbins, (n, ncols)).astype(np.int32)
    nb = np.full(ncols, nbins, dtype=np.int64)
    pairs = [(i, j) for i in range(ncols) for j in range(i + 1, ncols)]
    pa = np.array([p[0] for p in pairs], np.int64)
    pb = np.array([p[1] for p in pairs], np.int64)
    y0 = rng.integers(0, 3, n).astype(np.int64)
    fy = np.bincount(y0, minlength=3).astype(np.float64) / n

    rng2 = np.random.default_rng(42)
    yp = y0.copy()
    y_perms = np.empty((K, n), dtype=np.int64)
    ref = np.empty((K, pa.shape[0]), dtype=np.float64)
    for k in range(K):
        rng2.shuffle(yp)
        y_perms[k] = yp
        ref[k] = batch_pair_mi_prange(fd, pa, pb, nb, yp, fy)

    got = batch_pair_mi_perm_batched(fd, pa, pb, nb, y_perms, fy)
    assert np.array_equal(got, ref), f"max|d|={np.max(np.abs(got - ref)):.2e}"


def test_pooled_pair_floor_runs_and_is_nonneg():
    rng = np.random.default_rng(1)
    n, ncols, nbins = 6000, 10, 10
    fd = rng.integers(0, nbins, (n, ncols)).astype(np.int32)
    nb = np.full(ncols, nbins, dtype=np.int64)
    pa = np.array([0, 1, 2, 3], np.int64)
    pb = np.array([4, 5, 6, 7], np.int64)
    y = rng.integers(0, 2, n).astype(np.int64)
    fy = np.bincount(y, minlength=2).astype(np.float64) / n
    floor = pooled_pair_permutation_null_joint_mi_floor(fd, nb, pa, pb, y, fy, n_permutations=25, quantile=0.95, random_seed=7)
    assert np.isfinite(floor) and floor >= 0.0
    # deterministic under a fixed seed
    floor2 = pooled_pair_permutation_null_joint_mi_floor(fd, nb, pa, pb, y, fy, n_permutations=25, quantile=0.95, random_seed=7)
    assert floor == floor2


def test_triple_perm_batched_bit_identical_to_per_shuffle():
    """Order-3 sibling of test_perm_batched_bit_identical_to_per_shuffle (Wave 13 finding #2):
    batch_triple_mi_perm_batched hoists the permutation-invariant dense-renumbered triple code out of the
    K-shuffle loop, and must match calling batch_triple_mi_prange once per shuffle exactly."""
    rng = np.random.default_rng(2)
    n, ncols, nbins, K = 4000, 9, 8, 20
    fd = rng.integers(0, nbins, (n, ncols)).astype(np.int32)
    nb = np.full(ncols, nbins, dtype=np.int64)
    triples = [(i, j, k) for i in range(ncols) for j in range(i + 1, ncols) for k in range(j + 1, ncols)]
    ta = np.array([t[0] for t in triples], np.int64)
    tb = np.array([t[1] for t in triples], np.int64)
    tc = np.array([t[2] for t in triples], np.int64)
    y0 = rng.integers(0, 3, n).astype(np.int64)
    fy = np.bincount(y0, minlength=3).astype(np.float64) / n

    rng2 = np.random.default_rng(43)
    yp = y0.copy()
    y_perms = np.empty((K, n), dtype=np.int64)
    ref = np.empty((K, ta.shape[0]), dtype=np.float64)
    for k in range(K):
        rng2.shuffle(yp)
        y_perms[k] = yp
        ref[k] = batch_triple_mi_prange(fd, ta, tb, tc, nb, yp, fy)

    got = batch_triple_mi_perm_batched(fd, ta, tb, tc, nb, y_perms, fy)
    assert np.array_equal(got, ref), f"max|d|={np.max(np.abs(got - ref)):.2e}"


def test_pooled_triple_floor_runs_and_is_nonneg():
    rng = np.random.default_rng(3)
    n, ncols, nbins = 5000, 9, 8
    fd = rng.integers(0, nbins, (n, ncols)).astype(np.int32)
    nb = np.full(ncols, nbins, dtype=np.int64)
    ta = np.array([0, 1, 2], np.int64)
    tb = np.array([3, 4, 5], np.int64)
    tc = np.array([6, 7, 8], np.int64)
    y = rng.integers(0, 2, n).astype(np.int64)
    fy = np.bincount(y, minlength=2).astype(np.float64) / n
    floor = pooled_triple_permutation_null_joint_mi_floor(fd, nb, ta, tb, tc, y, fy, n_permutations=20, quantile=0.95, random_seed=11)
    assert np.isfinite(floor) and floor >= 0.0
    floor2 = pooled_triple_permutation_null_joint_mi_floor(fd, nb, ta, tb, tc, y, fy, n_permutations=20, quantile=0.95, random_seed=11)
    assert floor == floor2
