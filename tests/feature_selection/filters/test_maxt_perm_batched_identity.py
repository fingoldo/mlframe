"""The permutation-batched maxT pair-MI kernel is bit-identical to the per-shuffle batch_pair_mi_prange.

``pooled_pair_permutation_null_joint_mi_floor`` reuses each pair's joint encoding + x-marginal across all K
shuffles (only the (joint, y_perm) contingency re-runs). This must produce EXACTLY the same MI per (shuffle, pair)
as calling ``batch_pair_mi_prange`` once per shuffle, so the maxT floor is unchanged.
"""
import numpy as np

from mlframe.feature_selection.filters.info_theory import (
    batch_pair_mi_perm_batched,
    batch_pair_mi_prange,
)
from mlframe.feature_selection.filters._permutation_null import (
    pooled_pair_permutation_null_joint_mi_floor,
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
    floor = pooled_pair_permutation_null_joint_mi_floor(
        fd, nb, pa, pb, y, fy, n_permutations=25, quantile=0.95, random_seed=7
    )
    assert np.isfinite(floor) and floor >= 0.0
    # deterministic under a fixed seed
    floor2 = pooled_pair_permutation_null_joint_mi_floor(
        fd, nb, pa, pb, y, fy, n_permutations=25, quantile=0.95, random_seed=7
    )
    assert floor == floor2
