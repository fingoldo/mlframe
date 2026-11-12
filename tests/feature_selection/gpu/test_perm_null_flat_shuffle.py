"""The flat-key within-stratum shuffle of the GPU conditional permutation null permutes ``x`` inside each stratum, per column, and gives the same null.

One 1-D radix sort of ``column | stratum rank | random bits | x`` replaces the per-column argsort; the sorted keys carry the shuffled x in their low bits.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as m


def _case(n=4000, strata=37, kx=10, seed=0):
    """A z-sorted candidate: strata of unequal size, x codes 0..kx-1, dense stratum ranks."""
    rng = np.random.default_rng(seed)
    z = np.sort(rng.integers(0, strata, n))
    x = rng.integers(0, kx, n)
    rank = np.concatenate([[0], np.cumsum(z[1:] != z[:-1])]).astype(np.float64)
    return z, x, rank


def test_every_column_permutes_x_within_each_stratum():
    """Per stratum and per column the multiset of x is unchanged, the columns differ, and nothing leaves its stratum."""
    z, x, rank = _case()
    nperm = 12
    out = m._flat_within_stratum_shuffle(cp, cp.random.RandomState(3), cp.asarray(x), cp.asarray(rank)[:, None], nperm, int(x.max()) + 1, int(rank[-1]) + 1)
    out = cp.asnumpy(out)
    assert out.shape == (nperm, x.size)
    for j in range(nperm):
        for s in np.unique(z):
            m_ = z == s
            assert np.array_equal(np.sort(out[j, m_]), np.sort(x[m_]))
    assert len({out[j].tobytes() for j in range(nperm)}) == nperm
    assert (out != x[None, :]).any()


def test_a_key_that_does_not_fit_falls_back():
    """With too many strata and values for 20 random bits the helper answers None, so the caller keeps the argsort path."""
    _z, x, rank = _case(n=64)
    assert m._flat_within_stratum_shuffle(cp, cp.random.RandomState(0), cp.asarray(x), cp.asarray(rank)[:, None], 1 << 20, 1 << 20, 1 << 20) is None


def test_the_null_matches_an_exact_host_shuffle_in_distribution():
    """The CMI null of the flat shuffles has the mean of exact within-stratum host permutations, to four standard errors of the difference."""
    rng = np.random.default_rng(1)
    n, nperm = 20000, 400
    z, x, y = rng.integers(0, 60, n), rng.integers(0, 10, n), rng.integers(0, 2, n)
    z_dev, x_dev = cp.asarray(z), cp.asarray(x)
    z_for_cmi, order_d, z_rank_d = m._stratum_layout(cp, z_dev, None, None, n)
    flat = m._flat_within_stratum_shuffle(cp, cp.random.RandomState(7), x_dev[order_d], z_rank_d, nperm, 10, int(z_rank_d[-1, 0]) + 1)
    xp = cp.empty((n, nperm), dtype=cp.int64)
    xp[order_d, :] = flat.T
    got = cp.asnumpy(m._batched_cmi_resident_chunked(xp, y, z_for_cmi))
    host_rng = np.random.default_rng(0)
    strata = [np.flatnonzero(z == s) for s in range(60)]
    exact = np.empty((n, nperm), dtype=np.int64)
    for j in range(nperm):
        col = x.copy()
        for ix in strata:
            col[ix] = host_rng.permutation(x[ix])
        exact[:, j] = col
    ref = cp.asnumpy(m._batched_cmi_resident_chunked(cp.asarray(exact), y, z_for_cmi))
    se = np.sqrt(got.var() / nperm + ref.var() / nperm)
    assert abs(got.mean() - ref.mean()) < 4 * se


def test_a_shared_xz_entropy_gives_the_same_cmi_for_within_stratum_shuffles():
    """Shuffles inside the strata of z keep the (x, z) joint, so ``precomp_xz`` computed on one of them gives every column's CMI exactly as the per-column count does."""
    from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu, joint_entropy_gpu

    rng = np.random.default_rng(4)
    n, nperm, kz = 6000, 9, 700
    z, x, y = rng.integers(0, kz, n), rng.integers(0, 8, n), rng.integers(0, 3, n)
    z_dev = cp.asarray(z)
    _zc, order_d, z_rank_d = m._stratum_layout(cp, z_dev, None, None, n)
    flat = m._flat_within_stratum_shuffle(cp, cp.random.RandomState(1), cp.asarray(x)[order_d], z_rank_d, nperm, 8, int(z_rank_d[-1, 0]) + 1)
    xp = cp.empty((n, nperm), dtype=cp.int64)
    xp[order_d, :] = flat.T
    kz_eff = int(z.max()) + 1
    h, k = joint_entropy_gpu([xp[:, 0] * kz_eff + z_dev], [8 * kz_eff], 1.0 / n)
    with_pre = cp.asnumpy(batched_cmi_gpu(xp, y, z_dev, kx=8, precomp_xz=(float(h), int(k)), return_device=True))
    without = cp.asnumpy(batched_cmi_gpu(xp, y, z_dev, kx=8, return_device=True))
    np.testing.assert_allclose(with_pre, without, rtol=0, atol=1e-12)
