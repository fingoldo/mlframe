"""The batched permutation null of ``_count_nfailed_joint_indep_cupy`` counts the same failed permutations as the per-trial loop it replaced (same RandomState(base_seed + p) draws), for block sizes of one and many, and at thresholds inside the null's range."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import _need_cuda

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _inputs(n, kp, k1, k2, ky, seed):
    """Class codes of a pair, its two members and the target, with their relative frequencies."""
    rng = np.random.default_rng(seed)
    c1 = rng.integers(0, k1, n)
    c2 = rng.integers(0, k2, n)
    cp_ = (c1 * k2 + c2) % kp
    y = (rng.random(n) < 0.3 + 0.2 * (c1 % 2)).astype(np.int64) % ky

    def fr(c, k):
        """Relative class frequencies."""
        return np.bincount(c, minlength=k) / n

    return cp_.astype(np.int64), c1.astype(np.int64), c2.astype(np.int64), fr(cp_, kp), fr(c1, k1), fr(c2, k2), y, fr(y, ky)


def _loop_reference(pair, x1, x2, fp, f1, f2, y, fy, ii_obs, n_perms, base_seed):
    """The per-trial loop the batched version replaced."""
    N = y.size
    kp, k1, k2, ky = fp.size, f1.size, f2.size, fy.size
    yg = cp.asarray(y)
    dens = [cp.asarray(a[:, None] * fy[None, :]) for a in (fp, f1, f2)]
    failed = 0
    for p in range(n_perms):
        yp = cp.random.RandomState(base_seed + p).permutation(yg)
        terms = []
        for cls, K, den in ((pair, kp, dens[0]), (x1, k1, dens[1]), (x2, k2, dens[2])):
            joint = cp.bincount(cp.asarray(cls) * ky + yp, minlength=K * ky).reshape(K, ky) * (1.0 / N)
            terms.append(float(cp.where(joint > 0, joint * cp.log(joint / cp.maximum(den, 1e-300)), 0.0).sum()))
        failed += int((terms[0] - terms[1] - terms[2]) >= ii_obs)
    return failed


@pytest.mark.parametrize("n,kp,k1,k2,ky", [(3000, 12, 4, 6, 2), (20_000, 30, 5, 7, 3)])
@pytest.mark.parametrize("n_perms", [1, 7, 40])
def test_counts_equal_the_loop(n, kp, k1, k2, ky, n_perms):
    """Same failed count at a threshold at the middle of the null, at zero and above everything; a tiny block forces several blocks."""
    from mlframe.feature_selection.filters._cat_confirm_permutation import _count_nfailed_joint_indep_cupy

    args = _inputs(n, kp, k1, k2, ky, 3)
    pair, x1, x2, fp, f1, f2, y, fy = args
    probe = [_loop_reference(pair, x1, x2, fp, f1, f2, y, fy, thr, 40, 11) for thr in (-1.0, 0.0, 1.0)]
    assert probe[0] == 40 and probe[2] == 0
    for thr in (-1.0, 0.0, 0.002, 1.0):
        want = _loop_reference(pair, x1, x2, fp, f1, f2, y, fy, thr, n_perms, 11)
        got = _count_nfailed_joint_indep_cupy(pair, fp, x1, f1, x2, f2, y, fy, ii_obs=thr, n_perms=n_perms, base_seed=11)
        assert got == want, (thr, got, want)
