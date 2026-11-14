"""The stratum layout of a device-born support is computed once per support array and the nulls it feeds do not change."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as M


def _z(seed, n=4000, k=30):
    """A device support with ``k`` strata."""
    return cp.asarray(np.random.default_rng(seed).integers(0, k, n).astype(np.int64))


def test_same_support_returns_the_cached_layout_and_equals_a_fresh_one():
    """An identity match returns the same tuple; a recomputation gives the same arrays."""
    z = _z(1)
    first = M._stratum_layout(cp, z, None, None, z.size)
    assert M._stratum_layout(cp, z, None, None, z.size) is first
    fresh = M._stratum_layout_uncached(cp, z, None, None, z.size)
    for a, b in zip(first, fresh):
        assert bool((a == b).all())


def test_a_different_support_gets_its_own_layout_and_the_cache_is_bounded():
    """Another array (equal content or not) is a miss; only the last ``_LAYOUT_CACHE_MAX`` supports are kept."""
    zs = [_z(s) for s in range(M._LAYOUT_CACHE_MAX + 2)]
    lays = [M._stratum_layout(cp, z, None, None, z.size) for z in zs]
    assert len({id(lay) for lay in lays}) == len(zs)
    assert len(M._LAYOUT_CACHE) == M._LAYOUT_CACHE_MAX


def test_null_is_unchanged_by_the_cache():
    """The same (seed, salt) null from a cold and from a warm layout cache is identical."""
    rng = np.random.default_rng(3)
    n = 4000
    z = _z(4, n)
    x = cp.asarray(rng.integers(0, 8, n).astype(np.int64))
    y = rng.integers(0, 2, n)
    M._LAYOUT_CACHE.clear()
    cold = M.conditional_perm_null_gpu(x, y, z, order=None, z_rank=None, n_permutations=12, quantile=0.99, seed=1, salt=7)
    warm = M.conditional_perm_null_gpu(x, y, z, order=None, z_rank=None, n_permutations=12, quantile=0.99, seed=1, salt=7)
    assert cold == warm
