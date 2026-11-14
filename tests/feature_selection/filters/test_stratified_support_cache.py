"""The stratified support of the conditional permutation null is built once per (target, support) content."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _fe_cmi_redundancy_null as M


def _inp(seed, n=3000):
    """Null inputs for a random 6-bin candidate and a binary target."""
    rng = np.random.default_rng(seed)
    return M._NullInputs(rng.integers(0, 6, n).astype(np.int64), rng.integers(0, 2, n).astype(np.int64), 0, 0)


def _z(seed, n=3000, k=25):
    """A support with ``k`` strata."""
    return np.random.default_rng(seed).integers(0, k, n).astype(np.int64)


def test_same_content_hits_and_other_candidates_share_the_support():
    """Two candidates against one (y, z) get the same support object, even through a copy of z."""
    M._SUPPORT_CACHE.clear()
    z = _z(1)
    a = M._stratified_support_cached(_inp(1), z)
    b = M._stratified_support_cached(_inp(1), z.copy())
    assert a is b


def test_in_place_change_of_z_is_a_miss():
    """A support changed in place must not return the stale object."""
    M._SUPPORT_CACHE.clear()
    z = _z(2)
    a = M._stratified_support_cached(_inp(1), z)
    z[0] = (z[0] + 1) % 25
    b = M._stratified_support_cached(_inp(1), z)
    assert a is not b and np.array_equal(b.z, z)


def test_cache_is_bounded():
    """Only the last ``_SUPPORT_CACHE_MAX`` supports are kept."""
    M._SUPPORT_CACHE.clear()
    for s in range(M._SUPPORT_CACHE_MAX + 2):
        M._stratified_support_cached(_inp(1), _z(10 + s))
    assert len(M._SUPPORT_CACHE) == M._SUPPORT_CACHE_MAX


def test_cached_support_equals_a_fresh_build():
    """The cached object carries the same arrays a direct build gives."""
    M._SUPPORT_CACHE.clear()
    z, inp = _z(3), _inp(3)
    cached = M._stratified_support_cached(inp, z)
    fresh = M._StratifiedSupport(inp, z)
    for name in ("order", "sorted_z", "z_rank", "y_i", "z_i"):
        assert np.array_equal(getattr(cached, name), getattr(fresh, name))
    assert cached.has_groups == fresh.has_groups and cached.k_yz == fresh.k_yz and cached.k_z == fresh.k_z
