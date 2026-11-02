"""The cardinality cache key of ``joint_cardinalities_cupy`` hashes stable host arrays by identity-memoised content and keeps distinguishing different content."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _mi_greedy_cmi_fe_gpu as G


def test_host_key_array_is_stable_only_for_one_dimensional_contiguous_arrays():
    """The same object comes back for a 1-D contiguous host array; anything else is copied and flagged unstable."""
    a = np.arange(10, dtype=np.int64)
    got, stable = G._host_key_array(cp, a)
    assert got is a and stable
    b = np.arange(20, dtype=np.int64).reshape(4, 5)
    got, stable = G._host_key_array(cp, b)
    assert not stable and got.shape == (20,)
    c = np.arange(20, dtype=np.int64)[::2]
    got, stable = G._host_key_array(cp, c)
    assert not stable and got.flags.c_contiguous
    got, stable = G._host_key_array(cp, cp.arange(6))
    assert not stable and isinstance(got, np.ndarray)


def test_cardinalities_are_unchanged_and_follow_the_content():
    """Same cardinalities as the counted truth; two calls with equal content agree, and a different z gives its own cardinalities (no stale cache hit)."""
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.integers(0, 6, n).astype(np.int64)
    y = rng.integers(0, 2, n).astype(np.int64)
    z1 = rng.integers(0, 5, n).astype(np.int64)
    z2 = rng.integers(0, 40, n).astype(np.int64)
    k_z, k_xz, k_yz, k_xyz = G.joint_cardinalities_cupy(x, y, z1)
    assert k_z == len(np.unique(z1))
    assert k_xz == len(np.unique(x * 100 + z1))
    assert k_yz == len(np.unique(y * 100 + z1))
    assert k_xyz == len(np.unique((x * 10 + y) * 100 + z1))
    assert G.joint_cardinalities_cupy(x, y, z1) == (k_z, k_xz, k_yz, k_xyz)
    k2 = G.joint_cardinalities_cupy(x, y, z2)
    assert k2[0] == len(np.unique(z2)) and k2 != (k_z, k_xz, k_yz, k_xyz)
