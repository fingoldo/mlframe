"""B4 regression: ``conditional_mi`` cache key handling across all four combos.

Pre-B4 the function reused a single ``key`` local across four cache
branches. The legacy code happened to be safe because each branch
overwrites ``key`` before reading it -- but a future edit could trivially
break that invariant (e.g. adding a new branch that reads stale ``key``).

After B4 the four branches each own a separate local. This test enumerates
all four ``(can_use_x_cache, can_use_y_cache)`` truth assignments and pins
down which keys land in ``entropy_cache`` so the next refactor can't
quietly change the contract.
"""
from __future__ import annotations

import numpy as np
import pytest
from numba.core import types
from numba.typed import Dict as NumbaDict

from mlframe.feature_selection.filters import conditional_mi


def _build_data(n: int = 500, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 4, size=n).astype(np.int32)
    b = rng.integers(0, 4, size=n).astype(np.int32)
    c = rng.integers(0, 4, size=n).astype(np.int32)
    d = rng.integers(0, 4, size=n).astype(np.int32)
    factors = np.column_stack([a, b, c, d]).astype(np.int32)
    nbins = np.array([4, 4, 4, 4], dtype=np.int64)
    return factors, nbins


def _new_entropy_cache() -> dict:
    return NumbaDict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )


@pytest.mark.parametrize("can_use_x_cache,can_use_y_cache", [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])
def test_conditional_mi_cache_combos(can_use_x_cache, can_use_y_cache):
    factors, nbins = _build_data()
    x = np.array([0], dtype=np.int64)
    y = np.array([1], dtype=np.int64)
    z = np.array([2, 3], dtype=np.int64)
    cache = _new_entropy_cache()

    val = conditional_mi(
        factors_data=factors,
        x=x,
        y=y,
        z=z,
        var_is_nominal=None,
        factors_nbins=nbins,
        entropy_cache=cache,
        can_use_x_cache=can_use_x_cache,
        can_use_y_cache=can_use_y_cache,
    )
    assert np.isfinite(val)
    assert val >= 0.0

    # The Z entropy is always cached (its branch has no conditional gate).
    assert len(cache) >= 1, "entropy_z must always be cached when entropy_cache is provided"

    # When both x_cache and y_cache are off, only Z is cached.
    if not can_use_x_cache and not can_use_y_cache:
        assert len(cache) == 1
    # X-cache enabled => H(X, Z) is cached too.
    if can_use_x_cache:
        assert len(cache) >= 2
    # Y-cache enabled => H(Y, Z) is cached too.
    if can_use_y_cache:
        assert len(cache) >= 2
    # Both enabled => H(X, Y, Z) joins H(X, Z), H(Y, Z), H(Z) for a total of 4.
    if can_use_x_cache and can_use_y_cache:
        assert len(cache) == 4, f"expected 4 cached keys, got {len(cache)}: {dict(cache)}"


def test_conditional_mi_cache_round_trip_idempotent():
    """Calling twice with the same inputs hits the cache on every branch."""
    factors, nbins = _build_data()
    x = np.array([0], dtype=np.int64)
    y = np.array([1], dtype=np.int64)
    z = np.array([2, 3], dtype=np.int64)
    cache = _new_entropy_cache()

    first = conditional_mi(
        factors_data=factors, x=x, y=y, z=z,
        var_is_nominal=None, factors_nbins=nbins,
        entropy_cache=cache, can_use_x_cache=True, can_use_y_cache=True,
    )
    size_after_first = len(cache)
    second = conditional_mi(
        factors_data=factors, x=x, y=y, z=z,
        var_is_nominal=None, factors_nbins=nbins,
        entropy_cache=cache, can_use_x_cache=True, can_use_y_cache=True,
    )
    assert np.isclose(first, second, rtol=1e-12, atol=0)
    # Second call should not have inserted new keys.
    assert len(cache) == size_after_first
