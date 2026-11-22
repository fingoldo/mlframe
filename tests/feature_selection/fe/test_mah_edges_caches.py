"""The x-plan and multiset caches of ``mah_bin_edges`` return the edges the uncached computation returns, for the raw target and for its permutations."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _mah


def _reference_edges(x, y, initial_k=16):
    """``mah_bin_edges`` as it was before the caches: the inline quantile cuts, the label / quantile binning of y and the greedy merge."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    if x.size < 16:
        return np.array([], dtype=np.float64)
    qx_unique = np.unique(np.quantile(x, np.linspace(0, 1, initial_k + 1)))
    if qx_unique.size < 3:
        return np.array([], dtype=np.float64)
    initial_inner = qx_unique[1:-1].astype(np.float64)
    xb = np.searchsorted(initial_inner, x, side="right").astype(np.int64)
    yb, k_y = _mah._compute_y_binning(y, initial_k)
    if k_y == 0:
        return initial_inner
    k_x = int(xb.max()) + 1
    if k_x < 2 or k_y < 2:
        return initial_inner
    joint = np.bincount(xb * k_y + yb, minlength=k_x * k_y).reshape(k_x, k_y).astype(np.float64)
    _, row_merges, _ = _mah._greedy_merge_with_history(joint)
    return _mah._apply_merges_to_edges(initial_inner, row_merges)


def _targets(n, rng):
    """Integer targets of several shapes: binary, a few classes, many classes (quantile binned), a wide range, and a float one."""
    return {
        "binary": rng.integers(0, 2, n),
        "five_classes": rng.integers(0, 5, n),
        "many_classes": rng.integers(0, 200, n),
        "wide_range": rng.integers(0, 5_000_000, n),
        "negative": rng.integers(-30, 30, n),
        "float": rng.standard_normal(n),
        "constant": np.zeros(n, dtype=np.int64),
    }


@pytest.mark.parametrize("name", ["binary", "five_classes", "many_classes", "wide_range", "negative", "float", "constant"])
def test_cached_edges_equal_the_reference_for_the_target_and_its_permutations(name):
    """The first call (cache miss) and the later ones (cache hits, permuted target) give the reference edges."""
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.standard_normal(n) + 0.8 * (rng.random(n) < 0.3)
    y = _targets(n, rng)[name]
    for _ in range(4):
        yp = y[rng.permutation(n)]
        np.testing.assert_array_equal(_mah.mah_bin_edges(x, yp), _reference_edges(x, yp))
    np.testing.assert_array_equal(_mah.mah_bin_edges(x, y), _reference_edges(x, y))


def test_a_changed_column_is_not_served_from_the_plan_of_another():
    """Two columns of the same length and K get their own plans."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, 2000)
    a, b = rng.standard_normal(2000), rng.standard_normal(2000) ** 3
    np.testing.assert_array_equal(_mah.mah_bin_edges(a, y), _reference_edges(a, y))
    np.testing.assert_array_equal(_mah.mah_bin_edges(b, y), _reference_edges(b, y))
    np.testing.assert_array_equal(_mah.mah_bin_edges(a, y), _reference_edges(a, y))


def test_the_caches_are_bounded():
    """Neither cache grows beyond its cap."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 400)
    for i in range(_mah._X_PLAN_CACHE_MAX_ENTRIES + 20):
        _mah.mah_bin_edges(rng.standard_normal(400) + i, y)
    assert len(_mah._X_PLAN_CACHE) <= _mah._X_PLAN_CACHE_MAX_ENTRIES
    assert len(_mah._Y_MAP_CACHE) <= _mah._Y_MAP_CACHE_MAX_ENTRIES
