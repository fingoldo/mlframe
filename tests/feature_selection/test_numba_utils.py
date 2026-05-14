"""Unit tests for ``mlframe.feature_selection.filters._numba_utils``.

The three public ``@njit`` helpers (``arr2str``, ``count_cand_nbins``, ``unpack_and_sort``) are the cache-key / aggregation primitives consumed by
``conditional_mi`` and ``screen_predictors``. The invariants tested here back the comments in the source: collision-safety of the cache key, correct
aggregation of per-factor bin counts, and ordering-independence of the union representation used to canonicalise cache lookups.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils import (
    arr2str,
    count_cand_nbins,
    unpack_and_sort,
)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# arr2str
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestArr2Str:
    """Cache-key stringification of integer arrays. Must be deterministic, order-sensitive, and collision-safe across distinct multisets."""

    def test_equal_arrays_equal_strings(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([1, 2, 3], dtype=np.int64)
        assert arr2str(a) == arr2str(b)

    def test_different_arrays_different_strings(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([1, 2, 4], dtype=np.int64)
        assert arr2str(a) != arr2str(b)

    def test_order_matters(self):
        # The kernel iterates positionally; reversing the input MUST change the key (canonicalisation is the caller's job via ``unpack_and_sort``).
        a = np.array([1, 2, 3], dtype=np.int64)
        rev = a[::-1].copy()
        assert arr2str(a) != arr2str(rev)

    def test_empty_array(self):
        empty = np.array([], dtype=np.int64)
        assert arr2str(empty) == ""

    def test_single_element(self):
        a = np.array([42], dtype=np.int64)
        assert arr2str(a) == "42"

    def test_multidigit_separator_disambiguation(self):
        # Regression: the legacy naive concat collapsed sorted([1, 11]) and sorted([1, 1, 1]) both to "111". Underscore separator must keep them distinct.
        a = np.array([1, 11], dtype=np.int64)
        b = np.array([1, 1, 1], dtype=np.int64)
        assert arr2str(a) != arr2str(b)
        assert arr2str(a) == "1_11"
        assert arr2str(b) == "1_1_1"

    def test_returns_python_str(self):
        a = np.array([1, 2], dtype=np.int64)
        out = arr2str(a)
        # numba returns its UnicodeType which is interchangeable with Python str on the boundary.
        assert isinstance(out, str)
        assert out == "1_2"

    @pytest.mark.fast
    def test_biz_arr2str_no_collisions_on_uniform_random(self):
        # 10000 random length-5 keys with bin alphabet [0, 256). The "deterministic cache key" rule requires distinct arrays produce distinct strings on
        # realistic screening-path inputs; collision rate must be <0.1%.
        rng = np.random.default_rng(0)
        nbins = 256
        n_keys = 10_000
        seen: dict[str, tuple] = {}
        collisions = 0
        for _ in range(n_keys):
            arr = rng.integers(0, nbins, size=5).astype(np.int64)
            s = arr2str(arr)
            key = tuple(arr.tolist())
            if s in seen and seen[s] != key:
                collisions += 1
            else:
                seen[s] = key
        rate = collisions / n_keys
        assert rate < 0.001, f"collision rate {rate:.4%} exceeds 0.1% budget"


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# count_cand_nbins
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestCountCandNbins:
    """Sum of per-factor bin counts across a candidate's selected factor indices."""

    def test_basic_sum(self):
        X = np.array([0, 1, 2], dtype=np.int64)
        factors_nbins = np.array([3, 5, 7, 11], dtype=np.int64)
        assert count_cand_nbins(X, factors_nbins) == 3 + 5 + 7

    def test_single_factor(self):
        X = np.array([2], dtype=np.int64)
        factors_nbins = np.array([3, 5, 7, 11], dtype=np.int64)
        assert count_cand_nbins(X, factors_nbins) == 7

    def test_positive_for_nonempty(self):
        X = np.array([0, 1], dtype=np.int64)
        factors_nbins = np.array([4, 6], dtype=np.int64)
        out = count_cand_nbins(X, factors_nbins)
        assert out > 0
        assert isinstance(out, (int, np.integer))

    def test_repeated_indices(self):
        # The kernel does not deduplicate; same index counted twice is intentional in the screening path.
        X = np.array([1, 1, 1], dtype=np.int64)
        factors_nbins = np.array([2, 4], dtype=np.int64)
        assert count_cand_nbins(X, factors_nbins) == 12


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# unpack_and_sort
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestUnpackAndSort:
    """Concatenation + ascending sort of two integer iterables. Used to canonicalise the X u Z union for ``conditional_mi`` cache keys."""

    def test_sorted_ascending(self):
        x = np.array([3, 1], dtype=np.int64)
        z = np.array([4, 2], dtype=np.int64)
        out = unpack_and_sort(x, z)
        assert np.all(out[:-1] <= out[1:])

    def test_preserves_all_elements(self):
        x = np.array([5, 1, 3], dtype=np.int64)
        z = np.array([2, 4], dtype=np.int64)
        out = unpack_and_sort(x, z)
        assert len(out) == len(x) + len(z)
        assert sorted(out.tolist()) == sorted(x.tolist() + z.tolist())

    def test_ordering_independence(self):
        # Same multiset, different concat order -> same canonical key. This is the whole reason the helper exists.
        x1 = np.array([3, 1], dtype=np.int64)
        z1 = np.array([4, 2], dtype=np.int64)
        x2 = np.array([4, 2], dtype=np.int64)
        z2 = np.array([3, 1], dtype=np.int64)
        a = unpack_and_sort(x1, z1)
        b = unpack_and_sort(x2, z2)
        assert np.array_equal(a, b)

    def test_one_empty(self):
        x = np.array([], dtype=np.int64)
        z = np.array([7, 3, 5], dtype=np.int64)
        out = unpack_and_sort(x, z)
        assert out.tolist() == [3, 5, 7]

    def test_both_empty(self):
        x = np.array([], dtype=np.int64)
        z = np.array([], dtype=np.int64)
        out = unpack_and_sort(x, z)
        assert len(out) == 0

    def test_duplicates_preserved(self):
        x = np.array([1, 1], dtype=np.int64)
        z = np.array([1], dtype=np.int64)
        out = unpack_and_sort(x, z)
        assert out.tolist() == [1, 1, 1]

    def test_output_dtype_int64(self):
        x = np.array([1, 2], dtype=np.int64)
        z = np.array([3], dtype=np.int64)
        out = unpack_and_sort(x, z)
        assert out.dtype == np.int64
