"""Regression coverage for ``_get_col_codes_i64``'s column-codes cache (2026-07-21).

Found live via wellbore-50k cProfile: ``_resolve_pair_prevalence_gate``'s asymmetric-synergy branch
re-copies ``data[:, col_idx]`` into a fresh contiguous int64 array on EVERY call even though a small set
of bootstrap/anchor columns repeats across thousands of different partner pairs within one fit
(41902 ``np.ascontiguousarray`` calls / 22.7s tottime in that one function; a column-slice copy costs
~1ms vs ~0.4us for an already-contiguous array per a direct microbench). ``_col_codes_cache`` memoizes
per column index for the duration of one gate pre-pass -- selection-equivalence verified against a clean
origin/master checkout (same bit-identical pass/fail on the existing F2 synergy suite).
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mrmr_fe_step._step_pairs_rank import _get_col_codes_i64


def test_col_codes_cache_returns_correct_contiguous_int64_values():
    """A cached lookup must return the exact same values as an uncached ascontiguousarray copy."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 10, size=(200, 5)).astype(np.int32)
    cache: dict = {}
    codes = _get_col_codes_i64(data, 2, cache)
    assert codes.dtype == np.int64
    assert codes.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(codes, data[:, 2].astype(np.int64))


def test_col_codes_cache_hits_return_the_same_object_not_a_recopy():
    """A second lookup for the same column must hit the cache (object identity), not re-copy."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 10, size=(200, 5)).astype(np.int64)
    cache: dict = {}
    first = _get_col_codes_i64(data, 3, cache)
    second = _get_col_codes_i64(data, 3, cache)
    assert first is second, "repeat lookups for the same column must hit the cache, not re-copy"
    assert len(cache) == 1


def test_col_codes_cache_none_falls_back_to_uncached_behavior():
    """cache=None (the default) must preserve the pre-change always-copy behavior exactly."""
    rng = np.random.default_rng(2)
    data = rng.integers(0, 10, size=(50, 3)).astype(np.int64)
    a = _get_col_codes_i64(data, 0, None)
    b = _get_col_codes_i64(data, 0, None)
    assert a is not b, "cache=None must not memoize (each call independently correct)"
    np.testing.assert_array_equal(a, b)


def test_col_codes_cache_distinguishes_different_columns():
    """Different column indices must each get their own cache entry with the correct values."""
    rng = np.random.default_rng(3)
    data = rng.integers(0, 10, size=(100, 4)).astype(np.int64)
    cache: dict = {}
    c0 = _get_col_codes_i64(data, 0, cache)
    c1 = _get_col_codes_i64(data, 1, cache)
    assert len(cache) == 2
    np.testing.assert_array_equal(c0, data[:, 0])
    np.testing.assert_array_equal(c1, data[:, 1])
