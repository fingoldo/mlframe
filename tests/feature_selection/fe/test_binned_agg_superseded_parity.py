"""Parity of the superseded binned-numeric-agg reference helpers with the production replacements.

The production path (``fit_binned_numeric_agg``) runs ``_global_stats_all``, ``_per_cell_count_sum_njit`` and
``_per_cell_moments_stable``. The three helpers below are kept as references the replacements are checked against,
so each one must keep agreeing with the function that replaced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._binned_numeric_agg_fe import (
    _derive_cell_stats,
    _global_stat,
    _global_stats_all,
    _per_cell_count_sum_njit,
    _per_cell_moments_stable,
    _per_cell_raw_moments_njit,
    per_cell_stats_bincount,
)

STATS = ("mean", "std", "skew", "kurt")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("stat", STATS)
def test_global_stat_matches_fused_global_stats_all(seed, stat):
    """The superseded per-stat fallback agrees with the fused production form for every stat."""
    rng = np.random.default_rng(seed)
    v = rng.normal(loc=3.0, scale=2.0, size=600)
    v = v[np.isfinite(v)]
    ref = _global_stat(v, stat)
    fused = _global_stats_all(v, [stat])[stat]
    assert np.isclose(ref, fused, rtol=1e-9, atol=1e-12), (stat, ref, fused)


def test_global_stat_empty_column_is_zero_like_fused():
    """A column with no finite values yields 0.0 from both the reference and the fused form."""
    v = np.array([np.nan, np.inf, -np.inf])
    for stat in STATS:
        assert _global_stat(v, stat) == 0.0
        assert _global_stats_all(v[np.isfinite(v)], [stat])[stat] == 0.0


def test_per_cell_raw_moments_cnt_and_s1_are_bit_identical_to_pruned_kernel():
    """The reference kernel's count and sum outputs are bit-identical to the pruned pass prod calls."""
    rng = np.random.default_rng(7)
    n, n_cells = 2000, 17
    codes = rng.integers(0, n_cells, size=n).astype(np.int64)
    v = rng.normal(size=n)
    cnt_ref, s1_ref, _s2, _s3, _s4 = _per_cell_raw_moments_njit(codes, v, n_cells)
    cnt, s1 = _per_cell_count_sum_njit(codes, v, n_cells)
    assert np.array_equal(cnt, cnt_ref)
    assert np.array_equal(s1, s1_ref)


@pytest.mark.parametrize("seed", [3, 4])
def test_per_cell_stats_bincount_matches_production_composition(seed):
    """The convenience per-cell API returns exactly what the production moment pass composes to."""
    rng = np.random.default_rng(seed)
    n, n_cells = 1500, 9
    codes = rng.integers(0, n_cells, size=n).astype(np.int64)
    v = rng.normal(loc=-1.0, size=n)
    got = per_cell_stats_bincount(codes, v, n_cells, STATS)
    cnt, mean, cm2, cm3, cm4 = _per_cell_moments_stable(codes, v, n_cells)
    want = _derive_cell_stats(cnt, mean, cm2, cm3, cm4, STATS)
    for stat in STATS:
        assert np.array_equal(got[stat], want[stat], equal_nan=True), stat
