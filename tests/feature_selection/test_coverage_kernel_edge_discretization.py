"""Kernel-level edge coverage for the MRMR discretization primitive.

``discretize_array`` is the binning kernel every MI computation in MRMR feeds
through. These tests pin its GRACEFUL behaviour on the degenerate columns the
full-fit guards mostly reject upstream -- but the kernel itself must still produce
valid, in-range, deterministic bin codes (never a NaN code, never an out-of-range
code, never a crash) because it is also reached on partially-degenerate real frames
that pass the fit boundary (e.g. some-NaN columns).

The behaviours asserted here were verified against the current implementation; they
are the defensible graceful-degradation contract, so a regression that starts
emitting NaN / out-of-range codes -- or crashes on a constant / all-NaN column --
trips here.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.discretization import discretize_array


def _in_range(codes: np.ndarray, n_bins: int) -> bool:
    """Every bin code is a finite integer in ``[0, n_bins-1]`` (no NaN / garbage)."""
    arr = np.asarray(codes)
    if arr.dtype.kind not in "iu":
        return False
    return bool(arr.min() >= 0 and arr.max() <= n_bins - 1)


def test_all_nan_column_yields_single_in_range_code():
    """An all-NaN column collapses to one constant in-range bin code, no crash."""
    codes = discretize_array(np.full(50, np.nan), n_bins=5)
    assert _in_range(codes, 5)
    assert len(np.unique(codes)) == 1


def test_constant_column_yields_single_in_range_code():
    codes = discretize_array(np.full(50, 7.0), n_bins=5)
    assert _in_range(codes, 5)
    assert len(np.unique(codes)) == 1


def test_partial_nan_column_codes_are_finite_and_in_range():
    """A column with some NaN must produce all-finite, in-range codes; the NaN rows
    must map to a STABLE code (deterministic), never a NaN/garbage code."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=60)
    nan_pos = np.array([3, 7, 11, 40])
    a[nan_pos] = np.nan
    codes = discretize_array(a, n_bins=4)
    assert _in_range(codes, 4)
    # all NaN positions map to the same single code (deterministic, stable).
    assert len(set(codes[nan_pos].tolist())) == 1


def test_n_below_bins_does_not_crash_and_stays_in_range():
    """Fewer samples than requested bins must not crash; codes stay in range."""
    codes = discretize_array(np.array([1.0, 2.0, 3.0]), n_bins=10)
    assert _in_range(codes, 10)


def test_inf_values_do_not_emit_out_of_range_codes():
    """The kernel is reached below the fit-level inf guard on some paths; an inf
    must be clamped into a valid bin, never produce a NaN / out-of-range code."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    a[0] = np.inf
    a[1] = -np.inf
    codes = discretize_array(a, n_bins=5)
    assert _in_range(codes, 5)


def test_all_inf_column_yields_in_range_code():
    codes = discretize_array(np.full(50, np.inf), n_bins=5)
    assert _in_range(codes, 5)


def test_deterministic_codes_across_calls():
    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    a[rng.random(200) < 0.1] = np.nan
    c1 = discretize_array(a, n_bins=8)
    c2 = discretize_array(a, n_bins=8)
    assert np.array_equal(c1, c2)


def test_uniform_method_in_range():
    rng = np.random.default_rng(0)
    codes = discretize_array(rng.normal(size=200), n_bins=10, method="uniform")
    assert _in_range(codes, 10)
