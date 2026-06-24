"""Regression tests for NaN handling in the UNIFORM discretisation path.

Covers four bugs in the uniform binning kernels:

* **B1** ``discretize_uniform`` / ``discretize_uniform_parallel``: the uniform affine map left NaN
  unchanged through ``np.clip`` (a no-op on NaN), then ``.astype(int8)`` produced a garbage bin code
  (RuntimeWarning: invalid value encountered in cast) and NaN rows silently collided into bin 0 with
  real low values. NaN now routes to a dedicated NaN bin (``n_bins``), distinct from real bin 0.
* **B2** ``mlframe.core.arrays.arrayMinMax`` (used by the uniform path): a leading-NaN column seeded
  min=max=NaN under the no-NaN fastmath assumption, poisoning the whole column. min/max is now
  NaN-aware (finite-subset over the scan) while staying bit-identical on all-finite input.

The bit-identity test pins the all-finite uniform codes against a captured reference so the NaN
routing cannot regress the common case.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.core.arrays import arrayMinMax
from mlframe.feature_selection.filters.discretization import (
    discretize_array,
    discretize_uniform,
    discretize_uniform_parallel,
)


def _uniform_codes_via_clip_astype(arr: np.ndarray, n_bins: int, dtype=np.int8) -> np.ndarray:
    """Pre-fix uniform body (vectorised ``np.clip`` + ``astype``) on FINITE input, for bit-identity."""
    mn, mx = arrayMinMax(arr)
    rng = mx - mn
    if rng <= 0:
        return np.zeros_like(arr, dtype=dtype)
    rev = n_bins / rng
    return np.clip((arr - mn) * rev, 0, n_bins - 1).astype(dtype)


# ---------------------------------------------------------------------------
# B1: NaN row routes to dedicated bin, never collides with real bin 0 / garbage cast
# ---------------------------------------------------------------------------


def test_b1_uniform_nan_routes_to_dedicated_bin_no_warning():
    n_bins = 10
    arr = np.array([1.0, 2.0, 3.0, np.nan, 100.0, 50.0], dtype=np.float64)
    with warnings.catch_warnings():
        # Pre-fix this emitted "invalid value encountered in cast"; assert it is gone.
        warnings.simplefilter("error", RuntimeWarning)
        out = discretize_array(arr, n_bins=n_bins, method="uniform")

    nan_code = n_bins  # one past the top real code (0..n_bins-1)
    assert out[3] == nan_code, "NaN row must land in the dedicated NaN bin"
    # Real low value must be in bin 0 -- distinct from the NaN bin (no collision).
    assert out[0] == 0, "real low value must occupy bin 0"
    assert out[3] != out[0], "NaN must NOT collide with real bin-0 rows"


def test_b1_uniform_finite_rows_binned_identically_to_all_finite_case():
    """The finite rows of a NaN-bearing column bin EXACTLY as the same values without the NaN."""
    n_bins = 10
    with_nan = np.array([1.0, 2.0, 3.0, np.nan, 100.0, 50.0], dtype=np.float64)
    all_finite = np.array([1.0, 2.0, 3.0, 100.0, 50.0], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out_with_nan = discretize_array(with_nan, n_bins=n_bins, method="uniform")
    out_finite = discretize_array(all_finite, n_bins=n_bins, method="uniform")

    # finite positions of `with_nan` are indices [0,1,2,4,5]; map to all_finite [0,1,2,3,4].
    np.testing.assert_array_equal(out_with_nan[[0, 1, 2, 4, 5]], out_finite)


def test_b1_uniform_serial_and_parallel_agree_on_nan():
    n_bins = 10
    arr = np.array([1.0, 2.0, 3.0, np.nan, 100.0, 50.0], dtype=np.float64)
    mn, mx = arrayMinMax(arr)
    serial = discretize_uniform(arr=arr, n_bins=n_bins, dtype=np.int8)
    parallel = discretize_uniform_parallel(arr, n_bins, float(mn), float(mx), dtype=np.int8)
    np.testing.assert_array_equal(serial, parallel)
    assert serial[3] == n_bins


def test_b1_uniform_all_nan_column():
    n_bins = 10
    arr = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = discretize_array(arr, n_bins=n_bins, method="uniform")
    assert np.all(out == n_bins), "every row of an all-NaN column belongs to the NaN bin"


# ---------------------------------------------------------------------------
# B2: NaN-aware min/max -- leading NaN no longer poisons the column
# ---------------------------------------------------------------------------


def test_b2_arrayminmax_leading_nan():
    mn, mx = arrayMinMax(np.array([np.nan, 1.0, 2.0, 3.0, 100.0], dtype=np.float64))
    assert mn == 1.0 and mx == 100.0, "finite min/max must ignore a leading NaN"


def test_b2_arrayminmax_non_leading_and_trailing_nan():
    assert arrayMinMax(np.array([1.0, 2.0, np.nan, 3.0, 100.0])) == (1.0, 100.0)
    assert arrayMinMax(np.array([1.0, 2.0, 3.0, np.nan])) == (1.0, 3.0)


def test_b2_arrayminmax_all_nan_returns_sentinel():
    mn, mx = arrayMinMax(np.array([np.nan, np.nan], dtype=np.float64))
    assert np.isnan(mn) and np.isnan(mx), "all-NaN range returns NaN sentinels for the _rng guard"


def test_b2_uniform_leading_nan_column_not_all_nan():
    """A column whose FIRST element is NaN must still bin its finite rows correctly."""
    n_bins = 10
    arr = np.array([np.nan, 1.0, 2.0, 3.0, 100.0], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = discretize_array(arr, n_bins=n_bins, method="uniform")
    assert out[0] == n_bins, "leading NaN -> dedicated NaN bin"
    # The finite rows must span more than one bin (column is NOT collapsed to all-NaN/constant).
    finite_codes = out[1:]
    assert finite_codes.min() == 0 and finite_codes.max() == n_bins - 1, (
        "finite rows must span the full bin range, proving min/max was resolved over the finite subset"
    )


# ---------------------------------------------------------------------------
# Bit-identity: all-finite uniform codes unchanged vs the captured reference
# ---------------------------------------------------------------------------


def test_bit_identity_uniform_finite_matches_clip_astype_reference():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(1, 500))
        arr = rng.standard_normal(n) * rng.uniform(0.1, 1000.0) + rng.uniform(-500.0, 500.0)
        n_bins = int(rng.integers(2, 50))
        got = discretize_uniform(arr=arr, n_bins=n_bins, dtype=np.int8)
        ref = _uniform_codes_via_clip_astype(arr, n_bins, dtype=np.int8)
        np.testing.assert_array_equal(got, ref)


def test_bit_identity_uniform_captured_reference_vector():
    """Pin a concrete captured code vector so a future change to the affine map is caught explicitly."""
    arr = np.array([1000.0, 1010.0, 1050.0, 1099.0, 1100.0], dtype=np.float64)
    out = discretize_uniform(arr=arr, n_bins=10, dtype=np.int8)
    np.testing.assert_array_equal(out, np.array([0, 1, 5, 9, 9], dtype=np.int8))


def test_bit_identity_uniform_constant_column():
    out = discretize_uniform(arr=np.array([5.0] * 8, dtype=np.float64), n_bins=10, dtype=np.int8)
    np.testing.assert_array_equal(out, np.zeros(8, dtype=np.int8))


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v", "--no-cov", "-p", "no:randomly"]))
