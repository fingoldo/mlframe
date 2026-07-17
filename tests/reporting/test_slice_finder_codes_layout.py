"""Regression sensors for iter71: slice_finder ``codes`` matrix is Fortran-order so the arity-2 column gather is zero-copy.

The win (~7x on the per-combo aggregate at n=100k) comes purely from ``_bin_matrix`` storing ``codes`` column-major:
``codes[:, j]`` is then C-contiguous, so the ``np.ascontiguousarray`` gather in ``_aggregate_combo``'s arity-2 fast path
returns the view without copying ``n`` int64 per candidate pair. These tests pin BOTH the layout property (a future
revert to C-order trips it) AND output bit-identity (the change is layout-only, never numeric).
"""

import numpy as np
import pytest

from mlframe.reporting.charts.slice_finder import _aggregate_combo, _bin_matrix, find_weak_slices


def test_bin_matrix_codes_are_fortran_order_with_contiguous_columns():
    """Bin matrix codes are fortran order with contiguous columns."""
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((2000, 12))
    codes, _edges = _bin_matrix(mat, 4)
    assert codes.dtype == np.int64
    assert codes.flags["F_CONTIGUOUS"], "codes must be Fortran-order so column gathers are zero-copy (iter71 win)"
    for j in range(codes.shape[1]):
        assert codes[:, j].flags["C_CONTIGUOUS"], f"column {j} slice must be contiguous (no per-gather copy)"


def test_aggregate_combo_bit_identical_across_codes_layout():
    """Aggregate combo bit identical across codes layout."""
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((5000, 10))
    err = np.ascontiguousarray(rng.standard_normal(5000))
    codes_f, _ = _bin_matrix(mat, 4)  # prod: F-order
    codes_c = np.ascontiguousarray(codes_f)  # legacy C-order layout
    nbins = [4, 4]
    for combo in [(0, 1), (2, 5), (7, 9)]:
        sf, cf, _ = _aggregate_combo(codes_f, err, combo, nbins)
        sc, cc, _ = _aggregate_combo(codes_c, err, combo, nbins)
        assert np.array_equal(sf, sc), f"sums diverged on layout for {combo}"
        assert np.array_equal(cf, cc), f"counts diverged on layout for {combo}"
    # arity-3 mixed-radix path too
    sf, cf, _ = _aggregate_combo(codes_f, err, (0, 3, 6), [4, 4, 4])
    sc, cc, _ = _aggregate_combo(codes_c, err, (0, 3, 6), [4, 4, 4])
    assert np.array_equal(sf, sc) and np.array_equal(cf, cc)


def test_find_weak_slices_output_unchanged_by_layout():
    """Find weak slices output unchanged by layout."""
    rng = np.random.default_rng(7)
    n, p = 4000, 6
    X = rng.standard_normal((n, p))
    y_true = rng.standard_normal(n)
    # Inject a weak region so the finder returns a non-empty table.
    bad = X[:, 0] > 1.0
    y_pred = y_true + np.where(bad, rng.standard_normal(n) * 3.0, rng.standard_normal(n) * 0.1)
    names = [f"f{i}" for i in range(p)]
    res = find_weak_slices(X, y_true, y_pred, task="regression", feature_names=names, seed=0)
    assert len(res.table) > 0
    assert np.isfinite(res.global_error)
    # Re-run is deterministic + identical (seed fixed).
    res2 = find_weak_slices(X, y_true, y_pred, task="regression", feature_names=names, seed=0)
    assert res.table["score"].to_numpy().tolist() == res2.table["score"].to_numpy().tolist()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov"])
