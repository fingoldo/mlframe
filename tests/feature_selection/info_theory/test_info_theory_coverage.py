"""Additional coverage for info_theory.py -- Python wrapper branches.

The module is dominated by @njit functions whose interiors are invisible to coverage.py. Tests here focus on:
- Triggering each Python-level function call (counts the `def` line + dispatch)
- Exercising any non-njit code paths (probably none in this module)
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    entropy,
    mi,
    conditional_mi,
    merge_vars,
    compute_mi_from_classes,
)

try:
    from mlframe.feature_selection.filters.info_theory import entropy_miller_madow

    _HAVE_MM = True
except ImportError:
    _HAVE_MM = False


def _make_classes_and_freqs(n: int = 200, nbins_x: int = 3, nbins_y: int = 2, seed: int = 0):
    """Make classes and freqs."""
    rng = np.random.default_rng(seed)
    cx = rng.integers(0, nbins_x, n).astype(np.int32)
    cy = rng.integers(0, nbins_y, n).astype(np.int32)
    fx = np.bincount(cx, minlength=nbins_x).astype(np.float64) / n
    fy = np.bincount(cy, minlength=nbins_y).astype(np.float64) / n
    return cx, fx, cy, fy


@pytest.mark.fast
def test_compute_mi_from_classes_int32():
    """Compute mi from classes int32."""
    cx, fx, cy, fy = _make_classes_and_freqs(seed=1)
    out = compute_mi_from_classes(classes_x=cx, freqs_x=fx, classes_y=cy, freqs_y=fy, dtype=np.int32)
    assert np.isfinite(out)
    assert out >= 0.0


def test_compute_mi_from_classes_int64():
    """Compute mi from classes int64."""
    cx, fx, cy, fy = _make_classes_and_freqs(seed=2)
    cx64 = cx.astype(np.int64)
    cy64 = cy.astype(np.int64)
    out = compute_mi_from_classes(classes_x=cx64, freqs_x=fx, classes_y=cy64, freqs_y=fy, dtype=np.int64)
    assert np.isfinite(out)


def test_entropy_uniform_freqs():
    """Entropy uniform freqs."""
    freqs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    h = entropy(freqs)
    expected = np.log(4)
    assert abs(float(h) - float(expected)) < 1e-9


def test_entropy_degenerate():
    """Single non-zero cell -> entropy ~ 0."""
    freqs = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    h = entropy(freqs)
    assert float(h) < 0.01


def test_entropy_min_occupancy_branch():
    """min_occupancy > 0 takes the alternative filter branch."""
    freqs = np.array([0.5, 0.3, 0.1, 0.1], dtype=np.float64)
    h_no_floor = entropy(freqs)
    h_floor = entropy(freqs, min_occupancy=0.15)  # drops 2 of 4 bins
    # h_floor uses fewer bins -> different from h_no_floor
    assert np.isfinite(h_no_floor) and np.isfinite(h_floor)


def test_entropy_miller_madow_smoke():
    """Entropy miller madow smoke."""
    if not _HAVE_MM:
        pytest.skip("entropy_miller_madow not exported")
    freqs = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    h_plain = entropy(freqs)
    h_mm = entropy_miller_madow(freqs, n_samples=100)
    assert np.isfinite(h_plain) and np.isfinite(h_mm)


def test_merge_vars_two_columns():
    """Merge vars two columns."""
    rng = np.random.default_rng(3)
    n = 200
    factors_data = rng.integers(0, 3, size=(n, 4)).astype(np.int32)
    nbins = np.array([3, 3, 3, 3], dtype=np.int64)
    vars_indices = np.array([0, 1], dtype=np.int64)
    classes, freqs, n_bins = merge_vars(
        factors_data=factors_data,
        vars_indices=vars_indices,
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    assert classes.shape[0] == n
    assert freqs.ndim == 1
    assert int(n_bins) >= 1


def test_merge_vars_single_column():
    """Merge vars single column."""
    rng = np.random.default_rng(4)
    factors_data = rng.integers(0, 5, size=(100, 3)).astype(np.int32)
    nbins = np.array([5, 5, 5], dtype=np.int64)
    classes, _freqs, _n_bins = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    assert classes.shape[0] == 100


def test_mi_self_high():
    """mi(x, x) should be high since H(X, X) = H(X)."""
    rng = np.random.default_rng(5)
    factors_data = rng.integers(0, 4, size=(300, 2)).astype(np.int32)
    factors_data[:, 1] = factors_data[:, 0]  # copy
    nbins = np.array([4, 4], dtype=np.int64)
    out = mi(factors_data=factors_data, x=np.array([0]), y=np.array([1]), factors_nbins=nbins, dtype=np.int32)
    assert out > 0.5


def test_mi_independent_low():
    """mi(x, independent y) should be near 0."""
    rng = np.random.default_rng(6)
    factors_data = rng.integers(0, 3, size=(300, 2)).astype(np.int32)
    nbins = np.array([3, 3], dtype=np.int64)
    out = mi(factors_data=factors_data, x=np.array([0]), y=np.array([1]), factors_nbins=nbins, dtype=np.int32)
    assert out < 0.2


def test_conditional_mi_smoke():
    """conditional_mi with conditioning set of size 1."""
    rng = np.random.default_rng(7)
    factors_data = rng.integers(0, 3, size=(200, 3)).astype(np.int32)
    nbins = np.array([3, 3, 3], dtype=np.int64)
    var_is_nominal = np.array([True, True, True], dtype=np.bool_)
    out = conditional_mi(
        factors_data=factors_data,
        x=np.array([0]),
        y=np.array([1]),
        z=np.array([2]),
        var_is_nominal=var_is_nominal,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    assert np.isfinite(out)


def test_mi_verbose_branch():
    """verbose=True takes the print/log path."""
    rng = np.random.default_rng(8)
    factors_data = rng.integers(0, 3, size=(100, 2)).astype(np.int32)
    nbins = np.array([3, 3], dtype=np.int64)
    out = mi(factors_data=factors_data, x=np.array([0]), y=np.array([1]), factors_nbins=nbins, verbose=True, dtype=np.int32)
    assert np.isfinite(out)
