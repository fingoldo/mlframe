"""The entropy-consistent Miller-Madow MI option corrects each entropy term, so a sparse joint is not over-corrected."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.info_theory._entropy_kernels import entropy_miller_madow, mi, mi_miller_madow, mi_miller_madow_correct


def _data(identical: bool):
    """Two 8-level columns: identical (sparse joint) or independent (fully occupied joint)."""
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.integers(0, 8, n)
    b = a.copy() if identical else rng.integers(0, 8, n)
    return np.column_stack([a, b]).astype(np.int32), np.array([8, 8], dtype=np.int64), n


def test_identical_columns_recover_the_miller_madow_entropy():
    """For X == Y the MI is H(X); the consistent form equals the MM entropy, the product form subtracts (k-1)^2/(2n) instead."""
    data, nbins, n = _data(True)
    x, y = np.array([0]), np.array([1])
    plug = mi(data, x, y, nbins)
    consistent = mi_miller_madow(data, x, y, nbins, False, np.int32, True)
    product = mi_miller_madow(data, x, y, nbins)
    freqs = np.bincount(data[:, 0], minlength=8) / n
    assert abs(consistent - entropy_miller_madow(freqs, n)) < 1e-9
    assert abs(consistent - (plug + 7 / (2 * n))) < 1e-9
    assert abs(product - (plug - 49 / (2 * n))) < 1e-9


def test_full_joint_matches_the_product_form():
    """When the joint is fully occupied (k_xy = k_x * k_y) both forms coincide."""
    data, nbins, _ = _data(False)
    x, y = np.array([0]), np.array([1])
    a = mi_miller_madow(data, x, y, nbins, False, np.int32, True)
    b = mi_miller_madow(data, x, y, nbins)
    assert abs(a - b) < 1e-12


def test_correct_helper_default_is_unchanged():
    """Without k_xy the helper keeps the product form."""
    assert mi_miller_madow_correct(0.5, 5, 4, 1000) == 0.5 - 12 / 2000
    assert mi_miller_madow_correct(0.5, 5, 4, 1000, 20) == 0.5 + (5 + 4 - 20 - 1) / 2000
