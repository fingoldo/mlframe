"""The combo Miller-Madow kernels debit the plug-in bias with the OCCUPIED (cell, class) count, like ``joint_synergy_mi``; counting occupied cells only added a positive term instead."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_synergy_screen import _combo_mm_mi_cols_njit, _combo_mm_mi_njit, joint_synergy_mi


def _independent(seed, n=3000, k=300):
    """A k-level code column and an independent binary target."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, n).astype(np.int64), rng.integers(0, 2, n).astype(np.int64)


def test_combo_mm_mi_of_many_cell_independent_codes_matches_the_reference_debit():
    """300 cells x 2 classes at n=3000: the plug-in bias is ~0.05 nats, so the corrected value must sit near the reference, far below the plug-in level."""
    got, ref = [], []
    for seed in range(20):
        c, t = _independent(seed)
        got.append(_combo_mm_mi_njit(c.reshape(-1, 1), np.array([300], dtype=np.int64), t, 2, 300))
        ref.append(joint_synergy_mi(c, np.zeros_like(c), t, min_rows_per_cell=0.0))
    assert np.mean(got) < 0.012
    assert np.mean(got) == pytest.approx(np.mean(ref), abs=1e-9)


def test_combo_mm_mi_cols_kernel_agrees_with_the_matrix_kernel_on_the_debit():
    """The column-reading twin applies the same debit on a pair of codes."""
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 20, 3000).astype(np.int64), rng.integers(0, 15, 3000).astype(np.int64)
    t = rng.integers(0, 2, 3000).astype(np.int64)
    cards = np.array([20, 15], dtype=np.int64)
    m = _combo_mm_mi_njit(np.column_stack([a, b]), cards, t, 2, 300)
    c = _combo_mm_mi_cols_njit(a, b, b, 2, cards, t, 2, 300)
    assert c == pytest.approx(m, abs=1e-12)
    ref = joint_synergy_mi(a, b, t, min_rows_per_cell=0.0)
    assert m == pytest.approx(ref, abs=1e-9)


def test_combo_mm_mi_gpu_refuses_codes_beyond_the_declared_cardinality():
    """A code outside [0, card) would index outside the shared table on the device; it must raise before any launch."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _synergy_combo_gpu as g

    cols = [np.array([0, 1, 5, 1]), np.array([0, 1, 0, 1])]
    with pytest.raises(ValueError):
        g.combo_mm_mi_gpu(cols, [(0, 1)], [2, 2], np.array([0, 1, 0, 1]), 2)
