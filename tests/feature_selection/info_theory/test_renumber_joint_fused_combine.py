"""Regression: the fused `_combine_factorize_njit` multiply-add+refactorize must
produce a partition + nclasses BIT-IDENTICAL to the two-step numpy
`joint + c*mult` -> `_factorize_dense_njit` form that `_renumber_joint` used
before the fusion. Pins the bit-identity claim so a future "tweak the fused
loop" cannot silently alter which joint class ids the CMI/entropy path sees.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import (
    _combine_factorize_njit,
    _factorize_dense_njit,
    _renumber_joint,
)


def _two_step_ref(joint, c64, mult):
    """Two step ref."""
    return _factorize_dense_njit(joint + c64 * mult)


def test_combine_factorize_matches_two_step_numpy():
    """Combine factorize matches two step numpy."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(1, 400))
        nb = int(rng.integers(2, 40))
        c0 = rng.integers(0, nb, size=n).astype(np.int64)
        c1 = rng.integers(0, nb, size=n).astype(np.int64)
        j0, mult0 = _factorize_dense_njit(np.ascontiguousarray(c0))
        ja, ma = _two_step_ref(j0, c1, mult0)
        jb, mb = _combine_factorize_njit(j0, c1, mult0)
        assert ma == mb
        np.testing.assert_array_equal(ja, jb)


def test_combine_factorize_hash_fallback_path():
    """Large packed ids (> _FAC_ARRAY_CAP would overflow the direct buffer)
    must route through the typed.Dict fallback and still match the reference."""
    rng = np.random.default_rng(7)
    j0 = rng.integers(0, 4000, size=500).astype(np.int64)
    c1 = rng.integers(0, 4000, size=500).astype(np.int64)
    mult = 5000
    ja, ma = _two_step_ref(j0, c1, mult)
    jb, mb = _combine_factorize_njit(j0, c1, mult)
    assert ma == mb
    np.testing.assert_array_equal(ja, jb)


def test_renumber_joint_three_col_bit_identical_to_legacy():
    """Full 3-col `_renumber_joint` (two fused folds) matches the legacy
    all-numpy-multiply-add reference."""

    def legacy(*cols):
        """Helper that legacy."""
        j = np.ascontiguousarray(cols[0], dtype=np.int64).ravel()
        j, m = _factorize_dense_njit(j)
        for c in cols[1:]:
            c64 = np.ascontiguousarray(c, dtype=np.int64).ravel()
            j = j + c64 * m
            j, m = _factorize_dense_njit(j)
        return j, int(m)

    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(1, 300))
        cols = [rng.integers(0, int(rng.integers(2, 12)), size=n).astype(np.int64) for _ in range(3)]
        ja, ma = legacy(*cols)
        jb, mb = _renumber_joint(*cols)
        assert ma == mb
        np.testing.assert_array_equal(ja, jb)
