"""Regression: _renumber_joint tolerates a (n, 1) conditioning column (stray singleton 2nd dim).

Pre-fix, a (n, 1) col made the njit `_factorize_dense_njit` see a 2-D array, raising numba
`TypingError: Cannot unify Literal[int](0) and array(int64, 1d, C) for 'jmax'` at compile -- which the MRMR
unified_second_pass_gate FE swallowed ("continuing without ... columns"), silently dropping the CMI-greedy FE.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _renumber_joint


def test_renumber_joint_2d_singleton_matches_1d():
    """Renumber joint 2d singleton matches 1d."""
    rng = np.random.default_rng(0)
    c1d = rng.integers(0, 4, 200).astype(np.int64)
    c2d = c1d.reshape(-1, 1)  # the (n, 1) shape that tripped numba

    j1, m1 = _renumber_joint(c1d)
    j2, m2 = _renumber_joint(c2d)  # pre-fix: numba TypingError
    np.testing.assert_array_equal(j1, j2)
    assert m1 == m2


def test_renumber_joint_2d_singleton_multicol():
    """Renumber joint 2d singleton multicol."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 4, 200).astype(np.int64)
    b = rng.integers(0, 3, 200).astype(np.int64)
    jA, mA = _renumber_joint(a, b)
    jB, mB = _renumber_joint(a.reshape(-1, 1), b.reshape(-1, 1))
    np.testing.assert_array_equal(jA, jB)
    assert mA == mB
