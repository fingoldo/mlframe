"""The inline shuffle draws feed valid, uniform, thread-independent Fisher-Yates permutations."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._family_maxt_gate import _fill_permutations


def test_rows_are_permutations_and_depend_only_on_their_seed():
    """Every row is a permutation of range(n); row i of a block equals row 0 of a block seeded base + i."""
    n = 500
    block = np.empty((6, n), dtype=np.int32)
    _fill_permutations(block, 40)
    assert len(block) > 0 and all(np.array_equal(np.sort(r), np.arange(n)) for r in block)
    single = np.empty((1, n), dtype=np.int32)
    _fill_permutations(single, 43)
    np.testing.assert_array_equal(block[3], single[0])


def test_each_position_is_uniform_over_the_values():
    """Over many seeds the first position of a 5-element permutation takes every value with frequency 1/5 (within five standard errors)."""
    draws = 40000
    block = np.empty((draws, 5), dtype=np.int32)
    _fill_permutations(block, 7)
    freq = np.bincount(block[:, 0], minlength=5) / draws
    se = (0.2 * 0.8 / draws) ** 0.5
    assert np.abs(freq - 0.2).max() < 5 * se
    pair = np.bincount(block[:, 0] * 5 + block[:, 1], minlength=25).reshape(5, 5) / draws
    off = ~np.eye(5, dtype=bool)
    assert np.abs(pair[off] - 1 / 20).max() < 5 * (1 / 20 * 19 / 20 / draws) ** 0.5
