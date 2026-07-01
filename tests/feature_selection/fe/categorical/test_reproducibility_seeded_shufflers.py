"""Regression tests for seeded-RNG reproducibility of the cat-FE permutation/shuffle njit kernels.

Each kernel previously used the process-global numba RNG (``np.random.randint`` / ``np.random.shuffle``), so its permutation was non-reproducible across runs AND it
polluted the caller's RNG stream. The fix threads a ``base_seed`` and seeds an inline LCG. These tests pin: (a) same seed -> identical permutation, (b) different seed ->
different permutation, (c) numpy's process-global state is untouched.
"""
from __future__ import annotations

import numpy as np
import pytest


def _global_state_unchanged(before) -> bool:
    after = np.random.get_state()
    return before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_con1_conditional_shuffle_within_strata_seeded():
    from mlframe.feature_selection.filters._cat_confirm_permutation import _conditional_shuffle_within_strata

    y = np.array([0, 1, 0, 1, 0, 1, 1, 0] * 4, dtype=np.int64)
    base = np.arange(len(y)).astype(np.int64)

    st = np.random.get_state()
    a = base.copy()
    _conditional_shuffle_within_strata(a, y, 2, 4242)
    b = base.copy()
    _conditional_shuffle_within_strata(b, y, 2, 4242)
    c = base.copy()
    _conditional_shuffle_within_strata(c, y, 2, 9999)

    assert np.array_equal(a, b), "same base_seed must give identical permutation"
    assert not np.array_equal(a, c), "different base_seed must give a different permutation"
    assert _global_state_unchanged(st), "must not touch numpy global RNG state"


def test_con2_full_conditional_shuffle_ipf_seeded():
    from mlframe.feature_selection.filters._cat_confirm_permutation import _full_conditional_shuffle_ipf

    y = np.array([0, 1, 0, 1, 0, 1, 1, 0] * 4, dtype=np.int64)
    x1 = np.array([0, 0, 1, 1] * 8, dtype=np.int64)
    base = np.arange(len(y)).astype(np.int64)

    st = np.random.get_state()
    a = base.copy()
    _full_conditional_shuffle_ipf(a, x1, y, 2, 2, 777)
    b = base.copy()
    _full_conditional_shuffle_ipf(b, x1, y, 2, 2, 777)
    c = base.copy()
    _full_conditional_shuffle_ipf(c, x1, y, 2, 2, 778)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert _global_state_unchanged(st)


def test_con3_shuffle_and_compute_three_mis_seeded():
    from mlframe.feature_selection.filters._cat_confirm_permutation import _shuffle_and_compute_three_mis

    n = 200
    rng = np.random.default_rng(0)
    cls_pair = rng.integers(0, 6, n).astype(np.int64)
    cls_x1 = rng.integers(0, 3, n).astype(np.int64)
    cls_x2 = rng.integers(0, 3, n).astype(np.int64)
    y = rng.integers(0, 2, n).astype(np.int64)

    def _freqs(c, k):
        return (np.bincount(c, minlength=k).astype(np.float64) / n)

    fq_pair, fq_x1, fq_x2, fq_y = _freqs(cls_pair, 6), _freqs(cls_x1, 3), _freqs(cls_x2, 3), _freqs(y, 2)

    st = np.random.get_state()
    r1 = _shuffle_and_compute_three_mis(cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2, y.copy(), fq_y, 555, np.int64)
    r2 = _shuffle_and_compute_three_mis(cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2, y.copy(), fq_y, 555, np.int64)
    r3 = _shuffle_and_compute_three_mis(cls_pair, fq_pair, cls_x1, fq_x1, cls_x2, fq_x2, y.copy(), fq_y, 556, np.int64)

    assert r1 == r2, "same base_seed must give identical three-MI tuple"
    assert r1 != r3, "different base_seed must produce a different shuffle outcome"
    assert _global_state_unchanged(st)


def test_con4_group_aware_shuffle_seeded():
    from mlframe.feature_selection.filters._cat_target_encoding_and_weighted import _group_aware_shuffle

    n_groups = 12
    groups = np.repeat(np.arange(n_groups), 5).astype(np.int64)
    y = (np.arange(len(groups)) % 2).astype(np.int64)

    st = np.random.get_state()
    a = y.copy()
    _group_aware_shuffle(a, groups, n_groups, 31337)
    b = y.copy()
    _group_aware_shuffle(b, groups, n_groups, 31337)
    c = y.copy()
    _group_aware_shuffle(c, groups, n_groups, 42)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert _global_state_unchanged(st)
