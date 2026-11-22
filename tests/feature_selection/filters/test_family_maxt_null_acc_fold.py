"""The ``acc`` argument of the family null folds each block's maxima into one running vector, so a caller reads the result back once.

The fold must equal the maxima returned without ``acc`` for the same generator state, block by block, and it must return ``None``.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.feature_selection.filters._family_maxt_gate as gate

cp = pytest.importorskip("cupy")


def _inputs(seed: int = 7):
    """Five candidate columns of four bins on 400 rows, with a three-class label that carries signal on the first two columns."""
    rng = np.random.default_rng(seed)
    n = 400
    y = rng.integers(0, 3, n).astype(np.int64)
    cols = rng.integers(0, 4, (5, n)).astype(np.int64)
    cols[0] = (y + rng.integers(0, 2, n)) % 4
    cols[1] = (2 * y + rng.integers(0, 2, n)) % 4
    n_x = np.full(5, 4, dtype=np.int64)
    return cols, n_x, y


def test_folding_into_acc_returns_none_and_matches_the_maxima_block_by_block():
    """With ``acc`` the function returns ``None`` and ``acc`` holds the same maxima as the call without it, over several blocks."""
    cols, n_x, y = _inputs()
    n_perm = 20
    ref = gate.family_null_max(cols, n_x, y, 3, n_perm, np.random.default_rng(11), block=8)
    acc = cp.full(n_perm, -np.inf, dtype=cp.float64)
    out = gate.family_null_max(cols, n_x, y, 3, n_perm, np.random.default_rng(11), block=8, acc=acc)
    assert out is None
    np.testing.assert_allclose(cp.asnumpy(acc), ref, rtol=0, atol=0)


def test_a_second_fold_keeps_the_larger_maxima_per_permutation():
    """Folding a second column group into the same ``acc`` keeps, for each permutation, the larger of the two groups' maxima."""
    cols, n_x, y = _inputs(3)
    n_perm = 16
    a_ref = gate.family_null_max(cols[:2], n_x[:2], y, 3, n_perm, np.random.default_rng(5), block=16)
    b_ref = gate.family_null_max(cols[2:], n_x[2:], y, 3, n_perm, np.random.default_rng(5), block=16)
    acc = cp.full(n_perm, -np.inf, dtype=cp.float64)
    gate.family_null_max(cols[:2], n_x[:2], y, 3, n_perm, np.random.default_rng(5), block=16, acc=acc)
    gate.family_null_max(cols[2:], n_x[2:], y, 3, n_perm, np.random.default_rng(5), block=16, acc=acc)
    np.testing.assert_allclose(cp.asnumpy(acc), np.maximum(a_ref, b_ref), rtol=0, atol=0)
