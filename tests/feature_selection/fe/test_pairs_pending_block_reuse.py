"""``_score_pending_candidates`` hands the batched noise gate the candidates' codes as an ``(n, k)`` C-ordered narrow matrix.

The candidates are normally the columns, in order, of the matrix the block binner returned; the matrix is then narrowed in one pass instead of being rebuilt through a column-by-column copy into rows and a transpose back.
Anything that is not exactly those columns (a shuffled order, a column of another matrix, a copied column, a different row count) must take the old path, and both must give the gate the same matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_recompute as rec


def _entries(columns):
    """Pending entries ``(config, pair, op, uses_pw, codes)`` over the given code columns."""
    return [(("cfg", i), None, "op", False, col) for i, col in enumerate(columns)]


def _capture(monkeypatch, columns, n_rows, dtype=np.int32):
    """The ``disc_2d`` matrix that the batched noise gate receives for these candidate columns."""
    seen = {}

    def fake_dispatch(**kw):
        """Record the matrix handed to the gate and score it as all zeros."""
        seen["disc"] = kw["disc_2d"]
        return np.zeros(kw["disc_2d"].shape[1])

    monkeypatch.setattr(rec, "_dispatch_batch_mi_with_noise_gate", fake_dispatch)
    rec._score_pending_candidates(_entries(columns), n_rows, 10, dtype, None, None, None, 0, 0.0, None, lambda *a, **k: None, None, None)
    return seen["disc"]


def _matrix(n=300, k=7, dtype=np.int32, seed=0):
    """A small (n, k) code matrix with levels 0 to 9."""
    return np.random.default_rng(seed).integers(0, 10, (n, k)).astype(dtype)


def test_the_columns_of_one_matrix_are_found():
    """Column views of one C-ordered matrix, in order, are recognised and the matrix itself is returned (no copy)."""
    m = _matrix()
    assert rec._columns_of_one_matrix([m[:, j] for j in range(m.shape[1])], m.shape[0]) is m


@pytest.mark.parametrize("make", ["shuffled", "other_matrix", "copied", "fewer", "rows", "fortran"])
def test_anything_else_is_not_taken_for_one_matrix(make):
    """A swapped pair, a column of another matrix, a copied column, a missing column, another row count or a Fortran-ordered base all return None."""
    m = _matrix()
    cols = [m[:, j] for j in range(m.shape[1])]
    n = m.shape[0]
    if make == "shuffled":
        cols[0], cols[1] = cols[1], cols[0]
    elif make == "other_matrix":
        cols[3] = _matrix(seed=1)[:, 3]
    elif make == "copied":
        cols[2] = m[:, 2].copy()
    elif make == "fewer":
        cols = cols[:-1]
    elif make == "rows":
        n += 1
    elif make == "fortran":
        f = np.asfortranarray(m)
        cols = [f[:, j] for j in range(f.shape[1])]
    assert rec._columns_of_one_matrix(cols, n) is None


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_both_routes_give_the_gate_the_same_narrow_c_ordered_matrix(monkeypatch, dtype):
    """The one-pass route and the copy-and-transpose route hand the noise gate the same int8 C-ordered matrix."""
    m = _matrix(dtype=dtype)
    fast = _capture(monkeypatch, [m[:, j] for j in range(m.shape[1])], m.shape[0], dtype)
    slow = _capture(monkeypatch, [m[:, j].copy() for j in range(m.shape[1])], m.shape[0], dtype)
    assert fast.dtype == slow.dtype == np.int8 and fast.flags.c_contiguous and slow.flags.c_contiguous
    np.testing.assert_array_equal(fast, slow)
    np.testing.assert_array_equal(fast, m.astype(np.int8))
