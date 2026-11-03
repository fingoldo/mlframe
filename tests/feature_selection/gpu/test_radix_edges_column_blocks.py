"""When the whole-matrix transpose does not fit, the resident MI edges come from column blocks with the same values as the one-shot select."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import mlframe.feature_selection.filters._gpu_resident_select as S
from mlframe.feature_selection.filters._hermite_fe_mi import _radix_edges_in_column_blocks, _resident_mi_candidates


def test_blocks_equal_one_shot(monkeypatch):
    """A select that runs out of memory above 3 columns: the block path halves until it fits and concatenates the same edges."""
    rng = np.random.default_rng(0)
    X = cp.asarray(rng.standard_normal((5000, 11)).astype(np.float32))
    want = S._radix_select_interior_edges(X, 10)
    real = S._radix_select_interior_edges

    def picky(x, nb, **kw):
        """Out of memory for more than three columns."""
        if x.shape[1] > 3:
            raise cp.cuda.memory.OutOfMemoryError(1 << 30, 0, 0)
        return real(x, nb, **kw)

    monkeypatch.setattr(S, "_radix_select_interior_edges", picky)
    got = _radix_edges_in_column_blocks(cp, X, 10)
    assert got is not None and cp.array_equal(got, want)


def test_resident_candidates_use_blocks_on_oom(monkeypatch):
    """``_resident_mi_candidates`` takes the block path when the one-shot select runs out of memory (the edges equal the one-shot ones)."""
    rng = np.random.default_rng(1)
    X = cp.asarray(rng.standard_normal((4000, 6)).astype(np.float32))
    want = _resident_mi_candidates(X, 10, keep_dtype=True, relax_binning=False, x_is_cm=False)[3]
    real = S._radix_select_interior_edges

    def picky(x, nb, **kw):
        """Out of memory for the whole matrix only."""
        if x.shape[1] == 6:
            raise cp.cuda.memory.OutOfMemoryError(1 << 30, 0, 0)
        return real(x, nb, **kw)

    monkeypatch.setattr(S, "_radix_select_interior_edges", picky)
    got = _resident_mi_candidates(X, 10, keep_dtype=True, relax_binning=False, x_is_cm=False)[3]
    assert cp.array_equal(got, want)
