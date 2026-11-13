"""``_code_itemsize`` answers which integer width the family's codes need; scanning the parent of column views once must give exactly the answer of scanning every view."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._family_maxt_stream import _code_itemsize


def _reference(mats):
    """The width by the plain scan of every matrix that can exceed 127."""
    scan = [m for m in mats if m.size and not (m.dtype.kind == "i" and np.iinfo(m.dtype).max <= 127)]
    top = max((int(m.max()) for m in scan), default=0)
    return 1 if top <= 127 else (2 if top <= 32767 else 8)


def _columns(base):
    """Single-column ``(n, 1)`` views of ``base``, the way the pair step keeps its candidates' codes."""
    return [np.asarray(base[:, j]).reshape(-1, 1) for j in range(base.shape[1])]


@pytest.mark.parametrize("peak", [9, 127, 128, 32767, 32768, 200000])
@pytest.mark.parametrize("where", ["first", "last"])
def test_column_views_of_one_matrix_give_the_reference_width(peak, where):
    """A peak in the first or last column, at and around each width limit."""
    base = np.random.default_rng(0).integers(0, 10, (500, 12)).astype(np.int32)
    base[7, 0 if where == "first" else -1] = peak
    mats = _columns(base)
    assert _code_itemsize(mats) == _reference(mats)


def test_views_covering_a_small_part_of_a_big_parent_ignore_the_rest():
    """Two columns of a 40-column parent whose other columns hold large codes: the views alone decide (width 1)."""
    base = np.random.default_rng(1).integers(0, 10, (500, 40)).astype(np.int32)
    base[:, 10:] = 40000
    mats = _columns(base)[:2]
    assert _code_itemsize(mats) == _reference(mats) == 1


def test_a_parent_with_a_large_unviewed_code_does_not_inflate_the_answer():
    """Views cover most of the parent but not the column that holds the large code: the exact views decide."""
    base = np.random.default_rng(2).integers(0, 10, (500, 4)).astype(np.int32)
    base[:, 3] = 70000
    mats = _columns(base)[:3]
    assert _code_itemsize(mats) == _reference(mats) == 1


def test_mixed_independent_arrays_and_views():
    """Independent arrays, int8 matrices (skipped by type) and views of two parents together."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 10, (300, 5)).astype(np.int32)
    b = rng.integers(0, 10, (300, 5)).astype(np.int16)
    b[1, 2] = 400
    mats = _columns(a) + _columns(b) + [np.zeros((300, 2), dtype=np.int8), rng.integers(0, 5, (300, 1)).astype(np.int32)]
    assert _code_itemsize(mats) == _reference(mats) == 2


def test_empty_and_int8_only_families():
    """Nothing to scan gives the narrowest width."""
    assert _code_itemsize([]) == _reference([]) == 1
    assert _code_itemsize([np.zeros((10, 3), dtype=np.int8)]) == 1
