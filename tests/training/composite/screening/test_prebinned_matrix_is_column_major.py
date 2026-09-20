"""The prebinned code matrix is read one column at a time, so it must be stored that way.

``_mi_per_feature_prebinned`` walks a single int16 column element by element, and so do the per-candidate ``MI(T, X)``
and the ``mi_y`` comparison. In C order each of those reads jumps ``2 * F`` bytes, touching a fresh cache line per
element; the MI pass measured 239 ms against 35 ms at n=100k, F=100 on this host, and 1958 ms against 82 ms at F=300.
The values are unchanged: only the layout differs.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery.screening import _mi_per_feature_prebinned, _prebin_feature_columns


def _features(n: int = 2000, f: int = 12, seed: int = 0) -> np.ndarray:
    """A float32 feature block wide enough for the layout to matter."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, f)).astype(np.float32)


def test_the_prebinned_matrix_is_column_major():
    """The eager prebin path returns an F-contiguous matrix, which is how every consumer reads it."""
    codes = _prebin_feature_columns(_features(), nbins=8)
    assert codes.flags["F_CONTIGUOUS"], "the code matrix must be column-major for the per-column MI walk"


def test_per_feature_mi_is_identical_under_either_layout():
    """Layout is a storage choice: the MI vector must match the C-order one exactly, not approximately."""
    codes = _prebin_feature_columns(_features(), nbins=8)
    rng = np.random.default_rng(1)
    y = rng.integers(0, 8, size=codes.shape[0]).astype(codes.dtype)

    mi_f = _mi_per_feature_prebinned(codes, y, nbins=8)
    mi_c = _mi_per_feature_prebinned(np.ascontiguousarray(codes), y, nbins=8)
    np.testing.assert_array_equal(np.asarray(mi_f), np.asarray(mi_c))


def test_dropping_a_base_column_keeps_the_layout():
    """Each base screens with its own column removed; that delete must not silently flip the matrix back to C order."""
    codes = _prebin_feature_columns(_features(), nbins=8)
    without_base = np.delete(codes, 3, axis=1)
    assert without_base.flags["F_CONTIGUOUS"], "np.delete(axis=1) must preserve the column-major layout"


def test_the_exclude_col_vector_matches_deleting_that_column():
    """The no-copy exclude path and the materialised delete must agree, as they did before the layout change."""
    codes = _prebin_feature_columns(_features(), nbins=8)
    rng = np.random.default_rng(2)
    y = rng.integers(0, 8, size=codes.shape[0]).astype(codes.dtype)

    excluded = _mi_per_feature_prebinned(codes, y, nbins=8, exclude_col=3)
    deleted = _mi_per_feature_prebinned(np.delete(codes, 3, axis=1), y, nbins=8)
    np.testing.assert_array_equal(np.asarray(excluded), np.asarray(deleted))
