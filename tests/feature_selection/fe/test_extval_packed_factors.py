"""The packed external-factor matrix equals the column fill, is reused for the identical factor arrays and rebuilt for any other."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_emit_extval as E


def _factors(k=5, n=400, seed=0):
    """Distinct factor arrays."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n) for _ in range(k)]


def test_packed_matrix_equals_the_column_fill():
    """Column e of the packed matrix is factor e, C-contiguous, float64."""
    E._release_packed_factors()
    f = _factors()
    got = E._packed_factor_matrix(f, 400)
    want = np.empty((400, len(f)))
    for i, v in enumerate(f):
        want[:, i] = v
    assert got.flags.c_contiguous and got.dtype == np.float64
    np.testing.assert_array_equal(got, want)


def test_identical_arrays_reuse_the_matrix_and_any_change_rebuilds_it():
    """Same objects: the same matrix object. A different array, a different order, a different length of the list: a new, correct matrix."""
    E._release_packed_factors()
    f = _factors()
    a = E._packed_factor_matrix(f, 400)
    assert E._packed_factor_matrix(list(f), 400) is a
    g = list(f)
    g[2] = f[2].copy()
    b = E._packed_factor_matrix(g, 400)
    assert b is not a
    np.testing.assert_array_equal(b, a)
    c = E._packed_factor_matrix(f[::-1], 400)
    np.testing.assert_array_equal(c, a[:, ::-1])
    d = E._packed_factor_matrix(f[:3], 400)
    assert d.shape == (400, 3)
    assert E._release_packed_factors() == d.nbytes
    assert E._release_packed_factors() == 0
