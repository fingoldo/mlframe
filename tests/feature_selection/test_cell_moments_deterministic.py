"""The cell-moment kernels return the same bits on every call and the moments of the atomic reference."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _cell_moments_kernel as K


def _data(n=50_000, nc=40, nf=5, seed=0):
    """Cell codes, a value column with a few non-finite rows removed, weights and fold labels."""
    rng = np.random.default_rng(seed)
    codes = cp.asarray(rng.integers(0, nc, n).astype(np.int64))
    v = cp.asarray(rng.standard_normal(n) * 3 + 1)
    w = cp.asarray((rng.random(n) > 0.1).astype(np.float64) * rng.random(n))
    fin = cp.asarray((rng.random(n) > 0.05).astype(np.float64))
    folds = cp.asarray(rng.integers(0, nf, n).astype(np.int64))
    return codes, v, w, fin, folds, nc, nf


def _close(a, b):
    """Equal up to rounding of differently ordered sums."""
    for x, y in zip(a, b):
        np.testing.assert_allclose(cp.asnumpy(x), cp.asnumpy(y), rtol=1e-9, atol=1e-9)


def test_masked_is_bitwise_repeatable_and_matches_the_atomic_form():
    """Ten calls give one set of bits; the values match the atomic kernels."""
    codes, v, w, _fin, _f, nc, _nf = _data()
    first = K.masked_cell_moments_deterministic(cp, codes, v, w, nc)
    for _ in range(9):
        again = K.masked_cell_moments_deterministic(cp, codes, v, w, nc)
        assert all(np.array_equal(cp.asnumpy(a), cp.asnumpy(b)) for a, b in zip(first, again))
    _close(first, K.masked_cell_moments(cp, codes, v, w, nc))


def test_fold_is_bitwise_repeatable_and_matches_the_atomic_form():
    """The all-folds tables: repeatable bits, atomic values."""
    codes, v, _w, fin, folds, nc, nf = _data()
    first = K.fold_cell_moments_deterministic(cp, codes, v, fin, folds, nf, nc)
    assert first[0].shape == (nf, nc)
    for _ in range(9):
        again = K.fold_cell_moments_deterministic(cp, codes, v, fin, folds, nf, nc)
        assert all(np.array_equal(cp.asnumpy(a), cp.asnumpy(b)) for a, b in zip(first, again))
    _close(first, K.fold_cell_moments(cp, codes, v, fin, folds, nf, nc))
