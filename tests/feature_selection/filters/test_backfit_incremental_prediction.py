"""The Gauss-Seidel backfit keeps its prediction by increments and still equals the textbook backfit that re-sums every effect table per column."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils_offset import _backfit


def _reference(Q, y, even, n_bins, sweeps):
    """Textbook backfit in numpy: the prediction is rebuilt from all tables after every column update."""
    p, n = Q.shape
    f = np.zeros((p, n_bins))
    mu = y[even].mean() if even.any() else 0.0
    pred = np.zeros(n)
    for _ in range(sweeps):
        for j in range(p):
            r = y - mu - (pred - f[j][Q[j]])
            s = np.bincount(Q[j][even], weights=r[even], minlength=n_bins)
            c = np.bincount(Q[j][even], minlength=n_bins)
            f[j] = np.where(c > 0, s / np.maximum(c, 1), 0.0)
            pred = sum(f[jj][Q[jj]] for jj in range(p))
    return f, mu


@pytest.mark.parametrize("p,nb", [(1, 8), (6, 16), (25, 12)])
def test_equals_the_textbook_backfit(p, nb):
    """Random bins and a target built from some effects: the effect tables and the intercept agree to rounding."""
    rng = np.random.default_rng(p)
    n = 3000
    Q = rng.integers(0, nb, (p, n)).astype(np.int64)
    y = rng.standard_normal(n) + 0.5 * (Q[0] / nb)
    even = np.arange(n) % 2 == 0
    f, mu = _backfit(Q, y, even, nb, 4)
    f_ref, mu_ref = _reference(Q, y, even, nb, 4)
    assert abs(mu - mu_ref) < 1e-12
    assert np.allclose(f, f_ref, rtol=0, atol=1e-10)


def test_empty_bins_stay_zero():
    """A bin no even row falls into keeps the effect 0, as before."""
    Q = np.zeros((2, 40), dtype=np.int64)
    Q[1] = 3
    y = np.arange(40, dtype=np.float64)
    even = np.arange(40) % 2 == 0
    f, _ = _backfit(Q, y, even, 6, 3)
    assert (f[0, 1:] == 0).all() and (f[1, [0, 1, 2, 4, 5]] == 0).all()
