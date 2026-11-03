"""The recurrence build of the adaptive Fourier detector's coarse basis matches the direct sin / cos build and gives the same emitted columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe import _coarse_basis_njit, _coarse_basis_recur_njit, _uniform_grid_step
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import generate_extra_basis_features


def test_basis_matches_direct_build():
    """The 48-frequency chirp grid on z in [-4, 4]: centred bases and sums of squares agree to round-off."""
    z = np.random.default_rng(0).uniform(-4, 4, 20011)
    grid = [0.5 * k for k in range(1, 49)]
    assert _uniform_grid_step(grid) == 0.5
    want = _coarse_basis_njit(z, np.asarray(grid, dtype=np.float32))
    got = _coarse_basis_recur_njit(z, 0.5, len(grid))
    for w, g in zip(want, got):
        np.testing.assert_allclose(g, w, rtol=1e-9, atol=1e-9)


def test_non_uniform_grid_keeps_the_direct_build():
    """A grid that is not ``(k + 1) * step`` has no recurrence step."""
    assert _uniform_grid_step([0.5, 1.0, 1.75]) == 0.0
    assert _uniform_grid_step([1.0]) == 0.0


def test_same_emitted_columns(monkeypatch):
    """Periodic and chirp-like signals with both detectors on: the recurrence and the direct build emit the same columns and metadata."""
    rng = np.random.default_rng(4)
    n = 6000
    a, b = rng.uniform(-3, 3, n), rng.uniform(-2, 2, n)
    y = np.sin(2.7 * a) + 0.5 * np.cos(np.sign(b) * b * b * 3.1) + 0.3 * rng.standard_normal(n)
    X = pd.DataFrame({"a": a, "b": b, "c": rng.standard_normal(n)})
    kw = dict(extra_bases=("fourier",), y=y, fourier_adaptive=True, fourier_chirp=True, max_adaptive_cols=100)
    monkeypatch.setenv("MLFRAME_FOURIER_COARSE_RECUR", "0")
    want_cols, want_meta = generate_extra_basis_features(X, **kw)
    monkeypatch.setenv("MLFRAME_FOURIER_COARSE_RECUR", "1")
    got_cols, got_meta = generate_extra_basis_features(X, **kw)
    assert got_cols.equals(want_cols)
    assert repr(got_meta) == repr(want_meta)
