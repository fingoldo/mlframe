"""The cupy pairwise-complete correlation in tiles equals the whole-matrix result and the numpy backend."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_dedup as D


def test_tiled_cupy_corr_equals_whole_and_numpy(monkeypatch):
    """With a room small enough to force tiles the result equals the one-shot device result and the numpy backend (NaN pattern included)."""
    r = np.random.default_rng(0)
    n = 600
    Q = r.standard_normal((9, n)).astype(np.float32)
    R = r.standard_normal((7, n)).astype(np.float32)
    Q[2, :40] = np.nan
    R[3, 100:140] = np.nan
    Q[5] = Q[1] * 2.0
    whole = D._pc_corr_cupy(Q, R)
    monkeypatch.setattr("mlframe.feature_selection.filters._device_sort_safe.device_room", lambda cp_: 4 * (10 * 3 * n * 4 + 9 * 1024 * 1024 * 4))
    tiled = D._pc_corr_cupy(Q, R)
    np.testing.assert_allclose(tiled, whole, rtol=1e-5, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(tiled, D._pc_corr_numpy(Q, R), rtol=1e-3, atol=1e-4, equal_nan=True)
