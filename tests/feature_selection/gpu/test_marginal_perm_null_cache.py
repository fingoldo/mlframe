"""The marginal permutation null reuses the permutation matrix of the last (seed, rows, draws): the floors and means equal a fresh draw's exactly, the matrix is drawn once for many columns, another seed or row count draws again, and a matrix over the byte cap is not kept."""

from __future__ import annotations

import numpy as np
import pytest

from tests.feature_selection.gpu._gpu_support import has_device

from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as m


def _needs_gpu():
    """cupy with a usable CUDA device, else skip."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    return cp


def _null(cp, col, y, seed=7):
    """The marginal null of one device column."""
    return m.conditional_perm_null_gpu(col, y, None, order=None, z_rank=None, n_permutations=12, quantile=0.95, seed=seed, salt=1)


def test_cached_matrix_gives_the_fresh_draws_floors():
    """Several columns over the same rows: every (floor, mean) equals the one computed with the cache cleared before the call."""
    cp = _needs_gpu()
    rng = np.random.default_rng(0)
    n = 6000
    y = rng.integers(0, 2, n).astype(np.int64)
    cols = [cp.asarray(rng.integers(0, 8, n).astype(np.int64)) for _ in range(4)]
    cached = [_null(cp, c, y) for c in cols]
    fresh = []
    for c in cols:
        m._MARGINAL_PERM_CACHE.clear()
        fresh.append(_null(cp, c, y))
    assert cached == fresh


def test_matrix_is_drawn_once_per_key(monkeypatch):
    """Four columns, one draw; a different seed or row count draws again."""
    cp = _needs_gpu()
    calls = []
    real = m.device_argsort
    monkeypatch.setattr(m, "device_argsort", lambda keys, axis=0: (calls.append(1), real(keys, axis=axis))[1])
    m._MARGINAL_PERM_CACHE.clear()
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 2, n).astype(np.int64)
    cols = [cp.asarray(rng.integers(0, 5, n).astype(np.int64)) for _ in range(4)]
    for c in cols:
        _null(cp, c, y)
    assert len(calls) == 1
    _null(cp, cols[0], y, seed=8)
    assert len(calls) == 2
    _null(cp, cols[0][:2000], y[:2000])
    assert len(calls) == 3


def test_matrix_over_the_byte_cap_is_not_kept(monkeypatch):
    """With a cap below the matrix size every call draws again and nothing is stored."""
    cp = _needs_gpu()
    m._MARGINAL_PERM_CACHE.clear()
    monkeypatch.setattr(m, "_MARGINAL_PERM_CACHE_MAX_BYTES", 1)
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.integers(0, 2, n).astype(np.int64)
    col = cp.asarray(rng.integers(0, 5, n).astype(np.int64))
    first = _null(cp, col, y)
    assert _null(cp, col, y) == first
    assert m._MARGINAL_PERM_CACHE == []
