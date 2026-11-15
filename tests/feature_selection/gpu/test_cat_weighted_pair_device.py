"""The device weighted categorical pair search equals the numba weighted kernel, is reproducible, and oversize pairs go to the host through the dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _cat_weighted_pair_gpu as g
from mlframe.feature_selection.filters._cat_target_encoding_and_weighted import _pair_search_kernel_weighted_njit as host


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")


def _inputs(n=6000, levels=(3, 12, 7, 40, 5)):
    """Categorical columns of different cardinality, a 3-class target tied to two of them, float32 weights (the pipeline's critical dtype) with a few zero weights."""
    rng = np.random.default_rng(3)
    data = np.stack([rng.integers(0, r, n) for r in levels], axis=1).astype(np.int32)
    nbins = np.array(levels, dtype=np.int64)
    y = ((data[:, 0] + data[:, 1]) % 3).astype(np.int64)
    w = (rng.random(n) + 0.25).astype(np.float32)
    w[::97] = 0.0
    pairs = [(a, b) for a in range(len(levels)) for b in range(len(levels)) if a < b]
    return data, np.array([a for a, _ in pairs], dtype=np.int64), np.array([b for _, b in pairs], dtype=np.int64), rng.random(len(levels)) * 0.01, nbins, y, w


def test_device_equals_the_numba_weighted_kernel():
    """Joint MI, interaction information and occupied cells agree with the host kernel to 1e-12 relative for every pair that fits the table."""
    _needs_gpu()
    data, a, b, marg, nbins, y, w = _inputs()
    want_mi, want_ii, want_u = host(data, a, b, marg, nbins, y, w, np.int32)
    got_mi, got_ii, got_u, fits = g.weighted_pair_search_gpu(data, a, b, marg, nbins, y, w)
    assert fits.sum() >= 6
    np.testing.assert_allclose(got_mi[fits], want_mi[fits], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(got_ii[fits], want_ii[fits], rtol=1e-12, atol=1e-12)
    assert np.array_equal(got_u[fits], want_u[fits])


def test_reproducible_and_oversize_pairs_use_the_host(monkeypatch):
    """Two runs give identical bits; with a small table limit the dispatcher returns the host kernel's values for the pairs the device skips."""
    _needs_gpu()
    data, a, b, marg, nbins, y, w = _inputs()
    r1 = g.weighted_pair_search_gpu(data, a, b, marg, nbins, y, w)
    r2 = g.weighted_pair_search_gpu(data, a, b, marg, nbins, y, w)
    assert np.array_equal(r1[0], r2[0])
    monkeypatch.setattr(g, "_MAX_CELLS", 150)
    monkeypatch.setattr(g, "weighted_device_chosen", lambda n, k: True)
    assert not g.weighted_pair_search_gpu(data, a, b, marg, nbins, y, w)[3].all()
    got = g.weighted_pair_search_dispatch(host, data, a, b, marg, nbins, y, w, np.int32)
    want = host(data, a, b, marg, nbins, y, w, np.int32)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-12, atol=1e-13)
    assert np.array_equal(got[2], want[2])
