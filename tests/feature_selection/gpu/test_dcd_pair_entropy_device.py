"""The device pair joint entropies equal the numba kernels to rounding, oversize pairs keep the host value, and the batch entry point is unchanged with the device forced on."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _dcd_pair_su_batch as M
from mlframe.feature_selection.filters import _dcd_pair_su_gpu as g


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")


def _data(n=6000):
    """A code matrix with columns of 2, 5, 12 and 200 levels and all ordered pairs."""
    rng = np.random.default_rng(1)
    levels = [2, 5, 12, 200, 12]
    fd = np.stack([rng.integers(0, r, n) for r in levels], axis=1).astype(np.int32)
    nb = np.array(levels, dtype=np.int64)
    pairs = [(a, b) for a in range(5) for b in range(5) if a < b]
    return fd, np.array([a for a, _ in pairs], dtype=np.int64), np.array([b for _, b in pairs], dtype=np.int64), nb


def test_device_equals_host_to_rounding():
    """Every pair that fits agrees with the numba kernel to 1e-12."""
    _needs_gpu()
    fd, a, b, nb = _data()
    dev = g.joint_entropies_gpu(fd, a, b, nb)
    host = M._joint_entropies_host(fd, a, b, nb)
    ok = ~np.isnan(dev)
    assert ok.sum() >= 6
    np.testing.assert_allclose(dev[ok], host[ok], rtol=1e-12, atol=1e-12)


def test_oversize_pairs_use_the_host(monkeypatch):
    """With a small cell limit the device marks those pairs NaN and the dispatcher returns the host values for them."""
    _needs_gpu()
    fd, a, b, nb = _data()
    monkeypatch.setattr(g, "_MAX_CELLS", 40)
    monkeypatch.setattr(g, "entropy_device_chosen", lambda n, k: True)
    assert np.isnan(g.joint_entropies_gpu(fd, a, b, nb)).any()
    np.testing.assert_allclose(M.joint_entropies_of_pairs(fd, a, b, nb), M._joint_entropies_host(fd, a, b, nb), rtol=1e-12, atol=1e-12)
