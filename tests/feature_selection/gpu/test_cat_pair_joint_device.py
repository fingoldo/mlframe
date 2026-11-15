"""The device joint MI of categorical pairs equals the numba pass, oversize pairs fall back to the host, and the pair-table stage is unchanged by the device path."""

from __future__ import annotations

import numpy as np
import pytest

from tests.feature_selection.gpu._gpu_support import has_device

from mlframe.feature_selection.filters import _cat_pair_joint_gpu as g
from mlframe.feature_selection.filters._cat_pair_joint_njit import pair_joint_mi


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")


def _inputs(levels=(3, 12, 7, 40), n=6000, k_y=3):
    """Categorical code columns of different cardinality and a target tied to the first two."""
    rng = np.random.default_rng(2)
    codes = np.stack([rng.integers(0, r, n) for r in levels]).astype(np.int64)
    radix = codes.max(axis=1) + 1
    y = ((codes[0] + codes[1]) % k_y).astype(np.int64)
    pairs = [(a, b) for a in range(len(levels)) for b in range(len(levels)) if a != b]
    first = np.array([a for a, _ in pairs], dtype=np.int64)
    second = np.array([b for _, b in pairs], dtype=np.int64)
    return codes, radix, first, second, y, k_y


def test_device_equals_numba_pass():
    """Every pair that fits the shared table agrees with the host to 1e-12."""
    _needs_gpu()
    args = _inputs()
    host = pair_joint_mi(*args)
    dev = g.pair_joint_mi_gpu(*args)
    ok = ~np.isnan(dev)
    assert ok.sum() >= 10
    np.testing.assert_allclose(dev[ok], host[ok], rtol=1e-12, atol=1e-12)


def test_oversize_pairs_are_filled_by_the_dispatcher(monkeypatch):
    """With a tiny table limit the device returns NaN for those pairs and the dispatcher fills them from the numba pass."""
    _needs_gpu()
    args = _inputs()
    monkeypatch.setattr(g, "_MAX_CELLS", 200)
    monkeypatch.setattr(g, "pair_device_chosen", lambda n, p: True)
    assert np.isnan(g.pair_joint_mi_gpu(*args)).any()
    np.testing.assert_allclose(g.pair_joint_mi_dispatch(*args), pair_joint_mi(*args), rtol=1e-12, atol=1e-12)
