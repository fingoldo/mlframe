"""The batched wavelet leg selection gives the same admitted legs whatever the leg block size.

The leg code matrices are stacked and scored in blocks bounded by ``_LEG_BLOCK_BYTES``. A leg's MI depends on that leg alone, so a block of one leg
and a single stack of all legs must admit the same ``(j, k)`` legs in the same order. The device-born route stacks on the device, so the test forces
the host body.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from mlframe.feature_selection.filters import _wavelet_basis_fe_batched as B


@pytest.fixture
def host_body(monkeypatch):
    """Route the leg selection through the host body (the device-born route is switched off for the test)."""
    import mlframe.feature_selection.filters._gpu_strict_fe as S

    monkeypatch.setattr(S, "fe_gpu_device_born_wavelet_enabled", lambda: False)


def _case(n=3000, seed=0):
    """A binary target that depends on a threshold of one continuous column, with the column range the leg selection maps onto [0, 1]."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = ((x > 0.4) ^ (rng.random(n) < 0.1)).astype(np.int64)
    lo, span = float(x.min()), float(x.max() - x.min())
    return x, y, lo, span


@pytest.mark.usefixtures("host_body")
def test_one_leg_per_block_admits_the_same_legs(monkeypatch):
    """A budget of one leg per block admits exactly the legs a single stack admits, in the same order."""
    # The host body scores legs through the cupy MI kernel, so it cannot run without cupy. Production falls back to the CPU
    # leg selection when cupy is missing (_try_gpu_leg_selection); this test only covers the GPU body, so it skips without cupy.
    pytest.importorskip("cupy")
    x, y, lo, span = _case()
    kw = dict(max_scale=4, max_legs=40, scale_sigma=3.0)
    monkeypatch.setattr(B, "_LEG_BLOCK_BYTES", 1 << 40)
    whole = B.select_wavelet_legs_batched(x, y, lo, span, **kw)
    monkeypatch.setattr(B, "_LEG_BLOCK_BYTES", 1)
    blocked = B.select_wavelet_legs_batched(x, y, lo, span, **kw)
    assert whole, "the case must admit at least one leg, or the comparison proves nothing"
    assert blocked == whole


def test_device_route_one_leg_per_block_admits_the_same_legs(monkeypatch):
    """The device-born route scores legs in byte-bounded blocks: a budget of one leg per block admits the same legs, in the same order, as one block."""
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    import mlframe.feature_selection.filters._gpu_strict_fe as S

    monkeypatch.setattr(S, "fe_gpu_device_born_wavelet_enabled", lambda: True)
    x, y, lo, span = _case(seed=3)
    kw = dict(max_scale=4, max_legs=40, scale_sigma=3.0)
    monkeypatch.setattr(B, "_DEVICE_LEG_BLOCK_BYTES", 1 << 40)
    whole = B._select_wavelet_legs_batched_device(x, y, lo, span, **kw)
    monkeypatch.setattr(B, "_DEVICE_LEG_BLOCK_BYTES", 1)
    blocked = B._select_wavelet_legs_batched_device(x, y, lo, span, **kw)
    assert whole, "the device route must admit at least one leg, or the comparison proves nothing"
    assert blocked == whole
