"""The cross-basis resident scorer takes the route the kernel-tuning entry chooses (None -> the host scorer when the host is chosen) and its column blocks give the single-matrix MI."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _device_sort_safe as dss
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _gpu_resident_cross_basis as cb


def _cp():
    """cupy with a usable device, else skip."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
    return cp


def test_route_follows_the_tuner(monkeypatch):
    """With the host chosen ``raw_and_product_mi_resident`` answers None (the caller's host scorer runs); with the device chosen it returns the raw and engineered MI."""
    _cp()
    X, eng, specs, y, nbins = cb._sweep_inputs({"n_samples": 3000})
    monkeypatch.setattr(cb, "crossbasis_device_chosen", lambda n: False)
    assert cb.raw_and_product_mi_resident(X, eng, y, specs, nbins=nbins, basis="auto") is None
    monkeypatch.setattr(cb, "crossbasis_device_chosen", lambda n: True)
    got = cb.raw_and_product_mi_resident(X, eng, y, specs, nbins=nbins, basis="auto")
    assert got is not None and got[1].shape == (eng.shape[1],)


def test_column_blocks_give_the_single_matrix_mi(monkeypatch):
    """A room that holds about three columns forces many blocks; the MI equals the one-block run exactly."""
    cp = _cp()
    X, _eng, specs, y, nbins = cb._sweep_inputs({"n_samples": 3000})
    whole = cb._product_mi_blocked(cp, X, specs, y, nbins, "auto")
    monkeypatch.setattr(dss, "device_room", lambda cp_: 4 * 3 * 3 * 3000 * 8)
    blocked = cb._product_mi_blocked(cp, X, specs, y, nbins, "auto")
    assert whole.shape == blocked.shape == (len(specs),)
    np.testing.assert_array_equal(whole, blocked)
