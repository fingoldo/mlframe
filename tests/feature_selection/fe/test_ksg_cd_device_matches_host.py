"""The device Ross k-NN MI gives the host kernel's value per column, with singleton classes, several class counts and ties in the labels."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _ksg_cd_fast as host
from mlframe.feature_selection.filters import _ksg_cd_gpu as gpu


def _block(n, n_classes, p, seed):
    """A scaled, noised block with class-dependent columns and one singleton class."""
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, p))
    d = r.integers(0, n_classes, n).astype(np.int64)
    d[:2] = n_classes  # a class with two members
    d[2] = n_classes + 1  # a singleton class: dropped by both kernels
    d = np.unique(d, return_inverse=True)[1].astype(np.int64)
    X[:, : p // 2] += d[:, None] * 0.4
    X = X / X.std(axis=0)
    X += 1e-10 * r.standard_normal(X.shape)
    return np.ascontiguousarray(X), d


@pytest.mark.parametrize("n,n_classes,p,k", [(2500, 2, 6, 3), (12000, 3, 9, 3), (8000, 5, 4, 5)])
def test_device_mi_equals_host(n, n_classes, p, k):
    """Per-column MI agrees to rounding."""
    X, d = _block(n, n_classes, p, 0)
    want = host._mi_cd_columns(X, d, int(d.max()) + 1, k)
    got = gpu.mi_cd_columns_device(X, d, int(d.max()) + 1, k)
    assert got is not None
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    assert want[: p // 2].min() > want[p // 2 :].max()  # the informative columns score above the noise ones


def test_public_entry_points_route_through_the_device_when_forced(monkeypatch):
    """``ksg_mi_discrete_target_columns`` with the device chosen equals the host path chosen off."""
    X, d = _block(6000, 3, 5, 1)
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: False)
    want = host.ksg_mi_discrete_target_columns(X, d, n_neighbors=3, random_state=0)
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: True)
    got = host.ksg_mi_discrete_target_columns(X, d, n_neighbors=3, random_state=0)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_device_mi_accepts_float32_input_without_reading_past_the_buffer():
    """A float32 block is uploaded as float64 for the double* kernel, so its MI matches the host run on the same values."""
    X, d = _block(8000, 3, 6, 1)
    X32 = X.astype(np.float32)
    want = host._mi_cd_columns(np.ascontiguousarray(X32.astype(np.float64)), d, int(d.max()) + 1, 3)
    got = gpu.mi_cd_columns_device(X32, d, int(d.max()) + 1, 3)
    assert got is not None
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9), (got, want)
