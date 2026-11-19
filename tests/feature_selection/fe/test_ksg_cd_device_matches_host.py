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


@pytest.fixture(autouse=True)
def _exact_double_path(monkeypatch):
    """The equality tests below compare to 1e-9, which is the float64 path; the float32 path has its own tests at the end of the file."""
    monkeypatch.setenv("MLFRAME_KSG_F32", "0")


def _tied_block(n, p, seed):
    """A noised block whose columns take few distinct values, as a discretised feature does (the 1e-10 noise is lost in float32)."""
    r = np.random.default_rng(seed)
    d = r.integers(0, 3, n).astype(np.int64)
    X = r.standard_normal((n, p))
    X[:, : p // 2] += d[:, None] * 0.4
    X = np.round(X * 8) / 8
    X = X / X.std(axis=0)
    X += 1e-10 * r.standard_normal(X.shape)
    return np.ascontiguousarray(X), d


def test_float32_path_agrees_with_host_on_tie_free_columns(monkeypatch):
    """Continuous columns run in float32 and give the host MI to the rounding of the data, with the same ordering of informative above noise."""
    monkeypatch.setenv("MLFRAME_KSG_F32", "1")
    X, d = _block(20000, 3, 8, 0)
    want = host._mi_cd_columns(X, d, int(d.max()) + 1, 3)
    got = gpu.mi_cd_columns_device(X, d, int(d.max()) + 1, 3)
    assert got is not None
    np.testing.assert_allclose(got, want, rtol=0, atol=5e-4)
    assert got[:4].min() > got[4:].max()


def test_float32_path_leaves_tied_columns_to_float64(monkeypatch):
    """Columns with equal neighbours after the float32 cast (heavy ties) are scored in float64, so they equal the host value exactly; in float32 the noise columns would score about 1.9."""
    monkeypatch.setenv("MLFRAME_KSG_F32", "1")
    X, d = _tied_block(20000, 6, 1)
    want = host._mi_cd_columns(X, d, int(d.max()) + 1, 3)
    got = gpu.mi_cd_columns_device(X, d, int(d.max()) + 1, 3)
    assert got is not None
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    assert want[3:].max() < 0.1


def test_packed_float32_order_sorts_every_row_like_argsort():
    """The one-sort packed order of a float32 matrix (negative, positive and zero values) sorts each row exactly as a per-row argsort does."""
    r = np.random.default_rng(3)
    x = cp.asarray(r.standard_normal((7, 513)).astype(np.float32))
    x[0, :5] = 0.0
    order = gpu._packed_f32_order(cp, x)
    assert order is not None
    assert sorted(order[3].get().tolist()) == list(range(513))
    got = cp.take_along_axis(x, order, axis=1)
    np.testing.assert_array_equal(got.get(), np.sort(x.get(), axis=1))
