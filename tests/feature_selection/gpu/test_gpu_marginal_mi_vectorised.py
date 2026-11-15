"""``_gpu_marginal_mi`` counts every row in one pass and equals the njit Miller-Madow marginal MI row by row."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _usability_njit_pool as pool
from tests.feature_selection.gpu._gpu_support import has_device

pytestmark = pytest.mark.gpu


def _case(n=4000, m=37, ky=4, seed=0):
    """Dense code rows of unequal cardinality, some tied to the label, plus constant rows."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, ky, n)
    kx = rng.integers(1, 20, m)
    codes = np.stack([rng.integers(0, k, n) for k in kx])
    codes[:6] = (y[None, :] + rng.integers(0, 2, (6, n))) % ky
    kx[:6] = ky
    return codes.astype(np.int64), kx.astype(np.int64), y.astype(np.int64), ky


def test_matches_the_njit_marginal_mi_per_row(monkeypatch):
    """Every row equals ``_marginal_mi_njit`` to 1e-12, the dependent rows carry signal and the constant rows are exactly zero."""
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    codes, kx, y, ky = _case()
    n = codes.shape[1]
    p = np.bincount(y) / n
    h_y = float(-(p * np.log(p)).sum())
    monkeypatch.setattr(pool, "_cp", cp)
    got = cp.asnumpy(pool._gpu_marginal_mi(cp.asarray(codes), kx, cp.asarray(y), h_y, ky, n))
    want = np.array([pool._marginal_mi_njit(codes[r], int(kx[r]), y, h_y, ky, n) for r in range(codes.shape[0])])
    assert got.max() > 0.1
    assert np.abs(got - want).max() < 1e-12
    assert np.all(got[kx == 1] == 0.0)


def test_row_blocks_give_the_same_values(monkeypatch):
    """A cell budget that forces several row blocks changes nothing."""
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    codes, kx, y, ky = _case(m=40)
    n = codes.shape[1]
    p = np.bincount(y) / n
    h_y = float(-(p * np.log(p)).sum())
    monkeypatch.setattr(pool, "_cp", cp)
    whole = cp.asnumpy(pool._gpu_marginal_mi(cp.asarray(codes), kx, cp.asarray(y), h_y, ky, n))
    monkeypatch.setattr(pool, "_MARGINAL_MI_CELLS", int(kx.max()) * ky * 7)
    blocked = cp.asnumpy(pool._gpu_marginal_mi(cp.asarray(codes), kx, cp.asarray(y), h_y, ky, n))
    assert np.array_equal(whole, blocked)
