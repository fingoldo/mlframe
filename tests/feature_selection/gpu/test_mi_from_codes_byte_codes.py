"""``binned_mi_from_codes_gpu`` on a uint8 code matrix equals the int64 route exactly (same integer histograms), the byte matrix reaches the tiled kernel, and an oversized histogram still widens and works."""

from __future__ import annotations

import numpy as np
import pytest

from tests.feature_selection.gpu._gpu_support import has_device

from mlframe.feature_selection.filters import _fe_batched_mi as bm


def _cp():
    """cupy with a usable device, else skip."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    return cp


@pytest.mark.parametrize("n,K,kx,ky", [(1000, 5, 7, 3), (30_001, 19, 20, 2), (5000, 3, 200, 5)])
def test_uint8_equals_int64(n, K, kx, ky):
    """Identical MI per column for the byte and int64 matrices, also for a ragged column count."""
    cp = _cp()
    rng = np.random.default_rng(1)
    codes = rng.integers(0, kx, (n, K))
    y = rng.integers(0, ky, n)
    want = bm.binned_mi_from_codes_gpu(cp.asarray(codes.astype(np.int64)), cp.asarray(y.astype(np.int64)), kx_per_col=kx, ky=ky, codes_trusted=True)
    got = bm.binned_mi_from_codes_gpu(cp.asarray(codes.astype(np.uint8)), cp.asarray(y.astype(np.int64)), kx_per_col=kx, ky=ky, codes_trusted=True)
    np.testing.assert_array_equal(got, want)


def test_byte_matrix_reaches_the_byte_kernel(monkeypatch):
    """The tiled kernel is requested with byte_codes=True for a uint8 matrix and False for int64."""
    cp = _cp()
    seen = []
    real = bm._get_mi_from_codes_tile_kernel
    monkeypatch.setattr(bm, "_get_mi_from_codes_tile_kernel", lambda byte_codes=False: (seen.append(byte_codes), real(byte_codes))[1])
    rng = np.random.default_rng(2)
    c = rng.integers(0, 10, (2000, 4))
    y = cp.asarray(rng.integers(0, 2, 2000).astype(np.int64))
    bm.binned_mi_from_codes_gpu(cp.asarray(c.astype(np.uint8)), y, kx_per_col=10, ky=2, codes_trusted=True)
    bm.binned_mi_from_codes_gpu(cp.asarray(c.astype(np.int64)), y, kx_per_col=10, ky=2, codes_trusted=True)
    assert seen == [True, False]
