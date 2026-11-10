"""The batched CMI on a uint8 candidate matrix equals the int64 matrix's result exactly (counts are integers), on every histogram path."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu


@pytest.mark.parametrize("kx,ky,kz", [(10, 2, 10), (6, 2, 3), (10, 3, 40), (12, 2, 1)])
def test_uint8_equals_int64_conditional(kx, ky, kz):
    """Conditional CMI: small histograms (128 cells and below), the byte kernel's 129-256 cell band and the global-atomics band (above 256 cells)."""
    r = np.random.default_rng(kx * 100 + kz)
    n, k = 4001, 37
    x = r.integers(0, kx, size=(n, k))
    y = cp.asarray(r.integers(0, ky, size=n).astype(np.int64))
    z = cp.asarray(r.integers(0, kz, size=n).astype(np.int64))
    wide = batched_cmi_gpu(cp.asarray(x.astype(np.int64)), y, z, kx=kx, ky=ky, return_device=True, codes_trusted=True)
    byte = batched_cmi_gpu(cp.asarray(x.astype(np.uint8)), y, z, kx=kx, ky=ky, return_device=True, codes_trusted=True)
    np.testing.assert_array_equal(cp.asnumpy(byte), cp.asnumpy(wide))


def test_uint8_equals_int64_marginal():
    """Marginal MI (no conditioning set) takes the byte one-way histogram."""
    r = np.random.default_rng(1)
    n, k = 3000, 21
    x = r.integers(0, 9, size=(n, k))
    y = cp.asarray(r.integers(0, 2, size=n).astype(np.int64))
    wide = batched_cmi_gpu(cp.asarray(x.astype(np.int64)), y, None, kx=9, ky=2, return_device=True, codes_trusted=True)
    byte = batched_cmi_gpu(cp.asarray(x.astype(np.uint8)), y, None, kx=9, ky=2, return_device=True, codes_trusted=True)
    np.testing.assert_array_equal(cp.asnumpy(byte), cp.asnumpy(wide))
