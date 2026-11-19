"""The forward-recurrence block kernel ``clenshaw_basis_fwd`` gives the values of the per-degree Clenshaw kernel ``clenshaw_basis`` (kept as the reference and the rollback, ``MLFRAME_CLENSHAW_FWD=0``)."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_basis as basis_mod


@pytest.mark.parametrize("basis", sorted(basis_mod._BASIS_ID))
@pytest.mark.parametrize("degrees", [[1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 5, 9], [1], [0, 3]])
def test_fwd_block_equals_clenshaw_block(monkeypatch, basis, degrees):
    """Every basis and degree set: the forward block equals the Clenshaw block to float32 rounding."""
    Z = cp.asarray(np.random.default_rng(3).standard_normal((2000, 7)).astype(np.float32))
    monkeypatch.setenv("MLFRAME_CLENSHAW_FWD", "0")
    ref = basis_mod._clenshaw_basis_block_gpu(cp, Z, basis, degrees)
    monkeypatch.setenv("MLFRAME_CLENSHAW_FWD", "1")
    new = basis_mod._clenshaw_basis_block_gpu(cp, Z, basis, degrees)
    assert new.shape == ref.shape
    np.testing.assert_allclose(cp.asnumpy(new), cp.asnumpy(ref), rtol=1e-6, atol=1e-6)


def test_unsorted_degrees_use_the_reference_kernel(monkeypatch):
    """Descending degrees are not served by the forward kernel: the output is the reference kernel's, bit for bit."""
    Z = cp.asarray(np.random.default_rng(4).standard_normal((500, 3)).astype(np.float32))
    monkeypatch.setenv("MLFRAME_CLENSHAW_FWD", "1")
    a = basis_mod._clenshaw_basis_block_gpu(cp, Z, "chebyshev", [5, 2])
    monkeypatch.setenv("MLFRAME_CLENSHAW_FWD", "0")
    b = basis_mod._clenshaw_basis_block_gpu(cp, Z, "chebyshev", [5, 2])
    assert bool(cp.all(a == b))
