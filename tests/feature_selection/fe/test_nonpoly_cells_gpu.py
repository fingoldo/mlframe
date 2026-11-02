"""The device cells of fourier / rbf / sigmoid / pade equal the host cells, and the device router picks the basis the host router picks on data built for each of the eight bases."""

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe._nonpoly_cells import eval_basis_cell
from mlframe.feature_selection.filters.hermite_fe.shared import POLY_BASES

from ._basis_cases import DEGREES, _case
from tests._cuda_require import require_cuda_device


def _matrix(n: int) -> np.ndarray:
    """Columns of different shape: normal, uniform, exponential, lognormal and a large-scale one."""
    rng = np.random.default_rng(0)
    return np.column_stack([rng.normal(size=n), rng.uniform(0, 1, n), rng.exponential(size=n), np.exp(rng.normal(size=n)), rng.normal(size=n) * 1e3]).astype(np.float32)


@pytest.mark.parametrize("n", [129, 1009, 20000])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("basis", ["fourier", "rbf", "sigmoid", "pade"])
def test_device_cells_equal_host_cells(basis, dtype, n):
    """Every (column, degree) cell of the block equals the host cell; pade is compared away from the poles of its denominator, where float rounding of z is amplified."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters.hermite_fe._nonpoly_cells_gpu import nonpoly_block_gpu

    M = _matrix(n).astype(dtype)
    require_cuda_device()
    block, cells = nonpoly_block_gpu(cp, cp.asarray(M), basis, [1, 2, 3, 4, 9, 10])
    got = cp.asnumpy(block).astype(np.float64)
    tol = 5e-6 if dtype == np.float32 else 1e-9
    for j, (c, d) in enumerate(cells):
        z, params = POLY_BASES[basis]["fit"](M[:, c])
        want = eval_basis_cell(basis, z, d, params)
        mask = np.ones(n, dtype=bool)
        if basis == "pade":
            zz = np.asarray(z, dtype=np.float64)
            mask = np.abs(1.0 + zz**d) > 0.2
        np.testing.assert_allclose(got[mask, j], want[mask], atol=max(tol, 1e-5) if basis != "fourier" else tol, rtol=1e-4)


def test_degrees_without_a_cell_are_omitted():
    """rbf and sigmoid stop at nine centres; degree 10 yields no cell, as on the host."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters.hermite_fe._nonpoly_cells_gpu import nonpoly_block_gpu

    M = cp.asarray(_matrix(500))
    _block, cells = nonpoly_block_gpu(cp, M, "rbf", [9, 10])
    assert {d for _c, d in cells} == {9}


@pytest.mark.parametrize("basis", ["hermite", "legendre", "chebyshev", "laguerre", "fourier", "rbf", "sigmoid", "pade"])
def test_device_router_picks_the_crafted_basis(basis):
    """The batched device router reproduces the host choice on the column built for each basis."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_basis import _gpu_route_bases_batched
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import evaluable_bases

    x, y, _degree = _case(basis)
    require_cuda_device()
    chosen = _gpu_route_bases_batched(cp, cp.asarray(x.astype(np.float32))[:, None], cp.asarray(y.astype(np.float64)), list(evaluable_bases()), DEGREES, robust_axis=True)
    assert chosen == [basis]
