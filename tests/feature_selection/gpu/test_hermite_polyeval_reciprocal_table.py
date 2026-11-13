"""The Legendre and Laguerre evaluation kernels take 1/k from a per-block table: the values are the per-step division's, bit for bit."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _hermite_fe_mi as H

pytestmark = pytest.mark.gpu


def _reference(basis, x, c):
    """The same recurrence on the host in float64, dividing at every step like the old kernel."""
    xi = x.astype(np.float64)
    if len(c) == 1:
        return np.full_like(xi, c[0])
    prev, cur = np.ones_like(xi), (xi if basis == "legendre" else 1.0 - xi)
    s = c[0] + c[1] * cur
    for k in range(2, len(c)):
        inv = 1.0 / k
        nxt = ((2 * k - 1) * xi * cur - (k - 1) * prev) * inv if basis == "legendre" else (((2 * k - 1) - xi) * cur - (k - 1) * prev) * inv
        s = s + c[k] * nxt
        prev, cur = cur, nxt
    return s


@pytest.mark.parametrize("basis", ["legendre", "laguerre"])
@pytest.mark.parametrize("nc", [2, 3, 6, 40, 70])
def test_kernel_matches_the_division_recurrence(basis, nc):
    """Degrees below and above the 64-entry table (the tail divides in the kernel) agree with the host recurrence to a few ulp."""
    rng = np.random.default_rng(nc)
    x = rng.uniform(-1, 1, 5000).astype(np.float32)
    c = rng.standard_normal(nc) * 0.3
    out = cp.empty(x.shape[0])
    H._polyeval_kernel(basis, "float")(((x.shape[0] + 255) // 256,), (256,), (cp.asarray(x), cp.asarray(c), np.int32(nc), np.int32(x.shape[0]), out))
    ref = _reference(basis, x, c)
    assert np.allclose(cp.asnumpy(out), ref, rtol=1e-9, atol=1e-9)
