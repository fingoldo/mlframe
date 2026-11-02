"""The fused tau-grid kernel of the resident conditional gate equals the expression sequence it replaces, bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._resident_candidate_mi import _tau_grid_kernel


def _operands(n: int = 2500):
    """Operand columns with NaN, +-inf, zeros and ties, plus a threshold grid that hits tied values."""
    rng = np.random.default_rng(4)
    c = rng.standard_normal(n)
    c[::17] = np.nan
    c[5::97] = np.inf
    c[6::89] = -np.inf
    c[::23] = np.round(c[::23], 1)
    a = rng.standard_normal(n) * 1e3
    a[3::31] = np.inf
    a[4::41] = np.nan
    a[::13] = 0.0
    b = rng.standard_normal(n)
    taus = np.concatenate([np.quantile(c[np.isfinite(c)], np.linspace(0.05, 0.95, 9)), [c[np.isfinite(c)][0]]])
    return c, a, b, taus


@pytest.mark.parametrize("ctype,dtype", [("float", np.float32), ("double", np.float64)])
def test_mask_equals_compare_cast_multiply(ctype, dtype):
    """``(cv > tau) * av`` in double, then cast to the block dtype."""
    c, a, _b, taus = _operands()
    cg, ag, tg = cp.asarray(c), cp.asarray(a), cp.asarray(taus)
    want = ((cg[None, :] > tg[:, None]).astype(cp.float64) * ag[None, :]).astype(dtype)
    got = _tau_grid_kernel(cp, "mask", ctype)(cg[None, :], tg[:, None], ag[None, :])
    assert got.dtype == dtype
    np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint8), cp.asnumpy(want).view(np.uint8))


@pytest.mark.parametrize("ctype,dtype", [("float", np.float32), ("double", np.float64)])
def test_select_equals_where(ctype, dtype):
    """``where(cv > tau, av, bv)`` in double, then cast to the block dtype."""
    c, a, b, taus = _operands()
    cg, ag, bg, tg = cp.asarray(c), cp.asarray(a), cp.asarray(b), cp.asarray(taus)
    want = cp.where(cg[None, :] > tg[:, None], ag[None, :], bg[None, :]).astype(dtype)
    got = _tau_grid_kernel(cp, "select", ctype)(cg[None, :], tg[:, None], ag[None, :], bg[None, :])
    np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint8), cp.asnumpy(want).view(np.uint8))
