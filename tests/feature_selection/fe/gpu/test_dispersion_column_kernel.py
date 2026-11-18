"""The fused ``disp_columns`` kernel reproduces the cupy-op chain of the dispersion and residual builders exactly, for float32 and double operands, every emission kind, a floored sigma, NaN x_i and NaN x_j rows, and a narrower output dtype."""

from __future__ import annotations

import numpy as np
import pytest
from tests._cuda_require import require_cuda_device

from mlframe.feature_selection.filters import _extra_fe_families_dispersion_resident as dr


def _cp():
    """cupy with a usable device, else skip."""
    cp = pytest.importorskip("cupy")
    require_cuda_device()
    return cp


@pytest.mark.parametrize("real", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["z", "absz", "z2"])
def test_dispersion_modes_equal_the_cupy_chain(real, kind):
    """z = (x - mu[code]) / s with s < floor -> 1 and NaN x -> 0, then the fold, equal to the chain value for value (and the output cast)."""
    cp = _cp()
    rng = np.random.default_rng(0)
    n, nb = 5003, 10
    x = rng.standard_normal(n).astype(real)
    x[::97] = np.nan
    mean = rng.standard_normal(nb).astype(real)
    std = np.abs(rng.standard_normal(nb)).astype(real)
    std[3] = 0.0
    std[5] = 5e-10
    codes = rng.integers(0, nb, n).astype(np.int64)
    xg, mg, sg, cg = cp.asarray(x), cp.asarray(mean), cp.asarray(std), cp.asarray(codes)
    out = cp.empty(n, dtype=real)
    dr._fused_column(cp, out, xg, cg, mg, sg, dr._KIND_MODE[kind])
    mu, s = mg[cg], sg[cg]
    s = cp.where(s < dr._DISPERSION_SIGMA_FLOOR, 1.0, s)
    want = dr._emit_kind_gpu(cp, cp.where(cp.isfinite(xg), (xg - mu) / s, 0.0), kind)
    np.testing.assert_array_equal(cp.asnumpy(out), cp.asnumpy(want))
    out32 = cp.empty(n, dtype=np.float32)
    dr._fused_column(cp, out32, xg, cg, mg, sg, dr._KIND_MODE[kind])
    np.testing.assert_array_equal(cp.asnumpy(out32), cp.asnumpy(want).astype(np.float32))


@pytest.mark.parametrize("real", [np.float32, np.float64])
def test_residual_mode_equals_the_cupy_chain(real):
    """|x - mu[code]| with NaN x -> 0 equals the chain."""
    cp = _cp()
    rng = np.random.default_rng(1)
    n, nb = 4001, 8
    x = rng.standard_normal(n).astype(real)
    x[::50] = np.nan
    mean = rng.standard_normal(nb).astype(real)
    codes = rng.integers(0, nb, n).astype(np.int64)
    xg, mg, cg = cp.asarray(x), cp.asarray(mean), cp.asarray(codes)
    out = cp.empty(n, dtype=real)
    dr._fused_column(cp, out, xg, cg, mg, mg, 3)
    want = cp.where(cp.isfinite(xg), cp.abs(xg - mg[cg]), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(out), cp.asnumpy(want))


@pytest.mark.parametrize("real", [np.float32, np.float64])
def test_bin_codes_kernel_equals_the_searchsorted_reference(real):
    """``disp_bin_codes`` (one launch for several x_j with different edge counts) equals the per-column searchsorted / clip / NaN-to-bin-0 reference: tied edges, values on an edge, below and above all edges, NaN and inf."""
    cp = _cp()
    rng = np.random.default_rng(3)
    n = 7001
    cols, edge_rows = [], []
    for nb in (3, 10, 25):
        x = rng.standard_normal(n).astype(real)
        x[::53] = np.nan
        x[1::97] = np.inf
        edges = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, nb + 1)).astype(real)
        edges[2] = edges[1]  # a tied edge
        x[5], x[6] = edges[3], edges[-1]  # values exactly on an edge
        cols.append(x)
        edge_rows.append(edges)
    got = cp.asnumpy(dr._bin_codes_block(cp, [cp.asarray(c) for c in cols], edge_rows))
    for k, (x, e) in enumerate(zip(cols, edge_rows)):
        # the host digitiser compares the value rounded to the criterion dtype with the stored float64 edges; the kernel and its cupy reference do the same
        want = cp.asnumpy(dr._codes_from_edges_gpu(cp, cp.asarray(x), cp.asarray(e, dtype=cp.float64)))
        np.testing.assert_array_equal(got[k], want)
