"""The fused ``disp_column`` kernel reproduces the cupy-op chain of the dispersion and residual builders exactly, for float32 and double operands, every emission kind, a floored sigma, NaN x_i and NaN x_j rows, and a narrower output dtype."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _extra_fe_families_dispersion_resident as dr


def _cp():
    """cupy with a usable device, else skip."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")
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
