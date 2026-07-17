"""Regression: the device-born extra-basis matrix equals the host generate_extra_basis_features matrix.

SF1c device-borns the WHOLE extra-basis engineered matrix (spline / Fourier / chirp / wavelet) on the GPU from
the resident raw operands + the per-column fit ``meta``, so the host matrix never uploads at
``_orth_mi_backends.py:311``. Correctness bar: each device column reproduces the host formula verbatim (same
lo/span/mean/std/freq/power/knots from meta, same axis + clip). This test builds a frame whose target excites all
four detectors, generates the host matrix + meta, rebuilds it on the device via ``_build_extra_basis_matrix_gpu``,
and asserts every column matches to ~1e-5 -- the guarantee that the device-born path never silently changes an
engineered column (which would drift the binned-MI partition and thus selection). It also asserts all four bases
are actually exercised.

Tolerance 1e-5 (2026-07-13), not ~1e-8: under the default ``MLFRAME_CRIT_DTYPE_RELAXED=1`` both host and device
operate on the raw operand at f32, so the SAME residual arithmetic-precision gap documented in
``_gpu_resident_cross_basis.py``'s ``build_leg_product_matrix_gpu`` docstring applies here too (measured worst
case on this fixture: 3.66e-6). This test ALSO caught a genuine, larger, pre-existing bug (fixed alongside this
tolerance change): ``_bspline_col_gpu`` was missing the unconditional float64 upcast its host twin
``_bspline_basis_values`` has before the boundary-safety clip -- float32 can't represent the clip's 1e-12 margin
near a repeated boundary knot (epsilon below float32's ~1.19e-7 resolution at 1.0), so the clip silently no-opped
and the last B-spline basis function's degenerate-knot recursion collapsed to 0 instead of ~1.0 (maxerr was
exactly 1.0 pre-fix, an obvious formula bug, not a precision one)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def test_device_born_extra_basis_matches_host_all_bases():
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import (
        generate_extra_basis_features,
    )
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._extra_basis_resident import (
        _build_extra_basis_matrix_gpu,
    )

    rng = np.random.default_rng(0)
    n = 6000
    a = rng.uniform(-3, 3, n)
    b = rng.uniform(0.1, 5, n)
    d = rng.uniform(0, 2 * np.pi, n)
    X = pd.DataFrame({"a": a, "b": b, "d": d})
    # periodic (d) + chirp (a**2) + local threshold (b) -> excites fourier + chirp + wavelet + spline.
    y = np.sin(d) + np.sin(2.0 * ((a - a.mean()) / a.std()) ** 2) + (b > 2.5).astype(float) + 0.05 * rng.standard_normal(n)

    eng, meta = generate_extra_basis_features(
        X,
        cols=["a", "b", "d"],
        extra_bases=("spline", "fourier", "wavelet"),
        y=y,
        fourier_adaptive=True,
        fourier_chirp=True,
    )
    assert eng.shape[1] > 0, "no extra-basis columns emitted"
    bases = {m["basis"] + ("/q" if m.get("arg") == "quadratic" else "") for m in meta.values()}
    # all four device-ported families must be exercised so the parity actually covers each builder.
    assert {"spline", "fourier", "fourier/q", "wavelet"} <= bases, f"missing bases: {bases}"

    names = list(eng.columns)
    dev = cp.asnumpy(_build_extra_basis_matrix_gpu(cp, X, names, meta))
    host = eng.to_numpy(dtype=np.float64)
    assert dev.shape == host.shape
    # per-column max abs error -- relaxed-dtype arithmetic-precision gap (see module docstring); a
    # formula bug is orders of magnitude larger (this test caught one at maxerr=1.0, see docstring).
    maxerr = float(np.max(np.abs(dev - host)))
    assert maxerr < 1e-5, f"device extra-basis diverges from host by {maxerr:.3e}"
