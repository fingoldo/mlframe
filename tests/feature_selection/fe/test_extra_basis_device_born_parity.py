"""Regression: the device-born extra-basis matrix equals the host generate_extra_basis_features matrix.

SF1c device-borns the WHOLE extra-basis engineered matrix (spline / Fourier / chirp / wavelet) on the GPU from
the resident raw operands + the per-column fit ``meta``, so the host matrix never uploads at
``_orth_mi_backends.py:311``. Correctness bar: each device column reproduces the host formula verbatim (same
lo/span/mean/std/freq/power/knots from meta, same axis + clip), differing only in FP reduction order. This test
builds a frame whose target excites all four detectors, generates the host matrix + meta, rebuilds it on the
device via ``_build_extra_basis_matrix_gpu``, and asserts every column matches to ~1e-8 -- the guarantee that the
device-born path never silently changes an engineered column (which would drift the binned-MI partition and thus
selection). It also asserts all four bases are actually exercised."""
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
    y = (np.sin(d) + np.sin(2.0 * ((a - a.mean()) / a.std()) ** 2)
         + (b > 2.5).astype(float) + 0.05 * rng.standard_normal(n))

    eng, meta = generate_extra_basis_features(
        X, cols=["a", "b", "d"], extra_bases=("spline", "fourier", "wavelet"),
        y=y, fourier_adaptive=True, fourier_chirp=True,
    )
    assert eng.shape[1] > 0, "no extra-basis columns emitted"
    bases = {m["basis"] + ("/q" if m.get("arg") == "quadratic" else "") for m in meta.values()}
    # all four device-ported families must be exercised so the parity actually covers each builder.
    assert {"spline", "fourier", "fourier/q", "wavelet"} <= bases, f"missing bases: {bases}"

    names = list(eng.columns)
    dev = cp.asnumpy(_build_extra_basis_matrix_gpu(cp, X, names, meta))
    host = eng.to_numpy(dtype=np.float64)
    assert dev.shape == host.shape
    # per-column max abs error -- FP reduction order only; a formula bug would be orders of magnitude larger.
    maxerr = float(np.max(np.abs(dev - host)))
    assert maxerr < 1e-8, f"device extra-basis diverges from host by {maxerr:.3e}"
