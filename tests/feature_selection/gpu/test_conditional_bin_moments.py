"""The device per-bin moments of the conditional residual / dispersion families equal the numba loops, are reproducible, and the dispersion generator is unchanged by the batched path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.gpu._gpu_support import has_device

from mlframe.feature_selection.filters import _conditional_bin_moments as cbm


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")


def _inputs(p: int = 7, n: int = 20_000):
    """Codes over 10 bins and columns of very different scales with NaN and inf rows and an empty bin."""
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 9, n)  # bin 9 stays empty
    cols = [rng.standard_normal(n) * (10.0**k) + codes for k in range(p)]
    cols[1][::13] = np.nan
    cols[2][::17] = np.inf
    cols[3][:] = 5.0
    return codes, cols


def test_device_moments_equal_host():
    """Count, mean and centred sum of squares agree to 1e-12 relative on columns spanning six decades."""
    _needs_gpu()
    codes, cols = _inputs()
    for h, d in zip(cbm.bin_moments_host(codes, cols, 10), cbm.bin_moments_gpu(codes, cols, 10)):
        np.testing.assert_allclose(d, h, rtol=1e-12, atol=1e-12 * (1 + np.abs(h).max()))


def test_device_moments_reproducible_and_resident_input():
    """Two runs give identical bits, and a device matrix input gives the same result as host columns."""
    _needs_gpu()
    import cupy as cp

    codes, cols = _inputs()
    a = cbm.bin_moments_gpu(codes, cols, 10)
    b = cbm.bin_moments_gpu(codes, cols, 10)
    c = cbm.bin_moments_gpu(codes, cp.asarray(np.stack(cols)), 10)
    assert len(a) == len(b) == len(c) > 0, "the moments must come back"
    for x, y, z in zip(a, b, c):
        assert np.array_equal(x, y) and np.array_equal(x, z)


def test_dispersion_generator_matches_the_per_pair_host_path(monkeypatch):
    """Forcing the batched device moments gives the columns of the per-pair host statistic to 1e-9."""
    _needs_gpu()
    from mlframe.feature_selection.filters import _extra_fe_families_dispersion as disp

    rng = np.random.default_rng(5)
    X = pd.DataFrame({f"x{i}": rng.standard_normal(6000) * (1 + i) for i in range(4)})
    X.iloc[::9, 1] = np.nan
    X["x2"] = X["x2"] * (1 + np.abs(X["x0"]))
    monkeypatch.setattr(cbm, "moments_device_chosen", lambda n, p: False)
    host, _ = disp.generate_conditional_dispersion_features(X, list(X.columns))
    monkeypatch.setattr(cbm, "moments_device_chosen", lambda n, p: True)
    dev, _ = disp.generate_conditional_dispersion_features(X, list(X.columns))
    assert list(host.columns) == list(dev.columns)
    np.testing.assert_allclose(dev.to_numpy(), host.to_numpy(), rtol=1e-9, atol=1e-9)
