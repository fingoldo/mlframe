"""The column-moment kernels (numba, CUDA) give the meta scorer's fingerprint statistics of the pandas / numpy reference."""

from __future__ import annotations

import numpy as np
from tests.feature_selection.gpu._gpu_support import require_device
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _meta_fingerprint_gpu as MF
from mlframe.feature_selection.filters import _orthogonal_meta_scorer_fe as MS


def _frame(n=6000, k=12):
    """Heavy-tailed columns, one correlated with the target, one constant."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_t(4, size=(n, k)), columns=[f"x{i}" for i in range(k)])
    X["const"] = 3.0
    y = (X["x0"] + rng.normal(size=n) > 0).astype(float).to_numpy()
    return X, y


def _reference_moments(xs, yc):
    """The five sums with numpy: mean, m2, m3, m4 about the mean and the cross sum."""
    c = xs - xs.mean(axis=1, keepdims=True)
    return np.stack([xs.mean(axis=1), (c**2).mean(axis=1), (c**3).mean(axis=1), (c**4).mean(axis=1), c @ yc], axis=1)


def test_the_numba_kernel_gives_the_numpy_moments():
    """All five sums agree with numpy to 1e-9 relative."""
    X, y = _frame()
    xs = np.ascontiguousarray(X.to_numpy().T)
    yc = y - y.mean()
    np.testing.assert_allclose(MF._moments_host(xs, yc), _reference_moments(xs, yc), rtol=1e-9, atol=1e-10)


def test_the_cuda_kernel_gives_the_numpy_moments():
    """The CUDA kernel agrees with numpy on the same block."""
    require_device()
    X, y = _frame()
    xs = np.ascontiguousarray(X.to_numpy().T)
    yc = y - y.mean()
    got = MF._moments_device(xs, yc)
    assert got is not None, "the device path returned nothing on a host with a usable device"
    np.testing.assert_allclose(got, _reference_moments(xs, yc), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("forced", ["0", "1"])
def test_the_fingerprint_statistics_equal_the_pandas_reference(monkeypatch, forced):
    """``mean_abs_skew``, ``mean_kurtosis`` and ``mean_abs_pearson`` with the kernels equal the per-column pandas loops (``_VECTORISED = False``), on either backend."""
    X, y = _frame()
    cols = list(X.columns)
    monkeypatch.setenv("MLFRAME_META_FINGERPRINT_DEVICE", forced)
    got = (MS._mean_abs_skew_and_kurtosis(X), MS._mean_abs_pearson(X, cols, y))
    monkeypatch.setattr(MS, "_VECTORISED", False)
    want = (MS._mean_abs_skew_and_kurtosis(X), MS._mean_abs_pearson(X, cols, y))
    np.testing.assert_allclose([*got[0], got[1]], [*want[0], want[1]], rtol=1e-9)


def test_a_narrow_block_budget_gives_the_same_moments(monkeypatch):
    """With a block of three columns the moments equal the one-block ones."""
    X, y = _frame(n=2000, k=8)
    xs = np.ascontiguousarray(X.to_numpy().T)
    yc = y - y.mean()
    monkeypatch.setenv("MLFRAME_META_FINGERPRINT_DEVICE", "0")
    whole = MF.column_moments(xs, yc)
    monkeypatch.setattr(MF, "_BLOCK_BYTES", 3 * 8 * xs.shape[1])
    assert MF.column_block_width(xs.shape[1]) == 3
    assert np.array_equal(MF.column_moments(xs, yc), whole)
