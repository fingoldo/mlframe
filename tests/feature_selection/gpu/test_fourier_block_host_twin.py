"""The numba Fourier block twin equals the numpy chain and the device kernel; the fused sincos pass equals numpy for double axes and defers to numpy for single axes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.feature_selection.gpu._gpu_support import has_device

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _extra_basis_fourier_cuda as cuda_mod
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _extra_basis_resident as resident
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_block_host as host


def _metas(k: int, n_src: int) -> list:
    """Linear (powers 1 and 2) and chirp names, sine and cosine legs, spread over the sources."""
    metas = []
    for j in range(k):
        m = {"kind": "sin" if j % 2 == 0 else "cos", "freq": 0.25 * (1 + j % 12), "lo": -3.0, "span": 6.0, "src": j % n_src}
        if j % 4 == 3:
            m.update({"arg": "quadratic", "mean": 0.01, "std": 1.02, "lo": -4.0, "span": 8.0})
        else:
            m["power"] = 1 + (j % 3 == 1)
        metas.append(m)
    return metas


def _args(xs: np.ndarray, metas: list) -> tuple:
    """The parameter vectors of the block, formed as the device launcher forms them."""
    params = np.array([cuda_mod._axis_params(m) for m in metas], dtype=np.float64)
    src = np.array([m["src"] for m in metas])
    cosine = np.array([1 if m["kind"] != "sin" else 0 for m in metas])
    w = np.array([(2.0 * math.pi) * float(m["freq"]) for m in metas])
    return (xs, src, cosine, params[:, 4].astype(np.int64), params[:, 5].astype(np.int64), params[:, 0], params[:, 1], params[:, 2], params[:, 3], w)


def _needs_gpu():
    """cupy with a usable CUDA device, else skip."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    return cp


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("dt,tol", [(np.float32, 1e-5), (np.float64, 1e-12)])
def test_block_equals_the_numpy_chain(dt, tol, seed):
    """Every name of the block equals the per-column numpy expression of ``_fourier_col_gpu`` run on numpy arrays."""
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((5, 4001)).astype(dt)
    metas = _metas(24, 5)
    got = host.fourier_block_host(*_args(xs, metas))
    want = np.stack([resident._fourier_col_gpu(np, xs[m["src"]], m) for m in metas]).astype(dt)
    assert got.dtype == dt
    np.testing.assert_allclose(got, want, rtol=0, atol=tol)


@pytest.mark.parametrize("seed", [2, 3])
@pytest.mark.parametrize("dt,tol", [(np.float32, 2e-5), (np.float64, 1e-11)])
def test_block_equals_the_device_kernel(dt, tol, seed):
    """The host twin and the device kernel agree to the rounding of the element type."""
    cp = _needs_gpu()
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((6, 5003)).astype(dt)
    metas = _metas(32, 6)
    dev = cuda_mod.fourier_block_cuda(cp, [cp.asarray(xs[m["src"]]) for m in metas], metas)
    assert dev is not None
    assert np.allclose(host.fourier_block_host(*_args(xs, metas)), cp.asnumpy(dev), rtol=0, atol=tol)


def test_sincos_legs_double_axis_is_exact_and_single_axis_defers_to_numpy():
    """A double-precision axis over the row threshold gives numpy's values to the bit; a single-precision axis returns numpy's own arrays."""
    rng = np.random.default_rng(3)
    z = rng.random(30_000)
    s, c = host.sincos_legs(z, 1.75)
    ang = 2.0 * np.pi * 1.75 * z
    assert np.array_equal(s, np.sin(ang)) and np.array_equal(c, np.cos(ang))
    z32 = z.astype(np.float32)
    s32, c32 = host.sincos_legs(z32, 1.75)
    ang32 = 2.0 * np.pi * 1.75 * z32
    assert np.array_equal(s32, np.sin(ang32)) and np.array_equal(c32, np.cos(ang32))
    small = z[:100]
    s2, c2 = host.sincos_legs(small, 1.75)
    assert np.array_equal(s2, np.sin(2.0 * np.pi * 1.75 * small)) and np.array_equal(c2, np.cos(2.0 * np.pi * 1.75 * small))


def test_name_blocks_give_the_same_block(monkeypatch):
    """Forcing one name per block (each stacking only its own source column) gives the single-launch block bit for bit."""
    cp = _needs_gpu()
    from mlframe.feature_selection.filters import _device_blocks as db

    rng = np.random.default_rng(4)
    xs = rng.standard_normal((5, 3001)).astype(np.float32)
    metas = _metas(20, 5)
    cols = [cp.asarray(xs[m["src"]]) for m in metas]
    whole = cp.asnumpy(cuda_mod.fourier_block_cuda(cp, cols, metas))
    monkeypatch.setattr(db, "first_block", lambda *a, **k: 1)
    one = cp.asnumpy(cuda_mod.fourier_block_cuda(cp, cols, metas))
    assert np.array_equal(whole, one)
