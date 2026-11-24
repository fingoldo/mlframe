"""The logit-null kernels of the unary credit (``_unary_credit_logit_cuda``): the njit twin and the fused device kernel equal the numpy reference of ``_unary_credit_logit._deviance``, for both value types, over shapes
around the warp and the block, and the device path hands over to the host when it cannot run."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _unary_credit_logit as ucl
from mlframe.feature_selection.filters import _unary_credit_logit_cuda as kernels


def _problem(n: int, k: int, d: int, seed: int = 0, flip: float = 0.0):
    """``(offset, z rows with the zero row last, class bytes, numpy-reference LRT)``: standardised candidates, an offset and draws from a logit that uses the first candidate."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((k, n))
    z = (z - z.mean(axis=1, keepdims=True)) / np.where(z.std(axis=1, keepdims=True) > 0, z.std(axis=1, keepdims=True), 1.0)
    offset = 0.7 * rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-(offset + 0.4 * z[0])))
    y = (rng.random((d, n)) < p[None, :]).astype(np.uint8)
    if flip:
        y = ((z[0][None, :] > 0) ^ (rng.random((d, n)) < flip)).astype(np.uint8)  # near-separation by the first candidate
    steps = 5
    reference = ucl._deviance(offset, z.T.copy(), y.T.astype(np.float64), steps, np.float64)
    return offset, np.vstack([z, np.zeros((1, n))]), y, reference, steps


def _lrt(ll: np.ndarray) -> np.ndarray:
    """LRT of every candidate row against the last (zero) row."""
    return 2.0 * (ll[:-1] - ll[-1][None, :])


SHAPES = [(1, 1, 1), (31, 2, 3), (33, 2, 1), (64, 5, 7), (600, 13, 7), (3001, 4, 20)]


@pytest.mark.parametrize("n,k,d", SHAPES)
def test_the_njit_twin_equals_the_numpy_reference(n, k, d):
    """Same Newton arithmetic in double precision: the twin and the numpy passes agree to rounding, for one row up to a few thousand and from one draw to twenty."""
    offset, rows, y, reference, steps = _problem(n, k, d)
    ours = _lrt(kernels.loglik_host(offset, rows, y, steps))
    assert np.allclose(ours, reference, rtol=1e-9, atol=1e-9), float(np.abs(ours - reference).max())


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-8), (np.float32, 2e-3)])
@pytest.mark.parametrize("n,k,d", SHAPES)
def test_the_device_kernel_equals_the_numpy_reference_in_both_value_types(n, k, d, dtype, tol):
    """The fused kernel gives the reference LRT in double precision to 1e-8 and with float32 terms to 2e-3 (the LRT is of the order of 1 to 100), over shapes below, at and above the warp and block widths."""
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    offset, rows, y, reference, steps = _problem(n, k, d)
    ll = kernels.loglik_device(cp, cp.asarray(offset.astype(dtype)), cp.asarray(rows.astype(dtype)), cp.asarray(y), steps)
    assert ll.dtype == cp.float64 and ll.shape == (k + 1, d)
    ours = _lrt(cp.asnumpy(ll))
    assert np.all(np.isfinite(ours))
    assert np.allclose(ours, reference, rtol=tol, atol=tol), float(np.abs(ours - reference).max())


def test_near_separation_and_extreme_offsets_stay_finite_and_agree():
    """Classes almost separated by a candidate (3 percent flips) and an offset of up to 30 in absolute value: the saturated sigmoid and the guarded determinant give finite values equal on both kernels."""
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    offset, rows, y, _reference, steps = _problem(1500, 3, 5, seed=2, flip=0.03)
    offset = np.clip(offset * 40.0, -30.0, 30.0)
    host = _lrt(kernels.loglik_host(offset, rows, y, steps))
    device = _lrt(cp.asnumpy(kernels.loglik_device(cp, cp.asarray(offset), cp.asarray(rows), cp.asarray(y), steps)))
    assert np.all(np.isfinite(host)) and np.all(np.isfinite(device))
    assert np.allclose(host, device, rtol=1e-6, atol=1e-6), float(np.abs(host - device).max())


def test_a_constant_candidate_row_fits_the_intercept_only_model():
    """A zero candidate row is the offset-plus-intercept fit: its LRT against the zero row is zero on every draw."""
    offset, rows, y, _reference, steps = _problem(400, 2, 4)
    rows[0] = 0.0
    assert np.allclose(_lrt(kernels.loglik_host(offset, rows, y, steps))[0], 0.0, atol=1e-9)


def test_the_device_gives_way_to_the_host_when_it_cannot_run(monkeypatch):
    """A device that raises (out of memory, a driver error) returns ``None`` from ``try_loglik_device`` and the credit then finishes on the njit twin with the same verdict."""
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    offset, rows, y, _reference, steps = _problem(300, 2, 3)

    def boom(*_args, **_kwargs):
        """A device that cannot run the kernel."""
        raise RuntimeError("out of memory")

    monkeypatch.setattr(kernels, "loglik_device", boom)
    assert kernels.try_loglik_device(cp, cp.asarray(offset), cp.asarray(rows), cp.asarray(y), steps) is None


def test_the_memory_bound_refuses_a_working_set_larger_than_the_card(monkeypatch):
    """``fits_device`` compares the working set (candidate rows, offset, class bytes, result) with a quarter of the free memory."""
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    assert kernels.fits_device(cp, 10000, 14, 50, 4)
    free = int(cp.cuda.runtime.memGetInfo()[0])
    assert not kernels.fits_device(cp, free // 8, 14, 50, 4)  # about 14 rows x free/8 x 4 bytes: far beyond a quarter of the free memory
