"""A resident code matrix whose joint histogram exceeds shared memory still scores on the device and equals the host plug-in MI.

The wide-histogram branch used to call ``np.ascontiguousarray`` on the cupy matrix and raise, which the FE stages swallowed as "continuing without ... columns".
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _fe_batched_mi as M


def _host_mi(codes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plug-in MI (nats) per column from the joint counts."""
    n = len(y)
    out = []
    for j in range(codes.shape[1]):
        joint = np.zeros((int(codes[:, j].max()) + 1, int(y.max()) + 1))
        np.add.at(joint, (codes[:, j], y), 1)
        p = joint / n
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        out.append(float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum()))
    return np.maximum(np.array(out), 0.0)


@pytest.mark.parametrize("kx", [40, 700])
def test_wide_histogram_resident_codes_match_host(kx, monkeypatch):
    """A narrow histogram (shared-memory kernel) and one over the cap (the batched-bincount branch), passed as device arrays."""
    from mlframe.feature_selection.filters import _wavelet_basis_fe_batched as W

    wide_calls = []
    real = W.batched_binned_mi_gpu
    monkeypatch.setattr(W, "batched_binned_mi_gpu", lambda *a, **k: (wide_calls.append(1), real(*a, **k))[1])
    rng = np.random.default_rng(0)
    n, k, ky = 4000, 6, 20
    codes = rng.integers(0, kx, (n, k))
    y = rng.integers(0, ky, n)
    got = M.binned_mi_from_codes_gpu(cp.asarray(codes), y, kx_per_col=[kx] * k, ky=ky, codes_trusted=True)
    np.testing.assert_allclose(got, _host_mi(codes, y), rtol=1e-9, atol=1e-12)
    assert bool(wide_calls) == (kx * ky * 4 > M._MI_FROM_CODES_MAX_SHARED)
