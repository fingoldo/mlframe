"""The one-launch Fourier / chirp block equals the per-column device build bit for bit, for both element types and every axis variant."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def _metas(rng, k):
    """Meta dicts as the Fourier fitter emits them: sin and cos, powers 1 to 3, linear and chirp axes, spans that include degenerate values."""
    out = []
    for j in range(k):
        quad = j % 4 == 3
        m = {"basis": "fourier", "kind": "sin" if j % 2 == 0 else "cos", "freq": float(rng.integers(1, 9)) + float(rng.random()), "lo": float(rng.standard_normal()), "span": float(abs(rng.standard_normal()) + 0.5)}
        if quad:
            m.update(arg="quadratic", mean=float(rng.standard_normal()), std=float(abs(rng.standard_normal()) + 0.5))
        else:
            m["power"] = int(1 + j % 3)
        out.append(m)
    out[0]["span"] = 0.0  # floored at 1e-12
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_block_is_bitwise_the_per_column_build(dtype):
    """Every row of the block equals ``_fourier_col_gpu`` of the same column and meta to the last bit, with several names sharing a source column."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._extra_basis_fourier_cuda import fourier_block_cuda
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._extra_basis_resident import _fourier_col_gpu

    rng = np.random.default_rng(0)
    n, sources, k = 3000, 7, 40
    data = [cp.asarray(rng.standard_normal(n).astype(dtype) * (1 + s)) for s in range(sources)]
    columns = [data[int(rng.integers(0, sources))] for _ in range(k)]
    metas = _metas(rng, k)
    block = fourier_block_cuda(cp, columns, metas)
    assert block is not None and block.shape == (k, n) and block.dtype == np.dtype(dtype)
    got = cp.asnumpy(block)
    word = np.uint32 if dtype == np.float32 else np.uint64
    for j in range(k):
        ref = cp.asnumpy(_fourier_col_gpu(cp, columns[j], metas[j]))
        same = (got[j].view(word) == ref.view(word)) | (np.isnan(got[j]) & np.isnan(ref))
        assert same.all(), f"name {j} ({metas[j]}) differs at row {int(np.flatnonzero(~same)[0])}"


def test_unsupported_input_is_declined():
    """Integer columns, mixed dtypes and an empty list return ``None`` so the caller keeps the per-column build."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._extra_basis_fourier_cuda import fourier_block_cuda

    m = {"basis": "fourier", "kind": "sin", "freq": 2.0, "lo": 0.0, "span": 1.0, "power": 1}
    assert fourier_block_cuda(cp, [], []) is None
    assert fourier_block_cuda(cp, [cp.arange(10)], [m]) is None
    assert fourier_block_cuda(cp, [cp.zeros(10, dtype=cp.float32), cp.zeros(10, dtype=cp.float64)], [m, m]) is None
    assert fourier_block_cuda(cp, [cp.zeros(10, dtype=cp.float32), cp.zeros(11, dtype=cp.float32)], [m, m]) is None
