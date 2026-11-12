"""The tiled column mean/std kernel equals numpy and the one-block-per-column reference, including widths that are not a multiple of the tile."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _gpu_resident_basis as gpu_basis


@pytest.mark.parametrize("n,k", [(1000, 37), (7, 5), (4096, 64)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_tiled_matches_numpy_and_reference(n, k, dtype):
    """The tiled mean/std equals numpy and the reference kernel on widths that are not a multiple of the 32-column tile."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(n + k)
    host = (rng.standard_normal((n, k)) * 3.0 + 1.5).astype(dtype)
    M = cp.asarray(host)
    mean, std = gpu_basis._col_meanstd(cp, M)
    assert np.allclose(cp.asnumpy(mean), host.mean(axis=0), rtol=1e-5 if dtype == np.float32 else 1e-10)
    assert np.allclose(cp.asnumpy(std), host.std(axis=0), rtol=1e-5 if dtype == np.float32 else 1e-10)
    # the one-block-per-column reference kernel still agrees
    real_t = "float" if dtype == np.float32 else "double"
    ref = cp.RawKernel(gpu_basis._COL_MEANSTD_SRC.replace("{REAL}", real_t), "col_meanstd")
    rmean = cp.empty(k, dtype=M.dtype)
    rstd = cp.empty(k, dtype=M.dtype)
    ref((k,), (128,), (M, np.int64(n), np.int32(k), rmean, rstd))
    assert np.allclose(cp.asnumpy(mean), cp.asnumpy(rmean), rtol=1e-6 if dtype == np.float32 else 1e-12)
    assert np.allclose(cp.asnumpy(std), cp.asnumpy(rstd), rtol=1e-6 if dtype == np.float32 else 1e-12)
