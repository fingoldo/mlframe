"""The tiled column mean/std kernel equals numpy and the one-block-per-column reference, including widths that are not a multiple of the tile."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _gpu_resident_basis as gpu_basis

from . import _untiled_reference_kernels as ref_k

cupy = pytest.importorskip("cupy")

try:
    cupy.cuda.runtime.getDeviceCount()
    _HAVE_CUDA_DEVICE = True
except Exception:  # no driver / no device: the only thing this probe decides is whether the GPU tests can run
    _HAVE_CUDA_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAVE_CUDA_DEVICE, reason="no CUDA device")


@pytest.mark.parametrize("n,k", [(1000, 37), (7, 5), (4096, 64)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_tiled_matches_numpy_and_reference(n, k, dtype):
    """The tiled mean/std equals numpy and the reference kernel on widths that are not a multiple of the 32-column tile."""
    import cupy as cp

    rng = np.random.default_rng(n + k)
    host = (rng.standard_normal((n, k)) * 3.0 + 1.5).astype(dtype)
    M = cp.asarray(host)
    mean, std = gpu_basis._col_meanstd(cp, M)
    assert np.allclose(cp.asnumpy(mean), host.mean(axis=0), rtol=1e-5 if dtype == np.float32 else 1e-10)
    assert np.allclose(cp.asnumpy(std), host.std(axis=0), rtol=1e-5 if dtype == np.float32 else 1e-10)
    # the one-block-per-column reference kernel still agrees
    real_t = "float" if dtype == np.float32 else "double"
    ref = cp.RawKernel(ref_k._COL_MEANSTD_SRC.replace("{REAL}", real_t), "col_meanstd")
    rmean = cp.empty(k, dtype=M.dtype)
    rstd = cp.empty(k, dtype=M.dtype)
    ref((k,), (128,), (M, np.int64(n), np.int32(k), rmean, rstd))
    assert np.allclose(cp.asnumpy(mean), cp.asnumpy(rmean), rtol=1e-6 if dtype == np.float32 else 1e-12)
    assert np.allclose(cp.asnumpy(std), cp.asnumpy(rstd), rtol=1e-6 if dtype == np.float32 else 1e-12)


@pytest.mark.parametrize("n,k", [(1000, 37), (7, 5), (4096, 64)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_tiled_minmax_matches_numpy_and_reference(n, k, dtype):
    """The tiled column min/max equals numpy and the one-block-per-column reference, on widths that are not a multiple of the tile."""
    import cupy as cp

    rng = np.random.default_rng(n * 7 + k)
    host = (rng.standard_normal((n, k)) * 3.0 + 1.5).astype(dtype)
    M = cp.asarray(host)
    mn, mx = gpu_basis._col_minmax(cp, M)
    assert np.array_equal(cp.asnumpy(mn), host.min(axis=0))
    assert np.array_equal(cp.asnumpy(mx), host.max(axis=0))
    real_t = "float" if dtype == np.float32 else "double"
    ref = cp.RawKernel(ref_k._COL_MINMAX_SRC.replace("{REAL}", real_t), "col_minmax")
    rmn = cp.empty(k, dtype=M.dtype)
    rmx = cp.empty(k, dtype=M.dtype)
    ref((k,), (128,), (M, np.int64(n), np.int32(k), rmn, rmx))
    assert np.array_equal(cp.asnumpy(mn), cp.asnumpy(rmn))
    assert np.array_equal(cp.asnumpy(mx), cp.asnumpy(rmx))


@pytest.mark.parametrize("n,k", [(2000, 37), (50, 5), (4096, 64)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_tiled_heavy_tail_flags_equal_the_reference(n, k, dtype):
    """The tiled heavy-tail outlier flags equal the one-block-per-column reference on widths that are not a multiple of the tile."""
    import cupy as cp

    rng = np.random.default_rng(n + 3 * k)
    host = rng.standard_normal((n, k)).astype(dtype)
    host[: max(1, n // 50), ::3] += 200.0  # a clearly separated outlier group on every third column: flagged
    M = cp.asarray(host)
    med = cp.asarray(np.median(host, axis=0).astype(dtype))
    scale = cp.asarray((np.median(np.abs(host - np.median(host, axis=0)), axis=0) * 1.4826).astype(dtype))
    real_t = "float" if dtype == np.float32 else "double"
    args_tail = (np.float64(gpu_basis._GPU_ROBUST_AXIS_OUTER_K), np.float64(gpu_basis._GPU_ROBUST_AXIS_GAP), np.float64(gpu_basis._GPU_ROBUST_AXIS_MAX_FRAC), np.int64(n), np.int32(k))
    tiled = cp.empty(k, dtype=cp.int8)
    tiled_kernel = gpu_basis._get_heavy_tail_stats_tiled_kernel(cp, real_t)
    tiled_kernel(((k + 31) // 32,), (32, 8), (M, med, scale, *args_tail, tiled))
    ref = cp.empty(k, dtype=cp.int8)
    ref_kernel = ref_k.get_heavy_tail_stats_kernel(cp, real_t)
    ref_kernel((k,), (128,), (M, med, scale, *args_tail, ref))
    flags = cp.asnumpy(ref)
    assert flags.any() and not flags.all(), "the data must exercise both outcomes"
    assert np.array_equal(cp.asnumpy(tiled), flags)
