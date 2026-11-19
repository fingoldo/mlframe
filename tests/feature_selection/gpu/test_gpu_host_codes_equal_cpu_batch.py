"""The GPU binning of a candidate block returns the codes of the CPU batch discretiser, element for element (blocked pair recompute, 2026-11-18)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device: nothing to compare on
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._gpu_resident_fe import gpu_discretize_codes_host
from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch


@pytest.mark.parametrize("nbins, dtype", [(10, np.int8), (20, np.int16)])
def test_gpu_host_codes_equal_cpu_batch_with_ties_and_constants(nbins: int, dtype):
    """Rounded (tied), constant and few-valued columns bin to the same codes on the device and on the CPU."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30000, 60)).astype(np.float32)
    X[:, 5] = np.round(X[:, 5], 1)
    X[:, 7] = 2.0
    X[:, 9] = rng.integers(0, 3, 30000).astype(np.float32)
    gpu = np.asarray(gpu_discretize_codes_host(X, nbins, dtype=dtype))
    cpu = np.asarray(discretize_2d_quantile_batch(X, n_bins=nbins, dtype=dtype, parallel=True))
    assert gpu.dtype == cpu.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(gpu, cpu)
