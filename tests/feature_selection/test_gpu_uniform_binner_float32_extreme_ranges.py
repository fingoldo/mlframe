"""The GPU uniform binner bins float32 columns with overflowing or subnormal ranges like the CPU kernel."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization import discretize_2d_array
from mlframe.feature_selection.filters.discretization._discretization_cuda import discretize_2d_array_cuda


def _gpu():
    """Skip when no CUDA device is usable."""
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)
    except Exception as exc:  # no device or driver
        pytest.skip(f"no CUDA device: {exc}")


def _arr():
    """Three float32 columns: a range beyond float32's maximum, a subnormal one and an ordinary one."""
    return np.column_stack(
        [
            np.array([-3e38, 0.0, 3e38, 3e38, -3e38, 1e38], dtype=np.float32),
            np.array([0.0, 1e-45, 2e-45, 3e-45, 0.0, 1e-45], dtype=np.float32),
            np.linspace(0, 1, 6, dtype=np.float32),
        ]
    ).astype(np.float32)


def test_gpu_matches_cpu_on_extreme_float32_ranges():
    """Every row keeps its CPU bin instead of collapsing to 0."""
    _gpu()
    arr = _arr()
    cpu = discretize_2d_array(arr, n_bins=4, method="uniform")
    gpu = discretize_2d_array_cuda(arr, n_bins=4, method="uniform")
    np.testing.assert_array_equal(gpu, cpu)


def test_row_chunked_gpu_matches_cpu_on_extreme_float32_ranges():
    """The row-chunked uniform path bins the same columns like the CPU kernel."""
    _gpu()
    from mlframe.feature_selection.filters.discretization._discretization_cuda import discretize_2d_array_cuda_row_chunked

    arr = _arr()
    cpu = discretize_2d_array(arr, n_bins=4, method="uniform")
    gpu = discretize_2d_array_cuda_row_chunked(arr, n_bins=4, method="uniform", free_bytes=256)
    np.testing.assert_array_equal(gpu, cpu)
