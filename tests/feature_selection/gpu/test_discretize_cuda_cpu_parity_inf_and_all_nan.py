"""GPU quantile discretization matches the CPU path on a column with +inf and on an all-NaN column (tracker 2026-11-02 wave 12, PAR12D-3, PAR12D-4)."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters.discretization import _discretization_cuda as dc
from mlframe.feature_selection.filters.discretization import discretize_2d_array

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="needs a CUDA device")


def _frames():
    """A short column with +inf, an all-NaN column and an ordinary one."""
    rng = np.random.default_rng(0)
    arr = np.empty((50, 3), dtype=np.float64)
    arr[:, 0] = rng.normal(size=50)
    arr[:4, 0] = [0.0, 1.0, np.inf, 2.0]
    arr[:, 1] = np.nan
    arr[:, 2] = rng.normal(size=50)
    return arr


def test_gpu_codes_equal_cpu_codes_for_inf_and_all_nan_columns():
    """Before: the inf column collapsed to one bin on the GPU, the all-NaN column took the top code instead of 0."""
    arr = _frames()
    gpu = dc.discretize_2d_array_cuda(arr, 4, "quantile")
    cpu = discretize_2d_array(arr, 4, "quantile", prefer_gpu=False)
    np.testing.assert_array_equal(np.asarray(gpu), np.asarray(cpu))
