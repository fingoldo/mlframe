"""The CUDA 2-D discretisation gives the CPU bins for a Fortran-ordered input.

The wide-frame kernel indexes the array as row-major. A Fortran-ordered array (pandas ``to_numpy`` often returns one) was read from the wrong
cells, so the GPU bins disagreed with the CPU bins for every column.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - no usable CUDA runtime
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def test_fortran_ordered_wide_input_bins_like_the_c_ordered_one():
    """With 1000 columns (the GPU kernel's width) a Fortran-ordered input gives the same bins as the C-ordered copy."""
    from mlframe.feature_selection.filters import discretization as D

    rng = np.random.default_rng(0)
    x = rng.random((2000, 1000))
    cpu = np.asarray(D.discretize_2d_array(np.ascontiguousarray(x), n_bins=10, method="quantile"))
    fortran = np.asarray(D.discretize_2d_array(np.asfortranarray(x), n_bins=10, method="quantile"))
    np.testing.assert_array_equal(fortran, cpu)
