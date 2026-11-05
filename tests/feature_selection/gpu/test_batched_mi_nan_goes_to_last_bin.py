"""The in-kernel binning puts NaN in the last bin, as ``cp.searchsorted(side='right')`` does.

The binary search made every comparison false for a NaN value, so the value fell into bin 0 in the kernel while the reference codes put it
in the last bin. The mutual information from the two paths then differed.
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


def test_in_kernel_binning_matches_searchsorted_codes_with_nan():
    """MI from the in-kernel binning equals MI from the searchsorted codes when some values are NaN."""
    from mlframe.feature_selection.filters import _fe_batched_mi as fb

    rng = np.random.default_rng(0)
    n, K, nbins, ky = 2000, 2, 4, 2
    x = rng.normal(size=(n, K))
    x[::5, 0] = np.nan
    y = (rng.random(n) < 0.5).astype(np.int64)
    y[::5] = 1
    finite = x[np.isfinite(x[:, 0]), 0]
    edges = np.empty((nbins - 1, K))
    edges[:, 0] = np.quantile(finite, np.linspace(0, 1, nbins + 1)[1:-1])
    edges[:, 1] = np.quantile(x[:, 1], np.linspace(0, 1, nbins + 1)[1:-1])

    x_dev = cp.asarray(x)
    edges_dev = cp.asarray(edges)
    y_dev = cp.asarray(y)
    codes = cp.stack([cp.searchsorted(edges_dev[:, c], x_dev[:, c], side="right") for c in range(K)], axis=1).astype(cp.int32)
    mi_codes = np.asarray(cp.asnumpy(fb.binned_mi_from_codes_gpu(codes, y_dev, None, ky)))
    mi_values = np.asarray(cp.asnumpy(fb.binned_mi_from_values_gpu(x_dev, edges_dev, y_dev, nbins, ky)))
    np.testing.assert_allclose(mi_values.ravel(), mi_codes.ravel(), rtol=1e-9, atol=1e-12)
