"""Interior quantile edges for many bins through the grouped radix select equal cp.percentile's, value for value."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_resident_select import _radix_select_interior_edges, radix_quantiles_grouped


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_bins", [24, 32, 64, 100])
@pytest.mark.parametrize("k", [1, 7])
def test_equals_percentile(dtype, n_bins, k):
    """Normal, constant, tied and heavy-tailed columns where the single radix launch declines."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((20011, k)).astype(dtype)
    if k > 3:
        x[:, 1] = 3.0
        x[:, 2] = np.round(x[:, 2])
        x[:, 3] = rng.standard_cauchy(20011).astype(dtype)
    X = cp.asarray(x)
    got = radix_quantiles_grouped(X, np.linspace(0.0, 100.0, n_bins + 1)[1:-1] / 100.0)
    assert got is not None
    qs = cp.linspace(0.0, 100.0, n_bins + 1)
    want = (cp.percentile(X.ravel(), qs).reshape(-1, 1) if k == 1 else cp.percentile(X, qs, axis=0))[1:-1]
    assert cp.array_equal(got, want.astype(got.dtype))


def test_single_launch_declines_many_bins():
    """The reason for the grouping: one radix launch does not take 64 bins on a 48 KB shared-memory card."""
    X = cp.asarray(np.random.default_rng(0).standard_normal((5000, 3)).astype(np.float32))
    if int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlock", 48 * 1024)) <= 48 * 1024:
        assert _radix_select_interior_edges(X, 64) is None
