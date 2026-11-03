"""Column medians from the radix order statistics equal cp.median, value and dtype, for odd, even and tiny row counts."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_resident_select import radix_column_medians, radix_order_statistics


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [1, 2, 8, 9, 1001, 30000])
def test_equals_cp_median(dtype, n):
    """Mixed-sign columns at four scales (the double lerp of the quantile helper rounds differently from cp.median's mean), constant and tied columns."""
    rng = np.random.default_rng(7)
    M = cp.asarray((rng.standard_normal((n, 24)) * rng.choice([1e-3, 1.0, 1e4, 1e30], 24)).astype(dtype))
    M[:, 0] = 2.0
    M[:, 1] = cp.round(M[:, 1])
    med = radix_column_medians(M)
    assert med is not None
    want = cp.stack([cp.median(M[:, j]) for j in range(24)])
    assert med.dtype == want.dtype
    assert cp.array_equal(med, want)


def test_order_statistics_equal_sort():
    """Arbitrary ranks, repeated and unordered, equal the sorted columns at those ranks."""
    rng = np.random.default_rng(1)
    M = cp.asarray(rng.standard_normal((777, 5)).astype(np.float32))
    ranks = [0, 776, 388, 388, 10]
    assert cp.array_equal(radix_order_statistics(M, ranks), cp.sort(M, axis=0)[cp.asarray(ranks)])
    assert radix_order_statistics(M, [777]) is None
