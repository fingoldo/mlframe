"""Regression: uniform discretization on a denormal-tiny but positive range must not emit a
garbage code. ``rev_bin_width = n_bins / (max-min)`` overflows to inf when the range is a
subnormal, so ``(v-min)*inf`` is NaN at ``v == min`` (0*inf); NaN passed both the ``< 0`` and
``> hi`` clip guards (comparisons are False on NaN) and cast to a garbage ordinal (INT_MIN in a
wide dtype), silently poisoning the MI/SU histograms.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization import discretize_uniform, discretize_uniform_parallel


@pytest.mark.parametrize("kernel", [discretize_uniform, discretize_uniform_parallel])
def test_denormal_range_no_garbage_code(kernel):
    n_bins = 10
    mn = 5e-324  # smallest positive subnormal
    mx = 1e-323  # ~2x mn; max-min is itself a subnormal -> rev_bin_width overflows to inf
    assert mx > mn and not np.isfinite(n_bins / (mx - mn))  # the inf-overflow precondition holds
    arr = np.array([mn, mx, mn, (mn + mx) / 2.0], dtype=np.float64)
    out = kernel(arr, n_bins, mn, mx, np.int32)  # int32 so a NaN cast surfaces as INT_MIN, not int8's 0
    assert out.min() >= 0, f"garbage (negative) code emitted: {out}"
    assert out.max() <= n_bins, f"code exceeds the NaN-bin ceiling: {out}"
    assert out[0] == 0, f"value at the column floor must map to bin 0, got {out[0]}"
