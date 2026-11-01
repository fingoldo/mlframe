"""Per-group mean / std / count of one numeric column over integer group codes, bit-equal to ``DataFrame.groupby(codes)[col].mean() / .std(ddof=1) / .count()``.

pandas' cython group kernels run a Kahan-compensated sum in the column's own dtype for the mean, a Welford update in double for the variance (the value is widened first), and count the
non-NaN rows; they skip NaN. This kernel replays exactly those operations in the same row order, so a float32 and a double column give the pandas values to the last bit
(``test_group_stats_njit_equals_pandas``), without building a frame and a groupby per column.
"""

from __future__ import annotations


from ._numba_utils_shared import group_mean_std_count  # noqa: F401  -- hosted in _numba_utils_shared for the cross-module njit invariant
