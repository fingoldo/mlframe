"""Per-group mean / std / count of one numeric column over integer group codes, bit-equal to ``DataFrame.groupby(codes)[col].mean() / .std(ddof=1) / .count()``.

pandas' cython group kernels run a Kahan-compensated sum in the column's own dtype for the mean, a Welford update in float64 for the variance (the value is widened first), and count the
non-NaN rows; they skip NaN. This kernel replays exactly those operations in the same row order, so a float32 and a float64 column give the pandas values to the last bit
(``test_group_stats_njit_equals_pandas``), without building a frame and a groupby per column.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def group_mean_std_count(inverse: np.ndarray, x: np.ndarray, n_groups: int):
    """``(mean, std, count)`` per group code ``0..n_groups-1``: mean and std in the dtype of ``x`` (NaN for a group without a non-NaN row, std NaN below two rows), count as int64."""
    dtype = x.dtype
    sumx = np.zeros(n_groups, dtype=dtype)
    comp = np.zeros(n_groups, dtype=dtype)
    cnt = np.zeros(n_groups, dtype=np.int64)
    run_mean = np.zeros(n_groups, dtype=np.float64)  # Intentional float64: pandas accumulates the variance in float64 whatever the column dtype
    m2 = np.zeros(n_groups, dtype=np.float64)  # Intentional float64: same accumulator
    for i in range(x.shape[0]):
        v = x[i]
        if v == v:
            lab = inverse[i]
            cnt[lab] += 1
            y = v - comp[lab]
            t = sumx[lab] + y
            comp[lab] = t - sumx[lab] - y
            sumx[lab] = t
            old = run_mean[lab]
            run_mean[lab] = old + (v - old) / cnt[lab]
            m2[lab] = m2[lab] + (v - run_mean[lab]) * (v - old)
    mean = np.empty(n_groups, dtype=dtype)
    std = np.empty(n_groups, dtype=dtype)
    for g in range(n_groups):
        if cnt[g] == 0:
            mean[g] = np.nan
            std[g] = np.nan
        else:
            mean[g] = sumx[g] / cnt[g]
            if cnt[g] > 1:
                std[g] = np.sqrt(m2[g] / (cnt[g] - 1))
            else:
                std[g] = np.nan
    return mean, std, cnt
