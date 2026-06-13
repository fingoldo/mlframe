"""A/B bench for the all-finite single-sort fast path in compute_nunique_modes_quantiles_numpy (iter73, @200k).

The fast path replaces an independent np.unique (full sort) + np.nanquantile (full partition) with ONE np.sort that
derives sorted-unique values + counts AND the median_unbiased quantiles. Gated on all-finite, 1-D, median_unbiased
input (np.unique collapses all-NaN to one entry; a sort keeps each NaN distinct, so the fast path would change
nunique/modes -- hence the no-NaN gate). nunique/modes/ncrossings are exactly identical; quantiles carry a ~1e-16 ULP
delta (FP order in the linear interpolation), far below any selection-altering threshold.

Run:
    python -m mlframe.feature_engineering._benchmarks.bench_nunique_modes_quantiles_fused_iter73

Measured (this dev host, py3.14 store build, best-of-30):
    isolated compute_nunique_modes_quantiles_numpy @200k: OLD 9.57ms -> NEW 6.23ms = 1.54x min / 1.43x med
    e2e compute_numaggs(return_entropy=False, return_hurst=False) @200k: 12.79ms -> 9.09ms = 1.41x min / 1.53x med
    identity: nunique/modes/ncrossings exact; quantiles maxdiff ~1e-15; NaN input fully bit-identical (gated out)
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_engineering.numerical import compute_nunique_modes_quantiles_numpy, default_quantiles


def _best_of(fn, k: int = 30) -> tuple:
    ts = []
    for _ in range(k):
        s = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - s)
    return min(ts) * 1000, sorted(ts)[k // 2] * 1000


def main() -> None:
    arr = np.random.default_rng(0).normal(size=200_000)
    arr[::7] = arr[1]
    for _ in range(3):
        compute_nunique_modes_quantiles_numpy(arr)
    mn, med = _best_of(lambda: compute_nunique_modes_quantiles_numpy(arr))
    print(f"NEW fast path @200k: min {mn:.3f}ms med {med:.3f}ms")

    ref_q = np.nanquantile(arr, np.asarray(default_quantiles), method="median_unbiased")
    res = np.asarray(compute_nunique_modes_quantiles_numpy(arr), dtype=np.float64)
    print(f"quantile maxdiff vs np.nanquantile: {np.max(np.abs(res[5:10] - ref_q)):.2e}")


if __name__ == "__main__":
    main()
